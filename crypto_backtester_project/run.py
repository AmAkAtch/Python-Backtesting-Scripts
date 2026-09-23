#!/usr/bin/env python3
"""
Crypto swing-trading backtester + Optuna strategy search.

    python run.py selftest                 # offline end-to-end check (no network)
    python run.py universe                 # build/inspect the coin shortlist
    python run.py optimize --trials 500    # search; saves current_winner.json live
    python run.py validate                 # re-score the stored winner on fresh data
    python run.py report                   # rebuild WINNER_REPORT.md from the winner

Flags that apply anywhere:
    --coins N        override N_COINS
    --refresh        ignore the OHLCV cache and re-download
    --no-wf          disable the walk-forward holdout for this run
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from crypto_backtester import config as C


def _load_universe(args):
    from crypto_backtester.universe import build_universe, save_universe_log
    uni = build_universe(n_coins=args.coins or C.N_COINS, refresh=args.refresh)
    save_universe_log(uni)
    return uni


def cmd_universe(args):
    uni = _load_universe(args)
    acc = [r for r in uni.log if r["status"] == "accepted"]
    rej = [r for r in uni.log if r["status"] == "rejected"]
    print(f"\nACCEPTED ({len(acc)}):")
    for r in acc:
        print(f"  {r['base']:<8} {str(r.get('bars')):>5} bars  {r.get('first')} -> {r.get('last')}"
              f"   [{r.get('reason')}]")
    print(f"\nREJECTED ({len(rej)}):")
    for r in rej:
        print(f"  {r['base']:<8} {r['reason']}")
    print(f"\nGrid {uni.meta['grid_start']} .. {uni.meta['grid_end']}")
    print(f"Backtest starts {str(uni.dates[uni.start_idx])[:10]} "
          f"(coverage {uni.meta['coverage_at_start']:.1%} + {C.WARMUP_CAP} warm-up bars)")
    print(f"Details -> {C.OUT_DIR / 'universe_selection.json'}")
    return 0


def cmd_optimize(args):
    from crypto_backtester.report import write_report
    from crypto_backtester.search import load_winner, optimize, save_winner

    uni = _load_universe(args)
    print(f"\n[optimize] {args.trials} trials on {uni.n} coins, "
          f"{str(uni.dates[uni.start_idx])[:10]} -> {str(uni.dates[uni.meta['end_idx']])[:10]}")
    if C.USE_WALK_FORWARD:
        print(f"[optimize] walk-forward on: last {C.OOS_FRACTION:.0%} of the timeline is a holdout")

    study, obj, baseline = optimize(uni, n_trials=args.trials)

    best_params, best_value = study.best_params, study.best_value
    if baseline and baseline[0] > best_value:
        print(f"\n[optimize] nothing beat the stored champion ({baseline[0]:.5f} from {baseline[2]}); "
              f"keeping it as the winner.")
        best_params, best_value = baseline[1], baseline[0]

    final, s_is, s_oos, m_is, m_oos, res = obj.evaluate(best_params, collect=True)

    prev = load_winner(C.FINAL_WINNER)
    prev_score = None
    if prev:
        try:
            prev_score = obj.evaluate(prev["params_raw"])[0]
        except Exception:  # noqa: BLE001
            prev_score = None
    if prev_score is None or final > prev_score:
        save_winner(C.FINAL_WINNER, best_params, final, m_is, uni,
                    extra={"score_is": s_is,
                           "score_oos": None if not np.isfinite(s_oos) else s_oos,
                           "oos_metrics": None if not m_oos else
                           {k: v for k, v in m_oos.items() if not k.startswith("_")},
                           "n_trials": len(study.trials)})
        print(f"[optimize] FINAL WINNER saved -> {C.FINAL_WINNER}  (score {final:.5f})")
    else:
        print(f"[optimize] stored winner.json ({prev_score:.5f}) still better than this run "
              f"({final:.5f}); left untouched.")
        final, s_is, s_oos, m_is, m_oos, res = obj.evaluate(prev["params_raw"], collect=True)
        best_params = prev["params_raw"]

    path = write_report(uni, best_params, final, m_is, res, m_oos, study=study, baseline=baseline)
    print(f"[optimize] report -> {path}")
    _print_summary(m_is, m_oos, final, s_is, s_oos)
    return 0


def cmd_validate(args):
    from crypto_backtester.search import Objective, load_winner, describe, materialize
    uni = _load_universe(args)
    obj = Objective(uni)
    any_found = False
    for tag, path in (("winner.json", C.FINAL_WINNER), ("current_winner.json", C.CURRENT_WINNER)):
        w = load_winner(path)
        if not w:
            print(f"[validate] {tag}: not present")
            continue
        any_found = True
        final, s_is, s_oos, m_is, m_oos, _ = obj.evaluate(w["params_raw"])
        drift = "same data" if w.get("fingerprint") else "unknown"
        print(f"\n===== {tag} =====")
        print(f"stored score {w.get('score')}  ->  re-scored {final:.5f}  ({drift})")
        print(describe(materialize(w["params_raw"])))
        _print_summary(m_is, m_oos, final, s_is, s_oos)
    if not any_found:
        print("[validate] no saved configuration found; run `optimize` first.")
    return 0


def cmd_report(args):
    from crypto_backtester.report import write_report
    from crypto_backtester.search import Objective, load_winner
    uni = _load_universe(args)
    obj = Objective(uni)
    w = load_winner(C.FINAL_WINNER) or load_winner(C.CURRENT_WINNER)
    if not w:
        print("[report] no saved configuration found; run `optimize` first.")
        return 1
    final, s_is, s_oos, m_is, m_oos, res = obj.evaluate(w["params_raw"], collect=True)
    path = write_report(uni, w["params_raw"], final, m_is, res, m_oos)
    print(f"[report] -> {path}")
    return 0


def _print_summary(m_is, m_oos, final, s_is, s_oos):
    if not m_is.get("valid"):
        print("  (no valid metrics)")
        return
    print(f"\n  score {final:.5f}   (in-sample {s_is:.5f} / out-of-sample "
          f"{s_oos if np.isfinite(s_oos) else float('nan'):.5f})")
    print(f"  window            {m_is['start']} -> {m_is['end']}  ({m_is['years']:.1f}y)")
    print(f"  ROI on capital    {m_is['roi_on_capital']:>9.2%}   "
          f"({m_is['roi_on_capital_ann']:.2%}/yr on ${m_is['avg_capital_deployed']:,.0f} avg at risk)")
    print(f"  median year ROI   {m_is['median_year_return']:>9.2%}   "
          f"median coin ROI {m_is['median_coin_roi']:.2%}")
    print(f"  max drawdown      {m_is['max_dd_capital']:>9.2%}   "
          f"(raw equity curve {m_is['max_dd_equity']:.2%})")
    print(f"  profit factor     {m_is['profit_factor']:>9.2f}   win rate {m_is['win_rate']:.1%}")
    print(f"  trades            {m_is['n_trades']:>9,}   across {m_is['n_coins_traded']} coins")
    print(f"  final equity      ${m_is['final_equity']:>8,.0f}   "
          f"paid in ${m_is['total_contributed']:,.0f}")
    if m_oos and m_oos.get("valid"):
        print(f"  OOS ROI/yr        {m_oos['roi_on_capital_ann']:>9.2%}   "
              f"OOS DD {m_oos['max_dd_capital']:.2%}  ({m_oos['n_trades']} trades)")


def cmd_selftest(args):
    from crypto_backtester.selftest import run
    return run(n_trials=args.trials if args.trials != C.N_TRIALS else 25)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["selftest", "universe", "optimize", "validate", "report"])
    ap.add_argument("--trials", type=int, default=C.N_TRIALS)
    ap.add_argument("--coins", type=int, default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no-wf", action="store_true", help="disable the walk-forward holdout")
    args = ap.parse_args()

    if args.no_wf:
        C.USE_WALK_FORWARD = False
    if args.coins:
        C.N_COINS = args.coins

    return {
        "selftest": cmd_selftest, "universe": cmd_universe, "optimize": cmd_optimize,
        "validate": cmd_validate, "report": cmd_report,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
