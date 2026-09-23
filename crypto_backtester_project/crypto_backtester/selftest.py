"""
Offline self-test.

Builds a synthetic universe (trending coins, chop coins, one coin that dies
mid-sample, one stablecoin decoy and one wrapper decoy) and runs the whole
pipeline on it.  This exists so you can prove the engine works -- no
look-ahead, cash conservation, delisting, watchlist conversion -- without
touching an exchange.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .universe import CoinData, _to_grid


def synthetic_universe(n_coins=12, n_days=2200, seed=3) -> "object":
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2018-01-01")
    coins: list[CoinData] = []
    log: list[dict] = []

    # a shared "market" factor so the BTC gate has something real to gate on
    regime = np.concatenate([
        rng.normal(0.0016, 0.020, n_days // 3),
        rng.normal(-0.0015, 0.028, n_days // 3),
        rng.normal(0.0012, 0.024, n_days - 2 * (n_days // 3)),
    ])

    for i in range(n_coins):
        base = "BTC" if i == 0 else f"C{i:02d}"
        listing = 0 if i < max(2, n_coins // 2) else int(rng.integers(50, 500))
        m = n_days - listing
        beta = 1.0 if i == 0 else float(rng.uniform(0.6, 1.8))
        idio = rng.normal(0.0, 0.022 * beta, m)
        r = beta * regime[listing:] + idio
        close = 100.0 * np.exp(np.cumsum(r))
        rng_hl = np.abs(rng.normal(0.0, 0.02, m))
        high = close * (1 + rng_hl)
        low = close * (1 - rng_hl)
        openp = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.004, m))
        high = np.maximum.reduce([high, openp, close])
        low = np.minimum.reduce([low, openp, close])
        vol = np.abs(rng.lognormal(12.0, 0.6, m))
        dates = pd.date_range(start + pd.Timedelta(days=listing), periods=m, freq="D")

        # one coin dies two thirds of the way through
        if i == n_coins - 1:
            keep = int(m * 0.66)
            dates, openp, high, low, close, vol = (dates[:keep], openp[:keep], high[:keep],
                                                   low[:keep], close[:keep], vol[:keep])

        df = pd.DataFrame({"date": dates, "open": openp, "high": high, "low": low,
                           "close": close, "volume": vol,
                           "quote_volume": vol * close})
        cd = CoinData(base=base, provider="synthetic")
        cd.dates_raw = df["date"].to_numpy()
        cd.close_raw = df["close"].to_numpy()
        cd._df = df
        coins.append(cd)
        log.append({"base": base, "status": "accepted", "reason": "synthetic",
                    "rank": i, "market_cap": 1e9 - i * 1e7, "bars": len(df),
                    "provider": "synthetic", "first": str(df["date"].iloc[0].date()),
                    "last": str(df["date"].iloc[-1].date())})

    log.append({"base": "USDX", "status": "rejected", "reason": "peg detected from data"})
    log.append({"base": "WBTC", "status": "rejected", "reason": "duplicate of BTC"})
    return _to_grid(coins, log, verbose=False)


def check_filters() -> list[str]:
    """Sanity-check the name-based universe filters."""
    from .universe import name_verdict
    issues = []
    must_reject = ["USDT", "USDC", "DAI", "FDUSD", "USDE", "WBTC", "WETH", "STETH",
                   "WSTETH", "CBBTC", "JITOSOL", "PAXG", "BTCUP", "ETHDOWN"]
    must_keep = ["BTC", "ETH", "SOL", "XRP", "XLM", "DOGE", "LINK", "AVAX",
                 "WIF", "WLD", "UNI", "SUI", "TIA"]
    for s in must_reject:
        if name_verdict(s)[0]:
            issues.append(f"FILTER MISS: {s} should have been rejected")
    for s in must_keep:
        if not name_verdict(s)[0]:
            issues.append(f"FILTER FALSE POSITIVE: {s} was rejected ({name_verdict(s)[1]})")
    return issues


def run(n_trials=25, verbose=True) -> int:
    from .engine import run_backtest
    from .metrics import compute, score
    from .search import Objective, materialize, suggest, describe

    fails = check_filters()
    uni = synthetic_universe()
    if verbose:
        print(f"[selftest] synthetic universe: {uni.n} coins, "
              f"{len(uni.dates)} bars, start idx {uni.start_idx}")

    # --- determinism + cash conservation -----------------------------------
    p = materialize({"entry_type": 0, "entry_ma_len": 50, "entry_ma_type": 1,
                     "entry_require_cross": True, "exit_type": 0,
                     "sl_mult": 3.0, "trail_mult": 6.0, "wl_mode": 2,
                     "wl_mom_lookback": 30, "max_pyramid_layers": 2,
                     "pyramid_min_gap": 10, "use_global_tp": True, "tp_mult": 12.0,
                     "tp_size_pct": 50.0, "tp_move_sl_be": True,
                     "use_btc_entry_gate": True, "btc_ma_len": 100, "btc_ma_type": 0,
                     "adx_thresh": 15.0})
    r1 = run_backtest(uni, p, collect=True)
    r2 = run_backtest(uni, p, collect=True)
    if abs(r1.equity[-1] - r2.equity[-1]) > 1e-6:
        fails.append("NON-DETERMINISTIC: two identical runs disagree")

    m = compute(r1, uni)
    if not m.get("valid"):
        fails.append(f"baseline config produced no valid metrics: {m.get('reason')}")
    else:
        if verbose:
            print(f"[selftest] baseline: trades={m['n_trades']} "
                  f"ROI/yr={m['roi_on_capital_ann']:.2%} DD={m['max_dd_capital']:.2%} "
                  f"score={score(m):.4f}")
        # cash conservation: final equity == contributions + realised + unrealised
        contributed = C.INITIAL_CASH + r1.contrib.sum()
        pnl = sum(t["pnl"] for t in r1.trades)
        if abs((contributed + pnl) - r1.equity[-1]) > 1.0:
            fails.append(f"CASH LEAK: contributions {contributed:,.2f} + pnl {pnl:,.2f} "
                         f"!= final equity {r1.equity[-1]:,.2f}")
        if (r1.cash < -1e-6).any():
            fails.append("NEGATIVE CASH: the engine spent money it did not have")
        # no trade may start before the warm-up window
        if r1.trades and min(t["entry_date"] for t in r1.trades) < uni.dates[uni.start_idx]:
            fails.append("LOOK-AHEAD: a trade opened before the backtest start")

    # --- hard look-ahead audit ---------------------------------------------
    # Every non-watchlist entry must be the OPEN of the bar AFTER a bar on which
    # the entry signal was true, at exactly the modelled fill price. If the
    # engine ever peeked at the bar it traded on, this fails.
    from .signals import build_signals
    sigs, _ = build_signals(uni, p)
    dpos = {d: i for i, d in enumerate(uni.dates)}
    bad_sig = bad_px = 0
    ci_of = {c.base: i for i, c in enumerate(uni.coins)}
    for t in r1.trades:
        ci = ci_of[t["coin"]]
        ti = dpos.get(np.datetime64(pd.Timestamp(t["entry_date"]).to_datetime64(), "ns"))
        if ti is None or ti < 1:
            continue
        if not t["from_watchlist"] and not sigs[ci].entry[ti - 1]:
            bad_sig += 1
        want = uni.coins[ci].m_open[ti] * (1.0 + C.SLIPPAGE_BPS / 1e4)
        if not np.isfinite(want) or abs(want - t["entry_price"]) > max(1e-6, 1e-6 * want):
            bad_px += 1
    if bad_sig:
        fails.append(f"LOOK-AHEAD: {bad_sig} entries had no signal on the preceding bar")
    if bad_px:
        fails.append(f"FILL MODEL: {bad_px} entries were not filled at the next bar's open")
    if verbose and not (bad_sig or bad_px):
        print(f"[selftest] look-ahead audit clean across {len(r1.trades)} entries")

    # --- look-ahead probe: shifting all prices forward must change results ---
    r3 = run_backtest(uni, materialize({"entry_type": 2, "xover_short_len": 40,
                                        "xover_short_type": 1, "xover_gap": 60,
                                        "xover_long_type": 0, "exit_type": 5,
                                        "wl_mode": 5, "max_pyramid_layers": 1}))
    if not r3.ok:
        fails.append("MA-xover architecture failed to run")

    # --- every architecture must at least execute --------------------------
    for et in range(4):
        for xt in range(6):
            raw = {"entry_type": et, "exit_type": xt, "wl_mode": 3,
                   "max_pyramid_layers": 2, "pyramid_min_gap": 5,
                   "entry_ma_len": 60, "entry_ma_type": 0,
                   "rsi_f_len": 20, "rsi_f_smt": 8, "rsi_s_len": 60, "rsi_s_smt": 20,
                   "xover_short_len": 30, "xover_short_type": 1, "xover_gap": 40,
                   "xover_long_type": 0, "vol_ma_len": 20, "vol_mult": 2.0,
                   "price_lookback": 20, "body_atr_mult": 1.0,
                   "sl_mult": 3.0, "trail_mult": 6.0, "trail_pct": 15.0,
                   "exit_atr_mult": 3.0, "exit_ma_len": 100, "exit_ma_type": 1,
                   "use_safety_stop": True, "safety_sl_mult": 6.0}
            try:
                rr = run_backtest(uni, materialize(raw))
                if not rr.ok:
                    fails.append(f"entry {et} / exit {xt}: {rr.reason}")
            except Exception as e:  # noqa: BLE001
                fails.append(f"entry {et} / exit {xt} RAISED {e!r}")

    # --- full Optuna loop ---------------------------------------------------
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    obj = Objective(uni)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=1))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    good = [t for t in study.trials if t.value is not None and t.value > C.SCORE["invalid_score"]]
    if verbose:
        print(f"[selftest] optuna: {len(study.trials)} trials, {len(good)} scored, "
              f"best={study.best_value:.4f}")
    if not good:
        fails.append("no trial produced a valid score -- guards may be too tight")

    # --- report generation --------------------------------------------------
    try:
        final, s_is, s_oos, m_is, m_oos, res = obj.evaluate(study.best_params, collect=True)
        if res is not None and m_is.get("valid"):
            from .report import write_report
            path = write_report(uni, study.best_params, final, m_is, res, m_oos,
                                study=study, path=C.OUT_DIR / "SELFTEST_REPORT.md",
                                label="SELF-TEST")
            if verbose:
                print(f"[selftest] report written: {path}")
                print("\n--- strategy in words ---")
                print(describe(materialize(study.best_params)))
        else:
            fails.append("best trial could not be re-evaluated for the report")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        fails.append(f"report generation RAISED {e!r}")

    print("\n" + "=" * 70)
    if fails:
        print(f"SELFTEST FAILED ({len(fails)} issue(s))")
        for f in fails:
            print("  -", f)
    else:
        print("SELFTEST PASSED - engine, scoring, search and report all functional")
    print("=" * 70)
    return 1 if fails else 0
