"""
The search space and the Optuna driver (plan steps 4, 5, 9, 10, 13, 14).

Philosophy: hard-code nothing that could be learned.  Everything that changes
the shape of a trade -- indicator, length, type, threshold, multiplier, ranking
rule, layer cap -- is sampled.  The only fixed rules are the ones the plan
itself fixes (the $3 M liquidity floor, the $1,000 tranche, the monthly top-up)
plus the cost model, because letting an optimiser choose its own fees is how
you build a backtest that makes money and a live account that does not.

Conditional sampling: parameters are only suggested when the chosen
architecture actually uses them, so TPE never wastes density on dead
dimensions.  Shared blocks (RSI, MA cross-over) are suggested once and reused
by both the entry and the exit when the plan says they are shared.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np
import optuna

from . import config as C
from .engine import run_backtest
from .metrics import combine_is_oos, compute, score

optuna.logging.set_verbosity(optuna.logging.WARNING)

ENFORCE_RSI_ORDER = True  # swap fast/slow if sampled inverted (kills mirror dupes)
MA_TYPES = [0, 1, 2, 3, 4]


# ============================================================= search space ===
def suggest(trial: optuna.Trial) -> dict:
    p: dict = {}
    p["entry_type"] = trial.suggest_categorical("entry_type", [0, 1, 2, 3])
    p["exit_type"] = trial.suggest_categorical("exit_type", [0, 1, 2, 3, 4, 5])
    et, xt = p["entry_type"], p["exit_type"]

    # ---------------------------------------------------------- shared blocks
    need_rsi = (et == 1) or (xt == 4)
    if need_rsi:
        p["rsi_f_len"] = trial.suggest_int("rsi_f_len", 10, 100)
        p["rsi_f_smt"] = trial.suggest_int("rsi_f_smt", 5, 50)
        p["rsi_s_len"] = trial.suggest_int("rsi_s_len", 10, 100)
        p["rsi_s_smt"] = trial.suggest_int("rsi_s_smt", 5, 50)

    need_xover = (et == 2) or (xt == 5)
    if need_xover:
        p["xover_short_len"] = trial.suggest_int("xover_short_len", 20, 200)
        p["xover_short_type"] = trial.suggest_categorical("xover_short_type", MA_TYPES)
        p["xover_gap"] = trial.suggest_int("xover_gap", 10, 150)
        p["xover_long_type"] = trial.suggest_categorical("xover_long_type", MA_TYPES)

    # --------------------------------------------------------------- entries
    if et == 0:
        p["entry_ma_len"] = trial.suggest_int("entry_ma_len", 10, 250)
        p["entry_ma_type"] = trial.suggest_categorical("entry_ma_type", MA_TYPES)
        # extension: a *state* ("close is above") fires every single bar, a
        # *cross* fires once. Let the optimiser decide which it wants.
        p["entry_require_cross"] = trial.suggest_categorical("entry_require_cross", [True, False])
    elif et == 3:
        p["vol_ma_len"] = trial.suggest_int("vol_ma_len", 10, 100)
        p["vol_mult"] = trial.suggest_float("vol_mult", 1.5, 5.0)
        p["price_lookback"] = trial.suggest_int("price_lookback", 10, 60)
        p["body_atr_mult"] = trial.suggest_float("body_atr_mult", 0.5, 2.5)

    # --------------------------------------------------------------- filters
    p["use_btc_entry_gate"] = trial.suggest_categorical("use_btc_entry_gate", [True, False])
    p["use_btc_exit_override"] = trial.suggest_categorical("use_btc_exit_override", [True, False])
    if p["use_btc_entry_gate"] or p["use_btc_exit_override"]:
        p["btc_ma_len"] = trial.suggest_int("btc_ma_len", 20, 300)
        p["btc_ma_type"] = trial.suggest_categorical("btc_ma_type", MA_TYPES)

    p["adx_thresh"] = trial.suggest_categorical("adx_thresh", [0.0, 15.0, 20.0, 25.0])

    if et == 1:
        p["use_rsi_trend_filter"] = trial.suggest_categorical("use_rsi_trend_filter", [True, False])
        if p["use_rsi_trend_filter"]:
            p["rsi_trend_ma_len"] = trial.suggest_int("rsi_trend_ma_len", 20, 300)
            p["rsi_trend_ma_type"] = trial.suggest_categorical("rsi_trend_ma_type", MA_TYPES)

    # ----------------------------------------------------------- pyramiding
    p["max_pyramid_layers"] = trial.suggest_int("max_pyramid_layers", 1, 4)
    if p["max_pyramid_layers"] > 1:
        # extension: without a cadence rule a "state" entry adds a layer every
        # single bar until the cap; with one, add-ons are spaced deliberately.
        p["pyramid_min_gap"] = trial.suggest_int("pyramid_min_gap", 0, 30)
        p["pyramid_require_profit"] = trial.suggest_categorical("pyramid_require_profit", [True, False])

    # ------------------------------------------------------------- watchlist
    p["wl_mode"] = trial.suggest_categorical("wl_mode", [0, 1, 2, 3, 4, 5])
    if p["wl_mode"] == 2:
        p["wl_mom_lookback"] = trial.suggest_int("wl_mom_lookback", 5, 120)
        p["wl_mom_vs_btc"] = trial.suggest_categorical("wl_mom_vs_btc", [True, False])
    if p["wl_mode"] != 5:
        # extension: a three-month-old breakout signal is not the same trade
        # any more. 0 = keep forever (the literal reading of the plan).
        p["wl_max_age"] = trial.suggest_categorical("wl_max_age", [0, 5, 10, 21, 42, 90])

    # ------------------------------------------------- global take-profit ---
    p["use_global_tp"] = trial.suggest_categorical("use_global_tp", [True, False])
    if p["use_global_tp"]:
        p["tp_mult"] = trial.suggest_float("tp_mult", 5.0, 70.0)
        p["tp_size_pct"] = trial.suggest_float("tp_size_pct", 10.0, 90.0)
        p["tp_move_sl_be"] = trial.suggest_categorical("tp_move_sl_be", [True, False])

    # ------------------------------------------------------------ exits -----
    if xt == 0:
        p["sl_mult"] = trial.suggest_float("sl_mult", 1.5, 8.0)
        p["trail_mult"] = trial.suggest_float("trail_mult", 2.0, 15.0)
    elif xt == 1:
        p["trail_pct"] = trial.suggest_float("trail_pct", 5.0, 35.0)
    elif xt == 2:
        p["exit_atr_mult"] = trial.suggest_float("exit_atr_mult", 1.0, 6.0)
    elif xt == 3:
        p["exit_ma_len"] = trial.suggest_int("exit_ma_len", 20, 300)
        p["exit_ma_type"] = trial.suggest_categorical("exit_ma_type", MA_TYPES)

    if xt in (3, 4, 5):
        # extension: signal-only exits have no floor. A coin can go -85 % while
        # you wait for a cross. Optional catastrophic stop, optimiser's call.
        p["use_safety_stop"] = trial.suggest_categorical("use_safety_stop", [True, False])
        if p["use_safety_stop"]:
            p["safety_sl_mult"] = trial.suggest_float("safety_sl_mult", 2.0, 15.0)
    return p


def materialize(raw: dict) -> dict:
    """Fill in derived values + safe defaults. Pure function of `raw`."""
    p = dict(raw)
    p.setdefault("entry_require_cross", False)
    p.setdefault("adx_thresh", 0.0)
    p.setdefault("use_btc_entry_gate", False)
    p.setdefault("use_btc_exit_override", False)
    p.setdefault("use_rsi_trend_filter", False)
    p.setdefault("max_pyramid_layers", 1)
    p.setdefault("pyramid_min_gap", 0)
    p.setdefault("pyramid_require_profit", False)
    p.setdefault("wl_mode", 5)
    p.setdefault("wl_mom_lookback", 20)
    p.setdefault("wl_mom_vs_btc", False)
    p.setdefault("wl_max_age", 0)
    p.setdefault("use_global_tp", False)
    p.setdefault("use_safety_stop", False)
    p.setdefault("safety_sl_mult", 0.0)

    if ENFORCE_RSI_ORDER and "rsi_f_len" in p:
        if p["rsi_f_len"] > p["rsi_s_len"]:
            p["rsi_f_len"], p["rsi_s_len"] = p["rsi_s_len"], p["rsi_f_len"]
            p["rsi_f_smt"], p["rsi_s_smt"] = p["rsi_s_smt"], p["rsi_f_smt"]

    if "xover_short_len" in p:  # derived per the plan
        p["xover_long_len"] = int(min(300, p["xover_short_len"] + p["xover_gap"]))
    return p


def describe(p: dict) -> str:
    """Plain-English rendering of a config -- used by the report writer."""
    et, xt = p["entry_type"], p["exit_type"]
    L = []
    if et == 0:
        verb = "closes above" if not p.get("entry_require_cross") else "crosses above"
        L.append(f"Buy when the daily close {verb} its {p['entry_ma_len']}-day "
                 f"{C.MA_NAMES[p['entry_ma_type']]}.")
    elif et == 1:
        L.append(f"Buy when a {p['rsi_f_len']}-period RSI smoothed over {p['rsi_f_smt']} days "
                 f"crosses above a {p['rsi_s_len']}-period RSI smoothed over {p['rsi_s_smt']} days.")
    elif et == 2:
        L.append(f"Buy when the {p['xover_short_len']}-day {C.MA_NAMES[p['xover_short_type']]} "
                 f"crosses above the {p['xover_long_len']}-day {C.MA_NAMES[p['xover_long_type']]}.")
    else:
        L.append(f"Buy on a volume-confirmed breakout: volume > {p['vol_mult']:.2f}x its "
                 f"{p['vol_ma_len']}-day average, the close takes out the highest high of the "
                 f"prior {p['price_lookback']} days, and the candle body is at least "
                 f"{p['body_atr_mult']:.2f}x ATR(14).")

    if p.get("use_btc_entry_gate"):
        L.append(f"Only when BTC is above its {p['btc_ma_len']}-day {C.MA_NAMES[p['btc_ma_type']]} "
                 f"(macro risk-on gate).")
    if p.get("adx_thresh", 0) > 0:
        L.append(f"Only when ADX(14) >= {p['adx_thresh']:.0f} (trend must be real, not chop).")
    if et == 1 and p.get("use_rsi_trend_filter"):
        L.append(f"Only when price is above its {p['rsi_trend_ma_len']}-day "
                 f"{C.MA_NAMES[p['rsi_trend_ma_type']]} (RSI trend confirmation).")
    L.append(f"Never touch a coin whose trailing {C.LIQUIDITY_WINDOW}-day average dollar volume is "
             f"under ${C.LIQUIDITY_FLOOR_USD:,.0f}.")

    L.append(f"Each buy is a flat ${C.TRANCHE_USD:,.0f}, all-in. Up to "
             f"{p['max_pyramid_layers']} layer(s) per coin"
             + (f", at least {p['pyramid_min_gap']} day(s) apart" if p.get("pyramid_min_gap") else "")
             + (", and only while the existing layers are in profit." if p.get("pyramid_require_profit") else "."))

    if p["wl_mode"] == 5:
        L.append("No watchlist: a signal that arrives with an empty wallet is simply missed.")
    else:
        L.append(f"Signals that arrive with no cash become shadow positions and are queued; "
                 f"when cash frees up the queue is filled by {C.WL_NAMES[p['wl_mode']]}"
                 + (f" ({p['wl_mom_lookback']}-day momentum"
                    + (" relative to BTC" if p.get("wl_mom_vs_btc") else "") + ")"
                    if p["wl_mode"] == 2 else "")
                 + (f", dropping anything older than {p['wl_max_age']} days." if p.get("wl_max_age") else "."))
        L.append("A shadow position that hits its exit rule before it is ever funded is deleted, "
                 "never traded.")

    if xt == 0:
        L.append(f"Exit: initial stop {p['sl_mult']:.2f}x ATR(14) below entry, then a "
                 f"{p['trail_mult']:.2f}x ATR chandelier trail that only ever ratchets up.")
    elif xt == 1:
        L.append(f"Exit: liquidate on a {p['trail_pct']:.1f}% fall from the highest high since entry.")
    elif xt == 2:
        L.append(f"Exit: liquidate below peak - {p['exit_atr_mult']:.2f}x ATR(14).")
    elif xt == 3:
        L.append(f"Exit: liquidate when the close drops below its {p['exit_ma_len']}-day "
                 f"{C.MA_NAMES[p['exit_ma_type']]}.")
    elif xt == 4:
        L.append("Exit: liquidate when the fast smoothed RSI crosses back under the slow one.")
    else:
        L.append(f"Exit: liquidate when the {p['xover_short_len']}-day MA crosses under the "
                 f"{p['xover_long_len']}-day MA.")
    if p.get("use_safety_stop"):
        L.append(f"Plus a catastrophic backstop {p['safety_sl_mult']:.2f}x ATR below entry.")

    if p.get("use_global_tp"):
        L.append(f"Global partial take-profit: at entry + {p['tp_mult']:.1f}x ATR(14), sell "
                 f"{p['tp_size_pct']:.0f}% of the layer"
                 + (" and ratchet the stop to breakeven." if p.get("tp_move_sl_be") else "."))
    if p.get("use_btc_exit_override"):
        L.append(f"Macro kill-switch: if BTC closes under its {p['btc_ma_len']}-day "
                 f"{C.MA_NAMES[p['btc_ma_type']]}, every altcoin position is liquidated at the "
                 f"next open.")
    L.append("Any coin that is halted or delisted is written off per the configured policy.")
    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(L))


# ============================================================ winner storage ==
def _fingerprint(uni) -> str:
    import hashlib
    key = json.dumps({
        "coins": sorted(c.base for c in uni.coins),
        "start": uni.meta["grid_start"], "end": uni.meta["grid_end"],
        "tranche": C.TRANCHE_USD, "monthly": C.MONTHLY_CONTRIBUTION,
        "fee": C.TAKER_FEE_BPS, "slip": C.SLIPPAGE_BPS, "tax": C.TAX_RATE_ON_GAINS,
        "liq": C.LIQUIDITY_FLOOR_USD, "delist": C.DELIST_VALUATION,
        "wf": [C.USE_WALK_FORWARD, C.OOS_FRACTION, C.OOS_PENALTY_WEIGHT],
        "score": C.SCORE,
    }, sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def save_winner(path, raw: dict, sc: float, m: dict, uni, extra: dict | None = None) -> None:
    payload = {
        "schema": 2,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "score": float(sc),
        "params_raw": raw,
        "params_full": materialize(raw),
        "fingerprint": _fingerprint(uni),
        "universe": sorted(c.base for c in uni.coins),
        "window": {"start": m.get("start"), "end": m.get("end")},
        "metrics": {k: v for k, v in m.items() if not k.startswith("_")},
    }
    if extra:
        payload.update(extra)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    import os
    os.replace(tmp, path)


def load_winner(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


# ================================================================= objective ==
class Objective:
    def __init__(self, uni):
        self.uni = uni
        n = len(uni.dates)
        t0 = uni.start_idx
        t1 = int(uni.meta.get("end_idx", n - 1))
        span = max(t1 - t0, 1)
        if C.USE_WALK_FORWARD:
            cut = int(t0 + (1.0 - C.OOS_FRACTION) * span)
            self.is_win = (t0, cut)
            self.oos_win = (cut + 1, t1)
            self.is_scale = (cut - t0) / span
            self.oos_scale = (t1 - cut) / span
        else:
            self.is_win = (t0, t1)
            self.oos_win = None
            self.is_scale, self.oos_scale = 1.0, 1.0

    def evaluate(self, raw: dict, trial=None, collect=False):
        p = materialize(raw)
        r_is = run_backtest(self.uni, p, trial=trial, collect=collect,
                            start_idx=self.is_win[0], end_idx=self.is_win[1])
        m_is = compute(r_is, self.uni)
        s_is = score(m_is, self.is_scale)
        m_oos, s_oos = None, float("nan")
        if self.oos_win is not None:
            r_oos = run_backtest(self.uni, p, collect=collect,
                                 start_idx=self.oos_win[0], end_idx=self.oos_win[1])
            m_oos = compute(r_oos, self.uni)
            s_oos = score(m_oos, self.oos_scale)
        final = combine_is_oos(s_is, s_oos) if self.oos_win is not None else s_is
        return final, s_is, s_oos, m_is, m_oos, (r_is if collect else None)

    def __call__(self, trial: optuna.Trial) -> float:
        raw = suggest(trial)
        try:
            final, s_is, s_oos, m_is, m_oos, _ = self.evaluate(raw, trial=trial)
        except optuna.TrialPruned:
            raise
        except Exception as e:  # noqa: BLE001
            trial.set_user_attr("error", repr(e)[:300])
            return C.SCORE["invalid_score"]
        trial.set_user_attr("score_is", float(s_is))
        trial.set_user_attr("score_oos", float(s_oos) if np.isfinite(s_oos) else None)
        for k in ("roi_on_capital_ann", "median_year_return", "median_coin_roi",
                  "max_dd_capital", "profit_factor", "n_trades", "win_rate"):
            if m_is.get(k) is not None:
                v = m_is[k]
                trial.set_user_attr(k, float(v) if np.isfinite(float(v)) else None)
        if m_oos and m_oos.get("valid"):
            trial.set_user_attr("oos_roi_ann", float(m_oos["roi_on_capital_ann"]))
            trial.set_user_attr("oos_dd", float(m_oos["max_dd_capital"]))
        return float(final)


# ================================================================== driver ====
def optimize(uni, n_trials: int | None = None, verbose: bool = True):
    n_trials = n_trials or C.N_TRIALS
    obj = Objective(uni)

    # ---- step 13/14: re-score the stored champions to set the bar ---------
    baseline, seeds = None, []
    for tag, path in (("winner.json", C.FINAL_WINNER), ("current_winner.json", C.CURRENT_WINNER)):
        w = load_winner(path)
        if not w:
            continue
        try:
            final, s_is, s_oos, m_is, _, _ = obj.evaluate(w["params_raw"])
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] stored {tag} failed to re-evaluate: {e}")
            continue
        drift = "" if w.get("fingerprint") == _fingerprint(uni) else "  (universe/settings CHANGED)"
        if verbose:
            print(f"  baseline from {tag}: stored={w.get('score'):.5f} "
                  f"re-scored={final:.5f}{drift}")
        seeds.append(w["params_raw"])
        if baseline is None or final > baseline[0]:
            baseline = (final, w["params_raw"], tag)

    import warnings
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    sampler = optuna.samplers.TPESampler(
        seed=C.SEED, multivariate=True, group=True,
        n_startup_trials=max(30, n_trials // 12), constant_liar=(C.N_JOBS > 1))
    pruner = (optuna.pruners.MedianPruner(n_startup_trials=25,
                                          n_warmup_steps=C.PRUNER_WARMUP_YEARS)
              if C.USE_PRUNER else optuna.pruners.NopPruner())
    study = optuna.create_study(
        study_name=C.STUDY_NAME, direction="maximize",
        storage=f"sqlite:///{C.STUDY_DB}", load_if_exists=True,
        sampler=sampler, pruner=pruner)

    for s in seeds:
        try:
            study.enqueue_trial(s, skip_if_exists=True)
        except Exception:  # noqa: BLE001
            pass

    best_seen = baseline[0] if baseline else -np.inf
    t_start = time.time()
    state = {"best": best_seen, "n": 0}

    def cb(st: optuna.Study, tr: optuna.trial.FrozenTrial):
        state["n"] += 1
        if tr.value is None or tr.state != optuna.trial.TrialState.COMPLETE:
            return
        if tr.value > state["best"] + 1e-9:
            state["best"] = tr.value
            final, s_is, s_oos, m_is, m_oos, _ = obj.evaluate(tr.params)
            save_winner(C.CURRENT_WINNER, tr.params, final, m_is, uni,
                        extra={"score_is": s_is,
                               "score_oos": None if not np.isfinite(s_oos) else s_oos,
                               "oos_metrics": None if not m_oos else
                               {k: v for k, v in m_oos.items() if not k.startswith("_")},
                               "trial_number": tr.number})
            if verbose:
                print(f"  [{state['n']:>4}/{n_trials}] NEW BEST {final:.5f} "
                      f"(IS {s_is:.4f} / OOS {s_oos:.4f})  "
                      f"RoCaR={m_is.get('roi_on_capital_ann', 0):.1%}/yr  "
                      f"DD={m_is.get('max_dd_capital', 0):.1%}  "
                      f"trades={m_is.get('n_trades', 0)}  -> current_winner.json")
        elif verbose and state["n"] % 25 == 0:
            print(f"  [{state['n']:>4}/{n_trials}] best={state['best']:.5f} "
                  f"({(time.time() - t_start) / max(state['n'], 1):.2f}s/trial)")

    study.optimize(obj, n_trials=n_trials, n_jobs=C.N_JOBS, callbacks=[cb],
                   gc_after_trial=False, show_progress_bar=False)
    return study, obj, baseline
