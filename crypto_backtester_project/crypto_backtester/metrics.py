"""
Performance measurement and scoring  (plan step 12).

The ROI problem, and how it is solved here
------------------------------------------
The plan is explicit on three points: the monthly top-ups are deposits and not
profit, idle cash must not dilute the return, and shadow positions count for
nothing until real money is behind them.  That rules out the two obvious
formulas (final/contributions, and time-weighted return on total equity).

A third candidate -- chain-linking the daily return of the invested sleeve --
is *also* wrong here, and it is worth knowing why, because it is the trap most
homemade backtesters fall into.  When only one $1,000 position is open and it
gains 50 %, the sleeve return for that day is +50 %.  Chain-link a few hundred
days like that and you get a four-digit "ROI" sitting next to $13,000 of actual
profit.  Worse, an optimiser will *chase* it: the cheapest route to a huge
chain-linked number is to hold almost nothing, almost always.

So the headline number here is **return on capital at risk**:

    ROI  =  total trading P&L  /  average dollars actually invested

Because every tranche is a fixed $1,000 regardless of account size, this
strategy does not compound at the position level -- capacity grows linearly
with the deposits, not geometrically with returns.  A CAGR would therefore be a
fiction, so the annualised figure is the simple rate (ROI / years), and it
means exactly what it says: what one dollar at risk earned per year.

Drawdown is measured on the cumulative trading-P&L curve (equity minus
cumulative deposits), so a fresh $1,000 landing in the account can never paper
over a losing month.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

EPS = 1e-12


# ------------------------------------------------------------------ helpers --
def _max_dd_index(index: np.ndarray) -> tuple[float, int, int, int]:
    """Classic ratio drawdown for a positive index series."""
    if index.size == 0:
        return 0.0, 0, 0, 0
    peak = np.maximum.accumulate(index)
    dd = 1.0 - index / np.maximum(peak, EPS)
    i = int(np.argmax(dd))
    j = int(np.argmax(index[: i + 1])) if i > 0 else 0
    rec = index.size - i
    for k in range(i, index.size):
        if index[k] >= peak[i]:
            rec = k - i
            break
    return float(dd[i]), j, i, int(rec)


def _max_dd_pnl(pnl: np.ndarray, equity: np.ndarray) -> tuple[float, int, int, int]:
    """Drawdown of a cumulative P&L curve, normalised by the account equity at
    the running peak -- i.e. "what fraction of the account did the strategy
    hand back?".  Deposits cannot flatter it, because they are not in `pnl`."""
    if pnl.size == 0:
        return 0.0, 0, 0, 0
    peak = np.maximum.accumulate(pnl)
    at_peak = np.where(pnl >= peak - EPS, np.arange(pnl.size), -1)
    arg = np.maximum(np.maximum.accumulate(at_peak), 0)
    base = np.maximum(equity[arg], 2.0 * C.TRANCHE_USD)
    dd = (peak - pnl) / base
    i = int(np.argmax(dd))
    j = int(arg[i])
    rec = pnl.size - i
    for k in range(i, pnl.size):
        if pnl[k] >= peak[i]:
            rec = k - i
            break
    return float(max(dd[i], 0.0)), j, i, int(rec)


def xirr(dates, flows) -> float:
    d0 = pd.Timestamp(dates[0])
    yrs = np.array([(pd.Timestamp(x) - d0).days / 365.25 for x in dates])
    f = np.asarray(flows, float)

    def npv(r):
        return float(np.sum(f / np.power(1.0 + r, yrs)))

    lo, hi = -0.999, 10.0
    if npv(lo) * npv(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ================================================================== compute ===
def compute(res, uni=None) -> dict:
    if not res.ok:
        return {"valid": False, "reason": res.reason}

    dates = pd.DatetimeIndex(res.dates)
    days = max((dates[-1] - dates[0]).days, 1)
    years = days / 365.25

    tdf = pd.DataFrame(res.trades)
    if len(tdf) == 0:
        return {"valid": False, "reason": "no trades", "n_trades": 0}

    # --- capital actually at risk, and the trading-P&L curve ---------------
    cum_contrib = np.cumsum(res.contrib) + C.INITIAL_CASH
    pnl_curve = res.equity - cum_contrib  # cumulative trading profit, $
    basis = res.basis  # cost basis of open positions, $
    avg_cap = float(np.mean(basis))
    if avg_cap < C.TRANCHE_USD * 0.02:
        return {"valid": False, "reason": "effectively never invested", "n_trades": len(tdf)}
    peak_cap = float(np.max(basis))

    total_pnl_mtm = float(pnl_curve[-1])
    total_pnl = float(tdf["pnl"].sum())
    total_deployed = float(tdf["cost"].sum())

    roi_cap = total_pnl_mtm / max(avg_cap, EPS)
    roi_cap_ann = roi_cap / max(years, EPS)

    wins = tdf[tdf["pnl"] > 0]
    losses = tdf[tdf["pnl"] <= 0]
    gross_win = float(wins["pnl"].sum())
    gross_loss = float(-losses["pnl"].sum())

    # --- yearly, mark-to-market and contribution-neutral --------------------
    yr = dates.year.to_numpy()
    rows = []
    for y in np.unique(yr):
        m = yr == y
        idx = np.where(m)[0]
        p0 = pnl_curve[idx[0] - 1] if idx[0] > 0 else 0.0
        dp = float(pnl_curve[idx[-1]] - p0)
        cap_y = float(np.mean(basis[m]))
        tt = tdf[tdf["exit_year"] == y] if "exit_year" in tdf else tdf.iloc[0:0]
        rows.append({
            "year": int(y),
            "roi_year": dp / cap_y if cap_y > C.TRANCHE_USD * 0.02 else 0.0,
            "pnl_change": dp,
            "avg_capital": cap_y,
            "trades_closed": int(len(tt)),
            "realized_pnl": float(tt["pnl"].sum()) if len(tt) else 0.0,
            "end_equity": float(res.equity[idx[-1]]),
            "avg_invested": float(np.mean(res.mv[m])),
            "avg_cash_idle": float(np.mean(res.cash[m])),
            "idle_pct": float(np.mean(res.cash[m] / np.maximum(res.equity[m], EPS))),
            "avg_open": float(np.mean(res.n_open[m])),
            "max_open": int(np.max(res.n_open[m])),
            "avg_watchlist": float(np.mean(res.n_wl[m])),
            "contributions": float(np.sum(res.contrib[m])),
        })
    ydf = pd.DataFrame(rows)
    active = ydf[(ydf["trades_closed"] > 0) & (ydf["avg_capital"] > C.TRANCHE_USD * 0.02)]
    median_year = float(active["roi_year"].median()) if len(active) else 0.0

    # --- per coin -----------------------------------------------------------
    g = tdf.groupby("coin").agg(
        trades=("pnl", "size"), deployed=("cost", "sum"), pnl=("pnl", "sum"),
        win_rate=("pnl", lambda s: float((s > 0).mean())),
        med_ret=("ret", "median"), best=("ret", "max"), worst=("ret", "min"),
        avg_bars=("bars", "mean"),
    ).reset_index()
    g["roi"] = g["pnl"] / g["deployed"].clip(lower=EPS)
    median_coin = float(g["roi"].median()) if len(g) else 0.0

    # --- risk ---------------------------------------------------------------
    dd_cap, p0i, p1i, rec = _max_dd_pnl(pnl_curve, res.equity)
    dd_eq, _, _, _ = _max_dd_index(np.maximum(res.equity, EPS))

    dpnl = np.diff(pnl_curve, prepend=pnl_curve[0])
    daily_r = dpnl / max(avg_cap, EPS)  # constant denominator -> stable stats
    vol = float(np.std(daily_r)) * np.sqrt(365.25)
    dn = daily_r[daily_r < 0]
    voln = float(np.std(dn)) * np.sqrt(365.25) if dn.size else 0.0
    sharpe = roi_cap_ann / vol if vol > EPS else 0.0
    sortino = roi_cap_ann / voln if voln > EPS else 0.0
    calmar = roi_cap_ann / dd_cap if dd_cap > EPS else 0.0

    # --- secondary: chain-linked sleeve TWR (diagnostic only) ---------------
    r = np.clip(np.nan_to_num(res.r_sleeve), -0.999, 10.0)
    twr = np.cumprod(1.0 + r)

    # --- money weighted -----------------------------------------------------
    flows, fdates = [-C.INITIAL_CASH], [dates[0]]
    for i in range(len(dates)):
        if res.contrib[i] > 0:
            flows.append(-float(res.contrib[i]))
            fdates.append(dates[i])
    flows.append(float(res.equity[-1]))
    fdates.append(dates[-1])
    mwr = xirr(fdates, flows)

    out = {
        "valid": True,
        "start": str(dates[0].date()), "end": str(dates[-1].date()),
        "days": int(days), "years": years,
        # --- headline ROI family (all on money actually at risk) -----------
        "roi_on_capital": roi_cap,
        "roi_on_capital_ann": roi_cap_ann,
        "avg_capital_deployed": avg_cap,
        "peak_capital_deployed": peak_cap,
        "roi_turnover": total_pnl / max(total_deployed, EPS),
        "mwr_xirr": mwr,
        "median_year_return": median_year,
        "median_coin_roi": median_coin,
        "mean_coin_roi": float(g["roi"].mean()) if len(g) else 0.0,
        "pct_coins_profitable": float((g["roi"] > 0).mean()) if len(g) else 0.0,
        "sleeve_twr_total": float(twr[-1] - 1.0),
        # --- risk -----------------------------------------------------------
        "max_dd_capital": dd_cap,
        "max_dd_equity": dd_eq,
        "dd_peak_date": str(dates[p0i].date()), "dd_trough_date": str(dates[p1i].date()),
        "dd_recovery_days": rec,
        "ann_vol": vol, "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        # --- trade quality ---------------------------------------------------
        "n_trades": int(len(tdf)),
        "n_coins_traded": int(tdf["coin"].nunique()),
        "years_with_trades": int(len(active)),
        "total_deployed": total_deployed,
        "total_pnl": total_pnl,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > EPS else float("inf"),
        "win_rate": float((tdf["pnl"] > 0).mean()),
        "avg_win": float(wins["ret"].mean()) if len(wins) else 0.0,
        "avg_loss": float(losses["ret"].mean()) if len(losses) else 0.0,
        "expectancy": float(tdf["ret"].mean()),
        "median_trade_ret": float(tdf["ret"].median()),
        "avg_hold_days": float(tdf["bars"].mean()),
        "best_trade": float(tdf["ret"].max()), "worst_trade": float(tdf["ret"].min()),
        # --- capital usage ----------------------------------------------------
        "avg_exposure": float(np.mean(res.mv / np.maximum(res.equity, EPS))),
        "avg_idle_cash": float(np.mean(res.cash)),
        "final_equity": float(res.equity[-1]),
        "total_contributed": float(cum_contrib[-1]),
        "max_concurrent": int(res.stats.get("max_concurrent", 0)),
        "avg_watchlist": float(np.mean(res.n_wl)),
        "max_watchlist": int(np.max(res.n_wl)) if res.n_wl.size else 0,
        "watchlist_conversions": int(tdf["from_watchlist"].sum()) if "from_watchlist" in tdf else 0,
        "shadow_expired": int(res.stats.get("n_shadow_expired", 0)),
        "exit_mix": tdf["reason"].value_counts(normalize=True).round(4).to_dict(),
    }
    out["_yearly"] = ydf
    out["_by_coin"] = g.sort_values("roi", ascending=False)
    out["_twr"] = twr
    out["_pnl_curve"] = pnl_curve
    return out


# =================================================================== scoring ==
def _sq(x: float, scale: float) -> float:
    return float(np.tanh(x / max(scale, EPS)))


def score(m: dict, scale: float = 1.0) -> float:
    """Composite objective.

    Shape: a weighted, tanh-squashed blend of the four things the plan asks
    for, multiplied by a drawdown penalty and a sample-size confidence ramp.

    Squashing matters.  Without it a single 900 % outlier year dominates every
    other consideration and TPE spends the whole budget hunting lottery
    tickets; with it the marginal value of more return falls away, pushing the
    search towards configurations that are good on *all four* axes -- which is
    what "consistent" actually means.
    """
    S = C.SCORE
    if not m.get("valid"):
        return S["invalid_score"]
    # Guards are sized for the *full* history. A walk-forward holdout is only a
    # slice of it, so scale them down or every holdout is rejected as "too few
    # trades" and the out-of-sample check silently stops meaning anything.
    sc = max(min(float(scale), 1.0), 0.05)
    min_tr = max(12, int(round(S["min_trades"] * sc)))
    min_co = max(3, int(round(S["min_coins_traded"] * sc)))
    min_yr = max(1, int(round(S["min_years_with_trades"] * sc)))
    if (m["n_trades"] < min_tr
            or m["n_coins_traded"] < min_co
            or m["years_with_trades"] < min_yr):
        return S["invalid_score"]

    dd = m["max_dd_capital"]
    if dd >= S["dd_hard_cap"]:
        return S["invalid_score"] + 1.0 - min(dd, 5.0)

    pf = m["profit_factor"]
    pf = 3.0 if not np.isfinite(pf) else pf

    w = np.array([S["w_total_roi"], S["w_median_year"],
                  S["w_median_coin"], S["w_profit_factor"]], float)
    w = w / w.sum()
    terms = np.array([
        _sq(m["roi_on_capital_ann"], S["scale_total_roi"]),
        _sq(m["median_year_return"], S["scale_median_year"]),
        _sq(m["median_coin_roi"], S["scale_median_coin"]),
        _sq(pf - 1.0, S["scale_profit_factor"]),
    ])
    base = float(np.dot(w, terms))

    dd_factor = 1.0 / (1.0 + (dd / max(S["dd_tolerance"], EPS)) ** S["dd_power"])
    conf = min(1.0, m["n_trades"] / max(S["target_trades"] * sc, 10.0)) ** 0.25
    # a drawdown penalty must never *improve* a negative base
    return float(base * (dd_factor if base >= 0 else (2.0 - dd_factor)) * conf)


def combine_is_oos(s_is: float, s_oos: float) -> float:
    """Blend in-sample with the holdout.

    Applied as a drag rather than an average: the result can never sit far
    above what the untouched window says it should be, so a configuration that
    memorised the training data cannot win on its training score alone.
    """
    if not np.isfinite(s_oos):
        return s_is
    w = C.OOS_PENALTY_WEIGHT
    return float((1.0 - w) * s_is + w * min(s_is, s_oos))
