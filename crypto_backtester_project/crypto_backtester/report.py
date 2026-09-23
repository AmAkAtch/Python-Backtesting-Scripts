"""
The "explain the winner to a sceptic" document (plan step 15).

Writes WINNER_REPORT.md plus machine-readable side-cars (trades, equity,
yearly, per-coin CSVs) so every claim in the document can be re-derived.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config as C
from .search import describe, materialize


# ------------------------------------------------------------------ helpers --
def _pct(x, nd=2):
    try:
        v = float(x)
        return "n/a" if not np.isfinite(v) else f"{v * 100:.{nd}f}%"
    except Exception:  # noqa: BLE001
        return "n/a"


def _num(x, nd=2):
    try:
        v = float(x)
        return "inf" if not np.isfinite(v) else f"{v:,.{nd}f}"
    except Exception:  # noqa: BLE001
        return "n/a"


def _usd(x, nd=0):
    try:
        return f"${float(x):,.{nd}f}"
    except Exception:  # noqa: BLE001
        return "n/a"


def _md_table(df: pd.DataFrame, floatfmt="{:,.4f}") -> str:
    if df is None or len(df) == 0:
        return "_(empty)_\n"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
        else:
            d[c] = d[c].astype(str)
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    rows = ["| " + " | ".join(r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep] + rows) + "\n"


def _dd_episodes(pnl: np.ndarray, equity: np.ndarray, dates, top=6) -> pd.DataFrame:
    """Drawdown episodes of the cumulative trading-P&L curve."""
    peak = np.maximum.accumulate(pnl)
    under = pnl < peak - 1e-9
    eps, i, n = [], 0, len(pnl)
    while i < n:
        if not under[i]:
            i += 1
            continue
        j = i
        while j < n and under[j]:
            j += 1
        seg = pnl[i:j]
        k = int(np.argmin(seg))
        base = max(float(equity[max(i - 1, 0)]), 2.0 * C.TRANCHE_USD)
        eps.append({
            "peak": str(pd.Timestamp(dates[max(i - 1, 0)]).date()),
            "trough": str(pd.Timestamp(dates[i + k]).date()),
            "recovered": str(pd.Timestamp(dates[min(j, n - 1)]).date()) if j < n else "not yet",
            "depth_usd": float(peak[i] - seg[k]),
            "depth_pct": float((peak[i] - seg[k]) / base),
            "days": int(j - i),
            "days_to_recover": int(j - i - k),
        })
        i = j
    eps.sort(key=lambda e: -e["depth_pct"])
    return pd.DataFrame(eps[:top])


def _block_bootstrap(daily: np.ndarray, years: float, n_sims=2000, block=21, seed=11):
    """Stationary block bootstrap of the daily P&L-on-capital series."""
    n = daily.size
    if n < block * 4 or years <= 0:
        return None
    rng = np.random.default_rng(seed)
    rois = np.empty(n_sims)
    dds = np.empty(n_sims)
    for s in range(n_sims):
        parts, k = [], 0
        while k < n:
            st = int(rng.integers(0, n - block))
            parts.append(daily[st:st + block])
            k += block
        x = np.concatenate(parts)[:n]
        cum = np.cumsum(x)
        rois[s] = cum[-1] / years
        peak = np.maximum.accumulate(cum)
        dds[s] = float(np.max(peak - cum))
    return {"roi_ann": rois, "dd": dds}


# =================================================================== writer ===
def write_report(uni, raw_params: dict, sc: float, m: dict, res,
                 m_oos: dict | None = None, res_oos=None, study=None,
                 baseline=None, path=None, label="WINNER") -> str:
    p = materialize(raw_params)
    path = path or C.REPORT_MD
    out_dir = C.OUT_DIR
    dates = pd.DatetimeIndex(res.dates)
    pnl_curve = m["_pnl_curve"]
    ydf = m["_yearly"].copy()
    cdf = m["_by_coin"].copy()
    tdf = pd.DataFrame(res.trades)
    avg_cap = m["avg_capital_deployed"]

    # ---------- side-car CSVs ------------------------------------------------
    tdf.to_csv(out_dir / "winner_trades.csv", index=False)
    pd.DataFrame({
        "date": dates, "equity": res.equity, "invested_value": res.mv,
        "cash": res.cash, "cost_basis": res.basis,
        "cum_contributions": np.cumsum(res.contrib) + C.INITIAL_CASH,
        "cum_trading_pnl": pnl_curve,
        "open_positions": res.n_open, "watchlist": res.n_wl,
        "contribution": res.contrib, "sleeve_return": res.r_sleeve,
    }).to_csv(out_dir / "winner_equity.csv", index=False)
    ydf.to_csv(out_dir / "winner_yearly.csv", index=False)
    cdf.to_csv(out_dir / "winner_by_coin.csv", index=False)
    if res.missed:
        pd.DataFrame(res.missed).to_csv(out_dir / "winner_shadow_expired.csv", index=False)

    acc = [r for r in uni.log if r["status"] == "accepted"]
    rej = [r for r in uni.log if r["status"] == "rejected"]
    daily = np.diff(pnl_curve, prepend=pnl_curve[0]) / max(avg_cap, 1e-9)
    boot = _block_bootstrap(daily, m["years"])

    A: list[str] = []
    w = A.append

    # ======================================================== 1. front matter
    w(f"# {label} CONFIGURATION — Full Validation Dossier\n")
    w(f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
      f"composite score **{sc:.5f}** · Optuna study `{C.STUDY_NAME}`*\n")
    w("> This document exists so that somebody who does not trust the optimiser can check it. "
      "Every number below is reproducible from the CSVs written next to this file.\n")

    # ======================================================== 2. exec summary
    w("## 1. Executive summary\n")
    summary = pd.DataFrame([
        ("Backtest window", f"{m['start']} → {m['end']}  ({m['years']:.2f} years)"),
        ("Coins in universe", f"{uni.n} (traded: {m['n_coins_traded']})"),
        ("**Total ROI on capital at risk**", f"**{_pct(m['roi_on_capital'])}**"),
        ("**Annualised ROI on capital at risk**", f"**{_pct(m['roi_on_capital_ann'])} / yr**"),
        ("Average capital actually invested", _usd(avg_cap)),
        ("Peak capital invested at once", _usd(m["peak_capital_deployed"])),
        ("Median calendar-year ROI", _pct(m["median_year_return"])),
        ("Median per-coin ROI", _pct(m["median_coin_roi"])),
        ("**Max drawdown (deposit-adjusted)**", f"**{_pct(m['max_dd_capital'])}**"),
        ("Max drawdown (raw account curve)", _pct(m["max_dd_equity"])),
        ("Profit factor", _num(m["profit_factor"])),
        ("Win rate", _pct(m["win_rate"])),
        ("Closed tranches", f"{m['n_trades']:,}"),
        ("Sharpe / Sortino / Calmar",
         f"{_num(m['sharpe'])} / {_num(m['sortino'])} / {_num(m['calmar'])}"),
        ("Money-weighted return (XIRR on deposits)", _pct(m["mwr_xirr"])),
        ("Total cash paid in", _usd(m["total_contributed"])),
        ("Final account equity", _usd(m["final_equity"])),
        ("Total trading profit", _usd(m["total_pnl"])),
        ("Max positions held at once", f"{m['max_concurrent']}"),
    ], columns=["Metric", "Value"])
    w(_md_table(summary))

    w(f"""
**Reconciling the two headline numbers.** The account took in
{_usd(m['total_contributed'])} of deposits and finished at {_usd(m['final_equity'])}, so the
strategy itself produced {_usd(m['total_pnl'])}. That profit was earned by an average of only
{_usd(avg_cap)} being invested at any one moment — the rest of the balance was idle cash waiting
for a signal. Dividing the profit by the money that was actually at risk gives the
{_pct(m['roi_on_capital'])} headline, or {_pct(m['roi_on_capital_ann'])} per year, which is the
figure the plan asks for: ROI on invested capital, with idle cash and deposits excluded. The
XIRR of {_pct(m['mwr_xirr'])} is the same result seen from the depositor's chair, and it is lower
precisely because so much of each deposit sat in cash.
""")

    # ======================================================== 3. the strategy
    w("\n## 2. What this strategy actually does, in words\n")
    w(describe(p) + "\n")

    # ======================================================== 4. parameters
    w("\n## 3. Exact parameter set\n")
    w("Architecture: **" + C.ENTRY_NAMES[p["entry_type"]] + "** entry → **"
      + C.EXIT_NAMES[p["exit_type"]] + "** exit → **" + C.WL_NAMES[p["wl_mode"]] + "** queueing.\n")
    prows = []
    for k in sorted(p):
        v = p[k]
        note = ""
        if k == "entry_type":
            note = C.ENTRY_NAMES[v]
        elif k == "exit_type":
            note = C.EXIT_NAMES[v]
        elif k == "wl_mode":
            note = C.WL_NAMES[v]
        elif k.endswith("_type") and isinstance(v, int):
            note = C.MA_NAMES.get(v, "")
        elif k == "xover_long_len":
            note = "derived = min(300, short_len + gap)"
        prows.append((k, v, note))
    w(_md_table(pd.DataFrame(prows, columns=["Parameter", "Value", "Meaning"])))
    w("\nParameters absent from this table were never sampled: the chosen architecture does not "
      "use them, so the optimiser spent no search density on them.\n")

    # ======================================================== 5. the universe
    w("\n## 4. The universe this was traded on\n")
    w(f"Candidates ranked by **{C.UNIVERSE_SOURCE}**, filtered, then aligned onto one daily grid. "
      f"Price data source(s): {', '.join(uni.meta['providers'])}.\n")
    adf = pd.DataFrame(acc)
    cols = [c for c in ["base", "reason", "rank", "market_cap", "bars", "first", "last", "provider"]
            if c in adf.columns]
    adf = adf[cols]
    adf.columns = [{"base": "Coin", "reason": "Why included", "rank": "Rank",
                    "market_cap": "Market cap", "bars": "Bars", "first": "First bar",
                    "last": "Last bar", "provider": "Source"}[c] for c in cols]
    w(_md_table(adf, floatfmt="{:,.0f}"))

    w(f"\n**Rejected ({len(rej)}).** Noise removal is not cosmetic here: a stablecoin cannot trend, "
      "and a wrapped or liquid-staked token is the same bet you already hold, so keeping either "
      "one silently doubles a position and flatters diversification.\n")
    rdf = pd.DataFrame(rej)
    if len(rdf):
        counts = rdf["reason"].str.split("(").str[0].str.strip().value_counts()
        w(_md_table(counts.rename_axis("Rejection reason").reset_index(name="Count")))
        w("\n<details><summary>Full rejection list</summary>\n\n")
        w(_md_table(rdf[["base", "reason"]].rename(columns={"base": "Coin", "reason": "Reason"})))
        w("\n</details>\n")

    w(f"\n**Start-date logic.** The plan asks for the first date on which "
      f"{C.START_COVERAGE:.0%} of the shortlisted coins were tradeable. That bar was reached at "
      f"{uni.meta['coverage_at_start']:.1%} coverage; a further {C.WARMUP_CAP}-bar warm-up buffer "
      f"is added on top so that even a 300-day moving average is fully formed before the first "
      f"possible trade. Data grid: {uni.meta['grid_start']} → {uni.meta['grid_end']}; trading "
      f"begins {m['start']}.\n")

    # ======================================================== 6. capital rules
    w("\n## 5. Capital, cost and execution assumptions\n")
    cap = pd.DataFrame([
        ("Cash on day one", _usd(C.INITIAL_CASH)),
        ("New cash each month", f"{_usd(C.MONTHLY_CONTRIBUTION)} on the first bar of the month"),
        ("Size per entry", f"{_usd(C.TRANCHE_USD)} all-in (fees and slippage come out of it)"),
        ("Taker fee", f"{C.TAKER_FEE_BPS:.1f} bps per side"),
        ("Slippage", f"{C.SLIPPAGE_BPS:.1f} bps per side"
                     + (f" + {C.SLIPPAGE_ATR_COEF} × ATR/price" if C.SLIPPAGE_ATR_COEF else "")),
        ("Tax on realised gains", _pct(C.TAX_RATE_ON_GAINS, 0)),
        ("Signal → fill", "signal at the close of bar t, filled at the OPEN of bar t+1"),
        ("Stop fills", "at min(open, stop) — gap-downs are not handed back to you"),
        ("Stop vs take-profit in one bar", "the stop is assumed to hit first (pessimistic)"),
        ("Liquidity floor", f"{C.LIQUIDITY_WINDOW}-day average dollar volume ≥ "
                            f"{_usd(C.LIQUIDITY_FLOOR_USD)}"),
        ("Delisting policy", f"halted ≥ {C.DELIST_GAP_DAYS} bars → written off at "
                             f"{C.DELIST_VALUATION:.0%} of the last close"),
    ], columns=["Rule", "Setting"])
    w(_md_table(cap))

    # ======================================================== 7. year by year
    w("\n## 6. Year by year, and how the portfolio grew\n")
    y = ydf.copy()
    disp = pd.DataFrame({
        "Year": y["year"],
        "ROI on capital": y["roi_year"].map(lambda v: _pct(v)),
        "P&L change": y["pnl_change"].map(_usd),
        "Avg capital at risk": y["avg_capital"].map(_usd),
        "Trades closed": y["trades_closed"],
        "Realised P&L": y["realized_pnl"].map(_usd),
        "End equity": y["end_equity"].map(_usd),
        "Avg idle cash": y["avg_cash_idle"].map(_usd),
        "Idle %": y["idle_pct"].map(lambda v: _pct(v, 1)),
        "Avg open": y["avg_open"].round(2),
        "Max open": y["max_open"],
        "Avg watchlist": y["avg_watchlist"].round(2),
        "Paid in": y["contributions"].map(_usd),
    })
    w(_md_table(disp))
    w(f"\nMedian of the calendar-year ROIs (years with at least one closed trade): "
      f"**{_pct(m['median_year_return'])}**. Best year {_pct(y['roi_year'].max())}, worst year "
      f"{_pct(y['roi_year'].min())}. 'P&L change' is mark-to-market and deposit-neutral, so an "
      "open winner is credited to the year it rose in rather than the year it was finally sold.\n")
    w(f"\n**Idle cash.** The average uninvested balance across the whole test was "
      f"{_usd(m['avg_idle_cash'])}, i.e. average exposure of {_pct(m['avg_exposure'])}. This is "
      "deliberately excluded from the ROI numbers — the strategy is judged on money it actually "
      "risked — but the 'Idle %' column is the single most important capacity statistic in this "
      "document. It rises whenever deposits arrive faster than signals do, and it is the reason "
      "the account-level XIRR is so much lower than the return on capital at risk. If that column "
      "trends upward, the rules are too selective for the deposit schedule and either the tranche "
      "size, the layer cap or the entry filters need loosening.\n")

    # ======================================================== 8. drawdowns
    w("\n## 7. Drawdown anatomy\n")
    w(f"Deepest deposit-adjusted drawdown: **{_pct(m['max_dd_capital'])}** of account equity, "
      f"peaking {m['dd_peak_date']} and troughing {m['dd_trough_date']}, "
      f"{m['dd_recovery_days']} days to make it back.\n")
    eps = _dd_episodes(pnl_curve, res.equity, dates)
    if len(eps):
        e2 = eps.copy()
        e2["depth_usd"] = e2["depth_usd"].map(_usd)
        e2["depth_pct"] = e2["depth_pct"].map(lambda v: _pct(v))
        e2.columns = ["Peak", "Trough", "Recovered", "Depth ($)", "Depth (%)",
                      "Total days", "Days to recover"]
        w(_md_table(e2))
    w(f"\nThe raw account curve shows a smaller fall ({_pct(m['max_dd_equity'])}) purely because "
      "fresh deposits keep landing in it while positions are losing. The deposit-adjusted figure "
      "measures only what the strategy handed back, and is the one to budget against.\n")

    # ======================================================== 9. trade stats
    w("\n## 8. Trade statistics\n")
    ts = pd.DataFrame([
        ("Closed tranches", f"{m['n_trades']:,}"),
        ("Win rate", _pct(m["win_rate"])),
        ("Profit factor", _num(m["profit_factor"])),
        ("Average return per tranche", _pct(m["expectancy"])),
        ("Median return per tranche", _pct(m["median_trade_ret"])),
        ("Average winner / loser", f"{_pct(m['avg_win'])} / {_pct(m['avg_loss'])}"),
        ("Best / worst tranche", f"{_pct(m['best_trade'])} / {_pct(m['worst_trade'])}"),
        ("Average holding period", f"{m['avg_hold_days']:.1f} days"),
        ("Total capital pushed through trades", _usd(m["total_deployed"])),
        ("Turnover ROI (P&L ÷ capital deployed)", _pct(m["roi_turnover"])),
    ], columns=["Statistic", "Value"])
    w(_md_table(ts))

    if len(tdf):
        q = tdf["ret"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        w("\nPer-tranche return distribution (5 / 25 / 50 / 75 / 95th percentile): "
          + " | ".join(_pct(v) for v in q.values) + "\n")
        mix = pd.DataFrame(sorted(m["exit_mix"].items(), key=lambda kv: -kv[1]),
                           columns=["Exit reason", "Share"])
        mix["Share"] = mix["Share"].map(lambda v: _pct(v, 1))
        w("\n**How positions actually ended.** This is the most diagnostic table in the document: "
          "if one reason dominates, that one rule *is* the strategy and everything else is "
          "decoration.\n")
        w(_md_table(mix))
        if "layer" in tdf:
            lay = tdf.groupby("layer").agg(trades=("ret", "size"), avg_ret=("ret", "mean"),
                                           pnl=("pnl", "sum")).reset_index()
            lay["avg_ret"] = lay["avg_ret"].map(lambda v: _pct(v))
            lay["pnl"] = lay["pnl"].map(_usd)
            lay.columns = ["Pyramid layer", "Trades", "Avg return", "Total P&L"]
            w("\n**Do the add-on layers earn their keep?** If layer 2+ returns less than layer 1, "
              "pyramiding is averaging up into weakness and the layer cap should come down.\n")
            w(_md_table(lay))

    # ======================================================== 10. per coin
    w("\n## 9. Per-coin breakdown\n")
    c2 = cdf.copy()
    for col in ("roi", "med_ret", "best", "worst"):
        c2[col] = c2[col].map(lambda v: _pct(v))
    c2["win_rate"] = c2["win_rate"].map(lambda v: _pct(v, 0))
    c2["deployed"] = c2["deployed"].map(_usd)
    c2["pnl"] = c2["pnl"].map(_usd)
    c2["avg_bars"] = c2["avg_bars"].round(1)
    c2 = c2[["coin", "trades", "deployed", "pnl", "win_rate", "med_ret",
             "best", "worst", "avg_bars", "roi"]]
    c2.columns = ["Coin", "Trades", "Deployed", "P&L", "Win rate", "Median trade",
                  "Best", "Worst", "Avg days", "Coin ROI"]
    w(_md_table(c2))
    w(f"\n**{_pct(m['pct_coins_profitable'], 0)}** of traded coins finished positive; the median "
      f"coin returned {_pct(m['median_coin_roi'])} on the capital it was given.\n")
    gross = cdf["pnl"].clip(lower=0).sum()
    if gross > 0:
        top3 = cdf.nlargest(3, "pnl")
        w(f"\nConcentration check: the three best coins ({', '.join(top3['coin'])}) produced "
          f"**{_pct(top3['pnl'].sum() / gross, 1)}** of all gross profit. A result that hides "
          "entirely inside one or two names is a result that got lucky, not a strategy — this is "
          "the first number to look at before believing anything else in this document.\n")

    # ======================================================== 11. watchlist
    w("\n## 10. How the watchlist behaved\n")
    if p["wl_mode"] == 5:
        w("This configuration chose **no watchlist at all**. A signal that fired while the wallet "
          "was empty was simply missed, and the next signal after cash freed up was taken instead. "
          "The optimiser preferred that to all five queueing rules, which is itself informative: "
          "for this parameter set, a stale queued signal was worth less than a fresh one.\n")
    else:
        waited = tdf[tdf["from_watchlist"]] if "from_watchlist" in tdf else tdf.iloc[0:0]
        fresh = tdf[~tdf["from_watchlist"]] if "from_watchlist" in tdf else tdf.iloc[0:0]
        w(f"Queueing rule: **{C.WL_NAMES[p['wl_mode']]}**"
          + (f", ranked on {p['wl_mom_lookback']}-day momentum"
             + (" measured relative to BTC" if p.get("wl_mom_vs_btc") else "")
             if p["wl_mode"] == 2 else "")
          + (f", discarding anything older than {p['wl_max_age']} days" if p.get("wl_max_age") else "")
          + ".\n")
        wl_tbl = pd.DataFrame([
            ("Average shadow positions held", _num(m["avg_watchlist"])),
            ("Largest the watchlist ever got", f"{m['max_watchlist']}"),
            ("Shadow positions that became real trades", f"{m['watchlist_conversions']:,}"),
            ("Shadow positions that died before funding", f"{m['shadow_expired']:,}"),
            ("Average wait before funding",
             f"{waited['wait_days'].mean():.1f} days" if len(waited) else "n/a"),
            ("Avg return — funded from the watchlist",
             _pct(waited["ret"].mean()) if len(waited) else "n/a"),
            ("Avg return — funded immediately",
             _pct(fresh["ret"].mean()) if len(fresh) else "n/a"),
        ], columns=["Watchlist metric", "Value"])
        w(_md_table(wl_tbl))
        w("\nThe last two rows are what justify the queue. If waiting trades do *worse* than fresh "
          "ones, the watchlist is buying drift rather than opportunity and `WL_NONE` would have "
          "been the better answer; the fact that the optimiser kept the queue means the ranking "
          "rule was adding something, but check the size of the gap before trusting it.\n")

    # ======================================================== 12. OOS
    w("\n## 11. Out-of-sample check\n")
    if m_oos and m_oos.get("valid"):
        oo = pd.DataFrame([
            ("Window", f"{m_oos['start']} → {m_oos['end']}"),
            ("Annualised ROI on capital", _pct(m_oos["roi_on_capital_ann"])),
            ("Median year ROI", _pct(m_oos["median_year_return"])),
            ("Median coin ROI", _pct(m_oos["median_coin_roi"])),
            ("Max drawdown", _pct(m_oos["max_dd_capital"])),
            ("Profit factor", _num(m_oos["profit_factor"])),
            ("Win rate", _pct(m_oos["win_rate"])),
            ("Trades", f"{m_oos['n_trades']:,}"),
        ], columns=["Out-of-sample", "Value"])
        w(_md_table(oo))
        deg = m_oos["roi_on_capital_ann"] - m["roi_on_capital_ann"]
        word = "Decay" if deg < 0 else "Improvement"
        w(f"\n{word} from the training window to the untouched one: **{_pct(deg)}** "
          f"({_pct(m['roi_on_capital_ann'])} → {_pct(m_oos['roi_on_capital_ann'])} per year). "
          "Some decay is normal and expected. A collapse to zero or negative means the "
          "configuration memorised the training window, and the composite score already docks it "
          "for that — the final score blends in-sample with the *worse* of the two.\n")
    elif C.USE_WALK_FORWARD:
        w("_The holdout window produced too few trades to measure. That is itself a finding: this "
          "configuration is not active enough in recent conditions to be trusted forward._\n")
    else:
        w("_Walk-forward was disabled (`USE_WALK_FORWARD = False`), so every number in this "
          "document is in-sample. Treat them as an upper bound, not an expectation._\n")

    # ======================================================== 13. robustness
    w("\n## 12. Robustness, and how much of this is luck\n")
    if boot:
        w("**Block bootstrap** — 2,000 resamples of the daily P&L-on-capital series in 21-day "
          "blocks, which preserves short-horizon autocorrelation while destroying the specific "
          "order history happened to deal:\n")
        bt = pd.DataFrame([
            ("Annualised ROI — 5th percentile", _pct(np.percentile(boot["roi_ann"], 5))),
            ("Annualised ROI — median", _pct(np.percentile(boot["roi_ann"], 50))),
            ("Annualised ROI — 95th percentile", _pct(np.percentile(boot["roi_ann"], 95))),
            ("Probability of a positive result", _pct((boot["roi_ann"] > 0).mean(), 1)),
            ("Max drawdown — median", _usd(np.percentile(boot["dd"], 50) * avg_cap)),
            ("Max drawdown — 95th percentile (plan for this)",
             _usd(np.percentile(boot["dd"], 95) * avg_cap)),
        ], columns=["Bootstrap statistic", "Value"])
        w(_md_table(bt))
        w("\nRead the 95th-percentile drawdown as the one to budget for. The realised drawdown is "
          "a single draw from this distribution, not a ceiling.\n")

    if study is not None:
        try:
            import optuna
            comp = [t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
            pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
            vals = np.array([t.value for t in comp]) if comp else np.array([0.0])
            w(f"\n**Search context.** {len(comp):,} completed trials, {pruned:,} pruned. "
              f"Best {vals.max():.5f}, median {np.median(vals):.5f}, top-decile threshold "
              f"{np.percentile(vals, 90):.5f}. The gap between the winner and the top decile is a "
              "rough proxy for selection luck: if the winner towers over a dense cluster of "
              "near-identical scores, it is probably an outlier draw rather than a better idea.\n")
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imp = optuna.importance.get_param_importances(study)
            idf = pd.DataFrame(list(imp.items())[:15], columns=["Parameter", "Importance"])
            idf["Importance"] = idf["Importance"].map(lambda v: f"{v:.3f}")
            w("\n**What the score was actually sensitive to** (fANOVA importance):\n")
            w(_md_table(idf))
            w("\nParameters at the top are the ones worth re-validating by hand — nudge them ±20 % "
              "and re-run `validate`. Parameters near zero can be perturbed freely, which is good "
              "news: it means the winner is not balanced on a knife edge in those dimensions.\n")
        except Exception as e:  # noqa: BLE001
            w(f"\n_(parameter importance unavailable: {e})_\n")

    if baseline:
        w(f"\n**The bar it had to clear.** The stored champion in `{baseline[2]}` re-scored "
          f"{baseline[0]:.5f} on this data before the run started.\n")

    # ======================================================== 14. caveats
    w("\n## 13. Known biases — read this before risking money\n")
    w(f"""
1. **Survivorship.** The candidate list is ranked by *today's* market cap, so coins that died are
   structurally under-represented. {len(C.INCLUDE_DELISTED)} known-dead tickers are force-added via
   `INCLUDE_DELISTED`, and halted coins are written off at {C.DELIST_VALUATION:.0%} of last close.
   That is a patch, not a cure. The real fix is a point-in-time universe snapshot per rebalance
   date, which no free API provides cleanly.
2. **Look-ahead in universe selection.** Nobody knew in 2018 which of these names would matter.
   The $3 M liquidity floor mitigates it — a coin cannot be traded before it was liquid — but it
   does not remove it.
3. **Multiple testing.** This report describes the best of many configurations. With enough trials
   something always looks good in-sample; that is why section 11 and section 12 exist, and why the
   composite score is dragged toward the holdout result rather than averaged with it.
4. **Single quote asset, single venue.** All pairs are {C.QUOTE_ASSET} on one exchange family.
   Fills for a {_usd(C.TRANCHE_USD)} clip are realistic; the same rules at $100,000 a clip are not.
5. **Long-only spot.** No shorting, no leverage, no funding costs, no borrow.
6. **Daily bars.** The intrabar path is unknown, so stop-versus-target ordering inside one bar is
   resolved pessimistically. Real fills will differ, usually for the worse on gap days.
7. **Fees and slippage are assumptions**, not measurements: {C.TAKER_FEE_BPS:.0f} bps +
   {C.SLIPPAGE_BPS:.0f} bps per side. Re-run with `SLIPPAGE_BPS` doubled. If the edge disappears,
   it was never an edge — it was a rebate on an unrealistic fill.
8. **Deposits interact with results.** A strategy that trades rarely looks better per dollar at
   risk and worse per dollar deposited. Both numbers are in section 1 for exactly that reason.
""")

    # ======================================================== 15. how to check
    w("\n## 14. How to verify every claim here\n")
    w(f"""
| File | What it lets you check |
| --- | --- |
| `winner_trades.csv` | Every closed tranche: coin, both dates, both prices, cost, proceeds, P&L, layer, exit reason |
| `winner_equity.csv` | Daily cash, invested value, cost basis, cumulative deposits, cumulative trading P&L, open positions, watchlist depth |
| `winner_yearly.csv` | The year-by-year table above |
| `winner_by_coin.csv` | The per-coin table above |
| `winner_shadow_expired.csv` | Shadow positions that died before funding — the trades you did *not* take |
| `universe_selection.json` | Every coin considered and the exact reason it was kept or dropped |
| `winner_metrics.json` | Every metric in this document, machine-readable |
| `{C.FINAL_WINNER.name}` / `{C.CURRENT_WINNER.name}` | The parameter set, its score, and a fingerprint of the data and settings it was scored on |

Run `python run.py validate` at any time to re-score the stored winner against freshly downloaded
data. If the score moves materially, the edge was specific to one data snapshot rather than
structural.
""")

    text = "\n".join(A)
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(text)
    with open(out_dir / "winner_metrics.json", "w") as f:
        json.dump({k: v for k, v in m.items() if not k.startswith("_")},
                  f, indent=2, default=str)
    return str(path)
