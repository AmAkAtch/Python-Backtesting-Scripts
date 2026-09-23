# SELF-TEST CONFIGURATION — Full Validation Dossier

*Generated 2026-09-21 12:00 UTC · composite score **0.32979** · Optuna study `crypto_swing_v1`*

> This document exists so that somebody who does not trust the optimiser can check it. Every number below is reproducible from the CSVs written next to this file.

## 1. Executive summary

| Metric | Value |
| --- | --- |
| Backtest window | 2019-01-24 → 2022-10-13  (3.72 years) |
| Coins in universe | 12 (traded: 12) |
| **Total ROI on capital at risk** | **385.57%** |
| **Annualised ROI on capital at risk** | **103.70% / yr** |
| Average capital actually invested | $2,028 |
| Peak capital invested at once | $11,000 |
| Median calendar-year ROI | 61.25% |
| Median per-coin ROI | 1.87% |
| **Max drawdown (deposit-adjusted)** | **9.86%** |
| Max drawdown (raw account curve) | 9.45% |
| Profit factor | 1.73 |
| Win rate | 43.25% |
| Closed tranches | 326 |
| Sharpe / Sortino / Calmar | 1.00 / 1.01 / 10.52 |
| Money-weighted return (XIRR on deposits) | 8.36% |
| Total cash paid in | $46,000 |
| Final account equity | $53,821 |
| Total trading profit | $7,821 |
| Max positions held at once | 11 |


**Reconciling the two headline numbers.** The account took in
$46,000 of deposits and finished at $53,821, so the
strategy itself produced $7,821. That profit was earned by an average of only
$2,028 being invested at any one moment — the rest of the balance was idle cash waiting
for a signal. Dividing the profit by the money that was actually at risk gives the
385.57% headline, or 103.70% per year, which is the
figure the plan asks for: ROI on invested capital, with idle cash and deposits excluded. The
XIRR of 8.36% is the same result seen from the depositor's chair, and it is lower
precisely because so much of each deposit sat in cash.


## 2. What this strategy actually does, in words

1. Buy when the daily close crosses above its 15-day SMA.
2. Only when BTC is above its 209-day DEMA (macro risk-on gate).
3. Only when ADX(14) >= 15 (trend must be real, not chop).
4. Never touch a coin whose trailing 30-day average dollar volume is under $3,000,000.
5. Each buy is a flat $1,000, all-in. Up to 3 layer(s) per coin, at least 16 day(s) apart.
6. Signals that arrive with no cash become shadow positions and are queued; when cash frees up the queue is filled by WL_FCFS, dropping anything older than 90 days.
7. A shadow position that hits its exit rule before it is ever funded is deleted, never traded.
8. Exit: initial stop 4.41x ATR(14) below entry, then a 2.35x ATR chandelier trail that only ever ratchets up.
9. Global partial take-profit: at entry + 18.2x ATR(14), sell 47% of the layer.
10. Macro kill-switch: if BTC closes under its 209-day DEMA, every altcoin position is liquidated at the next open.
11. Any coin that is halted or delisted is written off per the configured policy.


## 3. Exact parameter set

Architecture: **ENTRY_MA_BREAKOUT** entry → **EXIT_HYBRID_ATR** exit → **WL_FCFS** queueing.

| Parameter | Value | Meaning |
| --- | --- | --- |
| adx_thresh | 15.0 |  |
| btc_ma_len | 209 |  |
| btc_ma_type | 2 | DEMA |
| entry_ma_len | 15 |  |
| entry_ma_type | 0 | SMA |
| entry_require_cross | True |  |
| entry_type | 0 | ENTRY_MA_BREAKOUT |
| exit_type | 0 | EXIT_HYBRID_ATR |
| max_pyramid_layers | 3 |  |
| pyramid_min_gap | 16 |  |
| pyramid_require_profit | False |  |
| safety_sl_mult | 0.0 |  |
| sl_mult | 4.410202077542829 |  |
| tp_move_sl_be | False |  |
| tp_mult | 18.164068694388806 |  |
| tp_size_pct | 46.742176828270495 |  |
| trail_mult | 2.349616469557302 |  |
| use_btc_entry_gate | True |  |
| use_btc_exit_override | True |  |
| use_global_tp | True |  |
| use_rsi_trend_filter | False |  |
| use_safety_stop | False |  |
| wl_max_age | 90 |  |
| wl_mode | 3 | WL_FCFS |
| wl_mom_lookback | 20 |  |
| wl_mom_vs_btc | False |  |


Parameters absent from this table were never sampled: the chosen architecture does not use them, so the optimiser spent no search density on them.


## 4. The universe this was traded on

Candidates ranked by **coingecko**, filtered, then aligned onto one daily grid. Price data source(s): synthetic.

| Coin | Why included | Rank | Market cap | Bars | First bar | Last bar | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BTC | synthetic | 0 | 1,000,000,000 | 2200 | 2018-01-01 | 2024-01-09 | synthetic |
| C01 | synthetic | 1 | 990,000,000 | 2200 | 2018-01-01 | 2024-01-09 | synthetic |
| C02 | synthetic | 2 | 980,000,000 | 2200 | 2018-01-01 | 2024-01-09 | synthetic |
| C03 | synthetic | 3 | 970,000,000 | 2200 | 2018-01-01 | 2024-01-09 | synthetic |
| C04 | synthetic | 4 | 960,000,000 | 2200 | 2018-01-01 | 2024-01-09 | synthetic |
| C05 | synthetic | 5 | 950,000,000 | 2200 | 2018-01-01 | 2024-01-09 | synthetic |
| C06 | synthetic | 6 | 940,000,000 | 1767 | 2019-03-10 | 2024-01-09 | synthetic |
| C07 | synthetic | 7 | 930,000,000 | 2046 | 2018-06-04 | 2024-01-09 | synthetic |
| C08 | synthetic | 8 | 920,000,000 | 1808 | 2019-01-28 | 2024-01-09 | synthetic |
| C09 | synthetic | 9 | 910,000,000 | 2112 | 2018-03-30 | 2024-01-09 | synthetic |
| C10 | synthetic | 10 | 900,000,000 | 1879 | 2018-11-18 | 2024-01-09 | synthetic |
| C11 | synthetic | 11 | 890,000,000 | 1284 | 2018-09-12 | 2022-03-18 | synthetic |


**Rejected (2).** Noise removal is not cosmetic here: a stablecoin cannot trend, and a wrapped or liquid-staked token is the same bet you already hold, so keeping either one silently doubles a position and flatters diversification.

| Rejection reason | Count |
| --- | --- |
| peg detected from data | 1 |
| duplicate of BTC | 1 |


<details><summary>Full rejection list</summary>


| Coin | Reason |
| --- | --- |
| USDX | peg detected from data |
| WBTC | duplicate of BTC |


</details>


**Start-date logic.** The plan asks for the first date on which 51% of the shortlisted coins were tradeable. That bar was reached at 83.3% coverage; a further 300-bar warm-up buffer is added on top so that even a 300-day moving average is fully formed before the first possible trade. Data grid: 2018-01-01 → 2024-01-09; trading begins 2019-01-24.


## 5. Capital, cost and execution assumptions

| Rule | Setting |
| --- | --- |
| Cash on day one | $1,000 |
| New cash each month | $1,000 on the first bar of the month |
| Size per entry | $1,000 all-in (fees and slippage come out of it) |
| Taker fee | 10.0 bps per side |
| Slippage | 15.0 bps per side |
| Tax on realised gains | 0% |
| Signal → fill | signal at the close of bar t, filled at the OPEN of bar t+1 |
| Stop fills | at min(open, stop) — gap-downs are not handed back to you |
| Stop vs take-profit in one bar | the stop is assumed to hit first (pessimistic) |
| Liquidity floor | 30-day average dollar volume ≥ $3,000,000 |
| Delisting policy | halted ≥ 3 bars → written off at 0% of the last close |


## 6. Year by year, and how the portfolio grew

| Year | ROI on capital | P&L change | Avg capital at risk | Trades closed | Realised P&L | End equity | Avg idle cash | Idle % | Avg open | Max open | Avg watchlist | Paid in |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2019 | 136.05% | $3,870 | $2,844 | 103 | $3,870 | $15,870 | $5,931 | 61.5% | 2.8500 | 9 | 1.4200 | $11,000 |
| 2020 | -13.56% | $-160 | $1,183 | 52 | $-132 | $27,709 | $21,175 | 95.0% | 1.1800 | 11 | 0.4400 | $12,000 |
| 2021 | -40.46% | $-514 | $1,271 | 78 | $-542 | $39,195 | $32,489 | 96.1% | 1.2700 | 10 | 0.4200 | $12,000 |
| 2022 | 149.17% | $4,627 | $3,101 | 93 | $4,627 | $53,821 | $41,873 | 92.6% | 3.1000 | 10 | 1.6400 | $10,000 |


Median of the calendar-year ROIs (years with at least one closed trade): **61.25%**. Best year 149.17%, worst year -40.46%. 'P&L change' is mark-to-market and deposit-neutral, so an open winner is credited to the year it rose in rather than the year it was finally sold.


**Idle cash.** The average uninvested balance across the whole test was $24,733, i.e. average exposure of 13.66%. This is deliberately excluded from the ROI numbers — the strategy is judged on money it actually risked — but the 'Idle %' column is the single most important capacity statistic in this document. It rises whenever deposits arrive faster than signals do, and it is the reason the account-level XIRR is so much lower than the return on capital at risk. If that column trends upward, the rules are too selective for the deposit schedule and either the tranche size, the layer cap or the entry filters need loosening.


## 7. Drawdown anatomy

Deepest deposit-adjusted drawdown: **9.86%** of account equity, peaking 2020-09-13 and troughing 2021-07-18, 30 days to make it back.

| Peak | Trough | Recovered | Depth ($) | Depth (%) | Total days | Days to recover |
| --- | --- | --- | --- | --- | --- | --- |
| 2020-09-13 | 2021-07-18 | 2021-08-17 | $2,567 | 9.86% | 337 | 30 |
| 2019-04-16 | 2019-04-25 | 2019-05-12 | $445 | 9.45% | 25 | 17 |
| 2019-07-16 | 2019-07-22 | 2019-08-01 | $889 | 9.25% | 15 | 10 |
| 2021-08-17 | 2022-03-30 | 2022-07-28 | $3,349 | 8.97% | 344 | 120 |
| 2019-12-06 | 2020-08-19 | 2020-08-29 | $1,340 | 8.00% | 266 | 10 |
| 2019-03-03 | 2019-03-23 | 2019-04-04 | $243 | 7.82% | 31 | 12 |


The raw account curve shows a smaller fall (9.45%) purely because fresh deposits keep landing in it while positions are losing. The deposit-adjusted figure measures only what the strategy handed back, and is the one to budget against.


## 8. Trade statistics

| Statistic | Value |
| --- | --- |
| Closed tranches | 326 |
| Win rate | 43.25% |
| Profit factor | 1.73 |
| Average return per tranche | 2.40% |
| Median return per tranche | -1.03% |
| Average winner / loser | 13.17% / -5.81% |
| Best / worst tranche | 86.30% / -100.00% |
| Average holding period | 8.5 days |
| Total capital pushed through trades | $326,000 |
| Turnover ROI (P&L ÷ capital deployed) | 2.40% |


Per-tranche return distribution (5 / 25 / 50 / 75 / 95th percentile): -10.53% | -5.24% | -1.03% | 5.51% | 32.65%


**How positions actually ended.** This is the most diagnostic table in the document: if one reason dominates, that one rule *is* the strategy and everything else is decoration.

| Exit reason | Share |
| --- | --- |
| STOP | 67.5% |
| BTC_MACRO_EXIT | 30.4% |
| END_OF_TEST | 1.8% |
| DELISTED | 0.3% |


**Do the add-on layers earn their keep?** If layer 2+ returns less than layer 1, pyramiding is averaging up into weakness and the layer cap should come down.

| Pyramid layer | Trades | Avg return | Total P&L |
| --- | --- | --- | --- |
| 1 | 307 | 2.80% | $8,597 |
| 2 | 19 | -4.08% | $-776 |


## 9. Per-coin breakdown

| Coin | Trades | Deployed | P&L | Win rate | Median trade | Best | Worst | Avg days | Coin ROI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C02 | 26 | $26,000 | $3,084 | 69% | 8.44% | 45.28% | -11.63% | 8.9000 | 11.86% |
| C06 | 27 | $27,000 | $1,980 | 56% | 0.34% | 86.30% | -13.52% | 10.3000 | 7.33% |
| C09 | 21 | $21,000 | $725 | 38% | -1.73% | 50.90% | -9.26% | 8.2000 | 3.45% |
| C05 | 24 | $24,000 | $604 | 46% | -0.75% | 26.32% | -10.68% | 8.5000 | 2.52% |
| C10 | 29 | $29,000 | $653 | 52% | 0.08% | 28.59% | -10.14% | 8.7000 | 2.25% |
| BTC | 31 | $31,000 | $593 | 39% | -2.44% | 39.82% | -13.63% | 9.1000 | 1.91% |
| C07 | 27 | $27,000 | $495 | 59% | 1.04% | 25.45% | -6.81% | 9.5000 | 1.83% |
| C04 | 29 | $29,000 | $404 | 28% | -2.21% | 42.91% | -9.75% | 7.4000 | 1.39% |
| C03 | 33 | $33,000 | $412 | 39% | -1.28% | 36.94% | -16.89% | 5.5000 | 1.25% |
| C08 | 27 | $27,000 | $-35 | 33% | -6.28% | 42.42% | -12.34% | 8.7000 | -0.13% |
| C11 | 22 | $22,000 | $-425 | 36% | -1.00% | 39.30% | -100.00% | 10.2000 | -1.93% |
| C01 | 30 | $30,000 | $-669 | 27% | -2.97% | 13.30% | -10.49% | 7.6000 | -2.23% |


**75%** of traded coins finished positive; the median coin returned 1.87% on the capital it was given.


Concentration check: the three best coins (C02, C06, C09) produced **64.7%** of all gross profit. A result that hides entirely inside one or two names is a result that got lucky, not a strategy — this is the first number to look at before believing anything else in this document.


## 10. How the watchlist behaved

Queueing rule: **WL_FCFS**, discarding anything older than 90 days.

| Watchlist metric | Value |
| --- | --- |
| Average shadow positions held | 0.93 |
| Largest the watchlist ever got | 9 |
| Shadow positions that became real trades | 75 |
| Shadow positions that died before funding | 160 |
| Average wait before funding | 5.9 days |
| Avg return — funded from the watchlist | 0.59% |
| Avg return — funded immediately | 2.94% |


The last two rows are what justify the queue. If waiting trades do *worse* than fresh ones, the watchlist is buying drift rather than opportunity and `WL_NONE` would have been the better answer; the fact that the optimiser kept the queue means the ranking rule was adding something, but check the size of the gap before trusting it.


## 11. Out-of-sample check

| Out-of-sample | Value |
| --- | --- |
| Window | 2022-10-14 → 2024-01-09 |
| Annualised ROI on capital | 28.61% |
| Median year ROI | -1.17% |
| Median coin ROI | 1.23% |
| Max drawdown | 53.06% |
| Profit factor | 1.15 |
| Win rate | 34.15% |
| Trades | 82 |


Decay from the training window to the untouched one: **-75.10%** (103.70% → 28.61% per year). Some decay is normal and expected. A collapse to zero or negative means the configuration memorised the training window, and the composite score already docks it for that — the final score blends in-sample with the *worse* of the two.


## 12. Robustness, and how much of this is luck

**Block bootstrap** — 2,000 resamples of the daily P&L-on-capital series in 21-day blocks, which preserves short-horizon autocorrelation while destroying the specific order history happened to deal:

| Bootstrap statistic | Value |
| --- | --- |
| Annualised ROI — 5th percentile | 4.74% |
| Annualised ROI — median | 96.39% |
| Annualised ROI — 95th percentile | 198.68% |
| Probability of a positive result | 95.7% |
| Max drawdown — median | $2,496 |
| Max drawdown — 95th percentile (plan for this) | $4,693 |


Read the 95th-percentile drawdown as the one to budget for. The realised drawdown is a single draw from this distribution, not a ceiling.


**Search context.** 8 completed trials, 17 pruned. Best 0.32979, median -2.34902, top-decile threshold 0.30124. The gap between the winner and the top decile is a rough proxy for selection luck: if the winner towers over a dense cluster of near-identical scores, it is probably an outlier draw rather than a better idea.


**What the score was actually sensitive to** (fANOVA importance):

| Parameter | Importance |
| --- | --- |
| pyramid_require_profit | 0.233 |
| adx_thresh | 0.216 |
| use_global_tp | 0.146 |
| btc_ma_len | 0.089 |
| wl_mode | 0.070 |
| btc_ma_type | 0.063 |
| entry_type | 0.046 |
| exit_type | 0.039 |
| max_pyramid_layers | 0.031 |
| use_btc_entry_gate | 0.028 |
| pyramid_min_gap | 0.023 |
| use_btc_exit_override | 0.014 |


Parameters at the top are the ones worth re-validating by hand — nudge them ±20 % and re-run `validate`. Parameters near zero can be perturbed freely, which is good news: it means the winner is not balanced on a knife edge in those dimensions.


## 13. Known biases — read this before risking money


1. **Survivorship.** The candidate list is ranked by *today's* market cap, so coins that died are
   structurally under-represented. 10 known-dead tickers are force-added via
   `INCLUDE_DELISTED`, and halted coins are written off at 0% of last close.
   That is a patch, not a cure. The real fix is a point-in-time universe snapshot per rebalance
   date, which no free API provides cleanly.
2. **Look-ahead in universe selection.** Nobody knew in 2018 which of these names would matter.
   The $3 M liquidity floor mitigates it — a coin cannot be traded before it was liquid — but it
   does not remove it.
3. **Multiple testing.** This report describes the best of many configurations. With enough trials
   something always looks good in-sample; that is why section 11 and section 12 exist, and why the
   composite score is dragged toward the holdout result rather than averaged with it.
4. **Single quote asset, single venue.** All pairs are USDT on one exchange family.
   Fills for a $1,000 clip are realistic; the same rules at $100,000 a clip are not.
5. **Long-only spot.** No shorting, no leverage, no funding costs, no borrow.
6. **Daily bars.** The intrabar path is unknown, so stop-versus-target ordering inside one bar is
   resolved pessimistically. Real fills will differ, usually for the worse on gap days.
7. **Fees and slippage are assumptions**, not measurements: 10 bps +
   15 bps per side. Re-run with `SLIPPAGE_BPS` doubled. If the edge disappears,
   it was never an edge — it was a rebate on an unrealistic fill.
8. **Deposits interact with results.** A strategy that trades rarely looks better per dollar at
   risk and worse per dollar deposited. Both numbers are in section 1 for exactly that reason.


## 14. How to verify every claim here


| File | What it lets you check |
| --- | --- |
| `winner_trades.csv` | Every closed tranche: coin, both dates, both prices, cost, proceeds, P&L, layer, exit reason |
| `winner_equity.csv` | Daily cash, invested value, cost basis, cumulative deposits, cumulative trading P&L, open positions, watchlist depth |
| `winner_yearly.csv` | The year-by-year table above |
| `winner_by_coin.csv` | The per-coin table above |
| `winner_shadow_expired.csv` | Shadow positions that died before funding — the trades you did *not* take |
| `universe_selection.json` | Every coin considered and the exact reason it was kept or dropped |
| `winner_metrics.json` | Every metric in this document, machine-readable |
| `winner.json` / `current_winner.json` | The parameter set, its score, and a fingerprint of the data and settings it was scored on |

Run `python run.py validate` at any time to re-score the stored winner against freshly downloaded
data. If the score moves materially, the edge was specific to one data snapshot rather than
structural.
