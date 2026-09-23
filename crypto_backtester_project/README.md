# Crypto swing-trading backtester + Optuna strategy search

A full implementation of the plan: fetch a clean coin universe, let Optuna choose the entry
architecture, the optional filters, the pyramiding rules, the watchlist ranking rule and the exit
architecture, simulate a $1,000-per-tranche portfolio funded $1,000 a month, score it, and write a
dossier explaining the winner.

```bash
pip install -r requirements.txt

python run.py selftest            # 60 s, no network: proves the engine works
python run.py universe            # build and inspect the coin shortlist
python run.py optimize --trials 500
python run.py validate            # re-score the stored winner on fresh data
python run.py report              # rebuild WINNER_REPORT.md
```

Everything you are likely to change is at the top of `crypto_backtester/config.py` — starting with
`N_COINS`.

---

## What each file does

| File | Role |
| --- | --- |
| `config.py` | The control panel. Universe size, capital rules, costs, scoring weights, paths. |
| `universe.py` | Multi-exchange OHLCV download + cache, and the coin filter pipeline. |
| `indicators.py` | SMA/EMA/DEMA/WMA/RMA, RSI, ATR, ADX, rolling max — with a cross-trial cache. |
| `signals.py` | Turns parameters into entry / filter / exit boolean arrays. Enforces the 1-bar lag. |
| `engine.py` | The portfolio simulator: cash, tranches, pyramiding, watchlist, stops, delistings. |
| `metrics.py` | Performance measurement and the composite score. |
| `search.py` | The Optuna search space, the objective, and winner persistence. |
| `report.py` | The in-depth winner document + CSV side-cars. |
| `selftest.py` | Synthetic-data end-to-end check, including a hard look-ahead audit. |

Outputs land in `output/`: `winner.json`, `current_winner.json`, `WINNER_REPORT.md`,
`winner_trades.csv`, `winner_equity.csv`, `winner_yearly.csv`, `winner_by_coin.csv`,
`winner_shadow_expired.csv`, `universe_selection.json`, plus the Optuna SQLite study so runs resume.

---

## The bar protocol (read this before changing the engine)

For master bar `t`, strictly in this order:

0. credit the monthly contribution if `t` is the first bar of a month
1. force-liquidate anything delisted or halted
2. BTC macro override + signal exits, filled at **OPEN[t]** (flags raised at the **CLOSE of t-1**)
3. entries, filled at **OPEN[t]**, using cash freed in step 2
4. intrabar stop / take-profit pass using HIGH[t]/LOW[t] against a stop frozen at the close of t-1
5. mark to market at CLOSE[t], then ratchet peaks and trailing stops for t+1

Consequences: no signal ever reads the bar it trades on; a stop can fire on the same bar an entry
fills; and when a stop and a target both sit inside one bar the **stop wins**, because the intrabar
path is unknown and the pessimistic branch is the only honest one.

`python run.py selftest` audits this directly — it re-derives every entry from the signal arrays and
fails if any fill was not the next bar's open at the modelled price.

---

## Things done differently from the literal plan, and why

Each of these is a genuine judgement call. All are switchable.

**1. ROI is return on capital at risk, not a chain-linked sleeve return.**
The plan says "do not count uninvested wallet funds, calculate ROI purely on the invested amount".
The tempting implementation — chain-link the daily return of the invested sleeve — is badly broken
here. When a single $1,000 position is open and gains 50 %, that day's sleeve return is +50 %; chain
a few hundred of those and you get a four-digit "ROI" next to $13,000 of real profit. Worse, an
optimiser *chases* it: the cheapest path to a huge chain-linked number is to hold almost nothing,
almost always. So the headline is `total P&L ÷ average dollars actually invested`, and because every
tranche is a fixed $1,000 the strategy does not compound at the position level — capacity grows
linearly with deposits — so the annualised figure is the simple rate, not a CAGR. A CAGR here would
be a fiction dressed as precision.

**2. Drawdown is measured on the cumulative trading-P&L curve.**
Raw account drawdown is meaningless when $1,000 lands in the account every month: fresh deposits
paper over losing stretches. The scored figure is `max giveback of cumulative P&L ÷ account equity at
the peak`. Both numbers are in the report so you can see the gap.

**3. A walk-forward holdout, on by default.**
The last 25 % of the timeline is never optimised on. The final score blends in-sample with the
*worse* of the two windows rather than averaging them, so a configuration cannot win on a training
score alone. The sanity guards (minimum trades, coins, years) are scaled to window length — without
that the short holdout is always rejected as "too few trades" and the check silently stops meaning
anything. Disable with `--no-wf`.

**4. Data-driven noise filters on top of the deny-lists.**
Curated lists go stale. Two detectors catch what they miss: a **peg detector** (daily return stdev
under 0.5 % → it is a stablecoin, whatever it is called) and a **duplicate detector** (return
correlation > 0.99 *and* a near-constant price ratio against an already-accepted coin → WBTC vs BTC,
wstETH vs ETH, cbBTC, jitoSOL, and every wrapper invented after this file was written).

**5. Exchange quote volume for the liquidity floor.**
Where the venue reports it, the $3 M floor uses true quote volume rather than `close × volume`,
which ignores the intrabar price path.

**6. Four optional parameters the plan does not mention.** All are `[True, False]` or have an
"off" value, so Optuna can decline them:
- `entry_require_cross` — `Close > MA` is a *state*, true every bar, which fires an entry daily
  until the layer cap. The cross version fires once. These are different strategies; let the search
  decide.
- `pyramid_min_gap` / `pyramid_require_profit` — cadence and confirmation for add-on layers, so
  pyramiding cannot mean "four layers in four days".
- `wl_max_age` — a 90-day-old breakout signal is not the same trade any more. `0` keeps the literal
  reading (never expire).
- `use_safety_stop` — exit types 3, 4 and 5 are signal-only and have no floor; a coin can fall 85 %
  while you wait for a cross. This adds an optional catastrophic backstop.

**7. `INCLUDE_DELISTED`.** Ranking by *today's* market cap is survivorship bias with a nice UI.
A list of known-dead tickers (LUNA, FTT, UST, WAVES, BSV…) is force-added so the backtest has to
survive them. It is a patch, not a cure — the report says so explicitly.

**8. Mirror-config collapse.** If the sampled fast RSI is slower than the slow RSI the pair is
swapped, so the search does not waste half its budget re-discovering the same configurations
backwards. Set `ENFORCE_RSI_ORDER = False` in `search.py` for the literal spec.

---

## The score

```
score = Σ wᵢ · tanh(metricᵢ / scaleᵢ)  ×  drawdown_penalty  ×  confidence_ramp
```

- terms: annualised ROI on capital, median calendar-year ROI, median per-coin ROI, profit factor
- `drawdown_penalty = 1 / (1 + (dd / tolerance)^power)`; above `dd_hard_cap` the config is invalid
- `confidence_ramp = min(1, trades / target)^0.25` — stops a 4-trade fluke from winning
- hard guards: minimum trades, minimum coins traded, minimum years with trades

Squashing is the point. Without `tanh`, one 900 % outlier year dominates everything and TPE spends
the whole budget hunting lottery tickets. With it, marginal return is worth progressively less and
the search is pushed toward configurations that are decent on *all four* axes — which is what
"consistent" actually means. All weights and scales live in `config.SCORE`.

---

## Winner files (plan steps 13 and 14)

- Every time a trial beats the running best, `current_winner.json` is overwritten immediately.
- On every start, **both** `winner.json` and `current_winner.json` are loaded, re-scored against the
  freshly built universe, printed, enqueued as the first Optuna trials, and the better of the two
  becomes the baseline to beat.
- At the end of a run, `winner.json` is replaced only if the run's champion beats the re-scored
  stored champion. `winner.json` always takes priority on the next load.
- Each file carries a **fingerprint** of the universe, date range, cost model and scoring weights.
  If it does not match, the loader says so — a stored score earned under different settings is not
  comparable, and the code re-scores rather than trusting the stored number.

---

## Performance

Roughly 0.2 s per trial on 12 coins × 1,800 bars; expect 1–2 s on 40 coins × 3,000 bars, so 500
trials is ten to fifteen minutes. The two things that make that possible are the indicator cache
(when 400 trials all ask for `EMA(BTC, 200)` it is computed once) and pre-flattening the signal
matrices into per-bar candidate lists instead of scanning every coin on every bar.

`USE_PRUNER = True` reports an intermediate result at each year boundary so hopeless configurations
die early. `N_JOBS > 1` works with the SQLite storage for small values, but the indicator cache is
per-process, so the speed-up is sub-linear.

---

## Before you trade any of this

Section 13 of the generated report is the honest list. The short version: the universe has
survivorship bias, the winner is the best of hundreds of configurations so some of its edge is
selection luck, daily bars hide the intrabar path, and the fee and slippage figures are assumptions
rather than measurements. Re-run with `SLIPPAGE_BPS` doubled. If the edge vanishes, it was never an
edge.
