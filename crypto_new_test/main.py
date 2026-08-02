"""
CRYPTO DIAMOND EXTRACTOR -- BACKTESTING ENGINE v1.0
====================================================
Top-100-by-market-cap crypto trend-following strategy: unified USDT cash
pool funded by a fixed monthly SIP, single-MA breakout entries gated by a
persistent BTC market-regime flag, four selectable exit families plus an
always-on BTC-regime/trend-reversal override, pyramiding into existing
winners, Bayesian (Optuna/TPE) hyperparameter search over IS data,
walk-forward OOS validation, and the same DSR + score-distribution
overfitting diagnostics as the NSE stock engine (optimized_v4.2) this was
built alongside. Money-weighted (IRR-style) annualized return, Wilder
ATR/ADX, NaN-safe indicators for newly-listed coins, a Binance spot cost
model, and two NEW mechanisms the stock engine doesn't need: a
recency-weighted per-calendar-year consistency score, and a hard
profit-concentration gate -- both aimed squarely at "don't reward a
strategy that got lucky in one early bull year and coasted on the
compounding."

DESIGN DECISIONS MADE UNDER "FULL DISCRETION" (flagged so they're easy to
revisit, not buried):

  1. BENCHMARK = BTC, reused for THREE jobs at once: (a) the market-regime
     filter gating every entry, (b) the DCA benchmark that ROI/alpha/IR are
     measured against, (c) the year classifier for the bull/bear regime
     score. This is deliberate, not laziness -- BTC dominance means "is
     BTC bullish" and "is the crypto market bullish" are the same question
     for backtest purposes, and reusing one series keeps entry filter,
     benchmark, and regime-scoring internally consistent instead of
     importing a second, harder-to-source "total market cap" index that
     would disagree with the filter about what "bullish" means.
  2. ANNUALIZATION BASIS = 365, not 252. Crypto trades every calendar day;
     carrying over the NSE engine's 252-day convention would silently
     under-annualize every Sharpe/Sortino/IR/IRR figure by ~1.45x. This is
     the single most important numeric adaptation in this file.
  3. RSI is gone entirely. Entry is now a pure MA-breakout: coin's close
     crosses above its OWN entry-MA (type + period optimized, 0=SMA,
     1=EMA, 2=DEMA, 3=WMA, 4=SMMA/RMA) while BTC's close is CURRENTLY
     above BTC's OWN filter-MA (independently optimized type + period) --
     BTC does not need to cross at the same moment, it's a standing flag,
     per your explicit correction.
  4. EXIT is a categorical choice per trial (Optuna picks ONE of 4
     families for the whole run, mirroring how the stock engine's
     n_exit_method works) PLUS an always-on override that can fire
     regardless of which family is active: BTC turns bearish OR price
     crosses under a SEPARATELY-optimized exit-MA, whichever happens
     first. The 4 selectable families:
        0 = original hybrid: partial-close 50% at an ATR profit target,
            move stop to breakeven, ATR-trail the remainder (adapted from
            your Pine script -- CLOSE-based, not HIGH-based, see #5).
        1 = %-trail from the post-entry high (stock engine's n_exit_m=1).
        2 = ATR-trail from the post-entry high (stock engine's n_exit_m=2).
        3 = crosses back under its own ENTRY-ma (the mirror-image of the
            entry signal).
  5. CLOSE-based signals throughout, not HIGH-based. Your Pine script's TP/
     trail checks use the bar's `high`, which is fine intrabar on
     TradingView but would give this daily-bar Python backtest a lookahead
     advantage it can't actually fill at. Every signal here fires off
     `close`, and every fill happens at the NEXT bar's `open` -- same
     two-phase signal/fill discipline the stock engine already uses.
  6. WIN-RATE HARD GATE lowered to 35% (from the stock engine's 50%).
     A trend system with a 65:5.6 reward:risk ratio (your Pine defaults)
     is SUPPOSED to lose often and win big -- gating at 50% would reject
     exactly the strategies this design is meant to find. Flagging this
     explicitly since it's a real deviation from the file you gave me.
  7. RECENCY WEIGHTING: linear ramp, weight(year) = 0.7 + 0.3 *
     (year - MIN_YEAR)/(MAX_YEAR - MIN_YEAR), i.e. the oldest year in the
     dataset counts at 0.7x and the most recent counts at 1.0x, scaled
     further by that year's data-coverage fraction (a partial first/last
     year contributes proportionally less). Applied ONLY to the new
     consistency/regime score terms below -- core Sharpe/Sortino/Calmar/IRR
     math is untouched, so you can always see the "raw" numbers too.
  8. NEW SCORE = same backbone as the stock engine (Calmar, Sortino,
     Information Ratio, EV-in-R, win-rate bonus, stat-confidence,
     drawdown-penalty) with weights rebalanced to make room for two new,
     additive terms:
       - CONSISTENCY term: recency-weighted mean per-calendar-year IRR,
         penalized by its recency-weighted std-dev and by its single worst
         year -- expressed in Calmar-like units (divided by overall max_dd)
         so it's on the same scale as the Calmar term it sits next to.
       - REGIME term: split years into bull/bear by BTC's OWN annual
         return (>+10% / <-10%), then score (a) how much of BTC's bull-year
         upside the strategy captured, and (b) how much BETTER than BTC
         the strategy did in bear years (defense, not just "didn't lose").
     Full weight table and formulas are in compute_score_crypto()'s
     docstring below.
  9. HARD GATE (NEW): profit concentration. If any single calendar year
     accounts for more than CONCENTRATION_GATE (55%) of total nominal
     profit, the trial is rejected outright (score = -999), independent of
     everything else. This is the direct, blunt-instrument answer to
     "don't let a strategy win purely because it happened to catch one
     early monster year." The recency weighting softens the SCORE; this
     gate refuses to deploy the extreme cases at all.

KEY ASSUMPTIONS & LIMITATIONS (same spirit as the stock engine's list)
  - Survivorship bias, by your explicit choice: universe = TODAY's top 100
    by market cap (ex-stablecoins/wrapped/leveraged tokens), applied back
    to START_DATE. A coin not in today's top 100 is invisible even if it
    was huge in 2018 (e.g. a coin that has since collapsed). Disclosed,
    not fixed -- same posture as the stock engine's LIMITATION-1.
  - A coin with fewer than MIN_HISTORY_DAYS of Binance USDT history is
    dropped entirely (separate, familiar exclusion -- same as the stock
    engine's <500-day rule).
  - Floor-clamp diagnostic (BUG-16 in the stock engine) is kept, but
    worded more cautiously here: unlike NSE circuit limits, a >99.9%
    one-day move is NOT structurally impossible in crypto (thin-book
    flash crashes, depegs). If it fires, treat it as "investigate", not
    an automatic "this is definitely a bug" the way the stock engine can.
  - One 70/30 IS/OOS split remains a single, high-variance robustness
    estimate, mitigated the same way: sub-period breakdown + DSR +
    score-distribution check, all reused unmodified from the stock engine
    (their math doesn't care what asset class fed them numbers).
  - Idle USDT sits in cash_pool earning ANNUAL_CASH_YIELD (default 0% --
    real USDT lending/staking yields exist but carry counterparty/
    regulatory risk this backtest isn't trying to model; raise it yourself
    if you want to credit e.g. a specific yield source).

SCORING (v1.1, see full breakdown in compute_score_crypto())
  score = [Calmar*0.20 + Sortino(cap 4.0)*0.10 + IR*0.20 + EV_in_R*0.10
           + WinRate_bonus*0.10 + Consistency*0.20 + Regime*0.10]
          * stat_confidence * drawdown_penalty
  Hard gates (all must pass or the trial scores -999): trades/months
  minimums, max_dd <= 50%, profit factor >= 1.10, ROI > 0, win rate >=
  MIN_WIN_RATE_GATE (35%, see decision #6), AND the profit-concentration
  gate (decision #9). Deployable-save gate is the same OOS-robustness
  >= 50% bar the stock engine uses.

  DECISION #10 (post-mortem, v1 -> v1.1): v1 additionally hard-gated on
  `roi > bench_roi * 1.20` -- beat BTC's own buy-and-hold-plus-DCA return
  by 20%, ported directly from the stock engine's "beat Nifty by 20%"
  gate. On real data this rejected literally every trial in a 20,000-trial
  IS run (2018-2024ish), because that window contains BTC's 2020-2021
  run -- DCA-ing straight into BTC through the 2018-2020 lows and riding
  that rally puts bench_roi somewhere in the hundreds-of-percent range,
  a bar that a diversified, partly-in-cash, trend-following system
  structurally can't clear in absolute dollar terms no matter how good
  its risk-adjusted behavior is. NSE's index doesn't compound like that,
  so the ported gate was invisible there; it's fatal here. REMOVED in
  v1.1 -- the IR term (already in the score) and the regime term's
  bull_capture ratio (already in the score) reward genuine outperformance
  continuously instead of demanding it as an all-or-nothing cliff, and
  ROI > 0 remains as the absolute-profitability floor.

  DECISIONS #11-14 (v1.1 -> v1.2, post-mortem on a run that printed a
  suspicious flat "Robustness Ratio: 0.0%" on an OOS leg that otherwise
  looked fine -- Sharpe 1.43, ROI 18%, MaxDD 10.3%):

  11. PARTIAL-YEAR SCORING GATE (NEW). compute_yearly_breakdown() IRR-
      annualizes whatever slice of a calendar year falls inside
      [start_day, end_day). For a partial year (e.g. the current
      in-progress year, or a WFO window that starts/ends mid-year) this
      can blow up to triple-digit noise -- a 6-month window that's up
      40% annualizes to ~90%, and a real run here showed a partial year
      reading +285% IRR off a coverage_frac of ~0.52. That's not signal,
      it's short-window IRR extrapolation. MIN_YEAR_COVERAGE_FOR_SCORING
      (0.75) now excludes any year below that coverage threshold from
      the CONSISTENCY and REGIME terms specifically -- the profit-
      concentration gate is untouched since it uses raw nominal dollars,
      not an annualized rate, and isn't subject to this distortion. This
      was very likely the dominant cause of the false-looking OOS
      collapse: the OOS window is short enough (~2.6 yrs) that one noisy
      partial year could swing the whole consistency term.
  12. ROBUSTNESS RATIO IS NO LONGER A HARD CLIFF AT OOS<=0. The old
      formula (`oos_score/is_score` if both positive, else flat 0.0)
      made a barely-negative OOS score and a catastrophically negative
      one print identically as "0.0%". It's now continuous
      (oos_score/abs(is_score) when OOS is negative), which changes
      NOTHING about the deploy decision (still requires both positive
      and >= ROBUSTNESS_DEPLOY_THRESHOLD to save) but stops hiding how
      close/far a rejected run actually was.
  13. CASH-FLOW-ADJUSTED DAILY RETURNS FOR SHARPE/SORTINO (BUG FIX).
      The daily return series feeding Sharpe/Sortino previously used raw
      daily_port_val deltas, which include that month's MONTHLY_SIP
      landing in cash_pool -- i.e. every SIP day showed an inflated
      "return" that's actually just new money arriving, worst in the
      earliest months when the deposit is a large fraction of portfolio
      value. Sharpe/Sortino now subtract that day's known inflow before
      computing the return. max_dd deliberately still uses the raw,
      contribution-inclusive series -- "did my balance ever fall X% from
      its all-time high, deposits and all" is the real lived experience
      of a DCA account and is left as-is. The monthly-return series
      feeding Information Ratio is now built by chain-linking these same
      flow-adjusted daily returns within each calendar month, instead of
      the old "shift the baseline by MONTHLY_SIP" approximation -- this
      is also what makes decision #14 possible.
  14. RANDOMIZED SIP-DATE ROBUSTNESS CHECK (NEW). The SIP used to always
      land on the first trading day of the month (`months[d]!=months[d-1]`
      doubled as both the injection trigger AND the monthly-return
      bucket boundary). Cash injection is now driven by an externally-
      built `sip_flag` array instead (build_sip_schedule()), decoupled
      from the monthly-bucket bookkeeping. This enables
      passes_temporal_robustness_check(): re-run a candidate champion
      TEMPORAL_ROBUSTNESS_RUNS times with the SIP landing on a different
      random day each time (same dollar amount, same benchmark timing,
      only the calendar day changes) and require most runs to hold onto
      most of the base score. A strategy whose edge secretly depends on
      always deploying capital at month-start conditions will show wide
      score variance here; a genuine trend-following edge should not
      care what day of the month the money shows up. This only runs when
      a new IS champion is found (and once more on the final champion),
      not on all 20,000 Optuna trials -- that would multiply the search
      budget by TEMPORAL_ROBUSTNESS_RUNS for no benefit, since the point
      is validating a candidate, not searching over SIP dates.
  15. UNIVERSE SHRUNK 100 -> TOP_N_COINS (default 40, tune 30-50).
      Thinner, deeper-ranked coins (60-100) contribute more noise/slippage
      risk than genuine opportunity; a smaller, more liquid universe
      trades off a few breakout opportunities for candidate quality. If
      trade counts fall too close to MIN_TRADES_GATE after shrinking,
      that's a real finding (smaller universe, fewer genuine setups), not
      a bug -- reconsider the gate deliberately rather than reflexively.
  16. SIP WITHHOLDING REMOVED; FIXED-CHUNK SIZING REPLACED WITH AN
      EQUAL-WEIGHT-TARGET SIZE (v1.5 -> v2.0, post-mortem on request).
      v1.5 injected MONTHLY_SIP into cash_pool only in months the pool
      was already below one full ticket, and deployed in fixed
      MONTHLY_SIP-sized chunks -- i.e. the SIP itself was a rationing
      mechanism for position size. Both changes are reverted/replaced:
        - SIP INJECTION is unconditional again: cash_pool += MONTHLY_SIP
          every sip_flag day, full stop, matching the benchmark's own
          unconditional inflow (removes the asymmetry decision #13's
          bench_sip_injected fix had to work around).
        - POSITION SIZE is now (cash_pool + invested_cost) / n_eligible,
          recomputed at each deployment step within the day -- an
          equal-weight target across however many coins are actually
          tradeable *as of today* (eligible_mask[d,:], NOT the full
          survivorship-biased end-state universe -- using the latter
          would leak forward information about coins that haven't
          listed yet). invested_cost is COST BASIS (shares_held *
          entry_prices), deliberately NOT mark-to-market value: sizing
          the next, unrelated trade off another position's *unrealized*
          paper gain would let one lucky early winner inflate every
          subsequent ticket before that gain is ever realized or given
          a chance to reverse -- the exact "one early monster year"
          distortion decision #9's profit-concentration gate exists to
          catch, just reintroduced through the sizing math instead of
          the score. Realized profit (a closed trade's proceeds landing
          back in cash_pool) DOES flow into the next target size, which
          is the legitimate form of compounding: money the strategy has
          actually banked, not money it's marking itself up. If the
          wallet holds less than the target, it invests everything it
          has left rather than skipping the trade, and a MIN_TICKET_SIZE
          floor stops the loop from chasing dust once cash_pool is
          nearly drained -- both aimed at the same goal you flagged:
          never let one ticket empty the wallet so completely that the
          next genuine signal has nothing to deploy.
"""

# ==========================================
# 0. PINE SCRIPT REFERENCE STRATEGY TOGGLE
# ==========================================
# True  -> on every run, evaluate your live Pine Script params first (full-
#          data report + its own IS/OOS split), let it seed the search as
#          the starting baseline if it's stronger than any loaded champion,
#          and enqueue it as an explicit Optuna trial so TPE explores its
#          neighborhood.
# False -> skip ALL of the above entirely and start a completely fresh
#          search (no Pine report, no baseline seeding, no enqueued trial).
RUN_PINESCRIPT_REFERENCE = True

import math
import json
import os
import time
import threading
import warnings
from collections import OrderedDict
from statistics import NormalDist
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from numba import njit
from tqdm import tqdm

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    print("optuna not found. Install with: pip install optuna")
    OPTUNA_AVAILABLE = False

warnings.filterwarnings('ignore')

# ==========================================
# 1. UNIVERSE DEFINITION
# ==========================================

# Disclosed, static exclusion lists (same pattern as the stock engine's
# DIVIDEND_KINGS_FALLBACK) -- these keep stablecoins, wrapped/pegged
# tokens, and leveraged/synthetic tokens out of the tradeable watchlist so
# they don't pollute it with assets that structurally can't "trend".
# Not exhaustive by design -- extend if CoinGecko's top 100 surfaces one
# this list missed; the fetch step below also prints whatever it excluded
# so a miss is visible, not silent.
STABLECOIN_SYMBOLS = {
    'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'FDUSD', 'USDD', 'USDP', 'GUSD',
    'PYUSD', 'USDE', 'FRAX', 'CRVUSD', 'LUSD', 'SUSD', 'EURT', 'EURS',
    'USTC', 'UST', 'USDS'
}
WRAPPED_SYMBOLS = {
    'WBTC', 'WETH', 'WSTETH', 'WEETH', 'WBETH', 'STETH', 'CBETH', 'RETH',
    'WBNB', 'WAVAX', 'WMATIC'
}
LEVERAGED_OR_SYNTHETIC_PATTERNS = ('UP', 'DOWN', '3L', '3S', 'BULL', 'BEAR')

MIN_HISTORY_DAYS = 400   # coin dropped entirely below this (mirrors stock's 500-day rule,
                          # slightly relaxed since the crypto universe is inherently younger)

# ── NEW v1.4: liquidity / noise-coin filtering ────────────────────────────
# Two layers, deliberately redundant:
#   1. A coarse screen at universe-fetch time using CoinGecko's TODAY's 24h
#      volume (see fetch_top100_universe()) -- cheap, catches the obvious
#      cases before any data is even downloaded.
#   2. A point-in-time rolling filter built from actual Binance USDT quote
#      volume (see build_liquidity_mask()) -- a coin that's liquid TODAY
#      but was thin in 2019 doesn't get treated as always-tradeable, and a
#      coin that's gone quiet recently after being liquid stops qualifying
#      for NEW entries from that point on (existing positions are never
#      force-closed by this, same non-destructive posture as
#      build_eligibility_mask()).
MIN_AVG_DAILY_VOLUME_USD = 3_000_000   # trailing-average USDT volume floor for a NEW entry
LIQUIDITY_LOOKBACK_DAYS  = 30           # trailing window the average is computed over

BINANCE_KLINES_URL   = "https://api.binance.com/api/v3/klines"
BINANCE_EXINFO_URL   = "https://api.binance.com/api/v3/exchangeInfo"
COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"

# Fallback if CoinGecko/Binance are unreachable when this runs -- guarantees
# BTC is present and a few unmistakable majors are too, same defensive
# posture as the stock engine's hardcoded fallback list.
FALLBACK_UNIVERSE = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
    'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'XLMUSDT', 'LTCUSDT', 'TRXUSDT',
    'DOTUSDT', 'MATICUSDT', 'SHIBUSDT', 'ATOMUSDT', 'UNIUSDT', 'ETCUSDT',
    'NEARUSDT', 'FILUSDT'
]


def _is_excluded(symbol_upper):
    if symbol_upper in STABLECOIN_SYMBOLS or symbol_upper in WRAPPED_SYMBOLS:
        return True
    for pat in LEVERAGED_OR_SYNTHETIC_PATTERNS:
        if symbol_upper.endswith(pat):
            return True
    return False


def _get_binance_usdt_symbols():
    """All symbols Binance actually lists as a spot XXXUSDT pair, so we
    never try to download a ticker that doesn't exist there."""
    try:
        r = requests.get(BINANCE_EXINFO_URL, timeout=15)
        r.raise_for_status()
        data = r.json()
        return {s['symbol'] for s in data['symbols']
                if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING'}
    except Exception as e:
        print(f"Could not fetch Binance exchangeInfo ({e}); will validate per-ticker on download instead.")
        return None


def fetch_top100_universe():
    """Fetch today's top-100-by-market-cap coins from CoinGecko, filter out
    stablecoins/wrapped/leveraged tokens, map to Binance USDT pairs, and
    HARD-GUARANTEE BTCUSDT sits at index 0 -- both because it's obviously
    going to rank #1 anyway, and because everything downstream (the regime
    filter, the benchmark, the bull/bear year classifier) assumes stock
    column 0 is BTC. This is enforced with an assertion later in
    prepare_matrix_data_crypto(), not just a comment."""
    print(f"Fetching top-{TOP_N_COINS}-by-market-cap universe from CoinGecko...")
    print("=" * 70)
    print("SURVIVORSHIP-BIAS WARNING (same posture as the stock engine):")
    print(f"  This pulls TODAY's top {TOP_N_COINS} and applies it back to {START_DATE}.")
    print(f"  A coin that mattered in 2018-2020 but isn't top-{TOP_N_COINS} today is")
    print("  invisible to this backtest. Disclosed, not fixed.")
    print("=" * 70)

    binance_usdt = _get_binance_usdt_symbols()

    tickers = []
    excluded_log = []
    try:
        r = requests.get(COINGECKO_MARKETS_URL, params={
            'vs_currency': 'usd', 'order': 'market_cap_desc',
            'per_page': TOP_N_COINS, 'page': 1, 'sparkline': 'false'
        }, timeout=20)
        r.raise_for_status()
        coins = r.json()
        for c in coins:
            sym = str(c.get('symbol', '')).upper()
            if not sym:
                continue
            if _is_excluded(sym):
                excluded_log.append(sym)
                continue
            # NEW v1.4: coarse noise-coin screen using CoinGecko's TODAY's 24h
            # volume -- cheap, catches the obvious cases before any Binance
            # data is even downloaded. The real, point-in-time filter is
            # build_liquidity_mask() further down (uses actual historical
            # Binance volume, not just today's snapshot); this is belt-and-
            # suspenders, not a substitute for it. BTC is exempt on principle
            # (it will always clear this bar anyway).
            vol_24h = c.get('total_volume', 0) or 0
            if sym != 'BTC' and vol_24h < MIN_AVG_DAILY_VOLUME_USD:
                excluded_log.append(f"{sym} (24h volume ${vol_24h:,.0f} below ${MIN_AVG_DAILY_VOLUME_USD:,.0f} floor)")
                continue
            binance_sym = f"{sym}USDT"
            if binance_usdt is not None and binance_sym not in binance_usdt:
                excluded_log.append(f"{sym} (no Binance USDT pair)")
                continue
            tickers.append(binance_sym)
    except Exception as e:
        print(f"CoinGecko fetch failed ({e}); falling back to a static major-coin list.")
        tickers = list(FALLBACK_UNIVERSE)

    # Hard-guarantee BTC first, de-dup, preserve rank order otherwise.
    tickers = [t for t in tickers if t != 'BTCUSDT']
    tickers = ['BTCUSDT'] + list(dict.fromkeys(tickers))

    if len(tickers) < max(10, TOP_N_COINS // 3):
        print("Universe too small after filtering; falling back to static major-coin list.")
        tickers = list(FALLBACK_UNIVERSE)

    print(f"Universe size after filtering: {len(tickers)} (excluded {len(excluded_log)}: "
          f"{', '.join(excluded_log[:15])}{' ...' if len(excluded_log) > 15 else ''})")
    sanity_majors = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XLMUSDT']
    present = [m for m in sanity_majors if m in tickers]
    print(f"Sanity check -- expected majors present: {present} "
          f"({'OK' if len(present) >= 3 else 'WARNING: fewer majors than expected, check filters'})")
    assert tickers[0] == 'BTCUSDT', "BTCUSDT must be at index 0 -- everything downstream depends on this."
    return tickers


# ==========================================
# 2. CONFIGURATIONS & SAVE SYSTEM
# ==========================================

START_DATE        = "2018-01-01"
BEST_PARAMS_FILE  = "best_params_crypto_v1.json"
INTERMEDIATE_FILE = "current_is_champion_crypto_v1.json"
ENGINE_VERSION    = "1.2.0"

WFO_IS_PCT  = 0.70
WFO_OOS_PCT = 0.30

BAYESIAN_TRIALS    = 20_000
NEIGHBOR_THRESHOLD = 0.70
MIN_TRADES_GATE    = 20
MIN_MONTHS_GATE    = 24
MIN_WIN_RATE_GATE  = 0.20   # decision #6 in the module docstring -- deliberately lower
                             # than the stock engine's 0.50: a trend system with a big
                             # reward:risk ratio is SUPPOSED to have a sub-50% win rate.
MONTHLY_SIP        = 2_000   # USDT/month -- your confirmed crypto budget, injected unconditionally
                             # every sip_flag day (decision #16 -- withholding removed)
MIN_TICKET_SIZE    = MONTHLY_SIP   # NEW: raised from 50 -- your explicit floor is "2000, no
                             # less". Below this the deployment loop skips the entry entirely
                             # for today (early on, when cash_pool is genuinely thin, missing
                             # an entry is fine and expected) rather than forcing a dust-sized
                             # position. Tied to MONTHLY_SIP directly instead of a separate
                             # hardcoded number so the two can't drift apart.

ROBUSTNESS_DEPLOY_THRESHOLD  = 0.50
HIGH_CASH_FRACTION_THRESHOLD = 0.30
MA_CACHE_MAX_ENTRIES         = 20_000

# ── NEW (decision #17): watchlist candidates are tracked as shadow
# positions with their own stop-loss/TP/trail state, so a watchlist
# candidate is removed by the SAME exit condition that would sell a real
# position -- see simulate_portfolio_crypto's WATCHLIST INVALIDATION block.
# The earlier grace-period/hard-age-cap mechanism (WL_GRACE_DAYS_DEFAULT /
# WL_MAX_AGE_DAYS_DEFAULT, wl_grace_days/wl_max_age_days) is fully removed,
# not just superseded -- there is no separate age cap anymore.

# ── TRANSACTION COST MODEL (decision from your point 9: "standard + some
# slippage") ────────────────────────────────────────────────────────────
# Binance spot taker fee, NO BNB discount / VIP tier assumed (conservative;
# raise/lower BINANCE_TAKER_PCT if you actually run a discounted tier).
# Slippage is set higher than the stock engine's 0.10% because thinner
# order books further down the top-100 and 24/7 gap risk both cut against
# next-bar-open fills more than an NSE large/mid-cap does.
BINANCE_TAKER_PCT = 0.00100
SLIPPAGE_PCT      = 0.00150
BUY_COST_PCT  = BINANCE_TAKER_PCT + SLIPPAGE_PCT   # 0.25% per side
SELL_COST_PCT = BINANCE_TAKER_PCT + SLIPPAGE_PCT   # symmetric -- no NSE-style
                                                     # asymmetric stamp duty/STT here

# ── IDLE CASH YIELD ────────────────────────────────────────────────────
# Default 0% -- deliberately conservative. Real USDT yield sources exist
# but vary by venue/regulatory status; raise this yourself if you want to
# credit a specific one you actually use.
ANNUAL_CASH_YIELD = 0.00

# Crypto trades every calendar day -- 365, not the stock engine's 252.
# This is decision #2 in the module docstring; it feeds every
# annualization (Sharpe, Sortino, IR, the money-weighted IRR).
CRYPTO_DAYS_PER_YEAR = 365.0

# ── NEW: recency weighting + regime + concentration gate (decisions #7-9) ──
RECENCY_WEIGHT_MIN       = 0.80
RECENCY_WEIGHT_MAX       = 1.00
BULL_YEAR_BTC_THRESHOLD  = 0.10    # BTC annual return above this => "bull year"
BEAR_YEAR_BTC_THRESHOLD  = -0.10   # BTC annual return below this => "bear year"
CONCENTRATION_GATE       = 0.8   # hard reject if one calendar year > 55% of total profit
YEAR_FULL_COVERAGE_DAYS  = 365.0   # denominator for a year's coverage-fraction weight

# ── Composite score weights (decision #8) -- sum to 1.00 ────────────────
W_CALMAR      = 0.20
W_SORTINO     = 0.10
W_IR          = 0.20
W_EV          = 0.10
W_WR_BONUS    = 0.10
W_CONSISTENCY = 0.20
W_REGIME      = 0.10

K_STD_PENALTY   = 1.00   # consistency term: penalty coefficient on recency-weighted std-dev
K_WORST_PENALTY = 0.50   # consistency term: penalty coefficient on the single worst year
W_BULL_CAPTURE  = 0.50   # regime term split
W_BEAR_DEFENSE  = 0.50

# ── NEW v1.2: universe size (decision #15) ──────────────────────────────
TOP_N_COINS = 200    # NEW v1.4: full top-100; liquidity filtering below (coarse
                     # 24h-volume screen at fetch + point-in-time trailing-volume
                     # mask) is what makes running the full 100 safe to search over
                     # instead of just noisier.

# ── NEW v1.2: partial-year scoring gate (decision #11) ──────────────────
MIN_YEAR_COVERAGE_FOR_SCORING = 0.75   # years below this coverage_frac are excluded
                                        # from the CONSISTENCY/REGIME terms (their IRR
                                        # is annualized from too few days to be trusted).
                                        # The profit-concentration hard gate is unaffected
                                        # -- it uses raw nominal dollars, not an annualized
                                        # rate, so it isn't subject to this distortion.

# ── NEW v1.2: SIP-date randomization / temporal robustness (decision #14) ──
SIP_RANDOM_MIN_DAY = 1
SIP_RANDOM_MAX_DAY = 28    # 28 keeps every month (incl. February) valid
TEMPORAL_ROBUSTNESS_RUNS      = 15     # Monte Carlo re-runs with randomized SIP dates
TEMPORAL_ROBUSTNESS_SCORE_MIN = 0.70   # each run must retain >= 70% of the base score
TEMPORAL_ROBUSTNESS_PASS_FRAC = 0.70   # >= 70% of the N runs must clear that bar

# ── NEW v1.3: Optuna parallelism (ported from the stock engine's OPTUNA_N_JOBS,
# which existed there but was never wired into the crypto engine's study.optimize
# call) ──────────────────────────────────────────────────────────────────────
OPTUNA_N_JOBS = max(1, (os.cpu_count() or 2) - 1)   # set to 1 for bit-exact reproducibility

# ── NEW v1.3: your live Pine Script strategy, hardcoded exactly as pasted --
# signal_method=1 (dual smoothed-RSI crossover + trend-SMA filter), exit_method=0
# (the hybrid ATR TP/breakeven/trail block, which is a CLOSE-based port of your
# Pine script's HIGH-based TP/trail -- see module docstring decision #5 for why
# that substitution is deliberate, not an oversight). This is evaluated and
# printed as the baseline 'winner' the moment the engine loads, BEFORE the
# Optuna search runs -- see run_optimization(). exit_ma_len/type and
# entry_ma_len/type are present only so the params dict has every key the
# simulator's schema expects; they're inert for signal_method==1 (the override
# exit uses the RSI crossunder instead, and entry_ma isn't read at all).
PINESCRIPT_REFERENCE_PARAMS = {
    'signal_method':  1,   # FIXED: was 0, which silently ignored the rsi_* params below
                           # and ran the inert entry_ma_len/type path instead. Your live
                           # Pine strategy is the dual smoothed-RSI crossover -- this must be 1.
    'use_trend_ma':   True,    # NEW: Toggle trend_ma filter
    'use_btc_filter': True,    # NEW: Toggle BTC regime filter
    'use_panic_exits': True,   # NEW: Toggle universal panic exits
    'rsi_fast_len':   38,   
    'rsi_fast_smt':   24,   
    'rsi_slow_len':   47,   
    'rsi_slow_smt':   35,   
    'trend_ma_len':   80,   
    'trend_ma_type':  0,    
    'btc_ma_len':     62,   
    'btc_ma_type':    0,    
    'entry_ma_len':   80,   
    'entry_ma_type':  0,
    'exit_ma_len':    80,   
    'exit_ma_type':   0,
    'wl_rank':        0,
    'adx_thresh':     0.0,
    'exit_method':    0,    
    'sl_mult':        5.6,
    'tp_mult':        65.3,
    'trail_mult':     8.9,
    'trail_pct':      15.0,  
    'exit_atr_mult':  3.0,   
}


def load_previous_winner(filename=BEST_PARAMS_FILE):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                return data.get('oos_score', -999999), data.get('is_score', -999999), data.get('params', None)
        except Exception:
            pass
    return -999999, -999999, None


def save_winner(oos_score, is_score, params, filename=BEST_PARAMS_FILE):
    clean_params = {k: float(v) if isinstance(v, (float, np.floating)) else int(v)
                    for k, v in params.items()}
    data = {
        'engine_version': ENGINE_VERSION,
        'oos_score': float(oos_score),
        'is_score':  float(is_score),
        'robustness_ratio': float(oos_score / is_score) if is_score > 0 else 0.0,
        'params': clean_params
    }
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving {filename}: {e}")


# ==========================================
# 3. FAST INDICATORS -- NaN-safe, 5 MA types
# ==========================================
# ma_type: 0=SMA, 1=EMA, 2=DEMA, 3=WMA, 4=SMMA/RMA (Wilder smoothing -- the
# type your Pine indicator's own ATR/ADX-style smoothing already uses,
# added here as a selectable trend-filter type per your request).

@njit(nogil=True, cache=True)
def calc_ma(prices, period, ma_type):
    n = len(prices)
    res = np.empty(n)
    res[:] = np.nan

    start = 0
    while start < n and np.isnan(prices[start]):
        start += 1
    if n - start < period:
        return res

    if ma_type == 0:  # SMA
        w_sum = np.sum(prices[start:start + period])
        res[start + period - 1] = w_sum / period
        for i in range(start + period, n):
            w_sum = w_sum - prices[i - period] + prices[i]
            res[i] = w_sum / period
    elif ma_type == 1 or ma_type == 2:  # EMA / DEMA
        ema1 = np.empty(n)
        ema1[:] = np.nan
        ema1[start + period - 1] = np.mean(prices[start:start + period])
        mult = 2 / (period + 1)
        for i in range(start + period, n):
            ema1[i] = (prices[i] - ema1[i - 1]) * mult + ema1[i - 1]
        if ma_type == 1:
            res = ema1
        else:  # DEMA
            ema2 = np.empty(n)
            ema2[:] = np.nan
            if start + period * 2 - 2 < n:
                ema2[start + period * 2 - 2] = np.mean(ema1[start + period - 1:start + period * 2 - 1])
                for i in range(start + period * 2 - 1, n):
                    ema2[i] = (ema1[i] - ema2[i - 1]) * mult + ema2[i - 1]
                for i in range(start + period * 2 - 2, n):
                    res[i] = 2 * ema1[i] - ema2[i]
    elif ma_type == 3:  # WMA
        weights = np.arange(1, period + 1, dtype=np.float64)
        w_sum = np.sum(weights)
        for i in range(start + period - 1, n):
            res[i] = np.sum(prices[i - period + 1:i + 1] * weights) / w_sum
    elif ma_type == 4:  # SMMA / RMA (Wilder smoothing)
        rma = np.mean(prices[start:start + period])
        res[start + period - 1] = rma
        for i in range(start + period, n):
            rma = (rma * (period - 1) + prices[i]) / period
            res[i] = rma
    return res


@njit(nogil=True, cache=True)
def calc_adx(highs, lows, closes, period=14):
    n = len(closes)
    adx = np.empty(n)
    adx[:] = np.nan
    start = 0
    while start < n and np.isnan(closes[start]):
        start += 1
    if n - start < period * 2:
        return adx

    tr       = np.zeros(n)
    plus_dm  = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(start + 1, n):
        high_diff = highs[i] - highs[i - 1]
        low_diff  = lows[i - 1] - lows[i]
        tr[i]        = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        plus_dm[i]   = high_diff if high_diff > low_diff and high_diff > 0 else 0.0
        minus_dm[i]  = low_diff  if low_diff  > high_diff and low_diff  > 0 else 0.0

    atr_s   = np.sum(tr[start+1:start+period+1])
    plus_s  = np.sum(plus_dm[start+1:start+period+1])
    minus_s = np.sum(minus_dm[start+1:start+period+1])

    for i in range(start + period, n):
        if i > start + period:
            atr_s   = atr_s   - atr_s   / period + tr[i]
            plus_s  = plus_s  - plus_s  / period + plus_dm[i]
            minus_s = minus_s - minus_s / period + minus_dm[i]

        if atr_s > 0:
            plus_di  = 100 * plus_s  / atr_s
            minus_di = 100 * minus_s / atr_s
            denom    = plus_di + minus_di
            dx       = 100 * abs(plus_di - minus_di) / denom if denom > 0 else 0.0
        else:
            dx = 0.0

        if i == start + period: adx[i] = dx
        else:                   adx[i] = (adx[i - 1] * (period - 1) + dx) / period
    return adx


@njit(nogil=True, cache=True)
def calc_atr_wilder(highs, lows, closes, period=14):
    n = len(closes)
    atr = np.empty(n)
    atr[:] = np.nan
    start = 0
    while start < n and np.isnan(closes[start]):
        start += 1
    if n - start < period + 1:
        return atr

    tr = np.zeros(n)
    for i in range(start + 1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))

    atr_s = np.sum(tr[start+1:start+period+1]) / period
    atr[start + period] = atr_s
    for i in range(start + period + 1, n):
        atr_s = (atr_s * (period - 1) + tr[i]) / period
        atr[i] = atr_s
    return atr


@njit(nogil=True, cache=True)
def calc_rsi_wilder(prices, period):
    """Standard Wilder-smoothed RSI (matches Pine's ta.rsi exactly). NaN-safe
    the same way every other indicator in this file is -- scans past a
    leading NaN run (newly-listed coin) before seeding the Wilder average.
    Callers that want the Pine strategy's 'smoothed RSI' (ta.sma(ta.rsi(...),
    smt)) should feed this straight into calc_ma(..., ma_type=0) afterward --
    see get_rsi_smoothed_cached() below, which does exactly that."""
    n = len(prices)
    rsi = np.empty(n)
    rsi[:] = np.nan
    start = 0
    while start < n and np.isnan(prices[start]):
        start += 1
    if n - start < period + 1:
        return rsi

    gains  = np.zeros(n)
    losses = np.zeros(n)
    for i in range(start + 1, n):
        diff = prices[i] - prices[i - 1]
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = -diff

    avg_gain = np.sum(gains[start+1:start+period+1]) / period
    avg_loss = np.sum(losses[start+1:start+period+1]) / period

    idx0 = start + period
    if avg_loss > 0:
        rsi[idx0] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    else:
        rsi[idx0] = 100.0 if avg_gain > 0 else 50.0

    for i in range(idx0 + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss > 0:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        else:
            rsi[i] = 100.0 if avg_gain > 0 else 50.0
    return rsi


# ==========================================
# 3b. BOUNDED, THREAD-SAFE MEMOIZATION FOR calc_ma (perf-only, identical
# pattern and correctness contract as the stock engine's cache)
# ==========================================

_MA_CACHE = OrderedDict()
_MA_CACHE_LOCK = threading.Lock()


def get_ma_cached(closes, stock_idx, period, ma_type):
    key = (stock_idx, int(period), int(ma_type))
    with _MA_CACHE_LOCK:
        cached = _MA_CACHE.get(key)
        if cached is not None:
            _MA_CACHE.move_to_end(key)
            return cached
    result = calc_ma(closes[:, stock_idx], period, ma_type)
    with _MA_CACHE_LOCK:
        _MA_CACHE[key] = result
        while len(_MA_CACHE) > MA_CACHE_MAX_ENTRIES:
            _MA_CACHE.popitem(last=False)
    return result


def clear_ma_cache():
    with _MA_CACHE_LOCK:
        _MA_CACHE.clear()


# ── NEW: cache for the Pine-strategy dual-RSI signal path (decision: keep
# a SEPARATE cache from _MA_CACHE, keyed on (stock_idx, rsi_len, smt_len) --
# a raw-RSI-then-SMA-smoothed series is a different computation from a
# price MA and must not collide in the same keyspace) ──────────────────────
_RSI_CACHE = OrderedDict()
_RSI_CACHE_LOCK = threading.Lock()


def get_rsi_smoothed_cached(closes, stock_idx, rsi_len, smt_len):
    """Matches Pine's calc_smoothed_rsi(src, rsi_l, smt_l) => ta.sma(ta.rsi(src,
    rsi_l), smt_l) exactly: Wilder RSI, then SMA-smoothed on top."""
    key = (stock_idx, int(rsi_len), int(smt_len))
    with _RSI_CACHE_LOCK:
        cached = _RSI_CACHE.get(key)
        if cached is not None:
            _RSI_CACHE.move_to_end(key)
            return cached
    raw_rsi = calc_rsi_wilder(closes[:, stock_idx], rsi_len)
    result  = calc_ma(raw_rsi, smt_len, 0)   # 0 = SMA, matches Pine's ta.sma smoothing
    with _RSI_CACHE_LOCK:
        _RSI_CACHE[key] = result
        while len(_RSI_CACHE) > MA_CACHE_MAX_ENTRIES:
            _RSI_CACHE.popitem(last=False)
    return result


def clear_rsi_cache():
    with _RSI_CACHE_LOCK:
        _RSI_CACHE.clear()


# ==========================================
# 4. PORTFOLIO SIMULATOR (crypto)
# ==========================================
# ENTRY: coin's close crosses above its OWN entry_ma AND BTC's close is
#   CURRENTLY above BTC's own filter-ma (standing flag, not a simultaneous
#   cross -- per your correction) AND (optional) ADX >= adx_threshold.
# EXIT: one selected family (exit_method 0-3) PLUS an always-on override
#   (BTC turns bearish OR price crosses under a separately-optimized
#   exit_ma) that can force a full close regardless of family. See the
#   module docstring decisions #4-5 for the exact per-family logic.
# Everything else (unified cash pool, monthly SIP, rank-and-deploy-to-one-
# best-candidate, pyramiding into winners, two-phase signal-at-close/
# fill-at-next-open, dated cash-flow tracking for the IRR solve, floor-
# clamp + cash-utilization diagnostics) is ported from the stock engine's
# simulate_portfolio essentially unchanged -- that machinery is asset-
# class agnostic.

@njit(nogil=True)
def simulate_portfolio_crypto(
        opens, closes, atr, adx, months, btc_ma, entry_fast, entry_slow, trend_ok, exit_ma,
        eligible_mask,
        wl_rank_method, adx_threshold,
        signal_method,
        exit_method, sl_mult, tp_mult, trail_mult, trail_pct, exit_atr_mult,
        start_day, end_day,
        starting_wealth,
        buy_cost_pct, sell_cost_pct, annual_cash_yield,
        sip_flag,
        use_btc_filter, use_panic_exits): # NEW BOOLEAN TOGGLES

    n_days, n_stocks = closes.shape
    if end_day < 0 or end_day >= n_days: end_day = n_days - 2
    if start_day < 1: start_day = 1

    daily_yield_mult = (1.0 + annual_cash_yield) ** (1.0 / CRYPTO_DAYS_PER_YEAR)
    btc_close = closes[:, 0]

    cash_pool              = starting_wealth
    total_invested_capital = 0.0

    in_pos           = np.zeros(n_stocks, dtype=np.bool_)
    entry_prices     = np.zeros(n_stocks)
    entry_days       = np.zeros(n_stocks, dtype=np.int32)
    shares_held      = np.zeros(n_stocks)
    high_since_entry = np.zeros(n_stocks)

    stop_loss_price  = np.zeros(n_stocks)
    tp_trigger_price = np.zeros(n_stocks)
    half_sold        = np.zeros(n_stocks, dtype=np.bool_)

    wl_active       = np.zeros(n_stocks, dtype=np.bool_)
    wl_entry_prices = np.zeros(n_stocks)
    wl_is_pyramid   = np.zeros(n_stocks, dtype=np.bool_)
    wl_entry_day    = np.zeros(n_stocks, dtype=np.int32)
    # NEW (decision #17): watchlist candidates are now tracked as shadow
    # positions -- same stop-loss/TP/trailing state a real position gets,
    # just with no capital behind it yet. wl_max_age_days no longer kicks a
    # candidate off the watchlist; only the SAME exit condition that would
    # sell a real position does (see WATCHLIST INVALIDATION below).
    wl_stop_loss_price  = np.zeros(n_stocks)
    wl_tp_trigger_price = np.zeros(n_stocks)
    wl_half_sold        = np.zeros(n_stocks, dtype=np.bool_)

    pending_exit    = np.zeros(n_stocks, dtype=np.bool_)
    pending_partial = np.zeros(n_stocks, dtype=np.bool_)

    total_wins     = 0.0
    total_losses   = 0.0
    winning_trades = 0
    losing_trades  = 0
    win_pct_sum    = 0.0
    loss_pct_sum   = 0.0
    total_bars_in_trades = 0

    n_floor_clamps  = 0
    cash_frac_sum   = 0.0
    cash_frac_days  = 0
    days_high_cash  = 0

    daily_port_val  = np.zeros(n_days)
    daily_bench_val = np.zeros(n_days)

    sip_injected    = np.zeros(n_days)
    bench_sip_injected = np.zeros(n_days)   # FIXED: benchmark's OWN inflow flag --
                                            # previously the daily benchmark return
                                            # subtracted the STRATEGY's sip_injected[d],
                                            # which is 0 on any month the strategy
                                            # withheld but BTC still got its $2000,
                                            # letting a fake return spike leak into
                                            # BTC's daily/monthly series (and from there
                                            # into Sharpe/Sortino/IR for the benchmark).
    daily_ret_p_arr = np.zeros(n_days)
    daily_ret_b_arr = np.zeros(n_days)

    MAX_CF     = n_days + 8
    MAX_MONTHS = n_days + 8
    port_monthly_returns  = np.zeros(MAX_MONTHS)
    bench_monthly_returns = np.zeros(MAX_MONTHS)
    month_cnt = 0

    month_compound_p     = 1.0
    month_compound_b     = 1.0
    seen_first_month_end = False
    
    # PREVIOUS FIX: Give benchmark identical starting capital
    bench_shares = 0.0
    if starting_wealth > 0 and btc_close[start_day] > 0:
        bench_shares = starting_wealth / btc_close[start_day]

    cf_days    = np.zeros(MAX_CF, dtype=np.int32)
    cf_amounts = np.zeros(MAX_CF)
    cf_cnt = 0
    if starting_wealth > 0:
        cf_days[0]    = start_day
        cf_amounts[0] = -starting_wealth
        cf_cnt = 1

    # NEW: benchmark gets its OWN cash-flow ledger, separate from the
    # strategy's cf_days/cf_amounts above. As of decision #16 both BTC's
    # SIP and the strategy's SIP land unconditionally every sip_flag day
    # with identical timing/amounts again -- kept as two separate ledgers
    # anyway since bench_shares/shares_held track different instruments
    # and nothing forces their cash-flow arrays to stay physically the
    # same object just because the schedules happen to match.
    bench_cf_days    = np.zeros(MAX_CF, dtype=np.int32)
    bench_cf_amounts = np.zeros(MAX_CF)
    bench_cf_cnt = 0
    if starting_wealth > 0:
        bench_cf_days[0]    = start_day
        bench_cf_amounts[0] = -starting_wealth
        bench_cf_cnt = 1

    for d in range(start_day, end_day):
        cash_pool *= daily_yield_mult

        # ── PHASE-A: SETTLE PENDING EXITS/PARTIALS ────────
        for s in range(n_stocks):
            if pending_partial[s] and in_pos[s]:
                fill_price = opens[d, s]
                if not (fill_price > 0): fill_price = closes[d - 1, s]
                sell_shares  = shares_held[s] * 0.5
                exit_val     = sell_shares * fill_price * (1.0 - sell_cost_pct)
                invested_val = sell_shares * entry_prices[s]
                profit       = exit_val - invested_val
                pct_change   = profit / invested_val if invested_val > 0 else 0.0

                if profit > 0:
                    total_wins     += profit
                    win_pct_sum    += pct_change
                    winning_trades += 1
                else:
                    total_losses   += abs(profit)
                    loss_pct_sum   += abs(pct_change)
                    losing_trades  += 1

                cash_pool          += exit_val
                shares_held[s]     -= sell_shares
                half_sold[s]        = True
                if stop_loss_price[s] < entry_prices[s]:
                    stop_loss_price[s] = entry_prices[s]
                pending_partial[s]   = False

            if pending_exit[s] and in_pos[s]:
                fill_price = opens[d, s]
                if not (fill_price > 0): fill_price = closes[d - 1, s]
                exit_val     = shares_held[s] * fill_price * (1.0 - sell_cost_pct)
                invested_val = shares_held[s] * entry_prices[s]
                profit       = exit_val - invested_val
                pct_change   = profit / invested_val if invested_val > 0 else 0.0

                if profit > 0:
                    total_wins     += profit
                    win_pct_sum    += pct_change
                    winning_trades += 1
                else:
                    total_losses   += abs(profit)
                    loss_pct_sum   += abs(pct_change)
                    losing_trades  += 1

                total_bars_in_trades += (d - 1 - entry_days[s])
                cash_pool        += exit_val
                in_pos[s]         = False
                shares_held[s]    = 0.0
                pending_exit[s]   = False
                half_sold[s]      = False

        # ── MONTHLY RETURN BUCKET CLOSE ────────
        if months[d] != months[d - 1]:
            if seen_first_month_end:
                port_monthly_returns[month_cnt]  = month_compound_p - 1.0
                bench_monthly_returns[month_cnt] = month_compound_b - 1.0
                month_cnt = min(month_cnt + 1, MAX_MONTHS - 1)
            seen_first_month_end = True
            month_compound_p = 1.0
            month_compound_b = 1.0

        # ── SIP CASH INJECTION ────────
        if sip_flag[d]:
            if btc_close[d] > 0:
                bench_shares += MONTHLY_SIP / btc_close[d]
                bench_sip_injected[d] = MONTHLY_SIP
            bench_cf_days[bench_cf_cnt]    = d
            bench_cf_amounts[bench_cf_cnt] = -MONTHLY_SIP
            bench_cf_cnt = min(bench_cf_cnt + 1, MAX_CF - 1)

            # decision #16: unconditional again -- lands every sip_flag day
            # regardless of how much is already sitting in cash_pool,
            # matching the benchmark's own unconditional inflow above.
            cash_pool              += MONTHLY_SIP
            total_invested_capital += MONTHLY_SIP
            sip_injected[d]         = MONTHLY_SIP

            cf_days[cf_cnt]    = d
            cf_amounts[cf_cnt] = -MONTHLY_SIP
            cf_cnt = min(cf_cnt + 1, MAX_CF - 1)

        curr_closes  = closes[d]
        btc_bullish  = btc_close[d] > btc_ma[d]

        # ── WATCHLIST INVALIDATION (decision #17: candidates are tracked as
        # shadow positions now -- the exact same exit rule that would sell a
        # REAL position (stop-loss/TP-trail/ATR-trail/signal-reversal, PLUS
        # the panic override) is what kicks a watchlist candidate too.
        # wl_max_age_days/wl_grace_days and the old grace-period mechanism
        # are fully removed (not just unused) -- a candidate only leaves the
        # watchlist by getting bought or by tripping the same exit condition
        # a funded position would trip. ──
        newly_invalidated = np.zeros(n_stocks, dtype=np.bool_)
        for s in range(n_stocks):
            if wl_active[s] and not wl_is_pyramid[s]:
                # pyramid candidates ride along with the REAL position they'd add
                # to -- that position's own PHASE-B exit (below) is what removes
                # them (via the stale-pyramid prune in CAPITAL DEPLOYMENT), so no
                # separate shadow-exit check runs for them here.
                if curr_closes[s] > 0:
                    high_since_entry[s] = max(high_since_entry[s], curr_closes[s])

                reversal = (entry_fast[d-1, s] >= entry_slow[d-1, s] and entry_fast[d, s] < entry_slow[d, s])

                override = False
                if use_panic_exits:
                    if use_btc_filter and not btc_bullish:
                        override = True
                    if signal_method == 0 and (closes[d-1, s] >= exit_ma[d-1, s] and curr_closes[s] < exit_ma[d, s]):
                        override = True
                    if signal_method == 1 and reversal:
                        override = True

                should_invalidate = False
                should_partial     = False

                if exit_method == 0:
                    if not wl_half_sold[s] and curr_closes[s] >= wl_tp_trigger_price[s]:
                        should_partial = True
                    if wl_half_sold[s]:
                        potential_new_sl = curr_closes[s] - atr[d, s] * trail_mult
                        if potential_new_sl > wl_stop_loss_price[s]:
                            wl_stop_loss_price[s] = potential_new_sl
                    if curr_closes[s] < wl_stop_loss_price[s]:
                        should_invalidate = True
                elif exit_method == 1:
                    if curr_closes[s] < high_since_entry[s] * (1.0 - trail_pct / 100.0):
                        should_invalidate = True
                elif exit_method == 2:
                    if curr_closes[s] < high_since_entry[s] - (exit_atr_mult * atr[d, s]):
                        should_invalidate = True
                elif exit_method == 3:
                    if reversal:
                        should_invalidate = True

                if override:
                    should_invalidate = True
                    should_partial     = False

                if should_partial and not should_invalidate:
                    wl_half_sold[s] = True
                    if wl_stop_loss_price[s] < wl_entry_prices[s]:
                        wl_stop_loss_price[s] = wl_entry_prices[s]   # shadow breakeven, same as a real position's partial exit

                if should_invalidate:
                    wl_active[s]         = False
                    wl_is_pyramid[s]     = False
                    wl_half_sold[s]      = False
                    newly_invalidated[s] = True

        # ── PHASE-B: FLAG EXITS/PARTIALS FOR ACTIVE POSITIONS ────────
        for s in range(n_stocks):
            if in_pos[s] and not pending_exit[s]:
                if curr_closes[s] > 0:
                    high_since_entry[s] = max(high_since_entry[s], curr_closes[s])

                signal_reversal = (entry_fast[d-1, s] >= entry_slow[d-1, s] and entry_fast[d, s] < entry_slow[d, s])
                
                # NEW: Optimizable Universal Panic Exits
                override_exit = False
                if use_panic_exits:
                    if use_btc_filter and not btc_bullish:
                        override_exit = True
                    if signal_method == 0 and (closes[d-1, s] >= exit_ma[d-1, s] and curr_closes[s] < exit_ma[d, s]):
                        override_exit = True
                    if signal_method == 1 and signal_reversal:
                        override_exit = True

                should_exit_full    = False
                should_exit_partial = False

                if exit_method == 0:
                    if not half_sold[s] and curr_closes[s] >= tp_trigger_price[s]:
                        should_exit_partial = True
                    if half_sold[s]:
                        potential_new_sl = curr_closes[s] - atr[d, s] * trail_mult
                        if potential_new_sl > stop_loss_price[s]:
                            stop_loss_price[s] = potential_new_sl
                    if curr_closes[s] < stop_loss_price[s]:
                        should_exit_full = True
                elif exit_method == 1:
                    if curr_closes[s] < high_since_entry[s] * (1.0 - trail_pct / 100.0):
                        should_exit_full = True
                elif exit_method == 2:
                    if curr_closes[s] < high_since_entry[s] - (exit_atr_mult * atr[d, s]):
                        should_exit_full = True
                elif exit_method == 3:
                    if signal_reversal:
                        should_exit_full = True

                if override_exit:
                    should_exit_full    = True
                    should_exit_partial = False 

                if should_exit_full:
                    pending_exit[s] = True
                elif should_exit_partial:
                    pending_partial[s] = True

        # ── WATCHLIST ADDITIONS ────────
        for s in range(n_stocks):
            if not wl_active[s] and not newly_invalidated[s] and eligible_mask[d, s]:
                if entry_fast[d-1, s] <= entry_slow[d-1, s] and entry_fast[d, s] > entry_slow[d, s] and trend_ok[d, s]:
                    
                    # NEW: Apply BTC filter ONLY if enabled
                    valid_entry = btc_bullish if use_btc_filter else True
                    
                    if valid_entry and adx_threshold > 0.0 and adx[d, s] < adx_threshold:
                        valid_entry = False
                    if valid_entry:
                        wl_active[s]        = True
                        wl_entry_prices[s]  = curr_closes[s]   # FIXED: was opens[d,s] -- the
                                                                # crossover is confirmed using
                                                                # day d's CLOSE (design decision
                                                                # #5), so the close is the actual
                                                                # price at signal-detection time;
                                                                # the open predates the move that
                                                                # produced the signal.
                        wl_is_pyramid[s]    = in_pos[s] and not pending_exit[s]
                        wl_entry_day[s]      = d
                        if not wl_is_pyramid[s]:
                            high_since_entry[s] = opens[d, s]
                            # BUG FIX: shadow SL/TP were never initialized here --
                            # they stayed at their np.zeros() default (or a STALE
                            # value left over from the last time this stock index
                            # held a position/candidate), which made
                            # `curr_closes[s] >= wl_tp_trigger_price[s]` true on
                            # day one (0.0 target) and forced an immediate,
                            # meaningless breakeven-stop -- effectively neutering
                            # every fresh candidate right after it was added. Now
                            # mirrors the exact ATR-based formula used at a REAL
                            # buy (ATR fallback to prior day, then a 2%-of-price
                            # floor if ATR itself is unavailable). Pyramid
                            # candidates skip this -- they ride the REAL
                            # position's own SL/TP, per the invalidation loop's
                            # `not wl_is_pyramid[s]` guard above.
                            entry_atr = atr[d, s] if atr[d, s] > 0 else atr[d - 1, s]
                            if not (entry_atr > 0):
                                entry_atr = curr_closes[s] * 0.02
                            wl_stop_loss_price[s]  = curr_closes[s] - entry_atr * sl_mult
                            wl_tp_trigger_price[s] = curr_closes[s] + entry_atr * tp_mult
                            wl_half_sold[s]        = False

        # ── CAPITAL DEPLOYMENT (decision #16: equal-weight target, not a
        # fixed MONTHLY_SIP chunk) ────────
        # n_eligible = how many coins are actually tradeable AS OF TODAY
        # (eligible_mask[d,:]) -- NOT the full end-state universe, which
        # would leak forward information about coins that haven't listed
        # yet. Recomputed every deployment step since it can change over
        # the backtest as coins list; invested_cost is recomputed every
        # step too since each buy changes it for the next candidate.
        n_eligible = 0
        for s in range(n_stocks):
            if eligible_mask[d, s]:
                n_eligible += 1
        if n_eligible < 1:
            n_eligible = 1

        while cash_pool >= MIN_TICKET_SIZE:
            best_rank = -999999.0
            best_s    = -1
            for s in range(n_stocks):
                if wl_active[s]:
                    if wl_is_pyramid[s] and not in_pos[s]:
                        wl_active[s]     = False
                        wl_is_pyramid[s] = False
                        continue

                    # PREVIOUS FIX: Generalize rank selection
                    rank = -999.0
                    if wl_rank_method == 0 and wl_entry_prices[s] > 0:
                        rank = (wl_entry_prices[s] - curr_closes[s]) / wl_entry_prices[s]
                    elif wl_rank_method == 1 and entry_fast[d, s] > 0:
                        rank = -abs(entry_fast[d, s] - entry_slow[d, s]) / entry_fast[d, s]
                    elif wl_rank_method == 2 and entry_slow[d, s] > 0:
                        rank = (entry_fast[d, s] - entry_slow[d, s]) / entry_slow[d, s]
                        
                    if rank > best_rank:
                        best_rank = rank
                        best_s    = s

            if best_s == -1: break

            buy_price = opens[d + 1, best_s] if d + 1 < n_days else curr_closes[best_s]
            if not (buy_price > 0):
                wl_active[best_s] = False
                continue

            # equal-weight target size: (idle cash + cost-basis of everything
            # already open) / n_eligible. Cost basis, NOT mark-to-market --
            # an open position's unrealized paper gain must not inflate the
            # size of the NEXT, unrelated trade (see decision #16). If the
            # wallet can't cover a full target, it spends whatever's left
            # instead of skipping the trade outright.
            invested_cost = 0.0
            for s2 in range(n_stocks):
                if in_pos[s2]:
                    invested_cost += shares_held[s2] * entry_prices[s2]
            target_size = (cash_pool + invested_cost) / n_eligible
            buy_amount  = target_size if target_size <= cash_pool else cash_pool
            if buy_amount < MIN_TICKET_SIZE:
                break

            cash_pool  -= buy_amount
            new_shares  = buy_amount / (buy_price * (1.0 + buy_cost_pct))

            if wl_is_pyramid[best_s] and in_pos[best_s]:
                old_cost = shares_held[best_s] * entry_prices[best_s]
                new_cost = new_shares * buy_price
                shares_held[best_s]  += new_shares
                entry_prices[best_s]  = (old_cost + new_cost) / shares_held[best_s]

                entry_atr = atr[d, best_s] if atr[d, best_s] > 0 else atr[d - 1, best_s]
                if not (entry_atr > 0):
                    entry_atr = buy_price * 0.02
                potential_new_sl = entry_prices[best_s] - entry_atr * sl_mult
                if potential_new_sl > stop_loss_price[best_s]:
                    stop_loss_price[best_s] = potential_new_sl
            else:
                in_pos[best_s]           = True
                entry_prices[best_s]     = buy_price
                entry_days[best_s]       = d
                shares_held[best_s]      = new_shares
                high_since_entry[best_s] = buy_price
                half_sold[best_s]        = False
                entry_atr = atr[d, best_s] if atr[d, best_s] > 0 else atr[d - 1, best_s]
                if not (entry_atr > 0):
                    entry_atr = buy_price * 0.02
                stop_loss_price[best_s]  = buy_price - entry_atr * sl_mult
                tp_trigger_price[best_s] = buy_price + entry_atr * tp_mult

            wl_active[best_s]       = False
            wl_is_pyramid[best_s]   = False

        # ── DAILY PORTFOLIO VALUATION ────────
        curr_val = cash_pool
        for s in range(n_stocks):
            if in_pos[s]:
                curr_val += shares_held[s] * curr_closes[s]

        if d > start_day:
            floor_val = daily_port_val[d - 1] * 0.001
            if curr_val < floor_val:
                n_floor_clamps += 1
            daily_port_val[d] = max(curr_val, floor_val)
        else:
            daily_port_val[d] = curr_val
        daily_bench_val[d] = bench_shares * btc_close[d]

        if daily_port_val[d] > 0:
            cash_frac = cash_pool / daily_port_val[d]
            cash_frac_sum  += cash_frac
            cash_frac_days += 1
            if cash_frac > HIGH_CASH_FRACTION_THRESHOLD:
                days_high_cash += 1

        # ── FLOW-ADJUSTED DAILY RETURN ────────
        if d > start_day:
            prev_p = daily_port_val[d - 1]
            if prev_p > 0:
                daily_ret_p_arr[d] = (daily_port_val[d] - prev_p - sip_injected[d]) / prev_p
            prev_b = daily_bench_val[d - 1]
            if prev_b > 0:
                daily_ret_b_arr[d] = (daily_bench_val[d] - prev_b - bench_sip_injected[d]) / prev_b
            month_compound_p *= (1.0 + daily_ret_p_arr[d])
            month_compound_b *= (1.0 + daily_ret_b_arr[d])

    final_wealth       = daily_port_val[end_day - 1]
    final_bench_wealth = daily_bench_val[end_day - 1]
    cf_days[cf_cnt]    = end_day - 1
    cf_amounts[cf_cnt] = final_wealth
    cf_cnt = min(cf_cnt + 1, MAX_CF - 1)

    bench_cf_days[bench_cf_cnt]    = end_day - 1
    bench_cf_amounts[bench_cf_cnt] = final_bench_wealth
    bench_cf_cnt = min(bench_cf_cnt + 1, MAX_CF - 1)

    trade_count = winning_trades + losing_trades
    avg_bars    = total_bars_in_trades / trade_count if trade_count > 0 else 0.0
    avg_runup   = win_pct_sum  / winning_trades if winning_trades > 0 else 0.0
    avg_loss_r  = loss_pct_sum / losing_trades  if losing_trades  > 0 else 0.0

    max_dd      = 0.0
    peak        = daily_port_val[start_day]
    curr_dd_dur = 0
    max_dd_dur  = 0
    returns_sum      = 0.0
    returns_sq_sum   = 0.0
    returns_cube_sum = 0.0
    returns_quad_sum = 0.0
    downside_sq    = 0.0
    valid_days     = 0

    for d in range(start_day + 1, end_day):
        v = daily_port_val[d]
        if v > peak:
            peak        = v
            curr_dd_dur = 0
        else:
            if peak > 0:
                dd = (v - peak) / peak
                if dd < max_dd: max_dd = dd
            curr_dd_dur += 1
            if curr_dd_dur > max_dd_dur: max_dd_dur = curr_dd_dur

        prev = daily_port_val[d - 1]
        if prev > 0:
            ret = daily_ret_p_arr[d]
            returns_sum      += ret
            returns_sq_sum   += ret * ret
            returns_cube_sum += ret * ret * ret
            returns_quad_sum += ret * ret * ret * ret
            valid_days       += 1
            if ret < 0:
                downside_sq += ret * ret

    sharpe  = 0.0
    sortino = 0.0
    skew    = 0.0
    kurt    = 0.0
    if valid_days > 1:
        mean_ret = returns_sum / valid_days
        var_ret  = (returns_sq_sum / valid_days) - (mean_ret ** 2)
        if var_ret > 0:
            std_ret = math.sqrt(var_ret)
            if std_ret > 0: sharpe = (mean_ret / std_ret) * math.sqrt(CRYPTO_DAYS_PER_YEAR)
        std_down = math.sqrt(downside_sq / valid_days)
        if std_down > 0: sortino = (mean_ret / std_down) * math.sqrt(CRYPTO_DAYS_PER_YEAR)

        if var_ret > 0:
            e_x  = mean_ret
            e_x2 = returns_sq_sum   / valid_days
            e_x3 = returns_cube_sum / valid_days
            e_x4 = returns_quad_sum / valid_days
            m2 = var_ret
            m3 = e_x3 - 3.0 * e_x * e_x2 + 2.0 * e_x ** 3
            m4 = e_x4 - 4.0 * e_x * e_x3 + 6.0 * e_x ** 2 * e_x2 - 3.0 * e_x ** 4
            skew = m3 / (m2 ** 1.5)
            kurt = m4 / (m2 ** 2)

    info_ratio = 0.0
    if month_cnt > MIN_MONTHS_GATE:
        mean_excess = 0.0
        sq_excess   = 0.0
        for i in range(month_cnt):
            ex = port_monthly_returns[i] - bench_monthly_returns[i]
            mean_excess += ex
            sq_excess   += ex * ex
        mean_excess /= month_cnt
        var_excess   = sq_excess / month_cnt - mean_excess ** 2
        if var_excess > 0:
            info_ratio = (mean_excess / math.sqrt(var_excess)) * math.sqrt(12)

    return (final_wealth, final_bench_wealth, total_invested_capital,
            total_wins, total_losses, trade_count,
            winning_trades, losing_trades,
            avg_bars, avg_runup, avg_loss_r,
            max_dd, max_dd_dur,
            sharpe, sortino, info_ratio,
            month_cnt,
            cf_days, cf_amounts, cf_cnt,
            bench_cf_days, bench_cf_amounts, bench_cf_cnt,
            n_floor_clamps,
            cash_frac_sum, cash_frac_days, days_high_cash,
            daily_port_val, daily_bench_val,
            skew, kurt, valid_days)

# ==========================================
# 4b. MONEY-WEIGHTED RETURN SOLVER (unchanged from the stock engine --
# the math is generic; only day_basis differs when called, see below)
# ==========================================

def money_weighted_annual_return(cf_days, cf_amounts, day_basis=CRYPTO_DAYS_PER_YEAR,
                                  r_lo=-0.999, r_hi=10.0, tol=1e-9, max_iter=200):
    if len(cf_amounts) < 2:
        return 0.0, False
    t0 = cf_days[0]
    times = [(d - t0) / day_basis for d in cf_days]

    def npv(r):
        total = 0.0
        for t, amt in zip(times, cf_amounts):
            try:
                total += amt / ((1.0 + r) ** t)
            except (OverflowError, ZeroDivisionError):
                return float('nan')
        return total

    f_lo, f_hi = npv(r_lo), npv(r_hi)
    if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
        return 0.0, False
    if (f_lo > 0) == (f_hi > 0):
        return 0.0, False

    for _ in range(max_iter):
        r_mid = 0.5 * (r_lo + r_hi)
        f_mid = npv(r_mid)
        if not math.isfinite(f_mid):
            return 0.0, False
        if abs(f_mid) < tol or (r_hi - r_lo) < 1e-10:
            return r_mid, True
        if (f_mid > 0) == (f_lo > 0):
            r_lo, f_lo = r_mid, f_mid
        else:
            r_hi, f_hi = r_mid, f_mid
    return 0.5 * (r_lo + r_hi), True


# ==========================================
# 4c. OVERFITTING DIAGNOSTIC -- DSR (Bailey & Lopez de Prado, 2014)
# Identical math to the stock engine -- generic, asset-class agnostic.
# ==========================================

_EULER_GAMMA = 0.5772156649015329
_NORM = NormalDist()


def expected_max_sharpe(sr_std, n_trials):
    if n_trials <= 1 or sr_std <= 0:
        return 0.0
    z1 = _NORM.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _NORM.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return sr_std * ((1 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)


def probabilistic_sharpe_ratio(sr_hat, sr_benchmark, T, skew, kurt):
    denom = math.sqrt(max(1e-12, 1 - skew * sr_hat + ((kurt - 1) / 4.0) * sr_hat ** 2))
    z = (sr_hat - sr_benchmark) * math.sqrt(max(T - 1, 1)) / denom
    return _NORM.cdf(z)


def deflated_sharpe_ratio(sr_hat_daily, all_trial_sharpes_daily, T, skew, kurt):
    n_trials = len(all_trial_sharpes_daily)
    if n_trials < 2:
        return probabilistic_sharpe_ratio(sr_hat_daily, 0.0, T, skew, kurt), 0.0, n_trials
    mean_sr = sum(all_trial_sharpes_daily) / n_trials
    var_sr  = sum((s - mean_sr) ** 2 for s in all_trial_sharpes_daily) / n_trials
    sr_std  = math.sqrt(var_sr)
    sr0 = expected_max_sharpe(sr_std, n_trials)
    dsr = probabilistic_sharpe_ratio(sr_hat_daily, sr0, T, skew, kurt)
    return dsr, sr0, n_trials


# ==========================================
# 5. DATA PREPARATION
# ==========================================

def build_eligibility_mask(n_days, n_stocks):
    """Point-in-time universe hook -- not used by default (same disclosed
    all-eligible posture you confirmed for point 2), kept for parity with
    the stock engine in case you later reconstruct historical top-100
    membership and want to plug it in."""
    return np.ones((n_days, n_stocks), dtype=np.bool_)


def build_liquidity_mask(quote_vol_matrix, lookback_days=LIQUIDITY_LOOKBACK_DAYS,
                          min_avg_usd=MIN_AVG_DAILY_VOLUME_USD):
    """NEW v1.4 -- noise-coin filtering for the full TOP_N_COINS universe.
    A NEW entry is only allowed on days where a coin's TRAILING
    `lookback_days`-day average USDT volume clears `min_avg_usd`. Point-in-
    time and time-varying on purpose: a coin thin in 2019 but liquid today
    doesn't get treated as always-tradeable, and a coin that goes quiet
    after being liquid stops qualifying for new entries from that point on.
    Pre-listing days (NaN in quote_vol_matrix) fall out of the rolling
    window naturally and read as ineligible, same as the MIN_HISTORY_DAYS
    exclusion elsewhere -- no separate NaN-handling needed. Existing
    positions are never force-closed by this, same non-destructive posture
    as build_eligibility_mask() above."""
    vol_df = pd.DataFrame(quote_vol_matrix)
    rolling_avg = vol_df.rolling(window=lookback_days, min_periods=lookback_days).mean().values
    return rolling_avg >= min_avg_usd


def _binance_download_klines(symbol, start_date_str, interval='1d'):
    """Paginated Binance spot klines download (1000 candles/call limit)."""
    start_ts = int(datetime.strptime(start_date_str, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    rows = []
    cursor = start_ts
    while cursor < end_ts:
        params = {'symbol': symbol, 'interval': interval, 'startTime': cursor, 'limit': 1000}
        try:
            r = requests.get(BINANCE_KLINES_URL, params=params, timeout=20)
            if r.status_code != 200:
                break
            batch = r.json()
        except Exception:
            break
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.15)   # be polite to the free public endpoint

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=[
        'OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime',
        'QuoteVol', 'Trades', 'TakerBaseVol', 'TakerQuoteVol', 'Ignore'
    ])
    df['Date'] = pd.to_datetime(df['OpenTime'], unit='ms').dt.normalize()
    for col in ['Open', 'High', 'Low', 'Close', 'QuoteVol']:
        df[col] = df[col].astype(float)
    # NEW v1.4: QuoteVol (USDT-denominated volume, straight from Binance --
    # more accurate than approximating via base-asset Volume * Close) is now
    # kept so build_liquidity_mask() can screen out thin-liquidity coin-days.
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'QuoteVol']].drop_duplicates('Date').set_index('Date')
    return df


def prepare_matrix_data_crypto(tickers, data_dir="data_crypto"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    raw_dfs      = {}
    master_dates = set()

    for ticker in tqdm(tickers, desc="Downloading Binance klines"):
        file_path = f"{data_dir}/{ticker}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            if 'QuoteVol' not in df.columns:
                # NEW v1.4: stale cache from before the liquidity filter
                # existed -- missing the column build_liquidity_mask() needs.
                # Self-healing: redownload once instead of silently disabling
                # the filter for this coin.
                df = _binance_download_klines(ticker, START_DATE)
                if df is not None and len(df) > 200:
                    df.to_csv(file_path)
        else:
            df = _binance_download_klines(ticker, START_DATE)
            if df is not None and len(df) > 200:
                df.to_csv(file_path)
        if df is not None and len(df) >= MIN_HISTORY_DAYS:
            raw_dfs[ticker] = df
            master_dates.update(df.index)
        elif ticker == 'BTCUSDT':
            raise RuntimeError("Could not download BTCUSDT history -- everything downstream "
                                "depends on BTC being present. Check network access to "
                                f"{BINANCE_KLINES_URL} and retry.")

    master_dates = sorted(list(master_dates))
    master_df    = pd.DataFrame(index=master_dates)
    master_df['Month'] = master_df.index.month
    master_df['Year']  = master_df.index.year

    n_days   = len(master_dates)
    stock_names = ['BTCUSDT'] + [t for t in raw_dfs.keys() if t != 'BTCUSDT']
    n_stocks = len(stock_names)

    opens      = np.zeros((n_days, n_stocks))
    highs      = np.zeros((n_days, n_stocks))
    lows       = np.zeros((n_days, n_stocks))
    closes     = np.zeros((n_days, n_stocks))
    atr_matrix = np.zeros((n_days, n_stocks))
    adx_matrix = np.zeros((n_days, n_stocks))
    quote_vol_matrix = np.full((n_days, n_stocks), np.nan)   # NEW v1.4 -- feeds build_liquidity_mask()

    for i, ticker in enumerate(tqdm(stock_names, desc="Building matrix")):
        df = raw_dfs[ticker].reindex(master_dates)
        opens[:,  i] = df['Open'].ffill().values
        highs[:,  i] = df['High'].ffill().values
        lows[:,   i] = df['Low'].ffill().values
        closes[:, i] = df['Close'].ffill().values
        atr_matrix[:, i] = calc_atr_wilder(highs[:, i], lows[:, i], closes[:, i], period=14)
        adx_matrix[:, i] = calc_adx(highs[:, i], lows[:, i], closes[:, i], period=14)
        if 'QuoteVol' in df.columns:
            # NOT forward-filled -- a genuine zero-volume day should count as
            # zero in the liquidity average, not silently inherit yesterday's
            # number the way price series legitimately do.
            quote_vol_matrix[:, i] = df['QuoteVol'].values

    assert stock_names[0] == 'BTCUSDT', "Column 0 must be BTCUSDT -- regime filter/benchmark depend on it."

    eligible_mask = build_eligibility_mask(n_days, n_stocks)
    liquidity_mask = build_liquidity_mask(quote_vol_matrix)
    n_before = int(eligible_mask.sum())
    eligible_mask = eligible_mask & liquidity_mask
    n_after = int(eligible_mask.sum())
    print(f"Liquidity filter (blocks NEW entries only, existing positions unaffected): "
          f"{n_before - n_after:,} of {n_before:,} eligible coin-days excluded "
          f"(trailing {LIQUIDITY_LOOKBACK_DAYS}-day avg USDT volume < ${MIN_AVG_DAILY_VOLUME_USD:,.0f}).")

    years_arr  = master_df['Year'].values.astype(np.int32)

    return (opens, closes, atr_matrix, adx_matrix,
            master_df['Month'].values, years_arr,
            stock_names, eligible_mask, master_dates)


# ==========================================
# 5b. SIP SCHEDULE (NEW v1.2, decision #14) -- built OUTSIDE the numba
# simulation from real calendar dates, then handed in as a plain boolean
# array. simulate_portfolio_crypto no longer decides for itself when the
# SIP lands; it just injects cash on whichever day sip_flag[d] is True.
# ==========================================

def build_sip_schedule(master_dates, mode='month_start', seed=None,
                        min_day=SIP_RANDOM_MIN_DAY, max_day=SIP_RANDOM_MAX_DAY):
    """Returns a boolean array (len == len(master_dates)) marking the day(s)
    each month's MONTHLY_SIP lands on.

    mode='month_start': flags the first trading day of every calendar month
      -- reproduces the engine's original, always-day-1 behavior. This is
      the DEFAULT schedule used for the main Optuna search, WFO validation,
      and all console reports, so headline numbers stay comparable to
      earlier runs of this script.

    mode='random': draws ONE uniformly random calendar day-of-month in
      [min_day, max_day] per (year, month) with a seeded RNG (same seed ->
      same schedule, for reproducibility), then flags the first trading day
      on/after that target within the month (crypto trades daily, so this
      is normally an exact hit). Used ONLY by
      passes_temporal_robustness_check() -- never by the main search, since
      that would multiply the trial budget by TEMPORAL_ROBUSTNESS_RUNS for
      no benefit.
    """
    n = len(master_dates)
    flags = np.zeros(n, dtype=np.bool_)

    if mode == 'month_start':
        prev_key = None
        for i, d in enumerate(master_dates):
            key = (d.year, d.month)
            if key != prev_key:
                flags[i] = True
                prev_key = key
        return flags

    if mode != 'random':
        raise ValueError(f"build_sip_schedule: unknown mode '{mode}'")

    rng = np.random.default_rng(seed)
    by_month = {}
    for i, d in enumerate(master_dates):
        key = (d.year, d.month)
        by_month.setdefault(key, []).append(i)

    for key in sorted(by_month.keys()):
        idxs = by_month[key]
        target_day = int(rng.integers(min_day, max_day + 1))
        chosen = idxs[-1]   # fallback: last trading day of the month
        for i in idxs:
            if master_dates[i].day >= target_day:
                chosen = i
                break
        flags[chosen] = True
    return flags


# ==========================================
# 6. YEARLY BREAKDOWN (new -- feeds recency weighting, the consistency
# term, the regime term, AND the profit-concentration hard gate)
# ==========================================

def compute_yearly_breakdown(daily_port_val, daily_bench_val, years_arr,
                              cf_days, cf_amounts, cf_cnt,
                              bench_cf_days, bench_cf_amounts, bench_cf_cnt,
                              start_day, end_day):
    """For every calendar year touched by [start_day, end_day), compute:
      - money-weighted IRR of the STRATEGY for that year alone (bookend
        the year's start/end portfolio value as synthetic cash flows,
        plus whatever real SIP cash flows landed inside that year --
        exactly the same technique run_wfo_validation already uses to
        chain IS->OOS starting_wealth, just applied year-by-year).
      - money-weighted IRR of BTC (the benchmark) for the same year, using
        the identical cash-flow timing/amounts (SIP dates and sizes are
        the same for both by construction) so the two are directly
        comparable.
      - nominal $ profit for that year (for the concentration gate).
      - coverage_frac: how much of a full year this slice actually spans
        (a partial first/last year -- or a coin-implied gap -- counts
        less, both for recency weighting and for the concentration gate's
        denominator).
    Returns a list of dicts, one per year, sorted ascending.
    """
    day_years = years_arr[start_day:end_day]
    if len(day_years) == 0:
        return []
    unique_years = sorted(set(int(y) for y in day_years))

    cf_days_arr = np.array(cf_days[:cf_cnt])
    cf_amts_arr = np.array(cf_amounts[:cf_cnt])
    # NEW: benchmark now has its own, independent cash-flow ledger (BTC's
    # SIP lands every month unconditionally; the strategy's SIP is withheld
    # in months it's still sitting on an undeployed ticket), so the two
    # schedules can diverge and must be masked into per-year slices
    # separately instead of reusing the strategy's year_cf_days/amounts.
    bench_cf_days_arr = np.array(bench_cf_days[:bench_cf_cnt])
    bench_cf_amts_arr = np.array(bench_cf_amounts[:bench_cf_cnt])

    out = []
    for yr in unique_years:
        yr_day_idxs = np.where(day_years == yr)[0] + start_day
        if len(yr_day_idxs) < 5:
            continue
        y_start, y_end = int(yr_day_idxs[0]), int(yr_day_idxs[-1])

        port_start_val = daily_port_val[max(y_start - 1, start_day)]
        port_end_val   = daily_port_val[y_end]
        bench_start_val = daily_bench_val[max(y_start - 1, start_day)]
        bench_end_val   = daily_bench_val[y_end]

        mask = (cf_days_arr >= y_start) & (cf_days_arr <= y_end)
        year_cf_days    = list(cf_days_arr[mask])
        year_cf_amounts = list(cf_amts_arr[mask])
        # exclude the global bookend entries (terminal/starting wealth cash
        # flows land outside a mid-run year's own boundary already, since
        # those are only appended at the very first/last day of the WHOLE
        # simulation -- but guard anyway) then bookend THIS year:
        port_cf_days    = [y_start] + year_cf_days + [y_end]
        port_cf_amounts = [-port_start_val] + year_cf_amounts + [port_end_val]

        bench_mask = (bench_cf_days_arr >= y_start) & (bench_cf_days_arr <= y_end)
        year_bench_cf_days    = list(bench_cf_days_arr[bench_mask])
        year_bench_cf_amounts = list(bench_cf_amts_arr[bench_mask])
        year_bench_cf_days    = [y_start] + year_bench_cf_days + [y_end]
        year_bench_cf_amounts = [-bench_start_val] + year_bench_cf_amounts + [bench_end_val]

        port_irr, port_ok  = money_weighted_annual_return(port_cf_days, port_cf_amounts)
        bench_irr, bench_ok = money_weighted_annual_return(year_bench_cf_days, year_bench_cf_amounts)

        nominal_contrib = sum(-a for a in year_cf_amounts if a < 0)   # SIP $ added this year
        nominal_profit  = (port_end_val - port_start_val) - nominal_contrib

        coverage_frac = min(1.0, len(yr_day_idxs) / YEAR_FULL_COVERAGE_DAYS)

        out.append({
            'year': yr,
            'port_irr': port_irr if port_ok else 0.0,
            'bench_irr': bench_irr if bench_ok else 0.0,
            'nominal_profit': nominal_profit,
            'coverage_frac': coverage_frac,
            'days': len(yr_day_idxs),
        })
    return out


def _recency_weight(year, min_year, max_year):
    if max_year <= min_year:
        return RECENCY_WEIGHT_MAX
    frac = (year - min_year) / (max_year - min_year)
    return RECENCY_WEIGHT_MIN + (RECENCY_WEIGHT_MAX - RECENCY_WEIGHT_MIN) * frac


# ==========================================
# 7. SCORING FUNCTION (crypto v1)
# ==========================================

def compute_score_crypto(metrics, yearly, is_oos=False):
    """
    score = [ Calmar*0.20 + Sortino(cap 4)*0.10 + IR*0.20 + EV_in_R*0.10
              + WinRate_bonus*0.10 + Consistency*0.20 + Regime*0.10 ]
            * stat_confidence * drawdown_penalty

    CONSISTENCY (Calmar-scaled, recency-weighted across calendar years):
      w_y      = recency_weight(year) * coverage_frac(year)
      mean_r   = weighted-mean(port_irr_y ; w_y)
      std_r    = weighted-std(port_irr_y ; w_y)
      worst_r  = min(port_irr_y)   -- unweighted on purpose: a bad crash
                 year is informative about tail risk regardless of when
                 it happened; the reward side is recency-weighted, the
                 tail-risk side isn't softened by the passage of time.
      consistency = (mean_r - K_STD*std_r - K_WORST*max(0,-worst_r)) / max_dd

    REGIME (bull/bear split by BTC's OWN annual return that year):
      bull years:  bench_irr_y > +10%  -> bull_capture = weighted-mean(port_irr_y / bench_irr_y)
      bear years:  bench_irr_y < -10%  -> bear_defense = weighted-mean(port_irr_y - bench_irr_y) / max_dd
      regime = W_BULL_CAPTURE*clip(bull_capture,-1,3) + W_BEAR_DEFENSE*clip(bear_defense,-2,2)
      (either side contributes 0 if that bucket has no qualifying years)

    HARD GATES (all must pass or score = -999): trade/month/drawdown/PF/
    ROI/win-rate gates (win-rate bar lowered per decision #6) PLUS the
    profit-concentration gate (decision #9). NOTE: an earlier version also
    hard-gated on beating BTC's own DCA return by 20% -- removed per
    decision #10 in the module docstring, it rejected every real trial
    because it demanded beating BTC's 2020-2021 run in absolute dollar
    terms. The IR and regime(bull_capture) terms below reward genuine
    outperformance continuously instead.
    """
    roi            = metrics['roi']
    bench_roi      = metrics['bench_roi']
    sortino        = metrics['sortino']
    ir             = metrics['ir']
    max_dd         = abs(metrics['max_dd'])
    trades         = metrics['trades']
    avg_runup      = metrics['avg_runup']
    avg_loss       = abs(metrics['avg_loss'])
    month_cnt      = metrics['month_cnt']
    winning_trades = metrics['winning_trades']
    losing_trades  = metrics['losing_trades']
    annual_return  = metrics['annual_return']

    target_trades = max(15, MIN_TRADES_GATE * (WFO_OOS_PCT / WFO_IS_PCT)) if is_oos else MIN_TRADES_GATE
    target_months = max(12, MIN_MONTHS_GATE * (WFO_OOS_PCT / WFO_IS_PCT)) if is_oos else MIN_MONTHS_GATE

    if trades    < target_trades: return -999.0
    if month_cnt < target_months: return -999.0
    if max_dd    > 0.50:          return -999.0
    if metrics['pf'] < 1.10:      return -999.0
    if roi       <= 0:            return -999.0
    if avg_loss  <= 0:            return -999.0
    if not math.isfinite(annual_return): return -999.0

    total_trades_counted = winning_trades + losing_trades
    actual_wr = winning_trades / total_trades_counted if total_trades_counted > 0 else 0.0
    if actual_wr < MIN_WIN_RATE_GATE: return -999.0

    # NEW hard gate: profit concentration (decision #9)
    if yearly:
        total_profit = sum(y['nominal_profit'] for y in yearly)
        if total_profit > 0:
            max_share = max(y['nominal_profit'] / total_profit for y in yearly)
            if max_share > CONCENTRATION_GATE:
                return -999.0

    if max_dd < 0.001: max_dd = 0.001
    calmar = annual_return / max_dd

    ev      = actual_wr * avg_runup - (1.0 - actual_wr) * avg_loss
    ev_in_r = ev / avg_loss

    wr_bonus = max(0.0, actual_wr - 0.50) * 4.0
    sortino_capped = min(sortino, 4.0)
    ir_scaled = max(0.0, ir) * math.sqrt(max(1.0, month_cnt) / 12.0)
    stat_conf = min(1.0, math.sqrt(trades / target_trades))

    if max_dd <= 0.20:   dd_penalty = 1.0
    elif max_dd <= 0.35: dd_penalty = 1.0 - (max_dd - 0.20) * 2.0
    else:                dd_penalty = 0.70 * math.exp(-2.0 * (max_dd - 0.35))

    # NEW (decision #11): drop years whose coverage_frac is too low to trust
    # an IRR-annualized figure from (a partial/in-progress calendar year can
    # annualize a few months of noise into a triple-digit reading). Only
    # affects the CONSISTENCY/REGIME terms below -- the profit-concentration
    # gate above uses raw nominal dollars and isn't subject to this.
    scoring_years = [y for y in yearly if y['coverage_frac'] >= MIN_YEAR_COVERAGE_FOR_SCORING]

    # ── CONSISTENCY term ────────────────────────────────────────────────
    consistency = 0.0
    if scoring_years:
        yrs = [y['year'] for y in scoring_years]
        min_year, max_year = min(yrs), max(yrs)
        w_list = [_recency_weight(y['year'], min_year, max_year) * y['coverage_frac'] for y in scoring_years]
        r_list = [y['port_irr'] for y in scoring_years]
        w_sum = sum(w_list)
        if w_sum > 0:
            mean_r = sum(w * r for w, r in zip(w_list, r_list)) / w_sum
            var_r  = sum(w * (r - mean_r) ** 2 for w, r in zip(w_list, r_list)) / w_sum
            std_r  = math.sqrt(max(0.0, var_r))
            worst_r = min(r_list)
            consistency = (mean_r - K_STD_PENALTY * std_r
                           - K_WORST_PENALTY * max(0.0, -worst_r)) / max_dd

    # ── REGIME term ─────────────────────────────────────────────────────
    regime = 0.0
    if scoring_years:
        bull_ratios = []
        bull_weights = []
        bear_diffs = []
        bear_weights = []
        sy_min_year = min(yr['year'] for yr in scoring_years)
        sy_max_year = max(yr['year'] for yr in scoring_years)
        for y in scoring_years:
            w = _recency_weight(y['year'], sy_min_year, sy_max_year) * y['coverage_frac']
            if y['bench_irr'] > BULL_YEAR_BTC_THRESHOLD:
                ratio = y['port_irr'] / y['bench_irr'] if y['bench_irr'] != 0 else 0.0
                bull_ratios.append(max(-1.0, min(3.0, ratio)))
                bull_weights.append(w)
            elif y['bench_irr'] < BEAR_YEAR_BTC_THRESHOLD:
                diff = (y['port_irr'] - y['bench_irr']) / max_dd
                bear_diffs.append(max(-2.0, min(2.0, diff)))
                bear_weights.append(w)
        bull_capture = (sum(r * w for r, w in zip(bull_ratios, bull_weights)) / sum(bull_weights)) if bull_weights else 0.0
        bear_defense = (sum(d * w for d, w in zip(bear_diffs, bear_weights)) / sum(bear_weights)) if bear_weights else 0.0
        regime = W_BULL_CAPTURE * bull_capture + W_BEAR_DEFENSE * bear_defense

    score = (
        calmar      * W_CALMAR +
        sortino_capped * W_SORTINO +
        ir_scaled   * W_IR +
        ev_in_r     * W_EV +
        wr_bonus    * W_WR_BONUS +
        consistency * W_CONSISTENCY +
        regime      * W_REGIME
    ) * stat_conf * dd_penalty

    return score


def diagnose_gates(metrics, yearly, is_oos=False):
    """Mirrors compute_score_crypto's hard gates exactly (same order, same
    thresholds) but instead of collapsing every failure into the same
    -999.0, reports each gate's actual value vs its threshold and whether
    it passed. Exists because a hard hit and a near-miss look identical as
    a bare score, and 'the OOS run scored -999' is not, by itself, enough
    information to tell overfitting from a data/coverage quirk. Returns a
    list of (gate_name, passed: bool, actual, threshold) tuples, in the
    same order compute_score_crypto checks them (so the FIRST failing row
    is the one that actually determined the -999 -- later rows are
    informational only, since compute_score_crypto would have already
    returned before reaching them)."""
    if not metrics:
        return [('has_any_invested_capital', False, 0, '>0 -- '
                 'evaluate_params_crypto returned an empty metrics dict, '
                 'meaning t_invested<=0 for this window (no SIP ever landed '
                 'inside [start_day, end_day) -- check the window boundaries, '
                 'not the strategy parameters)')]

    roi            = metrics['roi']
    max_dd         = abs(metrics['max_dd'])
    trades         = metrics['trades']
    month_cnt      = metrics['month_cnt']
    winning_trades = metrics['winning_trades']
    losing_trades  = metrics['losing_trades']
    annual_return  = metrics['annual_return']
    avg_loss       = abs(metrics['avg_loss'])
    pf             = metrics['pf']

    target_trades = max(15, MIN_TRADES_GATE * (WFO_OOS_PCT / WFO_IS_PCT)) if is_oos else MIN_TRADES_GATE
    target_months = max(12, MIN_MONTHS_GATE * (WFO_OOS_PCT / WFO_IS_PCT)) if is_oos else MIN_MONTHS_GATE
    total_trades_counted = winning_trades + losing_trades
    actual_wr = winning_trades / total_trades_counted if total_trades_counted > 0 else 0.0

    rows = [
        ('trades >= target',        trades >= target_trades,          trades, f'>= {target_trades:.1f}'),
        ('months >= target',        month_cnt >= target_months,       month_cnt, f'>= {target_months:.1f}'),
        ('max_dd <= 50%',           max_dd <= 0.50,                    f'{max_dd*100:.1f}%', '<= 50.0%'),
        ('profit_factor >= 1.10',   pf >= 1.10,                        f'{pf:.2f}', '>= 1.10'),
        ('roi > 0',                 roi > 0,                           f'{roi*100:.1f}%', '> 0%'),
        ('avg_loss > 0',            avg_loss > 0,                      f'{avg_loss:.4f}', '> 0 (needs >=1 losing trade)'),
        ('annual_return finite',    math.isfinite(annual_return),      annual_return, 'finite'),
        ('win_rate >= floor',       actual_wr >= MIN_WIN_RATE_GATE,    f'{actual_wr*100:.1f}%', f'>= {MIN_WIN_RATE_GATE*100:.0f}%'),
    ]

    if yearly:
        total_profit = sum(y['nominal_profit'] for y in yearly)
        if total_profit > 0:
            max_share = max(y['nominal_profit'] / total_profit for y in yearly)
            rows.append((f'profit_concentration <= {CONCENTRATION_GATE*100:.0f}%', max_share <= CONCENTRATION_GATE,
                         f'{max_share*100:.1f}%', f'<= {CONCENTRATION_GATE*100:.0f}%'))
        else:
            rows.append((f'profit_concentration <= {CONCENTRATION_GATE*100:.0f}%', True, 'n/a (total profit <= 0)', f'<= {CONCENTRATION_GATE*100:.0f}%'))
    else:
        rows.append((f'profit_concentration <= {CONCENTRATION_GATE*100:.0f}%', True, 'n/a (no yearly data)', f'<= {CONCENTRATION_GATE*100:.0f}%'))

    return rows


def print_gate_diagnosis(metrics, yearly, is_oos=False, label="GATE DIAGNOSIS"):
    rows = diagnose_gates(metrics, yearly, is_oos=is_oos)
    print(f"\n{'-'*60}\n{label}\n{'-'*60}")
    first_failure_shown = False
    for name, passed, actual, threshold in rows:
        marker = "PASS" if passed else "FAIL"
        print(f"  [{marker}] {name:<28} actual={actual!s:<12} needed {threshold}")
        if not passed and not first_failure_shown:
            print(f"       ^-- this is the gate that produced the -999 score "
                  f"(checks after this one are informational only)")
            first_failure_shown = True
    if first_failure_shown is False:
        print("  All gates listed here pass -- if the score was still -999, "
              "check for an empty metrics dict (t_invested<=0) above.")
    print("-" * 60)


# ==========================================
# 8. EVALUATE PARAMS
# ==========================================

def evaluate_params_crypto(p, opens, closes, atr, adx, months, years_arr,
                           eligible_mask, start_day=0, end_day=-1, is_oos=False,
                           starting_wealth=0.0, sip_flag=None):

    if sip_flag is None:
        raise ValueError("evaluate_params_crypto: sip_flag is required.")

    n_days, n_stocks = closes.shape
    if end_day < 0: end_day = n_days - 1

    signal_method = int(p.get('signal_method', 0))
    
    # NEW: Extract boolean toggles
    use_trend_ma = bool(p.get('use_trend_ma', True))
    use_btc_filter = bool(p.get('use_btc_filter', True))
    use_panic_exits = bool(p.get('use_panic_exits', True))

    entry_fast = np.zeros((n_days, n_stocks))
    entry_slow = np.zeros((n_days, n_stocks))
    trend_ok   = np.ones((n_days, n_stocks), dtype=np.bool_)
    exit_ma    = np.zeros((n_days, n_stocks))
    
    for s in range(n_stocks):
        exit_ma[:, s] = get_ma_cached(closes, s, p['exit_ma_len'], p['exit_ma_type'])
        if signal_method == 1:
            entry_fast[:, s] = get_rsi_smoothed_cached(closes, s, p['rsi_fast_len'], p['rsi_fast_smt'])
            entry_slow[:, s] = get_rsi_smoothed_cached(closes, s, p['rsi_slow_len'], p['rsi_slow_smt'])
        else:
            entry_ma_s = get_ma_cached(closes, s, p['entry_ma_len'], p['entry_ma_type'])
            entry_fast[:, s] = closes[:, s]
            entry_slow[:, s] = entry_ma_s
            
        # NEW: Apply trend filter universally, but only if enabled
        if use_trend_ma:
            trend_ma_s = get_ma_cached(closes, s, p['trend_ma_len'], p['trend_ma_type'])
            trend_ok[:, s] = closes[:, s] > trend_ma_s
        else:
            trend_ok[:, s] = True
            
    btc_ma = get_ma_cached(closes, 0, p['btc_ma_len'], p['btc_ma_type'])

    result = simulate_portfolio_crypto(
        opens, closes, atr, adx, months, btc_ma, entry_fast, entry_slow, trend_ok, exit_ma,
        eligible_mask,
        p['wl_rank'], p.get('adx_thresh', 0.0),
        signal_method,
        p['exit_method'], p['sl_mult'], p['tp_mult'], p['trail_mult'],
        p['trail_pct'], p['exit_atr_mult'],
        int(start_day), int(end_day),
        float(starting_wealth),
        BUY_COST_PCT, SELL_COST_PCT, ANNUAL_CASH_YIELD,
        sip_flag,
        use_btc_filter, use_panic_exits  # Passed boolean toggles to simulator
    )

    (f_wealth, f_bench, t_invested, wins, losses, trades,
     winning_trades, losing_trades, avg_bars, avg_runup, avg_loss,
     max_dd, max_dd_dur, sharpe, sortino, ir, month_cnt,
     cf_days, cf_amounts, cf_cnt,
     bench_cf_days, bench_cf_amounts, bench_cf_cnt,
     n_floor_clamps, cash_frac_sum, cash_frac_days, days_high_cash,
     daily_port_val, daily_bench_val,
     skew, kurt, valid_days) = result

    if t_invested <= 0: return -999.0, {}

    # `wins`/`losses` here are gross profit $ / gross loss $ (positive), unpacked
    # from simulate_portfolio_crypto's total_wins/total_losses -- NOT trade counts
    # (those are winning_trades/losing_trades below). Profit factor = gross profit
    # / gross loss; a position with only winners and zero losing trades has no
    # finite ratio, so it's floored at a large sentinel (999.0, matching this
    # file's existing -999.0 "clamp" convention) rather than left as inf/NaN,
    # which would otherwise corrupt anything downstream that sorts or averages pf.
    pf = (wins / losses) if losses > 0 else (999.0 if wins > 0 else 0.0)

    # PREVIOUS FIX: Accurate Baseline ROI & IRR Math
    capital_base = starting_wealth + t_invested
    roi = (f_wealth - capital_base) / capital_base if capital_base > 0 else 0.0
    
    # The benchmark must be measured against its OWN injected capital
    bench_capital_base = sum(-a for a in bench_cf_amounts[:bench_cf_cnt] if a < 0)
    bench_roi = (f_bench - bench_capital_base) / bench_capital_base if bench_capital_base > 0 else 0.0

    cf_days_list    = list(cf_days[:cf_cnt])
    cf_amounts_list = list(cf_amounts[:cf_cnt])
    annual_return, irr_converged = money_weighted_annual_return(cf_days_list, cf_amounts_list)
    years_elapsed = max(1.0 / 365.0, (int(end_day) - int(start_day)) / CRYPTO_DAYS_PER_YEAR)
    if not irr_converged:
        annual_return = (f_wealth / capital_base) ** (1.0 / years_elapsed) - 1.0 if capital_base > 0 else -1.0

    # NEW: benchmark now has its own cash-flow ledger (BTC's SIP lands every
    # month unconditionally; the strategy's is withheld while it's still
    # sitting on an undeployed ticket), so it can no longer borrow the
    # strategy's cf_days/cf_amounts with the last entry swapped for f_bench
    # -- that assumed identical cash-flow timing, which is no longer true.
    bench_cf_days_list    = list(bench_cf_days[:bench_cf_cnt])
    bench_cf_amounts_list = list(bench_cf_amounts[:bench_cf_cnt])
    bench_annual_return, bench_irr_converged = money_weighted_annual_return(
        bench_cf_days_list, bench_cf_amounts_list)
    if not bench_irr_converged:
        # bench_cf_amounts_list already includes the -starting_wealth entry
        # (if any) alongside every -MONTHLY_SIP contribution, so summing all
        # negative entries already gives the full capital base -- adding
        # starting_wealth again here would double-count it.
        bench_capital_base = sum(-a for a in bench_cf_amounts_list[:-1] if a < 0)
        bench_annual_return = ((f_bench / bench_capital_base) ** (1.0 / years_elapsed) - 1.0
                                if bench_capital_base > 0 else -1.0)

    avg_cash_frac      = (cash_frac_sum / cash_frac_days) if cash_frac_days > 0 else 0.0
    pct_days_high_cash = (days_high_cash / cash_frac_days * 100.0) if cash_frac_days > 0 else 0.0

    yearly = compute_yearly_breakdown(daily_port_val, daily_bench_val, years_arr,
                                       cf_days, cf_amounts, cf_cnt,
                                       bench_cf_days, bench_cf_amounts, bench_cf_cnt,
                                       int(start_day), int(end_day))

    metrics = {
        'roi': roi, 'bench_roi': bench_roi, 'alpha': roi - bench_roi,
        'pf': pf, 'wealth': f_wealth, 'bench_wealth': f_bench,
        'trades': trades, 'winning_trades': winning_trades, 'losing_trades': losing_trades,
        'sharpe': sharpe, 'sortino': sortino, 'ir': ir,
        'max_dd': max_dd, 'max_dd_dur': max_dd_dur,
        'avg_bars': avg_bars, 'avg_runup': avg_runup, 'avg_loss': avg_loss,
        't_invested': t_invested, 'month_cnt': month_cnt,
        'starting_wealth': starting_wealth,
        'annual_return': annual_return,
        'bench_annual_return': bench_annual_return, 
        'irr_converged': irr_converged,
        'cf_days': cf_days_list, 'cf_amounts': cf_amounts_list,
        'n_floor_clamps': int(n_floor_clamps),
        'avg_cash_frac': avg_cash_frac,
        'pct_days_high_cash': pct_days_high_cash,
        'yearly': yearly,
        'skew': float(skew), 'kurtosis': float(kurt), 'valid_days': int(valid_days),
        'signal_method': signal_method,
    }

    score = compute_score_crypto(metrics, yearly, is_oos=is_oos)
    return score, metrics

# ==========================================
# 9. NEIGHBORHOOD STABILITY
# ==========================================

def passes_neighborhood_check(p, base_score, opens, closes, atr, adx,
                               months, years_arr, eligible_mask,
                               start_day, end_day, sip_flag):
    if base_score <= 0: return True

    signal_method = p.get('signal_method', 0)

    # Perturb whichever entry-signal parameters are ACTUALLY active for this
    # champion's configuration (same principle as the stock engine's
    # neighborhood check -- perturbing an unused parameter trivially passes
    # and understates how brittle the fit really is).
    if signal_method == 1:
        perturbations = [
            {'rsi_fast_len': p['rsi_fast_len'] + 3},
            {'rsi_fast_len': max(3, p['rsi_fast_len'] - 3)},
            {'rsi_slow_len': p['rsi_slow_len'] + 3},
            {'rsi_slow_len': max(3, p['rsi_slow_len'] - 3)},
            {'rsi_fast_smt': p['rsi_fast_smt'] + 3},
            {'rsi_fast_smt': max(3, p['rsi_fast_smt'] - 3)},
            {'rsi_slow_smt': p['rsi_slow_smt'] + 3},
            {'rsi_slow_smt': max(3, p['rsi_slow_smt'] - 3)},
            {'trend_ma_len': p['trend_ma_len'] + 5},
            {'trend_ma_len': max(5, p['trend_ma_len'] - 5)},
        ]
    else:
        perturbations = [
            {'entry_ma_len': p['entry_ma_len'] + 3},
            {'entry_ma_len': max(3, p['entry_ma_len'] - 3)},
        ]

    perturbations += [
        {'exit_ma_len':  p['exit_ma_len'] + 3},
        {'exit_ma_len':  max(3, p['exit_ma_len'] - 3)},
        {'trail_pct':    p['trail_pct'] * 0.85},
        {'trail_pct':    p['trail_pct'] * 1.15},
        {'exit_atr_mult': p['exit_atr_mult'] * 0.80},
        {'exit_atr_mult': p['exit_atr_mult'] * 1.20},
    ]

    for delta in perturbations:
        n_p = p.copy()
        n_p.update(delta)
        if signal_method == 0:
            if n_p['entry_ma_len'] < 3 or n_p['entry_ma_len'] > 300: continue
        else:
            if n_p['rsi_fast_len'] < 3 or n_p['rsi_slow_len'] > 300: continue
        if n_p['exit_ma_len']  < 3 or n_p['exit_ma_len']  > 300: continue

        n_score, _ = evaluate_params_crypto(n_p, opens, closes, atr, adx,
                                            months, years_arr, eligible_mask,
                                            start_day, end_day, is_oos=False,
                                            sip_flag=sip_flag)
        if n_score < base_score * NEIGHBOR_THRESHOLD: return False
    return True


# ==========================================
# 9b. TEMPORAL ROBUSTNESS (NEW v1.2, decision #14) -- "did we get lucky on
# WHEN the cash landed, not just WHICH coins it landed in"
# ==========================================

def passes_temporal_robustness_check(p, base_score, opens, closes, atr, adx,
                                      months, years_arr, eligible_mask,
                                      master_dates, start_day, end_day,
                                      n_runs=TEMPORAL_ROBUSTNESS_RUNS,
                                      score_threshold=TEMPORAL_ROBUSTNESS_SCORE_MIN,
                                      pass_frac=TEMPORAL_ROBUSTNESS_PASS_FRAC,
                                      verbose=True):
    """Re-evaluates the SAME params n_runs times, each time with the monthly
    SIP landing on a different random calendar day instead of always day 1
    (same dollar amount, same benchmark timing -- only the day changes).
    Requires >= pass_frac of the runs to retain >= score_threshold of the
    base (month-start) score. A candidate whose edge secretly depends on
    always deploying capital right at month-start conditions will show wide
    score variance / frequent gate failures here; a genuine trend-following
    edge shouldn't care what day of the month the money shows up.
    """
    if base_score <= 0:
        return True, [], 0.0

    scores = []
    win_rates = []
    for seed in range(n_runs):
        sip_flag = build_sip_schedule(master_dates, mode='random', seed=seed)
        n_score, n_metrics = evaluate_params_crypto(
            p, opens, closes, atr, adx, months, years_arr, eligible_mask,
            start_day, end_day, is_oos=False, starting_wealth=0.0, sip_flag=sip_flag
        )
        scores.append(n_score)
        if n_metrics:
            wt, lt = n_metrics.get('winning_trades', 0), n_metrics.get('losing_trades', 0)
            if wt + lt > 0:
                win_rates.append(wt / (wt + lt))

    scores_arr   = np.array(scores)
    frac_holding = float(np.mean(scores_arr >= base_score * score_threshold))
    median_score = float(np.median(scores_arr))

    if verbose:
        wr_note = ""
        if win_rates:
            wr_arr = np.array(win_rates)
            wr_note = (f" | win-rate across runs: {wr_arr.mean()*100:.1f}% "
                       f"+/- {wr_arr.std()*100:.1f}pp")
        print(f"    [temporal robustness] {n_runs} random-SIP-date runs -- "
              f"median score {median_score:.4f} vs base {base_score:.4f} "
              f"({frac_holding*100:.0f}% of runs retained >= {score_threshold*100:.0f}% of "
              f"base, need >= {pass_frac*100:.0f}%){wr_note}")

    return frac_holding >= pass_frac, scores, median_score


# ==========================================
# 10. WALK-FORWARD VALIDATION
# ==========================================

def run_wfo_validation(best_params, opens, closes, atr, adx,
                       months, years_arr, eligible_mask, master_dates, sip_flag):
    n_days    = closes.shape[0]
    is_end    = int(n_days * WFO_IS_PCT)
    oos_start = is_end

    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION")
    print(f"   In-Sample days:     0 -> {is_end}  (~{is_end/365:.1f} years)")
    print(f"   Out-of-Sample days: {oos_start} -> {n_days}  (~{(n_days - oos_start)/365:.1f} years)")
    print("=" * 60)

    is_score, is_m = evaluate_params_crypto(
        best_params, opens, closes, atr, adx, months, years_arr, eligible_mask,
        start_day=0, end_day=is_end, is_oos=False, starting_wealth=0.0, sip_flag=sip_flag
    )
    is_end_wealth = is_m.get('wealth', 0.0) if is_score > -900 else 0.0

    oos_score, oos_m = evaluate_params_crypto(
        best_params, opens, closes, atr, adx, months, years_arr, eligible_mask,
        start_day=oos_start, end_day=n_days - 1, is_oos=True, starting_wealth=is_end_wealth,
        sip_flag=sip_flag
    )

    # decision #12: continuous, not a hard cliff at 0 the moment OOS goes
    # negative -- lets you see HOW close/far a rejected run actually was.
    # The deploy decision itself is untouched: still requires both legs
    # positive and robustness >= ROBUSTNESS_DEPLOY_THRESHOLD to save.
    if is_score > 0:
        robustness = (oos_score / is_score) if oos_score > 0 else (oos_score / abs(is_score))
    else:
        robustness = 0.0

    if is_score > 0:
        wt, lt = is_m.get('winning_trades', 0), is_m.get('losing_trades', 0)
        wr = wt / (wt + lt) * 100 if (wt + lt) > 0 else 0.0
        print(f"\nIn-Sample  Score: {is_score:.4f} | ROI: {is_m.get('roi', 0)*100:.1f}% | "
              f"Annual Return (IRR): {is_m.get('annual_return', 0)*100:.1f}% | "
              f"Sharpe: {is_m.get('sharpe', 0):.2f} | MaxDD: {abs(is_m.get('max_dd', 0))*100:.1f}% | "
              f"WinRate: {wr:.1f}% | Trades: {is_m.get('trades', 0)}")
    else:
        print_gate_diagnosis(is_m, is_m.get('yearly', []), is_oos=False, label="IN-SAMPLE GATE DIAGNOSIS")
    if oos_score > -900:
        wt, lt = oos_m.get('winning_trades', 0), oos_m.get('losing_trades', 0)
        wr = wt / (wt + lt) * 100 if (wt + lt) > 0 else 0.0
        print(f"Out-of-Sample Score: {oos_score:.4f} | ROI: {oos_m.get('roi', 0)*100:.1f}% | "
              f"Annual Return (IRR): {oos_m.get('annual_return', 0)*100:.1f}% | "
              f"Sharpe: {oos_m.get('sharpe', 0):.2f} | MaxDD: {abs(oos_m.get('max_dd', 0))*100:.1f}% | "
              f"WinRate: {wr:.1f}% | Trades: {oos_m.get('trades', 0)}")
        if oos_m.get('avg_cash_frac', 0) > 0.15:
            print(f"   (NOTE: averaged {oos_m['avg_cash_frac']*100:.0f}% of OOS portfolio value sitting "
                  f"in cash, {oos_m.get('pct_days_high_cash',0):.0f}% of days above "
                  f"{HIGH_CASH_FRACTION_THRESHOLD*100:.0f}% cash.)")
    else:
        print(f"Out-of-Sample Score: {oos_score:.4f} (strategy failed on OOS data)")
        print_gate_diagnosis(oos_m, oos_m.get('yearly', []), is_oos=True, label="OUT-OF-SAMPLE GATE DIAGNOSIS")

    print(f"\nRobustness Ratio: {robustness:.1%}  ", end="")
    if robustness >= 0.70:
        print("EXCELLENT (>70%) -- deploy with confidence")
    elif robustness >= ROBUSTNESS_DEPLOY_THRESHOLD:
        print(f"ACCEPTABLE ({ROBUSTNESS_DEPLOY_THRESHOLD:.0%}-70%) -- deploy cautiously   "
              f"<- meets this script's save threshold")
    elif robustness >= 0.30:
        print(f"POOR (30%-{ROBUSTNESS_DEPLOY_THRESHOLD:.0%}) -- likely overfit, do not deploy   "
              f"<- below this script's save threshold")
    elif robustness >= 0.0:
        print("REJECT (<30%) -- heavily overfit, discard")
    else:
        print(f"REJECT (OOS score negative, {robustness:.1%} of IS score's magnitude) -- "
              f"discard. See gate diagnosis above for what specifically broke.")

    # decision #14: does the champion's edge survive if the SIP had landed on
    # a different day each month? Run once here, on the IS window, since this
    # is the final report for the params actually being considered for save.
    if is_score > 0:
        temporal_ok, temporal_scores, temporal_median = passes_temporal_robustness_check(
            best_params, is_score, opens, closes, atr, adx,
            months, years_arr, eligible_mask, master_dates, 0, is_end
        )
        verdict = "PASSED" if temporal_ok else "FAILED"
        print(f"Temporal robustness (randomized SIP date): {verdict} -- if this fails, the "
              f"edge may partly be an artifact of always deploying capital at month-start "
              f"conditions rather than a genuine, calendar-independent signal.")

    return is_score, oos_score, oos_m, robustness, is_end_wealth


def report_yearly_table(best_params, opens, closes, atr, adx, months, years_arr,
                         eligible_mask, sip_flag):
    """Prints the strategy's own per-calendar-year IRR next to BTC's, with
    the recency weight and bull/bear tag applied that day -- the direct,
    eyeballable answer to 'did this rely on one lucky year'."""
    n_days = closes.shape[0]
    score, m = evaluate_params_crypto(best_params, opens, closes, atr, adx,
                                      months, years_arr, eligible_mask,
                                      start_day=0, end_day=n_days - 1, is_oos=False,
                                      starting_wealth=0.0, sip_flag=sip_flag)
    yearly = m.get('yearly', [])
    if not yearly:
        print("\n(No yearly breakdown available.)")
        return

    low_coverage_years = [y['year'] for y in yearly if y['coverage_frac'] < MIN_YEAR_COVERAGE_FOR_SCORING]
    if low_coverage_years:
        print(f"\n(Note: {low_coverage_years} excluded from the CONSISTENCY/REGIME score "
              f"terms -- coverage below {MIN_YEAR_COVERAGE_FOR_SCORING:.0%}, IRR too noisy "
              f"to trust. Still shown below for completeness.)")
    yrs = [y['year'] for y in yearly]
    min_year, max_year = min(yrs), max(yrs)
    total_profit = sum(y['nominal_profit'] for y in yearly)

    print("\n" + "=" * 78)
    print("YEAR-BY-YEAR BREAKDOWN (full-history champion, for eyeballing consistency)")
    print("=" * 78)
    print(f"{'Year':<6}{'Strategy IRR':>14}{'BTC IRR':>12}{'Regime':>8}{'RecencyW':>10}{'Profit Share':>14}")
    for y in yearly:
        regime = 'BULL' if y['bench_irr'] > BULL_YEAR_BTC_THRESHOLD else ('BEAR' if y['bench_irr'] < BEAR_YEAR_BTC_THRESHOLD else 'flat')
        w = _recency_weight(y['year'], min_year, max_year) * y['coverage_frac']
        share = (y['nominal_profit'] / total_profit * 100.0) if total_profit > 0 else 0.0
        print(f"{y['year']:<6}{y['port_irr']*100:>13.1f}%{y['bench_irr']*100:>11.1f}%{regime:>8}{w:>10.2f}{share:>13.1f}%")
    max_share = max((y['nominal_profit'] / total_profit) for y in yearly) if total_profit > 0 else 0.0
    print(f"\nMax single-year profit share: {max_share*100:.1f}%  "
          f"(hard gate rejects above {CONCENTRATION_GATE*100:.0f}%)")


def report_subperiod_breakdown_crypto(best_params, opens, closes, atr, adx,
                                       months, years_arr, eligible_mask, sip_flag,
                                       n_blocks=5):
    """Ported from the stock engine's report_subperiod_breakdown() --
    re-evaluates the already-found champion on n_blocks contiguous
    sub-windows of the FULL history (no re-optimization), so you can see
    whether it's consistent across regimes or just lived/died in one lucky
    stretch. Complements report_yearly_table (calendar-year granularity)
    with a fixed-block view that doesn't care about year boundaries."""
    n_days = closes.shape[0]
    edges = np.linspace(1, n_days - 1, n_blocks + 1).astype(int)

    print("\n" + "=" * 60)
    print(f"SUB-PERIOD CONSISTENCY CHECK ({n_blocks} blocks, same champion params)")
    print("=" * 60)
    rows = []
    for i in range(n_blocks):
        s_day, e_day = int(edges[i]), int(edges[i + 1])
        score, m = evaluate_params_crypto(
            best_params, opens, closes, atr, adx, months, years_arr, eligible_mask,
            start_day=s_day, end_day=e_day, is_oos=False, starting_wealth=0.0, sip_flag=sip_flag
        )
        rows.append((s_day, e_day, score, m))
        if score > -900:
            wt, lt = m.get('winning_trades', 0), m.get('losing_trades', 0)
            wr = wt / (wt + lt) * 100 if (wt + lt) > 0 else 0.0
            print(f"  Block {i+1} (days {s_day:5d}-{e_day:5d}, ~{(e_day-s_day)/365:.1f}y): "
                  f"Score {score:7.3f} | IRR {m.get('annual_return',0)*100:6.1f}% | "
                  f"Sharpe {m.get('sharpe',0):5.2f} | MaxDD {abs(m.get('max_dd',0))*100:5.1f}% | "
                  f"WinRate {wr:5.1f}% | Trades {m.get('trades',0)}")
        else:
            print(f"  Block {i+1} (days {s_day:5d}-{e_day:5d}): below minimum-trades/months gate "
                  f"for a block this short -- inconclusive, not a failure.")
    return rows


def run_overfitting_diagnostic(all_trial_sharpes_annualized, champion_metrics, T):
    if len(all_trial_sharpes_annualized) < 30:
        print("\n(Skipping overfitting diagnostic -- need at least ~30 completed trials.)")
        return None
    # NEW v1.3: use the champion's REAL skew/kurtosis (now computed inside
    # simulate_portfolio_crypto) with the same fallback logic the stock
    # engine's v5.0 uses -- a computed kurtosis of 0 means 'too few/uniform
    # daily returns to compute', not 'kurtosis is actually zero' (kurtosis is
    # bounded below by skew^2+1 for any real distribution), so only fall back
    # to the generic placeholder then. Previously this ALWAYS used the
    # hardcoded (-0.3, 5.0) placeholder, every run.
    kurt_computed = champion_metrics.get('kurtosis', 0.0)
    if kurt_computed > 0:
        skew = champion_metrics.get('skew', -0.3)
        kurt = kurt_computed
    else:
        skew = -0.3
        kurt = 5.0
    sr_hat_annual = champion_metrics.get('sharpe', 0.0)
    sr_hat_daily  = sr_hat_annual / math.sqrt(CRYPTO_DAYS_PER_YEAR)
    trial_sharpes_daily = [s / math.sqrt(CRYPTO_DAYS_PER_YEAR) for s in all_trial_sharpes_annualized]

    dsr, sr0_daily, n_trials = deflated_sharpe_ratio(sr_hat_daily, trial_sharpes_daily, T, skew, kurt)
    sr0_annual = sr0_daily * math.sqrt(CRYPTO_DAYS_PER_YEAR)

    print("\n" + "=" * 60)
    print("OVERFITTING DIAGNOSTIC 1/2 -- Deflated Sharpe Ratio")
    print("=" * 60)
    print(f"  Completed trials used:            {n_trials}")
    print(f"  Champion's annualized Sharpe:      {sr_hat_annual:.3f}")
    print(f"  Expected max Sharpe from {n_trials} noise trials: {sr0_annual:.3f}")
    print(f"  Deflated Sharpe Ratio (DSR, a probability): {dsr:.3f}")
    if dsr > 0.95:
        print("  -> Champion clears the noise threshold with high confidence.")
    elif dsr > 0.70:
        print("  -> Plausibly real edge, but not overwhelming. Treat cautiously.")
    else:
        print("  -> Cannot statistically distinguish this champion from noise. Weigh the OOS result heavily.")
    return dsr


def report_score_distribution_diagnostic(champion_score, all_trial_scores):
    if len(all_trial_scores) < 10:
        print("\n(Skipping score-distribution diagnostic -- fewer than 10 gate-passing trials.)")
        return None
    arr = np.array(all_trial_scores, dtype=np.float64)
    mean_s, std_s = float(arr.mean()), float(arr.std())
    pct_below = float((arr < champion_score).mean() * 100.0)
    z = (champion_score - mean_s) / std_s if std_s > 0 else float('inf')
    print("\n" + "=" * 60)
    print("OVERFITTING DIAGNOSTIC 2/2 -- Score-Distribution Sanity Check")
    print("=" * 60)
    print(f"  Other gate-passing trials: {len(arr)}")
    print(f"  Champion's composite score: {champion_score:.4f}")
    print(f"  Mean/std of other gate-passing scores: {mean_s:.4f} / {std_s:.4f}")
    print(f"  Champion beats {pct_below:.1f}% of other gate-passing trials ({z:.2f} std above their mean)")
    return {'mean': mean_s, 'std': std_s, 'pct_below': pct_below, 'z': z}


# ==========================================
# 11. CONSOLE REPORTER
# ==========================================

MA_TYPE_NAMES = {0: 'SMA', 1: 'EMA', 2: 'DEMA', 3: 'WMA', 4: 'SMMA'}
EXIT_METHOD_NAMES = {0: 'Hybrid ATR TP+Trail', 1: '%-Trail from High',
                     2: 'ATR-Trail from High', 3: 'Entry-MA Crossunder'}


def print_performance_report(score, metrics, p, label=""):
    print("\n" + "=" * 78)
    print(f"{label}")
    if score > -900:
        print(f"Composite Score: {score:.4f}")
    print("-" * 78)
    
    t_invested = metrics.get('t_invested', 0)
    start_w = metrics.get('starting_wealth', 0)
    wealth = metrics.get('wealth', 0)
    roi = metrics.get('roi', 0) * 100
    bench_roi = metrics.get('bench_roi', 0) * 100
    ann_ret = metrics.get('annual_return', 0) * 100
    bench_ann_ret = metrics.get('bench_annual_return', 0) * 100 
    
    sharpe = metrics.get('sharpe', 0)
    sortino = metrics.get('sortino', 0)
    skew = metrics.get('skew', 0)
    kurt = metrics.get('kurtosis', 0)
    ir = metrics.get('ir', 0)
    max_dd = abs(metrics.get('max_dd', 0)) * 100
    pf = metrics.get('pf', 0)
    
    trades = metrics.get('trades', 0)
    wt = metrics.get('winning_trades', 0)
    lt = metrics.get('losing_trades', 0)
    wr = (wt / trades) if trades > 0 else 0.0
    
    avg_bars = metrics.get('avg_bars', 0)
    avg_runup = metrics.get('avg_runup', 0)
    avg_loss = abs(metrics.get('avg_loss', 0))
    rr = avg_runup / avg_loss if avg_loss > 0 else 0.0
    
    ev = (wr * avg_runup) - ((1.0 - wr) * avg_loss)
    ev_in_r = ev / avg_loss if avg_loss > 0 else 0.0
    
    # 1. CAPITAL & RETURNS
    print("[CAPITAL & RETURNS]")
    print(f"Starting Capital Anchor:  ${start_w:,.0f}")
    print(f"Total SIP Injected:       ${t_invested:,.0f}")
    print(f"Final Portfolio Wealth:   ${wealth:,.0f}")
    print(f"System ROI (vs base):     {roi:.2f}%  (BTC B&H+DCA ROI: {bench_roi:.2f}%)")
    print(f"Excess Alpha (ROI):       {roi - bench_roi:+.2f}%")
    print(f"Annualized Return (IRR):  {ann_ret:.2f}%")
    print(f"BTC DCA Annualized (IRR): {bench_ann_ret:.2f}%   (apples-to-apples comparison)")
    print("-" * 78)
    
    # 2. RISK & PERFORMANCE
    print("[RISK & PERFORMANCE]")
    print(f"Sharpe Ratio:             {sharpe:.2f}")
    print(f"Sortino Ratio:            {sortino:.2f}  (capped to 4.0 in score)")
    print(f"Return Skew / Kurtosis:   {skew:.3f} / {kurt:.3f}")
    month_cnt = metrics.get('month_cnt', 1)
    ir_scaled = max(0.0, ir) * math.sqrt(max(1.0, month_cnt) / 12.0)
    print(f"Information Ratio:        {ir:.3f}  (scaled: {ir_scaled:.3f})")
    print(f"Max Portfolio Drawdown:   -{max_dd:.2f}%")
    print(f"Profit Factor:            {pf:.2f}")
    print("-" * 78)
    
    # 3. TRADE STATISTICS
    print("[TRADE STATISTICS]")
    print(f"Total Closed Trades:      {trades}")
    print(f"  Winning:                {wt}")
    print(f"  Losing:                 {lt}")
    print(f"Win Rate:                 {wr*100:.1f}%   <-- hard gate: must be >= {MIN_WIN_RATE_GATE*100:.0f}%")
    print(f"Avg Bars in Trade:        {avg_bars:.0f} days")
    print(f"Avg Win:                  +{avg_runup*100:.2f}%")
    print(f"Avg Loss:                 -{avg_loss*100:.2f}%")
    print(f"R:R Ratio:                {rr:.2f}x")
    print(f"Expected Value / R:       {ev_in_r:.3f}")
    print("-" * 78)
    
    # 4. SCORE BREAKDOWN
    calmar = (ann_ret/100) / (max_dd/100) if max_dd > 0 else 0.0
    print("[SCORE BREAKDOWN  (Calmar 20 / Sortino 10 / IR 20 / EV 10 / WR_Bonus 10 / Consistency 20 / Regime 10)]")
    print(f"Calmar (AnnRet/MaxDD):     {calmar:.3f}  x0.20")
    print(f"Sortino (cap 4.0):         {min(sortino, 4.0):.3f}  x0.10")
    print(f"IR scaled:                 {ir_scaled:.3f}  x0.20")
    print(f"EV in R:                   {ev_in_r:.3f}  x0.10")
    wr_bonus = max(0.0, wr - 0.50) * 4.0
    print(f"WR Bonus (WR-50)*4:        {wr_bonus:.3f}  x0.10")
    print(f"Consistency & Regime:      (Evaluated dynamically per-year, see breakdown table)")
    print("-" * 78)
    
    # 5. DIAGNOSTICS
    print("[DIAGNOSTICS]")
    print(f"Avg Cash Sitting Idle:     {metrics.get('avg_cash_frac', 0)*100:.1f}% of portfolio value")
    print(f"Days >30% in Cash:         {metrics.get('pct_days_high_cash', 0):.1f}% of this window")
    print(f"Floor-Clamp Triggers:      {metrics.get('n_floor_clamps', 0)}")
    print(f"Valid Return-Series Days:  {metrics.get('valid_days', 0)}")
    
    yearly = metrics.get('yearly', [])
    if yearly:
        total_profit = sum(y['nominal_profit'] for y in yearly)
        max_share = max((y['nominal_profit'] / total_profit) for y in yearly) if total_profit > 0 else 0.0
        print(f"Max single-year profit:    {max_share*100:.1f}% (gate rejects above {CONCENTRATION_GATE*100:.0f}%)")
    print("-" * 78)
    
    # 6. SIGNAL CONFIGURATION
    print("[SIGNAL CONFIGURATION]")
    signal_method = p.get('signal_method', 0)
    
    if signal_method == 1:
        print(f"Signal Model:              Pine dual-RSI crossover")
        print(f"Fast RSI:                  {p.get('rsi_fast_len')}/smt{p.get('rsi_fast_smt')}")
        print(f"Slow RSI:                  {p.get('rsi_slow_len')}/smt{p.get('rsi_slow_smt')}")
    else:
        print(f"Signal Model:              MA breakout")
        print(f"Entry MA:                  {p.get('entry_ma_len')} / {MA_TYPE_NAMES.get(p.get('entry_ma_type', 0), '?')}")
        
    print(f"Trend Filter:              {'Enabled (' + str(p.get('trend_ma_len')) + ' / ' + MA_TYPE_NAMES.get(p.get('trend_ma_type', 0), '?') + ')' if p.get('use_trend_ma', True) else 'Disabled'}")
    print(f"BTC Regime Filter:         {'Enabled (' + str(p.get('btc_ma_len')) + ' / ' + MA_TYPE_NAMES.get(p.get('btc_ma_type', 0), '?') + ')' if p.get('use_btc_filter', True) else 'Disabled'}")
    print(f"ADX Filter Threshold:      {p.get('adx_thresh', 0.0)}")
    
    exit_m = p.get('exit_method', 0)
    print(f"Exit Model Family:         {EXIT_METHOD_NAMES.get(exit_m, '?')}")
    if exit_m == 0:
        print(f"  -> Settings:             SL {p.get('sl_mult',0):.2f}x | TP {p.get('tp_mult',0):.2f}x | Trail {p.get('trail_mult',0):.2f}x")
    elif exit_m == 1:
        print(f"  -> Settings:             Trail {p.get('trail_pct',0):.2f}%")
    elif exit_m == 2:
        print(f"  -> Settings:             Exit ATR {p.get('exit_atr_mult',0):.2f}x")
    
    if p.get('use_panic_exits', True):
        overrides = []
        if p.get('use_btc_filter', True): overrides.append("BTC Bearish")
        if signal_method == 1: overrides.append("RSI Reversal")
        else: overrides.append(f"Price < Exit MA")
        print(f"Universal Panic Exits:     Enabled ({' OR '.join(overrides)})")
    else:
        print(f"Universal Panic Exits:     Disabled")
    print("-" * 78)
    
    # 7. COSTS & FIXED SETTINGS
    print("[COSTS & FIXED SETTINGS]")
    print(f"Buy/Sell cost:             {BUY_COST_PCT*100:.3f}% / {SELL_COST_PCT*100:.3f}%   Flat DP charge: $0.0   Idle cash yield: {ANNUAL_CASH_YIELD*100:.2f}%")
    print(f"Position Sizing:           Unified Cash Pool + Monthly SIP Pyramiding (Rank method: {p.get('wl_rank', 0)})")
    print(f"Execution Model:           Two-phase (signal @ close, fill @ next open)")
    print(f"Eligibility Mask:          Top {TOP_N_COINS} (survivorship-biased, disclosed)")
    print(f"Robustness Save Gate:      >= {ROBUSTNESS_DEPLOY_THRESHOLD*100:.0f}% OOS  AND  >= {TEMPORAL_ROBUSTNESS_PASS_FRAC*100:.0f}% temporal")
    print("=" * 78)
# ==========================================
# 12. OPTUNA SEARCH SPACE + MAIN OPTIMIZER
# ==========================================

def run_optimization():
    tickers = fetch_top100_universe()
    (opens, closes, atr, adx, months, years_arr,
     stock_names, eligible_mask, master_dates) = prepare_matrix_data_crypto(tickers)

    clear_ma_cache()
    clear_rsi_cache()

    # decision #14: the schedule used for the main search, WFO, and reports.
    # Always day-1-of-month, exactly reproducing pre-v1.2 behavior -- the
    # random-day schedule is generated separately, on demand, ONLY inside
    # passes_temporal_robustness_check().
    sip_flag_default = build_sip_schedule(master_dates, mode='month_start')

    n_days = closes.shape[0]
    is_end = int(n_days * WFO_IS_PCT)

    print(f"\nMatrix: {n_days} days x {closes.shape[1]} coins  (universe: top {TOP_N_COINS})")
    print(f"IS period:  days 0-{is_end}  ({is_end/365:.1f} yrs)")
    print(f"OOS period: days {is_end}-{n_days}  ({(n_days-is_end)/365:.1f} yrs -- never seen during opt)")

    # ==========================================
    # NEW v1.3: PINE SCRIPT REFERENCE STRATEGY -- run FIRST, every time the
    # engine loads, so you always see how your live TradingView strategy is
    # actually doing on the full backtest before anything else happens. This
    # is a REPORT ONLY at this point -- it does not touch BEST_PARAMS_FILE /
    # INTERMEDIATE_FILE by itself; whether it becomes the search's starting
    # baseline is decided further down, once the loaded-champion comparison
    # (below) has also run.
    # ==========================================
    if RUN_PINESCRIPT_REFERENCE:
        print("\n" + "#" * 78)
        print("# PINE SCRIPT REFERENCE STRATEGY -- 'Holy Grail Trend Hunter [Universal]'")
        print("# Your exact live TradingView params, run once on load so you can always")
        print("# see where the bot's own search stands relative to what you're actually")
        print("# trading right now.")
        print("#" * 78)
        pine_full_score, pine_full_metrics = evaluate_params_crypto(
            PINESCRIPT_REFERENCE_PARAMS, opens, closes, atr, adx, months, years_arr, eligible_mask,
            start_day=0, end_day=n_days - 1, is_oos=False, starting_wealth=0.0,
            sip_flag=sip_flag_default
        )
        if pine_full_score > -900:
            print_performance_report(pine_full_score, pine_full_metrics, PINESCRIPT_REFERENCE_PARAMS,
                                     "PINE SCRIPT REFERENCE -- LIFETIME FULL-DATA PERFORMANCE")
            report_yearly_table(PINESCRIPT_REFERENCE_PARAMS, opens, closes, atr, adx, months, years_arr,
                                eligible_mask, sip_flag_default)
        else:
            print("\nPine script reference strategy did NOT pass the hard quality gates on the "
                  "full dataset (this can legitimately happen -- e.g. not enough trades in this "
                  "universe/date range). Gate-by-gate detail below:")
            print_gate_diagnosis(pine_full_metrics, pine_full_metrics.get('yearly', []), is_oos=False,
                                 label="PINE SCRIPT REFERENCE -- GATE DIAGNOSIS")
        pine_is_score, _ = evaluate_params_crypto(
            PINESCRIPT_REFERENCE_PARAMS, opens, closes, atr, adx, months, years_arr, eligible_mask,
            start_day=0, end_day=is_end, is_oos=False, starting_wealth=0.0,
            sip_flag=sip_flag_default
        )
        if pine_is_score > -900:
            print("\nWalk-forward split for the Pine reference strategy itself (IS vs OOS, same "
                  "70/30 split the search uses) --")
            run_wfo_validation(PINESCRIPT_REFERENCE_PARAMS, opens, closes, atr, adx, months, years_arr,
                               eligible_mask, master_dates, sip_flag_default)
    else:
        print("\n(RUN_PINESCRIPT_REFERENCE = False -- skipping Pine script reference report, "
              "baseline seeding, and enqueued trial. Starting a completely fresh search.)")
        pine_full_score, pine_full_metrics = -999999.0, {}
        pine_is_score = -999999.0

    best_is_score = -999999.0
    best_params   = None
    all_trial_sharpes = []
    all_trial_scores  = []

    prev_oos, prev_is, prev_params = load_previous_winner(BEST_PARAMS_FILE)
    if prev_params is None:
        _, prev_is, prev_params = load_previous_winner(INTERMEDIATE_FILE)

    if prev_params is not None:
        print("\n" + "-" * 60)
        print("EVALUATING LOADED CHAMPION CONFIGURATION")
        print("-" * 60)
        score_is, metrics_is = evaluate_params_crypto(
            prev_params, opens, closes, atr, adx, months, years_arr, eligible_mask,
            start_day=0, end_day=is_end, is_oos=False, starting_wealth=0.0,
            sip_flag=sip_flag_default
        )
        score_full, metrics_full = evaluate_params_crypto(
            prev_params, opens, closes, atr, adx, months, years_arr, eligible_mask,
            start_day=0, end_day=n_days - 1, is_oos=False, starting_wealth=0.0,
            sip_flag=sip_flag_default
        )
        if score_is > -900:
            best_is_score = score_is
            best_params   = prev_params
            print(f"Loaded champion passes IS gates. Baseline: {best_is_score:.4f}")
            print_performance_report(score_full, metrics_full, prev_params,
                                     "LOADED CHAMPION (LIFETIME FULL-DATA PERFORMANCE)")
        else:
            print("Loaded champion no longer passes strict IS gates. Starting fresh.")
            print_gate_diagnosis(metrics_is, metrics_is.get('yearly', []), is_oos=False,
                                 label="WHY THE LOADED CHAMPION NO LONGER PASSES")

    # NEW v1.3: the Pine reference becomes the search's STARTING baseline
    # whenever it's stronger (on IS score) than whatever was loaded from disk
    # -- "run this exact Pine Script strategy as winner strategy" on load,
    # per your request, then let the search try to beat it from there. If a
    # previously-saved champion already beats the Pine reference on IS, that
    # stays the baseline instead (no regression just to force the Pine
    # params in) -- either way the Pine reference is ALWAYS enqueued as an
    # explicit trial below so TPE explores its neighborhood.
    if pine_is_score > -900 and pine_is_score > best_is_score:
        best_is_score = pine_is_score
        best_params   = dict(PINESCRIPT_REFERENCE_PARAMS)
        print(f"\nPine script reference is the strongest IS baseline available "
              f"({best_is_score:.4f}) -- search starts from it.")

    if OPTUNA_AVAILABLE:
        print(f"\nBayesian Optimization ({BAYESIAN_TRIALS} trials on IS data only, n_jobs={OPTUNA_N_JOBS})...")

        def optuna_objective(trial):
            use_trend_ma    = trial.suggest_categorical('use_trend_ma', [True, False])
            use_btc_filter  = trial.suggest_categorical('use_btc_filter', [True, False])
            use_panic_exits = trial.suggest_categorical('use_panic_exits', [True, False])
            signal_method   = trial.suggest_int('signal_method', 0, 1)
            exit_method     = trial.suggest_int('exit_method', 0, 3)

            p = {
                'signal_method':   signal_method,
                'use_trend_ma':    use_trend_ma,
                'use_btc_filter':  use_btc_filter,
                'use_panic_exits': use_panic_exits,
                
                # signal_method==0 params 
                'entry_ma_len':  trial.suggest_int('entry_ma_len', 10, 150),
                'entry_ma_type': trial.suggest_int('entry_ma_type', 0, 4),
                
                # signal_method==1 params 
                'rsi_fast_len':  trial.suggest_int('rsi_fast_len', 10, 60),
                'rsi_fast_smt':  trial.suggest_int('rsi_fast_smt', 5, 50),
                'rsi_slow_len':  trial.suggest_int('rsi_slow_len', 15, 80),
                'rsi_slow_smt':  trial.suggest_int('rsi_slow_smt', 10, 60),
                
                # Filters & Shared
                'trend_ma_len':  trial.suggest_int('trend_ma_len', 30, 150),
                'trend_ma_type': trial.suggest_int('trend_ma_type', 0, 4),
                'exit_ma_len':   trial.suggest_int('exit_ma_len', 10, 150),
                'exit_ma_type':  trial.suggest_int('exit_ma_type', 0, 4),
                'btc_ma_len':    trial.suggest_int('btc_ma_len', 10, 150),
                'btc_ma_type':   trial.suggest_int('btc_ma_type', 0, 4),
                
                'wl_rank':       trial.suggest_int('wl_rank', 0, 2),
                'adx_thresh':    trial.suggest_categorical('adx_thresh', [0.0, 15.0, 20.0, 25.0]),
                
                # Exit families
                'exit_method':   exit_method,
                'sl_mult':       trial.suggest_float('sl_mult', 1.5, 8.0),
                'tp_mult':       trial.suggest_float('tp_mult', 5.0, 70.0),
                'trail_mult':    trial.suggest_float('trail_mult', 2.0, 15.0),
                'trail_pct':     trial.suggest_float('trail_pct', 5.0, 35.0),
                'exit_atr_mult': trial.suggest_float('exit_atr_mult', 1.0, 6.0),
            }

            score, metrics = evaluate_params_crypto(p, opens, closes, atr, adx,
                                       months, years_arr, eligible_mask,
                                       start_day=0, end_day=is_end, is_oos=False,
                                       starting_wealth=0.0, sip_flag=sip_flag_default)
            if metrics:
                all_trial_sharpes.append(metrics.get('sharpe', 0.0))
                if score > -900:
                    all_trial_scores.append(score)
            return score if score > -900 else -999.0

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=300,
                                                multivariate=True, group=True)
        )
        # Always enqueue the exact Pine script params as a trial (regardless
        # of whether it ended up as best_params above) so TPE gets at least
        # one concrete, real-world-validated point to anchor its search near.
        if RUN_PINESCRIPT_REFERENCE:
            study.enqueue_trial({k: v for k, v in PINESCRIPT_REFERENCE_PARAMS.items()})
        if best_params is not None and best_params != PINESCRIPT_REFERENCE_PARAMS:
            study.enqueue_trial({k: v for k, v in best_params.items()})

        # NEW v1.3: protects best_is_score/best_params from a race between
        # concurrently-completing trials now that n_jobs > 1 is actually wired
        # up (ported from the stock engine's identical champion_lock).
        champion_lock = threading.Lock()

        def optuna_callback(study, trial):
            nonlocal best_is_score, best_params
            if trial.value is None or trial.value < -900: return
            with champion_lock:
                if trial.value <= best_is_score:
                    return
                p_full = dict(trial.params)
                if passes_neighborhood_check(
                    p_full, trial.value, opens, closes, atr, adx,
                    months, years_arr, eligible_mask, 0, is_end, sip_flag_default
                ):
                    temporal_ok, temporal_scores, temporal_median = passes_temporal_robustness_check(
                        p_full, trial.value, opens, closes, atr, adx,
                        months, years_arr, eligible_mask, master_dates, 0, is_end,
                        verbose=False
                    )
                    if not temporal_ok:
                        return   # edge looks timing-dependent (decision #14) -- not a genuine champion

                    best_is_score = trial.value
                    best_params   = p_full
                    score_chk, metrics = evaluate_params_crypto(
                        p_full, opens, closes, atr, adx, months, years_arr, eligible_mask,
                        0, is_end, is_oos=False, starting_wealth=0.0, sip_flag=sip_flag_default
                    )
                    print_performance_report(score_chk, metrics, p_full,
                                             f"NEW IS CHAMPION  (trial {trial.number})  "
                                             f"[temporal robustness: median {temporal_median:.4f} "
                                             f"vs base {trial.value:.4f}]")
                    save_winner(0.0, trial.value, p_full, filename=INTERMEDIATE_FILE)

        study.optimize(optuna_objective, n_trials=BAYESIAN_TRIALS,
                       callbacks=[optuna_callback], show_progress_bar=True,
                       n_jobs=OPTUNA_N_JOBS)

    if best_params is not None:
        is_score, oos_score, oos_metrics, robustness, is_end_wealth = run_wfo_validation(
            best_params, opens, closes, atr, adx, months, years_arr, eligible_mask,
            master_dates, sip_flag_default
        )

        report_yearly_table(best_params, opens, closes, atr, adx, months, years_arr,
                            eligible_mask, sip_flag_default)

        report_subperiod_breakdown_crypto(best_params, opens, closes, atr, adx,
                                          months, years_arr, eligible_mask, sip_flag_default)

        if oos_score > -900:
            run_overfitting_diagnostic(all_trial_sharpes, oos_metrics, T=int((n_days - is_end) * WFO_OOS_PCT))
            report_score_distribution_diagnostic(best_is_score, all_trial_scores)

        # NEW v1.3: temporal robustness is now an EXPLICIT final save gate,
        # not just something printed mid-WFO -- ported from the stock
        # engine's pattern (its run_temporal_robustness_check() gates
        # save_winner() directly; the crypto engine previously computed this
        # inside run_wfo_validation() but never actually required it to pass
        # before saving). Evaluated on the FULL dataset, same as the stock
        # engine's version -- a calendar-mechanics question, not a
        # market-regime one.
        temporal_final_ok = True
        temporal_final_median = None
        if oos_score > -900:
            full_score, _ = evaluate_params_crypto(
                best_params, opens, closes, atr, adx, months, years_arr, eligible_mask,
                start_day=0, end_day=n_days - 1, is_oos=False, starting_wealth=0.0,
                sip_flag=sip_flag_default
            )
            if full_score > 0:
                temporal_final_ok, _, temporal_final_median = passes_temporal_robustness_check(
                    best_params, full_score, opens, closes, atr, adx,
                    months, years_arr, eligible_mask, master_dates, 0, n_days - 1
                )

        if (oos_score > -900 and (oos_score > prev_oos or prev_oos <= 0)
                and robustness >= ROBUSTNESS_DEPLOY_THRESHOLD and temporal_final_ok):
            print("\nOOS + temporal-robustness validation passed. Saving fully verified champion...")
            save_winner(oos_score, is_score, best_params, filename=BEST_PARAMS_FILE)
            print_performance_report(oos_score, oos_metrics, best_params,
                                     "OUT-OF-SAMPLE VALIDATED CHAMPION")
        elif robustness < ROBUSTNESS_DEPLOY_THRESHOLD:
            print(f"\nStrategy rejected -- OOS robustness ({robustness:.1%}) below this "
                  f"script's {ROBUSTNESS_DEPLOY_THRESHOLD:.0%} save threshold.")
        elif not temporal_final_ok:
            print(f"\nStrategy rejected -- failed the final temporal (random-SIP-date) "
                  f"robustness check on the full dataset (median retained score "
                  f"{temporal_final_median:.4f} vs required {TEMPORAL_ROBUSTNESS_SCORE_MIN:.0%} "
                  f"of baseline across {TEMPORAL_ROBUSTNESS_PASS_FRAC:.0%} of runs). The edge may "
                  f"depend on first-of-month SIP timing rather than genuine signal quality.")
        else:
            print(f"\nNew IS champion found but OOS score ({oos_score:.4f}) "
                  f"did not beat previous champion OOS ({prev_oos:.4f}).")
    else:
        print("\nNo strategy found that passes all quality gates.")


if __name__ == "__main__":
    run_optimization()