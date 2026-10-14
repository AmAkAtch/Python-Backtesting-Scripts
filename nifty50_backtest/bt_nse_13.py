#!/usr/bin/env python3
"""
NSE EQUITY QUANTITATIVE SWING-TRADING FRAMEWORK (V13 - NO SLOT CAP, WEIGHT-RAMP SIZING, PLATEAU-AWARE SEARCH)
============================================================================================================
V13 (this file, bt_nse_13) - owner-directed redesign after the V12.4 runs (OOS placebo rank 0/52/71/15/49, none promoted):
  A. WATCHLIST: one entry per stock (V12.4 added a new entry every signal day); no age cap (an item leaves when its shadow position would
     have exited); fill-time revalidation (trend/ADX/macro) is an Optuna on/off choice; fills record wait_bars and gap_vs_trigger_pct.
  B. SIZING: max_concurrent_tranches is gone. Size = max(floor, equity x w); w constant within a day (even batches), ramps slowly while cash
     idles above a reserve, eases back while candidates wait for cash; no runts; whole shares only (unaffordable = one share above 2x target).
  C. ENTRIES: type 3 is a pure breakout (volume confirmation optional); type 5 is a buying-pressure surge (buy volume >= ratio x sell volume,
     estimated from where the close sits in the day's range; circuit days handled).
  D. ANTI-AVERAGING: up to avg_down_free_adds add-ons below the last entry price are allowed; up to 3 pyramid layers.
  E. LOTTERY GUARD measured in percent of equity at entry (rupee shares reward account growth, not luck).
  F. SEARCH: in-sample plateau penalty for top-decile trials (soft, capped, raw score stored); OOS is never used inside the search.
  G. SPEED/VISIBILITY: placebo seeds run in parallel processes (serial fallback); every slow validation phase logs its time.
  H. CLEANUP: legacy slot sizing and the V12.4 switch scaffolding removed; one definition of the fixed rules and engine defaults.

V12.4 - fixes found by reviewing the first REAL V12.3 run (11 of 14 gates, not promoted):
  P. WASTED-TRIAL DIAGNOSIS WAS WRONG (V11 to V12.3): "wasted" was counted as score <= -5, but only scores in [-10, -5) come from the trade-floor
     penalty; a genuinely poor strategy with plenty of trades also scores below -5. Real-data test: entry types 1 to 4 trade hundreds of times, only
     ~4% of random configs fail the floor, yet 99% scored <= -5. Each trial now records its closed-trade count (user attribute n_closed) and section 19
     separates 'below the trade floor' from 'traded enough but scored poorly'. Old studies without the attribute are labelled as 'cause not recorded'.
  Q. UNCLOSED-MARKS GATE MEASURED IN RETURN TERMS (owner delegated the decision): the rupee-share test (<15% of IS profit) failed all 8 shortlisted
     candidates because a compounding account earns most of its rupees just before the cut. The gate now asks: with open gains credited at only
     EOT_EXCESS_WEIGHT (30%), does the IS CAGR (median over the four cut dates) still beat the equal-weight survivor basket over the same window?
     The rupee share is still computed and printed (information). GATE_EOT_MODE = "share" restores the V12.3 test exactly.
  R. Cosmetic: failing-gate names in section 4 no longer carry trailing spaces.
  S. SEARCH BUDGET: N_TRIALS_PER_SEED 350 -> 500 (about 43% longer search). Convergence table of the real V12.3 run: seed 42 flat after 50% of its trials, seeds 1337
     and 2718 still rising in the last quarter. More trials only improve the in-sample search score; promotion still rests on the gates, not on the score.
V12.3 (bt_nse_12 before V12.4) - machinery fixes found by reviewing the first REAL V12.2 run (12 of 14 gates passed, not promoted):
  K. FULL GATES FOR RUNNERS-UP: the placebo and leave-top-5 gates only ever ran on the single champion, so a runner-up that clears
     every cheap gate was never tried. Now up to FULL_GATE_TOP_N candidates that clear ALL cheap gates get the full 14-gate test
     (placebo, leave-top-5 included) and the best fully-clean one is the champion. Candidates that already fail a cheap gate are
     skipped (they cannot be promoted anyway). Thresholds are NOT changed.
  L. PRE-SCREEN REPORTING: "13 passed" counted the three skipped gates as passes and never said WHICH gate failed. Section 4 now shows
     passed/evaluated and the failing gate names for every stage-2 candidate, plus which candidates got the full test.
  M. DRAWDOWN REALITY CHECK: the V12.2 champion had a fresh-start OOS drawdown of 45.6% against 15.2% for the index, and no gate saw
     it (the drawdown gate only reads the continuous run). V12.3 prints fresh-OOS drawdown vs index vs survivor basket under the gate
     table. An optional gate exists (GATE_FRESH_OOS_DD_VS_BASKET, default None = information only) for the owner to switch on.
  N. SEARCH DIAGNOSTICS: section 19 now shows the share of wasted (below trade-floor) trials by entry type, exit type, ADX, slots,
     holding period and floor, read straight from the stored trial values (no categorical-index decoding).
  O. REPORT: the 'Beats Conditioned Placebo Median' row now states it is weaker than the promotion gate; the walk-forward timing column is
     labelled as in-sample (the gate uses the full window, hence +9.67 vs +9.93); section 16 lists large moves in the last 10 sessions.
V12.2 (bt_nse_12 before V12.3) - fixes found by reviewing the first real V12.x runs (three champions, none promoted):
  G. FRESH-START OOS GATE: the V12.1 champion scored +47.8% in the continuous OOS slice (wealth and positions carried in from
     the in-sample bull run) but -2.2% when restarted cold on the OOS window, against +2.7% for the index. No gate saw that.
     Now a gate requires the cold-start OOS CAGR to beat NIFTYBEES, and the pre-screen runs it for every shortlisted candidate.
  H. UNCLOSED-MARKS GATE no longer depends on one arbitrary cut date: a long trend trade (CUPID, Oct 2023 to Mar 2024)
     straddled the IS/OOS cut in two runs. The gate now uses the MEDIAN share over the real cut and three earlier cuts.
  I. CORPORATE-ACTION RULE now requires the open gap and the close move to point the same way (ADANIENT 2015-06-03 had an open
     3x ABOVE the previous close and a close 39% BELOW it: a bad bar, not a split).
  J. REPORT: gate details are measured (neighbourhood pass-rate, timing p10, number of variants were hard-coded text);
     the calendar-year table now lists every year that the gates count (the partial first year was missing).
V12.1 - entry types 2 and 4 restored in the search; range check ignores label parameters; gate pre-screen for champion
  selection; concentration-by-symbol line; richer DATA_REPAIRS.csv.
V12.0 (file bt_nse_105) - fixes found by auditing the first real V11 run. The search machinery was misled by:
  A. A DEAD DRAWDOWN PENALTY: Yahoo printed NIFTYBEES at 1/10th price on 2019-12-19/20 (unit split), so the benchmark showed
     a -89.9% drawdown and the drawdown penalty was 0 for every trial. Benchmark beta/correlation were garbage too.
  B. UNADJUSTED CORPORATE ACTIONS (GPIL 5:1, TRENT, TMPV and VEDL demergers...) that look like -33% to -80% crashes with
     volume spikes: fake red candles for the volume exit, fake ADX spikes, fake stop-outs.
  C. A CAGR-ONLY OBJECTIVE that rewarded concentration: the winner held 4 positions, top-5 trades = 56% of profit, and random
     entries with the same exits scored almost as high. V12 scores each CALENDAR YEAR against the better of NIFTYBEES and the
     equal-weight basket of the same universe (kills survivorship credit), clips each year (kills lottery years), and
     penalises drawdown only beyond the benchmark's. It is the 'beat the benchmark year on year' goal, stated directly.
  D. GATES that were biased or blind: rupee-profit-by-year favoured late years of a compounding account (now log-return share);
     no gate on years beaten, on top-5-trade concentration, or on beating the survivor basket (now added).
  F. START STATE: initial capital is now Rs 1 lakh (your real scale) instead of Rs 5k, so floors/slots are judged in the regime you trade in.
  E. REPORT: brokerage was charged but not shown; missed-swing counted every day of a 'close > MA' state as a signal
     (now swing episodes); OOS score was hit by the 40-trade floor; no top-trades table; no search-health section.
Earlier changes (V11.0, file bt_nse_104):
 1. UNIVERSE: Nifty LargeMidcap 250 (= Nifty 100 + Midcap 150) pooled with Nifty Smallcap 100. Every symbol carries a
    bracket (LARGE/MID/SMALL) that sets its liquidity floor, slippage and impact. Smallcaps must clear a median daily
    traded-value floor so cash cannot freeze in a stock you cannot exit; upper brackets are not floored.
 2. LIQUIDITY: 30-day MEDIAN (not mean) traded value; missing history fails the test instead of passing at Rs 10 lakh.
 3. FEES: Upstox delivery brokerage (Rs 20 per executed order, both sides) + DP charges are now modelled.
    The V10.1 "ZERO_DELIVERY" mode under-charged every trade (about 1.0 pt per Rs 5k round trip).
 4. STOPS: decided on the close of bar t-1, filled at the OPEN of bar t (gaps included) - like every other exit.
    Breakeven stop sits at the true cost-covering price.
 5. TAKE-PROFIT BLOCK REMOVED. MA types reduced to SMA / EMA / WMA (RMA == EMA(2L-1); DEMA removed).
 6. SEARCH: 3 independent seeds x 350 trials; ONE plain objective (trimmed excess CAGR vs benchmark minus drawdown in
    excess of the benchmark's). No penalties inside the search. Ranges widened where V10.1 winners piled up at a limit
    (sl_mult, max_holding_bars, adx_thresh, macro_ma_len). Tranche floor {5k,10k,15k} is searched (real fees make it matter).
 7. ROBUSTNESS (applied afterwards to a diverse shortlist): full-parameter neighborhood incl. MA types and floor,
    deposit-day / start-date timing battery, seed agreement, 3 disjoint folds, 100-seed placebo, leave-top-5-out.
 8. PROMOTION GATES: winner.json / PASTE_INTO_LIVE.txt are written ONLY when every gate passes; otherwise the config is
    saved as winner_candidate.json / PASTE_INTO_LIVE_NOT_PROMOTED.txt and an earlier winner.json is left untouched.
 9. OOS is also reported as a slice of ONE continuous run (no restart, no forced liquidation at the split).
10. BENCHMARK: NIFTYBEES.NS (dividend-inclusive investable proxy). Regime filter still reads ^NSEI.
11. NEW REPORT SECTIONS: gates, deposit-timing table, missed-swing analysis, bracket breakdown, fee reality check,
    data-sanity (split/demerger artifacts, missing tickers). CASH_ANNUAL_YIELD stays 0 inside the objective on purpose.
12. STUDY HASH now covers the engine, fee/cost constants and universe signature, so old trials are never reused
    under new rules. Causality audit runs once per engine hash.
"""

from __future__ import annotations

import os
import io
import ast
import copy
import json
import time
import inspect
import hashlib
import logging
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field
import functools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import optuna

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ==============================================================================
# USER CONTROL PANEL
# ==============================================================================

ENGINE_VERSION: str = "V13.0"
# Universe: Nifty LargeMidcap 250 (= Nifty 100 + Nifty Midcap 150) pooled with Nifty Smallcap 100.
UNIVERSE_NAME: str = "POOL_LM250_SC100"

# (bracket, component index). Bracket drives liquidity floor + slippage + impact per symbol.
POOL_COMPONENTS: Dict[str, List[Tuple[str, str]]] = {
    "POOL_LM250_SC100": [("LARGE", "NIFTY100"), ("MID", "MIDCAP150"), ("SMALL", "SMALLCAP100")],
}
SINGLE_UNIVERSE_BRACKET: Dict[str, str] = {
    "NIFTY50": "LARGE", "NIFTY100": "LARGE", "MIDCAP150": "MID",
    "SMALLCAP100": "SMALL", "SMALLCAP250": "SMALL",
}

UNIVERSE_CONSTITUENT_FILES: Dict[str, str] = {
    "NIFTY50":     "constituents/ind_nifty50list.csv",
    "NIFTY100":    "constituents/ind_nifty100list.csv",
    "MIDCAP150":   "constituents/ind_niftymidcap150list.csv",
    "SMALLCAP100": "constituents/ind_niftysmallcap100list.csv",
    "SMALLCAP250": "constituents/ind_niftysmallcap250list.csv",
}

UNIVERSE_CONSTITUENT_URLS: Dict[str, List[str]] = {
    "NIFTY50": [
        "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
        "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    ],
    "NIFTY100": [
        "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
        "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
    ],
    "MIDCAP150": [
        "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
    ],
    "SMALLCAP100": [
        "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap100list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftysmallcap100list.csv",
    ],
    "SMALLCAP250": [
        "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
    ],
}

MACRO_INDEX_TICKER: str = "^NSEI"          # regime filter series (price index is fine for a trend filter)
BENCH_SYMBOL: str = "NIFTYBEES"            # benchmark series -> NIFTYBEES.NS (dividend-inclusive proxy)

FORCE_MACRO_REGIME_FILTER: Optional[bool] = None
FORCE_TRAIL_STOP: Optional[bool] = None  # Set True to lock trailing stops off (trail_atr_mult=0.0, trail_pct=0.0)

START_YEAR: int = 2015
INITIAL_CAPITAL: float = 100_000.0           # your real starting point (~Rs 1.2 lakh invested today). A Rs 5k cold start with a
                                            # Rs 10-15k tranche floor leaves the account in cash for months and unfairly punishes
                                            # high floors in the score. Set back to 5000.0 to replay a from-zero start.
MONTHLY_CONTRIBUTION: float = 5000.0
CASH_ANNUAL_YIELD: float = 0.0              # Deliberately 0: Optuna must not be rewarded for sitting in cash
MIN_HISTORY_DAYS: int = 300
WARMUP_BARS: int = 300

# --- Search budget: several independent short studies beat one long one (winner's-curse control)
SEEDS: List[int] = [42, 1337, 2718]
N_TRIALS_PER_SEED: int = 500                # V12.4: was 350; in the real V12.3 run seeds 1337 and 2718 were still improving in the last quarter (+0.61, +2.09). Studies resume, so this can be raised again (e.g. 700) without losing trials
N_STARTUP_TRIALS: int = 40
TOP_K_POSTHOC: int = 25                     # distinct top trials that get the full robustness battery

PARALLEL_DOWNLOAD_WORKERS: int = 8
FORCE_REFRESH: bool = False
CACHE_MAX_AGE_HOURS: float = 72.0
# ---- V13 engine rules ----
# V13_FIXED_RULES are decided by the owner and never searched. ENGINE_DEFAULTS fill in anything a config does not carry
# (the search records only what it samples, so reconstitute_params and the engine both start from these).
V13_FIXED_RULES: Dict[str, Any] = {
    "wl_dedupe": True,            # one watchlist entry per stock
    "wl_max_age_bars": 0,         # 0 = no age cap: an item leaves only when its shadow position would have exited
    "whole_shares": True,         # live buys whole shares; one share above 2x the target size is unaffordable
}
ENGINE_DEFAULTS: Dict[str, Any] = {
    **V13_FIXED_RULES,
    "wl_revalidate": True,        # re-check macro/ADX/trend state at fill time (searched)
    "avg_down_free_adds": 0,      # add-ons allowed below the last entry price before the anti-averaging guard bites (searched)
    "size_w0": 0.05, "size_w_max": 0.15, "size_reserve": 0.10, "size_ramp": 0.05,   # sizing (searched)
}

# ---- V13 sizing search choices (WEIGHT_RAMP). w = per-entry weight as a fraction of equity. ----
SIZE_W0_CHOICES: List[float] = [0.02, 0.03, 0.05, 0.08]        # starting weight; also the weight it eases back to
SIZE_W_MAX_CHOICES: List[float] = [0.10, 0.15, 0.25]           # ceiling the ramp may reach
SIZE_RESERVE_CHOICES: List[float] = [0.05, 0.10, 0.20]         # idle-cash share of equity tolerated before ramping up
SIZE_RAMP_CHOICES: List[float] = [0.02, 0.05, 0.10]            # daily multiplicative change of w
SIZE_RUNT_FRACTION: float = 0.6                                # an entry below this share of the target is not made
MAX_TRACKED_POSITIONS: int = 300                               # concurrency histogram length only (not a limit)

# --- Liquidity: median daily traded value over ADV_WINDOW sessions, per bracket.
#     Upper brackets need no floor; smallcaps must trade enough that cash never freezes in a position.
ADV_WINDOW: int = 30
ADV_MIN_PERIODS: int = 20
BRACKET_LIQUIDITY_FLOOR_INR: Dict[str, float] = {
    "LARGE": 0.0, "MID": 0.0, "SMALL": 10_000_000.0,   # Rs 1 crore median daily value for smallcaps
}
BRACKET_SLIPPAGE_BPS: Dict[str, float] = {"LARGE": 8.0, "MID": 15.0, "SMALL": 25.0}
BRACKET_IMPACT_COEF_BPS: Dict[str, float] = {"LARGE": 120.0, "MID": 200.0, "SMALL": 320.0}

TRANCHE_FLOOR_INR: float = 5_000.0          # default; Optuna picks from TRANCHE_FLOOR_CHOICES
TRANCHE_FLOOR_CHOICES: List[float] = [5_000.0, 10_000.0, 15_000.0]
MAX_POSITION_EQUITY_PCT: float = 0.25      # Hard cap on aggregate allocation per scrip at entry
MAX_ADV_PARTICIPATION: float = 0.015

NEIGHBORHOOD_DROP_LIMIT: float = 0.25
EOT_EXCESS_WEIGHT: float = 0.30  # Credit only a fraction of terminal-bar unrealized excess CAGR

# --- Objective: trimmed excess CAGR over the FULL-INVESTED benchmark, minus a drawdown penalty.
DD_PENALTY_WEIGHT: float = 0.25
MIN_CLOSED_TRADES: int = 40
# Objective V2 (annual, survivorship-neutral): for every calendar year in the window the strategy must beat the BETTER of
# (NIFTYBEES, equal-weight basket of the same universe). Each year's excess is clipped so that one lottery year or one
# lottery trade cannot dominate the score; drawdown is penalised only beyond the benchmark's own drawdown.
OBJ_YEAR_EXCESS_CLIP: float = 25.0          # percentage points; sensitivity table in the report re-scores at 15 / 25 / 40
OBJ_MIN_YEAR_BARS: int = 60                 # a (partial) calendar year needs this many sessions to count

# --- Promotion gates (all must pass for winner.json to be written)
GATE_MAX_DD_VS_INDEX: float = 1.25
GATE_PLACEBO_IS_PCT: float = 95.0
GATE_PLACEBO_OOS_PCT: float = 80.0
GATE_MAX_EOT_PROFIT_SHARE: float = 0.15
GATE_MAX_YEAR_PROFIT_SHARE: float = 0.40
GATE_MIN_SEED_AGREEMENT: int = 2
GATE_MIN_YEARS_BEATING: float = 0.70        # share of calendar years the strategy must beat NIFTYBEES (continuous run)
GATE_MAX_TOP5_TRADE_SHARE: float = 0.50     # lottery guard (measured in % of equity at entry, not rupees); threshold to be recalibrated on real V13 runs
GATE_FRESH_OOS_MUST_BEAT_INDEX: bool = True # V12.2: cold-start OOS CAGR must beat NIFTYBEES (the continuous slice alone is flattering)
GATE_EOT_MODE: str = "return"              # V12.4: "return" = organic IS CAGR (open gains at EOT_EXCESS_WEIGHT credit) must beat the survivor basket; "share" = V12.3 rupee-share test
FULL_GATE_TOP_N: int = 3                    # V12.3: how many cheap-gate-clean candidates also get the placebo / leave-5 gates before the champion is chosen (0 = off)
GATE_FRESH_OOS_DD_VS_BASKET: Optional[float] = None   # V12.3: if set (e.g. 1.5), fresh-start OOS max DD must be <= this multiple of the survivor basket's. None = information line only
EOT_CUT_BACKSHIFTS: Tuple[int, ...] = (0, 42, 84, 126)   # V12.2: IS window is also evaluated ending this many bars EARLIER; gate uses the median

# --- Data integrity (applied to the benchmark, the regime index and every constituent before anything is computed)
SPIKE_MIN_MOVE: float = 0.40                # one-bar move that starts a spike-and-revert check
SPIKE_REVERT_WITHIN: int = 5                # bars within which the price must return to its pre-spike level
SPIKE_REVERT_TOL: float = 0.15
CA_DROP_Q: float = 0.70                     # close/prev-close at or below this, gap-explained, on a calm market day => split/bonus/demerger
CA_JUMP_Q: float = 1.90                     # same for consolidation-like jumps (NSE circuits make genuine +90% days near-impossible)
CA_GAP_EXPLAINS: float = 0.80               # the OPEN must already sit at the new level (>= 80% of the move)
CA_MARKET_DAY_MAX: float = 0.06             # a day when the regime index itself moved more than this is never treated as a corporate action
LARGE_MOVE_LOG: float = 0.30                # moves this big that are NOT rebased are still listed for manual review
MANUAL_CORPORATE_ACTIONS: Dict[str, List[str]] = {}   # force-rebase: {"SYMBOL": ["YYYY-MM-DD"]}
KEEP_AS_REAL_MOVES: Dict[str, List[str]] = {}          # veto auto-rebase of a GENUINE crash/jump: {"SYMBOL": ["YYYY-MM-DD"]}
DATA_REPAIR_LOG: List[Dict[str, Any]] = []
PLACEBO_SEEDS_CONDITIONED: int = 100
PLACEBO_SEEDS_UNCONDITIONED: int = 30
# ---- In-loop robustness (in-sample ONLY: an OOS check inside the search would turn OOS into training data) ----
PLATEAU_IN_LOOP: bool = True                 # penalise fragile peaks while searching so TPE drifts toward plateaus
PLATEAU_TOP_QUANTILE: float = 0.90           # only trials whose raw score reaches this quantile of the study so far are checked
PLATEAU_MIN_COMPLETED: int = 40              # no checks until this many trials are complete
PLATEAU_MAX_NEIGHBORS: int = 12              # seeded sample of the +/-1 step neighbours (cost control; post-hoc test uses all)
PLATEAU_PENALTY_WEIGHT: float = 0.5          # score -= weight * max(0, raw - plateau)
PLATEAU_PENALTY_CAP: float = 0.5             # ... but never more than this share of |raw|; raw score is always stored
PLACEBO_WORKERS: int = 4                     # parallel processes for placebo seeds; 1 = serial. Falls back to serial on any failure.

# --- Installment-timing / start-date robustness (contribution lands on trading day N of each month)
TIMING_CONTRIB_OFFSETS: List[int] = [0, 5, 10, 15, 20]
TIMING_START_SHIFTS: List[int] = [0, 21, 42]

# Statutory Charges - Upstox equity delivery (verify DP + exchange rate against your contract note)
BROKERAGE_MODE: str = "FLAT"                # Upstox: Rs 20 per executed delivery order, both sides
FLAT_BROKERAGE_INR: float = 20.0
STT_RATE: float = 0.0010                    # 0.1% buy + 0.1% sell on delivery
EXCHANGE_TXN_RATE: float = 0.0000345        # NSE cash approx (Upstox calculator)
STAMP_DUTY_RATE: float = 0.00015            # 0.015% buy side
SEBI_CHARGE_RATE: float = 0.000001          # Rs 10 / crore
GST_RATE: float = 0.18
DP_CHARGE_INR: float = 20.0                 # per scrip per sell day (Upstox 16.5 + CDSL 3.5); GST added below
DP_CHARGE_GST: float = DP_CHARGE_INR * GST_RATE

EFFECTIVE_BUY_FEE_RATE: float = (
    STT_RATE + EXCHANGE_TXN_RATE + STAMP_DUTY_RATE + SEBI_CHARGE_RATE
    + GST_RATE * (EXCHANGE_TXN_RATE + SEBI_CHARGE_RATE)
)
EFFECTIVE_SELL_FEE_RATE: float = (
    STT_RATE + EXCHANGE_TXN_RATE + SEBI_CHARGE_RATE
    + GST_RATE * (EXCHANGE_TXN_RATE + SEBI_CHARGE_RATE)
)
FLAT_SELL_FIXED_INR: float = (
    (FLAT_BROKERAGE_INR * (1.0 + GST_RATE) if BROKERAGE_MODE == "FLAT" else 0.0)
    + DP_CHARGE_INR + DP_CHARGE_GST
)

DATA_DIR = Path("data_cache")
OUTPUT_DIR = Path("output") / UNIVERSE_NAME
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_WINNER_FILE = OUTPUT_DIR / "winner_candidate.json"
PASTE_FILE = OUTPUT_DIR / "PASTE_INTO_LIVE.txt"
SANITY_CSV_FILE = OUTPUT_DIR / "DATA_SANITY.csv"
REPAIRS_CSV_FILE = OUTPUT_DIR / "DATA_REPAIRS.csv"
CONT_TRADES_FILE = OUTPUT_DIR / "continuous_trades.csv"
AUDIT_STAMP_FILE = OUTPUT_DIR / "causality_audit.ok"
FINAL_WINNER_FILE = OUTPUT_DIR / "winner.json"
REPORT_FILE = OUTPUT_DIR / "WINNER_REPORT.md"
TRADES_CSV_FILE = OUTPUT_DIR / "winner_trades.csv"
STUDY_DB = f"sqlite:///{OUTPUT_DIR.resolve() / 'nse_swing_study.db'}"

STRESS_WINDOWS: Dict[str, Tuple[str, str]] = {
    "2011 Debt Crisis":          ("2011-08-01", "2011-12-31"),
    "2015-16 Commodity Plunge":  ("2015-08-01", "2016-02-29"),
    "2018 IL&FS Crash":          ("2018-09-01", "2018-12-31"),
    "2020 COVID Crash":          ("2020-01-01", "2020-06-30"),
    "2022 Inflation/War":        ("2022-01-01", "2022-06-30"),
}


def format_price(px: float) -> str:
    """Signed INR currency formatter."""
    if not np.isfinite(px):
        return "₹0.00"
    sign = "-" if px < 0 else ""
    return f"{sign}₹{abs(px):,.2f}"


def shift_1d(arr: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    """Non-wrapping causal 1D array shift."""
    res = np.empty_like(arr)
    res[0] = fill_value
    res[1:] = arr[:-1]
    return res


# ==============================================================================
# FAST NUMPY INDICATOR KERNELS
# ==============================================================================
def np_rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    w = max(1, int(window))
    out = np.full_like(arr, np.nan, dtype=np.float64)
    if len(arr) < w:
        return out
    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return out
    first_valid = int(np.argmax(valid_mask))
    clean = np.where(valid_mask, arr, 0.0)
    cumsum = np.cumsum(clean)
    cumsum = np.insert(cumsum, 0, 0.0)
    vals = (cumsum[w:] - cumsum[:-w]) / float(w)
    out[w - 1:] = vals
    out[:first_valid + w - 1] = np.nan
    return out


def np_rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    w = max(1, int(window))
    out = np.full_like(arr, np.nan, dtype=np.float64)
    n = len(arr)
    if n < w:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return out
    first_valid = int(np.argmax(valid_mask))
    windows = sliding_window_view(arr, window_shape=w)
    out[w - 1:] = np.max(windows, axis=-1)
    out[:first_valid + w - 1] = np.nan
    return out


def np_rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    w = max(1, int(window))
    out = np.full_like(arr, np.nan, dtype=np.float64)
    if len(arr) < w:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return out
    first_valid = int(np.argmax(valid_mask))
    windows = sliding_window_view(arr, window_shape=w)
    out[w - 1:] = np.std(windows, axis=-1, ddof=0)
    out[:first_valid + w - 1] = np.nan
    return out


def np_rolling_median(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    return pd.Series(arr).rolling(window, min_periods=min_periods).median().values.astype(np.float64)


def np_ewm_mean(arr: np.ndarray, span: int) -> np.ndarray:
    span = max(1, int(span))
    alpha = 2.0 / (span + 1.0)
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return out
    first_valid = int(np.argmax(valid_mask))
    out[first_valid] = arr[first_valid]
    for i in range(first_valid + 1, n):
        val = arr[i]
        if np.isnan(val):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * val + (1.0 - alpha) * out[i - 1]
    return out


# ==============================================================================
# STATUTORY FEE ACCOUNTING ENGINE
# ==============================================================================
@dataclass
class FeeBreakdown:
    brokerage: float = 0.0
    stt: float = 0.0
    exchange_charges: float = 0.0
    stamp_duty: float = 0.0
    sebi_charges: float = 0.0
    gst: float = 0.0
    dp_charges: float = 0.0
    slippage_cost: float = 0.0


def compute_buy_cost_audited(gross_inr: float, slip_cost: float) -> Tuple[float, FeeBreakdown]:
    if gross_inr <= 0:
        return 0.0, FeeBreakdown()
    brokerage = FLAT_BROKERAGE_INR if BROKERAGE_MODE == "FLAT" else 0.0
    stt = gross_inr * STT_RATE
    exch = gross_inr * EXCHANGE_TXN_RATE
    stamp = gross_inr * STAMP_DUTY_RATE
    sebi = gross_inr * SEBI_CHARGE_RATE
    gst = (brokerage + exch + sebi) * GST_RATE
    breakdown = FeeBreakdown(
        brokerage=brokerage, stt=stt, exchange_charges=exch,
        stamp_duty=stamp, sebi_charges=sebi, gst=gst,
        dp_charges=0.0, slippage_cost=slip_cost
    )
    return gross_inr + brokerage + stt + exch + stamp + sebi + gst, breakdown


def compute_sell_proceeds_audited(gross_inr: float, slip_cost: float, apply_dp: bool = True) -> Tuple[float, FeeBreakdown]:
    if gross_inr <= 0:
        return 0.0, FeeBreakdown()
    brokerage = FLAT_BROKERAGE_INR if BROKERAGE_MODE == "FLAT" else 0.0
    stt = gross_inr * STT_RATE
    exch = gross_inr * EXCHANGE_TXN_RATE
    sebi = gross_inr * SEBI_CHARGE_RATE
    gst = (brokerage + exch + sebi) * GST_RATE
    dp = (DP_CHARGE_INR + DP_CHARGE_GST) if apply_dp else 0.0
    breakdown = FeeBreakdown(
        brokerage=brokerage, stt=stt, exchange_charges=exch,
        stamp_duty=0.0, sebi_charges=sebi, gst=gst,
        dp_charges=dp, slippage_cost=slip_cost
    )
    proceeds = max(0.0, gross_inr - brokerage - stt - exch - sebi - gst - dp)
    return proceeds, breakdown


def compute_max_affordable_tranche(cash: float, adv_30d: float, base_bps: float, impact_bps: float) -> float:
    clean_adv = adv_30d if np.isfinite(adv_30d) else 1_000_000.0
    adv = max(clean_adv, 1_000_000.0)
    flat_buy_charge = (FLAT_BROKERAGE_INR * (1.0 + GST_RATE)) if BROKERAGE_MODE == "FLAT" else 0.0
    usable_cash = max(0.0, cash - flat_buy_charge)
    tranche_guess = usable_cash / (1.0 + EFFECTIVE_BUY_FEE_RATE + base_bps / 10000.0)
    for _ in range(3):
        part_rate = min(1.0, max(0.0, tranche_guess / adv))
        slip_mult = (base_bps + impact_bps * np.sqrt(part_rate)) / 10000.0
        tranche_guess = usable_cash / (1.0 + EFFECTIVE_BUY_FEE_RATE + slip_mult)
    return float(np.nan_to_num(tranche_guess * (1.0 - 1e-6), nan=0.0))


def breakeven_exit_price(total_cost: float, units: float, slip_mult: float) -> float:
    """Exit price at which net sell proceeds (after slippage, STT, exchange, GST, brokerage, DP) == total buy cost."""
    denom = units * max(1e-6, (1.0 - slip_mult - EFFECTIVE_SELL_FEE_RATE))
    return (total_cost + FLAT_SELL_FIXED_INR) / max(1e-9, denom)


def _buy_fill_audited(tranche_inr: float, ref_open_price: float, slip_mult: float,
                      whole_shares: bool = False) -> Tuple[float, float, float, FeeBreakdown]:
    fill_px = ref_open_price * (1.0 + slip_mult)
    units = tranche_inr / ref_open_price if ref_open_price > 0 else 0.0
    if whole_shares:
        # Live buys whole shares. If not even one share fits the target, allow ONE share only when it costs
        # at most twice the target; otherwise the entry is unaffordable (units = 0).
        units = float(np.floor(units))
        if units < 1.0 and ref_open_price > 0 and ref_open_price <= 2.0 * tranche_inr:
            units = 1.0
    gross_at_slip = units * fill_px
    slip_cost = units * (fill_px - ref_open_price)
    total_cash_cost, fees = compute_buy_cost_audited(gross_at_slip, slip_cost)
    return fill_px, units, total_cash_cost, fees


def _sell_fill_audited(units: float, ref_price: float, slip_mult: float, apply_dp: bool = True) -> Tuple[float, float, FeeBreakdown]:
    fill_px = ref_price * (1.0 - slip_mult)
    gross = units * fill_px
    slip_cost = units * (ref_price - fill_px)
    net_proceeds, fees = compute_sell_proceeds_audited(gross, slip_cost, apply_dp=apply_dp)
    return fill_px, net_proceeds, fees


# ==============================================================================
# UNIVERSE & REAL TRADING CALENDAR
# ==============================================================================
def load_universe_constituents(universe_name: str) -> List[str]:
    path = Path(UNIVERSE_CONSTITUENT_FILES[universe_name])
    if path.exists():
        df = pd.read_csv(path)
        sym_col = next((c for c in ("Symbol", "SYMBOL", "symbol") if c in df.columns), None)
        if sym_col:
            return sorted({str(s).strip().upper() for s in df[sym_col].tolist() if str(s).strip()})

    path.parent.mkdir(parents=True, exist_ok=True)
    urls = UNIVERSE_CONSTITUENT_URLS.get(universe_name, [])
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
    df = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read()
                path.write_bytes(content)
                df = pd.read_csv(io.BytesIO(content))
                break
        except Exception as e:
            logging.warning(f"Mirror failed for {url}: {e}")

    if (df is None or df.empty) and universe_name == "NIFTY50":
        try:
            wiki_tables = pd.read_html("https://en.wikipedia.org/wiki/NIFTY_50")
            for table in wiki_tables:
                sym_col = next((c for c in ("Symbol", "SYMBOL", "symbol") if c in table.columns), None)
                if sym_col:
                    symbols = sorted({str(s).strip().upper() for s in table[sym_col].tolist() if str(s).strip()})
                    pd.DataFrame({"Symbol": symbols}).to_csv(path, index=False)
                    return symbols
        except Exception as e:
            logging.warning(f"Wikipedia fallback failed: {e}")

    if df is None or df.empty:
        raise RuntimeError(f"Could not load constituent list for {universe_name}.")
    sym_col = next((c for c in ("Symbol", "SYMBOL", "symbol") if c in df.columns), None)
    return sorted({str(s).strip().upper() for s in df[sym_col].tolist() if str(s).strip()})


def load_universe_with_brackets(universe_name: str) -> Tuple[List[str], Dict[str, str]]:
    comps = POOL_COMPONENTS.get(universe_name)
    if comps is None:
        comps = [(SINGLE_UNIVERSE_BRACKET.get(universe_name, "LARGE"), universe_name)]
    order = {"LARGE": 0, "MID": 1, "SMALL": 2}
    bracket_map: Dict[str, str] = {}
    for bracket, comp in comps:
        for sym in load_universe_constituents(comp):
            if sym not in bracket_map or order[bracket] < order[bracket_map[sym]]:
                bracket_map[sym] = bracket
    return sorted(bracket_map), bracket_map


def _yf_ticker(symbol: str) -> str:
    return symbol if symbol.startswith("^") else f"{symbol}.NS"


def fetch_from_yfinance(symbol: str, start_year: int) -> Optional[pd.DataFrame]:
    ticker = _yf_ticker(symbol)
    start = f"{start_year}-01-01"
    try:
        raw = yf.download(ticker, start=start, progress=False, auto_adjust=True, threads=False)
    except Exception as e:
        logging.warning(f"yfinance download failed for {ticker}: {e}")
        return None
    if raw is None or raw.empty:
        return None
    raw = raw.reset_index()
    raw.columns = [c if isinstance(c, str) else c[0] for c in raw.columns]
    raw = raw.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    if not {"date", "open", "high", "low", "close", "volume"}.issubset(raw.columns):
        return None
    raw["quote_volume"] = raw["close"] * raw["volume"]
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None).dt.normalize()
    raw = raw.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    today = pd.Timestamp.now().normalize()
    raw = raw[raw["date"] < today]
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["open", "high", "low", "close"])
    # Reject invalid ticks and extreme split/data artifacts before indicators/ADV are built.
    raw = raw[
        (raw["open"] > 0) &
        (raw["close"] > 0) &
        (raw["high"] >= raw["low"]) &
        (raw["high"] / np.maximum(1e-8, raw["low"]) < 50.0)
    ]
    if len(raw) < MIN_HISTORY_DAYS:
        return None
    return raw.set_index("date")


def fetch_single_stock(symbol: str, start_year: int, refresh: bool) -> Tuple[str, Optional[pd.DataFrame], str]:
    safe_name = symbol.replace("^", "IDX_")
    cache_file = DATA_DIR / f"{safe_name}_1d_from{start_year}.parquet"
    meta_file = DATA_DIR / f"{safe_name}_1d_from{start_year}.meta.json"

    if not refresh and cache_file.exists() and meta_file.exists():
        age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600.0
        if age_hours < CACHE_MAX_AGE_HOURS:
            try:
                with open(meta_file, "r") as mf:
                    meta = json.load(mf)
                if meta.get("start_year") == start_year:
                    df = pd.read_parquet(cache_file)
                    if len(df) >= MIN_HISTORY_DAYS:
                        return symbol, df, "cache"
            except Exception:
                pass

    df = fetch_from_yfinance(symbol, start_year)
    if df is not None and len(df) >= MIN_HISTORY_DAYS:
        try:
            df.to_parquet(cache_file)
            with open(meta_file, "w") as mf:
                json.dump({"provider": "yfinance", "timestamp": time.time(), "bars": len(df), "start_year": start_year}, mf)
        except Exception:
            pass
        return symbol, df, "yfinance"
    return symbol, None, "none"


# ==============================================================================
# STATIC MARKET GRID & INDICATOR ENGINE
# ==============================================================================
@dataclass
class MarketGrid:
    symbols: List[str]
    dates: pd.DatetimeIndex
    open_mat: np.ndarray
    high_mat: np.ndarray
    low_mat: np.ndarray
    close_mat: np.ndarray
    volume_mat: np.ndarray
    atr14_mat: np.ndarray
    dvol30_mat: np.ndarray
    adx14_mat: np.ndarray
    delist_mat: np.ndarray
    alive_mat: np.ndarray
    macro_close: np.ndarray
    macro_open: np.ndarray
    months_arr: np.ndarray
    years_arr: np.ndarray
    bench_close: np.ndarray
    brackets: List[str]
    liq_floor: np.ndarray
    slip_bps: np.ndarray
    impact_bps: np.ndarray


class FastIndicators:
    @staticmethod
    def moving_average(arr: np.ndarray, length: int, kind: int) -> np.ndarray:
        # Supported kinds: 0 = SMA, 1 = EMA, 3 = WMA. (DEMA=2 and RMA=4 were removed in V11:
        # RMA(L) is mathematically EMA(2L-1); DEMA showed no measurable edge.)
        length = max(2, int(length))
        if kind == 0:   return np_rolling_mean(arr, length)
        elif kind == 1: return np_ewm_mean(arr, length)
        elif kind == 3:
            valid_mask = ~np.isnan(arr)
            out = np.full_like(arr, np.nan, dtype=np.float64)
            if not np.any(valid_mask):
                return out
            first_valid = int(np.argmax(valid_mask))
            n_valid = len(arr) - first_valid
            if n_valid < length:
                return out
            w = np.arange(1, length + 1, dtype=float)
            w_norm = w / w.sum()
            clean_tail = arr[first_valid:]
            conv = np.convolve(clean_tail, w_norm[::-1], mode='full')[:n_valid]
            conv[:length - 1] = np.nan
            out[first_valid:] = conv
            return out
        raise ValueError(f"Unsupported moving-average kind {kind} (allowed: 0=SMA, 1=EMA, 3=WMA)")

    @staticmethod
    def atr_1d(h: np.ndarray, l: np.ndarray, c: np.ndarray, length: int = 14) -> np.ndarray:
        cp = shift_1d(c, fill_value=c[0])
        tr1 = h - l
        tr2 = np.abs(h - cp)
        tr3 = np.abs(l - cp)
        tr = np.fmax(tr1, np.fmax(tr2, tr3))
        alpha = 1.0 / max(2, length)
        span = int(round((2.0 / alpha) - 1.0))
        return np_ewm_mean(tr, span)

    @staticmethod
    def adx_1d(h: np.ndarray, l: np.ndarray, c: np.ndarray, length: int = 14) -> np.ndarray:
        length = max(2, int(length))
        span = int(round((2.0 * length) - 1.0))
        up = h - shift_1d(h, fill_value=h[0])
        down = shift_1d(l, fill_value=l[0]) - l
        p_dm = np.where((up > down) & (up > 0.0), up, 0.0)
        m_dm = np.where((down > up) & (down > 0.0), down, 0.0)
        atr_arr = FastIndicators.atr_1d(h, l, c, length)
        p_di = 100.0 * np_ewm_mean(p_dm, span) / (atr_arr + 1e-9)
        m_di = 100.0 * np_ewm_mean(m_dm, span) / (atr_arr + 1e-9)
        dx = 100.0 * np.abs(p_di - m_di) / (p_di + m_di + 1e-9)
        return np_ewm_mean(dx, span)

    @staticmethod
    def rsi_smoothed(c: np.ndarray, length: int, smooth: int) -> np.ndarray:
        length, smooth = max(2, int(length)), max(1, int(smooth))
        delta = np.diff(c, prepend=c[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        span = int(round((2.0 * length) - 1.0))
        avg_gain = np_ewm_mean(gain, span)
        avg_loss = np_ewm_mean(loss, span)
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return np_rolling_mean(rsi, smooth)

    @staticmethod
    def bollinger_bands(c: np.ndarray, length: int, std_mult: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        length = max(2, int(length))
        mid = np_rolling_mean(c, length)
        std = np_rolling_std(c, length)
        return mid + (std_mult * std), mid, mid - (std_mult * std)


def repair_price_artifacts(df: pd.DataFrame, sym: str, mkt_ret: Optional[pd.Series],
                           allow_corporate: bool = True) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Yahoo returns two kinds of corrupt history that a trading engine will happily exploit:
      1. SPIKE-AND-REVERT bad prints (e.g. NIFTYBEES 2019-12-19/20 printed at 1/10th of its price for two days, which
         created a fake -89.9% drawdown and wrecked every beta/correlation statistic).
         -> the bad bars are rescaled back onto the surrounding price level.
      2. UNADJUSTED corporate actions (splits, bonuses, demergers: GPIL 5:1, TRENT, TMPV, VEDL). Each one is a fake
         -33% to -80% day plus a fake volume spike (a "high-volume red candle" exit / volume-breakout trigger).
         -> history BEFORE the event is rebased (prices x ratio, volume / ratio) so the event day becomes a normal day.
    Only drops <= -30% or jumps >= +90% whose OPEN already sits at the new level, on a calm index day, are rebased
    (NSE circuit limits make genuine moves of that size almost impossible). Anything else >= 30% is only LOGGED.
    """
    recs: List[Dict[str, Any]] = []
    if df is None or len(df) < 10:
        return df, recs
    d = df.copy()
    o = d["open"].values.astype(np.float64).copy()
    h = d["high"].values.astype(np.float64).copy()
    l = d["low"].values.astype(np.float64).copy()
    c = d["close"].values.astype(np.float64).copy()
    v = d["volume"].values.astype(np.float64).copy() if "volume" in d.columns else np.zeros(len(d))
    n = len(c)
    idx = d.index

    def _date(i: int) -> str:
        return str(pd.Timestamp(idx[i]).date())

    # 1) spike-and-revert
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.r_[1.0, c[1:] / np.where(c[:-1] > 0, c[:-1], np.nan)]
    cand = np.flatnonzero(np.abs(np.nan_to_num(q, nan=1.0) - 1.0) >= SPIKE_MIN_MOVE)
    skip_until = -1
    for i in cand:
        if i <= skip_until or i < 1 or i >= n - 1 or not np.isfinite(c[i - 1]) or c[i - 1] <= 0:
            continue
        for j in range(i, min(n - 1, i + SPIKE_REVERT_WITHIN)):
            if np.isfinite(c[j + 1]) and abs(c[j + 1] / c[i - 1] - 1.0) <= SPIKE_REVERT_TOL:
                f = c[i - 1] / c[i]
                sl = slice(i, j + 1)
                o[sl] *= f; h[sl] *= f; l[sl] *= f; c[sl] *= f
                recs.append({"symbol": sym, "date": _date(i), "type": "SPIKE_REVERT_FIXED", "ratio": float(q[i]),
                             "detail": f"{j - i + 1} bad bar(s) rescaled x{f:.3f}"})
                skip_until = j + 1
                break

    # 2) unadjusted corporate actions (chronological; later events never touch earlier-rebased bars twice)
    if allow_corporate:
        forced_dates = set(MANUAL_CORPORATE_ACTIONS.get(sym, []))
        veto_dates = set(KEEP_AS_REAL_MOVES.get(sym, []))
        with np.errstate(divide="ignore", invalid="ignore"):
            q_now = np.r_[1.0, c[1:] / np.where(c[:-1] > 0, c[:-1], np.nan)]
        cand_e = set(np.flatnonzero(np.abs(np.nan_to_num(q_now, nan=1.0) - 1.0) >= LARGE_MOVE_LOG).tolist())
        cand_e |= {i for i in range(1, n) if _date(i) in forced_dates} if forced_dates else set()
        for e in sorted(cand_e):
            if e < 1 or not (np.isfinite(c[e]) and np.isfinite(c[e - 1]) and c[e - 1] > 0 and c[e] > 0):
                continue
            qe = c[e] / c[e - 1]
            mag = abs(qe - 1.0)
            forced = _date(e) in forced_dates
            if _date(e) in veto_dates:
                recs.append({"symbol": sym, "date": _date(e), "type": "LARGE_MOVE_KEPT", "ratio": float(qe),
                             "detail": "kept by KEEP_AS_REAL_MOVES override"})
                continue
            if mag < LARGE_MOVE_LOG and not forced:
                continue
            gap = o[e] / c[e - 1] if (np.isfinite(o[e]) and o[e] > 0) else qe
            same_dir = (gap - 1.0) * (qe - 1.0) > 0.0          # V12.2: the open must have moved the SAME way as the close
            explained = bool(same_dir and abs(gap - 1.0) >= CA_GAP_EXPLAINS * mag)
            m_day = 0.0
            if mkt_ret is not None:
                try:
                    m_day = abs(float(mkt_ret.get(pd.Timestamp(idx[e]).normalize(), 0.0)))
                    m_day = m_day if np.isfinite(m_day) else 0.0
                except Exception:
                    m_day = 0.0
            if forced or ((qe <= CA_DROP_Q or qe >= CA_JUMP_Q) and explained and m_day < CA_MARKET_DAY_MAX):
                _pc, _nc, _no = float(c[e - 1]), float(c[e]), float(o[e])
                o[:e] *= qe; h[:e] *= qe; l[:e] *= qe; c[:e] *= qe; v[:e] /= qe
                recs.append({"symbol": sym, "date": _date(e), "type": "CORP_ACTION_REBASED", "ratio": float(qe),
                             "detail": f"pre-event prices x{qe:.4f}, volume /{qe:.4f}; prev close {_pc:.2f}, open {_no:.2f}, close {_nc:.2f}" + (" (manual)" if forced else "")})
            else:
                recs.append({"symbol": sym, "date": _date(e), "type": "LARGE_MOVE_KEPT", "ratio": float(qe),
                             "detail": f"{mag * 100:.0f}% move kept (index day move {m_day * 100:.1f}%, open-gap explains move: {bool(explained)}"
                                                  + ("" if same_dir else "; OPEN AND CLOSE MOVED OPPOSITE WAYS - suspect bar, not rebased") + ")"})

    if any(r["type"] != "LARGE_MOVE_KEPT" for r in recs):
        d["open"], d["high"], d["low"], d["close"] = o, h, l, c
        if "volume" in d.columns:
            d["volume"] = v
        if "quote_volume" in d.columns:
            d["quote_volume"] = c * v
    return d, recs


def build_market_universe(universe_name: str, start_year: int, refresh: bool,
                          extra_symbols: Optional[Dict[str, str]] = None) -> Tuple[MarketGrid, Dict[str, pd.DataFrame], pd.DataFrame, List[Dict[str, Any]]]:
    symbols, bracket_map = load_universe_with_brackets(universe_name)
    for _sym, _br in (extra_symbols or {}).items():      # live bot: holdings that are not in the pool still need rule evaluation
        if _sym not in bracket_map:
            bracket_map[_sym] = _br
            symbols = sorted(set(symbols) | {_sym})
    _, macro_df, _ = fetch_single_stock(MACRO_INDEX_TICKER, start_year, refresh)
    if macro_df is None:
        raise RuntimeError(f"Failed to load macro index data for {MACRO_INDEX_TICKER}.")
    _, bench_df, _ = fetch_single_stock(BENCH_SYMBOL, start_year, refresh)
    if bench_df is None:
        logging.warning(f"Benchmark {BENCH_SYMBOL}.NS unavailable; falling back to {MACRO_INDEX_TICKER} (price index, no dividends).")
        bench_df = macro_df

    raw_universe: Dict[str, pd.DataFrame] = {}
    provider_map: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=PARALLEL_DOWNLOAD_WORKERS) as executor:
        future_map = {executor.submit(fetch_single_stock, sym, start_year, refresh): sym for sym in symbols}
        for future in as_completed(future_map):
            sym, df, provider = future.result()
            if df is not None and len(df) >= MIN_HISTORY_DAYS:
                raw_universe[sym] = df
                provider_map[sym] = provider

    # ---- DATA INTEGRITY: repair benchmark, regime index and every constituent BEFORE any indicator is computed ----
    DATA_REPAIR_LOG.clear()
    bench_is_macro = bench_df is macro_df
    macro_df, r_m = repair_price_artifacts(macro_df, MACRO_INDEX_TICKER, None, allow_corporate=False)
    if bench_is_macro:
        bench_df, r_b = macro_df, []
    else:
        bench_df, r_b = repair_price_artifacts(bench_df, BENCH_SYMBOL, None, allow_corporate=False)
    mkt_ret = macro_df["close"].pct_change()
    mkt_ret.index = pd.DatetimeIndex(mkt_ret.index).normalize()
    all_recs: List[Dict[str, Any]] = list(r_m) + list(r_b)
    for sym_r in list(raw_universe.keys()):
        raw_universe[sym_r], recs_r = repair_price_artifacts(raw_universe[sym_r], sym_r, mkt_ret)
        all_recs.extend(recs_r)
    DATA_REPAIR_LOG.extend(all_recs)
    try:
        pd.DataFrame(all_recs).to_csv(REPAIRS_CSV_FILE, index=False)
    except Exception:
        pass
    n_fix = sum(1 for r in all_recs if r["type"] == "SPIKE_REVERT_FIXED")
    n_ca = sum(1 for r in all_recs if r["type"] == "CORP_ACTION_REBASED")
    n_kept = sum(1 for r in all_recs if r["type"] == "LARGE_MOVE_KEPT")
    logging.info(f"Data integrity: {n_fix} spike-and-revert bad prints fixed, {n_ca} unadjusted corporate actions rebased, "
                 f"{n_kept} large moves kept for manual review (see {REPAIRS_CSV_FILE.name}).")

    master_dates = pd.DatetimeIndex(sorted(macro_df.index.unique())).normalize()
    macro_df = macro_df.reindex(master_dates).ffill().dropna(subset=["close"])
    bench_close = bench_df["close"].reindex(master_dates).ffill().bfill().values.astype(np.float64)

    valid_symbols = sorted(raw_universe.keys())
    n_syms, n_bars = len(valid_symbols), len(master_dates)

    open_mat = np.full((n_syms, n_bars), np.nan, dtype=np.float64)
    high_mat = np.full((n_syms, n_bars), np.nan, dtype=np.float64)
    low_mat = np.full((n_syms, n_bars), np.nan, dtype=np.float64)
    close_mat = np.full((n_syms, n_bars), np.nan, dtype=np.float64)
    vol_mat = np.zeros((n_syms, n_bars), dtype=np.float64)
    atr14_mat = np.zeros((n_syms, n_bars), dtype=np.float64)
    adx14_mat = np.zeros((n_syms, n_bars), dtype=np.float64)
    dvol30_mat = np.zeros((n_syms, n_bars), dtype=np.float64)
    delist_mat = np.zeros((n_syms, n_bars), dtype=bool)
    alive_mat = np.zeros((n_syms, n_bars), dtype=bool)

    provenance_records = []
    universe: Dict[str, pd.DataFrame] = {}
    brackets = [bracket_map.get(s, "LARGE") for s in valid_symbols]
    liq_floor = np.array([BRACKET_LIQUIDITY_FLOOR_INR[b] for b in brackets], dtype=np.float64)
    slip_bps = np.array([BRACKET_SLIPPAGE_BPS[b] for b in brackets], dtype=np.float64)
    impact_bps = np.array([BRACKET_IMPACT_COEF_BPS[b] for b in brackets], dtype=np.float64)

    for i, sym in enumerate(valid_symbols):
        df = raw_universe[sym]
        reindexed = df.reindex(master_dates)
        raw_close = reindexed["close"].copy()

        first_idx, last_idx = df.index.min(), df.index.max()
        is_missing = raw_close.isna()
        trailing_missing_count = int((is_missing[::-1].cumprod()[::-1]).astype(int).sum())
        has_ever_traded = (~is_missing).cumsum() > 0
        is_delisted_perm = (is_missing[::-1].cumprod()[::-1].astype(bool) & has_ever_traded & (trailing_missing_count >= 20)).values

        reindexed["alive"] = ~is_missing
        reindexed["close"] = reindexed["close"].ffill()
        reindexed["open"] = reindexed["open"].ffill()
        reindexed["high"] = reindexed["high"].ffill()
        reindexed["low"] = reindexed["low"].ffill()
        reindexed["volume"] = reindexed["volume"].fillna(0.0)

        h_arr = reindexed["high"].values
        l_arr = reindexed["low"].values
        c_arr = reindexed["close"].values

        atr14 = FastIndicators.atr_1d(h_arr, l_arr, c_arr, 14)
        adx14 = FastIndicators.adx_1d(h_arr, l_arr, c_arr, 14)

        qv = reindexed["quote_volume"].values if "quote_volume" in reindexed else (c_arr * reindexed["volume"].values)
        # Median (not mean): a single volume-spike day cannot make an illiquid stock look liquid.
        # NaN (insufficient history) -> 0 so new listings FAIL any liquidity test until history exists.
        dvol30 = np_rolling_median(qv, ADV_WINDOW, ADV_MIN_PERIODS)
        dvol30 = np.nan_to_num(dvol30, nan=0.0)

        open_mat[i, :] = reindexed["open"].values
        high_mat[i, :] = h_arr
        low_mat[i, :] = l_arr
        close_mat[i, :] = c_arr
        vol_mat[i, :] = reindexed["volume"].values
        atr14_mat[i, :] = np.nan_to_num(atr14, nan=0.0)
        adx14_mat[i, :] = np.nan_to_num(adx14, nan=0.0)
        dvol30_mat[i, :] = dvol30
        delist_mat[i, :] = is_delisted_perm
        alive_mat[i, :] = (~is_missing).values
        universe[sym] = reindexed

        both_alive = (~is_missing).values[1:] & (~is_missing).values[:-1]
        rets_1d = np.where(both_alive, np.abs(c_arr[1:] / np.maximum(1e-9, c_arr[:-1]) - 1.0), 0.0)
        max_move_i = int(np.argmax(rets_1d)) if len(rets_1d) else 0
        provenance_records.append({
            "symbol": sym, "bracket": brackets[i], "provider": provider_map.get(sym, "unknown"),
            "first_date": str(first_idx.date()), "last_date": str(last_idx.date()),
            "total_bars": len(df), "is_delisted": bool(is_delisted_perm[-1]) if len(is_delisted_perm) else False,
            "max_1d_move_pct": float(rets_1d[max_move_i] * 100.0) if len(rets_1d) else 0.0,
            "max_1d_move_date": str(master_dates[max_move_i + 1].date()) if len(rets_1d) else "",
            "median_dvol_cr": float(np.nanmedian(dvol30[dvol30 > 0]) / 1e7) if np.any(dvol30 > 0) else 0.0,
        })

    grid = MarketGrid(
        symbols=valid_symbols, dates=master_dates,
        open_mat=open_mat, high_mat=high_mat, low_mat=low_mat, close_mat=close_mat, volume_mat=vol_mat,
        atr14_mat=atr14_mat, dvol30_mat=dvol30_mat, adx14_mat=adx14_mat,
        delist_mat=delist_mat, alive_mat=alive_mat,
        macro_close=macro_df["close"].values, macro_open=macro_df["open"].values,
        months_arr=master_dates.month.values, years_arr=master_dates.year.values,
        bench_close=bench_close, brackets=brackets, liq_floor=liq_floor,
        slip_bps=slip_bps, impact_bps=impact_bps
    )
    return grid, universe, macro_df, provenance_records


# ==============================================================================
# COMPILE SIGNALS & FUNNEL ATTRIBUTES
# ==============================================================================
def buy_sell_volume_split(h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Daily-bar proxy for buy vs sell volume: buy = V*(C-L)/(H-L). Circuit days (H == L): all buy if C >= previous C, else all sell."""
    rng = h - l
    safe = np.where(rng > 0, rng, 1.0)
    frac = np.where(rng > 0, (c - l) / safe, np.where(c >= shift_1d(c), 1.0, 0.0))
    frac = np.clip(np.nan_to_num(frac, nan=0.0), 0.0, 1.0)
    buy = v * frac
    return buy, v - buy


def buy_pressure_mask(h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray,
                      vol_ma_len: int, vol_mult: float, ratio: float) -> np.ndarray:
    """Entry type 5: volume above vol_mult x its prior average AND buy volume >= ratio x sell volume (completed bar)."""
    buy, sell = buy_sell_volume_split(h, l, c, v)
    vma = np_rolling_mean(shift_1d(v), int(vol_ma_len))
    return (~np.isnan(vma)) & (v > 0) & (v > vol_mult * vma) & (buy >= ratio * sell)


def compile_signals_fast(grid: MarketGrid, p: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_syms, n_bars = grid.close_mat.shape

    use_macro = p.get("use_market_macro_system", False)
    if use_macro:
        macro_ma = FastIndicators.moving_average(grid.macro_close, p["macro_ma_len"], p["macro_ma_type"])
        macro_ok = (~np.isnan(macro_ma)) & (grid.macro_close > macro_ma)
    else:
        macro_ok = np.ones(n_bars, dtype=bool)

    raw_signal_mat = np.zeros((n_syms, n_bars), dtype=bool)
    entry_mat = np.zeros((n_syms, n_bars), dtype=bool)
    exit_mat = np.zeros((n_syms, n_bars), dtype=bool)
    state_mat = np.zeros((n_syms, n_bars), dtype=bool)

    et, xt = p["entry_type"], p["exit_type"]
    adx_t = p.get("adx_thresh", 0.0)

    for i in range(n_syms):
        c_arr = grid.close_mat[i, :]
        o_arr = grid.open_mat[i, :]
        h_arr = grid.high_mat[i, :]
        l_arr = grid.low_mat[i, :]
        v_arr = grid.volume_mat[i, :]
        raw_entry = np.zeros(n_bars, dtype=bool)
        state_entry = np.zeros(n_bars, dtype=bool)

        if et == 0:
            ma = FastIndicators.moving_average(c_arr, p["entry_ma_len"], p["entry_ma_type"])
            raw_entry = (~np.isnan(ma)) & (c_arr > ma)
            state_entry = raw_entry
        elif et == 1:
            rf = FastIndicators.rsi_smoothed(c_arr, p["rsi_f_len"], p["rsi_f_smt"])
            rs = FastIndicators.rsi_smoothed(c_arr, p["rsi_s_len"], p["rsi_s_smt"])
            raw_entry = (~np.isnan(rf)) & (~np.isnan(rs)) & (rf > rs) & (shift_1d(rf) <= shift_1d(rs))
            if p.get("use_rsi_trend_filter", False):
                rma = FastIndicators.moving_average(c_arr, p["rsi_trend_ma_len"], p["rsi_trend_ma_type"])
                raw_entry = raw_entry & (~np.isnan(rma)) & (c_arr > rma)
                state_entry = (~np.isnan(rf)) & (~np.isnan(rs)) & (rf > rs) & (~np.isnan(rma)) & (c_arr > rma)
            else:
                state_entry = (~np.isnan(rf)) & (~np.isnan(rs)) & (rf > rs)
        elif et == 2:
            s_ma = FastIndicators.moving_average(c_arr, p["xover_short_len"], p["xover_short_type"])
            l_ma = FastIndicators.moving_average(c_arr, p["xover_long_len"], p["xover_long_type"])
            raw_entry = (~np.isnan(s_ma)) & (~np.isnan(l_ma)) & (s_ma > l_ma) & (shift_1d(s_ma) <= shift_1d(l_ma))
            state_entry = (~np.isnan(s_ma)) & (~np.isnan(l_ma)) & (s_ma > l_ma)
        elif et == 3:
            h_shifted = shift_1d(h_arr)
            hhv = np_rolling_max(h_shifted, int(p["price_lookback"]))
            body = c_arr - o_arr
            raw_entry = (~np.isnan(hhv)) & (c_arr > hhv) & (body >= (p["body_atr_mult"] * grid.atr14_mat[i, :]))
            if float(p["vol_mult"]) > 0.0:      # vol_mult = 0 -> pure breakout; legacy configs always have vol_mult >= 1.5
                vma = np_rolling_mean(shift_1d(v_arr), int(p["vol_ma_len"]))
                raw_entry = raw_entry & (~np.isnan(vma)) & (v_arr > (p["vol_mult"] * vma))
            base_ma = FastIndicators.moving_average(c_arr, int(p["price_lookback"]), 0)
            state_entry = (~np.isnan(base_ma)) & (c_arr > base_ma)
        elif et == 4:
            b_up, b_mid, _ = FastIndicators.bollinger_bands(c_arr, p["bb_entry_len"], p["bb_entry_std"])
            raw_entry = (~np.isnan(b_up)) & (c_arr > b_up) & (shift_1d(c_arr) <= shift_1d(b_up))
            state_entry = (~np.isnan(b_mid)) & (c_arr > b_mid)
        elif et == 5:
            raw_entry = buy_pressure_mask(h_arr, l_arr, c_arr, v_arr, p["bp_vol_ma_len"], p["bp_vol_mult"], p["bp_ratio"])
            state_entry = np.ones(n_bars, dtype=bool)      # a buying-pressure dip has no trend state to re-check at fill time

        raw_signal_mat[i, :] = raw_entry & grid.alive_mat[i, :]
        liq_ok = grid.dvol30_mat[i, :] >= max(float(grid.liq_floor[i]), 1.0)
        adx_ok = (grid.adx14_mat[i, :] >= adx_t) if adx_t > 0.0 else True

        entry_mat[i, :] = raw_entry & liq_ok & macro_ok & adx_ok & grid.alive_mat[i, :]
        state_mat[i, :] = state_entry & liq_ok & macro_ok & adx_ok & grid.alive_mat[i, :]

        if xt == 3:
            ma_val = FastIndicators.moving_average(c_arr, p["exit_ma_len"], p["exit_ma_type"])
            exit_mat[i, :] = (~np.isnan(ma_val)) & (c_arr < ma_val)
        elif xt == 4:
            rf = FastIndicators.rsi_smoothed(c_arr, p["exit_rsi_f_len"], p["exit_rsi_f_smt"])
            rs = FastIndicators.rsi_smoothed(c_arr, p["exit_rsi_s_len"], p["exit_rsi_s_smt"])
            exit_mat[i, :] = (~np.isnan(rf)) & (~np.isnan(rs)) & (rf < rs)
        elif xt == 5:
            s_ma = FastIndicators.moving_average(c_arr, p["exit_xover_short_len"], p["exit_xover_short_type"])
            l_ma = FastIndicators.moving_average(c_arr, p["exit_xover_long_len"], p["exit_xover_long_type"])
            exit_mat[i, :] = (~np.isnan(s_ma)) & (~np.isnan(l_ma)) & (s_ma < l_ma)
        elif xt == 6:
            v_shifted = shift_1d(v_arr)
            vma = np_rolling_mean(v_shifted, int(p["exit_vol_ma_len"]))
            exit_mat[i, :] = (~np.isnan(vma)) & (v_arr > (p["exit_vol_mult"] * vma)) & (c_arr < o_arr)
        elif xt == 7:
            _, b_mid, _ = FastIndicators.bollinger_bands(c_arr, p["bb_exit_len"], 2.0)
            exit_mat[i, :] = (~np.isnan(b_mid)) & (c_arr < b_mid)

    return raw_signal_mat, entry_mat, exit_mat, macro_ok, state_mat


# ==============================================================================
# EXECUTION & FORENSICS ENGINE
# ==============================================================================
@dataclass
class Tranche:
    coin: str
    coin_idx: int
    entry_bar: int
    entry_date: pd.Timestamp
    entry_price: float
    initial_units: float
    units: float
    cost_inr: float
    entry_atr: float
    current_sl: float
    stop_reason: str
    highest_high: float
    lowest_low: float
    peak_bar: int
    be_price: float = 0.0
    pending_stop: bool = False
    pending_reason: str = ""
    from_watchlist: bool = False
    wait_days: int = 0
    trigger_px: float = 0.0
    entry_equity: float = 0.0
    proceeds: float = 0.0
    fee_acc: FeeBreakdown = field(default_factory=FeeBreakdown)


def _accumulate_fees(dst: FeeBreakdown, src: FeeBreakdown) -> None:
    dst.brokerage += src.brokerage
    dst.stt += src.stt
    dst.exchange_charges += src.exchange_charges
    dst.sebi_charges += src.sebi_charges
    dst.gst += src.gst
    dst.dp_charges += src.dp_charges
    dst.slippage_cost += src.slippage_cost


def slip_fraction(grid: MarketGrid, c_i: int, part_rate: float) -> float:
    return (grid.slip_bps[c_i] + grid.impact_bps[c_i] * np.sqrt(part_rate)) / 10000.0


def make_tranche(grid: MarketGrid, c_i: int, t: int, fill_px: float, units: float, total_cost: float, fee_buy: FeeBreakdown,
                 slip_mult: float, entry_atr: float, sl_mult: float, from_watchlist: bool = False, wait_days: int = 0,
                 trigger_px: float = 0.0, entry_equity: float = 0.0) -> Tranche:
    """A position opened at the open of bar t. Its day-1 stop is judged on bar t's close and filled at the next open."""
    sl_price = fill_px - (sl_mult * entry_atr)
    return Tranche(
        coin=grid.symbols[c_i], coin_idx=c_i, entry_bar=t, entry_date=grid.dates[t],
        entry_price=fill_px, initial_units=units, units=units, cost_inr=total_cost,
        entry_atr=entry_atr, current_sl=sl_price, stop_reason="STOP_LOSS",
        highest_high=max(fill_px, grid.high_mat[c_i, t]), lowest_low=min(fill_px, grid.low_mat[c_i, t]),
        peak_bar=t, be_price=breakeven_exit_price(total_cost, units, slip_mult),
        pending_stop=bool(grid.close_mat[c_i, t] <= sl_price), pending_reason="STOP_LOSS",
        from_watchlist=from_watchlist, wait_days=wait_days, trigger_px=trigger_px,
        entry_equity=entry_equity, proceeds=0.0, fee_acc=fee_buy)


def tranche_open_exit_reason(tr: Tranche, t: int, grid: MarketGrid, exit_mat: np.ndarray, macro_bear_confirmed: bool,
                             xt: int, max_holding_bars: int) -> str:
    """Exit decided on information up to bar t-1 and filled at the open of bar t. Empty string = keep holding."""
    bars_held = t - tr.entry_bar
    if tr.pending_stop:
        return tr.pending_reason
    if macro_bear_confirmed:
        return "MACRO_REGIME_EXIT"
    if bars_held >= max_holding_bars:
        return "MAX_HOLDING_TIME"
    if xt in (3, 4, 5, 6, 7) and exit_mat[tr.coin_idx, t - 1]:
        if bars_held >= 3 or (tr.highest_high - tr.entry_price) >= (1.0 * tr.entry_atr):
            return f"SIGNAL_EXIT_TYPE_{xt}"
    return ""


def tranche_eod_update(tr: Tranche, t: int, grid: MarketGrid, xt: int, be_trigger_mult: float,
                       trail_pct_mult: float, trail_mult: float) -> None:
    """End-of-bar evaluation of a position that survived the open. A stop is judged on the CLOSE and sold at the next open."""
    c_i = tr.coin_idx
    h_bar, l_bar, c_bar = grid.high_mat[c_i, t], grid.low_mat[c_i, t], grid.close_mat[c_i, t]
    if h_bar > tr.highest_high:
        tr.highest_high = h_bar
        tr.peak_bar = t
    if l_bar < tr.lowest_low:
        tr.lowest_low = l_bar
    if c_bar <= tr.current_sl:
        tr.pending_stop = True
        tr.pending_reason = tr.stop_reason
        return
    if be_trigger_mult > 0.0 and tr.current_sl < tr.be_price:
        be_target = tr.entry_price + (be_trigger_mult * tr.entry_atr)
        if tr.highest_high >= be_target:
            tr.current_sl = max(tr.current_sl, tr.be_price)
            tr.stop_reason = "BREAKEVEN_SL"
    if xt == 1:
        pct_floor = tr.highest_high * trail_pct_mult
        if pct_floor > tr.current_sl:
            tr.current_sl = pct_floor
            tr.stop_reason = "TRAIL_PCT_STOP"
    elif trail_mult > 0.0:
        atr_floor = tr.highest_high - (trail_mult * grid.atr14_mat[c_i, t - 1])
        if atr_floor > tr.current_sl:
            tr.current_sl = atr_floor
            tr.stop_reason = "TRAIL_ATR_STOP"


def evaluate_entry(grid: MarketGrid, c_i: int, t: int, today_open: float, entry_atr: float, cash: float,
                   live_tranches: List[Tranche], day_equity: float, day_target: float, floor_inr: float,
                   free_adds: int, max_pyramid: int, whole_shares: bool) -> Dict[str, Any]:
    """Sizing and fill decision for one entry candidate at the open of bar t. Changes no state.
    The backtest passes the real open; the live bot passes its estimate of tomorrow's open."""
    coin_layers = sum(1 for tr in live_tranches if tr.coin_idx == c_i)
    if coin_layers > free_adds:
        highest_prior_entry = max(tr.entry_price for tr in live_tranches if tr.coin_idx == c_i)
        if today_open <= highest_prior_entry * 1.005:
            return {"ok": False, "reason": "anti_averaging_down", "starved": False}
    if cash < floor_inr:
        return {"ok": False, "reason": "cash_starved", "starved": True}
    max_pos_cap = day_equity * MAX_POSITION_EQUITY_PCT
    dynamic_slot_target = min(max_pos_cap, day_target)
    current_scrip_exposure = sum(tr.units * today_open for tr in live_tranches if tr.coin_idx == c_i)
    remaining_scrip_capacity = max(0.0, max_pos_cap - current_scrip_exposure)
    if remaining_scrip_capacity < floor_inr:
        return {"ok": False, "reason": "scrip_risk_cap", "starved": False}
    adv_30d = max(grid.dvol30_mat[c_i, t - 1], 1_000_000.0)
    liquidity_cap_inr = adv_30d * MAX_ADV_PARTICIPATION
    target_inr = min(dynamic_slot_target, liquidity_cap_inr, remaining_scrip_capacity)
    max_affordable = compute_max_affordable_tranche(cash, adv_30d, float(grid.slip_bps[c_i]), float(grid.impact_bps[c_i]))
    tranche_inr = min(max_affordable, max(floor_inr, target_inr))
    runt = tranche_inr < SIZE_RUNT_FRACTION * max(floor_inr, target_inr)
    slip_mult = slip_fraction(grid, c_i, min(1.0, max(0.0, tranche_inr / adv_30d)))
    fill_px, units, total_cost, fee_buy = _buy_fill_audited(tranche_inr, today_open, slip_mult, whole_shares)
    if cash >= total_cost and tranche_inr >= floor_inr and units > 0 and not runt and coin_layers < max_pyramid:
        if max_affordable < target_inr:
            binding = "cash_constrained"
        elif target_inr == liquidity_cap_inr:
            binding = "liquidity_cap"
        elif target_inr < floor_inr:
            binding = "tranche_floor"
        else:
            binding = "equity_slot"
        return {"ok": True, "binding": binding, "tranche_inr": tranche_inr, "fill_px": fill_px, "units": units,
                "total_cost": total_cost, "fee_buy": fee_buy, "slip_mult": slip_mult}
    if coin_layers >= max_pyramid:
        return {"ok": False, "reason": "pyramid_blocked", "starved": False}
    if units <= 0:
        return {"ok": False, "reason": "unaffordable", "starved": False}
    return {"ok": False, "reason": "cash_starved", "starved": True}      # cash short, below the floor, or a runt


class BacktestEngine:
    """
    Execution conventions (V11):
      * Every signal AND every stop is DECIDED on the close of bar t-1 and FILLED at the open of bar t.
        (Previously stops filled at the same bar's close, which is not achievable in live trading.)
      * A stop-loss breach on bar t's close therefore sets `pending_stop`; the position is sold at bar t+1's open,
        gaps included.
      * Breakeven stops sit at the true cost-covering exit price, not entry x 1.002.
      * Take-profit / partial exits were removed in V11.
    """
    def __init__(self, params: Dict[str, Any]):
        self.p = params

    def run_interval(self, grid: MarketGrid, raw_signal_mat: np.ndarray, entry_mat: np.ndarray,
                     exit_mat: np.ndarray, macro_ok: np.ndarray,
                     start_bar: int, end_bar: int, active_symbols_mask: Optional[np.ndarray] = None,
                     state_mat: Optional[np.ndarray] = None, contrib_offset: int = 0,
                     record_signals: bool = False) -> Dict[str, Any]:
        assert start_bar >= 1, "start_bar must be >= 1 for causal lookback."
        p = self.p
        dates = grid.dates
        max_holding_bars = p.get("max_holding_bars", 20)
        cfg = {**ENGINE_DEFAULTS, **p}
        size_w0, size_w_max = float(cfg["size_w0"]), float(cfg["size_w_max"])
        size_reserve, size_ramp = float(cfg["size_reserve"]), float(cfg["size_ramp"])
        size_w = size_w0
        be_trigger_mult = p.get("be_trigger_atr", 0.0)
        macro_active_exit = p.get("macro_active_exit", False)
        adx_t = p.get("adx_thresh", 0.0)
        floor_inr = float(p.get("tranche_floor", TRANCHE_FLOOR_INR))
        sl_mult_p = p.get("sl_mult", 3.0)

        active_state_mat = state_mat if state_mat is not None else entry_mat

        cash = INITIAL_CAPITAL
        total_inflow = INITIAL_CAPITAL
        live_tranches: List[Tranche] = []
        watchlist: List[Dict[str, Any]] = []
        closed_trades: List[Dict[str, Any]] = []

        n_interval = end_bar - start_bar
        active_capital_curve = np.zeros(n_interval, dtype=np.float64)
        total_equity_curve = np.zeros(n_interval, dtype=np.float64)
        twr_curve = np.zeros(n_interval, dtype=np.float64)
        concurrency_hist = np.zeros(MAX_TRACKED_POSITIONS + 1, dtype=np.int64)
        size_w_curve = np.zeros(n_interval, dtype=np.float64)
        idle_cash_curve = np.zeros(n_interval, dtype=np.float64)

        sig_id_counter = 0
        sig_meta: Dict[int, Tuple[int, int]] = {}
        sig_outcome: Dict[int, str] = {}

        funnel_dispositions = {
            "raw_triggers": 0, "macro_blocked": 0, "liquidity_blocked": 0,
            "adx_blocked": 0, "pyramid_blocked": 0, "revalidation_dropped": 0,
            "cash_starved_dropped": 0,
            "risk_cap_dropped": 0, "expired_unfilled": 0, "executed_fills": 0,
            "wl_duplicate_dropped": 0, "unaffordable_dropped": 0
        }

        binding_stats = {"tranche_ceiling": 0, "liquidity_cap": 0, "equity_slot": 0, "tranche_floor": 0, "cash_constrained": 0}
        max_pyramid = p.get("max_pyramid_layers", 1)
        trail_mult = p.get("trail_atr_mult", 0.0)
        trail_pct_mult = (1.0 - (p.get("trail_pct", 10.0) / 100.0))
        xt = p["exit_type"]
        wl_mode = p.get("wl_mode", "WL_NONE")
        wl_dedupe = bool(cfg["wl_dedupe"])
        wl_max_age = int(cfg["wl_max_age_bars"])
        wl_revalidate = bool(cfg["wl_revalidate"])
        free_adds = int(cfg["avg_down_free_adds"])
        whole_shares = bool(cfg["whole_shares"])

        def _slip(c_i: int, part_rate: float) -> float:
            return slip_fraction(grid, c_i, part_rate)

        def _set_outcome(item: Dict[str, Any], outcome: str) -> None:
            if record_signals:
                sig_outcome[item["sig_id"]] = outcome

        current_twr = 1.0
        prev_equity = cash
        peak_single_stock_pct = 0.0
        month_day = 0
        contrib_due = False

        for idx, t in enumerate(range(start_bar, end_bar)):
            curr_date = dates[t]
            inflow_today = 0.0

            # 0. Monthly Systematic Inflow (lands on trading day `contrib_offset` of each month; 0 = first)
            if t > start_bar:
                if grid.months_arr[t] != grid.months_arr[t - 1]:
                    if contrib_due:   # previous month was too short to reach the offset day: deposit late
                        cash += MONTHLY_CONTRIBUTION
                        total_inflow += MONTHLY_CONTRIBUTION
                        inflow_today += MONTHLY_CONTRIBUTION
                    contrib_due = True
                    month_day = 0
                else:
                    month_day += 1
                if contrib_due and month_day >= contrib_offset:
                    cash += MONTHLY_CONTRIBUTION
                    total_inflow += MONTHLY_CONTRIBUTION
                    inflow_today += MONTHLY_CONTRIBUTION
                    contrib_due = False

            if CASH_ANNUAL_YIELD > 0.0:
                cash += cash * (CASH_ANNUAL_YIELD / 248.0)

            # 1. Delisting Guard
            surviving_tranches = []
            for tr in live_tranches:
                if grid.delist_mat[tr.coin_idx, t]:
                    pnl = tr.proceeds - tr.cost_inr
                    vwap_exit = tr.proceeds / tr.initial_units if tr.initial_units > 0 else 0.0
                    closed_trades.append({
                        "coin": tr.coin, "pnl": pnl, "ret": pnl / tr.cost_inr if tr.cost_inr > 0 else 0.0,
                        "reason": "DELISTED", "bars": t - tr.entry_bar, "bars_to_peak": tr.peak_bar - tr.entry_bar,
                        "mfe_pct": ((tr.highest_high - tr.entry_price) / tr.entry_price) * 100.0,
                        "mae_pct": ((tr.lowest_low - tr.entry_price) / tr.entry_price) * 100.0,
                        "entry_date": tr.entry_date, "exit_date": curr_date, "exit_idx": idx,
                        "entry_price": tr.entry_price, "exit_price": vwap_exit,
                        "cost_inr": tr.cost_inr, "proceeds": tr.proceeds,
                        "frictions": tr.fee_acc, "from_watchlist": tr.from_watchlist,
                        "wait_bars": tr.wait_days, "entry_equity": tr.entry_equity,
                        "gap_vs_trigger_pct": ((tr.entry_price / tr.trigger_px - 1.0) * 100.0) if tr.trigger_px > 0 else 0.0,
                    })
                else:
                    surviving_tranches.append(tr)
            live_tranches = surviving_tranches

            # 2. Position management: decide on close(t-1) info, fill at open(t)
            macro_bear_confirmed = False
            if macro_active_exit and t >= 3:
                macro_bear_confirmed = (not macro_ok[t - 1]) and (not macro_ok[t - 2]) and (not macro_ok[t - 3])

            surviving_tranches = []
            for tr in live_tranches:
                c_i = tr.coin_idx

                if not grid.alive_mat[c_i, t]:
                    surviving_tranches.append(tr)
                    continue

                adv = max(grid.dvol30_mat[c_i, t - 1], 1_000_000.0)
                o_bar = grid.open_mat[c_i, t]
                bars_held = t - tr.entry_bar

                raw_exit_px = o_bar
                exit_reason = tranche_open_exit_reason(tr, t, grid, exit_mat, macro_bear_confirmed, xt, max_holding_bars)
                exit_triggered = bool(exit_reason)

                if exit_triggered:
                    part_rate_exit = min(1.0, max(0.0, (tr.units * raw_exit_px) / adv))
                    slip_exit = _slip(c_i, part_rate_exit)
                    fill_px, proceeds, fee_exit = _sell_fill_audited(tr.units, raw_exit_px, slip_exit, apply_dp=True)

                    cash += proceeds
                    total_proceeds = tr.proceeds + proceeds
                    pnl = total_proceeds - tr.cost_inr
                    vwap_exit = total_proceeds / tr.initial_units if tr.initial_units > 0 else fill_px
                    _accumulate_fees(tr.fee_acc, fee_exit)

                    closed_trades.append({
                        "coin": tr.coin, "pnl": pnl, "ret": pnl / tr.cost_inr if tr.cost_inr > 0 else 0.0,
                        "reason": exit_reason, "bars": bars_held, "bars_to_peak": tr.peak_bar - tr.entry_bar,
                        "mfe_pct": ((tr.highest_high - tr.entry_price) / tr.entry_price) * 100.0,
                        "mae_pct": ((tr.lowest_low - tr.entry_price) / tr.entry_price) * 100.0,
                        "entry_date": tr.entry_date, "exit_date": curr_date, "exit_idx": idx,
                        "entry_price": tr.entry_price, "exit_price": vwap_exit,
                        "cost_inr": tr.cost_inr, "proceeds": total_proceeds,
                        "frictions": tr.fee_acc, "from_watchlist": tr.from_watchlist,
                        "wait_bars": tr.wait_days, "entry_equity": tr.entry_equity,
                        "gap_vs_trigger_pct": ((tr.entry_price / tr.trigger_px - 1.0) * 100.0) if tr.trigger_px > 0 else 0.0,
                    })
                else:
                    tranche_eod_update(tr, t, grid, xt, be_trigger_mult, trail_pct_mult, trail_mult)
                    surviving_tranches.append(tr)
            live_tranches = surviving_tranches

            # 3. Ingest candidates decided on close(t-1)
            active_counts = {tr.coin_idx: 0 for tr in live_tranches}
            for tr in live_tranches:
                active_counts[tr.coin_idx] += 1

            new_candidates = []
            starved_today = False
            wl_coins = {it["coin_idx"] for it in watchlist} if wl_dedupe else set()
            sig_idx = np.flatnonzero(raw_signal_mat[:, t - 1])
            if active_symbols_mask is not None and len(sig_idx):
                sig_idx = sig_idx[active_symbols_mask[sig_idx]]

            # Fast path: on a day when NO order can be placed (no cash / no slot) and signals do not wait,
            # every filtered signal is dropped for the same reason -> count them vectorised (identical accounting).
            if (wl_mode == "WL_NONE") and (not record_signals) and cash < floor_inr and len(sig_idx):
                n_sig = int(len(sig_idx))
                funnel_dispositions["raw_triggers"] += n_sig
                sig_id_counter += n_sig
                if not macro_ok[t - 1]:
                    funnel_dispositions["macro_blocked"] += n_sig
                else:
                    liq_ok_v = grid.dvol30_mat[sig_idx, t - 1] >= np.maximum(grid.liq_floor[sig_idx], 1.0)
                    funnel_dispositions["liquidity_blocked"] += int((~liq_ok_v).sum())
                    rem = sig_idx[liq_ok_v]
                    ent_ok_v = entry_mat[rem, t - 1]
                    funnel_dispositions["adx_blocked"] += int((~ent_ok_v).sum())
                    rem = rem[ent_ok_v]
                    if len(rem):
                        if active_counts:
                            n_pyr = int(sum(1 for cc in rem if active_counts.get(int(cc), 0) >= max_pyramid))
                        else:
                            n_pyr = 0
                        funnel_dispositions["pyramid_blocked"] += n_pyr
                        n_rest = len(rem) - n_pyr
                        funnel_dispositions["cash_starved_dropped"] += n_rest
                        starved_today = starved_today or n_rest > 0
                sig_idx = sig_idx[:0]

            for c_i in sig_idx:
                c_i = int(c_i)
                if raw_signal_mat[c_i, t - 1]:
                    sig_id_counter += 1
                    funnel_dispositions["raw_triggers"] += 1
                    if record_signals:
                        sig_meta[sig_id_counter] = (c_i, t)

                    if not macro_ok[t - 1]:
                        funnel_dispositions["macro_blocked"] += 1
                        if record_signals: sig_outcome[sig_id_counter] = "macro_blocked"
                        continue
                    if grid.dvol30_mat[c_i, t - 1] < max(float(grid.liq_floor[c_i]), 1.0):
                        funnel_dispositions["liquidity_blocked"] += 1
                        if record_signals: sig_outcome[sig_id_counter] = "liquidity_blocked"
                        continue
                    if not entry_mat[c_i, t - 1]:
                        funnel_dispositions["adx_blocked"] += 1
                        if record_signals: sig_outcome[sig_id_counter] = "adx_blocked"
                        continue
                    if active_counts.get(c_i, 0) >= max_pyramid:
                        funnel_dispositions["pyramid_blocked"] += 1
                        if record_signals: sig_outcome[sig_id_counter] = "pyramid_blocked"
                        continue
                    if wl_dedupe and c_i in wl_coins:
                        funnel_dispositions["wl_duplicate_dropped"] += 1
                        if record_signals: sig_outcome[sig_id_counter] = "wl_duplicate"
                        continue

                    o_today = grid.open_mat[c_i, t]
                    a_yesterday = grid.atr14_mat[c_i, t - 1]
                    init_stop = o_today - (sl_mult_p * a_yesterday)
                    breakout_strength = (grid.close_mat[c_i, t - 1] - grid.open_mat[c_i, t - 1]) / max(1e-6, a_yesterday)
                    new_candidates.append({
                        "sig_id": sig_id_counter,
                        "coin": grid.symbols[c_i], "coin_idx": c_i, "signal_bar": t,
                        "trigger_price": grid.close_mat[c_i, t - 1], "shadow_stop": init_stop,
                        "highest_high": o_today, "breakout_quality": breakout_strength,
                        "entry_atr": a_yesterday, "unfilled_reason": None
                    })
                    wl_coins.add(c_i)

            new_candidates.sort(key=lambda x: x["breakout_quality"], reverse=True)
            watchlist.extend(new_candidates)

            # 4. Sizing & order fill with persistent-state revalidation
            day_block_cause: Optional[str] = None

            def _record_unfilled_disposition(item: Dict[str, Any], default_bucket: str = "expired_unfilled"):
                reason = day_block_cause if day_block_cause else item.get("unfilled_reason")
                if reason == "cash_starved":
                    funnel_dispositions["cash_starved_dropped"] += 1
                elif reason in ("pyramid_blocked", "anti_averaging_down"):
                    funnel_dispositions["pyramid_blocked"] += 1
                elif reason == "scrip_risk_cap":
                    funnel_dispositions["risk_cap_dropped"] += 1
                elif reason == "unaffordable":
                    funnel_dispositions["unaffordable_dropped"] += 1
                else:
                    funnel_dispositions[default_bucket] += 1
                _set_outcome(item, reason if reason else default_bucket)

            if watchlist and cash >= floor_inr:
                if wl_mode == "WL_DEEPEST_DISCOUNT":
                    watchlist.sort(key=lambda x: (x["trigger_price"] - grid.close_mat[x["coin_idx"], t - 1]) / max(1e-6, x["trigger_price"]), reverse=True)
                elif wl_mode == "WL_STRONGEST_MOMENTUM":
                    watchlist.sort(key=lambda x: x["breakout_quality"], reverse=True)

                unfilled = []
                # one equity snapshot and one target per day -> every entry of the batch gets the same size
                day_equity = cash + sum(tr.units * grid.open_mat[tr.coin_idx, t] for tr in live_tranches)
                day_target = max(floor_inr, day_equity * size_w)
                for item in watchlist:
                    c_i = item["coin_idx"]

                    if not grid.alive_mat[c_i, t]:
                        funnel_dispositions["revalidation_dropped"] += 1
                        _set_outcome(item, "revalidation_dropped")
                        continue

                    if wl_revalidate and t > item["signal_bar"]:
                        if not macro_ok[t - 1]:
                            funnel_dispositions["revalidation_dropped"] += 1
                            _set_outcome(item, "revalidation_dropped")
                            continue
                        if adx_t > 0.0 and grid.adx14_mat[c_i, t - 1] < adx_t:
                            funnel_dispositions["revalidation_dropped"] += 1
                            _set_outcome(item, "revalidation_dropped")
                            continue
                        if not active_state_mat[c_i, t - 1]:
                            funnel_dispositions["revalidation_dropped"] += 1
                            _set_outcome(item, "revalidation_dropped")
                            continue

                    dec = evaluate_entry(grid, c_i, t, grid.open_mat[c_i, t], item["entry_atr"], cash, live_tranches,
                                         day_equity, day_target, floor_inr, free_adds, max_pyramid, whole_shares)
                    if dec["ok"]:
                        binding_stats[dec["binding"]] += 1
                        cash -= dec["total_cost"]
                        funnel_dispositions["executed_fills"] += 1
                        _set_outcome(item, "executed")
                        live_tranches.append(make_tranche(
                            grid, c_i, t, dec["fill_px"], dec["units"], dec["total_cost"], dec["fee_buy"], dec["slip_mult"],
                            item["entry_atr"], sl_mult_p, from_watchlist=(t > item["signal_bar"]),
                            wait_days=(t - item["signal_bar"]), trigger_px=float(item["trigger_price"]), entry_equity=day_equity))
                    else:
                        item["unfilled_reason"] = dec["reason"]
                        starved_today = starved_today or dec["starved"]
                        unfilled.append(item)
                watchlist = unfilled
            elif watchlist:
                # Nothing could be attempted today: cash is below the floor. Remember it for disposition accounting.
                day_block_cause = "cash_starved"
                starved_today = True

            # Terminal disposition accounting for single-day watchlist modes
            if wl_mode == "WL_NONE" and watchlist:
                for item in watchlist:
                    _record_unfilled_disposition(item)
                watchlist = []

            # 5. Causal watchlist lifecycle update (EOD on unfilled items only)
            if wl_mode != "WL_NONE":
                surviving_watchlist = []
                for item in watchlist:
                    c_i = item["coin_idx"]
                    l_bar, h_bar = grid.low_mat[c_i, t], grid.high_mat[c_i, t]

                    shadow_stopped = (l_bar <= item["shadow_stop"])
                    shadow_exited = (xt in (3, 4, 5, 6, 7)) and exit_mat[c_i, t]
                    shadow_expired = wl_max_age > 0 and (t - item["signal_bar"]) >= wl_max_age

                    if shadow_expired or shadow_stopped or shadow_exited:
                        _record_unfilled_disposition(item, default_bucket="expired_unfilled")
                    else:
                        if h_bar > item["highest_high"]:
                            item["highest_high"] = h_bar
                        if trail_mult > 0.0:
                            item["shadow_stop"] = max(item["shadow_stop"], item["highest_high"] - (trail_mult * grid.atr14_mat[c_i, t]))
                        surviving_watchlist.append(item)
                watchlist = surviving_watchlist
            else:
                watchlist = []

            # 6. Horizon terminal mark (final bar only)
            if t == end_bar - 1:
                if wl_mode != "WL_NONE" and watchlist:
                    for item in watchlist:
                        _record_unfilled_disposition(item, default_bucket="expired_unfilled")
                    watchlist = []

                for tr in list(live_tranches):
                    c_i = tr.coin_idx
                    c_px = grid.close_mat[c_i, t]
                    if np.isfinite(c_px) and c_px > 0:
                        adv = max(grid.dvol30_mat[c_i, t - 1], 1_000_000.0)
                        part_rate = min(1.0, max(0.0, (tr.units * c_px) / adv))
                        slip_mult = _slip(c_i, part_rate)
                        fill_px, proceeds, fee_term = _sell_fill_audited(tr.units, c_px, slip_mult, apply_dp=True)
                        cash += proceeds
                        total_proceeds = tr.proceeds + proceeds
                        pnl = total_proceeds - tr.cost_inr
                        vwap_exit = total_proceeds / tr.initial_units if tr.initial_units > 0 else fill_px
                        _accumulate_fees(tr.fee_acc, fee_term)

                        closed_trades.append({
                            "coin": tr.coin, "pnl": pnl, "ret": pnl / tr.cost_inr if tr.cost_inr > 0 else 0.0,
                            "reason": "END_OF_TEST", "bars": t - tr.entry_bar,
                            "bars_to_peak": tr.peak_bar - tr.entry_bar,
                            "mfe_pct": ((tr.highest_high - tr.entry_price) / tr.entry_price) * 100.0,
                            "mae_pct": ((tr.lowest_low - tr.entry_price) / tr.entry_price) * 100.0,
                            "entry_date": tr.entry_date, "exit_date": curr_date, "exit_idx": idx,
                            "entry_price": tr.entry_price, "exit_price": vwap_exit,
                            "cost_inr": tr.cost_inr, "proceeds": total_proceeds,
                            "frictions": tr.fee_acc, "from_watchlist": tr.from_watchlist,
                        "wait_bars": tr.wait_days, "entry_equity": tr.entry_equity,
                        "gap_vs_trigger_pct": ((tr.entry_price / tr.trigger_px - 1.0) * 100.0) if tr.trigger_px > 0 else 0.0,
                        })
                live_tranches = []

            # 7. Daily time-weighted return (TWR) indexing & exposure tracking
            scrip_mkt_vals: Dict[int, float] = {}
            for tr in live_tranches:
                val = tr.units * grid.close_mat[tr.coin_idx, t]
                scrip_mkt_vals[tr.coin_idx] = scrip_mkt_vals.get(tr.coin_idx, 0.0) + val

            current_active = sum(scrip_mkt_vals.values())
            eod_equity = cash + current_active

            if eod_equity > 0 and scrip_mkt_vals:
                max_scrip_val_today = max(scrip_mkt_vals.values())
                peak_single_stock_pct = max(peak_single_stock_pct, max_scrip_val_today / eod_equity)

            if idx == 0:
                daily_twr_ret = 0.0
            else:
                capital_base = prev_equity + inflow_today
                daily_twr_ret = (eod_equity - capital_base) / max(1.0, capital_base)

            current_twr *= (1.0 + daily_twr_ret)
            prev_equity = eod_equity

            active_capital_curve[idx] = current_active
            total_equity_curve[idx] = eod_equity
            twr_curve[idx] = current_twr
            idle_cash_curve[idx] = cash / eod_equity if eod_equity > 0 else 0.0
            # sizing weight: decided on today's close, used for tomorrow's orders (causal)
            size_w_curve[idx] = size_w
            if starved_today:
                size_w = max(size_w0, size_w * (1.0 - size_ramp))
            elif eod_equity > 0 and cash / eod_equity > size_reserve:
                size_w = min(size_w_max, size_w * (1.0 + size_ramp))
            concurrency_hist[min(len(live_tranches), len(concurrency_hist) - 1)] += 1

        signal_log = []
        if record_signals:
            for sid, (c_i, sbar) in sig_meta.items():
                signal_log.append((sid, c_i, sbar, sig_outcome.get(sid, "open_at_end")))

        return {
            "trades": pd.DataFrame(closed_trades),
            "equity": total_equity_curve,
            "twr_curve": twr_curve,
            "active_capital": active_capital_curve,
            "total_inflow": total_inflow,
            "final_cash": cash,
            "dates": dates[start_bar:end_bar],
            "binding_stats": binding_stats,
            "funnel_stats": funnel_dispositions,
            "concurrency_hist": concurrency_hist,
            "size_w_curve": size_w_curve,
            "idle_cash_curve": idle_cash_curve,
            "peak_single_stock_pct": float(peak_single_stock_pct * 100.0),
            "signal_log": signal_log,
        }


# ==============================================================================
# AUDITED TIME-WEIGHTED RETURN (TWR) METRIC ENGINE
# ==============================================================================
def _year_returns(twr: np.ndarray, years: np.ndarray, min_bars: int = OBJ_MIN_YEAR_BARS) -> Dict[int, float]:
    """Flow-stripped calendar-year returns (%) from a TWR curve aligned bar-for-bar with `years`."""
    out: Dict[int, float] = {}
    if len(twr) < 2 or len(twr) != len(years):
        return out
    for y in np.unique(years):
        ix = np.flatnonzero(years == y)
        if len(ix) < min_bars:
            continue
        base = twr[ix[0] - 1] if ix[0] >= 1 else 1.0
        out[int(y)] = float((twr[ix[-1]] / max(1e-12, base) - 1.0) * 100.0)
    return out


def calculate_benchmarks(grid: MarketGrid, start_bar: int, end_bar: int, macro_ok: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Calculates TWR for Literal ^NSEI, Survivor Basket, and Reference 200 SMA Timed Benchmarks."""
    n_bars = end_bar - start_bar

    # 1. Literal benchmark TWR (NIFTYBEES.NS ETF series; falls back to ^NSEI if the ETF is unavailable)
    nsei_p = grid.bench_close[start_bar:end_bar]
    nsei_daily_ret = np.diff(nsei_p) / np.maximum(1e-6, nsei_p[:-1])
    nsei_twr = np.insert(np.cumprod(1.0 + nsei_daily_ret), 0, 1.0)
    nsei_cagr = (nsei_twr[-1] ** (248.0 / max(1, n_bars))) - 1.0

    peak_n = np.maximum.accumulate(nsei_twr)
    dd_n = np.where(peak_n > 0, (peak_n - nsei_twr) / peak_n, 0.0)
    nsei_max_dd = float(np.max(dd_n)) * 100.0 if len(dd_n) > 0 else 0.0

    # 2. Equal-Weight Survivor Basket TWR
    basket_close = grid.close_mat[:, start_bar:end_bar]
    alive_slice = grid.alive_mat[:, start_bar:end_bar]
    basket_returns = np.diff(basket_close, axis=1) / np.maximum(1e-6, basket_close[:, :-1])
    # Prevent a single corrupted small-cap tick from dominating the benchmark CAGR.
    basket_returns = np.clip(basket_returns, -0.95, 3.0)

    daily_basket_ret = np.zeros(n_bars - 1, dtype=np.float64)
    for b in range(n_bars - 1):
        alive_today = alive_slice[:, b] & alive_slice[:, b + 1]
        if np.any(alive_today):
            daily_basket_ret[b] = np.mean(basket_returns[alive_today, b])

    basket_twr = np.insert(np.cumprod(1.0 + daily_basket_ret), 0, 1.0)
    basket_cagr = (basket_twr[-1] ** (248.0 / max(1, n_bars))) - 1.0

    peak_b = np.maximum.accumulate(basket_twr)
    dd_b = np.where(peak_b > 0, (peak_b - basket_twr) / peak_b, 0.0)
    basket_max_dd = float(np.max(dd_b)) * 100.0 if len(dd_b) > 0 else 0.0

    # 3. Strategy Macro-Timed Benchmarks
    daily_cash_ret = CASH_ANNUAL_YIELD / 248.0
    if macro_ok is not None:
        macro_slice = macro_ok[start_bar:end_bar - 1]
        timed_nsei_daily = np.where(macro_slice, nsei_daily_ret, daily_cash_ret)
        timed_basket_daily = np.where(macro_slice, daily_basket_ret, daily_cash_ret)

        timed_nsei_twr = np.insert(np.cumprod(1.0 + timed_nsei_daily), 0, 1.0)
        timed_nsei_cagr = (timed_nsei_twr[-1] ** (248.0 / max(1, n_bars))) - 1.0
        timed_nsei_total_return = (timed_nsei_twr[-1] - 1.0) * 100.0

        timed_basket_twr = np.insert(np.cumprod(1.0 + timed_basket_daily), 0, 1.0)
        timed_basket_cagr = (timed_basket_twr[-1] ** (248.0 / max(1, n_bars))) - 1.0
        timed_basket_total_return = (timed_basket_twr[-1] - 1.0) * 100.0
    else:
        timed_nsei_cagr = nsei_cagr
        timed_nsei_total_return = (nsei_twr[-1] - 1.0) * 100.0
        timed_basket_cagr = basket_cagr
        timed_basket_total_return = (basket_twr[-1] - 1.0) * 100.0

    # 4. Standard Reference 200 SMA Timed Benchmark
    ref_200_ma = FastIndicators.moving_average(grid.macro_close, 200, 0)
    ref_macro_ok = (~np.isnan(ref_200_ma)) & (grid.macro_close > ref_200_ma)
    ref_slice = ref_macro_ok[start_bar:end_bar - 1]
    ref_timed_nsei_daily = np.where(ref_slice, nsei_daily_ret, daily_cash_ret)
    ref_timed_nsei_twr = np.insert(np.cumprod(1.0 + ref_timed_nsei_daily), 0, 1.0)
    ref_timed_nsei_cagr = (ref_timed_nsei_twr[-1] ** (248.0 / max(1, n_bars))) - 1.0

    yrs_slice = grid.years_arr[start_bar:end_bar]
    return {
        "idx_year": _year_returns(nsei_twr, yrs_slice),
        "bsk_year": _year_returns(basket_twr, yrs_slice),
        "nsei_twr_cagr": float(nsei_cagr * 100.0),
        "nsei_total_return": float((nsei_twr[-1] - 1.0) * 100.0),
        "nsei_max_dd": float(nsei_max_dd),
        "nsei_daily_ret": nsei_daily_ret,
        "basket_twr_cagr": float(basket_cagr * 100.0),
        "basket_total_return": float((basket_twr[-1] - 1.0) * 100.0),
        "basket_max_dd": float(basket_max_dd),
        "timed_nsei_cagr": float(timed_nsei_cagr * 100.0),
        "timed_nsei_total_return": float(timed_nsei_total_return),
        "timed_basket_cagr": float(timed_basket_cagr * 100.0),
        "timed_basket_total_return": float(timed_basket_total_return),
        "ref_200sma_timed_cagr": float(ref_timed_nsei_cagr * 100.0),
    }


def _calc_sub_metrics(tdf: pd.DataFrame, cagr: float, total_ret: float) -> Dict[str, Any]:
    if len(tdf) == 0:
        return {
            "trades": 0, "net_pnl": 0.0, "cagr": 0.0, "total_return": 0.0,
            "win_rate": 0.0, "pure_win_rate": 0.0, "be_rate": 0.0, "loss_rate": 0.0,
            "profit_factor": 0.0
        }
    net_pnl = float(tdf["pnl"].sum())
    
    is_be = tdf["reason"].str.contains("BREAKEVEN") | (tdf["ret"].abs() <= 0.0025)
    wins = tdf[(tdf["pnl"] > 0) & (~is_be)]["pnl"]
    losses = tdf[(tdf["pnl"] < 0) & (~is_be)]["pnl"].abs()
    be_trades = tdf[is_be]

    pure_wins_count = len(wins)
    be_count = len(be_trades)
    losses_count = len(losses)
    n_tot = len(tdf)

    win_rate = (len(tdf[tdf["pnl"] > 0]) / n_tot) * 100.0
    pure_win_rate = (pure_wins_count / n_tot) * 100.0
    be_rate = (be_count / n_tot) * 100.0
    loss_rate = (losses_count / n_tot) * 100.0

    pf = float(wins.sum() / (losses.sum() + 1e-9)) if len(losses) > 0 else 10.0
    return {
        "trades": n_tot, "net_pnl": net_pnl, "cagr": cagr, "total_return": total_ret,
        "win_rate": win_rate, "pure_win_rate": pure_win_rate, "be_rate": be_rate,
        "loss_rate": loss_rate, "profit_factor": pf
    }


def calculate_metrics(results: Dict[str, Any], grid: MarketGrid, start_bar: int, end_bar: int,
                      benchmark_cagr: Any, min_trades: Optional[int] = None) -> Dict[str, Any]:
    # `benchmark_cagr` is normally the dict from calculate_benchmarks (enables the annual objective); a bare float
    # still works and falls back to a clipped whole-window excess.
    bench_obj = benchmark_cagr if isinstance(benchmark_cagr, dict) else None
    benchmark_cagr = float(bench_obj["nsei_twr_cagr"]) if bench_obj is not None else float(benchmark_cagr)
    tdf = results["trades"]
    twr_series = results["twr_curve"]
    eq = results["equity"]
    ac = results["active_capital"]
    n_bars = end_bar - start_bar

    if len(twr_series) > 0:
        peak_twr = np.maximum.accumulate(twr_series)
        dd_curve = np.where(peak_twr > 0, (peak_twr - twr_series) / peak_twr, 0.0)
        max_dd = float(np.max(dd_curve)) * 100.0 if len(dd_curve) > 0 else 0.0
        total_ret_full = float((twr_series[-1] - 1.0) * 100.0)
        cagr_full = float(((twr_series[-1]) ** (248.0 / max(1, n_bars)) - 1.0) * 100.0)
    else:
        max_dd, total_ret_full, cagr_full = 0.0, 0.0, 0.0

    if len(tdf) > 0 and len(eq) > 0:
        cum_pnl = tdf["pnl"].cumsum().values
        peak_pnl = np.maximum.accumulate(cum_pnl)
        trade_dd_inr = peak_pnl - cum_pnl
        peak_contemp_equity = np.maximum.accumulate(eq)
        
        if "exit_idx" in tdf.columns:
            exit_indices = np.clip(tdf["exit_idx"].values.astype(int), 0, len(eq) - 1)
            trade_denominators = np.maximum(1000.0, peak_contemp_equity[exit_indices])
        else:
            trade_denominators = np.maximum(1000.0, float(np.max(eq)))
            
        trade_dd_pcts = (trade_dd_inr / trade_denominators) * 100.0
        closed_trade_dd_pct = float(np.max(trade_dd_pcts)) if len(trade_dd_pcts) > 0 else 0.0
    else:
        closed_trade_dd_pct = 0.0

    avg_active_cap = float(np.mean(ac)) if len(ac) > 0 else 0.0
    avg_total_equity = float(np.mean(eq)) if len(eq) > 0 else INITIAL_CAPITAL
    utilization = float(avg_active_cap / (avg_total_equity + 1e-9))

    full = _calc_sub_metrics(tdf, cagr_full, total_ret_full)

    eot_tdf = tdf[tdf["reason"] == "END_OF_TEST"] if len(tdf) > 0 else pd.DataFrame()
    eot_trades = len(eot_tdf)
    eot_pnl = float(eot_tdf["pnl"].sum()) if eot_trades > 0 else 0.0

    # Benchmark drawdown over the same window (from the benchmark series itself)
    bp = grid.bench_close[start_bar:end_bar]
    if len(bp) > 1:
        b_peak = np.maximum.accumulate(bp)
        bench_dd = float(np.max((b_peak - bp) / np.maximum(1e-9, b_peak))) * 100.0
    else:
        bench_dd = 0.0

    # Terminal-bar marks are not fully-realized alpha: credit only a fraction of the unclosed excess.
    final_equity = float(eq[-1]) if len(eq) > 0 else INITIAL_CAPITAL
    organic_final_equity = max(1e-6, final_equity - eot_pnl)
    final_twr = max(1e-6, float(twr_series[-1])) if len(twr_series) > 0 else 1.0
    organic_twr_ratio = np.clip(organic_final_equity / max(1e-6, final_equity), 0.0, 1.0)
    organic_twr_end = final_twr * organic_twr_ratio
    years_248 = max(1, n_bars) / 248.0
    cagr_organic = float(((organic_twr_end ** (1.0 / years_248)) - 1.0) * 100.0)

    unclosed_excess_cagr = max(0.0, cagr_full - cagr_organic)
    anti_cheat_cagr = cagr_organic + (EOT_EXCESS_WEIGHT * unclosed_excess_cagr)

    if len(tdf) > 5:
        trade_rets = tdf["ret"].values
        p95 = float(np.percentile(trade_rets, 95))
        clipped_rets = np.minimum(trade_rets, p95)
        denom_rets = float(np.sum(trade_rets))
        expectance_discount = float(np.clip(np.sum(clipped_rets) / denom_rets, 0.2, 1.0)) if denom_rets > 0 else 0.2
    else:
        expectance_discount = 1.0

    trimmed_cagr = anti_cheat_cagr * expectance_discount if anti_cheat_cagr > 0 else anti_cheat_cagr

    # OBJECTIVE V2 - annual, survivorship-neutral, lottery-resistant:
    #   for each calendar year: strategy TWR minus the BETTER of (NIFTYBEES, equal-weight basket of this universe),
    #   clipped to +/- OBJ_YEAR_EXCESS_CLIP points, averaged over years; minus a penalty only for drawdown WORSE than the
    #   benchmark's own. Cash earns 0, so hiding in cash cannot flatter the score.
    excess_cagr = trimmed_cagr - benchmark_cagr
    year_excess_raw: List[float] = []
    if bench_obj is not None and len(twr_series) > 1:
        s_year = _year_returns(twr_series, grid.years_arr[start_bar:end_bar])
        for y_, sr_ in s_year.items():
            ref_ = max(bench_obj.get("idx_year", {}).get(y_, -1e9), bench_obj.get("bsk_year", {}).get(y_, -1e9))
            if ref_ > -1e8:
                year_excess_raw.append(float(sr_ - ref_))
    if year_excess_raw:
        obj_core = float(np.mean(np.clip(year_excess_raw, -OBJ_YEAR_EXCESS_CLIP, OBJ_YEAR_EXCESS_CLIP)))
    else:
        obj_core = float(np.clip(cagr_full - benchmark_cagr, -OBJ_YEAR_EXCESS_CLIP, OBJ_YEAR_EXCESS_CLIP))
    dd_penalty = DD_PENALTY_WEIGHT * max(0.0, max_dd - bench_dd)
    score = obj_core - dd_penalty

    n_min = MIN_CLOSED_TRADES if min_trades is None else int(min_trades)
    if n_min > 0 and len(tdf) < n_min:
        score = -10.0 + (len(tdf) / float(n_min)) * 5.0   # graded, so TPE still sees a gradient

    eot_share = (eot_pnl / float(tdf["pnl"].sum())) if (len(tdf) > 0 and float(tdf["pnl"].sum()) > 0) else 0.0

    return {
        "score": float(np.nan_to_num(np.clip(score, -50.0, 200.0), nan=-10.0)), "full": full,
        "eot_trades": eot_trades, "eot_pnl": eot_pnl, "eot_profit_share": float(eot_share),
        "cagr_organic": float(cagr_organic), "trimmed_cagr": float(trimmed_cagr),
        "excess_cagr": float(excess_cagr), "bench_max_dd": float(bench_dd),
        "max_dd": float(max_dd), "closed_trade_dd_pct": float(closed_trade_dd_pct),
        "utilization": float(utilization * 100.0),
        "obj_core": float(obj_core), "dd_penalty": float(dd_penalty), "year_excess_raw": year_excess_raw,
        "excess_vs_basket_cagr": float(trimmed_cagr - float(bench_obj["basket_twr_cagr"])) if bench_obj is not None else 0.0,
    }


# ==============================================================================
# ANTI-OVERFITTING: MULTI-ARCH CAUSALITY & NEIGHBORHOOD AUDIT
# ==============================================================================
def logged_phase(name: str):
    """Logs start and elapsed time of a slow validation phase so the console never goes silent."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            t0 = time.time()
            logging.info(f"[phase] {name} ...")
            out = fn(*a, **k)
            logging.info(f"[phase] {name} done in {time.time() - t0:.0f}s")
            return out
        return wrapper
    return deco


def dynamic_parameter_neighbors(base_p: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = {
        "entry_ma_len": 5, "rsi_f_len": 2, "rsi_f_smt": 2, "rsi_s_len": 2, "rsi_s_smt": 2,
        "rsi_trend_ma_len": 10, "xover_short_len": 5, "xover_gap": 5, "vol_ma_len": 2,
        "vol_mult": 0.2, "price_lookback": 2, "body_atr_mult": 0.1, "bb_entry_len": 2,
        "bb_entry_std": 0.1, "macro_ma_len": 10, "sl_mult": 0.3,
        "trail_atr_mult": 0.5, "max_holding_bars": 5, "trail_pct": 2.0, "exit_ma_len": 5,
        "exit_rsi_f_len": 2, "exit_rsi_f_smt": 2, "exit_rsi_s_len": 2, "exit_rsi_s_smt": 2,
        "exit_xover_short_len": 5, "exit_xover_gap": 5, "exit_vol_ma_len": 5,
        "exit_vol_mult": 0.2, "bb_exit_len": 2, "bp_vol_ma_len": 2, "bp_vol_mult": 0.25, "bp_ratio": 0.5,
        "adx_thresh": 5.0, "be_trigger_atr": 1.0
    }

    neighbors = []
    for k, step in steps.items():
        if k not in base_p or (k == "vol_mult" and float(base_p[k]) == 0.0):   # vol_mult 0 = volume confirmation off; a +step would switch it on
            continue
        for mult in (+1, -1):
            cand = copy.deepcopy(base_p)
            if isinstance(cand[k], float):
                cand[k] = round(max(0.0, cand[k] + mult * step), 2)
            elif isinstance(cand[k], int):
                cand[k] = max(2, int(cand[k] + mult * step))

            if "xover_short_len" in cand and "xover_gap" in cand:
                cand["xover_long_len"] = min(200, cand["xover_short_len"] + cand["xover_gap"])
            if "exit_xover_short_len" in cand and "exit_xover_gap" in cand:
                cand["exit_xover_long_len"] = min(200, cand["exit_xover_short_len"] + cand["exit_xover_gap"])

            neighbors.append(cand)

    ma_type_keys = ("entry_ma_type", "xover_short_type", "xover_long_type", "rsi_trend_ma_type",
                    "macro_ma_type", "exit_ma_type", "exit_xover_short_type", "exit_xover_long_type")
    for k in ma_type_keys:
        if k in base_p:
            for alt in (0, 1, 3):
                if alt != base_p[k]:
                    cand_t = copy.deepcopy(base_p)
                    cand_t[k] = alt
                    neighbors.append(cand_t)

    if "tranche_floor" in base_p and float(base_p["tranche_floor"]) in TRANCHE_FLOOR_CHOICES:
        i_f = TRANCHE_FLOOR_CHOICES.index(float(base_p["tranche_floor"]))
        for j_f in (i_f - 1, i_f + 1):
            if 0 <= j_f < len(TRANCHE_FLOOR_CHOICES):
                cand_f = copy.deepcopy(base_p)
                cand_f["tranche_floor"] = TRANCHE_FLOOR_CHOICES[j_f]
                neighbors.append(cand_f)

    for k_c, choices in (("size_w0", SIZE_W0_CHOICES), ("size_w_max", SIZE_W_MAX_CHOICES),
                         ("size_reserve", SIZE_RESERVE_CHOICES), ("size_ramp", SIZE_RAMP_CHOICES)):
        if k_c in base_p and float(base_p[k_c]) in choices:
            i_c = choices.index(float(base_p[k_c]))
            for j_c in (i_c - 1, i_c + 1):
                if 0 <= j_c < len(choices):
                    cand_c = copy.deepcopy(base_p)
                    cand_c[k_c] = choices[j_c]
                    neighbors.append(cand_c)
    for k_b in ("wl_revalidate",):
        if k_b in base_p:
            cand_b = copy.deepcopy(base_p)
            cand_b[k_b] = not cand_b[k_b]
            neighbors.append(cand_b)

    if "macro_active_exit" in base_p:
        cand_macro = copy.deepcopy(base_p)
        cand_macro["macro_active_exit"] = not cand_macro["macro_active_exit"]
        neighbors.append(cand_macro)

    if "wl_mode" in base_p:
        cand_wl = copy.deepcopy(base_p)
        cand_wl["wl_mode"] = "WL_STRONGEST_MOMENTUM" if cand_wl["wl_mode"] != "WL_STRONGEST_MOMENTUM" else "WL_NONE"
        neighbors.append(cand_wl)

    return neighbors


def evaluate_neighborhood_stability(params: Dict[str, Any], candidate_score: float, grid: MarketGrid,
                                     start_bar: int, end_bar: int, benchmark_cagr: float,
                                     max_neighbors: Optional[int] = None) -> Tuple[bool, float, float, List[float]]:
    neighbors = dynamic_parameter_neighbors(params)
    if max_neighbors is not None and len(neighbors) > max_neighbors:      # deterministic: same params -> same sample
        _seed = int(hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:8], 16)
        _pick = np.random.default_rng(_seed).choice(len(neighbors), size=max_neighbors, replace=False)
        neighbors = [neighbors[i] for i in sorted(_pick)]
    neighbor_scores = []
    for n_p in neighbors:
        raw_sig, n_entry, n_exit, n_macro, n_state = compile_signals_fast(grid, n_p)
        engine = BacktestEngine(params=n_p)
        res = engine.run_interval(grid, raw_sig, n_entry, n_exit, n_macro, start_bar, end_bar, state_mat=n_state)
        metrics = calculate_metrics(res, grid, start_bar, end_bar, benchmark_cagr)
        neighbor_scores.append(metrics["score"])

    scores_arr = np.array(neighbor_scores)
    valid_mask = scores_arr > 0.0
    pass_rate = float(np.mean(valid_mask))
    p10_score = float(np.percentile(scores_arr, 10))
    median_neighbor_score = float(np.median(scores_arr))

    drop_tolerance = max(1.5, NEIGHBORHOOD_DROP_LIMIT * abs(candidate_score))   # score points (= CAGR pts)
    is_stable = (
        pass_rate >= 0.80 and
        median_neighbor_score >= (candidate_score - drop_tolerance) and
        p10_score > 0.0
    )
    plateau_score = min(candidate_score, median_neighbor_score)
    return is_stable, plateau_score, pass_rate, neighbor_scores


def slice_market_grid(grid: MarketGrid, cutoff_bar: int) -> MarketGrid:
    """Return a physically truncated MarketGrid so indicators can be recomputed independently."""
    return MarketGrid(
        symbols=grid.symbols,
        dates=grid.dates[:cutoff_bar],
        open_mat=grid.open_mat[:, :cutoff_bar].copy(),
        high_mat=grid.high_mat[:, :cutoff_bar].copy(),
        low_mat=grid.low_mat[:, :cutoff_bar].copy(),
        close_mat=grid.close_mat[:, :cutoff_bar].copy(),
        volume_mat=grid.volume_mat[:, :cutoff_bar].copy(),
        atr14_mat=grid.atr14_mat[:, :cutoff_bar].copy(),
        dvol30_mat=grid.dvol30_mat[:, :cutoff_bar].copy(),
        adx14_mat=grid.adx14_mat[:, :cutoff_bar].copy(),
        delist_mat=grid.delist_mat[:, :cutoff_bar].copy(),
        alive_mat=grid.alive_mat[:, :cutoff_bar].copy(),
        macro_close=grid.macro_close[:cutoff_bar].copy(),
        macro_open=grid.macro_open[:cutoff_bar].copy(),
        months_arr=grid.months_arr[:cutoff_bar].copy(),
        years_arr=grid.years_arr[:cutoff_bar].copy(),
        bench_close=grid.bench_close[:cutoff_bar].copy(),
        brackets=list(grid.brackets), liq_floor=grid.liq_floor.copy(),
        slip_bps=grid.slip_bps.copy(), impact_bps=grid.impact_bps.copy()
    )


def verify_behavioral_causality(grid: MarketGrid):
    logging.info("Executing Complete 6-Archetype Prefix-Invariance Causality Audit (Comparing Row Data)...")
    n_bars = len(grid.dates)
    if n_bars < 500:
        return

    t_cutoff = n_bars - 80
    t_cutoff_date = grid.dates[t_cutoff]
    # IMPORTANT: indicators/signals must be recomputed on the shorter grid.
    # Slicing signal arrays from the full grid would make this audit tautological.
    grid_trunc = slice_market_grid(grid, t_cutoff)

    test_cfgs = [
        {"entry_type": 0, "entry_ma_len": 50, "entry_ma_type": 1, "use_market_macro_system": True,
         "macro_ma_len": 100, "macro_ma_type": 0, "macro_active_exit": False, "adx_thresh": 0.0,
         "max_pyramid_layers": 1, "wl_mode": "WL_NONE",
         "max_holding_bars": 25, "sl_mult": 3.0, "exit_type": 0, "trail_atr_mult": 0.0, "tranche_floor": 5000.0},
        {"entry_type": 1, "rsi_f_len": 14, "rsi_f_smt": 6, "rsi_s_len": 40, "rsi_s_smt": 10,
         "use_rsi_trend_filter": False, "use_market_macro_system": True, "macro_ma_len": 100,
         "macro_ma_type": 0, "macro_active_exit": True, "adx_thresh": 20.0,
         "max_pyramid_layers": 1, "wl_mode": "WL_DEEPEST_DISCOUNT",
         "be_trigger_atr": 1.5, "tranche_floor": 10000.0,
         "max_holding_bars": 35, "sl_mult": 4.5, "exit_type": 1, "trail_pct": 20.0, "trail_atr_mult": 0.0},
        {"entry_type": 2, "xover_short_len": 15, "xover_short_type": 0, "xover_long_len": 60, "xover_long_type": 0, "xover_gap": 45,
         "use_market_macro_system": True, "macro_ma_len": 100, "macro_ma_type": 0, "macro_active_exit": False, "adx_thresh": 0.0,
         "max_pyramid_layers": 1, "wl_mode": "WL_NONE",
         "max_holding_bars": 20, "sl_mult": 3.0, "exit_type": 5, "trail_atr_mult": 0.0,
         "exit_xover_short_len": 15, "exit_xover_short_type": 0, "exit_xover_long_len": 60, "exit_xover_long_type": 0, "exit_xover_gap": 45},
        {"entry_type": 3, "vol_ma_len": 20, "vol_mult": 2.5, "price_lookback": 20, "body_atr_mult": 0.8,
         "use_market_macro_system": True, "macro_ma_len": 100, "macro_ma_type": 0, "macro_active_exit": False, "adx_thresh": 0.0,
         "max_pyramid_layers": 1, "wl_mode": "WL_NONE",
         "max_holding_bars": 20, "sl_mult": 3.0, "exit_type": 6, "trail_atr_mult": 0.0,
         "exit_vol_ma_len": 20, "exit_vol_mult": 1.5},
        {"entry_type": 4, "bb_entry_len": 50, "bb_entry_std": 3.0, "use_market_macro_system": False,
         "adx_thresh": 25.0, "max_pyramid_layers": 2, "wl_mode": "WL_NONE",
         "max_holding_bars": 55, "sl_mult": 9.0, "be_trigger_atr": 2.5, "tranche_floor": 15000.0, "exit_type": 1, "trail_pct": 29.0, "trail_atr_mult": 0.0,
         "bb_exit_len": 20},
        {"entry_type": 5, "bp_vol_ma_len": 20, "bp_vol_mult": 2.0, "bp_ratio": 3.0, "use_market_macro_system": False,
         "adx_thresh": 25.0, "max_pyramid_layers": 1, "wl_mode": "WL_NONE",
         "max_holding_bars": 40, "sl_mult": 4.0, "exit_type": 1, "trail_pct": 20.0, "trail_atr_mult": 0.0, "bb_exit_len": 20},
    ]

    for cfg in test_cfgs:
        raw_sig, entry_full, exit_full, macro_full, state_full = compile_signals_fast(grid, cfg)
        engine_full = BacktestEngine(params=cfg)
        res_full = engine_full.run_interval(grid, raw_sig, entry_full, exit_full, macro_full, 300, t_cutoff, state_mat=state_full)

        raw_trunc, entry_trunc, exit_trunc, macro_trunc, state_trunc = compile_signals_fast(grid_trunc, cfg)
        engine_trunc = BacktestEngine(params=cfg)
        f_prior = res_full["trades"]
        if len(f_prior) > 0:
            f_prior = f_prior[(f_prior["reason"] != "END_OF_TEST") & (f_prior["exit_date"] < t_cutoff_date)].sort_values(by=["coin", "entry_date"]).reset_index(drop=True)

        res_trunc = engine_trunc.run_interval(
            grid_trunc, raw_trunc, entry_trunc, exit_trunc, macro_trunc,
            300, t_cutoff, state_mat=state_trunc
        )
        t_prior = res_trunc["trades"]
        if len(t_prior) > 0:
            t_prior = t_prior[(t_prior["reason"] != "END_OF_TEST") & (t_prior["exit_date"] < t_cutoff_date)].sort_values(by=["coin", "entry_date"]).reset_index(drop=True)

        if len(f_prior) != len(t_prior):
            raise RuntimeError(f"Causality Drift in Archetype {cfg['entry_type']}: Trade count mismatch.")

        for row_i in range(len(f_prior)):
            f_row = f_prior.iloc[row_i]
            t_row = t_prior.iloc[row_i]
            if (f_row["coin"] != t_row["coin"] or
                f_row["entry_date"] != t_row["entry_date"] or
                f_row["exit_date"] != t_row["exit_date"] or
                abs(f_row["pnl"] - t_row["pnl"]) > 0.01):
                raise RuntimeError(f"Row-level Causality Drift on trade {f_row['coin']} on {f_row['entry_date']}")

    logging.info("Complete 5-Archetype Prefix-Invariance Audit Passed (Exact Row-Level PnL Verified).")


# ==============================================================================
# OPTUNA HYPERPARAMETER TUNER (Safe Static Search Space)
# ==============================================================================
def sample_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    p = {}
    p["entry_type"] = trial.suggest_categorical("entry_type", [0, 1, 2, 3, 4, 5])   # V13: 3 = pure breakout, 5 = buying-pressure surge
    p["adx_thresh"] = trial.suggest_categorical("adx_thresh", [0.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0])

    if p["entry_type"] == 0:
        p["entry_ma_len"] = trial.suggest_int("entry_ma_len", 10, 200, step=5)
        p["entry_ma_type"] = trial.suggest_categorical("entry_ma_type", [0, 1, 3])
    elif p["entry_type"] == 1:
        p["rsi_f_len"] = trial.suggest_int("rsi_f_len", 10, 80, step=2)
        p["rsi_f_smt"] = trial.suggest_int("rsi_f_smt", 6, 40, step=2)
        p["rsi_s_len"] = trial.suggest_int("rsi_s_len", 10, 80, step=2)
        p["rsi_s_smt"] = trial.suggest_int("rsi_s_smt", 6, 40, step=2)
        p["use_rsi_trend_filter"] = trial.suggest_categorical("use_rsi_trend_filter", [True, False])
        if p["use_rsi_trend_filter"]:
            p["rsi_trend_ma_len"] = trial.suggest_int("rsi_trend_ma_len", 20, 200, step=10)
            p["rsi_trend_ma_type"] = trial.suggest_categorical("rsi_trend_ma_type", [0, 1, 3])
    elif p["entry_type"] == 2:
        p["xover_short_len"] = trial.suggest_int("xover_short_len", 10, 100, step=5)
        p["xover_short_type"] = trial.suggest_categorical("xover_short_type", [0, 1, 3])
        gap = trial.suggest_int("xover_gap", 10, 80, step=5)
        p["xover_gap"] = gap
        p["xover_long_len"] = min(200, p["xover_short_len"] + gap)
        p["xover_long_type"] = trial.suggest_categorical("xover_long_type", [0, 1, 3])
    elif p["entry_type"] == 3:
        p["price_lookback"] = trial.suggest_int("price_lookback", 10, 50, step=2)
        p["body_atr_mult"] = round(trial.suggest_float("body_atr_mult", 0.3, 2.0, step=0.1), 1)
        if trial.suggest_categorical("breakout_use_volume", [True, False]):
            p["vol_ma_len"] = trial.suggest_int("vol_ma_len", 10, 50, step=2)
            p["vol_mult"] = round(trial.suggest_float("vol_mult", 1.5, 4.5, step=0.1), 1)
        else:
            p["vol_mult"] = 0.0
    elif p["entry_type"] == 5:
        p["bp_vol_ma_len"] = trial.suggest_int("bp_vol_ma_len", 10, 50, step=2)
        p["bp_vol_mult"] = round(trial.suggest_float("bp_vol_mult", 1.0, 4.0, step=0.25), 2)
        p["bp_ratio"] = round(trial.suggest_float("bp_ratio", 1.5, 6.0, step=0.5), 1)
    elif p["entry_type"] == 4:
        p["bb_entry_len"] = trial.suggest_int("bb_entry_len", 10, 50, step=2)
        p["bb_entry_std"] = round(trial.suggest_float("bb_entry_std", 1.5, 3.0, step=0.1), 1)

    if FORCE_MACRO_REGIME_FILTER is not None:
        p["use_market_macro_system"] = FORCE_MACRO_REGIME_FILTER
    else:
        p["use_market_macro_system"] = trial.suggest_categorical("use_market_macro_system", [True, False])

    if p["use_market_macro_system"]:
        p["macro_ma_len"] = trial.suggest_int("macro_ma_len", 20, 250, step=10)
        p["macro_ma_type"] = trial.suggest_categorical("macro_ma_type", [0, 1, 3])
        p["macro_active_exit"] = trial.suggest_categorical("macro_active_exit", [True, False])
    else:
        p["macro_active_exit"] = False

    # V13: no slot cap. Sizing is a weight on equity that ramps while cash sits idle (see SIZE_* constants).
    p["size_w0"] = float(trial.suggest_categorical("size_w0", SIZE_W0_CHOICES))
    p["size_w_max"] = float(trial.suggest_categorical("size_w_max", SIZE_W_MAX_CHOICES))
    p["size_reserve"] = float(trial.suggest_categorical("size_reserve", SIZE_RESERVE_CHOICES))
    p["size_ramp"] = float(trial.suggest_categorical("size_ramp", SIZE_RAMP_CHOICES))
    # V13 engine rules the owner fixed (not searched): one watchlist entry per stock, no age cap, whole shares.
    p.update(V13_FIXED_RULES)
    p["wl_revalidate"] = trial.suggest_categorical("wl_revalidate", [True, False])
    if p["entry_type"] in (0, 3, 4, 5):
        p["max_pyramid_layers"] = trial.suggest_categorical("max_pyramid_layers", [1, 2, 3])
    else:
        p["max_pyramid_layers"] = 1
    p["avg_down_free_adds"] = trial.suggest_categorical("avg_down_free_adds", [0, 1]) if p["max_pyramid_layers"] > 1 else 0

    p["wl_mode"] = trial.suggest_categorical("wl_mode", [
        "WL_NONE", "WL_DEEPEST_DISCOUNT", "WL_STRONGEST_MOMENTUM"
    ])

    # Take-profit / partial-exit block removed in V11 (worse in 4/4 prior studies; caps surges; costly DP on small lots).
    p["tranche_floor"] = float(trial.suggest_categorical("tranche_floor", TRANCHE_FLOOR_CHOICES))

    p["be_trigger_atr"] = trial.suggest_categorical("be_trigger_atr", [0.0, 1.5, 2.5, 3.5])
    p["max_holding_bars"] = trial.suggest_int("max_holding_bars", 10, 250, step=5)
    p["sl_mult"] = round(trial.suggest_float("sl_mult", 2.0, 9.0, step=0.2), 1)

    if FORCE_TRAIL_STOP is True:
        p["exit_type"] = trial.suggest_categorical("exit_type", [0, 3, 4, 5, 6, 7])
        p["trail_pct"] = 0.0
        p["trail_atr_mult"] = 0.0
    else:
        p["exit_type"] = trial.suggest_categorical("exit_type", [0, 1, 3, 4, 5, 6, 7])
        if p["exit_type"] == 1:
            p["trail_pct"] = round(trial.suggest_float("trail_pct", 5.0, 25.0, step=1.0), 1)
            p["trail_atr_mult"] = 0.0
        else:
            p["trail_atr_mult"] = trial.suggest_categorical("trail_atr_mult", [0.0, 2.5, 3.5, 5.0, 6.5])

    if p["exit_type"] == 3:
        p["exit_ma_len"] = trial.suggest_int("exit_ma_len", 10, 150, step=5)
        p["exit_ma_type"] = trial.suggest_categorical("exit_ma_type", [0, 1, 3])
    elif p["exit_type"] == 4:
        p["exit_rsi_f_len"] = trial.suggest_int("exit_rsi_f_len", 10, 60, step=2)
        p["exit_rsi_f_smt"] = trial.suggest_int("exit_rsi_f_smt", 6, 30, step=2)
        p["exit_rsi_s_len"] = trial.suggest_int("exit_rsi_s_len", 10, 60, step=2)
        p["exit_rsi_s_smt"] = trial.suggest_int("exit_rsi_s_smt", 6, 30, step=2)
    elif p["exit_type"] == 5:
        p["exit_xover_short_len"] = trial.suggest_int("exit_xover_short_len", 10, 80, step=5)
        p["exit_xover_short_type"] = trial.suggest_categorical("exit_xover_short_type", [0, 1, 3])
        exit_gap = trial.suggest_int("exit_xover_gap", 10, 60, step=5)
        p["exit_xover_gap"] = exit_gap
        p["exit_xover_long_len"] = min(150, p["exit_xover_short_len"] + exit_gap)
        p["exit_xover_long_type"] = trial.suggest_categorical("exit_xover_long_type", [0, 1, 3])
    elif p["exit_type"] == 6:
        p["exit_vol_ma_len"] = trial.suggest_int("exit_vol_ma_len", 10, 40, step=5)
        p["exit_vol_mult"] = round(trial.suggest_float("exit_vol_mult", 0.8, 3.5, step=0.1), 1)
    elif p["exit_type"] == 7:
        p["bb_exit_len"] = trial.suggest_int("bb_exit_len", 10, 60, step=2)

    return p


def _canonical_func_source(fn) -> str:
    """Extracts an AST dump of a function, stripping docstrings, comments, and whitespace formatting."""
    try:
        source = inspect.getsource(fn)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                if (node.body and isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, (ast.Constant, ast.Str))):
                    node.body.pop(0)
        return ast.dump(tree)
    except Exception:
        return inspect.getsource(fn)


ENGINE_HASH_INPUT: str = (
    _canonical_func_source(sample_hyperparameters) +
    _canonical_func_source(calculate_metrics) +
    _canonical_func_source(compile_signals_fast) +
    _canonical_func_source(BacktestEngine) +
    _canonical_func_source(FastIndicators) +
    _canonical_func_source(calculate_benchmarks) +
    _canonical_func_source(compute_max_affordable_tranche) +
    _canonical_func_source(breakeven_exit_price) +
    _canonical_func_source(buy_sell_volume_split) +
    _canonical_func_source(buy_pressure_mask) +
    _canonical_func_source(repair_price_artifacts) +
    _canonical_func_source(_year_returns) +
    _canonical_func_source(compute_buy_cost_audited) +
    _canonical_func_source(compute_sell_proceeds_audited) +
    str([FORCE_MACRO_REGIME_FILTER, FORCE_TRAIL_STOP, START_YEAR, BROKERAGE_MODE, FLAT_BROKERAGE_INR, STT_RATE,
         EXCHANGE_TXN_RATE, STAMP_DUTY_RATE, SEBI_CHARGE_RATE, GST_RATE, DP_CHARGE_INR, BRACKET_LIQUIDITY_FLOOR_INR,
         BRACKET_SLIPPAGE_BPS, BRACKET_IMPACT_COEF_BPS, TRANCHE_FLOOR_CHOICES, MAX_POSITION_EQUITY_PCT,
         MAX_ADV_PARTICIPATION, DD_PENALTY_WEIGHT, MIN_CLOSED_TRADES, EOT_EXCESS_WEIGHT, ADV_WINDOW, ADV_MIN_PERIODS,
         INITIAL_CAPITAL, MONTHLY_CONTRIBUTION, CASH_ANNUAL_YIELD, ENGINE_DEFAULTS, SIZE_W0_CHOICES, SIZE_W_MAX_CHOICES, SIZE_RESERVE_CHOICES, SIZE_RAMP_CHOICES, SIZE_RUNT_FRACTION, V13_FIXED_RULES, PLATEAU_IN_LOOP, PLATEAU_TOP_QUANTILE, PLATEAU_MIN_COMPLETED, PLATEAU_MAX_NEIGHBORS, PLATEAU_PENALTY_WEIGHT, PLATEAU_PENALTY_CAP, BENCH_SYMBOL,
         OBJ_YEAR_EXCESS_CLIP, OBJ_MIN_YEAR_BARS, SPIKE_MIN_MOVE, CA_DROP_Q, CA_JUMP_Q, CA_GAP_EXPLAINS, CA_MARKET_DAY_MAX, MANUAL_CORPORATE_ACTIONS])
)
PARAM_SPACE_HASH: str = hashlib.sha256(ENGINE_HASH_INPUT.encode()).hexdigest()[:8]
STUDY_NAME = f"nse_swing_{ENGINE_VERSION.lower()}_{UNIVERSE_NAME.lower()}_{PARAM_SPACE_HASH}"


def reconstitute_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuilds derived crossover lengths and guarantees all default configuration keys."""
    p = copy.deepcopy(params)
    for k in list(p.keys()):
        if isinstance(p[k], float):
            p[k] = round(p[k], 2)

    if p.get("entry_type") == 2 and "xover_gap" in p:
        p["xover_long_len"] = min(200, p["xover_short_len"] + p["xover_gap"])
    if p.get("exit_type") == 5 and "exit_xover_gap" in p:
        p["exit_xover_long_len"] = min(200, p["exit_xover_short_len"] + p["exit_xover_gap"])

    # Fallback defaults for conditional parameters not sampled in trial.params
    p.setdefault("use_market_macro_system", False if FORCE_MACRO_REGIME_FILTER is None else FORCE_MACRO_REGIME_FILTER)
    p.setdefault("macro_active_exit", False)
    p.setdefault("max_pyramid_layers", 1)
    p.setdefault("adx_thresh", 0.0)
    p.setdefault("be_trigger_atr", 0.0)
    p.setdefault("trail_atr_mult", 0.0)
    p.setdefault("wl_mode", "WL_NONE")
    p.setdefault("tranche_floor", TRANCHE_FLOOR_INR)
    # V13: values the search does not record in trial.params (fixed rules, and options that are off by default)
    p.update(V13_FIXED_RULES)
    p.setdefault("avg_down_free_adds", 0)
    if p.get("entry_type") == 3 and not p.get("breakout_use_volume", True):
        p["vol_mult"] = 0.0                      # pure breakout: volume confirmation switched off
    for legacy in ("use_global_tp", "tp_mult", "tp_size_pct", "tp_move_sl_be"):
        p.pop(legacy, None)
    return p


# ==============================================================================
# TWR BETA, CAPTURE, ABLATIONS & PLACEBO SUITE
# ==============================================================================
def compute_twr_analytics(strategy_twr: np.ndarray, nsei_daily_ret: np.ndarray) -> Dict[str, float]:
    """Computes true daily strategy TWR correlation, beta, and capture ratios."""
    if len(strategy_twr) <= 2 or len(nsei_daily_ret) <= 2:
        return {"twr_corr": 0.0, "twr_beta": 0.0, "up_capture": np.nan, "down_capture": np.nan}

    strat_daily_ret = np.diff(strategy_twr) / np.maximum(1e-6, strategy_twr[:-1])
    min_len = min(len(strat_daily_ret), len(nsei_daily_ret))
    r_s = strat_daily_ret[:min_len]
    r_m = nsei_daily_ret[:min_len]

    cov_matrix = np.cov(r_s, r_m)
    var_m = cov_matrix[1, 1]
    cov_sm = cov_matrix[0, 1]
    twr_beta = float(cov_sm / max(1e-9, var_m))

    corr_matrix = np.corrcoef(r_s, r_m)
    twr_corr = float(corr_matrix[0, 1])

    up_idx = r_m > 0
    down_idx = r_m < 0

    up_capture = float((np.mean(r_s[up_idx]) / max(1e-9, np.mean(r_m[up_idx]))) * 100.0) if np.any(up_idx) else np.nan
    down_capture = float((np.mean(r_s[down_idx]) / min(-1e-9, np.mean(r_m[down_idx]))) * 100.0) if np.any(down_idx) else np.nan

    return {
        "twr_corr": float(np.nan_to_num(twr_corr, nan=0.0)),
        "twr_beta": float(np.nan_to_num(twr_beta, nan=0.0)),
        "up_capture": float(up_capture),
        "down_capture": float(down_capture),
    }


_PLACEBO_CTX: Dict[str, Any] = {}


def _placebo_init(ctx: Dict[str, Any]) -> None:
    _PLACEBO_CTX.clear()
    _PLACEBO_CTX.update(ctx)
    _PLACEBO_CTX["engine"] = BacktestEngine(params=ctx["p"])


def _placebo_one(seed_i: int) -> Tuple[int, Optional[float], float]:
    """One placebo seed (own RNG stream, so serial and parallel runs give identical numbers)."""
    c = _PLACEBO_CTX
    grid, start_bar, end_bar = c["grid"], c["start_bar"], c["end_bar"]
    rng = np.random.default_rng([42, seed_i])
    random_raw = rng.random((len(grid.symbols), end_bar - start_bar)) < c["prob"]
    placebo_raw = np.zeros((len(grid.symbols), len(grid.dates)), dtype=bool)
    placebo_raw[:, start_bar:end_bar] = random_raw & grid.alive_mat[:, start_bar:end_bar]
    uncond = None
    if seed_i < c["n_uncond"]:
        uncond_entry = placebo_raw & c["macro_ok"] & c["liq_ok"] & grid.alive_mat
        res = c["engine"].run_interval(grid, placebo_raw, uncond_entry, c["exit_mat"], c["macro_ok"], start_bar, end_bar, state_mat=c["state_mat"])
        uncond = calculate_metrics(res, grid, start_bar, end_bar, c["bench"])["full"]["cagr"]
    cond_entry = placebo_raw & c["macro_ok"] & c["adx_ok"] & c["liq_ok"] & grid.alive_mat
    res = c["engine"].run_interval(grid, placebo_raw, cond_entry, c["exit_mat"], c["macro_ok"], start_bar, end_bar, state_mat=c["state_mat"])
    return seed_i, uncond, calculate_metrics(res, grid, start_bar, end_bar, c["bench"])["full"]["cagr"]


@logged_phase("placebo suite")
def run_matched_placebo_suite(grid: MarketGrid, p: Dict[str, Any], total_bars: int,
                              start_bar: int, end_bar: int, benchmark_cagr: float,
                              total_raw_triggers: int, n_seeds: int = 100, n_seeds_uncond: int = 30) -> Dict[str, Any]:
    """Unconditioned and conditioned random-entry placebos; keeps the full empirical distributions. Seeds run in parallel."""
    n_syms = len(grid.symbols)
    adx_t = p.get("adx_thresh", 0.0)
    _, _, exit_mat, macro_ok, state_mat = compile_signals_fast(grid, p)
    ctx = {
        "grid": grid, "p": p, "start_bar": start_bar, "end_bar": end_bar, "bench": benchmark_cagr,
        "prob": max(0.001, total_raw_triggers / float(total_bars * n_syms)), "n_uncond": n_seeds_uncond,
        "exit_mat": exit_mat, "macro_ok": macro_ok, "state_mat": state_mat,
        "liq_ok": grid.dvol30_mat >= np.maximum(grid.liq_floor, 1.0)[:, None],
        "adx_ok": (grid.adx14_mat >= adx_t) if adx_t > 0.0 else True,
    }
    workers = max(1, min(PLACEBO_WORKERS, (os.cpu_count() or 2) - 1, n_seeds))
    rows = None
    if workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                     initializer=_placebo_init, initargs=(ctx,)) as ex:
                rows = list(ex.map(_placebo_one, range(n_seeds), chunksize=max(1, n_seeds // (workers * 4))))
        except Exception as e:  # parallel is an optimisation only
            logging.warning(f"Parallel placebo failed ({type(e).__name__}: {e}); running serially.")
            rows = None
    if rows is None:
        _placebo_init(ctx)
        rows = [_placebo_one(i) for i in range(n_seeds)]
    rows.sort(key=lambda r: r[0])
    unconditioned_cagrs = [r[1] for r in rows if r[1] is not None]
    conditioned_cagrs = [r[2] for r in rows]

    return {
        "unconditioned": (
            float(np.median(unconditioned_cagrs)),
            float(np.percentile(unconditioned_cagrs, 5)),
            float(np.percentile(unconditioned_cagrs, 95))
        ),
        "conditioned": (
            float(np.median(conditioned_cagrs)),
            float(np.percentile(conditioned_cagrs, 5)),
            float(np.percentile(conditioned_cagrs, 95))
        ),
        "unconditioned_samples": unconditioned_cagrs,
        "conditioned_samples": conditioned_cagrs,
    }


@logged_phase("trailing-stop ablation")
def run_trailing_stop_ablation(grid: MarketGrid, p: Dict[str, Any], is_start: int, is_end: int,
                               oos_start: int, oos_end: int, is_bench_cagr: float,
                               oos_bench_cagr: float) -> str:
    """Evaluates the contribution of trailing stop parameters vs a pure fixed stop."""
    is_trail_active = (p.get("exit_type") == 1 and p.get("trail_pct", 0.0) > 0.0) or (p.get("trail_atr_mult", 0.0) > 0.0)
    if not is_trail_active:
        return "*Trailing Stop Ablation not applicable: The winning configuration does not employ a trailing stop mechanism.*\n"

    ablated_params = copy.deepcopy(p)
    if ablated_params.get("exit_type") == 1:
        ablated_params["exit_type"] = 0
        ablated_params["trail_pct"] = 0.0
    ablated_params["trail_atr_mult"] = 0.0

    raw_sig, entry_mat, exit_mat, macro_ok, state_mat = compile_signals_fast(grid, p)
    raw_sig_abl, entry_abl, exit_abl, macro_abl, state_abl = compile_signals_fast(grid, ablated_params)

    eng_base = BacktestEngine(params=p)
    eng_abl = BacktestEngine(params=ablated_params)

    r_is_base = eng_base.run_interval(grid, raw_sig, entry_mat, exit_mat, macro_ok, is_start, is_end, state_mat=state_mat)
    r_is_abl = eng_abl.run_interval(grid, raw_sig_abl, entry_abl, exit_abl, macro_abl, is_start, is_end, state_mat=state_abl)
    m_is_base = calculate_metrics(r_is_base, grid, is_start, is_end, is_bench_cagr)
    m_is_abl = calculate_metrics(r_is_abl, grid, is_start, is_end, is_bench_cagr)

    r_oos_base = eng_base.run_interval(grid, raw_sig, entry_mat, exit_mat, macro_ok, oos_start, oos_end, state_mat=state_mat)
    r_oos_abl = eng_abl.run_interval(grid, raw_sig_abl, entry_abl, exit_abl, macro_abl, oos_start, oos_end, state_mat=state_abl)
    m_oos_base = calculate_metrics(r_oos_base, grid, oos_start, oos_end, oos_bench_cagr, min_trades=0)
    m_oos_abl = calculate_metrics(r_oos_abl, grid, oos_start, oos_end, oos_bench_cagr, min_trades=0)

    active_trail_desc = f"trail_pct={p.get('trail_pct')}%" if p.get("exit_type") == 1 else f"trail_atr_mult={p.get('trail_atr_mult')}x"

    return f"""| Window | Configuration | Strategy CAGR | Max DD | Profit Factor | Completed Trades | Realized PnL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **In-Sample** | **Winner ({active_trail_desc})** | **{m_is_base['full']['cagr']:.2f}%** | **{m_is_base['max_dd']:.2f}%** | **{m_is_base['full']['profit_factor']:.2f}** | **{m_is_base['full']['trades']}** | **{format_price(m_is_base['full']['net_pnl'])}** |
| In-Sample | Ablated (No Trailing Stop) | {m_is_abl['full']['cagr']:.2f}% | {m_is_abl['max_dd']:.2f}% | {m_is_abl['full']['profit_factor']:.2f} | {m_is_abl['full']['trades']} | {format_price(m_is_abl['full']['net_pnl'])} |
| **Out-of-Sample** | **Winner ({active_trail_desc})** | **{m_oos_base['full']['cagr']:.2f}%** | **{m_oos_base['max_dd']:.2f}%** | **{m_oos_base['full']['profit_factor']:.2f}** | **{m_oos_base['full']['trades']}** | **{format_price(m_oos_base['full']['net_pnl'])}** |
| Out-of-Sample | Ablated (No Trailing Stop) | {m_oos_abl['full']['cagr']:.2f}% | {m_oos_abl['max_dd']:.2f}% | {m_oos_abl['full']['profit_factor']:.2f} | {m_oos_abl['full']['trades']} | {format_price(m_oos_abl['full']['net_pnl'])} |
"""


# ==============================================================================
# AUDIT AND FORENSIC REPORTING PIPELINE
# ==============================================================================
# ==============================================================================
# V11 HELPERS: data sanity, cached audit, timing robustness, gates, paste block
# ==============================================================================
from types import SimpleNamespace

STAGE2_N: int = 8            # candidates that receive the expensive neighbor + timing battery


def write_data_sanity(provenance_records: List[Dict[str, Any]], grid: MarketGrid) -> str:
    df = pd.DataFrame(provenance_records)
    try:
        df.to_csv(SANITY_CSV_FILE, index=False)
    except Exception:
        pass
    requested, _ = load_universe_with_brackets(UNIVERSE_NAME)
    missing = sorted(set(requested) - set(grid.symbols))
    jumpy = df[df["max_1d_move_pct"] > 40.0].sort_values("max_1d_move_pct", ascending=False) if len(df) else df
    short = df[df["total_bars"] < 750] if len(df) else df
    by_b = df.groupby("bracket").size().to_dict() if len(df) else {}
    logging.info(f"Universe: {len(grid.symbols)} of {len(requested)} constituents have usable history | brackets {by_b} | "
                 f"{len(missing)} missing | {len(jumpy)} with a >40% one-day move (check for splits/demergers) | {len(short)} short-history")
    if len(jumpy):
        logging.warning("Symbols with a >40% one-day close-to-close move (verify data adjustment): " +
                        ", ".join(f"{r.symbol} {r.max_1d_move_pct:.0f}% on {r.max_1d_move_date}" for r in jumpy.head(12).itertuples()))
    md = f"* Constituents requested: **{len(requested)}** | usable: **{len(grid.symbols)}** | by bracket: {by_b}\n"
    md += f"* No usable Yahoo history for **{len(missing)}** symbols" + (f": {', '.join(missing[:40])}{' ...' if len(missing) > 40 else ''}" if missing else "") + "\n"
    md += f"* Short history (<750 bars): **{len(short)}** symbols\n"
    if len(jumpy):
        md += "* **Check for split/demerger price artifacts** (one-day move > 40%): " + ", ".join(
            f"{r.symbol} ({r.max_1d_move_pct:.0f}% on {r.max_1d_move_date})" for r in jumpy.head(15).itertuples()) + "\n"
    else:
        md += "* No one-day moves above 40% found.\n"
    md += f"* Full per-symbol table: `{SANITY_CSV_FILE.name}`\n"
    return md


def run_causality_audit_cached(grid: MarketGrid) -> None:
    try:
        if AUDIT_STAMP_FILE.exists() and AUDIT_STAMP_FILE.read_text().strip() == PARAM_SPACE_HASH:
            logging.info("Causality audit already passed for this engine hash; skipping re-run.")
            return
    except Exception:
        pass
    verify_behavioral_causality(grid)
    try:
        AUDIT_STAMP_FILE.write_text(PARAM_SPACE_HASH)
    except Exception:
        pass


@logged_phase("timing robustness")
def run_timing_robustness(grid: MarketGrid, params: Dict[str, Any], start_bar: int, end_bar: int) -> Dict[str, Any]:
    """Re-runs the config with the monthly deposit landing on different trading days and with shifted start dates."""
    raw, ent, ext, mac, st = compile_signals_fast(grid, params)
    eng = BacktestEngine(params=params)
    rows = []
    for shift in TIMING_START_SHIFTS:
        s = start_bar + shift
        if end_bar - s < 200:
            continue
        bench = calculate_benchmarks(grid, s, end_bar, mac)
        for off in TIMING_CONTRIB_OFFSETS:
            res = eng.run_interval(grid, raw, ent, ext, mac, s, end_bar, state_mat=st, contrib_offset=off)
            m = calculate_metrics(res, grid, s, end_bar, bench)
            rows.append({"shift": shift, "offset": off, "cagr": m["full"]["cagr"], "bench": bench["nsei_twr_cagr"],
                         "excess": m["excess_cagr"], "score": m["score"], "trades": m["full"]["trades"]})
    if not rows:
        return {"rows": [], "p10_score": -10.0, "median_score": -10.0, "worst_excess": -99.0, "median_excess": -99.0, "ok": False}
    sc = np.array([r["score"] for r in rows]); ex = np.array([r["excess"] for r in rows])
    return {"rows": rows, "p10_score": float(np.percentile(sc, 10)), "median_score": float(np.median(sc)),
            "worst_excess": float(ex.min()), "median_excess": float(np.median(ex)), "ok": bool(np.percentile(sc, 10) > 0.0)}


def segment_from_continuous(cont: Dict[str, Any], grid: MarketGrid, is_start: int, seg_start: int, seg_end: int) -> Dict[str, Any]:
    """OOS as a slice of ONE continuous run (positions carried in, no restart, no forced liquidation at the split)."""
    tw = cont["twr_curve"]
    off_s, off_e = seg_start - is_start, seg_end - is_start
    base = tw[off_s - 1] if off_s >= 1 else 1.0
    seg = tw[off_s:off_e] / max(1e-9, base)
    n = len(seg)
    if n < 2:
        return {"cagr": 0.0, "max_dd": 0.0, "trades": 0, "net_pnl": 0.0, "bars": n}
    cagr = (seg[-1] ** (248.0 / n) - 1.0) * 100.0
    pk = np.maximum.accumulate(seg)
    dd = float(np.max((pk - seg) / np.maximum(1e-9, pk))) * 100.0
    tdf = cont["trades"]
    if len(tdf):
        xd = pd.to_datetime(tdf["exit_date"])
        mask = (xd >= grid.dates[seg_start]) & (xd <= grid.dates[seg_end - 1]) & (tdf["reason"] != "END_OF_TEST")
        nt, pnl = int(mask.sum()), float(tdf.loc[mask, "pnl"].sum())
    else:
        nt, pnl = 0, 0.0
    return {"cagr": float(cagr), "max_dd": dd, "trades": nt, "net_pnl": pnl, "bars": n}


def missed_swing_md(grid: MarketGrid, signal_log: List[Tuple[int, int, int, str]], horizon: int = 20) -> str:
    """Consecutive-day signals on the same symbol are ONE swing episode (a 'close > MA' state fires every day)."""
    if not signal_log:
        return "*No signals recorded.*\n"
    df = pd.DataFrame(signal_log, columns=["sid", "c_i", "sbar", "outcome"]).sort_values(["c_i", "sbar"]).reset_index(drop=True)
    df["ep"] = ((df["c_i"] != df["c_i"].shift()) | ((df["sbar"] - df["sbar"].shift()) > 1)).cumsum()
    order = ["executed", "cash_starved", "scrip_risk_cap", "expired_unfilled", "revalidation_dropped",
             "anti_averaging_down", "pyramid_blocked", "wl_duplicate", "unaffordable", "liquidity_blocked", "adx_blocked", "macro_blocked", "open_at_end"]
    prio = {o: i for i, o in enumerate(order)}
    df["prio"] = df["outcome"].map(prio).fillna(len(order))
    eps = df.groupby("ep").agg(c_i=("c_i", "first"), sbar=("sbar", "first"), prio=("prio", "min"), days=("sid", "count")).reset_index()
    eps["outcome"] = eps["prio"].map({i: o for o, i in prio.items()}).fillna("other")
    n_bars = len(grid.dates)
    fwd = []
    for r in eps.itertuples():
        e = min(int(r.sbar) + horizon, n_bars - 1)
        px0 = grid.open_mat[int(r.c_i), int(r.sbar)]
        fwd.append(grid.close_mat[int(r.c_i), e] / px0 - 1.0 if (e > r.sbar and np.isfinite(px0) and px0 > 0) else np.nan)
    eps["fwd"] = fwd
    eps = eps.dropna(subset=["fwd"])
    if eps.empty:
        return "*No evaluable swing episodes.*\n"
    g = eps.groupby("outcome")["fwd"].agg(["count", "mean", "median", lambda x: (x > 0.10).mean()])
    g.columns = ["n", "mean", "median", "big"]
    md = (f"*A swing episode = a run of consecutive daily signals on one symbol; its best disposition and the {horizon}-bar forward "
          f"return from its first day are shown.*\n\n| Best Disposition of the Episode | Episodes | Mean {horizon}-bar Fwd Return | Median | Share > +10% |\n| :--- | :--- | :--- | :--- | :--- |\n")
    for o in order:
        if o in g.index:
            r = g.loc[o]
            md += f"| {o} | {int(r['n'])} | {r['mean']*100:+.2f}% | {r['median']*100:+.2f}% | {r['big']*100:.1f}% |\n"
    big = eps[eps["fwd"] > 0.10]
    if len(big) > 0:
        md += f"\n* Of **{len(big)}** swing episodes that ran > +10% within {horizon} bars, **{(big['outcome'] == 'executed').mean() * 100:.1f}%** were bought "
        md += f"(**{(big['outcome'].isin(['cash_starved', 'scrip_risk_cap', 'expired_unfilled'])).mean() * 100:.1f}%** were lost to capital limits, "
        md += f"**{(big['outcome'].isin(['adx_blocked', 'liquidity_blocked', 'macro_blocked'])).mean() * 100:.1f}%** to the entry filters).\n"
    cap = eps[eps["outcome"].isin(["cash_starved", "scrip_risk_cap", "expired_unfilled"])]
    ex = eps[eps["outcome"] == "executed"]
    if len(cap) > 20 and len(ex) > 20:
        verdict = ("the ranking leaves better swings on the table" if cap["fwd"].mean() > ex["fwd"].mean() else "the ranking is choosing at least as well as the average dropped episode")
        md += f"* Capital-driven drops average **{cap['fwd'].mean()*100:+.2f}%** vs **{ex['fwd'].mean()*100:+.2f}%** for bought episodes: {verdict}.\n"
    return md


def top_trades_md(trades: pd.DataFrame, n: int = 10) -> str:
    if len(trades) == 0:
        return "*No trades.*\n"
    tot = float(trades["pnl"].sum())
    t = trades.sort_values("pnl", ascending=False).head(n)
    ev = {}
    for r in DATA_REPAIR_LOG:
        ev.setdefault(r["symbol"], []).append(pd.Timestamp(r["date"]))
    md = "| Symbol | Entry | Exit | Bars | Return | Net PnL | Share of Total | Exit Reason | Data-event nearby |\n| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for _, r in t.iterrows():
        near = any(abs((pd.Timestamp(r["entry_date"]) - d).days) <= 5 or abs((pd.Timestamp(r["exit_date"]) - d).days) <= 5
                   for d in ev.get(r["coin"], []))
        md += (f"| {r['coin']} | {pd.Timestamp(r['entry_date']).date()} | {pd.Timestamp(r['exit_date']).date()} | {int(r['bars'])} | "
               f"{r['ret']*100:+.1f}% | {format_price(r['pnl'])} | {(r['pnl']/tot*100 if tot > 0 else 0):.1f}% | {r['reason']} | {'YES - review' if near else 'no'} |\n")
    if tot > 0:
        _c = trades['pnl'] / trades['entry_equity'].clip(lower=1.0)
        _ct = float(_c.sum())
        if _ct > 0:
            md += (f"\n* Same, each trade measured as % of equity at entry (size-neutral): **{_c.nlargest(1).sum()/_ct:.1%} / {_c.nlargest(3).sum()/_ct:.1%} / "
                   f"{_c.nlargest(5).sum()/_ct:.1%} / {_c.nlargest(10).sum()/_ct:.1%}**.\n")
        md += (f"\n* Top-1 / top-3 / top-5 / top-10 trades (rupees) = **{trades['pnl'].nlargest(1).sum()/tot:.1%} / {trades['pnl'].nlargest(3).sum()/tot:.1%} / "
               f"{trades['pnl'].nlargest(5).sum()/tot:.1%} / {trades['pnl'].nlargest(10).sum()/tot:.1%}** of total profit.\n")
    return md


def repairs_md() -> str:
    if not DATA_REPAIR_LOG:
        return "* No data repairs were needed.\n"
    df = pd.DataFrame(DATA_REPAIR_LOG)
    cnt = df["type"].value_counts().to_dict()
    md = f"* Repairs by type: {cnt}. Full list in `{REPAIRS_CSV_FILE.name}`.\n\n"
    md += "| Type | Symbol | Date | Close ratio | Detail |\n| :--- | :--- | :--- | :--- | :--- |\n"
    show = df[df["type"] != "LARGE_MOVE_KEPT"].head(25).to_dict("records") + df[df["type"] == "LARGE_MOVE_KEPT"].head(15).to_dict("records")
    for r in show:
        md += f"| {r['type']} | {r['symbol']} | {r['date']} | {r['ratio']:.3f} | {r['detail']} |\n"
    return md


def objective_sensitivity_md(m: Dict[str, Any]) -> str:
    ye = np.array(m.get("year_excess_raw", []), dtype=float)
    if len(ye) == 0:
        return "*Not available.*\n"
    md = "| Year-excess clip (pts) | DD weight 0.15 | DD weight 0.25 | DD weight 0.40 |\n| :--- | :--- | :--- | :--- |\n"
    over_dd = max(0.0, m["max_dd"] - m["bench_max_dd"])
    for clip in (15.0, 25.0, 40.0):
        core = float(np.mean(np.clip(ye, -clip, clip)))
        md += f"| {clip:.0f} | " + " | ".join(f"{core - w * over_dd:+.2f}" for w in (0.15, 0.25, 0.40)) + " |\n"
    return md + f"\n* Unclipped mean yearly excess vs the better of NIFTYBEES/basket: **{float(np.mean(ye)):+.2f}** pts over {len(ye)} years; years beating that bar: **{int((ye > 0).sum())} of {len(ye)}**.\n"


def bracket_breakdown_md(grid: MarketGrid, trades: pd.DataFrame) -> str:
    if len(trades) == 0:
        return "*No trades.*\n"
    bmap = dict(zip(grid.symbols, grid.brackets))
    t = trades.copy()
    t["bracket"] = t["coin"].map(bmap)
    g = t.groupby("bracket").agg(trades=("pnl", "count"), net=("pnl", "sum"), win=("pnl", lambda x: (x > 0).mean() * 100.0),
                                 avg_ret=("ret", lambda x: x.mean() * 100.0), avg_cost=("cost_inr", "mean"))
    md = "| Bracket | Trades | Net PnL | Win Rate | Avg Return / Trade | Avg Ticket |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for b in ("LARGE", "MID", "SMALL"):
        if b in g.index:
            r = g.loc[b]
            md += f"| {b} | {int(r['trades'])} | {format_price(r['net'])} | {r['win']:.1f}% | {r['avg_ret']:+.2f}% | {format_price(r['avg_cost'])} |\n"
    return md


def fee_reality_md() -> str:
    md = "| Ticket | Buy-side charges | Sell-side charges (incl. DP) | Round trip | % of ticket |\n| :--- | :--- | :--- | :--- | :--- |\n"
    for tk in (5000, 7500, 10000, 15000, 25000, 50000, 100000):
        buy_total, _ = compute_buy_cost_audited(float(tk), 0.0)
        net, _ = compute_sell_proceeds_audited(float(tk), 0.0, apply_dp=True)
        b, s = buy_total - tk, tk - net
        md += f"| {format_price(tk)} | {format_price(b)} | {format_price(s)} | {format_price(b + s)} | {(b + s) / tk * 100:.2f}% |\n"
    return md


def build_paste_block(best_params: Dict[str, Any], status: str, failing: List[str], seed_agree: int) -> str:
    core_defaults = {"adx_thresh": 0.0, "use_market_macro_system": False, "macro_active_exit": False, "max_pyramid_layers": 1,
                     "wl_mode": "WL_NONE", "be_trigger_atr": 0.0, "trail_atr_mult": 0.0, "tranche_floor": TRANCHE_FLOOR_INR}
    cfg = dict(best_params)
    for k, v in core_defaults.items():
        cfg.setdefault(k, v)
    order = ["entry_type", "adx_thresh"] + [k for k in cfg if k not in ("entry_type", "adx_thresh", "exit_type")] + ["exit_type"]
    seen, lines = set(), []
    for k in order:
        if k in cfg and k not in seen:
            seen.add(k)
            lines.append(f"    {k!r}: {cfg[k]!r},")
    hdr = (f"# ===== PASTE INTO lv_nse: replace the whole DEFAULT_CONFIG dict =====\n"
           f"# status={status}  engine={ENGINE_VERSION}  universe={UNIVERSE_NAME}  param_hash={PARAM_SPACE_HASH}  seeds_agreeing={seed_agree}\n"
           f"# generated={pd.Timestamp.now().isoformat(timespec='seconds')}\n")
    if failing:
        hdr += "# FAILED GATES: " + "; ".join(failing) + "\n"
    hdr += (f"# bracket floors (median 30d value): {BRACKET_LIQUIDITY_FLOOR_INR} | slippage bps: {BRACKET_SLIPPAGE_BPS} | impact: {BRACKET_IMPACT_COEF_BPS}\n"
            f"CONFIG_ENGINE_VERSION = {ENGINE_VERSION!r}\nCONFIG_PARAM_HASH = {PARAM_SPACE_HASH!r}\n")
    return hdr + "DEFAULT_CONFIG = {\n" + "\n".join(lines) + "\n}\n"


def eot_share_multi_cut(grid: MarketGrid, eng: Any, sigs: Tuple[Any, ...], is_start: int, is_end: int, base_share: float,
                        base_organic: Optional[float] = None, base_basket: Optional[float] = None, base_headline: Optional[float] = None) -> Dict[str, Any]:
    """V12.2/V12.4: how much of the in-sample result depends on unclosed terminal marks, with the IS window ending at the real cut AND at earlier cuts.
    A long trend trade straddling one particular cut date must not decide promotion on its own.
    rows     = (cut back-shift, rupee share of IS profit that is an unclosed mark)           [V12.2 measure, now information]
    org_rows = (cut back-shift, organic CAGR, survivor-basket CAGR, headline CAGR)           [V12.4 gate measure, return terms]"""
    raw_sig, entry_m, exit_m, macro_m, state_m = sigs
    rows: List[Tuple[int, float]] = []
    org_rows: List[Tuple[int, float, float, float]] = []
    for b in EOT_CUT_BACKSHIFTS:
        e = is_end - int(b)
        if e - is_start < 400:
            continue
        if b == 0 and base_organic is not None and base_basket is not None:
            rows.append((0, float(base_share)))
            org_rows.append((0, float(base_organic), float(base_basket), float(base_headline if base_headline is not None else base_organic)))
            continue
        try:
            res = eng.run_interval(grid, raw_sig, entry_m, exit_m, macro_m, is_start, e, state_mat=state_m)
            bm = calculate_benchmarks(grid, is_start, e, macro_m)
            m = calculate_metrics(res, grid, is_start, e, bm, min_trades=0)
            rows.append((int(b), float(m["eot_profit_share"])))
            org_rows.append((int(b), float(m["cagr_organic"]), float(bm["basket_twr_cagr"]), float(m["full"]["cagr"])))
        except Exception as exc:   # pragma: no cover - defensive
            logging.warning(f"eot_share_multi_cut: cut -{b} failed: {exc}")
    vals = [v for _, v in rows]
    return {"rows": rows, "median": float(np.median(vals)) if vals else float(base_share), "max": float(max(vals)) if vals else float(base_share),
            "org_rows": org_rows,
            "org_median": float(np.median([r[1] for r in org_rows])) if org_rows else float("nan"),
            "basket_median": float(np.median([r[2] for r in org_rows])) if org_rows else float("nan"),
            "headline_median": float(np.median([r[3] for r in org_rows])) if org_rows else float("nan")}


def evaluate_gates(*, oos_slice: Dict[str, Any], oos_bench_cagr: float, oos_basket_cagr: float, is_cond_pct: float,
                   oos_cond_pct: float, is_metrics: Dict[str, Any], leave5_cagr: float, is_bench_cagr: float,
                   cont_trades: pd.DataFrame, cont_year: Dict[int, float], idx_year: Dict[int, float],
                   cont_max_dd: float, cont_bench_dd: float, stable: bool, timing_ok: bool, seed_agree: int,
                   fresh_oos_cagr: Optional[float] = None, fresh_oos_trades: Optional[int] = None, eot_multi: Optional[Dict[str, Any]] = None,
                   nb_pass_rate: Optional[float] = None, timing_p10: Optional[float] = None, n_timing: Optional[int] = None,
                   fresh_oos_dd: Optional[float] = None, oos_basket_dd: Optional[float] = None, oos_index_dd: Optional[float] = None) -> List[Tuple[str, bool, str]]:
    gates: List[Tuple[str, bool, str]] = []
    gates.append(("Continuous-slice OOS beats NIFTYBEES", oos_slice["cagr"] > oos_bench_cagr,
                  f"OOS {oos_slice['cagr']:.2f}% vs benchmark {oos_bench_cagr:.2f}%"))
    gates.append(("Continuous-slice OOS beats the equal-weight survivor basket of the same universe (survivorship-neutral)",
                  oos_slice["cagr"] > oos_basket_cagr, f"OOS {oos_slice['cagr']:.2f}% vs basket {oos_basket_cagr:.2f}%"))
    if fresh_oos_cagr is not None and GATE_FRESH_OOS_MUST_BEAT_INDEX:
        gates.append(("Fresh-start OOS (cold restart on the OOS window, starting capital) beats NIFTYBEES", fresh_oos_cagr > oos_bench_cagr,
                      f"fresh OOS {fresh_oos_cagr:.2f}% vs benchmark {oos_bench_cagr:.2f}% (survivor basket {oos_basket_cagr:.2f}%)"
                      + (f"; {fresh_oos_trades} trades" if fresh_oos_trades is not None else "")))
    if GATE_FRESH_OOS_DD_VS_BASKET is not None and fresh_oos_dd is not None and oos_basket_dd is not None:
        gates.append((f"Fresh-start OOS max drawdown <= {GATE_FRESH_OOS_DD_VS_BASKET:.2f}x the survivor basket's",
                      fresh_oos_dd <= GATE_FRESH_OOS_DD_VS_BASKET * max(1.0, oos_basket_dd),
                      f"{fresh_oos_dd:.1f}% vs basket {oos_basket_dd:.1f}%" + (f" (index {oos_index_dd:.1f}%)" if oos_index_dd is not None else "")))
    gates.append((f"IS placebo rank >= {GATE_PLACEBO_IS_PCT:.0f}th", is_cond_pct >= GATE_PLACEBO_IS_PCT, f"{is_cond_pct:.1f}th percentile"))
    gates.append((f"OOS placebo rank >= {GATE_PLACEBO_OOS_PCT:.0f}th", oos_cond_pct >= GATE_PLACEBO_OOS_PCT, f"{oos_cond_pct:.1f}th percentile"))
    if eot_multi and eot_multi.get("rows"):
        _share_txt = ", ".join(f"{'actual' if b_ == 0 else '-' + str(b_) + ' bars'} {s_:.1%}" for b_, s_ in eot_multi["rows"])
        if GATE_EOT_MODE == "share" or not eot_multi.get("org_rows"):
            gates.append((f"Unclosed terminal marks < {GATE_MAX_EOT_PROFIT_SHARE:.0%} of IS profit (median over {len(eot_multi['rows'])} cut dates)",
                          eot_multi["median"] < GATE_MAX_EOT_PROFIT_SHARE,
                          f"median {eot_multi['median']:.1%}, worst {eot_multi['max']:.1%}; by cut: " + _share_txt))
        else:
            _org = eot_multi["org_rows"]
            gates.append((f"Organic IS CAGR (open gains at {EOT_EXCESS_WEIGHT:.0%} credit) still beats the survivor basket (median over {len(_org)} cut dates)",
                          eot_multi["org_median"] > eot_multi["basket_median"],
                          f"organic median {eot_multi['org_median']:.1f}% vs basket {eot_multi['basket_median']:.1f}% (headline {eot_multi['headline_median']:.1f}%); "
                          f"by cut (organic vs basket): " + ", ".join(f"{'actual' if b_ == 0 else '-' + str(b_) + ' bars'} {o_:.1f}% vs {k_:.1f}%" for b_, o_, k_, _h in _org)
                          + f"; rupee share of profit still open (information): median {eot_multi['median']:.1%}, worst {eot_multi['max']:.1%}"))
    else:
        gates.append((f"Unclosed terminal marks < {GATE_MAX_EOT_PROFIT_SHARE:.0%} of IS profit", is_metrics["eot_profit_share"] < GATE_MAX_EOT_PROFIT_SHARE,
                      f"{is_metrics['eot_profit_share']:.1%}"))
    gates.append(("Leave-top-5-out IS CAGR still beats benchmark", leave5_cagr > is_bench_cagr, f"{leave5_cagr:.2f}% vs {is_bench_cagr:.2f}%"))

    # Year concentration on FLOW-STRIPPED log returns (rupee profit is biased toward late years in a compounding account)
    lr = {y: float(np.log1p(max(-0.99, r / 100.0))) for y, r in cont_year.items()}
    tot_lr = sum(lr.values())
    if tot_lr > 0 and lr:
        top_y = max(lr, key=lr.get)
        share = lr[top_y] / tot_lr
        gates.append((f"No single year > {GATE_MAX_YEAR_PROFIT_SHARE:.0%} of total log-return", share <= GATE_MAX_YEAR_PROFIT_SHARE,
                      f"best year {top_y} = {share:.1%}"))
    else:
        gates.append((f"No single year > {GATE_MAX_YEAR_PROFIT_SHARE:.0%} of total log-return", False, "no positive total return"))

    common = [y for y in cont_year if y in idx_year]
    if common:
        n_beat = sum(1 for y in common if cont_year[y] > idx_year[y])
        gates.append((f"Beats NIFTYBEES in >= {GATE_MIN_YEARS_BEATING:.0%} of calendar years (year-on-year goal)",
                      n_beat / len(common) >= GATE_MIN_YEARS_BEATING, f"{n_beat} of {len(common)} years"))
    else:
        gates.append((f"Beats NIFTYBEES in >= {GATE_MIN_YEARS_BEATING:.0%} of calendar years", False, "no comparable years"))

    # Lottery guard in PERCENT terms: each trade counts as its profit divided by portfolio equity when it was opened, so the
    # late, large positions of a compounding account do not dominate by size alone. The rupee reading is shown for reference.
    _lbl = f"Lottery guard: top-5 trades <= {GATE_MAX_TOP5_TRADE_SHARE:.0%} of total profit (each trade as % of equity at entry)"
    tot_pnl = float(cont_trades["pnl"].sum()) if len(cont_trades) else 0.0
    contrib = (cont_trades["pnl"] / cont_trades["entry_equity"].clip(lower=1.0)) if len(cont_trades) else pd.Series(dtype=float)
    tot_c = float(contrib.sum())
    if tot_c > 0:
        top5 = float(contrib.nlargest(5).sum() / tot_c)
        top5_inr = float(cont_trades["pnl"].nlargest(5).sum() / tot_pnl) if tot_pnl > 0 else float("nan")
        gates.append((_lbl, top5 <= GATE_MAX_TOP5_TRADE_SHARE, f"{top5:.1%} (in rupees it would read {top5_inr:.1%})"))
    else:
        gates.append((_lbl, False, "no positive total profit"))

    gates.append((f"Max drawdown <= {GATE_MAX_DD_VS_INDEX:.2f}x benchmark's", cont_max_dd <= GATE_MAX_DD_VS_INDEX * max(1.0, cont_bench_dd),
                  f"{cont_max_dd:.1f}% vs benchmark {cont_bench_dd:.1f}%"))
    gates.append(("Neighborhood stable (all active params +/- 1 step, MA types, floor)", stable,
                  (f"{nb_pass_rate:.0%} of neighbors beat the benchmark net of DD penalty (rule: >= 80%, median not far below the candidate, p10 > 0)"
                   if nb_pass_rate is not None else "rule: >= 80% of neighbors beat the benchmark, median not far below the candidate, p10 > 0")))
    gates.append(("Deposit-timing / start-date robust (p10 score > 0)", timing_ok,
                  (f"p10 score {timing_p10:+.2f} over {n_timing} timing variants" if timing_p10 is not None and n_timing is not None else "timing variants")))
    gates.append((f"Seeds agreeing on the strategy family >= {GATE_MIN_SEED_AGREEMENT}", seed_agree >= GATE_MIN_SEED_AGREEMENT, f"{seed_agree} of {len(SEEDS)} seeds"))
    return gates


def wasted_by_choice_md(allt: List[Any]) -> str:
    """V12.4: for each choice, how many trials fall below the trade floor (truly wasted) and how many traded enough but scored <= -5 (poor configs).
    Uses the closed-trade count recorded per trial; older studies without it fall back to a single 'score <= -5' column (cause not recorded)."""
    if not allt:
        return ""
    known = all(getattr(t, "n_closed", None) is not None for t in allt)
    rows_ = []
    for t in allt:
        d = {**t.params}
        v = float(t.value)
        if known:
            d["_floor"] = bool(int(t.n_closed) < MIN_CLOSED_TRADES)
            d["_poor"] = bool(v <= -5.0 and not d["_floor"])
        else:
            d["_le5"] = bool(v <= -5.0)
        rows_.append(d)
    df = pd.DataFrame(rows_)
    def _bin(col: str, edges: List[float], labels: List[str]) -> None:
        if col in df.columns:
            df[col + "_band"] = pd.cut(pd.to_numeric(df[col], errors="coerce"), bins=edges, labels=labels, include_lowest=True)
    _bin("max_holding_bars", [0, 60, 120, 180, 1000], ["10-60 bars", "61-120 bars", "121-180 bars", "181+ bars"])
    if known:
        md = (f"\n**Failure share by choice** ('Below trade floor' = fewer than {MIN_CLOSED_TRADES} closed trades, truly wasted; 'Poor' = traded enough but scored <= -5, "
              f"i.e. a genuinely bad configuration):\n\n| Parameter | Choice | Trials | Below trade floor | Poor (<= -5) | Poor share |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n")
    else:
        md = ("\n**Score <= -5 by choice** (this study did not record trade counts, so the cause is NOT known: a score <= -5 is either a trade-floor failure or a genuinely "
              "poor configuration):\n\n| Parameter | Choice | Trials | Score <= -5 | Share |\n| :--- | :--- | :--- | :--- | :--- |\n")
    for col in ("entry_type", "exit_type", "adx_thresh", "size_w0", "size_reserve", "size_ramp", "max_holding_bars_band", "tranche_floor"):
        if col not in df.columns:
            continue
        sub = df.dropna(subset=[col])
        if len(sub) < 20:
            continue
        if known:
            g = sub.groupby(col, observed=True)[["_floor", "_poor"]].agg(["count", "sum"])
            for ch_, r_ in g.iterrows():
                n_ = int(r_[("_floor", "count")])
                if n_ == 0:
                    continue
                md += f"| {col.replace('_band', '')} | {ch_} | {n_} | {int(r_[('_floor', 'sum')])} | {int(r_[('_poor', 'sum')])} | {r_[('_poor', 'sum')] / n_:.0%} |\n"
        else:
            g = sub.groupby(col, observed=True)["_le5"].agg(["count", "sum"])
            for ch_, r_ in g.iterrows():
                if int(r_["count"]) == 0:
                    continue
                md += f"| {col.replace('_band', '')} | {ch_} | {int(r_['count'])} | {int(r_['sum'])} | {r_['sum'] / r_['count']:.0%} |\n"
    return md


def search_diagnostics_md(seed_trials: Dict[int, List[Any]], seed_studies: Dict[int, Any]) -> str:
    """Is the SEARCH healthy? (edge pile-ups, wasted trials, convergence, what actually drives the score)"""
    allt = [t for trs in seed_trials.values() for t in trs]
    if not allt:
        return "*No trials.*\n"
    vals = np.array([t.value for t in allt])
    known = all(getattr(t, "n_closed", None) is not None for t in allt)
    pos = float(np.mean(vals > 0.0))
    if known:
        nc = np.array([int(t.n_closed) for t in allt])
        floor_fail = nc < MIN_CLOSED_TRADES
        wasted = float(np.mean(floor_fail))
        poor = float(np.mean((vals <= -5.0) & ~floor_fail))
        mid = float(np.mean((vals > -5.0) & (vals <= 0.0)))
        md = (f"* Trials: **{len(allt)}** over {len(seed_trials)} seeds. Below the {MIN_CLOSED_TRADES}-closed-trade floor (truly wasted budget): **{wasted:.0%}** | "
              f"traded enough but scored <= -5 (genuinely poor configurations): **{poor:.0%}** | scored between -5 and 0: **{mid:.0%}** | beating the benchmark: **{pos:.0%}**.\n")
        if wasted > 0.30:
            md += "* **WARNING:** more than 30% of trials never trade enough to be scored. Some part of the space (very high ADX, very high floor, very few slots, very long holds) starves the strategy of trades; see the table below before narrowing it.\n"
    else:
        le5 = float(np.mean(vals <= -5.0))
        mid = float(np.mean((vals > -5.0) & (vals <= 0.0)))
        md = (f"* Trials: **{len(allt)}** over {len(seed_trials)} seeds. Score <= -5: **{le5:.0%}** (trade-floor failures AND genuinely poor configurations; this study did not record "
              f"trade counts, so the split is unknown) | scored between -5 and 0: **{mid:.0%}** | beating the benchmark: **{pos:.0%}**.\n")

    md += wasted_by_choice_md(allt)

    md += "\n**Convergence** (best score found after X% of each seed's trials; if the last column is still rising the budget is too small):\n\n| Seed | 25% | 50% | 75% | 100% |\n| :--- | :--- | :--- | :--- | :--- |\n"
    for sd, trs in seed_trials.items():
        trs_o = sorted(trs, key=lambda z: z.tnum)
        if not trs_o:
            continue
        cells = []
        for frac in (0.25, 0.5, 0.75, 1.0):
            k = max(1, int(round(len(trs_o) * frac)))
            cells.append(f"{max(x.value for x in trs_o[:k]):+.2f}")
        md += f"| {sd} | " + " | ".join(cells) + " |\n"

    ok = [t for t in allt if t.value > -5.0]
    if len(ok) >= 40:
        top = sorted(ok, key=lambda x: x.value, reverse=True)[:max(30, int(0.05 * len(ok)))]
        dists: Dict[str, Any] = {}
        for t in allt:
            for k, d_ in t.dists.items():
                dists.setdefault(k, d_)
        rows = []
        for k, d_ in dists.items():
            vs = [t.params[k] for t in top if k in t.params]
            if len(vs) < 10:
                continue
            if hasattr(d_, "choices"):
                if k.endswith("_type") or k in ("wl_mode", "use_rsi_trend_filter"):
                    continue   # V12.1: nominal labels have no 'upper limit'; only ordinal/numeric choices are range-checked
                ch = list(d_.choices)
                if len(ch) < 3 or not all(isinstance(c_, (int, float)) and not isinstance(c_, bool) for c_ in ch):
                    continue
                pos_ = np.array([ch.index(v) / (len(ch) - 1) for v in vs if v in ch])
            else:
                lo, hi = float(d_.low), float(d_.high)
                if hi <= lo:
                    continue
                pos_ = np.array([(float(v) - lo) / (hi - lo) for v in vs])
            if len(pos_) < 10:
                continue
            up, dn = float(np.mean(pos_ >= 0.9)), float(np.mean(pos_ <= 0.1))
            flag = "PILING AT UPPER LIMIT - widen or check data/objective" if up >= 0.4 else ("PILING AT LOWER LIMIT - widen or check data/objective" if dn >= 0.4 else "ok")
            rows.append((k, float(np.median([float(v) for v in vs])), up, dn, flag))
        md += f"\n**Range check** (where the best {len(top)} trials sit inside each numeric range):\n\n| Parameter | Median of best | In top 10% of range | In bottom 10% of range | Verdict |\n| :--- | :--- | :--- | :--- | :--- |\n"
        for k, med, up, dn, flag in sorted(rows, key=lambda r: -max(r[2], r[3])):
            md += f"| {k} | {med:.4g} | {up:.0%} | {dn:.0%} | {flag} |\n"

        df = pd.DataFrame([{**t.params, "value": t.value} for t in ok])
        for col in ("entry_type", "exit_type", "wl_mode", "use_market_macro_system", "tranche_floor", "max_pyramid_layers", "size_w0", "size_w_max", "size_reserve", "size_ramp", "wl_revalidate", "avg_down_free_adds"):
            if col in df.columns and df[col].notna().sum() > 20:
                g = df.dropna(subset=[col]).groupby(col)["value"].agg(["count", "median", lambda x: x.quantile(0.9)])
                md += f"\n**{col}** (score by choice, scored trials only)\n\n| Choice | Trials | Median | p90 |\n| :--- | :--- | :--- | :--- |\n"
                for ch_, r_ in g.iterrows():
                    md += f"| {ch_} | {int(r_['count'])} | {r_['median']:+.2f} | {r_.iloc[2]:+.2f} |\n"

    try:
        import optuna.importance as oimp
        acc: Dict[str, List[float]] = {}
        for sd, st in seed_studies.items():
            imp = oimp.get_param_importances(st, target=lambda t_: max(float(t_.value), -5.0))
            for k, v in imp.items():
                acc.setdefault(k, []).append(float(v))
        if acc:
            top_imp = sorted(((k, float(np.mean(v))) for k, v in acc.items()), key=lambda kv: -kv[1])[:10]
            md += "\n**Parameter importance** (fANOVA, averaged over seeds; what actually moves the score):\n\n| Parameter | Importance |\n| :--- | :--- |\n"
            for k, v in top_imp:
                md += f"| {k} | {v:.1%} |\n"
    except Exception as exc:
        md += f"\n*Parameter importance unavailable ({type(exc).__name__}); `pip install scikit-learn` enables it.*\n"
    return md


def _placebo_pct(cagr_val: float, dist: List[float]) -> float:
    if not dist:
        return 50.0
    return float((np.sum(np.array(dist) < cagr_val) / len(dist)) * 100.0)


def prescreen_gates(grid: MarketGrid, rec: Dict[str, Any], is_start: int, is_end: int, oos_start: int, oos_end: int, n_bars: int,
                    full: bool = False, store: Optional[Dict[str, Any]] = None) -> Tuple[int, List[str]]:
    """V12.1-V12.3: run the promotion gates for ONE shortlisted candidate.
    full=False (default): every gate except the three expensive ones (both placebo gates, leave-top-5). Those three are NOT evaluated and are
      NOT counted: the return is (gates passed among the EVALUATED gates, names of failed gates); evaluated = passed + len(failed).
    full=True (V12.3): also runs the IS/OOS placebo suites and the leave-top-5-out run, so ALL gates are evaluated. The placebo suites are put
      into `store` (keys placebo_is / placebo_oos) so the final report can reuse them instead of recomputing.
    Never raises: on any error the candidate gets (0, ['prescreen error'])."""
    try:
        p = rec["params"]
        raw_sig, entry_m, exit_m, macro_m, state_m = compile_signals_fast(grid, p)
        eng = BacktestEngine(params=p)
        isb = calculate_benchmarks(grid, is_start, is_end, macro_m)
        oosb = calculate_benchmarks(grid, oos_start, oos_end, macro_m)
        isr = eng.run_interval(grid, raw_sig, entry_m, exit_m, macro_m, is_start, is_end, state_mat=state_m)
        ism = calculate_metrics(isr, grid, is_start, is_end, isb)
        cr = eng.run_interval(grid, raw_sig, entry_m, exit_m, macro_m, is_start, n_bars, state_mat=state_m)
        sl = segment_from_continuous(cr, grid, is_start, oos_start, oos_end)
        cb = calculate_benchmarks(grid, is_start, n_bars, macro_m)
        tw = cr["twr_curve"]
        pk = np.maximum.accumulate(tw)
        cdd = float(np.max((pk - tw) / np.maximum(1e-9, pk))) * 100.0 if len(tw) else 0.0
        cy = _year_returns(tw, grid.years_arr[is_start:n_bars])
        fr = eng.run_interval(grid, raw_sig, entry_m, exit_m, macro_m, oos_start, oos_end, state_mat=state_m)
        frm = calculate_metrics(fr, grid, oos_start, oos_end, oosb, min_trades=0)
        eotm = eot_share_multi_cut(grid, eng, (raw_sig, entry_m, exit_m, macro_m, state_m), is_start, is_end, float(ism["eot_profit_share"]),
                                  base_organic=float(ism["cagr_organic"]), base_basket=float(isb["basket_twr_cagr"]), base_headline=float(ism["full"]["cagr"]))

        is_cond_pct, oos_cond_pct, leave5 = 100.0, 100.0, float("inf")
        if full:
            is_pl = run_matched_placebo_suite(grid, p, is_end - is_start, is_start, is_end, isb, isr["funnel_stats"]["raw_triggers"],
                                              n_seeds=PLACEBO_SEEDS_CONDITIONED, n_seeds_uncond=PLACEBO_SEEDS_UNCONDITIONED)
            oos_pl = run_matched_placebo_suite(grid, p, oos_end - oos_start, oos_start, oos_end, oosb, fr["funnel_stats"]["raw_triggers"],
                                               n_seeds=PLACEBO_SEEDS_CONDITIONED, n_seeds_uncond=PLACEBO_SEEDS_UNCONDITIONED)
            is_cond_pct = _placebo_pct(ism["full"]["cagr"], is_pl["conditioned_samples"])
            oos_cond_pct = _placebo_pct(frm["full"]["cagr"], oos_pl["conditioned_samples"])
            mask = np.ones(len(grid.symbols), dtype=bool)
            if len(isr["trades"]) >= 10:
                top5 = isr["trades"].groupby("coin")["pnl"].sum().sort_values(ascending=False).head(5).index.tolist()
                for s_ in top5:
                    if s_ in grid.symbols:
                        mask[grid.symbols.index(s_)] = False
            r5 = eng.run_interval(grid, raw_sig, entry_m, exit_m, macro_m, is_start, is_end, active_symbols_mask=mask, state_mat=state_m)
            leave5 = float(calculate_metrics(r5, grid, is_start, is_end, isb)["full"]["cagr"])
            if store is not None:
                store["placebo_is"], store["placebo_oos"] = is_pl, oos_pl

        g = evaluate_gates(
            oos_slice=sl, oos_bench_cagr=oosb["nsei_twr_cagr"], oos_basket_cagr=oosb["basket_twr_cagr"],
            is_cond_pct=is_cond_pct, oos_cond_pct=oos_cond_pct, is_metrics=ism, leave5_cagr=leave5,
            is_bench_cagr=isb["nsei_twr_cagr"], cont_trades=cr["trades"], cont_year=cy, idx_year=cb["idx_year"],
            cont_max_dd=cdd, cont_bench_dd=cb["nsei_max_dd"], stable=bool(rec["stable"]),
            timing_ok=bool(rec["timing"] and rec["timing"]["ok"]), seed_agree=int(rec["seed_agree"]),
            fresh_oos_cagr=float(frm["full"]["cagr"]), fresh_oos_trades=int(len(fr["trades"])), eot_multi=eotm,
            fresh_oos_dd=float(frm["max_dd"]), oos_basket_dd=float(oosb["basket_max_dd"]), oos_index_dd=float(oosb["nsei_max_dd"]))
        if not full:   # V12.3: the three skipped gates are not evaluated, so they are neither passed nor failed
            g = [x for x in g if not (x[0].startswith("IS placebo") or x[0].startswith("OOS placebo") or x[0].startswith("Leave-top-5"))]
        return int(sum(1 for x in g if x[1])), [x[0] for x in g if not x[1]]
    except Exception as exc:   # pragma: no cover - defensive
        logging.warning(f"prescreen_gates failed for trial {rec['trial'].number}: {exc}")
        return 0, ["prescreen error"]

def run_optimization():
    logging.info(f"Initializing NSE Framework {ENGINE_VERSION}: Universe = {UNIVERSE_NAME}, Start Year = {START_YEAR}")
    logging.info(f"Parameter Space Signature: {PARAM_SPACE_HASH} | Seeds: {SEEDS} x {N_TRIALS_PER_SEED} trials")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    grid, universe, macro_df, provenance_records = build_market_universe(UNIVERSE_NAME, START_YEAR, FORCE_REFRESH)
    sanity_md = write_data_sanity(provenance_records, grid)
    universe_sig = hashlib.sha256("|".join(f"{s}:{b}" for s, b in zip(grid.symbols, grid.brackets)).encode()).hexdigest()[:6]

    run_causality_audit_cached(grid)

    n_bars = len(grid.dates)
    mature_coverage = np.sum(grid.alive_mat, axis=0)
    mature_pct = mature_coverage / float(len(grid.symbols))
    valid_start_indices = np.where(mature_pct >= 0.51)[0]

    trading_start_bar = max(int(valid_start_indices[0]) if len(valid_start_indices) > 0 else WARMUP_BARS, WARMUP_BARS)

    total_trading_bars = n_bars - trading_start_bar
    split_offset = int(total_trading_bars * 0.75)
    is_start, is_end = trading_start_bar, trading_start_bar + split_offset
    oos_start, oos_end = is_end, n_bars
    is_bars, oos_bars = is_end - is_start, oos_end - oos_start

    logging.info(
        f"Real NSE Calendar: Total Bars = {n_bars} | "
        f"Trading Starts = {grid.dates[trading_start_bar].date()} (Bar {trading_start_bar}) | "
        f"IS Window = {is_bars} sessions [{grid.dates[is_start].date()} -> {grid.dates[is_end-1].date()}] | "
        f"OOS Window = {oos_bars} sessions [{grid.dates[oos_start].date()} -> {grid.dates[oos_end-1].date()}]"
    )

    rough_macro_ma = FastIndicators.moving_average(grid.macro_close, 200, 0)
    rough_macro_ok = (~np.isnan(rough_macro_ma)) & (grid.macro_close > rough_macro_ma)

    is_bench = calculate_benchmarks(grid, is_start, is_end, rough_macro_ok)
    oos_bench = calculate_benchmarks(grid, oos_start, oos_end, rough_macro_ok)

    # ---- Objective: in-sample score, with a soft plateau penalty for trials that reach the study's top decile ----
    def objective(trial: optuna.Trial) -> float:
        params = sample_hyperparameters(trial)
        raw_sig, entry_mat, exit_mat, macro_ok, state_mat = compile_signals_fast(grid, params)
        res = BacktestEngine(params=params).run_interval(grid, raw_sig, entry_mat, exit_mat, macro_ok, is_start, is_end, state_mat=state_mat)
        trial.set_user_attr("n_closed", int(len(res["trades"])))   # lets the diagnostics separate floor failures from poor configs
        raw = float(calculate_metrics(res, grid, is_start, is_end, is_bench)["score"])
        trial.set_user_attr("raw_score", raw)
        if not PLATEAU_IN_LOOP or raw <= 0.0:
            return raw
        done_vals = [t.value for t in trial.study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        if len(done_vals) < PLATEAU_MIN_COMPLETED or raw < float(np.quantile(done_vals, PLATEAU_TOP_QUANTILE)):
            return raw
        _, plateau, _, _ = evaluate_neighborhood_stability(params, raw, grid, is_start, is_end, is_bench, max_neighbors=PLATEAU_MAX_NEIGHBORS)
        penalty = min(PLATEAU_PENALTY_WEIGHT * max(0.0, raw - plateau), PLATEAU_PENALTY_CAP * abs(raw))
        trial.set_user_attr("plateau_score", float(plateau))
        trial.set_user_attr("plateau_penalty", float(penalty))
        return raw - penalty

    seed_trials: Dict[int, List[Any]] = {}
    seed_studies: Dict[int, Any] = {}
    for seed in SEEDS:
        s_name = f"{STUDY_NAME}_{universe_sig}_s{seed}"
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=seed, n_startup_trials=N_STARTUP_TRIALS)
        study = optuna.create_study(study_name=s_name, storage=STUDY_DB, load_if_exists=True, direction="maximize", sampler=sampler)
        done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_remaining = max(0, N_TRIALS_PER_SEED - done)
        if n_remaining > 0:
            logging.info(f"[seed {seed}] running {n_remaining} trials in study '{s_name}'...")
            study.optimize(objective, n_trials=n_remaining, show_progress_bar=True)
        else:
            logging.info(f"[seed {seed}] loaded completed study with {done} trials.")
        seed_studies[seed] = study
        _pen = [t.user_attrs["plateau_penalty"] for t in study.trials if "plateau_penalty" in t.user_attrs]
        if _pen:
            logging.info(f"[seed {seed}] plateau check ran on {len(_pen)} top-decile trials; mean penalty {np.mean(_pen):.2f} score points, max {np.max(_pen):.2f}")
        seed_trials[seed] = [
            SimpleNamespace(number=f"s{seed}-{t.number}", seed=seed, value=float(t.value), params=dict(t.params),
                            dists=dict(t.distributions), tnum=int(t.number), n_closed=t.user_attrs.get("n_closed"))
            for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]

    def _family(prm: Dict[str, Any]) -> Tuple[Any, Any, bool]:
        return (prm.get("entry_type"), prm.get("exit_type"), bool(prm.get("use_market_macro_system", False)))

    seed_top_fams = {
        sd: {_family(t.params) for t in sorted([x for x in trs if x.value > -5.0], key=lambda z: z.value, reverse=True)[:25]}
        for sd, trs in seed_trials.items()
    }

    pool = sorted([t for trs in seed_trials.values() for t in trs if t.value > -5.0], key=lambda z: z.value, reverse=True)
    if not pool:
        pool = sorted([t for trs in seed_trials.values() for t in trs], key=lambda z: z.value, reverse=True)[:1]
    selected, seen_keys, bucket_counts = [], set(), {}
    for t in pool:
        key = json.dumps(t.params, sort_keys=True, default=str)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        fam4 = _family(t.params) + (t.params.get("tranche_floor"),)
        if bucket_counts.get(fam4, 0) >= 4:       # keep the shortlist diverse, not 25 clones of one region
            continue
        bucket_counts[fam4] = bucket_counts.get(fam4, 0) + 1
        selected.append(t)
        if len(selected) >= TOP_K_POSTHOC:
            break
    logging.info(f"Shortlisted {len(selected)} distinct candidates from {len(pool)} scored trials across {len(SEEDS)} seeds.")

    # ---- Stage 1: three disjoint temporal folds for every shortlisted candidate ----
    step_f = total_trading_bars // 3
    f1_s, f1_e = trading_start_bar, trading_start_bar + step_f
    f2_s, f2_e = trading_start_bar + step_f, trading_start_bar + 2 * step_f
    f3_s, f3_e = trading_start_bar + 2 * step_f, n_bars

    scored_candidates = []
    for t_cand in selected:
        p_cand = reconstitute_params(t_cand.params)
        cand_eng = BacktestEngine(params=p_cand)
        raw_c, ent_c, ext_c, mac_c, st_c = compile_signals_fast(grid, p_cand)
        fold_vals = []
        for (fs, fe) in ((f1_s, f1_e), (f2_s, f2_e), (f3_s, f3_e)):
            b_f = calculate_benchmarks(grid, fs, fe, mac_c)
            r_f = cand_eng.run_interval(grid, raw_c, ent_c, ext_c, mac_c, fs, fe, state_mat=st_c)
            m_f = calculate_metrics(r_f, grid, fs, fe, b_f)
            fold_vals.append((m_f["full"]["cagr"], b_f["nsei_twr_cagr"]))
        (c1, i1), (c2, i2), (c3, i3) = fold_vals
        fam = _family(t_cand.params)
        scored_candidates.append({
            "trial": t_cand, "params": p_cand, "score": t_cand.value,
            "beats_all": bool(c1 > i1 and c2 > i2 and c3 > i3), "beats_recent": bool(c3 > i3), "recent_margin": float(c3 - i3),
            "c1": c1, "c2": c2, "c3": c3, "i1": i1, "i2": i2, "i3": i3,
            "seed_agree": int(sum(1 for sd in SEEDS if fam in seed_top_fams.get(sd, set()))),
            "stable": None, "plateau": None, "pass_rate": None, "timing": None,
        })

    scored_candidates.sort(key=lambda x: (x["beats_all"], x["beats_recent"], x["recent_margin"], x["score"]), reverse=True)

    # ---- Stage 2: full neighborhood (all active params, MA types, floor) + deposit-timing battery ----
    stage2 = scored_candidates[:STAGE2_N]
    for rec in stage2:
        logging.info(f"Stage-2 battery for {rec['trial'].number} (score {rec['score']:.2f}) ...")
        is_stable, plateau, pass_rate, _ = evaluate_neighborhood_stability(
            params=rec["params"], candidate_score=rec["score"], grid=grid,
            start_bar=is_start, end_bar=is_end, benchmark_cagr=is_bench)
        rec["stable"], rec["plateau"], rec["pass_rate"] = bool(is_stable), float(plateau), float(pass_rate)
        rec["timing"] = run_timing_robustness(grid, rec["params"], is_start, is_end)

    for rec in stage2:
        rec["gates_passed"], rec["gates_failed"] = prescreen_gates(grid, rec, is_start, is_end, oos_start, oos_end, n_bars)
        logging.info(f"Pre-screen trial {rec['trial'].number}: {rec['gates_passed']} of {rec['gates_passed'] + len(rec['gates_failed'])} cheap gates passed; failing: {rec['gates_failed']}")

    def _checks(r):
        return int(bool(r["stable"])) + int(bool(r["timing"] and r["timing"]["ok"])) + int(r["beats_all"]) + int(r["seed_agree"] >= GATE_MIN_SEED_AGREEMENT)

    def _rank_key(r):
        # V12.3: (fully clean on ALL gates, fewest known failing gates, passed checks, score). Candidates that failed a cheap gate keep the V12.2 order among themselves.
        full_done = "full_failed" in r
        failed = r["full_failed"] if full_done else r["gates_failed"]
        return (int(full_done and not failed), -len(failed), _checks(r), r["score"])

    stage2.sort(key=_rank_key, reverse=True)
    # V12.3: full gate set (placebos + leave-top-5) for the best cheap-gate-clean candidates, so a clean runner-up can win
    if FULL_GATE_TOP_N > 0:
        for rec in [r for r in stage2 if not r["gates_failed"]][:FULL_GATE_TOP_N]:
            _store: Dict[str, Any] = {}
            _fp, _ff = prescreen_gates(grid, rec, is_start, is_end, oos_start, oos_end, n_bars, full=True, store=_store)
            rec["full_passed"], rec["full_failed"], rec["cache"] = _fp, _ff, _store
            logging.info(f"Full-gate test trial {rec['trial'].number}: {_fp} of {_fp + len(_ff)} gates passed; failing: {_ff}")
        stage2.sort(key=_rank_key, reverse=True)
    best_candidate_record = stage2[0]
    best_trial = best_candidate_record["trial"]
    best_params = best_candidate_record["params"]

    n_beat_all = sum(1 for c in scored_candidates if c["beats_all"])
    n_beat_recent = sum(1 for c in scored_candidates if c["beats_recent"])
    pool_size = len(scored_candidates)
    n_stable = sum(1 for c in stage2 if c["stable"])
    _full_done = [r_ for r_ in stage2 if "full_failed" in r_]
    selection_diagnosis_msg = (
        f"{n_beat_all} of {pool_size} shortlisted candidates beat the benchmark in all 3 folds ({n_beat_recent} in the latest fold); "
        f"{n_stable} of {len(stage2)} stage-2 candidates passed the neighborhood test. "
        f"Champion {best_trial.number} chosen by (1) being clean on ALL gates when the full test was run, (2) fewest known failing gates, (3) number of passed checks (stable / timing-robust / all-folds / seed-agreement), (4) score. "
        f"Cheap pre-screen (every gate except the placebo and leave-top-5 ones; passed of evaluated): " + "; ".join(
            f"#{r_['trial'].number}: {r_.get('gates_passed', 0)}/{r_.get('gates_passed', 0) + len(r_.get('gates_failed', []))}"
            + (f" fails [{'; '.join(_g.split(' (')[0].split(':')[0][:60].strip() for _g in r_['gates_failed'])}]" if r_.get("gates_failed") else "")
            for r_ in stage2) + ". "
        + (("Full gate test (placebo and leave-top-5 included) was run for " + str(len(_full_done)) + " candidate(s) that cleared every cheap gate: " + "; ".join(
            f"#{r_['trial'].number}: {r_['full_passed']}/{r_['full_passed'] + len(r_['full_failed'])}"
            + (f" fails [{'; '.join(_g.split(' (')[0].split(':')[0][:60].strip() for _g in r_['full_failed'])}]" if r_["full_failed"] else " ALL PASS") for r_ in _full_done) + ".")
           if _full_done else "No candidate cleared every cheap gate, so the full test was not run and no candidate can be promoted from this shortlist.")
    )
    logging.info(f"Champion selection: {selection_diagnosis_msg}")
    logging.info(f"Optimization Complete. Champion search score: {best_trial.value:.4f}")
    logging.info("Final validation now runs (placebo suites, ablation, timing battery, report). Each phase logs its own time below.")


    raw_sig_final, entry_final, exit_final, macro_final, state_final = compile_signals_fast(grid, best_params)
    final_engine = BacktestEngine(params=best_params)

    is_bench = calculate_benchmarks(grid, is_start, is_end, macro_final)
    oos_bench = calculate_benchmarks(grid, oos_start, oos_end, macro_final)

    # 1. RUN IN-SAMPLE
    is_results = final_engine.run_interval(grid, raw_sig_final, entry_final, exit_final, macro_final, is_start, is_end, state_mat=state_final)
    is_metrics = calculate_metrics(is_results, grid, is_start, is_end, is_bench)

    # 2. RUN FRESH OUT-OF-SAMPLE
    oos_results = final_engine.run_interval(grid, raw_sig_final, entry_final, exit_final, macro_final, oos_start, oos_end, state_mat=state_final)
    oos_metrics = calculate_metrics(oos_results, grid, oos_start, oos_end, oos_bench, min_trades=0)

    # 3. RUN UNIFIED CONTINUOUS BACKTEST FOR MULTI-CRISIS STRESS AUDIT
    continuous_results = final_engine.run_interval(grid, raw_sig_final, entry_final, exit_final, macro_final, is_start, n_bars, state_mat=state_final, record_signals=True)
    cont_trades = continuous_results["trades"]
    cont_twr = continuous_results["twr_curve"]

    stress_rows_md = ""
    for s_name, (s_start_str, s_end_str) in STRESS_WINDOWS.items():
        s_start_ts, s_end_ts = pd.Timestamp(s_start_str), pd.Timestamp(s_end_str)
        s_indices = np.where((grid.dates >= s_start_ts) & (grid.dates <= s_end_ts))[0]
        if len(s_indices) > 20:
            b_s, b_e = int(s_indices[0]), int(s_indices[-1]) + 1
            if b_s >= is_start:
                s_off_s, s_off_e = b_s - is_start, b_e - is_start
                s_twr_slice = cont_twr[s_off_s:s_off_e]
                s_ret = ((s_twr_slice[-1] / max(1e-6, s_twr_slice[0])) - 1.0) * 100.0
                peak_w = np.maximum.accumulate(s_twr_slice)
                dd_w = np.where(peak_w > 0, (peak_w - s_twr_slice) / peak_w, 0.0)
                s_dd = float(np.max(dd_w)) * 100.0

                s_b = calculate_benchmarks(grid, b_s, b_e, macro_final)
                risk_mask = (pd.to_datetime(cont_trades["entry_date"]) <= s_end_ts) & (pd.to_datetime(cont_trades["exit_date"]) >= s_start_ts)
                n_active_trades = int(risk_mask.sum())

                stress_rows_md += f"| **{s_name}** | {s_ret:+.2f}% | {s_dd:.2f}% | {s_b['nsei_total_return']:+.2f}% | {s_b['nsei_max_dd']:.2f}% | {s_b['timed_nsei_total_return']:+.2f}% | {n_active_trades} |\n"

    all_trades = pd.concat([is_results["trades"], oos_results["trades"]], ignore_index=True)

    # 4. TWR ANALYTICS & DUAL PLACEBO SUITE
    is_twr_analytics = compute_twr_analytics(is_results["twr_curve"], is_bench["nsei_daily_ret"])
    oos_twr_analytics = compute_twr_analytics(oos_results["twr_curve"], oos_bench["nsei_daily_ret"])

    _pc = best_candidate_record.get("cache") or {}
    is_placebo_suite = _pc.get("placebo_is") or run_matched_placebo_suite(
        grid, best_params, is_bars, is_start, is_end,
        is_bench, is_results["funnel_stats"]["raw_triggers"],
        n_seeds=PLACEBO_SEEDS_CONDITIONED, n_seeds_uncond=PLACEBO_SEEDS_UNCONDITIONED
    )

    oos_placebo_suite = _pc.get("placebo_oos") or run_matched_placebo_suite(
        grid, best_params, oos_bars, oos_start, oos_end,
        oos_bench, oos_results["funnel_stats"]["raw_triggers"],
        n_seeds=PLACEBO_SEEDS_CONDITIONED, n_seeds_uncond=PLACEBO_SEEDS_UNCONDITIONED
    )

    def calc_percentile(cagr_val: float, dist: List[float]) -> float:
        if not dist:
            return 50.0
        return float((np.sum(np.array(dist) < cagr_val) / len(dist)) * 100.0)

    is_cond_pct = calc_percentile(is_metrics["full"]["cagr"], is_placebo_suite["conditioned_samples"])
    oos_cond_pct = calc_percentile(oos_metrics["full"]["cagr"], oos_placebo_suite["conditioned_samples"])

    # Leave-Top-5-Survivors-Out Scrip Breakdown
    leave5_table_md = ""
    top_5_scrips = []
    if len(is_results["trades"]) >= 10:
        symbol_pnl = is_results["trades"].groupby("coin").agg(
            net_pnl=("pnl", "sum"),
            trades=("pnl", "count"),
            win_rate=("pnl", lambda x: (x > 0).mean() * 100.0)
        ).sort_values("net_pnl", ascending=False)

        top_5_summary = symbol_pnl.head(5).reset_index()
        top_5_scrips = top_5_summary["coin"].tolist()
        total_is_pnl = is_results["trades"]["pnl"].sum()

        leave5_table_md = "| Rank | Symbol | Net Profit (INR) | Trade Count | Win Rate (%) | Share of IS Profit |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        for rank_i, r in top_5_summary.iterrows():
            share_pnl = (r["net_pnl"] / max(1.0, total_is_pnl)) * 100.0
            leave5_table_md += f"| #{rank_i+1} | **{r['coin']}** | {format_price(r['net_pnl'])} | {int(r['trades'])} | {r['win_rate']:.1f}% | {share_pnl:.1f}% |\n"

        active_mask = np.ones(len(grid.symbols), dtype=bool)
        for s in top_5_scrips:
            if s in grid.symbols:
                active_mask[grid.symbols.index(s)] = False
    else:
        active_mask = np.ones(len(grid.symbols), dtype=bool)

    res_leave5 = final_engine.run_interval(grid, raw_sig_final, entry_final, exit_final, macro_final, is_start, is_end, active_symbols_mask=active_mask, state_mat=state_final)
    m_leave5 = calculate_metrics(res_leave5, grid, is_start, is_end, is_bench)
    leave5_cagr = m_leave5["full"]["cagr"]

    # Dual-Window Macro-Active Exit Ablation
    if best_params.get("use_market_macro_system", False):
        ablation_params = copy.deepcopy(best_params)
        ablation_params["macro_active_exit"] = not best_params.get("macro_active_exit", False)
        raw_sig_abl, entry_abl, exit_abl, macro_abl, state_abl = compile_signals_fast(grid, ablation_params)
        ablation_engine = BacktestEngine(params=ablation_params)

        res_abl_is = ablation_engine.run_interval(grid, raw_sig_abl, entry_abl, exit_abl, macro_abl, is_start, is_end, state_mat=state_abl)
        m_abl_is = calculate_metrics(res_abl_is, grid, is_start, is_end, is_bench)

        res_abl_oos = ablation_engine.run_interval(grid, raw_sig_abl, entry_abl, exit_abl, macro_abl, oos_start, oos_end, state_mat=state_abl)
        m_abl_oos = calculate_metrics(res_abl_oos, grid, oos_start, oos_end, oos_bench, min_trades=0)

        macro_ablation_md = f"""| Window | Configuration | Strategy CAGR | Max Drawdown | Profit Factor | Completed Trades |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **In-Sample** | **Winner (macro_active_exit={best_params.get('macro_active_exit', False)})** | **{is_metrics['full']['cagr']:.2f}%** | **{is_metrics['max_dd']:.2f}%** | **{is_metrics['full']['profit_factor']:.2f}** | **{is_metrics['full']['trades']}** |
| In-Sample | Ablated (macro_active_exit={ablation_params.get('macro_active_exit', False)}) | {m_abl_is['full']['cagr']:.2f}% | {m_abl_is['max_dd']:.2f}% | {m_abl_is['full']['profit_factor']:.2f} | {m_abl_is['full']['trades']} |
| **Out-of-Sample** | **Winner (macro_active_exit={best_params.get('macro_active_exit', False)})** | **{oos_metrics['full']['cagr']:.2f}%** | **{oos_metrics['max_dd']:.2f}%** | **{oos_metrics['full']['profit_factor']:.2f}** | **{oos_metrics['full']['trades']}** |
| Out-of-Sample | Ablated (macro_active_exit={ablation_params.get('macro_active_exit', False)}) | {m_abl_oos['full']['cagr']:.2f}% | {m_abl_oos['max_dd']:.2f}% | {m_abl_oos['full']['profit_factor']:.2f} | {m_abl_oos['full']['trades']} |
"""
    else:
        macro_ablation_md = "*Macro-Active Exit Ablation not applicable: The winning configuration has `use_market_macro_system = False` (market regime filter is disabled, so macro exits never trigger).*\n"

    # Trailing Stop Mechanism Ablation
    trailing_ablation_md = run_trailing_stop_ablation(
        grid, best_params, is_start, is_end, oos_start, oos_end,
        is_bench, oos_bench
    )

    # Post-Tax STCG Estimate
    def _compute_post_tax(tdf: pd.DataFrame, full_twr_series: np.ndarray, n_bars: int) -> Tuple[float, float]:
        if len(tdf) == 0 or len(full_twr_series) == 0:
            return 0.0, 0.0
        
        tdf_tax = tdf.copy()
        tdf_tax["exit_dt"] = pd.to_datetime(tdf_tax["exit_date"])
        tdf_tax["fy_year"] = np.where(tdf_tax["exit_dt"].dt.month >= 4, tdf_tax["exit_dt"].dt.year, tdf_tax["exit_dt"].dt.year - 1)
        
        cum_tax = 0.0
        loss_carry = 0.0
        tax_cutoff_date = pd.Timestamp("2024-07-23")

        for fy in sorted(tdf_tax["fy_year"].unique()):
            fy_trades = tdf_tax[tdf_tax["fy_year"] == fy]
            
            if fy == 2024:
                trades_pre = fy_trades[fy_trades["exit_dt"] < tax_cutoff_date]
                trades_post = fy_trades[fy_trades["exit_dt"] >= tax_cutoff_date]
                pnl_pre = trades_pre["pnl"].sum()
                pnl_post = trades_post["pnl"].sum()
                net_fy = pnl_pre + pnl_post
                
                if net_fy <= 0:
                    loss_carry += abs(net_fy)
                else:
                    taxable = net_fy - loss_carry
                    if taxable > 0:
                        gain_pre = max(0.0, pnl_pre)
                        gain_post = max(0.0, pnl_post)
                        tot_gain = gain_pre + gain_post
                        ratio_post = (gain_post / tot_gain) if tot_gain > 0 else 0.5
                        tax_fy = (taxable * (1.0 - ratio_post) * 0.15) + (taxable * ratio_post * 0.20)
                        cum_tax += tax_fy
                        loss_carry = 0.0
                    else:
                        loss_carry = abs(taxable)
            else:
                net_pnl_fy = fy_trades["pnl"].sum()
                rate = 0.20 if fy >= 2025 else 0.15
                if net_pnl_fy > 0:
                    taxable = net_pnl_fy - loss_carry
                    if taxable > 0:
                        cum_tax += taxable * rate
                        loss_carry = 0.0
                    else:
                        loss_carry = abs(taxable)
                else:
                    loss_carry += abs(net_pnl_fy)

        total_pre_tax_pnl = tdf_tax["pnl"].sum()
        post_tax_pnl = total_pre_tax_pnl - cum_tax
        
        pre_tax_twr_end = full_twr_series[-1] if len(full_twr_series) > 0 else 1.0
        if total_pre_tax_pnl > 0 and pre_tax_twr_end > 1.0:
            tax_retention = max(0.0, post_tax_pnl / total_pre_tax_pnl)
            post_tax_twr_end = 1.0 + (pre_tax_twr_end - 1.0) * tax_retention
            post_tax_cagr = ((post_tax_twr_end ** (248.0 / max(1, n_bars))) - 1.0) * 100.0
        else:
            post_tax_cagr = ((pre_tax_twr_end ** (248.0 / max(1, n_bars))) - 1.0) * 100.0
            
        return float(post_tax_pnl), float(post_tax_cagr)

    is_tax_pnl, is_tax_cagr = _compute_post_tax(is_results["trades"], is_results["twr_curve"], is_bars)
    oos_tax_pnl, oos_tax_cagr = _compute_post_tax(oos_results["trades"], oos_results["twr_curve"], oos_bars)

    # 7. TOP-K WALK-FORWARD STABILITY MATRIX (With Guaranteed Champion Visibility)
    f1_label = f"{grid.dates[f1_s].year}-{str(grid.dates[f1_e - 1].year)[2:]}"
    f2_label = f"{grid.dates[f2_s].year}-{str(grid.dates[f2_e - 1].year)[2:]}"
    f3_label = f"{grid.dates[f3_s].year}-{str(grid.dates[f3_e - 1].year)[2:]}"

    wf_folds_md = (
        f"| Candidate | Search Score | Fold 1 ({f1_label}) | Bench F1 | Fold 2 ({f2_label}) | Bench F2 | Fold 3 ({f3_label}) | Bench F3 | Neighbors Beating Index | Timing p10 Score (IS window) | Seeds | Status |\n"
        f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    )
    _stage2_ids = {id(r) for r in stage2}
    _wf_rows = list(stage2) + [r for r in scored_candidates if id(r) not in _stage2_ids][:4]
    for rank_idx, rec in enumerate(_wf_rows, start=1):
        is_champ = (rank_idx == 1)
        st_txt = "All-Era Champion" if rec["beats_all"] else ("Recent Outperformer" if rec["beats_recent"] else "Regime Sensitive")
        nb_txt = f"{rec['pass_rate']*100:.0f}% ({'stable' if rec['stable'] else 'UNSTABLE'})" if rec["pass_rate"] is not None else "not run"
        tm_txt = f"{rec['timing']['p10_score']:+.2f}" if rec["timing"] else "not run"
        lab = f"**★ CHAMPION ({rec['trial'].number})**" if is_champ else f"#{rank_idx} ({rec['trial'].number})"
        wf_folds_md += (
            f"| {lab} | {rec['score']:.2f} | {rec['c1']:.1f}% | {rec['i1']:.1f}% | {rec['c2']:.1f}% | {rec['i2']:.1f}% | "
            f"{rec['c3']:.1f}% | {rec['i3']:.1f}% | {nb_txt} | {tm_txt} | {rec['seed_agree']}/{len(SEEDS)} | {st_txt} |\n"
        )

    # 8. Exposure Telemetry
    def _compute_exposure_telemetry(tdf: pd.DataFrame, peak_stock_pct: float) -> Dict[str, Any]:
        if len(tdf) == 0:
            return {"distinct_scrips": 0, "max_single_stock_pct": 0.0, "top5_share": 0.0}
        n_distinct = int(tdf["coin"].nunique())
        
        pnl_by_coin = tdf.groupby("coin")["pnl"].sum()
        pos_pnl_coins = pnl_by_coin[pnl_by_coin > 0]
        tot_pos_pnl = float(pos_pnl_coins.sum()) if len(pos_pnl_coins) > 0 else 1.0
        top5_profit = float(pos_pnl_coins.nlargest(5).sum()) if len(pos_pnl_coins) > 0 else 0.0
        top5_share = float((top5_profit / tot_pos_pnl) * 100.0)

        return {
            "distinct_scrips": n_distinct,
            "max_single_stock_pct": peak_stock_pct,
            "top5_share": top5_share
        }

    is_exp_telemetry = _compute_exposure_telemetry(is_results["trades"], is_results["peak_single_stock_pct"])
    oos_exp_telemetry = _compute_exposure_telemetry(oos_results["trades"], oos_results["peak_single_stock_pct"])

    def _fmt_cap(val: float) -> str:
        return f"{val:.1f}%" if np.isfinite(val) else "N/A"

    # ---- Continuous-slice OOS, full-window timing battery, PROMOTION GATES ----
    oos_slice = segment_from_continuous(continuous_results, grid, is_start, oos_start, oos_end)
    cont_bench_full = calculate_benchmarks(grid, is_start, n_bars, macro_final)
    _tw = continuous_results["twr_curve"]
    _pk = np.maximum.accumulate(_tw)
    cont_max_dd = float(np.max((_pk - _tw) / np.maximum(1e-9, _pk))) * 100.0 if len(_tw) else 0.0
    final_timing = run_timing_robustness(grid, best_params, is_start, n_bars)
    cont_year = _year_returns(continuous_results["twr_curve"], grid.years_arr[is_start:n_bars])
    try:
        cont_trades.drop(columns=["frictions"], errors="ignore").to_csv(CONT_TRADES_FILE, index=False)
    except Exception:
        pass

    eot_multi = eot_share_multi_cut(grid, final_engine, (raw_sig_final, entry_final, exit_final, macro_final, state_final), is_start, is_end,
                                    float(is_metrics["eot_profit_share"]), base_organic=float(is_metrics["cagr_organic"]),
                                    base_basket=float(is_bench["basket_twr_cagr"]), base_headline=float(is_metrics["full"]["cagr"]))
    gates = evaluate_gates(
        oos_slice=oos_slice, oos_bench_cagr=oos_bench["nsei_twr_cagr"], oos_basket_cagr=oos_bench["basket_twr_cagr"],
        is_cond_pct=is_cond_pct, oos_cond_pct=oos_cond_pct,
        is_metrics=is_metrics, leave5_cagr=leave5_cagr, is_bench_cagr=is_bench["nsei_twr_cagr"], cont_trades=cont_trades,
        cont_year=cont_year, idx_year=cont_bench_full["idx_year"],
        cont_max_dd=cont_max_dd, cont_bench_dd=cont_bench_full["nsei_max_dd"],
        stable=bool(best_candidate_record["stable"]), timing_ok=bool(final_timing["ok"]), seed_agree=int(best_candidate_record["seed_agree"]),
        fresh_oos_cagr=float(oos_metrics["full"]["cagr"]), fresh_oos_trades=int(len(oos_results["trades"])), eot_multi=eot_multi,
        nb_pass_rate=(float(best_candidate_record["pass_rate"]) if best_candidate_record["pass_rate"] is not None else None),
        timing_p10=float(final_timing["p10_score"]), n_timing=int(len(final_timing["rows"])),
        fresh_oos_dd=float(oos_metrics["max_dd"]), oos_basket_dd=float(oos_bench["basket_max_dd"]), oos_index_dd=float(oos_bench["nsei_max_dd"]))
    all_pass = all(g[1] for g in gates)
    failing = [f"{g[0]} [{g[2]}]" for g in gates if not g[1]]
    status = "PROMOTED" if all_pass else "NOT PROMOTED"
    gate_md = "| Gate | Result | Detail |\n| :--- | :--- | :--- |\n" + "".join(
        f"| {g[0]} | {'PASS' if g[1] else '**FAIL**'} | {g[2]} |\n" for g in gates)
    try:
        _sp = cont_trades.groupby("coin")["pnl"].sum().sort_values(ascending=False)
        _conc_md = (f"*Concentration by symbol (information, continuous run): top symbol {_sp.index[0]} = {_sp.iloc[0] / max(1.0, float(_sp.sum())):.1%} of net profit, "
                    f"top 2 symbols = {_sp.head(2).sum() / max(1.0, float(_sp.sum())):.1%}, top 5 = {_sp.head(5).sum() / max(1.0, float(_sp.sum())):.1%}; "
                    f"{len(_sp)} symbols traded, {len(cont_trades)} trades. A stock bought in several layers counts once here.*\n\n") if len(_sp) else ""
    except Exception:
        _conc_md = ""
    _eot_md = ("*Unclosed gains at the in-sample cut, by cut date (information): " +
               ", ".join(f"{'actual cut' if b == 0 else '-' + str(b) + ' bars'} {s:.1%}" for b, s in eot_multi["rows"]) + ".*\n\n") if eot_multi.get("rows") else ""
    if eot_multi.get("org_rows"):
        _eot_md += (f"*Open-gain dependence in return terms (the gate when GATE_EOT_MODE is 'return'): organic IS CAGR (open gains at {EOT_EXCESS_WEIGHT:.0%} credit) vs survivor basket by cut: "
                    + ", ".join(f"{'actual cut' if b_ == 0 else '-' + str(b_) + ' bars'} {o_:.1f}% vs {k_:.1f}% (headline {h_:.1f}%)" for b_, o_, k_, h_ in eot_multi["org_rows"]) + ".*\n\n")
    _dd_md = (f"*Drawdown reality check (information): fresh-start OOS max drawdown **{oos_metrics['max_dd']:.1f}%** vs NIFTYBEES {oos_bench['nsei_max_dd']:.1f}% vs "
              f"equal-weight survivor basket {oos_bench['basket_max_dd']:.1f}%; continuous-slice OOS {oos_slice['max_dd']:.1f}%; whole continuous run {cont_max_dd:.1f}% "
              f"(NIFTYBEES {cont_bench_full['nsei_max_dd']:.1f}%). The max-drawdown gate above reads the continuous run only. "
              + (f"The optional fresh-OOS drawdown gate is ON ({GATE_FRESH_OOS_DD_VS_BASKET:.2f}x basket)." if GATE_FRESH_OOS_DD_VS_BASKET is not None
                 else "The optional fresh-OOS drawdown gate is OFF (set GATE_FRESH_OOS_DD_VS_BASKET to enable it).") + "*\n\n")
    gate_section_md = (f"## 0. Promotion Gates — Status: **{status}**\n"
                       f"{'All gates passed: winner.json and PASTE_INTO_LIVE.txt were written.' if all_pass else 'One or more gates failed: winner.json was NOT written and any earlier winner.json is preserved. The config is saved as winner_candidate.json / PASTE_INTO_LIVE_NOT_PROMOTED.txt for inspection only.'}\n\n{gate_md}\n{_conc_md}{_eot_md}{_dd_md}")

    winner_data = {
        "version": ENGINE_VERSION,
        "param_hash": PARAM_SPACE_HASH,
        "study_name": STUDY_NAME,
        "universe": UNIVERSE_NAME,
        "status": status,
        "gates": [{"name": g[0], "pass": g[1], "detail": g[2]} for g in gates],
        "is_score": float(is_metrics["score"]),
        "is_metrics": is_metrics,
        "oos_metrics": oos_metrics,
        "oos_continuous_slice": oos_slice,
        "is_post_tax": {"pnl": is_tax_pnl, "cagr": is_tax_cagr},
        "oos_post_tax": {"pnl": oos_tax_pnl, "cagr": oos_tax_cagr},
        "is_twr_analytics": is_twr_analytics,
        "oos_twr_analytics": oos_twr_analytics,
        "leave5_cagr": leave5_cagr,
        "top_5_scrips": top_5_scrips,
        "is_placebo_suite": {k: v for k, v in is_placebo_suite.items() if not k.endswith("_samples")},
        "oos_placebo_suite": {k: v for k, v in oos_placebo_suite.items() if not k.endswith("_samples")},
        "is_exposure_telemetry": is_exp_telemetry,
        "oos_exposure_telemetry": oos_exp_telemetry,
        "seed_agreement": int(best_candidate_record["seed_agree"]),
        "data_repairs": {t_: sum(1 for r in DATA_REPAIR_LOG if r["type"] == t_) for t_ in ("SPIKE_REVERT_FIXED", "CORP_ACTION_REBASED", "LARGE_MOVE_KEPT")},
        "continuous_year_returns": cont_year,
        "best_params": best_params,
        "updated_at": pd.Timestamp.now().isoformat(),
    }

    with open(CURRENT_WINNER_FILE, "w", encoding="utf-8") as f:
        json.dump(winner_data, f, indent=4, default=str)

    paste_txt = build_paste_block(best_params, status, failing, int(best_candidate_record["seed_agree"]))
    if all_pass:
        logging.info("ALL PROMOTION GATES PASSED -> writing winner.json and PASTE_INTO_LIVE.txt")
        with open(FINAL_WINNER_FILE, "w", encoding="utf-8") as f:
            json.dump(winner_data, f, indent=4, default=str)
        PASTE_FILE.write_text(paste_txt, encoding="utf-8")
    else:
        logging.warning("PROMOTION GATES FAILED: " + " | ".join(failing))
        PASTE_FILE.with_name("PASTE_INTO_LIVE_NOT_PROMOTED.txt").write_text(paste_txt, encoding="utf-8")

    export_df = all_trades.copy()
    if len(export_df) > 0 and "frictions" in export_df.columns:
        export_df["fee_brokerage"] = export_df["frictions"].apply(lambda f: f.brokerage)
        export_df["fee_stt"] = export_df["frictions"].apply(lambda f: f.stt)
        export_df["fee_exchange"] = export_df["frictions"].apply(lambda f: f.exchange_charges)
        export_df["fee_stamp"] = export_df["frictions"].apply(lambda f: f.stamp_duty)
        export_df["fee_sebi"] = export_df["frictions"].apply(lambda f: f.sebi_charges)
        export_df["fee_gst"] = export_df["frictions"].apply(lambda f: f.gst)
        export_df["fee_dp"] = export_df["frictions"].apply(lambda f: f.dp_charges)
        export_df["fee_slippage"] = export_df["frictions"].apply(lambda f: f.slippage_cost)
        export_df = export_df.drop(columns=["frictions"])
    export_df.to_csv(TRADES_CSV_FILE, index=False)

    f_disp = is_results["funnel_stats"]
    funnel_table_md = f"""| Stage in Funnel | Unique Signals | Disposition Description |
| :--- | :--- | :--- |
| **Total Raw Technical Triggers** | **{f_disp['raw_triggers']}** | Close > HHV, RSI cross, or MA trigger |
| ├── Macro Gate Blocked | -{f_disp['macro_blocked']} | NIFTY50 below Macro Moving Average |
| ├── Liquidity Floor Blocked | -{f_disp['liquidity_blocked']} | Median 30d traded value below bracket floor (SMALL: {format_price(BRACKET_LIQUIDITY_FLOOR_INR['SMALL'])}) |
| ├── ADX Trend Blocked | -{f_disp['adx_blocked']} | ADX < Selected Threshold |
| ├── Pyramid Limit Blocked | -{f_disp['pyramid_blocked']} | Scrip already at max tranches or averaging down |
| ├── Revalidation Dropped | -{f_disp['revalidation_dropped']} | Failed macro/ADX/state check at fill time |
| ├── Cash Starved Dropped | -{f_disp['cash_starved_dropped']} | Cash below tranche floor ({format_price(best_params.get('tranche_floor', TRANCHE_FLOOR_INR))}) |
| ├── Scrip Risk Cap Dropped | -{f_disp['risk_cap_dropped']} | Exceeded single-stock 25% exposure ceiling |
| ├── Expired in Watchlist | -{f_disp['expired_unfilled']} | Shadow position stopped/exited (or age cap if one is set) |
| ├── Watchlist Duplicate Dropped | -{f_disp['wl_duplicate_dropped']} | Stock already on the watchlist (one entry per stock) |
| ├── Unaffordable (whole shares) | -{f_disp['unaffordable_dropped']} | One share costs more than twice the target size |
| **Executed Trades on Ledger** | **{f_disp['executed_fills']}** | Successfully filled and audited |
| **Checksum Reconciliation** | **{sum([f_disp['macro_blocked'], f_disp['liquidity_blocked'], f_disp['adx_blocked'], f_disp['pyramid_blocked'], f_disp['revalidation_dropped'], f_disp['cash_starved_dropped'], f_disp['risk_cap_dropped'], f_disp['expired_unfilled'], f_disp['wl_duplicate_dropped'], f_disp['unaffordable_dropped'], f_disp['executed_fills']])}** | Must strictly equal Total Raw Triggers ({f_disp['raw_triggers']}) |
"""

    def _render_excursion(tdf: pd.DataFrame) -> str:
        if len(tdf) == 0:
            return "| Sub-Population | Average MFE (%) | Average MAE (%) | Peak Bar Location | Realized Win Efficiency |\n| :--- | :--- | :--- | :--- | :--- |\n| *No closed trades* | 0.0% | 0.0% | Bar 0 | 0.0% |\n"
        
        wins = tdf[tdf["pnl"] > 0]
        losses = tdf[tdf["pnl"] <= 0]

        win_mfe_str = f"+{wins['mfe_pct'].mean():.2f}%" if len(wins) > 0 and np.isfinite(wins['mfe_pct'].mean()) else "N/A"
        win_mae_str = f"{wins['mae_pct'].mean():.2f}%" if len(wins) > 0 and np.isfinite(wins['mae_pct'].mean()) else "N/A"
        win_peak_str = f"Bar {wins['bars_to_peak'].mean():.1f}" if len(wins) > 0 and np.isfinite(wins['bars_to_peak'].mean()) else "N/A"
        win_eff_str = f"{(wins['ret'].mean() * 100) / max(0.01, wins['mfe_pct'].mean()) * 100:.1f}%" if len(wins) > 0 and np.isfinite(wins['ret'].mean()) and wins['mfe_pct'].mean() > 0 else "N/A"

        loss_mfe_str = f"+{losses['mfe_pct'].mean():.2f}%" if len(losses) > 0 and np.isfinite(losses['mfe_pct'].mean()) else "N/A"
        loss_mae_str = f"{losses['mae_pct'].mean():.2f}%" if len(losses) > 0 and np.isfinite(losses['mae_pct'].mean()) else "N/A"
        loss_peak_str = f"Bar {losses['bars_to_peak'].mean():.1f}" if len(losses) > 0 and np.isfinite(losses['bars_to_peak'].mean()) else "N/A"

        tot_mfe_str = f"+{tdf['mfe_pct'].mean():.2f}%" if np.isfinite(tdf['mfe_pct'].mean()) else "N/A"
        tot_mae_str = f"{tdf['mae_pct'].mean():.2f}%" if np.isfinite(tdf['mae_pct'].mean()) else "N/A"
        tot_peak_str = f"Bar {tdf['bars_to_peak'].mean():.1f}" if np.isfinite(tdf['bars_to_peak'].mean()) else "N/A"
        tot_eff_str = f"Realized: {tdf['ret'].mean()*100:.2f}%" if np.isfinite(tdf['ret'].mean()) else "N/A"

        return f"""| Sub-Population | Average MFE (%) | Average MAE (%) | Peak Bar Location | Realized Win Efficiency |
| :--- | :--- | :--- | :--- | :--- |
| **Winning Positions** | {win_mfe_str} | {win_mae_str} | {win_peak_str} | {win_eff_str} |
| **Losing Positions** | {loss_mfe_str} | {loss_mae_str} | {loss_peak_str} | N/A (Loss Stop) |
| **Total Population** | {tot_mfe_str} | {tot_mae_str} | {tot_peak_str} | {tot_eff_str} |
"""

    is_excursion_md = _render_excursion(is_results["trades"])
    oos_excursion_md = _render_excursion(oos_results["trades"])

    def _render_frictions(tdf: pd.DataFrame) -> str:
        if len(tdf) == 0:
            return ""
        tot_stt = sum(tr["frictions"].stt for _, tr in tdf.iterrows())
        tot_exch = sum(tr["frictions"].exchange_charges for _, tr in tdf.iterrows())
        tot_sebi = sum(tr["frictions"].sebi_charges for _, tr in tdf.iterrows())
        tot_stamp = sum(tr["frictions"].stamp_duty for _, tr in tdf.iterrows())
        tot_gst = sum(tr["frictions"].gst for _, tr in tdf.iterrows())
        tot_dp = sum(tr["frictions"].dp_charges for _, tr in tdf.iterrows())
        tot_brk = sum(tr["frictions"].brokerage for _, tr in tdf.iterrows())
        tot_slip = sum(tr["frictions"].slippage_cost for _, tr in tdf.iterrows())
        tot_frictions = tot_brk + tot_stt + tot_exch + tot_sebi + tot_stamp + tot_gst + tot_dp + tot_slip
        return f"""| Statutory / Operational Fee | In-Sample Paid | Drag per Trade |
| :--- | :--- | :--- |
| **Brokerage (Upstox flat, per executed order)** | {format_price(tot_brk)} | {format_price(tot_brk/max(1, len(tdf)))} |
| **Securities Transaction Tax (STT)** | {format_price(tot_stt)} | {format_price(tot_stt/max(1, len(tdf)))} |
| **NSE Exchange Turnover Fees** | {format_price(tot_exch)} | {format_price(tot_exch/max(1, len(tdf)))} |
| **Stamp Duty (Buy Side)** | {format_price(tot_stamp)} | {format_price(tot_stamp/max(1, len(tdf)))} |
| **SEBI Turnover Charges** | {format_price(tot_sebi)} | {format_price(tot_sebi/max(1, len(tdf)))} |
| **GST (18% on Charges/Brokerage)** | {format_price(tot_gst)} | {format_price(tot_gst/max(1, len(tdf)))} |
| **Depository Participant (DP) Charges** | {format_price(tot_dp)} | {format_price(tot_dp/max(1, len(tdf)))} |
| **Bid-Ask Spread Slippage** | {format_price(tot_slip)} | {format_price(tot_slip/max(1, len(tdf)))} |
| **Total Realized Operational Cost** | **{format_price(tot_frictions)}** | **{format_price(tot_frictions/max(1, len(tdf)))}** |
"""

    frictions_md = _render_frictions(is_results["trades"])

    yearly_md = "| Calendar Year | Strategy TWR | ^NSEI Index Return | Equal-Weight Basket | Completed Trades | Win Rate |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for yr in sorted(cont_year):
        _ib = cont_bench_full["idx_year"].get(yr, float("nan"))
        _bb = cont_bench_full["bsk_year"].get(yr, float("nan"))
        yr_trades = cont_trades[pd.to_datetime(cont_trades["exit_date"]).dt.year == yr]
        yr_wr = (yr_trades["pnl"] > 0).mean() * 100.0 if len(yr_trades) > 0 else 0.0
        _first = int(np.flatnonzero(grid.years_arr == yr)[0])
        _lab = f"{yr} (partial)" if (_first < is_start or int(np.sum(grid.years_arr[is_start:] == yr)) < 200) else f"{yr}"
        yearly_md += f"| **{_lab}** | {cont_year[yr]:+.2f}% | {_ib:+.2f}% | {_bb:+.2f}% | {len(yr_trades)} | {yr_wr:.1f}% |\n"

    _cy = cont_year
    _common = [y for y in _cy if y in cont_bench_full["idx_year"]]
    if _common:
        _b_idx = sum(1 for y in _common if _cy[y] > cont_bench_full["idx_year"][y])
        _b_bsk = sum(1 for y in _common if _cy[y] > cont_bench_full["bsk_year"].get(y, 1e9))
        yearly_md += (f"\n* Years beating NIFTYBEES: **{_b_idx} of {len(_common)}**; years beating the equal-weight survivor basket: **{_b_bsk} of {len(_common)}**. "
                      f"(Counts use flow-stripped calendar-year TWR; years with fewer than {OBJ_MIN_YEAR_BARS} sessions are excluded.)\n")

    def _render_reasons(tdf: pd.DataFrame) -> str:
        if len(tdf) == 0:
            return ""
        reason_summary = tdf.groupby("reason").agg(
            trades=("pnl", "count"), net_pnl=("pnl", "sum"), win_rate=("pnl", lambda x: (x > 0).mean() * 100.0)
        ).reset_index()
        reason_summary["share"] = (reason_summary["trades"] / len(tdf)) * 100.0
        out = "| Exit Reason | Trades | Share (%) | Net PnL | Win Rate (%) | Intended Nature |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        for _, r in reason_summary.iterrows():
            nature = "Defensive Breakeven Stop" if "BREAKEVEN" in r['reason'] else ("Profit Taking / Timeout" if "HOLDING" in r['reason'] or "TP" in r['reason'] else "Risk Control Exit")
            out += f"| {r['reason']} | {int(r['trades'])} | {r['share']:.1f}% | {format_price(r['net_pnl'])} | {r['win_rate']:.1f}% | {nature} |\n"
        return out

    is_reason_md = _render_reasons(is_results["trades"])
    oos_reason_md = _render_reasons(oos_results["trades"])

    concurrency_md = "| Active Positions | Total Sessions (IS) | Time Proportion (%) |\n| :--- | :--- | :--- |\n"
    tot_bars_is = sum(is_results["concurrency_hist"])
    _ch = is_results["concurrency_hist"]
    _last = int(np.flatnonzero(_ch)[-1]) if _ch.any() else 0
    for slots, b_cnt in enumerate(_ch[:_last + 1]):
        concurrency_md += f"| **{slots} Positions Active** | {b_cnt} sessions | {(b_cnt/max(1, tot_bars_is))*100:.1f}% |\n"

    beats_index_is = bool(is_metrics['full']['cagr'] > is_bench['nsei_twr_cagr'])
    beats_index_oos = bool(oos_metrics['full']['cagr'] > oos_bench['nsei_twr_cagr'])
    beats_placebo_is = bool(is_metrics['full']['cagr'] > is_placebo_suite['conditioned'][0])
    beats_placebo_oos = bool(oos_metrics['full']['cagr'] > oos_placebo_suite['conditioned'][0])

    is_alpha_diff = is_metrics['full']['cagr'] - is_bench['nsei_twr_cagr']
    oos_alpha_diff = oos_metrics['full']['cagr'] - oos_bench['nsei_twr_cagr']

    _bm = np.abs(np.diff(grid.bench_close) / np.maximum(1e-9, grid.bench_close[:-1]))
    bench_max_move = float(np.max(_bm)) * 100.0 if len(_bm) else 0.0
    _bpk = np.maximum.accumulate(grid.bench_close)
    bench_full_dd = float(np.max((_bpk - grid.bench_close) / np.maximum(1e-9, _bpk))) * 100.0
    repairs_section_md = repairs_md()
    try:
        _rc = pd.Timestamp(grid.dates[-10])
        _rr = [r_ for r_ in DATA_REPAIR_LOG if pd.Timestamp(r_["date"]) >= _rc]
        if _rr:
            repairs_section_md += ("\n* **Large moves in the last 10 sessions (the final bar can be a partial or bad print; check the chart):** "
                                   + ", ".join(f"{r_['symbol']} {pd.Timestamp(r_['date']).date()} ({r_['type']}, close ratio {r_['ratio']:.2f})" for r_ in _rr) + "\n")
    except Exception:
        pass
    diag_md = search_diagnostics_md(seed_trials, seed_studies)
    sens_md = objective_sensitivity_md(is_metrics)
    toptr_md = top_trades_md(cont_trades)
    timing_md = "| Start shift | Deposit day | Strategy CAGR | Benchmark CAGR | Excess | Score | Trades |\n| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n" + "".join(
        f"| +{r['shift']} bars | day {r['offset']+1} | {r['cagr']:.2f}% | {r['bench']:.2f}% | {r['excess']:+.2f} | {r['score']:+.2f} | {r['trades']} |\n"
        for r in final_timing["rows"]) + f"\n* p10 score **{final_timing['p10_score']:+.2f}**, median **{final_timing['median_score']:+.2f}**, worst excess CAGR **{final_timing['worst_excess']:+.2f}** points.\n"
    nb_pass_txt = f"{best_candidate_record['pass_rate']*100:.0f}% of neighbors beat the index net of DD penalty ({'stable' if best_candidate_record['stable'] else 'UNSTABLE'})" if best_candidate_record["pass_rate"] is not None else "not run"
    missed_md = missed_swing_md(grid, continuous_results.get("signal_log", []))
    bracket_md = bracket_breakdown_md(grid, cont_trades)
    fee_md = fee_reality_md()

    report_md = f"""# NSE Quantitative Strategy Audit Dossier ({ENGINE_VERSION}) — {UNIVERSE_NAME}

{gate_section_md}
## 1. Segregated Performance Accounting (Pure Flow-Stripped TWR)

### A. Performance Baseline & Generalization Check
| Performance Metric | In-Sample ({is_bars} sessions) | Out-of-Sample ({oos_bars} sessions, Fresh Seed) | Generalization Ratio (OOS / IS) |
| :--- | :--- | :--- | :--- |
| **Beats Literal Index CAGR?** | **{'PASSED (+' + f"{is_alpha_diff:.2f}%" + ')' if beats_index_is else 'FAILED (' + f"{is_alpha_diff:.2f}%" + ')'}** | **{'PASSED (+' + f"{oos_alpha_diff:.2f}%" + ')' if beats_index_oos else 'FAILED (' + f"{oos_alpha_diff:.2f}%" + ')'}** | Direct Alpha over Buy & Hold ^NSEI |
| **Conditioned Placebo Alpha Rank** | **{is_cond_pct:.1f}th Percentile** | **{oos_cond_pct:.1f}th Percentile** | Empirical rank in {PLACEBO_SEEDS_CONDITIONED} ADX-conditioned placebo seeds |
| **Beats Conditioned Placebo Median? (weaker than the rank gate)** | **{'PASSED' if beats_placebo_is else 'FAILED'}** | **{'PASSED' if beats_placebo_oos else 'FAILED'}** | OOS median of random triggers {oos_placebo_suite['conditioned'][0]:.2f}%; promotion needs rank >= {GATE_PLACEBO_IS_PCT:.0f}th (IS) / {GATE_PLACEBO_OOS_PCT:.0f}th (OOS) |
| **Rule-Closed Net Profit (Pre-Tax)** | **{format_price(is_metrics['full']['net_pnl'])}** | **{format_price(oos_metrics['full']['net_pnl'])}** | — |
| **Estimated Post-Tax Net Profit (STCG)** | {format_price(is_tax_pnl)} | {format_price(oos_tax_pnl)} | 15% / 20% Net of FY Loss Set-Off |
| **Strategy Annualized TWR (CAGR)** | **{is_metrics['full']['cagr']:.2f}%** | **{oos_metrics['full']['cagr']:.2f}%** | {oos_metrics['full']['cagr']/max(0.01, is_metrics['full']['cagr']):.2f}x |
| **Continuous-Slice OOS CAGR (no restart, positions carried in)** | — | **{oos_slice['cagr']:.2f}%** | Max DD {oos_slice['max_dd']:.2f}% | {oos_slice['trades']} closed trades |
| **Organic CAGR (Anti-Cheat)** | {is_metrics['cagr_organic']:.2f}% | {oos_metrics['cagr_organic']:.2f}% | Terminal unclosed gains receive only {EOT_EXCESS_WEIGHT:.0%} incremental credit |
| **Estimated Post-Tax CAGR** | {is_tax_cagr:.2f}% | {oos_tax_cagr:.2f}% | Indian Fiscal Year audited |
| **Literal ^NSEI Index CAGR** | {is_bench['nsei_twr_cagr']:.2f}% | {oos_bench['nsei_twr_cagr']:.2f}% | Buy-and-Hold index |
| **Reference 200-SMA Timed Index CAGR** | {is_bench['ref_200sma_timed_cagr']:.2f}% | {oos_bench['ref_200sma_timed_cagr']:.2f}% | Fixed trend-following benchmark |
| **Equal-Weight Survivor Basket CAGR** | {is_bench['basket_twr_cagr']:.2f}% | {oos_bench['basket_twr_cagr']:.2f}% | Unweighted survivor basket |
| **Strategy TWR Max Drawdown** | **{is_metrics['max_dd']:.2f}%** | **{oos_metrics['max_dd']:.2f}%** | Pure flow-stripped drop |
| **Closed-Trade Cumulative Drawdown** | {is_metrics['closed_trade_dd_pct']:.2f}% | {oos_metrics['closed_trade_dd_pct']:.2f}% | Contemporaneous peak equity denominator |
| **Literal ^NSEI Index Max Drawdown** | {is_bench['nsei_max_dd']:.2f}% | {oos_bench['nsei_max_dd']:.2f}% | Index stress baseline |
| **Capital Utilization (Active / Equity)** | {is_metrics['utilization']:.2f}% | {oos_metrics['utilization']:.2f}% | Zero cash interest credit |
| **Trade Split (Win % / BE % / Loss %)** | **{is_metrics['full']['pure_win_rate']:.1f}% / {is_metrics['full']['be_rate']:.1f}% / {is_metrics['full']['loss_rate']:.1f}%** | **{oos_metrics['full']['pure_win_rate']:.1f}% / {oos_metrics['full']['be_rate']:.1f}% / {oos_metrics['full']['loss_rate']:.1f}%** | Breakeven stops (|ret|<=0.25%) isolated |
| **Nominal Win Rate (PnL > 0)** | {is_metrics['full']['win_rate']:.2f}% | {oos_metrics['full']['win_rate']:.2f}% | Includes marginal wins |
| **Strategy Profit Factor** | {is_metrics['full']['profit_factor']:.2f} | {oos_metrics['full']['profit_factor']:.2f} | Full trade population audited |
| **Completed Trades** | {is_metrics['full']['trades']} | {oos_metrics['full']['trades']} | Trade volume |
| **Raw Composite Fitness Score** | {is_metrics['score']:.4f} | {oos_metrics['score']:.4f} | Annual excess vs the better of NIFTYBEES / equal-weight basket (each year clipped) minus drawdown-beyond-benchmark penalty. OOS column is not subject to the minimum-trade cliff |

### B. Distinct Ticker & Point-in-Time Exposure Telemetry
| Concentration Metric | In-Sample (IS) | Out-of-Sample (OOS) | Operational Risk Assessment |
| :--- | :--- | :--- | :--- |
| **Distinct Tickers Traded** | {is_exp_telemetry['distinct_scrips']} symbols | {oos_exp_telemetry['distinct_scrips']} symbols | Breadth of execution basket |
| **Contemporaneous Peak Scrip Exposure** | {is_exp_telemetry['max_single_stock_pct']:.1f}% | {oos_exp_telemetry['max_single_stock_pct']:.1f}% | Sized at <=25% at entry; organic price growth floats above |
| **Top 5 Symbol Profit Share** | {is_exp_telemetry['top5_share']:.1f}% | {oos_exp_telemetry['top5_share']:.1f}% | Share of total profitable symbols |

## 2. Comprehensive Multi-Crisis Stress Audit (Continuous Pre-Invested Book)
| Crisis Period | Strategy TWR | Strategy Max DD | ^NSEI Return | ^NSEI Max DD | Timed ^NSEI Return | Active Trades |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{stress_rows_md}
## 3. Macro & Trailing Stop Ablation Audits
### A. Macro-Active Exit Ablation
{macro_ablation_md}
### B. Trailing Stop Mechanism Ablation
{trailing_ablation_md}
## 4. Top-K Walk-Forward Stability Matrix (3 Sequential Disjoint Folds)
> **Champion Promotion Diagnostic:** {selection_diagnosis_msg}
> 
> **Methodological Note (Fold 3 vs Literal OOS):** Fold 3 covers the final 33% of all available sessions ({f3_label}), spanning the late In-Sample bull market plus the entire Out-of-Sample test window. Literal OOS strictly isolates the final 25% of trading history. An algorithm can produce positive alpha over the pure OOS test window while still trailing the index during the broader bull run embedded in Fold 3.
>
> **Score note:** *Search Score* is the plain objective (trimmed excess CAGR vs benchmark, minus a penalty for drawdown beyond the benchmark's). Trials that reach the top decile of their study also receive a soft in-sample plateau penalty (a sample of neighbouring settings is scored; the raw score is kept as a trial attribute), so a Search Score below the raw objective means the setting looked fragile. Neighbourhood, deposit-timing and seed-agreement checks are applied afterwards to the shortlisted candidates and shown in the columns to the right.

{wf_folds_md}
*Timing p10 in this table is measured on the in-sample window for every candidate; the promotion gate in section 0 uses the full window (section 12), so the champion's two numbers can differ.*

## 5. TWR Beta, Capture Ratios & Dual Placebo Suite
| Metric | In-Sample Value | Out-of-Sample Value | Context / Benchmark Baseline |
| :--- | :--- | :--- | :--- |
| **Strategy TWR Beta to ^NSEI** | {is_twr_analytics['twr_beta']:.3f} | {oos_twr_analytics['twr_beta']:.3f} | Regression on daily returns |
| **TWR Correlation to ^NSEI** | {is_twr_analytics['twr_corr']:.3f} | {oos_twr_analytics['twr_corr']:.3f} | Market co-movement |
| **Up-Market Capture Ratio** | {_fmt_cap(is_twr_analytics['up_capture'])} | {_fmt_cap(oos_twr_analytics['up_capture'])} | Performance on positive index sessions |
| **Down-Market Capture Ratio** | {_fmt_cap(is_twr_analytics['down_capture'])} | {_fmt_cap(oos_twr_analytics['down_capture'])} | Absorption on negative index sessions |
| **Leave-Top-5-Survivors-Out CAGR** | **{leave5_cagr:.2f}%** | N/A | Compare to ^NSEI ({is_bench['nsei_twr_cagr']:.2f}%) |
| **Unconditioned Placebo Median** | **{is_placebo_suite['unconditioned'][0]:.2f}%** | **{oos_placebo_suite['unconditioned'][0]:.2f}%** | Pure random entry without ADX gate |
| **Conditioned Placebo Median** | **{is_placebo_suite['conditioned'][0]:.2f}%** | **{oos_placebo_suite['conditioned'][0]:.2f}%** | Matched ADX trend gate baseline |

### Top 5 Survivor Profit Drivers (Leave-Top-5-Out Breakdown)
{leave5_table_md}

## 6. Signal Funnel & Opportunity Attrition Matrix
{funnel_table_md}

## 7. Trade Excursion & Timing Decay Analysis (MFE / MAE)
### In-Sample Excursions
{is_excursion_md}
### Out-of-Sample Excursions
{oos_excursion_md}

## 8. Realized Statutory Frictions & Depository Drag
{frictions_md}

## 9. Annual Pure Time-Weighted Return (TWR) Attribution
{yearly_md}

## 10. Portfolio Slot Concurrency Distribution
{concurrency_md}

## 11. Trade-Reason Population Breakdown
### In-Sample Exits
{is_reason_md}
### Out-of-Sample Exits
{oos_reason_md}

## 12. Robustness Battery (Champion)
### A. Deposit-timing and start-date variants (monthly deposit lands on trading day N; start shifted by S bars; full window)
{timing_md}
* Neighborhood test perturbs every active parameter by one step (lengths, multipliers, MA types, tranche floor, toggles): pass-rate {nb_pass_txt}.

## 13. Missed-Swing Analysis (all raw signals in the continuous run)
{missed_md}

## 14. Where The Profit Came From (Bracket Breakdown, continuous run)
{bracket_md}

## 15. Fee Reality Check (current constants, zero slippage)
{fee_md}

## 16. Data Integrity & Sanity
{sanity_md}
* Benchmark series check: largest single-day move in the NIFTYBEES series actually used = **{bench_max_move:.2f}%**, max drawdown over the full history = **{bench_full_dd:.1f}%** (a value above ~45% would mean a bad print survived).

### Repairs applied before any indicator was computed
{repairs_section_md}

## 17. Objective Sensitivity (champion, in-sample)
{sens_md}

## 18. Top Trades of the Continuous Run (concentration check)
{toptr_md}

## 19. Search Diagnostics (is Optuna's search healthy?)
{diag_md}

## 20. Discovered Optimal Parameter Set
```json
{json.dumps(best_params, indent=4)}
"""
    report_md = report_md.replace("^NSEI", "NIFTYBEES")
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_md)
    logging.info(f"{ENGINE_VERSION} audit dossier written to {REPORT_FILE}")


if __name__ == "__main__":
    run_optimization()