"""
CRYPTO SIGNAL RESEARCH ENGINE -- v1.0
========================================================================
Sibling to main.py (the portfolio-simulation engine), NOT a replacement --
main.py is untouched. This file exists to answer a different, narrower
question: "is this entry+exit logic a good trade generator", isolated
from money management.

WHY THIS FILE EXISTS (context for future you)
  main.py fuses two separable questions into one score:
    (A) Is this entry/exit signal logic +EV?
    (B) If I ran it live with a real monthly budget and a single shared
        cash pool, how would my wealth have compounded?
  Optimizing signal parameters (MA lengths, exit family, RSI lengths...)
  against a (B)-style score means two equally-good signals can score very
  differently purely because of which coin happened to win the
  cash-allocation race on a given day -- capital-timing luck, not signal
  quality. This file removes ALL of (B): no shared cash pool, no monthly
  SIP, no wealth curve, no IRR, no BTC-return benchmarking/alpha/IR. Every
  eligible signal on every coin just takes a trade, independently, with a
  fixed nominal risk unit -- coins never compete for capital because
  there isn't a shared pot to compete for. Output is a TRADE LOG, not a
  portfolio value curve. Once a signal is found here that's genuinely
  good, main.py-style portfolio simulation is a separate, later exercise
  ("how much money should I actually put behind this"), not something
  that should shape which signal wins in the first place.

WHAT'S TOGGLEABLE / OPTIMIZED (all decided per-trial by Optuna, nothing
hardcoded except the fixed universe/date range)
  ENTRY TYPE (categorical, one per trial, applies to the whole universe):
    0 = MA Breakout        -- close crosses above its own entry_ma
    1 = RSI Crossover       -- smoothed rsi_fast crosses above rsi_slow
                               (matches the Pine script's calc_smoothed_rsi:
                               raw ta.rsi() then an SMA smoothing pass)
    2 = MA Crossover        -- short_ma crosses above long_ma (NEW)

  EXIT TYPE (categorical, one per trial, fully independent of entry type --
  any entry can pair with any exit, per your explicit instruction):
    0 = Hybrid ATR TP(50%) + breakeven + ATR trail (from the Pine script)
    1 = %-trail from post-entry high
    2 = ATR-trail from post-entry high
    3 = MA Crossunder       -- close crosses under a separately-optimized
                               exit_ma (NOT tied to the entry MA)
    4 = RSI Crossunder      -- separately-optimized exit-side RSI pair
    5 = MA Crossover exit   -- short_exit_ma crosses under long_exit_ma (NEW)

  TOGGLEABLE FILTERS (independent booleans, Optuna decides on/off):
    - use_btc_entry_gate:   require BTC bullish (close > BTC's own MA) to
                             take a NEW entry. Independent of...
    - use_btc_exit_override: force a full close if BTC turns bearish,
                             regardless of which exit family is active.
    - use_rsi_trend_filter: for entry_type 1 (RSI Crossover) ONLY -- the
                             Pine script's `close > trend_ma` condition,
                             now optional instead of mandatory.
  These three are searched independently, so Optuna can land anywhere
  from "pure RSI crossover, zero filters" to "MA crossover + BTC gate +
  BTC exit override + no trend filter" and anything between -- the data
  decides which filters actually help instead of us assuming.

  EVERY MA ANYWHERE (entry-MA, both legs of entry-crossover, exit-MA,
  both legs of exit-crossover, RSI's optional trend filter) independently
  optimizes:
    - type:   0=SMA 1=EMA 2=DEMA 3=WMA 4=SMMA/RMA
    - length: 20-300  (widened from main.py's 10-150, per your request)
  For both crossover pairs (entry_type 2, exit_type 5), the short leg's
  length is sampled first and the long leg is short + a positive gap, so
  Optuna can never sample a backwards pair.

POSITION SIZING / "MONEY" -- there isn't any, on purpose. Every trade's
  P&L is expressed in two scale-free units instead of dollars:
    - R-multiple: pnl_per_unit_price / (entry_atr * sl_mult). sl_mult
      here is NOT a live hard stop for every exit family (e.g. exit_type
      1's actual stop is the %-trail, not this ATR distance) -- it's used
      ONLY as a consistent "nominal risk unit" so trades under different
      exit families are still comparable in R. This mirrors how a trader
      thinks about "how many R did this trade make", independent of
      position size.
    - pct_return: (exit_price - entry_price) / entry_price, for anyone
      who wants a plain percentage view instead of R.
  A trade's blended exit price accounts for the hybrid method's 50%
  partial: exit_price = 0.5*partial_fill + 0.5*final_fill when a partial
  occurred, else just the final fill.

SCORING (trade-log based, no wealth curve)
  score = [ SQN_capped*0.30 + Expectancy_R*0.30 + MedianYearlyR*0.30
            + WinRateBonus*0.10 ] * stat_confidence * concentration_penalty
  - SQN (System Quality Number, Van Tharp): mean(R)/std(R) * sqrt(min(n,100)),
    capped at 6.0 for scoring stability. Trade-level analogue of Sharpe --
    no annualization basis needed since there's no daily portfolio series.
  - Expectancy_R: win_rate*avg_win_R - (1-win_rate)*avg_loss_R.
  - MedianYearlyR (the metric you asked for instead of a concentration
    gate): group ALL trades (across every coin) by calendar entry-year,
    sum R per year, take the MEDIAN across years. Directly answers "what
    does a typical year look like", and is naturally robust to one
    outlier year without needing a hard cutoff rule.
  - WinRateBonus: small, optional; a trend system is supposed to have a
    sub-50% win rate, so this stays low-weight on purpose.
  - concentration_penalty: SOFT, not a hard gate (previous 55% hard-reject
    rule is gone -- it would have penalized a strategy for a legitimate
    fat right tail, which is normal and expected for a trend system, not
    a bug). Only kicks in at genuinely extreme levels (>80% of total R
    from a single year OR a single coin): multiplies score by 0.7. Below
    that threshold, no penalty at all. Reported either way as a
    diagnostic even when it doesn't fire.

  HARD GATES (all must pass or score = -999): trades >= MIN_TRADES_GATE,
  distinct calendar years touched >= MIN_YEARS_GATE, profit_factor >=
  1.10, win_rate >= MIN_WIN_RATE_GATE (kept at 0.35, same rationale as
  main.py: a big reward:risk trend system is SUPPOSED to lose often),
  expectancy_R > 0 (replaces main.py's `roi > 0` -- this is the
  signal-level equivalent: is the AVERAGE trade profitable at all).

  NOTE ON WHAT'S DELIBERATELY ABSENT vs main.py: no ROI, no bench_roi, no
  alpha, no Information Ratio, no Calmar, no Sortino, no max-drawdown-of-
  wealth, no cash-utilization diagnostics, no robustness-vs-starting-
  wealth chaining between IS/OOS. All of those are properties of a WEALTH
  CURVE, and there isn't one here by design.

WHAT'S UNCHANGED IN SPIRIT FROM main.py (because it doesn't require a
wealth curve to make sense): IS/OOS walk-forward split (day-index based,
same as before -- a trade only counts if BOTH its entry and exit land
inside the window), neighborhood-parameter-perturbation stability check,
and the Deflated Sharpe Ratio overfitting diagnostic (here computed
against the trade-level R-multiple distribution's mean/std instead of a
daily portfolio Sharpe -- same Bailey & Lopez de Prado math, different
input series).

CHANGELOG
  v1.0  Initial split from main.py -- trade-log based, no portfolio sim.
  v1.1  Fixed two scoring bugs found by inspecting a real run: the yearly
        consistency term was a SUM (rewarded trade volume, not quality)
        instead of an average, and individual trade R wasn't winsorized
        (a couple of freak long-hold trades could dominate the mean).
  v1.2  Added PYRAMIDING (the same entry signal firing again while
        already in a position adds a layer, up to max_pyramid_layers --
        Optuna decides how many, 1 = off) and RECENCY WEIGHTING (the
        yearly consistency term now weights recent years more than old
        ones, so a strategy that made all its money early and has been
        idle since scores worse than one still working now). Also added
        a baseline-config loader (BASELINE_CONFIG_FILE) so a previous or
        externally-sourced config can be evaluated and used to seed a new
        search instead of starting from scratch, and consolidated every
        manually-adjustable constant into one config block at the top of
        the file (section 0).
"""

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
# 0. USER CONFIGURATION -- every manually-adjustable parameter lives here,
#    in one place. (Coin exclusion lists are DATA, not tuning knobs --
#    they stay in section 1 below, right where they're easy to find.)
# ==========================================

# -- Files --
# Paste a config (in the same shape save_winner() writes -- see the
# BASELINE CONFIG LOADER section below for the exact format and a legacy
# translator for older schemas) into BASELINE_CONFIG_FILE and the engine
# will evaluate it FIRST, print its performance, and seed Optuna's search
# with it as trial #1 -- so a new run starts from your best known config
# instead of from scratch.
BASELINE_CONFIG_FILE = "baseline_config.json"
BEST_PARAMS_FILE     = "best_params_crypto_signal_v1.json"   # engine writes ITS OWN winners here
ENGINE_VERSION        = "signal-1.2.0"

# -- Universe / data window --
START_DATE       = "2017-01-01"
MIN_HISTORY_DAYS = 300   # a coin needs at least this many days of history or it's dropped entirely

# -- Search budget --
BAYESIAN_TRIALS = 20_000
WFO_IS_PCT  = 0.70   # fraction of history used for in-sample optimization
WFO_OOS_PCT = 0.30   # remainder held out, never seen during search
ROBUSTNESS_DEPLOY_THRESHOLD = 0.50   # OOS/IS score ratio required to auto-save a champion

# -- Quality gates -- a trial failing ANY of these scores -999 --
MIN_TRADES_GATE   = 50
MIN_YEARS_GATE    = 3
MIN_WIN_RATE_GATE = 0.35   # trend systems are SUPPOSED to have a sub-50% win rate --
                            # this is deliberately lower than a 50% "default"

# -- Robustness-aware search (v1.3): instead of optimizing on the full
#    in-sample window and only checking true out-of-sample once at the
#    very end, the IS window itself gets split into an inner TRAIN slice
#    and an inner VALIDATION slice during the search, and every trial is
#    scored on the WORSE of the two. A parameter set can no longer win by
#    being lucky in one period -- it has to hold up in both. True OOS
#    (the WFO_OOS_PCT slice below) stays completely untouched during the
#    whole search either way, as the final honest check. --
INNER_VAL_PCT = 0.30          # fraction of the IS window reserved as inner-validation
MIN_YEARS_GATE_INNER = 2      # relaxed years-gate for the (shorter) inner-validation slice only
                               # -- final IS/OOS reporting still uses the full MIN_YEARS_GATE
ROBUST_FALLBACK_SCALE   = 0.05  # if a trial passes inner-train but fails inner-val, it still
ROBUST_FALLBACK_PENALTY = 5.0   # gets a (heavily discounted) fallback score instead of a flat
                                  # -999, so Optuna always has SOME gradient to search with even
                                  # if true dual-pass candidates are rare. Fallback-scored trials
                                  # are mathematically capped well below any real dual-pass trial
                                  # (see run_optimization's optuna_objective for the exact math),
                                  # so a genuinely robust candidate always outranks a lucky one.

# -- MA parameter search ranges (applies to every MA anywhere: entry-MA,
#    both legs of both crossover pairs, exit-MA, RSI's trend filter) --
MA_LEN_MIN, MA_LEN_MAX = 20, 300
MA_CACHE_MAX_ENTRIES   = 40_000   # perf only, raise if you have RAM to spare

# -- Pyramiding (v1.2): Optuna picks max_pyramid_layers per trial, so it
#    decides for itself whether adding to winners helps. 1 = pyramiding
#    off (each coin gets at most one open position at a time, as before).
MAX_PYRAMID_LAYERS_MIN = 1
MAX_PYRAMID_LAYERS_MAX = 4

# -- Recency weighting (v1.2, reshaped v1.3): a calendar year's contribution
#    to the scored consistency term decays from RECENCY_WEIGHT_MAX (the most
#    RECENT year in the window being scored) down to RECENCY_WEIGHT_MIN (the
#    OLDEST year), same bounded [MIN, MAX] range as before. Widen the gap to
#    lean harder toward "still working now" over "worked years ago and has
#    been quiet since".
#    v1.3 changed the SHAPE of the ramp from linear to a half-life-style
#    exponential decay (still normalized to hit exactly MIN at the window's
#    oldest year and MAX at its newest -- a drop-in replacement, nothing
#    downstream needs to change). Rationale: a straight line spreads the
#    30%-wide MIN..MAX gap evenly across however many years are in the
#    window, so a coin's best year sitting in the MIDDLE of a long window
#    (e.g. the 2020/2021 bull run inside an 8-year window) gets docked
#    almost as much as if it were the oldest year. A half-life decay instead
#    concentrates most of the weight in the last ~RECENCY_HALF_LIFE_YEARS
#    and lets everything older than that fall toward MIN together, which
#    better matches "does this still work now" without needing to know
#    exactly where in a halving cycle a given year sat (that's a harder,
#    separate problem -- see the REGIME note below). --
RECENCY_WEIGHT_MIN = 0.70
RECENCY_WEIGHT_MAX = 1.00
RECENCY_HALF_LIFE_YEARS = 2.0   # weight halves (within the MIN..MAX band)
                                # every this-many years back from the
                                # window's most recent year
# NOTE ON CRYPTO'S ~4-YEAR HALVING CYCLE: neither the old linear ramp nor
# this half-life version knows which phase of a halving cycle a given year
# was in -- both are pure calendar-distance decays. A genuinely regime-aware
# scheme would need bull/bear/accumulation years labeled explicitly (e.g. via
# BTC's own drawdown/return) and weighted by regime-similarity-to-now instead
# of by calendar distance. That's a real, separate improvement worth doing
# later; this change only fixes the "evenly-spread-over-the-window" issue,
# not the regime-blindness issue.

# -- R-multiple winsorization (v1.1): caps how much one freak trade can
#    dominate the AGGREGATE score (expectancy, SQN, yearly averages). The
#    raw/uncapped value is still stored and reported per-trade regardless
#    -- capping only affects what SCORES, never what you can SEE. --
R_WINSORIZE_CAP = 20.0

# -- Concentration diagnostic -- SOFT score penalty only, never a hard
#    reject (a fat right tail is normal/expected for a trend system) --
CONCENTRATION_SOFT_THRESHOLD = 0.80
CONCENTRATION_PENALTY_MULT   = 0.70

# -- Naive portfolio-risk diagnostics -- SOFT score penalties only, same
#    philosophy as concentration above. This engine deliberately has no
#    real wealth curve (see evaluate_params_signal's docstring) since
#    that's crypto_portfolio_optimizer.py's job -- but a signal whose
#    trades are individually fine yet cluster together in time (many
#    losers at once, a long unbroken losing streak, many coins crashing
#    on the same days) will make ANY capital-allocation scheme downstream
#    struggle, no matter how the portfolio optimizer's watchlist/funding
#    logic is tuned. These are cheap, trade-log-only proxies for that risk,
#    used to gently steer the search away from it -- not a real portfolio
#    simulation and not a hard gate, since some clustering is unavoidable
#    in a correlated asset class like crypto.
NAIVE_RISK_PCT_PER_TRADE = 0.01   # fixed-fractional risk assumed for the scored naive equity curve
NAIVE_DD_SOFT_THRESHOLD  = 0.50   # naive max drawdown above this triggers a penalty
NAIVE_DD_PENALTY_MULT    = 0.75
# losing streaks are compared to what's STATISTICALLY EXPECTED at this
# trial's own win rate (a 35%-win-rate trend system naturally has long
# losing streaks -- that's normal, not a red flag) rather than a fixed
# absolute count, so this doesn't unfairly punish a healthy low-win-rate
# system for behaving exactly as a low-win-rate system should.
STREAK_RATIO_SOFT_THRESHOLD = 1.5   # observed streak vs statistically-expected streak
STREAK_PENALTY_MULT         = 0.85
# fraction of all concurrently-open trades that were eventual losers, at
# the single worst (most-correlated) moment in the backtest
CORRELATED_LOSS_FRACTION_THRESHOLD = 0.70
CORRELATED_LOSS_MIN_COUNT          = 3     # ignore tiny-sample noise (e.g. 1-of-1 open = trivially 100%)
CORRELATED_LOSS_PENALTY_MULT       = 0.80

# -- Composite score weights -- must sum to 1.00 --
W_SQN        = 0.30
W_EXPECTANCY = 0.30
W_RECENCY    = 0.30   # scores recency_weighted_avg_r (see section 7)
W_WR_BONUS   = 0.10

SQN_CAP = 6.0   # Van Tharp's own scale calls >6 "holy grail" -- capped so
                # one freak trial can't dominate

# -- Neighborhood-stability perturbation check (a real edge shouldn't
#    collapse if you nudge a parameter slightly) --
NEIGHBOR_THRESHOLD = 0.80

# -- entry/exit signal type IDs (vocabulary, not really "tunable", but
#    kept here since every gate/weight above refers to the same trade
#    universe these define) --
ENTRY_MA_BREAKOUT = 0
ENTRY_RSI_XOVER   = 1
ENTRY_MA_XOVER    = 2

EXIT_HYBRID          = 0
EXIT_PCT_TRAIL       = 1
EXIT_ATR_TRAIL       = 2
EXIT_MA_CROSSUNDER   = 3
EXIT_RSI_CROSSUNDER  = 4
EXIT_MA_XOVER_EXIT   = 5


def _recency_weight(year, min_year, max_year):
    """Half-life-style exponential decay from RECENCY_WEIGHT_MAX (max_year)
    down to RECENCY_WEIGHT_MIN (min_year), normalized so the endpoints land
    on EXACTLY the same two values the old linear ramp used -- only the
    shape of the interior changed (concentrated near the recent end instead
    of spread evenly), so every caller/consumer of this function is
    unaffected. If the window spans only one year, returns the max weight
    (nothing to compare against)."""
    if max_year <= min_year:
        return RECENCY_WEIGHT_MAX
    years_back = max_year - year               # 0 at the newest year
    span_years = max_year - min_year
    raw = 0.5 ** (years_back / RECENCY_HALF_LIFE_YEARS)        # 1.0 at newest, decays going back
    raw_floor = 0.5 ** (span_years / RECENCY_HALF_LIFE_YEARS)  # raw's value at the OLDEST year
    denom = 1.0 - raw_floor
    if denom <= 1e-12:
        # half-life is so long relative to the window that raw barely moves
        # across it -- fall back to the linear ramp rather than divide by ~0
        frac = (year - min_year) / span_years
    else:
        frac = (raw - raw_floor) / denom
    return RECENCY_WEIGHT_MIN + (RECENCY_WEIGHT_MAX - RECENCY_WEIGHT_MIN) * frac


# ==========================================
# 1. UNIVERSE DEFINITION (unchanged from main.py -- data infra, not
#    portfolio-simulation, so it carries over as-is)
# ==========================================

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

MIN_HISTORY_DAYS = 400

BINANCE_KLINES_URL    = "https://api.binance.com/api/v3/klines"
BINANCE_EXINFO_URL    = "https://api.binance.com/api/v3/exchangeInfo"
COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"

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
    print("Fetching top-100-by-market-cap universe from CoinGecko...")
    binance_usdt = _get_binance_usdt_symbols()

    tickers = []
    excluded_log = []
    try:
        r = requests.get(COINGECKO_MARKETS_URL, params={
            'vs_currency': 'usd', 'order': 'market_cap_desc',
            'per_page': 100, 'page': 1, 'sparkline': 'false'
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
            binance_sym = f"{sym}USDT"
            if binance_usdt is not None and binance_sym not in binance_usdt:
                excluded_log.append(f"{sym} (no Binance USDT pair)")
                continue
            tickers.append(binance_sym)
    except Exception as e:
        print(f"CoinGecko fetch failed ({e}); falling back to a static major-coin list.")
        tickers = list(FALLBACK_UNIVERSE)

    tickers = [t for t in tickers if t != 'BTCUSDT']
    tickers = ['BTCUSDT'] + list(dict.fromkeys(tickers))

    if len(tickers) < 20:
        print("Universe too small after filtering; falling back to static major-coin list.")
        tickers = list(FALLBACK_UNIVERSE)

    print(f"Universe size after filtering: {len(tickers)} (excluded {len(excluded_log)})")
    assert tickers[0] == 'BTCUSDT', "BTCUSDT must be at index 0."
    return tickers


# ==========================================
# 2. SAVE / LOAD -- the engine's own output file, plus the baseline-config
#    loader for pasting in a config you want to test or seed a run with.
# ==========================================

def load_previous_winner(filename=BEST_PARAMS_FILE):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                return data.get('oos_score', -999999), data.get('is_score', -999999), data.get('params', None)
        except Exception:
            pass
    return -999999, -999999, None


def save_winner(oos_score, is_score, params, filename=BEST_PARAMS_FILE, tier=None,
                 naive_dd_1pct=None, naive_dd_2pct=None,
                 max_consecutive_losses=None, streak_ratio=None,
                 worst_day_loser_fraction=None, universe=None):
    clean_params = {}
    for k, v in params.items():
        if isinstance(v, (bool, np.bool_)):
            clean_params[k] = bool(v)
        elif isinstance(v, (float, np.floating)):
            clean_params[k] = float(v)
        else:
            clean_params[k] = int(v)
    data = {
        'engine_version': ENGINE_VERSION,
        'oos_score': float(oos_score),
        'is_score':  float(is_score),
        'robustness_ratio': float(oos_score / is_score) if is_score > 0 else 0.0,
        'robustness_tier': tier if tier is not None else 'unrated',
        'naive_dd_estimate_1pct_risk': float(naive_dd_1pct) if naive_dd_1pct is not None else None,
        'naive_dd_estimate_2pct_risk': float(naive_dd_2pct) if naive_dd_2pct is not None else None,
        'max_consecutive_losses': int(max_consecutive_losses) if max_consecutive_losses is not None else None,
        'streak_ratio': float(streak_ratio) if streak_ratio is not None else None,
        'worst_day_loser_fraction': float(worst_day_loser_fraction) if worst_day_loser_fraction is not None else None,
        # -- the EXACT ticker list (in the exact order) this run's data was
        #    built from. Persisted so downstream tools -- most importantly
        #    crypto_portfolio_optimizer.py -- can build their own universe
        #    as a filtered SUBSET of the coins this signal was actually
        #    optimized on, instead of independently re-querying CoinGecko
        #    at a different time and silently drifting onto a different
        #    coin set (market-cap rank changes daily). See fetch_top100_universe(). --
        'universe': list(universe) if universe is not None else None,
        'params': clean_params
    }
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving {filename}: {e}")


# -- BASELINE CONFIG LOADER --------------------------------------------
# Paste a config into BASELINE_CONFIG_FILE (same folder as this script)
# and a run will evaluate it FIRST, print its full performance report,
# and seed Optuna's search with it as trial #1. Native format is exactly
# what save_winner() writes:
#   {"engine_version": ..., "oos_score": ..., "is_score": ...,
#    "robustness_ratio": ..., "params": {<this engine's field names>}}
# A bare {"entry_type": 0, ...} params dict (no wrapper) also works.
#
# LEGACY SCHEMA SUPPORT: if the pasted params dict looks like it came from
# an older/different engine (keys like "signal_method", "use_trend_ma",
# "use_btc_filter", "use_panic_exits", "rsi_fast_len", "wl_rank" instead
# of this engine's "entry_type", "use_rsi_trend_filter", etc.), a
# best-effort translation is attempted -- see _LEGACY_KEY_MAP below. This
# is a GUESS at field correspondence, not a guarantee: it's printed in
# full so you can check it, and the translated result is echoed back in
# this engine's native format so you can paste THAT in going forward and
# skip the guessing entirely.

_LEGACY_KEY_MAP = {
    'signal_method':   'entry_type',           # assumed same numbering: 0=MA breakout, 1=RSI crossover
    'use_trend_ma':    'use_rsi_trend_filter',
    'use_btc_filter':  'use_btc_entry_gate',
    'use_panic_exits': 'use_btc_exit_override',
    'exit_method':     'exit_type',            # assumed same numbering 0-3 (this engine adds 4,5 on top)
    'rsi_fast_len':    'rsi_f_len',
    'rsi_fast_smt':    'rsi_f_smt',
    'rsi_slow_len':    'rsi_s_len',
    'rsi_slow_smt':    'rsi_s_smt',
    'trend_ma_len':    'rsi_trend_ma_len',
    'trend_ma_type':   'rsi_trend_ma_type',
    # unchanged names, listed for completeness / self-documentation:
    'entry_ma_len': 'entry_ma_len', 'entry_ma_type': 'entry_ma_type',
    'exit_ma_len': 'exit_ma_len', 'exit_ma_type': 'exit_ma_type',
    'btc_ma_len': 'btc_ma_len', 'btc_ma_type': 'btc_ma_type',
    'adx_thresh': 'adx_thresh', 'sl_mult': 'sl_mult', 'tp_mult': 'tp_mult',
    'trail_mult': 'trail_mult', 'trail_pct': 'trail_pct', 'exit_atr_mult': 'exit_atr_mult',
}
_LEGACY_DROPPED_KEYS = {
    'wl_rank': "portfolio watchlist-ranking concept -- doesn't exist in this "
               "engine (no shared capital to rank candidates for)",
}
_NATIVE_KEY_SIGNATURE = {'entry_type', 'exit_type', 'use_btc_entry_gate',
                          'use_btc_exit_override', 'use_rsi_trend_filter'}


def _looks_legacy(params):
    return bool(_NATIVE_KEY_SIGNATURE.isdisjoint(params.keys())) and \
           any(k in params for k in _LEGACY_KEY_MAP)


def translate_legacy_config(raw_params):
    translated = {}
    notes = []
    for k, v in raw_params.items():
        if k in _LEGACY_DROPPED_KEYS:
            notes.append(f"  DROPPED '{k}' = {v}  ({_LEGACY_DROPPED_KEYS[k]})")
            continue
        new_key = _LEGACY_KEY_MAP.get(k, k)
        if new_key != k:
            notes.append(f"  MAPPED  '{k}' -> '{new_key}'  (value {v} unchanged)")
        translated[new_key] = v
    if 'max_pyramid_layers' not in translated:
        translated['max_pyramid_layers'] = 1
        notes.append("  DEFAULTED 'max_pyramid_layers' = 1 (not present in legacy config -> pyramiding off)")
    return translated, notes


def _validate_and_fill_params(p):
    """Checks that every field the given entry_type/exit_type/toggle
    combination actually needs is present, fills safe defaults for
    fields that are always-suggested-but-conditionally-unused elsewhere
    in this engine, and returns (ok, filled_params, issues)."""
    p = dict(p)
    issues = []
    required_missing = []

    def need(key):
        if key not in p or p[key] is None:
            required_missing.append(key)

    et = p.get('entry_type')
    xt = p.get('exit_type')
    if et == ENTRY_MA_BREAKOUT:
        need('entry_ma_len'); need('entry_ma_type')
    elif et == ENTRY_RSI_XOVER:
        for k in ('rsi_f_len', 'rsi_f_smt', 'rsi_s_len', 'rsi_s_smt'):
            need(k)
        if p.get('use_rsi_trend_filter'):
            need('rsi_trend_ma_len'); need('rsi_trend_ma_type')
    elif et == ENTRY_MA_XOVER:
        for k in ('xover_short_len', 'xover_short_type', 'xover_long_len', 'xover_long_type'):
            need(k)
    else:
        issues.append(f"Unrecognized or missing 'entry_type': {et!r}")

    if xt == EXIT_MA_CROSSUNDER:
        need('exit_ma_len'); need('exit_ma_type')
    elif xt == EXIT_RSI_CROSSUNDER:
        for k in ('exit_rsi_f_len', 'exit_rsi_f_smt', 'exit_rsi_s_len', 'exit_rsi_s_smt'):
            need(k)
    elif xt == EXIT_MA_XOVER_EXIT:
        for k in ('exit_xover_short_len', 'exit_xover_short_type', 'exit_xover_long_len', 'exit_xover_long_type'):
            need(k)
    elif xt not in (EXIT_HYBRID, EXIT_PCT_TRAIL, EXIT_ATR_TRAIL):
        issues.append(f"Unrecognized or missing 'exit_type': {xt!r}")

    if p.get('use_btc_entry_gate') or p.get('use_btc_exit_override'):
        need('btc_ma_len'); need('btc_ma_type')
    else:
        p.setdefault('btc_ma_len', 62); p.setdefault('btc_ma_type', 0)

    for k, default in [('adx_thresh', 0.0), ('sl_mult', 4.0), ('tp_mult', 30.0),
                        ('trail_mult', 6.0), ('trail_pct', 15.0), ('exit_atr_mult', 3.0),
                        ('max_pyramid_layers', 1), ('use_btc_entry_gate', False),
                        ('use_btc_exit_override', False), ('use_rsi_trend_filter', False)]:
        if k not in p:
            p[k] = default
            issues.append(f"  (defaulted missing '{k}' = {default})")

    if required_missing:
        issues.insert(0, f"MISSING required fields for entry_type={et}/exit_type={xt}: {required_missing}")
        return False, p, issues
    return True, p, issues


def load_baseline_config(filename=BASELINE_CONFIG_FILE):
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, 'r') as f:
            raw = json.load(f)
    except Exception as e:
        print(f"\n** Could not parse {filename}: {e}. Ignoring baseline, starting fresh. **")
        return None

    raw_params = raw.get('params', raw) if isinstance(raw, dict) else None
    if not raw_params:
        print(f"\n** {filename} exists but has no usable 'params'. Ignoring, starting fresh. **")
        return None

    print(f"\n{'='*60}\nLOADED BASELINE CONFIG from {filename}\n{'='*60}")

    if _looks_legacy(raw_params):
        print("This looks like it's from a different/older schema -- attempting")
        print("a best-effort field translation (verify this against what you")
        print("actually meant; the exact native-schema version is echoed below):")
        params, notes = translate_legacy_config(raw_params)
        for n in notes:
            print(n)
    else:
        params = dict(raw_params)

    ok, params, issues = _validate_and_fill_params(params)
    for issue in issues:
        print(f"  {issue}")

    if not ok:
        print(f"\n** Baseline config is missing required fields and can't be run as-is. "
              f"Fix the JSON and rerun, or delete/rename {filename} to skip it. Starting fresh. **")
        return None

    print("\nNative-schema equivalent (paste this back into the file to skip translation next time):")
    print(json.dumps(params, indent=2, default=str))
    print("=" * 60)
    return params





# ==========================================
# 3. INDICATORS -- MA (5 types) + RSI (Wilder + SMA smoothing, matching
#    the Pine script's calc_smoothed_rsi)
# ==========================================

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
def calc_rsi_wilder(prices, period):
    """Standard Wilder RSI, matching TradingView's ta.rsi() -- the RAW
    RSI that the Pine script then smooths with an SMA pass (done
    separately below via calc_ma, matching calc_smoothed_rsi in the
    Pine source)."""
    n = len(prices)
    rsi = np.empty(n)
    rsi[:] = np.nan
    start = 0
    while start < n and np.isnan(prices[start]):
        start += 1
    if n - start < period + 1:
        return rsi

    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(start + 1, n):
        diff = prices[i] - prices[i - 1]
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = -diff

    avg_gain = np.mean(gains[start + 1:start + period + 1])
    avg_loss = np.mean(losses[start + 1:start + period + 1])
    idx0 = start + period
    if avg_loss > 0:
        rs = avg_gain / avg_loss
        rsi[idx0] = 100.0 - 100.0 / (1.0 + rs)
    else:
        rsi[idx0] = 100.0

    for i in range(idx0 + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
        else:
            rsi[i] = 100.0
    return rsi


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


# ==========================================
# 3b. CACHES (MA + raw RSI) -- perf only, same bounded/thread-safe pattern
# ==========================================

_MA_CACHE = OrderedDict()
_MA_CACHE_LOCK = threading.Lock()
_RSI_CACHE = OrderedDict()
_RSI_CACHE_LOCK = threading.Lock()


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


def get_raw_rsi_cached(closes, stock_idx, period):
    key = (stock_idx, int(period))
    with _RSI_CACHE_LOCK:
        cached = _RSI_CACHE.get(key)
        if cached is not None:
            _RSI_CACHE.move_to_end(key)
            return cached
    result = calc_rsi_wilder(closes[:, stock_idx], period)
    with _RSI_CACHE_LOCK:
        _RSI_CACHE[key] = result
        while len(_RSI_CACHE) > MA_CACHE_MAX_ENTRIES:
            _RSI_CACHE.popitem(last=False)
    return result


def get_smoothed_rsi(closes, stock_idx, rsi_len, smooth_len):
    raw = get_raw_rsi_cached(closes, stock_idx, rsi_len)
    return calc_ma(raw, smooth_len, 0)   # SMA smoothing, matches Pine's ta.sma(raw_rsi, smt_l)


def clear_caches():
    with _MA_CACHE_LOCK:
        _MA_CACHE.clear()
    with _RSI_CACHE_LOCK:
        _RSI_CACHE.clear()


# ==========================================
# 4. PER-COIN SIGNAL SIMULATOR
# ==========================================
# One coin at a time, fully independent of every other coin (no shared
# capital, so no competition, so no reason to simulate the whole universe
# in lockstep). Close-based signals, fill at next bar's open -- same
# two-phase discipline as main.py. Outputs a trade log (arrays), not a
# portfolio value curve.

@njit(nogil=True, cache=True)
def simulate_signal_trades(
        opens, closes, atr, adx,
        entry_ma, xover_short, xover_long,
        rsi_fast, rsi_slow, rsi_trend_ma,
        btc_close, btc_ma,
        exit_ma, exit_xover_short, exit_xover_long,
        exit_rsi_fast, exit_rsi_slow,
        entry_type, exit_type,
        use_btc_entry_gate, use_btc_exit_override, use_rsi_trend_filter,
        adx_threshold,
        sl_mult, tp_mult, trail_mult, trail_pct, exit_atr_mult,
        max_pyramid_layers,
        start_day, end_day):
    """
    PYRAMIDING (v1.2): if max_pyramid_layers > 1, the SAME entry signal
    firing again while already in a position adds another layer instead
    of being ignored, up to max_pyramid_layers total. This mirrors
    main.py's approach: the position's cost basis becomes the (equally-
    weighted) average of every layer's fill price, but the STOP/TP levels
    and the R-multiple's risk denominator stay anchored to the FIRST
    layer's entry (not re-anchored on each add) -- adding to a winner
    doesn't retroactively pretend you had a tighter stop the whole time.
    A pyramided position still exits as ONE trade (all layers close
    together, same exit signal) and is logged as ONE row in the trade
    log, with the layer count attached for reporting.
    """

    n_days = len(closes)
    if end_day < 0 or end_day >= n_days: end_day = n_days - 2
    if start_day < 1: start_day = 1
    if max_pyramid_layers < 1: max_pyramid_layers = 1

    MAX_TRADES = 2000
    t_entry_day   = np.zeros(MAX_TRADES, dtype=np.int32)
    t_exit_day    = np.zeros(MAX_TRADES, dtype=np.int32)
    t_entry_price = np.zeros(MAX_TRADES)
    t_exit_price  = np.zeros(MAX_TRADES)
    t_r_multiple  = np.zeros(MAX_TRADES)
    t_pct_return  = np.zeros(MAX_TRADES)
    t_bars_held   = np.zeros(MAX_TRADES, dtype=np.int32)
    t_layers      = np.zeros(MAX_TRADES, dtype=np.int32)
    t_cnt = 0

    in_pos              = False
    n_layers             = 0
    blended_entry_price  = 0.0
    entry_day            = 0
    entry_risk           = 0.0    # anchored to layer 1 only, never re-anchored
    stop_loss_price      = 0.0    # anchored to layer 1 only, never re-anchored
    tp_trigger_price     = 0.0    # anchored to layer 1 only, never re-anchored
    half_sold            = False
    partial_exit_price   = 0.0
    high_since_entry     = 0.0

    pending_entry   = False   # fresh open OR a pyramid add-on, settled next open
    pending_exit    = False
    pending_partial = False

    for d in range(start_day, end_day):

        # ---- Phase A: settle yesterday's signal at today's open ----
        if pending_partial and in_pos:
            fill_price = opens[d]
            if not (fill_price > 0):
                fill_price = closes[d - 1]
            partial_exit_price = fill_price
            half_sold = True
            if stop_loss_price < blended_entry_price:
                stop_loss_price = blended_entry_price
            pending_partial = False

        if pending_exit and in_pos:
            fill_price = opens[d]
            if not (fill_price > 0):
                fill_price = closes[d - 1]
            if half_sold:
                blended_exit = 0.5 * partial_exit_price + 0.5 * fill_price
            else:
                blended_exit = fill_price
            pnl_per_unit = blended_exit - blended_entry_price
            if t_cnt < MAX_TRADES:
                t_entry_day[t_cnt]   = entry_day
                t_exit_day[t_cnt]    = d
                t_entry_price[t_cnt] = blended_entry_price
                t_exit_price[t_cnt]  = blended_exit
                t_r_multiple[t_cnt]  = pnl_per_unit / entry_risk if entry_risk > 0 else 0.0
                t_pct_return[t_cnt]  = pnl_per_unit / blended_entry_price if blended_entry_price > 0 else 0.0
                t_bars_held[t_cnt]   = d - entry_day
                t_layers[t_cnt]      = n_layers
                t_cnt += 1
            in_pos = False
            n_layers = 0
            half_sold = False
            pending_exit = False

        if pending_entry:
            fill_price = opens[d]
            if not (fill_price > 0):
                fill_price = closes[d - 1]
            if fill_price > 0:
                if not in_pos:
                    # fresh position, layer 1
                    in_pos = True
                    n_layers = 1
                    blended_entry_price = fill_price
                    entry_day = d
                    high_since_entry = fill_price
                    half_sold = False
                    e_atr = atr[d] if atr[d] > 0 else atr[d - 1]
                    if not (e_atr > 0):
                        e_atr = fill_price * 0.02
                    entry_risk = e_atr * sl_mult
                    stop_loss_price  = fill_price - e_atr * sl_mult
                    tp_trigger_price = fill_price + e_atr * tp_mult
                else:
                    # pyramid add-on -- blend cost basis, everything else
                    # (stop/tp/entry_risk) stays anchored to layer 1
                    n_layers += 1
                    blended_entry_price = (blended_entry_price * (n_layers - 1) + fill_price) / n_layers
                    high_since_entry = max(high_since_entry, fill_price)
            pending_entry = False

        curr_close = closes[d]
        prev_close = closes[d - 1]

        btc_bullish = True
        if use_btc_entry_gate or use_btc_exit_override:
            btc_bullish = btc_close[d] > btc_ma[d]

        # ---- exit evaluation (close-based, flags for tomorrow's open) ----
        if in_pos and not pending_exit:
            if curr_close > 0:
                high_since_entry = max(high_since_entry, curr_close)

            override_exit = use_btc_exit_override and (not btc_bullish)

            should_exit_full    = False
            should_exit_partial = False

            if exit_type == EXIT_HYBRID:
                if not half_sold and curr_close >= tp_trigger_price:
                    should_exit_partial = True
                if half_sold:
                    potential_new_sl = curr_close - atr[d] * trail_mult
                    if potential_new_sl > stop_loss_price:
                        stop_loss_price = potential_new_sl
                if curr_close < stop_loss_price:
                    should_exit_full = True
            elif exit_type == EXIT_PCT_TRAIL:
                if curr_close < high_since_entry * (1.0 - trail_pct / 100.0):
                    should_exit_full = True
            elif exit_type == EXIT_ATR_TRAIL:
                if curr_close < high_since_entry - (exit_atr_mult * atr[d]):
                    should_exit_full = True
            elif exit_type == EXIT_MA_CROSSUNDER:
                if prev_close >= exit_ma[d - 1] and curr_close < exit_ma[d]:
                    should_exit_full = True
            elif exit_type == EXIT_RSI_CROSSUNDER:
                if exit_rsi_fast[d - 1] >= exit_rsi_slow[d - 1] and exit_rsi_fast[d] < exit_rsi_slow[d]:
                    should_exit_full = True
            elif exit_type == EXIT_MA_XOVER_EXIT:
                if exit_xover_short[d - 1] >= exit_xover_long[d - 1] and exit_xover_short[d] < exit_xover_long[d]:
                    should_exit_full = True

            if override_exit:
                should_exit_full    = True
                should_exit_partial = False

            if should_exit_full:
                pending_exit = True
            elif should_exit_partial:
                pending_partial = True

        # ---- entry evaluation -- same signal fires whether opening fresh
        #      or adding a pyramid layer to an existing winner ----
        can_add = (not in_pos) or (n_layers < max_pyramid_layers)
        if can_add and (not pending_entry) and (not pending_exit):
            entry_signal = False
            if entry_type == ENTRY_MA_BREAKOUT:
                if prev_close <= entry_ma[d - 1] and curr_close > entry_ma[d]:
                    entry_signal = True
            elif entry_type == ENTRY_RSI_XOVER:
                if rsi_fast[d - 1] <= rsi_slow[d - 1] and rsi_fast[d] > rsi_slow[d]:
                    entry_signal = True
                    if use_rsi_trend_filter and not (curr_close > rsi_trend_ma[d]):
                        entry_signal = False
            elif entry_type == ENTRY_MA_XOVER:
                if xover_short[d - 1] <= xover_long[d - 1] and xover_short[d] > xover_long[d]:
                    entry_signal = True

            if entry_signal and use_btc_entry_gate and not btc_bullish:
                entry_signal = False
            if entry_signal and adx_threshold > 0.0 and adx[d] < adx_threshold:
                entry_signal = False

            if entry_signal:
                pending_entry = True

    return (t_entry_day[:t_cnt], t_exit_day[:t_cnt], t_entry_price[:t_cnt],
            t_exit_price[:t_cnt], t_r_multiple[:t_cnt], t_pct_return[:t_cnt],
            t_bars_held[:t_cnt], t_layers[:t_cnt])


# ==========================================
# 4b. MONEY-WEIGHTED MATH REMOVED -- no cash flows to solve an IRR over.
# ==========================================

# ==========================================
# 4c. OVERFITTING DIAGNOSTIC -- DSR (Bailey & Lopez de Prado, 2014)
# Same math as main.py, applied to the trade-level R-multiple distribution
# (mean/std of R across all trades) instead of a daily portfolio series.
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


def deflated_sharpe_ratio(sr_hat, all_trial_srs, T, skew, kurt):
    n_trials = len(all_trial_srs)
    if n_trials < 2:
        return probabilistic_sharpe_ratio(sr_hat, 0.0, T, skew, kurt), 0.0, n_trials
    mean_sr = sum(all_trial_srs) / n_trials
    var_sr  = sum((s - mean_sr) ** 2 for s in all_trial_srs) / n_trials
    sr_std  = math.sqrt(var_sr)
    sr0 = expected_max_sharpe(sr_std, n_trials)
    dsr = probabilistic_sharpe_ratio(sr_hat, sr0, T, skew, kurt)
    return dsr, sr0, n_trials


# ==========================================
# 5. DATA PREPARATION (unchanged data-infra from main.py)
# ==========================================

def build_eligibility_mask(n_days, n_stocks):
    return np.ones((n_days, n_stocks), dtype=np.bool_)


def _binance_download_klines(symbol, start_date_str, interval='1d'):
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
        time.sleep(0.15)

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=[
        'OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime',
        'QuoteVol', 'Trades', 'TakerBaseVol', 'TakerQuoteVol', 'Ignore'
    ])
    df['Date'] = pd.to_datetime(df['OpenTime'], unit='ms').dt.normalize()
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = df[col].astype(float)
    df = df[['Date', 'Open', 'High', 'Low', 'Close']].drop_duplicates('Date').set_index('Date')
    return df


def prepare_matrix_data(tickers, data_dir="data_crypto"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    raw_dfs      = {}
    master_dates = set()

    for ticker in tqdm(tickers, desc="Downloading Binance klines"):
        file_path = f"{data_dir}/{ticker}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        else:
            df = _binance_download_klines(ticker, START_DATE)
            if df is not None and len(df) > 200:
                df.to_csv(file_path)
        if df is not None and len(df) >= MIN_HISTORY_DAYS:
            raw_dfs[ticker] = df
            master_dates.update(df.index)
        elif ticker == 'BTCUSDT':
            raise RuntimeError("Could not download BTCUSDT history.")

    master_dates = sorted(list(master_dates))
    master_df    = pd.DataFrame(index=master_dates)
    master_df['Year'] = master_df.index.year

    n_days   = len(master_dates)
    stock_names = ['BTCUSDT'] + [t for t in raw_dfs.keys() if t != 'BTCUSDT']
    n_stocks = len(stock_names)

    opens      = np.zeros((n_days, n_stocks))
    highs      = np.zeros((n_days, n_stocks))
    lows       = np.zeros((n_days, n_stocks))
    closes     = np.zeros((n_days, n_stocks))
    atr_matrix = np.zeros((n_days, n_stocks))
    adx_matrix = np.zeros((n_days, n_stocks))

    for i, ticker in enumerate(tqdm(stock_names, desc="Building matrix")):
        df = raw_dfs[ticker].reindex(master_dates)
        opens[:,  i] = df['Open'].ffill().values
        highs[:,  i] = df['High'].ffill().values
        lows[:,   i] = df['Low'].ffill().values
        closes[:, i] = df['Close'].ffill().values
        atr_matrix[:, i] = calc_atr_wilder(highs[:, i], lows[:, i], closes[:, i], period=14)
        adx_matrix[:, i] = calc_adx(highs[:, i], lows[:, i], closes[:, i], period=14)

    assert stock_names[0] == 'BTCUSDT'

    eligible_mask = build_eligibility_mask(n_days, n_stocks)
    years_arr = master_df['Year'].values.astype(np.int32)

    return (opens, closes, atr_matrix, adx_matrix, years_arr,
            stock_names, eligible_mask, master_dates)


# ==========================================
# 6. EVALUATE PARAMS -- runs every coin independently, aggregates the
#    trade log, scores it.
# ==========================================

_ZERO_CACHE = {}

def _zeros_like(n_days):
    if n_days not in _ZERO_CACHE:
        _ZERO_CACHE[n_days] = np.zeros(n_days)
    return _ZERO_CACHE[n_days]


def evaluate_params_signal(p, opens, closes, atr, adx, years_arr,
                            n_stocks, start_day=0, end_day=-1, is_oos=False,
                            min_years_required=None):
    n_days = closes.shape[0]
    if end_day < 0: end_day = n_days - 1

    entry_type = p['entry_type']
    exit_type  = p['exit_type']
    use_btc_entry_gate  = p['use_btc_entry_gate']
    use_btc_exit_override = p['use_btc_exit_override']
    use_rsi_trend_filter  = p['use_rsi_trend_filter']

    zeros = _zeros_like(n_days)

    # -- entry-side indicator arrays, computed only if needed --
    entry_ma_all    = None
    xover_short_all = None
    xover_long_all  = None
    rsi_fast_all    = None
    rsi_slow_all    = None
    rsi_trend_all   = None

    if entry_type == ENTRY_MA_BREAKOUT:
        entry_ma_all = np.zeros((n_days, n_stocks))
        for s in range(n_stocks):
            entry_ma_all[:, s] = get_ma_cached(closes, s, p['entry_ma_len'], p['entry_ma_type'])
    elif entry_type == ENTRY_RSI_XOVER:
        rsi_fast_all = np.zeros((n_days, n_stocks))
        rsi_slow_all = np.zeros((n_days, n_stocks))
        for s in range(n_stocks):
            rsi_fast_all[:, s] = get_smoothed_rsi(closes, s, p['rsi_f_len'], p['rsi_f_smt'])
            rsi_slow_all[:, s] = get_smoothed_rsi(closes, s, p['rsi_s_len'], p['rsi_s_smt'])
        if use_rsi_trend_filter:
            rsi_trend_all = np.zeros((n_days, n_stocks))
            for s in range(n_stocks):
                rsi_trend_all[:, s] = get_ma_cached(closes, s, p['rsi_trend_ma_len'], p['rsi_trend_ma_type'])
    elif entry_type == ENTRY_MA_XOVER:
        xover_short_all = np.zeros((n_days, n_stocks))
        xover_long_all  = np.zeros((n_days, n_stocks))
        for s in range(n_stocks):
            xover_short_all[:, s] = get_ma_cached(closes, s, p['xover_short_len'], p['xover_short_type'])
            xover_long_all[:, s]  = get_ma_cached(closes, s, p['xover_long_len'], p['xover_long_type'])

    # -- exit-side indicator arrays --
    exit_ma_all         = None
    exit_xover_short_all = None
    exit_xover_long_all  = None
    exit_rsi_fast_all    = None
    exit_rsi_slow_all    = None

    if exit_type == EXIT_MA_CROSSUNDER:
        exit_ma_all = np.zeros((n_days, n_stocks))
        for s in range(n_stocks):
            exit_ma_all[:, s] = get_ma_cached(closes, s, p['exit_ma_len'], p['exit_ma_type'])
    elif exit_type == EXIT_RSI_CROSSUNDER:
        exit_rsi_fast_all = np.zeros((n_days, n_stocks))
        exit_rsi_slow_all = np.zeros((n_days, n_stocks))
        for s in range(n_stocks):
            exit_rsi_fast_all[:, s] = get_smoothed_rsi(closes, s, p['exit_rsi_f_len'], p['exit_rsi_f_smt'])
            exit_rsi_slow_all[:, s] = get_smoothed_rsi(closes, s, p['exit_rsi_s_len'], p['exit_rsi_s_smt'])
    elif exit_type == EXIT_MA_XOVER_EXIT:
        exit_xover_short_all = np.zeros((n_days, n_stocks))
        exit_xover_long_all  = np.zeros((n_days, n_stocks))
        for s in range(n_stocks):
            exit_xover_short_all[:, s] = get_ma_cached(closes, s, p['exit_xover_short_len'], p['exit_xover_short_type'])
            exit_xover_long_all[:, s]  = get_ma_cached(closes, s, p['exit_xover_long_len'], p['exit_xover_long_type'])

    # -- BTC regime arrays (only if either toggle uses them) --
    if use_btc_entry_gate or use_btc_exit_override:
        btc_ma_all = get_ma_cached(closes, 0, p['btc_ma_len'], p['btc_ma_type'])
    else:
        btc_ma_all = zeros
    btc_close_col = closes[:, 0]

    all_entry_day, all_exit_day, all_entry_price, all_exit_price = [], [], [], []
    all_r, all_pct, all_bars, all_stock_idx, all_layers = [], [], [], [], []
    max_pyramid_layers = int(p.get('max_pyramid_layers', 1))

    for s in range(n_stocks):
        entry_ma    = entry_ma_all[:, s]    if entry_ma_all    is not None else zeros
        xover_short = xover_short_all[:, s] if xover_short_all is not None else zeros
        xover_long  = xover_long_all[:, s]  if xover_long_all  is not None else zeros
        rsi_fast    = rsi_fast_all[:, s]    if rsi_fast_all    is not None else zeros
        rsi_slow    = rsi_slow_all[:, s]    if rsi_slow_all    is not None else zeros
        rsi_trend   = rsi_trend_all[:, s]   if rsi_trend_all   is not None else zeros
        exit_ma          = exit_ma_all[:, s]          if exit_ma_all          is not None else zeros
        exit_xover_short = exit_xover_short_all[:, s] if exit_xover_short_all is not None else zeros
        exit_xover_long  = exit_xover_long_all[:, s]  if exit_xover_long_all  is not None else zeros
        exit_rsi_fast    = exit_rsi_fast_all[:, s]    if exit_rsi_fast_all    is not None else zeros
        exit_rsi_slow    = exit_rsi_slow_all[:, s]    if exit_rsi_slow_all    is not None else zeros

        result = simulate_signal_trades(
            opens[:, s], closes[:, s], atr[:, s], adx[:, s],
            entry_ma, xover_short, xover_long,
            rsi_fast, rsi_slow, rsi_trend,
            btc_close_col, btc_ma_all,
            exit_ma, exit_xover_short, exit_xover_long,
            exit_rsi_fast, exit_rsi_slow,
            entry_type, exit_type,
            use_btc_entry_gate, use_btc_exit_override, use_rsi_trend_filter,
            p.get('adx_thresh', 0.0),
            p['sl_mult'], p['tp_mult'], p['trail_mult'], p['trail_pct'], p['exit_atr_mult'],
            max_pyramid_layers,
            int(start_day), int(end_day)
        )
        (e_day, x_day, e_price, x_price, r_mult, pct_ret, bars, layers) = result
        if len(e_day) == 0:
            continue
        # only keep trades whose entry AND exit both land inside the window
        mask = (e_day >= start_day) & (x_day < end_day)
        if not mask.any():
            continue
        all_entry_day.append(e_day[mask])
        all_exit_day.append(x_day[mask])
        all_entry_price.append(e_price[mask])
        all_exit_price.append(x_price[mask])
        all_r.append(r_mult[mask])
        all_pct.append(pct_ret[mask])
        all_bars.append(bars[mask])
        all_layers.append(layers[mask])
        all_stock_idx.append(np.full(mask.sum(), s, dtype=np.int32))

    if not all_r:
        return -999.0, {}

    entry_days   = np.concatenate(all_entry_day)
    exit_days    = np.concatenate(all_exit_day)
    entry_prices = np.concatenate(all_entry_price)
    exit_prices  = np.concatenate(all_exit_price)
    r_multiple   = np.concatenate(all_r)
    pct_return   = np.concatenate(all_pct)
    bars_held    = np.concatenate(all_bars)
    layers_used  = np.concatenate(all_layers)
    stock_idx    = np.concatenate(all_stock_idx)
    entry_years  = years_arr[entry_days]

    # window's own year span, for recency weighting -- deliberately based
    # on every DAY in [start_day, end_day), not just years the strategy
    # happened to trade in. A strategy that only ever entered once, early,
    # and never traded again should NOT get treated as "perfectly recent"
    # just because it has a single data point -- recency is measured
    # against the actual calendar span being evaluated.
    window_years = years_arr[start_day:end_day]
    global_min_year = int(window_years.min())
    global_max_year = int(window_years.max())

    return compute_score_signal(r_multiple, pct_return, bars_held, entry_years,
                                 stock_idx, entry_days, exit_days,
                                 entry_prices, exit_prices, layers_used,
                                 global_min_year, global_max_year, is_oos=is_oos,
                                 min_years_required=min_years_required)


# ==========================================
# 7. SCORING (trade-log based)
# ==========================================

def compute_naive_risk_metrics(entry_days, exit_days, r_multiple,
                                risk_per_trade_pct=NAIVE_RISK_PCT_PER_TRADE):
    """
    Advisory, trade-log-only proxies for portfolio-level risk -- NOT a real
    portfolio simulation (that's crypto_portfolio_optimizer.py's job; this
    has no position sizing, concurrency limits, or capital constraints).
    Exists because trade-level stats (SQN, expectancy, profit factor) are
    computed on an unordered bag of R-multiples and can look perfectly
    healthy even when losses cluster together in calendar time -- which a
    real, capital-constrained portfolio would feel as a brutal, correlated
    drawdown. Cheap enough to run on every scored trial.

    Returns a dict with:
      naive_max_dd, naive_ulcer_index, naive_total_return -- from a naive
        fixed-fractional-risk equity curve, trades applied in EXIT-day
        order (ignores real concurrency, so this is a LOWER bound on real
        drawdown if trades genuinely overlap -- see estimate_naive_drawdown
        note in run_optimization for the same caveat).
      max_consecutive_losses -- longest run of consecutive losing trades
        in exit-day order.
      max_concurrent_open, max_concurrent_losers, worst_day_loser_fraction
        -- a sweep over calendar days (using entry_day..exit_day as each
        trade's "open" window) to find the single most-correlated moment:
        how many trades were open at once, how many of those were eventual
        losers, and what fraction that represents.
    """
    n = len(r_multiple)
    empty = {
        'naive_max_dd': 0.0, 'naive_ulcer_index': 0.0, 'naive_total_return': 0.0,
        'max_consecutive_losses': 0, 'expected_streak': 0.0, 'streak_ratio': 0.0,
        'max_concurrent_open': 0, 'max_concurrent_losers': 0,
        'worst_day_loser_fraction': 0.0,
    }
    if n == 0:
        return empty
    entry_days = np.asarray(entry_days)
    exit_days = np.asarray(exit_days)
    r = np.asarray(r_multiple, dtype=np.float64)

    # -- naive fixed-fractional equity curve: max drawdown + Ulcer Index --
    order = np.argsort(exit_days)
    r_sorted = r[order]
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    dd_sq_sum = 0.0
    for rr in r_sorted:
        equity *= (1.0 + rr * risk_per_trade_pct)
        equity = max(equity, 1e-6)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd
        dd_sq_sum += dd * dd
    ulcer_index = math.sqrt(dd_sq_sum / n)
    naive_total_return = equity - 1.0

    # -- longest consecutive losing streak, compared to what's statistically
    #    expected at this trial's own loss rate (Erdos-Renyi longest-run
    #    approximation) so a healthy low-win-rate trend system doesn't get
    #    unfairly flagged for behaving exactly as it should --
    max_streak = 0
    cur_streak = 0
    for rr in r_sorted:
        if rr < 0:
            cur_streak += 1
            if cur_streak > max_streak:
                max_streak = cur_streak
        else:
            cur_streak = 0
    loss_rate = float((r < 0).mean())
    if 0.0 < loss_rate < 1.0 and n > 5:
        expected_streak = math.log(n) / math.log(1.0 / loss_rate)
    else:
        expected_streak = 0.0
    streak_ratio = (max_streak / expected_streak) if expected_streak > 0 else 0.0

    # -- concurrent / correlated loss exposure: sweep-line over calendar
    #    days using each trade's [entry_day, exit_day] as its open window --
    day0 = int(entry_days.min())
    day1 = int(exit_days.max()) + 2
    span = day1 - day0 + 1
    open_delta = np.zeros(span)
    loser_delta = np.zeros(span)
    losers_mask = r < 0
    for i in range(n):
        s = int(entry_days[i]) - day0
        e = int(exit_days[i]) - day0 + 1
        open_delta[s] += 1
        open_delta[e] -= 1
        if losers_mask[i]:
            loser_delta[s] += 1
            loser_delta[e] -= 1
    open_curve = np.cumsum(open_delta)
    loser_curve = np.cumsum(loser_delta)
    peak_day = int(np.argmax(loser_curve))
    max_concurrent_losers = int(loser_curve[peak_day])
    worst_day_total_open = int(open_curve[peak_day])
    max_concurrent_open = int(open_curve.max())
    worst_day_loser_fraction = (max_concurrent_losers / worst_day_total_open
                                 if worst_day_total_open > 0 else 0.0)

    return {
        'naive_max_dd': max_dd, 'naive_ulcer_index': ulcer_index,
        'naive_total_return': naive_total_return,
        'max_consecutive_losses': max_streak, 'expected_streak': expected_streak,
        'streak_ratio': streak_ratio,
        'max_concurrent_open': max_concurrent_open,
        'max_concurrent_losers': max_concurrent_losers,
        'worst_day_loser_fraction': worst_day_loser_fraction,
    }


def compute_score_signal(r_multiple, pct_return, bars_held, entry_years,
                          stock_idx, entry_days, exit_days,
                          entry_prices, exit_prices, layers_used,
                          global_min_year, global_max_year, is_oos=False,
                          min_years_required=None):
    req_years = min_years_required if min_years_required is not None else MIN_YEARS_GATE
    """
    v1.2 SCORING -- adds recency weighting (v1.1 fixed the sum-vs-average
    and outlier-domination bugs; see CHANGELOG at top of file).

    RECENCY WEIGHTING: the consistency term is no longer a flat median
    across years -- each year's average trade quality is weighted by how
    RECENT that year is within the window being scored (weight ramps from
    RECENCY_WEIGHT_MIN for the oldest year in the window to
    RECENCY_WEIGHT_MAX for the most recent), then combined into a
    weighted mean. A strategy whose profit came entirely from years 1-2 of
    an 8-year window and did nothing since now scores WORSE on this term
    than one with the same total edge spread evenly, or concentrated
    recently -- directly targeting "made money early and then sat idle"
    per your instruction. Critically, the year range used for the weights
    is the WINDOW's actual span (global_min_year/global_max_year, passed
    in from every day in [start_day,end_day), not just the years the
    strategy happened to trade in) -- so a strategy with only one distant
    trading year doesn't accidentally get treated as "perfectly recent"
    just because it has no other data point to compare against.
    median_yearly_avg_r is still computed and reported alongside as a
    non-scored diagnostic, so you can see what the number would have been
    without the recency tilt.
    """
    n_trades = len(r_multiple)
    target_trades = max(15, int(MIN_TRADES_GATE * (WFO_OOS_PCT / WFO_IS_PCT))) if is_oos else MIN_TRADES_GATE

    r_capped = np.clip(r_multiple, -R_WINSORIZE_CAP, R_WINSORIZE_CAP)

    metrics = {
        'trades': int(n_trades),
        'r_multiple': r_capped,        # capped -- used for every stat below
        'r_multiple_raw': r_multiple,  # UNCAPPED -- for diagnostics/reporting only
        'pct_return': pct_return,
        'bars_held': bars_held,
        'entry_years': entry_years,
        'stock_idx': stock_idx,
        'entry_days': entry_days, 'exit_days': exit_days,
        'entry_prices': entry_prices, 'exit_prices': exit_prices,
        'layers_used': layers_used,
    }

    if n_trades < target_trades:
        return -999.0, metrics

    distinct_years = sorted(set(int(y) for y in entry_years))
    if len(distinct_years) < req_years:
        return -999.0, metrics

    wins_mask = r_capped > 0
    win_rate = float(wins_mask.mean())
    if win_rate < MIN_WIN_RATE_GATE:
        return -999.0, metrics

    gross_win  = float(r_capped[wins_mask].sum()) if wins_mask.any() else 0.0
    gross_loss = float(-r_capped[~wins_mask].sum()) if (~wins_mask).any() else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    if profit_factor < 1.10:
        return -999.0, metrics

    avg_win_r  = float(r_capped[wins_mask].mean()) if wins_mask.any() else 0.0
    avg_loss_r = float(-r_capped[~wins_mask].mean()) if (~wins_mask).any() else 0.0
    expectancy_r = win_rate * avg_win_r - (1.0 - win_rate) * avg_loss_r
    if expectancy_r <= 0:
        return -999.0, metrics

    mean_r   = float(r_capped.mean())
    median_r = float(np.median(r_capped))
    std_r    = float(r_capped.std())
    sqn = (mean_r / std_r) * math.sqrt(min(n_trades, 100)) if std_r > 0 else 0.0
    sqn_capped = min(sqn, SQN_CAP)

    # -- per-year average R (equal-weight scale, not a sum -- v1.1 fix) --
    year_r_avg = {}
    year_r_sum_raw = {}   # kept (uncapped) for the concentration diagnostic below
    for yr in distinct_years:
        yr_mask = entry_years == yr
        year_r_avg[yr] = float(r_capped[yr_mask].mean())
        year_r_sum_raw[yr] = float(r_multiple[yr_mask].sum())
    median_yearly_avg_r = float(np.median(list(year_r_avg.values())))   # reported only, not scored

    # -- recency-weighted average R (v1.2 -- THIS is what's scored) --
    weights = [_recency_weight(yr, global_min_year, global_max_year) for yr in distinct_years]
    w_sum = sum(weights)
    recency_weighted_avg_r = (sum(w * year_r_avg[yr] for w, yr in zip(weights, distinct_years)) / w_sum
                               if w_sum > 0 else median_yearly_avg_r)

    # -- concentration diagnostics (SOFT penalty only, not a hard gate;
    #    computed on RAW R since this is specifically about "did one crazy
    #    trade/year/coin account for an outsized share of nominal profit",
    #    which is exactly the thing capping would hide) --
    total_r_raw = float(r_multiple.sum())
    max_year_share = 0.0
    max_coin_share = 0.0
    if total_r_raw > 0:
        max_year_share = max(v / total_r_raw for v in year_r_sum_raw.values())
        coin_r_sums = {}
        for c in np.unique(stock_idx):
            coin_r_sums[int(c)] = float(r_multiple[stock_idx == c].sum())
        max_coin_share = max(v / total_r_raw for v in coin_r_sums.values())

    concentration_penalty = 1.0
    if max_year_share > CONCENTRATION_SOFT_THRESHOLD or max_coin_share > CONCENTRATION_SOFT_THRESHOLD:
        concentration_penalty = CONCENTRATION_PENALTY_MULT

    # -- naive portfolio-risk penalties (see compute_naive_risk_metrics'
    #    docstring): three independent, mild soft penalties, so a trial
    #    that trips more than one gets a meaningfully lower score without
    #    any single dimension being able to zero it out entirely --
    risk_m = compute_naive_risk_metrics(entry_days, exit_days, r_multiple)
    risk_penalty = 1.0
    if risk_m['naive_max_dd'] > NAIVE_DD_SOFT_THRESHOLD:
        risk_penalty *= NAIVE_DD_PENALTY_MULT
    if risk_m['streak_ratio'] > STREAK_RATIO_SOFT_THRESHOLD:
        risk_penalty *= STREAK_PENALTY_MULT
    if (risk_m['max_concurrent_losers'] >= CORRELATED_LOSS_MIN_COUNT
            and risk_m['worst_day_loser_fraction'] > CORRELATED_LOSS_FRACTION_THRESHOLD):
        risk_penalty *= CORRELATED_LOSS_PENALTY_MULT

    wr_bonus = max(0.0, win_rate - 0.50) * 4.0
    stat_conf = min(1.0, math.sqrt(n_trades / target_trades))

    score = (
        sqn_capped              * W_SQN +
        expectancy_r            * W_EXPECTANCY +
        recency_weighted_avg_r  * W_RECENCY +
        wr_bonus                * W_WR_BONUS
    ) * stat_conf * concentration_penalty * risk_penalty

    # -- entry-date clustering diagnostic (separate from R-based concentration
    #    above): what fraction of TRADE COUNT entered in the single busiest
    #    calendar year, regardless of profitability. High avg_bars_held +
    #    high entry clustering together are the signature of "found one
    #    historical regime and rode it across several coins" rather than a
    #    repeatable, frequently-firing signal. --
    entry_year_counts = {}
    for yr in distinct_years:
        entry_year_counts[yr] = int((entry_years == yr).sum())
    max_entry_year_concentration = max(entry_year_counts.values()) / n_trades if n_trades > 0 else 0.0

    mean_pct = float(pct_return.mean()) if pct_return is not None else 0.0
    median_pct = float(np.median(pct_return)) if pct_return is not None else 0.0
    avg_layers = float(layers_used.mean()) if layers_used is not None and len(layers_used) > 0 else 1.0
    pct_pyramided = float((layers_used > 1).mean()) if layers_used is not None and len(layers_used) > 0 else 0.0

    metrics.update({
        'win_rate': win_rate, 'profit_factor': profit_factor,
        'avg_win_r': avg_win_r, 'avg_loss_r': avg_loss_r,
        'expectancy_r': expectancy_r,
        'mean_r': mean_r, 'median_r': median_r, 'std_r': std_r,
        'mean_r_raw': float(r_multiple.mean()), 'median_r_raw': float(np.median(r_multiple)),
        'mean_pct': mean_pct, 'median_pct': median_pct,
        'max_single_trade_r': float(r_multiple.max()), 'min_single_trade_r': float(r_multiple.min()),
        'sqn': sqn, 'sqn_capped': sqn_capped,
        'median_yearly_avg_r': median_yearly_avg_r,          # reported diagnostic only
        'recency_weighted_avg_r': recency_weighted_avg_r,    # this is what's scored
        'year_r_avg': year_r_avg, 'year_r_sum_raw': year_r_sum_raw,
        'year_weights': dict(zip(distinct_years, weights)),
        'global_min_year': global_min_year, 'global_max_year': global_max_year,
        'max_year_share': max_year_share, 'max_coin_share': max_coin_share,
        'concentration_penalty': concentration_penalty,
        'distinct_years': distinct_years,
        'distinct_coins': int(len(np.unique(stock_idx))),
        'avg_bars_held': float(bars_held.mean()) if n_trades > 0 else 0.0,
        'entry_year_counts': entry_year_counts,
        'max_entry_year_concentration': max_entry_year_concentration,
        'avg_pyramid_layers': avg_layers, 'pct_trades_pyramided': pct_pyramided,
        'risk_penalty': risk_penalty,
        'naive_max_dd': risk_m['naive_max_dd'],
        'naive_ulcer_index': risk_m['naive_ulcer_index'],
        'max_consecutive_losses': risk_m['max_consecutive_losses'],
        'streak_ratio': risk_m['streak_ratio'],
        'max_concurrent_open': risk_m['max_concurrent_open'],
        'max_concurrent_losers': risk_m['max_concurrent_losers'],
        'worst_day_loser_fraction': risk_m['worst_day_loser_fraction'],
    })
    return score, metrics


def diagnose_gates_signal(r_multiple, pct_return, bars_held, entry_years,
                           stock_idx, is_oos=False):
    """Same-order gate breakdown as compute_score_signal, for debugging a
    -999 the way main.py's diagnose_gates does."""
    n_trades = len(r_multiple)
    target_trades = max(15, int(MIN_TRADES_GATE * (WFO_OOS_PCT / WFO_IS_PCT))) if is_oos else MIN_TRADES_GATE
    rows = [('trades >= target', n_trades >= target_trades, n_trades, f'>= {target_trades}')]
    if n_trades == 0:
        return rows
    distinct_years = sorted(set(int(y) for y in entry_years))
    rows.append(('distinct years >= gate', len(distinct_years) >= MIN_YEARS_GATE,
                 len(distinct_years), f'>= {MIN_YEARS_GATE}'))
    wins_mask = r_multiple > 0
    win_rate = float(wins_mask.mean())
    rows.append(('win_rate >= floor', win_rate >= MIN_WIN_RATE_GATE,
                 f'{win_rate*100:.1f}%', f'>= {MIN_WIN_RATE_GATE*100:.0f}%'))
    gross_win  = float(r_multiple[wins_mask].sum()) if wins_mask.any() else 0.0
    gross_loss = float(-r_multiple[~wins_mask].sum()) if (~wins_mask).any() else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    rows.append(('profit_factor >= 1.10', profit_factor >= 1.10, f'{profit_factor:.2f}', '>= 1.10'))
    avg_win_r  = float(r_multiple[wins_mask].mean()) if wins_mask.any() else 0.0
    avg_loss_r = float(-r_multiple[~wins_mask].mean()) if (~wins_mask).any() else 0.0
    expectancy_r = win_rate * avg_win_r - (1.0 - win_rate) * avg_loss_r
    rows.append(('expectancy_R > 0', expectancy_r > 0, f'{expectancy_r:.3f}R', '> 0'))
    return rows


def print_gate_diagnosis_signal(r_multiple, pct_return, bars_held, entry_years,
                                 stock_idx, is_oos=False, label="GATE DIAGNOSIS"):
    rows = diagnose_gates_signal(r_multiple, pct_return, bars_held, entry_years, stock_idx, is_oos)
    print(f"\n{'-'*60}\n{label}\n{'-'*60}")
    first_fail = False
    for name, passed, actual, threshold in rows:
        marker = "PASS" if passed else "FAIL"
        print(f"  [{marker}] {name:<26} actual={actual!s:<10} needed {threshold}")
        if not passed and not first_fail:
            print("       ^-- this is the gate that produced the -999 score")
            first_fail = True
    print("-" * 60)


# ==========================================
# 8. NEIGHBORHOOD STABILITY
# ==========================================

def passes_neighborhood_check(p, base_score, opens, closes, atr, adx,
                               years_arr, n_stocks, start_day, end_day):
    if base_score <= 0:
        return True

    perturbations = []
    if p['entry_type'] == ENTRY_MA_BREAKOUT:
        perturbations += [{'entry_ma_len': p['entry_ma_len'] + 10},
                           {'entry_ma_len': max(MA_LEN_MIN, p['entry_ma_len'] - 10)}]
    elif p['entry_type'] == ENTRY_RSI_XOVER:
        perturbations += [{'rsi_f_len': p['rsi_f_len'] + 5},
                           {'rsi_s_len': p['rsi_s_len'] + 5}]
    elif p['entry_type'] == ENTRY_MA_XOVER:
        perturbations += [{'xover_short_len': p['xover_short_len'] + 10},
                           {'xover_long_len': p['xover_long_len'] + 10}]

    perturbations += [{'sl_mult': p['sl_mult'] * 0.85}, {'sl_mult': p['sl_mult'] * 1.15}]

    for delta in perturbations:
        n_p = p.copy()
        n_p.update(delta)
        n_score, _ = evaluate_params_signal(n_p, opens, closes, atr, adx,
                                            years_arr, n_stocks,
                                            start_day, end_day, is_oos=False)
        if n_score < base_score * NEIGHBOR_THRESHOLD:
            return False
    return True


# ==========================================
# 9. WALK-FORWARD VALIDATION
# ==========================================

def run_wfo_validation(best_params, opens, closes, atr, adx, years_arr, n_stocks):
    n_days = closes.shape[0]
    is_end = int(n_days * WFO_IS_PCT)
    oos_start = is_end

    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION")
    print(f"   In-Sample days:     0 -> {is_end}")
    print(f"   Out-of-Sample days: {oos_start} -> {n_days}")
    print("=" * 60)

    is_score, is_m = evaluate_params_signal(best_params, opens, closes, atr, adx,
                                             years_arr, n_stocks,
                                             start_day=0, end_day=is_end, is_oos=False)
    oos_score, oos_m = evaluate_params_signal(best_params, opens, closes, atr, adx,
                                               years_arr, n_stocks,
                                               start_day=oos_start, end_day=n_days - 1, is_oos=True)

    robustness = max(0.0, oos_score / is_score) if is_score > 0 and oos_score > 0 else 0.0

    if is_score > 0:
        print(f"\nIn-Sample  Score: {is_score:.4f} | Trades: {is_m.get('trades',0)} | "
              f"WinRate: {is_m.get('win_rate',0)*100:.1f}% | PF: {is_m.get('profit_factor',0):.2f} | "
              f"ExpectancyR: {is_m.get('expectancy_r',0):.3f} | SQN: {is_m.get('sqn_capped',0):.2f} | "
              f"MedianYearlyAvgR: {is_m.get('median_yearly_avg_r',0):.2f} | "
              f"MeanR(raw): {is_m.get('mean_r_raw',0):.2f} | MedianR(raw): {is_m.get('median_r_raw',0):.2f}")
    else:
        r = is_m.get('r_multiple', np.array([]))
        print_gate_diagnosis_signal(r, is_m.get('pct_return', np.array([])),
                                    is_m.get('bars_held', np.array([])),
                                    is_m.get('entry_years', np.array([])),
                                    is_m.get('stock_idx', np.array([])),
                                    is_oos=False, label="IN-SAMPLE GATE DIAGNOSIS")

    if oos_score > -900:
        print(f"Out-of-Sample Score: {oos_score:.4f} | Trades: {oos_m.get('trades',0)} | "
              f"WinRate: {oos_m.get('win_rate',0)*100:.1f}% | PF: {oos_m.get('profit_factor',0):.2f} | "
              f"ExpectancyR: {oos_m.get('expectancy_r',0):.3f} | SQN: {oos_m.get('sqn_capped',0):.2f} | "
              f"MedianYearlyAvgR: {oos_m.get('median_yearly_avg_r',0):.2f} | "
              f"MeanR(raw): {oos_m.get('mean_r_raw',0):.2f} | MedianR(raw): {oos_m.get('median_r_raw',0):.2f}")
    else:
        print(f"Out-of-Sample Score: {oos_score:.4f} (strategy failed on OOS data)")
        r = oos_m.get('r_multiple', np.array([]))
        print_gate_diagnosis_signal(r, oos_m.get('pct_return', np.array([])),
                                    oos_m.get('bars_held', np.array([])),
                                    oos_m.get('entry_years', np.array([])),
                                    oos_m.get('stock_idx', np.array([])),
                                    is_oos=True, label="OUT-OF-SAMPLE GATE DIAGNOSIS")

    print(f"\nRobustness Ratio: {robustness:.1%}  ", end="")
    if robustness >= 0.70:
        print("EXCELLENT (>70%)")
    elif robustness >= 0.50:
        print("ACCEPTABLE (50-70%)")
    elif robustness >= 0.30:
        print("POOR (30-50%) -- likely overfit")
    else:
        print("REJECT (<30%) -- heavily overfit")

    return is_score, oos_score, is_m, oos_m, robustness


def report_yearly_table(metrics, label="YEAR-BY-YEAR BREAKDOWN"):
    year_r_avg = metrics.get('year_r_avg', {})
    year_r_sum_raw = metrics.get('year_r_sum_raw', {})
    entry_year_counts = metrics.get('entry_year_counts', {})
    year_weights = metrics.get('year_weights', {})
    if not year_r_avg:
        print("\n(No yearly breakdown available.)")
        return
    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)
    print("(Note: a trade's full result is counted in the year it was ENTERED, even if")
    print(" it was held for years -- a long-held trade can make one year look outsized.")
    print(" 'Recency Wt' is how much that year counts toward the score -- higher for")
    print(" more recent years, so profit from long ago counts for less.)")
    print(f"{'Year':<8}{'# Trades':>10}{'Avg R/trade':>14}{'Recency Wt':>12}{'Total R (real)':>16}{'Share':>10}")
    total_r_raw = sum(year_r_sum_raw.values())
    for yr in sorted(year_r_avg):
        share = (year_r_sum_raw[yr] / total_r_raw * 100.0) if total_r_raw > 0 else 0.0
        print(f"{yr:<8}{entry_year_counts.get(yr,0):>10}{year_r_avg[yr]:>14.2f}"
              f"{year_weights.get(yr,1.0):>12.2f}{year_r_sum_raw[yr]:>16.2f}{share:>9.1f}%")
    print(f"\nRecency-weighted avg-R/trade (THE scored consistency term): "
          f"{metrics.get('recency_weighted_avg_r', 0):.2f}")
    print(f"  (For comparison, plain median across years with no recency tilt: "
          f"{metrics.get('median_yearly_avg_r', 0):.2f} -- reported only, not scored)")
    print(f"Max single-year share of total profit: {metrics.get('max_year_share',0)*100:.1f}%  "
          f"(soft score penalty above {CONCENTRATION_SOFT_THRESHOLD*100:.0f}%)")
    print(f"Max single-coin share of total profit: {metrics.get('max_coin_share',0)*100:.1f}%  "
          f"(soft score penalty above {CONCENTRATION_SOFT_THRESHOLD*100:.0f}%)")
    print(f"Distinct coins traded: {metrics.get('distinct_coins', 0)}")

    avg_layers = metrics.get('avg_pyramid_layers', 1.0)
    pct_pyr = metrics.get('pct_trades_pyramided', 0.0)
    if avg_layers > 1.01 or pct_pyr > 0:
        print(f"Pyramiding: avg {avg_layers:.2f} layers/trade, {pct_pyr*100:.0f}% of trades added at least one layer")

    mean_pct, med_pct = metrics.get('mean_pct', 0)*100, metrics.get('median_pct', 0)*100
    mean_raw, med_raw = metrics.get('mean_r_raw', 0), metrics.get('median_r_raw', 0)
    gap_flag = "  <-- a handful of outsized trades are pulling the average up" \
               if mean_raw > med_raw * 2 and med_raw > 0 else ""
    print(f"\nTypical trade: entry to exit, price moved on average {mean_pct:+.1f}%, "
          f"but the MIDDLE trade only moved {med_pct:+.1f}%{gap_flag}")

    avg_days = metrics.get('avg_bars_held', 0)
    max_entry_conc = metrics.get('max_entry_year_concentration', 0)
    if avg_days > 180 or max_entry_conc > 0.40:
        print(f"\n** REGIME-CONCENTRATION CHECK **")
        print(f"   Average holding period: {avg_days:.0f} days ({avg_days/365:.1f} years).")
        print(f"   {max_entry_conc*100:.0f}% of all trades entered within a single calendar year.")
        print("   Long average holds combined with entries clustered in one window is the")
        print("   signature of a strategy that found ONE historical trend and rode it across")
        print("   several coins -- not necessarily a repeatable, frequently-firing signal.")
        print("   Weigh the out-of-sample result heavily before trusting this one.")


def print_top_trades(metrics, stock_names=None, master_dates=None, n=5):
    """Prints the biggest winners and losers by ACTUAL price move (%),
    with real coin names and dates -- the direct answer to 'what trades
    are actually driving these numbers', for a non-trader to eyeball.
    R-multiple is shown too, in parentheses, for anyone who wants it --
    but % is the number that means something without any trading
    background: it's just how much the coin's price moved between entry
    and exit."""
    pct = metrics.get('pct_return')
    r_raw = metrics.get('r_multiple_raw')
    if pct is None or len(pct) == 0:
        return
    stock_idx = metrics.get('stock_idx')
    entry_days = metrics.get('entry_days')
    exit_days = metrics.get('exit_days')
    bars_held = metrics.get('bars_held')

    order = np.argsort(pct)
    winners = order[::-1][:n]
    losers = order[:n]

    def _fmt_date(day_idx):
        if master_dates is not None and 0 <= day_idx < len(master_dates):
            d = master_dates[day_idx]
            return str(d.date()) if hasattr(d, 'date') else str(d)
        return f"day#{day_idx}"

    def _fmt_coin(s_idx):
        if stock_names is not None and 0 <= s_idx < len(stock_names):
            return stock_names[s_idx]
        return f"coin#{s_idx}"

    def _print_rows(idxs, title):
        print(f"\n{title}")
        print(f"  {'Coin':<12}{'Entry Date':>12}{'Exit Date':>12}{'Days Held':>10}{'% Move':>10}{'R':>9}")
        for i in idxs:
            print(f"  {_fmt_coin(int(stock_idx[i])):<12}{_fmt_date(int(entry_days[i])):>12}"
                  f"{_fmt_date(int(exit_days[i])):>12}{int(bars_held[i]):>10}"
                  f"{pct[i]*100:>+9.1f}%{'('+format(r_raw[i], '+.1f')+'R)':>9}")

    print("\n" + "-" * 74)
    print("TOP TRADES -- what actually happened, in plain terms (% = real price move)")
    print("-" * 74)
    _print_rows(winners, "Biggest winners (coin price moved up the most while held):")
    _print_rows(losers, "Biggest losers (coin price moved down the most while held):")
    print("-" * 74)


def run_overfitting_diagnostic(all_trial_srs, champion_metrics):
    if len(all_trial_srs) < 30:
        print("\n(Skipping DSR diagnostic -- need at least ~30 completed trials.)")
        return None
    r = champion_metrics.get('r_multiple', np.array([]))
    if len(r) < 3:
        return None
    skew = float(pd.Series(r).skew()) if len(r) > 2 else -0.3
    kurt = float(pd.Series(r).kurtosis() + 3) if len(r) > 3 else 5.0
    sr_hat = champion_metrics.get('mean_r', 0.0) / champion_metrics.get('std_r', 1.0) if champion_metrics.get('std_r', 0) > 0 else 0.0
    T = champion_metrics.get('trades', 0)

    dsr, sr0, n_trials = deflated_sharpe_ratio(sr_hat, all_trial_srs, T, skew, kurt)
    print("\n" + "=" * 60)
    print("OVERFITTING DIAGNOSTIC -- Deflated Sharpe Ratio (trade-level)")
    print("=" * 60)
    print(f"  Completed trials used:        {n_trials}")
    print(f"  Champion's trade-level Sharpe (mean_R/std_R): {sr_hat:.3f}")
    print(f"  Expected max from {n_trials} noise trials:    {sr0:.3f}")
    print(f"  Deflated Sharpe Ratio (probability):          {dsr:.3f}")
    if dsr > 0.95:
        print("  -> Clears the noise threshold with high confidence.")
    elif dsr > 0.70:
        print("  -> Plausibly real edge, not overwhelming. Weigh OOS heavily.")
    else:
        print("  -> Cannot statistically distinguish from noise.")
    return dsr


# ==========================================
# 10. CONSOLE REPORTER
# ==========================================

MA_TYPE_NAMES = {0: 'SMA', 1: 'EMA', 2: 'DEMA', 3: 'WMA', 4: 'SMMA'}
ENTRY_TYPE_NAMES = {0: 'MA Breakout', 1: 'RSI Crossover', 2: 'MA Crossover'}
EXIT_TYPE_NAMES = {0: 'Hybrid ATR TP+Trail', 1: '%-Trail from High',
                    2: 'ATR-Trail from High', 3: 'MA Crossunder',
                    4: 'RSI Crossunder', 5: 'MA Crossover Exit'}

LEGEND = """
------------------------------------------------------------------------
WHAT THE NUMBERS MEAN (plain-language)
------------------------------------------------------------------------
  R / R-multiple: how big a trade's profit or loss was, measured against
    how much risk was taken on entry (not dollars). +2R means the trade
    made twice what was initially risked; -1R means it lost the full
    initially-risked amount. Comparable across coins/prices this way.
  Win Rate: % of trades that were profitable. Trend systems often WIN
    LESS than half the time but make it up with much bigger winners --
    a low win rate is not automatically bad here.
  Profit Factor: total $ won / total $ lost. Above 1.0 = profitable
    overall; below 1.0 = losing overall, regardless of win rate.
  Expectancy (R): the average result of one trade, in R. This is the
    single best "is this worth trading" number -- positive and bigger
    is better.
  SQN: how CONSISTENT that edge is (mean R divided by how much R bounces
    around, adjusted for trade count). A strategy can have good
    expectancy but a low SQN if results are erratic.
  Recency-Weighted Avg-R/trade: take typical trade quality in each
    calendar year, then average across years -- but recent years count
    MORE than old ones. A strategy that made all its money early and has
    been quiet since scores worse here than one still working now, even
    if their lifetime totals are similar.
  Mean vs Median trade R (raw): if these two are far apart, a handful of
    huge winning trades are doing most of the work, and the strategy may
    be less repeatable than the average number suggests. Check the "Top
    Trades" list when you see a big gap.
  Concentration penalty: a soft score reduction if one single year or one
    single coin produced most of the total profit -- a flag, not an
    automatic rejection.
  Pyramiding: adding to a position that's already open when the same
    entry signal fires again, instead of only ever taking one entry per
    coin. More layers = more conviction added to a working trade.
------------------------------------------------------------------------"""



def print_performance_report(score, metrics, p, label="", stock_names=None, master_dates=None):
    print("\n" + "=" * 78)
    print(f"{label}")
    print("=" * 78)
    print(f"Score: {score:.4f}")
    print(f"Entry: {ENTRY_TYPE_NAMES.get(p['entry_type'],'?')}   "
          f"Exit: {EXIT_TYPE_NAMES.get(p['exit_type'],'?')}   "
          f"Max pyramid layers: {p.get('max_pyramid_layers', 1)}")
    print(f"BTC entry gate: {p['use_btc_entry_gate']}   "
          f"BTC exit override: {p['use_btc_exit_override']}   "
          f"RSI trend filter: {p['use_rsi_trend_filter']}")

    if p['entry_type'] == ENTRY_MA_BREAKOUT:
        print(f"  Entry MA: {p['entry_ma_len']} / {MA_TYPE_NAMES.get(p['entry_ma_type'],'?')}")
    elif p['entry_type'] == ENTRY_RSI_XOVER:
        print(f"  RSI fast: {p['rsi_f_len']}/{p['rsi_f_smt']}  RSI slow: {p['rsi_s_len']}/{p['rsi_s_smt']}")
        if p['use_rsi_trend_filter']:
            print(f"  Trend MA: {p['rsi_trend_ma_len']} / {MA_TYPE_NAMES.get(p['rsi_trend_ma_type'],'?')}")
    elif p['entry_type'] == ENTRY_MA_XOVER:
        print(f"  Short MA: {p['xover_short_len']} / {MA_TYPE_NAMES.get(p['xover_short_type'],'?')}   "
              f"Long MA: {p['xover_long_len']} / {MA_TYPE_NAMES.get(p['xover_long_type'],'?')}")

    if p['use_btc_entry_gate'] or p['use_btc_exit_override']:
        print(f"  BTC filter MA: {p['btc_ma_len']} / {MA_TYPE_NAMES.get(p['btc_ma_type'],'?')}")

    if p['exit_type'] == EXIT_MA_CROSSUNDER:
        print(f"  Exit MA: {p['exit_ma_len']} / {MA_TYPE_NAMES.get(p['exit_ma_type'],'?')}")
    elif p['exit_type'] == EXIT_RSI_CROSSUNDER:
        print(f"  Exit RSI fast: {p['exit_rsi_f_len']}/{p['exit_rsi_f_smt']}  "
              f"Exit RSI slow: {p['exit_rsi_s_len']}/{p['exit_rsi_s_smt']}")
    elif p['exit_type'] == EXIT_MA_XOVER_EXIT:
        print(f"  Exit short MA: {p['exit_xover_short_len']} / {MA_TYPE_NAMES.get(p['exit_xover_short_type'],'?')}   "
              f"Exit long MA: {p['exit_xover_long_len']} / {MA_TYPE_NAMES.get(p['exit_xover_long_type'],'?')}")

    if p['exit_type'] == EXIT_HYBRID:
        print(f"  SL mult: {p['sl_mult']:.2f}  TP mult: {p['tp_mult']:.2f}  Trail mult: {p['trail_mult']:.2f}")
    elif p['exit_type'] == EXIT_PCT_TRAIL:
        print(f"  Trail %: {p['trail_pct']:.2f}")
    elif p['exit_type'] == EXIT_ATR_TRAIL:
        print(f"  Exit ATR mult: {p['exit_atr_mult']:.2f}")
    print(f"  ADX threshold: {p.get('adx_thresh', 0.0)}   (Nominal risk unit sl_mult: {p['sl_mult']:.2f})")

    print(f"\nTrades: {metrics.get('trades',0)}  |  Win Rate: {metrics.get('win_rate',0)*100:.1f}% "
          f"(% of trades that made money)  |  Profit Factor: {metrics.get('profit_factor',0):.2f} "
          f"($ won per $ lost -- above 1.0 is profitable)")
    print(f"Typical trade price move: {metrics.get('mean_pct',0)*100:+.1f}% average, "
          f"{metrics.get('median_pct',0)*100:+.1f}% for the middle trade")
    print(f"Expectancy: {metrics.get('expectancy_r',0):.3f}R (avg profit per trade, in units of "
          f"risk taken)  |  SQN: {metrics.get('sqn_capped',0):.2f} (consistency of that edge, "
          f"Van Tharp scale: <1.6 poor, 1.6-2.5 avg, 2.5-4 good, >4 excellent)")
    print(f"Recency-Weighted Avg-R/trade: {metrics.get('recency_weighted_avg_r',0):.2f} "
          f"(typical trade quality, tilted toward recent years -- the core consistency check; "
          f"plain median w/o recency tilt: {metrics.get('median_yearly_avg_r',0):.2f})")
    print(f"Avg bars held: {metrics.get('avg_bars_held',0):.1f} days  |  Distinct coins: {metrics.get('distinct_coins',0)}")
    if metrics.get('avg_pyramid_layers', 1.0) > 1.01:
        print(f"Pyramiding: avg {metrics.get('avg_pyramid_layers',1.0):.2f} layers/trade, "
              f"{metrics.get('pct_trades_pyramided',0)*100:.0f}% of trades added at least one layer")
    print(f"Max year profit-share: {metrics.get('max_year_share',0)*100:.1f}%  |  "
          f"Max coin profit-share: {metrics.get('max_coin_share',0)*100:.1f}%  |  "
          f"Concentration penalty applied: {metrics.get('concentration_penalty',1.0) < 1.0}")
    if metrics.get('avg_bars_held', 0) > 180 or metrics.get('max_entry_year_concentration', 0) > 0.40:
        print(f"** Long avg. hold + entries clustered in one window -- check "
              f"'REGIME-CONCENTRATION CHECK' in the yearly table before trusting this. **")
    print_top_trades(metrics, stock_names=stock_names, master_dates=master_dates, n=5)
    print("=" * 78)


# ==========================================
# 11. OPTUNA SEARCH SPACE + MAIN OPTIMIZER
# ==========================================

def _suggest_params(trial):
    entry_type = trial.suggest_categorical('entry_type', [ENTRY_MA_BREAKOUT, ENTRY_RSI_XOVER, ENTRY_MA_XOVER])
    exit_type  = trial.suggest_categorical('exit_type', [EXIT_HYBRID, EXIT_PCT_TRAIL, EXIT_ATR_TRAIL,
                                                           EXIT_MA_CROSSUNDER, EXIT_RSI_CROSSUNDER, EXIT_MA_XOVER_EXIT])

    p = {
        'entry_type': entry_type,
        'exit_type':  exit_type,
        'use_btc_entry_gate':    trial.suggest_categorical('use_btc_entry_gate', [False, True]),
        'use_btc_exit_override': trial.suggest_categorical('use_btc_exit_override', [False, True]),
        'use_rsi_trend_filter':  trial.suggest_categorical('use_rsi_trend_filter', [False, True]),
        'adx_thresh': trial.suggest_categorical('adx_thresh', [0.0, 15.0, 20.0, 25.0]),
        'sl_mult':    trial.suggest_float('sl_mult', 1.5, 8.0),
        'tp_mult':    trial.suggest_float('tp_mult', 5.0, 70.0),
        'trail_mult': trial.suggest_float('trail_mult', 2.0, 15.0),
        'trail_pct':  trial.suggest_float('trail_pct', 5.0, 35.0),
        'exit_atr_mult': trial.suggest_float('exit_atr_mult', 1.0, 6.0),
        'max_pyramid_layers': trial.suggest_int('max_pyramid_layers', MAX_PYRAMID_LAYERS_MIN, MAX_PYRAMID_LAYERS_MAX),
    }

    if entry_type == ENTRY_MA_BREAKOUT:
        p['entry_ma_len']  = trial.suggest_int('entry_ma_len', MA_LEN_MIN, MA_LEN_MAX)
        p['entry_ma_type'] = trial.suggest_int('entry_ma_type', 0, 4)
    elif entry_type == ENTRY_RSI_XOVER:
        p['rsi_f_len'] = trial.suggest_int('rsi_f_len', 10, 100)
        p['rsi_f_smt'] = trial.suggest_int('rsi_f_smt', 5, 50)
        p['rsi_s_len'] = trial.suggest_int('rsi_s_len', 10, 100)
        p['rsi_s_smt'] = trial.suggest_int('rsi_s_smt', 5, 50)
        if p['use_rsi_trend_filter']:
            p['rsi_trend_ma_len']  = trial.suggest_int('rsi_trend_ma_len', MA_LEN_MIN, MA_LEN_MAX)
            p['rsi_trend_ma_type'] = trial.suggest_int('rsi_trend_ma_type', 0, 4)
    elif entry_type == ENTRY_MA_XOVER:
        short_len = trial.suggest_int('xover_short_len', MA_LEN_MIN, 200)
        gap       = trial.suggest_int('xover_gap', 10, 150)
        p['xover_short_len']  = short_len
        p['xover_short_type'] = trial.suggest_int('xover_short_type', 0, 4)
        p['xover_long_len']   = min(MA_LEN_MAX, short_len + gap)
        p['xover_long_type']  = trial.suggest_int('xover_long_type', 0, 4)

    if p['use_btc_entry_gate'] or p['use_btc_exit_override']:
        p['btc_ma_len']  = trial.suggest_int('btc_ma_len', MA_LEN_MIN, MA_LEN_MAX)
        p['btc_ma_type'] = trial.suggest_int('btc_ma_type', 0, 4)
    else:
        p['btc_ma_len'], p['btc_ma_type'] = 62, 0   # unused placeholders

    if exit_type == EXIT_MA_CROSSUNDER:
        p['exit_ma_len']  = trial.suggest_int('exit_ma_len', MA_LEN_MIN, MA_LEN_MAX)
        p['exit_ma_type'] = trial.suggest_int('exit_ma_type', 0, 4)
    elif exit_type == EXIT_RSI_CROSSUNDER:
        p['exit_rsi_f_len'] = trial.suggest_int('exit_rsi_f_len', 10, 100)
        p['exit_rsi_f_smt'] = trial.suggest_int('exit_rsi_f_smt', 5, 50)
        p['exit_rsi_s_len'] = trial.suggest_int('exit_rsi_s_len', 10, 100)
        p['exit_rsi_s_smt'] = trial.suggest_int('exit_rsi_s_smt', 5, 50)
    elif exit_type == EXIT_MA_XOVER_EXIT:
        e_short = trial.suggest_int('exit_xover_short_len', MA_LEN_MIN, 200)
        e_gap   = trial.suggest_int('exit_xover_gap', 10, 150)
        p['exit_xover_short_len']  = e_short
        p['exit_xover_short_type'] = trial.suggest_int('exit_xover_short_type', 0, 4)
        p['exit_xover_long_len']   = min(MA_LEN_MAX, e_short + e_gap)
        p['exit_xover_long_type']  = trial.suggest_int('exit_xover_long_type', 0, 4)

    return p

def is_too_close_to_benchmark(p, benchmark_p, tolerance=0.15):
    """
    Returns True if 'p' is on the same gradient/neighborhood as 'benchmark_p'.
    """
    if benchmark_p is None:
        return False

    # If the core signal family is different, it is a completely new setup
    if p['entry_type'] != benchmark_p['entry_type'] or p['exit_type'] != benchmark_p['exit_type']:
        return False

    # If in the same family, check if key lookback lengths are within +/- 15%
    same_family_keys = []
    if p['entry_type'] == ENTRY_MA_BREAKOUT:
        same_family_keys += ['entry_ma_len']
    elif p['entry_type'] == ENTRY_RSI_XOVER:
        same_family_keys += ['rsi_f_len', 'rsi_s_len']
    elif p['entry_type'] == ENTRY_MA_XOVER:
        same_family_keys += ['xover_short_len', 'xover_long_len']

    if p['exit_type'] == EXIT_MA_CROSSUNDER:
        same_family_keys += ['exit_ma_len']

    close_count = 0
    for k in same_family_keys:
        if k in p and k in benchmark_p:
            val_new = p[k]
            val_bench = benchmark_p[k]
            if abs(val_new - val_bench) / max(val_bench, 1) < tolerance:
                close_count += 1

    # If all primary lookbacks overlap closely, reject as 'not novel'
    return close_count == len(same_family_keys) and len(same_family_keys) > 0

def run_optimization():
    tickers = fetch_top100_universe()
    (opens, closes, atr, adx, years_arr,
     stock_names, eligible_mask, master_dates) = prepare_matrix_data(tickers)

    clear_caches()
    n_stocks = closes.shape[1]
    n_days = closes.shape[0]
    is_end = int(n_days * WFO_IS_PCT)
    inner_split = is_end - int((is_end - 0) * INNER_VAL_PCT)   # inner-train: [0, inner_split); inner-val: [inner_split, is_end)

    print(f"\nMatrix: {n_days} days x {n_stocks} coins")
    print(f"Inner-train (search):      days 0-{inner_split}")
    print(f"Inner-validation (search):  days {inner_split}-{is_end}  <- every trial must hold up here too, not just train")
    print(f"True OOS (final check only): days {is_end}-{n_days}  <- never touched during the whole search")
    print(LEGEND)
    print("\n** ROBUSTNESS-AWARE SEARCH (v1.3) **")
    print("Every trial is scored on the WORSE of inner-train vs inner-validation, not just")
    print("inner-train alone -- a parameter set can't win by being lucky in one period only.")
    print("If a trial passes inner-train but fails inner-validation's gates, it still gets a")
    print("heavily discounted fallback score (so the search always has SOME signal to follow)")
    print("instead of a flat rejection -- but it can never outrank a trial that genuinely")
    print("passes both.\n")

    best_is_score = -999999.0
    best_params = None
    all_trial_srs = []

    # -------------------------------------------------------------
    # 1. AUTO-LOAD BENCHMARK (Last Saved Winner or Baseline File)
    # -------------------------------------------------------------
    best_is_score = -999999.0
    best_params = None
    prev_winner_params = None

    # Try loading from the best params file first, then fallback to baseline config
    _, prev_is, loaded_params = load_previous_winner(BEST_PARAMS_FILE)
    if loaded_params is None:
        loaded_params = load_baseline_config(BASELINE_CONFIG_FILE)

    if loaded_params is not None:
        # Score the benchmark on the exact search period (inner_split)
        base_train_score, base_train_m = evaluate_params_signal(
            loaded_params, opens, closes, atr, adx, years_arr, n_stocks,
            start_day=0, end_day=inner_split, is_oos=False
        )
        base_val_score, base_val_m = evaluate_params_signal(
            loaded_params, opens, closes, atr, adx, years_arr, n_stocks,
            start_day=inner_split, end_day=is_end, is_oos=True,
            min_years_required=MIN_YEARS_GATE_INNER
        )
        
        # Dual-pass score logic matching the optimizer
        if base_train_score > -900 and base_val_score > -900:
            benchmark_score = min(base_train_score, base_val_score)
        else:
            benchmark_score = base_train_score if base_train_score > -900 else -999.0

        if benchmark_score > -900:
            best_is_score = benchmark_score
            best_params = loaded_params
            prev_winner_params = loaded_params
            print("\n" + "=" * 70)
            print(f"BENCHMARK LOADED: Score to beat = {best_is_score:.4f}")
            print(f"Entry: {ENTRY_TYPE_NAMES.get(loaded_params['entry_type'])} | "
                  f"Exit: {EXIT_TYPE_NAMES.get(loaded_params['exit_type'])}")
            print("Optuna must EXCEED this score on a DIFFERENT parameter gradient to save.")
            print("=" * 70 + "\n")

    if OPTUNA_AVAILABLE:
        print(f"\nBayesian Optimization ({BAYESIAN_TRIALS} trials, robustness-aware)...")

        def optuna_objective(trial):
            p = _suggest_params(trial)
            
            # FORCED NOVELTY: Reject trials that are merely fine-tuning the old winner
            if prev_winner_params is not None and is_too_close_to_benchmark(p, prev_winner_params):
                return -999.0

            train_score, train_m = evaluate_params_signal(
                p, opens, closes, atr, adx, years_arr, n_stocks,
                start_day=0, end_day=inner_split, is_oos=False
            )
            if train_score <= -900:
                return -999.0

            val_score, val_m = evaluate_params_signal(
                p, opens, closes, atr, adx, years_arr, n_stocks,
                start_day=inner_split, end_day=is_end, is_oos=True,
                min_years_required=MIN_YEARS_GATE_INNER
            )
            if val_score <= -900:
                return min(train_score, 20.0) * ROBUST_FALLBACK_SCALE - ROBUST_FALLBACK_PENALTY

            return min(train_score, val_score)

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(
                seed=int(time.time()), # Vary seed so every run explores differently
                n_startup_trials=500,  # Generous exploration phase
                multivariate=True, 
                group=True
            )
        )
        # NOTE: Do NOT call study.enqueue_trial(benchmark_params) here.
        # Leaving it out prevents Optuna from biasing the Bayesian density toward the old winner.

        def optuna_callback(study, trial):
            nonlocal best_is_score, best_params
            if trial.value is None or trial.value <= -900:
                return
            if trial.value > best_is_score:
                p_full = _suggest_params_from_trial(trial)
                if passes_neighborhood_check(p_full, max(trial.value, 0.01), opens, closes, atr, adx,
                                             years_arr, n_stocks, 0, inner_split):
                    best_is_score = trial.value
                    best_params = p_full
                    train_chk, train_m = evaluate_params_signal(p_full, opens, closes, atr, adx,
                                                                 years_arr, n_stocks, 0, inner_split, is_oos=False)
                    val_chk, val_m = evaluate_params_signal(p_full, opens, closes, atr, adx,
                                                             years_arr, n_stocks, inner_split, is_end, is_oos=True,
                                                             min_years_required=MIN_YEARS_GATE_INNER)
                    dual_pass = val_chk > -900
                    tag = "DUAL-PASS (train + inner-val)" if dual_pass else "FALLBACK (train only -- inner-val failed gates)"
                    print(f"\n[trial {trial.number}] inner-train score: {train_chk:.4f}   "
                          f"inner-val score: {val_chk if dual_pass else 'GATE FAIL'}   [{tag}]")
                    print_performance_report(train_chk, train_m, p_full, f"NEW CHAMPION (trial {trial.number})",
                                             stock_names=stock_names, master_dates=master_dates)

        study.optimize(optuna_objective, n_trials=BAYESIAN_TRIALS,
                       callbacks=[optuna_callback], show_progress_bar=True)

    if best_params is not None:
        is_score, oos_score, is_m, oos_m, robustness = run_wfo_validation(
            best_params, opens, closes, atr, adx, years_arr, n_stocks)
        report_yearly_table(is_m, label="IN-SAMPLE YEAR-BY-YEAR R BREAKDOWN")
        if oos_score > -900:
            run_overfitting_diagnostic(all_trial_srs, oos_m)
            report_yearly_table(oos_m, label="OUT-OF-SAMPLE YEAR-BY-YEAR R BREAKDOWN")

        # -- Naive portfolio-risk diagnostics (advisory, see
        #    compute_naive_risk_metrics' docstring). report_m's naive_max_dd
        #    was already computed at NAIVE_RISK_PCT_PER_TRADE (1%) inside
        #    compute_score_signal and factored into its score via
        #    risk_penalty above -- this just also computes the 2%-risk
        #    variant for context and prints everything clearly. Falls back
        #    to IS if OOS failed its gates (oos_m may have too short/empty
        #    a trade log in that case). --
        report_m = oos_m if oos_score > -900 else is_m
        naive_dd_1pct = report_m.get('naive_max_dd', 0.0)
        naive_dd_2pct = compute_naive_risk_metrics(
            report_m['entry_days'], report_m['exit_days'], report_m['r_multiple_raw'],
            risk_per_trade_pct=0.02)['naive_max_dd']
        max_streak = report_m.get('max_consecutive_losses', 0)
        streak_ratio = report_m.get('streak_ratio', 0.0)
        max_conc_open = report_m.get('max_concurrent_open', 0)
        max_conc_losers = report_m.get('max_concurrent_losers', 0)
        worst_day_frac = report_m.get('worst_day_loser_fraction', 0.0)
        risk_penalty_applied = report_m.get('risk_penalty', 1.0)

        print(f"\n{'='*78}")
        print("NAIVE PORTFOLIO-RISK DIAGNOSTICS (advisory, NOT a real portfolio")
        print("simulation -- run crypto_portfolio_optimizer.py for that; see")
        print("compute_naive_risk_metrics' docstring for exactly what this captures)")
        print(f"{'='*78}")
        print(f"  Naive max drawdown:   ~{naive_dd_1pct:.1%} @1% risk/trade, "
              f"~{naive_dd_2pct:.1%} @2% risk/trade")
        print(f"  Longest losing streak: {max_streak} trades in a row "
              f"({streak_ratio:.1f}x what this win rate would statistically predict)")
        print(f"  Worst correlated moment: {max_conc_losers}/{max_conc_open} concurrently-open "
              f"trades were eventual losers ({worst_day_frac:.0%})")
        if risk_penalty_applied < 1.0:
            print(f"  -> risk_penalty = {risk_penalty_applied:.2f}x was already applied to this "
                  f"trial's score for the reasons above.")
        if naive_dd_1pct > NAIVE_DD_SOFT_THRESHOLD:
            print("  *** Even at a conservative 1% risk per trade, this implies a >50%")
            print("      equity swing -- this signal's losing trades cluster heavily in")
            print("      calendar time (a market-wide event, most likely) even though its")
            print("      per-trade stats above look fine in isolation. The portfolio")
            print("      optimizer's max_dd gate is likely to reject nearly everything")
            print("      built on this signal. Worth confirming before spending an 8000-")
            print("      trial budget on it.")
        print(f"{'='*78}")

        # -- ALWAYS save + report the best candidate found, honestly tiered.
        #    A hard "don't save below 50%" gate meant a bad run left you with
        #    literally nothing. Now you always get the best available result
        #    with its real robustness clearly labeled, so YOU decide whether
        #    it's good enough to trade -- rather than the engine silently
        #    deciding for you by withholding it. --
        if oos_score > -900 and robustness >= 0.70:
            tier = "EXCELLENT (>70%) -- deploy with confidence"
        elif oos_score > -900 and robustness >= ROBUSTNESS_DEPLOY_THRESHOLD:
            tier = f"ACCEPTABLE ({ROBUSTNESS_DEPLOY_THRESHOLD:.0%}-70%) -- deploy cautiously"
        elif oos_score > -900 and robustness >= 0.30:
            tier = "CAUTION (30%-50%) -- meaningful risk this doesn't hold up; consider smaller size"
        elif oos_score > -900:
            tier = "POOR (<30%) -- high risk of not holding up; treat as a starting point, not a system"
        else:
            tier = "OOS GATE FAILURE -- this candidate never even cleared the OOS gates (see diagnosis above); the IS-side numbers below are the only evidence of any edge at all"
        if naive_dd_1pct > NAIVE_DD_SOFT_THRESHOLD:
            tier += f" | NAIVE DD ~{naive_dd_1pct:.0%} even at 1% risk/trade -- see note above"

        save_winner(oos_score, is_score, best_params, tier=tier,
                    naive_dd_1pct=naive_dd_1pct, naive_dd_2pct=naive_dd_2pct,
                    max_consecutive_losses=max_streak, streak_ratio=streak_ratio,
                    worst_day_loser_fraction=worst_day_frac, universe=tickers)
        print(f"\n{'='*78}\nFINAL RESULT -- ROBUSTNESS TIER: {tier}\n{'='*78}")
        print_performance_report(oos_score if oos_score > -900 else is_score,
                                 oos_m if oos_score > -900 else is_m,
                                 best_params,
                                 "BEST CANDIDATE FOUND (see tier above before trading this)",
                                 stock_names=stock_names, master_dates=master_dates)
        print(f"\nSaved to {BEST_PARAMS_FILE} regardless of tier -- open that file to see the tier")
        print("label alongside the params. A CAUTION or POOR tier is real information, not a")
        print("recommendation to trade it at full size (or at all) -- see it as your current")
        print("best-tested starting point while a better one keeps searching, not a finished system.")
    else:
        print("\nNo candidate cleared even the inner-train gates across every trial -- this is a")
        print("stronger signal than a low robustness score: the entry/exit families and parameter")
        print("ranges searched didn't find ANY combination with a positive, gate-passing edge even")
        print("before checking generalization. Worth widening the search (see MA_LEN_MIN/MAX, the")
        print("gate constants in section 0) or reconsidering whether these signal families fit this")
        print("data before adding more trials.")


def _suggest_params_from_trial(trial):
    """Rebuilds the full params dict from a completed trial's stored
    params (trial.params only contains what was actually suggested this
    trial, which is already conditionally correct thanks to _suggest_params'
    branching -- this just re-derives xover_long_len the same way)."""
    tp = dict(trial.params)
    p = {
        'entry_type': tp['entry_type'], 'exit_type': tp['exit_type'],
        'use_btc_entry_gate': tp['use_btc_entry_gate'],
        'use_btc_exit_override': tp['use_btc_exit_override'],
        'use_rsi_trend_filter': tp['use_rsi_trend_filter'],
        'adx_thresh': tp['adx_thresh'], 'sl_mult': tp['sl_mult'], 'tp_mult': tp['tp_mult'],
        'trail_mult': tp['trail_mult'], 'trail_pct': tp['trail_pct'], 'exit_atr_mult': tp['exit_atr_mult'],
        'max_pyramid_layers': tp['max_pyramid_layers'],
    }
    if tp['entry_type'] == ENTRY_MA_BREAKOUT:
        p['entry_ma_len'], p['entry_ma_type'] = tp['entry_ma_len'], tp['entry_ma_type']
    elif tp['entry_type'] == ENTRY_RSI_XOVER:
        p['rsi_f_len'], p['rsi_f_smt'] = tp['rsi_f_len'], tp['rsi_f_smt']
        p['rsi_s_len'], p['rsi_s_smt'] = tp['rsi_s_len'], tp['rsi_s_smt']
        if tp['use_rsi_trend_filter']:
            p['rsi_trend_ma_len'], p['rsi_trend_ma_type'] = tp['rsi_trend_ma_len'], tp['rsi_trend_ma_type']
    elif tp['entry_type'] == ENTRY_MA_XOVER:
        p['xover_short_len']  = tp['xover_short_len']
        p['xover_short_type'] = tp['xover_short_type']
        p['xover_long_len']   = min(MA_LEN_MAX, tp['xover_short_len'] + tp['xover_gap'])
        p['xover_long_type']  = tp['xover_long_type']

    if tp['use_btc_entry_gate'] or tp['use_btc_exit_override']:
        p['btc_ma_len'], p['btc_ma_type'] = tp['btc_ma_len'], tp['btc_ma_type']
    else:
        p['btc_ma_len'], p['btc_ma_type'] = 62, 0

    if tp['exit_type'] == EXIT_MA_CROSSUNDER:
        p['exit_ma_len'], p['exit_ma_type'] = tp['exit_ma_len'], tp['exit_ma_type']
    elif tp['exit_type'] == EXIT_RSI_CROSSUNDER:
        p['exit_rsi_f_len'], p['exit_rsi_f_smt'] = tp['exit_rsi_f_len'], tp['exit_rsi_f_smt']
        p['exit_rsi_s_len'], p['exit_rsi_s_smt'] = tp['exit_rsi_s_len'], tp['exit_rsi_s_smt']
    elif tp['exit_type'] == EXIT_MA_XOVER_EXIT:
        p['exit_xover_short_len']  = tp['exit_xover_short_len']
        p['exit_xover_short_type'] = tp['exit_xover_short_type']
        p['exit_xover_long_len']   = min(MA_LEN_MAX, tp['exit_xover_short_len'] + tp['exit_xover_gap'])
        p['exit_xover_long_type']  = tp['exit_xover_long_type']

    return p


if __name__ == "__main__":
    run_optimization()