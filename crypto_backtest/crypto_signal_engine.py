"""
CRYPTO ALPHA SIGNAL ENGINE
===========================
SCRIPT 1 of 4 in the consolidated pipeline. Sibling: crypto_portfolio_engine.py
(NSE has its own pair: nse_signal_engine.py / nse_portfolio_engine.py).

THE ONE QUESTION THIS FILE ANSWERS:
  "Is this entry/exit signal +EV, and consistently so across coins and
   years, independent of capital-timing luck?"

THE QUESTION IT DELIBERATELY DOES NOT ANSWER (see crypto_portfolio_engine.py):
  "How would my actual wealth have compounded running this with a real,
   shared, monthly-funded cash pool, position queues, and drawdown limits?"

Every trade here is scored independently in scale-free R-multiples. There
is no shared cash pool, so one coin can never "win the capital-allocation
race" and make a different coin's signal look better or worse than it
actually is. That separation is deliberate and load-bearing -- see the
project's design notes for why conflating these two questions was the
single biggest problem across the earlier iterations this consolidates.

OUTPUT: this script does not declare a winner or deploy money. It writes
a ranked SHORTLIST (crypto_signal_shortlist.json) of every distinct
config that cleared every signal-quality gate, each with full metrics
attached, for crypto_portfolio_engine.py to load and pressure-test under
real capital constraints. Nothing here decides what "wins" -- that is
Script 2's job, kept in a separate file on purpose so a genuinely good
signal can never be silently discarded just because Script 2's money-
management logic wasn't shaped to use it that day.

CONSOLIDATED FROM 5 PRIOR ITERATIONS -- what came from where:
  - R-multiple scoring, entry x exit combinatorics (3 entry families x 6
    exit families, all independently togglable filters), robustness-aware
    inner train/val Optuna search (every trial scored on the WORSE of an
    inner-train slice and an inner-validation slice, so nothing is ever
    optimized purely in-sample), SQN/expectancy/recency-weighted yearly R,
    winsorization, DSR diagnostic, baseline-config loader: from the most
    mature signal-research draft.
  - Latched/armed entry confluence -- a crossover ARMS a pending signal
    that waits for the trend/BTC filters to align (instead of requiring
    same-bar confluence, which throws away a lot of real setups where the
    filter just lags the crossover by a day or two) -- from an earlier
    iteration, generalized here into one boolean (`use_latched_entry`)
    that applies uniformly across all three entry families instead of a
    separate hand-maintained fork of the whole file.
  - Two-layer liquidity filtering (coarse 24h-volume screen at universe-
    fetch time + a point-in-time trailing-volume rolling mask that blocks
    NEW entries only) and the bounded thread-safe LRU indicator caches:
    pulled forward from the money-management draft, because a "trade" on
    a coin nobody could actually fill isn't a real signal either --
    liquidity is a data-eligibility question both scripts need answered
    the same way, not just a portfolio-sizing concern.
  - Cross-sectional per-coin median gate: two earlier iterations
    independently converged on "don't let 1-2 runaway coins carry the
    whole score." Kept here as an explicit hard gate (median across coins
    of that coin's own mean R must be positive), not just a diagnostic --
    a strategy that's only +EV because of SHIB/DOGE-style outliers should
    not clear this file.
  - Anchored-expanding-walk-forward FINAL STABILITY REPLAY (not full per-
    fold re-optimization -- see section 10 for exactly why that tradeoff
    was made): from the walk-forward-only draft, run once per shortlisted
    candidate as a cheap, honest "did this keep working every year, or did
    it live on one lucky stretch" check that a single 70/30 split can hide.

CHANGELOG
  v1.0  Initial consolidated build.
"""

import math
import json
import os
import time
import threading
import hashlib
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

# ==========================================================================
# 0. CONFIGURATION -- every manually-adjustable constant lives here.
# ==========================================================================

BASELINE_CONFIG_FILE = "baseline_signal_config.json"   # optional: paste a config here to seed the search
SHORTLIST_FILE        = "crypto_signal_shortlist.json"  # OUTPUT -- crypto_portfolio_engine.py reads this
ENGINE_VERSION         = "crypto-signal-1.0.0"

# -- Universe / data window --
START_DATE       = "2018-01-01"
MIN_HISTORY_DAYS = 400
TOP_N_COINS      = 60     # signal screening doesn't need the full 200-coin
                           # universe the portfolio engine might use -- a
                           # smaller, cleaner set keeps the search honest

# -- Search budget --
BAYESIAN_TRIALS = 20_000
WFO_IS_PCT  = 0.70
WFO_OOS_PCT = 0.30

# -- Quality gates -- a trial failing ANY of these scores -999 --
MIN_TRADES_GATE      = 50
MIN_YEARS_GATE        = 3
MIN_WIN_RATE_GATE     = 0.35   # trend systems are SUPPOSED to have a sub-50% win rate
MIN_PROFIT_FACTOR     = 1.10
MEDIAN_COIN_R_GATE    = 0.0    # median-across-coins mean-R must clear this (see docstring)

# -- Robustness-aware search: the IS window is split into an inner TRAIN
# slice and an inner VALIDATION slice, and every trial is scored on the
# WORSE of the two. True OOS (WFO_OOS_PCT) is never touched during search. --
INNER_VAL_PCT         = 0.30
MIN_YEARS_GATE_INNER   = 2
MIN_TRADES_INNER_VAL   = 20
ROBUST_FALLBACK_SCALE   = 0.05
ROBUST_FALLBACK_PENALTY = 5.0

# -- MA parameter search ranges --
MA_LEN_MIN, MA_LEN_MAX = 20, 300
MA_CACHE_MAX_ENTRIES   = 40_000

# -- Pyramiding --
MAX_PYRAMID_LAYERS_MIN = 1
MAX_PYRAMID_LAYERS_MAX = 4

# -- Recency weighting (oldest year in the scored window -> RECENCY_WEIGHT_MIN,
# most recent year -> RECENCY_WEIGHT_MAX) --
RECENCY_WEIGHT_MIN = 0.70
RECENCY_WEIGHT_MAX = 1.00

# -- R-multiple winsorization: caps how much one freak trade can dominate
# the AGGREGATE score. Raw value is still stored/reported regardless. --
R_WINSORIZE_CAP = 20.0

# -- Concentration -- SOFT penalty only, never a hard reject here. A fat
# right tail is normal/expected for a trend system; the PORTFOLIO engine
# is where a hard concentration gate belongs (deciding whether real money
# should go behind it), not this signal-quality screen. --
CONCENTRATION_SOFT_THRESHOLD = 0.80
CONCENTRATION_PENALTY_MULT   = 0.70

# -- Composite score weights -- must sum to 1.00 --
W_SQN        = 0.30
W_EXPECTANCY = 0.30
W_RECENCY    = 0.30
W_WR_BONUS   = 0.10
SQN_CAP = 6.0

# -- Neighborhood-stability perturbation check --
NEIGHBOR_THRESHOLD = 0.80

# -- Liquidity (two-layer -- see module docstring) --
MIN_AVG_DAILY_VOLUME_USD = 3_000_000
LIQUIDITY_LOOKBACK_DAYS  = 30

# -- Walk-forward FINAL stability replay (section 10) --
WF_INITIAL_TRAIN_YEARS = 3
WF_MIN_FOLD_TRADES     = 8     # deliberately low -- this is a stability
                                # replay of a FROZEN config on one calendar
                                # year, not a fresh statistical gate

# -- Shortlist output --
SHORTLIST_SIZE       = 15   # top-N distinct configs written for Script 2
SHORTLIST_POOL_CAP   = 60   # candidates kept in memory during search before final trim

# -- entry/exit signal type IDs --
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
    if max_year <= min_year:
        return RECENCY_WEIGHT_MAX
    frac = (year - min_year) / (max_year - min_year)
    return RECENCY_WEIGHT_MIN + (RECENCY_WEIGHT_MAX - RECENCY_WEIGHT_MIN) * frac


# ==========================================================================
# 1. UNIVERSE DEFINITION
# ==========================================================================

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


def fetch_top_universe(top_n=TOP_N_COINS):
    """Today's top-N-by-market-cap coins from CoinGecko, filtered and
    mapped to Binance USDT pairs. BTCUSDT is hard-guaranteed at index 0 --
    the BTC-regime filter and benchmark math downstream assume this.
    SURVIVORSHIP-BIAS DISCLOSURE: this is TODAY's top-N applied back to
    START_DATE. A coin that mattered in 2018-2020 but isn't top-N today is
    invisible to this backtest. Disclosed, not fixed (crypto has no public
    point-in-time constituent archive the way NSE does)."""
    print(f"Fetching top-{top_n}-by-market-cap universe from CoinGecko...")
    binance_usdt = _get_binance_usdt_symbols()

    tickers = []
    excluded_log = []
    try:
        r = requests.get(COINGECKO_MARKETS_URL, params={
            'vs_currency': 'usd', 'order': 'market_cap_desc',
            'per_page': top_n, 'page': 1, 'sparkline': 'false'
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
            vol_24h = c.get('total_volume', 0) or 0
            if sym != 'BTC' and vol_24h < MIN_AVG_DAILY_VOLUME_USD:
                excluded_log.append(f"{sym} (24h volume ${vol_24h:,.0f} below floor)")
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

    if len(tickers) < 15:
        print("Universe too small after filtering; falling back to static major-coin list.")
        tickers = list(FALLBACK_UNIVERSE)

    print(f"Universe size after filtering: {len(tickers)} (excluded {len(excluded_log)})")
    assert tickers[0] == 'BTCUSDT', "BTCUSDT must be at index 0."
    return tickers


# ==========================================================================
# 2. INDICATORS -- 5 MA types + Wilder RSI + Wilder ATR/ADX, all NaN-safe
# (scan past a leading NaN run so a newly-listed coin's pre-listing gap
# doesn't poison the indicator's state).
# ==========================================================================

@njit(nogil=True, cache=True)
def calc_ma(prices, period, ma_type):
    """ma_type: 0=SMA 1=EMA 2=DEMA 3=WMA 4=SMMA/RMA (Wilder)"""
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
    elif ma_type == 4:  # SMMA / RMA
        rma = np.mean(prices[start:start + period])
        res[start + period - 1] = rma
        for i in range(start + period, n):
            rma = (rma * (period - 1) + prices[i]) / period
            res[i] = rma
    return res


@njit(nogil=True, cache=True)
def calc_rsi_wilder(prices, period):
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


# ==========================================================================
# 2b. CACHES -- bounded, thread-safe (Optuna n_jobs>1 safe)
# ==========================================================================

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
    return calc_ma(raw, smooth_len, 0)


def clear_caches():
    with _MA_CACHE_LOCK:
        _MA_CACHE.clear()
    with _RSI_CACHE_LOCK:
        _RSI_CACHE.clear()


_ZERO_CACHE = {}


def _zeros_like(n_days):
    if n_days not in _ZERO_CACHE:
        _ZERO_CACHE[n_days] = np.zeros(n_days)
    return _ZERO_CACHE[n_days]


# ==========================================================================
# 3. PER-COIN SIGNAL SIMULATOR
# ==========================================================================
# One coin at a time, fully independent of every other coin -- no shared
# capital, so no reason to simulate the whole universe in lockstep. Close-
# based signals, fill at next bar's open. Outputs a trade log, not a
# portfolio value curve.

@njit(nogil=True, cache=True)
def simulate_signal_trades(
        opens, closes, atr, adx,
        entry_ma, xover_short, xover_long,
        rsi_fast, rsi_slow, rsi_trend_ma,
        btc_close, btc_ma,
        exit_ma, exit_xover_short, exit_xover_long,
        exit_rsi_fast, exit_rsi_slow,
        eligible,
        entry_type, exit_type,
        use_btc_entry_gate, use_btc_exit_override, use_rsi_trend_filter,
        use_latched_entry,
        adx_threshold,
        sl_mult, tp_mult, trail_mult, trail_pct, exit_atr_mult,
        max_pyramid_layers,
        start_day, end_day):
    """
    LATCHED ENTRY (use_latched_entry): instead of requiring the entry
    family's raw trigger AND every filter to align on the exact same bar,
    a trigger ARMS a pending signal that keeps waiting, bar after bar,
    until either (a) the filters catch up and it fires, or (b) the entry
    family's own reversal condition disarms it. This applies uniformly to
    all three entry families via one boolean instead of a separate,
    hand-maintained fork of the whole file -- generalizes an approach an
    earlier iteration only implemented for the RSI-crossover entry.

    PYRAMIDING: if max_pyramid_layers > 1, the SAME entry signal firing
    again while already in a position adds another layer (up to the cap)
    instead of being ignored. Cost basis becomes the equally-weighted
    average of every layer's fill price; the STOP/TP levels and the
    R-multiple's risk denominator stay anchored to the FIRST layer's
    entry, never re-anchored on an add.

    ELIGIBILITY (`eligible`, per-day bool): blocks NEW commitments of
    capital only (fresh opens AND pyramid adds) -- e.g. a liquidity-
    trailing-volume mask. An existing position is never force-closed by
    this; only what would deepen exposure is gated.
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
    entry_risk           = 0.0
    stop_loss_price      = 0.0
    tp_trigger_price     = 0.0
    half_sold            = False
    partial_exit_price   = 0.0
    high_since_entry     = 0.0

    pending_entry   = False
    pending_exit    = False
    pending_partial = False

    is_armed = False   # latched-entry state

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

        # ---- entry evaluation ----
        can_add = ((not in_pos) or (n_layers < max_pyramid_layers)) and eligible[d]
        if can_add and (not pending_entry) and (not pending_exit):

            trigger = False
            invalidator = False
            filter_ok = True

            if entry_type == ENTRY_MA_BREAKOUT:
                if prev_close <= entry_ma[d - 1] and curr_close > entry_ma[d]:
                    trigger = True
                if prev_close >= entry_ma[d - 1] and curr_close < entry_ma[d]:
                    invalidator = True
            elif entry_type == ENTRY_RSI_XOVER:
                if rsi_fast[d - 1] <= rsi_slow[d - 1] and rsi_fast[d] > rsi_slow[d]:
                    trigger = True
                if rsi_fast[d - 1] >= rsi_slow[d - 1] and rsi_fast[d] < rsi_slow[d]:
                    invalidator = True
                if use_rsi_trend_filter:
                    filter_ok = curr_close > rsi_trend_ma[d]
            elif entry_type == ENTRY_MA_XOVER:
                if xover_short[d - 1] <= xover_long[d - 1] and xover_short[d] > xover_long[d]:
                    trigger = True
                if xover_short[d - 1] >= xover_long[d - 1] and xover_short[d] < xover_long[d]:
                    invalidator = True

            btc_ok = (not use_btc_entry_gate) or btc_bullish
            adx_ok = (adx_threshold <= 0.0) or (adx[d] >= adx_threshold)

            entry_signal = False
            if use_latched_entry:
                if trigger:
                    is_armed = True
                if invalidator:
                    is_armed = False
                if is_armed and filter_ok and btc_ok and adx_ok:
                    entry_signal = True
                    is_armed = False
            else:
                entry_signal = trigger and filter_ok and btc_ok and adx_ok

            if entry_signal:
                pending_entry = True

    return (t_entry_day[:t_cnt], t_exit_day[:t_cnt], t_entry_price[:t_cnt],
            t_exit_price[:t_cnt], t_r_multiple[:t_cnt], t_pct_return[:t_cnt],
            t_bars_held[:t_cnt], t_layers[:t_cnt])


# ==========================================================================
# 3b. OVERFITTING DIAGNOSTIC -- DSR (Bailey & Lopez de Prado, 2014)
# ==========================================================================

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


# ==========================================================================
# 4. DATA PREPARATION
# ==========================================================================

def build_liquidity_mask(quote_vol_matrix, lookback_days=LIQUIDITY_LOOKBACK_DAYS,
                          min_avg_usd=MIN_AVG_DAILY_VOLUME_USD):
    """Point-in-time, time-varying: a NEW entry is only eligible on days
    where a coin's TRAILING lookback-day average USDT volume clears the
    floor. A coin thin in 2019 but liquid today isn't always-tradeable;
    a coin that goes quiet after being liquid stops qualifying for new
    entries from that point on. Pre-listing NaNs fall out of the rolling
    window naturally. Existing positions are never force-closed by this."""
    vol_df = pd.DataFrame(quote_vol_matrix)
    rolling_avg = vol_df.rolling(window=lookback_days, min_periods=lookback_days).mean().values
    return rolling_avg >= min_avg_usd


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
    for col in ['Open', 'High', 'Low', 'Close', 'QuoteVol']:
        df[col] = df[col].astype(float)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'QuoteVol']].drop_duplicates('Date').set_index('Date')
    return df


def prepare_matrix_data(tickers, data_dir="data_crypto_signal"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    raw_dfs      = {}
    master_dates = set()

    for ticker in tqdm(tickers, desc="Downloading Binance klines"):
        file_path = f"{data_dir}/{ticker}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            if 'QuoteVol' not in df.columns:
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
            raise RuntimeError("Could not download BTCUSDT history -- everything downstream depends on it.")

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
    quote_vol_matrix = np.full((n_days, n_stocks), np.nan)

    for i, ticker in enumerate(tqdm(stock_names, desc="Building matrix")):
        df = raw_dfs[ticker].reindex(master_dates)
        opens[:,  i] = df['Open'].ffill().values
        highs[:,  i] = df['High'].ffill().values
        lows[:,   i] = df['Low'].ffill().values
        closes[:, i] = df['Close'].ffill().values
        atr_matrix[:, i] = calc_atr_wilder(highs[:, i], lows[:, i], closes[:, i], period=14)
        adx_matrix[:, i] = calc_adx(highs[:, i], lows[:, i], closes[:, i], period=14)
        if 'QuoteVol' in df.columns:
            quote_vol_matrix[:, i] = df['QuoteVol'].values   # NOT ffilled -- a real zero-volume day should count as zero

    assert stock_names[0] == 'BTCUSDT'

    liquidity_mask = build_liquidity_mask(quote_vol_matrix)
    n_eligible_coin_days = int(np.nansum(liquidity_mask))
    print(f"Liquidity filter: {n_eligible_coin_days:,} coin-days eligible for new entries "
          f"(trailing {LIQUIDITY_LOOKBACK_DAYS}-day avg volume >= ${MIN_AVG_DAILY_VOLUME_USD:,.0f}).")

    years_arr = master_df['Year'].values.astype(np.int32)

    return (opens, closes, atr_matrix, adx_matrix, years_arr,
            stock_names, liquidity_mask, master_dates)


# ==========================================================================
# 5. EVALUATE PARAMS
# ==========================================================================

def collect_trade_log(p, opens, closes, atr, adx, years_arr, n_stocks,
                       eligible_mask, start_day=0, end_day=-1):
    """Runs the simulator across every coin and returns the concatenated,
    UNGATED trade log (or None if literally zero trades fired). Deliberately
    separate from evaluate_params_signal/compute_score_signal below: the
    walk-forward replay (section 9) needs to judge a single calendar-year
    fold on its own raw trades without being forced through hard gates
    (min-trades-for-a-FULL-search, win-rate floor, etc.) that were tuned
    for the aggregate multi-year search, not for reading one thin year in
    isolation. Scoring a fold through the full gate suite was silently
    relabeling a fold that had PLENTY of trades but genuinely lost money
    as "inconclusive" -- indistinguishable from a fold that simply didn't
    fire enough signals to say anything. Splitting this out fixes that."""
    n_days = closes.shape[0]
    if end_day < 0: end_day = n_days - 1

    entry_type = p['entry_type']
    exit_type  = p['exit_type']
    use_btc_entry_gate    = p['use_btc_entry_gate']
    use_btc_exit_override = p['use_btc_exit_override']
    use_rsi_trend_filter  = p['use_rsi_trend_filter']
    use_latched_entry     = p.get('use_latched_entry', False)

    zeros = _zeros_like(n_days)

    entry_ma_all = xover_short_all = xover_long_all = None
    rsi_fast_all = rsi_slow_all = rsi_trend_all = None

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

    exit_ma_all = exit_xover_short_all = exit_xover_long_all = None
    exit_rsi_fast_all = exit_rsi_slow_all = None

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
        eligible_col     = eligible_mask[:, s].astype(np.bool_)

        result = simulate_signal_trades(
            opens[:, s], closes[:, s], atr[:, s], adx[:, s],
            entry_ma, xover_short, xover_long,
            rsi_fast, rsi_slow, rsi_trend,
            btc_close_col, btc_ma_all,
            exit_ma, exit_xover_short, exit_xover_long,
            exit_rsi_fast, exit_rsi_slow,
            eligible_col,
            entry_type, exit_type,
            use_btc_entry_gate, use_btc_exit_override, use_rsi_trend_filter,
            use_latched_entry,
            p.get('adx_thresh', 0.0),
            p['sl_mult'], p['tp_mult'], p['trail_mult'], p['trail_pct'], p['exit_atr_mult'],
            max_pyramid_layers,
            int(start_day), int(end_day)
        )
        (e_day, x_day, e_price, x_price, r_mult, pct_ret, bars, layers) = result
        if len(e_day) == 0:
            continue
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
        return None

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

    window_years = years_arr[start_day:end_day]

    return {
        'entry_days': entry_days, 'exit_days': exit_days,
        'entry_prices': entry_prices, 'exit_prices': exit_prices,
        'r_multiple': r_multiple, 'pct_return': pct_return,
        'bars_held': bars_held, 'layers_used': layers_used,
        'stock_idx': stock_idx, 'entry_years': entry_years,
        'global_min_year': int(window_years.min()),
        'global_max_year': int(window_years.max()),
    }


def evaluate_params_signal(p, opens, closes, atr, adx, years_arr, n_stocks,
                            eligible_mask, start_day=0, end_day=-1, is_oos=False,
                            min_years_required=None, min_trades_required=None):
    log = collect_trade_log(p, opens, closes, atr, adx, years_arr, n_stocks,
                             eligible_mask, start_day, end_day)
    if log is None:
        return -999.0, {}

    return compute_score_signal(log['r_multiple'], log['pct_return'], log['bars_held'],
                                 log['entry_years'], log['stock_idx'], log['entry_days'],
                                 log['exit_days'], log['entry_prices'], log['exit_prices'],
                                 log['layers_used'], log['global_min_year'], log['global_max_year'],
                                 is_oos=is_oos, min_years_required=min_years_required,
                                 min_trades_required=min_trades_required)


# ==========================================================================
# 6. SCORING
# ==========================================================================

def compute_score_signal(r_multiple, pct_return, bars_held, entry_years,
                          stock_idx, entry_days, exit_days,
                          entry_prices, exit_prices, layers_used,
                          global_min_year, global_max_year, is_oos=False,
                          min_years_required=None, min_trades_required=None):
    req_years = min_years_required if min_years_required is not None else MIN_YEARS_GATE

    n_trades = len(r_multiple)
    if min_trades_required is not None:
        target_trades = min_trades_required
    else:
        target_trades = max(15, int(MIN_TRADES_GATE * (WFO_OOS_PCT / WFO_IS_PCT))) if is_oos else MIN_TRADES_GATE

    r_capped = np.clip(r_multiple, -R_WINSORIZE_CAP, R_WINSORIZE_CAP)

    metrics = {
        'trades': int(n_trades),
        'r_multiple': r_capped,
        'r_multiple_raw': r_multiple,
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
    if profit_factor < MIN_PROFIT_FACTOR:
        return -999.0, metrics

    avg_win_r  = float(r_capped[wins_mask].mean()) if wins_mask.any() else 0.0
    avg_loss_r = float(-r_capped[~wins_mask].mean()) if (~wins_mask).any() else 0.0
    expectancy_r = win_rate * avg_win_r - (1.0 - win_rate) * avg_loss_r
    if expectancy_r <= 0:
        return -999.0, metrics

    # -- cross-sectional per-coin median gate: don't let 1-2 runaway coins
    # carry a score that the TYPICAL coin never actually earned --
    per_coin_mean_r = {}
    for c in np.unique(stock_idx):
        per_coin_mean_r[int(c)] = float(r_multiple[stock_idx == c].mean())
    median_coin_mean_r = float(np.median(list(per_coin_mean_r.values())))
    if median_coin_mean_r <= MEDIAN_COIN_R_GATE:
        metrics['median_coin_mean_r'] = median_coin_mean_r
        return -999.0, metrics

    mean_r   = float(r_capped.mean())
    median_r = float(np.median(r_capped))
    std_r    = float(r_capped.std())
    sqn = (mean_r / std_r) * math.sqrt(min(n_trades, 100)) if std_r > 0 else 0.0
    sqn_capped = min(sqn, SQN_CAP)

    year_r_avg = {}
    year_r_sum_raw = {}
    for yr in distinct_years:
        yr_mask = entry_years == yr
        year_r_avg[yr] = float(r_capped[yr_mask].mean())
        year_r_sum_raw[yr] = float(r_multiple[yr_mask].sum())
    median_yearly_avg_r = float(np.median(list(year_r_avg.values())))

    weights = [_recency_weight(yr, global_min_year, global_max_year) for yr in distinct_years]
    w_sum = sum(weights)
    recency_weighted_avg_r = (sum(w * year_r_avg[yr] for w, yr in zip(weights, distinct_years)) / w_sum
                               if w_sum > 0 else median_yearly_avg_r)

    total_r_raw = float(r_multiple.sum())
    max_year_share = 0.0
    max_coin_share = 0.0
    if total_r_raw > 0:
        max_year_share = max(v / total_r_raw for v in year_r_sum_raw.values())
        coin_r_sums = {int(c): float(r_multiple[stock_idx == c].sum()) for c in np.unique(stock_idx)}
        max_coin_share = max(v / total_r_raw for v in coin_r_sums.values())

    concentration_penalty = 1.0
    if max_year_share > CONCENTRATION_SOFT_THRESHOLD or max_coin_share > CONCENTRATION_SOFT_THRESHOLD:
        concentration_penalty = CONCENTRATION_PENALTY_MULT

    wr_bonus = max(0.0, win_rate - 0.50) * 4.0
    stat_conf = min(1.0, math.sqrt(n_trades / target_trades))

    score = (
        sqn_capped              * W_SQN +
        expectancy_r            * W_EXPECTANCY +
        recency_weighted_avg_r  * W_RECENCY +
        wr_bonus                * W_WR_BONUS
    ) * stat_conf * concentration_penalty

    entry_year_counts = {yr: int((entry_years == yr).sum()) for yr in distinct_years}
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
        'median_yearly_avg_r': median_yearly_avg_r,
        'recency_weighted_avg_r': recency_weighted_avg_r,
        'median_coin_mean_r': median_coin_mean_r,
        'per_coin_mean_r': per_coin_mean_r,
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
    })
    return score, metrics


def diagnose_gates_signal(r_multiple, pct_return, bars_held, entry_years,
                           stock_idx, is_oos=False, min_trades_required=None):
    n_trades = len(r_multiple)
    if min_trades_required is not None:
        target_trades = min_trades_required
    else:
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
    rows.append(('profit_factor >= gate', profit_factor >= MIN_PROFIT_FACTOR, f'{profit_factor:.2f}', f'>= {MIN_PROFIT_FACTOR}'))
    avg_win_r  = float(r_multiple[wins_mask].mean()) if wins_mask.any() else 0.0
    avg_loss_r = float(-r_multiple[~wins_mask].mean()) if (~wins_mask).any() else 0.0
    expectancy_r = win_rate * avg_win_r - (1.0 - win_rate) * avg_loss_r
    rows.append(('expectancy_R > 0', expectancy_r > 0, f'{expectancy_r:.3f}R', '> 0'))
    per_coin_mean_r = [float(r_multiple[stock_idx == c].mean()) for c in np.unique(stock_idx)]
    median_coin_r = float(np.median(per_coin_mean_r)) if per_coin_mean_r else 0.0
    rows.append(('median-coin mean-R > gate', median_coin_r > MEDIAN_COIN_R_GATE,
                 f'{median_coin_r:.3f}R', f'> {MEDIAN_COIN_R_GATE}'))
    return rows


def print_gate_diagnosis_signal(r_multiple, pct_return, bars_held, entry_years,
                                 stock_idx, is_oos=False, label="GATE DIAGNOSIS",
                                 min_trades_required=None):
    rows = diagnose_gates_signal(r_multiple, pct_return, bars_held, entry_years, stock_idx,
                                  is_oos, min_trades_required)
    print(f"\n{'-'*60}\n{label}\n{'-'*60}")
    first_fail = False
    for name, passed, actual, threshold in rows:
        marker = "PASS" if passed else "FAIL"
        print(f"  [{marker}] {name:<26} actual={actual!s:<10} needed {threshold}")
        if not passed and not first_fail:
            print("       ^-- this is the gate that produced the -999 score")
            first_fail = True
    print("-" * 60)


# ==========================================================================
# 7. NEIGHBORHOOD STABILITY (conditional on which params are actually active)
# ==========================================================================

def passes_neighborhood_check(p, base_score, opens, closes, atr, adx,
                               years_arr, n_stocks, eligible_mask, start_day, end_day):
    if base_score <= 0:
        return True

    perturbations = []
    if p['entry_type'] == ENTRY_MA_BREAKOUT:
        perturbations += [{'entry_ma_len': p['entry_ma_len'] + 10},
                           {'entry_ma_len': max(MA_LEN_MIN, p['entry_ma_len'] - 10)}]
    elif p['entry_type'] == ENTRY_RSI_XOVER:
        perturbations += [{'rsi_f_len': p['rsi_f_len'] + 5},
                           {'rsi_s_len': p['rsi_s_len'] + 5}]
        if p['use_rsi_trend_filter']:
            perturbations += [{'rsi_trend_ma_len': p['rsi_trend_ma_len'] + 10}]
    elif p['entry_type'] == ENTRY_MA_XOVER:
        perturbations += [{'xover_short_len': p['xover_short_len'] + 10},
                           {'xover_long_len': p['xover_long_len'] + 10}]

    if p['exit_type'] == EXIT_MA_CROSSUNDER:
        perturbations += [{'exit_ma_len': p['exit_ma_len'] + 5}]
    elif p['exit_type'] == EXIT_PCT_TRAIL:
        perturbations += [{'trail_pct': p['trail_pct'] * 0.85}, {'trail_pct': p['trail_pct'] * 1.15}]
    elif p['exit_type'] == EXIT_ATR_TRAIL:
        perturbations += [{'exit_atr_mult': p['exit_atr_mult'] * 0.85}]
    elif p['exit_type'] == EXIT_HYBRID:
        perturbations += [{'sl_mult': p['sl_mult'] * 0.85}, {'sl_mult': p['sl_mult'] * 1.15}]

    if p.get('use_btc_entry_gate') or p.get('use_btc_exit_override'):
        perturbations += [{'btc_ma_len': p['btc_ma_len'] + 10}]

    for delta in perturbations:
        n_p = p.copy()
        n_p.update(delta)
        n_score, _ = evaluate_params_signal(n_p, opens, closes, atr, adx,
                                            years_arr, n_stocks, eligible_mask,
                                            start_day, end_day, is_oos=False)
        if n_score < base_score * NEIGHBOR_THRESHOLD:
            return False
    return True


# ==========================================================================
# 8. BASELINE CONFIG LOADER
# ==========================================================================

def load_baseline_config(filename=BASELINE_CONFIG_FILE):
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, 'r') as f:
            raw = json.load(f)
    except Exception as e:
        print(f"\n** Could not parse {filename}: {e}. Ignoring baseline. **")
        return None
    params = raw.get('params', raw) if isinstance(raw, dict) else None
    if not params:
        print(f"\n** {filename} exists but has no usable 'params'. Ignoring. **")
        return None
    print(f"\nLoaded baseline config from {filename}.")
    return params


# ==========================================================================
# 9. WALK-FORWARD FINAL STABILITY REPLAY
# ==========================================================================

def walk_forward_stability_check(p, opens, closes, atr, adx, years_arr, n_stocks,
                                  eligible_mask, full_start_day, full_end_day,
                                  initial_train_years=WF_INITIAL_TRAIN_YEARS):
    """
    Anchored-expanding-walk-forward-INSPIRED final replay, deliberately
    SIMPLIFIED from a true walk-forward search: this replays the ALREADY-
    FROZEN candidate params (found by the Optuna search above) across
    successive single-calendar-year test slices. It does NOT re-fit
    parameters per fold -- a true walk-forward search that re-optimizes
    on every expanding fold for every shortlist candidate would cost
    roughly (n_folds x full_search_budget) per candidate, which isn't
    affordable at this stage.

    What this DOES tell you, cheaply: does this specific frozen config
    keep working, year after year, as the window expands forward -- or
    did the aggregate IS/OOS number only look good because of one lucky
    early stretch that a single 70/30 split can hide? A config that fails
    this replay while still passing the aggregate split is a real flag.

    A fold is INCONCLUSIVE only when it has too few trades to say
    anything (< WF_MIN_FOLD_TRADES). A fold that fired PLENTY of trades
    and genuinely lost money is a real FAIL, not a shrug -- this reads
    the raw trade log directly (via collect_trade_log) instead of routing
    through the full aggregate hard-gate suite, specifically so a thin-
    but-real bad year can't get relabeled as a no-verdict the way it
    would if win-rate/profit-factor/etc gates were applied to one year
    of data in isolation.
    """
    window_years = years_arr[full_start_day:full_end_day]
    if len(window_years) == 0:
        return {'folds': [], 'pass_rate': 0.0, 'n_conclusive': 0, 'n_inconclusive': 0}
    min_year = int(window_years.min())
    max_year = int(window_years.max())

    folds = []
    test_year = min_year + initial_train_years
    while test_year <= max_year:
        day_idxs = np.where(years_arr == test_year)[0]
        day_idxs = day_idxs[(day_idxs >= full_start_day) & (day_idxs < full_end_day)]
        if len(day_idxs) < 30:
            test_year += 1
            continue
        s_day, e_day = int(day_idxs[0]), int(day_idxs[-1]) + 1
        log = collect_trade_log(p, opens, closes, atr, adx, years_arr, n_stocks,
                                 eligible_mask, start_day=s_day, end_day=e_day)
        n_trades = len(log['r_multiple']) if log is not None else 0
        conclusive = n_trades >= WF_MIN_FOLD_TRADES
        expectancy_r = None
        if conclusive:
            r = log['r_multiple']
            wins = r > 0
            wr = float(wins.mean())
            avg_win = float(r[wins].mean()) if wins.any() else 0.0
            avg_loss = float(-r[~wins].mean()) if (~wins).any() else 0.0
            expectancy_r = wr * avg_win - (1.0 - wr) * avg_loss
        folds.append({
            'test_year': test_year,
            'conclusive': conclusive,
            'expectancy_r': expectancy_r,
            'trades': n_trades,
        })
        test_year += 1

    conclusive_folds = [f for f in folds if f['conclusive']]
    n_profitable = sum(1 for f in conclusive_folds if (f['expectancy_r'] or 0) > 0)
    pass_rate = (n_profitable / len(conclusive_folds)) if conclusive_folds else 0.0

    return {
        'folds': folds,
        'pass_rate': pass_rate,
        'n_conclusive': len(conclusive_folds),
        'n_inconclusive': len(folds) - len(conclusive_folds),
    }


# ==========================================================================
# 10. SHORTLIST MANAGEMENT
# ==========================================================================

def _dedup_key(p):
    """Rounds params before hashing so near-identical Optuna trials don't
    all separately clutter the shortlist with the same underlying idea."""
    rounded = {}
    for k, v in sorted(p.items()):
        if isinstance(v, bool):
            rounded[k] = v
        elif isinstance(v, float):
            rounded[k] = round(v, 1)
        elif isinstance(v, int):
            rounded[k] = int(round(v / 5.0) * 5) if 'len' in k or 'thresh' in k else v
        else:
            rounded[k] = v
    return hashlib.sha1(json.dumps(rounded, sort_keys=True, default=str).encode()).hexdigest()


class ShortlistPool:
    """Keeps a bounded pool of distinct, gate-passing candidates found
    during the search, ranked by IS score. Final walk-forward replay +
    full reporting is deferred to the top SHORTLIST_SIZE at the very end,
    so the expensive diagnostics only ever run on genuine finalists."""

    def __init__(self, cap=SHORTLIST_POOL_CAP):
        self.cap = cap
        self.pool = {}   # key -> {'params':..., 'score':..., 'metrics':...}

    def add(self, params, score, metrics):
        key = _dedup_key(params)
        existing = self.pool.get(key)
        if existing is None or score > existing['score']:
            self.pool[key] = {'params': params, 'score': score, 'metrics': metrics}
        if len(self.pool) > self.cap:
            worst_key = min(self.pool, key=lambda k: self.pool[k]['score'])
            if worst_key != key:
                del self.pool[worst_key]

    def top(self, n):
        return sorted(self.pool.values(), key=lambda x: x['score'], reverse=True)[:n]


def write_shortlist(candidates, stock_names, min_year, max_year, filename=SHORTLIST_FILE):
    """candidates: list of dicts with params/is_score/oos_score/metrics/
    walk_forward. This is the interface contract with crypto_portfolio_engine.py
    -- every field a candidate might need to be fully reconstructed and
    re-simulated under real capital constraints must be present here."""
    out = {
        'engine_version': ENGINE_VERSION,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'universe': stock_names,
        'year_range': [min_year, max_year],
        'candidates': candidates,
    }

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return _clean(o.tolist())
        return o

    with open(filename, 'w') as f:
        json.dump(_clean(out), f, indent=2, default=str)
    print(f"\nWrote {len(candidates)} shortlisted candidates to {filename}")


# ==========================================================================
# 11. REPORTING
# ==========================================================================

ENTRY_TYPE_NAMES = {0: 'MA Breakout', 1: 'RSI Crossover', 2: 'MA Crossover'}
EXIT_TYPE_NAMES = {0: 'Hybrid ATR TP+Trail', 1: '%-Trail from High',
                    2: 'ATR-Trail from High', 3: 'MA Crossunder',
                    4: 'RSI Crossunder', 5: 'MA Crossover Exit'}


def print_performance_report(score, metrics, p, label=""):
    print("\n" + "=" * 78)
    print(f"{label}")
    print("=" * 78)
    print(f"Score: {score:.4f}")
    print(f"Entry: {ENTRY_TYPE_NAMES.get(p['entry_type'],'?')}   "
          f"Exit: {EXIT_TYPE_NAMES.get(p['exit_type'],'?')}   "
          f"Latched entry: {p.get('use_latched_entry', False)}   "
          f"Max pyramid layers: {p.get('max_pyramid_layers', 1)}")
    print(f"BTC entry gate: {p['use_btc_entry_gate']}   "
          f"BTC exit override: {p['use_btc_exit_override']}   "
          f"RSI trend filter: {p['use_rsi_trend_filter']}   "
          f"ADX threshold: {p.get('adx_thresh', 0.0)}")
    print(f"\nTrades: {metrics.get('trades',0)}  |  Win Rate: {metrics.get('win_rate',0)*100:.1f}%  |  "
          f"Profit Factor: {metrics.get('profit_factor',0):.2f}")
    print(f"Expectancy: {metrics.get('expectancy_r',0):.3f}R  |  SQN: {metrics.get('sqn_capped',0):.2f}  |  "
          f"Recency-weighted avg-R/trade: {metrics.get('recency_weighted_avg_r',0):.2f}")
    print(f"Median-coin mean-R: {metrics.get('median_coin_mean_r',0):.3f}R  "
          f"(the TYPICAL coin's own edge, not the portfolio-dominant one)")
    print(f"Distinct coins: {metrics.get('distinct_coins',0)}  |  Distinct years: {len(metrics.get('distinct_years',[]))}  |  "
          f"Max coin profit-share: {metrics.get('max_coin_share',0)*100:.1f}%  |  "
          f"Max year profit-share: {metrics.get('max_year_share',0)*100:.1f}%")
    if metrics.get('avg_pyramid_layers', 1.0) > 1.01:
        print(f"Pyramiding: avg {metrics.get('avg_pyramid_layers',1.0):.2f} layers/trade")
    print("=" * 78)


# ==========================================================================
# 12. OPTUNA SEARCH SPACE + MAIN
# ==========================================================================

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
        'use_latched_entry':     trial.suggest_categorical('use_latched_entry', [False, True]),
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
        p['btc_ma_len'], p['btc_ma_type'] = 62, 0

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


def _params_from_trial(trial):
    tp = dict(trial.params)
    p = {
        'entry_type': tp['entry_type'], 'exit_type': tp['exit_type'],
        'use_btc_entry_gate': tp['use_btc_entry_gate'],
        'use_btc_exit_override': tp['use_btc_exit_override'],
        'use_rsi_trend_filter': tp['use_rsi_trend_filter'],
        'use_latched_entry': tp['use_latched_entry'],
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


def run_signal_search(bayesian_trials=BAYESIAN_TRIALS, top_n_coins=TOP_N_COINS):
    tickers = fetch_top_universe(top_n_coins)
    (opens, closes, atr, adx, years_arr,
     stock_names, eligible_mask, master_dates) = prepare_matrix_data(tickers)

    clear_caches()
    n_stocks = closes.shape[1]
    n_days = closes.shape[0]
    is_end = int(n_days * WFO_IS_PCT)
    inner_split = is_end - int(is_end * INNER_VAL_PCT)

    print(f"\nMatrix: {n_days} days x {n_stocks} coins")
    print(f"Inner-train: days 0-{inner_split}  |  Inner-val: days {inner_split}-{is_end}  |  "
          f"True OOS (final check only): days {is_end}-{n_days}")

    pool = ShortlistPool()
    all_trial_srs = []

    baseline_params = load_baseline_config()
    if baseline_params is not None:
        base_score, base_metrics = evaluate_params_signal(
            baseline_params, opens, closes, atr, adx, years_arr, n_stocks, eligible_mask,
            start_day=0, end_day=is_end, is_oos=False)
        if base_score > -900:
            print_performance_report(base_score, base_metrics, baseline_params,
                                     "BASELINE CONFIG (in-sample) -- trial #1, the bar to beat")
            pool.add(baseline_params, base_score, base_metrics)

    if not OPTUNA_AVAILABLE:
        print("Optuna not installed -- cannot run the search. Install with: pip install optuna")
        return

    print(f"\nBayesian Optimization ({bayesian_trials} trials, robustness-aware)...")

    def optuna_objective(trial):
        p = _suggest_params(trial)
        train_score, train_m = evaluate_params_signal(p, opens, closes, atr, adx,
                                                       years_arr, n_stocks, eligible_mask,
                                                       start_day=0, end_day=inner_split, is_oos=False)
        if train_m and train_m.get('std_r', 0) > 0:
            all_trial_srs.append(train_m['mean_r'] / train_m['std_r'])
        if train_score <= -900:
            return -999.0

        val_score, val_m = evaluate_params_signal(p, opens, closes, atr, adx,
                                                   years_arr, n_stocks, eligible_mask,
                                                   start_day=inner_split, end_day=is_end, is_oos=True,
                                                   min_years_required=MIN_YEARS_GATE_INNER,
                                                   min_trades_required=MIN_TRADES_INNER_VAL)
        if val_score <= -900:
            return min(train_score, 20.0) * ROBUST_FALLBACK_SCALE - ROBUST_FALLBACK_PENALTY

        dual_score = min(train_score, val_score)
        if dual_score > 0:
            pool.add(p, dual_score, train_m)
        return dual_score

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=400, multivariate=True, group=True)
    )
    if baseline_params is not None:
        study.enqueue_trial({k: v for k, v in baseline_params.items()
                              if k in ['entry_type', 'exit_type', 'use_btc_entry_gate',
                                       'use_btc_exit_override', 'use_rsi_trend_filter',
                                       'use_latched_entry', 'adx_thresh', 'sl_mult', 'tp_mult',
                                       'trail_mult', 'trail_pct', 'exit_atr_mult', 'max_pyramid_layers']})

    study.optimize(optuna_objective, n_trials=bayesian_trials, show_progress_bar=True)

    print(f"\nSearch complete. {len(pool.pool)} distinct dual-pass candidates found.")
    print("Running final walk-forward stability replay + neighborhood check on the top "
          f"{SHORTLIST_SIZE} candidates...")

    finalists = pool.top(SHORTLIST_SIZE)
    shortlist_out = []
    for rank, cand in enumerate(finalists, start=1):
        p = cand['params']
        neighbor_ok = passes_neighborhood_check(p, cand['score'], opens, closes, atr, adx,
                                                 years_arr, n_stocks, eligible_mask, 0, inner_split)
        wf = walk_forward_stability_check(p, opens, closes, atr, adx, years_arr, n_stocks,
                                          eligible_mask, 0, n_days - 1)

        is_score, is_m = evaluate_params_signal(p, opens, closes, atr, adx, years_arr, n_stocks,
                                                 eligible_mask, 0, is_end, is_oos=False)
        oos_score, oos_m = evaluate_params_signal(p, opens, closes, atr, adx, years_arr, n_stocks,
                                                   eligible_mask, is_end, n_days - 1, is_oos=True)

        dsr = None
        if len(all_trial_srs) >= 30 and is_m.get('std_r', 0) > 0:
            skew = float(pd.Series(is_m['r_multiple']).skew())
            kurt = float(pd.Series(is_m['r_multiple']).kurtosis() + 3)
            sr_hat = is_m['mean_r'] / is_m['std_r']
            dsr, _, _ = deflated_sharpe_ratio(sr_hat, all_trial_srs, is_m['trades'], skew, kurt)

        label = f"SHORTLIST #{rank}  (neighbor-stable: {neighbor_ok}, WF pass-rate: {wf['pass_rate']*100:.0f}% over {wf['n_conclusive']} conclusive yrs)"
        print_performance_report(is_score, is_m, p, label)

        shortlist_out.append({
            'rank': rank,
            'config_id': _dedup_key(p),
            'params': p,
            'neighbor_stable': neighbor_ok,
            'is_score': is_score,
            'oos_score': oos_score,
            'oos_cleared_gates': oos_score > -900,
            'deflated_sharpe_ratio': dsr,
            'walk_forward': wf,
            'signal_metrics': {k: v for k, v in is_m.items()
                                if k not in ('r_multiple', 'r_multiple_raw', 'pct_return', 'bars_held',
                                             'entry_years', 'stock_idx', 'entry_days', 'exit_days',
                                             'entry_prices', 'exit_prices', 'layers_used')},
        })

    window_years = years_arr[0:n_days - 1]
    write_shortlist(shortlist_out, stock_names, int(window_years.min()), int(window_years.max()))
    print("\nDone. Hand crypto_signal_shortlist.json to crypto_portfolio_engine.py to see how "
          "these candidates actually hold up under a real shared cash pool.")


if __name__ == "__main__":
    run_signal_search()