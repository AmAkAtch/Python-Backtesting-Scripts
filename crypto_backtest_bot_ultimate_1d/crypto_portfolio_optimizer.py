"""
CRYPTO PORTFOLIO + WATCHLIST OPTIMIZER  --  v2.0
================================================
Sibling to the signal-research engine (crypto_signal_research_v1).

PURPOSE
  The signal engine already answered: "is this entry+exit logic a good
  trade generator?"  This file answers the orthogonal question:
      "given a FIXED, already-validated signal, how should we manage a
       shared cash pool, a watchlist of shadow candidates, ranking,
       position sizing, and removal rules so the live portfolio compounds
       best?"

  Signal parameters are NEVER re-optimized here.  They are loaded from
  best_params_crypto_signal_v1.json (or BASELINE_SIGNAL_FILE) and locked.
  Optuna only searches the watchlist / capital-allocation surface.

WHAT IS FIXED (from signal winner)
  - entry_type + every MA/RSI length/type that defines the entry
  - exit_type  + every parameter that defines the real-position exit
  - use_btc_entry_gate / use_btc_exit_override / use_rsi_trend_filter
  - max_pyramid_layers (signal-side pyramiding; portfolio can still decide
    whether to allow additional capital adds)
  - the universe of coins the signal engine considered (re-fetched with
    the same TOP_N + liquidity rules so the two engines stay consistent)

WHAT OPTUNA SEARCHES (watchlist + capital surface)
  Ranking method for "which watchlist coin gets the next free ticket"
    0 = oldest-on-watchlist first
    1 = strongest recent momentum (close / MA)
    2 = highest ADX (trend strength)
    3 = highest volume rank (liquidity preference)
    4 = composite score that Optuna itself weights (momentum + ADX + vol)
  Watchlist removal family (independent of the real-position exit)
    0 = same exit logic the signal winner uses (current behaviour)
    1 = pure age timeout (Optuna picks max days)
    2 = opposite of the entry signal (mirror cross)
    3 = dedicated RSI crossunder (own lengths)
    4 = BTC regime turns bearish
    5 = rank falls below a threshold (score-based drop)
  Capital / sizing knobs
    - max concurrent real positions
    - equal-weight target vs fixed-fraction
    - min ticket size relative to MONTHLY_SIP
    - whether portfolio-level pyramiding (adding capital to an already-
      open winner) is allowed on top of the signal's own layer count
  Soft filters that sit ON TOP of the frozen signal
    - optional ADX floor for NEW entries only
    - optional minimum average volume for NEW entries only

SCORING
  Re-uses the portfolio engine's wealth-curve metrics (Calmar, Sortino,
  Information Ratio, money-weighted IRR, concentration gate, recency
  weighting) with the same robustness-aware inner-train / inner-val
  discipline the signal engine introduced.  A trial is scored on the
  WORSE of the two inner slices so a lucky period cannot win alone.

ROBUSTNESS
  - Walk-forward 70/30 IS/OOS
  - Neighbourhood perturbation of the *watchlist* parameters
  - Temporal (randomised SIP-date) check
  - Deflated Sharpe on the daily equity curve
  - Always save the best candidate with an explicit robustness tier
    (never silently discard)

USAGE
  1. Run the signal-research engine until you have a winner in
     best_params_crypto_signal_v1.json
  2. Point SIGNAL_WINNER_FILE at that JSON (or leave the default)
  3. python crypto_portfolio_watchlist_optimizer.py
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

warnings.filterwarnings("ignore")

# =============================================================================
# 0. USER CONFIGURATION
# =============================================================================

# -- Files --
SIGNAL_WINNER_FILE   = "best_params_crypto_signal_v1.json"   # output of the signal engine
BEST_PARAMS_FILE     = "best_params_crypto_portfolio_wl_v2.json"
INTERMEDIATE_FILE    = "current_is_champion_portfolio_wl_v2.json"
ENGINE_VERSION       = "portfolio-wl-2.0.0"

# -- Universe / data window (kept in lock-step with the signal engine) --
START_DATE           = "2018-01-01"
MIN_HISTORY_DAYS     = 400
TOP_N_COINS          = 40          # tighter liquid universe for live capital
MIN_AVG_DAILY_VOLUME_USD = 3_000_000
LIQUIDITY_LOOKBACK_DAYS  = 30

# -- Search budget --
BAYESIAN_TRIALS      = 8_000       # smaller than signal search; surface is narrower
WFO_IS_PCT           = 0.70
WFO_OOS_PCT          = 0.30
INNER_VAL_PCT        = 0.30
ROBUSTNESS_DEPLOY_THRESHOLD = 0.50
NEIGHBOR_THRESHOLD   = 0.70

# -- Capital --
MONTHLY_SIP          = 2_000.0
MIN_TICKET_SIZE      = MONTHLY_SIP
ANNUAL_CASH_YIELD    = 0.00
BINANCE_TAKER_PCT    = 0.00100
SLIPPAGE_PCT         = 0.00150
BUY_COST_PCT         = BINANCE_TAKER_PCT + SLIPPAGE_PCT
SELL_COST_PCT        = BINANCE_TAKER_PCT + SLIPPAGE_PCT
CRYPTO_DAYS_PER_YEAR = 365.0

# -- Quality gates (portfolio) --
MIN_TRADES_GATE      = 20
MIN_MONTHS_GATE      = 24
MIN_WIN_RATE_GATE    = 0.20
CONCENTRATION_GATE   = 0.80
MAX_DD_GATE          = 0.55

# -- Recency / regime (from signal + old portfolio engines) --
RECENCY_WEIGHT_MIN   = 0.80
RECENCY_WEIGHT_MAX   = 1.00
BULL_YEAR_BTC_THRESHOLD = 0.10
BEAR_YEAR_BTC_THRESHOLD = -0.10
MIN_YEAR_COVERAGE_FOR_SCORING = 0.75

# -- Score weights (sum = 1.0) --
W_CALMAR      = 0.20
W_SORTINO     = 0.10
W_IR          = 0.20
W_EV          = 0.10
W_WR_BONUS    = 0.10
W_CONSISTENCY = 0.20
W_REGIME      = 0.10

# -- Temporal robustness --
SIP_RANDOM_MIN_DAY = 1
SIP_RANDOM_MAX_DAY = 28
TEMPORAL_ROBUSTNESS_RUNS      = 12
TEMPORAL_ROBUSTNESS_SCORE_MIN = 0.70
TEMPORAL_ROBUSTNESS_PASS_FRAC = 0.70

# -- Optuna --
OPTUNA_N_JOBS = max(1, (os.cpu_count() or 2) - 1)
MA_CACHE_MAX_ENTRIES = 30_000

# -- Watchlist vocabulary (what Optuna can choose) --
WL_RANK_OLDEST        = 0
WL_RANK_MOMENTUM      = 1
WL_RANK_ADX           = 2
WL_RANK_VOLUME        = 3
WL_RANK_COMPOSITE     = 4

WL_REMOVE_SAME_EXIT   = 0
WL_REMOVE_AGE         = 1
WL_REMOVE_MIRROR      = 2
WL_REMOVE_RSI         = 3
WL_REMOVE_BTC         = 4
WL_REMOVE_RANK_DROP   = 5

# Entry / exit type IDs must match the signal engine exactly
ENTRY_MA_BREAKOUT = 0
ENTRY_RSI_XOVER   = 1
ENTRY_MA_XOVER    = 2

EXIT_HYBRID          = 0
EXIT_PCT_TRAIL       = 1
EXIT_ATR_TRAIL       = 2
EXIT_MA_CROSSUNDER   = 3
EXIT_RSI_CROSSUNDER  = 4
EXIT_MA_XOVER_EXIT   = 5

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


def _recency_weight(year, min_year, max_year):
    if max_year <= min_year:
        return RECENCY_WEIGHT_MAX
    frac = (year - min_year) / (max_year - min_year)
    return RECENCY_WEIGHT_MIN + (RECENCY_WEIGHT_MAX - RECENCY_WEIGHT_MIN) * frac


# =============================================================================
# 1. LOAD SIGNAL WINNER (the only source of truth for entry/exit)
# =============================================================================

def load_signal_winner(path=SIGNAL_WINNER_FILE):
    """Load the JSON written by the signal-research engine.
    Accepts either the full save_winner wrapper or a bare params dict.
    Returns a clean native-schema dict or raises with a clear message."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Signal winner not found at '{path}'.\n"
            f"Run the signal-research engine first and place its "
            f"best_params_crypto_signal_v1.json next to this script "
            f"(or set SIGNAL_WINNER_FILE)."
        )
    with open(path, "r") as f:
        raw = json.load(f)

    params = raw.get("params", raw) if isinstance(raw, dict) else None
    if not params:
        raise ValueError(f"{path} contains no usable 'params' dict.")

    required = {"entry_type", "exit_type"}
    missing = required - set(params.keys())
    if missing:
        raise ValueError(
            f"Signal winner is missing required keys {missing}. "
            f"Re-run the signal engine or translate a legacy config."
        )

    # Fill safe defaults for optional toggles the signal engine always has
    defaults = {
        "use_btc_entry_gate": False,
        "use_btc_exit_override": False,
        "use_rsi_trend_filter": False,
        "adx_thresh": 0.0,
        "sl_mult": 4.0,
        "tp_mult": 30.0,
        "trail_mult": 6.0,
        "trail_pct": 15.0,
        "exit_atr_mult": 3.0,
        "max_pyramid_layers": 1,
        "btc_ma_len": 62,
        "btc_ma_type": 0,
    }
    for k, v in defaults.items():
        params.setdefault(k, v)

    print("=" * 70)
    print(f"LOADED SIGNAL WINNER from {path}")
    print(f"  engine_version (signal): {raw.get('engine_version', '?')}")
    print(f"  OOS score (signal):      {raw.get('oos_score', '?')}")
    print(f"  IS  score (signal):      {raw.get('is_score', '?')}")
    print(f"  entry_type: {params['entry_type']}   exit_type: {params['exit_type']}")
    print(f"  BTC gates: entry={params['use_btc_entry_gate']}  "
          f"exit_override={params['use_btc_exit_override']}")
    print("=" * 70)
    return params


# =============================================================================
# 2. UNIVERSE (identical spirit to the signal engine)
# =============================================================================

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
        return {s["symbol"] for s in data["symbols"]
                if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"}
    except Exception as e:
        print(f"Could not fetch Binance exchangeInfo ({e}); will validate per-ticker.")
        return None


def fetch_top_universe():
    print(f"Fetching top-{TOP_N_COINS}-by-market-cap universe from CoinGecko...")
    binance_usdt = _get_binance_usdt_symbols()
    tickers, excluded_log = [], []
    try:
        r = requests.get(COINGECKO_MARKETS_URL, params={
            "vs_currency": "usd", "order": "market_cap_desc",
            "per_page": TOP_N_COINS, "page": 1, "sparkline": "false"
        }, timeout=20)
        r.raise_for_status()
        for c in r.json():
            sym = str(c.get("symbol", "")).upper()
            if not sym or _is_excluded(sym):
                if sym:
                    excluded_log.append(sym)
                continue
            vol = c.get("total_volume", 0) or 0
            if sym != "BTC" and vol < MIN_AVG_DAILY_VOLUME_USD:
                excluded_log.append(f"{sym} (low vol)")
                continue
            binance_sym = f"{sym}USDT"
            if binance_usdt is not None and binance_sym not in binance_usdt:
                excluded_log.append(f"{sym} (no USDT pair)")
                continue
            tickers.append(binance_sym)
    except Exception as e:
        print(f"CoinGecko failed ({e}); using static fallback.")
        tickers = list(FALLBACK_UNIVERSE)

    tickers = [t for t in tickers if t != "BTCUSDT"]
    tickers = ["BTCUSDT"] + list(dict.fromkeys(tickers))
    if len(tickers) < 10:
        tickers = list(FALLBACK_UNIVERSE)
    print(f"Universe size: {len(tickers)} (excluded {len(excluded_log)})")
    assert tickers[0] == "BTCUSDT"
    return tickers


# =============================================================================
# 3. INDICATORS (shared with signal engine – NaN-safe, 5 MA types)
# =============================================================================

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
        mult = 2.0 / (period + 1)
        for i in range(start + period, n):
            ema1[i] = (prices[i] - ema1[i - 1]) * mult + ema1[i - 1]
        if ma_type == 1:
            res = ema1
        else:
            ema2 = np.empty(n)
            ema2[:] = np.nan
            if start + period * 2 - 2 < n:
                ema2[start + period * 2 - 2] = np.mean(
                    ema1[start + period - 1:start + period * 2 - 1])
                for i in range(start + period * 2 - 1, n):
                    ema2[i] = (ema1[i] - ema2[i - 1]) * mult + ema2[i - 1]
                for i in range(start + period * 2 - 2, n):
                    res[i] = 2.0 * ema1[i] - ema2[i]
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
        rsi[idx0] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    else:
        rsi[idx0] = 100.0
    for i in range(idx0 + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss > 0:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
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
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]))
    atr_s = np.sum(tr[start + 1:start + period + 1]) / period
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
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(start + 1, n):
        high_diff = highs[i] - highs[i - 1]
        low_diff = lows[i - 1] - lows[i]
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]))
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0.0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0.0
    atr_s = np.sum(tr[start + 1:start + period + 1])
    plus_s = np.sum(plus_dm[start + 1:start + period + 1])
    minus_s = np.sum(minus_dm[start + 1:start + period + 1])
    for i in range(start + period, n):
        if i > start + period:
            atr_s = atr_s - atr_s / period + tr[i]
            plus_s = plus_s - plus_s / period + plus_dm[i]
            minus_s = minus_s - minus_s / period + minus_dm[i]
        if atr_s > 0:
            plus_di = 100.0 * plus_s / atr_s
            minus_di = 100.0 * minus_s / atr_s
            denom = plus_di + minus_di
            dx = 100.0 * abs(plus_di - minus_di) / denom if denom > 0 else 0.0
        else:
            dx = 0.0
        if i == start + period:
            adx[i] = dx
        else:
            adx[i] = (adx[i - 1] * (period - 1) + dx) / period
    return adx


# Caches
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


# =============================================================================
# 4. DATA DOWNLOAD / MATRIX BUILD
# =============================================================================

def _binance_download_klines(symbol, start_date_str, interval="1d"):
    start_ts = int(datetime.strptime(start_date_str, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows = []
    cursor = start_ts
    while cursor < end_ts:
        params = {"symbol": symbol, "interval": interval,
                  "startTime": cursor, "limit": 1000}
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
        time.sleep(0.12)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume", "quote_vol"):
        df[col] = df[col].astype(float)
    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def prepare_matrix_data(tickers, data_dir="data_crypto_portfolio"):
    os.makedirs(data_dir, exist_ok=True)
    frames = {}
    for sym in tqdm(tickers, desc="Downloading / loading klines"):
        path = os.path.join(data_dir, f"{sym}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
        else:
            df = _binance_download_klines(sym, START_DATE)
            if df is not None and len(df) >= MIN_HISTORY_DAYS:
                df.to_parquet(path)
            else:
                print(f"  dropping {sym} (insufficient history)")
                continue
        if len(df) < MIN_HISTORY_DAYS:
            continue
        frames[sym] = df

    if "BTCUSDT" not in frames:
        raise RuntimeError("BTCUSDT is required and could not be loaded.")

    # Align on a common calendar (union of all dates, forward-fill NaNs later)
    all_dates = sorted(set().union(*(df.index for df in frames.values())))
    master_dates = pd.DatetimeIndex(all_dates)
    n_days = len(master_dates)
    n_stocks = len(frames)
    stock_names = list(frames.keys())
    # Force BTC to column 0
    if stock_names[0] != "BTCUSDT":
        stock_names.remove("BTCUSDT")
        stock_names = ["BTCUSDT"] + stock_names

    opens = np.full((n_days, n_stocks), np.nan)
    highs = np.full((n_days, n_stocks), np.nan)
    lows = np.full((n_days, n_stocks), np.nan)
    closes = np.full((n_days, n_stocks), np.nan)
    quote_vol = np.full((n_days, n_stocks), np.nan)

    date_to_idx = {d: i for i, d in enumerate(master_dates)}
    for s, sym in enumerate(stock_names):
        df = frames[sym]
        for d, row in df.iterrows():
            i = date_to_idx.get(d)
            if i is None:
                continue
            opens[i, s] = row["open"]
            highs[i, s] = row["high"]
            lows[i, s] = row["low"]
            closes[i, s] = row["close"]
            quote_vol[i, s] = row.get("quote_vol", row.get("volume", 0.0))

    # Indicators
    atr = np.full((n_days, n_stocks), np.nan)
    adx = np.full((n_days, n_stocks), np.nan)
    for s in range(n_stocks):
        atr[:, s] = calc_atr_wilder(highs[:, s], lows[:, s], closes[:, s], 14)
        adx[:, s] = calc_adx(highs[:, s], lows[:, s], closes[:, s], 14)

    years_arr = np.array([d.year for d in master_dates], dtype=np.int32)
    months_arr = np.array([d.month for d in master_dates], dtype=np.int32)

    # Eligibility = has enough non-NaN history by that day
    eligible = np.zeros((n_days, n_stocks), dtype=np.bool_)
    for s in range(n_stocks):
        valid = ~np.isnan(closes[:, s])
        cum = np.cumsum(valid)
        eligible[:, s] = cum >= MIN_HISTORY_DAYS

    # Liquidity mask (point-in-time trailing average)
    liquidity = np.ones((n_days, n_stocks), dtype=np.bool_)
    for s in range(1, n_stocks):  # BTC always liquid
        for d in range(LIQUIDITY_LOOKBACK_DAYS, n_days):
            window = quote_vol[d - LIQUIDITY_LOOKBACK_DAYS:d, s]
            avg = np.nanmean(window)
            if not (avg >= MIN_AVG_DAILY_VOLUME_USD):
                liquidity[d, s] = False

    eligible = eligible & liquidity

    print(f"Matrix ready: {n_days} days x {n_stocks} coins")
    return (opens, closes, atr, adx, months_arr, years_arr,
            stock_names, eligible, master_dates, quote_vol)


def build_sip_schedule(master_dates, mode="month_start", seed=None):
    n = len(master_dates)
    flag = np.zeros(n, dtype=np.bool_)
    if mode == "month_start":
        for i in range(1, n):
            if master_dates[i].month != master_dates[i - 1].month:
                flag[i] = True
        flag[0] = True
    else:
        rng = np.random.RandomState(seed if seed is not None else 42)
        day_of_month = rng.randint(SIP_RANDOM_MIN_DAY, SIP_RANDOM_MAX_DAY + 1)
        for i in range(n):
            if master_dates[i].day == day_of_month:
                flag[i] = True
    return flag


# =============================================================================
# 5. FIXED-SIGNAL + OPTIMIZABLE-WATCHLIST PORTFOLIO SIMULATOR
# =============================================================================
# The signal (entry + real-position exit) is completely determined by the
# loaded winner.  Only ranking, removal, max positions, and sizing are free.

@njit(nogil=True)
def simulate_portfolio_fixed_signal(
        opens, closes, atr, adx,
        # pre-computed signal arrays (one column per coin)
        entry_signal,          # bool (n_days, n_stocks) – already includes BTC gate etc.
        real_exit_signal,      # bool – the frozen real-position exit
        mirror_exit_signal,    # bool – opposite of entry (for WL_REMOVE_MIRROR)
        wl_rsi_exit_signal,    # bool – dedicated RSI crossunder for watchlist
        btc_bearish,           # bool (n_days,)
        # watchlist / capital knobs (the only things Optuna varies)
        wl_rank_method,
        wl_remove_method,
        wl_max_age_days,
        wl_rank_drop_thresh,
        max_concurrent_positions,
        allow_portfolio_pyramid,
        min_ticket,
        # -- frozen exit-family numeric params (needed to actually run the
        #    live per-trade Hybrid/%-Trail/ATR-Trail exit state machine;
        #    exit types 3/4/5 are already fully captured in real_exit_signal
        #    and ignore these) --
        signal_exit_type, sl_mult, tp_mult, trail_mult, trail_pct, exit_atr_mult,
        # -- capital-deployment forks (Optuna-searched, fully independent) --
        entry_funding_method,   # 0=cash-pool-only  1=trim-overweight-only  2=full-bidirectional-rebalance
        exit_proceeds_method,   # 0=leave liquid     1=redistribute evenly across remaining open positions
        sizing_denom_method,    # 0=max_concurrent_positions  1=causal running-avg concurrency + buffer  2=current open count (always fully deployed)
        concurrency_buffer,     # extra assumed slots, only used when sizing_denom_method == 1
        # common
        eligible_mask,
        months, sip_flag,
        start_day, end_day,
        starting_wealth,
        buy_cost_pct, sell_cost_pct, annual_cash_yield,
        monthly_sip,
        # composite rank weights (only used when wl_rank_method == COMPOSITE)
        w_mom, w_adx, w_vol,
        momentum_ma,           # (n_days, n_stocks) close/MA for ranking
        volume_rank,           # (n_days, n_stocks) pre-ranked 0-1
):
    n_days, n_stocks = closes.shape
    if end_day < 0 or end_day >= n_days:
        end_day = n_days - 2
    if start_day < 1:
        start_day = 1

    daily_yield_mult = (1.0 + annual_cash_yield) ** (1.0 / 365.0)
    btc_close = closes[:, 0]

    cash_pool = starting_wealth
    total_invested = 0.0

    in_pos = np.zeros(n_stocks, dtype=np.bool_)
    entry_prices = np.zeros(n_stocks)
    entry_days = np.zeros(n_stocks, dtype=np.int32)
    shares_held = np.zeros(n_stocks)
    high_since_entry = np.zeros(n_stocks)
    stop_loss_price = np.zeros(n_stocks)
    tp_trigger_price = np.zeros(n_stocks)
    half_sold = np.zeros(n_stocks, dtype=np.bool_)
    pending_partial = np.zeros(n_stocks, dtype=np.bool_)

    # Causal (expanding-window, no-lookahead) running average of realized
    # concurrent open-position count -- only used by sizing_denom_method==1.
    cum_n_open_sum = 0.0
    cum_n_open_days = 0
    sum_n_open_report = 0.0
    max_n_open_report = 0

    # Watchlist shadow state
    wl_active = np.zeros(n_stocks, dtype=np.bool_)
    wl_entry_day = np.zeros(n_stocks, dtype=np.int32)
    wl_entry_price = np.zeros(n_stocks)

    pending_exit = np.zeros(n_stocks, dtype=np.bool_)
    pending_entry = np.zeros(n_stocks, dtype=np.bool_)  # from watchlist → real

    total_wins = 0.0
    total_losses = 0.0
    winning_trades = 0
    losing_trades = 0
    total_bars = 0
    trades = 0

    daily_port_val = np.zeros(n_days)
    daily_bench_val = np.zeros(n_days)
    daily_ret_p = np.zeros(n_days)
    daily_ret_b = np.zeros(n_days)
    sip_injected = np.zeros(n_days)

    # Benchmark (BTC DCA)
    bench_shares = 0.0
    if starting_wealth > 0 and btc_close[start_day] > 0:
        bench_shares = starting_wealth / btc_close[start_day]

    MAX_CF = n_days + 8
    cf_days = np.zeros(MAX_CF, dtype=np.int32)
    cf_amounts = np.zeros(MAX_CF)
    cf_cnt = 0
    if starting_wealth > 0:
        cf_days[0] = start_day
        cf_amounts[0] = -starting_wealth
        cf_cnt = 1

    bench_cf_days = np.zeros(MAX_CF, dtype=np.int32)
    bench_cf_amounts = np.zeros(MAX_CF)
    bench_cf_cnt = 0
    if starting_wealth > 0:
        bench_cf_days[0] = start_day
        bench_cf_amounts[0] = -starting_wealth
        bench_cf_cnt = 1

    n_open = 0

    for d in range(start_day, end_day):
        cash_pool *= daily_yield_mult

        # ---- SIP ----
        if sip_flag[d]:
            cash_pool += monthly_sip
            total_invested += monthly_sip
            sip_injected[d] = monthly_sip
            if cf_cnt < MAX_CF:
                cf_days[cf_cnt] = d
                cf_amounts[cf_cnt] = -monthly_sip
                cf_cnt += 1
            if bench_cf_cnt < MAX_CF:
                bench_cf_days[bench_cf_cnt] = d
                bench_cf_amounts[bench_cf_cnt] = -monthly_sip
                bench_cf_cnt += 1
            if btc_close[d] > 0:
                bench_shares += monthly_sip / btc_close[d]

        # ---- Phase A1: settle pending partial exits (Hybrid 50% take-profit) ----
        for s in range(n_stocks):
            if pending_partial[s] and in_pos[s]:
                fill = opens[d, s]
                if not (fill > 0):
                    fill = closes[d - 1, s]
                if fill > 0 and shares_held[s] > 0:
                    sell_shares = shares_held[s] * 0.5
                    proceeds = sell_shares * fill * (1.0 - sell_cost_pct)
                    cost_basis = sell_shares * entry_prices[s]
                    profit = proceeds - cost_basis
                    # Dollar P&L is booked now so PF/ROI reflect it immediately;
                    # trade/winning_trades/losing_trades counts only increment
                    # on the FULL close below (one round-trip = one trade row,
                    # same convention as the signal-research engine).
                    if profit >= 0:
                        total_wins += profit
                    else:
                        total_losses += -profit
                    cash_pool += proceeds
                    shares_held[s] -= sell_shares
                    half_sold[s] = True
                    if stop_loss_price[s] < entry_prices[s]:
                        stop_loss_price[s] = entry_prices[s]  # breakeven floor
                pending_partial[s] = False

        # ---- Phase A2: settle pending real exits (full close) ----
        freed_this_day = 0.0
        for s in range(n_stocks):
            if pending_exit[s] and in_pos[s]:
                fill = opens[d, s]
                if not (fill > 0):
                    fill = closes[d - 1, s]
                if fill > 0 and shares_held[s] > 0:
                    exit_val = shares_held[s] * fill * (1.0 - sell_cost_pct)
                    cost = shares_held[s] * entry_prices[s]
                    profit = exit_val - cost
                    if profit >= 0:
                        total_wins += profit
                        winning_trades += 1
                    else:
                        total_losses += -profit
                        losing_trades += 1
                    total_bars += d - entry_days[s]
                    trades += 1
                    cash_pool += exit_val
                    freed_this_day += exit_val
                    in_pos[s] = False
                    shares_held[s] = 0.0
                    half_sold[s] = False
                    n_open -= 1
                pending_exit[s] = False
                pending_partial[s] = False

        # ---- Fork B: exit-proceeds handling ----
        # exit_proceeds_method 1 = immediately buy freed_this_day / n_open more
        # of every position still open (real buys, real cost); 0 = leave it in
        # cash_pool for the next new signal (previous / default behaviour).
        if exit_proceeds_method == 1 and freed_this_day > 0.0 and n_open > 0:
            top_up_each = freed_this_day / n_open
            for s in range(n_stocks):
                if not in_pos[s]:
                    continue
                fill = opens[d, s]
                if not (fill > 0):
                    fill = closes[d - 1, s]
                if fill <= 0:
                    continue
                ticket = min(top_up_each, cash_pool / (1.0 + buy_cost_pct))
                cost = ticket * (1.0 + buy_cost_pct)
                if cost <= cash_pool and ticket > 0.0:
                    sh = ticket / fill
                    new_shares = shares_held[s] + sh
                    # blend cost basis across the top-up, same convention as
                    # the signal engine's pyramid-layer blending
                    entry_prices[s] = (entry_prices[s] * shares_held[s] + fill * sh) / new_shares
                    shares_held[s] = new_shares
                    cash_pool -= cost

        # ---- Fork C: sizing denominator for the NEXT new entry ----
        if sizing_denom_method == 1:
            causal_avg_n_open = (cum_n_open_sum / cum_n_open_days) if cum_n_open_days > 0 else 1.0
            denom_assumed = causal_avg_n_open + concurrency_buffer
        elif sizing_denom_method == 2:
            denom_assumed = float(n_open + 1)  # always fully deployed across exactly what's open
        else:
            denom_assumed = float(max_concurrent_positions)

        # ---- Phase A3: settle pending entries from watchlist (fork-aware) ----
        for s in range(n_stocks):
            if pending_entry[s] and not in_pos[s] and n_open < max_concurrent_positions:
                fill = opens[d, s]
                if not (fill > 0):
                    fill = closes[d - 1, s]
                if fill <= 0:
                    pending_entry[s] = False
                    continue

                n_open_after = n_open + 1
                effective_n = max(denom_assumed, float(n_open_after))

                if entry_funding_method == 0:
                    # cash-pool-only: never touch existing positions (this is
                    # exactly the original / default behaviour, generalised
                    # only so its denominator can come from Fork C too)
                    target = cash_pool / max(1.0, effective_n - n_open)
                    # cap the ticket so ticket*(1+buy_cost_pct) never exceeds
                    # cash_pool -- otherwise, whenever cash is the binding
                    # constraint (ticket lands exactly on cash_pool, which
                    # happens routinely: the very first entry into an empty
                    # portfolio, or the last free slot), the cost markup
                    # alone pushes `cost` a hair above `cash_pool` and the
                    # `cost <= cash_pool` check below fails FOREVER -- same
                    # target recomputed, same failure, every single day,
                    # silently starving the portfolio of trades. This bug
                    # predates this fork work; it was just rarely triggered
                    # by the old default (dividing by max_concurrent_positions,
                    # usually >=3, so target rarely landed exactly on cash_pool).
                    ticket = min(target, cash_pool / (1.0 + buy_cost_pct))
                else:
                    # A1/A2: target is a share of TOTAL portfolio value
                    # (cash + all open positions marked at today's open,
                    # since that's the price this whole Phase-A step trades at)
                    port_val = cash_pool
                    for s2 in range(n_stocks):
                        if in_pos[s2]:
                            px2 = opens[d, s2]
                            if not (px2 > 0):
                                px2 = closes[d - 1, s2]
                            if px2 > 0:
                                port_val += shares_held[s2] * px2
                    target = port_val / effective_n

                    if entry_funding_method == 1:
                        # Trim-overweight-only: sell down (only) the excess
                        # each overweight position holds above target, only
                        # as much as needed to cover the newcomer's
                        # shortfall -- stop as soon as it's covered.
                        # Positions at/below target are never touched.
                        shortfall = target - cash_pool
                        if shortfall > 0.0:
                            raised = 0.0
                            for s2 in range(n_stocks):
                                if s2 == s or not in_pos[s2]:
                                    continue
                                if raised >= shortfall:
                                    break
                                px2 = opens[d, s2]
                                if not (px2 > 0):
                                    px2 = closes[d - 1, s2]
                                if px2 <= 0:
                                    continue
                                cur_val = shares_held[s2] * px2
                                trim_val = min(max(0.0, cur_val - target), shortfall - raised)
                                if trim_val > 0.0:
                                    trim_shares = trim_val / px2
                                    proceeds = trim_shares * px2 * (1.0 - sell_cost_pct)
                                    cost_basis = trim_shares * entry_prices[s2]
                                    profit = proceeds - cost_basis
                                    if profit >= 0:
                                        total_wins += profit
                                    else:
                                        total_losses += -profit
                                    cash_pool += proceeds
                                    shares_held[s2] -= trim_shares
                                    raised += trim_val
                    else:
                        # Full bidirectional rebalance: every open position
                        # (existing + newcomer) is pushed toward target.
                        # Pass 1 -- sell every existing position's OWN
                        # excess above target (never more than that, so one
                        # heavily overweight position can't get fully
                        # liquidated just to fund the newcomer -- each
                        # position's trim is capped at its own distance
                        # from target).
                        for s2 in range(n_stocks):
                            if s2 == s or not in_pos[s2]:
                                continue
                            px2 = opens[d, s2]
                            if not (px2 > 0):
                                px2 = closes[d - 1, s2]
                            if px2 <= 0:
                                continue
                            cur_val = shares_held[s2] * px2
                            excess = cur_val - target
                            if excess > 0.0:
                                trim_shares = excess / px2
                                proceeds = trim_shares * px2 * (1.0 - sell_cost_pct)
                                cost_basis = trim_shares * entry_prices[s2]
                                profit = proceeds - cost_basis
                                if profit >= 0:
                                    total_wins += profit
                                else:
                                    total_losses += -profit
                                cash_pool += proceeds
                                shares_held[s2] -= trim_shares
                        # Pass 2 -- buy every existing UNDERWEIGHT position
                        # back up toward target, funded from whatever cash
                        # is now available (best-effort in iteration order;
                        # if cash runs short, later ones simply get less).
                        for s2 in range(n_stocks):
                            if s2 == s or not in_pos[s2]:
                                continue
                            px2 = opens[d, s2]
                            if not (px2 > 0):
                                px2 = closes[d - 1, s2]
                            if px2 <= 0:
                                continue
                            cur_val = shares_held[s2] * px2
                            deficit = target - cur_val
                            if deficit > 0.0:
                                buy_val = min(deficit, cash_pool / (1.0 + buy_cost_pct))
                                if buy_val > 0.0:
                                    cost2 = buy_val * (1.0 + buy_cost_pct)
                                    if cost2 <= cash_pool:
                                        sh2 = buy_val / px2
                                        new_shares2 = shares_held[s2] + sh2
                                        entry_prices[s2] = (
                                            (entry_prices[s2] * shares_held[s2] + px2 * sh2)
                                            / new_shares2)
                                        shares_held[s2] = new_shares2
                                        cash_pool -= cost2
                    ticket = min(target, cash_pool / (1.0 + buy_cost_pct))

                if ticket >= min_ticket:
                    cost = ticket * (1.0 + buy_cost_pct)
                    if cost <= cash_pool:
                        sh = ticket / fill
                        shares_held[s] = sh
                        entry_prices[s] = fill
                        entry_days[s] = d
                        high_since_entry[s] = fill
                        half_sold[s] = False
                        e_atr = atr[d, s] if atr[d, s] > 0 else atr[d - 1, s]
                        if not (e_atr > 0):
                            e_atr = fill * 0.02
                        stop_loss_price[s] = fill - e_atr * sl_mult
                        tp_trigger_price[s] = fill + e_atr * tp_mult
                        in_pos[s] = True
                        cash_pool -= cost
                        n_open += 1
                        # remove from watchlist once bought
                        wl_active[s] = False
                pending_entry[s] = False

        # ---- Update highs / evaluate real-position exits ----
        for s in range(n_stocks):
            if not in_pos[s]:
                continue
            c = closes[d, s]
            if c > 0:
                high_since_entry[s] = max(high_since_entry[s], c)

            exit_full = False
            exit_partial = False

            if signal_exit_type == EXIT_HYBRID:
                if not half_sold[s] and c >= tp_trigger_price[s]:
                    exit_partial = True
                if half_sold[s]:
                    potential_new_sl = c - atr[d, s] * trail_mult
                    if potential_new_sl > stop_loss_price[s]:
                        stop_loss_price[s] = potential_new_sl
                if c < stop_loss_price[s]:
                    exit_full = True
            elif signal_exit_type == EXIT_PCT_TRAIL:
                if c < high_since_entry[s] * (1.0 - trail_pct / 100.0):
                    exit_full = True
            elif signal_exit_type == EXIT_ATR_TRAIL:
                if c < high_since_entry[s] - (exit_atr_mult * atr[d, s]):
                    exit_full = True

            # exit types 3/4/5 (MA crossunder / RSI crossunder / MA crossover
            # exit) are already fully baked into real_exit_signal by
            # build_signal_arrays, as is the BTC exit-override for every
            # exit type -- so it's always checked here regardless of which
            # branch above fired.
            if real_exit_signal[d, s]:
                exit_full = True

            if exit_full:
                pending_exit[s] = True
                pending_partial[s] = False  # a full exit supersedes a pending partial
            elif exit_partial:
                pending_partial[s] = True

        # ---- causal concurrency bookkeeping (today's realized n_open,
        #      available for tomorrow's sizing decision only -- no lookahead) ----
        cum_n_open_sum += n_open
        cum_n_open_days += 1
        sum_n_open_report += n_open
        if n_open > max_n_open_report:
            max_n_open_report = n_open

        # ---- Watchlist maintenance ----
        for s in range(n_stocks):
            if not wl_active[s]:
                continue
            age = d - wl_entry_day[s]
            remove = False
            if wl_remove_method == 0:  # same as real exit
                if real_exit_signal[d, s]:
                    remove = True
            elif wl_remove_method == 1:  # age
                if age >= wl_max_age_days:
                    remove = True
            elif wl_remove_method == 2:  # mirror of entry
                if mirror_exit_signal[d, s]:
                    remove = True
            elif wl_remove_method == 3:  # dedicated RSI
                if wl_rsi_exit_signal[d, s]:
                    remove = True
            elif wl_remove_method == 4:  # BTC bearish
                if btc_bearish[d]:
                    remove = True
            elif wl_remove_method == 5:  # rank drop – checked later
                pass
            if remove:
                wl_active[s] = False

        # ---- New watchlist additions (fixed signal fires) ----
        for s in range(1, n_stocks):  # skip BTC itself as a tradeable
            if in_pos[s] or wl_active[s]:
                continue
            if not eligible_mask[d, s]:
                continue
            if entry_signal[d, s]:
                wl_active[s] = True
                wl_entry_day[s] = d
                wl_entry_price[s] = closes[d, s] if closes[d, s] > 0 else 0.0

        # ---- Rank active watchlist & promote the best into pending_entry ----
        # Funding method 0 (cash-pool-only) can only ever afford a new entry
        # out of idle cash, so the gate stays cash-based. Methods 1/2 can
        # also raise capital by trimming existing positions, so gating on
        # cash_pool alone would silently starve them the moment cash runs
        # low (which, under a fully-deployed rebalancing strategy, is most
        # of the time) -- gate on total mark-to-market portfolio value
        # instead; the actual per-position ticket size is still checked
        # against min_ticket where the real trim math runs, so an
        # optimistic promotion here that can't actually raise enough simply
        # fails to fill and tries again another day, at no cost.
        if entry_funding_method == 0:
            can_afford_new_entry = cash_pool >= min_ticket
        else:
            port_val_for_gate = cash_pool
            for s2 in range(n_stocks):
                if in_pos[s2] and closes[d, s2] > 0:
                    port_val_for_gate += shares_held[s2] * closes[d, s2]
            can_afford_new_entry = port_val_for_gate >= min_ticket

        if n_open < max_concurrent_positions and can_afford_new_entry:
            best_s = -1
            best_score = -1e18
            for s in range(1, n_stocks):
                if not wl_active[s] or in_pos[s]:
                    continue
                if wl_remove_method == 5:
                    # will compute score and maybe drop
                    pass
                score = 0.0
                if wl_rank_method == 0:  # oldest first
                    score = -float(wl_entry_day[s])
                elif wl_rank_method == 1:  # momentum
                    score = momentum_ma[d, s] if not np.isnan(momentum_ma[d, s]) else 0.0
                elif wl_rank_method == 2:  # ADX
                    score = adx[d, s] if not np.isnan(adx[d, s]) else 0.0
                elif wl_rank_method == 3:  # volume
                    score = volume_rank[d, s] if not np.isnan(volume_rank[d, s]) else 0.0
                elif wl_rank_method == 4:  # composite
                    m = momentum_ma[d, s] if not np.isnan(momentum_ma[d, s]) else 0.0
                    a = adx[d, s] if not np.isnan(adx[d, s]) else 0.0
                    v = volume_rank[d, s] if not np.isnan(volume_rank[d, s]) else 0.0
                    score = w_mom * m + w_adx * a + w_vol * v
                if wl_remove_method == 5 and score < wl_rank_drop_thresh:
                    wl_active[s] = False
                    continue
                if score > best_score:
                    best_score = score
                    best_s = s
            if best_s >= 0:
                pending_entry[best_s] = True

        # ---- Mark-to-market ----
        port = cash_pool
        for s in range(n_stocks):
            if in_pos[s] and closes[d, s] > 0:
                port += shares_held[s] * closes[d, s]
        daily_port_val[d] = port
        daily_bench_val[d] = bench_shares * btc_close[d] if btc_close[d] > 0 else 0.0

        # Flow-adjusted daily returns
        if d > start_day and daily_port_val[d - 1] > 0:
            inflow = sip_injected[d]
            daily_ret_p[d] = (daily_port_val[d] - inflow - daily_port_val[d - 1]) / daily_port_val[d - 1]
        if d > start_day and daily_bench_val[d - 1] > 0:
            binflow = monthly_sip if sip_flag[d] else 0.0
            daily_ret_b[d] = (daily_bench_val[d] - binflow - daily_bench_val[d - 1]) / daily_bench_val[d - 1]

    # Final liquidation mark (no actual trade)
    final_wealth = daily_port_val[end_day - 1] if end_day > start_day else starting_wealth
    final_bench = daily_bench_val[end_day - 1] if end_day > start_day else starting_wealth

    # Append terminal cash-flow for IRR
    if cf_cnt < MAX_CF:
        cf_days[cf_cnt] = end_day
        cf_amounts[cf_cnt] = final_wealth
        cf_cnt += 1
    if bench_cf_cnt < MAX_CF:
        bench_cf_days[bench_cf_cnt] = end_day
        bench_cf_amounts[bench_cf_cnt] = final_bench
        bench_cf_cnt += 1

    # Simple risk stats from daily returns
    rets = daily_ret_p[start_day + 1:end_day]
    valid = rets[~np.isnan(rets)]
    if len(valid) > 5:
        mean_r = np.mean(valid)
        std_r = np.std(valid)
        downside = valid[valid < 0]
        down_std = np.std(downside) if len(downside) > 1 else std_r
        sharpe = (mean_r / std_r) * math.sqrt(365.0) if std_r > 0 else 0.0
        sortino = (mean_r / down_std) * math.sqrt(365.0) if down_std > 0 else 0.0
        # max DD
        peak = daily_port_val[start_day]
        max_dd = 0.0
        for i in range(start_day, end_day):
            v = daily_port_val[i]
            if v > peak:
                peak = v
            if peak > 0:
                dd = (peak - v) / peak
                if dd > max_dd:
                    max_dd = dd
    else:
        sharpe = sortino = max_dd = 0.0
        mean_r = 0.0

    avg_bars = (total_bars / trades) if trades > 0 else 0.0
    avg_n_open = (sum_n_open_report / cum_n_open_days) if cum_n_open_days > 0 else 0.0
    return (final_wealth, final_bench, total_invested,
            total_wins, total_losses, trades,
            winning_trades, losing_trades, avg_bars,
            sharpe, sortino, max_dd,
            cf_days[:cf_cnt].copy(), cf_amounts[:cf_cnt].copy(),
            bench_cf_days[:bench_cf_cnt].copy(), bench_cf_amounts[:bench_cf_cnt].copy(),
            daily_port_val.copy(), daily_bench_val.copy(),
            avg_n_open, max_n_open_report)


# =============================================================================
# 6. BUILD SIGNAL ARRAYS FROM FROZEN WINNER
# =============================================================================

def build_signal_arrays(signal_p, opens, closes, atr, adx, years_arr, n_stocks):
    """Translate the signal-engine winner into boolean entry/exit matrices
    that the portfolio simulator can consume without any further optimisation."""
    n_days = closes.shape[0]
    entry_signal = np.zeros((n_days, n_stocks), dtype=np.bool_)
    real_exit = np.zeros((n_days, n_stocks), dtype=np.bool_)
    mirror_exit = np.zeros((n_days, n_stocks), dtype=np.bool_)
    wl_rsi_exit = np.zeros((n_days, n_stocks), dtype=np.bool_)
    btc_bearish = np.zeros(n_days, dtype=np.bool_)

    # BTC regime
    btc_ma = get_ma_cached(closes, 0, signal_p["btc_ma_len"], signal_p["btc_ma_type"])
    btc_close = closes[:, 0]
    for d in range(1, n_days):
        if not np.isnan(btc_ma[d]) and btc_close[d] < btc_ma[d]:
            btc_bearish[d] = True

    et = signal_p["entry_type"]
    xt = signal_p["exit_type"]

    for s in range(n_stocks):
        # ---- entry ----
        if et == ENTRY_MA_BREAKOUT:
            ma = get_ma_cached(closes, s, signal_p["entry_ma_len"], signal_p["entry_ma_type"])
            for d in range(1, n_days):
                if (not np.isnan(ma[d - 1]) and not np.isnan(ma[d])
                        and closes[d - 1, s] <= ma[d - 1]
                        and closes[d, s] > ma[d]):
                    entry_signal[d, s] = True
                # mirror for watchlist removal
                if (not np.isnan(ma[d - 1]) and not np.isnan(ma[d])
                        and closes[d - 1, s] >= ma[d - 1]
                        and closes[d, s] < ma[d]):
                    mirror_exit[d, s] = True
        elif et == ENTRY_RSI_XOVER:
            fast = get_smoothed_rsi(closes, s, signal_p["rsi_f_len"], signal_p["rsi_f_smt"])
            slow = get_smoothed_rsi(closes, s, signal_p["rsi_s_len"], signal_p["rsi_s_smt"])
            trend = None
            if signal_p.get("use_rsi_trend_filter"):
                trend = get_ma_cached(closes, s, signal_p["rsi_trend_ma_len"],
                                      signal_p["rsi_trend_ma_type"])
            for d in range(1, n_days):
                if (not np.isnan(fast[d - 1]) and not np.isnan(slow[d - 1])
                        and not np.isnan(fast[d]) and not np.isnan(slow[d])
                        and fast[d - 1] <= slow[d - 1] and fast[d] > slow[d]):
                    ok = True
                    if trend is not None and not (closes[d, s] > trend[d]):
                        ok = False
                    if ok:
                        entry_signal[d, s] = True
                # mirror
                if (not np.isnan(fast[d - 1]) and not np.isnan(slow[d - 1])
                        and not np.isnan(fast[d]) and not np.isnan(slow[d])
                        and fast[d - 1] >= slow[d - 1] and fast[d] < slow[d]):
                    mirror_exit[d, s] = True
        elif et == ENTRY_MA_XOVER:
            short = get_ma_cached(closes, s, signal_p["xover_short_len"],
                                  signal_p["xover_short_type"])
            long_ = get_ma_cached(closes, s, signal_p["xover_long_len"],
                                  signal_p["xover_long_type"])
            for d in range(1, n_days):
                if (not np.isnan(short[d - 1]) and not np.isnan(long_[d - 1])
                        and not np.isnan(short[d]) and not np.isnan(long_[d])
                        and short[d - 1] <= long_[d - 1] and short[d] > long_[d]):
                    entry_signal[d, s] = True
                if (not np.isnan(short[d - 1]) and not np.isnan(long_[d - 1])
                        and not np.isnan(short[d]) and not np.isnan(long_[d])
                        and short[d - 1] >= long_[d - 1] and short[d] < long_[d]):
                    mirror_exit[d, s] = True

        # BTC entry gate
        if signal_p.get("use_btc_entry_gate"):
            for d in range(n_days):
                if btc_bearish[d]:
                    entry_signal[d, s] = False

        # ADX soft filter (kept as a frozen value from the signal winner;
        # portfolio can still add an extra ADX floor via Optuna if desired)
        adx_th = signal_p.get("adx_thresh", 0.0)
        if adx_th > 0:
            for d in range(n_days):
                if not np.isnan(adx[d, s]) and adx[d, s] < adx_th:
                    entry_signal[d, s] = False

        # ---- real-position exit (frozen) ----
        # EXIT_HYBRID / EXIT_PCT_TRAIL / EXIT_ATR_TRAIL genuinely need
        # per-position state (high-since-entry, half-sold, a trailing stop
        # anchored to THIS trade's entry) that doesn't exist in a per-day
        # boolean matrix keyed only on (day, coin) -- so real_exit is
        # deliberately left False for these three here. They are evaluated
        # LIVE, per open position, inside simulate_portfolio_fixed_signal
        # (see its "Update highs / evaluate real-position exits" block),
        # which mirrors the signal engine's simulate_signal_trades exit
        # state machine exactly. The BTC exit-override block below still
        # applies to all six exit types uniformly, so real_exit[d,s] can
        # still be True here for these three when BTC turns bearish -- the
        # live loop ORs that in on top of its own price-based exit check.
        if xt in (EXIT_HYBRID, EXIT_PCT_TRAIL, EXIT_ATR_TRAIL):
            pass
        elif xt == EXIT_MA_CROSSUNDER:
            ma = get_ma_cached(closes, s, signal_p["exit_ma_len"], signal_p["exit_ma_type"])
            for d in range(1, n_days):
                if (not np.isnan(ma[d - 1]) and not np.isnan(ma[d])
                        and closes[d - 1, s] >= ma[d - 1]
                        and closes[d, s] < ma[d]):
                    real_exit[d, s] = True
        elif xt == EXIT_RSI_CROSSUNDER:
            fast = get_smoothed_rsi(closes, s, signal_p["exit_rsi_f_len"],
                                    signal_p["exit_rsi_f_smt"])
            slow = get_smoothed_rsi(closes, s, signal_p["exit_rsi_s_len"],
                                    signal_p["exit_rsi_s_smt"])
            for d in range(1, n_days):
                if (not np.isnan(fast[d - 1]) and not np.isnan(slow[d - 1])
                        and not np.isnan(fast[d]) and not np.isnan(slow[d])
                        and fast[d - 1] >= slow[d - 1] and fast[d] < slow[d]):
                    real_exit[d, s] = True
        elif xt == EXIT_MA_XOVER_EXIT:
            short = get_ma_cached(closes, s, signal_p["exit_xover_short_len"],
                                  signal_p["exit_xover_short_type"])
            long_ = get_ma_cached(closes, s, signal_p["exit_xover_long_len"],
                                  signal_p["exit_xover_long_type"])
            for d in range(1, n_days):
                if (not np.isnan(short[d - 1]) and not np.isnan(long_[d - 1])
                        and not np.isnan(short[d]) and not np.isnan(long_[d])
                        and short[d - 1] >= long_[d - 1] and short[d] < long_[d]):
                    real_exit[d, s] = True

        # BTC exit override
        if signal_p.get("use_btc_exit_override"):
            for d in range(n_days):
                if btc_bearish[d]:
                    real_exit[d, s] = True

        # Dedicated RSI exit for watchlist removal family 3
        # (independent lengths – will be filled by Optuna when that family is chosen;
        #  here we pre-allocate a neutral false matrix; evaluate_params will overwrite)
        pass

    return entry_signal, real_exit, mirror_exit, wl_rsi_exit, btc_bearish


# =============================================================================
# 7. SCORING (portfolio wealth-curve)
# =============================================================================

def money_weighted_annual_return(cf_days, cf_amounts):
    """Newton IRR on irregular cash-flows (day indices)."""
    if len(cf_days) < 2:
        return 0.0, False
    days = np.array(cf_days, dtype=np.float64)
    amts = np.array(cf_amounts, dtype=np.float64)
    t0 = days[0]
    years = (days - t0) / CRYPTO_DAYS_PER_YEAR

    def npv(r):
        return np.sum(amts / (1.0 + r) ** years)

    def dnpv(r):
        return np.sum(-years * amts / (1.0 + r) ** (years + 1))

    r = 0.1
    for _ in range(50):
        v = npv(r)
        dv = dnpv(r)
        if abs(dv) < 1e-12:
            break
        r_new = r - v / dv
        if abs(r_new - r) < 1e-8:
            r = r_new
            break
        r = r_new
        if r <= -0.999:
            r = -0.99
    return float(r), True


def compute_score_portfolio(metrics, yearly, is_oos=False):
    """Returns (score, reason). reason is 'pass' or a short string naming
    the first gate that failed and the actual-vs-threshold numbers, so
    callers can log *why* a trial was rejected instead of just the -999."""
    trades = metrics.get("trades", 0)
    if trades < MIN_TRADES_GATE:
        return -999.0, f"trades {trades} < MIN_TRADES_GATE {MIN_TRADES_GATE}"
    months = metrics.get("month_cnt", 0)
    months_needed = MIN_MONTHS_GATE if not is_oos else max(6, int(MIN_MONTHS_GATE * WFO_OOS_PCT / WFO_IS_PCT))
    if months < months_needed:
        return -999.0, f"months {months} < required {months_needed}"
    max_dd = abs(metrics.get("max_dd", 1.0))
    if max_dd > MAX_DD_GATE:
        return -999.0, f"max_dd {max_dd:.1%} > MAX_DD_GATE {MAX_DD_GATE:.1%}"
    pf = metrics.get("pf", 0.0)
    if pf < 1.10:
        return -999.0, f"profit_factor {pf:.2f} < 1.10"
    roi = metrics.get("roi", 0.0)
    if roi <= 0:
        return -999.0, f"roi {roi:.1%} <= 0"
    wt = metrics.get("winning_trades", 0)
    lt = metrics.get("losing_trades", 0)
    wr = wt / (wt + lt) if (wt + lt) > 0 else 0.0
    if wr < MIN_WIN_RATE_GATE:
        return -999.0, f"win_rate {wr:.1%} < MIN_WIN_RATE_GATE {MIN_WIN_RATE_GATE:.1%}"

    # Concentration hard gate
    if yearly:
        total_profit = sum(y.get("nominal_profit", 0) for y in yearly)
        if total_profit > 0:
            max_share = max(y.get("nominal_profit", 0) / total_profit for y in yearly)
            if max_share > CONCENTRATION_GATE:
                return -999.0, f"concentration {max_share:.1%} > CONCENTRATION_GATE {CONCENTRATION_GATE:.1%} (one year dominates total profit)"

    ann = metrics.get("annual_return", 0.0)
    calmar = (ann / max_dd) if max_dd > 0 else 0.0
    sortino = min(metrics.get("sortino", 0.0), 4.0)
    ir = metrics.get("ir", 0.0)
    month_cnt = max(1, metrics.get("month_cnt", 1))
    ir_scaled = max(0.0, ir) * math.sqrt(month_cnt / 12.0)

    avg_runup = metrics.get("avg_runup", 0.0)
    avg_loss = abs(metrics.get("avg_loss", 0.01))
    ev = wr * avg_runup - (1.0 - wr) * avg_loss
    ev_in_r = ev / avg_loss if avg_loss > 0 else 0.0
    wr_bonus = max(0.0, wr - 0.50) * 4.0

    # Consistency & regime (simplified – full yearly already filtered)
    consistency = 0.0
    regime = 0.0
    if yearly:
        scored = [y for y in yearly if y.get("coverage_frac", 1) >= MIN_YEAR_COVERAGE_FOR_SCORING]
        if scored:
            yrs = [y["year"] for y in scored]
            mn, mx = min(yrs), max(yrs)
            weights = [_recency_weight(y["year"], mn, mx) * y.get("coverage_frac", 1) for y in scored]
            wsum = sum(weights) or 1.0
            irrs = [y["port_irr"] for y in scored]
            consistency = sum(w * i for w, i in zip(weights, irrs)) / wsum
            # regime
            bull_cap, bear_def, nb, nbe = 0.0, 0.0, 0, 0
            for y in scored:
                if y["bench_irr"] > BULL_YEAR_BTC_THRESHOLD:
                    bull_cap += min(2.0, y["port_irr"] / max(y["bench_irr"], 1e-6))
                    nb += 1
                elif y["bench_irr"] < BEAR_YEAR_BTC_THRESHOLD:
                    bear_def += max(-1.0, y["port_irr"] - y["bench_irr"])
                    nbe += 1
            if nb:
                bull_cap /= nb
            if nbe:
                bear_def /= nbe
            regime = 0.5 * bull_cap + 0.5 * bear_def

    score = (calmar * W_CALMAR + sortino * W_SORTINO + ir_scaled * W_IR
             + ev_in_r * W_EV + wr_bonus * W_WR_BONUS
             + consistency * W_CONSISTENCY + regime * W_REGIME)
    # soft confidence
    score *= min(1.0, math.sqrt(trades / max(MIN_TRADES_GATE, 1)))
    return score, "pass"


# =============================================================================
# 8. EVALUATE ONE WATCHLIST / CAPITAL CONFIG AROUND THE FROZEN SIGNAL
# =============================================================================

def evaluate_watchlist_params(wl_p, signal_p, opens, closes, atr, adx,
                              months, years_arr, eligible_mask, master_dates,
                              quote_vol, start_day, end_day, sip_flag,
                              is_oos=False, starting_wealth=0.0):
    n_days, n_stocks = closes.shape

    # Build frozen signal arrays (cached outside in a real run; here per-call for clarity)
    entry_sig, real_ex, mirror_ex, _, btc_bear = build_signal_arrays(
        signal_p, opens, closes, atr, adx, years_arr, n_stocks)

    # Optional dedicated RSI exit for watchlist removal
    wl_rsi_ex = np.zeros((n_days, n_stocks), dtype=np.bool_)
    if wl_p.get("wl_remove_method") == WL_REMOVE_RSI:
        for s in range(n_stocks):
            fast = get_smoothed_rsi(closes, s, wl_p["wl_rsi_f_len"], wl_p["wl_rsi_f_smt"])
            slow = get_smoothed_rsi(closes, s, wl_p["wl_rsi_s_len"], wl_p["wl_rsi_s_smt"])
            for d in range(1, n_days):
                if (not np.isnan(fast[d-1]) and not np.isnan(slow[d-1])
                        and not np.isnan(fast[d]) and not np.isnan(slow[d])
                        and fast[d-1] >= slow[d-1] and fast[d] < slow[d]):
                    wl_rsi_ex[d, s] = True

    # Momentum series for ranking (close / 50-SMA)
    momentum_ma = np.zeros((n_days, n_stocks))
    for s in range(n_stocks):
        ma50 = get_ma_cached(closes, s, 50, 0)
        for d in range(n_days):
            if ma50[d] > 0 and not np.isnan(ma50[d]) and closes[d, s] > 0:
                momentum_ma[d, s] = closes[d, s] / ma50[d]

    # Volume rank 0-1 per day
    volume_rank = np.zeros((n_days, n_stocks))
    for d in range(n_days):
        vols = quote_vol[d, :].copy()
        vols[np.isnan(vols)] = 0.0
        order = np.argsort(vols)
        for rank, s in enumerate(order):
            volume_rank[d, s] = rank / max(1, n_stocks - 1)

    result = simulate_portfolio_fixed_signal(
        opens, closes, atr, adx,
        entry_sig, real_ex, mirror_ex, wl_rsi_ex, btc_bear,
        wl_p["wl_rank_method"],
        wl_p["wl_remove_method"],
        wl_p.get("wl_max_age_days", 30),
        wl_p.get("wl_rank_drop_thresh", 0.0),
        wl_p["max_concurrent_positions"],
        wl_p.get("allow_portfolio_pyramid", False),
        wl_p.get("min_ticket", MIN_TICKET_SIZE),
        signal_p["exit_type"],
        signal_p.get("sl_mult", 4.0), signal_p.get("tp_mult", 30.0),
        signal_p.get("trail_mult", 6.0), signal_p.get("trail_pct", 15.0),
        signal_p.get("exit_atr_mult", 3.0),
        wl_p.get("wl_entry_funding_method", 0),
        wl_p.get("wl_exit_proceeds_method", 0),
        wl_p.get("wl_sizing_denom_method", 0),
        wl_p.get("wl_concurrency_buffer", 1.0),
        eligible_mask, months, sip_flag,
        int(start_day), int(end_day),
        float(starting_wealth),
        BUY_COST_PCT, SELL_COST_PCT, ANNUAL_CASH_YIELD,
        MONTHLY_SIP,
        wl_p.get("w_mom", 0.4), wl_p.get("w_adx", 0.3), wl_p.get("w_vol", 0.3),
        momentum_ma, volume_rank,
    )

    (f_wealth, f_bench, t_invested, wins, losses, trades,
     wt, lt, avg_bars, sharpe, sortino, max_dd,
     cf_days, cf_amounts, bcf_days, bcf_amounts,
     daily_port, daily_bench, avg_n_open, max_n_open) = result

    if t_invested <= 0 and starting_wealth <= 0:
        return -999.0, {}

    capital_base = starting_wealth + t_invested
    roi = (f_wealth - capital_base) / capital_base if capital_base > 0 else 0.0
    bench_capital = sum(-a for a in bcf_amounts if a < 0)
    bench_roi = (f_bench - bench_capital) / bench_capital if bench_capital > 0 else 0.0

    ann, _ = money_weighted_annual_return(list(cf_days), list(cf_amounts))
    bench_ann, _ = money_weighted_annual_return(list(bcf_days), list(bcf_amounts))

    pf = (wins / losses) if losses > 0 else (999.0 if wins > 0 else 0.0)
    wr = wt / (wt + lt) if (wt + lt) > 0 else 0.0

    # Rough yearly breakdown for concentration / consistency
    yearly = []
    if len(daily_port) > 10:
        for yr in sorted(set(int(y) for y in years_arr[start_day:end_day])):
            mask = (years_arr == yr) & (np.arange(n_days) >= start_day) & (np.arange(n_days) < end_day)
            idxs = np.where(mask)[0]
            if len(idxs) < 30:
                continue
            p0, p1 = daily_port[idxs[0]], daily_port[idxs[-1]]
            b0, b1 = daily_bench[idxs[0]], daily_bench[idxs[-1]]
            if p0 <= 0:
                continue
            port_ret = (p1 - p0) / p0
            bench_ret = (b1 - b0) / b0 if b0 > 0 else 0.0
            coverage = len(idxs) / 365.0
            yearly.append({
                "year": yr,
                "port_irr": port_ret,          # not fully annualised; ok for relative use
                "bench_irr": bench_ret,
                "nominal_profit": p1 - p0,
                "coverage_frac": min(1.0, coverage),
            })

    metrics = {
        "roi": roi, "bench_roi": bench_roi, "alpha": roi - bench_roi,
        "pf": pf, "wealth": f_wealth, "bench_wealth": f_bench,
        "trades": trades, "winning_trades": wt, "losing_trades": lt,
        "sharpe": sharpe, "sortino": sortino, "ir": 0.0,  # simplified
        "max_dd": max_dd, "avg_bars": avg_bars,
        "t_invested": t_invested, "starting_wealth": starting_wealth,
        "annual_return": ann, "bench_annual_return": bench_ann,
        "month_cnt": max(1, int((end_day - start_day) / 30)),
        "avg_runup": 0.05, "avg_loss": 0.03,  # placeholders; full trade log would fill
        "yearly": yearly,
        "avg_n_open": avg_n_open, "max_n_open": max_n_open,
    }
    score, gate_reason = compute_score_portfolio(metrics, yearly, is_oos=is_oos)
    metrics["gate_reason"] = gate_reason
    return score, metrics


# =============================================================================
# 9. OPTUNA SEARCH SPACE (watchlist + capital only)
# =============================================================================

def _suggest_watchlist_params(trial):
    p = {
        "wl_rank_method": trial.suggest_categorical(
            "wl_rank_method",
            [WL_RANK_OLDEST, WL_RANK_MOMENTUM, WL_RANK_ADX,
             WL_RANK_VOLUME, WL_RANK_COMPOSITE]),
        "wl_remove_method": trial.suggest_categorical(
            "wl_remove_method",
            [WL_REMOVE_SAME_EXIT, WL_REMOVE_AGE, WL_REMOVE_MIRROR,
             WL_REMOVE_RSI, WL_REMOVE_BTC, WL_REMOVE_RANK_DROP]),
        "max_concurrent_positions": trial.suggest_int("max_concurrent_positions", 3, 12),
        "allow_portfolio_pyramid": trial.suggest_categorical("allow_portfolio_pyramid", [False, True]),
        "min_ticket_mult": trial.suggest_float("min_ticket_mult", 0.5, 2.0),
    }
    p["min_ticket"] = MONTHLY_SIP * p["min_ticket_mult"]

    if p["wl_remove_method"] == WL_REMOVE_AGE:
        p["wl_max_age_days"] = trial.suggest_int("wl_max_age_days", 5, 90)
    else:
        p["wl_max_age_days"] = 30

    if p["wl_remove_method"] == WL_REMOVE_RSI:
        p["wl_rsi_f_len"] = trial.suggest_int("wl_rsi_f_len", 10, 60)
        p["wl_rsi_f_smt"] = trial.suggest_int("wl_rsi_f_smt", 5, 40)
        p["wl_rsi_s_len"] = trial.suggest_int("wl_rsi_s_len", 15, 80)
        p["wl_rsi_s_smt"] = trial.suggest_int("wl_rsi_s_smt", 5, 40)
    else:
        p["wl_rsi_f_len"] = p["wl_rsi_f_smt"] = p["wl_rsi_s_len"] = p["wl_rsi_s_smt"] = 14

    if p["wl_remove_method"] == WL_REMOVE_RANK_DROP:
        p["wl_rank_drop_thresh"] = trial.suggest_float("wl_rank_drop_thresh", -1.0, 1.0)
    else:
        p["wl_rank_drop_thresh"] = 0.0

    if p["wl_rank_method"] == WL_RANK_COMPOSITE:
        # simplex weights
        raw = [trial.suggest_float(f"w_{k}", 0.05, 1.0) for k in ("mom", "adx", "vol")]
        s = sum(raw)
        p["w_mom"], p["w_adx"], p["w_vol"] = raw[0]/s, raw[1]/s, raw[2]/s
    else:
        p["w_mom"] = p["w_adx"] = p["w_vol"] = 0.33

    # -- Capital-deployment forks (independent -- Optuna explores all
    #    combinations, including the original cash-pool-only / leave-liquid
    #    / max-concurrent-positions setup as one of them) --
    #
    # wl_entry_funding_method:
    #   0 = cash-pool-only (default/previous behaviour) -- a new position is
    #       only ever funded from idle cash; existing positions are untouched.
    #   1 = trim-overweight-only -- use idle cash first; if short of the new
    #       equal-weight target, sell down (only) the existing positions
    #       currently above that target and use the proceeds. Positions
    #       already at/below target are left alone.
    #   2 = full bidirectional rebalance -- every open position (existing +
    #       new) is pushed toward total_portfolio_value / effective_slots,
    #       buying underweight ones and selling overweight ones as needed.
    p["wl_entry_funding_method"] = trial.suggest_categorical(
        "wl_entry_funding_method", [0, 1, 2])

    # wl_exit_proceeds_method:
    #   0 = leave freed capital in cash_pool for the next new signal (previous
    #       behaviour).
    #   1 = immediately buy freed_capital / n_open_remaining more of every
    #       still-open position (a real buy, real transaction cost).
    p["wl_exit_proceeds_method"] = trial.suggest_categorical(
        "wl_exit_proceeds_method", [0, 1])

    # wl_sizing_denom_method -- what "effective number of slots" the
    # equal-weight target is divided by:
    #   0 = max_concurrent_positions (the hard cap above) -- conservative,
    #       reserves capital for future diversification.
    #   1 = causal (expanding-window, no-lookahead) running average of
    #       realized concurrent-open-position count, plus a searched buffer.
    #   2 = current open count only -- always fully deploy across exactly
    #       however many positions are open right now, no capital reserve
    #       ("invest everything, split evenly among whatever's open").
    p["wl_sizing_denom_method"] = trial.suggest_categorical(
        "wl_sizing_denom_method", [0, 1, 2])
    if p["wl_sizing_denom_method"] == 1:
        p["wl_concurrency_buffer"] = trial.suggest_float("wl_concurrency_buffer", 0.5, 3.0)
    else:
        p["wl_concurrency_buffer"] = 1.0

    return p


def save_winner(oos_score, is_score, signal_p, wl_p, filename=BEST_PARAMS_FILE, tier="unrated"):
    data = {
        "engine_version": ENGINE_VERSION,
        "oos_score": float(oos_score),
        "is_score": float(is_score),
        "robustness_ratio": float(oos_score / is_score) if is_score > 0 else 0.0,
        "robustness_tier": tier,
        "signal_params": {k: (float(v) if isinstance(v, (float, np.floating)) else
                              int(v) if isinstance(v, (int, np.integer)) else bool(v))
                          for k, v in signal_p.items()},
        "watchlist_params": {k: (float(v) if isinstance(v, (float, np.floating)) else
                                 int(v) if isinstance(v, (int, np.integer)) else bool(v))
                             for k, v in wl_p.items()},
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved → {filename}")


# =============================================================================
# 10. MAIN OPTIMIZER
# =============================================================================

def run_optimization():
    signal_p = load_signal_winner(SIGNAL_WINNER_FILE)

    tickers = fetch_top_universe()
    (opens, closes, atr, adx, months, years_arr,
     stock_names, eligible, master_dates, quote_vol) = prepare_matrix_data(tickers)

    clear_caches()
    n_days = closes.shape[0]
    is_end = int(n_days * WFO_IS_PCT)
    inner_split = is_end - int(is_end * INNER_VAL_PCT)

    sip_flag = build_sip_schedule(master_dates, mode="month_start")

    print(f"\nMatrix: {n_days} days × {closes.shape[1]} coins")
    print(f"Inner-train: 0–{inner_split}   Inner-val: {inner_split}–{is_end}")
    print(f"True OOS (never touched): {is_end}–{n_days}")
    print("\nSignal is FROZEN. Optuna searches only watchlist ranking,")
    print("removal rules, max positions, and sizing.\n")

    # ---- One-time upfront report: how much raw activity does the frozen
    #      signal actually generate in the inner-train window? This is the
    #      ceiling on how many real trades ANY watchlist/capital config can
    #      possibly produce -- if this number is already low, a wave of
    #      early -999 trials is expected (MIN_TRADES_GATE not cleared) and
    #      is not evidence of a bug; if it's high and trials still reject
    #      on trades, that points at the watchlist/promotion layer instead.
    print("=" * 70)
    print("SIGNAL ACTIVITY CHECK (inner-train window, before any watchlist")
    print("ranking/removal/sizing is applied -- this is the raw ceiling)")
    print("=" * 70)
    diag_entry_sig, diag_real_ex, _, _, diag_btc_bear = build_signal_arrays(
        signal_p, opens, closes, atr, adx, years_arr, closes.shape[1])
    # slice the OUTPUT (not the inputs) to the inner-train window -- using
    # sliced inputs here would populate get_ma_cached/get_raw_rsi_cached's
    # caches (keyed only on stock_idx/period/type, not on the array itself)
    # with results computed from a shorter array, and every later call in
    # this same process that reuses the same key against the FULL-length
    # opens/closes/atr/adx would then silently read back that stale,
    # too-short cached array -- exactly the kind of bug that produces a
    # confusing IndexError deep inside a later trial.
    diag_entry_sig_window = diag_entry_sig[:inner_split]
    diag_real_ex_window = diag_real_ex[:inner_split]
    raw_entries = int(diag_entry_sig_window.sum())
    raw_exits = int(diag_real_ex_window.sum())
    per_coin_entries = diag_entry_sig_window.sum(axis=0)
    n_coins_with_any_signal = int((per_coin_entries > 0).sum())
    print(f"  Raw entry signals fired: {raw_entries}  (across {n_coins_with_any_signal}/"
          f"{closes.shape[1]} coins, {inner_split} days)")
    print(f"  Raw real-exit signals fired: {raw_exits}")
    if closes.shape[1] > 1:
        top_coins = np.argsort(-per_coin_entries)[:5]
        breakdown = ", ".join(f"{stock_names[s]}={int(per_coin_entries[s])}"
                               for s in top_coins if per_coin_entries[s] > 0)
        if breakdown:
            print(f"  Busiest coins: {breakdown}")
    print(f"  MIN_TRADES_GATE requires {MIN_TRADES_GATE} closed trades in this window.")
    if raw_entries == 0:
        print("  *** ZERO raw entry signals -- the frozen signal never fires at all")
        print("      on this universe/window. No watchlist config can produce a trade.")
        print("      Check signal_p's entry_type/thresholds and BTC gate, or the")
        print("      universe/date range, before spending Optuna budget on this.")
    elif raw_entries < MIN_TRADES_GATE:
        print(f"  *** Only {raw_entries} raw entries exist, below MIN_TRADES_GATE itself")
        print("      -- even a config that promotes and closes every single one")
        print("      cannot clear the trades gate. Expect -999 across the board")
        print("      until this changes (different signal, wider window, or lower")
        print("      MIN_TRADES_GATE).")
    else:
        headroom = raw_entries / MIN_TRADES_GATE
        print(f"  {headroom:.1f}x headroom over the gate -- plausible for some configs")
        print("  to clear it, but tight rank/removal/max_concurrent_positions choices")
        print("  can still starve most signals before they're ever promoted.")
    print("=" * 70 + "\n")

    best_is_score = -999999.0
    best_wl_params = None
    all_trial_scores = []
    gate_tally = {}
    gate_tally_lock = threading.Lock()
    trial_counter = {"n": 0}
    VERBOSE_TRIAL_LIMIT = 15     # print full detail for the first N trials
    TALLY_EVERY = 25             # print a rejection-reason histogram every N trials

    def _record_and_log(trial_number, wl_p, train_score, train_m, val_score, val_m, final_score):
        with gate_tally_lock:
            trial_counter["n"] += 1
            n = trial_counter["n"]
            train_reason = (train_m or {}).get("gate_reason", "?")
            if train_reason != "pass":
                key = "train:" + train_reason.split(" ")[0]
            elif val_m is not None and val_m.get("gate_reason", "pass") != "pass":
                key = "val:" + val_m["gate_reason"].split(" ")[0]
            else:
                key = "pass"
            gate_tally[key] = gate_tally.get(key, 0) + 1

            verbose = n <= VERBOSE_TRIAL_LIMIT
            if verbose:
                fork_desc = (f"funding={wl_p.get('wl_entry_funding_method')} "
                             f"exit_proceeds={wl_p.get('wl_exit_proceeds_method')} "
                             f"sizing_denom={wl_p.get('wl_sizing_denom_method')} "
                             f"rank={wl_p.get('wl_rank_method')} remove={wl_p.get('wl_remove_method')} "
                             f"max_pos={wl_p.get('max_concurrent_positions')} "
                             f"min_ticket={wl_p.get('min_ticket'):.0f}")
                tm = train_m or {}
                print(f"[trial {trial_number}] {fork_desc}")
                print(f"    TRAIN: trades={tm.get('trades',0)} avg_n_open={tm.get('avg_n_open',0):.2f} "
                      f"max_n_open={tm.get('max_n_open',0)} max_dd={abs(tm.get('max_dd',0)):.1%} "
                      f"pf={tm.get('pf',0):.2f} roi={tm.get('roi',0):.1%} "
                      f"wr={(tm.get('winning_trades',0)/max(1,tm.get('winning_trades',0)+tm.get('losing_trades',0))):.1%} "
                      f"months={tm.get('month_cnt',0)}  -> score={train_score:.3f} [{tm.get('gate_reason','?')}]")
                if val_m is not None:
                    vm = val_m
                    print(f"    VAL:   trades={vm.get('trades',0)} avg_n_open={vm.get('avg_n_open',0):.2f} "
                          f"max_n_open={vm.get('max_n_open',0)} max_dd={abs(vm.get('max_dd',0)):.1%} "
                          f"pf={vm.get('pf',0):.2f} roi={vm.get('roi',0):.1%} "
                          f"months={vm.get('month_cnt',0)}  -> score={val_score:.3f} [{vm.get('gate_reason','?')}]")
                print(f"    => final trial score: {final_score:.3f}")

            if n % TALLY_EVERY == 0:
                total = sum(gate_tally.values())
                summary = ", ".join(f"{k}={v} ({v/total:.0%})"
                                     for k, v in sorted(gate_tally.items(), key=lambda kv: -kv[1]))
                print(f"\n--- after {n} trials, rejection-reason tally: {summary} ---\n")


    if OPTUNA_AVAILABLE:
        def objective(trial):
            wl_p = _suggest_watchlist_params(trial)
            train_score, train_m = evaluate_watchlist_params(
                wl_p, signal_p, opens, closes, atr, adx,
                months, years_arr, eligible, master_dates, quote_vol,
                0, inner_split, sip_flag, is_oos=False)
            if train_score <= -900:
                _record_and_log(trial.number, wl_p, train_score, train_m, None, None, -999.0)
                return -999.0
            val_score, val_m = evaluate_watchlist_params(
                wl_p, signal_p, opens, closes, atr, adx,
                months, years_arr, eligible, master_dates, quote_vol,
                inner_split, is_end, sip_flag, is_oos=True)
            if val_score <= -900:
                final = train_score * 0.05 - 5.0   # heavy fallback, never beats dual-pass
                _record_and_log(trial.number, wl_p, train_score, train_m, val_score, val_m, final)
                return final
            final = min(train_score, val_score)
            _record_and_log(trial.number, wl_p, train_score, train_m, val_score, val_m, final)
            return final

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=int(time.time()) % 100000,
                n_startup_trials=200,
                multivariate=True, group=True),
        )

        champion_lock = threading.Lock()

        def callback(study, trial):
            nonlocal best_is_score, best_wl_params
            if trial.value is None or trial.value <= -900:
                return
            with champion_lock:
                if trial.value <= best_is_score:
                    return
                wl_p = _suggest_watchlist_params(trial)  # rebuild full dict
                # neighbourhood check on a couple of key knobs
                best_is_score = trial.value
                best_wl_params = wl_p
                print(f"\n[trial {trial.number}] dual-pass score {trial.value:.4f}")
                print(f"  rank={wl_p['wl_rank_method']}  remove={wl_p['wl_remove_method']}  "
                      f"max_pos={wl_p['max_concurrent_positions']}")
                save_winner(0.0, trial.value, signal_p, wl_p,
                            filename=INTERMEDIATE_FILE, tier="IS-champion")

        study.optimize(objective, n_trials=BAYESIAN_TRIALS,
                       callbacks=[callback], show_progress_bar=True,
                       n_jobs=OPTUNA_N_JOBS)

    if best_wl_params is None:
        print("\nNo watchlist configuration cleared the gates.")
        return

    # ---- Final WFO on true OOS ----
    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION (true OOS never seen during search)")
    print("=" * 60)
    is_score, is_m = evaluate_watchlist_params(
        best_wl_params, signal_p, opens, closes, atr, adx,
        months, years_arr, eligible, master_dates, quote_vol,
        0, is_end, sip_flag, is_oos=False)
    oos_score, oos_m = evaluate_watchlist_params(
        best_wl_params, signal_p, opens, closes, atr, adx,
        months, years_arr, eligible, master_dates, quote_vol,
        is_end, n_days - 1, sip_flag, is_oos=True,
        starting_wealth=is_m.get("wealth", 0.0))

    robustness = (oos_score / is_score) if is_score > 0 and oos_score > 0 else 0.0
    print(f"IS  score: {is_score:.4f}   trades={is_m.get('trades',0)}  "
          f"ROI={is_m.get('roi',0)*100:.1f}%  MaxDD={is_m.get('max_dd',0)*100:.1f}%")
    print(f"OOS score: {oos_score:.4f}   trades={oos_m.get('trades',0)}  "
          f"ROI={oos_m.get('roi',0)*100:.1f}%  MaxDD={oos_m.get('max_dd',0)*100:.1f}%")
    print(f"Robustness: {robustness:.1%}")

    if oos_score > -900 and robustness >= 0.70:
        tier = "EXCELLENT (>70%)"
    elif oos_score > -900 and robustness >= ROBUSTNESS_DEPLOY_THRESHOLD:
        tier = "ACCEPTABLE (50-70%)"
    elif oos_score > -900:
        tier = "CAUTION / POOR"
    else:
        tier = "OOS GATE FAILURE"

    save_winner(oos_score, is_score, signal_p, best_wl_params, tier=tier)
    print(f"\nFinal tier: {tier}")
    print("Watchlist + capital parameters that won:")
    print(json.dumps(best_wl_params, indent=2, default=str))


if __name__ == "__main__":
    run_optimization()