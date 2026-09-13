"""
NSE ALPHA SIGNAL ENGINE
=========================
SCRIPT 3 of 4 in the consolidated pipeline. Sibling: nse_portfolio_engine.py
(crypto has its own pair: crypto_signal_engine.py / crypto_portfolio_engine.py
-- these two pairs are DELIBERATELY independent of each other, each with its
own copy of the indicator math, so the crypto pair and the NSE pair can each
be used standalone without needing the other asset class's files present).

THE ONE QUESTION THIS FILE ANSWERS:
  "Is this entry/exit signal +EV, and consistently so across stocks and
   years, independent of capital-timing luck?"

THE QUESTION IT DELIBERATELY DOES NOT ANSWER (see nse_portfolio_engine.py):
  "How would my actual wealth have compounded running this with a real,
   shared, monthly-funded (SIP) cash pool, position queues, NSE's actual
   statutory transaction costs, and drawdown limits?"

Every trade here is scored independently in scale-free R-multiples. There
is no shared cash pool, so one stock can never "win the capital-allocation
race" and make a different stock's signal look better or worse than it
actually is -- same separation of concerns as the crypto pair, for the
same reason.

OUTPUT: a ranked SHORTLIST (nse_signal_shortlist.json) of every distinct
config that cleared every signal-quality gate, for nse_portfolio_engine.py
to load and pressure-test under real capital constraints.

WHAT'S GENUINELY NSE-SPECIFIC HERE (beyond a straight port of the crypto
signal engine's architecture):
  - POINT-IN-TIME UNIVERSE RECONSTRUCTION: parses NSE's own historical
    index inclusion/exclusion archive and rolls TODAY's Nifty 50
    constituents backward, instead of just disclosing survivorship bias
    the way the crypto engine has to (crypto has no equivalent public
    point-in-time archive to parse). Best-effort, Nifty-50-only, and
    unverified against a live download -- see fetch_and_build_pit_universe()
    for exactly what that means and how to sanity-check it before trusting
    it with real capital. Falls back to disclosed survivorship bias
    (today's constituents applied backward) if no such file is present.
  - DIVIDEND SLEEVE: a disclosed list of high-dividend/defensive names
    gets its OWN, gentler exit family (hold through routine reversals,
    exit only on a genuine structural breakdown) instead of being traded
    like a momentum name -- and is exempt from the index-regime panic
    filter on BOTH the entry and exit side, since forcing a broad
    growth-market panic rule onto a stock chosen specifically for its
    defensive character would undermine the reason it's a separate sleeve
    in the first place.
  - GENERALIZED TREND FILTER: unlike the crypto engine (where the trend/
    regime filter only ever applied to the RSI-crossover entry family),
    NSE's own lineage always paired EVERY entry style with an optional
    "Super MA" trend filter -- generalized here into `use_trend_filter`,
    checked uniformly regardless of which of the 3 entry families fired.
  - EARLY-WARNING EXIT (7th exit family, EXIT_EARLY_TREND_BREAK): price
    breaking back below its own trend filter WHILE that trend filter is
    still rising is a materially different, earlier tell than a plain
    crossunder against a level -- gets out ahead of a slower confirming
    signal instead of riding a top-forming pattern all the way down to
    it. This is intentionally NOT just "MA_CROSSUNDER against the trend
    MA" -- the slope condition is what makes it a distinct, earlier
    warning rather than a duplicate of exit family #3.

Everything else (R-multiple scoring, robustness-aware inner train/val
Optuna search, SQN/expectancy/recency-weighted yearly R, winsorization,
soft concentration penalty, cross-sectional median-stock gate, DSR
diagnostic, bounded LRU caches, latched-entry confluence, ADX filter,
pyramiding, the walk-forward final stability replay) is the SAME
asset-agnostic machinery as crypto_signal_engine.py, adapted only where
the market genuinely requires it (day-count basis stays irrelevant to
R-multiple scoring either way; only the portfolio engine's annualization
needs 252 vs 365).

CHANGELOG
  v1.0  Initial consolidated build.
"""

import math
import json
import os
import io
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
    import yfinance as yf
except ImportError:
    yf = None
    print("yfinance not found. Install with: pip install yfinance")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    print("optuna not found. Install with: pip install optuna")
    OPTUNA_AVAILABLE = False

warnings.filterwarnings('ignore')

# ==========================================================================
# 0. CONFIGURATION
# ==========================================================================

BASELINE_CONFIG_FILE = "baseline_nse_signal_config.json"
SHORTLIST_FILE        = "nse_signal_shortlist.json"     # OUTPUT -- nse_portfolio_engine.py reads this
POINT_IN_TIME_UNIVERSE_FILE = None    # set to a file built by fetch_and_build_pit_universe()
ENGINE_VERSION = "nse-signal-1.0.0"

START_DATE       = "2015-01-01"
MIN_HISTORY_DAYS = 500

BAYESIAN_TRIALS = 20_000
WFO_IS_PCT  = 0.70
WFO_OOS_PCT = 0.30

MIN_TRADES_GATE   = 50
MIN_YEARS_GATE     = 3
MIN_WIN_RATE_GATE  = 0.35   # trend systems are SUPPOSED to have a sub-50% win rate
MIN_PROFIT_FACTOR  = 1.10
MEDIAN_STOCK_R_GATE = 0.0

# -- Signal-level drawdown gate -- reject a candidate outright if ANY stock's
# own R-multiple-based equity curve would have drawn down more than this.
# Same rationale as the crypto engine's identical gate: great expectancy/SQN
# can still hide a stock that would have wiped out a third of an account on
# its own. See compute_per_coin_drawdowns()'s docstring for exactly how this
# is computed without needing any shared cash pool. --
MAX_SIGNAL_DRAWDOWN     = 0.30   # 30% -- reject if any single stock's own equity curve draws down worse than this
DRAWDOWN_RISK_PER_TRADE = 0.02   # assumed fixed-fractional risk-per-trade for this gate's equity-curve
                                   # normalization ONLY -- unrelated to the portfolio engine's real sizing

INNER_VAL_PCT        = 0.30
MIN_YEARS_GATE_INNER  = 2
MIN_TRADES_INNER_VAL  = 20
ROBUST_FALLBACK_SCALE   = 0.05
ROBUST_FALLBACK_PENALTY = 5.0

MA_LEN_MIN, MA_LEN_MAX = 20, 300
MA_CACHE_MAX_ENTRIES   = 40_000

MAX_PYRAMID_LAYERS_MIN = 1
MAX_PYRAMID_LAYERS_MAX = 4

RECENCY_WEIGHT_MIN = 0.70
RECENCY_WEIGHT_MAX = 1.00

R_WINSORIZE_CAP = 20.0

CONCENTRATION_SOFT_THRESHOLD = 0.80
CONCENTRATION_PENALTY_MULT   = 0.70

W_SQN        = 0.30
W_EXPECTANCY = 0.30
W_RECENCY    = 0.30
W_WR_BONUS   = 0.10
SQN_CAP = 6.0

NEIGHBOR_THRESHOLD = 0.80

# -- liquidity: rolling average NOTIONAL (Volume x Close, INR) floor for a
# NEW entry -- the NSE analog of the crypto engine's USD quote-volume mask --
MIN_AVG_DAILY_NOTIONAL_INR = 5_000_000
LIQUIDITY_LOOKBACK_DAYS    = 30

# -- early-warning exit (EXIT_EARLY_TREND_BREAK): fixed, non-optimized
# lookback for the trend-MA slope check, kept fixed rather than another
# free parameter per Doc8's explicit parameter-discipline principle
# (every extra free parameter eats into how much a thin fold can actually
# distinguish signal from noise) --
EARLY_EXIT_TREND_SLOPE_LOOKBACK = 20

WF_INITIAL_TRAIN_YEARS = 3
WF_MIN_FOLD_TRADES     = 8

SHORTLIST_SIZE     = 15
SHORTLIST_POOL_CAP = 60

ENTRY_MA_BREAKOUT = 0
ENTRY_RSI_XOVER   = 1
ENTRY_MA_XOVER    = 2

EXIT_HYBRID           = 0
EXIT_PCT_TRAIL        = 1
EXIT_ATR_TRAIL        = 2
EXIT_MA_CROSSUNDER    = 3
EXIT_RSI_CROSSUNDER   = 4
EXIT_MA_XOVER_EXIT    = 5
EXIT_EARLY_TREND_BREAK = 6

DIV_EXIT_PEAK_DRAWDOWN     = 0
DIV_EXIT_TREND_MA_VIOLATION = 1
DIV_EXIT_TIME_DECAY         = 2


def _recency_weight(year, min_year, max_year):
    if max_year <= min_year:
        return RECENCY_WEIGHT_MAX
    frac = (year - min_year) / (max_year - min_year)
    return RECENCY_WEIGHT_MIN + (RECENCY_WEIGHT_MAX - RECENCY_WEIGHT_MIN) * frac


def compute_per_coin_drawdowns(r_multiple, entry_days, stock_idx, risk_per_trade=DRAWDOWN_RISK_PER_TRADE):
    """Per-stock equity-curve drawdown, identical technique to the crypto
    engine's version -- see that file's docstring for the full reasoning.
    Kept the name `per_coin` for exact parity with the crypto engine (this
    function is otherwise byte-for-byte identical); NSE vocabulary calls
    the same thing `per_stock` everywhere else in this file."""
    per_coin_dd = {}
    for c in np.unique(stock_idx):
        mask = stock_idx == c
        order = np.argsort(entry_days[mask])
        r_seq = r_multiple[mask][order]
        equity = 1.0
        peak = 1.0
        worst_dd = 0.0
        for r in r_seq:
            equity *= max(0.0, 1.0 + risk_per_trade * r)
            if equity > peak:
                peak = equity
            elif peak > 0:
                dd = (equity - peak) / peak
                if dd < worst_dd:
                    worst_dd = dd
        per_coin_dd[int(c)] = abs(worst_dd)
    return per_coin_dd


# ==========================================================================
# 1. UNIVERSE DEFINITION
# ==========================================================================

DIVIDEND_KINGS_FALLBACK = {
    'ITC.NS', 'COALINDIA.NS', 'ONGC.NS', 'POWERGRID.NS', 'NTPC.NS', 'PFC.NS',
    'RECLTD.NS', 'VEDL.NS', 'GAIL.NS', 'BPCL.NS', 'IOC.NS', 'PETRONET.NS',
    'SAIL.NS', 'NHPC.NS', 'NMDC.NS', 'HINDZINC.NS', 'CASTROLIND.NS'
}

FALLBACK_UNIVERSE = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'BAJFINANCE.NS',
    'KOTAKBANK.NS', 'LT.NS', 'HCLTECH.NS', 'ASIANPAINT.NS', 'AXISBANK.NS',
    'MARUTI.NS', 'TITAN.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'WIPRO.NS',
]


def fetch_dynamic_universe():
    """Today's Nifty 50 / Next 50 / Midcap 150 constituents, fetched live
    from niftyindices.com (spoofed headers -- NSE's own archive host blocks
    plain bot requests). SURVIVORSHIP-BIAS DISCLOSURE: this is TODAY's
    constituents applied back to START_DATE, unless POINT_IN_TIME_UNIVERSE_FILE
    is set (see fetch_and_build_pit_universe() below)."""
    print(f"Fetching Nifty universe (50 / Next 50 / Midcap 150)...")
    print(f"NOTE: uses TODAY's constituents applied back to {START_DATE} unless a "
          f"point-in-time universe file is configured -- see module docstring.")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                       '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }
    urls = [
        "https://niftyindices.com/IndexConstituent/ind_nifty50list.csv",
        "https://niftyindices.com/IndexConstituent/ind_niftynext50list.csv",
        "https://niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
    ]
    tickers = set()
    for url in urls:
        try:
            req = requests.get(url, headers=headers, timeout=15)
            if req.status_code == 200:
                df = pd.read_csv(io.StringIO(req.text))
                for symbol in df['Symbol']:
                    clean = str(symbol).strip()
                    if clean and clean != 'nan':
                        tickers.add(f"{clean}.NS")
        except Exception as e:
            print(f"Error fetching {url}: {e}")

    if len(tickers) < 100:
        print("Fallback to a static major-name list (NSE likely blocked the request).")
        tickers = set(FALLBACK_UNIVERSE) | DIVIDEND_KINGS_FALLBACK
    return list(tickers)


NSE_INDEX_INCL_EXCL_URL = "https://archives.nseindia.com/content/indices/IndexInclExcl.xls"
NSE_NIFTY50_ALIASES = {"NIFTY 50", "NIFTY50", "CNX NIFTY", "S&P CNX NIFTY", "S&P CNX NIFTY 50"}


def fetch_and_build_pit_universe(output_file="nifty50_point_in_time.json"):
    """Best-effort NIFTY 50 (ONLY -- no free archive found for Next 50 /
    Midcap 150) point-in-time membership reconstruction: parses NSE's own
    inclusion/exclusion history file and rolls TODAY's live list backward,
    undoing one change at a time. This is a genuine attempt to REDUCE
    survivorship bias, not just disclose it (crypto has no equivalent
    public archive, which is why the crypto engine can only disclose).
    Written against public documentation of this file's format, NOT
    verified against a live download -- run it, read the coverage report
    it prints, and spot-check a known historical change (e.g. a well-known
    index addition/removal you remember) before trusting it with real
    capital. Point this file's POINT_IN_TIME_UNIVERSE_FILE constant at the
    output to use it."""
    print("Point-in-time universe fetcher (NIFTY 50 only, best-effort)...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        resp = requests.get(NSE_INDEX_INCL_EXCL_URL, headers=headers, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"Could not download {NSE_INDEX_INCL_EXCL_URL}: {e}")
        print("NSE occasionally moves this file -- search 'niftyindices.com IndexInclExcl' for the current location.")
        return None

    raw = io.BytesIO(resp.content)
    df = None
    for engine in ('xlrd', 'openpyxl', None):
        try:
            raw.seek(0)
            df = pd.read_excel(raw, engine=engine) if engine else pd.read_excel(raw)
            break
        except Exception:
            continue
    if df is None:
        print("Downloaded but could not parse as .xls/.xlsx -- NSE may have changed format.")
        return None

    cols = {c: str(c).strip().lower() for c in df.columns}
    def find_col(*keywords):
        for c, low in cols.items():
            if all(k in low for k in keywords):
                return c
        return None

    col_index  = find_col("index")
    col_symbol = find_col("symbol") or find_col("ticker")
    col_date   = find_col("date")
    col_action = find_col("action") or find_col("remark") or find_col("event") or find_col("effect")

    if not all([col_symbol, col_date, col_action]):
        print(f"Could not identify symbol/date/action columns among {list(df.columns)} -- inspect manually.")
        return None

    events = []
    for _, row in df.iterrows():
        try:
            if col_index is not None:
                idx_name = str(row[col_index]).strip().upper()
                if idx_name and idx_name not in NSE_NIFTY50_ALIASES:
                    continue
            action_raw = str(row[col_action]).strip().upper()
            if 'IN' in action_raw and 'OUT' not in action_raw:
                action = 'IN'
            elif 'OUT' in action_raw or 'EXCL' in action_raw or 'REMOV' in action_raw or 'DEL' in action_raw:
                action = 'OUT'
            elif 'INCL' in action_raw or 'ADD' in action_raw:
                action = 'IN'
            else:
                continue
            dt = pd.to_datetime(row[col_date], errors='coerce')
            if pd.isna(dt):
                continue
            symbol = str(row[col_symbol]).strip().upper()
            if not symbol or symbol == 'NAN':
                continue
            events.append((dt.strftime('%Y-%m-%d'), f"{symbol}.NS", action))
        except Exception:
            continue

    if not events:
        print("Parsed the file but extracted zero usable NIFTY 50 events.")
        return None

    events.sort(key=lambda e: e[0])
    print(f"Parsed {len(events)} events, spanning {events[0][0]} to {events[-1][0]}.")

    try:
        req = requests.get("https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
                           headers=headers, timeout=10)
        req.raise_for_status()
        today_df = pd.read_csv(io.StringIO(req.text))
        today_set = {f"{s}.NS" for s in today_df['Symbol']}
    except Exception as e:
        print(f"Could not fetch today's live list to anchor the reconstruction ({e}). Aborting.")
        return None

    working_set = set(today_set)
    pit = {}
    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    pit[today_str] = sorted(working_set)
    for date_str, symbol, action in reversed(events):
        pit[date_str] = sorted(working_set)
        if action == 'IN':
            working_set.discard(symbol)
        else:
            working_set.add(symbol)
    pit[START_DATE] = sorted(working_set)

    with open(output_file, 'w') as f:
        json.dump(pit, f, indent=2)
    print(f"Wrote {len(pit)} membership snapshots to {output_file}.")
    print("Set POINT_IN_TIME_UNIVERSE_FILE to this path to use it. Next 50 / Midcap 150 "
          "still default to 'always eligible' -- spot-check before real use.")
    return output_file


def build_eligibility_mask(master_dates, stock_names, n_days, n_stocks):
    """True if a stock may be a NEW entry on that date. All-eligible unless
    POINT_IN_TIME_UNIVERSE_FILE points at a real reconstructed file --
    existing positions are never force-closed on removal."""
    mask = np.ones((n_days, n_stocks), dtype=np.bool_)
    if not POINT_IN_TIME_UNIVERSE_FILE or not os.path.exists(POINT_IN_TIME_UNIVERSE_FILE):
        return mask
    try:
        with open(POINT_IN_TIME_UNIVERSE_FILE, 'r') as f:
            pit = json.load(f)
    except Exception as e:
        print(f"Could not load point-in-time universe file ({e}); using all-eligible default.")
        return mask

    sorted_dates = sorted(pit.keys())
    name_to_idx = {name: i for i, name in enumerate(stock_names)}
    cursor = 0
    current_set = set()
    for d_idx, dt in enumerate(master_dates):
        dt_str = dt.strftime('%Y-%m-%d')
        while cursor < len(sorted_dates) and sorted_dates[cursor] <= dt_str:
            current_set = set(pit[sorted_dates[cursor]])
            cursor += 1
        if not current_set:
            continue
        mask[d_idx, :] = False
        for tkr in current_set:
            if tkr in name_to_idx:
                mask[d_idx, name_to_idx[tkr]] = True
    return mask


def build_liquidity_mask(notional_vol_matrix, lookback_days=LIQUIDITY_LOOKBACK_DAYS,
                          min_avg_inr=MIN_AVG_DAILY_NOTIONAL_INR):
    """Rolling trailing-average INR notional (Volume x Close) floor for a
    NEW entry -- the NSE analog of the crypto engine's USD quote-volume
    mask. A thin-but-currently-liquid midcap doesn't get treated as
    always-tradeable; a name that goes quiet stops qualifying for new
    entries from that point on. Existing positions are never force-closed."""
    vol_df = pd.DataFrame(notional_vol_matrix)
    rolling_avg = vol_df.rolling(window=lookback_days, min_periods=lookback_days).mean().values
    return rolling_avg >= min_avg_inr


# ==========================================================================
# 2. INDICATORS -- own copy (see module docstring for why this pair
# doesn't import from the crypto pair: keeping each asset-class pair
# fully standalone).
# ==========================================================================

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
    if ma_type == 0:
        w_sum = np.sum(prices[start:start + period])
        res[start + period - 1] = w_sum / period
        for i in range(start + period, n):
            w_sum = w_sum - prices[i - period] + prices[i]
            res[i] = w_sum / period
    elif ma_type == 1 or ma_type == 2:
        ema1 = np.empty(n)
        ema1[:] = np.nan
        ema1[start + period - 1] = np.mean(prices[start:start + period])
        mult = 2 / (period + 1)
        for i in range(start + period, n):
            ema1[i] = (prices[i] - ema1[i - 1]) * mult + ema1[i - 1]
        if ma_type == 1:
            res = ema1
        else:
            ema2 = np.empty(n)
            ema2[:] = np.nan
            if start + period * 2 - 2 < n:
                ema2[start + period * 2 - 2] = np.mean(ema1[start + period - 1:start + period * 2 - 1])
                for i in range(start + period * 2 - 1, n):
                    ema2[i] = (ema1[i] - ema2[i - 1]) * mult + ema2[i - 1]
                for i in range(start + period * 2 - 2, n):
                    res[i] = 2 * ema1[i] - ema2[i]
    elif ma_type == 3:
        weights = np.arange(1, period + 1, dtype=np.float64)
        w_sum = np.sum(weights)
        for i in range(start + period - 1, n):
            res[i] = np.sum(prices[i - period + 1:i + 1] * weights) / w_sum
    elif ma_type == 4:
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
    gains = np.zeros(n); losses = np.zeros(n)
    for i in range(start + 1, n):
        diff = prices[i] - prices[i - 1]
        if diff > 0: gains[i] = diff
        else: losses[i] = -diff
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
    tr = np.zeros(n); plus_dm = np.zeros(n); minus_dm = np.zeros(n)
    for i in range(start + 1, n):
        high_diff = highs[i] - highs[i - 1]
        low_diff  = lows[i - 1] - lows[i]
        tr[i]        = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        plus_dm[i]   = high_diff if high_diff > low_diff and high_diff > 0 else 0.0
        minus_dm[i]  = low_diff  if low_diff  > high_diff and low_diff  > 0 else 0.0
    atr_s = np.sum(tr[start+1:start+period+1])
    plus_s = np.sum(plus_dm[start+1:start+period+1])
    minus_s = np.sum(minus_dm[start+1:start+period+1])
    for i in range(start + period, n):
        if i > start + period:
            atr_s   = atr_s   - atr_s   / period + tr[i]
            plus_s  = plus_s  - plus_s  / period + plus_dm[i]
            minus_s = minus_s - minus_s / period + minus_dm[i]
        if atr_s > 0:
            plus_di = 100 * plus_s / atr_s
            minus_di = 100 * minus_s / atr_s
            denom = plus_di + minus_di
            dx = 100 * abs(plus_di - minus_di) / denom if denom > 0 else 0.0
        else:
            dx = 0.0
        if i == start + period: adx[i] = dx
        else: adx[i] = (adx[i - 1] * (period - 1) + dx) / period
    return adx


# ==========================================================================
# 2b. CACHES
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
# 3. PER-STOCK SIGNAL SIMULATOR
# ==========================================================================

@njit(nogil=True, cache=True)
def simulate_signal_trades(
        opens, closes, atr, adx,
        entry_ma, xover_short, xover_long,
        rsi_fast, rsi_slow,
        trend_ma,
        index_close, index_ma,
        exit_ma, exit_xover_short, exit_xover_long,
        exit_rsi_fast, exit_rsi_slow,
        eligible,
        entry_type, exit_type,
        use_index_entry_gate, use_index_exit_override, use_trend_filter,
        use_latched_entry,
        is_div_stock, use_dividend_bifurcation, div_exit_method, div_exit_val,
        adx_threshold,
        sl_mult, tp_mult, trail_mult, trail_pct, exit_atr_mult,
        max_pyramid_layers,
        early_exit_slope_lookback,
        start_day, end_day):
    """
    Same latched-entry / pyramiding / two-phase execution discipline as
    the crypto engine's simulator (see its docstring). NSE-SPECIFIC:

    - use_trend_filter applies to ALL THREE entry families uniformly
      (crypto's equivalent only ever gated the RSI entry family).
    - DIVIDEND BIFURCATION: when use_dividend_bifurcation and is_div_stock
      are both true, exit logic for this stock ignores exit_type entirely
      and uses div_exit_method instead (0=fixed %% drawdown from peak,
      1=%% violation of the trend MA, 2=consecutive days below the trend
      MA) -- and is exempt from use_index_exit_override, matching the
      source design: a defensive sleeve shouldn't be force-sold by a
      broad growth-market panic rule. Entry triggers are unaffected --
      only the exit character differs for a dividend name.
    - EXIT_EARLY_TREND_BREAK (6): price closes below its own trend_ma
      while trend_ma itself is STILL RISING (compared to
      early_exit_slope_lookback bars ago) -- a materially earlier,
      distinct tell from a plain crossunder against a fixed level.
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

    in_pos = False
    n_layers = 0
    blended_entry_price = 0.0
    entry_day = 0
    entry_risk = 0.0
    stop_loss_price = 0.0
    tp_trigger_price = 0.0
    half_sold = False
    partial_exit_price = 0.0
    high_since_entry = 0.0
    days_below_trend = 0.0

    pending_entry = False
    pending_exit = False
    pending_partial = False
    is_armed = False

    dividend_mode = use_dividend_bifurcation and is_div_stock

    for d in range(start_day, end_day):

        if pending_partial and in_pos:
            fill_price = opens[d]
            if not (fill_price > 0): fill_price = closes[d - 1]
            partial_exit_price = fill_price
            half_sold = True
            if stop_loss_price < blended_entry_price:
                stop_loss_price = blended_entry_price
            pending_partial = False

        if pending_exit and in_pos:
            fill_price = opens[d]
            if not (fill_price > 0): fill_price = closes[d - 1]
            if half_sold:
                blended_exit = 0.5 * partial_exit_price + 0.5 * fill_price
            else:
                blended_exit = fill_price
            pnl_per_unit = blended_exit - blended_entry_price
            if t_cnt < MAX_TRADES:
                t_entry_day[t_cnt] = entry_day
                t_exit_day[t_cnt] = d
                t_entry_price[t_cnt] = blended_entry_price
                t_exit_price[t_cnt] = blended_exit
                t_r_multiple[t_cnt] = pnl_per_unit / entry_risk if entry_risk > 0 else 0.0
                t_pct_return[t_cnt] = pnl_per_unit / blended_entry_price if blended_entry_price > 0 else 0.0
                t_bars_held[t_cnt] = d - entry_day
                t_layers[t_cnt] = n_layers
                t_cnt += 1
            in_pos = False
            n_layers = 0
            half_sold = False
            days_below_trend = 0.0
            pending_exit = False

        if pending_entry:
            fill_price = opens[d]
            if not (fill_price > 0): fill_price = closes[d - 1]
            if fill_price > 0:
                if not in_pos:
                    in_pos = True
                    n_layers = 1
                    blended_entry_price = fill_price
                    entry_day = d
                    high_since_entry = fill_price
                    half_sold = False
                    days_below_trend = 0.0
                    e_atr = atr[d] if atr[d] > 0 else atr[d - 1]
                    if not (e_atr > 0): e_atr = fill_price * 0.02
                    entry_risk = e_atr * sl_mult
                    stop_loss_price = fill_price - e_atr * sl_mult
                    tp_trigger_price = fill_price + e_atr * tp_mult
                else:
                    n_layers += 1
                    blended_entry_price = (blended_entry_price * (n_layers - 1) + fill_price) / n_layers
                    high_since_entry = max(high_since_entry, fill_price)
            pending_entry = False

        curr_close = closes[d]
        prev_close = closes[d - 1]

        index_bullish = True
        if use_index_entry_gate or use_index_exit_override:
            index_bullish = index_close[d] > index_ma[d]

        # ---- exit evaluation ----
        if in_pos and not pending_exit:
            if curr_close > 0:
                high_since_entry = max(high_since_entry, curr_close)

            should_exit_full = False
            should_exit_partial = False

            if dividend_mode:
                if div_exit_method == DIV_EXIT_PEAK_DRAWDOWN:
                    if curr_close < high_since_entry * (1.0 - div_exit_val / 100.0):
                        should_exit_full = True
                elif div_exit_method == DIV_EXIT_TREND_MA_VIOLATION:
                    if curr_close < trend_ma[d] * (1.0 - div_exit_val / 100.0):
                        should_exit_full = True
                elif div_exit_method == DIV_EXIT_TIME_DECAY:
                    if curr_close < trend_ma[d]:
                        days_below_trend += 1
                    else:
                        days_below_trend = 0.0
                    if days_below_trend > div_exit_val:
                        should_exit_full = True
                # dividend names are deliberately EXEMPT from use_index_exit_override
            else:
                if exit_type == EXIT_HYBRID:
                    if not half_sold and curr_close >= tp_trigger_price:
                        should_exit_partial = True
                    if half_sold:
                        pot_sl = curr_close - atr[d] * trail_mult
                        if pot_sl > stop_loss_price: stop_loss_price = pot_sl
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
                elif exit_type == EXIT_EARLY_TREND_BREAK:
                    lb = d - early_exit_slope_lookback
                    if lb >= 0 and not np.isnan(trend_ma[d]) and not np.isnan(trend_ma[lb]):
                        trend_rising = trend_ma[d] > trend_ma[lb]
                        if curr_close < trend_ma[d] and trend_rising:
                            should_exit_full = True

                if use_index_exit_override and not index_bullish:
                    should_exit_full = True
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
            elif entry_type == ENTRY_MA_XOVER:
                if xover_short[d - 1] <= xover_long[d - 1] and xover_short[d] > xover_long[d]:
                    trigger = True
                if xover_short[d - 1] >= xover_long[d - 1] and xover_short[d] < xover_long[d]:
                    invalidator = True

            filter_ok = (not use_trend_filter) or (curr_close > trend_ma[d])
            index_ok = (not use_index_entry_gate) or index_bullish or dividend_mode
            adx_ok = (adx_threshold <= 0.0) or (adx[d] >= adx_threshold)

            entry_signal = False
            if use_latched_entry:
                if trigger: is_armed = True
                if invalidator: is_armed = False
                if is_armed and filter_ok and index_ok and adx_ok:
                    entry_signal = True
                    is_armed = False
            else:
                entry_signal = trigger and filter_ok and index_ok and adx_ok

            if entry_signal:
                pending_entry = True

    return (t_entry_day[:t_cnt], t_exit_day[:t_cnt], t_entry_price[:t_cnt],
            t_exit_price[:t_cnt], t_r_multiple[:t_cnt], t_pct_return[:t_cnt],
            t_bars_held[:t_cnt], t_layers[:t_cnt])


# ==========================================================================
# 3b. DSR DIAGNOSTIC (same math, own copy)
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
    var_sr = sum((s - mean_sr) ** 2 for s in all_trial_srs) / n_trials
    sr_std = math.sqrt(var_sr)
    sr0 = expected_max_sharpe(sr_std, n_trials)
    dsr = probabilistic_sharpe_ratio(sr_hat, sr0, T, skew, kurt)
    return dsr, sr0, n_trials


# ==========================================================================
# 4. DATA PREPARATION
# ==========================================================================

def get_stock_data(ticker, data_dir="data_nse_signal"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    file_path = f"{data_dir}/{ticker}.csv"
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        except Exception:
            pass
    if yf is None:
        return None
    try:
        df = yf.download(ticker, start=START_DATE, progress=False, multi_level_index=False)
        if len(df) > 200:
            df.to_csv(file_path)
            return df
    except Exception:
        return None
    return None


def prepare_matrix_data(tickers):
    raw_dfs = {}
    master_dates = set()
    for ticker in tqdm(tickers, desc="Downloading NSE data"):
        df = get_stock_data(ticker)
        if df is not None and len(df) >= MIN_HISTORY_DAYS:
            raw_dfs[ticker] = df
            master_dates.update(df.index)

    if not raw_dfs:
        raise RuntimeError("No stocks downloaded -- check network access to Yahoo Finance.")

    master_dates = sorted(list(master_dates))
    master_df = pd.DataFrame(index=master_dates)
    master_df['Year'] = master_df.index.year

    n_days = len(master_dates)
    stock_names = list(raw_dfs.keys())
    n_stocks = len(stock_names)

    opens = np.zeros((n_days, n_stocks))
    highs = np.zeros((n_days, n_stocks))
    lows = np.zeros((n_days, n_stocks))
    closes = np.zeros((n_days, n_stocks))
    atr_matrix = np.zeros((n_days, n_stocks))
    adx_matrix = np.zeros((n_days, n_stocks))
    notional_vol_matrix = np.full((n_days, n_stocks), np.nan)
    is_div_stock = np.zeros(n_stocks, dtype=np.bool_)

    for i, ticker in enumerate(tqdm(stock_names, desc="Building matrix")):
        df = raw_dfs[ticker].reindex(master_dates)
        close_col = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
        opens[:, i] = df['Open'].ffill().values
        highs[:, i] = df['High'].ffill().values
        lows[:, i] = df['Low'].ffill().values
        closes[:, i] = close_col.ffill().values
        atr_matrix[:, i] = calc_atr_wilder(highs[:, i], lows[:, i], closes[:, i], period=14)
        adx_matrix[:, i] = calc_adx(highs[:, i], lows[:, i], closes[:, i], period=14)
        if 'Volume' in df.columns:
            notional_vol_matrix[:, i] = (df['Volume'].fillna(0) * close_col.ffill()).values
        if ticker in DIVIDEND_KINGS_FALLBACK:
            is_div_stock[i] = True

    liquidity_mask = build_liquidity_mask(notional_vol_matrix)
    pit_mask = build_eligibility_mask(master_dates, stock_names, n_days, n_stocks)
    eligible_mask = liquidity_mask & pit_mask
    print(f"Eligible coin-days for new entries after liquidity + point-in-time filtering: "
          f"{int(eligible_mask.sum()):,} / {n_days * n_stocks:,}")

    years_arr = master_df['Year'].values.astype(np.int32)
    return (opens, closes, atr_matrix, adx_matrix, years_arr,
            stock_names, eligible_mask, is_div_stock, master_dates)


# ==========================================================================
# 5. EVALUATE PARAMS
# ==========================================================================

def collect_trade_log(p, opens, closes, atr, adx, years_arr, n_stocks,
                       eligible_mask, is_div_stock, start_day=0, end_day=-1):
    n_days = closes.shape[0]
    if end_day < 0: end_day = n_days - 1

    entry_type = p['entry_type']; exit_type = p['exit_type']
    use_index_entry_gate = p['use_index_entry_gate']
    use_index_exit_override = p['use_index_exit_override']
    use_trend_filter = p['use_trend_filter']
    use_latched_entry = p.get('use_latched_entry', False)
    use_dividend_bifurcation = p.get('use_dividend_bifurcation', False)

    zeros = _zeros_like(n_days)

    entry_ma_all = xover_short_all = xover_long_all = None
    rsi_fast_all = rsi_slow_all = None

    if entry_type == ENTRY_MA_BREAKOUT:
        entry_ma_all = np.zeros((n_days, n_stocks))
        for s in range(n_stocks):
            entry_ma_all[:, s] = get_ma_cached(closes, s, p['entry_ma_len'], p['entry_ma_type'])
    elif entry_type == ENTRY_RSI_XOVER:
        rsi_fast_all = np.zeros((n_days, n_stocks)); rsi_slow_all = np.zeros((n_days, n_stocks))
        for s in range(n_stocks):
            rsi_fast_all[:, s] = get_smoothed_rsi(closes, s, p['rsi_f_len'], p['rsi_f_smt'])
            rsi_slow_all[:, s] = get_smoothed_rsi(closes, s, p['rsi_s_len'], p['rsi_s_smt'])
    elif entry_type == ENTRY_MA_XOVER:
        xover_short_all = np.zeros((n_days, n_stocks)); xover_long_all = np.zeros((n_days, n_stocks))
        for s in range(n_stocks):
            xover_short_all[:, s] = get_ma_cached(closes, s, p['xover_short_len'], p['xover_short_type'])
            xover_long_all[:, s] = get_ma_cached(closes, s, p['xover_long_len'], p['xover_long_type'])

    # trend_ma is needed whenever use_trend_filter OR the early-warning
    # exit family OR dividend bifurcation's trend-based methods are active
    trend_ma_all = None
    need_trend_ma = (use_trend_filter or exit_type == EXIT_EARLY_TREND_BREAK or
                      (use_dividend_bifurcation and p.get('div_exit_method', 0) in
                       (DIV_EXIT_TREND_MA_VIOLATION, DIV_EXIT_TIME_DECAY)))
    if need_trend_ma:
        trend_ma_all = np.zeros((n_days, n_stocks))
        for s in range(n_stocks):
            trend_ma_all[:, s] = get_ma_cached(closes, s, p['trend_ma_len'], p['trend_ma_type'])

    exit_ma_all = exit_xover_short_all = exit_xover_long_all = None
    exit_rsi_fast_all = exit_rsi_slow_all = None
    if exit_type == EXIT_MA_CROSSUNDER:
        exit_ma_all = np.zeros((n_days, n_stocks))
        for s in range(n_stocks):
            exit_ma_all[:, s] = get_ma_cached(closes, s, p['exit_ma_len'], p['exit_ma_type'])
    elif exit_type == EXIT_RSI_CROSSUNDER:
        exit_rsi_fast_all = np.zeros((n_days, n_stocks)); exit_rsi_slow_all = np.zeros((n_days, n_stocks))
        for s in range(n_stocks):
            exit_rsi_fast_all[:, s] = get_smoothed_rsi(closes, s, p['exit_rsi_f_len'], p['exit_rsi_f_smt'])
            exit_rsi_slow_all[:, s] = get_smoothed_rsi(closes, s, p['exit_rsi_s_len'], p['exit_rsi_s_smt'])
    elif exit_type == EXIT_MA_XOVER_EXIT:
        exit_xover_short_all = np.zeros((n_days, n_stocks)); exit_xover_long_all = np.zeros((n_days, n_stocks))
        for s in range(n_stocks):
            exit_xover_short_all[:, s] = get_ma_cached(closes, s, p['exit_xover_short_len'], p['exit_xover_short_type'])
            exit_xover_long_all[:, s] = get_ma_cached(closes, s, p['exit_xover_long_len'], p['exit_xover_long_type'])

    if use_index_entry_gate or use_index_exit_override:
        index_ma_all = get_ma_cached(closes, 0, p['index_ma_len'], p['index_ma_type'])
    else:
        index_ma_all = zeros
    index_close_col = closes[:, 0]

    div_exit_method = int(p.get('div_exit_method', 0))
    div_exit_val = float(p.get('div_exit_val', 15.0))
    early_lb = int(p.get('early_exit_slope_lookback', EARLY_EXIT_TREND_SLOPE_LOOKBACK))

    all_entry_day, all_exit_day, all_entry_price, all_exit_price = [], [], [], []
    all_r, all_pct, all_bars, all_stock_idx, all_layers = [], [], [], [], []
    max_pyramid_layers = int(p.get('max_pyramid_layers', 1))

    for s in range(n_stocks):
        entry_ma = entry_ma_all[:, s] if entry_ma_all is not None else zeros
        xover_short = xover_short_all[:, s] if xover_short_all is not None else zeros
        xover_long = xover_long_all[:, s] if xover_long_all is not None else zeros
        rsi_fast = rsi_fast_all[:, s] if rsi_fast_all is not None else zeros
        rsi_slow = rsi_slow_all[:, s] if rsi_slow_all is not None else zeros
        trend_ma = trend_ma_all[:, s] if trend_ma_all is not None else zeros
        exit_ma = exit_ma_all[:, s] if exit_ma_all is not None else zeros
        exit_xover_short = exit_xover_short_all[:, s] if exit_xover_short_all is not None else zeros
        exit_xover_long = exit_xover_long_all[:, s] if exit_xover_long_all is not None else zeros
        exit_rsi_fast = exit_rsi_fast_all[:, s] if exit_rsi_fast_all is not None else zeros
        exit_rsi_slow = exit_rsi_slow_all[:, s] if exit_rsi_slow_all is not None else zeros
        eligible_col = eligible_mask[:, s].astype(np.bool_)

        result = simulate_signal_trades(
            opens[:, s], closes[:, s], atr[:, s], adx[:, s],
            entry_ma, xover_short, xover_long,
            rsi_fast, rsi_slow,
            trend_ma,
            index_close_col, index_ma_all,
            exit_ma, exit_xover_short, exit_xover_long,
            exit_rsi_fast, exit_rsi_slow,
            eligible_col,
            entry_type, exit_type,
            use_index_entry_gate, use_index_exit_override, use_trend_filter,
            use_latched_entry,
            bool(is_div_stock[s]), use_dividend_bifurcation, div_exit_method, div_exit_val,
            p.get('adx_thresh', 0.0),
            p['sl_mult'], p['tp_mult'], p['trail_mult'], p['trail_pct'], p['exit_atr_mult'],
            max_pyramid_layers, early_lb,
            int(start_day), int(end_day)
        )
        (e_day, x_day, e_price, x_price, r_mult, pct_ret, bars, layers) = result
        if len(e_day) == 0:
            continue
        mask = (e_day >= start_day) & (x_day < end_day)
        if not mask.any():
            continue
        all_entry_day.append(e_day[mask]); all_exit_day.append(x_day[mask])
        all_entry_price.append(e_price[mask]); all_exit_price.append(x_price[mask])
        all_r.append(r_mult[mask]); all_pct.append(pct_ret[mask])
        all_bars.append(bars[mask]); all_layers.append(layers[mask])
        all_stock_idx.append(np.full(mask.sum(), s, dtype=np.int32))

    if not all_r:
        return None

    entry_days = np.concatenate(all_entry_day); exit_days = np.concatenate(all_exit_day)
    entry_prices = np.concatenate(all_entry_price); exit_prices = np.concatenate(all_exit_price)
    r_multiple = np.concatenate(all_r); pct_return = np.concatenate(all_pct)
    bars_held = np.concatenate(all_bars); layers_used = np.concatenate(all_layers)
    stock_idx = np.concatenate(all_stock_idx)
    entry_years = years_arr[entry_days]
    window_years = years_arr[start_day:end_day]

    return {
        'entry_days': entry_days, 'exit_days': exit_days,
        'entry_prices': entry_prices, 'exit_prices': exit_prices,
        'r_multiple': r_multiple, 'pct_return': pct_return,
        'bars_held': bars_held, 'layers_used': layers_used,
        'stock_idx': stock_idx, 'entry_years': entry_years,
        'global_min_year': int(window_years.min()), 'global_max_year': int(window_years.max()),
    }


def evaluate_params_signal(p, opens, closes, atr, adx, years_arr, n_stocks,
                            eligible_mask, is_div_stock, start_day=0, end_day=-1, is_oos=False,
                            min_years_required=None, min_trades_required=None):
    log = collect_trade_log(p, opens, closes, atr, adx, years_arr, n_stocks,
                             eligible_mask, is_div_stock, start_day, end_day)
    if log is None:
        return -999.0, {}
    return compute_score_signal(log['r_multiple'], log['pct_return'], log['bars_held'],
                                 log['entry_years'], log['stock_idx'], log['entry_days'],
                                 log['exit_days'], log['entry_prices'], log['exit_prices'],
                                 log['layers_used'], log['global_min_year'], log['global_max_year'],
                                 is_oos=is_oos, min_years_required=min_years_required,
                                 min_trades_required=min_trades_required)


# ==========================================================================
# 6. SCORING (identical formula/gates to the crypto engine, own copy;
# "median_coin_mean_r" renamed "median_stock_mean_r" for NSE vocabulary)
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
        'trades': int(n_trades), 'r_multiple': r_capped, 'r_multiple_raw': r_multiple,
        'pct_return': pct_return, 'bars_held': bars_held, 'entry_years': entry_years,
        'stock_idx': stock_idx, 'entry_days': entry_days, 'exit_days': exit_days,
        'entry_prices': entry_prices, 'exit_prices': exit_prices, 'layers_used': layers_used,
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

    gross_win = float(r_capped[wins_mask].sum()) if wins_mask.any() else 0.0
    gross_loss = float(-r_capped[~wins_mask].sum()) if (~wins_mask).any() else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    if profit_factor < MIN_PROFIT_FACTOR:
        return -999.0, metrics

    avg_win_r = float(r_capped[wins_mask].mean()) if wins_mask.any() else 0.0
    avg_loss_r = float(-r_capped[~wins_mask].mean()) if (~wins_mask).any() else 0.0
    expectancy_r = win_rate * avg_win_r - (1.0 - win_rate) * avg_loss_r
    if expectancy_r <= 0:
        return -999.0, metrics

    per_stock_mean_r = {int(c): float(r_multiple[stock_idx == c].mean()) for c in np.unique(stock_idx)}
    median_stock_mean_r = float(np.median(list(per_stock_mean_r.values())))
    if median_stock_mean_r <= MEDIAN_STOCK_R_GATE:
        metrics['median_stock_mean_r'] = median_stock_mean_r
        return -999.0, metrics

    # -- signal-level drawdown gate -- see MAX_SIGNAL_DRAWDOWN's docstring
    # (section 0) and the crypto engine's identical gate for the reasoning.
    per_stock_drawdown = compute_per_coin_drawdowns(r_multiple, entry_days, stock_idx)
    worst_stock_drawdown = max(per_stock_drawdown.values()) if per_stock_drawdown else 0.0
    if worst_stock_drawdown > MAX_SIGNAL_DRAWDOWN:
        metrics['worst_stock_drawdown'] = worst_stock_drawdown
        metrics['per_stock_drawdown'] = per_stock_drawdown
        return -999.0, metrics

    mean_r = float(r_capped.mean()); median_r = float(np.median(r_capped)); std_r = float(r_capped.std())
    sqn = (mean_r / std_r) * math.sqrt(min(n_trades, 100)) if std_r > 0 else 0.0
    sqn_capped = min(sqn, SQN_CAP)

    year_r_avg = {}; year_r_sum_raw = {}; year_coin_median_r = {}
    for yr in distinct_years:
        yr_mask = entry_years == yr
        year_r_avg[yr] = float(r_capped[yr_mask].mean())
        year_r_sum_raw[yr] = float(r_multiple[yr_mask].sum())
        # UPGRADE (per user request, same as the crypto engine's identical
        # change): median-across-coins-active-that-year, instead of a
        # trade-count-weighted pooled mean, drives the SCORED per-year
        # statistic -- see the crypto engine's comment for the full
        # reasoning. year_r_avg/year_r_sum_raw are kept as-is for the
        # concentration diagnostic, which is deliberately untouched.
        stocks_this_year = np.unique(stock_idx[yr_mask])
        per_stock_r_this_year = [float(r_capped[yr_mask & (stock_idx == c)].mean()) for c in stocks_this_year]
        year_coin_median_r[yr] = float(np.median(per_stock_r_this_year)) if per_stock_r_this_year else 0.0
    median_yearly_avg_r = float(np.median(list(year_coin_median_r.values())))

    weights = [_recency_weight(yr, global_min_year, global_max_year) for yr in distinct_years]
    w_sum = sum(weights)
    recency_weighted_avg_r = (sum(w * year_coin_median_r[yr] for w, yr in zip(weights, distinct_years)) / w_sum
                               if w_sum > 0 else median_yearly_avg_r)

    total_r_raw = float(r_multiple.sum())
    max_year_share = 0.0; max_stock_share = 0.0
    if total_r_raw > 0:
        max_year_share = max(v / total_r_raw for v in year_r_sum_raw.values())
        stock_r_sums = {int(c): float(r_multiple[stock_idx == c].sum()) for c in np.unique(stock_idx)}
        max_stock_share = max(v / total_r_raw for v in stock_r_sums.values())

    concentration_penalty = 1.0
    if max_year_share > CONCENTRATION_SOFT_THRESHOLD or max_stock_share > CONCENTRATION_SOFT_THRESHOLD:
        concentration_penalty = CONCENTRATION_PENALTY_MULT

    wr_bonus = max(0.0, win_rate - 0.50) * 4.0
    stat_conf = min(1.0, math.sqrt(n_trades / target_trades))

    score = (sqn_capped * W_SQN + expectancy_r * W_EXPECTANCY +
             recency_weighted_avg_r * W_RECENCY + wr_bonus * W_WR_BONUS) * stat_conf * concentration_penalty

    entry_year_counts = {yr: int((entry_years == yr).sum()) for yr in distinct_years}
    max_entry_year_concentration = max(entry_year_counts.values()) / n_trades if n_trades > 0 else 0.0
    mean_pct = float(pct_return.mean()) if pct_return is not None else 0.0
    median_pct = float(np.median(pct_return)) if pct_return is not None else 0.0
    avg_layers = float(layers_used.mean()) if layers_used is not None and len(layers_used) > 0 else 1.0
    pct_pyramided = float((layers_used > 1).mean()) if layers_used is not None and len(layers_used) > 0 else 0.0

    metrics.update({
        'win_rate': win_rate, 'profit_factor': profit_factor,
        'avg_win_r': avg_win_r, 'avg_loss_r': avg_loss_r, 'expectancy_r': expectancy_r,
        'mean_r': mean_r, 'median_r': median_r, 'std_r': std_r,
        'mean_r_raw': float(r_multiple.mean()), 'median_r_raw': float(np.median(r_multiple)),
        'mean_pct': mean_pct, 'median_pct': median_pct,
        'sqn': sqn, 'sqn_capped': sqn_capped,
        'median_yearly_avg_r': median_yearly_avg_r, 'recency_weighted_avg_r': recency_weighted_avg_r,
        'median_stock_mean_r': median_stock_mean_r, 'per_stock_mean_r': per_stock_mean_r,
        'worst_stock_drawdown': worst_stock_drawdown, 'per_stock_drawdown': per_stock_drawdown,
        'year_r_avg': year_r_avg, 'year_r_sum_raw': year_r_sum_raw, 'year_coin_median_r': year_coin_median_r,
        'year_weights': dict(zip(distinct_years, weights)),
        'global_min_year': global_min_year, 'global_max_year': global_max_year,
        'max_year_share': max_year_share, 'max_stock_share': max_stock_share,
        'concentration_penalty': concentration_penalty, 'distinct_years': distinct_years,
        'distinct_stocks': int(len(np.unique(stock_idx))),
        'avg_bars_held': float(bars_held.mean()) if n_trades > 0 else 0.0,
        'entry_year_counts': entry_year_counts, 'max_entry_year_concentration': max_entry_year_concentration,
        'avg_pyramid_layers': avg_layers, 'pct_trades_pyramided': pct_pyramided,
    })
    return score, metrics


def diagnose_gates_signal(r_multiple, pct_return, bars_held, entry_years, stock_idx,
                           entry_days=None, is_oos=False, min_trades_required=None):
    n_trades = len(r_multiple)
    target_trades = min_trades_required if min_trades_required is not None else (
        max(15, int(MIN_TRADES_GATE * (WFO_OOS_PCT / WFO_IS_PCT))) if is_oos else MIN_TRADES_GATE)
    rows = [('trades >= target', n_trades >= target_trades, n_trades, f'>= {target_trades}')]
    if n_trades == 0:
        return rows
    distinct_years = sorted(set(int(y) for y in entry_years))
    rows.append(('distinct years >= gate', len(distinct_years) >= MIN_YEARS_GATE, len(distinct_years), f'>= {MIN_YEARS_GATE}'))
    wins_mask = r_multiple > 0
    win_rate = float(wins_mask.mean())
    rows.append(('win_rate >= floor', win_rate >= MIN_WIN_RATE_GATE, f'{win_rate*100:.1f}%', f'>= {MIN_WIN_RATE_GATE*100:.0f}%'))
    gross_win = float(r_multiple[wins_mask].sum()) if wins_mask.any() else 0.0
    gross_loss = float(-r_multiple[~wins_mask].sum()) if (~wins_mask).any() else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    rows.append(('profit_factor >= gate', profit_factor >= MIN_PROFIT_FACTOR, f'{profit_factor:.2f}', f'>= {MIN_PROFIT_FACTOR}'))
    avg_win_r = float(r_multiple[wins_mask].mean()) if wins_mask.any() else 0.0
    avg_loss_r = float(-r_multiple[~wins_mask].mean()) if (~wins_mask).any() else 0.0
    expectancy_r = win_rate * avg_win_r - (1.0 - win_rate) * avg_loss_r
    rows.append(('expectancy_R > 0', expectancy_r > 0, f'{expectancy_r:.3f}R', '> 0'))
    per_stock_r = [float(r_multiple[stock_idx == c].mean()) for c in np.unique(stock_idx)]
    median_stock_r = float(np.median(per_stock_r)) if per_stock_r else 0.0
    rows.append(('median-stock mean-R > gate', median_stock_r > MEDIAN_STOCK_R_GATE, f'{median_stock_r:.3f}R', f'> {MEDIAN_STOCK_R_GATE}'))
    if entry_days is not None:
        per_stock_dd = compute_per_coin_drawdowns(r_multiple, entry_days, stock_idx)
        worst_dd = max(per_stock_dd.values()) if per_stock_dd else 0.0
        rows.append((f'worst-stock drawdown <= {MAX_SIGNAL_DRAWDOWN*100:.0f}%', worst_dd <= MAX_SIGNAL_DRAWDOWN,
                     f'{worst_dd*100:.1f}%', f'<= {MAX_SIGNAL_DRAWDOWN*100:.0f}%'))
    return rows


def print_gate_diagnosis_signal(r_multiple, pct_return, bars_held, entry_years, stock_idx,
                                 entry_days=None, is_oos=False, label="GATE DIAGNOSIS", min_trades_required=None):
    rows = diagnose_gates_signal(r_multiple, pct_return, bars_held, entry_years, stock_idx,
                                  entry_days, is_oos, min_trades_required)
    print(f"\n{'-'*60}\n{label}\n{'-'*60}")
    first_fail = False
    for name, passed, actual, threshold in rows:
        marker = "PASS" if passed else "FAIL"
        print(f"  [{marker}] {name:<28} actual={actual!s:<10} needed {threshold}")
        if not passed and not first_fail:
            print("       ^-- this is the gate that produced the -999 score")
            first_fail = True
    print("-" * 60)


# ==========================================================================
# 7. NEIGHBORHOOD STABILITY
# ==========================================================================

def passes_neighborhood_check(p, base_score, opens, closes, atr, adx, years_arr, n_stocks,
                               eligible_mask, is_div_stock, start_day, end_day):
    if base_score <= 0:
        return True
    perturbations = []
    if p['entry_type'] == ENTRY_MA_BREAKOUT:
        perturbations += [{'entry_ma_len': p['entry_ma_len'] + 10}, {'entry_ma_len': max(MA_LEN_MIN, p['entry_ma_len'] - 10)}]
    elif p['entry_type'] == ENTRY_RSI_XOVER:
        perturbations += [{'rsi_f_len': p['rsi_f_len'] + 5}, {'rsi_s_len': p['rsi_s_len'] + 5}]
    elif p['entry_type'] == ENTRY_MA_XOVER:
        perturbations += [{'xover_short_len': p['xover_short_len'] + 10}, {'xover_long_len': p['xover_long_len'] + 10}]

    if p.get('use_trend_filter') or p['exit_type'] == EXIT_EARLY_TREND_BREAK:
        perturbations += [{'trend_ma_len': p['trend_ma_len'] + 10}]

    if p['exit_type'] == EXIT_MA_CROSSUNDER:
        perturbations += [{'exit_ma_len': p['exit_ma_len'] + 5}]
    elif p['exit_type'] == EXIT_PCT_TRAIL:
        perturbations += [{'trail_pct': p['trail_pct'] * 0.85}, {'trail_pct': p['trail_pct'] * 1.15}]
    elif p['exit_type'] == EXIT_ATR_TRAIL:
        perturbations += [{'exit_atr_mult': p['exit_atr_mult'] * 0.85}]
    elif p['exit_type'] == EXIT_HYBRID:
        perturbations += [{'sl_mult': p['sl_mult'] * 0.85}, {'sl_mult': p['sl_mult'] * 1.15}]

    if p.get('use_index_entry_gate') or p.get('use_index_exit_override'):
        perturbations += [{'index_ma_len': p['index_ma_len'] + 10}]

    if p.get('use_dividend_bifurcation'):
        perturbations += [{'div_exit_val': p.get('div_exit_val', 15.0) * 0.85}]

    for delta in perturbations:
        n_p = p.copy()
        n_p.update(delta)
        n_score, _ = evaluate_params_signal(n_p, opens, closes, atr, adx, years_arr, n_stocks,
                                            eligible_mask, is_div_stock, start_day, end_day, is_oos=False)
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
                                  eligible_mask, is_div_stock, full_start_day, full_end_day,
                                  initial_train_years=WF_INITIAL_TRAIN_YEARS):
    """Same deliberately-simplified, frozen-param replay as the crypto
    engine -- see that file's docstring for exactly why full per-fold
    re-optimization isn't done here. A fold is INCONCLUSIVE only when it
    has too few trades to say anything; a fold with plenty of trades that
    genuinely lost money is a real FAIL, read directly off collect_trade_log
    rather than through the full aggregate gate suite."""
    window_years = years_arr[full_start_day:full_end_day]
    if len(window_years) == 0:
        return {'folds': [], 'pass_rate': 0.0, 'n_conclusive': 0, 'n_inconclusive': 0}
    min_year = int(window_years.min()); max_year = int(window_years.max())

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
                                 eligible_mask, is_div_stock, start_day=s_day, end_day=e_day)
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
        folds.append({'test_year': test_year, 'conclusive': conclusive, 'expectancy_r': expectancy_r, 'trades': n_trades})
        test_year += 1

    conclusive_folds = [f for f in folds if f['conclusive']]
    n_profitable = sum(1 for f in conclusive_folds if (f['expectancy_r'] or 0) > 0)
    pass_rate = (n_profitable / len(conclusive_folds)) if conclusive_folds else 0.0
    return {'folds': folds, 'pass_rate': pass_rate, 'n_conclusive': len(conclusive_folds),
            'n_inconclusive': len(folds) - len(conclusive_folds)}


# ==========================================================================
# 10. SHORTLIST MANAGEMENT
# ==========================================================================

def _dedup_key(p):
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
    def __init__(self, cap=SHORTLIST_POOL_CAP):
        self.cap = cap
        self.pool = {}

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
    out = {
        'engine_version': ENGINE_VERSION, 'generated_at': datetime.now(timezone.utc).isoformat(),
        'universe': stock_names, 'year_range': [min_year, max_year], 'candidates': candidates,
    }

    def _clean(o):
        if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list): return [_clean(v) for v in o]
        if isinstance(o, (np.floating, float)): return float(o)
        if isinstance(o, (np.integer, int)): return int(o)
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, np.ndarray): return _clean(o.tolist())
        return o

    with open(filename, 'w') as f:
        json.dump(_clean(out), f, indent=2, default=str)
    print(f"\nWrote {len(candidates)} shortlisted candidates to {filename}")


# ==========================================================================
# 11. REPORTING
# ==========================================================================

ENTRY_TYPE_NAMES = {0: 'MA Breakout', 1: 'RSI Crossover', 2: 'MA Crossover'}
EXIT_TYPE_NAMES = {0: 'Hybrid ATR TP+Trail', 1: '%-Trail from High', 2: 'ATR-Trail from High',
                    3: 'MA Crossunder', 4: 'RSI Crossunder', 5: 'MA Crossover Exit',
                    6: 'Early Trend-Break Warning'}


def print_performance_report(score, metrics, p, label=""):
    print("\n" + "=" * 78)
    print(f"{label}")
    print("=" * 78)
    print(f"Score: {score:.4f}")
    print(f"Entry: {ENTRY_TYPE_NAMES.get(p['entry_type'],'?')}   Exit: {EXIT_TYPE_NAMES.get(p['exit_type'],'?')}   "
          f"Latched: {p.get('use_latched_entry', False)}   Pyramid layers: {p.get('max_pyramid_layers', 1)}")
    print(f"Trend filter: {p.get('use_trend_filter')}   Index gate: {p['use_index_entry_gate']}   "
          f"Index override: {p['use_index_exit_override']}   Dividend bifurcation: {p.get('use_dividend_bifurcation', False)}   "
          f"ADX: {p.get('adx_thresh', 0.0)}")
    print(f"\nTrades: {metrics.get('trades',0)}  |  Win Rate: {metrics.get('win_rate',0)*100:.1f}%  |  "
          f"Profit Factor: {metrics.get('profit_factor',0):.2f}")
    print(f"Expectancy: {metrics.get('expectancy_r',0):.3f}R  |  SQN: {metrics.get('sqn_capped',0):.2f}  |  "
          f"Recency-weighted avg-R/trade: {metrics.get('recency_weighted_avg_r',0):.2f}")
    print(f"Median-stock mean-R: {metrics.get('median_stock_mean_r',0):.3f}R  |  "
          f"Distinct stocks: {metrics.get('distinct_stocks',0)}  |  Distinct years: {len(metrics.get('distinct_years',[]))}")
    print(f"Worst-stock drawdown: {metrics.get('worst_stock_drawdown',0)*100:.1f}%  "
          f"(gate: <= {MAX_SIGNAL_DRAWDOWN*100:.0f}%, assumes {DRAWDOWN_RISK_PER_TRADE*100:.0f}% risked per trade)")
    print("=" * 78)


# ==========================================================================
# 12. OPTUNA SEARCH SPACE + MAIN
# ==========================================================================

def _suggest_params(trial):
    entry_type = trial.suggest_categorical('entry_type', [ENTRY_MA_BREAKOUT, ENTRY_RSI_XOVER, ENTRY_MA_XOVER])
    exit_type = trial.suggest_categorical('exit_type', [EXIT_HYBRID, EXIT_PCT_TRAIL, EXIT_ATR_TRAIL,
                                                          EXIT_MA_CROSSUNDER, EXIT_RSI_CROSSUNDER,
                                                          EXIT_MA_XOVER_EXIT, EXIT_EARLY_TREND_BREAK])
    use_trend_filter = trial.suggest_categorical('use_trend_filter', [False, True])
    use_dividend_bifurcation = trial.suggest_categorical('use_dividend_bifurcation', [False, True])

    p = {
        'entry_type': entry_type, 'exit_type': exit_type,
        'use_index_entry_gate': trial.suggest_categorical('use_index_entry_gate', [False, True]),
        'use_index_exit_override': trial.suggest_categorical('use_index_exit_override', [False, True]),
        'use_trend_filter': use_trend_filter,
        'use_latched_entry': trial.suggest_categorical('use_latched_entry', [False, True]),
        'use_dividend_bifurcation': use_dividend_bifurcation,
        'adx_thresh': trial.suggest_categorical('adx_thresh', [0.0, 15.0, 20.0, 25.0]),
        'sl_mult': trial.suggest_float('sl_mult', 1.5, 8.0),
        'tp_mult': trial.suggest_float('tp_mult', 5.0, 70.0),
        'trail_mult': trial.suggest_float('trail_mult', 2.0, 15.0),
        'trail_pct': trial.suggest_float('trail_pct', 5.0, 35.0),
        'exit_atr_mult': trial.suggest_float('exit_atr_mult', 1.0, 6.0),
        'max_pyramid_layers': trial.suggest_int('max_pyramid_layers', MAX_PYRAMID_LAYERS_MIN, MAX_PYRAMID_LAYERS_MAX),
    }
    if entry_type == ENTRY_MA_BREAKOUT:
        p['entry_ma_len'] = trial.suggest_int('entry_ma_len', MA_LEN_MIN, MA_LEN_MAX)
        p['entry_ma_type'] = trial.suggest_int('entry_ma_type', 0, 4)
    elif entry_type == ENTRY_RSI_XOVER:
        p['rsi_f_len'] = trial.suggest_int('rsi_f_len', 10, 100)
        p['rsi_f_smt'] = trial.suggest_int('rsi_f_smt', 5, 50)
        p['rsi_s_len'] = trial.suggest_int('rsi_s_len', 10, 100)
        p['rsi_s_smt'] = trial.suggest_int('rsi_s_smt', 5, 50)
    elif entry_type == ENTRY_MA_XOVER:
        short_len = trial.suggest_int('xover_short_len', MA_LEN_MIN, 200)
        gap = trial.suggest_int('xover_gap', 10, 150)
        p['xover_short_len'] = short_len
        p['xover_short_type'] = trial.suggest_int('xover_short_type', 0, 4)
        p['xover_long_len'] = min(MA_LEN_MAX, short_len + gap)
        p['xover_long_type'] = trial.suggest_int('xover_long_type', 0, 4)

    if use_trend_filter or exit_type == EXIT_EARLY_TREND_BREAK or use_dividend_bifurcation:
        p['trend_ma_len'] = trial.suggest_int('trend_ma_len', MA_LEN_MIN, MA_LEN_MAX)
        p['trend_ma_type'] = trial.suggest_int('trend_ma_type', 0, 4)
    else:
        p['trend_ma_len'], p['trend_ma_type'] = 200, 0

    if p['use_index_entry_gate'] or p['use_index_exit_override']:
        p['index_ma_len'] = trial.suggest_int('index_ma_len', MA_LEN_MIN, MA_LEN_MAX)
        p['index_ma_type'] = trial.suggest_int('index_ma_type', 0, 4)
    else:
        p['index_ma_len'], p['index_ma_type'] = 62, 0

    if exit_type == EXIT_MA_CROSSUNDER:
        p['exit_ma_len'] = trial.suggest_int('exit_ma_len', MA_LEN_MIN, MA_LEN_MAX)
        p['exit_ma_type'] = trial.suggest_int('exit_ma_type', 0, 4)
    elif exit_type == EXIT_RSI_CROSSUNDER:
        p['exit_rsi_f_len'] = trial.suggest_int('exit_rsi_f_len', 10, 100)
        p['exit_rsi_f_smt'] = trial.suggest_int('exit_rsi_f_smt', 5, 50)
        p['exit_rsi_s_len'] = trial.suggest_int('exit_rsi_s_len', 10, 100)
        p['exit_rsi_s_smt'] = trial.suggest_int('exit_rsi_s_smt', 5, 50)
    elif exit_type == EXIT_MA_XOVER_EXIT:
        e_short = trial.suggest_int('exit_xover_short_len', MA_LEN_MIN, 200)
        e_gap = trial.suggest_int('exit_xover_gap', 10, 150)
        p['exit_xover_short_len'] = e_short
        p['exit_xover_short_type'] = trial.suggest_int('exit_xover_short_type', 0, 4)
        p['exit_xover_long_len'] = min(MA_LEN_MAX, e_short + e_gap)
        p['exit_xover_long_type'] = trial.suggest_int('exit_xover_long_type', 0, 4)

    if use_dividend_bifurcation:
        div_m = trial.suggest_int('div_exit_method', 0, 2)
        p['div_exit_method'] = div_m
        if div_m in (DIV_EXIT_PEAK_DRAWDOWN, DIV_EXIT_TREND_MA_VIOLATION):
            p['div_exit_val'] = trial.suggest_float('div_exit_val_pct', 5.0, 40.0)
        else:
            p['div_exit_val'] = float(trial.suggest_int('div_exit_val_days', 10, 100))
    else:
        p['div_exit_method'], p['div_exit_val'] = 0, 15.0

    p['early_exit_slope_lookback'] = EARLY_EXIT_TREND_SLOPE_LOOKBACK
    return p


def run_signal_search(bayesian_trials=BAYESIAN_TRIALS, tickers=None):
    tickers = tickers if tickers is not None else fetch_dynamic_universe()
    (opens, closes, atr, adx, years_arr, stock_names, eligible_mask,
     is_div_stock, master_dates) = prepare_matrix_data(tickers)

    clear_caches()
    n_stocks = closes.shape[1]
    n_days = closes.shape[0]
    is_end = int(n_days * WFO_IS_PCT)
    inner_split = is_end - int(is_end * INNER_VAL_PCT)

    print(f"\nMatrix: {n_days} days x {n_stocks} stocks")
    print(f"Inner-train: days 0-{inner_split}  |  Inner-val: days {inner_split}-{is_end}  |  "
          f"True OOS (final check only): days {is_end}-{n_days}")

    pool = ShortlistPool()
    all_trial_srs = []

    baseline_params = load_baseline_config()
    if baseline_params is not None:
        base_score, base_metrics = evaluate_params_signal(
            baseline_params, opens, closes, atr, adx, years_arr, n_stocks, eligible_mask, is_div_stock,
            start_day=0, end_day=is_end, is_oos=False)
        if base_score > -900:
            print_performance_report(base_score, base_metrics, baseline_params,
                                     "BASELINE CONFIG (in-sample) -- trial #1, the bar to beat")
            pool.add(baseline_params, base_score, base_metrics)

    if not OPTUNA_AVAILABLE:
        print("Optuna not installed -- cannot run the search.")
        return

    print(f"\nBayesian Optimization ({bayesian_trials} trials, robustness-aware)...")

    def optuna_objective(trial):
        p = _suggest_params(trial)
        train_score, train_m = evaluate_params_signal(p, opens, closes, atr, adx, years_arr, n_stocks,
                                                       eligible_mask, is_div_stock,
                                                       start_day=0, end_day=inner_split, is_oos=False)
        if train_m and train_m.get('std_r', 0) > 0:
            all_trial_srs.append(train_m['mean_r'] / train_m['std_r'])
        if train_score <= -900:
            return -999.0
        val_score, val_m = evaluate_params_signal(p, opens, closes, atr, adx, years_arr, n_stocks,
                                                   eligible_mask, is_div_stock,
                                                   start_day=inner_split, end_day=is_end, is_oos=True,
                                                   min_years_required=MIN_YEARS_GATE_INNER,
                                                   min_trades_required=MIN_TRADES_INNER_VAL)
        if val_score <= -900:
            return min(train_score, 20.0) * ROBUST_FALLBACK_SCALE - ROBUST_FALLBACK_PENALTY
        dual_score = min(train_score, val_score)
        if dual_score > 0:
            pool.add(p, dual_score, train_m)
        return dual_score

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=400,
                                                                    multivariate=True, group=True))
    if baseline_params is not None:
        keys = ['entry_type', 'exit_type', 'use_index_entry_gate', 'use_index_exit_override',
                'use_trend_filter', 'use_latched_entry', 'use_dividend_bifurcation', 'adx_thresh',
                'sl_mult', 'tp_mult', 'trail_mult', 'trail_pct', 'exit_atr_mult', 'max_pyramid_layers']
        study.enqueue_trial({k: v for k, v in baseline_params.items() if k in keys})

    study.optimize(optuna_objective, n_trials=bayesian_trials, show_progress_bar=True)

    print(f"\nSearch complete. {len(pool.pool)} distinct dual-pass candidates found.")
    print(f"Running final walk-forward stability replay + neighborhood check on the top {SHORTLIST_SIZE}...")

    finalists = pool.top(SHORTLIST_SIZE)
    shortlist_out = []
    for rank, cand in enumerate(finalists, start=1):
        p = cand['params']
        neighbor_ok = passes_neighborhood_check(p, cand['score'], opens, closes, atr, adx, years_arr,
                                                 n_stocks, eligible_mask, is_div_stock, 0, inner_split)
        wf = walk_forward_stability_check(p, opens, closes, atr, adx, years_arr, n_stocks,
                                          eligible_mask, is_div_stock, 0, n_days - 1)
        is_score, is_m = evaluate_params_signal(p, opens, closes, atr, adx, years_arr, n_stocks,
                                                 eligible_mask, is_div_stock, 0, is_end, is_oos=False)
        oos_score, oos_m = evaluate_params_signal(p, opens, closes, atr, adx, years_arr, n_stocks,
                                                   eligible_mask, is_div_stock, is_end, n_days - 1, is_oos=True)

        dsr = None
        if len(all_trial_srs) >= 30 and is_m.get('std_r', 0) > 0:
            skew = float(pd.Series(is_m['r_multiple']).skew())
            kurt = float(pd.Series(is_m['r_multiple']).kurtosis() + 3)
            sr_hat = is_m['mean_r'] / is_m['std_r']
            dsr, _, _ = deflated_sharpe_ratio(sr_hat, all_trial_srs, is_m['trades'], skew, kurt)

        label = f"SHORTLIST #{rank}  (neighbor-stable: {neighbor_ok}, WF pass-rate: {wf['pass_rate']*100:.0f}% over {wf['n_conclusive']} conclusive yrs)"
        print_performance_report(is_score, is_m, p, label)

        shortlist_out.append({
            'rank': rank, 'config_id': _dedup_key(p), 'params': p, 'neighbor_stable': neighbor_ok,
            'is_score': is_score, 'oos_score': oos_score, 'oos_cleared_gates': oos_score > -900,
            'deflated_sharpe_ratio': dsr, 'walk_forward': wf,
            'signal_metrics': {k: v for k, v in is_m.items()
                                if k not in ('r_multiple', 'r_multiple_raw', 'pct_return', 'bars_held',
                                             'entry_years', 'stock_idx', 'entry_days', 'exit_days',
                                             'entry_prices', 'exit_prices', 'layers_used')},
        })

    window_years = years_arr[0:n_days - 1]
    write_shortlist(shortlist_out, stock_names, int(window_years.min()), int(window_years.max()))
    print("\nDone. Hand nse_signal_shortlist.json to nse_portfolio_engine.py to see how these "
          "candidates actually hold up under a real shared SIP cash pool.")


if __name__ == "__main__":
    run_signal_search()