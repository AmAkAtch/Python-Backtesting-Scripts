"""
================================================================================
 MULTI-ASSET MOMENTUM STRATEGY BACKTESTING & HYPERPARAMETER OPTIMIZATION
 FRAMEWORK
================================================================================

Architecture (Nifty 50 + Nifty Next 50 + Nifty Midcap 50 universe, 15Y daily).

A note on environment: this script is network-complete (yfinance-based
download + local parquet caching) but was authored in a sandboxed dev
environment with no route to Yahoo Finance. `DataPipeline.download_universe`
will happily hit the real API when run in a normal environment; when it
cannot reach the network it transparently falls back to a seeded synthetic
OHLCV generator (`DataPipeline._synthesize`) so the rest of the pipeline
(indicators, portfolio engine, optimizer, robustness suite) is still fully
exercised and testable end-to-end. Swap `USE_SYNTHETIC_FALLBACK = False`
once you have live network access to force real downloads only.

Author: Principal Quantitative Software Engineer (framework scaffold)
================================================================================
"""

from __future__ import annotations

import os
import json
import math
import time
import random
import hashlib
import logging
import warnings
import dataclasses
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable, Any, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    import torch
    import torch.multiprocessing as mp
    from gpu_batch import GPUIndicatorBatch
    GPU_AVAILABLE = True
    # Ensure parallel processes spawn clean CUDA contexts
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
except ImportError:
    GPU_AVAILABLE = False

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):  # no-op decorator fallback so the @njit-decorated
        """Fallback no-op decorator so functions below still run (in pure Python,
        just without JIT compilation) when numba isn't installed."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def wrap(fn):
            return fn
        return wrap

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("momentum_framework")

# ==============================================================================
# GLOBAL CONSTANTS
# ==============================================================================

CACHE_DIR = "./data_cache"
RESULTS_DIR = "./results"
USE_SYNTHETIC_FALLBACK = True   # set False in a network-enabled environment
TRANSACTION_COST_PCT = 0.0015   # 0.15% per executed trade (STT+fees+brokerage+slippage)
MONTHLY_INJECTION = 10_000.0
MIN_ALLOCATION = 10_000.0
INDICATOR_PARAM_MIN = 5
INDICATOR_PARAM_MAX = 300
MAX_ENTRY_SLOTS = 10
MAX_EXIT_SLOTS = 10

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Representative NSE universe covering Nifty 50 / Next 50 / Midcap 50 (~150 names).
# In production, replace/extend this list with the full, point-in-time constituent
# list per index (survivorship bias is explicitly assumed away per spec).
NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS",
    "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS", "SUNPHARMA.NS",
    "ULTRACEMCO.NS", "NESTLEIND.NS", "WIPRO.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS",
    "M&M.NS", "TATASTEEL.NS", "TATAMOTORS.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "JSWSTEEL.NS", "COALINDIA.NS", "BAJAJFINSV.NS", "HCLTECH.NS", "DRREDDY.NS",
    "GRASIM.NS", "CIPLA.NS", "EICHERMOT.NS", "BRITANNIA.NS", "DIVISLAB.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "INDUSINDBK.NS", "APOLLOHOSP.NS", "TECHM.NS",
    "SBILIFE.NS", "BAJAJ-AUTO.NS", "SHRIRAMFIN.NS", "UPL.NS", "BPCL.NS",
    "LTIM.NS", "HDFCLIFE.NS",
]
NIFTY_NEXT_50 = [
    "ADANIGREEN.NS", "ADANIPOWER.NS", "AMBUJACEM.NS", "BANKBARODA.NS", "BEL.NS",
    "BOSCHLTD.NS", "CANBK.NS", "CHOLAFIN.NS", "COLPAL.NS", "DABUR.NS", "DLF.NS",
    "GAIL.NS", "GODREJCP.NS", "HAVELLS.NS", "ICICIGI.NS", "ICICIPRULI.NS",
    "IOC.NS", "IRCTC.NS", "JINDALSTEL.NS", "LICI.NS", "LODHA.NS", "MARICO.NS",
    "MOTHERSON.NS", "NAUKRI.NS", "PIDILITIND.NS", "PFC.NS", "PNB.NS", "RECLTD.NS",
    "SIEMENS.NS", "SRF.NS", "TATAPOWER.NS", "TORNTPHARM.NS", "TRENT.NS",
    "TVSMOTOR.NS", "VEDL.NS", "VBL.NS", "ZOMATO.NS", "ZYDUSLIFE.NS", "ABB.NS",
    "ATGL.NS", "BAJAJHLDNG.NS", "BERGEPAINT.NS", "DMART.NS", "INDIGO.NS",
    "MUTHOOTFIN.NS", "PAGEIND.NS", "PGHH.NS", "SHREECEM.NS", "TATACONSUM.NS", "UNITDSPR.NS",
]
NIFTY_MIDCAP_50 = [
    "AARTIIND.NS", "ABCAPITAL.NS", "ALKEM.NS", "APLAPOLLO.NS", "ASTRAL.NS",
    "AUBANK.NS", "AUROPHARMA.NS", "BALKRISIND.NS", "BANDHANBNK.NS", "BATAINDIA.NS",
    "BHARATFORG.NS", "BHEL.NS", "CGPOWER.NS", "COFORGE.NS", "CONCOR.NS",
    "CUMMINSIND.NS", "DEEPAKNTR.NS", "DIXON.NS", "ESCORTS.NS", "EXIDEIND.NS",
    "FEDERALBNK.NS", "FORTIS.NS", "GMRINFRA.NS", "GODREJPROP.NS", "GUJGASLTD.NS",
    "HDFCAMC.NS", "HONAUT.NS", "IDFCFIRSTB.NS", "IEX.NS", "INDHOTEL.NS",
    "INDUSTOWER.NS", "IPCALAB.NS", "JUBLFOOD.NS", "L&TFH.NS", "LALPATHLAB.NS",
    "LUPIN.NS", "MANAPPURAM.NS", "MAXHEALTH.NS", "MFSL.NS", "MRF.NS",
    "NAVINFLUOR.NS", "OBEROIRLTY.NS", "OFSS.NS", "PERSISTENT.NS", "PIIND.NS",
    "POLYCAB.NS", "SUNTV.NS", "SYNGENE.NS", "TIINDIA.NS", "VOLTAS.NS",
]
UNIVERSE: List[str] = NIFTY_50 + NIFTY_NEXT_50 + NIFTY_MIDCAP_50
assert len(UNIVERSE) == 150, f"Universe must contain 150 tickers, got {len(UNIVERSE)}"


# ==============================================================================
# SECTION: CONFIGURATION DATACLASSES
# ==============================================================================

@dataclass
class IndicatorSlot:
    """A single indicator slot used in either the entry or exit composer."""
    name: Optional[str] = None                 # key into IndicatorLibrary.REGISTRY, or None (empty slot)
    params: Dict[str, int] = field(default_factory=dict)
    comparator: str = "cross_above"             # cross_above | cross_below | greater_than | less_than
    threshold: float = 0.0                      # used by greater_than / less_than comparators

    def is_empty(self) -> bool:
        return self.name is None


@dataclass
class StrategyConfig:
    """Full, optimizer-tunable configuration for one momentum strategy instance."""
    use_heikin_ashi: bool = False
    entry_slots: List[IndicatorSlot] = field(default_factory=list)
    exit_slots: List[IndicatorSlot] = field(default_factory=list)

    # Exit engine parameters (no static % SL per spec -- dynamic/indicator driven only)
    take_profit_pct: Optional[float] = None      # e.g. 0.25 == +25%
    trailing_stop_pct: Optional[float] = None     # e.g. 0.10 == 10% trailing
    atr_chandelier_mult: Optional[float] = None   # ATR multiplier for chandelier exit
    atr_period: int = 22

    # Watchlist ranking model selector (1 of 6, see WatchlistEngine)
    watchlist_model: str = "weighted_score"

    # Weighted score model sub-weights (only used when watchlist_model == weighted_score)
    watchlist_weight_price_dist: float = 0.5
    watchlist_weight_age: float = 0.5
    watchlist_proximity_ma_period: int = 50

    # Risk overlay (SECTION 2 free-choice add-ons)
    regime_filter_enabled: bool = True
    regime_index_sma_period: int = 200
    sector_cap_pct: float = 0.25                  # max % of portfolio value in one sector
    vol_sizing_enabled: bool = True
    equal_weight_sizing_enabled: bool = False      # cost-basis equal-weight sizing (see PortfolioEngine.try_enter)
    cooldown_days: int = 0                         # min days after an exit before that ticker can re-enter (0 = off)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "StrategyConfig":
        entry = [IndicatorSlot(**s) for s in d.get("entry_slots", [])]
        exitl = [IndicatorSlot(**s) for s in d.get("exit_slots", [])]
        rest = {k: v for k, v in d.items() if k not in ("entry_slots", "exit_slots")}
        return StrategyConfig(entry_slots=entry, exit_slots=exitl, **rest)


# ==============================================================================
# SECTION: HEIKIN-ASHI TRANSFORM & PURE BUYING VOLUME
# ==============================================================================

class CandleTransform:
    """Heikin-Ashi transform and buyer-initiated volume estimation."""

    @staticmethod
    def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
        """Convert standard OHLC to Heikin-Ashi candles. df needs Open/High/Low/Close."""
        ha = pd.DataFrame(index=df.index)
        ha["Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0
        open_vals = df["Open"].to_numpy(dtype=np.float64)
        close_vals = ha["Close"].to_numpy(dtype=np.float64)
        ha["Open"] = _ha_open_numba(open_vals, close_vals, float(df["Close"].iloc[0]))
        ha["High"] = pd.concat([df["High"], ha["Open"], ha["Close"]], axis=1).max(axis=1)
        ha["Low"] = pd.concat([df["Low"], ha["Open"], ha["Close"]], axis=1).min(axis=1)
        if "Volume" in df.columns:
            ha["Volume"] = df["Volume"]
        return ha

    @staticmethod
    def ha_body_strength(ha_df: pd.DataFrame) -> pd.Series:
        """Body size relative to full candle range -- used by the HA Strength watchlist model."""
        rng = (ha_df["High"] - ha_df["Low"]).replace(0, np.nan)
        strength = (ha_df["Close"] - ha_df["Open"]).abs() / rng
        return strength.fillna(0.0)

    @staticmethod
    def pure_buying_volume(df: pd.DataFrame) -> pd.Series:
        """
        V_buy = Volume * (Close - Low) / (High - Low)
        Candle Range Delta estimate of buyer-initiated volume.
        """
        rng = (df["High"] - df["Low"]).replace(0, np.nan)
        v_buy = df["Volume"] * (df["Close"] - df["Low"]) / rng
        return v_buy.fillna(0.0)


# ==============================================================================
# SECTION: NUMBA-JIT CORE LOOPS (CPU-side accelerator, supporting the GPU path)
# ==============================================================================
# These back the handful of IndicatorLibrary functions that pandas is genuinely
# slow at: rolling windows whose statistic needs a Python-level callback per
# window (.rolling().apply(...)), and explicit sequential recursions using
# .iloc row-by-row. Both patterns pay huge per-call Python overhead in pandas;
# JIT-compiling the inner loop to machine code is a 10-50x class of win on
# exactly this kind of code, independent of and complementary to the GPU
# batch path (which handles cross-ticker vectorized rolling ops instead --
# GPU drives the "same op across 150 tickers at once" axis, Numba drives the
# "this one op has an irreducibly sequential/custom inner loop" axis).
#
# Every function below operates on plain float64 numpy arrays (numba does not
# understand pandas objects) and matches the NaN/warm-up semantics of the
# pandas-based implementation it replaces -- verified in validate_numba_parity().

@njit(cache=True)
def _ha_open_numba(open_vals: np.ndarray, ha_close_vals: np.ndarray, raw_close_0: float) -> np.ndarray:
    """
    Heikin-Ashi open price: ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    for i >= 1. Day 0 has no prior HA candle to derive from, so it's seeded
    from the RAW open/close instead: ha_open[0] = (raw_open[0] + raw_close[0]) / 2
    -- note this seed uses the raw close, NOT the Heikin-Ashi close, which is
    the standard HA bootstrap convention (and matches the original
    implementation this replaces; verified via parity test before use).
    A true recurrence (each day depends on the previous), and unlike the
    GPU-unsuitable EWM recurrences elsewhere in this file, this one runs on
    a SINGLE ticker at a time (no cross-ticker batch dimension to exploit
    even on GPU), so it stays a Numba target. Called unconditionally for
    every ticker on every trial (feeds the watchlist's HA-strength feature
    regardless of whether Heikin-Ashi mode itself is enabled).
    """
    n = len(open_vals)
    ha_open = np.empty(n)
    ha_open[0] = (open_vals[0] + raw_close_0) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close_vals[i - 1]) / 2.0
    return ha_open


@njit(cache=True)
def _wma_numba(values: np.ndarray, period: int) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan)
    weight_sum = period * (period + 1) / 2.0
    for i in range(period - 1, n):
        acc = 0.0
        valid = True
        for j in range(period):
            v = values[i - period + 1 + j]
            if np.isnan(v):
                valid = False
                break
            acc += v * (j + 1)
        if valid:
            out[i] = acc / weight_sum
    return out


@njit(cache=True)
def _kama_numba(price: np.ndarray, period: int, fast: float = 2.0 / 3.0, slow: float = 2.0 / 31.0) -> np.ndarray:
    n = len(price)
    out = np.full(n, np.nan)
    if period >= n:
        return out
    out[period] = price[period]
    for i in range(period + 1, n):
        change = abs(price[i] - price[i - period])
        vol = 0.0
        for j in range(i - period + 1, i + 1):
            vol += abs(price[j] - price[j - 1])
        er = change / vol if vol > 1e-12 else 0.0
        sc = (er * (fast - slow) + slow) ** 2
        out[i] = out[i - 1] + sc * (price[i] - out[i - 1])
    return out


@njit(cache=True)
def _linreg_slope_numba(values: np.ndarray, period: int) -> np.ndarray:
    """Closed-form OLS slope per rolling window (x = 0..period-1) -- replaces a
    per-window np.polyfit call, which is by far the single slowest indicator
    in the original library (a full least-squares solve per window, per day,
    per ticker)."""
    n = len(values)
    out = np.full(n, np.nan)
    sum_x = period * (period - 1) / 2.0
    sum_x2 = (period - 1) * period * (2 * period - 1) / 6.0
    denom = period * sum_x2 - sum_x * sum_x
    for i in range(period - 1, n):
        valid = True
        sum_y = 0.0
        sum_xy = 0.0
        for j in range(period):
            v = values[i - period + 1 + j]
            if np.isnan(v):
                valid = False
                break
            sum_y += v
            sum_xy += j * v
        if valid and denom != 0:
            out[i] = (period * sum_xy - sum_x * sum_y) / denom
    return out


@njit(cache=True)
def _rolling_mad_numba(values: np.ndarray, period: int) -> np.ndarray:
    """Rolling mean absolute deviation from the window mean (used by CCI)."""
    n = len(values)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        valid = True
        mean = 0.0
        for j in range(period):
            v = values[i - period + 1 + j]
            if np.isnan(v):
                valid = False
                break
            mean += v
        if not valid:
            continue
        mean /= period
        mad = 0.0
        for j in range(period):
            mad += abs(values[i - period + 1 + j] - mean)
        out[i] = mad / period
    return out


@njit(cache=True)
def _aroon_numba(values: np.ndarray, period: int, find_max: bool) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan)
    window = period + 1
    for i in range(window - 1, n):
        best_idx = 0
        best_val = values[i - window + 1]
        valid = not np.isnan(best_val)
        for j in range(1, window):
            v = values[i - window + 1 + j]
            if np.isnan(v):
                valid = False
                break
            if (find_max and v > best_val) or (not find_max and v < best_val):
                best_val = v
                best_idx = j
        if valid:
            out[i] = (period - (window - 1 - best_idx)) / period * 100.0
    return out


@njit(cache=True)
def _supertrend_numba(close: np.ndarray, upper_band: np.ndarray, lower_band: np.ndarray) -> np.ndarray:
    n = len(close)
    direction = np.ones(n)
    for i in range(1, n):
        if np.isnan(upper_band[i - 1]) or np.isnan(lower_band[i - 1]):
            direction[i] = direction[i - 1]
        elif close[i] > upper_band[i - 1]:
            direction[i] = 1.0
        elif close[i] < lower_band[i - 1]:
            direction[i] = -1.0
        else:
            direction[i] = direction[i - 1]
    return direction


@njit(cache=True)
def _monte_carlo_numba(pnls: np.ndarray, n_paths: int, seed: int, ruin_threshold: float):
    """
    Same logic as the original per-path numpy implementation (shuffle trade
    P&Ls, track running-max drawdown, flag ruin), but as one JIT-compiled
    loop instead of ~1,000 paths x 4 separate numpy calls each (permutation,
    cumsum, maximum.accumulate, concatenate) -- numpy's per-call overhead
    dominates for small trade counts, exactly the case Numba is built for.
    """
    np.random.seed(seed)
    n = len(pnls)
    drawdowns = np.empty(n_paths)
    ruin_count = 0
    for p in range(n_paths):
        path = np.random.permutation(pnls)
        equity = 0.0
        running_max = 0.0
        min_dd = 0.0
        min_equity = 0.0
        for i in range(n):
            equity += path[i]
            if equity > running_max:
                running_max = equity
            dd = equity - running_max
            if dd < min_dd:
                min_dd = dd
            if equity < min_equity:
                min_equity = equity
        drawdowns[p] = min_dd
        if min_equity <= ruin_threshold:
            ruin_count += 1
    return drawdowns, ruin_count


# ==============================================================================
# SECTION: TECHNICAL INDICATOR LIBRARY (50+ MODULAR INDICATORS)
# ==============================================================================

class IndicatorLibrary:
    """
    Registry of 50+ modular technical indicators. Every indicator is a pure
    function (df, **params) -> pd.Series, and every integer parameter is
    trainable within [INDICATOR_PARAM_MIN, INDICATOR_PARAM_MAX] by the
    optimizer. Composite/nested indicators combine two base indicators.
    """

    # ---- moving averages -----------------------------------------------
    @staticmethod
    def sma(df, period=20, col="Close"):
        return df[col].rolling(period, min_periods=period).mean()

    @staticmethod
    def ema(df, period=20, col="Close"):
        return df[col].ewm(span=period, adjust=False, min_periods=period).mean()

    @staticmethod
    def wma(df, period=20, col="Close"):
        vals = df[col].to_numpy(dtype=np.float64)
        out = _wma_numba(vals, period)
        return pd.Series(out, index=df.index)

    @staticmethod
    def dema(df, period=20, col="Close"):
        e1 = IndicatorLibrary.ema(df, period, col)
        e2 = e1.ewm(span=period, adjust=False, min_periods=period).mean()
        return 2 * e1 - e2

    @staticmethod
    def tema(df, period=20, col="Close"):
        e1 = IndicatorLibrary.ema(df, period, col)
        e2 = e1.ewm(span=period, adjust=False, min_periods=period).mean()
        e3 = e2.ewm(span=period, adjust=False, min_periods=period).mean()
        return 3 * e1 - 3 * e2 + e3

    @staticmethod
    def hull_ma(df, period=20, col="Close"):
        half = max(1, period // 2)
        sqrt_p = max(1, int(math.sqrt(period)))
        wma_half = IndicatorLibrary.wma(df, half, col)
        wma_full = IndicatorLibrary.wma(df, period, col)
        raw = 2 * wma_half - wma_full
        out = _wma_numba(raw.to_numpy(dtype=np.float64), sqrt_p)
        return pd.Series(out, index=df.index)

    @staticmethod
    def kama(df, period=10, col="Close"):
        price_vals = df[col].to_numpy(dtype=np.float64)
        out = _kama_numba(price_vals, period)
        return pd.Series(out, index=df.index)

    # ---- momentum / oscillators -----------------------------------------
    @staticmethod
    def rsi(df, period=14, col="Close"):
        delta = df[col].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50)

    @staticmethod
    def macd_line(df, fast=12, slow=26, col="Close"):
        return IndicatorLibrary.ema(df, fast, col) - IndicatorLibrary.ema(df, slow, col)

    @staticmethod
    def macd_signal(df, fast=12, slow=26, signal=9, col="Close"):
        macd = IndicatorLibrary.macd_line(df, fast, slow, col)
        return macd.ewm(span=signal, adjust=False, min_periods=signal).mean()

    @staticmethod
    def macd_hist(df, fast=12, slow=26, signal=9, col="Close"):
        return IndicatorLibrary.macd_line(df, fast, slow, col) - IndicatorLibrary.macd_signal(df, fast, slow, signal, col)

    @staticmethod
    def momentum(df, period=10, col="Close"):
        return df[col] - df[col].shift(period)

    @staticmethod
    def roc(df, period=10, col="Close"):
        return df[col].pct_change(period) * 100

    @staticmethod
    def stochastic_k(df, period=14):
        low_min = df["Low"].rolling(period).min()
        high_max = df["High"].rolling(period).max()
        return 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)

    @staticmethod
    def stochastic_d(df, period=14, smooth=3):
        return IndicatorLibrary.stochastic_k(df, period).rolling(smooth).mean()

    @staticmethod
    def williams_r(df, period=14):
        high_max = df["High"].rolling(period).max()
        low_min = df["Low"].rolling(period).min()
        return -100 * (high_max - df["Close"]) / (high_max - low_min).replace(0, np.nan)

    @staticmethod
    def cci(df, period=20):
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        sma_tp = tp.rolling(period).mean()
        mad_vals = _rolling_mad_numba(tp.to_numpy(dtype=np.float64), period)
        mad = pd.Series(mad_vals, index=df.index)
        return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

    @staticmethod
    def cmo(df, period=14, col="Close"):
        delta = df[col].diff()
        gain = delta.clip(lower=0).rolling(period).sum()
        loss = (-delta.clip(upper=0)).rolling(period).sum()
        return 100 * (gain - loss) / (gain + loss).replace(0, np.nan)

    @staticmethod
    def trix(df, period=15, col="Close"):
        e1 = df[col].ewm(span=period, adjust=False, min_periods=period).mean()
        e2 = e1.ewm(span=period, adjust=False, min_periods=period).mean()
        e3 = e2.ewm(span=period, adjust=False, min_periods=period).mean()
        return e3.pct_change() * 100

    @staticmethod
    def ultimate_osc(df, p1=7, p2=14, p3=28):
        prior_close = df["Close"].shift(1)
        bp = df["Close"] - pd.concat([df["Low"], prior_close], axis=1).min(axis=1)
        tr = pd.concat([df["High"], prior_close], axis=1).max(axis=1) - pd.concat([df["Low"], prior_close], axis=1).min(axis=1)
        avg1 = bp.rolling(p1).sum() / tr.rolling(p1).sum().replace(0, np.nan)
        avg2 = bp.rolling(p2).sum() / tr.rolling(p2).sum().replace(0, np.nan)
        avg3 = bp.rolling(p3).sum() / tr.rolling(p3).sum().replace(0, np.nan)
        return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7

    @staticmethod
    def awesome_osc(df, fast=5, slow=34):
        mid = (df["High"] + df["Low"]) / 2
        return mid.rolling(fast).mean() - mid.rolling(slow).mean()

    @staticmethod
    def dpo(df, period=20, col="Close"):
        shift = period // 2 + 1
        sma = df[col].rolling(period).mean()
        return df[col].shift(shift) - sma

    # ---- volatility / range -----------------------------------------------
    @staticmethod
    def true_range(df):
        prior_close = df["Close"].shift(1)
        return pd.concat([
            df["High"] - df["Low"],
            (df["High"] - prior_close).abs(),
            (df["Low"] - prior_close).abs(),
        ], axis=1).max(axis=1)

    @staticmethod
    def atr(df, period=14):
        return IndicatorLibrary.true_range(df).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    @staticmethod
    def bollinger_mid(df, period=20, col="Close"):
        return df[col].rolling(period).mean()

    @staticmethod
    def bollinger_upper(df, period=20, num_std=2, col="Close"):
        mid = IndicatorLibrary.bollinger_mid(df, period, col)
        std = df[col].rolling(period).std()
        return mid + num_std * std

    @staticmethod
    def bollinger_lower(df, period=20, num_std=2, col="Close"):
        mid = IndicatorLibrary.bollinger_mid(df, period, col)
        std = df[col].rolling(period).std()
        return mid - num_std * std

    @staticmethod
    def bollinger_bandwidth(df, period=20, num_std=2, col="Close"):
        upper = IndicatorLibrary.bollinger_upper(df, period, num_std, col)
        lower = IndicatorLibrary.bollinger_lower(df, period, num_std, col)
        mid = IndicatorLibrary.bollinger_mid(df, period, col)
        return (upper - lower) / mid.replace(0, np.nan)

    @staticmethod
    def keltner_upper(df, period=20, atr_mult=2):
        mid = IndicatorLibrary.ema(df, period)
        return mid + atr_mult * IndicatorLibrary.atr(df, period)

    @staticmethod
    def keltner_lower(df, period=20, atr_mult=2):
        mid = IndicatorLibrary.ema(df, period)
        return mid - atr_mult * IndicatorLibrary.atr(df, period)

    @staticmethod
    def historical_volatility(df, period=20, col="Close"):
        log_ret = np.log(df[col] / df[col].shift(1))
        return log_ret.rolling(period).std() * math.sqrt(252)

    @staticmethod
    def chaikin_volatility(df, period=10):
        hl = df["High"] - df["Low"]
        ema_hl = hl.ewm(span=period, adjust=False, min_periods=period).mean()
        return ema_hl.pct_change(period) * 100

    @staticmethod
    def donchian_upper(df, period=20):
        return df["High"].rolling(period).max()

    @staticmethod
    def donchian_lower(df, period=20):
        return df["Low"].rolling(period).min()

    @staticmethod
    def donchian_mid(df, period=20):
        return (IndicatorLibrary.donchian_upper(df, period) + IndicatorLibrary.donchian_lower(df, period)) / 2

    # ---- volume-based -------------------------------------------------------
    @staticmethod
    def obv(df):
        direction = np.sign(df["Close"].diff()).fillna(0)
        return (direction * df["Volume"]).cumsum()

    @staticmethod
    def cmf(df, period=20):
        mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
        mfv = mfm * df["Volume"]
        return mfv.rolling(period).sum() / df["Volume"].rolling(period).sum().replace(0, np.nan)

    @staticmethod
    def mfi(df, period=14):
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        raw_money_flow = tp * df["Volume"]
        pos_flow = raw_money_flow.where(tp > tp.shift(1), 0)
        neg_flow = raw_money_flow.where(tp < tp.shift(1), 0)
        pos_sum = pos_flow.rolling(period).sum()
        neg_sum = neg_flow.rolling(period).sum()
        mfr = pos_sum / neg_sum.replace(0, np.nan)
        return (100 - (100 / (1 + mfr))).fillna(50)

    @staticmethod
    def vwap_deviation(df, period=20):
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        vwap = (tp * df["Volume"]).rolling(period).sum() / df["Volume"].rolling(period).sum().replace(0, np.nan)
        return (df["Close"] - vwap) / vwap.replace(0, np.nan)

    @staticmethod
    def volume_roc(df, period=10):
        return df["Volume"].pct_change(period) * 100

    @staticmethod
    def pure_buy_volume_ratio(df, period=10):
        v_buy = CandleTransform.pure_buying_volume(df)
        return v_buy.rolling(period).sum() / df["Volume"].rolling(period).sum().replace(0, np.nan)

    @staticmethod
    def ad_line(df):
        clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
        return (clv * df["Volume"]).cumsum()

    # ---- trend strength -------------------------------------------------------
    @staticmethod
    def adx(df, period=14):
        up_move = df["High"].diff()
        down_move = -df["Low"].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr = IndicatorLibrary.true_range(df)
        atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    @staticmethod
    def plus_di(df, period=14):
        up_move = df["High"].diff()
        down_move = -df["Low"].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        tr = IndicatorLibrary.true_range(df)
        atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        return 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)

    @staticmethod
    def minus_di(df, period=14):
        up_move = df["High"].diff()
        down_move = -df["Low"].diff()
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr = IndicatorLibrary.true_range(df)
        atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        return 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)

    @staticmethod
    def aroon_up(df, period=25):
        out = _aroon_numba(df["High"].to_numpy(dtype=np.float64), period, True)
        return pd.Series(out, index=df.index)

    @staticmethod
    def aroon_down(df, period=25):
        out = _aroon_numba(df["Low"].to_numpy(dtype=np.float64), period, False)
        return pd.Series(out, index=df.index)

    @staticmethod
    def vortex_plus(df, period=14):
        vm_plus = (df["High"] - df["Low"].shift(1)).abs()
        tr_sum = IndicatorLibrary.true_range(df).rolling(period).sum()
        return vm_plus.rolling(period).sum() / tr_sum.replace(0, np.nan)

    @staticmethod
    def vortex_minus(df, period=14):
        vm_minus = (df["Low"] - df["High"].shift(1)).abs()
        tr_sum = IndicatorLibrary.true_range(df).rolling(period).sum()
        return vm_minus.rolling(period).sum() / tr_sum.replace(0, np.nan)

    # ---- price-structure -------------------------------------------------------
    @staticmethod
    def price_channel_pos(df, period=20):
        upper = df["High"].rolling(period).max()
        lower = df["Low"].rolling(period).min()
        return (df["Close"] - lower) / (upper - lower).replace(0, np.nan)

    @staticmethod
    def pct_from_high(df, period=252):
        roll_high = df["Close"].rolling(period).max()
        return (df["Close"] - roll_high) / roll_high.replace(0, np.nan) * 100

    @staticmethod
    def pct_from_low(df, period=252):
        roll_low = df["Close"].rolling(period).min()
        return (df["Close"] - roll_low) / roll_low.replace(0, np.nan) * 100

    @staticmethod
    def zscore(df, period=20, col="Close"):
        mean = df[col].rolling(period).mean()
        std = df[col].rolling(period).std()
        return (df[col] - mean) / std.replace(0, np.nan)

    @staticmethod
    def linreg_slope(df, period=20, col="Close"):
        out = _linreg_slope_numba(df[col].to_numpy(dtype=np.float64), period)
        return pd.Series(out, index=df.index)

    # ---- composite / nested indicators -----------------------------------------
    @staticmethod
    def ma_ratio(df, fast=10, slow=50, col="Close"):
        """Composite: fast SMA / slow SMA -- classic dual-MA momentum ratio."""
        return IndicatorLibrary.sma(df, fast, col) / IndicatorLibrary.sma(df, slow, col).replace(0, np.nan)

    @staticmethod
    def rsi_of_roc(df, roc_period=10, rsi_period=14, col="Close"):
        """Composite: RSI computed on top of Rate-of-Change (nested indicator)."""
        roc_series = IndicatorLibrary.roc(df, roc_period, col)
        tmp = df.copy()
        tmp["_roc"] = roc_series
        return IndicatorLibrary.rsi(tmp, rsi_period, col="_roc")

    @staticmethod
    def macd_hist_of_ha(df, fast=12, slow=26, signal=9):
        """Composite: MACD histogram computed on Heikin-Ashi close instead of raw close."""
        ha = CandleTransform.heikin_ashi(df)
        return IndicatorLibrary.macd_hist(ha, fast, slow, signal, col="Close")

    @staticmethod
    def atr_normalized_momentum(df, mom_period=10, atr_period=14, col="Close"):
        """Composite: raw momentum normalized by ATR -- volatility-adjusted momentum."""
        mom = IndicatorLibrary.momentum(df, mom_period, col)
        atr = IndicatorLibrary.atr(df, atr_period)
        return mom / atr.replace(0, np.nan)

    @staticmethod
    def volume_weighted_rsi(df, period=14, col="Close"):
        """Composite: RSI weighted by pure-buy-volume ratio."""
        rsi = IndicatorLibrary.rsi(df, period, col)
        vratio = IndicatorLibrary.pure_buy_volume_ratio(df, period)
        return rsi * (0.5 + vratio.clip(0, 1))

    @staticmethod
    def bb_percent_b(df, period=20, num_std=2, col="Close"):
        upper = IndicatorLibrary.bollinger_upper(df, period, num_std, col)
        lower = IndicatorLibrary.bollinger_lower(df, period, num_std, col)
        return (df[col] - lower) / (upper - lower).replace(0, np.nan)

    @staticmethod
    def chandelier_exit_long(df, period=22, atr_mult=3):
        """Composite: highest-high minus ATR multiple -- also reused directly by ExitEngine."""
        highest = df["High"].rolling(period).max()
        return highest - atr_mult * IndicatorLibrary.atr(df, period)

    @staticmethod
    def supertrend_direction(df, period=10, atr_mult=3):
        """Composite: simplified SuperTrend directional flag (+1/-1)."""
        hl2 = (df["High"] + df["Low"]) / 2
        atr = IndicatorLibrary.atr(df, period)
        upper_band = (hl2 + atr_mult * atr).to_numpy(dtype=np.float64)
        lower_band = (hl2 - atr_mult * atr).to_numpy(dtype=np.float64)
        close_vals = df["Close"].to_numpy(dtype=np.float64)
        direction = _supertrend_numba(close_vals, upper_band, lower_band)
        return pd.Series(direction, index=df.index)

    @staticmethod
    def ha_streak(df, lookback=10):
        """Composite: count of consecutive bullish Heikin-Ashi candles in lookback window."""
        ha = CandleTransform.heikin_ashi(df)
        bullish = (ha["Close"] > ha["Open"]).astype(int)
        return bullish.rolling(lookback).sum()

    # Registry mapping indicator name -> (callable, {param_name: (min,max,default)})
    REGISTRY: Dict[str, Dict[str, Any]] = {}


# Named parameter-range groups grounded in standard technical-analysis practice.
# Previously every trainable period used one blanket (INDICATOR_PARAM_MIN,
# INDICATOR_PARAM_MAX) = (5, 300) range regardless of indicator -- an Optuna
# trial sampling ADX(287) was exactly as likely as ADX(14), even though no
# practitioner uses ADX above ~30. That's pure wasted search budget: TPE has to
# spend trials learning "high ADX periods are useless" instead of ever getting
# to explore the region that matters. Tightening oscillators/ADX/Aroon to their
# real practical range concentrates the search where it can actually pay off.
# Moving averages and long-lookback trend filters are the deliberate exception:
# kept WIDER than common practice (which tops out ~200 for MAs) specifically to
# let the optimizer discover unconventional long-horizon windows the tightened
# groups would never let it try.
PARAM_RANGE_GROUPS: Dict[str, Tuple[int, int]] = {
    "moving_average": (5, 250),     # SMA/EMA/WMA/DEMA/TEMA/Hull/KAMA -- kept wide on purpose
    "fast_oscillator": (2, 50),     # RSI/Stoch/CCI/Williams %R/CMO/MFI/TRIX -- RSI(14)/(2) bookend real usage
    "adx_family": (5, 30),          # ADX/+DI/-DI/Vortex -- Wilder's original is 14, rarely tuned past 30
    "aroon": (10, 60),              # default 25, rarely tuned far outside this
    "volatility_band": (5, 60),     # ATR/Bollinger/Keltner/Donchian/hist&Chaikin vol -- ATR(14)/Bollinger(20) etc.
    "momentum_family": (5, 60),     # momentum/ROC/DPO/z-score/linreg slope -- short-to-medium lookbacks
    "long_lookback": (50, 300),     # 252-day high/low, regime filters -- meant to span up to ~1 trading year
    "streak_count": (3, 20),        # ha_streak -- small integer counts
}

# Per-indicator, per-parameter-name overrides where the group default isn't right
# on its own (MACD's three periods each need a distinct sub-range, etc). Falls
# back to the indicator's assigned group for any param not listed here.
PARAM_OVERRIDES: Dict[str, Dict[str, Tuple[int, int]]] = {
    "macd_line":       {"fast": (5, 20), "slow": (15, 60)},
    "macd_signal":     {"fast": (5, 20), "slow": (15, 60), "signal": (3, 15)},
    "macd_hist":       {"fast": (5, 20), "slow": (15, 60), "signal": (3, 15)},
    "macd_hist_of_ha": {"fast": (5, 20), "slow": (15, 60), "signal": (3, 15)},
    "ma_ratio":        {"fast": (5, 30), "slow": (20, 250)},
    "ultimate_osc":    {"p1": (5, 10), "p2": (10, 20), "p3": (20, 40)},
    "awesome_osc":     {"fast": (3, 10), "slow": (20, 40)},
    "stochastic_d":    {"smooth": (2, 10)},
    "rsi_of_roc":      {"roc_period": (5, 30)},
    "atr_normalized_momentum": {"mom_period": (5, 30)},
}

# Indicator -> parameter-range group. Anything not listed defaults to
# "momentum_family" (a moderate, non-extreme range) rather than silently
# reverting to the old wide blanket range.
INDICATOR_GROUPS: Dict[str, str] = {
    "sma": "moving_average", "ema": "moving_average", "wma": "moving_average",
    "dema": "moving_average", "tema": "moving_average", "hull_ma": "moving_average",
    "kama": "moving_average",
    "rsi": "fast_oscillator", "stochastic_k": "fast_oscillator", "stochastic_d": "fast_oscillator",
    "williams_r": "fast_oscillator", "cci": "fast_oscillator", "cmo": "fast_oscillator",
    "mfi": "fast_oscillator", "trix": "fast_oscillator", "volume_weighted_rsi": "fast_oscillator",
    "rsi_of_roc": "fast_oscillator",  # rsi_period only; roc_period overridden above
    "adx": "adx_family", "plus_di": "adx_family", "minus_di": "adx_family",
    "vortex_plus": "adx_family", "vortex_minus": "adx_family",
    "aroon_up": "aroon", "aroon_down": "aroon",
    "atr": "volatility_band", "bollinger_mid": "volatility_band", "bollinger_upper": "volatility_band",
    "bollinger_lower": "volatility_band", "bollinger_bandwidth": "volatility_band",
    "keltner_upper": "volatility_band", "keltner_lower": "volatility_band",
    "historical_volatility": "volatility_band", "chaikin_volatility": "volatility_band",
    "donchian_upper": "volatility_band", "donchian_lower": "volatility_band", "donchian_mid": "volatility_band",
    "cmf": "volatility_band", "bb_percent_b": "volatility_band", "chandelier_exit_long": "volatility_band",
    "supertrend_direction": "volatility_band",
    "atr_normalized_momentum": "momentum_family",  # mom_period overridden above; atr_period uses this group
    "momentum": "momentum_family", "roc": "momentum_family", "dpo": "momentum_family",
    "price_channel_pos": "momentum_family", "zscore": "momentum_family", "linreg_slope": "momentum_family",
    "vwap_deviation": "momentum_family", "volume_roc": "momentum_family",
    "pure_buy_volume_ratio": "momentum_family",
    "pct_from_high": "long_lookback", "pct_from_low": "long_lookback",
    "ha_streak": "streak_count",
}


def _param_range(indicator_name: str, param_name: str) -> Tuple[int, int]:
    """Resolve the (lo, hi) range for one indicator's parameter, checking the
    per-indicator override first, then falling back to the indicator's group."""
    override = PARAM_OVERRIDES.get(indicator_name, {}).get(param_name)
    if override is not None:
        return override
    group = INDICATOR_GROUPS.get(indicator_name, "momentum_family")
    return PARAM_RANGE_GROUPS[group]


def _build_registry():
    """
    Populate IndicatorLibrary.REGISTRY reflectively with trainable param ranges.
    See PARAM_RANGE_GROUPS/PARAM_OVERRIDES/INDICATOR_GROUPS above for the
    grouping rationale -- ranges are tailored per indicator family rather than
    one blanket range for everything.
    """
    specs = {
        "sma": {"period": 20}, "ema": {"period": 20}, "wma": {"period": 20},
        "dema": {"period": 20}, "tema": {"period": 20}, "hull_ma": {"period": 20},
        "kama": {"period": 10}, "rsi": {"period": 14}, "macd_line": {"fast": 12, "slow": 26},
        "macd_signal": {"fast": 12, "slow": 26, "signal": 9}, "macd_hist": {"fast": 12, "slow": 26, "signal": 9},
        "momentum": {"period": 10}, "roc": {"period": 10}, "stochastic_k": {"period": 14},
        "stochastic_d": {"period": 14, "smooth": 3}, "williams_r": {"period": 14}, "cci": {"period": 20},
        "cmo": {"period": 14}, "trix": {"period": 15}, "ultimate_osc": {"p1": 7, "p2": 14, "p3": 28},
        "awesome_osc": {"fast": 5, "slow": 34}, "dpo": {"period": 20}, "atr": {"period": 14},
        "bollinger_mid": {"period": 20}, "bollinger_upper": {"period": 20}, "bollinger_lower": {"period": 20},
        "bollinger_bandwidth": {"period": 20}, "keltner_upper": {"period": 20}, "keltner_lower": {"period": 20},
        "historical_volatility": {"period": 20}, "chaikin_volatility": {"period": 10},
        "donchian_upper": {"period": 20}, "donchian_lower": {"period": 20}, "donchian_mid": {"period": 20},
        "cmf": {"period": 20}, "mfi": {"period": 14}, "vwap_deviation": {"period": 20},
        "volume_roc": {"period": 10}, "pure_buy_volume_ratio": {"period": 10}, "obv": {}, "ad_line": {},
        "adx": {"period": 14}, "plus_di": {"period": 14}, "minus_di": {"period": 14},
        "aroon_up": {"period": 25}, "aroon_down": {"period": 25}, "vortex_plus": {"period": 14},
        "vortex_minus": {"period": 14}, "price_channel_pos": {"period": 20}, "pct_from_high": {"period": 252},
        "pct_from_low": {"period": 252}, "zscore": {"period": 20}, "linreg_slope": {"period": 20},
        "ma_ratio": {"fast": 10, "slow": 50}, "rsi_of_roc": {"roc_period": 10, "rsi_period": 14},
        "macd_hist_of_ha": {"fast": 12, "slow": 26, "signal": 9},
        "atr_normalized_momentum": {"mom_period": 10, "atr_period": 14},
        "volume_weighted_rsi": {"period": 14}, "bb_percent_b": {"period": 20},
        "chandelier_exit_long": {"period": 22}, "supertrend_direction": {"period": 10},
        "ha_streak": {"lookback": 10},
    }
    for name, defaults in specs.items():
        fn = getattr(IndicatorLibrary, name)
        param_ranges = {}
        for p, dv in defaults.items():
            lo, hi = _param_range(name, p)
            dv = int(np.clip(dv, lo, hi))  # keep the traditional default, clipped into the new range
            param_ranges[p] = (lo, hi, dv)
        IndicatorLibrary.REGISTRY[name] = {"fn": fn, "params": param_ranges}


_build_registry()
assert len(IndicatorLibrary.REGISTRY) >= 50, f"Indicator library must have >=50 indicators, has {len(IndicatorLibrary.REGISTRY)}"


# ==============================================================================
# SECTION: MARKET DATA PIPELINE (yfinance download + parquet cache)
# ==============================================================================

class DataPipeline:
    """Downloads (or loads cached) 15Y daily OHLCV data for the 150-ticker universe."""

    def __init__(self, universe: List[str] = UNIVERSE, years: int = 15, cache_dir: str = CACHE_DIR):
        self.universe = universe
        self.years = years
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, ticker: str) -> str:
        return os.path.join(self.cache_dir, f"{ticker.replace('.', '_')}.parquet")

    def download_universe(self, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Download once, cache to pickle, and return {ticker: OHLCV DataFrame}.
        Shows real-time progress bar in terminal.
        """
        data: Dict[str, pd.DataFrame] = {}
        network_ok = True

        # CHANGED: Wrap self.universe with tqdm progress bar
        progress_bar = tqdm(
            self.universe, 
            desc="📥 Loading Universe Data", 
            unit="ticker", 
            ncols=100
        )

        for ticker in progress_bar:
            # Update current status on progress bar
            progress_bar.set_postfix_str(f"Ticker: {ticker}")
            
            cache_path = self._cache_path(ticker)
            if not force_refresh and os.path.exists(cache_path):
                try:
                    data[ticker] = pd.read_pickle(cache_path)
                    continue
                except Exception:
                    pass  # fall through and re-fetch

            df = None
            if network_ok:
                try:
                    import yfinance as yf
                    raw = yf.download(
                        ticker, period=f"{self.years}y", interval="1d",
                        progress=False, auto_adjust=True, timeout=10,
                    )
                    if raw is not None and len(raw) > 0:
                        if isinstance(raw.columns, pd.MultiIndex):
                            raw.columns = raw.columns.get_level_values(0)
                        df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
                except Exception as e:
                    # Clear line brief for clean display
                    progress_bar.write(f"⚠️ yfinance unreachable ({e}); using synthetic fallback.")
                    network_ok = False

            if (df is None or len(df) == 0) and USE_SYNTHETIC_FALLBACK:
                df = self._synthesize(ticker)

            if df is None or len(df) == 0:
                progress_bar.write(f"⚠️ No data available for {ticker}; skipping.")
                continue

            df.to_pickle(cache_path)
            data[ticker] = df

        logger.info(f"Data pipeline ready: {len(data)}/{len(self.universe)} tickers loaded.")
        return data

    def _synthesize(self, ticker: str, start: str = None) -> pd.DataFrame:
        """
        Seeded synthetic OHLCV generator (geometric brownian motion + mild
        momentum autocorrelation) used only when live data is unreachable, so
        the rest of the pipeline remains fully testable offline.
        """
        # NOTE: Python's built-in hash() is randomized per-process (PYTHONHASHSEED) for
        # security reasons -- using it here would silently make this "seeded" generator
        # produce DIFFERENT synthetic data on every fresh process run, even for the exact
        # same ticker/years, which would quietly break reproducibility of every comparison
        # in this file (including the memory/re-validation feature in run_full_pipeline,
        # which specifically re-runs a prior config expecting the same data). hashlib's
        # md5 is stable across processes and Python versions, which is what we need here.
        stable_hash = int(hashlib.md5(ticker.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(stable_hash)
        requested_days = self.years * 252
        end = pd.Timestamp.today().normalize()
        # NOTE: pd.bdate_range(end=..., periods=N) can return fewer than N rows when
        # `end` itself isn't a business day (e.g. this code runs on a Saturday/Sunday
        # with no yfinance access) -- verified: bdate_range(end=<Saturday>, periods=2016)
        # returns 2015 rows, which used to crash DataFrame construction below with a
        # length mismatch. Derive n_days from what bdate_range actually returned
        # instead of assuming it matches the request.
        dates = pd.bdate_range(end=end, periods=requested_days)
        n_days = len(dates)
        mu, sigma = 0.0004, 0.018
        shocks = rng.normal(mu, sigma, n_days)
        # inject mild positive autocorrelation to make momentum strategies non-trivial
        for i in range(1, n_days):
            shocks[i] += 0.05 * shocks[i - 1]
        log_prices = np.cumsum(shocks) + math.log(rng.uniform(50, 3000))
        close = np.exp(log_prices)
        open_ = close * (1 + rng.normal(0, 0.003, n_days))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n_days)))
        volume = rng.integers(100_000, 5_000_000, n_days).astype(float)
        df = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates
        )
        return df


# ==============================================================================
# SECTION: STRATEGY COMPOSER (ENTRY / EXIT SIGNAL GENERATION)
# ==============================================================================

# Indicator names GPUIndicatorBatch can compute for the whole universe at once.
# Deliberately excludes EMA/ATR/RSI/MACD and anything else built on a recursive
# EWM chain: those require a Python-level loop over ~3,750 trading days (each
# day's update depends on the previous), which on a GPU means ~3,750 sequential
# kernel dispatches. Kernel launch latency (~5-20us each) dominates over the
# tiny amount of real math per step for a universe this size, the same reason
# RNNs are notoriously hard to accelerate on GPUs -- this is an architectural
# mismatch, not something that goes away with better hardware. Measured directly:
# even a real CUDA GPU won't out-launch pandas' already-C-optimized .ewm().
#
# What's left here are genuine single-shot batched tensor ops (rolling windows
# via one torch.unfold() call, no per-day Python loop) -- the actual class of
# operation GPUs are good at. Whether THIS category wins on a given machine is
# checked empirically at runtime by _gpu_beats_cpu() below rather than assumed.
GPU_INDICATOR_METHODS = {
    "sma", "momentum", "roc", "wma",
    "bollinger_mid", "bollinger_upper", "bollinger_lower", "bollinger_bandwidth",
    "donchian_upper", "donchian_lower", "donchian_mid",
    "stochastic_k", "williams_r", "historical_volatility",
}


class StrategyComposer:
    """Evaluates up to 10 entry + 10 exit indicator slots into boolean signals."""

    @staticmethod
    def _indicator_series(df: pd.DataFrame, slot: IndicatorSlot, ticker: Optional[str] = None,
                           gpu_cache: Optional[Dict[Tuple[str, Tuple], Dict[str, pd.Series]]] = None) -> pd.Series:
        if gpu_cache is not None and ticker is not None and slot.name in GPU_INDICATOR_METHODS:
            spec = IndicatorLibrary.REGISTRY[slot.name]["params"]
            kwargs = {k: v for k, v in slot.params.items() if k in spec}
            key = (slot.name, tuple(sorted(kwargs.items())))
            cached = gpu_cache.get(key)
            if cached is not None and ticker in cached:
                return cached[ticker]
        spec = IndicatorLibrary.REGISTRY[slot.name]
        fn = spec["fn"]
        # only pass params the function actually accepts / that were tuned
        kwargs = {k: v for k, v in slot.params.items() if k in spec["params"]}
        return fn(df, **kwargs)

    @staticmethod
    def _slot_signal(df: pd.DataFrame, slot: IndicatorSlot, ticker: Optional[str] = None,
                      gpu_cache: Optional[Dict[Tuple[str, Tuple], Dict[str, pd.Series]]] = None) -> pd.Series:
        if slot.is_empty():
            return pd.Series(False, index=df.index)
        ind = StrategyComposer._indicator_series(df, slot, ticker=ticker, gpu_cache=gpu_cache)
        close = df["Close"]
        if slot.comparator == "cross_above":
            return (close.shift(1) <= ind.shift(1)) & (close > ind)
        elif slot.comparator == "cross_below":
            return (close.shift(1) >= ind.shift(1)) & (close < ind)
        elif slot.comparator == "greater_than":
            return ind > slot.threshold
        elif slot.comparator == "less_than":
            return ind < slot.threshold
        else:
            raise ValueError(f"Unknown comparator: {slot.comparator}")

    @staticmethod
    def compute_composite_signal(df: pd.DataFrame, slots: List[IndicatorSlot], mode: str = "entry",
                                  ticker: Optional[str] = None,
                                  gpu_cache: Optional[Dict[Tuple[str, Tuple], Dict[str, pd.Series]]] = None) -> pd.Series:
        """
        AND-combine active entry slots (all must agree -- conservative momentum
        confirmation), OR-combine active exit slots (any one triggers an exit).
        Empty slots are ignored entirely.
        """
        active = [s for s in slots if not s.is_empty()]
        if not active:
            return pd.Series(False, index=df.index)
        sigs = [StrategyComposer._slot_signal(df, s, ticker=ticker, gpu_cache=gpu_cache) for s in active]
        stacked = pd.concat(sigs, axis=1)
        return stacked.all(axis=1) if mode == "entry" else stacked.any(axis=1)

    @staticmethod
    def prepare_source_df(raw_df: pd.DataFrame, use_heikin_ashi: bool) -> pd.DataFrame:
        """Optionally swap in Heikin-Ashi OHLC as the working candle series."""
        if not use_heikin_ashi:
            return raw_df
        ha = CandleTransform.heikin_ashi(raw_df)
        ha["Volume"] = raw_df.get("Volume", pd.Series(0, index=raw_df.index))
        return ha


# ==============================================================================
# SECTION: WATCHLIST ENGINE (6 PRIORITY RANKING MODELS)
# ==============================================================================

@dataclass
class WatchlistEntry:
    ticker: str
    signal_date: pd.Timestamp
    signal_price: float
    added_date: pd.Timestamp


class WatchlistEngine:
    """
    Ranks tickers waiting for capital using one of 6 selectable heuristic
    priority models, or (runtime-only, not part of Optuna's search space
    since the model doesn't exist during search) a fitted ML ranker.
    Higher score => higher execution priority.
    """

    MODELS = (
        "weighted_score", "unweighted_score", "proximity_model",
        "pure_momentum", "pure_buying_volume", "heikin_ashi_strength",
    )

    def __init__(self, cfg: StrategyConfig, ml_ranker: Optional["MLWatchlistRanker"] = None):
        self.cfg = cfg
        self.ml_ranker = ml_ranker

    def rank(
        self,
        watchlist: List[WatchlistEntry],
        current_date: pd.Timestamp,
        market_snapshot: Dict[str, Dict[str, float]],
    ) -> List[WatchlistEntry]:
        """market_snapshot[ticker] holds precomputed same-day features:
        {'price', 'ma', 'momentum', 'v_buy', 'ha_strength'}"""
        if not watchlist:
            return []
        model = self.cfg.watchlist_model
        scored: List[Tuple[float, WatchlistEntry]] = []

        prices_now = [market_snapshot.get(e.ticker, {}).get("price", e.signal_price) for e in watchlist]
        ages = [(current_date - e.added_date).days for e in watchlist]
        max_age = max(ages) if ages else 1
        max_age = max_age or 1

        for e, p_now, age in zip(watchlist, prices_now, ages):
            snap = market_snapshot.get(e.ticker, {})
            if model == "weighted_score":
                price_dist = 1.0 - min(abs(p_now - e.signal_price) / max(e.signal_price, 1e-6), 1.0)
                age_norm = age / max_age
                score = (self.cfg.watchlist_weight_price_dist * price_dist +
                         self.cfg.watchlist_weight_age * age_norm)
            elif model == "unweighted_score":
                score = age  # oldest-first, no weighting
            elif model == "proximity_model":
                ma = snap.get("ma", p_now)
                score = -abs(p_now - ma) / max(ma, 1e-6)  # closer to MA => higher (less negative)
            elif model == "pure_momentum":
                score = snap.get("momentum", 0.0)
            elif model == "pure_buying_volume":
                score = snap.get("v_buy", 0.0)
            elif model == "heikin_ashi_strength":
                score = snap.get("ha_strength", 0.0)
            elif model == "ml_ranked":
                if self.ml_ranker is None or self.ml_ranker.model is None:
                    score = 0.0  # no fitted model available -- degrades to a no-op ranking (insertion order)
                else:
                    ma = snap.get("ma", p_now)
                    # Mirrors the feature semantics MLWatchlistRanker.build_training_set used:
                    # price_dist there is "drift since the anchor point 'age' days ago" -- here
                    # that anchor is the price when the ticker first entered the watchlist.
                    features = {
                        "price_dist": (p_now - e.signal_price) / max(e.signal_price, 1e-6),
                        "age": float(age),
                        "ma_dist": (p_now - ma) / max(ma, 1e-6),
                        "momentum": snap.get("momentum", 0.0),
                        "v_buy": snap.get("v_buy", 0.0),
                        "ha_strength": snap.get("ha_strength", 0.0),
                    }
                    score = self.ml_ranker.score(features)
            else:
                raise ValueError(f"Unknown watchlist model: {model}")
            scored.append((score, e))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [e for _, e in scored]


# ==============================================================================
# SECTION: FLEXIBLE EXIT ENGINE (TP, TSL, ATR Chandelier, Indicator Exits)
# ==============================================================================

@dataclass
class Position:
    ticker: str
    shares: float
    entry_price: float
    entry_date: pd.Timestamp
    allocated_cash: float
    peak_price: float
    sector: str = "UNKNOWN"


class ExitEngine:
    """
    Fully optimizer-driven exit logic. No static % stop-loss exists (removed
    per spec); downside is instead managed via ATR chandelier stops / trailing
    stops / indicator exits, while % take-profit remains available.
    """

    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    def check_exit(
        self, pos: Position, current_price: float, atr_value: float, indicator_exit_signal: bool,
    ) -> Tuple[bool, str]:
        # 1. Percentage take-profit
        if self.cfg.take_profit_pct is not None:
            if current_price >= pos.entry_price * (1 + self.cfg.take_profit_pct):
                return True, "take_profit"

        # 2. Trailing stop-loss (% off running peak)
        if self.cfg.trailing_stop_pct is not None:
            if current_price <= pos.peak_price * (1 - self.cfg.trailing_stop_pct):
                return True, "trailing_stop"

        # 3. ATR-based chandelier exit
        if self.cfg.atr_chandelier_mult is not None and not np.isnan(atr_value):
            chandelier_level = pos.peak_price - self.cfg.atr_chandelier_mult * atr_value
            if current_price <= chandelier_level:
                return True, "atr_chandelier"

        # 4. Composite indicator exit slots (OR-combined upstream)
        if indicator_exit_signal:
            return True, "indicator_exit"

        return False, ""


# ==============================================================================
# SECTION: PORTFOLIO / CASH INJECTION / TRADE EXECUTION ENGINE
# ==============================================================================

class PortfolioEngine:
    """
    Manages cash injections, position sizing, watchlist overflow, and
    transaction-cost-adjusted trade execution with strict T+1 signal-to-fill
    lag (signals at day T close execute at day T+1 price).
    """

    def __init__(
        self, cfg: StrategyConfig, injection_days: Union[List[int], Dict[Tuple[int, int], int]],
        starting_cash: float = 0.0, sector_map: Optional[Dict[str, str]] = None,
        ml_ranker: Optional["MLWatchlistRanker"] = None,
    ):
        self.cfg = cfg
        # Either a fixed set of days-of-month, or a dict mapping (year, month) -> target
        # day for the SIP sensitivity suite's per-month "dynamic" schedules. Do NOT
        # collapse a dict into a set here -- that would destroy the per-month mapping.
        self.injection_days = injection_days if isinstance(injection_days, dict) else set(injection_days)
        self.cash = starting_cash
        self.positions: Dict[str, Position] = {}
        self.watchlist: List[WatchlistEntry] = []
        self.watchlist_engine = WatchlistEngine(cfg, ml_ranker=ml_ranker)
        self.sector_map = sector_map or {}
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.trade_log: List[dict] = []
        self._last_injected_month: Optional[Tuple[int, int]] = None
        self.last_exit_date: Dict[str, pd.Timestamp] = {}  # per-ticker, for cfg.cooldown_days
        # Explicit ledger of every external cash flow (SIP injections today; any future
        # withdrawals/redemptions would append here too). This is the source of truth
        # for XIRR (money-weighted return) and for the exact daily Time-Weighted Return
        # adjustment -- without it we cannot tell "the portfolio grew" apart from
        # "new capital arrived", which is what corrupted the old CAGR/Sharpe numbers.
        self.cash_flows: List[Tuple[pd.Timestamp, float]] = []

    # ---------------------------------------------------------------- cash
    def maybe_inject_cash(self, date: pd.Timestamp):
        """
        injection_days can be:
          - a fixed collection of days-of-month (e.g. [5]) -> injects on the first
            trading day on/after the 5th of every month, or
          - a dict mapping (year, month) -> an exact pd.Timestamp (the fully resolved
            injection date, as produced by RobustnessSuite's schedule builders), or
          - a dict mapping (year, month) -> an int day-of-month (legacy form, still
            resolved on/after that day, tolerant of weekends/holidays).
        Either way we inject at most once per (year, month).
        """
        key = (date.year, date.month)
        if key == self._last_injected_month:
            return
        if isinstance(self.injection_days, dict):
            target = self.injection_days.get(key)
            if target is None:
                return
            if isinstance(target, (pd.Timestamp, np.datetime64)):
                if pd.Timestamp(target) == date:
                    self.cash += MONTHLY_INJECTION
                    self.cash_flows.append((date, MONTHLY_INJECTION))
                    self._last_injected_month = key
            else:
                if date.day >= target:
                    self.cash += MONTHLY_INJECTION
                    self.cash_flows.append((date, MONTHLY_INJECTION))
                    self._last_injected_month = key
        else:
            target_day = min(self.injection_days) if self.injection_days else None
            if target_day is not None and date.day >= target_day:
                self.cash += MONTHLY_INJECTION
                self.cash_flows.append((date, MONTHLY_INJECTION))
                self._last_injected_month = key

    # ------------------------------------------------------------- trading
    def _sector_ok(self, ticker: str, portfolio_value: float, allocation: float) -> bool:
        sector = self.sector_map.get(ticker, "UNKNOWN")
        current_sector_value = sum(
            p.shares * p.entry_price for t, p in self.positions.items() if self.sector_map.get(t) == sector
        )
        if portfolio_value <= 0:
            return True
        return (current_sector_value + allocation) / portfolio_value <= self.cfg.sector_cap_pct

    def try_enter(self, ticker: str, price: float, date: pd.Timestamp, portfolio_value: float,
                  atr_value: Optional[float] = None) -> bool:
        if ticker in self.positions:
            return False
        # Cooldown: skip re-entry into a ticker for cfg.cooldown_days after its last exit --
        # prevents immediate whipsaw re-entry right after being stopped out.
        if self.cfg.cooldown_days > 0:
            last_exit = self.last_exit_date.get(ticker)
            if last_exit is not None and (date - last_exit).days < self.cfg.cooldown_days:
                return False

        if self.cfg.equal_weight_sizing_enabled:
            # Cost-basis equal-weight sizing: Target = (Cash Pool + Invested Cost) / N_eligible,
            # floored at MIN_ALLOCATION. Uses COST basis (sum of shares*entry_price), not
            # mark-to-market value, so unrealized gains on existing positions don't inflate
            # the size of unrelated new trades. N_eligible = current watchlist queue + this
            # candidate, i.e. how many tickers are actually contending for capital today.
            # Mutually exclusive with vol_sizing_enabled (both are sizing strategies; this
            # one takes precedence when both are on, since combining them isn't well-defined).
            invested_cost = sum(p.shares * p.entry_price for p in self.positions.values())
            n_eligible = max(1, len(self.watchlist) + 1)
            allocation = max((self.cash + invested_cost) / n_eligible, MIN_ALLOCATION)
        else:
            allocation = MIN_ALLOCATION
            if self.cfg.vol_sizing_enabled and atr_value and not np.isnan(atr_value) and atr_value > 0:
                # Risk-parity-style sizing: allocate MORE to lower-volatility names (smaller
                # ATR relative to price => larger vol_scalar), never LESS -- the spec's 10k
                # allocation floor is a hard minimum, so this only scales up, capped at 2x.
                target_daily_risk_pct = 0.02
                vol_scalar = float(np.clip((price * target_daily_risk_pct) / atr_value, 1.0, 2.0))
                allocation = MIN_ALLOCATION * vol_scalar
        if self.cash < allocation:
            # can't afford the scaled size -- fall back to the floor allocation if affordable
            if allocation > MIN_ALLOCATION and self.cash >= MIN_ALLOCATION:
                allocation = MIN_ALLOCATION
            else:
                return False
        if self.cfg.sector_cap_pct < 1.0 and not self._sector_ok(ticker, portfolio_value, allocation):
            return False
        cost = allocation * TRANSACTION_COST_PCT
        net_alloc = allocation - cost
        shares = net_alloc / price
        self.cash -= allocation
        self.positions[ticker] = Position(
            ticker=ticker, shares=shares, entry_price=price, entry_date=date,
            allocated_cash=allocation, peak_price=price, sector=self.sector_map.get(ticker, "UNKNOWN"),
        )
        self.trade_log.append({"date": str(date.date()), "ticker": ticker, "action": "BUY",
                                "price": price, "shares": shares, "allocation": allocation, "cash_after": self.cash})
        return True

    def exit_position(self, ticker: str, price: float, date: pd.Timestamp, reason: str):
        pos = self.positions.pop(ticker)
        proceeds = pos.shares * price
        cost = proceeds * TRANSACTION_COST_PCT
        net_proceeds = proceeds - cost
        self.cash += net_proceeds
        pnl = net_proceeds - pos.allocated_cash
        self.last_exit_date[ticker] = date
        self.trade_log.append({
            "date": str(date.date()), "ticker": ticker, "action": "SELL", "reason": reason,
            "price": price, "shares": pos.shares, "pnl": pnl, "cash_after": self.cash,
        })

    def add_to_watchlist(self, ticker: str, price: float, date: pd.Timestamp):
        if any(e.ticker == ticker for e in self.watchlist):
            return
        self.watchlist.append(WatchlistEntry(ticker=ticker, signal_date=date, signal_price=price, added_date=date))

    def process_watchlist(self, date: pd.Timestamp, market_snapshot: Dict[str, Dict[str, float]], portfolio_value: float):
        """Allocate any available cash to the top-ranked watchlist candidate(s)."""
        if not self.watchlist:
            return
        ranked = self.watchlist_engine.rank(self.watchlist, date, market_snapshot)
        still_waiting = []
        for entry in ranked:
            snap = market_snapshot.get(entry.ticker, {})
            price = snap.get("price")
            if price is None:
                still_waiting.append(entry)
                continue
            if self.cash >= MIN_ALLOCATION and entry.ticker not in self.positions:
                if self.try_enter(entry.ticker, price, date, portfolio_value, atr_value=snap.get("atr")):
                    continue
            still_waiting.append(entry)
        self.watchlist = still_waiting

    def mark_to_market(self, date: pd.Timestamp, price_lookup: Dict[str, float]):
        equity = self.cash
        for t, pos in self.positions.items():
            px = price_lookup.get(t, pos.entry_price)
            pos.peak_price = max(pos.peak_price, px)
            equity += pos.shares * px
        self.equity_curve.append((date, equity))


# ==============================================================================
# SECTION: CORE EVENT-DRIVEN BACKTESTER
# ==============================================================================

_GPU_BATCH_CACHE: Dict[int, Any] = {}
_GPU_WORTHWHILE_CACHE: Dict[int, bool] = {}


def _gpu_beats_cpu(market_data: Dict[str, pd.DataFrame], margin: float = 1.15) -> bool:
    """
    Empirically benchmarks GPU-batched vs CPU-pandas on a representative
    rolling-window op (Bollinger upper band, 20-period -- a genuine single-shot
    unfold-based batched tensor op, not a recursive/EWM one) using THIS
    process's actual hardware, and only green-lights GPU indicator routing if
    it wins by more than `margin`. This is deliberately not a hardcoded
    assumption: a CPU-only sandbox should measure GPU losing (torch-on-CPU vs
    pandas' C-optimized rolling implementation) and correctly disable itself;
    a machine with a real CUDA GPU and a large-enough universe should measure
    GPU winning and correctly enable itself. Runs once per market_data object
    and caches the verdict.
    """
    if not GPU_AVAILABLE:
        return False
    key = id(market_data)
    if key in _GPU_WORTHWHILE_CACHE:
        return _GPU_WORTHWHILE_CACHE[key]

    import time
    batch = GPUIndicatorBatch(market_data)
    # warm up (CUDA context / kernel compilation / first-call overhead shouldn't count)
    batch.bollinger_upper(20)

    t0 = time.perf_counter()
    for _ in range(3):
        batch.bollinger_upper(20)
    t_gpu = (time.perf_counter() - t0) / 3

    t0 = time.perf_counter()
    for _ in range(3):
        for t, df in market_data.items():
            IndicatorLibrary.bollinger_upper(df, 20)
    t_cpu = (time.perf_counter() - t0) / 3

    worthwhile = t_cpu > t_gpu * margin
    device_name = str(batch.device)
    logger.info(
        f"GPU calibration: rolling-window op took CPU={t_cpu*1000:.1f}ms vs "
        f"GPU({device_name})={t_gpu*1000:.1f}ms -> "
        f"{'GPU ENABLED for rolling-window indicators' if worthwhile else 'GPU DISABLED (CPU is faster on this hardware/universe size), using pandas/numba instead'}"
    )
    _GPU_WORTHWHILE_CACHE[key] = worthwhile
    _GPU_BATCH_CACHE[key] = batch  # reuse the batch we already built for calibration
    return worthwhile


def _get_gpu_batch(market_data: Dict[str, pd.DataFrame]):
    """
    Returns a cached GPUIndicatorBatch for this market_data object, building it
    (and uploading the whole universe to the GPU) only once, and only if
    _gpu_beats_cpu() has calibrated GPU as actually worthwhile on this
    hardware for this universe size -- see that function's docstring. Without
    this cache, every single Optuna trial's _precompute() would re-stack and
    re-upload the same 150-ticker x 15-year OHLCV tensors to the GPU from
    scratch -- for a few-hundred-trial search that's a few hundred redundant
    uploads of the same data, which would eat most of any real GPU speedup.
    Keyed by id(market_data) rather than a hash because these DataFrames are
    large and never mutated in place once loaded.
    """
    if not _gpu_beats_cpu(market_data):
        return None
    key = id(market_data)
    if key not in _GPU_BATCH_CACHE:
        _GPU_BATCH_CACHE[key] = GPUIndicatorBatch(market_data)
    return _GPU_BATCH_CACHE[key]


class Backtester:
    """
    Orchestrates the day-by-day simulation across the full universe:
    signal generation -> T+1 execution -> exit checks -> watchlist allocation
    -> cash injection -> equity mark-to-market.
    """

    def __init__(
        self, cfg: StrategyConfig, market_data: Dict[str, pd.DataFrame],
        injection_days: List[int], sector_map: Optional[Dict[str, str]] = None,
        benchmark: Optional[pd.Series] = None, ml_ranker: Optional["MLWatchlistRanker"] = None,
    ):
        self.cfg = cfg
        self.market_data = market_data
        self.injection_days = injection_days
        self.sector_map = sector_map or {t: t.split(".")[0][:3] for t in market_data}
        self.benchmark = benchmark  # e.g. a broad index close series for regime filter
        self.ml_ranker = ml_ranker  # only used when cfg.watchlist_model == "ml_ranked"

    # ---------------------------------------------------------- precompute
    def _precompute(self, tickers: List[str]) -> Dict[str, dict]:
        out = {}
        
        # Reuse the cached GPU batch (built once for this market_data, see
        # _get_gpu_batch) rather than re-uploading the whole universe to the
        # GPU on every trial -- only the requested period changes per trial,
        # and computing a new rolling window on already-resident tensors is cheap.
        # GPU batch operates on raw OHLC; Heikin-Ashi mode transforms candles
        # per-ticker first, so it can't use the shared raw-OHLC GPU tensors and
        # always falls back to the per-ticker pandas/numba path below.
        gpu_batch = _get_gpu_batch(self.market_data) if not self.cfg.use_heikin_ashi else None

        # ATR is EWM/recursive (Wilder smoothing) -- always CPU/pandas, see the note
        # on GPU_INDICATOR_METHODS above for why. SMA and momentum are genuine
        # single-shot batched ops and stay GPU-routed when GPU is calibrated as worthwhile.
        gpu_atrs = None
        gpu_mas = gpu_batch.sma(self.cfg.watchlist_proximity_ma_period) if gpu_batch else None
        gpu_mom10 = gpu_batch.momentum(10) if gpu_batch else None

        # GPU indicator cache: every unique (name, params) combo across ALL entry
        # AND exit slots is computed exactly ONCE for the whole universe here,
        # rather than recomputed per-ticker inside the loop below -- this is where
        # the GPU actually earns its keep: one batched op covers every ticker's
        # signal for this trial instead of ~150 separate pandas/numba calls.
        gpu_cache: Dict[Tuple[str, Tuple], Dict[str, pd.Series]] = {}
        if gpu_batch is not None:
            all_slots = [s for s in (self.cfg.entry_slots + self.cfg.exit_slots) if not s.is_empty()]
            for slot in all_slots:
                if slot.name not in GPU_INDICATOR_METHODS:
                    continue
                spec = IndicatorLibrary.REGISTRY[slot.name]["params"]
                kwargs = {k: v for k, v in slot.params.items() if k in spec}
                key = (slot.name, tuple(sorted(kwargs.items())))
                if key in gpu_cache:
                    continue
                try:
                    gpu_cache[key] = getattr(gpu_batch, slot.name)(**kwargs)
                except Exception:
                    pass  # fall back to per-ticker CPU/numba path for this combo

        for t in tickers:
            raw = self.market_data[t]
            src = StrategyComposer.prepare_source_df(raw, self.cfg.use_heikin_ashi)

            entry_sig = StrategyComposer.compute_composite_signal(
                src, self.cfg.entry_slots, "entry", ticker=t, gpu_cache=gpu_cache)
            exit_sig = StrategyComposer.compute_composite_signal(
                src, self.cfg.exit_slots, "exit", ticker=t, gpu_cache=gpu_cache)

            # Pull directly from GPU tensors
            atr_series = gpu_atrs[t] if gpu_atrs else IndicatorLibrary.atr(raw, self.cfg.atr_period)
            ma_series = gpu_mas[t] if gpu_mas else IndicatorLibrary.sma(raw, self.cfg.watchlist_proximity_ma_period)
            mom_series = gpu_mom10[t] if gpu_mom10 else IndicatorLibrary.momentum(raw, 10)

            v_buy_series = CandleTransform.pure_buying_volume(raw)
            ha = CandleTransform.heikin_ashi(raw)
            ha_strength_series = CandleTransform.ha_body_strength(ha)
            
            out[t] = {
                "raw": raw, "entry": entry_sig, "exit": exit_sig, "atr": atr_series,
                "ma": ma_series, "momentum": mom_series, "v_buy": v_buy_series,
                "ha_strength": ha_strength_series,
            }
        return out

    def run(self, tickers: Optional[List[str]] = None, start: Optional[str] = None, end: Optional[str] = None) -> "BacktestResult":
        tickers = tickers or list(self.market_data.keys())
        pre = self._precompute(tickers)

        # master trading calendar = union of all business days present in data
        all_dates = sorted(set().union(*[df["raw"].index for df in pre.values()]))
        all_dates = pd.DatetimeIndex(all_dates)
        if start:
            all_dates = all_dates[all_dates >= pd.Timestamp(start)]
        if end:
            all_dates = all_dates[all_dates <= pd.Timestamp(end)]

        regime_ok = pd.Series(True, index=all_dates)
        if self.cfg.regime_filter_enabled and self.benchmark is not None:
            bench_sma = self.benchmark.rolling(self.cfg.regime_index_sma_period).mean()
            regime_series = (self.benchmark > bench_sma).reindex(all_dates).ffill().fillna(True)
            regime_ok = regime_series

        # Fast path: with up to 10 AND-combined entry slots, a large fraction of
        # randomly-sampled Optuna configs never produce a single entry signal across
        # the whole universe (the composite AND requirement is too restrictive to
        # ever be simultaneously true). Those configs would otherwise still pay the
        # full ~3,750-day event loop -- ticker-by-ticker snapshot building, exit
        # checks, watchlist ranking -- just to discover nothing was ever bought.
        # Detect that upfront and construct the cash-only equity curve directly via
        # the same injection/mark-to-market calls the full loop would have made
        # (the only two operations that matter when no position is ever opened),
        # skipping everything else.
        any_entry_possible = any(
            bool(pre[t]["entry"].reindex(all_dates).fillna(False).any()) for t in tickers
        )
        if not any_entry_possible:
            portfolio = PortfolioEngine(self.cfg, self.injection_days, starting_cash=0.0, sector_map=self.sector_map, ml_ranker=self.ml_ranker)
            for date in all_dates:
                portfolio.maybe_inject_cash(date)
                portfolio.mark_to_market(date, {})
            return BacktestResult.from_portfolio(portfolio, self.cfg, benchmark=self.benchmark)

        portfolio = PortfolioEngine(self.cfg, self.injection_days, starting_cash=0.0, sector_map=self.sector_map, ml_ranker=self.ml_ranker)
        exit_engine = ExitEngine(self.cfg)

        # Signals derived from indicator crossings at Day T close are buffered here and only
        # executed on Day T+1 (next loop iteration), enforcing strict zero-lookahead per spec.
        # Price-triggered mechanical stops (TP / trailing stop / ATR chandelier) are NOT
        # buffered: those are resting stop orders that fire intraday against the current
        # day's price, which is standard execution-model treatment and distinct from an
        # indicator "signal" that requires the close to even be computed.
        pending_entries: Dict[str, None] = {}
        pending_indicator_exits: Dict[str, None] = {}
        # Latched "waiting mode" queue: an entry signal blocked by the regime filter is
        # NOT dropped -- it waits here and is retried every subsequent day until the
        # regime clears (entering then, at that day's price) or the ticker becomes
        # otherwise ineligible (already owned). Without this, a signal that fires while
        # the regime filter happens to be off is simply lost forever, which can silently
        # starve a strategy of trades during exactly the choppy periods a regime filter
        # is meant to sit out and then re-enter after.
        latched_entries: Dict[str, None] = {}

        for i, date in enumerate(all_dates):
            portfolio.maybe_inject_cash(date)

            # -------- 0. snapshot today's prices / features first, so pending T-1 signals fill at
            #             TODAY's price (i.e. Day T signal -> Day T+1 fill) ------------------------
            price_lookup, snapshot = {}, {}
            for t in tickers:
                d = pre[t]
                if date not in d["raw"].index:
                    continue
                price_lookup[t] = float(d["raw"].loc[date, "Close"])
                snapshot[t] = {
                    "price": price_lookup[t],
                    "ma": float(d["ma"].loc[date]) if date in d["ma"].index and not pd.isna(d["ma"].loc[date]) else price_lookup[t],
                    "momentum": float(d["momentum"].loc[date]) if date in d["momentum"].index and not pd.isna(d["momentum"].loc[date]) else 0.0,
                    "v_buy": float(d["v_buy"].loc[date]) if date in d["v_buy"].index and not pd.isna(d["v_buy"].loc[date]) else 0.0,
                    "ha_strength": float(d["ha_strength"].loc[date]) if date in d["ha_strength"].index and not pd.isna(d["ha_strength"].loc[date]) else 0.0,
                    "atr": float(d["atr"].loc[date]) if date in d["atr"].index and not pd.isna(d["atr"].loc[date]) else None,
                }

            # -------- 1. fill indicator-driven exits that were signalled on Day T-1 -----------------
            for t in list(pending_indicator_exits):
                if t in portfolio.positions and t in price_lookup:
                    portfolio.exit_position(t, price_lookup[t], date, reason="indicator_exit")
                pending_indicator_exits.pop(t, None)

            # -------- 2. check mechanical stop exits (TP / trailing / ATR chandelier) same-day -----
            for t in list(portfolio.positions.keys()):
                if t not in price_lookup:
                    continue
                pos = portfolio.positions[t]
                atr_val = float(pre[t]["atr"].loc[date]) if date in pre[t]["atr"].index else np.nan
                pos.peak_price = max(pos.peak_price, price_lookup[t])
                should_exit, reason = exit_engine.check_exit(pos, price_lookup[t], atr_val, indicator_exit_signal=False)
                if should_exit:
                    portfolio.exit_position(t, price_lookup[t], date, reason)

            # -------- 3. fill entries: fresh T-1 signals + latched signals waiting on regime -------
            regime_pass = bool(regime_ok.loc[date]) if date in regime_ok.index else True
            entries_to_try = list(pending_entries) + [t for t in latched_entries if t not in pending_entries]
            for t in entries_to_try:
                pending_entries.pop(t, None)
                if t not in price_lookup or t in portfolio.positions:
                    latched_entries.pop(t, None)  # data unavailable or already owned -- drop, not latch
                    continue
                if not regime_pass:
                    latched_entries[t] = None  # keep waiting for the regime filter to clear
                    continue
                latched_entries.pop(t, None)
                price = price_lookup[t]
                pv = portfolio.cash + sum(p.shares * price_lookup.get(tt, p.entry_price) for tt, p in portfolio.positions.items())
                entered = portfolio.try_enter(t, price, date, pv, atr_value=snapshot.get(t, {}).get("atr"))
                if not entered:
                    portfolio.add_to_watchlist(t, price, date)

            # -------- 4. generate NEW signals off today's close, queued for T+1 execution ----------
            for t in tickers:
                d = pre[t]
                if date in d["entry"].index and t not in portfolio.positions and bool(d["entry"].loc[date]):
                    pending_entries[t] = None
                if date in d["exit"].index and t in portfolio.positions and bool(d["exit"].loc[date]):
                    pending_indicator_exits[t] = None

            # -------- 5. allocate freed / injected cash to top watchlist candidate(s) ---------------
            pv = portfolio.cash + sum(p.shares * price_lookup.get(tt, p.entry_price) for tt, p in portfolio.positions.items())
            portfolio.process_watchlist(date, snapshot, pv)

            # -------- 6. mark to market ------------------------------------------------------------
            portfolio.mark_to_market(date, price_lookup)

        return BacktestResult.from_portfolio(portfolio, self.cfg, benchmark=self.benchmark)


# ==============================================================================
# SECTION: PERFORMANCE METRICS (MONEY-WEIGHTED vs TIME-WEIGHTED RETURN)
# ==============================================================================

class PerformanceMetrics:
    """
    Institutional-grade return/risk metrics for a portfolio that receives
    periodic external cash flows (a monthly SIP), following standard
    (GIPS-style) practice of answering two separate questions:

      - XIRR (money-weighted return): what did an investor who followed this
        exact contribution schedule actually earn? Depends on the TIMING and
        SIZE of contributions, which the strategy itself does not control --
        this is the number an investor statement reports.
      - True daily Time-Weighted Return (TWR) -> cagr/sharpe/sortino/calmar:
        did the STRATEGY have skill? Strips out the effect of cash-flow
        timing entirely by removing each injected amount from the return
        computation on the exact day it lands, then chain-links the
        resulting daily returns. This is the *exact* form (not the
        Modified-Dietz approximation, which exists only because most funds
        lack daily valuations -- we have a full daily equity curve, so we
        don't need to approximate).

    Why this file used to be wrong: (final_equity / first_day_equity)^(1/years)-1
    on a SIP portfolio conflates "new capital arrived" with "capital
    compounded", and equity.pct_change() records every injection day
    (including day 1, equity 0 -> 10,000) as an enormous fake "return" that
    poisons Sharpe/Sortino/Calmar and silently masks real drawdowns.
    """

    @staticmethod
    def xirr(cash_flows: List[Tuple[pd.Timestamp, float]], terminal_date: pd.Timestamp,
             terminal_value: float, guess: float = 0.10) -> float:
        if not cash_flows:
            return 0.0

        total_injected = sum(amt for _, amt in cash_flows)
        # Circuit breaker: zero PnL means exactly 0.0% XIRR
        if abs(terminal_value - total_injected) < 1e-4:
            return 0.0

        flows = [(pd.Timestamp(d), -float(amt)) for d, amt in cash_flows]
        flows.append((pd.Timestamp(terminal_date), float(terminal_value)))
        flows.sort(key=lambda x: x[0])
        t0 = flows[0][0]

        def npv(rate: float) -> float:
            total = 0.0
            for d, cf in flows:
                years = (d - t0).days / 365.25
                total += cf / ((1.0 + rate) ** years)
            return total

        if abs(npv(0.0)) < 1e-4:
            return 0.0

        # Dynamically search for a valid bracket (a, b) where npv(a) and npv(b) change signs
        a, b = None, None
        grid = np.linspace(-0.70, 2.0, 50)
        vals = [npv(r) for r in grid]
        
        for i in range(len(grid) - 1):
            if vals[i] * vals[i+1] <= 0:
                a, b = grid[i], grid[i+1]
                break

        if a is not None and b is not None:
            try:
                from scipy.optimize import brentq
                return float(brentq(npv, a, b, maxiter=500))
            except Exception:
                pass

        # Robust Newton fallback
        try:
            from scipy.optimize import newton
            res = newton(npv, x0=guess, maxiter=200)
            if not np.isnan(res) and -0.99 < res < 10.0:
                return float(res)
        except Exception:
            pass

        return 0.0
        
    @staticmethod
    def true_daily_returns(equity_curve: pd.Series, cash_flows: List[Tuple[pd.Timestamp, float]]) -> pd.Series:
        """Exact daily TWR: strip that day's injected cash before computing the return."""
        if len(equity_curve) < 2:
            return pd.Series(dtype=float)
        eq = equity_curve.sort_index()
        cf_series = pd.Series(0.0, index=eq.index)
        for d, amt in cash_flows:
            d = pd.Timestamp(d)
            if d in cf_series.index:
                cf_series.loc[d] += float(amt)

        prev = eq.shift(1)
        rets = ((eq - cf_series) / prev - 1.0).replace([np.inf, -np.inf], np.nan)
        # Days before any capital exists (e.g. pre-first-injection) have an
        # undefined "return", not an infinite one -- treat as flat, same as a
        # fund NAV series before its first subscription.
        rets = rets.fillna(0.0)
        return rets.iloc[1:]

    @staticmethod
    def twr_index(daily_returns: pd.Series) -> pd.Series:
        """Chain-linked growth-of-1 index built purely from TWR daily returns."""
        return (1.0 + daily_returns).cumprod()

    @staticmethod
    def full_report(
        equity_curve: pd.Series, cash_flows: List[Tuple[pd.Timestamp, float]],
        benchmark: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        empty = {"xirr": 0.0, "twr_cagr": 0.0, "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0,
                 "max_drawdown": 0.0, "alpha": 0.0, "beta": 0.0, "information_ratio": 0.0,
                 "tracking_error": 0.0}
        if len(equity_curve) < 2 or not cash_flows:
            return empty

        eq = equity_curve.sort_index()
        final_value = float(eq.iloc[-1])
        xirr_val = PerformanceMetrics.xirr(cash_flows, eq.index[-1], final_value)

        daily_ret = PerformanceMetrics.true_daily_returns(eq, cash_flows)
        idx = PerformanceMetrics.twr_index(daily_ret)
        if len(idx) < 2:
            return empty

        years = max((idx.index[-1] - idx.index[0]).days / 365.25, 1e-6)
        twr_cagr = float(idx.iloc[-1] ** (1 / years) - 1) if idx.iloc[-1] > 0 else -1.0

        running_max = idx.cummax()
        dd = (idx - running_max) / running_max.replace(0, np.nan)
        max_dd = float(dd.min()) if len(dd) else 0.0

        std = daily_ret.std()
        sharpe = float((daily_ret.mean() / std) * math.sqrt(252)) if std and std > 0 else 0.0
        downside_std = daily_ret[daily_ret < 0].std()
        sortino = float((daily_ret.mean() / downside_std) * math.sqrt(252)) if downside_std and downside_std > 0 else 0.0
        calmar = float(twr_cagr / abs(max_dd)) if max_dd != 0 else 0.0

        alpha = beta = info_ratio = tracking_error = 0.0
        if benchmark is not None and len(benchmark) > 2:
            bench_ret = benchmark.sort_index().pct_change().reindex(daily_ret.index).fillna(0.0)
            if bench_ret.std() > 0:
                cov = np.cov(daily_ret.values, bench_ret.values)
                beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 0.0
                # Geometric (CAGR-consistent) annualization, NOT raw arithmetic mean*252.
                # Arithmetic annualization has no normalization for how volatile the
                # return series is, so a low-trade-count strategy that hits a deep
                # drawdown then recovers can produce an absurd "annualized return" from
                # arithmetic mean alone (verified: a -97% drawdown scenario gave +15%
                # arithmetic vs the correct -5% geometric) -- exactly the failure mode
                # that produced a reported 1509% alpha on a real run. CAGR/Sharpe/Sortino
                # above already use daily_ret.mean() too, but always as part of a RATIO
                # (divided by std, or geometrically compounded), which is far more robust;
                # alpha previously used the raw arithmetic annualization with no such
                # normalization, so it alone inherited the full blow-up.
                n_days = len(daily_ret)
                strategy_growth = float((1.0 + daily_ret).prod())
                bench_growth = float((1.0 + bench_ret).prod())
                ann_strategy = strategy_growth ** (252.0 / n_days) - 1.0 if strategy_growth > 0 else -1.0
                ann_bench = bench_growth ** (252.0 / n_days) - 1.0 if bench_growth > 0 else -1.0
                alpha = float(ann_strategy - beta * ann_bench)
                excess = daily_ret - bench_ret
                tracking_error = float(excess.std() * math.sqrt(252))
                info_ratio = float((excess.mean() * 252) / tracking_error) if tracking_error > 0 else 0.0

        return {"xirr": float(xirr_val), "twr_cagr": twr_cagr, "sharpe": sharpe, "sortino": sortino,
                "calmar": calmar, "max_drawdown": max_dd, "alpha": alpha, "beta": beta,
                "information_ratio": info_ratio, "tracking_error": tracking_error}


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trade_log: List[dict]
    cfg: StrategyConfig
    total_invested: float
    final_value: float
    total_profit: float
    total_return: float
    cagr: float                # true daily-TWR-based CAGR (strategy skill, not contribution size)
    max_drawdown: float        # measured on the TWR index, so injections can't mask a real drawdown
    sharpe: float               # TWR-based, injection days no longer poison this
    sortino: float
    calmar: float
    n_trades: int
    win_rate: float
    profit_factor: float
    best_trade: float
    worst_trade: float
    avg_trade_pnl: float
    xirr: float = 0.0            # money-weighted return: what the actual SIP investor earned
    alpha: float = 0.0           # vs benchmark, annualized
    beta: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    twr_daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    cash_flows: List[Tuple[pd.Timestamp, float]] = field(default_factory=list)

    @staticmethod
    def from_portfolio(portfolio: PortfolioEngine, cfg: StrategyConfig,
                        benchmark: Optional[pd.Series] = None) -> "BacktestResult":
        if not portfolio.equity_curve:
            eq = pd.Series(dtype=float)
        else:
            dates, values = zip(*portfolio.equity_curve)
            eq = pd.Series(values, index=pd.DatetimeIndex(dates)).sort_index()

        if len(eq) < 2:
            return BacktestResult(
                eq, portfolio.trade_log, cfg, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0
            )

        # Capital & simple contribution-relative return (kept for reference --
        # this is NOT annualized and NOT a performance metric on its own).
        n_months = max(1, len(set((d.year, d.month) for d in eq.index)))
        total_invested = MONTHLY_INJECTION * n_months
        final_value = float(eq.iloc[-1])
        total_profit = final_value - total_invested
        total_return = total_profit / total_invested if total_invested else 0.0

        metrics = PerformanceMetrics.full_report(eq, portfolio.cash_flows, benchmark=benchmark)

        # Trade Execution Breakdown
        closed_sells = [t for t in portfolio.trade_log if t.get("action") == "SELL"]
        pnls = [t["pnl"] for t in closed_sells if "pnl" in t]
        n_trades = len(pnls)

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

        best_trade = float(max(pnls)) if pnls else 0.0
        worst_trade = float(min(pnls)) if pnls else 0.0
        avg_trade_pnl = float(np.mean(pnls)) if pnls else 0.0

        return BacktestResult(
            equity_curve=eq, trade_log=portfolio.trade_log, cfg=cfg,
            total_invested=float(total_invested), final_value=final_value,
            total_profit=float(total_profit), total_return=float(total_return),
            cagr=metrics["twr_cagr"], max_drawdown=metrics["max_drawdown"], sharpe=metrics["sharpe"],
            sortino=metrics["sortino"], calmar=metrics["calmar"], n_trades=n_trades,
            win_rate=float(win_rate), profit_factor=float(profit_factor),
            best_trade=best_trade, worst_trade=worst_trade, avg_trade_pnl=avg_trade_pnl,
            xirr=metrics["xirr"], alpha=metrics["alpha"], beta=metrics["beta"],
            information_ratio=metrics["information_ratio"], tracking_error=metrics["tracking_error"],
            twr_daily_returns=PerformanceMetrics.true_daily_returns(eq, portfolio.cash_flows),
            cash_flows=list(portfolio.cash_flows),
        )


# ==============================================================================
# SECTION: TWO-PASS ML WATCHLIST RANKER
# ==============================================================================

class MLWatchlistRanker:
    """
    Pass 1: run the strategy once to measure the empirical mean holding
            period H_bar (in trading days) across all closed trades.
    Pass 2: train a ranking model (LightGBM if available, else Ridge) that
            targets forward return over [t, t+H_bar] for each watchlist-style
            candidate, using the same feature set as WatchlistEngine.
    """

    FEATURES = ["price_dist", "age", "ma_dist", "momentum", "v_buy", "ha_strength"]

    def __init__(self):
        self.model = None
        self.h_bar: Optional[float] = None

    @staticmethod
    def measure_mean_holding_period(trade_log: List[dict]) -> float:
        buys = {t["ticker"]: t["date"] for t in trade_log if t["action"] == "BUY"}
        holds = []
        # naive pairing by ticker in chronological order (sufficient for a mean-holding estimate)
        buy_queue: Dict[str, List[str]] = {}
        for t in trade_log:
            if t["action"] == "BUY":
                buy_queue.setdefault(t["ticker"], []).append(t["date"])
            elif t["action"] == "SELL" and buy_queue.get(t["ticker"]):
                bdate = buy_queue[t["ticker"]].pop(0)
                holds.append((pd.Timestamp(t["date"]) - pd.Timestamp(bdate)).days)
        return float(np.mean(holds)) if holds else 10.0

    def build_training_set(
        self, market_data: Dict[str, pd.DataFrame], h_bar: int, sample_per_ticker: int = 200,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        rows, targets = [], []
        rng = np.random.default_rng(42)
        for ticker, df in market_data.items():
            if len(df) < h_bar + 60:
                continue
            ma = IndicatorLibrary.sma(df, 50)
            mom = IndicatorLibrary.momentum(df, 10)
            v_buy = CandleTransform.pure_buying_volume(df)
            ha = CandleTransform.heikin_ashi(df)
            ha_strength = CandleTransform.ha_body_strength(ha)
            fwd_ret = df["Close"].shift(-h_bar) / df["Close"] - 1.0

            valid_idx = df.index[50: len(df) - h_bar]
            if len(valid_idx) == 0:
                continue
            close_vals = df["Close"].values
            chosen = rng.choice(valid_idx, size=min(sample_per_ticker, len(valid_idx)), replace=False)
            for idx in chosen:
                pos = df.index.get_loc(idx)
                age = int(rng.integers(0, 20))  # synthetic watchlist-age proxy for offline training
                anchor_pos = max(0, pos - age)
                anchor_price = close_vals[anchor_pos]
                # price drift since the (simulated) original watchlist signal price --
                # NOT x/x-1, which is identically zero; this is the feature the
                # weighted_score watchlist model's "distance from signal price" mirrors.
                price_dist = float((close_vals[pos] - anchor_price) / max(anchor_price, 1e-6))
                row = {
                    "price_dist": price_dist,
                    "age": float(age),
                    "ma_dist": float((df["Close"].loc[idx] - ma.loc[idx]) / max(ma.loc[idx], 1e-6)) if not pd.isna(ma.loc[idx]) else 0.0,
                    "momentum": float(mom.loc[idx]) if not pd.isna(mom.loc[idx]) else 0.0,
                    "v_buy": float(v_buy.loc[idx]) if not pd.isna(v_buy.loc[idx]) else 0.0,
                    "ha_strength": float(ha_strength.loc[idx]) if not pd.isna(ha_strength.loc[idx]) else 0.0,
                }
                rows.append(row)
                targets.append(float(fwd_ret.loc[idx]) if not pd.isna(fwd_ret.loc[idx]) else 0.0)
        X = pd.DataFrame(rows, columns=self.FEATURES)
        y = pd.Series(targets)
        return X, y

    def fit(self, market_data: Dict[str, pd.DataFrame], trade_log: List[dict]):
        self.h_bar = max(3, int(round(self.measure_mean_holding_period(trade_log))))
        X, y = self.build_training_set(market_data, self.h_bar)
        if len(X) < 50:
            logger.warning("Insufficient samples for ML watchlist ranker; skipping fit.")
            return self
        try:
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, verbosity=-1)
            self.model.fit(X, y)
        except Exception as e:
            logger.warning(f"LightGBM unavailable ({e}); falling back to Ridge regression.")
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=1.0)
            self.model.fit(X, y)
        return self

    def score(self, features: Dict[str, float]) -> float:
        if self.model is None:
            return 0.0
        X = pd.DataFrame([{k: features.get(k, 0.0) for k in self.FEATURES}])
        return float(self.model.predict(X)[0])


# ==============================================================================
# SECTION: ROBUSTNESS, SENSITIVITY & VALIDATION SUITE
# ==============================================================================

class RobustnessSuite:
    """
    Implements:
      - 15-run SIP date sensitivity suite (10 fixed + 5 dynamic random schedules)
      - Parameter neighborhood stability check (+/-5% to +/-15% perturbation x5)
      - Deflated Sharpe Ratio (Marcos Lopez de Prado)
      - 1,000-path Monte Carlo trade-sequence permutation (95th pct DD, risk of ruin)
    """

    def __init__(self, market_data: Dict[str, pd.DataFrame], benchmark: Optional[pd.Series] = None):
        self.market_data = market_data
        self.benchmark = benchmark

    @staticmethod
    def _resolve_fixed_schedule(all_dates: pd.DatetimeIndex, target_day: int) -> Dict[Tuple[int, int], pd.Timestamp]:
        """Resolve a target day-of-month to the actual first trading day on/after it, per (year, month)."""
        schedule = {}
        by_month = pd.Series(all_dates, index=all_dates).groupby([all_dates.year, all_dates.month])
        for (y, m), grp in by_month:
            dates_in_month = list(grp)
            chosen = next((d for d in dates_in_month if d.day >= target_day), dates_in_month[-1])
            schedule[(y, m)] = chosen
        return schedule

    @staticmethod
    def _resolve_dynamic_schedule(all_dates: pd.DatetimeIndex, seed: int) -> Dict[Tuple[int, int], pd.Timestamp]:
        """Draw an independent random trading date within every (year, month) in the calendar."""
        rng = np.random.default_rng(seed)
        schedule = {}
        by_month = pd.Series(all_dates, index=all_dates).groupby([all_dates.year, all_dates.month])
        for (y, m), grp in by_month:
            dates_in_month = list(grp)
            schedule[(y, m)] = dates_in_month[int(rng.integers(0, len(dates_in_month)))]
        return schedule

    # ---------------------------------------------------------- SIP sensitivity
    def sip_sensitivity(self, cfg: StrategyConfig, tickers: List[str], n_fixed: int = 10, n_dynamic: int = 5,
                         seed: int = 7) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        all_dates = pd.DatetimeIndex(sorted(set().union(*[self.market_data[t].index for t in tickers if t in self.market_data])))

        schedules = []
        fixed_days = rng.choice(range(1, 29), size=n_fixed, replace=False)
        for d in fixed_days:
            schedules.append(("fixed", self._resolve_fixed_schedule(all_dates, int(d))))
        for sim_idx in range(n_dynamic):
            # a genuinely "dynamic" schedule draws an *independent* random injection
            # trading date for every single calendar month, rather than reusing one
            # static pool of days (which would silently collapse to a fixed schedule
            # under the once-per-month dedup in maybe_inject_cash).
            schedules.append(("dynamic", self._resolve_dynamic_schedule(all_dates, seed=1000 + seed + sim_idx)))

        returns, xirrs, cagrs = [], [], []
        for kind, days in schedules:
            bt = Backtester(cfg, self.market_data, injection_days=days, benchmark=self.benchmark)
            res = bt.run(tickers=tickers)
            returns.append(res.total_return)
            xirrs.append(res.xirr)
            cagrs.append(res.cagr)

        return {
            "schedules_run": len(schedules),
            "returns": returns,
            "xirrs": xirrs,
            "cagrs_time_weighted": cagrs,
            # std of the TWR-based CAGR is the real robustness signal: it should be small
            # (the strategy's skill shouldn't depend on which day of the month cash lands),
            # whereas std of XIRR will naturally move more since XIRR is money-weighted and
            # mechanically sensitive to contribution timing -- that's expected, not a flaw.
            "std_cagr_time_weighted": float(np.std(cagrs)) if cagrs else 0.0,
            "std_xirr": float(np.std(xirrs)) if xirrs else 0.0,
            "mean_return": float(np.mean(returns)) if returns else 0.0,
            "std_return": float(np.std(returns)) if returns else 0.0,
        }

    # ---------------------------------------------------------- parameter neighborhood
    def parameter_neighborhood_check(
        self, cfg: StrategyConfig, tickers: List[str], injection_days: List[int],
        n_neighbors: int = 5, pct_range: Tuple[float, float] = (0.05, 0.15), seed: int = 11,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        base_bt = Backtester(cfg, self.market_data, injection_days=injection_days, benchmark=self.benchmark)
        base_score = base_bt.run(tickers=tickers).sharpe

        neighbor_scores = []
        for _ in range(n_neighbors):
            perturbed = dataclasses.replace(cfg, entry_slots=[dataclasses.replace(s) for s in cfg.entry_slots],
                                             exit_slots=[dataclasses.replace(s) for s in cfg.exit_slots])
            for slot in perturbed.entry_slots + perturbed.exit_slots:
                if slot.is_empty():
                    continue
                new_params = {}
                for p, v in slot.params.items():
                    pct = rng.uniform(*pct_range) * rng.choice([-1, 1])
                    lo, hi = _param_range(slot.name, p)
                    new_val = int(np.clip(round(v * (1 + pct)), lo, hi))
                    new_params[p] = new_val
                slot.params = new_params
            nb_bt = Backtester(perturbed, self.market_data, injection_days=injection_days, benchmark=self.benchmark)
            neighbor_scores.append(nb_bt.run(tickers=tickers).sharpe)

        neighbor_scores = np.array(neighbor_scores)
        degradation = (base_score - neighbor_scores.mean()) / (abs(base_score) + 1e-9)
        is_plateau = bool(degradation < 0.35)  # heuristic: <35% relative Sharpe drop across neighborhood
        return {
            "base_sharpe": float(base_score), "neighbor_sharpes": neighbor_scores.tolist(),
            "mean_degradation_pct": float(degradation * 100), "is_stable_plateau": is_plateau,
        }

    # ---------------------------------------------------------- Walk-forward: anchored expanding folds
    def walk_forward_expanding_folds(
        self, cfg: StrategyConfig, tickers: List[str], injection_days: List[int], n_folds: int = 4,
    ) -> Dict[str, Any]:
        """
        Upgrade over the single static 70/30 split (train_test_split_check): runs
        n_folds anchored expanding-window folds -- fold k trains on [start,
        boundary_k] and tests on (boundary_k, boundary_{k+1}], for k=1..n_folds.
        Every later segment of history gets to act as an un-peeked out-of-sample
        test at least once (not just the final 30%), and any drift in how well the
        config generalizes across different time periods becomes visible instead
        of hidden behind one lucky or unlucky split point.
        """
        all_dates = pd.DatetimeIndex(sorted(set().union(
            *[self.market_data[t].index for t in tickers if t in self.market_data])))
        if len(all_dates) < 200:
            return {"error": "insufficient history for walk-forward folds"}

        boundaries = np.linspace(0, len(all_dates) - 1, n_folds + 2)[1:].astype(int)
        folds = []
        for k in range(n_folds):
            train_end = all_dates[boundaries[k]]
            test_end = all_dates[boundaries[k + 1]]
            train_bt = Backtester(cfg, self.market_data, injection_days=injection_days, benchmark=self.benchmark)
            train_res = train_bt.run(tickers=tickers, end=str(train_end.date()))
            test_bt = Backtester(cfg, self.market_data, injection_days=injection_days, benchmark=self.benchmark)
            test_res = test_bt.run(tickers=tickers, start=str((train_end + pd.Timedelta(days=1)).date()),
                                    end=str(test_end.date()))

            def decay(is_val, oos_val):
                return float((is_val - oos_val) / abs(is_val)) if abs(is_val) > 1e-9 else 0.0

            folds.append({
                "fold": k + 1, "train_end": str(train_end.date()), "test_end": str(test_end.date()),
                "train_sharpe": train_res.sharpe, "test_sharpe": test_res.sharpe,
                "train_cagr": train_res.cagr, "test_cagr": test_res.cagr,
                "test_n_trades": test_res.n_trades,
                "sharpe_decay_pct": decay(train_res.sharpe, test_res.sharpe) * 100,
            })

        mean_decay = float(np.mean([f["sharpe_decay_pct"] for f in folds])) / 100.0
        # Overfit if the AVERAGE decay is severe, or if MOST individual folds show severe
        # decay -- a strategy that only survives one lucky fold isn't robust, even if the
        # average looks acceptable.
        n_severe_folds = sum(1 for f in folds if f["sharpe_decay_pct"] > 50.0)
        return {
            "n_folds": n_folds, "folds": folds,
            "mean_sharpe_decay_pct": mean_decay * 100,
            "n_severe_folds": n_severe_folds,
            "likely_overfit": bool(mean_decay > 0.5 or n_severe_folds > n_folds // 2),
        }

    # ---------------------------------------------------------- Per-asset consistency (anti-overfitting)
    @staticmethod
    def per_asset_consistency_check(trade_log: List[dict], min_profitable_frac: float = 0.5) -> Dict[str, Any]:
        """
        A strategy that only wins because of 1-2 lucky monster-run tickers, while
        most of the traded universe actually loses money, isn't a real systemic
        edge -- it's a power-law fluke that Sharpe/CAGR alone won't reveal (those
        are computed on the AGGREGATE equity curve, which one huge winner can
        dominate). Groups closed trades by ticker, sums each ticker's net P&L, and
        requires at least min_profitable_frac of individually-traded tickers to be
        profitable on their own.
        """
        per_ticker_pnl: Dict[str, float] = {}
        for t in trade_log:
            if t.get("action") == "SELL" and "pnl" in t:
                per_ticker_pnl[t["ticker"]] = per_ticker_pnl.get(t["ticker"], 0.0) + t["pnl"]
        if not per_ticker_pnl:
            return {"n_tickers_traded": 0, "frac_profitable": 0.0, "median_pnl": 0.0, "passes_gate": False}
        pnls = list(per_ticker_pnl.values())
        n_profitable = sum(1 for p in pnls if p > 0)
        frac_profitable = n_profitable / len(pnls)
        return {
            "n_tickers_traded": len(pnls), "n_profitable": n_profitable,
            "frac_profitable": float(frac_profitable), "median_pnl": float(np.median(pnls)),
            "per_ticker_pnl": per_ticker_pnl, "passes_gate": bool(frac_profitable >= min_profitable_frac),
        }

    # ---------------------------------------------------------- Annual profit concentration (anti-overfitting)
    @staticmethod
    def annual_profit_concentration_check(trade_log: List[dict], max_single_year_frac: float = 0.55) -> Dict[str, Any]:
        """
        Rejects strategies whose lifetime profit is dominated by a single lucky
        year (e.g. one huge bull run) rather than a repeatable edge. Sums closed-
        trade P&L by the calendar year of the SELL date, and flags if any single
        year contributed more than max_single_year_frac of total NOMINAL
        (positive-years-only) profit.
        """
        yearly_pnl: Dict[int, float] = {}
        for t in trade_log:
            if t.get("action") == "SELL" and "pnl" in t:
                year = pd.Timestamp(t["date"]).year
                yearly_pnl[year] = yearly_pnl.get(year, 0.0) + t["pnl"]
        total_nominal_profit = sum(p for p in yearly_pnl.values() if p > 0)
        if total_nominal_profit <= 0:
            return {"max_year_frac": 0.0, "passes_gate": False, "yearly_pnl": yearly_pnl}
        max_year_frac = max((p / total_nominal_profit for p in yearly_pnl.values() if p > 0), default=0.0)
        return {
            "max_year_frac": float(max_year_frac), "yearly_pnl": yearly_pnl,
            "passes_gate": bool(max_year_frac <= max_single_year_frac),
        }

    # ---------------------------------------------------------- Walk-forward / out-of-sample split (single, legacy)
    def train_test_split_check(
        self, cfg: StrategyConfig, tickers: List[str], injection_days: List[int],
        train_frac: float = 0.7,
    ) -> Dict[str, Any]:
        """
        The single most important check missing from the pipeline: Optuna
        currently only ever sees the FULL history, so nothing here proves the
        winning config isn't just overfit to the 15-year sample it was
        selected on. This splits the available calendar chronologically into
        an in-sample training window (first train_frac) and a strictly later
        out-of-sample window, reruns the *same, already-fixed* config on
        each, and reports how much skill survives. A config whose
        out-of-sample Sharpe/CAGR collapses relative to in-sample is
        overfit, however good its full-history numbers look.

        NOTE: walk_forward_expanding_folds() above is the more rigorous upgrade
        (multiple anchored folds instead of one static split) and is what
        run_full_pipeline uses by default now; this single-split version is kept
        for anyone who wants the cheaper, lighter check.
        """
        all_dates = pd.DatetimeIndex(sorted(set().union(
            *[self.market_data[t].index for t in tickers if t in self.market_data])))
        if len(all_dates) < 100:
            return {"error": "insufficient history for a train/test split"}
        split_idx = int(len(all_dates) * train_frac)
        split_date = all_dates[split_idx]

        train_bt = Backtester(cfg, self.market_data, injection_days=injection_days, benchmark=self.benchmark)
        train_res = train_bt.run(tickers=tickers, end=str(split_date.date()))

        test_bt = Backtester(cfg, self.market_data, injection_days=injection_days, benchmark=self.benchmark)
        test_res = test_bt.run(tickers=tickers, start=str((split_date + pd.Timedelta(days=1)).date()))

        def decay(is_val, oos_val):
            if abs(is_val) < 1e-9:
                return 0.0
            return float((is_val - oos_val) / abs(is_val))

        sharpe_decay = decay(train_res.sharpe, test_res.sharpe)
        cagr_decay = decay(train_res.cagr, test_res.cagr)
        # Heuristic: >50% relative decay in either Sharpe or CAGR out-of-sample is a strong
        # overfitting flag; institutional reviewers typically want to see this near-zero
        # or even negative (out-of-sample outperforming in-sample, i.e. no overfit at all).
        likely_overfit = bool(sharpe_decay > 0.5 or cagr_decay > 0.5)

        return {
            "split_date": str(split_date.date()),
            "train_frac": train_frac,
            "in_sample": {
                "sharpe": train_res.sharpe, "cagr": train_res.cagr, "xirr": train_res.xirr,
                "max_drawdown": train_res.max_drawdown, "n_trades": train_res.n_trades,
            },
            "out_of_sample": {
                "sharpe": test_res.sharpe, "cagr": test_res.cagr, "xirr": test_res.xirr,
                "max_drawdown": test_res.max_drawdown, "n_trades": test_res.n_trades,
            },
            "sharpe_decay_pct": sharpe_decay * 100,
            "cagr_decay_pct": cagr_decay * 100,
            "likely_overfit": likely_overfit,
        }

    # ---------------------------------------------------------- Deflated Sharpe Ratio
    @staticmethod
    def deflated_sharpe_ratio(returns: pd.Series, n_trials: int, skew_adj: bool = True) -> float:
        """
        Marcos Lopez de Prado's Deflated Sharpe Ratio: probability the observed
        Sharpe Ratio is genuinely positive after correcting for multiple-testing
        (n_trials strategy configurations were searched over) and for the
        non-normality of the returns distribution.
        """
        from scipy.stats import norm, skew, kurtosis
        if len(returns) < 10 or returns.std() == 0:
            return 0.0
        sr = returns.mean() / returns.std() * math.sqrt(252)
        T = len(returns)
        gamma3 = skew(returns) if skew_adj else 0.0
        gamma4 = kurtosis(returns, fisher=False) if skew_adj else 3.0

        # expected max Sharpe under n_trials independent trials (order-statistics approximation)
        euler_gamma = 0.5772156649
        sr_std_estimate = 1.0  # normalized trial SR variance assumption
        expected_max_sr = sr_std_estimate * (
            (1 - euler_gamma) * norm.ppf(1 - 1 / n_trials) + euler_gamma * norm.ppf(1 - 1 / (n_trials * math.e))
        ) if n_trials > 1 else 0.0

        sr_std = math.sqrt(max((1 - gamma3 * sr + ((gamma4 - 1) / 4) * sr ** 2), 1e-9) / max(T - 1, 1))
        dsr = norm.cdf((sr - expected_max_sr) / sr_std) if sr_std > 0 else 0.0
        return float(dsr)

    # ---------------------------------------------------------- Monte Carlo permutation
    @staticmethod
    def monte_carlo_permutation(trade_pnls: List[float], n_paths: int = 1000, seed: int = 99) -> Dict[str, float]:
        """
        Shuffle the realized trade P&L sequence n_paths times to build a
        distribution of terminal drawdowns and estimate the 95th-percentile
        drawdown plus an empirical risk-of-ruin probability. JIT-compiled
        (see _monte_carlo_numba) -- this loop was previously ~1,000 paths x
        4 separate numpy calls each, now one compiled loop with zero
        per-call Python/numpy overhead (~10x faster).
        """
        if not trade_pnls:
            return {"p95_drawdown": 0.0, "risk_of_ruin": 0.0}
        pnls = np.array(trade_pnls, dtype=np.float64)
        ruin_threshold = -0.9 * abs(pnls.sum()) if pnls.sum() != 0 else -1e9

        drawdowns, ruin_count = _monte_carlo_numba(pnls, n_paths, seed, ruin_threshold)

        p95_dd = float(np.percentile(drawdowns, 5))  # 5th pct of dd (most negative) == 95th pct severity
        risk_of_ruin = ruin_count / n_paths
        return {"p95_drawdown": p95_dd, "risk_of_ruin": float(risk_of_ruin)}


# ==============================================================================
# SECTION: OPTUNA OPTIMIZER + DUAL REAL-TIME PERSISTENCE
# ==============================================================================

CURRENT_WINNER_PATH = os.path.join(RESULTS_DIR, "current_winner_config.json")
FINAL_WINNER_PATH = os.path.join(RESULTS_DIR, "final_winner_config.json")


def _load_prior_winner_config(path: str) -> Optional[StrategyConfig]:
    """Load a previously persisted winner config from disk, if it exists and parses."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        return StrategyConfig.from_dict(payload["config"])
    except Exception as e:
        logger.warning(f"Could not load prior config from {path}: {e}")
        return None


def _config_to_optuna_params(
    cfg: StrategyConfig, indicator_names: List[str], comparators: List[str], watchlist_models: List[str],
) -> Dict[str, Any]:
    """
    Reverse-maps a StrategyConfig back into the exact flat parameter dict
    Optuna's study.enqueue_trial() needs to reproduce it -- i.e. the same
    {param_name: value} shape that _suggest_config/_suggest_slot's
    trial.suggest_*() calls would have produced. This is what lets a
    previously-found winner be re-injected as a guaranteed first trial in a
    new search (see run_full_pipeline's "memory" section) instead of starting
    the TPE sampler from nothing every single run.

    Values are clipped into whatever the CURRENT indicator parameter ranges
    are (see PARAM_RANGE_GROUPS) before being enqueued -- if those ranges have
    since been tightened, Optuna's suggest_int() would otherwise raise a
    hard error on an out-of-range enqueued value. The fresh full-window
    backtest that validates "is this config still relevant" (see
    run_full_pipeline) runs the ORIGINAL unclipped values directly against
    IndicatorLibrary, so that validation is unaffected by this clipping --
    only the re-seeded search trial is.
    """
    params: Dict[str, Any] = {
        "n_entry_slots": int(np.clip(max(1, len(cfg.entry_slots)), 1, MAX_ENTRY_SLOTS)),
        "n_exit_slots": int(np.clip(len(cfg.exit_slots), 0, MAX_EXIT_SLOTS)),
    }

    def slot_params(prefix: str, slot: IndicatorSlot):
        active = not slot.is_empty() and slot.name in indicator_names
        params[f"{prefix}_active"] = active
        if not active:
            return
        params[f"{prefix}_name"] = slot.name
        spec = IndicatorLibrary.REGISTRY[slot.name]["params"]
        for p, (lo, hi, _dv) in spec.items():
            if lo == hi:
                continue  # fixed param, not sampled, nothing to enqueue
            raw_val = slot.params.get(p, _dv)
            params[f"{prefix}_{slot.name}_{p}"] = int(np.clip(raw_val, lo, hi))
        comparator = slot.comparator if slot.comparator in comparators else comparators[0]
        params[f"{prefix}_comparator"] = comparator
        if comparator in ("greater_than", "less_than"):
            params[f"{prefix}_threshold"] = float(np.clip(slot.threshold, 0.0, 100.0))

    entry_slots = cfg.entry_slots or [IndicatorSlot(name=None)]
    for i, slot in enumerate(entry_slots):
        slot_params(f"entry{i}", slot)
    for i, slot in enumerate(cfg.exit_slots):
        slot_params(f"exit{i}", slot)

    use_tp = cfg.take_profit_pct is not None
    use_tsl = cfg.trailing_stop_pct is not None
    use_chandelier = cfg.atr_chandelier_mult is not None
    params["use_tp"] = use_tp
    params["use_tsl"] = use_tsl
    params["use_atr_chandelier"] = use_chandelier
    if use_tp:
        params["take_profit_pct"] = float(np.clip(cfg.take_profit_pct, 0.05, 0.60))
    if use_tsl:
        params["trailing_stop_pct"] = float(np.clip(cfg.trailing_stop_pct, 0.03, 0.30))
    if use_chandelier:
        params["atr_chandelier_mult"] = float(np.clip(cfg.atr_chandelier_mult, 1.5, 5.0))

    params["use_heikin_ashi"] = cfg.use_heikin_ashi
    params["atr_period"] = int(np.clip(cfg.atr_period, 5, 60))
    # "ml_ranked" is a runtime-only mode (see WatchlistEngine), never one of Optuna's
    # sampled categorical choices -- fall back to the default heuristic if encountered
    # (shouldn't normally happen since ml_ranked configs are never persisted as winners).
    params["watchlist_model"] = cfg.watchlist_model if cfg.watchlist_model in watchlist_models else watchlist_models[0]
    params["wl_w_price_dist"] = float(np.clip(cfg.watchlist_weight_price_dist, 0.0, 1.0))
    params["wl_w_age"] = float(np.clip(cfg.watchlist_weight_age, 0.0, 1.0))
    params["wl_ma_period"] = int(np.clip(cfg.watchlist_proximity_ma_period, *PARAM_RANGE_GROUPS["moving_average"]))
    params["regime_filter_enabled"] = cfg.regime_filter_enabled
    params["regime_sma_period"] = int(np.clip(cfg.regime_index_sma_period, 50, 250))
    params["sector_cap_pct"] = float(np.clip(cfg.sector_cap_pct, 0.15, 1.0))
    params["vol_sizing_enabled"] = cfg.vol_sizing_enabled
    params["equal_weight_sizing_enabled"] = cfg.equal_weight_sizing_enabled
    params["cooldown_days"] = int(np.clip(cfg.cooldown_days, 0, 20))
    return params


class ObjectiveFunction:
    """Custom multi-objective loss combining Sharpe, CAGR, Max Drawdown, and a robustness penalty."""

    def __init__(self, w_sharpe=0.4, w_cagr=0.3, w_dd=0.2, w_robustness=0.1, max_drawdown_gate=-0.50):
        self.w_sharpe, self.w_cagr, self.w_dd, self.w_robustness = w_sharpe, w_cagr, w_dd, w_robustness
        # A drawdown this severe isn't investable with real capital no matter how good
        # Sharpe/CAGR look on paper -- verified directly: a config with Sharpe 1.923 and
        # CAGR 17.87% still "won" under the soft w_dd=0.2 weighting despite an 80.82% max
        # drawdown, because good Sharpe/CAGR could always outweigh the drawdown term. This
        # gate makes catastrophic drawdown disqualifying on its own, separate from the
        # weighted score.
        self.max_drawdown_gate = max_drawdown_gate

    def __call__(self, result: BacktestResult, robustness_penalty: float = 0.0) -> float:
        # Heavy penalty for dead strategies with 0 trades
        if result.n_trades == 0:
            return -999.0

        dd_term = -result.max_drawdown  # max_drawdown is negative; flip sign
        score = (self.w_sharpe * result.sharpe + self.w_cagr * result.cagr
                - self.w_dd * dd_term - self.w_robustness * robustness_penalty)

        # Escalating (not flat) penalty past the gate, so Optuna still gets a gradient
        # pushing away from worse and worse drawdowns rather than a wall where -55% and
        # -95% drawdown score identically.
        if result.max_drawdown < self.max_drawdown_gate:
            overshoot = self.max_drawdown_gate - result.max_drawdown  # positive amount past the gate
            score -= 5.0 * overshoot
        return float(score)

def print_detailed_winner_report(trial_number: int, score: float, result: BacktestResult, cfg: StrategyConfig):
    """Prints a detailed terminal report for newly discovered winning strategy configurations."""
    sep = "=" * 80
    subsep = "-" * 80
    
    active_entry = [f"{s.name}({s.params})" for s in cfg.entry_slots if not s.is_empty()]
    active_exit = [f"{s.name}({s.params})" for s in cfg.exit_slots if not s.is_empty()]

    report = f"""
{sep}
 🏆 NEW WINNER FOUND | Trial #{trial_number} | Composite Score: {score:.4f}
{sep}
 💰 PORTFOLIO & FINANCIAL PERFORMANCE
{subsep}
  Total Capital Invested:   ₹{result.total_invested:,.2f}
  Final Portfolio Value:    ₹{result.final_value:,.2f}
  Total Net Profit (ROI):   ₹{result.total_profit:,.2f} ({result.total_return * 100:.2f}% on contributions, not annualized)
  XIRR (money-weighted):    {result.xirr * 100:.2f}%   <- what a SIP investor actually earned
  CAGR (time-weighted):     {result.cagr * 100:.2f}%   <- strategy skill, excludes contribution timing
  Max Drawdown (TWR-based): {result.max_drawdown * 100:.2f}%

 📊 RISK & RETURN RATIOS
{subsep}
  Sharpe Ratio:             {result.sharpe:.3f}
  Sortino Ratio:            {result.sortino:.3f}
  Calmar Ratio:             {result.calmar:.3f}
  Profit Factor:            {result.profit_factor:.2f}
  Alpha (ann., vs bench):   {result.alpha * 100:.2f}%
  Beta (vs bench):          {result.beta:.2f}
  Information Ratio:        {result.information_ratio:.3f}

 📈 TRADE EXECUTION STATS
{subsep}
  Total Closed Trades:      {result.n_trades}
  Win Rate:                 {result.win_rate * 100:.2f}%
  Average Trade P&L:        ₹{result.avg_trade_pnl:,.2f}
  Best Single Trade:        ₹{result.best_trade:,.2f}
  Worst Single Trade:       ₹{result.worst_trade:,.2f}

 ⚙️ STRATEGY HYPERPARAMETERS
{subsep}
  Heikin-Ashi Mode:         {cfg.use_heikin_ashi}
  Watchlist Model:          {cfg.watchlist_model}
  Take Profit Target:       {f'{cfg.take_profit_pct*100:.1f}%' if cfg.take_profit_pct else 'Disabled'}
  Trailing Stop Loss:       {f'{cfg.trailing_stop_pct*100:.1f}%' if cfg.trailing_stop_pct else 'Disabled'}
  ATR Chandelier Exit:      {f'{cfg.atr_chandelier_mult:.2f}x (ATR Period {cfg.atr_period})' if cfg.atr_chandelier_mult else 'Disabled'}
  Regime Filter:            {f'Enabled (SMA {cfg.regime_index_sma_period})' if cfg.regime_filter_enabled else 'Disabled'}
  Vol Sizing / Sector Cap:  {cfg.vol_sizing_enabled} / {cfg.sector_cap_pct*100:.0f}%
  Entry Indicators:         {', '.join(active_entry) if active_entry else 'None'}
  Exit Indicators:          {', '.join(active_exit) if active_exit else 'None'}
{sep}
"""
    print(report)

class StrategyOptimizer:
    """
    Wraps the whole StrategyConfig search space (indicator slots, HA toggle,
    exit params, watchlist model) into an Optuna study, with immediate JSON
    persistence of the current best config on every new benchmark break.
    """

    INDICATOR_NAMES = list(IndicatorLibrary.REGISTRY.keys())
    COMPARATORS = ["cross_above", "cross_below", "greater_than", "less_than"]
    WATCHLIST_MODELS = list(WatchlistEngine.MODELS)

    def __init__(
        self, market_data: Dict[str, pd.DataFrame], tickers: List[str], injection_days: List[int],
        benchmark: Optional[pd.Series] = None, objective_fn: Optional[ObjectiveFunction] = None,
        robust_objective: bool = False, robust_n_sip: int = 3, live_robustness_check: bool = True,
    ):
        self.market_data = market_data
        self.tickers = tickers
        self.injection_days = injection_days
        self.benchmark = benchmark
        self.objective_fn = objective_fn or ObjectiveFunction()
        # Opt-in "robustness-in-the-loop" mode: rather than only validating the winning
        # config's SIP-timing stability *after* the search (see run_full_pipeline's
        # post-hoc 15-run suite), each trial also runs a small number of extra SIP-date
        # backtests and folds their return std-dev into the objective as a penalty. This
        # directly selects against configs that only look good under one lucky cash-timing
        # schedule, at the cost of ~(1 + robust_n_sip)x backtests per trial.
        self.robust_objective = robust_objective
        self.robust_n_sip = robust_n_sip
        # Every time a NEW best is found (not every trial -- new winners get rarer as
        # search progresses, so the added cost is bounded), run a cheap SIP-sensitivity
        # + parameter-neighborhood check right then rather than waiting for the whole
        # search to finish. This surfaces an overfit "winner" in real time instead of
        # discovering it only in the post-hoc report after the full trial budget is
        # already spent. Deliberately lighter than the final suite (5 SIP schedules +
        # 2 neighbors here vs 15 + 5 for the true final winner).
        self.live_robustness_check = live_robustness_check
        self.best_score = -np.inf
        self.best_cfg: Optional[StrategyConfig] = None
        self.best_result: Optional[BacktestResult] = None

    def _suggest_slot(self, trial, prefix: str) -> IndicatorSlot:
        use_slot = trial.suggest_categorical(f"{prefix}_active", [True, False])
        if not use_slot:
            return IndicatorSlot(name=None)
        name = trial.suggest_categorical(f"{prefix}_name", self.INDICATOR_NAMES)
        spec = IndicatorLibrary.REGISTRY[name]["params"]
        params = {}
        for p, (lo, hi, _dv) in spec.items():
            if lo == hi:
                params[p] = lo
            else:
                params[p] = trial.suggest_int(f"{prefix}_{name}_{p}", lo, hi)
        comparator = trial.suggest_categorical(f"{prefix}_comparator", self.COMPARATORS)
        threshold = trial.suggest_float(f"{prefix}_threshold", 0.0, 100.0) if comparator in ("greater_than", "less_than") else 0.0
        return IndicatorSlot(name=name, params=params, comparator=comparator, threshold=threshold)

    def _suggest_config(self, trial) -> StrategyConfig:
        n_entry = trial.suggest_int("n_entry_slots", 1, MAX_ENTRY_SLOTS)
        n_exit = trial.suggest_int("n_exit_slots", 0, MAX_EXIT_SLOTS)
        entry_slots = [self._suggest_slot(trial, f"entry{i}") for i in range(n_entry)]
        exit_slots = [self._suggest_slot(trial, f"exit{i}") for i in range(n_exit)]

        use_tp = trial.suggest_categorical("use_tp", [True, False])
        use_tsl = trial.suggest_categorical("use_tsl", [True, False])
        use_atr_chandelier = trial.suggest_categorical("use_atr_chandelier", [True, False])

        return StrategyConfig(
            use_heikin_ashi=trial.suggest_categorical("use_heikin_ashi", [True, False]),
            entry_slots=entry_slots,
            exit_slots=exit_slots,
            take_profit_pct=trial.suggest_float("take_profit_pct", 0.05, 0.60) if use_tp else None,
            trailing_stop_pct=trial.suggest_float("trailing_stop_pct", 0.03, 0.30) if use_tsl else None,
            atr_chandelier_mult=trial.suggest_float("atr_chandelier_mult", 1.5, 5.0) if use_atr_chandelier else None,
            atr_period=trial.suggest_int("atr_period", 5, 60),
            watchlist_model=trial.suggest_categorical("watchlist_model", self.WATCHLIST_MODELS),
            watchlist_weight_price_dist=trial.suggest_float("wl_w_price_dist", 0.0, 1.0),
            watchlist_weight_age=trial.suggest_float("wl_w_age", 0.0, 1.0),
            watchlist_proximity_ma_period=trial.suggest_int("wl_ma_period", *PARAM_RANGE_GROUPS["moving_average"]),
            regime_filter_enabled=trial.suggest_categorical("regime_filter_enabled", [True, False]),
            regime_index_sma_period=trial.suggest_int("regime_sma_period", 50, 250),
            sector_cap_pct=trial.suggest_float("sector_cap_pct", 0.15, 1.0),
            vol_sizing_enabled=trial.suggest_categorical("vol_sizing_enabled", [True, False]),
            equal_weight_sizing_enabled=trial.suggest_categorical("equal_weight_sizing_enabled", [True, False]),
            cooldown_days=trial.suggest_int("cooldown_days", 0, 20),
        )

    def _persist_current_winner(self, cfg: StrategyConfig, score: float, result: BacktestResult, trial_number: int):
        # 1. Output the detailed visual report directly to the terminal
        print_detailed_winner_report(trial_number, score, result, cfg)

        # 1b. Lightweight live robustness check -- see live_robustness_check docstring
        # in __init__. NOT the full 15-SIP/5-neighbor post-hoc suite (that stays
        # reserved for the true final winner in run_full_pipeline; running the full
        # suite on every improving trial would be prohibitively expensive), but
        # enough to catch an obviously fragile "winner" in real time.
        live_robustness = None
        if self.live_robustness_check and result.n_trades > 0:
            suite = RobustnessSuite(self.market_data, benchmark=self.benchmark)
            sip = suite.sip_sensitivity(cfg, self.tickers, n_fixed=3, n_dynamic=2)
            neigh = suite.parameter_neighborhood_check(cfg, self.tickers, self.injection_days, n_neighbors=2)
            live_robustness = {
                "std_cagr_time_weighted": sip["std_cagr_time_weighted"],
                "std_xirr": sip["std_xirr"],
                "neighborhood_degradation_pct": neigh["mean_degradation_pct"],
                "neighborhood_stable": neigh["is_stable_plateau"],
            }
            logger.info(
                f"  Live robustness (5 SIP schedules, 2 param neighbors): "
                f"SIP-CAGR std={live_robustness['std_cagr_time_weighted']:.3%}, "
                f"neighborhood degradation={live_robustness['neighborhood_degradation_pct']:.1f}% "
                f"({'stable' if live_robustness['neighborhood_stable'] else 'UNSTABLE -- may be overfit'})"
            )

        # 2. Persist complete metrics to current_winner_config.json
        payload = {
            "trial_number": trial_number,
            "score": score,
            "total_invested": result.total_invested,
            "final_value": result.final_value,
            "total_profit": result.total_profit,
            "total_return": result.total_return,
            "xirr_money_weighted": result.xirr,
            "cagr_time_weighted": result.cagr,
            "max_drawdown": result.max_drawdown,
            "sharpe": result.sharpe,
            "sortino": result.sortino,
            "calmar": result.calmar,
            "alpha": result.alpha,
            "beta": result.beta,
            "information_ratio": result.information_ratio,
            "tracking_error": result.tracking_error,
            "profit_factor": result.profit_factor,
            "win_rate": result.win_rate,
            "n_trades": result.n_trades,
            "avg_trade_pnl": result.avg_trade_pnl,
            "best_trade": result.best_trade,
            "worst_trade": result.worst_trade,
            "live_robustness_check": live_robustness,
            "config": cfg.to_dict(),
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        with open(CURRENT_WINNER_PATH, "w") as f:
            json.dump(payload, f, indent=2, default=str)

    def objective(self, trial) -> float:
        cfg = self._suggest_config(trial)
        bt = Backtester(cfg, self.market_data, self.injection_days, benchmark=self.benchmark)
        result = bt.run(tickers=self.tickers)

        robustness_penalty = 0.0
        if self.robust_objective and result.n_trades > 0:
            suite = RobustnessSuite(self.market_data, benchmark=self.benchmark)
            sip_report = suite.sip_sensitivity(cfg, self.tickers, n_fixed=self.robust_n_sip, n_dynamic=0)
            robustness_penalty = sip_report["std_cagr_time_weighted"]

        score = self.objective_fn(result, robustness_penalty=robustness_penalty)

        if score > self.best_score:
            self.best_score = score
            self.best_cfg = cfg
            self.best_result = result
            self._persist_current_winner(cfg, score, result, trial.number)

        return score

    def optimize(self, n_trials: int = 50, n_jobs: int = 1, seed: int = 42,
                 seed_configs: Optional[List[StrategyConfig]] = None) -> StrategyConfig:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # "Memory": re-inject any previously-found winner(s) as guaranteed first trials
        # rather than starting the TPE sampler from nothing every run. TPE builds its
        # probability model from every observed trial, so seeding with known-good
        # configs gives it a strong prior to refine around instead of exploring blind.
        # See run_full_pipeline for where these are loaded + freshly re-validated.
        if seed_configs:
            for sc in seed_configs:
                try:
                    study.enqueue_trial(_config_to_optuna_params(
                        sc, self.INDICATOR_NAMES, self.COMPARATORS, self.WATCHLIST_MODELS))
                except Exception as e:
                    logger.warning(f"Could not enqueue seed config for warm-start: {e}")

        # CHANGED: Enable show_progress_bar=True for real-time optimization status
        study.optimize(
            self.objective, 
            n_trials=n_trials, 
            n_jobs=n_jobs, 
            show_progress_bar=True
        )

        if self.best_cfg is not None:
            final_payload = {
                "score": self.best_score, "sharpe": self.best_result.sharpe, "cagr_time_weighted": self.best_result.cagr,
                "xirr_money_weighted": self.best_result.xirr, "max_drawdown": self.best_result.max_drawdown,
                "total_return": self.best_result.total_return, "alpha": self.best_result.alpha,
                "beta": self.best_result.beta, "information_ratio": self.best_result.information_ratio,
                # BUGFIX: this previously referenced a bare `result`, which does not exist in this
                # method's scope (it only ever existed inside objective()) -- every completed
                # optimize() run raised NameError right here, after all trials had already finished.
                "n_trades": self.best_result.n_trades, "config": self.best_cfg.to_dict(),
                "n_trials_run": n_trials, "timestamp": pd.Timestamp.now().isoformat(),
            }
            with open(FINAL_WINNER_PATH, "w") as f:
                json.dump(final_payload, f, indent=2, default=str)
            logger.info(f"Optimization complete. Final winner written to {FINAL_WINNER_PATH}")

        return self.best_cfg

# ==============================================================================
# SECTION: MAIN ENTRY POINT
# ==============================================================================

def run_full_pipeline(
    n_trials: int = 50,
    universe_subset: Optional[List[str]] = None,
    years: int = 15,
    n_jobs: int = 1,
    run_robustness: bool = True,
    robust_objective: bool = False,
    robust_n_sip: int = 3,
    live_robustness_check: bool = True,
) -> None:
    """
    End-to-end driver:
      1. Download / cache universe data.
      2. Run Optuna hyperparameter search over StrategyConfig space, with
         immediate current_winner_config.json persistence on every new best,
         plus (if live_robustness_check) a lightweight SIP+neighborhood check
         on every new winner as it's found, not just at the very end.
      3. Refit the two-pass ML watchlist ranker against the winning config's
         empirical trade log, then run ONE comparison backtest swapping the
         heuristic watchlist model for the fitted ranker to see whether it
         actually improves on what the search found.
      4. Run the full robustness suite (15-run SIP sensitivity, parameter
         neighborhood check, walk-forward out-of-sample check, Deflated
         Sharpe Ratio, Monte Carlo permutation).
      5. Persist final_winner_config.json and a robustness report.
    """
    tickers = universe_subset or UNIVERSE
    logger.info(f"Universe size: {len(tickers)} tickers, lookback: {years}y")

    pipeline = DataPipeline(universe=tickers, years=years)
    market_data = pipeline.download_universe()
    if not market_data:
        logger.error("No market data available; aborting.")
        return

    # Benchmark proxy for the regime filter: equal-weight composite of the loaded universe.
    aligned_closes = pd.concat(
        [df["Close"].rename(t) for t, df in market_data.items()], axis=1
    ).sort_index().ffill()
    benchmark = aligned_closes.mean(axis=1)

    injection_days = [5]  # default single SIP date for the main optimization run

    # ---- Memory: re-validate any previously found winner(s) before searching further ----
    # Every run persists current_winner_config.json (best seen so far in that run) and
    # final_winner_config.json (that run's ultimate best). On a fresh run, load both if
    # present and re-run each on THIS run's full window (current universe/years -- which
    # may differ from whatever they were last found on) to check they're still relevant
    # rather than assuming stale numbers from a JSON file still hold. The best validated
    # prior becomes both (a) the new search's starting floor -- it only reports a "new
    # winner" if something genuinely beats history, not just beats a cold random start --
    # and (b) a guaranteed-first trial that warm-starts Optuna's TPE sampler instead of
    # exploring blind every single run.
    prior_paths = {"current_winner": CURRENT_WINNER_PATH, "final_winner": FINAL_WINNER_PATH}
    validated_priors: List[Tuple[StrategyConfig, "BacktestResult", float]] = []  # (cfg, result, score)
    seen_cfg_dicts = set()
    prelim_objective = ObjectiveFunction()
    for label, path in prior_paths.items():
        prior_cfg = _load_prior_winner_config(path)
        if prior_cfg is None:
            continue
        cfg_key = json.dumps(prior_cfg.to_dict(), sort_keys=True, default=str)
        if cfg_key in seen_cfg_dicts:
            continue  # current == final winner from a run that never improved past trial 0
        seen_cfg_dicts.add(cfg_key)

        logger.info(f"Found prior {label} config at {path} -- re-validating on the current full window...")
        prior_bt = Backtester(prior_cfg, market_data, injection_days, benchmark=benchmark)
        prior_res = prior_bt.run(tickers=list(market_data.keys()))
        prior_score = prelim_objective(prior_res)
        logger.info(
            f"  {label}: Sharpe={prior_res.sharpe:.3f}, CAGR(TWR)={prior_res.cagr:.3%}, "
            f"XIRR={prior_res.xirr:.3%}, MaxDD={prior_res.max_drawdown:.3%}, "
            f"trades={prior_res.n_trades}, score={prior_score:.4f} "
            f"-- {'still looks viable' if prior_res.n_trades > 0 and prior_score > 0 else 'no longer productive on this window'}"
        )
        validated_priors.append((prior_cfg, prior_res, prior_score))

    seed_result_start = None
    if validated_priors:
        validated_priors.sort(key=lambda x: x[2], reverse=True)
        best_prior_cfg, best_prior_res, best_prior_score = validated_priors[0]
        seed_result_start = (best_prior_cfg, best_prior_res, best_prior_score)
        logger.info(f"Using best validated prior (score={best_prior_score:.4f}) as this run's starting floor "
                     f"and warm-start seed for {len(validated_priors)} config(s).")

    optimizer = StrategyOptimizer(market_data, list(market_data.keys()), injection_days, benchmark=benchmark,
                                   robust_objective=robust_objective, robust_n_sip=robust_n_sip,
                                   live_robustness_check=live_robustness_check)
    if seed_result_start is not None:
        best_prior_cfg, best_prior_res, best_prior_score = seed_result_start
        optimizer.best_score = best_prior_score
        optimizer.best_cfg = best_prior_cfg
        optimizer.best_result = best_prior_res

    logger.info(f"Starting Optuna search: {n_trials} trials, n_jobs={n_jobs}, "
                f"robust_objective={robust_objective}" + (f" ({robust_n_sip} SIP runs/trial)" if robust_objective else ""))
    best_cfg = optimizer.optimize(n_trials=n_trials, n_jobs=n_jobs,
                                   seed_configs=[c for c, _, _ in validated_priors] if validated_priors else None)

    if best_cfg is None:
        logger.error("Optimization produced no valid strategy; aborting.")
        return

    logger.info(f"Best score: {optimizer.best_score:.4f} | Sharpe: {optimizer.best_result.sharpe:.3f} "
                f"| CAGR(TWR): {optimizer.best_result.cagr:.3%} | XIRR: {optimizer.best_result.xirr:.3%} "
                f"| MaxDD: {optimizer.best_result.max_drawdown:.3%} | Alpha: {optimizer.best_result.alpha:.3%} "
                f"| Beta: {optimizer.best_result.beta:.2f} | InfoRatio: {optimizer.best_result.information_ratio:.3f}")

    # ---- Two-pass ML watchlist ranker, fit on the winning config's realized trades ----
    ranker = MLWatchlistRanker()
    ranker.fit(market_data, optimizer.best_result.trade_log)
    logger.info(f"ML watchlist ranker fit complete. Empirical mean holding period (H_bar): {ranker.h_bar} days")

    # ---- Does the ML ranker actually help? ----
    # The ranker above is fit AFTER the winning config is already selected, using
    # whichever heuristic watchlist model that config happened to use -- so nothing
    # about the search itself validates the ML ranker's real worth. Run one more
    # simulation with the SAME winning config but watchlist_model swapped to the
    # fitted ML ranker, and compare directly. If it looks like a genuine improvement
    # (not just noise), verify it survives a lightweight SIP-date sensitivity check
    # before trusting it -- the same overfitting risk that applies to indicator
    # params applies here too.
    suite = RobustnessSuite(market_data, benchmark=benchmark)
    ml_comparison = None
    if ranker.model is not None:
        ml_cfg = dataclasses.replace(best_cfg, watchlist_model="ml_ranked")
        ml_bt = Backtester(ml_cfg, market_data, injection_days, benchmark=benchmark, ml_ranker=ranker)
        ml_res = ml_bt.run(tickers=list(market_data.keys()))

        base = optimizer.best_result
        logger.info(
            f"ML-ranked watchlist comparison (same config, only ranking model swapped):\n"
            f"    Sharpe:  {base.sharpe:.3f} -> {ml_res.sharpe:.3f}\n"
            f"    CAGR(TWR): {base.cagr:.3%} -> {ml_res.cagr:.3%}\n"
            f"    XIRR:      {base.xirr:.3%} -> {ml_res.xirr:.3%}\n"
            f"    MaxDD:     {base.max_drawdown:.3%} -> {ml_res.max_drawdown:.3%}\n"
            f"    Trades:    {base.n_trades} -> {ml_res.n_trades}"
        )
        # Require improvement on BOTH Sharpe and CAGR -- a model that trades one off
        # against the other (e.g. higher Sharpe from just trading less) isn't a clean win.
        ml_looks_better = ml_res.sharpe > base.sharpe and ml_res.cagr > base.cagr
        ml_sip_report = None
        if ml_looks_better:
            logger.info("  -> ML ranker looks better on both Sharpe and CAGR; verifying via lightweight SIP-date check...")
            ml_sip_report = suite.sip_sensitivity(ml_cfg, list(market_data.keys()), n_fixed=5, n_dynamic=2)
            logger.info(f"  ML-ranked SIP sensitivity: std_cagr(TWR)={ml_sip_report['std_cagr_time_weighted']:.3%} "
                        f"(compare to the base config's own SIP std reported below)")
        else:
            logger.info("  -> ML ranker did not clearly improve on the heuristic model; keeping the heuristic winner.")

        ml_comparison = {
            "baseline": {"sharpe": base.sharpe, "cagr_time_weighted": base.cagr, "xirr": base.xirr,
                         "max_drawdown": base.max_drawdown, "n_trades": base.n_trades,
                         "watchlist_model": best_cfg.watchlist_model},
            "ml_ranked": {"sharpe": ml_res.sharpe, "cagr_time_weighted": ml_res.cagr, "xirr": ml_res.xirr,
                          "max_drawdown": ml_res.max_drawdown, "n_trades": ml_res.n_trades},
            "ml_improved_sharpe_and_cagr": ml_looks_better,
            "ml_sip_sensitivity": ml_sip_report,
        }
        with open(os.path.join(RESULTS_DIR, "ml_watchlist_comparison.json"), "w") as f:
            json.dump(ml_comparison, f, indent=2, default=str)
    else:
        logger.info("ML ranker did not fit (insufficient training samples) -- skipping comparison run.")

    if not run_robustness:
        return

    # ---- Robustness / sensitivity / validation suite ----
    logger.info("Running 15-run SIP date sensitivity suite...")
    sip_report = suite.sip_sensitivity(best_cfg, list(market_data.keys()))
    logger.info(f"SIP sensitivity: mean_return={sip_report['mean_return']:.3%} (contribution-relative), "
                f"std_xirr={sip_report['std_xirr']:.3%}, std_cagr(TWR)={sip_report['std_cagr_time_weighted']:.3%} "
                f"across {sip_report['schedules_run']} schedules -- the TWR std is the real skill-stability signal")

    logger.info("Running parameter neighborhood stability check...")
    neighborhood_report = suite.parameter_neighborhood_check(best_cfg, list(market_data.keys()), injection_days)
    logger.info(f"Neighborhood check: mean degradation={neighborhood_report['mean_degradation_pct']:.1f}%, "
                f"stable_plateau={neighborhood_report['is_stable_plateau']}")

    logger.info("Running walk-forward out-of-sample check (4 anchored expanding folds)...")
    walk_forward_report = suite.walk_forward_expanding_folds(best_cfg, list(market_data.keys()), injection_days, n_folds=4)
    if "error" not in walk_forward_report:
        fold_summary = ", ".join(
            f"F{f['fold']}:{f['sharpe_decay_pct']:.0f}%" for f in walk_forward_report["folds"]
        )
        logger.info(
            f"Walk-forward (4 folds): mean Sharpe decay={walk_forward_report['mean_sharpe_decay_pct']:.1f}% "
            f"[{fold_summary}], {walk_forward_report['n_severe_folds']}/{walk_forward_report['n_folds']} folds severe | "
            f"{'*** LIKELY OVERFIT ***' if walk_forward_report['likely_overfit'] else 'holds up out-of-sample'}"
        )

    logger.info("Running per-asset consistency check...")
    per_asset_report = RobustnessSuite.per_asset_consistency_check(optimizer.best_result.trade_log)
    logger.info(
        f"Per-asset consistency: {per_asset_report['n_profitable']}/{per_asset_report['n_tickers_traded']} "
        f"tickers individually profitable ({per_asset_report['frac_profitable']:.1%}) | "
        f"{'PASS' if per_asset_report['passes_gate'] else '*** FAIL -- edge may be 1-2 lucky tickers, not systemic ***'}"
    )

    logger.info("Running annual profit concentration check...")
    concentration_report = RobustnessSuite.annual_profit_concentration_check(optimizer.best_result.trade_log)
    logger.info(
        f"Annual concentration: single worst year = {concentration_report['max_year_frac']:.1%} of total profit | "
        f"{'PASS' if concentration_report['passes_gate'] else '*** FAIL -- may be a single lucky year, not a repeatable edge ***'}"
    )

    # NOTE: previously this used equity_curve.pct_change(), which records every SIP
    # injection day as a huge fake "return" (including an inf on day 1) -- same root
    # cause as the CAGR bug. Use the TWR-adjusted series stored on BacktestResult instead.
    daily_returns = optimizer.best_result.twr_daily_returns
    dsr = RobustnessSuite.deflated_sharpe_ratio(daily_returns, n_trials=n_trials)
    logger.info(f"Deflated Sharpe Ratio (n_trials={n_trials}): {dsr:.4f}")

    trade_pnls = [t["pnl"] for t in optimizer.best_result.trade_log if t.get("action") == "SELL"]
    mc_report = RobustnessSuite.monte_carlo_permutation(trade_pnls, n_paths=1000)
    logger.info(f"Monte Carlo (1000 paths): 95th pct drawdown={mc_report['p95_drawdown']:.2f}, "
                f"risk_of_ruin={mc_report['risk_of_ruin']:.3%}")

    robustness_payload = {
        "sip_sensitivity": sip_report,
        "parameter_neighborhood": neighborhood_report,
        "per_asset_consistency": per_asset_report,
        "annual_profit_concentration": concentration_report,
        "walk_forward_out_of_sample": walk_forward_report,
        "deflated_sharpe_ratio": dsr,
        "monte_carlo": mc_report,
        "ml_ranker_h_bar_days": ranker.h_bar,
        "ml_watchlist_comparison": ml_comparison,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    with open(os.path.join(RESULTS_DIR, "robustness_report.json"), "w") as f:
        json.dump(robustness_payload, f, indent=2, default=str)
    logger.info(f"Robustness report written to {os.path.join(RESULTS_DIR, 'robustness_report.json')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-asset momentum strategy backtester & optimizer")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--years", type=int, default=15, help="Years of daily history to backtest")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel Optuna workers (n_jobs)")
    parser.add_argument("--universe-limit", type=int, default=None,
                         help="Optional cap on number of tickers (useful for a fast local smoke test)")
    parser.add_argument("--skip-robustness", action="store_true", help="Skip the robustness/sensitivity suite")
    parser.add_argument("--skip-live-robustness", action="store_true",
                         help="Skip the lightweight SIP+neighborhood check that otherwise runs every time a "
                              "new best trial is found during search (not just at the very end)")
    parser.add_argument("--robust-objective", action="store_true",
                         help="Penalize SIP-timing instability inside every trial's objective, not just post-hoc "
                              "(costs ~1+robust-n-sip extra backtests per trial)")
    parser.add_argument("--robust-n-sip", type=int, default=3,
                         help="Number of extra SIP schedules run per trial when --robust-objective is set")
    args = parser.parse_args()

    subset = UNIVERSE[: args.universe_limit] if args.universe_limit else None
    run_full_pipeline(
        n_trials=args.trials, universe_subset=subset, years=args.years,
        n_jobs=args.jobs, run_robustness=not args.skip_robustness,
        robust_objective=args.robust_objective, robust_n_sip=args.robust_n_sip,
        live_robustness_check=not args.skip_live_robustness,
    )
