#!/usr/bin/env python3
"""
NSE EQUITY QUANTITATIVE SWING-TRADING FRAMEWORK (V10.1 - PRODUCTION PROMOTION ENGINE)
======================================================================================
Engine & Selection Corrections:
1. WALK-FORWARD CHAMPION VISIBILITY: Dedicated Row #1 in Section 4 always highlights the
   promoted champion with full fold breakdowns, regardless of raw Optuna rank.
2. AUDITED MULTI-FOLD PROMOTION DIAGNOSTICS: Surfaces clear diagnostic reporting on whether
   candidates cleared all folds, Fold 3, or fell back to best available margin.
3. EMPIRICAL PLACEBO PERCENTILE RANKING: Computes and displays the exact percentile rank of
   strategy CAGR against 50 ADX-conditioned placebo seeds instead of a binary median flag.
4. TRAILING STOP MECHANISM ABLATION: Counterfactual evaluation of trailing exits vs fixed
   exits across both In-Sample and Out-of-Sample windows.
5. FIXED CONDITIONAL PARAMETER INTEGRITY: Eliminates dynamic categorical distribution errors
   in Optuna while guaranteeing complete parameter records via deterministic reconstitution.
6. REALISTIC CONCENTRATION TELEMETRY: Clarifies entry-time position caps vs organic market
   appreciation in exposure telemetry tables.
7. ZERO CASH YIELD MANDATE: CASH_ANNUAL_YIELD = 0.0% preserved (all CAGR is pure trade alpha).
"""

from __future__ import annotations

import os
import io
import sys
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import optuna

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ==============================================================================
# USER CONTROL PANEL
# ==============================================================================

ENGINE_VERSION: str = "V10.1"
UNIVERSE_NAME: str = "NIFTY50"

UNIVERSE_CONSTITUENT_FILES: Dict[str, str] = {
    "NIFTY50":     "constituents/ind_nifty50list.csv",
    "NIFTY100":    "constituents/ind_nifty100list.csv",
    "MIDCAP150":   "constituents/ind_niftymidcap150list.csv",
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
    "SMALLCAP250": [
        "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
    ],
}

MACRO_INDEX_NAME: str = "NIFTY50"
MACRO_INDEX_TICKER: str = "^NSEI"

FORCE_MACRO_REGIME_FILTER: Optional[bool] = None
FORCE_TRAIL_STOP: Optional[bool] = None  # Set True to lock trailing stops off (trail_atr_mult=0.0, trail_pct=0.0)

START_YEAR: int = 2008
QUOTE_CURRENCY: str = "INR"
INITIAL_CAPITAL: float = 100000.0
MONTHLY_CONTRIBUTION: float = 10000.0
CASH_ANNUAL_YIELD: float = 0.0              # Strictly 0.0% cash yield (pure trade alpha)
MIN_HISTORY_DAYS: int = 300
WARMUP_BARS: int = 500
N_TRIALS: int = 1000
PARALLEL_DOWNLOAD_WORKERS: int = 8
FORCE_REFRESH: bool = False
CACHE_MAX_AGE_HOURS: float = 72.0
WL_MAX_AGE_BARS: int = 15

UNIVERSE_LIQUIDITY_FLOOR_INR: Dict[str, float] = {
    "NIFTY50":      0.0,
    "NIFTY100":     0.0,
    "MIDCAP150":    5_000_000.0,
    "SMALLCAP250":  1_000_000.0,
}
LIQUIDITY_FLOOR_INR: float = UNIVERSE_LIQUIDITY_FLOOR_INR[UNIVERSE_NAME]

TRANCHE_FLOOR_INR: float = 10_000.0
DEFAULT_MAX_CONCURRENT_TRANCHES: int = 8
MAX_POSITION_EQUITY_PCT: float = 0.25      # Hard cap on aggregate allocation per scrip at entry
MAX_ADV_PARTICIPATION: float = 0.015

UNIVERSE_SLIPPAGE_BPS: Dict[str, float] = {
    "NIFTY50": 5.0, "NIFTY100": 8.0, "MIDCAP150": 15.0, "SMALLCAP250": 25.0,
}
UNIVERSE_IMPACT_COEF_BPS: Dict[str, float] = {
    "NIFTY50": 80.0, "NIFTY100": 120.0, "MIDCAP150": 200.0, "SMALLCAP250": 320.0,
}
BASE_SLIPPAGE_BPS: float = UNIVERSE_SLIPPAGE_BPS[UNIVERSE_NAME]
IMPACT_COEF_BPS: float = UNIVERSE_IMPACT_COEF_BPS[UNIVERSE_NAME]
NEIGHBORHOOD_DROP_LIMIT: float = 0.25

# Statutory Charges (NSE Delivery Rates Card)
BROKERAGE_MODE: str = "ZERO_DELIVERY"
FLAT_BROKERAGE_INR: float = 20.0
STT_RATE: float = 0.0010
EXCHANGE_TXN_RATE: float = 0.0000322
STAMP_DUTY_RATE: float = 0.00015
SEBI_CHARGE_RATE: float = 0.000001
GST_RATE: float = 0.18
DP_CHARGE_INR: float = 15.0
DP_CHARGE_GST: float = DP_CHARGE_INR * GST_RATE

EFFECTIVE_BUY_FEE_RATE: float = (
    STT_RATE + EXCHANGE_TXN_RATE + STAMP_DUTY_RATE + SEBI_CHARGE_RATE
    + GST_RATE * (EXCHANGE_TXN_RATE + SEBI_CHARGE_RATE)
)
EFFECTIVE_SELL_FEE_RATE: float = (
    STT_RATE + EXCHANGE_TXN_RATE + SEBI_CHARGE_RATE
    + GST_RATE * (EXCHANGE_TXN_RATE + SEBI_CHARGE_RATE)
)

DATA_DIR = Path("data_cache")
OUTPUT_DIR = Path("output") / UNIVERSE_NAME
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_WINNER_FILE = OUTPUT_DIR / "current_winner.json"
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

    @property
    def total_frictions(self) -> float:
        return (self.brokerage + self.stt + self.exchange_charges +
                self.stamp_duty + self.sebi_charges + self.gst +
                self.dp_charges + self.slippage_cost)


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


def compute_max_affordable_tranche(cash: float, adv_30d: float) -> float:
    clean_adv = adv_30d if np.isfinite(adv_30d) else 1_000_000.0
    adv = max(clean_adv, 1_000_000.0)
    flat_buy_charge = (FLAT_BROKERAGE_INR * (1.0 + GST_RATE)) if BROKERAGE_MODE == "FLAT" else 0.0
    usable_cash = max(0.0, cash - flat_buy_charge)
    tranche_guess = usable_cash / (1.0 + EFFECTIVE_BUY_FEE_RATE + BASE_SLIPPAGE_BPS / 10000.0)
    for _ in range(3):
        part_rate = min(1.0, max(0.0, tranche_guess / adv))
        slip_mult = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate)) / 10000.0
        tranche_guess = usable_cash / (1.0 + EFFECTIVE_BUY_FEE_RATE + slip_mult)
    return float(np.nan_to_num(tranche_guess * (1.0 - 1e-6), nan=0.0))


def _buy_fill_audited(tranche_inr: float, ref_open_price: float, slip_mult: float) -> Tuple[float, float, float, FeeBreakdown]:
    fill_px = ref_open_price * (1.0 + slip_mult)
    units = tranche_inr / ref_open_price if ref_open_price > 0 else 0.0
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
    raw = raw.dropna(subset=["close"])
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


class FastIndicators:
    @staticmethod
    def moving_average(arr: np.ndarray, length: int, kind: int) -> np.ndarray:
        length = max(2, int(length))
        if kind == 0:   return np_rolling_mean(arr, length)
        elif kind == 1: return np_ewm_mean(arr, length)
        elif kind == 2:
            e1 = np_ewm_mean(arr, length)
            e2 = np_ewm_mean(e1, length)
            return 2.0 * e1 - e2
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
        elif kind == 4:
            alpha = 1.0 / length
            span = int(round((2.0 / alpha) - 1.0))
            return np_ewm_mean(arr, span)
        return np_rolling_mean(arr, length)

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


def build_market_universe(universe_name: str, start_year: int, refresh: bool) -> Tuple[MarketGrid, Dict[str, pd.DataFrame], pd.DataFrame, List[Dict[str, Any]]]:
    symbols = load_universe_constituents(universe_name)
    _, macro_df, _ = fetch_single_stock(MACRO_INDEX_TICKER, start_year, refresh)
    if macro_df is None:
        raise RuntimeError(f"Failed to load macro index data for {MACRO_INDEX_TICKER}.")

    raw_universe: Dict[str, pd.DataFrame] = {}
    provider_map: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=PARALLEL_DOWNLOAD_WORKERS) as executor:
        future_map = {executor.submit(fetch_single_stock, sym, start_year, refresh): sym for sym in symbols}
        for future in as_completed(future_map):
            sym, df, provider = future.result()
            if df is not None and len(df) >= MIN_HISTORY_DAYS:
                raw_universe[sym] = df
                provider_map[sym] = provider

    master_dates = pd.DatetimeIndex(sorted(macro_df.index.unique())).normalize()
    macro_df = macro_df.reindex(master_dates).ffill().dropna(subset=["close"])

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
        dvol30 = np_rolling_mean(qv, 30)
        dvol30 = np.nan_to_num(dvol30, nan=1_000_000.0)

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

        provenance_records.append({
            "symbol": sym, "provider": provider_map.get(sym, "unknown"),
            "first_date": str(first_idx.date()), "last_date": str(last_idx.date()),
            "total_bars": len(df), "is_delisted": bool(is_delisted_perm[-1]) if len(is_delisted_perm) else False
        })

    grid = MarketGrid(
        symbols=valid_symbols, dates=master_dates,
        open_mat=open_mat, high_mat=high_mat, low_mat=low_mat, close_mat=close_mat, volume_mat=vol_mat,
        atr14_mat=atr14_mat, dvol30_mat=dvol30_mat, adx14_mat=adx14_mat,
        delist_mat=delist_mat, alive_mat=alive_mat,
        macro_close=macro_df["close"].values, macro_open=macro_df["open"].values,
        months_arr=master_dates.month.values, years_arr=master_dates.year.values
    )
    return grid, universe, macro_df, provenance_records


# ==============================================================================
# COMPILE SIGNALS & FUNNEL ATTRIBUTES
# ==============================================================================
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
            v_shifted = shift_1d(v_arr)
            vma = np_rolling_mean(v_shifted, int(p["vol_ma_len"]))
            h_shifted = shift_1d(h_arr)
            hhv = np_rolling_max(h_shifted, int(p["price_lookback"]))
            body = c_arr - o_arr
            raw_entry = (~np.isnan(vma)) & (~np.isnan(hhv)) & (v_arr > (p["vol_mult"] * vma)) & (c_arr > hhv) & (body >= (p["body_atr_mult"] * grid.atr14_mat[i, :]))
            base_ma = FastIndicators.moving_average(c_arr, int(p["price_lookback"]), 0)
            state_entry = (~np.isnan(base_ma)) & (c_arr > base_ma)
        elif et == 4:
            b_up, b_mid, _ = FastIndicators.bollinger_bands(c_arr, p["bb_entry_len"], p["bb_entry_std"])
            raw_entry = (~np.isnan(b_up)) & (c_arr > b_up) & (shift_1d(c_arr) <= shift_1d(b_up))
            state_entry = (~np.isnan(b_mid)) & (c_arr > b_mid)

        raw_signal_mat[i, :] = raw_entry & grid.alive_mat[i, :]
        liq_ok = grid.dvol30_mat[i, :] >= LIQUIDITY_FLOOR_INR
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
    tid: int
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
    trough_bar: int
    layer: int
    tp_done: bool = False
    tp_proceeds: float = 0.0
    from_watchlist: bool = False
    wait_days: int = 0
    proceeds: float = 0.0
    fee_acc: FeeBreakdown = field(default_factory=FeeBreakdown)


class BacktestEngine:
    def __init__(self, params: Dict[str, Any]):
        self.p = params

    def run_interval(self, grid: MarketGrid, raw_signal_mat: np.ndarray, entry_mat: np.ndarray,
                     exit_mat: np.ndarray, macro_ok: np.ndarray,
                     start_bar: int, end_bar: int, active_symbols_mask: Optional[np.ndarray] = None,
                     state_mat: Optional[np.ndarray] = None) -> Dict[str, Any]:
        assert start_bar >= 1, "start_bar must be >= 1 for causal lookback."
        p = self.p
        dates = grid.dates
        max_holding_bars = p.get("max_holding_bars", 20)
        max_slots = p.get("max_concurrent_tranches", DEFAULT_MAX_CONCURRENT_TRANCHES)
        be_trigger_mult = p.get("be_trigger_atr", 0.0)
        macro_active_exit = p.get("macro_active_exit", False)
        adx_t = p.get("adx_thresh", 0.0)

        active_state_mat = state_mat if state_mat is not None else entry_mat

        n_syms = len(grid.symbols)
        cash = INITIAL_CAPITAL
        total_inflow = INITIAL_CAPITAL
        live_tranches: List[Tranche] = []
        watchlist: List[Dict[str, Any]] = []
        closed_trades: List[Dict[str, Any]] = []

        n_interval = end_bar - start_bar
        active_capital_curve = np.zeros(n_interval, dtype=np.float64)
        total_equity_curve = np.zeros(n_interval, dtype=np.float64)
        twr_curve = np.zeros(n_interval, dtype=np.float64)
        concurrency_hist = np.zeros(max_slots + 1, dtype=np.int64)

        tid_counter = 0
        sig_id_counter = 0

        funnel_dispositions = {
            "raw_triggers": 0, "macro_blocked": 0, "liquidity_blocked": 0,
            "adx_blocked": 0, "pyramid_blocked": 0, "revalidation_dropped": 0,
            "slot_saturated_dropped": 0, "cash_starved_dropped": 0,
            "risk_cap_dropped": 0, "expired_unfilled": 0, "executed_fills": 0
        }

        binding_stats = {"tranche_ceiling": 0, "liquidity_cap": 0, "equity_slot": 0, "tranche_floor": 0, "cash_constrained": 0}
        max_pyramid = p.get("max_pyramid_layers", 1)
        trail_mult = p.get("trail_atr_mult", 0.0)
        trail_pct_mult = (1.0 - (p.get("trail_pct", 10.0) / 100.0))
        use_tp = p.get("use_global_tp", False)
        tp_mult = p.get("tp_mult", 4.0)
        tp_size = p.get("tp_size_pct", 50.0) / 100.0
        tp_be = p.get("tp_move_sl_be", False)
        xt = p["exit_type"]
        wl_mode = p.get("wl_mode", "WL_NONE")

        current_twr = 1.0
        prev_equity = cash
        peak_single_stock_pct = 0.0

        for idx, t in enumerate(range(start_bar, end_bar)):
            curr_date = dates[t]
            inflow_today = 0.0

            # 0. Monthly Systematic Inflow
            if t > start_bar and grid.months_arr[t] != grid.months_arr[t - 1]:
                cash += MONTHLY_CONTRIBUTION
                total_inflow += MONTHLY_CONTRIBUTION
                inflow_today = MONTHLY_CONTRIBUTION

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
                    })
                else:
                    surviving_tranches.append(tr)
            live_tranches = surviving_tranches

            # 2. Strict Causal Intraday Position Management
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
                h_bar = grid.high_mat[c_i, t]
                l_bar = grid.low_mat[c_i, t]
                bars_held = t - tr.entry_bar

                exit_triggered = False
                exit_reason = ""
                raw_exit_px = o_bar

                # Phase 2A: Open Exits (non-stop-loss triggers only; these already follow the
                # signal-on-T-1/fill-on-T-open convention used elsewhere and are unaffected
                # by the stop-loss change below)
                if macro_bear_confirmed:
                    exit_triggered = True
                    exit_reason = "MACRO_REGIME_EXIT"
                    raw_exit_px = o_bar
                elif bars_held >= max_holding_bars:
                    exit_triggered = True
                    exit_reason = "MAX_HOLDING_TIME"
                    raw_exit_px = o_bar
                elif xt in (3, 4, 5, 6, 7) and exit_mat[c_i, t - 1]:
                    if bars_held >= 3 or (tr.highest_high - tr.entry_price) >= (1.0 * tr.entry_atr):
                        exit_triggered = True
                        exit_reason = f"SIGNAL_EXIT_TYPE_{xt}"
                        raw_exit_px = o_bar

                # Phase 2B: Close-of-Bar Exits. Stop-loss is decided off the EOD Close (not the
                # intraday Low, and not a same-bar Open gap-through) because the live companion
                # bot only ever observes one already-closed daily bar per run and has no way to
                # react to intrabar prices or gaps. Take-profit still uses the intraday High,
                # since the live bot checks that the same way.
                if not exit_triggered:
                    c_bar = grid.close_mat[c_i, t]
                    sl_breached = (c_bar <= tr.current_sl)
                    tp_price = tr.entry_price + (tp_mult * tr.entry_atr)
                    tp_breached = (use_tp and not tr.tp_done and (h_bar >= tp_price))

                    if sl_breached and tp_breached:
                        exit_triggered = True
                        exit_reason = tr.stop_reason
                        raw_exit_px = c_bar
                    elif sl_breached:
                        exit_triggered = True
                        exit_reason = tr.stop_reason
                        raw_exit_px = c_bar
                    elif tp_breached:
                        close_units = tr.units * tp_size
                        part_rate_tp = min(1.0, max(0.0, (close_units * tp_price) / adv))
                        slip_tp = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate_tp)) / 10000.0
                        _, credit, fee_tp = _sell_fill_audited(close_units, max(o_bar, tp_price), slip_tp, apply_dp=True)
                        cash += credit
                        tr.proceeds += credit
                        tr.tp_proceeds += credit
                        tr.units -= close_units
                        tr.tp_done = True
                        tr.fee_acc.brokerage += fee_tp.brokerage
                        tr.fee_acc.stt += fee_tp.stt
                        tr.fee_acc.exchange_charges += fee_tp.exchange_charges
                        tr.fee_acc.sebi_charges += fee_tp.sebi_charges
                        tr.fee_acc.gst += fee_tp.gst
                        tr.fee_acc.dp_charges += fee_tp.dp_charges
                        tr.fee_acc.slippage_cost += fee_tp.slippage_cost

                        if tp_be and tr.current_sl < (tr.entry_price * 1.002):
                            tr.current_sl = tr.entry_price * 1.002
                            tr.stop_reason = "BREAKEVEN_SL"

                if exit_triggered:
                    part_rate_exit = min(1.0, max(0.0, (tr.units * raw_exit_px) / adv))
                    slip_exit = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate_exit)) / 10000.0
                    fill_px, proceeds, fee_exit = _sell_fill_audited(tr.units, raw_exit_px, slip_exit, apply_dp=True)

                    cash += proceeds
                    total_proceeds = tr.proceeds + proceeds
                    pnl = total_proceeds - tr.cost_inr
                    vwap_exit = total_proceeds / tr.initial_units if tr.initial_units > 0 else fill_px

                    tr.fee_acc.brokerage += fee_exit.brokerage
                    tr.fee_acc.stt += fee_exit.stt
                    tr.fee_acc.exchange_charges += fee_exit.exchange_charges
                    tr.fee_acc.sebi_charges += fee_exit.sebi_charges
                    tr.fee_acc.gst += fee_exit.gst
                    tr.fee_acc.dp_charges += fee_exit.dp_charges
                    tr.fee_acc.slippage_cost += fee_exit.slippage_cost

                    closed_trades.append({
                        "coin": tr.coin, "pnl": pnl, "ret": pnl / tr.cost_inr if tr.cost_inr > 0 else 0.0,
                        "reason": exit_reason, "bars": bars_held, "bars_to_peak": tr.peak_bar - tr.entry_bar,
                        "mfe_pct": ((tr.highest_high - tr.entry_price) / tr.entry_price) * 100.0,
                        "mae_pct": ((tr.lowest_low - tr.entry_price) / tr.entry_price) * 100.0,
                        "entry_date": tr.entry_date, "exit_date": curr_date, "exit_idx": idx,
                        "entry_price": tr.entry_price, "exit_price": vwap_exit,
                        "cost_inr": tr.cost_inr, "proceeds": total_proceeds,
                        "frictions": tr.fee_acc, "from_watchlist": tr.from_watchlist,
                    })
                else:
                    if h_bar > tr.highest_high:
                        tr.highest_high = h_bar
                        tr.peak_bar = t
                    if l_bar < tr.lowest_low:
                        tr.lowest_low = l_bar
                        tr.trough_bar = t

                    if be_trigger_mult > 0.0 and tr.current_sl < (tr.entry_price * 1.002):
                        be_target = tr.entry_price + (be_trigger_mult * tr.entry_atr)
                        if tr.highest_high >= be_target:
                            tr.current_sl = max(tr.current_sl, tr.entry_price * 1.002)
                            tr.stop_reason = "BREAKEVEN_SL"

                    if xt == 1:
                        pct_floor = tr.highest_high * trail_pct_mult
                        if pct_floor > tr.current_sl:
                            tr.current_sl = pct_floor
                            tr.stop_reason = "TRAIL_PCT_STOP"
                    elif trail_mult > 0.0:
                        atr_bar_trailing = grid.atr14_mat[c_i, t - 1]
                        atr_floor = tr.highest_high - (trail_mult * atr_bar_trailing)
                        if atr_floor > tr.current_sl:
                            tr.current_sl = atr_floor
                            tr.stop_reason = "TRAIL_ATR_STOP"

                    surviving_tranches.append(tr)
            live_tranches = surviving_tranches

            # 3. Ingest Candidates from Verified `entry_mat`
            active_counts = {tr.coin_idx: 0 for tr in live_tranches}
            for tr in live_tranches:
                active_counts[tr.coin_idx] += 1

            new_candidates = []
            for c_i in range(n_syms):
                if active_symbols_mask is not None and not active_symbols_mask[c_i]:
                    continue

                if raw_signal_mat[c_i, t - 1]:
                    sig_id_counter += 1
                    funnel_dispositions["raw_triggers"] += 1

                    if not macro_ok[t - 1]:
                        funnel_dispositions["macro_blocked"] += 1
                        continue
                    if grid.dvol30_mat[c_i, t - 1] < LIQUIDITY_FLOOR_INR:
                        funnel_dispositions["liquidity_blocked"] += 1
                        continue
                    if not entry_mat[c_i, t - 1]:
                        funnel_dispositions["adx_blocked"] += 1
                        continue
                    if active_counts.get(c_i, 0) >= max_pyramid:
                        funnel_dispositions["pyramid_blocked"] += 1
                        continue

                    o_today = grid.open_mat[c_i, t]
                    a_yesterday = grid.atr14_mat[c_i, t - 1]
                    init_stop = o_today - (p.get("sl_mult", 3.0) * a_yesterday)
                    breakout_strength = (grid.close_mat[c_i, t - 1] - grid.open_mat[c_i, t - 1]) / max(1e-6, a_yesterday)
                    new_candidates.append({
                        "sig_id": sig_id_counter,
                        "coin": grid.symbols[c_i], "coin_idx": c_i, "signal_bar": t,
                        "trigger_price": grid.close_mat[c_i, t - 1], "shadow_stop": init_stop,
                        "highest_high": o_today, "breakout_quality": breakout_strength,
                        "entry_atr": a_yesterday, "unfilled_reason": None
                    })

            new_candidates.sort(key=lambda x: x["breakout_quality"], reverse=True)
            watchlist.extend(new_candidates)

            # 4. Sizing & Order Fill with Persistent State Revalidation
            def _record_unfilled_disposition(reason: Optional[str], default_bucket: str = "expired_unfilled"):
                if reason == "cash_starved":
                    funnel_dispositions["cash_starved_dropped"] += 1
                elif reason in ("pyramid_blocked", "anti_averaging_down"):
                    funnel_dispositions["pyramid_blocked"] += 1
                elif reason == "scrip_risk_cap":
                    funnel_dispositions["risk_cap_dropped"] += 1
                elif reason == "slot_saturated":
                    funnel_dispositions["slot_saturated_dropped"] += 1
                else:
                    funnel_dispositions[default_bucket] += 1

            if watchlist and cash >= TRANCHE_FLOOR_INR and len(live_tranches) < max_slots:
                if wl_mode == "WL_DEEPEST_DISCOUNT":
                    watchlist.sort(key=lambda x: (x["trigger_price"] - grid.close_mat[x["coin_idx"], t - 1]) / max(1e-6, x["trigger_price"]), reverse=True)
                elif wl_mode == "WL_STRONGEST_MOMENTUM":
                    watchlist.sort(key=lambda x: x["breakout_quality"], reverse=True)

                unfilled = []
                for item in watchlist:
                    c_i = item["coin_idx"]

                    if not grid.alive_mat[c_i, t]:
                        funnel_dispositions["revalidation_dropped"] += 1
                        continue

                    if t > item["signal_bar"]:
                        if not macro_ok[t - 1]:
                            funnel_dispositions["revalidation_dropped"] += 1
                            continue
                        if adx_t > 0.0 and grid.adx14_mat[c_i, t - 1] < adx_t:
                            funnel_dispositions["revalidation_dropped"] += 1
                            continue
                        if not active_state_mat[c_i, t - 1]:
                            funnel_dispositions["revalidation_dropped"] += 1
                            continue

                    coin_layers = sum(1 for tr in live_tranches if tr.coin_idx == c_i)
                    today_open = grid.open_mat[c_i, t]

                    # Anti-Averaging Down Rule
                    if coin_layers > 0:
                        highest_prior_entry = max(tr.entry_price for tr in live_tranches if tr.coin_idx == c_i)
                        if today_open <= highest_prior_entry * 1.005:
                            item["unfilled_reason"] = "anti_averaging_down"
                            unfilled.append(item)
                            continue

                    open_slots = max(1, max_slots - len(live_tranches))
                    if len(live_tranches) >= max_slots:
                        item["unfilled_reason"] = "slot_saturated"
                        unfilled.append(item)
                        continue
                    if cash < TRANCHE_FLOOR_INR:
                        item["unfilled_reason"] = "cash_starved"
                        unfilled.append(item)
                        continue

                    open_active_cap = sum(tr.units * grid.open_mat[tr.coin_idx, t] for tr in live_tranches)
                    current_equity = cash + open_active_cap
                    max_pos_cap = current_equity * MAX_POSITION_EQUITY_PCT
                    dynamic_slot_target = min(max_pos_cap, cash / float(open_slots))

                    current_scrip_exposure = sum(tr.units * today_open for tr in live_tranches if tr.coin_idx == c_i)
                    remaining_scrip_capacity = max(0.0, max_pos_cap - current_scrip_exposure)
                    if remaining_scrip_capacity < TRANCHE_FLOOR_INR:
                        item["unfilled_reason"] = "scrip_risk_cap"
                        unfilled.append(item)
                        continue

                    adv_30d = max(grid.dvol30_mat[c_i, t - 1], 1_000_000.0)
                    liquidity_cap_inr = adv_30d * MAX_ADV_PARTICIPATION
                    scaled_tranche_ceiling = max(500_000.0, current_equity * 0.35)
                    target_inr = min(dynamic_slot_target, scaled_tranche_ceiling, liquidity_cap_inr, remaining_scrip_capacity)

                    max_affordable = compute_max_affordable_tranche(cash, adv_30d)
                    tranche_inr = min(max_affordable, max(TRANCHE_FLOOR_INR, target_inr))

                    part_rate = min(1.0, max(0.0, tranche_inr / adv_30d))
                    slip_mult = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate)) / 10000.0

                    fill_px, units, total_cost, fee_buy = _buy_fill_audited(tranche_inr, today_open, slip_mult)

                    if (cash >= total_cost and tranche_inr >= TRANCHE_FLOOR_INR and units > 0 and
                            len(live_tranches) < max_slots and coin_layers < max_pyramid):

                        if max_affordable < target_inr:
                            binding_stats["cash_constrained"] += 1
                        elif target_inr == scaled_tranche_ceiling:
                            binding_stats["tranche_ceiling"] += 1
                        elif target_inr == liquidity_cap_inr:
                            binding_stats["liquidity_cap"] += 1
                        elif target_inr < TRANCHE_FLOOR_INR:
                            binding_stats["tranche_floor"] += 1
                        else:
                            binding_stats["equity_slot"] += 1

                        sl_price = fill_px - (p.get("sl_mult", 3.0) * item["entry_atr"])
                        cash -= total_cost
                        tid_counter += 1
                        funnel_dispositions["executed_fills"] += 1

                        h_today = grid.high_mat[c_i, t]
                        l_today = grid.low_mat[c_i, t]
                        c_today = grid.close_mat[c_i, t]

                        # Day-1 Entry Stop Check — decided off the Close, matching the live
                        # bot's EOD-only stop evaluation (Phase 2B above uses the same rule
                        # for positions opened on prior days).
                        if c_today <= sl_price:
                            part_rate_exit = min(1.0, max(0.0, (units * c_today) / adv_30d))
                            slip_exit = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate_exit)) / 10000.0
                            fill_exit_px, proceeds, fee_exit = _sell_fill_audited(units, c_today, slip_exit, apply_dp=True)
                            cash += proceeds
                            pnl = proceeds - total_cost

                            fee_buy.brokerage += fee_exit.brokerage
                            fee_buy.stt += fee_exit.stt
                            fee_buy.exchange_charges += fee_exit.exchange_charges
                            fee_buy.sebi_charges += fee_exit.sebi_charges
                            fee_buy.gst += fee_exit.gst
                            fee_buy.dp_charges += fee_exit.dp_charges
                            fee_buy.slippage_cost += fee_exit.slippage_cost

                            closed_trades.append({
                                "coin": item["coin"], "pnl": pnl, "ret": pnl / total_cost if total_cost > 0 else 0.0,
                                "reason": "STOP_LOSS", "bars": 0, "bars_to_peak": 0,
                                "mfe_pct": ((max(fill_px, h_today) - fill_px) / fill_px) * 100.0,
                                "mae_pct": ((l_today - fill_px) / fill_px) * 100.0,
                                "entry_date": curr_date, "exit_date": curr_date, "exit_idx": idx,
                                "entry_price": fill_px, "exit_price": fill_exit_px,
                                "cost_inr": total_cost, "proceeds": proceeds,
                                "frictions": fee_buy, "from_watchlist": (t > item["signal_bar"]),
                            })
                        else:
                            live_tranches.append(Tranche(
                                tid=tid_counter, coin=item["coin"], coin_idx=c_i, entry_bar=t, entry_date=curr_date,
                                entry_price=fill_px, initial_units=units, units=units, cost_inr=total_cost,
                                entry_atr=item["entry_atr"], current_sl=sl_price, stop_reason="STOP_LOSS",
                                highest_high=max(fill_px, h_today), lowest_low=min(fill_px, l_today),
                                peak_bar=t, trough_bar=t, layer=coin_layers + 1,
                                from_watchlist=(t > item["signal_bar"]), wait_days=(t - item["signal_bar"]),
                                proceeds=0.0, fee_acc=fee_buy
                            ))
                    else:
                        if coin_layers >= max_pyramid:
                            item["unfilled_reason"] = "pyramid_blocked"
                        elif cash < total_cost or tranche_inr < TRANCHE_FLOOR_INR:
                            item["unfilled_reason"] = "cash_starved"
                        else:
                            item["unfilled_reason"] = "slot_saturated"
                        unfilled.append(item)
                watchlist = unfilled

            # Terminal disposition accounting for single-day watchlist modes
            if wl_mode == "WL_NONE" and watchlist:
                for item in watchlist:
                    _record_unfilled_disposition(item.get("unfilled_reason"), default_bucket="slot_saturated_dropped")
                watchlist = []

            # 5. Causal Watchlist Lifecycle Update (EOD on Unfilled Items Only)
            if wl_mode != "WL_NONE":
                surviving_watchlist = []
                for item in watchlist:
                    c_i = item["coin_idx"]
                    l_bar, h_bar = grid.low_mat[c_i, t], grid.high_mat[c_i, t]

                    shadow_stopped = (l_bar <= item["shadow_stop"])
                    shadow_exited = (xt in (3, 4, 5, 6, 7)) and exit_mat[c_i, t]
                    shadow_expired = (t - item["signal_bar"]) >= WL_MAX_AGE_BARS

                    if shadow_expired or shadow_stopped or shadow_exited:
                        _record_unfilled_disposition(item.get("unfilled_reason"), default_bucket="expired_unfilled")
                    else:
                        if h_bar > item["highest_high"]:
                            item["highest_high"] = h_bar
                        if trail_mult > 0.0:
                            item["shadow_stop"] = max(item["shadow_stop"], item["highest_high"] - (trail_mult * grid.atr14_mat[c_i, t]))
                        surviving_watchlist.append(item)
                watchlist = surviving_watchlist
            else:
                watchlist = []

            # 6. Horizon Terminal Mark (Final Bar Only)
            if t == end_bar - 1:
                if wl_mode != "WL_NONE" and watchlist:
                    for item in watchlist:
                        _record_unfilled_disposition(item.get("unfilled_reason"), default_bucket="expired_unfilled")
                    watchlist = []

                for tr in list(live_tranches):
                    c_i = tr.coin_idx
                    c_px = grid.close_mat[c_i, t]
                    if np.isfinite(c_px) and c_px > 0:
                        adv = max(grid.dvol30_mat[c_i, t - 1], 1_000_000.0)
                        part_rate = min(1.0, max(0.0, (tr.units * c_px) / adv))
                        slip_mult = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate)) / 10000.0
                        fill_px, proceeds, fee_term = _sell_fill_audited(tr.units, c_px, slip_mult, apply_dp=True)
                        cash += proceeds
                        total_proceeds = tr.proceeds + proceeds
                        pnl = total_proceeds - tr.cost_inr
                        vwap_exit = total_proceeds / tr.initial_units if tr.initial_units > 0 else fill_px

                        tr.fee_acc.brokerage += fee_term.brokerage
                        tr.fee_acc.stt += fee_term.stt
                        tr.fee_acc.exchange_charges += fee_term.exchange_charges
                        tr.fee_acc.sebi_charges += fee_term.sebi_charges
                        tr.fee_acc.gst += fee_term.gst
                        tr.fee_acc.dp_charges += fee_term.dp_charges
                        tr.fee_acc.slippage_cost += fee_term.slippage_cost

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
                        })
                live_tranches = []

            # 7. Daily Time-Weighted Return (TWR) Indexing & Exposure Tracking
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
                equity_net_of_flows = eod_equity - inflow_today
                daily_twr_ret = (equity_net_of_flows - prev_equity) / max(1.0, prev_equity)

            current_twr *= (1.0 + daily_twr_ret)
            prev_equity = eod_equity

            active_capital_curve[idx] = current_active
            total_equity_curve[idx] = eod_equity
            twr_curve[idx] = current_twr
            concurrency_hist[min(len(live_tranches), max_slots)] += 1

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
            "peak_single_stock_pct": float(peak_single_stock_pct * 100.0),
        }


# ==============================================================================
# AUDITED TIME-WEIGHTED RETURN (TWR) METRIC ENGINE
# ==============================================================================
def calculate_benchmarks(grid: MarketGrid, start_bar: int, end_bar: int, macro_ok: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Calculates TWR for Literal ^NSEI, Survivor Basket, and Reference 200 SMA Timed Benchmarks."""
    n_bars = end_bar - start_bar

    # 1. Literal ^NSEI Index TWR
    nsei_p = grid.macro_close[start_bar:end_bar]
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

    return {
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
                      benchmark_cagr: float) -> Dict[str, Any]:
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

    if len(tdf) < 15 or full["profit_factor"] <= 1.00 or cagr_full <= 0.0:
        return {
            "score": -10.0, "full": full,
            "eot_trades": eot_trades, "eot_pnl": eot_pnl,
            "max_dd": float(max_dd), "closed_trade_dd_pct": float(closed_trade_dd_pct),
            "utilization": float(utilization * 100.0),
        }

    if len(tdf) > 5:
        trade_rets = tdf["ret"].values
        p95 = float(np.percentile(trade_rets, 95))
        clipped_rets = np.minimum(trade_rets, p95)
        expectance_discount = max(0.2, float(np.sum(clipped_rets) / max(1e-6, np.sum(trade_rets))))
    else:
        expectance_discount = 1.0

    trimmed_cagr = cagr_full * expectance_discount
    rf_cagr = CASH_ANNUAL_YIELD * 100.0
    cagr_excess_rf = max(0.0, trimmed_cagr - rf_cagr)

    exp_matched_benchmark = rf_cagr + utilization * max(0.0, benchmark_cagr - rf_cagr)
    alpha_excess = trimmed_cagr - exp_matched_benchmark

    if alpha_excess >= 0.0:
        alpha_mult = 1.0 + float(np.log1p(alpha_excess / 8.0))
    else:
        alpha_mult = float(np.exp(alpha_excess / 6.0))

    mar_ratio = cagr_excess_rf / max(4.0, max_dd)
    trade_confidence = min(1.0, len(tdf) / 50.0)
    pf_booster = 1.0 + float(np.log1p(max(0.0, full["profit_factor"] - 1.0)))

    score = np.log1p(max(0.0, mar_ratio)) * trade_confidence * pf_booster * alpha_mult

    return {
        "score": float(np.nan_to_num(score, nan=-10.0)), "full": full,
        "eot_trades": eot_trades, "eot_pnl": eot_pnl,
        "max_dd": float(max_dd), "closed_trade_dd_pct": float(closed_trade_dd_pct),
        "utilization": float(utilization * 100.0),
    }


# ==============================================================================
# ANTI-OVERFITTING: MULTI-ARCH CAUSALITY & NEIGHBORHOOD AUDIT
# ==============================================================================
def dynamic_parameter_neighbors(base_p: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = {
        "entry_ma_len": 5, "rsi_f_len": 2, "rsi_f_smt": 2, "rsi_s_len": 2, "rsi_s_smt": 2,
        "rsi_trend_ma_len": 10, "xover_short_len": 5, "xover_gap": 5, "vol_ma_len": 2,
        "vol_mult": 0.2, "price_lookback": 2, "body_atr_mult": 0.1, "bb_entry_len": 2,
        "bb_entry_std": 0.1, "macro_ma_len": 10, "sl_mult": 0.3, "tp_mult": 0.5,
        "trail_atr_mult": 0.5, "max_holding_bars": 5, "trail_pct": 2.0, "exit_ma_len": 5,
        "exit_rsi_f_len": 2, "exit_rsi_f_smt": 2, "exit_rsi_s_len": 2, "exit_rsi_s_smt": 2,
        "exit_xover_short_len": 5, "exit_xover_gap": 5, "exit_vol_ma_len": 5,
        "exit_vol_mult": 0.2, "bb_exit_len": 2, "max_concurrent_tranches": 1,
        "adx_thresh": 5.0, "be_trigger_atr": 1.0
    }

    neighbors = []
    for k, step in steps.items():
        if k not in base_p:
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
                                     start_bar: int, end_bar: int, benchmark_cagr: float) -> Tuple[bool, float, float, List[float]]:
    neighbors = dynamic_parameter_neighbors(params)
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

    is_stable = (
        pass_rate >= 0.80 and
        median_neighbor_score >= (candidate_score * (1.0 - NEIGHBORHOOD_DROP_LIMIT)) and
        p10_score > 0.0
    )
    plateau_score = min(candidate_score, median_neighbor_score)
    return is_stable, plateau_score, pass_rate, neighbor_scores


def verify_behavioral_causality(grid: MarketGrid):
    logging.info("Executing Complete 5-Archetype Prefix-Invariance Causality Audit (Comparing Row Data)...")
    n_bars = len(grid.dates)
    if n_bars < 500:
        return

    t_cutoff = n_bars - 80
    t_cutoff_date = grid.dates[t_cutoff]

    test_cfgs = [
        {"entry_type": 0, "entry_ma_len": 50, "entry_ma_type": 1, "use_market_macro_system": True,
         "macro_ma_len": 100, "macro_ma_type": 0, "macro_active_exit": False, "adx_thresh": 0.0,
         "max_pyramid_layers": 1, "wl_mode": "WL_NONE", "max_concurrent_tranches": 6,
         "use_global_tp": False, "max_holding_bars": 25, "sl_mult": 3.0, "exit_type": 0, "trail_atr_mult": 0.0},
        {"entry_type": 1, "rsi_f_len": 14, "rsi_f_smt": 6, "rsi_s_len": 40, "rsi_s_smt": 10,
         "use_rsi_trend_filter": False, "use_market_macro_system": True, "macro_ma_len": 100,
         "macro_ma_type": 0, "macro_active_exit": True, "adx_thresh": 20.0,
         "max_pyramid_layers": 1, "wl_mode": "WL_DEEPEST_DISCOUNT", "max_concurrent_tranches": 6,
         "use_global_tp": True, "tp_mult": 4.0, "tp_size_pct": 50.0, "tp_move_sl_be": True,
         "max_holding_bars": 35, "sl_mult": 4.5, "exit_type": 1, "trail_pct": 20.0, "trail_atr_mult": 0.0},
        {"entry_type": 2, "xover_short_len": 15, "xover_short_type": 0, "xover_long_len": 60, "xover_long_type": 0, "xover_gap": 45,
         "use_market_macro_system": True, "macro_ma_len": 100, "macro_ma_type": 0, "macro_active_exit": False, "adx_thresh": 0.0,
         "max_pyramid_layers": 1, "wl_mode": "WL_NONE", "max_concurrent_tranches": 6,
         "use_global_tp": False, "max_holding_bars": 20, "sl_mult": 3.0, "exit_type": 5, "trail_atr_mult": 0.0,
         "exit_xover_short_len": 15, "exit_xover_short_type": 0, "exit_xover_long_len": 60, "exit_xover_long_type": 0, "exit_xover_gap": 45},
        {"entry_type": 3, "vol_ma_len": 20, "vol_mult": 2.5, "price_lookback": 20, "body_atr_mult": 0.8,
         "use_market_macro_system": True, "macro_ma_len": 100, "macro_ma_type": 0, "macro_active_exit": False, "adx_thresh": 0.0,
         "max_pyramid_layers": 1, "wl_mode": "WL_NONE", "max_concurrent_tranches": 6,
         "use_global_tp": False, "max_holding_bars": 20, "sl_mult": 3.0, "exit_type": 6, "trail_atr_mult": 0.0,
         "exit_vol_ma_len": 20, "exit_vol_mult": 1.5},
        {"entry_type": 4, "bb_entry_len": 50, "bb_entry_std": 3.0, "use_market_macro_system": False,
         "adx_thresh": 25.0, "max_pyramid_layers": 2, "wl_mode": "WL_NONE", "max_concurrent_tranches": 11,
         "use_global_tp": True, "tp_mult": 5.0, "tp_size_pct": 50.0, "tp_move_sl_be": False,
         "max_holding_bars": 55, "sl_mult": 9.6, "exit_type": 1, "trail_pct": 29.0, "trail_atr_mult": 0.0,
         "bb_exit_len": 20},
    ]

    for cfg in test_cfgs:
        raw_sig, entry_full, exit_full, macro_full, state_full = compile_signals_fast(grid, cfg)
        engine_full = BacktestEngine(params=cfg)
        res_full = engine_full.run_interval(grid, raw_sig, entry_full, exit_full, macro_full, 300, t_cutoff, state_mat=state_full)
        f_prior = res_full["trades"]
        if len(f_prior) > 0:
            f_prior = f_prior[(f_prior["reason"] != "END_OF_TEST") & (f_prior["exit_date"] < t_cutoff_date)].sort_values(by=["coin", "entry_date"]).reset_index(drop=True)

        res_trunc = engine_full.run_interval(grid, raw_sig[:, :t_cutoff], entry_full[:, :t_cutoff], exit_full[:, :t_cutoff], macro_full[:t_cutoff], 300, t_cutoff, state_mat=state_full[:, :t_cutoff])
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
    p["entry_type"] = trial.suggest_categorical("entry_type", [0, 1, 2, 3, 4])
    p["adx_thresh"] = trial.suggest_categorical("adx_thresh", [0.0, 15.0, 20.0, 25.0, 30.0, 35.0])

    if p["entry_type"] == 0:
        p["entry_ma_len"] = trial.suggest_int("entry_ma_len", 10, 200, step=5)
        p["entry_ma_type"] = trial.suggest_categorical("entry_ma_type", [0, 1, 2, 3, 4])
    elif p["entry_type"] == 1:
        p["rsi_f_len"] = trial.suggest_int("rsi_f_len", 10, 80, step=2)
        p["rsi_f_smt"] = trial.suggest_int("rsi_f_smt", 6, 40, step=2)
        p["rsi_s_len"] = trial.suggest_int("rsi_s_len", 10, 80, step=2)
        p["rsi_s_smt"] = trial.suggest_int("rsi_s_smt", 6, 40, step=2)
        p["use_rsi_trend_filter"] = trial.suggest_categorical("use_rsi_trend_filter", [True, False])
        if p["use_rsi_trend_filter"]:
            p["rsi_trend_ma_len"] = trial.suggest_int("rsi_trend_ma_len", 20, 200, step=10)
            p["rsi_trend_ma_type"] = trial.suggest_categorical("rsi_trend_ma_type", [0, 1, 2, 3, 4])
    elif p["entry_type"] == 2:
        p["xover_short_len"] = trial.suggest_int("xover_short_len", 10, 100, step=5)
        p["xover_short_type"] = trial.suggest_categorical("xover_short_type", [0, 1, 2, 3, 4])
        gap = trial.suggest_int("xover_gap", 10, 80, step=5)
        p["xover_gap"] = gap
        p["xover_long_len"] = min(200, p["xover_short_len"] + gap)
        p["xover_long_type"] = trial.suggest_categorical("xover_long_type", [0, 1, 2, 3, 4])
    elif p["entry_type"] == 3:
        p["vol_ma_len"] = trial.suggest_int("vol_ma_len", 10, 50, step=2)
        p["vol_mult"] = round(trial.suggest_float("vol_mult", 1.5, 4.5, step=0.1), 1)
        p["price_lookback"] = trial.suggest_int("price_lookback", 10, 50, step=2)
        p["body_atr_mult"] = round(trial.suggest_float("body_atr_mult", 0.3, 2.0, step=0.1), 1)
    elif p["entry_type"] == 4:
        p["bb_entry_len"] = trial.suggest_int("bb_entry_len", 10, 50, step=2)
        p["bb_entry_std"] = round(trial.suggest_float("bb_entry_std", 1.5, 3.0, step=0.1), 1)

    if FORCE_MACRO_REGIME_FILTER is not None:
        p["use_market_macro_system"] = FORCE_MACRO_REGIME_FILTER
    else:
        p["use_market_macro_system"] = trial.suggest_categorical("use_market_macro_system", [True, False])

    if p["use_market_macro_system"]:
        p["macro_ma_len"] = trial.suggest_int("macro_ma_len", 50, 300, step=10)
        p["macro_ma_type"] = trial.suggest_categorical("macro_ma_type", [0, 1, 2, 3, 4])
        p["macro_active_exit"] = trial.suggest_categorical("macro_active_exit", [True, False])
    else:
        p["macro_active_exit"] = False

    p["max_concurrent_tranches"] = trial.suggest_int("max_concurrent_tranches", 4, 12, step=1)
    if p["entry_type"] in (0, 3, 4):
        p["max_pyramid_layers"] = trial.suggest_categorical("max_pyramid_layers", [1, 2])
    else:
        p["max_pyramid_layers"] = 1

    p["wl_mode"] = trial.suggest_categorical("wl_mode", [
        "WL_NONE", "WL_DEEPEST_DISCOUNT", "WL_STRONGEST_MOMENTUM"
    ])

    p["use_global_tp"] = trial.suggest_categorical("use_global_tp", [True, False])
    if p["use_global_tp"]:
        p["tp_mult"] = round(trial.suggest_float("tp_mult", 2.0, 7.0, step=0.5), 1)
        p["tp_size_pct"] = round(trial.suggest_float("tp_size_pct", 25.0, 75.0, step=5.0), 1)
        p["tp_move_sl_be"] = trial.suggest_categorical("tp_move_sl_be", [True, False])

    p["be_trigger_atr"] = trial.suggest_categorical("be_trigger_atr", [0.0, 1.5, 2.5, 3.5])
    p["max_holding_bars"] = trial.suggest_int("max_holding_bars", 10, 60, step=5)
    p["sl_mult"] = round(trial.suggest_float("sl_mult", 2.0, 6.0, step=0.2), 1)

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
        p["exit_ma_type"] = trial.suggest_categorical("exit_ma_type", [0, 1, 2, 3, 4])
    elif p["exit_type"] == 4:
        p["exit_rsi_f_len"] = trial.suggest_int("exit_rsi_f_len", 10, 60, step=2)
        p["exit_rsi_f_smt"] = trial.suggest_int("exit_rsi_f_smt", 6, 30, step=2)
        p["exit_rsi_s_len"] = trial.suggest_int("exit_rsi_s_len", 10, 60, step=2)
        p["exit_rsi_s_smt"] = trial.suggest_int("exit_rsi_s_smt", 6, 30, step=2)
    elif p["exit_type"] == 5:
        p["exit_xover_short_len"] = trial.suggest_int("exit_xover_short_len", 10, 80, step=5)
        p["exit_xover_short_type"] = trial.suggest_categorical("exit_xover_short_type", [0, 1, 2, 3, 4])
        exit_gap = trial.suggest_int("exit_xover_gap", 10, 60, step=5)
        p["exit_xover_gap"] = exit_gap
        p["exit_xover_long_len"] = min(150, p["exit_xover_short_len"] + exit_gap)
        p["exit_xover_long_type"] = trial.suggest_categorical("exit_xover_long_type", [0, 1, 2, 3, 4])
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
    str(FORCE_MACRO_REGIME_FILTER) +
    str(FORCE_TRAIL_STOP) +
    str(START_YEAR)
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
    p.setdefault("use_global_tp", False)
    p.setdefault("wl_mode", "WL_NONE")
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


def run_matched_placebo_suite(grid: MarketGrid, p: Dict[str, Any], total_bars: int,
                              start_bar: int, end_bar: int, benchmark_cagr: float,
                              total_raw_triggers: int, n_seeds: int = 50) -> Dict[str, Any]:
    """Runs Unconditioned and Conditioned Placebos and preserves full empirical distributions."""
    n_syms = len(grid.symbols)
    raw_trigger_prob = max(0.001, total_raw_triggers / float(total_bars * n_syms))
    np.random.seed(42)
    unconditioned_cagrs = []
    conditioned_cagrs = []
    adx_t = p.get("adx_thresh", 0.0)

    _, _, exit_mat, macro_ok, state_mat = compile_signals_fast(grid, p)
    engine = BacktestEngine(params=p)

    for _ in range(n_seeds):
        random_raw = (np.random.rand(n_syms, total_bars) < raw_trigger_prob)
        placebo_raw = np.zeros((n_syms, len(grid.dates)), dtype=bool)
        placebo_raw[:, start_bar:end_bar] = random_raw & grid.alive_mat[:, start_bar:end_bar]
        liq_ok = grid.dvol30_mat >= LIQUIDITY_FLOOR_INR

        uncond_entry = placebo_raw & macro_ok & liq_ok & grid.alive_mat
        res_uncond = engine.run_interval(grid, placebo_raw, uncond_entry, exit_mat, macro_ok, start_bar, end_bar, state_mat=state_mat)
        m_uncond = calculate_metrics(res_uncond, grid, start_bar, end_bar, benchmark_cagr)
        unconditioned_cagrs.append(m_uncond["full"]["cagr"])

        adx_ok = (grid.adx14_mat >= adx_t) if adx_t > 0.0 else True
        cond_entry = placebo_raw & macro_ok & adx_ok & liq_ok & grid.alive_mat
        res_cond = engine.run_interval(grid, placebo_raw, cond_entry, exit_mat, macro_ok, start_bar, end_bar, state_mat=state_mat)
        m_cond = calculate_metrics(res_cond, grid, start_bar, end_bar, benchmark_cagr)
        conditioned_cagrs.append(m_cond["full"]["cagr"])

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
    m_oos_base = calculate_metrics(r_oos_base, grid, oos_start, oos_end, oos_bench_cagr)
    m_oos_abl = calculate_metrics(r_oos_abl, grid, oos_start, oos_end, oos_bench_cagr)

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
def run_optimization():
    logging.info(f"Initializing NSE Framework {ENGINE_VERSION} (Production Promotion Engine): Universe = {UNIVERSE_NAME}, Start Year = {START_YEAR}")
    logging.info(f"Parameter Space Signature: {PARAM_SPACE_HASH} | Study: {STUDY_NAME}")
    grid, universe, macro_df, provenance_records = build_market_universe(UNIVERSE_NAME, START_YEAR, FORCE_REFRESH)

    verify_behavioral_causality(grid)

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

    # Dynamic crisis windows: derived directly from STRESS_WINDOWS
    is_crisis_slices = []
    for s_name, (s_start_str, s_end_str) in STRESS_WINDOWS.items():
        s_s_ts, s_e_ts = pd.Timestamp(s_start_str), pd.Timestamp(s_end_str)
        s_idx = np.where((grid.dates >= s_s_ts) & (grid.dates <= s_e_ts))[0]
        if len(s_idx) > 20 and s_idx[0] >= is_start and s_idx[-1] < is_end:
            c_s, c_e = int(s_idx[0]), int(s_idx[-1]) + 1
            bench_slice = calculate_benchmarks(grid, c_s, c_e)
            is_crisis_slices.append({
                "name": s_name, "start_bar": c_s, "end_bar": c_e,
                "nsei_max_dd": bench_slice["nsei_max_dd"]
            })

    baseline_score = -10.0
    if FINAL_WINNER_FILE.exists():
        try:
            with open(FINAL_WINNER_FILE, "r") as f:
                saved = json.load(f)
                if saved.get("version") == ENGINE_VERSION and saved.get("param_hash") == PARAM_SPACE_HASH:
                    baseline_score = saved.get("is_score", -10.0)
                    logging.info(f"Loaded All-Time Champion Hurdle Score from winner.json: {baseline_score:.4f}")
                else:
                    logging.info("Parameter space or version changed. Hurdle reset to baseline.")
        except Exception:
            pass

    def objective(trial: optuna.Trial) -> float:
        params = sample_hyperparameters(trial)
        raw_sig, entry_mat, exit_mat, macro_ok, state_mat = compile_signals_fast(grid, params)
        engine = BacktestEngine(params=params)
        res = engine.run_interval(grid, raw_sig, entry_mat, exit_mat, macro_ok, is_start, is_end, state_mat=state_mat)
        metrics = calculate_metrics(res, grid, is_start, is_end, is_bench["nsei_twr_cagr"])
        raw_score = metrics["score"]

        try:
            current_study_best = trial.study.best_value
        except ValueError:
            current_study_best = -10.0
        target_hurdle = max(current_study_best, baseline_score)

        if raw_score > target_hurdle and raw_score > 0.05:
            is_stable, plateau_score, pass_rate, _ = evaluate_neighborhood_stability(
                params=params, candidate_score=raw_score, grid=grid,
                start_bar=is_start, end_bar=is_end, benchmark_cagr=is_bench["nsei_twr_cagr"]
            )
            
            if not is_stable:
                stability_ratio = max(0.05, pass_rate)
                return max(0.01, float(plateau_score * 0.35 * stability_ratio))

            crisis_penalties = []
            for c_slice in is_crisis_slices:
                c_s_off = c_slice["start_bar"] - is_start
                c_e_off = c_slice["end_bar"] - is_start
                twr_slice = res["twr_curve"][c_s_off:c_e_off]
                peak_c = np.maximum.accumulate(twr_slice)
                dd_c = np.where(peak_c > 0, (peak_c - twr_slice) / peak_c, 0.0)
                c_strat_dd = float(np.max(dd_c)) * 100.0 if len(dd_c) > 0 else 0.0

                dd_diff = c_slice["nsei_max_dd"] - c_strat_dd
                if dd_diff < 0.0:
                    crisis_penalties.append(float(np.exp(dd_diff / 10.0)))
                else:
                    crisis_penalties.append(1.0)

            crisis_mult = float(np.min(crisis_penalties)) if len(crisis_penalties) > 0 else 1.0

            n_syms_grid = len(grid.symbols)
            jackknife_scores = []
            np.random.seed(trial.number)
            for _ in range(3):
                subset_mask = (np.random.rand(n_syms_grid) < 0.70)
                if np.sum(subset_mask) < 25:
                    subset_mask[:25] = True
                res_jk = engine.run_interval(grid, raw_sig, entry_mat, exit_mat, macro_ok, is_start, is_end, active_symbols_mask=subset_mask, state_mat=state_mat)
                m_jk = calculate_metrics(res_jk, grid, is_start, is_end, is_bench["nsei_twr_cagr"])
                jackknife_scores.append(m_jk["score"])

            p10_jk_score = float(np.percentile(jackknife_scores, 10))
            if p10_jk_score <= 0.0:
                return 0.01

            jk_ratio = min(1.0, max(0.1, p10_jk_score / max(1e-6, raw_score)))

            return float(plateau_score * crisis_mult * jk_ratio)
            
        return raw_score

    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=42, n_startup_trials=60)
    study = optuna.create_study(study_name=STUDY_NAME, storage=STUDY_DB, load_if_exists=True, direction="maximize", sampler=sampler)
    
    n_remaining = max(0, N_TRIALS - len(study.trials))
    if n_remaining > 0:
        logging.info(f"Executing {n_remaining} trials in study '{STUDY_NAME}'...")
        study.optimize(objective, n_trials=n_remaining, show_progress_bar=True)
    else:
        logging.info(f"Loaded completed study database with {len(study.trials)} trials.")

    # Multi-Fold Champion Selection Gate
    step_f = total_trading_bars // 3
    f1_s, f1_e = trading_start_bar, trading_start_bar + step_f
    f2_s, f2_e = trading_start_bar + step_f, trading_start_bar + 2 * step_f
    f3_s, f3_e = trading_start_bar + 2 * step_f, n_bars

    top_candidates = sorted(
        [t for t in study.trials if t.value is not None and t.value > 0.05],
        key=lambda t: t.value,
        reverse=True
    )[:15]

    scored_candidates = []
    if not top_candidates:
        fallback_trial = study.best_trial
        p_fb = reconstitute_params(fallback_trial.params)
        scored_candidates.append({
            "trial": fallback_trial,
            "beats_all": False, "beats_recent": False, "recent_margin": 0.0,
            "score": fallback_trial.value or 0.0,
            "c1": 0.0, "c2": 0.0, "c3": 0.0,
            "i1": 0.0, "i2": 0.0, "i3": 0.0,
        })
    else:
        for t_cand in top_candidates:
            p_cand = reconstitute_params(t_cand.params)
            cand_eng = BacktestEngine(params=p_cand)
            raw_c, ent_c, ext_c, mac_c, st_c = compile_signals_fast(grid, p_cand)

            b1 = calculate_benchmarks(grid, f1_s, f1_e, mac_c)
            b2 = calculate_benchmarks(grid, f2_s, f2_e, mac_c)
            b3 = calculate_benchmarks(grid, f3_s, f3_e, mac_c)

            r1 = cand_eng.run_interval(grid, raw_c, ent_c, ext_c, mac_c, f1_s, f1_e, state_mat=st_c)
            r2 = cand_eng.run_interval(grid, raw_c, ent_c, ext_c, mac_c, f2_s, f2_e, state_mat=st_c)
            r3 = cand_eng.run_interval(grid, raw_c, ent_c, ext_c, mac_c, f3_s, f3_e, state_mat=st_c)

            m1 = calculate_metrics(r1, grid, f1_s, f1_e, b1["nsei_twr_cagr"])
            m2 = calculate_metrics(r2, grid, f2_s, f2_e, b2["nsei_twr_cagr"])
            m3 = calculate_metrics(r3, grid, f3_s, f3_e, b3["nsei_twr_cagr"])

            c1, c2, c3 = m1["full"]["cagr"], m2["full"]["cagr"], m3["full"]["cagr"]
            i1, i2, i3 = b1["nsei_twr_cagr"], b2["nsei_twr_cagr"], b3["nsei_twr_cagr"]

            beats_all = bool(c1 > i1 and c2 > i2 and c3 > i3)
            beats_recent = bool(c3 > i3)
            recent_margin = float(c3 - i3)

            scored_candidates.append({
                "trial": t_cand,
                "beats_all": beats_all,
                "beats_recent": beats_recent,
                "recent_margin": recent_margin,
                "score": t_cand.value,
                "c1": c1, "c2": c2, "c3": c3,
                "i1": i1, "i2": i2, "i3": i3,
            })

    # Rank candidates prioritizing all-fold robustness, recent fold performance, and score
    scored_candidates.sort(
        key=lambda x: (x["beats_all"], x["beats_recent"], x["recent_margin"], x["score"]),
        reverse=True
    )
    best_candidate_record = scored_candidates[0]
    best_trial = best_candidate_record["trial"]
    best_params = reconstitute_params(best_trial.params)

    n_beat_all = sum(1 for c in scored_candidates if c["beats_all"])
    n_beat_recent = sum(1 for c in scored_candidates if c["beats_recent"])
    pool_size = len(scored_candidates)

    if n_beat_all == 0:
        if n_beat_recent == 0:
            selection_diagnosis_msg = (
                f"0 of {pool_size} candidates beat the index in all folds and 0 beat Fold 3; "
                f"champion selected by best available Fold 3 margin (Trial #{best_trial.number}, {best_candidate_record['recent_margin']:+.2f}pp vs ^NSEI)."
            )
        else:
            selection_diagnosis_msg = (
                f"0 of {pool_size} candidates beat all folds, but {n_beat_recent} beat recent Fold 3; "
                f"champion selected by Fold 3 outperformance (Trial #{best_trial.number}, {best_candidate_record['recent_margin']:+.2f}pp vs ^NSEI)."
            )
    else:
        selection_diagnosis_msg = (
            f"{n_beat_all} of {pool_size} candidates beat all 3 temporal folds. "
            f"Champion Trial #{best_trial.number} selected on all-fold dominance."
        )

    logging.info(f"Multi-Fold Gate Audit: {selection_diagnosis_msg}")

    logging.info(f"Optimization Complete. Best Verified Score: {best_trial.value:.4f}")

    raw_sig_final, entry_final, exit_final, macro_final, state_final = compile_signals_fast(grid, best_params)
    final_engine = BacktestEngine(params=best_params)

    is_bench = calculate_benchmarks(grid, is_start, is_end, macro_final)
    oos_bench = calculate_benchmarks(grid, oos_start, oos_end, macro_final)

    # 1. RUN IN-SAMPLE
    is_results = final_engine.run_interval(grid, raw_sig_final, entry_final, exit_final, macro_final, is_start, is_end, state_mat=state_final)
    is_metrics = calculate_metrics(is_results, grid, is_start, is_end, is_bench["nsei_twr_cagr"])

    # 2. RUN FRESH OUT-OF-SAMPLE
    oos_results = final_engine.run_interval(grid, raw_sig_final, entry_final, exit_final, macro_final, oos_start, oos_end, state_mat=state_final)
    oos_metrics = calculate_metrics(oos_results, grid, oos_start, oos_end, oos_bench["nsei_twr_cagr"])

    # 3. RUN UNIFIED CONTINUOUS BACKTEST FOR MULTI-CRISIS STRESS AUDIT
    continuous_results = final_engine.run_interval(grid, raw_sig_final, entry_final, exit_final, macro_final, is_start, n_bars, state_mat=state_final)
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

    is_placebo_suite = run_matched_placebo_suite(
        grid, best_params, is_bars, is_start, is_end,
        is_bench["nsei_twr_cagr"], is_results["funnel_stats"]["raw_triggers"], n_seeds=50
    )

    oos_placebo_suite = run_matched_placebo_suite(
        grid, best_params, oos_bars, oos_start, oos_end,
        oos_bench["nsei_twr_cagr"], oos_results["funnel_stats"]["raw_triggers"], n_seeds=50
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
    m_leave5 = calculate_metrics(res_leave5, grid, is_start, is_end, is_bench["nsei_twr_cagr"])
    leave5_cagr = m_leave5["full"]["cagr"]

    # Dual-Window Macro-Active Exit Ablation
    if best_params.get("use_market_macro_system", False):
        ablation_params = copy.deepcopy(best_params)
        ablation_params["macro_active_exit"] = not best_params.get("macro_active_exit", False)
        raw_sig_abl, entry_abl, exit_abl, macro_abl, state_abl = compile_signals_fast(grid, ablation_params)
        ablation_engine = BacktestEngine(params=ablation_params)

        res_abl_is = ablation_engine.run_interval(grid, raw_sig_abl, entry_abl, exit_abl, macro_abl, is_start, is_end, state_mat=state_abl)
        m_abl_is = calculate_metrics(res_abl_is, grid, is_start, is_end, is_bench["nsei_twr_cagr"])

        res_abl_oos = ablation_engine.run_interval(grid, raw_sig_abl, entry_abl, exit_abl, macro_abl, oos_start, oos_end, state_mat=state_abl)
        m_abl_oos = calculate_metrics(res_abl_oos, grid, oos_start, oos_end, oos_bench["nsei_twr_cagr"])

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
        is_bench["nsei_twr_cagr"], oos_bench["nsei_twr_cagr"]
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
        f"| Candidate Rank | Search Score* | Fold 1 ({f1_label}) | ^NSEI F1 | Fold 2 ({f2_label}) | ^NSEI F2 | Fold 3 ({f3_label}) | ^NSEI F3 | Consistency / Status |\n"
        f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    )

    # Row 1: The Selected Champion
    c_champ = best_candidate_record
    status_champ = "All-Era Champion" if c_champ["beats_all"] else ("Recent Outperformer" if c_champ["beats_recent"] else f"Best Available Fold 3 ({c_champ['recent_margin']:+.1f}pp)")
    wf_folds_md += (
        f"| **★ SELECTED CHAMPION (T{c_champ['trial'].number})** | **{c_champ['score']:.4f}** | "
        f"**{c_champ['c1']:.1f}%** | {c_champ['i1']:.1f}% | "
        f"**{c_champ['c2']:.1f}%** | {c_champ['i2']:.1f}% | "
        f"**{c_champ['c3']:.1f}%** | {c_champ['i3']:.1f}% | **{status_champ}** |\n"
    )

    # Subsequent rows: Runners-up
    for rank_idx, cand in enumerate(scored_candidates[1:4], start=2):
        status_cand = "Beat All Folds" if cand["beats_all"] else ("Beat Recent Fold" if cand["beats_recent"] else "Regime Sensitive")
        wf_folds_md += (
            f"| Runner-up #{rank_idx} (T{cand['trial'].number}) | {cand['score']:.4f} | "
            f"{cand['c1']:.1f}% | {cand['i1']:.1f}% | "
            f"{cand['c2']:.1f}% | {cand['i2']:.1f}% | "
            f"{cand['c3']:.1f}% | {cand['i3']:.1f}% | {status_cand} |\n"
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

    winner_data = {
        "version": ENGINE_VERSION,
        "param_hash": PARAM_SPACE_HASH,
        "study_name": STUDY_NAME,
        "universe": UNIVERSE_NAME,
        "is_score": float(is_metrics["score"]),
        "is_metrics": is_metrics,
        "oos_metrics": oos_metrics,
        "is_post_tax": {"pnl": is_tax_pnl, "cagr": is_tax_cagr},
        "oos_post_tax": {"pnl": oos_tax_pnl, "cagr": oos_tax_cagr},
        "is_twr_analytics": is_twr_analytics,
        "oos_twr_analytics": oos_twr_analytics,
        "leave5_cagr": leave5_cagr,
        "top_5_scrips": top_5_scrips,
        "is_placebo_suite": is_placebo_suite,
        "oos_placebo_suite": oos_placebo_suite,
        "is_exposure_telemetry": is_exp_telemetry,
        "oos_exposure_telemetry": oos_exp_telemetry,
        "best_params": best_params,
        "updated_at": pd.Timestamp.now().isoformat(),
    }

    with open(CURRENT_WINNER_FILE, "w", encoding="utf-8") as f:
        json.dump(winner_data, f, indent=4, default=str)

    if is_metrics["score"] > baseline_score:
        logging.info(f"New Champion Discovered: IS Score {is_metrics['score']:.4f} > All-Time Baseline {baseline_score:.4f}")
        with open(FINAL_WINNER_FILE, "w", encoding="utf-8") as f:
            json.dump(winner_data, f, indent=4, default=str)
    else:
        logging.info(f"Run Score {is_metrics['score']:.4f} did not beat Champion {baseline_score:.4f}. Winner file preserved.")

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
| ├── Liquidity Floor Blocked | -{f_disp['liquidity_blocked']} | ADV < {format_price(LIQUIDITY_FLOOR_INR)} |
| ├── ADX Trend Blocked | -{f_disp['adx_blocked']} | ADX < Selected Threshold |
| ├── Pyramid Limit Blocked | -{f_disp['pyramid_blocked']} | Scrip already at max tranches or averaging down |
| ├── Revalidation Dropped | -{f_disp['revalidation_dropped']} | Failed macro/ADX/state check at fill time |
| ├── Slot Saturated Dropped | -{f_disp['slot_saturated_dropped']} | No open portfolio slots available |
| ├── Cash Starved Dropped | -{f_disp['cash_starved_dropped']} | Cash below {format_price(TRANCHE_FLOOR_INR)} |
| ├── Scrip Risk Cap Dropped | -{f_disp['risk_cap_dropped']} | Exceeded single-stock 25% exposure ceiling |
| ├── Expired in Watchlist | -{f_disp['expired_unfilled']} | Exceeded {WL_MAX_AGE_BARS} bars or shadow stop |
| **Executed Trades on Ledger** | **{f_disp['executed_fills']}** | Successfully filled and audited |
| **Checksum Reconciliation** | **{sum([f_disp['macro_blocked'], f_disp['liquidity_blocked'], f_disp['adx_blocked'], f_disp['pyramid_blocked'], f_disp['revalidation_dropped'], f_disp['slot_saturated_dropped'], f_disp['cash_starved_dropped'], f_disp['risk_cap_dropped'], f_disp['expired_unfilled'], f_disp['executed_fills']])}** | Must strictly equal Total Raw Triggers ({f_disp['raw_triggers']}) |
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
        tot_slip = sum(tr["frictions"].slippage_cost for _, tr in tdf.iterrows())
        tot_frictions = tot_stt + tot_exch + tot_sebi + tot_stamp + tot_gst + tot_dp + tot_slip
        return f"""| Statutory / Operational Fee | In-Sample Paid | Drag per Trade |
| :--- | :--- | :--- |
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
    if len(continuous_results["twr_curve"]) > 0:
        years_list = sorted(np.unique(grid.years_arr[is_start:]))
        for yr in years_list:
            yr_indices = np.where(grid.years_arr == yr)[0]
            if len(yr_indices) > 0 and yr_indices[0] >= is_start:
                y_s, y_e = yr_indices[0] - is_start, yr_indices[-1] - is_start + 1
                y_s_grid, y_e_grid = yr_indices[0], yr_indices[-1] + 1
                
                yr_twr = continuous_results["twr_curve"][y_s:y_e]
                strat_yr_ret = ((yr_twr[-1] / max(1e-6, yr_twr[0])) - 1.0) * 100.0

                yr_b = calculate_benchmarks(grid, y_s_grid, y_e_grid, macro_final)
                nsei_yr_ret = yr_b["nsei_total_return"]
                basket_yr_ret = yr_b["basket_total_return"]

                yr_trades = cont_trades[pd.to_datetime(cont_trades["exit_date"]).dt.year == yr]
                yr_wr = (yr_trades["pnl"] > 0).mean() * 100.0 if len(yr_trades) > 0 else 0.0

                yearly_md += f"| **{yr}** | {strat_yr_ret:+.2f}% | {nsei_yr_ret:+.2f}% | {basket_yr_ret:+.2f}% | {len(yr_trades)} | {yr_wr:.1f}% |\n"

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

    concurrency_md = "| Active Concurrent Slots | Total Sessions (IS) | Time Proportion (%) |\n| :--- | :--- | :--- |\n"
    tot_bars_is = sum(is_results["concurrency_hist"])
    for slots, b_cnt in enumerate(is_results["concurrency_hist"]):
        concurrency_md += f"| **{slots} Positions Active** | {b_cnt} sessions | {(b_cnt/max(1, tot_bars_is))*100:.1f}% |\n"

    beats_index_is = bool(is_metrics['full']['cagr'] > is_bench['nsei_twr_cagr'])
    beats_index_oos = bool(oos_metrics['full']['cagr'] > oos_bench['nsei_twr_cagr'])
    beats_placebo_is = bool(is_metrics['full']['cagr'] > is_placebo_suite['conditioned'][0])
    beats_placebo_oos = bool(oos_metrics['full']['cagr'] > oos_placebo_suite['conditioned'][0])

    is_alpha_diff = is_metrics['full']['cagr'] - is_bench['nsei_twr_cagr']
    oos_alpha_diff = oos_metrics['full']['cagr'] - oos_bench['nsei_twr_cagr']

    report_md = f"""# NSE Quantitative Strategy Audit Dossier ({ENGINE_VERSION}) — {UNIVERSE_NAME}

## 1. Segregated Performance Accounting (Pure Flow-Stripped TWR)

### A. Performance Baseline & Generalization Check
| Performance Metric | In-Sample ({is_bars} sessions) | Out-of-Sample ({oos_bars} sessions, Fresh Seed) | Generalization Ratio (OOS / IS) |
| :--- | :--- | :--- | :--- |
| **Beats Literal Index CAGR?** | **{'PASSED (+' + f"{is_alpha_diff:.2f}%" + ')' if beats_index_is else 'FAILED (' + f"{is_alpha_diff:.2f}%" + ')'}** | **{'PASSED (+' + f"{oos_alpha_diff:.2f}%" + ')' if beats_index_oos else 'FAILED (' + f"{oos_alpha_diff:.2f}%" + ')'}** | Direct Alpha over Buy & Hold ^NSEI |
| **Conditioned Placebo Alpha Rank** | **{is_cond_pct:.1f}th Percentile** | **{oos_cond_pct:.1f}th Percentile** | Empirical rank in 50 ADX-conditioned placebo seeds |
| **Beats Conditioned Placebo Median?** | **{'PASSED' if beats_placebo_is else 'FAILED'}** | **{'PASSED' if beats_placebo_oos else 'FAILED'}** | Beat median of random triggers ({oos_placebo_suite['conditioned'][0]:.2f}%) |
| **Rule-Closed Net Profit (Pre-Tax)** | **{format_price(is_metrics['full']['net_pnl'])}** | **{format_price(oos_metrics['full']['net_pnl'])}** | — |
| **Estimated Post-Tax Net Profit (STCG)** | {format_price(is_tax_pnl)} | {format_price(oos_tax_pnl)} | 15% / 20% Net of FY Loss Set-Off |
| **Strategy Annualized TWR (CAGR)** | **{is_metrics['full']['cagr']:.2f}%** | **{oos_metrics['full']['cagr']:.2f}%** | {oos_metrics['full']['cagr']/max(0.01, is_metrics['full']['cagr']):.2f}x |
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
| **Raw Composite Fitness Score** | {is_metrics['score']:.4f} | {oos_metrics['score']:.4f} | Undiscounted full-horizon Calmar/MAR fitness |

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
> **\* Score Reconciliation:** The *Search Score* in this table is the search-time objective value after plateau neighborhood stability, crisis drawdown penalties, and jackknife sub-sampling. It is lower than the *Raw Composite Fitness Score* in Section 1, which measures undiscounted full-sample execution.

{wf_folds_md}
{wf_folds_md}
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

## 12. Discovered Optimal Parameter Set
```json
{json.dumps(best_params, indent=4)}
"""
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_md)
    logging.info(f"{ENGINE_VERSION} audit dossier written to {REPORT_FILE}")


if __name__ == "__main__":
    run_optimization()