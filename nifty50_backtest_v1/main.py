#!/usr/bin/env python3
"""
NSE EQUITY QUANTITATIVE SWING-TRADING FRAMEWORK (V3.0 - PURE SWING ENGINE)
==========================================================================
Architectural Overhaul:
1. UNIVERSAL TRAILING FLOOR: Active across ALL exit modes. Prevents unhedged drawdowns.
2. SWING-HORIZON CEILING: Mandatory max holding period (15–65 bars / ~3w–3m).
   Eliminates 8-year "zombie trades" (e.g. BAJFINANCE multi-year drift).
3. ZERO EOT EXPLOITATION: Fitness score calculated STRICTLY on organic, rule-closed
   trades (EOT_EXCESS_WEIGHT = 0.0). Organic PnL <= 0 immediately discards trial.
4. DECOUPLED DRAWDOWN REPORTING: Equity curve drawdown calculated independently of
   trade count, resolving the synthetic 100% drawdown bug during short stress tests.
5. FAST WMA CONVOLUTION: Vectorized numpy 1D convolution replacing rolling.apply().
"""

from __future__ import annotations

import os
import io
import sys
import copy
import json
import time
import logging
import urllib.request
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import optuna

try:
    import pandas_market_calendars as mcal
    _HAS_MCAL = True
except ImportError:
    _HAS_MCAL = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ==============================================================================
# USER CONTROL PANEL
# ==============================================================================

UNIVERSE_NAME: str = "NIFTY50"   # One of: NIFTY50, NIFTY100, MIDCAP150, SMALLCAP250

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

START_YEAR: int = 2010
QUOTE_CURRENCY: str = "INR"
INITIAL_CAPITAL: float = 5000.0
MONTHLY_CONTRIBUTION: float = 5000.0
MIN_HISTORY_DAYS: int = 300
WARMUP_BARS: int = 300
N_TRIALS: int = 500
PARALLEL_DOWNLOAD_WORKERS: int = 8
FORCE_REFRESH: bool = False
CACHE_MAX_AGE_HOURS: float = 72.0
WL_MAX_AGE_BARS: int = 21

# PURE SWING CONFIGURATION: Zero credit for unclosed positions
EOT_EXCESS_WEIGHT: float = 0.00

# Large-cap floor relaxed; impact/participation models guard liquidity
UNIVERSE_LIQUIDITY_FLOOR_INR: Dict[str, float] = {
    "NIFTY50":      0.0,
    "NIFTY100":     0.0,
    "MIDCAP150":    5_000_000.0,
    "SMALLCAP250":  1_000_000.0,
}
LIQUIDITY_FLOOR_INR: float = UNIVERSE_LIQUIDITY_FLOOR_INR[UNIVERSE_NAME]

TRANCHE_FLOOR_INR: float = 5_000.0
TRANCHE_CEILING_INR: float = 500_000.0
MAX_CONCURRENT_TRANCHES: int = 15
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

# Statutory Charges (NSE Delivery 2026 Rate Card)
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
STUDY_NAME = f"nse_swing_v3_{UNIVERSE_NAME.lower()}"

STRESS_START = "2020-01-01"
STRESS_END = "2020-06-30"


def format_price(px: float) -> str:
    """Signed INR currency formatter."""
    if not np.isfinite(px):
        return "\u20b90.00"
    sign = "-" if px < 0 else ""
    return f"{sign}\u20b9{abs(px):,.2f}"


# ==============================================================================
# NSE STATUTORY FEE MODEL
# ==============================================================================
def compute_buy_cost(gross_inr: float) -> float:
    if gross_inr <= 0:
        return 0.0
    brokerage = FLAT_BROKERAGE_INR if BROKERAGE_MODE == "FLAT" else 0.0
    stt = gross_inr * STT_RATE
    exch = gross_inr * EXCHANGE_TXN_RATE
    stamp = gross_inr * STAMP_DUTY_RATE
    sebi = gross_inr * SEBI_CHARGE_RATE
    gst = (brokerage + exch + sebi) * GST_RATE
    return gross_inr + brokerage + stt + exch + stamp + sebi + gst


def compute_sell_proceeds(gross_inr: float, apply_dp: bool = True) -> float:
    if gross_inr <= 0:
        return 0.0
    brokerage = FLAT_BROKERAGE_INR if BROKERAGE_MODE == "FLAT" else 0.0
    stt = gross_inr * STT_RATE
    exch = gross_inr * EXCHANGE_TXN_RATE
    sebi = gross_inr * SEBI_CHARGE_RATE
    gst = (brokerage + exch + sebi) * GST_RATE
    dp = (DP_CHARGE_INR + DP_CHARGE_GST) if apply_dp else 0.0
    return max(0.0, gross_inr - brokerage - stt - exch - sebi - gst - dp)


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


def _buy_fill(tranche_inr: float, ref_open_price: float, slip_mult: float) -> Tuple[float, float, float]:
    fill_px = ref_open_price * (1.0 + slip_mult)
    units = tranche_inr / ref_open_price if ref_open_price > 0 else 0.0
    gross_at_slip = units * fill_px
    total_cash_cost = compute_buy_cost(gross_at_slip)
    return fill_px, units, total_cash_cost


def _sell_fill(units: float, ref_price: float, slip_mult: float, apply_dp: bool = True) -> Tuple[float, float]:
    fill_px = ref_price * (1.0 - slip_mult)
    gross = units * fill_px
    net_proceeds = compute_sell_proceeds(gross, apply_dp=apply_dp)
    return fill_px, net_proceeds


# ==============================================================================
# 1. UNIVERSE CONSTITUENTS
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
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    df = None
    for url in urls:
        try:
            logging.info(f"Downloading constituents for {universe_name} from {url}...")
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
            logging.info("Falling back to Wikipedia constituent scrape...")
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
    if sym_col is None:
        raise ValueError(f"Could not find Symbol column in constituent file {path}.")

    return sorted({str(s).strip().upper() for s in df[sym_col].tolist() if str(s).strip()})


# ==============================================================================
# 2. CALENDAR & DATA INGESTION
# ==============================================================================
def build_nse_trading_calendar(start_date, end_date) -> pd.DatetimeIndex:
    if _HAS_MCAL:
        try:
            cal = mcal.get_calendar("NSE")
            schedule = cal.schedule(start_date=start_date, end_date=end_date)
            idx = schedule.index
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_localize(None)
            return pd.DatetimeIndex(idx).normalize()
        except Exception:
            pass
    return pd.bdate_range(start_date, end_date)


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
    raw["volume"] = raw["volume"].fillna(0.0)
    raw["quote_volume"] = raw["quote_volume"].fillna(0.0)
    if len(raw) < MIN_HISTORY_DAYS:
        return None
    return raw.set_index("date")


def fetch_single_stock(symbol: str, start_year: int, refresh: bool) -> Tuple[str, Optional[pd.DataFrame], str]:
    safe_name = symbol.replace("^", "IDX_")
    cache_file = DATA_DIR / f"{safe_name}_1d.parquet"
    meta_file = DATA_DIR / f"{safe_name}_1d.meta.json"

    if not refresh and cache_file.exists():
        age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600.0
        if age_hours < CACHE_MAX_AGE_HOURS:
            try:
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
                json.dump({"provider": "yfinance", "timestamp": time.time(), "bars": len(df)}, mf)
        except Exception as e:
            logging.warning(f"Cache write failed for {symbol}: {e}")
        return symbol, df, "yfinance"
    return symbol, None, "none"


def build_market_universe(universe_name: str, start_year: int, refresh: bool) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, List[Dict[str, Any]]]:
    symbols = load_universe_constituents(universe_name)
    logging.info(f"Loaded {len(symbols)} constituent symbols for {universe_name}.")

    _, macro_df, macro_prov = fetch_single_stock(MACRO_INDEX_TICKER, start_year, refresh)
    if macro_df is None:
        raise RuntimeError(f"Failed to load macro index data for {MACRO_INDEX_TICKER}.")

    raw_universe: Dict[str, pd.DataFrame] = {}
    provider_map: Dict[str, str] = {}

    logging.info(f"Downloading candles in parallel ({PARALLEL_DOWNLOAD_WORKERS} workers)...")
    with ThreadPoolExecutor(max_workers=PARALLEL_DOWNLOAD_WORKERS) as executor:
        future_map = {executor.submit(fetch_single_stock, sym, start_year, refresh): sym for sym in symbols}
        for future in as_completed(future_map):
            sym, df, provider = future.result()
            if df is not None and len(df) >= MIN_HISTORY_DAYS:
                raw_universe[sym] = df
                provider_map[sym] = provider

    if not raw_universe:
        raise RuntimeError("No constituent data loaded.")

    macro_end = macro_df.index.max()
    active_threshold = macro_end - pd.Timedelta(days=10)
    active_stocks = [s for s, df in raw_universe.items() if df.index.max() >= active_threshold]
    if not active_stocks:
        active_stocks = list(raw_universe.keys())
    common_end = min(raw_universe[s].index.max() for s in active_stocks)

    master_dates = build_nse_trading_calendar(macro_df.index.min(), common_end)
    master_dates = pd.DatetimeIndex(sorted(d for d in master_dates if macro_df.index.min() <= d <= common_end))

    macro_df = macro_df.reindex(master_dates).ffill().dropna(subset=["close"])

    universe: Dict[str, pd.DataFrame] = {}
    provenance_records: List[Dict[str, Any]] = []

    for sym, df in raw_universe.items():
        reindexed = df.loc[:common_end].reindex(master_dates)
        raw_close = reindexed["close"].copy()

        first_idx = df.index.min()
        last_idx = df.index.max()
        active_span_mask = (master_dates >= first_idx) & (master_dates <= min(last_idx, common_end))
        missing_in_span = int((raw_close.isna() & pd.Series(active_span_mask, index=master_dates)).sum())

        is_missing = raw_close.isna()
        trailing_terminal_missing = is_missing[::-1].cumprod()[::-1].astype(bool)
        has_ever_traded = (~is_missing).cumsum() > 0
        is_delisted_permanent = (trailing_terminal_missing & has_ever_traded).values

        reindexed["alive"] = ~is_missing
        reindexed["is_delisted"] = is_delisted_permanent

        reindexed["close"] = reindexed["close"].ffill()
        reindexed["open"] = reindexed["open"].ffill()
        reindexed["high"] = reindexed["high"].ffill()
        reindexed["low"] = reindexed["low"].ffill()
        reindexed["volume"] = reindexed["volume"].fillna(0.0)
        reindexed["quote_volume"] = reindexed["quote_volume"].fillna(0.0)
        universe[sym] = reindexed

        provenance_records.append({
            "symbol": sym,
            "provider": provider_map.get(sym, "unknown"),
            "first_date": str(first_idx.date()),
            "last_date": str(last_idx.date()),
            "total_bars": len(df),
            "missing_days_in_span": missing_in_span,
            "is_permanently_delisted_or_renamed": bool(is_delisted_permanent[-1]) if len(is_delisted_permanent) else False,
        })

    return universe, macro_df, provenance_records


# ==============================================================================
# 3. VECTORIZED INDICATORS (Optimized Vectorized 1D WMA)
# ==============================================================================
class Indicators:
    @staticmethod
    def moving_average(s: pd.Series, length: int, kind: int) -> pd.Series:
        length = max(2, int(length))
        if kind == 0:   return s.rolling(length, min_periods=length).mean()
        elif kind == 1: return s.ewm(span=length, adjust=False).mean()
        elif kind == 2:
            e1 = s.ewm(span=length, adjust=False).mean()
            e2 = e1.ewm(span=length, adjust=False).mean()
            return 2 * e1 - e2
        elif kind == 3:
            w = np.arange(1, length + 1, dtype=float)
            w_norm = w / w.sum()
            s_filled = s.ffill().bfill()
            conv = np.convolve(s_filled.values, w_norm[::-1], mode='full')[:len(s)]
            conv[:length - 1] = np.nan
            return pd.Series(conv, index=s.index)
        elif kind == 4: return s.ewm(alpha=1.0 / length, adjust=False).mean()
        return s.rolling(length, min_periods=length).mean()

    @staticmethod
    def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        cp = c.shift(1)
        tr = pd.concat([h - l, (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / max(2, length), adjust=False).mean()

    @staticmethod
    def rsi_smoothed(s: pd.Series, length: int, smooth: int) -> pd.Series:
        length, smooth = max(2, int(length)), max(1, int(smooth))
        delta = s.diff()
        gain = delta.where(delta > 0, 0.0).ewm(alpha=1.0 / length, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1.0 / length, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.rolling(smooth, min_periods=smooth).mean()

    @staticmethod
    def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
        length = max(2, int(length))
        h, l, c = df["high"], df["low"], df["close"]
        up = h - h.shift(1)
        down = l.shift(1) - l
        p_dm = np.where((up > down) & (up > 0), up, 0.0)
        m_dm = np.where((down > up) & (down > 0), down, 0.0)
        atr_s = Indicators.atr(df, length)
        p_di = 100.0 * pd.Series(p_dm, index=df.index).ewm(alpha=1.0 / length).mean() / (atr_s + 1e-9)
        m_di = 100.0 * pd.Series(m_dm, index=df.index).ewm(alpha=1.0 / length).mean() / (atr_s + 1e-9)
        dx = 100.0 * (p_di - m_di).abs() / (p_di + m_di + 1e-9)
        return dx.ewm(alpha=1.0 / length).mean()

    @staticmethod
    def bollinger_bands(s: pd.Series, length: int, std_mult: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        length = max(2, int(length))
        mid = s.rolling(length, min_periods=length).mean()
        std = s.rolling(length, min_periods=length).std()
        return mid + (std_mult * std), mid, mid - (std_mult * std)


# ==============================================================================
# 4. SIGNAL COMPILATION
# ==============================================================================
def compile_signals(universe: Dict[str, pd.DataFrame], macro_df: pd.DataFrame, p: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], pd.Series]:
    use_macro = p.get("use_market_macro_system", False)
    if use_macro:
        macro_ma = Indicators.moving_average(macro_df["close"], p["macro_ma_len"], p["macro_ma_type"])
        macro_ok = (macro_df["close"] > macro_ma).fillna(False)
    else:
        macro_ok = pd.Series(True, index=macro_df.index)

    signals: Dict[str, Dict[str, Any]] = {}
    for sym, df in universe.items():
        c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
        qv = df["quote_volume"]
        atr14 = Indicators.atr(df, 14)

        dvol = qv.rolling(30, min_periods=30).mean() if (qv > 0).any() else (c * v).rolling(30, min_periods=30).mean()
        dvol = dvol.fillna(1_000_000.0)
        liq_ok = (dvol >= LIQUIDITY_FLOOR_INR).fillna(True)

        adx_ok = pd.Series(True, index=df.index)
        if p.get("adx_thresh", 0.0) > 0.0:
            adx_ok = (Indicators.adx(df, 14) >= p["adx_thresh"]).fillna(False)

        et = p["entry_type"]
        raw_entry = pd.Series(False, index=df.index)
        if et == 0:
            ma = Indicators.moving_average(c, p["entry_ma_len"], p["entry_ma_type"])
            raw_entry = (c > ma).fillna(False)
        elif et == 1:
            rf = Indicators.rsi_smoothed(c, p["rsi_f_len"], p["rsi_f_smt"])
            rs = Indicators.rsi_smoothed(c, p["rsi_s_len"], p["rsi_s_smt"])
            raw_entry = ((rf > rs) & (rf.shift(1) <= rs.shift(1))).fillna(False)
            if p.get("use_rsi_trend_filter", False):
                rma = Indicators.moving_average(c, p["rsi_trend_ma_len"], p["rsi_trend_ma_type"])
                raw_entry = raw_entry & (c > rma).fillna(False)
        elif et == 2:
            s_ma = Indicators.moving_average(c, p["xover_short_len"], p["xover_short_type"])
            l_ma = Indicators.moving_average(c, p["xover_long_len"], p["xover_long_type"])
            raw_entry = ((s_ma > l_ma) & (s_ma.shift(1) <= l_ma.shift(1))).fillna(False)
        elif et == 3:
            vma = v.rolling(int(p["vol_ma_len"]), min_periods=int(p["vol_ma_len"])).mean()
            hhv = h.rolling(int(p["price_lookback"]), min_periods=int(p["price_lookback"])).max().shift(1)
            raw_entry = ((v > (p["vol_mult"] * vma)) & (c > hhv) & ((c - o) >= (p["body_atr_mult"] * atr14))).fillna(False)
        elif et == 4:
            b_up, _, _ = Indicators.bollinger_bands(c, p["bb_entry_len"], p["bb_entry_std"])
            raw_entry = ((c > b_up) & (c.shift(1) <= b_up.shift(1))).fillna(False)

        entry_mask = raw_entry & liq_ok & adx_ok & macro_ok & df["alive"]

        xt = p["exit_type"]
        exit_sig = pd.Series(False, index=df.index)
        if xt == 3:
            ma_val = Indicators.moving_average(c, p["exit_ma_len"], p["exit_ma_type"])
            exit_sig = (c < ma_val).fillna(False)
        elif xt == 4:
            rf = Indicators.rsi_smoothed(c, p["exit_rsi_f_len"], p["exit_rsi_f_smt"])
            rs = Indicators.rsi_smoothed(c, p["exit_rsi_s_len"], p["exit_rsi_s_smt"])
            exit_sig = (rf < rs).fillna(False)
        elif xt == 5:
            s_ma = Indicators.moving_average(c, p["exit_xover_short_len"], p["exit_xover_short_type"])
            l_ma = Indicators.moving_average(c, p["exit_xover_long_len"], p["exit_xover_long_type"])
            exit_sig = (s_ma < l_ma).fillna(False)
        elif xt == 6:
            vma = v.rolling(int(p["exit_vol_ma_len"]), min_periods=int(p["exit_vol_ma_len"])).mean()
            exit_sig = ((v > (p["exit_vol_mult"] * vma)) & (c < o)).fillna(False)
        elif xt == 7:
            _, b_mid, _ = Indicators.bollinger_bands(c, p["bb_exit_len"], 2.0)
            exit_sig = (c < b_mid).fillna(False)

        signals[sym] = {"entry": entry_mask, "exit_sig": exit_sig, "atr": atr14, "dvol": dvol, "df": df}

    return signals, macro_ok


# ==============================================================================
# 5. EXECUTION & SIMULATION ENGINE
# ==============================================================================
@dataclass
class Tranche:
    tid: int
    coin: str
    entry_bar: int
    entry_date: pd.Timestamp
    entry_price: float
    units: float
    cost_inr: float
    stop_loss: float
    highest_high: float
    layer: int
    tp_done: bool = False
    from_watchlist: bool = False
    wait_days: int = 0
    proceeds: float = 0.0


@dataclass
class Tranche:
    tid: int
    coin: str
    coin_idx: int
    entry_bar: int
    entry_date: pd.Timestamp
    entry_price: float
    units: float
    cost_inr: float
    stop_loss: float
    highest_high: float
    layer: int
    tp_done: bool = False
    from_watchlist: bool = False
    wait_days: int = 0
    proceeds: float = 0.0


class BacktestEngine:
    def __init__(self, params: Dict[str, Any]):
        self.p = params

    def run_interval(self, signals: Dict[str, Dict[str, Any]], macro_df: pd.DataFrame, macro_ok: pd.Series,
                      start_bar: int, end_bar: int) -> Dict[str, Any]:
        p = self.p
        dates = macro_df.index
        use_macro = p.get("use_market_macro_system", False)
        max_holding_bars = p.get("max_holding_bars", 63)

        # ----------------------------------------------------------------------
        # SPEED OPTIMIZATION: Extract contiguous NumPy matrices (Zero .iloc calls)
        # ----------------------------------------------------------------------
        symbols = list(signals.keys())
        n_syms = len(symbols)
        sym_to_idx = {s: i for i, s in enumerate(symbols)}

        open_mat = np.zeros((n_syms, len(dates)), dtype=np.float64)
        high_mat = np.zeros((n_syms, len(dates)), dtype=np.float64)
        low_mat = np.zeros((n_syms, len(dates)), dtype=np.float64)
        close_mat = np.zeros((n_syms, len(dates)), dtype=np.float64)
        atr_mat = np.zeros((n_syms, len(dates)), dtype=np.float64)
        dvol_mat = np.zeros((n_syms, len(dates)), dtype=np.float64)
        delist_mat = np.zeros((n_syms, len(dates)), dtype=bool)
        entry_mat = np.zeros((n_syms, len(dates)), dtype=bool)
        exit_mat = np.zeros((n_syms, len(dates)), dtype=bool)

        for i, sym in enumerate(symbols):
            df = signals[sym]["df"]
            open_mat[i, :] = df["open"].values
            high_mat[i, :] = df["high"].values
            low_mat[i, :] = df["low"].values
            close_mat[i, :] = df["close"].values
            atr_mat[i, :] = signals[sym]["atr"].values
            dvol_mat[i, :] = signals[sym]["dvol"].values
            delist_mat[i, :] = df["is_delisted"].values
            entry_mat[i, :] = signals[sym]["entry"].values
            exit_mat[i, :] = signals[sym]["exit_sig"].values

        macro_ok_arr = macro_ok.values
        months_arr = dates.month.values

        cash = INITIAL_CAPITAL
        total_inflow = INITIAL_CAPITAL
        live_tranches: List[Tranche] = []
        watchlist: List[Dict[str, Any]] = []
        closed_trades: List[Dict[str, Any]] = []

        active_capital_curve = np.zeros(end_bar - start_bar, dtype=np.float64)
        total_equity_curve = np.zeros(end_bar - start_bar, dtype=np.float64)
        tid_counter = 0

        binding_stats = {"tranche_ceiling": 0, "liquidity_cap": 0, "equity_slot": 0, "tranche_floor": 0, "cash_constrained": 0}
        max_pyramid = p.get("max_pyramid_layers", 1)

        trail_mult = p["trail_atr_mult"]
        trail_pct_mult = (1.0 - (p.get("trail_pct", 10.0) / 100.0))
        use_tp = p.get("use_global_tp", False)
        tp_mult = p.get("tp_mult", 5.0)
        tp_size = p.get("tp_size_pct", 50.0) / 100.0
        tp_be = p.get("tp_move_sl_be", False)
        xt = p["exit_type"]

        for idx, t in enumerate(range(start_bar, end_bar)):
            curr_date = dates[t]

            # 0. Monthly Inflow
            if t > start_bar and months_arr[t] != months_arr[t - 1]:
                cash += MONTHLY_CONTRIBUTION
                total_inflow += MONTHLY_CONTRIBUTION

            # 1. Delisting Guard
            surviving_tranches = []
            for tr in live_tranches:
                if delist_mat[tr.coin_idx, t]:
                    pnl = tr.proceeds - tr.cost_inr
                    closed_trades.append({
                        "coin": tr.coin, "pnl": pnl, "ret": pnl / tr.cost_inr if tr.cost_inr > 0 else 0.0,
                        "reason": "DELISTED", "bars": t - tr.entry_bar,
                        "entry_date": tr.entry_date, "exit_date": curr_date,
                        "entry_price": tr.entry_price, "exit_price": 0.0,
                        "cost_inr": tr.cost_inr, "proceeds": tr.proceeds,
                        "from_watchlist": tr.from_watchlist,
                    })
                else:
                    surviving_tranches.append(tr)
            live_tranches = surviving_tranches
            watchlist = [item for item in watchlist if not delist_mat[item["coin_idx"], t]]

            # 2. Macro Liquidation
            if use_macro and not macro_ok_arr[t - 1]:
                for tr in live_tranches:
                    c_i = tr.coin_idx
                    px = open_mat[c_i, t]
                    adv = max(dvol_mat[c_i, t - 1], 1_000_000.0)
                    part_rate = min(1.0, max(0.0, (tr.units * px) / adv))
                    slip_mult = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate)) / 10000.0
                    fill_px, proceeds = _sell_fill(tr.units, px, slip_mult, apply_dp=True)
                    cash += proceeds
                    tr.proceeds += proceeds
                    pnl = tr.proceeds - tr.cost_inr
                    closed_trades.append({
                        "coin": tr.coin, "pnl": pnl, "ret": pnl / tr.cost_inr if tr.cost_inr > 0 else 0.0,
                        "reason": "MACRO_EXIT", "bars": t - tr.entry_bar,
                        "entry_date": tr.entry_date, "exit_date": curr_date,
                        "entry_price": tr.entry_price, "exit_price": fill_px,
                        "cost_inr": tr.cost_inr, "proceeds": tr.proceeds,
                        "from_watchlist": tr.from_watchlist,
                    })
                live_tranches = []
                watchlist = []

            # 3. Position Intrabar Management
            surviving_tranches = []
            for tr in live_tranches:
                c_i = tr.coin_idx
                adv = max(dvol_mat[c_i, t - 1], 1_000_000.0)
                o_bar = open_mat[c_i, t]
                h_bar = high_mat[c_i, t]
                l_bar = low_mat[c_i, t]
                atr_bar = atr_mat[c_i, t - 1]

                if h_bar > tr.highest_high:
                    tr.highest_high = h_bar

                if use_tp and not tr.tp_done:
                    tp_price = tr.entry_price + (tp_mult * atr_bar)
                    if h_bar >= tp_price:
                        close_units = tr.units * tp_size
                        part_rate_tp = min(1.0, max(0.0, (close_units * tp_price) / adv))
                        slip_tp = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate_tp)) / 10000.0
                        _, credit = _sell_fill(close_units, max(o_bar, tp_price), slip_tp, apply_dp=True)
                        cash += credit
                        tr.proceeds += credit
                        tr.units -= close_units
                        tr.tp_done = True
                        if tp_be:
                            tr.stop_loss = max(tr.stop_loss, tr.entry_price)

                exit_triggered = False
                exit_reason = ""

                # Universal Trailing Floor
                tr.stop_loss = max(tr.stop_loss, tr.highest_high - (trail_mult * atr_bar))

                if l_bar <= tr.stop_loss:
                    exit_triggered = True
                    exit_reason = "STOP_LOSS"
                elif xt == 1 and l_bar <= (tr.highest_high * trail_pct_mult):
                    exit_triggered = True
                    exit_reason = "TRAIL_PCT_STOP"
                elif xt in (3, 4, 5, 6, 7) and exit_mat[c_i, t - 1]:
                    exit_triggered = True
                    exit_reason = f"SIGNAL_EXIT_TYPE_{xt}"
                elif (t - tr.entry_bar) >= max_holding_bars:
                    exit_triggered = True
                    exit_reason = "MAX_HOLDING_TIME"

                if exit_triggered:
                    if exit_reason in ("STOP_LOSS", "TRAIL_PCT_STOP"):
                        fill_px = min(o_bar, tr.stop_loss)
                        gross = tr.units * fill_px
                        proceeds = compute_sell_proceeds(gross, apply_dp=True)
                    else:
                        part_rate_exit = min(1.0, max(0.0, (tr.units * o_bar) / adv))
                        slip_exit = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate_exit)) / 10000.0
                        fill_px, proceeds = _sell_fill(tr.units, o_bar, slip_exit, apply_dp=True)

                    cash += proceeds
                    tr.proceeds += proceeds
                    pnl = tr.proceeds - tr.cost_inr
                    closed_trades.append({
                        "coin": tr.coin, "pnl": pnl, "ret": pnl / tr.cost_inr if tr.cost_inr > 0 else 0.0,
                        "reason": exit_reason, "bars": t - tr.entry_bar,
                        "entry_date": tr.entry_date, "exit_date": curr_date,
                        "entry_price": tr.entry_price, "exit_price": fill_px,
                        "cost_inr": tr.cost_inr, "proceeds": tr.proceeds,
                        "from_watchlist": tr.from_watchlist,
                    })
                else:
                    surviving_tranches.append(tr)
            live_tranches = surviving_tranches

            # 4. Shadow Watchlist
            surviving_watchlist = []
            for item in watchlist:
                c_i = item["coin_idx"]
                l_bar = low_mat[c_i, t]
                h_bar = high_mat[c_i, t]
                atr_bar = atr_mat[c_i, t - 1]

                if h_bar > item["highest_high"]:
                    item["highest_high"] = h_bar

                item["shadow_stop"] = max(item["shadow_stop"], item["highest_high"] - (trail_mult * atr_bar))

                shadow_stopped = l_bar <= item["shadow_stop"]
                shadow_exited = (xt in (3, 4, 5, 6, 7)) and exit_mat[c_i, t - 1]
                shadow_expired = (t - item["signal_bar"]) > WL_MAX_AGE_BARS

                if not (shadow_stopped or shadow_exited or shadow_expired):
                    surviving_watchlist.append(item)
            watchlist = surviving_watchlist

            # 5. Ingest New Signals
            active_counts: Dict[int, int] = {}
            for tr in live_tranches:
                active_counts[tr.coin_idx] = active_counts.get(tr.coin_idx, 0) + 1
            for item in watchlist:
                active_counts[item["coin_idx"]] = active_counts.get(item["coin_idx"], 0) + 1

            for c_i in range(n_syms):
                if entry_mat[c_i, t - 1]:
                    if active_counts.get(c_i, 0) < max_pyramid:
                        o_today = open_mat[c_i, t]
                        a_yesterday = atr_mat[c_i, t - 1]
                        init_shadow_stop = o_today - (p.get("sl_mult", 2.5) * a_yesterday)
                        watchlist.append({
                            "coin": symbols[c_i], "coin_idx": c_i, "signal_bar": t,
                            "trigger_price": close_mat[c_i, t - 1],
                            "shadow_stop": init_shadow_stop, "highest_high": o_today,
                        })
                        active_counts[c_i] = active_counts.get(c_i, 0) + 1

            # 6. Sizing & Orders (CALCULATE ACTIVE CAPITAL ONCE OUTSIDE LOOP)
            if watchlist:
                wl_mode = p.get("wl_mode", "WL_CLOSEST_BREAKOUT")
                if wl_mode == "WL_NONE":
                    watchlist = [item for item in watchlist if item["signal_bar"] == t]
                elif wl_mode == "WL_DEEPEST_DISCOUNT":
                    watchlist.sort(key=lambda x: (x["trigger_price"] - close_mat[x["coin_idx"], t - 1]) / max(1e-6, x["trigger_price"]), reverse=True)
                elif wl_mode == "WL_CLOSEST_BREAKOUT":
                    watchlist.sort(key=lambda x: abs(close_mat[x["coin_idx"], t - 1] - x["trigger_price"]) / max(1e-6, x["trigger_price"]))
                elif wl_mode == "WL_STRONGEST_MOMENTUM":
                    t_look = max(0, t - 14)
                    watchlist.sort(key=lambda x: close_mat[x["coin_idx"], t - 1] / max(1e-6, close_mat[x["coin_idx"], t_look]), reverse=True)
                elif wl_mode == "WL_FCFS":
                    watchlist.sort(key=lambda x: x["signal_bar"])
                elif wl_mode == "WL_LCFS":
                    watchlist.sort(key=lambda x: x["signal_bar"], reverse=True)

                unfilled = []
                # Compute open active capital once:
                open_active_capital = sum(tr.units * open_mat[tr.coin_idx, t] for tr in live_tranches)
                current_equity_for_sizing = cash + open_active_capital
                equity_slot_size = current_equity_for_sizing / float(MAX_CONCURRENT_TRANCHES)

                for item in watchlist:
                    c_i = item["coin_idx"]
                    adv_30d = max(dvol_mat[c_i, t - 1], 1_000_000.0)
                    liquidity_cap_inr = adv_30d * MAX_ADV_PARTICIPATION
                    target_inr = min(equity_slot_size, TRANCHE_CEILING_INR, liquidity_cap_inr)

                    max_affordable = compute_max_affordable_tranche(cash, adv_30d)
                    tranche_inr = min(max_affordable, max(TRANCHE_FLOOR_INR, target_inr))

                    part_rate = min(1.0, max(0.0, tranche_inr / adv_30d))
                    slip_mult = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate)) / 10000.0
                    coin_layers = sum(1 for tr in live_tranches if tr.coin_idx == c_i)

                    today_open = open_mat[c_i, t]
                    fill_px, units, total_cost = _buy_fill(tranche_inr, today_open, slip_mult)

                    if (cash >= total_cost and tranche_inr >= TRANCHE_FLOOR_INR and units > 0 and
                            len(live_tranches) < MAX_CONCURRENT_TRANCHES and coin_layers < max_pyramid):

                        if max_affordable < target_inr:
                            binding_stats["cash_constrained"] += 1
                        elif target_inr == TRANCHE_CEILING_INR:
                            binding_stats["tranche_ceiling"] += 1
                        elif target_inr == liquidity_cap_inr:
                            binding_stats["liquidity_cap"] += 1
                        elif target_inr < TRANCHE_FLOOR_INR:
                            binding_stats["tranche_floor"] += 1
                        else:
                            binding_stats["equity_slot"] += 1

                        sl_price = fill_px - (p.get("sl_mult", 2.5) * atr_mat[c_i, t - 1])
                        cash -= total_cost
                        tid_counter += 1
                        live_tranches.append(Tranche(
                            tid=tid_counter, coin=item["coin"], coin_idx=c_i, entry_bar=t, entry_date=curr_date,
                            entry_price=fill_px, units=units, cost_inr=total_cost, stop_loss=sl_price,
                            highest_high=fill_px, layer=coin_layers + 1,
                            from_watchlist=(t > item["signal_bar"]), wait_days=(t - item["signal_bar"]), proceeds=0.0,
                        ))
                    else:
                        unfilled.append(item)
                watchlist = unfilled if wl_mode != "WL_NONE" else []

            # 7. Horizon Terminal Pass
            if t == end_bar - 1:
                for tr in list(live_tranches):
                    c_i = tr.coin_idx
                    c_px = close_mat[c_i, t]
                    if np.isfinite(c_px) and c_px > 0:
                        adv = max(dvol_mat[c_i, t - 1], 1_000_000.0)
                        part_rate = min(1.0, max(0.0, (tr.units * c_px) / adv))
                        slip_mult = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate)) / 10000.0
                        fill_px, proceeds = _sell_fill(tr.units, c_px, slip_mult, apply_dp=True)
                        cash += proceeds
                        tr.proceeds += proceeds
                        pnl = tr.proceeds - tr.cost_inr
                        closed_trades.append({
                            "coin": tr.coin, "pnl": pnl, "ret": pnl / tr.cost_inr if tr.cost_inr > 0 else 0.0,
                            "reason": "END_OF_TEST", "bars": t - tr.entry_bar,
                            "entry_date": tr.entry_date, "exit_date": curr_date,
                            "entry_price": tr.entry_price, "exit_price": fill_px,
                            "cost_inr": tr.cost_inr, "proceeds": tr.proceeds,
                            "from_watchlist": tr.from_watchlist,
                        })
                live_tranches = []

            # 8. Mark to Market
            current_active = sum(tr.units * close_mat[tr.coin_idx, t] for tr in live_tranches)
            active_capital_curve[idx] = current_active
            total_equity_curve[idx] = cash + current_active

        return {
            "trades": pd.DataFrame(closed_trades),
            "equity": total_equity_curve,
            "active_capital": active_capital_curve,
            "total_inflow": total_inflow,
            "final_cash": cash,
            "dates": dates[start_bar:end_bar],
            "binding_stats": binding_stats,
        }

# ==============================================================================
# 6. METRICS & RIGOROUS ORGANIC FITNESS GATING
# ==============================================================================
def calculate_universe_dca_benchmark(universe: Dict[str, pd.DataFrame], start_bar: int, end_bar: int) -> Dict[str, Any]:
    symbols = list(universe.keys())
    n = len(symbols)
    if n == 0:
        return {"benchmark_roi": 0.0, "benchmark_pnl": 0.0, "total_contributed": INITIAL_CAPITAL, "benchmark_max_dd": 0.0}

    dates = universe[symbols[0]].index
    units = {s: 0.0 for s in symbols}
    total_contributed = INITIAL_CAPITAL
    cash_pool = 0.0
    per_leg_initial = INITIAL_CAPITAL / n

    for s in symbols:
        px = universe[s]["close"].iloc[start_bar]
        if np.isfinite(px) and px > 0:
            units[s] += per_leg_initial / px
        else:
            cash_pool += per_leg_initial

    n_bars = end_bar - start_bar
    equity_curve = []

    for t in range(n_bars):
        curr_bar = start_bar + t
        if t > 0 and dates[curr_bar].month != dates[curr_bar - 1].month:
            per_leg = MONTHLY_CONTRIBUTION / n
            for s in symbols:
                px = universe[s]["close"].iloc[curr_bar]
                if np.isfinite(px) and px > 0:
                    units[s] += per_leg / px
                else:
                    cash_pool += per_leg
            total_contributed += MONTHLY_CONTRIBUTION

        day_val = cash_pool
        for s in symbols:
            u = units[s]
            if u > 0:
                px = universe[s]["close"].iloc[curr_bar]
                if np.isfinite(px) and px > 0:
                    day_val += u * px
        equity_curve.append(day_val)

    eq_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(eq_arr)
    dd_curve = np.where(peak > 0, (peak - eq_arr) / peak, 0.0)
    max_dd = float(np.max(dd_curve)) * 100.0 if len(dd_curve) > 0 else 0.0

    final_val = eq_arr[-1] if len(eq_arr) > 0 else total_contributed
    net_pnl = final_val - total_contributed
    roi = (net_pnl / total_contributed) * 100.0 if total_contributed > 0 else 0.0

    return {
        "benchmark_roi": float(np.nan_to_num(roi, nan=0.0)),
        "benchmark_pnl": float(np.nan_to_num(net_pnl, nan=0.0)),
        "benchmark_max_dd": float(np.nan_to_num(max_dd, nan=0.0)),
        "total_contributed": total_contributed,
    }


def _calc_sub_metrics(tdf: pd.DataFrame, total_inflow: float, avg_active_cap: float) -> Dict[str, Any]:
    if len(tdf) == 0:
        return {"trades": 0, "net_pnl": 0.0, "account_roi": 0.0, "rocar": 0.0, "win_rate": 0.0, "profit_factor": 0.0}
    net_pnl = float(tdf["pnl"].sum())
    account_roi = float((net_pnl / total_inflow) * 100.0)
    rocar = float((net_pnl / avg_active_cap) * 100.0)
    wins = tdf[tdf["pnl"] > 0]["pnl"]
    losses = tdf[tdf["pnl"] <= 0]["pnl"].abs()
    pf = float(wins.sum() / (losses.sum() + 1e-9)) if len(losses) > 0 else 10.0
    win_rate = float((tdf["pnl"] > 0).mean()) * 100.0
    return {"trades": len(tdf), "net_pnl": net_pnl, "account_roi": account_roi, "rocar": rocar, "win_rate": win_rate, "profit_factor": pf}


def calculate_metrics(results: Dict[str, Any], benchmark_roi: float) -> Dict[str, Any]:
    tdf = results["trades"]
    eq = results["equity"]
    ac = results["active_capital"]
    d = results["dates"]

    # Independent Equity Curve Drawdown (fixes synthetic 100% drawdown bug)
    if len(eq) > 0:
        eq_clean = np.nan_to_num(eq, nan=results["total_inflow"])
        peak_eq = np.maximum.accumulate(eq_clean)
        dd_curve = np.where(peak_eq > 0, (peak_eq - eq_clean) / peak_eq, 0.0)
        max_dd = float(np.max(dd_curve)) * 100.0 if len(dd_curve) > 0 else 0.0
    else:
        max_dd = 0.0

    avg_active_cap = float(np.mean(ac)) if len(ac) > 0 and np.mean(ac) > 100.0 else TRANCHE_FLOOR_INR
    total_inflow = results["total_inflow"]

    full = _calc_sub_metrics(tdf, total_inflow, avg_active_cap)
    org_tdf = tdf[tdf["reason"] != "END_OF_TEST"] if len(tdf) > 0 else pd.DataFrame()
    organic = _calc_sub_metrics(org_tdf, total_inflow, avg_active_cap)

    eot_tdf = tdf[tdf["reason"] == "END_OF_TEST"] if len(tdf) > 0 else pd.DataFrame()
    eot_trades = len(eot_tdf)
    eot_pnl = float(eot_tdf["pnl"].sum()) if eot_trades > 0 else 0.0

    avg_total_equity = float(np.mean(eq)) if len(eq) > 0 else total_inflow
    utilization = float(avg_active_cap / (avg_total_equity + 1e-9))
    utilization = float(np.nan_to_num(utilization, nan=0.0))

    # STRICT ORGANIC GATE: Optuna receives zero reward for terminal paper drift
    if len(tdf) < 15 or organic["net_pnl"] <= 0.0 or organic["trades"] < 10:
        return {
            "score": -999.0, "full": full, "organic": organic,
            "eot_trades": eot_trades, "eot_pnl": eot_pnl,
            "max_dd": float(max_dd), "utilization": float(utilization * 100.0),
        }

    total_days = max(1, (d[-1] - d[0]).days)
    years = total_days / 365.25

    # Outlier trimmed Organic ROI
    sorted_org = org_tdf.sort_values(by="pnl", ascending=False)
    trimmed_org_pnl = sorted_org.iloc[2:]["pnl"].sum() if len(sorted_org) > 5 else organic["net_pnl"]
    ann_org_roi = (trimmed_org_pnl / total_inflow * 100.0) / years

    # Fitness driven 100% by closed, organic trading edge
    effective_ann_roi = max(0.0, ann_org_roi)

    clean_bench_roi = float(np.nan_to_num(benchmark_roi, nan=0.0))
    alpha_spread = organic["account_roi"] - clean_bench_roi
    alpha_booster = 1.0 + np.tanh(alpha_spread / 100.0)
    dd_penalty = ((1.0 - min(1.0, max_dd / 100.0)) ** 2)
    trade_confidence = min(1.0, organic["trades"] / 40.0)

    growth_component = effective_ann_roi * (0.5 + 0.5 * min(1.0, utilization * 5.0))
    growth_component = max(0.0, float(np.nan_to_num(growth_component, nan=0.0)))

    score = np.log1p(growth_component) * dd_penalty * trade_confidence * alpha_booster
    score = float(np.nan_to_num(score, nan=-999.0))

    return {
        "score": score, "full": full, "organic": organic,
        "eot_trades": eot_trades, "eot_pnl": eot_pnl,
        "max_dd": float(max_dd), "utilization": float(utilization * 100.0),
    }


# ==============================================================================
# 7. HYPERPARAMETER SAMPLING (Bounded to Swing-Trading Regimes)
# ==============================================================================
def sample_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    p = {}
    p["entry_type"] = trial.suggest_int("entry_type", 0, 4)
    if p["entry_type"] == 0:
        p["entry_ma_len"] = trial.suggest_int("entry_ma_len", 10, 200, step=5)
        p["entry_ma_type"] = trial.suggest_int("entry_ma_type", 0, 4)
    elif p["entry_type"] == 1:
        p["rsi_f_len"] = trial.suggest_int("rsi_f_len", 10, 80, step=2)
        p["rsi_f_smt"] = trial.suggest_int("rsi_f_smt", 6, 40, step=2)
        p["rsi_s_len"] = trial.suggest_int("rsi_s_len", 10, 80, step=2)
        p["rsi_s_smt"] = trial.suggest_int("rsi_s_smt", 6, 40, step=2)
        p["use_rsi_trend_filter"] = trial.suggest_categorical("use_rsi_trend_filter", [True, False])
        if p["use_rsi_trend_filter"]:
            p["rsi_trend_ma_len"] = trial.suggest_int("rsi_trend_ma_len", 20, 200, step=10)
            p["rsi_trend_ma_type"] = trial.suggest_int("rsi_trend_ma_type", 0, 4)
    elif p["entry_type"] == 2:
        p["xover_short_len"] = trial.suggest_int("xover_short_len", 10, 100, step=5)
        p["xover_short_type"] = trial.suggest_int("xover_short_type", 0, 4)
        gap = trial.suggest_int("xover_gap", 10, 80, step=5)
        p["xover_long_len"] = min(200, p["xover_short_len"] + gap)
        p["xover_long_type"] = trial.suggest_int("xover_long_type", 0, 4)
    elif p["entry_type"] == 3:
        p["vol_ma_len"] = trial.suggest_int("vol_ma_len", 10, 50, step=2)
        p["vol_mult"] = round(trial.suggest_float("vol_mult", 1.5, 4.0, step=0.1), 1)
        p["price_lookback"] = trial.suggest_int("price_lookback", 10, 40, step=2)
        p["body_atr_mult"] = round(trial.suggest_float("body_atr_mult", 0.5, 2.0, step=0.1), 1)
    elif p["entry_type"] == 4:
        p["bb_entry_len"] = trial.suggest_int("bb_entry_len", 10, 50, step=2)
        p["bb_entry_std"] = round(trial.suggest_float("bb_entry_std", 1.5, 3.0, step=0.1), 1)

    p["use_market_macro_system"] = trial.suggest_categorical("use_market_macro_system", [True, False])
    if p["use_market_macro_system"]:
        p["macro_ma_len"] = trial.suggest_int("macro_ma_len", 50, 250, step=10)
        p["macro_ma_type"] = trial.suggest_int("macro_ma_type", 0, 4)

    p["adx_thresh"] = trial.suggest_categorical("adx_thresh", [0.0, 15.0, 20.0])
    p["max_pyramid_layers"] = trial.suggest_int("max_pyramid_layers", 1, 3)
    p["wl_mode"] = trial.suggest_categorical("wl_mode", [
        "WL_DEEPEST_DISCOUNT", "WL_CLOSEST_BREAKOUT", "WL_STRONGEST_MOMENTUM", "WL_FCFS", "WL_LCFS", "WL_NONE"
    ])

    p["use_global_tp"] = trial.suggest_categorical("use_global_tp", [True, False])
    if p["use_global_tp"]:
        p["tp_mult"] = round(trial.suggest_float("tp_mult", 2.0, 15.0, step=0.5), 1)
        p["tp_size_pct"] = round(trial.suggest_float("tp_size_pct", 20.0, 80.0, step=5.0), 1)
        p["tp_move_sl_be"] = trial.suggest_categorical("tp_move_sl_be", [True, False])

    # MANDATORY SWING PARAMETERS
    p["max_holding_bars"] = trial.suggest_int("max_holding_bars", 15, 65, step=5)
    p["trail_atr_mult"] = round(trial.suggest_float("trail_atr_mult", 1.4, 5.0, step=0.2), 1)
    p["sl_mult"] = round(trial.suggest_float("sl_mult", 1.5, 4.5, step=0.1), 1)

    # Tactical Secondary Exit Mode
    p["exit_type"] = trial.suggest_categorical("exit_type", [0, 1, 3, 4, 5, 6, 7])
    if p["exit_type"] == 1:
        p["trail_pct"] = round(trial.suggest_float("trail_pct", 5.0, 20.0, step=0.5), 1)
    elif p["exit_type"] == 3:
        p["exit_ma_len"] = trial.suggest_int("exit_ma_len", 10, 150, step=5)
        p["exit_ma_type"] = trial.suggest_int("exit_ma_type", 0, 4)
    elif p["exit_type"] == 4:
        p["exit_rsi_f_len"] = trial.suggest_int("exit_rsi_f_len", 10, 60, step=2)
        p["exit_rsi_f_smt"] = trial.suggest_int("exit_rsi_f_smt", 6, 30, step=2)
        p["exit_rsi_s_len"] = trial.suggest_int("exit_rsi_s_len", 10, 60, step=2)
        p["exit_rsi_s_smt"] = trial.suggest_int("exit_rsi_s_smt", 6, 30, step=2)
    elif p["exit_type"] == 5:
        p["exit_xover_short_len"] = trial.suggest_int("exit_xover_short_len", 10, 80, step=5)
        p["exit_xover_short_type"] = trial.suggest_int("exit_xover_short_type", 0, 4)
        gap = trial.suggest_int("exit_xover_gap", 10, 60, step=5)
        p["exit_xover_long_len"] = min(150, p["exit_xover_short_len"] + gap)
        p["exit_xover_long_type"] = trial.suggest_int("exit_xover_long_type", 0, 4)
    elif p["exit_type"] == 6:
        p["exit_vol_ma_len"] = trial.suggest_int("exit_vol_ma_len", 10, 40, step=5)
        p["exit_vol_mult"] = round(trial.suggest_float("exit_vol_mult", 1.5, 4.0, step=0.1), 1)
    elif p["exit_type"] == 7:
        p["bb_exit_len"] = trial.suggest_int("bb_exit_len", 10, 60, step=2)

    return p


# ==============================================================================
# 8. DYNAMIC NEIGHBORHOOD AUDIT
# ==============================================================================
def dynamic_parameter_neighbors(base_p: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = {
        "entry_ma_len": 5, "rsi_f_len": 2, "rsi_s_len": 2, "rsi_trend_ma_len": 10,
        "xover_short_len": 5, "xover_gap": 5, "vol_ma_len": 2, "vol_mult": 0.1,
        "price_lookback": 2, "body_atr_mult": 0.1, "bb_entry_len": 2, "bb_entry_std": 0.1,
        "macro_ma_len": 10, "sl_mult": 0.2, "tp_mult": 0.5, "trail_atr_mult": 0.2,
        "max_holding_bars": 5, "trail_pct": 1.0, "exit_ma_len": 5, "exit_rsi_f_len": 2,
        "exit_rsi_s_len": 2, "exit_xover_short_len": 5, "exit_xover_gap": 5,
        "exit_vol_ma_len": 5, "exit_vol_mult": 0.2, "bb_exit_len": 2,
    }
    directions = [
        {"entry": +1, "macro": +1, "exit": -1, "tp": -1},
        {"entry": -1, "macro": -1, "exit": +1, "tp": +1},
        {"entry": +1, "macro": +1, "exit": +1, "tp": +1},
        {"entry": -1, "macro": -1, "exit": -1, "tp": -1},
    ]
    neighbors = []
    for d in directions:
        cand = copy.deepcopy(base_p)
        for k, step in steps.items():
            if k not in cand:
                continue
            cat = "exit" if ("exit" in k or "trail" in k or "sl" in k or "holding" in k) else ("macro" if "macro" in k else ("tp" if "tp" in k else "entry"))
            mult = d[cat]
            if isinstance(cand[k], float):
                cand[k] = round(max(0.2, cand[k] + mult * step), 1)
            elif isinstance(cand[k], int):
                cand[k] = max(2, int(cand[k] + mult * step))
        if "xover_short_len" in cand and "xover_gap" in cand:
            cand["xover_long_len"] = min(200, cand["xover_short_len"] + cand["xover_gap"])
        if "exit_xover_short_len" in cand and "exit_xover_gap" in cand:
            cand["exit_xover_long_len"] = min(200, cand["exit_xover_short_len"] + cand["exit_xover_gap"])
        neighbors.append(cand)
    return neighbors


def evaluate_neighborhood_stability(params: Dict[str, Any], candidate_score: float, universe: Dict[str, pd.DataFrame],
                                     macro_df: pd.DataFrame, start_bar: int, end_bar: int, benchmark_roi: float) -> Tuple[bool, float, List[float]]:
    neighbors = dynamic_parameter_neighbors(params)
    neighbor_scores = []
    for n_p in neighbors:
        n_signals, n_macro_ok = compile_signals(universe, macro_df, n_p)
        engine = BacktestEngine(params=n_p)
        res = engine.run_interval(n_signals, macro_df, n_macro_ok, start_bar, end_bar)
        metrics = calculate_metrics(res, benchmark_roi)
        neighbor_scores.append(metrics["score"])

    mean_neighbor_score = float(np.mean(neighbor_scores))
    min_neighbor_score = float(np.min(neighbor_scores))
    is_stable = (mean_neighbor_score >= (candidate_score * (1.0 - NEIGHBORHOOD_DROP_LIMIT)) and min_neighbor_score > 0.0)
    plateau_score = min(candidate_score, mean_neighbor_score)
    return is_stable, plateau_score, neighbor_scores


# ==============================================================================
# 9. BEHAVIORAL PREFIX-INVARIANCE AUDIT
# ==============================================================================
def verify_behavioral_causality(universe: Dict[str, pd.DataFrame], macro_df: pd.DataFrame):
    logging.info("Executing Behavioral Prefix-Invariance Audit across trading architectures...")
    n_bars = len(macro_df)
    if n_bars < 500:
        return

    t_cutoff = n_bars - 80
    t_cutoff_date = macro_df.index[t_cutoff]
    trunc_dates = macro_df.index[:t_cutoff]
    trunc_universe = {sym: df.loc[trunc_dates].copy() for sym, df in universe.items()}
    trunc_macro = macro_df.loc[trunc_dates].copy()

    test_configs: List[Dict[str, Any]] = [
        {"entry_type": 0, "entry_ma_len": 20, "entry_ma_type": 0, "use_market_macro_system": True,
         "macro_ma_len": 50, "macro_ma_type": 0, "adx_thresh": 0.0, "max_pyramid_layers": 1,
         "wl_mode": "WL_FCFS", "use_global_tp": False, "exit_type": 0, "sl_mult": 2.5, "trail_atr_mult": 3.0, "max_holding_bars": 40},
        {"entry_type": 1, "rsi_f_len": 14, "rsi_f_smt": 6, "rsi_s_len": 40, "rsi_s_smt": 10,
         "use_rsi_trend_filter": False, "use_market_macro_system": False, "adx_thresh": 0.0,
         "max_pyramid_layers": 1, "wl_mode": "WL_CLOSEST_BREAKOUT", "use_global_tp": False,
         "exit_type": 4, "sl_mult": 3.0, "trail_atr_mult": 3.0, "max_holding_bars": 40, "exit_rsi_f_len": 14, "exit_rsi_f_smt": 6, "exit_rsi_s_len": 40, "exit_rsi_s_smt": 10},
    ]

    for winner_path in (CURRENT_WINNER_FILE, FINAL_WINNER_FILE):
        if winner_path.exists():
            try:
                with open(winner_path, "r") as wf:
                    saved_champ = json.load(wf).get("best_params")
                    if saved_champ and saved_champ not in test_configs:
                        test_configs.append(saved_champ)
                        break
            except Exception:
                pass

    for cfg in test_configs:
        sigs_full, macro_ok_full = compile_signals(universe, macro_df, cfg)
        engine_full = BacktestEngine(params=cfg)
        res_full = engine_full.run_interval(sigs_full, macro_df, macro_ok_full, 300, t_cutoff)
        trades_full = res_full["trades"]

        sigs_trunc, macro_ok_trunc = compile_signals(trunc_universe, trunc_macro, cfg)
        engine_trunc = BacktestEngine(params=cfg)
        res_trunc = engine_trunc.run_interval(sigs_trunc, trunc_macro, macro_ok_trunc, 300, t_cutoff)
        trades_trunc = res_trunc["trades"]

        f_prior = trades_full[(trades_full["reason"] != "END_OF_TEST") & (trades_full["exit_date"] < t_cutoff_date)].sort_values(by=["coin", "entry_date"]).reset_index(drop=True)
        t_prior = trades_trunc[(trades_trunc["reason"] != "END_OF_TEST") & (trades_trunc["exit_date"] < t_cutoff_date)].sort_values(by=["coin", "entry_date"]).reset_index(drop=True)

        if len(f_prior) != len(t_prior):
            raise RuntimeError(f"Lookahead leak detected in Arch {cfg.get('entry_type')}: Full={len(f_prior)}, Trunc={len(t_prior)}")

        for i in range(len(f_prior)):
            r_f, r_t = f_prior.iloc[i], t_prior.iloc[i]
            if r_f["coin"] != r_t["coin"] or r_f["entry_date"] != r_t["entry_date"]:
                raise RuntimeError(f"Trade Alignment Drift in Arch {cfg.get('entry_type')}: {r_f['coin']} vs {r_t['coin']}")
            if abs(r_f["entry_price"] - r_t["entry_price"]) > 1e-4:
                raise RuntimeError(f"Entry Price Drift in Arch {cfg.get('entry_type')}")
            if abs(r_f["pnl"] - r_t["pnl"]) > 1e-4:
                raise RuntimeError(f"PnL Drift in Arch {cfg.get('entry_type')}")

    logging.info("Behavioral Causality Audit Passed (Zero Lookahead Confirmed).")


# ==============================================================================
# 10. OPTIMIZATION & REPORT
# ==============================================================================
def run_optimization():
    logging.info(f"Initializing NSE Framework V3.0 (Pure Swing): Universe = {UNIVERSE_NAME}, Start Year = {START_YEAR}")
    universe, macro_df, provenance_records = build_market_universe(UNIVERSE_NAME, START_YEAR, FORCE_REFRESH)

    verify_behavioral_causality(universe, macro_df)

    n_bars = len(macro_df)
    mature_coverage = pd.Series(0, index=macro_df.index)
    for df in universe.values():
        mature = df["alive"].rolling(WARMUP_BARS, min_periods=WARMUP_BARS).sum() == WARMUP_BARS
        mature_coverage = mature_coverage.add(mature.astype(int), fill_value=0)

    mature_pct = mature_coverage / float(len(universe))
    valid_start_indices = np.where(mature_pct >= 0.51)[0]
    if len(valid_start_indices) == 0:
        raw_cov = pd.Series(0, index=macro_df.index)
        for df in universe.values():
            raw_cov = raw_cov.add(df["alive"].astype(int), fill_value=0)
        raw_pct = raw_cov / float(len(universe))
        trading_start_bar = min(int(np.where(raw_pct >= 0.51)[0][0]) + WARMUP_BARS, n_bars - 100)
    else:
        trading_start_bar = int(valid_start_indices[0])

    total_trading_bars = n_bars - trading_start_bar
    split_offset = int(total_trading_bars * 0.75)
    is_start, is_end = trading_start_bar, trading_start_bar + split_offset
    oos_start, oos_end = is_end, n_bars
    is_bars, oos_bars = is_end - is_start, oos_end - oos_start

    logging.info(
        f"Timeline Verified: Total Bars = {n_bars} | "
        f"Trading Starts = {macro_df.index[trading_start_bar].date()} | "
        f"IS Window = {is_bars} bars [{macro_df.index[is_start].date()} -> {macro_df.index[is_end-1].date()}] | "
        f"OOS Window = {oos_bars} bars [{macro_df.index[oos_start].date()} -> {macro_df.index[oos_end-1].date()}]"
    )

    is_bench = calculate_universe_dca_benchmark(universe, is_start, is_end)
    oos_bench = calculate_universe_dca_benchmark(universe, oos_start, oos_end)

    baseline_score = -999.0
    if CURRENT_WINNER_FILE.exists():
        try:
            with open(CURRENT_WINNER_FILE, "r") as f:
                saved = json.load(f)
                baseline_score = saved.get("is_score", -999.0)
        except Exception:
            pass

    def objective(trial: optuna.Trial) -> float:
        params = sample_hyperparameters(trial)
        signals, macro_ok = compile_signals(universe, macro_df, params)
        engine = BacktestEngine(params=params)
        res = engine.run_interval(signals, macro_df, macro_ok, is_start, is_end)
        metrics = calculate_metrics(res, is_bench["benchmark_roi"])
        raw_score = metrics["score"]

        try:
            current_study_best = trial.study.best_value
        except ValueError:
            current_study_best = -999.0
        target_hurdle = max(current_study_best, baseline_score)

        if raw_score > target_hurdle and raw_score > 0.5:
            is_stable, plateau_score, neighbor_scores = evaluate_neighborhood_stability(
                params=params, candidate_score=raw_score, universe=universe, macro_df=macro_df,
                start_bar=is_start, end_bar=is_end, benchmark_roi=is_bench["benchmark_roi"]
            )
            if not is_stable:
                return min(raw_score * 0.4, target_hurdle - 0.05)
            return plateau_score
        return raw_score

    study = optuna.create_study(study_name=STUDY_NAME, storage=STUDY_DB, load_if_exists=True, direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_trial = study.best_trial
    best_params = best_trial.params
    if best_params.get("entry_type") == 2 and "xover_gap" in best_params:
        best_params["xover_long_len"] = min(200, best_params["xover_short_len"] + best_params["xover_gap"])
    if best_params.get("exit_type") == 5 and "exit_xover_gap" in best_params:
        best_params["exit_xover_long_len"] = min(200, best_params["exit_xover_short_len"] + best_params["exit_xover_gap"])

    logging.info(f"Optimization Complete. Best Verified Plateau Score: {study.best_value:.4f}")

    final_signals, final_macro_ok = compile_signals(universe, macro_df, best_params)
    final_engine = BacktestEngine(params=best_params)

    is_results = final_engine.run_interval(final_signals, macro_df, final_macro_ok, is_start, is_end)
    is_metrics = calculate_metrics(is_results, is_bench["benchmark_roi"])

    oos_results = final_engine.run_interval(final_signals, macro_df, final_macro_ok, oos_start, oos_end)
    oos_metrics = calculate_metrics(oos_results, oos_bench["benchmark_roi"])

    # COVID Stress Window
    stress_start_ts = pd.Timestamp(STRESS_START)
    stress_end_ts = pd.Timestamp(STRESS_END)
    stress_indices = np.where((macro_df.index >= stress_start_ts) & (macro_df.index <= stress_end_ts))[0]
    stress_metrics = None
    stress_bench = None

    if len(stress_indices) > 50:
        b_start, b_end = int(stress_indices[0]), int(stress_indices[-1]) + 1
        stress_bench = calculate_universe_dca_benchmark(universe, b_start, b_end)
        stress_res = final_engine.run_interval(final_signals, macro_df, final_macro_ok, b_start, b_end)
        stress_metrics = calculate_metrics(stress_res, stress_bench["benchmark_roi"])

    winner_data = {
        "universe": UNIVERSE_NAME, "is_score": is_metrics["score"], "is_metrics": is_metrics,
        "oos_metrics": oos_metrics, "stress_test": stress_metrics, "best_params": best_params,
        "updated_at": pd.Timestamp.now().isoformat(),
    }
    with open(CURRENT_WINNER_FILE, "w", encoding="utf-8") as f:
        json.dump(winner_data, f, indent=4, default=str)
    with open(FINAL_WINNER_FILE, "w", encoding="utf-8") as f:
        json.dump(winner_data, f, indent=4, default=str)

    all_trades = pd.concat([is_results["trades"], oos_results["trades"]], ignore_index=True)
    all_trades.to_csv(TRADES_CSV_FILE, index=False)

    reason_table_md = ""
    if len(all_trades) > 0:
        reason_summary = all_trades.groupby("reason").agg(
            trades=("pnl", "count"), net_pnl=("pnl", "sum"), win_rate=("pnl", lambda x: (x > 0).mean() * 100.0)
        ).reset_index()
        reason_summary["share"] = (reason_summary["trades"] / len(all_trades)) * 100.0
        reason_table_md = "| Exit Reason | Trades | Share (%) | Net PnL | Win Rate (%) |\n| :--- | :--- | :--- | :--- | :--- |\n"
        for _, r in reason_summary.iterrows():
            reason_table_md += f"| {r['reason']} | {int(r['trades'])} | {r['share']:.1f}% | {format_price(r['net_pnl'])} | {r['win_rate']:.1f}% |\n"

    binding = is_results["binding_stats"]
    total_evals = max(1, sum(binding.values()))
    binding_table_md = f"""| Constraint Mechanism | Times Bound | Share | Operational Status |
| :--- | :--- | :--- | :--- |
| **Cash Constrained** | {binding['cash_constrained']} | {binding['cash_constrained']/total_evals:.1%} | Funded with available balance |
| **Tranche Ceiling ({format_price(TRANCHE_CEILING_INR)})** | {binding['tranche_ceiling']} | {binding['tranche_ceiling']/total_evals:.1%} | Hard cap active |
| **Liquidity Cap ({MAX_ADV_PARTICIPATION:.1%} 30d ADV)** | {binding['liquidity_cap']} | {binding['liquidity_cap']/total_evals:.1%} | Volume bounds |
| **Equity Slot (Capital / {MAX_CONCURRENT_TRANCHES})** | {binding['equity_slot']} | {binding['equity_slot']/total_evals:.1%} | Proportional slot sizing |
| **Tranche Floor ({format_price(TRANCHE_FLOOR_INR)})** | {binding['tranche_floor']} | {binding['tranche_floor']/total_evals:.1%} | Minimum size floor |
"""

    sample_trades_md = "| Symbol | Entry Date | Entry Price | Exit Date | Exit Price | Reason | Net PnL | Return |\n| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    if len(all_trades) > 0:
        top_winners = all_trades.nlargest(3, "pnl")
        worst_losers = all_trades.nsmallest(3, "pnl")
        sample_df = pd.concat([top_winners, worst_losers]).drop_duplicates()
        for _, tr in sample_df.iterrows():
            sample_trades_md += (f"| {tr['coin']} | {str(tr['entry_date'])[:10]} | {format_price(tr['entry_price'])} | "
                                  f"{str(tr['exit_date'])[:10]} | {format_price(tr['exit_price'])} | {tr['reason']} | "
                                  f"{format_price(tr['pnl'])} | {tr['ret']*100:.1f}% |\n")

    prov_df = pd.DataFrame(provenance_records)
    total_gaps = int(prov_df["missing_days_in_span"].sum()) if len(prov_df) else 0
    delisted = prov_df[prov_df["is_permanently_delisted_or_renamed"] == True] if len(prov_df) else prov_df
    delisted_count = len(delisted)

    delisted_table_md = "\n*No terminal delistings/renames detected.*"
    if delisted_count > 0:
        delisted_table_md = ("\n#### Terminal Delisting / Rename Roster\n"
                              "| Symbol | First Active | Last Valid Bar | Total Bars |\n| :--- | :--- | :--- | :--- |\n")
        for _, drow in delisted.iterrows():
            delisted_table_md += f"| {drow['symbol']} | {drow['first_date']} | {drow['last_date']} | {drow['total_bars']} |\n"

    stress_section = ""
    if stress_metrics and stress_bench:
        stress_section = f"""
## 2. Isolated COVID-Crash Stress-Test ({STRESS_START} -> {STRESS_END})
| Performance Metric | Strategy | Universe DCA Benchmark | Alpha Advantage |
| :--- | :--- | :--- | :--- |
| **Organic Net PnL** | **{format_price(stress_metrics['organic']['net_pnl'])}** | **{format_price(stress_bench['benchmark_pnl'])}** | **{format_price(stress_metrics['organic']['net_pnl'] - stress_bench['benchmark_pnl'])}** |
| **Organic Account ROI** | **{stress_metrics['organic']['account_roi']:.2f}%** | **{stress_bench['benchmark_roi']:.2f}%** | **{stress_metrics['organic']['account_roi'] - stress_bench['benchmark_roi']:.2f}%** |
| **Max Portfolio Drawdown** | **{stress_metrics['max_dd']:.2f}%** | **{stress_bench['benchmark_max_dd']:.2f}%** | **{stress_bench['benchmark_max_dd'] - stress_metrics['max_dd']:.2f}% pts** |
| **Completed Trades** | {stress_metrics['organic']['trades']} | N/A | |
"""

    report_md = f"""# NSE Quantitative Strategy Audit Dossier (V3.0 Pure Swing) — {UNIVERSE_NAME}

## 1. Segregated Performance Accounting

### A. Organic Strategy Accounting (Closed Exclusively by Strategy Signals, Stops & Timeouts)
| Performance Metric | In-Sample ({is_bars} bars) | Out-of-Sample ({oos_bars} bars) |
| :--- | :--- | :--- |
| **Rule-Closed Net Profit** | **{format_price(is_metrics['organic']['net_pnl'])}** | **{format_price(oos_metrics['organic']['net_pnl'])}** |
| **Universe DCA Benchmark Profit** | **{format_price(is_bench['benchmark_pnl'])}** | **{format_price(oos_bench['benchmark_pnl'])}** |
| **Rule-Closed Account ROI** | {is_metrics['organic']['account_roi']:.2f}% | {oos_metrics['organic']['account_roi']:.2f}% |
| **Benchmark Account ROI** | {is_bench['benchmark_roi']:.2f}% | {oos_bench['benchmark_roi']:.2f}% |
| **Strategy Max Drawdown** | {is_metrics['max_dd']:.2f}% | {oos_metrics['max_dd']:.2f}% |
| **Benchmark Max Drawdown** | {is_bench['benchmark_max_dd']:.2f}% | {oos_bench['benchmark_max_dd']:.2f}% |
| **Capital Utilization** | {is_metrics['utilization']:.2f}% | {oos_metrics['utilization']:.2f}% |
| **Rule-Closed Profit Factor** | {is_metrics['organic']['profit_factor']:.2f} | {oos_metrics['organic']['profit_factor']:.2f} |
| **Rule-Closed Win Rate** | {is_metrics['organic']['win_rate']:.2f}% | {oos_metrics['organic']['win_rate']:.2f}% |
| **Completed Rule Trades** | {is_metrics['organic']['trades']} | {oos_metrics['organic']['trades']} |
| **Forced Terminal Trades (EOT)** | {is_metrics['eot_trades']} ({format_price(is_metrics['eot_pnl'])}) | {oos_metrics['eot_trades']} ({format_price(oos_metrics['eot_pnl'])}) |

### B. Full Portfolio Accounting (Including Terminal Unclosed Marks)
| Performance Metric | In-Sample | Out-of-Sample |
| :--- | :--- | :--- |
| **Full Strategy Net Profit** | **{format_price(is_metrics['full']['net_pnl'])}** | **{format_price(oos_metrics['full']['net_pnl'])}** |
| **Full Account ROI** | {is_metrics['full']['account_roi']:.2f}% | {oos_metrics['full']['account_roi']:.2f}% |
| **Full Profit Factor** | {is_metrics['full']['profit_factor']:.2f} | {oos_metrics['full']['profit_factor']:.2f} |
| **Total Completed Trades** | {is_metrics['full']['trades']} | {oos_metrics['full']['trades']} |
{stress_section}
## 3. Position-Size Binding Constraint Diagnostics
{binding_table_md}

## 4. Trade-Reason Population Breakdown
{reason_table_md}

## 5. Audit Trade Sampling (Extremes Inspection)
{sample_trades_md}

## 6. Discovered Optimal Parameter Set
```json
{json.dumps(best_params, indent=4)}
"""
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_md)
    logging.info(f"Audit dossier written to {REPORT_FILE}")


if __name__ == "__main__":
    run_optimization()