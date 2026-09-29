#!/usr/bin/env python3
"""
CRYPTO QUANTITATIVE SWING TRADING ENGINE (V10.2 - LIVE PAPER TRADING)
=====================================================================
1:1 Daily Execution & Paper Portfolio Management Engine
Translates the V10.2 Institutional Backtesting Framework to Live Operation:
- Evaluates signals exclusively on closed UTC daily bars (00:00 UTC / 05:30 AM IST).
- Replays missed days chronologically to ensure exits and trailing stops are tracked.
- Generates an actionable Exchange Action Feed for exact order mirroring.
- Supports manual wallet overrides, manual position injection, and watchlist editing.
- Dispatches an executive HTML email report via Gmail SMTP.
"""

from __future__ import annotations

import os
import sys
import json
import time
import copy
import smtplib
import ssl
from pathlib import Path
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Tuple, Any, Optional, Set

import numpy as np
import pandas as pd
import requests

# ==============================================================================
# 1. USER CONTROL PANEL, CREDENTIALS & MANUAL OVERRIDES
# ==============================================================================

# --- EMAIL NOTIFICATION CREDENTIALS (SET VIA ENV VARS OR UNCOMMENT BELOW) ---
# To use Gmail: generate a 16-character App Password via Google Account -> Security -> 2-Step Verification -> App Passwords
GMAIL_USER: str = os.getenv("GMAIL_USER", "your_email@gmail.com")
GMAIL_APP_PASSWORD: str = os.getenv("GMAIL_APP_PASSWORD", "your_app_password_here")
RECIPIENT_EMAIL: str = os.getenv("RECIPIENT_EMAIL", "recipient_email@gmail.com")
SEND_EMAIL_NOTIFICATION: bool = True     # Set False to disable email dispatch

# --- MANUAL WALLET OVERRIDE ---
# Set to None to let the bot manage wallet cash automatically via its ledger.
# Set to a float (e.g., 5000.0) to force-reset available cash.
MANUAL_WALLET_OVERRIDE: Optional[float] = 2000
# Set to True if you want to rerun in the afternoon with new cash or changed settings
FORCE_RERUN_TODAY: bool = True

# --- MANUAL POSITION INJECTIONS & FORCED EXITS ---
# Force-inject external trades to manage dynamically (SL, TP, and Trailing Stops apply):
# Example: [{"coin": "SOL", "units": 12.5, "entry_price": 142.30, "entry_date": "2026-09-24"}]
# --- MANUAL POSITION INJECTIONS & FORCED EXITS ---
# Enter your holdings directly in INR using "entry_price_inr"
MANUAL_POSITIONS_ADD: List[Dict[str, Any]] = [
    {"coin": "ONDO", "units": 43.1,  "entry_price_inr": 43.046, "entry_date": "2026-09-28"},
    {"coin": "AERO", "units": 15.1,  "entry_price_inr": 57.95,  "entry_date": "2026-09-28"},
    {"coin": "LTC",  "units": 0.138, "entry_price_inr": 7107.9, "entry_date": "2026-09-28"},
    {"coin": "H",    "units": 132.0, "entry_price_inr": 7.4618, "entry_date": "2026-09-28"},
]

# Force-exit positions immediately (Paper state exits, proceeds return to cash):
# Example: ["ETH", "AVAX"]
MANUAL_POSITIONS_REMOVE: List[str] = []

# Force-remove specific coins from the candidate watchlist:
# Example: ["DOGE"]
MANUAL_WATCHLIST_REMOVE: List[str] = []

# --- CORE UNIVERSE & EXECUTION PARAMETERS ---
USDT_INR_RATE: float = 100.15
TOP_N_COINS: int = 50
MACRO_INDEX_TICKER: str = "BTC"
INITIAL_CAPITAL: float = 1_000.0
CRYPTO_EXCHANGE_FEE_RATE: float = 0.0010  # 0.10% taker fee
BASE_SLIPPAGE_BPS: float = 6.0            # 6 bps base bid-ask spread
IMPACT_COEF_BPS: float = 120.0            # Market impact scaling coefficient
LIQUIDITY_FLOOR_USD: float = 2_000_000.0  # 30-day ADV floor
TRANCHE_FLOOR_USD: float = 50          # Minimum order allocation
MAX_POSITION_EQUITY_PCT: float = 0.25     # Max 25% portfolio equity per coin
MAX_ADV_PARTICIPATION: float = 0.015      # Max 1.5% of 30-day ADV
WL_MAX_AGE_BARS: int = 15
MIN_HISTORY_DAYS: int = 250
PARALLEL_DOWNLOAD_WORKERS: int = 8

# --- DIRECTORY & STORAGE PATHS ---
DATA_DIR = Path("data_cache")
OUTPUT_DIR = Path("output") / f"TOP_{TOP_N_COINS}"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = OUTPUT_DIR / "live_paper_state.json"
WINNER_CONFIG_FILE = OUTPUT_DIR / "winner.json"
CURRENT_WINNER_FILE = OUTPUT_DIR / "current_winner.json"

# Fallback parameters if winner.json is not yet generated
DEFAULT_CONFIG: Dict[str, Any] = {
        "entry_type": 3,
        "adx_thresh": 15.0,
        "vol_ma_len": 18,
        "vol_mult": 3.8,
        "price_lookback": 34,
        "body_atr_mult": 1.6,
        "use_market_macro_system": True,
        "macro_ma_len": 110,
        "macro_ma_type": 0,
        "macro_active_exit": False,
        "max_concurrent_tranches": 8,
        "max_pyramid_layers": 2,
        "wl_mode": "WL_STRONGEST_MOMENTUM",
        "use_global_tp": False,
        "be_trigger_atr": 1.5,
        "max_holding_bars": 90,
        "sl_mult": 3.6,
        "exit_type": 7,
        "trail_atr_mult": 5.0,
        "bb_exit_len": 28
    }


# ==============================================================================
# 2. FAST NUMPY INDICATOR KERNELS
# ==============================================================================

def shift_1d(arr: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    res = np.empty_like(arr)
    res[0] = fill_value
    res[1:] = arr[:-1]
    return res

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
        out[i] = out[i - 1] if np.isnan(val) else alpha * val + (1.0 - alpha) * out[i - 1]
    return out

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


# ==============================================================================
# 3. UNIVERSE FILTERING & MULTI-EXCHANGE DATA INGESTION
# ==============================================================================

STABLECOINS = {"USDT", "USDC", "BUSD", "DAI", "FDUSD", "TUSD", "USDD", "USDP", "FRAX", "PYUSD", "USDE", "EURT"}
FIAT_COMMODITY_PEGS = {"XAUT", "PAXG", "EUROC"}
WRAPPED_TOKEN_NAMES = {"WBTC", "WETH", "WMATIC", "WAVAX", "WBNB", "WSOL", "WFTM", "WNEAR", "CBBTC"}
LIQUID_STAKING_PREFIXES = ("ST", "WST", "R", "CB", "ANKR", "MSOL", "SAVAX", "BNSOL", "JITOSOL")
PROTECTED_TICKERS = {"STX", "STORJ", "ROSE", "RENDER", "RUNE", "RVN", "RAD", "REQ", "RLC", "RAY", "RON"}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR", "2L", "2S", "3L", "3S", "4L", "4S", "5L", "5S")

def fetch_live_ltp(coin: str) -> Optional[float]:
    """Gets real-time Last Traded Price (LTP) using mirrors unblocked in India."""
    clean = coin.upper().replace("-USD", "").replace("/USD", "").replace("/USDT", "").strip()
    try:
        url = f"https://data-api.binance.vision/api/v3/ticker/price?symbol={clean}USDT"
        r = _HTTP_SESSION.get(url, timeout=3)
        if r.status_code == 200:
            return float(r.json().get("price", 0.0))
    except Exception:
        pass

    try:
        url = f"https://api.bybit.com/v5/market/tickers?category=spot&symbol={clean}USDT"
        r = _HTTP_SESSION.get(url, timeout=3)
        if r.status_code == 200:
            items = r.json().get("result", {}).get("list", [])
            if items:
                return float(items[0].get("lastPrice", 0.0))
    except Exception:
        pass
    return None

def is_noise_or_wrapper(raw_symbol: str) -> bool:
    sym = raw_symbol.upper().replace("-USD", "").replace("/USD", "").replace("/USDT", "").strip()
    if sym in STABLECOINS or sym in FIAT_COMMODITY_PEGS or sym in WRAPPED_TOKEN_NAMES:
        return True
    if any(sym.endswith(suf) for suf in LEVERAGED_SUFFIXES):
        return True
    if sym.startswith("W") and len(sym) >= 4 and sym[1:] in {"BTC", "ETH", "SOL", "BNB", "AVAX", "ADA", "NEAR"}:
        return True
    if sym not in PROTECTED_TICKERS:
        for pfx in LIQUID_STAKING_PREFIXES:
            if sym.startswith(pfx) and len(sym) > len(pfx):
                base = sym[len(pfx):]
                if base in {"ETH", "SOL", "MATIC", "AVAX", "DOT", "ADA", "BNB", "NEAR"}:
                    return True
    return False

def filter_crypto_universe(symbols: List[str]) -> List[str]:
    seen, ordered = set(), []
    for s in symbols:
        clean = s.upper().replace("-USD", "").strip()
        if not is_noise_or_wrapper(clean) and clean not in seen:
            seen.add(clean)
            ordered.append(clean)
    return ordered

_HTTP_SESSION = requests.Session()
_HTTP_SESSION.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

def _clean_kline_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    # Strictly evaluate only completed daily bars before today's 00:00:00 UTC
    today_utc = pd.Timestamp.now("UTC").tz_localize(None).floor("D")
    df = df[df["date"] < today_utc].dropna(subset=["close"])
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["close"] > 0) & (df["open"] > 0) & (df["high"] >= df["low"])]
    df = df[(df["high"] / np.maximum(1e-8, df["low"])) < 50.0]
    return df.set_index("date")

def fetch_from_binance(symbol: str) -> Optional[pd.DataFrame]:
    start_ts = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp() * 1000)
    now_ts = int(pd.Timestamp.now("UTC").floor("D").timestamp() * 1000)
    for host in ["https://data-api.binance.vision", "https://api.binance.com"]:
        curr_start = start_ts
        all_rows = []
        success = True
        while curr_start < now_ts:
            url = f"{host}/api/v3/klines"
            params = {"symbol": f"{symbol}USDT", "interval": "1d", "startTime": curr_start, "limit": 1000}
            try:
                r = _HTTP_SESSION.get(url, params=params, timeout=6)
                if r.status_code != 200:
                    success = False
                    break
                data = r.json()
                if not data or not isinstance(data, list):
                    break
                all_rows.extend(data)
                if len(data) < 1000:
                    break
                curr_start = data[-1][0] + 86400000
                time.sleep(0.02)
            except Exception:
                success = False
                break
        if success and len(all_rows) >= MIN_HISTORY_DAYS:
            parsed = [[d[0], d[1], d[2], d[3], d[4], d[5], d[7]] for d in all_rows]
            df = pd.DataFrame(parsed, columns=["date", "open", "high", "low", "close", "volume", "quote_volume"])
            df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
            return _clean_kline_dataframe(df)
    return None

def fetch_from_bybit(symbol: str) -> Optional[pd.DataFrame]:
    url = "https://api.bybit.com/v5/market/kline"
    end_ts = int(pd.Timestamp.now("UTC").timestamp() * 1000)
    all_rows = []
    for _ in range(8):
        params = {"category": "spot", "symbol": f"{symbol}USDT", "interval": "D", "end": end_ts, "limit": 1000}
        try:
            r = _HTTP_SESSION.get(url, params=params, timeout=6)
            if r.status_code != 200:
                break
            res = r.json().get("result", {}).get("list", [])
            if not res:
                break
            all_rows.extend(res)
            if len(res) < 1000:
                break
            end_ts = int(res[-1][0]) - 1
            time.sleep(0.02)
        except Exception:
            break
    if len(all_rows) >= MIN_HISTORY_DAYS:
        parsed = [[int(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4]), float(d[5]), float(d[6])] for d in all_rows]
        df = pd.DataFrame(parsed, columns=["date", "open", "high", "low", "close", "volume", "quote_volume"])
        df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
        return _clean_kline_dataframe(df)
    return None

def fetch_from_yahoo(symbol: str) -> Optional[pd.DataFrame]:
    start_ts = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp())
    end_ts = int(pd.Timestamp.now("UTC").timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}-USD?period1={start_ts}&period2={end_ts}&interval=1d"
    try:
        r = _HTTP_SESSION.get(url, timeout=8)
        if r.status_code == 200:
            res = r.json()["chart"]["result"][0]
            timestamps = res["timestamp"]
            q = res["indicators"]["quote"][0]
            df = pd.DataFrame({
                "date": pd.to_datetime(timestamps, unit="s", utc=True),
                "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
                "volume": q["volume"],
                "quote_volume": [c * v if c and v else 0.0 for c, v in zip(q["close"], q["volume"])]
            })
            cleaned = _clean_kline_dataframe(df)
            if len(cleaned) >= MIN_HISTORY_DAYS:
                return cleaned
    except Exception:
        pass
    return None

def fetch_single_crypto(symbol: str, refresh: bool = False) -> Tuple[str, Optional[pd.DataFrame], str]:
    clean_sym = symbol.upper().replace("-USD", "").replace("/USD", "").replace("/USDT", "").strip()
    cache_file = DATA_DIR / f"{clean_sym}_live_1d.parquet"
    if not refresh and cache_file.exists():
        # Allow up to 12 hours local cache, but check if yesterday's closed candle is in cache
        yesterday_utc = pd.Timestamp.now("UTC").tz_localize(None).floor("D") - pd.Timedelta(days=1)
        try:
            df = pd.read_parquet(cache_file)
            if len(df) >= MIN_HISTORY_DAYS and df.index.max() >= yesterday_utc:
                return clean_sym, df, "cache"
        except Exception:
            pass

    for provider_name, fetch_fn in [
        ("binance", lambda: fetch_from_binance(clean_sym)),
        ("bybit", lambda: fetch_from_bybit(clean_sym)),
        ("yahoo", lambda: fetch_from_yahoo(clean_sym)),
    ]:
        try:
            df = fetch_fn()
            if df is not None and len(df) >= MIN_HISTORY_DAYS:
                df.to_parquet(cache_file)
                return clean_sym, df, provider_name
        except Exception:
            continue
    return clean_sym, None, "none"

def discover_top_universe(top_n: int = TOP_N_COINS) -> List[str]:
    cache_path = DATA_DIR / f"top_{top_n}_liquid_coins.json"
    manual_coins = [p["coin"].upper().strip() for p in MANUAL_POSITIONS_ADD if "coin" in p]
    candidates: List[str] = []

    # 1. Try reading from cache first
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
                if len(cached) >= min(15, top_n):
                    candidates = cached
        except Exception:
            pass

    # 2. If no cache, fetch 24h volume from Binance
    if not candidates:
        try:
            r = _HTTP_SESSION.get("https://data-api.binance.vision/api/v3/ticker/24hr", timeout=8)
            if r.status_code == 200:
                pairs = []
                for t in r.json():
                    s = t.get("symbol", "")
                    if s.endswith("USDT"):
                        pairs.append((s[:-4], float(t.get("quoteVolume", 0.0))))
                pairs.sort(key=lambda x: x[1], reverse=True)
                candidates = filter_crypto_universe([p[0] for p in pairs])
        except Exception:
            pass

    # 3. Fallback list if needed
    if len(candidates) < top_n:
        fallback = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOGE", "DOT", "LINK",
                    "NEAR", "LTC", "BCH", "UNI", "APT", "ATOM", "ICP", "FIL", "ETC", "XLM",
                    "RENDER", "HBAR", "AAVE", "INJ", "GRT", "VET", "ALGO", "OP", "ARB", "FTM",
                    "THETA", "SAND", "MANA", "AXS", "FLOW", "EOS", "KAVA", "EGLD", "XTZ", "RUNE"]
        candidates.extend(filter_crypto_universe(fallback))

    final_list = []
    for c in candidates:
        if c not in final_list:
            final_list.append(c)

    # 4. FORCE manual coins to the very front so they are NEVER truncated
    for mc in manual_coins:
        if mc in final_list:
            final_list.remove(mc)
        final_list.insert(0, mc)

    chosen = final_list[:top_n]
    with open(cache_path, "w") as f:
        json.dump(chosen, f)
    return chosen

# ==============================================================================
# 4. STATIC MARKET GRID & SIGNAL MATRIX
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

def build_live_market_grid(top_n: int = TOP_N_COINS) -> Tuple[MarketGrid, Dict[str, pd.DataFrame]]:
    symbols = discover_top_universe(top_n)
    _, macro_df, _ = fetch_single_crypto(MACRO_INDEX_TICKER, refresh=False)
    if macro_df is None:
        raise RuntimeError(f"Unable to load macro benchmark: {MACRO_INDEX_TICKER}")

    raw_universe: Dict[str, pd.DataFrame] = {}
    fetch_symbols = [s for s in symbols if s != MACRO_INDEX_TICKER]

    with ThreadPoolExecutor(max_workers=PARALLEL_DOWNLOAD_WORKERS) as executor:
        future_map = {executor.submit(fetch_single_crypto, sym): sym for sym in fetch_symbols}
        for future in as_completed(future_map):
            sym, df, _ = future.result()
            if df is not None and len(df) >= MIN_HISTORY_DAYS:
                raw_universe[sym] = df

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

    universe: Dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(valid_symbols):
        df = raw_universe[sym].reindex(master_dates)
        raw_close = df["close"].copy()
        is_missing = raw_close.isna()
        trailing_missing_count = int((is_missing[::-1].cumprod()[::-1]).astype(int).sum())
        has_ever_traded = (~is_missing).cumsum() > 0
        is_delisted_perm = (is_missing[::-1].cumprod()[::-1].astype(bool) & has_ever_traded & (trailing_missing_count >= 20)).values

        df["alive"] = ~is_missing
        df["close"] = df["close"].ffill()
        df["open"] = df["open"].ffill()
        df["high"] = df["high"].ffill()
        df["low"] = df["low"].ffill()
        df["volume"] = df["volume"].fillna(0.0)

        h_arr = df["high"].values
        l_arr = df["low"].values
        c_arr = df["close"].values

        atr14 = FastIndicators.atr_1d(h_arr, l_arr, c_arr, 14)
        adx14 = FastIndicators.adx_1d(h_arr, l_arr, c_arr, 14)
        qv = df["quote_volume"].values if "quote_volume" in df else (c_arr * df["volume"].values)
        dvol30 = np_rolling_mean(qv, 30)

        open_mat[i, :] = df["open"].values
        high_mat[i, :] = h_arr
        low_mat[i, :] = l_arr
        close_mat[i, :] = c_arr
        vol_mat[i, :] = df["volume"].values
        atr14_mat[i, :] = np.nan_to_num(atr14, nan=0.0)
        adx14_mat[i, :] = np.nan_to_num(adx14, nan=0.0)
        dvol30_mat[i, :] = np.nan_to_num(dvol30, nan=LIQUIDITY_FLOOR_USD)
        delist_mat[i, :] = is_delisted_perm
        alive_mat[i, :] = (~is_missing).values
        universe[sym] = df

    grid = MarketGrid(
        symbols=valid_symbols, dates=master_dates,
        open_mat=open_mat, high_mat=high_mat, low_mat=low_mat, close_mat=close_mat,
        volume_mat=vol_mat, atr14_mat=atr14_mat, dvol30_mat=dvol30_mat,
        adx14_mat=adx14_mat, delist_mat=delist_mat, alive_mat=alive_mat,
        macro_close=macro_df["close"].values, macro_open=macro_df["open"].values
    )
    return grid, universe

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
            vma = np_rolling_mean(shift_1d(v_arr), int(p["vol_ma_len"]))
            hhv = np_rolling_max(shift_1d(h_arr), int(p["price_lookback"]))
            body = c_arr - o_arr
            raw_entry = (~np.isnan(vma)) & (~np.isnan(hhv)) & (v_arr > (p["vol_mult"] * vma)) & (c_arr > hhv) & (body >= (p["body_atr_mult"] * grid.atr14_mat[i, :]))
            base_ma = FastIndicators.moving_average(c_arr, int(p["price_lookback"]), 0)
            state_entry = (~np.isnan(base_ma)) & (c_arr > base_ma)
        elif et == 4:
            b_up, b_mid, _ = FastIndicators.bollinger_bands(c_arr, p["bb_entry_len"], p["bb_entry_std"])
            raw_entry = (~np.isnan(b_up)) & (c_arr > b_up) & (shift_1d(c_arr) <= shift_1d(b_up))
            state_entry = (~np.isnan(b_mid)) & (c_arr > b_mid)

        raw_signal_mat[i, :] = raw_entry & grid.alive_mat[i, :]
        liq_ok = grid.dvol30_mat[i, :] >= LIQUIDITY_FLOOR_USD
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
            vma = np_rolling_mean(shift_1d(v_arr), int(p["exit_vol_ma_len"]))
            exit_mat[i, :] = (~np.isnan(vma)) & (v_arr > (p["exit_vol_mult"] * vma)) & (c_arr < o_arr)
        elif xt == 7:
            _, b_mid, _ = FastIndicators.bollinger_bands(c_arr, p["bb_exit_len"], 2.0)
            exit_mat[i, :] = (~np.isnan(b_mid)) & (c_arr < b_mid)

    return raw_signal_mat, entry_mat, exit_mat, macro_ok, state_mat


# ==============================================================================
# 5. AUDITED FRICTION & SIZING CALCULATORS
# ==============================================================================

def compute_max_affordable_tranche(cash: float, adv_30d: float) -> float:
    adv = max(adv_30d if np.isfinite(adv_30d) else LIQUIDITY_FLOOR_USD, LIQUIDITY_FLOOR_USD)
    usable_cash = max(0.0, cash)
    tranche_guess = usable_cash / (1.0 + CRYPTO_EXCHANGE_FEE_RATE + BASE_SLIPPAGE_BPS / 10000.0)
    for _ in range(3):
        part_rate = min(1.0, max(0.0, tranche_guess / adv))
        slip_mult = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate)) / 10000.0
        tranche_guess = usable_cash / (1.0 + CRYPTO_EXCHANGE_FEE_RATE + slip_mult)
    return float(np.nan_to_num(tranche_guess * (1.0 - 1e-6), nan=0.0))

def calc_buy_fill(tranche_usd: float, ref_price: float, adv_30d: float) -> Tuple[float, float, float]:
    part_rate = min(1.0, max(0.0, tranche_usd / max(adv_30d, LIQUIDITY_FLOOR_USD)))
    slip_mult = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate)) / 10000.0
    fill_px = ref_price * (1.0 + slip_mult)
    units = tranche_usd / ref_price if ref_price > 0 else 0.0
    gross = units * fill_px
    total_cost = gross * (1.0 + CRYPTO_EXCHANGE_FEE_RATE)
    return fill_px, units, total_cost

def calc_sell_fill(units: float, ref_price: float, adv_30d: float) -> Tuple[float, float]:
    part_rate = min(1.0, max(0.0, (units * ref_price) / max(adv_30d, LIQUIDITY_FLOOR_USD)))
    slip_mult = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate)) / 10000.0
    fill_px = ref_price * (1.0 - slip_mult)
    gross = units * fill_px
    proceeds = max(0.0, gross * (1.0 - CRYPTO_EXCHANGE_FEE_RATE))
    return fill_px, proceeds


# ==============================================================================
# 6. PERSISTENCE ENGINE & DATA STRUCTURES
# ==============================================================================

@dataclass
class Position:
    tid: int
    coin: str
    coin_idx: int
    entry_date: str
    entry_price: float
    initial_units: float
    units: float
    cost_usd: float
    entry_atr: float
    current_sl: float
    stop_reason: str
    highest_high: float
    lowest_low: float
    layer: int
    days_held: int = 0
    tp_done: bool = False
    tp_proceeds: float = 0.0
    from_watchlist: bool = False

@dataclass
class WatchlistItem:
    sig_id: int
    coin: str
    coin_idx: int
    signal_date: str
    trigger_price: float
    shadow_stop: float
    highest_high: float
    breakout_quality: float
    entry_atr: float
    bars_in_watchlist: int = 0

@dataclass
class ActionItem:
    action_type: str        # "BUY", "FULL_EXIT", "PARTIAL_TP", "UPDATE_SL"
    coin: str
    dollar_amount: float
    units: float
    estimated_price: float
    stop_loss: float
    reason: str
    notes: str = ""

def load_or_init_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
                return state
        except Exception as e:
            print(f"[WARN] Failed to read existing state file ({e}). Starting fresh ledger.")
    return {
        "last_processed_date": None,
        "wallet_cash": INITIAL_CAPITAL,
        "positions": {},
        "watchlist": [],
        "closed_trades": [],
        "manual_add_history": []
    }

def save_state(state: Dict[str, Any]):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4, default=str)


# ==============================================================================
# 7. SEQUENTIAL DAILY CATCH-UP & SIGNAL SIMULATION ENGINE
# ==============================================================================

class LiveCryptoExecutionEngine:
    def __init__(self, config: Dict[str, Any]):
        self.p = copy.deepcopy(config)
        self.state = load_or_init_state()
        self.action_feed: List[ActionItem] = []
        self.missed_alerts: List[str] = []

    def run_daily_cycle(self, grid: MarketGrid) -> Tuple[List[ActionItem], Dict[str, Any]]:
        p = self.p
        dates = grid.dates
        n_bars = len(dates)
        if n_bars < 50:
            raise RuntimeError("Insufficient historical bars in MarketGrid.")

        # Resolve last processed bar index
        last_date_str = self.state.get("last_processed_date")
        if last_date_str is None:
            # First run: start from the bar immediately prior to terminal to set up initial state
            start_bar = max(30, n_bars - 2)
        else:
            last_ts = pd.Timestamp(last_date_str).normalize()
            matching_idx = np.where(dates == last_ts)[0]
            if len(matching_idx) == 0:
                print(f"[WARN] Last processed date {last_date_str} not in grid. Defaulting to recent history.")
                start_bar = max(30, n_bars - 5)
            else:
                start_bar = int(matching_idx[0]) + 1

        # 1. Update wallet cash right away if you set an override!
        if MANUAL_WALLET_OVERRIDE is not None:
            self.state["wallet_cash"] = float(MANUAL_WALLET_OVERRIDE)

        # Track whether today's daily close was already recorded earlier
        already_processed_today = (self.state.get("last_processed_date") == str(dates[-1].date()))    

        # 2. Check if already up-to-date
        if start_bar >= n_bars:
            if not FORCE_RERUN_TODAY:
                print(f"[INFO] Engine already current. Latest completed bar was processed.")
                self._evaluate_current_positions_telemetry(grid, n_bars - 1)
                return self.action_feed, self.state
            else:
                print(f"[INFO] Same-day rerun active. Recalculating allocations for bar {n_bars - 1}...")
                start_bar = n_bars - 1  # Force it to re-evaluate today!

        raw_signal_mat, entry_mat, exit_mat, macro_ok, state_mat = compile_signals_fast(grid, p)

        # ----------------------------------------------------------------------
        # Catch-up Loop: Step sequentially across all unprocessed closed daily bars
        # ----------------------------------------------------------------------
        print(f"[INFO] Catch-up sync from bar {start_bar} ({dates[start_bar].date()}) to terminal bar {n_bars - 1} ({dates[-1].date()})...")

        for t in range(start_bar, n_bars):
            curr_date = dates[t]
            curr_date_str = str(curr_date.date())
            is_decision_day = (t == n_bars - 1)

            # A. Process Delisting Protection
            for coin, pos_dict in list(self.state["positions"].items()):
                c_i = pos_dict["coin_idx"]
                if grid.delist_mat[c_i, t]:
                    self.state["wallet_cash"] += pos_dict["tp_proceeds"]
                    self.state["closed_trades"].append({
                        "coin": coin, "entry_date": pos_dict["entry_date"], "exit_date": curr_date_str,
                        "pnl": pos_dict["tp_proceeds"] - pos_dict["cost_usd"], "reason": "DELISTED"
                    })
                    del self.state["positions"][coin]
                    if is_decision_day:
                        self.action_feed.append(ActionItem(
                            action_type="FULL_EXIT", coin=coin, dollar_amount=0.0, units=pos_dict["units"],
                            estimated_price=0.0, stop_loss=0.0, reason="DELISTED",
                            notes="Asset delisted from universe. Close any remaining exchange exposure immediately."
                        ))

            # B. Daily Position Evaluation & Trailing Stop Updates
            max_holding_bars = p.get("max_holding_bars", 25)
            use_tp = p.get("use_global_tp", False)
            tp_mult = p.get("tp_mult", 4.0)
            tp_size_pct = p.get("tp_size_pct", 50.0) / 100.0
            tp_move_sl_be = p.get("tp_move_sl_be", False)
            be_trigger_mult = p.get("be_trigger_atr", 0.0)
            trail_mult = p.get("trail_atr_mult", 0.0)
            trail_pct_mult = (1.0 - (p.get("trail_pct", 10.0) / 100.0))
            xt = p["exit_type"]

            macro_bear_confirmed = False
            if p.get("macro_active_exit", False) and t >= 3:
                macro_bear_confirmed = (not macro_ok[t - 1]) and (not macro_ok[t - 2]) and (not macro_ok[t - 3])

            for coin, pos_dict in list(self.state["positions"].items()):
                c_i = pos_dict["coin_idx"]
                pos = Position(**pos_dict)

                # ==============================================================
                # CRITICAL GUARD: Do NOT evaluate daily exits or trailing stops
                # for positions that entered TODAY! They only trail starting tomorrow.
                # ==============================================================
                if pos.entry_date == curr_date_str:
                    self.state["positions"][coin] = asdict(pos)
                    continue

                pos.days_held += 1

                if not grid.alive_mat[c_i, t]:
                    self.state["positions"][coin] = asdict(pos)
                    continue

                adv_30d = max(grid.dvol30_mat[c_i, t - 1], LIQUIDITY_FLOOR_USD)
                o_bar = grid.open_mat[c_i, t]
                h_bar = grid.high_mat[c_i, t]
                l_bar = grid.low_mat[c_i, t]
                c_bar = grid.close_mat[c_i, t]

                exit_triggered = False
                exit_reason = ""
                raw_exit_px = o_bar

                if macro_bear_confirmed:
                    exit_triggered = True
                    exit_reason = "MACRO_REGIME_EXIT"
                elif pos.days_held >= max_holding_bars:
                    exit_triggered = True
                    exit_reason = "MAX_HOLDING_TIME"
                elif xt in (3, 4, 5, 6, 7) and exit_mat[c_i, t - 1]:
                    if pos.days_held >= 3 or (pos.highest_high - pos.entry_price) >= (1.0 * pos.entry_atr):
                        exit_triggered = True
                        exit_reason = f"SIGNAL_EXIT_TYPE_{xt}"

                if not exit_triggered:
                    sl_breached = (c_bar <= pos.current_sl)
                    tp_price = pos.entry_price + (tp_mult * pos.entry_atr)
                    tp_breached = (use_tp and not pos.tp_done and (h_bar >= tp_price))

                    if sl_breached:
                        exit_triggered = True
                        exit_reason = pos.stop_reason
                        raw_exit_px = c_bar
                    elif tp_breached:
                        # Partial Take-Profit Execution
                        close_units = pos.units * tp_size_pct
                        fill_px, credit = calc_sell_fill(close_units, max(o_bar, tp_price), adv_30d)
                        self.state["wallet_cash"] += credit
                        pos.units -= close_units
                        pos.tp_proceeds += credit
                        pos.tp_done = True

                        if tp_move_sl_be and pos.current_sl < (pos.entry_price * 1.002):
                            pos.current_sl = pos.entry_price * 1.002
                            pos.stop_reason = "BREAKEVEN_SL"

                        if is_decision_day:
                            self.action_feed.append(ActionItem(
                                action_type="PARTIAL_TP", coin=coin, dollar_amount=credit, units=close_units,
                                estimated_price=fill_px, stop_loss=pos.current_sl, reason="GLOBAL_TAKE_PROFIT",
                                notes=f"Take Profit triggered: Sell {tp_size_pct*100:.0f}% of {coin}. Stop loss raised to Breakeven (${pos.current_sl:.4f})."
                            ))
                        else:
                            self.missed_alerts.append(f"[{curr_date_str}] MISSED TAKE PROFIT: {coin} reached TP target ${tp_price:.4f}. Executed 50% partial exit in paper ledger.")

                if exit_triggered:
                    fill_px, proceeds = calc_sell_fill(pos.units, raw_exit_px, adv_30d)
                    self.state["wallet_cash"] += proceeds
                    total_proceeds = pos.tp_proceeds + proceeds
                    pnl = total_proceeds - pos.cost_usd

                    self.state["closed_trades"].append({
                        "coin": coin, "entry_date": pos.entry_date, "exit_date": curr_date_str,
                        "entry_price": pos.entry_price, "exit_price": fill_px, "pnl": pnl,
                        "return_pct": (pnl / pos.cost_usd) * 100.0, "reason": exit_reason
                    })
                    del self.state["positions"][coin]

                    if is_decision_day:
                        self.action_feed.append(ActionItem(
                            action_type="FULL_EXIT", coin=coin, dollar_amount=proceeds, units=pos.units,
                            estimated_price=fill_px, stop_loss=0.0, reason=exit_reason,
                            notes=f"EXIT POSITION: Close 100% of {coin} immediately on exchange at market price."
                        ))
                    else:
                        self.missed_alerts.append(f"[{curr_date_str}] MISSED EXIT: {coin} closed due to {exit_reason} at approx ${fill_px:.4f}. Paper position closed; close immediately on exchange if still held!")
                else:
                    # Trailing & Breakeven Stop Adjustments
                    if h_bar > pos.highest_high:
                        pos.highest_high = h_bar
                    if l_bar < pos.lowest_low:
                        pos.lowest_low = l_bar

                    prev_sl = pos.current_sl
                    if be_trigger_mult > 0.0 and pos.current_sl < (pos.entry_price * 1.002):
                        if pos.highest_high >= (pos.entry_price + be_trigger_mult * pos.entry_atr):
                            pos.current_sl = max(pos.current_sl, pos.entry_price * 1.002)
                            pos.stop_reason = "BREAKEVEN_SL"

                    if xt == 1 and p.get("trail_pct", 0.0) > 0.0:
                        pct_floor = pos.highest_high * trail_pct_mult
                        if pct_floor > pos.current_sl:
                            pos.current_sl = pct_floor
                            pos.stop_reason = "TRAIL_PCT_STOP"
                    elif trail_mult > 0.0:
                        atr_recent = grid.atr14_mat[c_i, t - 1]
                        atr_floor = pos.highest_high - (trail_mult * atr_recent)
                        if atr_floor > pos.current_sl:
                            pos.current_sl = atr_floor
                            pos.stop_reason = "TRAIL_ATR_STOP"

                    if is_decision_day and pos.current_sl > prev_sl:
                        self.action_feed.append(ActionItem(
                            action_type="UPDATE_SL", coin=coin, dollar_amount=0.0, units=pos.units,
                            estimated_price=c_bar, stop_loss=pos.current_sl, reason=pos.stop_reason,
                            notes=f"Update Stop Loss order for {coin} to ${pos.current_sl:.4f} (was ${prev_sl:.4f})."
                        ))

                    self.state["positions"][coin] = asdict(pos)

            # C. Watchlist Lifecycle Evaluation
            wl_mode = p.get("wl_mode", "WL_NONE")
            surviving_watchlist = []
            for w_dict in self.state["watchlist"]:
                w_item = WatchlistItem(**w_dict)
                w_item.bars_in_watchlist += 1
                c_i = w_item.coin_idx

                l_bar = grid.low_mat[c_i, t]
                h_bar = grid.high_mat[c_i, t]

                shadow_stopped = (l_bar <= w_item.shadow_stop)
                shadow_exited = (xt in (3, 4, 5, 6, 7)) and exit_mat[c_i, t]
                shadow_expired = w_item.bars_in_watchlist >= WL_MAX_AGE_BARS

                if not (shadow_stopped or shadow_exited or shadow_expired):
                    if h_bar > w_item.highest_high:
                        w_item.highest_high = h_bar
                    if trail_mult > 0.0:
                        w_item.shadow_stop = max(w_item.shadow_stop, w_item.highest_high - (trail_mult * grid.atr14_mat[c_i, t]))
                    surviving_watchlist.append(asdict(w_item))
            self.state["watchlist"] = surviving_watchlist

            # D. Evaluate New Technical Triggers from Day t - 1
            max_slots = p.get("max_concurrent_tranches", 6)
            max_pyramid = p.get("max_pyramid_layers", 1)
            active_coins = {c: p_d["layer"] for c, p_d in self.state["positions"].items()}

            new_candidates = []
            existing_wl_coins = {item["coin"] for item in self.state["watchlist"]}
            for c_i in range(len(grid.symbols)):
                coin = grid.symbols[c_i]
                if coin in self.state["positions"] or coin in existing_wl_coins:
                    continue  # Skip coins you already hold or that are already queued
                if raw_signal_mat[c_i, t - 1]:
                    if not macro_ok[t - 1]:
                        continue
                    if grid.dvol30_mat[c_i, t - 1] < LIQUIDITY_FLOOR_USD:
                        continue
                    if not entry_mat[c_i, t - 1]:
                        continue
                    if active_coins.get(coin, 0) >= max_pyramid:
                        continue

                    o_today = grid.open_mat[c_i, t]
                    a_yesterday = grid.atr14_mat[c_i, t - 1]
                    init_stop = o_today - (p.get("sl_mult", 3.0) * a_yesterday)
                    breakout_strength = (grid.close_mat[c_i, t - 1] - grid.open_mat[c_i, t - 1]) / max(1e-6, a_yesterday)

                    new_candidates.append(asdict(WatchlistItem(
                        sig_id=int(time.time() * 1000) % 1000000 + c_i,
                        coin=coin, coin_idx=c_i, signal_date=curr_date_str,
                        trigger_price=grid.close_mat[c_i, t - 1], shadow_stop=init_stop,
                        highest_high=o_today, breakout_quality=breakout_strength,
                        entry_atr=a_yesterday, bars_in_watchlist=0
                    )))

            new_candidates.sort(key=lambda x: x["breakout_quality"], reverse=True)
            self.state["watchlist"].extend(new_candidates)

            # E. ORDER FILL EVALUATION
            # On past missed days: do NOT execute retroactive buys (days run dry for entries as requested)
            if not is_decision_day:
                if wl_mode == "WL_NONE":
                    self.state["watchlist"] = []
                continue

            # ------------------------------------------------------------------
            # TODAY'S LIVE DECISION DAY LOGIC
            # ------------------------------------------------------------------
            # 1. Apply Manual Wallet Override if active
            if MANUAL_WALLET_OVERRIDE is not None:
                print(f"[OVERRIDE] Resetting wallet balance to manual override: ${MANUAL_WALLET_OVERRIDE:,.2f}")
                self.state["wallet_cash"] = float(MANUAL_WALLET_OVERRIDE)

            # 2. Apply Manual Position Removals
            for coin_rm in MANUAL_POSITIONS_REMOVE:
                clean_rm = coin_rm.upper().strip()
                if clean_rm in self.state["positions"]:
                    p_rm = self.state["positions"][clean_rm]
                    c_i = p_rm["coin_idx"]
                    adv_30d = max(grid.dvol30_mat[c_i, t], LIQUIDITY_FLOOR_USD)
                    _, credit = calc_sell_fill(p_rm["units"], grid.close_mat[c_i, t], adv_30d)
                    self.state["wallet_cash"] += credit
                    del self.state["positions"][clean_rm]
                    self.action_feed.append(ActionItem(
                        action_type="FULL_EXIT", coin=clean_rm, dollar_amount=credit, units=p_rm["units"],
                        estimated_price=grid.close_mat[c_i, t], stop_loss=0.0, reason="MANUAL_OVERRIDE_EXIT",
                        notes=f"Manually forced exit for {clean_rm}. Order executed in paper portfolio."
                    ))

            # 3. Apply Manual Watchlist Removals
            for coin_wl_rm in MANUAL_WATCHLIST_REMOVE:
                clean_wl_rm = coin_wl_rm.upper().strip()
                self.state["watchlist"] = [item for item in self.state["watchlist"] if item["coin"] != clean_wl_rm]

            # 4. Apply Manual Position Injections
            for man_pos in MANUAL_POSITIONS_ADD:
                m_coin = man_pos.get("coin", "").upper().strip()
                if m_coin in grid.symbols and m_coin not in self.state["positions"]:
                    m_idx = grid.symbols.index(m_coin)
                    m_units = float(man_pos.get("units", 0.0))

                    # Automatically convert INR to USD if entry_price_inr is provided
                    if "entry_price_inr" in man_pos:
                        inr_px = float(man_pos["entry_price_inr"])
                        m_px = inr_px / USDT_INR_RATE
                        print(f"[INR CONVERSION] {m_coin}: ₹{inr_px:.4f} converted to ${m_px:.4f} USD (@ ₹{USDT_INR_RATE}/USDT)")
                    else:
                        m_px = float(man_pos.get("entry_price", grid.close_mat[m_idx, t]))

                    m_atr = float(grid.atr14_mat[m_idx, t])
                    m_sl = m_px - (p.get("sl_mult", 3.0) * m_atr)
                    m_cost = m_units * m_px

                    injected_pos = Position(
                        tid=int(time.time() * 1000) % 1000000, coin=m_coin, coin_idx=m_idx,
                        entry_date=man_pos.get("entry_date", curr_date_str), entry_price=m_px,
                        initial_units=m_units, units=m_units, cost_usd=m_cost, entry_atr=m_atr,
                        current_sl=m_sl, stop_reason="MANUAL_INJECTION_STOP",
                        highest_high=max(m_px, grid.high_mat[m_idx, t]),
                        lowest_low=min(m_px, grid.low_mat[m_idx, t]),
                        layer=1, days_held=0, tp_done=False
                    )
                    self.state["positions"][m_coin] = asdict(injected_pos)
                    print(f"[OVERRIDE] Successfully injected manual position: {m_coin} ({m_units} units @ ${m_px:,.2f})")

            # 5. Process Actionable Buy Signals for Today
            if self.state["watchlist"] and self.state["wallet_cash"] >= TRANCHE_FLOOR_USD and len(self.state["positions"]) < max_slots:
                if wl_mode == "WL_DEEPEST_DISCOUNT":
                    self.state["watchlist"].sort(key=lambda x: (x["trigger_price"] - grid.close_mat[x["coin_idx"], t]) / max(1e-6, x["trigger_price"]), reverse=True)
                elif wl_mode == "WL_STRONGEST_MOMENTUM":
                    self.state["watchlist"].sort(key=lambda x: x["breakout_quality"], reverse=True)

                unfilled = []
                for item_dict in self.state["watchlist"]:
                    c_i = item_dict["coin_idx"]
                    coin = item_dict["coin"]

                    if not grid.alive_mat[c_i, t]:
                        continue

                    # Filter revalidation
                    if not macro_ok[t]:
                        continue
                    if p.get("adx_thresh", 0.0) > 0.0 and grid.adx14_mat[c_i, t] < p["adx_thresh"]:
                        continue
                    if not state_mat[c_i, t]:
                        continue

                    coin_layers = sum(1 for c, p_d in self.state["positions"].items() if c == coin)
                    # Use live price (LTP) if running today; fallback to candle open
                    live_px = fetch_live_ltp(coin) if is_decision_day else None
                    today_open = live_px if (live_px is not None and live_px > 0) else grid.open_mat[c_i, t]

                    # Anti-averaging down check
                    if coin_layers > 0:
                        highest_prior_entry = max(p_d["entry_price"] for c, p_d in self.state["positions"].items() if c == coin)
                        if today_open <= highest_prior_entry * 1.005:
                            unfilled.append(item_dict)
                            continue

                    open_slots = max(1, max_slots - len(self.state["positions"]))
                    if len(self.state["positions"]) >= max_slots or self.state["wallet_cash"] < TRANCHE_FLOOR_USD:
                        unfilled.append(item_dict)
                        continue

                    open_active_cap = sum(p_d["units"] * grid.close_mat[p_d["coin_idx"], t] for p_d in self.state["positions"].values())
                    current_equity = self.state["wallet_cash"] + open_active_cap
                    max_pos_cap = current_equity * MAX_POSITION_EQUITY_PCT
                    dynamic_slot_target = min(max_pos_cap, self.state["wallet_cash"] / float(open_slots))

                    current_asset_exposure = sum(p_d["units"] * today_open for p_d in self.state["positions"].values() if p_d["coin"] == coin)
                    remaining_asset_capacity = max(0.0, max_pos_cap - current_asset_exposure)
                    if remaining_asset_capacity < TRANCHE_FLOOR_USD:
                        unfilled.append(item_dict)
                        continue

                    adv_30d = max(grid.dvol30_mat[c_i, t], LIQUIDITY_FLOOR_USD)
                    liquidity_cap_usd = adv_30d * MAX_ADV_PARTICIPATION
                    scaled_tranche_ceiling = max(10_000.0, current_equity * 0.35)
                    target_usd = min(dynamic_slot_target, scaled_tranche_ceiling, liquidity_cap_usd, remaining_asset_capacity)

                    max_affordable = compute_max_affordable_tranche(self.state["wallet_cash"], adv_30d)
                    tranche_usd = min(max_affordable, max(TRANCHE_FLOOR_USD, target_usd))

                    fill_px, units, total_cost = calc_buy_fill(tranche_usd, today_open, adv_30d)

                    if (self.state["wallet_cash"] >= total_cost and tranche_usd >= TRANCHE_FLOOR_USD and
                            units > 0 and len(self.state["positions"]) < max_slots and coin_layers < max_pyramid):

                        sl_price = fill_px - (p.get("sl_mult", 3.0) * item_dict["entry_atr"])
                        self.state["wallet_cash"] -= total_cost

                        new_pos = Position(
                            tid=int(time.time() * 1000) % 1000000, coin=coin, coin_idx=c_i,
                            entry_date=curr_date_str, entry_price=fill_px, initial_units=units,
                            units=units, cost_usd=total_cost, entry_atr=item_dict["entry_atr"],
                            current_sl=sl_price, stop_reason="STOP_LOSS",
                            highest_high=fill_px,
                            lowest_low=fill_px,
                            layer=coin_layers + 1, days_held=0, tp_done=False
                        )
                        self.state["positions"][coin] = asdict(new_pos)

                        self.action_feed.append(ActionItem(
                            action_type="BUY", coin=coin, dollar_amount=total_cost, units=units,
                            estimated_price=fill_px, stop_loss=sl_price, reason="SIGNAL_QUALIFIED",
                            notes=f"BUY SIGNAL: Allocate ${total_cost:,.2f} into {coin} (~{units:.4f} units @ ${fill_px:.4f}). Set Initial Stop Loss at ${sl_price:.4f}."
                        ))
                    else:
                        unfilled.append(item_dict)

                self.state["watchlist"] = unfilled

            if wl_mode == "WL_NONE":
                self.state["watchlist"] = []

        # Mark terminal bar as processed
        self.state["last_processed_date"] = str(dates[-1].date())
        save_state(self.state)
        return self.action_feed, self.state

    def _evaluate_current_positions_telemetry(self, grid: MarketGrid, t: int):
        """Read-only evaluation when already up-to-date."""
        for coin, p_dict in self.state["positions"].items():
            c_i = p_dict["coin_idx"]
            curr_px = grid.close_mat[c_i, t]
            sl_px = p_dict["current_sl"]
            if curr_px <= sl_px:
                self.action_feed.append(ActionItem(
                    action_type="FULL_EXIT", coin=coin, dollar_amount=0.0, units=p_dict["units"],
                    estimated_price=curr_px, stop_loss=sl_px, reason="STOP_LOSS_BREACHED",
                    notes=f"CRITICAL: {coin} price (${curr_px:.4f}) has breached stop loss (${sl_px:.4f}). Exit on exchange!"
                ))


# ==============================================================================
# 8. EXECUTIVE HTML EMAIL REPORT & DISPATCHER
# ==============================================================================

def generate_executive_html_email(actions: List[ActionItem], state: Dict[str, Any],
                                  grid: MarketGrid, config: Dict[str, Any],
                                  missed_alerts: List[str]) -> str:
    now_ist = pd.Timestamp.now("Asia/Kolkata").strftime("%A, %d %B %Y | %I:%M %p IST")
    latest_bar_date = state.get("last_processed_date", "N/A")

    # Financial Summary
    cash = state.get("wallet_cash", 0.0)
    positions = state.get("positions", {})
    t = len(grid.dates) - 1

    active_equity = 0.0
    pos_rows_html = ""
    for coin, p_dict in positions.items():
        c_i = p_dict["coin_idx"]
        cur_px = grid.close_mat[c_i, t]
        val = p_dict["units"] * cur_px
        active_equity += val
        cost = p_dict["cost_usd"]
        pnl = val - cost
        pnl_pct = (pnl / cost) * 100.0 if cost > 0 else 0.0
        pnl_color = "#10b981" if pnl >= 0 else "#ef4444"
        pnl_sign = "+" if pnl >= 0 else ""

        sl = p_dict["current_sl"]
        dist_sl = ((cur_px - sl) / cur_px) * 100.0 if cur_px > 0 else 0.0

        pos_rows_html += f"""
        <tr style="border-bottom: 1px solid #1e293b; font-size: 13px;">
            <td style="padding: 10px; font-weight: 700; color: #f8fafc;">{coin}</td>
            <td style="padding: 10px; color: #94a3b8;">${p_dict['entry_price']:,.4f}</td>
            <td style="padding: 10px; color: #f8fafc; font-weight: 600;">${cur_px:,.4f}</td>
            <td style="padding: 10px; color: {pnl_color}; font-weight: 700;">{pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)</td>
            <td style="padding: 10px; color: #f59e0b;">${sl:,.4f} <span style="font-size: 11px; color: #64748b;">({dist_sl:.1f}%)</span></td>
            <td style="padding: 10px; color: #94a3b8;">{p_dict['days_held']} / {config.get('max_holding_bars', 25)}d</td>
            <td style="padding: 10px; text-align: center;">{'<span style="color:#10b981;">&#10004; Taken</span>' if p_dict.get('tp_done') else '<span style="color:#64748b;">Pending</span>'}</td>
        </tr>
        """

    total_equity = cash + active_equity
    macro_px = grid.macro_close[-1]
    macro_ma_len = config.get("macro_ma_len", 100)
    macro_ma = FastIndicators.moving_average(grid.macro_close, macro_ma_len, config.get("macro_ma_type", 0))[-1]
    macro_bullish = macro_px > macro_ma if not np.isnan(macro_ma) else True
    macro_badge = f'<span style="background-color: {"#065f46" if macro_bullish else "#7f1d1d"}; color: {"#34d399" if macro_bullish else "#f87171"}; padding: 4px 8px; border-radius: 4px; font-weight: 700;">{"BULLISH REGIME" if macro_bullish else "BEARISH REGIME"} (BTC > {macro_ma_len}MA)</span>'

    # Action Items Table
    action_rows_html = ""
    if not actions:
        action_rows_html = """
        <tr>
            <td colspan="5" style="padding: 20px; text-align: center; color: #94a3b8; font-style: italic;">
                No immediate manual exchange actions required for today. Portfolio allocation is fully optimized.
            </td>
        </tr>
        """
    else:
        for a in actions:
            badge_color = "#10b981" if a.action_type == "BUY" else ("#ef4444" if a.action_type == "FULL_EXIT" else "#f59e0b")
            action_rows_html += f"""
            <tr style="border-bottom: 1px solid #1e293b; font-size: 13px;">
                <td style="padding: 12px;"><span style="background-color: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color}55; padding: 4px 8px; border-radius: 4px; font-weight: 700;">{a.action_type}</span></td>
                <td style="padding: 12px; font-weight: 700; color: #f8fafc;">{a.coin}</td>
                <td style="padding: 12px; color: #38bdf8; font-weight: 600;">{f"${a.dollar_amount:,.2f}" if a.dollar_amount > 0 else "-"}</td>
                <td style="padding: 12px; color: #e2e8f0;">{f"{a.units:.4f}" if a.units > 0 else "-"}</td>
                <td style="padding: 12px; color: #cbd5e1;">{a.notes}</td>
            </tr>
            """

    # Watchlist Rows
    wl_rows_html = ""
    watchlist = state.get("watchlist", [])
    if not watchlist:
        wl_rows_html = "<tr><td colspan='4' style='padding: 15px; text-align: center; color: #64748b;'>Watchlist is currently empty.</td></tr>"
    else:
        for w in watchlist[:8]:
            wl_rows_html += f"""
            <tr style="border-bottom: 1px solid #1e293b; font-size: 12px; color: #94a3b8;">
                <td style="padding: 8px; font-weight: 600; color: #f8fafc;">{w['coin']}</td>
                <td style="padding: 8px;">${w['trigger_price']:,.4f}</td>
                <td style="padding: 8px; color: #f59e0b;">${w['shadow_stop']:,.4f}</td>
                <td style="padding: 8px;">{w['bars_in_watchlist']}d / {WL_MAX_AGE_BARS}d</td>
            </tr>
            """

    # Missed Alerts Box
    missed_html = ""
    if missed_alerts:
        alerts_li = "".join([f"<li style='margin-bottom: 4px;'>{alert}</li>" for alert in missed_alerts])
        missed_html = f"""
        <div style="background-color: #451a03; border-left: 4px solid #f59e0b; padding: 14px; margin-bottom: 24px; border-radius: 4px;">
            <h4 style="margin: 0 0 6px 0; color: #fbbf24; font-size: 14px;">⚠️ Missed Days Catch-Up Audit</h4>
            <ul style="margin: 0; padding-left: 20px; color: #fde68a; font-size: 12px;">{alerts_li}</ul>
        </div>
        """

    # Assembled HTML Template
    return f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="background-color: #090d16; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #e2e8f0; margin: 0; padding: 24px;">
        <div style="max-width: 860px; margin: 0 auto; background-color: #0f172a; border: 1px solid #1e293b; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.5);">
            
            <!-- HEADER -->
            <div style="background: linear-gradient(135deg, #1e293b, #0f172a); padding: 24px 30px; border-bottom: 1px solid #334155;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h1 style="margin: 0; font-size: 22px; color: #f8fafc; font-weight: 800; letter-spacing: -0.5px;">CRYPTO QUANT V10.2 · LIVE DISPATCH</h1>
                        <p style="margin: 4px 0 0 0; font-size: 12px; color: #94a3b8;">Executed at {now_ist} · Evaluated closed bar: <strong style="color: #38bdf8;">{latest_bar_date}</strong></p>
                    </div>
                    <div style="text-align: right;">{macro_badge}</div>
                </div>
            </div>

            <div style="padding: 24px 30px;">
                {missed_html}

                <!-- EXECUTIVE TELEMETRY TILES -->
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px;">
                    <div style="background-color: #1e293b; padding: 14px; border-radius: 8px; border: 1px solid #334155;">
                        <span style="font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">Total Net Equity</span>
                        <div style="font-size: 20px; font-weight: 800; color: #f8fafc; margin-top: 4px;">${total_equity:,.2f}</div>
                    </div>
                    <div style="background-color: #1e293b; padding: 14px; border-radius: 8px; border: 1px solid #334155;">
                        <span style="font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">Liquid Wallet Cash</span>
                        <div style="font-size: 20px; font-weight: 800; color: #38bdf8; margin-top: 4px;">${cash:,.2f}</div>
                    </div>
                    <div style="background-color: #1e293b; padding: 14px; border-radius: 8px; border: 1px solid #334155;">
                        <span style="font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">Active Exposure</span>
                        <div style="font-size: 20px; font-weight: 800; color: #f8fafc; margin-top: 4px;">${active_equity:,.2f}</div>
                    </div>
                    <div style="background-color: #1e293b; padding: 14px; border-radius: 8px; border: 1px solid #334155;">
                        <span style="font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">Portfolio Slots</span>
                        <div style="font-size: 20px; font-weight: 800; color: #f59e0b; margin-top: 4px;">{len(positions)} / {config.get('max_concurrent_tranches', 6)}</div>
                    </div>
                </div>

                <!-- IMMEDIATE EXCHANGE ACTIONS -->
                <h3 style="font-size: 15px; text-transform: uppercase; letter-spacing: 0.5px; color: #f8fafc; margin: 24px 0 10px 0; display: flex; align-items: center;">
                    <span style="color: #38bdf8; margin-right: 8px;">⚡</span> Immediate Exchange Actions to Mirror
                </h3>
                <div style="background-color: #141d2f; border: 1px solid #1e293b; border-radius: 8px; overflow: hidden; margin-bottom: 24px;">
                    <table style="width: 100%; border-collapse: collapse; text-align: left;">
                        <thead>
                            <tr style="background-color: #1a2438; border-bottom: 1px solid #334155; font-size: 11px; color: #94a3b8; text-transform: uppercase;">
                                <th style="padding: 10px 12px;">Action</th>
                                <th style="padding: 10px 12px;">Coin</th>
                                <th style="padding: 10px 12px;">Amount</th>
                                <th style="padding: 10px 12px;">Est. Units</th>
                                <th style="padding: 10px 12px;">Instructions</th>
                            </tr>
                        </thead>
                        <tbody>{action_rows_html}</tbody>
                    </table>
                </div>

                <!-- ACTIVE PORTFOLIO -->
                <h3 style="font-size: 15px; text-transform: uppercase; letter-spacing: 0.5px; color: #f8fafc; margin: 24px 0 10px 0;">
                    📊 Active Paper Portfolio ({len(positions)} Positions)
                </h3>
                <div style="background-color: #141d2f; border: 1px solid #1e293b; border-radius: 8px; overflow: hidden; margin-bottom: 24px;">
                    <table style="width: 100%; border-collapse: collapse; text-align: left;">
                        <thead>
                            <tr style="background-color: #1a2438; border-bottom: 1px solid #334155; font-size: 11px; color: #94a3b8; text-transform: uppercase;">
                                <th style="padding: 10px;">Coin</th>
                                <th style="padding: 10px;">Entry</th>
                                <th style="padding: 10px;">Current</th>
                                <th style="padding: 10px;">Unrealized PnL</th>
                                <th style="padding: 10px;">Stop Loss</th>
                                <th style="padding: 10px;">Holding Time</th>
                                <th style="padding: 10px; text-align: center;">50% TP</th>
                            </tr>
                        </thead>
                        <tbody>{pos_rows_html if pos_rows_html else "<tr><td colspan='7' style='padding: 15px; text-align: center; color: #64748b;'>No active open positions.</td></tr>"}</tbody>
                    </table>
                </div>

                <!-- WATCHLIST CANDIDATES -->
                <h3 style="font-size: 15px; text-transform: uppercase; letter-spacing: 0.5px; color: #f8fafc; margin: 24px 0 10px 0;">
                    🎯 Active Watchlist Candidates ({len(watchlist)})
                </h3>
                <div style="background-color: #141d2f; border: 1px solid #1e293b; border-radius: 8px; overflow: hidden;">
                    <table style="width: 100%; border-collapse: collapse; text-align: left;">
                        <thead>
                            <tr style="background-color: #1a2438; border-bottom: 1px solid #334155; font-size: 11px; color: #94a3b8; text-transform: uppercase;">
                                <th style="padding: 8px;">Coin</th>
                                <th style="padding: 8px;">Trigger Price</th>
                                <th style="padding: 8px;">Shadow Stop</th>
                                <th style="padding: 8px;">Watchlist Age</th>
                            </tr>
                        </thead>
                        <tbody>{wl_rows_html}</tbody>
                    </table>
                </div>

                <!-- FOOTER -->
                <div style="margin-top: 30px; padding-top: 15px; border-top: 1px solid #1e293b; font-size: 11px; color: #64748b; text-align: center;">
                    V10.2 Quantitative Strategy Engine · Fully Automated Non-Custodial Paper Ledger
                </div>

            </div>
        </div>
    </body>
    </html>
    """

def dispatch_gmail_notification(subject: str, html_body: str):
    if not (GMAIL_USER and GMAIL_APP_PASSWORD and RECIPIENT_EMAIL):
        print("[INFO] Gmail credentials not set or incomplete. Skipping email dispatch.")
        return

    if "your_email" in GMAIL_USER or "your_app_password" in GMAIL_APP_PASSWORD:
        print("[INFO] Default placeholder credentials detected. Please configure real Gmail credentials to receive emails.")
        return

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"Crypto Quant Engine <{GMAIL_USER}>"
        msg["To"] = RECIPIENT_EMAIL
        msg.attach(MIMEText(html_body, "html"))

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_USER, RECIPIENT_EMAIL, msg.as_string())
        print(f"[SUCCESS] Executive HTML report successfully dispatched to {RECIPIENT_EMAIL}")
    except Exception as e:
        print(f"[ERROR] Failed to send email via Gmail SMTP: {e}")


# ==============================================================================
# 9. MAIN ORCHESTRATION PIPELINE
# ==============================================================================

def load_active_config() -> Dict[str, Any]:
    """Loads winner.json if present; otherwise defaults to DEFAULT_CONFIG."""
    for cand_file in [FINAL_WINNER_FILE if 'FINAL_WINNER_FILE' in globals() else WINNER_CONFIG_FILE, CURRENT_WINNER_FILE]:
        if cand_file.exists():
            try:
                with open(cand_file, "r") as f:
                    data = json.load(f)
                    best_params = data.get("best_params", {})
                    if best_params:
                        print(f"[CONFIG] Loaded optimized parameters from {cand_file.name}")
                        return {**DEFAULT_CONFIG, **best_params}
            except Exception:
                pass
    print("[CONFIG] Using default built-in strategy configuration.")
    return copy.deepcopy(DEFAULT_CONFIG)

def main():
    print("=" * 80)
    print(f"CRYPTO QUANTITATIVE LIVE PAPER TRADING ENGINE (V10.2 INSTITUTIONAL)")
    print(f"Execution Time: {pd.Timestamp.now('Asia/Kolkata').strftime('%Y-%m-%d %H:%M:%S IST')}")
    print("=" * 80)

    # 1. Load Strategy Configuration
    config = load_active_config()

    # 2. Ingest Multi-Exchange Daily Market Grid
    print("[INIT] Fetching clean multi-exchange market grid...")
    grid, _ = build_live_market_grid(top_n=TOP_N_COINS)
    print(f"[INIT] Market grid loaded: {len(grid.symbols)} coins across {len(grid.dates)} bars up to {grid.dates[-1].date()}.")

    # 3. Initialize Execution Engine & Run Daily Catch-up Cycle
    engine = LiveCryptoExecutionEngine(config=config)
    actions, state = engine.run_daily_cycle(grid)

    # 4. Display Exchange Action Feed on Console
    print("\n" + "=" * 80)
    print("EXCHANGE ACTION FEED (MIRROR THESE ON YOUR EXCHANGE)")
    print("=" * 80)
    if not actions:
        print(">> No orders required today. Portfolio is active and holding.")
    else:
        for idx, a in enumerate(actions, 1):
            print(f"{idx}. [{a.action_type}] {a.coin}: {a.notes}")
    print("=" * 80 + "\n")

    # 5. Display Paper Portfolio Status
    cash = state.get("wallet_cash", 0.0)
    positions = state.get("positions", {})
    t = len(grid.dates) - 1
    active_equity = sum(p_d["units"] * grid.close_mat[p_d["coin_idx"], t] for p_d in positions.values())
    print(f"Current Portfolio Valuation:")
    print(f"  Available Cash:   ${cash:,.2f}")
    print(f"  Invested Equity:  ${active_equity:,.2f}")
    print(f"  Total Net Worth:  ${cash + active_equity:,.2f}")
    print(f"  Open Positions:   {len(positions)} / {config.get('max_concurrent_tranches', 6)}\n")

    # 6. Generate & Send Executive HTML Email
    if SEND_EMAIL_NOTIFICATION:
        html_report = generate_executive_html_email(actions, state, grid, config, engine.missed_alerts)
        subject = f"Crypto Quant Dispatch: {len(actions)} Actions · Net Worth: ${cash + active_equity:,.2f}"
        dispatch_gmail_notification(subject, html_report)

    print("[DONE] Daily cycle completed successfully.")

if __name__ == "__main__":
    main()