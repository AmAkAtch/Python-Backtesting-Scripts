#!/usr/bin/env python3
"""
CRYPTO LIVE BOT V14.0-CRYPTO -- paper trading signal generator.
Run it AFTER 05:30 IST (the daily candle closes at 00:00 UTC = 05:30 IST).

  * 100% Rulebook Parity with `bt_crypto_14.py` (V14.0-CRYPTO):
    - 2D Vectorized Indicator Bank + Composable Signals (Solo or 1-2 Confirmation Filters,
      0-2 OR Exits, PCT/ATR Trail, and pyramid spacing rules).
    - CoinDCX spot fees (0.60% per side), per-bar dynamic liquidity tiers (LARGE / MID / SMALL),
      2x breakout/surge slippage multiplier, and 24/7 token-swap/redenomination guard.
  * The ledger holds FACTS only (cash, positions with entry date / price / units / cost,
    watchlist, last plan, closed trades). Stops, trailing levels, breakeven moves, holding clocks,
    and exits are re-evaluated from price history on EVERY run under the current winner config.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import io
import json
import logging
import os
import shutil
import smtplib
import ssl
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import requests

# ==============================================================================
# 1. SETTINGS (edit this block)
# ==============================================================================
DRY_RUN: bool = True                       # True = compute + show + email, save NOTHING. Set False once verified.
SEND_EMAIL_NOTIFICATION: bool = True
ALLOW_NOT_PROMOTED_CONFIG: bool = True     # True = use winner_candidate.json when winner.json is absent / not promoted
LIVE_CONFIG_OVERRIDE: Optional[str] = None # Path to a specific winner json; None = winner.json, else winner_candidate.json

STARTING_CASH_INR: float = 10000.0             # Free cash on first run (1 INR = 1 ledger unit; 0 = no new BUY orders planned)
COINDCX_ONLY: bool = False                 # False = spot pairs such as ONDO/USDT on CoinDCX (no INR-market filter)
USDINR_FALLBACK: float = 98.0              # Fallback rate if live USDT/INR cannot be fetched
LIVE_QUOTE_DRIFT_WARN_PCT: float = 3.0     # Warn when live price drifts from the bar open used by the plan
COLAB_WORK_DIR: str = "/content/drive/MyDrive/crypto_live_v14"
WORK_DIR_OVERRIDE: Optional[str] = None    # Local override (e.g. crypto_backtest folder to share data_cache & winner json)

# Coins you ALREADY hold on CoinDCX (optional). Added once each.
MANUAL_HOLDINGS_ADD: List[Dict[str, Any]] = [
    {"coin": "ONDO", "units": 43.1, "entry_price_inr": 43.046, "entry_date": "2026-09-25"},
    {"coin": "LTC", "units": 0.138, "entry_price_inr": 7107.9, "entry_date": "2026-09-25"},
    {"coin": "H", "units": 132, "entry_price_inr": 7.4618, "entry_date": "2026-08-21"},
]
# Positions sold manually: {"coin": "X", "entry_date": "YYYY-MM-DD", "exit_date": "YYYY-MM-DD", "exit_price_inr": 123.4}
MANUAL_HOLDINGS_REMOVE: List[Dict[str, Any]] = []
# Cash deposited (+) or withdrawn (-), applied once by id: {"id": "2026-11-25-sip", "amount_inr": 5000.0}
CASH_ADJUSTMENTS: List[Dict[str, Any]] = []
# Order ids from the plan that you did NOT execute:
ORDER_SKIPS: List[str] = []
# First run only: rebuild standing watchlist over the last N bars (0 = off).
WATCHLIST_WARM_START_BARS: int = 60

# Credentials: environment / .env / Colab secrets
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass
try:
    from google.colab import userdata  # type: ignore
    GMAIL_USER = userdata.get("GMAIL_USER") or os.environ.get("GMAIL_USER", "")
    GMAIL_APP_PASSWORD = userdata.get("GMAIL_APP_PASSWORD") or os.environ.get("GMAIL_APP_PASSWORD", "")
    RECIPIENT_EMAIL = userdata.get("RECIPIENT_EMAIL") or os.environ.get("RECIPIENT_EMAIL", "")
except Exception:
    GMAIL_USER = os.environ.get("GMAIL_USER", "")
    GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
    RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL", "")

# ==============================================================================
# 2. WORK FOLDER
# ==============================================================================
def _resolve_work_dir() -> Path:
    if WORK_DIR_OVERRIDE:
        return Path(WORK_DIR_OVERRIDE)
    try:
        from google.colab import drive  # type: ignore
        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
        return Path(COLAB_WORK_DIR)
    except ImportError:
        return Path(os.environ.get("CRYPTO_LIVE_DIR", "")) if os.environ.get("CRYPTO_LIVE_DIR") else Path(__file__).resolve().parent / "crypto_live_v14"


WORK_DIR: Path = _resolve_work_dir()
WORK_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_DIR)
STATE_DIR: Path = WORK_DIR / "live_v14_state"
USDINR: float = USDINR_FALLBACK

# ==============================================================================
# 3. ENGINE CORE (V14.0-CRYPTO PARITY)
# ==============================================================================
ENGINE_VERSION: str = "V14.0-CRYPTO"
UNIVERSE_NAME: str = "TOP_CRYPTO_LIQUID"
TOP_N_COINS: int = 50

MACRO_INDEX_NAME: str = "BTC"
MACRO_INDEX_TICKER: str = "BTC"
BENCH_SYMBOL: str = "BTC"

FORCE_MACRO_REGIME_FILTER: Optional[bool] = None
FORCE_TRAIL_STOP: Optional[bool] = None

BARS_PER_YEAR: float = 365.0
START_YEAR: int = 2017
QUOTE_CURRENCY: str = "USD"
INITIAL_CAPITAL: float = 1_000.0
MONTHLY_CONTRIBUTION: float = 1_000.0
CASH_ANNUAL_YIELD: float = 0.0
MIN_HISTORY_DAYS: int = 300
WARMUP_BARS: int = 300

PARALLEL_DOWNLOAD_WORKERS: int = 8
FORCE_REFRESH: bool = False
CACHE_MAX_AGE_HOURS: float = 48.0

V13_FIXED_RULES: Dict[str, Any] = {
    "wl_dedupe": True,
    "wl_max_age_bars": 0,
    "whole_shares": False,
}
ENGINE_DEFAULTS: Dict[str, Any] = {
    **V13_FIXED_RULES,
    "wl_revalidate": True,
    "avg_down_free_adds": 0,
    "pyramid_rule": 0,
    "pyramid_gap_bars": 0,
    "pyramid_progress_atr": 0.0,
    "size_w0": 0.05,
    "size_w_max": 0.15,
    "size_reserve": 0.10,
    "size_ramp": 0.05,
}

PYRAMID_GAP_CHOICES: List[int] = [5, 10, 20]
PYRAMID_PROGRESS_CHOICES: List[float] = [1.0, 1.5, 2.0]
SIZE_RUNT_FRACTION: float = 0.6

ADV_WINDOW: int = 30
ADV_MIN_PERIODS: int = 30
LIQUIDITY_FLOOR_USD: float = 2_000_000.0
TIER_ADV_LARGE_USD: float = 300_000_000.0
TIER_ADV_MID_USD: float = 50_000_000.0
BRACKET_SLIPPAGE_BPS: Dict[str, float] = {"LARGE": 6.0, "MID": 12.0, "SMALL": 25.0}
BRACKET_IMPACT_COEF_BPS: Dict[str, float] = {"LARGE": 120.0, "MID": 180.0, "SMALL": 300.0}
VOL_BREAKOUT_SLIP_MULT: float = 2.0
SLIPPAGE_SCALE: float = 1.0

TRANCHE_FLOOR_USD: float = 50.0
MAX_POSITION_EQUITY_PCT: float = 0.25
MAX_ADV_PARTICIPATION: float = 0.015

EXCLUDED_SYMBOLS: Set[str] = {
    "USD1", "USDS", "FIGR_HELOC", "SUSDS", "SUSDE", "USDG", "USDY", "USDTB", "BUIDL", "USYC", "BFUSD"
}
ALLOWED_PROVIDERS: Optional[Set[str]] = {"binance", "bybit", "okx"}
PEGGED_BAND: float = 0.04
PEGGED_SHARE: float = 0.90

SPIKE_MIN_MOVE: float = 0.50
SPIKE_REVERT_WITHIN: int = 5
SPIKE_REVERT_TOL: float = 0.30
SPIKE_OPEN_EXPLAINS: float = 0.50
CA_DROP_Q: float = 0.70
CA_JUMP_Q: float = 1.90
CA_GAP_EXPLAINS: float = 0.80
CA_MARKET_DAY_MAX: float = 0.15
LARGE_MOVE_LOG: float = 0.40
MANUAL_CORPORATE_ACTIONS: Dict[str, List[str]] = {}
KEEP_AS_REAL_MOVES: Dict[str, List[str]] = {}
DATA_REPAIR_LOG: List[Dict[str, Any]] = []
DELIST_HAIRCUT: float = 0.50

CRYPTO_EXCHANGE_FEE_RATE: float = 0.0060
EFFECTIVE_BUY_FEE_RATE: float = CRYPTO_EXCHANGE_FEE_RATE
EFFECTIVE_SELL_FEE_RATE: float = CRYPTO_EXCHANGE_FEE_RATE
FLAT_SELL_FIXED_USD: float = 0.0

DATA_DIR = Path("data_cache")
OUTPUT_DIR = Path("output") / f"TOP_{TOP_N_COINS}"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPAIRS_CSV_FILE = OUTPUT_DIR / "DATA_REPAIRS.csv"

STABLECOINS: Set[str] = {
    "USD1", "USDS", "USDG", "FIGR_HELOC",
    "USDT", "USDC", "BUSD", "DAI", "FDUSD", "TUSD", "USDD", "USDP", "FRAX",
    "LUSD", "GUSD", "PYUSD", "EURT", "EURS", "USTC", "USDJ", "CUSD", "SUSD",
    "USDE", "USD0", "CRVUSD", "GHO", "FIRST", "MIM", "FEI", "ALUSD", "RLUSD"
}
FIAT_COMMODITY_PEGS: Set[str] = {"XAUT", "PAXG", "EUROC", "XSGD", "EURCV"}
WRAPPED_TOKEN_NAMES: Set[str] = {
    "WBTC", "WETH", "WMATIC", "WAVAX", "WBNB", "WSOL", "WFTM", "WTRX", "WKAVA",
    "WBETH", "WROSE", "WNEAR", "WCELO", "WKLAY", "WBTT", "WGLMR", "WQTUM", "CBBTC"
}
LIQUID_STAKING_PREFIXES: Tuple[str, ...] = (
    "ST", "WST", "R", "CB", "ANKR", "MSOL", "SAVAX", "BNSOL", "JITOSOL", "OSOL"
)
PROTECTED_TICKERS: Set[str] = {
    "STX", "STORJ", "ROSE", "RENDER", "RUNE", "RVN", "RAD", "REQ", "RLC",
    "RARE", "RAY", "RON", "RDNT", "RPL", "STEEM", "STRAX", "STRK"
}
LEVERAGED_SUFFIXES: Tuple[str, ...] = (
    "UP", "DOWN", "BULL", "BEAR", "2L", "2S", "3L", "3S", "4L", "4S", "5L", "5S"
)


def is_noise_or_wrapper_coin(raw_symbol: str) -> bool:
    sym = raw_symbol.upper().replace("-USD", "").replace("/USD", "").replace("/USDT", "").strip()
    if sym in STABLECOINS or sym in FIAT_COMMODITY_PEGS or sym in WRAPPED_TOKEN_NAMES:
        return True
    if any(sym.endswith(suf) for suf in LEVERAGED_SUFFIXES):
        return True
    if sym.startswith("W") and len(sym) >= 4 and sym[1:] in {
        "BTC", "ETH", "SOL", "BNB", "AVAX", "ADA", "MATIC", "POL", "DOT", "LINK", "NEAR"
    }:
        return True
    if sym not in PROTECTED_TICKERS:
        for pfx in LIQUID_STAKING_PREFIXES:
            if sym.startswith(pfx) and len(sym) > len(pfx):
                base = sym[len(pfx):]
                if base in {"ETH", "SOL", "MATIC", "AVAX", "DOT", "ADA", "BNB", "ATOM", "NEAR"}:
                    return True
    return False


def filter_crypto_universe(symbols: List[str]) -> List[str]:
    filtered, seen, ordered = [], set(), []
    for s in symbols:
        clean = s.upper().replace("-USD", "").strip()
        if not is_noise_or_wrapper_coin(clean):
            filtered.append(clean)
    for s in filtered:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def format_price(px: float) -> str:
    if not np.isfinite(px):
        return "$0.00"
    sign = "-" if px < 0 else ""
    abs_px = abs(px)
    if abs_px < 1e-9:
        return "$0.00"
    if abs_px >= 1.0:
        return f"{sign}${abs_px:,.2f}"
    elif abs_px >= 0.01:
        return f"{sign}${abs_px:,.4f}"
    else:
        return f"{sign}${abs_px:.7f}"


def shift_1d(arr: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    res = np.empty_like(arr)
    res[0] = fill_value
    res[1:] = arr[:-1]
    return res


def shift_2d(mat: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    out = np.empty_like(mat)
    out[:, 0] = fill_value
    out[:, 1:] = mat[:, :-1]
    return out


def np_rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    w = max(1, int(window))
    out = np.full_like(arr, np.nan, dtype=np.float64)
    if len(arr) < w:
        return out
    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return out
    clean = np.where(valid_mask, arr, 0.0)
    cumsum = np.insert(np.cumsum(clean), 0, 0.0)
    cntsum = np.insert(np.cumsum(valid_mask.astype(np.int32)), 0, 0)
    win_sum = cumsum[w:] - cumsum[:-w]
    win_cnt = cntsum[w:] - cntsum[:-w]
    out[w - 1:] = np.where(win_cnt == w, win_sum / float(w), np.nan)
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
        out[i] = out[i - 1] if np.isnan(val) else (alpha * val + (1.0 - alpha) * out[i - 1])
    return out


class FastIndicators:
    @staticmethod
    def moving_average(arr: np.ndarray, length: int, kind: int) -> np.ndarray:
        length = max(2, int(length))
        if kind == 0:
            return np_rolling_mean(arr, length)
        elif kind == 1:
            return np_ewm_mean(arr, length)
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
            conv = np.convolve(arr[first_valid:], w_norm[::-1], mode="full")[:n_valid]
            conv[:length - 1] = np.nan
            out[first_valid:] = conv
            return out
        raise ValueError(f"Unsupported moving-average kind {kind} (allowed: 0=SMA, 1=EMA, 3=WMA)")

    @staticmethod
    def atr_1d(h: np.ndarray, l: np.ndarray, c: np.ndarray, length: int = 14) -> np.ndarray:
        started = np.cumsum(~np.isnan(c)) > 0
        cp = shift_1d(c, fill_value=np.nan)
        tr1 = h - l
        tr2 = np.abs(h - cp)
        tr3 = np.abs(l - cp)
        tr = np.where(started, np.fmax(tr1, np.fmax(tr2, tr3)), np.nan)
        alpha = 1.0 / max(2, length)
        span = int(round((2.0 / alpha) - 1.0))
        return np_ewm_mean(tr, span)

    @staticmethod
    def adx_1d(h: np.ndarray, l: np.ndarray, c: np.ndarray, length: int = 14) -> np.ndarray:
        length = max(2, int(length))
        span = int(round((2.0 * length) - 1.0))
        up = h - shift_1d(h, fill_value=np.nan)
        down = shift_1d(l, fill_value=np.nan) - l
        valid = (~np.isnan(up)) & (~np.isnan(down))
        p_dm = np.where(valid, np.where((up > down) & (up > 0.0), up, 0.0), np.nan)
        m_dm = np.where(valid, np.where((down > up) & (down > 0.0), down, 0.0), np.nan)
        atr_arr = FastIndicators.atr_1d(h, l, c, length)
        p_di = 100.0 * np_ewm_mean(p_dm, span) / (atr_arr + 1e-9)
        m_di = 100.0 * np_ewm_mean(m_dm, span) / (atr_arr + 1e-9)
        dx = 100.0 * np.abs(p_di - m_di) / (p_di + m_di + 1e-9)
        return np_ewm_mean(dx, span)


@dataclass
class FeeBreakdown:
    exchange_fee: float = 0.0
    slippage_cost: float = 0.0


def compute_buy_cost_audited(gross_usd: float, slip_cost: float) -> Tuple[float, FeeBreakdown]:
    if gross_usd <= 0:
        return 0.0, FeeBreakdown()
    exch = gross_usd * CRYPTO_EXCHANGE_FEE_RATE
    return gross_usd + exch, FeeBreakdown(exchange_fee=exch, slippage_cost=slip_cost)


def compute_sell_proceeds_audited(gross_usd: float, slip_cost: float) -> Tuple[float, FeeBreakdown]:
    if gross_usd <= 0:
        return 0.0, FeeBreakdown()
    exch = gross_usd * CRYPTO_EXCHANGE_FEE_RATE
    return max(0.0, gross_usd - exch), FeeBreakdown(exchange_fee=exch, slippage_cost=slip_cost)


def compute_max_affordable_tranche(cash: float, adv_30d: float, base_bps: float, impact_bps: float) -> float:
    clean_adv = adv_30d if np.isfinite(adv_30d) else LIQUIDITY_FLOOR_USD
    adv = max(clean_adv, LIQUIDITY_FLOOR_USD)
    usable_cash = max(0.0, cash)
    tranche_guess = usable_cash / ((1.0 + EFFECTIVE_BUY_FEE_RATE) * (1.0 + base_bps / 10000.0))
    for _ in range(3):
        part_rate = min(1.0, max(0.0, tranche_guess / adv))
        slip_mult = (base_bps + impact_bps * np.sqrt(part_rate)) / 10000.0
        tranche_guess = usable_cash / ((1.0 + EFFECTIVE_BUY_FEE_RATE) * (1.0 + slip_mult))
    return float(np.nan_to_num(tranche_guess * (1.0 - 1e-6), nan=0.0))


def breakeven_exit_price(total_cost: float, units: float, slip_mult: float) -> float:
    denom = units * max(1e-6, (1.0 - slip_mult - EFFECTIVE_SELL_FEE_RATE))
    return (total_cost + FLAT_SELL_FIXED_USD) / max(1e-9, denom)


def _buy_fill_audited(tranche_usd: float, ref_open_price: float, slip_mult: float,
                      whole_shares: bool = False) -> Tuple[float, float, float, FeeBreakdown]:
    fill_px = ref_open_price * (1.0 + slip_mult)
    units = tranche_usd / ref_open_price if ref_open_price > 0 else 0.0
    if whole_shares:
        units = float(np.floor(units))
        if units < 1.0 and ref_open_price > 0 and ref_open_price <= 2.0 * tranche_usd:
            units = 1.0
    gross_at_slip = units * fill_px
    slip_cost = units * (fill_px - ref_open_price)
    total_cash_cost, fees = compute_buy_cost_audited(gross_at_slip, slip_cost)
    return fill_px, units, total_cash_cost, fees


def _sell_fill_audited(units: float, ref_price: float, slip_mult: float) -> Tuple[float, float, FeeBreakdown]:
    fill_px = ref_price * (1.0 - slip_mult)
    gross = units * fill_px
    slip_cost = units * (ref_price - fill_px)
    net_proceeds, fees = compute_sell_proceeds_audited(gross, slip_cost)
    return fill_px, net_proceeds, fees


DEFAULT_CRYPTO_LIST: List[str] = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOGE", "DOT", "LINK",
    "NEAR", "LTC", "BCH", "UNI", "APT", "ATOM", "ICP", "FIL", "ETC", "XLM",
    "RENDER", "HBAR", "AAVE", "INJ", "GRT", "VET", "ALGO", "OP", "ARB", "FTM",
    "THETA", "SAND", "MANA", "AXS", "FLOW", "EOS", "KAVA", "EGLD", "XTZ", "RUNE"
]

_HTTP_SESSION = requests.Session()
_HTTP_SESSION.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})


def _clean_kline_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    today_utc = pd.Timestamp.now("UTC").tz_localize(None).floor("D")
    df = df[df["date"] < today_utc].dropna(subset=["close"])
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df["close"] > 0) & (df["open"] > 0) & (df["high"] >= df["low"])]
    ratio = df["high"] / np.maximum(1e-8, df["low"])
    df = df[ratio < 50.0]
    df["high"] = np.maximum(df["high"], np.maximum(df["open"], df["close"]))
    df["low"] = np.minimum(df["low"], np.minimum(df["open"], df["close"]))
    return df.set_index("date")


def fetch_from_binance(symbol: str, start_year: int) -> Optional[pd.DataFrame]:
    hosts = ["https://data-api.binance.vision", "https://api.binance.com"]
    start_ts = int(pd.Timestamp(f"{start_year}-01-01", tz="UTC").timestamp() * 1000)
    now_ts = int(pd.Timestamp.now("UTC").floor("D").timestamp() * 1000)
    for host in hosts:
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
    for _ in range(12):
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


def fetch_from_okx(symbol: str) -> Optional[pd.DataFrame]:
    url = "https://www.okx.com/api/v5/market/history-candles"
    inst_id = f"{symbol}-USDT"
    all_rows = []
    after = ""
    for _ in range(30):
        params = {"instId": inst_id, "bar": "1Dutc", "limit": 100}
        if after:
            params["after"] = after
        try:
            r = _HTTP_SESSION.get(url, params=params, timeout=6)
            if r.status_code != 200:
                break
            data = r.json().get("data", [])
            if not data:
                break
            all_rows.extend(data)
            after = data[-1][0]
            if len(data) < 100:
                break
            time.sleep(0.02)
        except Exception:
            break
    if len(all_rows) >= MIN_HISTORY_DAYS:
        parsed = [[int(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4]), float(d[5]), float(d[6])] for d in all_rows]
        df = pd.DataFrame(parsed, columns=["date", "open", "high", "low", "close", "volume", "quote_volume"])
        df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
        return _clean_kline_dataframe(df)
    return None


def fetch_from_yahoo(symbol: str, start_year: int) -> Optional[pd.DataFrame]:
    start_ts = int(pd.Timestamp(f"{start_year}-01-01", tz="UTC").timestamp())
    end_ts = int(pd.Timestamp.now("UTC").timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}-USD?period1={start_ts}&period2={end_ts}&interval=1d"
    try:
        r = _HTTP_SESSION.get(url, timeout=8)
        if r.status_code != 200:
            return None
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


def fetch_single_crypto(symbol: str, start_year: int, refresh: bool) -> Tuple[str, Optional[pd.DataFrame], str]:
    clean_sym = symbol.upper().replace("-USD", "").replace("/USD", "").replace("/USDT", "").strip()
    cache_file = DATA_DIR / f"{clean_sym}_1d_from{start_year}.parquet"
    meta_file = DATA_DIR / f"{clean_sym}_1d_from{start_year}.meta.json"

    if not refresh and cache_file.exists() and meta_file.exists():
        age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600.0
        if age_hours < CACHE_MAX_AGE_HOURS:
            try:
                with open(meta_file, "r") as mf:
                    meta = json.load(mf)
                if meta.get("start_year") == start_year:
                    df = pd.read_parquet(cache_file)
                    if len(df) >= MIN_HISTORY_DAYS:
                        return clean_sym, df, meta.get("provider", "cache")
            except Exception:
                pass

    for provider_name, fetch_fn in [
        ("binance", lambda: fetch_from_binance(clean_sym, start_year)),
        ("bybit", lambda: fetch_from_bybit(clean_sym)),
        ("okx", lambda: fetch_from_okx(clean_sym)),
        ("yahoo", lambda: fetch_from_yahoo(clean_sym, start_year)),
    ]:
        try:
            df = fetch_fn()
            if df is not None and len(df) >= MIN_HISTORY_DAYS:
                df.to_parquet(cache_file)
                with open(meta_file, "w") as mf:
                    json.dump({
                        "provider": provider_name, "timestamp": time.time(),
                        "bars": len(df), "start_year": start_year
                    }, mf)
                return clean_sym, df, provider_name
        except Exception:
            continue

    return clean_sym, None, "none"


def load_universe_constituents(universe_name: str, top_n: int = TOP_N_COINS) -> List[str]:
    cache_path = DATA_DIR / f"{universe_name.lower()}_top{top_n}_list.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
                filtered = filter_crypto_universe(data)
                if len(filtered) >= min(15, top_n):
                    return filtered[:top_n]
        except Exception:
            pass

    candidates: List[str] = []
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            api_data = json.loads(resp.read().decode("utf-8"))
            raw_syms = [coin["symbol"].upper() for coin in api_data]
            candidates = filter_crypto_universe(raw_syms)
    except Exception as e:
        logging.warning(f"CoinGecko constituent discovery unavailable ({e}). Falling back to Binance volume.")

    if len(candidates) < top_n:
        try:
            resp = _HTTP_SESSION.get("https://api.binance.com/api/v3/ticker/24hr", timeout=8)
            if resp.status_code == 200:
                pairs = []
                for t in resp.json():
                    s = t.get("symbol", "")
                    if s.endswith("USDT"):
                        pairs.append((s[:-4], float(t.get("quoteVolume", 0.0))))
                pairs.sort(key=lambda x: x[1], reverse=True)
                candidates.extend(filter_crypto_universe([p[0] for p in pairs]))
        except Exception as e:
            logging.warning(f"Binance volume ranking fallback failed: {e}")

    if len(candidates) < top_n:
        candidates.extend(filter_crypto_universe(DEFAULT_CRYPTO_LIST))

    seen, final_ranked = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            final_ranked.append(c)

    chosen = final_ranked[:top_n]
    with open(cache_path, "w") as f:
        json.dump(chosen, f)
    return chosen


_REDENOM_RATIOS = (0.001, 0.01, 0.05, 0.10, 0.20, 0.25, 0.3333, 0.50, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 100.0, 1000.0)


def _is_plausible_split_ratio(qe: float) -> bool:
    for r in _REDENOM_RATIOS:
        if abs(qe - r) / r <= 0.08:
            return True
    return False


def repair_price_artifacts(df: pd.DataFrame, sym: str, mkt_ret: Optional[pd.Series],
                           allow_corporate: bool = True) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
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

    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.r_[1.0, c[1:] / np.where(c[:-1] > 0, c[:-1], np.nan)]
    cand = np.flatnonzero(np.abs(np.nan_to_num(q, nan=1.0) - 1.0) >= SPIKE_MIN_MOVE)
    skip_until = -1
    for i in cand:
        if i <= skip_until or i < 1 or i >= n - 1 or not np.isfinite(c[i - 1]) or c[i - 1] <= 0:
            continue
        if np.isfinite(o[i]) and o[i] > 0:
            _gap_i = o[i] / c[i - 1] - 1.0
            if _gap_i * (q[i] - 1.0) <= 0.0 or abs(_gap_i) < SPIKE_OPEN_EXPLAINS * abs(q[i] - 1.0):
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
            same_dir = (gap - 1.0) * (qe - 1.0) > 0.0
            explained = bool(same_dir and abs(gap - 1.0) >= CA_GAP_EXPLAINS * mag)
            split_like = _is_plausible_split_ratio(qe)
            m_day = 0.0
            if mkt_ret is not None:
                try:
                    m_day = abs(float(mkt_ret.get(pd.Timestamp(idx[e]).normalize(), 0.0)))
                    m_day = m_day if np.isfinite(m_day) else 0.0
                except Exception:
                    m_day = 0.0
            if forced or ((qe <= CA_DROP_Q or qe >= CA_JUMP_Q) and explained and split_like and m_day < CA_MARKET_DAY_MAX):
                _pc, _nc, _no = float(c[e - 1]), float(c[e]), float(o[e])
                o[:e] *= qe; h[:e] *= qe; l[:e] *= qe; c[:e] *= qe; v[:e] /= qe
                recs.append({"symbol": sym, "date": _date(e), "type": "CORP_ACTION_REBASED", "ratio": float(qe),
                             "detail": f"pre-event prices x{qe:.4f}, volume /{qe:.4f}; prev close {_pc:.4f}, open {_no:.4f}, close {_nc:.4f}" + (" (manual)" if forced else "")})
            else:
                recs.append({"symbol": sym, "date": _date(e), "type": "LARGE_MOVE_KEPT", "ratio": float(qe),
                             "detail": f"{mag * 100:.0f}% move kept (BTC day move {m_day * 100:.1f}%, open-gap explains: {bool(explained)}, redenom-ratio: {bool(split_like)})"})

    if any(r["type"] != "LARGE_MOVE_KEPT" for r in recs):
        d["open"], d["high"], d["low"], d["close"] = o, h, l, c
        if "volume" in d.columns:
            d["volume"] = v
    return d, recs


def tier_cost_arrays(dvol: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    large = dvol >= TIER_ADV_LARGE_USD
    mid = (dvol >= TIER_ADV_MID_USD) & (~large)
    slip = np.where(large, BRACKET_SLIPPAGE_BPS["LARGE"], np.where(mid, BRACKET_SLIPPAGE_BPS["MID"], BRACKET_SLIPPAGE_BPS["SMALL"]))
    imp = np.where(large, BRACKET_IMPACT_COEF_BPS["LARGE"], np.where(mid, BRACKET_IMPACT_COEF_BPS["MID"], BRACKET_IMPACT_COEF_BPS["SMALL"]))
    code = np.where(large, 0, np.where(mid, 1, 2))
    return slip.astype(np.float64), imp.astype(np.float64), code


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


class VectorIndicatorBank:
    def __init__(self, grid: MarketGrid):
        self.grid = grid
        self._cache: Dict[Tuple[Any, ...], np.ndarray] = {}
        self.started_mask = np.cumsum(grid.alive_mat, axis=1) > 0

    def _masked_close(self) -> np.ndarray:
        return np.where(self.started_mask, self.grid.close_mat, np.nan)

    def sma(self, length: int) -> np.ndarray:
        w = max(2, int(length))
        key = ("sma", w)
        if key in self._cache:
            return self._cache[key]
        mat = self._masked_close()
        valid = ~np.isnan(mat)
        clean = np.where(valid, mat, 0.0)
        csum = np.pad(np.cumsum(clean, axis=1), ((0, 0), (1, 0)), constant_values=0.0)
        ccnt = np.pad(np.cumsum(valid.astype(np.int32), axis=1), ((0, 0), (1, 0)), constant_values=0)
        win_sum = csum[:, w:] - csum[:, :-w]
        win_cnt = ccnt[:, w:] - ccnt[:, :-w]
        out = np.full_like(mat, np.nan, dtype=np.float64)
        out[:, w - 1:] = np.where(win_cnt == w, win_sum / float(w), np.nan)
        self._cache[key] = out
        return out

    def ema(self, span: int) -> np.ndarray:
        span = max(2, int(span))
        key = ("ema", span)
        if key in self._cache:
            return self._cache[key]
        mat = self._masked_close()
        alpha = 2.0 / (span + 1.0)
        n_syms, n_bars = mat.shape
        out = np.full((n_syms, n_bars), np.nan, dtype=np.float64)
        out[:, 0] = mat[:, 0]
        for t in range(1, n_bars):
            prev = out[:, t - 1]
            cur = mat[:, t]
            hp, hc = ~np.isnan(prev), ~np.isnan(cur)
            out[:, t] = np.where(hc & hp, alpha * cur + (1.0 - alpha) * prev, np.where(hc, cur, prev))
        self._cache[key] = out
        return out

    def wma(self, length: int) -> np.ndarray:
        w = max(2, int(length))
        key = ("wma", w)
        if key in self._cache:
            return self._cache[key]
        mat = self._masked_close()
        n_syms, n_bars = mat.shape
        out = np.full((n_syms, n_bars), np.nan, dtype=np.float64)
        if n_bars >= w:
            weights = np.arange(1, w + 1, dtype=np.float64)
            weights /= weights.sum()
            wins = sliding_window_view(mat, window_shape=w, axis=1)
            out[:, w - 1:] = np.tensordot(wins, weights, axes=([2], [0]))
        self._cache[key] = out
        return out

    def ma(self, length: int, kind: int) -> np.ndarray:
        if kind == 0:
            return self.sma(length)
        elif kind == 1:
            return self.ema(length)
        elif kind == 3:
            return self.wma(length)
        raise ValueError(f"Unsupported MA kind: {kind}")

    def rolling_max_high(self, window: int) -> np.ndarray:
        w = max(2, int(window))
        key = ("hhv", w)
        if key in self._cache:
            return self._cache[key]
        h_shift = shift_2d(np.where(self.started_mask, self.grid.high_mat, np.nan))
        n_syms, n_bars = h_shift.shape
        out = np.full((n_syms, n_bars), np.nan, dtype=np.float64)
        if n_bars >= w:
            wins = sliding_window_view(h_shift, window_shape=w, axis=1)
            out[:, w - 1:] = np.max(wins, axis=-1)
        self._cache[key] = out
        return out

    def bollinger(self, length: int, std_mult: float) -> Tuple[np.ndarray, np.ndarray]:
        w = max(2, int(length))
        mid = self.sma(w)
        key_std = ("std", w)
        if key_std not in self._cache:
            mat = self._masked_close()
            n_syms, n_bars = mat.shape
            std_out = np.full((n_syms, n_bars), np.nan, dtype=np.float64)
            if n_bars >= w:
                wins = sliding_window_view(mat, window_shape=w, axis=1)
                std_out[:, w - 1:] = np.std(wins, axis=-1, ddof=0)
            self._cache[key_std] = std_out
        std = self._cache[key_std]
        return mid + (std_mult * std), mid

    def rsi_smoothed(self, length: int, smooth: int) -> np.ndarray:
        length, smooth = max(2, int(length)), max(1, int(smooth))
        key = ("rsi", length, smooth)
        if key in self._cache:
            return self._cache[key]
        c = self._masked_close()
        delta = c - shift_2d(c)
        valid_delta = ~np.isnan(delta)
        gain = np.where(valid_delta, np.maximum(delta, 0.0), np.nan)
        loss = np.where(valid_delta, np.maximum(-delta, 0.0), np.nan)

        span = int(round((2.0 * length) - 1.0))
        alpha = 2.0 / (span + 1.0)
        n_syms, n_bars = c.shape
        avg_g = np.full((n_syms, n_bars), np.nan, dtype=np.float64)
        avg_l = np.full((n_syms, n_bars), np.nan, dtype=np.float64)
        avg_g[:, 0] = gain[:, 0]
        avg_l[:, 0] = loss[:, 0]
        for t in range(1, n_bars):
            pg, pl = avg_g[:, t - 1], avg_l[:, t - 1]
            cg, cl = gain[:, t], loss[:, t]
            hp, hc = ~np.isnan(pg), ~np.isnan(cg)
            avg_g[:, t] = np.where(hc & hp, alpha * cg + (1.0 - alpha) * pg, np.where(hc, cg, pg))
            avg_l[:, t] = np.where(hc & hp, alpha * cl + (1.0 - alpha) * pl, np.where(hc, cl, pl))

        rs = avg_g / (avg_l + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        if smooth > 1:
            valid = ~np.isnan(rsi)
            clean = np.where(valid, rsi, 0.0)
            csum = np.pad(np.cumsum(clean, axis=1), ((0, 0), (1, 0)))
            ccnt = np.pad(np.cumsum(valid.astype(np.int32), axis=1), ((0, 0), (1, 0)))
            out = np.full_like(rsi, np.nan)
            out[:, smooth - 1:] = np.where(
                (ccnt[:, smooth:] - ccnt[:, :-smooth]) == smooth,
                (csum[:, smooth:] - csum[:, :-smooth]) / float(smooth),
                np.nan,
            )
        else:
            out = rsi
        self._cache[key] = out
        return out

    def buy_pressure_mask(self, vol_ma_len: int, vol_mult: float, ratio: float) -> np.ndarray:
        h, l, c, v = self.grid.high_mat, self.grid.low_mat, self.grid.close_mat, self.grid.volume_mat
        rng = h - l
        safe_rng = np.where(rng > 0, rng, 1.0)
        frac = np.where(rng > 0, (c - l) / safe_rng, np.where(c >= shift_2d(c), 1.0, 0.0))
        frac = np.clip(np.nan_to_num(frac, nan=0.0), 0.0, 1.0)
        buy = v * frac
        sell = v - buy

        w = max(2, int(vol_ma_len))
        v_shift = shift_2d(v, fill_value=0.0)
        csum = np.pad(np.cumsum(v_shift, axis=1), ((0, 0), (1, 0)))
        vma = np.full_like(v, np.nan)
        vma[:, w - 1:] = (csum[:, w:] - csum[:, :-w]) / float(w)
        return (~np.isnan(vma)) & (v > 0) & (v > vol_mult * vma) & (buy >= ratio * sell)


def build_market_universe(universe_name: str, start_year: int, refresh: bool,
                          extra_symbols: Optional[Iterable[str]] = None) -> Tuple[MarketGrid, Dict[str, pd.DataFrame], pd.DataFrame, List[Dict[str, Any]]]:
    symbols = list(load_universe_constituents(universe_name, top_n=TOP_N_COINS))
    held_only: Set[str] = set()
    for _s in (extra_symbols or ()):
        _s = str(_s).upper().strip()
        if _s and _s not in symbols:
            symbols.append(_s)
            held_only.add(_s)
    _, macro_df, macro_prov = fetch_single_crypto(MACRO_INDEX_TICKER, start_year, refresh)
    if macro_df is None:
        raise RuntimeError(f"Failed to load crypto macro benchmark: {MACRO_INDEX_TICKER}")

    raw_universe: Dict[str, pd.DataFrame] = {}
    provider_map: Dict[str, str] = {MACRO_INDEX_TICKER: macro_prov}
    fetch_symbols = [s for s in symbols if s != MACRO_INDEX_TICKER]
    with ThreadPoolExecutor(max_workers=PARALLEL_DOWNLOAD_WORKERS) as executor:
        future_map = {executor.submit(fetch_single_crypto, sym, start_year, refresh): sym for sym in fetch_symbols}
        for future in as_completed(future_map):
            sym, df, provider = future.result()
            if df is not None and len(df) >= MIN_HISTORY_DAYS:
                raw_universe[sym] = df
                provider_map[sym] = provider
    if MACRO_INDEX_TICKER in symbols and len(macro_df) >= MIN_HISTORY_DAYS:
        raw_universe[MACRO_INDEX_TICKER] = macro_df.copy()

    dropped: List[str] = []
    for sym_h in list(raw_universe.keys()):
        if sym_h == MACRO_INDEX_TICKER or sym_h in held_only:
            continue
        cl_h = raw_universe[sym_h]["close"].dropna()
        why = ""
        if sym_h in EXCLUDED_SYMBOLS:
            why = "excluded list"
        elif ALLOWED_PROVIDERS is not None and provider_map.get(sym_h, "unknown") not in ALLOWED_PROVIDERS:
            why = f"provider '{provider_map.get(sym_h, 'unknown')}' is not an exchange feed"
        elif len(cl_h) > 30 and float(np.mean(np.abs(cl_h.values / max(1e-12, float(np.median(cl_h.values))) - 1.0) <= PEGGED_BAND)) >= PEGGED_SHARE:
            why = "pegged asset (stablecoin-like price)"
        if why:
            dropped.append(f"{sym_h} ({why})")
            raw_universe.pop(sym_h, None)

    DATA_REPAIR_LOG.clear()
    macro_df, r_m = repair_price_artifacts(macro_df, MACRO_INDEX_TICKER, None, allow_corporate=False)
    mkt_ret = macro_df["close"].pct_change()
    mkt_ret.index = pd.DatetimeIndex(mkt_ret.index).normalize()
    all_recs: List[Dict[str, Any]] = list(r_m)
    for sym_r in list(raw_universe.keys()):
        if sym_r == MACRO_INDEX_TICKER:
            raw_universe[sym_r] = macro_df.copy()
            continue
        raw_universe[sym_r], recs_r = repair_price_artifacts(raw_universe[sym_r], sym_r, mkt_ret)
        all_recs.extend(recs_r)
    DATA_REPAIR_LOG.extend(all_recs)
    try:
        pd.DataFrame(all_recs).to_csv(REPAIRS_CSV_FILE, index=False)
    except Exception:
        pass

    master_dates = pd.DatetimeIndex(sorted(macro_df.index.unique())).normalize()
    macro_df = macro_df.reindex(master_dates).ffill().dropna(subset=["close"])
    bench_close = macro_df["close"].reindex(master_dates).ffill().bfill().values.astype(np.float64)

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
    slip_mat = np.zeros((n_syms, n_bars), dtype=np.float64)
    impact_mat = np.zeros((n_syms, n_bars), dtype=np.float64)
    brackets: List[str] = []

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
        reindexed["high"] = np.maximum(reindexed["high"].ffill(), np.maximum(reindexed["open"], reindexed["close"]))
        reindexed["low"] = np.minimum(reindexed["low"].ffill(), np.minimum(reindexed["open"], reindexed["close"]))
        reindexed["volume"] = reindexed["volume"].fillna(0.0)

        h_arr = reindexed["high"].values
        l_arr = reindexed["low"].values
        c_arr = reindexed["close"].values

        atr14 = FastIndicators.atr_1d(h_arr, l_arr, c_arr, 14)
        adx14 = FastIndicators.adx_1d(h_arr, l_arr, c_arr, 14)

        qv = reindexed["quote_volume"].values if "quote_volume" in reindexed else (c_arr * reindexed["volume"].values)
        dvol30 = np_rolling_median(qv, ADV_WINDOW, ADV_MIN_PERIODS)
        dvol30 = np.nan_to_num(dvol30, nan=0.0)
        s_bps, i_bps, tier_code = tier_cost_arrays(dvol30)

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
        slip_mat[i, :] = s_bps
        impact_mat[i, :] = i_bps
        universe[sym] = reindexed

        live_codes = tier_code[(~is_missing).values & (dvol30 > 0)]
        modal = int(np.bincount(live_codes, minlength=3).argmax()) if len(live_codes) else 2
        brackets.append(("LARGE", "MID", "SMALL")[modal])

        both_alive = (~is_missing).values[1:] & (~is_missing).values[:-1]
        rets_1d = np.where(both_alive, np.abs(c_arr[1:] / np.maximum(1e-9, c_arr[:-1]) - 1.0), 0.0)
        max_move_i = int(np.argmax(rets_1d)) if len(rets_1d) else 0
        provenance_records.append({
            "symbol": sym, "bracket": brackets[i], "provider": provider_map.get(sym, "unknown"),
            "first_date": str(first_idx.date()), "last_date": str(last_idx.date()),
            "total_bars": len(df), "is_delisted": bool(is_delisted_perm[-1]) if len(is_delisted_perm) else False,
            "max_1d_move_pct": float(rets_1d[max_move_i] * 100.0) if len(rets_1d) else 0.0,
            "max_1d_move_date": str(master_dates[max_move_i + 1].date()) if len(rets_1d) else "",
            "median_dvol_musd": float(np.nanmedian(dvol30[dvol30 > 0]) / 1e6) if np.any(dvol30 > 0) else 0.0,
        })

    grid = MarketGrid(
        symbols=valid_symbols, dates=master_dates,
        open_mat=open_mat, high_mat=high_mat, low_mat=low_mat, close_mat=close_mat, volume_mat=vol_mat,
        atr14_mat=atr14_mat, dvol30_mat=dvol30_mat, adx14_mat=adx14_mat,
        delist_mat=delist_mat, alive_mat=alive_mat,
        macro_close=macro_df["close"].values, macro_open=macro_df["open"].values,
        months_arr=master_dates.month.values, years_arr=master_dates.year.values,
        bench_close=bench_close, brackets=brackets,
        liq_floor=np.full(n_syms, LIQUIDITY_FLOOR_USD, dtype=np.float64),
        slip_bps=slip_mat, impact_bps=impact_mat
    )
    return grid, universe, macro_df, provenance_records


def _eval_entry_state_filter(bank: VectorIndicatorBank, grid: MarketGrid, fname: str, prefix: str, p: Dict[str, Any]) -> np.ndarray:
    c = grid.close_mat
    if fname == "NONE":
        return np.ones_like(c, dtype=bool)
    elif fname == "TREND_MA":
        ma = bank.ma(p[f"{prefix}_ma_len"], p[f"{prefix}_ma_type"])
        return (~np.isnan(ma)) & (c > ma)
    elif fname == "RSI_STATE":
        rf = bank.rsi_smoothed(p[f"{prefix}_rsi_f_len"], p[f"{prefix}_rsi_f_smt"])
        rs = bank.rsi_smoothed(p[f"{prefix}_rsi_s_len"], p[f"{prefix}_rsi_s_smt"])
        return (~np.isnan(rf)) & (~np.isnan(rs)) & (rf > rs)
    elif fname == "VOLUME_SURGE":
        w = max(2, int(p[f"{prefix}_vol_ma_len"]))
        v = grid.volume_mat
        v_shift = shift_2d(v, fill_value=0.0)
        csum = np.pad(np.cumsum(v_shift, axis=1), ((0, 0), (1, 0)))
        vma = np.full_like(v, np.nan)
        vma[:, w - 1:] = (csum[:, w:] - csum[:, :-w]) / float(w)
        return (~np.isnan(vma)) & (v > p[f"{prefix}_vol_mult"] * vma)
    elif fname == "BUY_PRESSURE":
        return bank.buy_pressure_mask(p[f"{prefix}_bp_len"], p[f"{prefix}_bp_vol_mult"], p[f"{prefix}_bp_ratio"])
    elif fname == "BB_ABOVE_MID":
        _, b_mid = bank.bollinger(p[f"{prefix}_bb_len"], 2.0)
        return (~np.isnan(b_mid)) & (c > b_mid)
    raise ValueError(f"Unknown entry filter: {fname}")


def _eval_exit_signal(bank: VectorIndicatorBank, grid: MarketGrid, xname: str, prefix: str, p: Dict[str, Any]) -> np.ndarray:
    c, o, v = grid.close_mat, grid.open_mat, grid.volume_mat
    if xname == "NONE":
        return np.zeros_like(c, dtype=bool)
    elif xname == "MA_BREAK":
        ma = bank.ma(p[f"{prefix}_ma_len"], p[f"{prefix}_ma_type"])
        return (~np.isnan(ma)) & (c < ma)
    elif xname == "RSI_CROSS_DOWN":
        rf = bank.rsi_smoothed(p[f"{prefix}_rsi_f_len"], p[f"{prefix}_rsi_f_smt"])
        rs = bank.rsi_smoothed(p[f"{prefix}_rsi_s_len"], p[f"{prefix}_rsi_s_smt"])
        return (~np.isnan(rf)) & (~np.isnan(rs)) & (rf < rs)
    elif xname == "MA_XOVER_DOWN":
        s_ma = bank.ma(p[f"{prefix}_xover_s_len"], p[f"{prefix}_xover_s_type"])
        l_ma = bank.ma(p[f"{prefix}_xover_l_len"], p[f"{prefix}_xover_l_type"])
        return (~np.isnan(s_ma)) & (~np.isnan(l_ma)) & (s_ma < l_ma)
    elif xname == "CLIMAX_VOLUME_RED":
        w = max(2, int(p[f"{prefix}_vol_ma_len"]))
        v_shift = shift_2d(v, fill_value=0.0)
        csum = np.pad(np.cumsum(v_shift, axis=1), ((0, 0), (1, 0)))
        vma = np.full_like(v, np.nan)
        vma[:, w - 1:] = (csum[:, w:] - csum[:, :-w]) / float(w)
        return (~np.isnan(vma)) & (v > p[f"{prefix}_vol_mult"] * vma) & (c < o)
    elif xname == "BB_MID_BREAK":
        _, b_mid = bank.bollinger(p[f"{prefix}_bb_len"], 2.0)
        return (~np.isnan(b_mid)) & (c < b_mid)
    raise ValueError(f"Unknown exit signal: {xname}")


def compile_signals_composable(
    grid: MarketGrid, p: Dict[str, Any], bank: Optional[VectorIndicatorBank] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if bank is None:
        bank = VectorIndicatorBank(grid)

    c, o, v = grid.close_mat, grid.open_mat, grid.volume_mat
    n_bars = c.shape[1]

    if p.get("use_market_macro_system", False):
        macro_ma = FastIndicators.moving_average(grid.macro_close, p["macro_ma_len"], p["macro_ma_type"])
        macro_ok = (~np.isnan(macro_ma)) & (grid.macro_close > macro_ma)
    else:
        macro_ok = np.ones(n_bars, dtype=bool)

    et = p["entry_type"]
    if et == 0:
        ma = bank.ma(p["entry_ma_len"], p["entry_ma_type"])
        raw_entry = (~np.isnan(ma)) & (c > ma)
        state_entry = raw_entry
    elif et == 1:
        rf = bank.rsi_smoothed(p["rsi_f_len"], p["rsi_f_smt"])
        rs = bank.rsi_smoothed(p["rsi_s_len"], p["rsi_s_smt"])
        raw_entry = (~np.isnan(rf)) & (~np.isnan(rs)) & (rf > rs) & (shift_2d(rf) <= shift_2d(rs))
        if p.get("use_rsi_trend_filter", False):
            rma = bank.ma(p["rsi_trend_ma_len"], p["rsi_trend_ma_type"])
            raw_entry = raw_entry & (~np.isnan(rma)) & (c > rma)
            state_entry = (~np.isnan(rf)) & (~np.isnan(rs)) & (rf > rs) & (~np.isnan(rma)) & (c > rma)
        else:
            state_entry = (~np.isnan(rf)) & (~np.isnan(rs)) & (rf > rs)
    elif et == 2:
        s_ma = bank.ma(p["xover_short_len"], p["xover_short_type"])
        l_ma = bank.ma(p["xover_long_len"], p["xover_long_type"])
        raw_entry = (~np.isnan(s_ma)) & (~np.isnan(l_ma)) & (s_ma > l_ma) & (shift_2d(s_ma) <= shift_2d(l_ma))
        state_entry = (~np.isnan(s_ma)) & (~np.isnan(l_ma)) & (s_ma > l_ma)
    elif et == 3:
        hhv = bank.rolling_max_high(int(p["price_lookback"]))
        raw_entry = (~np.isnan(hhv)) & (c > hhv) & ((c - o) >= (p["body_atr_mult"] * grid.atr14_mat))
        if float(p.get("vol_mult", 0.0)) > 0.0:
            w = max(2, int(p["vol_ma_len"]))
            v_shift = shift_2d(v, fill_value=0.0)
            csum = np.pad(np.cumsum(v_shift, axis=1), ((0, 0), (1, 0)))
            vma = np.full_like(v, np.nan)
            vma[:, w - 1:] = (csum[:, w:] - csum[:, :-w]) / float(w)
            raw_entry = raw_entry & (~np.isnan(vma)) & (v > p["vol_mult"] * vma)
        base_ma = bank.sma(int(p["price_lookback"]))
        state_entry = (~np.isnan(base_ma)) & (c > base_ma)
    elif et == 4:
        b_up, b_mid = bank.bollinger(p["bb_entry_len"], p["bb_entry_std"])
        raw_entry = (~np.isnan(b_up)) & (c > b_up) & (shift_2d(c) <= shift_2d(b_up))
        state_entry = (~np.isnan(b_mid)) & (c > b_mid)
    elif et == 5:
        raw_entry = bank.buy_pressure_mask(p["bp_vol_ma_len"], p["bp_vol_mult"], p["bp_ratio"])
        state_entry = np.ones_like(c, dtype=bool)
    elif et == 6:
        ma = bank.ma(p["entry_ma_len"], p["entry_ma_type"])
        above = (~np.isnan(ma)) & (c > ma)
        raw_entry = above & ((o <= ma) | (shift_2d(c) < shift_2d(ma)))
        state_entry = above
    else:
        raise ValueError(f"Unsupported entry_type: {et}")

    f1_mask = _eval_entry_state_filter(bank, grid, p.get("entry_filter_1", "NONE"), "ef1", p)
    f2_mask = _eval_entry_state_filter(bank, grid, p.get("entry_filter_2", "NONE"), "ef2", p)
    combined_raw = raw_entry & f1_mask & f2_mask
    combined_state = state_entry & f1_mask & f2_mask

    liq_ok = grid.dvol30_mat >= np.maximum(grid.liq_floor, 1.0)[:, None]
    adx_t = float(p.get("adx_thresh", 0.0))
    adx_ok = (grid.adx14_mat >= adx_t) if adx_t > 0.0 else True

    raw_signal_mat = combined_raw & grid.alive_mat
    entry_mat = raw_signal_mat & liq_ok & macro_ok[None, :] & adx_ok
    state_mat = combined_state & liq_ok & macro_ok[None, :] & adx_ok & grid.alive_mat
    episode_onset_mat = raw_signal_mat & (~shift_2d(raw_signal_mat, fill_value=False))

    if "exit_primary" in p or "exit_secondary" in p:
        x1_mask = _eval_exit_signal(bank, grid, p.get("exit_primary", "NONE"), "ex1", p)
        x2_mask = _eval_exit_signal(bank, grid, p.get("exit_secondary", "NONE"), "ex2", p)
        exit_mat = x1_mask | x2_mask
    else:
        xt = p.get("exit_type", 0)
        if xt == 3:
            ma_val = bank.ma(p["exit_ma_len"], p["exit_ma_type"])
            exit_mat = (~np.isnan(ma_val)) & (c < ma_val)
        elif xt == 4:
            rf = bank.rsi_smoothed(p["exit_rsi_f_len"], p["exit_rsi_f_smt"])
            rs = bank.rsi_smoothed(p["exit_rsi_s_len"], p["exit_rsi_s_smt"])
            exit_mat = (~np.isnan(rf)) & (~np.isnan(rs)) & (rf < rs)
        elif xt == 5:
            s_ma = bank.ma(p["exit_xover_short_len"], p["exit_xover_short_type"])
            l_ma = bank.ma(p["exit_xover_long_len"], p["exit_xover_long_type"])
            exit_mat = (~np.isnan(s_ma)) & (~np.isnan(l_ma)) & (s_ma < l_ma)
        elif xt == 6:
            w = max(2, int(p["exit_vol_ma_len"]))
            v_shift = shift_2d(v, fill_value=0.0)
            csum = np.pad(np.cumsum(v_shift, axis=1), ((0, 0), (1, 0)))
            vma = np.full_like(v, np.nan)
            vma[:, w - 1:] = (csum[:, w:] - csum[:, :-w]) / float(w)
            exit_mat = (~np.isnan(vma)) & (v > p["exit_vol_mult"] * vma) & (c < o)
        elif xt == 7:
            _, b_mid = bank.bollinger(p["bb_exit_len"], 2.0)
            exit_mat = (~np.isnan(b_mid)) & (c < b_mid)
        else:
            exit_mat = np.zeros_like(c, dtype=bool)

    return raw_signal_mat, entry_mat, exit_mat, macro_ok, state_mat, episode_onset_mat


@dataclass
class Tranche:
    coin: str
    coin_idx: int
    entry_bar: int
    entry_date: pd.Timestamp
    entry_price: float
    initial_units: float
    units: float
    cost_usd: float
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


def slip_fraction(grid: MarketGrid, c_i: int, part_rate: float, bar: int, mult: float = 1.0) -> float:
    b = max(0, bar)
    return SLIPPAGE_SCALE * mult * (grid.slip_bps[c_i, b] + grid.impact_bps[c_i, b] * np.sqrt(part_rate)) / 10000.0


def _resolve_trail_mode(p: Dict[str, Any]) -> Tuple[str, float, float]:
    if "trail_mode" in p:
        tm = str(p["trail_mode"])
        if tm == "PCT" and float(p.get("trail_pct", 0.0)) > 0.0:
            return "PCT", 1.0 - (float(p["trail_pct"]) / 100.0), 0.0
        elif tm == "ATR" and float(p.get("trail_atr_mult", 0.0)) > 0.0:
            return "ATR", 0.0, float(p["trail_atr_mult"])
        return "NONE", 0.0, 0.0
    if p.get("exit_type") == 1 and float(p.get("trail_pct", 0.0)) > 0.0:
        return "PCT", 1.0 - (float(p["trail_pct"]) / 100.0), 0.0
    if float(p.get("trail_atr_mult", 0.0)) > 0.0:
        return "ATR", 0.0, float(p["trail_atr_mult"])
    return "NONE", 0.0, 0.0


def make_tranche(grid: MarketGrid, c_i: int, t: int, fill_px: float, units: float, total_cost: float, fee_buy: FeeBreakdown,
                 slip_mult: float, entry_atr: float, sl_mult: float, from_watchlist: bool = False, wait_days: int = 0,
                 trigger_px: float = 0.0, entry_equity: float = 0.0) -> Tranche:
    sl_price = fill_px - (sl_mult * entry_atr)
    return Tranche(
        coin=grid.symbols[c_i], coin_idx=c_i, entry_bar=t, entry_date=grid.dates[t],
        entry_price=fill_px, initial_units=units, units=units, cost_usd=total_cost,
        entry_atr=entry_atr, current_sl=sl_price, stop_reason="STOP_LOSS",
        highest_high=max(fill_px, grid.high_mat[c_i, t]), lowest_low=min(fill_px, grid.low_mat[c_i, t]),
        peak_bar=t, be_price=breakeven_exit_price(total_cost, units, slip_mult),
        pending_stop=bool(grid.close_mat[c_i, t] <= sl_price), pending_reason="STOP_LOSS",
        from_watchlist=from_watchlist, wait_days=wait_days, trigger_px=trigger_px,
        entry_equity=entry_equity, proceeds=0.0, fee_acc=fee_buy
    )


def tranche_open_exit_reason(tr: Tranche, t: int, grid: MarketGrid, exit_mat: np.ndarray, macro_bear_confirmed: bool,
                             has_signal_exit: bool, max_holding_bars: int, exit_label: str = "SIGNAL_EXIT") -> str:
    bars_held = t - tr.entry_bar
    if tr.pending_stop:
        return tr.pending_reason
    if macro_bear_confirmed:
        return "MACRO_REGIME_EXIT"
    if bars_held >= max_holding_bars:
        return "MAX_HOLDING_TIME"
    if has_signal_exit and exit_mat[tr.coin_idx, t - 1]:
        if bars_held >= 3 or (tr.highest_high - tr.entry_price) >= (1.0 * tr.entry_atr):
            return exit_label
    return ""


def tranche_eod_update(tr: Tranche, t: int, grid: MarketGrid, trail_mode: str, be_trigger_mult: float,
                       trail_pct_mult: float, trail_mult: float) -> None:
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
    if trail_mode == "PCT" and trail_pct_mult > 0.0:
        pct_floor = tr.highest_high * trail_pct_mult
        if pct_floor > tr.current_sl:
            tr.current_sl = pct_floor
            tr.stop_reason = "TRAIL_PCT_STOP"
    elif trail_mode == "ATR" and trail_mult > 0.0:
        atr_now = grid.atr14_mat[c_i, t] if grid.atr14_mat[c_i, t] > 0 else tr.entry_atr
        atr_floor = tr.highest_high - (trail_mult * atr_now)
        if atr_floor > tr.current_sl:
            tr.current_sl = atr_floor
            tr.stop_reason = "TRAIL_ATR_STOP"


def evaluate_entry(grid: MarketGrid, c_i: int, t: int, today_open: float, entry_atr: float, cash: float,
                   live_tranches: List[Tranche], day_equity: float, day_target: float, floor_usd: float,
                   free_adds: int, max_pyramid: int, whole_shares: bool, entry_mult: float = 1.0,
                   pyr_rule: int = 0, pyr_gap_bars: int = 0, pyr_progress_atr: float = 0.0) -> Dict[str, Any]:
    coin_layers = sum(1 for tr in live_tranches if tr.coin_idx == c_i)
    if coin_layers >= max_pyramid:
        return {"ok": False, "reason": "pyramid_blocked", "starved": False}
    if coin_layers > free_adds:
        highest_prior_entry = max(tr.entry_price for tr in live_tranches if tr.coin_idx == c_i)
        if today_open <= highest_prior_entry * 1.005:
            return {"ok": False, "reason": "anti_averaging_down", "starved": False}
    if coin_layers >= 1 and pyr_rule:
        last_tr = max((tr for tr in live_tranches if tr.coin_idx == c_i), key=lambda tr: tr.entry_bar)
        if pyr_rule == 1 and (t - last_tr.entry_bar) < pyr_gap_bars:
            return {"ok": False, "reason": "pyramid_blocked", "starved": False}
        if pyr_rule == 2 and today_open < last_tr.entry_price + pyr_progress_atr * last_tr.entry_atr:
            return {"ok": False, "reason": "pyramid_blocked", "starved": False}
    if cash < floor_usd:
        return {"ok": False, "reason": "cash_starved", "starved": True}

    max_pos_cap = day_equity * MAX_POSITION_EQUITY_PCT
    dynamic_slot_target = min(max_pos_cap, day_target)
    current_scrip_exposure = sum(tr.units * today_open for tr in live_tranches if tr.coin_idx == c_i)
    remaining_scrip_capacity = max(0.0, max_pos_cap - current_scrip_exposure)
    if remaining_scrip_capacity < floor_usd:
        return {"ok": False, "reason": "scrip_risk_cap", "starved": False}

    adv_30d = max(grid.dvol30_mat[c_i, t - 1], LIQUIDITY_FLOOR_USD)
    liquidity_cap_usd = adv_30d * MAX_ADV_PARTICIPATION
    target_usd = min(dynamic_slot_target, liquidity_cap_usd, remaining_scrip_capacity)
    max_affordable = compute_max_affordable_tranche(
        cash, adv_30d,
        float(SLIPPAGE_SCALE * entry_mult * grid.slip_bps[c_i, t - 1]),
        float(SLIPPAGE_SCALE * entry_mult * grid.impact_bps[c_i, t - 1]),
    )
    tranche_usd = min(max_affordable, max(floor_usd, target_usd))
    runt = tranche_usd < SIZE_RUNT_FRACTION * max(floor_usd, target_usd)
    slip_mult = slip_fraction(grid, c_i, min(1.0, max(0.0, tranche_usd / adv_30d)), t - 1, entry_mult)
    fill_px, units, total_cost, fee_buy = _buy_fill_audited(tranche_usd, today_open, slip_mult, whole_shares)

    if whole_shares and units == 1.0 and total_cost > min(cash, remaining_scrip_capacity):
        return {"ok": False, "reason": "unaffordable", "starved": False}

    if cash >= total_cost and tranche_usd >= floor_usd and units > 0 and not runt:
        if max_affordable < target_usd:
            binding = "cash_constrained"
        elif target_usd == liquidity_cap_usd:
            binding = "liquidity_cap"
        elif target_usd < floor_usd:
            binding = "tranche_floor"
        else:
            binding = "equity_slot"
        return {
            "ok": True, "binding": binding, "tranche_usd": tranche_usd, "fill_px": fill_px, "units": units,
            "total_cost": total_cost, "fee_buy": fee_buy, "slip_mult": slip_mult
        }
    if units <= 0:
        return {"ok": False, "reason": "unaffordable", "starved": False}
    return {"ok": False, "reason": "cash_starved", "starved": True}


def reconstitute_params(params: Dict[str, Any]) -> Dict[str, Any]:
    p = copy.deepcopy(params)
    for k in list(p.keys()):
        if isinstance(p[k], float):
            p[k] = round(p[k], 2)

    if "exit_primary" not in p and "exit_type" in p:
        xt = int(p["exit_type"])
        p["use_exit_secondary"] = False
        p["exit_secondary"] = "NONE"
        if xt == 3:
            p["use_exit_primary"] = True
            p["exit_primary"] = "MA_BREAK"
            p["ex1_ma_len"] = int(p.get("exit_ma_len", 50))
            p["ex1_ma_type"] = int(p.get("exit_ma_type", 0))
        elif xt == 4:
            p["use_exit_primary"] = True
            p["exit_primary"] = "RSI_CROSS_DOWN"
            p["ex1_rsi_f_len"] = int(p.get("exit_rsi_f_len", 14))
            p["ex1_rsi_f_smt"] = int(p.get("exit_rsi_f_smt", 6))
            p["ex1_rsi_s_len"] = int(p.get("exit_s_len", 40))
            p["ex1_rsi_s_smt"] = int(p.get("exit_rsi_s_smt", 10))
        elif xt == 5:
            p["use_exit_primary"] = True
            p["exit_primary"] = "MA_XOVER_DOWN"
            s_len = int(p.get("exit_xover_short_len", 15))
            gap = int(p.get("exit_xover_gap", 30))
            p["ex1_xover_s_len"] = s_len
            p["ex1_xover_s_type"] = int(p.get("exit_xover_short_type", 0))
            p["ex1_xover_gap"] = gap
            p["ex1_xover_l_len"] = int(p.get("exit_xover_long_len", min(150, s_len + gap)))
            p["ex1_xover_l_type"] = int(p.get("exit_xover_long_type", 0))
        elif xt == 6:
            p["use_exit_primary"] = True
            p["exit_primary"] = "CLIMAX_VOLUME_RED"
            p["ex1_vol_ma_len"] = int(p.get("exit_vol_ma_len", 20))
            p["ex1_vol_mult"] = float(p.get("exit_vol_mult", 2.0))
        elif xt == 7:
            p["use_exit_primary"] = True
            p["exit_primary"] = "BB_MID_BREAK"
            p["ex1_bb_len"] = int(p.get("bb_exit_len", 20))
        else:
            p["use_exit_primary"] = False
            p["exit_primary"] = "NONE"

    if "trail_mode" not in p:
        if p.get("exit_type") == 1 and float(p.get("trail_pct", 0.0)) > 0.0:
            p["trail_mode"] = "PCT"
        elif float(p.get("trail_atr_mult", 0.0)) > 0.0:
            p["trail_mode"] = "ATR"
        else:
            p["trail_mode"] = "NONE"

    if not p.get("use_entry_filter_1", "entry_filter_1" in p and p["entry_filter_1"] != "NONE"):
        p["entry_filter_1"] = "NONE"
        p["entry_filter_2"] = "NONE"
    elif p.get("entry_filter_1", "NONE") == "NONE":
        p["entry_filter_1"] = "NONE"
        p["entry_filter_2"] = "NONE"
    elif not p.get("use_entry_filter_2", "entry_filter_2" in p and p["entry_filter_2"] != "NONE") or p.get("entry_filter_2") == p.get("entry_filter_1"):
        p["entry_filter_2"] = "NONE"

    if not p.get("use_exit_primary", "exit_primary" in p and p["exit_primary"] != "NONE"):
        p["exit_primary"] = "NONE"
        p["exit_secondary"] = "NONE"
    elif p.get("exit_primary", "NONE") == "NONE":
        p["exit_primary"] = "NONE"
        p["exit_secondary"] = "NONE"
    elif not p.get("use_exit_secondary", "exit_secondary" in p and p["exit_secondary"] != "NONE") or p.get("exit_secondary") == p.get("exit_primary"):
        p["exit_secondary"] = "NONE"

    if p.get("entry_type") == 2 and "xover_gap" in p:
        p["xover_long_len"] = min(200, p["xover_short_len"] + p["xover_gap"])
    if "ex1_xover_s_len" in p and "ex1_xover_gap" in p:
        p["ex1_xover_l_len"] = min(150, p["ex1_xover_s_len"] + p["ex1_xover_gap"])
    if "ex2_xover_s_len" in p and "ex2_xover_gap" in p:
        p["ex2_xover_l_len"] = min(150, p["ex2_xover_s_len"] + p["ex2_xover_gap"])

    p.setdefault("entry_filter_1", "NONE")
    p.setdefault("entry_filter_2", "NONE")
    p.setdefault("exit_primary", "NONE")
    p.setdefault("exit_secondary", "NONE")
    p.setdefault("trail_mode", "NONE")
    p.setdefault("use_market_macro_system", False if FORCE_MACRO_REGIME_FILTER is None else FORCE_MACRO_REGIME_FILTER)
    p.setdefault("macro_active_exit", False)
    p.setdefault("max_pyramid_layers", 1)
    p.setdefault("adx_thresh", 0.0)
    p.setdefault("be_trigger_atr", 0.0)
    p.setdefault("trail_atr_mult", 0.0)
    p.setdefault("trail_pct", 0.0)
    p.setdefault("wl_mode", "WL_NONE")
    p.setdefault("tranche_floor", TRANCHE_FLOOR_USD)
    p.update(V13_FIXED_RULES)
    p.setdefault("avg_down_free_adds", 0)
    p.setdefault("pyramid_rule", 0)
    p.setdefault("pyramid_gap_bars", 0)
    p.setdefault("pyramid_progress_atr", 0.0)
    if p.get("entry_type") == 3 and not p.get("breakout_use_volume", True):
        p["vol_mult"] = 0.0
    for legacy in ("use_global_tp", "tp_mult", "tp_size_pct", "tp_move_sl_be"):
        p.pop(legacy, None)
    return p


bt = sys.modules[__name__]
bt.DATA_DIR.mkdir(parents=True, exist_ok=True)
bt.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 4. CONFIG & RULES
# ==============================================================================
LIVE_ENGINE_VERSION = bt.ENGINE_VERSION


@dataclass
class Rules:
    """Everything the engine rules read from the winner config, resolved once."""
    p: Dict[str, Any]
    trail_mode: str
    trail_mult: float
    trail_pct_mult: float
    be_trigger_mult: float
    macro_active_exit: bool
    max_holding_bars: int
    sl_mult: float
    floor_usd: float
    max_pyramid: int
    free_adds: int
    pyr_rule: int
    pyr_gap_bars: int
    pyr_progress_atr: float
    whole_shares: bool
    entry_mult: float
    has_signal_exit: bool
    exit_label: str
    wl_mode: str
    wl_dedupe: bool
    wl_max_age: int
    wl_revalidate: bool
    adx_t: float
    size_w0: float
    size_w_max: float
    size_reserve: float
    size_ramp: float

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "Rules":
        p = bt.reconstitute_params(dict(params))
        c = {**bt.ENGINE_DEFAULTS, **p}
        t_mode, t_pct_mult, t_atr_mult = bt._resolve_trail_mode(p)
        if "exit_primary" in p or "exit_secondary" in p:
            x1, x2 = p.get("exit_primary", "NONE"), p.get("exit_secondary", "NONE")
            active_exits = [x for x in (x1, x2) if x != "NONE"]
            has_sig_exit = bool(active_exits)
            ex_label = "SIGNAL_EXIT_" + "+".join(active_exits) if active_exits else "SIGNAL_EXIT"
        else:
            xt = int(p.get("exit_type", 0))
            has_sig_exit = xt in (3, 4, 5, 6, 7)
            ex_label = f"SIGNAL_EXIT_TYPE_{xt}"

        is_surge_entry = (
            int(p.get("entry_type", 0)) in (3, 5)
            or p.get("entry_filter_1", "NONE") in ("VOLUME_SURGE", "BUY_PRESSURE")
            or p.get("entry_filter_2", "NONE") in ("VOLUME_SURGE", "BUY_PRESSURE")
        )
        return cls(
            p=p,
            trail_mode=t_mode,
            trail_mult=t_atr_mult,
            trail_pct_mult=t_pct_mult,
            be_trigger_mult=float(p.get("be_trigger_atr", 0.0)),
            macro_active_exit=bool(p.get("macro_active_exit", False)),
            max_holding_bars=int(p.get("max_holding_bars", 20)),
            sl_mult=float(p.get("sl_mult", 3.0)),
            floor_usd=float(p.get("tranche_floor", bt.TRANCHE_FLOOR_USD)),
            max_pyramid=int(p.get("max_pyramid_layers", 1)),
            free_adds=int(c["avg_down_free_adds"]),
            pyr_rule=int(c["pyramid_rule"]),
            pyr_gap_bars=int(c["pyramid_gap_bars"]),
            pyr_progress_atr=float(c["pyramid_progress_atr"]),
            whole_shares=bool(c["whole_shares"]),
            entry_mult=(bt.VOL_BREAKOUT_SLIP_MULT if is_surge_entry else 1.0),
            has_signal_exit=has_sig_exit,
            exit_label=ex_label,
            wl_mode=str(p.get("wl_mode", "WL_NONE")),
            wl_dedupe=bool(c["wl_dedupe"]),
            wl_max_age=int(c["wl_max_age_bars"]),
            wl_revalidate=bool(c["wl_revalidate"]),
            adx_t=float(p.get("adx_thresh", 0.0)),
            size_w0=float(c["size_w0"]),
            size_w_max=float(c["size_w_max"]),
            size_reserve=float(c["size_reserve"]),
            size_ramp=float(c["size_ramp"]),
        )


def load_config() -> Tuple[Rules, Dict[str, Any]]:
    """Winner config, re-read on EVERY run. winner.json (promoted) is preferred over winner_candidate.json."""
    if LIVE_CONFIG_OVERRIDE:
        path = Path(LIVE_CONFIG_OVERRIDE)
    elif (WORK_DIR / "winner.json").exists():
        path = WORK_DIR / "winner.json"
    elif (WORK_DIR / "winner_candidate.json").exists():
        path = WORK_DIR / "winner_candidate.json"
    else:
        sys.exit(f"No winner config found. Copy winner.json or winner_candidate.json from the backtester's output folder into {WORK_DIR}")
    doc = json.load(open(path, encoding="utf-8"))
    status = str(doc.get("status", "UNKNOWN"))
    if doc.get("version") != LIVE_ENGINE_VERSION:
        sys.exit(f"Config {path.name} was produced by {doc.get('version')}, this live bot is {LIVE_ENGINE_VERSION}. Re-run the backtester.")
    if "NOT PROMOTED" in status.upper() and not ALLOW_NOT_PROMOTED_CONFIG:
        sys.exit(f"Config {path.name} is {status} and ALLOW_NOT_PROMOTED_CONFIG is False.")
    content = hashlib.sha256(json.dumps(doc["best_params"], sort_keys=True, default=str).encode()).hexdigest()[:8]
    meta = {
        "path": str(path), "status": status, "param_hash": content,
        "file_hash": doc.get("param_hash"), "updated_at": doc.get("updated_at"), "version": doc.get("version"),
    }
    return Rules.from_params(doc["best_params"]), meta


# ==============================================================================
# 5. STATE (facts only) AND SMALL HELPERS
# ==============================================================================
def new_state(last_bar_date: str, rules: Rules, cfg_hash: str) -> Dict[str, Any]:
    return {
        "schema": 2, "config_hash": cfg_hash, "last_bar_date": last_bar_date,
        "cash": float(STARTING_CASH_INR), "fx_account": USDINR, "size_w": rules.size_w0,
        "positions": [], "watchlist": [], "plan": None, "closed_trades": [],
        "applied_manual_ids": [], "applied_cash_ids": [], "applied_remove_ids": [], "next_id": 1,
    }


def load_state() -> Optional[Dict[str, Any]]:
    f = STATE_DIR / "live_state_v14.json"
    if not f.exists():
        return None
    st = json.load(open(f, encoding="utf-8"))
    if st.get("schema") != 2:
        sys.exit("Unknown state schema; refusing to continue.")
    return st


def save_state(st: Dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    f = STATE_DIR / "live_state_v14.json"
    if f.exists():
        shutil.copy2(f, STATE_DIR / "live_state_v14.json.bak")
    tmp = STATE_DIR / "live_state_v14.json.tmp"
    json.dump(st, open(tmp, "w", encoding="utf-8"), indent=1, default=str)
    os.replace(tmp, f)


def pos_fx(st: Dict[str, Any], pos: Dict[str, Any]) -> float:
    return float(pos.get("fx_lock") or st.get("fx_account") or USDINR)


def sym_index(grid: MarketGrid) -> Dict[str, int]:
    return {s: i for i, s in enumerate(grid.symbols)}


def bar_index(grid: MarketGrid, date_str: str) -> int:
    return int(grid.dates.searchsorted(pd.Timestamp(date_str)))


def entry_atr_of(grid: MarketGrid, pos: Dict[str, Any], c_i: int, e_bar: int) -> float:
    return float(pos["entry_atr"]) if pos.get("entry_atr") else float(grid.atr14_mat[c_i, max(e_bar - 1, 0)])


def _proxy_tranche(grid: MarketGrid, rules: Rules, pos: Dict[str, Any], c_i: int):
    e = min(bar_index(grid, pos["entry_date"]), len(grid.dates) - 1)
    return bt.make_tranche(
        grid, c_i, e, float(pos["entry_price"]), float(pos["units"]), float(pos["cost_usd"]),
        bt.FeeBreakdown(), 0.0, entry_atr_of(grid, pos, c_i, e), rules.sl_mult
    )


# ==============================================================================
# 6. POSITION REPLAY, PLAN APPLICATION, WATCHLIST & SIZING
# ==============================================================================
def macro_bear(rules: Rules, macro_ok: np.ndarray, t: int) -> bool:
    return bool(rules.macro_active_exit and t >= 3 and (not macro_ok[t - 1]) and (not macro_ok[t - 2]) and (not macro_ok[t - 3]))


def replay_position(grid: MarketGrid, rules: Rules, sig, pos: Dict[str, Any], n: int) -> Dict[str, Any]:
    """Replays bars entry+1 .. n-1 with recovery-aware stop handling, then evaluates bar n."""
    _, _, exit_mat, macro_ok, _, _ = sig
    idx = sym_index(grid)
    out: Dict[str, Any] = {"pos": pos, "tranche": None, "verdict": "", "overdue_since": None, "note": ""}
    if pos["coin"] not in idx:
        out["note"] = "NO PRICE DATA - manage manually"
        return out
    c_i = idx[pos["coin"]]
    e_bar = bar_index(grid, pos["entry_date"])
    if e_bar >= n:
        out["note"] = "entered after the last completed bar - nothing to evaluate yet"
        return out
    entry_px = float(pos["entry_price"])
    units = float(pos["units"])
    adv = max(float(grid.dvol30_mat[c_i, max(e_bar - 1, 0)]), bt.LIQUIDITY_FLOOR_USD)
    slip = bt.slip_fraction(grid, c_i, min(1.0, float(pos["cost_usd"]) / adv), max(e_bar - 1, 0))
    tr = bt.make_tranche(
        grid, c_i, e_bar, entry_px, units, float(pos["cost_usd"]), bt.FeeBreakdown(), slip,
        entry_atr_of(grid, pos, c_i, e_bar), rules.sl_mult
    )
    out["tranche"] = tr
    for t in range(e_bar + 1, n):
        if not grid.alive_mat[c_i, t]:
            continue
        reason = bt.tranche_open_exit_reason(
            tr, t, grid, exit_mat, macro_bear(rules, macro_ok, t),
            rules.has_signal_exit, rules.max_holding_bars, rules.exit_label
        )
        if reason and out["overdue_since"] is None:
            out["overdue_since"] = str(grid.dates[t].date())
        tr.pending_stop = False
        tr.pending_reason = ""
        bt.tranche_eod_update(
            tr, t, grid, rules.trail_mode, rules.be_trigger_mult, rules.trail_pct_mult, rules.trail_mult
        )

    out["verdict"] = bt.tranche_open_exit_reason(
        tr, n, grid, exit_mat, macro_bear(rules, macro_ok, n),
        rules.has_signal_exit, rules.max_holding_bars, rules.exit_label
    )
    if not out["verdict"]:
        out["overdue_since"] = None
    return out


def apply_plan(st: Dict[str, Any], grid: MarketGrid, rules: Rules, t: int) -> List[str]:
    notes: List[str] = []
    plan = st.get("plan")
    if not plan:
        return notes
    idx = sym_index(grid)
    cash = float(st["cash"])
    for o in [o for o in plan["orders"] if o["side"] == "SELL"]:
        if o["id"] in ORDER_SKIPS:
            notes.append(f"{o['id']}: skipped by you (still held)")
            continue
        pos = next((p for p in st["positions"] if p["id"] == o["pos_id"]), None)
        if pos is None or pos["coin"] not in idx:
            continue
        c_i = idx[pos["coin"]]
        raw_px = float(grid.open_mat[c_i, t])
        adv = max(float(grid.dvol30_mat[c_i, t - 1]), bt.LIQUIDITY_FLOOR_USD)
        slip = bt.slip_fraction(grid, c_i, min(1.0, max(0.0, pos["units"] * raw_px / adv)), t - 1)
        fill_px, proceeds, _ = bt._sell_fill_audited(float(pos["units"]), raw_px, slip)
        cash += proceeds
        lock = pos_fx(st, pos)
        st["closed_trades"].append({
            "coin": pos["coin"], "entry_date": pos["entry_date"], "exit_date": str(grid.dates[t].date()),
            "entry_price": pos["entry_price"], "exit_price": round(fill_px, 4), "units": pos["units"],
            "pnl": round(proceeds - pos["cost_usd"], 2), "reason": o["reason"], "fx_lock": lock,
        })
        st["positions"] = [p for p in st["positions"] if p["id"] != pos["id"]]
        notes.append(f"SOLD {pos['coin']} x{pos['units']:g} at open {fill_px:.2f} ({o['reason']})")

    live = [_proxy_tranche(grid, rules, p, idx[p["coin"]]) for p in st["positions"] if p["coin"] in idx]
    day_equity = cash + sum(tr.units * grid.open_mat[tr.coin_idx, t] for tr in live)
    day_target = max(rules.floor_usd, day_equity * st["size_w"])
    for o in [o for o in plan["orders"] if o["side"] == "BUY"]:
        if o["id"] in ORDER_SKIPS:
            notes.append(f"{o['id']}: skipped by you (not bought)")
            continue
        c_i = idx[o["coin"]]
        if not grid.alive_mat[c_i, t]:
            notes.append(f"BUY {o['coin']} not filled: no trading at the open")
            continue
        dec = bt.evaluate_entry(
            grid, c_i, t, float(grid.open_mat[c_i, t]), float(o["entry_atr"]), cash, live,
            day_equity, day_target, rules.floor_usd, rules.free_adds, rules.max_pyramid,
            rules.whole_shares, rules.entry_mult, rules.pyr_rule, rules.pyr_gap_bars, rules.pyr_progress_atr
        )
        if not dec["ok"]:
            notes.append(f"BUY {o['coin']} not filled at the actual open ({dec['reason']})")
            continue
        cash -= dec["total_cost"]
        fx = float(o.get("fx") or st.get("fx_account") or USDINR)
        pos = {
            "id": f"P{st['next_id']}", "coin": o["coin"], "entry_date": str(grid.dates[t].date()),
            "entry_price": round(dec["fill_px"], 4), "units": dec["units"], "cost_usd": round(dec["total_cost"], 2),
            "entry_atr": float(o["entry_atr"]), "source": "bot", "fx_lock": fx,
            "entry_price_inr": round(dec["fill_px"] * fx, 4),
        }
        st["next_id"] += 1
        st["positions"].append(pos)
        live.append(_proxy_tranche(grid, rules, pos, c_i))
        notes.append(f"BOUGHT {o['coin']} x{dec['units']:g} at open {dec['fill_px']:.2f}")
    st["cash"] = cash
    for it in st["watchlist"]:
        if it["signal_date"] == str(grid.dates[t].date()) and it["coin"] in idx:
            o_t = float(grid.open_mat[idx[it["coin"]], t])
            it["highest_high"], it["shadow_stop"] = o_t, o_t - rules.sl_mult * it["entry_atr"]
    return notes


def watchlist_lifecycle(st: Dict[str, Any], grid: MarketGrid, rules: Rules, sig, t: int) -> None:
    if rules.wl_mode == "WL_NONE":
        st["watchlist"] = []
        return
    idx = sym_index(grid)
    exit_mat = sig[2]
    keep = []
    for it in st["watchlist"]:
        if it["coin"] not in idx:
            continue
        c_i = idx[it["coin"]]
        sig_bar = bar_index(grid, it["signal_date"])
        stopped = grid.close_mat[c_i, t] <= it["shadow_stop"]
        exited = rules.has_signal_exit and bool(exit_mat[c_i, t])
        timeout = (t - sig_bar) >= rules.max_holding_bars
        expired = rules.wl_max_age > 0 and (t - sig_bar) >= rules.wl_max_age
        if stopped or exited or timeout or expired:
            continue
        if grid.high_mat[c_i, t] > it["highest_high"]:
            it["highest_high"] = float(grid.high_mat[c_i, t])
        if rules.trail_mode == "PCT" and rules.trail_pct_mult > 0.0:
            it["shadow_stop"] = max(it["shadow_stop"], it["highest_high"] * rules.trail_pct_mult)
        elif rules.trail_mode == "ATR" and rules.trail_mult > 0.0:
            atr_now = float(grid.atr14_mat[c_i, t]) if grid.atr14_mat[c_i, t] > 0 else float(it["entry_atr"])
            it["shadow_stop"] = max(it["shadow_stop"], it["highest_high"] - rules.trail_mult * atr_now)
        keep.append(it)
    st["watchlist"] = keep


def warm_start_watchlist(st: Dict[str, Any], grid: MarketGrid, rules: Rules, sig, n: int) -> int:
    if rules.wl_mode == "WL_NONE" or WATCHLIST_WARM_START_BARS <= 0:
        return 0
    raw, entry_mat, _, macro_ok, _, _ = sig
    held: Dict[str, int] = {}
    for p in st["positions"]:
        held[p["coin"]] = held.get(p["coin"], 0) + 1
    tmp: Dict[str, Any] = {"watchlist": []}
    for t in range(max(1, n - WATCHLIST_WARM_START_BARS), n):
        queued = {it["coin"] for it in tmp["watchlist"]}
        for c_i in np.flatnonzero(raw[:, t - 1]):
            c_i = int(c_i)
            coin = grid.symbols[c_i]
            if (
                coin in ENTRY_BLOCKED or not macro_ok[t - 1]
                or grid.dvol30_mat[c_i, t - 1] < max(float(grid.liq_floor[c_i]), 1.0)
                or not entry_mat[c_i, t - 1] or held.get(coin, 0) >= rules.max_pyramid
                or (rules.wl_dedupe and coin in queued) or not grid.alive_mat[c_i, t]
            ):
                continue
            a_y = float(grid.atr14_mat[c_i, t - 1])
            o_t = float(grid.open_mat[c_i, t])
            tmp["watchlist"].append({
                "coin": coin, "signal_date": str(grid.dates[t].date()),
                "trigger_price": float(grid.close_mat[c_i, t - 1]),
                "shadow_stop": o_t - rules.sl_mult * a_y, "highest_high": o_t,
                "breakout_quality": float((grid.close_mat[c_i, t - 1] - grid.open_mat[c_i, t - 1]) / max(1e-6, a_y)),
                "entry_atr": a_y,
            })
            queued.add(coin)
        watchlist_lifecycle(tmp, grid, rules, sig, t)
    st["watchlist"] = tmp["watchlist"]
    return len(st["watchlist"])


def update_size_w(st: Dict[str, Any], grid: MarketGrid, rules: Rules, t: int, starved: bool) -> None:
    idx = sym_index(grid)
    eod_equity = st["cash"] + sum(p["units"] * grid.close_mat[idx[p["coin"]], t] for p in st["positions"] if p["coin"] in idx)
    w = min(max(st["size_w"], rules.size_w0), rules.size_w_max)
    if starved:
        w = max(rules.size_w0, w * (1.0 - rules.size_ramp))
    elif eod_equity > 0 and st["cash"] / eod_equity > rules.size_reserve:
        w = min(rules.size_w_max, w * (1.0 + rules.size_ramp))
    st["size_w"] = w


def make_plan(st: Dict[str, Any], grid: MarketGrid, rules: Rules, sig, n: int, est_open: Callable[[int], float],
              alive_at_open: Optional[Callable[[int], bool]] = None) -> Dict[str, Any]:
    raw, entry_mat, _, macro_ok, state_mat, _ = sig
    active_state = state_mat if state_mat is not None else entry_mat
    idx = sym_index(grid)
    cash = float(st["cash"])
    orders: List[Dict[str, Any]] = []
    views: List[Dict[str, Any]] = []
    live_tr = []

    for pos in st["positions"]:
        r = replay_position(grid, rules, sig, pos, n)
        tr = r["tranche"]
        if r["verdict"] and tr is not None:
            c_i = tr.coin_idx
            px = est_open(c_i)
            adv = max(float(grid.dvol30_mat[c_i, n - 1]), bt.LIQUIDITY_FLOOR_USD)
            slip = bt.slip_fraction(grid, c_i, min(1.0, max(0.0, pos["units"] * px / adv)), n - 1)
            _, proceeds, _ = bt._sell_fill_audited(float(pos["units"]), px, slip)
            cash += proceeds
            orders.append({
                "id": f"{grid.dates[n - 1].date()}:SELL:{pos['coin']}:{pos['id']}", "side": "SELL",
                "coin": pos["coin"], "pos_id": pos["id"], "units": pos["units"], "est_price": px,
                "reason": r["verdict"], "overdue_since": r["overdue_since"], "fx": pos_fx(st, pos),
            })
        elif tr is not None:
            live_tr.append(tr)
        views.append(r)

    active_counts: Dict[int, int] = {}
    for tr in live_tr:
        active_counts[tr.coin_idx] = active_counts.get(tr.coin_idx, 0) + 1
    wl = [] if rules.wl_mode == "WL_NONE" else [dict(it) for it in st["watchlist"]]
    wl_coins = {it["coin"] for it in wl} if rules.wl_dedupe else set()
    new_items: List[Dict[str, Any]] = []
    gate_counts = {"signals": 0, "macro": 0, "liquidity": 0, "entry_filter": 0, "pyramid": 0, "duplicate": 0, "not_tradable": 0}
    for c_i in np.flatnonzero(raw[:, n - 1]):
        c_i = int(c_i)
        gate_counts["signals"] += 1
        if grid.symbols[c_i] in ENTRY_BLOCKED:
            gate_counts["not_tradable"] += 1
            continue
        if not macro_ok[n - 1]:
            gate_counts["macro"] += 1
            continue
        if grid.dvol30_mat[c_i, n - 1] < max(float(grid.liq_floor[c_i]), 1.0):
            gate_counts["liquidity"] += 1
            continue
        if not entry_mat[c_i, n - 1]:
            gate_counts["entry_filter"] += 1
            continue
        if active_counts.get(c_i, 0) >= rules.max_pyramid:
            gate_counts["pyramid"] += 1
            continue
        if rules.wl_dedupe and grid.symbols[c_i] in wl_coins:
            gate_counts["duplicate"] += 1
            continue
        o_est = est_open(c_i)
        a_y = float(grid.atr14_mat[c_i, n - 1])
        new_items.append({
            "coin": grid.symbols[c_i], "signal_date": None, "trigger_price": float(grid.close_mat[c_i, n - 1]),
            "shadow_stop": o_est - rules.sl_mult * a_y, "highest_high": o_est,
            "breakout_quality": float((grid.close_mat[c_i, n - 1] - grid.open_mat[c_i, n - 1]) / max(1e-6, a_y)),
            "entry_atr": a_y,
        })
        wl_coins.add(grid.symbols[c_i])
    new_items.sort(key=lambda x: x["breakout_quality"], reverse=True)
    for it in new_items:
        it["_new"] = True
    wl.extend(new_items)

    starved = False
    unfilled: List[Dict[str, Any]] = []
    skipped_info: List[Dict[str, Any]] = []
    if wl and cash >= rules.floor_usd:
        if rules.wl_mode == "WL_DEEPEST_DISCOUNT":
            wl.sort(key=lambda x: (x["trigger_price"] - grid.close_mat[idx[x["coin"]], n - 1]) / max(1e-6, x["trigger_price"]), reverse=True)
        elif rules.wl_mode == "WL_STRONGEST_MOMENTUM":
            wl.sort(key=lambda x: x["breakout_quality"], reverse=True)
        day_equity = cash + sum(tr.units * est_open(tr.coin_idx) for tr in live_tr)
        day_target = max(rules.floor_usd, day_equity * st["size_w"])
        for it in wl:
            c_i = idx[it["coin"]]
            if not (alive_at_open(c_i) if alive_at_open else grid.alive_mat[c_i, n - 1]):
                continue
            if rules.wl_revalidate and not it.get("_new"):
                if not macro_ok[n - 1] or (rules.adx_t > 0.0 and grid.adx14_mat[c_i, n - 1] < rules.adx_t) or not active_state[c_i, n - 1]:
                    continue
            dec = bt.evaluate_entry(
                grid, c_i, n, est_open(c_i), float(it["entry_atr"]), cash, live_tr,
                day_equity, day_target, rules.floor_usd, rules.free_adds, rules.max_pyramid,
                rules.whole_shares, rules.entry_mult, rules.pyr_rule, rules.pyr_gap_bars, rules.pyr_progress_atr
            )
            if dec["ok"]:
                cash -= dec["total_cost"]
                orders.append({
                    "id": f"{grid.dates[n - 1].date()}:BUY:{it['coin']}", "side": "BUY", "coin": it["coin"],
                    "units": dec["units"], "est_price": dec["fill_px"], "est_cost": dec["total_cost"],
                    "entry_atr": it["entry_atr"], "reason": "WATCHLIST" if not it.get("_new") else "SIGNAL",
                    "stop_at_fill": dec["fill_px"] - rules.sl_mult * it["entry_atr"], "fx": USDINR,
                })
                live_tr.append(bt.make_tranche(
                    grid, c_i, n - 1, dec["fill_px"], dec["units"], dec["total_cost"], dec["fee_buy"],
                    dec["slip_mult"], float(it["entry_atr"]), rules.sl_mult
                ))
            else:
                starved = starved or dec["starved"]
                unfilled.append(it)
                skipped_info.append({"coin": it["coin"], "reason": dec["reason"]})
    elif wl:
        starved = True
        unfilled = wl
        skipped_info = [{"coin": it["coin"], "reason": "cash_starved"} for it in wl]

    next_wl = []
    if rules.wl_mode != "WL_NONE":
        for it in unfilled:
            next_wl.append({k: v for k, v in it.items() if k != "_new"})
    return {
        "made_after_bar": str(grid.dates[n - 1].date()), "orders": orders, "starved": starved,
        "views": views, "watchlist_after": next_wl, "skipped": skipped_info, "gates": gate_counts,
        "cash_after_plan": cash,
    }


def run_bar(st: Dict[str, Any], grid: MarketGrid, rules: Rules, sig, t: int, first_new_bar: bool) -> List[str]:
    notes: List[str] = delist_guard(st, grid, t)
    starved_today = False
    for it in st["watchlist"]:
        if it["signal_date"] is None:
            it["signal_date"] = str(grid.dates[t].date())
    if first_new_bar and st.get("plan"):
        notes += apply_plan(st, grid, rules, t)
        starved_today = bool(st["plan"].get("starved"))
    st["plan"] = None
    watchlist_lifecycle(st, grid, rules, sig, t)
    update_size_w(st, grid, rules, t, starved_today)
    st["last_bar_date"] = str(grid.dates[t].date())
    return notes


def finalize_plan(st: Dict[str, Any], plan: Dict[str, Any]) -> None:
    st["plan"] = {"made_after_bar": plan["made_after_bar"], "orders": plan["orders"], "starved": plan["starved"]}
    st["watchlist"] = plan["watchlist_after"]


# ==============================================================================
# 7. LEDGER INPUTS, LIVE HELPERS, REPORTING & MAIN
# ==============================================================================
ENTRY_BLOCKED: Set[str] = set()
LIVE_QUOTES: Dict[str, Tuple[float, float]] = {}


def apply_manual_inputs(st: Dict[str, Any], grid: MarketGrid) -> List[str]:
    notes: List[str] = []
    idx = sym_index(grid)
    for h in MANUAL_HOLDINGS_ADD:
        coin = str(h["coin"]).upper().strip()
        price_inr = h.get("entry_price_inr")
        hid = f"{coin}|{h['entry_date']}|{h.get('units')}|{h.get('invested_inr')}|{price_inr if price_inr is not None else h.get('entry_price')}"
        if hid in st["applied_manual_ids"]:
            continue
        rate = float(h.get("usdinr") or USDINR)
        if price_inr is not None:
            entry_usd = float(price_inr) / rate
            money = float(h["invested_inr"]) if h.get("invested_inr") is not None else float(h["units"]) * float(price_inr)
        else:
            entry_usd = float(h["entry_price"])
            money = float(h["invested_inr"]) if h.get("invested_inr") is not None else float(h["units"]) * entry_usd
        bot_units = money / entry_usd
        st["positions"].append({
            "id": f"P{st['next_id']}", "coin": coin, "entry_date": h["entry_date"],
            "entry_price": entry_usd, "units": bot_units, "cost_usd": round(money, 2),
            "source": "manual", "fx_lock": rate,
            "entry_price_inr": float(price_inr) if price_inr is not None else entry_usd * rate,
        })
        st["next_id"] += 1
        st["applied_manual_ids"].append(hid)
        notes.append(f"ADOPTED {coin}: invested {money:,.2f}; entry price Rs {entry_usd * rate:,.4f} at frozen Rs {rate:.2f}/USD.")
    for r in MANUAL_HOLDINGS_REMOVE:
        rid = f"{r['coin']}|{r['entry_date']}|{r['exit_date']}|{r.get('exit_price_inr', r.get('exit_price'))}"
        if rid in st["applied_remove_ids"]:
            continue
        pos = next((p for p in st["positions"] if p["coin"] == r["coin"] and p["entry_date"] == r["entry_date"]), None)
        if pos is None:
            notes.append(f"REMOVE {r['coin']} {r['entry_date']}: no such position in the ledger")
            continue
        lock = pos_fx(st, pos)
        exit_usd = float(r["exit_price_inr"]) / lock if r.get("exit_price_inr") is not None else float(r["exit_price"])
        proceeds, _ = bt.compute_sell_proceeds_audited(pos["units"] * exit_usd, 0.0)
        st["cash"] += proceeds
        st["closed_trades"].append({
            "coin": pos["coin"], "entry_date": pos["entry_date"], "exit_date": r["exit_date"],
            "entry_price": pos["entry_price"], "exit_price": exit_usd, "units": pos["units"],
            "pnl": round(proceeds - pos["cost_usd"], 2), "reason": "MANUAL_SELL", "fx_lock": lock,
        })
        st["positions"] = [p for p in st["positions"] if p["id"] != pos["id"]]
        st["applied_remove_ids"].append(rid)
        notes.append(f"RECORDED manual sale of {r['coin']}")
    for c in CASH_ADJUSTMENTS:
        if c["id"] in st["applied_cash_ids"]:
            continue
        st["cash"] += float(c["amount_inr"])
        st["applied_cash_ids"].append(c["id"])
        notes.append(f"CASH {float(c['amount_inr']):+,.2f} ({c['id']})")
    return notes


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def now_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz="Asia/Kolkata")


def bar_cutoff() -> pd.Timestamp:
    return now_utc().tz_localize(None).floor("D")


def expected_last_bar() -> pd.Timestamp:
    return bar_cutoff() - pd.Timedelta(days=1)


def fetch_usdinr() -> Tuple[float, str]:
    try:
        r = bt._HTTP_SESSION.get("https://api.coindcx.com/exchange/ticker", timeout=8)
        if r.status_code == 200:
            for m in r.json():
                if str(m.get("market", "")).upper() == "USDTINR":
                    px = float(m.get("last_price"))
                    if 40.0 < px < 300.0:
                        return px, "CoinDCX USDT/INR"
    except Exception:
        pass
    return USDINR_FALLBACK, f"fallback {USDINR_FALLBACK:g} (live rate unavailable)"


def fetch_coindcx_inr_coins() -> Optional[Set[str]]:
    try:
        r = bt._HTTP_SESSION.get("https://api.coindcx.com/exchange/v1/markets_details", timeout=10)
        if r.status_code != 200:
            return None
        coins: Set[str] = set()
        for m in r.json():
            if str(m.get("base_currency_short_name", "")).upper() == "INR" and str(m.get("status", "active")).lower() == "active":
                c = str(m.get("target_currency_short_name", "")).upper().strip()
                if c:
                    coins.add(c)
        return coins if len(coins) >= 20 else None
    except Exception:
        return None


def fetch_live_quote(sym: str) -> Optional[Tuple[float, float]]:
    today = bar_cutoff()
    for host in ("https://data-api.binance.vision", "https://api.binance.com"):
        try:
            r = bt._HTTP_SESSION.get(f"{host}/api/v3/klines", params={"symbol": f"{sym}USDT", "interval": "1d", "limit": 1}, timeout=8)
            if r.status_code == 200:
                row = r.json()[-1]
                if pd.Timestamp(int(row[0]), unit="ms") == today:
                    return float(row[1]), float(row[4])
        except Exception:
            continue
    try:
        r = bt._HTTP_SESSION.get("https://www.okx.com/api/v5/market/candles", params={"instId": f"{sym}-USDT", "bar": "1Dutc", "limit": 1}, timeout=8)
        if r.status_code == 200:
            data = r.json().get("data", [])
            if data and pd.Timestamp(int(data[0][0]), unit="ms") == today:
                return float(data[0][1]), float(data[0][4])
    except Exception:
        pass
    return None


def attach_live_quotes(wanted: Iterable[str]) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    todo = sorted(set(wanted))
    with ThreadPoolExecutor(max_workers=bt.PARALLEL_DOWNLOAD_WORKERS) as ex:
        futs = {ex.submit(fetch_live_quote, s): s for s in todo}
        for fu in as_completed(futs):
            q = fu.result()
            if q is not None:
                out[futs[fu]] = q
    return out


def install_fresh_fetch() -> None:
    original = bt.fetch_single_crypto

    def live_fetch(symbol: str, start_year: int, refresh: bool):
        clean = symbol.upper().replace("-USD", "").replace("/USD", "").replace("/USDT", "").strip()
        cache_file = bt.DATA_DIR / f"{clean}_1d_from{start_year}.parquet"
        force = bool(refresh)
        if not force and cache_file.exists():
            try:
                last = pd.Timestamp(pd.read_parquet(cache_file).index.max()).normalize()
                force = last < expected_last_bar()
            except Exception:
                force = True
        return original(symbol, start_year, force)

    bt.fetch_single_crypto = live_fetch


def sync_universe_list() -> None:
    name = f"{bt.UNIVERSE_NAME.lower()}_top{bt.TOP_N_COINS}_list.json"
    dst = bt.DATA_DIR / name
    if dst.exists():
        return
    for cand in (WORK_DIR.parent / "crypto_backtest" / "data_cache" / name, WORK_DIR / name):
        if cand.exists():
            bt.DATA_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(cand, dst)
            return


def inr(x: float) -> str:
    return bt.format_price(x)


def px(x: float) -> str:
    return bt.format_price(float(x))


def delist_guard(st: Dict[str, Any], grid: MarketGrid, t: int) -> List[str]:
    notes: List[str] = []
    idx = sym_index(grid)
    for pos in list(st["positions"]):
        c_i = idx.get(pos["coin"])
        if c_i is not None and grid.delist_mat[c_i, t]:
            last_px = float(grid.close_mat[c_i, t])
            liq = pos["units"] * last_px * (1.0 - bt.DELIST_HAIRCUT) if (np.isfinite(last_px) and last_px > 0) else 0.0
            st["cash"] += liq
            lock = pos_fx(st, pos)
            st["closed_trades"].append({
                "coin": pos["coin"], "entry_date": pos["entry_date"], "exit_date": str(grid.dates[t].date()),
                "entry_price": pos["entry_price"], "exit_price": round(last_px * (1.0 - bt.DELIST_HAIRCUT), 6),
                "units": pos["units"], "pnl": round(liq - pos["cost_usd"], 2), "reason": "DELISTED", "fx_lock": lock,
            })
            st["positions"] = [p for p in st["positions"] if p["id"] != pos["id"]]
            notes.append(f"DELISTED {pos['coin']}: liquidated at last close less {bt.DELIST_HAIRCUT:.0%} haircut.")
    return notes


def _live_txt(coin: str) -> str:
    q = LIVE_QUOTES.get(coin)
    if not q:
        return "live price unavailable (plan used the last close)"
    lo, la = q
    d = (la / lo - 1.0) * 100.0
    flag = "  !! LARGE DRIFT: your real fill will differ a lot from the paper fill" if abs(d) >= LIVE_QUOTE_DRIFT_WARN_PCT else ""
    return f"bar open {px(lo)}, live {px(la)} ({d:+.2f}% vs open){flag}"


def build_report(st, meta, grid: MarketGrid, plan, notes, warnings) -> Tuple[str, str, str]:
    n = len(grid.dates)
    last = str(grid.dates[n - 1].date())
    idx = sym_index(grid)
    sells = [o for o in plan["orders"] if o["side"] == "SELL"]
    buys = [o for o in plan["orders"] if o["side"] == "BUY"]
    prefix = "[DRY RUN] " if DRY_RUN else ""
    subject = f"{prefix}{ENGINE_VERSION} plan NOW: {len(sells)} SELL, {len(buys)} BUY (data through {last})"
    banner = f"CONFIG {meta['version']} / {meta['param_hash']} / {meta['status']}"
    rows_pos = []
    eq = st["cash"]
    for r in plan["views"]:
        p = r["pos"]
        tr = r["tranche"]
        c_i = idx.get(p["coin"])
        last_px = float(grid.close_mat[c_i, n - 1]) if c_i is not None else float("nan")
        eq += p["units"] * (last_px if last_px == last_px else p["entry_price"])
        pnl = (last_px / p["entry_price"] - 1.0) * 100.0 if last_px == last_px else float("nan")
        stop = f"{px(tr.current_sl)} ({tr.stop_reason})" if tr is not None else "-"
        held = (n - 1 - bar_index(grid, p["entry_date"])) if c_i is not None else 0
        verdict = ("SELL: " + r["verdict"] + (f" (due since {r['overdue_since']})" if r["overdue_since"] else "")) if r["verdict"] else "hold"
        rows_pos.append((p["coin"], p["entry_date"], p["units"], p["entry_price"], last_px, pnl, held, stop, verdict + (f" | {r['note']}" if r["note"] else "")))
    lines = [
        subject, banner,
        f"Data through {last}. USD/INR {USDINR:.2f}. Cash {inr(st['cash'])} | est. equity {inr(eq)} | size weight {st['size_w']:.3f} | positions {len(st['positions'])}"
    ]
    if DRY_RUN:
        lines.append("DRY RUN: nothing was saved.")
    for w in warnings:
        lines.append("WARNING: " + w)
    if notes:
        lines += ["", "LEDGER NOTES"] + ["  " + x for x in notes]
    lines += ["", "SELL NOW"]
    lines += (
        ["  " + f"{o['coin']:<10} x{o['units']:.6g}  ~{px(o['est_price'])} ({inr(o['units'] * o['est_price'])})  {o['reason']}\n      {_live_txt(o['coin'])}" for o in sells]
        or ["  none"]
    )
    lines += ["", "BUY NOW"]
    lines += (
        ["  " + f"{o['coin']:<10} x{o['units']:.6g}  ~{px(o['est_price'])}  spend ~{inr(o['est_cost'])}  stop {px(o['stop_at_fill'])}  [{o['reason']}]\n      {_live_txt(o['coin'])}" for o in buys]
        or ["  none"]
    )
    lines += ["", "POSITIONS"]
    lines += ["  " + f"{c:<10} {d}  x{u:.6g} @{px(ep)}  now {px(lp)} ({pl:+.1f}%)  held {h}d  stop {s}  -> {v}" for c, d, u, ep, lp, pl, h, s, v in rows_pos]
    text = "\n".join(lines)
    html = "<html><body style='font-family:Consolas,monospace;font-size:13px'><pre>" + text.replace("&", "&amp;").replace("<", "&lt;") + "</pre></body></html>"
    return subject, text, html

from html import escape as esc


def fmt_qty(x: float) -> str:
    x = float(x)
    if abs(x) >= 1000:
        return f"{x:,.2f}"
    if abs(x) >= 1:
        return f"{x:,.4f}".rstrip("0").rstrip(".")
    return f"{x:.6f}".rstrip("0").rstrip(".")


def rs(x: float, dec: int = 0) -> str:
    """Indian digit grouping: Rs 12,34,567."""
    if x is None or not np.isfinite(x):
        return "₹0"
    sign = "-" if x < -0.5 * 10 ** (-dec) else ""
    s = f"{abs(x):.{dec}f}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail, parts = whole[:-3], whole[-3:], []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return f"{sign}₹{whole}" + (f".{frac}" if dec else "")


def rs_px(x: float) -> str:
    ax = abs(float(x))
    return rs(x, 4 if ax < 1 else (2 if ax < 1000 else 0))


def _reason(r: str) -> str:
    m = {
        "SIGNAL": "Fresh breakout signal",
        "WATCHLIST": "Waiting signal now fundable",
        "STOP_LOSS": "Initial stop hit",
        "TRAIL_PCT_STOP": "Trailing % stop hit",
        "TRAIL_ATR_STOP": "Trailing ATR stop hit",
        "BREAKEVEN_SL": "Breakeven stop hit",
        "MAX_HOLDING_TIME": "Holding-time limit reached",
        "MACRO_REGIME_EXIT": "Macro regime turned bearish",
        "MANUAL_SELL": "Sold by you",
        "DELISTED": "Delisted",
    }
    if str(r).startswith("SIGNAL_EXIT"):
        return f"Signal exit ({str(r).replace('SIGNAL_EXIT_', '').replace('_', ' ')})"
    return m.get(str(r), str(r).replace("_", " ").capitalize())


def build_email(st, meta, grid: MarketGrid, plan, notes, warnings, rules: Rules, sig) -> Tuple[str, str]:
    n = len(grid.dates)
    last = str(grid.dates[n - 1].date())
    idx = sym_index(grid)
    fee = bt.EFFECTIVE_SELL_FEE_RATE
    now = now_ist().strftime("%A, %d %B %Y · %I:%M %p IST")
    orders = plan["orders"]
    sells = [o for o in orders if o["side"] == "SELL"]
    buys = [o for o in orders if o["side"] == "BUY"]

    # ---- positions in INR at their frozen rates
    layer_seen: Dict[str, int] = {}
    prows: List[Dict[str, Any]] = []
    tot_inv = tot_val = tot_val_usd = 0.0
    for r in sorted(plan["views"], key=lambda v: (v["pos"]["coin"], v["pos"]["entry_date"])):
        p, tr = r["pos"], r["tranche"]
        lock = pos_fx(st, p)
        c_i = idx.get(p["coin"])
        last_px = float(grid.close_mat[c_i, n - 1]) if c_i is not None else float(p["entry_price"])
        inv = float(p["cost_usd"])
        val = p["units"] * last_px
        layer_seen[p["coin"]] = layer_seen.get(p["coin"], 0) + 1
        held = (n - 1 - bar_index(grid, p["entry_date"])) if c_i is not None else 0
        tot_inv += inv
        tot_val += val
        tot_val_usd += p["units"] * last_px
        prows.append({
            "coin": p["coin"], "layer": layer_seen[p["coin"]], "manual": p.get("source") == "manual",
            "units": p["units"], "user_units": p.get("user_units"),
            "entry_inr": float(p.get("entry_price_inr") or p["entry_price"] * lock),
            "real_units": inv / max(1e-12, float(p.get("entry_price_inr") or p["entry_price"] * lock)),
            "inv": inv, "val": val, "pnl": val - inv,
            "pct": (val / inv - 1.0) * 100.0 if inv > 0 else 0.0,
            "px": last_px * lock,
            "stop": (tr.current_sl * lock) if tr is not None else None,
            "dist": ((1.0 - tr.current_sl / last_px) * 100.0) if (tr is not None and last_px > 0) else None,
            "stop_reason": tr.stop_reason if tr is not None else "",
            "held": max(held, 0), "verdict": r["verdict"], "overdue": r["overdue_since"], "note": r["note"],
        })

    cash_inr = float(st["cash"])
    equity_inr = cash_inr + tot_val
    open_pnl = tot_val - tot_inv
    open_pct = (tot_val / tot_inv - 1.0) * 100.0 if tot_inv > 0 else 0.0
    realised = sum(float(t["pnl"]) for t in st["closed_trades"])
    wins = sum(1 for t in st["closed_trades"] if float(t["pnl"]) > 0)
    n_closed = len(st["closed_trades"])

    use_macro = bool(rules.p.get("use_market_macro_system"))
    ok = bool(sig[3][n - 1]) if use_macro else True
    if use_macro:
        badge = (
            f'<span style="background:{"#0b3d2e" if ok else "#4a1620"};color:{"#4ade80" if ok else "#fb7185"};'
            f'border:1px solid {"#14664c" if ok else "#7f2434"};padding:5px 12px;border-radius:20px;'
            f'font-size:11px;font-weight:700;letter-spacing:.8px;">{"● BULLISH" if ok else "● BEARISH"} · BTC vs {int(rules.p.get("macro_ma_len", 0))}d MA</span>'
        )
    else:
        badge = (
            f'<span style="background:#1a2338;color:#94a3b8;border:1px solid #2a3654;padding:5px 12px;'
            f'border-radius:20px;font-size:11px;font-weight:700;letter-spacing:.8px;">REGIME FILTER OFF</span>'
        )
    dry = (
        '<span style="background:#3b2a07;color:#fbbf24;border:1px solid #6b4a0c;padding:5px 12px;'
        'border-radius:20px;font-size:11px;font-weight:700;letter-spacing:.8px;margin-left:6px;">DRY RUN</span>'
        if DRY_RUN else ""
    )

    G, R_, MUTE, TXT = "#4ade80", "#fb7185", "#8a97b4", "#e7ecf6"

    def col(x: float) -> str:
        return G if x >= 0 else R_

    th = 'style="padding:10px 14px;text-align:left;font-weight:600;"'

    def section(icon: str, title: str, sub: str = "") -> str:
        return (
            f'<div style="margin:30px 0 12px 0;"><span style="color:#d4af37;font-size:13px;">{icon}</span> '
            f'<span style="font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#f3f5fa;font-weight:700;">{title}</span>'
            + (f'<span style="font-size:11px;color:{MUTE};margin-left:10px;">{sub}</span>' if sub else "") + '</div>'
        )

    def card(inner: str) -> str:
        return f'<div style="background:#101a2e;border:1px solid #1d2a47;border-radius:12px;overflow:hidden;">{inner}</div>'

    def kpi(label: str, big: str, small: str = "", color: str = TXT, side: str = "l") -> str:
        pad = "0 7px 14px 0" if side == "l" else "0 0 14px 7px"
        return (
            f'<td style="width:50%;padding:{pad};vertical-align:top;"><div style="background:#101a2e;border:1px solid #1d2a47;border-radius:12px;padding:16px 18px;">'
            f'<div style="font-size:10px;letter-spacing:1.6px;text-transform:uppercase;color:{MUTE};">{label}</div>'
            f'<div style="font-size:23px;font-weight:800;color:{color};margin-top:6px;letter-spacing:-.3px;">{big}</div>'
            f'<div style="font-size:11px;color:{MUTE};margin-top:3px;">{small}</div></div></td>'
        )

    def box(title: str, items: List[str], bg: str, border: str, fg: str) -> str:
        lis = "".join(f'<li style="margin:0 0 6px 0;">{esc(x)}</li>' for x in items)
        return (
            f'<div style="background:{bg};border:1px solid {border};border-left:4px solid {border};border-radius:10px;padding:14px 18px;margin-bottom:14px;">'
            f'<div style="font-size:11px;letter-spacing:1.6px;text-transform:uppercase;color:{border};font-weight:700;margin-bottom:8px;">{title}</div>'
            f'<ul style="margin:0;padding-left:18px;color:{fg};font-size:12.5px;line-height:1.55;">{lis}</ul></div>'
        )

    def live_line(coin: str) -> str:
        q = LIVE_QUOTES.get(coin)
        if not q:
            return f'<span style="color:{MUTE};">live price unavailable · plan sized on the last close</span>'
        lo, la = q
        d = (la / lo - 1.0) * 100.0
        warn = abs(d) >= LIVE_QUOTE_DRIFT_WARN_PCT
        return (
            f'<span style="color:{"#fbbf24" if warn else MUTE};">live price is {d:+.2f}% vs this bar\'s open'
            + (" · expect a noticeably different fill" if warn else "") + '</span>'
        )

    act = ""
    for o in sells:
        pos = next((p for p in st["positions"] if p["id"] == o["pos_id"]), None)
        lock = float(o.get("fx") or st.get("fx_account") or USDINR)
        gross = o["units"] * o["est_price"] * (1.0 - fee)
        inv = float(pos["cost_usd"]) if pos else 0.0
        res = gross - inv
        act += (
            f'<tr><td style="padding:16px 14px 4px 14px;"><span style="background:#4a1620;color:{R_};border:1px solid #7f2434;padding:4px 10px;border-radius:6px;font-size:11px;font-weight:800;letter-spacing:1px;">SELL</span>'
            f' <span style="font-size:16px;font-weight:800;color:#fff;margin-left:8px;">{esc(o["coin"])}</span></td>'
            f'<td style="padding:16px 14px 4px 14px;text-align:right;"><span style="font-size:17px;font-weight:800;color:#fff;">≈ {rs(gross)}</span>'
            f'<div style="font-size:11px;color:{col(res)};">{"+" if res >= 0 else "−"}{rs(abs(res))} vs invested</div></td></tr>'
            f'<tr><td colspan="2" style="padding:0 14px 16px 14px;border-bottom:1px solid #1d2a47;font-size:12px;color:#b6c0d6;line-height:1.6;">'
            f'Sell <b style="color:#fff;">all your {esc(o["coin"])}</b> now at market · ref. price {rs_px(o["est_price"] * lock)} · '
            f'<b style="color:#fff;">{esc(_reason(o["reason"]))}</b>{" · due since " + esc(o["overdue_since"]) if o.get("overdue_since") else ""}<br>{live_line(o["coin"])}</td></tr>'
        )
    for o in buys:
        fx = float(o.get("fx") or USDINR)
        act += (
            f'<tr><td style="padding:16px 14px 4px 14px;"><span style="background:#0b3d2e;color:{G};border:1px solid #14664c;padding:4px 10px;border-radius:6px;font-size:11px;font-weight:800;letter-spacing:1px;">BUY</span>'
            f' <span style="font-size:16px;font-weight:800;color:#fff;margin-left:8px;">{esc(o["coin"])}</span></td>'
            f'<td style="padding:16px 14px 4px 14px;text-align:right;"><span style="font-size:17px;font-weight:800;color:#fff;">{rs(o["est_cost"])}</span>'
            f'<div style="font-size:11px;color:{MUTE};">to spend, fees included</div></td></tr>'
            f'<tr><td colspan="2" style="padding:0 14px 16px 14px;border-bottom:1px solid #1d2a47;font-size:12px;color:#b6c0d6;line-height:1.6;">'
            f'Get about <b style="color:#fff;">{fmt_qty(o["est_cost"] / max(1e-12, o["est_price"] * fx * (1.0 + bt.EFFECTIVE_BUY_FEE_RATE)))} {esc(o["coin"])}</b> · ref. price {rs_px(o["est_price"] * fx)} · '
            f'stop alert {rs_px(o["stop_at_fill"] * fx)} (close-based) · <b style="color:#fff;">{esc(_reason(o["reason"]))}</b><br>{live_line(o["coin"])}</td></tr>'
        )
    if not act:
        act = (
            f'<tr><td style="padding:34px 14px;text-align:center;"><div style="font-size:26px;">🌙</div><div style="font-size:14px;font-weight:700;color:#fff;margin-top:6px;">All quiet</div>'
            f'<div style="font-size:12px;color:{MUTE};margin-top:4px;">No orders today. Hold what you have.</div></td></tr>'
        )
    skipped = plan.get("skipped") or []
    skip_txt = (
        f'<div style="font-size:11px;color:{MUTE};padding:10px 14px;border-top:1px solid #1d2a47;">Signals that could not be funded: '
        + ", ".join(f'{esc(x["coin"])} ({esc(str(x["reason"]).replace("_", " "))})' for x in skipped) + "</div>"
    ) if skipped else ""

    prow_html = ""
    for r in prows:
        tag = ' <span style="font-size:9px;background:#26304a;color:#9fb0d0;padding:2px 6px;border-radius:4px;letter-spacing:.8px;vertical-align:middle;">YOURS</span>' if r["manual"] else ""
        layer = f' <span style="color:{MUTE};font-size:10px;">L{r["layer"]}</span>' if r["layer"] > 1 else ""
        if r["verdict"]:
            status = f'<span style="color:{R_};font-weight:700;">SELL · {esc(_reason(r["verdict"]))}</span>'
        else:
            status = f'<span style="color:{MUTE};">Hold</span>'
        stop_txt = (
            f'{rs_px(r["stop"])}<div style="font-size:10px;color:{MUTE};">{r["dist"]:.1f}% below · {esc(str(r["stop_reason"]).replace("_", " ").lower())}</div>'
            if r["stop"] is not None else "–"
        )
        prow_html += (
            f'<tr style="border-top:1px solid #1d2a47;font-size:12.5px;">'
            f'<td style="padding:13px 14px;"><b style="color:#fff;font-size:14px;">{esc(r["coin"])}</b>{layer}{tag}<div style="font-size:10px;color:{MUTE};">≈ {fmt_qty(r["real_units"])} units · {r["held"]}d</div></td>'
            f'<td style="padding:13px 8px;color:#cbd5e8;">{rs(r["inv"])}<div style="font-size:10px;color:{MUTE};">@ {rs_px(r["entry_inr"])}</div></td>'
            f'<td style="padding:13px 8px;color:#fff;font-weight:600;">{rs(r["val"])}<div style="font-size:10px;color:{MUTE};">@ {rs_px(r["px"])}</div></td>'
            f'<td style="padding:13px 8px;color:{col(r["pnl"])};font-weight:800;white-space:nowrap;">{"+" if r["pnl"] >= 0 else "−"}{rs(abs(r["pnl"]))}<div style="font-size:10.5px;font-weight:700;">{r["pct"]:+.1f}%</div></td>'
            f'<td style="padding:13px 8px;color:#f0c674;">{stop_txt}</td>'
            f'<td style="padding:13px 14px 13px 8px;">{status}</td></tr>'
        )
    if not prow_html:
        prow_html = f'<tr><td colspan="6" style="padding:28px;text-align:center;color:{MUTE};font-size:13px;">No open positions.</td></tr>'

    by_coin: Dict[str, Dict[str, Any]] = {}
    for w in st["watchlist"]:
        by_coin.setdefault(w["coin"], w)
    wl = sorted(by_coin.values(), key=lambda w: w["breakout_quality"], reverse=True)[:8]
    wl_html = ""
    for w in wl:
        wl_html += (
            f'<tr style="border-top:1px solid #1d2a47;font-size:12px;color:#b6c0d6;"><td style="padding:10px 14px;"><b style="color:#fff;">{esc(w["coin"])}</b></td>'
            f'<td style="padding:10px 8px;">{rs_px(w["trigger_price"] * USDINR)}</td><td style="padding:10px 8px;color:#f0c674;">{rs_px(w["shadow_stop"] * USDINR)}</td>'
            f'<td style="padding:10px 14px 10px 8px;">{max(0, n - 1 - bar_index(grid, w["signal_date"]))}d waiting</td></tr>'
        )
    wl_block = (
        section("◎", "Watchlist", f"{len(by_coin)} coin(s) waiting for a free slot or cash")
        + card(f'<table style="width:100%;border-collapse:collapse;"><thead><tr style="background:#16233d;font-size:10px;letter-spacing:1.2px;text-transform:uppercase;color:{MUTE};">'
               f'<th {th}>Coin</th><th style="padding:10px 8px;text-align:left;">Trigger</th><th style="padding:10px 8px;text-align:left;">Shadow stop</th><th {th}>Age</th></tr></thead><tbody>{wl_html}</tbody></table>')
    ) if wl_html else ""

    notes_block = box("Ledger notes", [x for x in notes if not x.startswith("USD/INR")], "#0d2236", "#38bdf8", "#bfe3fa") if any(not x.startswith("USD/INR") for x in notes) else ""
    warn_block = box("Read first", list(warnings), "#2b1f08", "#fbbf24", "#fde7a8") if warnings else ""
    closed_html = ""
    for t in st["closed_trades"][-6:][::-1]:
        pi = float(t["pnl"])
        closed_html += (
            f'<tr style="border-top:1px solid #1d2a47;font-size:12px;color:#b6c0d6;"><td style="padding:10px 14px;"><b style="color:#fff;">{esc(t["coin"])}</b></td>'
            f'<td style="padding:10px 8px;">{esc(str(t["entry_date"]))} → {esc(str(t["exit_date"]))}</td><td style="padding:10px 8px;">{esc(_reason(t["reason"]))}</td>'
            f'<td style="padding:10px 14px 10px 8px;text-align:right;color:{col(pi)};font-weight:700;">{"+" if pi >= 0 else "−"}{rs(abs(pi))}</td></tr>'
        )
    closed_block = (
        section("✓", "Recent closed trades", f"{n_closed} closed · {wins} winners")
        + card(f'<table style="width:100%;border-collapse:collapse;"><tbody>{closed_html}</tbody></table>')
    ) if closed_html else ""

    g = plan["gates"]
    _drop = ", ".join(f"{k.replace('_', ' ')} {g[k]}" for k in ("not_tradable", "macro", "liquidity", "entry_filter", "pyramid", "duplicate") if g.get(k))
    funnel = (f'{g["signals"]} signal(s) on the last bar' + (f" ({_drop} filtered out)" if _drop else "")) if g["signals"] else "No breakout signals on the last bar"
    sub_usd = f"≈ {format_price(st['cash'] + tot_val_usd)} on the ledger"

    html_out = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body style="margin:0;padding:22px 10px;background:#060a13;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:{TXT};">
<div style="max-width:720px;margin:0 auto;background:#0b1424;border:1px solid #1d2a47;border-radius:18px;overflow:hidden;">
 <div style="background:linear-gradient(135deg,#19264a 0%,#0b1424 70%);padding:28px 30px 24px 30px;border-bottom:1px solid #d4af37;">
  <table style="width:100%;border-collapse:collapse;"><tr><td style="vertical-align:top;">
   <div style="font-size:11px;letter-spacing:4px;color:#d4af37;font-weight:700;">CRYPTO QUANT</div>
   <div style="font-size:24px;font-weight:800;color:#fff;margin-top:6px;letter-spacing:-.4px;">Daily Dispatch</div>
   <div style="font-size:12px;color:{MUTE};margin-top:6px;">{now}</div>
   <div style="font-size:11px;color:{MUTE};margin-top:2px;">Data through <b style="color:#9fd1ff;">{last}</b> · new bar opened 05:30 IST · {esc(meta["version"])} · cfg {esc(meta["param_hash"])}</div>
   <div style="margin-top:14px;white-space:nowrap;">{badge}{dry}</div>
  </td></tr></table>
 </div>
 <div style="padding:26px 30px 30px 30px;">
  {warn_block}{notes_block}
  <div style="text-align:center;padding:6px 0 22px 0;">
   <div style="font-size:11px;letter-spacing:2.4px;color:{MUTE};text-transform:uppercase;">Paper net worth</div>
   <div style="font-size:40px;font-weight:800;color:#fff;letter-spacing:-1px;margin-top:6px;">{rs(equity_inr)}</div>
   <div style="font-size:11px;color:{MUTE};margin-top:2px;">{esc(sub_usd)}</div>
   <div style="margin-top:6px;"><span style="background:{"#0b3d2e" if open_pnl >= 0 else "#4a1620"};color:{col(open_pnl)};padding:4px 12px;border-radius:20px;font-size:12px;font-weight:700;">{"▲" if open_pnl >= 0 else "▼"} {rs(abs(open_pnl))} open P&amp;L · {open_pct:+.1f}%</span></div>
  </div>
  <table style="width:100%;border-collapse:collapse;"><tr>
   {kpi("Free cash", rs(cash_inr), "available for new buys", "#7dd3fc", "l")}{kpi("Invested", rs(tot_inv), f"{len(prows)} open position(s)", TXT, "r")}
  </tr><tr>
   {kpi("Realised P&amp;L", ("+" if realised >= 0 else "−") + rs(abs(realised)), f"{n_closed} closed trade(s)", col(realised), "l")}{kpi("Position size weight", f"{st['size_w'] * 100:.1f}%", "of equity per new buy", TXT, "r")}
  </tr></table>

  {section("⚡", "Action centre", "place these on CoinDCX now · market orders")}
  {card(f'<table style="width:100%;border-collapse:collapse;">{act}</table>{skip_txt}')}

  {section("▣", "Open positions", "₹ values use the rate frozen when each holding was first counted")}
  {card(f'<table style="width:100%;border-collapse:collapse;"><thead><tr style="background:#16233d;font-size:10px;letter-spacing:1.2px;text-transform:uppercase;color:{MUTE};"><th {th}>Coin</th><th style="padding:10px 8px;text-align:left;">Invested</th><th style="padding:10px 8px;text-align:left;">Now</th><th style="padding:10px 8px;text-align:left;">P&amp;L</th><th style="padding:10px 8px;text-align:left;">Stop alert</th><th {th}>Next open</th></tr></thead><tbody>{prow_html}</tbody></table>')}
  {wl_block}
  {closed_block}
  <div style="margin-top:28px;padding-top:16px;border-top:1px solid #1d2a47;font-size:11px;color:#6b7896;line-height:1.7;text-align:center;">
   {esc(funnel)}.<br>
   Stops are <b>close-based alerts</b>: if a daily close is at or below the stop, sell at the next open. The paper ledger fills at the bar's open; your market order fills near the live price.<br>
   Inside the bot every money amount is a plain number, shown here with the ₹ sign (₹1 = $1 for money). Only per-token prices are converted to ₹, at the USD/INR rate recorded when each holding or order was first counted, so they never move with the exchange rate. Today's rate ({USDINR:.2f}) is used only for orders shown for the first time.<br>
   Paper trading signals, not financial advice.
  </div>
 </div>
</div></body></html>"""
    prefix = "[DRY RUN] " if DRY_RUN else ""
    parts = []
    if buys:
        parts.append(f"{len(buys)} BUY")
    if sells:
        parts.append(f"{len(sells)} SELL")
    action = " · ".join(parts) if parts else "No action"
    subject = f"{prefix}Crypto Quant · {action} · Net worth {rs(equity_inr)} · {pd.Timestamp(last).strftime('%d %b')}"
    return subject, html_out

def send_email(subject: str, html: str, text: str = "") -> None:
    if not (GMAIL_USER and GMAIL_APP_PASSWORD and RECIPIENT_EMAIL) or "your_" in GMAIL_USER + GMAIL_APP_PASSWORD:
        print("[INFO] Email credentials not configured; skipping email.")
        return
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"], msg["From"], msg["To"] = subject, f"{ENGINE_VERSION} <{GMAIL_USER}>", RECIPIENT_EMAIL
        if text:
            msg.attach(MIMEText(text, "plain"))
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as server:
            server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_USER, RECIPIENT_EMAIL, msg.as_string())
        print(f"[SUCCESS] Plan emailed to {RECIPIENT_EMAIL}")
    except Exception as e:
        print(f"[ERROR] Email failed: {e}")


def main() -> None:
    global USDINR
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info(f"Work folder: {WORK_DIR}")
    rules, meta = load_config()
    cfg_hash = f"{meta['version']}:{meta['param_hash']}"
    USDINR, fx_src = fetch_usdinr()
    logging.info(f"{'DRY RUN. ' if DRY_RUN else ''}Config {meta['path']} | {meta['status']} | hash {meta['param_hash']} | USD/INR {USDINR:.2f} ({fx_src}) | {now_ist():%Y-%m-%d %H:%M IST}")

    install_fresh_fetch()
    sync_universe_list()
    extra = sorted({h["coin"].upper() for h in MANUAL_HOLDINGS_ADD} | {p["coin"] for p in (load_state() or {"positions": []})["positions"]})
    grid, _, _, _ = bt.build_market_universe(bt.UNIVERSE_NAME, bt.START_YEAR, False, extra_symbols=extra)
    bank = VectorIndicatorBank(grid)
    sig = bt.compile_signals_composable(grid, rules.p, bank=bank)
    n = len(grid.dates)
    warnings: List[str] = []
    if grid.dates[-1] < expected_last_bar():
        warnings.append(f"newest complete bar is {grid.dates[-1].date()} vs expected {expected_last_bar().date()}.")

    pool = set(bt.load_universe_constituents(bt.UNIVERSE_NAME, bt.TOP_N_COINS))
    cdx = fetch_coindcx_inr_coins() if COINDCX_ONLY else None
    ENTRY_BLOCKED.clear()
    for s in grid.symbols:
        if s not in pool or (cdx is not None and s not in cdx):
            ENTRY_BLOCKED.add(s)

    st = load_state()
    if st is not None:
        st.setdefault("fx_account", USDINR)
    LIVE_QUOTES.clear()
    held_now = {p["coin"] for p in (st["positions"] if st else [])} | {h["coin"].upper() for h in MANUAL_HOLDINGS_ADD}
    LIVE_QUOTES.update(attach_live_quotes([s for s in grid.symbols if (s not in ENTRY_BLOCKED) or s in held_now]))
    first_run = st is None
    notes: List[str] = [f"USD/INR {USDINR:.2f} ({fx_src})"]
    if first_run:
        st = new_state(str(grid.dates[n - 1].date()), rules, cfg_hash)
        notes.append(f"First run: paper ledger created with {float(STARTING_CASH_INR):,.2f} cash.")
    else:
        if st["config_hash"] != cfg_hash:
            notes.append(f"CONFIG CHANGED ({st['config_hash']} -> {cfg_hash}).")
            st["config_hash"] = cfg_hash
        last_done = bar_index(grid, st["last_bar_date"])
        if last_done + 1 < n:
            for k, t in enumerate(range(last_done + 1, n)):
                notes += run_bar(st, grid, rules, sig, t, first_new_bar=(k == 0))
    notes += apply_manual_inputs(st, grid)
    notes += delist_guard(st, grid, n - 1)
    if first_run:
        n_wl = warm_start_watchlist(st, grid, rules, sig, n)
        if rules.wl_mode != "WL_NONE":
            notes.append(f"Watchlist rebuilt from last {WATCHLIST_WARM_START_BARS} bars: {n_wl} items.")

    idx_map = {s: i for i, s in enumerate(grid.symbols)}
    live_open = {idx_map[s]: q[0] for s, q in LIVE_QUOTES.items() if s in idx_map}
    est_open = lambda c_i: float(live_open.get(c_i, grid.close_mat[c_i, n - 1]))
    plan = make_plan(st, grid, rules, sig, n, est_open)
    finalize_plan(st, plan)
    subject, text, _plain_html = build_report(st, meta, grid, plan, notes, warnings)
    print(text)
    if SEND_EMAIL_NOTIFICATION:
        mail_subject, html = build_email(st, meta, grid, plan, notes, warnings, rules, sig)
        send_email(mail_subject, html, text)
    if DRY_RUN:
        print("\n[DRY RUN] state not saved.")
    else:
        save_state({k: v for k, v in st.items() if not k.startswith("_")})
        out = STATE_DIR / f"orders_{grid.dates[n - 1].date()}.csv"
        pd.DataFrame(plan["orders"]).to_csv(out, index=False)
        print(f"\nState saved. Orders written to {out}")


if __name__ == "__main__":
    main()