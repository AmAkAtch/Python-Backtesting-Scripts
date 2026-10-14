#!/usr/bin/env python3
"""
NSE LIVE BOT V13 -- runs AFTER the market close and tells you what to do at TOMORROW'S OPEN.

Design rules (owner decisions, 8-9 Oct 2026)
  * One rulebook. Entry sizing, exits, stops and trailing logic are the SAME functions the backtester uses
    (make_tranche, tranche_open_exit_reason, tranche_eod_update, evaluate_entry, compile_signals_fast).
  * The state file stores only FACTS: what was bought, when, at what price, how many, and your cash.
    Stops, trailing levels, breakeven, holding time and exit verdicts are recomputed from price history
    on EVERY run under the CURRENT winner config. Change the winner and every position and watchlist item is
    judged by the new rules; if the new winner says exit, the plan says SELL.
  * Decide on the close of bar t, trade at the open of bar t+1 (same convention as the backtest).
  * Idempotent same-day reruns: pre-plan watchlist is preserved until the next trading bar arrives.
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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ==============================================================================
# 1. SETTINGS (edit this block)
# ==============================================================================
DRY_RUN: bool = False                      # True = compute + show + email, save NOTHING. Set False once the first plan looks right.
SEND_EMAIL_NOTIFICATION: bool = True
ALLOW_NOT_PROMOTED_CONFIG: bool = True     # owner decision 9 Oct 2026: use the current candidate even though it is not promoted
LIVE_CONFIG_OVERRIDE: Optional[str] = None # path to a winner json; None = winner.json, else winner_candidate.json
MARKET_CLOSE_IST_HOUR: float = 16.0        # warn and exclude today's incomplete bar if run before this hour IST

STARTING_CASH_INR: float = 2000         # YOUR FREE CASH in the broker account on the first run. 0 = no buy orders can be planned.
COLAB_WORK_DIR: str = "/content/drive/MyDrive/nse_live_v13"      # on Google Colab everything lives here (config, cache, state)
WORK_DIR_OVERRIDE: Optional[str] = None                          # local testing: e.g. the nifty50_backtest folder to reuse its data cache

# Holdings you really own (from the Upstox trade register). Added to the ledger once each (by symbol|date|units|price).
MANUAL_HOLDINGS_ADD: List[Dict[str, Any]] = [
    {"coin": "ADANIGREEN", "units": 6, "entry_price": 1377.80, "entry_date": "2026-07-28"},
    {"coin": "BPCL", "units": 20, "entry_price": 293.90, "entry_date": "2026-07-28"},
    {"coin": "CGPOWER", "units": 14, "entry_price": 740.55, "entry_date": "2026-03-13"},
    {"coin": "CHOLAFIN", "units": 4, "entry_price": 1765.60, "entry_date": "2026-09-16"},
    {"coin": "GAIL", "units": 45, "entry_price": 175.78, "entry_date": "2026-06-18"},
    {"coin": "HDBFS", "units": 12, "entry_price": 684.85, "entry_date": "2026-07-29"},
    {"coin": "HUDCO", "units": 38, "entry_price": 208.71, "entry_date": "2026-07-03"},
    {"coin": "ITC", "units": 20, "entry_price": 348.90, "entry_date": "2026-01-02"},
    {"coin": "IOC", "units": 23, "entry_price": 131.38, "entry_date": "2026-01-02"},
    {"coin": "ITCHOTELS", "units": 48, "entry_price": 164.93, "entry_date": "2026-08-17"},
    {"coin": "JSWENERGY", "units": 15, "entry_price": 515.05, "entry_date": "2026-05-15"},
    {"coin": "MOTILALOFS", "units": 9, "entry_price": 869.30, "entry_date": "2026-07-27"},
    {"coin": "NATCOPHARM", "units": 8, "entry_price": 927.55, "entry_date": "2026-03-23"},
    {"coin": "NATIONALUM", "units": 14, "entry_price": 374.00, "entry_date": "2026-09-03"},
    {"coin": "NHPC", "units": 67, "entry_price": 78.50, "entry_date": "2026-05-12"},
    {"coin": "NTPCGREEN", "units": 73, "entry_price": 110.15, "entry_date": "2026-05-15"},
    {"coin": "PHOENIXLTD", "units": 4, "entry_price": 1908.70, "entry_date": "2026-08-11"},
    {"coin": "TATAPOWER", "units": 20, "entry_price": 395.00, "entry_date": "2026-04-08"},
]
# Positions you sold by hand: {"coin": "X", "entry_date": "YYYY-MM-DD", "exit_date": "YYYY-MM-DD", "exit_price": 123.4}
MANUAL_HOLDINGS_REMOVE: List[Dict[str, Any]] = []
# Cash you deposited (+) or withdrew (-), applied once by id: {"id": "2026-11-03-sip", "amount": 5000.0}
CASH_ADJUSTMENTS: List[Dict[str, Any]] = []
# Order ids from the plan that you did NOT execute (the ledger assumes every planned order was executed at the open).
ORDER_SKIPS: List[str] = []
# First run only: approximate standing watchlist by replaying the last N bars of signals and shadow-position life. 0 = off.
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

LIVE_ENGINE_VERSION = "V13.0"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

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
        return Path(__file__).resolve().parent / "nse_live_v13"


WORK_DIR: Path = _resolve_work_dir()
WORK_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_DIR)
STATE_DIR: Path = WORK_DIR / "live_v13_state"

# ==============================================================================
# 3. ENGINE CORE
# ==============================================================================
UNIVERSE_NAME: str = "POOL_LM250_SC100"

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

MACRO_INDEX_TICKER: str = "^NSEI"
FORCE_MACRO_REGIME_FILTER: Optional[bool] = None
START_YEAR: int = 2021
MIN_HISTORY_DAYS: int = 300
PARALLEL_DOWNLOAD_WORKERS: int = 1
CACHE_MAX_AGE_HOURS: float = 72.0

V13_FIXED_RULES: Dict[str, Any] = {
    "wl_dedupe": True,
    "whole_shares": True,
}

ENGINE_DEFAULTS: Dict[str, Any] = {
    **V13_FIXED_RULES,
    "wl_revalidate": True,
    "avg_down_free_adds": 0,
    "size_w0": 0.05, "size_w_max": 0.15, "size_reserve": 0.10, "size_ramp": 0.05,
}

SIZE_RUNT_FRACTION: float = 0.6
ADV_WINDOW: int = 30
ADV_MIN_PERIODS: int = 20

BRACKET_LIQUIDITY_FLOOR_INR: Dict[str, float] = {
    "LARGE": 0.0, "MID": 0.0, "SMALL": 10_000_000.0,
}
BRACKET_SLIPPAGE_BPS: Dict[str, float] = {"LARGE": 8.0, "MID": 15.0, "SMALL": 25.0}
BRACKET_IMPACT_COEF_BPS: Dict[str, float] = {"LARGE": 120.0, "MID": 200.0, "SMALL": 320.0}

TRANCHE_FLOOR_INR: float = 5_000.0
MAX_POSITION_EQUITY_PCT: float = 0.25
MAX_ADV_PARTICIPATION: float = 0.015

SPIKE_MIN_MOVE: float = 0.40
SPIKE_REVERT_WITHIN: int = 5
SPIKE_REVERT_TOL: float = 0.15
CA_DROP_Q: float = 0.70
CA_JUMP_Q: float = 1.90
CA_GAP_EXPLAINS: float = 0.80
CA_MARKET_DAY_MAX: float = 0.06
LARGE_MOVE_LOG: float = 0.30

MANUAL_CORPORATE_ACTIONS: Dict[str, List[str]] = {}
KEEP_AS_REAL_MOVES: Dict[str, List[str]] = {}
DATA_REPAIR_LOG: List[Dict[str, Any]] = []
DROPPED_TODAY_BARS: int = 0

BROKERAGE_MODE: str = "FLAT"
FLAT_BROKERAGE_INR: float = 20.0
STT_RATE: float = 0.0010
EXCHANGE_TXN_RATE: float = 0.0000345
STAMP_DUTY_RATE: float = 0.00015
SEBI_CHARGE_RATE: float = 0.000001
GST_RATE: float = 0.18
DP_CHARGE_INR: float = 20.0
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
REPAIRS_CSV_FILE = OUTPUT_DIR / "DATA_REPAIRS.csv"


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
    denom = units * max(1e-6, (1.0 - slip_mult - EFFECTIVE_SELL_FEE_RATE))
    return (total_cost + FLAT_SELL_FIXED_INR) / max(1e-9, denom)


def _buy_fill_audited(tranche_inr: float, ref_open_price: float, slip_mult: float,
                      whole_shares: bool = False) -> Tuple[float, float, float, FeeBreakdown]:
    fill_px = ref_open_price * (1.0 + slip_mult)
    units = tranche_inr / ref_open_price if ref_open_price > 0 else 0.0
    if whole_shares:
        units = float(np.floor(units))
        if units < 1.0 and ref_open_price > 0 and ref_open_price <= 2.0 * tranche_inr:
            units = 1.0
    gross_at_slip = units * fill_px
    slip_cost = units * (fill_px - ref_open_price)
    total_cash_cost, fees = compute_buy_cost_audited(gross_at_slip, slip_cost)
    return fill_px, units, total_cash_cost, fees


def _buy_fill_units_audited(units: float, ref_open_price: float, slip_mult: float) -> Tuple[float, float, FeeBreakdown]:
    """Audited cost for a fixed number of planned shares at the actual open price."""
    fill_px = ref_open_price * (1.0 + slip_mult)
    gross_at_slip = units * fill_px
    slip_cost = units * (fill_px - ref_open_price)
    total_cash_cost, fees = compute_buy_cost_audited(gross_at_slip, slip_cost)
    return fill_px, total_cash_cost, fees


def _sell_fill_audited(units: float, ref_price: float, slip_mult: float, apply_dp: bool = True) -> Tuple[float, float, FeeBreakdown]:
    fill_px = ref_price * (1.0 - slip_mult)
    gross = units * fill_px
    slip_cost = units * (ref_price - fill_px)
    net_proceeds, fees = compute_sell_proceeds_audited(gross, slip_cost, apply_dp=apply_dp)
    return fill_px, net_proceeds, fees


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

    if df is None or df.empty:
        raise RuntimeError(f"Could not load constituent list for {universe_name}.")
    sym_col = next((c for c in ("Symbol", "SYMBOL", "symbol") if c in df.columns), None)
    if not sym_col:
        raise RuntimeError(f"No Symbol column found in constituent list for {universe_name}.")
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
    global DROPPED_TODAY_BARS
    ticker = _yf_ticker(symbol)
    start = f"{start_year}-01-01"
    raw = None
    for attempt in range(3):
        try:
            raw = yf.download(ticker, start=start, progress=False, auto_adjust=True, threads=False)
            if raw is not None and not raw.empty:
                break
        except Exception as e:
            if attempt == 2:
                logging.warning(f"yfinance download failed for {ticker} after 3 attempts: {e}")
            else:
                time.sleep(1.0 * (attempt + 1))
    if raw is None or raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            raw.columns = raw.columns.get_level_values(0)
        elif "Close" in raw.columns.get_level_values(1):
            raw.columns = raw.columns.get_level_values(1)
        else:
            raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]

    raw = raw.reset_index()
    raw.columns = [c[0] if isinstance(c, tuple) else str(c) for c in raw.columns]
    raw.columns = [c.lower().strip() for c in raw.columns]
    raw = raw.loc[:, ~raw.columns.duplicated()].copy()

    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(raw.columns):
        return None

    raw["quote_volume"] = raw["close"] * raw["volume"]
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None).dt.normalize()
    raw = raw.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    now_ist = dt.datetime.now(IST)
    today_ist = pd.Timestamp(now_ist.date())
    if (now_ist.hour + now_ist.minute / 60.0) >= MARKET_CLOSE_IST_HOUR:
        raw = raw[raw["date"] <= today_ist]
    else:
        if (raw["date"] >= today_ist).any():
            DROPPED_TODAY_BARS += 1
        raw = raw[raw["date"] < today_ist]

    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["open", "high", "low", "close"])

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
                with open(meta_file, "r", encoding="utf-8") as mf:
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
            with open(meta_file, "w", encoding="utf-8") as mf:
                json.dump({"provider": "yfinance", "timestamp": time.time(), "bars": len(df), "start_year": start_year}, mf)
        except Exception:
            pass
        return symbol, df, "yfinance"

    # Fallback to existing cache if live download failed
    if cache_file.exists():
        try:
            df_cached = pd.read_parquet(cache_file)
            if len(df_cached) >= MIN_HISTORY_DAYS:
                logging.warning(f"yfinance failed for {symbol}; falling back to cached parquet.")
                return symbol, df_cached, "stale_cache"
        except Exception:
            pass

    return symbol, None, "none"


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
    brackets: List[str]
    liq_floor: np.ndarray
    slip_bps: np.ndarray
    impact_bps: np.ndarray
    stale_symbols: List[str] = field(default_factory=list)


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
            clean_tail = arr[first_valid:]
            conv = np.convolve(clean_tail, w_norm[::-1], mode="full")[:n_valid]
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

    # 2) unadjusted corporate actions
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
                          extra_symbols: Optional[Dict[str, str]] = None) -> MarketGrid:
    symbols, bracket_map = load_universe_with_brackets(universe_name)
    for _sym, _br in (extra_symbols or {}).items():
        if _sym not in bracket_map:
            bracket_map[_sym] = _br
            symbols = sorted(set(symbols) | {_sym})

    _, macro_df, _ = fetch_single_stock(MACRO_INDEX_TICKER, start_year, refresh)
    if macro_df is None:
        raise RuntimeError(f"Failed to load macro index data for {MACRO_INDEX_TICKER}.")

    raw_universe: Dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=PARALLEL_DOWNLOAD_WORKERS) as executor:
        future_map = {executor.submit(fetch_single_stock, sym, start_year, refresh): sym for sym in symbols}
        for future in as_completed(future_map):
            sym, df, _ = future.result()
            if df is not None and len(df) >= MIN_HISTORY_DAYS:
                raw_universe[sym] = df

    DATA_REPAIR_LOG.clear()
    macro_df, r_m = repair_price_artifacts(macro_df, MACRO_INDEX_TICKER, None, allow_corporate=False)
    mkt_ret = macro_df["close"].pct_change()
    mkt_ret.index = pd.DatetimeIndex(mkt_ret.index).normalize()
    all_recs: List[Dict[str, Any]] = list(r_m)
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
    logging.info(
        f"Data integrity: {n_fix} spike-and-revert bad prints fixed, {n_ca} unadjusted corporate actions rebased, "
        f"{n_kept} large moves kept for manual review (see {REPAIRS_CSV_FILE.name})."
    )

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

    brackets = [bracket_map.get(s, "LARGE") for s in valid_symbols]
    liq_floor = np.array([BRACKET_LIQUIDITY_FLOOR_INR[b] for b in brackets], dtype=np.float64)
    slip_bps = np.array([BRACKET_SLIPPAGE_BPS[b] for b in brackets], dtype=np.float64)
    impact_bps = np.array([BRACKET_IMPACT_COEF_BPS[b] for b in brackets], dtype=np.float64)

    stale_symbols: List[str] = []
    last_master_date = master_dates[-1]

    for i, sym in enumerate(valid_symbols):
        df = raw_universe[sym]
        if df.index.max() < last_master_date:
            stale_symbols.append(sym)
        reindexed = df.reindex(master_dates)
        raw_close = reindexed["close"].copy()

        is_missing = raw_close.isna()
        trailing_missing_count = int((is_missing[::-1].cumprod()[::-1]).astype(int).sum())
        has_ever_traded = (~is_missing).cumsum() > 0
        is_delisted_perm = (is_missing[::-1].cumprod()[::-1].astype(bool) & has_ever_traded & (trailing_missing_count >= 20)).values

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

    return MarketGrid(
        symbols=valid_symbols, dates=master_dates,
        open_mat=open_mat, high_mat=high_mat, low_mat=low_mat, close_mat=close_mat, volume_mat=vol_mat,
        atr14_mat=atr14_mat, dvol30_mat=dvol30_mat, adx14_mat=adx14_mat,
        delist_mat=delist_mat, alive_mat=alive_mat,
        macro_close=macro_df["close"].values, macro_open=macro_df["open"].values,
        brackets=brackets, liq_floor=liq_floor,
        slip_bps=slip_bps, impact_bps=impact_bps,
        stale_symbols=stale_symbols,
    )


def buy_sell_volume_split(h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rng = h - l
    safe = np.where(rng > 0, rng, 1.0)
    frac = np.where(rng > 0, (c - l) / safe, np.where(c >= shift_1d(c), 1.0, 0.0))
    frac = np.clip(np.nan_to_num(frac, nan=0.0), 0.0, 1.0)
    buy = v * frac
    return buy, v - buy


def buy_pressure_mask(h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray,
                      vol_ma_len: int, vol_mult: float, ratio: float) -> np.ndarray:
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
            if float(p["vol_mult"]) > 0.0:
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
            state_entry = np.ones(n_bars, dtype=bool)

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


@dataclass
class Tranche:
    coin: str
    coin_idx: int
    entry_bar: int
    entry_date: pd.Timestamp
    entry_price: float
    units: float
    cost_inr: float
    entry_atr: float
    current_sl: float
    stop_reason: str
    highest_high: float
    lowest_low: float
    be_price: float = 0.0
    pending_stop: bool = False
    pending_reason: str = ""


def slip_fraction(grid: MarketGrid, c_i: int, part_rate: float) -> float:
    return (grid.slip_bps[c_i] + grid.impact_bps[c_i] * np.sqrt(part_rate)) / 10000.0


def make_tranche(grid: MarketGrid, c_i: int, t: int, fill_px: float, units: float, total_cost: float,
                 slip_mult: float, entry_atr: float, sl_mult: float) -> Tranche:
    """A position opened at the open of bar t. Its day-1 stop and trailing updates are judged on bar t's close."""
    sl_price = fill_px - (sl_mult * entry_atr)
    return Tranche(
        coin=grid.symbols[c_i], coin_idx=c_i, entry_bar=t, entry_date=grid.dates[t],
        entry_price=fill_px, units=units, cost_inr=total_cost,
        entry_atr=entry_atr, current_sl=sl_price, stop_reason="STOP_LOSS",
        highest_high=max(fill_px, grid.high_mat[c_i, t]), lowest_low=min(fill_px, grid.low_mat[c_i, t]),
        be_price=breakeven_exit_price(total_cost, units, slip_mult),
        pending_stop=bool(grid.close_mat[c_i, t] <= sl_price), pending_reason="STOP_LOSS"
    )


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
    elif trail_mult > 0.0 and t >= 1:
        atr_floor = tr.highest_high - (trail_mult * grid.atr14_mat[c_i, t - 1])
        if atr_floor > tr.current_sl:
            tr.current_sl = atr_floor
            tr.stop_reason = "TRAIL_ATR_STOP"


def evaluate_entry(grid: MarketGrid, c_i: int, t: int, today_open: float, entry_atr: float, cash: float,
                   live_tranches: List[Tranche], day_equity: float, day_target: float, floor_inr: float,
                   free_adds: int, max_pyramid: int, whole_shares: bool) -> Dict[str, Any]:
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
    return {"ok": False, "reason": "cash_starved", "starved": True}


def reconstitute_params(params: Dict[str, Any]) -> Dict[str, Any]:
    p = copy.deepcopy(params)
    for k in list(p.keys()):
        if isinstance(p[k], float):
            p[k] = round(p[k], 2)

    if p.get("entry_type") == 2 and "xover_gap" in p:
        p["xover_long_len"] = min(200, p["xover_short_len"] + p["xover_gap"])
    if p.get("exit_type") == 5 and "exit_xover_gap" in p:
        p["exit_xover_long_len"] = min(200, p["exit_xover_short_len"] + p["exit_xover_gap"])

    p.setdefault("use_market_macro_system", False if FORCE_MACRO_REGIME_FILTER is None else FORCE_MACRO_REGIME_FILTER)
    p.setdefault("macro_active_exit", False)
    p.setdefault("max_pyramid_layers", 1)
    p.setdefault("adx_thresh", 0.0)
    p.setdefault("be_trigger_atr", 0.0)
    p.setdefault("trail_atr_mult", 0.0)
    p.setdefault("wl_mode", "WL_NONE")
    p.setdefault("tranche_floor", TRANCHE_FLOOR_INR)
    p.update(V13_FIXED_RULES)
    p.setdefault("avg_down_free_adds", 0)
    if p.get("entry_type") == 3 and not p.get("breakout_use_volume", True):
        p["vol_mult"] = 0.0
    for legacy in ("use_global_tp", "tp_mult", "tp_size_pct", "tp_move_sl_be"):
        p.pop(legacy, None)
    return p


# ==============================================================================
# 4. CONFIG
# ==============================================================================
@dataclass
class Rules:
    """Everything the engine rules read from the winner config, resolved once."""
    p: Dict[str, Any]
    xt: int
    trail_mult: float
    trail_pct_mult: float
    be_trigger_mult: float
    macro_active_exit: bool
    max_holding_bars: int
    sl_mult: float
    floor_inr: float
    max_pyramid: int
    free_adds: int
    whole_shares: bool
    wl_mode: str
    wl_dedupe: bool
    wl_revalidate: bool
    adx_t: float
    size_w0: float
    size_w_max: float
    size_reserve: float
    size_ramp: float

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "Rules":
        p = reconstitute_params(dict(params))
        c = {**ENGINE_DEFAULTS, **p}
        return cls(
            p=p, xt=int(p["exit_type"]), trail_mult=float(p.get("trail_atr_mult", 0.0)),
            trail_pct_mult=1.0 - float(p.get("trail_pct", 10.0)) / 100.0, be_trigger_mult=float(p.get("be_trigger_atr", 0.0)),
            macro_active_exit=bool(p.get("macro_active_exit", False)), max_holding_bars=int(p.get("max_holding_bars", 20)),
            sl_mult=float(p.get("sl_mult", 3.0)), floor_inr=float(p.get("tranche_floor", TRANCHE_FLOOR_INR)),
            max_pyramid=int(p.get("max_pyramid_layers", 1)), free_adds=int(c["avg_down_free_adds"]),
            whole_shares=bool(c["whole_shares"]), wl_mode=str(p.get("wl_mode", "WL_NONE")), wl_dedupe=bool(c["wl_dedupe"]),
            wl_revalidate=bool(c["wl_revalidate"]), adx_t=float(p.get("adx_thresh", 0.0)),
            size_w0=float(c["size_w0"]), size_w_max=float(c["size_w_max"]), size_reserve=float(c["size_reserve"]),
            size_ramp=float(c["size_ramp"])
        )


def load_config() -> Tuple[Rules, Dict[str, Any]]:
    if LIVE_CONFIG_OVERRIDE:
        path = Path(LIVE_CONFIG_OVERRIDE)
    elif (WORK_DIR / "winner.json").exists():
        path = WORK_DIR / "winner.json"
    elif (WORK_DIR / "winner_candidate.json").exists():
        path = WORK_DIR / "winner_candidate.json"
    else:
        sys.exit(f"No winner config found. Copy winner.json or winner_candidate.json from the backtester's output folder into {WORK_DIR}")
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    status = str(doc.get("status", "UNKNOWN"))
    if doc.get("version") != LIVE_ENGINE_VERSION:
        sys.exit(f"Config {path.name} was produced by {doc.get('version')}, this live bot is {LIVE_ENGINE_VERSION}. Re-run the backtester.")
    if "NOT PROMOTED" in status.upper() and not ALLOW_NOT_PROMOTED_CONFIG:
        sys.exit(f"Config {path.name} is {status} and ALLOW_NOT_PROMOTED_CONFIG is False.")
    content = hashlib.sha256(json.dumps(doc["best_params"], sort_keys=True, default=str).encode()).hexdigest()[:8]
    meta = {"path": str(path), "status": status, "param_hash": content, "file_hash": doc.get("param_hash"),
            "updated_at": doc.get("updated_at"), "version": doc.get("version")}
    return Rules.from_params(doc["best_params"]), meta


# ==============================================================================
# 5. STATE (facts only) AND SMALL HELPERS
# ==============================================================================
def new_state(last_bar_date: str, rules: Rules, cfg_hash: str) -> Dict[str, Any]:
    return {"schema": 2, "config_hash": cfg_hash, "last_bar_date": last_bar_date, "cash": float(STARTING_CASH_INR),
            "size_w": rules.size_w0, "positions": [], "watchlist": [], "plan": None, "closed_trades": [],
            "applied_manual_ids": [], "applied_cash_ids": [], "applied_remove_ids": [], "next_id": 1}


def load_state() -> Optional[Dict[str, Any]]:
    f = STATE_DIR / "live_state_v13.json"
    if not f.exists():
        return None
    with open(f, "r", encoding="utf-8") as fp:
        st = json.load(fp)
    if st.get("schema") != 2:
        sys.exit("Unknown state schema; refusing to continue.")
    return st


def save_state(st: Dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    f = STATE_DIR / "live_state_v13.json"
    if f.exists():
        shutil.copy2(f, STATE_DIR / "live_state_v13.json.bak")
    tmp = STATE_DIR / "live_state_v13.json.tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(st, fp, indent=1, default=str)
    os.replace(tmp, f)


def sym_index(grid: MarketGrid) -> Dict[str, int]:
    return {s: i for i, s in enumerate(grid.symbols)}


def bar_index(grid: MarketGrid, date_str: str) -> int:
    return int(grid.dates.searchsorted(pd.Timestamp(date_str)))


def entry_atr_of(grid: MarketGrid, pos: Dict[str, Any], c_i: int, e_bar: int) -> float:
    return float(pos["entry_atr"]) if pos.get("entry_atr") else float(grid.atr14_mat[c_i, max(e_bar - 1, 0)])


def _proxy_tranche(grid: MarketGrid, rules: Rules, pos: Dict[str, Any], c_i: int) -> Tranche:
    e = min(bar_index(grid, pos["entry_date"]), len(grid.dates) - 1)
    return make_tranche(grid, c_i, e, float(pos["entry_price"]), float(pos["units"]), float(pos["cost_inr"]),
                        0.0, entry_atr_of(grid, pos, c_i, e), rules.sl_mult)


# ==============================================================================
# 6. REPLAY: rebuild every position's rule state from price history under the CURRENT config
# ==============================================================================
def macro_bear(rules: Rules, macro_ok: np.ndarray, t: int) -> bool:
    return bool(rules.macro_active_exit and t >= 3 and (not macro_ok[t - 1]) and (not macro_ok[t - 2]) and (not macro_ok[t - 3]))


def replay_position(grid: MarketGrid, rules: Rules, sig, pos: Dict[str, Any], n: int) -> Dict[str, Any]:
    """Replays bars e_bar .. n-1 exactly as the backtest engine does, then asks the engine's exit rule about bar n (tomorrow's open)."""
    _, _, exit_mat, macro_ok, _ = sig
    idx = sym_index(grid)
    out: Dict[str, Any] = {"pos": pos, "tranche": None, "verdict": "", "overdue_since": None, "note": ""}
    if pos["coin"] not in idx:
        out["note"] = "NO PRICE DATA - manage manually"
        return out
    c_i = idx[pos["coin"]]
    e_bar = bar_index(grid, pos["entry_date"])
    if e_bar >= n:
        out["note"] = "Entered after the last completed bar - awaiting first daily close"
        return out

    entry_px = float(pos["entry_price"])
    bar_low = float(grid.low_mat[c_i, e_bar])
    bar_high = float(grid.high_mat[c_i, e_bar])
    data_open = float(grid.open_mat[c_i, e_bar])
    # Corporate action check: only rescale if entry_px lies >5% outside the entire entry day's [low, high] range
    if data_open > 0 and bar_low > 0 and (entry_px < bar_low * 0.95 or entry_px > bar_high * 1.05):
        out["note"] = f"Entry {entry_px:.2f} outside day range [{bar_low:.2f}..{bar_high:.2f}]: rebased to open {data_open:.2f}"
        entry_px = data_open

    units = float(pos["units"])
    adv = max(float(grid.dvol30_mat[c_i, max(e_bar - 1, 0)]), 1_000_000.0)
    slip = slip_fraction(grid, c_i, min(1.0, float(pos["cost_inr"]) / adv))
    tr = make_tranche(grid, c_i, e_bar, entry_px, units, float(pos["cost_inr"]), slip,
                      entry_atr_of(grid, pos, c_i, e_bar), rules.sl_mult)
    # Critical: run end-of-day update on the entry bar itself (bar e_bar) so day-1 breakeven/trailing highs are captured
    tranche_eod_update(tr, e_bar, grid, rules.xt, rules.be_trigger_mult, rules.trail_pct_mult, rules.trail_mult)
    out["tranche"] = tr

    for t in range(e_bar + 1, n):
        if not grid.alive_mat[c_i, t]:
            continue
        reason = tranche_open_exit_reason(tr, t, grid, exit_mat, macro_bear(rules, macro_ok, t), rules.xt, rules.max_holding_bars)
        if reason:
            out["verdict"], out["overdue_since"] = reason, str(grid.dates[t].date())
            return out
        tranche_eod_update(tr, t, grid, rules.xt, rules.be_trigger_mult, rules.trail_pct_mult, rules.trail_mult)

    out["verdict"] = tranche_open_exit_reason(tr, n, grid, exit_mat, macro_bear(rules, macro_ok, n), rules.xt, rules.max_holding_bars)
    return out


# ==============================================================================
# 7. APPLY LAST PLAN AT THE OPEN OF BAR t, THEN WATCHLIST AND SIZE UPDATES
# ==============================================================================
def apply_plan(st: Dict[str, Any], grid: MarketGrid, rules: Rules, t: int) -> List[str]:
    notes: List[str] = []
    plan = st.get("plan")
    if not plan:
        return notes
    idx = sym_index(grid)
    cash = float(st["cash"])

    # 1. Sells first
    for o in [o for o in plan["orders"] if o["side"] == "SELL"]:
        if o["id"] in ORDER_SKIPS:
            notes.append(f"{o['id']}: skipped by user (still held)")
            continue
        pos = next((p for p in st["positions"] if p["id"] == o["pos_id"]), None)
        if pos is None or pos["coin"] not in idx:
            continue
        c_i = idx[pos["coin"]]
        raw_px = float(grid.open_mat[c_i, t])
        adv = max(float(grid.dvol30_mat[c_i, t - 1]), 1_000_000.0)
        slip = slip_fraction(grid, c_i, min(1.0, max(0.0, pos["units"] * raw_px / adv)))
        fill_px, proceeds, _ = _sell_fill_audited(float(pos["units"]), raw_px, slip, apply_dp=True)
        cash += proceeds
        st["closed_trades"].append({
            "coin": pos["coin"], "entry_date": pos["entry_date"], "exit_date": str(grid.dates[t].date()),
            "entry_price": pos["entry_price"], "exit_price": round(fill_px, 4), "units": pos["units"],
            "pnl": round(proceeds - pos["cost_inr"], 2), "reason": o["reason"]
        })
        st["positions"] = [p for p in st["positions"] if p["id"] != pos["id"]]
        notes.append(f"SOLD {pos['coin']} x{pos['units']:g} at open {fill_px:.2f} ({o['reason']})")

    # 2. Buys: record the exact planned quantity at the actual open price so ledger matches broker fills
    for o in [o for o in plan["orders"] if o["side"] == "BUY"]:
        if o["id"] in ORDER_SKIPS:
            notes.append(f"{o['id']}: skipped by user (not bought)")
            continue
        if o["coin"] not in idx:
            continue
        c_i = idx[o["coin"]]
        if not grid.alive_mat[c_i, t]:
            notes.append(f"BUY {o['coin']} not filled: no trading in the stock at the open (halted or delisted)")
            continue
        planned_units = float(o["units"])
        raw_px = float(grid.open_mat[c_i, t])
        adv = max(float(grid.dvol30_mat[c_i, t - 1]), 1_000_000.0)
        slip = slip_fraction(grid, c_i, min(1.0, max(0.0, (planned_units * raw_px) / adv)))
        fill_px, total_cost, _ = _buy_fill_units_audited(planned_units, raw_px, slip)
        cash -= total_cost
        pos = {
            "id": f"P{st['next_id']}", "coin": o["coin"], "entry_date": str(grid.dates[t].date()),
            "entry_price": round(fill_px, 4), "units": planned_units, "cost_inr": round(total_cost, 2),
            "entry_atr": float(o["entry_atr"]), "source": "bot"
        }
        st["next_id"] += 1
        st["positions"].append(pos)
        notes.append(f"BOUGHT {o['coin']} x{planned_units:g} at open {fill_px:.2f} (cost {inr(total_cost)})")

    st["cash"] = cash

    # Watchlist items created for this bar: re-anchor their shadow stop to the REAL open
    for it in st["watchlist"]:
        if it["signal_date"] == str(grid.dates[t].date()) and it["coin"] in idx:
            o_t = float(grid.open_mat[idx[it["coin"]], t])
            it["highest_high"], it["shadow_stop"] = o_t, o_t - rules.sl_mult * it["entry_atr"]
    return notes


def watchlist_lifecycle(st: Dict[str, Any], grid: MarketGrid, rules: Rules, sig, t: int) -> None:
    """End-of-bar-t life of items that are still waiting (engine step 5)."""
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
        stopped = grid.low_mat[c_i, t] <= it["shadow_stop"]
        exited = rules.xt in (3, 4, 5, 6, 7) and bool(exit_mat[c_i, t])
        if stopped or exited:
            continue
        if grid.high_mat[c_i, t] > it["highest_high"]:
            it["highest_high"] = float(grid.high_mat[c_i, t])
        if rules.trail_mult > 0.0:
            it["shadow_stop"] = max(it["shadow_stop"], it["highest_high"] - rules.trail_mult * float(grid.atr14_mat[c_i, t]))
        keep.append(it)
    st["watchlist"] = keep


def warm_start_watchlist(st: Dict[str, Any], grid: MarketGrid, rules: Rules, sig, n: int) -> int:
    """Approximate the standing watchlist over the last WATCHLIST_WARM_START_BARS bars."""
    if rules.wl_mode == "WL_NONE" or WATCHLIST_WARM_START_BARS <= 0:
        return 0
    _, entry_mat, _, _, _ = sig
    held: Dict[str, int] = {}
    for p in st["positions"]:
        held[p["coin"]] = held.get(p["coin"], 0) + 1
    tmp: Dict[str, Any] = {"watchlist": []}
    for t in range(max(1, n - WATCHLIST_WARM_START_BARS), n):
        queued = {it["coin"] for it in tmp["watchlist"]}
        for c_i in np.flatnonzero(entry_mat[:, t - 1]):
            c_i = int(c_i)
            coin = grid.symbols[c_i]
            if held.get(coin, 0) >= rules.max_pyramid or (rules.wl_dedupe and coin in queued) or not grid.alive_mat[c_i, t]:
                continue
            a_y = float(grid.atr14_mat[c_i, t - 1])
            o_t = float(grid.open_mat[c_i, t])
            tmp["watchlist"].append({
                "coin": coin, "signal_date": str(grid.dates[t].date()),
                "trigger_price": float(grid.close_mat[c_i, t - 1]),
                "shadow_stop": o_t - rules.sl_mult * a_y, "highest_high": o_t,
                "breakout_quality": float((grid.close_mat[c_i, t - 1] - grid.open_mat[c_i, t - 1]) / max(1e-6, a_y)),
                "entry_atr": a_y
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


# ==============================================================================
# 8. PLAN FOR THE OPEN OF BAR n (n = number of completed bars)
# ==============================================================================
def make_plan(st: Dict[str, Any], grid: MarketGrid, rules: Rules, sig, n: int, est_open: Callable[[int], float],
              alive_at_open: Optional[Callable[[int], bool]] = None) -> Dict[str, Any]:
    raw, entry_mat, exit_mat, macro_ok, state_mat = sig
    active_state = state_mat if state_mat is not None else entry_mat
    idx = sym_index(grid)
    cash = float(st["cash"])
    orders: List[Dict[str, Any]] = []
    views: List[Dict[str, Any]] = []
    live_tr: List[Tranche] = []

    # ---- 1. Sells (engine step 2)
    for pos in st["positions"]:
        r = replay_position(grid, rules, sig, pos, n)
        tr = r["tranche"]
        if r["verdict"] and tr is not None:
            c_i = tr.coin_idx
            px = est_open(c_i)
            adv = max(float(grid.dvol30_mat[c_i, n - 1]), 1_000_000.0)
            slip = slip_fraction(grid, c_i, min(1.0, max(0.0, pos["units"] * px / adv)))
            _, proceeds, _ = _sell_fill_audited(float(pos["units"]), px, slip, apply_dp=True)
            cash += proceeds
            orders.append({
                "id": f"{grid.dates[n - 1].date()}:SELL:{pos['coin']}:{pos['id']}", "side": "SELL", "coin": pos["coin"],
                "pos_id": pos["id"], "units": pos["units"], "est_price": px, "reason": r["verdict"],
                "overdue_since": r["overdue_since"]
            })
        elif tr is not None:
            live_tr.append(tr)
        views.append(r)

    # ---- 2. Candidates decided on the close of bar n-1 (engine step 3)
    active_counts: Dict[int, int] = {}
    for tr in live_tr:
        active_counts[tr.coin_idx] = active_counts.get(tr.coin_idx, 0) + 1
    wl = [] if rules.wl_mode == "WL_NONE" else [dict(it) for it in st["watchlist"]]
    wl_coins = {it["coin"] for it in wl} if rules.wl_dedupe else set()
    new_items: List[Dict[str, Any]] = []
    gate_counts = {"signals": 0, "macro": 0, "liquidity": 0, "adx": 0, "pyramid": 0, "duplicate": 0, "queued": 0}

    for c_i in np.flatnonzero(raw[:, n - 1]):
        c_i = int(c_i)
        gate_counts["signals"] += 1
        if not macro_ok[n - 1]:
            gate_counts["macro"] += 1
            continue
        if grid.dvol30_mat[c_i, n - 1] < max(float(grid.liq_floor[c_i]), 1.0):
            gate_counts["liquidity"] += 1
            continue
        if not entry_mat[c_i, n - 1]:
            gate_counts["adx"] += 1
            continue
        if active_counts.get(c_i, 0) >= rules.max_pyramid:
            gate_counts["pyramid"] += 1
            continue
        if rules.wl_dedupe and grid.symbols[c_i] in wl_coins:
            gate_counts["duplicate"] += 1
            continue
        gate_counts["queued"] += 1
        o_est = est_open(c_i)
        a_y = float(grid.atr14_mat[c_i, n - 1])
        new_items.append({
            "coin": grid.symbols[c_i], "signal_date": None, "trigger_price": float(grid.close_mat[c_i, n - 1]),
            "shadow_stop": o_est - rules.sl_mult * a_y, "highest_high": o_est,
            "breakout_quality": float((grid.close_mat[c_i, n - 1] - grid.open_mat[c_i, n - 1]) / max(1e-6, a_y)),
            "entry_atr": a_y
        })
        wl_coins.add(grid.symbols[c_i])

    new_items.sort(key=lambda x: x["breakout_quality"], reverse=True)
    for it in new_items:
        it["_new"] = True
    wl.extend(new_items)

    # ---- 3. Sizing and fills at the estimated open (engine step 4)
    starved = False
    unfilled: List[Dict[str, Any]] = []
    skipped_info: List[Dict[str, Any]] = []
    if wl and cash >= rules.floor_inr:
        if rules.wl_mode == "WL_DEEPEST_DISCOUNT":
            wl.sort(key=lambda x: (x["trigger_price"] - grid.close_mat[idx[x["coin"]], n - 1]) / max(1e-6, x["trigger_price"]), reverse=True)
        elif rules.wl_mode == "WL_STRONGEST_MOMENTUM":
            wl.sort(key=lambda x: x["breakout_quality"], reverse=True)
        day_equity = cash + sum(tr.units * est_open(tr.coin_idx) for tr in live_tr)
        day_target = max(rules.floor_inr, day_equity * st["size_w"])
        for it in wl:
            c_i = idx[it["coin"]]
            if not (alive_at_open(c_i) if alive_at_open else grid.alive_mat[c_i, n - 1]):
                continue
            if rules.wl_revalidate and not it.get("_new"):
                if not macro_ok[n - 1] or (rules.adx_t > 0.0 and grid.adx14_mat[c_i, n - 1] < rules.adx_t) or not active_state[c_i, n - 1]:
                    continue
            dec = evaluate_entry(grid, c_i, n, est_open(c_i), float(it["entry_atr"]), cash, live_tr, day_equity, day_target,
                                 rules.floor_inr, rules.free_adds, rules.max_pyramid, rules.whole_shares)
            if dec["ok"]:
                cash -= dec["total_cost"]
                orders.append({
                    "id": f"{grid.dates[n - 1].date()}:BUY:{it['coin']}", "side": "BUY", "coin": it["coin"], "units": dec["units"],
                    "est_price": dec["fill_px"], "est_cost": dec["total_cost"], "entry_atr": it["entry_atr"],
                    "reason": "WATCHLIST" if not it.get("_new") else "SIGNAL",
                    "stop_at_fill": dec["fill_px"] - rules.sl_mult * it["entry_atr"]
                })
                live_tr.append(make_tranche(grid, c_i, n - 1, dec["fill_px"], dec["units"], dec["total_cost"],
                                            dec["slip_mult"], float(it["entry_atr"]), rules.sl_mult))
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
        "made_after_bar": str(grid.dates[n - 1].date()), "orders": orders, "starved": starved, "views": views,
        "watchlist_after": next_wl, "skipped": skipped_info, "gates": gate_counts, "cash_after_plan": cash
    }


# ==============================================================================
# 9. ONE FULL DAY & DELIST CHECK
# ==============================================================================
def check_delisted_positions(st: Dict[str, Any], grid: MarketGrid, t: int) -> List[str]:
    """Never auto-writes off a real position on a Yahoo data gap. Emits a loud warning instead."""
    warns: List[str] = []
    idx = sym_index(grid)
    for pos in st["positions"]:
        c_i = idx.get(pos["coin"])
        if c_i is not None and grid.delist_mat[c_i, t]:
            warns.append(
                f"{pos['coin']} ({pos['id']}) has 20+ missing bars (possible delisting, suspension, or ticker change). "
                f"Position kept in ledger; if sold manually, add to MANUAL_HOLDINGS_REMOVE."
            )
    return warns


def run_bar(st: Dict[str, Any], grid: MarketGrid, rules: Rules, sig, t: int, first_new_bar: bool) -> List[str]:
    """Process completed bar t: promote planned watchlist & apply orders at open (first new bar only), then EOD lifecycle."""
    notes: List[str] = []
    starved_today = False
    if first_new_bar and st.get("plan"):
        if "watchlist_after" in st["plan"]:
            st["watchlist"] = st["plan"]["watchlist_after"]
        for it in st["watchlist"]:
            if it["signal_date"] is None:
                it["signal_date"] = str(grid.dates[t].date())
        notes += apply_plan(st, grid, rules, t)
        starved_today = bool(st["plan"].get("starved"))
    st["plan"] = None
    watchlist_lifecycle(st, grid, rules, sig, t)
    update_size_w(st, grid, rules, t, starved_today)
    st["last_bar_date"] = str(grid.dates[t].date())
    return notes


def finalize_plan(st: Dict[str, Any], plan: Dict[str, Any]) -> None:
    """Store plan and post-plan watchlist inside st['plan'] so same-day reruns remain 100% idempotent."""
    st["plan"] = {
        "made_after_bar": plan["made_after_bar"],
        "orders": plan["orders"],
        "starved": plan["starved"],
        "watchlist_after": plan["watchlist_after"],
    }


# ==============================================================================
# 10. LEDGER INPUTS
# ==============================================================================
def apply_manual_inputs(st: Dict[str, Any], grid: MarketGrid) -> List[str]:
    notes: List[str] = []
    idx = sym_index(grid)
    for h in MANUAL_HOLDINGS_ADD:
        hid = f"{h['coin']}|{h['entry_date']}|{h['units']}|{h['entry_price']}"
        if hid in st["applied_manual_ids"]:
            continue
        gross = float(h["units"]) * float(h["entry_price"])
        total_cost, _ = compute_buy_cost_audited(gross, 0.0)
        st["positions"].append({
            "id": f"P{st['next_id']}", "coin": h["coin"], "entry_date": h["entry_date"],
            "entry_price": float(h["entry_price"]), "units": float(h["units"]),
            "cost_inr": round(total_cost, 2), "source": "manual"
        })
        st["next_id"] += 1
        st["applied_manual_ids"].append(hid)
        notes.append(
            f"ADOPTED {h['coin']} x{h['units']} @ {h['entry_price']} ({h['entry_date']})"
            + ("" if h["coin"] in idx else " [no price data: excluded from rule evaluation]")
        )
    for r in MANUAL_HOLDINGS_REMOVE:
        rid = f"{r['coin']}|{r['entry_date']}|{r['exit_date']}|{r['exit_price']}"
        if rid in st["applied_remove_ids"]:
            continue
        pos = next((p for p in st["positions"] if p["coin"] == r["coin"] and p["entry_date"] == r["entry_date"]), None)
        if pos is None:
            notes.append(f"REMOVE {r['coin']} {r['entry_date']}: no such position in the ledger")
            continue
        gross = pos["units"] * float(r["exit_price"])
        proceeds, _ = compute_sell_proceeds_audited(gross, 0.0, apply_dp=True)
        st["cash"] += proceeds
        st["closed_trades"].append({
            "coin": pos["coin"], "entry_date": pos["entry_date"], "exit_date": r["exit_date"],
            "entry_price": pos["entry_price"], "exit_price": r["exit_price"], "units": pos["units"],
            "pnl": round(proceeds - pos["cost_inr"], 2), "reason": "MANUAL_SELL"
        })
        st["positions"] = [p for p in st["positions"] if p["id"] != pos["id"]]
        st["applied_remove_ids"].append(rid)
        notes.append(f"RECORDED manual sale of {r['coin']}")
    for c in CASH_ADJUSTMENTS:
        if c["id"] in st["applied_cash_ids"]:
            continue
        st["cash"] += float(c["amount"])
        st["applied_cash_ids"].append(c["id"])
        notes.append(f"CASH {c['amount']:+,.0f} ({c['id']})")
    return notes


# ==============================================================================
# 11. REPORTING (PREMIUM EXECUTIVE DASHBOARD WITH DYNAMIC EXIT WATCH)
# ==============================================================================
def inr(x: float) -> str:
    return f"₹{x:,.2f}" if np.isfinite(x) else "-"


def _format_exit_watch(tr: Optional[Tranche], c_i: Optional[int], last_px: float, held_bars: int,
                       rules: Rules, exit_mat: np.ndarray, macro_ok: np.ndarray, n: int) -> Tuple[str, str]:
    """Builds HTML and plain-text Exit Watch summaries showing ONLY the exit rules active in the current config."""
    if tr is None or c_i is None or not np.isfinite(last_px) or last_px <= 0:
        return '<span style="color:#ef4444;font-weight:600;">No price data</span>', "No data"

    html_lines: List[str] = []
    txt_parts: List[str] = []

    # 1. Stop level (always active: initial, breakeven, or trailing)
    sl_px = tr.current_sl
    dist_pct = ((last_px - sl_px) / last_px) * 100.0
    sl_label_map = {
        "STOP_LOSS": "Initial",
        "BREAKEVEN_SL": "Breakeven",
        "TRAIL_PCT_STOP": "Trail %",
        "TRAIL_ATR_STOP": "Trail ATR",
    }
    sl_type = sl_label_map.get(tr.stop_reason, tr.stop_reason)
    if dist_pct <= 0:
        sl_col = "#ef4444"
        dist_str = f"BREACHED ({dist_pct:+.1f}%)"
    elif dist_pct < 4.0:
        sl_col = "#f59e0b"
        dist_str = f"-{dist_pct:.1f}% below close"
    else:
        sl_col = "#10b981"
        dist_str = f"-{dist_pct:.1f}% below close"
    html_lines.append(
        f'<div><span style="color:#94a3b8;">Stop ({sl_type}):</span> '
        f'<strong style="color:#f8fafc;">{inr(sl_px)}</strong> '
        f'<span style="color:{sl_col};font-weight:600;">({dist_str})</span></div>'
    )
    txt_parts.append(f"Stop({sl_type}): {inr(sl_px)} ({dist_str})")

    # 2. Signal exit (shown only for exit_type 3..7; omitted for 0 and 1)
    if rules.xt in (3, 4, 5, 6, 7):
        sig_on = bool(exit_mat[c_i, n - 1])
        gate_ok = held_bars >= 3 or (tr.highest_high - tr.entry_price) >= (1.0 * tr.entry_atr)
        if sig_on and gate_ok:
            sig_html = '<span style="color:#ef4444;font-weight:700;">ACTIVE (Exit Triggered)</span>'
            sig_txt = "ACTIVE"
        elif sig_on:
            sig_html = '<span style="color:#f59e0b;font-weight:600;">ARMED (Wait Bar 3 / +1 ATR)</span>'
            sig_txt = "ARMED"
        else:
            sig_html = '<span style="color:#10b981;">INTACT</span>'
            sig_txt = "INTACT"
        html_lines.append(f'<div><span style="color:#94a3b8;">Signal (T{rules.xt}):</span> {sig_html}</div>')
        txt_parts.append(f"Sig(T{rules.xt}): {sig_txt}")

    # 3. Time limit (bars held vs max_holding_bars)
    time_col = "#ef4444" if held_bars >= rules.max_holding_bars else ("#f59e0b" if held_bars >= rules.max_holding_bars - 2 else "#cbd5e1")
    html_lines.append(
        f'<div><span style="color:#94a3b8;">Time:</span> '
        f'<span style="color:{time_col};font-weight:600;">{held_bars} / {rules.max_holding_bars}d</span></div>'
    )
    txt_parts.append(f"Time: {held_bars}/{rules.max_holding_bars}d")

    # 4. Macro regime exit (shown only if macro_active_exit is enabled)
    if rules.macro_active_exit:
        bear_streak = 0
        for k in range(1, 4):
            if n - k >= 0 and not macro_ok[n - k]:
                bear_streak += 1
            else:
                break
        if bear_streak >= 3:
            m_html = '<span style="color:#ef4444;font-weight:700;">BEAR EXIT (3/3d below MA)</span>'
            m_txt = "BEAR EXIT"
        elif bear_streak > 0:
            m_html = f'<span style="color:#f59e0b;font-weight:600;">WARN ({bear_streak}/3d below MA)</span>'
            m_txt = f"WARN({bear_streak}/3d)"
        else:
            m_html = '<span style="color:#10b981;">BULL / OK</span>'
            m_txt = "OK"
        html_lines.append(f'<div><span style="color:#94a3b8;">Macro:</span> {m_html}</div>')
        txt_parts.append(f"Macro: {m_txt}")

    return "".join(html_lines), " | ".join(txt_parts)


def build_report(st: Dict[str, Any], meta: Dict[str, Any], grid: MarketGrid, rules: Rules,
                 sig, plan: Dict[str, Any], notes: List[str], warnings: List[str]) -> Tuple[str, str, str]:
    n = len(grid.dates)
    last_date = str(grid.dates[n - 1].date())
    idx = sym_index(grid)
    _, _, exit_mat, macro_ok, _ = sig
    sells = [o for o in plan["orders"] if o["side"] == "SELL"]
    buys = [o for o in plan["orders"] if o["side"] == "BUY"]
    prefix = "[DRY RUN] " if DRY_RUN else ""
    subject = f"{prefix}NSE Quant V13 Plan: {len(sells)} SELL | {len(buys)} BUY ({last_date})"

    total_cost = 0.0
    total_current_val = 0.0
    rows_holdings = []

    for r in plan["views"]:
        p = r["pos"]
        tr = r["tranche"]
        c_i = idx.get(p["coin"])
        has_data = c_i is not None and np.isfinite(grid.close_mat[c_i, n - 1])
        last_px = float(grid.close_mat[c_i, n - 1]) if has_data else float("nan")
        entry_px = float(p["entry_price"])
        units = float(p["units"])
        pos_cost = float(p["cost_inr"])

        pos_val = (units * last_px) if has_data else pos_cost
        pnl_inr = pos_val - pos_cost
        pnl_pct = (pnl_inr / pos_cost * 100.0) if (has_data and pos_cost > 0) else 0.0

        total_cost += pos_cost
        total_current_val += pos_val

        # Held bars as of tomorrow's open (bar n), clamped to >= 0
        held_bars = max(0, n - bar_index(grid, p["entry_date"])) if c_i is not None else 0
        verdict = r["verdict"] if r["verdict"] else ("HOLD" if has_data else "NO DATA")
        ew_html, ew_txt = _format_exit_watch(tr, c_i, last_px, held_bars, rules, exit_mat, macro_ok, n)

        rows_holdings.append({
            "coin": p["coin"],
            "entry_date": p["entry_date"],
            "units": units,
            "entry_px": entry_px,
            "last_px": last_px,
            "has_data": has_data,
            "pnl_pct": pnl_pct,
            "pnl_inr": pnl_inr,
            "held_bars": held_bars,
            "verdict": verdict,
            "overdue": r.get("overdue_since"),
            "note": r.get("note", ""),
            "exit_watch_html": ew_html,
            "exit_watch_txt": ew_txt,
        })

    rows_holdings.sort(key=lambda x: (0 if x["verdict"] not in ("HOLD", "NO DATA") else (1 if x["verdict"] == "NO DATA" else 2), -x["pnl_pct"]))

    free_cash = float(st["cash"])
    total_equity = free_cash + total_current_val
    overall_pnl_inr = total_current_val - total_cost
    overall_pnl_pct = (overall_pnl_inr / total_cost * 100.0) if total_cost > 0 else 0.0
    unique_scrips = len({p["coin"] for p in st["positions"]})
    total_tranches = len(st["positions"])
    scrip_summary = f"{unique_scrips} scrips" + (f" ({total_tranches} tranches)" if total_tranches != unique_scrips else "")

    # --- Plain-Text Fallback ---
    banner = f"CONFIG {meta['version']} / {meta['param_hash']} / {meta['status']}"
    lines = [
        subject, banner,
        f"Data: {last_date} | Cash: {inr(free_cash)} | Equity: {inr(total_equity)} | Active: {scrip_summary}",
        "",
        "SELL AT OPEN: " + (", ".join(f"{o['coin']} x{o['units']:g} ({o['reason']})" for o in sells) or "None"),
        "BUY AT OPEN:  " + (", ".join(f"{o['coin']} x{o['units']:g} @ ~{inr(o['est_price'])}" for o in buys) or "None"),
        "",
        "HOLDINGS & EXIT WATCH:"
    ]
    for rh in rows_holdings:
        note_str = f" [NOTE: {rh['note']}]" if rh["note"] else ""
        lines.append(
            f"  - {rh['coin']} x{rh['units']:g} | Entry {inr(rh['entry_px'])} ({rh['entry_date']}) | "
            f"Last {inr(rh['last_px'])} | P&L {inr(rh['pnl_inr'])} ({rh['pnl_pct']:+.1f}%) | "
            f"{rh['exit_watch_txt']} | Action: {rh['verdict']}{note_str}"
        )
    if warnings:
        lines.append("\nWARNINGS:\n" + "\n".join(f"  ! {w}" for w in warnings))
    text = "\n".join(lines)

    # --- Premium HTML Template ---
    badge_status = (
        '<span style="background-color:#f59e0b;color:#111827;padding:3px 8px;border-radius:4px;font-weight:700;font-size:11px;">DRY RUN</span>'
        if DRY_RUN else
        '<span style="background-color:#10b981;color:#ffffff;padding:3px 8px;border-radius:4px;font-weight:700;font-size:11px;">LIVE RUN</span>'
    )
    pnl_color = "#10b981" if overall_pnl_inr >= 0 else "#ef4444"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>{subject}</title>
    </head>
    <body style="margin:0;padding:24px;background-color:#0b0f19;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#e2e8f0;">
      <table align="center" border="0" cellpadding="0" cellspacing="0" width="100%" style="max-width:980px;margin:auto;">

        <!-- HEADER -->
        <tr>
          <td style="padding:16px 20px;background:linear-gradient(135deg,#1e293b 0%,#0f172a 100%);border-radius:12px;border:1px solid #334155;">
            <table width="100%" border="0" cellpadding="0" cellspacing="0">
              <tr>
                <td>
                  <div style="font-size:11px;color:#94a3b8;letter-spacing:1.5px;text-transform:uppercase;font-weight:700;">Quant Execution Terminal • {meta['version']}</div>
                  <div style="font-size:22px;color:#ffffff;font-weight:800;margin-top:4px;">Daily Pre-Market Execution Plan</div>
                  <div style="font-size:12px;color:#64748b;margin-top:4px;">Bar Date: <span style="color:#cbd5e1;">{last_date}</span> • Param Hash: <span style="color:#cbd5e1;">{meta['param_hash']}</span></div>
                </td>
                <td align="right" valign="top">
                  {badge_status}
                </td>
              </tr>
            </table>
          </td>
        </tr>

        <!-- EXECUTIVE SUMMARY STAT CARDS -->
        <tr>
          <td style="padding-top:16px;">
            <table width="100%" border="0" cellpadding="0" cellspacing="0">
              <tr>
                <td width="24%" style="padding:12px 16px;background:#1e293b;border-radius:8px;border:1px solid #334155;">
                  <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;font-weight:600;">Total Portfolio Equity</div>
                  <div style="font-size:18px;color:#ffffff;font-weight:700;margin-top:4px;">{inr(total_equity)}</div>
                </td>
                <td width="1%"></td>
                <td width="24%" style="padding:12px 16px;background:#1e293b;border-radius:8px;border:1px solid #334155;">
                  <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;font-weight:600;">Free Available Cash</div>
                  <div style="font-size:18px;color:#38bdf8;font-weight:700;margin-top:4px;">{inr(free_cash)}</div>
                </td>
                <td width="1%"></td>
                <td width="24%" style="padding:12px 16px;background:#1e293b;border-radius:8px;border:1px solid #334155;">
                  <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;font-weight:600;">Holdings Unrealized P&amp;L</div>
                  <div style="font-size:18px;color:{pnl_color};font-weight:700;margin-top:4px;">{inr(overall_pnl_inr)} <span style="font-size:12px;">({overall_pnl_pct:+.1f}%)</span></div>
                </td>
                <td width="1%"></td>
                <td width="24%" style="padding:12px 16px;background:#1e293b;border-radius:8px;border:1px solid #334155;">
                  <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;font-weight:600;">Active Scrips / Target W</div>
                  <div style="font-size:18px;color:#cbd5e1;font-weight:700;margin-top:4px;">{scrip_summary} <span style="font-size:12px;color:#94a3b8;">({st['size_w']*100:.1f}%)</span></div>
                </td>
              </tr>
            </table>
          </td>
        </tr>
    """

    # --- ACTION ORDERS ---
    if sells or buys:
        html += """
        <tr>
          <td style="padding-top:20px;">
            <div style="font-size:13px;font-weight:700;color:#f8fafc;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:8px;">⚡ Orders to Execute at Market Open (09:15 AM)</div>
            <table width="100%" border="0" cellpadding="8" cellspacing="0" style="background:#1e293b;border:1px solid #334155;border-radius:8px;font-size:13px;border-collapse:collapse;">
              <thead>
                <tr style="background:#0f172a;color:#94a3b8;border-bottom:1px solid #334155;text-align:left;font-size:11px;text-transform:uppercase;">
                  <th style="padding:10px 12px;">Side</th>
                  <th style="padding:10px 12px;">Scrip</th>
                  <th style="padding:10px 12px;">Quantity</th>
                  <th style="padding:10px 12px;">Est. Price</th>
                  <th style="padding:10px 12px;">Order Value</th>
                  <th style="padding:10px 12px;">Trigger / Exit Reason</th>
                </tr>
              </thead>
              <tbody>
        """
        for s in sells:
            overdue_str = f" <span style='color:#ef4444;font-size:11px;'>(Due: {s['overdue_since']})</span>" if s.get("overdue_since") else ""
            html += f"""
                <tr style="border-bottom:1px solid #334155;background:#450a0a33;">
                  <td style="padding:10px 12px;"><span style="background:#dc2626;color:#ffffff;font-weight:700;padding:2px 8px;border-radius:4px;font-size:11px;">SELL</span></td>
                  <td style="padding:10px 12px;font-weight:700;color:#ffffff;">{s['coin']}</td>
                  <td style="padding:10px 12px;color:#cbd5e1;">{s['units']:g}</td>
                  <td style="padding:10px 12px;color:#cbd5e1;">{inr(s['est_price'])}</td>
                  <td style="padding:10px 12px;color:#cbd5e1;">{inr(s['units'] * s['est_price'])}</td>
                  <td style="padding:10px 12px;color:#f87171;font-weight:600;">{s['reason']}{overdue_str}</td>
                </tr>
            """
        for b in buys:
            html += f"""
                <tr style="border-bottom:1px solid #334155;background:#064e3b33;">
                  <td style="padding:10px 12px;"><span style="background:#16a34a;color:#ffffff;font-weight:700;padding:2px 8px;border-radius:4px;font-size:11px;">BUY</span></td>
                  <td style="padding:10px 12px;font-weight:700;color:#ffffff;">{b['coin']}</td>
                  <td style="padding:10px 12px;color:#cbd5e1;">{b['units']:g}</td>
                  <td style="padding:10px 12px;color:#cbd5e1;">{inr(b['est_price'])}</td>
                  <td style="padding:10px 12px;color:#cbd5e1;">{inr(b['est_cost'])}</td>
                  <td style="padding:10px 12px;color:#4ade80;">[{b['reason']}] Initial SL: {inr(b['stop_at_fill'])}</td>
                </tr>
            """
        html += """
              </tbody>
            </table>
          </td>
        </tr>
        """
    else:
        html += """
        <tr>
          <td style="padding-top:16px;">
            <div style="background:#1e293b;border:1px solid #334155;padding:12px 16px;border-radius:8px;color:#94a3b8;font-size:13px;">
              ☕ <strong>No orders scheduled for tomorrow's open.</strong> Portfolio allocations remain intact.
            </div>
          </td>
        </tr>
        """

    # --- TABLE 2: CURRENT HOLDINGS & EXIT WATCH ---
    html += """
        <tr>
          <td style="padding-top:24px;">
            <div style="font-size:13px;font-weight:700;color:#f8fafc;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:8px;">📊 Portfolio Holdings &amp; Exit Watch</div>
            <table width="100%" border="0" cellpadding="8" cellspacing="0" style="background:#1e293b;border:1px solid #334155;border-radius:8px;font-size:12px;border-collapse:collapse;">
              <thead>
                <tr style="background:#0f172a;color:#94a3b8;border-bottom:1px solid #334155;text-align:left;font-size:11px;text-transform:uppercase;">
                  <th style="padding:10px 10px;">Scrip</th>
                  <th style="padding:10px 10px;">Entry Date</th>
                  <th style="padding:10px 10px;text-align:right;">Qty</th>
                  <th style="padding:10px 10px;text-align:right;">Entry Px</th>
                  <th style="padding:10px 10px;text-align:right;">Last Px</th>
                  <th style="padding:10px 10px;text-align:right;">Net P&amp;L</th>
                  <th style="padding:10px 10px;">Exit Watch (Active Config Rules)</th>
                  <th style="padding:10px 10px;text-align:center;">Action</th>
                </tr>
              </thead>
              <tbody>
    """

    for r in rows_holdings:
        pnl_badge_col = "#10b981" if r["pnl_inr"] >= 0 else "#ef4444"
        pnl_cell = (
            f'<span style="color:{pnl_badge_col};font-weight:700;">{r["pnl_pct"]:+.1f}%</span>'
            f'<br><span style="font-size:10px;color:#94a3b8;">{inr(r["pnl_inr"])}</span>'
            if r["has_data"] else "-"
        )
        note_html = f'<div style="font-size:10px;color:#f59e0b;margin-top:2px;">⚠️ {r["note"]}</div>' if r["note"] else ""

        if r["verdict"] == "HOLD":
            action_pill = '<span style="background:#064e3b;color:#34d399;padding:2px 8px;border-radius:4px;font-weight:600;font-size:11px;">HOLD</span>'
            row_bg = "transparent"
        elif r["verdict"] == "NO DATA":
            action_pill = '<span style="background:#78350f;color:#fcd34d;padding:2px 8px;border-radius:4px;font-weight:700;font-size:11px;">NO DATA</span>'
            row_bg = "#78350f22"
        else:
            overdue_tag = f'<div style="font-size:10px;color:#f87171;margin-top:2px;">Due: {r["overdue"]}</div>' if r.get("overdue") else ""
            action_pill = (
                f'<span style="background:#7f1d1d;color:#fca5a5;padding:2px 8px;border-radius:4px;font-weight:700;font-size:11px;">'
                f'EXIT: {r["verdict"]}</span>{overdue_tag}'
            )
            row_bg = "#450a0a22"

        html += f"""
                <tr style="border-bottom:1px solid #334155;background:{row_bg};">
                  <td style="padding:9px 10px;font-weight:700;color:#ffffff;">{r['coin']}{note_html}</td>
                  <td style="padding:9px 10px;color:#94a3b8;">{r['entry_date']}</td>
                  <td style="padding:9px 10px;text-align:right;color:#cbd5e1;">{r['units']:g}</td>
                  <td style="padding:9px 10px;text-align:right;color:#cbd5e1;">{inr(r['entry_px'])}</td>
                  <td style="padding:9px 10px;text-align:right;font-weight:600;color:#ffffff;">{inr(r['last_px'])}</td>
                  <td style="padding:9px 10px;text-align:right;">{pnl_cell}</td>
                  <td style="padding:9px 10px;line-height:1.45;font-size:11px;">{r['exit_watch_html']}</td>
                  <td style="padding:9px 10px;text-align:center;">{action_pill}</td>
                </tr>
        """

    html += """
              </tbody>
            </table>
          </td>
        </tr>
    """

    # --- TABLE 3: RANKED WATCHLIST (post-plan standing queue) ---
    watchlist = plan.get("watchlist_after", [])
    if watchlist:
        sorted_wl = sorted(watchlist, key=lambda x: x.get("breakout_quality", 0.0), reverse=True)
        html += """
        <tr>
          <td style="padding-top:24px;">
            <div style="font-size:13px;font-weight:700;color:#f8fafc;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:8px;">🎯 Ranked Watchlist Pipeline</div>
            <table width="100%" border="0" cellpadding="8" cellspacing="0" style="background:#1e293b;border:1px solid #334155;border-radius:8px;font-size:12px;border-collapse:collapse;">
              <thead>
                <tr style="background:#0f172a;color:#94a3b8;border-bottom:1px solid #334155;text-align:left;font-size:11px;text-transform:uppercase;">
                  <th style="padding:10px 10px;width:50px;">Rank</th>
                  <th style="padding:10px 10px;">Scrip</th>
                  <th style="padding:10px 10px;">Signal Date</th>
                  <th style="padding:10px 10px;text-align:right;">Trigger Px</th>
                  <th style="padding:10px 10px;text-align:right;">Shadow Stop</th>
                  <th style="padding:10px 10px;text-align:right;">Breakout Score</th>
                  <th style="padding:10px 10px;text-align:right;">ATR (14)</th>
                </tr>
              </thead>
              <tbody>
        """
        for i, it in enumerate(sorted_wl, 1):
            score = it.get("breakout_quality", 0.0)
            score_color = "#38bdf8" if score > 0 else "#94a3b8"
            html += f"""
                <tr style="border-bottom:1px solid #334155;">
                  <td style="padding:8px 10px;color:#94a3b8;font-weight:700;">#{i}</td>
                  <td style="padding:8px 10px;font-weight:700;color:#ffffff;">{it['coin']}</td>
                  <td style="padding:8px 10px;color:#94a3b8;">{it.get('signal_date') or 'Today (' + last_date + ')'}</td>
                  <td style="padding:8px 10px;text-align:right;color:#cbd5e1;">{inr(it.get('trigger_price', 0.0))}</td>
                  <td style="padding:8px 10px;text-align:right;color:#cbd5e1;">{inr(it.get('shadow_stop', 0.0))}</td>
                  <td style="padding:8px 10px;text-align:right;font-weight:600;color:{score_color};">{score:.2f}</td>
                  <td style="padding:8px 10px;text-align:right;color:#cbd5e1;">{inr(it.get('entry_atr', 0.0))}</td>
                </tr>
            """
        html += """
              </tbody>
            </table>
          </td>
        </tr>
        """

    # --- TABLE 4: SYSTEM NOTES & COMPLETE SIGNALS FUNNEL ---
    g = plan.get("gates", {})
    if notes or warnings or plan.get("skipped") or g:
        html += """
        <tr>
          <td style="padding-top:24px;">
            <div style="font-size:12px;font-weight:700;color:#94a3b8;text-transform:uppercase;margin-bottom:8px;">Ledger Diagnostics &amp; Signals Funnel</div>
            <div style="background:#0f172a;border:1px solid #334155;border-radius:8px;padding:12px;font-family:monospace;font-size:11px;color:#94a3b8;line-height:1.6;">
        """
        for w in warnings:
            html += f'<div style="color:#ef4444;font-weight:600;">⚠️ WARNING: {w}</div>'
        for n_item in notes:
            html += f'<div>• {n_item}</div>'
        if plan.get("skipped"):
            html += f'<div style="color:#f59e0b;">• Unfunded Candidates: {", ".join(x["coin"] + " (" + x["reason"] + ")" for x in plan["skipped"])}</div>'
        if g:
            html += (
                f'<div style="color:#64748b;margin-top:4px;">'
                f'Signal Funnel: {g.get("signals", 0)} raw &rarr; '
                f'-{g.get("macro", 0)} macro &rarr; '
                f'-{g.get("liquidity", 0)} liquidity &rarr; '
                f'-{g.get("adx", 0)} ADX/filter &rarr; '
                f'-{g.get("pyramid", 0)} max-pyramid &rarr; '
                f'-{g.get("duplicate", 0)} wl-duplicate = '
                f'<strong style="color:#cbd5e1;">{g.get("queued", 0)} queued</strong>'
                f'</div>'
            )
        html += """
            </div>
          </td>
        </tr>
        """

    html += """
        <!-- FOOTER -->
        <tr>
          <td style="padding-top:24px;padding-bottom:24px;text-align:center;font-size:11px;color:#64748b;">
            NSE Live Bot Automation Engine • Generated at market close for next day execution.
          </td>
        </tr>
      </table>
    </body>
    </html>
    """

    return subject, text, html


def send_email(subject: str, text: str, html: str) -> None:
    if not (GMAIL_USER and GMAIL_APP_PASSWORD and RECIPIENT_EMAIL) or "your_" in GMAIL_USER + GMAIL_APP_PASSWORD:
        print("[INFO] Email credentials not configured; skipping email.")
        return
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"], msg["From"], msg["To"] = subject, f"NSE Quant Engine <{GMAIL_USER}>", RECIPIENT_EMAIL
        msg.attach(MIMEText(text, "plain", "utf-8"))
        msg.attach(MIMEText(html, "html", "utf-8"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as server:
            server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_USER, RECIPIENT_EMAIL, msg.as_string())
        print(f"[SUCCESS] Plan emailed to {RECIPIENT_EMAIL}")
    except Exception as e:
        print(f"[ERROR] Email failed: {e}")


# ==============================================================================
# 12. MAIN
# ==============================================================================
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rules, meta = load_config()
    cfg_hash = f"{meta['version']}:{meta['param_hash']}"
    logging.info(f"{'DRY RUN. ' if DRY_RUN else ''}Config {meta['path']} | {meta['status']} | hash {meta['param_hash']}")

    # Load state BEFORE building the market universe so existing positions & watchlist items outside the pool are fetched
    st = load_state()
    first_run = st is None

    extra: Dict[str, str] = {h["coin"]: "MID" for h in MANUAL_HOLDINGS_ADD}
    if st is not None:
        for p in st.get("positions", []):
            extra.setdefault(p["coin"], "MID")
        for it in st.get("watchlist", []):
            extra.setdefault(it["coin"], "MID")
        if st.get("plan") and "watchlist_after" in st["plan"]:
            for it in st["plan"]["watchlist_after"]:
                extra.setdefault(it["coin"], "MID")

    grid = build_market_universe(UNIVERSE_NAME, START_YEAR, True, extra_symbols=extra)
    sig = compile_signals_fast(grid, rules.p)
    n = len(grid.dates)
    now = dt.datetime.now(IST)
    warnings: List[str] = []

    if now.weekday() < 5 and (now.hour + now.minute / 60.0) < MARKET_CLOSE_IST_HOUR:
        warnings.append(
            f"Run started at {now:%H:%M} IST (before {MARKET_CLOSE_IST_HOUR:.0f}:00 IST market close). "
            f"Today's incomplete bar was excluded ({DROPPED_TODAY_BARS} tickers had intraday prints); plan is based on {grid.dates[-1].date()}."
        )
    if len(grid.stale_symbols) > max(5, int(0.10 * len(grid.symbols))):
        warnings.append(
            f"Partial Yahoo Finance update detected: {len(grid.stale_symbols)}/{len(grid.symbols)} stocks lack bars for "
            f"{grid.dates[-1].date()} (forward-filled from prior close). Consider re-running later this evening."
        )
    if (now.date() - grid.dates[-1].date()).days > 5:
        warnings.append(f"Price data ends {grid.dates[-1].date()}: more than 5 days old.")

    notes: List[str] = []
    if first_run:
        st = new_state(str(grid.dates[n - 1].date()), rules, cfg_hash)
        notes.append(f"First run: ledger created with cash {inr(st['cash'])}.")
        if st["cash"] <= 0:
            warnings.append("STARTING_CASH_INR is 0, so no BUY orders can be planned. Set it to your free broker cash.")
    else:
        if st["config_hash"] != cfg_hash:
            notes.append(f"CONFIG CHANGED ({st['config_hash']} -> {cfg_hash}): every position and watchlist item is re-judged by the new rules.")
            st["config_hash"] = cfg_hash
        last_done = bar_index(grid, st["last_bar_date"])
        if last_done + 1 < n:
            for k, t in enumerate(range(last_done + 1, n)):
                notes += run_bar(st, grid, rules, sig, t, first_new_bar=(k == 0))

    notes += apply_manual_inputs(st, grid)
    warnings += check_delisted_positions(st, grid, n - 1)

    if first_run:
        n_wl = warm_start_watchlist(st, grid, rules, sig, n)
        if rules.wl_mode != "WL_NONE":
            notes.append(f"Watchlist rebuilt from the last {WATCHLIST_WARM_START_BARS} bars: {n_wl} items (approximation of the backtest's standing queue).")

    est_open = lambda c_i: float(grid.close_mat[c_i, n - 1])
    plan = make_plan(st, grid, rules, sig, n, est_open)
    finalize_plan(st, plan)

    subject, text, html = build_report(st, meta, grid, rules, sig, plan, notes, warnings)
    print(text)
    if SEND_EMAIL_NOTIFICATION:
        send_email(subject, text, html)
    if DRY_RUN:
        print("\n[DRY RUN] state not saved.")
    else:
        save_state({k: v for k, v in st.items() if not k.startswith("_")})
        out = STATE_DIR / f"orders_{grid.dates[n - 1].date()}.csv"
        pd.DataFrame(plan["orders"]).to_csv(out, index=False)
        print(f"\nState saved. Orders written to {out}")


if __name__ == "__main__":
    main()