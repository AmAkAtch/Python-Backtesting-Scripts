#!/usr/bin/env python3
"""
NSE LIVE BOT V14.0 -- runs AFTER the market close and tells you what to do at TOMORROW'S OPEN.

Design rules & Parity Updates (V14.0):
  * One rulebook. Entry sizing, exits, stops and trailing logic are the SAME functions the backtester uses
    (make_tranche, tranche_open_exit_reason, tranche_eod_update, evaluate_entry, compile_signals_composable).
  * Composable signal and multi-exit support matching `bt_nse_14.py`.
  * The state file stores only FACTS: what was bought, when, at what price, how many, and your cash.
  * Decide on the close of bar t, trade at the open of bar t+1.
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
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import yfinance as yf

# ==============================================================================
# 1. SETTINGS (edit this block)
# ==============================================================================
DRY_RUN: bool = True                       # True = compute + show + email, save NOTHING. Set False once verified.
SEND_EMAIL_NOTIFICATION: bool = True
ALLOW_NOT_PROMOTED_CONFIG: bool = True     # Use current candidate even if unpromoted
LIVE_CONFIG_OVERRIDE: Optional[str] = None # Path to winner json; None = winner.json or winner_candidate.json
MARKET_CLOSE_IST_HOUR: float = 16.0        # Exclude today's incomplete bar if run before this hour IST

STARTING_CASH_INR: float = 0.0             # Free cash in broker account on first run. 0 = no BUY orders planned.
COLAB_WORK_DIR: str = "/content/drive/MyDrive/nse_live_v14"
WORK_DIR_OVERRIDE: Optional[str] = None    # Local testing override

# Holdings you own (from Upstox trade register). Added once each.
MANUAL_HOLDINGS_ADD: List[Dict[str, Any]] = []
MANUAL_HOLDINGS_REMOVE: List[Dict[str, Any]] = []
CASH_ADJUSTMENTS: List[Dict[str, Any]] = []
ORDER_SKIPS: List[str] = []
WATCHLIST_WARM_START_BARS: int = 60

# Credentials
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

LIVE_ENGINE_VERSION = "V14.0"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

# ==============================================================================
# 2. WORK FOLDER SETUP
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
        return Path(__file__).resolve().parent / "nse_live_v14"


WORK_DIR: Path = _resolve_work_dir()
WORK_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_DIR)
STATE_DIR: Path = WORK_DIR / "live_v14_state"

# ==============================================================================
# 3. ENGINE CORE & VECTORIZED INDICATORS (Parity with bt_nse_14.py)
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
    "pyramid_rule": 0,
    "pyramid_gap_bars": 0,
    "pyramid_progress_atr": 0.0,
    "size_w0": 0.05,
    "size_w_max": 0.15,
    "size_reserve": 0.10,
    "size_ramp": 0.05,
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
DELIST_RECOVERY_FRACTION: float = 0.50

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
        return (~np.isnan(rf)) & (~np.isnan(rs)) & (rf < rs) & (shift_2d(rf) >= shift_2d(rs))
    elif xname == "MA_XOVER_DOWN":
        s_ma = bank.ma(p[f"{prefix}_xover_s_len"], p[f"{prefix}_xover_s_type"])
        l_ma = bank.ma(p[f"{prefix}_xover_l_len"], p[f"{prefix}_xover_l_type"])
        return (~np.isnan(s_ma)) & (~np.isnan(l_ma)) & (s_ma < l_ma) & (shift_2d(s_ma) >= shift_2d(l_ma))
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


def make_tranche(grid: MarketGrid, c_i: int, t: int, fill_px: float, units: float, total_cost: float,
                 slip_mult: float, entry_atr: float, sl_mult: float) -> Tranche:
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
    elif trail_mode == "ATR" and trail_mult > 0.0 and t >= 1:
        atr_now = grid.atr14_mat[c_i, t] if grid.atr14_mat[c_i, t] > 0 else tr.entry_atr
        atr_floor = tr.highest_high - (trail_mult * atr_now)
        if atr_floor > tr.current_sl:
            tr.current_sl = atr_floor
            tr.stop_reason = "TRAIL_ATR_STOP"


def reconstitute_params(params: Dict[str, Any]) -> Dict[str, Any]:
    p = copy.deepcopy(params)
    for k in list(p.keys()):
        if isinstance(p[k], float):
            p[k] = round(p[k], 2)

    if p.get("entry_type") == 2 and "xover_gap" in p:
        p["xover_long_len"] = min(200, p["xover_short_len"] + p["xover_gap"])

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
    p.setdefault("tranche_floor", TRANCHE_FLOOR_INR)
    p.update(V13_FIXED_RULES)
    p.setdefault("avg_down_free_adds", 0)
    p.setdefault("pyramid_rule", 0)
    p.setdefault("pyramid_gap_bars", 0)
    p.setdefault("pyramid_progress_atr", 0.0)
    if p.get("entry_type") == 3 and not p.get("breakout_use_volume", True):
        p["vol_mult"] = 0.0
    return p


# ==============================================================================
# 4. CONFIG
# ==============================================================================
@dataclass
class Rules:
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
    pyr_rule: int
    pyr_gap: int
    pyr_prog: float

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "Rules":
        p = reconstitute_params(dict(params))
        c = {**ENGINE_DEFAULTS, **p}
        return cls(
            p=p, xt=int(p.get("exit_type", 0)), trail_mult=float(p.get("trail_atr_mult", 0.0)),
            trail_pct_mult=1.0 - float(p.get("trail_pct", 10.0)) / 100.0, be_trigger_mult=float(p.get("be_trigger_atr", 0.0)),
            macro_active_exit=bool(p.get("macro_active_exit", False)), max_holding_bars=int(p.get("max_holding_bars", 20)),
            sl_mult=float(p.get("sl_mult", 3.0)), floor_inr=float(p.get("tranche_floor", TRANCHE_FLOOR_INR)),
            max_pyramid=int(p.get("max_pyramid_layers", 1)), free_adds=int(c["avg_down_free_adds"]),
            whole_shares=bool(c["whole_shares"]), wl_mode=str(p.get("wl_mode", "WL_NONE")), wl_dedupe=bool(c["wl_dedupe"]),
            wl_revalidate=bool(c["wl_revalidate"]), adx_t=float(p.get("adx_thresh", 0.0)),
            size_w0=float(c["size_w0"]), size_w_max=float(c["size_w_max"]), size_reserve=float(c["size_reserve"]),
            size_ramp=float(c["size_ramp"]), pyr_rule=int(c["pyramid_rule"]), pyr_gap=int(c["pyramid_gap_bars"]),
            pyr_prog=float(c["pyramid_progress_atr"])
        )


def load_config() -> Tuple[Rules, Dict[str, Any]]:
    if LIVE_CONFIG_OVERRIDE:
        path = Path(LIVE_CONFIG_OVERRIDE)
    elif (WORK_DIR / "winner.json").exists():
        path = WORK_DIR / "winner.json"
    elif (WORK_DIR / "winner_candidate.json").exists():
        path = WORK_DIR / "winner_candidate.json"
    else:
        sys.exit(f"No winner config found. Copy winner.json or winner_candidate.json into {WORK_DIR}")
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    status = str(doc.get("status", "UNKNOWN"))
    if doc.get("version") != LIVE_ENGINE_VERSION:
        sys.exit(f"Config {path.name} was produced by {doc.get('version')}, this live bot is {LIVE_ENGINE_VERSION}.")
    if "NOT PROMOTED" in status.upper() and not ALLOW_NOT_PROMOTED_CONFIG:
        sys.exit(f"Config {path.name} is {status} and ALLOW_NOT_PROMOTED_CONFIG is False.")
    content = hashlib.sha256(json.dumps(doc["best_params"], sort_keys=True, default=str).encode()).hexdigest()[:8]
    meta = {"path": str(path), "status": status, "param_hash": content, "file_hash": doc.get("param_hash"),
            "updated_at": doc.get("updated_at"), "version": doc.get("version")}
    return Rules.from_params(doc["best_params"]), meta


# ==============================================================================
# 5. STATE & HELPERS
# ==============================================================================
def new_state(last_bar_date: str, rules: Rules, cfg_hash: str) -> Dict[str, Any]:
    return {"schema": 2, "config_hash": cfg_hash, "last_bar_date": last_bar_date, "cash": float(STARTING_CASH_INR),
            "size_w": rules.size_w0, "positions": [], "watchlist": [], "plan": None, "closed_trades": [],
            "applied_manual_ids": [], "applied_cash_ids": [], "applied_remove_ids": [], "next_id": 1}


def load_state() -> Optional[Dict[str, Any]]:
    f = STATE_DIR / "live_state_v14.json"
    if not f.exists():
        return None
    with open(f, "r", encoding="utf-8") as fp:
        st = json.load(fp)
    if st.get("schema") != 2:
        sys.exit("Unknown state schema; refusing to continue.")
    return st


def save_state(st: Dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    f = STATE_DIR / "live_state_v14.json"
    if f.exists():
        shutil.copy2(f, STATE_DIR / "live_state_v14.json.bak")
    tmp = STATE_DIR / "live_state_v14.json.tmp"
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
# 6. REPLAY
# ==============================================================================
def macro_bear(rules: Rules, macro_ok: np.ndarray, t: int) -> bool:
    return bool(rules.macro_active_exit and t >= 3 and (not macro_ok[t - 1]) and (not macro_ok[t - 2]) and (not macro_ok[t - 3]))


def replay_position(grid: MarketGrid, rules: Rules, sig, pos: Dict[str, Any], n: int) -> Dict[str, Any]:
    _, _, exit_mat, macro_ok, _, _ = sig
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
    if data_open > 0 and bar_low > 0 and (entry_px < bar_low * 0.95 or entry_px > bar_high * 1.05):
        entry_px = data_open

    units = float(pos["units"])
    tr = make_tranche(grid, c_i, e_bar, entry_px, units, float(pos["cost_inr"]), 0.0,
                      entry_atr_of(grid, pos, c_i, e_bar), rules.sl_mult)
    tranche_eod_update(tr, e_bar, grid, rules.xt, rules.be_trigger_mult, rules.trail_pct_mult, rules.trail_mult)
    out["tranche"] = tr

    has_signal_exit = bool(np.any(exit_mat))
    if "exit_primary" in rules.p or "exit_secondary" in rules.p:
        x1, x2 = rules.p.get("exit_primary", "NONE"), rules.p.get("exit_secondary", "NONE")
        active_exits = [x for x in (x1, x2) if x != "NONE"]
        exit_label = "SIGNAL_EXIT_" + "+".join(active_exits) if active_exits else "SIGNAL_EXIT"
    else:
        exit_label = f"SIGNAL_EXIT_TYPE_{rules.xt}"

    for t in range(e_bar + 1, n):
        if not grid.alive_mat[c_i, t]:
            continue
        reason = tranche_open_exit_reason(tr, t, grid, exit_mat, macro_bear(rules, macro_ok, t), has_signal_exit, rules.max_holding_bars, exit_label)
        if reason:
            out["verdict"], out["overdue_since"] = reason, str(grid.dates[t].date())
            return out
        tranche_eod_update(tr, t, grid, rules.xt, rules.be_trigger_mult, rules.trail_pct_mult, rules.trail_mult)

    out["verdict"] = tranche_open_exit_reason(tr, n, grid, exit_mat, macro_bear(rules, macro_ok, n), has_signal_exit, rules.max_holding_bars, exit_label)
    return out


# ==============================================================================
# 7. PLAN EXECUTION
# ==============================================================================
def apply_plan(st: Dict[str, Any], grid: MarketGrid, rules: Rules, t: int) -> List[str]:
    notes: List[str] = []
    plan = st.get("plan")
    if not plan:
        return notes
    idx = sym_index(grid)
    cash = float(st["cash"])

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

    for o in [o for o in plan["orders"] if o["side"] == "BUY"]:
        if o["id"] in ORDER_SKIPS:
            notes.append(f"{o['id']}: skipped by user (not bought)")
            continue
        if o["coin"] not in idx:
            continue
        c_i = idx[o["coin"]]
        if not grid.alive_mat[c_i, t]:
            notes.append(f"BUY {o['coin']} not filled: no trading in the stock at the open")
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
    has_signal_exit = bool(np.any(exit_mat))
    for it in st["watchlist"]:
        if it["coin"] not in idx:
            continue
        c_i = idx[it["coin"]]
        stopped = grid.close_mat[c_i, t] <= it["shadow_stop"]
        exited = has_signal_exit and bool(exit_mat[c_i, t])
        if stopped or exited:
            continue
        if grid.high_mat[c_i, t] > it["highest_high"]:
            it["highest_high"] = float(grid.high_mat[c_i, t])
        if rules.trail_mult > 0.0:
            it["shadow_stop"] = max(it["shadow_stop"], it["highest_high"] - rules.trail_mult * float(grid.atr14_mat[c_i, t]))
        keep.append(it)
    st["watchlist"] = keep


def warm_start_watchlist(st: Dict[str, Any], grid: MarketGrid, rules: Rules, sig, n: int) -> int:
    if rules.wl_mode == "WL_NONE" or WATCHLIST_WARM_START_BARS <= 0:
        return 0
    _, entry_mat, _, _, _, _ = sig
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
                "coin": coin, "signal_date": None, "trigger_price": float(grid.close_mat[c_i, t - 1]),
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
# 8. MAKE PLAN
# ==============================================================================
def make_plan(st: Dict[str, Any], grid: MarketGrid, rules: Rules, sig, n: int, est_open: Callable[[int], float],
              alive_at_open: Optional[Callable[[int], bool]] = None) -> Dict[str, Any]:
    raw, entry_mat, exit_mat, macro_ok, state_mat, _ = sig
    active_state = state_mat if state_mat is not None else entry_mat
    idx = sym_index(grid)
    cash = float(st["cash"])
    orders: List[Dict[str, Any]] = []
    views: List[Dict[str, Any]] = []
    live_tr: List[Tranche] = []

    has_signal_exit = bool(np.any(exit_mat))
    if "exit_primary" in rules.p or "exit_secondary" in rules.p:
        x1, x2 = rules.p.get("exit_primary", "NONE"), rules.p.get("exit_secondary", "NONE")
        active_exits = [x for x in (x1, x2) if x != "NONE"]
        exit_label = "SIGNAL_EXIT_" + "+".join(active_exits) if active_exits else "SIGNAL_EXIT"
    else:
        exit_label = f"SIGNAL_EXIT_TYPE_{rules.xt}"

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
        if not entry_mat[n - 1, c_i] if entry_mat.ndim == 2 and entry_mat.shape[0] != len(grid.symbols) else not entry_mat[c_i, n - 1]:
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
                                 rules.floor_inr, rules.free_adds, rules.max_pyramid, rules.whole_shares,
                                 rules.pyr_rule, rules.pyr_gap, rules.pyr_prog)
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
# 9. RUN BAR & MANUAL INPUTS
# ==============================================================================
def check_delisted_positions(st: Dict[str, Any], grid: MarketGrid, t: int) -> List[str]:
    warns: List[str] = []
    idx = sym_index(grid)
    for pos in st["positions"]:
        c_i = idx.get(pos["coin"])
        if c_i is not None and grid.delist_mat[c_i, t]:
            warns.append(
                f"{pos['coin']} ({pos['id']}) has 20+ missing bars (possible delisting). "
                f"Position kept in ledger; if sold manually, add to MANUAL_HOLDINGS_REMOVE."
            )
    return warns


def run_bar(st: Dict[str, Any], grid: MarketGrid, rules: Rules, sig, t: int, first_new_bar: bool) -> List[str]:
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
# 10. REPORTING & EMAIL
# ==============================================================================
def build_report(st: Dict[str, Any], meta: Dict[str, Any], grid: MarketGrid, rules: Rules,
                 sig, plan: Dict[str, Any], notes: List[str], warnings: List[str]) -> Tuple[str, str, str]:
    n = len(grid.dates)
    last_date = str(grid.dates[n - 1].date())
    idx = sym_index(grid)
    sells = [o for o in plan["orders"] if o["side"] == "SELL"]
    buys = [o for o in plan["orders"] if o["side"] == "BUY"]
    prefix = "[DRY RUN] " if DRY_RUN else ""
    subject = f"{prefix}NSE Quant V14 Plan: {len(sells)} SELL | {len(buys)} BUY ({last_date})"

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

        held_bars = max(0, n - bar_index(grid, p["entry_date"])) if c_i is not None else 0
        verdict = r["verdict"] if r["verdict"] else ("HOLD" if has_data else "NO DATA")
        _, _, exit_mat, macro_ok, _, _ = sig
        ew_html, ew_txt = _format_exit_watch(tr, c_i, last_px, held_bars, rules, exit_mat, macro_ok, n)

        rows_holdings.append({
            "coin": p["coin"], "entry_date": p["entry_date"], "units": units, "entry_px": entry_px,
            "last_px": last_px, "has_data": has_data, "pnl_pct": pnl_pct, "pnl_inr": pnl_inr,
            "held_bars": held_bars, "verdict": verdict, "overdue": r.get("overdue_since"),
            "note": r.get("note", ""), "exit_watch_html": ew_html, "exit_watch_txt": ew_txt,
        })

    rows_holdings.sort(key=lambda x: (0 if x["verdict"] not in ("HOLD", "NO DATA") else (1 if x["verdict"] == "NO DATA" else 2), -x["pnl_pct"]))

    free_cash = float(st["cash"])
    total_equity = free_cash + total_current_val
    overall_pnl_inr = total_current_val - total_cost
    overall_pnl_pct = (overall_pnl_inr / total_cost * 100.0) if total_cost > 0 else 0.0
    unique_scrips = len({p["coin"] for p in st["positions"]})
    total_tranches = len(st["positions"])
    scrip_summary = f"{unique_scrips} scrips" + (f" ({total_tranches} tranches)" if total_tranches != unique_scrips else "")

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

    badge_status = (
        '<span style="background-color:#f59e0b;color:#111827;padding:3px 8px;border-radius:4px;font-weight:700;font-size:11px;">DRY RUN</span>'
        if DRY_RUN else
        '<span style="background-color:#10b981;color:#ffffff;padding:3px 8px;border-radius:4px;font-weight:700;font-size:11px;">LIVE RUN</span>'
    )
    pnl_color = "#10b981" if overall_pnl_inr >= 0 else "#ef4444"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>{subject}</title></head>
    <body style="margin:0;padding:24px;background-color:#0b0f19;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#e2e8f0;">
      <table align="center" border="0" cellpadding="0" cellspacing="0" width="100%" style="max-width:980px;margin:auto;">
        <tr>
          <td style="padding:16px 20px;background:linear-gradient(135deg,#1e293b 0%,#0f172a 100%);border-radius:12px;border:1px solid #334155;">
            <table width="100%" border="0" cellpadding="0" cellspacing="0">
              <tr>
                <td>
                  <div style="font-size:11px;color:#94a3b8;letter-spacing:1.5px;text-transform:uppercase;font-weight:700;">Quant Execution Terminal • {meta['version']}</div>
                  <div style="font-size:22px;color:#ffffff;font-weight:800;margin-top:4px;">Daily Pre-Market Execution Plan</div>
                  <div style="font-size:12px;color:#64748b;margin-top:4px;">Bar Date: <span style="color:#cbd5e1;">{last_date}</span> • Param Hash: <span style="color:#cbd5e1;">{meta['param_hash']}</span></div>
                </td>
                <td align="right" valign="top">{badge_status}</td>
              </tr>
            </table>
          </td>
        </tr>
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

    if sells or buys:
        html += """
        <tr>
          <td style="padding-top:20px;">
            <div style="font-size:13px;font-weight:700;color:#f8fafc;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:8px;">⚡ Orders to Execute at Market Open (09:15 AM)</div>
            <table width="100%" border="0" cellpadding="8" cellspacing="0" style="background:#1e293b;border:1px solid #334155;border-radius:8px;font-size:13px;border-collapse:collapse;">
              <thead>
                <tr style="background:#0f172a;color:#94a3b8;border-bottom:1px solid #334155;text-align:left;font-size:11px;text-transform:uppercase;">
                  <th style="padding:10px 12px;">Side</th><th style="padding:10px 12px;">Scrip</th><th style="padding:10px 12px;">Quantity</th><th style="padding:10px 12px;">Est. Price</th><th style="padding:10px 12px;">Order Value</th><th style="padding:10px 12px;">Trigger / Exit Reason</th>
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

    html += """
        <tr>
          <td style="padding-top:24px;">
            <div style="font-size:13px;font-weight:700;color:#f8fafc;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:8px;">📊 Portfolio Holdings &amp; Exit Watch</div>
            <table width="100%" border="0" cellpadding="8" cellspacing="0" style="background:#1e293b;border:1px solid #334155;border-radius:8px;font-size:12px;border-collapse:collapse;">
              <thead>
                <tr style="background:#0f172a;color:#94a3b8;border-bottom:1px solid #334155;text-align:left;font-size:11px;text-transform:uppercase;">
                  <th style="padding:10px 10px;">Scrip</th><th style="padding:10px 10px;">Entry Date</th><th style="padding:10px 10px;text-align:right;">Qty</th><th style="padding:10px 10px;text-align:right;">Entry Px</th><th style="padding:10px 10px;text-align:right;">Last Px</th><th style="padding:10px 10px;text-align:right;">Net P&amp;L</th><th style="padding:10px 10px;">Exit Watch (Active Config Rules)</th><th style="padding:10px 10px;text-align:center;">Action</th>
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
                  <th style="padding:10px 10px;width:50px;">Rank</th><th style="padding:10px 10px;">Scrip</th><th style="padding:10px 10px;">Signal Date</th><th style="padding:10px 10px;text-align:right;">Trigger Px</th><th style="padding:10px 10px;text-align:right;">Shadow Stop</th><th style="padding:10px 10px;text-align:right;">Breakout Score</th><th style="padding:10px 10px;text-align:right;">ATR (14)</th>
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
# 11. MAIN
# ==============================================================================
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rules, meta = load_config()
    cfg_hash = f"{meta['version']}:{meta['param_hash']}"
    logging.info(f"{'DRY RUN. ' if DRY_RUN else ''}Config {meta['path']} | {meta['status']} | hash {meta['param_hash']}")

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
    sig = compile_signals_composable(grid, rules.p)
    n = len(grid.dates)
    now = dt.datetime.now(IST)
    warnings: List[str] = []

    if now.weekday() < 5 and (now.hour + now.minute / 60.0) < MARKET_CLOSE_IST_HOUR:
        warnings.append(
            f"Run started at {now:%H:%M} IST (before {MARKET_CLOSE_IST_HOUR:.0f}:00 IST market close). "
            f"Today's incomplete bar was excluded ({DROPPED_TODAY_BARS} tickers had intraday prints); plan is based on {grid.dates[-1].date()}."
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
            notes.append(f"Watchlist rebuilt from the last {WATCHLIST_WARM_START_BARS} bars: {n_wl} items.")

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