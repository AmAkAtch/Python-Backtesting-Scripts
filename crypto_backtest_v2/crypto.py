#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crypto_backtest_optimizer.py
================================================================================
Multi-Asset Cryptocurrency Trend-Following Backtester & Hyperparameter
Optimizer with configurable coin universe capping and dynamic start date
alignment (manual date or >=60% coin availability).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Sequence

import numpy as np
import pandas as pd

from numba import njit, prange

import optuna
from optuna.samplers import TPESampler, RandomSampler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numba")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# ############################  USER SETTINGS  ################################
# ==============================================================================
# --- Optimization run ---
N_TRIALS = 20000                # Optuna trials to run
SAMPLER = "tpe"                 # "tpe" (TPESampler) or "random" (RandomSampler)
N_STARTUP_TRIALS = 100          # random trials before TPE starts modeling
N_JOBS = 1                      # parallel Optuna workers (>1 uses multiple CPU cores)
SEED = 42                       # reproducibility seed
PROGRESS_EVERY_N_TRIALS = 10    # print a progress/ETA line at least this often

# --- Data, Universe & Start Date Controls ---
SYNTHETIC = False               # True = generate fake data, no network needed
N_COINS_TARGET = 100            # tickers to attempt to fetch (real mode) or synthesize
MAX_COINS: Optional[int] = 100 # e.g. 50 to test top 50 coins by volume; None = all fetched
START_DATE: Optional[str] = None # e.g. "2021-01-01"; None = auto-detect via AUTO_START_MIN_COINS_PCT
AUTO_START_MIN_COINS_PCT = 0.51 # 0.60 = start date when >=60% of universe is listed (0.0 to disable)

MIN_LIQUID_ASSETS = 70          # baseline liquid altcoins floor (auto-adjusted if MAX_COINS < 70)
CACHE_DIR = "./crypto_cache"
UNIVERSE_FILE = None            # e.g. "my_tickers.txt" to override DEFAULT_UNIVERSE
REFRESH_CACHE = False           # True = ignore cache and re-download everything

# --- Capital, tickets & friction model ---
INITIAL_CAPITAL = 1000.0
DCA_AMOUNT = 1000.0
DCA_INTERVAL_DAYS = 30
TICKET_SIZE = 1000.0
TAKER_FEE = 0.0010              # 0.10%
TAX_RATE = 0.01                 # 1.0% on gross sale proceeds
SLIPPAGE_BASE = 0.0035          # 0.35% base
SLIPPAGE_ATR_MULT = 0.05        # + 5% * (ATR14 / Close)
LIQUIDITY_FLOOR_USD = 3_000_000.0   # 30-day avg dollar volume required for a NEW entry fill
LIQUIDITY_WINDOW = 30

# --- Priority / Must-Have Coins ---
# Shorthand ("BTC") or full Yahoo ticker ("BTC-USD") both work
PRIORITY_COINS = ["BTC", "ETH", "SOL", "DOGE", "XRP"]
ALLOW_BTC_TRADING = True       # Set True so BTC can be bought/sold like any other altcoin

# --- Scoring ---
MIN_EXPECTED_TRADES = 150       # Activity_Damper saturates to 1.0 at this trade count

# --- Output & logging ---
OUTPUT_DIR = "./results"
VERBOSITY = "INFO"              # DEBUG, INFO, WARNING, ...
SELF_TEST = False               # True = force a fast synthetic smoke test
# ==============================================================================

# ==============================================================================
# LOGGING
# ==============================================================================
logger = logging.getLogger("crypto_backtest")


def setup_logging(verbosity: str = "INFO", log_file: Optional[str] = None) -> None:
    logger.setLevel(getattr(logging, verbosity.upper(), logging.INFO))
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)


# ==============================================================================
# ENUMS
# ==============================================================================
MA_SMA, MA_EMA, MA_DEMA, MA_WMA, MA_RMA = 0, 1, 2, 3, 4
MA_TYPE_NAMES = {0: "SMA", 1: "EMA", 2: "DEMA", 3: "WMA", 4: "SMMA/RMA"}

ENTRY_MA_BREAKOUT, ENTRY_RSI_XOVER, ENTRY_MA_XOVER = 0, 1, 2

EXIT_HYBRID_ATR = 0
EXIT_PCT_TRAIL = 1
EXIT_ATR_TRAIL = 2
EXIT_MA_CROSSUNDER = 3
EXIT_RSI_CROSSUNDER = 4
EXIT_MA_XOVER_EXIT = 5

WL_DEEPEST_DISCOUNT, WL_CLOSEST_BREAKOUT, WL_STRONGEST_MOMENTUM, WL_FCFS = 0, 1, 2, 3

MAX_PYRAMID_LAYERS_CAP = 4

# ==============================================================================
# UNIVERSE DEFINITION
# ==============================================================================
STABLECOIN_AND_WRAPPED_BLACKLIST = {
    # USD & Fiat Stables
    "USDT-USD", "USDC-USD", "BUSD-USD", "DAI-USD", "TUSD-USD", "USDD-USD",
    "FDUSD-USD", "USDP-USD", "GUSD-USD", "USTC-USD", "FRAX-USD", "LUSD-USD",
    "SUSD-USD", "EURS-USD", "EURT-USD", "USDN-USD", "CUSD-USD", "OUSD-USD",
    "MIM-USD", "USDX-USD", "RSV-USD", "HUSD-USD", "PYUSD-USD", "USDE-USD",
    "USD0-USD", "USDS-USD", "GHO-USD", "CRVUSD-USD", "RLUSD-USD", "TBTC-USD",
    # Wrapped & Liquid Staking / Restaking Tokens
    "WBTC-USD", "WETH-USD", "STETH-USD", "WSTETH-USD", "CBETH-USD", "RETH-USD",
    "WEETH-USD", "WBETH-USD", "SFRXETH-USD", "EZETH-USD", "MSOL-USD", "JITOSOL-USD",
}

MACRO_REFERENCE_TICKER = "BTC-USD"

DEFAULT_UNIVERSE: List[str] = sorted(set([
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
    "DOGE-USD", "TRX-USD", "AVAX-USD", "DOT-USD", "LINK-USD", "LTC-USD",
    "BCH-USD", "ATOM-USD", "XLM-USD", "ETC-USD", "FIL-USD", "APT-USD",
    "ARB-USD", "OP-USD", "NEAR-USD", "ICP-USD", "HBAR-USD", "VET-USD",
    "ALGO-USD", "EGLD-USD", "SAND-USD", "MANA-USD", "AXS-USD", "THETA-USD",
    "FTM-USD", "EOS-USD", "XTZ-USD", "CHZ-USD", "ENJ-USD", "ZEC-USD",
    "DASH-USD", "XMR-USD", "KSM-USD", "WAVES-USD", "COMP-USD", "MKR-USD",
    "SNX-USD", "YFI-USD", "UNI-USD", "AAVE-USD", "CRV-USD", "SUSHI-USD",
    "1INCH-USD", "GRT-USD", "RUNE-USD", "KAVA-USD", "ZIL-USD", "MIOTA-USD",
    "NEO-USD", "QTUM-USD", "ONT-USD", "ICX-USD", "BAT-USD", "ZRX-USD",
    "OMG-USD", "LRC-USD", "STORJ-USD", "ANKR-USD", "CELR-USD", "CTSI-USD",
    "SKL-USD", "OCEAN-USD", "BAND-USD", "REN-USD", "BAL-USD", "KNC-USD",
    "UMA-USD", "RSR-USD", "CVC-USD", "DENT-USD", "HOT-USD", "SC-USD",
    "ZEN-USD", "RVN-USD", "DGB-USD", "NANO-USD", "LSK-USD", "FLOW-USD",
    "GALA-USD", "IMX-USD", "GMT-USD", "APE-USD", "DYDX-USD", "ENS-USD",
    "LDO-USD", "INJ-USD", "STX-USD", "ROSE-USD", "ONE-USD", "CELO-USD",
    "KDA-USD", "GLMR-USD", "MOVR-USD", "ASTR-USD", "MINA-USD", "XEC-USD",
    "IOTX-USD", "ANT-USD", "MTL-USD", "POWR-USD", "FUN-USD", "LOOM-USD",
    "GNO-USD", "BNT-USD", "NMR-USD", "OXT-USD", "SXP-USD", "PERP-USD",
    "MASK-USD", "YGG-USD", "ALICE-USD", "TLM-USD", "SLP-USD", "ILV-USD",
    "GHST-USD", "AUDIO-USD", "RLC-USD", "STMX-USD", "CTXC-USD", "WAXP-USD",
    "ARDR-USD", "STRAX-USD", "VTHO-USD", "WTC-USD", "ELF-USD", "POLY-USD",
    "REQ-USD", "DATA-USD", "DUSK-USD", "ORN-USD", "TRB-USD", "FET-USD",
    "AGIX-USD", "RLY-USD", "API3-USD", "FORTH-USD", "POND-USD", "ALPACA-USD",
    "BADGER-USD", "FARM-USD", "BOND-USD", "DODO-USD", "MIR-USD", "LINA-USD",
    "REEF-USD", "BEL-USD", "TVK-USD", "TWT-USD", "HARD-USD", "DEXE-USD",
    "POLS-USD", "PHA-USD", "PERL-USD", "TROY-USD", "DOCK-USD", "WING-USD",
    "FOR-USD", "UFT-USD", "VITE-USD", "MDT-USD", "ARPA-USD", "COTI-USD",
    "CHR-USD", "STPT-USD", "KEY-USD", "WAN-USD", "FIRO-USD", "NKN-USD",
    "LUNC-USD", "LUNA-USD", "OSMO-USD", "JUNO-USD", "SCRT-USD", "AKT-USD",
    "AR-USD", "HNT-USD", "IOTA-USD", "KAS-USD", "SUI-USD", "SEI-USD",
    "TIA-USD", "PYTH-USD", "JTO-USD", "WLD-USD", "RNDR-USD", "AGLD-USD",
    "GMX-USD", "CFX-USD", "FLR-USD", "ID-USD", "HOOK-USD", "MAGIC-USD",
    "HIGH-USD", "SSV-USD", "RDNT-USD", "BLUR-USD", "LOOKS-USD", "JASMY-USD",
    "ACH-USD", "SUPER-USD", "VOXEL-USD", "GTC-USD", "T-USD", "BICO-USD",
    "SPELL-USD", "TRU-USD", "QUICK-USD", "MULTI-USD", "ALPHA-USD",
    "RAY-USD", "ORCA-USD", "STG-USD", "JOE-USD", "SFP-USD", "BAKE-USD",
    "AUCTION-USD", "XVS-USD", "CREAM-USD", "PROM-USD", "OGN-USD", "NULS-USD",
    "FIDA-USD", "MEDIA-USD", "STEP-USD", "PORTO-USD", "LAZIO-USD", "PSG-USD",
    "BAR-USD", "JUV-USD", "ATM-USD", "ACM-USD", "CITY-USD", "FLM-USD",
    "DEGO-USD", "LIT-USD", "AVA-USD", "MBOX-USD", "IDEX-USD", "PUNDIX-USD",
    "COS-USD", "CKB-USD", "ARK-USD", "MTH-USD", "AION-USD", "WABI-USD",
    "AMB-USD", "BLZ-USD", "NAS-USD", "GAS-USD", "LEND-USD", "NULS-USD",
]))


# ==============================================================================
# CONFIG DATACLASSES
# ==============================================================================
@dataclass
class RunConfig:
    mode: str = "single"
    n_trials: int = N_TRIALS
    n_coins_target: int = N_COINS_TARGET
    max_coins: Optional[int] = MAX_COINS
    start_date: Optional[str] = START_DATE
    auto_start_min_coins_pct: float = AUTO_START_MIN_COINS_PCT
    min_liquid_assets: int = MIN_LIQUID_ASSETS
    cache_dir: str = CACHE_DIR
    universe_file: Optional[str] = UNIVERSE_FILE
    refresh_cache: bool = REFRESH_CACHE
    synthetic: bool = SYNTHETIC
    seed: int = SEED
    sampler: str = SAMPLER
    n_startup_trials: int = N_STARTUP_TRIALS
    n_jobs: int = N_JOBS
    initial_capital: float = INITIAL_CAPITAL
    dca_amount: float = DCA_AMOUNT
    dca_interval_days: int = DCA_INTERVAL_DAYS
    ticket_size: float = TICKET_SIZE
    taker_fee: float = TAKER_FEE
    tax_rate: float = TAX_RATE
    slippage_base: float = SLIPPAGE_BASE
    slippage_atr_mult: float = SLIPPAGE_ATR_MULT
    liquidity_floor_usd: float = LIQUIDITY_FLOOR_USD
    liquidity_window: int = LIQUIDITY_WINDOW
    min_expected_trades: int = MIN_EXPECTED_TRADES
    output_dir: str = OUTPUT_DIR
    verbosity: str = VERBOSITY
    self_test: bool = SELF_TEST
    progress_every: int = PROGRESS_EVERY_N_TRIALS
    priority_coins: List[str] = field(default_factory=lambda: list(PRIORITY_COINS))
    allow_btc_trading: bool = ALLOW_BTC_TRADING

    def validate(self) -> None:
        if self.mode != "single":
            raise ValueError("--mode 'walkforward' is not implemented in this build.")
        if self.sampler not in ("tpe", "random"):
            raise ValueError("--sampler must be 'tpe' or 'random'")
        if self.n_trials < 1:
            raise ValueError("--n-trials must be >= 1")
        if self.max_coins is not None:
            if self.max_coins < 1:
                raise ValueError("--max-coins must be >= 1")
            if self.min_liquid_assets > self.max_coins:
                self.min_liquid_assets = self.max_coins
        if not (0.0 <= self.auto_start_min_coins_pct <= 1.0):
            raise ValueError("--auto-start-pct must be between 0.0 and 1.0")


@dataclass
class StrategyParams:
    entry_type: int = ENTRY_MA_BREAKOUT
    entry_ma_len: int = 50
    entry_ma_type: int = MA_SMA

    rsi_f_len: int = 14
    rsi_f_smt: int = 5
    rsi_s_len: int = 28
    rsi_s_smt: int = 10

    xover_short_len: int = 50
    xover_short_type: int = MA_SMA
    xover_gap: int = 50
    xover_long_type: int = MA_SMA

    use_btc_entry_gate: bool = True
    btc_ma_len: int = 100
    btc_ma_type: int = MA_SMA

    adx_thresh: float = 20.0

    use_rsi_trend_filter: bool = False
    rsi_trend_ma_len: int = 100
    rsi_trend_ma_type: int = MA_SMA

    use_latched_entry: bool = False
    max_pyramid_layers: int = 1

    exit_type: int = EXIT_HYBRID_ATR
    tp_mult: float = 20.0
    trail_mult: float = 6.0
    sl_mult: float = 3.0
    trail_pct: float = 15.0
    exit_atr_mult: float = 3.0
    exit_ma_len: int = 50
    exit_ma_type: int = MA_SMA

    use_btc_exit_override: bool = False
    watchlist_rank_mode: int = WL_DEEPEST_DISCOUNT

    def xover_long_len(self) -> int:
        return min(300, self.xover_short_len + self.xover_gap)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["xover_long_len"] = self.xover_long_len()
        return d


@dataclass
class AlignedData:
    dates: np.ndarray               # datetime64[D], shape [T]
    days_since_start: np.ndarray    # int32, shape [T]
    tickers: List[str]              # length N
    O: np.ndarray                   # float64 [T, N]
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    V: np.ndarray
    listed: np.ndarray              # bool [T, N]
    btc_close: np.ndarray           # float64 [T]

    @property
    def T(self) -> int:
        return self.C.shape[0]

    @property
    def N(self) -> int:
        return self.C.shape[1]


# ==============================================================================
# INDICATORS
# ==============================================================================
@njit(parallel=True, cache=True)
def sma_2d(price_2d, length):
    T, N = price_2d.shape
    out = np.full((T, N), np.nan)
    for j in prange(N):
        buf = np.zeros(length)
        count = 0
        buf_idx = 0
        window_sum = 0.0
        for i in range(T):
            v = price_2d[i, j]
            if np.isnan(v):
                window_sum = 0.0
                count = 0
                buf_idx = 0
                continue
            if count < length:
                buf[count] = v
                window_sum += v
                count += 1
                if count == length:
                    out[i, j] = window_sum / length
            else:
                oldest = buf[buf_idx]
                window_sum += v - oldest
                buf[buf_idx] = v
                buf_idx = (buf_idx + 1) % length
                out[i, j] = window_sum / length
    return out


@njit(parallel=True, cache=True)
def ema_2d(price_2d, length):
    T, N = price_2d.shape
    out = np.full((T, N), np.nan)
    alpha = 2.0 / (length + 1.0)
    for j in prange(N):
        seed_sum = 0.0
        count = 0
        ema_val = 0.0
        seeded = False
        for i in range(T):
            v = price_2d[i, j]
            if np.isnan(v):
                seed_sum = 0.0
                count = 0
                seeded = False
                continue
            if not seeded:
                seed_sum += v
                count += 1
                if count == length:
                    ema_val = seed_sum / length
                    out[i, j] = ema_val
                    seeded = True
            else:
                ema_val = alpha * v + (1.0 - alpha) * ema_val
                out[i, j] = ema_val
    return out


@njit(parallel=True, cache=True)
def rma_2d(price_2d, length):
    T, N = price_2d.shape
    out = np.full((T, N), np.nan)
    alpha = 1.0 / length
    for j in prange(N):
        seed_sum = 0.0
        count = 0
        val = 0.0
        seeded = False
        for i in range(T):
            v = price_2d[i, j]
            if np.isnan(v):
                seed_sum = 0.0
                count = 0
                seeded = False
                continue
            if not seeded:
                seed_sum += v
                count += 1
                if count == length:
                    val = seed_sum / length
                    out[i, j] = val
                    seeded = True
            else:
                val = alpha * v + (1.0 - alpha) * val
                out[i, j] = val
    return out


@njit(parallel=True, cache=True)
def wma_2d(price_2d, length):
    T, N = price_2d.shape
    out = np.full((T, N), np.nan)
    denom = length * (length + 1) / 2.0
    for j in prange(N):
        window = np.zeros(length)
        count = 0
        for i in range(T):
            v = price_2d[i, j]
            if np.isnan(v):
                count = 0
                continue
            if count < length:
                window[count] = v
                count += 1
            else:
                for k in range(length - 1):
                    window[k] = window[k + 1]
                window[length - 1] = v
            if count == length:
                s = 0.0
                for k in range(length):
                    s += window[k] * (k + 1)
                out[i, j] = s / denom
    return out


@njit(parallel=True, cache=True)
def dema_core_2d(ema1, ema2):
    T, N = ema1.shape
    out = np.full((T, N), np.nan)
    for j in prange(N):
        for i in range(T):
            a = ema1[i, j]
            b = ema2[i, j]
            if not (np.isnan(a) or np.isnan(b)):
                out[i, j] = 2.0 * a - b
    return out


def dema_2d(price_2d: np.ndarray, length: int) -> np.ndarray:
    ema1 = ema_2d(price_2d, length)
    ema2 = ema_2d(ema1, length)
    return dema_core_2d(ema1, ema2)


def compute_ma(price_2d: np.ndarray, length: int, ma_type: int) -> np.ndarray:
    length = max(1, int(length))
    if ma_type == MA_SMA:
        return sma_2d(price_2d, length)
    elif ma_type == MA_EMA:
        return ema_2d(price_2d, length)
    elif ma_type == MA_DEMA:
        return dema_2d(price_2d, length)
    elif ma_type == MA_WMA:
        return wma_2d(price_2d, length)
    elif ma_type == MA_RMA:
        return rma_2d(price_2d, length)
    raise ValueError(f"Unknown ma_type: {ma_type}")


def compute_ma_1d(price_1d: np.ndarray, length: int, ma_type: int) -> np.ndarray:
    return compute_ma(price_1d.reshape(-1, 1), length, ma_type)[:, 0]


@njit(parallel=True, cache=True)
def rsi_2d(price_2d, length):
    T, N = price_2d.shape
    out = np.full((T, N), np.nan)
    for j in prange(N):
        seed_gain = 0.0
        seed_loss = 0.0
        count = 0
        avg_gain = 0.0
        avg_loss = 0.0
        seeded = False
        prev_valid = False
        prev_price = 0.0
        for i in range(T):
            v = price_2d[i, j]
            if np.isnan(v):
                count = 0
                seeded = False
                prev_valid = False
                seed_gain = 0.0
                seed_loss = 0.0
                continue
            if not prev_valid:
                prev_price = v
                prev_valid = True
                continue
            change = v - prev_price
            prev_price = v
            gain = change if change > 0.0 else 0.0
            loss = -change if change < 0.0 else 0.0
            if not seeded:
                seed_gain += gain
                seed_loss += loss
                count += 1
                if count == length:
                    avg_gain = seed_gain / length
                    avg_loss = seed_loss / length
                    seeded = True
                    out[i, j] = 100.0 if avg_loss == 0.0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
            else:
                avg_gain = (avg_gain * (length - 1) + gain) / length
                avg_loss = (avg_loss * (length - 1) + loss) / length
                out[i, j] = 100.0 if avg_loss == 0.0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def smoothed_rsi(price_2d: np.ndarray, rsi_len: int, smooth_len: int) -> np.ndarray:
    raw = rsi_2d(price_2d, max(1, int(rsi_len)))
    return sma_2d(raw, max(1, int(smooth_len)))


@njit(parallel=True, cache=True)
def true_range_2d(H, L, C):
    T, N = H.shape
    tr = np.full((T, N), np.nan)
    for j in prange(N):
        prev_close = 0.0
        prev_valid = False
        for i in range(T):
            h, l, c = H[i, j], L[i, j], C[i, j]
            if np.isnan(h) or np.isnan(l) or np.isnan(c):
                prev_valid = False
                continue
            if not prev_valid:
                tr[i, j] = h - l
            else:
                a = h - l
                b = abs(h - prev_close)
                d = abs(l - prev_close)
                m = a
                if b > m:
                    m = b
                if d > m:
                    m = d
                tr[i, j] = m
            prev_close = c
            prev_valid = True
    return tr


def atr_2d(H: np.ndarray, L: np.ndarray, C: np.ndarray, length: int = 14) -> np.ndarray:
    tr = true_range_2d(H, L, C)
    return rma_2d(tr, length)


@njit(parallel=True, cache=True)
def directional_moves_2d(H, L):
    T, N = H.shape
    plus_dm = np.full((T, N), np.nan)
    minus_dm = np.full((T, N), np.nan)
    for j in prange(N):
        prev_h = 0.0
        prev_l = 0.0
        prev_valid = False
        for i in range(T):
            h, l = H[i, j], L[i, j]
            if np.isnan(h) or np.isnan(l):
                prev_valid = False
                continue
            if not prev_valid:
                plus_dm[i, j] = 0.0
                minus_dm[i, j] = 0.0
            else:
                up = h - prev_h
                down = prev_l - l
                pdm = up if (up > down and up > 0.0) else 0.0
                mdm = down if (down > up and down > 0.0) else 0.0
                plus_dm[i, j] = pdm
                minus_dm[i, j] = mdm
            prev_h, prev_l = h, l
            prev_valid = True
    return plus_dm, minus_dm


@njit(parallel=True, cache=True)
def dx_from_smoothed_2d(sm_tr, sm_pdm, sm_mdm):
    T, N = sm_tr.shape
    dx = np.full((T, N), np.nan)
    for j in prange(N):
        for i in range(T):
            st = sm_tr[i, j]
            if np.isnan(st):
                continue
            pdi = 100.0 * sm_pdm[i, j] / st if st != 0.0 else 0.0
            mdi = 100.0 * sm_mdm[i, j] / st if st != 0.0 else 0.0
            denom = pdi + mdi
            dx[i, j] = 100.0 * abs(pdi - mdi) / denom if denom != 0.0 else 0.0
    return dx


def adx_2d(H: np.ndarray, L: np.ndarray, C: np.ndarray, length: int = 14) -> np.ndarray:
    plus_dm, minus_dm = directional_moves_2d(H, L)
    tr = true_range_2d(H, L, C)
    sm_tr = rma_2d(tr, length)
    sm_pdm = rma_2d(plus_dm, length)
    sm_mdm = rma_2d(minus_dm, length)
    dx = dx_from_smoothed_2d(sm_tr, sm_pdm, sm_mdm)
    return rma_2d(dx, length)


@njit(parallel=True, cache=True)
def rolling_dollar_volume_ok_2d(C, V, window, floor_usd):
    T, N = C.shape
    out = np.zeros((T, N), dtype=np.bool_)
    for j in prange(N):
        buf = np.zeros(window)
        count = 0
        buf_idx = 0
        s = 0.0
        for i in range(T):
            c, v = C[i, j], V[i, j]
            if np.isnan(c) or np.isnan(v):
                s = 0.0
                count = 0
                buf_idx = 0
                continue
            dv = c * v
            if count < window:
                buf[count] = dv
                s += dv
                count += 1
                if count == window:
                    out[i, j] = (s / window) >= floor_usd
            else:
                oldest = buf[buf_idx]
                s += dv - oldest
                buf[buf_idx] = dv
                buf_idx = (buf_idx + 1) % window
                out[i, j] = (s / window) >= floor_usd
    return out


# ==============================================================================
# DATA INGESTION, CACHING & ALIGNMENT
# ==============================================================================
def is_blacklisted(ticker: str, allow_btc: bool = False) -> bool:
    t = ticker.upper()
    if t in STABLECOIN_AND_WRAPPED_BLACKLIST:
        return True
    if not allow_btc and t == MACRO_REFERENCE_TICKER:
        return True
    return False

def is_noise_or_stablecoin(df: pd.DataFrame, ticker: str) -> bool:
    """
    Algorithmic filter: rejects coins that trade pegged to $1 or lack the 
    volatility required for trend-following to overcome friction.
    """
    if df.empty or len(df) < 30 or "Close" not in df.columns:
        return True

    c = df["Close"].dropna()
    if len(c) < 30:
        return True

    # Check 1: Dollar Peg Trap (Median price near $1.00 with narrow range)
    median_price = float(c.median())
    if 0.90 <= median_price <= 1.10:
        p05 = float(c.quantile(0.05))
        p95 = float(c.quantile(0.95))
        # If 90% of all historical prints stay within [0.93, 1.07], it's a stablecoin
        if p05 >= 0.93 and p95 <= 1.07:
            logger.debug(f"Dropped {ticker}: flagged as dynamic stablecoin (median=${median_price:.3f})")
            return True

    # Check 2: Minimum Volatility Floor
    # Trend-following needs volatility to outrun 0.45% round-trip fees/slippage.
    # Annualized daily volatility below 25% indicates a dead or pegged instrument.
    daily_returns = c.pct_change().dropna()
    if len(daily_returns) > 30:
        ann_vol = float(daily_returns.std() * np.sqrt(365))
        if ann_vol < 0.25:
            logger.debug(f"Dropped {ticker}: low volatility ({ann_vol*100:.1f}% ann. vol)")
            return True

    return False

def standardize_ticker(ticker: str) -> str:
    """Converts shorthand like 'SOL' or 'solana' to Yahoo format 'SOL-USD'."""
    t = ticker.strip().upper()
    if t == "SOLANA":
        t = "SOL"
    if not t.endswith("-USD") and "-" not in t:
        return f"{t}-USD"
    return t


def verify_priority_coins(final_tickers: List[str], priority_list: List[str]) -> None:
    """Verifies that mandatory coins made it into the final aligned universe."""
    universe_set = set(final_tickers)
    std_priority = [standardize_ticker(c) for c in priority_list]
    
    found = [c for c in std_priority if c in universe_set]
    missing = [c for c in std_priority if c not in universe_set]
    
    logger.info("=" * 60)
    logger.info(f"PRIORITY COIN AUDIT: {len(found)}/{len(std_priority)} present in active backtest")
    logger.info(f"  Present ({len(found)}): {', '.join(found)}")
    if missing:
        logger.warning(f"  MISSING ({len(missing)}): {', '.join(missing)}")
        logger.warning("  (Check if missing coins failed Yahoo download or were cut by liquidity filters)")
    logger.info("=" * 60)

def load_universe_list(cfg: RunConfig) -> List[str]:
    if cfg.universe_file:
        with open(cfg.universe_file) as f:
            tickers = [ln.strip().upper() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        logger.info(f"Loaded {len(tickers)} tickers from {cfg.universe_file}")
    else:
        tickers = list(DEFAULT_UNIVERSE)
    if MACRO_REFERENCE_TICKER not in tickers:
        tickers.append(MACRO_REFERENCE_TICKER)
    return sorted(set(tickers))


def _cache_path(cache_dir: str, ticker: str) -> Path:
    return Path(cache_dir) / f"{ticker.replace('/', '_')}.parquet"


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path)
        return
    except Exception:
        pass
    try:
        df.reset_index().to_feather(path.with_suffix(".feather"))
        return
    except Exception:
        pass
    df.to_csv(path.with_suffix(".csv"))


def _load_cache(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists():
            return pd.read_parquet(path)
        fpath = path.with_suffix(".feather")
        if fpath.exists():
            df = pd.read_feather(fpath)
            return df.set_index(df.columns[0])
        cpath = path.with_suffix(".csv")
        if cpath.exists():
            return pd.read_csv(cpath, index_col=0, parse_dates=True)
    except Exception:
        pass
    return None


def fetch_from_yfinance(tickers: List[str], cache_dir: str, refresh: bool) -> Dict[str, pd.DataFrame]:
    import yfinance as yf

    out: Dict[str, pd.DataFrame] = {}
    to_download: List[str] = []
    for t in tickers:
        path = _cache_path(cache_dir, t)
        if not refresh:
            cached = _load_cache(path)
            if cached is not None and len(cached) > 30:
                out[t] = cached
                continue
        to_download.append(t)

    if not to_download:
        logger.info(f"All {len(out)} tickers loaded from cache ({cache_dir}).")
        return out

    logger.info(f"Fetching {len(to_download)} tickers from Yahoo Finance...")
    batch_size = 20
    for bi in range(0, len(to_download), batch_size):
        batch = to_download[bi:bi + batch_size]
        data = None
        try:
            data = yf.download(batch, period="max", interval="1d", group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
        except Exception as e:
            logger.warning(f"Batch fetch error ({e}); falling back per-ticker")

        for t in batch:
            df = None
            try:
                if data is not None and isinstance(data.columns, pd.MultiIndex):
                    if t in data.columns.get_level_values(0):
                        sub = data[t].dropna(how="all")
                        if not sub.empty:
                            df = sub[["Open", "High", "Low", "Close", "Volume"]]
                elif data is not None and len(batch) == 1:
                    sub = data.dropna(how="all")
                    if not sub.empty:
                        df = sub[["Open", "High", "Low", "Close", "Volume"]]
            except Exception:
                pass

            if df is None or df.empty:
                try:
                    h = yf.Ticker(t).history(period="max", interval="1d", auto_adjust=True)
                    if not h.empty:
                        df = h[["Open", "High", "Low", "Close", "Volume"]]
                except Exception:
                    df = None

            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df = df[~df.index.duplicated(keep="last")].sort_index()
                _save_cache(df, _cache_path(cache_dir, t))
                out[t] = df
                logger.info(f"  {t}: {len(df)} bars")
            else:
                logger.warning(f"  {t}: no usable data returned (skipping)")
        time.sleep(0.3)
    return out


def generate_synthetic_universe(n_coins: int, n_days: int, seed: int) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")

    btc_ret = rng.normal(0.0006, 0.035, n_days)
    btc_ret[int(n_days * 0.30):int(n_days * 0.45)] -= 0.012
    btc_ret[int(n_days * 0.60):int(n_days * 0.75)] += 0.007
    btc_close = 20000.0 * np.exp(np.cumsum(btc_ret))
    btc_open = np.roll(btc_close, 1); btc_open[0] = btc_close[0]
    btc_high = np.maximum(btc_open, btc_close) * (1 + np.abs(rng.normal(0, 0.01, n_days)))
    btc_low = np.minimum(btc_open, btc_close) * (1 - np.abs(rng.normal(0, 0.01, n_days)))
    btc_vol = rng.uniform(2e9, 6e9, n_days)
    btc_df = pd.DataFrame({"Open": btc_open, "High": btc_high, "Low": btc_low,
                            "Close": btc_close, "Volume": btc_vol}, index=dates)

    coins: Dict[str, pd.DataFrame] = {}
    for k in range(n_coins):
        beta = rng.uniform(0.3, 1.6)
        idio_vol = rng.uniform(0.02, 0.09)
        drift = rng.normal(0.0002, 0.0015)
        start_offset = int(rng.integers(0, max(1, n_days // 3)))
        length = n_days - start_offset
        if length < 60:
            continue
        idio = rng.normal(drift, idio_vol, length)
        ret = beta * btc_ret[start_offset:] + idio
        base_price = rng.uniform(0.01, 500.0)
        close = base_price * np.exp(np.cumsum(ret))
        openp = np.roll(close, 1); openp[0] = close[0]
        high = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.02, length)))
        low = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.02, length)))
        tier = rng.choice([0, 1, 2], p=[0.25, 0.35, 0.40])
        vol_scale = [3e4, 3e5, 4e6][tier]
        volume = rng.lognormal(mean=math.log(vol_scale), sigma=0.6, size=length)
        idx = dates[start_offset:]
        if rng.random() < 0.08:
            cutoff = int(rng.integers(length // 2, length))
            idx = idx[:cutoff]
            openp, high, low, close, volume = openp[:cutoff], high[:cutoff], low[:cutoff], close[:cutoff], volume[:cutoff]
        coins[f"SYN{k:03d}-USD"] = pd.DataFrame(
            {"Open": openp, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx
        )
    return coins, btc_df


def _fill_short_gaps(series: pd.Series, max_gap: int) -> Tuple[pd.Series, np.ndarray]:
    notna = series.notna().to_numpy()
    n = len(notna)
    if not notna.any():
        return series, notna
    first_i = int(np.argmax(notna))
    last_i = n - 1 - int(np.argmax(notna[::-1]))
    in_life = np.zeros(n, dtype=bool)
    in_life[first_i:last_i + 1] = True
    candidate = (~notna) & in_life
    fillable = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if candidate[i]:
            j = i
            while j < n and candidate[j]:
                j += 1
            if (j - i) <= max_gap:
                fillable[i:j] = True
            i = j
        else:
            i += 1
    filled_vals = series.ffill()
    result = series.copy()
    result[fillable] = filled_vals[fillable]
    return result, (notna | fillable)


def align_universe(
    coin_data: Dict[str, pd.DataFrame],
    btc_df: pd.DataFrame,
    max_ffill_gap: int = 3,
    start_date: Optional[str] = None,
    auto_start_pct: float = 0.0,
) -> AlignedData:
    all_dates = btc_df.index
    for df in coin_data.values():
        all_dates = all_dates.union(df.index)
    full_range = pd.date_range(all_dates.min(), all_dates.max(), freq="D")
    T = len(full_range)
    tickers = sorted(coin_data.keys())
    N = len(tickers)

    O = np.full((T, N), np.nan); H = np.full((T, N), np.nan)
    L = np.full((T, N), np.nan); C = np.full((T, N), np.nan)
    V = np.full((T, N), np.nan)
    listed = np.zeros((T, N), dtype=bool)

    for j, t in enumerate(tickers):
        df = coin_data[t].reindex(full_range)
        close_f, valid = _fill_short_gaps(df["Close"], max_ffill_gap)
        open_f, _ = _fill_short_gaps(df["Open"], max_ffill_gap)
        high_f, _ = _fill_short_gaps(df["High"], max_ffill_gap)
        low_f, _ = _fill_short_gaps(df["Low"], max_ffill_gap)
        vol_f, _ = _fill_short_gaps(df["Volume"], max_ffill_gap)
        O[:, j] = np.where(valid, open_f.to_numpy(), np.nan)
        H[:, j] = np.where(valid, high_f.to_numpy(), np.nan)
        L[:, j] = np.where(valid, low_f.to_numpy(), np.nan)
        C[:, j] = np.where(valid, close_f.to_numpy(), np.nan)
        V[:, j] = np.where(valid, vol_f.fillna(0.0).to_numpy(), np.nan)
        listed[:, j] = valid

    btc_close_f, btc_valid = _fill_short_gaps(btc_df["Close"].reindex(full_range), max_ffill_gap)
    btc_close = np.where(btc_valid, btc_close_f.to_numpy(), np.nan)
    btc_close = pd.Series(btc_close).ffill().bfill().to_numpy()

    # --- Determine starting index (start_date vs auto_start_pct) ---
    start_idx = 0
    if start_date is not None:
        target_dt = pd.Timestamp(start_date).tz_localize(None)
        if target_dt <= full_range[0]:
            logger.info(f"Specified start_date ({start_date}) is <= dataset start. Using bar 0.")
            start_idx = 0
        elif target_dt > full_range[-1]:
            raise ValueError(f"Requested start_date {start_date} is after data end ({full_range[-1].strftime('%Y-%m-%d')}).")
        else:
            start_idx = int(np.searchsorted(full_range, target_dt))
            logger.info(f"Using manual start date: {target_dt.strftime('%Y-%m-%d')} (bar {start_idx}/{T}).")
    elif auto_start_pct > 0.0 and N > 0:
        active_counts = listed.sum(axis=1)
        req_count = math.ceil(auto_start_pct * N)
        qualifying = np.where(active_counts >= req_count)[0]
        if len(qualifying) > 0:
            start_idx = int(qualifying[0])
            start_dt = full_range[start_idx]
            act = active_counts[start_idx]
            logger.info(
                f"Auto-detected start date: {start_dt.strftime('%Y-%m-%d')} "
                f"({act}/{N} = {act/N*100:.1f}% coins tradeable, threshold: {auto_start_pct*100:.0f}%). "
                f"Trimming timeline from bar {start_idx}/{T}."
            )
        else:
            max_act = int(active_counts.max()) if len(active_counts) > 0 else 0
            logger.warning(
                f"Universe never reached {auto_start_pct*100:.0f}% concurrently active coins "
                f"(peak: {max_act}/{N} = {max_act/N*100:.1f}%). Using start of dataset."
            )
            start_idx = 0

    if start_idx > 0:
        full_range = full_range[start_idx:]
        O = O[start_idx:, :]
        H = H[start_idx:, :]
        L = L[start_idx:, :]
        C = C[start_idx:, :]
        V = V[start_idx:, :]
        listed = listed[start_idx:, :]
        btc_close = btc_close[start_idx:]

    days_since_start = (full_range - full_range[0]).days.to_numpy().astype(np.int32)

    return AlignedData(
        dates=full_range.to_numpy(),
        days_since_start=days_since_start,
        tickers=tickers,
        O=O, H=H, L=L, C=C, V=V,
        listed=listed,
        btc_close=btc_close,
    )


def validate_universe(aligned: AlignedData, min_liquid_assets: int) -> None:
    target_floor = min(min_liquid_assets, aligned.N)
    min_bars = min(180, max(30, aligned.T // 2))
    usable = int(np.sum(aligned.listed.sum(axis=0) >= min_bars))
    if usable < target_floor:
        raise RuntimeError(
            f"Only {usable} tradable assets have >={min_bars} days of usable history "
            f"after filtering (need >={target_floor}). Universe has {aligned.N} tickers."
        )
    logger.info(f"Universe validated: {usable}/{aligned.N} assets have >={min_bars}d history "
                f"(floor: {target_floor}).")


def load_or_fetch_universe(cfg: RunConfig) -> AlignedData:
    if cfg.synthetic:
        logger.info(f"Generating synthetic universe: {cfg.n_coins_target} coins (seed={cfg.seed})")
        coins, btc_df = generate_synthetic_universe(cfg.n_coins_target, 1500, cfg.seed)
    else:
        tickers = load_universe_list(cfg)
        
        # Ensure priority coins are in the download queue
        std_priority = [standardize_ticker(c) for c in cfg.priority_coins]
        for p_coin in std_priority:
            if p_coin not in tickers:
                tickers.append(p_coin)

        tradable_req = [t for t in tickers if not is_blacklisted(t, allow_btc=cfg.allow_btc_trading)]
        logger.info(f"Universe: {len(tickers)} tickers requested "
                    f"({len(tradable_req)} tradable candidates + BTC macro reference)")
        raw = fetch_from_yfinance(tickers, cfg.cache_dir, cfg.refresh_cache)
        if MACRO_REFERENCE_TICKER not in raw:
            raise RuntimeError(
                f"Could not fetch {MACRO_REFERENCE_TICKER} (required for macro regime filter)."
            )
        
        # Copy BTC for macro filter without popping it, so it can also be traded
        btc_df = raw[MACRO_REFERENCE_TICKER].copy()
        
        # Filter static blacklist and dynamic stablecoin noise
        raw_candidates = {t: df for t, df in raw.items() if not is_blacklisted(t, allow_btc=cfg.allow_btc_trading)}
        coins = {
            t: df for t, df in raw_candidates.items() 
            if not is_noise_or_stablecoin(df, t)
        }

    # --- Apply MAX_COINS filter with Priority Pinning ---
    std_priority_set = {standardize_ticker(c) for c in cfg.priority_coins}
    
    if cfg.max_coins is not None and len(coins) > cfg.max_coins:
        def _coin_score(item: Tuple[str, pd.DataFrame]) -> float:
            _, df = item
            if "Close" in df.columns and "Volume" in df.columns:
                return float((df["Close"] * df["Volume"]).fillna(0.0).sum())
            return float(len(df))

        # 1. Separate priority coins so they cannot be bumped
        pinned_coins = {t: df for t, df in coins.items() if t in std_priority_set}
        remaining_candidates = {t: df for t, df in coins.items() if t not in std_priority_set}

        # 2. Fill remaining slots with top volume coins
        slots_left = max(0, cfg.max_coins - len(pinned_coins))
        sorted_remaining = sorted(remaining_candidates.items(), key=_coin_score, reverse=True)
        top_volume_coins = dict(sorted_remaining[:slots_left])

        # 3. Combine pinned coins + top volume coins
        coins = {**pinned_coins, **top_volume_coins}
        logger.info(f"Universe trimmed: {len(pinned_coins)} priority coins pinned + "
                    f"{len(top_volume_coins)} top-volume coins (total={len(coins)}).")

    aligned = align_universe(
        coins,
        btc_df,
        start_date=cfg.start_date,
        auto_start_pct=cfg.auto_start_min_coins_pct,
    )
    
    # --- Final Verification Audit ---
    verify_priority_coins(aligned.tickers, cfg.priority_coins)
    
    validate_universe(aligned, cfg.min_liquid_assets if not cfg.synthetic else 1)
    logger.info(f"Aligned universe: {aligned.T} bars x {aligned.N} coins "
                f"({pd.Timestamp(aligned.dates[0]).strftime('%Y-%m-%d')} -> {pd.Timestamp(aligned.dates[-1]).strftime('%Y-%m-%d')})")
    return aligned

# ==============================================================================
# PER-TRIAL INDICATOR BUNDLE ASSEMBLY
# ==============================================================================
CH_ENTRY1, CH_ENTRY2, CH_EXIT1, CH_EXIT2 = 0, 1, 2, 3
N_CHANNELS = 4


@dataclass
class GlobalIndicators:
    atr14: np.ndarray
    atr14_ok: np.ndarray
    adx14: np.ndarray
    adx14_ok: np.ndarray
    liquidity_ok: np.ndarray


def precompute_global_indicators(aligned: AlignedData, cfg: RunConfig) -> GlobalIndicators:
    logger.info("Precomputing trial-independent indicators (ATR-14, ADX-14, liquidity gate)...")
    atr14 = atr_2d(aligned.H, aligned.L, aligned.C, 14)
    adx14 = adx_2d(aligned.H, aligned.L, aligned.C, 14)
    liq = rolling_dollar_volume_ok_2d(aligned.C, aligned.V, cfg.liquidity_window, cfg.liquidity_floor_usd)
    return GlobalIndicators(
        atr14=np.nan_to_num(atr14), atr14_ok=~np.isnan(atr14),
        adx14=np.nan_to_num(adx14), adx14_ok=~np.isnan(adx14),
        liquidity_ok=liq,
    )


def build_trial_indicator_bundle(aligned: AlignedData, gi: GlobalIndicators, p: StrategyParams) -> Dict[str, np.ndarray]:
    T, N = aligned.C.shape
    ind_data = np.zeros((T, N, N_CHANNELS))
    ind_ok = np.zeros((T, N, N_CHANNELS), dtype=np.bool_)

    need_rsi = (p.entry_type == ENTRY_RSI_XOVER) or (p.exit_type == EXIT_RSI_CROSSUNDER)
    need_xover = (p.entry_type == ENTRY_MA_XOVER) or (p.exit_type == EXIT_MA_XOVER_EXIT)

    rsi_f = rsi_s = xshort = xlong = None
    if need_rsi:
        rsi_f = smoothed_rsi(aligned.C, p.rsi_f_len, p.rsi_f_smt)
        rsi_s = smoothed_rsi(aligned.C, p.rsi_s_len, p.rsi_s_smt)
    if need_xover:
        xshort = compute_ma(aligned.C, p.xover_short_len, p.xover_short_type)
        xlong = compute_ma(aligned.C, p.xover_long_len(), p.xover_long_type)

    if p.entry_type == ENTRY_MA_BREAKOUT:
        entry_ma = compute_ma(aligned.C, p.entry_ma_len, p.entry_ma_type)
        ind_data[:, :, CH_ENTRY1] = aligned.C
        ind_ok[:, :, CH_ENTRY1] = aligned.listed
        ind_data[:, :, CH_ENTRY2] = np.nan_to_num(entry_ma)
        ind_ok[:, :, CH_ENTRY2] = ~np.isnan(entry_ma)
    elif p.entry_type == ENTRY_RSI_XOVER:
        ind_data[:, :, CH_ENTRY1] = np.nan_to_num(rsi_f); ind_ok[:, :, CH_ENTRY1] = ~np.isnan(rsi_f)
        ind_data[:, :, CH_ENTRY2] = np.nan_to_num(rsi_s); ind_ok[:, :, CH_ENTRY2] = ~np.isnan(rsi_s)
    elif p.entry_type == ENTRY_MA_XOVER:
        ind_data[:, :, CH_ENTRY1] = np.nan_to_num(xshort); ind_ok[:, :, CH_ENTRY1] = ~np.isnan(xshort)
        ind_data[:, :, CH_ENTRY2] = np.nan_to_num(xlong); ind_ok[:, :, CH_ENTRY2] = ~np.isnan(xlong)

    if p.exit_type == EXIT_MA_CROSSUNDER:
        exit_ma = compute_ma(aligned.C, p.exit_ma_len, p.exit_ma_type)
        ind_data[:, :, CH_EXIT1] = aligned.C
        ind_ok[:, :, CH_EXIT1] = aligned.listed
        ind_data[:, :, CH_EXIT2] = np.nan_to_num(exit_ma)
        ind_ok[:, :, CH_EXIT2] = ~np.isnan(exit_ma)
    elif p.exit_type == EXIT_RSI_CROSSUNDER:
        ind_data[:, :, CH_EXIT1] = np.nan_to_num(rsi_f); ind_ok[:, :, CH_EXIT1] = ~np.isnan(rsi_f)
        ind_data[:, :, CH_EXIT2] = np.nan_to_num(rsi_s); ind_ok[:, :, CH_EXIT2] = ~np.isnan(rsi_s)
    elif p.exit_type == EXIT_MA_XOVER_EXIT:
        ind_data[:, :, CH_EXIT1] = np.nan_to_num(xshort); ind_ok[:, :, CH_EXIT1] = ~np.isnan(xshort)
        ind_data[:, :, CH_EXIT2] = np.nan_to_num(xlong); ind_ok[:, :, CH_EXIT2] = ~np.isnan(xlong)

    btc_ma = compute_ma_1d(aligned.btc_close, p.btc_ma_len, p.btc_ma_type)

    use_trend = (p.entry_type == ENTRY_RSI_XOVER) and p.use_rsi_trend_filter
    if use_trend:
        rsi_trend_ma = compute_ma(aligned.C, p.rsi_trend_ma_len, p.rsi_trend_ma_type)
        rsi_trend_ma_ok = ~np.isnan(rsi_trend_ma)
    else:
        rsi_trend_ma = np.zeros((T, N))
        rsi_trend_ma_ok = np.zeros((T, N), dtype=np.bool_)

    return dict(
        ind_data=np.nan_to_num(ind_data), ind_ok=ind_ok,
        btc_ma=np.nan_to_num(btc_ma), btc_ma_ok=~np.isnan(btc_ma),
        rsi_trend_ma=np.nan_to_num(rsi_trend_ma), rsi_trend_ma_ok=rsi_trend_ma_ok,
    )


# ==============================================================================
# NUMBA PORTFOLIO SIMULATION CORE
# ==============================================================================
LMAX = MAX_PYRAMID_LAYERS_CAP


@njit(fastmath=True, cache=True)
def run_backtest_core(
    O, H, L, C, V, listed, liquidity_ok,
    atr14, atr14_ok,
    btc_close, btc_ma, btc_ma_ok,
    adx14, adx14_ok,
    rsi_trend_ma, rsi_trend_ma_ok,
    ind_data, ind_ok,
    days_since_start,
    entry_type, exit_type, watchlist_rank_mode,
    use_btc_entry_gate, use_btc_exit_override,
    use_rsi_trend_filter, use_latched_entry,
    max_pyramid_layers, adx_thresh,
    tp_mult, trail_mult, sl_mult, trail_pct_frac, exit_atr_mult,
    initial_capital, dca_amount, dca_interval_days, ticket_size,
    taker_fee, tax_rate, slippage_base, slippage_atr_mult,
):
    T, N = C.shape

    cash = initial_capital
    next_dca_day = dca_interval_days
    total_paid_in = initial_capital

    layer_active = np.zeros((N, LMAX), dtype=np.int8)
    layer_entry_price = np.zeros((N, LMAX))
    layer_entry_bar = np.full((N, LMAX), -1, dtype=np.int32)
    layer_units = np.zeros((N, LMAX))
    layer_cost_basis = np.zeros((N, LMAX))
    layer_stop = np.zeros((N, LMAX))
    layer_partial_done = np.zeros((N, LMAX), dtype=np.int8)

    pos_peak_high = np.zeros(N)
    pos_realized_pnl = np.zeros(N)
    pos_capital_deployed = np.zeros(N)

    prev_above_entry = np.zeros(N, dtype=np.int8)
    latch_armed = np.zeros(N, dtype=np.int8)

    wl_active = np.zeros(N, dtype=np.int8)
    wl_entry_price = np.zeros(N)
    wl_trigger_bar = np.full(N, -1, dtype=np.int32)
    wl_target_layer = np.zeros(N, dtype=np.int32)
    wl_peak_high = np.zeros(N)

    pend_entry = np.zeros(N, dtype=np.int8)
    pend_entry_layer = np.zeros(N, dtype=np.int32)
    pend_exit_full = np.zeros(N, dtype=np.int8)
    pend_exit_layer = np.zeros((N, LMAX), dtype=np.int8)
    pend_exit_partial_layer = np.full(N, -1, dtype=np.int32)

    max_trades = N * 400 + 200
    tr_coin = np.zeros(max_trades, dtype=np.int32)
    tr_entry_bar = np.zeros(max_trades, dtype=np.int32)
    tr_exit_bar = np.zeros(max_trades, dtype=np.int32)
    tr_entry_price = np.zeros(max_trades)
    tr_exit_price = np.zeros(max_trades)
    tr_units = np.zeros(max_trades)
    tr_pnl = np.zeros(max_trades)
    tr_count = 0

    equity_curve = np.zeros(T)
    cash_curve = np.zeros(T)
    active_pos_curve = np.zeros(T, dtype=np.int32)
    coin_equity = np.zeros((T, N))
    trail_pct = trail_pct_frac

    
    for t in range(T):
        # 1. Execute queued orders at bar Open
        if t > 0:
            tp = t - 1
            for j in range(N):
                if not listed[t, j]:
                    continue
                o = O[t, j]
                cprev = C[tp, j]
                a = atr14[tp, j] if (atr14_ok[tp, j] and cprev > 0.0) else 0.0
                slip = slippage_base + slippage_atr_mult * (a / cprev) if cprev > 0.0 else slippage_base

                if pend_exit_full[j] == 1:
                    sell_price = o * (1.0 - slip)
                    for l in range(LMAX):
                        if layer_active[j, l] == 1:
                            units = layer_units[j, l]
                            gross = units * sell_price
                            net = gross * (1.0 - taker_fee - tax_rate)
                            pnl = net - layer_cost_basis[j, l]
                            cash += net
                            pos_realized_pnl[j] += pnl
                            if tr_count < max_trades:
                                tr_coin[tr_count] = j
                                tr_entry_bar[tr_count] = layer_entry_bar[j, l]
                                tr_exit_bar[tr_count] = t
                                tr_entry_price[tr_count] = layer_entry_price[j, l]
                                tr_exit_price[tr_count] = sell_price
                                tr_units[tr_count] = units
                                tr_pnl[tr_count] = pnl
                                tr_count += 1
                            layer_active[j, l] = 0
                            layer_units[j, l] = 0.0
                            layer_cost_basis[j, l] = 0.0
                            layer_partial_done[j, l] = 0
                    pend_exit_full[j] = 0
                    pos_peak_high[j] = 0.0

                for l in range(LMAX):
                    if pend_exit_layer[j, l] == 1:
                        pend_exit_layer[j, l] = 0
                        if layer_active[j, l] == 1:
                            sell_price = o * (1.0 - slip)
                            units = layer_units[j, l]
                            gross = units * sell_price
                            net = gross * (1.0 - taker_fee - tax_rate)
                            pnl = net - layer_cost_basis[j, l]
                            cash += net
                            pos_realized_pnl[j] += pnl
                            if tr_count < max_trades:
                                tr_coin[tr_count] = j
                                tr_entry_bar[tr_count] = layer_entry_bar[j, l]
                                tr_exit_bar[tr_count] = t
                                tr_entry_price[tr_count] = layer_entry_price[j, l]
                                tr_exit_price[tr_count] = sell_price
                                tr_units[tr_count] = units
                                tr_pnl[tr_count] = pnl
                                tr_count += 1
                            layer_active[j, l] = 0
                            layer_units[j, l] = 0.0
                            layer_cost_basis[j, l] = 0.0
                            layer_partial_done[j, l] = 0

                pl = pend_exit_partial_layer[j]
                if pl >= 0 and layer_active[j, pl] == 1:
                    sell_price = o * (1.0 - slip)
                    sold_units = layer_units[j, pl] * 0.5
                    sold_cost = layer_cost_basis[j, pl] * 0.5
                    gross = sold_units * sell_price
                    net = gross * (1.0 - taker_fee - tax_rate)
                    pnl = net - sold_cost
                    cash += net
                    pos_realized_pnl[j] += pnl
                    if tr_count < max_trades:
                        tr_coin[tr_count] = j
                        tr_entry_bar[tr_count] = layer_entry_bar[j, pl]
                        tr_exit_bar[tr_count] = t
                        tr_entry_price[tr_count] = layer_entry_price[j, pl]
                        tr_exit_price[tr_count] = sell_price
                        tr_units[tr_count] = sold_units
                        tr_pnl[tr_count] = pnl
                        tr_count += 1
                    layer_units[j, pl] -= sold_units
                    layer_cost_basis[j, pl] -= sold_cost
                    layer_partial_done[j, pl] = 1
                    layer_stop[j, pl] = layer_entry_price[j, pl]
                    pend_exit_partial_layer[j] = -1

                if pend_entry[j] == 1:
                    buy_price = o * (1.0 + slip)
                    l = pend_entry_layer[j]
                    n_active_before = 0
                    for ll in range(LMAX):
                        n_active_before += layer_active[j, ll]
                    units = ticket_size / (buy_price * (1.0 + taker_fee))
                    layer_active[j, l] = 1
                    layer_entry_price[j, l] = buy_price
                    layer_entry_bar[j, l] = t
                    layer_units[j, l] = units
                    layer_cost_basis[j, l] = ticket_size
                    layer_partial_done[j, l] = 0
                    if exit_type == EXIT_HYBRID_ATR:
                        layer_stop[j, l] = buy_price - atr14[t, j] * sl_mult if atr14_ok[t, j] else buy_price * 0.5
                    pos_capital_deployed[j] += ticket_size
                    if n_active_before == 0:
                        pos_peak_high[j] = H[t, j]
                    pend_entry[j] = 0

        # 2. DCA injection
        while days_since_start[t] >= next_dca_day:
            cash += dca_amount
            total_paid_in += dca_amount
            next_dca_day += dca_interval_days

        # 3. Exit logic & trailing updates
        for j in range(N):
            if not listed[t, j]:
                continue
            n_active = 0
            for l in range(LMAX):
                n_active += layer_active[j, l]

            if n_active > 0:
                btc_override = (use_btc_exit_override == 1) and btc_ma_ok[t] and (btc_close[t] <= btc_ma[t])
                if btc_override:
                    pend_exit_full[j] = 1
                else:
                    if exit_type == EXIT_HYBRID_ATR:
                        for l in range(LMAX):
                            if layer_active[j, l] == 0:
                                continue
                            if layer_partial_done[j, l] == 0:
                                if atr14_ok[t, j]:
                                    tp_level = layer_entry_price[j, l] + atr14[t, j] * tp_mult
                                    if C[t, j] >= tp_level:
                                        pend_exit_partial_layer[j] = l
                                if C[t, j] <= layer_stop[j, l]:
                                    pend_exit_layer[j, l] = 1
                            else:
                                if atr14_ok[t, j]:
                                    cand = C[t, j] - atr14[t, j] * trail_mult
                                    if cand > layer_stop[j, l]:
                                        layer_stop[j, l] = cand
                                if C[t, j] <= layer_stop[j, l]:
                                    pend_exit_layer[j, l] = 1
                    elif exit_type == EXIT_PCT_TRAIL:
                        if H[t, j] > pos_peak_high[j]:
                            pos_peak_high[j] = H[t, j]
                        if C[t, j] < pos_peak_high[j] * (1.0 - trail_pct):
                            pend_exit_full[j] = 1
                    elif exit_type == EXIT_ATR_TRAIL:
                        if H[t, j] > pos_peak_high[j]:
                            pos_peak_high[j] = H[t, j]
                        if atr14_ok[t, j] and C[t, j] < (pos_peak_high[j] - atr14[t, j] * exit_atr_mult):
                            pend_exit_full[j] = 1
                    elif exit_type in (EXIT_MA_CROSSUNDER, EXIT_RSI_CROSSUNDER, EXIT_MA_XOVER_EXIT):
                        ok1 = ind_ok[t, j, CH_EXIT1]; ok2 = ind_ok[t, j, CH_EXIT2]
                        if ok1 and ok2 and ind_data[t, j, CH_EXIT1] < ind_data[t, j, CH_EXIT2]:
                            pend_exit_full[j] = 1

            if wl_active[j] == 1:
                if H[t, j] > wl_peak_high[j]:
                    wl_peak_high[j] = H[t, j]
                purge = False
                if exit_type == EXIT_HYBRID_ATR:
                    if atr14_ok[t, j] and C[t, j] <= (wl_entry_price[j] - atr14[t, j] * sl_mult):
                        purge = True
                elif exit_type == EXIT_PCT_TRAIL:
                    if C[t, j] < wl_peak_high[j] * (1.0 - trail_pct):
                        purge = True
                elif exit_type == EXIT_ATR_TRAIL:
                    if atr14_ok[t, j] and C[t, j] < (wl_peak_high[j] - atr14[t, j] * exit_atr_mult):
                        purge = True
                elif exit_type in (EXIT_MA_CROSSUNDER, EXIT_RSI_CROSSUNDER, EXIT_MA_XOVER_EXIT):
                    ok1 = ind_ok[t, j, CH_EXIT1]; ok2 = ind_ok[t, j, CH_EXIT2]
                    if ok1 and ok2 and ind_data[t, j, CH_EXIT1] < ind_data[t, j, CH_EXIT2]:
                        purge = True
                if wl_target_layer[j] > 0:
                    n_now = 0
                    for l in range(LMAX):
                        n_now += layer_active[j, l]
                    if n_now == 0:
                        purge = True
                if purge:
                    wl_active[j] = 0

        # 4. Entry signals
        for j in range(N):
            if not listed[t, j] or pend_exit_full[j] == 1 or wl_active[j] == 1:
                continue
            n_active = 0
            for l in range(LMAX):
                n_active += layer_active[j, l]
            if n_active >= max_pyramid_layers:
                continue

            ok1 = ind_ok[t, j, CH_ENTRY1]; ok2 = ind_ok[t, j, CH_ENTRY2]
            raw_cross = False
            if ok1 and ok2:
                is_above_now = ind_data[t, j, CH_ENTRY1] > ind_data[t, j, CH_ENTRY2]
                if t > 0 and prev_above_entry[j] == 0 and is_above_now:
                    raw_cross = True
                prev_above_entry[j] = 1 if is_above_now else 0
            else:
                is_above_now = False

            btc_gate_ok = True
            if use_btc_entry_gate == 1:
                btc_gate_ok = btc_ma_ok[t] and (btc_close[t] > btc_ma[t])
            adx_gate_ok = True
            if adx_thresh > 0.0:
                adx_gate_ok = adx14_ok[t, j] and (adx14[t, j] >= adx_thresh)
            trend_gate_ok = True
            if entry_type == ENTRY_RSI_XOVER and use_rsi_trend_filter == 1:
                trend_gate_ok = rsi_trend_ma_ok[t, j] and (C[t, j] > rsi_trend_ma[t, j])
            liq_gate_ok = liquidity_ok[t, j]
            filters_pass = btc_gate_ok and adx_gate_ok and trend_gate_ok and liq_gate_ok

            effective_signal = False
            if use_latched_entry == 1:
                if latch_armed[j] == 0 and raw_cross:
                    latch_armed[j] = 1
                if latch_armed[j] == 1:
                    if filters_pass:
                        effective_signal = True
                        latch_armed[j] = 0
                    elif not is_above_now:
                        latch_armed[j] = 0
            else:
                effective_signal = raw_cross and filters_pass

            if effective_signal:
                wl_active[j] = 1
                wl_entry_price[j] = C[t, j]
                wl_trigger_bar[j] = t
                wl_target_layer[j] = n_active
                wl_peak_high[j] = H[t, j]

        # 5. Funding from watchlist
        while cash >= ticket_size:
            best_j = -1
            best_score = -1.0e300
            for j in range(N):
                if wl_active[j] == 0 or not listed[t, j]:
                    continue
                n_active = 0
                for l in range(LMAX):
                    n_active += layer_active[j, l]
                if n_active != wl_target_layer[j] or n_active >= max_pyramid_layers:
                    continue

                btc_gate_ok = (use_btc_entry_gate != 1) or (btc_ma_ok[t] and btc_close[t] > btc_ma[t])
                adx_gate_ok = (adx_thresh <= 0.0) or (adx14_ok[t, j] and adx14[t, j] >= adx_thresh)
                trend_gate_ok = (entry_type != ENTRY_RSI_XOVER or use_rsi_trend_filter != 1) or (
                    rsi_trend_ma_ok[t, j] and C[t, j] > rsi_trend_ma[t, j]
                )
                liq_gate_ok = liquidity_ok[t, j]
                if not (btc_gate_ok and adx_gate_ok and trend_gate_ok and liq_gate_ok):
                    continue

                wep = wl_entry_price[j]
                if watchlist_rank_mode == WL_DEEPEST_DISCOUNT:
                    score = (wep - C[t, j]) / wep if wep > 0.0 else -1.0e300
                elif watchlist_rank_mode == WL_CLOSEST_BREAKOUT:
                    score = -abs(C[t, j] - wep) / C[t, j] if C[t, j] > 0.0 else -1.0e300
                elif watchlist_rank_mode == WL_STRONGEST_MOMENTUM:
                    score = (C[t, j] - wep) / wep if wep > 0.0 else -1.0e300
                else:
                    score = -float(wl_trigger_bar[j])

                if score > best_score:
                    best_score = score
                    best_j = j

            if best_j < 0:
                break
            cash -= ticket_size
            pend_entry[best_j] = 1
            pend_entry_layer[best_j] = wl_target_layer[best_j]
            wl_active[best_j] = 0

        # 6. Portfolio valuation
        mv = 0.0
        for j in range(N):
            for l in range(LMAX):
                if layer_active[j, l] == 1:
                    mv += layer_units[j, l] * C[t, j]
            pend_val = ticket_size if pend_entry[j] == 1 else 0.0
            mv += pend_val
            unreal = 0.0
            for l in range(LMAX):
                if layer_active[j, l] == 1:
                    unreal += layer_units[j, l] * (C[t, j] - layer_entry_price[j, l])
            coin_equity[t, j] = pos_realized_pnl[j] + unreal
        # Count distinct assets with at least one active tranche
        concurrent_count = 0
        for j in range(N):
            has_pos = 0
            for l in range(LMAX):
                if layer_active[j, l] == 1:
                    has_pos = 1
                    break
            concurrent_count += has_pos

        active_pos_curve[t] = concurrent_count
        cash_curve[t] = cash
        equity_curve[t] = cash + mv

    return (
        tr_coin[:tr_count], tr_entry_bar[:tr_count], tr_exit_bar[:tr_count],
        tr_entry_price[:tr_count], tr_exit_price[:tr_count], tr_units[:tr_count],
        tr_pnl[:tr_count], pos_capital_deployed, coin_equity, equity_curve,
        cash, total_paid_in, tr_count >= max_trades,
        cash_curve, active_pos_curve,
    )


def run_one_trial(aligned: AlignedData, gi: GlobalIndicators, p: StrategyParams, cfg: RunConfig):
    bundle = build_trial_indicator_bundle(aligned, gi, p)
    O_ = np.nan_to_num(aligned.O); H_ = np.nan_to_num(aligned.H)
    L_ = np.nan_to_num(aligned.L); C_ = np.nan_to_num(aligned.C)
    V_ = np.nan_to_num(aligned.V)
    return run_backtest_core(
        O_, H_, L_, C_, V_, aligned.listed, gi.liquidity_ok,
        gi.atr14, gi.atr14_ok,
        aligned.btc_close, bundle["btc_ma"], bundle["btc_ma_ok"],
        gi.adx14, gi.adx14_ok,
        bundle["rsi_trend_ma"], bundle["rsi_trend_ma_ok"],
        bundle["ind_data"], bundle["ind_ok"],
        aligned.days_since_start,
        p.entry_type, p.exit_type, p.watchlist_rank_mode,
        int(p.use_btc_entry_gate), int(p.use_btc_exit_override),
        int(p.use_rsi_trend_filter), int(p.use_latched_entry),
        p.max_pyramid_layers, p.adx_thresh,
        p.tp_mult, p.trail_mult, p.sl_mult, p.trail_pct / 100.0, p.exit_atr_mult,
        cfg.initial_capital, cfg.dca_amount, cfg.dca_interval_days, cfg.ticket_size,
        cfg.taker_fee, cfg.tax_rate, cfg.slippage_base, cfg.slippage_atr_mult,
    )


# ==============================================================================
# METRICS & COMPOSITE SCORE
# ==============================================================================
def compute_metrics(trial_result, aligned: AlignedData, cfg: RunConfig) -> Dict[str, Any]:
    (tr_coin, tr_entry_bar, tr_exit_bar, tr_entry_price, tr_exit_price, tr_units,
     tr_pnl, pos_capital_deployed, coin_equity, equity_curve, final_cash,
     total_paid_in, overflow, cash_curve, active_pos_curve) = trial_result

    N = aligned.N
    years = pd.DatetimeIndex(aligned.dates).year.to_numpy()
    total_trades = int(len(tr_pnl))

    traded_mask = pos_capital_deployed > 0
    coin_pnl = np.zeros(N)
    if total_trades > 0:
        np.add.at(coin_pnl, tr_coin, tr_pnl)
    coin_lifetime_return = np.full(N, np.nan)
    coin_lifetime_return[traded_mask] = coin_pnl[traded_mask] / pos_capital_deployed[traded_mask]

    valid_returns = coin_lifetime_return[traded_mask]
    median_coin_return = float(np.median(valid_returns)) if valid_returns.size > 0 else 0.0

    annual_median_returns: List[float] = []
    if total_trades > 0:
        exit_years = years[tr_exit_bar]
        unique_years = sorted(set(exit_years.tolist()))
        for y in unique_years:
            mask_y = exit_years == y
            if not mask_y.any():
                continue
            coin_pnl_y = np.zeros(N)
            np.add.at(coin_pnl_y, tr_coin[mask_y], tr_pnl[mask_y])
            coins_active_y = np.unique(tr_coin[mask_y])
            rets_y = [coin_pnl_y[c] / pos_capital_deployed[c] for c in coins_active_y if pos_capital_deployed[c] > 0]
            if rets_y:
                annual_median_returns.append(float(np.median(rets_y)))

    if len(annual_median_returns) >= 2:
        mu = float(np.mean(annual_median_returns))
        sigma = float(np.std(annual_median_returns))
        annual_consistency_factor = mu / (sigma + 1e-4)
    elif len(annual_median_returns) == 1:
        annual_consistency_factor = annual_median_returns[0]
    else:
        annual_consistency_factor = 0.0

    annual_consistency_factor = max(0.0, annual_consistency_factor)

    per_coin_year_dd: List[float] = []
    if total_trades > 0:
        coins_with_exits = set(np.unique(tr_coin).tolist())
        for c in np.where(traded_mask)[0]:
            if c not in coins_with_exits:
                continue
            entries_c = tr_entry_bar[tr_coin == c]
            exits_c = tr_exit_bar[tr_coin == c]
            first_bar = int(entries_c.min())
            last_bar = int(exits_c.max())
            window_years = sorted(set(years[first_bar:last_bar + 1].tolist()))
            series_full = coin_equity[first_bar:last_bar + 1, c]
            yrs_slice = years[first_bar:last_bar + 1]
            capital_c = max(pos_capital_deployed[c], 1.0)
            for y in window_years:
                s = series_full[yrs_slice == y]
                if s.size < 2:
                    continue
                running_peak = np.maximum.accumulate(s)
                dd = (running_peak - s) / capital_c
                per_coin_year_dd.append(float(np.max(dd)))

    avg_max_dd = float(np.mean(per_coin_year_dd)) if per_coin_year_dd else 0.0
    drawdown_penalty = 1.0 / (1.0 + (avg_max_dd / 0.20) ** 2)
    activity_damper = min(1.0, total_trades / max(1, cfg.min_expected_trades))

    score = median_coin_return * annual_consistency_factor * drawdown_penalty * activity_damper

    return dict(
        score=float(score),
        median_coin_return=median_coin_return,
        annual_consistency_factor=float(annual_consistency_factor),
        drawdown_penalty=float(drawdown_penalty),
        activity_damper=float(activity_damper),
        avg_max_drawdown_per_coin_per_year=avg_max_dd,
        total_trades=total_trades,
        n_coins_traded=int(traded_mask.sum()),
        annual_median_returns=annual_median_returns,
        final_equity=float(equity_curve[-1]),
        total_paid_in=float(total_paid_in),
        final_cash=float(final_cash),
        trade_log_overflow=bool(overflow),
    )


# ==============================================================================
# OPTUNA SEARCH SPACE
# ==============================================================================
def suggest_params(trial: optuna.Trial) -> StrategyParams:
    p = StrategyParams()
    p.entry_type = trial.suggest_categorical("entry_type", [ENTRY_MA_BREAKOUT, ENTRY_RSI_XOVER, ENTRY_MA_XOVER])

    if p.entry_type == ENTRY_MA_BREAKOUT:
        p.entry_ma_len = trial.suggest_int("entry_ma_len", 10, 250)
        p.entry_ma_type = trial.suggest_categorical("entry_ma_type", [0, 1, 2, 3, 4])

    need_rsi_params = (p.entry_type == ENTRY_RSI_XOVER)
    need_xover_params = (p.entry_type == ENTRY_MA_XOVER)

    p.exit_type = trial.suggest_categorical(
        "exit_type", [EXIT_HYBRID_ATR, EXIT_PCT_TRAIL, EXIT_ATR_TRAIL,
                      EXIT_MA_CROSSUNDER, EXIT_RSI_CROSSUNDER, EXIT_MA_XOVER_EXIT]
    )

    if p.exit_type == EXIT_RSI_CROSSUNDER:
        need_rsi_params = True
    if p.exit_type == EXIT_MA_XOVER_EXIT:
        need_xover_params = True

    if need_rsi_params:
        p.rsi_f_len = trial.suggest_int("rsi_f_len", 10, 100)
        p.rsi_f_smt = trial.suggest_int("rsi_f_smt", 5, 50)
        p.rsi_s_len = trial.suggest_int("rsi_s_len", 10, 100)
        p.rsi_s_smt = trial.suggest_int("rsi_s_smt", 5, 50)

    if need_xover_params:
        p.xover_short_len = trial.suggest_int("xover_short_len", 20, 200)
        p.xover_short_type = trial.suggest_categorical("xover_short_type", [0, 1, 2, 3, 4])
        p.xover_gap = trial.suggest_int("xover_gap", 10, 150)
        p.xover_long_type = trial.suggest_categorical("xover_long_type", [0, 1, 2, 3, 4])

    if p.exit_type == EXIT_MA_CROSSUNDER:
        p.exit_ma_len = trial.suggest_int("exit_ma_len", 20, 300)
        p.exit_ma_type = trial.suggest_categorical("exit_ma_type", [0, 1, 2, 3, 4])

    p.use_btc_entry_gate = trial.suggest_categorical("use_btc_entry_gate", [True, False])
    if p.use_btc_entry_gate:
        p.btc_ma_len = trial.suggest_int("btc_ma_len", 20, 300)
        p.btc_ma_type = trial.suggest_categorical("btc_ma_type", [0, 1, 2, 3, 4])

    p.adx_thresh = trial.suggest_categorical("adx_thresh", [0.0, 15.0, 20.0, 25.0])

    if p.entry_type == ENTRY_RSI_XOVER:
        p.use_rsi_trend_filter = trial.suggest_categorical("use_rsi_trend_filter", [True, False])
        if p.use_rsi_trend_filter:
            p.rsi_trend_ma_len = trial.suggest_int("rsi_trend_ma_len", 20, 300)
            p.rsi_trend_ma_type = trial.suggest_categorical("rsi_trend_ma_type", [0, 1, 2, 3, 4])

    p.use_latched_entry = trial.suggest_categorical("use_latched_entry", [True, False])
    p.max_pyramid_layers = trial.suggest_int("max_pyramid_layers", 1, 4)

    if p.exit_type == EXIT_HYBRID_ATR:
        p.tp_mult = trial.suggest_float("tp_mult", 5.0, 70.0)
        p.trail_mult = trial.suggest_float("trail_mult", 2.0, 15.0)
        p.sl_mult = trial.suggest_float("sl_mult", 1.5, 8.0)
    elif p.exit_type == EXIT_PCT_TRAIL:
        p.trail_pct = trial.suggest_float("trail_pct", 5.0, 35.0)
    elif p.exit_type == EXIT_ATR_TRAIL:
        p.exit_atr_mult = trial.suggest_float("exit_atr_mult", 1.0, 6.0)

    p.use_btc_exit_override = trial.suggest_categorical("use_btc_exit_override", [True, False])
    p.watchlist_rank_mode = trial.suggest_categorical(
        "watchlist_rank_mode", [WL_DEEPEST_DISCOUNT, WL_CLOSEST_BREAKOUT, WL_STRONGEST_MOMENTUM, WL_FCFS]
    )
    return p


def build_sampler(cfg: RunConfig):
    if cfg.sampler == "random":
        return RandomSampler(seed=cfg.seed)
    return TPESampler(multivariate=False, n_startup_trials=cfg.n_startup_trials, seed=cfg.seed)


def objective_factory(aligned: AlignedData, gi: GlobalIndicators, cfg: RunConfig):
    def objective(trial: optuna.Trial) -> float:
        p = suggest_params(trial)
        try:
            result = run_one_trial(aligned, gi, p, cfg)
            metrics = compute_metrics(result, aligned, cfg)
        except Exception as e:
            logger.debug(f"Trial {trial.number} raised {type(e).__name__}: {e}")
            return -1e9
        for k, v in metrics.items():
            if k != "annual_median_returns":
                trial.set_user_attr(k, v)
        trial.set_user_attr("params_full", p.as_dict())
        score = metrics["score"]
        if not np.isfinite(score):
            return -1e9
        return score
    return objective


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class ProgressReporter:
    def __init__(self, total: int, every_n: int = 10, max_seconds_between: float = 30.0,
                 min_seconds_between: float = 2.0):
        self.total = total
        self.every_n = max(1, every_n)
        self.max_seconds_between = max_seconds_between
        self.min_seconds_between = min_seconds_between
        self.start_time = time.time()
        self.last_print_time = self.start_time
        self.last_print_count = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        done = trial.number + 1
        now = time.time()
        is_last = done >= self.total
        trials_since = done - self.last_print_count
        time_since = now - self.last_print_time
        should_print = (
            is_last
            or (trials_since >= self.every_n and time_since >= self.min_seconds_between)
            or time_since >= self.max_seconds_between
        )
        if not should_print:
            return
        elapsed = now - self.start_time
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - done) / rate if rate > 0 else float("nan")
        pct = 100.0 * done / self.total
        try:
            best = study.best_value
            best_n = study.best_trial.number
            best_str = f"best={best:.5f} (trial #{best_n})"
        except ValueError:
            best_str = "best=(none yet)"
        eta_str = _fmt_duration(remaining) if np.isfinite(remaining) else "?"
        logger.info(
            f"Progress: {done}/{self.total} ({pct:5.1f}%)  "
            f"elapsed={_fmt_duration(elapsed)}  eta={eta_str}  "
            f"rate={rate:.2f} trials/s  {best_str}"
        )
        self.last_print_time = now
        self.last_print_count = done


# ==============================================================================
# TOP-5 LEADERBOARD PERSISTENCE
# ==============================================================================
class TopKLeaderboard:
    def __init__(self, path: str, k: int = 5):
        self.path = Path(path)
        self.k = k
        self.entries: List[Dict[str, Any]] = []

    def _write(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({"generated_at": datetime.now(timezone.utc).isoformat(),
                       "top_k": self.entries}, f, indent=2, default=str)
        os.replace(tmp, self.path)

    def maybe_add(self, trial: optuna.trial.FrozenTrial) -> None:
        score = trial.value
        if score is None or not np.isfinite(score) or score <= -1e8:
            return
        if len(self.entries) >= self.k and score <= self.entries[-1]["score"]:
            return
        entry = {
            "rank": None,
            "trial_number": trial.number,
            "score": float(score),
            "params": trial.user_attrs.get("params_full", trial.params),
            "metrics": {k: v for k, v in trial.user_attrs.items() if k not in ("params_full",)},
            "found_at": datetime.now(timezone.utc).isoformat(),
        }
        self.entries.append(entry)
        self.entries.sort(key=lambda e: e["score"], reverse=True)
        self.entries = self.entries[: self.k]
        for i, e in enumerate(self.entries):
            e["rank"] = i + 1
        self._write()
        rank = next(e["rank"] for e in self.entries if e["trial_number"] == trial.number)
        tag = "New #1" if rank == 1 else f"New top-{self.k} (#{rank})"
        logger.info(f"  {tag}: score={score:.5f}  trial=#{trial.number}  "
                    f"median_ret={trial.user_attrs.get('median_coin_return', float('nan')):+.3f}  "
                    f"trades={trial.user_attrs.get('total_trades', '?')}")

    def callback(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        self.maybe_add(trial)


# ==============================================================================
# REPORTING
# ==============================================================================
def generate_validation_audit_report(
    best_trial: optuna.trial.FrozenTrial,
    aligned: AlignedData,
    gi: GlobalIndicators,
    cfg: RunConfig,
    output_dir: str,
) -> str:
    """
    Simulates the champion strategy parameter set and generates an exhaustive
    quantitative audit dossier (Markdown + JSON) covering universe composition,
    annual cash drag, concurrent exposure, friction drag, and outlier concentration.
    """
    p_dict = best_trial.user_attrs.get("params_full", best_trial.params)
    
    # Reconstruct StrategyParams dataclass
    p = StrategyParams(
        entry_type=p_dict.get("entry_type", 0),
        entry_ma_len=p_dict.get("entry_ma_len", 50),
        entry_ma_type=p_dict.get("entry_ma_type", 0),
        rsi_f_len=p_dict.get("rsi_f_len", 14),
        rsi_f_smt=p_dict.get("rsi_f_smt", 5),
        rsi_s_len=p_dict.get("rsi_s_len", 28),
        rsi_s_smt=p_dict.get("rsi_s_smt", 10),
        xover_short_len=p_dict.get("xover_short_len", 50),
        xover_short_type=p_dict.get("xover_short_type", 0),
        xover_gap=p_dict.get("xover_gap", 50),
        xover_long_type=p_dict.get("xover_long_type", 0),
        use_btc_entry_gate=bool(p_dict.get("use_btc_entry_gate", True)),
        btc_ma_len=p_dict.get("btc_ma_len", 100),
        btc_ma_type=p_dict.get("btc_ma_type", 0),
        adx_thresh=float(p_dict.get("adx_thresh", 20.0)),
        use_rsi_trend_filter=bool(p_dict.get("use_rsi_trend_filter", False)),
        rsi_trend_ma_len=p_dict.get("rsi_trend_ma_len", 100),
        rsi_trend_ma_type=p_dict.get("rsi_trend_ma_type", 0),
        use_latched_entry=bool(p_dict.get("use_latched_entry", False)),
        max_pyramid_layers=p_dict.get("max_pyramid_layers", 1),
        exit_type=p_dict.get("exit_type", 0),
        tp_mult=float(p_dict.get("tp_mult", 20.0)),
        trail_mult=float(p_dict.get("trail_mult", 6.0)),
        sl_mult=float(p_dict.get("sl_mult", 3.0)),
        trail_pct=float(p_dict.get("trail_pct", 15.0)),
        exit_atr_mult=float(p_dict.get("exit_atr_mult", 3.0)),
        exit_ma_len=p_dict.get("exit_ma_len", 50),
        exit_ma_type=p_dict.get("exit_ma_type", 0),
        use_btc_exit_override=bool(p_dict.get("use_btc_exit_override", False)),
        watchlist_rank_mode=p_dict.get("watchlist_rank_mode", 0),
    )

    # Full execution trace of the champion strategy
    trial_res = run_one_trial(aligned, gi, p, cfg)
    (tr_coin, tr_entry_bar, tr_exit_bar, tr_entry_price, tr_exit_price, tr_units,
     tr_pnl, pos_capital_deployed, coin_equity, equity_curve, final_cash,
     total_paid_in, overflow, cash_curve, active_pos_curve) = trial_res

    T, N = aligned.C.shape
    dates = pd.DatetimeIndex(aligned.dates)
    years = dates.year.to_numpy()
    unique_years = sorted(np.unique(years).tolist())
    total_trades = len(tr_pnl)

    # --- 1. Portfolio & Risk Adjustments (DCA-Insulated Returns) ---
    dca_dates = np.zeros(T, dtype=bool)
    next_dca = cfg.dca_interval_days
    for t in range(T):
        if aligned.days_since_start[t] >= next_dca:
            dca_dates[t] = True
            next_dca += cfg.dca_interval_days

    # Daily organic returns stripping out cash injections
    daily_returns = np.zeros(T)
    for t in range(1, T):
        prev_eq = equity_curve[t - 1]
        injection = cfg.dca_amount if dca_dates[t] else 0.0
        daily_returns[t] = (equity_curve[t] - injection - prev_eq) / prev_eq if prev_eq > 0 else 0.0

    mean_ret = np.mean(daily_returns[1:])
    std_ret = np.std(daily_returns[1:])
    downside_std = np.std(daily_returns[1:][daily_returns[1:] < 0])
    
    sharpe = float((mean_ret / (std_ret + 1e-9)) * np.sqrt(365))
    sortino = float((mean_ret / (downside_std + 1e-9)) * np.sqrt(365))

    # Portfolio Drawdown
    running_peak = np.maximum.accumulate(equity_curve)
    drawdowns = (running_peak - equity_curve) / np.maximum(running_peak, 1.0)
    max_dd_pct = float(np.max(drawdowns)) * 100.0
    max_dd_usd = float(np.max(running_peak - equity_curve))

    total_days = max(1, (dates[-1] - dates[0]).days)
    final_eq = float(equity_curve[-1])
    net_profit = final_eq - total_paid_in
    roi_pct = (net_profit / total_paid_in) * 100.0
    cagr_pct = ((final_eq / total_paid_in) ** (365.0 / total_days) - 1.0) * 100.0 if final_eq > 0 else -100.0
    calmar = float(cagr_pct / (max_dd_pct + 1e-4))

    # --- 2. Trade Mechanics & Friction Breakdown ---
    wins = tr_pnl[tr_pnl > 0]
    losses = tr_pnl[tr_pnl <= 0]
    win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
    profit_factor = (np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 and abs(np.sum(losses)) > 0 else float("nan")
    payoff_ratio = (np.mean(wins) / abs(np.mean(losses))) if len(wins) > 0 and len(losses) > 0 else float("nan")

    # Estimated transaction costs paid
    est_buy_fees = total_trades * cfg.ticket_size * cfg.taker_fee
    est_sell_volume = np.sum(tr_units * tr_exit_price)
    est_sell_fees = est_sell_volume * cfg.taker_fee
    est_taxes = est_sell_volume * cfg.tax_rate
    total_friction_usd = est_buy_fees + est_sell_fees + est_taxes

    # --- 3. Asset Concentration & Outlier Audit ---
    coin_pnl_map = np.zeros(N)
    if total_trades > 0:
        np.add.at(coin_pnl_map, tr_coin, tr_pnl)
    
    sorted_coin_indices = np.argsort(coin_pnl_map)[::-1]
    top1_coin_pnl = float(coin_pnl_map[sorted_coin_indices[0]]) if total_trades > 0 else 0.0
    top3_coin_pnl = float(np.sum(coin_pnl_map[sorted_coin_indices[:3]])) if total_trades > 0 else 0.0
    top5_coin_pnl = float(np.sum(coin_pnl_map[sorted_coin_indices[:5]])) if total_trades > 0 else 0.0

    top3_conc_pct = (top3_coin_pnl / net_profit * 100.0) if net_profit > 0 else 0.0
    top1_ticker = aligned.tickers[sorted_coin_indices[0]] if total_trades > 0 else "N/A"

    # --- 4. Year-by-Year Breakdown Table ---
    annual_rows = []
    for y in unique_years:
        mask_y = years == y
        if not mask_y.any():
            continue
        
        y_bars = np.where(mask_y)[0]
        y_start_eq = float(equity_curve[y_bars[0]])
        y_end_eq = float(equity_curve[y_bars[-1]])
        y_eq_change = y_end_eq - y_start_eq
        
        # Idle Cash & Concurrent Positions for Year y
        y_cash = cash_curve[mask_y]
        y_equity = equity_curve[mask_y]
        avg_idle_cash = float(np.mean(y_cash))
        avg_idle_pct = float(np.mean(y_cash / np.maximum(y_equity, 1.0)) * 100.0)
        max_concurrent_y = int(np.max(active_pos_curve[mask_y]))
        avg_concurrent_y = float(np.mean(active_pos_curve[mask_y]))

        # Trades & Return inside Year y
        y_trade_mask = (tr_exit_bar >= y_bars[0]) & (tr_exit_bar <= y_bars[-1])
        y_trades = int(np.sum(y_trade_mask))
        
        # Realized Median Coin Return for Year y
        if y_trades > 0:
            y_coins = np.unique(tr_coin[y_trade_mask])
            y_rets = []
            for c_idx in y_coins:
                c_pnl_y = np.sum(tr_pnl[y_trade_mask & (tr_coin == c_idx)])
                cap = pos_capital_deployed[c_idx]
                if cap > 0:
                    y_rets.append(c_pnl_y / cap)
            med_ret_y = float(np.median(y_rets) * 100.0) if y_rets else 0.0
        else:
            med_ret_y = 0.0

        annual_rows.append({
            "year": y,
            "start_equity": y_start_eq,
            "end_equity": y_end_eq,
            "net_change": y_eq_change,
            "trades": y_trades,
            "median_return_pct": med_ret_y,
            "avg_idle_cash": avg_idle_cash,
            "idle_cash_pct": avg_idle_pct,
            "max_concurrent": max_concurrent_y,
            "avg_concurrent": avg_concurrent_y,
        })

    # --- 5. Generate Markdown Report ---
    report_lines = [
        "# Quantitative Strategy Validation Dossier",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"**Backtest Timeline:** {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')} ({total_days} days)  ",
        f"**Universe:** {aligned.N} tradable assets (evaluated from top volume & priority list)",
        "",
        "---",
        "## 1. Executive Performance & Capital Efficiency",
        "",
        "| Metric | Result | Institutional Benchmark |",
        "| :--- | :--- | :--- |",
        f"| **Initial Capital** | ${cfg.initial_capital:,.2f} | — |",
        f"| **Total Capital Invested (DCA)** | ${total_paid_in:,.2f} | — |",
        f"| **Ending Portfolio Equity** | ${final_eq:,.2f} | — |",
        f"| **Net Realized & Unrealized Profit** | ${net_profit:+,.2f} | — |",
        f"| **Total ROI** | {roi_pct:+.2f}% | — |",
        f"| **CAGR (Annualized Compound Return)** | {cagr_pct:.2f}% | > 20% Target |",
        f"| **Sharpe Ratio (Annualized)** | {sharpe:.2f} | > 1.20 Validated |",
        f"| **Sortino Ratio (Downside Risk)** | {sortino:.2f} | > 1.50 Desired |",
        f"| **Calmar Ratio (CAGR / MaxDD)** | {calmar:.2f} | > 1.00 Robust |",
        f"| **Portfolio Max Drawdown** | -{max_dd_pct:.2f}% (-${max_dd_usd:,.2f}) | < 25% Preferred |",
        f"| **Overall Average Idle Cash Drag** | {np.mean(cash_curve / equity_curve)*100:.1f}% | < 40% (Capital Drag Warning) |",
        f"| **Peak Concurrent Positions Held** | {int(np.max(active_pos_curve))} coins | Portfolio Capacity Cap |",
        "",
        "---",
        "## 2. Trade Mechanics & Friction Audit",
        "",
        "| Execution Variable | Value | Notes |",
        "| :--- | :--- | :--- |",
        f"| **Total Closed Exits (Full/Partial)** | {total_trades:,} | Statistically significant (>150 required) |",
        f"| **Win Rate** | {win_rate:.2f}% | Trend-following standard: 35% - 50% |",
        f"| **Profit Factor** | {profit_factor:.2f} | Gross Gains / Gross Losses |",
        f"| **Payoff Ratio (Win/Loss)** | {payoff_ratio:.2f} | Critical: Must exceed 2.0 if win rate < 40% |",
        f"| **Total Friction Deducted** | -${total_friction_usd:,.2f} | Brokerage fee (0.10%) + Tax (1.0%) + Slippage |",
        f"| **Friction / Net Profit Ratio** | {(total_friction_usd / max(1.0, net_profit))*100:.1f}% | Strategy drag under realistic exchange models |",
        "",
        "---",
        "## 3. Outlier & Concentration Risk (Curve-Fit Check)",
        "> **Quant Warning:** If the Top 3 coins generate more than 60-70% of total profits, the strategy is likely riding a single fluke bull run rather than an institutional edge.",
        "",
        f"- **Top Performer ({top1_ticker}):** ${top1_coin_pnl:+,.2f} ({top1_coin_pnl / max(1.0, net_profit) * 100:.1f}% of net profits)",
        f"- **Top 3 Assets Combined:** ${top3_coin_pnl:+,.2f} (**{top3_conc_pct:.1f}%** of total profits)",
        f"- **Top 5 Assets Combined:** ${top5_coin_pnl:+,.2f} ({(top5_coin_pnl / max(1.0, net_profit) * 100):.1f}% of total profits)",
        f"- **Asset Breadth:** {int(np.sum(coin_pnl_map > 0))} coins profitable / {int(np.sum(pos_capital_deployed > 0))} coins actually traded.",
        "",
        "---",
        "## 4. Annual Performance & Cash Allocation Breakdown",
        "",
        "| Year | Starting Equity | Ending Equity | Realized Trades | Median Coin Ret | Avg Idle Cash ($) | Idle Cash (%) | Peak Concurrent Coins |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for row in annual_rows:
        report_lines.append(
            f"| **{row['year']}** | ${row['start_equity']:,.0f} | ${row['end_equity']:,.0f} | "
            f"{row['trades']} | {row['median_return_pct']:+.1f}% | ${row['avg_idle_cash']:,.0f} | "
            f"{row['idle_cash_pct']:.1f}% | {row['max_concurrent']} |"
        )

    report_lines.extend([
        "",
        "---",
        "## 5. Optimal Hyperparameter Configuration",
        "```json",
        json.dumps(p.as_dict(), indent=2),
        "```",
        "",
        "---",
        "## 6. Complete Universe Backtested",
        f"Total Tickers: {len(aligned.tickers)}",
        "",
        "```text",
        ", ".join(aligned.tickers),
        "```",
    ])

    report_content = "\n".join(report_lines)

    # Save to disk
    md_path = Path(output_dir) / "strategy_audit_dossier.md"
    json_path = Path(output_dir) / "strategy_audit_metrics.json"

    with open(md_path, "w") as f:
        f.write(report_content)

    with open(json_path, "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "trial_number": best_trial.number,
            "score": best_trial.value,
            "cagr_pct": cagr_pct,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_dd_pct": max_dd_pct,
            "total_trades": total_trades,
            "net_profit": net_profit,
            "top3_concentration_pct": top3_conc_pct,
            "annual_breakdown": annual_rows,
            "universe": aligned.tickers,
            "params": p.as_dict(),
        }, f, indent=2)

    logger.info(f"Complete Validation Dossier saved: {md_path}")
    logger.info(f"Machine-readable Audit Metrics saved: {json_path}")
    return report_content

def print_summary(study: optuna.Study, leaderboard: TopKLeaderboard, cfg: RunConfig, elapsed: float) -> None:
    logger.info("=" * 78)
    logger.info(f"Study complete: {len(study.trials)} trials in {elapsed:.1f}s "
                f"({elapsed / max(1, len(study.trials)):.3f}s/trial)")
    n_finite = sum(1 for t in study.trials if t.value is not None and t.value > -1e8)
    logger.info(f"  {n_finite}/{len(study.trials)} trials produced a usable score")
    logger.info("-" * 78)
    logger.info(f"TOP {len(leaderboard.entries)} (saved to {leaderboard.path}):")
    for e in leaderboard.entries:
        m_ = e["metrics"]
        logger.info(
            f"  #{e['rank']}  score={e['score']:.5f}  trial={e['trial_number']:>5}  "
            f"median_ret={m_.get('median_coin_return', float('nan')):+.3f}  "
            f"consistency={m_.get('annual_consistency_factor', float('nan')):+.3f}  "
            f"dd_pen={m_.get('drawdown_penalty', float('nan')):.3f}  "
            f"activity={m_.get('activity_damper', float('nan')):.3f}  "
            f"trades={m_.get('total_trades', '?')}  coins={m_.get('n_coins_traded', '?')}"
        )
    logger.info("=" * 78)


# ==============================================================================
# CLI
# ==============================================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> RunConfig:
    ap = argparse.ArgumentParser(
        description="Multi-asset crypto backtester + Optuna hyperparameter search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    ap.add_argument("--n-coins-target", type=int, default=N_COINS_TARGET)
    ap.add_argument("--max-coins", type=int, default=MAX_COINS,
                    help="max number of coins to include in backtest (selected by top historical volume)")
    ap.add_argument("--start-date", type=str, default=START_DATE,
                    help="explicit backtest start date (YYYY-MM-DD); overrides auto-start percentage")
    ap.add_argument("--auto-start-pct", type=float, default=AUTO_START_MIN_COINS_PCT,
                    help="fraction of coins that must be tradeable to set backtest start date (e.g. 0.60 for 60%%)")
    ap.add_argument("--min-liquid-assets", type=int, default=MIN_LIQUID_ASSETS)
    ap.add_argument("--cache-dir", type=str, default=CACHE_DIR)
    ap.add_argument("--universe-file", type=str, default=UNIVERSE_FILE)
    ap.add_argument("--refresh-cache", action="store_true", default=REFRESH_CACHE)
    ap.add_argument("--synthetic", action="store_true", default=SYNTHETIC)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--sampler", choices=["tpe", "random"], default=SAMPLER)
    ap.add_argument("--n-startup-trials", type=int, default=N_STARTUP_TRIALS)
    ap.add_argument("--n-jobs", type=int, default=N_JOBS)
    ap.add_argument("--min-expected-trades", type=int, default=MIN_EXPECTED_TRADES)
    ap.add_argument("--progress-every", type=int, default=PROGRESS_EVERY_N_TRIALS)
    ap.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    ap.add_argument("--verbosity", type=str, default=VERBOSITY)
    ap.add_argument("--self-test", action="store_true", default=SELF_TEST)
    args = ap.parse_args(argv)

    cfg = RunConfig(
        n_trials=args.n_trials,
        n_coins_target=args.n_coins_target,
        max_coins=args.max_coins,
        start_date=args.start_date,
        auto_start_min_coins_pct=args.auto_start_pct,
        min_liquid_assets=args.min_liquid_assets,
        cache_dir=args.cache_dir,
        universe_file=args.universe_file,
        refresh_cache=args.refresh_cache,
        synthetic=args.synthetic,
        seed=args.seed,
        sampler=args.sampler,
        n_startup_trials=args.n_startup_trials,
        n_jobs=args.n_jobs,
        min_expected_trades=args.min_expected_trades,
        output_dir=args.output_dir,
        verbosity=args.verbosity,
        self_test=args.self_test,
        progress_every=args.progress_every,
    )
    if args.self_test:
        cfg.synthetic = True
        cfg.n_trials = 25
        cfg.n_coins_target = 20
        cfg.max_coins = 15
        cfg.min_liquid_assets = 1
        cfg.output_dir = "./results_self_test"
    cfg.validate()
    return cfg


# ==============================================================================
# MAIN
# ==============================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = parse_args(argv)
    setup_logging(cfg.verbosity)

    logger.info("crypto_backtest_optimizer -- starting run")
    logger.info(f"mode={cfg.mode}  sampler={cfg.sampler}  n_trials={cfg.n_trials}  "
                f"max_coins={cfg.max_coins}  start_date={cfg.start_date}  "
                f"auto_start_pct={cfg.auto_start_min_coins_pct}")
    np.random.seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)

    try:
        aligned = load_or_fetch_universe(cfg)
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return 1

    gi = precompute_global_indicators(aligned, cfg)

    sampler = build_sampler(cfg)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=f"crypto_bt_{int(time.time())}")
    leaderboard = TopKLeaderboard(str(Path(cfg.output_dir) / "top_5_winners.json"), k=5)
    progress = ProgressReporter(total=cfg.n_trials, every_n=cfg.progress_every)
    objective = objective_factory(aligned, gi, cfg)

    t0 = time.time()
    study.optimize(objective, n_trials=cfg.n_trials, n_jobs=cfg.n_jobs,
                   callbacks=[leaderboard.callback, progress], show_progress_bar=False)
    elapsed = time.time() - t0

    print_summary(study, leaderboard, cfg, elapsed)

    if study.best_trial is not None:
        logger.info("Generating full institutional validation dossier for Trial #%d...", study.best_trial.number)
        generate_validation_audit_report(study.best_trial, aligned, gi, cfg, cfg.output_dir)

    trials_path = Path(cfg.output_dir) / "all_trials.csv"
    try:
        study.trials_dataframe().to_csv(trials_path, index=False)
        logger.info(f"Full trial history written to {trials_path}")
    except Exception as e:
        logger.warning(f"Could not write trial history CSV: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())