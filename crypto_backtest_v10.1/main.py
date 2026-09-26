#!/usr/bin/env python3
"""
CRYPTO QUANTITATIVE SWING-TRADING FRAMEWORK (V10.2 - INSTITUTIONAL PROMOTION ENGINE)
===================================================================================
Integrated Synthesis of V10.1 High-Performance Matrix Engine & V7.2 Safeguards:

1. MATRIX VECTORIZATION (from V10.1): High-speed contiguous 2D NumPy array grid (MarketGrid)
   eliminates Pandas .iloc lookups during simulation for sub-second optimization runs.
2. MULTI-EXCHANGE REST INGESTION (from V7.2): Direct REST ingestion waterfall (Binance -> Bybit ->
   OKX -> Yahoo) with JSON metadata sidecars, terminal synchronization, and Parquet caching.
3. PREFIX-INVARIANCE AUDIT (from V7.2): Mathematical proof of zero lookahead leak executed via
   matrix slicing (slice_market_grid) before optimization begins.
4. SYMBOL JACKKNIFING (from V7.2): Matrix-native sub-universe validation (3x 70% symbol masks)
   rejects candidate strategies whose edge is concentrated in isolated asset pumps.
5. EOT ANTI-CHEATING DISCOUNT (from V7.2): Penalizes artificial CAGR inflation from unclosed
   positions marked to market on the terminal bar (END_OF_TEST).
6. 24/7/365 CONTINUOUS TWR & AUDITED FRICTIONS (from V10.1): Multi-benchmark tracking (BTC,
   Equal-Weight Basket, Timed Regimes), TWR Beta, Up/Down capture, and explicit fee/slippage ledger.
7. TOP-N CONSTITUENT SELECTOR: Configurable TOP_N_COINS universe capacity.
8. SANITIZED BASKET BENCHMARK: Daily return clipping prevents bad-tick CAGR blowup.
"""

from __future__ import annotations

import os
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
from typing import Dict, List, Tuple, Any, Optional, Set

import numpy as np
import pandas as pd
import requests
import optuna

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ==============================================================================
# USER CONTROL PANEL
# ==============================================================================

ENGINE_VERSION: str = "V10.2-INSTITUTIONAL"
UNIVERSE_NAME: str = "TOP_CRYPTO_LIQUID"

# --- Configurable Universe Cap ---
TOP_N_COINS: int = 50                   # Set to Top 50 coins by market cap & volume

MACRO_INDEX_NAME: str = "BTC"
MACRO_INDEX_TICKER: str = "BTC"

FORCE_MACRO_REGIME_FILTER: Optional[bool] = None
FORCE_TRAIL_STOP: Optional[bool] = None

START_YEAR: int = 2017
QUOTE_CURRENCY: str = "USD"
INITIAL_CAPITAL: float = 10_000.0
MONTHLY_CONTRIBUTION: float = 500.0
CASH_ANNUAL_YIELD: float = 0.0          # 0.0% pure risk-free yield
MIN_HISTORY_DAYS: int = 250
WARMUP_BARS: int = 300
N_TRIALS: int = 1000
PARALLEL_DOWNLOAD_WORKERS: int = 8
FORCE_REFRESH: bool = False
CACHE_MAX_AGE_HOURS: float = 48.0
WL_MAX_AGE_BARS: int = 15

# Anti-Cheating & Fitness Sizing
EOT_EXCESS_WEIGHT: float = 0.30         # 0.0 = pure rule-closed trades, 0.30 = standard incremental credit

# Crypto Exchange Fee & Execution Parameters
CRYPTO_EXCHANGE_FEE_RATE: float = 0.0010  # 0.10% (10 bps) standard taker fee
BASE_SLIPPAGE_BPS: float = 6.0            # 6 bps base bid-ask spread
IMPACT_COEF_BPS: float = 120.0            # Market impact scaling coefficient
LIQUIDITY_FLOOR_USD: float = 2_000_000.0  # 30-day ADV floor
TRANCHE_FLOOR_USD: float = 50.0           # Minimum order allocation
DEFAULT_MAX_CONCURRENT_TRANCHES: int = 6
MAX_POSITION_EQUITY_PCT: float = 0.25     # Hard cap on aggregate allocation per asset
MAX_ADV_PARTICIPATION: float = 0.015      # Max 1.5% of 30-day ADV
NEIGHBORHOOD_DROP_LIMIT: float = 0.25

DATA_DIR = Path("data_cache_crypto")
OUTPUT_DIR = Path("output_crypto") / f"{UNIVERSE_NAME}_TOP_{TOP_N_COINS}"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_WINNER_FILE = OUTPUT_DIR / "current_winner.json"
FINAL_WINNER_FILE = OUTPUT_DIR / "winner.json"
REPORT_FILE = OUTPUT_DIR / "WINNER_REPORT.md"
TRADES_CSV_FILE = OUTPUT_DIR / "winner_trades.csv"
STUDY_DB = f"sqlite:///{OUTPUT_DIR.resolve() / 'crypto_swing_study.db'}"

# Historical Stress Regimes
STRESS_WINDOWS: Dict[str, Tuple[str, str]] = {
    "2018 Crypto Winter":           ("2018-01-08", "2018-12-15"),
    "2020 COVID Liquidation Crash": ("2020-02-15", "2020-04-15"),
    "2021 May Deleveraging Crash":   ("2021-05-08", "2021-07-25"),
    "2022 Terra/Luna & 3AC Crash":  ("2022-05-01", "2022-07-01"),
    "2022 FTX Insolvency Crisis":    ("2022-11-01", "2022-12-31"),
}

# ==============================================================================
# CRYPTO NOISE & WRAPPER FILTERING ENGINE
# ==============================================================================

STABLECOINS: Set[str] = {
    "USDT", "USDC", "BUSD", "DAI", "FDUSD", "TUSD", "USDD", "USDP", "FRAX",
    "LUSD", "GUSD", "PYUSD", "EURT", "EURS", "USTC", "USDJ", "CUSD", "SUSD",
    "USDE", "USD0", "CRVUSD", "GHO", "FIRST", "MIM", "FEI", "ALUSD", "RLUSD"
}

FIAT_COMMODITY_PEGS: Set[str] = {
    "XAUT", "PAXG", "EUROC", "XSGD", "EURCV"
}

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
    filtered = []
    for s in symbols:
        clean = s.upper().replace("-USD", "").strip()
        if not is_noise_or_wrapper_coin(clean):
            filtered.append(clean)
    seen = set()
    ordered = []
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
# CRYPTO SPOT FEE & EXECUTION SLIPPAGE ENGINE
# ==============================================================================

@dataclass
class FeeBreakdown:
    exchange_fee: float = 0.0
    slippage_cost: float = 0.0

    @property
    def total_frictions(self) -> float:
        return self.exchange_fee + self.slippage_cost

    def __hash__(self) -> int:
        return hash((round(self.exchange_fee, 8), round(self.slippage_cost, 8)))


def compute_buy_cost_audited(gross_usd: float, slip_cost: float) -> Tuple[float, FeeBreakdown]:
    if gross_usd <= 0:
        return 0.0, FeeBreakdown()
    exch_fee = gross_usd * CRYPTO_EXCHANGE_FEE_RATE
    breakdown = FeeBreakdown(exchange_fee=exch_fee, slippage_cost=slip_cost)
    return gross_usd + exch_fee, breakdown


def compute_sell_proceeds_audited(gross_usd: float, slip_cost: float) -> Tuple[float, FeeBreakdown]:
    if gross_usd <= 0:
        return 0.0, FeeBreakdown()
    exch_fee = gross_usd * CRYPTO_EXCHANGE_FEE_RATE
    breakdown = FeeBreakdown(exchange_fee=exch_fee, slippage_cost=slip_cost)
    proceeds = max(0.0, gross_usd - exch_fee)
    return proceeds, breakdown


def compute_max_affordable_tranche(cash: float, adv_30d: float) -> float:
    clean_adv = adv_30d if np.isfinite(adv_30d) else LIQUIDITY_FLOOR_USD
    adv = max(clean_adv, LIQUIDITY_FLOOR_USD)
    usable_cash = max(0.0, cash)
    tranche_guess = usable_cash / (1.0 + CRYPTO_EXCHANGE_FEE_RATE + BASE_SLIPPAGE_BPS / 10000.0)
    for _ in range(3):
        part_rate = min(1.0, max(0.0, tranche_guess / adv))
        slip_mult = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate)) / 10000.0
        tranche_guess = usable_cash / (1.0 + CRYPTO_EXCHANGE_FEE_RATE + slip_mult)
    return float(np.nan_to_num(tranche_guess * (1.0 - 1e-6), nan=0.0))


def _buy_fill_audited(tranche_usd: float, ref_open_price: float, slip_mult: float) -> Tuple[float, float, float, FeeBreakdown]:
    fill_px = ref_open_price * (1.0 + slip_mult)
    units = tranche_usd / ref_open_price if ref_open_price > 0 else 0.0
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


# ==============================================================================
# MULTI-EXCHANGE DATA INGESTION ENGINE WITH METADATA SIDECARS
# ==============================================================================

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

    # Sanitize invalid ticks & wild daily spikes (>50x intraday gap)
    df = df[(df["close"] > 0) & (df["open"] > 0) & (df["high"] >= df["low"])]
    ratio = df["high"] / np.maximum(1e-8, df["low"])
    df = df[ratio < 50.0]

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
            "open": q["open"],
            "high": q["high"],
            "low": q["low"],
            "close": q["close"],
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

    seen = set()
    final_ranked = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            final_ranked.append(c)

    chosen = final_ranked[:top_n]
    with open(cache_path, "w") as f:
        json.dump(chosen, f)
    logging.info(f"Loaded Top {len(chosen)} liquid universe constituents.")
    return chosen


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


def slice_market_grid(grid: MarketGrid, cutoff_bar: int) -> MarketGrid:
    return MarketGrid(
        symbols=grid.symbols,
        dates=grid.dates[:cutoff_bar],
        open_mat=grid.open_mat[:, :cutoff_bar].copy(),
        high_mat=grid.high_mat[:, :cutoff_bar].copy(),
        low_mat=grid.low_mat[:, :cutoff_bar].copy(),
        close_mat=grid.close_mat[:, :cutoff_bar].copy(),
        volume_mat=grid.volume_mat[:, :cutoff_bar].copy(),
        atr14_mat=grid.atr14_mat[:, :cutoff_bar].copy(),
        dvol30_mat=grid.dvol30_mat[:, :cutoff_bar].copy(),
        adx14_mat=grid.adx14_mat[:, :cutoff_bar].copy(),
        delist_mat=grid.delist_mat[:, :cutoff_bar].copy(),
        alive_mat=grid.alive_mat[:, :cutoff_bar].copy(),
        macro_close=grid.macro_close[:cutoff_bar].copy(),
        macro_open=grid.macro_open[:cutoff_bar].copy(),
        months_arr=grid.months_arr[:cutoff_bar].copy(),
        years_arr=grid.years_arr[:cutoff_bar].copy()
    )


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
    symbols = load_universe_constituents(universe_name, top_n=TOP_N_COINS)
    _, macro_df, macro_prov = fetch_single_crypto(MACRO_INDEX_TICKER, start_year, refresh)
    if macro_df is None:
        raise RuntimeError(f"Failed to load crypto macro benchmark: {MACRO_INDEX_TICKER}")
    logging.info(f"Loaded macro benchmark {MACRO_INDEX_TICKER} from [{macro_prov}] ({len(macro_df)} bars).")

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
        dvol30 = np.nan_to_num(dvol30, nan=LIQUIDITY_FLOOR_USD)

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
    cost_usd: float
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
                     state_mat: Optional[np.ndarray] = None, window_label: str = "IN_SAMPLE") -> Dict[str, Any]:
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
        peak_single_asset_pct = 0.0

        for idx, t in enumerate(range(start_bar, end_bar)):
            curr_date = dates[t]
            inflow_today = 0.0

            # 0. Systematic Monthly Inflow
            if t > start_bar and grid.months_arr[t] != grid.months_arr[t - 1]:
                cash += MONTHLY_CONTRIBUTION
                total_inflow += MONTHLY_CONTRIBUTION
                inflow_today = MONTHLY_CONTRIBUTION

            if CASH_ANNUAL_YIELD > 0.0:
                cash += cash * (CASH_ANNUAL_YIELD / 365.0)

            # 1. Delisting Guard
            surviving_tranches = []
            for tr in live_tranches:
                if grid.delist_mat[tr.coin_idx, t]:
                    pnl = tr.proceeds - tr.cost_usd
                    vwap_exit = tr.proceeds / tr.initial_units if tr.initial_units > 0 else 0.0
                    closed_trades.append({
                        "window": window_label, "coin": tr.coin, "pnl": pnl, "ret": pnl / tr.cost_usd if tr.cost_usd > 0 else 0.0,
                        "reason": "DELISTED", "bars": t - tr.entry_bar, "bars_to_peak": tr.peak_bar - tr.entry_bar,
                        "mfe_pct": ((tr.highest_high - tr.entry_price) / tr.entry_price) * 100.0,
                        "mae_pct": ((tr.lowest_low - tr.entry_price) / tr.entry_price) * 100.0,
                        "entry_date": tr.entry_date, "exit_date": curr_date, "exit_idx": idx,
                        "global_entry_bar": tr.entry_bar, "global_exit_bar": t,
                        "entry_price": tr.entry_price, "exit_price": vwap_exit,
                        "cost_usd": tr.cost_usd, "proceeds": tr.proceeds,
                        "frictions": tr.fee_acc, "from_watchlist": tr.from_watchlist,
                    })
                else:
                    surviving_tranches.append(tr)
            live_tranches = surviving_tranches

            # 2. Daily Position Management
            macro_bear_confirmed = False
            if macro_active_exit and t >= 3:
                macro_bear_confirmed = (not macro_ok[t - 1]) and (not macro_ok[t - 2]) and (not macro_ok[t - 3])

            surviving_tranches = []
            for tr in live_tranches:
                c_i = tr.coin_idx

                if not grid.alive_mat[c_i, t]:
                    surviving_tranches.append(tr)
                    continue

                adv = max(grid.dvol30_mat[c_i, t - 1], LIQUIDITY_FLOOR_USD)
                o_bar = grid.open_mat[c_i, t]
                h_bar = grid.high_mat[c_i, t]
                l_bar = grid.low_mat[c_i, t]
                bars_held = t - tr.entry_bar

                exit_triggered = False
                exit_reason = ""
                raw_exit_px = o_bar

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
                        _, credit, fee_tp = _sell_fill_audited(close_units, max(o_bar, tp_price), slip_tp)
                        cash += credit
                        tr.proceeds += credit
                        tr.tp_proceeds += credit
                        tr.units -= close_units
                        tr.tp_done = True
                        tr.fee_acc.exchange_fee += fee_tp.exchange_fee
                        tr.fee_acc.slippage_cost += fee_tp.slippage_cost

                        if tp_be and tr.current_sl < (tr.entry_price * 1.002):
                            tr.current_sl = tr.entry_price * 1.002
                            tr.stop_reason = "BREAKEVEN_SL"

                if exit_triggered:
                    part_rate_exit = min(1.0, max(0.0, (tr.units * raw_exit_px) / adv))
                    slip_exit = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate_exit)) / 10000.0
                    fill_px, proceeds, fee_exit = _sell_fill_audited(tr.units, raw_exit_px, slip_exit)

                    cash += proceeds
                    total_proceeds = tr.proceeds + proceeds
                    pnl = total_proceeds - tr.cost_usd
                    vwap_exit = total_proceeds / tr.initial_units if tr.initial_units > 0 else fill_px

                    tr.fee_acc.exchange_fee += fee_exit.exchange_fee
                    tr.fee_acc.slippage_cost += fee_exit.slippage_cost

                    closed_trades.append({
                        "window": window_label, "coin": tr.coin, "pnl": pnl, "ret": pnl / tr.cost_usd if tr.cost_usd > 0 else 0.0,
                        "reason": exit_reason, "bars": bars_held, "bars_to_peak": tr.peak_bar - tr.entry_bar,
                        "mfe_pct": ((tr.highest_high - tr.entry_price) / tr.entry_price) * 100.0,
                        "mae_pct": ((tr.lowest_low - tr.entry_price) / tr.entry_price) * 100.0,
                        "entry_date": tr.entry_date, "exit_date": curr_date, "exit_idx": idx,
                        "global_entry_bar": tr.entry_bar, "global_exit_bar": t,
                        "entry_price": tr.entry_price, "exit_price": vwap_exit,
                        "cost_usd": tr.cost_usd, "proceeds": total_proceeds,
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

            # 3. New Candidates Evaluation
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
                    if grid.dvol30_mat[c_i, t - 1] < LIQUIDITY_FLOOR_USD:
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

            # 4. Sizing & Order Fill
            def _record_unfilled(reason: Optional[str], default_bucket: str = "expired_unfilled"):
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

            if watchlist and cash >= TRANCHE_FLOOR_USD and len(live_tranches) < max_slots:
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
                    if cash < TRANCHE_FLOOR_USD:
                        item["unfilled_reason"] = "cash_starved"
                        unfilled.append(item)
                        continue

                    open_active_cap = sum(tr.units * grid.open_mat[tr.coin_idx, t] for tr in live_tranches)
                    current_equity = cash + open_active_cap
                    max_pos_cap = current_equity * MAX_POSITION_EQUITY_PCT
                    dynamic_slot_target = min(max_pos_cap, cash / float(open_slots))

                    current_asset_exposure = sum(tr.units * today_open for tr in live_tranches if tr.coin_idx == c_i)
                    remaining_asset_capacity = max(0.0, max_pos_cap - current_asset_exposure)
                    if remaining_asset_capacity < TRANCHE_FLOOR_USD:
                        item["unfilled_reason"] = "scrip_risk_cap"
                        unfilled.append(item)
                        continue

                    adv_30d = max(grid.dvol30_mat[c_i, t - 1], LIQUIDITY_FLOOR_USD)
                    liquidity_cap_usd = adv_30d * MAX_ADV_PARTICIPATION
                    scaled_tranche_ceiling = max(10_000.0, current_equity * 0.35)
                    target_usd = min(dynamic_slot_target, scaled_tranche_ceiling, liquidity_cap_usd, remaining_asset_capacity)

                    max_affordable = compute_max_affordable_tranche(cash, adv_30d)
                    tranche_usd = min(max_affordable, max(TRANCHE_FLOOR_USD, target_usd))

                    part_rate = min(1.0, max(0.0, tranche_usd / adv_30d))
                    slip_mult = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate)) / 10000.0

                    fill_px, units, total_cost, fee_buy = _buy_fill_audited(tranche_usd, today_open, slip_mult)

                    if (cash >= total_cost and tranche_usd >= TRANCHE_FLOOR_USD and units > 0 and
                            len(live_tranches) < max_slots and coin_layers < max_pyramid):

                        if max_affordable < target_usd:
                            binding_stats["cash_constrained"] += 1
                        elif target_usd == scaled_tranche_ceiling:
                            binding_stats["tranche_ceiling"] += 1
                        elif target_usd == liquidity_cap_usd:
                            binding_stats["liquidity_cap"] += 1
                        elif target_usd < TRANCHE_FLOOR_USD:
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

                        if c_today <= sl_price:
                            part_rate_exit = min(1.0, max(0.0, (units * c_today) / adv_30d))
                            slip_exit = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate_exit)) / 10000.0
                            fill_exit_px, proceeds, fee_exit = _sell_fill_audited(units, c_today, slip_exit)
                            cash += proceeds
                            pnl = proceeds - total_cost

                            fee_buy.exchange_fee += fee_exit.exchange_fee
                            fee_buy.slippage_cost += fee_exit.slippage_cost

                            closed_trades.append({
                                "window": window_label, "coin": item["coin"], "pnl": pnl, "ret": pnl / total_cost if total_cost > 0 else 0.0,
                                "reason": "STOP_LOSS", "bars": 0, "bars_to_peak": 0,
                                "mfe_pct": ((max(fill_px, h_today) - fill_px) / fill_px) * 100.0,
                                "mae_pct": ((l_today - fill_px) / fill_px) * 100.0,
                                "entry_date": curr_date, "exit_date": curr_date, "exit_idx": idx,
                                "global_entry_bar": t, "global_exit_bar": t,
                                "entry_price": fill_px, "exit_price": fill_exit_px,
                                "cost_usd": total_cost, "proceeds": proceeds,
                                "frictions": fee_buy, "from_watchlist": (t > item["signal_bar"]),
                            })
                        else:
                            live_tranches.append(Tranche(
                                tid=tid_counter, coin=item["coin"], coin_idx=c_i, entry_bar=t, entry_date=curr_date,
                                entry_price=fill_px, initial_units=units, units=units, cost_usd=total_cost,
                                entry_atr=item["entry_atr"], current_sl=sl_price, stop_reason="STOP_LOSS",
                                highest_high=max(fill_px, h_today), lowest_low=min(fill_px, l_today),
                                peak_bar=t, trough_bar=t, layer=coin_layers + 1,
                                from_watchlist=(t > item["signal_bar"]), wait_days=(t - item["signal_bar"]),
                                proceeds=0.0, fee_acc=fee_buy
                            ))
                    else:
                        if coin_layers >= max_pyramid:
                            item["unfilled_reason"] = "pyramid_blocked"
                        elif cash < total_cost or tranche_usd < TRANCHE_FLOOR_USD:
                            item["unfilled_reason"] = "cash_starved"
                        else:
                            item["unfilled_reason"] = "slot_saturated"
                        unfilled.append(item)
                watchlist = unfilled

            if wl_mode == "WL_NONE" and watchlist:
                for item in watchlist:
                    _record_unfilled(item.get("unfilled_reason"), default_bucket="slot_saturated_dropped")
                watchlist = []

            # 5. Watchlist Lifecycle
            if wl_mode != "WL_NONE":
                surviving_watchlist = []
                for item in watchlist:
                    c_i = item["coin_idx"]
                    l_bar, h_bar = grid.low_mat[c_i, t], grid.high_mat[c_i, t]

                    shadow_stopped = (l_bar <= item["shadow_stop"])
                    shadow_exited = (xt in (3, 4, 5, 6, 7)) and exit_mat[c_i, t]
                    shadow_expired = (t - item["signal_bar"]) >= WL_MAX_AGE_BARS

                    if shadow_expired or shadow_stopped or shadow_exited:
                        _record_unfilled(item.get("unfilled_reason"), default_bucket="expired_unfilled")
                    else:
                        if h_bar > item["highest_high"]:
                            item["highest_high"] = h_bar
                        if trail_mult > 0.0:
                            item["shadow_stop"] = max(item["shadow_stop"], item["highest_high"] - (trail_mult * grid.atr14_mat[c_i, t]))
                        surviving_watchlist.append(item)
                watchlist = surviving_watchlist

            # 6. Horizon Terminal Mark
            if t == end_bar - 1:
                for tr in list(live_tranches):
                    c_i = tr.coin_idx
                    c_px = grid.close_mat[c_i, t]
                    if np.isfinite(c_px) and c_px > 0:
                        adv = max(grid.dvol30_mat[c_i, t - 1], LIQUIDITY_FLOOR_USD)
                        part_rate = min(1.0, max(0.0, (tr.units * c_px) / adv))
                        slip_mult = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate)) / 10000.0
                        fill_px, proceeds, fee_term = _sell_fill_audited(tr.units, c_px, slip_mult)
                        cash += proceeds
                        total_proceeds = tr.proceeds + proceeds
                        pnl = total_proceeds - tr.cost_usd
                        vwap_exit = total_proceeds / tr.initial_units if tr.initial_units > 0 else fill_px

                        tr.fee_acc.exchange_fee += fee_term.exchange_fee
                        tr.fee_acc.slippage_cost += fee_term.slippage_cost

                        closed_trades.append({
                            "window": window_label, "coin": tr.coin, "pnl": pnl, "ret": pnl / tr.cost_usd if tr.cost_usd > 0 else 0.0,
                            "reason": "END_OF_TEST", "bars": t - tr.entry_bar,
                            "bars_to_peak": tr.peak_bar - tr.entry_bar,
                            "mfe_pct": ((tr.highest_high - tr.entry_price) / tr.entry_price) * 100.0,
                            "mae_pct": ((tr.lowest_low - tr.entry_price) / tr.entry_price) * 100.0,
                            "entry_date": tr.entry_date, "exit_date": curr_date, "exit_idx": idx,
                            "global_entry_bar": tr.entry_bar, "global_exit_bar": t,
                            "entry_price": tr.entry_price, "exit_price": vwap_exit,
                            "cost_usd": tr.cost_usd, "proceeds": total_proceeds,
                            "frictions": tr.fee_acc, "from_watchlist": tr.from_watchlist,
                        })
                live_tranches = []

            # 7. Time-Weighted Return (TWR) Tracking
            asset_mkt_vals: Dict[int, float] = {}
            for tr in live_tranches:
                val = tr.units * grid.close_mat[tr.coin_idx, t]
                asset_mkt_vals[tr.coin_idx] = asset_mkt_vals.get(tr.coin_idx, 0.0) + val

            current_active = sum(asset_mkt_vals.values())
            eod_equity = cash + current_active

            if eod_equity > 0 and asset_mkt_vals:
                max_asset_today = max(asset_mkt_vals.values())
                peak_single_asset_pct = max(peak_single_asset_pct, max_asset_today / eod_equity)

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
            "peak_single_asset_pct": float(peak_single_asset_pct * 100.0),
        }


# ==============================================================================
# AUDITED TIME-WEIGHTED RETURN (TWR) 365-DAY METRIC ENGINE
# ==============================================================================

def calculate_benchmarks(grid: MarketGrid, start_bar: int, end_bar: int, macro_ok: Optional[np.ndarray] = None) -> Dict[str, Any]:
    n_bars = end_bar - start_bar

    # 1. Macro Index (BTC) TWR
    btc_p = grid.macro_close[start_bar:end_bar]
    btc_daily_ret = np.diff(btc_p) / np.maximum(1e-6, btc_p[:-1])
    btc_twr = np.insert(np.cumprod(1.0 + btc_daily_ret), 0, 1.0)
    btc_cagr = (btc_twr[-1] ** (365.0 / max(1, n_bars))) - 1.0

    peak_b = np.maximum.accumulate(btc_twr)
    dd_b = np.where(peak_b > 0, (peak_b - btc_twr) / peak_b, 0.0)
    btc_max_dd = float(np.max(dd_b)) * 100.0 if len(dd_b) > 0 else 0.0

    # 2. Equal-Weight Clean Crypto Basket (Protected against Bad Ticks)
    basket_close = grid.close_mat[:, start_bar:end_bar]
    alive_slice = grid.alive_mat[:, start_bar:end_bar]
    prev_close = np.maximum(1e-6, basket_close[:, :-1])
    raw_basket_returns = (basket_close[:, 1:] - prev_close) / prev_close
    
    # Clip extreme single-day asset return anomalies [-95%, +300%] to prevent corrupted Yahoo data from exploding CAGR
    clean_basket_returns = np.clip(raw_basket_returns, -0.95, 3.0)

    daily_basket_ret = np.zeros(n_bars - 1, dtype=np.float64)
    for b in range(n_bars - 1):
        alive_today = alive_slice[:, b] & alive_slice[:, b + 1]
        if np.any(alive_today):
            daily_basket_ret[b] = np.mean(clean_basket_returns[alive_today, b])

    basket_twr = np.insert(np.cumprod(1.0 + daily_basket_ret), 0, 1.0)
    basket_cagr = (basket_twr[-1] ** (365.0 / max(1, n_bars))) - 1.0

    peak_bk = np.maximum.accumulate(basket_twr)
    dd_bk = np.where(peak_bk > 0, (peak_bk - basket_twr) / peak_bk, 0.0)
    basket_max_dd = float(np.max(dd_bk)) * 100.0 if len(dd_bk) > 0 else 0.0

    # 3. Macro Regime-Timed Benchmarks
    daily_cash_ret = CASH_ANNUAL_YIELD / 365.0
    if macro_ok is not None:
        macro_slice = macro_ok[start_bar:end_bar - 1]
        timed_btc_daily = np.where(macro_slice, btc_daily_ret, daily_cash_ret)
        timed_btc_twr = np.insert(np.cumprod(1.0 + timed_btc_daily), 0, 1.0)
        timed_btc_cagr = (timed_btc_twr[-1] ** (365.0 / max(1, n_bars))) - 1.0
        timed_btc_total_return = (timed_btc_twr[-1] - 1.0) * 100.0
    else:
        timed_btc_cagr = btc_cagr
        timed_btc_total_return = (btc_twr[-1] - 1.0) * 100.0

    return {
        "macro_twr_cagr": float(btc_cagr * 100.0),
        "macro_total_return": float((btc_twr[-1] - 1.0) * 100.0),
        "macro_max_dd": float(btc_max_dd),
        "macro_daily_ret": btc_daily_ret,
        "basket_twr_cagr": float(basket_cagr * 100.0),
        "basket_total_return": float((basket_twr[-1] - 1.0) * 100.0),
        "basket_max_dd": float(basket_max_dd),
        "timed_macro_cagr": float(timed_btc_cagr * 100.0),
        "timed_macro_total_return": float(timed_btc_total_return),
    }


def _calc_sub_metrics(tdf: pd.DataFrame, cagr: float, total_ret: float) -> Dict[str, Any]:
    if len(tdf) == 0:
        return {
            "trades": 0, "net_pnl": 0.0, "cagr": 0.0, "total_return": 0.0,
            "win_rate": 0.0, "pure_win_rate": 0.0, "be_rate": 0.0, "loss_rate": 0.0,
            "profit_factor": 0.0
        }
    net_pnl = float(tdf["pnl"].sum())
    is_be = tdf["reason"].str.contains("BREAKEVEN") | (tdf["ret"].abs() <= 0.0030)
    wins = tdf[(tdf["pnl"] > 0) & (~is_be)]["pnl"]
    losses = tdf[(tdf["pnl"] < 0) & (~is_be)]["pnl"].abs()
    be_trades = tdf[is_be]

    n_tot = len(tdf)
    win_rate = (len(tdf[tdf["pnl"] > 0]) / n_tot) * 100.0
    pure_win_rate = (len(wins) / n_tot) * 100.0
    be_rate = (len(be_trades) / n_tot) * 100.0
    loss_rate = (len(losses) / n_tot) * 100.0
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
    total_inflow = results["total_inflow"]

    if len(twr_series) > 0:
        peak_twr = np.maximum.accumulate(twr_series)
        dd_curve = np.where(peak_twr > 0, (peak_twr - twr_series) / peak_twr, 0.0)
        max_dd = float(np.max(dd_curve)) * 100.0 if len(dd_curve) > 0 else 0.0
        total_ret_full = float((twr_series[-1] - 1.0) * 100.0)
        cagr_full = float(((twr_series[-1]) ** (365.0 / max(1, n_bars)) - 1.0) * 100.0)
    else:
        max_dd, total_ret_full, cagr_full = 0.0, 0.0, 0.0

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
            "cagr_organic": 0.0,
            "max_dd": float(max_dd),
            "utilization": float(utilization * 100.0),
        }

    # Anti-Cheating Discounting (V7.2 Port)
    final_equity = float(eq[-1]) if len(eq) > 0 else total_inflow
    organic_final_equity = max(1e-6, final_equity - eot_pnl)
    final_twr = max(1e-6, float(twr_series[-1]))
    organic_twr_ratio = np.clip(organic_final_equity / max(1e-6, final_equity), 0.0, 1.0)
    organic_twr_end = final_twr * organic_twr_ratio
    years_365 = max(1, n_bars) / 365.0
    cagr_organic = float(((organic_twr_end ** (1.0 / years_365)) - 1.0) * 100.0)

    unclosed_excess_cagr = max(0.0, cagr_full - cagr_organic)
    anti_cheat_cagr = max(0.0, cagr_organic) + (EOT_EXCESS_WEIGHT * unclosed_excess_cagr)

    # Winsorized Expectancy Discounting (95th percentile clipping)
    if len(tdf) > 5:
        trade_rets = tdf["ret"].values
        p95 = float(np.percentile(trade_rets, 95))
        clipped_rets = np.minimum(trade_rets, p95)
        expectance_discount = max(0.2, float(np.sum(clipped_rets) / max(1e-6, np.sum(trade_rets))))
    else:
        expectance_discount = 1.0

    trimmed_cagr = anti_cheat_cagr * expectance_discount
    rf_cagr = CASH_ANNUAL_YIELD * 100.0
    cagr_excess_rf = max(0.0, trimmed_cagr - rf_cagr)

    exp_matched_benchmark = rf_cagr + utilization * max(0.0, benchmark_cagr - rf_cagr)
    alpha_excess = trimmed_cagr - exp_matched_benchmark

    if alpha_excess >= 0.0:
        alpha_mult = 1.0 + float(np.log1p(alpha_excess / 12.0))
    else:
        alpha_mult = float(np.exp(alpha_excess / 8.0))

    mar_ratio = cagr_excess_rf / max(6.0, max_dd)
    trade_confidence = min(1.0, len(tdf) / 50.0)
    pf_booster = 1.0 + float(np.log1p(max(0.0, full["profit_factor"] - 1.0)))

    score = np.log1p(max(0.0, mar_ratio)) * trade_confidence * pf_booster * alpha_mult

    return {
        "score": float(np.nan_to_num(score, nan=-10.0)), "full": full,
        "eot_trades": eot_trades, "eot_pnl": eot_pnl,
        "cagr_organic": float(cagr_organic),
        "max_dd": float(max_dd),
        "utilization": float(utilization * 100.0),
    }


# ==============================================================================
# LOOKAHEAD LEAK PREFIX-INVARIANCE AUDIT (MATRIX ARCHITECTURE)
# ==============================================================================

def verify_causality_matrix(grid: MarketGrid):
    logging.info("Executing Behavioral Prefix-Invariance Audit on contiguous NumPy MarketGrid...")
    n_bars = len(grid.dates)
    if n_bars < 500:
        return

    t_cutoff = n_bars - 80
    t_cutoff_date = grid.dates[t_cutoff]
    grid_trunc = slice_market_grid(grid, t_cutoff)

    test_configs = [
        {"entry_type": 0, "entry_ma_len": 20, "entry_ma_type": 0, "use_market_macro_system": True,
         "macro_ma_len": 50, "macro_ma_type": 0, "adx_thresh": 0.0, "max_pyramid_layers": 1,
         "wl_mode": "WL_NONE", "use_global_tp": False, "exit_type": 0, "sl_mult": 2.5, "trail_atr_mult": 4.0},
        {"entry_type": 1, "rsi_f_len": 14, "rsi_f_smt": 5, "rsi_s_len": 40, "rsi_s_smt": 10,
         "use_rsi_trend_filter": False, "use_market_macro_system": False, "adx_thresh": 0.0,
         "max_pyramid_layers": 1, "wl_mode": "WL_STRONGEST_MOMENTUM", "use_global_tp": False,
         "exit_type": 4, "sl_mult": 3.0, "exit_rsi_f_len": 14, "exit_rsi_f_smt": 5, "exit_rsi_s_len": 40, "exit_rsi_s_smt": 10},
        {"entry_type": 2, "xover_short_len": 20, "xover_short_type": 1, "xover_gap": 30, "xover_long_len": 50,
         "xover_long_type": 1, "use_market_macro_system": True, "macro_ma_len": 100, "macro_ma_type": 1,
         "adx_thresh": 15.0, "max_pyramid_layers": 1, "wl_mode": "WL_NONE", "use_global_tp": False,
         "exit_type": 5, "sl_mult": 2.5, "exit_xover_short_len": 20, "exit_xover_short_type": 1, "exit_xover_gap": 30,
         "exit_xover_long_len": 50, "exit_xover_long_type": 1},
        {"entry_type": 3, "vol_ma_len": 20, "vol_mult": 2.5, "price_lookback": 20, "body_atr_mult": 1.2,
         "use_market_macro_system": False, "adx_thresh": 0.0, "max_pyramid_layers": 1, "wl_mode": "WL_DEEPEST_DISCOUNT",
         "use_global_tp": False, "exit_type": 6, "sl_mult": 3.0, "exit_vol_ma_len": 20, "exit_vol_mult": 3.0},
        {"entry_type": 4, "bb_entry_len": 20, "bb_entry_std": 2.0, "use_market_macro_system": True,
         "macro_ma_len": 100, "macro_ma_type": 0, "adx_thresh": 0.0, "max_pyramid_layers": 1,
         "wl_mode": "WL_NONE", "use_global_tp": False, "exit_type": 7, "sl_mult": 2.5, "bb_exit_len": 20}
    ]

    for cfg in test_configs:
        cfg = reconstitute_params(cfg)
        raw_full, ent_full, ext_full, mac_full, st_full = compile_signals_fast(grid, cfg)
        engine_full = BacktestEngine(params=cfg)
        res_full = engine_full.run_interval(grid, raw_full, ent_full, ext_full, mac_full, 300, t_cutoff, state_mat=st_full)
        trades_full = res_full["trades"]

        raw_tr, ent_tr, ext_tr, mac_tr, st_tr = compile_signals_fast(grid_trunc, cfg)
        engine_tr = BacktestEngine(params=cfg)
        res_tr = engine_tr.run_interval(grid_trunc, raw_tr, ent_tr, ext_tr, mac_tr, 300, t_cutoff, state_mat=st_tr)
        trades_tr = res_tr["trades"]

        f_prior = trades_full[(trades_full["reason"] != "END_OF_TEST") & (trades_full["exit_date"] < t_cutoff_date)].sort_values(by=["coin", "entry_date"]).reset_index(drop=True)
        t_prior = trades_tr[(trades_tr["reason"] != "END_OF_TEST") & (trades_tr["exit_date"] < t_cutoff_date)].sort_values(by=["coin", "entry_date"]).reset_index(drop=True)

        if len(f_prior) != len(t_prior):
            raise RuntimeError(
                f"Prefix-Invariance Violation in Arch {cfg.get('entry_type')}! Trade count mismatch: "
                f"Full={len(f_prior)}, Truncated={len(t_prior)}. Lookahead leak detected in NumPy kernels!"
            )

        for i in range(len(f_prior)):
            r_f, r_t = f_prior.iloc[i], t_prior.iloc[i]
            if r_f["coin"] != r_t["coin"] or r_f["entry_date"] != r_t["entry_date"]:
                raise RuntimeError(f"Trade Alignment Drift in Arch {cfg.get('entry_type')} on trade {i}: {r_f['coin']} vs {r_t['coin']}")
            if abs(r_f["entry_price"] - r_t["entry_price"]) > 1e-4:
                raise RuntimeError(f"Entry Price Drift in Arch {cfg.get('entry_type')}: Full={r_f['entry_price']}, Trunc={r_t['entry_price']}")
            if abs(r_f["pnl"] - r_t["pnl"]) > 1e-4:
                raise RuntimeError(f"PnL Drift in Arch {cfg.get('entry_type')}: Full={r_f['pnl']}, Trunc={r_t['pnl']}")

    logging.info(f"Causality Audit Passed across all {len(test_configs)} trading architectures (Zero Lookahead Proven).")


# ==============================================================================
# ANTI-OVERFITTING: NEIGHBORHOOD AUDIT & CAUSALITY
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
    pass_rate = float(np.mean(valid_mask)) if len(scores_arr) > 0 else 0.0
    p10_score = float(np.percentile(scores_arr, 10)) if len(scores_arr) > 0 else -10.0
    median_neighbor_score = float(np.median(scores_arr)) if len(scores_arr) > 0 else -10.0

    is_stable = (
        pass_rate >= 0.80 and
        median_neighbor_score >= (candidate_score * (1.0 - NEIGHBORHOOD_DROP_LIMIT)) and
        p10_score > 0.0
    )
    plateau_score = min(candidate_score, median_neighbor_score)
    return is_stable, plateau_score, pass_rate, neighbor_scores


# ==============================================================================
# OPTUNA HYPERPARAMETER SEARCH SPACE
# ==============================================================================

def sample_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    p = {}
    p["entry_type"] = trial.suggest_categorical("entry_type", [0, 1, 2, 3, 4])
    p["adx_thresh"] = trial.suggest_categorical("adx_thresh", [0.0, 15.0, 20.0, 25.0, 30.0])

    if p["entry_type"] == 0:
        p["entry_ma_len"] = trial.suggest_int("entry_ma_len", 10, 150, step=5)
        p["entry_ma_type"] = trial.suggest_categorical("entry_ma_type", [0, 1, 2, 3, 4])
    elif p["entry_type"] == 1:
        p["rsi_f_len"] = trial.suggest_int("rsi_f_len", 8, 50, step=2)
        p["rsi_f_smt"] = trial.suggest_int("rsi_f_smt", 4, 30, step=2)
        p["rsi_s_len"] = trial.suggest_int("rsi_s_len", 10, 60, step=2)
        p["rsi_s_smt"] = trial.suggest_int("rsi_s_smt", 4, 30, step=2)
        p["use_rsi_trend_filter"] = trial.suggest_categorical("use_rsi_trend_filter", [True, False])
        if p["use_rsi_trend_filter"]:
            p["rsi_trend_ma_len"] = trial.suggest_int("rsi_trend_ma_len", 20, 150, step=10)
            p["rsi_trend_ma_type"] = trial.suggest_categorical("rsi_trend_ma_type", [0, 1, 2, 3, 4])
    elif p["entry_type"] == 2:
        p["xover_short_len"] = trial.suggest_int("xover_short_len", 8, 60, step=4)
        p["xover_short_type"] = trial.suggest_categorical("xover_short_type", [0, 1, 2, 3, 4])
        gap = trial.suggest_int("xover_gap", 10, 60, step=5)
        p["xover_gap"] = gap
        p["xover_long_len"] = min(180, p["xover_short_len"] + gap)
        p["xover_long_type"] = trial.suggest_categorical("xover_long_type", [0, 1, 2, 3, 4])
    elif p["entry_type"] == 3:
        p["vol_ma_len"] = trial.suggest_int("vol_ma_len", 10, 40, step=2)
        p["vol_mult"] = round(trial.suggest_float("vol_mult", 1.4, 4.0, step=0.1), 1)
        p["price_lookback"] = trial.suggest_int("price_lookback", 10, 40, step=2)
        p["body_atr_mult"] = round(trial.suggest_float("body_atr_mult", 0.4, 2.2, step=0.1), 1)
    elif p["entry_type"] == 4:
        p["bb_entry_len"] = trial.suggest_int("bb_entry_len", 10, 40, step=2)
        p["bb_entry_std"] = round(trial.suggest_float("bb_entry_std", 1.5, 3.0, step=0.1), 1)

    if FORCE_MACRO_REGIME_FILTER is not None:
        p["use_market_macro_system"] = FORCE_MACRO_REGIME_FILTER
    else:
        p["use_market_macro_system"] = trial.suggest_categorical("use_market_macro_system", [True, False])

    if p["use_market_macro_system"]:
        p["macro_ma_len"] = trial.suggest_int("macro_ma_len", 50, 200, step=10)
        p["macro_ma_type"] = trial.suggest_categorical("macro_ma_type", [0, 1, 2, 3, 4])
        p["macro_active_exit"] = trial.suggest_categorical("macro_active_exit", [True, False])
    else:
        p["macro_active_exit"] = False

    p["max_concurrent_tranches"] = trial.suggest_int("max_concurrent_tranches", 4, 10, step=1)
    p["max_pyramid_layers"] = 1 if p["entry_type"] not in (0, 3, 4) else trial.suggest_categorical("max_pyramid_layers", [1, 2])
    p["wl_mode"] = trial.suggest_categorical("wl_mode", ["WL_NONE", "WL_DEEPEST_DISCOUNT", "WL_STRONGEST_MOMENTUM"])

    p["use_global_tp"] = trial.suggest_categorical("use_global_tp", [True, False])
    if p["use_global_tp"]:
        p["tp_mult"] = round(trial.suggest_float("tp_mult", 2.0, 7.0, step=0.5), 1)
        p["tp_size_pct"] = round(trial.suggest_float("tp_size_pct", 25.0, 75.0, step=5.0), 1)
        p["tp_move_sl_be"] = trial.suggest_categorical("tp_move_sl_be", [True, False])

    p["be_trigger_atr"] = trial.suggest_categorical("be_trigger_atr", [0.0, 1.5, 2.5, 3.5])
    p["max_holding_bars"] = trial.suggest_int("max_holding_bars", 10, 45, step=5)
    p["sl_mult"] = round(trial.suggest_float("sl_mult", 2.0, 6.0, step=0.2), 1)

    if FORCE_TRAIL_STOP is True:
        p["exit_type"] = trial.suggest_categorical("exit_type", [0, 3, 4, 5, 6, 7])
        p["trail_pct"] = 0.0
        p["trail_atr_mult"] = 0.0
    else:
        p["exit_type"] = trial.suggest_categorical("exit_type", [0, 1, 3, 4, 5, 6, 7])
        if p["exit_type"] == 1:
            p["trail_pct"] = round(trial.suggest_float("trail_pct", 6.0, 30.0, step=1.0), 1)
            p["trail_atr_mult"] = 0.0
        else:
            p["trail_atr_mult"] = trial.suggest_categorical("trail_atr_mult", [0.0, 2.5, 3.5, 5.0])

    if p["exit_type"] == 3:
        p["exit_ma_len"] = trial.suggest_int("exit_ma_len", 10, 120, step=5)
        p["exit_ma_type"] = trial.suggest_categorical("exit_ma_type", [0, 1, 2, 3, 4])
    elif p["exit_type"] == 4:
        p["exit_rsi_f_len"] = trial.suggest_int("exit_rsi_f_len", 8, 50, step=2)
        p["exit_rsi_f_smt"] = trial.suggest_int("exit_rsi_f_smt", 4, 24, step=2)
        p["exit_rsi_s_len"] = trial.suggest_int("exit_rsi_s_len", 10, 50, step=2)
        p["exit_rsi_s_smt"] = trial.suggest_int("exit_rsi_s_smt", 4, 24, step=2)
    elif p["exit_type"] == 5:
        p["exit_xover_short_len"] = trial.suggest_int("exit_xover_short_len", 8, 60, step=4)
        p["exit_xover_short_type"] = trial.suggest_categorical("exit_xover_short_type", [0, 1, 2, 3, 4])
        exit_gap = trial.suggest_int("exit_xover_gap", 10, 50, step=5)
        p["exit_xover_gap"] = exit_gap
        p["exit_xover_long_len"] = min(150, p["exit_xover_short_len"] + exit_gap)
        p["exit_xover_long_type"] = trial.suggest_categorical("exit_xover_long_type", [0, 1, 2, 3, 4])
    elif p["exit_type"] == 6:
        p["exit_vol_ma_len"] = trial.suggest_int("exit_vol_ma_len", 10, 30, step=5)
        p["exit_vol_mult"] = round(trial.suggest_float("exit_vol_mult", 0.8, 3.0, step=0.1), 1)
    elif p["exit_type"] == 7:
        p["bb_exit_len"] = trial.suggest_int("bb_exit_len", 10, 50, step=2)

    return p


def reconstitute_params(params: Dict[str, Any]) -> Dict[str, Any]:
    p = copy.deepcopy(params)
    for k in list(p.keys()):
        if isinstance(p[k], float):
            p[k] = round(p[k], 2)

    if p.get("entry_type") == 2 and "xover_gap" in p:
        p["xover_long_len"] = min(180, p["xover_short_len"] + p["xover_gap"])
    if p.get("exit_type") == 5 and "exit_xover_gap" in p:
        p["exit_xover_long_len"] = min(150, p["exit_xover_short_len"] + p["exit_xover_gap"])

    p.setdefault("use_market_macro_system", False if FORCE_MACRO_REGIME_FILTER is None else FORCE_MACRO_REGIME_FILTER)
    p.setdefault("macro_active_exit", False)
    p.setdefault("max_pyramid_layers", 1)
    p.setdefault("adx_thresh", 0.0)
    p.setdefault("be_trigger_atr", 0.0)
    p.setdefault("trail_atr_mult", 0.0)
    p.setdefault("use_global_tp", False)
    p.setdefault("wl_mode", "WL_NONE")
    return p


def _canonical_func_source(fn) -> str:
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
    str(TOP_N_COINS) +
    str(FORCE_MACRO_REGIME_FILTER) +
    str(FORCE_TRAIL_STOP) +
    str(START_YEAR) +
    str(EOT_EXCESS_WEIGHT)
)
PARAM_SPACE_HASH: str = hashlib.sha256(ENGINE_HASH_INPUT.encode()).hexdigest()[:8]
STUDY_NAME = f"crypto_swing_{ENGINE_VERSION.lower()}_{UNIVERSE_NAME.lower()}_top{TOP_N_COINS}_{PARAM_SPACE_HASH}"


# ==============================================================================
# TWR BETA, CAPTURE, ABLATIONS & MATCHED PLACEBOS
# ==============================================================================

def compute_twr_analytics(strategy_twr: np.ndarray, btc_daily_ret: np.ndarray) -> Dict[str, float]:
    if len(strategy_twr) <= 2 or len(btc_daily_ret) <= 2:
        return {"twr_corr": 0.0, "twr_beta": 0.0, "up_capture": np.nan, "down_capture": np.nan}

    strat_daily_ret = np.diff(strategy_twr) / np.maximum(1e-6, strategy_twr[:-1])
    min_len = min(len(strat_daily_ret), len(btc_daily_ret))
    r_s = strat_daily_ret[:min_len]
    r_m = btc_daily_ret[:min_len]

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
        liq_ok = grid.dvol30_mat >= LIQUIDITY_FLOOR_USD

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
# AUDIT & OPTIMIZATION PIPELINE
# ==============================================================================

def run_optimization():
    logging.info(f"Initializing Crypto Framework {ENGINE_VERSION}: Universe = {UNIVERSE_NAME}, Top {TOP_N_COINS} Coins, Start Year = {START_YEAR}")
    logging.info(f"Param Signature: {PARAM_SPACE_HASH} | Study DB: {STUDY_DB}")

    grid, universe, macro_df, provenance_records = build_market_universe(UNIVERSE_NAME, START_YEAR, FORCE_REFRESH)
    n_bars = len(grid.dates)

    # Prefix-Invariance Audit
    verify_causality_matrix(grid)

    mature_coverage = np.sum(grid.alive_mat, axis=0)
    mature_pct = mature_coverage / float(len(grid.symbols))
    valid_start_indices = np.where(mature_pct >= 0.40)[0]
    trading_start_bar = max(int(valid_start_indices[0]) if len(valid_start_indices) > 0 else WARMUP_BARS, WARMUP_BARS)

    total_trading_bars = n_bars - trading_start_bar
    split_offset = int(total_trading_bars * 0.75)
    is_start, is_end = trading_start_bar, trading_start_bar + split_offset
    oos_start, oos_end = is_end, n_bars
    is_bars, oos_bars = is_end - is_start, oos_end - oos_start

    logging.info(
        f"24/7/365 Continuous Calendar: Total Bars = {n_bars} | "
        f"Trading Starts = {grid.dates[trading_start_bar].date()} | "
        f"IS Window = {is_bars} days [{grid.dates[is_start].date()} -> {grid.dates[is_end-1].date()}] | "
        f"OOS Window = {oos_bars} days [{grid.dates[oos_start].date()} -> {grid.dates[oos_end-1].date()}]"
    )

    rough_macro_ma = FastIndicators.moving_average(grid.macro_close, 200, 0)
    rough_macro_ok = (~np.isnan(rough_macro_ma)) & (grid.macro_close > rough_macro_ma)

    is_bench = calculate_benchmarks(grid, is_start, is_end, rough_macro_ok)
    oos_bench = calculate_benchmarks(grid, oos_start, oos_end, rough_macro_ok)

    is_crisis_slices = []
    for s_name, (s_start_str, s_end_str) in STRESS_WINDOWS.items():
        s_s_ts, s_e_ts = pd.Timestamp(s_start_str), pd.Timestamp(s_end_str)
        s_idx = np.where((grid.dates >= s_s_ts) & (grid.dates <= s_e_ts))[0]
        if len(s_idx) > 15 and s_idx[0] >= is_start and s_idx[-1] < is_end:
            c_s, c_e = int(s_idx[0]), int(s_idx[-1]) + 1
            bench_slice = calculate_benchmarks(grid, c_s, c_e)
            is_crisis_slices.append({
                "name": s_name, "start_bar": c_s, "end_bar": c_e,
                "macro_max_dd": bench_slice["macro_max_dd"]
            })

    baseline_score = -10.0
    if FINAL_WINNER_FILE.exists():
        try:
            with open(FINAL_WINNER_FILE, "r") as f:
                saved = json.load(f)
                if saved.get("version") == ENGINE_VERSION and saved.get("param_hash") == PARAM_SPACE_HASH:
                    baseline_score = saved.get("is_score", -10.0)
                    logging.info(f"Loaded All-Time Champion Hurdle Score from winner.json: {baseline_score:.4f}")
        except Exception:
            pass

    def objective(trial: optuna.Trial) -> float:
        params = sample_hyperparameters(trial)
        raw_sig, entry_mat, exit_mat, macro_ok, state_mat = compile_signals_fast(grid, params)
        engine = BacktestEngine(params=params)
        res = engine.run_interval(grid, raw_sig, entry_mat, exit_mat, macro_ok, is_start, is_end, state_mat=state_mat, window_label="IN_SAMPLE")
        metrics = calculate_metrics(res, grid, is_start, is_end, is_bench["macro_twr_cagr"])
        raw_score = metrics["score"]

        try:
            current_study_best = trial.study.best_value
        except ValueError:
            current_study_best = -10.0
        target_hurdle = max(current_study_best, baseline_score)

        if raw_score > target_hurdle and raw_score > 0.05:
            is_stable, plateau_score, pass_rate, _ = evaluate_neighborhood_stability(
                params=params, candidate_score=raw_score, grid=grid,
                start_bar=is_start, end_bar=is_end, benchmark_cagr=is_bench["macro_twr_cagr"]
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

                dd_diff = c_slice["macro_max_dd"] - c_strat_dd
                if dd_diff < 0.0:
                    crisis_penalties.append(float(np.exp(dd_diff / 15.0)))
                else:
                    crisis_penalties.append(1.0)

            crisis_mult = float(np.min(crisis_penalties)) if len(crisis_penalties) > 0 else 1.0

            # Symbol Jackknifing: 3x 70% symbol sub-universes (V7.2 Port)
            n_syms = len(grid.symbols)
            min_active = max(5, int(n_syms * 0.40))
            jk_scores = []
            rng = np.random.default_rng(trial.number)

            for _ in range(3):
                mask = rng.random(n_syms) < 0.70
                if np.sum(mask) < min_active:
                    mask[:min_active] = True

                res_jk = engine.run_interval(
                    grid, raw_sig, entry_mat, exit_mat, macro_ok,
                    is_start, is_end, active_symbols_mask=mask, state_mat=state_mat
                )
                m_jk = calculate_metrics(res_jk, grid, is_start, is_end, is_bench["macro_twr_cagr"])
                jk_scores.append(m_jk["score"])

            p10_jk_score = float(np.percentile(jk_scores, 10))
            if p10_jk_score <= 0.0:
                logging.info(f"Trial {trial.number} REJECTED [Concentrated Coin Alpha]: p10 jackknife <= 0")
                return 0.01

            jk_ratio = min(1.0, max(0.10, p10_jk_score / max(1e-6, raw_score)))

            final_score = float(plateau_score * crisis_mult * jk_ratio)
            return final_score

        return raw_score

    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=42, n_startup_trials=50)
    study = optuna.create_study(study_name=STUDY_NAME, storage=STUDY_DB, load_if_exists=True, direction="maximize", sampler=sampler)

    n_remaining = max(0, N_TRIALS - len(study.trials))
    if n_remaining > 0:
        logging.info(f"Executing {n_remaining} trials in study '{STUDY_NAME}'...")
        study.optimize(objective, n_trials=n_remaining, show_progress_bar=True)

    # 3-Fold Walk-Forward Champion Gate
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
            "trial": fallback_trial, "beats_all": False, "beats_recent": False,
            "recent_margin": 0.0, "score": fallback_trial.value or 0.0,
            "c1": 0.0, "c2": 0.0, "c3": 0.0, "i1": 0.0, "i2": 0.0, "i3": 0.0,
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

            m1 = calculate_metrics(r1, grid, f1_s, f1_e, b1["macro_twr_cagr"])
            m2 = calculate_metrics(r2, grid, f2_s, f2_e, b2["macro_twr_cagr"])
            m3 = calculate_metrics(r3, grid, f3_s, f3_e, b3["macro_twr_cagr"])

            c1, c2, c3 = m1["full"]["cagr"], m2["full"]["cagr"], m3["full"]["cagr"]
            i1, i2, i3 = b1["macro_twr_cagr"], b2["macro_twr_cagr"], b3["macro_twr_cagr"]

            scored_candidates.append({
                "trial": t_cand,
                "beats_all": bool(c1 > i1 and c2 > i2 and c3 > i3),
                "beats_recent": bool(c3 > i3),
                "recent_margin": float(c3 - i3),
                "score": t_cand.value,
                "c1": c1, "c2": c2, "c3": c3,
                "i1": i1, "i2": i2, "i3": i3,
            })

    scored_candidates.sort(
        key=lambda x: (x["beats_all"], x["beats_recent"], x["recent_margin"], x["score"]),
        reverse=True
    )
    best_candidate_record = scored_candidates[0]
    best_trial = best_candidate_record["trial"]
    best_params = reconstitute_params(best_trial.params)

    raw_sig_final, entry_final, exit_final, macro_final, state_final = compile_signals_fast(grid, best_params)
    final_engine = BacktestEngine(params=best_params)

    is_bench = calculate_benchmarks(grid, is_start, is_end, macro_final)
    oos_bench = calculate_benchmarks(grid, oos_start, oos_end, macro_final)

    is_results = final_engine.run_interval(grid, raw_sig_final, entry_final, exit_final, macro_final, is_start, is_end, state_mat=state_final, window_label="IN_SAMPLE")
    is_metrics = calculate_metrics(is_results, grid, is_start, is_end, is_bench["macro_twr_cagr"])

    oos_results = final_engine.run_interval(grid, raw_sig_final, entry_final, exit_final, macro_final, oos_start, oos_end, state_mat=state_final, window_label="OUT_OF_SAMPLE")
    oos_metrics = calculate_metrics(oos_results, grid, oos_start, oos_end, oos_bench["macro_twr_cagr"])

    continuous_results = final_engine.run_interval(grid, raw_sig_final, entry_final, exit_final, macro_final, is_start, n_bars, state_mat=state_final, window_label="CONTINUOUS")
    cont_trades = continuous_results["trades"]
    cont_twr = continuous_results["twr_curve"]

    stress_rows_md = ""
    for s_name, (s_start_str, s_end_str) in STRESS_WINDOWS.items():
        s_start_ts, s_end_ts = pd.Timestamp(s_start_str), pd.Timestamp(s_end_str)
        s_indices = np.where((grid.dates >= s_start_ts) & (grid.dates <= s_end_ts))[0]
        if len(s_indices) > 15:
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

                stress_rows_md += f"| **{s_name}** | {s_ret:+.2f}% | {s_dd:.2f}% | {s_b['macro_total_return']:+.2f}% | {s_b['macro_max_dd']:.2f}% | {s_b['timed_macro_total_return']:+.2f}% | {n_active_trades} |\n"

    # Concatenate all closed trades safely with window identifiers
    all_trades = pd.concat([is_results["trades"], oos_results["trades"]], ignore_index=True)

    # Compute TWR Beta & Capture Analytics
    is_twr_analytics = compute_twr_analytics(is_results["twr_curve"], is_bench["macro_daily_ret"])
    oos_twr_analytics = compute_twr_analytics(oos_results["twr_curve"], oos_bench["macro_daily_ret"])

    # Run Matched Placebo Suite
    logging.info("Executing Matched Placebo Suite (Noise Controls)...")
    is_placebo_suite = run_matched_placebo_suite(
        grid, best_params, is_bars, is_start, is_end,
        is_bench["macro_twr_cagr"], is_results["funnel_stats"]["raw_triggers"], n_seeds=50
    )
    oos_placebo_suite = run_matched_placebo_suite(
        grid, best_params, oos_bars, oos_start, oos_end,
        oos_bench["macro_twr_cagr"], oos_results["funnel_stats"]["raw_triggers"], n_seeds=50
    )

    trailing_ablation_md = run_trailing_stop_ablation(
        grid, best_params, is_start, is_end, oos_start, oos_end,
        is_bench["macro_twr_cagr"], oos_bench["macro_twr_cagr"]
    )

    winner_data = {
        "version": ENGINE_VERSION,
        "param_hash": PARAM_SPACE_HASH,
        "study_name": STUDY_NAME,
        "universe": UNIVERSE_NAME,
        "top_n_coins": TOP_N_COINS,
        "is_score": float(is_metrics["score"]),
        "is_metrics": is_metrics,
        "oos_metrics": oos_metrics,
        "is_twr_analytics": is_twr_analytics,
        "oos_twr_analytics": oos_twr_analytics,
        "is_placebo_suite": {
            "conditioned_median": is_placebo_suite["conditioned"][0],
            "conditioned_p5": is_placebo_suite["conditioned"][1],
            "conditioned_p95": is_placebo_suite["conditioned"][2],
        },
        "oos_placebo_suite": {
            "conditioned_median": oos_placebo_suite["conditioned"][0],
            "conditioned_p5": oos_placebo_suite["conditioned"][1],
            "conditioned_p95": oos_placebo_suite["conditioned"][2],
        },
        "best_params": best_params,
        "updated_at": pd.Timestamp.now().isoformat(),
    }

    with open(CURRENT_WINNER_FILE, "w", encoding="utf-8") as f:
        json.dump(winner_data, f, indent=4, default=str)

    if is_metrics["score"] > baseline_score:
        logging.info(f"New Champion Discovered: IS Score {is_metrics['score']:.4f} > All-Time Baseline {baseline_score:.4f}")
        with open(FINAL_WINNER_FILE, "w", encoding="utf-8") as f:
            json.dump(winner_data, f, indent=4, default=str)

    # Flatten friction dataclasses cleanly into primitives before CSV export
    export_df = all_trades.copy()
    if len(export_df) > 0 and "frictions" in export_df.columns:
        export_df["fee_exchange"] = export_df["frictions"].apply(lambda f: f.exchange_fee if hasattr(f, "exchange_fee") else 0.0)
        export_df["fee_slippage"] = export_df["frictions"].apply(lambda f: f.slippage_cost if hasattr(f, "slippage_cost") else 0.0)
        export_df = export_df.drop(columns=["frictions"])
    export_df.to_csv(TRADES_CSV_FILE, index=False)

    f_disp = is_results["funnel_stats"]
    funnel_table_md = f"""| Stage in Funnel | Unique Signals | Disposition Description |
| :--- | :--- | :--- |
| **Total Raw Technical Triggers** | **{f_disp['raw_triggers']}** | Close > HHV, RSI cross, or MA trigger |
| ├── Macro Gate Blocked | -{f_disp['macro_blocked']} | {MACRO_INDEX_NAME} below Macro Moving Average |
| ├── Liquidity Floor Blocked | -{f_disp['liquidity_blocked']} | ADV < {format_price(LIQUIDITY_FLOOR_USD)} |
| ├── ADX Trend Blocked | -{f_disp['adx_blocked']} | ADX < Selected Threshold |
| ├── Pyramid Limit Blocked | -{f_disp['pyramid_blocked']} | Asset already at max layers or averaging down |
| ├── Revalidation Dropped | -{f_disp['revalidation_dropped']} | Failed macro/ADX/state check at fill time |
| ├── Slot Saturated Dropped | -{f_disp['slot_saturated_dropped']} | No open portfolio slots available |
| ├── Cash Starved Dropped | -{f_disp['cash_starved_dropped']} | Cash below {format_price(TRANCHE_FLOOR_USD)} |
| ├── Scrip Risk Cap Dropped | -{f_disp['risk_cap_dropped']} | Exceeded single-asset 25% exposure ceiling |
| ├── Expired in Watchlist | -{f_disp['expired_unfilled']} | Exceeded {WL_MAX_AGE_BARS} bars or shadow stop |
| **Executed Trades on Ledger** | **{f_disp['executed_fills']}** | Successfully filled and audited |
"""

    report_md = f"""# Crypto Quantitative Strategy Audit Dossier ({ENGINE_VERSION}) — {UNIVERSE_NAME} (Top {TOP_N_COINS})

## 1. Segregated Performance Accounting (Pure 365-Day Continuous TWR)

### A. Performance Baseline & Generalization Check
| Performance Metric | In-Sample ({is_bars} days) | Out-of-Sample ({oos_bars} days, Fresh Seed) | Generalization Assessment |
| :--- | :--- | :--- | :--- |
| **Strategy Annualized TWR (CAGR)** | **{is_metrics['full']['cagr']:.2f}%** | **{oos_metrics['full']['cagr']:.2f}%** | 365 Continuous Days/Year |
| **Strategy Organic CAGR (Anti-Cheat)** | {is_metrics['cagr_organic']:.2f}% | {oos_metrics['cagr_organic']:.2f}% | Deflated by terminal unclosed equity |
| **Benchmark ({MACRO_INDEX_NAME}) CAGR** | {is_bench['macro_twr_cagr']:.2f}% | {oos_bench['macro_twr_cagr']:.2f}% | Buy-and-Hold {MACRO_INDEX_NAME} |
| **Equal-Weight Filtered Basket CAGR** | {is_bench['basket_twr_cagr']:.2f}% | {oos_bench['basket_twr_cagr']:.2f}% | Unweighted clean crypto set (clipped) |
| **Strategy TWR Max Drawdown** | **{is_metrics['max_dd']:.2f}%** | **{oos_metrics['max_dd']:.2f}%** | Pure capital decline |
| **Benchmark ({MACRO_INDEX_NAME}) Max DD** | {is_bench['macro_max_dd']:.2f}% | {oos_bench['macro_max_dd']:.2f}% | Market systemic peak-to-trough |
| **Strategy Profit Factor** | {is_metrics['full']['profit_factor']:.2f} | {oos_metrics['full']['profit_factor']:.2f} | Net Wins / Net Losses |
| **Completed Trades** | {is_metrics['full']['trades']} | {oos_metrics['full']['trades']} | Executed volume |
| **Net Realized PnL** | **{format_price(is_metrics['full']['net_pnl'])}** | **{format_price(oos_metrics['full']['net_pnl'])}** | Frictions deducted |
| **TWR Beta vs {MACRO_INDEX_NAME}** | {is_twr_analytics['twr_beta']:.2f} | {oos_twr_analytics['twr_beta']:.2f} | Systematic market exposure |
| **TWR Correlation vs {MACRO_INDEX_NAME}** | {is_twr_analytics['twr_corr']:.2f} | {oos_twr_analytics['twr_corr']:.2f} | Co-movement correlation |
| **Up-Market Capture** | {is_twr_analytics['up_capture']:.1f}% | {oos_twr_analytics['up_capture']:.1f}% | Return captured when BTC positive |
| **Down-Market Capture** | {is_twr_analytics['down_capture']:.1f}% | {oos_twr_analytics['down_capture']:.1f}% | Return captured when BTC negative |

### B. Matched Placebo Noise Suite (50 Seeds Control)
| Evaluation Window | Strategy CAGR | Conditioned Noise Median | Noise [P5, P95] Range | Statistical Edge Status |
| :--- | :--- | :--- | :--- | :--- |
| **In-Sample** | **{is_metrics['full']['cagr']:.2f}%** | {is_placebo_suite['conditioned'][0]:.2f}% | [{is_placebo_suite['conditioned'][1]:.2f}%, {is_placebo_suite['conditioned'][2]:.2f}%] | {'CONFIRMED (> P95)' if is_metrics['full']['cagr'] > is_placebo_suite['conditioned'][2] else 'INSIDE NOISE BAND'} |
| **Out-of-Sample** | **{oos_metrics['full']['cagr']:.2f}%** | {oos_placebo_suite['conditioned'][0]:.2f}% | [{oos_placebo_suite['conditioned'][1]:.2f}%, {oos_placebo_suite['conditioned'][2]:.2f}%] | {'CONFIRMED (> P95)' if oos_metrics['full']['cagr'] > oos_placebo_suite['conditioned'][2] else 'INSIDE NOISE BAND'} |

## 2. Crypto Black Swan Stress Audit
| Historical Crash Period | Strategy Return | Strategy Max DD | {MACRO_INDEX_NAME} Return | {MACRO_INDEX_NAME} Max DD | Timed Macro Return | Active Trades |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{stress_rows_md}
## 3. Trailing Stop Mechanism Ablation
{trailing_ablation_md}
## 4. Signal Funnel & Opportunity Attrition Matrix
{funnel_table_md}
## 5. Discovered Optimal Parameter Set
```json
{json.dumps(best_params, indent=4)}
"""
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_md)
    logging.info(f"{ENGINE_VERSION} audit dossier written to {REPORT_FILE}")


if __name__ == "__main__":
    run_optimization()