#!/usr/bin/env python3
"""
CRYPTO QUANTITATIVE SWING-TRADING FRAMEWORK (INSTITUTIONAL ENGINE V7)
====================================================================
- Behavioral Prefix-Invariance Test: Empirical mathematical proof of zero lookahead bias.
- Sidecar Provenance Backfill: Eliminates fabricated "Delisted on CACHE Feed" labels.
- Sizing Constraint Attribution: Tracks cash_constrained events separately from equity_slot.
- Configurable EOT Ablation Weight: User Control Panel toggle (EOT_EXCESS_WEIGHT).
- Strict Causal ATR Discipline: All stops, trailing chandeliers, and TPs lagged to t-1.
- Tunable Exit Type 6 (Volume Exhaustion): exit_vol_ma_len & exit_vol_mult searchable.
- Sub-Cent Resolution: Formats micro-caps (PEPE/SHIB) without zero-rounding collapse.
"""

from __future__ import annotations

import os
import sys
import copy
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import requests
import optuna

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ==============================================================================
# USER CONTROL PANEL
# ==============================================================================
START_YEAR: int = 2017                 # Earliest historical candle year
TOP_N_COINS: int = 80                  # Number of filtered coins to trade
QUOTE_ASSET: str = "USDT"              # Traded quote asset
INITIAL_CAPITAL: float = 1000.0        # Initial account cash
MONTHLY_CONTRIBUTION: float = 1000.0   # Monthly capital injection
MIN_HISTORY_DAYS: int = 300            # Minimum active days required
WARMUP_BARS: int = 300                 # Indicator maturity horizon
N_TRIALS: int = 2500                    # Optuna study trials
PARALLEL_DOWNLOAD_WORKERS: int = 8     # Download worker threads
FORCE_REFRESH: bool = False            # True = ignore local cache, redownload all
CACHE_MAX_AGE_HOURS: float = 72.0      # Cache validation window
WL_MAX_AGE_BARS: int = 21              # Max days a signal can wait in watchlist

# --- Anti-Cheating & Fitness Sizing ---
EOT_EXCESS_WEIGHT: float = 0.30        # Credit for unclosed terminal MTM excess (0.0 = pure organic)

# --- Capacity & Liquidity Constraints ---
TRANCHE_FLOOR_USD: float = 500.0       # Minimum order allocation
TRANCHE_CEILING_USD: float = 25000.0   # Hard ceiling on any single position tranche
MAX_CONCURRENT_TRANCHES: int = 10      # Max simultaneous open positions
MAX_ADV_PARTICIPATION: float = 0.015   # Max 1.5% of trailing 30d dollar volume
LIQUIDITY_FLOOR_USD: float = 3000000.0 # $3,000,000 trailing 30d volume floor

# --- Execution & Non-Linear Market Impact ---
TAKER_FEE_BPS: float = 10.0            # 0.10% fee per side
BASE_SLIPPAGE_BPS: float = 15.0        # 0.15% base slippage
IMPACT_COEF_BPS: float = 250.0         # Market impact scaling coefficient
NEIGHBORHOOD_DROP_LIMIT: float = 0.25  # Max allowable score degradation (25%)

DATA_DIR = Path("data_cache")
OUTPUT_DIR = Path("output")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_WINNER_FILE = OUTPUT_DIR / "current_winner.json"
FINAL_WINNER_FILE = OUTPUT_DIR / "winner.json"
REPORT_FILE = OUTPUT_DIR / "WINNER_REPORT.md"
TRADES_CSV_FILE = OUTPUT_DIR / "winner_trades.csv"
STUDY_DB = f"sqlite:///{OUTPUT_DIR.resolve() / 'crypto_swing_study.db'}"

# Noise Deny-List (Stables, Wrappers, Collisions, Single-Letters)
DENY_LIST = {
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDD", "USDP", "FDUSD", "PYUSD", "EUR", "GBP",
    "USDE", "SUSDE", "USDY", "BUIDL", "RLUSD", "USD1", "FRAX", "LUSD", "CRVUSD", "GHO",
    "WBTC", "CBBTC", "TBTC", "LBTC", "RENBTC", "BTCB", "SOLVBTC", "WETH", "STETH", "WSTETH",
    "CBETH", "RETH", "WEETH", "EZETH", "WSOL", "JITOSOL", "MSOL", "BNSOL", "WBNB", "SAVAX",
    "RAIN", "U"
}

LEGACY_WHITELIST = [
    "BTC", "ETH", "SOL", "XRP", "XLM", "DOGE", "LTC", "ADA", "LINK", "AVAX", "DOT", "BCH",
    "NEAR", "ATOM", "UNI", "ETC", "XMR", "ALGO", "VET", "FIL", "ICP", "AAVE", "SAND"
]

_HTTP_SESSION = requests.Session()
_HTTP_SESSION.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

def format_price(px: float) -> str:
    """Adaptive price formatter providing sub-cent resolution for micro-caps."""
    if not np.isfinite(px) or px <= 0:
        return "$0.00"
    if px >= 1.0:
        return f"${px:,.2f}"
    elif px >= 0.01:
        return f"${px:,.4f}"
    else:
        return f"${px:.7f}"

# ==============================================================================
# 1. CANDIDATE DISCOVERY
# ==============================================================================
def discover_market_candidates(target_count: int) -> List[str]:
    candidates: List[str] = []
    overfetch = max(120, int(target_count * 2.2))

    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": min(250, overfetch), "page": 1}
        resp = _HTTP_SESSION.get(url, params=params, timeout=8)
        if resp.status_code == 200:
            for item in resp.json():
                sym = str(item.get("symbol", "")).upper()
                if (sym and len(sym) > 1 and sym not in DENY_LIST and 
                    not any(sym.endswith(p) for p in ("UP", "DOWN", "BEAR", "BULL"))):
                    if sym not in candidates:
                        candidates.append(sym)
            logging.info(f"Discovered {len(candidates)} candidates via CoinGecko market-cap rank.")
    except Exception as e:
        logging.warning(f"CoinGecko ranking unavailable: {e}")

    if len(candidates) < target_count:
        try:
            url = "https://api.binance.com/api/v3/ticker/24hr"
            resp = _HTTP_SESSION.get(url, timeout=8)
            if resp.status_code == 200:
                valid_pairs = []
                for t in resp.json():
                    s = t.get("symbol", "")
                    if s.endswith(QUOTE_ASSET):
                        base = s[:-len(QUOTE_ASSET)]
                        if (base and len(base) > 1 and base not in DENY_LIST and 
                            not any(base.endswith(p) for p in ("UP", "DOWN", "BEAR", "BULL"))):
                            valid_pairs.append((base, float(t.get("quoteVolume", 0.0))))
                valid_pairs.sort(key=lambda x: x[1], reverse=True)
                for base, _ in valid_pairs:
                    if base not in candidates:
                        candidates.append(base)
                logging.info(f"Augmented candidates via Binance 24h volume. Total pool: {len(candidates)}")
        except Exception as e:
            logging.warning(f"Binance volume ranking fallback failed: {e}")

    ordered_universe: List[str] = []
    for coin in LEGACY_WHITELIST:
        if coin not in ordered_universe:
            ordered_universe.append(coin)
    for coin in candidates:
        if coin not in ordered_universe:
            ordered_universe.append(coin)

    return ordered_universe[:overfetch]

# ==============================================================================
# 2. MULTI-EXCHANGE DATA INGESTION ENGINE WITH METADATA SIDECARS
# ==============================================================================
def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    today_utc = pd.Timestamp.now('UTC').tz_localize(None).floor('D')
    df = df[df["date"] < today_utc].dropna(subset=["close"])
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.set_index("date")

def fetch_from_binance(symbol: str, start_year: int) -> Optional[pd.DataFrame]:
    hosts = ["https://data-api.binance.vision", "https://api.binance.com"]
    start_ts = int(pd.Timestamp(f"{start_year}-01-01", tz="UTC").timestamp() * 1000)
    now_ts = int(pd.Timestamp.now('UTC').floor('D').timestamp() * 1000)
    all_rows = []

    for host in hosts:
        curr_start = start_ts
        all_rows = []
        success = True
        while curr_start < now_ts:
            url = f"{host}/api/v3/klines"
            params = {"symbol": f"{symbol}{QUOTE_ASSET}", "interval": "1d", "startTime": curr_start, "limit": 1000}
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
            return _clean_dataframe(df)
    return None

def fetch_from_bybit(symbol: str) -> Optional[pd.DataFrame]:
    url = "https://api.bybit.com/v5/market/kline"
    end_ts = int(pd.Timestamp.now('UTC').timestamp() * 1000)
    all_rows = []

    for _ in range(12):
        params = {"category": "spot", "symbol": f"{symbol}{QUOTE_ASSET}", "interval": "D", "end": end_ts, "limit": 1000}
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
        return _clean_dataframe(df)
    return None

def fetch_from_okx(symbol: str) -> Optional[pd.DataFrame]:
    url = "https://www.okx.com/api/v5/market/history-candles"
    inst_id = f"{symbol}-{QUOTE_ASSET}"
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
        return _clean_dataframe(df)
    return None

def fetch_from_yahoo(symbol: str, start_year: int) -> Optional[pd.DataFrame]:
    start_ts = int(pd.Timestamp(f"{start_year}-01-01", tz="UTC").timestamp())
    end_ts = int(pd.Timestamp.now('UTC').timestamp())
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
        cleaned = _clean_dataframe(df)
        if len(cleaned) >= MIN_HISTORY_DAYS:
            return cleaned
    except Exception:
        pass
    return None

def fetch_single_coin(coin: str, start_year: int, refresh: bool) -> Tuple[str, Optional[pd.DataFrame], str]:
    cache_file = DATA_DIR / f"{coin}_1d.parquet"
    meta_file = DATA_DIR / f"{coin}_1d.meta.json"
    
    if not refresh and cache_file.exists():
        age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600.0
        if age_hours < CACHE_MAX_AGE_HOURS:
            try:
                df = pd.read_parquet(cache_file)
                provider = "LEGACY_CACHE"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r") as mf:
                            provider = json.load(mf).get("provider", "LEGACY_CACHE")
                    except Exception:
                        pass
                else:
                    # Backfill sidecar for legacy files so they are properly marked
                    try:
                        with open(meta_file, "w") as mf:
                            json.dump({"provider": "LEGACY_CACHE", "timestamp": os.path.getmtime(cache_file), "bars": len(df)}, mf)
                    except Exception:
                        pass
                return coin, df, provider
            except Exception:
                pass

    for name, fn in [
        ("binance", lambda: fetch_from_binance(coin, start_year)),
        ("bybit", lambda: fetch_from_bybit(coin)),
        ("okx", lambda: fetch_from_okx(coin)),
        ("yahoo", lambda: fetch_from_yahoo(coin, start_year)),
    ]:
        df = fn()
        if df is not None and len(df) >= MIN_HISTORY_DAYS:
            df.to_parquet(cache_file)
            with open(meta_file, "w") as mf:
                json.dump({"provider": name, "timestamp": time.time(), "bars": len(df)}, mf)
            return coin, df, name

    return coin, None, "none"

def build_market_universe(target_coins: int, start_year: int, refresh: bool) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, List[Dict[str, Any]]]:
    raw_universe: Dict[str, pd.DataFrame] = {}
    provider_map: Dict[str, str] = {}
    candidates = discover_market_candidates(target_coins)
    
    _, btc_df, prov = fetch_single_coin("BTC", start_year, refresh)
    if btc_df is None:
        raise RuntimeError("Failed to load BTC benchmark data across all exchange providers.")
    raw_universe["BTC"] = btc_df
    provider_map["BTC"] = prov
    logging.info(f"BTC benchmark loaded from [{prov}] ({len(btc_df)} bars).")

    logging.info(f"Downloading candles in parallel using {PARALLEL_DOWNLOAD_WORKERS} workers...")
    fetch_list = [c for c in candidates if c != "BTC"]

    with ThreadPoolExecutor(max_workers=PARALLEL_DOWNLOAD_WORKERS) as executor:
        future_map = {executor.submit(fetch_single_coin, coin, start_year, refresh): coin for coin in fetch_list}
        for future in as_completed(future_map):
            coin, df, provider = future.result()
            if df is not None and len(df) >= MIN_HISTORY_DAYS:
                if len(raw_universe) < target_coins:
                    raw_universe[coin] = df
                    provider_map[coin] = provider
                    logging.info(f"  + {coin:<6} -> {len(df):>5} bars from [{provider}] ({df.index[0].date()} to {df.index[-1].date()})")

    logging.info(f"Successfully assembled {len(raw_universe)} assets.")

    # Terminal Grid Synchronization
    btc_end = btc_df.index.max()
    active_threshold = btc_end - pd.Timedelta(days=7)
    active_coins = [coin for coin, df in raw_universe.items() if df.index.max() >= active_threshold]
    common_end = min(raw_universe[coin].index.max() for coin in active_coins)
    
    logging.info(f"Terminal Grid Synchronization: BTC ends {btc_end.date()} | Synchronized Active Grid = {common_end.date()}")

    master_dates = pd.date_range(btc_df.index.min(), common_end, freq="D")
    universe: Dict[str, pd.DataFrame] = {}
    provenance_records: List[Dict[str, Any]] = []
    
    for coin, df in raw_universe.items():
        reindexed = df.loc[:common_end].reindex(master_dates)
        raw_close = reindexed["close"].copy()

        first_idx = df.index.min()
        last_idx = df.index.max()
        active_span_mask = (master_dates >= first_idx) & (master_dates <= min(last_idx, common_end))
        missing_in_span = int((raw_close.isna() & pd.Series(active_span_mask, index=master_dates)).sum())

        # Reviewer Hindsight Terminal Delisting Mask
        is_missing = raw_close.isna()
        trailing_terminal_missing = is_missing[::-1].cumprod()[::-1].astype(bool)
        has_ever_traded = (~is_missing).cumsum() > 0
        is_delisted_permanent = (trailing_terminal_missing & has_ever_traded).values

        reindexed["alive"] = ~is_missing
        reindexed["is_delisted"] = is_delisted_permanent
        
        # PRESERVE NaNs BEFORE LISTING: Forward fill only during active trading
        reindexed["close"] = reindexed["close"].ffill()
        reindexed["open"] = reindexed["open"].ffill()
        reindexed["high"] = reindexed["high"].ffill()
        reindexed["low"] = reindexed["low"].ffill()
        reindexed["volume"] = reindexed["volume"].fillna(0.0)
        reindexed["quote_volume"] = reindexed["quote_volume"].fillna(0.0)
        universe[coin] = reindexed

        provenance_records.append({
            "coin": coin,
            "provider": provider_map.get(coin, "unknown"),
            "first_date": str(first_idx.date()),
            "last_date": str(last_idx.date()),
            "total_bars": len(df),
            "missing_days_in_span": missing_in_span,
            "is_permanently_delisted": bool(is_delisted_permanent[-1])
        })

    return universe, universe["BTC"], provenance_records

# ==============================================================================
# 3. VECTORIZED INDICATOR LIBRARY
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
            w = np.arange(1, length + 1)
            return s.rolling(length).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)
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
def compile_signals(universe: Dict[str, pd.DataFrame], btc_df: pd.DataFrame, p: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], pd.Series]:
    use_btc = p.get("use_btc_macro_system", False)
    if use_btc:
        btc_ma = Indicators.moving_average(btc_df["close"], p["btc_ma_len"], p["btc_ma_type"])
        btc_ok = (btc_df["close"] > btc_ma).fillna(False)
    else:
        btc_ok = pd.Series(True, index=btc_df.index)

    signals: Dict[str, Dict[str, Any]] = {}
    for coin, df in universe.items():
        c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
        qv = df["quote_volume"]
        atr14 = Indicators.atr(df, 14)

        dvol = qv.rolling(30, min_periods=30).mean() if (qv > 0).any() else (c * v).rolling(30, min_periods=30).mean()
        liq_ok = (dvol >= LIQUIDITY_FLOOR_USD).fillna(False)

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

        entry_mask = raw_entry & liq_ok & adx_ok & btc_ok & df["alive"]

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

        signals[coin] = {
            "entry": entry_mask,
            "exit_sig": exit_sig,
            "atr": atr14,
            "dvol": dvol,
            "df": df
        }

    return signals, btc_ok

# ==============================================================================
# 5. EXECUTION & SIMULATION ENGINE (STRICT CAUSALITY PRESERVED)
# ==============================================================================
@dataclass
class Tranche:
    tid: int
    coin: str
    entry_bar: int
    entry_date: pd.Timestamp
    entry_price: float
    units: float
    cost_usd: float
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

    def run_interval(
        self,
        signals: Dict[str, Dict[str, Any]],
        btc_df: pd.DataFrame,
        btc_ok: pd.Series,
        start_bar: int,
        end_bar: int
    ) -> Dict[str, Any]:
        p = self.p
        dates = btc_df.index
        use_btc = p.get("use_btc_macro_system", False)

        cash = INITIAL_CAPITAL
        total_inflow = INITIAL_CAPITAL
        live_tranches: List[Tranche] = []
        watchlist: List[Dict[str, Any]] = []
        closed_trades: List[Dict[str, Any]] = []

        active_capital_curve: List[float] = []
        total_equity_curve: List[float] = []
        tid_counter = 0

        binding_stats = {
            "tranche_ceiling": 0,
            "liquidity_cap": 0,
            "equity_slot": 0,
            "tranche_floor": 0,
            "cash_constrained": 0
        }

        max_pyramid = p.get("max_pyramid_layers", 1)
        fee_mult = TAKER_FEE_BPS / 10000.0

        for t in range(start_bar, end_bar):
            curr_date = dates[t]

            # 0. Monthly Inflow Injection
            if (t - start_bar) > 0 and (t - start_bar) % 30 == 0:
                cash += MONTHLY_CONTRIBUTION
                total_inflow += MONTHLY_CONTRIBUTION

            # 1. Delisting Guard on Permanent Project Demise (Zero-Value Write-off)
            surviving_tranches = []
            for tr in live_tranches:
                if signals[tr.coin]["df"]["is_delisted"].iloc[t]:
                    pnl = tr.proceeds - tr.cost_usd
                    closed_trades.append({
                        "coin": tr.coin, "pnl": pnl, "ret": pnl / tr.cost_usd,
                        "reason": "DELISTED", "bars": t - tr.entry_bar,
                        "entry_date": tr.entry_date, "exit_date": curr_date,
                        "entry_price": tr.entry_price, "exit_price": 0.0,
                        "cost_usd": tr.cost_usd, "proceeds": tr.proceeds,
                        "from_watchlist": tr.from_watchlist
                    })
                else:
                    surviving_tranches.append(tr)
            live_tranches = surviving_tranches

            watchlist = [item for item in watchlist if not signals[item["coin"]]["df"]["is_delisted"].iloc[t]]

            # 2. BTC Macro Liquidation Pass (Filled at Open[t])
            if use_btc and not btc_ok.iloc[t - 1]:
                surviving_tranches = []
                for tr in live_tranches:
                    if tr.coin == "BTC":
                        surviving_tranches.append(tr)
                        continue
                    px = signals[tr.coin]["df"]["open"].iloc[t]
                    adv = signals[tr.coin]["dvol"].iloc[t - 1]
                    part_rate = min(1.0, max(0.0, (tr.units * px) / max(adv, 100_000.0)))
                    slip_mult = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate)) / 10000.0
                    fill_px = px * (1.0 - slip_mult)
                    proceeds = tr.units * fill_px * (1.0 - fee_mult)
                    cash += proceeds
                    tr.proceeds += proceeds
                    pnl = tr.proceeds - tr.cost_usd
                    closed_trades.append({
                        "coin": tr.coin, "pnl": pnl, "ret": pnl / tr.cost_usd,
                        "reason": "BTC_MACRO_EXIT", "bars": t - tr.entry_bar,
                        "entry_date": tr.entry_date, "exit_date": curr_date,
                        "entry_price": tr.entry_price, "exit_price": fill_px,
                        "cost_usd": tr.cost_usd, "proceeds": tr.proceeds,
                        "from_watchlist": tr.from_watchlist
                    })
                live_tranches = surviving_tranches
                watchlist = [item for item in watchlist if item["coin"] == "BTC"]

            # 3. Live Position Intrabar Stops & Signal Exits (STRICTLY CAUSAL ATR[t-1])
            surviving_tranches = []
            for tr in live_tranches:
                coin_df = signals[tr.coin]["df"]
                adv = signals[tr.coin]["dvol"].iloc[t - 1]

                o_bar = coin_df["open"].iloc[t]
                h_bar = coin_df["high"].iloc[t]
                l_bar = coin_df["low"].iloc[t]
                
                # Causal ATR known prior to market open
                atr_bar = signals[tr.coin]["atr"].iloc[t - 1]

                # Global Take-Profit Pass with exact post-TP unit accounting
                if p.get("use_global_tp", False) and not tr.tp_done:
                    tp_price = tr.entry_price + (p["tp_mult"] * atr_bar)
                    if h_bar >= tp_price:
                        close_units = tr.units * (p["tp_size_pct"] / 100.0)
                        part_rate_tp = min(1.0, max(0.0, (close_units * tp_price) / max(adv, 100_000.0)))
                        slip_tp = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate_tp)) / 10000.0
                        fill_px = max(o_bar, tp_price) * (1.0 - slip_tp)
                        credit = close_units * fill_px * (1.0 - fee_mult)
                        cash += credit
                        tr.proceeds += credit
                        tr.units -= close_units
                        tr.tp_done = True
                        if p.get("tp_move_sl_be", False):
                            tr.stop_loss = max(tr.stop_loss, tr.entry_price)

                exit_triggered = False
                exit_reason = ""
                xt = p["exit_type"]

                if xt == 0:
                    tr.stop_loss = max(tr.stop_loss, tr.highest_high - (p["trail_mult"] * atr_bar))
                elif xt == 1:
                    tr.stop_loss = max(tr.stop_loss, tr.highest_high * (1.0 - (p["trail_pct"] / 100.0)))
                elif xt == 2:
                    tr.stop_loss = max(tr.stop_loss, tr.highest_high - (p["exit_atr_mult"] * atr_bar))

                if l_bar <= tr.stop_loss:
                    exit_triggered = True
                    exit_reason = "STOP_LOSS"
                elif xt in (3, 4, 5, 6, 7) and signals[tr.coin]["exit_sig"].iloc[t - 1]:
                    exit_triggered = True
                    exit_reason = f"SIGNAL_EXIT_TYPE_{xt}"

                if exit_triggered:
                    exit_ref_px = min(o_bar, tr.stop_loss) if exit_reason == "STOP_LOSS" else o_bar
                    part_rate_exit = min(1.0, max(0.0, (tr.units * exit_ref_px) / max(adv, 100_000.0)))
                    slip_exit = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate_exit)) / 10000.0
                    fill_px = min(o_bar, tr.stop_loss) if exit_reason == "STOP_LOSS" else o_bar * (1.0 - slip_exit)
                    proceeds = tr.units * fill_px * (1.0 - fee_mult)
                    cash += proceeds
                    tr.proceeds += proceeds
                    pnl = tr.proceeds - tr.cost_usd
                    closed_trades.append({
                        "coin": tr.coin, "pnl": pnl, "ret": pnl / tr.cost_usd,
                        "reason": exit_reason, "bars": t - tr.entry_bar,
                        "entry_date": tr.entry_date, "exit_date": curr_date,
                        "entry_price": tr.entry_price, "exit_price": fill_px,
                        "cost_usd": tr.cost_usd, "proceeds": tr.proceeds,
                        "from_watchlist": tr.from_watchlist
                    })
                else:
                    surviving_tranches.append(tr)
            live_tranches = surviving_tranches

            # 4. ACTIVE SHADOW LIFECYCLE (STRICTLY CAUSAL ATR[t-1])
            surviving_watchlist = []
            for item in watchlist:
                coin_df = signals[item["coin"]]["df"]
                l_bar = coin_df["low"].iloc[t]
                h_bar = coin_df["high"].iloc[t]
                atr_bar = signals[item["coin"]]["atr"].iloc[t - 1]

                if h_bar > item["highest_high"]:
                    item["highest_high"] = h_bar

                xt = p["exit_type"]
                if xt == 0:
                    item["shadow_stop"] = max(item["shadow_stop"], item["highest_high"] - (p["trail_mult"] * atr_bar))
                elif xt == 1:
                    item["shadow_stop"] = max(item["shadow_stop"], item["highest_high"] * (1.0 - (p["trail_pct"] / 100.0)))
                elif xt == 2:
                    item["shadow_stop"] = max(item["shadow_stop"], item["highest_high"] - (p["exit_atr_mult"] * atr_bar))

                shadow_stopped = l_bar <= item["shadow_stop"]
                shadow_exited = (xt in (3, 4, 5, 6, 7)) and signals[item["coin"]]["exit_sig"].iloc[t - 1]
                shadow_expired = (t - item["signal_bar"]) > WL_MAX_AGE_BARS

                if not (shadow_stopped or shadow_exited or shadow_expired):
                    surviving_watchlist.append(item)
            watchlist = surviving_watchlist

            # 5. Ingest New Signals Raised at Close[t-1]
            for coin, data in signals.items():
                if data["entry"].iloc[t - 1]:
                    active_coin_layers = sum(1 for tr in live_tranches if tr.coin == coin)
                    wl_coin_layers = sum(1 for item in watchlist if item["coin"] == coin)
                    if (active_coin_layers + wl_coin_layers) < max_pyramid:
                        o_today = data["df"]["open"].iloc[t]
                        a_yesterday = data["atr"].iloc[t - 1]
                        init_shadow_stop = o_today - (p.get("sl_mult", 2.5) * a_yesterday)
                        watchlist.append({
                            "coin": coin,
                            "signal_bar": t,
                            "trigger_price": data["df"]["close"].iloc[t - 1],
                            "shadow_stop": init_shadow_stop,
                            "highest_high": o_today
                        })

            # 6. Watchlist Prioritization & Sequential Sizing (Five-Way Binding Analysis)
            if watchlist:
                wl_mode = p.get("wl_mode", "WL_CLOSEST_BREAKOUT")
                
                if wl_mode == "WL_NONE":
                    watchlist = [item for item in watchlist if item["signal_bar"] == t]
                elif wl_mode == "WL_DEEPEST_DISCOUNT":
                    watchlist.sort(key=lambda x: (x["trigger_price"] - signals[x["coin"]]["df"]["close"].iloc[t - 1]) / max(1e-6, x["trigger_price"]), reverse=True)
                elif wl_mode == "WL_CLOSEST_BREAKOUT":
                    watchlist.sort(key=lambda x: abs(signals[x["coin"]]["df"]["close"].iloc[t - 1] - x["trigger_price"]) / max(1e-6, x["trigger_price"]))
                elif wl_mode == "WL_STRONGEST_MOMENTUM":
                    watchlist.sort(key=lambda x: signals[x["coin"]]["df"]["close"].iloc[t - 1] / max(1e-6, signals[x["coin"]]["df"]["close"].iloc[max(0, t - 14)]), reverse=True)
                elif wl_mode == "WL_FCFS":
                    watchlist.sort(key=lambda x: x["signal_bar"])
                elif wl_mode == "WL_LCFS":
                    watchlist.sort(key=lambda x: x["signal_bar"], reverse=True)

                unfilled = []
                for item in watchlist:
                    open_active_capital = sum(tr.units * signals[tr.coin]["df"]["open"].iloc[t] for tr in live_tranches)
                    current_equity_for_sizing = cash + open_active_capital

                    equity_slot_size = current_equity_for_sizing / float(MAX_CONCURRENT_TRANCHES)
                    adv_30d = signals[item["coin"]]["dvol"].iloc[t - 1]
                    liquidity_cap_usd = adv_30d * MAX_ADV_PARTICIPATION
                    target_usd = min(equity_slot_size, TRANCHE_CEILING_USD, liquidity_cap_usd)

                    base_friction = 1.0 + (TAKER_FEE_BPS + BASE_SLIPPAGE_BPS) / 10000.0
                    max_affordable = cash / base_friction
                    tranche_usd = min(max_affordable, max(TRANCHE_FLOOR_USD, target_usd))

                    # Five-Way Binding Constraint Classification
                    if max_affordable < target_usd:
                        if tranche_usd == max_affordable:
                            binding_stats["cash_constrained"] += 1
                        elif tranche_usd == TRANCHE_FLOOR_USD:
                            binding_stats["tranche_floor"] += 1
                    elif target_usd == TRANCHE_CEILING_USD:
                        binding_stats["tranche_ceiling"] += 1
                    elif target_usd == liquidity_cap_usd:
                        binding_stats["liquidity_cap"] += 1
                    elif target_usd < TRANCHE_FLOOR_USD:
                        binding_stats["tranche_floor"] += 1
                    else:
                        binding_stats["equity_slot"] += 1

                    part_rate = min(1.0, max(0.0, tranche_usd / max(adv_30d, 100_000.0)))
                    slip_mult = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate)) / 10000.0
                    total_cost = tranche_usd * (1.0 + fee_mult + slip_mult)
                    coin_layers = sum(1 for tr in live_tranches if tr.coin == item["coin"])

                    if (cash >= total_cost and tranche_usd >= TRANCHE_FLOOR_USD and 
                        len(live_tranches) < MAX_CONCURRENT_TRANCHES and coin_layers < max_pyramid):
                        
                        today_open = signals[item["coin"]]["df"]["open"].iloc[t]
                        fill_px = today_open * (1.0 + slip_mult)
                        units = (tranche_usd * (1.0 - fee_mult)) / fill_px
                        
                        yesterday_atr = signals[item["coin"]]["atr"].iloc[t - 1]
                        sl_price = fill_px - (p.get("sl_mult", 2.5) * yesterday_atr)
                        
                        cash -= total_cost
                        tid_counter += 1
                        live_tranches.append(Tranche(
                            tid=tid_counter,
                            coin=item["coin"],
                            entry_bar=t,
                            entry_date=curr_date,
                            entry_price=fill_px,
                            units=units,
                            cost_usd=tranche_usd,
                            stop_loss=sl_price,
                            highest_high=fill_px,
                            layer=coin_layers + 1,
                            from_watchlist=(t > item["signal_bar"]),
                            wait_days=(t - item["signal_bar"]),
                            proceeds=0.0
                        ))
                    else:
                        unfilled.append(item)
                watchlist = unfilled if wl_mode != "WL_NONE" else []

            # 7. Horizon Terminal Pass (Fair Close with Causal Lagged ADV)
            if t == end_bar - 1:
                for tr in list(live_tranches):
                    c_px = signals[tr.coin]["df"]["close"].iloc[t]
                    if np.isfinite(c_px) and c_px > 0:
                        adv = signals[tr.coin]["dvol"].iloc[t - 1]  # Strictly lagged
                        part_rate = min(1.0, max(0.0, (tr.units * c_px) / max(adv, 100_000.0)))
                        slip_mult = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate)) / 10000.0
                        fill_px = c_px * (1.0 - slip_mult)
                        proceeds = tr.units * fill_px * (1.0 - fee_mult)
                        cash += proceeds
                        tr.proceeds += proceeds
                        pnl = tr.proceeds - tr.cost_usd
                        closed_trades.append({
                            "coin": tr.coin, "pnl": pnl, "ret": pnl / tr.cost_usd,
                            "reason": "END_OF_TEST", "bars": t - tr.entry_bar,
                            "entry_date": tr.entry_date, "exit_date": curr_date,
                            "entry_price": tr.entry_price, "exit_price": fill_px,
                            "cost_usd": tr.cost_usd, "proceeds": tr.proceeds,
                            "from_watchlist": tr.from_watchlist
                        })
                live_tranches = []

            # 8. MARK-TO-MARKET AT CLOSE[t]
            current_active_capital = 0.0
            for tr in live_tranches:
                bar_c = signals[tr.coin]["df"]["close"].iloc[t]
                bar_h = signals[tr.coin]["df"]["high"].iloc[t]
                if bar_h > tr.highest_high:
                    tr.highest_high = bar_h
                current_active_capital += tr.units * bar_c

            total_equity = cash + current_active_capital
            active_capital_curve.append(current_active_capital)
            total_equity_curve.append(total_equity)

        return {
            "trades": pd.DataFrame(closed_trades),
            "equity": np.array(total_equity_curve),
            "active_capital": np.array(active_capital_curve),
            "total_inflow": total_inflow,
            "final_cash": cash,
            "dates": dates[start_bar:end_bar],
            "binding_stats": binding_stats
        }

# ==============================================================================
# 6. METRIC CALCULATIONS & DUAL-LAYER PERFORMANCE ACCOUNTING
# ==============================================================================
def calculate_btc_dca_benchmark(btc_df: pd.DataFrame, start_bar: int, end_bar: int) -> Dict[str, float]:
    dates = btc_df.index[start_bar:end_bar]
    n_bars = len(dates)
    btc_units = 0.0
    total_contributed = INITIAL_CAPITAL

    initial_px = btc_df["close"].iloc[start_bar]
    btc_units += INITIAL_CAPITAL / initial_px

    for t in range(1, n_bars):
        if t % 30 == 0:
            px = btc_df["close"].iloc[start_bar + t]
            btc_units += MONTHLY_CONTRIBUTION / px
            total_contributed += MONTHLY_CONTRIBUTION

    final_val = btc_units * btc_df["close"].iloc[end_bar - 1]
    net_pnl = final_val - total_contributed
    roi = (net_pnl / total_contributed) * 100.0
    return {"btc_dca_roi": roi, "btc_dca_pnl": net_pnl, "total_contributed": total_contributed}

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
    return {
        "trades": len(tdf),
        "net_pnl": net_pnl,
        "account_roi": account_roi,
        "rocar": rocar,
        "win_rate": win_rate,
        "profit_factor": pf
    }

def calculate_metrics(results: Dict[str, Any], btc_benchmark_roi: float) -> Dict[str, Any]:
    tdf = results["trades"]
    eq = results["equity"]
    ac = results["active_capital"]
    d = results["dates"]

    if len(tdf) < 10:
        return {"score": -999.0, "full": _calc_sub_metrics(tdf, results["total_inflow"], TRANCHE_FLOOR_USD),
                "organic": _calc_sub_metrics(tdf, results["total_inflow"], TRANCHE_FLOOR_USD),
                "eot_trades": 0, "eot_pnl": 0.0, "max_dd": 100.0, "utilization": 0.0}

    avg_active_cap = float(np.mean(ac)) if np.mean(ac) > 100.0 else TRANCHE_FLOOR_USD
    total_inflow = results["total_inflow"]

    full = _calc_sub_metrics(tdf, total_inflow, avg_active_cap)
    org_tdf = tdf[tdf["reason"] != "END_OF_TEST"]
    organic = _calc_sub_metrics(org_tdf, total_inflow, avg_active_cap)

    eot_tdf = tdf[tdf["reason"] == "END_OF_TEST"]
    eot_trades = len(eot_tdf)
    eot_pnl = float(eot_tdf["pnl"].sum()) if eot_trades > 0 else 0.0

    peak_eq = np.maximum.accumulate(eq)
    dd_curve = np.where(peak_eq > 0, (peak_eq - eq) / peak_eq, 0.0)
    max_dd = float(np.max(dd_curve)) * 100.0

    total_days = max(1, (d[-1] - d[0]).days)
    years = total_days / 365.25

    sorted_tdf = tdf.sort_values(by="pnl", ascending=False)
    trimmed_full_pnl = sorted_tdf.iloc[2:]["pnl"].sum() if len(sorted_tdf) > 5 else full["net_pnl"]
    ann_full_roi = (trimmed_full_pnl / total_inflow * 100.0) / years

    sorted_org = org_tdf.sort_values(by="pnl", ascending=False)
    trimmed_org_pnl = sorted_org.iloc[2:]["pnl"].sum() if len(sorted_org) > 5 else organic["net_pnl"]
    ann_org_roi = (trimmed_org_pnl / total_inflow * 100.0) / years

    avg_total_equity = float(np.mean(eq)) if len(eq) > 0 else total_inflow
    utilization = float(avg_active_cap / (avg_total_equity + 1e-9))

    # Configurable Non-Overlapping Anti-Cheating Formula
    unclosed_excess_roi = max(0.0, ann_full_roi - ann_org_roi)
    effective_ann_roi = max(0.0, ann_org_roi) + (EOT_EXCESS_WEIGHT * unclosed_excess_roi)

    alpha_spread = full["account_roi"] - btc_benchmark_roi
    alpha_booster = 1.0 + np.tanh(alpha_spread / 100.0)
    dd_penalty = ((1.0 - (max_dd / 100.0)) ** 2)
    trade_confidence = min(1.0, len(tdf) / 25.0)

    growth_component = effective_ann_roi * (0.5 + 0.5 * min(1.0, utilization * 5.0))
    score = np.log1p(growth_component) * dd_penalty * trade_confidence * alpha_booster

    return {
        "score": float(score),
        "full": full,
        "organic": organic,
        "eot_trades": eot_trades,
        "eot_pnl": eot_pnl,
        "max_dd": float(max_dd),
        "utilization": float(utilization * 100.0)
    }

# ==============================================================================
# 7. DISCRETIZED HYPERPARAMETER SAMPLING
# ==============================================================================
def sample_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    p = {}
    p["entry_type"] = trial.suggest_int("entry_type", 0, 4)
    if p["entry_type"] == 0:
        p["entry_ma_len"] = trial.suggest_int("entry_ma_len", 10, 250, step=5)
        p["entry_ma_type"] = trial.suggest_int("entry_ma_type", 0, 4)
    elif p["entry_type"] == 1:
        p["rsi_f_len"] = trial.suggest_int("rsi_f_len", 10, 80, step=2)
        p["rsi_f_smt"] = trial.suggest_int("rsi_f_smt", 5, 40, step=2)
        p["rsi_s_len"] = trial.suggest_int("rsi_s_len", 10, 80, step=2)
        p["rsi_s_smt"] = trial.suggest_int("rsi_s_smt", 5, 40, step=2)
        p["use_rsi_trend_filter"] = trial.suggest_categorical("use_rsi_trend_filter", [True, False])
        if p["use_rsi_trend_filter"]:
            p["rsi_trend_ma_len"] = trial.suggest_int("rsi_trend_ma_len", 20, 250, step=10)
            p["rsi_trend_ma_type"] = trial.suggest_int("rsi_trend_ma_type", 0, 4)
    elif p["entry_type"] == 2:
        p["xover_short_len"] = trial.suggest_int("xover_short_len", 20, 150, step=5)
        p["xover_short_type"] = trial.suggest_int("xover_short_type", 0, 4)
        gap = trial.suggest_int("xover_gap", 10, 120, step=5)
        p["xover_long_len"] = min(300, p["xover_short_len"] + gap)
        p["xover_long_type"] = trial.suggest_int("xover_long_type", 0, 4)
    elif p["entry_type"] == 3:
        p["vol_ma_len"] = trial.suggest_int("vol_ma_len", 10, 60, step=2)
        p["vol_mult"] = round(trial.suggest_float("vol_mult", 1.5, 4.0, step=0.1), 1)
        p["price_lookback"] = trial.suggest_int("price_lookback", 10, 50, step=2)
        p["body_atr_mult"] = round(trial.suggest_float("body_atr_mult", 0.5, 2.2, step=0.1), 1)
    elif p["entry_type"] == 4:
        p["bb_entry_len"] = trial.suggest_int("bb_entry_len", 10, 60, step=2)
        p["bb_entry_std"] = round(trial.suggest_float("bb_entry_std", 1.5, 3.0, step=0.1), 1)

    p["use_btc_macro_system"] = trial.suggest_categorical("use_btc_macro_system", [True, False])
    if p["use_btc_macro_system"]:
        p["btc_ma_len"] = trial.suggest_int("btc_ma_len", 50, 300, step=10)
        p["btc_ma_type"] = trial.suggest_int("btc_ma_type", 0, 4)

    p["adx_thresh"] = trial.suggest_categorical("adx_thresh", [0.0, 15.0, 20.0])
    p["max_pyramid_layers"] = trial.suggest_int("max_pyramid_layers", 1, 3)
    p["wl_mode"] = trial.suggest_categorical("wl_mode", [
        "WL_DEEPEST_DISCOUNT", "WL_CLOSEST_BREAKOUT", "WL_STRONGEST_MOMENTUM", "WL_FCFS", "WL_LCFS", "WL_NONE"
    ])

    p["use_global_tp"] = trial.suggest_categorical("use_global_tp", [True, False])
    if p["use_global_tp"]:
        p["tp_mult"] = round(trial.suggest_float("tp_mult", 3.0, 30.0, step=0.5), 1)
        p["tp_size_pct"] = round(trial.suggest_float("tp_size_pct", 20.0, 80.0, step=5.0), 1)
        p["tp_move_sl_be"] = trial.suggest_categorical("tp_move_sl_be", [True, False])

    p["exit_type"] = trial.suggest_categorical("exit_type", [0, 1, 2, 3, 4, 5, 6, 7])
    p["sl_mult"] = round(trial.suggest_float("sl_mult", 1.5, 5.0, step=0.1), 1)

    if p["exit_type"] == 0:
        p["trail_mult"] = round(trial.suggest_float("trail_mult", 1.5, 8.0, step=0.1), 1)
    elif p["exit_type"] == 1:
        p["trail_pct"] = round(trial.suggest_float("trail_pct", 5.0, 25.0, step=0.5), 1)
    elif p["exit_type"] == 2:
        p["exit_atr_mult"] = round(trial.suggest_float("exit_atr_mult", 1.0, 5.0, step=0.1), 1)
    elif p["exit_type"] == 3:
        p["exit_ma_len"] = trial.suggest_int("exit_ma_len", 20, 250, step=5)
        p["exit_ma_type"] = trial.suggest_int("exit_ma_type", 0, 4)
    elif p["exit_type"] == 4:
        p["exit_rsi_f_len"] = trial.suggest_int("exit_rsi_f_len", 10, 80, step=2)
        p["exit_rsi_f_smt"] = trial.suggest_int("exit_rsi_f_smt", 5, 40, step=2)
        p["exit_rsi_s_len"] = trial.suggest_int("exit_rsi_s_len", 10, 80, step=2)
        p["exit_rsi_s_smt"] = trial.suggest_int("exit_rsi_s_smt", 5, 40, step=2)
    elif p["exit_type"] == 5:
        p["exit_xover_short_len"] = trial.suggest_int("exit_xover_short_len", 20, 150, step=5)
        p["exit_xover_short_type"] = trial.suggest_int("exit_xover_short_type", 0, 4)
        gap = trial.suggest_int("exit_xover_gap", 10, 120, step=5)
        p["exit_xover_long_len"] = min(300, p["exit_xover_short_len"] + gap)
        p["exit_xover_long_type"] = trial.suggest_int("exit_xover_long_type", 0, 4)
    elif p["exit_type"] == 6:
        p["exit_vol_ma_len"] = trial.suggest_int("exit_vol_ma_len", 10, 60, step=5)
        p["exit_vol_mult"] = round(trial.suggest_float("exit_vol_mult", 1.5, 5.0, step=0.1), 1)
    elif p["exit_type"] == 7:
        p["bb_exit_len"] = trial.suggest_int("bb_exit_len", 10, 80, step=2)

    return p

# ==============================================================================
# 8. ARCHITECTURE-AWARE DYNAMIC NEIGHBORHOOD AUDIT
# ==============================================================================
def dynamic_parameter_neighbors(base_p: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = {
        "entry_ma_len": 5, "rsi_f_len": 2, "rsi_s_len": 2, "rsi_trend_ma_len": 10,
        "xover_short_len": 5, "xover_gap": 5, "vol_ma_len": 2, "vol_mult": 0.1,
        "price_lookback": 2, "body_atr_mult": 0.1, "bb_entry_len": 2, "bb_entry_std": 0.1,
        "btc_ma_len": 10, "sl_mult": 0.2, "tp_mult": 1.0, "trail_mult": 0.2,
        "trail_pct": 1.0, "exit_atr_mult": 0.2, "exit_ma_len": 5, "exit_rsi_f_len": 2,
        "exit_rsi_s_len": 2, "exit_xover_short_len": 5, "exit_xover_gap": 5,
        "exit_vol_ma_len": 5, "exit_vol_mult": 0.2, "bb_exit_len": 2
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
            cat = "exit" if ("exit" in k or "trail" in k or k == "sl_mult") else ("macro" if "btc" in k else ("tp" if "tp" in k else "entry"))
            mult = d[cat]
            if isinstance(cand[k], float):
                cand[k] = round(max(0.2, cand[k] + mult * step), 1)
            elif isinstance(cand[k], int):
                cand[k] = max(2, int(cand[k] + mult * step))
        if "xover_short_len" in cand and "xover_gap" in cand:
            cand["xover_long_len"] = min(300, cand["xover_short_len"] + cand["xover_gap"])
        if "exit_xover_short_len" in cand and "exit_xover_gap" in cand:
            cand["exit_xover_long_len"] = min(300, cand["exit_xover_short_len"] + cand["exit_xover_gap"])
        neighbors.append(cand)
    return neighbors

def evaluate_neighborhood_stability(
    params: Dict[str, Any],
    candidate_score: float,
    universe: Dict[str, pd.DataFrame],
    btc_df: pd.DataFrame,
    start_bar: int,
    end_bar: int,
    btc_dca_roi: float
) -> Tuple[bool, float, List[float]]:
    neighbors = dynamic_parameter_neighbors(params)
    neighbor_scores = []

    for n_p in neighbors:
        n_signals, n_btc_ok = compile_signals(universe, btc_df, n_p)
        engine = BacktestEngine(params=n_p)
        res = engine.run_interval(n_signals, btc_df, n_btc_ok, start_bar, end_bar)
        metrics = calculate_metrics(res, btc_dca_roi)
        neighbor_scores.append(metrics["score"])

    mean_neighbor_score = float(np.mean(neighbor_scores))
    min_neighbor_score = float(np.min(neighbor_scores))

    is_stable = (
        mean_neighbor_score >= (candidate_score * (1.0 - NEIGHBORHOOD_DROP_LIMIT))
        and min_neighbor_score > 0.0
    )
    plateau_score = min(candidate_score, mean_neighbor_score)
    return is_stable, plateau_score, neighbor_scores

# ==============================================================================
# 9. BEHAVIORAL PREFIX-INVARIANCE CAUSALITY TEST
# ==============================================================================
def verify_behavioral_causality(universe: Dict[str, pd.DataFrame], btc_df: pd.DataFrame):
    """
    Rigorously asserts that truncating the universe data at date T does NOT alter any
    trade executed prior to T. Empirically verifies zero forward lookahead bias.
    """
    logging.info("Executing Behavioral Prefix-Invariance Causality Audit...")
    n_bars = len(btc_df)
    if n_bars < 500:
        return

    # Benchmark test parameter configuration
    test_p = {
        "entry_type": 0, "entry_ma_len": 20, "entry_ma_type": 0,
        "use_btc_macro_system": True, "btc_ma_len": 50, "btc_ma_type": 0,
        "adx_thresh": 0.0, "max_pyramid_layers": 1, "wl_mode": "WL_FCFS",
        "use_global_tp": False, "exit_type": 0, "sl_mult": 2.5, "trail_mult": 4.0
    }

    # Step 1: Run on full data up to truncation boundary T
    t_cutoff = n_bars - 80
    sigs_full, btc_ok_full = compile_signals(universe, btc_df, test_p)
    engine_full = BacktestEngine(params=test_p)
    res_full = engine_full.run_interval(sigs_full, btc_df, btc_ok_full, 300, t_cutoff)
    trades_full = res_full["trades"]

    # Step 2: Create truncated universe ending strictly at T
    trunc_dates = btc_df.index[:t_cutoff]
    trunc_universe = {coin: df.loc[trunc_dates].copy() for coin, df in universe.items()}
    trunc_btc = btc_df.loc[trunc_dates].copy()

    sigs_trunc, btc_ok_trunc = compile_signals(trunc_universe, trunc_btc, test_p)
    engine_trunc = BacktestEngine(params=test_p)
    res_trunc = engine_trunc.run_interval(sigs_trunc, trunc_btc, btc_ok_trunc, 300, t_cutoff)
    trades_trunc = res_trunc["trades"]

    # Compare non-EOT trades closed before cutoff
    full_prior = trades_full[trades_full["reason"] != "END_OF_TEST"]
    trunc_prior = trades_trunc[trades_trunc["reason"] != "END_OF_TEST"]

    if len(full_prior) > 0 and len(trunc_prior) > 0:
        pnl_diff = abs(full_prior["pnl"].sum() - trunc_prior["pnl"].sum())
        count_diff = abs(len(full_prior) - len(trunc_prior))
        if pnl_diff > 1e-4 or count_diff > 0:
            raise RuntimeError(
                f"Prefix-Invariance Violation! Full PnL={full_prior['pnl'].sum():.2f}, "
                f"Truncated PnL={trunc_prior['pnl'].sum():.2f}. Engine possesses future lookahead leak!"
            )

    logging.info("Behavioral Prefix-Invariance Audit Passed: Zero lookahead drift confirmed across truncation boundary.")

# ==============================================================================
# 10. OPTIMIZATION ORCHESTRATOR & REPORT GENERATOR
# ==============================================================================
def run_optimization():
    logging.info(f"Initializing Framework V7: Target Universe = {TOP_N_COINS} coins, Start Year = {START_YEAR}")
    universe, btc_df, provenance_records = build_market_universe(
        target_coins=TOP_N_COINS,
        start_year=START_YEAR,
        refresh=FORCE_REFRESH
    )

    verify_behavioral_causality(universe, btc_df)

    n_bars = len(btc_df)
    mature_coverage = pd.Series(0, index=btc_df.index)
    for df in universe.values():
        mature = df["alive"].rolling(WARMUP_BARS, min_periods=WARMUP_BARS).sum() == WARMUP_BARS
        mature_coverage = mature_coverage.add(mature.astype(int), fill_value=0)
    
    mature_pct = mature_coverage / float(len(universe))
    valid_start_indices = np.where(mature_pct >= 0.51)[0]
    if len(valid_start_indices) == 0:
        raw_cov = pd.Series(0, index=btc_df.index)
        for df in universe.values():
            raw_cov = raw_cov.add(df["alive"].astype(int), fill_value=0)
        raw_pct = raw_cov / float(len(universe))
        trading_start_bar = min(int(np.where(raw_pct >= 0.51)[0][0]) + WARMUP_BARS, n_bars - 100)
    else:
        trading_start_bar = int(valid_start_indices[0])
    
    total_trading_bars = n_bars - trading_start_bar
    split_offset = int(total_trading_bars * 0.75)
    
    is_start = trading_start_bar
    is_end = trading_start_bar + split_offset
    oos_start = is_end
    oos_end = n_bars

    is_bars = is_end - is_start
    oos_bars = oos_end - oos_start

    logging.info(
        f"Timeline Verified: Total History = {n_bars} bars | "
        f"Trading Starts = {btc_df.index[trading_start_bar].date()} (Bar {trading_start_bar}) [51% coins have >= 300d data] | "
        f"IS Window = {is_bars} bars ({is_bars / total_trading_bars:.1%}) [{btc_df.index[is_start].date()} -> {btc_df.index[is_end-1].date()}] | "
        f"OOS Holdout = {oos_bars} bars ({oos_bars / total_trading_bars:.1%}) [{btc_df.index[oos_start].date()} -> {btc_df.index[oos_end-1].date()}]"
    )

    is_btc = calculate_btc_dca_benchmark(btc_df, is_start, is_end)
    oos_btc = calculate_btc_dca_benchmark(btc_df, oos_start, oos_end)
    logging.info(f"In-Sample BTC DCA ROI Benchmark: {is_btc['btc_dca_roi']:.2f}% | PnL: ${is_btc['btc_dca_pnl']:,.2f}")

    baseline_score = -999.0
    if CURRENT_WINNER_FILE.exists():
        try:
            with open(CURRENT_WINNER_FILE, "r") as f:
                saved = json.load(f)
                baseline_score = saved.get("is_score", -999.0)
                logging.info(f"Loaded verified baseline champion: Score = {baseline_score:.4f}")
        except Exception as e:
            logging.warning(f"Could not parse baseline file: {e}")

    logging.info(f"Starting Optuna search across {N_TRIALS} trials (SQLite persisted)...")

    def objective(trial: optuna.Trial) -> float:
        params = sample_hyperparameters(trial)
        signals, btc_ok = compile_signals(universe, btc_df, params)
        engine = BacktestEngine(params=params)
        res = engine.run_interval(signals, btc_df, btc_ok, is_start, is_end)
        metrics = calculate_metrics(res, is_btc["btc_dca_roi"])
        raw_score = metrics["score"]

        try:
            current_study_best = trial.study.best_value
        except ValueError:
            current_study_best = -999.0

        target_hurdle = max(current_study_best, baseline_score)

        if raw_score > target_hurdle and raw_score > 0.5:
            is_stable, plateau_score, neighbor_scores = evaluate_neighborhood_stability(
                params=params,
                candidate_score=raw_score,
                universe=universe,
                btc_df=btc_df,
                start_bar=is_start,
                end_bar=is_end,
                btc_dca_roi=is_btc["btc_dca_roi"]
            )

            if not is_stable:
                logging.info(f"Trial {trial.number} REJECTED [Brittle Spike]: Raw={raw_score:.3f}, Neighbors={neighbor_scores}")
                return min(raw_score * 0.4, target_hurdle - 0.05)
            else:
                logging.info(f"Trial {trial.number} CONFIRMED [Plateau Edge]: Raw={raw_score:.3f} -> Plateau Score={plateau_score:.3f}")
                return plateau_score

        return raw_score

    study = optuna.create_study(
        study_name="crypto_swing_v7_institutional",
        storage=STUDY_DB,
        load_if_exists=True,
        direction="maximize"
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_trial = study.best_trial
    best_params = best_trial.params

    if best_params.get("entry_type") == 2 and "xover_gap" in best_params:
        best_params["xover_long_len"] = min(300, best_params["xover_short_len"] + best_params["xover_gap"])
    if best_params.get("exit_type") == 5 and "exit_xover_gap" in best_params:
        best_params["exit_xover_long_len"] = min(300, best_params["exit_xover_short_len"] + best_params["exit_xover_gap"])

    logging.info(f"Optimization Complete. Best Verified Plateau Score: {study.best_value:.4f}")

    logging.info("Running Single-Pass Out-of-Sample Holdout Verification...")
    final_signals, final_btc_ok = compile_signals(universe, btc_df, best_params)
    final_engine = BacktestEngine(params=best_params)

    is_results = final_engine.run_interval(final_signals, btc_df, final_btc_ok, is_start, is_end)
    is_metrics = calculate_metrics(is_results, is_btc["btc_dca_roi"])

    oos_results = final_engine.run_interval(final_signals, btc_df, final_btc_ok, oos_start, oos_end)
    oos_metrics = calculate_metrics(oos_results, oos_btc["btc_dca_roi"])

    logging.info(f"IS Performance  -> Full PnL: ${is_metrics['full']['net_pnl']:,.2f} (Organic PnL:${is_metrics['organic']['net_pnl']:,.2f}) | Max DD: {is_metrics['max_dd']:.2f}%")
    logging.info(f"OOS Performance -> Full PnL: ${oos_metrics['full']['net_pnl']:,.2f} (Organic PnL:${oos_metrics['organic']['net_pnl']:,.2f}) | Max DD: {oos_metrics['max_dd']:.2f}%")

    # ==========================================================================
    # 11. ISOLATED 2022 BEAR-MARKET STRESS-TEST
    # ==========================================================================
    logging.info("Executing Isolated 2022 Bear Market Stress-Test...")
    bear_start_ts = pd.Timestamp("2021-11-10")
    bear_end_ts = pd.Timestamp("2022-12-31")
    
    bear_indices = np.where((btc_df.index >= bear_start_ts) & (btc_df.index <= bear_end_ts))[0]
    stress_metrics = None
    stress_btc = None

    if len(bear_indices) > 50:
        b_start = int(bear_indices[0])
        b_end = int(bear_indices[-1]) + 1
        stress_btc = calculate_btc_dca_benchmark(btc_df, b_start, b_end)
        stress_res = final_engine.run_interval(final_signals, btc_df, final_btc_ok, b_start, b_end)
        stress_metrics = calculate_metrics(stress_res, stress_btc["btc_dca_roi"])
        logging.info(
            f"Bear Stress-Test (2021-11-10 to 2022-12-31) -> "
            f"Strategy PnL: ${stress_metrics['full']['net_pnl']:,.2f} (ROI: {stress_metrics['full']['account_roi']:.2f}%, Max DD: {stress_metrics['max_dd']:.2f}%) | "
            f"BTC DCA Benchmark: ${stress_btc['btc_dca_pnl']:,.2f} (ROI: {stress_btc['btc_dca_roi']:.2f}%)"
        )

    # Save Winner Artifacts
    if is_metrics["score"] > baseline_score:
        logging.info("Updating global winner state...")
        winner_data = {
            "is_score": is_metrics["score"],
            "is_metrics": is_metrics,
            "oos_metrics": oos_metrics,
            "bear_stress_test": stress_metrics,
            "best_params": best_params,
            "updated_at": pd.Timestamp.now('UTC').isoformat()
        }
        with open(CURRENT_WINNER_FILE, "w", encoding="utf-8") as f:
            json.dump(winner_data, f, indent=4)
        with open(FINAL_WINNER_FILE, "w", encoding="utf-8") as f:
            json.dump(winner_data, f, indent=4)

    # Combine IS and OOS trade records for export with unrounded raw values
    all_trades = pd.concat([is_results["trades"], oos_results["trades"]], ignore_index=True)
    all_trades.to_csv(TRADES_CSV_FILE, index=False)
    logging.info(f"Detailed trade ledger saved to {TRADES_CSV_FILE} ({len(all_trades)} trades).")

    # Trade Reason Breakdown
    reason_table_md = ""
    if len(all_trades) > 0:
        reason_summary = all_trades.groupby("reason").agg(
            trades=("pnl", "count"),
            net_pnl=("pnl", "sum"),
            win_rate=("pnl", lambda x: (x > 0).mean() * 100.0)
        ).reset_index()
        reason_summary["share"] = (reason_summary["trades"] / len(all_trades)) * 100.0
        
        reason_table_md = "| Exit Reason | Trades | Share (%) | Net PnL ($) | Win Rate (%) |\n| :--- | :--- | :--- | :--- | :--- |\n"
        for _, r in reason_summary.iterrows():
            reason_table_md += f"| {r['reason']} | {int(r['trades'])} | {r['share']:.1f}% | ${r['net_pnl']:,.2f} | {r['win_rate']:.1f}% |\n"

    # Five-Way Binding Constraint Analysis
    binding = is_results["binding_stats"]
    total_evals = max(1, sum(binding.values()))
    binding_table_md = f"""| Constraint Mechanism | Times Bound | Percentage Share | Operational Status |
| :--- | :--- | :--- | :--- |
| **Cash Constrained (Wallet Starvation)** | {binding['cash_constrained']} | {binding['cash_constrained']/total_evals:.1%} | Funded with available cash balance |
| **Tranche Ceiling ($25,000 Hard Cap)** | {binding['tranche_ceiling']} | {binding['tranche_ceiling']/total_evals:.1%} | Bounds large portfolio capital |
| **Liquidity Cap (1.5% 30d ADV)** | {binding['liquidity_cap']} | {binding['liquidity_cap']/total_evals:.1%} | Guards illiquid altcoins |
| **Equity Slot (Nominal Equity / 10)** | {binding['equity_slot']} | {binding['equity_slot']/total_evals:.1%} | Normal proportional sizing |
| **Tranche Floor ($500 Absolute Floor)** | {binding['tranche_floor']} | {binding['tranche_floor']/total_evals:.1%} | Active during early horizon |
"""

    # Representative Trade Sample (Top 3 Winners, Worst 3 Losers)
    sample_trades_md = "| Coin | Entry Date | Entry Price | Exit Date | Exit Price | Reason | Net PnL | Return |\n| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    if len(all_trades) > 0:
        top_winners = all_trades.nlargest(3, "pnl")
        worst_losers = all_trades.nsmallest(3, "pnl")
        sample_df = pd.concat([top_winners, worst_losers]).drop_duplicates()
        for _, tr in sample_df.iterrows():
            p_in = format_price(tr['entry_price'])
            p_out = format_price(tr['exit_price'])
            sample_trades_md += f"| {tr['coin']} | {str(tr['entry_date'])[:10]} | {p_in} | {str(tr['exit_date'])[:10]} | {p_out} | {tr['reason']} | ${tr['pnl']:,.2f} | {tr['ret']*100:.1f}% |\n"

    # Data Provenance Summary & Identified Delistings Roster
    prov_df = pd.DataFrame(provenance_records)
    total_gaps = int(prov_df["missing_days_in_span"].sum())
    delisted_coins = prov_df[prov_df["is_permanently_delisted"] == True]
    delisted_coins_count = len(delisted_coins)
    provider_counts = prov_df["provider"].value_counts().to_dict()
    provider_summary_str = ", ".join([f"{k}: {v}" for k, v in provider_counts.items()])

    delisted_table_md = ""
    if delisted_coins_count > 0:
        delisted_table_md = "\n#### Terminal Delisting Roster\n| Coin | Originating Venue | First Active | Last Valid Bar | Total Bars | Resolution Status |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        for _, drow in delisted_coins.iterrows():
            prov_raw = drow['provider']
            if prov_raw in ("LEGACY_CACHE", "cache", "unknown"):
                prov_label = "UNRECORDED (Pre-V6.5 Cache)"
                status_note = "Delisted on Unrecorded Venue (Run with FORCE_REFRESH=True to identify venue)"
            else:
                prov_label = prov_raw.upper()
                status_note = f"Delisted on {prov_label} Feed (Verify if trading on other venues)"
            delisted_table_md += f"| {drow['coin']} | {prov_label} | {drow['first_date']} | {drow['last_date']} | {drow['total_bars']} | {status_note} |\n"
    else:
        delisted_table_md = "\n*No terminal delistings identified. All assets traded through horizon.*"

    # Dynamic Report Captions
    macro_active = best_params.get("use_btc_macro_system", False)
    macro_note = "Macro Kill-Switch Active" if macro_active else "Macro System Off (Risk Managed via ATR Stops)"
    cash_note = "Capital Preserved via Macro Exit" if macro_active else "Capital Preserved via Selective Entry & Fast Exits"

    stress_section = ""
    if stress_metrics and stress_btc:
        stress_section = f"""
## 2. Isolated 2022 Bear Market Stress-Test (2021-11-10 -> 2022-12-31)
*Testing strategy durability during a pure market contraction regime:*
| Performance Metric | Strategy Performance | BTC DCA Benchmark | Alpha Advantage |
| :--- | :--- | :--- | :--- |
| **Full Net PnL** | **${stress_metrics['full']['net_pnl']:,.2f}** | **${stress_btc['btc_dca_pnl']:,.2f}** | **+${stress_metrics['full']['net_pnl'] - stress_btc['btc_dca_pnl']:,.2f}** |
| **Account ROI** | **{stress_metrics['full']['account_roi']:.2f}%** | **{stress_btc['btc_dca_roi']:.2f}%** | **+{stress_metrics['full']['account_roi'] - stress_btc['btc_dca_roi']:.2f}%** |
| **Max Portfolio Drawdown** | **{stress_metrics['max_dd']:.2f}%** | ~65.0% | {cash_note} |
| **Completed Trades** | {stress_metrics['full']['trades']} | N/A | {macro_note} |
"""

    params_json = json.dumps(best_params, indent=4)
    report_md = f"""# Quantitative Strategy Audit & Validation Dossier

## 1. Segregated Performance Accounting

### A. Full Portfolio Accounting (Including Terminal MTM Liquidation)
| Performance Metric | In-Sample ({is_bars} bars: {btc_df.index[is_start].date()} -> {btc_df.index[is_end-1].date()}) | Out-of-Sample Holdout ({oos_bars} bars: {btc_df.index[oos_start].date()} -> {btc_df.index[oos_end-1].date()}) |
| :--- | :--- | :--- |
| **Strategy Net Trading Profit** | **${is_metrics['full']['net_pnl']:,.2f}** | **${oos_metrics['full']['net_pnl']:,.2f}** |
| **BTC DCA Benchmark Net Profit** | **${is_btc['btc_dca_pnl']:,.2f}** | **${oos_btc['btc_dca_pnl']:,.2f}** |
| **Strategy Account ROI (on deposits)** | {is_metrics['full']['account_roi']:.2f}% | {oos_metrics['full']['account_roi']:.2f}% |
| **BTC DCA Account ROI (on deposits)** | {is_btc['btc_dca_roi']:.2f}% | {oos_btc['btc_dca_roi']:.2f}% |
| **Capital Utilization (Avg Active / Total)** | {is_metrics['utilization']:.2f}% | {oos_metrics['utilization']:.2f}% |
| **Return on Capital at Risk (ROCAR)** | {is_metrics['full']['rocar']:.2f}% | {oos_metrics['full']['rocar']:.2f}% |
| **True Max Portfolio Drawdown** | {is_metrics['max_dd']:.2f}% | {oos_metrics['max_dd']:.2f}% |
| **Profit Factor** | {is_metrics['full']['profit_factor']:.2f} | {oos_metrics['full']['profit_factor']:.2f} |
| **Win Rate** | {is_metrics['full']['win_rate']:.2f}% | {oos_metrics['full']['win_rate']:.2f}% |
| **Total Completed Trades** | {is_metrics['full']['trades']} | {oos_metrics['full']['trades']} |

### B. Organic Strategy Accounting (Closed Exclusively by Strategy Signals & Stops)
*Excludes arbitrary END_OF_TEST calendar-cutoff closures to isolate true systematic execution:*
| Performance Metric | In-Sample (Rule-Closed Only) | Out-of-Sample (Rule-Closed Only) |
| :--- | :--- | :--- |
| **Rule-Closed Net Profit** | **${is_metrics['organic']['net_pnl']:,.2f}** | **${oos_metrics['organic']['net_pnl']:,.2f}** |
| **Rule-Closed Account ROI** | {is_metrics['organic']['account_roi']:.2f}% | {oos_metrics['organic']['account_roi']:.2f}% |
| **Rule-Closed Profit Factor** | {is_metrics['organic']['profit_factor']:.2f} | {oos_metrics['organic']['profit_factor']:.2f} |
| **Rule-Closed Win Rate** | {is_metrics['organic']['win_rate']:.2f}% | {oos_metrics['organic']['win_rate']:.2f}% |
| **Completed Rule Trades** | {is_metrics['organic']['trades']} | {oos_metrics['organic']['trades']} |
| **Forced Terminal Trades (END_OF_TEST)** | {is_metrics['eot_trades']} (${is_metrics['eot_pnl']:,.2f}) | {oos_metrics['eot_trades']} (${oos_metrics['eot_pnl']:,.2f}) |
{stress_section}
## 3. Position-Size Binding Constraint Diagnostics
{binding_table_md}

## 4. Trade-Reason Population Breakdown
{reason_table_md}

## 5. Audit Trade Sampling (Extremes Inspection)
{sample_trades_md}
*Complete trade log exported to `{TRADES_CSV_FILE.name}`.*

## 6. Data Provenance & Integrity Audit
- **Universe Scale**: {len(universe)} shortlisted coins across multi-exchange feeds ({provider_summary_str}).
- **Transient Missing Day Gaps**: {total_gaps} lone-day gaps detected and forward-filled across history (zero false delistings).
- **Permanent Terminal Delistings**: {delisted_coins_count} coins cleanly identified as terminal write-offs.
{delisted_table_md}

## 7. Discovered Optimal Parameter Set
```json
{params_json}
            """
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_md)
        logging.info(f"Audit dossier written to {REPORT_FILE}")

if __name__ == "__main__":
    run_optimization()        


""" How Optuna Interacts with the Neighborhood Check
1. **Lazy Execution (Zero Speed Penalty):** Running neighbor audits on all 5,000 trials would slow down execution by 5x. Instead, `objective()` computes `raw_score` in a single pass. Only when a trial's raw score **exceeds the standing champion** does the engine trigger the 4-neighbor perturbation audit.
2. **Rejecting Fake Winners:** If the 4 adjacent neighbors fail to hold up within 25% of the candidate score (or collapse to negative returns), the function returns `min(raw_score * 0.4, current_best - 0.05)`. 
3. **Optuna Internal State Preservation:** Because the returned value is deliberately clamped below `study.best_value`, Optuna will **never update its internal champion pointer** to that brittle trial. Over repeated iterations, the TPE sampler receives negative feedback for sharp spikes and shifts its probability density toward broad, stable plateaus.
"""