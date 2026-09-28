#!/usr/bin/env python3
"""
CRYPTO LIVE-PRICE SWING-TRADING COMPANION
V7.4 — V10.2 ENGINE-PARITY + LIVE-PRICE EXECUTION

Purpose:
  Decision-support + portfolio bookkeeping only. No exchange order API.

Parity contract with bt_crypto_103.py:
  * exact V10.2 indicator kernels / entry / state / exit definitions
  * last fully closed UTC daily candle generates the signal
  * current spot price is the live entry/exit reference
  * stop/TP checks use current live spot; no current partial daily candle enters indicators
  * TP uses T high
  * exact watchlist revalidation/expiry/ranking/anti-averaging-down
  * exact dynamic slot/equity/ADV sizing math
  * audited crypto fee/slippage formulas
  * 20-bar permanent-delist guard
  * no fabricated historical execution catch-up; live mode acts only on the latest closed day + current spot

Live-specific adaptation:
  * INR wallet representation with current USD/INR conversion
  * no terminal END_OF_TEST liquidation
  * persistent state and email reporting
  * fractional crypto units are retained, matching the backtest
"""
from __future__ import annotations

import argparse
import copy
import html
import json
import logging
import os
import smtplib
import ssl
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from swing_live_parity_core_v10x import (
    FastIndicators,
    SignalArrays,
    compile_single_symbol,
    np_rolling_mean,
    reconstitute_params,
)

# ==============================================================================
# CONFIG — strategy/risk values match bt_crypto_103.py
# ==============================================================================

ENGINE_VERSION = "V10.2-INSTITUTIONAL"
UNIVERSE_NAME = "TOP_CRYPTO_LIQUID"
TOP_N_COINS = 50
START_YEAR = 2017
MIN_HISTORY_DAYS = 250
WARMUP_BARS = 300
IST = ZoneInfo("Asia/Kolkata")
MACRO_INDEX_TICKER = "BTC"

BASE_DIR = Path(os.environ.get("SWING_BOT_BASE_DIR", Path.home() / "CryptoSwingBot_V7"))
DATA_DIR = BASE_DIR / "data_cache"
OUTPUT_DIR = BASE_DIR / "output_crypto" / f"{UNIVERSE_NAME}_TOP_{TOP_N_COINS}"
STATE_FILE = Path(os.environ.get("CRYPTO_STATE_PATH", str(BASE_DIR / "state.json")))
WINNER_PATH = Path(os.environ.get(
    "CRYPTO_WINNER_JSON_PATH",
    str(OUTPUT_DIR / "winner.json"),
))
LEGACY_WINNER_PATH = BASE_DIR / "output_crypto" / f"{UNIVERSE_NAME}_TOP_{TOP_N_COINS}" / "winner.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set None to keep persisted rolling cash. To bootstrap/reset wallet cash, set a value.
SET_WALLET_CASH_INR: Optional[float] = None

FORCE_RERUN_TODAY = True
DEFAULT_USD_INR_RATE = 95.4
LIQUIDITY_FLOOR_USD = 2_000_000.0
TRANCHE_FLOOR_USD = 50.0
DEFAULT_MAX_CONCURRENT_TRANCHES = 6
MAX_POSITION_EQUITY_PCT = 0.25
MAX_ADV_PARTICIPATION = 0.015
WL_MAX_AGE_BARS = 15
LIVE_PRICE_MAX_AGE_SECONDS = float(os.environ.get("CRYPTO_LIVE_PRICE_MAX_AGE_SECONDS", "90"))

CRYPTO_EXCHANGE_FEE_RATE = 0.0010
BASE_SLIPPAGE_BPS = 6.0
IMPACT_COEF_BPS = 120.0

DENY_LIST = {
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDD", "USDP", "FDUSD", "PYUSD", "EUR", "GBP",
    "USDE", "SUSDE", "USDY", "BUIDL", "RLUSD", "USD1", "FRAX", "LUSD", "CRVUSD", "GHO",
    "WBTC", "CBBTC", "TBTC", "LBTC", "RENBTC", "BTCB", "SOLVBTC", "WETH", "STETH", "WSTETH",
    "CBETH", "RETH", "WEETH", "EZETH", "WSOL", "JITOSOL", "MSOL", "BNSOL", "WBNB", "SAVAX"
}
LEGACY_WHITELIST = [
    "BTC", "ETH", "SOL", "XRP", "XLM", "DOGE", "LTC", "ADA", "LINK", "AVAX", "DOT", "BCH",
    "NEAR", "ATOM", "UNI", "ETC", "XMR", "ALGO", "VET", "FIL", "ICP", "AAVE", "SAND"
]
STABLECOINS = {
    "USDT", "USDC", "BUSD", "DAI", "FDUSD", "TUSD", "USDD", "USDP", "FRAX",
    "LUSD", "GUSD", "PYUSD", "EURT", "EURS", "USTC", "USDJ", "CUSD", "SUSD",
    "USDE", "USD0", "CRVUSD", "GHO", "FIRST", "MIM", "FEI", "ALUSD", "RLUSD"
}
FIAT_COMMODITY_PEGS = {"XAUT", "PAXG", "EUROC", "XSGD", "EURCV"}
WRAPPED_TOKEN_NAMES = {
    "WBTC", "WETH", "WMATIC", "WAVAX", "WBNB", "WSOL", "WFTM", "WTRX", "WKAVA",
    "WBETH", "WROSE", "WNEAR", "WCELO", "WKLAY", "WBTT", "WGLMR", "WQTUM", "CBBTC"
}
LIQUID_STAKING_PREFIXES = (
    "ST", "WST", "R", "CB", "ANKR", "MSOL", "SAVAX", "BNSOL", "JITOSOL", "OSOL"
)
PROTECTED_TICKERS = {
    "STX", "STORJ", "ROSE", "RENDER", "RUNE", "RVN", "RAD", "REQ", "RLC",
    "RARE", "RAY", "RON", "RDNT", "RPL", "STEEM", "STRAX", "STRK"
}
LEVERAGED_SUFFIXES = (
    "UP", "DOWN", "BULL", "BEAR", "2L", "2S", "3L", "3S", "4L", "4S", "5L", "5S"
)

try:
    from google.colab import drive, userdata  # type: ignore
    IN_COLAB = True
except ImportError:
    drive = None
    userdata = None
    IN_COLAB = False

if IN_COLAB:
    try:
        drive.mount("/content/drive", force_remount=False)
        if str(BASE_DIR) == str(Path.home() / "CryptoSwingBot_V7"):
            BASE_DIR = Path("/content/drive/MyDrive/CryptoSwingBot_V7")
            DATA_DIR = BASE_DIR / "data_cache"
            OUTPUT_DIR = BASE_DIR / "output_crypto" / f"{UNIVERSE_NAME}_TOP_{TOP_N_COINS}"
            STATE_FILE = Path(os.environ.get("CRYPTO_STATE_PATH", str(BASE_DIR / "state.json")))
            WINNER_PATH = Path(os.environ.get("CRYPTO_WINNER_JSON_PATH", str(OUTPUT_DIR / "winner.json")))
            DATA_DIR.mkdir(parents=True, exist_ok=True); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        pass

logger = logging.getLogger("crypto_live_v7_3")
logger.setLevel(logging.INFO)
if not logger.handlers:
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(sh)
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    fh = logging.FileHandler(BASE_DIR / "crypto_live_v7_3.log")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(fh)

_HTTP_SESSION = requests.Session()
_HTTP_SESSION.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})


def get_secret(name: str) -> Optional[str]:
    if IN_COLAB and userdata is not None:
        try:
            v = userdata.get(name)
            if v:
                return str(v)
        except Exception:
            pass
    return os.environ.get(name)


def send_email(subject: str, body: str) -> None:
    if os.environ.get("DRY_RUN", "false").lower() in ("1", "true", "yes", "y", "on"):
        logger.info("[DRY RUN] Email suppressed: %s", subject)
        return
    user, pw, to = get_secret("GMAIL_USER"), get_secret("GMAIL_APP_PASSWORD"), get_secret("RECIPIENT_EMAIL")
    if not user or not pw or not to:
        logger.warning("Missing Gmail secrets; email skipped.")
        return
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    msg.attach(MIMEText(body, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as server:
            server.login(user, pw)
            server.sendmail(user, to, msg.as_string())
        logger.info("Email delivered: %s", subject)
    except Exception as e:
        logger.error("Email send failed: %s", e)


# ==============================================================================
# V10.2 FEE / FILL ENGINE
# ==============================================================================

def _fee_dict() -> Dict[str, float]:
    return {"exchange_fee": 0.0, "slippage_cost": 0.0}


def _merge_fees(acc: Dict[str, float], inc: Dict[str, float]) -> None:
    for k, v in inc.items():
        acc[k] = float(acc.get(k, 0.0) + v)


def compute_buy_cost_audited(gross_usd: float, slip_cost: float) -> Tuple[float, Dict[str, float]]:
    if gross_usd <= 0:
        return 0.0, _fee_dict()
    fee = gross_usd * CRYPTO_EXCHANGE_FEE_RATE
    return gross_usd + fee, {"exchange_fee": fee, "slippage_cost": slip_cost}


def compute_sell_proceeds_audited(gross_usd: float, slip_cost: float) -> Tuple[float, Dict[str, float]]:
    if gross_usd <= 0:
        return 0.0, _fee_dict()
    fee = gross_usd * CRYPTO_EXCHANGE_FEE_RATE
    return max(0.0, gross_usd - fee), {"exchange_fee": fee, "slippage_cost": slip_cost}


def compute_max_affordable_tranche(cash_usd: float, adv_30d: float) -> float:
    clean_adv = adv_30d if np.isfinite(adv_30d) else LIQUIDITY_FLOOR_USD
    adv = max(clean_adv, LIQUIDITY_FLOOR_USD)
    usable_cash = max(0.0, cash_usd)
    guess = usable_cash / (1.0 + CRYPTO_EXCHANGE_FEE_RATE + BASE_SLIPPAGE_BPS / 10000.0)
    for _ in range(3):
        part_rate = min(1.0, max(0.0, guess / adv))
        slip_mult = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate)) / 10000.0
        guess = usable_cash / (1.0 + CRYPTO_EXCHANGE_FEE_RATE + slip_mult)
    return float(np.nan_to_num(guess * (1.0 - 1e-6), nan=0.0))


def _buy_fill_audited(tranche_usd: float, ref_open_price: float, slip_mult: float):
    fill_px = ref_open_price * (1.0 + slip_mult)
    units = tranche_usd / ref_open_price if ref_open_price > 0 else 0.0
    gross = units * fill_px
    slip_cost = units * (fill_px - ref_open_price)
    total, fees = compute_buy_cost_audited(gross, slip_cost)
    return fill_px, units, total, fees


def _sell_fill_audited(units: float, ref_price: float, slip_mult: float):
    fill_px = ref_price * (1.0 - slip_mult)
    gross = units * fill_px
    slip_cost = units * (ref_price - fill_px)
    proceeds, fees = compute_sell_proceeds_audited(gross, slip_cost)
    return fill_px, proceeds, fees


FALLBACK_PARAMS = {
    "entry_type": 3, "adx_thresh": 15.0, "vol_ma_len": 24, "vol_mult": 3.5,
    "price_lookback": 38, "body_atr_mult": 1.3, "use_market_macro_system": False,
    "max_concurrent_tranches": 7, "max_pyramid_layers": 2, "wl_mode": "WL_NONE",
    "use_global_tp": False, "be_trigger_atr": 1.5, "max_holding_bars": 30, "sl_mult": 4.4,
    "exit_type": 7, "trail_atr_mult": 0.0, "bb_exit_len": 44, "macro_active_exit": False,
}


def load_strategy_params() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    for path in (WINNER_PATH, LEGACY_WINNER_PATH):
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                p = data.get("best_params") or data.get("params")
                if p:
                    p = reconstitute_params(p, max_xover_long=180, max_exit_xover_long=150)
                    logger.info("Loaded V10.2 champion from %s (engine=%s, score=%s)",
                                path, data.get("version", "?"), data.get("is_score", "?"))
                    return p, data
            except Exception as e:
                logger.warning("Winner load failed %s: %s", path, e)
    logger.warning("No winner.json found; using fallback.")
    return reconstitute_params(FALLBACK_PARAMS, max_xover_long=180, max_exit_xover_long=150), {}


# ==============================================================================
# UNIVERSE
# ==============================================================================

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
    filtered = [s.upper().replace("-USD", "").strip() for s in symbols if not is_noise_or_wrapper_coin(s)]
    seen, ordered = set(), []
    for s in filtered:
        if s not in seen:
            seen.add(s); ordered.append(s)
    return ordered


def load_universe_constituents() -> List[str]:
    cache_path = DATA_DIR / f"{UNIVERSE_NAME.lower()}_top{TOP_N_COINS}_list.json"
    if cache_path.exists():
        try:
            vals = filter_crypto_universe(json.loads(cache_path.read_text(encoding="utf-8")))
            if len(vals) >= min(15, TOP_N_COINS):
                return vals[:TOP_N_COINS]
        except Exception:
            pass

    candidates: List[str] = []
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
            candidates = filter_crypto_universe([x["symbol"] for x in raw])
    except Exception as e:
        logger.warning("CoinGecko discovery unavailable: %s", e)

    if len(candidates) < TOP_N_COINS:
        try:
            r = _HTTP_SESSION.get("https://api.binance.com/api/v3/ticker/24hr", timeout=8)
            if r.status_code == 200:
                pairs = []
                for item in r.json():
                    s = item.get("symbol", "")
                    if s.endswith("USDT"):
                        pairs.append((s[:-4], float(item.get("quoteVolume", 0.0))))
                pairs.sort(key=lambda x: x[1], reverse=True)
                candidates.extend(filter_crypto_universe([x[0] for x in pairs]))
        except Exception as e:
            logger.warning("Binance volume fallback failed: %s", e)
    candidates.extend(filter_crypto_universe(LEGACY_WHITELIST))
    chosen = filter_crypto_universe(candidates)[:TOP_N_COINS]
    cache_path.write_text(json.dumps(chosen), encoding="utf-8")
    logger.info("Universe: %d coins.", len(chosen))
    return chosen


# ==============================================================================
# DATA
# ==============================================================================

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    # Exact V10.2 data cutoff: exclude the current UTC calendar day (possibly partial).
    today_utc = pd.Timestamp.now("UTC").tz_localize(None).floor("D")
    df = df[df["date"] < today_utc].dropna(subset=["close"])
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[
        (df["close"] > 0) & (df["open"] > 0) & (df["high"] >= df["low"]) &
        ((df["high"] / np.maximum(1e-8, df["low"])) < 50.0)
    ]
    return df.set_index("date")


def fetch_from_binance(symbol: str) -> Optional[pd.DataFrame]:
    hosts = ["https://data-api.binance.vision", "https://api.binance.com"]
    start_ts = int(pd.Timestamp(f"{START_YEAR}-01-01", tz="UTC").timestamp() * 1000)
    now_ts = int(pd.Timestamp.now("UTC").floor("D").timestamp() * 1000)
    for host in hosts:
        curr = start_ts; rows = []; ok = True
        while curr < now_ts:
            try:
                r = _HTTP_SESSION.get(
                    f"{host}/api/v3/klines",
                    params={"symbol": f"{symbol}USDT", "interval": "1d", "startTime": curr, "limit": 1000},
                    timeout=6,
                )
                if r.status_code != 200:
                    ok = False; break
                data = r.json()
                if not data or not isinstance(data, list):
                    break
                rows.extend(data)
                if len(data) < 1000:
                    break
                curr = data[-1][0] + 86400000
                time.sleep(0.02)
            except Exception:
                ok = False; break
        if ok and len(rows) >= MIN_HISTORY_DAYS:
            frame = pd.DataFrame(
                [[d[0], d[1], d[2], d[3], d[4], d[5], d[7]] for d in rows],
                columns=["date", "open", "high", "low", "close", "volume", "quote_volume"],
            )
            frame["date"] = pd.to_datetime(frame["date"], unit="ms", utc=True)
            return _clean_df(frame)
    return None


def fetch_from_bybit(symbol: str) -> Optional[pd.DataFrame]:
    url = "https://api.bybit.com/v5/market/kline"
    end_ts = int(pd.Timestamp.now("UTC").timestamp() * 1000)
    rows = []
    for _ in range(12):
        try:
            r = _HTTP_SESSION.get(url, params={
                "category": "spot", "symbol": f"{symbol}USDT", "interval": "D", "end": end_ts, "limit": 1000
            }, timeout=6)
            if r.status_code != 200:
                break
            data = r.json().get("result", {}).get("list", [])
            if not data:
                break
            rows.extend(data)
            if len(data) < 1000:
                break
            end_ts = int(data[-1][0]) - 1
            time.sleep(0.02)
        except Exception:
            break
    if len(rows) >= MIN_HISTORY_DAYS:
        frame = pd.DataFrame(
            [[int(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4]), float(d[5]), float(d[6])] for d in rows],
            columns=["date", "open", "high", "low", "close", "volume", "quote_volume"],
        )
        frame["date"] = pd.to_datetime(frame["date"], unit="ms", utc=True)
        return _clean_df(frame)
    return None


def fetch_from_okx(symbol: str) -> Optional[pd.DataFrame]:
    url = "https://www.okx.com/api/v5/market/history-candles"
    rows, after = [], ""
    for _ in range(30):
        params = {"instId": f"{symbol}-USDT", "bar": "1Dutc", "limit": 100}
        if after:
            params["after"] = after
        try:
            r = _HTTP_SESSION.get(url, params=params, timeout=6)
            if r.status_code != 200:
                break
            data = r.json().get("data", [])
            if not data:
                break
            rows.extend(data)
            after = data[-1][0]
            if len(data) < 100:
                break
            time.sleep(0.02)
        except Exception:
            break
    if len(rows) >= MIN_HISTORY_DAYS:
        frame = pd.DataFrame(
            [[int(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4]), float(d[5]), float(d[6])] for d in rows],
            columns=["date", "open", "high", "low", "close", "volume", "quote_volume"],
        )
        frame["date"] = pd.to_datetime(frame["date"], unit="ms", utc=True)
        return _clean_df(frame)
    return None


def fetch_from_yahoo(symbol: str) -> Optional[pd.DataFrame]:
    start_ts = int(pd.Timestamp(f"{START_YEAR}-01-01", tz="UTC").timestamp())
    end_ts = int(pd.Timestamp.now("UTC").timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}-USD?period1={start_ts}&period2={end_ts}&interval=1d"
    try:
        r = _HTTP_SESSION.get(url, timeout=8)
        if r.status_code != 200:
            return None
        res = r.json()["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        frame = pd.DataFrame({
            "date": pd.to_datetime(res["timestamp"], unit="s", utc=True),
            "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
            "volume": q["volume"],
            "quote_volume": [c * v if c and v else 0.0 for c, v in zip(q["close"], q["volume"])],
        })
        cleaned = _clean_df(frame)
        return cleaned if len(cleaned) >= MIN_HISTORY_DAYS else None
    except Exception:
        return None


def fetch_coin(coin: str) -> Optional[pd.DataFrame]:
    cache_file = DATA_DIR / f"{coin}_1d_from{START_YEAR}.parquet"
    meta_file = DATA_DIR / f"{coin}_1d_from{START_YEAR}.meta.json"
    if cache_file.exists() and meta_file.exists():
        try:
            age_h = (time.time() - cache_file.stat().st_mtime) / 3600.0
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if age_h < 48.0 and meta.get("start_year") == START_YEAR:
                df = pd.read_parquet(cache_file)
                if len(df) >= MIN_HISTORY_DAYS and _refresh_cache_if_last_closed_missing(df):
                    return df
        except Exception:
            pass
    for provider, fn in (
        ("binance", lambda: fetch_from_binance(coin)),
        ("bybit", lambda: fetch_from_bybit(coin)),
        ("okx", lambda: fetch_from_okx(coin)),
        ("yahoo", lambda: fetch_from_yahoo(coin)),
    ):
        try:
            df = fn()
            if df is not None and len(df) >= MIN_HISTORY_DAYS:
                df.to_parquet(cache_file)
                meta_file.write_text(json.dumps({
                    "provider": provider, "timestamp": time.time(), "bars": len(df), "start_year": START_YEAR
                }), encoding="utf-8")
                return df
        except Exception:
            continue
    return None


def build_market(universe_coins: List[str], state_probe: Dict[str, Any]) -> Tuple[pd.DatetimeIndex, Dict[str, Dict[str, Any]]]:
    wanted = set(universe_coins) | set(state_probe.get("positions", {}).keys())
    wl = state_probe.get("watchlist", [])
    if isinstance(wl, dict):
        wanted |= set(wl.keys())
    else:
        wanted |= {w.get("coin") for w in wl if isinstance(w, dict)}
    wanted.add("BTC")

    raw: Dict[str, pd.DataFrame] = {}
    for coin in sorted(x for x in wanted if x):
        df = fetch_coin(coin)
        if df is not None and len(df) >= MIN_HISTORY_DAYS:
            raw[coin] = df
    if "BTC" not in raw:
        raise RuntimeError("Failed to load BTC benchmark data.")

    btc_dates = pd.DatetimeIndex(sorted(raw["BTC"].index.unique())).normalize()
    market: Dict[str, Dict[str, Any]] = {}
    for coin, df in raw.items():
        re = df.reindex(btc_dates)
        raw_close = re["close"].copy()
        is_missing = raw_close.isna()
        has_ever_traded = (~is_missing).cumsum() > 0
        trailing_missing_count = int((is_missing[::-1].cumprod()[::-1]).astype(int).sum())
        is_delisted = (
            is_missing[::-1].cumprod()[::-1].astype(bool)
            & has_ever_traded & (trailing_missing_count >= 20)
        ).to_numpy(bool)
        alive = (~is_missing).to_numpy(bool)

        re["close"] = re["close"].ffill()
        re["open"] = re["open"].ffill()
        re["high"] = re["high"].ffill()
        re["low"] = re["low"].ffill()
        re["volume"] = re["volume"].fillna(0.0)
        re["quote_volume"] = re["quote_volume"].fillna(0.0)

        o = re["open"].to_numpy(float); h = re["high"].to_numpy(float)
        l = re["low"].to_numpy(float); c = re["close"].to_numpy(float)
        v = re["volume"].to_numpy(float); qv = re["quote_volume"].to_numpy(float)
        dvol = np.nan_to_num(np_rolling_mean(qv, 30), nan=LIQUIDITY_FLOOR_USD)

        market[coin] = {
            "df": re, "alive": alive, "delist": is_delisted,
            "open": o, "high": h, "low": l, "close": c, "volume": v,
            "quote_volume": qv, "dvol": dvol,
        }
    return btc_dates, market


def compile_all_signals(dates, market, p):
    btc = market["BTC"]["close"]
    if p.get("use_market_macro_system", False):
        mma = FastIndicators.moving_average(btc, p["macro_ma_len"], p["macro_ma_type"])
        macro_ok = (~np.isnan(mma)) & (btc > mma)
    else:
        macro_ok = np.ones(len(dates), dtype=bool)

    for coin, m in market.items():
        m["sig"] = compile_single_symbol(
            m["open"], m["high"], m["low"], m["close"], m["volume"], m["dvol"],
            m["alive"], macro_ok, p, LIQUIDITY_FLOOR_USD,
        )
    return macro_ok




def _expected_last_closed_utc_date() -> pd.Timestamp:
    return pd.Timestamp.now("UTC").tz_localize(None).floor("D") - pd.Timedelta(days=1)


def _refresh_cache_if_last_closed_missing(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    return pd.Timestamp(df.index.max()).normalize() >= _expected_last_closed_utc_date()


def _parse_binance_price_payload(payload: Any) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if isinstance(payload, list):
        for row in payload:
            try:
                result[str(row["symbol"]).upper()] = float(row["price"])
            except Exception:
                continue
    return result


def fetch_live_crypto_prices(coins: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch current spot prices. This is market-data only; no order API is used."""
    wanted = sorted(set(c.upper() for c in coins if c))
    result: Dict[str, Dict[str, Any]] = {}
    now = datetime.now(IST)
    missing = list(wanted)

    # Binance supports a batched current-price endpoint.
    for chunk_start in range(0, len(wanted), 50):
        chunk = wanted[chunk_start:chunk_start + 50]
        try:
            symbols = json.dumps([f"{c}USDT" for c in chunk], separators=(",", ":"))
            r = _HTTP_SESSION.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbols": symbols}, timeout=8,
            )
            if r.status_code == 200:
                prices = _parse_binance_price_payload(r.json())
                for c in chunk:
                    sym = f"{c}USDT"
                    if sym in prices and np.isfinite(prices[sym]) and prices[sym] > 0:
                        result[c] = {"price": prices[sym], "provider": "binance", "quote_time": now.isoformat(), "age_sec": 0.0}
        except Exception as e:
            logger.warning("Binance live-price batch failed: %s", e)

    missing = [c for c in wanted if c not in result]
    for c in missing:
        # Yahoo fallback is intentionally per-symbol and only used when Binance has no quote.
        try:
            ts = int(time.time())
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{c}-USD?period1={ts-86400}&period2={ts}&interval=1m"
            r = _HTTP_SESSION.get(url, timeout=8)
            if r.status_code == 200:
                res = r.json()["chart"]["result"][0]
                vals = [x for x in res["indicators"]["quote"][0].get("close", []) if x is not None]
                stamps = [x for x in res.get("timestamp", []) if x is not None]
                if vals:
                    stamp = pd.to_datetime(stamps[-1], unit="s", utc=True) if stamps else pd.Timestamp.now("UTC")
                    age = max(0.0, (pd.Timestamp.now("UTC") - stamp).total_seconds())
                    result[c] = {"price": float(vals[-1]), "provider": "yahoo", "quote_time": stamp.isoformat(), "age_sec": age}
        except Exception:
            pass
    return result


# ==============================================================================
# STATE
# ==============================================================================

def _blank_state():
    # Strategy cash is authoritative in USD, matching the V10.2 backtest.
    # cash_inr is only a display/legacy snapshot derived from current FX.
    return {
        "cash_usd": 0.0,
        "cash_inr": 0.0,
        "positions": {},
        "watchlist": [],
        "last_processed_date": None,
    }


def _normalize_position(coin: str, pos: Dict[str, Any], dates: pd.DatetimeIndex, fx: float) -> Dict[str, Any]:
    p = dict(pos)
    price_usd = float(p.get("entry_price_usd", 0.0))
    price_inr = float(p.get("entry_price_inr", price_usd * fx))
    cost_inr = float(p.get("cost_inr", p.get("invested_amount", 0.0)))
    units = float(p.get("units", 0.0))
    if units <= 0 and price_inr > 0 and cost_inr > 0:
        units = cost_inr / price_inr
    cost_usd = float(p.get("cost_usd", units * price_usd))
    entry_date = str(p.get("entry_date", dates[0].date()))[:10]
    matches = np.where(dates.strftime("%Y-%m-%d") == entry_date)[0]
    entry_bar = int(p.get("entry_bar", matches[0] if len(matches) else 0))
    fees = p.get("fee_acc", _fee_dict())
    return {
        "coin": coin, "entry_bar": entry_bar, "entry_date": entry_date,
        "entry_price_usd": price_usd, "entry_price_inr": price_inr,
        "initial_units": float(p.get("initial_units", units)), "units": units,
        "cost_usd": cost_usd, "cost_inr": cost_inr if cost_inr > 0 else cost_usd * fx,
        "entry_atr": float(p.get("entry_atr", max(price_usd * 0.02, 1e-8))),
        "current_sl_usd": float(p.get("current_sl_usd", p.get("stop_loss_usd", price_usd * 0.9))),
        "stop_reason": p.get("stop_reason", "STOP_LOSS"),
        "highest_high_usd": float(p.get("highest_high_usd", price_usd)),
        "lowest_low_usd": float(p.get("lowest_low_usd", price_usd)),
        "peak_bar": int(p.get("peak_bar", entry_bar)), "trough_bar": int(p.get("trough_bar", entry_bar)),
        "layer": int(p.get("layer", 1)), "tp_done": bool(p.get("tp_done", False)),
        "tp_proceeds_usd": float(p.get("tp_proceeds_usd", 0.0)),
        "from_watchlist": bool(p.get("from_watchlist", False)),
        "wait_days": int(p.get("wait_days", 0)),
        "proceeds_usd": float(p.get("proceeds_usd", 0.0)),
        "fee_acc": {k: float(v) for k, v in fees.items()},
        "is_manual": bool(p.get("is_manual", False)),
    }


def load_state(dates, fx):
    state = _blank_state()
    if STATE_FILE.exists():
        try:
            old = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            if "cash_usd" in old:
                state["cash_usd"] = float(old.get("cash_usd", 0.0))
            else:
                # One-time migration of the legacy INR cash ledger.
                state["cash_usd"] = float(old.get("cash_inr", 0.0)) / max(fx, 1e-9)
            state["cash_inr"] = state["cash_usd"] * fx
            state["positions"] = {}
            for c, raw_pos in old.get("positions", {}).items():
                pos_list = raw_pos if isinstance(raw_pos, list) else [raw_pos]
                state["positions"][c] = [_normalize_position(c, x, dates, fx) for x in pos_list]
            wl = old.get("watchlist", [])
            if isinstance(wl, dict):
                wl = [{"coin": c, **info} for c, info in wl.items()]
            state["watchlist"] = list(wl)
            state["last_processed_date"] = old.get("last_processed_date")
        except Exception as e:
            logger.warning("State load/migration failed: %s", e)
    return state


def save_state(state):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    tmp.replace(STATE_FILE)


def apply_manual_overrides(state, dates, today_str, fx, p):
    raw = os.environ.get("CRYPTO_MANUAL_OVERRIDES_JSON")
    cfg = json.loads(raw) if raw else {"force_remove_positions": [], "force_remove_watchlist": [],
                                        "force_add_positions": {}, "force_add_watchlist": []}
    for c in cfg.get("force_remove_positions", []):
        state["positions"].pop(c, None)
    remove_wl = set(cfg.get("force_remove_watchlist", []))
    state["watchlist"] = [w for w in state["watchlist"] if w.get("coin") not in remove_wl]
    for coin, info in cfg.get("force_add_positions", {}).items():
        if coin in state["positions"]:
            continue
        price_inr = float(info["entry_price_inr"])
        price_usd = float(info.get("entry_price_usd", price_inr / fx))
        cost_inr = float(info.get("cost_inr", TRANCHE_FLOOR_USD * fx))
        state["positions"][coin] = [_normalize_position(coin, {
            "entry_date": info.get("entry_date", today_str),
            "entry_price_inr": price_inr, "entry_price_usd": price_usd,
            "cost_inr": cost_inr, "cost_usd": float(info.get("cost_usd", cost_inr / fx)),
            "units": float(info.get("units", cost_inr / price_inr)),
            "entry_atr": float(info.get("entry_atr", price_usd * 0.02)),
            "current_sl_usd": float(info.get("stop_loss_usd", price_usd * 0.9)),
            "layer": int(info.get("layer", 1)), "is_manual": True,
        }, dates, fx)]
    adds = cfg.get("force_add_watchlist", [])
    if isinstance(adds, dict):
        adds = [{"coin": c, **info} for c, info in adds.items()]
    for info in adds:
        state["watchlist"].append({
            "coin": info["coin"], "signal_bar_date": info.get("eligible_date", today_str),
            "signal_date": info.get("signal_date", today_str), "trigger_price": float(info.get("trigger_price", 0.0)),
            "shadow_stop": float(info.get("shadow_stop", 0.0)), "highest_high": float(info.get("trigger_price", 0.0)),
            "breakout_quality": float(info.get("breakout_quality", 0.0)),
            "entry_atr": float(info.get("entry_atr", 0.0)), "unfilled_reason": None,
        })


def fetch_fx() -> float:
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/INR=X?range=5d&interval=1d"
        res = _HTTP_SESSION.get(url, timeout=6).json()["chart"]["result"][0]
        rates = [x for x in res["indicators"]["quote"][0]["close"] if x is not None]
        if rates and rates[-1] > 50:
            return float(rates[-1])
    except Exception:
        pass
    return DEFAULT_USD_INR_RATE


# ==============================================================================
# LIVE ENGINE
# ==============================================================================

def _make_event(bucket, event, is_today, actionable, retro):
    (actionable[bucket] if is_today else retro).append(event)



def run_cycle():
    p, meta = load_strategy_params()
    fx = fetch_fx()
    universe = load_universe_constituents()

    probe = _blank_state()
    if STATE_FILE.exists():
        try:
            probe = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    raw_manual = os.environ.get("CRYPTO_MANUAL_OVERRIDES_JSON")
    manual_coins = set()
    if raw_manual:
        try:
            cfg = json.loads(raw_manual)
            manual_coins |= set((cfg.get("force_add_positions") or {}).keys())
            adds = cfg.get("force_add_watchlist") or []
            manual_coins |= set(adds.keys()) if isinstance(adds, dict) else {x.get("coin") for x in adds if isinstance(x, dict) and x.get("coin")}
        except Exception as e:
            logger.warning("Invalid CRYPTO_MANUAL_OVERRIDES_JSON before fetch: %s", e)
    universe = sorted(set(universe) | {c for c in manual_coins if c})

    # Historical dataset deliberately ends at the last COMPLETE UTC daily candle.
    dates, market = build_market(universe, probe)
    if len(dates) < 2:
        raise RuntimeError("Not enough closed daily crypto history.")
    last_closed_idx = len(dates) - 1
    expected_closed = _expected_last_closed_utc_date()
    if dates[last_closed_idx].normalize() < expected_closed:
        logger.warning("Historical cache is behind the last expected closed UTC day (%s < %s).", dates[last_closed_idx].date(), expected_closed.date())
    closed_date_str = str(dates[last_closed_idx].date())

    # Current price is separate from the historical signal dataset.
    live_coins = sorted(market.keys())
    live_prices = fetch_live_crypto_prices(live_coins)
    if "BTC" not in live_prices:
        raise RuntimeError("Could not obtain current BTC spot price. Refusing to generate crypto live signals.")
    for coin, q in live_prices.items():
        if coin in market:
            market[coin]["live_price"] = float(q["price"])
            market[coin]["quote_time"] = q["quote_time"]
            market[coin]["quote_age_sec"] = float(q["age_sec"])
            market[coin]["live_available"] = True
    for coin in market:
        market[coin]["live_available"] = coin in live_prices

    # Warn, but don't silently use an old quote as "current".
    stale = {c: q for c, q in live_prices.items() if q["age_sec"] > LIVE_PRICE_MAX_AGE_SECONDS}
    if stale:
        logger.warning("%d crypto quotes exceed %.0fs freshness threshold; affected names will not generate new actions.", len(stale), LIVE_PRICE_MAX_AGE_SECONDS)

    macro_ok = compile_all_signals(dates, market, p)
    state = load_state(dates, fx)
    if SET_WALLET_CASH_INR is not None:
        state["cash_usd"] = float(SET_WALLET_CASH_INR) / max(fx, 1e-9)
        logger.info("Cash override applied: ₹%.2f ($%.2f)", SET_WALLET_CASH_INR, state["cash_usd"])
    else:
        logger.info("Rolling strategy cash: $%.2f (₹%.2f)", state["cash_usd"], state["cash_usd"] * fx)
    state["cash_inr"] = state["cash_usd"] * fx
    apply_manual_overrides(state, dates, closed_date_str, fx, p)

    for w in state["watchlist"]:
        if "signal_bar" not in w:
            vals = np.where(dates.strftime("%Y-%m-%d") == str(w.get("signal_bar_date", w.get("signal_date", closed_date_str)))[:10])[0]
            w["signal_bar"] = int(vals[0]) if len(vals) else last_closed_idx
        w["coin"] = w.get("coin", w.get("ticker"))
    for c, positions in state["positions"].items():
        for pos in positions:
            if "entry_bar" not in pos:
                vals = np.where(dates.strftime("%Y-%m-%d") == str(pos.get("entry_date", closed_date_str))[:10])[0]
                pos["entry_bar"] = int(vals[0]) if len(vals) else last_closed_idx

    buys, exits, tps, retro = [], [], [], []
    raw_signals = 0
    consumed = set(state.get("consumed_signal_keys", []))
    live_idx = last_closed_idx + 1  # synthetic current execution index; not a candle in history

    def remember(key: str) -> None:
        consumed.add(key)

    # ------------------------------------------------------------------
    # PASS A: positions. Signals are from the last fully closed day;
    # execution/reference price is the CURRENT live spot.
    # ------------------------------------------------------------------
    for coin in list(state["positions"]):
        m = market.get(coin)
        q = live_prices.get(coin)
        if m is None or q is None or float(q["age_sec"]) > LIVE_PRICE_MAX_AGE_SECONDS:
            logger.warning("No fresh live price for held crypto %s; no exit action generated.", coin)
            continue
        sig: SignalArrays = m["sig"]
        live_px = float(q["price"])
        prev_atr = float(sig.atr[last_closed_idx]) if np.isfinite(sig.atr[last_closed_idx]) else 0.0
        adv = max(float(m["dvol"][last_closed_idx]), LIQUIDITY_FLOOR_USD)

        for pos in list(state["positions"][coin]):
            bars_held = max(0, live_idx - int(pos["entry_bar"]))
            macro_bear = bool(p.get("macro_active_exit", False) and last_closed_idx >= 2 and
                              (not macro_ok[last_closed_idx]) and (not macro_ok[last_closed_idx-1]) and (not macro_ok[last_closed_idx-2]))
            strategy_exit = int(p["exit_type"]) in (3, 4, 5, 6, 7) and bool(sig.exit_sig[last_closed_idx])
            eligible_signal_exit = bars_held >= 3 or (pos["highest_high_usd"] - pos["entry_price_usd"]) >= pos["entry_atr"]
            reason = None
            if macro_bear:
                reason = "BTC_MACRO_REGIME_EXIT"
            elif bars_held >= p.get("max_holding_bars", 20):
                reason = "MAX_HOLDING_TIME"
            elif strategy_exit and eligible_signal_exit:
                reason = f"STRATEGY_EXIT_TYPE_{p['exit_type']}"

            tp_price = pos["entry_price_usd"] + p.get("tp_mult", 4.0) * pos["entry_atr"]
            if reason is None and live_px <= pos["current_sl_usd"]:
                reason = pos["stop_reason"]
            if reason is None and p.get("use_global_tp", False) and not pos.get("tp_done", False) and live_px >= tp_price:
                close_units = pos["units"] * (p.get("tp_size_pct", 50.0) / 100.0)
                if close_units > 0:
                    part_rate = min(1.0, max(0.0, (close_units * live_px) / adv))
                    slip = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate)) / 10000.0
                    eff_px, credit, fee = _sell_fill_audited(close_units, live_px, slip)
                    state["cash_usd"] += credit
                    pos["proceeds_usd"] += credit
                    pos["tp_proceeds_usd"] += credit
                    pos["units"] -= close_units
                    pos["tp_done"] = True
                    _merge_fees(pos["fee_acc"], fee)
                    if p.get("tp_move_sl_be", False) and pos["current_sl_usd"] < pos["entry_price_usd"] * 1.002:
                        pos["current_sl_usd"] = pos["entry_price_usd"] * 1.002
                        pos["stop_reason"] = "BREAKEVEN_SL"
                    tps.append({
                        "date": closed_date_str, "coin": coin, "layer": pos["layer"],
                        "action": f"TAKE PROFIT (SELL {p.get('tp_size_pct',50.0):.0f}%)",
                        "price_usd": live_px, "effective_price_usd": eff_px,
                        "price_inr": live_px * fx, "freed_cash_inr": credit * fx,
                    })

            if reason is not None:
                part_rate = min(1.0, max(0.0, (pos["units"] * live_px) / adv))
                slip = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate)) / 10000.0
                eff_px, proceeds, fee = _sell_fill_audited(pos["units"], live_px, slip)
                state["cash_usd"] += proceeds
                pos["proceeds_usd"] += proceeds
                _merge_fees(pos["fee_acc"], fee)
                pnl_usd = pos["proceeds_usd"] - pos["cost_usd"]
                exits.append({
                    "date": closed_date_str, "coin": coin, "layer": pos["layer"],
                    "action": "CLOSE 100%", "reason": reason,
                    "reference_price_usd": live_px, "effective_price_usd": eff_px,
                    "entry_price_usd": pos["entry_price_usd"], "exit_price_usd": live_px,
                    "entry_price_inr": pos["entry_price_usd"] * fx, "exit_price_inr": live_px * fx,
                    "ret_pct": pnl_usd / max(1e-9, pos["cost_usd"]) * 100.0,
                    "freed_cash_inr": proceeds * fx, "pnl_usd": pnl_usd,
                })
                state["positions"][coin].remove(pos)
                continue

            # State update after exit decision uses the completed candle's high/low and ATR;
            # current spot is included as a point observation, not as a partial daily candle.
            h_last = float(m["high"][last_closed_idx])
            l_last = float(m["low"][last_closed_idx])
            if max(live_px, h_last) > pos["highest_high_usd"]:
                pos["highest_high_usd"] = max(live_px, h_last); pos["peak_bar"] = live_idx
            if min(live_px, l_last) < pos["lowest_low_usd"]:
                pos["lowest_low_usd"] = min(live_px, l_last); pos["trough_bar"] = live_idx
            be = p.get("be_trigger_atr", 0.0)
            if be > 0.0 and pos["current_sl_usd"] < pos["entry_price_usd"] * 1.002:
                if pos["highest_high_usd"] >= pos["entry_price_usd"] + be * pos["entry_atr"]:
                    pos["current_sl_usd"] = max(pos["current_sl_usd"], pos["entry_price_usd"] * 1.002)
                    pos["stop_reason"] = "BREAKEVEN_SL"
            if int(p["exit_type"]) == 1:
                floor = pos["highest_high_usd"] * (1.0 - p.get("trail_pct", 10.0) / 100.0)
                if floor > pos["current_sl_usd"]:
                    pos["current_sl_usd"], pos["stop_reason"] = floor, "TRAIL_PCT_STOP"
            elif p.get("trail_atr_mult", 0.0) > 0.0 and np.isfinite(prev_atr):
                floor = pos["highest_high_usd"] - p["trail_atr_mult"] * prev_atr
                if floor > pos["current_sl_usd"]:
                    pos["current_sl_usd"], pos["stop_reason"] = floor, "TRAIL_ATR_STOP"

    state["positions"] = {c: xs for c, xs in state["positions"].items() if xs}

    # ------------------------------------------------------------------
    # PASS B: revalidate old watchlist, then add the LAST CLOSED daily signal.
    # ------------------------------------------------------------------
    wl_mode = p.get("wl_mode", "WL_NONE")
    max_pyramid = int(p.get("max_pyramid_layers", 1))
    survivors = []
    for item in state["watchlist"]:
        coin = item.get("coin"); m = market.get(coin); q = live_prices.get(coin)
        if m is None or q is None:
            if wl_mode != "WL_NONE": survivors.append(item)
            continue
        sig = m["sig"]; live_px = float(q["price"])
        if live_px <= float(item.get("shadow_stop", -np.inf)):
            continue
        if int(p["exit_type"]) in (3,4,5,6,7) and bool(sig.exit_sig[last_closed_idx]):
            continue
        age = max(0, last_closed_idx - int(item.get("signal_bar", last_closed_idx)))
        if age >= WL_MAX_AGE_BARS:
            continue
        h_last = float(m["high"][last_closed_idx])
        if h_last > item.get("highest_high", h_last):
            item["highest_high"] = h_last
        atr_last = float(sig.atr[last_closed_idx]) if np.isfinite(sig.atr[last_closed_idx]) else float(item.get("entry_atr", 0.0))
        if p.get("trail_atr_mult", 0.0) > 0.0 and np.isfinite(atr_last):
            item["shadow_stop"] = max(float(item["shadow_stop"]), float(item["highest_high"]) - p["trail_atr_mult"] * atr_last)
        item["age_bars"] = age
        survivors.append(item)
    state["watchlist"] = survivors

    for coin, m in market.items():
        q = live_prices.get(coin)
        if q is None:
            continue
        sig = m["sig"]
        if not bool(sig.raw_signal[last_closed_idx]):
            continue
        raw_signals += 1
        if not bool(sig.entry[last_closed_idx]):
            continue
        if not bool(macro_ok[last_closed_idx]):
            continue
        if len(state["positions"].get(coin, [])) >= max_pyramid:
            continue
        key = f"{coin}|{closed_date_str}|{int(p.get('entry_type',0))}|{int(p.get('exit_type',0))}"
        if key in consumed:
            continue
        atr_last = float(sig.atr[last_closed_idx]) if np.isfinite(sig.atr[last_closed_idx]) else 0.0
        state["watchlist"].append({
            "coin": coin, "signal_bar": last_closed_idx, "signal_bar_date": closed_date_str,
            "signal_date": closed_date_str, "trigger_price": float(m["close"][last_closed_idx]),
            "shadow_stop": float(q["price"] - p.get("sl_mult", 3.0) * atr_last),
            "highest_high": float(m["high"][last_closed_idx]),
            "breakout_quality": (float(m["close"][last_closed_idx]) - float(m["open"][last_closed_idx])) / max(1e-6, atr_last),
            "entry_atr": atr_last, "unfilled_reason": None, "signal_key": key,
        })

    def wl_score(item):
        coin = item["coin"]; px = float(live_prices[coin]["price"])
        if wl_mode == "WL_DEEPEST_DISCOUNT":
            return (item["trigger_price"] - px) / max(1e-6, item["trigger_price"])
        if wl_mode == "WL_STRONGEST_MOMENTUM":
            return float(item.get("breakout_quality", 0.0))
        return -float(item.get("age_bars", 0))
    state["watchlist"].sort(key=wl_score, reverse=True)

    # ------------------------------------------------------------------
    # PASS C: fill now at the CURRENT spot price.
    # ------------------------------------------------------------------
    max_slots = int(p.get("max_concurrent_tranches", DEFAULT_MAX_CONCURRENT_TRANCHES))
    unfilled = []
    for item in state["watchlist"]:
        coin = item["coin"]; m = market.get(coin); q = live_prices.get(coin)
        if m is None or q is None or q["age_sec"] > LIVE_PRICE_MAX_AGE_SECONDS:
            if wl_mode != "WL_NONE": unfilled.append(item)
            continue
        if sum(len(v) for v in state["positions"].values()) >= max_slots:
            if wl_mode != "WL_NONE": item["unfilled_reason"] = "slot_saturated"; unfilled.append(item)
            continue
        coin_layers = len(state["positions"].get(coin, []))
        if coin_layers >= max_pyramid:
            if wl_mode != "WL_NONE": item["unfilled_reason"] = "pyramid_blocked"; unfilled.append(item)
            continue
        live_px = float(q["price"])
        if coin_layers:
            highest_entry = max(x["entry_price_usd"] for x in state["positions"][coin])
            if live_px <= highest_entry * 1.005:
                if wl_mode != "WL_NONE": item["unfilled_reason"] = "anti_averaging_down"; unfilled.append(item)
                continue

        adv = max(float(m["dvol"][last_closed_idx]), LIQUIDITY_FLOOR_USD)
        open_slots = max(1, max_slots - sum(len(v) for v in state["positions"].values()))
        open_active_usd = sum(
            pos["units"] * float(live_prices.get(c, {}).get("price", market[c]["close"][last_closed_idx]))
            for c, layers in state["positions"].items() if c in market
            for pos in layers
        )
        current_equity = state["cash_usd"] + open_active_usd
        max_pos_cap = current_equity * MAX_POSITION_EQUITY_PCT
        dynamic_slot_target = min(max_pos_cap, state["cash_usd"] / float(open_slots))
        current_asset_exposure = sum(x["units"] * live_px for x in state["positions"].get(coin, []))
        remaining_asset_capacity = max(0.0, max_pos_cap - current_asset_exposure)
        if remaining_asset_capacity < TRANCHE_FLOOR_USD:
            if wl_mode != "WL_NONE": item["unfilled_reason"] = "scrip_risk_cap"; unfilled.append(item)
            continue
        liquidity_cap = adv * MAX_ADV_PARTICIPATION
        scaled_ceiling = max(10_000.0, current_equity * 0.35)
        target = min(dynamic_slot_target, scaled_ceiling, liquidity_cap, remaining_asset_capacity)
        tranche = min(compute_max_affordable_tranche(state["cash_usd"], adv), max(TRANCHE_FLOOR_USD, target))
        if tranche < TRANCHE_FLOOR_USD:
            if wl_mode != "WL_NONE": item["unfilled_reason"] = "cash_starved"; unfilled.append(item)
            continue
        part_rate = min(1.0, max(0.0, tranche / adv))
        slip = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate)) / 10000.0
        eff_entry, units, total_cost, fee_buy = _buy_fill_audited(tranche, live_px, slip)
        if state["cash_usd"] + 1e-9 < total_cost or units <= 0:
            if wl_mode != "WL_NONE": item["unfilled_reason"] = "cash_starved"; unfilled.append(item)
            continue
        sl = eff_entry - p.get("sl_mult", 3.0) * item["entry_atr"]
        state["cash_usd"] -= total_cost
        layer = coin_layers + 1
        pos = {
            "coin": coin, "entry_bar": live_idx, "entry_date": str(datetime.now(IST).date()),
            "entry_price_usd": eff_entry, "reference_entry_price_usd": live_px,
            "entry_price_inr": eff_entry * fx, "reference_entry_price_inr": live_px * fx,
            "initial_units": units, "units": units, "cost_usd": total_cost, "cost_inr": total_cost * fx,
            "entry_atr": item["entry_atr"], "current_sl_usd": sl, "stop_reason": "STOP_LOSS",
            "highest_high_usd": max(eff_entry, float(m["high"][last_closed_idx]), live_px),
            "lowest_low_usd": min(eff_entry, float(m["low"][last_closed_idx]), live_px),
            "peak_bar": live_idx, "trough_bar": live_idx, "layer": layer,
            "tp_done": False, "tp_proceeds_usd": 0.0, "from_watchlist": False,
            "wait_days": max(0, last_closed_idx - int(item.get("signal_bar", last_closed_idx))),
            "proceeds_usd": 0.0, "fee_acc": fee_buy, "is_manual": False,
        }
        state["positions"].setdefault(coin, []).append(pos)
        remember(item.get("signal_key", f"{coin}|{closed_date_str}"))
        buys.append({
            "date": closed_date_str, "coin": coin, "layer": layer,
            "action": f"BUY (Layer {layer})", "price_usd": live_px, "effective_price_usd": eff_entry,
            "price_inr": live_px * fx, "allocate_inr": total_cost * fx,
            "signal_date": item.get("signal_date", closed_date_str),
        })

    if wl_mode == "WL_NONE":
        unfilled = []
    state["watchlist"] = unfilled
    state["last_processed_date"] = closed_date_str
    state["last_live_price_time_ist"] = datetime.now(IST).isoformat()
    state["last_live_prices"] = {c: {"price": q["price"], "provider": q["provider"], "quote_time": q["quote_time"], "age_sec": q["age_sec"]} for c, q in live_prices.items()}
    state["consumed_signal_keys"] = sorted(consumed)[-500:]
    state["cash_inr"] = state["cash_usd"] * fx
    save_state(state)

    current_signals = []
    for coin, m in market.items():
        q = live_prices.get(coin)
        if not q:
            continue
        sig = m["sig"]
        if bool(sig.raw_signal[last_closed_idx]):
            current_signals.append({
                "coin": coin, "signal_close_usd": float(m["close"][last_closed_idx]),
                "live_price_usd": float(q["price"]), "adx": float(sig.adx[last_closed_idx]) if np.isfinite(sig.adx[last_closed_idx]) else 0.0,
                "qualifying": bool(sig.entry[last_closed_idx]), "layers": len(state["positions"].get(coin, [])),
                "age_sec": q["age_sec"],
            })

    render_report(meta, p, state, market, dates, last_closed_idx, fx, buys, exits, tps, retro, current_signals, raw_signals, live_prices)


def render_report(meta, p, state, market, idx, fx, buys, exits, tps, retro, current_signals, raw_signals, live_prices):
    market_value_inr = 0.0
    rows = []
    for coin in sorted(state["positions"]):
        q = live_prices.get(coin)
        if not q:
            continue
        px = float(q["price"])
        for pos in state["positions"][coin]:
            rem = pos["units"] * px
            market_value_inr += rem * fx
            total_value = pos.get("proceeds_usd", 0.0) + rem
            pnl_pct = (total_value - pos["cost_usd"]) / max(1e-9, pos["cost_usd"]) * 100.0
            rows.append(
                f"<tr><td>{html.escape(coin)} L{pos['layer']}</td><td>{pos['entry_date']}</td>"
                f"<td>₹{pos.get('reference_entry_price_inr',pos['entry_price_usd']*fx):,.4f}</td><td>{pos['units']:.8f}</td>"
                f"<td>₹{px*fx:,.4f}</td><td>{pnl_pct:+.2f}%</td><td>₹{pos['current_sl_usd']*fx:,.4f}</td></tr>"
            )
    cash_inr = state["cash_usd"] * fx
    equity = cash_inr + market_value_inr
    buys_html = "".join(
        f"<div class='buy'>BUY — <b>{b['coin']}</b> L{b['layer']} @ LIVE SPOT <b>${b['price_usd']:,.6f}</b> / ₹{b['price_inr']:,.4f} "
        f"(estimated effective ${b['effective_price_usd']:,.6f}; allocate ₹{b['allocate_inr']:,.2f}; signal {b.get('signal_date','--')})</div>" for b in buys
    ) or "<div class='muted'>No new live buy actions.</div>"
    exits_html = "".join(
        f"<div class='sell'>{html.escape(e['coin'])} L{e['layer']} — {html.escape(e['reason'])}, EXIT LIVE SPOT <b>${e['exit_price_usd']:,.6f}</b> / ₹{e['exit_price_inr']:,.4f} "
        f"(estimated effective ${e['effective_price_usd']:,.6f}), P&L {e.get('ret_pct',0):+.2f}%</div>" for e in exits
    ) or "<div class='muted'>No live exit actions.</div>"
    tp_html = "".join(
        f"<div class='tp'>{html.escape(t['coin'])} L{t['layer']} — partial TP at live spot ${t['price_usd']:,.6f} / ₹{t['price_inr']:,.4f}, credited ₹{t['freed_cash_inr']:,.2f}</div>" for t in tps
    )
    sig_rows = "".join(
        f"<tr><td>{html.escape(x['coin'])}</td><td>₹{x['signal_close_usd']*fx:,.4f}</td><td>₹{x['live_price_usd']*fx:,.4f}</td>"
        f"<td>{x['adx']:.2f}</td><td>{'YES' if x['qualifying'] else 'NO'}</td><td>{x['age_sec']:.0f}s</td></tr>" for x in current_signals
    ) or "<tr><td colspan='6' class='muted'>No entry trigger on the last fully closed daily candle.</td></tr>"
    wl_rows = []
    for w in state["watchlist"][:30]:
        coin = w["coin"]; px = live_prices.get(coin, {}).get("price", w.get("trigger_price", 0.0))
        pullback = (w.get("trigger_price",0.0) - px) / max(1e-9,w.get("trigger_price",0.0)) * 100.0
        age = max(0, idx - int(w.get("signal_bar",idx)))
        wl_rows.append(
            f"<tr><td>{html.escape(coin)}</td><td>{w.get('signal_date','--')}</td><td>₹{w.get('trigger_price',0)*fx:,.4f}</td><td>₹{px*fx:,.4f}</td>"
            f"<td>{pullback:+.2f}%</td><td>{age}</td><td>₹{w.get('shadow_stop',0)*fx:,.4f}</td></tr>"
        )
    body = f"""<!doctype html><html><head><style>
    body{{font-family:Arial,sans-serif;background:#0f172a;color:#f8fafc;padding:20px}} .card{{max-width:1020px;margin:auto;background:#1e293b;border:1px solid #334155;border-radius:12px;padding:22px}}
    .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}} .stat{{background:#0f172a;padding:12px;border-radius:8px}} .buy,.sell,.tp{{padding:9px;margin:6px 0;border-radius:6px}} .buy{{background:#064e3b}} .sell{{background:#450a0a}} .tp{{background:#1e3a8a}} .muted{{color:#94a3b8}}
    table{{width:100%;border-collapse:collapse;font-size:12px}} th,td{{padding:7px;border-bottom:1px solid #334155;text-align:right}} th:first-child,td:first-child{{text-align:left}}
    </style></head><body><div class='card'>
    <h2>🪙 Crypto V7.4 Live-Price Companion</h2>
    <div class='muted'>V10.2 signal kernel | signal source: last fully closed UTC daily candle ({str(_expected_last_closed_utc_date().date())}) | execution/reference: current spot | winner {html.escape(str(meta.get('version','fallback')))} | USD/INR {fx:.2f}</div>
    <div class='grid'><div class='stat'><b>Cash</b><br>₹{cash_inr:,.2f}</div><div class='stat'><b>Open Market</b><br>₹{market_value_inr:,.2f}</div><div class='stat'><b>Portfolio</b><br>₹{equity:,.2f}</div><div class='stat'><b>Open Layers</b><br>{sum(len(v) for v in state['positions'].values())}</div></div>
    <h3>Today's live actions</h3>{buys_html}{exits_html}{tp_html}
    <h3>Open positions</h3><table><tr><th>Coin</th><th>Entry</th><th>Entry Spot</th><th>Units</th><th>Current Spot</th><th>P&L</th><th>Stop</th></tr>{''.join(rows) or "<tr><td colspan='7' class='muted'>No open positions.</td></tr>"}</table>
    <h3>Last-closed-candle signals vs current spot</h3><table><tr><th>Coin</th><th>Signal Close ₹</th><th>Current ₹</th><th>ADX</th><th>Qualifies</th><th>Quote Age</th></tr>{sig_rows}</table>
    <h3>Watchlist ({len(state['watchlist'])})</h3><table><tr><th>Coin</th><th>Signal</th><th>Trigger ₹</th><th>Current ₹</th><th>Pullback</th><th>Age</th><th>Shadow Stop</th></tr>{''.join(wl_rows) or "<tr><td colspan='7' class='muted'>Watchlist empty.</td></tr>"}</table>
    <div class='muted'>Current-price quotes: {len(live_prices)}/{len(market)}. Freshness limit: {LIVE_PRICE_MAX_AGE_SECONDS:.0f}s. Raw entry triggers on the last closed candle: {raw_signals}. The current partial UTC candle is never fed into the signal indicators. No exchange order API is called.</div>
    </div></body></html>"""
    send_email(f"🪙 Crypto V7.4 LIVE {str(_expected_last_closed_utc_date().date())} [{len(buys)} buys | {len(exits)} exits | ₹{equity:,.0f}]", body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        os.environ["DRY_RUN"] = "true"
    try:
        run_cycle()
    except Exception:
        tb = traceback.format_exc()
        logger.error("Fatal crypto live companion crash:\n%s", tb)
        send_email(
            f"⚠️ Crypto V10.2-Parity Bot CRASHED — {datetime.now(IST):%Y-%m-%d}",
            f"<pre>{html.escape(tb)}</pre>",
        )


if __name__ == "__main__":
    main()
