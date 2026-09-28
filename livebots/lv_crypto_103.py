%%writefile crypto_live_companion.py
#!/usr/bin/env python3
"""
CRYPTO LIVE-PRICE SWING-TRADING COMPANION
Hardcoded UnifiedSwingBot Colab Drive Paths with Multi-Source Spot Fallback
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
# HARDCODED UNIFIED PATHS
# ==============================================================================
BASE_DIR = Path("/content/drive/MyDrive/UnifiedSwingBot")
DATA_DIR = BASE_DIR / "crypto" / "data_cache"
STATE_FILE = BASE_DIR / "crypto" / "state.json"
WINNER_PATH = BASE_DIR / "winners" / "CRYPTO" / "winner.json"
LOG_FILE = BASE_DIR / "crypto" / "crypto_live.log"

DATA_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE_NAME = "TOP_CRYPTO_LIQUID"
TOP_N_COINS = 50
START_YEAR = 2017
MIN_HISTORY_DAYS = 250
IST = ZoneInfo("Asia/Kolkata")
MACRO_INDEX_TICKER = "BTC"
SET_WALLET_CASH_INR: Optional[float] = None
DEFAULT_USD_INR_RATE = 95.4
LIQUIDITY_FLOOR_USD = 2_000_000.0
TRANCHE_FLOOR_USD = 50.0
DEFAULT_MAX_CONCURRENT_TRANCHES = 6
MAX_POSITION_EQUITY_PCT = 0.25
MAX_ADV_PARTICIPATION = 0.015
WL_MAX_AGE_BARS = 15

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
LIQUID_STAKING_PREFIXES = ("ST", "WST", "R", "CB", "ANKR", "MSOL", "SAVAX", "BNSOL", "JITOSOL", "OSOL")
PROTECTED_TICKERS = {
    "STX", "STORJ", "ROSE", "RENDER", "RUNE", "RVN", "RAD", "REQ", "RLC",
    "RARE", "RAY", "RON", "RDNT", "RPL", "STEEM", "STRAX", "STRK"
}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR", "2L", "2S", "3L", "3S", "4L", "4S", "5L", "5S")

logger = logging.getLogger("crypto_live")
logger.setLevel(logging.INFO)
if not logger.handlers:
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(sh)
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(fh)

_HTTP_SESSION = requests.Session()
_HTTP_SESSION.headers.update({"User-Agent": "Mozilla/5.0"})


def send_email(subject: str, body: str) -> None:
    user = os.environ.get("GMAIL_USER")
    pw = os.environ.get("GMAIL_APP_PASSWORD")
    to = os.environ.get("RECIPIENT_EMAIL")
    if not user or not pw or not to:
        logger.warning("Missing Gmail credentials in environment; email skipped.")
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
    if WINNER_PATH.exists():
        try:
            data = json.loads(WINNER_PATH.read_text(encoding="utf-8"))
            p = data.get("best_params") or data.get("params")
            if p:
                p = reconstitute_params(p, max_xover_long=180, max_exit_xover_long=150)
                logger.info("Loaded CRYPTO champion from %s", WINNER_PATH)
                return p, data
        except Exception as e:
            logger.warning("Winner load failed %s: %s", WINNER_PATH, e)
    logger.warning("No CRYPTO winner.json found at %s; using fallback.", WINNER_PATH)
    return reconstitute_params(FALLBACK_PARAMS, max_xover_long=180, max_exit_xover_long=150), {}


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
    cache_path = DATA_DIR / "universe_top50_list.json"
    if cache_path.exists():
        try:
            vals = filter_crypto_universe(json.loads(cache_path.read_text(encoding="utf-8")))
            if len(vals) >= min(15, TOP_N_COINS):
                return vals[:TOP_N_COINS]
        except Exception:
            pass

    candidates: List[str] = []
    try:
        r = _HTTP_SESSION.get("https://api.bybit.com/v5/market/tickers?category=spot", timeout=8)
        if r.status_code == 200:
            data = r.json().get("result", {}).get("list", [])
            for item in data:
                s = item.get("symbol", "")
                if s.endswith("USDT"):
                    candidates.append(s[:-4])
    except Exception as e:
        logger.warning("Bybit universe discovery failed: %s", e)

    if not candidates:
        try:
            url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
                candidates = [x["symbol"].upper() for x in raw]
        except Exception:
            pass

    candidates.extend(LEGACY_WHITELIST)
    chosen = filter_crypto_universe(candidates)[:TOP_N_COINS]
    cache_path.write_text(json.dumps(chosen), encoding="utf-8")
    return chosen


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    today_utc = pd.Timestamp.now("UTC").tz_localize(None).floor("D")
    df = df[df["date"] < today_utc].dropna(subset=["close"])
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[
        (df["close"] > 0) & (df["open"] > 0) & (df["high"] >= df["low"]) &
        ((df["high"] / np.maximum(1e-8, df["low"])) < 50.0)
    ]
    return df.set_index("date")


def fetch_from_bybit_kline(symbol: str) -> Optional[pd.DataFrame]:
    url = "https://api.bybit.com/v5/market/kline"
    end_ts = int(pd.Timestamp.now("UTC").timestamp() * 1000)
    rows = []
    for _ in range(12):
        try:
            r = _HTTP_SESSION.get(
                url,
                params={"category": "spot", "symbol": f"{symbol}USDT", "interval": "D", "end": end_ts, "limit": 1000},
                timeout=6,
            )
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


def fetch_coin(coin: str) -> Optional[pd.DataFrame]:
    cache_file = DATA_DIR / f"{coin}_1d.parquet"
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            expected = pd.Timestamp.now("UTC").tz_localize(None).floor("D") - pd.Timedelta(days=1)
            if len(df) >= MIN_HISTORY_DAYS and pd.Timestamp(df.index.max()).normalize() >= expected:
                return df
        except Exception:
            pass

    df = fetch_from_bybit_kline(coin)
    if df is not None and len(df) >= MIN_HISTORY_DAYS:
        df.to_parquet(cache_file)
        return df
    return None


def build_market(universe_coins: List[str], state_probe: Dict[str, Any]) -> Tuple[pd.DatetimeIndex, Dict[str, Dict[str, Any]]]:
    wanted = set(universe_coins) | set(state_probe.get("positions", {}).keys())
    wl = state_probe.get("watchlist", [])
    wanted |= {w.get("coin") for w in wl if isinstance(w, dict) and w.get("coin")}
    wanted.add("BTC")

    raw: Dict[str, pd.DataFrame] = {}
    for coin in sorted(x for x in wanted if x):
        df = fetch_coin(coin)
        if df is not None and len(df) >= MIN_HISTORY_DAYS:
            raw[coin] = df
    if "BTC" not in raw:
        raise RuntimeError("Failed to load BTC historical benchmark data.")

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


def fetch_live_crypto_prices(coins: List[str]) -> Dict[str, Dict[str, Any]]:
    wanted = set(c.upper() for c in coins if c)
    result: Dict[str, Dict[str, Any]] = {}
    now = datetime.now(IST)

    try:
        r = _HTTP_SESSION.get("https://api.bybit.com/v5/market/tickers?category=spot", timeout=8)
        if r.status_code == 200:
            for item in r.json().get("result", {}).get("list", []):
                sym = item.get("symbol", "").upper()
                if sym.endswith("USDT"):
                    c = sym[:-4]
                    if c in wanted:
                        px = float(item.get("lastPrice", 0.0))
                        if px > 0:
                            result[c] = {"price": px, "provider": "bybit", "quote_time": now.isoformat(), "age_sec": 0.0}
    except Exception as e:
        logger.warning("Bybit spot fetch error: %s", e)

    missing = [c for c in wanted if c not in result]
    if missing:
        try:
            r = _HTTP_SESSION.get("https://api.binance.com/api/v3/ticker/price", timeout=6)
            if r.status_code == 200:
                for row in r.json():
                    sym = str(row.get("symbol", "")).upper()
                    if sym.endswith("USDT"):
                        c = sym[:-4]
                        if c in missing:
                            px = float(row.get("price", 0.0))
                            if px > 0:
                                result[c] = {"price": px, "provider": "binance", "quote_time": now.isoformat(), "age_sec": 0.0}
        except Exception:
            pass

    if "BTC" not in result:
        try:
            url = "https://query1.finance.yahoo.com/v8/finance/chart/BTC-USD?interval=1m&range=1d"
            res = _HTTP_SESSION.get(url, timeout=6).json()["chart"]["result"][0]
            btc_px = float(res["meta"]["regularMarketPrice"])
            result["BTC"] = {"price": btc_px, "provider": "yahoo", "quote_time": now.isoformat(), "age_sec": 0.0}
        except Exception as e:
            logger.warning("BTC Yahoo spot fallback failed: %s", e)

    return result


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


def _blank_state():
    return {"cash_usd": 0.0, "cash_inr": 0.0, "positions": {}, "watchlist": [], "last_processed_date": None}


def _normalize_position(coin: str, pos: Dict[str, Any], dates: pd.DatetimeIndex, fx: float) -> Dict[str, Any]:
    p = dict(pos)
    price_usd = float(p.get("entry_price_usd", 0.0))
    price_inr = float(p.get("entry_price_inr", price_usd * fx))
    cost_inr = float(p.get("cost_inr", 0.0))
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
        "current_sl_usd": float(p.get("current_sl_usd", price_usd * 0.9)),
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
    }


def load_state(dates, fx):
    state = _blank_state()
    if STATE_FILE.exists():
        try:
            old = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            state["cash_usd"] = float(old.get("cash_usd", float(old.get("cash_inr", 0.0)) / max(fx, 1e-9)))
            state["cash_inr"] = state["cash_usd"] * fx
            state["positions"] = {
                c: [_normalize_position(c, x, dates, fx) for x in (xs if isinstance(xs, list) else [xs])]
                for c, xs in old.get("positions", {}).items()
            }
            wl = old.get("watchlist", [])
            state["watchlist"] = list(wl.values()) if isinstance(wl, dict) else list(wl)
            state["last_processed_date"] = old.get("last_processed_date")
        except Exception as e:
            logger.warning("Crypto state load failed: %s", e)
    return state


def save_state(state):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    tmp.replace(STATE_FILE)


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

    dates, market = build_market(universe, probe)
    last_closed_idx = len(dates) - 1
    closed_date_str = str(dates[last_closed_idx].date())

    live_prices = fetch_live_crypto_prices(list(market.keys()))
    if "BTC" not in live_prices:
        raise RuntimeError("No current BTC price available.")

    for coin, q in live_prices.items():
        if coin in market:
            market[coin]["live_price"] = float(q["price"])
            market[coin]["quote_time"] = q["quote_time"]
            market[coin]["quote_age_sec"] = float(q["age_sec"])
            market[coin]["live_available"] = True

    macro_ok = compile_all_signals(dates, market, p)
    state = load_state(dates, fx)
    if SET_WALLET_CASH_INR is not None:
        state["cash_usd"] = float(SET_WALLET_CASH_INR) / max(fx, 1e-9)
    state["cash_inr"] = state["cash_usd"] * fx

    for w in state["watchlist"]:
        if "signal_bar" not in w:
            vals = np.where(dates.strftime("%Y-%m-%d") == str(w.get("signal_date", closed_date_str))[:10])[0]
            w["signal_bar"] = int(vals[0]) if len(vals) else last_closed_idx
        w["coin"] = w.get("coin", w.get("ticker"))

    for c, positions in state["positions"].items():
        for pos in positions:
            if "entry_bar" not in pos:
                vals = np.where(dates.strftime("%Y-%m-%d") == str(pos.get("entry_date", closed_date_str))[:10])[0]
                pos["entry_bar"] = int(vals[0]) if len(vals) else last_closed_idx

    buys, exits, tps = [], [], []
    signal_events = []
    consumed = set(state.get("consumed_signal_keys", []))
    live_idx = last_closed_idx + 1

    # PASS A: Existing positions
    for coin in list(state["positions"]):
        m = market.get(coin)
        q = live_prices.get(coin)
        if m is None or q is None:
            continue
        sig: SignalArrays = m["sig"]
        live_px = float(q["price"])
        adv = max(float(m["dvol"][last_closed_idx]), LIQUIDITY_FLOOR_USD)
        prev_atr = float(sig.atr[last_closed_idx]) if np.isfinite(sig.atr[last_closed_idx]) else 0.0

        for pos in list(state["positions"][coin]):
            bars_held = max(0, live_idx - int(pos["entry_bar"]))
            strategy_exit = int(p["exit_type"]) in (3, 4, 5, 6, 7) and bool(sig.exit_sig[last_closed_idx])
            eligible_signal_exit = bars_held >= 3 or (pos["highest_high_usd"] - pos["entry_price_usd"]) >= pos["entry_atr"]
            reason = None
            if bars_held >= p.get("max_holding_bars", 20):
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
                    tps.append({
                        "coin": coin, "layer": pos["layer"], "price_usd": live_px,
                        "effective_price_usd": eff_px, "price_inr": live_px * fx, "freed_cash_inr": credit * fx,
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
                    "coin": coin, "layer": pos["layer"], "reason": reason, "exit_price_usd": live_px,
                    "effective_price_usd": eff_px, "ret_pct": pnl_usd / max(1e-9, pos["cost_usd"]) * 100.0,
                    "freed_cash_inr": proceeds * fx,
                })
                state["positions"][coin].remove(pos)
                continue

            h_last = float(m["high"][last_closed_idx])
            l_last = float(m["low"][last_closed_idx])
            if max(live_px, h_last) > pos["highest_high_usd"]:
                pos["highest_high_usd"] = max(live_px, h_last)
            if min(live_px, l_last) < pos["lowest_low_usd"]:
                pos["lowest_low_usd"] = min(live_px, l_last)

            be = p.get("be_trigger_atr", 0.0)
            if be > 0.0 and pos["current_sl_usd"] < pos["entry_price_usd"] * 1.002:
                if pos["highest_high_usd"] >= pos["entry_price_usd"] + be * pos["entry_atr"]:
                    pos["current_sl_usd"] = max(pos["current_sl_usd"], pos["entry_price_usd"] * 1.002)
                    pos["stop_reason"] = "BREAKEVEN_SL"

            if p.get("trail_atr_mult", 0.0) > 0.0 and np.isfinite(prev_atr):
                floor = pos["highest_high_usd"] - p["trail_atr_mult"] * prev_atr
                if floor > pos["current_sl_usd"]:
                    pos["current_sl_usd"], pos["stop_reason"] = floor, "TRAIL_ATR_STOP"

    state["positions"] = {c: xs for c, xs in state["positions"].items() if xs}

    # PASS B: Watchlist & Signals from LAST CLOSED daily bar
    for coin, m in market.items():
        q = live_prices.get(coin)
        if q is None:
            continue
        sig = m["sig"]
        entry_signal = bool(sig.entry[last_closed_idx])
        raw_signal = bool(sig.raw_signal[last_closed_idx])
        macro_pass = bool(macro_ok[last_closed_idx])
        live_px = float(q["price"])
        signal_px = float(m["close"][last_closed_idx])

        if raw_signal and entry_signal and macro_pass:
            key = f"{coin}|{closed_date_str}"
            if key not in consumed:
                atr_last = float(sig.atr[last_closed_idx]) if np.isfinite(sig.atr[last_closed_idx]) else 0.0
                signal_events.append({
                    "coin": coin, "signal_type": "ENTRY", "signal_date": closed_date_str,
                    "signal_price": signal_px, "live_price": live_px, "status": "PENDING",
                    "reason": "Valid entry signal on last closed candle", "signal_key": key,
                })
                state["watchlist"].append({
                    "coin": coin, "signal_bar": last_closed_idx, "signal_date": closed_date_str,
                    "trigger_price": signal_px, "shadow_stop": live_px - p.get("sl_mult", 3.0) * atr_last,
                    "highest_high": float(m["high"][last_closed_idx]), "entry_atr": atr_last, "signal_key": key,
                })
            else:
                signal_events.append({
                    "coin": coin, "signal_type": "ENTRY", "signal_date": closed_date_str,
                    "signal_price": signal_px, "live_price": live_px, "status": "ALREADY CONSUMED",
                    "reason": "Signal key already processed", "signal_key": key,
                })

        if bool(sig.exit_sig[last_closed_idx]):
            signal_events.append({
                "coin": coin, "signal_type": "EXIT", "signal_date": closed_date_str,
                "signal_price": signal_px, "live_price": live_px,
                "status": "POSITION OPEN" if coin in state["positions"] else "NO POSITION",
                "reason": "Strategy exit signal on last closed candle", "signal_key": f"EXIT|{coin}|{closed_date_str}",
            })

    # PASS C: Fill at current live spot (Executed once across all accumulated signals)
    max_slots = int(p.get("max_concurrent_tranches", DEFAULT_MAX_CONCURRENT_TRANCHES))
    max_pyramid = int(p.get("max_pyramid_layers", 1))
    unfilled = []

    for item in state["watchlist"]:
        coin = item["coin"]
        m = market.get(coin)
        q = live_prices.get(coin)
        if m is None or q is None or sum(len(v) for v in state["positions"].values()) >= max_slots:
            unfilled.append(item)
            continue

        coin_layers = len(state["positions"].get(coin, []))
        if coin_layers >= max_pyramid:
            unfilled.append(item)
            continue

        live_px = float(q["price"])
        if coin_layers > 0:
            highest_entry = max(x["entry_price_usd"] for x in state["positions"][coin])
            if live_px <= highest_entry * 1.005:
                unfilled.append(item)
                continue

        adv = max(float(m["dvol"][last_closed_idx]), LIQUIDITY_FLOOR_USD)
        open_slots = max(1, max_slots - sum(len(v) for v in state["positions"].values()))
        open_usd = sum(pos["units"] * float(live_prices.get(c, {}).get("price", 0.0))
                       for c, layers in state["positions"].items() for pos in layers)
        current_equity = state["cash_usd"] + open_usd
        max_pos_cap = current_equity * MAX_POSITION_EQUITY_PCT
        target = min(max_pos_cap, state["cash_usd"] / float(open_slots))
        tranche = min(compute_max_affordable_tranche(state["cash_usd"], adv), max(TRANCHE_FLOOR_USD, target))
        if tranche < TRANCHE_FLOOR_USD:
            unfilled.append(item)
            continue

        part_rate = min(1.0, max(0.0, tranche / adv))
        slip = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate)) / 10000.0
        eff_entry, units, total_cost, fee_buy = _buy_fill_audited(tranche, live_px, slip)
        if state["cash_usd"] + 1e-9 < total_cost or units <= 0:
            unfilled.append(item)
            continue

        state["cash_usd"] -= total_cost
        layer = coin_layers + 1
        pos = {
            "coin": coin, "entry_bar": live_idx, "entry_date": str(datetime.now(IST).date()),
            "entry_price_usd": eff_entry, "entry_price_inr": eff_entry * fx, "units": units,
            "cost_usd": total_cost, "cost_inr": total_cost * fx, "entry_atr": item["entry_atr"],
            "current_sl_usd": eff_entry - p.get("sl_mult", 3.0) * item["entry_atr"], "stop_reason": "STOP_LOSS",
            "highest_high_usd": max(eff_entry, live_px), "lowest_low_usd": min(eff_entry, live_px),
            "layer": layer, "tp_done": False, "tp_proceeds_usd": 0.0, "proceeds_usd": 0.0, "fee_acc": fee_buy,
        }
        state["positions"].setdefault(coin, []).append(pos)
        consumed.add(item.get("signal_key", f"{coin}|{closed_date_str}"))
        buys.append({
            "coin": coin, "layer": layer, "price_usd": live_px, "effective_price_usd": eff_entry,
            "price_inr": live_px * fx, "allocate_inr": total_cost * fx, "signal_date": item.get("signal_date"),
        })

    state["watchlist"] = unfilled if p.get("wl_mode") != "WL_NONE" else []
    state["last_processed_date"] = closed_date_str
    state["consumed_signal_keys"] = sorted(consumed)[-500:]
    state["cash_inr"] = state["cash_usd"] * fx
    save_state(state)

    render_report(
        meta,
        p,
        state,
        market,
        dates,
        last_closed_idx,
        fx,
        buys,
        exits,
        tps,
        live_prices,
        signal_events=signal_events,
    )


def _money_inr(v: float, decimals: int = 2) -> str:
    return f"₹{float(v):,.{decimals}f}"


def _money_usd(v: float, decimals: int = 2) -> str:
    return f"${float(v):,.{decimals}f}"


def _pct(v: float, decimals: int = 2) -> str:
    return f"{float(v):+,.{decimals}f}%"


def _safe_pct(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return 0.0
    return float(numerator / denominator * 100.0)


def _pnl_color(v: float) -> str:
    if v > 0:
        return "#22c55e"
    if v < 0:
        return "#ef4444"
    return "#94a3b8"


def _signal_badge(active: bool, yes_text: str = "ACTIVE", no_text: str = "INACTIVE") -> str:
    if active:
        return (
            f"<span class='badge badge-green'>"
            f"<span class='dot'></span>{html.escape(yes_text)}</span>"
        )
    return (
        f"<span class='badge badge-gray'>"
        f"<span class='dot'></span>{html.escape(no_text)}</span>"
    )


def _risk_badge(cushion_pct: float) -> str:
    if cushion_pct <= 0:
        cls = "badge-red"
        label = "AT / BELOW EXIT"
    elif cushion_pct < 3:
        cls = "badge-orange"
        label = "TIGHT"
    elif cushion_pct < 7:
        cls = "badge-yellow"
        label = "WATCH"
    else:
        cls = "badge-green"
        label = "HEALTHY"

    return f"<span class='badge {cls}'>{label}</span>"


def _holding_bar(bars_held: int, max_bars: int) -> str:
    if max_bars <= 0:
        return ""

    pct = min(100.0, max(0.0, bars_held / max_bars * 100.0))

    if pct >= 90:
        cls = "bar-red"
    elif pct >= 70:
        cls = "bar-orange"
    else:
        cls = "bar-green"

    return f"""
    <div class="holding-wrap">
        <div class="holding-track">
            <div class="holding-fill {cls}" style="width:{pct:.1f}%"></div>
        </div>
        <span>{bars_held}/{max_bars} bars</span>
    </div>
    """


def _explain_entry_signal(p: Dict[str, Any]) -> str:
    parts = []
    parts.append(f"Entry type {p.get('entry_type', 'configured')}")
    if p.get("adx_thresh") is not None:
        parts.append(f"ADX threshold {p['adx_thresh']}")
    if p.get("vol_ma_len") is not None and p.get("vol_mult") is not None:
        parts.append(f"volume confirmation: {p['vol_ma_len']}-bar average × {p['vol_mult']}")
    if p.get("price_lookback") is not None:
        parts.append(f"price lookback: {p['price_lookback']} bars")
    if p.get("body_atr_mult") is not None:
        parts.append(f"body ≥ {p['body_atr_mult']}× ATR")
    if p.get("use_market_macro_system", False):
        parts.append("BTC macro filter enabled")
    else:
        parts.append("BTC macro filter disabled")
    return "; ".join(parts)


def _explain_exit_signal(p: Dict[str, Any]) -> str:
    parts = []
    parts.append(f"Exit type {p.get('exit_type', 'configured')}")
    if p.get("max_holding_bars") is not None:
        parts.append(f"max hold {p['max_holding_bars']} bars")
    if p.get("sl_mult") is not None:
        parts.append(f"protective stop {p['sl_mult']}× ATR")
    if p.get("trail_atr_mult", 0):
        parts.append(f"trailing component {p['trail_atr_mult']}× ATR")
    if p.get("use_global_tp", False):
        parts.append("global TP enabled")
    return "; ".join(parts)


def render_report(
    meta,
    p,
    state,
    market,
    dates,
    idx,
    fx,
    buys,
    exits,
    tps,
    live_prices,
    signal_events=None,
):
    signal_events = signal_events or []
    now_ist = datetime.now(IST)
    signal_date = str(dates[idx].date())

    free_cash_usd = float(state.get("cash_usd", 0.0))
    free_cash_inr = free_cash_usd * fx

    invested_usd = 0.0
    market_value_usd = 0.0
    unrealized_pnl_usd = 0.0
    position_count = 0
    portfolio_rows = []

    for coin in sorted(state["positions"]):
        q = live_prices.get(coin)
        if not q:
            continue
        live_px = float(q["price"])
        m = market.get(coin)

        for pos in state["positions"][coin]:
            position_count += 1
            cost_usd = float(pos.get("cost_usd", 0.0))
            units = float(pos.get("units", 0.0))
            market_value = units * live_px
            pnl_usd = float(pos.get("proceeds_usd", 0.0)) + market_value - cost_usd

            invested_usd += cost_usd
            market_value_usd += market_value
            unrealized_pnl_usd += pnl_usd

            pnl_pct = _safe_pct(pnl_usd, cost_usd)
            entry_px = float(pos.get("entry_price_usd", 0.0))
            stop_px = float(pos.get("current_sl_usd", 0.0))
            cushion_usd = live_px - stop_px
            cushion_pct = _safe_pct(cushion_usd, live_px)
            cushion_inr = cushion_usd * fx
            stop_value_usd = units * stop_px
            stop_loss_from_now_usd = market_value - stop_value_usd

            bars_held = max(0, (idx + 1) - int(pos.get("entry_bar", idx)))
            max_hold = int(p.get("max_holding_bars", 30))

            strategy_exit_active = False
            if m is not None:
                try:
                    strategy_exit_active = bool(m["sig"].exit_sig[idx])
                except Exception:
                    strategy_exit_active = False

            current_value_inr = market_value * fx
            cost_inr = cost_usd * fx
            allocation_pct = _safe_pct(market_value_usd, max(1e-9, market_value_usd + free_cash_usd))
            pnl_html = f"<span style='color:{_pnl_color(pnl_pct)};font-weight:800'>{_pct(pnl_pct)}</span>"
            strategy_exit_html = _signal_badge(strategy_exit_active, "EXIT SIGNAL", "HOLD")
            cushion_html = f"<div class='cushion-value'>{_money_inr(cushion_inr)}</div><div class='muted'>{cushion_pct:+.2f}% from exit</div>{_risk_badge(cushion_pct)}"
            exit_distance_text = "Price is at/below protective exit" if cushion_pct <= 0 else f"{_money_usd(cushion_usd)} above protective exit"

            portfolio_rows.append(f"""
                <tr>
                    <td>
                        <div class="coin-name">{html.escape(coin)}</div>
                        <div class="muted">Layer {pos.get('layer', 1)}</div>
                    </td>
                    <td>
                        <div>{_money_inr(entry_px * fx, 4)}</div>
                        <div class="muted">{pos.get('entry_date', '-')}</div>
                    </td>
                    <td>
                        <div>{_money_inr(live_px * fx, 4)}</div>
                        <div class="muted">{live_prices[coin].get('provider', 'unknown').upper()}</div>
                    </td>
                    <td>
                        <div>{_money_inr(cost_inr)}</div>
                        <div class="muted">{units:,.6f} units</div>
                    </td>
                    <td>
                        <div>{_money_inr(current_value_inr)}</div>
                        <div>{pnl_html}</div>
                    </td>
                    <td>
                        <div class="exit-price">{_money_inr(stop_px * fx, 4)}</div>
                        <div class="muted">{html.escape(str(pos.get('stop_reason', 'Protective exit')))}</div>
                    </td>
                    <td>
                        {cushion_html}
                        <div class="muted exit-distance">{html.escape(exit_distance_text)}</div>
                    </td>
                    <td>
                        {strategy_exit_html}
                        <div style="margin-top:7px">{_holding_bar(bars_held, max_hold)}</div>
                        <div class="muted" style="margin-top:5px">Stop-hit impact: {_money_inr(stop_loss_from_now_usd * fx)}</div>
                    </td>
                    <td>
                        <div>{allocation_pct:.1f}%</div>
                        <div class="muted">of portfolio</div>
                    </td>
                </tr>
            """)

    portfolio_value_usd = free_cash_usd + market_value_usd
    portfolio_value_inr = portfolio_value_usd * fx
    unrealized_pnl_inr = unrealized_pnl_usd * fx
    invested_pct = _safe_pct(invested_usd, max(1e-9, portfolio_value_usd))
    cash_pct = _safe_pct(free_cash_usd, max(1e-9, portfolio_value_usd))
    max_slots = int(p.get("max_concurrent_tranches", DEFAULT_MAX_CONCURRENT_TRANCHES))
    free_slots = max(0, max_slots - position_count)

    btc_price = float(live_prices.get("BTC", {}).get("price", 0.0))
    macro_active = bool(p.get("use_market_macro_system", False))
    macro_status = "ENABLED" if macro_active else "DISABLED"
    macro_text = "Macro filter configured" if macro_active else "No BTC macro filter applied"

    if buys:
        buy_cards = "".join(f"""
            <div class="action-card buy-card">
                <div class="action-icon">↗</div>
                <div class="action-content">
                    <div class="action-title">BUY <span>{html.escape(b['coin'])}</span> · Layer {b['layer']}</div>
                    <div class="action-main">{_money_inr(b['price_inr'], 4)}</div>
                    <div class="action-meta">Effective fill: {_money_inr(b['effective_price_usd'] * fx, 4)} &nbsp;·&nbsp; Allocation: {_money_inr(b['allocate_inr'])}</div>
                    <div class="action-meta">Signal candle: {html.escape(str(b.get('signal_date', '-')))}</div>
                </div>
            </div>""" for b in buys)
    else:
        buy_cards = "<div class='empty-card'>No new buy executions in this cycle.</div>"

    if exits:
        exit_cards = "".join(f"""
            <div class="action-card exit-card">
                <div class="action-icon">↘</div>
                <div class="action-content">
                    <div class="action-title">EXIT <span>{html.escape(e['coin'])}</span> · Layer {e['layer']}</div>
                    <div class="action-main">{_money_inr(e['exit_price_usd'] * fx, 4)}</div>
                    <div class="action-meta">Reason: <strong>{html.escape(e['reason'])}</strong></div>
                    <div class="action-meta">Realized return: <strong>{_pct(e['ret_pct'])}</strong></div>
                </div>
            </div>""" for e in exits)
    else:
        exit_cards = "<div class='empty-card'>No new exits in this cycle.</div>"

    signal_rows = "".join(f"""
        <tr>
            <td>
                <div class="coin-name">{html.escape(str(event.get('coin', '')))}</div>
                <div class="muted">{html.escape(str(event.get('signal_date', '-')))}</div>
            </td>
            <td>{_signal_badge(event.get('signal_type') == 'ENTRY', str(event.get('signal_type', 'ENTRY')), 'SIGNAL')}</td>
            <td>{_money_inr(float(event.get('signal_price', 0.0)) * fx, 4)}</td>
            <td>{_money_inr(float(event.get('live_price', 0.0)) * fx, 4)}</td>
            <td><strong>{html.escape(str(event.get('status', 'DETECTED')))}</strong></td>
            <td>{html.escape(str(event.get('reason', 'Strategy signal detected')))}</td>
        </tr>""" for event in signal_events) or """
        <tr><td colspan="6" class="empty-table">No new entry signals detected on the last closed daily candle.</td></tr>"""

    entry_explanation = _explain_entry_signal(p)
    exit_explanation = _explain_exit_signal(p)

    body = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 0; background: #070b14; color: #e5e7eb; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
.wrapper {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
.header {{ background: linear-gradient(135deg, #111827 0%, #0f172a 60%, #111827 100%); border: 1px solid #263244; border-radius: 18px; padding: 26px; margin-bottom: 18px; }}
.brand {{ font-size: 12px; letter-spacing: 2px; color: #94a3b8; font-weight: 700; text-transform: uppercase; }}
.title {{ font-size: 30px; font-weight: 800; margin-top: 5px; color: #f8fafc; }}
.subtitle {{ color: #94a3b8; margin-top: 7px; font-size: 13px; }}
.status-row {{ margin-top: 18px; }}
.badge {{ display: inline-block; padding: 5px 9px; border-radius: 999px; font-size: 10px; font-weight: 800; letter-spacing: .4px; white-space: nowrap; }}
.badge-green {{ background: #052e1b; color: #4ade80; border: 1px solid #166534; }}
.badge-red {{ background: #3f0d0d; color: #f87171; border: 1px solid #991b1b; }}
.badge-orange {{ background: #431407; color: #fb923c; border: 1px solid #9a3412; }}
.badge-yellow {{ background: #422006; color: #facc15; border: 1px solid #854d0e; }}
.badge-gray {{ background: #1e293b; color: #94a3b8; border: 1px solid #334155; }}
.dot {{ display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: currentColor; margin-right: 5px; }}
.grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 18px; }}
.card {{ background: #0f172a; border: 1px solid #263244; border-radius: 14px; padding: 17px; }}
.card-label {{ color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing: .7px; font-weight: 700; }}
.card-value {{ color: #f8fafc; font-size: 21px; font-weight: 800; margin-top: 7px; }}
.card-sub {{ color: #64748b; font-size: 11px; margin-top: 4px; }}
.section {{ background: #0f172a; border: 1px solid #263244; border-radius: 16px; padding: 20px; margin-bottom: 18px; }}
.section-title {{ font-size: 16px; font-weight: 800; color: #f8fafc; margin-bottom: 4px; }}
.section-description {{ color: #64748b; font-size: 12px; margin-bottom: 15px; }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
.info-box {{ background: #111827; border: 1px solid #253044; border-radius: 12px; padding: 15px; }}
.info-label {{ color: #64748b; font-size: 10px; text-transform: uppercase; font-weight: 800; letter-spacing: .7px; }}
.info-value {{ margin-top: 7px; color: #e2e8f0; font-size: 13px; line-height: 1.55; }}
.action-card {{ display: flex; gap: 14px; padding: 14px; border-radius: 12px; margin-bottom: 9px; }}
.buy-card {{ background: linear-gradient(90deg, #052e1b, #071f18); border: 1px solid #166534; }}
.exit-card {{ background: linear-gradient(90deg, #3f0d0d, #211015); border: 1px solid #991b1b; }}
.action-icon {{ font-size: 22px; width: 30px; }}
.action-title {{ font-weight: 800; font-size: 13px; }}
.action-title span {{ color: #f8fafc; }}
.action-main {{ font-size: 20px; font-weight: 800; margin-top: 4px; }}
.action-meta {{ color: #94a3b8; font-size: 11px; margin-top: 5px; }}
.empty-card {{ background: #111827; border: 1px dashed #334155; color: #64748b; padding: 14px; border-radius: 12px; font-size: 12px; }}
table {{ width: 100%; border-collapse: collapse; }}
th {{ background: #111827; color: #64748b; font-size: 9px; text-transform: uppercase; letter-spacing: .7px; text-align: left; padding: 11px 9px; border-bottom: 1px solid #263244; }}
td {{ padding: 13px 9px; border-bottom: 1px solid #1e293b; font-size: 11px; vertical-align: middle; }}
tr:last-child td {{ border-bottom: none; }}
.coin-name {{ color: #f8fafc; font-weight: 800; font-size: 13px; }}
.muted {{ color: #64748b; font-size: 10px; margin-top: 3px; }}
.exit-price {{ color: #fbbf24; font-weight: 800; }}
.cushion-value {{ font-weight: 800; color: #e2e8f0; }}
.exit-distance {{ max-width: 160px; line-height: 1.4; }}
.holding-wrap {{ min-width: 100px; }}
.holding-track {{ height: 5px; background: #1e293b; border-radius: 99px; overflow: hidden; margin-bottom: 4px; }}
.holding-fill {{ height: 100%; border-radius: 99px; }}
.bar-green {{ background: #22c55e; }}
.bar-orange {{ background: #f97316; }}
.bar-red {{ background: #ef4444; }}
.summary-line {{ display: flex; justify-content: space-between; border-bottom: 1px solid #1e293b; padding: 9px 0; font-size: 12px; }}
.summary-line:last-child {{ border-bottom: none; }}
.summary-label {{ color: #94a3b8; }}
.summary-value {{ color: #f8fafc; font-weight: 700; }}
.empty-table {{ text-align: center; color: #64748b; padding: 25px; }}
.footer {{ color: #475569; font-size: 10px; text-align: center; padding: 8px 0 20px; line-height: 1.6; }}
@media only screen and (max-width: 850px) {{
    .grid {{ grid-template-columns: repeat(2, 1fr); }}
    .two-col {{ grid-template-columns: 1fr; }}
    .wrapper {{ padding: 10px; }}
    .section {{ overflow-x: auto; }}
    table {{ min-width: 1000px; }}
}}
</style>
</head>
<body>
<div class="wrapper">
    <div class="header">
        <div class="brand">UNIFIED SWING BOT · CRYPTO</div>
        <div class="title">Daily Portfolio Intelligence</div>
        <div class="subtitle">Last closed candle: <strong>{html.escape(signal_date)} UTC</strong> &nbsp;·&nbsp; Live execution: <strong>{now_ist:%d %b %Y %H:%M:%S IST}</strong></div>
        <div class="status-row">
            {_signal_badge(position_count > 0, f"{position_count} OPEN POSITIONS", "NO OPEN POSITIONS")} &nbsp;
            {_signal_badge(len(buys) > 0, f"{len(buys)} BUY EXECUTED", "NO NEW BUY")} &nbsp;
            {_signal_badge(len(exits) > 0, f"{len(exits)} EXIT EXECUTED", "NO NEW EXIT")}
        </div>
    </div>
    <div class="grid">
        <div class="card"><div class="card-label">Total Equity</div><div class="card-value">{_money_inr(portfolio_value_inr)}</div><div class="card-sub">Free cash + current positions</div></div>
        <div class="card"><div class="card-label">Free Cash</div><div class="card-value">{_money_inr(free_cash_inr)}</div><div class="card-sub">{cash_pct:.1f}% of equity · immediately deployable</div></div>
        <div class="card"><div class="card-label">Invested Capital</div><div class="card-value">{_money_inr(invested_usd * fx)}</div><div class="card-sub">{invested_pct:.1f}% of portfolio cost basis</div></div>
        <div class="card"><div class="card-label">Active P&L</div><div class="card-value" style="color:{_pnl_color(_safe_pct(unrealized_pnl_usd, invested_usd))}">{_money_inr(unrealized_pnl_inr)}</div><div class="card-sub">{_pct(_safe_pct(unrealized_pnl_usd, invested_usd))} across open positions</div></div>
    </div>
    <div class="section">
        <div class="section-title">Market & Engine Status</div>
        <div class="section-description">Context used by this cycle before execution.</div>
        <div class="grid">
            <div class="card"><div class="card-label">BTC Live Price</div><div class="card-value">{_money_usd(btc_price)}</div><div class="card-sub">Macro reference asset</div></div>
            <div class="card"><div class="card-label">USD / INR</div><div class="card-value">₹{fx:,.2f}</div><div class="card-sub">Conversion used for reporting</div></div>
            <div class="card"><div class="card-label">Open Slots</div><div class="card-value">{free_slots} <span style="font-size:13px;color:#64748b">/ {max_slots}</span></div><div class="card-sub">{position_count} active tranche(s)</div></div>
            <div class="card"><div class="card-label">Macro Regime</div><div class="card-value">{macro_status}</div><div class="card-sub">{html.escape(macro_text)}</div></div>
        </div>
    </div>
    <div class="section">
        <div class="section-title">Signal Engine</div>
        <div class="section-description">What the bot considers an entry and what can force an exit.</div>
        <div class="two-col">
            <div class="info-box"><div class="info-label">ENTRY SIGNAL</div><div class="info-value"><strong>A valid entry must be generated on the <u>last closed daily candle</u>.</strong><br><br>{html.escape(entry_explanation)}<br><br><span style="color:#64748b">Execution takes place at the current available spot price.</span></div></div>
            <div class="info-box"><div class="info-label">EXIT SIGNAL</div><div class="info-value"><strong>Exit decisions are evaluated against the latest closed daily signal plus live protective risk.</strong><br><br>{html.escape(exit_explanation)}<br><br><span style="color:#64748b">Can exit on strategy signal, protective stop, or maximum holding period.</span></div></div>
        </div>
    </div>
    <div class="section">
        <div class="section-title">Today's Executed Actions</div>
        <div class="section-description">Orders generated during this live cycle.</div>
        <h4 style="color:#4ade80;margin-bottom:8px">BUY ACTIVITY</h4>
        {buy_cards}
        <h4 style="color:#f87171;margin-top:20px;margin-bottom:8px">EXIT ACTIVITY</h4>
        {exit_cards}
    </div>
    <div class="section">
        <div class="section-title">Open Portfolio</div>
        <div class="section-description">Current live valuation of every open tranche.</div>
        <table>
            <thead>
                <tr>
                    <th>Asset</th><th>Buy Price</th><th>Live Price</th><th>Invested</th><th>Market Value / P&L</th><th>Protective Exit</th><th>Cushion</th><th>Exit Signal / Holding</th><th>Allocation</th>
                </tr>
            </thead>
            <tbody>
                {''.join(portfolio_rows) if portfolio_rows else '<tr><td colspan="9" class="empty-table">No open positions.</td></tr>'}
            </tbody>
        </table>
    </div>
    <div class="section">
        <div class="section-title">Signal Watchlist</div>
        <div class="section-description">Signals detected from the latest closed daily candle.</div>
        <table>
            <thead>
                <tr><th>Asset</th><th>Signal</th><th>Signal Price</th><th>Live Price</th><th>Status</th><th>Comment</th></tr>
            </thead>
            <tbody>{signal_rows}</tbody>
        </table>
    </div>
    <div class="section">
        <div class="section-title">Wallet & Capital Deployment</div>
        <div class="two-col">
            <div class="info-box">
                <div class="info-label">CAPITAL SUMMARY</div>
                <div class="summary-line"><span class="summary-label">Total Equity</span><span class="summary-value">{_money_inr(portfolio_value_inr)}</span></div>
                <div class="summary-line"><span class="summary-label">Free Cash</span><span class="summary-value">{_money_inr(free_cash_inr)}</span></div>
                <div class="summary-line"><span class="summary-label">Invested Cost Basis</span><span class="summary-value">{_money_inr(invested_usd * fx)}</span></div>
                <div class="summary-line"><span class="summary-label">Current Market Value</span><span class="summary-value">{_money_inr(market_value_usd * fx)}</span></div>
                <div class="summary-line"><span class="summary-label">Active P&L</span><span class="summary-value" style="color:{_pnl_color(_safe_pct(unrealized_pnl_usd, invested_usd))}">{_money_inr(unrealized_pnl_inr)}</span></div>
            </div>
            <div class="info-box">
                <div class="info-label">RISK / CAPACITY</div>
                <div class="summary-line"><span class="summary-label">Maximum Tranches</span><span class="summary-value">{max_slots}</span></div>
                <div class="summary-line"><span class="summary-label">Active Tranches</span><span class="summary-value">{position_count}</span></div>
                <div class="summary-line"><span class="summary-label">Available Slots</span><span class="summary-value">{free_slots}</span></div>
                <div class="summary-line"><span class="summary-label">Cash Allocation</span><span class="summary-value">{cash_pct:.1f}%</span></div>
                <div class="summary-line"><span class="summary-label">Invested Allocation</span><span class="summary-value">{invested_pct:.1f}%</span></div>
            </div>
        </div>
    </div>
    <div class="footer">UnifiedSwingBot · Crypto Live Companion<br>Signal source: last closed daily candle · Execution source: current live spot · Reporting currency: INR</div>
</div>
</body>
</html>"""

    send_email(
        f"🪙 Crypto Intelligence | {len(buys)} BUY · {len(exits)} EXIT · {_money_inr(portfolio_value_inr, 0)} Equity",
        body,
    )


def main():
    try:
        run_cycle()
    except Exception:
        tb = traceback.format_exc()
        logger.error("Fatal crypto companion crash:\n%s", tb)
        send_email(f"⚠️ Crypto Bot CRASHED — {datetime.now(IST):%Y-%m-%d}", f"<pre>{html.escape(tb)}</pre>")


if __name__ == "__main__":
    main()