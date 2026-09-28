#!/usr/bin/env python3
"""
NSE EQUITY LIVE SNAPSHOT SWING-TRADING COMPANION
V11.2 — V10.1 ENGINE-PARITY + LIVE-LTP SNAPSHOT

Purpose:
  Decision-support + portfolio bookkeeping only. No broker/order API.

Parity contract with bt_nse_103.py:
  * exact V10.1 indicator kernels and signal definitions
  * signal on T-1 -> fill/decision on T
  * macro/max-hold/signal exits at T open using T-1 state
  * live snapshot close = current LTP; stop/exit reference = current LTP
  * TP uses T high and is processed after open exits but before state updates
  * exact watchlist revalidation, expiry, ranking, anti-averaging-down
  * exact dynamic slot/equity/ADV sizing math
  * audited NSE fee/slippage model
  * 20-bar permanent-delist guard
  * no fabricated historical catch-up fills; live mode only acts on the current snapshot

Live-specific adaptation:
  * whole NSE shares only (backtest permits fractional units)
  * actual cash is persisted; optional explicit cash reset is user-controlled
  * no terminal END_OF_TEST liquidation
"""
from __future__ import annotations

import argparse
import copy
import functools
import html
import io
import json
import logging
import os
import random
import smtplib
import ssl
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from swing_live_parity_core_v10x import (
    FastIndicators,
    SignalArrays,
    compile_single_symbol,
    np_rolling_mean,
    reconstitute_params,
)

# ==============================================================================
# CONFIG — strategy/risk values intentionally match bt_nse_103.py
# ==============================================================================

ENGINE_VERSION = "V10.1"
UNIVERSE_NAME = os.environ.get("STOCK_UNIVERSE", "NIFTY50").upper()
IST = ZoneInfo("Asia/Kolkata")

BASE_DIR = Path(os.environ.get("SWING_BOT_BASE_DIR", Path.home() / "UnifiedSwingBot"))
DATA_DIR = BASE_DIR / "stocks"
LOG_DIR = BASE_DIR / "logs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

WINNER_JSON_PATH = os.environ.get(
    "WINNER_JSON_PATH",
    str(BASE_DIR / "output" / UNIVERSE_NAME / "winner.json"),
)
LEGACY_WINNER_JSON_PATH = Path("/content/drive/MyDrive/UnifiedSwingBot/NIFTY50/winner.json")
WINNER_PATH = Path(WINNER_JSON_PATH)

STATE_PATH = Path(os.environ.get(
    "NSE_STATE_PATH",
    str(DATA_DIR / f"state_{UNIVERSE_NAME.lower()}.json"),
))

# Set None after your initial cash bootstrap to preserve rolling cash.
SET_AVAILABLE_CASH_INR: Optional[float] = None

STOCK_FORCE_RERUN = os.environ.get("STOCK_FORCE_RERUN", "true").strip().lower() in (
    "1", "true", "yes", "y", "on"
)
STOCK_UNIVERSE_LIMIT = None
STOCK_EXTRA_TICKERS = {
    x.strip().upper() for x in os.environ.get("STOCK_EXTRA_TICKERS", "").split(",") if x.strip()
}
STOCK_EXCLUDE_TICKERS = {
    x.strip().upper() for x in os.environ.get("STOCK_EXCLUDE_TICKERS", "").split(",") if x.strip()
}
DRY_RUN = os.environ.get("DRY_RUN", "false").strip().lower() in ("1", "true", "yes", "y", "on")

# Backtest constants, kept as constants here to prevent accidental drift.
UNIVERSE_LIQUIDITY_FLOOR_INR = {
    "NIFTY50": 0.0,
    "NIFTY100": 0.0,
    "MIDCAP150": 5_000_000.0,
    "SMALLCAP250": 1_000_000.0,
}
LIQUIDITY_FLOOR_INR = UNIVERSE_LIQUIDITY_FLOOR_INR[UNIVERSE_NAME]
TRANCHE_FLOOR_INR = 10_000.0
DEFAULT_MAX_CONCURRENT_TRANCHES = 8
MAX_POSITION_EQUITY_PCT = 0.25
MAX_ADV_PARTICIPATION = 0.015
WL_MAX_AGE_BARS = 15

UNIVERSE_SLIPPAGE_BPS = {
    "NIFTY50": 5.0, "NIFTY100": 8.0, "MIDCAP150": 15.0, "SMALLCAP250": 25.0,
}
UNIVERSE_IMPACT_COEF_BPS = {
    "NIFTY50": 80.0, "NIFTY100": 120.0, "MIDCAP150": 200.0, "SMALLCAP250": 320.0,
}
BASE_SLIPPAGE_BPS = UNIVERSE_SLIPPAGE_BPS[UNIVERSE_NAME]
IMPACT_COEF_BPS = UNIVERSE_IMPACT_COEF_BPS[UNIVERSE_NAME]

BROKERAGE_MODE = "ZERO_DELIVERY"
FLAT_BROKERAGE_INR = 20.0
STT_RATE = 0.0010
EXCHANGE_TXN_RATE = 0.0000322
STAMP_DUTY_RATE = 0.00015
SEBI_CHARGE_RATE = 0.000001
GST_RATE = 0.18
DP_CHARGE_INR = 15.0
DP_CHARGE_GST = DP_CHARGE_INR * GST_RATE
EFFECTIVE_BUY_FEE_RATE = (
    STT_RATE + EXCHANGE_TXN_RATE + STAMP_DUTY_RATE + SEBI_CHARGE_RATE
    + GST_RATE * (EXCHANGE_TXN_RATE + SEBI_CHARGE_RATE)
)
EFFECTIVE_SELL_FEE_RATE = (
    STT_RATE + EXCHANGE_TXN_RATE + SEBI_CHARGE_RATE
    + GST_RATE * (EXCHANGE_TXN_RATE + SEBI_CHARGE_RATE)
)

LIVE_QUOTE_MAX_AGE_MINUTES = float(os.environ.get("LIVE_QUOTE_MAX_AGE_MINUTES", "5"))
LIVE_INTRADAY_INTERVAL = "1m"

NSE_INDEX_SOURCES = {
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

MACRO_INDEX_TICKER = "^NSEI"
START_YEAR = 2008
MIN_HISTORY_DAYS = 300
FETCH_BATCH_SIZE = 40
DOWNLOAD_DELAY_RANGE = (0.2, 0.4)

# ==============================================================================
# LOGGING / SECRETS
# ==============================================================================

logger = logging.getLogger("nse_live_v11_1")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

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
        if str(BASE_DIR) == str(Path.home() / "UnifiedSwingBot"):
            BASE_DIR = Path("/content/drive/MyDrive/UnifiedSwingBot")
            DATA_DIR = BASE_DIR / "stocks"
            LOG_DIR = BASE_DIR / "logs"
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            WINNER_PATH = Path(os.environ.get(
                "WINNER_JSON_PATH",
                str(BASE_DIR / "output" / UNIVERSE_NAME / "winner.json"),
            ))
            STATE_PATH = Path(os.environ.get(
                "NSE_STATE_PATH",
                str(DATA_DIR / f"state_{UNIVERSE_NAME.lower()}.json"),
            ))
    except Exception as e:
        logger.warning("Colab Drive mount failed: %s", e)

# Rebind the dedicated file handler after any Colab Drive remap.
for _h in list(logger.handlers):
    if isinstance(_h, logging.FileHandler):
        logger.removeHandler(_h)
        try:
            _h.close()
        except Exception:
            pass
_file_handler = logging.FileHandler(LOG_DIR / "bot_stocks_v11_1.log")
_file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
logger.addHandler(_file_handler)


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
    if DRY_RUN:
        logger.info("[DRY RUN] Email suppressed: %s", subject)
        return
    user = get_secret("GMAIL_USER")
    pw = get_secret("GMAIL_APP_PASSWORD")
    to = get_secret("RECIPIENT_EMAIL")
    if not user or not pw or not to:
        logger.warning("Missing Gmail secrets; email skipped.")
        return
    msg = email.utils
    del msg
    mime = __import__("email.mime.multipart", fromlist=["MIMEMultipart"]).MIMEMultipart("alternative")
    mime["Subject"], mime["From"], mime["To"] = subject, user, to
    mime.attach(__import__("email.mime.text", fromlist=["MIMEText"]).MIMEText(body, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as server:
            server.login(user, pw)
            server.sendmail(user, to, mime.as_string())
        logger.info("Email delivered: %s", subject)
    except Exception as e:
        logger.error("Email send failed: %s", e)


# ==============================================================================
# BACKTEST FEE / FILL ENGINE — exact V10.1 formulas
# ==============================================================================

def _fee_dict() -> Dict[str, float]:
    return {
        "brokerage": 0.0, "stt": 0.0, "exchange_charges": 0.0,
        "stamp_duty": 0.0, "sebi_charges": 0.0, "gst": 0.0,
        "dp_charges": 0.0, "slippage_cost": 0.0,
    }


def _merge_fees(acc: Dict[str, float], inc: Dict[str, float]) -> None:
    for k, v in inc.items():
        acc[k] = float(acc.get(k, 0.0) + v)


def compute_buy_cost_audited(gross_inr: float, slip_cost: float) -> Tuple[float, Dict[str, float]]:
    if gross_inr <= 0:
        return 0.0, _fee_dict()
    brokerage = FLAT_BROKERAGE_INR if BROKERAGE_MODE == "FLAT" else 0.0
    stt = gross_inr * STT_RATE
    exch = gross_inr * EXCHANGE_TXN_RATE
    stamp = gross_inr * STAMP_DUTY_RATE
    sebi = gross_inr * SEBI_CHARGE_RATE
    gst = (brokerage + exch + sebi) * GST_RATE
    f = {
        "brokerage": brokerage, "stt": stt, "exchange_charges": exch,
        "stamp_duty": stamp, "sebi_charges": sebi, "gst": gst,
        "dp_charges": 0.0, "slippage_cost": slip_cost,
    }
    return gross_inr + brokerage + stt + exch + stamp + sebi + gst, f


def compute_sell_proceeds_audited(gross_inr: float, slip_cost: float, apply_dp: bool = True) -> Tuple[float, Dict[str, float]]:
    if gross_inr <= 0:
        return 0.0, _fee_dict()
    brokerage = FLAT_BROKERAGE_INR if BROKERAGE_MODE == "FLAT" else 0.0
    stt = gross_inr * STT_RATE
    exch = gross_inr * EXCHANGE_TXN_RATE
    sebi = gross_inr * SEBI_CHARGE_RATE
    gst = (brokerage + exch + sebi) * GST_RATE
    dp = (DP_CHARGE_INR + DP_CHARGE_GST) if apply_dp else 0.0
    f = {
        "brokerage": brokerage, "stt": stt, "exchange_charges": exch,
        "stamp_duty": 0.0, "sebi_charges": sebi, "gst": gst,
        "dp_charges": dp, "slippage_cost": slip_cost,
    }
    proceeds = max(0.0, gross_inr - brokerage - stt - exch - sebi - gst - dp)
    return proceeds, f


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


def _sell_fill_audited(units: float, ref_price: float, slip_mult: float) -> Tuple[float, float, Dict[str, float]]:
    fill_px = ref_price * (1.0 - slip_mult)
    gross = units * fill_px
    slip_cost = units * (ref_price - fill_px)
    return (fill_px, *compute_sell_proceeds_audited(gross, slip_cost, apply_dp=True))


def _whole_share_buy(
    cash: float, target_inr: float, ref_open: float, adv_30d: float
) -> Optional[Tuple[float, int, float, Dict[str, float], float]]:
    if ref_open <= 0:
        return None
    adv = max(adv_30d, 1_000_000.0)
    tranche = min(compute_max_affordable_tranche(cash, adv), max(TRANCHE_FLOOR_INR, target_inr))
    if tranche < TRANCHE_FLOOR_INR:
        return None

    slip_mult = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(min(1.0, max(0.0, tranche / adv)))) / 10000.0
    units = int(tranche // ref_open)
    if units <= 0:
        return None

    # Whole-share live adaptation: keep the V10 target and audit exact realized cost.
    for _ in range(3):
        fill_px = ref_open * (1.0 + slip_mult)
        gross = units * fill_px
        slip_cost = units * (fill_px - ref_open)
        total_cost, fees = compute_buy_cost_audited(gross, slip_cost)
        if total_cost <= cash + 1e-9:
            return fill_px, units, total_cost, fees, tranche
        units -= 1
        if units <= 0:
            return None

    return None


# ==============================================================================
# WINNER / STATE
# ==============================================================================

FALLBACK_PARAMS = {
    "entry_type": 1,
    "adx_thresh": 35.0,
    "rsi_f_len": 76,
    "rsi_f_smt": 10,
    "rsi_s_len": 74,
    "rsi_s_smt": 6,
    "use_rsi_trend_filter": False,
    "use_market_macro_system": True,
    "macro_ma_len": 167,
    "macro_ma_type": 2,
    "macro_active_exit": False,
    "max_concurrent_tranches": 5,
    "wl_mode": "WL_STRONGEST_MOMENTUM",
    "use_global_tp": False,
    "be_trigger_atr": 3.5,
    "max_holding_bars": 55,
    "sl_mult": 5.6,
    "exit_type": 0,
    "trail_atr_mult": 0.0,
    "max_pyramid_layers": 1,
}


def load_strategy_params() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    candidates = [WINNER_PATH]
    if LEGACY_WINNER_JSON_PATH.exists():
        candidates.append(LEGACY_WINNER_JSON_PATH)
    for path in candidates:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                p = data.get("best_params") or data.get("params")
                if p:
                    p = reconstitute_params(p, max_xover_long=200, max_exit_xover_long=200)
                    logger.info("Loaded V10.1 champion from %s (engine=%s, score=%s)",
                                path, data.get("version", "?"), data.get("is_score", "?"))
                    return p, data
            except Exception as e:
                logger.warning("Winner file %s could not be loaded: %s", path, e)
    logger.warning("No winner.json found; using conservative fallback.")
    return reconstitute_params(FALLBACK_PARAMS), {}


def _blank_state() -> Dict[str, Any]:
    return {"cash_inr": 0.0, "positions": {}, "watchlist": [], "last_processed_date": None}


def _normalize_position(t: str, pos: Dict[str, Any], dates: pd.DatetimeIndex) -> Dict[str, Any]:
    p = dict(pos)
    entry_price = float(p.get("entry_price", 0.0))
    invested = float(p.get("cost_inr", p.get("invested_amount", 0.0)))
    units = int(p.get("units", 0))
    if units <= 0 and entry_price > 0 and invested > 0:
        units = max(1, int(invested // entry_price))
    entry_date = str(p.get("entry_date", p.get("entry_bar_date", dates[0].date())))
    entry_bar = p.get("entry_bar")
    if entry_bar is None:
        matches = np.where(dates.strftime("%Y-%m-%d") == entry_date[:10])[0]
        entry_bar = int(matches[0]) if len(matches) else 0
    f = p.get("fee_acc", _fee_dict())
    out = {
        "ticker": t, "entry_bar": int(entry_bar), "entry_date": entry_date[:10],
        "entry_price": entry_price, "initial_units": int(p.get("initial_units", units)),
        "units": units, "cost_inr": invested if invested > 0 else units * entry_price,
        "entry_atr": float(p.get("entry_atr", max(entry_price * 0.02, 0.01))),
        "current_sl": float(p.get("current_sl", p.get("stop_loss", entry_price * 0.9))),
        "stop_reason": p.get("stop_reason", "STOP_LOSS"),
        "highest_high": float(p.get("highest_high", entry_price)),
        "lowest_low": float(p.get("lowest_low", entry_price)),
        "peak_bar": int(p.get("peak_bar", entry_bar)),
        "trough_bar": int(p.get("trough_bar", entry_bar)),
        "layer": int(p.get("layer", 1)),
        "tp_done": bool(p.get("tp_done", False)),
        "tp_proceeds": float(p.get("tp_proceeds", 0.0)),
        "from_watchlist": bool(p.get("from_watchlist", False)),
        "wait_days": int(p.get("wait_days", 0)),
        "proceeds": float(p.get("proceeds", 0.0)),
        "fee_acc": {k: float(v) for k, v in f.items()},
    }
    return out


def load_state(dates: pd.DatetimeIndex) -> Dict[str, Any]:
    state = _blank_state()
    if STATE_PATH.exists():
        try:
            with STATE_PATH.open("r", encoding="utf-8") as f:
                old = json.load(f)
            state["cash_inr"] = float(old.get("cash_inr", old.get("available_cash_inr", 0.0)))
            state["positions"] = {
                t: [_normalize_position(t, x, dates) for x in xs]
                for t, xs in old.get("positions", {}).items()
            }
            wl = old.get("watchlist", [])
            if isinstance(wl, dict):
                wl = [{"ticker": t, **info} for t, info in wl.items()]
            state["watchlist"] = list(wl)
            state["last_processed_date"] = old.get("last_processed_date", old.get("last_run_date"))
        except Exception as e:
            logger.warning("State migration/load failed; starting fresh: %s", e)
    return state


def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.replace(STATE_PATH)


def apply_manual_overrides(state: Dict[str, Any], dates: pd.DatetimeIndex, today_str: str) -> None:
    raw = os.environ.get("STOCK_MANUAL_OVERRIDES_JSON")
    if raw:
        try:
            cfg = json.loads(raw)
        except Exception as e:
            logger.warning("Invalid STOCK_MANUAL_OVERRIDES_JSON (%s); ignoring overrides.", e)
            cfg = {}
    else:
        cfg = {
            "force_remove_positions": [], "force_remove_watchlist": [],
            "force_add_positions": {}, "force_add_watchlist": {}, "force_update_positions": {},
        }
    for t in cfg.get("force_remove_positions", []):
        state["positions"].pop(t, None)
    state["watchlist"] = [
        w for w in state["watchlist"] if w.get("ticker", w.get("coin")) not in set(cfg.get("force_remove_watchlist", []))
    ]
    for t, info in cfg.get("force_add_positions", {}).items():
        if t in state["positions"]:
            continue
        state["positions"][t] = [_normalize_position(t, {
            "entry_price": float(info["entry_price"]),
            "entry_date": info.get("entry_date", today_str),
            "cost_inr": float(info.get("cost_inr", info.get("invested_amount", TRANCHE_FLOOR_INR))),
            "units": int(info.get("units", 0)),
            "entry_atr": float(info.get("entry_atr", float(info["entry_price"]) * 0.02)),
            "current_sl": float(info.get("current_sl", float(info["entry_price"]) * 0.9)),
            "layer": int(info.get("layer", 1)),
        }, dates)]
    for t, info in cfg.get("force_add_watchlist", {}).items():
        state["watchlist"].append({
            "ticker": t, "signal_bar_date": info.get("eligible_date", today_str),
            "signal_date": info.get("signal_date", today_str),
            "trigger_price": float(info.get("signal_close", info.get("trigger_price", 0.0))),
            "shadow_stop": float(info.get("shadow_stop", 0.0)),
            "highest_high": float(info.get("signal_close", info.get("trigger_price", 0.0))),
            "breakout_quality": float(info.get("breakout_quality", 0.0)),
            "entry_atr": float(info.get("entry_atr", 0.0)),
            "unfilled_reason": None,
        })
    for t, info in cfg.get("force_update_positions", {}).items():
        layer_no = int(info.get("layer", 1))
        for pos in state["positions"].get(t, []):
            if pos.get("layer") == layer_no:
                for k in ("entry_price", "current_sl", "entry_atr", "cost_inr"):
                    if k in info:
                        pos[k] = float(info[k])
                if "units" in info:
                    pos["units"] = int(info["units"])
                break


# ==============================================================================
# UNIVERSE + DATA
# ==============================================================================

def _nse_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    })
    try:
        s.get("https://www.nseindia.com", timeout=10)
    except Exception:
        pass
    return s


def fetch_stock_universe() -> List[str]:
    cache = DATA_DIR / f"universe_{UNIVERSE_NAME.lower()}.json"
    today = datetime.now(IST).date().isoformat()
    if cache.exists():
        try:
            d = json.loads(cache.read_text(encoding="utf-8"))
            if d.get("date") == today and d.get("tickers"):
                return d["tickers"]
        except Exception:
            pass
    s = _nse_session()
    for url in NSE_INDEX_SOURCES.get(UNIVERSE_NAME, NSE_INDEX_SOURCES["NIFTY50"]):
        try:
            r = s.get(url, timeout=15)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            col = next(c for c in ("Symbol", "SYMBOL", "symbol") if c in df.columns)
            tickers = sorted({f"{str(x).strip().upper()}.NS" for x in df[col] if str(x).strip()})
            cache.write_text(json.dumps({"date": today, "tickers": tickers}), encoding="utf-8")
            return tickers
        except Exception as e:
            logger.warning("NSE constituent mirror failed: %s", e)
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))["tickers"]
        except Exception:
            pass
    raise RuntimeError(f"Could not load {UNIVERSE_NAME} constituents.")


def _latest_allowed_nse_date() -> datetime.date:
    now = datetime.now(IST)
    if now.time().hour > 15 or (now.time().hour == 15 and now.time().minute >= 45):
        return now.date()
    return (now - timedelta(days=1)).date()




def _normalize_intraday_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        return idx.tz_localize(IST)
    return idx.tz_convert(IST)


def _extract_live_intraday_frame(raw: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    frame = _extract_batch(raw, ticker)
    if frame is None or frame.empty:
        return None
    frame = frame.copy()
    frame.columns = [str(c) for c in frame.columns]
    needed = {"Open", "High", "Low", "Close", "Volume"}
    if not needed.issubset(frame.columns):
        return None
    local_idx = _normalize_intraday_index(pd.DatetimeIndex(frame.index))
    frame.index = local_idx
    now = datetime.now(IST)
    today = now.date()
    frame = frame[(frame.index.date == today) & (frame.index.time >= datetime.strptime("09:15", "%H:%M").time())]
    if frame.empty:
        return None
    for c in ("Open", "High", "Low", "Close", "Volume"):
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
    if frame.empty:
        return None
    last_ts = frame.index[-1]
    age_min = max(0.0, (now - last_ts.to_pydatetime()).total_seconds() / 60.0)
    return pd.DataFrame({
        "open": [float(frame["Open"].iloc[0])],
        "high": [float(frame["High"].max())],
        "low": [float(frame["Low"].min())],
        "close": [float(frame["Close"].iloc[-1])],
        "volume": [float(frame["Volume"].fillna(0.0).sum())],
        "quote_time": [last_ts.isoformat()],
        "quote_age_min": [age_min],
    }, index=pd.DatetimeIndex([pd.Timestamp(today)]))


def fetch_live_nse_snapshot(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch today's live NSE session using 1-minute bars.

    The resulting bar is intentionally NOT written into the daily cache.
    Its Close is the latest available LTP; Open/High/Low/Volume are the
    session aggregates observed by yfinance up to the snapshot time.
    """
    if not tickers:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    try:
        raw = yf.download(
            tickers,
            period="1d",
            interval=LIVE_INTRADAY_INTERVAL,
            prepost=False,
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="ticker",
        )
    except Exception as e:
        logger.warning("NSE live 1m snapshot batch failed: %s", e)
        raw = None

    def ingest(raw_df: Optional[pd.DataFrame], ticker: str) -> bool:
        frame = _extract_live_intraday_frame(raw_df, ticker) if raw_df is not None else None
        if frame is None or frame.empty:
            return False
        row = frame.iloc[0]
        age = float(row["quote_age_min"])
        if age > LIVE_QUOTE_MAX_AGE_MINUTES:
            logger.warning("Stale NSE quote for %s: %.1f min old (limit %.1f).", ticker, age, LIVE_QUOTE_MAX_AGE_MINUTES)
        out[ticker] = {
            "open": float(row["open"]), "high": float(row["high"]), "low": float(row["low"]),
            "price": float(row["close"]), "volume": float(row["volume"]),
            "quote_time": str(row["quote_time"]), "quote_age_min": age,
        }
        return True

    missing = []
    for ticker in tickers:
        if not ingest(raw, ticker):
            missing.append(ticker)

    # Per-symbol retry only for missing names; avoids exploding API calls for the normal case.
    for ticker in missing:
        try:
            retry_df = yf.download(
                ticker,
                period="1d",
                interval=LIVE_INTRADAY_INTERVAL,
                prepost=False,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            ingest(retry_df, ticker)
        except Exception as e:
            logger.warning("Live NSE quote unavailable for %s: %s", ticker, e)
    return out


def append_nse_live_snapshot(
    ticker_data: Dict[str, pd.DataFrame],
    macro_df: pd.DataFrame,
    snapshot: Dict[str, Dict[str, Any]],
) -> Tuple[pd.Timestamp, pd.DataFrame]:
    """Append an in-memory current-session bar to every available symbol and benchmark."""
    live_date = pd.Timestamp(datetime.now(IST).date()).normalize()
    def add_row(df: pd.DataFrame, q: Optional[Dict[str, Any]]) -> pd.DataFrame:
        row = {
            "open": np.nan if q is None else q["open"],
            "high": np.nan if q is None else q["high"],
            "low": np.nan if q is None else q["low"],
            "close": np.nan if q is None else q["price"],
            "volume": 0.0 if q is None else q["volume"],
        }
        idx = pd.DatetimeIndex([live_date])
        x = pd.DataFrame(row, index=idx)
        x["quote_volume"] = x["close"].fillna(0.0) * x["volume"]
        z = df.copy()
        z.index = pd.to_datetime(z.index).tz_localize(None).normalize()
        z = z[z.index < live_date]
        return pd.concat([z, x]).sort_index()

    for ticker in list(ticker_data):
        ticker_data[ticker] = add_row(ticker_data[ticker], snapshot.get(ticker))

    macro_df = add_row(macro_df, snapshot.get(MACRO_INDEX_TICKER))
    return live_date, macro_df


def _extract_batch(raw: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    if raw is None or raw.empty:
        return None
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw.copy()
    levels0 = raw.columns.get_level_values(0)
    if ticker in levels0:
        return raw[ticker].copy()
    # yfinance can invert MultiIndex levels in some releases.
    if ticker in raw.columns.get_level_values(-1):
        return raw.xs(ticker, axis=1, level=-1).copy()
    return None


def _clean_stock_frame(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    if "date" not in df.columns:
        df = df.reset_index()
        df = df.rename(columns={"Date": "date"})
    needed = {"date", "open", "high", "low", "close", "volume"}
    if not needed.issubset(df.columns):
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.drop_duplicates("date").sort_values("date")
    cutoff = _latest_allowed_nse_date()
    df = df[df["date"].dt.date <= cutoff]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[
        (df["open"] > 0) & (df["close"] > 0) &
        (df["high"] >= df["low"]) &
        ((df["high"] / np.maximum(1e-8, df["low"])) < 50.0)
    ]
    if len(df) < MIN_HISTORY_DAYS:
        return None
    out = df.set_index("date")
    out["quote_volume"] = out["close"] * out["volume"].fillna(0.0)
    return out


def fetch_price_data(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    need_new: List[str] = []
    old_map: Dict[str, pd.DataFrame] = {}
    allowed = _latest_allowed_nse_date()

    for ticker in tickers:
        path = DATA_DIR / f"{ticker.replace('^','IDX_')}.parquet"
        if path.exists():
            try:
                old = pd.read_parquet(path)
                old.index = pd.to_datetime(old.index).tz_localize(None).normalize()
                if old.index.max().date() >= allowed:
                    results[ticker] = old
                    continue
                old_map[ticker] = old
            except Exception:
                pass
        need_new.append(ticker)

    if need_new:
        for i in range(0, len(need_new), FETCH_BATCH_SIZE):
            batch = need_new[i:i + FETCH_BATCH_SIZE]
            starts = []
            for t in batch:
                old = old_map.get(t)
                starts.append(old.index.max() - pd.Timedelta(days=10) if old is not None else pd.Timestamp(f"{START_YEAR}-01-01"))
            start = min(starts)
            try:
                raw = yf.download(
                    batch, start=start.strftime("%Y-%m-%d"), progress=False,
                    auto_adjust=True, threads=False, group_by="ticker",
                )
            except Exception as e:
                logger.warning("yfinance batch failed (%d): %s", len(batch), e)
                raw = None

            for t in batch:
                fresh = _clean_stock_frame(_extract_batch(raw, t))
                old = old_map.get(t)
                if old is not None and fresh is not None:
                    combined = pd.concat([old, fresh])
                    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                    fresh = _clean_stock_frame(combined.reset_index())
                elif old is not None and fresh is None:
                    fresh = old
                if fresh is not None:
                    path = DATA_DIR / f"{t.replace('^','IDX_')}.parquet"
                    fresh.to_parquet(path)
                    results[t] = fresh

            if i + FETCH_BATCH_SIZE < len(need_new):
                time.sleep(random.uniform(*DOWNLOAD_DELAY_RANGE))
    return results


# ==============================================================================
# MARKET GRID + SIGNALS
# ==============================================================================

def build_live_matrix(
    ticker_data: Dict[str, pd.DataFrame], macro_df: pd.DataFrame
) -> Tuple[pd.DatetimeIndex, Dict[str, Dict[str, Any]]]:
    master_dates = pd.DatetimeIndex(sorted(pd.to_datetime(macro_df.index).unique())).normalize()
    market: Dict[str, Dict[str, Any]] = {}

    for ticker, df in ticker_data.items():
        re = df.reindex(master_dates)
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

        c = re["close"].to_numpy(float)
        o = re["open"].to_numpy(float)
        h = re["high"].to_numpy(float)
        l = re["low"].to_numpy(float)
        v = re["volume"].to_numpy(float)
        dvol = np.nan_to_num(np_rolling_mean(c * v, 30), nan=1_000_000.0)

        market[ticker] = {
            "df": re, "alive": alive, "delist": is_delisted,
            "open": o, "high": h, "low": l, "close": c, "volume": v, "dvol": dvol,
        }
    return master_dates, market


def compile_all_signals(master_dates: pd.DatetimeIndex, market: Dict[str, Dict[str, Any]], macro_df: pd.DataFrame, p: Dict[str, Any]):
    macro = macro_df.reindex(master_dates).ffill()
    mc = macro["close"].to_numpy(float)
    if p.get("use_market_macro_system", False):
        mma = FastIndicators.moving_average(mc, p["macro_ma_len"], p["macro_ma_type"])
        macro_ok = (~np.isnan(mma)) & (mc > mma)
    else:
        macro_ok = np.ones(len(master_dates), dtype=bool)

    for ticker, m in market.items():
        sig = compile_single_symbol(
            m["open"], m["high"], m["low"], m["close"], m["volume"], m["dvol"],
            m["alive"], macro_ok, p, LIQUIDITY_FLOOR_INR,
        )
        m["sig"] = sig
    return macro_ok


# ==============================================================================
# LIVE BAR ENGINE
# ==============================================================================

def _idx_from_date(dates: pd.DatetimeIndex, value: str) -> Optional[int]:
    key = str(value)[:10]
    matches = np.where(dates.strftime("%Y-%m-%d") == key)[0]
    return int(matches[0]) if len(matches) else None


def _position_market_value(state, market, latest_idx) -> float:
    total = 0.0
    for ticker, layers in state["positions"].items():
        if ticker not in market:
            continue
        px = float(market[ticker]["close"][latest_idx])
        for pos in layers:
            total += pos["units"] * px
    return total


def _make_event(bucket: str, event: Dict[str, Any], is_today: bool,
                actionable: Dict[str, list], retro: list) -> None:
    if is_today:
        actionable[bucket].append(event)
    else:
        retro.append(event)



def run_cycle() -> None:
    p, winner_meta = load_strategy_params()
    today_ist = datetime.now(IST)
    live_date = pd.Timestamp(today_ist.date()).normalize()

    if today_ist.weekday() >= 5:
        logger.warning("NSE live companion started on weekend (%s). No live session expected.", today_ist.date())

    constituents = fetch_stock_universe()
    probe = _blank_state()
    if STATE_PATH.exists():
        try:
            probe = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    held = set(probe.get("positions", {}).keys())
    wl_probe = probe.get("watchlist", [])
    held_wl = set(wl_probe.keys()) if isinstance(wl_probe, dict) else {w.get("ticker") for w in wl_probe if isinstance(w, dict)}
    held_wl.discard(None)
    tickers = sorted((set(constituents) | STOCK_EXTRA_TICKERS | held | held_wl) - STOCK_EXCLUDE_TICKERS)
    if STOCK_UNIVERSE_LIMIT:
        tickers = tickers[:STOCK_UNIVERSE_LIMIT]

    logger.info("Loading %d NSE symbols + benchmark historical data.", len(tickers))
    raw = fetch_price_data(tickers + [MACRO_INDEX_TICKER])
    macro_df = raw.pop(MACRO_INDEX_TICKER, None)
    if macro_df is None or macro_df.empty:
        raise RuntimeError("Missing ^NSEI daily benchmark data.")

    logger.info("Fetching current NSE 1-minute snapshot; current LTP will be today's signal Close.")
    snapshot = fetch_live_nse_snapshot(tickers + [MACRO_INDEX_TICKER])
    if MACRO_INDEX_TICKER not in snapshot:
        raise RuntimeError("Could not obtain a current ^NSEI snapshot. Refusing to generate live signals.")

    live_date, macro_df = append_nse_live_snapshot(raw, macro_df, snapshot)
    logger.info("Live NSE snapshot date=%s, symbols quoted=%d/%d, ^NSEI LTP=%.2f",
                live_date.date(), len(snapshot), len(tickers) + 1, snapshot[MACRO_INDEX_TICKER]["price"])

    master_dates, market = build_live_matrix(raw, macro_df)
    if len(master_dates) < 2 or master_dates[-1] != live_date:
        raise RuntimeError("Live snapshot was not appended as today's working bar.")
    today_idx = len(master_dates) - 1
    yesterday_idx = today_idx - 1
    today_str = str(live_date.date())

    state = load_state(master_dates)
    if SET_AVAILABLE_CASH_INR is not None:
        state["cash_inr"] = float(SET_AVAILABLE_CASH_INR)
        logger.info("Cash override applied: ₹%.2f", state["cash_inr"])
    apply_manual_overrides(state, master_dates, today_str)

    # Normalize persisted indices without replaying historical fills. Live mode is a
    # current-snapshot decision engine, not a synthetic historical backfill engine.
    for item in state["watchlist"]:
        if item.get("signal_bar") is None:
            sb = _idx_from_date(master_dates, item.get("signal_date", today_str))
            item["signal_bar"] = int(sb) if sb is not None else max(0, yesterday_idx)
        item["ticker"] = item.get("ticker", item.get("coin"))
    for ticker, layers in state["positions"].items():
        for pos in layers:
            if "entry_bar" not in pos:
                eb = _idx_from_date(master_dates, pos.get("entry_date", today_str))
                pos["entry_bar"] = int(eb) if eb is not None else yesterday_idx

    # Compile against the actual current snapshot bar.
    macro_ok = compile_all_signals(master_dates, market, macro_df, p)

    # Make live quote metadata explicit for reports and quote-age checks.
    for ticker, q in snapshot.items():
        if ticker in market:
            market[ticker]["live_price"] = float(q["price"])
            market[ticker]["quote_time"] = q["quote_time"]
            market[ticker]["quote_age_min"] = q["quote_age_min"]
            market[ticker]["live_available"] = True
    for ticker in market:
        if "live_available" not in market[ticker]:
            market[ticker]["live_available"] = False

    all_buys, all_exits, all_tp, retro = [], [], [], []
    raw_signals = 0
    executed_signal_keys = set(state.get("consumed_signal_keys", []))

    # Keep enough history for old keys from becoming unbounded, but never discard today's key.
    def remember_signal(key: str) -> None:
        executed_signal_keys.add(key)

    # ------------------------------------------------------------------
    # PASS A: Existing positions — current live price is the action price.
    # ------------------------------------------------------------------
    max_holding_default = 20
    for ticker in list(state["positions"]):
        m = market.get(ticker)
        q = snapshot.get(ticker)
        if m is None or q is None:
            logger.warning("No live quote for held position %s; no action generated.", ticker)
            continue
        sig: SignalArrays = m["sig"]
        live_px = float(q["price"])
        live_high = float(q["high"])
        live_low = float(q["low"])
        prev_atr = float(sig.atr[yesterday_idx]) if np.isfinite(sig.atr[yesterday_idx]) else float(sig.atr[today_idx]) if np.isfinite(sig.atr[today_idx]) else 0.0
        adv_prev = max(float(m["dvol"][yesterday_idx]), 1_000_000.0)

        for pos in list(state["positions"][ticker]):
            bars_held = max(0, today_idx - int(pos["entry_bar"]))
            macro_bear_confirmed = bool(p.get("macro_active_exit", False) and yesterday_idx >= 2 and
                (not macro_ok[yesterday_idx]) and (not macro_ok[yesterday_idx-1]) and (not macro_ok[yesterday_idx-2]))
            strategy_exit = int(p["exit_type"]) in (3, 4, 5, 6, 7) and bool(sig.exit_sig[today_idx])
            signal_eligible = bars_held >= 3 or (pos["highest_high"] - pos["entry_price"]) >= pos["entry_atr"]

            reason = None
            if macro_bear_confirmed:
                reason = "MACRO_REGIME_EXIT"
            elif bars_held >= p.get("max_holding_bars", max_holding_default):
                reason = "MAX_HOLDING_TIME"
            elif strategy_exit and signal_eligible:
                reason = f"SIGNAL_EXIT_TYPE_{p['exit_type']}"

            tp_price = pos["entry_price"] + p.get("tp_mult", 4.0) * pos["entry_atr"]
            if reason is None and live_px <= pos["current_sl"]:
                reason = pos["stop_reason"]
            if reason is None and p.get("use_global_tp", False) and not pos.get("tp_done", False) and live_px >= tp_price:
                close_units = min(pos["units"], max(1, int(np.floor(pos["units"] * p.get("tp_size_pct", 50.0) / 100.0))))
                if close_units > 0:
                    part_rate = min(1.0, max(0.0, (close_units * live_px) / adv_prev))
                    slip = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate)) / 10000.0
                    fill_px, credit, fee = _sell_fill_audited(close_units, live_px * 1.0, slip, apply_dp=True)
                    state["cash_inr"] += credit
                    pos["proceeds"] += credit
                    pos["tp_proceeds"] += credit
                    pos["units"] -= close_units
                    pos["tp_done"] = True
                    _merge_fees(pos["fee_acc"], fee)
                    if p.get("tp_move_sl_be", False) and pos["current_sl"] < pos["entry_price"] * 1.002:
                        pos["current_sl"] = pos["entry_price"] * 1.002
                        pos["stop_reason"] = "BREAKEVEN_SL"
                    all_tp.append({
                        "date": today_str, "ticker": ticker, "layer": pos["layer"],
                        "action": f"TAKE PROFIT (SELL {p.get('tp_size_pct', 50.0):.0f}%)",
                        "price": live_px, "effective_price": fill_px, "entry": pos["entry_price"],
                        "freed_cash_inr": credit, "reference_price": live_px,
                    })

            if reason is not None:
                part_rate = min(1.0, max(0.0, (pos["units"] * live_px) / adv_prev))
                slip = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(part_rate)) / 10000.0
                fill_px, proceeds, fee = _sell_fill_audited(pos["units"], live_px, slip, apply_dp=True)
                state["cash_inr"] += proceeds
                pos["proceeds"] += proceeds
                _merge_fees(pos["fee_acc"], fee)
                pnl = pos["proceeds"] - pos["cost_inr"]
                all_exits.append({
                    "date": today_str, "ticker": ticker, "layer": pos["layer"],
                    "action": "CLOSE 100%", "reason": reason,
                    "reference_price": live_px, "effective_price": fill_px,
                    "entry_price": pos["entry_price"], "exit_price": live_px,
                    "pnl_inr": pnl, "ret_pct": pnl / max(1e-9, pos["cost_inr"]) * 100.0,
                    "freed_cash_inr": proceeds,
                })
                state["positions"][ticker].remove(pos)
                continue

            # V10.1 state update after exit decision: today's intraday high/low can move
            # the peak/trough, while ATR trailing uses the last completed ATR to remain causal.
            if live_high > pos["highest_high"]:
                pos["highest_high"] = live_high
                pos["peak_bar"] = today_idx
            if live_low < pos["lowest_low"]:
                pos["lowest_low"] = live_low
                pos["trough_bar"] = today_idx
            be = p.get("be_trigger_atr", 0.0)
            if be > 0.0 and pos["current_sl"] < pos["entry_price"] * 1.002:
                if pos["highest_high"] >= pos["entry_price"] + be * pos["entry_atr"]:
                    pos["current_sl"] = max(pos["current_sl"], pos["entry_price"] * 1.002)
                    pos["stop_reason"] = "BREAKEVEN_SL"
            if int(p["exit_type"]) == 1:
                floor = pos["highest_high"] * (1.0 - p.get("trail_pct", 10.0) / 100.0)
                if floor > pos["current_sl"]:
                    pos["current_sl"], pos["stop_reason"] = floor, "TRAIL_PCT_STOP"
            elif p.get("trail_atr_mult", 0.0) > 0.0 and np.isfinite(prev_atr):
                floor = pos["highest_high"] - p["trail_atr_mult"] * prev_atr
                if floor > pos["current_sl"]:
                    pos["current_sl"], pos["stop_reason"] = floor, "TRAIL_ATR_STOP"

    state["positions"] = {t: xs for t, xs in state["positions"].items() if xs}

    # ------------------------------------------------------------------
    # PASS B: Candidate/watchlist evaluation using today's LIVE signal.
    # ------------------------------------------------------------------
    max_pyramid = int(p.get("max_pyramid_layers", 1))
    wl_mode = p.get("wl_mode", "WL_NONE")

    new_candidates = []
    for ticker, m in market.items():
        q = snapshot.get(ticker)
        if q is None or not m.get("live_available", False):
            continue
        sig = m["sig"]
        if bool(sig.raw_signal[today_idx]):
            raw_signals += 1
            if bool(sig.entry[today_idx]) and bool(macro_ok[today_idx]) and len(state["positions"].get(ticker, [])) < max_pyramid:
                key = f"{ticker}|{today_str}|{int(p.get('entry_type',0))}|{int(p.get('exit_type',0))}"
                if key not in executed_signal_keys:
                    atr_live = float(sig.atr[today_idx]) if np.isfinite(sig.atr[today_idx]) else float(sig.atr[yesterday_idx])
                    breakout_quality = (float(m["close"][today_idx]) - float(m["open"][today_idx])) / max(1e-6, atr_live)
                    new_candidates.append({
                        "ticker": ticker, "signal_bar": today_idx, "signal_date": today_str,
                        "trigger_price": float(m["close"][today_idx]),
                        "shadow_stop": float(q["price"] - p.get("sl_mult", 3.0) * atr_live),
                        "highest_high": float(q["high"]), "breakout_quality": breakout_quality,
                        "entry_atr": atr_live, "unfilled_reason": None, "signal_key": key,
                    })
                    if wl_mode == "WL_NONE":
                        # WL_NONE is same-snapshot immediate execution in live mode.
                        pass

    state["watchlist"].extend(new_candidates)

    # Clean/revalidate existing watchlist. A current LTP is the live decision price;
    # previous completed ATR is used for trailing shadow stops.
    survivors_wl = []
    for item in state["watchlist"]:
        ticker = item.get("ticker")
        m = market.get(ticker)
        q = snapshot.get(ticker)
        if m is None or q is None:
            if wl_mode != "WL_NONE":
                survivors_wl.append(item)
            continue
        sig = m["sig"]
        live_px = float(q["price"])
        live_high = float(q["high"])
        live_low = float(q["low"])
        if live_high > item.get("highest_high", live_high):
            item["highest_high"] = live_high
        if live_low <= item.get("shadow_stop", -np.inf):
            continue
        if int(p["exit_type"]) in (3, 4, 5, 6, 7) and bool(sig.exit_sig[today_idx]):
            continue
        age = max(0, today_idx - int(item.get("signal_bar", today_idx)))
        if age >= WL_MAX_AGE_BARS:
            continue
        atr_prev = float(sig.atr[yesterday_idx]) if np.isfinite(sig.atr[yesterday_idx]) else float(item.get("entry_atr", 0.0))
        if p.get("trail_atr_mult", 0.0) > 0.0 and np.isfinite(atr_prev):
            item["shadow_stop"] = max(float(item["shadow_stop"]), float(item["highest_high"]) - p["trail_atr_mult"] * atr_prev)
        item["age_bars"] = age
        survivors_wl.append(item)
    state["watchlist"] = survivors_wl

    def wl_score(item: Dict[str, Any]) -> float:
        ticker = item["ticker"]
        px = float(snapshot[ticker]["price"])
        if wl_mode == "WL_DEEPEST_DISCOUNT":
            return (item["trigger_price"] - px) / max(1e-6, item["trigger_price"])
        if wl_mode == "WL_STRONGEST_MOMENTUM":
            return float(item.get("breakout_quality", 0.0))
        return -float(item.get("age_bars", 0))

    state["watchlist"].sort(key=wl_score, reverse=True)

    # ------------------------------------------------------------------
    # PASS C: Immediate live-LTP fills.
    # ------------------------------------------------------------------
    max_slots = int(p.get("max_concurrent_tranches", DEFAULT_MAX_CONCURRENT_TRANCHES))
    unfilled = []
    for item in state["watchlist"]:
        ticker = item["ticker"]
        m = market.get(ticker)
        q = snapshot.get(ticker)
        if m is None or q is None:
            if wl_mode != "WL_NONE":
                unfilled.append(item)
            continue
        if sum(len(v) for v in state["positions"].values()) >= max_slots:
            if wl_mode != "WL_NONE":
                item["unfilled_reason"] = "slot_saturated"; unfilled.append(item)
            continue
        if int(p.get("max_pyramid_layers", 1)) <= len(state["positions"].get(ticker, [])):
            if wl_mode != "WL_NONE":
                item["unfilled_reason"] = "pyramid_blocked"; unfilled.append(item)
            continue

        live_px = float(q["price"])
        existing_layers = state["positions"].get(ticker, [])
        if existing_layers:
            highest_entry = max(pos["entry_price"] for pos in existing_layers)
            if live_px <= highest_entry * 1.005:
                if wl_mode != "WL_NONE":
                    item["unfilled_reason"] = "anti_averaging_down"; unfilled.append(item)
                continue

        adv_prev = max(float(m["dvol"][yesterday_idx]), 1_000_000.0)
        open_slots = max(1, max_slots - sum(len(v) for v in state["positions"].values()))
        current_active = sum(
            pos["units"] * float(snapshot.get(t, {}).get("price", market[t]["close"][yesterday_idx]))
            for t, layers in state["positions"].items() if t in market
            for pos in layers
        )
        current_equity = state["cash_inr"] + current_active
        max_pos_cap = current_equity * MAX_POSITION_EQUITY_PCT
        dynamic_slot_target = min(max_pos_cap, state["cash_inr"] / float(open_slots))
        current_scrip_exposure = sum(pos["units"] * live_px for pos in existing_layers)
        remaining_capacity = max(0.0, max_pos_cap - current_scrip_exposure)
        if remaining_capacity < TRANCHE_FLOOR_INR:
            if wl_mode != "WL_NONE":
                item["unfilled_reason"] = "scrip_risk_cap"; unfilled.append(item)
            continue
        liquidity_cap = adv_prev * MAX_ADV_PARTICIPATION
        scaled_ceiling = max(500_000.0, current_equity * 0.35)
        target = min(dynamic_slot_target, scaled_ceiling, liquidity_cap, remaining_capacity)
        fill = _whole_share_buy(state["cash_inr"], target, live_px, adv_prev)
        if fill is None:
            if wl_mode != "WL_NONE":
                item["unfilled_reason"] = "cash_starved"; unfilled.append(item)
            continue
        effective_px, units, total_cost, fee_buy, tranche = fill
        if state["cash_inr"] + 1e-9 < total_cost:
            if wl_mode != "WL_NONE":
                item["unfilled_reason"] = "cash_starved"; unfilled.append(item)
            continue

        layer = len(existing_layers) + 1
        state["cash_inr"] -= total_cost
        pos = {
            "ticker": ticker, "entry_bar": today_idx, "entry_date": today_str,
            "entry_price": effective_px, "reference_entry_price": live_px,
            "initial_units": units, "units": units, "cost_inr": total_cost,
            "entry_atr": item["entry_atr"], "current_sl": effective_px - p.get("sl_mult", 3.0) * item["entry_atr"],
            "stop_reason": "STOP_LOSS", "highest_high": max(effective_px, float(q["high"])),
            "lowest_low": min(effective_px, float(q["low"])), "peak_bar": today_idx, "trough_bar": today_idx,
            "layer": layer, "tp_done": False, "tp_proceeds": 0.0,
            "from_watchlist": today_idx > int(item.get("signal_bar", today_idx)),
            "wait_days": max(0, today_idx - int(item.get("signal_bar", today_idx))),
            "proceeds": 0.0, "fee_acc": fee_buy,
        }
        state["positions"].setdefault(ticker, []).append(pos)
        state["watchlist"] = [w for w in state["watchlist"] if w is not item]
        remember_signal(item.get("signal_key", f"{ticker}|{today_str}"))
        all_buys.append({
            "date": today_str, "ticker": ticker, "layer": layer,
            "action": f"BUY (Layer {layer})", "price": live_px, "effective_price": effective_px,
            "allocate_inr": total_cost, "signal_date": item.get("signal_date"),
            "reference_price": live_px,
        })

    # WL_NONE never carries an unfilled signal forward.
    if wl_mode == "WL_NONE":
        unfilled = []
    state["watchlist"] = unfilled

    state["last_processed_date"] = today_str
    state["last_snapshot_time_ist"] = datetime.now(IST).isoformat()
    state["last_snapshot_quotes"] = {t: {"price": q["price"], "quote_time": q["quote_time"], "age_min": q["quote_age_min"]} for t, q in snapshot.items()}
    state["consumed_signal_keys"] = sorted(executed_signal_keys)[-500:]
    save_state(state)

    current_signals = []
    for ticker, m in market.items():
        q = snapshot.get(ticker)
        if q is None:
            continue
        sig = m["sig"]
        if bool(sig.raw_signal[today_idx]):
            current_signals.append({
                "ticker": ticker, "close": float(q["price"]),
                "atr": float(sig.atr[today_idx]) if np.isfinite(sig.atr[today_idx]) else 0.0,
                "adx": float(sig.adx[today_idx]) if np.isfinite(sig.adx[today_idx]) else 0.0,
                "qualifying": bool(sig.entry[today_idx]),
                "layers": len(state["positions"].get(ticker, [])),
                "quote_age_min": float(q["quote_age_min"]),
            })

    render_report(p, winner_meta, state, market, master_dates, today_idx,
                  all_buys, all_exits, all_tp, retro, current_signals, raw_signals,
                  snapshot)


def render_report(p, winner_meta, state, market, dates, latest_idx, buys, exits, tps, retro, current_signals, raw_signals, snapshot):
    latest_date = str(dates[latest_idx].date())
    current_market = 0.0
    position_rows = []
    for ticker in sorted(state["positions"]):
        if ticker not in market or ticker not in snapshot:
            continue
        px = float(snapshot[ticker]["price"])
        for pos in state["positions"][ticker]:
            remaining = pos["units"] * px
            current_market += remaining
            total_value = pos.get("proceeds", 0.0) + remaining
            pnl = total_value - pos["cost_inr"]
            position_rows.append(
                f"<tr><td>{html.escape(ticker)} L{pos['layer']}</td><td>{pos['entry_date']}</td>"
                f"<td>₹{pos.get('reference_entry_price',pos['entry_price']):,.2f}</td><td>{pos['units']:,}</td>"
                f"<td>₹{px:,.2f}</td><td>{pnl:+,.2f}</td><td>{pnl/max(1e-9,pos['cost_inr'])*100:+.2f}%</td>"
                f"<td>₹{pos['current_sl']:,.2f} ({html.escape(pos['stop_reason'])})</td></tr>"
            )
    equity = state["cash_inr"] + current_market
    invested = sum(pos["cost_inr"] for xs in state["positions"].values() for pos in xs)
    portfolio_pnl_pct = (current_market - invested) / invested * 100 if invested > 0 else 0.0
    quote_count = len(snapshot)
    stale = [t for t, q in snapshot.items() if q["quote_age_min"] > LIVE_QUOTE_MAX_AGE_MINUTES]
    buy_html = "".join(
        f"<div class='buy'>BUY — <b>{html.escape(b['ticker'])}</b> L{b['layer']} @ LTP <b>₹{b['price']:,.2f}</b> "
        f"(estimated effective ₹{b['effective_price']:,.2f}; cost ₹{b['allocate_inr']:,.0f}; signal {b.get('signal_date','--')})</div>" for b in buys
    ) or "<div class='muted'>No new live buy actions.</div>"
    exit_html = "".join(
        f"<div class='sell'>{html.escape(e['ticker'])} L{e['layer']} — {html.escape(e['reason'])}, "
        f"EXIT LTP <b>₹{e.get('exit_price',0):,.2f}</b> (estimated effective ₹{e.get('effective_price',0):,.2f}), "
        f"P&L {e.get('ret_pct',0):+.2f}%</div>" for e in exits
    ) or "<div class='muted'>No live exit actions.</div>"
    tp_html = "".join(
        f"<div class='tp'>{html.escape(t['ticker'])} L{t['layer']} — partial TP at LTP ₹{t['price']:,.2f}, credited ₹{t['freed_cash_inr']:,.2f}</div>" for t in tps
    )
    sig_rows = "".join(
        f"<tr><td>{html.escape(x['ticker'])}</td><td>₹{x['close']:,.2f}</td><td>{x['adx']:.2f}</td>"
        f"<td>{'YES' if x['qualifying'] else 'NO'}</td><td>{x['quote_age_min']:.1f}m</td><td>{x['layers']}</td></tr>" for x in current_signals
    ) or "<tr><td colspan='6' class='muted'>No raw triggers on the live snapshot bar.</td></tr>"
    wl_rows = []
    for item in state["watchlist"][:30]:
        t = item["ticker"]; px = snapshot.get(t, {}).get("price", item.get("trigger_price",0.0))
        pullback = (item.get("trigger_price",0.0)-px)/max(1e-9,item.get("trigger_price",0.0))*100.0
        age = latest_idx-int(item.get("signal_bar",latest_idx))
        wl_rows.append(f"<tr><td>{html.escape(t)}</td><td>{item.get('signal_date','--')}</td><td>₹{item.get('trigger_price',0):,.2f}</td>"
                       f"<td>₹{px:,.2f}</td><td>{pullback:+.2f}%</td><td>{age}</td><td>₹{item.get('shadow_stop',0):,.2f}</td></tr>")
    body = f"""<!doctype html><html><head><style>
    body{{font-family:Arial,sans-serif;background:#f8fafc;color:#0f172a;padding:20px}}
    .card{{max-width:980px;margin:auto;background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:22px}}
    .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}} .stat{{background:#f1f5f9;padding:12px;border-radius:8px}}
    .buy,.sell,.tp,.retro{{padding:9px;margin:6px 0;border-radius:6px}} .buy{{background:#ecfdf5;color:#166534}} .sell{{background:#fef2f2;color:#991b1b}} .tp{{background:#eff6ff;color:#1d4ed8}}
    .muted{{color:#64748b}} table{{width:100%;border-collapse:collapse;font-size:12px}} th,td{{padding:7px;border-bottom:1px solid #e2e8f0;text-align:right}} th:first-child,td:first-child{{text-align:left}}
    </style></head><body><div class='card'>
    <h2>📈 NSE V11.2 Live Snapshot Companion — {html.escape(UNIVERSE_NAME)}</h2>
    <div class='muted'>V10.1 parity indicators | live snapshot bar {latest_date} | current LTP treated as Close | winner {html.escape(str(winner_meta.get('version','fallback')))}</div>
    <div class='grid'><div class='stat'><b>Cash</b><br>₹{state['cash_inr']:,.2f}</div><div class='stat'><b>Open Market</b><br>₹{current_market:,.2f}</div>
    <div class='stat'><b>Equity</b><br>₹{equity:,.2f}</div><div class='stat'><b>Open Layers</b><br>{sum(len(v) for v in state['positions'].values())}</div></div>
    <h3>Today's live actions</h3>{buy_html}{exit_html}{tp_html}
    <h3>Open holdings</h3><table><tr><th>Ticker</th><th>Entry</th><th>Entry LTP</th><th>Units</th><th>Current LTP</th><th>P&L ₹</th><th>P&L %</th><th>Stop</th></tr>
    {''.join(position_rows) or "<tr><td colspan='8' class='muted'>No open positions.</td></tr>"}</table>
    <h3>Live snapshot signals</h3><table><tr><th>Ticker</th><th>LTP / Close</th><th>ADX</th><th>Qualifies</th><th>Quote Age</th><th>Layers</th></tr>{sig_rows}</table>
    <h3>Watchlist ({len(state['watchlist'])})</h3><table><tr><th>Ticker</th><th>Signal</th><th>Trigger</th><th>Current</th><th>Pullback</th><th>Age</th><th>Shadow Stop</th></tr>
    {''.join(wl_rows) or "<tr><td colspan='7' class='muted'>Watchlist empty.</td></tr>"}</table>
    <div class='muted'>Live quotes: {quote_count}/{len(market)+1}. Stale beyond {LIVE_QUOTE_MAX_AGE_MINUTES:.1f}m: {len(stale)}. Raw entry triggers on today's live snapshot: {raw_signals}. No broker/order API is called.</div>
    </div></body></html>"""
    send_email(f"📈 NSE V11.2 [{UNIVERSE_NAME}] LIVE {latest_date} [{len(buys)} buys | {len(exits)} exits | {len(tps)} TP | ₹{equity:,.0f}]", body)


def main() -> None:
    try:
        run_cycle()
    except Exception:
        tb = traceback.format_exc()
        logger.error("Fatal NSE live companion crash:\n%s", tb)
        send_email(
            f"⚠️ NSE V10.1-Parity Bot CRASHED — {datetime.now(IST):%Y-%m-%d}",
            f"<pre>{html.escape(tb)}</pre>",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        DRY_RUN = True
    main()
