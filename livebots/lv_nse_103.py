%%writefile nse_live_companion.py
#!/usr/bin/env python3
"""
NSE EQUITY LIVE SNAPSHOT SWING-TRADING COMPANION
Hardcoded UnifiedSwingBot Colab Drive Paths
"""
from __future__ import annotations

import argparse
import copy
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
from typing import Any, Dict, List, Optional, Tuple
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
# HARDCODED UNIFIED PATHS
# ==============================================================================
UNIVERSE_NAME = "NIFTY50"
IST = ZoneInfo("Asia/Kolkata")

BASE_DIR = Path("/content/drive/MyDrive/UnifiedSwingBot")
DATA_DIR = BASE_DIR / "nse" / "data_cache"
STATE_PATH = BASE_DIR / "nse" / f"state_{UNIVERSE_NAME.lower()}.json"
WINNER_PATH = BASE_DIR / "winners" / UNIVERSE_NAME / "winner.json"
LOG_FILE = BASE_DIR / "nse" / "nse_live.log"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Risk & Execution Constants
LIQUIDITY_FLOOR_INR = 0.0
TRANCHE_FLOOR_INR = 10_000.0
DEFAULT_MAX_CONCURRENT_TRANCHES = 8
MAX_POSITION_EQUITY_PCT = 0.25
MAX_ADV_PARTICIPATION = 0.015
WL_MAX_AGE_BARS = 15

BASE_SLIPPAGE_BPS = 5.0
IMPACT_COEF_BPS = 80.0

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

LIVE_QUOTE_MAX_AGE_MINUTES = 10.0
LIVE_INTRADAY_INTERVAL = "1m"
MACRO_INDEX_TICKER = "^NSEI"
START_YEAR = 2008
MIN_HISTORY_DAYS = 300
FETCH_BATCH_SIZE = 40

# ==============================================================================
# LOGGING & MAILER
# ==============================================================================
logger = logging.getLogger("nse_live")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def send_email(subject: str, body: str) -> None:
    user = os.environ.get("GMAIL_USER")
    pw = os.environ.get("GMAIL_APP_PASSWORD")
    to = os.environ.get("RECIPIENT_EMAIL")
    if not user or not pw or not to:
        logger.warning("Missing Gmail credentials in environment; email skipped.")
        return
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    mime = MIMEMultipart("alternative")
    mime["Subject"], mime["From"], mime["To"] = subject, user, to
    mime.attach(MIMEText(body, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as server:
            server.login(user, pw)
            server.sendmail(user, to, mime.as_string())
        logger.info("Email delivered: %s", subject)
    except Exception as e:
        logger.error("Email send failed: %s", e)


# ==============================================================================
# AUDITED FEE & FILL ENGINE
# ==============================================================================
def _fee_dict() -> Dict[str, float]:
    return {
        "brokerage": 0.0, "stt": 0.0, "exchange_charges": 0.0,
        "stamp_duty": 0.0, "sebi_charges": 0.0, "gst": 0.0,
        "dp_charges": 0.0, "slippage_cost": 0.0
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
        "dp_charges": 0.0, "slippage_cost": slip_cost
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
        "dp_charges": dp, "slippage_cost": slip_cost
    }
    return max(0.0, gross_inr - brokerage - stt - exch - sebi - gst - dp), f


def compute_max_affordable_tranche(cash: float, adv_30d: float) -> float:
    clean_adv = adv_30d if np.isfinite(adv_30d) else 1_000_000.0
    adv = max(clean_adv, 1_000_000.0)
    usable_cash = max(0.0, cash)
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


def _whole_share_buy(cash: float, target_inr: float, ref_open: float, adv_30d: float):
    if not np.isfinite(cash) or cash <= 0 or not np.isfinite(target_inr) or target_inr < TRANCHE_FLOOR_INR:
        return None
    if not np.isfinite(ref_open) or ref_open <= 0:
        return None
    adv = max(float(adv_30d) if np.isfinite(adv_30d) else 1_000_000.0, 1_000_000.0)
    tranche = min(compute_max_affordable_tranche(cash, adv), target_inr)
    if tranche < TRANCHE_FLOOR_INR:
        return None
    slip_mult = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(min(1.0, max(0.0, tranche / adv)))) / 10000.0
    units = int(tranche // (ref_open * (1.0 + slip_mult)))
    if units <= 0:
        return None
    for _ in range(10):
        fill_px = ref_open * (1.0 + slip_mult)
        gross = units * fill_px
        slip_cost = units * (fill_px - ref_open)
        total_cost, fees = compute_buy_cost_audited(gross, slip_cost)
        if total_cost <= cash + 1e-9 and total_cost <= target_inr + 1e-9:
            return fill_px, units, total_cost, fees, tranche
        units -= 1
        if units <= 0:
            return None
    return None


FALLBACK_PARAMS: Dict[str, Any] = {
    "entry_type": 1, "adx_thresh": 35.0, "rsi_f_len": 76, "rsi_f_smt": 10,
    "rsi_s_len": 74, "rsi_s_smt": 6, "use_rsi_trend_filter": False,
    "use_market_macro_system": True, "macro_ma_len": 167, "macro_ma_type": 2,
    "macro_active_exit": False, "max_concurrent_tranches": 5, "wl_mode": "WL_STRONGEST_MOMENTUM",
    "use_global_tp": False, "be_trigger_atr": 3.5, "max_holding_bars": 55, "sl_mult": 5.6,
    "exit_type": 0, "trail_atr_mult": 0.0, "max_pyramid_layers": 1,
}


def load_strategy_params() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if WINNER_PATH.exists():
        try:
            with WINNER_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            p = data.get("best_params") or data.get("params")
            if p:
                p = reconstitute_params(p, max_xover_long=200, max_exit_xover_long=200)
                logger.info("Loaded NSE champion from %s", WINNER_PATH)
                return p, data
        except Exception as e:
            logger.warning("Winner load failed: %s", e)
    logger.warning("No NSE winner.json found at %s; using fallback.", WINNER_PATH)
    return reconstitute_params(FALLBACK_PARAMS), {}


def _blank_state() -> Dict[str, Any]:
    return {"cash_inr": 0.0, "positions": {}, "watchlist": [], "consumed_signal_keys": [], "last_processed_date": None}


def _normalize_position(t: str, pos: Dict[str, Any], dates: pd.DatetimeIndex) -> Dict[str, Any]:
    p = dict(pos)
    entry_price = float(p.get("entry_price", 0.0))
    invested = float(p.get("cost_inr", 0.0))
    units = int(p.get("units", 0))
    if units <= 0 and entry_price > 0 and invested > 0:
        units = max(1, int(invested // entry_price))
    entry_date = str(p.get("entry_date", dates[0].date()))[:10]
    matches = np.where(dates.strftime("%Y-%m-%d") == entry_date)[0]
    entry_bar = int(p.get("entry_bar", matches[0] if len(matches) else 0))
    return {
        "ticker": t, "entry_bar": entry_bar, "entry_date": entry_date,
        "entry_price": entry_price, "initial_units": int(p.get("initial_units", units)),
        "units": units, "cost_inr": invested if invested > 0 else units * entry_price,
        "entry_atr": float(p.get("entry_atr", max(entry_price * 0.02, 0.01))),
        "current_sl": float(p.get("current_sl", entry_price * 0.9)),
        "stop_reason": p.get("stop_reason", "STOP_LOSS"),
        "highest_high": float(p.get("highest_high", entry_price)),
        "lowest_low": float(p.get("lowest_low", entry_price)),
        "peak_bar": int(p.get("peak_bar", entry_bar)), "trough_bar": int(p.get("trough_bar", entry_bar)),
        "layer": int(p.get("layer", 1)), "tp_done": bool(p.get("tp_done", False)),
        "tp_proceeds": float(p.get("tp_proceeds", 0.0)), "proceeds": float(p.get("proceeds", 0.0)),
        "fee_acc": {k: float(v) for k, v in p.get("fee_acc", _fee_dict()).items()},
    }


def load_state(dates: pd.DatetimeIndex) -> Dict[str, Any]:
    state = _blank_state()
    if STATE_PATH.exists():
        try:
            with STATE_PATH.open("r", encoding="utf-8") as f:
                old = json.load(f)
            state["cash_inr"] = float(old.get("cash_inr", 0.0))
            state["positions"] = {t: [_normalize_position(t, x, dates) for x in xs] for t, xs in old.get("positions", {}).items()}
            wl = old.get("watchlist", [])
            state["watchlist"] = list(wl.values()) if isinstance(wl, dict) else list(wl)
            state["consumed_signal_keys"] = list(old.get("consumed_signal_keys", []))
            state["last_processed_date"] = old.get("last_processed_date")
        except Exception as e:
            logger.warning("NSE state load failed: %s", e)
    return state


def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.replace(STATE_PATH)


# ==============================================================================
# DATA & INTRADAY SNAPSHOT
# ==============================================================================
def fetch_stock_universe() -> List[str]:
    url = "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv"
    try:
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0"})
        r = s.get(url, timeout=15)
        df = pd.read_csv(io.StringIO(r.text))
        col = next(c for c in ("Symbol", "SYMBOL", "symbol") if c in df.columns)
        return sorted({f"{str(x).strip().upper()}.NS" for x in df[col] if str(x).strip()})
    except Exception:
        return [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LICI.NS",
            "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS", "HCLTECH.NS", "ASIANPAINT.NS"
        ]


def fetch_live_nse_snapshot(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    if not tickers:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    try:
        raw = yf.download(
            tickers, period="1d", interval="1m", progress=False, threads=False, group_by="ticker"
        )
    except Exception as e:
        logger.warning("NSE live 1m download failed: %s", e)
        return {}

    now = datetime.now(IST)
    for ticker in tickers:
        try:
            frame = raw[ticker].dropna(subset=["Open", "Close"]) if isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.levels[0] else raw.dropna(subset=["Open", "Close"])
            if frame.empty:
                continue
            out[ticker] = {
                "open": float(frame["Open"].iloc[0]),
                "high": float(frame["High"].max()),
                "low": float(frame["Low"].min()),
                "price": float(frame["Close"].iloc[-1]),
                "volume": float(frame["Volume"].fillna(0.0).sum()),
                "quote_time": now.isoformat(),
            }
        except Exception:
            continue
    return out


def append_nse_live_snapshot(raw_data: Dict[str, pd.DataFrame], macro_df: pd.DataFrame, snapshot: Dict[str, Dict[str, Any]]) -> Tuple[pd.Timestamp, pd.DataFrame]:
    live_date = pd.Timestamp(datetime.now(IST).date()).normalize()

    def add_row(df: pd.DataFrame, q: Optional[Dict[str, Any]]) -> pd.DataFrame:
        row = {
            "open": np.nan if q is None else q["open"],
            "high": np.nan if q is None else q["high"],
            "low": np.nan if q is None else q["low"],
            "close": np.nan if q is None else q["price"],
            "volume": 0.0 if q is None else q["volume"],
        }
        x = pd.DataFrame(row, index=pd.DatetimeIndex([live_date]))
        z = df[df.index < live_date]
        return pd.concat([z, x]).sort_index()

    for ticker in list(raw_data):
        raw_data[ticker] = add_row(raw_data[ticker], snapshot.get(ticker))
    macro_df = add_row(macro_df, snapshot.get(MACRO_INDEX_TICKER))
    return live_date, macro_df


def fetch_price_data(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    results = {}
    cutoff = (datetime.now(IST) - timedelta(days=1)).date()
    for t in tickers:
        p = DATA_DIR / f"{t.replace('^','IDX_')}.parquet"
        if p.exists():
            try:
                df = pd.read_parquet(p)
                if df.index.max().date() >= cutoff:
                    results[t] = df
                    continue
            except Exception:
                pass
        try:
            d = yf.download(t, start=f"{START_YEAR}-01-01", progress=False, auto_adjust=True)
            if not d.empty and len(d) >= MIN_HISTORY_DAYS:
                d = d.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
                d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
                d = d[d.index.date <= cutoff]
                d.to_parquet(p)
                results[t] = d
        except Exception:
            continue
    return results


def build_live_matrix(raw_data, macro_df):
    master_dates = pd.DatetimeIndex(sorted(pd.to_datetime(macro_df.index).unique())).normalize()
    market = {}
    for ticker, df in raw_data.items():
        re = df.reindex(master_dates)
        alive = (~re["close"].isna()).to_numpy(bool)
        re = re.ffill()
        c, o, h, l, v = re["close"].to_numpy(float), re["open"].to_numpy(float), re["high"].to_numpy(float), re["low"].to_numpy(float), re["volume"].fillna(0.0).to_numpy(float)
        dvol = np.nan_to_num(np_rolling_mean(c * v, 30), nan=1_000_000.0)
        market[ticker] = {"alive": alive, "open": o, "high": h, "low": l, "close": c, "volume": v, "dvol": dvol}
    return master_dates, market


# ==============================================================================
# PREMIUM REPORTING UI (MATCHING CRYPTO INTELLIGENCE FORMAT)
# ==============================================================================
def _money_inr(v: float, decimals: int = 2) -> str:
    try: return f"₹{float(v):,.{decimals}f}"
    except Exception: return "₹0.00"

def _pct(v: float, decimals: int = 2) -> str:
    try: return f"{float(v):+,.{decimals}f}%"
    except Exception: return "0.00%"

def _safe_pct(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return 0.0
    return float(numerator / denominator * 100.0)

def _pnl_color(v: float) -> str:
    if v > 0: return "#22c55e"
    if v < 0: return "#ef4444"
    return "#94a3b8"

def _signal_badge(active: bool, yes_text: str = "ACTIVE", no_text: str = "INACTIVE") -> str:
    if active:
        return f"<span class='badge badge-green'><span class='dot'></span>{html.escape(yes_text)}</span>"
    return f"<span class='badge badge-gray'><span class='dot'></span>{html.escape(no_text)}</span>"

def _risk_badge(cushion_pct: float) -> str:
    if cushion_pct <= 0: cls, label = "badge-red", "AT / BELOW EXIT"
    elif cushion_pct < 3: cls, label = "badge-orange", "TIGHT"
    elif cushion_pct < 7: cls, label = "badge-yellow", "WATCH"
    else: cls, label = "badge-green", "HEALTHY"
    return f"<span class='badge {cls}'>{label}</span>"

def _holding_bar(bars_held: int, max_bars: int) -> str:
    if max_bars <= 0: return ""
    pct = min(100.0, max(0.0, bars_held / max_bars * 100.0))
    cls = "bar-red" if pct >= 90 else ("bar-orange" if pct >= 70 else "bar-green")
    return f"""<div class="holding-wrap"><div class="holding-track"><div class="holding-fill {cls}" style="width:{pct:.1f}%"></div></div><span>{bars_held}/{max_bars} bars</span></div>"""

def _explain_entry_signal(p: Dict[str, Any]) -> str:
    parts = [f"Entry type {p.get('entry_type', 'configured')}"]
    if p.get("adx_thresh") is not None: parts.append(f"ADX threshold {p['adx_thresh']}")
    if p.get("vol_ma_len") is not None and p.get("vol_mult") is not None:
        parts.append(f"Volume confirmation: {p['vol_ma_len']}-bar avg × {p['vol_mult']}")
    if p.get("price_lookback") is not None: parts.append(f"Price lookback: {p['price_lookback']} bars")
    if p.get("body_atr_mult") is not None: parts.append(f"Body ≥ {p['body_atr_mult']}× ATR")
    parts.append("Macro filter enabled" if p.get("use_market_macro_system", False) else "Macro filter disabled")
    return "; ".join(parts)

def _explain_exit_signal(p: Dict[str, Any]) -> str:
    parts = [f"Exit type {p.get('exit_type', 'configured')}"]
    if p.get("max_holding_bars") is not None: parts.append(f"Max hold {p['max_holding_bars']} bars")
    if p.get("sl_mult") is not None: parts.append(f"Protective stop {p['sl_mult']}× ATR")
    if p.get("trail_atr_mult", 0): parts.append(f"Trailing component {p['trail_atr_mult']}× ATR")
    if p.get("use_global_tp", False): parts.append("Global TP enabled")
    return "; ".join(parts)


def render_report(p, state, market, snapshot, idx, today, buys, exits, signals, macro_ok):
    now_ist = datetime.now(IST)
    free_cash = float(state.get("cash_inr", 0.0))
    max_slots = int(p.get("max_concurrent_tranches", DEFAULT_MAX_CONCURRENT_TRANCHES))
    
    invested_inr = 0.0
    market_val_inr = 0.0
    unrealized_pnl = 0.0
    position_count = 0
    portfolio_rows = []

    for ticker in sorted(state.get("positions", {})):
        q = snapshot.get(ticker)
        m = market.get(ticker)
        px = float(q.get("price", 0.0)) if q else 0.0

        for pos in state["positions"][ticker]:
            position_count += 1
            cost = float(pos.get("cost_inr", 0.0))
            units = int(pos.get("units", 0))

            if px > 0:
                cur_val = units * px
                pnl = (float(pos.get("proceeds", 0.0)) + cur_val) - cost
                invested_inr += cost
                market_val_inr += cur_val
                unrealized_pnl += pnl
                pnl_pct = _safe_pct(pnl, cost)
                stop_px = float(pos.get("current_sl", 0.0))
                cushion_inr = px - stop_px
                cushion_pct = _safe_pct(cushion_inr, px)
                bars_held = max(0, idx - int(pos.get("entry_bar", idx)))
                max_hold = int(p.get("max_holding_bars", 55))

                strategy_exit_active = False
                if m is not None:
                    try: strategy_exit_active = bool(m["sig"].exit_sig[idx])
                    except Exception: pass

                alloc_pct = _safe_pct(cur_val, max(1e-9, free_cash + market_val_inr))
                pnl_html = f"<span style='color:{_pnl_color(pnl_pct)};font-weight:800'>{_pct(pnl_pct)}</span>"
                strategy_exit_html = _signal_badge(strategy_exit_active, "EXIT SIGNAL", "HOLD")
                cushion_html = f"<div class='cushion-value'>{_money_inr(cushion_inr)}</div><div class='muted'>{cushion_pct:+.2f}% from exit</div>{_risk_badge(cushion_pct)}"
                exit_dist_text = "Price at/below protective exit" if cushion_pct <= 0 else f"{_money_inr(cushion_inr)} above protective exit"

                portfolio_rows.append(f"""
                <tr>
                    <td><div class="coin-name">{html.escape(ticker)}</div><div class="muted">Layer {pos.get('layer', 1)}</div></td>
                    <td><div>{_money_inr(pos.get('entry_price', 0.0))}</div><div class="muted">{html.escape(str(pos.get('entry_date', '-')))}</div></td>
                    <td><div>{_money_inr(px)}</div><div class="muted">3:15 Snapshot</div></td>
                    <td><div>{_money_inr(cost)}</div><div class="muted">{units:,} shares</div></td>
                    <td><div>{_money_inr(cur_val)}</div><div>{pnl_html}</div></td>
                    <td><div class="exit-price">{_money_inr(stop_px)}</div><div class="muted">{html.escape(str(pos.get('stop_reason', 'Protective exit')))}</div></td>
                    <td>{cushion_html}<div class="muted exit-distance">{html.escape(exit_dist_text)}</div></td>
                    <td>{strategy_exit_html}<div style="margin-top:7px">{_holding_bar(bars_held, max_hold)}</div></td>
                    <td><div>{alloc_pct:.1f}%</div><div class="muted">of portfolio</div></td>
                </tr>""")
            else:
                portfolio_rows.append(f"""
                <tr>
                    <td><div class="coin-name">{html.escape(ticker)}</div><div class="muted">Layer {pos.get('layer', 1)}</div></td>
                    <td colspan="8"><span class="badge badge-orange">LIVE PRICE UNAVAILABLE</span><div class="muted">Position remains open; current quote was missing during snapshot.</div></td>
                </tr>""")

    total_equity = free_cash + market_val_inr
    cash_pct = _safe_pct(free_cash, max(1e-9, total_equity))
    invested_pct = _safe_pct(invested_inr, max(1e-9, total_equity))
    free_slots = max(0, max_slots - position_count)
    macro_pass = bool(macro_ok[idx]) if len(macro_ok) > idx else False

    buy_cards = "".join(f"""
        <div class="action-card buy-card">
            <div class="action-icon">↗</div>
            <div class="action-content">
                <div class="action-title">BUY <span>{html.escape(b['ticker'])}</span> · Layer {b.get('layer', 1)}</div>
                <div class="action-main">{_money_inr(b['price'])}</div>
                <div class="action-meta">Fill: {_money_inr(b['effective_price'])} &nbsp;·&nbsp; Units: {b['units']:,} &nbsp;·&nbsp; Cost: {_money_inr(b['cost'])}</div>
            </div>
        </div>""" for b in buys) or "<div class='empty-card'>No new buy executions in this cycle.</div>"

    exit_cards = "".join(f"""
        <div class="action-card exit-card">
            <div class="action-icon">↘</div>
            <div class="action-content">
                <div class="action-title">EXIT <span>{html.escape(e['ticker'])}</span> · Layer {e.get('layer', 1)}</div>
                <div class="action-main">{_money_inr(e['exit_price'])}</div>
                <div class="action-meta">Reason: <strong>{html.escape(e['reason'])}</strong> &nbsp;·&nbsp; P&L: <strong>{_pct(e.get('ret_pct', 0))}</strong></div>
            </div>
        </div>""" for e in exits) or "<div class='empty-card'>No new exits in this cycle.</div>"

    signal_rows = "".join(f"""
        <tr>
            <td><div class="coin-name">{html.escape(str(ev.get('ticker', '')))}</div><div class="muted">{ev.get('signal_date', today)}</div></td>
            <td>{_signal_badge(ev.get('status') == 'EXECUTED', 'EXECUTED', ev.get('status', 'BLOCKED'))}</td>
            <td>{_money_inr(ev.get('signal_price', 0.0))}</td>
            <td>{_money_inr(ev.get('live_price', 0.0))}</td>
            <td><strong>{html.escape(str(ev.get('status', 'DETECTED')))}</strong></td>
            <td>{html.escape(str(ev.get('reason', 'Strategy trigger')))}</td>
        </tr>""" for ev in signals) or "<tr><td colspan='6' class='empty-table'>No new entry signals detected on today's 3:15 PM snapshot.</td></tr>"

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
        <div class="brand">UNIFIED SWING BOT · NSE EQUITY</div>
        <div class="title">📈 NIFTY50 Daily Intelligence</div>
        <div class="subtitle">Snapshot bar: <strong>{html.escape(today)} 3:15 PM IST</strong> &nbsp;·&nbsp; Execution: <strong>{now_ist:%d %b %Y %H:%M:%S IST}</strong></div>
        <div class="status-row">
            {_signal_badge(position_count > 0, f"{position_count} OPEN POSITIONS", "NO OPEN POSITIONS")} &nbsp;
            {_signal_badge(len(buys) > 0, f"{len(buys)} BUY EXECUTED", "NO NEW BUY")} &nbsp;
            {_signal_badge(len(exits) > 0, f"{len(exits)} EXIT EXECUTED", "NO NEW EXIT")}
        </div>
    </div>
    <div class="grid">
        <div class="card"><div class="card-label">Total Equity</div><div class="card-value">{_money_inr(total_equity)}</div><div class="card-sub">Free cash + positions mark-to-market</div></div>
        <div class="card"><div class="card-label">Free Cash</div><div class="card-value">{_money_inr(free_cash)}</div><div class="card-sub">{cash_pct:.1f}% of equity · immediately deployable</div></div>
        <div class="card"><div class="card-label">Invested Capital</div><div class="card-value">{_money_inr(invested_inr)}</div><div class="card-sub">{invested_pct:.1f}% of portfolio cost basis</div></div>
        <div class="card"><div class="card-label">Active P&L</div><div class="card-value" style="color:{_pnl_color(_safe_pct(unrealized_pnl, invested_inr))}">{_money_inr(unrealized_pnl)}</div><div class="card-sub">{_pct(_safe_pct(unrealized_pnl, invested_inr))} across open tranches</div></div>
    </div>
    <div class="section">
        <div class="section-title">Market & Engine Status</div>
        <div class="section-description">Macro benchmark context used during this 3:15 PM snapshot pass.</div>
        <div class="grid">
            <div class="card"><div class="card-label">^NSEI LTP</div><div class="card-value">{_money_inr(float(snapshot.get(MACRO_INDEX_TICKER, {}).get('price', 0.0)))}</div><div class="card-sub">Nifty 50 Index</div></div>
            <div class="card"><div class="card-label">Macro Regime</div><div class="card-value">{'RISK-ON' if macro_pass else 'RISK-OFF'}</div><div class="card-sub">{'Enabled' if p.get('use_market_macro_system') else 'Disabled'}</div></div>
            <div class="card"><div class="card-label">Open Slots</div><div class="card-value">{free_slots} <span style="font-size:13px;color:#64748b">/ {max_slots}</span></div><div class="card-sub">{position_count} active tranche(s)</div></div>
            <div class="card"><div class="card-label">Universe</div><div class="card-value">NIFTY 50</div><div class="card-sub">Audited institutional sizing</div></div>
        </div>
    </div>
    <div class="section">
        <div class="section-title">Signal Engine</div>
        <div class="section-description">Strategy parameters governing entry qualification and risk management.</div>
        <div class="two-col">
            <div class="info-box"><div class="info-label">ENTRY SIGNAL</div><div class="info-value"><strong>Evaluated on the 3:15 PM LTP snapshot treated as today's close.</strong><br><br>{html.escape(entry_explanation)}</div></div>
            <div class="info-box"><div class="info-label">EXIT SIGNAL</div><div class="info-value"><strong>Exits execute dynamically on strategy signals, time stops, or protective stops.</strong><br><br>{html.escape(exit_explanation)}</div></div>
        </div>
    </div>
    <div class="section">
        <div class="section-title">Today's Executed Actions</div>
        <div class="section-description">Trades generated during this 3:15 PM snapshot cycle.</div>
        <h4 style="color:#4ade80;margin-bottom:8px">BUY ACTIVITY</h4>
        {buy_cards}
        <h4 style="color:#f87171;margin-top:20px;margin-bottom:8px">EXIT ACTIVITY</h4>
        {exit_cards}
    </div>
    <div class="section">
        <div class="section-title">Open Portfolio</div>
        <div class="section-description">Live valuation, dynamic cushions, and protective stops for held positions.</div>
        <table>
            <thead>
                <tr>
                    <th>Symbol</th><th>Buy Price</th><th>3:15 LTP</th><th>Invested</th><th>Market Value / P&L</th><th>Protective Exit</th><th>Cushion</th><th>Risk / Hold</th><th>Allocation</th>
                </tr>
            </thead>
            <tbody>
                {''.join(portfolio_rows) if portfolio_rows else '<tr><td colspan="9" class="empty-table">No open positions.</td></tr>'}
            </tbody>
        </table>
    </div>
    <div class="section">
        <div class="section-title">Signal Decisions</div>
        <div class="section-description">Audit log of qualifying setups and execution decisions today.</div>
        <table>
            <thead>
                <tr><th>Symbol</th><th>Status</th><th>Signal Price</th><th>Live Price</th><th>Decision</th><th>Reason</th></tr>
            </thead>
            <tbody>{signal_rows}</tbody>
        </table>
    </div>
    <div class="section">
        <div class="section-title">Capital Allocation Summary</div>
        <div class="two-col">
            <div class="info-box">
                <div class="info-label">CAPITAL SUMMARY</div>
                <div class="summary-line"><span class="summary-label">Total Equity</span><span class="summary-value">{_money_inr(total_equity)}</span></div>
                <div class="summary-line"><span class="summary-label">Free Cash</span><span class="summary-value">{_money_inr(free_cash)}</span></div>
                <div class="summary-line"><span class="summary-label">Invested Capital</span><span class="summary-value">{_money_inr(invested_inr)}</span></div>
                <div class="summary-line"><span class="summary-label">Unrealized P&L</span><span class="summary-value" style="color:{_pnl_color(_safe_pct(unrealized_pnl, invested_inr))}">{_money_inr(unrealized_pnl)}</span></div>
            </div>
            <div class="info-box">
                <div class="info-label">CAPACITY</div>
                <div class="summary-line"><span class="summary-label">Max Allowed Tranches</span><span class="summary-value">{max_slots}</span></div>
                <div class="summary-line"><span class="summary-label">Active Tranches</span><span class="summary-value">{position_count}</span></div>
                <div class="summary-line"><span class="summary-label">Available Slots</span><span class="summary-value">{free_slots}</span></div>
                <div class="summary-line"><span class="summary-label">Cash Allocation</span><span class="summary-value">{cash_pct:.1f}%</span></div>
            </div>
        </div>
    </div>
    <div class="footer">UnifiedSwingBot · NSE Live Companion<br>Decision-support snapshot only. No exchange order API is called.</div>
</div>
</body>
</html>"""
    return body, total_equity


# ==============================================================================
# MAIN CYCLE (RUNS AT 3:15 PM IST)
# ==============================================================================
def run_cycle():
    p, meta = load_strategy_params()
    tickers = fetch_stock_universe()
    if not tickers:
        raise RuntimeError("NIFTY50 universe is empty.")

    raw = fetch_price_data(tickers + [MACRO_INDEX_TICKER])
    macro_df = raw.pop(MACRO_INDEX_TICKER, None)
    if macro_df is None or macro_df.empty:
        raise RuntimeError("Missing ^NSEI benchmark data.")

    snapshot = fetch_live_nse_snapshot(tickers + [MACRO_INDEX_TICKER])
    if MACRO_INDEX_TICKER not in snapshot:
        raise RuntimeError("Live ^NSEI quote not found. Aborting before state mutation.")

    live_date, macro_df = append_nse_live_snapshot(raw, macro_df, snapshot)
    dates, market = build_live_matrix(raw, macro_df)
    if len(dates) < 2:
        raise RuntimeError("Insufficient NSE matrix history.")

    idx = len(dates) - 1
    today = str(live_date.date())
    mc = macro_df.reindex(dates).ffill()["close"].to_numpy(float)

    if p.get("use_market_macro_system", False):
        mma = FastIndicators.moving_average(mc, p["macro_ma_len"], p["macro_ma_type"])
        macro_ok = (~np.isnan(mma)) & (mc > mma)
    else:
        macro_ok = np.ones(len(dates), dtype=bool)

    for t, m in market.items():
        m["sig"] = compile_single_symbol(
            m["open"], m["high"], m["low"], m["close"], m["volume"], m["dvol"],
            m["alive"], macro_ok, p, LIQUIDITY_FLOOR_INR
        )

    state = load_state(dates)
    state.setdefault("positions", {})
    state.setdefault("consumed_signal_keys", [])
    consumed = set(state["consumed_signal_keys"])

    buys, exits, signals = [], [], []
    maxslots = int(p.get("max_concurrent_tranches", DEFAULT_MAX_CONCURRENT_TRANCHES))
    maxlayers = int(p.get("max_pyramid_layers", 1))

    # PASS A: Existing-position exits & dynamic trailing updates
    for t in list(state["positions"]):
        m = market.get(t)
        q = snapshot.get(t)
        if not m or not q:
            logger.warning("Skipping %s risk evaluation: live quote unavailable.", t)
            continue
        px = float(q.get("price", 0.0))
        if not np.isfinite(px) or px <= 0:
            continue

        adv = max(float(m["dvol"][max(0, idx - 1)]), 1_000_000.0)
        prev_atr = float(m["sig"].atr[max(0, idx - 1)]) if np.isfinite(m["sig"].atr[max(0, idx - 1)]) else float(m["sig"].atr[idx])

        for pos in list(state["positions"][t]):
            held = max(0, idx - int(pos["entry_bar"]))
            stop = float(pos.get("current_sl", 0.0))
            reason = None

            if held >= int(p.get("max_holding_bars", 55)):
                reason = "MAX_HOLDING_TIME"
            elif int(p["exit_type"]) in (3, 4, 5, 6, 7) and bool(m["sig"].exit_sig[idx]):
                reason = f"STRATEGY_EXIT_{p['exit_type']}"
            elif not np.isfinite(stop) or stop <= 0:
                reason = "INVALID_PROTECTIVE_STOP"
            elif px <= stop:
                reason = pos.get("stop_reason", "STOP_LOSS")

            if reason:
                units = int(pos.get("units", 0))
                if units <= 0:
                    continue
                slip = (BASE_SLIPPAGE_BPS + IMPACT_COEF_BPS * np.sqrt(min(1.0, (units * px) / adv))) / 10000.0
                fill, proceeds, fee = _sell_fill_audited(units, px, slip)
                state["cash_inr"] += proceeds
                pos["proceeds"] += proceeds
                _merge_fees(pos["fee_acc"], fee)
                pnl = pos["proceeds"] - pos["cost_inr"]
                exits.append({
                    "ticker": t, "layer": pos["layer"], "reason": reason,
                    "exit_price": px, "pnl": pnl, "ret_pct": _safe_pct(pnl, pos["cost_inr"])
                })
                state["positions"][t].remove(pos)
                continue

            # Update highest high & ratcheting stop loss
            if px > pos.get("highest_high", px):
                pos["highest_high"] = px
                pos["peak_bar"] = idx
            if px < pos.get("lowest_low", px):
                pos["lowest_low"] = px
                pos["trough_bar"] = idx

            be = p.get("be_trigger_atr", 0.0)
            if be > 0.0 and pos["current_sl"] < pos["entry_price"] * 1.002:
                if pos["highest_high"] >= pos["entry_price"] + be * pos["entry_atr"]:
                    pos["current_sl"] = max(pos["current_sl"], pos["entry_price"] * 1.002)
                    pos["stop_reason"] = "BREAKEVEN_SL"

            if p.get("trail_atr_mult", 0.0) > 0.0 and np.isfinite(prev_atr):
                floor = pos["highest_high"] - p["trail_atr_mult"] * prev_atr
                if floor > pos["current_sl"]:
                    pos["current_sl"], pos["stop_reason"] = floor, "TRAIL_ATR_STOP"

    state["positions"] = {t: v for t, v in state["positions"].items() if v}

    # PASS B & C: Signal evaluation & sizing against total portfolio equity
    count = lambda: sum(len(v) for v in state["positions"].values())

    for t, m in market.items():
        q = snapshot.get(t)
        if not q:
            continue
        px = float(q.get("price", 0.0))
        if not np.isfinite(px) or px <= 0 or not bool(m["sig"].entry[idx]):
            continue

        key = f"{t}|{today}"
        ev = {
            "ticker": t, "signal_type": "ENTRY", "signal_date": today,
            "signal_price": float(m["close"][idx]), "live_price": px,
            "status": "DETECTED", "reason": "Strategy entry signal detected"
        }

        if not bool(macro_ok[idx]):
            ev.update(status="BLOCKED_MACRO", reason="Market macro filter is risk-off")
            signals.append(ev)
            continue
        if key in consumed:
            ev.update(status="ALREADY_CONSUMED", reason="Signal was already executed")
            signals.append(ev)
            continue
        if count() >= maxslots:
            ev.update(status="BLOCKED_MAX_SLOTS", reason=f"Maximum {maxslots} open tranches reached")
            signals.append(ev)
            continue

        layers = len(state["positions"].get(t, []))
        if layers >= maxlayers:
            ev.update(status="BLOCKED_MAX_LAYERS", reason=f"Maximum {maxlayers} layer(s) allowed")
            signals.append(ev)
            continue

        # Anti-averaging-down guard for pyramid layers
        if layers > 0:
            highest_entry = max(x["entry_price"] for x in state["positions"][t])
            if px <= highest_entry * 1.005:
                ev.update(status="BLOCKED_PYRAMID_PRICE", reason="Current price is not above previous entry by 0.5%")
                signals.append(ev)
                continue

        # Dynamic slot target based on TOTAL portfolio equity
        current_active = sum(
            pos["units"] * float(snapshot.get(t_sym, {}).get("price", 0.0))
            for t_sym, pos_list in state["positions"].items()
            for pos in pos_list
        )
        total_equity = state["cash_inr"] + current_active
        max_pos_cap = total_equity * MAX_POSITION_EQUITY_PCT
        slots = max(1, maxslots - count())
        dynamic_target = min(max_pos_cap, state["cash_inr"] / float(slots))
        adv = max(float(m["dvol"][max(0, idx - 1)]), 1_000_000.0)

        if dynamic_target < TRANCHE_FLOOR_INR:
            ev.update(status="BLOCKED_POSITION_CAP", reason=f"Capped allocation {_money_inr(dynamic_target)} is below floor {_money_inr(TRANCHE_FLOOR_INR)}")
            signals.append(ev)
            continue

        fill = _whole_share_buy(state["cash_inr"], dynamic_target, px, adv)
        if not fill:
            ev.update(status="BLOCKED_AFFORDABILITY", reason="No whole-share fill fits cash, fees, and allocation cap")
            signals.append(ev)
            continue

        eff, units, cost, fees, _ = fill
        atr = float(m["sig"].atr[idx])
        stop = eff - float(p.get("sl_mult", 3.0)) * atr
        if not np.isfinite(atr) or atr <= 0 or not np.isfinite(stop) or stop <= 0:
            ev.update(status="BLOCKED_INVALID_STOP", reason="Computed protective stop is invalid")
            signals.append(ev)
            continue

        state["cash_inr"] -= cost
        layer = layers + 1
        pos = {
            "ticker": t, "entry_bar": idx, "entry_date": today, "entry_price": eff,
            "initial_units": units, "units": units, "cost_inr": cost, "entry_atr": atr,
            "current_sl": stop, "stop_reason": "STOP_LOSS", "highest_high": px, "lowest_low": px,
            "layer": layer, "fee_acc": fees, "tp_done": False, "tp_proceeds": 0.0, "proceeds": 0.0
        }
        state["positions"].setdefault(t, []).append(pos)
        consumed.add(key)
        buys.append({
            "ticker": t, "price": px, "effective_price": eff, "units": units,
            "cost": cost, "layer": layer, "signal_date": today
        })
        ev.update(status="EXECUTED", reason=f"Bought {units:,} shares; allocated {_money_inr(cost)}")
        signals.append(ev)

    state["consumed_signal_keys"] = sorted(consumed)[-500:]
    state["last_processed_date"] = today
    save_state(state)

    body, final_equity = render_report(p, state, market, snapshot, idx, today, buys, exits, signals, macro_ok)
    send_email(
        f"📈 NSE Intelligence | {len(buys)} BUY · {len(exits)} EXIT · {_money_inr(final_equity, 0)} Equity",
        body
    )
    return {"buys": buys, "exits": exits, "signals": signals, "equity": final_equity}


def main():
    try:
        run_cycle()
    except Exception:
        tb = traceback.format_exc()
        logger.error("Fatal NSE companion crash:\n%s", tb)
        send_email(f"⚠️ NSE Bot CRASHED — {datetime.now(IST):%Y-%m-%d}", f"<pre>{html.escape(tb)}</pre>")
        raise


if __name__ == "__main__":
    main()