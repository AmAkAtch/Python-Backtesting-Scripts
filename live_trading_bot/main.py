from __future__ import annotations
"""
Daily Swing-Trade Signal Bot — v6.0 (stocks + crypto)

One file, two independent engines sharing infrastructure (data fetching,
state, email, formatting). Stocks keep the original SMA/LMA crossover
engine unchanged. Crypto has been REPLACED with a 1:1 port of the
crypto_new_test/main.py backtester (engine v1.2.0, decision #17)'s live
logic: dual smoothed-RSI crossover entries, a standing BTC-regime filter,
the hybrid ATR TP(partial 50%)+breakeven+ATR-trail exit family, an
always-on panic-exit override, pyramiding into winners, and watchlist
candidates tracked as shadow positions (their own stop-loss/TP, removed
by the same exit condition that would sell a real position) -- so the bot
you run every day makes the exact same decisions the backtest validated.

Everything you'd normally want to tweak (money, strategy parameters,
which tickers to trade, manual position overrides) lives in the
CONFIGURATION block near the top of this file. Stocks and crypto each
run as their own cycle (own state file, own email).

KNOWN, DELIBERATE SIMPLIFICATIONS vs. the backtester (flagged so they're
easy to revisit, not buried):
  - EOD same-bar fills. The backtester fills at the NEXT bar's open
    (no-lookahead discipline for a research backtest); this bot runs once
    per day after the close and acts on that same close, since there's no
    "next bar" to wait for in live trading -- economically the same idea
    as "act on today's signal", just naming today's close as the price
    instead of tomorrow's open. This matches how the existing stock engine
    already behaves (nothing new here).
  - Sizing now mirrors the backtester's decision #16 equal-weight-target
    formula directly: CRYPTO_TOTAL_CAPITAL is a manually-entered figure
    (edit it whenever you deposit/withdraw), and each suggested buy is
    sized at (cash_pool + invested_cost) / n_eligible, cash_pool derived
    fresh each run as CRYPTO_TOTAL_CAPITAL minus the cost basis of every
    open position -- no separately-tracked, driftable ledger. One
    remaining difference from the backtester: there's no monthly SIP
    schedule here (no new money "arrives" on a date) -- CRYPTO_TOTAL_CAPITAL
    IS the full pool, all the time; you grow it by editing the number.
  - wl_rank methods 1/2 (MA-slope-based ranking) need yesterday's fast/slow
    RSI values, which ARE tracked per-ticker now (see watch_row below), so
    all three ranking methods work. Only method 0 (percent-below-entry) is
    exercised by the params you're currently running.
"""

import argparse
import functools
import hashlib
import io
import json
import logging
import os
import random
import smtplib
import ssl
import time
import traceback
import warnings
import sys
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm



try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


warnings.filterwarnings('ignore')


try:
    from google.colab import drive, userdata  # type: ignore
    IN_COLAB = True
except ImportError:
    drive = None
    userdata = None
    IN_COLAB = False

# ════════════════════════════════════════════════════════════════════════
#  LOGGING
# ════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("swing_bot")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    
    # Prevents Windows charmap UnicodeEncodeError on emojis
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
        
    _console = logging.StreamHandler(sys.stdout)
    _console.setFormatter(_fmt)
    logger.addHandler(_console)


# ══════════════════════════════════════════════════════════════════════════════
#  ⚙️  CONFIGURATION — this is the only section you should need to edit.
# ══════════════════════════════════════════════════════════════════════════════

# ---- money -----------------------------------------------------------------
STOCK_AVAILABLE_FUNDS = 4000
STOCK_CHUNK_SIZE = 8000

# CRYPTO_TOTAL_CAPITAL -- EDIT THIS whenever you deposit/withdraw or just
# want to change how much you're committing. This is the total USD you're
# treating as "mine, deployable" across EVERY open crypto position plus
# whatever's sitting uninvested -- not a per-run budget. Each run derives
# cash_pool = CRYPTO_TOTAL_CAPITAL - (cost basis of all open positions),
# then sizes each suggested buy as an equal-weight target across every
# coin that's tradeable today: (cash_pool + invested_cost) / n_eligible,
# same formula and reasoning as the backtester's decision #16 (cost-basis,
# not mark-to-market, so one position's unrealized paper gain can't
# inflate the size of the next unrelated buy). If the wallet can't cover a
# full target, the bot suggests whatever's left instead of skipping --
# unless that's below CRYPTO_MIN_TICKET_SIZE, in which case it's fine to
# miss the entry rather than suggest a dust-sized buy.
CRYPTO_TOTAL_CAPITAL = 4000
CRYPTO_MIN_TICKET_SIZE = 2000   # mirrors the backtester's MIN_TICKET_SIZE (== MONTHLY_SIP)

# ---- run behavior ------------------------------------------------------------
STOCK_FORCE_RERUN = True
CRYPTO_FORCE_RERUN = True

STOCK_UNIVERSE_LIMIT = None   # cap on how many auto-discovered stock tickers to scan (None = no cap)
CRYPTO_UNIVERSE_LIMIT = None  # same, for crypto

# Gap larger than this (trading days) between runs triggers a loud warning so
# you notice if the bot has been silently failing to run. The replay engine
# below handles the gap correctly either way -- this is just a visibility flag.
GAP_WARNING_DAYS = 5

# If a stock index's constituent list hasn't refreshed in this many calendar
# days, its universe is "stale" and gets escalated from a log line to a
# banner in the email.
UNIVERSE_STALE_WARNING_DAYS = 7

# ---- manual ticker add/remove (universe-level) ------------------------------
# Stocks are auto-discovered from the NSE index lists below. Use these two
# sets to manually widen or narrow that list without touching index logic.
STOCK_EXTRA_TICKERS: set[str] = set()     # e.g. {"TATASTEEL.NS"} -- always scan these too
STOCK_EXCLUDE_TICKERS: set[str] = set()   # e.g. {"YESBANK.NS"} -- never scan/trade these

# ── DYNAMIC UNIVERSE (mirrors the backtester's fetch_top100_universe) ──
# True (default): every crypto cycle fetches today's top CRYPTO_TOP_N_FETCH
# coins by market cap from CoinGecko, drops stablecoins/wrapped/leveraged
# tokens and anything too thin to trade, and uses whatever survives (usually
# ends up somewhere around 100 coins out of a 200-coin candidate pool -- same
# ratio the backtester settled on). False: trade only the fixed CRYPTO_UNIVERSE
# list below instead (useful for a small, hand-picked universe or if you'd
# rather not depend on CoinGecko being reachable).
CRYPTO_USE_DYNAMIC_UNIVERSE = True
CRYPTO_TOP_N_FETCH = 200                        # candidate pool pulled from CoinGecko BEFORE filtering
CRYPTO_MIN_AVG_DAILY_VOLUME_USD = 3_000_000     # 24h USD volume floor for a coin to qualify (mirrors
                                                 # the backtester's MIN_AVG_DAILY_VOLUME_USD)
CRYPTO_UNIVERSE_STALE_WARNING_DAYS = 3          # cached dynamic universe older than this escalates to
                                                 # an email banner -- shorter than the stock engine's
                                                 # UNIVERSE_STALE_WARNING_DAYS since coin rankings and
                                                 # listings move faster than NSE index constituents.

# Fixed list used when CRYPTO_USE_DYNAMIC_UNIVERSE is False, AND as the
# last-resort fallback if a CoinGecko fetch fails with no usable cache on
# disk either. yfinance ticker format (e.g. "BTC-USD"). BTC-USD MUST stay in
# this list (or in CRYPTO_EXTRA_TICKERS) -- everything downstream (the regime
# filter, the RSI-reversal panic exit) assumes BTC's own data is being
# fetched. The crypto cycle also hard-guarantees BTC-USD is fetched even if
# it were accidentally removed from here (see run_cycle_crypto).
CRYPTO_UNIVERSE = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD",
    "DOGE-USD", "AVAX-USD", "DOT-USD", "LINK-USD", "MATIC-USD", "LTC-USD",
]
CRYPTO_EXTRA_TICKERS: set[str] = set()    # extra coins always included, on top of whichever universe mode is active
CRYPTO_EXCLUDE_TICKERS: set[str] = set()  # coins to always skip, regardless of universe mode

# Disclosed, static exclusion lists -- identical intent to the backtester's
# STABLECOIN_SYMBOLS/WRAPPED_SYMBOLS/LEVERAGED_OR_SYNTHETIC_PATTERNS: keep
# assets that structurally can't "trend" out of the dynamic universe. Not
# exhaustive by design -- the fetch step logs whatever it excluded so a miss
# is visible, not silent.
CRYPTO_STABLECOIN_SYMBOLS = {
    'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'FDUSD', 'USDD', 'USDP', 'GUSD',
    'PYUSD', 'USDE', 'FRAX', 'CRVUSD', 'LUSD', 'SUSD', 'EURT', 'EURS',
    'USTC', 'UST', 'USDS',
}
CRYPTO_WRAPPED_SYMBOLS = {
    'WBTC', 'WETH', 'WSTETH', 'WEETH', 'WBETH', 'STETH', 'CBETH', 'RETH',
    'WBNB', 'WAVAX', 'WMATIC',
}
CRYPTO_LEVERAGED_OR_SYNTHETIC_PATTERNS = ('UP', 'DOWN', '3L', '3S', 'BULL', 'BEAR')

COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"

# ---- manual position/watchlist overrides ------------------------------------
# Force-add or force-remove a specific OPEN POSITION or WATCHLIST entry,
# independent of what the bot's own signal logic would do. Same schema for
# both asset classes:
#   force_remove_positions: ["TICKER", ...]
#   force_add_positions:    {"TICKER": {"entry_price": 123.45, "entry_date": "YYYY-MM-DD"}}
#   force_remove_watchlist: ["TICKER", ...]
#   force_add_watchlist:    {"TICKER": {"wl_entry_price": 123.45, "date_added": "YYYY-MM-DD"}}
STOCK_MANUAL_OVERRIDES: dict[str, Any] = {
    "force_remove_positions": [],
    "force_add_positions": {},
    "force_remove_watchlist": [],
    "force_add_watchlist": {},
}

CRYPTO_MANUAL_OVERRIDES: dict[str, Any] = {
    "force_remove_positions": [],
    "force_add_positions": {},
    "force_remove_watchlist": [],
    "force_add_watchlist": {},
}

# ---- strategy parameters -----------------------------------------------------
STOCK_PARAMS_META = {
    "engine_version": "5.0.0",
    "oos_score": 2.153335728175182,
    "is_score": 2.0706792546733706,
    "robustness_ratio": 1.0399175648837269,
}

STOCK_PARAMS = {
    "s_ma": 20,
    "l_ma": 81,
    "sl_ma": 167,
    "div_exit_m": 1,
    "t_s": 2,
    "t_l": 0,
    "t_sl": 2,
    "wl_rank": 1,
    "entry_f": 1,
    "n_exit_m": 1,
    "n_trail_p": 22.17630256750009,
    "n_atr_m": 2.408113662644546,
    "adx_thresh": 25.0,
    "div_exit_v": 7.719158747538546,
}

# Tickers that use the alternate "peak-drawdown / SuperMA / days-below-long-MA"
# exit rule instead of the standard trailing-stop/ATR rule (originally modeled
# on dividend-paying stocks you'd rather hold through noise).
STOCK_ALT_EXIT_TICKERS = {'ITC.NS', 'COALINDIA.NS', 'ONGC.NS', 'POWERGRID.NS', 'NTPC.NS', 'PFC.NS',
                          'RECLTD.NS', 'VEDL.NS', 'GAIL.NS', 'BPCL.NS', 'IOC.NS', 'PETRONET.NS',
                          'SAIL.NS', 'NHPC.NS', 'NMDC.NS', 'HINDZINC.NS', 'CASTROLIND.NS'}

# ── CRYPTO PARAMS -- 1:1 port of the crypto_new_test/main.py backtester's
# param schema (engine v1.2.0+decision #17). This is the exact champion the
# walk-forward search produced; see the backtester's module docstring for
# what each field means. use_trend_ma/use_btc_filter/use_panic_exits are
# stored as real Python bools (the backtester's JSON export uses 0/1,
# converted here once). BUG FIX: this dict used to be a bare literal, never
# assigned to anything -- CRYPTO_PARAMS/CRYPTO_PARAMS_META didn't exist, so
# the ASSETS registry below crashed with NameError the moment this module
# loaded. wl_grace_days/wl_max_age_days REMOVED (decision #17 in the
# backtester replaced the grace-period/max-age mechanism entirely -- a
# watchlist candidate is now a shadow position with its own stop/TP,
# removed by the same exit condition that would sell a real position; see
# _crypto_wl_shadow_exit_step() further down).
CRYPTO_PARAMS_META = {
    "engine_version": "1.2.0",
    "oos_score": 0.0,
    "is_score": 1.5761502939967806,
    "robustness_ratio": 0.0,
}

CRYPTO_PARAMS = {
    "use_trend_ma": 0,
        "use_btc_filter": 1,
        "use_panic_exits": 1,
        "signal_method": 1,
        "exit_method": 0,
        "entry_ma_len": 88,
        "entry_ma_type": 2,
        "rsi_fast_len": 55,
        "rsi_fast_smt": 11,
        "rsi_slow_len": 78,
        "rsi_slow_smt": 21,
        "trend_ma_len": 144,
        "trend_ma_type": 2,
        "exit_ma_len": 33,
        "exit_ma_type": 2,
        "btc_ma_len": 74,
        "btc_ma_type": 1,
        "wl_rank": 0,
        "adx_thresh": 15.0,
        "sl_mult": 2.9057366206868434,
        "tp_mult": 14.340429945524633,
        "trail_mult": 13.491195609646326,
        "trail_pct": 19.697422661758537,
        "exit_atr_mult": 1.8627830146544484
}

CRYPTO_ALT_EXIT_TICKERS: set[str] = set()   # unused by the new crypto engine; kept for ASSETS-registry shape

# ---- NSE stock universe sources ----------------------------------------------
# NSE moved its CSV archive host from archives.nseindia.com to
# nsearchives.nseindia.com; the old host now regularly 503s from cloud/
# datacenter IPs (Colab included). Each index is tried in order and the first
# mirror that works wins; niftyindices.com (the official index maintainer) is
# an independent third fallback.
NSE_INDEX_SOURCES = {
    "NIFTY50": [
        "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
        "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
        "https://niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    ],
    "NIFTYNEXT50": [
        "https://nsearchives.nseindia.com/content/indices/ind_niftynext50list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftynext50list.csv",
        "https://niftyindices.com/IndexConstituent/ind_niftynext50list.csv",
    ],
    "NIFTYMIDCAP150": [
        "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
        "https://niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
    ],
}

# ---- misc ---------------------------------------------------------------------
YF_AUTO_ADJUST = True
DOWNLOAD_DELAY_RANGE = (0.2, 0.5)
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 2.0

RECIPIENT_EMAIL_OVERRIDE = None
DRY_RUN = False  # set via --dry-run; when True, emails are logged, not sent


# ══════════════════════════════════════════════════════════════════════════════
#  DIRECTORIES
# ══════════════════════════════════════════════════════════════════════════════

if IN_COLAB:
    drive.mount('/content/drive', force_remount=False)
    BASE_DIR = '/content/drive/MyDrive/UnifiedSwingBot'
else:
    BASE_DIR = os.environ.get('SWING_BOT_BASE_DIR', str(Path.home() / 'UnifiedSwingBot'))

STOCK_DIR = f'{BASE_DIR}/stocks'
CRYPTO_DIR = f'{BASE_DIR}/crypto'
LOG_DIR = f'{BASE_DIR}/logs'
for d in [STOCK_DIR, CRYPTO_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

_file_handler = logging.FileHandler(f"{LOG_DIR}/bot.log")
_file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
logger.addHandler(_file_handler)


def get_secret(name: str) -> str | None:
    if IN_COLAB:
        try:
            return userdata.get(name)
        except Exception:
            return None
    return os.environ.get(name)


# ════════════════════════════════════════════════════════════════════════
#  EMAIL FLAVOR CONTENT -- purely cosmetic, no effect on trading logic.
#  pick_daily() selects one entry per calendar day deterministically, so it's
#  stable across multiple runs on the same day but rotates daily.
# ════════════════════════════════════════════════════════════════════════

DAILY_QUOTES = [
    "The market rewards patience far more often than it rewards prediction.",
    "A plan you can follow beats a forecast you can't.",
    "Position sizing decides survival; stock picking only decides speed.",
    "Every trailing stop is a promise you made to your calmer self.",
    "Compounding is silent work that only looks impressive in hindsight.",
    "Risk isn't the trade you lose -- it's the one sized too big to survive.",
    "Discipline is choosing what you'd want most over what you want right now.",
    "The best exit is the one you decided on before you got emotional.",
    "Diversification is really just admitting you don't know which idea wins.",
    "Small, repeatable edges beat big, unrepeatable ones.",
    "Your system doesn't need to be right often -- just right on size.",
    "The trend is just the market agreeing with itself, for now.",
]

DAILY_TRIVIA = [
    "The Bombay Stock Exchange, founded in 1875, is one of Asia's oldest exchanges.",
    "SIP (Systematic Investment Plan) simply means investing a fixed amount at fixed intervals, rain or shine.",
    "The term 'bull market' is thought to come from how a bull attacks -- thrusting its horns upward.",
    "The Nifty 50 index was launched by NSE in 1996.",
    "Rupee-cost averaging can lower your average buy price automatically during choppy, sideways markets.",
    "Warren Buffett is often said to have bought his first stock at age eleven.",
    "The '4% rule' is a rough, widely-cited guideline for sustainable retirement withdrawal rates -- not a guarantee.",
    "Compound interest has long been nicknamed the 'eighth wonder of the world' by its admirers.",
    "India's mutual fund industry has grown into one of the fastest-expanding pools of retail savings globally.",
    "A stop-loss order doesn't guarantee your exact exit price -- fast-moving markets can cause slippage past it.",
    "ATR (Average True Range) was originally developed by J. Welles Wilder for commodity traders in the 1970s.",
    "The ADX indicator measures trend strength, not trend direction -- a rising ADX just means conviction is growing.",
    "Bitcoin's block reward halves roughly every four years, by design.",
    "Crypto markets trade 24/7 -- there's no closing bell, so gaps happen mid-week, not just over weekends.",
]


def pick_daily(items: list[str], date_str: str) -> str:
    """Deterministic 'line of the day': same for every run today, rotates tomorrow."""
    idx = int(hashlib.md5(date_str.encode()).hexdigest(), 16) % len(items)
    return items[idx]


# ════════════════════════════════════════════════════════════════════════
#  NETWORK HELPERS (retry/backoff + NSE session warm-up)
# ════════════════════════════════════════════════════════════════════════

def retry(times: int = RETRY_ATTEMPTS, delay: float = RETRY_BASE_DELAY,
          backoff: float = 2.0, exceptions: tuple = (Exception,)) -> Callable:
    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _delay = delay
            last_exc = None
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    logger.warning(f"{fn.__name__} failed (attempt {attempt}/{times}): {e}")
                    if attempt < times:
                        time.sleep(_delay)
                        _delay *= backoff
            raise last_exc
        return wrapper
    return deco


def _nse_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    })
    try:
        s.get("https://www.nseindia.com", timeout=10)
    except Exception as e:
        logger.warning(f"NSE session warm-up failed (continuing anyway): {e}")
    return s


@retry(exceptions=(requests.RequestException,))
def _fetch_nse_list(session: requests.Session, url: str) -> pd.Series:
    resp = session.get(url, timeout=10)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))['Symbol']


@retry(exceptions=(Exception,))
def _safe_yf_download(*args, **kwargs) -> pd.DataFrame:
    return yf.download(*args, auto_adjust=YF_AUTO_ADJUST, threads=False, **kwargs)


# ════════════════════════════════════════════════════════════════════════
#  STATE / EMAIL / FORMATTING UTILITIES
# ════════════════════════════════════════════════════════════════════════

def load_state(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path) as f:
                s = json.load(f)
            s.setdefault('positions', {}); s.setdefault('watchlist', {}); s.setdefault('last_run_date', None)
            return s
        except Exception as e:
            logger.warning(f"Could not parse state file {path}, starting fresh: {e}")
    return {"positions": {}, "watchlist": {}, "last_run_date": None}


def save_state(state: dict, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(state, f, indent=2, default=str)


@retry(exceptions=(smtplib.SMTPException, OSError))
def _send_smtp(user: str, pw: str, to: str, msg: MIMEMultipart) -> None:
    if not user or not pw:
        raise ValueError("GMAIL_USER or GMAIL_APP_PASSWORD is missing. Check your .env file.")
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=ssl.create_default_context()) as server:
        server.login(user, pw)
        server.sendmail(user, to, msg.as_string())


def send_email(subject: str, html_body: str) -> None:
    if DRY_RUN:
        logger.info(f"[DRY RUN] Would send email: {subject}")
        return
    try:
        user = get_secret('GMAIL_USER')
        pw = get_secret('GMAIL_APP_PASSWORD')
        to = RECIPIENT_EMAIL_OVERRIDE or get_secret('RECIPIENT_EMAIL')
        msg = MIMEMultipart('alternative')
        msg['Subject'], msg['From'], msg['To'] = subject, user, to
        msg.attach(MIMEText(html_body, 'html'))
        _send_smtp(user, pw, to, msg)
        logger.info(f"✅ Email sent: {subject}")
    except Exception as e:
        logger.error(f"❌ Email failed: {e}\n{html_body}")


def fmt_curr(x: float, prefix: str) -> str:
    return f"{prefix}{x:,.2f}"


def fmt_pct(x: float) -> str:
    return f"{x:+.2f}%"


def colored_pct(x: float) -> str:
    """Green/red inline-styled percentage span."""
    color = "#1b5e20" if x >= 0 else "#b71c1c"
    return f'<span style="color:{color};font-weight:600;">{fmt_pct(x)}</span>'


def build_table(headers: list[str], rows: list[list], empty_msg: str) -> str:
    """Generic table with a dark header band and zebra striping (Buys, Watchlist)."""
    if not rows:
        return f'<p style="color:#888;font-size:13px;font-style:italic;">{empty_msg}</p>'
    h = ''.join(
        f'<th style="border:1px solid #ddd;padding:8px;background:#1a237e;'
        f'color:#fff;font-size:12px;text-align:left;">{x}</th>' for x in headers
    )
    body = []
    for i, r in enumerate(rows):
        bg = "#ffffff" if i % 2 == 0 else "#f7f7fb"
        tds = ''.join(f'<td style="border:1px solid #e5e5e5;padding:7px;font-size:13px;">{x}</td>' for x in r)
        body.append(f'<tr style="background:{bg};">{tds}</tr>')
    return f'<table style="border-collapse:collapse;width:100%;font-size:13px;">{h}{"".join(body)}</table>'


# Row styling for the Open Positions table, keyed by whether a stop-out today
# would realize a profit ("locked_in"), a loss ("at_risk"), or can't be
# evaluated as a single price ("unknown").
POSITION_STATUS_STYLE = {
    "locked_in": {"bg": "#e8f5e9", "text": "#1b5e20", "badge": "✅ Locked-In", "badge_bg": "#2e7d32"},
    "at_risk":   {"bg": "#ffebee", "text": "#b71c1c", "badge": "⚠️ At Risk",   "badge_bg": "#c62828"},
    "unknown":   {"bg": "#f5f5f5", "text": "#616161", "badge": "— N/A",        "badge_bg": "#9e9e9e"},
}


def build_positions_table(rows: list[dict]) -> str:
    if not rows:
        return '<p style="color:#888;font-size:13px;font-style:italic;">No open positions.</p>'
    headers = ["Ticker", "Entry Date", "Held", "Entry", "Current", "P&L",
               "Exit Trigger", "Cushion", "Status"]
    h = ''.join(
        f'<th style="border:1px solid #ddd;padding:8px;background:#1a237e;'
        f'color:#fff;font-size:12px;text-align:left;">{x}</th>' for x in headers
    )
    body = []
    for r in rows:
        s = POSITION_STATUS_STYLE[r["status"]]
        badge = (f'<span style="background:{s["badge_bg"]};color:#fff;padding:3px 9px;'
                 f'border-radius:10px;font-size:11px;white-space:nowrap;">{s["badge"]}</span>')
        cells = [r["ticker"], r["entry_date"], f'{r["held_days"]}d', r["entry_str"],
                 r["cur_str"], r["pnl_str"], r["stop_str"], r["cushion_str"], badge]
        tds = ''.join(
            f'<td style="border:1px solid #e0d9d9;padding:7px;font-size:13px;color:{s["text"]};">{c}</td>'
            for c in cells
        )
        body.append(f'<tr style="background:{s["bg"]};">{tds}</tr>')
    return f'<table style="border-collapse:collapse;width:100%;font-size:13px;">{h}{"".join(body)}</table>'


def build_sells_table(rows: list[dict]) -> str:
    if not rows:
        return '<p style="color:#888;font-size:13px;font-style:italic;">No exits today.</p>'
    headers = ["Ticker", "Reason", "Entry", "Exit", "Gain"]
    h = ''.join(
        f'<th style="border:1px solid #ddd;padding:8px;background:#1a237e;'
        f'color:#fff;font-size:12px;text-align:left;">{x}</th>' for x in headers
    )
    body = []
    for i, r in enumerate(rows):
        bg = "#ffffff" if i % 2 == 0 else "#f7f7fb"
        cells = [r["ticker"], r["reason"], r["entry_str"], r["exit_str"], colored_pct(r["gain_pct"])]
        tds = ''.join(f'<td style="border:1px solid #e5e5e5;padding:7px;font-size:13px;">{x}</td>' for x in cells)
        body.append(f'<tr style="background:{bg};">{tds}</tr>')
    return f'<table style="border-collapse:collapse;width:100%;font-size:13px;">{h}{"".join(body)}</table>'


def kpi_card(label: str, value: str, sub: str = "", color: str = "#1a237e") -> str:
    sub_html = f'<div style="font-size:11px;color:#999;margin-top:3px;">{sub}</div>' if sub else ''
    return (f'<td style="padding:16px 8px;text-align:center;background:#fafafa;'
            f'border:1px solid #ececec;">'
            f'<div style="font-size:10.5px;color:#9e9e9e;text-transform:uppercase;'
            f'letter-spacing:0.6px;">{label}</div>'
            f'<div style="font-size:21px;font-weight:700;color:{color};margin-top:5px;">{value}</div>'
            f'{sub_html}</td>')


# ════════════════════════════════════════════════════════════════════════
#  INDICATORS -- kept bit-identical to the backtester used for optimization
# ════════════════════════════════════════════════════════════════════════

def calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    tr = np.zeros(n)
    start = next((i for i, v in enumerate(close) if not np.isnan(v)), n)
    if n - start < period + 1: return np.full(n, np.nan)
    for i in range(start + 1, n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr = np.full(n, np.nan)
    atr_s = np.mean(tr[start+1 : start+period+1])
    atr[start + period] = atr_s
    for i in range(start + period + 1, n):
        atr_s = (atr_s * (period - 1) + tr[i]) / period
        atr[i] = atr_s
    return atr


def calc_ma(prices: np.ndarray, period: int, ma_type: int = 0) -> np.ndarray:
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    res = np.full(n, np.nan)
    start = next((i for i, v in enumerate(prices) if not np.isnan(v)), n)
    if n - start < period: return res

    if ma_type == 0: # SMA
        for i in range(start + period - 1, n): res[i] = np.mean(prices[i-period+1 : i+1])
    elif ma_type in (1, 2): # EMA/DEMA
        ema1 = np.full(n, np.nan)
        ema1[start + period - 1] = np.mean(prices[start:start + period])
        mult = 2 / (period + 1)
        for i in range(start + period, n): ema1[i] = (prices[i] - ema1[i-1]) * mult + ema1[i-1]
        res = ema1 if ma_type == 1 else 2 * ema1 - calc_ma(ema1, period, 1)
    elif ma_type == 3: # WMA
        weights = np.arange(1, period + 1, dtype=np.float64)
        for i in range(start + period - 1, n): res[i] = np.sum(prices[i-period+1 : i+1] * weights) / weights.sum()
    return res


def calc_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    adx = np.full(n, np.nan)
    start = next((i for i, v in enumerate(close) if not np.isnan(v)), n)
    if n - start < period * 2: return adx

    tr, pdm, ndm = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(start + 1, n):
        hd, ld = high[i] - high[i-1], low[i-1] - low[i]
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        pdm[i] = hd if hd > ld and hd > 0 else 0.0
        ndm[i] = ld if ld > hd and ld > 0 else 0.0

    atr_s, p_s, n_s = np.sum(tr[start+1:start+period+1]), np.sum(pdm[start+1:start+period+1]), np.sum(ndm[start+1:start+period+1])
    for i in range(start + period, n):
        if i > start + period:
            atr_s = atr_s - atr_s/period + tr[i]
            p_s = p_s - p_s/period + pdm[i]
            n_s = n_s - n_s/period + ndm[i]
        dx = 100 * abs((100*p_s/atr_s) - (100*n_s/atr_s)) / ((100*p_s/atr_s) + (100*n_s/atr_s)) if atr_s > 0 else 0.0
        adx[i] = dx if i == start + period else (adx[i-1]*(period-1) + dx)/period
    return adx


def calc_rsi_wilder(prices: np.ndarray, period: int) -> np.ndarray:
    """Standard Wilder-smoothed RSI -- matches the backtester's calc_rsi_wilder
    (and Pine's ta.rsi) exactly. NaN-safe the same way the other indicators
    here are: scans past a leading NaN run (newly-listed coin) before seeding
    the Wilder average."""
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    rsi = np.full(n, np.nan)
    start = next((i for i, v in enumerate(prices) if not np.isnan(v)), n)
    if n - start < period + 1:
        return rsi

    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(start + 1, n):
        diff = prices[i] - prices[i - 1]
        if diff > 0: gains[i] = diff
        else: losses[i] = -diff

    avg_gain = np.mean(gains[start+1:start+period+1])
    avg_loss = np.mean(losses[start+1:start+period+1])
    idx0 = start + period
    rsi[idx0] = (100.0 - 100.0/(1.0+avg_gain/avg_loss)) if avg_loss > 0 else (100.0 if avg_gain > 0 else 50.0)
    for i in range(idx0 + 1, n):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
        rsi[i] = (100.0 - 100.0/(1.0+avg_gain/avg_loss)) if avg_loss > 0 else (100.0 if avg_gain > 0 else 50.0)
    return rsi


def calc_rsi_smoothed(prices: np.ndarray, rsi_len: int, smt_len: int) -> np.ndarray:
    """Matches the backtester's get_rsi_smoothed_cached(): Wilder RSI, then
    SMA-smoothed on top (ma_type=0) -- this is the Pine strategy's
    calc_smoothed_rsi(src, rsi_l, smt_l) => ta.sma(ta.rsi(src, rsi_l), smt_l)."""
    raw_rsi = calc_rsi_wilder(prices, rsi_len)
    return calc_ma(raw_rsi, smt_len, 0)


# ════════════════════════════════════════════════════════════════════════
#  MANUAL OVERRIDES
# ════════════════════════════════════════════════════════════════════════

def apply_manual_overrides(state: dict, overrides: dict, today_str: str) -> list[str]:
    logs = []
    for t in overrides.get("force_remove_positions", []):
        if t in state['positions']:
            del state['positions'][t]
            logs.append(f"Override: force-removed position {t}")
    for t in overrides.get("force_remove_watchlist", []):
        if state['watchlist'].pop(t, None) is not None:
            logs.append(f"Override: force-removed watchlist {t}")
    for t, info in overrides.get("force_add_positions", {}).items():
        entry = float(info["entry_price"])
        state['positions'][t] = {
            "entry_price": entry,
            "entry_date": info.get("entry_date", today_str),
            "high_since_entry": float(info.get("high_since_entry", entry)),
            "days_below_long": 0,
            # crypto-only fields, filled in with sane defaults so a force-added
            # position doesn't crash the crypto exit engine if used there:
            "shares": float(info.get("shares", 1.0)),
            "stop_loss_price": float(info.get("stop_loss_price", entry * 0.5)),
            "tp_trigger_price": float(info.get("tp_trigger_price", entry * 2.0)),
            "half_sold": bool(info.get("half_sold", False)),
        }
        logs.append(f"Override: force-added position {t}")
    for t, info in overrides.get("force_add_watchlist", {}).items():
        entry = float(info["wl_entry_price"])
        state['watchlist'][t] = {
            "wl_entry_price": entry,
            "date_added": info.get("date_added", today_str),
            "high_since_entry": float(info.get("high_since_entry", entry)),
            "is_pyramid": bool(info.get("is_pyramid", False)),
            # NEW (decision #17): shadow SL/TP, same defaults as
            # force_add_positions above, so a force-added watchlist entry
            # doesn't crash _crypto_wl_shadow_exit_step if used there.
            "stop_loss_price": float(info.get("stop_loss_price", entry * 0.5)),
            "tp_trigger_price": float(info.get("tp_trigger_price", entry * 2.0)),
            "half_sold": bool(info.get("half_sold", False)),
        }
        logs.append(f"Override: force-added watchlist {t}")
    return logs


# ════════════════════════════════════════════════════════════════════════
#  STOCK REPLAY ENGINE (unchanged) -- SMA/LMA crossover + trailing/ATR/
#  SuperMA-family exits.
#
#  Walks one ticker's position/watchlist state forward day-by-day across
#  whatever window needs re-checking, instead of only comparing the single
#  most recent bar.
# ════════════════════════════════════════════════════════════════════════

def _index_after(date_index: pd.DatetimeIndex, date_str: str) -> int:
    """First array position strictly after date_str -- where a replay window should start."""
    return int(date_index.searchsorted(pd.Timestamp(date_str), side='right'))


def _exit_check(is_alt: bool, p: dict, close: float, slma: float, lma: float,
                 high_since_entry: float, atr_val: float, days_below_long: int) -> tuple[str, int]:
    """One day's exit/invalidation test, used for BOTH open positions and watchlist entries."""
    if is_alt:
        if p['div_exit_m'] == 0 and close < high_since_entry * (1 - p['div_exit_v'] / 100):
            return f"Peak Drop {p['div_exit_v']:.1f}%", days_below_long
        if p['div_exit_m'] == 1 and not np.isnan(slma) and close < slma * (1 - p['div_exit_v'] / 100):
            return "SuperMA violation", days_below_long
        if p['div_exit_m'] == 2:
            days_below_long = days_below_long + 1 if (not np.isnan(lma) and close < lma) else 0
            if days_below_long > p['div_exit_v']:
                return "Days below long MA", days_below_long
        return "", days_below_long
    else:
        if p['n_exit_m'] == 1 and close < high_since_entry * (1 - p['n_trail_p'] / 100):
            return "Fixed Trail Hit", days_below_long
        if p['n_exit_m'] == 2 and not np.isnan(atr_val) and close < high_since_entry - p['n_atr_m'] * atr_val:
            return "ATR Trail Hit", days_below_long
        return "", days_below_long


def compute_stop_price(is_alt: bool, p: dict, high_since_entry: float, slma_val: float,
                        atr_val: float, days_below_long: int) -> tuple[float | None, str]:
    """Given the currently active exit rule for this ticker (per its params, not a
    hypothetical), returns the price level at which the *next* daily check would
    trigger a sell, plus a short label naming which rule is live."""
    if is_alt:
        if p['div_exit_m'] == 0:
            return high_since_entry * (1 - p['div_exit_v'] / 100), f"Peak Trail -{p['div_exit_v']:.1f}%"
        if p['div_exit_m'] == 1:
            if np.isnan(slma_val):
                return None, "SuperMA not yet available"
            return slma_val * (1 - p['div_exit_v'] / 100), f"SuperMA -{p['div_exit_v']:.1f}%"
        return None, f"{days_below_long}/{p['div_exit_v']:.0f}d below Long MA"
    else:
        if p['n_exit_m'] == 0:
            return None, "MA Crossunder (no fixed price)"
        if p['n_exit_m'] == 1:
            return high_since_entry * (1 - p['n_trail_p'] / 100), f"Fixed Trail -{p['n_trail_p']:.1f}%"
        if np.isnan(atr_val):
            return None, "ATR Trail (ATR n/a)"
        return high_since_entry - p['n_atr_m'] * atr_val, f"ATR Trail -{p['n_atr_m']:.2f}×ATR"


def process_ticker(t: str, df: pd.DataFrame, p: dict, state: dict, today_str: str,
                    last_run_date: str | None, diag: dict, alt_exit_tickers: set[str]) -> dict | None:
    """Replays ticker t's position/watchlist state from the last verified day through
    today (STOCK engine only)."""
    closes, highs, lows, atr = df['Close'].values, df['High'].values, df['Low'].values, df['ATR'].values
    sma  = calc_ma(closes, p['s_ma'],  p['t_s'])
    lma  = calc_ma(closes, p['l_ma'],  p['t_l'])
    slma = calc_ma(closes, p['sl_ma'], p['t_sl'])
    adx  = calc_adx(highs, lows, closes)
    is_alt = t in alt_exit_tickers
    n = len(df)
    if n < 2 or np.isnan(sma[-2]) or np.isnan(lma[-2]):
        return None

    pos = state['positions'].get(t)
    wl  = state['watchlist'].get(t)

    ref_date = last_run_date
    if wl is not None and wl.get('high_since_entry') is None:
        ref_date = wl['date_added'] if ref_date is None else min(ref_date, wl['date_added'])

    if ref_date is None:
        start_idx = n - 1          # true first-ever run: only the latest bar
    else:
        start_idx = max(1, _index_after(df.index, ref_date))
        if start_idx > n - 1:
            start_idx = n - 1      # nothing new since ref_date; still check today once

    gap = (n - 1) - start_idx
    if gap >= GAP_WARNING_DAYS:
        logger.warning(f"{t}: replaying {gap} unprocessed trading day(s) "
                        f"(bot appears to have missed runs) -- catching up now.")

    sell_event = None
    wl_events = []

    for i in range(start_idx, n):
        close_i, slma_i, lma_i, atr_i = closes[i], slma[i], lma[i], atr[i]
        date_i = df.index[i].strftime('%Y-%m-%d')
        is_today = (i == n - 1)

        if pos is not None:
            pos['high_since_entry'] = max(pos.get('high_since_entry', close_i), close_i)
            exit_reason = ""
            if is_alt:
                exit_reason, pos['days_below_long'] = _exit_check(
                    True, p, close_i, slma_i, lma_i, pos['high_since_entry'], atr_i,
                    pos.get('days_below_long', 0))
            else:
                if p['n_exit_m'] == 0 and not np.isnan(sma[i-1]) and not np.isnan(lma[i-1]) \
                        and sma[i-1] >= lma[i-1] and sma[i] < lma[i]:
                    exit_reason = "MA Crossunder"
                else:
                    exit_reason, _ = _exit_check(False, p, close_i, slma_i, lma_i,
                                                  pos['high_since_entry'], atr_i, 0)
            if exit_reason:
                sell_event = {
                    "ticker": t, "reason": exit_reason, "entry": pos['entry_price'],
                    "exit_price": close_i, "date": date_i, "retroactive": not is_today,
                }
                del state['positions'][t]
                pos = None
            continue

        if wl is not None:
            if wl.get('high_since_entry') is None:
                a_idx = _index_after(df.index, wl['date_added']) - 1
                a_idx = max(0, a_idx)
                wl['high_since_entry'] = float(np.nanmax(closes[a_idx:i])) if i > a_idx else float(wl['wl_entry_price'])
            wl['high_since_entry'] = max(wl['high_since_entry'], close_i)

            invalid_reason = ""
            if not np.isnan(sma[i-1]) and not np.isnan(lma[i-1]) and sma[i-1] >= lma[i-1] and sma[i] < lma[i]:
                invalid_reason = "MA Crossunder"
            else:
                invalid_reason, _ = _exit_check(is_alt, p, close_i, slma_i, lma_i,
                                                 wl['high_since_entry'], atr_i, 0)
            if invalid_reason:
                wl_events.append({"ticker": t, "action": "removed", "reason": invalid_reason,
                                   "date": date_i, "retroactive": not is_today})
                del state['watchlist'][t]
                wl = None
            else:
                continue

        if i >= 1 and not np.isnan(sma[i-1]) and not np.isnan(lma[i-1]) \
                and sma[i-1] <= lma[i-1] and sma[i] > lma[i]:
            diag["xover"] += 1
            blocked = (p['entry_f'] == 1 and close_i <= slma_i) or (p['entry_f'] == 2 and close_i >= slma_i)
            if blocked:
                diag["regime_block"] += 1
                continue
            adx_i = adx[i]
            if p['adx_thresh'] > 0 and (np.isnan(adx_i) or adx_i < p['adx_thresh']):
                diag["adx_block"] += 1
                if not np.isnan(adx_i) and adx_i >= p['adx_thresh'] - 5:
                    diag["near_miss"].append(f"{t}: {adx_i:.1f}")
                continue
            wl = {"wl_entry_price": float(close_i), "date_added": date_i, "high_since_entry": float(close_i)}
            state['watchlist'][t] = wl
            wl_events.append({"ticker": t, "action": "added", "price": close_i,
                               "date": date_i, "retroactive": not is_today})

    stop_info = None
    if pos is not None:
        stop_price, stop_label = compute_stop_price(
            is_alt, p, pos['high_since_entry'], slma[-1], atr[-1], pos.get('days_below_long', 0))
        stop_info = {"price": stop_price, "label": stop_label}

    result = None
    if t in state['watchlist']:
        result = {'close': closes[-1], 'sma': sma[-1], 'lma': lma[-1]}
    return {"sell_event": sell_event, "wl_events": wl_events, "watch_row": result,
            "today_close": closes[-1] if not np.isnan(closes[-1]) else None,
            "stop_info": stop_info}


# ════════════════════════════════════════════════════════════════════════
#  CRYPTO REPLAY ENGINE -- 1:1 port of crypto_new_test/main.py's live logic:
#  standing BTC-regime filter, dual smoothed-RSI crossover entry, hybrid
#  ATR TP(50% partial)+breakeven+ATR-trail exit family, always-on panic
#  override, pyramiding, watchlist grace-days/max-age.
# ════════════════════════════════════════════════════════════════════════

_MA_TYPE_NAMES = {0: 'SMA', 1: 'EMA', 2: 'DEMA', 3: 'WMA'}


def _crypto_signal_arrays(df: pd.DataFrame, p: dict):
    """Builds this ticker's own signal-relevant series -- mirrors the
    backtester's per-ticker setup inside evaluate_params_crypto()."""
    closes = df['Close'].values
    if p['signal_method'] == 1:
        fast = calc_rsi_smoothed(closes, p['rsi_fast_len'], p['rsi_fast_smt'])
        slow = calc_rsi_smoothed(closes, p['rsi_slow_len'], p['rsi_slow_smt'])
    else:
        fast = closes
        slow = calc_ma(closes, p['entry_ma_len'], p['entry_ma_type'])
    if p['use_trend_ma']:
        trend_ma = calc_ma(closes, p['trend_ma_len'], p['trend_ma_type'])
        trend_ok = closes > trend_ma
    else:
        trend_ok = np.ones(len(closes), dtype=bool)
    exit_ma = calc_ma(closes, p['exit_ma_len'], p['exit_ma_type'])
    adx = calc_adx(df['High'].values, df['Low'].values, closes)
    return fast, slow, trend_ok, exit_ma, adx


def _crypto_reversal(fast: np.ndarray, slow: np.ndarray, i: int) -> bool:
    return (not np.isnan(fast[i-1]) and not np.isnan(slow[i-1])
            and fast[i-1] >= slow[i-1] and fast[i] < slow[i])


def _crypto_cross_up(fast: np.ndarray, slow: np.ndarray, i: int) -> bool:
    return (i >= 1 and not np.isnan(fast[i-1]) and not np.isnan(slow[i-1])
            and fast[i-1] <= slow[i-1] and fast[i] > slow[i])


def _crypto_panic_override(p: dict, btc_bullish_today: bool, fast: np.ndarray, slow: np.ndarray,
                            closes: np.ndarray, exit_ma: np.ndarray, i: int) -> bool:
    """Always-on override that can force a full exit / instant watchlist drop
    regardless of the selected exit family -- mirrors simulate_portfolio_crypto's
    `override`/`override_exit` computation exactly."""
    if not p['use_panic_exits']:
        return False
    if p['use_btc_filter'] and not btc_bullish_today:
        return True
    if p['signal_method'] == 0:
        if (i >= 1 and not np.isnan(exit_ma[i-1]) and not np.isnan(exit_ma[i])
                and closes[i-1] >= exit_ma[i-1] and closes[i] < exit_ma[i]):
            return True
    else:
        if _crypto_reversal(fast, slow, i):
            return True
    return False


def _crypto_exit_step(p: dict, pos: dict, close_i: float, atr_i: float) -> tuple[bool, bool]:
    """One day's exit test for exit_method families 0-3, mirrors
    simulate_portfolio_crypto's PHASE-B block. Mutates pos in place
    (stop_loss_price ratchet / half_sold flag). Returns (should_exit_full,
    should_exit_partial)."""
    em = p['exit_method']
    should_full = False
    should_partial = False
    if em == 0:
        if not pos['half_sold'] and close_i >= pos['tp_trigger_price']:
            should_partial = True
        if pos['half_sold'] and not np.isnan(atr_i):
            new_sl = close_i - atr_i * p['trail_mult']
            if new_sl > pos['stop_loss_price']:
                pos['stop_loss_price'] = new_sl
        if close_i < pos['stop_loss_price']:
            should_full = True
    elif em == 1:
        if close_i < pos['high_since_entry'] * (1.0 - p['trail_pct'] / 100.0):
            should_full = True
    elif em == 2:
        if not np.isnan(atr_i) and close_i < pos['high_since_entry'] - p['exit_atr_mult'] * atr_i:
            should_full = True
    # em == 3 (entry-signal crossunder) is handled by the caller via _crypto_reversal /
    # the panic-override reversal check, since that IS the "mirror-image of entry" rule.
    return should_full, should_partial


def _crypto_shadow_sl_tp(p: dict, entry_price: float, atr_today: float, atr_prev: float) -> tuple[float, float]:
    """ATR-based shadow stop/TP for a fresh watchlist candidate -- mirrors
    exactly what a real position gets at buy time in run_cycle_crypto's
    deployment loop AND the backtester's CAPITAL DEPLOYMENT block: try
    TODAY's ATR first, fall back to YESTERDAY's if today's isn't available
    yet, and only fall back to a flat 2%-of-price floor if neither is
    (previously this only tried today's ATR before jumping straight to the
    2% floor, skipping the yesterday fallback the backtester always has)."""
    entry_atr = atr_today if (not np.isnan(atr_today) and atr_today > 0) else atr_prev
    if not (not np.isnan(entry_atr) and entry_atr > 0):
        entry_atr = entry_price * 0.02
    return entry_price - entry_atr * p['sl_mult'], entry_price + entry_atr * p['tp_mult']


def _crypto_wl_shadow_exit_step(p: dict, wl: dict, close_i: float, atr_i: float) -> tuple[bool, bool]:
    """Watchlist counterpart of _crypto_exit_step() -- decision #17 in the
    backtester replaced the old grace-days/max-age mechanism entirely with
    this: a watchlist candidate is tracked as a shadow position (its own
    stop_loss_price/tp_trigger_price/half_sold, mirroring a real position's
    fields exactly) and is only removed by tripping the SAME exit condition
    that would sell a real position, or by getting bought. No separate age
    cap exists anymore. Mutates wl in place. Returns (should_exit_full,
    should_exit_partial)."""
    em = p['exit_method']
    should_full = False
    should_partial = False
    if em == 0:
        if not wl['half_sold'] and close_i >= wl['tp_trigger_price']:
            should_partial = True
        if wl['half_sold'] and not np.isnan(atr_i):
            new_sl = close_i - atr_i * p['trail_mult']
            if new_sl > wl['stop_loss_price']:
                wl['stop_loss_price'] = new_sl
        if close_i < wl['stop_loss_price']:
            should_full = True
    elif em == 1:
        if close_i < wl['high_since_entry'] * (1.0 - p['trail_pct'] / 100.0):
            should_full = True
    elif em == 2:
        if not np.isnan(atr_i) and close_i < wl['high_since_entry'] - p['exit_atr_mult'] * atr_i:
            should_full = True
    return should_full, should_partial


def _crypto_stop_price(p: dict, pos: dict, atr_last: float) -> dict:
    """Reporting-only: the price level at which the *next* daily check would
    trigger a sell for whichever exit family is active, mirroring
    compute_stop_price()'s role for the stock engine."""
    em = p['exit_method']
    if em == 0:
        label = "Breakeven+ATR Trail (post-TP)" if pos['half_sold'] else f"Initial Stop ({p['sl_mult']:.2f}×ATR)"
        return {"price": pos['stop_loss_price'], "label": label}
    if em == 1:
        return {"price": pos['high_since_entry'] * (1 - p['trail_pct'] / 100.0),
                "label": f"Fixed Trail -{p['trail_pct']:.1f}%"}
    if em == 2:
        if np.isnan(atr_last):
            return {"price": None, "label": "ATR Trail (ATR n/a today)"}
        return {"price": pos['high_since_entry'] - p['exit_atr_mult'] * atr_last,
                "label": f"ATR Trail -{p['exit_atr_mult']:.2f}×ATR"}
    return {"price": None, "label": "RSI/Signal Crossunder (no fixed price)"}


def process_ticker_crypto(t: str, df: pd.DataFrame, p: dict, state: dict, today_str: str,
                           last_run_date: str | None, diag: dict,
                           btc_bullish_by_date: dict[str, bool]) -> dict | None:
    """Replays ticker t's position/watchlist state from the last verified day
    through today, using the crypto engine (dual-RSI / BTC-filter / hybrid
    exit / pyramiding). Mutates state['positions']/state['watchlist'] in
    place. Returns a dict mirroring process_ticker()'s stock-engine shape."""
    closes = df['Close'].values
    atr = df['ATR'].values
    fast, slow, trend_ok, exit_ma, adx = _crypto_signal_arrays(df, p)
    n = len(df)
    if n < 2 or np.isnan(fast[-2]) or np.isnan(slow[-2]):
        return None

    pos = state['positions'].get(t)
    wl = state['watchlist'].get(t)

    ref_date = last_run_date
    if wl is not None and wl.get('high_since_entry') is None:
        ref_date = wl['date_added'] if ref_date is None else min(ref_date, wl['date_added'])

    if ref_date is None:
        start_idx = n - 1
    else:
        start_idx = max(1, _index_after(df.index, ref_date))
        if start_idx > n - 1:
            start_idx = n - 1

    gap = (n - 1) - start_idx
    if gap >= GAP_WARNING_DAYS:
        logger.warning(f"{t}: replaying {gap} unprocessed trading day(s) (crypto) -- catching up now.")

    sell_event = None
    wl_events = []

    for i in range(start_idx, n):
        close_i = closes[i]
        atr_i = atr[i]
        date_i = df.index[i].strftime('%Y-%m-%d')
        is_today = (i == n - 1)
        btc_bullish_i = btc_bullish_by_date.get(date_i, True)  # default bullish if BTC data missing that day

        # ── existing position: exit checks (checked first, same as the backtester) ──
        if pos is not None:
            pos['high_since_entry'] = max(pos.get('high_since_entry', close_i), close_i)
            override = _crypto_panic_override(p, btc_bullish_i, fast, slow, closes, exit_ma, i)
            should_full, should_partial = _crypto_exit_step(p, pos, close_i, atr_i)
            if p['exit_method'] == 3 and _crypto_reversal(fast, slow, i):
                should_full = True
            if override:
                should_full, should_partial = True, False

            if should_partial and not should_full:
                sell_shares = pos['shares'] * 0.5
                proceeds = sell_shares * close_i
                cost = sell_shares * pos['entry_price']
                pnl_pct_half = (proceeds - cost) / cost * 100 if cost > 0 else 0.0
                pos['shares'] -= sell_shares
                pos['half_sold'] = True
                if pos['stop_loss_price'] < pos['entry_price']:
                    pos['stop_loss_price'] = pos['entry_price']   # move stop to breakeven
                tag = f" (on {date_i}, detected today)" if not is_today else ""
                wl_events.append({"ticker": t, "action": "partial_exit",
                                   "reason": f"TP hit -- sold 50% @ {close_i:.4f} ({pnl_pct_half:+.1f}%)",
                                   "date": date_i, "retroactive": not is_today})

            if should_full:
                exit_price = close_i
                if override and p['use_btc_filter'] and not btc_bullish_i:
                    reason = "Panic Exit -- BTC Turned Bearish"
                elif override:
                    reason = "Panic Exit -- RSI Reversal" if p['signal_method'] == 1 else "Panic Exit -- Price < Exit MA"
                else:
                    reason = {0: "Stop-Loss Hit", 1: "Trail Stop Hit", 2: "ATR Trail Hit",
                              3: "Signal Reversal"}.get(p['exit_method'], "Exit Signal")
                sell_event = {"ticker": t, "reason": reason, "entry": pos['entry_price'],
                               "exit_price": exit_price, "date": date_i, "retroactive": not is_today}
                del state['positions'][t]
                pos = None
            # NOTE: deliberately no `continue` here (unlike the stock engine) -- a
            # ticker can still register a fresh/pyramid signal the same bar it was
            # exited or while still open, matching the backtester's ordering.

        # ── watchlist maintenance (decision #17: shadow-position exits) ──
        if wl is not None and not wl.get('is_pyramid'):
            # pyramid candidates ride along with the REAL position they'd add
            # to -- that position's own exit above is what removes them (via
            # the stale-pyramid prune in the deployment loop), same as the
            # backtester's `if wl_active and not wl_is_pyramid` guard.
            if wl.get('high_since_entry') is None:
                a_idx = max(0, _index_after(df.index, wl['date_added']) - 1)
                wl['high_since_entry'] = float(np.nanmax(closes[a_idx:i])) if i > a_idx else float(wl['wl_entry_price'])
            wl['high_since_entry'] = max(wl['high_since_entry'], close_i)

            reversal = _crypto_reversal(fast, slow, i)
            override = _crypto_panic_override(p, btc_bullish_i, fast, slow, closes, exit_ma, i)
            should_full, should_partial = _crypto_wl_shadow_exit_step(p, wl, close_i, atr_i)
            if p['exit_method'] == 3 and reversal:
                should_full = True
            if override:
                should_full, should_partial = True, False

            if should_partial and not should_full:
                if wl['stop_loss_price'] < wl['wl_entry_price']:
                    wl['stop_loss_price'] = wl['wl_entry_price']   # shadow breakeven, same as a real position's partial exit
                wl['half_sold'] = True
                wl_events.append({"ticker": t, "action": "partial_exit",
                                   "reason": f"Shadow TP hit @ {close_i:.4f} (watchlist, no capital committed yet)",
                                   "date": date_i, "retroactive": not is_today})

            if should_full:
                reason = "Panic Override" if override else {
                    0: "Stop-Loss Hit", 1: "Trail Stop Hit", 2: "ATR Trail Hit", 3: "Signal Reversal",
                }.get(p['exit_method'], "Exit Signal")
                wl_events.append({"ticker": t, "action": "removed", "reason": reason,
                                   "date": date_i, "retroactive": not is_today})
                del state['watchlist'][t]
                wl = None

        # ── fresh entry signal -- works for a flat ticker OR a pyramid into an open position ──
        if wl is None and _crypto_cross_up(fast, slow, i):
            diag["xover"] += 1
            trend_pass = bool(trend_ok[i]) if p['use_trend_ma'] else True
            if not trend_pass:
                diag["regime_block"] += 1
            else:
                btc_pass = btc_bullish_i if p['use_btc_filter'] else True
                if not btc_pass:
                    diag["regime_block"] += 1
                else:
                    adx_i = adx[i]
                    if p['adx_thresh'] > 0 and (np.isnan(adx_i) or adx_i < p['adx_thresh']):
                        diag["adx_block"] += 1
                    else:
                        is_pyramid = pos is not None
                        if is_pyramid:
                            wl = {"wl_entry_price": float(close_i), "date_added": date_i,
                                  "high_since_entry": pos['high_since_entry'], "is_pyramid": True}
                        else:
                            atr_prev_i = atr[i - 1] if i >= 1 else float('nan')
                            shadow_sl, shadow_tp = _crypto_shadow_sl_tp(p, close_i, atr_i, atr_prev_i)
                            wl = {"wl_entry_price": float(close_i), "date_added": date_i,
                                  "high_since_entry": float(close_i), "is_pyramid": False,
                                  "stop_loss_price": shadow_sl, "tp_trigger_price": shadow_tp,
                                  "half_sold": False}
                        state['watchlist'][t] = wl
                        wl_events.append({"ticker": t, "action": "added", "price": close_i,
                                           "pyramid": is_pyramid, "date": date_i, "retroactive": not is_today})

    stop_info = None
    if pos is not None:
        stop_info = _crypto_stop_price(p, pos, atr[-1])

    result = None
    if t in state['watchlist']:
        wl_now = state['watchlist'][t]
        result = {'close': closes[-1], 'entry_price': wl_now['wl_entry_price'],
                  'is_pyramid': wl_now.get('is_pyramid', False), 'fast': fast[-1], 'slow': slow[-1]}
    return {"sell_event": sell_event, "wl_events": wl_events, "watch_row": result,
            "today_close": closes[-1] if not np.isnan(closes[-1]) else None,
            "stop_info": stop_info}


# ════════════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ════════════════════════════════════════════════════════════════════════

def update_ticker_csv(ticker: str, data_dir: str) -> pd.DataFrame | None:
    path = f"{data_dir}/{ticker}.csv"
    try:
        if os.path.exists(path):
            df_old = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
            df_new = _safe_yf_download(
                ticker, start=(df_old.index.max().date() - timedelta(days=10)).isoformat(),
                multi_level_index=False, progress=False,
            )
            if df_new is not None and not df_new.empty:
                if isinstance(df_new.columns, pd.MultiIndex):
                    df_new.columns = df_new.columns.get_level_values(0)
                df_new.index = pd.to_datetime(df_new.index)
                comb = pd.concat([df_old.drop(columns=['ATR'], errors='ignore'), df_new])
                comb = comb[~comb.index.duplicated(keep='last')].sort_index()
            else:
                comb = df_old
        else:
            comb = _safe_yf_download(ticker, start="2015-01-01", progress=False, multi_level_index=False)
            if comb is not None and not comb.empty and isinstance(comb.columns, pd.MultiIndex):
                comb.columns = comb.columns.get_level_values(0)

        if comb is None or len(comb) < 100:
            return None

        comb = comb.dropna(subset=['Close', 'High', 'Low'])
        comb['ATR'] = calc_atr(comb['High'].values, comb['Low'].values, comb['Close'].values, 14)
        comb.to_csv(path, index_label='Date')
        return comb
    except Exception as e:
        logger.warning(f"update_ticker_csv({ticker}) failed: {e}")
        return None


def fetch_stock_universe() -> dict:
    """Fetches/refreshes NSE index constituent lists (with mirror fallback and
    on-disk caching) and flags which indices are stale."""
    cache_path = f'{STOCK_DIR}/universe_cache.json'
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                raw_cache = json.load(f)
            cache = ({"tickers": {"LEGACY": raw_cache}, "last_success": {"LEGACY": None}}
                     if isinstance(raw_cache, list) else raw_cache)
        except Exception as e:
            logger.warning(f"Could not parse universe cache, starting fresh: {e}")
            cache = {"tickers": {}, "last_success": {}}
    else:
        cache = {"tickers": {}, "last_success": {}}
    cache.setdefault("tickers", {})
    cache.setdefault("last_success", {})

    today_date_str = datetime.now().date().isoformat()
    tickers: set[str] = set()
    session = _nse_session()
    any_live_fetch = False

    for index_name, urls in NSE_INDEX_SOURCES.items():
        symbols, last_err = None, None
        for url in urls:
            try:
                symbols = _fetch_nse_list(session, url)
                break
            except Exception as e:
                last_err = e
        if symbols is not None:
            index_tickers = sorted(set((symbols + ".NS").tolist()))
            cache["tickers"][index_name] = index_tickers
            cache["last_success"][index_name] = today_date_str
            any_live_fetch = True
            logger.info(f"{index_name}: refreshed {len(index_tickers)} tickers.")
        else:
            logger.warning(f"{index_name}: all mirrors failed, last error: {last_err}")
        tickers.update(cache["tickers"].get(index_name, []))

    if any_live_fetch:
        try:
            with open(cache_path, 'w') as f:
                json.dump(cache, f)
        except Exception as e:
            logger.warning(f"Could not save universe cache: {e}")

    stale_indices: list[tuple[str, int | None]] = []
    for index_name in NSE_INDEX_SOURCES:
        last = cache["last_success"].get(index_name)
        if last is None:
            stale_indices.append((index_name, None))
        else:
            age = (datetime.now().date() - datetime.fromisoformat(last).date()).days
            if age >= UNIVERSE_STALE_WARNING_DAYS:
                stale_indices.append((index_name, age))

    stale_html = ""
    if stale_indices:
        lines = [f"<b>{n}</b>: never successfully fetched" if age is None
                 else f"<b>{n}</b>: last refreshed {age} day(s) ago" for n, age in stale_indices]
        parts = [f"{n} (never fetched)" if age is None else f"{n} ({age}d stale)" for n, age in stale_indices]
        logger.error(f"⚠️ Universe staleness threshold exceeded: {', '.join(parts)}")
        stale_html = (f"<div style='background:#ffebee;border-left:4px solid #c62828;padding:10px 14px;"
                      f"margin-bottom:14px;font-size:13px;color:#b71c1c;'>"
                      f"<b>⚠️ Universe data is stale (≥{UNIVERSE_STALE_WARNING_DAYS}d):</b><br>"
                      f"{'<br>'.join(lines)}<br>"
                      f"<span style='font-size:11.5px;color:#555;'>NSE index constituent fetches have been "
                      f"failing -- the bot is trading off a cached list, so recent index additions/removals "
                      f"won't be reflected. Check network access to nsearchives.nseindia.com.</span></div>")
    if not any_live_fetch:
        logger.warning(f"All NSE index fetches failed this run — trading off cached universe "
                        f"({len(tickers)} tickers).")

    return {"tickers": tickers, "stale_html": stale_html, "any_live_fetch": any_live_fetch}


def _is_noise_coin(symbol_upper: str) -> bool:
    if symbol_upper in CRYPTO_STABLECOIN_SYMBOLS or symbol_upper in CRYPTO_WRAPPED_SYMBOLS:
        return True
    for pat in CRYPTO_LEVERAGED_OR_SYNTHETIC_PATTERNS:
        if symbol_upper.endswith(pat):
            return True
    return False


def fetch_crypto_universe_dynamic() -> dict:
    """Fetches today's top CRYPTO_TOP_N_FETCH coins by market cap from
    CoinGecko, drops stablecoins/wrapped/leveraged tokens and anything below
    the volume floor (mirrors the backtester's fetch_top100_universe() /
    MIN_AVG_DAILY_VOLUME_USD screen exactly), and maps the survivors to
    yfinance "SYM-USD" tickers. Caches the result on disk so a CoinGecko
    outage falls back to yesterday's good list instead of collapsing all
    the way down to the tiny static CRYPTO_UNIVERSE."""
    cache_path = f'{CRYPTO_DIR}/dynamic_universe_cache.json'
    cache = {"tickers": [], "last_success": None}
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except Exception as e:
            logger.warning(f"Could not parse crypto universe cache, starting fresh: {e}")

    excluded_log: list[str] = []
    tickers: list[str] = []
    any_live_fetch = False
    try:
        r = requests.get(COINGECKO_MARKETS_URL, params={
            'vs_currency': 'usd', 'order': 'market_cap_desc',
            'per_page': CRYPTO_TOP_N_FETCH, 'page': 1, 'sparkline': 'false',
        }, timeout=20)
        r.raise_for_status()
        coins = r.json()
        for c in coins:
            sym = str(c.get('symbol', '')).upper()
            if not sym:
                continue
            if _is_noise_coin(sym):
                excluded_log.append(sym)
                continue
            vol_24h = c.get('total_volume', 0) or 0
            if sym != 'BTC' and vol_24h < CRYPTO_MIN_AVG_DAILY_VOLUME_USD:
                excluded_log.append(f"{sym} (24h vol ${vol_24h:,.0f} < ${CRYPTO_MIN_AVG_DAILY_VOLUME_USD:,.0f})")
                continue
            tickers.append(f"{sym}-USD")
        any_live_fetch = True
    except Exception as e:
        logger.warning(f"CoinGecko top-{CRYPTO_TOP_N_FETCH} fetch failed: {e}")

    # BTC-USD always first, de-duped, rank order preserved otherwise.
    tickers = [t for t in tickers if t != 'BTC-USD']
    tickers = ['BTC-USD'] + list(dict.fromkeys(tickers))

    today_date_str = datetime.now().date().isoformat()
    if any_live_fetch and len(tickers) >= max(10, CRYPTO_TOP_N_FETCH // 4):
        cache = {"tickers": tickers, "last_success": today_date_str}
        try:
            with open(cache_path, 'w') as f:
                json.dump(cache, f)
        except Exception as e:
            logger.warning(f"Could not save crypto universe cache: {e}")
        logger.info(f"Crypto universe: {len(tickers)} coins after filtering top-{CRYPTO_TOP_N_FETCH} "
                    f"by market cap (excluded {len(excluded_log)}: "
                    f"{', '.join(excluded_log[:15])}{' ...' if len(excluded_log) > 15 else ''})")
    elif cache.get("tickers"):
        logger.warning("CoinGecko fetch failed or returned too few coins today -- using cached universe.")
        tickers = cache["tickers"]
    else:
        logger.warning("No CoinGecko data and no cache available -- falling back to static CRYPTO_UNIVERSE.")
        tickers = list(CRYPTO_UNIVERSE)

    stale_html = ""
    last = cache.get("last_success")
    if last is not None:
        age = (datetime.now().date() - datetime.fromisoformat(last).date()).days
        if age >= CRYPTO_UNIVERSE_STALE_WARNING_DAYS:
            logger.error(f"⚠️ Crypto dynamic-universe cache is {age}d stale.")
            stale_html = (f"<div style='background:#ffebee;border-left:4px solid #c62828;padding:10px 14px;"
                          f"margin-bottom:14px;font-size:13px;color:#b71c1c;'>"
                          f"<b>⚠️ Dynamic universe is {age}d stale:</b> CoinGecko fetches have been "
                          f"failing, so today's run is trading off a cached top-{CRYPTO_TOP_N_FETCH} list. "
                          f"Check network access to api.coingecko.com.</div>")
    elif not any_live_fetch:
        stale_html = ("<div style='background:#ffebee;border-left:4px solid #c62828;padding:10px 14px;"
                       "margin-bottom:14px;font-size:13px;color:#b71c1c;'>"
                       "<b>⚠️ Never successfully fetched the dynamic universe -- trading off the static "
                       "CRYPTO_UNIVERSE fallback list.</b></div>")

    return {"tickers": set(tickers), "stale_html": stale_html, "any_live_fetch": any_live_fetch}


def fetch_crypto_universe() -> dict:
    """Entry point the ASSETS registry calls. Dispatches to the dynamic
    top-N-by-market-cap fetch (default) or the fixed CRYPTO_UNIVERSE list,
    per CRYPTO_USE_DYNAMIC_UNIVERSE."""
    if CRYPTO_USE_DYNAMIC_UNIVERSE:
        return fetch_crypto_universe_dynamic()
    return {"tickers": set(CRYPTO_UNIVERSE), "stale_html": "", "any_live_fetch": True}


# ════════════════════════════════════════════════════════════════════════
#  ASSET REGISTRY -- wires the config block above to the shared engine below
# ════════════════════════════════════════════════════════════════════════

ASSETS: dict[str, dict] = {
    "stock": {
        "label": "Stock", "emoji": "📈", "currency": "₹",
        "funds": STOCK_AVAILABLE_FUNDS, "chunk_size": STOCK_CHUNK_SIZE,
        "force_rerun": STOCK_FORCE_RERUN, "universe_limit": STOCK_UNIVERSE_LIMIT,
        "params": STOCK_PARAMS, "params_meta": STOCK_PARAMS_META,
        "manual_overrides": STOCK_MANUAL_OVERRIDES, "alt_exit_tickers": STOCK_ALT_EXIT_TICKERS,
        "extra_tickers": STOCK_EXTRA_TICKERS, "exclude_tickers": STOCK_EXCLUDE_TICKERS,
        "data_dir": STOCK_DIR, "state_path": f"{STOCK_DIR}/state.json",
        "universe_fetcher": fetch_stock_universe,
    },
    "crypto": {
        "label": "Crypto", "emoji": "🪙", "currency": "$",
        "total_capital": CRYPTO_TOTAL_CAPITAL, "min_ticket_size": CRYPTO_MIN_TICKET_SIZE,
        "force_rerun": CRYPTO_FORCE_RERUN, "universe_limit": CRYPTO_UNIVERSE_LIMIT,
        "params": CRYPTO_PARAMS, "params_meta": CRYPTO_PARAMS_META,
        "manual_overrides": CRYPTO_MANUAL_OVERRIDES, "alt_exit_tickers": CRYPTO_ALT_EXIT_TICKERS,
        "extra_tickers": CRYPTO_EXTRA_TICKERS, "exclude_tickers": CRYPTO_EXCLUDE_TICKERS,
        "data_dir": CRYPTO_DIR, "state_path": f"{CRYPTO_DIR}/state.json",
        "universe_fetcher": fetch_crypto_universe,
    },
}


# ════════════════════════════════════════════════════════════════════════
#  STOCK CYCLE (unchanged engine)
# ════════════════════════════════════════════════════════════════════════

def run_cycle(asset_key: str) -> None:
    if asset_key != "stock":
        # Crypto has its own dedicated cycle (run_cycle_crypto) because its
        # param schema and signal/exit engine are completely different now --
        # routing it through here would crash on the first p['s_ma'] lookup.
        run_cycle_crypto()
        return

    cfg = ASSETS[asset_key]
    label, emoji, currency = cfg["label"], cfg["emoji"], cfg["currency"]
    data_dir, state_path = cfg["data_dir"], cfg["state_path"]
    p, meta = cfg["params"], cfg["params_meta"]
    alt_exit_tickers = cfg["alt_exit_tickers"]
    overrides_cfg = cfg["manual_overrides"]
    cur = lambda x: fmt_curr(x, currency)

    logger.info(f"=== {label} Cycle Started: {datetime.now()} ===")
    logger.info(f"Params: {p}")

    state = load_state(state_path)
    last_run_date = state.get('last_run_date')

    univ = cfg["universe_fetcher"]()
    tickers = set(alt_exit_tickers) | univ["tickers"] | set(cfg["extra_tickers"])
    tickers -= set(cfg["exclude_tickers"])

    tracked = set(state['positions']) | set(state['watchlist'])
    tracked |= set(overrides_cfg.get("force_add_positions", {}).keys())
    tracked |= set(overrides_cfg.get("force_add_watchlist", {}).keys())

    limit = cfg["universe_limit"]
    discovered = sorted(tickers)[:limit] if limit else sorted(tickers)
    universe = sorted(set(discovered) | tracked)

    min_hist = max(p['s_ma'], p['l_ma'], p['sl_ma'], 28) + 50
    ticker_data: dict[str, pd.DataFrame] = {}
    for t in tqdm(universe, desc=f"{label} Data"):
        df = update_ticker_csv(t, data_dir)
        if df is not None and len(df) >= min_hist:
            ticker_data[t] = df
        time.sleep(random.uniform(*DOWNLOAD_DELAY_RANGE))

    if not ticker_data:
        msg = f"❌ No {label.lower()} data fetched — universe or network entirely unavailable. Bot did NOT run today."
        logger.error(msg)
        send_email(f"⚠️ {label} Bot — NO DATA (run skipped)",
                    f"<p>{msg}</p><p>Check network access from this environment.</p>")
        return

    today_str = max(df.index.max() for df in ticker_data.values()).strftime('%Y-%m-%d')
    if state.get('last_run_date') == today_str and not cfg["force_rerun"]:
        logger.info(f"{label} data unchanged. Skipping duplicate email.")
        return

    stale_tickers: list[tuple[str, str]] = []
    fresh_data: dict[str, pd.DataFrame] = {}
    for t, df in ticker_data.items():
        last_date = df.index.max().strftime('%Y-%m-%d')
        if last_date == today_str:
            fresh_data[t] = df
        else:
            stale_tickers.append((t, last_date))
    ticker_data = fresh_data

    stale_open_positions = sorted(t for t, _ in stale_tickers if t in state['positions'])
    stale_frac = len(stale_tickers) / (len(stale_tickers) + len(ticker_data)) if (stale_tickers or ticker_data) else 0.0
    if stale_tickers:
        logger.warning(f"{len(stale_tickers)} ticker(s) had data older than today's session "
                        f"({today_str}) and were skipped: "
                        f"{', '.join(f'{t}({d})' for t, d in stale_tickers[:15])}"
                        f"{' ...' if len(stale_tickers) > 15 else ''}")
    if stale_open_positions:
        logger.error(f"⚠️ {len(stale_open_positions)} OPEN position(s) have stale data today and were "
                      f"NOT re-evaluated for exit: {', '.join(stale_open_positions)}")
    if stale_frac >= 0.5 and ticker_data:
        logger.error(f"⚠️ {stale_frac:.0%} of fetched tickers are stale today -- this looks like a "
                      f"partial data-provider outage, not isolated ticker issues.")

    override_logs = apply_manual_overrides(state, overrides_cfg, today_str)
    for line in override_logs:
        logger.info(line)

    sell_events, wl_changes = [], []
    diag = {"xover": 0, "regime_block": 0, "adx_block": 0, "near_miss": []}
    latest_cache, stop_cache, price_snapshot = {}, {}, {}

    for t, df in ticker_data.items():
        r = process_ticker(t, df, p, state, today_str, last_run_date, diag, alt_exit_tickers)
        if r is None:
            continue
        if r["today_close"] is not None:
            price_snapshot[t] = r["today_close"]
        if r["sell_event"]:
            sell_events.append(r["sell_event"])
        for we in r["wl_events"]:
            tag = f" (on {we['date']}, detected today)" if we["retroactive"] else ""
            if we["action"] == "added":
                wl_changes.append(f"Added {we['ticker']} @ {cur(we['price'])}{tag}")
            else:
                wl_changes.append(f"Removed {we['ticker']} ({we['reason']}){tag}")
        if r["watch_row"] is not None:
            latest_cache[t] = r["watch_row"]
        if r.get("stop_info") is not None:
            stop_cache[t] = r["stop_info"]

    scored = []
    for t, wl in state['watchlist'].items():
        c = latest_cache.get(t)
        if not c: continue
        sc = ((wl['wl_entry_price'] - c['close'])/wl['wl_entry_price'] if p['wl_rank'] == 0
              else (-abs(c['close'] - c['lma'])/c['close'] if p['wl_rank'] == 1
              else (c['close'] - c['sma'])/c['sma']))
        days_tracked = (datetime.strptime(today_str, "%Y-%m-%d") - datetime.strptime(wl['date_added'], "%Y-%m-%d")).days
        scored.append((t, sc, c['close'], days_tracked))
    scored.sort(key=lambda x: -x[1])

    buys = []
    chunks = int(cfg["funds"] // cfg["chunk_size"])
    for _ in range(chunks):
        if not scored: break
        t, sc, pr, _dt = scored.pop(0)
        buys.append([t, f"{sc:.4f}", cur(pr)])
        state['positions'][t] = {"entry_price": float(pr), "entry_date": today_str,
                                  "high_since_entry": float(pr), "days_below_long": 0}
        state['watchlist'].pop(t, None)

    state['last_run_date'] = today_str
    save_state(state, state_path)

    # ── SELLS table ──
    sell_rows = []
    for e in sell_events:
        tag = f" (occurred {e['date']}, detected today)" if e["retroactive"] else ""
        gain_pct = (e['exit_price'] - e['entry']) / e['entry'] * 100
        sell_rows.append({"ticker": e["ticker"], "reason": e["reason"] + tag,
                           "entry_str": cur(e["entry"]), "exit_str": cur(e["exit_price"]),
                           "gain_pct": gain_pct})

    open_rows = []
    total_invested = total_current = 0.0
    status_counts = {"locked_in": 0, "at_risk": 0, "unknown": 0}
    cushion_vals: list[float] = []
    best_pos: tuple[str, float] | None = None
    worst_pos: tuple[str, float] | None = None
    oldest_pos: tuple[str, int] | None = None
    alt_exit_holdings = 0

    for t, pos in sorted(state['positions'].items()):
        cur_price = price_snapshot.get(t)
        entry = pos['entry_price']
        total_invested += entry

        if cur_price is not None:
            total_current += cur_price
            cur_str = cur(cur_price)
            pnl_pct = (cur_price - entry) / entry * 100
            pnl_str = colored_pct(pnl_pct)
            if best_pos is None or pnl_pct > best_pos[1]:
                best_pos = (t, pnl_pct)
            if worst_pos is None or pnl_pct < worst_pos[1]:
                worst_pos = (t, pnl_pct)
        else:
            cur_str, pnl_str = "n/a", "n/a"

        stop = stop_cache.get(t)
        stop_price = stop["price"] if stop else None
        stop_label = stop["label"] if stop else "not recalculated today"

        if stop_price is None:
            status = "unknown"
            stop_str = f"n/a <br><span style='font-size:10.5px;color:#999;'>({stop_label})</span>"
            cushion_str = "n/a"
        else:
            status = "locked_in" if entry < stop_price else "at_risk"
            stop_str = f"{cur(stop_price)} <br><span style='font-size:10.5px;color:#999;'>({stop_label})</span>"
            if cur_price is not None and cur_price != 0:
                cushion_pct = (cur_price - stop_price) / cur_price * 100
                cushion_vals.append(cushion_pct)
                cushion_str = f"{cushion_pct:.1f}%"
            else:
                cushion_str = "n/a"

        status_counts[status] += 1

        entry_date = pos.get('entry_date', today_str)
        held_days = (datetime.strptime(today_str, "%Y-%m-%d") - datetime.strptime(entry_date, "%Y-%m-%d")).days
        if oldest_pos is None or held_days > oldest_pos[1]:
            oldest_pos = (t, held_days)
        if t in alt_exit_tickers:
            alt_exit_holdings += 1

        open_rows.append({
            "ticker": t, "entry_date": entry_date, "held_days": held_days,
            "entry_str": cur(entry), "cur_str": cur_str, "pnl_str": pnl_str,
            "stop_str": stop_str, "cushion_str": cushion_str, "status": status,
        })

    n_positions = len(state['positions'])
    avg_cushion = f"{np.mean(cushion_vals):.1f}%" if cushion_vals else "n/a"
    pnl_summary_pct = (total_current - total_invested) / total_invested * 100 if total_invested else None

    quote_of_day = pick_daily(DAILY_QUOTES, today_str)
    trivia_of_day = pick_daily(DAILY_TRIVIA, today_str)

    stale_data_html = ""
    if stale_tickers:
        shown = stale_tickers[:20]
        stale_data_lines = [f"{t} (last data: {d})" for t, d in shown]
        more = f" ...and {len(stale_tickers) - 20} more" if len(stale_tickers) > 20 else ""
        pos_line = (f"<br><b>⚠️ Includes {len(stale_open_positions)} OPEN position(s) not re-evaluated "
                    f"today: {', '.join(stale_open_positions)}</b>") if stale_open_positions else ""
        outage_line = (f"<br><span style='color:#b71c1c;'><b>{stale_frac:.0%} of fetched tickers are "
                        f"stale -- likely a data-provider issue, not isolated tickers.</b></span>"
                        ) if stale_frac >= 0.5 else ""
        stale_data_html = (f"<div style='background:#fff3e0;border-left:4px solid #e65100;padding:10px 14px;"
                            f"margin-bottom:14px;font-size:13px;color:#5d3d00;'>"
                            f"<b>🕒 {len(stale_tickers)} ticker(s) had old price data today (skipped, "
                            f"not treated as current):</b><br>"
                            f"{', '.join(stale_data_lines)}{more}{pos_line}{outage_line}</div>")
    override_html = (f"<div style='background:#f3e5f5;border-left:4px solid #6a1b9a;padding:10px 14px;"
                      f"margin-bottom:14px;font-size:13px;'><b>Manual overrides applied:</b><br>"
                      f"{'<br>'.join(override_logs)}</div>") if override_logs else ""
    diag_html = (f"<div style='background:#fff3e0;border-left:4px solid #ef6c00;padding:10px 14px;"
                 f"margin-bottom:14px;font-size:13px;'><b>Diagnostic:</b> {diag['xover']} signals seen, "
                 f"{diag['regime_block']} blocked by Regime filter, {diag['adx_block']} blocked by ADX.<br>"
                 f"Near Misses: {', '.join(diag['near_miss']) or 'None'}</div>") if not buys and not scored else ""
    wl_html = (f"<div style='background:#e3f2fd;border-left:4px solid #1565c0;padding:10px 14px;"
               f"margin-bottom:14px;font-size:13px;'><b>Watchlist changes:</b><br>"
               f"{'<br>'.join(wl_changes)}</div>") if wl_changes else ""

    pnl_color = "#1b5e20" if (pnl_summary_pct or 0) >= 0 else "#b71c1c"
    kpi_row = "".join([
        kpi_card("Invested Capital", cur(total_invested), f"{n_positions} open position(s)"),
        kpi_card("Current Value", cur(total_current) if total_current else "n/a"),
        kpi_card("Unrealized P&L", fmt_pct(pnl_summary_pct) if pnl_summary_pct is not None else "n/a",
                 color=pnl_color),
        kpi_card("Portfolio Health",
                 f"{status_counts['locked_in']} 🟢 / {status_counts['at_risk']} 🔴",
                 f"Locked-in vs. At-risk · avg cushion {avg_cushion}"),
    ])

    best_line = f"Best open: <b>{best_pos[0]}</b> {colored_pct(best_pos[1])}" if best_pos else "Best open: n/a"
    worst_line = f"Worst open: <b>{worst_pos[0]}</b> {colored_pct(worst_pos[1])}" if worst_pos else "Worst open: n/a"
    oldest_line = f"Longest held: <b>{oldest_pos[0]}</b> ({oldest_pos[1]}d)" if oldest_pos else "Longest held: n/a"
    extra_line = (f" &nbsp;|&nbsp; Alt-exit holdings: <b>{alt_exit_holdings}/{n_positions}</b>"
                  if alt_exit_tickers and n_positions else "")

    watchlist_rows = [[t, f"{s:.4f}", cur(pr), f"{dt}d"] for t, s, pr, dt in scored]

    oos_display = f"{meta['oos_score']:.2f}" if meta.get('oos_score') is not None else "n/a"

    html = f"""<html><body style="font-family:'Segoe UI',Arial,sans-serif;color:#222;background:#f0f0f3;
    padding:20px 0;margin:0;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr><td align="center">
    <table role="presentation" width="640" cellpadding="0" cellspacing="0"
           style="background:#ffffff;border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,0.08);">

      <tr><td style="background:#0d1b4c;padding:22px 28px;">
        <div style="color:#fff;font-size:20px;font-weight:700;">{emoji} {label} Swing-Trade Signal Bot</div>
        <div style="color:#9fa8da;font-size:12.5px;margin-top:3px;">
          {today_str} &nbsp;·&nbsp; Engine v{meta['engine_version']} &nbsp;·&nbsp; OOS score {oos_display}
        </div>
      </td></tr>

      <tr><td style="padding:20px 28px 4px 28px;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="8"><tr>{kpi_row}</tr></table>
      </td></tr>

      <tr><td style="padding:6px 28px 0 28px;font-size:12.5px;color:#555;">
        {best_line} &nbsp;|&nbsp; {worst_line}<br>{oldest_line}{extra_line}
      </td></tr>

      <tr><td style="padding:16px 28px 0 28px;">
        <div style="background:#fff8e1;border-left:4px solid #f9a825;padding:11px 15px;
                    font-size:13px;color:#5d4a00;border-radius:0 4px 4px 0;">
          <b>💡 Trading Wisdom —</b> {quote_of_day}
        </div>
      </td></tr>

      <tr><td style="padding:10px 28px 0 28px;">
        <div style="background:#e0f7fa;border-left:4px solid #00838f;padding:11px 15px;
                    font-size:13px;color:#004d54;border-radius:0 4px 4px 0;">
          <b>📚 Did You Know —</b> {trivia_of_day}
        </div>
      </td></tr>

      <tr><td style="padding:20px 28px 0 28px;">{univ['stale_html']}{stale_data_html}{override_html}{wl_html}{diag_html}</td></tr>

      <tr><td style="padding:6px 28px 0 28px;">
        <h3 style="background:#2e7d32;color:white;padding:8px 12px;margin:0;font-size:14px;
                   border-radius:4px;">🟢 BUYS</h3>
        <div style="padding:8px 0 16px 0;">{build_table(["Ticker", "Score", "Price"], buys, "No buys today.")}</div>
      </td></tr>

      <tr><td style="padding:0 28px;">
        <h3 style="background:#c62828;color:white;padding:8px 12px;margin:0;font-size:14px;
                   border-radius:4px;">🔴 SELLS</h3>
        <div style="padding:8px 0 16px 0;">{build_sells_table(sell_rows)}</div>
      </td></tr>

      <tr><td style="padding:0 28px;">
        <h3 style="background:#6a1b9a;color:white;padding:8px 12px;margin:0;font-size:14px;
                   border-radius:4px;">📂 OPEN POSITIONS</h3>
        <div style="padding:8px 0 4px 0;">{build_positions_table(open_rows)}</div>
        <div style="font-size:11px;color:#999;padding:2px 2px 16px 2px;">
          🟢 Locked-In = exit trigger is above your entry price (a stop-out can no longer realize a loss).
          🔴 At Risk = exit trigger is still below entry. Cushion % = how far today's price sits above the trigger.
        </div>
      </td></tr>

      <tr><td style="padding:0 28px;">
        <h3 style="background:#1565c0;color:white;padding:8px 12px;margin:0;font-size:14px;
                   border-radius:4px;">📋 WATCHLIST</h3>
        <div style="padding:8px 0 16px 0;">
          {build_table(["Ticker", "Score", "Price", "Tracked"], watchlist_rows, "Watchlist empty.")}
        </div>
      </td></tr>

      <tr><td style="padding:14px 28px 22px 28px;border-top:1px solid #eee;text-align:center;">
        <div style="font-size:11px;color:#aaa;">
          Generated automatically by your Swing-Trade Signal Bot &middot; not financial advice &mdash;
          just your own validated strategy talking back to you.<br>
          "Wealth is built in decades, not days."
        </div>
      </td></tr>

    </table>
    </td></tr></table>
    </body></html>"""

    send_email(f"{emoji} {label} Bot — {today_str} ({len(buys)} Buy / {len(sell_rows)} Sell)", html)


# ════════════════════════════════════════════════════════════════════════
#  CRYPTO CYCLE -- dedicated pipeline (dual-RSI + BTC regime + hybrid exit
#  + pyramiding), separate from the stock cycle above because the param
#  schema and every signal/exit/ranking rule is different now.
# ════════════════════════════════════════════════════════════════════════

def run_cycle_crypto() -> None:
    cfg = ASSETS["crypto"]
    data_dir, state_path = cfg["data_dir"], cfg["state_path"]
    p, meta = cfg["params"], cfg["params_meta"]
    overrides_cfg = cfg["manual_overrides"]
    cur = lambda x: fmt_curr(x, cfg["currency"])

    logger.info(f"=== Crypto Cycle Started: {datetime.now()} ===")
    logger.info(f"Params: {p}")

    state = load_state(state_path)
    last_run_date = state.get('last_run_date')

    univ = cfg["universe_fetcher"]()
    tickers = univ["tickers"] | set(cfg["extra_tickers"])
    tickers -= set(cfg["exclude_tickers"])
    tickers.add("BTC-USD")   # hard-guarantee BTC is always fetched -- the regime filter and the
                              # RSI-reversal panic exit both depend on it, even if it were ever
                              # accidentally dropped from CRYPTO_UNIVERSE / excluded by mistake.

    tracked = set(state['positions']) | set(state['watchlist'])
    tracked |= set(overrides_cfg.get("force_add_positions", {}).keys())
    tracked |= set(overrides_cfg.get("force_add_watchlist", {}).keys())

    limit = cfg["universe_limit"]
    discovered = sorted(tickers)[:limit] if limit else sorted(tickers)
    universe = sorted(set(discovered) | tracked | {"BTC-USD"})

    min_hist = max(p['rsi_slow_len'] + p['rsi_slow_smt'], p['trend_ma_len'],
                    p['btc_ma_len'], p['exit_ma_len'], p['entry_ma_len'], 28) + 50
    ticker_data: dict[str, pd.DataFrame] = {}
    for t in tqdm(universe, desc="Crypto Data"):
        df = update_ticker_csv(t, data_dir)
        if df is not None and len(df) >= min_hist:
            ticker_data[t] = df
        time.sleep(random.uniform(*DOWNLOAD_DELAY_RANGE))

    if "BTC-USD" not in ticker_data:
        msg = ("❌ BTC-USD data unavailable — the crypto engine's regime filter and RSI-reversal "
               "panic exit both depend on it. Bot did NOT run today.")
        logger.error(msg)
        send_email("⚠️ Crypto Bot — NO BTC DATA (run skipped)",
                    f"<p>{msg}</p><p>Check network access to Yahoo Finance from this environment.</p>")
        return
    if not ticker_data:
        msg = "❌ No crypto data fetched — universe or network entirely unavailable. Bot did NOT run today."
        logger.error(msg)
        send_email("⚠️ Crypto Bot — NO DATA (run skipped)",
                    f"<p>{msg}</p><p>Check network access from this environment.</p>")
        return

    today_str = max(df.index.max() for df in ticker_data.values()).strftime('%Y-%m-%d')
    if state.get('last_run_date') == today_str and not cfg["force_rerun"]:
        logger.info("Crypto data unchanged. Skipping duplicate email.")
        return

    stale_tickers: list[tuple[str, str]] = []
    fresh_data: dict[str, pd.DataFrame] = {}
    for t, df in ticker_data.items():
        last_date = df.index.max().strftime('%Y-%m-%d')
        if last_date == today_str:
            fresh_data[t] = df
        else:
            stale_tickers.append((t, last_date))
    ticker_data = fresh_data

    if "BTC-USD" not in ticker_data:
        msg = "❌ BTC-USD data is stale today — the regime filter can't be trusted. Bot did NOT run today."
        logger.error(msg)
        send_email("⚠️ Crypto Bot — STALE BTC DATA (run skipped)", f"<p>{msg}</p>")
        return

    stale_open_positions = sorted(t for t, _ in stale_tickers if t in state['positions'])
    stale_frac = len(stale_tickers) / (len(stale_tickers) + len(ticker_data)) if (stale_tickers or ticker_data) else 0.0
    if stale_tickers:
        logger.warning(f"{len(stale_tickers)} ticker(s) had data older than today's session "
                        f"({today_str}) and were skipped: "
                        f"{', '.join(f'{t}({d})' for t, d in stale_tickers[:15])}"
                        f"{' ...' if len(stale_tickers) > 15 else ''}")
    if stale_open_positions:
        logger.error(f"⚠️ {len(stale_open_positions)} OPEN position(s) have stale data today and were "
                      f"NOT re-evaluated for exit: {', '.join(stale_open_positions)}")

    override_logs = apply_manual_overrides(state, overrides_cfg, today_str)
    for line in override_logs:
        logger.info(line)

    # ── BTC regime series: BTC's own close vs. BTC's own filter-MA, a standing
    # flag (not a simultaneous-cross requirement) -- decision #3 in the
    # backtester's module docstring. ──
    btc_df = ticker_data["BTC-USD"]
    btc_closes = btc_df['Close'].values
    btc_ma = calc_ma(btc_closes, p['btc_ma_len'], p['btc_ma_type'])
    btc_bullish_by_date = {
        btc_df.index[i].strftime('%Y-%m-%d'): (bool(btc_closes[i] > btc_ma[i]) if not np.isnan(btc_ma[i]) else True)
        for i in range(len(btc_df))
    }

    sell_events, wl_changes = [], []
    diag = {"xover": 0, "regime_block": 0, "adx_block": 0, "near_miss": []}
    latest_cache, stop_cache, price_snapshot = {}, {}, {}

    for t, df in ticker_data.items():
        r = process_ticker_crypto(t, df, p, state, today_str, last_run_date, diag, btc_bullish_by_date)
        if r is None:
            continue
        if r["today_close"] is not None:
            price_snapshot[t] = r["today_close"]
        if r["sell_event"]:
            sell_events.append(r["sell_event"])
        for we in r["wl_events"]:
            tag = f" (on {we['date']}, detected today)" if we["retroactive"] else ""
            if we["action"] == "added":
                pyr = " [PYRAMID]" if we.get("pyramid") else ""
                wl_changes.append(f"Added {we['ticker']} @ {cur(we['price'])}{pyr}{tag}")
            elif we["action"] == "partial_exit":
                wl_changes.append(f"{we['ticker']}: {we['reason']}{tag}")
            else:
                wl_changes.append(f"Removed {we['ticker']} ({we['reason']}){tag}")
        if r["watch_row"] is not None:
            latest_cache[t] = r["watch_row"]
        if r.get("stop_info") is not None:
            stop_cache[t] = r["stop_info"]

    # ── RANKING (mirrors the backtester's wl_rank methods 0/1/2) ──
    scored = []
    for t, wl in state['watchlist'].items():
        c = latest_cache.get(t)
        if not c: continue
        if p['wl_rank'] == 0 and c['entry_price'] > 0:
            sc = (c['entry_price'] - c['close']) / c['entry_price']
        elif p['wl_rank'] == 1 and c.get('fast') and c['fast'] and not np.isnan(c['fast']) and c['fast'] != 0:
            sc = -abs(c['fast'] - c['slow']) / c['fast']
        elif p['wl_rank'] == 2 and c.get('slow') and not np.isnan(c['slow']) and c['slow'] != 0:
            sc = (c['fast'] - c['slow']) / c['slow']
        else:
            sc = 0.0
        days_tracked = (datetime.strptime(today_str, "%Y-%m-%d")
                         - datetime.strptime(state['watchlist'][t]['date_added'], "%Y-%m-%d")).days
        scored.append((t, sc, c['close'], days_tracked, c.get('is_pyramid', False)))
    scored.sort(key=lambda x: -x[1])

    # ── CAPITAL DEPLOYMENT -- equal-weight-target sizing, ported directly
    # from the backtester's decision #16: cash_pool derived fresh each run
    # as CRYPTO_TOTAL_CAPITAL minus the cost basis of every open position
    # (no separately-tracked, driftable ledger -- you edit CRYPTO_TOTAL_
    # CAPITAL directly whenever you deposit/withdraw, and this always
    # reflects it correctly). Each suggested buy is sized at (cash_pool +
    # invested_cost) / n_eligible -- COST BASIS, not mark-to-market, so one
    # position's unrealized paper gain can't inflate the size of the next,
    # unrelated buy (same reasoning as the backtester's profit-concentration
    # gate, just reachable through sizing instead of scoring). n_eligible is
    # len(ticker_data): every coin with usable data today, mirroring the
    # backtester's eligible_mask[d,:] (today's tradeable universe, not just
    # today's watchlist candidates -- dividing by the full universe is what
    # stops one entry from starving the others). Pyramiding into an existing
    # position averages cost basis and only ever RAISES the stop, exactly
    # mirroring simulate_portfolio_crypto's deployment loop. A "stale"
    # pyramid ticket (the position it was meant to add to already exited
    # before capital was free) is dropped without spending anything on it,
    # matching the backtester's `if wl_is_pyramid and not in_pos: drop`.
    buys = []
    n_eligible = max(1, len(ticker_data))
    while scored:
        t, sc, pr, _dt, is_pyr = scored.pop(0)
        if is_pyr and t not in state['positions']:
            state['watchlist'].pop(t, None)
            continue
        if not (pr > 0):
            state['watchlist'].pop(t, None)
            continue

        invested_cost = sum(ps['shares'] * ps['entry_price'] for ps in state['positions'].values())
        cash_pool = cfg["total_capital"] - invested_cost
        target_size = (cash_pool + invested_cost) / n_eligible
        buy_amount = target_size if target_size <= cash_pool else cash_pool
        if buy_amount < cfg["min_ticket_size"]:
            break   # early on, when capital is genuinely thin, it's fine to miss the entry

        new_shares = buy_amount / pr
        atr_last = ticker_data[t]['ATR'].values[-1] if t in ticker_data else np.nan
        # FIXED: fall back to YESTERDAY's ATR before the flat 2%-of-price floor,
        # matching the backtester's CAPITAL DEPLOYMENT chain exactly (today ->
        # yesterday -> 2% floor). Previously this jumped straight from today's
        # ATR to the 2% floor, skipping the middle step.
        atr_prev = (ticker_data[t]['ATR'].values[-2]
                    if t in ticker_data and len(ticker_data[t]) >= 2 else np.nan)

        if is_pyr and t in state['positions']:
            pos = state['positions'][t]
            old_cost = pos['shares'] * pos['entry_price']
            new_cost = new_shares * pr
            pos['shares'] += new_shares
            pos['entry_price'] = (old_cost + new_cost) / pos['shares']
            # FIXED: use the same today-then-yesterday-then-2% ATR fallback as
            # everywhere else, instead of silently skipping the ratchet entirely
            # when today's ATR happens to be unavailable -- the backtester's
            # pyramid-ratchet block always computes SOME candidate stop.
            ratchet_atr = atr_last if (not np.isnan(atr_last) and atr_last > 0) else atr_prev
            if not (not np.isnan(ratchet_atr) and ratchet_atr > 0):
                ratchet_atr = pr * 0.02
            candidate_sl = pos['entry_price'] - ratchet_atr * p['sl_mult']
            if candidate_sl > pos['stop_loss_price']:
                pos['stop_loss_price'] = candidate_sl
            buys.append([t + " [pyramid]", f"{sc:.4f}", cur(pr), cur(buy_amount)])
        else:
            entry_atr = atr_last if (not np.isnan(atr_last) and atr_last > 0) else atr_prev
            if not (not np.isnan(entry_atr) and entry_atr > 0):
                entry_atr = pr * 0.02
            state['positions'][t] = {
                "entry_price": float(pr), "entry_date": today_str,
                "shares": float(new_shares), "high_since_entry": float(pr),
                "stop_loss_price": float(pr - entry_atr * p['sl_mult']),
                "tp_trigger_price": float(pr + entry_atr * p['tp_mult']),
                "half_sold": False,
            }
            buys.append([t, f"{sc:.4f}", cur(pr), cur(buy_amount)])

        state['watchlist'].pop(t, None)

    state['last_run_date'] = today_str
    save_state(state, state_path)

    # ── SELLS table ──
    sell_rows = []
    for e in sell_events:
        tag = f" (occurred {e['date']}, detected today)" if e["retroactive"] else ""
        gain_pct = (e['exit_price'] - e['entry']) / e['entry'] * 100
        sell_rows.append({"ticker": e["ticker"], "reason": e["reason"] + tag,
                           "entry_str": cur(e["entry"]), "exit_str": cur(e["exit_price"]),
                           "gain_pct": gain_pct})

    # ── OPEN POSITIONS table + portfolio-health stats ──
    open_rows = []
    total_invested = total_current = 0.0
    status_counts = {"locked_in": 0, "at_risk": 0, "unknown": 0}
    cushion_vals: list[float] = []
    best_pos: tuple[str, float] | None = None
    worst_pos: tuple[str, float] | None = None
    oldest_pos: tuple[str, int] | None = None

    for t, pos in sorted(state['positions'].items()):
        cur_price = price_snapshot.get(t)
        entry = pos['entry_price']
        shares = pos.get('shares', 0.0)
        total_invested += entry * shares

        if cur_price is not None:
            total_current += cur_price * shares
            cur_str = cur(cur_price)
            pnl_pct = (cur_price - entry) / entry * 100
            pnl_str = colored_pct(pnl_pct)
            if best_pos is None or pnl_pct > best_pos[1]:
                best_pos = (t, pnl_pct)
            if worst_pos is None or pnl_pct < worst_pos[1]:
                worst_pos = (t, pnl_pct)
        else:
            cur_str, pnl_str = "n/a", "n/a"

        stop = stop_cache.get(t)
        stop_price = stop["price"] if stop else None
        stop_label = stop["label"] if stop else "not recalculated today"

        if stop_price is None:
            status = "unknown"
            stop_str = f"n/a <br><span style='font-size:10.5px;color:#999;'>({stop_label})</span>"
            cushion_str = "n/a"
        else:
            status = "locked_in" if entry < stop_price else "at_risk"
            stop_str = f"{cur(stop_price)} <br><span style='font-size:10.5px;color:#999;'>({stop_label})</span>"
            if cur_price is not None and cur_price != 0:
                cushion_pct = (cur_price - stop_price) / cur_price * 100
                cushion_vals.append(cushion_pct)
                cushion_str = f"{cushion_pct:.1f}%"
            else:
                cushion_str = "n/a"

        status_counts[status] += 1

        entry_date = pos.get('entry_date', today_str)
        held_days = (datetime.strptime(today_str, "%Y-%m-%d") - datetime.strptime(entry_date, "%Y-%m-%d")).days
        if oldest_pos is None or held_days > oldest_pos[1]:
            oldest_pos = (t, held_days)

        half_tag = " (half-sold)" if pos.get('half_sold') else ""
        open_rows.append({
            "ticker": t + half_tag, "entry_date": entry_date, "held_days": held_days,
            "entry_str": cur(entry), "cur_str": cur_str, "pnl_str": pnl_str,
            "stop_str": stop_str, "cushion_str": cushion_str, "status": status,
        })

    n_positions = len(state['positions'])
    avg_cushion = f"{np.mean(cushion_vals):.1f}%" if cushion_vals else "n/a"
    pnl_summary_pct = (total_current - total_invested) / total_invested * 100 if total_invested else None

    quote_of_day = pick_daily(DAILY_QUOTES, today_str)
    trivia_of_day = pick_daily(DAILY_TRIVIA, today_str)

    btc_today_bullish = btc_bullish_by_date.get(today_str, True)
    regime_html = (f"<div style='background:{'#e8f5e9' if btc_today_bullish else '#ffebee'};"
                   f"border-left:4px solid {'#2e7d32' if btc_today_bullish else '#c62828'};"
                   f"padding:10px 14px;margin-bottom:14px;font-size:13px;"
                   f"color:{'#1b5e20' if btc_today_bullish else '#b71c1c'};'>"
                   f"<b>{'🟢 BTC Regime: BULLISH' if btc_today_bullish else '🔴 BTC Regime: BEARISH'}</b> "
                   f"— {'new entries allowed' if btc_today_bullish else 'new entries blocked, panic exits active'} "
                   f"(BTC {p['btc_ma_len']}-{_MA_TYPE_NAMES.get(p['btc_ma_type'], '?')} filter)</div>")

    stale_data_html = ""
    if stale_tickers:
        shown = stale_tickers[:20]
        stale_data_lines = [f"{t} (last data: {d})" for t, d in shown]
        more = f" ...and {len(stale_tickers) - 20} more" if len(stale_tickers) > 20 else ""
        pos_line = (f"<br><b>⚠️ Includes {len(stale_open_positions)} OPEN position(s) not re-evaluated "
                    f"today: {', '.join(stale_open_positions)}</b>") if stale_open_positions else ""
        stale_data_html = (f"<div style='background:#fff3e0;border-left:4px solid #e65100;padding:10px 14px;"
                            f"margin-bottom:14px;font-size:13px;color:#5d3d00;'>"
                            f"<b>🕒 {len(stale_tickers)} ticker(s) had old price data today (skipped, "
                            f"not treated as current):</b><br>"
                            f"{', '.join(stale_data_lines)}{more}{pos_line}</div>")
    override_html = (f"<div style='background:#f3e5f5;border-left:4px solid #6a1b9a;padding:10px 14px;"
                      f"margin-bottom:14px;font-size:13px;'><b>Manual overrides applied:</b><br>"
                      f"{'<br>'.join(override_logs)}</div>") if override_logs else ""
    diag_html = (f"<div style='background:#fff3e0;border-left:4px solid #ef6c00;padding:10px 14px;"
                 f"margin-bottom:14px;font-size:13px;'><b>Diagnostic:</b> {diag['xover']} signals seen, "
                 f"{diag['regime_block']} blocked by trend/BTC filter, {diag['adx_block']} blocked by ADX.</div>"
                 ) if not buys and not scored else ""
    wl_html = (f"<div style='background:#e3f2fd;border-left:4px solid #1565c0;padding:10px 14px;"
               f"margin-bottom:14px;font-size:13px;'><b>Watchlist changes:</b><br>"
               f"{'<br>'.join(wl_changes)}</div>") if wl_changes else ""

    pnl_color = "#1b5e20" if (pnl_summary_pct or 0) >= 0 else "#b71c1c"

    cash_available = cfg["total_capital"] - total_invested
    kpi_row = "".join([
        kpi_card("Invested Capital", cur(total_invested), f"of {cur(cfg['total_capital'])} total · {n_positions} position(s)"),
        kpi_card("Cash Available", cur(cash_available), "basis for next suggested allocation"),
        kpi_card("Current Value", cur(total_current) if total_current else "n/a"),
        kpi_card("Unrealized P&L", fmt_pct(pnl_summary_pct) if pnl_summary_pct is not None else "n/a",
                 color=pnl_color),
        kpi_card("Portfolio Health",
                 f"{status_counts['locked_in']} 🟢 / {status_counts['at_risk']} 🔴",
                 f"Locked-in vs. At-risk · avg cushion {avg_cushion}"),
    ])

    best_line = f"Best open: <b>{best_pos[0]}</b> {colored_pct(best_pos[1])}" if best_pos else "Best open: n/a"
    worst_line = f"Worst open: <b>{worst_pos[0]}</b> {colored_pct(worst_pos[1])}" if worst_pos else "Worst open: n/a"
    oldest_line = f"Longest held: <b>{oldest_pos[0]}</b> ({oldest_pos[1]}d)" if oldest_pos else "Longest held: n/a"

    watchlist_rows = [[t + (" [PYRAMID]" if is_pyr else ""), f"{s:.4f}", cur(pr), f"{dt}d"]
                       for t, s, pr, dt, is_pyr in scored]

    oos_display = f"{meta['oos_score']:.2f}" if meta.get('oos_score') is not None else "n/a"

    html = f"""<html><body style="font-family:'Segoe UI',Arial,sans-serif;color:#222;background:#f0f0f3;
    padding:20px 0;margin:0;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr><td align="center">
    <table role="presentation" width="640" cellpadding="0" cellspacing="0"
           style="background:#ffffff;border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,0.08);">

      <tr><td style="background:#0d1b4c;padding:22px 28px;">
        <div style="color:#fff;font-size:20px;font-weight:700;">🪙 Crypto Swing-Trade Signal Bot</div>
        <div style="color:#9fa8da;font-size:12.5px;margin-top:3px;">
          {today_str} &nbsp;·&nbsp; Engine v{meta['engine_version']} &nbsp;·&nbsp; IS score {meta.get('is_score', 'n/a')} &nbsp;·&nbsp; OOS score {oos_display}
        </div>
      </td></tr>

      <tr><td style="padding:20px 28px 4px 28px;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="8"><tr>{kpi_row}</tr></table>
      </td></tr>

      <tr><td style="padding:6px 28px 0 28px;font-size:12.5px;color:#555;">
        {best_line} &nbsp;|&nbsp; {worst_line}<br>{oldest_line}
      </td></tr>

      <tr><td style="padding:16px 28px 0 28px;">
        <div style="background:#fff8e1;border-left:4px solid #f9a825;padding:11px 15px;
                    font-size:13px;color:#5d4a00;border-radius:0 4px 4px 0;">
          <b>💡 Trading Wisdom —</b> {quote_of_day}
        </div>
      </td></tr>

      <tr><td style="padding:10px 28px 0 28px;">
        <div style="background:#e0f7fa;border-left:4px solid #00838f;padding:11px 15px;
                    font-size:13px;color:#004d54;border-radius:0 4px 4px 0;">
          <b>📚 Did You Know —</b> {trivia_of_day}
        </div>
      </td></tr>

      <tr><td style="padding:20px 28px 0 28px;">{univ['stale_html']}{regime_html}{stale_data_html}{override_html}{wl_html}{diag_html}</td></tr>

      <tr><td style="padding:6px 28px 0 28px;">
        <h3 style="background:#2e7d32;color:white;padding:8px 12px;margin:0;font-size:14px;
                   border-radius:4px;">🟢 BUYS</h3>
        <div style="padding:8px 0 16px 0;">{build_table(["Ticker", "Score", "Price", "Suggested Allocation"], buys, "No buys today.")}</div>
      </td></tr>

      <tr><td style="padding:0 28px;">
        <h3 style="background:#c62828;color:white;padding:8px 12px;margin:0;font-size:14px;
                   border-radius:4px;">🔴 SELLS</h3>
        <div style="padding:8px 0 16px 0;">{build_sells_table(sell_rows)}</div>
      </td></tr>

      <tr><td style="padding:0 28px;">
        <h3 style="background:#6a1b9a;color:white;padding:8px 12px;margin:0;font-size:14px;
                   border-radius:4px;">📂 OPEN POSITIONS</h3>
        <div style="padding:8px 0 4px 0;">{build_positions_table(open_rows)}</div>
        <div style="font-size:11px;color:#999;padding:2px 2px 16px 2px;">
          🟢 Locked-In = exit trigger is above your entry price (a stop-out can no longer realize a loss).
          🔴 At Risk = exit trigger is still below entry. Cushion % = how far today's price sits above the trigger.
          "(half-sold)" = TP already hit once; remaining half is trailing on the breakeven+ATR stop.
        </div>
      </td></tr>

      <tr><td style="padding:0 28px;">
        <h3 style="background:#1565c0;color:white;padding:8px 12px;margin:0;font-size:14px;
                   border-radius:4px;">📋 WATCHLIST</h3>
        <div style="padding:8px 0 16px 0;">
          {build_table(["Ticker", "Score", "Price", "Tracked"], watchlist_rows, "Watchlist empty.")}
        </div>
      </td></tr>

      <tr><td style="padding:14px 28px 22px 28px;border-top:1px solid #eee;text-align:center;">
        <div style="font-size:11px;color:#aaa;">
          Generated automatically by your Swing-Trade Signal Bot &middot; not financial advice &mdash;
          just your own validated strategy talking back to you.<br>
          "Wealth is built in decades, not days."
        </div>
      </td></tr>

    </table>
    </td></tr></table>
    </body></html>"""

    send_email(f"🪙 Crypto Bot — {today_str} ({len(buys)} Buy / {len(sell_rows)} Sell)", html)


# ════════════════════════════════════════════════════════════════════════
#  EXECUTION
# ════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily Swing-Trade Signal Bot (stocks + crypto)")
    parser.add_argument("--dry-run", action="store_true", help="Run normally but log emails instead of sending them")
    args, _ = parser.parse_known_args()
    return args


def run_cycle_safe(name: str, fn: Callable) -> None:
    try:
        fn()
    except Exception:
        tb = traceback.format_exc()
        logger.error(f"{name} cycle crashed:\n{tb}")
        send_email(f"⚠️ {name} Bot CRASHED — {datetime.now():%Y-%m-%d}",
                    f"<p>The {name.lower()} cycle raised an unhandled exception and did not complete.</p>"
                    f"<pre style='white-space:pre-wrap;font-size:12px;background:#f5f5f5;padding:10px;'>{tb}</pre>")


if __name__ == "__main__":
    args = parse_args()
    DRY_RUN = args.dry_run
    if DRY_RUN:
        logger.info("Running in --dry-run mode: emails will be logged, not sent.")
    run_cycle_safe("Stock", lambda: run_cycle("stock"))
    run_cycle_safe("Crypto", lambda: run_cycle_crypto())