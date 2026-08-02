"""
DIAMOND EXTRACTOR -- BACKTESTING ENGINE v5.0
=============================================
NSE equity trend-following strategy: unified cash pool funded by a fixed
monthly SIP, MA-crossover entries with an optional trend/pullback filter,
two-phase exits (signal at close, fill at next open), pyramiding into
existing winners, separate exit logic for a disclosed dividend-stock
watchlist, Bayesian (Optuna/TPE) hyperparameter search over IS data,
walk-forward OOS validation, and overfitting diagnostics (DSR + a
descriptive score-distribution check).

HOW TO USE
  1. (Optional, one-time) run fetch_and_build_point_in_time_universe() to
     reduce survivorship bias, then set POINT_IN_TIME_UNIVERSE_FILE below.
  2. Run this file directly: `python main.py`.
  3. It searches BAYESIAN_TRIALS parameter combos on the IS window, then
     walk-forward validates the best one on held-out OOS data. A strategy
     is only saved to BEST_PARAMS_FILE if it clears both the OOS and the
     temporal (random-SIP-day) robustness gates -- see SCORING below.
  4. Re-running resumes from the last saved champion (BEST_PARAMS_FILE, or
     the IS-only INTERMEDIATE_FILE if no fully-validated champion exists
     yet) and keeps searching from there.

SCORING
  score = [Calmar*0.30 + Sortino(cap 4.0)*0.10 + IR*0.30 + EV_in_R*0.15
           + WinRate_bonus*0.15] * stat_confidence * drawdown_penalty
  Hard gates (any failure -> score = -999): min trades/months, max_dd <=
  50%, profit factor >= 1.10, ROI > 0 and > 1.1x benchmark ROI, win rate
  >= MIN_WIN_RATE_GATE. Deployable only if BOTH OOS robustness (OOS score /
  IS score) and temporal robustness (score under 15 randomized-SIP-day
  reruns / baseline) clear their thresholds (50% by default).

KEY LIMITATIONS (read before trusting any absolute number)
  - Survivorship bias: universe defaults to TODAY's constituents applied
    backward, unless POINT_IN_TIME_UNIVERSE_FILE is set (Nifty 50 only --
    see fetch_and_build_point_in_time_universe()).
  - One 70/30 IS/OOS split is a single, high-variance robustness estimate,
    mitigated (not solved) by the sub-period breakdown, the two overfitting
    diagnostics, and the temporal robustness check.
  - Deflated Sharpe Ratio is computed on Sharpe, not on the composite score
    Optuna actually optimizes (no published deflation theory exists for an
    arbitrary composite score) -- treat it as one indicative angle.
  - max_dd uses the raw (cash-inclusive) value series; Sharpe/Sortino/skew/
    kurtosis use the cash-flow-adjusted series. Both are intentional, and
    answer different questions -- see simulate_portfolio.

CHANGELOG (v5.0 vs v4.2 -- full v1-v4.1 history trimmed from this header,
still available in version control)
  - Fixed Sharpe/Sortino/skew/kurtosis to exclude SIP deposits from the
    daily return series (they were being counted as investment profit).
  - Cash-flow/monthly-return arrays now sized off n_days instead of a
    fixed cap, so they can no longer silently overwrite old data.
  - Optuna now actually uses TPE + multivariate/group sampling and
    actually passes n_jobs (both were previously claimed but not wired up).
  - Added temporal (random-SIP-day) robustness check as a second save gate.
  - OOS robustness ratio is now continuous instead of clipping any
    non-positive OOS result to a flat 0%.
  - Skew/kurtosis feeding the DSR diagnostic are now computed from the
    champion's real returns instead of hardcoded placeholders.
  - Added a best-effort point-in-time Nifty 50 universe fetcher.
  - Fixed MIN_WIN_RATE_GATE (0.50 -> 0.45) to match its own documentation.
  - Fixed a benchmark-ROI denominator mismatch that could hard-reject a
    genuinely sound OOS result (see evaluate_params' bench_roi comment).
  - Console output cleaned up: OOS/IS gate failures now print the actual
    metrics and which gate(s) they missed, instead of just "-999".
  - ROI-vs-benchmark hard gate set to 1.10x (was drifting between 1.0x and
    1.2x across manual edits) -- a deliberate, modest-but-real outperformance
    margin: enough to justify running an active strategy over a passive
    index fund (costs, taxes, effort, model risk), not so strict that a
    noisy ~3.4y OOS window gets rejected on sampling variance alone.
    diagnose_hard_gates() now reads this from one constant so the printed
    diagnostic can never drift out of sync with the actual gate again.
  - Neighborhood stability check now perturbs every parameter that's
    actually ACTIVE for the champion's specific entry/exit configuration
    (sl_ma, div_exit_v, adx_thresh), not just s_ma/l_ma/n_trail_p/n_atr_m.
    Previously, a champion using the crossunder exit (n_exit_m==0) would
    have 4 of 7 perturbations silently test unused parameters and trivially
    pass, understating how brittle the fit actually was.
"""

import math
import json
import os
import io
import threading
import warnings
from collections import OrderedDict
from statistics import NormalDist

import numpy as np
import pandas as pd
import requests
from numba import njit
from tqdm import tqdm

try:
    import yfinance as yf
except ImportError:
    yf = None
    print("yfinance not found. Install with: pip install yfinance")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    print("optuna not found. Install with: pip install optuna")
    OPTUNA_AVAILABLE = False

warnings.filterwarnings('ignore')

# ==========================================
# 1. UNIVERSE DEFINITION
# ==========================================

DIVIDEND_KINGS_FALLBACK = {
    'ITC.NS', 'COALINDIA.NS', 'ONGC.NS', 'POWERGRID.NS', 'NTPC.NS', 'PFC.NS',
    'RECLTD.NS', 'VEDL.NS', 'GAIL.NS', 'BPCL.NS', 'IOC.NS', 'PETRONET.NS',
    'SAIL.NS', 'NHPC.NS', 'NMDC.NS', 'HINDZINC.NS', 'CASTROLIND.NS'
}

# Point-in-time membership hook (survivorship-bias mitigation). Point this
# at a JSON file shaped {"YYYY-MM-DD": ["RELIANCE.NS", ...]} to restrict new
# watchlist entries to stocks actually in-index on that date. None = every
# stock eligible every day (v3/v4 default, disclosed survivorship bias).
POINT_IN_TIME_UNIVERSE_FILE = None


def fetch_dynamic_universe():
    print("Fetching Nifty universe (50 / Next 50 / Midcap 150)...")
    print("NOTE: uses TODAY's constituents applied back to", START_DATE, "-- survivorship-biased, see module docstring.")
    
    # Spoof a real browser to bypass NSE bot protection
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5'
    }
    
    # Use the more reliable niftyindices.com endpoints
    urls = [
        "https://niftyindices.com/IndexConstituent/ind_nifty50list.csv",
        "https://niftyindices.com/IndexConstituent/ind_niftynext50list.csv",
        "https://niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv"
    ]
    
    tickers = set()
    for url in urls:
        try:
            req = requests.get(url, headers=headers, timeout=15)
            if req.status_code == 200:
                df = pd.read_csv(io.StringIO(req.text))
                for symbol in df['Symbol']:
                    # Filter out any weird spaces/NaNs just in case
                    clean_symbol = str(symbol).strip()
                    if clean_symbol and clean_symbol != 'nan':
                        tickers.add(f"{clean_symbol}.NS")
            else:
                print(f"Failed to fetch {url} - Status Code: {req.status_code}")
        except Exception as e:
            print(f"Error fetching {url}: {e}")

    if len(tickers) < 100:
        print("Fallback to predefined list... (NSE likely blocked the request)")
        tickers = set(['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS',
                       'ICICIBANK.NS', 'ITC.NS'] + list(DIVIDEND_KINGS_FALLBACK))
    
    return list(tickers)

# ==========================================
# 2. CONFIGURATIONS & SAVE SYSTEM
# ==========================================

START_DATE        = "2015-01-01"
BEST_PARAMS_FILE  = "best_params_v4.json"          # fully OOS+temporal-validated champion
INTERMEDIATE_FILE = "current_is_champion_v4.json"  # IS-only champion, saved mid-search

ENGINE_VERSION = "5.0.0"   # stamped into save_winner() output

WFO_IS_PCT   = 0.70
WFO_OOS_PCT  = 0.30

BAYESIAN_TRIALS      = 20_000
NEIGHBOR_THRESHOLD   = 0.75
MIN_TRADES_GATE      = 40
MIN_MONTHS_GATE      = 24
MIN_WIN_RATE_GATE    = 0.5
MONTHLY_SIP          = 8_000
MIN_TICKET_SIZE      = MONTHLY_SIP   # NEW (ported from the crypto engine's decision #16):
                                      # floor under a now-VARIABLE ticket size (see
                                      # decision #18 in the changelog) -- kept EXACTLY
                                      # equal to the fixed Rs.8,000 SIP figure per explicit
                                      # instruction, not loosened just because it's now variable.

ROBUSTNESS_DEPLOY_THRESHOLD = 0.50   # OOS-score / IS-score gate, and printed banding

# ROI must beat the benchmark's ROI by at least this multiple to pass the
# hard gate (in compute_score_v4) -- see the CHANGELOG entry above for why
# 1.10 was chosen. diagnose_hard_gates() reads this same constant so the
# printed diagnostic can never drift out of sync with the real gate again.
BENCH_OUTPERFORMANCE_MULT = 1.10

HIGH_CASH_FRACTION_THRESHOLD = 0.30  # reporting-only threshold, doesn't affect simulation

# ── NEW (ported from the crypto engine, decisions #18-20): recency-weighted
# consistency/regime scoring + a profit-concentration hard gate, both aimed
# at "don't reward a strategy that got lucky in one early bull year and
# coasted on the compounding." Nifty plays the same dual role BTC plays in
# the crypto engine: benchmark AND bull/bear-year classifier at once.
RECENCY_WEIGHT_MIN        = 0.80
RECENCY_WEIGHT_MAX        = 1.00
BULL_YEAR_NIFTY_THRESHOLD = 0.10    # Nifty annual return above this => "bull year"
BEAR_YEAR_NIFTY_THRESHOLD = -0.10   # Nifty annual return below this => "bear year"
CONCENTRATION_GATE        = 0.55    # hard reject if one calendar year > 55% of total profit
YEAR_FULL_COVERAGE_DAYS   = 252.0   # NSE trading days (vs the crypto engine's 365)
MIN_YEAR_COVERAGE_FOR_SCORING = 0.75  # years below this coverage_frac are excluded from
                                        # the CONSISTENCY/REGIME terms -- IRR annualized from
                                        # too few days isn't trustworthy. The concentration
                                        # gate is unaffected (raw nominal dollars, not a rate).

# ── Composite score weights (sum to 1.00) -- rebalanced from v4.2's flat
# Calmar 0.30/Sortino 0.10/IR 0.30/EV 0.15/WR 0.15 to make room for the two
# new additive terms below, identical rebalancing to the crypto engine's.
W_CALMAR      = 0.20
W_SORTINO     = 0.10
W_IR          = 0.20
W_EV          = 0.10
W_WR_BONUS    = 0.10
W_CONSISTENCY = 0.20
W_REGIME      = 0.10

K_STD_PENALTY   = 1.00   # consistency term: penalty coefficient on recency-weighted std-dev
K_WORST_PENALTY = 0.50   # consistency term: penalty coefficient on the single worst year
W_BULL_CAPTURE  = 0.50   # regime term split
W_BEAR_DEFENSE  = 0.50

# ── NEW: optional watchlist-age weighting -- a candidate's rank (from
# wl_rank_method) can optionally be boosted by how long it's been sitting
# in the watchlist, so a strategy-favored-but-fresh candidate doesn't
# perpetually starve an older one out of ever getting funded. Off by
# default (use_wl_age_weight=False baseline); searched as a toggle so
# Optuna keeps it only if it actually helps. WL_AGE_NORM_DAYS is the
# saturation point for the age boost (age_frac caps at 1.0 once a
# candidate has waited this many trading days) -- kept as a fixed
# constant rather than another optimized dimension to keep the search
# space from growing further; revisit if a champion's age weighting looks
# artificially capped.
WL_AGE_NORM_DAYS = 20.0

OPTUNA_N_JOBS = max(1, (os.cpu_count() or 2) - 1)  # set to 1 for bit-exact reproducibility

# Temporal (random-SIP-day) robustness check: reruns the chosen champion
# with an independently-randomized SIP day each month, to catch a strategy
# that's really just curve-fit to "SIP always lands on the 1st".
TEMPORAL_ROBUSTNESS_RUNS      = 15
TEMPORAL_ROBUSTNESS_THRESHOLD = 0.65

MA_CACHE_MAX_ENTRIES = 20_000   # bounded LRU cache size for get_ma_cached()

# Transaction costs -- NSE delivery trade, zero-brokerage discount broker,
# statutory rates as of mid-2026. Re-verify against your own contract note.
BUY_STATUTORY_PCT  = 0.00119   # STT + stamp duty + exchange + SEBI + GST
SELL_STATUTORY_PCT = 0.00104   # STT + exchange + SEBI + GST
SLIPPAGE_PCT       = 0.00100   # execution-slippage allowance
DP_FLAT_FEE_RS     = 20.0      # flat depository charge, once per exit
BUY_COST_PCT  = BUY_STATUTORY_PCT  + SLIPPAGE_PCT
SELL_COST_PCT = SELL_STATUTORY_PCT + SLIPPAGE_PCT

# Idle cash yield -- conservative proxy for liquid-fund returns. 0.0
# reproduces the old zero-yield assumption exactly.
ANNUAL_CASH_YIELD = 0.00

TRADING_DAYS_PER_YEAR = 252.0   # basis for all annualization in this file


def load_previous_winner(filename=BEST_PARAMS_FILE):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                return data.get('oos_score', -999999), data.get('is_score', -999999), data.get('params', None)
        except Exception:
            pass
    return -999999, -999999, None


def save_winner(oos_score, is_score, params, filename=BEST_PARAMS_FILE, robustness_ratio=None):
    clean_params = {k: float(v) if isinstance(v, (float, np.floating)) else int(v)
                    for k, v in params.items()}
    # Callers that already computed the robustness ratio (post-WFO) pass it
    # through so the saved JSON always matches what was printed/gated on.
    if robustness_ratio is None:
        robustness_ratio = float(oos_score / is_score) if is_score > 0 else 0.0
    data = {
        'engine_version': ENGINE_VERSION,
        'oos_score': float(oos_score),
        'is_score':  float(is_score),
        'robustness_ratio': float(robustness_ratio),
        'params': clean_params
    }
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving {filename}: {e}")


# ==========================================
# 3. FAST INDICATORS (numba, NaN-safe)
# ==========================================
# fastmath is deliberately off -- it breaks np.isnan() checks this module
# relies on for newly-listed stocks. nogil=True lets Optuna's n_jobs>1
# threading actually parallelize these calls.

@njit(nogil=True, cache=True)
def calc_ma(prices, period, ma_type):
    n = len(prices)
    res = np.empty(n)
    res[:] = np.nan

    # Find first real (non-NaN) data point -- a stock listed after
    # START_DATE has leading NaN that must not poison the MA's state.
    start = 0
    while start < n and np.isnan(prices[start]):
        start += 1
    if n - start < period:
        return res

    if ma_type == 0:  # SMA
        w_sum = np.sum(prices[start:start + period])
        res[start + period - 1] = w_sum / period
        for i in range(start + period, n):
            w_sum = w_sum - prices[i - period] + prices[i]
            res[i] = w_sum / period
    elif ma_type == 1 or ma_type == 2:  # EMA / DEMA
        ema1 = np.empty(n)
        ema1[:] = np.nan
        ema1[start + period - 1] = np.mean(prices[start:start + period])
        mult = 2 / (period + 1)
        for i in range(start + period, n):
            ema1[i] = (prices[i] - ema1[i - 1]) * mult + ema1[i - 1]
        if ma_type == 1:
            res = ema1
        else:  # DEMA
            ema2 = np.empty(n)
            ema2[:] = np.nan
            if start + period * 2 - 2 < n:
                ema2[start + period * 2 - 2] = np.mean(ema1[start + period - 1:start + period * 2 - 1])
                for i in range(start + period * 2 - 1, n):
                    ema2[i] = (ema1[i] - ema2[i - 1]) * mult + ema2[i - 1]
                for i in range(start + period * 2 - 2, n):
                    res[i] = 2 * ema1[i] - ema2[i]
    elif ma_type == 3:  # WMA
        weights = np.arange(1, period + 1, dtype=np.float64)
        w_sum = np.sum(weights)
        for i in range(start + period - 1, n):
            res[i] = np.sum(prices[i - period + 1:i + 1] * weights) / w_sum
    return res


@njit(nogil=True, cache=True)
def calc_adx(highs, lows, closes, period=14):
    n = len(closes)
    adx = np.empty(n)
    adx[:] = np.nan

    start = 0
    while start < n and np.isnan(closes[start]):
        start += 1
    if n - start < period * 2:
        return adx

    tr       = np.zeros(n)
    plus_dm  = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(start + 1, n):
        high_diff = highs[i] - highs[i - 1]
        low_diff  = lows[i - 1] - lows[i]
        tr[i]        = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        plus_dm[i]   = high_diff if high_diff > low_diff and high_diff > 0 else 0.0
        minus_dm[i]  = low_diff  if low_diff  > high_diff and low_diff  > 0 else 0.0

    atr_s   = np.sum(tr[start+1:start+period+1])
    plus_s  = np.sum(plus_dm[start+1:start+period+1])
    minus_s = np.sum(minus_dm[start+1:start+period+1])

    for i in range(start + period, n):
        if i > start + period:
            atr_s   = atr_s   - atr_s   / period + tr[i]
            plus_s  = plus_s  - plus_s  / period + plus_dm[i]
            minus_s = minus_s - minus_s / period + minus_dm[i]

        if atr_s > 0:
            plus_di  = 100 * plus_s  / atr_s
            minus_di = 100 * minus_s / atr_s
            denom    = plus_di + minus_di
            dx       = 100 * abs(plus_di - minus_di) / denom if denom > 0 else 0.0
        else:
            dx = 0.0

        if i == start + period: adx[i] = dx
        else:                   adx[i] = (adx[i - 1] * (period - 1) + dx) / period
    return adx


@njit(nogil=True, cache=True)
def calc_atr_wilder(highs, lows, closes, period=14):
    """Wilder-smoothed ATR, used consistently everywhere in this file."""
    n = len(closes)
    atr = np.empty(n)
    atr[:] = np.nan
    start = 0
    while start < n and np.isnan(closes[start]):
        start += 1
    if n - start < period + 1:
        return atr

    tr = np.zeros(n)
    for i in range(start + 1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))

    atr_s = np.sum(tr[start+1:start+period+1]) / period
    atr[start + period] = atr_s
    for i in range(start + period + 1, n):
        atr_s = (atr_s * (period - 1) + tr[i]) / period
        atr[i] = atr_s
    return atr


# ==========================================
# 3b. BOUNDED, THREAD-SAFE MEMOIZATION FOR calc_ma
# ==========================================
# Pure speed optimization. Cache key is (stock_idx, period, ma_type) only --
# safe because `closes` is built once per run_optimization() call and never
# mutated. Always call clear_ma_cache() before reusing evaluate_params()
# against a NEW dataset in the same process.

_MA_CACHE = OrderedDict()
_MA_CACHE_LOCK = threading.Lock()


def get_ma_cached(closes, stock_idx, period, ma_type):
    key = (stock_idx, int(period), int(ma_type))
    with _MA_CACHE_LOCK:
        cached = _MA_CACHE.get(key)
        if cached is not None:
            _MA_CACHE.move_to_end(key)
            return cached
    # Computed outside the lock so calc_ma's nogil parallelism isn't
    # serialized; a rare cache-miss race just recomputes the same array
    # twice, which is wasted work but never incorrect.
    result = calc_ma(closes[:, stock_idx], period, ma_type)
    with _MA_CACHE_LOCK:
        _MA_CACHE[key] = result
        while len(_MA_CACHE) > MA_CACHE_MAX_ENTRIES:
            _MA_CACHE.popitem(last=False)
    return result


def clear_ma_cache():
    with _MA_CACHE_LOCK:
        _MA_CACHE.clear()


# ==========================================
# 4. PORTFOLIO SIMULATOR
# ==========================================

@njit(nogil=True)
def simulate_portfolio(
        opens, closes, atr, adx, sip_trigger, index_closes, is_div_stock,
        short_ma, long_ma, super_ma, eligible_mask,
        wl_rank_method, entry_filter, adx_threshold,
        n_exit_method, n_trail_pct, n_atr_mult,
        div_exit_method, div_exit_val,
        start_day, end_day,
        starting_wealth, bench_starting_wealth,
        buy_cost_pct, sell_cost_pct, dp_flat_fee, annual_cash_yield,
        nifty_ma, use_nifty_filter,
        use_wl_age_weight, wl_age_weight):

    n_days, n_stocks = closes.shape
    if end_day < 0 or end_day >= n_days: end_day = n_days - 2
    if start_day < 1: start_day = 1

    daily_yield_mult = (1.0 + annual_cash_yield) ** (1.0 / TRADING_DAYS_PER_YEAR)

    # ── UNIFIED CASH POOL ──────────────────────────────────────────────────
    cash_pool              = starting_wealth
    total_invested_capital = 0.0

    in_pos           = np.zeros(n_stocks, dtype=np.bool_)
    entry_prices     = np.zeros(n_stocks)
    entry_days       = np.zeros(n_stocks, dtype=np.int32)
    shares_held      = np.zeros(n_stocks)
    high_since_entry = np.zeros(n_stocks)
    days_below_long  = np.zeros(n_stocks)
    wl_days_below_long = np.zeros(n_stocks)   # NEW: watchlist-side mirror of days_below_long,
                                               # needed for div_exit_method==2 parity (see
                                               # WATCHLIST INVALIDATION below)

    wl_active       = np.zeros(n_stocks, dtype=np.bool_)
    wl_entry_prices = np.zeros(n_stocks)
    wl_is_pyramid   = np.zeros(n_stocks, dtype=np.bool_)
    wl_entry_day    = np.zeros(n_stocks, dtype=np.int32)   # NEW: feeds the optional
                                                            # watchlist-age ranking boost

    pending_exit = np.zeros(n_stocks, dtype=np.bool_)

    total_wins     = 0.0
    total_losses   = 0.0
    winning_trades = 0
    losing_trades  = 0
    win_pct_sum    = 0.0
    loss_pct_sum   = 0.0
    total_bars_in_trades = 0

    n_floor_clamps  = 0
    cash_frac_sum   = 0.0
    cash_frac_days  = 0
    days_high_cash  = 0

    daily_port_val  = np.zeros(n_days)
    daily_bench_val = np.zeros(n_days)

    # Per-day record of external cash injected (the monthly SIP) -- used to
    # strip cash-flow noise out of the daily return series before it feeds
    # Sharpe/Sortino/skew/kurtosis below.
    daily_cash_injection = np.zeros(n_days)

    # Cash-flow / monthly-return arrays sized off n_days (a provable upper
    # bound: at most one SIP event and one month-boundary per trading day)
    # instead of a fixed cap, so a long-running backtest can't silently
    # overflow and corrupt IRR/IR. `+8` is slack for the start/terminal entries.
    MAX_CF     = n_days + 8
    MAX_MONTHS = n_days + 8
    cf_days    = np.zeros(MAX_CF, dtype=np.int32)
    cf_amounts = np.zeros(MAX_CF)
    cf_cnt = 0
    if starting_wealth > 0:
        cf_days[0]    = start_day
        cf_amounts[0] = -starting_wealth
        cf_cnt = 1

    port_monthly_returns  = np.zeros(MAX_MONTHS)
    bench_monthly_returns = np.zeros(MAX_MONTHS)
    month_cnt = 0

    last_port_month_end  = 0.0
    last_bench_month_end = 0.0
    have_prior_month_end = False
    # bench_starting_wealth mirrors starting_wealth for the benchmark side,
    # so the OOS "beat benchmark" gate compares two numbers measured on the
    # same footing (both carrying forward from the IS period).
    bench_shares = bench_starting_wealth / index_closes[start_day] if index_closes[start_day] > 0 else 0.0

    for d in range(start_day, end_day):

        # ── IDLE CASH YIELD ──────────────────────────────────────────────
        cash_pool *= daily_yield_mult

        # ── PHASE-A: SETTLE PENDING EXITS AT TODAY'S OPEN ─────────────────
        for s in range(n_stocks):
            if pending_exit[s] and in_pos[s]:
                fill_price = opens[d, s]
                if not (fill_price > 0):   # NaN-safe: NaN>0 is False too
                    fill_price = closes[d - 1, s]
                exit_val     = shares_held[s] * fill_price * (1.0 - sell_cost_pct)
                exit_val     = max(0.0, exit_val - dp_flat_fee)
                invested_val = shares_held[s] * entry_prices[s]
                profit       = exit_val - invested_val
                pct_change   = profit / invested_val if invested_val > 0 else 0.0

                if profit > 0:
                    total_wins     += profit
                    win_pct_sum    += pct_change
                    winning_trades += 1
                else:
                    total_losses   += abs(profit)
                    loss_pct_sum   += abs(pct_change)
                    losing_trades  += 1

                total_bars_in_trades += (d - 1 - entry_days[s])
                cash_pool        += exit_val
                in_pos[s]         = False
                shares_held[s]    = 0.0
                pending_exit[s]   = False

        # ── MONTHLY SIP INJECTION ──────────────────────────────────────────
        if sip_trigger[d]:
            p_current = daily_port_val[d - 1] if d > start_day else starting_wealth
            b_current = bench_shares * index_closes[d - 1] if bench_shares > 0 else 0.0

            if have_prior_month_end:
                port_monthly_returns[month_cnt]  = (p_current - last_port_month_end) / last_port_month_end
                bench_monthly_returns[month_cnt] = (b_current - last_bench_month_end) / last_bench_month_end if last_bench_month_end > 0 else 0.0
                month_cnt = min(month_cnt + 1, MAX_MONTHS - 1)

            last_port_month_end  = p_current + MONTHLY_SIP
            last_bench_month_end = b_current + MONTHLY_SIP
            have_prior_month_end = True

            cash_pool               += MONTHLY_SIP
            total_invested_capital  += MONTHLY_SIP
            daily_cash_injection[d] += MONTHLY_SIP
            if index_closes[d] > 0:
                bench_shares += MONTHLY_SIP / index_closes[d]

            if cf_cnt >= MAX_CF:
                raise ValueError("cash-flow array overflow -- should be mathematically impossible")
            cf_days[cf_cnt]    = d
            cf_amounts[cf_cnt] = -MONTHLY_SIP
            cf_cnt += 1

        curr_closes = closes[d]
        nifty_bullish = index_closes[d] > nifty_ma[d]   # NEW: Nifty-regime panic-exit signal

        # ── WATCHLIST INVALIDATION (FIXED for shadow-position parity with
        # PHASE-B below, ported from the crypto engine's decision #17: a
        # watchlist candidate should be dropped by the SAME rule that would
        # actually sell it once bought -- no more, no less. Previously the
        # short/long MA crossunder applied to EVERY watchlist candidate
        # unconditionally, including dividend candidates -- even though a
        # real dividend POSITION is never exited by the growth crossunder,
        # only by div_exit_method (see PHASE-B). That let a perfectly fine
        # dividend candidate get dropped off the watchlist for a reason that
        # would never have sold it as a real position. Also adds the
        # previously-missing div_exit_method==2 (time-decay) watchlist
        # check, and the new optional Nifty panic-exit override for growth
        # (non-dividend) candidates only -- see module docstring.) ──
        newly_invalidated = np.zeros(n_stocks, dtype=np.bool_)
        for s in range(n_stocks):
            if wl_active[s]:
                if not wl_is_pyramid[s] and curr_closes[s] > 0:
                    high_since_entry[s] = max(high_since_entry[s], curr_closes[s])

                wl_invalid = False
                if is_div_stock[s]:
                    if div_exit_method == 0:
                        if curr_closes[s] < high_since_entry[s] * (1.0 - div_exit_val / 100.0): wl_invalid = True
                    elif div_exit_method == 1:
                        if curr_closes[s] < super_ma[d, s] * (1.0 - div_exit_val / 100.0): wl_invalid = True
                    elif div_exit_method == 2:
                        if curr_closes[s] < long_ma[d, s]: wl_days_below_long[s] += 1
                        else: wl_days_below_long[s] = 0
                        if wl_days_below_long[s] > div_exit_val: wl_invalid = True
                else:
                    if n_exit_method == 0:
                        if short_ma[d-1, s] >= long_ma[d-1, s] and short_ma[d, s] < long_ma[d, s]: wl_invalid = True
                    elif n_exit_method == 1:
                        if curr_closes[s] < high_since_entry[s] * (1.0 - n_trail_pct / 100.0): wl_invalid = True
                    elif n_exit_method == 2:
                        if curr_closes[s] < high_since_entry[s] - (n_atr_mult * atr[d, s]): wl_invalid = True
                    if use_nifty_filter and not nifty_bullish:
                        wl_invalid = True

                if wl_invalid:
                    wl_active[s]          = False
                    wl_is_pyramid[s]      = False
                    wl_days_below_long[s] = 0.0
                    newly_invalidated[s]  = True

        # ── PHASE-B: FLAG EXITS FOR NEXT-BAR SETTLEMENT ───────────────────
        for s in range(n_stocks):
            if in_pos[s] and not pending_exit[s]:
                if curr_closes[s] > 0:
                    high_since_entry[s] = max(high_since_entry[s], curr_closes[s])

                should_exit = False
                if is_div_stock[s]:
                    if div_exit_method == 0:
                        if curr_closes[s] < high_since_entry[s] * (1.0 - div_exit_val / 100.0): should_exit = True
                    elif div_exit_method == 1:
                        if curr_closes[s] < super_ma[d, s] * (1.0 - div_exit_val / 100.0): should_exit = True
                    elif div_exit_method == 2:
                        if curr_closes[s] < long_ma[d, s]: days_below_long[s] += 1
                        else: days_below_long[s] = 0
                        if days_below_long[s] > div_exit_val: should_exit = True
                else:
                    if n_exit_method == 0:
                        if short_ma[d-1, s] >= long_ma[d-1, s] and short_ma[d, s] < long_ma[d, s]: should_exit = True
                    elif n_exit_method == 1:
                        if curr_closes[s] < high_since_entry[s] * (1.0 - n_trail_pct / 100.0): should_exit = True
                    elif n_exit_method == 2:
                        if curr_closes[s] < high_since_entry[s] - (n_atr_mult * atr[d, s]): should_exit = True
                    # NEW: optional market-wide panic exit (Nifty regime turns
                    # bearish) -- growth stocks only, see module docstring.
                    if use_nifty_filter and not nifty_bullish:
                        should_exit = True

                if should_exit:
                    pending_exit[s] = True

        # ── WATCHLIST ADDITIONS ─────────────────────────────────────────
        for s in range(n_stocks):
            if not wl_active[s] and not newly_invalidated[s] and eligible_mask[d, s]:
                if short_ma[d-1, s] <= long_ma[d-1, s] and short_ma[d, s] > long_ma[d, s]:
                    valid_entry = True
                    if entry_filter == 1 and curr_closes[s] <= super_ma[d, s]: valid_entry = False
                    elif entry_filter == 2 and curr_closes[s] >= super_ma[d, s]: valid_entry = False
                    if valid_entry and adx_threshold > 0.0 and adx[d, s] < adx_threshold: valid_entry = False
                    # NEW: optional entry gate -- growth stocks only skip new
                    # entries while Nifty itself is bearish. Dividend stocks
                    # are deliberately exempt from this filter on both the
                    # entry and exit side (see module docstring) -- they were
                    # chosen specifically for a steadier, more defensive
                    # profile, and force-applying a broad growth-market panic
                    # rule to them would work against the reason they're a
                    # separate sleeve of the portfolio in the first place.
                    if valid_entry and use_nifty_filter and not is_div_stock[s] and not nifty_bullish:
                        valid_entry = False
                    if valid_entry:
                        wl_active[s]        = True
                        wl_entry_prices[s]  = curr_closes[s]   # FIXED (ported from the crypto
                                                                # engine): was opens[d,s] -- the
                                                                # crossover is confirmed using
                                                                # day d's CLOSE, so the close is
                                                                # the actual price at signal-
                                                                # detection time; the open predates
                                                                # the move that produced the signal.
                        wl_is_pyramid[s]    = in_pos[s] and not pending_exit[s]
                        wl_entry_day[s]     = d
                        if not wl_is_pyramid[s]:
                            high_since_entry[s] = opens[d, s]

        # ── CAPITAL DEPLOYMENT (ported from the crypto engine's decision #16:
        # equal-weight target instead of a fixed MONTHLY_SIP chunk per buy --
        # adapts ticket size to how many genuine opportunities exist right
        # now, instead of either starving a lone candidate to Rs.8,000
        # forever or running out of cash mid-way through a pile of
        # simultaneous signals. MIN_TICKET_SIZE is kept EQUAL to the fixed
        # Rs.8,000 SIP figure by explicit instruction -- this is a floor
        # under a now-variable ticket size, not a loosening of it.) ──
        n_eligible = 0
        for s in range(n_stocks):
            if eligible_mask[d, s]:
                n_eligible += 1
        if n_eligible < 1:
            n_eligible = 1

        while cash_pool >= MIN_TICKET_SIZE:
            best_rank  = -999999.0
            best_s     = -1
            for s in range(n_stocks):
                if wl_active[s]:
                    if wl_is_pyramid[s] and not in_pos[s]:
                        wl_active[s]     = False
                        wl_is_pyramid[s] = False
                        continue

                    rank = -999.0
                    if wl_rank_method == 0 and wl_entry_prices[s] > 0:
                        rank = (wl_entry_prices[s] - curr_closes[s]) / wl_entry_prices[s]
                    elif wl_rank_method == 1 and curr_closes[s] > 0:
                        rank = -abs(curr_closes[s] - long_ma[d, s]) / curr_closes[s]
                    elif wl_rank_method == 2 and short_ma[d, s] > 0:
                        rank = (curr_closes[s] - short_ma[d, s]) / short_ma[d, s]
                    # NEW: optional watchlist-age boost -- additive, so it
                    # nudges the existing rank toward older candidates
                    # instead of replacing the strategy's own ranking logic.
                    # age_frac saturates at 1.0 once a candidate has waited
                    # WL_AGE_NORM_DAYS trading days, so an extremely stale
                    # candidate doesn't get an unbounded advantage.
                    if use_wl_age_weight and rank > -999.0:
                        age_frac = (d - wl_entry_day[s]) / WL_AGE_NORM_DAYS
                        if age_frac > 1.0: age_frac = 1.0
                        elif age_frac < 0.0: age_frac = 0.0
                        rank = rank + wl_age_weight * age_frac
                    if rank > best_rank:
                        best_rank = rank
                        best_s    = s

            if best_s == -1: break

            buy_price = opens[d + 1, best_s] if d + 1 < n_days else curr_closes[best_s]
            if not (buy_price > 0):
                wl_active[best_s] = False
                continue

            # Equal-weight target size: (idle cash + cost-basis of everything
            # already open) / n_eligible. Cost basis, NOT mark-to-market --
            # an open position's unrealized paper gain must not inflate the
            # size of the NEXT, unrelated trade. If the wallet can't cover a
            # full target, it spends whatever's left instead of skipping the
            # trade outright; MIN_TICKET_SIZE stops the loop chasing dust
            # once cash_pool is nearly drained.
            invested_cost = 0.0
            for s2 in range(n_stocks):
                if in_pos[s2]:
                    invested_cost += shares_held[s2] * entry_prices[s2]
            target_size = (cash_pool + invested_cost) / n_eligible
            buy_amount  = target_size if target_size <= cash_pool else cash_pool
            if buy_amount < MIN_TICKET_SIZE:
                break

            cash_pool  -= buy_amount
            new_shares  = buy_amount / (buy_price * (1.0 + buy_cost_pct))

            if wl_is_pyramid[best_s] and in_pos[best_s]:
                old_cost = shares_held[best_s] * entry_prices[best_s]
                new_cost = new_shares * buy_price
                shares_held[best_s]  += new_shares
                entry_prices[best_s]  = (old_cost + new_cost) / shares_held[best_s]
            else:
                in_pos[best_s]           = True
                entry_prices[best_s]     = buy_price
                entry_days[best_s]       = d
                shares_held[best_s]      = new_shares
                high_since_entry[best_s] = buy_price

            wl_active[best_s]     = False
            wl_is_pyramid[best_s] = False

        # ── DAILY PORTFOLIO VALUATION ──────────────────────────────────────
        curr_val = cash_pool
        for s in range(n_stocks):
            if in_pos[s]:
                curr_val += shares_held[s] * curr_closes[s]

        if d > start_day:
            floor_val = daily_port_val[d - 1] * 0.001
            if curr_val < floor_val:
                # Should never fire on real NSE data (circuit limits alone
                # rule out a >99.9% one-day move). If it fires, treat the
                # run as suspect -- it means a NaN/zero-price leak upstream.
                n_floor_clamps += 1
            daily_port_val[d] = max(curr_val, floor_val)
        else:
            daily_port_val[d] = curr_val
        daily_bench_val[d] = bench_shares * index_closes[d]

        # ── CASH-UTILIZATION DIAGNOSTIC (reporting only) ──────────────────
        if daily_port_val[d] > 0:
            cash_frac = cash_pool / daily_port_val[d]
            cash_frac_sum  += cash_frac
            cash_frac_days += 1
            if cash_frac > HIGH_CASH_FRACTION_THRESHOLD:
                days_high_cash += 1

    # ── TERMINAL CASH FLOW ──────────────────────────────────────────────
    final_wealth       = daily_port_val[end_day - 1]
    final_bench_wealth = daily_bench_val[end_day - 1]
    if cf_cnt >= MAX_CF:
        raise ValueError("cash-flow array overflow at terminal entry -- should be impossible")
    cf_days[cf_cnt]    = end_day - 1
    cf_amounts[cf_cnt] = final_wealth
    cf_cnt += 1

    # ── METRICS ────────────────────────────────────────────────────────────
    trade_count        = winning_trades + losing_trades

    avg_bars   = total_bars_in_trades / trade_count if trade_count > 0 else 0.0
    avg_runup  = win_pct_sum  / winning_trades if winning_trades > 0 else 0.0
    avg_loss_r = loss_pct_sum / losing_trades  if losing_trades  > 0 else 0.0

    # max_dd deliberately walks the RAW daily_port_val (cash included, not
    # SIP-adjusted) -- "how far did my total account balance fall from its
    # peak" is a different, both-valid question from "how well did the
    # money that was actually invested perform" (the adjusted series below).
    max_dd      = 0.0
    peak        = daily_port_val[start_day]
    curr_dd_dur = 0
    max_dd_dur  = 0
    returns_sum      = 0.0
    returns_sq_sum   = 0.0
    returns_cube_sum = 0.0
    returns_quad_sum = 0.0
    downside_sq      = 0.0
    valid_days       = 0
    down_days        = 0

    for d in range(start_day + 1, end_day):
        v = daily_port_val[d]
        if v > peak:
            peak        = v
            curr_dd_dur = 0
        else:
            if peak > 0:
                dd = (v - peak) / peak
                if dd < max_dd: max_dd = dd
            curr_dd_dur += 1
            if curr_dd_dur > max_dd_dur: max_dd_dur = curr_dd_dur

        prev = daily_port_val[d - 1]
        if prev > 0:
            # Subtract today's SIP injection before computing the return --
            # otherwise every SIP date reads as a fabricated one-day gain
            # (scoring deposit speed, not investment performance).
            ret = (v - daily_cash_injection[d] - prev) / prev
            returns_sum      += ret
            returns_sq_sum   += ret * ret
            returns_cube_sum += ret * ret * ret
            returns_quad_sum += ret * ret * ret * ret
            valid_days       += 1
            if ret < 0:
                downside_sq += ret * ret
                down_days   += 1

    sharpe  = 0.0
    sortino = 0.0
    skew    = 0.0
    kurt    = 0.0
    if valid_days > 1:
        mean_ret = returns_sum / valid_days
        var_ret  = (returns_sq_sum / valid_days) - (mean_ret ** 2)
        if var_ret > 0:
            std_ret = math.sqrt(var_ret)
            if std_ret > 0: sharpe = (mean_ret / std_ret) * math.sqrt(TRADING_DAYS_PER_YEAR)
        # Sortino divides by ALL valid days, not just down days (matches
        # the textbook definition).
        std_down = math.sqrt(downside_sq / valid_days)
        if std_down > 0: sortino = (mean_ret / std_down) * math.sqrt(TRADING_DAYS_PER_YEAR)

        # Real sample skew / (non-excess) kurtosis of this same
        # cash-flow-adjusted return series, via raw-to-central-moment
        # conversion. Left at 0.0 (handled by the caller) if var_ret is 0.
        if var_ret > 0:
            e_x  = mean_ret
            e_x2 = returns_sq_sum   / valid_days
            e_x3 = returns_cube_sum / valid_days
            e_x4 = returns_quad_sum / valid_days
            m2 = var_ret
            m3 = e_x3 - 3.0 * e_x * e_x2 + 2.0 * e_x ** 3
            m4 = e_x4 - 4.0 * e_x * e_x3 + 6.0 * e_x ** 2 * e_x2 - 3.0 * e_x ** 4
            skew = m3 / (m2 ** 1.5)
            kurt = m4 / (m2 ** 2)

    info_ratio = 0.0
    if month_cnt > MIN_MONTHS_GATE:
        mean_excess = 0.0
        sq_excess   = 0.0
        for i in range(month_cnt):
            ex = port_monthly_returns[i] - bench_monthly_returns[i]
            mean_excess += ex
            sq_excess   += ex * ex
        mean_excess /= month_cnt
        var_excess   = sq_excess / month_cnt - mean_excess ** 2
        if var_excess > 0:
            info_ratio = (mean_excess / math.sqrt(var_excess)) * math.sqrt(12)

    return (final_wealth, final_bench_wealth, total_invested_capital,
            total_wins, total_losses, trade_count,
            winning_trades, losing_trades,
            avg_bars, avg_runup, avg_loss_r,
            max_dd, max_dd_dur,
            sharpe, sortino, info_ratio,
            month_cnt,
            cf_days, cf_amounts, cf_cnt,
            n_floor_clamps,
            cash_frac_sum, cash_frac_days, days_high_cash,
            daily_port_val, daily_bench_val,
            skew, kurt, valid_days)


# ==========================================
# 4b. MONEY-WEIGHTED RETURN SOLVER
# ==========================================

def money_weighted_annual_return(cf_days, cf_amounts, day_basis=TRADING_DAYS_PER_YEAR,
                                  r_lo=-0.999, r_hi=10.0, tol=1e-9, max_iter=200):
    """
    Solve for the annualized money-weighted rate of return (XIRR-style)
    given a dated cash-flow stream, via bisection on NPV(r)=0.

    Cash flows are always "a run of outflows then one terminal inflow" --
    exactly one sign change -- so by Descartes' rule of signs there is at
    most one real root for r > -1, guaranteeing a sign-changing bracket
    finds a UNIQUE root rather than converging to the wrong one of several.

    Returns (r, converged). converged=False if no bracket exists (e.g. a
    strategy that lost essentially all capital).
    """
    if len(cf_amounts) < 2:
        return 0.0, False
    t0 = cf_days[0]
    times = [(d - t0) / day_basis for d in cf_days]

    def npv(r):
        total = 0.0
        for t, amt in zip(times, cf_amounts):
            try:
                total += amt / ((1.0 + r) ** t)
            except (OverflowError, ZeroDivisionError):
                return float('nan')
        return total

    f_lo, f_hi = npv(r_lo), npv(r_hi)
    if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
        return 0.0, False
    if (f_lo > 0) == (f_hi > 0):
        return 0.0, False

    for _ in range(max_iter):
        r_mid = 0.5 * (r_lo + r_hi)
        f_mid = npv(r_mid)
        if not math.isfinite(f_mid):
            return 0.0, False
        if abs(f_mid) < tol or (r_hi - r_lo) < 1e-10:
            return r_mid, True
        if (f_mid > 0) == (f_lo > 0):
            r_lo, f_lo = r_mid, f_mid
        else:
            r_hi, f_hi = r_mid, f_mid
    return 0.5 * (r_lo + r_hi), True


# ==========================================
# 4c. OVERFITTING DIAGNOSTIC -- Deflated / Probabilistic Sharpe Ratio
# (Bailey & Lopez de Prado, 2014)
# ==========================================

_EULER_GAMMA = 0.5772156649015329
_NORM = NormalDist()


def expected_max_sharpe(sr_std, n_trials):
    """Expected max Sharpe you'd see across n_trials independent
    zero-skill strategies, given the observed cross-sectional Sharpe std."""
    if n_trials <= 1 or sr_std <= 0:
        return 0.0
    z1 = _NORM.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _NORM.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return sr_std * ((1 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)


def probabilistic_sharpe_ratio(sr_hat, sr_benchmark, T, skew, kurt):
    """P(true Sharpe > sr_benchmark), given observed per-period sr_hat over
    T observations, adjusted for skew/kurtosis. sr_hat/sr_benchmark must be
    in the same (non-annualized) units."""
    denom = math.sqrt(max(1e-12, 1 - skew * sr_hat + ((kurt - 1) / 4.0) * sr_hat ** 2))
    z = (sr_hat - sr_benchmark) * math.sqrt(max(T - 1, 1)) / denom
    return _NORM.cdf(z)


def deflated_sharpe_ratio(sr_hat_daily, all_trial_sharpes_daily, T, skew, kurt):
    """Returns (DSR, threshold_SR0_daily, n_trials_used)."""
    n_trials = len(all_trial_sharpes_daily)
    if n_trials < 2:
        return probabilistic_sharpe_ratio(sr_hat_daily, 0.0, T, skew, kurt), 0.0, n_trials
    mean_sr = sum(all_trial_sharpes_daily) / n_trials
    var_sr  = sum((s - mean_sr) ** 2 for s in all_trial_sharpes_daily) / n_trials
    sr_std  = math.sqrt(var_sr)
    sr0 = expected_max_sharpe(sr_std, n_trials)
    dsr = probabilistic_sharpe_ratio(sr_hat_daily, sr0, T, skew, kurt)
    return dsr, sr0, n_trials


# ==========================================
# 4d. POINT-IN-TIME UNIVERSE -- BEST-EFFORT FETCHER
# ==========================================
# Reconstructs point-in-time NIFTY 50 (ONLY -- not Next 50/Midcap 150, no
# free officially-maintained archive found for those) membership from NSE's
# own inclusion/exclusion history file, rolling backward from today's live
# list. Written against public docs of that file, NOT verified against a
# live download -- run it, read the coverage report, and spot-check known
# historical changes (e.g. HDFC merging into HDFC Bank in 2023) before
# trusting it for real capital.
#
# Usage:
#   >>> from main import fetch_and_build_point_in_time_universe
#   >>> fetch_and_build_point_in_time_universe("nifty50_pit.json")
# then set POINT_IN_TIME_UNIVERSE_FILE = "nifty50_pit.json" above.

NSE_INDEX_INCL_EXCL_URL = "https://archives.nseindia.com/content/indices/IndexInclExcl.xls"
NSE_NIFTY50_ALIASES = {"NIFTY 50", "NIFTY50", "CNX NIFTY", "S&P CNX NIFTY", "S&P CNX NIFTY 50"}


def fetch_and_build_point_in_time_universe(output_file="nifty50_point_in_time.json"):
    """Best-effort NIFTY 50 (ONLY) point-in-time membership reconstruction.
    See section comment above for exact scope."""
    print("Point-in-time universe fetcher (NIFTY 50 only)...")

    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        resp = requests.get(NSE_INDEX_INCL_EXCL_URL, headers=headers, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"Could not download {NSE_INDEX_INCL_EXCL_URL}: {e}")
        print("NSE occasionally moves this file -- search 'niftyindices.com IndexInclExcl' for the current location.")
        return None

    raw = io.BytesIO(resp.content)
    df = None
    for engine in ('xlrd', 'openpyxl', None):
        try:
            raw.seek(0)
            df = pd.read_excel(raw, engine=engine) if engine else pd.read_excel(raw)
            break
        except Exception:
            continue
    if df is None:
        print("Downloaded but could not parse as .xls/.xlsx -- NSE may have changed format.")
        return None

    cols = {c: str(c).strip().lower() for c in df.columns}
    def find_col(*keywords):
        for c, low in cols.items():
            if all(k in low for k in keywords):
                return c
        return None

    col_index  = find_col("index")
    col_symbol = find_col("symbol") or find_col("ticker")
    col_date   = find_col("date")
    col_action = find_col("action") or find_col("remark") or find_col("event") or find_col("effect")

    print(f"Columns: {list(df.columns)}")
    print(f"Detected -> index: {col_index!r}  symbol: {col_symbol!r}  date: {col_date!r}  action: {col_action!r}")
    if not all([col_symbol, col_date, col_action]):
        print("Could not identify symbol/date/action columns -- inspect the file manually.")
        return None

    events = []   # (date_str, symbol, 'IN'|'OUT')
    unmatched_index_names = set()
    for _, row in df.iterrows():
        try:
            if col_index is not None:
                idx_name = str(row[col_index]).strip().upper()
                if idx_name not in NSE_NIFTY50_ALIASES:
                    if idx_name and idx_name != 'NAN':
                        unmatched_index_names.add(idx_name)
                    continue
            action_raw = str(row[col_action]).strip().upper()
            if 'IN' in action_raw and 'OUT' not in action_raw:
                action = 'IN'
            elif 'OUT' in action_raw or 'EXCL' in action_raw or 'REMOV' in action_raw or 'DEL' in action_raw:
                action = 'OUT'
            elif 'INCL' in action_raw or 'ADD' in action_raw:
                action = 'IN'
            else:
                continue
            dt = pd.to_datetime(row[col_date], errors='coerce')
            if pd.isna(dt):
                continue
            symbol = str(row[col_symbol]).strip().upper()
            if not symbol or symbol == 'NAN':
                continue
            events.append((dt.strftime('%Y-%m-%d'), f"{symbol}.NS", action))
        except Exception:
            continue

    if not events:
        print("Parsed the file but extracted zero usable NIFTY 50 events.")
        if unmatched_index_names:
            print(f"Unmatched index names seen: {sorted(unmatched_index_names)[:20]}")
            print("Add the correct spelling to NSE_NIFTY50_ALIASES and rerun.")
        return None

    events.sort(key=lambda e: e[0])
    earliest, latest = events[0][0], events[-1][0]
    print(f"Parsed {len(events)} events, spanning {earliest} to {latest}.")

    months_stale = (pd.Timestamp.now() - pd.Timestamp(latest)).days / 30.44
    if months_stale > 8:
        print(f"WARNING: latest event is ~{months_stale:.0f} months old -- archive may be stale "
              f"(Nifty 50 rebalances ~every 6 months). Cross-check against today's live list.")
    if earliest > START_DATE:
        print(f"WARNING: earliest event ({earliest}) is after START_DATE ({START_DATE}) -- membership "
              f"before {earliest} is extrapolated backward, unverified.")

    try:
        req = requests.get("https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
                           headers=headers, timeout=10)
        req.raise_for_status()
        today_df = pd.read_csv(io.StringIO(req.text))
        today_set = {f"{s}.NS" for s in today_df['Symbol']}
    except Exception as e:
        print(f"Could not fetch today's live list to anchor the reconstruction ({e}). Aborting.")
        return None
    print(f"Anchored to today's live list: {len(today_set)} symbols.")

    # Roll backward from today, undoing one event at a time.
    working_set = set(today_set)
    pit = {}
    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    pit[today_str] = sorted(working_set)
    for date_str, symbol, action in reversed(events):
        pit[date_str] = sorted(working_set)
        if action == 'IN':
            working_set.discard(symbol)
        else:
            working_set.add(symbol)
    pit[START_DATE] = sorted(working_set)

    with open(output_file, 'w') as f:
        json.dump(pit, f, indent=2)
    print(f"Wrote {len(pit)} membership snapshots to {output_file}.")
    print("Next 50 / Midcap 150 still default to 'always eligible'. Spot-check before real use.")
    return output_file


# ==========================================
# 5. DATA PREPARATION
# ==========================================

def build_eligibility_mask(master_dates, stock_names, n_days, n_stocks):
    """Returns (n_days, n_stocks) bool array: True if a stock may be a NEW
    entry on that date. Defaults to all-True unless POINT_IN_TIME_UNIVERSE_FILE
    points at a real reconstructed-membership file (existing positions are
    never force-closed on removal)."""
    mask = np.ones((n_days, n_stocks), dtype=np.bool_)
    if not POINT_IN_TIME_UNIVERSE_FILE or not os.path.exists(POINT_IN_TIME_UNIVERSE_FILE):
        return mask
    try:
        with open(POINT_IN_TIME_UNIVERSE_FILE, 'r') as f:
            pit = json.load(f)
    except Exception as e:
        print(f"Could not load point-in-time universe file ({e}); using all-eligible default.")
        return mask

    sorted_dates = sorted(pit.keys())
    name_to_idx = {name: i for i, name in enumerate(stock_names)}
    cursor = 0
    current_set = set()
    for d_idx, dt in enumerate(master_dates):
        dt_str = dt.strftime('%Y-%m-%d')
        while cursor < len(sorted_dates) and sorted_dates[cursor] <= dt_str:
            current_set = set(pit[sorted_dates[cursor]])
            cursor += 1
        if not current_set:
            continue
        mask[d_idx, :] = False
        for tkr in current_set:
            if tkr in name_to_idx:
                mask[d_idx, name_to_idx[tkr]] = True
    return mask


def prepare_matrix_data(tickers):
    if not os.path.exists("data"): os.makedirs("data")
    raw_dfs      = {}
    master_dates = set()

    for ticker in tqdm(tickers, desc="Downloading Data"):
        file_path = f"data/{ticker}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        else:
            try:
                df = yf.download(ticker, start=START_DATE, progress=False, multi_level_index=False)
                if len(df) > 200: df.to_csv(file_path)
            except Exception: continue
        # A currently-listed stock with <500 trading days of history is
        # dropped entirely here -- a second, distinct exclusion on top of
        # the disclosed survivorship bias above (nothing to backfill it with).
        if df is not None and len(df) > 500:
            raw_dfs[ticker] = df
            master_dates.update(df.index)

    nifty_path = "data/NIFTY_BENCHMARK.csv"
    if os.path.exists(nifty_path):
        nifty_df = pd.read_csv(nifty_path, parse_dates=['Date'], index_col='Date')
    else:
        nifty_df = yf.download("^NSEI", start=START_DATE, progress=False, multi_level_index=False)
        nifty_df.to_csv(nifty_path)

    master_dates = sorted(list(master_dates))
    master_df    = pd.DataFrame(index=master_dates)
    master_df['Month'] = master_df.index.month
    master_df['Year']  = master_df.index.year

    nifty_reindexed = nifty_df.reindex(master_dates).ffill()
    index_closes    = nifty_reindexed['Close'].values.flatten().astype(np.float64)

    n_days   = len(master_dates)
    n_stocks = len(raw_dfs)

    opens        = np.zeros((n_days, n_stocks))
    highs        = np.zeros((n_days, n_stocks))
    lows         = np.zeros((n_days, n_stocks))
    closes       = np.zeros((n_days, n_stocks))
    atr_matrix   = np.zeros((n_days, n_stocks))
    adx_matrix   = np.zeros((n_days, n_stocks))
    is_div_stock = np.zeros(n_stocks, dtype=np.bool_)
    stock_names  = list(raw_dfs.keys())

    for i, ticker in enumerate(tqdm(stock_names, desc="Building Matrix")):
        df = raw_dfs[ticker].reindex(master_dates)
        # yfinance's default (auto_adjust=True) already returns
        # split/dividend-adjusted OHLC and omits 'Adj Close' -- this
        # handles both the new and old yfinance default.
        close_col = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
        # ffill only fills gaps within a stock's own trading history
        # (holidays etc.) -- genuine pre-listing rows stay NaN, which
        # calc_ma/calc_adx handle correctly via their NaN-scan.
        opens[:,  i] = df['Open'].ffill().values
        highs[:,  i] = df['High'].ffill().values
        lows[:,   i] = df['Low'].ffill().values
        closes[:, i] = close_col.ffill().values
        atr_matrix[:, i] = calc_atr_wilder(highs[:, i], lows[:, i], closes[:, i], period=14)
        adx_matrix[:, i] = calc_adx(highs[:, i], lows[:, i], closes[:, i], period=14)
        if ticker in DIVIDEND_KINGS_FALLBACK:
            is_div_stock[i] = True

    eligible_mask = build_eligibility_mask(master_dates, stock_names, n_days, n_stocks)
    years_arr = master_df['Year'].values.astype(np.int32)

    return (opens, closes, atr_matrix, adx_matrix,
            master_df['Month'].values, index_closes,
            is_div_stock, stock_names, eligible_mask, years_arr)


# ==========================================
# 5b. YEARLY BREAKDOWN (ported from the crypto engine -- feeds recency
# weighting, the consistency term, the regime term, AND the profit-
# concentration hard gate). Unlike the crypto engine, this reuses a SINGLE
# cash-flow ledger for both the strategy and Nifty instead of two separate
# ones -- stock has no SIP-withholding mechanic, so strategy and benchmark
# cash flows land on identical days/amounts by construction, and a second
# ledger would just be unused complexity. If a withhold-style mechanic is
# ever added here, this needs the same benchmark-ledger split the crypto
# engine has -- see the CHANGELOG note on this exact question.
# ==========================================

def compute_yearly_breakdown(daily_port_val, daily_bench_val, years_arr,
                              cf_days, cf_amounts, cf_cnt,
                              start_day, end_day):
    """For every calendar year touched by [start_day, end_day), compute the
    strategy's and Nifty's money-weighted IRR for that year alone (bookend
    the year's start/end value as synthetic cash flows, plus whatever real
    SIP cash flows landed inside that year), nominal $ profit (for the
    concentration gate), and coverage_frac (a partial first/last year
    counts less, both for recency weighting and the concentration gate's
    denominator)."""
    day_years = years_arr[start_day:end_day]
    if len(day_years) == 0:
        return []
    unique_years = sorted(set(int(y) for y in day_years))

    cf_days_arr = np.array(cf_days[:cf_cnt])
    cf_amts_arr = np.array(cf_amounts[:cf_cnt])

    out = []
    for yr in unique_years:
        yr_day_idxs = np.where(day_years == yr)[0] + start_day
        if len(yr_day_idxs) < 5:
            continue
        y_start, y_end = int(yr_day_idxs[0]), int(yr_day_idxs[-1])

        port_start_val  = daily_port_val[max(y_start - 1, start_day)]
        port_end_val    = daily_port_val[y_end]
        bench_start_val = daily_bench_val[max(y_start - 1, start_day)]
        bench_end_val   = daily_bench_val[y_end]

        mask = (cf_days_arr >= y_start) & (cf_days_arr <= y_end)
        year_cf_days    = list(cf_days_arr[mask])
        year_cf_amounts = list(cf_amts_arr[mask])
        port_cf_days     = [y_start] + year_cf_days + [y_end]
        port_cf_amounts  = [-port_start_val] + year_cf_amounts + [port_end_val]
        bench_cf_days    = [y_start] + year_cf_days + [y_end]
        bench_cf_amounts = [-bench_start_val] + year_cf_amounts + [bench_end_val]

        port_irr, port_ok   = money_weighted_annual_return(port_cf_days, port_cf_amounts)
        bench_irr, bench_ok = money_weighted_annual_return(bench_cf_days, bench_cf_amounts)

        nominal_contrib = sum(-a for a in year_cf_amounts if a < 0)
        nominal_profit  = (port_end_val - port_start_val) - nominal_contrib
        coverage_frac   = min(1.0, len(yr_day_idxs) / YEAR_FULL_COVERAGE_DAYS)

        out.append({
            'year': yr,
            'port_irr': port_irr if port_ok else 0.0,
            'bench_irr': bench_irr if bench_ok else 0.0,
            'nominal_profit': nominal_profit,
            'coverage_frac': coverage_frac,
            'days': len(yr_day_idxs),
        })
    return out


def _recency_weight(year, min_year, max_year):
    if max_year <= min_year:
        return RECENCY_WEIGHT_MAX
    frac = (year - min_year) / (max_year - min_year)
    return RECENCY_WEIGHT_MIN + (RECENCY_WEIGHT_MAX - RECENCY_WEIGHT_MIN) * frac


# ==========================================
# 6. SCORING FUNCTION
# ==========================================
# score = Calmar*0.30 + Sortino(cap4)*0.10 + IR*0.30 + EV_in_R*0.15
#         + WinRate_bonus*0.15, scaled by stat_confidence * drawdown_penalty.
# Calmar uses the money-weighted IRR, not lump-sum CAGR.

def compute_score_v4(metrics, yearly, is_oos=False):
    """
    score = [ Calmar*0.20 + Sortino(cap4)*0.10 + IR*0.20 + EV_in_R*0.10
              + WinRate_bonus*0.10 + Consistency*0.20 + Regime*0.10 ]
            * stat_confidence * drawdown_penalty

    CONSISTENCY and REGIME are ported from the crypto engine unchanged in
    formula (see that file's compute_score_crypto docstring for the full
    derivation) -- recency-weighted per-calendar-year IRR consistency, and
    a bull/bear split scored against Nifty's own annual return that year.

    HARD GATES: same as before, PLUS a new profit-concentration gate -- if
    any single calendar year accounts for more than CONCENTRATION_GATE
    (55%) of total nominal profit, the trial is rejected outright,
    independent of everything else. The recency weighting softens the
    SCORE; this gate refuses to deploy the extreme cases at all.
    """
    roi            = metrics['roi']
    bench_roi      = metrics['bench_roi']
    sortino        = metrics['sortino']
    ir             = metrics['ir']
    max_dd         = abs(metrics['max_dd'])
    trades         = metrics['trades']
    avg_runup      = metrics['avg_runup']
    avg_loss       = abs(metrics['avg_loss'])
    month_cnt      = metrics['month_cnt']
    winning_trades = metrics['winning_trades']
    losing_trades  = metrics['losing_trades']
    annual_return  = metrics['annual_return']

    target_trades = max(15, MIN_TRADES_GATE * (WFO_OOS_PCT / WFO_IS_PCT)) if is_oos else MIN_TRADES_GATE
    target_months = max(12, MIN_MONTHS_GATE * (WFO_OOS_PCT / WFO_IS_PCT)) if is_oos else MIN_MONTHS_GATE

    if trades    < target_trades: return -999.0
    if month_cnt < target_months: return -999.0
    if max_dd    > 0.50:          return -999.0
    if metrics['pf'] < 1.10:     return -999.0
    if roi       <= 0:            return -999.0
    if roi       < bench_roi * BENCH_OUTPERFORMANCE_MULT: return -999.0
    if avg_loss  <= 0:            return -999.0
    if not math.isfinite(annual_return): return -999.0

    total_trades_counted = winning_trades + losing_trades
    actual_wr = winning_trades / total_trades_counted if total_trades_counted > 0 else 0.0
    if actual_wr < MIN_WIN_RATE_GATE: return -999.0

    # NEW hard gate: profit concentration (ported from the crypto engine)
    if yearly:
        total_profit = sum(y['nominal_profit'] for y in yearly)
        if total_profit > 0:
            max_share = max(y['nominal_profit'] / total_profit for y in yearly)
            if max_share > CONCENTRATION_GATE:
                return -999.0

    if max_dd < 0.001: max_dd = 0.001
    calmar = annual_return / max_dd

    ev      = actual_wr * avg_runup - (1.0 - actual_wr) * avg_loss
    ev_in_r = ev / avg_loss

    wr_bonus = max(0.0, actual_wr - 0.50) * 4.0

    sortino_capped = min(sortino, 4.0)

    ir_scaled = max(0.0, ir) * math.sqrt(max(1.0, month_cnt) / 12.0)
    stat_conf = min(1.0, math.sqrt(trades / target_trades))

    if max_dd <= 0.20:   dd_penalty = 1.0
    elif max_dd <= 0.35: dd_penalty = 1.0 - (max_dd - 0.20) * 2.0
    else:                dd_penalty = 0.70 * math.exp(-2.0 * (max_dd - 0.35))

    # NEW: drop years whose coverage_frac is too low to trust an
    # IRR-annualized figure from (ported from the crypto engine's decision
    # #11). Only affects CONSISTENCY/REGIME below -- the concentration gate
    # above uses raw nominal dollars and isn't subject to this distortion.
    scoring_years = [y for y in yearly if y['coverage_frac'] >= MIN_YEAR_COVERAGE_FOR_SCORING]

    # ── CONSISTENCY term ────────────────────────────────────────────────
    consistency = 0.0
    if scoring_years:
        yrs = [y['year'] for y in scoring_years]
        min_year, max_year = min(yrs), max(yrs)
        w_list = [_recency_weight(y['year'], min_year, max_year) * y['coverage_frac'] for y in scoring_years]
        r_list = [y['port_irr'] for y in scoring_years]
        w_sum = sum(w_list)
        if w_sum > 0:
            mean_r = sum(w * r for w, r in zip(w_list, r_list)) / w_sum
            var_r  = sum(w * (r - mean_r) ** 2 for w, r in zip(w_list, r_list)) / w_sum
            std_r  = math.sqrt(max(0.0, var_r))
            worst_r = min(r_list)
            consistency = (mean_r - K_STD_PENALTY * std_r
                           - K_WORST_PENALTY * max(0.0, -worst_r)) / max_dd

    # ── REGIME term (bull/bear split by Nifty's OWN annual return) ───────
    regime = 0.0
    if scoring_years:
        bull_ratios, bull_weights = [], []
        bear_diffs,  bear_weights = [], []
        sy_min_year = min(yr['year'] for yr in scoring_years)
        sy_max_year = max(yr['year'] for yr in scoring_years)
        for y in scoring_years:
            w = _recency_weight(y['year'], sy_min_year, sy_max_year) * y['coverage_frac']
            if y['bench_irr'] > BULL_YEAR_NIFTY_THRESHOLD:
                ratio = y['port_irr'] / y['bench_irr'] if y['bench_irr'] != 0 else 0.0
                bull_ratios.append(max(-1.0, min(3.0, ratio)))
                bull_weights.append(w)
            elif y['bench_irr'] < BEAR_YEAR_NIFTY_THRESHOLD:
                diff = (y['port_irr'] - y['bench_irr']) / max_dd
                bear_diffs.append(max(-2.0, min(2.0, diff)))
                bear_weights.append(w)
        bull_capture = (sum(r * w for r, w in zip(bull_ratios, bull_weights)) / sum(bull_weights)) if bull_weights else 0.0
        bear_defense = (sum(d * w for d, w in zip(bear_diffs, bear_weights)) / sum(bear_weights)) if bear_weights else 0.0
        regime = W_BULL_CAPTURE * bull_capture + W_BEAR_DEFENSE * bear_defense

    score = (
        calmar         * W_CALMAR +
        sortino_capped * W_SORTINO +
        ir_scaled      * W_IR +
        ev_in_r        * W_EV +
        wr_bonus       * W_WR_BONUS +
        consistency    * W_CONSISTENCY +
        regime         * W_REGIME
    ) * stat_conf * dd_penalty

    return score


def diagnose_hard_gates(metrics, yearly, is_oos=False):
    """Returns a list of (gate_name, value_str, threshold_str, passed:bool)
    so a -999 rejection can be explained instead of just printed as -999."""
    trades         = metrics['trades']
    month_cnt      = metrics['month_cnt']
    max_dd         = abs(metrics['max_dd'])
    pf             = metrics['pf']
    roi            = metrics['roi']
    bench_roi      = metrics['bench_roi']
    avg_loss       = abs(metrics['avg_loss'])
    winning_trades = metrics['winning_trades']
    losing_trades  = metrics['losing_trades']
    total = winning_trades + losing_trades
    actual_wr = winning_trades / total if total > 0 else 0.0

    target_trades = max(15, MIN_TRADES_GATE * (WFO_OOS_PCT / WFO_IS_PCT)) if is_oos else MIN_TRADES_GATE
    target_months = max(12, MIN_MONTHS_GATE * (WFO_OOS_PCT / WFO_IS_PCT)) if is_oos else MIN_MONTHS_GATE

    pf_str = "inf" if math.isinf(pf) else f"{pf:.2f}"

    rows = [
        ("Trades",     f"{trades}",           f">= {target_trades:.0f}",      trades >= target_trades),
        ("Months",     f"{month_cnt}",         f">= {target_months:.0f}",      month_cnt >= target_months),
        ("Max DD",     f"{max_dd*100:.1f}%",   "<= 50.0%",                    max_dd <= 0.50),
        ("Profit Factor", pf_str,              ">= 1.10",                     pf >= 1.10),
        ("ROI",        f"{roi*100:.1f}%",      "> 0%",                        roi > 0),
        ("ROI vs Bench", f"{roi*100:.1f}%",
         f"> {bench_roi*BENCH_OUTPERFORMANCE_MULT*100:.1f}% ({BENCH_OUTPERFORMANCE_MULT:.2f}x bench)",
         roi >= bench_roi * BENCH_OUTPERFORMANCE_MULT),
        ("Avg Loss",   f"{avg_loss*100:.1f}%", "> 0%",                        avg_loss > 0),
        ("Win Rate",   f"{actual_wr*100:.1f}%", f">= {MIN_WIN_RATE_GATE*100:.0f}%", actual_wr >= MIN_WIN_RATE_GATE),
    ]

    if yearly:
        total_profit = sum(y['nominal_profit'] for y in yearly)
        if total_profit > 0:
            max_share = max(y['nominal_profit'] / total_profit for y in yearly)
            rows.append((f'Profit Concentration', f'{max_share*100:.1f}%',
                         f'<= {CONCENTRATION_GATE*100:.0f}%', max_share <= CONCENTRATION_GATE))
        else:
            rows.append(('Profit Concentration', 'n/a (total profit <= 0)',
                         f'<= {CONCENTRATION_GATE*100:.0f}%', True))

    return rows


def print_gate_diagnosis(metrics, yearly, is_oos=False, indent="  "):
    """Prints which hard gate(s) a -999 rejection actually tripped."""
    gates = diagnose_hard_gates(metrics, yearly, is_oos=is_oos)
    failed = [g for g in gates if not g[3]]
    print(f"{indent}Gate check (failures marked with *):")
    for name, val, thresh, passed in gates:
        mark = " " if passed else "*"
        print(f"{indent}{mark} {name:<14} {val:>10}   need {thresh}")
    if not failed:
        print(f"{indent}(All gates technically passed -- rejection likely came from a "
              f"non-finite annual_return, see irr_converged.)")


# ==========================================
# 7. EVALUATE PARAMS
# ==========================================

def evaluate_params(p, opens, closes, atr, adx, sip_trigger, index_closes, is_div_stock,
                    eligible_mask, years_arr, start_day=0, end_day=-1, is_oos=False,
                    starting_wealth=0.0, bench_starting_wealth=0.0):

    if p['s_ma'] >= p['l_ma']: return -999.0, {}

    n_days, n_stocks = closes.shape
    if end_day < 0: end_day = n_days - 1

    short_ma = np.zeros((n_days, n_stocks))
    long_ma  = np.zeros((n_days, n_stocks))
    super_ma = np.zeros((n_days, n_stocks))

    for s in range(n_stocks):
        short_ma[:, s] = get_ma_cached(closes, s, p['s_ma'],  p['t_s'])
        long_ma[:,  s] = get_ma_cached(closes, s, p['l_ma'],  p['t_l'])
        super_ma[:, s] = get_ma_cached(closes, s, p['sl_ma'], p['t_sl'])

    # NEW: optional Nifty-regime panic-exit/entry filter (ported from the
    # crypto engine's BTC filter, adapted so it only applies to growth
    # stocks -- see module docstring). A single 1D MA over index_closes,
    # cheap enough to not need its own cache the way per-stock MAs do.
    use_nifty_filter = bool(p.get('use_nifty_filter', False))
    nifty_ma = calc_ma(index_closes, int(p.get('nifty_ma_len', 50)), int(p.get('nifty_ma_type', 0)))

    # NEW: optional watchlist-age ranking boost -- see module constants.
    use_wl_age_weight = bool(p.get('use_wl_age_weight', False))
    wl_age_weight     = float(p.get('wl_age_weight', 0.0))

    result = simulate_portfolio(
        opens, closes, atr, adx, sip_trigger, index_closes, is_div_stock,
        short_ma, long_ma, super_ma, eligible_mask,
        p['wl_rank'], p['entry_f'], p.get('adx_thresh', 0.0),
        p['n_exit_m'], p['n_trail_p'], p['n_atr_m'],
        p['div_exit_m'], p['div_exit_v'],
        int(start_day), int(end_day),
        float(starting_wealth), float(bench_starting_wealth),
        BUY_COST_PCT, SELL_COST_PCT, DP_FLAT_FEE_RS, ANNUAL_CASH_YIELD,
        nifty_ma, use_nifty_filter,
        use_wl_age_weight, wl_age_weight
    )

    (f_wealth, f_bench, t_invested, wins, losses, trades,
     winning_trades, losing_trades, avg_bars, avg_runup, avg_loss,
     max_dd, max_dd_dur, sharpe, sortino, ir, month_cnt,
     cf_days, cf_amounts, cf_cnt,
     n_floor_clamps, cash_frac_sum, cash_frac_days, days_high_cash,
     daily_port_val, daily_bench_val,
     skew, kurt, valid_days) = result

    if t_invested <= 0: return -999.0, {}

    roi = (f_wealth - (starting_wealth + t_invested)) / (starting_wealth + t_invested) if (starting_wealth + t_invested) > 0 else 0.0
    # bench_roi mirrors roi's own denominator (starting wealth + this
    # window's contributions) so the "beat benchmark by 20%" gate compares
    # like with like, instead of measuring the benchmark from zero on an
    # OOS call where the strategy side carries forward IS-period capital.
    # SAFE as a shared denominator here specifically because stock has no
    # SIP-withholding mechanic -- Nifty and the strategy always receive the
    # identical MONTHLY_SIP on the identical days, so t_invested IS what
    # Nifty received too. See the section-5b comment for what would need to
    # change if that ever stops being true.
    bench_denom = bench_starting_wealth + t_invested
    bench_roi   = (f_bench - bench_denom) / bench_denom if bench_denom > 0 else 0.0
    pf          = wins / losses if losses > 0 else float('inf')

    cf_days_list    = list(cf_days[:cf_cnt])
    cf_amounts_list = list(cf_amounts[:cf_cnt])
    annual_return, irr_converged = money_weighted_annual_return(cf_days_list, cf_amounts_list)
    if not irr_converged:
        capital_base = starting_wealth + t_invested
        years = max(1.0, t_invested / (MONTHLY_SIP * 12))
        annual_return = (f_wealth / capital_base) ** (1.0 / years) - 1.0 if capital_base > 0 else -1.0

    avg_cash_frac      = (cash_frac_sum / cash_frac_days) if cash_frac_days > 0 else 0.0
    pct_days_high_cash = (days_high_cash / cash_frac_days * 100.0) if cash_frac_days > 0 else 0.0

    yearly = compute_yearly_breakdown(daily_port_val, daily_bench_val, years_arr,
                                       cf_days, cf_amounts, cf_cnt,
                                       int(start_day), int(end_day))

    metrics = {
        'roi': roi, 'bench_roi': bench_roi, 'alpha': roi - bench_roi,
        'pf': pf, 'wealth': f_wealth, 'bench_wealth': f_bench,
        'trades': trades, 'winning_trades': winning_trades, 'losing_trades': losing_trades,
        'sharpe': sharpe, 'sortino': sortino, 'ir': ir,
        'max_dd': max_dd, 'max_dd_dur': max_dd_dur,
        'avg_bars': avg_bars, 'avg_runup': avg_runup, 'avg_loss': avg_loss,
        't_invested': t_invested, 'month_cnt': month_cnt,
        'starting_wealth': starting_wealth, 'bench_starting_wealth': bench_starting_wealth,
        'annual_return': annual_return, 'irr_converged': irr_converged,
        'cf_days': cf_days_list, 'cf_amounts': cf_amounts_list,
        'n_floor_clamps': int(n_floor_clamps),
        'avg_cash_frac': avg_cash_frac,
        'pct_days_high_cash': pct_days_high_cash,
        'yearly': yearly,
        'skew': skew, 'kurtosis': kurt, 'valid_days': int(valid_days),
    }

    score = compute_score_v4(metrics, yearly, is_oos=is_oos)
    return score, metrics


# ==========================================
# 8. NEIGHBORHOOD STABILITY
# ==========================================

def passes_neighborhood_check(p, base_score, opens, closes, atr, adx,
                               sip_trigger, index_closes, is_div_stock, eligible_mask,
                               years_arr, start_day, end_day):
    """
    Perturbs every parameter that's actually ACTIVE for this champion's
    specific entry/exit configuration, not just s_ma/l_ma/n_trail_p/n_atr_m,
    and requires the score to hold up within NEIGHBOR_THRESHOLD at each one.

    Perturbations are built CONDITIONALLY on which mode is in use -- e.g.
    n_trail_p is only perturbed if n_exit_m==1 actually reads it. Testing an
    inactive parameter would trivially pass (the score can't change) and
    silently understate how brittle the fit really is.
    """
    if base_score <= 0: return True

    perturbations = [
        {'s_ma': p['s_ma'] + 2, 'l_ma': p['l_ma'] + 2},
        {'s_ma': p['s_ma'] - 2, 'l_ma': p['l_ma'] - 2},
        {'s_ma': p['s_ma'] + 3, 'l_ma': p['l_ma'] - 3},
    ]

    # Super MA only matters for entry_f in {1,2} (momentum/pullback filter)
    # or div_exit_m==1 (Super MA violation exit for dividend stocks).
    if p['entry_f'] in (1, 2) or p['div_exit_m'] == 1:
        perturbations += [
            {'sl_ma': max(50, int(p['sl_ma'] * 0.95))},
            {'sl_ma': int(p['sl_ma'] * 1.05)},
        ]

    # Growth-stock exit params only matter for the exit method actually active.
    if p['n_exit_m'] == 1:
        perturbations += [
            {'n_trail_p': p['n_trail_p'] * 0.85},
            {'n_trail_p': p['n_trail_p'] * 1.15},
        ]
    elif p['n_exit_m'] == 2:
        perturbations += [
            {'n_atr_m': p['n_atr_m'] * 0.80},
            {'n_atr_m': p['n_atr_m'] * 1.20},
        ]

    # Dividend-stock exit value, scaled to whichever exit mode is active
    # (peak-drawdown % and super-MA-violation % behave the same way;
    # time-decay is a day count, so it's nudged additively instead).
    if p['div_exit_m'] in (0, 1):
        perturbations += [
            {'div_exit_v': p['div_exit_v'] * 0.85},
            {'div_exit_v': p['div_exit_v'] * 1.15},
        ]
    else:
        perturbations += [
            {'div_exit_v': max(1.0, p['div_exit_v'] - 10)},
            {'div_exit_v': p['div_exit_v'] + 10},
        ]

    # ADX filter, nudged to a neighboring allowed value -- only if it's on.
    if p.get('adx_thresh', 0.0) > 0.0:
        adx_ladder = [15.0, 20.0, 25.0]
        idx = adx_ladder.index(p['adx_thresh']) if p['adx_thresh'] in adx_ladder else 1
        for neighbor_idx in (idx - 1, idx + 1):
            if 0 <= neighbor_idx < len(adx_ladder):
                perturbations.append({'adx_thresh': adx_ladder[neighbor_idx]})

    # NEW: Nifty-regime filter, only perturbed if it's actually active --
    # ported from the same conditional-perturbation principle as the ADX
    # ladder above and the crypto engine's own use_btc_filter perturbation.
    if p.get('use_nifty_filter', False):
        perturbations += [
            {'nifty_ma_len': max(10, int(p.get('nifty_ma_len', 50) * 0.90))},
            {'nifty_ma_len': int(p.get('nifty_ma_len', 50) * 1.10)},
        ]

    # NEW: watchlist-age weight, only perturbed if it's actually active.
    if p.get('use_wl_age_weight', False):
        perturbations += [
            {'wl_age_weight': max(0.0, p.get('wl_age_weight', 0.0) * 0.80)},
            {'wl_age_weight': p.get('wl_age_weight', 0.0) * 1.20},
        ]

    for delta in perturbations:
        n_p = p.copy()
        n_p.update(delta)

        if n_p['s_ma'] >= n_p['l_ma']: continue
        if n_p['s_ma'] < 3:            continue
        if n_p['l_ma'] > 200:          continue
        if n_p.get('n_trail_p', 15) < 3: continue

        n_score, _ = evaluate_params(n_p, opens, closes, atr, adx,
                                     sip_trigger, index_closes, is_div_stock, eligible_mask,
                                     years_arr, start_day, end_day, is_oos=False)
        if n_score < base_score * NEIGHBOR_THRESHOLD: return False
    return True


# ==========================================
# 9. WALK-FORWARD VALIDATION
# ==========================================

def run_wfo_validation(best_params, opens, closes, atr, adx,
                       sip_trigger, index_closes, is_div_stock, eligible_mask, years_arr):
    n_days    = closes.shape[0]
    is_end    = int(n_days * WFO_IS_PCT)
    oos_start = is_end

    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION")
    print(f"  In-Sample:     days 0-{is_end}      (~{is_end/252:.1f}y)")
    print(f"  Out-of-Sample: days {oos_start}-{n_days}   (~{(n_days - oos_start)/252:.1f}y, never seen during search)")
    print("=" * 60)

    is_score, is_m = evaluate_params(
        best_params, opens, closes, atr, adx,
        sip_trigger, index_closes, is_div_stock, eligible_mask, years_arr,
        start_day=0, end_day=is_end, is_oos=False,
        starting_wealth=0.0
    )

    is_end_wealth       = is_m.get('wealth', 0.0)      if is_score > -900 else 0.0
    is_end_bench_wealth = is_m.get('bench_wealth', 0.0) if is_score > -900 else 0.0

    oos_score, oos_m = evaluate_params(
        best_params, opens, closes, atr, adx,
        sip_trigger, index_closes, is_div_stock, eligible_mask, years_arr,
        start_day=oos_start, end_day=n_days - 1, is_oos=True,
        starting_wealth=is_end_wealth, bench_starting_wealth=is_end_bench_wealth
    )

    if oos_score <= -900:
        robustness = -1.0   # categorical "hard gate failure" marker, not a ratio
    elif is_score > 0:
        robustness = oos_score / abs(is_score)
    else:
        robustness = 0.0

    if is_score > 0:
        wt, lt = is_m.get('winning_trades', 0), is_m.get('losing_trades', 0)
        wr = wt / (wt + lt) * 100 if (wt + lt) > 0 else 0.0
        print(f"\nIn-Sample  Score: {is_score:.4f} | ROI: {is_m.get('roi', 0)*100:.1f}% | "
              f"IRR: {is_m.get('annual_return', 0)*100:.1f}% | Sharpe: {is_m.get('sharpe', 0):.2f} | WinRate: {wr:.1f}%")
    else:
        print(f"\nIn-Sample  Score: {is_score:.4f} (below gate)")
        print_gate_diagnosis(is_m, is_m.get('yearly', []), is_oos=False)

    if oos_score > 0:
        wt, lt = oos_m.get('winning_trades', 0), oos_m.get('losing_trades', 0)
        wr = wt / (wt + lt) * 100 if (wt + lt) > 0 else 0.0
        print(f"Out-of-Sample Score: {oos_score:.4f} | ROI: {oos_m.get('roi', 0)*100:.1f}% | "
              f"IRR: {oos_m.get('annual_return', 0)*100:.1f}% | Sharpe: {oos_m.get('sharpe', 0):.2f} | WinRate: {wr:.1f}%")
        if oos_m.get('avg_cash_frac', 0) > 0.15:
            print(f"  (NOTE: {oos_m['avg_cash_frac']*100:.0f}% of OOS portfolio value sat in cash on average, "
                  f"{oos_m.get('pct_days_high_cash',0):.0f}% of days above {HIGH_CASH_FRACTION_THRESHOLD*100:.0f}% cash.)")
    else:
        print(f"Out-of-Sample Score: {oos_score:.4f} (failed OOS hard gates)")
        # This is the key fix: show WHICH gate(s) actually failed instead
        # of leaving the reader to guess between 6+ possible causes.
        if oos_m:
            print_gate_diagnosis(oos_m, oos_m.get('yearly', []), is_oos=True)
        else:
            print("  (No metrics available -- likely s_ma >= l_ma or zero invested capital.)")

    if robustness <= -1.0:
        print("\nRobustness Ratio: N/A -- OOS failed a hard gate outright. REJECT.")
    else:
        print(f"\nRobustness Ratio: {robustness:.1%}  ", end="")
        if robustness >= 0.70:
            print("EXCELLENT (>70%) -- deploy with confidence")
        elif robustness >= ROBUSTNESS_DEPLOY_THRESHOLD:
            print(f"ACCEPTABLE ({ROBUSTNESS_DEPLOY_THRESHOLD:.0%}-70%) -- deploy cautiously  <- meets save threshold")
        elif robustness >= 0.30:
            print(f"POOR (30%-{ROBUSTNESS_DEPLOY_THRESHOLD:.0%}) -- likely overfit, do not deploy  <- below save threshold")
        else:
            print("REJECT (<30%) -- heavily overfit, discard")

    return is_score, oos_score, oos_m, robustness, is_end_wealth


def report_subperiod_breakdown(best_params, opens, closes, atr, adx,
                               sip_trigger, index_closes, is_div_stock, eligible_mask, years_arr,
                               n_blocks=5):
    """Re-evaluates the already-found champion on n_blocks contiguous
    sub-windows of the full history (no re-optimization) -- checks whether
    it's consistent across regimes or just lived/died in one lucky stretch."""
    n_days = closes.shape[0]
    edges = np.linspace(1, n_days - 1, n_blocks + 1).astype(int)

    print("\n" + "=" * 60)
    print(f"SUB-PERIOD CONSISTENCY CHECK ({n_blocks} blocks, same champion params)")
    print("=" * 60)
    rows = []
    for i in range(n_blocks):
        s_day, e_day = int(edges[i]), int(edges[i + 1])
        score, m = evaluate_params(
            best_params, opens, closes, atr, adx,
            sip_trigger, index_closes, is_div_stock, eligible_mask, years_arr,
            start_day=s_day, end_day=e_day, is_oos=False, starting_wealth=0.0
        )
        rows.append((s_day, e_day, score, m))
        if score > -900:
            wt, lt = m.get('winning_trades', 0), m.get('losing_trades', 0)
            wr = wt / (wt + lt) * 100 if (wt + lt) > 0 else 0.0
            print(f"  Block {i+1} (days {s_day:5d}-{e_day:5d}, ~{(e_day-s_day)/252:.1f}y): "
                  f"Score {score:7.3f} | IRR {m.get('annual_return',0)*100:6.1f}% | "
                  f"Sharpe {m.get('sharpe',0):5.2f} | MaxDD {abs(m.get('max_dd',0))*100:5.1f}% | "
                  f"WinRate {wr:5.1f}% | Trades {m.get('trades',0)}")
        else:
            print(f"  Block {i+1} (days {s_day:5d}-{e_day:5d}): below minimum-trades/months gate "
                  f"for a block this short -- inconclusive, not a failure.")
    return rows


def run_overfitting_diagnostic(all_trial_sharpes_annualized, champion_metrics, T):
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014), using every
    completed Optuna trial's Sharpe plus the champion's own skew/kurtosis."""
    if len(all_trial_sharpes_annualized) < 30:
        print("\n(Skipping DSR diagnostic -- need >=30 completed trials with valid Sharpe.)")
        return None

    # Kurtosis is bounded below by skew^2+1 for any real distribution, so
    # a value <=0 signals "couldn't be computed" (too few valid days or
    # zero return variance) -- fall back to a generic estimate only then.
    kurt_computed = champion_metrics.get('kurtosis', 0.0)
    if kurt_computed > 0:
        skew = champion_metrics.get('skew', -0.3)
        kurt = kurt_computed
        moment_source = "champion's own daily returns"
    else:
        skew = -0.3
        kurt = 5.0
        moment_source = "generic placeholder -- too few/uniform daily returns to compute the champion's own"

    sr_hat_annual = champion_metrics.get('sharpe', 0.0)
    sr_hat_daily  = sr_hat_annual / math.sqrt(TRADING_DAYS_PER_YEAR)
    trial_sharpes_daily = [s / math.sqrt(TRADING_DAYS_PER_YEAR) for s in all_trial_sharpes_annualized]

    dsr, sr0_daily, n_trials = deflated_sharpe_ratio(sr_hat_daily, trial_sharpes_daily, T, skew, kurt)
    sr0_annual = sr0_daily * math.sqrt(TRADING_DAYS_PER_YEAR)

    print("\n" + "=" * 60)
    print("OVERFITTING DIAGNOSTIC 1/2 -- Deflated Sharpe Ratio")
    print("=" * 60)
    print(f"  Trials used: {n_trials}   Sample size (T): {T}   Skew/Kurt: {skew:.3f} / {kurt:.3f}  ({moment_source})")
    print(f"  Champion Sharpe (annualized): {sr_hat_annual:.3f}   Expected max noise Sharpe: {sr0_annual:.3f}")
    print(f"  Deflated Sharpe Ratio (probability): {dsr:.3f}  ", end="")
    if dsr > 0.95:
        print("-> clears the noise threshold with high confidence")
    elif dsr > 0.70:
        print("-> plausibly real edge, not overwhelming -- treat cautiously")
    else:
        print("-> can't statistically distinguish from the best of thousands of noise trials -- high overfit risk")
    print("  CAVEAT: computed on Sharpe (has known sampling theory), not the composite score actually")
    print("  optimized. Read as one indicative angle, not an exact p-value -- weigh the OOS result more.")
    return dsr


def report_score_distribution_diagnostic(champion_score, all_trial_scores):
    """Purely descriptive: where does the champion's score sit among every
    OTHER trial that already passed every hard gate? Not a new statistical
    test -- read alongside the DSR figure and the OOS result, not instead."""
    if len(all_trial_scores) < 10:
        print("\n(Skipping score-distribution diagnostic -- fewer than 10 other gate-passing trials.)")
        return None

    arr = np.array(all_trial_scores, dtype=np.float64)
    mean_s, std_s = float(arr.mean()), float(arr.std())
    pct_below = float((arr < champion_score).mean() * 100.0)
    z = (champion_score - mean_s) / std_s if std_s > 0 else float('inf')

    print("\n" + "=" * 60)
    print("OVERFITTING DIAGNOSTIC 2/2 -- Score-Distribution Sanity Check")
    print("=" * 60)
    print(f"  Other gate-passing trials: {len(arr)}   Champion score: {champion_score:.4f}")
    print(f"  Mean/std of others: {mean_s:.4f} / {std_s:.4f}   Champion beats {pct_below:.1f}% of them ({z:.2f} std above mean)")
    if pct_below > 99.0:
        print("  -> A clear outlier even among gate-passing trials. Still just the best of ~12,000 attempts --")
        print("     weigh alongside DSR and the OOS result, not instead of them.")
    else:
        print("  -> Several other trials scored comparably -- the OOS walk-forward result matters more than this.")
    return {'mean': mean_s, 'std': std_s, 'pct_below': pct_below, 'z': z}


# ==========================================
# 9b. TEMPORAL ROBUSTNESS -- RANDOM SIP-DAY CHECK
# ==========================================
# Every evaluate_params() call needs a sip_trigger array (one True per
# calendar month, marking which trading day the SIP lands on).
# build_sip_trigger_mask() is the only place that gets built:
#   - fixed_offset=0 (default) = always the month's 1st trading day.
#   - rng=<Generator> = independently randomizes the offset each month,
#     used by the check below to test whether the edge depends on SIP timing.

def build_sip_trigger_mask(months, rng=None, fixed_offset=0):
    n_days = len(months)
    trigger = np.zeros(n_days, dtype=np.bool_)
    if n_days == 0:
        return trigger
    idx = 0
    while idx < n_days:
        j = idx
        while j < n_days and months[j] == months[idx]:
            j += 1
        month_len = j - idx
        if rng is not None:
            offset = int(rng.integers(0, month_len))
        else:
            offset = min(fixed_offset, month_len - 1)
        trigger[idx + offset] = True
        idx = j
    return trigger


def run_temporal_robustness_check(best_params, opens, closes, atr, adx,
                                   months, index_closes, is_div_stock, eligible_mask, years_arr,
                                   baseline_score, n_runs=TEMPORAL_ROBUSTNESS_RUNS):
    """Reruns the already-chosen champion n_runs times with an
    independently random SIP day each month (nothing re-optimized) and
    checks whether the score survives. Evaluated on the FULL dataset --
    this is a calendar-mechanics question, not a market-regime one."""
    n_days = closes.shape[0]
    scores = []
    n_gate_failures = 0

    print("\n" + "=" * 60)
    print(f"TEMPORAL ROBUSTNESS -- RANDOM SIP-DAY CHECK ({n_runs} reruns)")
    print("=" * 60)

    for i in range(n_runs):
        rng = np.random.default_rng(seed=5000 + i)
        trigger = build_sip_trigger_mask(months, rng=rng)
        score, _ = evaluate_params(best_params, opens, closes, atr, adx,
                                    trigger, index_closes, is_div_stock, eligible_mask, years_arr,
                                    start_day=0, end_day=n_days - 1, is_oos=False,
                                    starting_wealth=0.0)
        if score > -900:
            scores.append(score)
        else:
            n_gate_failures += 1

    if not scores:
        print("  All randomized-SIP-day reruns FAILED the hard gates entirely.")
        print("  -> Viability depends on first-of-month SIP timing. REJECT.")
        return 0.0, []

    arr = np.array(scores)
    mean_s, min_s, max_s = float(arr.mean()), float(arr.min()), float(arr.max())
    ratio = mean_s / abs(baseline_score) if baseline_score > 0 else 0.0

    print(f"  Baseline (1st-of-month) score: {baseline_score:.4f}")
    print(f"  Reruns passing gates: {len(scores)}/{n_runs}   mean/min/max: {mean_s:.4f} / {min_s:.4f} / {max_s:.4f}")
    if n_gate_failures > 0:
        print(f"  WARNING: {n_gate_failures}/{n_runs} reruns failed hard gates entirely.")
    print(f"  Temporal Robustness Ratio: {ratio:.1%}  ", end="")
    if ratio >= 0.70 and n_gate_failures == 0:
        print("STRONG -- edge does not depend on SIP timing")
    elif ratio >= TEMPORAL_ROBUSTNESS_THRESHOLD and n_gate_failures <= n_runs * 0.2:
        print("ACCEPTABLE -- some sensitivity to cash-arrival timing")
    else:
        print("WEAK -- edge appears to depend on first-of-month SIP timing")

    return ratio, scores


# ==========================================
# 10. CONSOLE REPORTER
# ==========================================

def _fmt_pf(pf_val):
    return "inf (no losing trades)" if (isinstance(pf_val, float) and math.isinf(pf_val)) else f"{pf_val:.2f}"


def print_performance_report(score, metrics, p, label=""):
    ma_names = ["SMA", "EMA", "DEMA", "WMA"]
    ts, tl, tsl = ma_names[p['t_s']], ma_names[p['t_l']], ma_names[p['t_sl']]

    entry_names = [
        "Unfiltered (Any Crossover Valid)",
        f"Momentum (Price > {tsl}{p['sl_ma']})",
        f"Value/Pullback (Price < {tsl}{p['sl_ma']})"
    ]
    norm_exit_names = [
        f"Crossunder ({ts}{p['s_ma']} < {tl}{p['l_ma']})",
        f"Fixed Trailing Stop ({p['n_trail_p']:.1f}% from Peak)",
        f"Volatility Trailing Stop ({p['n_atr_m']:.2f}x 14-ATR(Wilder) from Peak)"
    ]
    rank_names = [
        "Max Drawdown (Deepest Discount from Signal)",
        "Risk-Off (Closest to Long MA Support)",
        "Momentum (Highest Velocity above Short MA)"
    ]
    div_exit_names = [
        "Fixed Peak Drawdown (%)",
        "Super MA Violation (%)",
        "Time Decay (Consecutive Days Below Long MA)"
    ]

    wt = metrics.get('winning_trades', 0)
    lt = metrics.get('losing_trades',  0)
    actual_wr   = wt / (wt + lt) if (wt + lt) > 0 else 0.0
    adx_info    = f"ADX>{p.get('adx_thresh', 0):.0f}" if p.get('adx_thresh', 0) > 0 else "Disabled"

    avg_runup   = metrics.get('avg_runup', 0)
    avg_loss_m  = abs(metrics.get('avg_loss', 1))
    sortino_raw = metrics.get('sortino', 0)
    sortino_cap = min(sortino_raw, 4.0)
    ir          = metrics.get('ir', 0)
    month_cnt   = metrics.get('month_cnt', 1)
    ir_scaled   = max(0.0, ir) * math.sqrt(max(1.0, month_cnt) / 12.0)
    wr_bonus    = max(0.0, actual_wr - 0.50) * 4.0

    ev       = actual_wr * avg_runup - (1.0 - actual_wr) * avg_loss_m
    ev_in_r  = ev / avg_loss_m if avg_loss_m > 0 else 0.0

    start_w = metrics.get('starting_wealth', 0.0)
    annual_return = metrics.get('annual_return', 0.0)
    irr_note = "" if metrics.get('irr_converged', True) else "  (solver did not converge -- approximate)"

    floor_clamp_warning = ""
    if metrics.get('n_floor_clamps', 0) > 0:
        floor_clamp_warning = (
            f"\n*** WARNING: floor clamp triggered {metrics['n_floor_clamps']}x this run -- "
            f"should never happen on real data. Check for a NaN/zero-price leak before trusting these numbers. ***\n"
        )

    bench_annual_return = None
    cf_days = metrics.get('cf_days', [])
    cf_amounts = metrics.get('cf_amounts', [])
    if len(cf_days) >= 2 and metrics.get('bench_wealth', 0) > 0:
        bench_cf_amounts = list(cf_amounts[:-1]) + [metrics['bench_wealth']]
        # Swap in the benchmark's own carried-over starting wealth (not the
        # strategy's) so this IRR reflects a genuine "just the index, same
        # cash flows" trajectory.
        if metrics.get('starting_wealth', 0.0) > 0 and len(bench_cf_amounts) >= 1:
            bench_cf_amounts[0] = -metrics.get('bench_starting_wealth', 0.0)
        bench_annual_return, bench_irr_ok = money_weighted_annual_return(cf_days, bench_cf_amounts)
        if not bench_irr_ok:
            bench_annual_return = None

    print(f"""
======================================================
{label if label else 'DIAMOND EXTRACTOR — CHAMPION STRATEGY REPORT (v5.0)'}
Composite Score: {score:.4f}{floor_clamp_warning}
------------------------------------------------------
[CAPITAL & RETURNS]
Starting Capital Anchor:  Rs.{start_w:,.0f}
Total SIP Injected:       Rs.{metrics['t_invested']:,.0f}
Final Portfolio Wealth:   Rs.{metrics['wealth']:,.0f}
System ROI (vs base):     {metrics['roi']*100:.2f}%  (Index SIP ROI: {metrics['bench_roi']*100:.2f}%)
Excess Alpha (ROI):       {metrics['alpha']*100:+.2f}%
Annualized Return (IRR):  {annual_return*100:.2f}%{irr_note}
""" + (f"Nifty SIP Annualized (IRR): {bench_annual_return*100:.2f}%   (apples-to-apples comparison)\n" if bench_annual_return is not None else "") + f"""------------------------------------------------------
[RISK & PERFORMANCE]
Sharpe Ratio:             {metrics['sharpe']:.2f}
Sortino Ratio:            {sortino_raw:.2f}  (capped to {sortino_cap:.2f} in score)
Return Skew / Kurtosis:   {metrics.get('skew', 0):.3f} / {metrics.get('kurtosis', 0):.3f}
Information Ratio:        {ir:.3f}  (scaled: {ir_scaled:.3f})
Max Portfolio Drawdown:   {metrics['max_dd']*100:.2f}%
Profit Factor:            {_fmt_pf(metrics.get('pf', 0.0))}
------------------------------------------------------
[TRADE STATISTICS]
Total Closed Trades:      {metrics['trades']}
  Winning:                {wt}
  Losing:                 {lt}
Win Rate:                 {actual_wr*100:.1f}%   <-- hard gate: must be >= {MIN_WIN_RATE_GATE*100:.0f}%
Avg Bars in Trade:        {metrics['avg_bars']:.0f} days
Avg Win:                  +{avg_runup*100:.2f}%
Avg Loss:                 -{avg_loss_m*100:.2f}%
R:R Ratio:                {avg_runup/avg_loss_m if avg_loss_m > 0 else 0:.2f}x
Expected Value / R:       {ev_in_r:.3f}
------------------------------------------------------
[SCORE BREAKDOWN  (Calmar 30 / Sortino 10 / IR 30 / EV 15 / WR_Bonus 15)]
Calmar (AnnRet/MaxDD):     {(annual_return/max(abs(metrics['max_dd']),0.001)):.3f}  x0.30
Sortino (cap 4.0):        {sortino_cap:.2f}   x0.10 = {sortino_cap*0.10:.3f}
IR scaled:                {ir_scaled:.3f}  x0.30 = {ir_scaled*0.30:.3f}
EV in R:                  {ev_in_r:.3f}  x0.15 = {ev_in_r*0.15:.3f}
WR Bonus (WR-50)*4:       {wr_bonus:.3f}  x0.15 = {wr_bonus*0.15:.3f}
------------------------------------------------------
[DIAGNOSTICS]
Avg Cash Sitting Idle:     {metrics.get('avg_cash_frac', 0)*100:.1f}% of portfolio value
Days >{HIGH_CASH_FRACTION_THRESHOLD*100:.0f}% in Cash:        {metrics.get('pct_days_high_cash', 0):.1f}% of this window
Floor-Clamp Triggers:     {metrics.get('n_floor_clamps', 0)}  (should be 0)
Valid Return-Series Days: {metrics.get('valid_days', 0)}
------------------------------------------------------
[SIGNAL CONFIGURATION]
MA Signal:                {ts}{p['s_ma']} x {tl}{p['l_ma']}
Super MA (Filter):        {tsl}{p['sl_ma']}
Entry Filter:             {entry_names[p['entry_f']]}
ADX Filter:               {adx_info}
Growth Exit:              {norm_exit_names[p['n_exit_m']]}
Watchlist Priority:       {rank_names[p['wl_rank']]}
Div Stock Exit:           {div_exit_names[p['div_exit_m']]} @ {p['div_exit_v']:.1f}
------------------------------------------------------
[COSTS & FIXED SETTINGS]
Buy/Sell cost:             {BUY_COST_PCT*100:.3f}% / {SELL_COST_PCT*100:.3f}%   Flat DP charge: Rs.{DP_FLAT_FEE_RS:.0f}   Idle cash yield: {ANNUAL_CASH_YIELD*100:.2f}%
Position Sizing:          Fixed Rs.{MONTHLY_SIP:,} per signal   Pyramiding: ON
Exit Model:                Two-phase (signal @ close, fill @ next open)
Eligibility Mask:         {'Point-in-time file loaded' if POINT_IN_TIME_UNIVERSE_FILE else 'All-eligible (survivorship-biased, disclosed)'}
Robustness Save Gate:     >= {ROBUSTNESS_DEPLOY_THRESHOLD:.0%} OOS  AND  >= {TEMPORAL_ROBUSTNESS_THRESHOLD:.0%} temporal
======================================================
""")


# ==========================================
# 11. PARAM HELPERS (Optuna <-> canonical)
# ==========================================

def _params_to_trial_dict(p):
    trial = {k: v for k, v in p.items() if k != 'div_exit_v'}
    div_m = int(p.get('div_exit_m', 0))
    div_v = float(p.get('div_exit_v', 10.0))
    if   div_m == 0: trial['div_val_peak_pct'] = div_v
    elif div_m == 1: trial['div_val_ma_pct']   = div_v
    else:            trial['div_val_days']      = int(div_v)
    for stale in ('rs_thresh', 'mkt_regime', 'vol_sizing', 'vol_cap_r', 'use_rs',
                  'div_val', 'div_val_int',
                  'div_val_type_0', 'div_val_type_1', 'div_val_type_2_int'):
        trial.pop(stale, None)
    if 'adx_thresh' in trial:
        trial['adx_thresh'] = float(trial['adx_thresh'])
    return trial


def _trial_to_params(trial_params):
    p = dict(trial_params)
    if   'div_val_peak_pct' in p: p['div_exit_v'] = p.pop('div_val_peak_pct')
    elif 'div_val_ma_pct'   in p: p['div_exit_v'] = p.pop('div_val_ma_pct')
    elif 'div_val_days'     in p: p['div_exit_v'] = float(p.pop('div_val_days'))
    return p


# ==========================================
# 12. MAIN OPTIMIZER
# ==========================================

def run_optimization():
    tickers = fetch_dynamic_universe()
    (opens, closes, atr, adx,
     months, index_closes,
     is_div_stock, stock_names, eligible_mask, years_arr) = prepare_matrix_data(tickers)

    clear_ma_cache()  # cache is keyed without a data fingerprint -- must clear before a fresh matrix

    sip_trigger = build_sip_trigger_mask(months, fixed_offset=0)

    n_days = closes.shape[0]
    is_end = int(n_days * WFO_IS_PCT)

    print(f"\nMatrix: {n_days} days x {closes.shape[1]} stocks")
    print(f"IS period:  days 0-{is_end}  ({is_end/252:.1f} yrs)")
    print(f"OOS period: days {is_end}-{n_days}  ({(n_days-is_end)/252:.1f} yrs -- never seen during opt)")

    best_is_score = -999999.0
    best_params   = None
    all_trial_sharpes = []   # feeds the Sharpe-based DSR diagnostic
    all_trial_scores  = []   # feeds the score-distribution diagnostic

    prev_oos, prev_is, prev_params = load_previous_winner(BEST_PARAMS_FILE)
    if prev_params is None:
        _, prev_is, prev_params = load_previous_winner(INTERMEDIATE_FILE)

    if prev_params is not None:
        print("\n" + "-" * 60)
        print("EVALUATING LOADED CHAMPION CONFIGURATION")
        print("-" * 60)

        score_is, metrics_is = evaluate_params(
            prev_params, opens, closes, atr, adx,
            sip_trigger, index_closes, is_div_stock, eligible_mask, years_arr,
            start_day=0, end_day=is_end, is_oos=False, starting_wealth=0.0
        )
        score_full, metrics_full = evaluate_params(
            prev_params, opens, closes, atr, adx,
            sip_trigger, index_closes, is_div_stock, eligible_mask, years_arr,
            start_day=0, end_day=n_days - 1, is_oos=False, starting_wealth=0.0
        )

        if score_is > -900:
            best_is_score = score_is
            best_params   = prev_params
            print(f"Loaded champion passes IS gates. Baseline: {best_is_score:.4f}")
            print("NOTE: the report below evaluates the WHOLE dataset (IS+OOS combined), not a")
            print("held-out test -- it is expected to look strong since most of that window is")
            print("the data this champion was originally optimized on. See WALK-FORWARD VALIDATION")
            print("further down for the actual held-out-only result.")
            print_performance_report(score_full, metrics_full, prev_params,
                                     "LOADED CHAMPION (LIFETIME FULL-DATA PERFORMANCE)")
        else:
            print("Loaded champion no longer passes strict IS gates. Starting fresh.")

    # ==========================================
    # BAYESIAN OPTIMIZATION
    # ==========================================

    if OPTUNA_AVAILABLE:
        print(f"\nBayesian Optimization ({BAYESIAN_TRIALS} trials on IS data only, n_jobs={OPTUNA_N_JOBS})...")

        def optuna_objective(trial):
            s_ma  = trial.suggest_int('s_ma',  20,  80)
            l_ma  = trial.suggest_int('l_ma',  40, 150)
            sl_ma = trial.suggest_int('sl_ma', 150, 300)
            if s_ma >= l_ma: raise optuna.TrialPruned()

            div_exit_m = trial.suggest_int('div_exit_m', 0, 2)
            if   div_exit_m == 0: div_val = trial.suggest_float('div_val_peak_pct', 10.0, 40.0)
            elif div_exit_m == 1: div_val = trial.suggest_float('div_val_ma_pct',    0.0, 20.0)
            else:                 div_val = float(trial.suggest_int('div_val_days', 10, 100))

            p = {
                's_ma':       s_ma,
                'l_ma':       l_ma,
                'sl_ma':      sl_ma,
                't_s':        trial.suggest_int('t_s',  0, 3),
                't_l':        trial.suggest_int('t_l',  0, 3),
                't_sl':       trial.suggest_int('t_sl', 0, 3),
                'wl_rank':    trial.suggest_int('wl_rank',  0, 2),
                'entry_f':    trial.suggest_int('entry_f',  0, 2),
                'n_exit_m':   trial.suggest_int('n_exit_m', 0, 2),
                'n_trail_p':  trial.suggest_float('n_trail_p', 5.0, 35.0),
                'n_atr_m':    trial.suggest_float('n_atr_m',   1.0,  5.0),
                'div_exit_m': div_exit_m,
                'div_exit_v': float(div_val),
                'adx_thresh': trial.suggest_categorical('adx_thresh', [0.0, 15.0, 20.0, 25.0]),
                # NEW: optional Nifty-regime panic-exit/entry filter, ported
                # from the crypto engine's use_btc_filter -- growth stocks
                # only (see module docstring). Toggle is searched so Optuna
                # decides whether it actually helps this configuration
                # instead of it being forced on or off.
                'use_nifty_filter': trial.suggest_categorical('use_nifty_filter', [True, False]),
                'nifty_ma_len':     trial.suggest_int('nifty_ma_len', 20, 150),
                'nifty_ma_type':    trial.suggest_int('nifty_ma_type', 0, 3),
                # NEW: optional watchlist-age ranking boost -- off by default,
                # searched as a toggle so Optuna only keeps it if it actually
                # helps (see WL_AGE_NORM_DAYS module comment).
                'use_wl_age_weight': trial.suggest_categorical('use_wl_age_weight', [True, False]),
                'wl_age_weight':     trial.suggest_float('wl_age_weight', 0.0, 0.50),
            }

            score, metrics = evaluate_params(p, opens, closes, atr, adx,
                                       sip_trigger, index_closes, is_div_stock, eligible_mask, years_arr,
                                       start_day=0, end_day=is_end, is_oos=False,
                                       starting_wealth=0.0)
            if metrics:
                all_trial_sharpes.append(metrics.get('sharpe', 0.0))
                if score > -900:
                    all_trial_scores.append(score)
            return score if score > -900 else -999.0

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, group=True)
        )
        if best_params is not None:
            study.enqueue_trial(_params_to_trial_dict(best_params))

        # Lock protects best_is_score/best_params from a race between
        # concurrently-completing trials when n_jobs > 1.
        champion_lock = threading.Lock()

        def optuna_callback(study, trial):
            nonlocal best_is_score, best_params

            if trial.value is None or trial.value < -900: return

            with champion_lock:
                if trial.value <= best_is_score:
                    return
                p_full = _trial_to_params(trial.params)

                if passes_neighborhood_check(
                    p_full, trial.value, opens, closes, atr, adx,
                    sip_trigger, index_closes, is_div_stock, eligible_mask, years_arr, 0, is_end
                ):
                    best_is_score = trial.value
                    best_params   = p_full

                    score_chk, metrics = evaluate_params(
                        p_full, opens, closes, atr, adx,
                        sip_trigger, index_closes, is_div_stock, eligible_mask, years_arr,
                        0, is_end, is_oos=False, starting_wealth=0.0
                    )
                    print_performance_report(score_chk, metrics, p_full,
                                             f"NEW IS CHAMPION  (trial {trial.number})")
                    save_winner(0.0, trial.value, p_full, filename=INTERMEDIATE_FILE)

        study.optimize(optuna_objective, n_trials=BAYESIAN_TRIALS,
                       callbacks=[optuna_callback], show_progress_bar=True,
                       n_jobs=OPTUNA_N_JOBS)

    # ==========================================
    # WALK-FORWARD VALIDATION
    # ==========================================
    if best_params is not None:
        is_score, oos_score, oos_metrics, robustness, is_end_wealth = run_wfo_validation(
            best_params, opens, closes, atr, adx,
            sip_trigger, index_closes, is_div_stock, eligible_mask, years_arr
        )

        report_subperiod_breakdown(best_params, opens, closes, atr, adx,
                                   sip_trigger, index_closes, is_div_stock, eligible_mask, years_arr)

        temporal_ratio = 0.0   # only computed below if OOS cleared its own hard gates

        if oos_score > -900:
            T = oos_metrics.get('valid_days', n_days - is_end)
            run_overfitting_diagnostic(all_trial_sharpes, oos_metrics, T=T)
            report_score_distribution_diagnostic(best_is_score, all_trial_scores)

            full_score, full_metrics = evaluate_params(
                best_params, opens, closes, atr, adx,
                sip_trigger, index_closes, is_div_stock, eligible_mask, years_arr,
                start_day=0, end_day=n_days - 1, is_oos=False, starting_wealth=0.0
            )
            temporal_ratio, _ = run_temporal_robustness_check(
                best_params, opens, closes, atr, adx,
                months, index_closes, is_div_stock, eligible_mask, years_arr,
                baseline_score=full_score, n_runs=TEMPORAL_ROBUSTNESS_RUNS
            )

        if (oos_score > -900 and (oos_score > prev_oos or prev_oos <= 0)
                and robustness >= ROBUSTNESS_DEPLOY_THRESHOLD
                and temporal_ratio >= TEMPORAL_ROBUSTNESS_THRESHOLD):
            print("\nOOS + temporal-robustness validation passed. Saving fully verified champion...")
            save_winner(oos_score, is_score, best_params, filename=BEST_PARAMS_FILE,
                       robustness_ratio=robustness)
            print_performance_report(oos_score, oos_metrics, best_params,
                                     "OUT-OF-SAMPLE VALIDATED CHAMPION")
        elif robustness < ROBUSTNESS_DEPLOY_THRESHOLD:
            robustness_desc = "N/A (OOS failed a hard gate outright)" if robustness <= -1.0 else f"{robustness:.1%}"
            print(f"\nStrategy rejected -- OOS robustness ({robustness_desc}) below this "
                  f"script's {ROBUSTNESS_DEPLOY_THRESHOLD:.0%} save threshold.")
        elif temporal_ratio < TEMPORAL_ROBUSTNESS_THRESHOLD:
            print(f"\nStrategy rejected -- temporal (random-SIP-day) robustness ratio "
                  f"({temporal_ratio:.1%}) below this script's "
                  f"{TEMPORAL_ROBUSTNESS_THRESHOLD:.0%} save threshold. The edge may depend "
                  f"on first-of-month SIP timing rather than genuine signal quality.")
        else:
            print(f"\nNew IS champion found but OOS score ({oos_score:.4f}) "
                  f"did not beat previous champion OOS ({prev_oos:.4f}).")
    else:
        print("\nNo strategy found that passes all quality gates.")


if __name__ == "__main__":
    run_optimization()
