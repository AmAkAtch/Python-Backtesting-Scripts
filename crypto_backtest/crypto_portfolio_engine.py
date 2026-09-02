"""
CRYPTO PORTFOLIO & WEALTH ENGINE
==================================
SCRIPT 2 of 4 in the consolidated pipeline. Sibling: crypto_signal_engine.py
(which must be run first -- this script reads its output).

THE ONE QUESTION THIS FILE ANSWERS:
  "Given a signal that already proved itself independently (+EV, robust
   across coins/years, in crypto_signal_engine.py's shortlist), how does
   it actually compound when subjected to a real, shared, monthly-funded
   cash pool, position queues, realistic Binance costs, and drawdown
   constraints?"

THIS FILE DOES NOT SEARCH FOR SIGNALS. Every entry/exit parameter it
tests comes frozen from crypto_signal_shortlist.json. This script's own
config block controls only genuinely portfolio-level POLICY (SIP size,
position-sizing rule, watchlist ranking rule, transaction costs) --
policy that's the same regardless of which shortlisted signal is being
tested, so every candidate is judged on a level playing field. Nothing
here can silently discard a good signal for being outside some narrow
entry/exit template -- if it's in the shortlist, this file can run it,
because the entry/exit taxonomy and every toggle (BTC gate/override, RSI
trend filter, ADX threshold, latched-entry confluence, pyramiding) here
is the exact same taxonomy crypto_signal_engine.py searched over.

WATCHLIST CANDIDATES ARE SHADOW POSITIONS, not a hand-waved grace period.
A coin whose signal fired but has no capital behind it yet sits on the
watchlist with the SAME stop-loss/TP/trail state a funded position would
have, and leaves the watchlist via the exact same exit rule that would
have sold it -- never a separate arbitrary max-age/grace-period heuristic.

CONSOLIDATED FROM THE PORTFOLIO-SIMULATION LINEAGE:
  - Unified cash pool + unconditional monthly SIP, equal-weight target
    position sizing `(cash + cost-basis of open positions) / n_eligible`
    (recomputed every deployment step, cost-basis not mark-to-market, so
    one winner's unrealized paper gain never inflates an unrelated next
    trade's size) -- replaces an earlier, more primitive fixed-dollar-
    chunk deployment loop for the reasons documented at the time: fixed
    chunks either starve simultaneous opportunities or leave cash idle.
  - Two-phase signal-at-close/fill-at-next-open execution discipline,
    identical to the signal engine, so a trade that would have filled
    there fills the same way here.
  - Recency-weighted CONSISTENCY term + bull/bear REGIME term (scored
    against BTC's own annual return), partial-year coverage gate (a
    half-year window was once seen annualizing into a fake +285% IRR
    without this -- excluded from the scored terms, not from the report).
  - A HARD profit-concentration gate here (unlike the signal engine's
    SOFT penalty) -- deliberate difference: this file is deciding whether
    real money should go behind a candidate, so an extreme one-year/one-
    coin dependency is disqualifying here in a way it correctly isn't at
    the signal-screening stage.
  - Temporal robustness check: rerun a candidate with the SIP landing on
    a randomized day each month; an edge that only survives because it
    always deploys capital at month-start conditions is a real red flag
    a fixed-schedule backtest can hide.
  - Exact money-weighted XIRR via bisection (a single sign-change cash-
    flow stream guarantees a unique real root by Descartes' rule of
    signs), and cash-flow-adjusted daily returns for Sharpe/Sortino/
    skew/kurtosis (a naive P&L delta would count every SIP deposit as a
    fabricated one-day gain).
  - Tiered, always-save-something champion policy (EXCELLENT/ACCEPTABLE/
    CAUTION/POOR/GATE-FAILURE) instead of all-or-nothing.

A DELIBERATE, NARROW IMPORT from crypto_signal_engine.py: only the
stateless indicator math (calc_ma, calc_rsi_wilder, calc_atr_wilder,
calc_adx and their caches) is shared, specifically so the two files can
never silently drift into disagreeing about what "RSI(14)" means. The
actual simulation loop, scoring, and every gate are fully independent --
that separation is the whole point of keeping these as two files.

CHANGELOG
  v1.0  Initial consolidated build.
"""

import math
import json
import os
import time
import threading
import warnings
from datetime import datetime, timezone
from statistics import NormalDist

import numpy as np
import pandas as pd
import requests
from numba import njit
from tqdm import tqdm

try:
    from crypto_signal_engine import (
        calc_ma, calc_rsi_wilder, calc_atr_wilder, calc_adx,
        get_ma_cached, get_raw_rsi_cached, get_smoothed_rsi, clear_caches,
        ENTRY_MA_BREAKOUT, ENTRY_RSI_XOVER, ENTRY_MA_XOVER,
        EXIT_HYBRID, EXIT_PCT_TRAIL, EXIT_ATR_TRAIL, EXIT_MA_CROSSUNDER,
        EXIT_RSI_CROSSUNDER, EXIT_MA_XOVER_EXIT,
        build_liquidity_mask, _binance_download_klines,
    )
except ImportError as e:
    raise SystemExit(
        "crypto_portfolio_engine.py must sit in the same folder as "
        f"crypto_signal_engine.py (only indicator math is shared, so the "
        f"two files can never quietly disagree about what an indicator "
        f"means). Import error: {e}"
    )

warnings.filterwarnings('ignore')

# ==========================================================================
# 0. CONFIGURATION -- portfolio-level POLICY only. Entry/exit params come
# frozen from the shortlist; nothing here re-optimizes signal logic.
# ==========================================================================

SHORTLIST_FILE = "crypto_signal_shortlist.json"        # INPUT -- from crypto_signal_engine.py
CHAMPION_FILE  = "crypto_portfolio_champion.json"       # OUTPUT -- fully validated winner
INTERMEDIATE_FILE = "crypto_portfolio_current_best.json"
ENGINE_VERSION = "crypto-portfolio-1.0.0"

MAX_CANDIDATES_TO_TEST = 15   # how many shortlist entries to actually run (top-ranked first)

MONTHLY_SIP      = 2_000.0
# BUG FIX (found from a real run): this used to be `MIN_TICKET_SIZE = MONTHLY_SIP`,
# ported from a source script's explicit "$2,000, no less" instruction. That
# instruction was never stress-tested against equal-weight sizing over a LARGE
# universe: target_size = cash_pool / n_eligible means the ticket floor must be
# cleared by cash_pool alone before the FIRST trade can ever fire. With
# n_eligible=29 that requires 29 months (~2.4 years) of pure SIP accumulation
# with ZERO trades; with TOP_N_COINS=200 it's ~16.7 years. A real run with a
# 29-coin universe showed every single shortlisted candidate hard-failing
# every gate -- confirmed empirically (holding everything else fixed, dropping
# this floor from $2,000 to $50 took trade count from 115 to 179 on the same
# data, nearly matching the signal engine's own unconstrained count). This
# floor's only real job is "don't bother with dust trades" -- it should never
# have been tied to the SIP amount at all.
MIN_TICKET_SIZE  = 50.0
CRYPTO_DAYS_PER_YEAR = 365.0

BINANCE_TAKER_PCT = 0.00100
SLIPPAGE_PCT      = 0.00150
BUY_COST_PCT  = BINANCE_TAKER_PCT + SLIPPAGE_PCT
SELL_COST_PCT = BINANCE_TAKER_PCT + SLIPPAGE_PCT
ANNUAL_CASH_YIELD = 0.00   # deliberately conservative default, matching the source lineage

WFO_IS_PCT  = 0.70
WFO_OOS_PCT = 0.30

MIN_TRADES_GATE   = 20
MIN_MONTHS_GATE   = 24
MIN_WIN_RATE_GATE = 0.20   # a big reward:risk trend system is SUPPOSED to have a low win rate

ROBUSTNESS_DEPLOY_THRESHOLD = 0.50

RECENCY_WEIGHT_MIN = 0.80
RECENCY_WEIGHT_MAX = 1.00
BULL_YEAR_BTC_THRESHOLD = 0.10
BEAR_YEAR_BTC_THRESHOLD = -0.10
CONCENTRATION_GATE_HARD = 0.80    # HARD reject above this (see module docstring)
MIN_YEAR_COVERAGE_FOR_SCORING = 0.75
YEAR_FULL_COVERAGE_DAYS = 365.0

# BUG FIX (found from a real run): a W_IR=0.20 weight was defined here for a
# monthly Information-Ratio-vs-benchmark term that was never actually
# implemented -- the simulator never tracked monthly return buckets, so
# compute_score_portfolio's weighted sum silently only ever used 0.80 of its
# intended 1.00 weight mass. Rather than bolt on a whole new monthly-return
# subsystem for a metric the yearly CONSISTENCY and REGIME terms below
# already substantially cover (both compare portfolio IRR against the
# benchmark's own IRR, just at yearly instead of monthly granularity), W_IR
# is removed and the remaining weights are rescaled by 1/0.80 to preserve
# their original RELATIVE proportions while actually summing to 1.00.
W_CALMAR = 0.25
W_SORTINO = 0.125
W_EV = 0.125
W_WR_BONUS = 0.125
W_CONSISTENCY = 0.25
W_REGIME = 0.125
K_STD_PENALTY = 1.00
K_WORST_PENALTY = 0.50
W_BULL_CAPTURE = 0.50
W_BEAR_DEFENSE = 0.50

# -- capital-allocation policy when >1 watchlist candidate competes for
# the same limited cash on the same day -- applies uniformly to every
# candidate this file tests, since it's a money-management choice, not a
# signal-quality one. 0=deepest %% discount from trigger price, 1=closest
# to its own trend/entry level, 2=strongest momentum. --
WL_RANK_METHOD = 0

# -- temporal (random-SIP-day) robustness --
TEMPORAL_ROBUSTNESS_RUNS      = 15
TEMPORAL_ROBUSTNESS_SCORE_MIN = 0.70
TEMPORAL_ROBUSTNESS_PASS_FRAC = 0.70
SIP_RANDOM_MIN_DAY = 1
SIP_RANDOM_MAX_DAY = 28

MA_CACHE_LOCAL_NOTE = "MA/RSI caches are imported from crypto_signal_engine and shared as-is."


def _recency_weight(year, min_year, max_year):
    if max_year <= min_year:
        return RECENCY_WEIGHT_MAX
    frac = (year - min_year) / (max_year - min_year)
    return RECENCY_WEIGHT_MIN + (RECENCY_WEIGHT_MAX - RECENCY_WEIGHT_MIN) * frac


# ==========================================================================
# 1. SHORTLIST LOADER -- the interface contract with crypto_signal_engine.py
# ==========================================================================

def load_shortlist(filename=SHORTLIST_FILE):
    if not os.path.exists(filename):
        raise SystemExit(
            f"{filename} not found. Run crypto_signal_engine.py first -- "
            f"this script has nothing to test without its shortlist."
        )
    with open(filename, 'r') as f:
        data = json.load(f)
    candidates = data.get('candidates', [])
    universe   = data.get('universe', [])
    if not candidates:
        raise SystemExit(f"{filename} has no candidates -- nothing cleared the signal-quality gates.")
    print(f"Loaded {len(candidates)} candidates from {filename} "
          f"(signal engine version {data.get('engine_version', '?')}, "
          f"universe of {len(universe)} coins, "
          f"year range {data.get('year_range')}).")
    return candidates, universe


# ==========================================================================
# 2. DATA PREPARATION -- fetches EXACTLY the universe the shortlist was
# built on, so signal-stage and portfolio-stage results are apples-to-apples.
# ==========================================================================

def prepare_matrix_data(tickers, start_date, data_dir="data_crypto_portfolio"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    raw_dfs = {}
    master_dates = set()

    for ticker in tqdm(tickers, desc="Downloading Binance klines"):
        file_path = f"{data_dir}/{ticker}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            if 'QuoteVol' not in df.columns:
                df = _binance_download_klines(ticker, start_date)
                if df is not None and len(df) > 200:
                    df.to_csv(file_path)
        else:
            df = _binance_download_klines(ticker, start_date)
            if df is not None and len(df) > 200:
                df.to_csv(file_path)
        if df is not None and len(df) > 200:
            raw_dfs[ticker] = df
            master_dates.update(df.index)
        elif ticker == 'BTCUSDT':
            raise RuntimeError("Could not download BTCUSDT history.")

    master_dates = sorted(list(master_dates))
    master_df = pd.DataFrame(index=master_dates)
    master_df['Month'] = master_df.index.month
    master_df['Year']  = master_df.index.year

    n_days = len(master_dates)
    stock_names = ['BTCUSDT'] + [t for t in raw_dfs.keys() if t != 'BTCUSDT']
    n_stocks = len(stock_names)

    opens  = np.zeros((n_days, n_stocks))
    highs  = np.zeros((n_days, n_stocks))
    lows   = np.zeros((n_days, n_stocks))
    closes = np.zeros((n_days, n_stocks))
    atr_matrix = np.zeros((n_days, n_stocks))
    adx_matrix = np.zeros((n_days, n_stocks))
    quote_vol_matrix = np.full((n_days, n_stocks), np.nan)

    for i, ticker in enumerate(tqdm(stock_names, desc="Building matrix")):
        df = raw_dfs[ticker].reindex(master_dates)
        opens[:,  i] = df['Open'].ffill().values
        highs[:,  i] = df['High'].ffill().values
        lows[:,   i] = df['Low'].ffill().values
        closes[:, i] = df['Close'].ffill().values
        atr_matrix[:, i] = calc_atr_wilder(highs[:, i], lows[:, i], closes[:, i], period=14)
        adx_matrix[:, i] = calc_adx(highs[:, i], lows[:, i], closes[:, i], period=14)
        if 'QuoteVol' in df.columns:
            quote_vol_matrix[:, i] = df['QuoteVol'].values

    assert stock_names[0] == 'BTCUSDT'
    liquidity_mask = build_liquidity_mask(quote_vol_matrix)
    years_arr  = master_df['Year'].values.astype(np.int32)
    months_arr = master_df['Month'].values.astype(np.int32)

    return (opens, closes, atr_matrix, adx_matrix, months_arr, years_arr,
            stock_names, liquidity_mask, master_dates)


def build_sip_schedule(master_dates, mode='month_start', seed=None,
                        min_day=SIP_RANDOM_MIN_DAY, max_day=SIP_RANDOM_MAX_DAY):
    n = len(master_dates)
    flags = np.zeros(n, dtype=np.bool_)
    if mode == 'month_start':
        prev_key = None
        for i, d in enumerate(master_dates):
            key = (d.year, d.month)
            if key != prev_key:
                flags[i] = True
                prev_key = key
        return flags
    if mode != 'random':
        raise ValueError(f"unknown sip schedule mode {mode!r}")
    rng = np.random.default_rng(seed)
    by_month = {}
    for i, d in enumerate(master_dates):
        key = (d.year, d.month)
        by_month.setdefault(key, []).append(i)
    for key in sorted(by_month.keys()):
        idxs = by_month[key]
        target_day = int(rng.integers(min_day, max_day + 1))
        chosen = idxs[-1]
        for i in idxs:
            if master_dates[i].day >= target_day:
                chosen = i
                break
        flags[chosen] = True
    return flags


# ==========================================================================
# 3. PORTFOLIO SIMULATOR -- shared cash pool, monthly SIP, watchlist shadow
# positions, equal-weight sizing. Supports the FULL entry x exit taxonomy
# from crypto_signal_engine.py (3 entry families x 6 exit families x every
# toggle including latched-entry), generalized from a money-management
# pattern that originally only supported one hardcoded entry/exit family.
# ==========================================================================

@njit(nogil=True)
def simulate_portfolio_crypto(
        opens, closes, atr, adx, sip_flag,
        entry_ma, xover_short, xover_long,
        rsi_fast, rsi_slow, rsi_trend_ma,
        btc_close, btc_ma,
        exit_ma, exit_xover_short, exit_xover_long,
        exit_rsi_fast, exit_rsi_slow,
        eligible_mask,
        entry_type, exit_type,
        use_btc_entry_gate, use_btc_exit_override, use_rsi_trend_filter,
        use_latched_entry, adx_threshold,
        sl_mult, tp_mult, trail_mult, trail_pct, exit_atr_mult,
        max_pyramid_layers, wl_rank_method,
        start_day, end_day,
        starting_wealth, bench_starting_wealth,
        buy_cost_pct, sell_cost_pct, annual_cash_yield,
        monthly_sip, min_ticket_size):
    """
    monthly_sip / min_ticket_size are passed explicitly rather than read as
    bare module globals. numba freezes a referenced module-level global's
    value into the compiled function AT FIRST COMPILATION -- reassigning the
    Python-level constant afterward (e.g. while testing) silently has NO
    EFFECT on already-compiled code. That's not a correctness bug for a
    single production run (the constant is set once, before the function is
    ever called), but it's a real fragility: it masked a fix during testing
    and makes this function's behavior depend on *when* something happened
    to be set rather than what was passed in. Explicit parameters remove
    that trap entirely.
    """

    n_days, n_stocks = closes.shape
    if end_day < 0 or end_day >= n_days: end_day = n_days - 2
    if start_day < 1: start_day = 1
    if max_pyramid_layers < 1: max_pyramid_layers = 1

    daily_yield_mult = (1.0 + annual_cash_yield) ** (1.0 / 365.0)
    btc_col = closes[:, 0]

    cash_pool = starting_wealth
    total_invested_capital = 0.0

    in_pos           = np.zeros(n_stocks, dtype=np.bool_)
    n_layers         = np.zeros(n_stocks, dtype=np.int32)
    entry_prices     = np.zeros(n_stocks)
    entry_days       = np.zeros(n_stocks, dtype=np.int32)
    shares_held      = np.zeros(n_stocks)
    high_since_entry = np.zeros(n_stocks)
    stop_loss_price  = np.zeros(n_stocks)
    tp_trigger_price = np.zeros(n_stocks)
    half_sold        = np.zeros(n_stocks, dtype=np.bool_)
    entry_risk       = np.zeros(n_stocks)   # anchored to layer 1 only
    is_armed         = np.zeros(n_stocks, dtype=np.bool_)   # latched-entry state, real-position side

    wl_active        = np.zeros(n_stocks, dtype=np.bool_)
    wl_entry_price   = np.zeros(n_stocks)
    wl_is_pyramid    = np.zeros(n_stocks, dtype=np.bool_)
    wl_stop_loss     = np.zeros(n_stocks)
    wl_tp_trigger    = np.zeros(n_stocks)
    wl_half_sold     = np.zeros(n_stocks, dtype=np.bool_)
    wl_high_since    = np.zeros(n_stocks)
    wl_is_armed      = np.zeros(n_stocks, dtype=np.bool_)   # latched-entry state, watchlist side

    pending_exit    = np.zeros(n_stocks, dtype=np.bool_)
    pending_partial = np.zeros(n_stocks, dtype=np.bool_)

    total_wins = 0.0
    total_losses = 0.0
    winning_trades = 0
    losing_trades = 0
    win_pct_sum = 0.0
    loss_pct_sum = 0.0
    total_bars_in_trades = 0

    daily_port_val  = np.zeros(n_days)
    daily_bench_val = np.zeros(n_days)
    sip_injected        = np.zeros(n_days)
    bench_sip_injected  = np.zeros(n_days)
    daily_ret_p_arr = np.zeros(n_days)
    daily_ret_b_arr = np.zeros(n_days)

    MAX_CF = n_days + 8
    cf_days = np.zeros(MAX_CF, dtype=np.int32)
    cf_amounts = np.zeros(MAX_CF)
    cf_cnt = 0
    if starting_wealth > 0:
        cf_days[0] = start_day
        cf_amounts[0] = -starting_wealth
        cf_cnt = 1

    bench_cf_days = np.zeros(MAX_CF, dtype=np.int32)
    bench_cf_amounts = np.zeros(MAX_CF)
    bench_cf_cnt = 0
    if bench_starting_wealth > 0:
        bench_cf_days[0] = start_day
        bench_cf_amounts[0] = -bench_starting_wealth
        bench_cf_cnt = 1

    bench_shares = bench_starting_wealth / btc_col[start_day] if btc_col[start_day] > 0 else 0.0

    for d in range(start_day, end_day):
        cash_pool *= daily_yield_mult

        # ---- PHASE A: settle pending exits/partials at today's open ----
        for s in range(n_stocks):
            if pending_partial[s] and in_pos[s]:
                fill_price = opens[d, s]
                if not (fill_price > 0): fill_price = closes[d - 1, s]
                sell_shares = shares_held[s] * 0.5
                exit_val = sell_shares * fill_price * (1.0 - sell_cost_pct)
                invested_val = sell_shares * entry_prices[s]
                profit = exit_val - invested_val
                pct_change = profit / invested_val if invested_val > 0 else 0.0
                if profit > 0:
                    total_wins += profit; win_pct_sum += pct_change; winning_trades += 1
                else:
                    total_losses += abs(profit); loss_pct_sum += abs(pct_change); losing_trades += 1
                cash_pool += exit_val
                shares_held[s] -= sell_shares
                half_sold[s] = True
                if stop_loss_price[s] < entry_prices[s]:
                    stop_loss_price[s] = entry_prices[s]
                pending_partial[s] = False

            if pending_exit[s] and in_pos[s]:
                fill_price = opens[d, s]
                if not (fill_price > 0): fill_price = closes[d - 1, s]
                exit_val = shares_held[s] * fill_price * (1.0 - sell_cost_pct)
                invested_val = shares_held[s] * entry_prices[s]
                profit = exit_val - invested_val
                pct_change = profit / invested_val if invested_val > 0 else 0.0
                if profit > 0:
                    total_wins += profit; win_pct_sum += pct_change; winning_trades += 1
                else:
                    total_losses += abs(profit); loss_pct_sum += abs(pct_change); losing_trades += 1
                total_bars_in_trades += (d - 1 - entry_days[s])
                cash_pool += exit_val
                in_pos[s] = False
                n_layers[s] = 0
                shares_held[s] = 0.0
                pending_exit[s] = False
                half_sold[s] = False

        # ---- SIP injection ----
        if sip_flag[d]:
            if btc_col[d] > 0:
                bench_shares += monthly_sip / btc_col[d]
                bench_sip_injected[d] = monthly_sip
            if bench_cf_cnt < MAX_CF:
                bench_cf_days[bench_cf_cnt] = d
                bench_cf_amounts[bench_cf_cnt] = -monthly_sip
                bench_cf_cnt += 1
            cash_pool += monthly_sip
            total_invested_capital += monthly_sip
            sip_injected[d] = monthly_sip
            if cf_cnt < MAX_CF:
                cf_days[cf_cnt] = d
                cf_amounts[cf_cnt] = -monthly_sip
                cf_cnt += 1

        curr_closes = closes[d]
        prev_closes = closes[d - 1]
        btc_bullish = True
        if use_btc_entry_gate or use_btc_exit_override:
            btc_bullish = btc_close[d] > btc_ma[d]

        # ---- WATCHLIST INVALIDATION (shadow-position exit logic -- same
        # rule that would sell a REAL position, applied to unfunded ones) ----
        newly_invalidated = np.zeros(n_stocks, dtype=np.bool_)
        for s in range(n_stocks):
            if wl_active[s] and not wl_is_pyramid[s]:
                if curr_closes[s] > 0:
                    wl_high_since[s] = max(wl_high_since[s], curr_closes[s])

                should_invalidate = False
                should_partial = False

                if exit_type == 0:  # EXIT_HYBRID
                    if not wl_half_sold[s] and curr_closes[s] >= wl_tp_trigger[s]:
                        should_partial = True
                    if wl_half_sold[s]:
                        pot_sl = curr_closes[s] - atr[d, s] * trail_mult
                        if pot_sl > wl_stop_loss[s]: wl_stop_loss[s] = pot_sl
                    if curr_closes[s] < wl_stop_loss[s]:
                        should_invalidate = True
                elif exit_type == 1:  # EXIT_PCT_TRAIL
                    if curr_closes[s] < wl_high_since[s] * (1.0 - trail_pct / 100.0):
                        should_invalidate = True
                elif exit_type == 2:  # EXIT_ATR_TRAIL
                    if curr_closes[s] < wl_high_since[s] - (exit_atr_mult * atr[d, s]):
                        should_invalidate = True
                elif exit_type == 3:  # EXIT_MA_CROSSUNDER
                    if prev_closes[s] >= exit_ma[d - 1, s] and curr_closes[s] < exit_ma[d, s]:
                        should_invalidate = True
                elif exit_type == 4:  # EXIT_RSI_CROSSUNDER
                    if exit_rsi_fast[d - 1, s] >= exit_rsi_slow[d - 1, s] and exit_rsi_fast[d, s] < exit_rsi_slow[d, s]:
                        should_invalidate = True
                elif exit_type == 5:  # EXIT_MA_XOVER_EXIT
                    if exit_xover_short[d - 1, s] >= exit_xover_long[d - 1, s] and exit_xover_short[d, s] < exit_xover_long[d, s]:
                        should_invalidate = True

                if use_btc_exit_override and not btc_bullish:
                    should_invalidate = True
                    should_partial = False

                if should_partial and not should_invalidate:
                    wl_half_sold[s] = True
                    if wl_stop_loss[s] < wl_entry_price[s]:
                        wl_stop_loss[s] = wl_entry_price[s]

                if should_invalidate:
                    wl_active[s] = False
                    wl_is_pyramid[s] = False
                    wl_half_sold[s] = False
                    wl_is_armed[s] = False
                    newly_invalidated[s] = True

        # ---- PHASE B: flag exits for REAL positions ----
        for s in range(n_stocks):
            if in_pos[s] and not pending_exit[s]:
                if curr_closes[s] > 0:
                    high_since_entry[s] = max(high_since_entry[s], curr_closes[s])

                should_exit_full = False
                should_exit_partial = False

                if exit_type == 0:
                    if not half_sold[s] and curr_closes[s] >= tp_trigger_price[s]:
                        should_exit_partial = True
                    if half_sold[s]:
                        pot_sl = curr_closes[s] - atr[d, s] * trail_mult
                        if pot_sl > stop_loss_price[s]: stop_loss_price[s] = pot_sl
                    if curr_closes[s] < stop_loss_price[s]:
                        should_exit_full = True
                elif exit_type == 1:
                    if curr_closes[s] < high_since_entry[s] * (1.0 - trail_pct / 100.0):
                        should_exit_full = True
                elif exit_type == 2:
                    if curr_closes[s] < high_since_entry[s] - (exit_atr_mult * atr[d, s]):
                        should_exit_full = True
                elif exit_type == 3:
                    if prev_closes[s] >= exit_ma[d - 1, s] and curr_closes[s] < exit_ma[d, s]:
                        should_exit_full = True
                elif exit_type == 4:
                    if exit_rsi_fast[d - 1, s] >= exit_rsi_slow[d - 1, s] and exit_rsi_fast[d, s] < exit_rsi_slow[d, s]:
                        should_exit_full = True
                elif exit_type == 5:
                    if exit_xover_short[d - 1, s] >= exit_xover_long[d - 1, s] and exit_xover_short[d, s] < exit_xover_long[d, s]:
                        should_exit_full = True

                if use_btc_exit_override and not btc_bullish:
                    should_exit_full = True
                    should_exit_partial = False

                if should_exit_full:
                    pending_exit[s] = True
                elif should_exit_partial:
                    pending_partial[s] = True

        # ---- WATCHLIST ADDITIONS ----
        adx_ok_today = (adx_threshold <= 0.0)
        for s in range(n_stocks):
            if not wl_active[s] and not newly_invalidated[s] and eligible_mask[d, s]:
                trigger = False
                filter_ok = True
                if entry_type == 0:  # ENTRY_MA_BREAKOUT
                    if prev_closes[s] <= entry_ma[d - 1, s] and curr_closes[s] > entry_ma[d, s]:
                        trigger = True
                elif entry_type == 1:  # ENTRY_RSI_XOVER
                    if rsi_fast[d - 1, s] <= rsi_slow[d - 1, s] and rsi_fast[d, s] > rsi_slow[d, s]:
                        trigger = True
                        if use_rsi_trend_filter:
                            filter_ok = curr_closes[s] > rsi_trend_ma[d, s]
                elif entry_type == 2:  # ENTRY_MA_XOVER
                    if xover_short[d - 1, s] <= xover_long[d - 1, s] and xover_short[d, s] > xover_long[d, s]:
                        trigger = True

                if not trigger:
                    continue

                adx_ok = (adx_threshold <= 0.0) or (adx[d, s] >= adx_threshold)
                btc_ok = (not use_btc_entry_gate) or btc_bullish

                add_now = False
                if use_latched_entry:
                    add_now = True   # arm regardless of filters; re-checked at funding time
                else:
                    add_now = filter_ok and adx_ok and btc_ok

                if add_now:
                    wl_active[s] = True
                    wl_entry_price[s] = curr_closes[s]
                    wl_is_pyramid[s] = in_pos[s] and not pending_exit[s] and (n_layers[s] < max_pyramid_layers)
                    wl_is_armed[s] = use_latched_entry
                    if not wl_is_pyramid[s]:
                        wl_high_since[s] = opens[d, s] if d + 1 < n_days else curr_closes[s]
                        e_atr = atr[d, s] if atr[d, s] > 0 else atr[d - 1, s]
                        if not (e_atr > 0): e_atr = curr_closes[s] * 0.02
                        wl_stop_loss[s] = curr_closes[s] - e_atr * sl_mult
                        wl_tp_trigger[s] = curr_closes[s] + e_atr * tp_mult
                        wl_half_sold[s] = False

        # ---- CAPITAL DEPLOYMENT (equal-weight target sizing) ----
        n_eligible = 0
        for s in range(n_stocks):
            if eligible_mask[d, s]:
                n_eligible += 1
        if n_eligible < 1: n_eligible = 1

        while cash_pool >= min_ticket_size:
            best_rank = -999999.0
            best_s = -1
            for s in range(n_stocks):
                if not wl_active[s]:
                    continue
                if wl_is_pyramid[s] and not in_pos[s]:
                    wl_active[s] = False; wl_is_pyramid[s] = False
                    continue

                # latched candidates must clear their filters/BTC/ADX gate
                # NOW, at funding time, not just at watchlist-add time
                if wl_is_armed[s]:
                    ok = True
                    if entry_type == 1 and use_rsi_trend_filter:
                        ok = ok and (curr_closes[s] > rsi_trend_ma[d, s])
                    if use_btc_entry_gate:
                        ok = ok and btc_bullish
                    if adx_threshold > 0.0:
                        ok = ok and (adx[d, s] >= adx_threshold)
                    if not ok:
                        continue

                rank = -999.0
                if wl_rank_method == 0 and wl_entry_price[s] > 0:
                    rank = (wl_entry_price[s] - curr_closes[s]) / wl_entry_price[s]
                elif wl_rank_method == 1 and curr_closes[s] > 0:
                    rank = -abs(curr_closes[s] - wl_entry_price[s]) / curr_closes[s]
                elif wl_rank_method == 2 and wl_entry_price[s] > 0:
                    rank = (curr_closes[s] - wl_entry_price[s]) / wl_entry_price[s]
                if rank > best_rank:
                    best_rank = rank
                    best_s = s

            if best_s == -1:
                break

            buy_price = opens[d + 1, best_s] if d + 1 < n_days else curr_closes[best_s]
            if not (buy_price > 0):
                wl_active[best_s] = False
                continue

            invested_cost = 0.0
            for s2 in range(n_stocks):
                if in_pos[s2]:
                    invested_cost += shares_held[s2] * entry_prices[s2]
            target_size = (cash_pool + invested_cost) / n_eligible
            buy_amount = target_size if target_size <= cash_pool else cash_pool
            if buy_amount < min_ticket_size:
                break

            cash_pool -= buy_amount
            new_shares = buy_amount / (buy_price * (1.0 + buy_cost_pct))

            if wl_is_pyramid[best_s] and in_pos[best_s]:
                old_cost = shares_held[best_s] * entry_prices[best_s]
                new_cost = new_shares * buy_price
                shares_held[best_s] += new_shares
                entry_prices[best_s] = (old_cost + new_cost) / shares_held[best_s]
                n_layers[best_s] += 1
                e_atr = atr[d, best_s] if atr[d, best_s] > 0 else atr[d - 1, best_s]
                if not (e_atr > 0): e_atr = buy_price * 0.02
                pot_sl = entry_prices[best_s] - e_atr * sl_mult
                if pot_sl > stop_loss_price[best_s]: stop_loss_price[best_s] = pot_sl
            else:
                in_pos[best_s] = True
                n_layers[best_s] = 1
                entry_prices[best_s] = buy_price
                entry_days[best_s] = d
                shares_held[best_s] = new_shares
                high_since_entry[best_s] = buy_price
                half_sold[best_s] = False
                e_atr = atr[d, best_s] if atr[d, best_s] > 0 else atr[d - 1, best_s]
                if not (e_atr > 0): e_atr = buy_price * 0.02
                entry_risk[best_s] = e_atr * sl_mult
                stop_loss_price[best_s] = buy_price - e_atr * sl_mult
                tp_trigger_price[best_s] = buy_price + e_atr * tp_mult

            wl_active[best_s] = False
            wl_is_pyramid[best_s] = False
            wl_is_armed[best_s] = False

        # ---- DAILY VALUATION ----
        curr_val = cash_pool
        for s in range(n_stocks):
            if in_pos[s]:
                curr_val += shares_held[s] * curr_closes[s]

        if d > start_day:
            floor_val = daily_port_val[d - 1] * 0.001
            daily_port_val[d] = max(curr_val, floor_val)
        else:
            daily_port_val[d] = curr_val
        daily_bench_val[d] = bench_shares * btc_col[d]

        if d > start_day:
            prev_p = daily_port_val[d - 1]
            if prev_p > 0:
                daily_ret_p_arr[d] = (daily_port_val[d] - prev_p - sip_injected[d]) / prev_p
            prev_b = daily_bench_val[d - 1]
            if prev_b > 0:
                daily_ret_b_arr[d] = (daily_bench_val[d] - prev_b - bench_sip_injected[d]) / prev_b

    final_wealth = daily_port_val[end_day - 1]
    final_bench = daily_bench_val[end_day - 1]
    if cf_cnt < MAX_CF:
        cf_days[cf_cnt] = end_day - 1; cf_amounts[cf_cnt] = final_wealth; cf_cnt += 1
    if bench_cf_cnt < MAX_CF:
        bench_cf_days[bench_cf_cnt] = end_day - 1; bench_cf_amounts[bench_cf_cnt] = final_bench; bench_cf_cnt += 1

    trade_count = winning_trades + losing_trades
    avg_bars = total_bars_in_trades / trade_count if trade_count > 0 else 0.0
    avg_runup = win_pct_sum / winning_trades if winning_trades > 0 else 0.0
    avg_loss_r = loss_pct_sum / losing_trades if losing_trades > 0 else 0.0

    max_dd = 0.0
    peak = daily_port_val[start_day]
    returns_sum = 0.0; returns_sq_sum = 0.0; returns_cube_sum = 0.0; returns_quad_sum = 0.0
    downside_sq = 0.0; valid_days = 0

    for d in range(start_day + 1, end_day):
        v = daily_port_val[d]
        if v > peak:
            peak = v
        else:
            if peak > 0:
                dd = (v - peak) / peak
                if dd < max_dd: max_dd = dd
        prev = daily_port_val[d - 1]
        if prev > 0:
            ret = daily_ret_p_arr[d]
            returns_sum += ret; returns_sq_sum += ret * ret
            returns_cube_sum += ret ** 3; returns_quad_sum += ret ** 4
            valid_days += 1
            if ret < 0: downside_sq += ret * ret

    sharpe = 0.0; sortino = 0.0; skew = 0.0; kurt = 0.0
    if valid_days > 1:
        mean_ret = returns_sum / valid_days
        var_ret = (returns_sq_sum / valid_days) - mean_ret ** 2
        if var_ret > 0:
            std_ret = math.sqrt(var_ret)
            if std_ret > 0: sharpe = (mean_ret / std_ret) * math.sqrt(365.0)
            e_x2 = returns_sq_sum / valid_days
            e_x3 = returns_cube_sum / valid_days
            e_x4 = returns_quad_sum / valid_days
            m2 = var_ret
            m3 = e_x3 - 3.0 * mean_ret * e_x2 + 2.0 * mean_ret ** 3
            m4 = e_x4 - 4.0 * mean_ret * e_x3 + 6.0 * mean_ret ** 2 * e_x2 - 3.0 * mean_ret ** 4
            skew = m3 / (m2 ** 1.5)
            kurt = m4 / (m2 ** 2)
        std_down = math.sqrt(downside_sq / valid_days)
        if std_down > 0: sortino = (mean_ret / std_down) * math.sqrt(365.0)

    return (final_wealth, final_bench, total_invested_capital,
            total_wins, total_losses, trade_count, winning_trades, losing_trades,
            avg_bars, avg_runup, avg_loss_r, max_dd, sharpe, sortino,
            cf_days, cf_amounts, cf_cnt, bench_cf_days, bench_cf_amounts, bench_cf_cnt,
            daily_port_val, daily_bench_val, skew, kurt, valid_days)


# ==========================================================================
# 4. MONEY-WEIGHTED RETURN SOLVER (XIRR via bisection)
# ==========================================================================

def money_weighted_annual_return(cf_days, cf_amounts, day_basis=CRYPTO_DAYS_PER_YEAR,
                                  r_lo=-0.999, r_hi=10.0, tol=1e-9, max_iter=200):
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


# ==========================================================================
# 4b. DSR DIAGNOSTIC (same math as the signal engine's, generic)
# ==========================================================================

_EULER_GAMMA = 0.5772156649015329
_NORM = NormalDist()


def deflated_sharpe_ratio(sr_hat_daily, all_trial_sharpes_daily, T, skew, kurt):
    n_trials = len(all_trial_sharpes_daily)
    if n_trials < 2:
        denom = math.sqrt(max(1e-12, 1 - skew * sr_hat_daily + ((kurt - 1) / 4.0) * sr_hat_daily ** 2))
        z = sr_hat_daily * math.sqrt(max(T - 1, 1)) / denom
        return _NORM.cdf(z), 0.0, n_trials
    mean_sr = sum(all_trial_sharpes_daily) / n_trials
    var_sr = sum((s - mean_sr) ** 2 for s in all_trial_sharpes_daily) / n_trials
    sr_std = math.sqrt(var_sr)
    if n_trials <= 1 or sr_std <= 0:
        sr0 = 0.0
    else:
        z1 = _NORM.inv_cdf(1.0 - 1.0 / n_trials)
        z2 = _NORM.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
        sr0 = sr_std * ((1 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)
    denom = math.sqrt(max(1e-12, 1 - skew * sr_hat_daily + ((kurt - 1) / 4.0) * sr_hat_daily ** 2))
    z = (sr_hat_daily - sr0) * math.sqrt(max(T - 1, 1)) / denom
    return _NORM.cdf(z), sr0, n_trials


# ==========================================================================
# 5. YEARLY BREAKDOWN -- feeds recency weighting, consistency/regime terms,
# and the hard concentration gate.
# ==========================================================================

def compute_yearly_breakdown(daily_port_val, daily_bench_val, years_arr,
                              cf_days, cf_amounts, cf_cnt,
                              bench_cf_days, bench_cf_amounts, bench_cf_cnt,
                              start_day, end_day):
    day_years = years_arr[start_day:end_day]
    if len(day_years) == 0:
        return []
    unique_years = sorted(set(int(y) for y in day_years))
    cf_days_arr = np.array(cf_days[:cf_cnt]); cf_amts_arr = np.array(cf_amounts[:cf_cnt])
    bench_cf_days_arr = np.array(bench_cf_days[:bench_cf_cnt]); bench_cf_amts_arr = np.array(bench_cf_amounts[:bench_cf_cnt])

    out = []
    for yr in unique_years:
        yr_idxs = np.where(day_years == yr)[0] + start_day
        if len(yr_idxs) < 5:
            continue
        y_start, y_end = int(yr_idxs[0]), int(yr_idxs[-1])
        port_start = daily_port_val[max(y_start - 1, start_day)]
        port_end   = daily_port_val[y_end]
        bench_start = daily_bench_val[max(y_start - 1, start_day)]
        bench_end   = daily_bench_val[y_end]

        mask = (cf_days_arr >= y_start) & (cf_days_arr <= y_end)
        year_cf_days = list(cf_days_arr[mask]); year_cf_amts = list(cf_amts_arr[mask])
        port_cf_days = [y_start] + year_cf_days + [y_end]
        port_cf_amts = [-port_start] + year_cf_amts + [port_end]

        bmask = (bench_cf_days_arr >= y_start) & (bench_cf_days_arr <= y_end)
        year_bcf_days = list(bench_cf_days_arr[bmask]); year_bcf_amts = list(bench_cf_amts_arr[bmask])
        bench_cf_days_l = [y_start] + year_bcf_days + [y_end]
        bench_cf_amts_l = [-bench_start] + year_bcf_amts + [bench_end]

        port_irr, port_ok = money_weighted_annual_return(port_cf_days, port_cf_amts)
        bench_irr, bench_ok = money_weighted_annual_return(bench_cf_days_l, bench_cf_amts_l)

        nominal_contrib = sum(-a for a in year_cf_amts if a < 0)
        nominal_profit = (port_end - port_start) - nominal_contrib
        coverage_frac = min(1.0, len(yr_idxs) / YEAR_FULL_COVERAGE_DAYS)

        out.append({
            'year': yr, 'port_irr': port_irr if port_ok else 0.0,
            'bench_irr': bench_irr if bench_ok else 0.0,
            'nominal_profit': nominal_profit, 'coverage_frac': coverage_frac,
            'days': len(yr_idxs),
        })
    return out


# ==========================================================================
# 6. SCORING
# ==========================================================================

def compute_score_portfolio(metrics, yearly, is_oos=False):
    roi = metrics['roi']; sortino = metrics['sortino']
    max_dd = abs(metrics['max_dd']); trades = metrics['trades']
    avg_runup = metrics['avg_runup']; avg_loss = abs(metrics['avg_loss'])
    winning_trades = metrics['winning_trades']; losing_trades = metrics['losing_trades']
    annual_return = metrics['annual_return']

    target_trades = max(15, MIN_TRADES_GATE * (WFO_OOS_PCT / WFO_IS_PCT)) if is_oos else MIN_TRADES_GATE

    if trades < target_trades: return -999.0
    if max_dd > 0.60: return -999.0
    if metrics['pf'] < 1.05: return -999.0
    if roi <= 0: return -999.0
    if avg_loss <= 0: return -999.0
    if not math.isfinite(annual_return): return -999.0

    total_counted = winning_trades + losing_trades
    actual_wr = winning_trades / total_counted if total_counted > 0 else 0.0
    if actual_wr < MIN_WIN_RATE_GATE: return -999.0

    # HARD concentration gate -- deliberately stricter than the signal
    # engine's soft penalty. See module docstring for why.
    if yearly:
        total_profit = sum(y['nominal_profit'] for y in yearly)
        if total_profit > 0:
            max_share = max(y['nominal_profit'] / total_profit for y in yearly)
            if max_share > CONCENTRATION_GATE_HARD:
                return -999.0

    if max_dd < 0.001: max_dd = 0.001
    calmar = annual_return / max_dd
    ev = actual_wr * avg_runup - (1.0 - actual_wr) * avg_loss
    ev_in_r = ev / avg_loss
    wr_bonus = max(0.0, actual_wr - 0.50) * 4.0
    sortino_capped = min(sortino, 4.0)
    stat_conf = min(1.0, math.sqrt(trades / target_trades))

    if max_dd <= 0.20: dd_penalty = 1.0
    elif max_dd <= 0.35: dd_penalty = 1.0 - (max_dd - 0.20) * 2.0
    else: dd_penalty = 0.70 * math.exp(-2.0 * (max_dd - 0.35))

    scoring_years = [y for y in yearly if y['coverage_frac'] >= MIN_YEAR_COVERAGE_FOR_SCORING]

    consistency = 0.0
    if scoring_years:
        yrs = [y['year'] for y in scoring_years]
        min_year, max_year = min(yrs), max(yrs)
        w_list = [_recency_weight(y['year'], min_year, max_year) * y['coverage_frac'] for y in scoring_years]
        r_list = [y['port_irr'] for y in scoring_years]
        w_sum = sum(w_list)
        if w_sum > 0:
            mean_r = sum(w * r for w, r in zip(w_list, r_list)) / w_sum
            var_r = sum(w * (r - mean_r) ** 2 for w, r in zip(w_list, r_list)) / w_sum
            std_r = math.sqrt(max(0.0, var_r))
            worst_r = min(r_list)
            consistency = (mean_r - K_STD_PENALTY * std_r - K_WORST_PENALTY * max(0.0, -worst_r)) / max_dd

    regime = 0.0
    if scoring_years:
        bull_ratios, bull_w, bear_diffs, bear_w = [], [], [], []
        sy_min = min(y['year'] for y in scoring_years); sy_max = max(y['year'] for y in scoring_years)
        for y in scoring_years:
            w = _recency_weight(y['year'], sy_min, sy_max) * y['coverage_frac']
            if y['bench_irr'] > BULL_YEAR_BTC_THRESHOLD:
                ratio = y['port_irr'] / y['bench_irr'] if y['bench_irr'] != 0 else 0.0
                bull_ratios.append(max(-1.0, min(3.0, ratio))); bull_w.append(w)
            elif y['bench_irr'] < BEAR_YEAR_BTC_THRESHOLD:
                diff = (y['port_irr'] - y['bench_irr']) / max_dd
                bear_diffs.append(max(-2.0, min(2.0, diff))); bear_w.append(w)
        bull_capture = (sum(r * w for r, w in zip(bull_ratios, bull_w)) / sum(bull_w)) if bull_w else 0.0
        bear_defense = (sum(d * w for d, w in zip(bear_diffs, bear_w)) / sum(bear_w)) if bear_w else 0.0
        regime = W_BULL_CAPTURE * bull_capture + W_BEAR_DEFENSE * bear_defense

    score = (calmar * W_CALMAR + sortino_capped * W_SORTINO + ev_in_r * W_EV +
             wr_bonus * W_WR_BONUS + consistency * W_CONSISTENCY + regime * W_REGIME) * stat_conf * dd_penalty
    return score


def diagnose_gates_portfolio(metrics, yearly, is_oos=False):
    if not metrics:
        return [('has_any_invested_capital', False, 0,
                 '>0 -- evaluate_candidate_portfolio returned an empty metrics dict, meaning '
                 't_invested<=0 for this window (no SIP ever landed inside [start_day, end_day) '
                 '-- check the window boundaries and sip_flag, not the strategy parameters)')]
    trades = metrics.get('trades', 0)
    max_dd = abs(metrics.get('max_dd', 0))
    pf = metrics.get('pf', 0)
    roi = metrics.get('roi', 0)
    avg_loss = abs(metrics.get('avg_loss', 0))
    wt, lt = metrics.get('winning_trades', 0), metrics.get('losing_trades', 0)
    wr = wt / (wt + lt) if (wt + lt) > 0 else 0.0
    target_trades = max(15, MIN_TRADES_GATE * (WFO_OOS_PCT / WFO_IS_PCT)) if is_oos else MIN_TRADES_GATE
    rows = [
        ('trades >= target', trades >= target_trades, trades, f'>= {target_trades:.0f}'),
        ('max_dd <= 60%', max_dd <= 0.60, f'{max_dd*100:.1f}%', '<= 60.0%'),
        ('profit_factor >= 1.05', pf >= 1.05, f'{pf:.2f}', '>= 1.05'),
        ('roi > 0', roi > 0, f'{roi*100:.1f}%', '> 0%'),
        ('avg_loss > 0', avg_loss > 0, f'{avg_loss:.4f}', '> 0'),
        ('win_rate >= floor', wr >= MIN_WIN_RATE_GATE, f'{wr*100:.1f}%', f'>= {MIN_WIN_RATE_GATE*100:.0f}%'),
    ]
    if yearly:
        total_profit = sum(y['nominal_profit'] for y in yearly)
        if total_profit > 0:
            max_share = max(y['nominal_profit'] / total_profit for y in yearly)
            rows.append((f'profit_concentration <= {CONCENTRATION_GATE_HARD*100:.0f}%',
                         max_share <= CONCENTRATION_GATE_HARD, f'{max_share*100:.1f}%',
                         f'<= {CONCENTRATION_GATE_HARD*100:.0f}%'))
    return rows


def print_gate_diagnosis(metrics, yearly, is_oos=False, label="GATE DIAGNOSIS"):
    print(f"\n{'-'*60}\n{label}\n{'-'*60}")
    first_fail = False
    for name, passed, actual, threshold in diagnose_gates_portfolio(metrics, yearly, is_oos):
        marker = "PASS" if passed else "FAIL"
        print(f"  [{marker}] {name:<30} actual={actual!s:<10} needed {threshold}")
        if not passed and not first_fail:
            print("       ^-- this is the gate that produced the -999 score")
            first_fail = True
    print("-" * 60)


# ==========================================================================
# 7. EVALUATE A CANDIDATE (from the shortlist) UNDER REAL CAPITAL
# ==========================================================================

def _prep_indicator_arrays(p, closes, n_days, n_stocks):
    entry_type = p['entry_type']; exit_type = p['exit_type']
    zeros = np.zeros((n_days, n_stocks))

    entry_ma = xover_short = xover_long = zeros
    rsi_fast = rsi_slow = rsi_trend_ma = zeros

    if entry_type == ENTRY_MA_BREAKOUT:
        entry_ma = np.zeros((n_days, n_stocks))
        for s in range(n_stocks):
            entry_ma[:, s] = get_ma_cached(closes, s, p['entry_ma_len'], p['entry_ma_type'])
    elif entry_type == ENTRY_RSI_XOVER:
        rsi_fast = np.zeros((n_days, n_stocks)); rsi_slow = np.zeros((n_days, n_stocks))
        for s in range(n_stocks):
            rsi_fast[:, s] = get_smoothed_rsi(closes, s, p['rsi_f_len'], p['rsi_f_smt'])
            rsi_slow[:, s] = get_smoothed_rsi(closes, s, p['rsi_s_len'], p['rsi_s_smt'])
        if p.get('use_rsi_trend_filter'):
            rsi_trend_ma = np.zeros((n_days, n_stocks))
            for s in range(n_stocks):
                rsi_trend_ma[:, s] = get_ma_cached(closes, s, p['rsi_trend_ma_len'], p['rsi_trend_ma_type'])
    elif entry_type == ENTRY_MA_XOVER:
        xover_short = np.zeros((n_days, n_stocks)); xover_long = np.zeros((n_days, n_stocks))
        for s in range(n_stocks):
            xover_short[:, s] = get_ma_cached(closes, s, p['xover_short_len'], p['xover_short_type'])
            xover_long[:, s] = get_ma_cached(closes, s, p['xover_long_len'], p['xover_long_type'])

    exit_ma = exit_xover_short = exit_xover_long = zeros
    exit_rsi_fast = exit_rsi_slow = zeros
    if exit_type == EXIT_MA_CROSSUNDER:
        exit_ma = np.zeros((n_days, n_stocks))
        for s in range(n_stocks):
            exit_ma[:, s] = get_ma_cached(closes, s, p['exit_ma_len'], p['exit_ma_type'])
    elif exit_type == EXIT_RSI_CROSSUNDER:
        exit_rsi_fast = np.zeros((n_days, n_stocks)); exit_rsi_slow = np.zeros((n_days, n_stocks))
        for s in range(n_stocks):
            exit_rsi_fast[:, s] = get_smoothed_rsi(closes, s, p['exit_rsi_f_len'], p['exit_rsi_f_smt'])
            exit_rsi_slow[:, s] = get_smoothed_rsi(closes, s, p['exit_rsi_s_len'], p['exit_rsi_s_smt'])
    elif exit_type == EXIT_MA_XOVER_EXIT:
        exit_xover_short = np.zeros((n_days, n_stocks)); exit_xover_long = np.zeros((n_days, n_stocks))
        for s in range(n_stocks):
            exit_xover_short[:, s] = get_ma_cached(closes, s, p['exit_xover_short_len'], p['exit_xover_short_type'])
            exit_xover_long[:, s] = get_ma_cached(closes, s, p['exit_xover_long_len'], p['exit_xover_long_type'])

    if p.get('use_btc_entry_gate') or p.get('use_btc_exit_override'):
        btc_ma = get_ma_cached(closes, 0, p['btc_ma_len'], p['btc_ma_type'])
    else:
        btc_ma = np.zeros(n_days)

    return (entry_ma, xover_short, xover_long, rsi_fast, rsi_slow, rsi_trend_ma,
            btc_ma, exit_ma, exit_xover_short, exit_xover_long, exit_rsi_fast, exit_rsi_slow)


def evaluate_candidate_portfolio(p, opens, closes, atr, adx, years_arr, eligible_mask,
                                  sip_flag, start_day=0, end_day=-1, is_oos=False,
                                  starting_wealth=0.0, bench_starting_wealth=0.0):
    n_days, n_stocks = closes.shape
    if end_day < 0: end_day = n_days - 1

    (entry_ma, xover_short, xover_long, rsi_fast, rsi_slow, rsi_trend_ma, btc_ma,
     exit_ma, exit_xover_short, exit_xover_long, exit_rsi_fast, exit_rsi_slow
     ) = _prep_indicator_arrays(p, closes, n_days, n_stocks)

    result = simulate_portfolio_crypto(
        opens, closes, atr, adx, sip_flag,
        entry_ma, xover_short, xover_long, rsi_fast, rsi_slow, rsi_trend_ma,
        closes[:, 0], btc_ma,
        exit_ma, exit_xover_short, exit_xover_long, exit_rsi_fast, exit_rsi_slow,
        eligible_mask,
        int(p['entry_type']), int(p['exit_type']),
        bool(p['use_btc_entry_gate']), bool(p['use_btc_exit_override']), bool(p['use_rsi_trend_filter']),
        bool(p.get('use_latched_entry', False)), float(p.get('adx_thresh', 0.0)),
        float(p['sl_mult']), float(p['tp_mult']), float(p['trail_mult']), float(p['trail_pct']), float(p['exit_atr_mult']),
        int(p.get('max_pyramid_layers', 1)), int(WL_RANK_METHOD),
        int(start_day), int(end_day),
        float(starting_wealth), float(bench_starting_wealth),
        BUY_COST_PCT, SELL_COST_PCT, ANNUAL_CASH_YIELD,
        MONTHLY_SIP, MIN_TICKET_SIZE
    )

    (f_wealth, f_bench, t_invested, wins, losses, trades, winning_trades, losing_trades,
     avg_bars, avg_runup, avg_loss, max_dd, sharpe, sortino,
     cf_days, cf_amounts, cf_cnt, bench_cf_days, bench_cf_amounts, bench_cf_cnt,
     daily_port_val, daily_bench_val, skew, kurt, valid_days) = result

    if t_invested <= 0:
        return -999.0, {}

    capital_base = starting_wealth + t_invested
    roi = (f_wealth - capital_base) / capital_base if capital_base > 0 else 0.0
    bench_capital_base = sum(-a for a in bench_cf_amounts[:bench_cf_cnt] if a < 0)
    bench_roi = (f_bench - bench_capital_base) / bench_capital_base if bench_capital_base > 0 else 0.0
    pf = (wins / losses) if losses > 0 else (999.0 if wins > 0 else 0.0)

    cf_days_list = list(cf_days[:cf_cnt]); cf_amounts_list = list(cf_amounts[:cf_cnt])
    annual_return, converged = money_weighted_annual_return(cf_days_list, cf_amounts_list)
    years_elapsed = max(1.0 / 365.0, (int(end_day) - int(start_day)) / CRYPTO_DAYS_PER_YEAR)
    if not converged:
        annual_return = (f_wealth / capital_base) ** (1.0 / years_elapsed) - 1.0 if capital_base > 0 else -1.0

    yearly = compute_yearly_breakdown(daily_port_val, daily_bench_val, years_arr,
                                       cf_days, cf_amounts, cf_cnt,
                                       bench_cf_days, bench_cf_amounts, bench_cf_cnt,
                                       int(start_day), int(end_day))

    month_cnt = max(1, int(round((end_day - start_day) / 30.44)))

    metrics = {
        'roi': roi, 'bench_roi': bench_roi, 'alpha': roi - bench_roi,
        'pf': pf, 'wealth': f_wealth, 'bench_wealth': f_bench,
        'trades': trades, 'winning_trades': winning_trades, 'losing_trades': losing_trades,
        'sharpe': sharpe, 'sortino': sortino, 'max_dd': max_dd,
        'avg_bars': avg_bars, 'avg_runup': avg_runup, 'avg_loss': avg_loss,
        't_invested': t_invested, 'month_cnt': month_cnt,
        'starting_wealth': starting_wealth, 'bench_starting_wealth': bench_starting_wealth,
        'annual_return': annual_return, 'irr_converged': converged,
        'yearly': yearly, 'skew': float(skew), 'kurtosis': float(kurt), 'valid_days': int(valid_days),
    }
    score = compute_score_portfolio(metrics, yearly, is_oos=is_oos)
    return score, metrics


# ==========================================================================
# 8. WALK-FORWARD IS/OOS VALIDATION (wealth chained IS -> OOS)
# ==========================================================================

def run_wfo_validation(p, opens, closes, atr, adx, years_arr, eligible_mask, sip_flag):
    n_days = closes.shape[0]
    is_end = int(n_days * WFO_IS_PCT)

    is_score, is_m = evaluate_candidate_portfolio(p, opens, closes, atr, adx, years_arr, eligible_mask,
                                                   sip_flag, start_day=0, end_day=is_end, is_oos=False)
    is_end_wealth = is_m.get('wealth', 0.0) if is_score > -900 else 0.0
    is_end_bench = is_m.get('bench_wealth', 0.0) if is_score > -900 else 0.0

    oos_score, oos_m = evaluate_candidate_portfolio(p, opens, closes, atr, adx, years_arr, eligible_mask,
                                                     sip_flag, start_day=is_end, end_day=n_days - 1, is_oos=True,
                                                     starting_wealth=is_end_wealth, bench_starting_wealth=is_end_bench)

    if oos_score <= -900:
        robustness = -1.0
    elif is_score > 0:
        robustness = oos_score / is_score if oos_score > 0 else oos_score / abs(is_score)
    else:
        robustness = 0.0

    return is_score, is_m, oos_score, oos_m, robustness, is_end_wealth, is_end_bench


def run_temporal_robustness_check(p, opens, closes, atr, adx, years_arr, eligible_mask, master_dates,
                                   baseline_score, n_runs=TEMPORAL_ROBUSTNESS_RUNS):
    if baseline_score <= 0:
        return True, 0.0, []
    n_days = closes.shape[0]
    scores = []
    for seed in range(n_runs):
        sip_flag = build_sip_schedule(master_dates, mode='random', seed=seed)
        s, _ = evaluate_candidate_portfolio(p, opens, closes, atr, adx, years_arr, eligible_mask,
                                             sip_flag, start_day=0, end_day=n_days - 1, is_oos=False)
        scores.append(s)
    scores_arr = np.array(scores)
    frac_holding = float(np.mean(scores_arr >= baseline_score * TEMPORAL_ROBUSTNESS_SCORE_MIN))
    median_score = float(np.median(scores_arr))
    return frac_holding >= TEMPORAL_ROBUSTNESS_PASS_FRAC, median_score, scores


# ==========================================================================
# 9. SAVE / REPORTING
# ==========================================================================

def save_champion(oos_score, is_score, robustness, temporal_median, p, tier, filename=CHAMPION_FILE):
    clean = {k: (float(v) if isinstance(v, (float, np.floating)) else
                 int(v) if isinstance(v, (int, np.integer)) and not isinstance(v, bool) else
                 bool(v) if isinstance(v, (bool, np.bool_)) else v)
             for k, v in p.items()}
    data = {
        'engine_version': ENGINE_VERSION,
        'oos_score': float(oos_score), 'is_score': float(is_score),
        'robustness_ratio': float(robustness), 'temporal_median_score': float(temporal_median),
        'robustness_tier': tier, 'params': clean,
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def print_candidate_report(rank, p, is_score, is_m, oos_score, oos_m, robustness,
                            temporal_ok, temporal_median, neighbor_note=""):
    from crypto_signal_engine import ENTRY_TYPE_NAMES, EXIT_TYPE_NAMES
    print("\n" + "=" * 78)
    print(f"CANDIDATE #{rank}  {neighbor_note}")
    print("=" * 78)
    print(f"Entry: {ENTRY_TYPE_NAMES.get(p['entry_type'],'?')}   Exit: {EXIT_TYPE_NAMES.get(p['exit_type'],'?')}   "
          f"Latched: {p.get('use_latched_entry', False)}   Pyramid layers: {p.get('max_pyramid_layers',1)}")
    print(f"In-Sample  Score: {is_score:.4f}" + (f"  ROI: {is_m.get('roi',0)*100:.1f}%  IRR: {is_m.get('annual_return',0)*100:.1f}%  "
          f"Sharpe: {is_m.get('sharpe',0):.2f}  MaxDD: {abs(is_m.get('max_dd',0))*100:.1f}%  Trades: {is_m.get('trades',0)}"
          if is_score > -900 else "  (failed IS gates)"))
    if oos_score > -900:
        print(f"Out-of-Sample Score: {oos_score:.4f}  ROI: {oos_m.get('roi',0)*100:.1f}%  IRR: {oos_m.get('annual_return',0)*100:.1f}%  "
              f"Sharpe: {oos_m.get('sharpe',0):.2f}  MaxDD: {abs(oos_m.get('max_dd',0))*100:.1f}%  Trades: {oos_m.get('trades',0)}")
    else:
        print(f"Out-of-Sample Score: {oos_score:.4f}  (failed OOS gates)")
    rob_str = "N/A" if robustness <= -1.0 else f"{robustness:.1%}"
    print(f"Robustness Ratio: {rob_str}   Temporal robustness: {'PASS' if temporal_ok else 'FAIL'} "
          f"(median retained score {temporal_median:.4f})")
    print("=" * 78)


# ==========================================================================
# 10. MAIN
# ==========================================================================

def run_portfolio_validation():
    candidates, universe = load_shortlist()
    candidates = candidates[:MAX_CANDIDATES_TO_TEST]

    if not universe:
        raise SystemExit("Shortlist has no universe list -- cannot fetch matching data.")

    print(f"\nFetching data for the shortlist's own {len(universe)}-coin universe "
          f"(so signal-stage and portfolio-stage results are apples-to-apples)...")
    (opens, closes, atr, adx, months_arr, years_arr,
     stock_names, eligible_mask, master_dates) = prepare_matrix_data(universe, "2018-01-01")

    clear_caches()
    n_days = closes.shape[0]
    sip_flag_default = build_sip_schedule(master_dates, mode='month_start')

    print(f"\nMatrix: {n_days} days x {closes.shape[1]} coins")
    print(f"Testing {len(candidates)} shortlisted candidates under a real ${MONTHLY_SIP:,.0f}/month "
          f"shared cash pool, equal-weight sizing, and Binance-realistic costs...\n")

    results = []
    for cand in candidates:
        p = cand['params']
        rank_in = cand.get('rank', '?')
        is_score, is_m, oos_score, oos_m, robustness, is_end_wealth, is_end_bench = run_wfo_validation(
            p, opens, closes, atr, adx, years_arr, eligible_mask, sip_flag_default)

        temporal_ok, temporal_median = True, 0.0
        if is_score > 0:
            temporal_ok, temporal_median, _ = run_temporal_robustness_check(
                p, opens, closes, atr, adx, years_arr, eligible_mask, master_dates, is_score)

        print_candidate_report(rank_in, p, is_score, is_m, oos_score, oos_m, robustness,
                                temporal_ok, temporal_median,
                                neighbor_note=f"(signal-engine rank #{rank_in}, signal IS score {cand.get('is_score', 0):.3f})")

        # BUG FIX (found from a real run): a candidate that fails used to print
        # only a bare "-999.0000 (failed IS gates)" with no further detail --
        # exactly the situation where detail matters most, and especially bad
        # when EVERY candidate fails at once (a systemic problem, not one bad
        # config), since there was previously no way to tell which gate tripped
        # without editing the script. Failures now get the full itemized
        # breakdown automatically; passing candidates stay concise since
        # print_candidate_report already covers them.
        if is_score <= -900:
            print_gate_diagnosis(is_m, is_m.get('yearly', []), is_oos=False,
                                 label=f"Candidate #{rank_in} -- why IS failed")
        elif oos_score <= -900:
            print_gate_diagnosis(oos_m, oos_m.get('yearly', []), is_oos=True,
                                 label=f"Candidate #{rank_in} -- passed IS, why OOS failed")

        results.append({
            'signal_rank': rank_in, 'params': p, 'is_score': is_score, 'oos_score': oos_score,
            'robustness': robustness, 'temporal_ok': temporal_ok, 'temporal_median': temporal_median,
            'is_metrics': is_m, 'oos_metrics': oos_m,
        })

    # rank: prefer candidates that cleared OOS gates AND temporal robustness,
    # then by OOS score -- exactly the point of this whole file: a candidate
    # that ranked #1 on signal quality alone might not win here, and a
    # candidate that ranked lower on signal quality might, if it happens to
    # fit the realities of shared capital better. Nothing here overwrites
    # or hides the signal-engine ranking -- both are reported.
    def sort_key(r):
        cleared = r['oos_score'] > -900 and r['temporal_ok']
        return (cleared, r['oos_score'] if r['oos_score'] > -900 else -9999)
    results.sort(key=sort_key, reverse=True)

    print("\n" + "=" * 78)
    print("FINAL RANKING -- portfolio-validated, not signal-only")
    print("=" * 78)
    print(f"{'Rank':<6}{'Signal Rank':<13}{'OOS Score':<12}{'Robustness':<12}{'Temporal':<10}{'Entry/Exit'}")
    from crypto_signal_engine import ENTRY_TYPE_NAMES, EXIT_TYPE_NAMES
    for i, r in enumerate(results, start=1):
        p = r['params']
        rob = "N/A" if r['robustness'] <= -1.0 else f"{r['robustness']*100:.0f}%"
        oos_str = f"{r['oos_score']:.3f}" if r['oos_score'] > -900 else "GATE FAIL"
        print(f"{i:<6}{r['signal_rank']:<13}{oos_str:<12}{rob:<12}{'PASS' if r['temporal_ok'] else 'FAIL':<10}"
              f"{ENTRY_TYPE_NAMES.get(p['entry_type'],'?')} / {EXIT_TYPE_NAMES.get(p['exit_type'],'?')}")

    if not results:
        print("\nNo candidates were tested.")
        return

    winner = results[0]
    if winner['oos_score'] > -900 and winner['robustness'] >= 0.70 and winner['temporal_ok']:
        tier = "EXCELLENT (>70% robustness, temporal-stable) -- deploy with confidence"
    elif winner['oos_score'] > -900 and winner['robustness'] >= ROBUSTNESS_DEPLOY_THRESHOLD and winner['temporal_ok']:
        tier = f"ACCEPTABLE ({ROBUSTNESS_DEPLOY_THRESHOLD:.0%}-70%) -- deploy cautiously"
    elif winner['oos_score'] > -900 and winner['robustness'] >= 0.30:
        tier = "CAUTION (30%-50% or failed temporal check) -- meaningful risk; consider smaller size"
    elif winner['oos_score'] > -900:
        tier = "POOR (<30%) -- high risk of not holding up; treat as a starting point, not a system"
    else:
        tier = "OOS GATE FAILURE -- even the best shortlisted candidate didn't hold up under real capital constraints"

    save_champion(winner['oos_score'], winner['is_score'], winner['robustness'],
                  winner['temporal_median'], winner['params'], tier)
    print(f"\n{'='*78}\nWINNER -- ROBUSTNESS TIER: {tier}\n{'='*78}")
    print(f"Saved to {CHAMPION_FILE}. This was signal-engine rank #{winner['signal_rank']} of "
          f"{len(candidates)} tested -- ", end="")
    if winner['signal_rank'] != 1:
        print("NOT the top-ranked signal. Exactly the scenario this two-file split exists to catch: "
              "a stronger standalone signal that doesn't fit the realities of shared capital as well "
              "as one that ranked lower on signal quality alone.")
    else:
        print("also the top-ranked signal -- capital constraints didn't change the verdict this time.")


if __name__ == "__main__":
    run_portfolio_validation()