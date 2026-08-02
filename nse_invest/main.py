"""
Walk-Forward Validated MA-Crossover Tranche Strategy Search
=============================================================

WHY THIS LOOKS DIFFERENT FROM THE PREVIOUS SAMPLES
----------------------------------------------------
Every earlier version (samples 5-9, "previous winner") ran ONE Monte Carlo
search against the ENTIRE history, then used "robustness" checks (neighbor MA
perturbation, multi-year win-rate, kill switches) computed over that SAME full
history. None of that is out-of-sample testing -- it's in-sample selection
with guardrails. With tens of thousands of random draws against ~11 years of
monthly data, finding a config that clears even a strict-looking bar (IR>0.5,
60% vintage win rate, survives ±2-day MA perturbation) is close to GUARANTEED
by chance alone, regardless of whether real edge exists.

This version fixes that with ANCHORED EXPANDING WALK-FORWARD VALIDATION:

  Fold 1: fit on [start .. year Y]   -> freeze params -> test, untouched, on year Y+1
  Fold 2: fit on [start .. year Y+1] -> freeze params -> test, untouched, on year Y+2
  Fold 3: fit on [start .. year Y+2] -> freeze params -> test, untouched, on year Y+3
  ... and so on through the present.

Every year of your data gets used for fitting AND gets exactly one genuine,
never-peeked-at forward test. Nothing is wasted, and nothing is leaked.
The parameters are also allowed to be DIFFERENT in every fold -- if they swing
wildly from fold to fold, that instability is itself diagnostic: it tells you
the search is fitting noise in the training window rather than finding a
stable edge. The per-fold table at the end shows you exactly this.

DELIBERATE SIMPLIFICATIONS VS SAMPLE 9
----------------------------------------
RSI mode, volume-spike mode, and DEMA/WMA variants were dropped. This is not
an oversight -- it's a direct response to the "don't waste a small dataset"
concern. Walk-forward means the EARLIEST fold trains on as little as ~5 years
of data. Every extra free parameter (sample 9 has ~16) eats further into that
already-thin training window's ability to distinguish signal from noise. The
parameter set kept here (MA pair, MA type, trend filter, exit mode, ranking
mode, trailing stop %, structural dip %) is closer to sample 8 -- fewer knobs,
each fold's in-sample fit more trustworthy.

WHAT THIS DOES NOT DO
-----------------------
It does not implement a Deflated Sharpe Ratio / Probability-of-Backtest-
Overfitting correction. That requires careful bookkeeping of the FULL trial
distribution per fold and is easy to get subtly wrong in a way that produces
false confidence. What this script gives you instead, per fold: how many
random draws were attempted, and how many cleared the in-sample consistency
gate at all -- a cheap, honest "how hard did we have to search" signal. If you
want the full deflated-Sharpe treatment later, bolt it on using the per-fold
trial counts already being tracked below.

WALK-FORWARD IS NOT A LARGE-SAMPLE TEST EITHER
------------------------------------------------
With ~11 years of NIFTY data and a 5-year initial training window, you get on
the order of 6-8 out-of-sample fold results. That is NOT a lot of independent
observations -- treat the aggregate OOS stats as directional, not conclusive.
This script will print that caveat explicitly rather than let a clean-looking
number imply more confidence than the sample size supports.
"""

import os
import warnings
import numpy as np
import pandas as pd
from numba import njit

try:
    from tqdm import tqdm
except ImportError:  # tqdm is a convenience, not a requirement
    def tqdm(iterable, **kwargs):
        return iterable

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
NIFTY_UNIVERSE = [
    # --- GIANTS & GROWTH (Focus: Momentum/Capital Appreciation) ---
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 
    'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'BAJFINANCE.NS', 'KOTAKBANK.NS', 'LT.NS', 
    'HCLTECH.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'MARUTI.NS', 'TITAN.NS', 'SUNPHARMA.NS', 
    'BAJAJFINSV.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'NESTLEIND.NS', 'TECHM.NS', 
    'ADANIENT.NS', 'ADANIPORTS.NS', 'TATAMOTORS.NS', 'GRASIM.NS', 
    'DIVISLAB.NS', 'SBILIFE.NS', 'DRREDDY.NS', 'CIPLA.NS', 'BRITANNIA.NS', 
    'EICHERMOT.NS', 'INDUSINDBK.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'TATACONSUM.NS', 
    'UPL.NS', 'APOLLOHOSP.NS', 'PIDILITIND.NS', 
    
    # --- MIDCAP / NEXT 50 GROWTH ---
    'GODREJCP.NS', 'DABUR.NS', 'SHREECEM.NS', 'DLF.NS', 'HAVELLS.NS', 'SRF.NS', 
    'ICICIPRULI.NS', 'JINDALSTEL.NS', 'AMBUJACEM.NS', 'CHOLAFIN.NS', 'BERGEPAINT.NS', 
    'BANKBARODA.NS', 'CANBK.NS', 'BEL.NS', 'SIEMENS.NS', 'BOSCHLTD.NS', 'MCDOWELL-N.NS', 
    'MARICO.NS', 'TORNTPHARM.NS', 'PIIND.NS', 'NAUKRI.NS', 'HAL.NS', 'TRENT.NS', 
    'TVSMOTOR.NS', 'ZOMATO.NS', 'VBL.NS', 'ABB.NS', 'PAGEIND.NS', 'COLPAL.NS',
    
    # --- HIGH MOMENTUM MIDCAPS ---
    'POLYCAB.NS', 'DIXON.NS', 'PERSISTENT.NS', 'LTIM.NS', 'KPITTECH.NS', 'COFORGE.NS',
    'ASTRAL.NS', 'BALKRISIND.NS', 'FEDERALBNK.NS', 'IDFCFIRSTB.NS', 'ASHOKLEY.NS',
    'CUMMINSIND.NS', 'OBEROIRLTY.NS', 'ESCORTS.NS', 'JUBLFOOD.NS', 'MRF.NS', 'MUTHOOTFIN.NS',
    'PETRONET.NS', 'SHRIRAMFIN.NS', 'TATACHEM.NS', 'TATAPOWER.NS',
    'VOLTAS.NS', 'ZEEL.NS', 'AUROPHARMA.NS', 'LUPIN.NS', 'ALKEM.NS', 'BHEL.NS',
    'BSE.NS', 'ANGELONE.NS', 'MCX.NS'
]

# NOTE on survivorship bias: this list is CURRENT index membership backtested
# to 2015. Stocks that earned index inclusion because they ran hard (Trent,
# Zomato, Dixon, BEL...) are present from day one, years before any real-time
# investor could have known to include them. This inflates every backtest
# below versus what was achievable in real time. If you build a point-in-time
# constituent history from NSE's index fact sheets, swap it in here.
# Similarly: delisted/suspended names are simply absent. NSE bhavcopy archives
# (daily full-market CSVs) contain every symbol that traded that day,
# including delisted ones -- that's the fix for missing yfinance coverage,
# not manual entry.

HIGH_DIVIDEND_STOCKS = [
    'ITC.NS', 'COALINDIA.NS', 'ONGC.NS', 'POWERGRID.NS', 'NTPC.NS',
    'VEDL.NS', 'PFC.NS', 'RECLTD.NS', 'BPCL.NS', 'IOC.NS', 'GAIL.NS', 'SAIL.NS'
]

START_DATE = "2015-01-01"
MONTHLY_BUDGET = 8000.0
TRANCHE_SIZE = 8000.0           # Fixed-size tranches -- deliberately NOT a % of NAV.
                                 # A %-of-NAV sizing scheme lets late-stage bets balloon
                                 # in absolute size as the account compounds, concentrating
                                 # the terminal result in a handful of late, oversized bets.
                                 # Fixed tranches keep "edge per trade" the thing being tested.
FRICTION_RATE = 0.003            # 0.3% flat slippage + statutory tax friction

# --- Walk-forward configuration ---
INITIAL_TRAIN_YEARS = 5          # First fold trains on this many years before any OOS test
ITERATIONS_PER_FOLD = 60000        # Random draws searched WITHIN each fold's training window
NEIGHBOR_THRESHOLD = 0.80         # In-sample neighbor-stability requirement (unchanged concept,
                                   # just now applied per-fold instead of once globally)
MIN_YEARS_ACTIVE = 3              # Minimum invested years required within a fold's training
                                   # window before a config is even considered
WIN_RATE_THRESHOLD = 0.55         # Fraction of in-sample years a config must beat the index in.
                                   # (Earlier samples drifted this from 0.60 down to 0.45 without
                                   # justification when a riskier sizing scheme was introduced --
                                   # don't do that. If you change this, write down why.)

if not os.path.exists("data"):
    os.makedirs("data")

# ==============================================================================
# 2. INDICATOR KERNELS
# ==============================================================================
@njit(fastmath=True, cache=True)
def calc_sma_2d(prices, period):
    num_days, num_stocks = prices.shape
    sma = np.full((num_days, num_stocks), np.nan, dtype=np.float64)
    if num_days < period:
        return sma
    for s in range(num_stocks):
        start_idx = -1
        for i in range(num_days):
            if not np.isnan(prices[i, s]):
                start_idx = i
                break
        if start_idx == -1 or num_days - start_idx < period:
            continue
        w_sum = 0.0
        for i in range(start_idx, start_idx + period):
            w_sum += prices[i, s]
        sma[start_idx + period - 1, s] = w_sum / period
        for i in range(start_idx + period, num_days):
            if np.isnan(prices[i, s]):
                sma[i, s] = sma[i - 1, s]
            else:
                w_sum = w_sum - prices[i - period, s] + prices[i, s]
                sma[i, s] = w_sum / period
    return sma


@njit(fastmath=True, cache=True)
def calc_ema_2d(prices, period):
    num_days, num_stocks = prices.shape
    ema = np.full((num_days, num_stocks), np.nan, dtype=np.float64)
    if num_days < period:
        return ema
    mult = 2.0 / (period + 1.0)
    for s in range(num_stocks):
        start_idx = -1
        for i in range(num_days):
            if not np.isnan(prices[i, s]):
                start_idx = i
                break
        if start_idx == -1 or num_days - start_idx < period:
            continue
        w_sum = 0.0
        for i in range(start_idx, start_idx + period):
            w_sum += prices[i, s]
        ema[start_idx + period - 1, s] = w_sum / period
        for i in range(start_idx + period, num_days):
            if np.isnan(prices[i, s]):
                ema[i, s] = ema[i - 1, s]
            else:
                ema[i, s] = (prices[i, s] - ema[i - 1, s]) * mult + ema[i - 1, s]
    return ema


def get_ma(arr, ma_type, period):
    """ma_type: 0 = SMA, 1 = EMA"""
    return calc_ema_2d(arr, period) if ma_type else calc_sma_2d(arr, period)


# ==============================================================================
# 3. PORTFOLIO ENGINE (fixed-tranche, dividend-aware exits, annual vintage tracking)
# ==============================================================================
@njit(fastmath=True, cache=True)
def evaluate_portfolio(closes, opens, index_closes, index_opens, short_ma, long_ma, super_ma,
                        years, months, filter_enabled, tsl_pct, struct_dip_pct, exit_mode,
                        ranking_mode, max_lookback, div_mask):

    num_days, num_stocks = closes.shape
    port_cash = 0.0
    shares = np.zeros(num_stocks, dtype=np.float64)
    peak_prices = np.zeros(num_stocks, dtype=np.float64)

    wishlist_active = np.zeros(num_stocks, dtype=np.bool_)
    wishlist_target = np.zeros(num_stocks, dtype=np.float64)

    bench_cash = 0.0
    bench_shares = 0.0

    port_year_start_val = np.zeros(30, dtype=np.float64)
    bench_year_start_val = np.zeros(30, dtype=np.float64)
    capital_injected_yr = np.zeros(30, dtype=np.float64)

    port_year_returns = np.zeros(30, dtype=np.float64)
    bench_year_returns = np.zeros(30, dtype=np.float64)

    # super_ma is ALWAYS computed (not gated by filter_enabled) so the structural-dip
    # exit remains a live risk control even when the trend filter is toggled off for
    # entries -- entry filtering and exit risk-management are decoupled on purpose.
    start_day = max_lookback + 2 if max_lookback < num_days else 260

    for i in range(start_day, num_days - 1):
        curr_month, prev_month = months[i], months[i - 1]
        curr_year, prev_year = years[i], years[i - 1]
        yr_idx = curr_year - years[0]
        if yr_idx < 0 or yr_idx >= 30:
            continue

        # --- ANNUAL ACCOUNTING SNAPSHOTS ---
        if curr_year != prev_year:
            p_val = port_cash
            for s in range(num_stocks):
                p_val += shares[s] * (closes[i - 1, s] if not np.isnan(closes[i - 1, s]) else 0.0)
            port_year_start_val[yr_idx] = p_val
            bench_year_start_val[yr_idx] = bench_cash + (bench_shares * index_closes[i - 1])

            prev_yr_idx = yr_idx - 1
            if prev_yr_idx >= 0 and port_year_start_val[prev_yr_idx] > 0:
                inj = capital_injected_yr[prev_yr_idx]
                p_base = port_year_start_val[prev_yr_idx] + (inj * 0.5)
                b_base = bench_year_start_val[prev_yr_idx] + (inj * 0.5)
                port_year_returns[prev_yr_idx] = (p_val - port_year_start_val[prev_yr_idx] - inj) / p_base if p_base > 0 else 0.0
                bench_year_returns[prev_yr_idx] = (bench_year_start_val[yr_idx] - bench_year_start_val[prev_yr_idx] - inj) / b_base if b_base > 0 else 0.0

        # --- MONTHLY BUDGET INJECTIONS ---
        if curr_month != prev_month:
            port_cash += MONTHLY_BUDGET
            bench_cash += MONTHLY_BUDGET
            capital_injected_yr[yr_idx] += MONTHLY_BUDGET

            b_price = index_opens[i] if not np.isnan(index_opens[i]) else index_closes[i]
            if b_price > 0:
                bench_shares += (bench_cash * (1.0 - FRICTION_RATE)) / b_price
                bench_cash = 0.0

        # --- BIFURCATED EXITS (dividend-mask aware) ---
        for s in range(num_stocks):
            if shares[s] > 0 and not np.isnan(closes[i - 1, s]):
                if closes[i - 1, s] > peak_prices[s]:
                    peak_prices[s] = closes[i - 1, s]

                sell_signal = False
                is_crashing_structurally = (not np.isnan(super_ma[i - 1, s])) and (closes[i - 1, s] < (super_ma[i - 1, s] * (1.0 - struct_dip_pct)))
                trailing_stop_hit = closes[i - 1, s] < (peak_prices[s] * (1.0 - tsl_pct))
                crossunder = (short_ma[i - 2, s] >= long_ma[i - 2, s]) and (short_ma[i - 1, s] < long_ma[i - 1, s])

                if div_mask[s]:
                    # High-dividend names: hold through routine MA crossunders, only
                    # exit on a genuine structural breakdown.
                    if is_crashing_structurally:
                        sell_signal = True
                else:
                    if exit_mode == 0:
                        sell_signal = crossunder
                    elif exit_mode == 1:
                        sell_signal = trailing_stop_hit or is_crashing_structurally
                    elif exit_mode == 2:
                        sell_signal = crossunder or trailing_stop_hit or is_crashing_structurally

                if sell_signal:
                    sell_price = opens[i, s] if not np.isnan(opens[i, s]) else closes[i, s]
                    if sell_price > 0:
                        port_cash += (shares[s] * sell_price) * (1.0 - FRICTION_RATE)
                        shares[s] = 0.0
                        peak_prices[s] = 0.0
                        wishlist_active[s] = False

        # --- SIGNAL MONITORING ---
        for s in range(num_stocks):
            if np.isnan(closes[i - 1, s]) or np.isnan(short_ma[i - 1, s]) or np.isnan(long_ma[i - 1, s]):
                continue
            if short_ma[i - 2, s] >= long_ma[i - 2, s] and short_ma[i - 1, s] < long_ma[i - 1, s]:
                wishlist_active[s] = False
            if short_ma[i - 2, s] <= long_ma[i - 2, s] and short_ma[i - 1, s] > long_ma[i - 1, s]:
                trend_ok = True
                if filter_enabled and (not np.isnan(super_ma[i - 1, s]) and closes[i - 1, s] <= super_ma[i - 1, s]):
                    trend_ok = False
                if trend_ok:
                    wishlist_active[s] = True
                    wishlist_target[s] = closes[i - 1, s]

        # --- FIXED-TRANCHE DEPLOYMENT ---
        while port_cash >= TRANCHE_SIZE:
            best_score = 999999.0 if ranking_mode == 2 else -999999.0
            best_stock = -1

            for s in range(num_stocks):
                if wishlist_active[s]:
                    buy_price = opens[i, s] if not np.isnan(opens[i, s]) else closes[i, s]
                    if buy_price <= 0 or np.isnan(buy_price):
                        continue
                    if buy_price > wishlist_target[s]:
                        continue

                    score = 0.0
                    if ranking_mode == 0:
                        score = (closes[i - 1, s] - closes[i - 21, s]) / closes[i - 21, s] if closes[i - 21, s] > 0 else -999.0
                    elif ranking_mode == 1:
                        score = (wishlist_target[s] - buy_price) / wishlist_target[s] if wishlist_target[s] > 0 else -999.0
                    elif ranking_mode == 2:
                        score = (buy_price - long_ma[i - 1, s]) / long_ma[i - 1, s] if long_ma[i - 1, s] > 0 else 999.0

                    if ranking_mode == 2:
                        if score < best_score:
                            best_score = score
                            best_stock = s
                    else:
                        if score > best_score:
                            best_score = score
                            best_stock = s

            if best_stock != -1:
                final_buy_price = opens[i, best_stock] if not np.isnan(opens[i, best_stock]) else closes[i, best_stock]
                shares[best_stock] += (TRANCHE_SIZE * (1.0 - FRICTION_RATE)) / final_buy_price
                if peak_prices[best_stock] == 0.0 or final_buy_price > peak_prices[best_stock]:
                    peak_prices[best_stock] = final_buy_price
                port_cash -= TRANCHE_SIZE
                wishlist_active[best_stock] = False
            else:
                break

    # --- FINAL-YEAR FLUSH ---
    # The annual-return finalization above only fires when the loop observes the FIRST
    # day of the *following* year. That's fine for any year in the middle of the data,
    # but breaks for the LAST year present in this array -- which is exactly the case
    # for every walk-forward test slice (deliberately built to end on the test year's
    # last trading day) and also for the most recent/incomplete year in a full-history
    # run. Without this flush, that final year's return silently stays at its zero-
    # initialized default, which reads as a phantom LOSS in the win/loss check below
    # regardless of what actually happened.
    final_yr_idx = years[num_days - 1] - years[0]
    if 0 <= final_yr_idx < 30 and capital_injected_yr[final_yr_idx] > 0:
        p_val = port_cash
        for s in range(num_stocks):
            p_val += shares[s] * (closes[num_days - 1, s] if not np.isnan(closes[num_days - 1, s]) else 0.0)
        b_val = bench_cash + (bench_shares * index_closes[num_days - 1])

        inj = capital_injected_yr[final_yr_idx]
        p_base = port_year_start_val[final_yr_idx] + (inj * 0.5)
        b_base = bench_year_start_val[final_yr_idx] + (inj * 0.5)
        if p_base > 0:
            port_year_returns[final_yr_idx] = (p_val - port_year_start_val[final_yr_idx] - inj) / p_base
        if b_base > 0:
            bench_year_returns[final_yr_idx] = (b_val - bench_year_start_val[final_yr_idx] - inj) / b_base

    # --- TERMINAL METRICS ---
    final_p_val = port_cash
    for s in range(num_stocks):
        final_p_val += shares[s] * (closes[-1, s] if not np.isnan(closes[-1, s]) else 0.0)
    final_b_val = bench_cash + (bench_shares * index_closes[-1])
    terminal_alpha = ((final_p_val - final_b_val) / final_b_val) if final_b_val > 0 else 0.0

    total_weight = 0.0
    win_score = 0.0
    years_active = 0
    vintage_wins = 0

    for y in range(30):
        if capital_injected_yr[y] > 0:
            years_active += 1
            w = float(years_active)
            total_weight += w
            if port_year_returns[y] > bench_year_returns[y]:
                win_score += w
                vintage_wins += 1

    weighted_win_rate = (win_score / total_weight) if total_weight > 0 else 0.0
    consistency_pass = (years_active >= MIN_YEARS_ACTIVE) and (float(vintage_wins) >= (float(years_active) * WIN_RATE_THRESHOLD))

    composite_score = (0.75 * weighted_win_rate) + (0.25 * terminal_alpha)

    return (composite_score, consistency_pass, final_p_val, final_b_val, weighted_win_rate,
            terminal_alpha, port_year_returns, bench_year_returns, capital_injected_yr)


# ==============================================================================
# 4. DATA LOADER (live data -- requires yfinance + network access)
# ==============================================================================
def load_data():
    import yfinance as yf  # imported lazily so the rest of this module is testable offline

    print("--- Bulk Downloading Market Matrices ---")
    raw_data = yf.download(NIFTY_UNIVERSE, start=START_DATE, progress=True)
    closes_df = raw_data['Close'].ffill()
    opens_df = raw_data['Open'].ffill()

    valid_tickers = [c for c in closes_df.columns if closes_df[c].notna().sum() > 400]

    closes = np.ascontiguousarray(closes_df[valid_tickers].values, dtype=np.float64)
    opens = np.ascontiguousarray(opens_df[valid_tickers].values, dtype=np.float64)

    nifty_data = yf.download("^NSEI", start=START_DATE, progress=False)
    index_closes = np.ascontiguousarray(nifty_data['Close'].values.flatten(), dtype=np.float64)
    index_opens = np.ascontiguousarray(nifty_data['Open'].values.flatten(), dtype=np.float64)

    years = np.ascontiguousarray(closes_df.index.year.values, dtype=np.int32)
    months = np.ascontiguousarray(closes_df.index.month.values, dtype=np.int32)

    div_mask = np.zeros(len(valid_tickers), dtype=np.bool_)
    for idx, t in enumerate(valid_tickers):
        if t in HIGH_DIVIDEND_STOCKS:
            div_mask[idx] = True

    return closes, opens, index_closes, index_opens, years, months, div_mask


# ==============================================================================
# 5. PER-FOLD IN-SAMPLE SEARCH
#    (operates ONLY on whatever slice of data it's handed -- the walk-forward
#     driver controls what that slice contains, this function never sees
#     anything outside it)
# ==============================================================================
def search_best_params(closes, opens, index_closes, index_opens, years, months, div_mask,
                        num_iterations, quiet=True):
    best_score = -999.0
    best_params = None
    best_metrics = None
    attempts_passed_gate = 0

    iterator = range(num_iterations) if quiet else tqdm(range(num_iterations), desc="  searching", leave=False)

    for _ in iterator:
        s_ma = int(np.random.randint(10, 80))
        l_ma = int(np.random.randint(50, 160))
        if s_ma >= l_ma:
            continue

        t_s = int(np.random.randint(0, 2))
        t_l = int(np.random.randint(0, 2))
        t_sl = int(np.random.randint(0, 2))

        filter_enabled = bool(np.random.choice([True, False]))
        sup_ma = int(np.random.randint(180, 260)) if filter_enabled else 200

        exit_mode = int(np.random.randint(0, 3))
        ranking_mode = int(np.random.randint(0, 3))
        tsl_pct = float(np.random.uniform(0.10, 0.35))
        struct_dip_pct = float(np.random.uniform(0.04, 0.20))

        max_lookback = max(l_ma, sup_ma)
        # Need enough history beyond burn-in to actually evaluate anything meaningful.
        # This guard matters a lot more here than in a full-history search, because
        # early walk-forward folds may only have ~5 years of data to begin with.
        if max_lookback + 250 >= closes.shape[0]:
            continue

        arr_s = get_ma(closes, t_s, s_ma)
        arr_l = get_ma(closes, t_l, l_ma)
        arr_sup = get_ma(closes, t_sl, sup_ma)

        result = evaluate_portfolio(
            closes, opens, index_closes, index_opens, arr_s, arr_l, arr_sup,
            years, months, filter_enabled, tsl_pct, struct_dip_pct, exit_mode,
            ranking_mode, max_lookback, div_mask
        )
        score, consistency, p_val, b_val, win_rate, alpha = result[:6]

        if consistency:
            attempts_passed_gate += 1

        if consistency and score > best_score:
            neighbor_passed = True
            for offset in (-2, 2):
                n_s, n_l = s_ma + offset, l_ma + offset
                if n_s >= n_l or n_s < 5 or n_l > 175:
                    continue
                n_arr_s = get_ma(closes, t_s, n_s)
                n_arr_l = get_ma(closes, t_l, n_l)
                n_result = evaluate_portfolio(
                    closes, opens, index_closes, index_opens, n_arr_s, n_arr_l, arr_sup,
                    years, months, filter_enabled, tsl_pct, struct_dip_pct, exit_mode,
                    ranking_mode, max_lookback, div_mask
                )
                n_score, n_consistency = n_result[0], n_result[1]
                if not n_consistency or n_score < (score * NEIGHBOR_THRESHOLD):
                    neighbor_passed = False
                    break

            if neighbor_passed:
                best_score = score
                best_params = {
                    's_ma': s_ma, 'l_ma': l_ma, 'sup_ma': sup_ma,
                    't_s': t_s, 't_l': t_l, 't_sl': t_sl,
                    'filter_enabled': filter_enabled, 'exit_mode': exit_mode,
                    'ranking_mode': ranking_mode, 'tsl_pct': tsl_pct,
                    'struct_dip_pct': struct_dip_pct
                }
                best_metrics = {'port_val': p_val, 'bench_val': b_val, 'win_rate': win_rate, 'alpha': alpha}

    if best_params is None:
        return None

    return {
        'params': best_params,
        'score': best_score,
        'metrics': best_metrics,
        'trials_run': num_iterations,
        'trials_passed_gate': attempts_passed_gate,
    }


# ==============================================================================
# 6. ANCHORED EXPANDING WALK-FORWARD DRIVER
# ==============================================================================
def run_walkforward(closes, opens, index_closes, index_opens, years, months, div_mask,
                     initial_train_years=INITIAL_TRAIN_YEARS,
                     iterations_per_fold=ITERATIONS_PER_FOLD):

    min_year = int(years[0])
    max_year = int(years[-1])

    fold_results = []
    test_year = min_year + initial_train_years

    while test_year <= max_year:
        train_end_year = test_year - 1

        train_idx_candidates = np.where(years <= train_end_year)[0]
        if len(train_idx_candidates) == 0:
            test_year += 1
            continue
        train_end_idx = train_idx_candidates[-1]

        test_idx_candidates = np.where(years <= test_year)[0]
        test_end_idx = test_idx_candidates[-1]

        # --- FIT: search using ONLY data through the end of train_end_year ---
        c_train = np.ascontiguousarray(closes[:train_end_idx + 1])
        o_train = np.ascontiguousarray(opens[:train_end_idx + 1])
        ic_train = np.ascontiguousarray(index_closes[:train_end_idx + 1])
        io_train = np.ascontiguousarray(index_opens[:train_end_idx + 1])
        y_train = np.ascontiguousarray(years[:train_end_idx + 1])
        m_train = np.ascontiguousarray(months[:train_end_idx + 1])

        best = search_best_params(c_train, o_train, ic_train, io_train, y_train, m_train,
                                   div_mask, iterations_per_fold)

        if best is None:
            print(f"[{test_year}] No config cleared the in-sample gate training on "
                  f"{min_year}-{train_end_year}. Skipping this fold.")
            test_year += 1
            continue

        # --- TEST: freeze those params, extend the window through test_year, ---
        # --- and read off ONLY that one new year's isolated return.          ---
        c_ext = np.ascontiguousarray(closes[:test_end_idx + 1])
        o_ext = np.ascontiguousarray(opens[:test_end_idx + 1])
        ic_ext = np.ascontiguousarray(index_closes[:test_end_idx + 1])
        io_ext = np.ascontiguousarray(index_opens[:test_end_idx + 1])
        y_ext = np.ascontiguousarray(years[:test_end_idx + 1])
        m_ext = np.ascontiguousarray(months[:test_end_idx + 1])

        p = best['params']
        arr_s = get_ma(c_ext, p['t_s'], p['s_ma'])
        arr_l = get_ma(c_ext, p['t_l'], p['l_ma'])
        arr_sup = get_ma(c_ext, p['t_sl'], p['sup_ma'])
        max_lookback = max(p['l_ma'], p['sup_ma'])

        result = evaluate_portfolio(
            c_ext, o_ext, ic_ext, io_ext, arr_s, arr_l, arr_sup,
            y_ext, m_ext, p['filter_enabled'], p['tsl_pct'], p['struct_dip_pct'],
            p['exit_mode'], p['ranking_mode'], max_lookback, div_mask
        )
        port_yr_ret, bench_yr_ret = result[6], result[7]

        yr_idx = test_year - min_year
        oos_port = port_yr_ret[yr_idx]
        oos_bench = bench_yr_ret[yr_idx]
        oos_win = oos_port > oos_bench

        fold_results.append({
            'test_year': test_year,
            'train_window': f"{min_year}-{train_end_year}",
            'params': p,
            'in_sample_score': best['score'],
            'in_sample_win_rate': best['metrics']['win_rate'],
            'trials_run': best['trials_run'],
            'trials_passed_gate': best['trials_passed_gate'],
            'oos_port_return': oos_port,
            'oos_bench_return': oos_bench,
            'oos_win': oos_win,
        })

        t_str = lambda t: "EMA" if t else "SMA"
        print(f"[Fold -> {test_year}] fit on {min_year}-{train_end_year} "
              f"({best['trials_passed_gate']}/{best['trials_run']} draws passed gate) "
              f"| {t_str(p['t_s'])}{p['s_ma']}/{t_str(p['t_l'])}{p['l_ma']} "
              f"| OOS: strategy {oos_port*100:+.1f}% vs index {oos_bench*100:+.1f}% "
              f"-> {'WIN' if oos_win else 'LOSS'}")

        test_year += 1

    return fold_results


# ==============================================================================
# 7. REPORTING
# ==============================================================================
def summarize_folds(fold_results):
    print("\n" + "=" * 78)
    print("WALK-FORWARD OUT-OF-SAMPLE SUMMARY")
    print("=" * 78)

    if not fold_results:
        print("No folds produced a valid out-of-sample test. Nothing to report.")
        return

    n = len(fold_results)
    wins = sum(1 for f in fold_results if f['oos_win'])
    excess = np.array([f['oos_port_return'] - f['oos_bench_return'] for f in fold_results])

    print(f"\n{'Test Year':<10} | {'Train Window':<12} | {'Short/Long MA':<16} | "
          f"{'OOS Strategy':<13} | {'OOS Index':<10} | {'Result':<6}")
    print("-" * 78)
    for f in fold_results:
        p = f['params']
        t_str = lambda t: "EMA" if t else "SMA"
        ma_str = f"{t_str(p['t_s'])}{p['s_ma']}/{t_str(p['t_l'])}{p['l_ma']}"
        print(f"{f['test_year']:<10} | {f['train_window']:<12} | {ma_str:<16} | "
              f"{f['oos_port_return']*100:>+11.1f}% | {f['oos_bench_return']*100:>+8.1f}% | "
              f"{'WIN' if f['oos_win'] else 'LOSS'}")

    print("-" * 78)
    print(f"Out-of-sample fold win rate : {wins}/{n} ({wins/n*100:.0f}%)")
    print(f"Mean OOS excess return      : {np.mean(excess)*100:+.2f}%")
    print(f"Std dev of OOS excess       : {np.std(excess)*100:.2f}%")
    if np.std(excess) > 0:
        print(f"Pseudo risk-adjusted edge   : {np.mean(excess) / np.std(excess):.2f}  "
              f"(mean / std of {n} annual excess returns -- NOT a proper Sharpe, just a ratio)")

    print("\nPARAMETER STABILITY ACROSS FOLDS (read this before trusting the win rate above):")
    short_mas = [f['params']['s_ma'] for f in fold_results]
    long_mas = [f['params']['l_ma'] for f in fold_results]
    print(f"  Short MA picked per fold: {short_mas}")
    print(f"  Long MA picked per fold : {long_mas}")
    if n >= 3 and (np.std(short_mas) > 15 or np.std(long_mas) > 25):
        print("  -> These swing a lot fold to fold. That's a sign the search is finding")
        print("     different noise in each training window rather than a stable edge.")
    else:
        print("  -> Relatively stable across folds -- a mildly encouraging sign, though")
        print(f"     with only {n} folds this is not strong statistical evidence either way.")

    print(f"\nCAVEAT: {n} out-of-sample observations is a small sample. Treat the win rate")
    print("and excess-return numbers above as directional, not conclusive. More years of")
    print("history (or shortening the initial training window, at the cost of noisier")
    print("early folds) are the only ways to get more OOS observations from this dataset.")


# ==============================================================================
# 8. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    closes, opens, index_closes, index_opens, years, months, div_mask = load_data()
    fold_results = run_walkforward(closes, opens, index_closes, index_opens, years, months, div_mask)
    summarize_folds(fold_results)