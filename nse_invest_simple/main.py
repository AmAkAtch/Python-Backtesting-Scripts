import yfinance as yf
import pandas as pd
import numpy as np
import os
import requests
import io
import math
import json
from numba import njit
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. UNIVERSE DEFINITION & CLASSIFICATION
# ==========================================
DIVIDEND_KINGS = {
    'ITC.NS', 'COALINDIA.NS', 'ONGC.NS', 'POWERGRID.NS', 'NTPC.NS', 'PFC.NS', 
    'RECLTD.NS', 'VEDL.NS', 'GAIL.NS', 'BPCL.NS', 'IOC.NS', 'PETRONET.NS', 
    'SAIL.NS', 'NHPC.NS', 'NMDC.NS', 'HINDZINC.NS', 'CASTROLIND.NS'
}

def fetch_dynamic_universe():
    print("Fetching broad Nifty universe (Nifty 50, Next 50, Midcap 150)...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    urls = [
        "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftynext50list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv"
    ]
    tickers = set()
    for url in urls:
        try:
            req = requests.get(url, headers=headers, timeout=10)
            if req.status_code == 200:
                df = pd.read_csv(io.StringIO(req.text))
                for symbol in df['Symbol']:
                    tickers.add(f"{symbol}.NS")
        except: pass
        
    if len(tickers) < 100:
        print("Fallback to predefined comprehensive list...")
        tickers = set(['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'ITC.NS'] + list(DIVIDEND_KINGS))
    return list(tickers)

# ==========================================
# 2. CONFIGURATIONS & SAVE SYSTEM
# ==========================================
START_DATE = "2015-01-01"
NUM_ITERATIONS = 100000   
COST_PCT = 0.0020  
NEIGHBOR_THRESHOLD = 0.85  
BEST_PARAMS_FILE = "best_params.json"

def load_previous_winner():
    if os.path.exists(BEST_PARAMS_FILE):
        try:
            with open(BEST_PARAMS_FILE, 'r') as f:
                data = json.load(f)
                return data.get('score', -999999), data.get('params', None)
        except Exception: pass
    return -999999, None

def save_winner(score, params):
    clean_params = {k: float(v) if isinstance(v, float) else int(v) for k, v in params.items()}
    data = {'score': float(score), 'params': clean_params}
    try:
        with open(BEST_PARAMS_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e: print(f"Error saving to JSON: {e}")

# ==========================================
# 3. FAST INDICATORS
# ==========================================
@njit(fastmath=True, cache=True)
def calc_ma(prices, period, ma_type):
    n = len(prices)
    res = np.empty(n); res[:] = np.nan
    if n < period: return res
    
    if ma_type == 0: # SMA
        w_sum = np.sum(prices[:period])
        res[period-1] = w_sum / period
        for i in range(period, n):
            w_sum = w_sum - prices[i-period] + prices[i]
            res[i] = w_sum / period
            
    elif ma_type == 1 or ma_type == 2: # EMA / DEMA
        ema1 = np.empty(n); ema1[:] = np.nan
        ema1[period-1] = np.mean(prices[:period])
        mult = 2 / (period + 1)
        for i in range(period, n):
            ema1[i] = (prices[i] - ema1[i-1]) * mult + ema1[i-1]
            
        if ma_type == 1: res = ema1
        else: # DEMA
            ema2 = np.empty(n); ema2[:] = np.nan
            ema2[period*2-2] = np.mean(ema1[period-1:period*2-1])
            for i in range(period*2-1, n):
                ema2[i] = (ema1[i] - ema2[i-1]) * mult + ema2[i-1]
            for i in range(period*2-2, n): res[i] = 2 * ema1[i] - ema2[i]
                
    elif ma_type == 3: # WMA
        weights = np.arange(1, period + 1)
        w_sum = np.sum(weights)
        for i in range(period - 1, n):
            res[i] = np.sum(prices[i - period + 1 : i + 1] * weights) / w_sum
            
    return res

# ==========================================
# 4. CHRONOLOGICAL PORTFOLIO SIMULATOR
# ==========================================
@njit(fastmath=True)
def simulate_portfolio(opens, closes, atr, months, index_closes, is_div_stock,
                       short_ma, long_ma, super_ma, wl_rank_method, 
                       entry_filter, n_exit_method, n_trail_pct, n_atr_mult,
                       div_exit_method, div_exit_val):
    
    n_days, n_stocks = closes.shape
    
    # Capital Pools
    monthly_allowance = 0.0      
    accumulated_capital = 0.0    
    total_invested_capital = 0.0 
    
    # State tracking
    in_pos = np.zeros(n_stocks, dtype=np.bool_)
    entry_prices = np.zeros(n_stocks)
    entry_days = np.zeros(n_stocks, dtype=np.int32)
    shares_held = np.zeros(n_stocks)
    high_since_entry = np.zeros(n_stocks)
    days_below_long = np.zeros(n_stocks)
    
    wl_active = np.zeros(n_stocks, dtype=np.bool_)
    wl_entry_prices = np.zeros(n_stocks)
    
    # Performance metrics
    total_wins = 0.0
    total_losses = 0.0
    winning_trades = 0
    losing_trades = 0
    win_pct_sum = 0.0
    loss_pct_sum = 0.0
    total_bars_in_trades = 0
    
    daily_port_val = np.zeros(n_days)
    daily_bench_val = np.zeros(n_days)
    
    port_monthly_returns = np.zeros(150) 
    bench_monthly_returns = np.zeros(150)
    month_cnt = 0
    
    last_port_month_end = 0.0
    last_bench_month_end = 0.0
    bench_shares = 0.0
    
    for d in range(1, n_days - 1):
        # 0. MONTHLY REFRESH
        if months[d] != months[d-1]:
            if d > 20:
                p_current = daily_port_val[d-1]
                b_current = bench_shares * index_closes[d-1] if bench_shares > 0 else 0.0
                
                if last_port_month_end > 0:
                    port_monthly_returns[month_cnt] = (p_current - last_port_month_end) / last_port_month_end
                    bench_monthly_returns[month_cnt] = (b_current - last_bench_month_end) / last_bench_month_end
                    month_cnt += 1
                
                last_port_month_end = p_current + 8000.0
                last_bench_month_end = b_current + 8000.0
            else:
                last_port_month_end = 8000.0
                last_bench_month_end = 8000.0
                
            monthly_allowance = 8000.0 
            total_invested_capital += 8000.0 
            if index_closes[d] > 0: bench_shares += 8000.0 / index_closes[d]

        curr_opens = opens[d+1] 
        curr_closes = closes[d] 
        
        # 1. PROCESS EXITS & RELEASE SETTLED CASH
        for s in range(n_stocks):
            if in_pos[s]:
                high_since_entry[s] = max(high_since_entry[s], curr_closes[s])
                should_exit = False
                
                if is_div_stock[s]:
                    # Dividend Stock Logic remains the same
                    if div_exit_method == 0: 
                        if curr_closes[s] < high_since_entry[s] * (1.0 - div_exit_val / 100.0): should_exit = True
                    elif div_exit_method == 1: 
                        if curr_closes[s] < super_ma[d, s] * (1.0 - div_exit_val / 100.0): should_exit = True
                    elif div_exit_method == 2: 
                        if curr_closes[s] < long_ma[d, s]: days_below_long[s] += 1
                        else: days_below_long[s] = 0
                        if days_below_long[s] > div_exit_val: should_exit = True
                else:
                    # NEW Normal Stock Exit Logic
                    if n_exit_method == 0: # Standard Crossunder
                        if short_ma[d-1, s] >= long_ma[d-1, s] and short_ma[d, s] < long_ma[d, s]:
                            should_exit = True
                    elif n_exit_method == 1: # Fixed Percentage Trailing Stop
                        if curr_closes[s] < high_since_entry[s] * (1.0 - n_trail_pct / 100.0):
                            should_exit = True
                    elif n_exit_method == 2: # Volatility (ATR) Trailing Stop
                        if curr_closes[s] < high_since_entry[s] - (n_atr_mult * atr[d, s]):
                            should_exit = True
                        
                if should_exit:
                    exit_val = shares_held[s] * curr_opens[s] * (1 - COST_PCT)
                    invested_val = shares_held[s] * entry_prices[s]
                    profit = exit_val - invested_val
                    
                    pct_change = 0.0
                    if invested_val > 0: pct_change = profit / invested_val
                    
                    if profit > 0: 
                        total_wins += profit
                        win_pct_sum += pct_change
                        winning_trades += 1
                    else: 
                        total_losses += abs(profit)
                        loss_pct_sum += pct_change
                        losing_trades += 1
                        
                    total_bars_in_trades += (d - entry_days[s])
                    accumulated_capital += exit_val 
                    in_pos[s] = False
                    shares_held[s] = 0

            # Watchlist Invalidations
            if wl_active[s]:
                if curr_closes[s] > wl_entry_prices[s] or (short_ma[d-1, s] >= long_ma[d-1, s] and short_ma[d, s] < long_ma[d, s]):
                    wl_active[s] = False

        # 2. NEW ENTRIES TO WATCHLIST (WITH REGIME FILTER)
        for s in range(n_stocks):
            if not in_pos[s] and not wl_active[s]:
                # Did a crossover occur?
                if short_ma[d-1, s] <= long_ma[d-1, s] and short_ma[d, s] > long_ma[d, s]:
                    valid_entry = True
                    
                    # Apply Regime/Trend Filters
                    if entry_filter == 1: # Must be above Super MA (Momentum)
                        if curr_closes[s] <= super_ma[d, s]: valid_entry = False
                    elif entry_filter == 2: # Must be below Super MA (Mean Reversion / Value)
                        if curr_closes[s] >= super_ma[d, s]: valid_entry = False
                        
                    if valid_entry:
                        wl_active[s] = True
                        wl_entry_prices[s] = curr_opens[s] 

        # 3. REAL-TIME SEQUENTIAL BUY LOGIC (₹8,000 Chunks)
        while True:
            has_sip_chunk = (monthly_allowance >= 8000.0)
            has_settled_chunk = (accumulated_capital >= 8000.0)
            
            if not (has_sip_chunk or has_settled_chunk):
                break 
                
            best_score = -999999.0
            best_s = -1
            for s in range(n_stocks):
                if wl_active[s]:
                    score = -999.0
                    if wl_rank_method == 0 and wl_entry_prices[s] > 0: 
                        score = (wl_entry_prices[s] - curr_closes[s]) / wl_entry_prices[s]
                    elif wl_rank_method == 1 and curr_closes[s] > 0: 
                        score = -abs(curr_closes[s] - long_ma[d, s]) / curr_closes[s]
                    elif wl_rank_method == 2 and short_ma[d, s] > 0: 
                        score = (curr_closes[s] - short_ma[d, s]) / short_ma[d, s]
                        
                    if score > best_score: 
                        best_score = score
                        best_s = s
            
            if best_s != -1:
                buy_price = curr_opens[best_s]
                if buy_price > 0:
                    shares = 8000.0 / (buy_price * (1 + COST_PCT))
                    in_pos[best_s] = True
                    entry_prices[best_s] = buy_price
                    entry_days[best_s] = d
                    shares_held[best_s] = shares
                    high_since_entry[best_s] = buy_price
                    wl_active[best_s] = False 
                    
                    if has_sip_chunk: monthly_allowance -= 8000.0
                    else: accumulated_capital -= 8000.0
                else: wl_active[best_s] = False 
            else: 
                break 
            
        # 4. DAILY PORTFOLIO VALUATION TRACKER
        curr_val = accumulated_capital + monthly_allowance
        for s in range(n_stocks):
            if in_pos[s]: curr_val += shares_held[s] * curr_closes[s]
        
        if curr_val > 0: daily_port_val[d] = curr_val
        else: daily_port_val[d] = daily_port_val[d-1] if d > 0 else 0
        daily_bench_val[d] = bench_shares * index_closes[d]

    # --- PORTFOLIO METRICS MATH ---
    final_wealth = daily_port_val[n_days-2]
    final_bench_wealth = daily_bench_val[n_days-2]
    trade_count = winning_trades + losing_trades
    
    avg_bars = total_bars_in_trades / trade_count if trade_count > 0 else 0.0
    avg_runup = win_pct_sum / winning_trades if winning_trades > 0 else 0.0
    avg_loss = loss_pct_sum / losing_trades if losing_trades > 0 else 0.0
    
    max_dd = 0.0
    peak = daily_port_val[1]
    curr_dd_dur = 0
    max_dd_dur = 0
    returns_sum = 0.0
    returns_sq_sum = 0.0
    valid_days = 0
    
    for d in range(2, n_days - 1):
        if daily_port_val[d] > peak:
            peak = daily_port_val[d]
            curr_dd_dur = 0
        else:
            if peak > 0:
                dd = (daily_port_val[d] - peak) / peak
                if dd < max_dd: max_dd = dd
            curr_dd_dur += 1
            if curr_dd_dur > max_dd_dur: max_dd_dur = curr_dd_dur
            
        prev = daily_port_val[d-1]
        if prev > 0:
            ret = (daily_port_val[d] - prev) / prev
            returns_sum += ret
            returns_sq_sum += ret * ret
            valid_days += 1
            
    sharpe = 0.0
    if valid_days > 1:
        mean_ret = returns_sum / valid_days
        var_ret = (returns_sq_sum / valid_days) - (mean_ret * mean_ret)
        if var_ret > 0:
            std_ret = math.sqrt(var_ret)
            if std_ret > 0:
                sharpe = (mean_ret / std_ret) * math.sqrt(252)

    info_ratio = 0.0
    if month_cnt > 2:
        excess_returns = port_monthly_returns[:month_cnt] - bench_monthly_returns[:month_cnt]
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)
        if std_excess > 0: info_ratio = (mean_excess / std_excess) * math.sqrt(12)

    return final_wealth, final_bench_wealth, total_invested_capital, total_wins, total_losses, trade_count, avg_bars, avg_runup, avg_loss, max_dd, max_dd_dur, sharpe, info_ratio

# ==========================================
# 5. DATA PREP & EVALUATION
# ==========================================
def prepare_matrix_data(tickers):
    if not os.path.exists("data"): os.makedirs("data")
    raw_dfs = {}; master_dates = set()
    
    for ticker in tqdm(tickers, desc="Downloading Data"):
        file_path = f"data/{ticker}.csv"
        if os.path.exists(file_path): 
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        else:
            try:
                df = yf.download(ticker, start=START_DATE, progress=False, multi_level_index=False)
                if len(df) > 200: df.to_csv(file_path)
            except: continue
        if df is not None and len(df) > 500:
            # Calculate ATR (Average True Range) before matrix conversion
            df['tr0'] = abs(df['High'] - df['Low'])
            df['tr1'] = abs(df['High'] - df['Close'].shift())
            df['tr2'] = abs(df['Low'] - df['Close'].shift())
            df['TR'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
            df['ATR'] = df['TR'].rolling(window=14, min_periods=1).mean().fillna(0)
            
            raw_dfs[ticker] = df; master_dates.update(df.index)
            
    nifty_path = "data/NIFTY_BENCHMARK.csv"
    if os.path.exists(nifty_path):
        nifty_df = pd.read_csv(nifty_path, parse_dates=['Date'], index_col='Date')
    else:
        nifty_df = yf.download("^NSEI", start=START_DATE, progress=False, multi_level_index=False)
        nifty_df.to_csv(nifty_path)
        
    master_dates = sorted(list(master_dates))
    master_df = pd.DataFrame(index=master_dates)
    master_df['Month'] = master_df.index.month
    
    nifty_reindexed = nifty_df.reindex(master_dates).ffill().bfill()
    index_closes = nifty_reindexed['Close'].values.flatten()
    
    n_days = len(master_dates); n_stocks = len(raw_dfs)
    
    opens = np.zeros((n_days, n_stocks))
    closes = np.zeros((n_days, n_stocks)) 
    atr = np.zeros((n_days, n_stocks))
    is_div_stock = np.zeros(n_stocks, dtype=np.bool_)
    stock_names = list(raw_dfs.keys())
    
    for i, ticker in enumerate(stock_names):
        df = raw_dfs[ticker].reindex(master_dates).ffill().bfill()
        opens[:, i] = df['Open'].values
        atr[:, i] = df['ATR'].values
        if 'Adj Close' in df.columns: closes[:, i] = df['Adj Close'].values 
        else: closes[:, i] = df['Close'].values 
            
        if ticker in DIVIDEND_KINGS: is_div_stock[i] = True
            
    return opens, closes, atr, master_df['Month'].values, index_closes, is_div_stock, stock_names

def evaluate_params(p, opens, closes, atr, months, index_closes, is_div_stock):
    if p['s_ma'] >= p['l_ma']: return -999, {}
    
    n_days, n_stocks = closes.shape
    short_ma = np.zeros((n_days, n_stocks)); long_ma = np.zeros((n_days, n_stocks)); super_ma = np.zeros((n_days, n_stocks))
    
    for s in range(n_stocks):
        short_ma[:, s] = calc_ma(closes[:, s], p['s_ma'], p['t_s'])
        long_ma[:, s] = calc_ma(closes[:, s], p['l_ma'], p['t_l'])
        super_ma[:, s] = calc_ma(closes[:, s], p['sl_ma'], p['t_sl'])
        
    # Passing the new 16 parameters cleanly
    f_wealth, f_bench, t_invested, wins, losses, trades, avg_bars, avg_runup, avg_loss, max_dd, max_dd_dur, sharpe, ir = simulate_portfolio(
        opens, closes, atr, months, index_closes, is_div_stock, 
        short_ma, long_ma, super_ma, p['wl_rank'], 
        p['entry_f'], p['n_exit_m'], p['n_trail_p'], p['n_atr_m'],
        p['div_exit_m'], p['div_exit_v']
    )
    
    if t_invested <= 0 or trades < 10: return -999, {}
    
    roi = (f_wealth - t_invested) / t_invested
    bench_roi = (f_bench - t_invested) / t_invested
    excess_alpha = roi - bench_roi
    pf = wins / losses if losses > 0 else 2.5
    
    if pf < 1.0 or roi < 0: return -999, {}
    
    score = roi * pf * (1.0 + sharpe) + (ir * 0.2)
    
    return score, {
        'roi': roi, 'bench_roi': bench_roi, 'alpha': excess_alpha, 'pf': pf, 'wealth': f_wealth, 
        'bench_wealth': f_bench, 'trades': trades, 'sharpe': sharpe, 'ir': ir,
        'max_dd': max_dd, 'max_dd_dur': max_dd_dur, 'avg_bars': avg_bars, 
        'avg_runup': avg_runup, 'avg_loss': avg_loss, 't_invested': t_invested
    }

# ==========================================
# 6. OPTIMIZER & CONSOLE REPORTER
# ==========================================
def print_performance_report(score, metrics, p):
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
        f"Volatility Trailing Stop ({p['n_atr_m']:.2f}x 14-ATR from Peak)"
    ]
    rank_names = [
        "Max Drawdown (Deepest Discount from Signal)", 
        "Risk-Off (Closest to Long MA Support)", 
        "Momentum (Highest Velocity above Short MA)"
    ]
    exit_names = [
        "Fixed Peak Drawdown (%)", 
        "Super MA Violation (%)", 
        "Time Decay (Consecutive Days Below Long MA)"
    ]
    
    print(f"""
======================================================
🏆 STABLE PORTFOLIO RECORD SET (SAVED) 🏆
Optimization Score: {score:.4f}
------------------------------------------------------
[CAPITAL & RETURNS]
💰 Total Injected (SIP):  ₹{metrics['t_invested']:,.0f}
💎 Final End Wealth:      ₹{metrics['wealth']:,.0f}
📈 Total System ROI:      {metrics['roi']*100:.2f}% (Vs Index: {metrics['bench_roi']*100:.2f}%)
📊 Terminal Excess Alpha: +{metrics['alpha']*100:.2f}%
------------------------------------------------------
[RISK & PERFORMANCE METRICS]
⚖️ Sharpe Ratio:          {metrics['sharpe']:.2f}
🛡️ Information Ratio:     {metrics['ir']:.3f}
📉 Max Port. Drawdown:    {metrics['max_dd']*100:.2f}%
📊 Profit Factor:         {metrics['pf']:.2f}
------------------------------------------------------
[TRADE STATISTICS (THE TRUE EDGE)]
🔄 Total Trades Closed:   {metrics['trades']}
🎯 Estimated Win Rate:    {(metrics['pf'] / (1 + metrics['pf'])) * 100:.1f}% 
⏱️ Avg Bars in Trade:     {metrics['avg_bars']:.0f} days
🚀 Avg Win (Run-up):      +{metrics['avg_runup']*100:.2f}%
📉 Avg Loss (Drawdown):   {metrics['avg_loss']*100:.2f}%
------------------------------------------------------
[TRADING LOGIC CONFIGURATION]
• Signal Trigger:         {ts}{p['s_ma']} crossover {tl}{p['l_ma']}
• Entry Regime Filter:    {entry_names[p['entry_f']]}
• Growth Stock Exit:      {norm_exit_names[p['n_exit_m']]}
• Purchase Priority:      {rank_names[p['wl_rank']]}
• Black Swan Exit Rule:   {exit_names[p['div_exit_m']]} set at: {p['div_exit_v']:.1f}
======================================================
    """)


def run_optimization():
    tickers = fetch_dynamic_universe()
    opens, closes, atr, months, index_closes, is_div_stock, stock_names = prepare_matrix_data(tickers)
    print(f"\nMatrix Built: {closes.shape[0]} Days x {closes.shape[1]} Stocks.")
    
    best_score = -999999.0 
    
    # --- 1. TEST PREVIOUS WINNER ---
    prev_score, prev_params = load_previous_winner()
    if prev_params is not None:
        if 'entry_f' in prev_params:
            print("\n--- Retesting Previous Champion ---")
            score, metrics = evaluate_params(prev_params, opens, closes, atr, months, index_closes, is_div_stock)
            if score > -900:
                best_score = score
                print(f"✅ Champion Retained! Score: {score:.4f} | ROI: {metrics['roi']*100:.2f}% | Sharpe: {metrics['sharpe']:.2f}")
                
                # --> NEW: Print the full report for the loaded champion
                print_performance_report(score, metrics, prev_params)
                
                save_winner(score, prev_params) 
            else:
                print("❌ Previous Champion failed current risk parameters based on updated data. Resetting baseline.")
        else:
            print("⚠️ Architecture upgraded. Previous parameters lack new Regimes/Stops. Starting fresh search.")
            
    # --- 2. START RANDOM SEARCH ---
    print("\nStarting Advanced Optimization with Regime & Trailing Stop Validation...")
    r_short = np.random.randint(5, 60, NUM_ITERATIONS)
    r_long = np.random.randint(20, 150, NUM_ITERATIONS)
    r_super = np.random.randint(100, 250, NUM_ITERATIONS)
    r_types = np.random.randint(0, 4, (NUM_ITERATIONS, 3)) 
    r_wl_rank = np.random.randint(0, 3, NUM_ITERATIONS)
    
    # Regime Filters and Normal Stock Exit Optimization
    r_entry_filter = np.random.randint(0, 3, NUM_ITERATIONS)
    r_norm_exit_m = np.random.randint(0, 3, NUM_ITERATIONS)
    r_norm_trail_p = np.random.uniform(5.0, 25.0, NUM_ITERATIONS)  
    r_norm_atr_m = np.random.uniform(1.0, 4.0, NUM_ITERATIONS)     
    
    r_div_exit = np.random.randint(0, 3, NUM_ITERATIONS)
    r_div_val = np.zeros(NUM_ITERATIONS)
    for i in range(NUM_ITERATIONS):
        if r_div_exit[i] == 0: r_div_val[i] = np.random.uniform(10.0, 40.0) 
        elif r_div_exit[i] == 1: r_div_val[i] = np.random.uniform(0.0, 20.0) 
        else: r_div_val[i] = np.random.randint(10, 100) 

    for i in tqdm(range(NUM_ITERATIONS), desc="Simulating"):
        p = {
            's_ma': r_short[i], 'l_ma': r_long[i], 'sl_ma': r_super[i],
            't_s': r_types[i][0], 't_l': r_types[i][1], 't_sl': r_types[i][2],
            'wl_rank': r_wl_rank[i], 
            'entry_f': r_entry_filter[i], 'n_exit_m': r_norm_exit_m[i], 
            'n_trail_p': r_norm_trail_p[i], 'n_atr_m': r_norm_atr_m[i],
            'div_exit_m': r_div_exit[i], 'div_exit_v': r_div_val[i]
        }
        
        score, metrics = evaluate_params(p, opens, closes, atr, months, index_closes, is_div_stock)
        
        if score > -900 and score > best_score:
            
            # --- NEIGHBORHOOD STABILITY CHECK ---
            neighbor_passed = True
            for offset in [-2, 2]:
                n_p = p.copy()
                n_p['s_ma'] += offset
                n_p['l_ma'] += offset
                
                if n_p['s_ma'] >= n_p['l_ma'] or n_p['s_ma'] < 3 or n_p['l_ma'] > 160: 
                    continue
                    
                n_score, _ = evaluate_params(n_p, opens, closes, atr, months, index_closes, is_div_stock)
                if n_score < (score * NEIGHBOR_THRESHOLD):
                    neighbor_passed = False
                    break
            
            if neighbor_passed:
                best_score = score
                save_winner(score, p) 
                
                # --> NEW: Call the extracted printing function here instead of dumping 30 lines
                print_performance_report(score, metrics, p)

if __name__ == "__main__":
    run_optimization()