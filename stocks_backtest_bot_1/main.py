import yfinance as yf
import pandas as pd
import numpy as np
import os
from numba import njit
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================

# EXPANDED LIST: NIFTY 100 + Prominent MIDCAPS (Simulating a Nifty 200/250 Universe)
# This includes high-momentum midcaps like Trent, BEL, HAL, Polycab, Dixon, etc.
NIFTY_UNIVERSE = [
    # --- GIANTS (Nifty 50) ---
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 
    'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'BAJFINANCE.NS', 'KOTAKBANK.NS', 'LT.NS', 
    'HCLTECH.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'MARUTI.NS', 'TITAN.NS', 'SUNPHARMA.NS', 
    'BAJAJFINSV.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'NESTLEIND.NS', 'ONGC.NS', 'NTPC.NS', 
    'POWERGRID.NS', 'JSWSTEEL.NS', 'TATASTEEL.NS', 'M&M.NS', 'TECHM.NS', 'HDFCLIFE.NS', 
    'ADANIENT.NS', 'ADANIPORTS.NS', 'COALINDIA.NS', 'TATAMOTORS.NS', 'GRASIM.NS', 
    'DIVISLAB.NS', 'SBILIFE.NS', 'DRREDDY.NS', 'CIPLA.NS', 'BPCL.NS', 'BRITANNIA.NS', 
    'EICHERMOT.NS', 'INDUSINDBK.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'TATACONSUM.NS', 
    'UPL.NS', 'APOLLOHOSP.NS', 'PIDILITIND.NS', 
    
    # --- MIDCAP / NEXT 50 FAVORITES ---
    'GODREJCP.NS', 'DABUR.NS', 'SHREECEM.NS', 'VEDL.NS', 'DLF.NS', 'HAVELLS.NS', 'SRF.NS', 
    'ICICIPRULI.NS', 'JINDALSTEL.NS', 'GAIL.NS', 'AMBUJACEM.NS', 'CHOLAFIN.NS', 'BERGEPAINT.NS', 
    'BANKBARODA.NS', 'CANBK.NS', 'BEL.NS', 'SIEMENS.NS', 'BOSCHLTD.NS', 'MCDOWELL-N.NS', 
    'MARICO.NS', 'IOC.NS', 'TORNTPHARM.NS', 'PIIND.NS', 'NAUKRI.NS', 'HAL.NS', 'TRENT.NS', 
    'TVSMOTOR.NS', 'ZOMATO.NS', 'VBL.NS', 'ABB.NS', 'PAGEIND.NS', 'COLPAL.NS',
    
    # --- HIGH MOMENTUM MIDCAPS (The "Juice" for Moving Averages) ---
    'POLYCAB.NS', 'DIXON.NS', 'PERSISTENT.NS', 'LTIM.NS', 'KPITTECH.NS', 'COFORGE.NS',
    'ASTRAL.NS', 'BALKRISIND.NS', 'FEDERALBNK.NS', 'IDFCFIRSTB.NS', 'ASHOKLEY.NS',
    'CUMMINSIND.NS', 'OBEROIRLTY.NS', 'ESCORTS.NS', 'JUBLFOOD.NS', 'MRF.NS', 'MUTHOOTFIN.NS',
    'PETRONET.NS', 'PFC.NS', 'RECLTD.NS', 'SHRIRAMFIN.NS', 'TATACHEM.NS', 'TATAPOWER.NS',
    'VOLTAS.NS', 'ZEEL.NS', 'AUROPHARMA.NS', 'LUPIN.NS', 'ALKEM.NS', 'BHEL.NS', 'SAIL.NS'
]

START_DATE = "2015-01-01"
INITIAL_CAPITAL = 10000.0   
NUM_ITERATIONS = 500000      
MIN_TRADES_REQUIRED = 50     

# INCREASED COST for Midcaps (Slippage is higher than Nifty 50)
COST_PCT = 0.0020  # 0.20% Round trip 

if not os.path.exists("data"):
    os.makedirs("data")

# ==========================================
# 2. DATA LOADING
# ==========================================
def get_stock_data(ticker):
    file_path = f"data/{ticker}.csv"
    if os.path.exists(file_path):
        try: return pd.read_csv(file_path)
        except: pass
    try:
        df = yf.download(ticker, start=START_DATE, progress=False, multi_level_index=False)
        if len(df) > 500:
            df.to_csv(file_path)
            return df
    except Exception: return None
    return None

# ==========================================
# 3. FAST INDICATORS
# ==========================================
@njit(fastmath=True, cache=True)
def calc_ema(prices, period):
    n = len(prices)
    ema = np.empty(n); ema[:] = np.nan
    if n < period: return ema
    ema[period-1] = np.mean(prices[:period])
    mult = 2 / (period + 1)
    for i in range(period, n):
        ema[i] = (prices[i] - ema[i-1]) * mult + ema[i-1]
    return ema

@njit(fastmath=True, cache=True)
def calc_sma(prices, period):
    n = len(prices)
    sma = np.empty(n); sma[:] = np.nan
    if n < period: return sma
    w_sum = np.sum(prices[:period])
    sma[period-1] = w_sum / period
    for i in range(period, n):
        w_sum = w_sum - prices[i-period] + prices[i]
        sma[i] = w_sum / period
    return sma

# ==========================================
# 4. BACKTEST LOGIC (NO LOOK-AHEAD BIAS)
# ==========================================
@njit(fastmath=True)
def backtest_logic(opens, highs, lows, closes, 
                   short_ma, long_ma, super_ma, 
                   use_trailing, trail_pct, 
                   filter_mode): 
    
    n = len(closes)
    in_pos = False
    entry_price = 0.0
    high_since_entry = 0.0
    
    total_pnl = 0.0
    wins = 0
    trades = 0
    current_equity = INITIAL_CAPITAL 
    peak_equity = INITIAL_CAPITAL
    max_drawdown_pct = 0.0
    
    start_idx = 0
    for i in range(n):
        if not np.isnan(super_ma[i]) and not np.isnan(long_ma[i]):
            start_idx = i; break
    if start_idx == 0: start_idx = 301 
            
    # Iterate until n-1 to trade on i+1
    for i in range(start_idx, n - 1):
        
        # Execution on Next Open
        next_open = opens[i+1]
        next_low = lows[i+1]
        next_high = highs[i+1]
        
        # Signal on Current Close
        curr_close = closes[i]
        
        # --- EXIT ---
        if in_pos:
            high_since_entry = max(high_since_entry, next_high)
            should_exit = False
            exit_price = 0.0
            
            # 1. Trailing Stop
            if use_trailing:
                stop_price = high_since_entry * (1.0 - trail_pct / 100.0)
                if next_low <= stop_price:
                    should_exit = True
                    exit_price = next_open if next_open < stop_price else stop_price
            
            # 2. Crossover Exit
            if not should_exit:
                if short_ma[i-1] >= long_ma[i-1] and short_ma[i] < long_ma[i]:
                    should_exit = True
                    exit_price = next_open
            
            if should_exit:
                # PnL with 0.20% Cost
                gross_pnl_pct = (exit_price - entry_price) / entry_price
                net_pnl_pct = gross_pnl_pct - COST_PCT
                
                trade_pnl = current_equity * net_pnl_pct
                total_pnl += trade_pnl
                current_equity += trade_pnl
                
                trades += 1
                if trade_pnl > 0: wins += 1
                
                if current_equity > peak_equity: peak_equity = current_equity
                dd = (peak_equity - current_equity) / peak_equity
                if dd > max_drawdown_pct: max_drawdown_pct = dd
                
                in_pos = False
                high_since_entry = 0.0

        # --- ENTRY ---
        elif not in_pos:
            crossover = short_ma[i-1] <= long_ma[i-1] and short_ma[i] > long_ma[i]
            
            trend_ok = False
            if filter_mode == 0: trend_ok = True
            elif filter_mode == 1: 
                if curr_close > super_ma[i]: trend_ok = True
            elif filter_mode == 2: 
                if curr_close < super_ma[i]: trend_ok = True
            
            if crossover and trend_ok:
                in_pos = True
                entry_price = next_open
                high_since_entry = next_open

    return total_pnl, trades, wins, max_drawdown_pct

# ==========================================
# 5. ARCHITECTURE SEARCH OPTIMIZER
# ==========================================
def run_simulation():
    print("--- 1. Loading Data (Nifty 100 + Midcap Favorites) ---")
    data_store = {}
    for ticker in tqdm(NIFTY_UNIVERSE):
        df = get_stock_data(ticker)
        if df is not None:
            # LIQUIDITY FILTER: Ignore stocks with very low avg volume
            if df['Volume'].mean() < 50000: continue
            
            data_store[ticker] = {
                'open': np.ascontiguousarray(df['Open'].values.flatten()),
                'high': np.ascontiguousarray(df['High'].values.flatten()),
                'low': np.ascontiguousarray(df['Low'].values.flatten()),
                'close': np.ascontiguousarray(df['Close'].values.flatten())
            }

    print(f"Loaded {len(data_store)} stocks.")
    print(f"--- 2. Simulating {NUM_ITERATIONS} Architectures ---")
    print(f"--- Cost per trade: {COST_PCT*100:.2f}% (Midcap Adjusted) ---")
    
    # --- RANDOM PARAMS ---
    r_short = np.random.randint(5, 61, NUM_ITERATIONS)
    r_long = np.random.randint(10, 91, NUM_ITERATIONS)
    r_super = np.random.randint(50, 301, NUM_ITERATIONS)
    r_pct = np.random.uniform(1.0, 50.0, NUM_ITERATIONS)
    r_use = np.random.randint(0, 2, NUM_ITERATIONS)
    r_type_short = np.random.randint(0, 2, NUM_ITERATIONS)
    r_type_long = np.random.randint(0, 2, NUM_ITERATIONS)
    r_type_super = np.random.randint(0, 2, NUM_ITERATIONS)
    r_filter_mode = np.random.randint(0, 3, NUM_ITERATIONS)

    best_score = -np.inf
    best_params = {}
    
    for i in tqdm(range(NUM_ITERATIONS)):
        s_ma = r_short[i]
        l_ma = r_long[i]
        if s_ma >= l_ma: continue

        type_s = r_type_short[i]; type_l = r_type_long[i]; type_sl = r_type_super[i]
        f_mode = r_filter_mode[i]; tr_pct = r_pct[i]; use_tr = bool(r_use[i])
        sl_ma = r_super[i]

        stock_pnls = []; stock_wins = 0; total_trades = 0; max_dds = []

        for ticker, data in data_store.items():
            closes = data['close']
            
            if type_s == 0: arr_short = calc_sma(closes, s_ma)
            else: arr_short = calc_ema(closes, s_ma)
                
            if type_l == 0: arr_long = calc_sma(closes, l_ma)
            else: arr_long = calc_ema(closes, l_ma)
                
            if type_sl == 0: arr_super = calc_sma(closes, sl_ma)
            else: arr_super = calc_ema(closes, sl_ma)
            
            pnl, trades, wins, dd = backtest_logic(
                data['open'], data['high'], data['low'], closes,
                arr_short, arr_long, arr_super, use_tr, tr_pct, f_mode
            )
            
            pnl_pct = (pnl / INITIAL_CAPITAL) * 100
            stock_pnls.append(pnl_pct)
            if pnl > 0: stock_wins += 1
            total_trades += trades
            max_dds.append(dd)
            
        if total_trades < MIN_TRADES_REQUIRED: continue
            
        median_pnl = np.median(stock_pnls)
        win_rate_stocks = stock_wins / len(stock_pnls)
        avg_dd = np.mean(max_dds)
        if avg_dd == 0: avg_dd = 0.001
        
        # Stability Score
        stability_score = (median_pnl * win_rate_stocks) / (1 + (avg_dd * 10))
        
        if stability_score > best_score:
            best_score = stability_score
            
            # Helper strings for logging
            t_s_str = "EMA" if type_s else "SMA"
            t_l_str = "EMA" if type_l else "SMA"
            t_sl_str = "EMA" if type_sl else "SMA"
            
            if f_mode == 0: filt_str = "OFF"
            elif f_mode == 1: filt_str = "Price > Super"
            else: filt_str = "Price < Super"
            
            # --- NEW: Format Trailing Stop String ---
            trail_msg = f"{tr_pct:.1f}%" if use_tr else "OFF"

            # --- UPDATED CONSOLE PRINT ---
            tqdm.write(f"🚀 NEW BEST: Score {best_score:.2f} | {t_s_str}{s_ma}/{t_l_str}{l_ma} | Filter: {filt_str} | Trail: {trail_msg} | Median Ret: {median_pnl:.1f}%")

            best_params = {
                'Short': f"{s_ma} {t_s_str}", 
                'Long': f"{l_ma} {t_l_str}",
                'Super': f"{sl_ma} {t_sl_str}", 
                'Filter': filt_str,
                'Trail': trail_msg,  # Use the formatted string here too
                'Score': f"{stability_score:.2f}",
                'Median Return': f"{median_pnl:.1f}%",
                'Win Rate (Stocks)': f"{win_rate_stocks*100:.1f}%",
                'Avg DD': f"{avg_dd*100:.1f}%"
            }

    print("\n" + "="*60)
    print("🏆 ULTIMATE ARCHITECTURE FOUND (NIFTY 250 UNIVERSE) 🏆")
    print("="*60)
    for k, v in best_params.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    run_simulation()