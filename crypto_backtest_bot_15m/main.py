import ccxt
import pandas as pd
import numpy as np
import random
import os
import time
from tqdm import tqdm
from numba import njit
from datetime import datetime

# ==========================================
# 1. DYNAMIC CONFIGURATION
# ==========================================
if not os.path.exists('data'):
    os.makedirs('data')

# Top Liquid Pairs
COINS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 
    'ADA/USDT', 'LINK/USDT', 'LTC/USDT', 'BCH/USDT', 'SUI/USDT', 
    'RUNE/USDT', 'XLM/USDT', 'SHIB/USDT', 'RENDER/USDT', 'JUP/USDT', 
    'ONDO/USDT', 'OM/USDT', 'BAT/USDT', 'GLM/USDT', 'AUCTION/USDT',
    'AVAX/USDT', 'DOGE/USDT', 'DOT/USDT', 'TRX/USDT', 'MATIC/USDT',
    'UNI/USDT', 'NEAR/USDT', 'APT/USDT', 'FIL/USDT', 'ATOM/USDT',
    'IMX/USDT', 'ARB/USDT', 'OP/USDT', 'INJ/USDT', 'STX/USDT',
    'ETC/USDT', 'VET/USDT', 'TIA/USDT', 'LDO/USDT', 'GRT/USDT'
]

TIMEFRAME = '15m'
START_DATE = "2022-01-01 00:00:00" 

TOTAL_TESTS = 300000 
TRADING_FEE = 0.002 
FIXED_TRADE_SIZE = 1000.0 

# Decay rate for recency weighting (0.05 = gentle bias to recent years)
DECAY_RATE = 0.05 
ROBUSTNESS_THRESHOLD = 0.60 

RANGES = {
    'fast_rsi_len': (5, 60),       
    'fast_smooth': (1, 50),        
    'slow_rsi_len': (10, 90),     
    'slow_smooth': (2, 70),        
    'trend_ma': (20, 250),        
    'atr_period': (14, 14),       
    'atr_stop_mult': (5, 10.0),   
    'atr_tp_mult': (30.0, 80.0), 
    'atr_trail_mult': (5, 10.0),
    'btc_filter_ma': (60, 150) 
}

# ==========================================
# 2. NUMBA INDICATORS
# ==========================================
@njit(fastmath=True)
def numba_sma(arr, length):
    out = np.empty_like(arr)
    out[:] = np.nan
    cumsum = 0.0
    for i in range(len(arr)):
        if not np.isnan(arr[i]):
            cumsum += arr[i]
            if i >= length:
                cumsum -= arr[i - length]
                out[i] = cumsum / length
            elif i == length - 1:
                out[i] = cumsum / length
    return out

@njit(fastmath=True)
def numba_rma(arr, length):
    out = np.empty_like(arr)
    out[:] = np.nan
    alpha = 1.0 / length
    avg = 0.0
    count = 0
    for i in range(len(arr)):
        if not np.isnan(arr[i]):
            if count < length:
                avg += arr[i]
                count += 1
                if count == length:
                    out[i] = avg / length
            else:
                out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out

@njit(fastmath=True)
def numba_rsi(close, length):
    delta = np.empty_like(close)
    delta[0] = 0
    delta[1:] = close[1:] - close[:-1]
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = numba_rma(gain, length)
    avg_loss = numba_rma(loss, length)
    out = np.empty_like(close)
    out[:] = np.nan
    for i in range(len(close)):
        if avg_loss[i] == 0:
            out[i] = 100 if avg_gain[i] != 0 else 50
        else:
            rs = avg_gain[i] / avg_loss[i]
            out[i] = 100 - (100 / (1 + rs))
    return out

@njit(fastmath=True)
def numba_atr(high, low, close, length):
    tr = np.empty_like(close)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, max(hc, lc))
    return numba_rma(tr, length)

# ==========================================
# 3. BACKTEST LOGIC
# ==========================================
@njit(fastmath=True)
def backtest_yearly_runner(opens, highs, lows, closes, years, 
                           rsi_fast, rsi_slow, ma_trend, atr, btc_bullish,
                           tp_mult, sl_mult, trail_mult, trading_fee, start_idx):
    
    yearly_pnl = np.zeros(150) 
    
    total_trades = 0
    gross_win_dollars = 0.0
    gross_loss_dollars = 0.0
    
    in_pos = False
    half_sold = False 
    entry_price = 0.0
    stop_loss_price = 0.0
    tp_trigger_price = 0.0 
    units = 0.0 
    
    n = len(closes)
    
    for i in range(start_idx, n-1):
        yr_idx = int(years[i+1]) - 2000 
        if yr_idx < 0: yr_idx = 0
        
        curr_open = opens[i+1]
        curr_high = highs[i+1]
        curr_low = lows[i+1]
        curr_close = closes[i+1]
        curr_atr = atr[i]
        
        # 1. MANAGE POSITION
        if in_pos:
            net_pnl_event = 0.0
            
            # Trail Stop
            if half_sold:
                potential_new_sl = curr_high - (curr_atr * trail_mult)
                if potential_new_sl > stop_loss_price:
                    stop_loss_price = potential_new_sl
            
            # Check Exits
            sl_hit = curr_low <= stop_loss_price
            tp_hit = (not half_sold) and (curr_high >= tp_trigger_price)
            
            if sl_hit:
                exit_p = stop_loss_price
                if curr_open < stop_loss_price: exit_p = curr_open 
                
                revenue = units * exit_p
                cost_of_chunk = units * entry_price
                gross_pnl = revenue - cost_of_chunk
                fees = (cost_of_chunk + revenue) * trading_fee
                
                net_pnl_event = gross_pnl - fees
                yearly_pnl[yr_idx] += net_pnl_event
                
                if net_pnl_event > 0: gross_win_dollars += net_pnl_event
                else: gross_loss_dollars += abs(net_pnl_event)
                
                in_pos = False
                half_sold = False
                units = 0.0
                
            elif tp_hit and not sl_hit:
                sell_p = tp_trigger_price
                if curr_open > tp_trigger_price: sell_p = curr_open 
                
                units_to_sell = units * 0.5
                revenue = units_to_sell * sell_p
                cost_of_chunk = units_to_sell * entry_price
                
                gross_pnl = revenue - cost_of_chunk
                fees = (cost_of_chunk + revenue) * trading_fee
                
                net_pnl_event = gross_pnl - fees
                yearly_pnl[yr_idx] += net_pnl_event
                
                if net_pnl_event > 0: gross_win_dollars += net_pnl_event
                else: gross_loss_dollars += abs(net_pnl_event)
                
                units -= units_to_sell 
                half_sold = True
                if stop_loss_price < entry_price:
                    stop_loss_price = entry_price
            
            else:
                crossunder = (rsi_fast[i-1] >= rsi_slow[i-1]) and (rsi_fast[i] < rsi_slow[i])
                if crossunder:
                    revenue = units * curr_open
                    cost_of_chunk = units * entry_price
                    gross_pnl = revenue - cost_of_chunk
                    fees = (cost_of_chunk + revenue) * trading_fee
                    
                    net_pnl_event = gross_pnl - fees
                    yearly_pnl[yr_idx] += net_pnl_event
                    
                    if net_pnl_event > 0: gross_win_dollars += net_pnl_event
                    else: gross_loss_dollars += abs(net_pnl_event)

                    in_pos = False
                    half_sold = False
                    units = 0.0

        # 2. CHECK ENTRY
        if not in_pos:
            crossover = (rsi_fast[i-1] <= rsi_slow[i-1]) and (rsi_fast[i] > rsi_slow[i])
            trend_bullish = closes[i] > ma_trend[i]
            market_bullish = btc_bullish[i] == 1.0
            
            if crossover and trend_bullish and market_bullish:
                entry_price = curr_open
                units = FIXED_TRADE_SIZE / entry_price
                stop_loss_price = entry_price - (curr_atr * sl_mult)
                tp_trigger_price = entry_price + (curr_atr * tp_mult)
                in_pos = True
                half_sold = False
                total_trades += 1
            
    return yearly_pnl, total_trades, gross_win_dollars, gross_loss_dollars

# ==========================================
# 4. DATA FETCHING
# ==========================================
def fetch_full_history(symbol, start_date_str):
    safe_sym = symbol.replace('/', '_')
    fname = f"data/{safe_sym}_{TIMEFRAME}_full.csv"
    
    if os.path.exists(fname):
        return pd.read_csv(fname)
    
    exchange = ccxt.binance()
    since = exchange.parse8601(start_date_str)
    all_ohlcv = []
    
    print(f"Downloading {symbol}...")
    try:
        while True:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000: break
            time.sleep(0.1)
            
        if not all_ohlcv: return None
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.to_csv(fname, index=False)
        return df
    except Exception as e:
        print(f"Error {symbol}: {e}")
        return None

def prepare_btc_filter():
    print("Preparing BTC Trend Filter...")
    df_btc = fetch_full_history('BTC/USDT', START_DATE)
    if df_btc is None: return None
    df_btc['timestamp'] = pd.to_datetime(df_btc['timestamp'], unit='ms')
    df_btc = df_btc.set_index('timestamp').resample('1D').ffill()
    return df_btc['close']

# ==========================================
# 5. CORE EVALUATOR (UPDATED: MEDIAN LOGIC)
# ==========================================
def evaluate_params(p, market_data, valid_coins, yearly_weights, min_year, max_year):
    if p['f_len'] >= p['s_len']: return -999, {}
    
    # 1. Global Metrics (For Kill Switch & Profit Factor)
    global_yearly_pnl = {y: 0.0 for y in range(min_year, max_year + 1)}
    total_gross_win = 0.0
    total_gross_loss = 0.0
    total_trades_all = 0
    
    # 2. Individual Coin Scores (For Median Calculation)
    coin_scores = []
    
    valid_runs = 0
    
    for coin in valid_coins:
        d = market_data[coin]
        warmup = max(p['s_len'] + p['s_smt'], p['ma'], p['btc_ma']) + 2
        if len(d['c']) <= warmup: continue
        
        rsi_f = numba_sma(numba_rsi(d['c'], p['f_len']), p['f_smt'])
        rsi_s = numba_sma(numba_rsi(d['c'], p['s_len']), p['s_smt'])
        ma = numba_sma(d['c'], p['ma'])
        atr = numba_atr(d['h'], d['l'], d['c'], p['atr_per'])
        
        btc_ma = numba_sma(d['btc_close'], p['btc_ma'])
        btc_bullish = np.where(d['btc_close'] > btc_ma, 1.0, 0.0)
        
        yearly_res, trades, g_win, g_loss = backtest_yearly_runner(
            d['o'], d['h'], d['l'], d['c'], d['y'],
            rsi_f, rsi_s, ma, atr, btc_bullish,
            p['tp_m'], p['sl_m'], p['tr_m'], 
            TRADING_FEE, int(warmup)
        )
        
        if trades > 0:
            total_trades_all += trades
            total_gross_win += g_win
            total_gross_loss += g_loss
            valid_runs += 1
            
            # A. Update Global Portfolio Tracker (For Kill Switch)
            for y in range(min_year, max_year + 1):
                y_idx = y - 2000
                if 0 <= y_idx < 150:
                    global_yearly_pnl[y] += yearly_res[y_idx]
            
            # B. Calculate Score for THIS COIN (Weighted by Recency)
            # This is the "Universal Soldier" check
            this_coin_score = 0.0
            for y in range(min_year, max_year + 1):
                y_idx = y - 2000
                if 0 <= y_idx < 150:
                    this_coin_score += (yearly_res[y_idx] * yearly_weights[y])
            
            coin_scores.append(this_coin_score)

    if valid_runs == 0 or total_trades_all < 50: 
        return -999, {}

    # --- DEFENSE LAYER 1: KILL SWITCH (Portfolio View) ---
    # We still check the AGGREGATE portfolio. If the portfolio blows up, it fails.
    # We don't care if the "Median Coin" survived if the "Portfolio" died.
    recent_years = [max_year, max_year-1, max_year-2]
    yearly_loss_limit = -(FIXED_TRADE_SIZE * 0.5) 
    
    for y in recent_years:
        if y in global_yearly_pnl and global_yearly_pnl[y] < yearly_loss_limit:
            return -999, {}

    # --- DEFENSE LAYER 2: PROFIT FACTOR ---
    pf = total_gross_win / total_gross_loss if total_gross_loss > 0 else 10.0
    if pf < 1.05: return -999, {}

    # --- OFFENSE LAYER: MEDIAN SCORE ---
    # Instead of summing all scores (where SHIB can dominate), we take the Median.
    # This forces the strategy to be profitable on the "Middle Coin".
    median_coin_score = np.median(coin_scores)
    
    # We multiply by PF to reward efficiency, matching your original preference
    final_score = median_coin_score * pf
    
    metrics = {
        'pf': pf,
        'trades': total_trades_all,
        'yearly_pnl': global_yearly_pnl
    }
    
    return final_score, metrics

# ==========================================
# 6. NEIGHBOR GENERATOR
# ==========================================
def get_neighbors(p):
    neighbors = []
    n1 = p.copy(); n1['f_len'] = max(5, n1['f_len'] - 2); n1['s_len'] = max(10, n1['s_len'] - 3); neighbors.append(n1)
    n2 = p.copy(); n2['f_len'] = n2['f_len'] + 2; n2['s_len'] = n2['s_len'] + 3; neighbors.append(n2)
    n3 = p.copy(); n3['ma'] = int(n3['ma'] * 0.95); neighbors.append(n3)
    n4 = p.copy(); n4['ma'] = int(n4['ma'] * 1.05); neighbors.append(n4)
    n5 = p.copy(); n5['sl_m'] = max(1.1, n5['sl_m'] - 0.2); n5['tp_m'] = n5['tp_m'] - 0.5; neighbors.append(n5)
    n6 = p.copy(); n6['sl_m'] = n6['sl_m'] + 0.2; n6['tp_m'] = n6['tp_m'] + 0.5; neighbors.append(n6)
    n7 = p.copy(); n7['btc_ma'] = int(n7['btc_ma'] * 0.9); neighbors.append(n7)
    n8 = p.copy(); n8['btc_ma'] = int(n8['btc_ma'] * 1.1); neighbors.append(n8)
    return neighbors

# ==========================================
# 7. OPTIMIZER MAIN
# ==========================================
def optimize():
    print("\n--- 1. LOADING & ALIGNING DATA ---")
    
    btc_series = prepare_btc_filter()
    if btc_series is None: return
    
    market_data = {}
    valid_coins = []
    max_year_found = 0
    min_year_found = 2100
    
    for coin in COINS:
        df = fetch_full_history(coin, START_DATE)
        if df is None or len(df) < 500: continue
        
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['year'] = df['datetime'].dt.year
        
        curr_max = df['year'].max(); curr_min = df['year'].min()
        if curr_max > max_year_found: max_year_found = curr_max
        if curr_min < min_year_found: min_year_found = curr_min
        
        d = {
            'o': df['open'].values.astype(np.float64),
            'h': df['high'].values.astype(np.float64),
            'l': df['low'].values.astype(np.float64),
            'c': df['close'].values.astype(np.float64),
            'y': df['year'].values.astype(np.int32),
            'ts': df['timestamp'].values
        }
        
        coin_dates = df['datetime']
        aligned_btc = btc_series.reindex(coin_dates).fillna(method='ffill').values
        d['btc_close'] = aligned_btc.astype(np.float64)
        
        market_data[coin] = d
        valid_coins.append(coin)

    print(f"Loaded {len(valid_coins)} coins. Time Window: {min_year_found}-{max_year_found}")
    
    yearly_weights = {}
    print("\n--- DYNAMIC RECENCY WEIGHTS ---")
    for y in range(min_year_found, max_year_found + 1):
        if y != max_year_found:
            age = max_year_found - y -1
            weight = 1.0 / ((1 + DECAY_RATE) ** age)
        else:
            weight = 0.1 # Preserving your specific current year weight preference
        yearly_weights[y] = weight
        print(f"Year {y}: {weight:.3f}")
        
    print(f"\nOptimization Mode: Median Logic + Robustness Check > {ROBUSTNESS_THRESHOLD*100}%")
    
    best_score = -999999
    best_params = None
    
    for i in tqdm(range(TOTAL_TESTS)):
        
        p = {
            'f_len': random.randint(*RANGES['fast_rsi_len']),
            'f_smt': random.randint(*RANGES['fast_smooth']),
            's_len': random.randint(*RANGES['slow_rsi_len']),
            's_smt': random.randint(*RANGES['slow_smooth']),
            'ma': random.randint(*RANGES['trend_ma']),
            'atr_per': 14,
            'sl_m': random.uniform(*RANGES['atr_stop_mult']),
            'tp_m': random.uniform(*RANGES['atr_tp_mult']),
            'tr_m': random.uniform(*RANGES['atr_trail_mult']),
            'btc_ma': random.randint(*RANGES['btc_filter_ma'])
        }
        
        score, metrics = evaluate_params(p, market_data, valid_coins, yearly_weights, min_year_found, max_year_found)
        
        if score > best_score:
            neighbors = get_neighbors(p)
            neighbor_scores = []
            
            for np_params in neighbors:
                n_score, _ = evaluate_params(np_params, market_data, valid_coins, yearly_weights, min_year_found, max_year_found)
                if n_score > -900: 
                    neighbor_scores.append(n_score)
            
            if len(neighbor_scores) > 0:
                avg_neighbor_score = sum(neighbor_scores) / len(neighbor_scores)
                
                # Protect against division by zero if score is tiny
                if abs(score) < 0.01: stability_ratio = 0
                else: stability_ratio = avg_neighbor_score / score
                
                if stability_ratio >= ROBUSTNESS_THRESHOLD:
                    best_score = score
                    best_params = p
                    
                    display_years = sorted(list(metrics['yearly_pnl'].keys()))[-9:]
                    breakdown = " | ".join([f"{y}:${int(metrics['yearly_pnl'][y])}" for y in display_years])
                    
                    print(f"\nNEW MEDIAN-ROBUST BEST! Score: {score:.0f} (Stability: {stability_ratio*100:.0f}%)")
                    print(f"PF: {metrics['pf']:.2f} | Trades: {metrics['trades']}")
                    print(f"Breakdown: {breakdown}")
                    print(f"Params: Fast RSI {p['f_len']}/{p['f_smt']} | Slow RSI {p['s_len']}/{p['s_smt']} | MA {p['ma']} | BTC {p['btc_ma']}")
                    print(f"Risk: TP {p['tp_m']:.1f} | SL {p['sl_m']:.1f} | Trail {p['tr_m']:.1f}")

    print("\nOPTIMIZATION COMPLETE")
    print("Best Parameters:", best_params)

if __name__ == "__main__":
    optimize()