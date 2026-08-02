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
# 1. CONFIGURATION
# ==========================================
if not os.path.exists('data'):
    os.makedirs('data')

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

TIMEFRAME = '1d'
START_DATE = "2018-01-01 00:00:00" 

TOTAL_TESTS = 100000 
TRADING_FEE = 0.002 
FIXED_TRADE_SIZE = 1000.0 

# Scoring
DECAY_RATE = 0.05 
ROBUSTNESS_THRESHOLD = 0.60 

# Parameter Ranges (MA Focus)
RANGES = {
    'short_ma': (5, 60),
    'long_ma': (20, 100),
    'super_ma': (100, 250), # The "Trend Filter"
    'atr_period': (14, 14),
    'atr_stop_mult': (3.0, 8.0),   
    'atr_tp_mult': (10.0, 50.0), 
    'atr_trail_mult': (3.0, 10.0),
    'btc_filter_ma': (50, 200) 
}

# ==========================================
# 2. NUMBA INDICATORS (Added EMA/SMA Switch)
# ==========================================
@njit(fastmath=True)
def calc_sma(arr, length):
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
def calc_ema(arr, length):
    n = len(arr)
    out = np.empty(n); out[:] = np.nan
    if n < length: return out
    out[length-1] = np.mean(arr[:length])
    mult = 2.0 / (length + 1)
    for i in range(length, n):
        out[i] = (arr[i] - out[i-1]) * mult + out[i-1]
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
# 3. BACKTEST LOGIC (MA CROSSOVER EDITION)
# ==========================================
@njit(fastmath=True)
def backtest_ma_logic(opens, highs, lows, closes, years, 
                      short_ma, long_ma, super_ma, atr, btc_bullish,
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
        curr_atr = atr[i]
        
        # 1. MANAGE POSITION
        if in_pos:
            net_pnl_event = 0.0
            
            # Trail Stop
            if half_sold:
                potential_new_sl = curr_high - (curr_atr * trail_mult)
                if potential_new_sl > stop_loss_price:
                    stop_loss_price = potential_new_sl
            
            # Exits
            sl_hit = curr_low <= stop_loss_price
            tp_hit = (not half_sold) and (curr_high >= tp_trigger_price)
            
            # Crossunder Exit (Momentum Invalidated)
            crossunder = (short_ma[i-1] >= long_ma[i-1]) and (short_ma[i] < long_ma[i])
            
            if sl_hit:
                exit_p = stop_loss_price if curr_open > stop_loss_price else curr_open
                revenue = units * exit_p
                cost = units * entry_price
                pnl = revenue - cost
                fees = (revenue + cost) * trading_fee
                net_pnl_event = pnl - fees
                
                yearly_pnl[yr_idx] += net_pnl_event
                if net_pnl_event > 0: gross_win_dollars += net_pnl_event
                else: gross_loss_dollars += abs(net_pnl_event)
                
                in_pos = False; half_sold = False; units = 0.0
                
            elif tp_hit and not sl_hit:
                sell_p = tp_trigger_price if curr_open < tp_trigger_price else curr_open
                units_sell = units * 0.5
                revenue = units_sell * sell_p
                cost = units_sell * entry_price
                pnl = revenue - cost
                fees = (revenue + cost) * trading_fee
                net_pnl_event = pnl - fees
                
                yearly_pnl[yr_idx] += net_pnl_event
                if net_pnl_event > 0: gross_win_dollars += net_pnl_event
                else: gross_loss_dollars += abs(net_pnl_event)
                
                units -= units_sell; half_sold = True
                if stop_loss_price < entry_price: stop_loss_price = entry_price
            
            elif crossunder:
                revenue = units * curr_open
                cost = units * entry_price
                pnl = revenue - cost
                fees = (revenue + cost) * trading_fee
                net_pnl_event = pnl - fees
                
                yearly_pnl[yr_idx] += net_pnl_event
                if net_pnl_event > 0: gross_win_dollars += net_pnl_event
                else: gross_loss_dollars += abs(net_pnl_event)

                in_pos = False; half_sold = False; units = 0.0

        # 2. ENTRY LOGIC
        if not in_pos:
            crossover = (short_ma[i-1] <= long_ma[i-1]) and (short_ma[i] > long_ma[i])
            
            # FILTERS:
            # 1. Coin must be above Super Trend MA
            trend_ok = closes[i] > super_ma[i]
            # 2. BTC must be Bullish
            btc_ok = btc_bullish[i] == 1.0
            
            if crossover and trend_ok and btc_ok:
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
    if os.path.exists(fname): return pd.read_csv(fname)
    
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
        print(f"Error {symbol}: {e}"); return None

def prepare_btc_filter():
    print("Preparing BTC Trend Filter...")
    df_btc = fetch_full_history('BTC/USDT', START_DATE)
    if df_btc is None: return None
    df_btc['timestamp'] = pd.to_datetime(df_btc['timestamp'], unit='ms')
    df_btc = df_btc.set_index('timestamp').resample('1D').ffill()
    return df_btc['close']

# ==========================================
# 5. CORE EVALUATOR (Median + PF + Dollars)
# ==========================================
def evaluate_params(p, market_data, valid_coins, yearly_weights, min_year, max_year):
    if p['s_len'] >= p['l_len']: return -999, {}
    
    global_yearly_pnl = {y: 0.0 for y in range(min_year, max_year + 1)}
    total_gross_win = 0.0
    total_gross_loss = 0.0
    total_trades_all = 0
    
    coin_scores = []
    valid_runs = 0
    
    for coin in valid_coins:
        d = market_data[coin]
        warmup = max(p['l_len'], p['su_len'], p['btc_ma']) + 2
        if len(d['c']) <= warmup: continue
        
        # Indicator Switching (EMA vs SMA)
        if p['t_s'] == 0: short_ma = calc_sma(d['c'], p['s_len'])
        else: short_ma = calc_ema(d['c'], p['s_len'])
            
        if p['t_l'] == 0: long_ma = calc_sma(d['c'], p['l_len'])
        else: long_ma = calc_ema(d['c'], p['l_len'])
            
        if p['t_su'] == 0: super_ma = calc_sma(d['c'], p['su_len'])
        else: super_ma = calc_ema(d['c'], p['su_len'])
        
        atr = numba_atr(d['h'], d['l'], d['c'], p['atr'])
        btc_ma = calc_sma(d['btc_close'], p['btc_ma']) # BTC usually standard SMA
        btc_bullish = np.where(d['btc_close'] > btc_ma, 1.0, 0.0)
        
        yearly_res, trades, g_win, g_loss = backtest_ma_logic(
            d['o'], d['h'], d['l'], d['c'], d['y'],
            short_ma, long_ma, super_ma, atr, btc_bullish,
            p['tp_m'], p['sl_m'], p['tr_m'], 
            TRADING_FEE, int(warmup)
        )
        
        if trades > 0:
            total_trades_all += trades
            total_gross_win += g_win
            total_gross_loss += g_loss
            valid_runs += 1
            
            # Kill Switch Tracking
            for y in range(min_year, max_year + 1):
                y_idx = y - 2000
                if 0 <= y_idx < 150:
                    global_yearly_pnl[y] += yearly_res[y_idx]
            
            # Individual Score
            this_coin_score = 0.0
            for y in range(min_year, max_year + 1):
                y_idx = y - 2000
                if 0 <= y_idx < 150:
                    this_coin_score += (yearly_res[y_idx] * yearly_weights[y])
            coin_scores.append(this_coin_score)

    if valid_runs == 0 or total_trades_all < 50: return -999, {}

    # 1. Kill Switch
    recent_years = [max_year, max_year-1, max_year-2]
    yearly_loss_limit = -(FIXED_TRADE_SIZE * 0.5) 
    for y in recent_years:
        if y in global_yearly_pnl and global_yearly_pnl[y] < yearly_loss_limit:
            return -999, {}

    # 2. Profit Factor
    pf = total_gross_win / total_gross_loss if total_gross_loss > 0 else 1.5
    if pf < 1.05: return -999, {}

    # 3. Median Score * PF
    median_score = np.median(coin_scores)
    final_score = median_score * pf
    
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
    # MA Tweaks
    n1 = p.copy(); n1['s_len'] = max(5, int(n1['s_len']*0.9)); n1['l_len'] = max(10, int(n1['l_len']*0.9)); neighbors.append(n1)
    n2 = p.copy(); n2['s_len'] = int(n2['s_len']*1.1); n2['l_len'] = int(n2['l_len']*1.1); neighbors.append(n2)
    # Risk Tweaks
    n3 = p.copy(); n3['sl_m'] = max(1.5, n3['sl_m'] - 0.5); neighbors.append(n3)
    n4 = p.copy(); n4['sl_m'] = n4['sl_m'] + 0.5; neighbors.append(n4)
    return neighbors

# ==========================================
# 7. OPTIMIZER MAIN
# ==========================================
def optimize():
    print("\n--- 1. LOADING DATA (MA EDITION) ---")
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
            'btc_close': btc_series.reindex(df['datetime']).fillna(method='ffill').values.astype(np.float64)
        }
        market_data[coin] = d; valid_coins.append(coin)

    yearly_weights = {}
    print("\n--- WEIGHTS ---")
    for y in range(min_year_found, max_year_found + 1):
        if y != max_year_found:
            age = max_year_found - y -1
            weight = 1.0 / ((1 + DECAY_RATE) ** age)
        else:
            weight = 0.1 # Preserving your specific current year weight preference
        yearly_weights[y] = weight
        print(f"Year {y}: {weight:.3f}")
    print("\n--- STARTING MA OPTIMIZATION ---")
    best_score = -99999
    
    for i in tqdm(range(TOTAL_TESTS)):
        # Random Param Gen
        p = {
            's_len': random.randint(*RANGES['short_ma']),
            'l_len': random.randint(*RANGES['long_ma']),
            'su_len': random.randint(*RANGES['super_ma']),
            't_s': random.randint(0,1), # 0=SMA, 1=EMA
            't_l': random.randint(0,1),
            't_su': random.randint(0,1),
            'atr': 14,
            'sl_m': random.uniform(*RANGES['atr_stop_mult']),
            'tp_m': random.uniform(*RANGES['atr_tp_mult']),
            'tr_m': random.uniform(*RANGES['atr_trail_mult']),
            'btc_ma': random.randint(*RANGES['btc_filter_ma'])
        }
        
        score, metrics = evaluate_params(p, market_data, valid_coins, yearly_weights, min_year_found, max_year_found)
        
        if score > best_score:
            neighbors = get_neighbors(p)
            n_scores = []
            for n_p in neighbors:
                ns, _ = evaluate_params(n_p, market_data, valid_coins, yearly_weights, min_year_found, max_year_found)
                if ns > -900: n_scores.append(ns)
            
            stability = (sum(n_scores)/len(n_scores))/score if (n_scores and abs(score)>1) else 0
            
            if stability >= ROBUSTNESS_THRESHOLD:
                best_score = score
                
                # Format
                ts = "EMA" if p['t_s'] else "SMA"
                tl = "EMA" if p['t_l'] else "SMA"
                tsu = "EMA" if p['t_su'] else "SMA"
                dys = sorted(list(metrics['yearly_pnl'].keys()))[-6:]
                bk = " | ".join([f"{y}:${int(metrics['yearly_pnl'][y])}" for y in dys])
                
                tqdm.write(f"\n🚀 MA BEST! Score: {score:.0f} (PF: {metrics['pf']:.2f})")
                tqdm.write(f"Params: {ts}{p['s_len']} / {tl}{p['l_len']} | Super: {tsu}{p['su_len']} | BTC_MA: {p['btc_ma']}")
                tqdm.write(f"Risk: TP {p['tp_m']:.1f} | SL {p['sl_m']:.1f} | Trail {p['tr_m']:.1f}")
                tqdm.write(f"Breakdown: {bk}")

if __name__ == "__main__":
    optimize()