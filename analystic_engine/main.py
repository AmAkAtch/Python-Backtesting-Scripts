import os
import json
import random
import logging
import itertools
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from numba import njit
import warnings

# Mute yfinance logs and standard warnings completely for a clean console
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# Cache Configuration Directories
TICKER_CACHE_FILE = "market_universes_cache.json"
PRICE_CACHE_DIR = "market_price_data_cache"

if not os.path.exists(PRICE_CACHE_DIR):
    os.makedirs(PRICE_CACHE_DIR)

# =========================================================================
# 1. SANITIZED TICKER INGESTION LAYER
# =========================================================================
def fetch_and_cache_tickers():
    """Fetches the asset lists dynamically and saves the ticker map to disk."""
    if os.path.exists(TICKER_CACHE_FILE):
        print(f"🧬 Local asset list found ('{TICKER_CACHE_FILE}'). Loading profiles...")
        with open(TICKER_CACHE_FILE, 'r') as f:
            return json.load(f)
            
    print("🚀 Cache miss on asset lists. Querying global live pipelines...")
    universes = {"CRYPTO_TOP_30": [], "NASDAQ_TOP_50": [], "NIFTY_BROAD_250": []}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    # Crypto: Fetch Top 100 from CoinGecko and filter out the noise makers
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1"
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            crypto_data = response.json()
            EXCLUDE_SYMBOLS = {
                'USDT', 'USDC', 'DAI', 'FDUSD', 'TUSD', 'USDD', 'USDE', 'FRAX', 'PYUSD',
                'WBTC', 'WETH', 'STETH', 'WEETH', 'CBETH', 'RETH', 'WBNB', 'LEO', 'TON'
            }
            clean_crypto = []
            for coin in crypto_data:
                sym = str(coin['symbol']).upper()
                if sym in EXCLUDE_SYMBOLS or not sym.isalnum() or len(sym) > 8:
                    continue
                clean_crypto.append(f"{sym}-USD")
                if len(clean_crypto) == 30:
                    break
            universes["CRYPTO_TOP_30"] = clean_crypto
    except Exception as e:
        print(f"⚠️ Crypto fetch failed ({e}). Using core defaults.")
        universes["CRYPTO_TOP_30"] = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD']

    # Nasdaq: Fetch Top 50 via Wikipedia
    try:
        nasdaq_tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100", attrs={"id": "constellations"})
        universes["NASDAQ_TOP_50"] = nasdaq_tables[0]['Ticker'].tolist()[:50]
    except Exception as e:
        print(f"⚠️ Nasdaq fetch failed ({e}). Using core defaults.")
        universes["NASDAQ_TOP_50"] = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL']

    # Nifty 50 + Next 50 + Midcap 150 via Official NSE India Endpoints
    nifty_urls = {
        "N50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
        "NN50": "https://archives.nseindia.com/content/indices/ind_niftynext50list.csv",
        "M150": "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv"
    }
    nifty_combined = ['^NSEI']
    for key, url in nifty_urls.items():
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                df = pd.read_csv(url)
                if 'Symbol' in df.columns:
                    nifty_combined.extend([f"{sym.strip()}.NS" for sym in df['Symbol'].dropna()])
        except Exception as e:
            print(f"⚠️ Nifty {key} pipeline failed ({e}).")
            
    universes["NIFTY_BROAD_250"] = list(set(nifty_combined))

    with open(TICKER_CACHE_FILE, 'w') as f:
        json.dump(universes, f, indent=4)
    print("💾 Ticker lists compiled and cached.")
    return universes


# =========================================================================
# 2. LOCAL HISTORICAL DATA STORAGE LAYER (SAVING FUEL)
# =========================================================================
def get_cached_historical_data(ticker):
    """Loads historical price series from local binary storage if it exists,

    otherwise downloads via yfinance and saves it locally.
    """
    safe_filename = ticker.replace("^", "INDEX_").replace(".", "_")
    filepath = os.path.join(PRICE_CACHE_DIR, f"{safe_filename}.npy")
    
    # Check if we already have the values saved locally
    if os.path.exists(filepath):
        return np.load(filepath)
        
    # Cache Miss: Download data from the web
    try:
        df = yf.download(ticker, period="5y", progress=False)
        if df.empty:
            return None
            
        # Parse close series safely handling any multi-index structures
        if isinstance(df.columns, pd.MultiIndex):
            closes = df['Close'][ticker].dropna().values.flatten()
        else:
            closes = df['Close'].dropna().values.flatten()
            
        if len(closes) > 100:
            np.save(filepath, closes) # Write clean binary array straight to disk
            return closes
    except Exception:
        pass
    return None


# =========================================================================
# 3. ACCELERATED COMPILER COMPUTATION ENGINE
# =========================================================================
@njit(nogil=True, cache=True)
def calc_ma(prices, period, ma_type):
    n = len(prices)
    res = np.empty(n)
    res[:] = np.nan
    start = 0
    while start < n and np.isnan(prices[start]): start += 1
    if n - start < period: return res

    if ma_type == 0:  # SMA
        w_sum = np.sum(prices[start:start + period])
        res[start + period - 1] = w_sum / period
        for i in range(start + period, n):
            w_sum = w_sum - prices[i - period] + prices[i]
            res[i] = w_sum / period
    elif ma_type == 1:  # EMA
        res[start + period - 1] = np.mean(prices[start:start + period])
        mult = 2 / (period + 1)
        for i in range(start + period, n):
            res[i] = (prices[i] - res[i - 1]) * mult + res[i - 1]
    elif ma_type == 2:  # DEMA
        ema1 = np.empty(n)
        ema1[:] = np.nan
        ema1[start + period - 1] = np.mean(prices[start:start + period])
        mult = 2 / (period + 1)
        for i in range(start + period, n):
            ema1[i] = (prices[i] - ema1[i - 1]) * mult + ema1[i - 1]
        
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

@njit(nogil=True)
def evaluate_zone_level(closes, ma1, ma2, max_zone_width_pct=5.0, buffer_envelope=0.005, verification_duration=5, bounce_window=20):
    n = len(closes)
    touches = 0
    bounces = 0
    breaks = 0
    cooldown = 0
    bounce_magnitude_sum = 0.0
    
    for i in range(250, n - max(verification_duration, bounce_window)):
        if cooldown > 0:
            cooldown -= 1
            continue
            
        u_i = max(ma1[i], ma2[i])
        l_i = min(ma1[i], ma2[i])
        if np.isnan(u_i) or l_i == 0: continue
        
        # Verify if the two lines are strictly in range of each other
        zone_width = ((u_i - l_i) / l_i) * 100.0
        if zone_width > max_zone_width_pct: continue
            
        price = closes[i]
        top_edge = u_i * (1.0 + buffer_envelope)
        bottom_edge = l_i * (1.0 - buffer_envelope)
        
        # Price is interacting within the boundaries of the cloud
        if price <= top_edge and price >= bottom_edge:
            touches += 1
            u_prev = max(ma1[i-1], ma2[i-1])
            entered_from_above = closes[i-1] > u_prev
            
            forward_prices = closes[i+1 : i+1+verification_duration]
            horizon_prices = closes[i+1 : i+1+bounce_window]
            
            forward_l = np.empty(verification_duration)
            forward_u = np.empty(verification_duration)
            for k in range(verification_duration):
                forward_l[k] = min(ma1[i+1+k], ma2[i+1+k])
                forward_u[k] = max(ma1[i+1+k], ma2[i+1+k])
            
            if entered_from_above:
                if np.all(forward_prices < forward_l):
                    breaks += 1
                    cooldown = verification_duration * 2
                else:
                    bounces += 1
                    local_peak = np.max(horizon_prices)
                    bounce_magnitude_sum += ((local_peak - price) / price) * 100.0
                    cooldown = 5
            else:
                if np.all(forward_prices > forward_u):
                    breaks += 1
                    cooldown = verification_duration * 2
                else:
                    bounces += 1
                    local_trough = np.min(horizon_prices)
                    bounce_magnitude_sum += ((price - local_trough) / price) * 100.0
                    cooldown = 5
                    
    return bounces, breaks, touches, bounce_magnitude_sum


# =========================================================================
# 4. RANDOMIZED VARIABLE-OFFSET CONFIGURATION OPTIMIZER
# =========================================================================
def run_cached_universe_optimizer(iterations=100):
    universes = fetch_and_cache_tickers()
    ma_labels = {0: 'SMA', 1: 'EMA', 2: 'DEMA', 3: 'WMA'}
    
    for universe_name, tickers in universes.items():
        print(f"\n========================================================================")
        print(f" OPTIMIZING: {universe_name} ({len(tickers)} Assets Elements)")
        print(f"========================================================================")
        
        # Ingest/load historical price data once per universe
        loaded_datasets = {}
        for ticker in tickers:
            closes = get_cached_historical_data(ticker)
            if closes is not None:
                loaded_datasets[ticker] = closes
                
        print(f"📊 Loaded {len(loaded_datasets)} historical arrays from local disk. Processing loop...")
        
        best_fitness = 0.0
        champion = {}
        
        for run_idx in range(iterations):
            # 1. Choose structural architecture parameters randomly
            first_ma_type = random.choice([0, 1, 2, 3])
            first_ma_len = random.randint(15, 150)
            
            # 2. Hardcoded logic constraint: second_ma = first_ma + random_offset (0 to 10)
            offset = random.randint(0, 10)
            second_ma_type = random.choice([0, 1, 2, 3])
            second_ma_len = first_ma_len + offset
            
            # 3. Cloud filtering threshold parameters 
            max_zone_width = random.uniform(1.5, 10.0)
            
            total_touches, total_bounces, total_breaks, total_magnitude = 0, 0, 0, 0.0
            
            for ticker, closes in loaded_datasets.items():
                ma1 = calc_ma(closes, first_ma_len, first_ma_type)
                ma2 = calc_ma(closes, second_ma_len, second_ma_type)
                
                bounces, breaks, touches, bounce_mag = evaluate_zone_level(
                    closes, ma1, ma2, max_zone_width_pct=max_zone_width
                )
                
                total_touches += touches
                total_bounces += bounces
                total_breaks += breaks
                total_magnitude += bounce_mag
                
            if total_touches > 100:
                bounce_rate = (total_bounces / total_touches) * 100.0
                break_rate = (total_breaks / total_touches) * 100.0
                avg_mag = (total_magnitude / total_bounces) if total_bounces > 0 else 0.0
                
                # Fitness strategy maximizing bounce accuracy & clean breakdowns
                fitness = bounce_rate + break_rate
                
                if bounce_rate > 50.0 and fitness > best_fitness:
                    best_fitness = fitness
                    champion = {
                        'ma1': f"{first_ma_len}-Period {ma_labels[first_ma_type]}",
                        'ma2': f"{second_ma_len}-Period {ma_labels[second_ma_type]}",
                        'offset': offset,
                        'max_width': max_zone_width,
                        'bounce_rate': bounce_rate,
                        'break_rate': break_rate,
                        'avg_mag': avg_mag,
                        'touches': total_touches
                    }
                    
        if champion:
            print(f"\n[🥇 CHAMPION CLOUD MATRIX FOUND]:")
            print(f"  • Line 1 Structure:       {champion['ma1']}")
            print(f"  • Line 2 Structure:       {champion['ma2']} (Offset Delta: +{champion['offset']})")
            print(f"  • Max Width Constrained:  {champion['max_width']:.2f}%")
            print(f"  ------------------------------------------------")
            print(f"  • Cloud Bounce Rate:      {champion['bounce_rate']:.2f}%")
            print(f"  • Cloud Breakdown Rate:   {champion['break_rate']:.2f}%")
            print(f"  • Post-Bounce Magnitude:  +{champion['avg_mag']:.2f}% (Peak trend over 20 days)")
            print(f"  • Documented S/R Events:  {champion['touches']} touches verified")
        else:
            print("❌ Failed to isolate an optimal configuration clearing safety constraints.")

if __name__ == "__main__":
    # Adjust iteration limits depending on how deep you want the random grid search to explore
    run_cached_universe_optimizer(iterations=150)