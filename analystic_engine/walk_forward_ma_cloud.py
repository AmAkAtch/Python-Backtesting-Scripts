import os
import json
import random
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from numba import njit
import warnings

warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HISTORY_PERIOD = "10y"
TICKER_CACHE_FILE = "market_universes_cache.json"
PRICE_CACHE_DIR = "market_price_data_cache"
N_FOLDS = 5
ITERATIONS_PER_FOLD = 150
MIN_TOUCHES = 100
BOUNCE_WINDOW = 20
VERIFICATION_DURATION = 5
N_PERMUTATIONS = 3000
MAX_SAMPLE_FOR_PERM_TEST = 4000   # subsample large groups for permutation speed
RANDOM_SEED = 42

if not os.path.exists(PRICE_CACHE_DIR):
    os.makedirs(PRICE_CACHE_DIR)

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

# =========================================================================
# 1. TICKER INGESTION (unchanged from prior version)
# =========================================================================
def fetch_and_cache_tickers():
    if os.path.exists(TICKER_CACHE_FILE):
        print(f"Local asset list found ('{TICKER_CACHE_FILE}'). Loading profiles...")
        with open(TICKER_CACHE_FILE, 'r') as f:
            return json.load(f)

    print("Cache miss on asset lists. Querying live pipelines...")
    universes = {"CRYPTO_TOP_30": [], "NASDAQ_TOP_50": [], "NIFTY_BROAD_250": []}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

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
        print(f"Crypto fetch failed ({e}). Using core defaults.")
        universes["CRYPTO_TOP_30"] = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD']

    try:
        nasdaq_tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100", attrs={"id": "constellations"})
        universes["NASDAQ_TOP_50"] = nasdaq_tables[0]['Ticker'].tolist()[:50]
    except Exception as e:
        print(f"Nasdaq fetch failed ({e}). Using core defaults.")
        universes["NASDAQ_TOP_50"] = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL']

    nifty_urls = {
        "N50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
        "NN50": "https://archives.nseindia.com/content/indices/ind_niftynext50list.csv",
        "M150": "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
    }
    nifty_combined = ['^NSEI']
    session = requests.Session()
    session.headers.update({
        'User-Agent': headers['User-Agent'],
        'Referer': 'https://www.nseindia.com/',
        'Accept-Language': 'en-US,en;q=0.9',
    })
    try:
        session.get("https://www.nseindia.com", timeout=10)
    except Exception:
        pass
    for key, url in nifty_urls.items():
        try:
            res = session.get(url, timeout=10)
            if res.status_code == 200:
                from io import StringIO
                df = pd.read_csv(StringIO(res.text))
                if 'Symbol' in df.columns:
                    nifty_combined.extend([f"{sym.strip()}.NS" for sym in df['Symbol'].dropna()])
        except Exception as e:
            print(f"Nifty {key} pipeline failed ({e}).")
    universes["NIFTY_BROAD_250"] = list(set(nifty_combined))

    with open(TICKER_CACHE_FILE, 'w') as f:
        json.dump(universes, f, indent=4)
    print("Ticker lists compiled and cached.")
    return universes


def get_cached_historical_data(ticker, period=HISTORY_PERIOD):
    safe_filename = ticker.replace("^", "INDEX_").replace(".", "_")
    filepath = os.path.join(PRICE_CACHE_DIR, f"{safe_filename}_{period}.npy")
    if os.path.exists(filepath):
        return np.load(filepath)
    try:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            closes = df['Close'][ticker].dropna().values.flatten()
        else:
            closes = df['Close'].dropna().values.flatten()
        if len(closes) > 300:
            np.save(filepath, closes)
            return closes
    except Exception:
        pass
    return None


# =========================================================================
# 2. MOVING AVERAGES
# =========================================================================
@njit(nogil=True, cache=True)
def calc_ma(prices, period, ma_type):
    n = len(prices)
    res = np.empty(n)
    res[:] = np.nan
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
    elif ma_type == 1:  # EMA
        res[start + period - 1] = np.mean(prices[start:start + period])
        mult = 2 / (period + 1)
        for i in range(start + period, n):
            res[i] = (prices[i] - res[i - 1]) * mult + res[i - 1]
    elif ma_type == 2:  # DEMA
        ema1 = np.empty(n); ema1[:] = np.nan
        ema1[start + period - 1] = np.mean(prices[start:start + period])
        mult = 2 / (period + 1)
        for i in range(start + period, n):
            ema1[i] = (prices[i] - ema1[i - 1]) * mult + ema1[i - 1]
        ema2 = np.empty(n); ema2[:] = np.nan
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


# =========================================================================
# 3. TOUCH DETECTION + REACTION MEASUREMENT (fixed reference, no chasing)
# =========================================================================

@njit(nogil=True)
def find_touches_and_reactions(closes, ma1, ma2, eval_start, eval_end,
                               max_zone_width_pct, buffer_envelope,
                               bounce_window, verification_duration):
    n = len(closes)
    hi = min(eval_end, n - max(verification_duration, bounce_window))
    lo = max(250, eval_start)

    # <-- FIX: Early exit if the window is smaller than the 250-bar MA warmup period
    if hi <= lo:
        return np.empty(0), np.empty(0)

    reactions = np.empty(hi - lo)
    frozen_breaks = np.empty(hi - lo)
    count = 0
    cooldown = 0

    for i in range(lo, hi):
        if cooldown > 0:
            cooldown -= 1
            continue
        u_i = max(ma1[i], ma2[i])
        l_i = min(ma1[i], ma2[i])
        if np.isnan(u_i) or l_i == 0:
            continue
        zone_width = ((u_i - l_i) / l_i) * 100.0
        if zone_width > max_zone_width_pct:
            continue

        price = closes[i]
        if price <= 0:  
            continue

        top_edge = u_i * (1.0 + buffer_envelope)
        bottom_edge = l_i * (1.0 - buffer_envelope)

        if price <= top_edge and price >= bottom_edge:
            u_prev = max(ma1[i - 1], ma2[i - 1])
            entered_from_above = closes[i - 1] > u_prev
            horizon = closes[i + 1: i + 1 + bounce_window]
            frozen_horizon = closes[i + 1: i + 1 + verification_duration]

            if entered_from_above:
                frozen_level = bottom_edge 
                reaction = (np.max(horizon) - price) / price * 100.0
                broke = 1
                for k in range(verification_duration):
                    if frozen_horizon[k] >= frozen_level:
                        broke = 0
                        break
            else:
                frozen_level = top_edge 
                reaction = (price - np.min(horizon)) / price * 100.0
                broke = 1
                for k in range(verification_duration):
                    if frozen_horizon[k] <= frozen_level:
                        broke = 0
                        break

            reactions[count] = reaction
            frozen_breaks[count] = broke
            count += 1
            cooldown = 5

    return reactions[:count], frozen_breaks[:count]

@njit(nogil=True)
def compute_baseline_reactions(closes, eval_start, eval_end, bounce_window):
    n = len(closes)
    hi = min(eval_end, n - bounce_window)
    lo = max(0, eval_start)
    up_react = np.empty(max(hi - lo, 0))
    down_react = np.empty(max(hi - lo, 0))
    count = 0
    for i in range(lo, hi):
        price = closes[i]
        if price <= 0:  # <-- FIX: Skip zero/negative prices to avoid division by zero
            continue
        horizon = closes[i + 1: i + 1 + bounce_window]
        if len(horizon) == 0:
            continue
        up_react[count] = (np.max(horizon) - price) / price * 100.0
        down_react[count] = (price - np.min(horizon)) / price * 100.0
        count += 1
    return up_react[:count], down_react[:count]

# =========================================================================
# 4. SIGNIFICANCE TESTING (dependency-free permutation test)
# =========================================================================
def permutation_test(sample_a, sample_b, n_perm=N_PERMUTATIONS, seed=None):
    """One-sided permutation test: is median(sample_a) > median(sample_b)
    more than would occur by chance? Returns (p_value, effect_size)."""
    rng = np.random.default_rng(seed)
    if len(sample_a) == 0 or len(sample_b) == 0:
        return 1.0, 0.0

    if len(sample_a) > MAX_SAMPLE_FOR_PERM_TEST:
        sample_a = rng.choice(sample_a, MAX_SAMPLE_FOR_PERM_TEST, replace=False)
    if len(sample_b) > MAX_SAMPLE_FOR_PERM_TEST:
        sample_b = rng.choice(sample_b, MAX_SAMPLE_FOR_PERM_TEST, replace=False)

    observed = np.median(sample_a) - np.median(sample_b)
    pooled = np.concatenate([sample_a, sample_b])
    na = len(sample_a)

    count_ge = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = np.median(pooled[:na]) - np.median(pooled[na:])
        if diff >= observed:
            count_ge += 1
    p_value = (count_ge + 1) / (n_perm + 1)  # add-one smoothing
    return p_value, observed


# =========================================================================
# 5. WALK-FORWARD FOLD CONSTRUCTION
# =========================================================================
def get_fold_boundaries(n, n_folds=N_FOLDS):
    return np.linspace(0, n, n_folds + 1).astype(int)


def sample_config():
    first_ma_type = random.choice([0, 1, 2, 3])
    first_ma_len = random.randint(15, 150)
    offset = random.randint(0, 10)
    second_ma_type = random.choice([0, 1, 2, 3])
    second_ma_len = first_ma_len + offset
    max_zone_width = random.uniform(1.5, 10.0)
    return {
        'first_ma_type': first_ma_type, 'first_ma_len': first_ma_len,
        'second_ma_type': second_ma_type, 'second_ma_len': second_ma_len,
        'offset': offset, 'max_zone_width': max_zone_width,
    }


# =========================================================================
# 6. OPTIMIZER
# =========================================================================
def run_optimizer(iterations=ITERATIONS_PER_FOLD, n_folds=N_FOLDS):
    universes = fetch_and_cache_tickers()
    ma_labels = {0: 'SMA', 1: 'EMA', 2: 'DEMA', 3: 'WMA'}
    bonferroni_alpha = 0.05 / iterations

    for universe_name, tickers in universes.items():
        print(f"\n========================================================================")
        print(f" S/R CLOUD SEARCH: {universe_name} ({len(tickers)} assets)")
        print(f"========================================================================")

        loaded_datasets, fold_boundaries = {}, {}
        for ticker in tickers:
            closes = get_cached_historical_data(ticker)
            if closes is not None and len(closes) > 300:
                loaded_datasets[ticker] = closes
                fold_boundaries[ticker] = get_fold_boundaries(len(closes), n_folds)

        print(f"Loaded {len(loaded_datasets)} historical arrays (period={HISTORY_PERIOD}).")
        if not loaded_datasets:
            continue

        fold_summaries = []

        for fold_idx in range(n_folds - 1):
            ranges_train, ranges_test = {}, {}
            for ticker, edges in fold_boundaries.items():
                train_end = edges[fold_idx + 1]
                test_start, test_end = edges[fold_idx + 1], edges[fold_idx + 2]
                ranges_train[ticker] = (0, train_end)
                ranges_test[ticker] = (test_start, test_end)

            print(f"\n--- Fold {fold_idx + 1}/{n_folds - 1} ---")

            # Baseline (null) distributions don't depend on config - compute once per fold.
            baseline_train = np.concatenate([
                np.concatenate(compute_baseline_reactions(closes, *ranges_train[t], BOUNCE_WINDOW))
                for t, closes in loaded_datasets.items()
            ])
            baseline_test = np.concatenate([
                np.concatenate(compute_baseline_reactions(closes, *ranges_test[t], BOUNCE_WINDOW))
                for t, closes in loaded_datasets.items()
            ])

            best_effect, best_config, best_touch_reactions = -np.inf, None, None

            for _ in range(iterations):
                config = sample_config()
                touch_reactions_all = []
                for ticker, closes in loaded_datasets.items():
                    ma1 = calc_ma(closes, config['first_ma_len'], config['first_ma_type'])
                    ma2 = calc_ma(closes, config['second_ma_len'], config['second_ma_type'])
                    reactions, _ = find_touches_and_reactions(
                        closes, ma1, ma2, *ranges_train[ticker],
                        config['max_zone_width'], 0.005, BOUNCE_WINDOW, VERIFICATION_DURATION
                    )
                    if len(reactions) > 0:
                        touch_reactions_all.append(reactions)

                if not touch_reactions_all:
                    continue
                pooled_reactions = np.concatenate(touch_reactions_all)
                if len(pooled_reactions) <= MIN_TOUCHES:
                    continue

                # cheap selection signal: effect size vs baseline median
                effect = np.median(pooled_reactions) - np.median(baseline_train)
                if effect > best_effect:
                    best_effect = effect
                    best_config = config
                    best_touch_reactions = pooled_reactions

            if best_config is None:
                print("  No config cleared MIN_TOUCHES on train data.")
                fold_summaries.append(None)
                continue

            # Rigorous significance test on TRAIN (sanity check only)
            p_train, eff_train = permutation_test(best_touch_reactions, baseline_train, seed=RANDOM_SEED)

            # Recompute this exact config's touches on the TEST fold and test independently
            test_reactions_all, test_frozen_breaks_all = [], []
            for ticker, closes in loaded_datasets.items():
                ma1 = calc_ma(closes, best_config['first_ma_len'], best_config['first_ma_type'])
                ma2 = calc_ma(closes, best_config['second_ma_len'], best_config['second_ma_type'])
                reactions, breaks = find_touches_and_reactions(
                    closes, ma1, ma2, *ranges_test[ticker],
                    best_config['max_zone_width'], 0.005, BOUNCE_WINDOW, VERIFICATION_DURATION
                )
                if len(reactions) > 0:
                    test_reactions_all.append(reactions)
                    test_frozen_breaks_all.append(breaks)

            test_reactions = np.concatenate(test_reactions_all) if test_reactions_all else np.array([])
            test_breaks = np.concatenate(test_frozen_breaks_all) if test_frozen_breaks_all else np.array([])

            p_test, eff_test = permutation_test(test_reactions, baseline_test, seed=RANDOM_SEED + 1)
            frozen_break_rate = (test_breaks.mean() * 100.0) if len(test_breaks) > 0 else float('nan')

            print(f"  Candidate: {best_config['first_ma_len']}-{ma_labels[best_config['first_ma_type']]} / "
                  f"{best_config['second_ma_len']}-{ma_labels[best_config['second_ma_type']]} "
                  f"(max_width={best_config['max_zone_width']:.2f}%)")
            print(f"  Train:  n_touches={len(best_touch_reactions)}  "
                  f"median_reaction={np.median(best_touch_reactions):.2f}%  "
                  f"vs baseline={np.median(baseline_train):.2f}%  "
                  f"effect={eff_train:+.2f}pp  p={p_train:.4f}")
            print(f"  TEST:   n_touches={len(test_reactions)}  "
                  f"median_reaction={(np.median(test_reactions) if len(test_reactions) else float('nan')):.2f}%  "
                  f"vs baseline={np.median(baseline_test):.2f}%  "
                  f"effect={eff_test:+.2f}pp  p={p_test:.4f}  "
                  f"frozen_break_rate={frozen_break_rate:.1f}%")

            significant = (len(test_reactions) > MIN_TOUCHES) and (p_test < bonferroni_alpha) and (eff_test > 0)
            print(f"  -> {'SIGNIFICANT out-of-sample (survives Bonferroni correction)' if significant else 'NOT significant out-of-sample'}")

            fold_summaries.append({
                'config': best_config, 'p_test': p_test, 'eff_test': eff_test,
                'n_touches_test': len(test_reactions), 'significant': significant,
                'frozen_break_rate': frozen_break_rate,
            })

        valid = [f for f in fold_summaries if f is not None]
        sig_folds = [f for f in valid if f['significant']]
        print(f"\n[SUMMARY: {universe_name}]")
        print(f"  Folds tested: {len(valid)}   Statistically significant OOS: {len(sig_folds)}")
        if sig_folds:
            avg_effect = np.mean([f['eff_test'] for f in sig_folds])
            print(f"  Mean out-of-sample effect size among significant folds: +{avg_effect:.2f}pp "
                  f"over the unconditional baseline")
            print("  -> There is a real, statistically defensible signal here (bonferroni-corrected,")
            print("     out-of-sample, vs. a proper null). Still not a strategy - no costs, no")
            print("     entries/exits, no position sizing - but this is a legitimate starting point.")
        else:
            print("  -> No fold survived correction for multiple testing on held-out data.")
            print("     Best interpretation: this MA-cloud family does not carry a detectable")
            print("     S/R signal beyond what random price levels already show in this universe.")


if __name__ == "__main__":
    run_optimizer(iterations=ITERATIONS_PER_FOLD, n_folds=N_FOLDS)