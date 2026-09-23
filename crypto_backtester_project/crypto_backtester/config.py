"""
================================================================================
 USER CONTROL PANEL  --  everything you are likely to change lives here.
================================================================================
Nothing in this file is a strategy parameter.  Strategy parameters (MA lengths,
thresholds, exit multipliers, ...) are *all* sampled by Optuna -- see search.py.
This file only holds the things Optuna must NOT be allowed to cheat on:
universe size, capital rules, transaction costs, scoring philosophy.
================================================================================
"""
from __future__ import annotations

import os
from pathlib import Path

# ==============================================================================
# 1. UNIVERSE  (step 2 + 3 of the plan)
# ==============================================================================
N_COINS = 40  # <<< how many coins to fetch / trade.  CHANGE ME.

QUOTE_ASSET = "USDT"  # quote currency of the traded pair
INTERVAL = "1d"  # bar size.  '1d' is what the whole design assumes.

# Where the ranked candidate list comes from:
#   'coingecko'      -> top-N by market cap (needs api.coingecko.com)
#   'binance_volume' -> top-N by 24h quote volume on Binance (no extra host)
#   'static'         -> use STATIC_UNIVERSE below, skip all discovery
UNIVERSE_SOURCE = "coingecko"

# OHLCV providers tried in order until one returns data.
OHLCV_PROVIDERS = ("binance", "bybit", "okx")

# Over-fetch the ranked list before filtering, because a large slice of the
# top-200 is stablecoins / wrappers / LSTs that we are about to throw away.
UNIVERSE_OVERFETCH = 5.0  # fetch N_COINS * this many candidates

# Coins that MUST be in the final list if the exchange has them at all.
# (step 3: "check for some legacy coins ... they need to be in the final list")
CORE_LEGACY = [
    "BTC", "ETH", "SOL", "XRP", "XLM", "DOGE", "LTC", "BCH", "ADA", "ETC",
    "LINK", "TRX", "DOT", "AVAX", "ATOM", "BNB", "XTZ", "ALGO", "VET", "NEO",
    "EOS", "DASH", "ZEC", "IOTA", "QTUM", "XMR",
]

# Optional: force-include coins that are *dead or delisted today*.  Adding these
# is the single cheapest way to cut survivorship bias -- see README.  They will
# simply be skipped if the provider has no history for them.
INCLUDE_DELISTED = [
    "LUNA", "UST", "FTT", "SRM", "WAVES", "BSV", "OMG", "ANT", "MIR", "SCRT",
]

STATIC_UNIVERSE: list[str] = []  # only used when UNIVERSE_SOURCE == 'static'

MIN_HISTORY_DAYS = 400  # a coin needs at least this much history to be tradable

# ------------------------------------------------------------------ noise lists
# Fiat-pegged / yield-bearing dollar tokens.  Researched against the 2026
# stablecoin league table (USDT, USDC, USDe, DAI/USDS, PYUSD, USD1, FDUSD,
# RLUSD, TUSD, USDD are the current top-10 by cap).
STABLE_DENY = {
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDD", "USDP", "PAX", "GUSD",
    "FRAX", "LUSD", "SUSD", "USDE", "SUSDE", "USDS", "SUSDS", "FDUSD",
    "PYUSD", "USD1", "RLUSD", "CRVUSD", "GHO", "USDY", "BUIDL", "USDB",
    "USDX", "USDL", "USDG", "USDF", "USDO", "USR", "DEUSD", "USDA", "USDM",
    "MIM", "DOLA", "ALUSD", "USDN", "USDJ", "VAI", "HUSD", "EURS", "EURT",
    "EURC", "EURI", "AGEUR", "STASIS", "XSGD", "XIDR", "IDRT", "BIDR",
    "TRYB", "BRZ", "CNHT", "MXNT", "ZCHF", "JPYC", "GYEN", "AEUR", "UST",
    "USTC", "USDQ", "AUSD", "M", "OUSD", "YUSD", "FEI", "TRIBE", "NUSD",
}

# Wrapped / bridged / liquid-staked / restaked representations of another asset.
# These are duplicate price series -- keeping them just double-counts a bet.
WRAPPED_DENY = {
    # BTC representations
    "WBTC", "CBBTC", "LBTC", "TBTC", "RENBTC", "HBTC", "BTCB", "SOLVBTC",
    "XBTC", "PUMPBTC", "ENZOBTC", "UNIBTC", "STBTC", "BTCBR", "BRBTC",
    # ETH representations + LSTs/LRTs
    "WETH", "STETH", "WSTETH", "WBETH", "WEETH", "EETH", "RETH", "CBETH",
    "METH", "EZETH", "RSETH", "ANKRETH", "SFRXETH", "FRXETH", "OSETH",
    "SWETH", "LSETH", "RSWETH", "PUFETH", "WOETH", "ETHX", "OETH", "MSETH",
    # SOL representations + LSTs
    "WSOL", "MSOL", "JITOSOL", "BNSOL", "JUPSOL", "STSOL", "BSOL", "INF",
    "HSOL", "VSOL", "EDGESOL", "DSOL", "LST",
    # other chains' wrapped natives / bridged
    "WBNB", "WAVAX", "WMATIC", "WPOL", "WTRX", "WHYPE", "WS", "WKAVA",
    "WCFX", "WFTM", "WBTT", "WNXM", "WEMIX", "WBETA", "BETH", "WNEAR",
    "SAVAX", "ANKRBNB", "SLISBNB", "STKATOM", "STATOM", "STTIA", "STDYDX",
    "STKBNB", "STONE", "SUPEROETH", "RSTETH",
    # tokenised commodities (not crypto beta, and thin)
    "PAXG", "XAUT", "KAU", "AGAU", "TGOLD",
}

# Regex-ish suffix/prefix patterns for leveraged / index products.
LEVERAGED_PATTERNS = (
    "UP", "DOWN", "BULL", "BEAR", "3L", "3S", "5L", "5S", "HEDGE", "HALF",
)

# Data-driven noise detection (catches pegs/wrappers that post-date this file)
DETECT_STABLE_BY_DATA = True
STABLE_MAX_DAILY_VOL = 0.005  # daily return stdev below 0.5 %/day -> a peg
DETECT_DUPLICATES_BY_DATA = True
DUP_MIN_CORR = 0.990  # return correlation with an already-accepted coin
DUP_MAX_RATIO_CV = 0.030  # + price-ratio coeff. of variation -> same asset

# ==============================================================================
# 2. CAPITAL & EXECUTION  (steps 6-8)
# ==============================================================================
TRANCHE_USD = 1_000.0  # all-in cash spent per entry (fees+slippage included)
MONTHLY_CONTRIBUTION = 1_000.0  # new cash added on the 1st trading day of a month
INITIAL_CASH = 1_000.0  # cash on the very first bar

TAKER_FEE_BPS = 10.0  # 0.10 % per side  (Binance spot taker, no BNB discount)
SLIPPAGE_BPS = 15.0  # 0.15 % base slippage per side on market orders
SLIPPAGE_ATR_COEF = 0.00  # extra slippage = coef * (ATR/close); 0 disables
TAX_RATE_ON_GAINS = 0.00  # e.g. 0.30 -> 30 % of each realised gain is paid

# Liquidity floor (step 5.4) -- a fixed, non-optimised rule.
LIQUIDITY_WINDOW = 30
LIQUIDITY_FLOOR_USD = 3_000_000.0

# Delisting / halted-bar policy (step 10, "Delisting Gap").
DELIST_GAP_DAYS = 3  # this many consecutive missing/frozen bars = halted
DELIST_VALUATION = 0.0  # 0.0 = the spec's total write-off. 1.0 = last close.
BTC_EXIT_LIQUIDATES_BTC = False  # spec says "all open *altcoin* positions"
BTC_EXIT_CLEARS_WATCHLIST = True

# ==============================================================================
# 3. BACKTEST WINDOW  (step: "51 % of coins tradeable")
# ==============================================================================
START_COVERAGE = 0.51  # begin when >= 51 % of the universe has data
HARD_START = None  # e.g. "2019-01-01" to override the coverage rule
HARD_END = None  # e.g. "2026-06-30"
WARMUP_CAP = 300  # longest lookback any indicator may use

# Walk-forward split.  Optuna only ever sees IS; OOS is a holdout used to
# penalise over-fit configs and is reported honestly in the document.
USE_WALK_FORWARD = True
OOS_FRACTION = 0.25  # last 25 % of the timeline is never optimised on
OOS_PENALTY_WEIGHT = 0.50  # 0 = ignore OOS, 1 = OOS matters as much as IS

# ==============================================================================
# 4. SCORING  (step 12)
# ==============================================================================
# Every term is squashed with tanh(x / scale) so no single metric can run away
# with the score, and so TPE sees a smooth, bounded objective.
SCORE = {
    # weights -- they are renormalised, so relative size is what matters
    "w_total_roi": 0.40,  # 12.1 total ROI over the whole universe/history
    "w_median_year": 0.25,  # 12.2 median ROI per calendar year
    "w_median_coin": 0.20,  # 12.3 median ROI per coin
    "w_profit_factor": 0.15,  # stability bonus (not in the spec; cheap insurance)
    # tanh scales: the value at which a term reaches ~76 % of its maximum
    "scale_total_roi": 0.50,  # 50 %/yr CAGR on deployed capital
    "scale_median_year": 0.40,
    "scale_median_coin": 0.60,
    "scale_profit_factor": 1.00,  # on (PF - 1)
    # drawdown (12.4) is a *multiplier*, not an additive term
    "dd_tolerance": 0.25,  # 25 % contribution-adjusted DD halves nothing yet
    "dd_power": 2.0,  # steepness of the penalty
    "dd_hard_cap": 0.60,  # above this the config is treated as invalid
    # sanity guards -- stops Optuna "winning" with 4 lucky trades
    "min_trades": 40,
    "min_coins_traded": 6,
    "min_years_with_trades": 3,
    "target_trades": 150,  # confidence ramp: score * min(1, n/target)**0.25
    "invalid_score": -5.0,
}

# ==============================================================================
# 5. OPTUNA
# ==============================================================================
N_TRIALS = 400
N_JOBS = 1  # >1 needs a real DB; sqlite + threads is fine for <= 4
SEED = 7
STUDY_NAME = "crypto_swing_v1"
USE_PRUNER = True  # prune hopeless configs at year boundaries
PRUNER_WARMUP_YEARS = 3

# ==============================================================================
# 6. PATHS
# ==============================================================================
ROOT = Path(os.environ.get("CBT_HOME", Path(__file__).resolve().parent.parent))
DATA_DIR = ROOT / "data_cache"
OUT_DIR = ROOT / "output"
CURRENT_WINNER = OUT_DIR / "current_winner.json"
FINAL_WINNER = OUT_DIR / "winner.json"
STUDY_DB = OUT_DIR / f"{STUDY_NAME}.db"
REPORT_MD = OUT_DIR / "WINNER_REPORT.md"

CACHE_MAX_AGE_HOURS = 12  # re-download OHLCV older than this
REQUEST_TIMEOUT = 30
MAX_RETRIES = 4

for _d in (DATA_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 7. ENUM LABELS (used by the report writer so the doc reads like English)
# ==============================================================================
MA_NAMES = {0: "SMA", 1: "EMA", 2: "DEMA", 3: "WMA", 4: "SMMA/RMA"}
ENTRY_NAMES = {
    0: "ENTRY_MA_BREAKOUT",
    1: "ENTRY_RSI_XOVER",
    2: "ENTRY_MA_XOVER",
    3: "ENTRY_VOL_BREAKOUT",
}
EXIT_NAMES = {
    0: "EXIT_HYBRID_ATR",
    1: "EXIT_PCT_TRAIL",
    2: "EXIT_ATR_TRAIL",
    3: "EXIT_MA_CROSSUNDER",
    4: "EXIT_RSI_CROSSUNDER",
    5: "EXIT_MA_XOVER_EXIT",
}
WL_NAMES = {
    0: "WL_DEEPEST_DISCOUNT",
    1: "WL_CLOSEST_BREAKOUT",
    2: "WL_STRONGEST_MOMENTUM",
    3: "WL_FCFS",
    4: "WL_LCFS",
    5: "WL_NONE",
}
