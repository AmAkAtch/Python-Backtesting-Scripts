# NSE Quantitative Strategy Audit Dossier (V10.1) — NIFTY50

## 1. Segregated Performance Accounting (Pure Flow-Stripped TWR)

### A. Performance Baseline & Generalization Check
| Performance Metric | In-Sample (3069 sessions) | Out-of-Sample (1024 sessions, Fresh Seed) | Generalization Ratio (OOS / IS) |
| :--- | :--- | :--- | :--- |
| **Beats Literal Index CAGR?** | **PASSED (+5.82%)** | **PASSED (+3.44%)** | Direct Alpha over Buy & Hold ^NSEI |
| **Conditioned Placebo Alpha Rank** | **70.0th Percentile** | **60.0th Percentile** | Empirical rank in 50 ADX-conditioned placebo seeds |
| **Beats Conditioned Placebo Median?** | **PASSED** | **PASSED** | Beat median of random triggers (10.73%) |
| **Rule-Closed Net Profit (Pre-Tax)** | **₹3,717,356.48** | **₹158,204.80** | — |
| **Estimated Post-Tax Net Profit (STCG)** | ₹3,159,753.01 | ₹131,262.71 | 15% / 20% Net of FY Loss Set-Off |
| **Strategy Annualized TWR (CAGR)** | **15.96%** | **11.65%** | 0.73x |
| **Estimated Post-Tax CAGR** | 14.70% | 9.93% | Indian Fiscal Year audited |
| **Literal ^NSEI Index CAGR** | 10.14% | 8.21% | Buy-and-Hold index |
| **Reference 200-SMA Timed Index CAGR** | 5.79% | 4.06% | Fixed trend-following benchmark |
| **Equal-Weight Survivor Basket CAGR** | 20.87% | 17.61% | Unweighted survivor basket |
| **Strategy TWR Max Drawdown** | **18.01%** | **16.66%** | Pure flow-stripped drop |
| **Closed-Trade Cumulative Drawdown** | 15.85% | 12.08% | Contemporaneous peak equity denominator |
| **Literal ^NSEI Index Max Drawdown** | 38.44% | 15.77% | Index stress baseline |
| **Capital Utilization (Active / Equity)** | 63.60% | 43.80% | Zero cash interest credit |
| **Trade Split (Win % / BE % / Loss %)** | **58.4% / 14.7% / 26.9%** | **52.8% / 17.0% / 30.2%** | Breakeven stops (|ret|<=0.25%) isolated |
| **Nominal Win Rate (PnL > 0)** | 59.90% | 52.83% | Includes marginal wins |
| **Strategy Profit Factor** | 3.23 | 2.54 | Full trade population audited |
| **Completed Trades** | 197 | 53 | Trade volume |
| **Raw Composite Fitness Score** | 2.2223 | 1.6023 | Undiscounted full-horizon Calmar/MAR fitness |

### B. Distinct Ticker & Point-in-Time Exposure Telemetry
| Concentration Metric | In-Sample (IS) | Out-of-Sample (OOS) | Operational Risk Assessment |
| :--- | :--- | :--- | :--- |
| **Distinct Tickers Traded** | 47 symbols | 32 symbols | Breadth of execution basket |
| **Contemporaneous Peak Scrip Exposure** | 28.4% | 25.3% | Sized at <=25% at entry; organic price growth floats above |
| **Top 5 Symbol Profit Share** | 48.4% | 57.2% | Share of total profitable symbols |

## 2. Comprehensive Multi-Crisis Stress Audit (Continuous Pre-Invested Book)
| Crisis Period | Strategy TWR | Strategy Max DD | ^NSEI Return | ^NSEI Max DD | Timed ^NSEI Return | Active Trades |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2011 Debt Crisis** | +0.00% | 0.00% | -16.18% | 17.63% | +0.00% | 0 |
| **2015-16 Commodity Plunge** | +0.00% | 0.00% | -18.21% | 18.84% | +0.00% | 0 |
| **2018 IL&FS Crash** | -7.63% | 11.51% | -6.21% | 13.45% | -2.62% | 7 |
| **2020 COVID Crash** | -14.57% | 17.84% | -15.44% | 38.44% | -5.32% | 10 |
| **2022 Inflation/War** | +4.83% | 2.41% | -10.47% | 16.47% | -1.17% | 6 |

## 3. Macro & Trailing Stop Ablation Audits
### A. Macro-Active Exit Ablation
| Window | Configuration | Strategy CAGR | Max Drawdown | Profit Factor | Completed Trades |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **In-Sample** | **Winner (macro_active_exit=False)** | **15.96%** | **18.01%** | **3.23** | **197** |
| In-Sample | Ablated (macro_active_exit=True) | 13.78% | 20.25% | 3.29 | 216 |
| **Out-of-Sample** | **Winner (macro_active_exit=False)** | **11.65%** | **16.66%** | **2.54** | **53** |
| Out-of-Sample | Ablated (macro_active_exit=True) | 7.01% | 16.31% | 1.65 | 64 |

### B. Trailing Stop Mechanism Ablation
*Trailing Stop Ablation not applicable: The winning configuration does not employ a trailing stop mechanism.*

## 4. Top-K Walk-Forward Stability Matrix (3 Sequential Disjoint Folds)
> **Champion Promotion Diagnostic:** 6 of 15 candidates beat all 3 temporal folds. Champion Trial #913 selected on all-fold dominance.
> 
> **Methodological Note (Fold 3 vs Literal OOS):** Fold 3 covers the final 33% of all available sessions (2021-26), spanning the late In-Sample bull market plus the entire Out-of-Sample test window. Literal OOS strictly isolates the final 25% of trading history. An algorithm can produce positive alpha over the pure OOS test window while still trailing the index during the broader bull run embedded in Fold 3.
>
> **\* Score Reconciliation:** The *Search Score* in this table is the search-time objective value after plateau neighborhood stability, crisis drawdown penalties, and jackknife sub-sampling. It is lower than the *Raw Composite Fitness Score* in Section 1, which measures undiscounted full-sample execution.

| Candidate Rank | Search Score* | Fold 1 (2010-15) | ^NSEI F1 | Fold 2 (2015-21) | ^NSEI F2 | Fold 3 (2021-26) | ^NSEI F3 | Consistency / Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **★ SELECTED CHAMPION (T913)** | **2.2223** | **13.4%** | 9.7% | **14.9%** | 11.4% | **14.5%** | 8.5% | **All-Era Champion** |
| Runner-up #2 (T84) | 2.0027 | 12.6% | 9.7% | 12.9% | 11.4% | 12.5% | 8.5% | Beat All Folds |
| Runner-up #3 (T916) | 2.1236 | 20.7% | 9.7% | 11.5% | 11.4% | 11.9% | 8.5% | Beat All Folds |
| Runner-up #4 (T772) | 2.2092 | 9.7% | 9.7% | 18.5% | 11.4% | 11.1% | 8.5% | Beat All Folds |

| Candidate Rank | Search Score* | Fold 1 (2010-15) | ^NSEI F1 | Fold 2 (2015-21) | ^NSEI F2 | Fold 3 (2021-26) | ^NSEI F3 | Consistency / Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **★ SELECTED CHAMPION (T913)** | **2.2223** | **13.4%** | 9.7% | **14.9%** | 11.4% | **14.5%** | 8.5% | **All-Era Champion** |
| Runner-up #2 (T84) | 2.0027 | 12.6% | 9.7% | 12.9% | 11.4% | 12.5% | 8.5% | Beat All Folds |
| Runner-up #3 (T916) | 2.1236 | 20.7% | 9.7% | 11.5% | 11.4% | 11.9% | 8.5% | Beat All Folds |
| Runner-up #4 (T772) | 2.2092 | 9.7% | 9.7% | 18.5% | 11.4% | 11.1% | 8.5% | Beat All Folds |

## 5. TWR Beta, Capture Ratios & Dual Placebo Suite
| Metric | In-Sample Value | Out-of-Sample Value | Context / Benchmark Baseline |
| :--- | :--- | :--- | :--- |
| **Strategy TWR Beta to ^NSEI** | 0.367 | 0.411 | Regression on daily returns |
| **TWR Correlation to ^NSEI** | 0.458 | 0.487 | Market co-movement |
| **Up-Market Capture Ratio** | 50.8% | 52.0% | Performance on positive index sessions |
| **Down-Market Capture Ratio** | 39.7% | 41.5% | Absorption on negative index sessions |
| **Leave-Top-5-Survivors-Out CAGR** | **13.07%** | N/A | Compare to ^NSEI (10.14%) |
| **Unconditioned Placebo Median** | **12.90%** | **10.08%** | Pure random entry without ADX gate |
| **Conditioned Placebo Median** | **15.11%** | **10.73%** | Matched ADX trend gate baseline |

### Top 5 Survivor Profit Drivers (Leave-Top-5-Out Breakdown)
| Rank | Symbol | Net Profit (INR) | Trade Count | Win Rate (%) | Share of IS Profit |
| :--- | :--- | :--- | :--- | :--- | :--- |
| #1 | **ADANIENT** | ₹691,944.92 | 5 | 80.0% | 18.6% |
| #2 | **TECHM** | ₹609,493.38 | 9 | 77.8% | 16.4% |
| #3 | **INDIGO** | ₹420,823.90 | 4 | 100.0% | 11.3% |
| #4 | **RELIANCE** | ₹233,387.51 | 5 | 80.0% | 6.3% |
| #5 | **WIPRO** | ₹206,334.94 | 6 | 66.7% | 5.6% |


## 6. Signal Funnel & Opportunity Attrition Matrix
| Stage in Funnel | Unique Signals | Disposition Description |
| :--- | :--- | :--- |
| **Total Raw Technical Triggers** | **7789** | Close > HHV, RSI cross, or MA trigger |
| ├── Macro Gate Blocked | -3259 | NIFTY50 below Macro Moving Average |
| ├── Liquidity Floor Blocked | -0 | ADV < ₹0.00 |
| ├── ADX Trend Blocked | -3611 | ADX < Selected Threshold |
| ├── Pyramid Limit Blocked | -156 | Scrip already at max tranches or averaging down |
| ├── Revalidation Dropped | -326 | Failed macro/ADX/state check at fill time |
| ├── Slot Saturated Dropped | -76 | No open portfolio slots available |
| ├── Cash Starved Dropped | -0 | Cash below ₹10,000.00 |
| ├── Scrip Risk Cap Dropped | -0 | Exceeded single-stock 25% exposure ceiling |
| ├── Expired in Watchlist | -164 | Exceeded 15 bars or shadow stop |
| **Executed Trades on Ledger** | **197** | Successfully filled and audited |
| **Checksum Reconciliation** | **7789** | Must strictly equal Total Raw Triggers (7789) |


## 7. Trade Excursion & Timing Decay Analysis (MFE / MAE)
### In-Sample Excursions
| Sub-Population | Average MFE (%) | Average MAE (%) | Peak Bar Location | Realized Win Efficiency |
| :--- | :--- | :--- | :--- | :--- |
| **Winning Positions** | +19.20% | -6.01% | Bar 42.7 | 71.3% |
| **Losing Positions** | +7.21% | -9.51% | Bar 13.9 | N/A (Loss Stop) |
| **Total Population** | +14.39% | -7.41% | Bar 31.2 | Realized: 5.14% |

### Out-of-Sample Excursions
| Sub-Population | Average MFE (%) | Average MAE (%) | Peak Bar Location | Realized Win Efficiency |
| :--- | :--- | :--- | :--- | :--- |
| **Winning Positions** | +20.66% | -3.60% | Bar 45.7 | 70.7% |
| **Losing Positions** | +5.36% | -7.51% | Bar 19.8 | N/A (Loss Stop) |
| **Total Population** | +13.45% | -5.44% | Bar 33.5 | Realized: 5.11% |


## 8. Realized Statutory Frictions & Depository Drag
| Statutory / Operational Fee | In-Sample Paid | Drag per Trade |
| :--- | :--- | :--- |
| **Securities Transaction Tax (STT)** | ₹148,981.90 | ₹756.25 |
| **NSE Exchange Turnover Fees** | ₹4,797.22 | ₹24.35 |
| **Stamp Duty (Buy Side)** | ₹10,882.15 | ₹55.24 |
| **SEBI Turnover Charges** | ₹148.98 | ₹0.76 |
| **GST (18% on Charges/Brokerage)** | ₹890.32 | ₹4.52 |
| **Depository Participant (DP) Charges** | ₹3,486.90 | ₹17.70 |
| **Bid-Ask Spread Slippage** | ₹95,995.98 | ₹487.29 |
| **Total Realized Operational Cost** | **₹265,183.45** | **₹1,346.11** |


## 9. Annual Pure Time-Weighted Return (TWR) Attribution
| Calendar Year | Strategy TWR | ^NSEI Index Return | Equal-Weight Basket | Completed Trades | Win Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2011** | -5.16% | -24.90% | -14.96% | 6 | 33.3% |
| **2012** | +24.63% | +23.92% | +40.55% | 12 | 75.0% |
| **2013** | +3.32% | +5.18% | +13.07% | 19 | 52.6% |
| **2014** | +27.79% | +33.14% | +50.48% | 24 | 62.5% |
| **2015** | +2.85% | -5.35% | +2.61% | 9 | 55.6% |
| **2016** | +17.46% | +5.06% | +11.29% | 12 | 66.7% |
| **2017** | +46.59% | +28.75% | +42.63% | 17 | 70.6% |
| **2018** | +8.84% | +4.03% | +3.25% | 21 | 61.9% |
| **2019** | +6.21% | +12.75% | +18.49% | 8 | 62.5% |
| **2020** | +20.35% | +14.77% | +30.15% | 19 | 47.4% |
| **2021** | +26.88% | +23.79% | +46.59% | 25 | 52.0% |
| **2022** | +4.55% | +2.72% | +8.28% | 11 | 63.6% |
| **2023** | +17.99% | +19.42% | +36.28% | 15 | 66.7% |
| **2024** | +38.89% | +8.75% | +19.79% | 21 | 66.7% |
| **2025** | -5.49% | +10.05% | +12.20% | 7 | 0.0% |
| **2026** | +2.76% | -10.33% | -2.93% | 5 | 60.0% |


## 10. Portfolio Slot Concurrency Distribution
| Active Concurrent Slots | Total Sessions (IS) | Time Proportion (%) |
| :--- | :--- | :--- |
| **0 Positions Active** | 678 sessions | 22.1% |
| **1 Positions Active** | 305 sessions | 9.9% |
| **2 Positions Active** | 165 sessions | 5.4% |
| **3 Positions Active** | 151 sessions | 4.9% |
| **4 Positions Active** | 284 sessions | 9.3% |
| **5 Positions Active** | 1486 sessions | 48.4% |


## 11. Trade-Reason Population Breakdown
### In-Sample Exits
| Exit Reason | Trades | Share (%) | Net PnL | Win Rate (%) | Intended Nature |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BREAKEVEN_SL | 24 | 12.2% | -₹172,391.12 | 0.0% | Defensive Breakeven Stop |
| MAX_HOLDING_TIME | 149 | 75.6% | ₹5,152,145.85 | 79.2% | Profit Taking / Timeout |
| STOP_LOSS | 24 | 12.2% | -₹1,262,398.26 | 0.0% | Risk Control Exit |

### Out-of-Sample Exits
| Exit Reason | Trades | Share (%) | Net PnL | Win Rate (%) | Intended Nature |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BREAKEVEN_SL | 8 | 15.1% | -₹6,067.14 | 0.0% | Defensive Breakeven Stop |
| MAX_HOLDING_TIME | 41 | 77.4% | ₹216,128.92 | 68.3% | Profit Taking / Timeout |
| STOP_LOSS | 4 | 7.5% | -₹51,856.98 | 0.0% | Risk Control Exit |


## 12. Discovered Optimal Parameter Set
```json
{
    "entry_type": 1,
    "adx_thresh": 35.0,
    "rsi_f_len": 76,
    "rsi_f_smt": 10,
    "rsi_s_len": 74,
    "rsi_s_smt": 6,
    "use_rsi_trend_filter": false,
    "use_market_macro_system": true,
    "macro_ma_len": 290,
    "macro_ma_type": 2,
    "macro_active_exit": false,
    "max_concurrent_tranches": 5,
    "wl_mode": "WL_STRONGEST_MOMENTUM",
    "use_global_tp": false,
    "be_trigger_atr": 3.5,
    "max_holding_bars": 55,
    "sl_mult": 5.6,
    "exit_type": 0,
    "trail_atr_mult": 0.0,
    "max_pyramid_layers": 1
}
