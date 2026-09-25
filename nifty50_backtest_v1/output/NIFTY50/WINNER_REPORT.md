# NSE Quantitative Strategy Audit Dossier (V10.1) — NIFTY50

## 1. Segregated Performance Accounting (Pure Flow-Stripped TWR)

### A. Performance Baseline & Generalization Check
| Performance Metric | In-Sample (3069 sessions) | Out-of-Sample (1024 sessions, Fresh Seed) | Generalization Ratio (OOS / IS) |
| :--- | :--- | :--- | :--- |
| **Beats Literal Index CAGR?** | **PASSED (+4.37%)** | **PASSED (+0.89%)** | Direct Alpha over Buy & Hold ^NSEI |
| **Conditioned Placebo Alpha Rank** | **100.0th Percentile** | **76.0th Percentile** | Empirical rank in 50 ADX-conditioned placebo seeds |
| **Beats Conditioned Placebo Median?** | **PASSED** | **PASSED** | Beat median of random triggers (6.40%) |
| **Rule-Closed Net Profit (Pre-Tax)** | **₹2,885,577.27** | **₹119,085.56** | — |
| **Estimated Post-Tax Net Profit (STCG)** | ₹2,452,740.68 | ₹100,979.88 | 15% / 20% Net of FY Loss Set-Off |
| **Strategy Annualized TWR (CAGR)** | **14.51%** | **9.10%** | 0.63x |
| **Estimated Post-Tax CAGR** | 13.32% | 7.86% | Indian Fiscal Year audited |
| **Literal ^NSEI Index CAGR** | 10.14% | 8.21% | Buy-and-Hold index |
| **Reference 200-SMA Timed Index CAGR** | 5.79% | 4.06% | Fixed trend-following benchmark |
| **Equal-Weight Survivor Basket CAGR** | 20.87% | 17.61% | Unweighted survivor basket |
| **Strategy TWR Max Drawdown** | **18.17%** | **20.55%** | Pure flow-stripped drop |
| **Closed-Trade Cumulative Drawdown** | 11.17% | 16.83% | Contemporaneous peak equity denominator |
| **Literal ^NSEI Index Max Drawdown** | 38.44% | 15.77% | Index stress baseline |
| **Capital Utilization (Active / Equity)** | 57.36% | 47.14% | Zero cash interest credit |
| **Trade Split (Win % / BE % / Loss %)** | **59.4% / 1.3% / 39.3%** | **61.8% / 2.2% / 36.0%** | Breakeven stops (|ret|<=0.25%) isolated |
| **Nominal Win Rate (PnL > 0)** | 60.40% | 61.80% | Includes marginal wins |
| **Strategy Profit Factor** | 2.04 | 1.66 | Full trade population audited |
| **Completed Trades** | 303 | 89 | Trade volume |
| **Composite Fitness Score** | 1.6250 | 0.7672 | Multi-term Calmar fitness |

### B. Distinct Ticker & Point-in-Time Exposure Telemetry
| Concentration Metric | In-Sample (IS) | Out-of-Sample (OOS) | Operational Risk Assessment |
| :--- | :--- | :--- | :--- |
| **Distinct Tickers Traded** | 46 symbols | 43 symbols | Breadth of execution basket |
| **Contemporaneous Peak Scrip Exposure** | 27.9% | 26.2% | Sized at <=25% at entry; organic price growth floats above |
| **Top 5 Symbol Profit Share** | 32.3% | 49.1% | Share of total profitable symbols |

## 2. Comprehensive Multi-Crisis Stress Audit (Continuous Pre-Invested Book)
| Crisis Period | Strategy TWR | Strategy Max DD | ^NSEI Return | ^NSEI Max DD | Timed ^NSEI Return | Active Trades |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2011 Debt Crisis** | -2.25% | 2.25% | -16.18% | 17.63% | -1.92% | 2 |
| **2015-16 Commodity Plunge** | -1.13% | 4.53% | -18.21% | 18.84% | -4.50% | 3 |
| **2018 IL&FS Crash** | -0.86% | 6.76% | -6.21% | 13.45% | -8.03% | 13 |
| **2020 COVID Crash** | -2.36% | 7.32% | -15.44% | 38.44% | -5.35% | 9 |
| **2022 Inflation/War** | +3.33% | 7.24% | -10.47% | 16.47% | -10.03% | 13 |

## 3. Macro & Trailing Stop Ablation Audits
### A. Macro-Active Exit Ablation
| Window | Configuration | Strategy CAGR | Max Drawdown | Profit Factor | Completed Trades |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **In-Sample** | **Winner (macro_active_exit=False)** | **14.51%** | **18.17%** | **2.04** | **303** |
| In-Sample | Ablated (macro_active_exit=True) | 10.84% | 17.41% | 1.71 | 315 |
| **Out-of-Sample** | **Winner (macro_active_exit=False)** | **9.10%** | **20.55%** | **1.66** | **89** |
| Out-of-Sample | Ablated (macro_active_exit=True) | 8.80% | 19.00% | 1.75 | 91 |

### B. Trailing Stop Mechanism Ablation
| Window | Configuration | Strategy CAGR | Max DD | Profit Factor | Completed Trades | Realized PnL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **In-Sample** | **Winner (trail_atr_mult=6.5x)** | **14.51%** | **18.17%** | **2.04** | **303** | **₹2,885,577.27** |
| In-Sample | Ablated (No Trailing Stop) | 14.16% | 13.65% | 2.10 | 295 | ₹2,684,040.17 |
| **Out-of-Sample** | **Winner (trail_atr_mult=6.5x)** | **9.10%** | **20.55%** | **1.66** | **89** | **₹119,085.56** |
| Out-of-Sample | Ablated (No Trailing Stop) | 10.12% | 20.60% | 1.80 | 89 | ₹137,903.36 |

## 4. Top-K Walk-Forward Stability Matrix (3 Sequential Disjoint Folds)
> **Champion Promotion Diagnostic:** 1 of 15 candidates beat all 3 temporal folds. Champion Trial #181 selected on all-fold dominance.
> 
> **Methodological Note (Fold 3 vs Literal OOS):** Fold 3 covers the final 33% of all available sessions (2021-26), spanning the late In-Sample bull market plus the entire Out-of-Sample test window. Literal OOS strictly isolates the final 25% of trading history. An algorithm can produce positive alpha over the pure OOS test window while still trailing the index during the broader bull run embedded in Fold 3.

| Candidate Rank | Score | Fold 1 (2010-15) | ^NSEI F1 | Fold 2 (2015-21) | ^NSEI F2 | Fold 3 (2021-26) | ^NSEI F3 | Consistency / Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **★ SELECTED CHAMPION (T181)** | **1.4396** | **18.1%** | 9.7% | **12.4%** | 11.4% | **9.9%** | 8.5% | **All-Era Champion** |
| Runner-up #2 (T928) | 1.4290 | 8.9% | 9.7% | 17.5% | 11.4% | 12.8% | 8.5% | Beat Recent Fold |
| Runner-up #3 (T729) | 1.5310 | 10.9% | 9.7% | 6.6% | 11.4% | 8.7% | 8.5% | Beat Recent Fold |
| Runner-up #4 (T907) | 1.4801 | 12.4% | 9.7% | 12.9% | 11.4% | 6.5% | 8.5% | Regime Sensitive |

## 5. TWR Beta, Capture Ratios & Dual Placebo Suite
| Metric | In-Sample Value | Out-of-Sample Value | Context / Benchmark Baseline |
| :--- | :--- | :--- | :--- |
| **Strategy TWR Beta to ^NSEI** | 0.352 | 0.462 | Regression on daily returns |
| **TWR Correlation to ^NSEI** | 0.492 | 0.572 | Market co-movement |
| **Up-Market Capture Ratio** | 49.5% | 54.6% | Performance on positive index sessions |
| **Down-Market Capture Ratio** | 40.0% | 48.2% | Absorption on negative index sessions |
| **Leave-Top-5-Survivors-Out CAGR** | **10.65%** | N/A | Compare to ^NSEI (10.14%) |
| **Unconditioned Placebo Median** | **8.99%** | **7.12%** | Pure random entry without ADX gate |
| **Conditioned Placebo Median** | **9.54%** | **6.40%** | Matched ADX trend gate baseline |

### Top 5 Survivor Profit Drivers (Leave-Top-5-Out Breakdown)
| Rank | Symbol | Net Profit (INR) | Trade Count | Win Rate (%) | Share of IS Profit |
| :--- | :--- | :--- | :--- | :--- | :--- |
| #1 | **BHARTIARTL** | ₹294,136.44 | 5 | 60.0% | 10.2% |
| #2 | **JSWSTEEL** | ₹224,140.79 | 7 | 85.7% | 7.8% |
| #3 | **ADANIENT** | ₹208,004.54 | 6 | 83.3% | 7.2% |
| #4 | **WIPRO** | ₹184,592.90 | 12 | 58.3% | 6.4% |
| #5 | **TATASTEEL** | ₹174,950.11 | 4 | 75.0% | 6.1% |


## 6. Signal Funnel & Opportunity Attrition Matrix
| Stage in Funnel | Unique Signals | Disposition Description |
| :--- | :--- | :--- |
| **Total Raw Technical Triggers** | **1853** | Close > HHV, RSI cross, or MA trigger |
| ├── Macro Gate Blocked | -513 | NIFTY50 below Macro Moving Average |
| ├── Liquidity Floor Blocked | -0 | ADV < ₹0.00 |
| ├── ADX Trend Blocked | -894 | ADX < Selected Threshold |
| ├── Pyramid Limit Blocked | -29 | Scrip already at max tranches or averaging down |
| ├── Revalidation Dropped | -36 | Failed macro/ADX/state check at fill time |
| ├── Slot Saturated Dropped | -53 | No open portfolio slots available |
| ├── Cash Starved Dropped | -1 | Cash below ₹10,000.00 |
| ├── Scrip Risk Cap Dropped | -0 | Exceeded single-stock 25% exposure ceiling |
| ├── Expired in Watchlist | -24 | Exceeded 15 bars or shadow stop |
| **Executed Trades on Ledger** | **303** | Successfully filled and audited |
| **Checksum Reconciliation** | **1853** | Must strictly equal Total Raw Triggers (1853) |


## 7. Trade Excursion & Timing Decay Analysis (MFE / MAE)
### In-Sample Excursions
| Sub-Population | Average MFE (%) | Average MAE (%) | Peak Bar Location | Realized Win Efficiency |
| :--- | :--- | :--- | :--- | :--- |
| **Winning Positions** | +19.89% | -4.27% | Bar 38.0 | 57.9% |
| **Losing Positions** | +4.69% | -8.73% | Bar 12.1 | N/A (Loss Stop) |
| **Total Population** | +13.87% | -6.03% | Bar 27.7 | Realized: 3.65% |

### Out-of-Sample Excursions
| Sub-Population | Average MFE (%) | Average MAE (%) | Peak Bar Location | Realized Win Efficiency |
| :--- | :--- | :--- | :--- | :--- |
| **Winning Positions** | +15.54% | -3.94% | Bar 37.8 | 57.8% |
| **Losing Positions** | +3.98% | -7.35% | Bar 13.5 | N/A (Loss Stop) |
| **Total Population** | +11.12% | -5.24% | Bar 28.5 | Realized: 2.68% |


## 8. Realized Statutory Frictions & Depository Drag
| Statutory / Operational Fee | In-Sample Paid | Drag per Trade |
| :--- | :--- | :--- |
| **Securities Transaction Tax (STT)** | ₹175,923.08 | ₹580.60 |
| **NSE Exchange Turnover Fees** | ₹5,664.72 | ₹18.70 |
| **Stamp Duty (Buy Side)** | ₹12,962.56 | ₹42.78 |
| **SEBI Turnover Charges** | ₹175.92 | ₹0.58 |
| **GST (18% on Charges/Brokerage)** | ₹1,051.32 | ₹3.47 |
| **Depository Participant (DP) Charges** | ₹7,611.00 | ₹25.12 |
| **Bid-Ask Spread Slippage** | ₹110,832.33 | ₹365.78 |
| **Total Realized Operational Cost** | **₹314,220.93** | **₹1,037.03** |


## 9. Annual Pure Time-Weighted Return (TWR) Attribution
| Calendar Year | Strategy TWR | ^NSEI Index Return | Equal-Weight Basket | Completed Trades | Win Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2011** | -2.11% | -24.90% | -14.96% | 10 | 50.0% |
| **2012** | +11.71% | +23.92% | +40.55% | 27 | 63.0% |
| **2013** | +8.87% | +5.18% | +13.07% | 30 | 46.7% |
| **2014** | +62.53% | +33.14% | +50.48% | 31 | 77.4% |
| **2015** | +4.82% | -5.35% | +2.61% | 20 | 50.0% |
| **2016** | +5.58% | +5.06% | +11.29% | 8 | 62.5% |
| **2017** | +30.25% | +28.75% | +42.63% | 33 | 63.6% |
| **2018** | +14.54% | +4.03% | +3.25% | 27 | 63.0% |
| **2019** | +14.31% | +12.75% | +18.49% | 24 | 50.0% |
| **2020** | +6.36% | +14.77% | +30.15% | 17 | 76.5% |
| **2021** | +13.14% | +23.79% | +46.59% | 36 | 66.7% |
| **2022** | -2.35% | +2.72% | +8.28% | 17 | 35.3% |
| **2023** | +25.71% | +19.42% | +36.28% | 24 | 75.0% |
| **2024** | +25.47% | +8.75% | +19.79% | 33 | 66.7% |
| **2025** | -4.27% | +10.05% | +12.20% | 17 | 47.1% |
| **2026** | +0.83% | -10.33% | -2.93% | 11 | 63.6% |


## 10. Portfolio Slot Concurrency Distribution
| Active Concurrent Slots | Total Sessions (IS) | Time Proportion (%) |
| :--- | :--- | :--- |
| **0 Positions Active** | 449 sessions | 14.6% |
| **1 Positions Active** | 271 sessions | 8.8% |
| **2 Positions Active** | 311 sessions | 10.1% |
| **3 Positions Active** | 230 sessions | 7.5% |
| **4 Positions Active** | 234 sessions | 7.6% |
| **5 Positions Active** | 368 sessions | 12.0% |
| **6 Positions Active** | 354 sessions | 11.5% |
| **7 Positions Active** | 852 sessions | 27.8% |


## 11. Trade-Reason Population Breakdown
### In-Sample Exits
| Exit Reason | Trades | Share (%) | Net PnL | Win Rate (%) | Intended Nature |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GAP_STOP_LOSS | 2 | 0.7% | -₹143,368.32 | 0.0% | Risk Control Exit |
| GAP_TRAIL_ATR_STOP | 9 | 3.0% | -₹10,410.76 | 55.6% | Risk Control Exit |
| MAX_HOLDING_TIME | 146 | 48.2% | ₹4,289,524.88 | 87.0% | Profit Taking / Timeout |
| SIGNAL_EXIT_TYPE_6 | 59 | 19.5% | ₹686,827.65 | 59.3% | Risk Control Exit |
| STOP_LOSS | 29 | 9.6% | -₹894,933.24 | 0.0% | Risk Control Exit |
| TRAIL_ATR_STOP | 58 | 19.1% | -₹1,042,062.93 | 27.6% | Risk Control Exit |

### Out-of-Sample Exits
| Exit Reason | Trades | Share (%) | Net PnL | Win Rate (%) | Intended Nature |
| :--- | :--- | :--- | :--- | :--- | :--- |
| END_OF_TEST | 1 | 1.1% | ₹4,092.61 | 100.0% | Risk Control Exit |
| GAP_STOP_LOSS | 2 | 2.2% | -₹16,647.89 | 0.0% | Risk Control Exit |
| GAP_TRAIL_ATR_STOP | 2 | 2.2% | -₹16,727.20 | 0.0% | Risk Control Exit |
| MAX_HOLDING_TIME | 40 | 44.9% | ₹214,992.23 | 92.5% | Profit Taking / Timeout |
| SIGNAL_EXIT_TYPE_6 | 21 | 23.6% | ₹30,414.65 | 52.4% | Risk Control Exit |
| STOP_LOSS | 5 | 5.6% | -₹51,272.55 | 0.0% | Risk Control Exit |
| TRAIL_ATR_STOP | 18 | 20.2% | -₹45,766.30 | 33.3% | Risk Control Exit |


## 12. Discovered Optimal Parameter Set
```json
{
    "entry_type": 1,
    "adx_thresh": 25.0,
    "rsi_f_len": 18,
    "rsi_f_smt": 20,
    "rsi_s_len": 76,
    "rsi_s_smt": 14,
    "use_rsi_trend_filter": false,
    "use_market_macro_system": true,
    "macro_ma_len": 210,
    "macro_ma_type": 3,
    "macro_active_exit": false,
    "max_concurrent_tranches": 7,
    "wl_mode": "WL_STRONGEST_MOMENTUM",
    "use_global_tp": true,
    "tp_mult": 5.0,
    "tp_size_pct": 50.0,
    "tp_move_sl_be": false,
    "be_trigger_atr": 0.0,
    "max_holding_bars": 55,
    "sl_mult": 4.8,
    "exit_type": 6,
    "trail_atr_mult": 6.5,
    "exit_vol_ma_len": 30,
    "exit_vol_mult": 3.4,
    "max_pyramid_layers": 1
}
