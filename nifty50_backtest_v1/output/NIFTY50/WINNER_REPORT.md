# NSE Quantitative Strategy Audit Dossier (V3.0 Pure Swing) — NIFTY50

## 1. Segregated Performance Accounting

### A. Organic Strategy Accounting (Closed Exclusively by Strategy Signals, Stops & Timeouts)
| Performance Metric | In-Sample (3045 bars) | Out-of-Sample (1016 bars) |
| :--- | :--- | :--- |
| **Rule-Closed Net Profit** | **₹1,769,195.52** | **₹14,384.64** |
| **Universe DCA Benchmark Profit** | **₹2,501,943.99** | **₹60,404.76** |
| **Rule-Closed Account ROI** | 250.95% | 5.99% |
| **Benchmark Account ROI** | 354.89% | 25.17% |
| **Strategy Max Drawdown** | 21.77% | 12.49% |
| **Benchmark Max Drawdown** | 39.08% | 10.70% |
| **Capital Utilization** | 98.30% | 97.95% |
| **Rule-Closed Profit Factor** | 1.72 | 1.08 |
| **Rule-Closed Win Rate** | 41.33% | 44.03% |
| **Completed Rule Trades** | 1442 | 477 |
| **Forced Terminal Trades (EOT)** | 15 (₹94,662.05) | 15 (-₹855.56) |

### B. Full Portfolio Accounting (Including Terminal Unclosed Marks)
| Performance Metric | In-Sample | Out-of-Sample |
| :--- | :--- | :--- |
| **Full Strategy Net Profit** | **₹1,863,857.56** | **₹13,529.07** |
| **Full Account ROI** | 264.38% | 5.64% |
| **Full Profit Factor** | 1.76 | 1.08 |
| **Total Completed Trades** | 1457 | 492 |

## 2. Isolated COVID-Crash Stress-Test (2020-01-01 -> 2020-06-30)
| Performance Metric | Strategy | Universe DCA Benchmark | Alpha Advantage |
| :--- | :--- | :--- | :--- |
| **Organic Net PnL** | **-₹0.05** | **₹355.03** | **-₹355.08** |
| **Organic Account ROI** | **-0.00%** | **1.18%** | **-1.18%** |
| **Max Portfolio Drawdown** | **12.95%** | **30.45%** | **17.50% pts** |
| **Completed Trades** | 10 | N/A | |

## 3. Position-Size Binding Constraint Diagnostics
| Constraint Mechanism | Times Bound | Share | Operational Status |
| :--- | :--- | :--- | :--- |
| **Cash Constrained** | 286 | 19.6% | Funded with available balance |
| **Tranche Ceiling (₹500,000.00)** | 0 | 0.0% | Hard cap active |
| **Liquidity Cap (1.5% 30d ADV)** | 0 | 0.0% | Volume bounds |
| **Equity Slot (Capital / 15)** | 1074 | 73.7% | Proportional slot sizing |
| **Tranche Floor (₹5,000.00)** | 97 | 6.7% | Minimum size floor |


## 4. Trade-Reason Population Breakdown
| Exit Reason | Trades | Share (%) | Net PnL | Win Rate (%) |
| :--- | :--- | :--- | :--- | :--- |
| END_OF_TEST | 30 | 1.5% | ₹93,806.49 | 53.3% |
| MAX_HOLDING_TIME | 572 | 29.3% | ₹3,719,482.18 | 96.2% |
| STOP_LOSS | 1347 | 69.1% | -₹1,935,902.03 | 19.0% |


## 5. Audit Trade Sampling (Extremes Inspection)
| Symbol | Entry Date | Entry Price | Exit Date | Exit Price | Reason | Net PnL | Return |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SHRIRAMFIN | 2020-11-02 | ₹125.17 | 2021-01-11 | ₹236.73 | MAX_HOLDING_TIME | ₹73,693.76 | 88.7% |
| ADANIENT | 2022-06-20 | ₹2,103.76 | 2022-08-23 | ₹2,923.01 | STOP_LOSS | ₹57,365.59 | 38.6% |
| JSWSTEEL | 2021-01-28 | ₹349.38 | 2021-04-08 | ₹538.08 | MAX_HOLDING_TIME | ₹55,273.88 | 53.6% |
| ETERNAL | 2022-06-08 | ₹66.04 | 2022-06-30 | ₹53.78 | STOP_LOSS | -₹27,929.47 | -18.8% |
| ETERNAL | 2022-06-17 | ₹65.69 | 2022-06-30 | ₹53.77 | STOP_LOSS | -₹26,972.73 | -18.3% |
| ADANIPORTS | 2021-06-16 | ₹730.20 | 2021-06-17 | ₹629.54 | STOP_LOSS | -₹18,680.38 | -14.0% |


## 6. Discovered Optimal Parameter Set
```json
{
    "entry_type": 0,
    "entry_ma_len": 60,
    "entry_ma_type": 3,
    "use_market_macro_system": false,
    "adx_thresh": 15.0,
    "max_pyramid_layers": 2,
    "wl_mode": "WL_DEEPEST_DISCOUNT",
    "use_global_tp": false,
    "max_holding_bars": 50,
    "trail_atr_mult": 4.6,
    "sl_mult": 2.5,
    "exit_type": 0
}
