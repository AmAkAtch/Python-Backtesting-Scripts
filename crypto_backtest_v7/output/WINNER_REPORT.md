# Quantitative Strategy Audit & Validation Dossier

## 1. Segregated Performance Accounting

### A. Full Portfolio Accounting (Including Terminal MTM Liquidation)
| Performance Metric | In-Sample (1186 bars: 2022-05-24 -> 2025-08-21) | Out-of-Sample Holdout (396 bars: 2025-08-22 -> 2026-09-21) |
| :--- | :--- | :--- |
| **Strategy Net Trading Profit** | **$154,570.49** | **$7,318.03** |
| **BTC DCA Benchmark Net Profit** | **$84,501.85** | **$1,231.11** |
| **Strategy Account ROI (on deposits)** | 386.43% | 52.27% |
| **BTC DCA Account ROI (on deposits)** | 211.25% | 8.79% |
| **Capital Utilization (Avg Active / Total)** | 29.89% | 19.34% |
| **Return on Capital at Risk (ROCAR)** | 900.78% | 483.35% |
| **True Max Portfolio Drawdown** | 22.70% | 14.51% |
| **Profit Factor** | 2.08 | 2.81 |
| **Win Rate** | 45.67% | 48.76% |
| **Total Completed Trades** | 427 | 121 |

### B. Organic Strategy Accounting (Closed Exclusively by Strategy Signals & Stops)
*Excludes arbitrary END_OF_TEST calendar-cutoff closures to isolate true systematic execution:*
| Performance Metric | In-Sample (Rule-Closed Only) | Out-of-Sample (Rule-Closed Only) |
| :--- | :--- | :--- |
| **Rule-Closed Net Profit** | **$156,771.98** | **$4,061.54** |
| **Rule-Closed Account ROI** | 391.93% | 29.01% |
| **Rule-Closed Profit Factor** | 2.11 | 2.02 |
| **Rule-Closed Win Rate** | 45.88% | 48.25% |
| **Completed Rule Trades** | 425 | 114 |
| **Forced Terminal Trades (END_OF_TEST)** | 2 ($-2,201.49) | 7 ($3,256.49) |

## 2. Isolated 2022 Bear Market Stress-Test (2021-11-10 -> 2022-12-31)
*Testing strategy durability during a pure market contraction regime:*
| Performance Metric | Strategy Performance | BTC DCA Benchmark | Alpha Advantage |
| :--- | :--- | :--- | :--- |
| **Full Net PnL** | **$-1,190.04** | **$-5,778.51** | **+$4,588.48** |
| **Account ROI** | **-8.50%** | **-41.28%** | **+32.77%** |
| **Max Portfolio Drawdown** | **18.30%** | ~65.0% | Capital Preserved via Macro Exit |
| **Completed Trades** | 25 | N/A | Macro Kill-Switch Active |

## 3. Position-Size Binding Constraint Diagnostics
| Constraint Mechanism | Times Bound | Percentage Share | Operational Status |
| :--- | :--- | :--- | :--- |
| **Cash Constrained (Wallet Starvation)** | 44 | 10.3% | Funded with available cash balance |
| **Tranche Ceiling ($25,000 Hard Cap)** | 0 | 0.0% | Bounds large portfolio capital |
| **Liquidity Cap (1.5% 30d ADV)** | 0 | 0.0% | Guards illiquid altcoins |
| **Equity Slot (Nominal Equity / 10)** | 383 | 89.7% | Normal proportional sizing |
| **Tranche Floor ($500 Absolute Floor)** | 0 | 0.0% | Active during early horizon |


## 4. Trade-Reason Population Breakdown
| Exit Reason | Trades | Share (%) | Net PnL ($) | Win Rate (%) |
| :--- | :--- | :--- | :--- | :--- |
| BTC_MACRO_EXIT | 38 | 6.9% | $2,539.22 | 42.1% |
| END_OF_TEST | 9 | 1.6% | $1,055.00 | 44.4% |
| SIGNAL_EXIT_TYPE_4 | 448 | 81.8% | $230,468.54 | 52.0% |
| STOP_LOSS | 53 | 9.7% | $-72,174.23 | 1.9% |


## 5. Audit Trade Sampling (Extremes Inspection)
| Coin | Entry Date | Entry Price | Exit Date | Exit Price | Reason | Net PnL | Return |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| BGB | 2024-12-09 | $2.66 | 2024-12-27 | $7.28 | SIGNAL_EXIT_TYPE_4 | $24,502.49 | 172.7% |
| XRP | 2024-11-11 | $0.5895 | 2024-12-07 | $2.42 | SIGNAL_EXIT_TYPE_4 | $18,892.21 | 310.0% |
| PENGU | 2025-07-11 | $0.0192 | 2025-07-29 | $0.0404 | SIGNAL_EXIT_TYPE_4 | $17,657.92 | 110.4% |
| HBAR | 2025-01-17 | $0.3634 | 2025-02-02 | $0.2670 | STOP_LOSS | $-4,509.68 | -26.7% |
| ZEC | 2024-12-06 | $70.84 | 2024-12-09 | $50.44 | STOP_LOSS | $-4,242.39 | -28.9% |
| BGB | 2024-12-28 | $8.12 | 2024-12-29 | $6.19 | STOP_LOSS | $-4,102.19 | -23.9% |

*Complete trade log exported to `winner_trades.csv`.*

## 6. Data Provenance & Integrity Audit
- **Universe Scale**: 80 shortlisted coins across multi-exchange feeds (binance: 59, bybit: 12, yahoo: 5, okx: 4).
- **Transient Missing Day Gaps**: 5 lone-day gaps detected and forward-filled across history (zero false delistings).
- **Permanent Terminal Delistings**: 2 coins cleanly identified as terminal write-offs.

#### Terminal Delisting Roster
| Coin | Originating Venue | First Active | Last Valid Bar | Total Bars | Resolution Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| XMR | BINANCE | 2019-03-15 | 2024-02-20 | 1804 | Delisted on BINANCE Feed (Verify if trading on other venues) |
| LIT | BINANCE | 2021-02-04 | 2025-02-10 | 1468 | Delisted on BINANCE Feed (Verify if trading on other venues) |


## 7. Discovered Optimal Parameter Set
```json
{
    "entry_type": 3,
    "vol_ma_len": 20,
    "vol_mult": 1.7,
    "price_lookback": 38,
    "body_atr_mult": 0.9,
    "use_btc_macro_system": true,
    "btc_ma_len": 120,
    "btc_ma_type": 0,
    "adx_thresh": 15.0,
    "max_pyramid_layers": 3,
    "wl_mode": "WL_DEEPEST_DISCOUNT",
    "use_global_tp": true,
    "tp_mult": 17.0,
    "tp_size_pct": 25.0,
    "tp_move_sl_be": true,
    "exit_type": 4,
    "sl_mult": 2.9000000000000004,
    "exit_rsi_f_len": 40,
    "exit_rsi_f_smt": 13,
    "exit_rsi_s_len": 26,
    "exit_rsi_s_smt": 25
}
            