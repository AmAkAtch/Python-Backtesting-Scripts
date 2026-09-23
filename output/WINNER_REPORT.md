# Quantitative Strategy Audit & Validation Dossier

## 1. Segregated Performance Accounting

### A. Full Portfolio Accounting (Including Terminal MTM Liquidation)
| Performance Metric | In-Sample (1245 bars: 2022-03-06 -> 2025-08-01) | Out-of-Sample Holdout (416 bars: 2025-08-02 -> 2026-09-21) |
| :--- | :--- | :--- |
| **Strategy Net Trading Profit** | **$191,380.93** | **$1,967.60** |
| **BTC DCA Benchmark Net Profit** | **$87,906.48** | **$844.87** |
| **Strategy Account ROI (on deposits)** | 455.67% | 14.05% |
| **BTC DCA Account ROI (on deposits)** | 209.30% | 6.03% |
| **Capital Utilization (Avg Active / Total)** | 22.62% | 15.72% |
| **Return on Capital at Risk (ROCAR)** | 1358.71% | 175.02% |
| **True Max Portfolio Drawdown** | 15.67% | 26.11% |
| **Profit Factor** | 3.14 | 1.49 |
| **Win Rate** | 42.70% | 42.53% |
| **Total Completed Trades** | 281 | 87 |

### B. Organic Strategy Accounting (Closed Exclusively by Strategy Signals & Stops)
*Excludes arbitrary END_OF_TEST calendar-cutoff closures to isolate true systematic execution:*
| Performance Metric | In-Sample (Rule-Closed Only) | Out-of-Sample (Rule-Closed Only) |
| :--- | :--- | :--- |
| **Rule-Closed Net Profit** | **$191,380.93** | **$895.06** |
| **Rule-Closed Account ROI** | 455.67% | 6.39% |
| **Rule-Closed Profit Factor** | 3.14 | 1.23 |
| **Rule-Closed Win Rate** | 42.70% | 39.24% |
| **Completed Rule Trades** | 281 | 79 |
| **Forced Terminal Trades (END_OF_TEST)** | 0 ($0.00) | 8 ($1,072.54) |

## 2. Isolated 2022 Bear Market Stress-Test (2021-11-10 -> 2022-12-31)
*Testing strategy durability during a pure market contraction regime:*
| Performance Metric | Strategy Performance | BTC DCA Benchmark | Alpha Advantage |
| :--- | :--- | :--- | :--- |
| **Full Net PnL** | **$-1,790.20** | **$-5,778.51** | **+$3,988.32** |
| **Account ROI** | **-12.79%** | **-41.28%** | **+28.49%** |
| **Max Portfolio Drawdown** | **14.50%** | ~65.0% | Capital Preserved via Macro Exit |
| **Completed Trades** | 37 | N/A | Macro Kill-Switch Active |

## 3. Position-Size Binding Constraint Diagnostics
| Constraint Mechanism | Times Bound | Percentage Share | Operational Status |
| :--- | :--- | :--- | :--- |
| **Cash Constrained (Wallet Starvation)** | 918 | 72.9% | Funded with available cash balance |
| **Tranche Ceiling ($25,000 Hard Cap)** | 0 | 0.0% | Bounds large portfolio capital |
| **Liquidity Cap (1.5% 30d ADV)** | 0 | 0.0% | Guards illiquid altcoins |
| **Equity Slot (Nominal Equity / 10)** | 267 | 21.2% | Normal proportional sizing |
| **Tranche Floor ($500 Absolute Floor)** | 74 | 5.9% | Active during early horizon |


## 4. Trade-Reason Population Breakdown
| Exit Reason | Trades | Share (%) | Net PnL ($) | Win Rate (%) |
| :--- | :--- | :--- | :--- | :--- |
| BTC_MACRO_EXIT | 41 | 11.1% | $4,426.60 | 29.3% |
| END_OF_TEST | 8 | 2.2% | $1,072.54 | 75.0% |
| SIGNAL_EXIT_TYPE_3 | 313 | 85.1% | $188,783.55 | 44.4% |
| STOP_LOSS | 6 | 1.6% | $-934.16 | 0.0% |


## 5. Audit Trade Sampling (Extremes Inspection)
| Coin | Entry Date | Entry Price | Exit Date | Exit Price | Reason | Net PnL | Return |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XRP | 2024-11-14 | $0.6909 | 2024-12-06 | $2.24 | SIGNAL_EXIT_TYPE_3 | $21,007.22 | 223.7% |
| PENGU | 2025-07-11 | $0.0192 | 2025-07-25 | $0.0376 | SIGNAL_EXIT_TYPE_3 | $18,320.31 | 95.7% |
| XRP | 2024-11-15 | $0.7749 | 2024-12-06 | $2.24 | SIGNAL_EXIT_TYPE_3 | $17,394.99 | 188.6% |
| LIT | 2025-02-02 | $1.21 | 2025-02-03 | $0.8490 | BTC_MACRO_EXIT | $-5,195.34 | -29.9% |
| PENGU | 2025-04-28 | $0.0128 | 2025-05-04 | $0.0104 | SIGNAL_EXIT_TYPE_3 | $-3,529.12 | -19.1% |
| CAKE | 2025-05-28 | $2.78 | 2025-05-31 | $2.28 | SIGNAL_EXIT_TYPE_3 | $-3,305.61 | -18.1% |

*Complete trade log exported to `winner_trades.csv`.*

## 6. Data Provenance & Integrity Audit
- **Universe Scale**: 80 shortlisted coins across multi-exchange feeds (LEGACY_CACHE: 80).
- **Transient Missing Day Gaps**: 5 lone-day gaps detected and forward-filled across history (zero false delistings).
- **Permanent Terminal Delistings**: 2 coins cleanly identified as terminal write-offs.

#### Terminal Delisting Roster
| Coin | Originating Venue | First Active | Last Valid Bar | Total Bars | Resolution Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| XMR | UNRECORDED (Pre-V6.5 Cache) | 2019-03-15 | 2024-02-20 | 1804 | Delisted on Unrecorded Venue (Run with FORCE_REFRESH=True to identify venue) |
| LIT | UNRECORDED (Pre-V6.5 Cache) | 2021-02-04 | 2025-02-10 | 1468 | Delisted on Unrecorded Venue (Run with FORCE_REFRESH=True to identify venue) |


## 7. Discovered Optimal Parameter Set
```json
{
    "entry_type": 3,
    "vol_ma_len": 30,
    "vol_mult": 2.3,
    "price_lookback": 14,
    "body_atr_mult": 1.5,
    "use_btc_macro_system": true,
    "btc_ma_len": 80,
    "btc_ma_type": 3,
    "adx_thresh": 0.0,
    "max_pyramid_layers": 2,
    "wl_mode": "WL_LCFS",
    "use_global_tp": true,
    "tp_mult": 26.0,
    "tp_size_pct": 75.0,
    "tp_move_sl_be": true,
    "exit_type": 3,
    "sl_mult": 4.9,
    "exit_ma_len": 20,
    "exit_ma_type": 2
}
            