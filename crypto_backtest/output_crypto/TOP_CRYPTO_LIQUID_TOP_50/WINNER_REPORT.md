# Crypto Quantitative Strategy Audit Dossier (V10.2-INSTITUTIONAL) — TOP_CRYPTO_LIQUID (Top 50)

## 1. Segregated Performance Accounting (Pure 365-Day Continuous TWR)

### A. Performance Baseline & Generalization Check
| Performance Metric | In-Sample (1650 days) | Out-of-Sample (551 days, Fresh Seed) | Generalization Assessment |
| :--- | :--- | :--- | :--- |
| **Strategy Annualized TWR (CAGR)** | **158.07%** | **60.75%** | 365 Continuous Days/Year |
| **Strategy Organic CAGR (Anti-Cheat)** | 158.07% | 42.08% | Deflated by terminal unclosed equity |
| **Benchmark (BTC) CAGR** | 58.38% | -1.87% | Buy-and-Hold BTC |
| **Beats Index?** | PASSED | PASSED | Direct CAGR comparison |
| **Equal-Weight Filtered Basket CAGR** | 58.16% | 15.51% | Unweighted clean crypto set (clipped) |
| **Strategy TWR Max Drawdown** | **41.74%** | **21.73%** | Pure capital decline |
| **Benchmark (BTC) Max DD** | 76.63% | 52.97% | Market systemic peak-to-trough |
| **Strategy Profit Factor** | 4.62 | 3.30 | Net Wins / Net Losses |
| **Completed Trades** | 78 | 27 | Executed volume |
| **Net Realized PnL** | **$999,567.63** | **$15,789.89** | Frictions deducted |
| **TWR Beta vs BTC** | 0.27 | 0.22 | Systematic market exposure |
| **TWR Correlation vs BTC** | 0.26 | 0.23 | Co-movement correlation |
| **Up-Market Capture** | 37.3% | 44.2% | Return captured when BTC positive |
| **Down-Market Capture** | 13.3% | 25.2% | Return captured when BTC negative |

### B. Matched Placebo Noise Suite (50 Seeds Control)
| Evaluation Window | Strategy CAGR | Conditioned Noise Median | Noise [P5, P95] Range | Statistical Edge Status |
| :--- | :--- | :--- | :--- | :--- |
| **In-Sample** | **158.07%** | 22.36% | [-4.52%, 57.95%] | Empirical percentile: **100.0th** |
| **Out-of-Sample** | **60.75%** | 2.73% | [-17.19%, 29.39%] | Empirical percentile: **100.0th** |

## 2. Crypto Black Swan Stress Audit
| Historical Crash Period | Strategy Return | Strategy Max DD | BTC Return | BTC Max DD | Timed Macro Return | Active Trades |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2021 May Deleveraging Crash** | +0.00% | 0.00% | -39.89% | 49.39% | -5.17% | 0 |
| **2022 Terra/Luna & 3AC Crash** | +0.00% | 0.00% | -49.88% | 52.20% | +0.00% | 0 |
| **2022 FTX Insolvency Crisis** | -7.85% | 7.85% | -19.24% | 25.91% | -9.45% | 2 |

## 3. Macro & Trailing Stop Ablation Audits
### A. Macro-Active Exit Ablation
| Configuration | IS CAGR | OOS CAGR | IS Score | OOS Score |\n| :--- | ---: | ---: | ---: | ---: |\n| **Macro-active exit ON** | 158.07% | 60.75% | 11.7975 | 2.4040 |\n| **Counterfactual OFF** | 174.49% | 93.41% | 12.6817 | 1.3250 |\n
### B. Trailing Stop Mechanism Ablation
*Trailing Stop Ablation not applicable: The winning configuration does not employ a trailing stop mechanism.*


## 4. Top-K Walk-Forward Stability Matrix
> **Champion Promotion Diagnostic:** Champion cleared all three sequential folds against the macro benchmark.

| Candidate | Search Score | Fold 1 vs Index | Fold 2 vs Index | Fold 3 vs Index | Beats All 3 | Beats Recent | Recent Margin |\n| :--- | ---: | ---: | ---: | ---: | :---: | :---: | ---: |\n| **#1 Trial 698** | 11.7975 | +235.06% | +30.92% | +75.58% | YES | YES | +75.58% |\n| **#2 Trial 358** | 11.9415 | +242.99% | -7.34% | +106.98% | NO | YES | +106.98% |\n| **#3 Trial 513** | 11.8537 | +139.95% | -32.52% | +70.34% | NO | YES | +70.34% |\n| **#4 Trial 722** | 12.5654 | +36.01% | -21.62% | +68.64% | NO | YES | +68.64% |\n| **#5 Trial 873** | 12.5122 | +149.80% | -14.92% | +65.99% | NO | YES | +65.99% |\n| **#6 Trial 748** | 12.5361 | +147.20% | -27.12% | +64.97% | NO | YES | +64.97% |\n| **#7 Trial 794** | 12.3194 | +93.04% | -49.91% | +63.68% | NO | YES | +63.68% |\n| **#8 Trial 912** | 11.9311 | +176.61% | -22.40% | +62.90% | NO | YES | +62.90% |\n| **#9 Trial 751** | 11.8102 | +123.68% | -29.44% | +50.57% | NO | YES | +50.57% |\n| **#10 Trial 802** | 12.2867 | +57.64% | -27.24% | +45.18% | NO | YES | +45.18% |\n

## 5. Matched Placebo Percentile & Concentration Diagnostics
| Diagnostic | Result |\n| :--- | :--- |\n| Top 5 symbol profit drivers removed | DOGE, XLM, PEPE, TRX, AVAX |\n| Baseline IS CAGR | 158.07% |\n| Leave-top-5-out IS CAGR | 81.21% |\n

## 6. Exposure Telemetry
| Window | Peak Single-Asset Exposure |\n| :--- | ---: |\n| In-Sample | 66.69% |\n| Out-of-Sample | 57.36% |\n

## 7. Signal Funnel & Opportunity Attrition Matrix
| Stage in Funnel | Unique Signals | Disposition Description |
| :--- | :--- | :--- |
| **Total Raw Technical Triggers** | **226** | Close > HHV, RSI cross, or MA trigger |
| ├── Macro Gate Blocked | -19 | BTC below Macro Moving Average |
| ├── Liquidity Floor Blocked | -16 | ADV < $2,000,000.00 |
| ├── ADX Trend Blocked | -24 | ADX < Selected Threshold |
| ├── Pyramid Limit Blocked | -51 | Asset already at max layers or averaging down |
| ├── Revalidation Dropped | -1 | Failed macro/ADX/state check at fill time |
| ├── Slot Saturated Dropped | -24 | No open portfolio slots available |
| ├── Cash Starved Dropped | -5 | Cash below $50.00 |
| ├── Scrip Risk Cap Dropped | -0 | Exceeded single-asset 25% exposure ceiling |
| ├── Expired in Watchlist | -8 | Exceeded 15 bars or shadow stop |
| **Executed Trades on Ledger** | **78** | Successfully filled and audited |
| **Checksum: raw triggers - all dispositions - executed** | **0** | Expected 0 |


## 8. Trade Excursion & Timing Decay Analysis (MFE / MAE)
### In-Sample
| Metric | Median | Mean |\n| :--- | ---: | ---: |\n| MFE | 21.31% | 79.63% |\n| MAE | -11.59% | -13.80% |\n| Bars to Peak | 6.0 | 10.8 |\n
### Out-of-Sample
| Metric | Median | Mean |\n| :--- | ---: | ---: |\n| MFE | 19.05% | 36.58% |\n| MAE | -3.42% | -9.70% |\n| Bars to Peak | 2.0 | 4.9 |\n

## 9. Trade-Reason Population Breakdown
### In-Sample
| Exit Reason | Trades | Share |\n| :--- | ---: | ---: |\n| BREAKEVEN_SL | 29 | 37.2% |\n| MAX_HOLDING_TIME | 26 | 33.3% |\n| MACRO_REGIME_EXIT | 12 | 15.4% |\n| SIGNAL_EXIT_TYPE_7 | 11 | 14.1% |\n
### Out-of-Sample
| Exit Reason | Trades | Share |\n| :--- | ---: | ---: |\n| MACRO_REGIME_EXIT | 8 | 29.6% |\n| BREAKEVEN_SL | 6 | 22.2% |\n| SIGNAL_EXIT_TYPE_7 | 5 | 18.5% |\n| MAX_HOLDING_TIME | 4 | 14.8% |\n| END_OF_TEST | 4 | 14.8% |\n

## 10. Annual Pure Time-Weighted Return Attribution
| Calendar Year | Strategy TWR | BTC Return | Completed Trades | Win Rate |\n| :--- | ---: | ---: | ---: | ---: |\n| **2020** | +1.22% | +164.54% | 4 | 50.0% |\n| **2021** | +1534.70% | +57.57% | 21 | 57.1% |\n| **2022** | -19.07% | -65.34% | 7 | 0.0% |\n| **2023** | +56.92% | +154.46% | 24 | 33.3% |\n| **2024** | +242.71% | +111.81% | 21 | 42.9% |\n| **2025** | +10.11% | -7.34% | 12 | 41.7% |\n| **2026** | +37.19% | -4.92% | 16 | 50.0% |\n

## 11. Discovered Optimal Parameter Set
```json
{
    "entry_type": 3,
    "adx_thresh": 20.0,
    "vol_ma_len": 40,
    "vol_mult": 3.5,
    "price_lookback": 40,
    "body_atr_mult": 1.4,
    "use_market_macro_system": true,
    "macro_ma_len": 50,
    "macro_ma_type": 0,
    "macro_active_exit": true,
    "max_concurrent_tranches": 4,
    "max_pyramid_layers": 1,
    "wl_mode": "WL_STRONGEST_MOMENTUM",
    "use_global_tp": false,
    "be_trigger_atr": 1.5,
    "max_holding_bars": 35,
    "sl_mult": 6.0,
    "exit_type": 7,
    "trail_atr_mult": 0.0,
    "bb_exit_len": 50
}
