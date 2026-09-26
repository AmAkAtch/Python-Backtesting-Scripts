# Crypto Quantitative Strategy Audit Dossier (V10.2-INSTITUTIONAL) — TOP_CRYPTO_LIQUID (Top 50)

## 1. Segregated Performance Accounting (Pure 365-Day Continuous TWR)

### A. Performance Baseline & Generalization Check
| Performance Metric | In-Sample (1769 days) | Out-of-Sample (590 days, Fresh Seed) | Generalization Assessment |
| :--- | :--- | :--- | :--- |
| **Strategy Annualized TWR (CAGR)** | **138.16%** | **29.56%** | 365 Continuous Days/Year |
| **Strategy Organic CAGR (Anti-Cheat)** | 138.16% | 24.97% | Deflated by terminal unclosed equity |
| **Benchmark (BTC) CAGR** | 72.24% | -8.78% | Buy-and-Hold BTC |
| **Equal-Weight Filtered Basket CAGR** | -79.66% | -81.54% | Unweighted clean crypto set (clipped) |
| **Strategy TWR Max Drawdown** | **29.83%** | **17.80%** | Pure capital decline |
| **Benchmark (BTC) Max DD** | 76.63% | 53.06% | Market systemic peak-to-trough |
| **Strategy Profit Factor** | 6.61 | 2.22 | Net Wins / Net Losses |
| **Completed Trades** | 125 | 52 | Executed volume |
| **Net Realized PnL** | **$1,085,512.48** | **$7,748.50** | Frictions deducted |
| **TWR Beta vs BTC** | 0.27 | 0.18 | Systematic market exposure |
| **TWR Correlation vs BTC** | 0.28 | 0.22 | Co-movement correlation |
| **Up-Market Capture** | 34.4% | 34.6% | Return captured when BTC positive |
| **Down-Market Capture** | 13.1% | 23.8% | Return captured when BTC negative |

### B. Matched Placebo Noise Suite (50 Seeds Control)
| Evaluation Window | Strategy CAGR | Conditioned Noise Median | Noise [P5, P95] Range | Statistical Edge Status |
| :--- | :--- | :--- | :--- | :--- |
| **In-Sample** | **138.16%** | 13.51% | [-0.82%, 32.46%] | CONFIRMED (> P95) |
| **Out-of-Sample** | **29.56%** | 1.83% | [-13.01%, 40.89%] | INSIDE NOISE BAND |

## 2. Crypto Black Swan Stress Audit
| Historical Crash Period | Strategy Return | Strategy Max DD | BTC Return | BTC Max DD | Timed Macro Return | Active Trades |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2021 May Deleveraging Crash** | -14.77% | 21.81% | -39.88% | 49.31% | -39.88% | 4 |
| **2022 Terra/Luna & 3AC Crash** | +0.00% | 0.00% | -49.91% | 52.09% | -49.91% | 0 |
| **2022 FTX Insolvency Crisis** | -14.55% | 14.55% | -19.22% | 25.82% | -19.22% | 4 |

## 3. Trailing Stop Mechanism Ablation
*Trailing Stop Ablation not applicable: The winning configuration does not employ a trailing stop mechanism.*

## 4. Signal Funnel & Opportunity Attrition Matrix
| Stage in Funnel | Unique Signals | Disposition Description |
| :--- | :--- | :--- |
| **Total Raw Technical Triggers** | **260** | Close > HHV, RSI cross, or MA trigger |
| ├── Macro Gate Blocked | -0 | BTC below Macro Moving Average |
| ├── Liquidity Floor Blocked | -56 | ADV < $2,000,000.00 |
| ├── ADX Trend Blocked | -0 | ADX < Selected Threshold |
| ├── Pyramid Limit Blocked | -25 | Asset already at max layers or averaging down |
| ├── Revalidation Dropped | -0 | Failed macro/ADX/state check at fill time |
| ├── Slot Saturated Dropped | -50 | No open portfolio slots available |
| ├── Cash Starved Dropped | -2 | Cash below $50.00 |
| ├── Scrip Risk Cap Dropped | -2 | Exceeded single-asset 25% exposure ceiling |
| ├── Expired in Watchlist | -0 | Exceeded 15 bars or shadow stop |
| **Executed Trades on Ledger** | **125** | Successfully filled and audited |

## 5. Discovered Optimal Parameter Set
```json
{
    "entry_type": 3,
    "adx_thresh": 15.0,
    "vol_ma_len": 24,
    "vol_mult": 3.5,
    "price_lookback": 38,
    "body_atr_mult": 1.3,
    "use_market_macro_system": false,
    "max_concurrent_tranches": 7,
    "max_pyramid_layers": 2,
    "wl_mode": "WL_NONE",
    "use_global_tp": false,
    "be_trigger_atr": 1.5,
    "max_holding_bars": 30,
    "sl_mult": 4.4,
    "exit_type": 7,
    "trail_atr_mult": 0.0,
    "bb_exit_len": 44,
    "macro_active_exit": false
}
