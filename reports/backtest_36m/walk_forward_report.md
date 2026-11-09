# Milestone 2: 36-Month 4-Pair Walk-Forward Empirical Backtest Report

**Execution Period**: 2023-01-01 00:00:00 to 2025-12-31 23:30:00 UTC (36 Months, 52,608 30m bars per symbol)  
**Evaluation Architecture**: 8 Rolling Windows (12M Sub-Train / 3M Test, 100-Bar Dynamic Memory Embargo)  
**Pairs Evaluated**: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT (139,808 Out-of-Sample Decision Packets)  
**Date Generated**: 2026-11-09 05:49:40 UTC  

---

## 1. Executive Portfolio Performance Summary

| Metric | Portfolio Aggregate | BTCUSDT | ETHUSDT | SOLUSDT | BNBUSDT |
|---|---|---|---|---|---|
| **Sharpe Ratio (Ann. sqrt(17520))** | **-2.4509** | -19.4618 | 2.8040 | -6.3439 | -5.5538 |
| **Win Rate** | **46.98%** | 44.18% | 57.13% | 45.93% | 44.67% |
| **Profit Factor** | **0.9548** | 0.7369 | 1.2233 | 0.9332 | 0.9390 |
| **Max Drawdown (%)** | **124.77%** | 26.92% | 26.72% | 48.54% | 36.73% |
| **Total Executed Trades** | **5,055** | 999 | 773 | 2,359 | 924 |
| **Mean ROC-AUC (Long / Short)** | **0.5091 / 0.5183** | 0.5188 / 0.5275 | 0.5092 / 0.5156 | 0.5012 / 0.5093 | 0.5072 / 0.5208 |
| **Mean Binned ECE (Long / Short)** | **0.0217 / 0.0212** | 0.0263 / 0.0210 | 0.0223 / 0.0208 | 0.0190 / 0.0215 | 0.0195 / 0.0216 |
| **OOD Rejection Rate (%)** | **1.15%** | 1.11% | 1.12% | 1.26% | 1.09% |
| **Economic Consistency Rate (%)** | **1.74%** | 3.72% | 1.63% | 1.09% | 0.54% |

---

## 2. Walk-Forward Window Performance per Symbol

### 2.1 BTCUSDT Window Breakdown

| Win ID | Test Bars | Trades | Win Rate | Profit Factor | Sharpe Ratio | Max DD (%) | Cum Return | AUC (L / S) | ECE (L / S) | OOD (%) | Econ (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Win 1 | 4,369 | 129 | 56.59% | 1.13 | 6.82 | 14.59% | 0.0688 | 0.528 / 0.518 | 0.0146 / 0.0107 | 1.05% | 8.6% |
| Win 2 | 4,369 | 344 | 43.31% | 0.79 | -13.65 | 26.92% | -0.2590 | 0.521 / 0.555 | 0.0477 / 0.0243 | 1.35% | 5.7% |
| Win 3 | 4,369 | 136 | 48.53% | 0.94 | -3.18 | 19.50% | -0.0385 | 0.503 / 0.512 | 0.0303 / 0.0155 | 0.76% | 1.1% |
| Win 4 | 4,369 | 45 | 28.89% | 0.44 | -46.04 | 10.11% | -0.1031 | 0.518 / 0.541 | 0.0232 / 0.0370 | 1.30% | 2.7% |
| Win 5 | 4,369 | 260 | 41.92% | 0.84 | -9.27 | 21.22% | -0.1377 | 0.518 / 0.506 | 0.0196 / 0.0355 | 1.46% | 0.9% |
| Win 6 | 4,369 | 21 | 42.86% | 0.55 | -33.18 | 3.72% | -0.0278 | 0.544 / 0.512 | 0.0098 / 0.0216 | 1.21% | 0.3% |
| Win 7 | 4,369 | 6 | 50.00% | 0.59 | -30.36 | 0.32% | -0.0045 | 0.510 / 0.541 | 0.0310 / 0.0098 | 0.76% | 2.5% |
| Win 8 | 4,369 | 58 | 41.38% | 0.62 | -26.83 | 11.12% | -0.0859 | 0.509 / 0.535 | 0.0339 / 0.0140 | 0.98% | 8.0% |

### 2.2 ETHUSDT Window Breakdown

| Win ID | Test Bars | Trades | Win Rate | Profit Factor | Sharpe Ratio | Max DD (%) | Cum Return | AUC (L / S) | ECE (L / S) | OOD (%) | Econ (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Win 1 | 4,369 | 60 | 53.33% | 1.27 | 13.21 | 11.74% | 0.0699 | 0.509 / 0.531 | 0.0144 / 0.0026 | 0.85% | 2.7% |
| Win 2 | 4,369 | 48 | 52.08% | 1.10 | 4.70 | 7.96% | 0.0215 | 0.506 / 0.533 | 0.0478 / 0.0307 | 1.46% | 2.5% |
| Win 3 | 4,369 | 91 | 35.16% | 0.56 | -31.00 | 26.72% | -0.2476 | 0.518 / 0.498 | 0.0277 / 0.0318 | 1.05% | 1.1% |
| Win 4 | 4,369 | 68 | 50.00% | 1.04 | 2.11 | 4.84% | 0.0095 | 0.513 / 0.523 | 0.0231 / 0.0191 | 1.26% | 0.8% |
| Win 5 | 4,369 | 192 | 48.44% | 1.11 | 5.29 | 11.80% | 0.0857 | 0.501 / 0.500 | 0.0164 / 0.0302 | 1.12% | 0.3% |
| Win 6 | 4,369 | 1 | 100.00% | 2.00 | 0.00 | 0.00% | 0.0146 | 0.511 / 0.505 | 0.0167 / 0.0360 | 1.19% | 1.1% |
| Win 7 | 4,369 | 283 | 48.06% | 0.91 | -5.46 | 16.65% | -0.0883 | 0.504 / 0.519 | 0.0216 / 0.0105 | 0.94% | 1.6% |
| Win 8 | 4,369 | 30 | 70.00% | 1.80 | 33.58 | 7.28% | 0.0978 | 0.511 / 0.515 | 0.0105 / 0.0055 | 1.10% | 2.9% |

### 2.3 SOLUSDT Window Breakdown

| Win ID | Test Bars | Trades | Win Rate | Profit Factor | Sharpe Ratio | Max DD (%) | Cum Return | AUC (L / S) | ECE (L / S) | OOD (%) | Econ (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Win 1 | 4,369 | 126 | 43.65% | 1.18 | 8.78 | 15.90% | 0.1543 | 0.500 / 0.501 | 0.0297 / 0.0266 | 1.81% | 0.0% |
| Win 2 | 4,369 | 14 | 42.86% | 0.56 | -33.79 | 4.97% | -0.0449 | 0.510 / 0.521 | 0.0473 / 0.0372 | 1.21% | 0.0% |
| Win 3 | 4,369 | 873 | 48.68% | 1.00 | 0.18 | 43.62% | 0.0132 | 0.499 / 0.509 | 0.0160 / 0.0170 | 1.24% | 0.0% |
| Win 4 | 4,369 | 107 | 51.40% | 1.03 | 1.80 | 13.52% | 0.0164 | 0.507 / 0.522 | 0.0160 / 0.0118 | 1.10% | 2.7% |
| Win 5 | 4,369 | 347 | 48.70% | 0.96 | -2.33 | 40.87% | -0.1013 | 0.490 / 0.493 | 0.0177 / 0.0222 | 1.19% | 3.8% |
| Win 6 | 4,369 | 640 | 47.19% | 0.93 | -4.18 | 48.54% | -0.1952 | 0.506 / 0.511 | 0.0184 / 0.0379 | 0.80% | 0.2% |
| Win 7 | 4,369 | 40 | 35.00% | 0.49 | -37.28 | 11.56% | -0.0956 | 0.491 / 0.512 | 0.0017 / 0.0100 | 0.82% | 0.1% |
| Win 8 | 4,369 | 212 | 50.00% | 1.31 | 16.07 | 11.17% | 0.2882 | 0.505 / 0.504 | 0.0051 / 0.0091 | 1.90% | 2.0% |

### 2.4 BNBUSDT Window Breakdown

| Win ID | Test Bars | Trades | Win Rate | Profit Factor | Sharpe Ratio | Max DD (%) | Cum Return | AUC (L / S) | ECE (L / S) | OOD (%) | Econ (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Win 1 | 4,369 | 380 | 48.16% | 0.91 | -5.46 | 36.73% | -0.1787 | 0.532 / 0.511 | 0.0104 / 0.0146 | 1.17% | 2.1% |
| Win 2 | 4,369 | 75 | 45.33% | 1.13 | 6.86 | 7.62% | 0.0439 | 0.522 / 0.516 | 0.0290 / 0.0341 | 0.39% | 0.5% |
| Win 3 | 4,369 | 43 | 41.86% | 0.66 | -22.47 | 19.41% | -0.1232 | 0.497 / 0.512 | 0.0311 / 0.0061 | 1.44% | 0.5% |
| Win 4 | 4,369 | 71 | 40.85% | 0.67 | -23.79 | 10.98% | -0.0918 | 0.501 / 0.536 | 0.0106 / 0.0168 | 0.96% | 0.1% |
| Win 5 | 4,369 | 105 | 47.62% | 1.12 | 6.33 | 8.09% | 0.0451 | 0.505 / 0.515 | 0.0189 / 0.0094 | 1.26% | 0.2% |
| Win 6 | 4,369 | 63 | 38.10% | 0.64 | -26.21 | 9.77% | -0.0749 | 0.513 / 0.514 | 0.0059 / 0.0529 | 1.88% | 0.2% |
| Win 7 | 4,369 | 2 | 50.00% | 1.47 | 25.01 | 0.00% | 0.0024 | 0.483 / 0.536 | 0.0310 / 0.0138 | 0.78% | 0.1% |
| Win 8 | 4,369 | 185 | 45.41% | 0.92 | -4.70 | 17.37% | -0.0597 | 0.504 / 0.525 | 0.0187 / 0.0255 | 0.87% | 0.6% |

---

## 3. Global Causal Attribution Hierarchy (TreeSHAP Driver Importance)

The table displays the empirical global importance ranking (mean absolute SHAP $|\phi|$ in logit margin space) evaluated natively across all out-of-sample decision bars:

| Rank | BTCUSDT Feature (Mean |SHAP|) | ETHUSDT Feature (Mean |SHAP|) | SOLUSDT Feature (Mean |SHAP|) | BNBUSDT Feature (Mean |SHAP|) |
|---|---|---|---|---|
| 1 | `vpin_toxicity_pct` (0.03594) | `btc_lead_catchup_gap` (0.05195) | `btc_lead_catchup_gap` (0.04030) | `btc_lead_catchup_gap` (0.03011) |
| 2 | `kyle_lambda_log` (0.02970) | `shannon_entropy_50` (0.02584) | `taker_cvd_z_24` (0.03427) | `shannon_entropy_50` (0.02908) |
| 3 | `shannon_entropy_50` (0.02487) | `vpin_toxicity_pct` (0.02194) | `kyle_lambda_log` (0.03293) | `vpin_toxicity_pct` (0.02532) |
| 4 | `hurst_exponent_100` (0.01933) | `taker_cvd_z_24` (0.02194) | `fracdiff_momentum_z` (0.02720) | `hurst_exponent_100` (0.02282) |
| 5 | `taker_cvd_z_24` (0.01759) | `kyle_lambda_log` (0.02005) | `hurst_exponent_100` (0.02577) | `taker_cvd_z_24` (0.02050) |
| 6 | `fracdiff_momentum_z` (0.01132) | `hurst_exponent_100` (0.01554) | `vpin_toxicity_pct` (0.02417) | `kyle_lambda_log` (0.01915) |
| 7 | `ofi_imbalance_z` (0.00619) | `fracdiff_momentum_z` (0.00981) | `shannon_entropy_50` (0.01843) | `fracdiff_momentum_z` (0.01248) |
| 8 | `funding_crowded_pos_z` (0.00563) | `ofi_imbalance_z` (0.00979) | `ofi_imbalance_z` (0.00927) | `ofi_imbalance_z` (0.00921) |
| 9 | `funding_crowded_neg_z` (0.00212) | `funding_crowded_pos_z` (0.00650) | `funding_crowded_pos_z` (0.00329) | `funding_crowded_pos_z` (0.00564) |
| 10 | `oi_directional_score` (0.00000) | `funding_crowded_neg_z` (0.00257) | `funding_crowded_neg_z` (0.00088) | `funding_crowded_neg_z` (0.00201) |
| 11 | `btc_lead_catchup_gap` (0.00000) | `oi_directional_score` (0.00000) | `oi_directional_score` (0.00000) | `oi_directional_score` (0.00000) |

---

## 4. Methodological Invariants Verified

1. **Zero Look-Ahead Bias**: Causal temporal ordering enforced. Sub-train (t <= subtrain_end) and Calibration (cal_start <= t <= cal_end) separated by 100-bar memory embargo. Test evaluation strictly at t >= test_start.
2. **Invariant #9 Compliance**: 100% AST validated. Zero default parameters in functions, zero default fields in classes, zero silent fallbacks.
3. **Dual-Booster Monotonic Constraints**: Applied economic signs (+1/-1) across all 11 canonical features, preventing noise overfitting.
4. **Conformal Subspace OOD Filter**: Evaluated continuous 6D order flow and volatility state with Ledoit-Wolf regularized covariance. Filtered anomalous market conditions before execution.
5. **Venn-Abers Calibration & Coherence Gate**: Pre-computed isotonic regression grids mapped raw scores into calibrated probabilities guaranteed to satisfy P_long + P_short <= 1.0.

## 5. Next Steps for Milestone 3 (Lead Quant Researcher Evaluation)

- Audit individual pair risk-adjusted return profiles against production targets (Sharpe >= 1.8, Max DD <= 12%).
- Formulate mathematical hyperparameter and policy optimizations if any pair underperforms.
- Proceed to Phase 4 Final Empirical Synthesis.