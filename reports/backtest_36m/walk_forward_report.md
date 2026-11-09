# Milestone 2: 36-Month 4-Pair Walk-Forward Empirical Backtest Report

**Execution Period**: 2023-01-01 00:00:00 to 2025-12-31 23:30:00 UTC (36 Months, 52,608 30m bars per symbol)  
**Evaluation Architecture**: 8 Rolling Windows (12M Sub-Train / 3M Test, 100-Bar Dynamic Memory Embargo)  
**Pairs Evaluated**: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT (139,808 Out-of-Sample Decision Packets)  
**Date Generated**: 2026-11-09 09:34:36 UTC  

---

## 1. Executive Portfolio Performance Summary

| Metric | Portfolio Aggregate | BTCUSDT | ETHUSDT | SOLUSDT | BNBUSDT |
|---|---|---|---|---|---|
| **Sharpe Ratio (Ann. sqrt(17520))** | **-2.1968** | -1.0120 | -3.8244 | 2.5773 | -7.1146 |
| **Win Rate** | **47.84%** | 48.98% | 48.48% | 50.89% | 45.71% |
| **Profit Factor** | **0.9608** | 1.0379 | 1.1208 | 1.0543 | 0.9543 |
| **Max Drawdown (%)** | **309.72%** | 34.39% | 42.30% | 272.14% | 65.68% |
| **Total Executed Trades** | **11,711** | 1,439 | 1,012 | 7,663 | 1,597 |
| **Mean ROC-AUC (Long / Short)** | **0.5031 / 0.5092** | 0.5090 / 0.5129 | 0.5014 / 0.5124 | 0.4991 / 0.5054 | 0.5027 / 0.5061 |
| **Mean Binned ECE (Long / Short)** | **0.0264 / 0.0206** | 0.0278 / 0.0198 | 0.0210 / 0.0177 | 0.0269 / 0.0216 | 0.0300 / 0.0233 |
| **OOD Rejection Rate (%)** | **1.15%** | 1.11% | 1.12% | 1.26% | 1.09% |
| **Economic Consistency Rate (%)** | **1.25%** | 2.51% | 1.16% | 1.12% | 0.22% |

---

## 2. Walk-Forward Window Performance per Symbol

### 2.1 BTCUSDT Window Breakdown

| Win ID | Test Bars | Trades | Win Rate | Profit Factor | Sharpe Ratio | Max DD (%) | Cum Return | AUC (L / S) | ECE (L / S) | OOD (%) | Econ (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Win 1 | 4,369 | 908 | 49.34% | 0.98 | -0.99 | 34.39% | -0.0595 | 0.510 / 0.498 | 0.0314 / 0.0138 | 1.05% | 3.4% |
| Win 2 | 4,369 | 278 | 47.48% | 1.00 | 0.04 | 10.54% | 0.0006 | 0.512 / 0.529 | 0.0556 / 0.0262 | 1.35% | 6.4% |
| Win 3 | 4,369 | 48 | 64.58% | 1.88 | 38.46 | 3.60% | 0.1013 | 0.498 / 0.510 | 0.0217 / 0.0249 | 0.76% | 0.2% |
| Win 4 | 4,369 | 16 | 37.50% | 0.73 | -19.27 | 2.96% | -0.0143 | 0.525 / 0.533 | 0.0290 / 0.0362 | 1.30% | 1.7% |
| Win 5 | 4,369 | 109 | 44.04% | 0.86 | -8.62 | 12.77% | -0.0562 | 0.499 / 0.496 | 0.0172 / 0.0132 | 1.46% | 1.7% |
| Win 6 | 4,369 | 35 | 60.00% | 1.09 | 5.16 | 4.56% | 0.0069 | 0.514 / 0.505 | 0.0049 / 0.0172 | 1.21% | 0.3% |
| Win 7 | 4,369 | 23 | 43.48% | 0.57 | -32.92 | 2.47% | -0.0273 | 0.506 / 0.521 | 0.0373 / 0.0077 | 0.76% | 0.6% |
| Win 8 | 4,369 | 22 | 45.45% | 1.19 | 10.05 | 3.53% | 0.0117 | 0.508 / 0.510 | 0.0255 / 0.0194 | 0.98% | 5.9% |

### 2.2 ETHUSDT Window Breakdown

| Win ID | Test Bars | Trades | Win Rate | Profit Factor | Sharpe Ratio | Max DD (%) | Cum Return | AUC (L / S) | ECE (L / S) | OOD (%) | Econ (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Win 1 | 4,369 | 78 | 50.00% | 1.14 | 7.61 | 12.58% | 0.0538 | 0.498 / 0.515 | 0.0072 / 0.0044 | 0.85% | 0.7% |
| Win 2 | 4,369 | 118 | 53.39% | 1.23 | 10.99 | 11.92% | 0.1038 | 0.492 / 0.521 | 0.0457 / 0.0351 | 1.46% | 1.3% |
| Win 3 | 4,369 | 21 | 23.81% | 0.19 | -97.98 | 15.37% | -0.1410 | 0.505 / 0.501 | 0.0128 / 0.0163 | 1.05% | 1.0% |
| Win 4 | 4,369 | 219 | 43.84% | 0.83 | -11.08 | 21.04% | -0.1580 | 0.508 / 0.522 | 0.0236 / 0.0187 | 1.26% | 0.5% |
| Win 5 | 4,369 | 25 | 60.00% | 1.79 | 34.31 | 4.03% | 0.0734 | 0.515 / 0.515 | 0.0163 / 0.0112 | 1.12% | 2.7% |
| Win 6 | 4,369 | 373 | 48.26% | 0.97 | -1.73 | 42.30% | -0.0471 | 0.514 / 0.513 | 0.0195 / 0.0292 | 1.19% | 1.3% |
| Win 7 | 4,369 | 167 | 44.91% | 0.79 | -14.13 | 19.26% | -0.1345 | 0.484 / 0.509 | 0.0288 / 0.0160 | 0.94% | 1.7% |
| Win 8 | 4,369 | 11 | 63.64% | 2.02 | 41.41 | 1.33% | 0.0325 | 0.495 / 0.503 | 0.0138 / 0.0108 | 1.10% | 0.2% |

### 2.3 SOLUSDT Window Breakdown

| Win ID | Test Bars | Trades | Win Rate | Profit Factor | Sharpe Ratio | Max DD (%) | Cum Return | AUC (L / S) | ECE (L / S) | OOD (%) | Econ (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Win 1 | 4,369 | 699 | 51.79% | 1.13 | 6.98 | 55.70% | 0.6177 | 0.513 / 0.493 | 0.0218 / 0.0156 | 1.81% | 0.1% |
| Win 2 | 4,369 | 4,279 | 46.09% | 0.88 | -6.86 | 272.14% | -2.3027 | 0.500 / 0.498 | 0.0599 / 0.0481 | 1.21% | 0.0% |
| Win 3 | 4,369 | 1,490 | 48.66% | 1.06 | 2.95 | 64.86% | 0.3710 | 0.495 / 0.508 | 0.0153 / 0.0218 | 1.24% | 0.2% |
| Win 4 | 4,369 | 156 | 53.85% | 1.13 | 6.91 | 17.98% | 0.0894 | 0.496 / 0.522 | 0.0243 / 0.0117 | 1.10% | 3.2% |
| Win 5 | 4,369 | 175 | 51.43% | 0.93 | -4.19 | 28.72% | -0.1095 | 0.500 / 0.500 | 0.0124 / 0.0241 | 1.19% | 0.4% |
| Win 6 | 4,369 | 730 | 47.95% | 0.95 | -3.06 | 55.98% | -0.1667 | 0.506 / 0.509 | 0.0216 / 0.0314 | 0.80% | 3.1% |
| Win 7 | 4,369 | 109 | 51.38% | 1.09 | 5.18 | 14.07% | 0.0390 | 0.475 / 0.506 | 0.0434 / 0.0018 | 0.82% | 0.4% |
| Win 8 | 4,369 | 25 | 56.00% | 1.27 | 12.71 | 4.46% | 0.0318 | 0.508 / 0.508 | 0.0166 / 0.0181 | 1.90% | 1.6% |

### 2.4 BNBUSDT Window Breakdown

| Win ID | Test Bars | Trades | Win Rate | Profit Factor | Sharpe Ratio | Max DD (%) | Cum Return | AUC (L / S) | ECE (L / S) | OOD (%) | Econ (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Win 1 | 4,369 | 564 | 53.72% | 1.14 | 7.67 | 27.28% | 0.3503 | 0.513 / 0.510 | 0.0048 / 0.0080 | 1.17% | 0.5% |
| Win 2 | 4,369 | 214 | 45.79% | 1.01 | 0.29 | 9.36% | 0.0046 | 0.514 / 0.513 | 0.0340 / 0.0234 | 0.39% | 0.4% |
| Win 3 | 4,369 | 268 | 41.79% | 0.61 | -25.65 | 65.68% | -0.6343 | 0.477 / 0.501 | 0.0469 / 0.0183 | 1.44% | 0.3% |
| Win 4 | 4,369 | 50 | 50.00% | 1.09 | 4.99 | 5.78% | 0.0177 | 0.491 / 0.517 | 0.0107 / 0.0109 | 0.96% | 0.2% |
| Win 5 | 4,369 | 26 | 50.00% | 1.63 | 26.12 | 3.60% | 0.0502 | 0.500 / 0.496 | 0.0236 / 0.0097 | 1.26% | 0.3% |
| Win 6 | 4,369 | 48 | 33.33% | 0.38 | -56.07 | 11.73% | -0.0992 | 0.552 / 0.497 | 0.0211 / 0.0436 | 1.88% | 0.1% |
| Win 7 | 4,369 | 144 | 44.44% | 0.79 | -13.66 | 14.73% | -0.0989 | 0.473 / 0.508 | 0.0588 / 0.0255 | 0.78% | 0.0% |
| Win 8 | 4,369 | 283 | 46.64% | 0.99 | -0.60 | 35.00% | -0.0128 | 0.500 / 0.505 | 0.0398 / 0.0469 | 0.87% | 0.1% |

---

## 3. Global Causal Attribution Hierarchy (TreeSHAP Driver Importance)

The table displays the empirical global importance ranking (mean absolute SHAP $|\phi|$ in logit margin space) evaluated natively across all out-of-sample decision bars:

| Rank | BTCUSDT Feature (Mean |SHAP|) | ETHUSDT Feature (Mean |SHAP|) | SOLUSDT Feature (Mean |SHAP|) | BNBUSDT Feature (Mean |SHAP|) |
|---|---|---|---|---|
| 1 | `vpin_toxicity_pct` (0.03200) | `btc_lead_catchup_gap` (0.06085) | `kyle_lambda_log` (0.04475) | `shannon_entropy_50` (0.03463) |
| 2 | `kyle_lambda_log` (0.02747) | `shannon_entropy_50` (0.03081) | `btc_lead_catchup_gap` (0.04042) | `btc_lead_catchup_gap` (0.02927) |
| 3 | `shannon_entropy_50` (0.02445) | `taker_cvd_z_24` (0.01845) | `taker_cvd_z_24` (0.02752) | `hurst_exponent_100` (0.02442) |
| 4 | `hurst_exponent_100` (0.02226) | `hurst_exponent_100` (0.01836) | `hurst_exponent_100` (0.02697) | `vpin_toxicity_pct` (0.02417) |
| 5 | `taker_cvd_z_24` (0.01703) | `kyle_lambda_log` (0.01797) | `vpin_toxicity_pct` (0.02504) | `taker_cvd_z_24` (0.02076) |
| 6 | `oi_directional_score` (0.01187) | `vpin_toxicity_pct` (0.01558) | `oi_directional_score` (0.02248) | `kyle_lambda_log` (0.01499) |
| 7 | `fracdiff_momentum_z` (0.00885) | `fracdiff_momentum_z` (0.01362) | `shannon_entropy_50` (0.01728) | `fracdiff_momentum_z` (0.01149) |
| 8 | `funding_crowded_pos_z` (0.00623) | `oi_directional_score` (0.01131) | `fracdiff_momentum_z` (0.01378) | `ofi_imbalance_z` (0.00789) |
| 9 | `ofi_imbalance_z` (0.00592) | `ofi_imbalance_z` (0.00788) | `ofi_imbalance_z` (0.00856) | `funding_crowded_pos_z` (0.00660) |
| 10 | `funding_crowded_neg_z` (0.00132) | `funding_crowded_pos_z` (0.00599) | `funding_crowded_pos_z` (0.00553) | `oi_directional_score` (0.00622) |
| 11 | `btc_lead_catchup_gap` (0.00000) | `funding_crowded_neg_z` (0.00043) | `funding_crowded_neg_z` (0.00260) | `funding_crowded_neg_z` (0.00134) |

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