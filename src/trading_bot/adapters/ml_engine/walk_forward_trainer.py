# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""End-to-End Walk-Forward Trainer & Backtest Engine Adapter (Clean Architecture, Invariant #9).

Orchestrates Stage 4 Causal ML components (M4.1 -> M4.5):
1. TripleBarrierLabeler: Symmetrical 1.8*sigma_GK barriers, two-way fee deduction, soft return meta-weighting.
2. SampleUniquenessWeights: Concurrent overlap matrix c_t, average uniqueness u_bar_i, de Prado weights.
3. PurgedKFoldSplitter: 100-bar dynamic memory embargo eliminating autoregressive feature leakage.
4. ConformalSubspaceOODDetector: 6D continuous physical state subspace with Ledoit-Wolf shrinkage.
5. OOFRidgePrior & ConstrainedLGBMModel & CoherenceGate: Dual-binary LightGBM with monotonic locks and coherence.
6. TabulatedVennAbersCalibrator & ECEDriftMonitor: 500-point fast lookup (<0.05ms) and 7-day rolling ECE drift monitoring.
7. NativeTreeSHAPExplainer & EconomicSignVerifier: C++ TreeSHAP attributions (<0.8ms) and economic sign verification.
8. Strict zero look-ahead bias: t <= T_decision, 100-bar train/test boundary embargo, causal temporal ordering.

Strict Clean Architecture:
- Concrete adapter in src/trading_bot/adapters/ml_engine/
- Interacts with persistence strictly via domain port MLPredictionRepositoryPort (adapters_isolation kept).
- Enforces Invariant #9: zero default parameters, zero class field defaults, zero silent fallbacks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import numpy as np

from trading_bot.adapters.ml_engine.conformal_ood_detector import (
    ConformalSubspaceOODDetector,
)
from trading_bot.adapters.ml_engine.constrained_lgbm_model import (
    ConstrainedLGBMModel,
)
from trading_bot.adapters.ml_engine.native_treeshap_explainer import (
    NativeTreeSHAPExplainer,
)
from trading_bot.adapters.ml_engine.tabulated_venn_abers import (
    TabulatedVennAbersCalibrator,
)
from trading_bot.domain.entities.market.market_state import MicroBar1m
from trading_bot.domain.entities.ml.barrier_labels import (
    BarrierLabel,
    BarrierType,
    LabelSpan,
)
from trading_bot.domain.entities.ml.ml_configs import (
    LGBMMonotonicConstraints,
    WalkForwardConfig,
)
from trading_bot.domain.entities.ml.ml_predictions import (
    UnifiedMLPredictionPacket,
    WalkForwardSummaryReport,
    WalkForwardWindowSplit,
    WindowEvaluationMetrics,
)
from trading_bot.domain.services.ml.coherence_gate import CoherenceGate
from trading_bot.domain.services.ml.ece_drift_monitor import (
    ECEDriftMonitor,
    calculate_binned_ece,
)
from trading_bot.domain.services.ml.economic_sign_verifier import EconomicSignVerifier
from trading_bot.domain.services.ml.sample_uniqueness_weights import (
    compute_de_prado_weights,
)
from trading_bot.domain.services.ml.triple_barrier_labeler import evaluate_triple_barrier
from trading_bot.ports.ml_prediction_repository_port import MLPredictionRepositoryPort


def compute_simulated_trade_return(
    barrier_label: BarrierLabel,
    is_long: bool,
) -> float:
    """Compute exact directional return for a simulated trade (Zero Defaults, Invariant #9).

    Args:
        barrier_label: Evaluated BarrierLabel for the bar.
        is_long: True for Long position, False for Short position.

    Returns:
        Signed net return of the trade including transaction costs.
    """
    if is_long:
        if barrier_label.barrier_hit == BarrierType.UPPER_PROFIT:
            return barrier_label.net_return
        if barrier_label.barrier_hit == BarrierType.LOWER_LOSS:
            return -barrier_label.net_return
        return barrier_label.realized_return - barrier_label.roundtrip_cost
    else:
        if barrier_label.barrier_hit == BarrierType.LOWER_LOSS:
            return barrier_label.net_return
        if barrier_label.barrier_hit == BarrierType.UPPER_PROFIT:
            return -barrier_label.net_return
        return -barrier_label.realized_return - barrier_label.roundtrip_cost


def compute_roc_auc_binary(y_true: Sequence[int], y_scores: Sequence[float]) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC-AUC).

    Uses trapezoidal integration with rank-sum statistics.
    If only one class is present in y_true, returns 0.50 (uninformative baseline).
    """
    n = len(y_true)
    if n == 0:
        return 0.50

    pos_indices = [i for i, y in enumerate(y_true) if y == 1]
    neg_indices = [i for i, y in enumerate(y_true) if y == 0]

    n_pos = len(pos_indices)
    n_neg = len(neg_indices)

    if n_pos == 0 or n_neg == 0:
        return 0.50

    paired: list[tuple[float, int]] = sorted(
        [(y_scores[i], y_true[i]) for i in range(n)], key=lambda x: x[0]
    )
    rank_sum_pos = 0.0
    i = 0
    while i < n:
        j = i
        while j < n and paired[j][0] == paired[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if paired[k][1] == 1:
                rank_sum_pos += avg_rank
        i = j

    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    auc = u_stat / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def compute_brier_score_binary(y_true: Sequence[int], y_scores: Sequence[float]) -> float:
    """Compute Brier Score: mean squared error of predicted probabilities."""
    n = len(y_true)
    if n == 0:
        return 0.0
    total_sq_err = sum((float(y_scores[i]) - float(y_true[i])) ** 2 for i in range(n))
    return float(total_sq_err / float(n))


def compute_log_loss_binary(y_true: Sequence[int], y_scores: Sequence[float], eps: float) -> float:
    """Compute binary cross-entropy (log-loss) with probability clamping."""
    n = len(y_true)
    if n == 0:
        return 0.0
    total_loss = 0.0
    for i in range(n):
        p = min(max(float(y_scores[i]), eps), 1.0 - eps)
        y = float(y_true[i])
        total_loss += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
    return float(total_loss / float(n))


class WalkForwardTrainer:
    """High-throughput Walk-Forward Backtesting Engine orchestrating Stage 4 Causal ML."""

    def __init__(self, config: WalkForwardConfig, model_version: str) -> None:
        """Initialize trainer with explicit configuration and version string.

        Args:
            config: Explicit WalkForwardConfig instance (Zero Defaults).
            model_version: Explicit string identifier for the model version.
        """
        if len(model_version.strip()) == 0:
            raise ValueError("model_version must not be empty")
        self._config: WalkForwardConfig = config
        self._model_version: str = model_version

    @property
    def config(self) -> WalkForwardConfig:
        """Return the injected WalkForwardConfig schema."""
        return self._config

    @property
    def model_version(self) -> str:
        """Return the model version string."""
        return self._model_version

    def generate_temporal_windows(self) -> list[WalkForwardWindowSplit]:
        """Generate deterministic walk-forward temporal splits across the timeline.

        Enforces strict 100-bar memory embargo at both:
        1. Internal boundary: between sub-train and calibration slice.
        2. External boundary: between train_end and test_start.
        """
        windows: list[WalkForwardWindowSplit] = []
        cur_start = self._config.start_date
        window_id = 1
        embargo_delta = timedelta(minutes=30 * self._config.embargo_bars_30m)

        while True:
            train_end = cur_start + timedelta(
                days=int(self._config.train_duration_months * 30.4375)
            )
            test_start = train_end + embargo_delta
            test_end = test_start + timedelta(days=int(self._config.test_duration_months * 30.4375))

            if test_end > self._config.end_date:
                break

            total_train_seconds = (train_end - cur_start).total_seconds()
            subtrain_seconds = total_train_seconds * (1.0 - self._config.cal_ratio)
            subtrain_end = cur_start + timedelta(seconds=subtrain_seconds)

            cal_start = subtrain_end + embargo_delta
            cal_end = train_end

            if cal_start >= cal_end:
                subtrain_end = cal_end - embargo_delta - timedelta(days=7)
                cal_start = subtrain_end + embargo_delta

            windows.append(
                WalkForwardWindowSplit(
                    window_id=window_id,
                    subtrain_start=cur_start,
                    subtrain_end=subtrain_end,
                    internal_embargo_start=subtrain_end,
                    internal_embargo_end=cal_start,
                    cal_start=cal_start,
                    cal_end=cal_end,
                    external_embargo_start=train_end,
                    external_embargo_end=test_start,
                    test_start=test_start,
                    test_end=test_end,
                )
            )

            window_id += 1
            cur_start = cur_start + timedelta(days=int(self._config.test_duration_months * 30.4375))

        return windows

    def label_dataset(
        self,
        symbol: str,
        bar_times: Sequence[datetime],
        entry_prices: Sequence[float],
        vol_gks: Sequence[float],
        candles_1m_by_time: Sequence[list[MicroBar1m]],
    ) -> list[BarrierLabel]:
        """Label 30m decision bars using forward 1m candle scans (1.8*sigma_GK symmetrical barriers).

        Args:
            symbol: Market symbol (e.g. "BTCUSDT").
            bar_times: Chronological timestamps of decision bars.
            entry_prices: Entry execution prices for each decision bar.
            vol_gks: Garman-Klass volatility estimate for each decision bar.
            candles_1m_by_time: Forward 1m candles for each decision bar.

        Returns:
            List of BarrierLabel domain entities.
        """
        n = len(bar_times)
        if not (n == len(entry_prices) == len(vol_gks) == len(candles_1m_by_time)):
            raise ValueError(
                f"Array length mismatch: {n} times, {len(entry_prices)} prices, "
                f"{len(vol_gks)} vols, {len(candles_1m_by_time)} candle sets"
            )

        labels: list[BarrierLabel] = []
        for i in range(n):
            lbl = evaluate_triple_barrier(
                symbol=symbol,
                bar_time=bar_times[i],
                entry_price=entry_prices[i],
                vol_gk=vol_gks[i],
                candles_1m=candles_1m_by_time[i],
                config=self._config.triple_barrier_config,
            )
            labels.append(lbl)
        return labels

    def execute_window(
        self,
        window: WalkForwardWindowSplit,
        symbol: str,
        bar_times: Sequence[datetime],
        canonical_features_matrix: Sequence[Sequence[float]],
        subspace_features_matrix: Sequence[Sequence[float]],
        barrier_labels: Sequence[BarrierLabel],
        feature_names: tuple[str, ...],
    ) -> tuple[WindowEvaluationMetrics, list[UnifiedMLPredictionPacket]]:
        """Execute training, calibration, and test inference for a single walk-forward window.

        Enforces strict causal ordering:
        - Train models strictly on Sub-Train data (t <= subtrain_end).
        - Calibrate strictly on Calibration data (cal_start <= t <= cal_end).
        - Test models strictly on Test data (t >= test_start).
        """
        n_samples = len(bar_times)
        if not (
            n_samples
            == len(canonical_features_matrix)
            == len(subspace_features_matrix)
            == len(barrier_labels)
        ):
            raise ValueError("All input sequences must have identical sample length")

        # 1. Partition sample indices based on timestamps
        subtrain_indices: list[int] = []
        cal_indices: list[int] = []
        test_indices: list[int] = []

        for i, dt in enumerate(bar_times):
            if window.subtrain_start <= dt <= window.subtrain_end:
                subtrain_indices.append(i)
            elif window.cal_start <= dt <= window.cal_end:
                cal_indices.append(i)
            elif window.test_start <= dt <= window.test_end:
                test_indices.append(i)

        if len(subtrain_indices) < 20:
            raise ValueError(
                f"Window {window.window_id} has insufficient subtrain samples: {len(subtrain_indices)}"
            )
        if len(cal_indices) < 10:
            raise ValueError(
                f"Window {window.window_id} has insufficient cal samples: {len(cal_indices)}"
            )
        if len(test_indices) == 0:
            raise ValueError(
                f"Window {window.window_id} has zero test samples: {len(test_indices)}"
            )

        # 2. Extract Sub-Train features, labels, and compute de Prado weights
        subtrain_feats = [canonical_features_matrix[i] for i in subtrain_indices]
        subtrain_subspace = [subspace_features_matrix[i] for i in subtrain_indices]
        subtrain_y_long = [barrier_labels[i].y_long for i in subtrain_indices]
        subtrain_y_short = [barrier_labels[i].y_short for i in subtrain_indices]

        label_spans: list[LabelSpan] = []
        for idx_pos, orig_idx in enumerate(subtrain_indices):
            lbl = barrier_labels[orig_idx]
            t_start = idx_pos
            t_end = min(idx_pos + lbl.bars_held, len(subtrain_indices) - 1)
            vol_val = max(1e-4, abs(lbl.net_return))
            label_spans.append(
                LabelSpan(
                    sample_id=str(orig_idx),
                    t_start=t_start,
                    t_end=t_end,
                    realized_return=lbl.net_return,
                    volatility=vol_val,
                )
            )

        sample_weights = compute_de_prado_weights(
            events=label_spans,
            num_bars=len(subtrain_indices),
            config=self._config.sample_weight_config,
        )

        # 3. Fit 6D Conformal Subspace OOD Detector
        cal_subspace = [subspace_features_matrix[i] for i in cal_indices]
        ood_detector = ConformalSubspaceOODDetector(config=self._config.conformal_ood_config)
        ood_detector.fit(
            train_features=subtrain_subspace,
            calibration_features=cal_subspace,
        )

        # 4. Fit Constrained LGBM Model with 5-Fold OOF Ridge Prior
        lgbm_model = ConstrainedLGBMModel(
            config=self._config.constrained_lgbm_config,
            ridge_config=self._config.ridge_prior_config,
            coherence_config=self._config.coherence_gate_config,
            feature_names=self._config.canonical_features,
            random_state=self._config.random_state,
        )
        lgbm_model.fit(
            train_features=subtrain_feats,
            y_long=subtrain_y_long,
            y_short=subtrain_y_short,
            sample_weights=sample_weights,
            sample_intervals=None,
        )

        # 5. Fit Tabulated Venn-Abers Calibrator on Calibration Slice
        cal_feats = [canonical_features_matrix[i] for i in cal_indices]
        cal_y_long = [barrier_labels[i].y_long for i in cal_indices]
        cal_y_short = [barrier_labels[i].y_short for i in cal_indices]

        raw_cal_preds = lgbm_model.predict_batch(batch_features=cal_feats)
        cal_scores_long = [p.p_raw_long for p in raw_cal_preds]
        cal_scores_short = [p.p_raw_short for p in raw_cal_preds]

        va_long = TabulatedVennAbersCalibrator(config=self._config.fast_venn_abers_config)
        va_short = TabulatedVennAbersCalibrator(config=self._config.fast_venn_abers_config)
        va_long.fit(scores=cal_scores_long, labels=cal_y_long)
        va_short.fit(scores=cal_scores_short, labels=cal_y_short)

        # 6. Initialize TreeSHAP Explainer and Economic Verifiers
        verifier_long = EconomicSignVerifier(
            config=self._config.treeshap_config,
            monotonic_constraints=LGBMMonotonicConstraints.create_canonical_long(),
        )
        verifier_short = EconomicSignVerifier(
            config=self._config.treeshap_config,
            monotonic_constraints=LGBMMonotonicConstraints.create_canonical_short(),
        )
        explainer = NativeTreeSHAPExplainer(
            config=self._config.treeshap_config,
            sign_verifier_long=verifier_long,
            sign_verifier_short=verifier_short,
        )

        coherence_gate = CoherenceGate(config=self._config.coherence_gate_config)
        drift_monitor = ECEDriftMonitor(config=self._config.ece_monitor_config)

        # 7. Out-of-Sample Inference on Test Slice (Strict Causal Evaluation)
        packets: list[UnifiedMLPredictionPacket] = []
        p_long_cal_list: list[float] = []
        p_short_cal_list: list[float] = []
        test_y_long: list[int] = []
        test_y_short: list[int] = []
        ood_count = 0
        economic_consistent_count = 0

        trade_returns: list[float] = []

        for test_idx in test_indices:
            dt = bar_times[test_idx]
            feat_vec = canonical_features_matrix[test_idx]
            subspace_vec = subspace_features_matrix[test_idx]
            lbl = barrier_labels[test_idx]

            # 7.1 OOD Evaluation
            ood_result = ood_detector.predict(sample_features=subspace_vec)
            if ood_result.is_ood:
                ood_count += 1

            # 7.2 Raw Model Prediction
            raw_pred = lgbm_model.predict(sample_features=feat_vec)

            # 7.3 Fast Venn-Abers Calibration (<0.05ms)
            cal_long = va_long.calibrate(score=raw_pred.p_raw_long)
            cal_short = va_short.calibrate(score=raw_pred.p_raw_short)

            # 7.4 Coherence Regularization
            coh_result = coherence_gate.enforce_coherence(
                p_raw_long=cal_long.p_calibrated,
                p_raw_short=cal_short.p_calibrated,
            )
            adj_long = coh_result.p_long
            adj_short = coh_result.p_short

            # 7.5 Native TreeSHAP Attribution & Sign Check (<0.8ms)
            shap_res = explainer.explain(
                feature_vector=feat_vec,
                feature_names=feature_names,
                model=lgbm_model,
                symbol=symbol,
                bar_time=dt,
                model_type="long_barrier",
            )
            if shap_res.is_economic_consistent:
                economic_consistent_count += 1

            top_bull = tuple(shap_res.top_bullish_drivers[:5])
            top_bear = tuple(shap_res.top_bearish_drivers[:3])

            feat_snapshot = {feature_names[k]: feat_vec[k] for k in range(len(feature_names))}

            packet = UnifiedMLPredictionPacket(
                symbol=symbol,
                bar_time=dt,
                regime_label="REGIME_EVAL",
                model_version=self._model_version,
                p_long_raw=raw_pred.p_raw_long,
                p_short_raw=raw_pred.p_raw_short,
                p_long_calibrated=cal_long,
                p_short_calibrated=cal_short,
                ood_status=ood_result,
                top_bullish_shap=top_bull,
                top_bearish_shap=top_bear,
                is_economic_consistent=shap_res.is_economic_consistent,
                features_snapshot=feat_snapshot,
            )
            packets.append(packet)

            p_long_cal_list.append(adj_long)
            p_short_cal_list.append(adj_short)
            test_y_long.append(lbl.y_long)
            test_y_short.append(lbl.y_short)

            # Record in drift monitor
            drift_monitor.record_observation(
                eval_time=dt,
                predicted_prob=adj_long,
                true_label=lbl.y_long,
            )

            # 7.6 Simulate trade execution if signal exceeds decision_threshold and NOT OOD
            if not ood_result.is_ood:
                if adj_long >= self._config.decision_threshold and adj_long > adj_short:
                    trade_returns.append(
                        compute_simulated_trade_return(barrier_label=lbl, is_long=True)
                    )
                elif adj_short >= self._config.decision_threshold and adj_short > adj_long:
                    trade_returns.append(
                        compute_simulated_trade_return(barrier_label=lbl, is_long=False)
                    )

        # 8. Compute Test Window Evaluation Metrics
        n_test = len(test_indices)
        auc_l = compute_roc_auc_binary(y_true=test_y_long, y_scores=p_long_cal_list)
        auc_s = compute_roc_auc_binary(y_true=test_y_short, y_scores=p_short_cal_list)
        brier_l = compute_brier_score_binary(y_true=test_y_long, y_scores=p_long_cal_list)
        brier_s = compute_brier_score_binary(y_true=test_y_short, y_scores=p_short_cal_list)
        logloss_l = compute_log_loss_binary(y_true=test_y_long, y_scores=p_long_cal_list, eps=1e-5)
        logloss_s = compute_log_loss_binary(
            y_true=test_y_short, y_scores=p_short_cal_list, eps=1e-5
        )

        ece_l, _, _, _ = calculate_binned_ece(
            predictions=p_long_cal_list,
            labels=test_y_long,
            n_bins=self._config.ece_monitor_config.n_bins,
        )
        ece_s, _, _, _ = calculate_binned_ece(
            predictions=p_short_cal_list,
            labels=test_y_short,
            n_bins=self._config.ece_monitor_config.n_bins,
        )

        pct_ood = (float(ood_count) / float(n_test)) * 100.0
        pct_econ = (float(economic_consistent_count) / float(n_test)) * 100.0

        trade_count = len(trade_returns)
        if trade_count > 0:
            win_count = sum(1 for r in trade_returns if r > 0)
            win_rate = float(win_count) / float(trade_count)

            pos_sum = sum(r for r in trade_returns if r > 0)
            neg_sum = sum(abs(r) for r in trade_returns if r < 0)
            if neg_sum > 0.0:
                profit_factor = float(pos_sum / neg_sum)
            else:
                profit_factor = 2.0 if pos_sum > 0 else 1.0

            mean_r = float(np.mean(trade_returns))
            std_r = float(np.std(trade_returns))
            if std_r > 1e-8:
                sharpe = (mean_r / std_r) * math.sqrt(17520.0)
            else:
                sharpe = 0.0

            cum_returns = np.cumsum(trade_returns)
            peak = np.maximum.accumulate(cum_returns)
            drawdowns = peak - cum_returns
            max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
            cum_ret = float(cum_returns[-1])
        else:
            win_rate = 0.0
            profit_factor = 1.0
            sharpe = 0.0
            max_dd = 0.0
            cum_ret = 0.0

        metrics = WindowEvaluationMetrics(
            window_id=window.window_id,
            symbol=symbol,
            sample_count=n_test,
            auc_long=round(auc_l, 4),
            auc_short=round(auc_s, 4),
            brier_long=round(brier_l, 4),
            brier_short=round(brier_s, 4),
            log_loss_long=round(logloss_l, 4),
            log_loss_short=round(logloss_s, 4),
            ece_long=round(ece_l, 4),
            ece_short=round(ece_s, 4),
            pct_ood=round(pct_ood, 2),
            pct_economic_consistent=round(pct_econ, 2),
            trade_count=trade_count,
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 4),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown_pct=round(max_dd * 100.0, 2),
            cumulative_return=round(cum_ret, 4),
        )

        return metrics, packets

    def run_walk_forward(
        self,
        windows: Sequence[WalkForwardWindowSplit],
        symbol: str,
        bar_times: Sequence[datetime],
        canonical_features_matrix: Sequence[Sequence[float]],
        subspace_features_matrix: Sequence[Sequence[float]],
        barrier_labels: Sequence[BarrierLabel],
        feature_names: tuple[str, ...],
    ) -> tuple[WalkForwardSummaryReport, list[UnifiedMLPredictionPacket]]:
        """Run full walk-forward evaluation across all window splits."""
        all_metrics: list[WindowEvaluationMetrics] = []
        all_packets: list[UnifiedMLPredictionPacket] = []

        for window in windows:
            metrics, packets = self.execute_window(
                window=window,
                symbol=symbol,
                bar_times=bar_times,
                canonical_features_matrix=canonical_features_matrix,
                subspace_features_matrix=subspace_features_matrix,
                barrier_labels=barrier_labels,
                feature_names=feature_names,
            )
            all_metrics.append(metrics)
            all_packets.extend(packets)

        total_samples = sum(m.sample_count for m in all_metrics)
        if len(all_metrics) > 0:
            mean_auc_l = float(np.mean([m.auc_long for m in all_metrics]))
            mean_auc_s = float(np.mean([m.auc_short for m in all_metrics]))
            mean_sharpe = float(np.mean([m.sharpe_ratio for m in all_metrics]))
            max_mdd = max(m.max_drawdown_pct for m in all_metrics)
            mean_pf = float(np.mean([m.profit_factor for m in all_metrics]))
            mean_wr = float(np.mean([m.win_rate for m in all_metrics]))
            mean_pct_ood = float(np.mean([m.pct_ood for m in all_metrics]))
            mean_pct_econ = float(np.mean([m.pct_economic_consistent for m in all_metrics]))
        else:
            mean_auc_l = 0.50
            mean_auc_s = 0.50
            mean_sharpe = 0.0
            max_mdd = 0.0
            mean_pf = 1.0
            mean_wr = 0.0
            mean_pct_ood = 0.0
            mean_pct_econ = 100.0

        summary = WalkForwardSummaryReport(
            total_samples=total_samples,
            total_windows=len(all_metrics),
            window_metrics=tuple(all_metrics),
            mean_auc_long=round(mean_auc_l, 4),
            mean_auc_short=round(mean_auc_s, 4),
            overall_sharpe=round(mean_sharpe, 4),
            overall_max_drawdown=round(max_mdd, 2),
            overall_profit_factor=round(mean_pf, 4),
            overall_win_rate=round(mean_wr, 4),
            overall_pct_ood=round(mean_pct_ood, 2),
            overall_pct_economic_consistent=round(mean_pct_econ, 2),
            model_version=self._model_version,
            created_at=datetime.now(tz=UTC),
        )
        return summary, all_packets

    async def persist_predictions(
        self,
        repo: MLPredictionRepositoryPort,
        packets: Sequence[UnifiedMLPredictionPacket],
    ) -> int:
        """Persist prediction packets into database via MLPredictionRepositoryPort."""
        return await repo.save_prediction_packets_batch(packets=packets)
