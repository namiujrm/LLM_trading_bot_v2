"""Rolling Expected Calibration Error (ECE) Drift Monitor Service — Zero Defaults (Invariant #9).

Pure Python domain service implementing calibration quality tracking and drift detection (Proposal P5).
Calculates Expected Calibration Error (ECE) across M uniform probability bins:
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|

Monitors ECE over a rolling 7-day window (336 bars of 30m).
Triggers `is_drift_detected = True` when ECE exceeds `drift_warning_threshold` (typically 0.08).

Strict Clean Architecture:
- 100% Pure Python (only `datetime`, `collections.deque`, `math`, `collections.abc.Sequence`, and domain entities).
- ZERO external dependencies (NO numpy, NO scipy, NO sklearn).
- Invariant #9: zero default parameters, zero class field defaults, zero silent fallbacks.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from datetime import datetime

from trading_bot.domain.entities.ml.ml_configs import ECEMonitorConfig
from trading_bot.domain.entities.ml.ml_predictions import ECEDriftRecord


def calculate_binned_ece(
    predictions: Sequence[float],
    labels: Sequence[int],
    n_bins: int,
) -> tuple[float, tuple[float, ...], tuple[float, ...], tuple[int, ...]]:
    """Pure mathematical function to calculate binned Expected Calibration Error (ECE).

    Args:
        predictions: Sequence of predicted probabilities in [0, 1].
        labels: Sequence of ground truth binary labels in {0, 1}.
        n_bins: Number of equal-width bins spanning [0, 1].

    Returns:
        Tuple of (ece, bin_accuracies, bin_confidences, bin_counts).

    Raises:
        ValueError: If n_bins < 2, lengths differ, or values are invalid.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, received {n_bins}")
    if len(predictions) != len(labels):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(labels)} labels")

    n_samples = len(predictions)
    if n_samples == 0:
        empty_accuracies = tuple(0.0 for _ in range(n_bins))
        empty_confidences = tuple((m + 0.5) / n_bins for m in range(n_bins))
        empty_counts = tuple(0 for _ in range(n_bins))
        return 0.0, empty_accuracies, empty_confidences, empty_counts

    bin_counts: list[int] = [0 for _ in range(n_bins)]
    bin_acc_sum: list[float] = [0.0 for _ in range(n_bins)]
    bin_conf_sum: list[float] = [0.0 for _ in range(n_bins)]

    for p, y in zip(predictions, labels, strict=True):
        if not math.isfinite(p):
            raise ValueError(f"Prediction must be a finite float, received {p}")
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Prediction must be in range [0, 1], received {p}")
        if y not in (0, 1):
            raise ValueError(f"Label must be binary 0 or 1, received {y}")

        # Bin index with clamping for p == 1.0
        bin_idx = min(int(p * n_bins), n_bins - 1)
        bin_counts[bin_idx] += 1
        bin_acc_sum[bin_idx] += float(y)
        bin_conf_sum[bin_idx] += p

    accuracies: list[float] = []
    confidences: list[float] = []
    total_ece: float = 0.0

    for m in range(n_bins):
        count = bin_counts[m]
        if count > 0:
            acc = bin_acc_sum[m] / count
            conf = bin_conf_sum[m] / count
            weight = count / n_samples
            total_ece += weight * abs(acc - conf)
            accuracies.append(acc)
            confidences.append(conf)
        else:
            accuracies.append(0.0)
            confidences.append((m + 0.5) / n_bins)

    return total_ece, tuple(accuracies), tuple(confidences), tuple(bin_counts)


def compute_ece_drift_record(
    predictions: Sequence[float],
    labels: Sequence[int],
    config: ECEMonitorConfig,
    eval_time: datetime,
) -> ECEDriftRecord:
    """Pure functional entry point to compute an ECEDriftRecord.

    Args:
        predictions: Sequence of predicted probabilities in [0, 1].
        labels: Sequence of ground truth binary labels in {0, 1}.
        config: Explicit ECEMonitorConfig instance.
        eval_time: Timestamp of evaluation.

    Returns:
        ECEDriftRecord instance.
    """
    ece, accs, confs, counts = calculate_binned_ece(
        predictions=predictions,
        labels=labels,
        n_bins=config.n_bins,
    )
    sample_count = len(predictions)
    is_drift = (
        sample_count >= config.min_samples_to_evaluate and ece > config.drift_warning_threshold
    )

    return ECEDriftRecord(
        eval_time=eval_time,
        ece=ece,
        sample_count=sample_count,
        is_drift_detected=is_drift,
        bin_accuracies=accs,
        bin_confidences=confs,
        bin_counts=counts,
    )


class ECEDriftMonitor:
    """Domain Service for tracking Expected Calibration Error drift over a rolling window.

    Maintains a FIFO queue of recent (eval_time, prediction, label) observations
    bounded by `config.window_bars_30m`.
    """

    def __init__(self, config: ECEMonitorConfig) -> None:
        """Initialize ECEDriftMonitor with explicit configuration.

        Args:
            config: Explicit ECEMonitorConfig instance.
        """
        self._config: ECEMonitorConfig = config
        self._observations: deque[tuple[datetime, float, int]] = deque(
            maxlen=config.window_bars_30m
        )

    @property
    def config(self) -> ECEMonitorConfig:
        """Return the injected ECEMonitorConfig."""
        return self._config

    @property
    def current_sample_count(self) -> int:
        """Return the number of observations currently in the rolling window."""
        return len(self._observations)

    def record_observation(
        self,
        eval_time: datetime,
        predicted_prob: float,
        true_label: int,
    ) -> None:
        """Record a single prediction outcome observation into the rolling window.

        Args:
            eval_time: Timestamp of observation.
            predicted_prob: Model calibrated probability in [0, 1].
            true_label: True binary outcome in {0, 1}.

        Raises:
            ValueError: If predicted_prob is not finite or out of [0, 1], or true_label not in {0, 1}.
        """
        if not math.isfinite(predicted_prob):
            raise ValueError(f"predicted_prob must be a finite float, received {predicted_prob}")
        if not (0.0 <= predicted_prob <= 1.0):
            raise ValueError(f"predicted_prob must be in range [0, 1], received {predicted_prob}")
        if true_label not in (0, 1):
            raise ValueError(f"true_label must be 0 or 1, received {true_label}")

        self._observations.append((eval_time, predicted_prob, true_label))

    def record_batch(
        self,
        batch: Sequence[tuple[datetime, float, int]],
    ) -> None:
        """Record a batch of observations into the rolling window.

        Args:
            batch: Sequence of (eval_time, predicted_prob, true_label) tuples.
        """
        for t, p, y in batch:
            self.record_observation(eval_time=t, predicted_prob=p, true_label=y)

    def evaluate_drift(self, eval_time: datetime) -> ECEDriftRecord:
        """Evaluate calibration drift across observations currently in the rolling window.

        Args:
            eval_time: Timestamp of evaluation.

        Returns:
            ECEDriftRecord instance.
        """
        preds = [obs[1] for obs in self._observations]
        labels = [obs[2] for obs in self._observations]
        return compute_ece_drift_record(
            predictions=preds,
            labels=labels,
            config=self._config,
            eval_time=eval_time,
        )

    def clear(self) -> None:
        """Clear all stored observations from the rolling window."""
        self._observations.clear()
