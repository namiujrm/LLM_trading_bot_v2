"""Unit tests for Pure Python ECE Drift Monitor Service (Clean Architecture & Invariant #9).

Verifies:
- calculate_binned_ece pure mathematical function correctness (perfect calibration, miscalibration, empty bins).
- compute_ece_drift_record entrypoint and ECEDriftRecord invariant validation.
- ECEDriftMonitor rolling window FIFO buffer, drift threshold alerts, and burn-in sample guards.
- 100% Domain Layer Purity (zero external computational imports).
"""

from __future__ import annotations

import math
import sys
from datetime import UTC, datetime

import pytest

from trading_bot.domain.entities.ml.ml_configs import ECEMonitorConfig
from trading_bot.domain.entities.ml.ml_predictions import ECEDriftRecord
from trading_bot.domain.services.ml.ece_drift_monitor import (
    ECEDriftMonitor,
    calculate_binned_ece,
    compute_ece_drift_record,
)


def test_calculate_binned_ece_perfect_calibration() -> None:
    """Verifies that perfectly calibrated predictions yield an ECE of 0.0."""
    # Bin 0: p=0.0, y=0 -> acc=0.0, conf=0.0
    # Bin 9: p=1.0, y=1 -> acc=1.0, conf=1.0
    preds: list[float] = [0.0, 0.0, 1.0, 1.0]
    labels: list[int] = [0, 0, 1, 1]
    n_bins: int = 10

    ece, accs, confs, counts = calculate_binned_ece(
        predictions=preds,
        labels=labels,
        n_bins=n_bins,
    )

    assert math.isclose(ece, 0.0, abs_tol=1e-9)
    assert counts[0] == 2
    assert counts[9] == 2
    assert accs[0] == 0.0
    assert confs[0] == 0.0
    assert accs[9] == 1.0
    assert confs[9] == 1.0


def test_calculate_binned_ece_complete_miscalibration() -> None:
    """Verifies that maximally miscalibrated predictions yield an ECE of 1.0."""
    # Predict 1.0 with certainty, all labels are 0
    preds: list[float] = [1.0] * 10
    labels: list[int] = [0] * 10
    n_bins: int = 10

    ece, accs, confs, counts = calculate_binned_ece(
        predictions=preds,
        labels=labels,
        n_bins=n_bins,
    )

    assert math.isclose(ece, 1.0, abs_tol=1e-9)
    assert counts[9] == 10
    assert accs[9] == 0.0
    assert confs[9] == 1.0


def test_calculate_binned_ece_empty_input() -> None:
    """Verifies that an empty input yields ECE 0.0 and zero counts without error."""
    ece, accs, confs, counts = calculate_binned_ece(
        predictions=[],
        labels=[],
        n_bins=10,
    )

    assert ece == 0.0
    assert len(accs) == 10
    assert len(confs) == 10
    assert len(counts) == 10
    assert all(c == 0 for c in counts)
    assert all(a == 0.0 for a in accs)


def test_calculate_binned_ece_boundary_cases() -> None:
    """Verifies boundary mapping for 0.0 and 1.0 into bins 0 and n_bins - 1."""
    preds: list[float] = [0.0, 0.5, 1.0]
    labels: list[int] = [0, 1, 1]
    n_bins: int = 10

    ece, _accs, _confs, counts = calculate_binned_ece(
        predictions=preds,
        labels=labels,
        n_bins=n_bins,
    )

    assert counts[0] == 1  # 0.0 in bin 0
    assert counts[5] == 1  # 0.5 in bin 5 (0.5 * 10 = 5)
    assert counts[9] == 1  # 1.0 in bin 9 (clamped to n_bins - 1)
    assert 0.0 <= ece <= 1.0


def test_calculate_binned_ece_empty_bins_handling() -> None:
    """Verifies that empty bins contribute 0.0 to ECE and do not raise ZeroDivisionError."""
    # Only bin 2 has samples: p = 0.25 (falls in [0.2, 0.3)), 4 samples
    preds: list[float] = [0.25, 0.25, 0.25, 0.25]
    labels: list[int] = [1, 0, 0, 0]  # accuracy = 0.25, conf = 0.25
    n_bins: int = 10

    ece, accs, _confs, counts = calculate_binned_ece(
        predictions=preds,
        labels=labels,
        n_bins=n_bins,
    )

    assert math.isclose(ece, 0.0, abs_tol=1e-9)
    assert counts[2] == 4
    for m in range(n_bins):
        if m != 2:
            assert counts[m] == 0
            assert accs[m] == 0.0


def test_calculate_binned_ece_fail_fast_validations() -> None:
    """Verifies fail-fast exceptions on invalid arguments."""
    # Length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        calculate_binned_ece(predictions=[0.5, 0.6], labels=[1], n_bins=10)

    # n_bins < 2
    with pytest.raises(ValueError, match="n_bins must be at least 2"):
        calculate_binned_ece(predictions=[0.5], labels=[1], n_bins=1)

    # Prediction out of bounds (< 0.0)
    with pytest.raises(ValueError, match="Prediction must be in range"):
        calculate_binned_ece(predictions=[-0.01], labels=[0], n_bins=10)

    # Prediction out of bounds (> 1.0)
    with pytest.raises(ValueError, match="Prediction must be in range"):
        calculate_binned_ece(predictions=[1.01], labels=[1], n_bins=10)

    # Non-finite prediction (NaN)
    with pytest.raises(ValueError, match="finite float"):
        calculate_binned_ece(predictions=[float("nan")], labels=[1], n_bins=10)

    # Non-binary label
    with pytest.raises(ValueError, match="Label must be binary"):
        calculate_binned_ece(predictions=[0.5], labels=[2], n_bins=10)


def test_compute_ece_drift_record_and_mce() -> None:
    """Verifies compute_ece_drift_record generates valid ECEDriftRecord with correct MCE."""
    cfg = ECEMonitorConfig(
        window_bars_30m=336,
        n_bins=10,
        drift_warning_threshold=0.08,
        min_samples_to_evaluate=30,
    )
    eval_time = datetime(2026, 11, 8, 12, 0, 0, tzinfo=UTC)

    # 40 samples, all predicted 0.90, all ground truth 0 -> massive miscalibration
    preds: list[float] = [0.90] * 40
    labels: list[int] = [0] * 40

    record: ECEDriftRecord = compute_ece_drift_record(
        predictions=preds,
        labels=labels,
        config=cfg,
        eval_time=eval_time,
    )

    assert record.eval_time == eval_time
    assert record.sample_count == 40
    assert record.is_drift_detected is True
    assert math.isclose(record.ece, 0.90, abs_tol=1e-6)
    assert math.isclose(record.max_calibration_error, 0.90, abs_tol=1e-6)
    assert record.n_bins == 10


def test_ece_drift_monitor_fifo_capacity() -> None:
    """Verifies ECEDriftMonitor respects window_bars_30m maxlen FIFO bound."""
    cfg = ECEMonitorConfig(
        window_bars_30m=10,
        n_bins=5,
        drift_warning_threshold=0.08,
        min_samples_to_evaluate=5,
    )
    monitor = ECEDriftMonitor(config=cfg)

    t0 = datetime(2026, 11, 8, 0, 0, 0, tzinfo=UTC)
    for _i in range(15):
        monitor.record_observation(eval_time=t0, predicted_prob=0.5, true_label=1)

    assert monitor.current_sample_count == 10


def test_ece_drift_monitor_drift_alert_activation() -> None:
    """Verifies that severe miscalibration triggers is_drift_detected when burn-in passed."""
    cfg = ECEMonitorConfig(
        window_bars_30m=336,
        n_bins=10,
        drift_warning_threshold=0.08,
        min_samples_to_evaluate=30,
    )
    monitor = ECEDriftMonitor(config=cfg)
    t = datetime(2026, 11, 8, 12, 0, 0, tzinfo=UTC)

    # Feed 35 miscalibrated observations: predicted 0.85, true outcome 0 (error = 0.85 > 0.08)
    for _ in range(35):
        monitor.record_observation(eval_time=t, predicted_prob=0.85, true_label=0)

    record = monitor.evaluate_drift(eval_time=t)
    assert record.sample_count == 35
    assert record.is_drift_detected is True
    assert record.ece > 0.08


def test_ece_drift_monitor_min_samples_guard() -> None:
    """Verifies that is_drift_detected remains False before min_samples_to_evaluate is reached."""
    cfg = ECEMonitorConfig(
        window_bars_30m=336,
        n_bins=10,
        drift_warning_threshold=0.08,
        min_samples_to_evaluate=30,
    )
    monitor = ECEDriftMonitor(config=cfg)
    t = datetime(2026, 11, 8, 12, 0, 0, tzinfo=UTC)

    # Feed only 10 miscalibrated observations (< 30)
    for _ in range(10):
        monitor.record_observation(eval_time=t, predicted_prob=0.99, true_label=0)

    record = monitor.evaluate_drift(eval_time=t)
    assert record.sample_count == 10
    assert record.ece > 0.08
    assert record.is_drift_detected is False  # Suppressed by sample count guard


def test_ece_drift_monitor_record_batch_and_clear() -> None:
    """Verifies record_batch and clear methods."""
    cfg = ECEMonitorConfig(
        window_bars_30m=50,
        n_bins=5,
        drift_warning_threshold=0.08,
        min_samples_to_evaluate=10,
    )
    monitor = ECEDriftMonitor(config=cfg)
    t = datetime(2026, 11, 8, 12, 0, 0, tzinfo=UTC)

    batch = [(t, 0.2, 0), (t, 0.4, 0), (t, 0.6, 1), (t, 0.8, 1)]
    monitor.record_batch(batch=batch)
    assert monitor.current_sample_count == 4

    monitor.clear()
    assert monitor.current_sample_count == 0


def test_domain_purity_and_zero_dependencies() -> None:
    """Verifies that ece_drift_monitor does not import heavy libraries (Clean Architecture purity)."""
    import trading_bot.domain.services.ml.ece_drift_monitor as module

    module_name = module.__name__
    imported_modules = sys.modules

    assert f"{module_name}" in imported_modules
    heavy_libs = {"numpy", "scipy", "sklearn", "polars", "lightgbm"}

    # Inspect module globals to ensure zero heavy library imports
    for lib in heavy_libs:
        assert lib not in module.__dict__, (
            f"Forbidden dependency `{lib}` imported in domain service!"
        )
