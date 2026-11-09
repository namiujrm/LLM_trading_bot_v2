# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Unit tests for Pre-Tabulated Venn-Abers Calibrator Adapter (Clean Architecture & Invariant #9).

Verifies:
- Conformance to ProbabilityCalibratorPort Protocol.
- Fail-fast validation on unfitted state, invalid inputs, and single-class data.
- Isotonic PAV monotonicity, lower/upper bound ordering (p0 <= p1), and sandwich property (p0 <= pcal <= p1).
- Numerical safety of denominator (1 - p0 + p1 >= 1.0).
- Interface symmetry across calibrate(), calibrate_batch(), and calibrate_array().
- Calibration quality improvement (reduction of Expected Calibration Error).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from trading_bot.adapters.ml_engine.tabulated_venn_abers import (
    TabulatedVennAbersCalibrator,
)
from trading_bot.domain.entities.ml.ml_configs import FastVennAbersConfig
from trading_bot.domain.services.ml.ece_drift_monitor import calculate_binned_ece
from trading_bot.ports.ml_model_port import ProbabilityCalibratorPort


def _create_default_config() -> FastVennAbersConfig:
    return FastVennAbersConfig(
        grid_points=500,
        prob_clip_min=0.001,
        prob_clip_max=0.999,
        denom_eps=1e-6,
        uncertainty_threshold=0.15,
        calibration_floor=0.01,
    )


def test_port_conformance() -> None:
    """Verifies that TabulatedVennAbersCalibrator conforms to ProbabilityCalibratorPort."""
    cfg = _create_default_config()
    calibrator = TabulatedVennAbersCalibrator(config=cfg)
    assert isinstance(calibrator, ProbabilityCalibratorPort)
    assert not calibrator.is_fitted
    assert calibrator.config == cfg


def test_unfitted_calibrator_raises_runtime_error() -> None:
    """Verifies that calling any inference method before fit() raises RuntimeError."""
    cfg = _create_default_config()
    calibrator = TabulatedVennAbersCalibrator(config=cfg)

    with pytest.raises(RuntimeError, match="has not been fitted"):
        _ = calibrator.grid_scores

    with pytest.raises(RuntimeError, match="has not been fitted"):
        _ = calibrator.table_p0

    with pytest.raises(RuntimeError, match="has not been fitted"):
        _ = calibrator.table_p1

    with pytest.raises(RuntimeError, match="has not been fitted"):
        _ = calibrator.table_pcal

    with pytest.raises(RuntimeError, match="has not been fitted"):
        _ = calibrator.table_delta

    with pytest.raises(RuntimeError, match="has not been fitted"):
        calibrator.calibrate(score=0.5)

    with pytest.raises(RuntimeError, match="has not been fitted"):
        calibrator.calibrate_batch(scores=[0.5])

    with pytest.raises(RuntimeError, match="has not been fitted"):
        calibrator.calibrate_array(scores=np.array([0.5]))


def test_isotonic_pav_monotonicity_and_ordering() -> None:
    """Verifies monotonicity of p0 and p1 tables, ordering p0 <= p1, and sandwich property."""
    cfg = _create_default_config()
    calibrator = TabulatedVennAbersCalibrator(config=cfg)

    # Generate synthetic calibration data: 200 samples
    rng = np.random.default_rng(42)
    raw_scores = rng.uniform(0.05, 0.95, size=200)
    # Binary labels with positive correlation to raw scores
    true_probs = 1.0 / (1.0 + np.exp(-5.0 * (raw_scores - 0.5)))
    labels = (rng.uniform(0.0, 1.0, size=200) < true_probs).astype(int)

    calibrator.fit(scores=raw_scores.tolist(), labels=labels.tolist())
    assert calibrator.is_fitted

    p0 = calibrator.table_p0
    p1 = calibrator.table_p1
    pcal = calibrator.table_pcal

    # Check monotonicity of grid
    assert np.all(np.diff(calibrator.grid_scores) > 0)

    # Check monotonicity of p0 and p1 (non-decreasing within tolerance)
    assert np.all(np.diff(p0) >= -1e-12)
    assert np.all(np.diff(p1) >= -1e-12)

    # Check lower-upper ordering: p0 <= p1 everywhere
    assert np.all(p0 <= p1 + 1e-12)

    # Check Sandwich invariant: p0 <= pcal <= p1 everywhere
    assert np.all(pcal >= p0 - 1e-12)
    assert np.all(pcal <= p1 + 1e-12)

    # Check denominator lower bound: 1 - p0 + p1 >= 1.0
    denom = 1.0 - p0 + p1
    assert np.all(denom >= 1.0 - 1e-12)


def test_sandwich_property_and_uncertainty_interval() -> None:
    """Verifies that calibrate() produces outputs satisfying the sandwich property and width."""
    cfg = _create_default_config()
    calibrator = TabulatedVennAbersCalibrator(config=cfg)

    # Synthetic calibration data
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    labels = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
    calibrator.fit(scores=scores, labels=labels)

    test_queries = [0.0, 0.05, 0.25, 0.45, 0.55, 0.75, 0.95, 1.0]
    for q in test_queries:
        res = calibrator.calibrate(score=q)
        assert 0.0 <= res.p_calibrated <= 1.0
        assert 0.0 <= res.p_lower <= 1.0
        assert 0.0 <= res.p_upper <= 1.0
        assert res.p_lower <= res.p_upper + 1e-7
        assert res.p_lower - 1e-7 <= res.p_calibrated <= res.p_upper + 1e-7
        assert math.isclose(res.uncertainty_width, res.p_upper - res.p_lower, abs_tol=1e-6)
        assert res.raw_score == q


def test_fail_fast_on_invalid_inputs() -> None:
    """Verifies fail-fast guards on fit() and calibrate()."""
    cfg = _create_default_config()
    calibrator = TabulatedVennAbersCalibrator(config=cfg)

    # Length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        calibrator.fit(scores=[0.1, 0.2], labels=[0])

    # Insufficient samples (< 10)
    with pytest.raises(ValueError, match="Insufficient calibration samples"):
        calibrator.fit(scores=[0.1] * 5, labels=[0] * 5)

    # Non-finite scores
    with pytest.raises(ValueError, match="non-finite values"):
        calibrator.fit(scores=[float("nan")] * 10, labels=[0] * 5 + [1] * 5)

    # Scores out of [0, 1]
    with pytest.raises(ValueError, match="range"):
        calibrator.fit(scores=[-0.1] + [0.5] * 9, labels=[0] * 5 + [1] * 5)

    # Non-binary labels
    with pytest.raises(ValueError, match="strictly binary"):
        calibrator.fit(scores=[0.5] * 10, labels=[0] * 5 + [2] * 5)

    # Single-class dataset
    with pytest.raises(ValueError, match="single-class"):
        calibrator.fit(scores=[0.5] * 10, labels=[1] * 10)

    # Fit valid data first to test inference exceptions
    calibrator.fit(scores=[float(i) / 10.0 for i in range(10)], labels=[0] * 5 + [1] * 5)

    # Non-finite score in calibrate
    with pytest.raises(ValueError, match="finite float"):
        calibrator.calibrate(score=float("nan"))

    # Out of range score in calibrate
    with pytest.raises(ValueError, match="range"):
        calibrator.calibrate(score=1.5)

    with pytest.raises(ValueError, match="range"):
        calibrator.calibrate(score=-0.1)


def test_interface_symmetry_single_batch_array() -> None:
    """Verifies numerical symmetry between calibrate, calibrate_batch, and calibrate_array."""
    cfg = _create_default_config()
    calibrator = TabulatedVennAbersCalibrator(config=cfg)

    rng = np.random.default_rng(123)
    train_scores = rng.uniform(0.1, 0.9, size=100)
    train_labels = (train_scores > 0.5).astype(int)
    calibrator.fit(scores=train_scores.tolist(), labels=train_labels.tolist())

    test_scores = [0.15, 0.35, 0.50, 0.65, 0.85]
    single_results = [calibrator.calibrate(s) for s in test_scores]
    batch_results = calibrator.calibrate_batch(scores=test_scores)
    array_results = calibrator.calibrate_array(scores=np.array(test_scores))

    for i in range(len(test_scores)):
        sr = single_results[i]
        br = batch_results[i]
        ar = array_results[i]

        assert math.isclose(sr.p_calibrated, br.p_calibrated, abs_tol=1e-12)
        assert math.isclose(sr.p_lower, br.p_lower, abs_tol=1e-12)
        assert math.isclose(sr.p_upper, br.p_upper, abs_tol=1e-12)
        assert math.isclose(sr.uncertainty_width, br.uncertainty_width, abs_tol=1e-12)

        assert math.isclose(sr.p_calibrated, float(ar[0]), abs_tol=1e-12)
        assert math.isclose(sr.p_lower, float(ar[1]), abs_tol=1e-12)
        assert math.isclose(sr.p_upper, float(ar[2]), abs_tol=1e-12)
        assert math.isclose(sr.uncertainty_width, float(ar[3]), abs_tol=1e-12)


def test_calibration_error_reduction_on_miscalibrated_model() -> None:
    """Verifies that Venn-Abers calibration reduces Expected Calibration Error (ECE) >= 25%."""
    cfg = _create_default_config()
    calibrator = TabulatedVennAbersCalibrator(config=cfg)

    rng = np.random.default_rng(777)
    # Ground truth latent probability
    x = rng.uniform(0.0, 1.0, size=1000)
    p_true = x.copy()
    y = (rng.uniform(0.0, 1.0, size=1000) < p_true).astype(int)

    # Severely uncalibrated raw model (cubed probabilities: underconfident in middle, shrunk to 0)
    raw_scores = np.power(p_true, 3.0)

    # Split calibration (500) and test (500)
    cal_scores = raw_scores[:500].tolist()
    cal_labels = y[:500].tolist()
    test_scores = raw_scores[500:].tolist()
    test_labels = y[500:].tolist()

    # Raw ECE on test set
    raw_ece, _, _, _ = calculate_binned_ece(
        predictions=test_scores,
        labels=test_labels,
        n_bins=10,
    )

    # Fit Venn-Abers on calibration split
    calibrator.fit(scores=cal_scores, labels=cal_labels)

    # Calibrate test scores
    cal_results = calibrator.calibrate_batch(scores=test_scores)
    cal_preds = [r.p_calibrated for r in cal_results]

    cal_ece, _, _, _ = calculate_binned_ece(
        predictions=cal_preds,
        labels=test_labels,
        n_bins=10,
    )

    ece_reduction_pct = (raw_ece - cal_ece) / raw_ece * 100.0

    # Gate 4.4 requires >= 25% ECE reduction
    assert ece_reduction_pct >= 25.0, (
        f"ECE reduction {ece_reduction_pct:.2f}% did not meet 25% threshold "
        f"(raw_ece={raw_ece:.4f}, cal_ece={cal_ece:.4f})"
    )
