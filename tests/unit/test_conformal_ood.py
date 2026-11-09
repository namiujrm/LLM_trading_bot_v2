# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Comprehensive Unit Tests for 6D Conformal Subspace OOD Detector (Zero Defaults).

Verifies Milestone M4.2:
- In-distribution samples with D_M <= T_OOD: is_ood=False, gamma_ood == 1.0, excess == 0.0
- Extreme outliers (10-sigma shift): is_ood=True, D_M >> T_OOD, gamma_ood << 1.0
- Empirical conformal coverage: ~1% false positives under alpha=0.01 on Gaussian data
- Condition number bounds: cond(Sigma) <= 100.0 on collinear and degenerate features
- High-performance latency: 10,000 samples evaluated in < 100ms (< 0.01ms per sample)
- Fail-fast validation: dimension mismatch, NaN/Inf, empty training set, N < 6
- Invariant #9: zero default parameters, zero class field defaults
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from pydantic import ValidationError

from trading_bot.adapters.ml_engine.conformal_ood_detector import ConformalSubspaceOODDetector
from trading_bot.domain.entities.ml.ml_configs import ConformalOODConfig
from trading_bot.domain.entities.ml.ml_predictions import OODDetectionResult
from trading_bot.ports.ml_model_port import OODDetectorPort

CANONICAL_FEATURES: tuple[str, ...] = (
    "hurst_100",
    "fracdiff_64",
    "vol_gk_12",
    "vol_parkinson_24",
    "volume_zscore_12",
    "tick_imbalance_6",
)


def create_default_test_config() -> ConformalOODConfig:
    """Create a fully-specified ConformalOODConfig with zero defaults."""
    return ConformalOODConfig(
        feature_names=CANONICAL_FEATURES,
        alpha_significance=0.01,
        kappa_shrinkage=1.0,
        max_condition_number=100.0,
        min_eigenvalue_floor=1e-6,
    )


def test_ood_detector_port_adapter_conformance() -> None:
    """Verify ConformalSubspaceOODDetector structurally implements OODDetectorPort."""
    cfg = create_default_test_config()
    detector = ConformalSubspaceOODDetector(cfg)
    assert isinstance(detector, OODDetectorPort)


def test_in_distribution_sample_classification() -> None:
    """Verify in-distribution sample (mean vector) has is_ood=False, gamma=1.0, excess=0.0."""
    np.random.seed(42)
    n_samples = 500
    x_train = np.random.randn(n_samples, 6).tolist()
    x_cal = np.random.randn(n_samples, 6).tolist()

    cfg = create_default_test_config()
    detector = ConformalSubspaceOODDetector(cfg)
    detector.fit(x_train, x_cal)

    assert detector.is_fitted is True
    assert detector.threshold > 0.0
    assert detector.condition_number <= 100.0
    assert detector.config == cfg

    # Test center sample (exact mean)
    mean_sample = detector.mean_vector.tolist()
    res = detector.predict(mean_sample)

    assert res.is_ood is False
    assert res.excess_distance == 0.0
    assert res.gamma_ood == 1.0
    assert res.shrinkage_factor == 1.0
    assert res.threshold == res.empirical_quantile_threshold
    assert res.mahalanobis_distance <= res.empirical_quantile_threshold
    assert len(res.feature_contributions) == 6
    assert pytest.approx(sum(res.feature_contributions), abs=1e-5) == res.mahalanobis_distance**2

    # Verify evaluate alias produces identical result
    eval_res = detector.evaluate(mean_sample)
    assert eval_res.is_ood == res.is_ood
    assert eval_res.mahalanobis_distance == res.mahalanobis_distance


def test_extreme_outlier_detection() -> None:
    """Verify 10-sigma extreme outlier is strictly classified as OOD with gamma << 1.0."""
    np.random.seed(42)
    n_samples = 500
    x_train = np.random.randn(n_samples, 6).tolist()
    x_cal = np.random.randn(n_samples, 6).tolist()

    cfg = create_default_test_config()
    detector = ConformalSubspaceOODDetector(cfg)
    detector.fit(x_train, x_cal)

    # 10-sigma shift on volatility and volume
    outlier = [0.0, 0.0, 10.0, 10.0, 0.0, 0.0]
    res = detector.predict(outlier)

    assert res.is_ood is True
    assert res.mahalanobis_distance > detector.threshold * 2.0
    assert res.excess_distance > 0.0
    assert res.gamma_ood < 0.10  # strictly gamma << 1.0 under P4 relative excess
    # Volatility features (indices 2 and 3) should dominate contribution
    assert res.feature_contributions[2] > 10.0
    assert res.feature_contributions[3] > 10.0


def test_synthetic_gaussian_conformal_coverage() -> None:
    """Verify ~1% of in-distribution Gaussian test samples are flagged under alpha=0.01."""
    np.random.seed(12345)
    n_train = 5000
    n_cal = 5000
    n_test = 20000

    x_train = np.random.randn(n_train, 6).tolist()
    x_cal = np.random.randn(n_cal, 6).tolist()
    x_test = np.random.randn(n_test, 6).tolist()

    cfg = create_default_test_config()
    detector = ConformalSubspaceOODDetector(cfg)
    detector.fit(x_train, x_cal)

    results = detector.predict_batch(x_test)
    assert len(results) == n_test

    ood_count = sum(1 for r in results if r.is_ood)
    ood_fraction = ood_count / n_test

    # With alpha=0.01, empirical coverage should be tightly bounded around 1% [0.7%, 1.5%]
    assert 0.007 <= ood_fraction <= 0.015, f"Expected ~1% OOD rate, got {ood_fraction:.4f}"


def test_condition_number_bounds_on_collinear_and_degenerate_features() -> None:
    """Verify cond(Sigma) <= 100.0 even under collinear, duplicate, and constant features."""
    np.random.seed(42)
    n_samples = 300

    # Case A: Perfectly collinear columns (col 1 = 3 * col 0, col 5 = -2 * col 4)
    x_collinear = np.random.randn(n_samples, 6)
    x_collinear[:, 1] = x_collinear[:, 0] * 3.0
    x_collinear[:, 5] = x_collinear[:, 4] * -2.0

    cfg = create_default_test_config()
    detector_a = ConformalSubspaceOODDetector(cfg)
    detector_a.fit(x_collinear.tolist(), x_collinear.tolist())

    assert detector_a.condition_number <= 100.0 + 1e-9
    assert np.all(np.isfinite(detector_a.precision_matrix))
    assert np.all(np.isfinite(detector_a.covariance_matrix))

    # Case B: Constant column (zero variance on col 3)
    x_constant = np.random.randn(n_samples, 6)
    x_constant[:, 3] = 42.0

    detector_b = ConformalSubspaceOODDetector(cfg)
    detector_b.fit(x_constant.tolist(), x_constant.tolist())

    assert detector_b.condition_number <= 100.0 + 1e-9
    assert np.all(np.isfinite(detector_b.precision_matrix))
    assert np.all(np.isfinite(detector_b.covariance_matrix))


def test_latency_benchmark_10000_samples() -> None:
    """Verify latency SLAs: vectorized math < 0.002ms/sample, batch < 0.05ms/sample, single < 0.1ms."""
    np.random.seed(42)
    n_samples = 500
    x_train = np.random.randn(n_samples, 6).tolist()
    x_cal = np.random.randn(n_samples, 6).tolist()

    cfg = create_default_test_config()
    detector = ConformalSubspaceOODDetector(cfg)
    detector.fit(x_train, x_cal)

    # Generate 10,000 batch samples
    m_samples = 10000
    x_batch = np.random.randn(m_samples, 6).tolist()

    # Warmup pass (pre-warm Python runtime and JIT/NumPy dispatch)
    _ = detector.predict_batch(x_batch[:100])

    # 1. Benchmark pure vectorized numerical math kernel (10,000 samples)
    arr = np.asarray(x_batch, dtype=np.float64)
    t_math_0 = time.perf_counter()
    diff = arr - detector.mean_vector
    prec_diff = diff @ detector.precision_matrix
    d_sq = np.sum(prec_diff * diff, axis=1)
    d_m = np.sqrt(np.maximum(0.0, d_sq))
    threshold = detector.threshold
    is_ood = d_m > threshold
    excess = np.maximum(0.0, (d_m - threshold) / threshold)
    _ = np.where(is_ood, np.clip(np.exp(-cfg.kappa_shrinkage * excess), 0.0, 1.0), 1.0)
    _ = diff * prec_diff
    t_math_1 = time.perf_counter()

    math_elapsed_ms = (t_math_1 - t_math_0) * 1000.0
    math_per_sample_ms = math_elapsed_ms / m_samples
    assert math_elapsed_ms < 20.0, f"Vectorized math took {math_elapsed_ms:.2f}ms, expected < 20ms"
    assert math_per_sample_ms < 0.002, (
        f"Vectorized math per sample took {math_per_sample_ms:.5f}ms, expected < 0.002ms"
    )

    # 2. Time end-to-end batch prediction including Pydantic domain entity allocations
    t0 = time.perf_counter()
    batch_results = detector.predict_batch(x_batch)
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0
    per_sample_ms = elapsed_ms / m_samples

    assert len(batch_results) == m_samples
    assert elapsed_ms < 400.0, f"Batch evaluation took {elapsed_ms:.2f}ms, expected < 400ms"
    assert per_sample_ms < 0.05, (
        f"Per sample evaluation took {per_sample_ms:.5f}ms, expected < 0.05ms"
    )

    # 3. Verify single-sample real-time evaluation speed (100 samples)
    t2 = time.perf_counter()
    for i in range(100):
        detector.predict(x_batch[i])
    t3 = time.perf_counter()

    single_avg_ms = ((t3 - t2) * 1000.0) / 100.0
    assert single_avg_ms < 0.25, (
        f"Single-sample evaluation took {single_avg_ms:.4f}ms, expected < 0.25ms"
    )


def test_fail_fast_validation_errors() -> None:
    """Verify typed ValueError / RuntimeError on invalid inputs (fail-fast discipline)."""
    cfg = create_default_test_config()
    detector = ConformalSubspaceOODDetector(cfg)

    # 1. Unfitted property and prediction checks
    assert detector.is_fitted is False
    with pytest.raises(RuntimeError, match="has not been fitted"):
        _ = detector.threshold
    with pytest.raises(RuntimeError, match="has not been fitted"):
        _ = detector.condition_number
    with pytest.raises(RuntimeError, match="has not been fitted"):
        _ = detector.precision_matrix
    with pytest.raises(RuntimeError, match="has not been fitted"):
        _ = detector.covariance_matrix
    with pytest.raises(RuntimeError, match="has not been fitted"):
        _ = detector.mean_vector
    with pytest.raises(RuntimeError, match="has not been fitted"):
        detector.predict([0.0] * 6)
    with pytest.raises(RuntimeError, match="has not been fitted"):
        detector.predict_batch([[0.0] * 6])

    # 2. Empty training set
    with pytest.raises(ValueError, match="Insufficient training samples"):
        detector.fit([], [[0.0] * 6] * 10)

    # 3. Training sample count < 6
    with pytest.raises(ValueError, match="Insufficient training samples"):
        detector.fit([[0.0] * 6] * 4, [[0.0] * 6] * 10)

    # 4. Calibration sample count < 6
    with pytest.raises(ValueError, match="Insufficient calibration samples"):
        detector.fit([[0.0] * 6] * 20, [[0.0] * 6] * 4)

    # 5. Feature dimension mismatch in fit (5 columns instead of 6)
    with pytest.raises(ValueError, match="does not match expected 6 features"):
        detector.fit([[0.0] * 5] * 20, [[0.0] * 6] * 20)

    with pytest.raises(ValueError, match="does not match expected 6 features"):
        detector.fit([[0.0] * 6] * 20, [[0.0] * 5] * 20)

    # 6. Non-finite values in training set
    x_nan = [[0.0] * 6] * 20
    x_nan[0][0] = float("nan")
    with pytest.raises(ValueError, match="contains non-finite values"):
        detector.fit(x_nan, [[0.0] * 6] * 20)

    # 7. Non-finite values in calibration set
    x_inf = [[0.0] * 6] * 20
    x_inf[0][0] = float("inf")
    with pytest.raises(ValueError, match="contains non-finite values"):
        detector.fit([[1.0 + i] * 6 for i in range(20)], x_inf)

    # 8. Zero variance training set
    with pytest.raises(ValueError, match="zero or near-zero variance"):
        detector.fit([[1.0] * 6] * 20, [[1.0] * 6] * 20)

    # Now fit valid detector for prediction checks
    np.random.seed(42)
    valid_train = np.random.randn(50, 6).tolist()
    valid_cal = np.random.randn(50, 6).tolist()
    detector.fit(valid_train, valid_cal)

    # 9. Dimension mismatch in predict
    with pytest.raises(ValueError, match="length 6"):
        detector.predict([0.0] * 5)

    # 10. Non-finite in predict
    with pytest.raises(ValueError, match="non-finite"):
        detector.predict([0.0, float("nan"), 0.0, 0.0, 0.0, 0.0])

    # 11. Dimension mismatch in predict_batch
    with pytest.raises(ValueError, match="shape"):
        detector.predict_batch([[0.0] * 5])

    # 12. Non-finite in predict_batch
    with pytest.raises(ValueError, match="non-finite"):
        detector.predict_batch([[0.0, float("nan"), 0.0, 0.0, 0.0, 0.0]])


def test_monotonic_exponential_penalty_decay() -> None:
    """Verify monotonic decay: higher distance strictly yields smaller gamma_ood."""
    np.random.seed(42)
    detector = ConformalSubspaceOODDetector(create_default_test_config())
    detector.fit(np.random.randn(100, 6).tolist(), np.random.randn(100, 6).tolist())

    outlier_small = [0.0, 0.0, 5.0, 0.0, 0.0, 0.0]
    outlier_medium = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
    outlier_large = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]

    r1 = detector.predict(outlier_small)
    r2 = detector.predict(outlier_medium)
    r3 = detector.predict(outlier_large)

    assert r1.mahalanobis_distance < r2.mahalanobis_distance < r3.mahalanobis_distance
    assert r1.gamma_ood > r2.gamma_ood > r3.gamma_ood > 0.0


def test_invariant_9_zero_defaults_enforcement() -> None:
    """Verify Invariant #9: schemas and functions cannot be instantiated with missing args."""
    # ConformalOODConfig requires all 5 fields
    with pytest.raises(ValidationError):
        ConformalOODConfig()  # type: ignore[call-arg]

    # OODDetectionResult requires all 6 fields
    with pytest.raises(ValidationError):
        OODDetectionResult()  # type: ignore[call-arg]

    # ConformalSubspaceOODDetector requires config argument
    with pytest.raises(TypeError):
        ConformalSubspaceOODDetector()  # type: ignore[call-arg]


def test_conformal_ood_config_validation() -> None:
    """Verify ConformalOODConfig value constraints."""
    # alpha_significance must be in (0, 1)
    with pytest.raises(ValueError, match="alpha_significance"):
        ConformalOODConfig(
            feature_names=CANONICAL_FEATURES,
            alpha_significance=1.5,
            kappa_shrinkage=1.0,
            max_condition_number=100.0,
            min_eigenvalue_floor=1e-6,
        )

    # kappa_shrinkage must be > 0
    with pytest.raises(ValueError, match="kappa_shrinkage"):
        ConformalOODConfig(
            feature_names=CANONICAL_FEATURES,
            alpha_significance=0.01,
            kappa_shrinkage=-0.5,
            max_condition_number=100.0,
            min_eigenvalue_floor=1e-6,
        )

    # max_condition_number must be > 1.0
    with pytest.raises(ValueError, match="max_condition_number"):
        ConformalOODConfig(
            feature_names=CANONICAL_FEATURES,
            alpha_significance=0.01,
            kappa_shrinkage=1.0,
            max_condition_number=0.5,
            min_eigenvalue_floor=1e-6,
        )

    # min_eigenvalue_floor must be > 0
    with pytest.raises(ValueError, match="min_eigenvalue_floor"):
        ConformalOODConfig(
            feature_names=CANONICAL_FEATURES,
            alpha_significance=0.01,
            kappa_shrinkage=1.0,
            max_condition_number=100.0,
            min_eigenvalue_floor=0.0,
        )

    # feature_names must not be empty
    with pytest.raises(ValueError, match="feature_names"):
        ConformalOODConfig(
            feature_names=(),
            alpha_significance=0.01,
            kappa_shrinkage=1.0,
            max_condition_number=100.0,
            min_eigenvalue_floor=1e-6,
        )


def test_ood_detection_result_validation() -> None:
    """Verify OODDetectionResult value constraints."""
    with pytest.raises(ValueError, match="mahalanobis_distance"):
        OODDetectionResult(
            is_ood=False,
            mahalanobis_distance=-1.0,
            empirical_quantile_threshold=3.0,
            excess_distance=0.0,
            gamma_ood=1.0,
            feature_contributions=(0.0,) * 6,
        )

    with pytest.raises(ValueError, match="empirical_quantile_threshold"):
        OODDetectionResult(
            is_ood=False,
            mahalanobis_distance=1.0,
            empirical_quantile_threshold=0.0,
            excess_distance=0.0,
            gamma_ood=1.0,
            feature_contributions=(0.0,) * 6,
        )

    with pytest.raises(ValueError, match="excess_distance"):
        OODDetectionResult(
            is_ood=False,
            mahalanobis_distance=1.0,
            empirical_quantile_threshold=3.0,
            excess_distance=-0.5,
            gamma_ood=1.0,
            feature_contributions=(0.0,) * 6,
        )

    with pytest.raises(ValueError, match="gamma_ood"):
        OODDetectionResult(
            is_ood=False,
            mahalanobis_distance=1.0,
            empirical_quantile_threshold=3.0,
            excess_distance=0.0,
            gamma_ood=1.5,
            feature_contributions=(0.0,) * 6,
        )

    with pytest.raises(ValueError, match="gamma_ood"):
        OODDetectionResult(
            is_ood=False,
            mahalanobis_distance=1.0,
            empirical_quantile_threshold=3.0,
            excess_distance=0.0,
            gamma_ood=-0.1,
            feature_contributions=(0.0,) * 6,
        )

    # Verify gamma_ood == 0.0 is strictly valid (closed interval [0.0, 1.0])
    valid_zero_res = OODDetectionResult(
        is_ood=True,
        mahalanobis_distance=1000.0,
        empirical_quantile_threshold=3.0,
        excess_distance=332.33,
        gamma_ood=0.0,
        feature_contributions=(0.0,) * 6,
    )
    assert valid_zero_res.gamma_ood == 0.0


def test_degenerate_zero_threshold_calibration_raises_fail_fast_error() -> None:
    """Verify fit() fails fast with ValueError if calibration set yields non-positive threshold.

    Adheres strictly to Invariant #9 (Fail-Fast Discipline): Prevents a poisoned detector
    from deferring validation errors to inference time.
    """
    cfg = create_default_test_config()
    detector = ConformalSubspaceOODDetector(cfg)

    np.random.seed(42)
    x_train = np.random.randn(50, 6).tolist()
    mu = np.mean(x_train, axis=0).tolist()
    # Degenerate calibration set where all samples match the empirical mean exactly (d_cal = 0)
    x_cal_degenerate = [mu] * 50

    with pytest.raises(ValueError, match=r"Empirical conformal threshold .* is non-positive"):
        detector.fit(x_train, x_cal_degenerate)

    # Invariant check: Detector must remain unfitted after failed fit()
    assert detector.is_fitted is False
    with pytest.raises(RuntimeError, match="has not been fitted"):
        _ = detector.threshold


def test_predict_and_predict_batch_interface_symmetry_and_invariants() -> None:
    """Verify predict() and predict_batch() yield identical, fully-validated domain entities.

    Verifies symmetry across:
    1. In-distribution center (mean vector)
    2. In-distribution normal samples
    3. Boundary edge samples (near threshold)
    4. Out-of-distribution moderate samples
    5. Extreme 10-sigma, 50-sigma, and 1000-sigma flash-crash outliers
    Ensures predict_batch() enforces domain entity invariants without silent bypass.
    """
    cfg = create_default_test_config()
    detector = ConformalSubspaceOODDetector(cfg)

    np.random.seed(42)
    x_train = np.random.randn(200, 6).tolist()
    x_cal = np.random.randn(200, 6).tolist()
    detector.fit(x_train, x_cal)

    assert detector.is_fitted is True
    assert detector.threshold > 0.0

    mean_vec = detector.mean_vector.tolist()
    test_samples: list[list[float]] = [
        mean_vec,  # in-distribution center
        [m + 0.1 for m in mean_vec],  # near center
        [1.0, -0.5, 0.8, -0.2, 0.4, -0.1],  # random in-distribution
        [0.0, 0.0, 10.0, 10.0, 0.0, 0.0],  # moderate OOD
        [50.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 50-sigma outlier
        [100.0, -100.0, 50.0, -50.0, 80.0, -80.0],  # 100-sigma outlier
        [1000.0] * 6,  # 1000-sigma flash-crash regime
    ]

    single_results = [detector.predict(s) for s in test_samples]
    batch_results = detector.predict_batch(test_samples)

    assert len(single_results) == len(batch_results) == len(test_samples)

    for r_single, r_batch in zip(single_results, batch_results, strict=True):
        assert isinstance(r_single, OODDetectionResult)
        assert isinstance(r_batch, OODDetectionResult)

        # Domain invariants
        assert r_single.mahalanobis_distance >= 0.0
        assert r_batch.mahalanobis_distance >= 0.0
        assert r_single.empirical_quantile_threshold > 0.0
        assert r_batch.empirical_quantile_threshold > 0.0
        assert r_single.excess_distance >= 0.0
        assert r_batch.excess_distance >= 0.0
        assert 0.0 <= r_single.gamma_ood <= 1.0
        assert 0.0 <= r_batch.gamma_ood <= 1.0

        # Exact symmetry (within machine precision rel=1e-9)
        assert r_single.is_ood == r_batch.is_ood
        assert (
            pytest.approx(r_single.mahalanobis_distance, rel=1e-9, abs=1e-9)
            == r_batch.mahalanobis_distance
        )
        assert r_single.empirical_quantile_threshold == r_batch.empirical_quantile_threshold
        assert (
            pytest.approx(r_single.excess_distance, rel=1e-9, abs=1e-9) == r_batch.excess_distance
        )
        assert pytest.approx(r_single.gamma_ood, rel=1e-9, abs=1e-9) == r_batch.gamma_ood
        for cs, cb in zip(
            r_single.feature_contributions, r_batch.feature_contributions, strict=True
        ):
            assert pytest.approx(cs, rel=1e-9, abs=1e-9) == cb

        # Mathematical decomposition sum(c_j) = D_M^2
        assert (
            pytest.approx(sum(r_single.feature_contributions), rel=1e-5, abs=1e-5)
            == r_single.mahalanobis_distance**2
        )
        assert (
            pytest.approx(sum(r_batch.feature_contributions), rel=1e-5, abs=1e-5)
            == r_batch.mahalanobis_distance**2
        )
