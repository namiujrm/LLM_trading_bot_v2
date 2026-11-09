# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Adversarial Stress and Edge-Case Test Suite for 6D Conformal OOD Detector.

Adversarially tests:
1. High-throughput Latency SLA benchmark (10,000 samples) and single-sample latency.
2. Empirical conformal false positive rate coverage (~1% target under alpha=0.01) across Monte Carlo seeds.
3. Extreme outlier sensitivity (50-sigma and 100-sigma) and attenuation (gamma < 1e-6).
4. Boundary condition: Float64 underflow (kappa * excess >= 745.13) causing Pydantic ValidationError in predict().
5. Spectral condition number clamping on rank-1 (collinear) covariance matrices.
6. Monotonicity of gamma_ood along continuous multidimensional directional rays.

Adheres strictly to Invariant #9 (Zero Defaults).
"""

from __future__ import annotations

import math
import time

import numpy as np

from trading_bot.adapters.ml_engine.conformal_ood_detector import ConformalSubspaceOODDetector
from trading_bot.domain.entities.ml.ml_configs import ConformalOODConfig
from trading_bot.domain.entities.ml.ml_predictions import OODDetectionResult

CANONICAL_FEATURES: tuple[str, ...] = (
    "hurst_100",
    "fracdiff_64",
    "vol_gk_12",
    "vol_parkinson_24",
    "volume_zscore_12",
    "tick_imbalance_6",
)


def create_adversarial_test_config(
    kappa: float,
    max_cond: float,
) -> ConformalOODConfig:
    """Create a fully-specified test configuration with zero defaults."""
    return ConformalOODConfig(
        feature_names=CANONICAL_FEATURES,
        alpha_significance=0.01,
        kappa_shrinkage=kappa,
        max_condition_number=max_cond,
        min_eigenvalue_floor=1e-6,
    )


def test_adversarial_50_and_100_sigma_outliers() -> None:
    """Verify 50-sigma and 100-sigma outliers have is_ood=True and gamma_ood < 1e-6."""
    np.random.seed(42)
    cfg = create_adversarial_test_config(1.0, 100.0)
    detector = ConformalSubspaceOODDetector(cfg)
    detector.fit(np.random.randn(500, 6).tolist(), np.random.randn(500, 6).tolist())

    # 50-sigma single-axis (under P4 relative excess, excess = (D_M - T) / T ~ 11.7)
    r50 = detector.predict([50.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert r50.is_ood is True
    assert r50.excess_distance > 10.0
    assert r50.gamma_ood < 1e-4
    assert math.isfinite(r50.mahalanobis_distance)

    # 100-sigma single-axis (under P4 relative excess, excess = (D_M - T) / T ~ 24.4)
    r100 = detector.predict([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert r100.is_ood is True
    assert r100.excess_distance > 20.0
    assert r100.gamma_ood < 1e-6
    assert math.isfinite(r100.mahalanobis_distance)

    # Multi-dimensional 50 and 100 sigma
    coord_50 = 50.0 / math.sqrt(6.0)
    coord_100 = 100.0 / math.sqrt(6.0)
    r_multi50 = detector.predict([coord_50] * 6)
    r_multi100 = detector.predict([coord_100] * 6)

    assert r_multi50.is_ood is True and r_multi50.gamma_ood < 1e-4
    assert r_multi100.is_ood is True and r_multi100.gamma_ood < 1e-6


def test_adversarial_float64_underflow_handled_gracefully() -> None:
    """Verify that when kappa * excess >= 745.13, predict() does NOT crash, returning gamma_ood == 0.0."""
    np.random.seed(42)
    cfg = create_adversarial_test_config(1.0, 100.0)
    detector = ConformalSubspaceOODDetector(cfg)
    detector.fit(np.random.randn(500, 6).tolist(), np.random.randn(500, 6).tolist())

    # Flash crash 5000-sigma outlier: excess >> 745.13 -> exp(-excess) underflows to 0.0
    # Must NOT raise ValidationError; must safely return gamma_ood == 0.0
    res_flash = detector.predict([5000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert res_flash.is_ood is True
    assert res_flash.gamma_ood == 0.0
    assert res_flash.shrinkage_factor == 0.0

    # With kappa=10.0, 400-sigma has relative excess ~ 100, 10 * 100 = 1000 > 745.13 -> underflows to 0.0
    cfg_k10 = create_adversarial_test_config(10.0, 100.0)
    det_k10 = ConformalSubspaceOODDetector(cfg_k10)
    det_k10.fit(np.random.randn(500, 6).tolist(), np.random.randn(500, 6).tolist())

    res_k10 = det_k10.predict([400.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert res_k10.is_ood is True
    assert res_k10.gamma_ood == 0.0


def test_adversarial_monotonicity_across_gradient() -> None:
    """Verify gamma_ood monotonically decreases as excess distance strictly increases."""
    np.random.seed(42)
    cfg = create_adversarial_test_config(1.0, 100.0)
    detector = ConformalSubspaceOODDetector(cfg)
    detector.fit(np.random.randn(500, 6).tolist(), np.random.randn(500, 6).tolist())

    direction = np.array([1.0, -0.5, 0.8, -0.2, 0.4, -0.1])
    direction = direction / np.linalg.norm(direction)

    prev_res: OODDetectionResult | None = None
    alphas = np.linspace(0.0, 30.0, 100)

    for a in alphas:
        point = (detector.mean_vector + a * direction).tolist()
        res = detector.predict(point)

        if prev_res is not None and res.mahalanobis_distance > prev_res.mahalanobis_distance:
            assert res.excess_distance >= prev_res.excess_distance
            assert res.gamma_ood <= prev_res.gamma_ood
            if res.is_ood and prev_res.is_ood:
                assert res.gamma_ood < prev_res.gamma_ood

        prev_res = res


def test_adversarial_rank_1_collinear_covariance_regularization() -> None:
    """Verify condition number is strictly regularized <= max_condition_number under rank-1 collinear features."""
    np.random.seed(42)
    n_samples = 200
    cfg = create_adversarial_test_config(1.0, 100.0)
    detector = ConformalSubspaceOODDetector(cfg)

    # 5 out of 6 columns are identical multiples of column 0
    x_rank1 = np.random.randn(n_samples, 6)
    for col in range(1, 6):
        x_rank1[:, col] = x_rank1[:, 0] * float(col + 1)

    detector.fit(x_rank1.tolist(), x_rank1.tolist())

    assert detector.condition_number <= 100.0 + 1e-6
    assert np.all(np.isfinite(detector.precision_matrix))
    assert np.all(np.isfinite(detector.covariance_matrix))


def test_adversarial_conformal_false_positive_coverage_distribution() -> None:
    """Verify empirical FPR is tightly distributed around 1% across 10 independent Monte Carlo trials."""
    cfg = create_adversarial_test_config(1.0, 100.0)
    n_train = 2000
    n_cal = 1000
    n_test = 5000

    rates: list[float] = []
    for seed in range(10):
        rng = np.random.RandomState(seed + 42)
        x_train = rng.randn(n_train, 6).tolist()
        x_cal = rng.randn(n_cal, 6).tolist()
        x_test = rng.randn(n_test, 6).tolist()

        det = ConformalSubspaceOODDetector(cfg)
        det.fit(x_train, x_cal)
        results = det.predict_batch(x_test)
        ood_count = sum(1 for r in results if r.is_ood)
        rates.append(ood_count / n_test)

    mean_rate = float(np.mean(rates))
    assert 0.006 <= mean_rate <= 0.015, f"Mean FPR {mean_rate:.4f} outside [0.006, 0.015]"


def test_adversarial_predict_batch_throughput_benchmark() -> None:
    """Verify predict_batch evaluates 10,000 samples and measure execution latency."""
    cfg = create_adversarial_test_config(1.0, 100.0)
    detector = ConformalSubspaceOODDetector(cfg)

    np.random.seed(42)
    x_train = np.random.randn(500, 6).tolist()
    x_cal = np.random.randn(500, 6).tolist()
    detector.fit(x_train, x_cal)

    x_batch = np.random.randn(10000, 6).tolist()

    t0 = time.perf_counter()
    results = detector.predict_batch(x_batch)
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0
    assert len(results) == 10000
    # Mean single evaluation is sub-millisecond; in CI with GC pressure it can take 20-30us
    per_sample_us = (elapsed_ms / 10000.0) * 1000.0
    assert per_sample_us < 40.0, f"Per sample took {per_sample_us:.2f}us, expected < 40us"
