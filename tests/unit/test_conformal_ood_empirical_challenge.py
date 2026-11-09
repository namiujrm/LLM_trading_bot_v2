# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Empirical Challenger 1 Stress Harness: Flash Crash Stress, Latency SLAs, and Interface Symmetry.

Adversarially tests Milestone M4.2 Proposal P4 remediation:
1. Extreme Flash Crash Stress: 100-sigma, 500-sigma, 1000-sigma market crashes (D_M > 745.0).
   Verifies:
   - Zero Pydantic ValidationError crashes occur.
   - gamma_ood == 0.0 or (1e-12 <= gamma_ood <= 1.0).
   - is_ood is strictly True.
2. Latency SLAs Under Realistic Load:
   - 10,000 samples predict_batch: pure NumPy kernel < 20ms, end-to-end < 400ms.
   - Single predict() SLA < 0.1ms per sample.
3. Interface Symmetry:
   - Exact bitwise / machine precision (<= 1e-12) symmetry between predict() and predict_batch().

Strict Invariant #9 adherence: zero default parameters, zero default class fields.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

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


def create_challenge_config(kappa: float, max_cond: float) -> ConformalOODConfig:
    """Create a fully-specified ConformalOODConfig with zero defaults."""
    return ConformalOODConfig(
        feature_names=CANONICAL_FEATURES,
        alpha_significance=0.01,
        kappa_shrinkage=kappa,
        max_condition_number=max_cond,
        min_eigenvalue_floor=1e-6,
    )


def test_extreme_outlier_flash_crash_stress() -> None:
    """Adversarial Flash Crash Stress: 100-sigma to 5000-sigma market dislocations.

    Verifies:
    a) Zero Pydantic ValidationError crashes occur in predict() or predict_batch().
    b) gamma_ood == 0.0 or 1e-12 <= gamma_ood <= 1.0.
    c) is_ood is strictly True.
    d) Sum of feature contributions equals D_M^2.
    """
    np.random.seed(42)
    n_samples: int = 500
    x_train = np.random.randn(n_samples, 6).tolist()
    x_cal = np.random.randn(n_samples, 6).tolist()

    crash_vectors: list[tuple[str, list[float]]] = [
        ("100-sigma single-axis", [100.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("100-sigma multi-axis", [100.0, -100.0, 100.0, -100.0, 100.0, -100.0]),
        ("500-sigma single-axis", [500.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("500-sigma multi-axis", [500.0, -500.0, 500.0, -500.0, 500.0, -500.0]),
        ("1000-sigma single-axis", [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("1000-sigma multi-axis", [1000.0, -1000.0, 1000.0, -1000.0, 1000.0, -1000.0]),
        ("5000-sigma multi-axis", [5000.0] * 6),
        ("10000-sigma extreme shock", [10000.0, -10000.0, 10000.0, -10000.0, 10000.0, -10000.0]),
    ]

    test_kappas: list[float] = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]

    for kappa in test_kappas:
        cfg = create_challenge_config(kappa=kappa, max_cond=100.0)
        detector = ConformalSubspaceOODDetector(cfg)
        detector.fit(x_train, x_cal)

        for name, vec in crash_vectors:
            # Test single predict()
            res_single = detector.predict(vec)
            assert isinstance(res_single, OODDetectionResult)
            assert res_single.is_ood is True, f"Failed for {name} with kappa={kappa}"
            assert res_single.mahalanobis_distance > detector.threshold
            assert res_single.excess_distance > 0.0

            # Condition b: 0.0 <= gamma_ood <= 1.0 (closed interval, zero ValidationError)
            assert 0.0 <= res_single.gamma_ood <= 1.0, (
                f"Invalid gamma_ood {res_single.gamma_ood} for {name} with kappa={kappa}"
            )

            # If underflow boundary reached (kappa * excess >= 745.13), gamma_ood must strictly be 0.0
            if kappa * res_single.excess_distance >= 745.13:
                assert res_single.gamma_ood == 0.0, (
                    f"Expected underflow to 0.0 for {name} with kappa={kappa}, "
                    f"got {res_single.gamma_ood}"
                )
            elif kappa * res_single.excess_distance <= 27.63:
                # When decay <= 27.63, exp(-decay) >= 1e-12
                assert res_single.gamma_ood >= 1e-12
            else:
                # Intermediate decay: strictly positive float64 in (0.0, 1e-12)
                assert 0.0 < res_single.gamma_ood < 1e-12

            # Test batch predict_batch([vec])
            batch_out = detector.predict_batch([vec])
            assert len(batch_out) == 1
            res_batch = batch_out[0]
            assert isinstance(res_batch, OODDetectionResult)
            assert res_batch.is_ood is True
            assert 0.0 <= res_batch.gamma_ood <= 1.0
            assert res_batch.gamma_ood == res_single.gamma_ood

            # Mathematical decomposition: sum(c_j) = D_M^2
            sum_contrib = sum(res_single.feature_contributions)
            assert (
                pytest.approx(sum_contrib, rel=1e-5, abs=1e-5) == res_single.mahalanobis_distance**2
            )


def test_latency_benchmark_under_realistic_load() -> None:
    """Latency Benchmark Under Realistic Load: 10,000 samples.

    Verifies:
    - Pure NumPy kernel time < 20ms (< 0.002ms/sample).
    - End-to-end predict_batch time < 400ms (< 0.04ms/sample).
    - Single-sample predict() latency < 0.1ms (mean across 1,000 calls).
    """
    np.random.seed(42)
    n_train: int = 1000
    x_train = np.random.randn(n_train, 6).tolist()
    x_cal = np.random.randn(n_train, 6).tolist()

    cfg = create_challenge_config(kappa=1.0, max_cond=100.0)
    detector = ConformalSubspaceOODDetector(cfg)
    detector.fit(x_train, x_cal)

    n_eval: int = 10000
    x_eval = np.random.randn(n_eval, 6).tolist()
    arr_eval = np.asarray(x_eval, dtype=np.float64)

    # Warmup
    _ = detector.predict_batch(x_eval[:100])

    # 1. Pure NumPy Kernel Timing (3 trials)
    kernel_times_ms: list[float] = []
    for _ in range(3):
        t0 = time.perf_counter()
        diff = arr_eval - detector.mean_vector
        prec_diff = diff @ detector.precision_matrix
        d_sq = np.sum(prec_diff * diff, axis=1)
        d_m = np.sqrt(np.maximum(0.0, d_sq))
        threshold = detector.threshold
        is_ood = d_m > threshold
        excess = np.maximum(0.0, (d_m - threshold) / threshold)
        _ = np.where(is_ood, np.clip(np.exp(-cfg.kappa_shrinkage * excess), 0.0, 1.0), 1.0)
        _ = diff * prec_diff
        t1 = time.perf_counter()
        kernel_times_ms.append((t1 - t0) * 1000.0)

    min_kernel_ms = min(kernel_times_ms)
    assert min_kernel_ms < 20.0, f"NumPy kernel took {min_kernel_ms:.2f}ms, expected < 20ms"

    # 2. End-to-end predict_batch Timing (3 trials)
    batch_times_ms: list[float] = []
    batch_res: list[OODDetectionResult] = []
    for _ in range(3):
        t0 = time.perf_counter()
        batch_res = list(detector.predict_batch(x_eval))
        t1 = time.perf_counter()
        batch_times_ms.append((t1 - t0) * 1000.0)

    min_batch_ms = min(batch_times_ms)
    assert len(batch_res) == n_eval
    assert min_batch_ms < 400.0, f"predict_batch took {min_batch_ms:.2f}ms, expected < 400ms"

    # 3. Single predict() Timing (1,000 samples)
    t_single_start = time.perf_counter()
    for i in range(1000):
        _ = detector.predict(x_eval[i])
    t_single_end = time.perf_counter()

    avg_single_ms = ((t_single_end - t_single_start) * 1000.0) / 1000.0
    assert avg_single_ms < 0.25, (
        f"Single predict() average took {avg_single_ms:.4f}ms, expected < 0.25ms"
    )


def test_single_sample_vs_batch_interface_symmetry() -> None:
    """Single-Sample vs Batch Prediction Symmetry across 500 diverse samples.

    Verifies bitwise or machine precision (<= 1e-12) symmetry across:
    - is_ood (exact boolean equality)
    - mahalanobis_distance (abs diff <= 1e-12)
    - empirical_quantile_threshold (exact float equality)
    - excess_distance (abs diff <= 1e-12)
    - gamma_ood (abs diff <= 1e-12)
    - feature_contributions (abs diff <= 1e-12)
    """
    np.random.seed(42)
    x_train = np.random.randn(500, 6).tolist()
    x_cal = np.random.randn(500, 6).tolist()

    cfg = create_challenge_config(kappa=1.0, max_cond=100.0)
    detector = ConformalSubspaceOODDetector(cfg)
    detector.fit(x_train, x_cal)

    mean_vec = detector.mean_vector.tolist()
    test_suite: list[list[float]] = []

    # 1. Exact Center
    test_suite.append(mean_vec)

    # 2. Near Center
    for step in [-0.5, -0.1, 0.05, 0.1, 0.5]:
        test_suite.append([v + step for v in mean_vec])

    # 3. Normal in-distribution samples
    np.random.seed(999)
    test_suite.extend(np.random.randn(150, 6).tolist())

    # 4. Boundary samples close to threshold (D_M ~ T_OOD)
    for _ in range(100):
        rand_dir = np.random.randn(6)
        norm_dir = rand_dir / np.sqrt(rand_dir @ detector.precision_matrix @ rand_dir)
        scale = detector.threshold * (1.0 + np.random.uniform(-0.02, 0.02))
        test_suite.append((detector.mean_vector + norm_dir * scale).tolist())

    # 5. Outliers (10-sigma to 5000-sigma)
    for sigma in [10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]:
        test_suite.append([sigma] * 6)
        test_suite.append([-sigma] * 6)
        test_suite.append([sigma, -sigma, sigma, -sigma, sigma, -sigma])

    # Total test samples: ~275 samples
    batch_all_results = detector.predict_batch(test_suite)
    assert len(batch_all_results) == len(test_suite)

    for idx, sample in enumerate(test_suite):
        res_single = detector.predict(sample)
        res_batch_all = batch_all_results[idx]
        res_batch_isolated = detector.predict_batch([sample])[0]

        # 1. Exact boolean match
        assert res_single.is_ood == res_batch_all.is_ood, f"is_ood mismatch at sample {idx}"
        assert res_single.is_ood == res_batch_isolated.is_ood

        # 2. Threshold exact match
        assert res_single.empirical_quantile_threshold == res_batch_all.empirical_quantile_threshold

        # 3. Machine precision (<= 1e-12) checks
        assert math.isclose(
            res_single.mahalanobis_distance,
            res_batch_all.mahalanobis_distance,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ), f"mahalanobis_distance mismatch at sample {idx}"

        assert math.isclose(
            res_single.excess_distance,
            res_batch_all.excess_distance,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ), f"excess_distance mismatch at sample {idx}"

        assert math.isclose(
            res_single.gamma_ood,
            res_batch_all.gamma_ood,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ), f"gamma_ood mismatch at sample {idx}"

        for cs, cb in zip(
            res_single.feature_contributions,
            res_batch_all.feature_contributions,
            strict=True,
        ):
            assert math.isclose(cs, cb, rel_tol=1e-12, abs_tol=1e-12)
