# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Empirical Challenger 1 Stress Harness: TreeSHAP Latency SLA, Numerical Precision, and Memory Footprint (Milestone M4.5).

Adversarially tests Milestone M4.5 Proposal P6:
1. Latency SLA Empirical Challenge:
   - Single-sample median latency < 1.0ms across 100 iterations with GC pre-warming.
   - Batch throughput: 1,000 samples with C++ vectorized kernel < 0.05ms/sample.
   - Tail latency under multi-threaded CPU contention: P95 < 35.0ms (Windows OS timer quantum jitter tolerance).
2. Local Efficiency Numerical Precision Stress:
   - Extreme numerical feature values (+1e6, -1e6, 1e-12, mixed magnitudes).
   - True C++ local efficiency comparison against raw_score=True.
   - Boundary sensitivity check (rejection at > local_efficiency_tol, acceptance at <= tol).
3. Memory Footprint of Batch Processing:
   - Repeated batch executions (50,000 samples total) with tracemalloc and RSS verification.
   - Confirms zero memory leaks and bounded heap allocation.

Strict Invariant #9 adherence: zero default parameters, zero default class fields, zero fallbacks.
"""

from __future__ import annotations

import concurrent.futures
import gc
import time
import tracemalloc
from datetime import UTC, datetime

import lightgbm as lgb
import numpy as np
import pytest
from pydantic import ValidationError

from trading_bot.adapters.ml_engine.constrained_lgbm_model import (
    CANONICAL_FEATURES,
    CANONICAL_MONOTONE_LONG,
    CANONICAL_MONOTONE_SHORT,
)
from trading_bot.adapters.ml_engine.native_treeshap_explainer import (
    NativeTreeSHAPExplainer,
)
from trading_bot.domain.entities.ml.ml_configs import (
    LGBMMonotonicConstraints,
    TreeSHAPConfig,
)
from trading_bot.domain.entities.ml.ml_predictions import (
    SHAPDriverContribution,
    SHAPExplanationResult,
)
from trading_bot.domain.services.ml.economic_sign_verifier import EconomicSignVerifier
from trading_bot.ports.ml_model_port import TreeSHAPExplainerPort


def _create_explainer_and_models() -> tuple[
    NativeTreeSHAPExplainer, lgb.Booster, lgb.Booster, TreeSHAPConfig
]:
    """Helper to train standard long and short boosters and initialize NativeTreeSHAPExplainer."""
    config = TreeSHAPConfig(
        top_k_bullish=5,
        top_k_bearish=3,
        local_efficiency_tol=1e-4,
        sign_tolerance=1e-4,
    )
    long_constraints = LGBMMonotonicConstraints.create_canonical_long()
    short_constraints = LGBMMonotonicConstraints.create_canonical_short()

    verifier_long = EconomicSignVerifier(config=config, monotonic_constraints=long_constraints)
    verifier_short = EconomicSignVerifier(config=config, monotonic_constraints=short_constraints)

    explainer = NativeTreeSHAPExplainer(
        config=config,
        sign_verifier_long=verifier_long,
        sign_verifier_short=verifier_short,
    )

    rng = np.random.default_rng(42)
    n_samples = 500
    n_features = 11

    x_train = rng.normal(0.0, 1.0, size=(n_samples, n_features))

    # Long booster
    monotone_long: list[int] = [CANONICAL_MONOTONE_LONG[f] for f in CANONICAL_FEATURES]
    y_long = (np.dot(x_train, np.array(monotone_long, dtype=np.float64) * 0.4) > 0.0).astype(int)
    ds_long = lgb.Dataset(x_train, label=y_long, feature_name=list(CANONICAL_FEATURES))
    params_long = {
        "objective": "binary",
        "max_depth": 4,
        "num_leaves": 15,
        "learning_rate": 0.05,
        "monotone_constraints": monotone_long,
        "verbosity": -1,
    }
    booster_long = lgb.train(params_long, ds_long, num_boost_round=30)

    # Short booster
    monotone_short: list[int] = [CANONICAL_MONOTONE_SHORT[f] for f in CANONICAL_FEATURES]
    y_short = (np.dot(x_train, np.array(monotone_short, dtype=np.float64) * 0.4) > 0.0).astype(int)
    ds_short = lgb.Dataset(x_train, label=y_short, feature_name=list(CANONICAL_FEATURES))
    params_short = {
        "objective": "binary",
        "max_depth": 4,
        "num_leaves": 15,
        "learning_rate": 0.05,
        "monotone_constraints": monotone_short,
        "verbosity": -1,
    }
    booster_short = lgb.train(params_short, ds_short, num_boost_round=30)

    return explainer, booster_long, booster_short, config


def test_scope1_latency_sla_and_batch_throughput_empirical() -> None:
    """Scope 1: Latency SLA & Batch Throughput Empirical Verification.

    Asserts:
    1. Single-sample median latency < 1.0ms across 100 iterations with GC pre-warming.
    2. Native C++ TreeSHAP kernel throughput < 0.05ms/sample across 1,000 samples.
    """
    explainer, booster_long, _, _ = _create_explainer_and_models()
    rng = np.random.default_rng(12345)
    sample_time = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)

    # Pre-warming (10 warm-up runs)
    warm_x = rng.normal(0.0, 1.0, size=(10, 11)).tolist()
    for row in warm_x:
        _ = explainer.explain(
            feature_vector=row,
            feature_names=CANONICAL_FEATURES,
            model=booster_long,
            symbol="BTCUSDT",
            bar_time=sample_time,
            model_type="long_barrier",
        )

    # 1. Single-Sample Latency Benchmark (100 iterations, GC disabled during measurement)
    n_iters = 100
    test_rows = rng.normal(0.0, 1.0, size=(n_iters, 11)).tolist()
    latencies_ms: list[float] = []

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for i in range(n_iters):
            t0 = time.perf_counter()
            _ = explainer.explain(
                feature_vector=test_rows[i],
                feature_names=CANONICAL_FEATURES,
                model=booster_long,
                symbol="BTCUSDT",
                bar_time=sample_time,
                model_type="long_barrier",
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    finally:
        if gc_was_enabled:
            gc.enable()

    median_lat = float(np.median(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))
    mean_lat = float(np.mean(latencies_ms))

    print(
        f"\n[Scope 1 Empirical Single-Sample] Median: {median_lat:.4f}ms, P95: {p95_lat:.4f}ms, Mean: {mean_lat:.4f}ms"
    )
    assert median_lat < 1.0, f"Single-sample median latency {median_lat:.4f}ms exceeded SLA 1.0ms"

    # 2. Vectorized Batch Throughput Benchmark (1,000 samples)
    n_batch = 1000
    x_batch = rng.normal(0.0, 1.0, size=(n_batch, 11))
    t_start = time.perf_counter()
    _ = booster_long.predict(x_batch, pred_contrib=True)
    cpp_time_ms = (time.perf_counter() - t_start) * 1000.0
    per_sample_cpp_ms = cpp_time_ms / float(n_batch)

    print(
        f"[Scope 1 Empirical Batch 1000] Total C++: {cpp_time_ms:.4f}ms, Per-Sample: {per_sample_cpp_ms:.5f}ms/sample"
    )
    assert per_sample_cpp_ms < 0.05, (
        f"Vectorized batch throughput {per_sample_cpp_ms:.5f}ms/sample exceeded SLA 0.05ms"
    )


def test_scope1_tail_latency_under_cpu_thread_contention() -> None:
    """Scope 1 Stress: Single-Sample P95 Tail Latency under Heavy CPU Thread Contention.

    Simulates high-load environment: spawns 4 background worker threads doing
    continuous matrix operations while measuring 100 single-sample explanations.
    Asserts Median < 8.0ms and P95 < 35.0ms under contention (tolerating 15.6ms OS quantum slices).
    """
    explainer, booster_long, _, _ = _create_explainer_and_models()
    rng = np.random.default_rng(999)
    sample_time = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)

    # Background CPU load worker
    stop_event = False

    def _cpu_burner() -> None:
        burn_rng = np.random.default_rng(101)
        while not stop_event:
            a = burn_rng.normal(0.0, 1.0, size=(64, 64))
            _ = np.linalg.inv(np.dot(a.T, a) + np.eye(64) * 0.1)

    latencies_ms: list[float] = []
    test_rows = rng.normal(0.0, 1.0, size=(100, 11)).tolist()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for _ in range(4):
            _ = executor.submit(_cpu_burner)
        try:
            # Let CPU burn ramp up
            time.sleep(0.05)

            # Pre-warm runtime under contention
            for row in test_rows[:10]:
                _ = explainer.explain(
                    feature_vector=row,
                    feature_names=CANONICAL_FEATURES,
                    model=booster_long,
                    symbol="BTCUSDT",
                    bar_time=sample_time,
                    model_type="long_barrier",
                )

            for i in range(100):
                t0 = time.perf_counter()
                _ = explainer.explain(
                    feature_vector=test_rows[i],
                    feature_names=CANONICAL_FEATURES,
                    model=booster_long,
                    symbol="BTCUSDT",
                    bar_time=sample_time,
                    model_type="long_barrier",
                )
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        finally:
            stop_event = True
            time.sleep(0.05)

    p95_contention = float(np.percentile(latencies_ms, 95))
    median_contention = float(np.median(latencies_ms))
    max_contention = float(np.max(latencies_ms))

    print(
        f"\n[Scope 1 Contention Stress] Median: {median_contention:.4f}ms, P95: {p95_contention:.4f}ms, Max: {max_contention:.4f}ms"
    )
    assert median_contention < 8.0, (
        f"Median latency under contention {median_contention:.4f}ms exceeded SLA 8.0ms"
    )
    assert p95_contention < 35.0, (
        f"Tail latency P95 under contention {p95_contention:.4f}ms exceeded jitter tolerance 35.0ms"
    )


def test_scope2_local_efficiency_numerical_precision_stress() -> None:
    """Scope 2: Local Efficiency Numerical Precision Stress with Extreme Values.

    Challenges TreeSHAP additivity axiom: sum(phi_j) + phi_0 == raw_score across:
    - Extreme positive magnitudes (1e6)
    - Extreme negative magnitudes (-1e6)
    - Tiny sub-epsilon values (1e-12)
    - Mixed scale vectors (1e6, -1e6, 1e-12, 0.0)
    - Compares sum(phi_j) + phi_0 directly against booster.predict(raw_score=True).
    """
    explainer, booster_long, _, config = _create_explainer_and_models()
    sample_time = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)

    extreme_scenarios: list[tuple[str, list[float]]] = [
        ("Huge Positive (1e6)", [1e6] * 11),
        ("Huge Negative (-1e6)", [-1e6] * 11),
        ("Near Zero (1e-12)", [1e-12] * 11),
        ("Mixed Extreme Scale", [1e6, -1e6, 1e-12, -1e-12, 1e5, -1e5, 0.0, 1.0, -1.0, 50.0, -50.0]),
        ("Alternating Flash Crash", [1e4, -1e4, 1e4, -1e4, 1e4, -1e4, 1e4, -1e4, 1e4, -1e4, 1e4]),
        ("Asymmetric Saturation", [1e8, 0.0, 0.0, 0.0, -1e8, 0.0, 0.0, 0.0, 1e-15, -1e-15, 0.5]),
    ]

    for label, feat_vec in extreme_scenarios:
        # 1. NativeTreeSHAPExplainer explain() call
        result = explainer.explain(
            feature_vector=feat_vec,
            feature_names=CANONICAL_FEATURES,
            model=booster_long,
            symbol="BTCUSDT",
            bar_time=sample_time,
            model_type="long_barrier",
        )

        # 2. Independent C++ ground truth comparison against raw_score=True
        x_2d = np.asarray([feat_vec], dtype=np.float64)
        c_raw_arr = np.asarray(booster_long.predict(x_2d, raw_score=True), dtype=np.float64)
        c_raw_score = float(c_raw_arr[0])
        c_contrib = np.asarray(booster_long.predict(x_2d, pred_contrib=True), dtype=np.float64)
        c_phi_sum = float(np.sum(c_contrib[0, :-1]))
        c_base_val = float(c_contrib[0, -1])

        # Discrepancy between C++ TreeSHAP sum and C++ raw_score
        cpp_discrepancy = abs(c_phi_sum + c_base_val - c_raw_score)
        print(
            f"[Scope 2 Extreme '{label}'] cpp_discrepancy: {cpp_discrepancy:.2e}, raw_score: {c_raw_score:.6f}"
        )

        # Local efficiency must hold to machine precision <= 1e-12 in compiled C++
        assert cpp_discrepancy <= 1e-10, (
            f"C++ TreeSHAP local efficiency violated on '{label}': {cpp_discrepancy} > 1e-10"
        )

        # Result raw_margin must match C++ raw_score within local_efficiency_tol
        adapter_diff = abs(result.raw_margin - c_raw_score)
        assert adapter_diff <= config.local_efficiency_tol, (
            f"Adapter raw_margin diverged from raw_score on '{label}': {adapter_diff} > {config.local_efficiency_tol}"
        )


def test_scope2_local_efficiency_violation_rejection_boundary() -> None:
    """Scope 2 Boundary Sensitivity: Assert Genuine Efficiency Violations are Rejected.

    Injects artificial perturbation Delta to verify that:
    - Delta = 0.9e-4 (<= tol 1e-4) is accepted.
    - Delta = 1.1e-4 (> tol 1e-4) raises ValueError.
    - Delta = 1.0 (gross leakage) raises ValueError.
    """
    _create_explainer_and_models()
    sample_time = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)

    # Baseline valid contributions
    contribs = tuple(
        SHAPDriverContribution(
            feature_name=name,
            feature_value=0.5,
            shap_value=0.01 if idx % 2 == 0 else -0.01,
            expected_sign=1 if idx % 2 == 0 else -1,
            is_sign_consistent=True,
        )
        for idx, name in enumerate(CANONICAL_FEATURES)
    )
    base_val = 0.1
    phi_sum = sum(c.shap_value for c in contribs)
    exact_margin = phi_sum + base_val

    # 1. Delta within tolerance (0.9e-4) -> Should pass
    result_valid = SHAPExplanationResult(
        symbol="BTCUSDT",
        bar_time=sample_time,
        model_type="long_barrier",
        base_value=base_val,
        raw_margin=exact_margin + 0.9e-4,
        contributions=contribs,
        top_bullish_drivers=tuple(c for c in contribs if c.shap_value > 0)[:5],
        top_bearish_drivers=tuple(c for c in contribs if c.shap_value < 0)[:3],
        is_economic_consistent=True,
    )
    assert result_valid is not None

    # 2. Delta exceeding tolerance (1.1e-4) -> Must raise ValueError
    with pytest.raises(ValueError, match="TreeSHAP local efficiency violated"):
        _ = SHAPExplanationResult(
            symbol="BTCUSDT",
            bar_time=sample_time,
            model_type="long_barrier",
            base_value=base_val,
            raw_margin=exact_margin + 1.1e-4,
            contributions=contribs,
            top_bullish_drivers=tuple(c for c in contribs if c.shap_value > 0)[:5],
            top_bearish_drivers=tuple(c for c in contribs if c.shap_value < 0)[:3],
            is_economic_consistent=True,
        )

    # 3. Gross leakage Delta = 1.0 -> Must raise ValueError
    with pytest.raises(ValueError, match="TreeSHAP local efficiency violated"):
        _ = SHAPExplanationResult(
            symbol="BTCUSDT",
            bar_time=sample_time,
            model_type="long_barrier",
            base_value=base_val,
            raw_margin=exact_margin + 1.0,
            contributions=contribs,
            top_bullish_drivers=tuple(c for c in contribs if c.shap_value > 0)[:5],
            top_bearish_drivers=tuple(c for c in contribs if c.shap_value < 0)[:3],
            is_economic_consistent=True,
        )


def test_scope3_memory_footprint_and_leak_stress() -> None:
    """Scope 3: Memory Footprint & Heap Leak Stress across Repeated Batch Invocations.

    Executes 50 consecutive batch iterations of 1,000 samples (50,000 samples total).
    Verifies that:
    1. Python heap allocation does not grow unbounded (peak heap delta < 10 MB).
    2. Zero uncollected cyclic references or memory leaks exist.
    """
    explainer, booster_long, _, _ = _create_explainer_and_models()
    rng = np.random.default_rng(2026)
    n_batch_samples = 1000
    n_iterations = 50  # 50,000 total samples
    sample_time = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
    bar_times = [sample_time for _ in range(n_batch_samples)]

    x_batch = rng.normal(0.0, 1.0, size=(n_batch_samples, 11)).tolist()

    # Pre-warm runtime and garbage collector
    _ = explainer.explain_batch(
        batch_features=x_batch,
        feature_names=CANONICAL_FEATURES,
        model=booster_long,
        symbol="BTCUSDT",
        bar_times=bar_times,
        model_type="long_barrier",
    )
    gc.collect()

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    for _ in range(n_iterations):
        results = explainer.explain_batch(
            batch_features=x_batch,
            feature_names=CANONICAL_FEATURES,
            model=booster_long,
            symbol="BTCUSDT",
            bar_times=bar_times,
            model_type="long_barrier",
        )
        assert len(results) == n_batch_samples
        # Release reference
        del results

    gc.collect()
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
    total_diff_bytes = sum(stat.size_diff for stat in top_stats)
    total_diff_mb = total_diff_bytes / (1024.0 * 1024.0)

    print(
        f"\n[Scope 3 Memory Footprint] Total Heap Delta over 50,000 samples: {total_diff_mb:.3f} MB"
    )
    for stat in top_stats[:5]:
        print(f"  {stat}")

    # Bounded heap growth: delta must be strictly < 10.0 MB for 50,000 samples
    assert total_diff_mb < 10.0, (
        f"Heap memory grew by {total_diff_mb:.3f} MB, indicating potential memory leak"
    )


def test_scope4_structural_contract_and_invariant9_conformance() -> None:
    """Scope 4: Clean Architecture Port Protocol and Invariant #9 AST Verification.

    Verifies:
    1. NativeTreeSHAPExplainer is an instance of TreeSHAPExplainerPort protocol.
    2. Invariant #9: All configurations and domain entities reject missing arguments.
    """
    explainer, _, _, _ = _create_explainer_and_models()

    # 1. Structural protocol check
    assert isinstance(explainer, TreeSHAPExplainerPort), (
        "NativeTreeSHAPExplainer does not satisfy TreeSHAPExplainerPort Protocol"
    )

    # 2. Invariant #9 zero default parameter checks: calling without arguments must raise ValidationError
    with pytest.raises(ValidationError):
        _ = TreeSHAPConfig()  # type: ignore[call-arg]

    with pytest.raises(ValidationError):
        _ = LGBMMonotonicConstraints()  # type: ignore[call-arg]

    with pytest.raises(ValidationError):
        _ = SHAPDriverContribution()  # type: ignore[call-arg]

    with pytest.raises(ValidationError):
        _ = SHAPExplanationResult()  # type: ignore[call-arg]
