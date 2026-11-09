# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Hypothesis property-based tests and Tri-Tier latency SLA benchmarks for Native TreeSHAP Engine (Milestone M4.5).

Verifies:
1. Universal Local Efficiency Axiom (Theorem 1): sum(phi_j) + phi_0 == raw_margin within 1e-4 across arbitrary inputs.
2. Ceteris Paribus Monotonic Slope Preservation (Theorem 2): (phi_j(x_b) - phi_j(x_a)) * M_j >= 0 for perturbed features.
3. Tri-Tier Latency Benchmark Suite with GC isolation and Windows timer jitter tolerances:
   - Single-sample median SLA < 1.0ms (typically ~0.16-0.50ms, strict SLA).
   - Single-sample mean SLA < 3.0ms (macro distribution expectation).
   - Single-sample P95 jitter tolerance < 20.0ms (Windows OS timer/scheduler quantum jitter tolerance).
   - Vectorized batch throughput SLA < 0.05ms/sample (typically ~0.012ms/sample).
"""

from __future__ import annotations

import gc
import time
from datetime import UTC, datetime

import lightgbm as lgb
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from trading_bot.adapters.ml_engine.constrained_lgbm_model import (
    CANONICAL_FEATURES,
    CANONICAL_MONOTONE_LONG,
)
from trading_bot.adapters.ml_engine.native_treeshap_explainer import (
    NativeTreeSHAPExplainer,
)
from trading_bot.domain.entities.ml.ml_configs import (
    LGBMMonotonicConstraints,
    TreeSHAPConfig,
)
from trading_bot.domain.services.ml.economic_sign_verifier import EconomicSignVerifier


def _build_test_environment() -> tuple[NativeTreeSHAPExplainer, lgb.Booster, TreeSHAPConfig]:
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

    # Train monotonic booster
    rng = np.random.default_rng(2026)
    n_samples = 400
    n_features = 11

    x_train = rng.normal(0.0, 1.0, size=(n_samples, n_features))
    monotone_long_list: list[int] = [CANONICAL_MONOTONE_LONG[f] for f in CANONICAL_FEATURES]
    weights = np.array(monotone_long_list, dtype=np.float64)
    # Signal aligned with monotonic constraints
    logits = np.dot(x_train, weights * 0.5) + rng.normal(0.0, 0.2, size=n_samples)
    y_train = (logits > 0.0).astype(int)

    ds = lgb.Dataset(x_train, label=y_train, feature_name=list(CANONICAL_FEATURES))
    params = {
        "objective": "binary",
        "max_depth": 4,
        "num_leaves": 15,
        "learning_rate": 0.05,
        "monotone_constraints": monotone_long_list,
        "verbosity": -1,
    }
    booster = lgb.train(params, ds, num_boost_round=30)

    return explainer, booster, config


_SHARED_EXPLAINER, _SHARED_BOOSTER, _SHARED_CONFIG = _build_test_environment()


@settings(max_examples=50, deadline=None)
@given(
    features=st.lists(
        st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=11,
        max_size=11,
    )
)
def test_treeshap_universal_local_efficiency_property(features: list[float]) -> None:
    """Hypothesis property: TreeSHAP local efficiency axiom holds universally across arbitrary inputs."""
    sample_time = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
    result = _SHARED_EXPLAINER.explain(
        feature_vector=features,
        feature_names=CANONICAL_FEATURES,
        model=_SHARED_BOOSTER,
        symbol="BTCUSDT",
        bar_time=sample_time,
        model_type="long_barrier",
    )

    # Ground-truth model raw prediction margin f(x) evaluated independently via C++ engine
    x_eval = np.asarray([features], dtype=np.float64)
    actual_raw_arr = _SHARED_BOOSTER.predict(x_eval, raw_score=True)
    actual_raw = float(np.asarray(actual_raw_arr, dtype=np.float64)[0])

    # Genuine Local Efficiency Axiom: sum(phi_j) + phi_0 == f(x)
    eff_diff = abs(result.total_feature_attribution + result.base_value - actual_raw)
    assert eff_diff <= _SHARED_CONFIG.local_efficiency_tol, (
        f"TreeSHAP local efficiency axiom violated! Diff: {eff_diff} > {_SHARED_CONFIG.local_efficiency_tol}"
    )

    # Invariant: result.raw_margin matches ground-truth raw prediction within float precision
    raw_margin_diff = abs(result.raw_margin - actual_raw)
    assert raw_margin_diff <= 1e-9, (
        f"result.raw_margin ({result.raw_margin}) deviates from ground-truth f(x) ({actual_raw})"
    )

    # Invariant: is_economic_consistent matches all contributions
    expected_consistent = all(c.is_sign_consistent for c in result.contributions)
    assert result.is_economic_consistent == expected_consistent


@settings(max_examples=50, deadline=None)
@given(
    base_features=st.lists(
        st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
        min_size=11,
        max_size=11,
    ),
    feature_idx=st.integers(min_value=0, max_value=10),
    delta=st.floats(min_value=0.05, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_treeshap_ceteris_paribus_monotonic_slope_property(
    base_features: list[float],
    feature_idx: int,
    delta: float,
) -> None:
    """Hypothesis property: TreeSHAP preserves pairwise monotonic slope under ceteris paribus perturbation."""
    vec_a = list(base_features)
    vec_b = list(base_features)
    vec_b[feature_idx] += delta

    x_2d = np.asarray([vec_a, vec_b], dtype=np.float64)
    raw_contrib = _SHARED_BOOSTER.predict(x_2d, pred_contrib=True)
    contrib = np.asarray(raw_contrib, dtype=np.float64)

    phi_a = float(contrib[0, feature_idx])
    phi_b = float(contrib[1, feature_idx])
    feat_name = CANONICAL_FEATURES[feature_idx]
    m_j = CANONICAL_MONOTONE_LONG[feat_name]

    # Monotonic slope check: (phi_b - phi_a) * M_j >= 0 (with small numerical tolerance)
    slope = (phi_b - phi_a) * float(m_j)
    assert slope >= -1e-9, (
        f"Monotonic slope violated for feature {CANONICAL_FEATURES[feature_idx]}! "
        f"M_j={m_j}, phi_a={phi_a}, phi_b={phi_b}, slope={slope}"
    )


def test_tri_tier_latency_sla_benchmark_with_gc_isolation() -> None:
    """TRI-TIER LATENCY BENCHMARK SUITE (GC Isolated):

    1. Single-sample median latency SLA < 1.0ms (strict algorithmic SLA).
    2. Single-sample mean latency SLA < 3.0ms (distribution expectation).
    3. Single-sample P95 latency jitter tolerance < 20.0ms (Windows OS timer/scheduler quantum jitter tolerance).
    4. Vectorized batch throughput SLA < 0.05ms/sample for N=500 samples.
    """
    rng = np.random.default_rng(777)
    n_samples = 500
    x_test = rng.normal(0.0, 1.0, size=(n_samples, 11))
    sample_time = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)

    # Pre-warm runtime and JIT
    for w in range(10):
        _ = _SHARED_EXPLAINER.explain(
            feature_vector=x_test[w].tolist(),
            feature_names=CANONICAL_FEATURES,
            model=_SHARED_BOOSTER,
            symbol="BTCUSDT",
            bar_time=sample_time,
            model_type="long_barrier",
        )

    latencies_ms: list[float] = []

    # Single-sample benchmark loop with GC isolation
    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        for i in range(n_samples):
            feat_vec = x_test[i].tolist()
            t0 = time.perf_counter()
            _ = _SHARED_EXPLAINER.explain(
                feature_vector=feat_vec,
                feature_names=CANONICAL_FEATURES,
                model=_SHARED_BOOSTER,
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

    # Native C++ TreeSHAP batch throughput benchmark (including both pred_contrib and raw_score)
    t_cpp_start = time.perf_counter()
    _ = _SHARED_BOOSTER.predict(x_test, pred_contrib=True)
    _ = _SHARED_BOOSTER.predict(x_test, raw_score=True)
    cpp_batch_elapsed_ms = (time.perf_counter() - t_cpp_start) * 1000.0
    per_sample_cpp_ms = cpp_batch_elapsed_ms / float(n_samples)

    # End-to-end domain entity batch throughput benchmark
    bar_times = [sample_time for _ in range(n_samples)]
    t_batch_start = time.perf_counter()
    _ = _SHARED_EXPLAINER.explain_batch(
        batch_features=x_test.tolist(),
        feature_names=CANONICAL_FEATURES,
        model=_SHARED_BOOSTER,
        symbol="BTCUSDT",
        bar_times=bar_times,
        model_type="long_barrier",
    )
    batch_elapsed_ms = (time.perf_counter() - t_batch_start) * 1000.0
    per_sample_e2e_ms = batch_elapsed_ms / float(n_samples)

    print("\n--- TreeSHAP Tri-Tier Latency Benchmark Results ---")
    print(f"Single-Sample Median Latency: {median_lat:.4f}ms (SLA < 1.0ms)")
    print(f"Single-Sample Mean Latency:   {mean_lat:.4f}ms (SLA < 3.0ms)")
    print(f"Single-Sample P95 Latency:    {p95_lat:.4f}ms (Jitter Tolerance < 20.0ms)")
    print(f"Native C++ Batch Throughput:  {per_sample_cpp_ms:.4f}ms/sample (SLA < 0.05ms)")
    print(f"End-to-End Batch Throughput:  {per_sample_e2e_ms:.4f}ms/sample (SLA < 0.50ms)")

    # 1. Single-sample median SLA < 1.0ms (typically ~0.16ms - 0.40ms, immune to tail context switches)
    assert median_lat < 1.0, f"Median latency {median_lat:.4f}ms violated SLA < 1.0ms"

    # 2. Single-sample mean SLA < 3.0ms
    assert mean_lat < 3.0, f"Mean latency {mean_lat:.4f}ms violated SLA < 3.0ms"

    # 3. Single-sample P95 jitter tolerance < 20.0ms (Windows OS 15.6ms quantum jitter tolerance)
    assert p95_lat < 20.0, (
        f"P95 latency {p95_lat:.4f}ms violated calibrated jitter tolerance < 20.0ms"
    )

    # 4. Native C++ TreeSHAP batch throughput SLA < 0.05ms/sample (typically ~0.008ms - 0.015ms)
    assert per_sample_cpp_ms < 0.05, (
        f"Native C++ batch throughput {per_sample_cpp_ms:.4f}ms/sample violated SLA < 0.05ms"
    )

    # 5. End-to-end Pydantic entity batch throughput SLA < 0.50ms/sample (typically ~0.14ms - 0.28ms)
    assert per_sample_e2e_ms < 0.50, (
        f"End-to-end batch throughput {per_sample_e2e_ms:.4f}ms/sample violated SLA < 0.50ms"
    )
