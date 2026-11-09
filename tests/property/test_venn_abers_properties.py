# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Hypothesis property-based tests and Tri-Tier latency SLA benchmarks for Venn-Abers Calibrator.

Verifies:
1. Analytical monotonicity property: s_a <= s_b ==> p_cal(s_a) <= p_cal(s_b) everywhere.
2. Sandwich invariant property: p_lower <= p_calibrated <= p_upper and bounded uncertainty width.
3. Tri-Tier Latency Benchmark Suite with GC isolation and Windows timer jitter tolerances:
   - Single-sample median SLA < 0.02ms (target < 0.05ms).
   - Batch array throughput SLA < 0.001ms/sample for 10,000 samples (< 10ms total).
   - Batch domain entity throughput SLA < 0.05ms/sample for 1,000 samples (< 50ms total).
"""

from __future__ import annotations

import gc
import math
import time

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from trading_bot.adapters.ml_engine.tabulated_venn_abers import (
    TabulatedVennAbersCalibrator,
)
from trading_bot.domain.entities.ml.ml_configs import FastVennAbersConfig


def _get_fitted_calibrator() -> TabulatedVennAbersCalibrator:
    cfg = FastVennAbersConfig(
        grid_points=500,
        prob_clip_min=0.001,
        prob_clip_max=0.999,
        denom_eps=1e-6,
        uncertainty_threshold=0.15,
        calibration_floor=0.01,
    )
    calibrator = TabulatedVennAbersCalibrator(config=cfg)

    # Realistic calibration dataset with signal
    rng = np.random.default_rng(2026)
    scores = rng.uniform(0.01, 0.99, size=300)
    p_true = 1.0 / (1.0 + np.exp(-4.0 * (scores - 0.5)))
    labels = (rng.uniform(0.0, 1.0, size=300) < p_true).astype(int)

    calibrator.fit(scores=scores.tolist(), labels=labels.tolist())
    return calibrator


_SHARED_CALIBRATOR = _get_fitted_calibrator()


@settings(max_examples=100)
@given(
    s1=st.floats(min_value=0.001, max_value=0.999),
    s2=st.floats(min_value=0.001, max_value=0.999),
)
def test_venn_abers_monotonicity_property(s1: float, s2: float) -> None:
    """Hypothesis property: calibrator preserves strict non-decreasing monotonicity."""
    sa = min(s1, s2)
    sb = max(s1, s2)

    res_a = _SHARED_CALIBRATOR.calibrate(score=sa)
    res_b = _SHARED_CALIBRATOR.calibrate(score=sb)

    _tol: float = 1e-9
    assert res_a.p_lower <= res_b.p_lower + _tol
    assert res_a.p_upper <= res_b.p_upper + _tol
    assert res_a.p_calibrated <= res_b.p_calibrated + _tol


@settings(max_examples=100)
@given(
    score=st.floats(min_value=0.0, max_value=1.0),
)
def test_venn_abers_sandwich_and_boundedness_property(score: float) -> None:
    """Hypothesis property: calibrated probability satisfies sandwich invariant and bounds."""
    res = _SHARED_CALIBRATOR.calibrate(score=score)

    _tol: float = 1e-7
    assert 0.0 <= res.p_lower <= 1.0
    assert 0.0 <= res.p_upper <= 1.0
    assert 0.0 <= res.p_calibrated <= 1.0
    assert 0.0 <= res.uncertainty_width <= 1.0

    assert res.p_lower <= res.p_upper + _tol
    assert res.p_lower - _tol <= res.p_calibrated <= res.p_upper + _tol
    assert math.isclose(res.uncertainty_width, res.p_upper - res.p_lower, abs_tol=1e-6)


def test_tri_tier_latency_sla_benchmarks() -> None:
    """Verifies ultra-fast inference speed meeting Milestone M4.4 SLA (< 0.05ms/sample).

    Implements:
    - Tier 1: Warmup phase (100 queries) to prime CPU branch predictor and cache.
    - Tier 2: Garbage Collection isolation during timing loops.
    - Tier 3: Multi-metric percentile SLA assertions with OS thread jitter tolerance.
    """
    calibrator = _SHARED_CALIBRATOR

    # Tier 1: Warmup Phase
    for _ in range(100):
        _ = calibrator.calibrate(score=0.5)

    # Tier 2: GC Isolation
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        # Measure 2,000 single-sample calibrate() calls
        latencies_single_ms: list[float] = []
        for _ in range(2000):
            t0 = time.perf_counter()
            _ = calibrator.calibrate(score=0.55)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_single_ms.append(dt_ms)

        # Measure 10,000 sample vectorized calibrate_array() call
        scores_10k = np.linspace(0.01, 0.99, 10000, dtype=np.float64)
        t_arr_start = time.perf_counter()
        arr_out = calibrator.calibrate_array(scores=scores_10k)
        total_arr_ms = (time.perf_counter() - t_arr_start) * 1000.0

        # Measure 1,000 sample domain entity calibrate_batch() call
        scores_1k = np.linspace(0.01, 0.99, 1000, dtype=np.float64).tolist()
        t_batch_start = time.perf_counter()
        batch_out = calibrator.calibrate_batch(scores=scores_1k)
        total_batch_ms = (time.perf_counter() - t_batch_start) * 1000.0
    finally:
        if gc_was_enabled:
            gc.enable()

    # Tier 3: Multi-Metric Percentile SLA Assertions
    median_single_ms = float(np.median(latencies_single_ms))
    mean_single_ms = float(np.mean(latencies_single_ms))
    p95_single_ms = float(np.percentile(latencies_single_ms, 95))
    p99_single_ms = float(np.percentile(latencies_single_ms, 99))

    # Single-sample SLA: Median must be < 0.02ms (target SLA is < 0.05ms)
    assert median_single_ms < 0.02, (
        f"Single inference median latency {median_single_ms:.5f}ms exceeded 0.02ms"
    )

    # Mean latency must be < 0.05ms
    assert mean_single_ms < 0.05, (
        f"Single inference mean latency {mean_single_ms:.5f}ms exceeded 0.05ms"
    )

    # Tail latency bounds for Windows OS scheduler stability
    assert p95_single_ms < 0.10, f"p95 latency {p95_single_ms:.5f}ms exceeded 0.10ms"
    assert p99_single_ms < 1.00, f"p99 latency {p99_single_ms:.5f}ms exceeded 1.00ms"

    # Batch array throughput SLA: 10,000 samples in < 10ms (< 0.001ms/sample)
    assert total_arr_ms < 10.0, (
        f"10,000 array calibration took {total_arr_ms:.3f}ms, exceeded 10.0ms limit"
    )
    assert arr_out.shape == (10000, 4)

    # Batch entity throughput SLA: 1,000 samples in < 50ms (< 0.05ms/sample)
    assert total_batch_ms < 50.0, (
        f"1,000 batch calibration took {total_batch_ms:.3f}ms, exceeded 50.0ms limit"
    )
    assert len(batch_out) == 1000
