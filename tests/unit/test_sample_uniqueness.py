"""Unit tests for López de Prado Sample Uniqueness and Weights (Zero Defaults).

Verifies concurrency calculation, harmonic average uniqueness, composite sample weighting,
mathematical conservation laws, and Fail-Fast validation.
"""

from __future__ import annotations

import pytest

from trading_bot.domain.entities.ml.barrier_labels import LabelSpan
from trading_bot.domain.entities.ml.ml_configs import SampleWeightConfig
from trading_bot.domain.exceptions import NumericalInstabilityException
from trading_bot.domain.services.ml.sample_uniqueness_weights import (
    SampleUniquenessWeightsService,
    compute_de_prado_weights,
)


@pytest.fixture
def weight_config() -> SampleWeightConfig:
    return SampleWeightConfig(min_volatility_epsilon=1e-6)


@pytest.fixture
def service(weight_config: SampleWeightConfig) -> SampleUniquenessWeightsService:
    return SampleUniquenessWeightsService(config=weight_config)


def test_empty_events_returns_empty_results(service: SampleUniquenessWeightsService) -> None:
    events: list[LabelSpan] = []
    concurrency = service.compute_concurrency(events=events, num_bars=10)
    assert concurrency == [0] * 10

    uniqueness = service.compute_average_uniqueness(events=events, concurrency=concurrency)
    assert uniqueness == []

    records = service.compute_weights(events=events, num_bars=10)
    assert records == []


def test_single_sample_evaluation(service: SampleUniquenessWeightsService) -> None:
    events = [
        LabelSpan(
            sample_id="s0",
            t_start=2,
            t_end=5,
            realized_return=0.03,
            volatility=0.015,
        )
    ]
    concurrency = service.compute_concurrency(events=events, num_bars=8)
    expected_c = [0, 0, 1, 1, 1, 1, 0, 0]
    assert concurrency == expected_c

    records = service.compute_weights(events=events, num_bars=8)
    assert len(records) == 1
    rec = records[0]
    assert rec.sample_id == "s0"
    assert rec.concurrency_avg == pytest.approx(1.0)
    assert rec.uniqueness == pytest.approx(1.0)
    assert rec.normalized_weight == pytest.approx(1.0)


def test_non_overlapping_samples_have_unit_uniqueness(
    service: SampleUniquenessWeightsService,
) -> None:
    events = [
        LabelSpan(sample_id="s0", t_start=0, t_end=2, realized_return=0.02, volatility=0.01),
        LabelSpan(sample_id="s1", t_start=3, t_end=5, realized_return=0.02, volatility=0.01),
        LabelSpan(sample_id="s2", t_start=6, t_end=8, realized_return=0.02, volatility=0.01),
    ]
    concurrency = service.compute_concurrency(events=events, num_bars=10)
    assert max(concurrency) == 1

    records = service.compute_weights(events=events, num_bars=10)
    assert len(records) == 3
    for r in records:
        assert r.uniqueness == pytest.approx(1.0)
        assert r.concurrency_avg == pytest.approx(1.0)
        assert r.normalized_weight == pytest.approx(1.0)


def test_completely_overlapping_samples_decay_uniqueness(
    service: SampleUniquenessWeightsService,
) -> None:
    # 4 identical samples on bars [2, 6]
    events = [
        LabelSpan(sample_id=f"s{i}", t_start=2, t_end=6, realized_return=0.04, volatility=0.02)
        for i in range(4)
    ]
    concurrency = service.compute_concurrency(events=events, num_bars=10)
    assert concurrency[2:7] == [4, 4, 4, 4, 4]

    records = service.compute_weights(events=events, num_bars=10)
    assert len(records) == 4
    for r in records:
        assert r.concurrency_avg == pytest.approx(4.0)
        assert r.uniqueness == pytest.approx(0.25)
        assert r.normalized_weight == pytest.approx(1.0)  # All identical, so normalized to 1.0


def test_hand_calculated_partial_overlap_exact_values(
    service: SampleUniquenessWeightsService,
) -> None:
    """Exact verification against analytical derivation:

    Sample 0: [0, 2], ret=0.02, vol=0.01
    Sample 1: [1, 3], ret=0.04, vol=0.02
    Sample 2: [3, 4], ret=0.01, vol=0.01
    Timeline: [0, 1, 2, 3, 4] -> c_t = [1, 2, 2, 2, 1]
    u_0 = (1 + 0.5 + 0.5) / 3 = 2/3
    u_1 = (0.5 + 0.5 + 0.5) / 3 = 1/2
    u_2 = (0.5 + 1.0) / 2 = 3/4
    w_raw_0 = (2/3) * (0.02 / 0.01) = 4/3
    w_raw_1 = (1/2) * (0.04 / 0.02) = 1.0
    w_raw_2 = (3/4) * (0.01 / 0.01) = 3/4
    sum_w_raw = 4/3 + 1 + 3/4 = 37/12
    Scale = 3 / (37/12) = 36/37
    w_norm_0 = (4/3) * (36/37) = 48/37 =~ 1.297297
    w_norm_1 = 1.0 * (36/37) = 36/37 =~ 0.972973
    w_norm_2 = (3/4) * (36/37) = 27/37 =~ 0.729730
    """
    events = [
        LabelSpan(sample_id="s0", t_start=0, t_end=2, realized_return=0.02, volatility=0.01),
        LabelSpan(sample_id="s1", t_start=1, t_end=3, realized_return=0.04, volatility=0.02),
        LabelSpan(sample_id="s2", t_start=3, t_end=4, realized_return=0.01, volatility=0.01),
    ]
    concurrency = service.compute_concurrency(events=events, num_bars=5)
    assert concurrency == [1, 2, 2, 2, 1]

    records = service.compute_weights(events=events, num_bars=5)
    assert len(records) == 3

    assert records[0].uniqueness == pytest.approx(2.0 / 3.0)
    assert records[1].uniqueness == pytest.approx(0.5)
    assert records[2].uniqueness == pytest.approx(0.75)

    assert records[0].raw_weight == pytest.approx(4.0 / 3.0)
    assert records[1].raw_weight == pytest.approx(1.0)
    assert records[2].raw_weight == pytest.approx(0.75)

    assert records[0].normalized_weight == pytest.approx(48.0 / 37.0)
    assert records[1].normalized_weight == pytest.approx(36.0 / 37.0)
    assert records[2].normalized_weight == pytest.approx(27.0 / 37.0)

    # Invariants
    total_weight = sum(r.normalized_weight for r in records)
    assert total_weight == pytest.approx(3.0)
    assert (total_weight / 3.0) == pytest.approx(1.0)


def test_all_zero_returns_yields_unit_normalized_weights(
    service: SampleUniquenessWeightsService,
) -> None:
    events = [
        LabelSpan(sample_id="s0", t_start=0, t_end=2, realized_return=0.0, volatility=0.01),
        LabelSpan(sample_id="s1", t_start=1, t_end=3, realized_return=0.0, volatility=0.02),
    ]
    records = service.compute_weights(events=events, num_bars=4)
    assert len(records) == 2
    for r in records:
        assert r.raw_weight == 0.0
        assert r.normalized_weight == 1.0
    assert sum(r.normalized_weight for r in records) == 2.0


def test_negative_return_uses_absolute_magnitude(service: SampleUniquenessWeightsService) -> None:
    events = [
        LabelSpan(sample_id="s_pos", t_start=0, t_end=3, realized_return=0.05, volatility=0.01),
        LabelSpan(sample_id="s_neg", t_start=4, t_end=7, realized_return=-0.05, volatility=0.01),
    ]
    records = service.compute_weights(events=events, num_bars=8)
    assert len(records) == 2
    assert records[0].raw_weight == pytest.approx(records[1].raw_weight)
    assert records[0].normalized_weight == pytest.approx(records[1].normalized_weight)
    assert records[0].normalized_weight == pytest.approx(1.0)


def test_zero_volatility_clipped_to_epsilon(service: SampleUniquenessWeightsService) -> None:
    events = [
        LabelSpan(sample_id="s0", t_start=0, t_end=1, realized_return=0.02, volatility=0.0),
        LabelSpan(sample_id="s1", t_start=2, t_end=3, realized_return=0.02, volatility=0.0),
    ]
    records = service.compute_weights(events=events, num_bars=4)
    assert len(records) == 2
    # volatility denominator is 1e-6
    expected_raw = 1.0 * (0.02 / 1e-6)
    assert records[0].raw_weight == pytest.approx(expected_raw)
    assert records[0].normalized_weight == pytest.approx(1.0)
    assert records[1].normalized_weight == pytest.approx(1.0)


def test_single_bar_holding_span(service: SampleUniquenessWeightsService) -> None:
    events = [
        LabelSpan(sample_id="s0", t_start=3, t_end=3, realized_return=0.01, volatility=0.01),
        LabelSpan(sample_id="s1", t_start=3, t_end=3, realized_return=0.02, volatility=0.01),
    ]
    concurrency = service.compute_concurrency(events=events, num_bars=5)
    assert concurrency[3] == 2

    records = service.compute_weights(events=events, num_bars=5)
    assert records[0].uniqueness == pytest.approx(0.5)
    assert records[1].uniqueness == pytest.approx(0.5)
    # raw weights: 0.5 * 1.0 = 0.5, 0.5 * 2.0 = 1.0. Sum = 1.5, N = 2.
    # norm weights: 0.5 * (2 / 1.5) = 2/3, 1.0 * (2 / 1.5) = 4/3.
    assert records[0].normalized_weight == pytest.approx(2.0 / 3.0)
    assert records[1].normalized_weight == pytest.approx(4.0 / 3.0)
    assert sum(r.normalized_weight for r in records) == pytest.approx(2.0)


def test_prefix_sum_matches_direct_summation(service: SampleUniquenessWeightsService) -> None:
    events = [
        LabelSpan(sample_id="s0", t_start=1, t_end=6, realized_return=0.02, volatility=0.01),
        LabelSpan(sample_id="s1", t_start=3, t_end=8, realized_return=0.03, volatility=0.01),
        LabelSpan(sample_id="s2", t_start=5, t_end=9, realized_return=0.01, volatility=0.01),
    ]
    num_bars = 12
    concurrency = service.compute_concurrency(events=events, num_bars=num_bars)
    uniqueness_fast = service.compute_average_uniqueness(events=events, concurrency=concurrency)

    # Compute naive direct loop
    for i, ev in enumerate(events):
        direct_sum = sum(1.0 / concurrency[t] for t in range(ev.t_start, ev.t_end + 1))
        direct_u = direct_sum / float(ev.t_end - ev.t_start + 1)
        assert uniqueness_fast[i] == pytest.approx(direct_u, rel=1e-12)


def test_functional_compute_de_prado_weights(weight_config: SampleWeightConfig) -> None:
    events = [
        LabelSpan(sample_id="s0", t_start=0, t_end=1, realized_return=0.01, volatility=0.01),
        LabelSpan(sample_id="s1", t_start=2, t_end=3, realized_return=0.02, volatility=0.01),
    ]
    weights = compute_de_prado_weights(events=events, num_bars=5, config=weight_config)
    assert len(weights) == 2
    # w_raw: 1.0 * 1.0 = 1.0, 1.0 * 2.0 = 2.0. Sum = 3.0, N = 2.
    # norm: 1.0 * (2/3) = 2/3, 2.0 * (2/3) = 4/3.
    assert weights[0] == pytest.approx(2.0 / 3.0)
    assert weights[1] == pytest.approx(4.0 / 3.0)
    assert sum(weights) == pytest.approx(2.0)


def test_validation_errors_fail_fast(service: SampleUniquenessWeightsService) -> None:
    # 1. t_start > t_end
    with pytest.raises(ValueError, match=r"t_start \(5\) > t_end \(2\)"):
        service.compute_weights(
            events=[
                LabelSpan(
                    sample_id="err", t_start=5, t_end=2, realized_return=0.01, volatility=0.01
                )
            ],
            num_bars=10,
        )

    # 2. t_start < 0
    with pytest.raises(ValueError, match="negative t_start"):
        service.compute_weights(
            events=[
                LabelSpan(
                    sample_id="err", t_start=-1, t_end=2, realized_return=0.01, volatility=0.01
                )
            ],
            num_bars=10,
        )

    # 3. t_end >= num_bars
    with pytest.raises(ValueError, match=r"t_end \(10\) >= num_bars \(10\)"):
        service.compute_weights(
            events=[
                LabelSpan(
                    sample_id="err", t_start=2, t_end=10, realized_return=0.01, volatility=0.01
                )
            ],
            num_bars=10,
        )

    # 4. negative volatility
    with pytest.raises(ValueError, match="negative volatility"):
        service.compute_weights(
            events=[
                LabelSpan(
                    sample_id="err", t_start=2, t_end=4, realized_return=0.01, volatility=-0.01
                )
            ],
            num_bars=10,
        )

    # 5. non-finite return
    with pytest.raises(NumericalInstabilityException, match="non-finite return"):
        service.compute_weights(
            events=[
                LabelSpan(
                    sample_id="err",
                    t_start=2,
                    t_end=4,
                    realized_return=float("nan"),
                    volatility=0.01,
                )
            ],
            num_bars=10,
        )

    # 6. non-finite volatility
    with pytest.raises(NumericalInstabilityException, match="non-finite volatility"):
        service.compute_weights(
            events=[
                LabelSpan(
                    sample_id="err",
                    t_start=2,
                    t_end=4,
                    realized_return=0.01,
                    volatility=float("inf"),
                )
            ],
            num_bars=10,
        )

    # 7. Invalid config epsilon
    with pytest.raises(ValueError, match="min_volatility_epsilon"):
        SampleUniquenessWeightsService(config=SampleWeightConfig(min_volatility_epsilon=0.0))


def test_50k_concurrent_spans_latency_sla_under_200ms(
    service: SampleUniquenessWeightsService,
) -> None:
    """Benchmark verifying compute_normalized_weights processes 50,000 highly concurrent spans in < 200ms."""
    import random
    import time

    # Generate 50,000 synthetic spans over 1,000 bars with deterministic seed
    random.seed(42)
    n_spans = 50_000
    num_bars = 1_000
    events: list[LabelSpan] = []
    for i in range(n_spans):
        t_start = random.randint(0, num_bars - 20)
        t_end = t_start + random.randint(1, 18)
        events.append(
            LabelSpan(
                sample_id=f"span_{i}",
                t_start=t_start,
                t_end=t_end,
                realized_return=random.gauss(0.001, 0.02),
                volatility=max(1e-5, random.gauss(0.015, 0.005)),
            )
        )

    # High-resolution benchmark of pure service execution
    t0 = time.perf_counter_ns()
    weights = service.compute_normalized_weights(events=events, num_bars=num_bars)
    t1 = time.perf_counter_ns()
    elapsed_ms = (t1 - t0) / 1_000_000.0

    # 1. Structural invariants
    assert len(weights) == n_spans

    # 2. Mathematical conservation law: sum(w_norm) == N
    assert sum(weights) == pytest.approx(float(n_spans), rel=1e-9)

    # 3. Latency SLA constraint (< 200ms)
    assert elapsed_ms < 200.0, f"Latency SLA violated: {elapsed_ms:.2f}ms >= 200.0ms target"
