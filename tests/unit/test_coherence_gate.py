"""Unit and Property Tests for CoherenceGate Domain Service (Proposal P3).

Verifies the mathematical correctness of dual-binary coherence enforcement:
- Mutual exclusivity axiom: P_long + P_short <= 1.0
- Preservation of coherent probabilities (S <= 1.0)
- Conflict attenuation (S > 1.0)
- Extreme whipsaw circuit breaker (alpha = 0.0)
- Conviction ratio preservation
- Strict Fail-Fast input validation (Invariant #9)
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from trading_bot.domain.entities.ml.ml_configs import CoherenceGateConfig
from trading_bot.domain.entities.ml.ml_predictions import BarrierPredictionResult
from trading_bot.domain.services.ml.coherence_gate import (
    CoherenceGate,
    enforce_dual_binary_coherence,
)


@pytest.fixture
def default_coherence_config() -> CoherenceGateConfig:
    return CoherenceGateConfig(
        gamma_conflict=1.0,
        max_combined_probability=1.0,
        min_probability_floor=0.0,
    )


def test_coherent_inputs_preserve_exact_values(
    default_coherence_config: CoherenceGateConfig,
) -> None:
    gate = CoherenceGate(config=default_coherence_config)
    res = gate.enforce_coherence(p_raw_long=0.40, p_raw_short=0.35)

    assert res.p_long == pytest.approx(0.40)
    assert res.p_short == pytest.approx(0.35)
    assert res.p_raw_long == pytest.approx(0.40)
    assert res.p_raw_short == pytest.approx(0.35)
    assert res.is_coherent is True
    assert res.coherence_penalty == pytest.approx(0.0)
    assert res.raw_sum == pytest.approx(0.75)
    assert res.coherent_sum == pytest.approx(0.75)
    assert res.raw_conflict_excess == pytest.approx(0.0)


def test_boundary_input_sum_exactly_one(default_coherence_config: CoherenceGateConfig) -> None:
    gate = CoherenceGate(config=default_coherence_config)
    res = gate.enforce_coherence(p_raw_long=0.60, p_raw_short=0.40)

    assert res.p_long == pytest.approx(0.60)
    assert res.p_short == pytest.approx(0.40)
    assert res.is_coherent is True
    assert res.coherence_penalty == pytest.approx(0.0)
    assert res.coherent_sum == pytest.approx(1.0)


def test_incoherent_input_attenuation(default_coherence_config: CoherenceGateConfig) -> None:
    gate = CoherenceGate(config=default_coherence_config)
    # p_raw_long = 0.85, p_raw_short = 0.82 -> raw_sum = 1.67, excess = 0.67
    # alpha = 1.0 - 1.0 * 0.67 = 0.33
    res = gate.enforce_coherence(p_raw_long=0.85, p_raw_short=0.82)

    assert res.is_coherent is False
    assert res.coherence_penalty == pytest.approx(0.67)
    assert res.p_long == pytest.approx(0.85 * 0.33)
    assert res.p_short == pytest.approx(0.82 * 0.33)
    assert res.coherent_sum <= 1.0
    assert res.raw_sum == pytest.approx(1.67)
    assert res.raw_conflict_excess == pytest.approx(0.67)


def test_extreme_conflict_circuit_breaker() -> None:
    config = CoherenceGateConfig(
        gamma_conflict=1.5,
        max_combined_probability=1.0,
        min_probability_floor=0.0,
    )
    gate = CoherenceGate(config=config)
    # p_raw_long = 0.90, p_raw_short = 0.90 -> raw_sum = 1.80, excess = 0.80
    # penalty = 1.5 * 0.80 = 1.20 -> alpha = max(0, 1 - 1.20) = 0.0
    res = gate.enforce_coherence(p_raw_long=0.90, p_raw_short=0.90)

    assert res.is_coherent is False
    assert res.coherence_penalty == pytest.approx(1.0)
    assert res.p_long == pytest.approx(0.0)
    assert res.p_short == pytest.approx(0.0)
    assert res.coherent_sum == pytest.approx(0.0)


def test_ratio_preservation_when_attenuated(default_coherence_config: CoherenceGateConfig) -> None:
    gate = CoherenceGate(config=default_coherence_config)
    res = gate.enforce_coherence(p_raw_long=0.80, p_raw_short=0.40)

    assert res.is_coherent is False
    assert res.p_long > 0.0
    assert res.p_short > 0.0
    # Ratio p_long / p_short must equal p_raw_long / p_raw_short = 2.0
    assert (res.p_long / res.p_short) == pytest.approx(res.p_raw_long / res.p_raw_short)


def test_enforce_batch_consistency(default_coherence_config: CoherenceGateConfig) -> None:
    gate = CoherenceGate(config=default_coherence_config)
    pairs = [
        (0.2, 0.3),
        (0.8, 0.6),
        (0.0, 0.0),
        (1.0, 1.0),
    ]
    batch_res = gate.enforce_batch(pairs)
    assert len(batch_res) == len(pairs)
    for pair, res in zip(pairs, batch_res, strict=True):
        single_res = gate.enforce_coherence(p_raw_long=pair[0], p_raw_short=pair[1])
        assert res.p_long == single_res.p_long
        assert res.p_short == single_res.p_short
        assert res.is_coherent == single_res.is_coherent
        assert res.coherence_penalty == single_res.coherence_penalty


def test_functional_entrypoint(default_coherence_config: CoherenceGateConfig) -> None:
    res = enforce_dual_binary_coherence(
        p_raw_long=0.7,
        p_raw_short=0.5,
        config=default_coherence_config,
    )
    assert res.is_coherent is False
    assert res.coherent_sum <= 1.0


def test_fail_fast_on_invalid_probabilities(default_coherence_config: CoherenceGateConfig) -> None:
    gate = CoherenceGate(config=default_coherence_config)

    with pytest.raises(ValueError, match="finite float"):
        gate.enforce_coherence(p_raw_long=float("nan"), p_raw_short=0.5)

    with pytest.raises(ValueError, match="finite float"):
        gate.enforce_coherence(p_raw_long=0.5, p_raw_short=float("inf"))

    with pytest.raises(ValueError, match="range"):
        gate.enforce_coherence(p_raw_long=-0.1, p_raw_short=0.5)

    with pytest.raises(ValueError, match="range"):
        gate.enforce_coherence(p_raw_long=0.5, p_raw_short=1.1)


def test_fail_fast_on_invalid_config() -> None:
    with pytest.raises(ValueError, match="gamma_conflict"):
        CoherenceGateConfig(
            gamma_conflict=0.0,
            max_combined_probability=1.0,
            min_probability_floor=0.0,
        )

    with pytest.raises(ValueError, match="max_combined_probability"):
        CoherenceGateConfig(
            gamma_conflict=1.0,
            max_combined_probability=0.0,
            min_probability_floor=0.0,
        )

    with pytest.raises(ValueError, match="min_probability_floor"):
        CoherenceGateConfig(
            gamma_conflict=1.0,
            max_combined_probability=1.0,
            min_probability_floor=-0.1,
        )

    with pytest.raises(ValueError, match="less than"):
        CoherenceGateConfig(
            gamma_conflict=1.0,
            max_combined_probability=1.0,
            min_probability_floor=1.0,
        )

    with pytest.raises(ValueError, match="max_combined_probability"):
        CoherenceGateConfig(
            gamma_conflict=1.0,
            max_combined_probability=1.05,
            min_probability_floor=0.0,
        )


@given(
    p_long=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    p_short=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    gamma=st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_property_coherence_guarantee_universal(
    p_long: float, p_short: float, gamma: float
) -> None:
    """Mathematical property test: for all p_long, p_short in [0, 1] and gamma >= 1.0,
    adjusted probabilities NEVER exceed 1.0, NEVER inflate beyond raw, and stay in [0, 1].
    """
    config = CoherenceGateConfig(
        gamma_conflict=gamma,
        max_combined_probability=1.0,
        min_probability_floor=0.0,
    )
    gate = CoherenceGate(config=config)
    res = gate.enforce_coherence(p_raw_long=p_long, p_raw_short=p_short)

    assert 0.0 <= res.p_long <= 1.0
    assert 0.0 <= res.p_short <= 1.0
    assert res.p_long <= p_long + 1e-7
    assert res.p_short <= p_short + 1e-7
    assert (res.p_long + res.p_short) <= 1.0 + 1e-7


def test_boundary_epsilon_inputs(default_coherence_config: CoherenceGateConfig) -> None:
    gate = CoherenceGate(config=default_coherence_config)

    # 1. Floating-point boundary from Challenger 1: p_long = 1.0, p_short = 5.96e-8
    res1 = gate.enforce_coherence(p_raw_long=1.0, p_raw_short=5.960464477539063e-08)
    assert res1.is_coherent is False
    assert 0.0 <= res1.p_long <= 1.0
    assert 0.0 <= res1.p_short <= 1.0
    assert res1.coherent_sum <= 1.0 + 1e-7
    assert res1.p_long <= 1.0 + 1e-7

    # 2. Boundary from Challenger 2: p_long = 0.5, p_short = 0.5 + 1e-8
    res2 = gate.enforce_coherence(p_raw_long=0.5, p_raw_short=0.5 + 1e-8)
    assert res2.is_coherent is False
    assert res2.coherent_sum <= 1.0 + 1e-7
    assert res2.p_long <= 0.5 + 1e-7
    assert res2.p_short <= 0.5 + 1e-8 + 1e-7

    # 3. Sub-epsilon coherent boundary: p_long = 0.5, p_short = 0.5 + 5e-10
    res3 = gate.enforce_coherence(p_raw_long=0.5, p_raw_short=0.5 + 5e-10)
    assert res3.is_coherent is True
    assert res3.coherent_sum <= 1.0 + 1e-7
    assert res3.coherence_penalty == pytest.approx(0.0)


def test_custom_max_combined_probability_sub_one() -> None:
    config = CoherenceGateConfig(
        gamma_conflict=1.0,
        max_combined_probability=0.90,
        min_probability_floor=0.0,
    )
    gate = CoherenceGate(config=config)
    # raw_sum = 0.92 > max_comb (0.90), but raw_sum <= 1.0
    res = gate.enforce_coherence(p_raw_long=0.46, p_raw_short=0.46)
    assert res.is_coherent is False
    assert res.raw_sum == pytest.approx(0.92)
    assert res.coherent_sum <= 0.90 + 1e-7
    assert res.coherence_penalty > 0.0


def test_barrier_prediction_result_invariants_direct() -> None:
    # 1. Valid coherent prediction
    res = BarrierPredictionResult(
        p_long=0.4,
        p_short=0.3,
        p_raw_long=0.4,
        p_raw_short=0.3,
        is_coherent=True,
        coherence_penalty=0.0,
    )
    assert res.is_coherent is True

    # 2. Coherent but raw_sum > 1.0 must fail
    with pytest.raises(ValueError, match="is_coherent cannot be True"):
        BarrierPredictionResult(
            p_long=0.5,
            p_short=0.5,
            p_raw_long=0.6,
            p_raw_short=0.5,
            is_coherent=True,
            coherence_penalty=0.0,
        )

    # 3. Coherent but coherence_penalty > 0 must fail
    with pytest.raises(ValueError, match=r"coherence_penalty must be 0\.0"):
        BarrierPredictionResult(
            p_long=0.4,
            p_short=0.3,
            p_raw_long=0.4,
            p_raw_short=0.3,
            is_coherent=True,
            coherence_penalty=0.1,
        )

    # 4. Incoherent with raw_sum <= 1.0 (from custom sub-one threshold) is allowed
    res_incoherent = BarrierPredictionResult(
        p_long=0.44,
        p_short=0.44,
        p_raw_long=0.46,
        p_raw_short=0.46,
        is_coherent=False,
        coherence_penalty=0.02,
    )
    assert res_incoherent.is_coherent is False

    # 5. Inflation: p_long > p_raw_long must fail
    with pytest.raises(ValueError, match="cannot exceed raw probability"):
        BarrierPredictionResult(
            p_long=0.55,
            p_short=0.3,
            p_raw_long=0.5,
            p_raw_short=0.3,
            is_coherent=False,
            coherence_penalty=0.0,
        )

    # 6. Coherence violation: p_long + p_short > 1.0 must fail
    with pytest.raises(ValueError, match="Coherence violation"):
        BarrierPredictionResult(
            p_long=0.6,
            p_short=0.5,
            p_raw_long=0.7,
            p_raw_short=0.7,
            is_coherent=False,
            coherence_penalty=0.2,
        )

    # 7. Negative penalty must fail
    with pytest.raises(ValueError, match="coherence_penalty must be non-negative"):
        BarrierPredictionResult(
            p_long=0.4,
            p_short=0.3,
            p_raw_long=0.4,
            p_raw_short=0.3,
            is_coherent=False,
            coherence_penalty=-0.05,
        )
