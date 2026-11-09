"""Unit tests for FeeToEdgeGatingService domain service.

Verifies Gói 1 (FeeEdge_10) quantitative logic, basis points computation,
and Invariant #9 compliance.
"""

from __future__ import annotations

import pytest

from trading_bot.domain.services.execution.fee_to_edge_gate import (
    FeeToEdgeConfig,
    FeeToEdgeGatingService,
    FeeToEdgeVerdict,
)


@pytest.fixture
def fee_config() -> FeeToEdgeConfig:
    return FeeToEdgeConfig(
        multiplier=1.0,
        roundtrip_cost_rate=0.0010,  # 10 bps roundtrip cost
        min_calibrated_prob=0.52,  # 52% conviction threshold
    )


def test_fee_to_edge_gate_cleared(fee_config: FeeToEdgeConfig) -> None:
    service = FeeToEdgeGatingService(fee_config)
    # p_cal = 0.58, vol = 0.015 (1.5%)
    # expected_edge = (2 * 0.58 - 1) * (1.8 * 0.015) = 0.16 * 0.027 = 0.00432 (43.2 bps)
    # hurdle = 1.0 * 0.0010 = 0.0010 (10.0 bps)
    verdict = service.evaluate(p_cal=0.58, garman_klass_vol=0.015)

    assert isinstance(verdict, FeeToEdgeVerdict)
    assert verdict.is_cleared is True
    assert verdict.reason == "CLEARED"
    assert verdict.expected_edge_bps == pytest.approx(43.2, rel=1e-4)
    assert verdict.hurdle_bps == pytest.approx(10.0, rel=1e-4)


def test_fee_to_edge_gate_conviction_below_min(fee_config: FeeToEdgeConfig) -> None:
    service = FeeToEdgeGatingService(fee_config)
    # p_cal = 0.51 (< 0.52), vol = 0.030
    verdict = service.evaluate(p_cal=0.51, garman_klass_vol=0.030)

    assert verdict.is_cleared is False
    assert verdict.reason == "CONVICTION_BELOW_MIN"
    assert verdict.expected_edge_bps == pytest.approx((2 * 0.51 - 1) * (1.8 * 0.030) * 10000.0)
    assert verdict.hurdle_bps == pytest.approx(10.0)


def test_fee_to_edge_gate_below_fee_hurdle(fee_config: FeeToEdgeConfig) -> None:
    service = FeeToEdgeGatingService(fee_config)
    # Conviction passes (0.53 >= 0.52), but low volatility makes edge too thin
    # expected_edge = (2 * 0.53 - 1) * (1.8 * 0.002) = 0.06 * 0.0036 = 0.000216 (2.16 bps)
    # hurdle = 10.0 bps
    verdict = service.evaluate(p_cal=0.53, garman_klass_vol=0.002)

    assert verdict.is_cleared is False
    assert verdict.reason == "EXPECTED_EDGE_BELOW_FEE_HURDLE"
    assert verdict.expected_edge_bps == pytest.approx(2.16, rel=1e-4)
    assert verdict.hurdle_bps == pytest.approx(10.0, rel=1e-4)


def test_fee_to_edge_gate_exact_hurdle_threshold() -> None:
    # Set hurdle to exactly 18.0 bps
    # expected_edge = (2 * 0.60 - 1) * (1.8 * 0.005) = 0.20 * 0.009 = 0.0018 (18.0 bps)
    cfg = FeeToEdgeConfig(
        multiplier=1.8,
        roundtrip_cost_rate=0.0010,
        min_calibrated_prob=0.50,
    )
    service = FeeToEdgeGatingService(cfg)
    verdict = service.evaluate(p_cal=0.60, garman_klass_vol=0.005)

    assert verdict.is_cleared is True
    assert verdict.reason == "CLEARED"
    assert verdict.expected_edge_bps == pytest.approx(18.0, rel=1e-4)
    assert verdict.hurdle_bps == pytest.approx(18.0, rel=1e-4)


def test_fee_to_edge_gate_zero_or_negative_edge(fee_config: FeeToEdgeConfig) -> None:
    service = FeeToEdgeGatingService(fee_config)
    # p_cal = 0.50 -> expected_edge = 0.0
    verdict_neutral = service.evaluate(p_cal=0.50, garman_klass_vol=0.02)
    assert verdict_neutral.is_cleared is False
    assert verdict_neutral.reason == "CONVICTION_BELOW_MIN"
    assert verdict_neutral.expected_edge_bps == pytest.approx(0.0)

    # p_cal = 0.40 -> expected_edge < 0.0
    verdict_negative = service.evaluate(p_cal=0.40, garman_klass_vol=0.02)
    assert verdict_negative.is_cleared is False
    assert verdict_negative.reason == "CONVICTION_BELOW_MIN"
    assert verdict_negative.expected_edge_bps < 0.0
