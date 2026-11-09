"""Unit tests for DynamicRoutingMatrix domain service.

Verifies state-to-model tier assignment, execution priority, and Invariant #9 compliance.
"""

from __future__ import annotations

import pytest

from trading_bot.domain.entities.common.enums import PlaybookType
from trading_bot.domain.services.execution.routing_matrix import (
    DynamicRoutingMatrix,
    RoutingDecision,
    RoutingRequest,
)


@pytest.fixture
def routing_matrix() -> DynamicRoutingMatrix:
    return DynamicRoutingMatrix(
        fast_tier_name="fast_tactical",
        deep_tier_name="deep_reasoning",
        max_uncertainty_for_fast=0.10,
    )


def test_routing_matrix_no_trade_regime_skip(
    routing_matrix: DynamicRoutingMatrix,
) -> None:
    req = RoutingRequest(
        playbook=PlaybookType.NO_TRADE_REGIME,
        symbol="BTCUSDT",
        is_btc_lead=True,
        uncertainty_width=0.04,
        is_ood=False,
    )
    decision = routing_matrix.resolve_route(req)

    assert isinstance(decision, RoutingDecision)
    assert decision.tier == "fast_tactical"
    assert decision.priority == 3
    assert decision.rationale == "NO_TRADE_REGIME_SKIP"


def test_routing_matrix_high_uncertainty_routes_deep(
    routing_matrix: DynamicRoutingMatrix,
) -> None:
    # uncertainty_width = 0.15 > 0.10
    req_btc = RoutingRequest(
        playbook=PlaybookType.TREND_CONTINUATION_BREAKOUT,
        symbol="BTCUSDT",
        is_btc_lead=True,
        uncertainty_width=0.15,
        is_ood=False,
    )
    decision_btc = routing_matrix.resolve_route(req_btc)

    assert decision_btc.tier == "deep_reasoning"
    assert decision_btc.priority == 1
    assert decision_btc.rationale == "DEEP_REASONING_COMPLEX_REGIME"

    # Altcoin high uncertainty -> priority 2
    req_alt = RoutingRequest(
        playbook=PlaybookType.TREND_CONTINUATION_BREAKOUT,
        symbol="SOLUSDT",
        is_btc_lead=False,
        uncertainty_width=0.15,
        is_ood=False,
    )
    decision_alt = routing_matrix.resolve_route(req_alt)

    assert decision_alt.tier == "deep_reasoning"
    assert decision_alt.priority == 2
    assert decision_alt.rationale == "DEEP_REASONING_COMPLEX_REGIME"


def test_routing_matrix_liquidation_reversal_always_deep(
    routing_matrix: DynamicRoutingMatrix,
) -> None:
    # Low uncertainty (0.03 <= 0.10), but playbook is LIQUIDATION_EXHAUSTION_REVERSAL -> forces deep tier
    req = RoutingRequest(
        playbook=PlaybookType.LIQUIDATION_EXHAUSTION_REVERSAL,
        symbol="ETHUSDT",
        is_btc_lead=False,
        uncertainty_width=0.03,
        is_ood=False,
    )
    decision = routing_matrix.resolve_route(req)

    assert decision.tier == "deep_reasoning"
    assert decision.priority == 2
    assert decision.rationale == "DEEP_REASONING_COMPLEX_REGIME"


def test_routing_matrix_fast_tactical_execution(
    routing_matrix: DynamicRoutingMatrix,
) -> None:
    # Normal trend or general alpha with low uncertainty
    req_btc = RoutingRequest(
        playbook=PlaybookType.GENERAL_ALPHA_MOMENTUM,
        symbol="BTCUSDT",
        is_btc_lead=True,
        uncertainty_width=0.05,
        is_ood=False,
    )
    decision_btc = routing_matrix.resolve_route(req_btc)

    assert decision_btc.tier == "fast_tactical"
    assert decision_btc.priority == 1
    assert decision_btc.rationale == "FAST_TACTICAL_EXECUTION"

    req_alt = RoutingRequest(
        playbook=PlaybookType.CROSS_ASSET_LEAD_LAG_CATCHUP,
        symbol="BNBUSDT",
        is_btc_lead=False,
        uncertainty_width=0.05,
        is_ood=False,
    )
    decision_alt = routing_matrix.resolve_route(req_alt)

    assert decision_alt.tier == "fast_tactical"
    assert decision_alt.priority == 2
    assert decision_alt.rationale == "FAST_TACTICAL_EXECUTION"
