"""Execution, playbook validation, fee gating, routing, and barrier evaluation domain services."""

from __future__ import annotations

from trading_bot.domain.services.execution.barrier_evaluator import (
    BarrierConfig,
    BarrierEvaluator,
)
from trading_bot.domain.services.execution.fee_to_edge_gate import (
    FeeToEdgeConfig,
    FeeToEdgeGatingService,
    FeeToEdgeVerdict,
)
from trading_bot.domain.services.execution.playbook_validator import (
    PlaybookConfig,
    PlaybookPreconditionsInput,
    PlaybookValidationResult,
    PlaybookValidator,
)
from trading_bot.domain.services.execution.routing_matrix import (
    DynamicRoutingMatrix,
    RoutingDecision,
    RoutingRequest,
)

__all__ = [
    "BarrierConfig",
    "BarrierEvaluator",
    "DynamicRoutingMatrix",
    "FeeToEdgeConfig",
    "FeeToEdgeGatingService",
    "FeeToEdgeVerdict",
    "PlaybookConfig",
    "PlaybookPreconditionsInput",
    "PlaybookValidationResult",
    "PlaybookValidator",
    "RoutingDecision",
    "RoutingRequest",
]
