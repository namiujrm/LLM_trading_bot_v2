"""Dynamic Routing Matrix Domain Service — State-to-Model Tier and Priority Dispatch.

Routes trade reasoning between fast tactical tiers (e.g. Claude Haiku / Gemini Flash)
and deep reasoning tiers (e.g. Claude Sonnet / DeepSeek R1).
Enforces Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
"""

from __future__ import annotations

from pydantic import BaseModel

from trading_bot.domain.entities.common.enums import PlaybookType


class RoutingRequest(BaseModel):
    """Input parameters for model routing resolution (Zero Defaults)."""

    playbook: PlaybookType
    symbol: str
    is_btc_lead: bool
    uncertainty_width: float
    is_ood: bool


class RoutingDecision(BaseModel):
    """Resolved model tier and priority decision (Zero Defaults)."""

    tier: str
    priority: int
    rationale: str


class DynamicRoutingMatrix:
    """Dispatches routing requests to fast or deep LLM inference tiers."""

    def __init__(
        self,
        fast_tier_name: str,
        deep_tier_name: str,
        max_uncertainty_for_fast: float,
    ) -> None:
        self.fast_tier_name: str = fast_tier_name
        self.deep_tier_name: str = deep_tier_name
        self.max_uncertainty_for_fast: float = max_uncertainty_for_fast

    def resolve_route(self, request: RoutingRequest) -> RoutingDecision:
        """Resolve route tier and execution priority.

        Rules:
        - If request.playbook == NO_TRADE_REGIME:
            tier=fast_tier_name, priority=3, rationale="NO_TRADE_REGIME_SKIP".
        - If uncertainty_width > max_uncertainty_for_fast or playbook == LIQUIDATION_EXHAUSTION_REVERSAL:
            tier=deep_tier_name, priority=(1 if request.is_btc_lead else 2), rationale="DEEP_REASONING_COMPLEX_REGIME".
        - Else:
            tier=fast_tier_name, priority=(1 if request.is_btc_lead else 2), rationale="FAST_TACTICAL_EXECUTION".
        """
        if request.playbook == PlaybookType.NO_TRADE_REGIME:
            return RoutingDecision(
                tier=self.fast_tier_name,
                priority=3,
                rationale="NO_TRADE_REGIME_SKIP",
            )

        priority = 1 if request.is_btc_lead else 2

        if (
            request.uncertainty_width > self.max_uncertainty_for_fast
            or request.playbook == PlaybookType.LIQUIDATION_EXHAUSTION_REVERSAL
        ):
            return RoutingDecision(
                tier=self.deep_tier_name,
                priority=priority,
                rationale="DEEP_REASONING_COMPLEX_REGIME",
            )

        return RoutingDecision(
            tier=self.fast_tier_name,
            priority=priority,
            rationale="FAST_TACTICAL_EXECUTION",
        )
