"""Fee-to-Edge Gate Domain Service — Gói 1 (FeeEdge_10) filter.

Filters out noise signals before activating Multi-Agent Playbooks and LLM deliberations.
Enforces Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
"""

from __future__ import annotations

from pydantic import BaseModel


class FeeToEdgeConfig(BaseModel):
    """Configuration for Fee-to-Edge gating (Zero Defaults)."""

    multiplier: float
    roundtrip_cost_rate: float
    min_calibrated_prob: float


class FeeToEdgeVerdict(BaseModel):
    """Verdict from Fee-to-Edge gating evaluation (Zero Defaults)."""

    is_cleared: bool
    expected_edge_bps: float
    hurdle_bps: float
    reason: str


class FeeToEdgeGatingService:
    """Evaluates expected mathematical edge against roundtrip transaction fee hurdle."""

    def __init__(self, config: FeeToEdgeConfig) -> None:
        self.config: FeeToEdgeConfig = config

    def evaluate(self, p_cal: float, garman_klass_vol: float) -> FeeToEdgeVerdict:
        """Evaluate expected edge against transaction hurdle.

        Formula:
            E[Edge] = (2 * P_cal - 1) * (1.8 * sigma_GK)
            Hurdle = multiplier * roundtrip_cost_rate
            Values expressed in basis points (rate * 10000.0).
        """
        expected_edge = (2.0 * p_cal - 1.0) * (1.8 * garman_klass_vol)
        expected_edge_bps = expected_edge * 10000.0
        hurdle = self.config.multiplier * self.config.roundtrip_cost_rate
        hurdle_bps = hurdle * 10000.0

        if p_cal < self.config.min_calibrated_prob:
            return FeeToEdgeVerdict(
                is_cleared=False,
                expected_edge_bps=expected_edge_bps,
                hurdle_bps=hurdle_bps,
                reason="CONVICTION_BELOW_MIN",
            )

        if expected_edge >= (hurdle - 1e-9):
            return FeeToEdgeVerdict(
                is_cleared=True,
                expected_edge_bps=expected_edge_bps,
                hurdle_bps=hurdle_bps,
                reason="CLEARED",
            )

        return FeeToEdgeVerdict(
            is_cleared=False,
            expected_edge_bps=expected_edge_bps,
            hurdle_bps=hurdle_bps,
            reason="EXPECTED_EDGE_BELOW_FEE_HURDLE",
        )
