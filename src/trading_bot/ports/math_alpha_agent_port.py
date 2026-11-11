"""Math Alpha Agent Port (Clean Architecture, Invariant #9).

Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks.
Depends ONLY on domain entities and standard typing.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from trading_bot.domain.entities.quant.symbolic_alpha import (
    ALL_CAUSAL_CLUSTERS,
    CLUSTER_0_REGIME,
    CLUSTER_1_MICRO,
    CLUSTER_2_DERIV,
    CLUSTER_3_CROSS,
    SymbolicAlphaCandidate,
)

__all__ = [
    "ALL_CAUSAL_CLUSTERS",
    "CLUSTER_0_REGIME",
    "CLUSTER_1_MICRO",
    "CLUSTER_2_DERIV",
    "CLUSTER_3_CROSS",
    "MathAlphaAgentPort",
]


@runtime_checkable
class MathAlphaAgentPort(Protocol):
    """Protocol port interface for Math Alpha Agent generating symbolic expressions."""

    async def generate_candidates_for_cluster(
        self,
        cluster_name: str,
        blindspots: list[dict[str, Any]],
        decayed_alphas: list[str],
        target_count: int,
    ) -> list[SymbolicAlphaCandidate]:
        """Generate symbolic alpha candidates targeted at a specific causal cluster."""
        ...
