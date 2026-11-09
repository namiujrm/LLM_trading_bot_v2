"""Rate Limit Governor Port — Interface for multi-provider API rate pacing and route allocation.

Enforces Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class RouteQuotaSpec(BaseModel):
    """Specification of rate limits and concurrency constraints for a single model route."""

    route_id: str
    model_name: str
    rpm_limit: int
    tpm_limit: int
    rpd_limit: int
    min_interval_ms: int
    max_concurrency: int


@runtime_checkable
class RateLimitGovernorPort(Protocol):
    """Port interface for acquiring available LLM routes and managing provider cooldowns."""

    async def acquire_available_route(self, tier: str, estimated_tokens: int) -> RouteQuotaSpec:
        """Acquire an available route within the given tier that satisfies rate limits."""
        ...

    async def mark_route_cooldown(self, route_id: str, cooldown_seconds: float) -> None:
        """Mark a route into cooldown for the specified duration (e.g. following HTTP 429)."""
        ...
