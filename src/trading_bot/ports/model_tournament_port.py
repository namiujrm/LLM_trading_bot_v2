"""Model Tournament Notifier Port (Clean Architecture, Invariant #9).

Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks.
Depends ONLY on domain entities and standard typing.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from trading_bot.domain.entities.quant.symbolic_alpha import (
    ModelTournamentReport,
    TournamentVerdict,
)


@runtime_checkable
class ModelTournamentNotifierPort(Protocol):
    """Protocol port interface for presenting Champion vs. Challenger tournament reports."""

    async def present_tournament_report(
        self,
        report: ModelTournamentReport,
    ) -> TournamentVerdict:
        """Present tournament comparison report and obtain promotion verdict."""
        ...
