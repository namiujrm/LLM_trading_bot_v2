"""Macro Sentinel Port — Interface for detecting macroeconomic blackout windows.

Enforces Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
Clean Architecture: Depends ONLY on standard library and Pydantic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class MacroBlackoutState(BaseModel):
    """Current state of macroeconomic event blackout (Zero Defaults)."""

    is_blackout: bool
    upcoming_event: str | None
    time_to_event_minutes: float | None
    reason: str


@runtime_checkable
class MacroSentinelPort(Protocol):
    """Port interface for querying macroeconomic blackout windows."""

    async def check_blackout(
        self,
        symbol: str,
        current_time_utc: datetime,
    ) -> MacroBlackoutState:
        """Check if symbol is currently in or approaching a macroeconomic event blackout."""
        ...
