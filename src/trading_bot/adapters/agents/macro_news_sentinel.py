"""Macro News Sentinel Adapter — Evaluates economic calendar schedules for blackout windows.

Enforces Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
Clean Architecture: Implements MacroSentinelPort from trading_bot.ports.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from pydantic import BaseModel

from trading_bot.ports.macro_sentinel_port import (
    MacroBlackoutState,
    MacroSentinelPort,
)


class MacroEventSchedule(BaseModel):
    """Specification of an scheduled macroeconomic event with blackout windows (Zero Defaults)."""

    event_name: str
    event_time_utc: datetime
    blackout_before_minutes: float
    blackout_after_minutes: float


def _ensure_utc(dt: datetime) -> datetime:
    """Normalize datetime to UTC timezone-aware for deterministic delta calculation."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class MacroNewsSentinel(MacroSentinelPort):
    """Evaluates upcoming macroeconomic events and determines trading blackout state."""

    def __init__(self, schedules: list[MacroEventSchedule]) -> None:
        self.schedules: list[MacroEventSchedule] = schedules

    async def check_blackout(
        self,
        symbol: str,
        current_time_utc: datetime,
    ) -> MacroBlackoutState:
        """Evaluate if current_time_utc falls within [event_time - before, event_time + after].

        If active blackout is detected, returns is_blackout=True.
        Otherwise finds nearest upcoming event or reports clear state.
        """
        now = _ensure_utc(current_time_utc)

        # 1. Check if any event blackout window is currently active
        for schedule in self.schedules:
            event_time = _ensure_utc(schedule.event_time_utc)
            start_window = event_time - timedelta(minutes=schedule.blackout_before_minutes)
            end_window = event_time + timedelta(minutes=schedule.blackout_after_minutes)

            if start_window <= now <= end_window:
                time_to_event = (event_time - now).total_seconds() / 60.0
                return MacroBlackoutState(
                    is_blackout=True,
                    upcoming_event=schedule.event_name,
                    time_to_event_minutes=time_to_event,
                    reason=f"MACRO_BLACKOUT_ACTIVE_{schedule.event_name}",
                )

        # 2. Find nearest upcoming event in the future (event_time >= now)
        upcoming_schedules = [s for s in self.schedules if _ensure_utc(s.event_time_utc) >= now]

        if len(upcoming_schedules) > 0:
            nearest = min(
                upcoming_schedules,
                key=lambda s: (_ensure_utc(s.event_time_utc) - now).total_seconds(),
            )
            time_to_nearest = (_ensure_utc(nearest.event_time_utc) - now).total_seconds() / 60.0
            return MacroBlackoutState(
                is_blackout=False,
                upcoming_event=nearest.event_name,
                time_to_event_minutes=time_to_nearest,
                reason="CLEAR_UPCOMING_EVENT_SCHEDULED",
            )

        return MacroBlackoutState(
            is_blackout=False,
            upcoming_event=None,
            time_to_event_minutes=None,
            reason="CLEAR_NO_SCHEDULED_EVENTS",
        )
