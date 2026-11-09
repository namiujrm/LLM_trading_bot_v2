"""Unit tests for MacroNewsSentinel adapter.

Verifies economic event schedule blackout detection, time delta computation,
and Invariant #9 compliance.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trading_bot.adapters.agents.macro_news_sentinel import (
    MacroEventSchedule,
    MacroNewsSentinel,
)
from trading_bot.ports.macro_sentinel_port import MacroBlackoutState


@pytest.fixture
def fmc_event_time() -> datetime:
    return datetime(2026, 11, 10, 18, 0, 0, tzinfo=UTC)


@pytest.fixture
def sample_schedules(fmc_event_time: datetime) -> list[MacroEventSchedule]:
    return [
        MacroEventSchedule(
            event_name="FOMC_RATE_DECISION",
            event_time_utc=fmc_event_time,
            blackout_before_minutes=15.0,  # 17:45 to 18:30 blackout
            blackout_after_minutes=30.0,
        ),
        MacroEventSchedule(
            event_name="CPI_RELEASE",
            event_time_utc=fmc_event_time + timedelta(days=2),
            blackout_before_minutes=15.0,
            blackout_after_minutes=15.0,
        ),
    ]


@pytest.mark.asyncio
async def test_macro_sentinel_active_blackout_before_event(
    sample_schedules: list[MacroEventSchedule],
    fmc_event_time: datetime,
) -> None:
    sentinel = MacroNewsSentinel(sample_schedules)
    # 10 minutes before FOMC (within 15m blackout window)
    now = fmc_event_time - timedelta(minutes=10)

    state = await sentinel.check_blackout(symbol="BTCUSDT", current_time_utc=now)

    assert isinstance(state, MacroBlackoutState)
    assert state.is_blackout is True
    assert state.upcoming_event == "FOMC_RATE_DECISION"
    assert state.time_to_event_minutes == pytest.approx(10.0, rel=1e-4)
    assert state.reason == "MACRO_BLACKOUT_ACTIVE_FOMC_RATE_DECISION"


@pytest.mark.asyncio
async def test_macro_sentinel_active_blackout_after_event(
    sample_schedules: list[MacroEventSchedule],
    fmc_event_time: datetime,
) -> None:
    sentinel = MacroNewsSentinel(sample_schedules)
    # 20 minutes after FOMC (within 30m post-event blackout window)
    now = fmc_event_time + timedelta(minutes=20)

    state = await sentinel.check_blackout(symbol="BTCUSDT", current_time_utc=now)

    assert state.is_blackout is True
    assert state.upcoming_event == "FOMC_RATE_DECISION"
    assert state.time_to_event_minutes == pytest.approx(-20.0, rel=1e-4)
    assert state.reason == "MACRO_BLACKOUT_ACTIVE_FOMC_RATE_DECISION"


@pytest.mark.asyncio
async def test_macro_sentinel_clear_with_upcoming_event(
    sample_schedules: list[MacroEventSchedule],
    fmc_event_time: datetime,
) -> None:
    sentinel = MacroNewsSentinel(sample_schedules)
    # 45 minutes before FOMC (outside 15m blackout window)
    now = fmc_event_time - timedelta(minutes=45)

    state = await sentinel.check_blackout(symbol="BTCUSDT", current_time_utc=now)

    assert state.is_blackout is False
    assert state.upcoming_event == "FOMC_RATE_DECISION"
    assert state.time_to_event_minutes == pytest.approx(45.0, rel=1e-4)
    assert state.reason == "CLEAR_UPCOMING_EVENT_SCHEDULED"


@pytest.mark.asyncio
async def test_macro_sentinel_all_events_in_past(
    sample_schedules: list[MacroEventSchedule],
    fmc_event_time: datetime,
) -> None:
    sentinel = MacroNewsSentinel(sample_schedules)
    # 3 days after FOMC (all events have passed)
    now = fmc_event_time + timedelta(days=3)

    state = await sentinel.check_blackout(symbol="BTCUSDT", current_time_utc=now)

    assert state.is_blackout is False
    assert state.upcoming_event is None
    assert state.time_to_event_minutes is None
    assert state.reason == "CLEAR_NO_SCHEDULED_EVENTS"


@pytest.mark.asyncio
async def test_macro_sentinel_empty_schedules() -> None:
    sentinel = MacroNewsSentinel([])
    now = datetime(2026, 11, 10, 12, 0, 0, tzinfo=UTC)

    state = await sentinel.check_blackout(symbol="BTCUSDT", current_time_utc=now)

    assert state.is_blackout is False
    assert state.upcoming_event is None
    assert state.time_to_event_minutes is None
    assert state.reason == "CLEAR_NO_SCHEDULED_EVENTS"


@pytest.mark.asyncio
async def test_macro_sentinel_naive_datetime_handling(
    sample_schedules: list[MacroEventSchedule],
    fmc_event_time: datetime,
) -> None:
    sentinel = MacroNewsSentinel(sample_schedules)
    # Naive datetime (implicitly interpreted as UTC) 5 minutes before FOMC
    naive_now = datetime(2026, 11, 10, 17, 55, 0)

    state = await sentinel.check_blackout(symbol="BTCUSDT", current_time_utc=naive_now)

    assert state.is_blackout is True
    assert state.upcoming_event == "FOMC_RATE_DECISION"
    assert state.time_to_event_minutes == pytest.approx(5.0, rel=1e-4)
