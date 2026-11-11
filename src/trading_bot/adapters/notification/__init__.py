"""Notification Adapters (Clean Architecture, Invariant #9)."""

from __future__ import annotations

from trading_bot.adapters.notification.console_tournament_notifier import (
    ConsoleAndFileTournamentNotifierAdapter,
)

__all__: list[str] = [
    "ConsoleAndFileTournamentNotifierAdapter",
]
