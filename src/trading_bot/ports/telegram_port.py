"""Telegram Domain Ports (Clean Architecture, Invariant #9).

Enforces strictly ZERO default parameters, ZERO class field defaults, and ZERO silent fallbacks.
Depends ONLY on pure domain entities and standard library typing.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from trading_bot.domain.entities.quant.symbolic_alpha import ModelTournamentReport
from trading_bot.domain.entities.telegram.telegram_models import (
    TelegramCommandRequest,
    TelegramCycleReport,
    TelegramInlineButton,
)


@runtime_checkable
class TelegramNotifierPort(Protocol):
    """Protocol port interface for dispatching outbound alerts and reports to Telegram."""

    async def send_text_message(
        self,
        text: str,
        reply_markup: list[list[TelegramInlineButton]] | dict[str, Any] | None,
    ) -> bool:
        """Send a formatted text message with optional inline keyboard buttons."""
        ...

    async def send_cycle_report(
        self,
        report: TelegramCycleReport | dict[str, Any],
    ) -> bool:
        """Send a 30-minute trading cycle summary report card."""
        ...

    async def send_emergency_alert(
        self,
        title: str,
        message: str,
    ) -> bool:
        """Send a high-priority emergency alert."""
        ...

    async def send_order_placed_alert(
        self,
        symbol: str,
        side: str,
        order_type: str,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        quantity: float,
        risk_reward_ratio: float,
        order_id: str,
    ) -> bool:
        """Send automated real-time notification when a bracket order is placed on exchange."""
        ...

    async def send_position_closed_alert(
        self,
        symbol: str,
        side: str,
        close_price: float,
        pnl_usd: float,
        pnl_pct: float,
        r_multiple: float,
        exit_reason: str,
    ) -> bool:
        """Send notification when a position is closed (TP, SL, or manual close)."""
        ...

    async def send_error_alert(
        self,
        component: str,
        error_message: str,
        is_fatal: bool,
    ) -> bool:
        """Send real-time alert when an error or exception occurs in the system."""
        ...

    async def send_tournament_verdict(
        self,
        report: ModelTournamentReport,
    ) -> bool:
        """Send the weekend Champion vs. Challenger tournament comparison report."""
        ...


@runtime_checkable
class TelegramCommandHandlerPort(Protocol):
    """Protocol port interface for processing inbound user commands and callbacks."""

    async def handle_command(
        self,
        request: TelegramCommandRequest,
    ) -> str:
        """Process a validated user slash command and return the response text."""
        ...

    async def handle_callback(
        self,
        callback_id: str,
        callback_data: str,
    ) -> str:
        """Process an inline keyboard button callback query and return notification text."""
        ...
