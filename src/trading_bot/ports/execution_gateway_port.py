"""Execution Gateway Port Interface (Protocol)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from trading_bot.domain.entities.account_state import (
    AccountPortfolioContext,
    ActivePositionSnapshot,
    PendingOrderSnapshot,
)
from trading_bot.domain.entities.trade_blueprint import BracketOrderTicket


@runtime_checkable
class ExecutionGatewayPort(Protocol):
    """Abstract interface for order execution on exchange/paper simulator (Zero Defaults)."""

    async def place_bracket_order(self, ticket: BracketOrderTicket) -> str:
        """Submit a bracket order (entry, SL, TP) and return the parent order ID."""
        ...

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an active order by symbol and order ID."""
        ...

    async def get_position_notional(self, symbol: str) -> float:
        """Fetch current open position notional value in USD for a symbol."""
        ...

    async def fetch_account_portfolio_context(self) -> AccountPortfolioContext:
        """Fetch the unified live portfolio and margin state context."""
        ...

    async def fetch_active_positions(self) -> list[ActivePositionSnapshot]:
        """Fetch all currently open positions with margin and breakeven metrics."""
        ...

    async def fetch_pending_orders(self) -> list[PendingOrderSnapshot]:
        """Fetch all currently open pending entry orders reserving margin."""
        ...
