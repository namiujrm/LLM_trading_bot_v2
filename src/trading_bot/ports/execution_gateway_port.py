"""Execution Gateway Port Interface (Protocol)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from trading_bot.domain.entities.execution.trade_blueprint import BracketOrderTicket
from trading_bot.domain.entities.market.market_state import EnrichedMicroBarRecord
from trading_bot.domain.entities.market.symbol_filters import SymbolFilterRule
from trading_bot.domain.entities.risk.account_state import (
    AccountPortfolioContext,
    ActivePositionSnapshot,
    PendingOrderSnapshot,
)


@runtime_checkable
class ExecutionGatewayPort(Protocol):
    """Abstract interface for order execution on exchange/paper simulator (Zero Defaults)."""

    async def place_bracket_order(self, ticket: BracketOrderTicket) -> str:
        """Submit a bracket order (entry, SL, TP) and return the parent order ID."""
        ...

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an active order by symbol and order ID."""
        ...

    async def cancel_algo_order(self, symbol: str, algo_id: str) -> bool:
        """Cancel an active algo conditional order (SL/TP) by symbol and algo ID."""
        ...

    async def fetch_open_algo_orders(self, symbol: str) -> list[dict[str, Any]]:
        """Fetch all currently open algo conditional orders for a symbol."""
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

    async def fetch_symbol_filters(self) -> dict[str, SymbolFilterRule]:
        """Fetch normalized exchange trading rules and precision constraints for symbols."""
        ...

    async def fetch_recent_klines_1m(self, symbol: str, limit: int) -> list[EnrichedMicroBarRecord]:
        """Fetch recent 1m klines from exchange for buffer warm-up."""
        ...

    async def create_user_data_stream(self) -> str:
        """Start a user data stream and return the listenKey."""
        ...

    async def keepalive_user_data_stream(self, listen_key: str) -> None:
        """Keepalive a user data stream listenKey."""
        ...

    async def close_user_data_stream(self, listen_key: str) -> None:
        """Close an active user data stream listenKey."""
        ...


@runtime_checkable
class HistoricalMultiTFWarmupPort(Protocol):
    """Extended exchange gateway port for fetching historical 30m klines during warmup."""

    async def fetch_recent_klines_30m(
        self, symbol: str, limit: int
    ) -> list[EnrichedMicroBarRecord]:
        """Fetch recent completed 30m klines from exchange for deep indicator warm-up."""
        ...
