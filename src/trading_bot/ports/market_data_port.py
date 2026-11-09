"""Market Data Port Interface (Protocol)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from trading_bot.domain.entities.market.market_state import MicroBar1m, MultiTFBar
from trading_bot.domain.entities.risk.account_state import AccountPortfolioContext


@runtime_checkable
class MarketDataPort(Protocol):
    """Abstract interface for fetching and persisting market klines and ticks (Zero Defaults)."""

    async def fetch_recent_klines_1m(self, symbol: str, limit: int) -> list[MicroBar1m]:
        """Fetch the most recent N 1-minute bars for a symbol."""
        ...

    async def save_klines_1m(self, symbol: str, bars: list[MicroBar1m]) -> None:
        """Persist a batch of enriched 1-minute bars."""
        ...

    async def get_latest_price(self, symbol: str) -> float:
        """Get the current real-time mark or mid price for a symbol."""
        ...

    async def fetch_multi_tf_bars(
        self, symbol: str, timeframe: str, limit: int
    ) -> list[MultiTFBar]:
        """Fetch multi-timeframe aggregated bars (15m, 30m, 1h, 4h)."""
        ...

    async def save_account_equity_snapshot(self, context: AccountPortfolioContext) -> None:
        """Persist a point-in-time account equity and margin snapshot."""
        ...
