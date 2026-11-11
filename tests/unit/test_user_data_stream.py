"""Unit tests for BinanceUserDataStream (Invariant #9 Compliant)."""

from __future__ import annotations

from typing import Any

import pytest

from trading_bot.adapters.binance.user_data_stream import BinanceUserDataStream
from trading_bot.domain.entities.execution.trade_blueprint import BracketOrderTicket
from trading_bot.domain.entities.market.market_state import EnrichedMicroBarRecord
from trading_bot.domain.entities.market.symbol_filters import SymbolFilterRule
from trading_bot.domain.entities.risk.account_state import (
    AccountPortfolioContext,
    ActivePositionSnapshot,
    PendingOrderSnapshot,
)
from trading_bot.infrastructure.config_schemas import BinanceConfig
from trading_bot.ports.execution_gateway_port import ExecutionGatewayPort


class MockGateway(ExecutionGatewayPort):
    def __init__(self) -> None:
        self.created_listen_key: str = "mock_key_abc123"
        self.keepalive_calls: int = 0
        self.closed_listen_keys: list[str] = []

    async def place_bracket_order(self, ticket: BracketOrderTicket) -> str:
        return ticket.order_id

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        return True

    async def cancel_algo_order(self, symbol: str, algo_id: str) -> bool:
        return True

    async def fetch_open_algo_orders(self, symbol: str) -> list[dict[str, Any]]:
        return []

    async def get_position_notional(self, symbol: str) -> float:
        return 0.0

    async def fetch_account_portfolio_context(self) -> AccountPortfolioContext:
        raise NotImplementedError

    async def fetch_active_positions(self) -> list[ActivePositionSnapshot]:
        return []

    async def fetch_pending_orders(self) -> list[PendingOrderSnapshot]:
        return []

    async def fetch_symbol_filters(self) -> dict[str, SymbolFilterRule]:
        return {}

    async def fetch_recent_klines_1m(self, symbol: str, limit: int) -> list[EnrichedMicroBarRecord]:
        return []

    async def create_user_data_stream(self) -> str:
        return self.created_listen_key

    async def keepalive_user_data_stream(self, listen_key: str) -> None:
        self.keepalive_calls += 1

    async def close_user_data_stream(self, listen_key: str) -> None:
        self.closed_listen_keys.append(listen_key)


@pytest.mark.asyncio
async def test_order_trade_update_sl_filled_triggers_tp_cancel() -> None:
    gateway = MockGateway()
    cfg = BinanceConfig(
        api_key="mock_key",
        api_secret="mock_secret",
        testnet=True,
        recv_window_ms=5000,
    )
    callbacks_received: list[tuple[str, str, str, str]] = []

    async def on_leg_filled(sym: str, leg: str, parent: str, opposing: str) -> None:
        callbacks_received.append((sym, leg, parent, opposing))

    stream = BinanceUserDataStream(
        binance_config=cfg,
        gateway=gateway,
        session=None,
        on_bracket_leg_filled=on_leg_filled,
    )

    # Simulate WebSocket message for SL order fill
    msg_sl = {
        "e": "ORDER_TRADE_UPDATE",
        "o": {
            "s": "BTCUSDT",
            "c": "order_btc_999_sl",
            "X": "FILLED",
            "ot": "STOP_MARKET",
            "S": "SELL",
        },
    }
    await stream.handle_ws_message(msg_sl)

    assert len(callbacks_received) == 1
    assert callbacks_received[0] == (
        "BTCUSDT",
        "SL",
        "order_btc_999",
        "order_btc_999_tp",
    )


@pytest.mark.asyncio
async def test_order_trade_update_tp_filled_triggers_sl_cancel() -> None:
    gateway = MockGateway()
    cfg = BinanceConfig(
        api_key="mock_key",
        api_secret="mock_secret",
        testnet=True,
        recv_window_ms=5000,
    )
    callbacks_received: list[tuple[str, str, str, str]] = []

    async def on_leg_filled(sym: str, leg: str, parent: str, opposing: str) -> None:
        callbacks_received.append((sym, leg, parent, opposing))

    stream = BinanceUserDataStream(
        binance_config=cfg,
        gateway=gateway,
        session=None,
        on_bracket_leg_filled=on_leg_filled,
    )

    # Simulate WebSocket message for TP order fill
    msg_tp = {
        "e": "ORDER_TRADE_UPDATE",
        "o": {
            "s": "ETHUSDT",
            "c": "order_eth_888_tp",
            "X": "FILLED",
            "ot": "TAKE_PROFIT_MARKET",
            "S": "BUY",
        },
    }
    await stream.handle_ws_message(msg_tp)

    assert len(callbacks_received) == 1
    assert callbacks_received[0] == (
        "ETHUSDT",
        "TP",
        "order_eth_888",
        "order_eth_888_sl",
    )


@pytest.mark.asyncio
async def test_order_trade_update_new_status_ignored() -> None:
    gateway = MockGateway()
    cfg = BinanceConfig(
        api_key="mock_key",
        api_secret="mock_secret",
        testnet=True,
        recv_window_ms=5000,
    )
    callbacks_received: list[tuple[str, str, str, str]] = []

    async def on_leg_filled(sym: str, leg: str, parent: str, opposing: str) -> None:
        callbacks_received.append((sym, leg, parent, opposing))

    stream = BinanceUserDataStream(
        binance_config=cfg,
        gateway=gateway,
        session=None,
        on_bracket_leg_filled=on_leg_filled,
    )

    msg_new = {
        "e": "ORDER_TRADE_UPDATE",
        "o": {
            "s": "BTCUSDT",
            "c": "order_btc_999_sl",
            "X": "NEW",
            "ot": "STOP_MARKET",
        },
    }
    await stream.handle_ws_message(msg_new)
    assert len(callbacks_received) == 0
