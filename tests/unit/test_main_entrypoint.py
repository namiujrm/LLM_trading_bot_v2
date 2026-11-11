"""Comprehensive Unit Tests for Main Entrypoint & TradingBotSystemDaemon (Milestone 8.3).

Enforces strictly ZERO default parameters, ZERO class field defaults, and ZERO silent fallbacks.
Tests:
- Daemon initialization with mock and environment configurations.
- Lifecycle: initialize(), run(), shutdown(), and re-entrancy guards.
- Signal handler registration and cross-platform safety.
- Telegram command hooks wiring (/status, /positions, /mode, /panic_close_all).
- Action hooks: Breakeven trigger and emergency position closure.
- 30-minute cycle summary interceptor and Telegram cycle report card notification.
- Main entrypoint function execution and error handling.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading_bot.adapters.telegram.telegram_bot_adapter import TelegramBotAdapter
from trading_bot.application.use_cases.evaluate_tiered_funnel import TieredFunnelInput
from trading_bot.application.use_cases.run_30m_trading_cycle import (
    SymbolCycleResult,
    TradingCycleSummary,
)
from trading_bot.domain.entities.common.enums import (
    EntryOrderType,
    ExecutionMode,
    TradeSide,
)
from trading_bot.domain.entities.market.market_state import SemanticQuantPacket
from trading_bot.domain.entities.risk.account_state import (
    AccountPortfolioContext,
    ActivePositionSnapshot,
    PendingOrderSnapshot,
)
from trading_bot.domain.entities.telegram.telegram_models import TelegramCommandRequest
from trading_bot.infrastructure.config_schemas import TelegramConfig
from trading_bot.interfaces.main import TradingBotSystemDaemon, main
from trading_bot.ports.execution_gateway_port import ExecutionGatewayPort


class MockResponse:
    """Mock aiohttp ClientResponse supporting async context manager (Zero Defaults)."""

    def __init__(self, status: int, data: dict[str, Any], raise_exc: Exception | None) -> None:
        self.status: int = status
        self._data: dict[str, Any] = data
        self._raise_exc: Exception | None = raise_exc

    async def json(self) -> dict[str, Any]:
        return self._data

    async def __aenter__(self) -> MockResponse:
        if self._raise_exc is not None:
            raise self._raise_exc
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        pass


class MockSession:
    """Mock aiohttp ClientSession recording requests (Zero Defaults)."""

    def __init__(self, responses: list[MockResponse]) -> None:
        self.responses: list[MockResponse] = responses
        self.recorded_requests: list[dict[str, Any]] = []
        self.closed: bool = False

    def post(
        self,
        url: str,
        json: dict[str, Any] | None,
        timeout: Any,
    ) -> MockResponse:
        self.recorded_requests.append({"url": url, "json": json, "timeout": timeout})
        if len(self.responses) > 0:
            return self.responses.pop(0)
        return MockResponse(200, {"ok": True, "result": []}, None)

    async def close(self) -> None:
        self.closed = True


class MockDatabaseManager:
    """Mock database manager tracking pool lifecycle and health (Zero Defaults)."""

    def __init__(self, is_healthy: bool) -> None:
        self._is_healthy: bool = is_healthy
        self.is_initialized: bool = False
        self.initialize_called: bool = False
        self.close_called: bool = False
        self._pool: Any = object()

    async def initialize(self) -> None:
        self.initialize_called = True
        self.is_initialized = True

    async def close(self) -> None:
        self.close_called = True

    async def cancel_algo_order(self, symbol: str, algo_id: str) -> bool:
        return True

    async def fetch_open_algo_orders(self, symbol: str) -> list[dict[str, Any]]:
        return []

    async def fetch_symbol_filters(self) -> dict[str, Any]:
        return {}

    async def fetch_recent_klines_1m(self, symbol: str, limit: int) -> list[Any]:
        return []

    async def create_user_data_stream(self) -> str:
        return "mock_listen_key"

    async def keepalive_user_data_stream(self, listen_key: str) -> None:
        pass

    async def close_user_data_stream(self, listen_key: str) -> None:
        pass
        self.is_initialized = False
        self._pool = None

    async def healthcheck(self) -> bool:
        return self._is_healthy


class MockExecutionGateway(ExecutionGatewayPort):
    """Mock execution gateway for USD-M Futures (Zero Defaults)."""

    def __init__(
        self,
        context: AccountPortfolioContext,
        positions: list[ActivePositionSnapshot],
        pending_orders: list[PendingOrderSnapshot],
    ) -> None:
        self.context: AccountPortfolioContext = context
        self.positions: list[ActivePositionSnapshot] = positions
        self.pending_orders: list[PendingOrderSnapshot] = pending_orders
        self.cancelled_orders: list[tuple[str, str]] = []
        self.closed_positions: list[tuple[str, float]] = []
        self.breakeven_updates: list[tuple[str, float]] = []
        self.sync_time_called: bool = False
        self.close_called: bool = False

    async def sync_time_offset(self) -> int:
        self.sync_time_called = True
        return 42

    async def fetch_account_portfolio_context(self) -> AccountPortfolioContext:
        return self.context

    async def fetch_active_positions(self) -> list[ActivePositionSnapshot]:
        return list(self.positions)

    async def fetch_pending_orders(self) -> list[PendingOrderSnapshot]:
        return list(self.pending_orders)

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        self.cancelled_orders.append((symbol, order_id))
        return True

    async def close_position_market(self, symbol: str, quantity: float) -> bool:
        self.closed_positions.append((symbol, quantity))
        return True

    async def update_stop_loss_to_breakeven(self, symbol: str, be_price: float) -> bool:
        self.breakeven_updates.append((symbol, be_price))
        return True

    async def place_bracket_order(self, ticket: Any) -> str:
        return "order_123"

    async def get_position_notional(self, symbol: str) -> float:
        return 1000.0

    async def close(self) -> None:
        self.close_called = True

    async def cancel_algo_order(self, symbol: str, algo_id: str) -> bool:
        return True

    async def fetch_open_algo_orders(self, symbol: str) -> list[dict[str, Any]]:
        return []

    async def fetch_symbol_filters(self) -> dict[str, Any]:
        return {}

    async def fetch_recent_klines_1m(self, symbol: str, limit: int) -> list[Any]:
        return []

    async def create_user_data_stream(self) -> str:
        return "mock_listen_key"

    async def keepalive_user_data_stream(self, listen_key: str) -> None:
        pass

    async def close_user_data_stream(self, listen_key: str) -> None:
        pass


class MockLivePaperRunner:
    """Mock LivePaperRunner simulating trading cycle orchestration (Zero Defaults)."""

    def __init__(self, symbols: list[str]) -> None:
        self.symbols: list[str] = symbols
        self.is_running: bool = False
        self.start_called: bool = False
        self.stop_called: bool = False
        self.cycle_history: list[TradingCycleSummary] = []
        self._stop_event: asyncio.Event = asyncio.Event()

    async def start(self) -> None:
        self.start_called = True
        self.is_running = True
        await self._stop_event.wait()

    async def stop(self) -> None:
        self.stop_called = True
        self.is_running = False
        self._stop_event.set()

    async def handle_cycle_ready(
        self,
        cycle_time_utc: datetime,
        market_inputs: dict[str, TieredFunnelInput],
        quant_packets: dict[str, SemanticQuantPacket],
        current_prices: dict[str, float],
        atr_30m_values: dict[str, float],
    ) -> None:
        pass

    def set_on_concluded_callback(self, callback: Any) -> None:
        pass


class MockWSIngestor:
    """Mock WebSocket ingestor simulating streaming status (Zero Defaults)."""

    def __init__(self, is_running: bool) -> None:
        self._is_running: bool = is_running
        self.stop_called: bool = False

    async def stop(self) -> None:
        self.stop_called = True
        self._is_running = False


def create_sample_position(
    pos_id: str,
    symbol: str,
    side: TradeSide,
    entry_price: float,
    qty: float,
) -> ActivePositionSnapshot:
    """Create a sample ActivePositionSnapshot without defaults (Zero Defaults)."""
    return ActivePositionSnapshot(
        position_id=pos_id,
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        mark_price=entry_price * 1.01 if side == TradeSide.LONG else entry_price * 0.99,
        stop_loss_price=entry_price * 0.98 if side == TradeSide.LONG else entry_price * 1.02,
        take_profit_price=entry_price * 1.05 if side == TradeSide.LONG else entry_price * 0.95,
        liquidation_price=entry_price * 0.85 if side == TradeSide.LONG else entry_price * 1.15,
        quantity=qty,
        notional_usd=entry_price * qty,
        isolated_margin_usd=(entry_price * qty) / 5.0,
        leverage=5,
        unrealized_pnl_usd=50.0,
        unrealized_pnl_pct=0.05,
        unrealized_r_multiple=1.5,
        is_sl_at_breakeven=False,
        effective_at_risk_usd=100.0,
        btc_beta=1.0,
    )


def create_sample_order(order_id: str, symbol: str) -> PendingOrderSnapshot:
    """Create a sample PendingOrderSnapshot without defaults (Zero Defaults)."""
    return PendingOrderSnapshot(
        order_id=order_id,
        symbol=symbol,
        side=TradeSide.LONG,
        order_type=EntryOrderType.LIMIT_PULLBACK_15M,
        limit_price=65000.0,
        notional_usd=3250.0,
        reserved_margin_usd=650.0,
        created_at_ms=1700000000000,
    )


def create_sample_account_context(
    positions: list[ActivePositionSnapshot],
    orders: list[PendingOrderSnapshot],
) -> AccountPortfolioContext:
    """Create a sample AccountPortfolioContext without defaults (Zero Defaults)."""
    return AccountPortfolioContext(
        execution_mode=ExecutionMode.PAPER,
        timestamp_ms=1700000000000,
        wallet_balance_usd=10000.0,
        unrealized_pnl_usd=50.0,
        equity_nav_usd=10050.0,
        used_margin_usd=2000.0,
        available_margin_usd=8050.0,
        margin_utilization_pct=0.199,
        current_daily_pnl_usd=150.0,
        current_daily_dd_pct=0.015,
        remaining_daily_dd_budget_pct=0.035,
        current_max_dd_pct=0.025,
        remaining_max_dd_budget_pct=0.075,
        consecutive_losses_count=0,
        active_positions=positions,
        pending_orders=orders,
        gross_exposure_usd=5000.0,
        net_directional_delta_usd=5000.0,
        net_btc_beta_exposure_usd=5000.0,
        raw_open_heat_pct=0.02,
        effective_open_heat_pct=0.02,
        same_direction_active_count=len(positions),
    )


def create_sample_cycle_summary() -> TradingCycleSummary:
    """Create a sample TradingCycleSummary without defaults (Zero Defaults)."""
    btc_res = SymbolCycleResult(
        symbol="BTCUSDT",
        funnel_passed=True,
        funnel_stage="PASSED",
        thesis=None,
        blueprint=None,
        verdict=None,
        order_id="order_btc_01",
        is_shadow=False,
        rejection_reason=None,
    )
    alt_res = {
        "ETHUSDT": SymbolCycleResult(
            symbol="ETHUSDT",
            funnel_passed=False,
            funnel_stage="STAGE_2_SPREAD",
            thesis=None,
            blueprint=None,
            verdict=None,
            order_id=None,
            is_shadow=False,
            rejection_reason="Excessive spread",
        )
    }
    return TradingCycleSummary(
        cycle_timestamp_ms=1700000000000,
        btc_result=btc_res,
        altcoin_results=alt_res,
        executed_orders_count=1,
        shadow_orders_count=0,
    )


def create_test_env() -> dict[str, str]:
    """Create test environment mapping without defaults (Zero Defaults)."""
    return {
        "EXECUTION_MODE": "PAPER",
        "SYMBOLS": "BTCUSDT,ETHUSDT",
        "BINANCE_TESTNET": "true",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "test_db",
        "POSTGRES_USER": "test_user",
        "POSTGRES_PASSWORD": "test_password",
        "POSTGRES_MAX_CONNECTIONS": "5",
        "BINANCE_API_KEY": "test_key",
        "BINANCE_API_SECRET": "test_secret",
        "BINANCE_RECV_WINDOW_MS": "5000",
        "TELEGRAM_BOT_TOKEN": "12345:TEST_TOKEN",
        "TELEGRAM_CHAT_ID": "5577408890",
        "TELEGRAM_POLL_TIMEOUT": "10",
        "LOG_DIR": "logs",
        "DATASET_36M_PATH": "__absent_in_unit_tests__",
    }


# =============================================================================
# Test Cases
# =============================================================================


@pytest.mark.asyncio
async def test_daemon_initialization_properties() -> None:
    """Test initial daemon properties and configuration extraction."""
    env = create_test_env()
    session = MockSession([])
    pos = create_sample_position("pos_btc", "BTCUSDT", TradeSide.LONG, 65000.0, 0.1)
    order = create_sample_order("ord_01", "BTCUSDT")
    ctx = create_sample_account_context([pos], [order])
    gateway = MockExecutionGateway(ctx, [pos], [order])
    db_mgr = MockDatabaseManager(True)
    runner = MockLivePaperRunner(["BTCUSDT", "ETHUSDT"])
    tg_cfg = TelegramConfig(bot_token="test_tok", chat_id="5577408890")
    adapter = TelegramBotAdapter(tg_cfg, session, 10)  # type: ignore[arg-type]

    daemon = TradingBotSystemDaemon(
        env_mapping=env,
        session=session,  # type: ignore[arg-type]
        db_manager=db_mgr,  # type: ignore[arg-type]
        gateway=gateway,
        ws_ingestor=None,
        telegram_adapter=adapter,
        runner=runner,  # type: ignore[arg-type]
    )

    assert daemon.is_initialized is False
    assert daemon.is_running is False
    assert daemon.execution_mode == "PAPER"
    assert daemon.symbols == ["BTCUSDT", "ETHUSDT"]
    assert daemon.testnet is True
    assert daemon.telegram_adapter is adapter
    assert daemon.gateway is gateway
    assert daemon.db_manager is db_mgr
    assert daemon.runner is runner


@pytest.mark.asyncio
async def test_daemon_initialize_wires_subsystems() -> None:
    """Test that initialize() wires gateway, DB, Telegram hooks, and sends startup notification."""
    env = create_test_env()
    session = MockSession([])
    pos = create_sample_position("pos_btc", "BTCUSDT", TradeSide.LONG, 65000.0, 0.1)
    order = create_sample_order("ord_01", "BTCUSDT")
    ctx = create_sample_account_context([pos], [order])
    gateway = MockExecutionGateway(ctx, [pos], [order])
    db_mgr = MockDatabaseManager(True)
    runner = MockLivePaperRunner(["BTCUSDT", "ETHUSDT"])
    tg_cfg = TelegramConfig(bot_token="test_tok", chat_id="5577408890")
    adapter = TelegramBotAdapter(tg_cfg, session, 10)  # type: ignore[arg-type]

    daemon = TradingBotSystemDaemon(
        env_mapping=env,
        session=session,  # type: ignore[arg-type]
        db_manager=db_mgr,  # type: ignore[arg-type]
        gateway=gateway,
        ws_ingestor=None,
        telegram_adapter=adapter,
        runner=runner,  # type: ignore[arg-type]
    )

    await daemon.initialize()

    assert daemon.is_initialized is True
    assert db_mgr.initialize_called is True
    assert gateway.sync_time_called is True
    assert adapter.execution_mode == "PAPER"
    assert adapter.account_context is not None
    assert adapter.account_context.equity_nav_usd == 10050.0

    # Verify startup notification sent to Telegram
    assert len(session.recorded_requests) >= 1
    first_req = session.recorded_requests[0]
    assert "sendMessage" in first_req["url"]
    assert "Hệ thống đã khởi động thành công" in first_req["json"]["text"]

    # Re-initialization guard
    await daemon.initialize()
    assert daemon.is_initialized is True


@pytest.mark.asyncio
async def test_handle_status_command() -> None:
    """Test /status command handler formats NAV, daily PnL, Drawdown, mode, DB and WS health."""
    env = create_test_env()
    session = MockSession([])
    pos = create_sample_position("pos_btc", "BTCUSDT", TradeSide.LONG, 65000.0, 0.1)
    order = create_sample_order("ord_01", "BTCUSDT")
    ctx = create_sample_account_context([pos], [order])
    gateway = MockExecutionGateway(ctx, [pos], [order])
    db_mgr = MockDatabaseManager(True)
    runner = MockLivePaperRunner(["BTCUSDT", "ETHUSDT"])
    tg_cfg = TelegramConfig(bot_token="test_tok", chat_id="5577408890")
    adapter = TelegramBotAdapter(tg_cfg, session, 10)  # type: ignore[arg-type]

    daemon = TradingBotSystemDaemon(
        env_mapping=env,
        session=session,  # type: ignore[arg-type]
        db_manager=db_mgr,  # type: ignore[arg-type]
        gateway=gateway,
        ws_ingestor=None,
        telegram_adapter=adapter,
        runner=runner,  # type: ignore[arg-type]
    )
    await daemon.initialize()

    cmd_req = TelegramCommandRequest(
        command="status",
        args=[],
        user_id="5577408890",
        chat_id="5577408890",
        raw_text="/status",
    )
    status_text = await daemon.handle_status_command(cmd_req)

    assert "HỆ THỐNG GIAO DỊCH - TRẠNG THÁI" in status_text
    assert "• *Execution Mode:* `PAPER`" in status_text
    assert "• *DB Health:* `🟢 ONLINE`" in status_text
    assert "• *Portfolio NAV:* `$10,050.00`" in status_text
    assert "• *Daily PnL:* 📈 `+$150.00`" in status_text
    assert "• *Drawdown:* `0.01%`" in status_text
    assert "• *Open Positions:* `1`" in status_text
    assert "• *Pending Orders:* `1`" in status_text


@pytest.mark.asyncio
async def test_handle_positions_command() -> None:
    """Test /positions command fetches live positions with BE/Close inline buttons."""
    env = create_test_env()
    session = MockSession([])
    pos = create_sample_position("pos_btc_01", "BTCUSDT", TradeSide.LONG, 65000.0, 0.1)
    order = create_sample_order("ord_01", "BTCUSDT")
    ctx = create_sample_account_context([pos], [order])
    gateway = MockExecutionGateway(ctx, [pos], [order])
    db_mgr = MockDatabaseManager(True)
    runner = MockLivePaperRunner(["BTCUSDT", "ETHUSDT"])
    tg_cfg = TelegramConfig(bot_token="test_tok", chat_id="5577408890")
    adapter = TelegramBotAdapter(tg_cfg, session, 10)  # type: ignore[arg-type]

    daemon = TradingBotSystemDaemon(
        env_mapping=env,
        session=session,  # type: ignore[arg-type]
        db_manager=db_mgr,  # type: ignore[arg-type]
        gateway=gateway,
        ws_ingestor=None,
        telegram_adapter=adapter,
        runner=runner,  # type: ignore[arg-type]
    )
    await daemon.initialize()

    positions_text = await daemon.handle_positions_command()

    assert "DANH SÁCH VỊ THẾ ĐANG MỞ" in positions_text
    assert "BTCUSDT" in positions_text
    assert "Entry: `$65,000.0000`" in positions_text

    # Verify inline keyboard was sent in Telegram sendMessage payload
    assert len(session.recorded_requests) >= 2
    last_req = session.recorded_requests[-1]
    assert "reply_markup" in last_req["json"]
    markup = last_req["json"]["reply_markup"]
    assert "inline_keyboard" in markup
    buttons = markup["inline_keyboard"][0]
    assert buttons[0]["text"] == "🛡️ BE BTCUSDT"
    assert buttons[0]["callback_data"] == "be:pos_btc_01"
    assert buttons[1]["text"] == "🚨 Đóng BTCUSDT"
    assert buttons[1]["callback_data"] == "close:pos_btc_01"


@pytest.mark.asyncio
async def test_on_mode_change_hook() -> None:
    """Test mode change action hook dynamically updates daemon and Telegram adapter."""
    env = create_test_env()
    session = MockSession([])
    pos = create_sample_position("pos_btc_01", "BTCUSDT", TradeSide.LONG, 65000.0, 0.1)
    ctx = create_sample_account_context([pos], [])
    gateway = MockExecutionGateway(ctx, [pos], [])
    db_mgr = MockDatabaseManager(True)
    runner = MockLivePaperRunner(["BTCUSDT"])
    tg_cfg = TelegramConfig(bot_token="test_tok", chat_id="5577408890")
    adapter = TelegramBotAdapter(tg_cfg, session, 10)  # type: ignore[arg-type]

    daemon = TradingBotSystemDaemon(
        env_mapping=env,
        session=session,  # type: ignore[arg-type]
        db_manager=db_mgr,  # type: ignore[arg-type]
        gateway=gateway,
        ws_ingestor=None,
        telegram_adapter=adapter,
        runner=runner,  # type: ignore[arg-type]
    )
    await daemon.initialize()

    # Valid mode change to LIVE
    res_live = await daemon.on_mode_change_hook("LIVE")
    assert "LIVE" in res_live
    assert daemon.execution_mode == "LIVE"
    assert adapter.execution_mode == "LIVE"

    # Valid mode change to TESTNET
    res_testnet = await daemon.on_mode_change_hook("testnet")
    assert "TESTNET" in res_testnet
    assert daemon.execution_mode == "TESTNET"
    assert adapter.execution_mode == "TESTNET"

    # Invalid mode rejection
    res_invalid = await daemon.on_mode_change_hook("UNKNOWN_MODE")
    assert "Chế độ không hợp lệ" in res_invalid
    assert daemon.execution_mode == "TESTNET"


@pytest.mark.asyncio
async def test_on_panic_close_hook() -> None:
    """Test panic close action hook cancels all pending orders, closes positions, and pauses."""
    env = create_test_env()
    session = MockSession([])
    pos1 = create_sample_position("pos_btc_01", "BTCUSDT", TradeSide.LONG, 65000.0, 0.1)
    pos2 = create_sample_position("pos_eth_01", "ETHUSDT", TradeSide.SHORT, 3500.0, 1.0)
    order1 = create_sample_order("ord_btc_01", "BTCUSDT")
    order2 = create_sample_order("ord_eth_01", "ETHUSDT")
    ctx = create_sample_account_context([pos1, pos2], [order1, order2])
    gateway = MockExecutionGateway(ctx, [pos1, pos2], [order1, order2])
    db_mgr = MockDatabaseManager(True)
    runner = MockLivePaperRunner(["BTCUSDT", "ETHUSDT"])
    tg_cfg = TelegramConfig(bot_token="test_tok", chat_id="5577408890")
    adapter = TelegramBotAdapter(tg_cfg, session, 10)  # type: ignore[arg-type]

    daemon = TradingBotSystemDaemon(
        env_mapping=env,
        session=session,  # type: ignore[arg-type]
        db_manager=db_mgr,  # type: ignore[arg-type]
        gateway=gateway,
        ws_ingestor=None,
        telegram_adapter=adapter,
        runner=runner,  # type: ignore[arg-type]
    )
    await daemon.initialize()

    res = await daemon.on_panic_close_hook()

    assert "PANIC CLOSE ALL" in res
    assert "Đã hủy 2 lệnh chờ" in res
    assert "đóng 2 vị thế thành công" in res
    assert adapter.is_paused is True
    assert len(gateway.cancelled_orders) == 2
    assert len(gateway.closed_positions) == 2


@pytest.mark.asyncio
async def test_on_breakeven_hook() -> None:
    """Test breakeven trigger action hook adjusts Stop Loss to entry price."""
    env = create_test_env()
    session = MockSession([])
    pos = create_sample_position("pos_btc_01", "BTCUSDT", TradeSide.LONG, 65000.0, 0.1)
    ctx = create_sample_account_context([pos], [])
    gateway = MockExecutionGateway(ctx, [pos], [])
    db_mgr = MockDatabaseManager(True)
    runner = MockLivePaperRunner(["BTCUSDT"])
    tg_cfg = TelegramConfig(bot_token="test_tok", chat_id="5577408890")
    adapter = TelegramBotAdapter(tg_cfg, session, 10)  # type: ignore[arg-type]

    daemon = TradingBotSystemDaemon(
        env_mapping=env,
        session=session,  # type: ignore[arg-type]
        db_manager=db_mgr,  # type: ignore[arg-type]
        gateway=gateway,
        ws_ingestor=None,
        telegram_adapter=adapter,
        runner=runner,  # type: ignore[arg-type]
    )
    await daemon.initialize()

    # Success case
    be_res = await daemon.on_breakeven_hook("pos_btc_01")
    assert "Đã dời SL về BE (65000.0)" in be_res
    assert len(gateway.breakeven_updates) == 1
    assert gateway.breakeven_updates[0] == ("BTCUSDT", 65000.0)

    # Unknown position ID
    be_fail = await daemon.on_breakeven_hook("unknown_pos_id")
    assert "Không tìm thấy vị thế mở" in be_fail


@pytest.mark.asyncio
async def test_on_close_position_hook() -> None:
    """Test single position close hook cancels symbol orders and submits Market exit."""
    env = create_test_env()
    session = MockSession([])
    pos = create_sample_position("pos_btc_01", "BTCUSDT", TradeSide.LONG, 65000.0, 0.1)
    order = create_sample_order("ord_btc_01", "BTCUSDT")
    ctx = create_sample_account_context([pos], [order])
    gateway = MockExecutionGateway(ctx, [pos], [order])
    db_mgr = MockDatabaseManager(True)
    runner = MockLivePaperRunner(["BTCUSDT"])
    tg_cfg = TelegramConfig(bot_token="test_tok", chat_id="5577408890")
    adapter = TelegramBotAdapter(tg_cfg, session, 10)  # type: ignore[arg-type]

    daemon = TradingBotSystemDaemon(
        env_mapping=env,
        session=session,  # type: ignore[arg-type]
        db_manager=db_mgr,  # type: ignore[arg-type]
        gateway=gateway,
        ws_ingestor=None,
        telegram_adapter=adapter,
        runner=runner,  # type: ignore[arg-type]
    )
    await daemon.initialize()

    close_res = await daemon.on_close_position_hook("pos_btc_01")
    assert "Đã gửi lệnh Market đóng khẩn cấp" in close_res
    assert len(gateway.cancelled_orders) == 1
    assert gateway.cancelled_orders[0] == ("BTCUSDT", "ord_btc_01")
    assert len(gateway.closed_positions) == 1
    assert gateway.closed_positions[0] == ("BTCUSDT", 0.1)

    # Unknown position ID
    fail_res = await daemon.on_close_position_hook("unknown_pos_id")
    assert "Không tìm thấy vị thế mở" in fail_res


@pytest.mark.asyncio
async def test_send_cycle_notification_hook() -> None:
    """Test 30-minute trading cycle summary notification hook formats report card."""
    env = create_test_env()
    session = MockSession([])
    pos = create_sample_position("pos_btc_01", "BTCUSDT", TradeSide.LONG, 65000.0, 0.1)
    ctx = create_sample_account_context([pos], [])
    gateway = MockExecutionGateway(ctx, [pos], [])
    db_mgr = MockDatabaseManager(True)
    runner = MockLivePaperRunner(["BTCUSDT"])
    tg_cfg = TelegramConfig(bot_token="test_tok", chat_id="5577408890")
    adapter = TelegramBotAdapter(tg_cfg, session, 10)  # type: ignore[arg-type]

    daemon = TradingBotSystemDaemon(
        env_mapping=env,
        session=session,  # type: ignore[arg-type]
        db_manager=db_mgr,  # type: ignore[arg-type]
        gateway=gateway,
        ws_ingestor=None,
        telegram_adapter=adapter,
        runner=runner,  # type: ignore[arg-type]
    )
    await daemon.initialize()

    summary = create_sample_cycle_summary()
    sent = await daemon.send_cycle_notification(summary)
    assert sent is True

    # Verify cycle report card sent in recorded Telegram requests
    assert len(session.recorded_requests) >= 2
    last_req = session.recorded_requests[-1]
    assert "[30m CYCLE REPORT]" in last_req["json"]["text"]
    assert "*NAV:* `$10,050.00`" in last_req["json"]["text"]
    assert "*Daily PnL:*" in last_req["json"]["text"]
    assert "Arbiter Verdict:* BTC: PASSED" in last_req["json"]["text"]


@pytest.mark.asyncio
async def test_daemon_run_and_shutdown_lifecycle() -> None:
    """Test full daemon execution loop, polling task start, and graceful shutdown."""
    env = create_test_env()
    session = MockSession([])
    pos = create_sample_position("pos_btc_01", "BTCUSDT", TradeSide.LONG, 65000.0, 0.1)
    ctx = create_sample_account_context([pos], [])
    gateway = MockExecutionGateway(ctx, [pos], [])
    db_mgr = MockDatabaseManager(True)
    runner = MockLivePaperRunner(["BTCUSDT"])
    ws_ing = MockWSIngestor(True)
    tg_cfg = TelegramConfig(bot_token="test_tok", chat_id="5577408890")
    adapter = TelegramBotAdapter(tg_cfg, session, 10)  # type: ignore[arg-type]

    daemon = TradingBotSystemDaemon(
        env_mapping=env,
        session=session,  # type: ignore[arg-type]
        db_manager=db_mgr,  # type: ignore[arg-type]
        gateway=gateway,
        ws_ingestor=ws_ing,  # type: ignore[arg-type]
        telegram_adapter=adapter,
        runner=runner,  # type: ignore[arg-type]
    )

    # Launch daemon.run() in background
    run_task = asyncio.create_task(daemon.run())
    await asyncio.sleep(0.05)

    assert daemon.is_running is True
    assert adapter.is_polling is True
    assert runner.start_called is True

    # Request graceful shutdown
    daemon.request_shutdown()
    await run_task

    assert daemon.is_running is False
    assert adapter.is_polling is False
    assert runner.stop_called is True
    assert ws_ing.stop_called is True
    assert db_mgr.close_called is True
    assert gateway.close_called is True

    # Verify shutdown notification sent to Telegram
    assert len(session.recorded_requests) >= 2
    shutdown_req = session.recorded_requests[-1]
    assert "Hệ thống đang tắt an toàn" in shutdown_req["json"]["text"]


@pytest.mark.asyncio
async def test_signal_handlers_registration() -> None:
    """Test cross-platform signal handler setup does not raise."""
    env = create_test_env()
    session = MockSession([])
    tg_cfg = TelegramConfig(bot_token="test_tok", chat_id="5577408890")
    adapter = TelegramBotAdapter(tg_cfg, session, 10)  # type: ignore[arg-type]

    daemon = TradingBotSystemDaemon(
        env_mapping=env,
        session=session,  # type: ignore[arg-type]
        db_manager=None,
        gateway=None,
        ws_ingestor=None,
        telegram_adapter=adapter,
        runner=None,
    )

    # Registration must be safe on Windows and Unix
    daemon.setup_signal_handlers()

    # Trigger request shutdown manually to confirm event set
    daemon.request_shutdown()
    assert daemon.is_shutdown_requested is True


@pytest.mark.asyncio
async def test_main_function_entrypoint_success() -> None:
    """Test async main() entrypoint returns 0 on graceful termination."""
    with patch("trading_bot.interfaces.main.TradingBotSystemDaemon") as mock_daemon_cls:
        mock_instance = AsyncMock()
        mock_instance.setup_signal_handlers = MagicMock()
        mock_instance.initialize = AsyncMock()
        mock_instance.run = AsyncMock()
        mock_instance.shutdown = AsyncMock()
        mock_daemon_cls.return_value = mock_instance

        exit_code = await main()
        assert exit_code == 0
        mock_instance.setup_signal_handlers.assert_called_once()
        mock_instance.initialize.assert_awaited_once()
        mock_instance.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_main_function_entrypoint_exception_handling() -> None:
    """Test async main() entrypoint returns 1 and invokes shutdown on fatal error."""
    with patch("trading_bot.interfaces.main.TradingBotSystemDaemon") as mock_daemon_cls:
        mock_instance = AsyncMock()
        mock_instance.setup_signal_handlers = MagicMock()
        mock_instance.initialize = AsyncMock(side_effect=RuntimeError("Fatal DB failure"))
        mock_instance.shutdown = AsyncMock()
        mock_daemon_cls.return_value = mock_instance

        exit_code = await main()
        assert exit_code == 1
        mock_instance.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_daemon_initializes_model_services_and_wires_telegram_hooks(tmp_path: Path) -> None:
    """Test daemon initializes FileModelRegistry, ModelTrainingService, ModelPromotionService, and wires hooks."""
    mock_session = AsyncMock()
    mock_db = MagicMock()
    mock_db.get_pool.return_value = None
    mock_gateway = AsyncMock()
    mock_gateway.sync_time_offset.return_value = 10.0
    mock_runner = MagicMock()
    mock_runner.handle_cycle_ready = AsyncMock()

    mock_telegram = MagicMock(spec=TelegramBotAdapter)
    mock_telegram.config.chat_id = "5577408890"
    mock_telegram.start = AsyncMock()
    mock_telegram.stop = AsyncMock()
    mock_telegram.send_message = AsyncMock()
    mock_telegram.send_text_message = AsyncMock()
    mock_telegram.set_challenger_pending = MagicMock()
    mock_telegram.set_execution_mode = MagicMock()
    mock_telegram.set_account_context = MagicMock()
    mock_telegram.set_on_mode_change = MagicMock()
    mock_telegram.set_on_panic_close = MagicMock()
    mock_telegram.set_on_breakeven = MagicMock()
    mock_telegram.set_on_close_position = MagicMock()
    mock_telegram.set_on_approve_model = MagicMock()
    mock_telegram.set_on_retrain = MagicMock()
    mock_telegram.register_custom_command = MagicMock()

    env = create_test_env()
    env["MODELS_DIR"] = str(tmp_path / "models")

    daemon = TradingBotSystemDaemon(
        env_mapping=env,
        session=mock_session,
        db_manager=mock_db,
        gateway=mock_gateway,
        ws_ingestor=None,
        telegram_adapter=mock_telegram,
        runner=mock_runner,
    )

    await daemon.initialize()

    assert daemon.model_registry is not None
    assert daemon.training_service is not None
    assert daemon.promotion_service is not None

    mock_telegram.set_on_approve_model.assert_called_once()
    mock_telegram.set_on_retrain.assert_called_once()

    # Verify on_retrain_hook launches background training & sends comparison report + inline button
    retrain_res = await daemon.on_retrain_hook("fast")
    assert "Challenger retraining job launched" in retrain_res
    if daemon.background_tasks:
        await asyncio.gather(*list(daemon.background_tasks))

    mock_telegram.set_challenger_pending.assert_called_with(True)
    mock_telegram.send_text_message.assert_awaited()
    sent_text, sent_kb = mock_telegram.send_text_message.call_args[0]
    assert "BÁO CÁO ĐỐI ĐẦU CHAMPION vs CHALLENGER" in sent_text
    assert sent_kb is not None
    assert sent_kb[0][0].callback_data == "confirm_approve_model"

    await daemon.shutdown()
