"""Unit tests for BinanceFuturesGatewayAdapter and Invariant #9 compliance."""

from __future__ import annotations

import ast
import hashlib
import hmac
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from scripts.check_no_defaults_or_hardcode import ZeroDefaultAndHardcodeVisitor

from trading_bot.adapters.binance.binance_futures_gateway import (
    BinanceFuturesGatewayAdapter,
)
from trading_bot.domain.entities.common.enums import (
    EntryOrderType,
    ExecutionMode,
    TradeSide,
)
from trading_bot.domain.entities.execution.trade_blueprint import BracketOrderTicket
from trading_bot.infrastructure.config_schemas import BinanceConfig
from trading_bot.ports.execution_gateway_port import ExecutionGatewayPort


class MockResponse:
    """Mock aiohttp response context manager (Zero Defaults)."""

    def __init__(self, status: int, data: Any, text_data: str) -> None:
        self.status: int = status
        self._data: Any = data
        self._text_data: str = text_data

    async def json(self) -> Any:
        return self._data

    async def text(self) -> str:
        return self._text_data

    async def __aenter__(self) -> MockResponse:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        pass


class MockClientSession:
    """Mock aiohttp ClientSession intercepting requests (Zero Defaults)."""

    def __init__(
        self,
        handler: Callable[[str, str, dict[str, Any] | None, dict[str, str] | None], MockResponse],
    ) -> None:
        self.handler: Callable[
            [str, str, dict[str, Any] | None, dict[str, str] | None], MockResponse
        ] = handler
        self.closed: bool = False
        self.calls: list[tuple[str, str, dict[str, Any] | None, dict[str, str] | None]] = []

    def request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> MockResponse:
        self.calls.append((method, url, params, headers))
        return self.handler(method, url, params, headers)

    async def close(self) -> None:
        self.closed = True


def make_binance_config(testnet: bool) -> BinanceConfig:
    """Helper factory for BinanceConfig without default arguments."""
    return BinanceConfig(
        api_key="test_api_key_12345",
        api_secret="test_api_secret_67890",
        testnet=testnet,
        recv_window_ms=5000,
    )


def make_bracket_ticket(
    symbol: str,
    side: TradeSide,
    entry_type: EntryOrderType,
    entry_price: float,
    stop_loss_price: float,
    take_profit_price: float,
    quantity: float,
    leverage: int,
) -> BracketOrderTicket:
    """Helper factory for BracketOrderTicket without default arguments."""
    return BracketOrderTicket(
        order_id="ticket-uuid-001",
        symbol=symbol,
        side=side,
        entry_type=entry_type,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        quantity=quantity,
        leverage=leverage,
    )


def test_isinstance_execution_gateway_port() -> None:
    """Validate BinanceFuturesGatewayAdapter conforms to ExecutionGatewayPort Protocol."""
    cfg = make_binance_config(False)
    adapter = BinanceFuturesGatewayAdapter(cfg, None)
    assert isinstance(adapter, ExecutionGatewayPort)


def test_base_url_resolution() -> None:
    """Validate base_url switches correctly between testnet and realnet."""
    testnet_cfg = make_binance_config(True)
    realnet_cfg = make_binance_config(False)

    testnet_adapter = BinanceFuturesGatewayAdapter(testnet_cfg, None)
    realnet_adapter = BinanceFuturesGatewayAdapter(realnet_cfg, None)

    assert testnet_adapter.base_url == "https://testnet.binancefuture.com"
    assert realnet_adapter.base_url == "https://fapi.binance.com"


def test_sign_params_hmac_and_headers() -> None:
    """Validate HMAC-SHA256 signature, timestamp, and recvWindow generation."""
    cfg = make_binance_config(True)
    adapter = BinanceFuturesGatewayAdapter(cfg, None)

    params: dict[str, Any] = {"symbol": "BTCUSDT", "side": "BUY"}
    signed = adapter.sign_params(params)

    assert "timestamp" in signed
    assert "recvWindow" in signed
    assert signed["recvWindow"] == 5000
    assert "signature" in signed

    # Verify signature independently
    secret_bytes = cfg.api_secret.encode("utf-8")
    query_parts = f"symbol=BTCUSDT&side=BUY&timestamp={signed['timestamp']}&recvWindow=5000"
    expected_sig = hmac.new(secret_bytes, query_parts.encode("utf-8"), hashlib.sha256).hexdigest()
    assert signed["signature"] == expected_sig


@pytest.mark.asyncio
async def test_place_bracket_order_market_long() -> None:
    """Validate place_bracket_order submits entry, SL, and TP orders for Long Market."""
    cfg = make_binance_config(True)

    order_requests: list[dict[str, Any]] = []

    def handle_request(
        method: str,
        url: str,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> MockResponse:
        assert params is not None
        assert headers is not None
        assert headers.get("X-MBX-APIKEY") == cfg.api_key
        order_requests.append(dict(params))
        if len(order_requests) == 1:
            return MockResponse(200, {"orderId": 987654321, "status": "FILLED"}, "OK")
        if len(order_requests) == 2:
            return MockResponse(200, {"orderId": 987654322, "status": "NEW"}, "OK")
        return MockResponse(200, {"orderId": 987654323, "status": "NEW"}, "OK")

    mock_session = MockClientSession(handle_request)
    adapter = BinanceFuturesGatewayAdapter(cfg, mock_session)  # pyright: ignore[reportArgumentType]

    ticket = make_bracket_ticket(
        symbol="BTCUSDT",
        side=TradeSide.LONG,
        entry_type=EntryOrderType.MARKET,
        entry_price=50000.0,
        stop_loss_price=49000.0,
        take_profit_price=52000.0,
        quantity=0.25,
        leverage=10,
    )

    parent_id = await adapter.place_bracket_order(ticket)

    assert parent_id == "987654321"
    assert len(order_requests) == 3

    # Check 1. Entry order
    entry_req = order_requests[0]
    assert entry_req["symbol"] == "BTCUSDT"
    assert entry_req["side"] == "BUY"
    assert entry_req["type"] == "MARKET"
    assert entry_req["quantity"] == "0.25"
    assert entry_req["newClientOrderId"] == "ticket-uuid-001"

    # Check 2. Stop Loss order
    sl_req = order_requests[1]
    assert sl_req["symbol"] == "BTCUSDT"
    assert sl_req["side"] == "SELL"
    assert sl_req["type"] == "STOP_MARKET"
    assert sl_req["stopPrice"] == "49000.0"
    assert sl_req["quantity"] == "0.25"
    assert sl_req["reduceOnly"] == "true"
    assert sl_req["newClientOrderId"] == "ticket-uuid-001_sl"

    # Check 3. Take Profit order
    tp_req = order_requests[2]
    assert tp_req["symbol"] == "BTCUSDT"
    assert tp_req["side"] == "SELL"
    assert tp_req["type"] == "TAKE_PROFIT_MARKET"
    assert tp_req["stopPrice"] == "52000.0"
    assert tp_req["quantity"] == "0.25"
    assert tp_req["reduceOnly"] == "true"
    assert tp_req["newClientOrderId"] == "ticket-uuid-001_tp"


@pytest.mark.asyncio
async def test_place_bracket_order_limit_short() -> None:
    """Validate place_bracket_order submits limit entry with GTC and correct sides for Short."""
    cfg = make_binance_config(False)

    order_requests: list[dict[str, Any]] = []

    def handle_request(
        method: str,
        url: str,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> MockResponse:
        assert params is not None
        order_requests.append(dict(params))
        return MockResponse(200, {"orderId": 11223344, "status": "NEW"}, "OK")

    mock_session = MockClientSession(handle_request)
    adapter = BinanceFuturesGatewayAdapter(cfg, mock_session)  # pyright: ignore[reportArgumentType]

    ticket = make_bracket_ticket(
        symbol="ETHUSDT",
        side=TradeSide.SHORT,
        entry_type=EntryOrderType.LIMIT_PULLBACK_15M,
        entry_price=3000.0,
        stop_loss_price=3100.0,
        take_profit_price=2800.0,
        quantity=1.5,
        leverage=5,
    )

    parent_id = await adapter.place_bracket_order(ticket)

    assert parent_id == "11223344"
    assert len(order_requests) == 3

    # Check Entry
    assert order_requests[0]["side"] == "SELL"
    assert order_requests[0]["type"] == "LIMIT"
    assert order_requests[0]["price"] == "3000.0"
    assert order_requests[0]["timeInForce"] == "GTC"

    # Check SL
    assert order_requests[1]["side"] == "BUY"
    assert order_requests[1]["type"] == "STOP_MARKET"
    assert order_requests[1]["stopPrice"] == "3100.0"

    # Check TP
    assert order_requests[2]["side"] == "BUY"
    assert order_requests[2]["type"] == "TAKE_PROFIT_MARKET"
    assert order_requests[2]["stopPrice"] == "2800.0"


@pytest.mark.asyncio
async def test_cancel_order_numeric_and_client_id() -> None:
    """Validate cancel_order handles both numeric orderId and string clientOrderId."""
    cfg = make_binance_config(True)
    captured_params: list[dict[str, Any]] = []

    def handle_request(
        method: str,
        url: str,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> MockResponse:
        assert method == "DELETE"
        assert "/fapi/v1/order" in url
        assert params is not None
        captured_params.append(dict(params))
        return MockResponse(200, {"orderId": 556677, "status": "CANCELED"}, "OK")

    mock_session = MockClientSession(handle_request)
    adapter = BinanceFuturesGatewayAdapter(cfg, mock_session)  # pyright: ignore[reportArgumentType]

    res_num = await adapter.cancel_order("BTCUSDT", "556677")
    assert res_num is True
    assert captured_params[0]["orderId"] == "556677"
    assert "origClientOrderId" not in captured_params[0]

    res_client = await adapter.cancel_order("BTCUSDT", "custom_client_id_abc")
    assert res_client is True
    assert captured_params[1]["origClientOrderId"] == "custom_client_id_abc"
    assert "orderId" not in captured_params[1]


@pytest.mark.asyncio
async def test_get_position_notional() -> None:
    """Validate get_position_notional computes absolute total notional across positions."""
    cfg = make_binance_config(True)

    def handle_request(
        method: str,
        url: str,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> MockResponse:
        assert method == "GET"
        assert "/fapi/v2/positionRisk" in url
        data = [
            {"symbol": "BTCUSDT", "positionAmt": "-0.5", "notional": "-25000.50"},
            {"symbol": "ETHUSDT", "positionAmt": "0.0", "notional": "0.0"},
        ]
        return MockResponse(200, data, "OK")

    mock_session = MockClientSession(handle_request)
    adapter = BinanceFuturesGatewayAdapter(cfg, mock_session)  # pyright: ignore[reportArgumentType]

    notional = await adapter.get_position_notional("BTCUSDT")
    assert notional == pytest.approx(25000.50)


@pytest.mark.asyncio
async def test_fetch_active_positions_filtering_and_mapping() -> None:
    """Validate fetch_active_positions filters zero positions and maps all snapshot fields."""
    cfg = make_binance_config(False)

    raw_positions = [
        {
            "symbol": "BTCUSDT",
            "positionAmt": "0.10",
            "entryPrice": "50000.0",
            "markPrice": "51000.0",
            "unRealizedProfit": "100.0",
            "liquidationPrice": "42000.0",
            "leverage": "10",
            "isolatedMargin": "500.0",
            "notional": "5100.0",
            "stopLossPrice": "49000.0",
            "takeProfitPrice": "53000.0",
        },
        {
            "symbol": "SOLUSDT",
            "positionAmt": "0.0",
            "entryPrice": "0.0",
            "markPrice": "150.0",
            "unRealizedProfit": "0.0",
            "liquidationPrice": "0.0",
            "leverage": "20",
            "isolatedMargin": "0.0",
            "notional": "0.0",
        },
        {
            "symbol": "ETHUSDT",
            "positionAmt": "-2.0",
            "entryPrice": "3000.0",
            "markPrice": "2900.0",
            "unRealizedProfit": "200.0",
            "liquidationPrice": "3600.0",
            "leverage": "5",
            "isolatedMargin": "1200.0",
            "notional": "-5800.0",
        },
    ]

    def handle_request(
        method: str,
        url: str,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> MockResponse:
        return MockResponse(200, raw_positions, "OK")

    mock_session = MockClientSession(handle_request)
    adapter = BinanceFuturesGatewayAdapter(cfg, mock_session)  # pyright: ignore[reportArgumentType]

    positions = await adapter.fetch_active_positions()

    # Position with 0 size (SOLUSDT) must be filtered out
    assert len(positions) == 2

    # Check BTCUSDT position
    btc = positions[0]
    assert btc.symbol == "BTCUSDT"
    assert btc.side == TradeSide.LONG
    assert btc.quantity == 0.10
    assert btc.entry_price == 50000.0
    assert btc.mark_price == 51000.0
    assert btc.notional_usd == 5100.0
    assert btc.leverage == 10
    assert btc.isolated_margin_usd == 500.0
    assert btc.unrealized_pnl_usd == 100.0
    assert btc.unrealized_pnl_pct == pytest.approx(0.20)
    assert btc.btc_beta == 1.0
    assert btc.stop_loss_price == 49000.0
    assert btc.take_profit_price == 53000.0
    assert btc.is_sl_at_breakeven is False
    assert btc.effective_at_risk_usd == pytest.approx(100.0)  # (50000 - 49000) * 0.10

    # Check ETHUSDT short position
    eth = positions[1]
    assert eth.symbol == "ETHUSDT"
    assert eth.side == TradeSide.SHORT
    assert eth.quantity == 2.0
    assert eth.entry_price == 3000.0
    assert eth.notional_usd == 5800.0
    assert eth.btc_beta == 1.2


@pytest.mark.asyncio
async def test_fetch_pending_orders_mapping() -> None:
    """Validate fetch_pending_orders maps open limit and market orders correctly."""
    cfg = make_binance_config(True)

    raw_orders = [
        {
            "orderId": 889900,
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "price": "49500.0",
            "origQty": "0.20",
            "time": 1700000000000,
        },
        {
            "orderId": 889901,
            "symbol": "SOLUSDT",
            "side": "SELL",
            "type": "MARKET",
            "price": "0.0",
            "stopPrice": "145.0",
            "origQty": "10.0",
            "updateTime": 1700000001000,
        },
    ]

    def handle_request(
        method: str,
        url: str,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> MockResponse:
        return MockResponse(200, raw_orders, "OK")

    mock_session = MockClientSession(handle_request)
    adapter = BinanceFuturesGatewayAdapter(cfg, mock_session)  # pyright: ignore[reportArgumentType]

    orders = await adapter.fetch_pending_orders()

    assert len(orders) == 2

    # Check Limit Order
    lim = orders[0]
    assert lim.order_id == "889900"
    assert lim.symbol == "BTCUSDT"
    assert lim.side == TradeSide.LONG
    assert lim.order_type == EntryOrderType.LIMIT_PULLBACK_15M
    assert lim.limit_price == 49500.0
    assert lim.notional_usd == pytest.approx(9900.0)
    assert lim.reserved_margin_usd == pytest.approx(9900.0)
    assert lim.created_at_ms == 1700000000000

    # Check Market Order
    mkt = orders[1]
    assert mkt.order_id == "889901"
    assert mkt.symbol == "SOLUSDT"
    assert mkt.side == TradeSide.SHORT
    assert mkt.order_type == EntryOrderType.MARKET
    assert mkt.limit_price == 145.0
    assert mkt.notional_usd == pytest.approx(1450.0)


@pytest.mark.asyncio
async def test_fetch_account_portfolio_context() -> None:
    """Validate fetch_account_portfolio_context maps balance, margin, exposure, and heat."""
    cfg = make_binance_config(True)

    account_data = {
        "updateTime": 1710000000000,
        "totalWalletBalance": "10000.0",
        "totalUnrealizedProfit": "500.0",
        "totalMarginBalance": "10500.0",
        "totalInitialMargin": "2000.0",
        "availableBalance": "8500.0",
        "current_daily_pnl_usd": "500.0",
        "current_daily_dd_pct": "0.01",
        "remaining_daily_dd_budget_pct": "0.04",
        "current_max_dd_pct": "0.02",
        "remaining_max_dd_budget_pct": "0.08",
        "consecutive_losses_count": 1,
        "positions": [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0.10",
                "entryPrice": "50000.0",
                "markPrice": "55000.0",
                "unRealizedProfit": "500.0",
                "liquidationPrice": "40000.0",
                "leverage": "10",
                "isolatedMargin": "500.0",
                "notional": "5500.0",
                "stopLossPrice": "49000.0",
                "takeProfitPrice": "60000.0",
            }
        ],
        "open_orders": [],
    }

    def handle_request(
        method: str,
        url: str,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> MockResponse:
        return MockResponse(200, account_data, "OK")

    mock_session = MockClientSession(handle_request)
    adapter = BinanceFuturesGatewayAdapter(cfg, mock_session)  # pyright: ignore[reportArgumentType]

    ctx = await adapter.fetch_account_portfolio_context()

    assert ctx.execution_mode == ExecutionMode.TESTNET
    assert ctx.timestamp_ms == 1710000000000
    assert ctx.wallet_balance_usd == 10000.0
    assert ctx.unrealized_pnl_usd == 500.0
    assert ctx.equity_nav_usd == 10500.0
    assert ctx.used_margin_usd == 2000.0
    assert ctx.available_margin_usd == 8500.0
    assert ctx.margin_utilization_pct == pytest.approx(2000.0 / 10500.0)
    assert len(ctx.active_positions) == 1
    assert len(ctx.pending_orders) == 0
    assert ctx.gross_exposure_usd == 5500.0
    assert ctx.net_directional_delta_usd == 5500.0
    assert ctx.net_btc_beta_exposure_usd == 5500.0
    assert ctx.raw_open_heat_pct > 0.0
    assert ctx.effective_open_heat_pct > 0.0


@pytest.mark.asyncio
async def test_api_error_response_raises_runtime_error() -> None:
    """Validate that HTTP 4xx/5xx responses raise informative RuntimeError."""
    cfg = make_binance_config(True)

    def handle_request(
        method: str,
        url: str,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> MockResponse:
        return MockResponse(400, {"code": -1102, "msg": "Param error"}, "Param error")

    mock_session = MockClientSession(handle_request)
    adapter = BinanceFuturesGatewayAdapter(cfg, mock_session)  # pyright: ignore[reportArgumentType]

    ticket = make_bracket_ticket(
        symbol="BTCUSDT",
        side=TradeSide.LONG,
        entry_type=EntryOrderType.MARKET,
        entry_price=50000.0,
        stop_loss_price=49000.0,
        take_profit_price=52000.0,
        quantity=0.1,
        leverage=10,
    )

    with pytest.raises(RuntimeError, match="Binance API error \\[400\\]"):
        await adapter.place_bracket_order(ticket)


@pytest.mark.asyncio
async def test_session_close_and_context_manager() -> None:
    """Validate async context manager and session cleanup."""
    cfg = make_binance_config(True)

    def handle_request(
        method: str,
        url: str,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> MockResponse:
        return MockResponse(200, [], "OK")

    mock_session = MockClientSession(handle_request)
    adapter = BinanceFuturesGatewayAdapter(cfg, mock_session)  # pyright: ignore[reportArgumentType]

    async with adapter as gw:
        assert gw.base_url == "https://testnet.binancefuture.com"

    # Since session was injected from caller (not owned), gw does not close caller's session
    assert mock_session.closed is False


def test_ast_compliance_zero_defaults_zero_fallbacks() -> None:
    """Validate that gateway adapter and test file strictly comply with Invariant #9."""
    repo_root = Path(__file__).resolve().parent.parent.parent

    target_files = [
        repo_root / "src" / "trading_bot" / "adapters" / "binance" / "binance_futures_gateway.py",
        Path(__file__).resolve(),
    ]

    all_violations: list[str] = []
    for f in target_files:
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        visitor = ZeroDefaultAndHardcodeVisitor(f)
        visitor.visit(tree)
        all_violations.extend(visitor.violations)

    assert not all_violations, "Invariant #9 violations found:\n" + "\n".join(all_violations)
