"""Unit tests for Telegram pure domain entities, ports, and configuration loaders."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from trading_bot.domain.entities.quant.symbolic_alpha import ModelTournamentReport
from trading_bot.domain.entities.telegram.telegram_models import (
    InvalidTelegramCommandException,
    TelegramBotCommand,
    TelegramCallbackQuery,
    TelegramCommandRequest,
    TelegramCycleReport,
    TelegramInlineButton,
    TelegramInlineKeyboard,
    TelegramMessage,
    TelegramParsingException,
    TelegramUpdate,
)
from trading_bot.infrastructure.config_schemas import ConfigLoader
from trading_bot.ports.telegram_port import (
    TelegramCommandHandlerPort,
    TelegramNotifierPort,
)


def test_telegram_inline_button_serialization() -> None:
    """Verify TelegramInlineButton serializes correctly and enforces explicit fields."""
    btn1 = TelegramInlineButton(text="Confirm", callback_data="btn_confirm", url=None)
    assert btn1.text == "Confirm"
    assert btn1.callback_data == "btn_confirm"
    assert btn1.url is None
    assert btn1.to_dict() == {"text": "Confirm", "callback_data": "btn_confirm"}

    btn2 = TelegramInlineButton(text="Docs", callback_data=None, url="https://example.com")
    assert btn2.to_dict() == {"text": "Docs", "url": "https://example.com"}

    # Zero-default discipline: omitting fields must raise ValidationError
    with pytest.raises(ValidationError):
        TelegramInlineButton(text="NoCallbackOrUrl")  # pyright: ignore[reportCallIssue]


def test_telegram_inline_keyboard_serialization() -> None:
    """Verify TelegramInlineKeyboard creates proper Telegram Bot API structure."""
    btn_be = TelegramInlineButton(text="🛡️ Breakeven", callback_data="be_BTCUSDT", url=None)
    btn_close = TelegramInlineButton(text="🚨 Close", callback_data="close_BTCUSDT", url=None)
    kb = TelegramInlineKeyboard(inline_keyboard=[[btn_be, btn_close]])

    payload = kb.to_dict()
    assert "inline_keyboard" in payload
    assert len(payload["inline_keyboard"]) == 1
    assert len(payload["inline_keyboard"][0]) == 2
    assert payload["inline_keyboard"][0][0] == {
        "text": "🛡️ Breakeven",
        "callback_data": "be_BTCUSDT",
    }
    assert payload["inline_keyboard"][0][1] == {
        "text": "🚨 Close",
        "callback_data": "close_BTCUSDT",
    }


def test_telegram_message_to_api_payload() -> None:
    """Verify TelegramMessage payload generation with Markdown and keyboard."""
    btn = TelegramInlineButton(text="Status", callback_data="cmd_status", url=None)
    kb = TelegramInlineKeyboard(inline_keyboard=[[btn]])

    msg = TelegramMessage(
        message_id=101,
        chat_id="5577408890",
        text="Hello Operator",
        parse_mode="Markdown",
        reply_markup=kb,
        date=1700000000,
    )

    payload = msg.to_api_payload()
    assert payload["chat_id"] == "5577408890"
    assert payload["text"] == "Hello Operator"
    assert payload["parse_mode"] == "Markdown"
    assert payload["reply_markup"] == {
        "inline_keyboard": [[{"text": "Status", "callback_data": "cmd_status"}]]
    }


def test_telegram_message_raw_list_reply_markup() -> None:
    """Verify TelegramMessage accepts raw list of TelegramInlineButton rows."""
    btn = TelegramInlineButton(text="Panic", callback_data="panic_confirm", url=None)
    msg = TelegramMessage(
        message_id=None,
        chat_id="5577408890",
        text="Confirm Emergency Panic",
        parse_mode=None,
        reply_markup=[[btn]],
        date=None,
    )
    payload = msg.to_api_payload()
    assert payload["reply_markup"] == {
        "inline_keyboard": [[{"text": "Panic", "callback_data": "panic_confirm"}]]
    }


def test_telegram_update_from_message_dict() -> None:
    """Verify TelegramUpdate.from_dict parses incoming getUpdates message."""
    raw_payload: dict[str, Any] = {
        "update_id": 998877,
        "message": {
            "message_id": 1234,
            "from": {"id": 5577408890, "is_bot": False, "first_name": "Operator"},
            "chat": {"id": 5577408890, "type": "private"},
            "date": 1700001234,
            "text": "/status",
        },
    }
    update = TelegramUpdate.from_dict(raw_payload)
    assert update.update_id == 998877
    assert update.message is not None
    assert update.message.chat_id == "5577408890"
    assert update.message.text == "/status"
    assert update.message.message_id == 1234
    assert update.callback_query is None


def test_telegram_update_from_callback_query_dict() -> None:
    """Verify TelegramUpdate.from_dict parses incoming inline callback query."""
    raw_payload: dict[str, Any] = {
        "update_id": 998878,
        "callback_query": {
            "id": "cb_query_42",
            "from": {"id": 5577408890, "is_bot": False, "first_name": "Operator"},
            "message": {
                "message_id": 5678,
                "chat": {"id": 5577408890, "type": "private"},
                "text": "Choose action",
            },
            "data": "be_ETHUSDT",
        },
    }
    update = TelegramUpdate.from_dict(raw_payload)
    assert update.update_id == 998878
    assert update.message is None
    assert update.callback_query is not None
    assert update.callback_query.callback_id == "cb_query_42"
    assert update.callback_query.from_user_id == "5577408890"
    assert update.callback_query.chat_id == "5577408890"
    assert update.callback_query.data == "be_ETHUSDT"
    assert update.callback_query.message_id == 5678


def test_telegram_callback_query_entity() -> None:
    """Verify TelegramCallbackQuery explicit initialization and zero-default discipline."""
    cb = TelegramCallbackQuery(
        callback_id="cb_test_01",
        from_user_id="5577408890",
        chat_id="5577408890",
        data="close_all",
        message_id=999,
    )
    assert cb.callback_id == "cb_test_01"
    assert cb.from_user_id == "5577408890"
    assert cb.chat_id == "5577408890"
    assert cb.data == "close_all"
    assert cb.message_id == 999

    with pytest.raises(ValidationError):
        TelegramCallbackQuery(callback_id="missing_fields")  # pyright: ignore[reportCallIssue]


def test_telegram_update_missing_required_fields_raises() -> None:
    """Verify TelegramUpdate raises TelegramParsingException on missing required keys."""
    with pytest.raises(TelegramParsingException, match="Missing required 'update_id'"):
        TelegramUpdate.from_dict({})

    with pytest.raises(TelegramParsingException, match=r"Missing 'chat\.id'"):
        TelegramUpdate.from_dict({"update_id": 1, "message": {"text": "hi"}})

    with pytest.raises(TelegramParsingException, match="Malformed callback_query"):
        TelegramUpdate.from_dict({"update_id": 2, "callback_query": {"id": "123"}})


def test_telegram_command_request_parse() -> None:
    """Verify TelegramCommandRequest parses valid slash commands with arguments."""
    req1 = TelegramCommandRequest.parse("/status", "5577408890", "5577408890")
    assert req1.command == "status"
    assert req1.args == []
    assert req1.user_id == "5577408890"
    assert req1.chat_id == "5577408890"

    req2 = TelegramCommandRequest.parse("/mode testnet", "5577408890", "5577408890")
    assert req2.command == "mode"
    assert req2.args == ["testnet"]

    req3 = TelegramCommandRequest.parse("/pause 2.5", "5577408890", "5577408890")
    assert req3.command == "pause"
    assert req3.args == ["2.5"]

    # With bot mention e.g. /status@mybot
    req4 = TelegramCommandRequest.parse(
        "/status@nmai_llm_quant_trading_bot", "5577408890", "5577408890"
    )
    assert req4.command == "status"
    assert req4.args == []


def test_telegram_command_request_invalid_raises() -> None:
    """Verify TelegramCommandRequest raises InvalidTelegramCommandException on invalid commands."""
    with pytest.raises(InvalidTelegramCommandException, match="not a valid slash command"):
        TelegramCommandRequest.parse("status", "5577408890", "5577408890")

    with pytest.raises(InvalidTelegramCommandException, match="Empty command string"):
        TelegramCommandRequest.parse("   ", "5577408890", "5577408890")

    with pytest.raises(InvalidTelegramCommandException, match="Command name cannot be empty"):
        TelegramCommandRequest.parse("/", "5577408890", "5577408890")


def test_telegram_cycle_report_format() -> None:
    """Verify TelegramCycleReport formats into proper Markdown summary."""
    report = TelegramCycleReport(
        cycle_timestamp_ms=1710000000000,
        execution_mode="PAPER",
        portfolio_nav_usd=10250.75,
        daily_pnl_usd=250.75,
        current_drawdown_pct=0.45,
        open_positions_count=2,
        executed_orders_count=1,
        shadow_orders_count=3,
        btc_funnel_passed=True,
        altcoins_scanned_count=3,
        shap_top_features={"hurst_15m": 0.042, "vpin_roll": -0.015},
        calibrated_probabilities={"BTCUSDT": 0.68, "ETHUSDT": 0.54},
        arbiter_verdict_summary="Approved 1 ticket with 0.25 Quarter-Kelly sizing.",
    )
    formatted = report.format_telegram_markdown()
    assert "🟢 [30m CYCLE REPORT]" in formatted
    assert "Mode: PAPER" in formatted
    assert "$10,250.75" in formatted
    assert "+$250.75" in formatted
    assert "hurst_15m" in formatted
    assert "68.00%" in formatted
    assert "*Arbiter Verdict:* Approved 1 ticket" in formatted


def test_telegram_cycle_report_negative_pnl_format() -> None:
    """Verify TelegramCycleReport formats negative PnL and non-passed funnel properly."""
    report = TelegramCycleReport(
        cycle_timestamp_ms=1710000000000,
        execution_mode="LIVE",
        portfolio_nav_usd=9800.0,
        daily_pnl_usd=-200.0,
        current_drawdown_pct=2.0,
        open_positions_count=0,
        executed_orders_count=0,
        shadow_orders_count=1,
        btc_funnel_passed=False,
        altcoins_scanned_count=4,
        shap_top_features={},
        calibrated_probabilities={},
        arbiter_verdict_summary="Funnel blocked at Stage 2.",
    )
    formatted = report.format_telegram_markdown()
    assert "⚪ [30m CYCLE REPORT]" in formatted
    assert "Mode: LIVE" in formatted
    assert "📉 `-$200.00`" in formatted or "-$200.00" in formatted


def test_telegram_cycle_report_escapes_underscores_in_verdict() -> None:
    """Verify underscores in arbiter verdict are escaped to prevent Markdown entity errors."""
    report = TelegramCycleReport(
        cycle_timestamp_ms=1710000000000,
        execution_mode="PAPER",
        portfolio_nav_usd=10000.0,
        daily_pnl_usd=0.0,
        current_drawdown_pct=0.0,
        open_positions_count=0,
        executed_orders_count=0,
        shadow_orders_count=0,
        btc_funnel_passed=False,
        altcoins_scanned_count=4,
        shap_top_features={},
        calibrated_probabilities={},
        arbiter_verdict_summary="BTC: STAGE_B_FEE_EDGE | Alt Executed: 0 | Shadow: 0",
    )
    formatted = report.format_telegram_markdown()
    assert "STAGE\\_B\\_FEE\\_EDGE" in formatted


def test_telegram_bot_command_enum() -> None:
    """Verify TelegramBotCommand enum members."""
    assert TelegramBotCommand.STATUS.value == "status"
    assert TelegramBotCommand.POSITIONS.value == "positions"
    assert TelegramBotCommand.PANIC_CLOSE_ALL.value == "panic_close_all"
    assert TelegramBotCommand.APPROVE_MODEL.value == "approve_model"


class DummyTelegramNotifier:
    """Mock implementation conforming to TelegramNotifierPort."""

    def __init__(self) -> None:
        self.sent_messages: list[str] = []

    async def send_text_message(
        self,
        text: str,
        reply_markup: list[list[TelegramInlineButton]] | dict[str, Any] | None,
    ) -> bool:
        self.sent_messages.append(text)
        return True

    async def send_cycle_report(
        self,
        report: TelegramCycleReport | dict[str, Any],
    ) -> bool:
        return True

    async def send_emergency_alert(
        self,
        title: str,
        message: str,
    ) -> bool:
        self.sent_messages.append(f"{title}: {message}")
        return True

    async def send_tournament_verdict(
        self,
        report: ModelTournamentReport,
    ) -> bool:
        return True

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
        self.sent_messages.append(f"ORDER_PLACED: {symbol} {side} @ {entry_price}")
        return True

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
        self.sent_messages.append(f"POSITION_CLOSED: {symbol} {side} PnL={pnl_usd}")
        return True

    async def send_error_alert(
        self,
        component: str,
        error_message: str,
        is_fatal: bool,
    ) -> bool:
        self.sent_messages.append(f"ERROR: {component}: {error_message}")
        return True


class DummyTelegramCommandHandler:
    """Mock implementation conforming to TelegramCommandHandlerPort."""

    async def handle_command(
        self,
        request: TelegramCommandRequest,
    ) -> str:
        return f"Handled {request.command}"

    async def handle_callback(
        self,
        callback_id: str,
        callback_data: str,
    ) -> str:
        return f"Callback {callback_id}: {callback_data}"


@pytest.mark.asyncio
async def test_telegram_ports_runtime_conformance() -> None:
    """Verify runtime checkable protocol adherence for TelegramNotifierPort and TelegramCommandHandlerPort."""
    notifier = DummyTelegramNotifier()
    assert isinstance(notifier, TelegramNotifierPort)

    handler = DummyTelegramCommandHandler()
    assert isinstance(handler, TelegramCommandHandlerPort)

    # Exercise async methods
    res_msg = await notifier.send_text_message("test message", None)
    assert res_msg is True
    assert notifier.sent_messages == ["test message"]

    res_order = await notifier.send_order_placed_alert(
        symbol="BTCUSDT",
        side="BUY",
        order_type="LIMIT",
        entry_price=65000.0,
        stop_loss_price=64000.0,
        take_profit_price=67000.0,
        quantity=0.1,
        risk_reward_ratio=2.0,
        order_id="ord-123",
    )
    assert res_order is True

    res_close = await notifier.send_position_closed_alert(
        symbol="BTCUSDT",
        side="BUY",
        close_price=66000.0,
        pnl_usd=100.0,
        pnl_pct=1.54,
        r_multiple=1.0,
        exit_reason="TAKE_PROFIT",
    )
    assert res_close is True

    res_err = await notifier.send_error_alert(
        component="watchdog",
        error_message="Test error",
        is_fatal=False,
    )
    assert res_err is True

    cmd_req = TelegramCommandRequest(
        command="status",
        args=[],
        user_id="5577408890",
        chat_id="5577408890",
        raw_text="/status",
    )
    res_cmd = await handler.handle_command(cmd_req)
    assert res_cmd == "Handled status"

    res_cb = await handler.handle_callback("cb_1", "data_1")
    assert res_cb == "Callback cb_1: data_1"


def test_config_loader_telegram_config() -> None:
    """Verify ConfigLoader loads TelegramConfig from mapping and raises on missing key."""
    env_map = {
        "TELEGRAM_BOT_TOKEN": "9180299198:AAHsdHpZpd0_RXbrziSU1iviHqbEcYHWWPw",
        "TELEGRAM_CHAT_ID": "5577408890",
    }
    cfg = ConfigLoader.load_telegram_config(env_map)
    assert cfg.bot_token == "9180299198:AAHsdHpZpd0_RXbrziSU1iviHqbEcYHWWPw"
    assert cfg.chat_id == "5577408890"

    with pytest.raises(KeyError, match="TELEGRAM_CHAT_ID"):
        ConfigLoader.load_telegram_config({"TELEGRAM_BOT_TOKEN": "token_only"})


def test_config_loader_telegram_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ConfigLoader.load_telegram_config_from_env reads os.environ."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token_from_env")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat_from_env")
    cfg = ConfigLoader.load_telegram_config_from_env()
    assert cfg.bot_token == "token_from_env"
    assert cfg.chat_id == "chat_from_env"


def test_config_loader_binance_config() -> None:
    """Verify ConfigLoader loads BinanceConfig from mapping and handles testnet parsing."""
    env_map = {
        "BINANCE_API_KEY": "test_key",
        "BINANCE_API_SECRET": "test_sec",
        "BINANCE_TESTNET": "true",
        "BINANCE_RECV_WINDOW_MS": "5000",
    }
    cfg = ConfigLoader.load_binance_config(env_map)
    assert cfg.api_key == "test_key"
    assert cfg.api_secret == "test_sec"
    assert cfg.testnet is True
    assert cfg.recv_window_ms == 5000

    env_map_live = {
        "BINANCE_API_KEY": "live_key",
        "BINANCE_API_SECRET": "live_sec",
        "BINANCE_TESTNET": "false",
        "BINANCE_RECV_WINDOW_MS": "3000",
    }
    cfg_live = ConfigLoader.load_binance_config(env_map_live)
    assert cfg_live.testnet is False
    assert cfg_live.recv_window_ms == 3000

    with pytest.raises(KeyError, match="BINANCE_API_SECRET"):
        ConfigLoader.load_binance_config({"BINANCE_API_KEY": "key_only"})


def test_config_loader_binance_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ConfigLoader.load_binance_config_from_env reads os.environ."""
    monkeypatch.setenv("BINANCE_API_KEY", "b_key")
    monkeypatch.setenv("BINANCE_API_SECRET", "b_sec")
    monkeypatch.setenv("BINANCE_TESTNET", "1")
    monkeypatch.setenv("BINANCE_RECV_WINDOW_MS", "6000")

    cfg = ConfigLoader.load_binance_config_from_env()
    assert cfg.api_key == "b_key"
    assert cfg.testnet is True
    assert cfg.recv_window_ms == 6000
