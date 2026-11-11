"""Comprehensive Unit Tests for TelegramBotAdapter (Milestone 8.2).

Enforces strictly ZERO default parameters, ZERO class field defaults, and ZERO silent fallbacks.
Tests:
- TelegramNotifierPort methods (send_text_message, send_cycle_report, send_emergency_alert, send_tournament_verdict).
- Inline keyboard serialization, markdown 400 retry with plain text.
- Long-polling loop (poll_once, start_polling, stop_polling, graceful cancellation).
- Whitelist security (chat_id matching, unauthorized rejection for message & callback).
- Slash command routing (/status, /positions, /shadow, /quota, /mode, /pause, /resume, /panic_close_all, /approve_model).
- 2-step confirmation flows for /mode live and /panic_close_all.
- Callback query handling (confirm_mode:live, confirm_panic_close, cancel_action, be:id, close:id, approve_model).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from trading_bot.adapters.telegram.telegram_bot_adapter import TelegramBotAdapter
from trading_bot.domain.entities.common.enums import (
    EntryOrderType,
    ExecutionMode,
    TradeSide,
)
from trading_bot.domain.entities.quant.symbolic_alpha import (
    ModelTournamentReport,
    TournamentVerdict,
)
from trading_bot.domain.entities.risk.account_state import (
    AccountPortfolioContext,
    ActivePositionSnapshot,
    PendingOrderSnapshot,
)
from trading_bot.domain.entities.telegram.telegram_models import (
    TelegramCommandRequest,
    TelegramCycleReport,
    TelegramInlineButton,
    TelegramSecurityViolationException,
    TelegramUpdate,
)
from trading_bot.infrastructure.config_schemas import TelegramConfig


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


def create_test_config() -> TelegramConfig:
    return TelegramConfig(
        bot_token="test_token_12345",
        chat_id="5577408890",
    )


def create_sample_account_context() -> AccountPortfolioContext:
    pos = ActivePositionSnapshot(
        position_id="pos_btc_01",
        symbol="BTCUSDT",
        side=TradeSide.LONG,
        entry_price=90000.0,
        mark_price=91500.0,
        stop_loss_price=89000.0,
        take_profit_price=94000.0,
        liquidation_price=70000.0,
        quantity=0.1,
        notional_usd=9150.0,
        isolated_margin_usd=1830.0,
        leverage=5,
        unrealized_pnl_usd=150.0,
        unrealized_pnl_pct=1.67,
        unrealized_r_multiple=1.5,
        is_sl_at_breakeven=False,
        effective_at_risk_usd=100.0,
        btc_beta=1.0,
    )
    order = PendingOrderSnapshot(
        order_id="ord_eth_01",
        symbol="ETHUSDT",
        side=TradeSide.LONG,
        order_type=EntryOrderType.LIMIT_PULLBACK_15M,
        limit_price=3100.0,
        notional_usd=1550.0,
        reserved_margin_usd=310.0,
        created_at_ms=1700000000000,
    )
    return AccountPortfolioContext(
        execution_mode=ExecutionMode.PAPER,
        timestamp_ms=1700000000000,
        wallet_balance_usd=10000.0,
        unrealized_pnl_usd=150.0,
        equity_nav_usd=10150.0,
        used_margin_usd=1830.0,
        available_margin_usd=8320.0,
        margin_utilization_pct=18.03,
        current_daily_pnl_usd=150.0,
        current_daily_dd_pct=0.5,
        remaining_daily_dd_budget_pct=2.5,
        current_max_dd_pct=1.2,
        remaining_max_dd_budget_pct=4.8,
        consecutive_losses_count=0,
        active_positions=[pos],
        pending_orders=[order],
        gross_exposure_usd=9150.0,
        net_directional_delta_usd=9150.0,
        net_btc_beta_exposure_usd=9150.0,
        raw_open_heat_pct=18.03,
        effective_open_heat_pct=18.03,
        same_direction_active_count=1,
    )


def create_sample_tournament_report(is_promotable: bool) -> ModelTournamentReport:
    if is_promotable:
        rec_verdict = TournamentVerdict.PROMOTE_CHALLENGER_IMMEDIATELY
        rationale = "Challenger demonstrates superior net Sharpe and Brier skill score."
    else:
        rec_verdict = TournamentVerdict.KEEP_CHAMPION
        rationale = "Challenger failed minimum Sharpe threshold."

    return ModelTournamentReport(
        created_at_utc="2026-11-10T12:00:00Z",
        champion_model_id="champion_v1",
        challenger_model_id="challenger_v2",
        champion_feature_count=20,
        challenger_feature_count=25,
        champion_net_sharpe=2.1,
        challenger_net_sharpe=2.8,
        champion_max_drawdown_pct=5.5,
        challenger_max_drawdown_pct=4.2,
        champion_calmar_ratio=3.5,
        challenger_calmar_ratio=4.8,
        champion_win_rate=0.58,
        challenger_win_rate=0.64,
        champion_profit_factor=1.85,
        challenger_profit_factor=2.20,
        champion_expectancy_r=0.45,
        challenger_expectancy_r=0.62,
        champion_brier_score=0.18,
        challenger_brier_score=0.14,
        champion_brier_skill_score=0.12,
        challenger_brier_skill_score=0.22,
        champion_log_loss=0.42,
        challenger_log_loss=0.35,
        champion_venn_abers_spread=0.08,
        challenger_venn_abers_spread=0.05,
        is_promotable=is_promotable,
        recommended_verdict=rec_verdict,
        approval_checklist={"sharpe_improvement": is_promotable},
        verdict_rationale=rationale,
    )


# =============================================================================
# 1. Adapter Initialization & Properties Tests
# =============================================================================


@pytest.mark.asyncio
async def test_telegram_bot_adapter_init_and_properties() -> None:
    config = create_test_config()
    session = MockSession([])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    assert adapter.config.bot_token == "test_token_12345"
    assert adapter.config.chat_id == "5577408890"
    assert not adapter.is_polling
    assert adapter.execution_mode == "PAPER"
    assert not adapter.is_paused
    assert not adapter.challenger_pending
    assert adapter.last_update_id == 0
    assert adapter.account_context is None
    assert adapter.shadow_stats == {}
    assert adapter.quota_stats == {}

    # Setters verification
    adapter.set_execution_mode("testnet")
    assert adapter.execution_mode == "TESTNET"

    adapter.set_paused(True, 2.0)
    assert adapter.is_paused

    adapter.set_challenger_pending(True)
    assert adapter.challenger_pending

    adapter.set_shadow_stats({"evaluated_count": 10, "defense_rate_pct": 90.0})
    assert adapter.shadow_stats["evaluated_count"] == 10

    adapter.set_quota_stats({"fast_rpm": 5})
    assert adapter.quota_stats["fast_rpm"] == 5

    ctx = create_sample_account_context()
    adapter.set_account_context(ctx)
    assert adapter.account_context is not None
    assert adapter.account_context.equity_nav_usd == 10150.0


# =============================================================================
# 2. TelegramNotifierPort Outbound Messaging Tests
# =============================================================================


@pytest.mark.asyncio
async def test_send_text_message_success() -> None:
    config = create_test_config()
    res = MockResponse(200, {"ok": True, "result": {"message_id": 101}}, None)
    session = MockSession([res])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    success = await adapter.send_text_message("Hello Telegram", None)
    assert success
    assert len(session.recorded_requests) == 1
    req = session.recorded_requests[0]
    assert req["url"] == "https://api.telegram.org/bottest_token_12345/sendMessage"
    assert req["json"]["chat_id"] == "5577408890"
    assert req["json"]["text"] == "Hello Telegram"
    assert req["json"]["parse_mode"] == "Markdown"
    assert "reply_markup" not in req["json"]


@pytest.mark.asyncio
async def test_send_text_message_with_inline_buttons() -> None:
    config = create_test_config()
    res = MockResponse(200, {"ok": True, "result": {"message_id": 102}}, None)
    session = MockSession([res])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    btn1 = TelegramInlineButton(text="Confirm", callback_data="confirm", url=None)
    btn2 = TelegramInlineButton(text="Docs", callback_data=None, url="https://docs.com")
    keyboard = [[btn1, btn2]]

    success = await adapter.send_text_message("Choose option:", keyboard)
    assert success
    req = session.recorded_requests[0]
    assert "reply_markup" in req["json"]
    markup = req["json"]["reply_markup"]
    assert "inline_keyboard" in markup
    assert len(markup["inline_keyboard"][0]) == 2
    assert markup["inline_keyboard"][0][0]["text"] == "Confirm"
    assert markup["inline_keyboard"][0][0]["callback_data"] == "confirm"
    assert markup["inline_keyboard"][0][1]["url"] == "https://docs.com"


@pytest.mark.asyncio
async def test_send_text_message_with_dict_reply_markup() -> None:
    config = create_test_config()
    res = MockResponse(200, {"ok": True, "result": {"message_id": 103}}, None)
    session = MockSession([res])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    custom_markup = {"inline_keyboard": [[{"text": "Custom", "callback_data": "custom"}]]}
    success = await adapter.send_text_message("Custom markup", custom_markup)
    assert success
    req = session.recorded_requests[0]
    assert req["json"]["reply_markup"] == custom_markup


@pytest.mark.asyncio
async def test_send_text_message_retry_plain_on_400_markdown_rejection() -> None:
    config = create_test_config()
    # First response fails with 400 Bad Request (bad markdown entity)
    res1 = MockResponse(
        400,
        {"ok": False, "description": "Bad Request: can't parse entities"},
        None,
    )
    # Second response (retry as plain text) succeeds with 200
    res2 = MockResponse(200, {"ok": True, "result": {"message_id": 104}}, None)
    session = MockSession([res1, res2])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    success = await adapter.send_text_message("Unescaped *markdown _test", None)
    assert success
    assert len(session.recorded_requests) == 2
    req1 = session.recorded_requests[0]
    assert req1["json"]["parse_mode"] == "Markdown"
    req2 = session.recorded_requests[1]
    assert "parse_mode" not in req2["json"]
    assert req2["json"]["text"] == "Unescaped *markdown _test"


@pytest.mark.asyncio
async def test_send_text_message_network_error_returns_false() -> None:
    config = create_test_config()
    res = MockResponse(500, {}, ConnectionError("Telegram API down"))
    session = MockSession([res])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    success = await adapter.send_text_message("Test", None)
    assert not success


@pytest.mark.asyncio
async def test_send_text_message_api_error_returns_false() -> None:
    config = create_test_config()
    res = MockResponse(200, {"ok": False, "description": "Unauthorized"}, None)
    session = MockSession([res])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    success = await adapter.send_text_message("Test", None)
    assert not success


@pytest.mark.asyncio
async def test_send_cycle_report_domain_and_dict() -> None:
    config = create_test_config()
    res1 = MockResponse(200, {"ok": True, "result": {"message_id": 105}}, None)
    res2 = MockResponse(200, {"ok": True, "result": {"message_id": 106}}, None)
    session = MockSession([res1, res2])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    cycle_entity = TelegramCycleReport(
        cycle_timestamp_ms=1700000000000,
        execution_mode="PAPER",
        portfolio_nav_usd=10500.0,
        daily_pnl_usd=250.0,
        current_drawdown_pct=0.45,
        open_positions_count=2,
        executed_orders_count=5,
        shadow_orders_count=1,
        btc_funnel_passed=True,
        altcoins_scanned_count=15,
        shap_top_features={"volatility_ratio": 0.42},
        calibrated_probabilities={"SOLUSDT": 0.68},
        arbiter_verdict_summary="Passed all risk bounds",
    )

    success1 = await adapter.send_cycle_report(cycle_entity)
    assert success1
    assert "30m CYCLE REPORT" in session.recorded_requests[0]["json"]["text"]

    # Test with dict
    cycle_dict = cycle_entity.model_dump()
    success2 = await adapter.send_cycle_report(cycle_dict)
    assert success2
    assert "30m CYCLE REPORT" in session.recorded_requests[1]["json"]["text"]


@pytest.mark.asyncio
async def test_send_emergency_alert() -> None:
    config = create_test_config()
    res = MockResponse(200, {"ok": True, "result": {"message_id": 107}}, None)
    session = MockSession([res])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    success = await adapter.send_emergency_alert("CIRCUIT_BREAKER", "Drawdown limit reached: 3.5%")
    assert success
    req = session.recorded_requests[0]
    assert "EMERGENCY ALERT: CIRCUIT_BREAKER" in req["json"]["text"]
    assert "Drawdown limit reached" in req["json"]["text"]


@pytest.mark.asyncio
async def test_send_tournament_verdict_promotable_and_non_promotable() -> None:
    config = create_test_config()
    res1 = MockResponse(200, {"ok": True, "result": {"message_id": 108}}, None)
    res2 = MockResponse(200, {"ok": True, "result": {"message_id": 109}}, None)
    session = MockSession([res1, res2])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    promotable_report = create_sample_tournament_report(True)
    success1 = await adapter.send_tournament_verdict(promotable_report)
    assert success1
    assert adapter.challenger_pending
    req1 = session.recorded_requests[0]
    assert "WEEKEND MODEL TOURNAMENT VERDICT" in req1["json"]["text"]
    assert "YES (Ready to Promote)" in req1["json"]["text"]
    assert req1["json"]["reply_markup"] is not None
    assert req1["json"]["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == "approve_model"

    non_promotable_report = create_sample_tournament_report(False)
    success2 = await adapter.send_tournament_verdict(non_promotable_report)
    assert success2
    req2 = session.recorded_requests[1]
    assert "NO (Retain Champion)" in req2["json"]["text"]
    assert "reply_markup" not in req2["json"]


# =============================================================================
# 3. get_updates and answer_callback_query Tests
# =============================================================================


@pytest.mark.asyncio
async def test_get_updates_success_and_error() -> None:
    config = create_test_config()
    updates_payload = {
        "ok": True,
        "result": [
            {
                "update_id": 1001,
                "message": {
                    "message_id": 50,
                    "chat": {"id": 5577408890},
                    "text": "/status",
                    "date": 1700000000,
                },
            },
            {
                "update_id": 1002,
                "callback_query": {
                    "id": "cb_99",
                    "from": {"id": 5577408890},
                    "message": {
                        "message_id": 51,
                        "chat": {"id": 5577408890},
                    },
                    "data": "confirm_mode:live",
                },
            },
        ],
    }
    res1 = MockResponse(200, updates_payload, None)
    res2 = MockResponse(500, {}, None)
    res3 = MockResponse(200, {}, ConnectionError("timeout"))
    session = MockSession([res1, res2, res3])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    updates = await adapter.get_updates(0, 30)
    assert len(updates) == 2
    assert updates[0].update_id == 1001
    assert updates[0].message is not None
    assert updates[0].message.text == "/status"
    assert updates[1].update_id == 1002
    assert updates[1].callback_query is not None
    assert updates[1].callback_query.data == "confirm_mode:live"

    # Server error
    err_updates = await adapter.get_updates(1003, 30)
    assert len(err_updates) == 0

    # Network error
    net_updates = await adapter.get_updates(1003, 30)
    assert len(net_updates) == 0


@pytest.mark.asyncio
async def test_answer_callback_query() -> None:
    config = create_test_config()
    res1 = MockResponse(200, {"ok": True, "result": True}, None)
    res2 = MockResponse(500, {"ok": False}, None)
    res3 = MockResponse(200, {}, ConnectionError("Network failure"))
    session = MockSession([res1, res2, res3])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    assert await adapter.answer_callback_query("cb_1", "Done")
    assert not await adapter.answer_callback_query("cb_2", "Fail")
    assert not await adapter.answer_callback_query("cb_3", "Error")


# =============================================================================
# 4. Security Whitelist Tests
# =============================================================================


@pytest.mark.asyncio
async def test_whitelist_security_blocks_unauthorized_user() -> None:
    config = create_test_config()  # authorized is "5577408890"
    session = MockSession([])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    # Direct call to handle_command with wrong chat_id raises TelegramSecurityViolationException
    req = TelegramCommandRequest.parse("/status", "999999", "999999")
    with pytest.raises(TelegramSecurityViolationException):
        await adapter.handle_command(req)

    # Dispatched update from wrong chat_id is silently dropped by _dispatch_update
    raw_unauthorized = {
        "update_id": 2001,
        "message": {
            "message_id": 99,
            "chat": {"id": 999999},
            "text": "/panic_close_all",
            "date": 1700000000,
        },
    }
    unauthorized_upd = TelegramUpdate.from_dict(raw_unauthorized)
    await adapter.dispatch_update(unauthorized_upd)
    # No message sent to telegram
    assert len(session.recorded_requests) == 0

    # Dispatched callback from wrong chat_id is also dropped
    raw_unauthorized_cb = {
        "update_id": 2002,
        "callback_query": {
            "id": "cb_fake",
            "from": {"id": 999999},
            "message": {"message_id": 99, "chat": {"id": 999999}},
            "data": "confirm_panic_close",
        },
    }
    unauthorized_cb_upd = TelegramUpdate.from_dict(raw_unauthorized_cb)
    await adapter.dispatch_update(unauthorized_cb_upd)
    assert len(session.recorded_requests) == 0


# =============================================================================
# 5. Interactive Slash Command Routing Tests
# =============================================================================


@pytest.mark.asyncio
async def test_cmd_status_without_and_with_context() -> None:
    config = create_test_config()
    session = MockSession([])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    req = TelegramCommandRequest.parse("/status", "5577408890", "5577408890")
    res1 = await adapter.handle_command(req)
    assert "HỆ THỐNG GIAO DỊCH - TRẠNG THÁI" in res1
    assert "*Execution Mode:* `PAPER`" in res1
    assert "ONLINE / HEALTHY" in res1

    # Now with AccountPortfolioContext
    ctx = create_sample_account_context()
    adapter.set_account_context(ctx)
    res2 = await adapter.handle_command(req)
    assert "Portfolio NAV:* `$10,150.00`" in res2
    assert "Daily PnL:* 📈 `+$150.00`" in res2
    assert "Drawdown:* `0.50%`" in res2
    assert "Open Positions:* `1`" in res2


@pytest.mark.asyncio
async def test_cmd_positions_empty_and_active() -> None:
    config = create_test_config()
    session = MockSession([])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    req = TelegramCommandRequest.parse("/positions", "5577408890", "5577408890")
    # Empty positions
    res1 = await adapter.handle_command(req)
    assert "Hiện không có vị thế nào đang mở" in res1

    # Active positions
    ctx = create_sample_account_context()
    adapter.set_account_context(ctx)
    res2 = await adapter.handle_command(req)
    assert "DANH SÁCH VỊ THẾ ĐANG MỞ" in res2
    assert "BTCUSDT" in res2
    # Verify inline buttons attached
    last_req = session.recorded_requests[-1]
    assert "reply_markup" in last_req["json"]
    buttons = last_req["json"]["reply_markup"]["inline_keyboard"][0]
    assert buttons[0]["callback_data"] == "be:pos_btc_01"
    assert buttons[1]["callback_data"] == "close:pos_btc_01"


@pytest.mark.asyncio
async def test_cmd_shadow_stats() -> None:
    config = create_test_config()
    session = MockSession([])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    adapter.set_shadow_stats(
        {
            "evaluated_count": 45,
            "vetoed_count": 12,
            "defense_rate_pct": 88.5,
            "avoided_loss_usd": 1250.75,
        }
    )
    req = TelegramCommandRequest.parse("/shadow", "5577408890", "5577408890")
    res = await adapter.handle_command(req)
    assert "RISK ARBITER - SHADOW STATS" in res
    assert "Evaluated:* `45`" in res
    assert "Vetoed / Filtered Orders:* `12`" in res
    assert "Defense Rate:* `88.5%`" in res
    assert "Avoided Negative PnL:* `$1,250.75`" in res


@pytest.mark.asyncio
async def test_cmd_quota_stats() -> None:
    config = create_test_config()
    session = MockSession([])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    adapter.set_quota_stats(
        {
            "fast_rpm": 12,
            "fast_rpm_limit": 60,
            "deep_rpm": 4,
            "deep_rpm_limit": 20,
            "judge_rpm": 2,
            "judge_rpm_limit": 10,
            "rpd_count": 140,
        }
    )
    req = TelegramCommandRequest.parse("/quota", "5577408890", "5577408890")
    res = await adapter.handle_command(req)
    assert "9ROUTER API GATEWAY - QUOTA STATS" in res
    assert "Fast Tier:* `12/60 RPM`" in res
    assert "Daily Total Requests (RPD):* `140`" in res


@pytest.mark.asyncio
async def test_cmd_mode_switching_and_confirmation() -> None:
    config = create_test_config()
    session = MockSession([])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    mode_hook = MagicMock()

    async def async_mode_hook(mode: str) -> str:
        mode_hook(mode)
        return "hook_called"

    adapter.set_on_mode_change(async_mode_hook)

    # 1. /mode with no arguments returns usage
    req_empty = TelegramCommandRequest.parse("/mode", "5577408890", "5577408890")
    res_empty = await adapter.handle_command(req_empty)
    assert "Cú pháp đổi chế độ" in res_empty

    # 2. /mode testnet directly switches mode
    req_testnet = TelegramCommandRequest.parse("/mode testnet", "5577408890", "5577408890")
    res_testnet = await adapter.handle_command(req_testnet)
    assert "Đã chuyển execution mode sang: `TESTNET`" in res_testnet
    assert adapter.execution_mode == "TESTNET"
    mode_hook.assert_called_with("TESTNET")

    # 3. /mode invalid returns error
    req_inv = TelegramCommandRequest.parse("/mode invalid_mode", "5577408890", "5577408890")
    res_inv = await adapter.handle_command(req_inv)
    assert "Chế độ không hợp lệ" in res_inv

    # 4. /mode live requires 2-step confirmation (does NOT change mode immediately)
    req_live = TelegramCommandRequest.parse("/mode live", "5577408890", "5577408890")
    res_live = await adapter.handle_command(req_live)
    assert "XÁC NHẬN CHUYỂN SANG LIVE MODE" in res_live
    assert adapter.execution_mode == "TESTNET"  # Remains TESTNET until confirmed
    last_req = session.recorded_requests[-1]
    assert (
        last_req["json"]["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
        == "confirm_mode:live"
    )


@pytest.mark.asyncio
async def test_cmd_pause_and_resume() -> None:
    config = create_test_config()
    session = MockSession([])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    pause_hook = MagicMock()
    resume_hook = MagicMock()

    async def async_pause(hours: float | None) -> str:
        pause_hook(hours)
        return "paused"

    async def async_resume() -> str:
        resume_hook()
        return "resumed"

    adapter.set_on_pause(async_pause)
    adapter.set_on_resume(async_resume)

    # Pause with hours
    req_pause = TelegramCommandRequest.parse("/pause 4", "5577408890", "5577408890")
    res_pause = await adapter.handle_command(req_pause)
    assert adapter.is_paused
    assert "trong `4.0` giờ" in res_pause
    pause_hook.assert_called_with(4.0)

    # Resume
    req_resume = TelegramCommandRequest.parse("/resume", "5577408890", "5577408890")
    res_resume = await adapter.handle_command(req_resume)
    assert not adapter.is_paused
    assert "Đã tiếp tục giao dịch" in res_resume
    resume_hook.assert_called_once()


@pytest.mark.asyncio
async def test_cmd_panic_close_all_requires_confirmation() -> None:
    config = create_test_config()
    session = MockSession([])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    req = TelegramCommandRequest.parse("/panic_close_all", "5577408890", "5577408890")
    res = await adapter.handle_command(req)
    assert "CẢNH BÁO KHẨN CẤP: PANIC CLOSE ALL" in res
    last_req = session.recorded_requests[-1]
    buttons = last_req["json"]["reply_markup"]["inline_keyboard"][0]
    assert buttons[0]["callback_data"] == "confirm_panic_close"
    assert buttons[1]["callback_data"] == "cancel_action"


@pytest.mark.asyncio
async def test_cmd_approve_model() -> None:
    config = create_test_config()
    session = MockSession([])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    approve_hook = MagicMock()

    async def async_approve() -> str:
        approve_hook()
        return "approved"

    adapter.set_on_approve_model(async_approve)

    # When not pending
    req = TelegramCommandRequest.parse("/approve_model", "5577408890", "5577408890")
    res1 = await adapter.handle_command(req)
    assert "không có Challenger model nào đang chờ" in res1
    approve_hook.assert_not_called()

    # When pending, displays 2-step prompt with confirm and cancel buttons
    adapter.set_challenger_pending(True)
    res2 = await adapter.handle_command(req)
    assert "XÁC NHẬN THĂNG HẠNG CHALLENGER MODEL" in res2
    assert any("confirm_approve_model" in str(r) for r in session.recorded_requests)
    approve_hook.assert_not_called()

    # Step 2: User confirms promotion
    ans = await adapter.handle_callback("cb_confirm", "confirm_approve_model")
    assert "Đã thăng hạng Challenger thành công" in ans
    assert not adapter.challenger_pending
    approve_hook.assert_called_once()


@pytest.mark.asyncio
async def test_cmd_retrain() -> None:
    config = create_test_config()
    session = MockSession([])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    retrain_hook = MagicMock()

    async def async_retrain(mode: str) -> str:
        retrain_hook(mode)
        return f"retrained_{mode}"

    adapter.set_on_retrain(async_retrain)

    # 1. /retrain with default (fast)
    req_fast = TelegramCommandRequest.parse("/retrain", "5577408890", "5577408890")
    res_fast = await adapter.handle_command(req_fast)
    assert "FAST" in res_fast
    retrain_hook.assert_called_with("fast")

    # 2. /retrain full
    req_full = TelegramCommandRequest.parse("/retrain full", "5577408890", "5577408890")
    res_full = await adapter.handle_command(req_full)
    assert "FULL" in res_full
    retrain_hook.assert_called_with("full")


@pytest.mark.asyncio
async def test_custom_command_registration() -> None:
    config = create_test_config()
    session = MockSession([])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    async def custom_ping(req: TelegramCommandRequest) -> str:
        return f"PONG {req.args[0] if len(req.args) > 0 else 'DEFAULT'}"

    adapter.register_custom_command("ping", custom_ping)

    req = TelegramCommandRequest.parse("/ping 123", "5577408890", "5577408890")
    res = await adapter.handle_command(req)
    assert res == "PONG 123"


@pytest.mark.asyncio
async def test_unknown_command_returns_help() -> None:
    config = create_test_config()
    session = MockSession([])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    req = TelegramCommandRequest.parse("/unknown_command", "5577408890", "5577408890")
    res = await adapter.handle_command(req)
    assert "Lệnh không xác định: `/unknown_command`" in res
    assert "Danh sách lệnh khả dụng" in res


# =============================================================================
# 6. Inline Keyboard Callback Queries & 2-Step Execution Tests
# =============================================================================


@pytest.mark.asyncio
async def test_callback_confirm_mode_live() -> None:
    config = create_test_config()
    session = MockSession([])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    mode_hook = MagicMock()

    async def async_mode_change(m: str) -> str:
        mode_hook(m)
        return "ok"

    adapter.set_on_mode_change(async_mode_change)

    ans = await adapter.handle_callback("cb_1", "confirm_mode:live")
    assert "Chế độ LIVE đã được kích hoạt" in ans
    assert adapter.execution_mode == "LIVE"
    mode_hook.assert_called_with("LIVE")
    # Verify notification sent to Telegram
    assert any(
        "EXECUTION MODE ĐÃ CHUYỂN SANG LIVE" in r["json"]["text"] for r in session.recorded_requests
    )


@pytest.mark.asyncio
async def test_callback_confirm_panic_close() -> None:
    config = create_test_config()
    session = MockSession([])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    panic_hook = MagicMock()

    async def async_panic() -> str:
        panic_hook()
        return "panicked"

    adapter.set_on_panic_close(async_panic)

    ans = await adapter.handle_callback("cb_2", "confirm_panic_close")
    assert "Đang thực thi Panic Close All" in ans
    panic_hook.assert_called_once()
    assert any(
        "PANIC CLOSE ALL ĐÃ ĐƯỢC KÍCH HOẠT" in r["json"]["text"] for r in session.recorded_requests
    )


@pytest.mark.asyncio
async def test_callback_cancel_action() -> None:
    config = create_test_config()
    session = MockSession([])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    ans = await adapter.handle_callback("cb_3", "cancel_action")
    assert "Thao tác đã được hủy bỏ" in ans
    assert any("hủy bỏ an toàn" in r["json"]["text"] for r in session.recorded_requests)


@pytest.mark.asyncio
async def test_callback_breakeven_and_close_position() -> None:
    config = create_test_config()
    session = MockSession([])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    be_hook = MagicMock()
    close_hook = MagicMock()

    async def async_be(pos_id: str) -> str:
        be_hook(pos_id)
        return "be_done"

    async def async_close(pos_id: str) -> str:
        close_hook(pos_id)
        return "close_done"

    adapter.set_on_breakeven(async_be)
    adapter.set_on_close_position(async_close)

    ans_be = await adapter.handle_callback("cb_4", "be:pos_btc_01")
    assert "Đã dời SL về BE cho vị thế pos_btc_01" in ans_be
    be_hook.assert_called_with("pos_btc_01")

    ans_close = await adapter.handle_callback("cb_5", "close:pos_btc_01")
    assert "Đã gửi lệnh đóng vị thế pos_btc_01" in ans_close
    close_hook.assert_called_with("pos_btc_01")


@pytest.mark.asyncio
async def test_callback_approve_model() -> None:
    config = create_test_config()
    session = MockSession([])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    approve_hook = MagicMock()

    async def async_approve() -> str:
        approve_hook()
        return "model_promoted"

    adapter.set_on_approve_model(async_approve)
    adapter.set_challenger_pending(True)

    # 1. Clicking inline approve_model displays 2-step confirmation prompt
    ans_prompt = await adapter.handle_callback("cb_6", "approve_model")
    assert "XÁC NHẬN THĂNG HẠNG CHALLENGER MODEL" in ans_prompt
    approve_hook.assert_not_called()

    # 2. Clicking cancel_approve_model cancels
    ans_cancel = await adapter.handle_callback("cb_cancel", "cancel_approve_model")
    assert "hủy bỏ" in ans_cancel
    approve_hook.assert_not_called()

    # 3. Clicking confirm_approve_model executes promotion
    ans_confirm = await adapter.handle_callback("cb_confirm", "confirm_approve_model")
    assert "Đã thăng hạng Challenger thành công" in ans_confirm
    assert not adapter.challenger_pending
    approve_hook.assert_called_once()


@pytest.mark.asyncio
async def test_callback_unknown() -> None:
    config = create_test_config()
    session = MockSession([])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    ans = await adapter.handle_callback("cb_7", "unknown_callback_data")
    assert "không xác định" in ans


# =============================================================================
# 7. Long-Polling Background Task & poll_once Tests
# =============================================================================


@pytest.mark.asyncio
async def test_poll_once_advances_offset_and_dispatches() -> None:
    config = create_test_config()
    updates_payload = {
        "ok": True,
        "result": [
            {
                "update_id": 5001,
                "message": {
                    "message_id": 10,
                    "chat": {"id": 5577408890},
                    "text": "/status",
                    "date": 1700000000,
                },
            }
        ],
    }
    res = MockResponse(200, updates_payload, None)
    session = MockSession([res])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    processed_updates = await adapter.poll_once()
    assert len(processed_updates) == 1
    assert adapter.last_update_id == 5001
    # Verify response to /status was dispatched
    assert any(
        "text" in r["json"] and "HỆ THỐNG GIAO DỊCH" in r["json"]["text"]
        for r in session.recorded_requests
    )


@pytest.mark.asyncio
async def test_start_and_stop_polling() -> None:
    config = create_test_config()
    res = MockResponse(200, {"ok": True, "result": []}, None)
    session = MockSession([res])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    await adapter.start_polling()
    assert adapter.is_polling

    # Starting again should be a no-op
    await adapter.start_polling()
    assert adapter.is_polling

    # Yield execution briefly
    await asyncio.sleep(0.01)

    await adapter.stop_polling()
    assert not adapter.is_polling


@pytest.mark.asyncio
async def test_send_order_placed_alert() -> None:
    config = create_test_config()
    session = MockSession([])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    success = await adapter.send_order_placed_alert(
        symbol="BTCUSDT",
        side="BUY",
        order_type="LIMIT_PULLBACK_15M",
        entry_price=65000.0,
        stop_loss_price=64000.0,
        take_profit_price=67000.0,
        quantity=0.1,
        risk_reward_ratio=2.0,
        order_id="ord_test_01",
    )
    assert success is True
    assert len(session.recorded_requests) == 1
    req = session.recorded_requests[0]
    assert "LỆNH ĐÃ ĐƯỢC ĐẶT TỰ ĐỘNG" in req["json"]["text"]
    assert "BTCUSDT" in req["json"]["text"]
    assert "65,000.00" in req["json"]["text"]
    # Check attached inline keyboard for BE and Close
    buttons = req["json"]["reply_markup"]["inline_keyboard"][0]
    assert buttons[0]["callback_data"] == "be:BTCUSDT"
    assert buttons[1]["callback_data"] == "close:BTCUSDT"


@pytest.mark.asyncio
async def test_send_position_closed_alert() -> None:
    config = create_test_config()
    session = MockSession([])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    # Win alert
    win_success = await adapter.send_position_closed_alert(
        symbol="BTCUSDT",
        side="LONG",
        close_price=67000.0,
        pnl_usd=200.0,
        pnl_pct=2.0,
        r_multiple=2.0,
        exit_reason="HIT_TP | Target hit",
    )
    assert win_success is True
    req_win = session.recorded_requests[-1]
    assert "THẮNG (WIN)" in req_win["json"]["text"]
    assert "+$200.00" in req_win["json"]["text"]

    # Loss alert
    loss_success = await adapter.send_position_closed_alert(
        symbol="ETHUSDT",
        side="SHORT",
        close_price=3500.0,
        pnl_usd=-100.0,
        pnl_pct=-1.5,
        r_multiple=-1.0,
        exit_reason="STOPPED_OUT | Stop hit",
    )
    assert loss_success is True
    req_loss = session.recorded_requests[-1]
    assert "THUA (LOSS)" in req_loss["json"]["text"]
    assert "-$100.00" in req_loss["json"]["text"]


@pytest.mark.asyncio
async def test_send_error_alert() -> None:
    config = create_test_config()
    session = MockSession([])
    adapter = TelegramBotAdapter(config, session, 30)  # pyright: ignore[reportArgumentType]

    # Warning
    w_success = await adapter.send_error_alert(
        component="BinanceGateway",
        error_message="Drift detected -1021",
        is_fatal=False,
    )
    assert w_success is True
    assert "WARNING" in session.recorded_requests[-1]["json"]["text"]

    # Fatal
    f_success = await adapter.send_error_alert(
        component="MainDaemon",
        error_message="Database pool exhausted",
        is_fatal=True,
    )
    assert f_success is True
    assert "FATAL ERROR" in session.recorded_requests[-1]["json"]["text"]
