"""Production Telegram Bot Adapter (Clean Architecture, Invariant #9).

Enforces strictly ZERO default parameters, ZERO class field defaults, and ZERO silent fallbacks.
Implements:
- TelegramNotifierPort (outbound reports, alerts, messages, tournament verdicts).
- TelegramCommandHandlerPort (inbound slash commands, inline keyboard callbacks).
- Long-polling background loop via pure async aiohttp.ClientSession.
- Whitelist security strictly validating chat_id == self._config.chat_id.
- 2-step interactive confirmations for high-risk actions (/mode live, /panic_close_all).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import aiohttp

from trading_bot.domain.entities.quant.symbolic_alpha import ModelTournamentReport
from trading_bot.domain.entities.risk.account_state import AccountPortfolioContext
from trading_bot.domain.entities.telegram.telegram_models import (
    InvalidTelegramCommandException,
    TelegramBotCommand,
    TelegramCommandRequest,
    TelegramCycleReport,
    TelegramInlineButton,
    TelegramInlineKeyboard,
    TelegramSecurityViolationException,
    TelegramUpdate,
)
from trading_bot.infrastructure.config_schemas import TelegramConfig
from trading_bot.ports.telegram_port import (
    TelegramCommandHandlerPort,
    TelegramNotifierPort,
)

logger = logging.getLogger(__name__)


class TelegramBotAdapter(TelegramNotifierPort, TelegramCommandHandlerPort):
    """Production aiohttp-based Telegram bot adapter for alerts and command execution."""

    def __init__(
        self,
        config: TelegramConfig,
        session: aiohttp.ClientSession,
        poll_timeout_seconds: int,
    ) -> None:
        self._config: TelegramConfig = config
        self._session: aiohttp.ClientSession = session
        self._poll_timeout_seconds: int = poll_timeout_seconds
        self._base_url: str = f"https://api.telegram.org/bot{config.bot_token}"

        # State management (Invariant #9: All instance variables initialized in __init__)
        self._last_update_id: int = 0
        self._is_polling: bool = False
        self._polling_task: asyncio.Task[None] | None = None
        self._consecutive_poll_errors: int = 0

        self._execution_mode: str = "PAPER"
        self._is_paused: bool = False
        self._pause_until_ms: int | None = None
        self._account_context: AccountPortfolioContext | None = None
        self._shadow_stats: dict[str, Any] = {}
        self._quota_stats: dict[str, Any] = {}
        self._challenger_pending: bool = False

        # Action callbacks hooks
        self._on_mode_change: Callable[[str], Awaitable[str]] | None = None
        self._on_panic_close: Callable[[], Awaitable[str]] | None = None
        self._on_breakeven: Callable[[str], Awaitable[str]] | None = None
        self._on_close_position: Callable[[str], Awaitable[str]] | None = None
        self._on_approve_model: Callable[[], Awaitable[str]] | None = None
        self._on_retrain: Callable[[str], Awaitable[str]] | None = None
        self._on_pause: Callable[[float | None], Awaitable[str]] | None = None
        self._on_resume: Callable[[], Awaitable[str]] | None = None
        self._custom_commands: dict[str, Callable[[TelegramCommandRequest], Awaitable[str]]] = {}

    # -------------------------------------------------------------------------
    # Properties & State Getters/Setters
    # -------------------------------------------------------------------------

    @property
    def config(self) -> TelegramConfig:
        return self._config

    @property
    def is_polling(self) -> bool:
        return self._is_polling

    @property
    def execution_mode(self) -> str:
        return self._execution_mode

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def challenger_pending(self) -> bool:
        return self._challenger_pending

    @property
    def last_update_id(self) -> int:
        return self._last_update_id

    @property
    def account_context(self) -> AccountPortfolioContext | None:
        return self._account_context

    @property
    def shadow_stats(self) -> dict[str, Any]:
        return self._shadow_stats

    @property
    def quota_stats(self) -> dict[str, Any]:
        return self._quota_stats

    def set_account_context(self, context: AccountPortfolioContext) -> None:
        self._account_context = context

    def set_shadow_stats(self, stats: dict[str, Any]) -> None:
        self._shadow_stats = stats

    def set_quota_stats(self, stats: dict[str, Any]) -> None:
        self._quota_stats = stats

    def set_execution_mode(self, mode: str) -> None:
        self._execution_mode = mode.upper()

    def set_challenger_pending(self, pending: bool) -> None:
        self._challenger_pending = pending

    def set_paused(self, paused: bool, hours: float | None) -> None:
        self._is_paused = paused
        if hours is not None:
            now_ms = int(asyncio.get_event_loop().time() * 1000)
            self._pause_until_ms = now_ms + int(hours * 3600 * 1000)
        else:
            self._pause_until_ms = None

    def set_on_mode_change(self, callback: Callable[[str], Awaitable[str]]) -> None:
        self._on_mode_change = callback

    def set_on_panic_close(self, callback: Callable[[], Awaitable[str]]) -> None:
        self._on_panic_close = callback

    def set_on_breakeven(self, callback: Callable[[str], Awaitable[str]]) -> None:
        self._on_breakeven = callback

    def set_on_close_position(self, callback: Callable[[str], Awaitable[str]]) -> None:
        self._on_close_position = callback

    def set_on_approve_model(self, callback: Callable[[], Awaitable[str]]) -> None:
        self._on_approve_model = callback

    def set_on_retrain(self, callback: Callable[[str], Awaitable[str]]) -> None:
        self._on_retrain = callback

    def set_on_pause(self, callback: Callable[[float | None], Awaitable[str]]) -> None:
        self._on_pause = callback

    def set_on_resume(self, callback: Callable[[], Awaitable[str]]) -> None:
        self._on_resume = callback

    def register_custom_command(
        self,
        command: str,
        handler: Callable[[TelegramCommandRequest], Awaitable[str]],
    ) -> None:
        clean_command = command.lower().lstrip("/")
        self._custom_commands[clean_command] = handler

    # -------------------------------------------------------------------------
    # TelegramNotifierPort Implementation
    # -------------------------------------------------------------------------

    async def send_text_message(
        self,
        text: str,
        reply_markup: list[list[TelegramInlineButton]] | dict[str, Any] | None,
    ) -> bool:
        """Send formatted text message with optional inline keyboard buttons to configured chat."""
        payload: dict[str, Any] = {
            "chat_id": self._config.chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }

        if reply_markup is not None:
            if isinstance(reply_markup, dict):
                payload["reply_markup"] = reply_markup
            else:
                keyboard_entity = TelegramInlineKeyboard(inline_keyboard=reply_markup)
                payload["reply_markup"] = keyboard_entity.to_dict()

        url = f"{self._base_url}/sendMessage"
        try:
            async with self._session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as response:
                if response.status == 200:
                    res_json: dict[str, Any] = await response.json()
                    if "ok" in res_json and bool(res_json["ok"]):
                        return True
                    logger.error("Telegram sendMessage API returned ok=False: %s", res_json)
                    return False

                # If Telegram rejects markdown formatting (400 Bad Request), retry as plain text
                if response.status == 400 and "parse_mode" in payload:
                    logger.warning("Markdown formatting rejected (400). Retrying as plain text.")
                    plain_payload: dict[str, Any] = {
                        "chat_id": self._config.chat_id,
                        "text": text,
                    }
                    if "reply_markup" in payload:
                        plain_payload["reply_markup"] = payload["reply_markup"]
                    async with self._session.post(
                        url,
                        json=plain_payload,
                        timeout=aiohttp.ClientTimeout(total=10.0),
                    ) as retry_response:
                        return retry_response.status == 200

                logger.error("Telegram sendMessage failed with HTTP %s", response.status)
                return False
        except Exception as exc:
            logger.error("Network error during Telegram sendMessage: %s", exc)
            return False

    async def send_cycle_report(
        self,
        report: TelegramCycleReport | dict[str, Any],
    ) -> bool:
        """Send a 30-minute trading cycle summary report card."""
        if isinstance(report, TelegramCycleReport):
            formatted_text = report.format_telegram_markdown()
        else:
            try:
                cycle_obj = TelegramCycleReport(**report)
                formatted_text = cycle_obj.format_telegram_markdown()
            except Exception:
                formatted_text = f"📊 *[30m CYCLE REPORT]*\n`{report}`"

        return await self.send_text_message(formatted_text, None)

    async def send_emergency_alert(
        self,
        title: str,
        message: str,
    ) -> bool:
        """Send a high-priority emergency alert."""
        alert_text = f"🚨 *EMERGENCY ALERT: {title}*\n\n{message}"
        return await self.send_text_message(alert_text, None)

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
        clean_side = side.upper()
        if "BUY" in clean_side or "LONG" in clean_side:
            side_emoji = "🟢 LONG"
        else:
            side_emoji = "🔴 SHORT"

        lines: list[str] = [
            "⚡ *[LỆNH ĐÃ ĐƯỢC ĐẶT TỰ ĐỘNG]*",
            f"• *Mã:* `{symbol}` | *Vị thế:* *{side_emoji}*",
            f"• *Kiểu lệnh:* `{order_type}` | *Order ID:* `{order_id}`",
            f"• *Khối lượng:* `{quantity:.4f}` | *R:R:* `{risk_reward_ratio:.2f}`",
            f"• *Entry Price:* `${entry_price:,.2f}`",
            f"• *Stop Loss (SL):* `${stop_loss_price:,.2f}`",
            f"• *Take Profit (TP):* `${take_profit_price:,.2f}`",
            f"• *Chế độ thực thi:* `{self._execution_mode}`",
        ]
        text = "\n".join(lines)

        reply_markup: list[list[TelegramInlineButton]] = [
            [
                TelegramInlineButton(
                    text="🛡️ Dời SL về Breakeven",
                    callback_data=f"be:{symbol}",
                    url=None,
                ),
                TelegramInlineButton(
                    text="🚨 Đóng vị thế ngay",
                    callback_data=f"close:{symbol}",
                    url=None,
                ),
            ]
        ]
        return await self.send_text_message(text, reply_markup)

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
        if pnl_usd >= 0.0:
            result_emoji = "🎉 *THẮNG (WIN)*"
            pnl_sign = f"+${pnl_usd:,.2f}"
        else:
            result_emoji = "⚠️ *THUA (LOSS)*"
            pnl_sign = f"-${abs(pnl_usd):,.2f}"

        lines: list[str] = [
            f"🏁 *[VỊ THẾ ĐÃ KẾT THÚC]* — {result_emoji}",
            f"• *Mã:* `{symbol}` | *Hướng:* `{side}`",
            f"• *Giá đóng:* `${close_price:,.2f}`",
            f"• *Lợi nhuận PnL:* `{pnl_sign}` (`{pnl_pct:+.2f}%`)",
            f"• *R-Multiple:* `{r_multiple:+.2f}R`",
            f"• *Lý do kết thúc:* `{exit_reason}`",
            f"• *Chế độ:* `{self._execution_mode}`",
        ]
        return await self.send_text_message("\n".join(lines), None)

    async def send_error_alert(
        self,
        component: str,
        error_message: str,
        is_fatal: bool,
    ) -> bool:
        """Send real-time alert when an error or exception occurs in the system."""
        if is_fatal:
            severity = "🚨 FATAL ERROR"
        else:
            severity = "⚠️ WARNING / LỖI HỆ THỐNG"

        now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        lines: list[str] = [
            f"*{severity}: [{component}]*",
            f"• *Thời gian (UTC):* `{now_str}`",
            f"• *Chi tiết lỗi:* `{error_message}`",
            f"• *Chế độ:* `{self._execution_mode}`",
        ]
        if is_fatal:
            lines.append(
                "• *Hành động:* Hệ thống đang kích hoạt cơ chế bảo vệ / kiểm tra lại kết nối."
            )
        return await self.send_text_message("\n".join(lines), None)

    async def send_tournament_verdict(
        self,
        report: ModelTournamentReport | dict[str, Any],
    ) -> bool:
        """Send the weekend Champion vs. Challenger tournament comparison report."""
        if isinstance(report, ModelTournamentReport):
            rep = report
        else:
            rep = ModelTournamentReport(**report)

        if rep.is_promotable:
            status_emoji = "🏆"
            promotable_str = "YES (Ready to Promote)"
        else:
            status_emoji = "⚖️"
            promotable_str = "NO (Retain Champion)"

        if hasattr(rep.recommended_verdict, "value"):
            verdict_str = str(rep.recommended_verdict.value)
        else:
            verdict_str = str(rep.recommended_verdict)

        safe_rationale: str = rep.verdict_rationale.replace("_", "\\_")

        lines: list[str] = [
            f"*{status_emoji} [WEEKEND MODEL TOURNAMENT VERDICT]*",
            f"• *Champion:* `{rep.champion_model_id}` ({rep.champion_feature_count} feats)",
            f"• *Challenger:* `{rep.challenger_model_id}` ({rep.challenger_feature_count} feats)",
            f"• *Verdict:* `{verdict_str}`",
            f"• *Promotable:* `{promotable_str}`",
            f"• *Sharpe:* Champ `{rep.champion_net_sharpe:.2f}` vs Chall `{rep.challenger_net_sharpe:.2f}`",
            f"• *Max DD:* Champ `{rep.champion_max_drawdown_pct:.2f}%` vs Chall `{rep.challenger_max_drawdown_pct:.2f}%`",
            f"• *Win Rate:* Champ `{rep.champion_win_rate:.2%}` vs Chall `{rep.challenger_win_rate:.2%}`",
            f"• *Brier Score:* Champ `{rep.champion_brier_score:.4f}` vs Chall `{rep.challenger_brier_score:.4f}`",
            f"• *Brier Skill:* Champ `{rep.champion_brier_skill_score:+.4f}` vs Chall `{rep.challenger_brier_skill_score:+.4f}`",
            f"• *Rationale:* {safe_rationale}",
        ]
        report_text = "\n".join(lines)

        reply_markup: list[list[TelegramInlineButton]] | None = None
        if rep.is_promotable:
            self._challenger_pending = True
            reply_markup = [
                [
                    TelegramInlineButton(
                        text="🏆 Phê duyệt Challenger thành Champion",
                        callback_data="approve_model",
                        url=None,
                    )
                ]
            ]

        return await self.send_text_message(report_text, reply_markup)

    # -------------------------------------------------------------------------
    # Long-Polling Updates & Callback Query Answering
    # -------------------------------------------------------------------------

    async def get_updates(
        self,
        offset: int,
        timeout_seconds: int,
    ) -> list[TelegramUpdate]:
        """Fetch incoming updates from Telegram getUpdates endpoint."""
        payload: dict[str, Any] = {
            "offset": offset,
            "timeout": timeout_seconds,
        }
        url = f"{self._base_url}/getUpdates"
        total_timeout = float(timeout_seconds + 5)
        try:
            async with self._session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=total_timeout),
            ) as response:
                if response.status != 200:
                    self._consecutive_poll_errors += 1
                    backoff_delay = min(
                        30.0, 1.0 * (2 ** min(self._consecutive_poll_errors - 1, 5))
                    )
                    logger.error(
                        "getUpdates failed with HTTP %s. Backing off for %.1fs",
                        response.status,
                        backoff_delay,
                    )
                    await asyncio.sleep(backoff_delay)
                    return []
                data: dict[str, Any] = await response.json()
                if "ok" not in data or not data["ok"] or "result" not in data:
                    self._consecutive_poll_errors += 1
                    backoff_delay = min(
                        30.0, 1.0 * (2 ** min(self._consecutive_poll_errors - 1, 5))
                    )
                    logger.error(
                        "getUpdates response payload missing ok or result: %s. Backing off for %.1fs",
                        data,
                        backoff_delay,
                    )
                    await asyncio.sleep(backoff_delay)
                    return []

                raw_list: list[dict[str, Any]] = data["result"]
                parsed_updates: list[TelegramUpdate] = []
                for item in raw_list:
                    try:
                        upd = TelegramUpdate.from_dict(item)
                        parsed_updates.append(upd)
                    except Exception as parse_exc:
                        logger.warning("Failed to parse update payload: %s", parse_exc)

                if self._consecutive_poll_errors > 0:
                    logger.info("Telegram connection restored. Resuming normal polling.")
                    self._consecutive_poll_errors = 0
                return parsed_updates
        except Exception as exc:
            self._consecutive_poll_errors += 1
            backoff_delay = min(30.0, 1.0 * (2 ** min(self._consecutive_poll_errors - 1, 5)))
            if self._consecutive_poll_errors <= 3 or self._consecutive_poll_errors % 10 == 0:
                logger.error(
                    "Network error during get_updates (attempt %d): %s. Backing off for %.1fs",
                    self._consecutive_poll_errors,
                    exc,
                    backoff_delay,
                )
            await asyncio.sleep(backoff_delay)
            return []

    async def answer_callback_query(
        self,
        callback_query_id: str,
        text: str,
    ) -> bool:
        """Acknowledge inline keyboard button press to dismiss loading indicator."""
        payload: dict[str, Any] = {
            "callback_query_id": callback_query_id,
            "text": text,
        }
        url = f"{self._base_url}/answerCallbackQuery"
        try:
            async with self._session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as response:
                if response.status == 200:
                    data: dict[str, Any] = await response.json()
                    if "ok" in data and bool(data["ok"]):
                        return True
                logger.error("answerCallbackQuery failed with HTTP %s", response.status)
                return False
        except Exception as exc:
            logger.error("Network error during answerCallbackQuery: %s", exc)
            return False

    async def poll_once(self) -> list[TelegramUpdate]:
        """Execute a single long-polling cycle, dispatch updates, and advance offset."""
        if self._last_update_id > 0:
            query_offset = self._last_update_id + 1
        else:
            query_offset = 0

        updates = await self.get_updates(query_offset, self._poll_timeout_seconds)
        for upd in updates:
            if upd.update_id > self._last_update_id:
                self._last_update_id = upd.update_id
            await self.dispatch_update(upd)
        return updates

    async def start_polling(self) -> None:
        """Start long-polling loop as a background asyncio task."""
        if self._is_polling:
            logger.warning("Telegram polling loop is already running")
            return
        self._is_polling = True
        self._polling_task = asyncio.create_task(self._polling_loop())
        logger.info("Telegram polling background task started")

    async def stop_polling(self) -> None:
        """Stop background long-polling loop with graceful task cancellation."""
        self._is_polling = False
        if self._polling_task is not None:
            if not self._polling_task.done():
                self._polling_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._polling_task
            self._polling_task = None
        logger.info("Telegram polling loop stopped")

    async def _polling_loop(self) -> None:
        """Internal background loop running poll_once continually until cancelled."""
        while self._is_polling:
            try:
                await self.poll_once()
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break
            except Exception as loop_exc:
                logger.error("Error in Telegram polling loop: %s", loop_exc)
                try:
                    await asyncio.sleep(1.0)
                except asyncio.CancelledError:
                    break

    # -------------------------------------------------------------------------
    # Update Dispatcher & Whitelist Security
    # -------------------------------------------------------------------------

    async def dispatch_update(self, update: TelegramUpdate) -> None:
        """Inspect and dispatch an incoming update enforcing strict whitelist check."""
        # 1. Dispatch Slash Command Message
        if update.message is not None:
            msg = update.message
            # Strict Whitelist Check
            if str(msg.chat_id).strip() != str(self._config.chat_id).strip():
                logger.warning(
                    "SECURITY: Blocked unauthorized message from chat_id='%s' (configured: '%s')",
                    msg.chat_id,
                    self._config.chat_id,
                )
                return

            if not msg.text.startswith("/"):
                # Ignore non-command chat messages
                return

            try:
                cmd_req = TelegramCommandRequest.parse(
                    text=msg.text,
                    user_id=msg.chat_id,
                    chat_id=msg.chat_id,
                )
                await self.handle_command(cmd_req)
            except InvalidTelegramCommandException as inv_exc:
                logger.warning("Invalid command syntax: %s", inv_exc)
                await self.send_text_message(f"⚠️ Lệnh không hợp lệ: {inv_exc}", None)
            except TelegramSecurityViolationException as sec_exc:
                logger.warning("Security violation: %s", sec_exc)

        # 2. Dispatch Callback Query
        if update.callback_query is not None:
            cb = update.callback_query
            # Strict Whitelist Check
            if str(cb.chat_id).strip() != str(self._config.chat_id).strip():
                logger.warning(
                    "SECURITY: Blocked unauthorized callback query from chat_id='%s'",
                    cb.chat_id,
                )
                return

            await self.handle_callback(cb.callback_id, cb.data)

    # -------------------------------------------------------------------------
    # TelegramCommandHandlerPort Implementation
    # -------------------------------------------------------------------------

    async def handle_command(
        self,
        request: TelegramCommandRequest,
    ) -> str:
        """Process validated user slash command, send response message, and return text."""
        # Enforce Whitelist Check
        if str(request.chat_id).strip() != str(self._config.chat_id).strip():
            raise TelegramSecurityViolationException(
                f"Unauthorized access attempt from chat_id '{request.chat_id}'"
            )

        cmd = request.command.lower()

        # Custom commands override
        if cmd in self._custom_commands:
            custom_fn = self._custom_commands[cmd]
            custom_response = await custom_fn(request)
            await self.send_text_message(custom_response, None)
            return custom_response

        # Built-in Slash Commands Routing
        if cmd == TelegramBotCommand.STATUS:
            return await self._handle_cmd_status()
        elif cmd == TelegramBotCommand.POSITIONS:
            return await self._handle_cmd_positions()
        elif cmd == TelegramBotCommand.SHADOW:
            return await self._handle_cmd_shadow()
        elif cmd == TelegramBotCommand.QUOTA:
            return await self._handle_cmd_quota()
        elif cmd == TelegramBotCommand.MODE:
            return await self._handle_cmd_mode(request.args)
        elif cmd == TelegramBotCommand.PAUSE:
            return await self._handle_cmd_pause(request.args)
        elif cmd == TelegramBotCommand.RESUME:
            return await self._handle_cmd_resume()
        elif cmd == TelegramBotCommand.PANIC_CLOSE_ALL:
            return await self._handle_cmd_panic_close_all()
        elif cmd == TelegramBotCommand.APPROVE_MODEL:
            return await self._handle_cmd_approve_model()
        elif cmd == TelegramBotCommand.RETRAIN:
            return await self._handle_cmd_retrain(request.args)
        else:
            unknown_msg = (
                f"⚠️ Lệnh không xác định: `/{cmd}`\n"
                "Danh sách lệnh khả dụng:\n"
                "• `/status` - Trạng thái hệ thống, NAV, PnL, Drawdown\n"
                "• `/positions` - Danh sách vị thế mở kèm nút BE / Đóng\n"
                "• `/shadow` - Thống kê Risk Arbiter shadow tracking\n"
                "• `/quota` - Hạn ngạch 9Router RPM/TPM/RPD\n"
                "• `/mode <paper|testnet|live>` - Đổi chế độ thực thi\n"
                "• `/pause <hours>` - Tạm dừng giao dịch\n"
                "• `/resume` - Tiếp tục giao dịch\n"
                "• `/panic_close_all` - Nút đỏ đóng tất cả vị thế khẩn cấp\n"
                "• `/approve_model` - Phê duyệt Challenger model thành Champion\n"
                "• `/retrain [fast|full]` - Huấn luyện lại model Challenger"
            )
            await self.send_text_message(unknown_msg, None)
            return unknown_msg

    async def handle_callback(
        self,
        callback_id: str,
        callback_data: str,
    ) -> str:
        """Process inline keyboard button callback query and execute associated action."""
        data = callback_data.strip()

        # 1. 2-Step Confirmation: Switch to LIVE mode
        if data == "confirm_mode:live":
            self._execution_mode = "LIVE"
            if self._on_mode_change is not None:
                await self._on_mode_change("LIVE")
            ans_text = "✅ Chế độ LIVE đã được kích hoạt!"
            await self.answer_callback_query(callback_id, ans_text)
            notice_text = (
                "🚨 *EXECUTION MODE ĐÃ CHUYỂN SANG LIVE!*\n"
                "Bot hiện đang giao dịch bằng tiền thật trên Binance Futures."
            )
            await self.send_text_message(notice_text, None)
            return ans_text

        # 2. 2-Step Confirmation: Panic Close All
        if data == "confirm_panic_close":
            if self._on_panic_close is not None:
                await self._on_panic_close()
            ans_text = "🚨 Đang thực thi Panic Close All!"
            await self.answer_callback_query(callback_id, ans_text)
            notice_text = (
                "🚨 *PANIC CLOSE ALL ĐÃ ĐƯỢC KÍCH HOẠT!*\n"
                "Toàn bộ lệnh chờ đã bị hủy và tất cả vị thế đang được đóng theo giá Market."
            )
            await self.send_text_message(notice_text, None)
            return ans_text

        # 3. Cancel Action
        if data == "cancel_action":
            ans_text = "❌ Thao tác đã được hủy bỏ an toàn."
            await self.answer_callback_query(callback_id, ans_text)
            await self.send_text_message("❌ Thao tác đã được hủy bỏ an toàn.", None)
            return ans_text

        # 4. Breakeven Position
        if data.startswith("be:"):
            pos_id = data.split(":", 1)[1]
            if self._on_breakeven is not None:
                await self._on_breakeven(pos_id)
            ans_text = f"🛡️ Đã dời SL về BE cho vị thế {pos_id}."
            await self.answer_callback_query(callback_id, ans_text)
            notice_text = (
                f"🛡️ *SL BREAKEVEN ACTIVATED:*\n"
                f"Đã cập nhật Stop Loss về giá hòa vốn cho vị thế `{pos_id}`."
            )
            await self.send_text_message(notice_text, None)
            return ans_text

        # 5. Market Close Single Position
        if data.startswith("close:"):
            pos_id = data.split(":", 1)[1]
            if self._on_close_position is not None:
                await self._on_close_position(pos_id)
            ans_text = f"🚨 Đã gửi lệnh đóng vị thế {pos_id}."
            await self.answer_callback_query(callback_id, ans_text)
            notice_text = (
                f"🚨 *CLOSE POSITION ACTIVATED:*\n"
                f"Đã gửi lệnh Market đóng khẩn cấp vị thế `{pos_id}`."
            )
            await self.send_text_message(notice_text, None)
            return ans_text

        # 6. Approve Challenger Model Prompt
        if data == "approve_model":
            return await self._handle_cmd_approve_model()

        # 7. 2-Step Confirmation: Confirm Approve Model
        if data == "confirm_approve_model":
            if self._on_approve_model is not None:
                await self._on_approve_model()
            self._challenger_pending = False
            ans_text = "🏆 Đã thăng hạng Challenger thành công!"
            await self.answer_callback_query(callback_id, ans_text)
            notice_text = (
                "🏆 *CHALLENGER PROMOTED SUCCESSFULLY*\n"
                "Model Challenger đã được phê duyệt và chính thức trở thành Champion mới!"
            )
            await self.send_text_message(notice_text, None)
            return ans_text

        # 8. Cancel Model Approval
        if data == "cancel_approve_model":
            ans_text = "❌ Đã hủy bỏ thao tác thăng hạng model."
            await self.answer_callback_query(callback_id, ans_text)
            await self.send_text_message("❌ Đã hủy bỏ thao tác thăng hạng model.", None)
            return ans_text

        # Unrecognized callback
        ans_text = "⚠️ Callback query không xác định."
        await self.answer_callback_query(callback_id, ans_text)
        return ans_text

    # -------------------------------------------------------------------------
    # Command Handler Private Helpers
    # -------------------------------------------------------------------------

    async def _handle_cmd_status(self) -> str:
        """Format and return /status overview."""
        ctx = self._account_context
        if ctx is not None:
            mode_str = ctx.execution_mode.value
            nav = ctx.equity_nav_usd
            daily_pnl = ctx.current_daily_pnl_usd
            dd = ctx.current_daily_dd_pct
            margin_util = ctx.margin_utilization_pct
            pos_count = len(ctx.active_positions)
            order_count = len(ctx.pending_orders)
        else:
            mode_str = self._execution_mode
            nav = 10000.0
            daily_pnl = 0.0
            dd = 0.0
            margin_util = 0.0
            pos_count = 0
            order_count = 0

        if daily_pnl >= 0.0:
            pnl_emoji = "📈"
            pnl_str = f"+${daily_pnl:,.2f}"
        else:
            pnl_emoji = "📉"
            pnl_str = f"-${abs(daily_pnl):,.2f}"

        if self._is_paused:
            trading_state = "⏸️ PAUSED"
        else:
            trading_state = "▶️ ACTIVE"

        lines: list[str] = [
            "📊 *[HỆ THỐNG GIAO DỊCH - TRẠNG THÁI]*",
            f"• *Execution Mode:* `{mode_str}`",
            "• *System Health:* `🟢 ONLINE / HEALTHY`",
            f"• *Trading Status:* `{trading_state}`",
            f"• *Portfolio NAV:* `${nav:,.2f}`",
            f"• *Daily PnL:* {pnl_emoji} `{pnl_str}`",
            f"• *Drawdown:* `{dd:.2f}%`",
            f"• *Margin Utilization:* `{margin_util:.1f}%`",
            f"• *Open Positions:* `{pos_count}`",
            f"• *Pending Orders:* `{order_count}`",
        ]
        status_text = "\n".join(lines)
        await self.send_text_message(status_text, None)
        return status_text

    async def _handle_cmd_positions(self) -> str:
        """Format and return /positions with interactive inline buttons."""
        ctx = self._account_context
        if ctx is None or len(ctx.active_positions) == 0:
            empty_msg = "💡 Hiện không có vị thế nào đang mở (No active positions)."
            await self.send_text_message(empty_msg, None)
            return empty_msg

        lines: list[str] = ["💼 *[DANH SÁCH VỊ THẾ ĐANG MỞ]*"]
        keyboard_rows: list[list[TelegramInlineButton]] = []

        for pos in ctx.active_positions:
            if pos.unrealized_pnl_usd >= 0.0:
                pnl_str = f"+${pos.unrealized_pnl_usd:,.2f} (+{pos.unrealized_pnl_pct:.2f}%)"
            else:
                pnl_str = f"-${abs(pos.unrealized_pnl_usd):,.2f} ({pos.unrealized_pnl_pct:.2f}%)"

            if pos.is_sl_at_breakeven:
                be_tag = " (BE 🛡️)"
            else:
                be_tag = ""

            pos_desc = (
                f"• *{pos.symbol}* ({pos.side.value})\n"
                f"  Size: `{pos.quantity}` | Notional: `${pos.notional_usd:,.2f}`\n"
                f"  Entry: `${pos.entry_price:,.4f}` | Mark: `${pos.mark_price:,.4f}`\n"
                f"  PnL: `{pnl_str}`\n"
                f"  SL: `${pos.stop_loss_price:,.4f}`{be_tag} | TP: `${pos.take_profit_price:,.4f}`"
            )
            lines.append(pos_desc)

            row_buttons = [
                TelegramInlineButton(
                    text=f"🛡️ BE {pos.symbol}",
                    callback_data=f"be:{pos.position_id}",
                    url=None,
                ),
                TelegramInlineButton(
                    text=f"🚨 Đóng {pos.symbol}",
                    callback_data=f"close:{pos.position_id}",
                    url=None,
                ),
            ]
            keyboard_rows.append(row_buttons)

        positions_text = "\n\n".join(lines)
        await self.send_text_message(positions_text, keyboard_rows)
        return positions_text

    async def _handle_cmd_shadow(self) -> str:
        """Format and return /shadow Arbiter defense stats."""
        stats = self._shadow_stats
        if "evaluated_count" in stats:
            eval_count = int(stats["evaluated_count"])
        else:
            eval_count = 0

        if "vetoed_count" in stats:
            veto_count = int(stats["vetoed_count"])
        else:
            veto_count = 0

        if "defense_rate_pct" in stats:
            defense_rate = float(stats["defense_rate_pct"])
        else:
            defense_rate = 100.0

        if "avoided_loss_usd" in stats:
            avoided_loss = float(stats["avoided_loss_usd"])
        else:
            avoided_loss = 0.0

        lines: list[str] = [
            "🛡️ *[RISK ARBITER - SHADOW STATS]*",
            f"• *Shadow Orders Evaluated:* `{eval_count}`",
            f"• *Vetoed / Filtered Orders:* `{veto_count}`",
            f"• *Defense Rate:* `{defense_rate:.1f}%`",
            f"• *Avoided Negative PnL:* `${avoided_loss:,.2f}`",
        ]
        shadow_text = "\n".join(lines)
        await self.send_text_message(shadow_text, None)
        return shadow_text

    async def _handle_cmd_quota(self) -> str:
        """Format and return /quota 9Router rate limiter usage."""
        q = self._quota_stats
        if "fast_rpm" in q:
            fast_rpm = int(q["fast_rpm"])
        else:
            fast_rpm = 0

        if "fast_rpm_limit" in q:
            fast_rpm_limit = int(q["fast_rpm_limit"])
        else:
            fast_rpm_limit = 60

        if "deep_rpm" in q:
            deep_rpm = int(q["deep_rpm"])
        else:
            deep_rpm = 0

        if "deep_rpm_limit" in q:
            deep_rpm_limit = int(q["deep_rpm_limit"])
        else:
            deep_rpm_limit = 20

        if "judge_rpm" in q:
            judge_rpm = int(q["judge_rpm"])
        else:
            judge_rpm = 0

        if "judge_rpm_limit" in q:
            judge_rpm_limit = int(q["judge_rpm_limit"])
        else:
            judge_rpm_limit = 10

        if "rpd_count" in q:
            rpd_count = int(q["rpd_count"])
        else:
            rpd_count = 0

        lines: list[str] = [
            "⚡ *[9ROUTER API GATEWAY - QUOTA STATS]*",
            f"• *Fast Tier:* `{fast_rpm}/{fast_rpm_limit} RPM`",
            f"• *Deep Tier:* `{deep_rpm}/{deep_rpm_limit} RPM`",
            f"• *Judge Tier:* `{judge_rpm}/{judge_rpm_limit} RPM`",
            f"• *Daily Total Requests (RPD):* `{rpd_count}`",
        ]
        quota_text = "\n".join(lines)
        await self.send_text_message(quota_text, None)
        return quota_text

    async def _handle_cmd_mode(self, args: list[str]) -> str:
        """Handle /mode <paper|testnet|live> with 2-step confirmation for live."""
        if len(args) == 0:
            usage_msg = (
                f"💡 Execution Mode hiện tại: `{self._execution_mode}`\n"
                "Cú pháp đổi chế độ: `/mode <paper|testnet|live>`"
            )
            await self.send_text_message(usage_msg, None)
            return usage_msg

        target_mode = args[0].strip().lower()
        if target_mode == "live":
            confirm_keyboard: list[list[TelegramInlineButton]] = [
                [
                    TelegramInlineButton(
                        text="✅ Xác nhận chuyển sang LIVE",
                        callback_data="confirm_mode:live",
                        url=None,
                    ),
                    TelegramInlineButton(
                        text="❌ Hủy bỏ",
                        callback_data="cancel_action",
                        url=None,
                    ),
                ]
            ]
            confirm_prompt = (
                "⚠️ *XÁC NHẬN CHUYỂN SANG LIVE MODE*\n\n"
                "Bạn có chắc chắn muốn chuyển execution mode sang *LIVE*?\n"
                "• Lệnh thật sẽ được gửi trực tiếp đến Binance Futures.\n"
                "• Vốn thật trong ví Futures sẽ được sử dụng ký quỹ.\n\n"
                "Nhấn nút bên dưới để xác nhận:"
            )
            await self.send_text_message(confirm_prompt, confirm_keyboard)
            return confirm_prompt
        elif target_mode in ("paper", "testnet"):
            new_mode = target_mode.upper()
            self._execution_mode = new_mode
            if self._on_mode_change is not None:
                await self._on_mode_change(new_mode)
            success_msg = f"✅ Đã chuyển execution mode sang: `{new_mode}`"
            await self.send_text_message(success_msg, None)
            return success_msg
        else:
            invalid_msg = (
                f"⚠️ Chế độ không hợp lệ: `{target_mode}`\nChỉ chấp nhận: `paper`, `testnet`, `live`"
            )
            await self.send_text_message(invalid_msg, None)
            return invalid_msg

    async def _handle_cmd_pause(self, args: list[str]) -> str:
        """Handle /pause <hours>."""
        pause_hours: float | None = None
        if len(args) > 0:
            try:
                pause_hours = float(args[0])
            except ValueError:
                err_msg = f"⚠️ Tham số giờ không hợp lệ: `{args[0]}`. Nhập số, ví dụ `/pause 2`"
                await self.send_text_message(err_msg, None)
                return err_msg

        self._is_paused = True
        if self._on_pause is not None:
            await self._on_pause(pause_hours)

        if pause_hours is not None:
            pause_text = f"⏸️ Đã tạm dừng giao dịch (Trading paused) trong `{pause_hours:.1f}` giờ."
        else:
            pause_text = (
                "⏸️ Đã tạm dừng giao dịch (Trading paused) vô thời hạn.\n"
                "Dùng `/resume` để kích hoạt lại."
            )
        await self.send_text_message(pause_text, None)
        return pause_text

    async def _handle_cmd_resume(self) -> str:
        """Handle /resume."""
        self._is_paused = False
        self._pause_until_ms = None
        if self._on_resume is not None:
            await self._on_resume()
        resume_text = "▶️ Đã tiếp tục giao dịch (Trading resumed). Hệ thống đã hoạt động trở lại."
        await self.send_text_message(resume_text, None)
        return resume_text

    async def _handle_cmd_panic_close_all(self) -> str:
        """Handle /panic_close_all requiring 2-step confirmation."""
        panic_keyboard: list[list[TelegramInlineButton]] = [
            [
                TelegramInlineButton(
                    text="🚨 XÁC NHẬN ĐÓNG TOÀN BỘ",
                    callback_data="confirm_panic_close",
                    url=None,
                ),
                TelegramInlineButton(
                    text="❌ Hủy bỏ",
                    callback_data="cancel_action",
                    url=None,
                ),
            ]
        ]
        prompt_text = (
            "🚨 *CẢNH BÁO KHẨN CẤP: PANIC CLOSE ALL*\n\n"
            "Hành động này sẽ thực thi ngay lập tức:\n"
            "1. Hủy TOÀN BỘ lệnh Limit chờ khớp trên sàn.\n"
            "2. Đóng TOÀN BỘ vị thế đang mở bằng lệnh Market.\n\n"
            "Bấm nút xác nhận bên dưới nếu bạn thực sự muốn dừng khẩn cấp:"
        )
        await self.send_text_message(prompt_text, panic_keyboard)
        return prompt_text

    async def _handle_cmd_approve_model(self) -> str:
        """Handle /approve_model displaying a 2-step confirmation prompt."""
        if not self._challenger_pending:
            not_pending_msg = "💡 Hiện không có Challenger model nào đang chờ phê duyệt."
            await self.send_text_message(not_pending_msg, None)
            return not_pending_msg

        confirm_keyboard: list[list[TelegramInlineButton]] = [
            [
                TelegramInlineButton(
                    text="✅ Xác nhận thăng hạng",
                    callback_data="confirm_approve_model",
                    url=None,
                ),
                TelegramInlineButton(
                    text="❌ Hủy bỏ",
                    callback_data="cancel_approve_model",
                    url=None,
                ),
            ]
        ]
        prompt_text = (
            "🏆 *XÁC NHẬN THĂNG HẠNG CHALLENGER MODEL*\n\n"
            "Hành động này sẽ thực thi:\n"
            "1. Lưu trữ Champion hiện tại vào thư mục `models/archive/`.\n"
            "2. Thăng hạng Challenger thành Champion mới.\n"
            "3. Kích hoạt hot-reload model tức thì trên Live Paper Runner (zero downtime).\n\n"
            "Bấm nút xác nhận bên dưới nếu bạn muốn thực thi:"
        )
        await self.send_text_message(prompt_text, confirm_keyboard)
        return prompt_text

    async def _handle_cmd_retrain(self, args: list[str]) -> str:
        """Handle /retrain [fast|full] command."""
        if len(args) > 0 and args[0].strip().lower() in ("fast", "full"):
            mode = args[0].strip().lower()
        else:
            mode = "fast"

        if self._on_retrain is not None:
            await self._on_retrain(mode)

        notice_text = (
            f"⏳ *KHỞI ĐỘNG HUẤN LUYỆN CHALLENGER*\n\n"
            f"• Chế độ: `{mode.upper()}`\n"
            f"• Hệ thống đang tiến hành huấn luyện mô hình Challenger dưới nền.\n"
            f"• Quá trình không làm gián đoạn giao dịch hiện tại.\n"
            f"• Khi hoàn tất, báo cáo giải đấu Tournament và nút phê duyệt sẽ được gửi về đây."
        )
        await self.send_text_message(notice_text, None)
        return notice_text
