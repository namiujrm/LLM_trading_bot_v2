"""Pure Domain Entities for Telegram Command Center (Clean Architecture, Invariant #9).

Enforces strictly ZERO default parameters, ZERO class field defaults, and ZERO silent fallbacks.
Pure Python and Pydantic only: no I/O, no network calls, no external third-party dependencies.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from trading_bot.domain.exceptions import DomainException


class TelegramParsingException(DomainException, ValueError):
    """Raised when Telegram update or payload cannot be parsed."""


class InvalidTelegramCommandException(DomainException, ValueError):
    """Raised when an incoming message is not a valid Telegram slash command."""


class TelegramSecurityViolationException(DomainException, PermissionError):
    """Raised when an unauthorized user or chat attempts to execute commands."""


class TelegramBotCommand(StrEnum):
    """Supported Telegram bot commands."""

    STATUS = "status"
    POSITIONS = "positions"
    SHADOW = "shadow"
    QUOTA = "quota"
    MODE = "mode"
    PAUSE = "pause"
    RESUME = "resume"
    PANIC_CLOSE_ALL = "panic_close_all"
    APPROVE_MODEL = "approve_model"
    RETRAIN = "retrain"


class TelegramInlineButton(BaseModel):
    """Inline keyboard button entity (Zero Defaults)."""

    text: str
    callback_data: str | None
    url: str | None

    def to_dict(self) -> dict[str, str]:
        """Serialize button to Telegram Bot API dict format."""
        btn_dict: dict[str, str] = {"text": self.text}
        if self.callback_data is not None:
            btn_dict["callback_data"] = self.callback_data
        if self.url is not None:
            btn_dict["url"] = self.url
        return btn_dict


class TelegramInlineKeyboard(BaseModel):
    """Inline keyboard markup entity composed of button rows (Zero Defaults)."""

    inline_keyboard: list[list[TelegramInlineButton]]

    def to_dict(self) -> dict[str, list[list[dict[str, str]]]]:
        """Serialize keyboard to Telegram Bot API reply_markup format."""
        serialized_rows: list[list[dict[str, str]]] = []
        for row in self.inline_keyboard:
            row_buttons: list[dict[str, str]] = []
            for btn in row:
                row_buttons.append(btn.to_dict())
            serialized_rows.append(row_buttons)
        return {"inline_keyboard": serialized_rows}


class TelegramMessage(BaseModel):
    """Telegram message entity representing incoming or outgoing text message (Zero Defaults)."""

    message_id: int | None
    chat_id: str
    text: str
    parse_mode: str | None
    reply_markup: TelegramInlineKeyboard | list[list[TelegramInlineButton]] | None
    date: int | None

    def to_api_payload(self) -> dict[str, Any]:
        """Convert message to Telegram Bot API sendMessage JSON payload."""
        payload: dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": self.text,
        }
        if self.parse_mode is not None:
            payload["parse_mode"] = self.parse_mode
        if self.reply_markup is not None:
            if isinstance(self.reply_markup, TelegramInlineKeyboard):
                payload["reply_markup"] = self.reply_markup.to_dict()
            else:
                nested_rows: list[list[dict[str, str]]] = []
                for row in self.reply_markup:
                    btn_list: list[dict[str, str]] = []
                    for btn in row:
                        btn_list.append(btn.to_dict())
                    nested_rows.append(btn_list)
                payload["reply_markup"] = {"inline_keyboard": nested_rows}
        return payload


class TelegramCallbackQuery(BaseModel):
    """Callback query generated when an inline keyboard button is pressed (Zero Defaults)."""

    callback_id: str
    from_user_id: str
    chat_id: str
    data: str
    message_id: int | None


class TelegramUpdate(BaseModel):
    """Update object received from Telegram getUpdates polling (Zero Defaults)."""

    update_id: int
    message: TelegramMessage | None
    callback_query: TelegramCallbackQuery | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TelegramUpdate:
        """Parse raw Telegram getUpdates dict into a typed domain entity."""
        if "update_id" not in payload:
            raise TelegramParsingException("Missing required 'update_id' in update payload")

        parsed_update_id: int = int(payload["update_id"])
        parsed_message: TelegramMessage | None = None
        parsed_callback_query: TelegramCallbackQuery | None = None

        if "message" in payload and payload["message"] is not None:
            msg_data: dict[str, Any] = payload["message"]
            if "chat" not in msg_data or "id" not in msg_data["chat"]:
                raise TelegramParsingException("Missing 'chat.id' in update message")

            msg_chat_id: str = str(msg_data["chat"]["id"])
            msg_id_val: int | None = None
            if "message_id" in msg_data and msg_data["message_id"] is not None:
                msg_id_val = int(msg_data["message_id"])

            msg_text_val: str = ""
            if "text" in msg_data and msg_data["text"] is not None:
                msg_text_val = str(msg_data["text"])

            msg_date_val: int | None = None
            if "date" in msg_data and msg_data["date"] is not None:
                msg_date_val = int(msg_data["date"])

            parsed_message = TelegramMessage(
                message_id=msg_id_val,
                chat_id=msg_chat_id,
                text=msg_text_val,
                parse_mode=None,
                reply_markup=None,
                date=msg_date_val,
            )

        if "callback_query" in payload and payload["callback_query"] is not None:
            cb_data: dict[str, Any] = payload["callback_query"]
            if "id" not in cb_data or "from" not in cb_data or "id" not in cb_data["from"]:
                raise TelegramParsingException("Malformed callback_query payload")

            cb_id: str = str(cb_data["id"])
            cb_user_id: str = str(cb_data["from"]["id"])
            cb_data_str: str = ""
            if "data" in cb_data and cb_data["data"] is not None:
                cb_data_str = str(cb_data["data"])

            cb_chat_id: str = cb_user_id
            cb_msg_id: int | None = None
            if "message" in cb_data and cb_data["message"] is not None:
                cb_msg_obj: dict[str, Any] = cb_data["message"]
                if "chat" in cb_msg_obj and "id" in cb_msg_obj["chat"]:
                    cb_chat_id = str(cb_msg_obj["chat"]["id"])
                if "message_id" in cb_msg_obj and cb_msg_obj["message_id"] is not None:
                    cb_msg_id = int(cb_msg_obj["message_id"])

            parsed_callback_query = TelegramCallbackQuery(
                callback_id=cb_id,
                from_user_id=cb_user_id,
                chat_id=cb_chat_id,
                data=cb_data_str,
                message_id=cb_msg_id,
            )

        return cls(
            update_id=parsed_update_id,
            message=parsed_message,
            callback_query=parsed_callback_query,
        )


class TelegramCommandRequest(BaseModel):
    """Parsed user slash command request (Zero Defaults)."""

    command: str
    args: list[str]
    user_id: str
    chat_id: str
    raw_text: str

    @classmethod
    def parse(
        cls,
        text: str,
        user_id: str,
        chat_id: str,
    ) -> TelegramCommandRequest:
        """Parse raw text into a validated TelegramCommandRequest.

        Raises InvalidTelegramCommandException if input does not begin with '/'.
        """
        clean_text: str = text.strip()
        if len(clean_text) == 0:
            raise InvalidTelegramCommandException("Empty command string provided")

        if not clean_text.startswith("/"):
            raise InvalidTelegramCommandException(
                f"Text '{clean_text}' is not a valid slash command (must begin with '/')"
            )

        tokens: list[str] = clean_text.split()
        command_token: str = tokens[0][1:]
        if "@" in command_token:
            command_token = command_token.split("@")[0]

        command_name: str = command_token.lower()
        if len(command_name) == 0:
            raise InvalidTelegramCommandException("Command name cannot be empty")

        parsed_args: list[str] = tokens[1:]
        return cls(
            command=command_name,
            args=parsed_args,
            user_id=user_id,
            chat_id=chat_id,
            raw_text=text,
        )


class TelegramCycleReport(BaseModel):
    """Institutional 30-minute trading cycle summary for Telegram reporting (Zero Defaults)."""

    cycle_timestamp_ms: int
    execution_mode: str
    portfolio_nav_usd: float
    daily_pnl_usd: float
    current_drawdown_pct: float
    open_positions_count: int
    executed_orders_count: int
    shadow_orders_count: int
    btc_funnel_passed: bool
    altcoins_scanned_count: int
    shap_top_features: dict[str, float]
    calibrated_probabilities: dict[str, float]
    arbiter_verdict_summary: str

    def format_telegram_markdown(self) -> str:
        """Format cycle metrics into an institutional Telegram Markdown summary."""
        if self.btc_funnel_passed:
            status_emoji: str = "🟢"
        else:
            status_emoji = "⚪"

        if self.daily_pnl_usd >= 0.0:
            pnl_emoji: str = "📈"
            pnl_str: str = f"+${self.daily_pnl_usd:,.2f}"
        else:
            pnl_emoji = "📉"
            pnl_str = f"-${abs(self.daily_pnl_usd):,.2f}"

        safe_verdict: str = self.arbiter_verdict_summary.replace("_", "\\_")

        lines: list[str] = [
            f"*{status_emoji} [30m CYCLE REPORT]* `Mode: {self.execution_mode}`",
            f"• *Timestamp:* `{self.cycle_timestamp_ms}`",
            f"• *NAV:* `${self.portfolio_nav_usd:,.2f}` | *DD:* `{self.current_drawdown_pct:.2f}%`",
            f"• *Daily PnL:* {pnl_emoji} `{pnl_str}`",
            f"• *Positions:* `{self.open_positions_count}` | *Exec:* `{self.executed_orders_count}` | *Shadow:* `{self.shadow_orders_count}`",
            f"• *Altcoins Scanned:* `{self.altcoins_scanned_count}`",
            f"• *Arbiter Verdict:* {safe_verdict}",
        ]

        if len(self.shap_top_features) > 0:
            lines.append("*Top SHAP Features:*")
            sorted_shap = sorted(
                self.shap_top_features.items(),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )
            for feat_name, feat_val in sorted_shap[:5]:
                lines.append(f"  `{feat_name}`: `{feat_val:+.4f}`")

        if len(self.calibrated_probabilities) > 0:
            lines.append("*ML Calibrated Probs:*")
            for sym, prob_val in self.calibrated_probabilities.items():
                lines.append(f"  `{sym}`: `{prob_val:.2%}`")

        return "\n".join(lines)
