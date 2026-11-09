"""Structured JSON Lines Daily Rotating Logger for Multi-Subsystem Trading Bot.

Enforces Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterable
from datetime import UTC, date, datetime
from enum import Enum, StrEnum
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, cast

from trading_bot.infrastructure.config_schemas import LoggingConfig


class LogCategory(StrEnum):
    """Subsystem log category partitions."""

    TRADING = "trading"
    ML_ENGINE = "ml_engine"
    AGENTS = "agents"
    SYSTEM = "system"


LOG_LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

CATEGORY_PREFIX_MAP: dict[LogCategory, str] = {
    LogCategory.TRADING: "orders",
    LogCategory.ML_ENGINE: "predictions",
    LogCategory.AGENTS: "deliberation",
    LogCategory.SYSTEM: "app",
}

STANDARD_LOGRECORD_ATTRS: set[str] = {
    "args",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "message",
    "module",
    "msecs",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "taskName",
    "thread",
    "threadName",
}


def get_log_dir_for_category(log_dir: str, category: LogCategory) -> Path:
    """Return the dedicated subdirectory path for a specific log category."""
    return Path(log_dir) / category.value


def get_log_file_prefix(category: LogCategory) -> str:
    """Return the file prefix mapped to the given log category."""
    if category not in CATEGORY_PREFIX_MAP:
        raise ValueError(f"Unknown log category: {category}")
    return CATEGORY_PREFIX_MAP[category]


def get_log_file_path(log_dir: str, category: LogCategory, date_str: str) -> Path:
    """Generate the structured log file path: {log_dir}/{category}/{prefix}_{YYYY-MM-DD}.jsonl."""
    prefix: str = get_log_file_prefix(category)
    return get_log_dir_for_category(log_dir, category) / f"{prefix}_{date_str}.jsonl"


def _json_serial_default(obj: Any) -> Any:
    """Default JSON serializer fallback for non-primitive types."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return obj.as_posix()

    if isinstance(obj, (set, frozenset)):
        serialized_items: list[object] = list(cast(Iterable[object], obj))
        return serialized_items
    return str(obj)


class JsonLinesFormatter(logging.Formatter):
    """Structured JSON Lines log record formatter."""

    def __init__(self, category: LogCategory) -> None:
        super().__init__()
        self.category: LogCategory = category

    def format(self, record: logging.LogRecord) -> str:
        record_dt = datetime.fromtimestamp(record.created, tz=UTC)
        log_entry: dict[str, Any] = {
            "timestamp": record_dt.isoformat(),
            "level": record.levelname,
            "category": self.category.value,
            "logger_name": record.name,
            "message": record.getMessage(),
        }

        # Gather custom extra fields passed into the log record
        custom_extras: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if (
                key not in STANDARD_LOGRECORD_ATTRS
                and not key.startswith("_")
                and key != "category"
            ):
                custom_extras[key] = value

        # Explicit payload or context handling
        if "payload" in record.__dict__:
            log_entry["payload"] = record.__dict__["payload"]
        elif "context" in record.__dict__:
            log_entry["context"] = record.__dict__["context"]
        elif len(custom_extras) > 0:
            log_entry["payload"] = custom_extras
            log_entry["context"] = custom_extras

        # Ensure context is included if passed alongside payload
        if "context" in record.__dict__ and "context" not in log_entry:
            log_entry["context"] = record.__dict__["context"]

        # Preserve any exception tracebacks
        if record.exc_info is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=_json_serial_default, ensure_ascii=False)


def close_bot_logger(logger: logging.Logger) -> None:
    """Close and detach all handlers from the specified logger."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def get_bot_logger_for_date(
    category: LogCategory,
    config: LoggingConfig,
    date_str: str,
) -> logging.Logger:
    """Initialize and return a rotating file logger for a category on a specific date."""
    category_dir: Path = get_log_dir_for_category(config.log_dir, category)
    category_dir.mkdir(parents=True, exist_ok=True)

    file_path: Path = get_log_file_path(config.log_dir, category, date_str)
    logger_name: str = f"trading_bot.{category.value}"
    logger: logging.Logger = logging.getLogger(logger_name)

    level_str: str = config.min_level.upper()
    if level_str not in LOG_LEVEL_MAP:
        raise ValueError(f"Invalid log level: {config.min_level}")
    level: int = LOG_LEVEL_MAP[level_str]

    logger.setLevel(level)
    logger.propagate = False

    close_bot_logger(logger)

    file_handler: TimedRotatingFileHandler = TimedRotatingFileHandler(
        filename=str(file_path),
        when="midnight",
        interval=1,
        backupCount=config.retention_days,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(JsonLinesFormatter(category))
    logger.addHandler(file_handler)

    if config.console_output:
        console_handler: logging.Handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(JsonLinesFormatter(category))
        logger.addHandler(console_handler)

    return logger


def get_bot_logger(
    category: LogCategory,
    config: LoggingConfig,
) -> logging.Logger:
    """Initialize and return a rotating file logger for a category for current UTC date."""
    date_str: str = datetime.now(UTC).strftime("%Y-%m-%d")
    return get_bot_logger_for_date(category, config, date_str)


class StructuredBotLogger:
    """High-level structured logger wrapper ensuring typed event logging with zero defaults."""

    def __init__(self, category: LogCategory, config: LoggingConfig) -> None:
        self.category: LogCategory = category
        self.config: LoggingConfig = config
        self.logger: logging.Logger = get_bot_logger(category, config)

    def log(self, level: int, message: str, payload: dict[str, Any]) -> None:
        """Emit a structured log record with an explicit payload."""
        self.logger.log(level, message, extra={"payload": payload, "context": payload})

    def debug(self, message: str, payload: dict[str, Any]) -> None:
        """Emit a DEBUG structured log record."""
        self.logger.debug(message, extra={"payload": payload, "context": payload})

    def info(self, message: str, payload: dict[str, Any]) -> None:
        """Emit an INFO structured log record."""
        self.logger.info(message, extra={"payload": payload, "context": payload})

    def warning(self, message: str, payload: dict[str, Any]) -> None:
        """Emit a WARNING structured log record."""
        self.logger.warning(message, extra={"payload": payload, "context": payload})

    def error(self, message: str, payload: dict[str, Any]) -> None:
        """Emit an ERROR structured log record."""
        self.logger.error(message, extra={"payload": payload, "context": payload})

    def critical(self, message: str, payload: dict[str, Any]) -> None:
        """Emit a CRITICAL structured log record."""
        self.logger.critical(message, extra={"payload": payload, "context": payload})

    def log_order_event(
        self,
        event_type: str,
        order_id: str,
        symbol: str,
        details: dict[str, Any],
    ) -> None:
        """Log a dedicated trading order event."""
        payload: dict[str, Any] = {
            "event_type": event_type,
            "order_id": order_id,
            "symbol": symbol,
            "details": details,
        }
        self.info(f"Order event: {event_type} for {symbol}", payload)

    def log_ml_prediction(
        self,
        model_name: str,
        symbol: str,
        probability: float,
        details: dict[str, Any],
    ) -> None:
        """Log a dedicated ML prediction event."""
        payload: dict[str, Any] = {
            "model_name": model_name,
            "symbol": symbol,
            "probability": probability,
            "details": details,
        }
        self.info(
            f"ML prediction: {model_name} on {symbol} prob={probability:.4f}",
            payload,
        )

    def log_agent_deliberation(
        self,
        agent_name: str,
        symbol: str,
        decision: str,
        details: dict[str, Any],
    ) -> None:
        """Log an agent deliberation outcome."""
        payload: dict[str, Any] = {
            "agent_name": agent_name,
            "symbol": symbol,
            "decision": decision,
            "details": details,
        }
        self.info(
            f"Agent deliberation: {agent_name} on {symbol} decision={decision}",
            payload,
        )

    def log_system_event(
        self,
        event_type: str,
        component: str,
        message: str,
        details: dict[str, Any],
    ) -> None:
        """Log a system or infrastructure lifecycle event."""
        payload: dict[str, Any] = {
            "event_type": event_type,
            "component": component,
            "details": details,
        }
        self.info(f"System event: [{component}] {message}", payload)

    def close(self) -> None:
        """Close handlers associated with this structured logger."""
        close_bot_logger(self.logger)
