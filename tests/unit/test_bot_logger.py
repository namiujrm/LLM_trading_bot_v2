"""Unit tests for Structured JSON Lines Bot Logger and Invariant #9 compliance."""

from __future__ import annotations

import ast
import json
import logging
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import pytest
from scripts.check_no_defaults_or_hardcode import ZeroDefaultAndHardcodeVisitor

from trading_bot.infrastructure.config_schemas import LoggingConfig
from trading_bot.infrastructure.logging import (
    JsonLinesFormatter,
    LogCategory,
    StructuredBotLogger,
    close_bot_logger,
    get_bot_logger,
    get_bot_logger_for_date,
    get_log_dir_for_category,
    get_log_file_path,
    get_log_file_prefix,
)


class SampleEnum(StrEnum):
    """Sample enum for testing serialization."""

    ALPHA = "alpha_signal"
    BETA = "beta_signal"


def test_log_categories_enum_members() -> None:
    """Validate all 4 subsystem categories are defined with correct string values."""
    assert LogCategory.TRADING.value == "trading"
    assert LogCategory.ML_ENGINE.value == "ml_engine"
    assert LogCategory.AGENTS.value == "agents"
    assert LogCategory.SYSTEM.value == "system"


def test_file_prefix_and_path_generation(tmp_path: Path) -> None:
    """Validate directory and file path conventions for all categories."""
    date_str: str = "2026-11-09"
    log_dir_str: str = str(tmp_path)

    expected_mappings: list[tuple[LogCategory, str, str]] = [
        (LogCategory.TRADING, "trading", "orders"),
        (LogCategory.ML_ENGINE, "ml_engine", "predictions"),
        (LogCategory.AGENTS, "agents", "deliberation"),
        (LogCategory.SYSTEM, "system", "app"),
    ]

    for category, expected_dir, expected_prefix in expected_mappings:
        assert get_log_file_prefix(category) == expected_prefix
        category_dir = get_log_dir_for_category(log_dir_str, category)
        assert category_dir == tmp_path / expected_dir
        expected_file = tmp_path / expected_dir / f"{expected_prefix}_{date_str}.jsonl"
        assert get_log_file_path(log_dir_str, category, date_str) == expected_file


def test_directory_creation_for_all_four_categories(tmp_path: Path) -> None:
    """Validate that initializing loggers creates subdirectories on disk."""
    config = LoggingConfig(
        log_dir=str(tmp_path),
        retention_days=7,
        min_level="INFO",
        console_output=False,
    )
    date_str: str = "2026-11-09"

    for category in [
        LogCategory.TRADING,
        LogCategory.ML_ENGINE,
        LogCategory.AGENTS,
        LogCategory.SYSTEM,
    ]:
        logger = get_bot_logger_for_date(category, config, date_str)
        logger.info(f"Init test for {category.value}")
        close_bot_logger(logger)

        category_dir = tmp_path / category.value
        assert category_dir.exists()
        assert category_dir.is_dir()

        prefix = get_log_file_prefix(category)
        log_file = category_dir / f"{prefix}_{date_str}.jsonl"
        assert log_file.exists()


def test_log_entries_are_valid_json_lines_with_required_fields(tmp_path: Path) -> None:
    """Validate every log line is valid JSON Lines with required schema keys."""
    config = LoggingConfig(
        log_dir=str(tmp_path),
        retention_days=14,
        min_level="INFO",
        console_output=False,
    )
    date_str: str = "2026-11-09"
    test_message: str = "Test execution order placed"

    logger = get_bot_logger_for_date(LogCategory.TRADING, config, date_str)
    logger.info(test_message)
    close_bot_logger(logger)

    log_file = tmp_path / "trading" / f"orders_{date_str}.jsonl"
    assert log_file.exists()

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert "timestamp" in entry
    assert "level" in entry
    assert "category" in entry
    assert "logger_name" in entry
    assert "message" in entry

    # Validate timestamp is valid ISO-8601 UTC
    parsed_dt = datetime.fromisoformat(entry["timestamp"])
    assert parsed_dt.tzinfo is not None

    assert entry["level"] == "INFO"
    assert entry["category"] == "trading"
    assert entry["logger_name"] == "trading_bot.trading"
    assert entry["message"] == test_message


def test_contextual_payload_serialization(tmp_path: Path) -> None:
    """Validate numeric metrics, nested dicts, and non-primitive objects serialization."""
    config = LoggingConfig(
        log_dir=str(tmp_path),
        retention_days=7,
        min_level="DEBUG",
        console_output=False,
    )
    date_str: str = "2026-11-09"

    logger = get_bot_logger_for_date(LogCategory.ML_ENGINE, config, date_str)

    sample_uuid = uuid.uuid4()
    now_utc = datetime.now(UTC)
    payload_data = {
        "model_id": "lgbm_wf_fold3",
        "symbol": "BTCUSDT",
        "calibrated_prob": 0.6284,
        "raw_prob": 0.612,
        "ood_score": 0.012,
        "is_ood": False,
        "sample_uuid": sample_uuid,
        "created_at": now_utc,
        "enum_val": SampleEnum.ALPHA,
        "path_val": Path("data/features_36m"),
        "tags": {"crypto", "momentum"},
        "nested_metrics": {
            "gk_volatility": 0.0215,
            "hurts_exp": 0.58,
            "feature_importances": [0.12, 0.08, 0.05],
        },
    }

    logger.info("ML inference evaluated", extra={"payload": payload_data})
    close_bot_logger(logger)

    log_file = tmp_path / "ml_engine" / f"predictions_{date_str}.jsonl"
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert "payload" in entry
    payload = entry["payload"]

    assert payload["model_id"] == "lgbm_wf_fold3"
    assert payload["symbol"] == "BTCUSDT"
    assert payload["calibrated_prob"] == 0.6284
    assert payload["is_ood"] is False
    assert payload["sample_uuid"] == str(sample_uuid)
    assert payload["enum_val"] == "alpha_signal"
    assert payload["path_val"] == "data/features_36m"
    assert set(payload["tags"]) == {"crypto", "momentum"}
    assert payload["nested_metrics"]["gk_volatility"] == 0.0215
    assert payload["nested_metrics"]["feature_importances"] == [0.12, 0.08, 0.05]


def test_logging_levels_filtering(tmp_path: Path) -> None:
    """Validate records below min_level are filtered out while higher levels are recorded."""
    config = LoggingConfig(
        log_dir=str(tmp_path),
        retention_days=7,
        min_level="WARNING",
        console_output=False,
    )
    date_str: str = "2026-11-09"

    logger = get_bot_logger_for_date(LogCategory.SYSTEM, config, date_str)
    logger.debug("Debug should be dropped")
    logger.info("Info should be dropped")
    logger.warning("Warning should be recorded")
    logger.error("Error should be recorded")
    logger.critical("Critical should be recorded")
    close_bot_logger(logger)

    log_file = tmp_path / "system" / f"app_{date_str}.jsonl"
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3

    entries = [json.loads(line) for line in lines]
    levels = [e["level"] for e in entries]
    assert levels == ["WARNING", "ERROR", "CRITICAL"]


def test_invalid_min_level_raises_error(tmp_path: Path) -> None:
    """Validate invalid min_level fails fast with ValueError."""
    config = LoggingConfig(
        log_dir=str(tmp_path),
        retention_days=7,
        min_level="INVALID_LEVEL_NAME",
        console_output=False,
    )
    with pytest.raises(ValueError, match="Invalid log level"):
        get_bot_logger_for_date(LogCategory.SYSTEM, config, "2026-11-09")


def test_rotating_file_handler_properties(tmp_path: Path) -> None:
    """Validate TimedRotatingFileHandler setup, rotation schedule, and backupCount."""
    config = LoggingConfig(
        log_dir=str(tmp_path),
        retention_days=30,
        min_level="INFO",
        console_output=False,
    )
    logger = get_bot_logger(LogCategory.TRADING, config)

    timed_handlers = [h for h in logger.handlers if isinstance(h, TimedRotatingFileHandler)]
    assert len(timed_handlers) == 1
    handler = timed_handlers[0]

    assert handler.when == "MIDNIGHT"
    assert handler.backupCount == 30
    assert handler.encoding == "utf-8"
    assert isinstance(handler.formatter, JsonLinesFormatter)
    close_bot_logger(logger)


def test_utf8_encoding_preserves_vietnamese_and_emojis(tmp_path: Path) -> None:
    """Validate UTF-8 handling for Vietnamese diacritics and emojis without corruption."""
    config = LoggingConfig(
        log_dir=str(tmp_path),
        retention_days=7,
        min_level="INFO",
        console_output=False,
    )
    date_str: str = "2026-11-09"

    logger = get_bot_logger_for_date(LogCategory.AGENTS, config, date_str)
    vietnamese_msg: str = "Chiến lược giao dịch: vào lệnh MUA 0.5 BTC, ngưỡng cắt lỗ 95,000 USD"
    emoji_msg: str = "🚀 Lệnh khớp thành công 🎯 PnL: +2.5% 🔥"
    multilingual_msg: str = "アルゴリズム取引 [東京セッション] 100%"

    logger.info(vietnamese_msg)
    logger.info(emoji_msg)
    logger.info(multilingual_msg)
    close_bot_logger(logger)

    log_file = tmp_path / "agents" / f"deliberation_{date_str}.jsonl"
    content = log_file.read_text(encoding="utf-8")
    lines = content.strip().splitlines()
    assert len(lines) == 3

    entry0 = json.loads(lines[0])
    entry1 = json.loads(lines[1])
    entry2 = json.loads(lines[2])

    assert entry0["message"] == vietnamese_msg
    assert entry1["message"] == emoji_msg
    assert entry2["message"] == multilingual_msg


def test_structured_bot_logger_helper_methods(tmp_path: Path) -> None:
    """Validate StructuredBotLogger event methods and level helpers."""
    config = LoggingConfig(
        log_dir=str(tmp_path),
        retention_days=7,
        min_level="DEBUG",
        console_output=False,
    )

    s_logger = StructuredBotLogger(LogCategory.TRADING, config)

    s_logger.log_order_event(
        event_type="ORDER_SUBMITTED",
        order_id="ord_9901",
        symbol="BTCUSDT",
        details={"side": "BUY", "price": 98500.0, "size": 0.25},
    )

    s_logger.log_ml_prediction(
        model_name="lgbm_directional_30m",
        symbol="BTCUSDT",
        probability=0.6415,
        details={"venn_abers_lower": 0.61, "venn_abers_upper": 0.67},
    )

    s_logger.log_agent_deliberation(
        agent_name="risk_arbiter",
        symbol="BTCUSDT",
        decision="APPROVE_FULL",
        details={"open_heat_pct": 3.8, "max_heat_pct": 6.0},
    )

    s_logger.log_system_event(
        event_type="GATEWAY_READY",
        component="binance_futures",
        message="Websocket stream authenticated",
        details={"latency_ms": 18.2},
    )

    s_logger.debug("Debug test message", {"test_key": "debug_val"})
    s_logger.warning("Warning test message", {"warn_key": 100})
    s_logger.error("Error test message", {"err_code": "ERR_01"})
    s_logger.critical("Critical test message", {"fatal": True})

    s_logger.close()

    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    log_file = tmp_path / "trading" / f"orders_{date_str}.jsonl"
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 8

    entries = [json.loads(line) for line in lines]

    # Check order event
    order_entry = entries[0]
    assert "ORDER_SUBMITTED" in order_entry["message"]
    assert order_entry["payload"]["event_type"] == "ORDER_SUBMITTED"
    assert order_entry["payload"]["order_id"] == "ord_9901"
    assert order_entry["payload"]["symbol"] == "BTCUSDT"
    assert order_entry["payload"]["details"]["price"] == 98500.0

    # Check ML prediction
    ml_entry = entries[1]
    assert ml_entry["payload"]["model_name"] == "lgbm_directional_30m"
    assert ml_entry["payload"]["probability"] == 0.6415

    # Check agent deliberation
    agent_entry = entries[2]
    assert agent_entry["payload"]["agent_name"] == "risk_arbiter"
    assert agent_entry["payload"]["decision"] == "APPROVE_FULL"

    # Check system event
    sys_entry = entries[3]
    assert sys_entry["payload"]["component"] == "binance_futures"

    # Check level helpers
    assert entries[4]["level"] == "DEBUG"
    assert entries[5]["level"] == "WARNING"
    assert entries[6]["level"] == "ERROR"
    assert entries[7]["level"] == "CRITICAL"


def test_console_output_toggle(tmp_path: Path) -> None:
    """Validate console handler presence matches console_output configuration."""
    cfg_no_console = LoggingConfig(
        log_dir=str(tmp_path),
        retention_days=7,
        min_level="INFO",
        console_output=False,
    )
    logger_no_console = get_bot_logger(LogCategory.SYSTEM, cfg_no_console)
    assert len(logger_no_console.handlers) == 1
    assert isinstance(logger_no_console.handlers[0], TimedRotatingFileHandler)
    close_bot_logger(logger_no_console)

    cfg_with_console = LoggingConfig(
        log_dir=str(tmp_path),
        retention_days=7,
        min_level="INFO",
        console_output=True,
    )
    logger_with_console = get_bot_logger(LogCategory.SYSTEM, cfg_with_console)
    assert len(logger_with_console.handlers) == 2
    handler_types = {type(h) for h in logger_with_console.handlers}
    assert TimedRotatingFileHandler in handler_types
    assert logging.StreamHandler in handler_types
    close_bot_logger(logger_with_console)


def test_exception_serialization_in_jsonl(tmp_path: Path) -> None:
    """Validate exception tracebacks are captured in 'exception' JSON field."""
    config = LoggingConfig(
        log_dir=str(tmp_path),
        retention_days=7,
        min_level="ERROR",
        console_output=False,
    )
    date_str: str = "2026-11-09"
    logger = get_bot_logger_for_date(LogCategory.SYSTEM, config, date_str)

    try:
        _ = 1 / 0
    except ZeroDivisionError:
        logger.error("Arithmetic failure occurred", exc_info=True)
    close_bot_logger(logger)

    log_file = tmp_path / "system" / f"app_{date_str}.jsonl"
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert "exception" in entry
    assert "ZeroDivisionError: division by zero" in entry["exception"]


def test_ast_static_gate_compliance() -> None:
    """Validate that bot_logger.py and config_schemas.py satisfy Invariant #9 (0 violations)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    target_files = [
        repo_root / "src" / "trading_bot" / "infrastructure" / "logging" / "bot_logger.py",
        repo_root / "src" / "trading_bot" / "infrastructure" / "logging" / "__init__.py",
        repo_root / "src" / "trading_bot" / "infrastructure" / "config_schemas.py",
        repo_root / "tests" / "unit" / "test_bot_logger.py",
    ]

    all_violations: list[str] = []
    for file_path in target_files:
        assert file_path.exists(), f"File does not exist: {file_path}"
        code = file_path.read_text(encoding="utf-8")
        tree = ast.parse(code, filename=str(file_path))
        visitor = ZeroDefaultAndHardcodeVisitor(file_path)
        visitor.visit(tree)
        all_violations.extend(visitor.violations)

    assert len(all_violations) == 0, f"Invariant #9 violations detected: {all_violations}"
