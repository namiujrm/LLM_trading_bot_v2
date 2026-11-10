"""Unit tests for LivePaperRunner daemon and multi-stream orchestrator."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from trading_bot.application.services.live_bar_pipeline import LiveBarPipelineService
from trading_bot.application.services.shadow_order_tracker import ShadowOrderTracker
from trading_bot.application.use_cases.run_30m_trading_cycle import (
    Run30mTradingCycleUseCase,
    SymbolCycleResult,
    TradingCycleSummary,
)
from trading_bot.domain.entities.common.enums import (
    OutcomeCategory,
    ShadowOrderStatus,
    TradeSide,
)
from trading_bot.domain.entities.execution.shadow_order import ShadowOrderRecord
from trading_bot.domain.entities.market.market_state import (
    EnrichedMicroBarRecord,
)
from trading_bot.domain.entities.memory.episodic_case import EpisodicCase
from trading_bot.infrastructure.config_schemas import LoggingConfig
from trading_bot.infrastructure.logging.bot_logger import (
    LogCategory,
    StructuredBotLogger,
)
from trading_bot.interfaces.live_paper_runner import (
    LivePaperRunner,
    LivePaperRunnerConfig,
)


def make_test_bar(symbol: str, open_ts_ms: int, close_price: float) -> EnrichedMicroBarRecord:
    return EnrichedMicroBarRecord(
        symbol=symbol,
        open_time_ms=open_ts_ms,
        open=close_price,
        high=close_price + 5.0,
        low=close_price - 5.0,
        close=close_price,
        volume=100.0,
        quote_volume=100.0 * close_price,
        taker_buy_vol=50.0,
        taker_buy_quote_vol=50.0 * close_price,
        trade_count=100,
        ofi_sum=1.0,
        obi_mean=0.1,
        obi_std=0.05,
        spread_bps_mean=2.0,
        liq_long_usd=0.0,
        liq_short_usd=0.0,
    )


def make_test_runner_config(duration_minutes: int | None = 1) -> LivePaperRunnerConfig:
    return LivePaperRunnerConfig(
        symbols=["BTCUSDT", "ETHUSDT"],
        duration_minutes=duration_minutes,
        testnet=True,
        log_dir="logs",
        max_tracking_duration_ms=21600000,
        check_interval_seconds=0.05,
    )


@pytest.fixture
def mock_runner_dependencies() -> dict[str, Any]:
    log_cfg = LoggingConfig(
        log_dir="logs",
        retention_days=1,
        min_level="INFO",
        console_output=False,
    )
    trading_logger = StructuredBotLogger(LogCategory.TRADING, log_cfg)
    agent_logger = StructuredBotLogger(LogCategory.AGENTS, log_cfg)
    system_logger = StructuredBotLogger(LogCategory.SYSTEM, log_cfg)

    db_manager = AsyncMock()
    memory_repo = AsyncMock()
    gateway = AsyncMock()
    gateway.sync_time_offset = AsyncMock(return_value=12)

    ws_ingestor = AsyncMock()
    ws_ingestor.start = AsyncMock()
    ws_ingestor.stop = AsyncMock()

    bar_pipeline = AsyncMock(spec=LiveBarPipelineService)
    bar_pipeline.on_1m_bar = AsyncMock()

    shadow_tracker = MagicMock(spec=ShadowOrderTracker)
    shadow_tracker.on_1m_bar = MagicMock(return_value=[])

    judge_adapter = AsyncMock()

    use_case = AsyncMock(spec=Run30mTradingCycleUseCase)

    return {
        "db_manager": db_manager,
        "memory_repo": memory_repo,
        "gateway": gateway,
        "ws_ingestor": ws_ingestor,
        "bar_pipeline": bar_pipeline,
        "shadow_tracker": shadow_tracker,
        "judge_adapter": judge_adapter,
        "use_case": use_case,
        "trading_logger": trading_logger,
        "agent_logger": agent_logger,
        "system_logger": system_logger,
    }


def test_live_paper_runner_init(mock_runner_dependencies: dict[str, Any]) -> None:
    cfg = make_test_runner_config()
    runner = LivePaperRunner(
        config=cfg,
        db_manager=mock_runner_dependencies["db_manager"],
        memory_repo=mock_runner_dependencies["memory_repo"],
        gateway=mock_runner_dependencies["gateway"],
        ws_ingestor=mock_runner_dependencies["ws_ingestor"],
        bar_pipeline=mock_runner_dependencies["bar_pipeline"],
        shadow_tracker=mock_runner_dependencies["shadow_tracker"],
        judge_adapter=mock_runner_dependencies["judge_adapter"],
        use_case=mock_runner_dependencies["use_case"],
        trading_logger=mock_runner_dependencies["trading_logger"],
        agent_logger=mock_runner_dependencies["agent_logger"],
        system_logger=mock_runner_dependencies["system_logger"],
    )
    assert runner.is_running is False
    assert len(runner.cycle_history) == 0


@pytest.mark.asyncio
async def test_handle_bar_closed_no_concluded_orders(
    mock_runner_dependencies: dict[str, Any],
) -> None:
    cfg = make_test_runner_config()
    runner = LivePaperRunner(
        config=cfg,
        db_manager=mock_runner_dependencies["db_manager"],
        memory_repo=mock_runner_dependencies["memory_repo"],
        gateway=mock_runner_dependencies["gateway"],
        ws_ingestor=mock_runner_dependencies["ws_ingestor"],
        bar_pipeline=mock_runner_dependencies["bar_pipeline"],
        shadow_tracker=mock_runner_dependencies["shadow_tracker"],
        judge_adapter=mock_runner_dependencies["judge_adapter"],
        use_case=mock_runner_dependencies["use_case"],
        trading_logger=mock_runner_dependencies["trading_logger"],
        agent_logger=mock_runner_dependencies["agent_logger"],
        system_logger=mock_runner_dependencies["system_logger"],
    )

    bar = make_test_bar("BTCUSDT", 1731162600000, 90000.0)
    await runner.handle_bar_closed(bar)

    mock_runner_dependencies["bar_pipeline"].on_1m_bar.assert_awaited_once_with(bar)
    mock_runner_dependencies["shadow_tracker"].on_1m_bar.assert_called_once_with("BTCUSDT", bar)
    mock_runner_dependencies["judge_adapter"].conduct_autopsy.assert_not_called()


@pytest.mark.asyncio
async def test_handle_bar_closed_with_concluded_order_triggers_judge(
    mock_runner_dependencies: dict[str, Any],
) -> None:
    concluded_record = ShadowOrderRecord(
        shadow_id="shadow_btc_1",
        parent_order_id=None,
        symbol="BTCUSDT",
        side=TradeSide.LONG,
        entry_price=90000.0,
        stop_loss_price=89000.0,
        take_profit_price=92000.0,
        veto_category=None,
        is_post_sl_tracker=False,
        entry_timestamp_ms=1731162600000,
        max_tracking_duration_ms=21600000,
        mae_price=89500.0,
        mfe_price=92100.0,
        mae_atr_multiple=0.8,
        mfe_atr_multiple=2.2,
        status=ShadowOrderStatus.HIT_TP,
    )
    mock_runner_dependencies["shadow_tracker"].on_1m_bar.return_value = [concluded_record]

    mock_case = EpisodicCase(
        case_id="case_123",
        symbol="BTCUSDT",
        timestamp_ms=1731162600000,
        vector_16d=[0.1] * 16,
        outcome=OutcomeCategory.VALIDATED_DEFENSE,
        confidence_score=0.75,
        lesson_summary="Test tactical lesson",
        playbook="TREND_CONTINUATION_BREAKOUT",
    )
    mock_runner_dependencies["judge_adapter"].conduct_autopsy.return_value = mock_case

    cfg = make_test_runner_config()
    runner = LivePaperRunner(
        config=cfg,
        db_manager=mock_runner_dependencies["db_manager"],
        memory_repo=mock_runner_dependencies["memory_repo"],
        gateway=mock_runner_dependencies["gateway"],
        ws_ingestor=mock_runner_dependencies["ws_ingestor"],
        bar_pipeline=mock_runner_dependencies["bar_pipeline"],
        shadow_tracker=mock_runner_dependencies["shadow_tracker"],
        judge_adapter=mock_runner_dependencies["judge_adapter"],
        use_case=mock_runner_dependencies["use_case"],
        trading_logger=mock_runner_dependencies["trading_logger"],
        agent_logger=mock_runner_dependencies["agent_logger"],
        system_logger=mock_runner_dependencies["system_logger"],
    )

    bar = make_test_bar("BTCUSDT", 1731162600000, 92050.0)
    await runner.handle_bar_closed(bar)

    mock_runner_dependencies["judge_adapter"].conduct_autopsy.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_cycle_ready_dispatches_trading_cycle(
    mock_runner_dependencies: dict[str, Any],
) -> None:
    now_dt = datetime.now(tz=UTC)
    mock_summary = TradingCycleSummary(
        cycle_timestamp_ms=int(now_dt.timestamp() * 1000),
        btc_result=SymbolCycleResult(
            symbol="BTCUSDT",
            funnel_passed=True,
            funnel_stage="STAGE_C_PLAYBOOK",
            thesis=None,
            blueprint=None,
            verdict=None,
            order_id="order_btc_1",
            is_shadow=False,
            rejection_reason=None,
        ),
        altcoin_results={},
        executed_orders_count=1,
        shadow_orders_count=0,
    )
    mock_runner_dependencies["use_case"].execute_cycle.return_value = mock_summary

    cfg = make_test_runner_config()
    runner = LivePaperRunner(
        config=cfg,
        db_manager=mock_runner_dependencies["db_manager"],
        memory_repo=mock_runner_dependencies["memory_repo"],
        gateway=mock_runner_dependencies["gateway"],
        ws_ingestor=mock_runner_dependencies["ws_ingestor"],
        bar_pipeline=mock_runner_dependencies["bar_pipeline"],
        shadow_tracker=mock_runner_dependencies["shadow_tracker"],
        judge_adapter=mock_runner_dependencies["judge_adapter"],
        use_case=mock_runner_dependencies["use_case"],
        trading_logger=mock_runner_dependencies["trading_logger"],
        agent_logger=mock_runner_dependencies["agent_logger"],
        system_logger=mock_runner_dependencies["system_logger"],
    )

    await runner.handle_cycle_ready(
        cycle_time_utc=now_dt,
        market_inputs={},
        quant_packets={},
        current_prices={},
        atr_30m_values={},
    )

    assert len(runner.cycle_history) == 1
    assert runner.cycle_history[0].executed_orders_count == 1
    mock_runner_dependencies["use_case"].execute_cycle.assert_awaited_once()


@pytest.mark.asyncio
async def test_runner_start_and_graceful_stop(
    mock_runner_dependencies: dict[str, Any],
) -> None:
    # 0 duration minutes for immediate exit in loop
    cfg = make_test_runner_config(duration_minutes=0)
    runner = LivePaperRunner(
        config=cfg,
        db_manager=mock_runner_dependencies["db_manager"],
        memory_repo=mock_runner_dependencies["memory_repo"],
        gateway=mock_runner_dependencies["gateway"],
        ws_ingestor=mock_runner_dependencies["ws_ingestor"],
        bar_pipeline=mock_runner_dependencies["bar_pipeline"],
        shadow_tracker=mock_runner_dependencies["shadow_tracker"],
        judge_adapter=mock_runner_dependencies["judge_adapter"],
        use_case=mock_runner_dependencies["use_case"],
        trading_logger=mock_runner_dependencies["trading_logger"],
        agent_logger=mock_runner_dependencies["agent_logger"],
        system_logger=mock_runner_dependencies["system_logger"],
    )

    await runner.start()

    assert runner.is_running is False
    mock_runner_dependencies["db_manager"].initialize.assert_awaited_once()
    mock_runner_dependencies["gateway"].sync_time_offset.assert_awaited_once()
    mock_runner_dependencies["ws_ingestor"].start.assert_awaited_once()
    mock_runner_dependencies["ws_ingestor"].stop.assert_awaited_once()
    mock_runner_dependencies["db_manager"].close.assert_awaited_once()
