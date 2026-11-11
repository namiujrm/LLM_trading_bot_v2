"""Unit tests for LivePaperRunner daemon and multi-stream orchestrator (Milestone 1).

Enforces strictly ZERO default parameters, ZERO class field defaults, and ZERO silent fallbacks.
Tests:
- Fallback models completely eliminated from runtime interface.
- Injection of real ML model ports into LivePaperRunner.__init__.
- Zero-downtime hot-reloading via reload_active_model().
- Fail-Safe Capital Protection in run_30m_cycle(): STAGE_C_ML_ERROR on inference failure,
  zero trades executed, and emergency alert dispatched to Telegram.
"""

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
from trading_bot.domain.entities.ml.model_registry_entities import (
    ModelArtifactMetadata,
    ModelRole,
)
from trading_bot.infrastructure.config_schemas import LoggingConfig
from trading_bot.infrastructure.logging.bot_logger import (
    LogCategory,
    StructuredBotLogger,
)
from trading_bot.interfaces.live_paper_runner import (
    LivePaperRunner,
    LivePaperRunnerConfig,
)
from trading_bot.ports.ml_model_port import (
    CausalBarrierModelPort,
    OODDetectorPort,
    TreeSHAPExplainerPort,
)
from trading_bot.ports.telegram_port import TelegramNotifierPort


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


def make_test_runner_config(duration_minutes: int | None) -> LivePaperRunnerConfig:
    return LivePaperRunnerConfig(
        symbols=["BTCUSDT", "ETHUSDT"],
        duration_minutes=duration_minutes,
        testnet=True,
        log_dir="logs",
        max_tracking_duration_ms=21600000,
        check_interval_seconds=0.05,
    )


def make_test_model_metadata(model_id: str, role: ModelRole) -> ModelArtifactMetadata:
    return ModelArtifactMetadata(
        model_id=model_id,
        role=role,
        feature_names=["f1", "f2", "f3"],
        trained_at_utc=datetime.now(tz=UTC),
        training_sample_count=1000,
        net_sharpe=2.1,
        brier_score=0.18,
        brier_skill_score=0.12,
        max_drawdown_pct=0.05,
        file_checksum="sha256_mock_checksum",
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
    bar_pipeline.reload_models = MagicMock()

    shadow_tracker = MagicMock(spec=ShadowOrderTracker)
    shadow_tracker.on_1m_bar = MagicMock(return_value=[])

    judge_adapter = AsyncMock()
    use_case = AsyncMock(spec=Run30mTradingCycleUseCase)

    barrier_model = MagicMock(spec=CausalBarrierModelPort)
    ood_detector = MagicMock(spec=OODDetectorPort)
    shap_explainer = MagicMock(spec=TreeSHAPExplainerPort)
    metadata = make_test_model_metadata("mock_champion_v1", ModelRole.CHAMPION)
    telegram_notifier = AsyncMock(spec=TelegramNotifierPort)
    telegram_notifier.send_emergency_alert = AsyncMock(return_value=True)

    return {
        "db_manager": db_manager,
        "memory_repo": memory_repo,
        "gateway": gateway,
        "ws_ingestor": ws_ingestor,
        "bar_pipeline": bar_pipeline,
        "shadow_tracker": shadow_tracker,
        "judge_adapter": judge_adapter,
        "use_case": use_case,
        "barrier_model": barrier_model,
        "ood_detector": ood_detector,
        "shap_explainer": shap_explainer,
        "model_metadata": metadata,
        "telegram_notifier": telegram_notifier,
        "trading_logger": trading_logger,
        "agent_logger": agent_logger,
        "system_logger": system_logger,
    }


def create_test_runner(
    cfg: LivePaperRunnerConfig,
    deps: dict[str, Any],
) -> LivePaperRunner:
    return LivePaperRunner(
        config=cfg,
        db_manager=deps["db_manager"],
        memory_repo=deps["memory_repo"],
        gateway=deps["gateway"],
        ws_ingestor=deps["ws_ingestor"],
        bar_pipeline=deps["bar_pipeline"],
        shadow_tracker=deps["shadow_tracker"],
        judge_adapter=deps["judge_adapter"],
        use_case=deps["use_case"],
        barrier_model=deps["barrier_model"],
        ood_detector=deps["ood_detector"],
        shap_explainer=deps["shap_explainer"],
        model_metadata=deps["model_metadata"],
        telegram_notifier=deps["telegram_notifier"],
        trading_logger=deps["trading_logger"],
        agent_logger=deps["agent_logger"],
        system_logger=deps["system_logger"],
        user_data_stream=None,
    )


def test_fallback_models_completely_removed() -> None:
    """Verify that heuristic fallback classes are completely removed from live_paper_runner."""
    import trading_bot.interfaces.live_paper_runner as runner_module

    assert not hasattr(runner_module, "FallbackBarrierModel")
    assert not hasattr(runner_module, "FallbackOODDetector")
    assert not hasattr(runner_module, "FallbackTreeSHAPExplainer")


def test_live_paper_runner_init(mock_runner_dependencies: dict[str, Any]) -> None:
    cfg = make_test_runner_config(duration_minutes=1)
    runner = create_test_runner(cfg, mock_runner_dependencies)

    assert runner.is_running is False
    assert len(runner.cycle_history) == 0
    assert runner.barrier_model is mock_runner_dependencies["barrier_model"]
    assert runner.ood_detector is mock_runner_dependencies["ood_detector"]
    assert runner.shap_explainer is mock_runner_dependencies["shap_explainer"]
    assert runner.active_model_metadata is mock_runner_dependencies["model_metadata"]
    assert runner.telegram_notifier is mock_runner_dependencies["telegram_notifier"]


def test_runner_reload_active_model(mock_runner_dependencies: dict[str, Any]) -> None:
    """Test hot-reloading active model suite in-memory with zero downtime."""
    cfg = make_test_runner_config(duration_minutes=1)
    runner = create_test_runner(cfg, mock_runner_dependencies)

    new_barrier_model = MagicMock(spec=CausalBarrierModelPort)
    new_ood_detector = MagicMock(spec=OODDetectorPort)
    new_shap_explainer = MagicMock(spec=TreeSHAPExplainerPort)
    new_metadata = make_test_model_metadata("promoted_challenger_v2", ModelRole.CHAMPION)

    runner.reload_active_model(
        new_barrier_model=new_barrier_model,
        new_ood_detector=new_ood_detector,
        new_shap_explainer=new_shap_explainer,
        new_metadata=new_metadata,
    )

    assert runner.barrier_model is new_barrier_model
    assert runner.ood_detector is new_ood_detector
    assert runner.shap_explainer is new_shap_explainer
    assert runner.active_model_metadata is new_metadata
    mock_runner_dependencies["bar_pipeline"].reload_models.assert_called_once_with(
        barrier_model=new_barrier_model,
        ood_detector=new_ood_detector,
        treeshap_explainer=new_shap_explainer,
    )


@pytest.mark.asyncio
async def test_run_30m_cycle_fail_safe_capital_protection_on_error(
    mock_runner_dependencies: dict[str, Any],
) -> None:
    """Test Fail-Safe Capital Protection: STAGE_C_ML_ERROR, zero trades, Telegram alert."""
    mock_runner_dependencies["use_case"].execute_cycle.side_effect = RuntimeError(
        "Corrupted booster weights or mathematical divergence"
    )

    cfg = make_test_runner_config(duration_minutes=1)
    runner = create_test_runner(cfg, mock_runner_dependencies)

    now_dt = datetime.now(tz=UTC)
    market_inputs: dict[str, Any] = {"BTCUSDT": MagicMock(), "ETHUSDT": MagicMock()}

    summary = await runner.run_30m_cycle(
        cycle_time_utc=now_dt,
        market_inputs=market_inputs,
        quant_packets={},
        current_prices={},
        atr_30m_values={},
    )

    # 1. 100% capital protection: zero trades executed, zero shadow orders
    assert summary.executed_orders_count == 0
    assert summary.shadow_orders_count == 0

    # 2. Rejection reason is strictly STAGE_C_ML_ERROR and funnel_passed is False
    assert summary.btc_result.funnel_passed is False
    assert summary.btc_result.rejection_reason == "STAGE_C_ML_ERROR"
    assert summary.btc_result.funnel_stage == "STAGE_C_PLAYBOOK"

    assert "ETHUSDT" in summary.altcoin_results
    assert summary.altcoin_results["ETHUSDT"].funnel_passed is False
    assert summary.altcoin_results["ETHUSDT"].rejection_reason == "STAGE_C_ML_ERROR"

    # 3. Emergency alert dispatched to Telegram
    mock_runner_dependencies["telegram_notifier"].send_emergency_alert.assert_awaited_once()

    # 4. Cycle recorded in history
    assert len(runner.cycle_history) == 1
    assert runner.cycle_history[0] is summary


@pytest.mark.asyncio
async def test_handle_bar_closed_no_concluded_orders(
    mock_runner_dependencies: dict[str, Any],
) -> None:
    cfg = make_test_runner_config(duration_minutes=1)
    runner = create_test_runner(cfg, mock_runner_dependencies)

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

    cfg = make_test_runner_config(duration_minutes=1)
    runner = create_test_runner(cfg, mock_runner_dependencies)

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

    cfg = make_test_runner_config(duration_minutes=1)
    runner = create_test_runner(cfg, mock_runner_dependencies)

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
    cfg = make_test_runner_config(duration_minutes=0)
    runner = create_test_runner(cfg, mock_runner_dependencies)

    await runner.start()

    assert runner.is_running is False
    mock_runner_dependencies["db_manager"].initialize.assert_awaited_once()
    mock_runner_dependencies["gateway"].sync_time_offset.assert_awaited_once()
    mock_runner_dependencies["ws_ingestor"].start.assert_awaited_once()
    mock_runner_dependencies["ws_ingestor"].stop.assert_awaited_once()
    mock_runner_dependencies["db_manager"].close.assert_awaited_once()


@pytest.mark.asyncio
async def test_runner_handle_bracket_leg_filled_cancels_opposing(
    mock_runner_dependencies: dict[str, Any],
) -> None:
    """Verify that handle_bracket_leg_filled issues cancellation for opposing leg."""
    cfg = make_test_runner_config(duration_minutes=None)
    runner = create_test_runner(cfg, mock_runner_dependencies)
    gateway = mock_runner_dependencies["gateway"]
    gateway.cancel_algo_order = AsyncMock(return_value=True)

    await runner.handle_bracket_leg_filled("BTCUSDT", "SL", "order_123", "order_123_tp")
    gateway.cancel_algo_order.assert_awaited_once_with("BTCUSDT", "order_123_tp")


@pytest.mark.asyncio
async def test_runner_reconcile_orphan_bracket_orders(
    mock_runner_dependencies: dict[str, Any],
) -> None:
    """Verify that reconcile_orphan_bracket_orders cleans up unlinked algo orders for flat positions."""
    cfg = make_test_runner_config(duration_minutes=None)
    runner = create_test_runner(cfg, mock_runner_dependencies)
    gateway = mock_runner_dependencies["gateway"]
    gateway.fetch_active_positions = AsyncMock(return_value=[])
    gateway.fetch_open_algo_orders = AsyncMock(
        return_value=[{"algoId": 12345, "clientAlgoId": "order_tp"}]
    )
    gateway.cancel_algo_order = AsyncMock(return_value=True)

    await runner.reconcile_orphan_bracket_orders()
    assert gateway.cancel_algo_order.await_count >= 1
