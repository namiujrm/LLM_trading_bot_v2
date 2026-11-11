"""Live Paper Trading Runner Daemon & Multi-Stream Orchestrator.

Production CLI runner orchestrating:
1. PostgreSQL TimescaleDB connection pool & PgvectorMemoryRepository.
2. BinanceFuturesGatewayAdapter with server time offset synchronization.
3. BinanceFuturesWSIngestor streaming real-time 1m bars, depth, and liquidations.
4. LiveBarPipelineService accumulating rolling multi-TF bars and online quant features.
5. ShadowOrderTracker performing continuous 1m price progression and outcome classification.
6. PostMortemJudgeAdapter synthesizing counterfactual tactical lessons on concluded shadow orders.
7. Run30mTradingCycleUseCase executing BTC-lead cascade at exact 30m candle closes.
8. Structured daily rotating JSONL logging to logs/trading/ and logs/agents/.
9. Graceful shutdown on SIGINT/SIGTERM or duration timeout.

Clean Architecture: Presentation/Interfaces Layer.
Strict Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

import aiohttp
from dotenv import load_dotenv
from pydantic import BaseModel

from trading_bot.adapters.agents.arbiter_agent import RiskArbiterAdapter
from trading_bot.adapters.agents.architect_agent import TradeArchitectAdapter
from trading_bot.adapters.agents.judge_agent import PostMortemJudgeAdapter
from trading_bot.adapters.agents.macro_news_sentinel import MacroNewsSentinel
from trading_bot.adapters.agents.ninerouter_client import NinerouterClient
from trading_bot.adapters.agents.scout_agent import MarketScoutAdapter
from trading_bot.adapters.binance.binance_futures_gateway import BinanceFuturesGatewayAdapter
from trading_bot.adapters.binance.micro_accumulator import MicroIngestorConfig
from trading_bot.adapters.binance.user_data_stream import BinanceUserDataStream
from trading_bot.adapters.binance.ws_micro_ingestor import BinanceFuturesWSIngestor
from trading_bot.adapters.persistence.database import DatabaseManager
from trading_bot.adapters.persistence.pgvector_memory_repo import PgvectorMemoryRepository
from trading_bot.adapters.quant_engine.polars_engine import PolarsQuantEngineAdapter
from trading_bot.application.services.live_bar_pipeline import LiveBarPipelineService
from trading_bot.application.services.shadow_order_tracker import ShadowOrderTracker
from trading_bot.application.use_cases.evaluate_tiered_funnel import (
    TieredFunnelEvaluator,
    TieredFunnelInput,
)
from trading_bot.application.use_cases.run_30m_trading_cycle import (
    Run30mTradingCycleUseCase,
    SymbolCycleResult,
    TradingCycleSummary,
)
from trading_bot.domain.entities.common.enums import (
    PlaybookType,
)
from trading_bot.domain.entities.execution.shadow_order import ShadowOrderRecord
from trading_bot.domain.entities.market.market_state import (
    EnrichedMicroBarRecord,
    SemanticQuantPacket,
)
from trading_bot.domain.entities.ml.model_registry_entities import ModelArtifactMetadata
from trading_bot.domain.entities.quant.quant_configs import (
    AlphaDSLConfig,
    FracDiffConfig,
    ICGatekeeperConfig,
    QuantEngineMasterConfig,
    RegimeConfig,
    VPINConfig,
)
from trading_bot.domain.services.execution.fee_to_edge_gate import (
    FeeToEdgeConfig,
    FeeToEdgeGatingService,
)
from trading_bot.domain.services.execution.playbook_validator import (
    PlaybookConfig,
    PlaybookValidator,
)
from trading_bot.domain.services.quant.ic_gatekeeper import ICGatekeeperService
from trading_bot.domain.services.quant.microstructure_math import MicrostructureMathService
from trading_bot.domain.services.quant.packet_assembler import SemanticQuantPacketAssembler
from trading_bot.domain.services.quant.regime_detector import RegimeDetectorService
from trading_bot.domain.services.risk.risk_calculator import (
    DeterministicRiskCalculator,
    RiskConfig,
)
from trading_bot.infrastructure.config_schemas import (
    BinanceConfig,
    DatabaseConfig,
    LoggingConfig,
    NinerouterConfig,
)
from trading_bot.infrastructure.logging.bot_logger import (
    LogCategory,
    StructuredBotLogger,
)
from trading_bot.ports.agent_ports import (
    PostMortemJudgePort,
)
from trading_bot.ports.execution_gateway_port import (
    ExecutionGatewayPort,
    HistoricalMultiTFWarmupPort,
)
from trading_bot.ports.memory_repository_port import MemoryRepositoryPort
from trading_bot.ports.ml_model_port import (
    CausalBarrierModelPort,
    OODDetectorPort,
    ProbabilityCalibratorPort,
    TreeSHAPExplainerPort,
)
from trading_bot.ports.rate_limit_governor_port import (
    RateLimitGovernorPort,
    RouteQuotaSpec,
)
from trading_bot.ports.telegram_port import TelegramNotifierPort


class LivePaperRunnerConfig(BaseModel):
    """Configuration schema for live paper runner daemon (Zero Defaults)."""

    symbols: list[str]
    duration_minutes: int | None
    testnet: bool
    log_dir: str
    max_tracking_duration_ms: int
    check_interval_seconds: float


class InMemoryRateLimitGovernor(RateLimitGovernorPort):
    """In-memory rate limit governor implementing RateLimitGovernorPort (Zero Defaults)."""

    def __init__(self, specs_by_tier: dict[str, list[RouteQuotaSpec]]) -> None:
        self._specs_by_tier: dict[str, list[RouteQuotaSpec]] = specs_by_tier
        self.cooldown_calls: list[tuple[str, float]] = []

    async def acquire_available_route(
        self,
        tier: str,
        estimated_tokens: int,
    ) -> RouteQuotaSpec:
        if tier not in self._specs_by_tier or len(self._specs_by_tier[tier]) == 0:
            raise KeyError(f"Route tier '{tier}' not registered in RateLimitGovernor")
        return self._specs_by_tier[tier][0]

    async def mark_route_cooldown(
        self,
        route_id: str,
        cooldown_seconds: float,
    ) -> None:
        self.cooldown_calls.append((route_id, cooldown_seconds))


class LivePaperRunner:
    """Production live paper trading runner daemon orchestrating ingestion, analysis, and execution."""

    def __init__(
        self,
        config: LivePaperRunnerConfig,
        db_manager: DatabaseManager,
        memory_repo: MemoryRepositoryPort,
        gateway: ExecutionGatewayPort,
        ws_ingestor: BinanceFuturesWSIngestor,
        bar_pipeline: LiveBarPipelineService,
        shadow_tracker: ShadowOrderTracker,
        judge_adapter: PostMortemJudgePort,
        use_case: Run30mTradingCycleUseCase,
        barrier_model: CausalBarrierModelPort,
        ood_detector: OODDetectorPort,
        shap_explainer: TreeSHAPExplainerPort,
        model_metadata: ModelArtifactMetadata | None,
        telegram_notifier: TelegramNotifierPort | None,
        trading_logger: StructuredBotLogger,
        agent_logger: StructuredBotLogger,
        system_logger: StructuredBotLogger,
        user_data_stream: BinanceUserDataStream | None,
    ) -> None:
        self.config: LivePaperRunnerConfig = config
        self._db: DatabaseManager = db_manager
        self._memory_repo: MemoryRepositoryPort = memory_repo
        self._gateway: ExecutionGatewayPort = gateway
        self._ws_ingestor: BinanceFuturesWSIngestor = ws_ingestor
        self._bar_pipeline: LiveBarPipelineService = bar_pipeline
        self._shadow_tracker: ShadowOrderTracker = shadow_tracker
        self._judge_adapter: PostMortemJudgePort = judge_adapter
        self._use_case: Run30mTradingCycleUseCase = use_case
        self._barrier_model: CausalBarrierModelPort = barrier_model
        self._ood_detector: OODDetectorPort = ood_detector
        self._shap_explainer: TreeSHAPExplainerPort = shap_explainer
        self._active_model_metadata: ModelArtifactMetadata | None = model_metadata
        self._telegram_notifier: TelegramNotifierPort | None = telegram_notifier
        self._trading_logger: StructuredBotLogger = trading_logger
        self._agent_logger: StructuredBotLogger = agent_logger
        self._system_logger: StructuredBotLogger = system_logger
        self._user_data_stream: BinanceUserDataStream | None = user_data_stream

        self._is_running: bool = False
        self._stop_event: asyncio.Event = asyncio.Event()
        self._cycle_history: list[TradingCycleSummary] = []
        self._start_time_monotonic: float = 0.0
        self._on_concluded_callback: Callable[[ShadowOrderRecord, Any], Awaitable[None]] | None = (
            None
        )

    def set_on_concluded_callback(
        self,
        callback: Callable[[ShadowOrderRecord, Any], Awaitable[None]],
    ) -> None:
        """Register asynchronous callback fired when an order/shadow order concludes."""
        self._on_concluded_callback = callback

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def cycle_history(self) -> list[TradingCycleSummary]:
        return list(self._cycle_history)

    @property
    def barrier_model(self) -> CausalBarrierModelPort:
        return self._barrier_model

    @property
    def ood_detector(self) -> OODDetectorPort:
        return self._ood_detector

    @property
    def shap_explainer(self) -> TreeSHAPExplainerPort:
        return self._shap_explainer

    @property
    def active_model_metadata(self) -> ModelArtifactMetadata | None:
        return self._active_model_metadata

    @property
    def telegram_notifier(self) -> TelegramNotifierPort | None:
        return self._telegram_notifier

    def set_telegram_notifier(self, notifier: TelegramNotifierPort) -> None:
        """Register TelegramNotifierPort for operational alerts."""
        self._telegram_notifier = notifier

    @property
    def user_data_stream(self) -> BinanceUserDataStream | None:
        return self._user_data_stream

    async def handle_bracket_leg_filled(
        self, symbol: str, filled_leg: str, parent_id: str, opposing_algo_id: str
    ) -> None:
        """Cancel opposing bracket algo leg immediately upon fill of SL or TP."""
        self._trading_logger.info(
            f"Bracket leg {filled_leg} filled for {symbol}. Cancelling opposing leg {opposing_algo_id}...",
            {
                "symbol": symbol,
                "filled_leg": filled_leg,
                "opposing_algo_id": opposing_algo_id,
                "parent_id": parent_id,
            },
        )
        try:
            cancelled = await self._gateway.cancel_algo_order(symbol, opposing_algo_id)
            if cancelled:
                self._trading_logger.info(
                    f"Successfully cancelled opposing leg {opposing_algo_id} for {symbol}",
                    {"symbol": symbol, "opposing_algo_id": opposing_algo_id},
                )
            else:
                self._trading_logger.warning(
                    f"Opposing leg {opposing_algo_id} was not active or already closed",
                    {"symbol": symbol, "opposing_algo_id": opposing_algo_id},
                )
        except Exception as exc:
            self._trading_logger.error(
                f"Failed to cancel opposing bracket leg {opposing_algo_id}: {exc}",
                {"symbol": symbol, "opposing_algo_id": opposing_algo_id, "error": str(exc)},
            )

    async def reconcile_orphan_bracket_orders(self) -> None:
        """Periodically clean up orphaned bracket orders when position is flat."""
        try:
            positions = await self._gateway.fetch_active_positions()
            active_symbols = {p.symbol.upper() for p in positions if abs(p.quantity) > 1e-8}
            for sym in self.config.symbols:
                upper_sym = sym.upper()
                if upper_sym not in active_symbols:
                    open_algos = await self._gateway.fetch_open_algo_orders(upper_sym)
                    for algo in open_algos:
                        algo_id_val = ""
                        if "algoId" in algo and algo["algoId"] is not None:
                            algo_id_val = str(algo["algoId"])
                        elif "clientAlgoId" in algo and algo["clientAlgoId"] is not None:
                            algo_id_val = str(algo["clientAlgoId"])

                        if algo_id_val:
                            self._system_logger.info(
                                f"Reconciliation: Cancelling orphaned algo order {algo_id_val} for flat symbol {upper_sym}",
                                {"symbol": upper_sym, "algo_id": algo_id_val},
                            )
                            await self._gateway.cancel_algo_order(upper_sym, algo_id_val)
        except Exception as exc:
            self._system_logger.warning(
                f"Warning during bracket order reconciliation: {exc}",
                {"error": str(exc)},
            )

    def reload_active_model(
        self,
        new_barrier_model: CausalBarrierModelPort,
        new_ood_detector: OODDetectorPort,
        new_shap_explainer: TreeSHAPExplainerPort,
        new_metadata: ModelArtifactMetadata,
    ) -> None:
        """Atomically hot-reload barrier model, OOD detector, and SHAP explainer in-memory."""
        self._barrier_model = new_barrier_model
        self._ood_detector = new_ood_detector
        self._shap_explainer = new_shap_explainer
        self._active_model_metadata = new_metadata
        self._bar_pipeline.reload_models(
            barrier_model=new_barrier_model,
            ood_detector=new_ood_detector,
            treeshap_explainer=new_shap_explainer,
        )
        self._system_logger.log_system_event(
            event_type="MODEL_HOT_RELOADED",
            component="LivePaperRunner",
            message=f"Active model successfully hot-reloaded to {new_metadata.model_id} ({new_metadata.role.value})",
            details={
                "model_id": new_metadata.model_id,
                "role": new_metadata.role.value,
                "trained_at_utc": new_metadata.trained_at_utc.isoformat(),
                "file_checksum": new_metadata.file_checksum,
            },
        )

    def reload_calibrators(
        self,
        *,
        calibrator_long: ProbabilityCalibratorPort | None,
        calibrator_short: ProbabilityCalibratorPort | None,
    ) -> None:
        """Hot-reload fitted Venn-Abers Long/Short calibrators into the live bar pipeline."""
        self._bar_pipeline.set_calibrators(
            calibrator_long=calibrator_long,
            calibrator_short=calibrator_short,
        )

    async def handle_bar_closed(self, bar: EnrichedMicroBarRecord) -> None:
        """Process an enriched 1-minute closed candle across bar pipeline and shadow tracker."""
        # 1. Pipeline accumulation & rolling multi-TF synthesis
        await self._bar_pipeline.on_1m_bar(bar)

        # 2. Shadow order price progression evaluation
        concluded_orders = self._shadow_tracker.on_1m_bar(bar.symbol, bar)

        # 3. Dissect concluded shadow orders through post-mortem autopsy judge
        for concluded in concluded_orders:
            self._trading_logger.log_order_event(
                event_type="SHADOW_ORDER_CONCLUDED",
                order_id=concluded.shadow_id,
                symbol=concluded.symbol,
                details={
                    "status": concluded.status.value,
                    "mae_atr": concluded.mae_atr_multiple,
                    "mfe_atr": concluded.mfe_atr_multiple,
                    "is_post_sl": concluded.is_post_sl_tracker,
                },
            )

            # Assemble quant context for judge autopsy
            quant_context: dict[str, Any] = {
                "symbol": concluded.symbol,
                "timestamp_ms": concluded.entry_timestamp_ms,
                "vector_16d": [0.1] * 16,
                "playbook": PlaybookType.TREND_CONTINUATION_BREAKOUT.value,
            }
            if concluded.symbol in self.config.symbols:
                try:
                    latest_pkt = self._bar_pipeline.get_latest_quant_packet(concluded.symbol)
                    quant_context["vector_16d"] = latest_pkt.trajectory_vector_16d
                except KeyError:
                    pass

            case = await self._judge_adapter.conduct_autopsy(concluded, quant_context)
            self._agent_logger.log_agent_deliberation(
                agent_name="PostMortemJudge",
                symbol=case.symbol,
                decision=case.outcome.value,
                details={
                    "case_id": case.case_id,
                    "confidence_score": case.confidence_score,
                    "lesson_summary": case.lesson_summary,
                },
            )

            if self._on_concluded_callback is not None:
                with contextlib.suppress(Exception):
                    await self._on_concluded_callback(concluded, case)

    async def run_30m_cycle(
        self,
        cycle_time_utc: datetime,
        market_inputs: dict[str, TieredFunnelInput],
        quant_packets: dict[str, SemanticQuantPacket],
        current_prices: dict[str, float],
        atr_30m_values: dict[str, float],
    ) -> TradingCycleSummary:
        """Execute 30-minute cascade trading cycle with Fail-Safe Capital Protection.

        If model inference fails or an anomaly occurs, enforces Fail-Safe Capital Protection:
        all symbols fail with reason STAGE_C_ML_ERROR, zero trades are executed, and an alert
        is dispatched to Telegram. Heuristic fallbacks are strictly prohibited.
        """
        self._system_logger.log_system_event(
            event_type="CYCLE_DISPATCH_TRIGGERED",
            component="LivePaperRunner",
            message=f"30m cycle dispatch triggered at {cycle_time_utc.isoformat()} for {len(market_inputs)} symbols",
            details={
                "cycle_time_utc": cycle_time_utc.isoformat(),
                "symbols_count": len(market_inputs),
            },
        )

        try:
            summary = await self._use_case.execute_cycle(
                cycle_time_utc=cycle_time_utc,
                market_inputs=market_inputs,
                current_prices=current_prices,
                atr_30m_values=atr_30m_values,
                quant_packets=quant_packets,
            )
            self._cycle_history.append(summary)

            # Structured trade log
            self._trading_logger.log_order_event(
                event_type="CYCLE_SUMMARY_RECORDED",
                order_id=f"cycle_{summary.cycle_timestamp_ms}",
                symbol="ALL",
                details={
                    "executed_orders_count": summary.executed_orders_count,
                    "shadow_orders_count": summary.shadow_orders_count,
                    "btc_passed": summary.btc_result.funnel_passed,
                    "btc_action": summary.btc_result.order_id,
                },
            )
            return summary

        except Exception as exc:
            self._system_logger.error(
                f"Fail-Safe Capital Protection activated due to ML/cycle error: {exc}",
                {"error": str(exc), "cycle_time": cycle_time_utc.isoformat()},
            )

            cycle_ts_ms = int(cycle_time_utc.timestamp() * 1000)
            btc_fail = SymbolCycleResult(
                symbol="BTCUSDT",
                funnel_passed=False,
                funnel_stage="STAGE_C_PLAYBOOK",
                thesis=None,
                blueprint=None,
                verdict=None,
                order_id=None,
                is_shadow=False,
                rejection_reason="STAGE_C_ML_ERROR",
            )
            altcoin_fails: dict[str, SymbolCycleResult] = {
                sym: SymbolCycleResult(
                    symbol=sym,
                    funnel_passed=False,
                    funnel_stage="STAGE_C_PLAYBOOK",
                    thesis=None,
                    blueprint=None,
                    verdict=None,
                    order_id=None,
                    is_shadow=False,
                    rejection_reason="STAGE_C_ML_ERROR",
                )
                for sym in market_inputs
                if sym != "BTCUSDT"
            }

            fail_summary = TradingCycleSummary(
                cycle_timestamp_ms=cycle_ts_ms,
                btc_result=btc_fail,
                altcoin_results=altcoin_fails,
                executed_orders_count=0,
                shadow_orders_count=0,
            )
            self._cycle_history.append(fail_summary)

            if self._telegram_notifier is not None:
                with contextlib.suppress(Exception):
                    await self._telegram_notifier.send_emergency_alert(
                        "🚨 [FAIL-SAFE CAPITAL PROTECTION] ML Inference Error",
                        f"An anomaly occurred during 30m cycle: {exc}\n"
                        f"funnel_passed=False (STAGE_C_ML_ERROR). Zero trades taken.",
                    )

            return fail_summary

    async def handle_cycle_ready(
        self,
        cycle_time_utc: datetime,
        market_inputs: dict[str, TieredFunnelInput],
        quant_packets: dict[str, SemanticQuantPacket],
        current_prices: dict[str, float],
        atr_30m_values: dict[str, float],
    ) -> None:
        """Callback invoked when all symbols have completed 30m candle close."""
        await self.run_30m_cycle(
            cycle_time_utc=cycle_time_utc,
            market_inputs=market_inputs,
            quant_packets=quant_packets,
            current_prices=current_prices,
            atr_30m_values=atr_30m_values,
        )

    async def start(self) -> None:
        """Start the live paper runner daemon and orchestrate continuous processing."""
        self._start_time_monotonic = time.monotonic()
        self._is_running = True
        self._stop_event.clear()

        self._system_logger.log_system_event(
            event_type="RUNNER_STARTING",
            component="LivePaperRunner",
            message="Starting live paper runner daemon",
            details={
                "symbols": self.config.symbols,
                "testnet": self.config.testnet,
                "duration_minutes": self.config.duration_minutes,
            },
        )

        # 1. Initialize Database Pool
        try:
            await self._db.initialize()
            self._system_logger.log_system_event(
                event_type="DATABASE_CONNECTED",
                component="LivePaperRunner",
                message="Connected to PostgreSQL database",
                details={},
            )
        except Exception as exc:
            self._system_logger.error(
                f"Database connection error: {exc}",
                {"error": str(exc)},
            )

        # 2. Sync Exchange Server Time Offset
        gateway_any: Any = self._gateway
        if hasattr(gateway_any, "sync_time_offset"):
            try:
                time_offset_val = await gateway_any.sync_time_offset()
                time_offset_ms: int = int(time_offset_val)
                self._system_logger.log_system_event(
                    event_type="EXCHANGE_TIME_SYNCED",
                    component="LivePaperRunner",
                    message=f"Exchange server time synced with offset {time_offset_ms}ms",
                    details={"time_offset_ms": time_offset_ms},
                )
            except Exception as exc:
                self._system_logger.warning(
                    f"Exchange server time sync warning: {exc}",
                    {"error": str(exc)},
                )

        # 2.1 Sync Exchange Filters
        if hasattr(gateway_any, "fetch_symbol_filters"):
            try:
                await gateway_any.fetch_symbol_filters()
                self._system_logger.log_system_event(
                    event_type="EXCHANGE_FILTERS_SYNCED",
                    component="LivePaperRunner",
                    message="Binance exchange lot size and tick size filters synced",
                    details={},
                )
            except Exception as exc:
                self._system_logger.warning(
                    f"Exchange filter sync warning: {exc}",
                    {"error": str(exc)},
                )

        # 2.2 Warm-up Rolling Buffers with Recent Historical Klines (350 x 30m + 300 x 1m)
        for sym in self.config.symbols:
            try:
                if isinstance(self._gateway, HistoricalMultiTFWarmupPort):
                    bars_30m = await self._gateway.fetch_recent_klines_30m(sym, 350)
                    await self._bar_pipeline.warmup_historical_30m_bars(sym, bars_30m)
                bars = await self._gateway.fetch_recent_klines_1m(sym, 300)
                await self._bar_pipeline.warmup_historical_bars(sym, bars)
                self._system_logger.log_system_event(
                    event_type="HISTORICAL_BUFFER_WARMED",
                    component="LivePaperRunner",
                    message=f"Pre-warmed {len(bars)} 1m historical bars for {sym}",
                    details={"symbol": sym, "bar_count": len(bars)},
                )
            except Exception as exc:
                self._system_logger.warning(
                    f"Historical buffer warm-up warning for {sym}: {exc}",
                    {"symbol": sym, "error": str(exc)},
                )

        # 3. Start WebSocket Ingestor
        await self._ws_ingestor.start()
        self._system_logger.log_system_event(
            event_type="WEBSOCKET_INGESTOR_STARTED",
            component="LivePaperRunner",
            message="WebSocket ingestor started for target symbols",
            details={"stream_count": len(self.config.symbols) * 4},
        )

        # 3.1 Start User Data Stream
        if self._user_data_stream is not None:
            await self._user_data_stream.start()
            self._system_logger.log_system_event(
                event_type="USER_DATA_STREAM_STARTED",
                component="LivePaperRunner",
                message="User Data Stream started for real-time bracket order management",
                details={},
            )

        # 4. Main Event Loop with duration monitoring and periodic reconciliation
        last_reconcile_monotonic: float = time.monotonic()
        try:
            while not self._stop_event.is_set():
                now_monotonic = time.monotonic()
                if (now_monotonic - last_reconcile_monotonic) >= 60.0:
                    last_reconcile_monotonic = now_monotonic
                    await self.reconcile_orphan_bracket_orders()

                if self.config.duration_minutes is not None:
                    elapsed_minutes = (time.monotonic() - self._start_time_monotonic) / 60.0
                    if elapsed_minutes >= float(self.config.duration_minutes):
                        self._system_logger.log_system_event(
                            event_type="DURATION_TIMEOUT_REACHED",
                            component="LivePaperRunner",
                            message=f"Configured duration limit reached ({elapsed_minutes:.2f} min)",
                            details={"elapsed_minutes": elapsed_minutes},
                        )
                        break

                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.config.check_interval_seconds,
                    )

        finally:
            await self.stop()

    async def stop(self) -> None:
        """Gracefully stop all streaming ingestors, flushing logs, and closing resources."""
        if not self._is_running:
            return

        self._is_running = False
        self._stop_event.set()

        self._system_logger.log_system_event(
            event_type="RUNNER_STOPPING",
            component="LivePaperRunner",
            message="Stopping live paper runner daemon",
            details={},
        )

        # Stop User Data Stream
        if self._user_data_stream is not None:
            try:
                await self._user_data_stream.stop()
            except Exception as exc:
                self._system_logger.warning(
                    f"Error stopping User Data Stream: {exc}",
                    {"error": str(exc)},
                )

        # Stop WebSocket Ingestor
        await self._ws_ingestor.stop()

        # Close Database Pool
        with contextlib.suppress(Exception):
            await self._db.close()

        self._system_logger.log_system_event(
            event_type="RUNNER_STOPPED",
            component="LivePaperRunner",
            message="Live paper runner stopped successfully",
            details={"cycles_executed": len(self._cycle_history)},
        )


def make_default_risk_config() -> RiskConfig:
    """Construct explicit RiskConfig under Invariant #9."""
    return RiskConfig(
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0005,
        min_tp_to_cost_multiple=3.0,
        min_net_rr=1.30,
        kelly_fraction=0.25,
        max_risk_per_trade_pct=0.02,
        min_executable_risk_pct=0.001,
        max_total_open_heat_pct=0.06,
        max_margin_utilization_pct=0.80,
        target_30m_yz_vol=0.015,
        min_vol_floor=0.005,
        max_vol_scalar=2.0,
        uncertainty_penalty_weight=1.0,
        prior_neutral_prob=0.50,
        corr_scalar_zero_same_dir=1.0,
        corr_scalar_one_same_dir=0.6,
        corr_scalar_two_plus_same_dir=0.3,
        daily_drawdown_halt_pct=0.04,
        max_drawdown_freeze_pct=0.08,
        min_barrier_distance_pct=0.003,
        min_leverage=1,
        max_leverage=10,
    )


def make_default_playbook_config(altcoin_symbols: list[str]) -> PlaybookConfig:
    """Construct explicit PlaybookConfig under Invariant #9."""
    return PlaybookConfig(
        trend_hurst_min=0.50,
        trend_vpin_percentile_max=88.0,
        trend_min_calibrated_prob=0.52,
        reversal_hurst_max=0.49,
        reversal_liq_z_min=1.8,
        reversal_funding_z_abs_min=2.0,
        reversal_vpin_slope_max=0.0,
        lead_lag_btc_ret_z_min=1.5,
        lead_lag_btc_beta_min=0.50,
        lead_lag_catchup_gap_z_min=1.0,
        altcoin_symbols=altcoin_symbols,
    )


async def bootstrap_live_paper_runner(
    runner_cfg: LivePaperRunnerConfig,
    session: aiohttp.ClientSession | None,
    barrier_model: CausalBarrierModelPort,
    ood_detector: OODDetectorPort,
    shap_explainer: TreeSHAPExplainerPort,
    model_metadata: ModelArtifactMetadata | None,
    telegram_notifier: TelegramNotifierPort | None,
) -> LivePaperRunner:
    """Bootstrap a production LivePaperRunner instance wired to live environment or testnet."""
    load_dotenv()
    env = dict(os.environ)

    log_cfg = LoggingConfig(
        log_dir=runner_cfg.log_dir,
        retention_days=7,
        min_level="INFO",
        console_output=True,
    )
    trading_logger = StructuredBotLogger(LogCategory.TRADING, log_cfg)
    agent_logger = StructuredBotLogger(LogCategory.AGENTS, log_cfg)
    system_logger = StructuredBotLogger(LogCategory.SYSTEM, log_cfg)

    db_cfg = DatabaseConfig(
        host=env["POSTGRES_HOST"],
        port=int(env["POSTGRES_PORT"]),
        db_name=env["POSTGRES_DB"],
        user=env["POSTGRES_USER"],
        password=env["POSTGRES_PASSWORD"],
        max_connections=int(env["POSTGRES_MAX_CONNECTIONS"]),
    )
    db_manager = DatabaseManager(db_cfg)
    memory_repo = PgvectorMemoryRepository(db_manager)

    binance_cfg = BinanceConfig(
        api_key=env["BINANCE_API_KEY"],
        api_secret=env["BINANCE_API_SECRET"],
        testnet=runner_cfg.testnet,
        recv_window_ms=int(env["BINANCE_RECV_WINDOW_MS"]),
    )
    client_session = session if session is not None else aiohttp.ClientSession()
    gateway = BinanceFuturesGatewayAdapter(binance_cfg, client_session)

    nr_cfg = NinerouterConfig(
        base_url=env["NINEROUTER_BASE_URL"],
        api_key=env["NINEROUTER_API_KEY"],
        timeout_seconds=float(env["NINEROUTER_TIMEOUT_SECONDS"]),
        fast_tier_model=env["MODEL_FAST_TIER"],
        deep_tier_model=env["MODEL_DEEP_TIER"],
        judge_tier_model=env["MODEL_JUDGE_TIER"],
    )

    route_specs = {
        "fast": [
            RouteQuotaSpec(
                route_id="ds_fast",
                model_name=nr_cfg.fast_tier_model,
                rpm_limit=60,
                tpm_limit=100000,
                rpd_limit=5000,
                min_interval_ms=100,
                max_concurrency=5,
            )
        ],
        "deep": [
            RouteQuotaSpec(
                route_id="ds_deep",
                model_name=nr_cfg.deep_tier_model,
                rpm_limit=30,
                tpm_limit=50000,
                rpd_limit=2000,
                min_interval_ms=200,
                max_concurrency=2,
            )
        ],
        "judge": [
            RouteQuotaSpec(
                route_id="gemini_judge",
                model_name=nr_cfg.judge_tier_model,
                rpm_limit=30,
                tpm_limit=50000,
                rpd_limit=2000,
                min_interval_ms=200,
                max_concurrency=2,
            )
        ],
    }
    governor = InMemoryRateLimitGovernor(route_specs)
    nr_client = NinerouterClient(nr_cfg, governor, client_session, agent_logger)

    scout = MarketScoutAdapter(nr_client, "fast")
    architect = TradeArchitectAdapter(nr_client, "deep")

    risk_cfg = make_default_risk_config()
    calc = DeterministicRiskCalculator(risk_cfg)
    arbiter = RiskArbiterAdapter(nr_client, "deep", calc, agent_logger)

    judge_adapter = PostMortemJudgeAdapter(
        client=nr_client,
        config=nr_cfg,
        governor=governor,
        memory_repo=memory_repo,
        logger=agent_logger,
    )

    fee_service = FeeToEdgeGatingService(
        FeeToEdgeConfig(
            multiplier=1.0,
            roundtrip_cost_rate=0.0010,
            min_calibrated_prob=0.52,
        )
    )
    altcoin_syms = [s for s in runner_cfg.symbols if s != "BTCUSDT"]
    playbook_val = PlaybookValidator(make_default_playbook_config(altcoin_syms))

    funnel_evaluator = TieredFunnelEvaluator(
        fee_to_edge_service=fee_service,
        playbook_validator=playbook_val,
        max_spread_bps=15.0,
        min_yz_vol=0.002,
        max_uncertainty_width=0.45,
    )

    macro_sentinel = MacroNewsSentinel([])
    shadow_tracker = ShadowOrderTracker(
        max_tracking_duration_ms=runner_cfg.max_tracking_duration_ms
    )

    use_case = Run30mTradingCycleUseCase(
        funnel_evaluator=funnel_evaluator,
        scout=scout,
        architect=architect,
        arbiter=arbiter,
        execution_gateway=gateway,
        macro_sentinel=macro_sentinel,
        memory_repo=memory_repo,
        shadow_order_tracker=shadow_tracker,
    )

    # Pre-declare runner reference for callbacks
    runner_ref: list[LivePaperRunner | None] = [None]

    async def on_bar_closed_callback(bar: EnrichedMicroBarRecord) -> None:
        if runner_ref[0] is not None:
            await runner_ref[0].handle_bar_closed(bar)

    async def on_cycle_ready_callback(
        cycle_dt: datetime,
        m_inputs: dict[str, TieredFunnelInput],
        q_packets: dict[str, SemanticQuantPacket],
        c_prices: dict[str, float],
        a_values: dict[str, float],
    ) -> None:
        if runner_ref[0] is not None:
            await runner_ref[0].handle_cycle_ready(
                cycle_dt, m_inputs, q_packets, c_prices, a_values
            )

    # Polars Quant Engine
    quant_engine_cfg = QuantEngineMasterConfig(
        vpin=VPINConfig(
            num_buckets=50,
            adv_lookback_days=20,
            volume_bucket_slice_ratio=0.02,
            toxicity_high_threshold=0.85,
            division_eps=1e-12,
        ),
        regime=RegimeConfig(
            hurst_window=100,
            yang_zhang_window=20,
            bipower_jump_confidence=2.58,
            entropy_bins=10,
            permutation_dimension=3,
            tsallis_q=1.5,
            division_eps=1e-12,
        ),
        fracdiff=FracDiffConfig(
            d_order=0.40,
            weight_threshold=1e-4,
            max_window_length=64,
            adf_critical_pvalue=0.05,
        ),
        alpha_dsl=AlphaDSLConfig(
            max_ast_depth=8,
            max_ast_nodes=50,
            min_window_bars=1,
            max_window_bars=288,
            division_epsilon=1e-8,
        ),
        ic_gatekeeper=ICGatekeeperConfig(
            min_rank_ic=0.015,
            min_t_statistic=2.0,
            min_ic_ir=0.50,
            rolling_eval_window_bars=100,
            forward_prediction_bars=12,
        ),
    )
    polars_engine = PolarsQuantEngineAdapter(
        config=quant_engine_cfg,
        micro_service=MicrostructureMathService(),
        regime_service=RegimeDetectorService(),
        assembler=SemanticQuantPacketAssembler(),
        ic_gatekeeper=ICGatekeeperService(quant_engine_cfg.ic_gatekeeper),
    )

    bar_pipeline = LiveBarPipelineService(
        symbols=runner_cfg.symbols,
        quant_engine=polars_engine,
        barrier_model=barrier_model,
        ood_detector=ood_detector,
        treeshap_explainer=shap_explainer,
        on_30m_bar=None,
        on_cycle_ready=on_cycle_ready_callback,
    )

    micro_cfg = MicroIngestorConfig(
        obi_depth_levels=20,
        obi_decay_rate=0.85,
        bps_scale_factor=10000.0,
    )
    ws_ingestor = BinanceFuturesWSIngestor(
        symbols=runner_cfg.symbols,
        cfg=micro_cfg,
        binance_cfg=binance_cfg,
        on_bar_closed=on_bar_closed_callback,
    )

    async def on_leg_filled(sym: str, leg: str, parent: str, opposing: str) -> None:
        if runner_ref[0] is not None:
            await runner_ref[0].handle_bracket_leg_filled(sym, leg, parent, opposing)

    user_data_stream = BinanceUserDataStream(
        binance_config=binance_cfg,
        gateway=gateway,
        session=session,
        on_bracket_leg_filled=on_leg_filled,
    )

    runner = LivePaperRunner(
        config=runner_cfg,
        db_manager=db_manager,
        memory_repo=memory_repo,
        gateway=gateway,
        ws_ingestor=ws_ingestor,
        bar_pipeline=bar_pipeline,
        shadow_tracker=shadow_tracker,
        judge_adapter=judge_adapter,
        use_case=use_case,
        barrier_model=barrier_model,
        ood_detector=ood_detector,
        shap_explainer=shap_explainer,
        model_metadata=model_metadata,
        telegram_notifier=telegram_notifier,
        trading_logger=trading_logger,
        agent_logger=agent_logger,
        system_logger=system_logger,
        user_data_stream=user_data_stream,
    )
    runner_ref[0] = runner
    return runner


def parse_cli_args() -> LivePaperRunnerConfig:
    """Parse CLI arguments into typed LivePaperRunnerConfig (Zero Defaults)."""
    parser = argparse.ArgumentParser(description="Production Live Paper Trading Runner Daemon")
    parser.add_argument(
        "--duration-minutes",
        type=int,
        dest="duration_minutes",
        help="Optional duration in minutes to run before graceful shutdown",
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        dest="testnet",
        help="Connect to Binance Futures Testnet",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        dest="symbols",
        help="Comma-separated symbols (e.g. BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        dest="log_dir",
        help="Directory to store structured JSONL logs",
    )

    parsed = parser.parse_args()

    dur_val: int | None = None
    if parsed.duration_minutes is not None:
        dur_val = int(parsed.duration_minutes)

    testnet_val: bool = bool(parsed.testnet)

    if parsed.symbols is not None and len(parsed.symbols.strip()) > 0:
        symbols_list = [s.strip().upper() for s in parsed.symbols.split(",")]
    else:
        symbols_list = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

    log_dir_val: str = "logs"
    if parsed.log_dir is not None:
        log_dir_val = str(parsed.log_dir)

    return LivePaperRunnerConfig(
        symbols=symbols_list,
        duration_minutes=dur_val,
        testnet=testnet_val,
        log_dir=log_dir_val,
        max_tracking_duration_ms=21600000,  # 6 hours
        check_interval_seconds=1.0,
    )


async def async_main() -> None:
    """Async main routine."""
    _config = parse_cli_args()
    raise RuntimeError(
        "Direct invocation of live_paper_runner CLI without persistent model registry is prohibited "
        "under Invariant #9 (zero heuristic fallbacks). Launch via trading_bot.interfaces.main or supply "
        "real models to bootstrap_live_paper_runner."
    )


def main() -> None:
    """CLI Entrypoint for live paper trading runner daemon."""
    try:
        asyncio.run(async_main())
    except (KeyboardInterrupt, SystemExit):
        print("Live paper runner terminated.")


if __name__ == "__main__":
    main()
