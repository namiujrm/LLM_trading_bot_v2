"""Unified System Daemon Entrypoint & Orchestrator (Milestone 8.3).

Coordinates:
1. Configuration loading from environment mappings (Invariant #9: Zero Defaults).
2. PostgreSQL TimescaleDB connection pool initialization and health checking.
3. Binance Futures REST gateway with server time offset synchronization.
4. Binance Futures WebSocket ingestor for real-time market data streaming.
5. Telegram Bot Adapter with long-polling and interactive command center.
6. Live Paper Runner orchestrating 30-minute BTC-lead cascade trading cycle.
7. Signal handling (SIGINT, SIGTERM) with cross-platform OS safety.
8. Telegram command hooks (/status, /positions, /mode, /panic_close_all, BE, Close).

Clean Architecture: Presentation/Interfaces Layer.
Strict Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import asyncpg
from dotenv import load_dotenv

from trading_bot.adapters.binance.binance_futures_gateway import BinanceFuturesGatewayAdapter
from trading_bot.adapters.binance.ws_micro_ingestor import BinanceFuturesWSIngestor
from trading_bot.adapters.ml_engine.constrained_lgbm_model import CANONICAL_FEATURES
from trading_bot.adapters.ml_engine.file_model_registry import FileModelRegistry
from trading_bot.adapters.ml_engine.model_trainer_adapter import LGBMModelTrainerAdapter
from trading_bot.adapters.persistence.database import DatabaseManager
from trading_bot.adapters.telegram.telegram_bot_adapter import TelegramBotAdapter
from trading_bot.application.services.model_promotion_service import ModelPromotionService
from trading_bot.application.services.model_training_service import ModelTrainingService
from trading_bot.application.use_cases.evaluate_tiered_funnel import TieredFunnelInput
from trading_bot.application.use_cases.run_30m_trading_cycle import TradingCycleSummary
from trading_bot.domain.entities.common.enums import TradeSide
from trading_bot.domain.entities.execution.shadow_order import ShadowOrderRecord
from trading_bot.domain.entities.market.market_state import SemanticQuantPacket
from trading_bot.domain.entities.ml.ml_configs import TreeSHAPConfig
from trading_bot.domain.entities.risk.account_state import (
    AccountPortfolioContext,
    ActivePositionSnapshot,
)
from trading_bot.domain.entities.telegram.telegram_models import (
    TelegramCommandRequest,
    TelegramCycleReport,
)
from trading_bot.domain.exceptions import (
    ChampionModelNotFoundError,
    ModelArtifactCorruptError,
)
from trading_bot.infrastructure.config_schemas import (
    ConfigLoader,
    DatabaseConfig,
)
from trading_bot.interfaces.live_paper_runner import (
    LivePaperRunner,
    LivePaperRunnerConfig,
    bootstrap_live_paper_runner,
)
from trading_bot.ports.execution_gateway_port import ExecutionGatewayPort
from trading_bot.ports.market_data_port import MarketDataPort

logger = logging.getLogger(__name__)


class TradingBotSystemDaemon:
    """Unified application daemon coordinating trading runner and Telegram command center."""

    def __init__(
        self,
        env_mapping: Mapping[str, str],
        session: aiohttp.ClientSession | None,
        db_manager: DatabaseManager | None,
        gateway: ExecutionGatewayPort | None,
        ws_ingestor: BinanceFuturesWSIngestor | None,
        telegram_adapter: TelegramBotAdapter | None,
        runner: LivePaperRunner | None,
    ) -> None:
        self._env: Mapping[str, str] = env_mapping
        self._session: aiohttp.ClientSession | None = session
        self._owns_session: bool = session is None
        self._db_manager: DatabaseManager | None = db_manager
        self._gateway: ExecutionGatewayPort | None = gateway
        self._ws_ingestor: BinanceFuturesWSIngestor | None = ws_ingestor
        self._telegram_adapter: TelegramBotAdapter | None = telegram_adapter
        self._runner: LivePaperRunner | None = runner
        self._model_registry: FileModelRegistry | None = None
        self._training_service: ModelTrainingService | None = None
        self._promotion_service: ModelPromotionService | None = None
        self._evolution_task: asyncio.Task[None] | None = None
        self._time_sync_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()

        self._is_initialized: bool = False
        self._is_running: bool = False
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._runner_task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        if "EXECUTION_MODE" in env_mapping:
            self._execution_mode: str = env_mapping["EXECUTION_MODE"].strip().upper()
        else:
            self._execution_mode = "PAPER"

        if "SYMBOLS" in env_mapping and len(env_mapping["SYMBOLS"].strip()) > 0:
            self._symbols: list[str] = [
                s.strip().upper() for s in env_mapping["SYMBOLS"].split(",")
            ]
        else:
            self._symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

        if "BINANCE_TESTNET" in env_mapping:
            raw_testnet = env_mapping["BINANCE_TESTNET"].strip().lower()
            self._testnet: bool = raw_testnet in ("true", "1", "yes")
        else:
            self._testnet = False

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_shutdown_requested(self) -> bool:
        return self._shutdown_event.is_set()

    @property
    def execution_mode(self) -> str:
        return self._execution_mode

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    @property
    def testnet(self) -> bool:
        return self._testnet

    @property
    def telegram_adapter(self) -> TelegramBotAdapter | None:
        return self._telegram_adapter

    @property
    def gateway(self) -> ExecutionGatewayPort | None:
        return self._gateway

    @property
    def db_manager(self) -> DatabaseManager | None:
        return self._db_manager

    @property
    def ws_ingestor(self) -> BinanceFuturesWSIngestor | None:
        return self._ws_ingestor

    @property
    def runner(self) -> LivePaperRunner | None:
        return self._runner

    @property
    def model_registry(self) -> FileModelRegistry | None:
        return self._model_registry

    @property
    def training_service(self) -> ModelTrainingService | None:
        return self._training_service

    @property
    def promotion_service(self) -> ModelPromotionService | None:
        return self._promotion_service

    # -------------------------------------------------------------------------
    # Lifecycle: Initialize, Run, Shutdown
    # -------------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize all subsystems, establish connections, and wire command hooks."""
        if self._is_initialized:
            logger.warning("TradingBotSystemDaemon is already initialized")
            return

        self._loop = asyncio.get_running_loop()

        # 1. Ensure aiohttp ClientSession exists
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True

        # 2. Initialize Database pool
        if self._db_manager is None:
            db_cfg = DatabaseConfig(
                host=self._env["POSTGRES_HOST"],
                port=int(self._env["POSTGRES_PORT"]),
                db_name=self._env["POSTGRES_DB"],
                user=self._env["POSTGRES_USER"],
                password=self._env["POSTGRES_PASSWORD"],
                max_connections=int(self._env["POSTGRES_MAX_CONNECTIONS"]),
            )
            self._db_manager = DatabaseManager(db_cfg)

        try:
            await self._db_manager.initialize()
            logger.info("Database pool initialized successfully")
        except Exception as exc:
            logger.error("Failed to initialize database pool: %s", exc)

        # 3. Initialize Binance REST Gateway and synchronize time offset
        if self._gateway is None:
            binance_cfg = ConfigLoader.load_binance_config(self._env)
            self._gateway = BinanceFuturesGatewayAdapter(binance_cfg, self._session)

        gateway_any: Any = self._gateway
        if hasattr(gateway_any, "sync_time_offset"):
            try:
                drift_ms = await gateway_any.sync_time_offset()
                logger.info("Binance server time synced (offset: %d ms)", int(drift_ms))
            except Exception as exc:
                logger.warning("Failed to sync Binance time offset: %s", exc)

        # 4. Initialize Telegram Bot Adapter
        if self._telegram_adapter is None:
            tg_cfg = ConfigLoader.load_telegram_config(self._env)
            if "TELEGRAM_POLL_TIMEOUT" in self._env:
                poll_timeout = int(self._env["TELEGRAM_POLL_TIMEOUT"])
            else:
                poll_timeout = 30
            self._telegram_adapter = TelegramBotAdapter(
                config=tg_cfg,
                session=self._session,
                poll_timeout_seconds=poll_timeout,
            )

        self._telegram_adapter.set_execution_mode(self._execution_mode)

        # 5. Initialize Model Registry & Model Training Service
        if "MODELS_DIR" in self._env and len(self._env["MODELS_DIR"].strip()) > 0:
            models_dir = Path(self._env["MODELS_DIR"].strip())
        else:
            models_dir = Path("models")

        tree_shap_cfg = TreeSHAPConfig(
            top_k_bullish=5,
            top_k_bearish=3,
            local_efficiency_tol=1e-4,
            sign_tolerance=1e-4,
        )
        db_pool: asyncpg.Pool[asyncpg.Record] | None = None
        with contextlib.suppress(Exception):
            db_pool = self._db_manager.get_pool()
        self._model_registry = FileModelRegistry(
            base_dir=models_dir,
            db_pool=db_pool,
            tree_shap_config=tree_shap_cfg,
        )
        trainer_adapter = LGBMModelTrainerAdapter.create_canonical(
            tree_shap_config=tree_shap_cfg,
        )
        self._training_service = ModelTrainingService(
            model_registry=self._model_registry,
            model_trainer=trainer_adapter,
            canonical_features=list(CANONICAL_FEATURES),
            default_sample_count=250,
        )

        # 6. Load or Bootstrap Champion Model
        try:
            (
                barrier_model,
                ood_detector,
                shap_explainer,
                model_meta,
            ) = await self._model_registry.load_champion_model()
            logger.info("Loaded active Champion model: %s", model_meta.model_id)
        except (ChampionModelNotFoundError, ModelArtifactCorruptError) as exc:
            logger.warning(
                "Champion model not found or invalid in %s: %s. Auto-bootstrapping baseline...",
                models_dir,
                exc,
            )
            model_meta = await self._training_service.bootstrap_initial_champion(
                market_data_port=self._gateway
                if isinstance(self._gateway, MarketDataPort)
                else None,
                historical_bars=None,
            )
            (
                barrier_model,
                ood_detector,
                shap_explainer,
                _,
            ) = await self._model_registry.load_champion_model()
            logger.info("Auto-bootstrapped and loaded Champion model: %s", model_meta.model_id)

        # 7. Initialize Live Runner if needed
        if self._runner is None:
            if "LOG_DIR" in self._env:
                log_dir_val = self._env["LOG_DIR"]
            else:
                log_dir_val = "logs"

            runner_cfg = LivePaperRunnerConfig(
                symbols=self._symbols,
                duration_minutes=None,
                testnet=self._testnet,
                log_dir=log_dir_val,
                max_tracking_duration_ms=21600000,
                check_interval_seconds=1.0,
            )
            self._runner = await bootstrap_live_paper_runner(
                runner_cfg=runner_cfg,
                session=self._session,
                barrier_model=barrier_model,
                ood_detector=ood_detector,
                shap_explainer=shap_explainer,
                model_metadata=model_meta,
                telegram_notifier=self._telegram_adapter,
            )

        # 8. Initialize Model Promotion Service
        self._promotion_service = ModelPromotionService(
            model_registry=self._model_registry,
            live_paper_runner=self._runner,
            notifier=self._telegram_adapter,
        )

        # 9. Intercept cycle ready and trade conclusions for Telegram notifications
        self._wire_cycle_summary_hook()
        self._wire_concluded_order_hook()

        # 10. Wire action hooks between Telegram and system
        self._wire_telegram_hooks()

        # 11. Initial account context sync
        try:
            ctx = await self.sync_account_context()
            logger.info(
                "Initial account context synced: NAV=$%.2f, mode=%s",
                ctx.equity_nav_usd,
                ctx.execution_mode.value,
            )
        except Exception as exc:
            logger.warning("Initial account context sync warning: %s", exc)

        # 9. Send startup notification to Telegram
        startup_msg = (
            f"🚀 [LLM Trading Bot v2] Hệ thống đã khởi động thành công! "
            f"Chế độ: {self._execution_mode}"
        )
        await self._telegram_adapter.send_text_message(startup_msg, None)

        self._is_initialized = True
        logger.info(
            "TradingBotSystemDaemon initialized successfully in mode %s",
            self._execution_mode,
        )

    async def run(self) -> None:
        """Start Telegram long-polling and Live Paper Runner, then wait for shutdown signal."""
        if not self._is_initialized:
            await self.initialize()

        self._is_running = True
        self._shutdown_event.clear()

        # Start Telegram polling
        if self._telegram_adapter is not None:
            await self._telegram_adapter.start_polling()

        # Start Runner background task
        if self._runner is not None:
            self._runner_task = asyncio.create_task(self._runner.start())

        # Start Sunday evolution background scheduler
        self._evolution_task = asyncio.create_task(self._run_sunday_evolution_scheduler())

        # Start periodic Binance server time sync scheduler (every 15m)
        self._time_sync_task = asyncio.create_task(self._run_periodic_time_sync())

        logger.info("TradingBotSystemDaemon running. Waiting for shutdown signal...")

        try:
            await self._shutdown_event.wait()
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Gracefully terminate all subsystems, flush logs, and release network resources."""
        if not self._is_running and self._shutdown_event.is_set():
            return

        self._is_running = False
        self._shutdown_event.set()

        logger.info("Initiating graceful shutdown of TradingBotSystemDaemon...")

        # 1. Send shutdown notification to Telegram
        if self._telegram_adapter is not None:
            try:
                await self._telegram_adapter.send_text_message(
                    "🛑 *[LLM Trading Bot v2]* Hệ thống đang tắt an toàn (Graceful Shutdown)...",
                    None,
                )
            except Exception as exc:
                logger.warning("Could not send Telegram shutdown notification: %s", exc)

        # 2. Stop Telegram polling
        if self._telegram_adapter is not None:
            try:
                await self._telegram_adapter.stop_polling()
            except Exception as exc:
                logger.error("Error stopping Telegram poller: %s", exc)

        # 3. Stop Runner
        if self._runner is not None:
            try:
                await self._runner.stop()
            except Exception as exc:
                logger.error("Error stopping Live Runner: %s", exc)

        if self._runner_task is not None and not self._runner_task.done():
            self._runner_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._runner_task
            self._runner_task = None

        # Stop Sunday evolution background scheduler
        if self._evolution_task is not None and not self._evolution_task.done():
            self._evolution_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._evolution_task
            self._evolution_task = None

        # Stop periodic time sync scheduler
        if self._time_sync_task is not None and not self._time_sync_task.done():
            self._time_sync_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._time_sync_task
            self._time_sync_task = None

        for bg_task in list(self._background_tasks):
            if not bg_task.done():
                bg_task.cancel()
        self._background_tasks.clear()

        # 4. Stop independent WS ingestor if present
        if self._ws_ingestor is not None and hasattr(self._ws_ingestor, "stop"):
            try:
                await self._ws_ingestor.stop()
            except Exception as exc:
                logger.error("Error stopping WebSocket Ingestor: %s", exc)

        # 5. Close Database pool
        if self._db_manager is not None:
            try:
                await self._db_manager.close()
            except Exception as exc:
                logger.error("Error closing database pool: %s", exc)

        # 6. Close Binance gateway session
        if self._gateway is not None:
            gateway_any: Any = self._gateway
            if hasattr(gateway_any, "close"):
                try:
                    await gateway_any.close()
                except Exception as exc:
                    logger.error("Error closing gateway: %s", exc)

        # 7. Close aiohttp session if owned
        if self._owns_session and self._session is not None and not self._session.closed:
            try:
                await self._session.close()
                self._session = None
            except Exception as exc:
                logger.error("Error closing aiohttp ClientSession: %s", exc)

        logger.info("TradingBotSystemDaemon shutdown complete.")

    def request_shutdown(self) -> None:
        """Trigger graceful shutdown from signal handler or external thread."""
        logger.info("Shutdown requested.")
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._shutdown_event.set)
        else:
            self._shutdown_event.set()

    def setup_signal_handlers(self) -> None:
        """Register cross-platform OS signals (SIGINT, SIGTERM) with Windows safety."""
        if sys.platform != "win32":
            try:
                loop = asyncio.get_running_loop()
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(sig, self.request_shutdown)
            except (NotImplementedError, RuntimeError):
                for sig in (signal.SIGINT, signal.SIGTERM):
                    signal.signal(sig, lambda s, f: self.request_shutdown())
        else:
            for sig in (signal.SIGINT, signal.SIGTERM):
                with contextlib.suppress(ValueError, OSError):
                    signal.signal(sig, lambda s, f: self.request_shutdown())

    # -------------------------------------------------------------------------
    # Account Synchronization & Health Checks
    # -------------------------------------------------------------------------

    async def sync_account_context(self) -> AccountPortfolioContext:
        """Fetch live account portfolio context from Binance gateway and update adapter."""
        if self._gateway is None:
            raise RuntimeError("Gateway is not initialized")

        ctx = await self._gateway.fetch_account_portfolio_context()
        if self._telegram_adapter is not None:
            self._telegram_adapter.set_account_context(ctx)
        return ctx

    def _is_db_healthy(self) -> bool:
        """Check whether PostgreSQL Database pool is active and healthy."""
        if self._db_manager is None:
            return False
        db_any: Any = self._db_manager
        if hasattr(db_any, "_pool"):
            pool_val: Any = db_any._pool
            return pool_val is not None
        return True

    def _is_ws_healthy(self) -> bool:
        """Check whether Binance WebSocket ingestor is active and streaming."""
        if self._ws_ingestor is not None:
            ws_any: Any = self._ws_ingestor
            if hasattr(ws_any, "_is_running"):
                is_run: Any = ws_any._is_running
                return bool(is_run)
        if self._runner is not None:
            runner_any: Any = self._runner
            if hasattr(runner_any, "_ws_ingestor"):
                r_ws: Any = runner_any._ws_ingestor
                if hasattr(r_ws, "_is_running"):
                    r_run: Any = r_ws._is_running
                    return bool(r_run)
        return True

    # -------------------------------------------------------------------------
    # Telegram Hooks & Commands Wiring
    # -------------------------------------------------------------------------

    def _wire_telegram_hooks(self) -> None:
        """Wire callback hooks and commands to Telegram bot adapter."""
        if self._telegram_adapter is None:
            return

        self._telegram_adapter.set_on_mode_change(self.on_mode_change_hook)
        self._telegram_adapter.set_on_panic_close(self.on_panic_close_hook)
        self._telegram_adapter.set_on_breakeven(self.on_breakeven_hook)
        self._telegram_adapter.set_on_close_position(self.on_close_position_hook)
        self._telegram_adapter.set_on_approve_model(self.on_approve_model_hook)
        self._telegram_adapter.set_on_retrain(self.on_retrain_hook)
        self._telegram_adapter.register_custom_command("status", self.handle_status_command)
        adapter_any: Any = self._telegram_adapter
        adapter_any._handle_cmd_positions = self.handle_positions_command

    def _wire_cycle_summary_hook(self) -> None:
        """Attach notification hook to LivePaperRunner cycle completion and automated order placement alerts."""
        if self._runner is None:
            return

        original_handle_cycle_ready = self._runner.handle_cycle_ready

        async def hooked_handle_cycle_ready(
            cycle_time_utc: datetime,
            market_inputs: dict[str, TieredFunnelInput],
            quant_packets: dict[str, SemanticQuantPacket],
            current_prices: dict[str, float],
            atr_30m_values: dict[str, float],
        ) -> None:
            await original_handle_cycle_ready(
                cycle_time_utc,
                market_inputs,
                quant_packets,
                current_prices,
                atr_30m_values,
            )
            if self._runner is not None and len(self._runner.cycle_history) > 0:
                latest_summary = self._runner.cycle_history[-1]

                # Automated order notification when orders are executed
                if self._telegram_adapter is not None:
                    all_results = [
                        latest_summary.btc_result,
                        *list(latest_summary.altcoin_results.values()),
                    ]
                    for r in all_results:
                        if (
                            r.funnel_passed
                            and r.order_id is not None
                            and not r.is_shadow
                            and r.blueprint is not None
                        ):
                            bp = r.blueprint
                            side_str = bp.thesis.proposed_side.value
                            entry_type_str = bp.order_type.value
                            entry_p = float(bp.entry_price)
                            sl_p = float(bp.stop_loss_price)
                            tp_p = float(bp.take_profit_price)
                            if r.verdict is not None:
                                qty = float(r.verdict.approved_notional_usd / max(entry_p, 1e-6))
                                rr = float(r.verdict.net_rr)
                            else:
                                qty = 0.0
                                risk = abs(entry_p - sl_p)
                                reward = abs(tp_p - entry_p)
                                rr = reward / max(risk, 1e-6)

                            with contextlib.suppress(Exception):
                                await self._telegram_adapter.send_order_placed_alert(
                                    symbol=r.symbol,
                                    side=side_str,
                                    order_type=entry_type_str,
                                    entry_price=entry_p,
                                    stop_loss_price=sl_p,
                                    take_profit_price=tp_p,
                                    quantity=qty,
                                    risk_reward_ratio=rr,
                                    order_id=r.order_id,
                                )

                await self.send_cycle_notification(latest_summary)

        self._runner.handle_cycle_ready = hooked_handle_cycle_ready

    def _wire_concluded_order_hook(self) -> None:
        """Attach notification hook to LivePaperRunner for concluded shadow/real trades."""
        if self._runner is None:
            return

        async def on_order_concluded(concluded: ShadowOrderRecord, case: Any) -> None:
            if self._telegram_adapter is None:
                return

            status_str = concluded.status.value
            side_str = concluded.side.value

            if "TP" in status_str:
                exit_price = float(concluded.mfe_price)
                r_mult = float(concluded.mfe_atr_multiple)
            else:
                exit_price = float(concluded.mae_price)
                r_mult = float(-concluded.mae_atr_multiple)

            if concluded.entry_price > 0.0:
                if concluded.side == TradeSide.LONG:
                    real_price_diff = exit_price - concluded.entry_price
                else:
                    real_price_diff = concluded.entry_price - exit_price
                pnl_pct = real_price_diff / concluded.entry_price
            else:
                pnl_pct = 0.0

            pnl_usd = float(pnl_pct * 1000.0)
            lesson = getattr(case, "lesson_summary", status_str)

            with contextlib.suppress(Exception):
                await self._telegram_adapter.send_position_closed_alert(
                    symbol=concluded.symbol,
                    side=side_str,
                    close_price=exit_price,
                    pnl_usd=pnl_usd,
                    pnl_pct=pnl_pct * 100.0,
                    r_multiple=r_mult,
                    exit_reason=f"{status_str} | {lesson}",
                )

        runner_any: Any = self._runner
        if hasattr(runner_any, "set_on_concluded_callback"):
            runner_any.set_on_concluded_callback(on_order_concluded)

    async def handle_status_command(self, request: TelegramCommandRequest) -> str:
        """Format and return live /status overview with DB/WS health and portfolio state."""
        nav = 10000.0
        daily_pnl = 0.0
        dd = 0.0
        margin_util = 0.0
        pos_count = 0
        order_count = 0

        if self._gateway is not None:
            try:
                ctx = await self.sync_account_context()
                nav = ctx.equity_nav_usd
                daily_pnl = ctx.current_daily_pnl_usd
                dd = ctx.current_daily_dd_pct
                margin_util = ctx.margin_utilization_pct
                pos_count = len(ctx.active_positions)
                order_count = len(ctx.pending_orders)
            except Exception as exc:
                logger.warning("Error fetching account context for status: %s", exc)

        db_ok = self._is_db_healthy()
        if db_ok:
            db_status = "🟢 ONLINE"
        else:
            db_status = "🔴 OFFLINE"

        ws_ok = self._is_ws_healthy()
        if ws_ok:
            ws_status = "🟢 CONNECTED"
        else:
            ws_status = "🔴 DISCONNECTED"

        if daily_pnl >= 0.0:
            pnl_emoji = "📈"
            pnl_str = f"+${daily_pnl:,.2f}"
        else:
            pnl_emoji = "📉"
            pnl_str = f"-${abs(daily_pnl):,.2f}"

        is_paused = False
        if self._telegram_adapter is not None and self._telegram_adapter.is_paused:
            is_paused = True

        if is_paused:
            trading_state = "⏸️ PAUSED"
        else:
            trading_state = "▶️ ACTIVE"

        lines: list[str] = [
            "📊 *[HỆ THỐNG GIAO DỊCH - TRẠNG THÁI]*",
            f"• *Execution Mode:* `{self._execution_mode}`",
            f"• *DB Health:* `{db_status}`",
            f"• *WS Ingestor:* `{ws_status}`",
            f"• *Trading Status:* `{trading_state}`",
            f"• *Portfolio NAV:* `${nav:,.2f}`",
            f"• *Daily PnL:* {pnl_emoji} `{pnl_str}`",
            f"• *Drawdown:* `{dd:.2f}%`",
            f"• *Margin Utilization:* `{margin_util:.1f}%`",
            f"• *Open Positions:* `{pos_count}`",
            f"• *Pending Orders:* `{order_count}`",
        ]
        return "\n".join(lines)

    async def handle_positions_command(self) -> str:
        """Fetch live positions from Binance gateway and format with BE/Close inline buttons."""
        if self._gateway is not None:
            with contextlib.suppress(Exception):
                await self.sync_account_context()

        if self._telegram_adapter is not None:
            adapter_any: Any = self._telegram_adapter
            adapter_cls: Any = TelegramBotAdapter
            base_fn: Any = adapter_cls._handle_cmd_positions
            res_val: Any = await base_fn(adapter_any)
            return str(res_val)
        return "Telegram adapter not available"

    async def on_mode_change_hook(self, mode: str) -> str:
        """Action hook to hot-swap execution mode (PAPER, TESTNET, LIVE)."""
        clean_mode = mode.strip().upper()
        if clean_mode not in ("PAPER", "TESTNET", "LIVE"):
            return f"Chế độ không hợp lệ: {mode}"

        self._execution_mode = clean_mode
        if self._telegram_adapter is not None:
            self._telegram_adapter.set_execution_mode(clean_mode)

        if self._runner is not None and hasattr(self._runner, "config"):
            self._runner.config.testnet = clean_mode == "TESTNET"

        logger.info("Execution mode hot-swapped to %s", clean_mode)
        return f"Đã chuyển execution mode sang: {clean_mode}"

    async def on_panic_close_hook(self) -> str:
        """Action hook to cancel all pending orders and close all open positions at Market."""
        if self._gateway is None:
            return "Gateway not available"

        # 1. Cancel all pending orders
        cancelled_orders_count = 0
        try:
            pending = await self._gateway.fetch_pending_orders()
            for o in pending:
                ok = await self._gateway.cancel_order(o.symbol, o.order_id)
                if ok:
                    cancelled_orders_count += 1
        except Exception as exc:
            logger.error("Error cancelling pending orders during panic close: %s", exc)

        # 2. Close all active positions
        closed_positions_count = 0
        try:
            positions = await self._gateway.fetch_active_positions()
            for p in positions:
                gateway_any: Any = self._gateway
                if hasattr(gateway_any, "close_position_market"):
                    await gateway_any.close_position_market(p.symbol, p.quantity)
                    closed_positions_count += 1
                elif hasattr(gateway_any, "_request"):
                    if p.side == TradeSide.LONG:
                        exit_side = "SELL"
                    else:
                        exit_side = "BUY"
                    close_params = {
                        "symbol": p.symbol.upper(),
                        "side": exit_side,
                        "type": "MARKET",
                        "quantity": str(p.quantity),
                        "reduceOnly": "true",
                    }
                    await gateway_any._request("POST", "/fapi/v1/order", close_params, True)
                    closed_positions_count += 1
        except Exception as exc:
            logger.error("Error closing active positions during panic close: %s", exc)

        # 3. Pause trading
        if self._telegram_adapter is not None:
            self._telegram_adapter.set_paused(True, None)

        return (
            f"🚨 PANIC CLOSE ALL: Đã hủy {cancelled_orders_count} lệnh chờ và "
            f"đóng {closed_positions_count} vị thế thành công. Hệ thống đã chuyển sang PAUSED."
        )

    async def on_breakeven_hook(self, position_id: str) -> str:
        """Action hook to move stop-loss to entry price (breakeven) for a given position."""
        if self._gateway is None:
            return "Gateway not available"

        try:
            positions = await self._gateway.fetch_active_positions()
        except Exception as exc:
            return f"Lỗi truy vấn vị thế: {exc}"

        target: ActivePositionSnapshot | None = None
        for p in positions:
            if p.position_id == position_id or p.symbol.upper() == position_id.upper():
                target = p
                break

        if target is None:
            return f"Không tìm thấy vị thế mở với ID {position_id}."

        be_price = target.entry_price

        gateway_any: Any = self._gateway
        if hasattr(gateway_any, "update_stop_loss_to_breakeven"):
            await gateway_any.update_stop_loss_to_breakeven(target.symbol, be_price)
        elif hasattr(gateway_any, "_request"):
            if target.side == TradeSide.LONG:
                exit_side = "SELL"
            else:
                exit_side = "BUY"
            sl_params = {
                "symbol": target.symbol.upper(),
                "side": exit_side,
                "type": "STOP_MARKET",
                "stopPrice": str(be_price),
                "quantity": str(target.quantity),
                "reduceOnly": "true",
                "newClientOrderId": f"be_{target.position_id}",
            }
            try:
                await gateway_any._request("POST", "/fapi/v1/order", sl_params, True)
            except Exception as exc:
                logger.error("Failed to place breakeven SL order: %s", exc)
                return f"Lỗi dời SL về BE: {exc}"

        return f"Đã dời SL về BE ({be_price}) cho vị thế {target.symbol} ({position_id})."

    async def on_close_position_hook(self, position_id: str) -> str:
        """Action hook to close a specific position via Market order and cancel its open orders."""
        if self._gateway is None:
            return "Gateway not available"

        try:
            positions = await self._gateway.fetch_active_positions()
        except Exception as exc:
            return f"Lỗi truy vấn vị thế: {exc}"

        target: ActivePositionSnapshot | None = None
        for p in positions:
            if p.position_id == position_id or p.symbol.upper() == position_id.upper():
                target = p
                break

        if target is None:
            return f"Không tìm thấy vị thế mở với ID {position_id}."

        # 1. Cancel pending orders for this symbol
        try:
            pending = await self._gateway.fetch_pending_orders()
            for o in pending:
                if o.symbol.upper() == target.symbol.upper():
                    await self._gateway.cancel_order(o.symbol, o.order_id)
        except Exception as exc:
            logger.warning("Error cancelling open orders for %s: %s", target.symbol, exc)

        # 2. Market close position
        gateway_any: Any = self._gateway
        if hasattr(gateway_any, "close_position_market"):
            await gateway_any.close_position_market(target.symbol, target.quantity)
        elif hasattr(gateway_any, "_request"):
            if target.side == TradeSide.LONG:
                exit_side = "SELL"
            else:
                exit_side = "BUY"
            close_params = {
                "symbol": target.symbol.upper(),
                "side": exit_side,
                "type": "MARKET",
                "quantity": str(target.quantity),
                "reduceOnly": "true",
            }
            try:
                await gateway_any._request("POST", "/fapi/v1/order", close_params, True)
            except Exception as exc:
                logger.error("Failed to close position: %s", exc)
                return f"Lỗi đóng vị thế: {exc}"

        return f"Đã gửi lệnh Market đóng khẩn cấp vị thế {target.symbol} ({position_id})."

    async def on_approve_model_hook(self) -> str:
        """Promote Challenger model to Champion and hot-reload."""
        if self._promotion_service is None:
            return "❌ Promotion service chưa được khởi tạo."
        if self._telegram_adapter is not None:
            user_id = self._telegram_adapter.config.chat_id
        else:
            user_id = "admin"
        record = await self._promotion_service.promote_and_reload(
            user_id=user_id,
            rationale="Approved by user via Telegram command center",
        )
        return f"Promoted {record.promoted_challenger_id} to Champion"

    async def on_retrain_hook(self, mode: str) -> str:
        """Launch background async retraining of Challenger model."""
        training_svc = self._training_service
        if training_svc is None:
            return "❌ Training service chưa được khởi tạo."

        async def run_training_job() -> None:
            try:
                meta, _, _, _ = await training_svc.train_challenger_model(
                    mode=mode,
                    market_data_port=self._gateway
                    if isinstance(self._gateway, MarketDataPort)
                    else None,
                    historical_bars=None,
                    feature_names=None,
                )
                if self._telegram_adapter is not None:
                    self._telegram_adapter.set_challenger_pending(True)
                    msg = (
                        f"🤖 *CHALLENGER MODEL ĐÃ HUẤN LUYỆN XONG*\n\n"
                        f"• *ID:* `{meta.model_id}`\n"
                        f"• *Net Sharpe:* `{meta.net_sharpe:.2f}`\n"
                        f"• *Brier Skill Score:* `{meta.brier_skill_score:.4f}`\n"
                        f"• *Max Drawdown:* `{meta.max_drawdown_pct:.2%}`\n"
                        f"• *Số mẫu:* `{meta.training_sample_count}`\n\n"
                        f"🏆 Dùng `/approve_model` để xác nhận thăng hạng model này thành Champion mới!"
                    )
                    await self._telegram_adapter.send_text_message(msg, None)
            except Exception as exc:
                logger.error("Background Challenger retraining failed: %s", exc, exc_info=True)
                if self._telegram_adapter is not None:
                    await self._telegram_adapter.send_error_alert(
                        component="ModelTrainingService",
                        error_message=str(exc),
                        is_fatal=False,
                    )

        training_task = asyncio.create_task(run_training_job())
        self._background_tasks.add(training_task)
        training_task.add_done_callback(self._background_tasks.discard)
        return f"Challenger retraining job launched in background (mode: {mode})"

    async def _run_sunday_evolution_scheduler(self) -> None:
        """Periodically check and trigger automated weekly evolution on Sunday 00:00 UTC."""
        logger.info("Sunday 00:00 UTC weekly evolution scheduler started.")
        last_triggered_week: int = -1
        while self._is_running and not self._shutdown_event.is_set():
            try:
                now_utc = datetime.now(UTC)
                if now_utc.weekday() == 6 and now_utc.hour == 0:
                    current_week = now_utc.isocalendar()[1]
                    if current_week != last_triggered_week:
                        last_triggered_week = current_week
                        logger.info(
                            "Sunday 00:00 UTC reached. Triggering weekly Challenger retraining."
                        )
                        if self._telegram_adapter is not None:
                            await self._telegram_adapter.send_text_message(
                                "📅 *SUNDAY 00:00 UTC EVOLUTION:*\n"
                                "Bắt đầu chu trình tự động huấn luyện và đánh giá Challenger hàng tuần...",
                                None,
                            )
                        await self.on_retrain_hook("full")

                for _ in range(60):
                    if not self._is_running or self._shutdown_event.is_set():
                        break
                    await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error in Sunday evolution scheduler: %s", exc)
                await asyncio.sleep(10.0)

    async def _run_periodic_time_sync(self) -> None:
        """Periodically re-synchronize Binance server time offset every 15 minutes."""
        logger.info("Periodic Binance server time sync scheduler started (interval: 15m).")
        while self._is_running:
            try:
                await asyncio.sleep(900.0)
                if self._gateway is not None:
                    gateway_any: Any = self._gateway
                    if hasattr(gateway_any, "sync_time_offset"):
                        drift_ms = await gateway_any.sync_time_offset()
                        logger.info(
                            "Periodic Binance server time synced (offset: %d ms)", int(drift_ms)
                        )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Periodic Binance time sync warning: %s", exc)

    async def send_cycle_notification(self, summary: TradingCycleSummary) -> bool:
        """Format 30-minute cycle report card and send to Telegram."""
        if self._telegram_adapter is None:
            return False

        nav = 10000.0
        daily_pnl = 0.0
        dd = 0.0
        pos_count = 0

        if self._gateway is not None:
            try:
                ctx = await self._gateway.fetch_account_portfolio_context()
                self._telegram_adapter.set_account_context(ctx)
                nav = ctx.equity_nav_usd
                daily_pnl = ctx.current_daily_pnl_usd
                dd = ctx.current_daily_dd_pct
                pos_count = len(ctx.active_positions)
            except Exception as exc:
                logger.warning("Failed to fetch account context for cycle report: %s", exc)

        if summary.btc_result.funnel_passed:
            btc_stage = "PASSED"
        else:
            btc_stage = summary.btc_result.funnel_stage.replace("_", " ")

        arbiter_summary = (
            f"BTC: {btc_stage} | Alt Executed: {summary.executed_orders_count} | "
            f"Shadow: {summary.shadow_orders_count}"
        )

        cycle_report = TelegramCycleReport(
            cycle_timestamp_ms=summary.cycle_timestamp_ms,
            execution_mode=self._execution_mode,
            portfolio_nav_usd=nav,
            daily_pnl_usd=daily_pnl,
            current_drawdown_pct=dd,
            open_positions_count=pos_count,
            executed_orders_count=summary.executed_orders_count,
            shadow_orders_count=summary.shadow_orders_count,
            btc_funnel_passed=summary.btc_result.funnel_passed,
            altcoins_scanned_count=len(summary.altcoin_results),
            shap_top_features={},
            calibrated_probabilities={},
            arbiter_verdict_summary=arbiter_summary,
        )
        return await self._telegram_adapter.send_cycle_report(cycle_report)


async def main() -> int:
    """Unified application entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    load_dotenv()
    env = dict(os.environ)

    daemon = TradingBotSystemDaemon(
        env_mapping=env,
        session=None,
        db_manager=None,
        gateway=None,
        ws_ingestor=None,
        telegram_adapter=None,
        runner=None,
    )
    daemon.setup_signal_handlers()

    try:
        await daemon.initialize()
        await daemon.run()
        return 0
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Application interrupted by user.")
        await daemon.shutdown()
        return 0
    except Exception as exc:
        logger.error("Unhandled fatal exception in main daemon: %s", exc, exc_info=True)
        if daemon.telegram_adapter is not None:
            with contextlib.suppress(Exception):
                await daemon.telegram_adapter.send_error_alert(
                    component="TradingBotSystemDaemon",
                    error_message=str(exc),
                    is_fatal=True,
                )
        await daemon.shutdown()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
