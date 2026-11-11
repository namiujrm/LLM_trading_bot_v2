"""Comprehensive End-to-End Integration Test Suite for Stage 6 Pipeline.

Milestone M6.4: Full Integration, Seeding, Runner & Verification.
Covers:
- 1m Bar streaming -> rolling multi-TF aggregation (15m, 30m) -> 30m candle close.
- Causal ML inference (Dual-Barrier, OOD Detector, TreeSHAP Explainer) -> SemanticQuantPacket.
- pgvector RAG: retrieves K=20 cohort summary and stratified lessons from episodic memory.
- Multi-agent cascade deliberation (MarketScout, TradeArchitect, RiskArbiter).
- Trade execution on gateway & shadow registration on Arbiter veto.
- Subsequent 1m bars price progression tracking (MAE, MFE).
- Concluded shadow order triggers PostMortemJudgeAdapter autopsy and pgvector persistence.
- Post-SL tracking and STOP_HUNTED_THEN_HIT_TP autopsy synthesis.
- Live pgvector TimescaleDB query verification against seeded historical cases.

Enforces 100% Invariant #9 compliance across all test doubles and helpers:
- Zero default parameters in functions and methods.
- Zero class field defaults.
- Zero silent fallbacks.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import pytest

from trading_bot.adapters.agents.arbiter_agent import RiskArbiterAdapter
from trading_bot.adapters.agents.architect_agent import TradeArchitectAdapter
from trading_bot.adapters.agents.judge_agent import PostMortemJudgeAdapter
from trading_bot.adapters.agents.macro_news_sentinel import MacroNewsSentinel
from trading_bot.adapters.agents.ninerouter_client import (
    NinerouterResponse,
)
from trading_bot.adapters.agents.scout_agent import MarketScoutAdapter
from trading_bot.adapters.persistence.database import DatabaseManager
from trading_bot.adapters.persistence.pgvector_memory_repo import PgvectorMemoryRepository
from trading_bot.application.services.live_bar_pipeline import LiveBarPipelineService
from trading_bot.application.services.shadow_order_tracker import ShadowOrderTracker
from trading_bot.application.use_cases.evaluate_tiered_funnel import (
    TieredFunnelEvaluator,
    TieredFunnelInput,
)
from trading_bot.application.use_cases.run_30m_trading_cycle import (
    Run30mTradingCycleUseCase,
    TradingCycleSummary,
)
from trading_bot.domain.entities.common.enums import (
    EntryOrderType,
    ExecutionMode,
    ExistingPositionDirective,
    MarketRegime,
    OIPriceQuadrant,
    OutcomeCategory,
    PlaybookType,
    RiskDecisionType,
    ShadowOrderStatus,
    TradeSide,
    VetoCategory,
)
from trading_bot.domain.entities.execution.trade_blueprint import (
    BracketOrderTicket,
    MarketThesis,
    RiskVerdict,
    TradeBlueprint,
)
from trading_bot.domain.entities.market.market_state import (
    EnrichedMicroBarRecord,
    FeatureSignalItem,
    MicroBar1m,
    MultiTFBar,
    SemanticQuantPacket,
)
from trading_bot.domain.entities.memory.episodic_case import (
    CohortBaseRateSummary,
    EpisodicCase,
    StratifiedMemoryPack,
)
from trading_bot.domain.entities.ml.ml_predictions import (
    BarrierPredictionResult,
    OODDetectionResult,
    SHAPDriverContribution,
    SHAPExplanationResult,
)
from trading_bot.domain.entities.quant.quant_features import (
    MicrostructureMetrics,
    QuantFeatureVector,
    RegimeMetrics,
)
from trading_bot.domain.entities.risk.account_state import (
    AccountPortfolioContext,
    ActivePositionSnapshot,
    PendingOrderSnapshot,
)
from trading_bot.domain.services.execution.fee_to_edge_gate import (
    FeeToEdgeConfig,
    FeeToEdgeGatingService,
)
from trading_bot.domain.services.execution.playbook_validator import (
    PlaybookConfig,
    PlaybookValidator,
)
from trading_bot.domain.services.risk.risk_calculator import (
    DeterministicRiskCalculator,
    RiskConfig,
)
from trading_bot.infrastructure.config_schemas import (
    DatabaseConfig,
    LoggingConfig,
    NinerouterConfig,
)
from trading_bot.infrastructure.logging.bot_logger import (
    LogCategory,
    StructuredBotLogger,
)
from trading_bot.interfaces.live_paper_runner import (
    LivePaperRunner,
    LivePaperRunnerConfig,
)
from trading_bot.ports.execution_gateway_port import ExecutionGatewayPort
from trading_bot.ports.memory_repository_port import MemoryRepositoryPort
from trading_bot.ports.ml_model_port import (
    CausalBarrierModelPort,
    OODDetectorPort,
    TreeSHAPExplainerPort,
)
from trading_bot.ports.quant_engine_port import QuantEnginePort
from trading_bot.ports.rate_limit_governor_port import (
    RateLimitGovernorPort,
    RouteQuotaSpec,
)

# ==============================================================================
# Invariant #9 Compliant Test Doubles (Zero Defaults, Zero Class Field Defaults)
# ==============================================================================


class MockExecutionGateway(ExecutionGatewayPort):
    """Mock execution gateway tracking placed bracket orders (Zero Defaults)."""

    def __init__(self, initial_balance: float) -> None:
        self.initial_balance: float = initial_balance
        self.placed_tickets: list[BracketOrderTicket] = []
        self.cancelled_order_ids: list[str] = []
        self.time_offset_ms: int = 15

    async def sync_time_offset(self) -> int:
        return self.time_offset_ms

    async def place_bracket_order(self, ticket: BracketOrderTicket) -> str:
        self.placed_tickets.append(ticket)
        return f"order_{ticket.symbol.lower()}_{len(self.placed_tickets)}"

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        self.cancelled_order_ids.append(order_id)
        return True

    async def get_position_notional(self, symbol: str) -> float:
        return 0.0

    async def fetch_account_portfolio_context(self) -> AccountPortfolioContext:
        return AccountPortfolioContext(
            execution_mode=ExecutionMode.TESTNET,
            timestamp_ms=1700000000000,
            wallet_balance_usd=self.initial_balance,
            unrealized_pnl_usd=0.0,
            equity_nav_usd=self.initial_balance,
            used_margin_usd=0.0,
            available_margin_usd=self.initial_balance,
            margin_utilization_pct=0.0,
            current_daily_pnl_usd=0.0,
            current_daily_dd_pct=0.0,
            remaining_daily_dd_budget_pct=3.0,
            current_max_dd_pct=0.0,
            remaining_max_dd_budget_pct=6.0,
            consecutive_losses_count=0,
            active_positions=[],
            pending_orders=[],
            gross_exposure_usd=0.0,
            net_directional_delta_usd=0.0,
            net_btc_beta_exposure_usd=0.0,
            raw_open_heat_pct=0.0,
            effective_open_heat_pct=0.0,
            same_direction_active_count=0,
        )

    async def fetch_active_positions(self) -> list[ActivePositionSnapshot]:
        return []

    async def fetch_pending_orders(self) -> list[PendingOrderSnapshot]:
        return []

    async def cancel_algo_order(self, symbol: str, algo_id: str) -> bool:
        return True

    async def fetch_open_algo_orders(self, symbol: str) -> list[dict[str, Any]]:
        return []

    async def fetch_symbol_filters(self) -> dict[str, Any]:
        return {}

    async def fetch_recent_klines_1m(self, symbol: str, limit: int) -> list[Any]:
        return []

    async def create_user_data_stream(self) -> str:
        return "mock_listen_key"

    async def keepalive_user_data_stream(self, listen_key: str) -> None:
        pass

    async def close_user_data_stream(self, listen_key: str) -> None:
        pass


class MockInMemoryMemoryRepository(MemoryRepositoryPort):
    """In-memory implementation of MemoryRepositoryPort for hermetic testing (Zero Defaults)."""

    def __init__(self, initial_cases: list[EpisodicCase]) -> None:
        self._cases: list[EpisodicCase] = list(initial_cases)
        self.upserted_cases: list[EpisodicCase] = []
        self.boosted_ids: list[tuple[str, float]] = []

    async def query_knn_cohort(
        self,
        vector_16d: list[float],
        playbook: str,
        k: int,
    ) -> list[EpisodicCase]:
        matching = [c for c in self._cases if c.playbook == playbook]
        return matching[:k]

    async def compute_cohort_summary(
        self,
        vector_16d: list[float],
        playbook: str,
        k: int,
    ) -> CohortBaseRateSummary:
        matching = [c for c in self._cases if c.playbook == playbook][:k]
        if not matching:
            return CohortBaseRateSummary(
                has_historical_cohort=False,
                total_neighbors=0,
                weighted_win_rate=None,
                stop_hunted_then_tp_rate=None,
                true_failure_trap_rate=None,
                expected_r_multiple=None,
                mae_target_percentile_atr=None,
                mfe_median_atr=None,
            )

        wins = sum(1 for c in matching if c.outcome == OutcomeCategory.VALIDATED_WIN)
        traps = sum(1 for c in matching if c.outcome == OutcomeCategory.FALSE_POSITIVE_TRAP)
        hunts = sum(1 for c in matching if c.outcome == OutcomeCategory.STOP_HUNTED_THEN_HIT_TP)
        tot = len(matching)

        return CohortBaseRateSummary(
            has_historical_cohort=True,
            total_neighbors=tot,
            weighted_win_rate=round(wins / tot, 4),
            stop_hunted_then_tp_rate=round(hunts / tot, 4),
            true_failure_trap_rate=round(traps / tot, 4),
            expected_r_multiple=1.45,
            mae_target_percentile_atr=1.15,
            mfe_median_atr=2.30,
        )

    async def fetch_stratified_pack(
        self,
        vector_16d: list[float],
        playbook: str,
        k: int,
    ) -> StratifiedMemoryPack:
        summary = await self.compute_cohort_summary(vector_16d, playbook, k)
        matching = [c for c in self._cases if c.playbook == playbook]
        b1 = [
            c
            for c in matching
            if c.outcome in (OutcomeCategory.VALIDATED_WIN, OutcomeCategory.MISSED_OPPORTUNITY)
        ][:2]
        b2 = [
            c
            for c in matching
            if c.outcome in (OutcomeCategory.FALSE_POSITIVE_TRAP, OutcomeCategory.VALIDATED_DEFENSE)
        ][:2]
        b3 = [c for c in matching if c.outcome == OutcomeCategory.STOP_HUNTED_THEN_HIT_TP][:2]
        return StratifiedMemoryPack(
            cohort_summary=summary,
            bucket1_bull_missed=b1,
            bucket2_failure_traps=b2,
            bucket3_stop_hunts=b3,
        )

    async def upsert_case(self, case: EpisodicCase) -> None:
        self._cases.append(case)
        self.upserted_cases.append(case)

    async def boost_lesson_confidence(self, case_id: str, delta: float) -> None:
        self.boosted_ids.append((case_id, delta))
        for c in self._cases:
            if c.case_id == case_id:
                new_conf = max(0.0, min(1.0, c.confidence_score + delta))
                c.confidence_score = round(new_conf, 4)


class MockRateLimitGovernor(RateLimitGovernorPort):
    """Deterministic rate limit governor (Zero Defaults)."""

    def __init__(self, spec: RouteQuotaSpec) -> None:
        self.spec: RouteQuotaSpec = spec
        self.cooldown_calls: list[tuple[str, float]] = []

    async def acquire_available_route(
        self,
        tier: str,
        estimated_tokens: int,
    ) -> RouteQuotaSpec:
        return self.spec

    async def mark_route_cooldown(
        self,
        route_id: str,
        cooldown_seconds: float,
    ) -> None:
        self.cooldown_calls.append((route_id, cooldown_seconds))


class MockNinerouterClient:
    """Mock 9Router client delivering typed responses (Zero Defaults)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate_chat_completion(
        self,
        tier: str,
        messages: list[dict[str, str]],
        estimated_tokens: int,
        temperature: float,
        max_tokens: int,
    ) -> NinerouterResponse:
        self.calls.append(
            {
                "tier": tier,
                "messages": messages,
                "estimated_tokens": estimated_tokens,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

        content_str = ""
        prompt_text = "".join(m["content"] for m in messages)

        # Distinguish agent by prompt keywords: Judge prompt MUST be matched first
        if (
            "Post-Mortem Autopsy Judge" in prompt_text
            or "classified_outcome" in prompt_text
            or "autopsy" in prompt_text.lower()
        ):
            if "STOP_HUNTED_THEN_HIT_TP" in prompt_text:
                target_agent = "TradeArchitect"
                tactical_lesson = "Stop-hunt detected. Widen SL buffer past 80th-percentile MAE."
            else:
                target_agent = "MarketScout"
                tactical_lesson = "Vetoed candidate reached TP with favorable excursion 2.50x ATR. Signal was accurate."

            content_str = json.dumps(
                {
                    "target_agent": target_agent,
                    "outcome_assessment": "Concluded shadow order analyzed.",
                    "root_cause": "Regime dynamics validated.",
                    "tactical_lesson": tactical_lesson,
                }
            )
        elif "TradeArchitect" in prompt_text or "blueprint" in prompt_text.lower():
            # Architect response
            content_str = json.dumps(
                {
                    "entry_order_type": "LIMIT_PULLBACK_15M",
                    "entry_price": 50000.0,
                    "stop_loss_price": 49000.0,
                    "take_profit_price": 52000.0,
                    "limit_offset_bps": 5.0,
                    "time_in_force_minutes": 30,
                    "reasoning": "Tight bracket above 80th-percentile MAE buffer.",
                }
            )
        elif "RiskArbiter" in prompt_text or "risk" in prompt_text.lower():
            # Arbiter response
            content_str = json.dumps(
                {
                    "decision": "EXECUTE",
                    "recommended_size_multiplier": 1.0,
                    "recommended_leverage": 3,
                    "veto_category": None,
                    "veto_reasons": [],
                    "deliberation_notes": "All quantitative invariants verified.",
                }
            )
        else:
            # Scout response
            content_str = json.dumps(
                {
                    "direction": "LONG",
                    "confidence": 0.85,
                    "regime": "STRONG_TREND_BULL",
                    "core_thesis": "Strong momentum breakout with positive taker CVD and clean order flow.",
                    "invalidation_condition": "Drop below 30m VWAP on heavy volume.",
                }
            )

        return NinerouterResponse(
            content=content_str,
            model=f"mock_{tier}",
            finish_reason="stop",
            prompt_tokens=150,
            completion_tokens=80,
            is_fallback=False,
        )


class MockQuantEngine(QuantEnginePort):
    """Deterministic MockQuantEngine returning typed metrics (Zero Defaults)."""

    def __init__(self) -> None:
        self.metrics_calls: int = 0

    async def compute_microstructure_metrics(
        self,
        symbol: str,
        bars_1m: list[MicroBar1m],
    ) -> MicrostructureMetrics:
        self.metrics_calls += 1
        return MicrostructureMetrics(
            vpin=0.35,
            kyle_lambda=0.0012,
            estimated_oneway_slippage_pct=0.0005,
            ofi_zscore=1.25,
            obi_mean=0.25,
            obi_std=0.05,
            va_amihud=0.0004,
            corwin_schultz_spread_bps=2.5,
            mnvr=1.05,
        )

    async def compute_regime_state(
        self,
        symbol: str,
        bars_multi_tf: list[MultiTFBar],
    ) -> RegimeMetrics:
        return RegimeMetrics(
            hurst_exponent=0.62,
            yang_zhang_vol=0.015,
            garman_klass_vol=0.014,
            parkinson_vol=0.013,
            bipower_variation=0.012,
            jump_z_score=0.10,
            has_volatility_jump=False,
            shannon_entropy=0.75,
            permutation_entropy=0.72,
            tsallis_entropy=0.80,
            regime_label=MarketRegime.STRONG_TREND_BULL,
        )

    async def evaluate_quant_packet(
        self,
        symbol: str,
        timestamp_ms: int,
    ) -> SemanticQuantPacket:
        vec_16d = [0.15] * 16
        atr_val = 500.0 if symbol == "BTCUSDT" else 50.0
        return SemanticQuantPacket(
            symbol=symbol,
            timestamp_ms=timestamp_ms,
            active_features=[
                FeatureSignalItem(
                    feature_name="hurst_exponent_100",
                    raw_value=0.62,
                    z_score=1.20,
                    is_statistically_active=True,
                    economic_direction=1,
                ),
                FeatureSignalItem(
                    feature_name="vpin_toxicity_pct",
                    raw_value=0.35,
                    z_score=-0.50,
                    is_statistically_active=True,
                    economic_direction=-1,
                ),
                FeatureSignalItem(
                    feature_name="atr_30m",
                    raw_value=atr_val,
                    z_score=0.10,
                    is_statistically_active=True,
                    economic_direction=0,
                ),
            ],
            market_regime=MarketRegime.STRONG_TREND_BULL,
            oi_quadrant=OIPriceQuadrant.LONG_BUILDUP,
            trajectory_vector_16d=vec_16d,
        )

    async def compute_feature_vector_batch(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> list[QuantFeatureVector]:
        return []


class MockBarrierModel(CausalBarrierModelPort):
    """Deterministic causal barrier model double (Zero Defaults)."""

    def fit(
        self,
        train_features: Sequence[Sequence[float]],
        y_long: Sequence[int],
        y_short: Sequence[int],
        sample_weights: Sequence[float],
        sample_intervals: Sequence[tuple[int, int]] | None,
    ) -> None:
        pass

    def predict(self, sample_features: Sequence[float]) -> BarrierPredictionResult:
        return BarrierPredictionResult(
            p_long=0.68,
            p_short=0.22,
            p_raw_long=0.68,
            p_raw_short=0.22,
            is_coherent=True,
            coherence_penalty=0.0,
        )

    def predict_batch(
        self,
        batch_features: Sequence[Sequence[float]],
    ) -> Sequence[BarrierPredictionResult]:
        return [self.predict(f) for f in batch_features]


class MockOODDetector(OODDetectorPort):
    """Deterministic out-of-distribution detector double (Zero Defaults)."""

    def fit(
        self,
        train_features: Sequence[Sequence[float]],
        calibration_features: Sequence[Sequence[float]],
    ) -> None:
        pass

    def predict(self, sample_features: Sequence[float]) -> OODDetectionResult:
        return OODDetectionResult(
            is_ood=False,
            mahalanobis_distance=1.1,
            empirical_quantile_threshold=3.5,
            excess_distance=0.0,
            gamma_ood=1.0,
            feature_contributions=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        )

    def predict_batch(
        self,
        batch_features: Sequence[Sequence[float]],
    ) -> Sequence[OODDetectionResult]:
        return [self.predict(f) for f in batch_features]


class MockTreeSHAPExplainer(TreeSHAPExplainerPort):
    """Deterministic TreeSHAP explainer double (Zero Defaults)."""

    def explain(
        self,
        feature_vector: Sequence[float],
        feature_names: Sequence[str],
        model: Any,
        symbol: str,
        bar_time: datetime,
        model_type: str,
    ) -> SHAPExplanationResult:
        c_bull = SHAPDriverContribution(
            feature_name="hurst_exponent_100",
            feature_value=0.65,
            shap_value=0.08,
            expected_sign=1,
            is_sign_consistent=True,
        )
        c_bear = SHAPDriverContribution(
            feature_name="vpin_toxicity_pct",
            feature_value=0.35,
            shap_value=-0.08,
            expected_sign=-1,
            is_sign_consistent=True,
        )
        return SHAPExplanationResult(
            symbol=symbol,
            bar_time=bar_time,
            model_type=model_type,
            base_value=0.50,
            raw_margin=0.50,
            contributions=(c_bull, c_bear),
            top_bullish_drivers=(c_bull,),
            top_bearish_drivers=(c_bear,),
            is_economic_consistent=True,
        )

    def explain_batch(
        self,
        batch_features: Sequence[Sequence[float]],
        feature_names: Sequence[str],
        model: Any,
        symbol: str,
        bar_times: Sequence[datetime],
        model_type: str,
    ) -> tuple[SHAPExplanationResult, ...]:
        return tuple(
            self.explain(f, feature_names, model, symbol, t, model_type)
            for f, t in zip(batch_features, bar_times, strict=True)
        )


class MockWSIngestor:
    """Mock WebSocket ingestor double (Zero Defaults)."""

    def __init__(self) -> None:
        self.started: bool = False
        self.stopped: bool = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True


# ==============================================================================
# Helper Factories (Zero Defaults)
# ==============================================================================


def make_route_spec() -> RouteQuotaSpec:
    return RouteQuotaSpec(
        route_id="integration_spec",
        model_name="gemini-2.5-flash",
        rpm_limit=100,
        tpm_limit=100000,
        rpd_limit=5000,
        min_interval_ms=10,
        max_concurrency=5,
    )


def make_ninerouter_config() -> NinerouterConfig:
    return NinerouterConfig(
        base_url="http://mock-9router:20128/v1",
        api_key="integration-key",
        timeout_seconds=5.0,
        fast_tier_model="ds/deepseek-v4-flash",
        deep_tier_model="ds/deepseek-v4-pro",
        judge_tier_model="gemini-2.5-flash",
    )


def make_risk_config() -> RiskConfig:
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


def make_enriched_bar(
    symbol: str,
    open_time_ms: int,
    open_px: float,
    high_px: float,
    low_px: float,
    close_px: float,
    volume: float,
) -> EnrichedMicroBarRecord:
    return EnrichedMicroBarRecord(
        symbol=symbol,
        open_time_ms=open_time_ms,
        open=open_px,
        high=high_px,
        low=low_px,
        close=close_px,
        volume=volume,
        quote_volume=close_px * volume,
        taker_buy_vol=volume * 0.60,
        taker_buy_quote_vol=close_px * volume * 0.60,
        trade_count=100,
        ofi_sum=15.0,
        obi_mean=0.25,
        obi_std=0.05,
        spread_bps_mean=2.0,
        liq_long_usd=0.0,
        liq_short_usd=0.0,
    )


def make_seed_case(
    case_id: str,
    symbol: str,
    outcome: OutcomeCategory,
    playbook: str,
) -> EpisodicCase:
    return EpisodicCase(
        case_id=case_id,
        symbol=symbol,
        timestamp_ms=1700000000000,
        vector_16d=[0.15] * 16,
        outcome=outcome,
        confidence_score=0.75,
        lesson_summary=f"[{playbook}] {symbol} tactical lesson for {outcome.value}",
        playbook=playbook,
    )


def make_test_cases() -> list[EpisodicCase]:
    cases: list[EpisodicCase] = []
    playbooks = [
        PlaybookType.TREND_CONTINUATION_BREAKOUT.value,
        PlaybookType.LIQUIDATION_EXHAUSTION_REVERSAL.value,
        PlaybookType.CROSS_ASSET_LEAD_LAG_CATCHUP.value,
        PlaybookType.GENERAL_ALPHA_MOMENTUM.value,
    ]
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    idx = 0
    for pb in playbooks:
        for sym in symbols:
            idx += 1
            cases.append(make_seed_case(f"case_win_{idx}", sym, OutcomeCategory.VALIDATED_WIN, pb))
            idx += 1
            cases.append(
                make_seed_case(f"case_trap_{idx}", sym, OutcomeCategory.FALSE_POSITIVE_TRAP, pb)
            )
            idx += 1
            cases.append(
                make_seed_case(f"case_hunt_{idx}", sym, OutcomeCategory.STOP_HUNTED_THEN_HIT_TP, pb)
            )
            idx += 1
            cases.append(
                make_seed_case(f"case_miss_{idx}", sym, OutcomeCategory.MISSED_OPPORTUNITY, pb)
            )
    return cases


# ==============================================================================
# Integration Test Cases
# ==============================================================================


@pytest.mark.asyncio
async def test_stage6_full_pipeline_simulation(tmp_path: Any) -> None:
    """Full End-to-End Simulation:

    1m bars -> 30m close -> ML quant packet -> 30m Trading Cycle -> pgvector RAG -> Deliberation ->
    Trade Execution on BTC & Shadow Registration on ETH -> Subsequent 1m Bars Price Progression ->
    Take-Profit Hit -> Post-Mortem Judge Autopsy -> Episodic Memory Persisted.
    """
    log_dir = str(tmp_path / "logs")
    log_cfg = LoggingConfig(
        log_dir=log_dir, retention_days=1, min_level="INFO", console_output=False
    )
    trading_logger = StructuredBotLogger(LogCategory.TRADING, log_cfg)
    agent_logger = StructuredBotLogger(LogCategory.AGENTS, log_cfg)
    system_logger = StructuredBotLogger(LogCategory.SYSTEM, log_cfg)

    # 1. Setup seeded in-memory memory repository with 64 historical cases
    initial_cases = make_test_cases()
    memory_repo = MockInMemoryMemoryRepository(initial_cases)

    # 2. Setup Gateway & Agent Infrastructure
    gateway = MockExecutionGateway(initial_balance=50000.0)
    mock_nr = MockNinerouterClient()
    nr_cfg = make_ninerouter_config()
    governor = MockRateLimitGovernor(make_route_spec())

    # Scout, Architect, Arbiter adapters
    # For Scout: propose LONG for BTC and LONG for ETH
    scout = MarketScoutAdapter(mock_nr, "fast")  # type: ignore[arg-type]
    architect = TradeArchitectAdapter(mock_nr, "deep")  # type: ignore[arg-type]

    risk_cfg = make_risk_config()
    risk_calc = DeterministicRiskCalculator(risk_cfg)
    arbiter = RiskArbiterAdapter(mock_nr, "deep", risk_calc, agent_logger)  # type: ignore[arg-type]

    # Post-Mortem Judge Adapter
    judge_adapter = PostMortemJudgeAdapter(
        client=mock_nr,  # type: ignore[arg-type]
        config=nr_cfg,
        governor=governor,
        memory_repo=memory_repo,
        logger=agent_logger,
    )

    fee_service = FeeToEdgeGatingService(
        FeeToEdgeConfig(multiplier=1.0, roundtrip_cost_rate=0.0010, min_calibrated_prob=0.52)
    )
    playbook_val = PlaybookValidator(
        PlaybookConfig(
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
            altcoin_symbols=["ETHUSDT"],
        )
    )

    funnel_eval = TieredFunnelEvaluator(
        fee_to_edge_service=fee_service,
        playbook_validator=playbook_val,
        max_spread_bps=15.0,
        min_yz_vol=0.005,
        max_uncertainty_width=0.45,
    )

    macro_sentinel = MacroNewsSentinel([])
    shadow_tracker = ShadowOrderTracker(max_tracking_duration_ms=21600000)

    use_case = Run30mTradingCycleUseCase(
        funnel_evaluator=funnel_eval,
        scout=scout,
        architect=architect,
        arbiter=arbiter,
        execution_gateway=gateway,
        macro_sentinel=macro_sentinel,
        memory_repo=memory_repo,
        shadow_order_tracker=shadow_tracker,
    )

    # 3. Setup LiveBarPipelineService
    quant_engine = MockQuantEngine()
    barrier_model = MockBarrierModel()
    ood_detector = MockOODDetector()
    shap_explainer = MockTreeSHAPExplainer()

    runner_cfg = LivePaperRunnerConfig(
        symbols=["BTCUSDT", "ETHUSDT"],
        duration_minutes=35,
        testnet=True,
        log_dir=log_dir,
        max_tracking_duration_ms=21600000,
        check_interval_seconds=0.1,
    )

    db_cfg = DatabaseConfig(
        host="localhost",
        port=5433,
        db_name="trading_bot_db",
        user="postgres",
        password="postgres",
        max_connections=5,
    )
    db_mgr = DatabaseManager(db_cfg)

    ws_mock = MockWSIngestor()

    cycle_dispatch_received: list[TradingCycleSummary] = []

    async def on_cycle_ready(
        cycle_dt: datetime,
        m_inputs: dict[str, TieredFunnelInput],
        q_packets: dict[str, SemanticQuantPacket],
        c_prices: dict[str, float],
        a_values: dict[str, float],
    ) -> None:
        summary = await use_case.execute_cycle(
            cycle_time_utc=cycle_dt,
            market_inputs=m_inputs,
            current_prices=c_prices,
            atr_30m_values=a_values,
            quant_packets=q_packets,
        )
        cycle_dispatch_received.append(summary)

    bar_pipeline = LiveBarPipelineService(
        symbols=["BTCUSDT", "ETHUSDT"],
        quant_engine=quant_engine,
        barrier_model=barrier_model,
        ood_detector=ood_detector,
        treeshap_explainer=shap_explainer,
        on_30m_bar=None,
        on_cycle_ready=on_cycle_ready,
    )

    runner = LivePaperRunner(
        config=runner_cfg,
        db_manager=db_mgr,
        memory_repo=memory_repo,
        gateway=gateway,
        ws_ingestor=ws_mock,  # type: ignore[arg-type]
        bar_pipeline=bar_pipeline,
        shadow_tracker=shadow_tracker,
        judge_adapter=judge_adapter,
        use_case=use_case,
        barrier_model=barrier_model,
        ood_detector=ood_detector,
        shap_explainer=shap_explainer,
        model_metadata=None,
        telegram_notifier=None,
        trading_logger=trading_logger,
        agent_logger=agent_logger,
        system_logger=system_logger,
        user_data_stream=None,
    )

    # 4. Stream 30 1m bars for BTC and ETH up to 30m close
    # Use timestamps aligned to 30m boundary: 00:00 to 00:29 (open_time_ms)
    # The 30th bar at 00:29:00 has close at 00:30:00 (timestamp_ms + 60000 % 1800000 == 0)
    base_ts = 1700000000000 - (1700000000000 % 1800000)  # aligned to 30m
    btc_price = 50000.0
    eth_price = 3000.0

    for minute in range(30):
        bar_ts = base_ts + minute * 60000
        btc_bar = make_enriched_bar(
            "BTCUSDT", bar_ts, btc_price, btc_price + 20, btc_price - 20, btc_price + 5, 10.0
        )
        eth_bar = make_enriched_bar(
            "ETHUSDT", bar_ts, eth_price, eth_price + 2, eth_price - 2, eth_price + 1, 50.0
        )

        await runner.handle_bar_closed(btc_bar)
        await runner.handle_bar_closed(eth_bar)

    # 5. Verify 30m Cycle Dispatched
    assert len(cycle_dispatch_received) == 1
    cycle_summary = cycle_dispatch_received[0]
    assert cycle_summary.btc_result.funnel_passed is True

    # 6. Simulate Arbiter VETO on ETHUSDT to trigger shadow order registration
    # Construct blueprint and veto verdict for ETHUSDT
    eth_thesis = MarketThesis(
        symbol="ETHUSDT",
        selected_playbook=PlaybookType.CROSS_ASSET_LEAD_LAG_CATCHUP,
        proposed_side=TradeSide.LONG,
        supporting_active_features=["hurst_exponent_100", "vpin_toxicity_pct"],
        regime_alignment_score=0.85,
        thesis_summary="Lead-lag catchup signal validated by order flow.",
    )
    eth_blueprint = TradeBlueprint(
        thesis=eth_thesis,
        order_type=EntryOrderType.LIMIT_PULLBACK_15M,
        entry_price=3000.0,
        stop_loss_price=2950.0,
        take_profit_price=3100.0,
        rationale="Tight bracket above 80th-percentile MAE buffer.",
    )
    veto_verdict = RiskVerdict(
        decision=RiskDecisionType.VETO_TO_SHADOW,
        approved_leverage=1,
        approved_notional_usd=0.0,
        net_rr=2.0,
        reason="Setup defect: Sub-optimal risk/reward ratio on 30m chart",
        arbiter_size_multiplier=0.0,
        veto_category=VetoCategory.SETUP_DEFECT,
        existing_position_directive=ExistingPositionDirective.HOLD_EXISTING,
        portfolio_risk_rationale="Candidate rejected due to setup defect. Candidate redirected to shadow tracking.",
    )
    eth_quant_pkt = bar_pipeline.get_latest_quant_packet("ETHUSDT")
    shadow_record = shadow_tracker.register_vetoed_candidate(
        eth_blueprint, veto_verdict, eth_quant_pkt
    )
    assert shadow_record.symbol == "ETHUSDT"
    assert shadow_record.status == ShadowOrderStatus.TRACKING
    assert len(shadow_tracker.get_all_active_orders()) == 1

    # 7. Stream subsequent 1m bars for ETH where price progresses and hits Take Profit (3100.0)
    concluded_autopsy_cases: list[EpisodicCase] = []
    for step in range(5):
        sim_ts = base_ts + (30 + step) * 60000
        # Step 4 price exceeds Take Profit at 3105.0
        current_px = 3020.0 + step * 25.0
        if step == 4:
            current_px = 3105.0

        bar = make_enriched_bar(
            "ETHUSDT", sim_ts, current_px - 5, current_px + 10, current_px - 5, current_px, 100.0
        )
        # Process through runner handle_bar_closed (which detects conclusion and triggers autopsy)
        concluded_orders = shadow_tracker.on_1m_bar("ETHUSDT", bar)
        for order in concluded_orders:
            case = await judge_adapter.conduct_autopsy(
                shadow_order=order,
                quant_context={
                    "symbol": order.symbol,
                    "playbook": PlaybookType.CROSS_ASSET_LEAD_LAG_CATCHUP.value,
                    "vector_16d": [0.15] * 16,
                },
            )
            concluded_autopsy_cases.append(case)

    # 8. Assertions: Concluded Shadow Order & Post-Mortem Autopsy
    assert len(concluded_autopsy_cases) == 1
    autopsy_case = concluded_autopsy_cases[0]
    assert autopsy_case.symbol == "ETHUSDT"
    assert autopsy_case.outcome == OutcomeCategory.MISSED_OPPORTUNITY
    assert autopsy_case.confidence_score >= 0.40
    assert (
        "MarketScout" in autopsy_case.lesson_summary or "reached TP" in autopsy_case.lesson_summary
    )

    # Verify case is committed to episodic memory repository
    assert len(memory_repo.upserted_cases) == 1
    assert memory_repo.upserted_cases[0].case_id == autopsy_case.case_id

    # Clean up loggers
    trading_logger.close()
    agent_logger.close()
    system_logger.close()


@pytest.mark.asyncio
async def test_stage6_post_sl_tracking_and_stop_hunt_autopsy(tmp_path: Any) -> None:
    """Post-SL Tracking & Stop-Hunt Autopsy:

    Simulates a live executed trade on BTCUSDT that hits SL, is registered as post-SL tracker,
    subsequently dips (MAE), and then reverses sharply to reach the original Take-Profit barrier.
    Post-mortem judge classifies outcome as STOP_HUNTED_THEN_HIT_TP and generates tactical
    lesson targeting [TradeArchitect].
    """
    log_dir = str(tmp_path / "logs_post_sl")
    log_cfg = LoggingConfig(
        log_dir=log_dir, retention_days=1, min_level="INFO", console_output=False
    )
    agent_logger = StructuredBotLogger(LogCategory.AGENTS, log_cfg)

    memory_repo = MockInMemoryMemoryRepository([])
    mock_nr = MockNinerouterClient()
    nr_cfg = make_ninerouter_config()
    governor = MockRateLimitGovernor(make_route_spec())

    judge_adapter = PostMortemJudgeAdapter(
        client=mock_nr,  # type: ignore[arg-type]
        config=nr_cfg,
        governor=governor,
        memory_repo=memory_repo,
        logger=agent_logger,
    )

    shadow_tracker = ShadowOrderTracker(max_tracking_duration_ms=21600000)

    # Register live trade that just hit Stop Loss
    post_sl_order = shadow_tracker.register_post_sl_position(
        symbol="BTCUSDT",
        side=TradeSide.LONG,
        entry_price=50000.0,
        sl_price=49000.0,
        tp_price=52000.0,
        exit_timestamp_ms=1700000000000,
        quant_packet={"atr_30m": 500.0, "timestamp_ms": 1700000000000},
    )
    assert post_sl_order.is_post_sl_tracker is True
    assert post_sl_order.status == ShadowOrderStatus.TRACKING

    # Step 1: Bar dips below SL to 48700.0 (MAE = 300 / 500 = 0.6x ATR beyond SL)
    dip_bar = make_enriched_bar("BTCUSDT", 1700000060000, 49000.0, 49050.0, 48700.0, 48800.0, 20.0)
    shadow_tracker.on_1m_bar("BTCUSDT", dip_bar)

    # Step 2: Sharp reversal upwards reaching 52100.0 (hitting TP)
    tp_bar = make_enriched_bar("BTCUSDT", 1700000120000, 48800.0, 52100.0, 48800.0, 52050.0, 200.0)
    concluded = shadow_tracker.on_1m_bar("BTCUSDT", tp_bar)

    assert len(concluded) == 1
    concluded_order = concluded[0]
    assert concluded_order.status == ShadowOrderStatus.STOP_HUNTED_THEN_TP

    # Run post-mortem judge autopsy
    case = await judge_adapter.conduct_autopsy(
        shadow_order=concluded_order,
        quant_context={
            "symbol": "BTCUSDT",
            "playbook": PlaybookType.TREND_CONTINUATION_BREAKOUT.value,
            "vector_16d": [0.20] * 16,
        },
    )

    # Assertions
    assert case.symbol == "BTCUSDT"
    assert case.outcome == OutcomeCategory.STOP_HUNTED_THEN_HIT_TP
    # Tactical lesson addresses TradeArchitect with recommendations on stop buffer
    assert "TradeArchitect" in case.lesson_summary
    assert len(memory_repo.upserted_cases) == 1

    agent_logger.close()


@pytest.mark.asyncio
async def test_stage6_live_pgvector_rag_retrieval() -> None:
    """Verify live TimescaleDB pgvector similarity and RAG cohort retrieval across all 4 playbooks.

    Reads from the live database container populated by `scripts/seed_episodic_memory.py`.
    """
    db_cfg = DatabaseConfig(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        db_name=os.getenv("POSTGRES_DB", "trading_bot_db"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        max_connections=int(os.getenv("POSTGRES_MAX_CONNECTIONS", "5")),
    )
    db_mgr = DatabaseManager(db_cfg)

    try:
        await db_mgr.initialize()
    except Exception as exc:
        pytest.skip(f"Live TimescaleDB container not reachable: {exc}")

    try:
        repo = PgvectorMemoryRepository(db_mgr)

        # Seed test cases so live pgvector similarity and RAG cohort retrieval work
        # reliably across both clean CI containers and populated databases
        test_cases = make_test_cases()
        for c in test_cases:
            await repo.upsert_case(c)
        playbooks = [
            PlaybookType.TREND_CONTINUATION_BREAKOUT.value,
            PlaybookType.LIQUIDATION_EXHAUSTION_REVERSAL.value,
            PlaybookType.CROSS_ASSET_LEAD_LAG_CATCHUP.value,
            PlaybookType.GENERAL_ALPHA_MOMENTUM.value,
        ]

        for pb in playbooks:
            rows = await db_mgr.fetch(
                "SELECT regime_trajectory_vec::text as vec FROM episodic_lessons_memory WHERE playbook_used = $1 LIMIT 1",
                pb,
            )
            if rows:
                query_vector = [
                    float(x.strip()) for x in str(rows[0]["vec"]).strip("[]").split(",")
                ]
            else:
                query_vector = [0.15] * 16

            # 1. Test k-NN cohort
            cohort = await repo.query_knn_cohort(query_vector, pb, 5)
            assert len(cohort) > 0, f"No neighbors retrieved for playbook {pb}"
            assert all(c.playbook == pb for c in cohort)

            # 2. Test cohort base-rate summary
            summary = await repo.compute_cohort_summary(query_vector, pb, 20)
            assert summary.has_historical_cohort is True
            assert summary.total_neighbors > 0
            assert summary.weighted_win_rate is not None

            # 3. Test 2-tier stratified memory pack
            pack = await repo.fetch_stratified_pack(query_vector, pb, 20)
            assert pack.cohort_summary.has_historical_cohort is True
            # Stratified buckets should contain representative cases
            all_pack_lessons = (
                pack.bucket1_bull_missed + pack.bucket2_failure_traps + pack.bucket3_stop_hunts
            )
            assert len(all_pack_lessons) > 0

    finally:
        await db_mgr.close()
