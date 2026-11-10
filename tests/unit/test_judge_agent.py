"""Unit tests for PostMortemJudgeAdapter (judge_agent.py).

Tests Gemini model routing via 9Router, fallback on 404, prompt generation,
tactical lesson synthesis, confidence score adjustments, and repository upsert.
Enforces Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from trading_bot.adapters.agents.judge_agent import PostMortemJudgeAdapter
from trading_bot.adapters.agents.ninerouter_client import (
    NinerouterClient,
    NinerouterResponse,
)
from trading_bot.domain.entities.common.enums import (
    OutcomeCategory,
    ShadowOrderStatus,
    TradeSide,
    VetoCategory,
)
from trading_bot.domain.entities.execution.shadow_order import ShadowOrderRecord
from trading_bot.domain.entities.memory.episodic_case import (
    CohortBaseRateSummary,
    EpisodicCase,
    StratifiedMemoryPack,
)
from trading_bot.infrastructure.config_schemas import NinerouterConfig
from trading_bot.ports.memory_repository_port import MemoryRepositoryPort
from trading_bot.ports.rate_limit_governor_port import (
    RateLimitGovernorPort,
    RouteQuotaSpec,
)


class MockMemoryRepository(MemoryRepositoryPort):
    """Mock repository recording upserted cases and boosted lessons."""

    def __init__(self) -> None:
        self.upserted_cases: list[EpisodicCase] = []
        self.boosted_lessons: list[tuple[str, float]] = []

    async def query_knn_cohort(
        self,
        vector_16d: list[float],
        playbook: str,
        k: int,
    ) -> list[EpisodicCase]:
        return []

    async def compute_cohort_summary(
        self,
        vector_16d: list[float],
        playbook: str,
        k: int,
    ) -> CohortBaseRateSummary:
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

    async def fetch_stratified_pack(
        self,
        vector_16d: list[float],
        playbook: str,
        k: int,
    ) -> StratifiedMemoryPack:
        summary = await self.compute_cohort_summary(vector_16d, playbook, k)
        return StratifiedMemoryPack(
            cohort_summary=summary,
            bucket1_bull_missed=[],
            bucket2_failure_traps=[],
            bucket3_stop_hunts=[],
        )

    async def upsert_case(self, case: EpisodicCase) -> None:
        self.upserted_cases.append(case)

    async def boost_lesson_confidence(self, case_id: str, delta: float) -> None:
        self.boosted_lessons.append((case_id, delta))


class MockGovernor(RateLimitGovernorPort):
    """Mock RateLimitGovernor allocating fake routes."""

    async def acquire_available_route(
        self,
        tier: str,
        estimated_tokens: int,
    ) -> RouteQuotaSpec:
        return RouteQuotaSpec(
            route_id=f"route_{tier}",
            model_name="gemini-2.5-flash",
            rpm_limit=15,
            tpm_limit=100000,
            rpd_limit=1500,
            min_interval_ms=100,
            max_concurrency=2,
        )

    async def mark_route_cooldown(self, route_id: str, cooldown_seconds: float) -> None:
        pass


@pytest.fixture
def ninerouter_config() -> NinerouterConfig:
    return NinerouterConfig(
        base_url="http://localhost:20128/v1",
        api_key="mock_key",
        timeout_seconds=30.0,
        fast_tier_model="ds/deepseek-v4-flash",
        deep_tier_model="ds/deepseek-v4-pro",
        judge_tier_model="gemini-2.5-flash",
    )


@pytest.fixture
def sample_shadow_order() -> ShadowOrderRecord:
    return ShadowOrderRecord(
        shadow_id="shadow_btc_101",
        parent_order_id=None,
        symbol="BTCUSDT",
        side=TradeSide.LONG,
        entry_price=65000.0,
        stop_loss_price=64000.0,
        take_profit_price=67000.0,
        veto_category=VetoCategory.SETUP_DEFECT,
        is_post_sl_tracker=False,
        entry_timestamp_ms=1710000000000,
        max_tracking_duration_ms=21600000,
        mae_price=64200.0,
        mfe_price=67200.0,
        mae_atr_multiple=0.8,
        mfe_atr_multiple=2.2,
        status=ShadowOrderStatus.HIT_TP,
    )


@pytest.fixture
def sample_quant_context() -> dict[str, Any]:
    return {
        "playbook": "TREND_CONTINUATION_BREAKOUT",
        "vector_16d": [0.15] * 16,
        "applied_case_ids": ["prior_case_001", "prior_case_002"],
    }


@pytest.mark.asyncio
async def test_successful_judge_gemini_call(
    ninerouter_config: NinerouterConfig,
    sample_shadow_order: ShadowOrderRecord,
    sample_quant_context: dict[str, Any],
) -> None:
    """Test successful Gemini response parsing and EpisodicCase creation."""
    client = MagicMock(spec=NinerouterClient)
    gemini_json = json.dumps(
        {
            "target_agent": "MarketScout",
            "outcome_assessment": "Vetoed long trade reached TP cleanly.",
            "root_cause": "OFI and micro-momentum were genuine drivers.",
            "tactical_lesson": "Do not veto breakout setups when Yang-Zhang volatility expansion exceeds 1.5x.",
        }
    )
    client.generate_chat_completion = AsyncMock(
        return_value=NinerouterResponse(
            content=gemini_json,
            model="gemini-2.5-flash",
            finish_reason="stop",
            prompt_tokens=450,
            completion_tokens=120,
            is_fallback=False,
        )
    )

    memory_repo = MockMemoryRepository()
    governor = MockGovernor()

    adapter = PostMortemJudgeAdapter(
        client=client,
        config=ninerouter_config,
        governor=governor,
        memory_repo=memory_repo,
        logger=None,
    )

    case = await adapter.conduct_autopsy(sample_shadow_order, sample_quant_context)

    # Verify LLM called with tier="judge"
    client.generate_chat_completion.assert_called_once()
    assert client.generate_chat_completion.call_args.kwargs["tier"] == "judge"

    # Verify outcome classification
    assert case.outcome == OutcomeCategory.MISSED_OPPORTUNITY
    # Baseline 0.55 - 0.12 = 0.43 for MISSED_OPPORTUNITY
    assert case.confidence_score == 0.43
    assert case.symbol == "BTCUSDT"
    assert len(case.vector_16d) == 16
    assert "[MarketScout]" in case.lesson_summary
    assert "Yang-Zhang volatility" in case.lesson_summary

    # Verify upserted to memory repo
    assert len(memory_repo.upserted_cases) == 1
    assert memory_repo.upserted_cases[0].case_id == case.case_id

    # Verify prior case IDs were penalized (-0.12)
    assert len(memory_repo.boosted_lessons) == 2
    assert memory_repo.boosted_lessons[0] == ("prior_case_001", -0.12)
    assert memory_repo.boosted_lessons[1] == ("prior_case_002", -0.12)


@pytest.mark.asyncio
async def test_fallback_on_404_to_deep_tier(
    ninerouter_config: NinerouterConfig,
    sample_shadow_order: ShadowOrderRecord,
    sample_quant_context: dict[str, Any],
) -> None:
    """Test fallback from judge tier to deep tier when judge returns 404."""
    client = MagicMock(spec=NinerouterClient)

    deep_json = json.dumps(
        {
            "target_agent": "MarketScout",
            "outcome_assessment": "Deep tier evaluated missed opportunity.",
            "root_cause": "Scout was too conservative.",
            "tactical_lesson": "Deep tier lesson: lower entry threshold in bull trend.",
        }
    )

    # First call (judge) raises 404, second call (deep) succeeds
    client.generate_chat_completion = AsyncMock(
        side_effect=[
            RuntimeError("9Router error [404]: Model gemini-2.5-flash not found"),
            NinerouterResponse(
                content=deep_json,
                model="ds/deepseek-v4-pro",
                finish_reason="stop",
                prompt_tokens=400,
                completion_tokens=100,
                is_fallback=False,
            ),
        ]
    )

    memory_repo = MockMemoryRepository()
    governor = MockGovernor()

    adapter = PostMortemJudgeAdapter(
        client=client,
        config=ninerouter_config,
        governor=governor,
        memory_repo=memory_repo,
        logger=None,
    )

    case = await adapter.conduct_autopsy(sample_shadow_order, sample_quant_context)

    assert client.generate_chat_completion.call_count == 2
    assert client.generate_chat_completion.call_args_list[0].kwargs["tier"] == "judge"
    assert client.generate_chat_completion.call_args_list[1].kwargs["tier"] == "deep"

    assert "[MarketScout]" in case.lesson_summary
    assert "Deep tier lesson" in case.lesson_summary


@pytest.mark.asyncio
async def test_fallback_to_deterministic_rule_when_both_fail(
    ninerouter_config: NinerouterConfig,
    sample_shadow_order: ShadowOrderRecord,
    sample_quant_context: dict[str, Any],
) -> None:
    """Test fallback to deterministic lesson synthesis when both judge and deep fail."""
    client = MagicMock(spec=NinerouterClient)
    # Both judge and deep raise connection errors
    client.generate_chat_completion = AsyncMock(
        side_effect=[
            ConnectionRefusedError("Connection to 9Router failed on judge tier"),
            ConnectionRefusedError("Connection to 9Router failed on deep tier"),
        ]
    )

    memory_repo = MockMemoryRepository()
    governor = MockGovernor()

    adapter = PostMortemJudgeAdapter(
        client=client,
        config=ninerouter_config,
        governor=governor,
        memory_repo=memory_repo,
        logger=None,
    )

    case = await adapter.conduct_autopsy(sample_shadow_order, sample_quant_context)

    # Must produce factual deterministic lesson
    assert "[MarketScout]" in case.lesson_summary
    assert "BTCUSDT" in case.lesson_summary
    assert "vetoed candidate reached TP" in case.lesson_summary
    assert case.confidence_score == 0.43


@pytest.mark.asyncio
async def test_confidence_scores_across_all_outcomes(
    ninerouter_config: NinerouterConfig,
) -> None:
    """Test confidence score update logic across all outcome categories."""
    client = MagicMock(spec=NinerouterClient)
    client.generate_chat_completion = AsyncMock(
        side_effect=RuntimeError("Offline")
    )  # triggers deterministic fallback

    memory_repo = MockMemoryRepository()
    governor = MockGovernor()

    adapter = PostMortemJudgeAdapter(
        client=client,
        config=ninerouter_config,
        governor=governor,
        memory_repo=memory_repo,
        logger=None,
    )

    # 1. VALIDATED_DEFENSE: +0.08 -> 0.55 + 0.08 = 0.63
    order_def = ShadowOrderRecord(
        shadow_id="shadow_def_01",
        parent_order_id=None,
        symbol="BTCUSDT",
        side=TradeSide.LONG,
        entry_price=65000.0,
        stop_loss_price=64000.0,
        take_profit_price=67000.0,
        veto_category=VetoCategory.SETUP_DEFECT,
        is_post_sl_tracker=False,
        entry_timestamp_ms=1710000000000,
        max_tracking_duration_ms=21600000,
        mae_price=63900.0,
        mfe_price=65100.0,
        mae_atr_multiple=1.1,
        mfe_atr_multiple=0.1,
        status=ShadowOrderStatus.HIT_SL,
    )
    case_def = await adapter.conduct_autopsy(
        order_def, {"vector_16d": [0.0] * 16, "playbook": "TREND_CONTINUATION_BREAKOUT"}
    )
    assert case_def.outcome == OutcomeCategory.VALIDATED_DEFENSE
    assert case_def.confidence_score == 0.63
    assert "[RiskArbiter]" in case_def.lesson_summary

    # 2. CAPACITY_SKIPPED_WIN: 0.00 delta -> 0.55 (IMMUNITY PROTECTED)
    order_cap = ShadowOrderRecord(
        shadow_id="shadow_cap_01",
        parent_order_id=None,
        symbol="ETHUSDT",
        side=TradeSide.LONG,
        entry_price=3500.0,
        stop_loss_price=3400.0,
        take_profit_price=3700.0,
        veto_category=VetoCategory.PORTFOLIO_CAPACITY_VETO,
        is_post_sl_tracker=False,
        entry_timestamp_ms=1710000000000,
        max_tracking_duration_ms=21600000,
        mae_price=3480.0,
        mfe_price=3720.0,
        mae_atr_multiple=0.2,
        mfe_atr_multiple=2.2,
        status=ShadowOrderStatus.HIT_TP,
    )
    case_cap = await adapter.conduct_autopsy(
        order_cap, {"vector_16d": [0.0] * 16, "playbook": "TREND_CONTINUATION_BREAKOUT"}
    )
    assert case_cap.outcome == OutcomeCategory.CAPACITY_SKIPPED_WIN
    assert case_cap.confidence_score == 0.55
    assert "immunity protection" in case_cap.lesson_summary

    # 3. STOP_HUNTED_THEN_HIT_TP: 0.00 delta -> 0.55
    order_sh = ShadowOrderRecord(
        shadow_id="shadow_sh_01",
        parent_order_id="live_01",
        symbol="SOLUSDT",
        side=TradeSide.LONG,
        entry_price=150.0,
        stop_loss_price=145.0,
        take_profit_price=160.0,
        veto_category=None,
        is_post_sl_tracker=True,
        entry_timestamp_ms=1710000000000,
        max_tracking_duration_ms=21600000,
        mae_price=144.0,
        mfe_price=161.0,
        mae_atr_multiple=0.6,
        mfe_atr_multiple=2.2,
        status=ShadowOrderStatus.STOP_HUNTED_THEN_TP,
    )
    case_sh = await adapter.conduct_autopsy(
        order_sh, {"vector_16d": [0.0] * 16, "playbook": "TREND_CONTINUATION_BREAKOUT"}
    )
    assert case_sh.outcome == OutcomeCategory.STOP_HUNTED_THEN_HIT_TP
    assert case_sh.confidence_score == 0.55
    assert "[TradeArchitect]" in case_sh.lesson_summary

    # 4. FALSE_POSITIVE_TRAP: -0.12 delta -> 0.55 - 0.12 = 0.43
    order_trap = ShadowOrderRecord(
        shadow_id="shadow_trap_01",
        parent_order_id="live_02",
        symbol="BNBUSDT",
        side=TradeSide.LONG,
        entry_price=600.0,
        stop_loss_price=590.0,
        take_profit_price=620.0,
        veto_category=None,
        is_post_sl_tracker=True,
        entry_timestamp_ms=1710000000000,
        max_tracking_duration_ms=21600000,
        mae_price=580.0,
        mfe_price=602.0,
        mae_atr_multiple=2.0,
        mfe_atr_multiple=0.2,
        status=ShadowOrderStatus.TIMEOUT_CLOSED,
    )
    case_trap = await adapter.conduct_autopsy(
        order_trap, {"vector_16d": [0.0] * 16, "playbook": "TREND_CONTINUATION_BREAKOUT"}
    )
    assert case_trap.outcome == OutcomeCategory.FALSE_POSITIVE_TRAP
    assert case_trap.confidence_score == 0.43
    assert "[RiskArbiter]" in case_trap.lesson_summary


@pytest.mark.asyncio
async def test_validation_errors_on_missing_or_invalid_vector(
    ninerouter_config: NinerouterConfig,
    sample_shadow_order: ShadowOrderRecord,
) -> None:
    """Verify KeyError and ValueError on invalid 16D vector context."""
    client = MagicMock(spec=NinerouterClient)
    memory_repo = MockMemoryRepository()
    governor = MockGovernor()

    adapter = PostMortemJudgeAdapter(
        client=client,
        config=ninerouter_config,
        governor=governor,
        memory_repo=memory_repo,
        logger=None,
    )

    # Missing vector
    with pytest.raises(KeyError, match="vector_16d"):
        await adapter.conduct_autopsy(sample_shadow_order, {"playbook": "GENERAL"})

    # Invalid vector length (15 instead of 16)
    with pytest.raises(ValueError, match="vector_16d must be a list of 16 floats"):
        await adapter.conduct_autopsy(
            sample_shadow_order, {"vector_16d": [0.1] * 15, "playbook": "GENERAL"}
        )
