"""Unit tests for Risk Arbiter Agent Adapter (arbiter_agent.py).

Enforces Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from trading_bot.adapters.agents.arbiter_agent import RiskArbiterAdapter
from trading_bot.adapters.agents.ninerouter_client import (
    NinerouterClient,
    NinerouterResponse,
)
from trading_bot.domain.entities.common.enums import (
    EntryOrderType,
    ExecutionMode,
    ExistingPositionDirective,
    PlaybookType,
    RiskDecisionType,
    TradeSide,
    VetoCategory,
)
from trading_bot.domain.entities.execution.trade_blueprint import (
    MarketThesis,
    RiskVerdict,
    TradeBlueprint,
)
from trading_bot.domain.entities.memory.episodic_case import (
    CohortBaseRateSummary,
    StratifiedMemoryPack,
)
from trading_bot.domain.entities.risk.account_state import AccountPortfolioContext
from trading_bot.domain.services.risk.risk_calculator import (
    DeterministicRiskCalculator,
    RiskConfig,
)
from trading_bot.infrastructure.logging.bot_logger import StructuredBotLogger


@pytest.fixture
def risk_config() -> RiskConfig:
    return RiskConfig(
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0005,
        min_tp_to_cost_multiple=3.0,
        min_net_rr=1.5,
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


@pytest.fixture
def account_context() -> AccountPortfolioContext:
    return AccountPortfolioContext(
        execution_mode=ExecutionMode.PAPER,
        timestamp_ms=1700000000000,
        wallet_balance_usd=10000.0,
        unrealized_pnl_usd=0.0,
        equity_nav_usd=10000.0,
        used_margin_usd=1000.0,
        available_margin_usd=9000.0,
        margin_utilization_pct=0.10,
        current_daily_pnl_usd=50.0,
        current_daily_dd_pct=0.0,
        remaining_daily_dd_budget_pct=0.04,
        current_max_dd_pct=0.01,
        remaining_max_dd_budget_pct=0.07,
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


@pytest.fixture
def valid_blueprint() -> TradeBlueprint:
    thesis = MarketThesis(
        symbol="BTCUSDT",
        selected_playbook=PlaybookType.TREND_CONTINUATION_BREAKOUT,
        proposed_side=TradeSide.LONG,
        supporting_active_features=["hurst_exp"],
        regime_alignment_score=0.85,
        thesis_summary="Bullish trend setup",
    )
    return TradeBlueprint(
        thesis=thesis,
        order_type=EntryOrderType.LIMIT_PULLBACK_15M,
        entry_price=60000.0,
        stop_loss_price=59400.0,  # 1.0% SL
        take_profit_price=61500.0,  # 2.5% TP
        rationale="15m VWAP test entry",
    )


@pytest.fixture
def memory_pack() -> StratifiedMemoryPack:
    cohort = CohortBaseRateSummary(
        has_historical_cohort=True,
        total_neighbors=20,
        weighted_win_rate=0.62,
        stop_hunted_then_tp_rate=0.10,
        true_failure_trap_rate=0.15,
        expected_r_multiple=1.8,
        mae_target_percentile_atr=1.8,
        mfe_median_atr=3.0,
    )
    return StratifiedMemoryPack(
        cohort_summary=cohort,
        bucket1_bull_missed=[],
        bucket2_failure_traps=[],
        bucket3_stop_hunts=[],
    )


@pytest.mark.asyncio
async def test_arbiter_deterministic_fallback_mode_emits_critical_log(
    risk_config: RiskConfig,
    account_context: AccountPortfolioContext,
    valid_blueprint: TradeBlueprint,
    memory_pack: StratifiedMemoryPack,
) -> None:
    mock_client: AsyncMock = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = NinerouterResponse(
        content="",
        model="deterministic_fallback",
        finish_reason="fallback",
        prompt_tokens=0,
        completion_tokens=0,
        is_fallback=True,
    )

    mock_logger: MagicMock = MagicMock(spec=StructuredBotLogger)
    calc: DeterministicRiskCalculator = DeterministicRiskCalculator(risk_config)

    adapter: RiskArbiterAdapter = RiskArbiterAdapter(
        client=mock_client,
        tier="deep_reasoning",
        deterministic_calculator=calc,
        logger=mock_logger,
    )

    verdict: RiskVerdict = await adapter.evaluate_risk(
        blueprint=valid_blueprint,
        account_context=account_context,
        memory_pack=memory_pack,
    )

    # 1. Verify CRITICAL log was emitted
    mock_logger.critical.assert_called_once()
    critical_msg = mock_logger.critical.call_args[0][0]
    assert "9Router offline/exhausted. Deterministic Rule-Only Fallback applied" in critical_msg

    # 2. Verify deterministic calculation approved trade and set valid limits
    assert verdict.decision == RiskDecisionType.APPROVE_FULL
    assert verdict.approved_notional_usd > 0.0
    assert 1 <= verdict.approved_leverage <= 10
    assert verdict.arbiter_size_multiplier == 1.0
    assert verdict.existing_position_directive == ExistingPositionDirective.HOLD_EXISTING
    assert "deterministic rule-only fallback" in verdict.portfolio_risk_rationale.lower()


@pytest.mark.asyncio
async def test_arbiter_deterministic_fallback_rejection_on_drawdown(
    risk_config: RiskConfig,
    account_context: AccountPortfolioContext,
    valid_blueprint: TradeBlueprint,
) -> None:
    # Breached daily drawdown (5% vs 4% limit)
    account_context.current_daily_dd_pct = 0.05

    mock_client: AsyncMock = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = NinerouterResponse(
        content="",
        model="deterministic_fallback",
        finish_reason="fallback",
        prompt_tokens=0,
        completion_tokens=0,
        is_fallback=True,
    )

    mock_logger: MagicMock = MagicMock(spec=StructuredBotLogger)
    calc: DeterministicRiskCalculator = DeterministicRiskCalculator(risk_config)

    adapter: RiskArbiterAdapter = RiskArbiterAdapter(
        client=mock_client,
        tier="deep_reasoning",
        deterministic_calculator=calc,
        logger=mock_logger,
    )

    verdict: RiskVerdict = await adapter.evaluate_risk(
        blueprint=valid_blueprint,
        account_context=account_context,
        memory_pack=None,
    )

    assert verdict.decision == RiskDecisionType.VETO_TO_SHADOW
    assert verdict.approved_notional_usd == 0.0
    assert verdict.approved_leverage == 1
    assert verdict.veto_category == VetoCategory.PORTFOLIO_CAPACITY_VETO
    assert "CIRCUIT_BREAKER" in str(verdict.reason)


@pytest.mark.asyncio
async def test_arbiter_bounds_llm_approved_size_and_leverage(
    risk_config: RiskConfig,
    account_context: AccountPortfolioContext,
    valid_blueprint: TradeBlueprint,
) -> None:
    # LLM hallucinates an excessive approval: notional 100,000 USD and leverage 50x
    mock_client: AsyncMock = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = NinerouterResponse(
        content=(
            "```json\n"
            "{\n"
            '  "decision": "APPROVE_FULL",\n'
            '  "approved_leverage": 50,\n'
            '  "approved_notional_usd": 100000.0,\n'
            '  "net_rr": 2.1,\n'
            '  "reason": null,\n'
            '  "arbiter_size_multiplier": 1.0,\n'
            '  "veto_category": null,\n'
            '  "existing_position_directive": "HOLD_EXISTING",\n'
            '  "portfolio_risk_rationale": "High conviction."\n'
            "}\n"
            "```"
        ),
        model="claude-3-7-sonnet",
        finish_reason="stop",
        prompt_tokens=2200,
        completion_tokens=150,
        is_fallback=False,
    )

    calc: DeterministicRiskCalculator = DeterministicRiskCalculator(risk_config)
    adapter: RiskArbiterAdapter = RiskArbiterAdapter(
        client=mock_client,
        tier="deep_reasoning",
        deterministic_calculator=calc,
        logger=None,
    )

    verdict: RiskVerdict = await adapter.evaluate_risk(
        blueprint=valid_blueprint,
        account_context=account_context,
        memory_pack=None,
    )

    # Invariant: leverage must be bounded by deterministic calculator max_leverage (10)
    assert verdict.approved_leverage <= 10
    # Invariant: notional must be bounded by Quarter-Kelly / max risk math
    assert verdict.approved_notional_usd < 100000.0
    assert verdict.decision == RiskDecisionType.APPROVE_FULL


@pytest.mark.asyncio
async def test_arbiter_deterministic_veto_overrides_llm_approval(
    risk_config: RiskConfig,
    account_context: AccountPortfolioContext,
    valid_blueprint: TradeBlueprint,
) -> None:
    # Breached daily drawdown
    account_context.current_daily_dd_pct = 0.05

    # LLM attempts to approve anyway
    mock_client: AsyncMock = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = NinerouterResponse(
        content=(
            "```json\n"
            "{\n"
            '  "decision": "APPROVE_FULL",\n'
            '  "approved_leverage": 3,\n'
            '  "approved_notional_usd": 3000.0,\n'
            '  "net_rr": 2.1,\n'
            '  "reason": null,\n'
            '  "arbiter_size_multiplier": 1.0,\n'
            '  "veto_category": null,\n'
            '  "existing_position_directive": "HOLD_EXISTING",\n'
            '  "portfolio_risk_rationale": "Approve despite drawdown."\n'
            "}\n"
            "```"
        ),
        model="claude-3-7-sonnet",
        finish_reason="stop",
        prompt_tokens=2200,
        completion_tokens=150,
        is_fallback=False,
    )

    calc: DeterministicRiskCalculator = DeterministicRiskCalculator(risk_config)
    adapter: RiskArbiterAdapter = RiskArbiterAdapter(
        client=mock_client,
        tier="deep_reasoning",
        deterministic_calculator=calc,
        logger=None,
    )

    verdict: RiskVerdict = await adapter.evaluate_risk(
        blueprint=valid_blueprint,
        account_context=account_context,
        memory_pack=None,
    )

    # Deterministic safety gate overrides LLM approval!
    assert verdict.decision == RiskDecisionType.VETO_TO_SHADOW
    assert verdict.approved_notional_usd == 0.0
    assert verdict.approved_leverage == 1
    assert "CIRCUIT_BREAKER" in str(verdict.reason)


@pytest.mark.asyncio
async def test_arbiter_flat_thesis_immediate_veto(
    risk_config: RiskConfig,
    account_context: AccountPortfolioContext,
) -> None:
    flat_thesis = MarketThesis(
        symbol="BTCUSDT",
        selected_playbook=PlaybookType.NO_TRADE_REGIME,
        proposed_side=TradeSide.FLAT,
        supporting_active_features=[],
        regime_alignment_score=0.5,
        thesis_summary="No trade regime",
    )
    flat_blueprint = TradeBlueprint(
        thesis=flat_thesis,
        order_type=EntryOrderType.LIMIT_PULLBACK_15M,
        entry_price=60000.0,
        stop_loss_price=59400.0,
        take_profit_price=61500.0,
        rationale="Flat rationale",
    )

    mock_client: AsyncMock = AsyncMock(spec=NinerouterClient)
    calc: DeterministicRiskCalculator = DeterministicRiskCalculator(risk_config)
    adapter: RiskArbiterAdapter = RiskArbiterAdapter(
        client=mock_client,
        tier="deep_reasoning",
        deterministic_calculator=calc,
        logger=None,
    )

    verdict: RiskVerdict = await adapter.evaluate_risk(
        blueprint=flat_blueprint,
        account_context=account_context,
        memory_pack=None,
    )

    assert verdict.decision == RiskDecisionType.VETO_TO_SHADOW
    assert verdict.approved_notional_usd == 0.0
    assert not mock_client.generate_chat_completion.called
