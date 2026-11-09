"""Unit tests for Trade Architect Agent Adapter (architect_agent.py).

Enforces Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from trading_bot.adapters.agents.architect_agent import TradeArchitectAdapter
from trading_bot.adapters.agents.ninerouter_client import (
    NinerouterClient,
    NinerouterResponse,
)
from trading_bot.domain.entities.common.enums import EntryOrderType, PlaybookType, TradeSide
from trading_bot.domain.entities.execution.trade_blueprint import (
    MarketThesis,
    TradeBlueprint,
)
from trading_bot.domain.entities.memory.episodic_case import CohortBaseRateSummary


@pytest.mark.asyncio
async def test_architect_agent_successful_generation() -> None:
    mock_client: AsyncMock = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = NinerouterResponse(
        content=(
            "```json\n"
            "{\n"
            '  "order_type": "LIMIT_PULLBACK_15M",\n'
            '  "entry_price": 60100.0,\n'
            '  "stop_loss_price": 59500.0,\n'
            '  "take_profit_price": 61200.0,\n'
            '  "rationale": "Entry calibrated to 15m VWAP support."\n'
            "}\n"
            "```"
        ),
        model="claude-3-5-haiku",
        finish_reason="stop",
        prompt_tokens=1400,
        completion_tokens=120,
        is_fallback=False,
    )

    adapter: TradeArchitectAdapter = TradeArchitectAdapter(
        client=mock_client,
        tier="fast_tactical",
    )

    thesis: MarketThesis = MarketThesis(
        symbol="BTCUSDT",
        selected_playbook=PlaybookType.TREND_CONTINUATION_BREAKOUT,
        proposed_side=TradeSide.LONG,
        supporting_active_features=["hurst_exp"],
        regime_alignment_score=0.85,
        thesis_summary="Bullish trend continuation",
    )

    cohort_summary: CohortBaseRateSummary = CohortBaseRateSummary(
        has_historical_cohort=True,
        total_neighbors=20,
        weighted_win_rate=0.65,
        stop_hunted_then_tp_rate=0.10,
        true_failure_trap_rate=0.15,
        expected_r_multiple=1.8,
        mae_target_percentile_atr=1.8,
        mfe_median_atr=3.2,
    )

    bp: TradeBlueprint = await adapter.construct_trade_blueprint(
        thesis=thesis,
        current_price=60200.0,
        atr_30m=300.0,
        cohort_summary=cohort_summary,
    )

    assert bp.thesis.symbol == "BTCUSDT"
    assert bp.order_type == EntryOrderType.LIMIT_PULLBACK_15M
    assert bp.entry_price == 60100.0
    assert bp.stop_loss_price == 59500.0
    assert bp.take_profit_price == 61200.0
    assert bp.rationale == "Entry calibrated to 15m VWAP support."

    # Verify cohort stats were passed in prompt
    call_args = mock_client.generate_chat_completion.call_args[1]
    messages = call_args["messages"]
    user_prompt = messages[1]["content"]
    assert "cohort_stats" in user_prompt
    assert "mae_target_percentile_atr" in user_prompt


@pytest.mark.asyncio
async def test_architect_agent_deterministic_fallback_long() -> None:
    mock_client: AsyncMock = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = NinerouterResponse(
        content="",
        model="deterministic_fallback",
        finish_reason="fallback",
        prompt_tokens=0,
        completion_tokens=0,
        is_fallback=True,
    )

    adapter: TradeArchitectAdapter = TradeArchitectAdapter(
        client=mock_client,
        tier="fast_tactical",
    )

    thesis: MarketThesis = MarketThesis(
        symbol="BTCUSDT",
        selected_playbook=PlaybookType.TREND_CONTINUATION_BREAKOUT,
        proposed_side=TradeSide.LONG,
        supporting_active_features=[],
        regime_alignment_score=0.8,
        thesis_summary="Bullish breakout",
    )

    bp: TradeBlueprint = await adapter.construct_trade_blueprint(
        thesis=thesis,
        current_price=60000.0,
        atr_30m=300.0,
        cohort_summary=None,
    )

    assert bp.entry_price == 60000.0
    # Long fallback: SL = current_price - 2.0 * ATR = 59400
    assert bp.stop_loss_price == 59400.0
    # Long fallback: TP = current_price + 3.0 * ATR = 60900
    assert bp.take_profit_price == 60900.0
    assert "deterministic fallback" in bp.rationale.lower()


@pytest.mark.asyncio
async def test_architect_agent_deterministic_fallback_short() -> None:
    mock_client: AsyncMock = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = NinerouterResponse(
        content="",
        model="deterministic_fallback",
        finish_reason="fallback",
        prompt_tokens=0,
        completion_tokens=0,
        is_fallback=True,
    )

    adapter: TradeArchitectAdapter = TradeArchitectAdapter(
        client=mock_client,
        tier="fast_tactical",
    )

    thesis: MarketThesis = MarketThesis(
        symbol="ETHUSDT",
        selected_playbook=PlaybookType.LIQUIDATION_EXHAUSTION_REVERSAL,
        proposed_side=TradeSide.SHORT,
        supporting_active_features=[],
        regime_alignment_score=0.75,
        thesis_summary="Short reversal",
    )

    bp: TradeBlueprint = await adapter.construct_trade_blueprint(
        thesis=thesis,
        current_price=3000.0,
        atr_30m=20.0,
        cohort_summary=None,
    )

    assert bp.entry_price == 3000.0
    # Short fallback: SL = current_price + 2.0 * ATR = 3040.0
    assert bp.stop_loss_price == 3040.0
    # Short fallback: TP = current_price - 3.0 * ATR = 2940.0
    assert bp.take_profit_price == 2940.0
