"""Unit tests for Market Scout Agent Adapter (scout_agent.py).

Enforces Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from trading_bot.adapters.agents.ninerouter_client import (
    NinerouterClient,
    NinerouterResponse,
)
from trading_bot.adapters.agents.scout_agent import MarketScoutAdapter
from trading_bot.domain.entities.common.enums import PlaybookType, TradeSide
from trading_bot.domain.entities.execution.trade_blueprint import MarketThesis
from trading_bot.ports.macro_sentinel_port import MacroBlackoutState


@pytest.mark.asyncio
async def test_scout_agent_successful_llm_parse() -> None:
    mock_client: AsyncMock = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = NinerouterResponse(
        content=(
            "```json\n"
            "{\n"
            '  "symbol": "BTCUSDT",\n'
            '  "selected_playbook": "TREND_CONTINUATION_BREAKOUT",\n'
            '  "proposed_side": "LONG",\n'
            '  "supporting_active_features": ["hurst_exponent", "vpin"],\n'
            '  "regime_alignment_score": 0.90,\n'
            '  "thesis_summary": "Bullish momentum confirmed across micro metrics."\n'
            "}\n"
            "```"
        ),
        model="gemini-2.5-flash",
        finish_reason="stop",
        prompt_tokens=1200,
        completion_tokens=150,
        is_fallback=False,
    )

    adapter: MarketScoutAdapter = MarketScoutAdapter(
        client=mock_client,
        tier="fast_tactical",
    )

    macro_state: MacroBlackoutState = MacroBlackoutState(
        is_blackout=False,
        upcoming_event=None,
        time_to_event_minutes=None,
        reason="CLEAR",
    )

    thesis: MarketThesis = await adapter.analyze_market(
        symbol="BTCUSDT",
        quant_packet={"hurst_exponent": 0.65, "vpin": 0.25},
        playbook=PlaybookType.TREND_CONTINUATION_BREAKOUT,
        macro_state=macro_state,
        btc_thesis=None,
    )

    assert thesis.symbol == "BTCUSDT"
    assert thesis.selected_playbook == PlaybookType.TREND_CONTINUATION_BREAKOUT
    assert thesis.proposed_side == TradeSide.LONG
    assert thesis.regime_alignment_score == 0.90
    assert "hurst_exponent" in thesis.supporting_active_features
    assert mock_client.generate_chat_completion.called


@pytest.mark.asyncio
async def test_scout_agent_deterministic_fallback() -> None:
    mock_client: AsyncMock = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = NinerouterResponse(
        content="",
        model="deterministic_fallback",
        finish_reason="fallback",
        prompt_tokens=0,
        completion_tokens=0,
        is_fallback=True,
    )

    adapter: MarketScoutAdapter = MarketScoutAdapter(
        client=mock_client,
        tier="fast_tactical",
    )

    macro_state: MacroBlackoutState = MacroBlackoutState(
        is_blackout=False,
        upcoming_event=None,
        time_to_event_minutes=None,
        reason="CLEAR",
    )

    thesis: MarketThesis = await adapter.analyze_market(
        symbol="ETHUSDT",
        quant_packet={"hurst_exponent": 0.45},
        playbook=PlaybookType.LIQUIDATION_EXHAUSTION_REVERSAL,
        macro_state=macro_state,
        btc_thesis=None,
    )

    assert thesis.symbol == "ETHUSDT"
    assert thesis.selected_playbook == PlaybookType.LIQUIDATION_EXHAUSTION_REVERSAL
    assert thesis.proposed_side == TradeSide.FLAT
    assert len(thesis.supporting_active_features) == 0
    assert thesis.regime_alignment_score == 0.5
    assert "deterministic fallback" in thesis.thesis_summary.lower()


@pytest.mark.asyncio
async def test_scout_agent_with_btc_thesis_context() -> None:
    mock_client: AsyncMock = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = NinerouterResponse(
        content=(
            "```json\n"
            "{\n"
            '  "symbol": "SOLUSDT",\n'
            '  "selected_playbook": "CROSS_ASSET_LEAD_LAG_CATCHUP",\n'
            '  "proposed_side": "LONG",\n'
            '  "supporting_active_features": ["btc_lead_beta"],\n'
            '  "regime_alignment_score": 0.82,\n'
            '  "thesis_summary": "Catching up to BTC lead."\n'
            "}\n"
            "```"
        ),
        model="gemini-2.5-flash",
        finish_reason="stop",
        prompt_tokens=1300,
        completion_tokens=120,
        is_fallback=False,
    )

    adapter: MarketScoutAdapter = MarketScoutAdapter(
        client=mock_client,
        tier="fast_tactical",
    )

    macro_state: MacroBlackoutState = MacroBlackoutState(
        is_blackout=False,
        upcoming_event=None,
        time_to_event_minutes=None,
        reason="CLEAR",
    )

    btc_thesis: MarketThesis = MarketThesis(
        symbol="BTCUSDT",
        selected_playbook=PlaybookType.TREND_CONTINUATION_BREAKOUT,
        proposed_side=TradeSide.LONG,
        supporting_active_features=["hurst_exp"],
        regime_alignment_score=0.92,
        thesis_summary="BTC breaking out aggressively.",
    )

    thesis: MarketThesis = await adapter.analyze_market(
        symbol="SOLUSDT",
        quant_packet={"btc_lead_beta": 1.45},
        playbook=PlaybookType.CROSS_ASSET_LEAD_LAG_CATCHUP,
        macro_state=macro_state,
        btc_thesis=btc_thesis,
    )

    assert thesis.symbol == "SOLUSDT"
    assert thesis.selected_playbook == PlaybookType.CROSS_ASSET_LEAD_LAG_CATCHUP
    assert thesis.proposed_side == TradeSide.LONG
    # Verify prompt received btc_context
    call_args = mock_client.generate_chat_completion.call_args[1]
    messages = call_args["messages"]
    user_prompt = messages[1]["content"]
    assert "btc_context" in user_prompt
    assert "BTC breaking out aggressively" in user_prompt
