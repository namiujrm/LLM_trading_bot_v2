"""Unit tests for Structured Output Parser (output_parser.py).

Enforces Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
"""

from __future__ import annotations

import pytest

from trading_bot.adapters.agents.output_parser import (
    extract_json_from_raw_text,
    parse_market_thesis,
    parse_risk_verdict,
    parse_trade_blueprint,
)
from trading_bot.domain.entities.common.enums import (
    EntryOrderType,
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


def test_extract_json_with_code_fences() -> None:
    raw: str = (
        "Here is the thesis:\n"
        "```json\n"
        '{\n  "symbol": "BTCUSDT",\n  "decision": "APPROVE"\n}\n'
        "```\n"
        "End of message."
    )
    result = extract_json_from_raw_text(raw)
    assert result["symbol"] == "BTCUSDT"
    assert result["decision"] == "APPROVE"


def test_extract_json_without_fences() -> None:
    raw: str = 'Prefix text {"symbol": "ETHUSDT", "score": 0.85} suffix text'
    result = extract_json_from_raw_text(raw)
    assert result["symbol"] == "ETHUSDT"
    assert result["score"] == 0.85


def test_extract_json_with_think_tags() -> None:
    raw: str = (
        "<think>Let me evaluate this trade carefully...</think>\n"
        "```\n"
        '{"symbol": "SOLUSDT", "side": "LONG"}\n'
        "```"
    )
    result = extract_json_from_raw_text(raw)
    assert result["symbol"] == "SOLUSDT"
    assert result["side"] == "LONG"


def test_extract_json_malformed_raises_error() -> None:
    raw: str = "This response contains no JSON at all."
    with pytest.raises(ValueError, match="Could not extract valid JSON"):
        extract_json_from_raw_text(raw)


def test_parse_market_thesis_valid() -> None:
    raw: str = (
        "```json\n"
        "{\n"
        '  "symbol": "BTCUSDT",\n'
        '  "selected_playbook": "TREND_CONTINUATION_BREAKOUT",\n'
        '  "proposed_side": "LONG",\n'
        '  "supporting_active_features": ["hurst_exp", "vpin"],\n'
        '  "regime_alignment_score": 0.88,\n'
        '  "thesis_summary": "Strong trend continuation breakout."\n'
        "}\n"
        "```"
    )
    thesis: MarketThesis = parse_market_thesis(
        raw_text=raw,
        symbol="BTCUSDT",
        fallback_playbook=PlaybookType.NO_TRADE_REGIME,
    )
    assert thesis.symbol == "BTCUSDT"
    assert thesis.selected_playbook == PlaybookType.TREND_CONTINUATION_BREAKOUT
    assert thesis.proposed_side == TradeSide.LONG
    assert thesis.supporting_active_features == ["hurst_exp", "vpin"]
    assert thesis.regime_alignment_score == 0.88
    assert thesis.thesis_summary == "Strong trend continuation breakout."


def test_parse_market_thesis_malformed_fallback() -> None:
    raw: str = "Total gibberish that cannot be parsed as JSON"
    thesis: MarketThesis = parse_market_thesis(
        raw_text=raw,
        symbol="SOLUSDT",
        fallback_playbook=PlaybookType.GENERAL_ALPHA_MOMENTUM,
    )
    assert thesis.symbol == "SOLUSDT"
    assert thesis.selected_playbook == PlaybookType.GENERAL_ALPHA_MOMENTUM
    assert thesis.proposed_side == TradeSide.FLAT
    assert len(thesis.supporting_active_features) == 0
    assert thesis.regime_alignment_score == 0.5
    assert "fallback" in thesis.thesis_summary.lower()


def test_parse_market_thesis_invalid_enum_fallback() -> None:
    raw: str = (
        '{"symbol": "BNBUSDT", "selected_playbook": "UNKNOWN_PLAYBOOK", "proposed_side": "UP"}'
    )
    thesis: MarketThesis = parse_market_thesis(
        raw_text=raw,
        symbol="BNBUSDT",
        fallback_playbook=PlaybookType.NO_TRADE_REGIME,
    )
    assert thesis.symbol == "BNBUSDT"
    assert thesis.selected_playbook == PlaybookType.NO_TRADE_REGIME
    assert thesis.proposed_side == TradeSide.FLAT


def test_parse_trade_blueprint_valid() -> None:
    thesis = MarketThesis(
        symbol="BTCUSDT",
        selected_playbook=PlaybookType.TREND_CONTINUATION_BREAKOUT,
        proposed_side=TradeSide.LONG,
        supporting_active_features=["hurst_exp"],
        regime_alignment_score=0.8,
        thesis_summary="Bullish thesis",
    )
    raw: str = (
        "```json\n"
        "{\n"
        '  "order_type": "LIMIT_PULLBACK_15M",\n'
        '  "entry_price": 60000.0,\n'
        '  "stop_loss_price": 59400.0,\n'
        '  "take_profit_price": 61500.0,\n'
        '  "rationale": "Pullback entry to 15m VWAP."\n'
        "}\n"
        "```"
    )
    bp: TradeBlueprint = parse_trade_blueprint(
        raw_text=raw,
        thesis=thesis,
        current_price=60200.0,
        atr_30m=300.0,
    )
    assert bp.thesis.symbol == "BTCUSDT"
    assert bp.order_type == EntryOrderType.LIMIT_PULLBACK_15M
    assert bp.entry_price == 60000.0
    assert bp.stop_loss_price == 59400.0
    assert bp.take_profit_price == 61500.0
    assert bp.rationale == "Pullback entry to 15m VWAP."


def test_parse_trade_blueprint_malformed_fallback_long() -> None:
    thesis = MarketThesis(
        symbol="BTCUSDT",
        selected_playbook=PlaybookType.TREND_CONTINUATION_BREAKOUT,
        proposed_side=TradeSide.LONG,
        supporting_active_features=[],
        regime_alignment_score=0.8,
        thesis_summary="Bullish thesis",
    )
    bp: TradeBlueprint = parse_trade_blueprint(
        raw_text="Not a json",
        thesis=thesis,
        current_price=60000.0,
        atr_30m=300.0,
    )
    assert bp.entry_price == 60000.0
    # SL = 60000 - 2.0 * 300 = 59400
    assert bp.stop_loss_price == 59400.0
    # TP = 60000 + 3.0 * 300 = 60900
    assert bp.take_profit_price == 60900.0
    assert "fallback" in bp.rationale.lower()


def test_parse_trade_blueprint_malformed_fallback_short() -> None:
    thesis = MarketThesis(
        symbol="BTCUSDT",
        selected_playbook=PlaybookType.LIQUIDATION_EXHAUSTION_REVERSAL,
        proposed_side=TradeSide.SHORT,
        supporting_active_features=[],
        regime_alignment_score=0.8,
        thesis_summary="Bearish reversal",
    )
    bp: TradeBlueprint = parse_trade_blueprint(
        raw_text="Not a json",
        thesis=thesis,
        current_price=60000.0,
        atr_30m=300.0,
    )
    assert bp.entry_price == 60000.0
    # SHORT SL = 60000 + 2.0 * 300 = 60600
    assert bp.stop_loss_price == 60600.0
    # SHORT TP = 60000 - 3.0 * 300 = 59100
    assert bp.take_profit_price == 59100.0


def test_parse_risk_verdict_valid() -> None:
    raw: str = (
        "```json\n"
        "{\n"
        '  "decision": "APPROVE_FULL",\n'
        '  "approved_leverage": 5,\n'
        '  "approved_notional_usd": 5000.0,\n'
        '  "net_rr": 2.1,\n'
        '  "reason": null,\n'
        '  "arbiter_size_multiplier": 1.0,\n'
        '  "veto_category": null,\n'
        '  "existing_position_directive": "HOLD_EXISTING",\n'
        '  "portfolio_risk_rationale": "Portfolio margin healthy."\n'
        "}\n"
        "```"
    )
    verdict: RiskVerdict = parse_risk_verdict(raw_text=raw)
    assert verdict.decision == RiskDecisionType.APPROVE_FULL
    assert verdict.approved_leverage == 5
    assert verdict.approved_notional_usd == 5000.0
    assert verdict.net_rr == 2.1
    assert verdict.reason is None
    assert verdict.arbiter_size_multiplier == 1.0
    assert verdict.veto_category is None
    assert verdict.existing_position_directive == ExistingPositionDirective.HOLD_EXISTING
    assert verdict.portfolio_risk_rationale == "Portfolio margin healthy."


def test_parse_risk_verdict_malformed_fallback() -> None:
    verdict: RiskVerdict = parse_risk_verdict(raw_text="Broken output")
    assert verdict.decision == RiskDecisionType.VETO_TO_SHADOW
    assert verdict.approved_leverage == 1
    assert verdict.approved_notional_usd == 0.0
    assert verdict.net_rr == 0.0
    assert verdict.arbiter_size_multiplier == 0.0
    assert verdict.veto_category == VetoCategory.SETUP_DEFECT
    assert verdict.existing_position_directive == ExistingPositionDirective.HOLD_EXISTING
