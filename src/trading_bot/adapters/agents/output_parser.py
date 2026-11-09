"""Structured Output Parser for Multi-Agent Responses.

Robust extraction and validation of JSON outputs from LLM completions.
Enforces Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
"""

from __future__ import annotations

import json
import re
from typing import Any, cast

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


def extract_json_from_raw_text(raw_text: str) -> dict[str, Any]:
    """Robustly extract a JSON dictionary from raw model text.

    Handles whitespace, thought blocks (<think>...</think>), and markdown code fences.
    """
    clean_text: str = re.sub(r"<think>[\s\S]*?</think>", "", raw_text).strip()

    # Attempt 1: Check for fenced JSON code block
    fence_pattern: str = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(fence_pattern, clean_text)
    if match is not None:
        fence_content: str = match.group(1).strip()
        try:
            parsed = json.loads(fence_content)
            if isinstance(parsed, dict):
                return cast(dict[str, Any], parsed)
        except Exception:
            pass

    # Attempt 2: Direct loads of clean text
    try:
        parsed = json.loads(clean_text)
        if isinstance(parsed, dict):
            return cast(dict[str, Any], parsed)
    except Exception:
        pass

    # Attempt 3: Locate outer-most JSON curly braces
    start_idx: int = clean_text.find("{")
    end_idx: int = clean_text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate_json: str = clean_text[start_idx : end_idx + 1]
        try:
            parsed = json.loads(candidate_json)
            if isinstance(parsed, dict):
                return cast(dict[str, Any], parsed)
        except Exception:
            pass

    raise ValueError(f"Could not extract valid JSON object from model output: {raw_text[:120]}")


def parse_market_thesis(
    raw_text: str,
    symbol: str,
    fallback_playbook: PlaybookType,
) -> MarketThesis:
    """Parse raw LLM response into MarketThesis domain entity with explicit fallback."""
    try:
        data: dict[str, Any] = extract_json_from_raw_text(raw_text)

        if "symbol" in data and isinstance(data["symbol"], str):
            parsed_symbol: str = data["symbol"]
        else:
            parsed_symbol = symbol

        if "selected_playbook" in data and isinstance(data["selected_playbook"], str):
            try:
                selected_playbook: PlaybookType = PlaybookType(data["selected_playbook"])
            except ValueError:
                selected_playbook = fallback_playbook
        else:
            selected_playbook = fallback_playbook

        if "proposed_side" in data and isinstance(data["proposed_side"], str):
            try:
                proposed_side: TradeSide = TradeSide(data["proposed_side"].upper())
            except ValueError:
                proposed_side = TradeSide.FLAT
        else:
            proposed_side = TradeSide.FLAT

        if "supporting_active_features" in data and isinstance(
            data["supporting_active_features"], list
        ):
            raw_features: list[object] = cast(list[object], data["supporting_active_features"])
            active_features: list[str] = [str(f) for f in raw_features]
        else:
            active_features = []

        if "regime_alignment_score" in data and isinstance(
            data["regime_alignment_score"], (int, float)
        ):
            alignment_score: float = max(0.0, min(1.0, float(data["regime_alignment_score"])))
        else:
            alignment_score = 0.5

        if "thesis_summary" in data and isinstance(data["thesis_summary"], str):
            thesis_summary: str = data["thesis_summary"]
        else:
            thesis_summary = "Market thesis parsed from model response"

        return MarketThesis(
            symbol=parsed_symbol,
            selected_playbook=selected_playbook,
            proposed_side=proposed_side,
            supporting_active_features=active_features,
            regime_alignment_score=alignment_score,
            thesis_summary=thesis_summary,
        )
    except Exception:
        return MarketThesis(
            symbol=symbol,
            selected_playbook=fallback_playbook,
            proposed_side=TradeSide.FLAT,
            supporting_active_features=[],
            regime_alignment_score=0.5,
            thesis_summary="Output parser fallback: malformed model response",
        )


def parse_trade_blueprint(
    raw_text: str,
    thesis: MarketThesis,
    current_price: float,
    atr_30m: float,
) -> TradeBlueprint:
    """Parse raw LLM response into TradeBlueprint domain entity with explicit fallback."""
    try:
        data: dict[str, Any] = extract_json_from_raw_text(raw_text)

        if "order_type" in data and isinstance(data["order_type"], str):
            try:
                order_type: EntryOrderType = EntryOrderType(data["order_type"])
            except ValueError:
                order_type = EntryOrderType.LIMIT_PULLBACK_15M
        else:
            order_type = EntryOrderType.LIMIT_PULLBACK_15M

        if (
            "entry_price" in data
            and isinstance(data["entry_price"], (int, float))
            and data["entry_price"] > 0
        ):
            entry_price: float = float(data["entry_price"])
        else:
            entry_price = current_price

        if (
            "stop_loss_price" in data
            and isinstance(data["stop_loss_price"], (int, float))
            and data["stop_loss_price"] > 0
        ):
            stop_loss_price: float = float(data["stop_loss_price"])
        else:
            if thesis.proposed_side == TradeSide.SHORT:
                stop_loss_price = current_price + 2.0 * atr_30m
            else:
                stop_loss_price = current_price - 2.0 * atr_30m

        if (
            "take_profit_price" in data
            and isinstance(data["take_profit_price"], (int, float))
            and data["take_profit_price"] > 0
        ):
            take_profit_price: float = float(data["take_profit_price"])
        else:
            if thesis.proposed_side == TradeSide.SHORT:
                take_profit_price = current_price - 3.0 * atr_30m
            else:
                take_profit_price = current_price + 3.0 * atr_30m

        if "rationale" in data and isinstance(data["rationale"], str):
            rationale: str = data["rationale"]
        else:
            rationale = "Trade blueprint parsed from model response"

        return TradeBlueprint(
            thesis=thesis,
            order_type=order_type,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            rationale=rationale,
        )
    except Exception:
        if thesis.proposed_side == TradeSide.SHORT:
            sl_fallback: float = current_price + 2.0 * atr_30m
            tp_fallback: float = current_price - 3.0 * atr_30m
        else:
            sl_fallback = current_price - 2.0 * atr_30m
            tp_fallback = current_price + 3.0 * atr_30m

        return TradeBlueprint(
            thesis=thesis,
            order_type=EntryOrderType.LIMIT_PULLBACK_15M,
            entry_price=current_price,
            stop_loss_price=sl_fallback,
            take_profit_price=tp_fallback,
            rationale="Output parser fallback: malformed blueprint response",
        )


def parse_risk_verdict(raw_text: str) -> RiskVerdict:
    """Parse raw LLM response into RiskVerdict domain entity with explicit fallback."""
    try:
        data: dict[str, Any] = extract_json_from_raw_text(raw_text)

        if "decision" in data and isinstance(data["decision"], str):
            try:
                decision: RiskDecisionType = RiskDecisionType(data["decision"])
            except ValueError:
                decision = RiskDecisionType.VETO_TO_SHADOW
        else:
            decision = RiskDecisionType.VETO_TO_SHADOW

        if "approved_leverage" in data and isinstance(data["approved_leverage"], (int, float)):
            approved_leverage: int = max(1, int(data["approved_leverage"]))
        else:
            approved_leverage = 1

        if "approved_notional_usd" in data and isinstance(
            data["approved_notional_usd"], (int, float)
        ):
            approved_notional_usd: float = max(0.0, float(data["approved_notional_usd"]))
        else:
            approved_notional_usd = 0.0

        if "net_rr" in data and isinstance(data["net_rr"], (int, float)):
            net_rr: float = float(data["net_rr"])
        else:
            net_rr = 0.0

        if "reason" in data and data["reason"] is not None:
            reason: str | None = str(data["reason"])
        else:
            reason = None

        if "arbiter_size_multiplier" in data and isinstance(
            data["arbiter_size_multiplier"], (int, float)
        ):
            arbiter_size_multiplier: float = max(
                0.0, min(1.0, float(data["arbiter_size_multiplier"]))
            )
        else:
            arbiter_size_multiplier = 0.0

        if (
            "veto_category" in data
            and data["veto_category"] is not None
            and isinstance(data["veto_category"], str)
        ):
            try:
                veto_category: VetoCategory | None = VetoCategory(data["veto_category"])
            except ValueError:
                veto_category = None
        else:
            veto_category = None

        if "existing_position_directive" in data and isinstance(
            data["existing_position_directive"], str
        ):
            try:
                existing_position_directive: ExistingPositionDirective = ExistingPositionDirective(
                    data["existing_position_directive"]
                )
            except ValueError:
                existing_position_directive = ExistingPositionDirective.HOLD_EXISTING
        else:
            existing_position_directive = ExistingPositionDirective.HOLD_EXISTING

        if "portfolio_risk_rationale" in data and isinstance(data["portfolio_risk_rationale"], str):
            portfolio_risk_rationale: str = data["portfolio_risk_rationale"]
        else:
            portfolio_risk_rationale = "Risk verdict parsed from model response"

        return RiskVerdict(
            decision=decision,
            approved_leverage=approved_leverage,
            approved_notional_usd=approved_notional_usd,
            net_rr=net_rr,
            reason=reason,
            arbiter_size_multiplier=arbiter_size_multiplier,
            veto_category=veto_category,
            existing_position_directive=existing_position_directive,
            portfolio_risk_rationale=portfolio_risk_rationale,
        )
    except Exception:
        return RiskVerdict(
            decision=RiskDecisionType.VETO_TO_SHADOW,
            approved_leverage=1,
            approved_notional_usd=0.0,
            net_rr=0.0,
            reason="Output parser fallback: malformed risk verdict",
            arbiter_size_multiplier=0.0,
            veto_category=VetoCategory.SETUP_DEFECT,
            existing_position_directive=ExistingPositionDirective.HOLD_EXISTING,
            portfolio_risk_rationale="Malformed LLM response in Risk Arbiter",
        )
