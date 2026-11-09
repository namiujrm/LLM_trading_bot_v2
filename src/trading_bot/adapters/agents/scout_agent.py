"""Market Scout Agent Adapter implementing MarketScoutPort.

Evaluates quantitative metrics, regime, macro blackout, and cross-asset BTC context.
Enforces Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
"""

from __future__ import annotations

import json
from typing import Any

from trading_bot.adapters.agents.ninerouter_client import (
    NinerouterClient,
    NinerouterResponse,
)
from trading_bot.adapters.agents.output_parser import parse_market_thesis
from trading_bot.domain.entities.common.enums import PlaybookType, TradeSide
from trading_bot.domain.entities.execution.trade_blueprint import MarketThesis
from trading_bot.ports.agent_ports import MarketScoutPort
from trading_bot.ports.macro_sentinel_port import MacroBlackoutState


class MarketScoutAdapter(MarketScoutPort):
    """Concrete adapter for Market Scout agent communicating via 9Router."""

    def __init__(self, client: NinerouterClient, tier: str) -> None:
        self._client: NinerouterClient = client
        self._tier: str = tier

    def _build_scout_prompt(
        self,
        symbol: str,
        quant_packet: dict[str, Any],
        playbook: PlaybookType,
        macro_state: MacroBlackoutState,
        btc_thesis: MarketThesis | None,
    ) -> list[dict[str, str]]:
        """Construct the prompt messages for Market Scout."""
        system_prompt: str = (
            "You are the Market Scout quantitative agent in a high-frequency derivative trading system.\n"
            "Your role is to analyze quantitative signals, TreeSHAP drivers, microstructure metrics, "
            "and macroeconomic blackout states within the pre-classified playbook framework.\n"
            "Produce ONLY valid JSON conforming to the following schema:\n"
            "{\n"
            '  "symbol": string,\n'
            '  "selected_playbook": "TREND_CONTINUATION_BREAKOUT" | "LIQUIDATION_EXHAUSTION_REVERSAL" | '
            '"CROSS_ASSET_LEAD_LAG_CATCHUP" | "GENERAL_ALPHA_MOMENTUM" | "NO_TRADE_REGIME",\n'
            '  "proposed_side": "LONG" | "SHORT" | "FLAT",\n'
            '  "supporting_active_features": list[string],\n'
            '  "regime_alignment_score": float (between 0.0 and 1.0),\n'
            '  "thesis_summary": string\n'
            "}\n"
            "If macro blackout is active or market signals are noise, set proposed_side to 'FLAT' "
            "and selected_playbook to 'NO_TRADE_REGIME'."
        )

        user_content_dict: dict[str, Any] = {
            "symbol": symbol,
            "playbook_candidate": playbook.value,
            "macro_blackout": {
                "is_blackout": macro_state.is_blackout,
                "upcoming_event": macro_state.upcoming_event,
                "time_to_event_minutes": macro_state.time_to_event_minutes,
                "reason": macro_state.reason,
            },
            "quant_packet": quant_packet,
        }

        if btc_thesis is not None:
            user_content_dict["btc_context"] = {
                "btc_symbol": btc_thesis.symbol,
                "btc_proposed_side": btc_thesis.proposed_side.value,
                "btc_playbook": btc_thesis.selected_playbook.value,
                "btc_regime_alignment": btc_thesis.regime_alignment_score,
                "btc_summary": btc_thesis.thesis_summary,
            }

        user_prompt: str = json.dumps(user_content_dict, indent=2)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def analyze_market(
        self,
        symbol: str,
        quant_packet: dict[str, Any],
        playbook: PlaybookType,
        macro_state: MacroBlackoutState,
        btc_thesis: MarketThesis | None,
    ) -> MarketThesis:
        """Analyze market using 9Router completion or deterministic fallback."""
        messages: list[dict[str, str]] = self._build_scout_prompt(
            symbol=symbol,
            quant_packet=quant_packet,
            playbook=playbook,
            macro_state=macro_state,
            btc_thesis=btc_thesis,
        )

        response: NinerouterResponse = await self._client.generate_chat_completion(
            tier=self._tier,
            messages=messages,
            estimated_tokens=1200,
            temperature=0.1,
            max_tokens=1000,
        )

        if response.is_fallback:
            fallback_side: TradeSide = TradeSide.FLAT
            if "proposed_side" in quant_packet and isinstance(
                quant_packet["proposed_side"], TradeSide
            ):
                fallback_side = quant_packet["proposed_side"]
            elif (
                "proposed_side" in quant_packet
                and str(quant_packet["proposed_side"]) in TradeSide._value2member_map_
            ):
                fallback_side = TradeSide(str(quant_packet["proposed_side"]))
            elif "p_long_cal" in quant_packet and "p_short_cal" in quant_packet:
                p_long: float = float(quant_packet["p_long_cal"])
                p_short: float = float(quant_packet["p_short_cal"])
                if p_long > p_short and p_long >= 0.50:
                    fallback_side = TradeSide.LONG
                elif p_short > p_long and p_short >= 0.50:
                    fallback_side = TradeSide.SHORT

            return MarketThesis(
                symbol=symbol,
                selected_playbook=playbook,
                proposed_side=fallback_side,
                supporting_active_features=[],
                regime_alignment_score=0.5,
                thesis_summary="Deterministic fallback: 9Router offline",
            )

        return parse_market_thesis(
            raw_text=response.content,
            symbol=symbol,
            fallback_playbook=playbook,
        )
