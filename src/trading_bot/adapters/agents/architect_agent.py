"""Trade Architect Agent Adapter implementing TradeArchitectPort.

Constructs limit/market execution blueprints with ATR-calibrated barriers.
Enforces Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
"""

from __future__ import annotations

import json
from typing import Any

from trading_bot.adapters.agents.ninerouter_client import (
    NinerouterClient,
    NinerouterResponse,
)
from trading_bot.adapters.agents.output_parser import parse_trade_blueprint
from trading_bot.domain.entities.common.enums import EntryOrderType, TradeSide
from trading_bot.domain.entities.execution.trade_blueprint import (
    MarketThesis,
    TradeBlueprint,
)
from trading_bot.domain.entities.memory.episodic_case import CohortBaseRateSummary
from trading_bot.ports.agent_ports import TradeArchitectPort


class TradeArchitectAdapter(TradeArchitectPort):
    """Concrete adapter for Trade Architect agent communicating via 9Router."""

    def __init__(self, client: NinerouterClient, tier: str) -> None:
        self._client: NinerouterClient = client
        self._tier: str = tier

    def _build_architect_prompt(
        self,
        thesis: MarketThesis,
        current_price: float,
        atr_30m: float,
        cohort_summary: CohortBaseRateSummary | None,
    ) -> list[dict[str, str]]:
        """Construct the prompt messages for Trade Architect."""
        system_prompt: str = (
            "You are the Trade Architect tactical execution engineer in a crypto quantitative fund.\n"
            "Your objective is to determine entry order type, entry price, stop-loss price, "
            "and take-profit price based on the Market Thesis, current 30m ATR, and historical MAE/MFE distribution.\n"
            "Produce ONLY valid JSON conforming to the following schema:\n"
            "{\n"
            '  "order_type": "LIMIT_PULLBACK_15M" | "MARKET",\n'
            '  "entry_price": float,\n'
            '  "stop_loss_price": float,\n'
            '  "take_profit_price": float,\n'
            '  "rationale": string\n'
            "}\n"
            "For LONG trades: stop_loss_price must be strictly below entry_price, take_profit_price strictly above.\n"
            "For SHORT trades: stop_loss_price must be strictly above entry_price, take_profit_price strictly below.\n"
            "Ensure net risk-to-reward ratio (reward / risk) is at least 1.5."
        )

        user_content_dict: dict[str, Any] = {
            "thesis": {
                "symbol": thesis.symbol,
                "selected_playbook": thesis.selected_playbook.value,
                "proposed_side": thesis.proposed_side.value,
                "regime_alignment_score": thesis.regime_alignment_score,
                "thesis_summary": thesis.thesis_summary,
            },
            "market_context": {
                "current_price": current_price,
                "atr_30m": atr_30m,
            },
        }

        if cohort_summary is not None:
            user_content_dict["cohort_stats"] = {
                "has_historical_cohort": cohort_summary.has_historical_cohort,
                "total_neighbors": cohort_summary.total_neighbors,
                "weighted_win_rate": cohort_summary.weighted_win_rate,
                "mae_target_percentile_atr": cohort_summary.mae_target_percentile_atr,
                "mfe_median_atr": cohort_summary.mfe_median_atr,
            }

        user_prompt: str = json.dumps(user_content_dict, indent=2)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def construct_trade_blueprint(
        self,
        thesis: MarketThesis,
        current_price: float,
        atr_30m: float,
        cohort_summary: CohortBaseRateSummary | None,
    ) -> TradeBlueprint:
        """Construct trade blueprint using 9Router completion or deterministic fallback."""
        messages: list[dict[str, str]] = self._build_architect_prompt(
            thesis=thesis,
            current_price=current_price,
            atr_30m=atr_30m,
            cohort_summary=cohort_summary,
        )

        response: NinerouterResponse = await self._client.generate_chat_completion(
            tier=self._tier,
            messages=messages,
            estimated_tokens=1400,
            temperature=0.1,
            max_tokens=1000,
        )

        if response.is_fallback:
            if thesis.proposed_side == TradeSide.SHORT:
                sl_price: float = current_price + 2.0 * atr_30m
                tp_price: float = current_price - 3.0 * atr_30m
            else:
                sl_price = current_price - 2.0 * atr_30m
                tp_price = current_price + 3.0 * atr_30m

            return TradeBlueprint(
                thesis=thesis,
                order_type=EntryOrderType.LIMIT_PULLBACK_15M,
                entry_price=current_price,
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
                rationale="Deterministic fallback: 9Router offline",
            )

        return parse_trade_blueprint(
            raw_text=response.content,
            thesis=thesis,
            current_price=current_price,
            atr_30m=atr_30m,
        )
