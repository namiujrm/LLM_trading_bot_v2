"""Risk Arbiter Agent Adapter implementing RiskArbiterPort.

Dual-Shield risk management: semantic LLM prosecutor paired with strict deterministic bounding.
Enforces Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from trading_bot.adapters.agents.ninerouter_client import (
    NinerouterClient,
    NinerouterResponse,
)
from trading_bot.adapters.agents.output_parser import parse_risk_verdict
from trading_bot.domain.entities.common.enums import (
    EntryOrderType,
    ExistingPositionDirective,
    RiskDecisionType,
    TradeSide,
    VetoCategory,
)
from trading_bot.domain.entities.execution.trade_blueprint import (
    RiskVerdict,
    TradeBlueprint,
)
from trading_bot.domain.entities.memory.episodic_case import StratifiedMemoryPack
from trading_bot.domain.entities.risk.account_state import AccountPortfolioContext
from trading_bot.domain.services.risk.risk_calculator import (
    DeterministicRiskCalculator,
    SizingResult,
)
from trading_bot.infrastructure.logging.bot_logger import StructuredBotLogger
from trading_bot.ports.agent_ports import RiskArbiterPort

_std_logger: logging.Logger = logging.getLogger(__name__)


class RiskArbiterAdapter(RiskArbiterPort):
    """Concrete adapter for Risk Arbiter agent communicating via 9Router with Deterministic Fallback."""

    def __init__(
        self,
        client: NinerouterClient,
        tier: str,
        deterministic_calculator: DeterministicRiskCalculator,
        logger: StructuredBotLogger | None,
    ) -> None:
        self._client: NinerouterClient = client
        self._tier: str = tier
        self._calc: DeterministicRiskCalculator = deterministic_calculator
        self._logger: StructuredBotLogger | None = logger

    def _build_arbiter_prompt(
        self,
        blueprint: TradeBlueprint,
        account_context: AccountPortfolioContext,
        memory_pack: StratifiedMemoryPack | None,
    ) -> list[dict[str, str]]:
        """Construct the prompt messages for Risk Arbiter."""
        system_prompt: str = (
            "You are the Risk Arbiter and Portfolio-Aware Semantic Risk Prosecutor.\n"
            "Evaluate portfolio health, margin utilization, beta exposure, drawdown headroom, "
            "and historical failure memory traps before granting trade authorization.\n"
            "Apply the 4 Portfolio-Aware Rules:\n"
            "1. Effective Open Heat vs Breakeven: do not count positions whose SL is at breakeven.\n"
            "2. Correlated Beta Stacking: if >=2 correlated positions are in loss, reduce size or veto.\n"
            "3. Drawdown Budget: if remaining daily drawdown budget <= 1.2% or consecutive losses >= 2, "
            "reduce size or veto.\n"
            "4. Same-symbol conflict: forbid duplicate symbol entries if existing position is not at breakeven.\n"
            "Produce ONLY valid JSON conforming to the following schema:\n"
            "{\n"
            '  "decision": "APPROVE_FULL" | "APPROVE_REDUCED_SIZE" | "VETO_TO_SHADOW",\n'
            '  "approved_leverage": integer,\n'
            '  "approved_notional_usd": float,\n'
            '  "net_rr": float,\n'
            '  "reason": string | null,\n'
            '  "arbiter_size_multiplier": float (1.0, 0.5, or 0.0),\n'
            '  "veto_category": "SETUP_DEFECT" | "PORTFOLIO_CAPACITY_VETO" | null,\n'
            '  "existing_position_directive": "HOLD_EXISTING" | "TIGHTEN_EXISTING_SL_TO_BE" | "FORBID_DUPLICATE_SYMBOL",\n'
            '  "portfolio_risk_rationale": string\n'
            "}"
        )

        user_content_dict: dict[str, Any] = {
            "blueprint": {
                "symbol": blueprint.thesis.symbol,
                "playbook": blueprint.thesis.selected_playbook.value,
                "proposed_side": blueprint.thesis.proposed_side.value,
                "order_type": blueprint.order_type.value,
                "entry_price": blueprint.entry_price,
                "stop_loss_price": blueprint.stop_loss_price,
                "take_profit_price": blueprint.take_profit_price,
                "rationale": blueprint.rationale,
            },
            "portfolio_context": {
                "wallet_balance_usd": account_context.wallet_balance_usd,
                "equity_nav_usd": account_context.equity_nav_usd,
                "available_margin_usd": account_context.available_margin_usd,
                "used_margin_usd": account_context.used_margin_usd,
                "margin_utilization_pct": account_context.margin_utilization_pct,
                "effective_open_heat_pct": account_context.effective_open_heat_pct,
                "remaining_daily_dd_budget_pct": account_context.remaining_daily_dd_budget_pct,
                "consecutive_losses_count": account_context.consecutive_losses_count,
                "active_position_count": len(account_context.active_positions),
                "same_direction_active_count": account_context.same_direction_active_count,
            },
        }

        if memory_pack is not None:
            user_content_dict["memory_pack"] = {
                "has_historical_cohort": memory_pack.cohort_summary.has_historical_cohort,
                "weighted_win_rate": memory_pack.cohort_summary.weighted_win_rate,
                "true_failure_trap_rate": memory_pack.cohort_summary.true_failure_trap_rate,
                "stop_hunts_count": len(memory_pack.bucket3_stop_hunts),
                "failure_traps_count": len(memory_pack.bucket2_failure_traps),
            }

        user_prompt: str = json.dumps(user_content_dict, indent=2)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _derive_calibrated_win_prob(
        self,
        blueprint: TradeBlueprint,
        memory_pack: StratifiedMemoryPack | None,
    ) -> float:
        """Derive estimated win probability from episodic memory or regime alignment score."""
        if (
            memory_pack is not None
            and memory_pack.cohort_summary.has_historical_cohort
            and memory_pack.cohort_summary.weighted_win_rate is not None
        ):
            return memory_pack.cohort_summary.weighted_win_rate

        return 0.52 + 0.10 * blueprint.thesis.regime_alignment_score

    async def evaluate_risk(
        self,
        blueprint: TradeBlueprint,
        account_context: AccountPortfolioContext,
        memory_pack: StratifiedMemoryPack | None,
    ) -> RiskVerdict:
        """Evaluate risk with dual-shield protection (LLM Semantic + Deterministic Invariant)."""
        if blueprint.thesis.proposed_side == TradeSide.FLAT:
            return RiskVerdict(
                decision=RiskDecisionType.VETO_TO_SHADOW,
                approved_leverage=1,
                approved_notional_usd=0.0,
                net_rr=0.0,
                reason="Thesis proposed side is FLAT",
                arbiter_size_multiplier=0.0,
                veto_category=VetoCategory.SETUP_DEFECT,
                existing_position_directive=ExistingPositionDirective.HOLD_EXISTING,
                portfolio_risk_rationale="Scout thesis is FLAT, no trade executed.",
            )

        win_prob: float = self._derive_calibrated_win_prob(
            blueprint=blueprint,
            memory_pack=memory_pack,
        )

        messages: list[dict[str, str]] = self._build_arbiter_prompt(
            blueprint=blueprint,
            account_context=account_context,
            memory_pack=memory_pack,
        )

        response: NinerouterResponse = await self._client.generate_chat_completion(
            tier=self._tier,
            messages=messages,
            estimated_tokens=2200,
            temperature=0.1,
            max_tokens=1200,
        )

        if response.is_fallback:
            # Deterministic Rule-Only Fallback
            det_result: SizingResult = self._calc.evaluate_with_portfolio_context(
                context=account_context,
                arbiter_size_multiplier=1.0,
                entry_price=blueprint.entry_price,
                stop_loss_price=blueprint.stop_loss_price,
                take_profit_price=blueprint.take_profit_price,
                is_limit_entry=(blueprint.order_type == EntryOrderType.LIMIT_PULLBACK_15M),
                kyle_slippage_pct=0.0003,
                calibrated_win_prob=win_prob,
                venn_abers_uncertainty_width=0.05,
                ood_shrinkage_factor=1.0,
                current_yz_vol=self._calc.cfg.target_30m_yz_vol,
                pre_event_blackout=False,
            )

            critical_msg: str = (
                "9Router offline/exhausted. Deterministic Rule-Only Fallback applied"
            )
            log_payload: dict[str, Any] = {
                "symbol": blueprint.thesis.symbol,
                "approved": det_result.approved,
                "rejection_reason": det_result.rejection_reason,
                "position_notional_usd": det_result.position_notional_usd,
                "recommended_leverage": det_result.recommended_leverage,
            }
            if self._logger is not None:
                self._logger.critical(critical_msg, log_payload)
            else:
                _std_logger.critical("%s: %s", critical_msg, str(log_payload))

            if det_result.approved:
                return RiskVerdict(
                    decision=RiskDecisionType.APPROVE_FULL,
                    approved_leverage=det_result.recommended_leverage,
                    approved_notional_usd=det_result.position_notional_usd,
                    net_rr=det_result.net_rr,
                    reason=None,
                    arbiter_size_multiplier=1.0,
                    veto_category=None,
                    existing_position_directive=ExistingPositionDirective.HOLD_EXISTING,
                    portfolio_risk_rationale=(
                        "Deterministic Rule-Only Fallback approved based on mathematical limits."
                    ),
                )

            rej_str: str = str(det_result.rejection_reason)
            if "MARGIN" in rej_str or "HEAT" in rej_str or "DRAWDOWN" in rej_str:
                veto_cat: VetoCategory | None = VetoCategory.PORTFOLIO_CAPACITY_VETO
            else:
                veto_cat = VetoCategory.SETUP_DEFECT

            return RiskVerdict(
                decision=RiskDecisionType.VETO_TO_SHADOW,
                approved_leverage=1,
                approved_notional_usd=0.0,
                net_rr=det_result.net_rr,
                reason=det_result.rejection_reason,
                arbiter_size_multiplier=0.0,
                veto_category=veto_cat,
                existing_position_directive=ExistingPositionDirective.HOLD_EXISTING,
                portfolio_risk_rationale=(
                    f"Deterministic Rule-Only Fallback rejected: {det_result.rejection_reason}"
                ),
            )

        # 9Router active: parse verdict and apply deterministic mathematical bounds
        llm_verdict: RiskVerdict = parse_risk_verdict(response.content)

        det_result = self._calc.evaluate_with_portfolio_context(
            context=account_context,
            arbiter_size_multiplier=llm_verdict.arbiter_size_multiplier,
            entry_price=blueprint.entry_price,
            stop_loss_price=blueprint.stop_loss_price,
            take_profit_price=blueprint.take_profit_price,
            is_limit_entry=(blueprint.order_type == EntryOrderType.LIMIT_PULLBACK_15M),
            kyle_slippage_pct=0.0003,
            calibrated_win_prob=win_prob,
            venn_abers_uncertainty_width=0.05,
            ood_shrinkage_factor=1.0,
            current_yz_vol=self._calc.cfg.target_30m_yz_vol,
            pre_event_blackout=False,
        )

        if not det_result.approved:
            rej_str = str(det_result.rejection_reason)
            if "MARGIN" in rej_str or "HEAT" in rej_str or "DRAWDOWN" in rej_str:
                veto_cat = VetoCategory.PORTFOLIO_CAPACITY_VETO
            else:
                veto_cat = VetoCategory.SETUP_DEFECT

            return RiskVerdict(
                decision=RiskDecisionType.VETO_TO_SHADOW,
                approved_leverage=1,
                approved_notional_usd=0.0,
                net_rr=det_result.net_rr,
                reason=f"Deterministic bounding veto: {det_result.rejection_reason}",
                arbiter_size_multiplier=0.0,
                veto_category=veto_cat,
                existing_position_directive=llm_verdict.existing_position_directive,
                portfolio_risk_rationale=(
                    f"{llm_verdict.portfolio_risk_rationale} | "
                    f"Overridden by deterministic bounds: {det_result.rejection_reason}"
                ),
            )

        if llm_verdict.decision == RiskDecisionType.VETO_TO_SHADOW:
            return RiskVerdict(
                decision=RiskDecisionType.VETO_TO_SHADOW,
                approved_leverage=1,
                approved_notional_usd=0.0,
                net_rr=det_result.net_rr,
                reason=llm_verdict.reason,
                arbiter_size_multiplier=0.0,
                veto_category=llm_verdict.veto_category,
                existing_position_directive=llm_verdict.existing_position_directive,
                portfolio_risk_rationale=llm_verdict.portfolio_risk_rationale,
            )

        # Enforce that LLM approved size/leverage cannot exceed deterministic limits
        if llm_verdict.approved_notional_usd > 0.0:
            capped_notional: float = min(
                llm_verdict.approved_notional_usd,
                det_result.position_notional_usd,
            )
        else:
            capped_notional = det_result.position_notional_usd

        capped_leverage: int = min(
            llm_verdict.approved_leverage,
            det_result.recommended_leverage,
        )
        capped_leverage = max(
            self._calc.cfg.min_leverage,
            min(self._calc.cfg.max_leverage, capped_leverage),
        )

        return RiskVerdict(
            decision=llm_verdict.decision,
            approved_leverage=capped_leverage,
            approved_notional_usd=round(capped_notional, 2),
            net_rr=det_result.net_rr,
            reason=llm_verdict.reason,
            arbiter_size_multiplier=llm_verdict.arbiter_size_multiplier,
            veto_category=None,
            existing_position_directive=llm_verdict.existing_position_directive,
            portfolio_risk_rationale=llm_verdict.portfolio_risk_rationale,
        )
