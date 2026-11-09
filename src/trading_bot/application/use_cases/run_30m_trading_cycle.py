"""30-Minute Trading Cycle Use Case & BTC-Lead Cascade Orchestrator.

Orchestrates the 30-minute execution cycle:
1. Macro Sentinel blackout query on BTCUSDT. Halts execution if blackout is active.
2. Unified portfolio and margin state fetch via ExecutionGatewayPort.
3. BTC P0 Evaluation:
   - Funnel evaluation through TieredFunnelEvaluator.
   - MarketScoutPort deliberation to form BTC MarketThesis.
   - TradeArchitectPort blueprint construction (if not FLAT).
   - RiskArbiterPort evaluation against live account context and memory.
   - ExecutionGatewayPort bracket order placement on approval (or shadow marking on veto).
4. Altcoin P1 Concurrent Execution (ETHUSDT, SOLUSDT, BNBUSDT, etc.):
   - Evaluated concurrently via asyncio.gather.
   - Ingests lead BTC MarketThesis into altcoin scout deliberations.
   - Constructs blueprints and evaluates risk verdicts.
   - Dispatches approved bracket orders or flags shadow orders.

Enforces Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
Clean Architecture: Application layer depends ONLY on Domain and Ports.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from trading_bot.application.use_cases.evaluate_tiered_funnel import (
    FunnelStage,
    TieredFunnelEvaluator,
    TieredFunnelInput,
)
from trading_bot.domain.entities.common.enums import (
    RiskDecisionType,
    TradeSide,
)
from trading_bot.domain.entities.execution.trade_blueprint import (
    BracketOrderTicket,
    MarketThesis,
    RiskVerdict,
    TradeBlueprint,
)
from trading_bot.domain.entities.memory.episodic_case import (
    CohortBaseRateSummary,
    StratifiedMemoryPack,
)
from trading_bot.domain.entities.risk.account_state import AccountPortfolioContext
from trading_bot.ports.agent_ports import (
    MarketScoutPort,
    RiskArbiterPort,
    TradeArchitectPort,
)
from trading_bot.ports.execution_gateway_port import ExecutionGatewayPort
from trading_bot.ports.macro_sentinel_port import (
    MacroBlackoutState,
    MacroSentinelPort,
)
from trading_bot.ports.memory_repository_port import MemoryRepositoryPort


class SymbolCycleResult(BaseModel):
    """Result of 30-minute execution cycle for a single symbol (Zero Defaults)."""

    symbol: str
    funnel_passed: bool
    funnel_stage: str
    thesis: MarketThesis | None
    blueprint: TradeBlueprint | None
    verdict: RiskVerdict | None
    order_id: str | None
    is_shadow: bool
    rejection_reason: str | None


class TradingCycleSummary(BaseModel):
    """Unified summary of the 30-minute cascade execution cycle (Zero Defaults)."""

    cycle_timestamp_ms: int
    btc_result: SymbolCycleResult
    altcoin_results: dict[str, SymbolCycleResult]
    executed_orders_count: int
    shadow_orders_count: int


class Run30mTradingCycleUseCase:
    """Orchestrates 30-minute BTC-lead cascade trading cycle across all target symbols."""

    def __init__(
        self,
        funnel_evaluator: TieredFunnelEvaluator,
        scout: MarketScoutPort,
        architect: TradeArchitectPort,
        arbiter: RiskArbiterPort,
        execution_gateway: ExecutionGatewayPort,
        macro_sentinel: MacroSentinelPort,
        memory_repo: MemoryRepositoryPort | None,
    ) -> None:
        self.funnel_evaluator: TieredFunnelEvaluator = funnel_evaluator
        self.scout: MarketScoutPort = scout
        self.architect: TradeArchitectPort = architect
        self.arbiter: RiskArbiterPort = arbiter
        self.execution_gateway: ExecutionGatewayPort = execution_gateway
        self.macro_sentinel: MacroSentinelPort = macro_sentinel
        self.memory_repo: MemoryRepositoryPort | None = memory_repo

    async def execute_cycle(
        self,
        cycle_time_utc: datetime,
        market_inputs: dict[str, TieredFunnelInput],
        current_prices: dict[str, float],
        atr_30m_values: dict[str, float],
        quant_packets: dict[str, dict[str, Any]],
    ) -> TradingCycleSummary:
        """Execute the 30-minute trading cycle with BTC-lead cascade and altcoin parallelism."""
        cycle_timestamp_ms = int(cycle_time_utc.timestamp() * 1000)

        # 1. Macro Blackout Sentinel Check
        macro_state = await self.macro_sentinel.check_blackout("BTCUSDT", cycle_time_utc)
        if macro_state.is_blackout:
            btc_res = SymbolCycleResult(
                symbol="BTCUSDT",
                funnel_passed=False,
                funnel_stage=FunnelStage.STAGE_A_RULES.value,
                thesis=None,
                blueprint=None,
                verdict=None,
                order_id=None,
                is_shadow=False,
                rejection_reason=f"MACRO_BLACKOUT: {macro_state.reason}",
            )
            halted_altcoins: dict[str, SymbolCycleResult] = {}
            for sym in market_inputs:
                if sym != "BTCUSDT":
                    halted_altcoins[sym] = SymbolCycleResult(
                        symbol=sym,
                        funnel_passed=False,
                        funnel_stage=FunnelStage.STAGE_A_RULES.value,
                        thesis=None,
                        blueprint=None,
                        verdict=None,
                        order_id=None,
                        is_shadow=False,
                        rejection_reason=f"MACRO_BLACKOUT: {macro_state.reason}",
                    )
            return TradingCycleSummary(
                cycle_timestamp_ms=cycle_timestamp_ms,
                btc_result=btc_res,
                altcoin_results=halted_altcoins,
                executed_orders_count=0,
                shadow_orders_count=0,
            )

        # 2. Live Account Portfolio Context Fetch
        portfolio_context = await self.execution_gateway.fetch_account_portfolio_context()

        # 3. BTC P0 Evaluation
        btc_result: SymbolCycleResult
        lead_btc_thesis: MarketThesis | None = None

        if "BTCUSDT" in market_inputs:
            btc_inp = market_inputs["BTCUSDT"]
            btc_res, raw_btc_thesis = await self._process_symbol(
                symbol="BTCUSDT",
                inp=btc_inp,
                macro_state=macro_state,
                portfolio_context=portfolio_context,
                current_prices=current_prices,
                atr_30m_values=atr_30m_values,
                quant_packets=quant_packets,
                btc_thesis=None,
            )
            btc_result = btc_res
            if raw_btc_thesis is not None and raw_btc_thesis.proposed_side != TradeSide.FLAT:
                lead_btc_thesis = raw_btc_thesis
            else:
                lead_btc_thesis = None
        else:
            btc_result = SymbolCycleResult(
                symbol="BTCUSDT",
                funnel_passed=False,
                funnel_stage=FunnelStage.STAGE_A_RULES.value,
                thesis=None,
                blueprint=None,
                verdict=None,
                order_id=None,
                is_shadow=False,
                rejection_reason="BTCUSDT_NOT_IN_MARKET_INPUTS",
            )
            lead_btc_thesis = None

        # 4. Altcoins P1 Parallel Execution
        alt_symbols = [s for s in market_inputs if s != "BTCUSDT"]
        alt_tasks = [
            self._process_symbol(
                symbol=sym,
                inp=market_inputs[sym],
                macro_state=macro_state,
                portfolio_context=portfolio_context,
                current_prices=current_prices,
                atr_30m_values=atr_30m_values,
                quant_packets=quant_packets,
                btc_thesis=lead_btc_thesis,
            )
            for sym in alt_symbols
        ]
        alt_outputs = await asyncio.gather(*alt_tasks)

        altcoin_results: dict[str, SymbolCycleResult] = {}
        for sym, (alt_res, _) in zip(alt_symbols, alt_outputs, strict=True):
            altcoin_results[sym] = alt_res

        # 5. Summary Counts
        executed_orders_count = 0
        if btc_result.order_id is not None:
            executed_orders_count += 1
        for alt_res in altcoin_results.values():
            if alt_res.order_id is not None:
                executed_orders_count += 1

        shadow_orders_count = 0
        if btc_result.is_shadow:
            shadow_orders_count += 1
        for alt_res in altcoin_results.values():
            if alt_res.is_shadow:
                shadow_orders_count += 1

        return TradingCycleSummary(
            cycle_timestamp_ms=cycle_timestamp_ms,
            btc_result=btc_result,
            altcoin_results=altcoin_results,
            executed_orders_count=executed_orders_count,
            shadow_orders_count=shadow_orders_count,
        )

    async def _process_symbol(
        self,
        symbol: str,
        inp: TieredFunnelInput,
        macro_state: MacroBlackoutState,
        portfolio_context: AccountPortfolioContext,
        current_prices: dict[str, float],
        atr_30m_values: dict[str, float],
        quant_packets: dict[str, dict[str, Any]],
        btc_thesis: MarketThesis | None,
    ) -> tuple[SymbolCycleResult, MarketThesis | None]:
        """Process a single symbol through funnel, agent crew deliberation, risk arbitration, and execution."""
        # 1. Funnel Evaluation
        funnel_verdict = self.funnel_evaluator.evaluate(inp)
        if not funnel_verdict.passed:
            return (
                SymbolCycleResult(
                    symbol=symbol,
                    funnel_passed=False,
                    funnel_stage=funnel_verdict.stage.value,
                    thesis=None,
                    blueprint=None,
                    verdict=None,
                    order_id=None,
                    is_shadow=False,
                    rejection_reason=funnel_verdict.rejection_reason,
                ),
                None,
            )

        # 2. Scout Market Deliberation
        quant_packet = quant_packets[symbol]
        thesis = await self.scout.analyze_market(
            symbol=symbol,
            quant_packet=quant_packet,
            playbook=funnel_verdict.playbook,
            macro_state=macro_state,
            btc_thesis=btc_thesis,
        )

        if thesis.proposed_side == TradeSide.FLAT:
            return (
                SymbolCycleResult(
                    symbol=symbol,
                    funnel_passed=True,
                    funnel_stage=funnel_verdict.stage.value,
                    thesis=thesis,
                    blueprint=None,
                    verdict=None,
                    order_id=None,
                    is_shadow=False,
                    rejection_reason="SCOUT_PROPOSED_FLAT",
                ),
                thesis,
            )

        # 3. Retrieve Episodic Memory if repository is configured
        cohort_summary: CohortBaseRateSummary | None = None
        memory_pack: StratifiedMemoryPack | None = None
        if self.memory_repo is not None and "vector_16d" in quant_packet:
            cohort_summary = await self.memory_repo.compute_cohort_summary(
                quant_packet["vector_16d"],
                funnel_verdict.playbook.value,
                20,
            )
            memory_pack = await self.memory_repo.fetch_stratified_pack(
                quant_packet["vector_16d"],
                funnel_verdict.playbook.value,
                20,
            )

        # 4. Construct Trade Blueprint
        current_price = current_prices[symbol]
        atr_30m = atr_30m_values[symbol]
        blueprint = await self.architect.construct_trade_blueprint(
            thesis=thesis,
            current_price=current_price,
            atr_30m=atr_30m,
            cohort_summary=cohort_summary,
        )

        # 5. Evaluate Risk Verdict
        verdict = await self.arbiter.evaluate_risk(
            blueprint=blueprint,
            account_context=portfolio_context,
            memory_pack=memory_pack,
        )

        # 6. Execution Dispatch or Shadow Order Marking
        if verdict.decision in (
            RiskDecisionType.APPROVE_FULL,
            RiskDecisionType.APPROVE_REDUCED_SIZE,
        ):
            quantity = verdict.approved_notional_usd / blueprint.entry_price
            ticket = BracketOrderTicket(
                order_id=f"order_{symbol}_{uuid.uuid4().hex[:10]}",
                symbol=blueprint.thesis.symbol,
                side=blueprint.thesis.proposed_side,
                entry_type=blueprint.order_type,
                entry_price=blueprint.entry_price,
                stop_loss_price=blueprint.stop_loss_price,
                take_profit_price=blueprint.take_profit_price,
                quantity=quantity,
                leverage=verdict.approved_leverage,
            )
            order_id = await self.execution_gateway.place_bracket_order(ticket)
            return (
                SymbolCycleResult(
                    symbol=symbol,
                    funnel_passed=True,
                    funnel_stage=funnel_verdict.stage.value,
                    thesis=thesis,
                    blueprint=blueprint,
                    verdict=verdict,
                    order_id=order_id,
                    is_shadow=False,
                    rejection_reason=None,
                ),
                thesis,
            )

        if verdict.decision == RiskDecisionType.VETO_TO_SHADOW:
            return (
                SymbolCycleResult(
                    symbol=symbol,
                    funnel_passed=True,
                    funnel_stage=funnel_verdict.stage.value,
                    thesis=thesis,
                    blueprint=blueprint,
                    verdict=verdict,
                    order_id=None,
                    is_shadow=True,
                    rejection_reason=verdict.reason,
                ),
                thesis,
            )

        return (
            SymbolCycleResult(
                symbol=symbol,
                funnel_passed=True,
                funnel_stage=funnel_verdict.stage.value,
                thesis=thesis,
                blueprint=blueprint,
                verdict=verdict,
                order_id=None,
                is_shadow=False,
                rejection_reason=verdict.reason,
            ),
            thesis,
        )
