"""Unit tests for Run30mTradingCycleUseCase application orchestrator.

Verifies:
- BTC-lead cascade: BTC scout generates thesis, altcoins receive BTC thesis in analyze_market
- BTC flat/failure: altcoins receive btc_thesis=None
- Order placement on approved risk (APPROVE_FULL, APPROVE_REDUCED_SIZE)
- Shadow order flagging on veto (VETO_TO_SHADOW)
- Altcoin parallel execution across ETH, SOL, BNB
- Macro blackout halting (stops execution immediately)
- Invariant #9 compliance (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from trading_bot.application.services.shadow_order_tracker import ShadowOrderTracker
from trading_bot.application.use_cases.evaluate_tiered_funnel import (
    TieredFunnelEvaluator,
    TieredFunnelInput,
)
from trading_bot.application.use_cases.run_30m_trading_cycle import (
    Run30mTradingCycleUseCase,
    TradingCycleSummary,
)
from trading_bot.domain.entities.common.enums import (
    EntryOrderType,
    ExecutionMode,
    ExistingPositionDirective,
    OIPriceQuadrant,
    PlaybookType,
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
    EpisodicCase,
    StratifiedMemoryPack,
)
from trading_bot.domain.entities.risk.account_state import (
    AccountPortfolioContext,
    ActivePositionSnapshot,
    PendingOrderSnapshot,
)
from trading_bot.domain.services.execution.fee_to_edge_gate import (
    FeeToEdgeConfig,
    FeeToEdgeGatingService,
)
from trading_bot.domain.services.execution.playbook_validator import (
    PlaybookConfig,
    PlaybookValidator,
)
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

# ==============================================================================
# Test Doubles / Mocks (100% Invariant #9 Compliant: Zero Defaults)
# ==============================================================================


class MockMacroSentinel(MacroSentinelPort):
    def __init__(self, is_blackout: bool, reason: str) -> None:
        self.is_blackout: bool = is_blackout
        self.reason: str = reason
        self.calls_count: int = 0

    async def check_blackout(
        self,
        symbol: str,
        current_time_utc: datetime,
    ) -> MacroBlackoutState:
        self.calls_count += 1
        return MacroBlackoutState(
            is_blackout=self.is_blackout,
            upcoming_event="CPI_RELEASE" if self.is_blackout else None,
            time_to_event_minutes=10.0 if self.is_blackout else None,
            reason=self.reason,
        )


class MockExecutionGateway(ExecutionGatewayPort):
    def __init__(self, portfolio_context: AccountPortfolioContext) -> None:
        self.portfolio_context: AccountPortfolioContext = portfolio_context
        self.placed_tickets: list[BracketOrderTicket] = []
        self.fetch_context_calls: int = 0

    async def place_bracket_order(self, ticket: BracketOrderTicket) -> str:
        self.placed_tickets.append(ticket)
        return f"exchange_order_{ticket.symbol}_{len(self.placed_tickets)}"

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        return True

    async def get_position_notional(self, symbol: str) -> float:
        return 0.0

    async def fetch_account_portfolio_context(self) -> AccountPortfolioContext:
        self.fetch_context_calls += 1
        return self.portfolio_context

    async def fetch_active_positions(self) -> list[ActivePositionSnapshot]:
        return []

    async def fetch_pending_orders(self) -> list[PendingOrderSnapshot]:
        return []


class MockMarketScout(MarketScoutPort):
    def __init__(self, proposed_sides: dict[str, TradeSide]) -> None:
        self.proposed_sides: dict[str, TradeSide] = proposed_sides
        self.received_btc_theses: dict[str, MarketThesis | None] = {}
        self.calls: list[str] = []

    async def analyze_market(
        self,
        symbol: str,
        quant_packet: dict[str, Any],
        playbook: PlaybookType,
        macro_state: MacroBlackoutState,
        btc_thesis: MarketThesis | None,
    ) -> MarketThesis:
        self.calls.append(symbol)
        self.received_btc_theses[symbol] = btc_thesis
        side = self.proposed_sides[symbol]
        return MarketThesis(
            symbol=symbol,
            selected_playbook=playbook,
            proposed_side=side,
            supporting_active_features=["trend_continuation_z"],
            regime_alignment_score=0.88,
            thesis_summary=f"Analysis for {symbol} yielding {side.value}",
        )


class MockTradeArchitect(TradeArchitectPort):
    def __init__(self) -> None:
        self.constructed_symbols: list[str] = []

    async def construct_trade_blueprint(
        self,
        thesis: MarketThesis,
        current_price: float,
        atr_30m: float,
        cohort_summary: CohortBaseRateSummary | None,
    ) -> TradeBlueprint:
        self.constructed_symbols.append(thesis.symbol)
        sl_price = (
            current_price - 1.5 * atr_30m
            if thesis.proposed_side == TradeSide.LONG
            else current_price + 1.5 * atr_30m
        )
        tp_price = (
            current_price + 3.0 * atr_30m
            if thesis.proposed_side == TradeSide.LONG
            else current_price - 3.0 * atr_30m
        )
        return TradeBlueprint(
            thesis=thesis,
            order_type=EntryOrderType.MARKET,
            entry_price=current_price,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            rationale="ATR-MAE calibrated blueprint",
        )


class MockRiskArbiter(RiskArbiterPort):
    def __init__(self, decisions: dict[str, RiskDecisionType]) -> None:
        self.decisions: dict[str, RiskDecisionType] = decisions
        self.evaluated_blueprints: list[TradeBlueprint] = []

    async def evaluate_risk(
        self,
        blueprint: TradeBlueprint,
        account_context: AccountPortfolioContext,
        memory_pack: StratifiedMemoryPack | None,
    ) -> RiskVerdict:
        self.evaluated_blueprints.append(blueprint)
        decision = self.decisions[blueprint.thesis.symbol]
        approved_notional = 5000.0 if decision != RiskDecisionType.VETO_TO_SHADOW else 0.0
        reason = (
            "APPROVED_BY_QUARTER_KELLY"
            if approved_notional > 0
            else "PORTFOLIO_HEAT_LIMIT_EXCEEDED"
        )
        return RiskVerdict(
            decision=decision,
            approved_leverage=5,
            approved_notional_usd=approved_notional,
            net_rr=2.0,
            reason=reason,
            arbiter_size_multiplier=1.0,
            veto_category=None,
            existing_position_directive=ExistingPositionDirective.HOLD_EXISTING,
            portfolio_risk_rationale="Simulated arbiter verdict",
        )


class MockMemoryRepo(MemoryRepositoryPort):
    def __init__(self) -> None:
        self.query_count: int = 0

    async def query_knn_cohort(
        self,
        vector_16d: list[float],
        playbook: str,
        k: int,
    ) -> list[EpisodicCase]:
        self.query_count += 1
        return []

    async def compute_cohort_summary(
        self,
        vector_16d: list[float],
        playbook: str,
        k: int,
    ) -> CohortBaseRateSummary:
        self.query_count += 1
        return CohortBaseRateSummary(
            has_historical_cohort=True,
            total_neighbors=20,
            weighted_win_rate=0.65,
            stop_hunted_then_tp_rate=0.10,
            true_failure_trap_rate=0.15,
            expected_r_multiple=1.8,
            mae_target_percentile_atr=0.85,
            mfe_median_atr=1.5,
        )

    async def fetch_stratified_pack(
        self,
        vector_16d: list[float],
        playbook: str,
        k: int,
    ) -> StratifiedMemoryPack:
        self.query_count += 1
        summary = await self.compute_cohort_summary(vector_16d, playbook, k)
        return StratifiedMemoryPack(
            cohort_summary=summary,
            bucket1_bull_missed=[],
            bucket2_failure_traps=[],
            bucket3_stop_hunts=[],
        )

    async def upsert_case(self, case: EpisodicCase) -> None:
        pass

    async def boost_lesson_confidence(self, case_id: str, delta: float) -> None:
        pass


# ==============================================================================
# Helper Factories
# ==============================================================================


def make_portfolio_context() -> AccountPortfolioContext:
    return AccountPortfolioContext(
        execution_mode=ExecutionMode.TESTNET,
        timestamp_ms=1700000000000,
        wallet_balance_usd=50000.0,
        unrealized_pnl_usd=0.0,
        equity_nav_usd=50000.0,
        used_margin_usd=0.0,
        available_margin_usd=50000.0,
        margin_utilization_pct=0.0,
        current_daily_pnl_usd=0.0,
        current_daily_dd_pct=0.0,
        remaining_daily_dd_budget_pct=3.0,
        current_max_dd_pct=0.0,
        remaining_max_dd_budget_pct=6.0,
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


def make_funnel_evaluator() -> TieredFunnelEvaluator:
    fee_service = FeeToEdgeGatingService(
        FeeToEdgeConfig(
            multiplier=1.0,
            roundtrip_cost_rate=0.0010,
            min_calibrated_prob=0.52,
        )
    )
    pb_validator = PlaybookValidator(
        PlaybookConfig(
            trend_hurst_min=0.50,
            trend_vpin_percentile_max=88.0,
            trend_min_calibrated_prob=0.52,
            reversal_hurst_max=0.49,
            reversal_liq_z_min=1.8,
            reversal_funding_z_abs_min=2.0,
            reversal_vpin_slope_max=0.0,
            lead_lag_btc_ret_z_min=1.5,
            lead_lag_btc_beta_min=0.50,
            lead_lag_catchup_gap_z_min=1.0,
            altcoin_symbols=["ETHUSDT", "SOLUSDT", "BNBUSDT"],
        )
    )
    return TieredFunnelEvaluator(
        fee_to_edge_service=fee_service,
        playbook_validator=pb_validator,
        max_spread_bps=5.0,
        min_yz_vol=0.002,
        max_uncertainty_width=0.15,
    )


def make_symbol_input(symbol: str, passed: bool) -> TieredFunnelInput:
    if passed:
        p_long = 0.60
        p_short = 0.40
    else:
        p_long = 0.45
        p_short = 0.45
    return TieredFunnelInput(
        symbol=symbol,
        spread_bps=2.0,
        yang_zhang_vol=0.015,
        garman_klass_vol=0.015,
        is_blackout=False,
        is_account_frozen=False,
        p_long_cal=p_long,
        p_short_cal=p_short,
        uncertainty_width=0.08,
        is_ood=False,
        hurst_exp=0.55,
        oi_quadrant=OIPriceQuadrant.LONG_BUILDUP,
        vpin_percentile=60.0,
        liq_long_z=0.1,
        liq_short_z=0.1,
        funding_z=0.0,
        vpin_slope_3bars=-0.02,
        btc_15m_return_z=0.0,
        btc_lead_beta=1.0,
        lead_lag_catchup_gap_z=0.0,
    )


# ==============================================================================
# Test Cases
# ==============================================================================


@pytest.mark.asyncio
async def test_run_cycle_btc_lead_cascade_success() -> None:
    """Verifies BTC generates thesis first, and altcoins receive BTC's thesis in analyze_market."""
    funnel = make_funnel_evaluator()
    sentinel = MockMacroSentinel(is_blackout=False, reason="NO_BLACKOUT")
    gateway = MockExecutionGateway(make_portfolio_context())
    scout = MockMarketScout(
        {
            "BTCUSDT": TradeSide.LONG,
            "ETHUSDT": TradeSide.LONG,
            "SOLUSDT": TradeSide.LONG,
        }
    )
    architect = MockTradeArchitect()
    arbiter = MockRiskArbiter(
        {
            "BTCUSDT": RiskDecisionType.APPROVE_FULL,
            "ETHUSDT": RiskDecisionType.APPROVE_REDUCED_SIZE,
            "SOLUSDT": RiskDecisionType.APPROVE_FULL,
        }
    )
    memory_repo = MockMemoryRepo()

    use_case = Run30mTradingCycleUseCase(
        funnel_evaluator=funnel,
        scout=scout,
        architect=architect,
        arbiter=arbiter,
        execution_gateway=gateway,
        macro_sentinel=sentinel,
        memory_repo=memory_repo,
        shadow_order_tracker=None,
    )

    cycle_time = datetime(2026, 11, 9, 14, 30, tzinfo=UTC)
    market_inputs = {
        "BTCUSDT": make_symbol_input("BTCUSDT", True),
        "ETHUSDT": make_symbol_input("ETHUSDT", True),
        "SOLUSDT": make_symbol_input("SOLUSDT", True),
    }
    current_prices = {
        "BTCUSDT": 90000.0,
        "ETHUSDT": 3200.0,
        "SOLUSDT": 180.0,
    }
    atr_30m_values = {
        "BTCUSDT": 500.0,
        "ETHUSDT": 30.0,
        "SOLUSDT": 3.5,
    }
    quant_packets: dict[str, dict[str, Any]] = {
        "BTCUSDT": {"vector_16d": [0.1] * 16},
        "ETHUSDT": {"vector_16d": [0.2] * 16},
        "SOLUSDT": {"vector_16d": [0.3] * 16},
    }

    summary = await use_case.execute_cycle(
        cycle_time_utc=cycle_time,
        market_inputs=market_inputs,
        current_prices=current_prices,
        atr_30m_values=atr_30m_values,
        quant_packets=quant_packets,
    )

    # 1. Summary validation
    assert isinstance(summary, TradingCycleSummary)
    assert summary.executed_orders_count == 3
    assert summary.shadow_orders_count == 0

    # 2. BTC result validation
    assert summary.btc_result.funnel_passed is True
    assert summary.btc_result.order_id is not None
    assert summary.btc_result.thesis is not None
    assert summary.btc_result.thesis.proposed_side == TradeSide.LONG

    # 3. Cascade check: Altcoins received BTC thesis
    assert scout.received_btc_theses["BTCUSDT"] is None
    assert scout.received_btc_theses["ETHUSDT"] is not None
    assert scout.received_btc_theses["ETHUSDT"].symbol == "BTCUSDT"
    assert scout.received_btc_theses["ETHUSDT"].proposed_side == TradeSide.LONG
    assert scout.received_btc_theses["SOLUSDT"] is not None
    assert scout.received_btc_theses["SOLUSDT"].symbol == "BTCUSDT"

    # 4. Gateway execution check
    assert len(gateway.placed_tickets) == 3
    btc_ticket = next(t for t in gateway.placed_tickets if t.symbol == "BTCUSDT")
    assert btc_ticket.side == TradeSide.LONG
    assert btc_ticket.entry_price == 90000.0
    assert btc_ticket.quantity == pytest.approx(5000.0 / 90000.0)


@pytest.mark.asyncio
async def test_run_cycle_btc_flat_propagates_none_to_altcoins() -> None:
    """When BTC scout proposes FLAT, btc_thesis=None must be passed to altcoins."""
    funnel = make_funnel_evaluator()
    sentinel = MockMacroSentinel(is_blackout=False, reason="NO_BLACKOUT")
    gateway = MockExecutionGateway(make_portfolio_context())
    scout = MockMarketScout(
        {
            "BTCUSDT": TradeSide.FLAT,
            "ETHUSDT": TradeSide.LONG,
        }
    )
    architect = MockTradeArchitect()
    arbiter = MockRiskArbiter(
        {
            "BTCUSDT": RiskDecisionType.APPROVE_FULL,
            "ETHUSDT": RiskDecisionType.APPROVE_FULL,
        }
    )

    use_case = Run30mTradingCycleUseCase(
        funnel_evaluator=funnel,
        scout=scout,
        architect=architect,
        arbiter=arbiter,
        execution_gateway=gateway,
        macro_sentinel=sentinel,
        memory_repo=None,
        shadow_order_tracker=None,
    )

    cycle_time = datetime(2026, 11, 9, 14, 30, tzinfo=UTC)
    market_inputs = {
        "BTCUSDT": make_symbol_input("BTCUSDT", True),
        "ETHUSDT": make_symbol_input("ETHUSDT", True),
    }
    current_prices = {"BTCUSDT": 90000.0, "ETHUSDT": 3200.0}
    atr_30m_values = {"BTCUSDT": 500.0, "ETHUSDT": 30.0}
    quant_packets: dict[str, dict[str, Any]] = {"BTCUSDT": {}, "ETHUSDT": {}}

    summary = await use_case.execute_cycle(
        cycle_time_utc=cycle_time,
        market_inputs=market_inputs,
        current_prices=current_prices,
        atr_30m_values=atr_30m_values,
        quant_packets=quant_packets,
    )

    assert summary.btc_result.rejection_reason == "SCOUT_PROPOSED_FLAT"
    assert summary.btc_result.order_id is None
    # Altcoin received btc_thesis=None
    assert scout.received_btc_theses["ETHUSDT"] is None
    # Only ETH order was placed
    assert summary.executed_orders_count == 1
    assert len(gateway.placed_tickets) == 1
    assert gateway.placed_tickets[0].symbol == "ETHUSDT"


@pytest.mark.asyncio
async def test_run_cycle_veto_flags_shadow_orders() -> None:
    """When arbiter issues VETO_TO_SHADOW, order must NOT be placed and is_shadow flagged."""
    funnel = make_funnel_evaluator()
    sentinel = MockMacroSentinel(is_blackout=False, reason="NO_BLACKOUT")
    gateway = MockExecutionGateway(make_portfolio_context())
    scout = MockMarketScout(
        {
            "BTCUSDT": TradeSide.LONG,
            "SOLUSDT": TradeSide.SHORT,
        }
    )
    architect = MockTradeArchitect()
    arbiter = MockRiskArbiter(
        {
            "BTCUSDT": RiskDecisionType.VETO_TO_SHADOW,  # Vetoed
            "SOLUSDT": RiskDecisionType.APPROVE_FULL,
        }
    )

    use_case = Run30mTradingCycleUseCase(
        funnel_evaluator=funnel,
        scout=scout,
        architect=architect,
        arbiter=arbiter,
        execution_gateway=gateway,
        macro_sentinel=sentinel,
        memory_repo=None,
        shadow_order_tracker=None,
    )

    cycle_time = datetime(2026, 11, 9, 14, 30, tzinfo=UTC)
    market_inputs = {
        "BTCUSDT": make_symbol_input("BTCUSDT", True),
        "SOLUSDT": make_symbol_input("SOLUSDT", True),
    }
    current_prices = {"BTCUSDT": 90000.0, "SOLUSDT": 180.0}
    atr_30m_values = {"BTCUSDT": 500.0, "SOLUSDT": 3.5}
    quant_packets: dict[str, dict[str, Any]] = {"BTCUSDT": {}, "SOLUSDT": {}}

    summary = await use_case.execute_cycle(
        cycle_time_utc=cycle_time,
        market_inputs=market_inputs,
        current_prices=current_prices,
        atr_30m_values=atr_30m_values,
        quant_packets=quant_packets,
    )

    # BTC was vetoed to shadow
    assert summary.btc_result.is_shadow is True
    assert summary.btc_result.order_id is None
    assert summary.shadow_orders_count == 1
    assert summary.executed_orders_count == 1

    # Only SOL order reached gateway
    assert len(gateway.placed_tickets) == 1
    assert gateway.placed_tickets[0].symbol == "SOLUSDT"


@pytest.mark.asyncio
async def test_run_cycle_macro_blackout_halts_execution() -> None:
    """When macro blackout is active, entire cycle halts immediately with 0 orders placed."""
    funnel = make_funnel_evaluator()
    sentinel = MockMacroSentinel(
        is_blackout=True,
        reason="US_CPI_RELEASE_IN_10M",
    )
    gateway = MockExecutionGateway(make_portfolio_context())
    scout = MockMarketScout({"BTCUSDT": TradeSide.LONG})
    architect = MockTradeArchitect()
    arbiter = MockRiskArbiter({"BTCUSDT": RiskDecisionType.APPROVE_FULL})

    use_case = Run30mTradingCycleUseCase(
        funnel_evaluator=funnel,
        scout=scout,
        architect=architect,
        arbiter=arbiter,
        execution_gateway=gateway,
        macro_sentinel=sentinel,
        memory_repo=None,
        shadow_order_tracker=None,
    )

    cycle_time = datetime(2026, 11, 9, 13, 20, tzinfo=UTC)
    market_inputs = {
        "BTCUSDT": make_symbol_input("BTCUSDT", True),
        "ETHUSDT": make_symbol_input("ETHUSDT", True),
        "SOLUSDT": make_symbol_input("SOLUSDT", True),
    }
    current_prices = {"BTCUSDT": 90000.0, "ETHUSDT": 3200.0, "SOLUSDT": 180.0}
    atr_30m_values = {"BTCUSDT": 500.0, "ETHUSDT": 30.0, "SOLUSDT": 3.5}
    quant_packets: dict[str, dict[str, Any]] = {"BTCUSDT": {}, "ETHUSDT": {}, "SOLUSDT": {}}

    summary = await use_case.execute_cycle(
        cycle_time_utc=cycle_time,
        market_inputs=market_inputs,
        current_prices=current_prices,
        atr_30m_values=atr_30m_values,
        quant_packets=quant_packets,
    )

    # 1. Macro sentinel called exactly once
    assert sentinel.calls_count == 1
    # 2. Portfolio context was NOT fetched (halted before gateway)
    assert gateway.fetch_context_calls == 0
    # 3. No scout/architect/arbiter calls
    assert len(scout.calls) == 0
    assert len(architect.constructed_symbols) == 0
    assert len(arbiter.evaluated_blueprints) == 0
    # 4. Zero orders executed or shadowed
    assert summary.executed_orders_count == 0
    assert summary.shadow_orders_count == 0
    assert len(gateway.placed_tickets) == 0
    # 5. Symbols marked with macro blackout reason
    assert summary.btc_result.rejection_reason is not None
    assert "MACRO_BLACKOUT" in summary.btc_result.rejection_reason
    assert summary.altcoin_results["ETHUSDT"].rejection_reason is not None
    assert "MACRO_BLACKOUT" in summary.altcoin_results["ETHUSDT"].rejection_reason
    assert summary.altcoin_results["SOLUSDT"].rejection_reason is not None
    assert "MACRO_BLACKOUT" in summary.altcoin_results["SOLUSDT"].rejection_reason


@pytest.mark.asyncio
async def test_run_cycle_parallel_altcoin_execution() -> None:
    """Verifies that all 3 altcoins are processed concurrently and populated in results."""
    funnel = make_funnel_evaluator()
    sentinel = MockMacroSentinel(is_blackout=False, reason="NO_BLACKOUT")
    gateway = MockExecutionGateway(make_portfolio_context())
    scout = MockMarketScout(
        {
            "BTCUSDT": TradeSide.LONG,
            "ETHUSDT": TradeSide.LONG,
            "SOLUSDT": TradeSide.LONG,
            "BNBUSDT": TradeSide.LONG,
        }
    )
    architect = MockTradeArchitect()
    arbiter = MockRiskArbiter(
        {
            "BTCUSDT": RiskDecisionType.APPROVE_FULL,
            "ETHUSDT": RiskDecisionType.APPROVE_FULL,
            "SOLUSDT": RiskDecisionType.APPROVE_FULL,
            "BNBUSDT": RiskDecisionType.APPROVE_FULL,
        }
    )

    use_case = Run30mTradingCycleUseCase(
        funnel_evaluator=funnel,
        scout=scout,
        architect=architect,
        arbiter=arbiter,
        execution_gateway=gateway,
        macro_sentinel=sentinel,
        memory_repo=None,
        shadow_order_tracker=None,
    )

    cycle_time = datetime(2026, 11, 9, 14, 30, tzinfo=UTC)
    market_inputs = {
        "BTCUSDT": make_symbol_input("BTCUSDT", True),
        "ETHUSDT": make_symbol_input("ETHUSDT", True),
        "SOLUSDT": make_symbol_input("SOLUSDT", True),
        "BNBUSDT": make_symbol_input("BNBUSDT", True),
    }
    current_prices = {
        "BTCUSDT": 90000.0,
        "ETHUSDT": 3200.0,
        "SOLUSDT": 180.0,
        "BNBUSDT": 620.0,
    }
    atr_30m_values = {
        "BTCUSDT": 500.0,
        "ETHUSDT": 30.0,
        "SOLUSDT": 3.5,
        "BNBUSDT": 8.0,
    }
    quant_packets: dict[str, dict[str, Any]] = {
        "BTCUSDT": {},
        "ETHUSDT": {},
        "SOLUSDT": {},
        "BNBUSDT": {},
    }

    summary = await use_case.execute_cycle(
        cycle_time_utc=cycle_time,
        market_inputs=market_inputs,
        current_prices=current_prices,
        atr_30m_values=atr_30m_values,
        quant_packets=quant_packets,
    )

    assert set(summary.altcoin_results.keys()) == {"ETHUSDT", "SOLUSDT", "BNBUSDT"}
    assert summary.executed_orders_count == 4
    assert len(gateway.placed_tickets) == 4


@pytest.mark.asyncio
async def test_run_cycle_funnel_rejection_skips_crew() -> None:
    """When a symbol fails the funnel, scout/architect/arbiter must not be called for it."""
    funnel = make_funnel_evaluator()
    sentinel = MockMacroSentinel(is_blackout=False, reason="NO_BLACKOUT")
    gateway = MockExecutionGateway(make_portfolio_context())
    scout = MockMarketScout({"BTCUSDT": TradeSide.LONG})
    architect = MockTradeArchitect()
    arbiter = MockRiskArbiter({"BTCUSDT": RiskDecisionType.APPROVE_FULL})

    use_case = Run30mTradingCycleUseCase(
        funnel_evaluator=funnel,
        scout=scout,
        architect=architect,
        arbiter=arbiter,
        execution_gateway=gateway,
        macro_sentinel=sentinel,
        memory_repo=None,
        shadow_order_tracker=None,
    )

    cycle_time = datetime(2026, 11, 9, 14, 30, tzinfo=UTC)
    # BTC fails funnel
    market_inputs = {"BTCUSDT": make_symbol_input("BTCUSDT", False)}
    current_prices = {"BTCUSDT": 90000.0}
    atr_30m_values = {"BTCUSDT": 500.0}
    quant_packets: dict[str, dict[str, Any]] = {"BTCUSDT": {}}

    summary = await use_case.execute_cycle(
        cycle_time_utc=cycle_time,
        market_inputs=market_inputs,
        current_prices=current_prices,
        atr_30m_values=atr_30m_values,
        quant_packets=quant_packets,
    )

    assert summary.btc_result.funnel_passed is False
    assert summary.btc_result.thesis is None
    assert summary.btc_result.blueprint is None
    assert summary.btc_result.order_id is None
    assert summary.executed_orders_count == 0
    assert len(scout.calls) == 0
    assert len(architect.constructed_symbols) == 0
    assert len(arbiter.evaluated_blueprints) == 0


@pytest.mark.asyncio
async def test_run_cycle_registers_shadow_orders_on_veto() -> None:
    """When arbiter issues VETO or VETO_TO_SHADOW and shadow_tracker is present, register in tracker."""
    funnel = make_funnel_evaluator()
    sentinel = MockMacroSentinel(is_blackout=False, reason="NO_BLACKOUT")
    gateway = MockExecutionGateway(make_portfolio_context())
    scout = MockMarketScout(
        {
            "BTCUSDT": TradeSide.LONG,
            "ETHUSDT": TradeSide.LONG,
        }
    )
    architect = MockTradeArchitect()
    arbiter = MockRiskArbiter(
        {
            "BTCUSDT": RiskDecisionType.VETO_TO_SHADOW,
            "ETHUSDT": RiskDecisionType.VETO,
        }
    )
    tracker = ShadowOrderTracker(max_tracking_duration_ms=21600000)

    use_case = Run30mTradingCycleUseCase(
        funnel_evaluator=funnel,
        scout=scout,
        architect=architect,
        arbiter=arbiter,
        execution_gateway=gateway,
        macro_sentinel=sentinel,
        memory_repo=None,
        shadow_order_tracker=tracker,
    )

    cycle_time = datetime(2026, 11, 9, 14, 30, tzinfo=UTC)
    market_inputs = {
        "BTCUSDT": make_symbol_input("BTCUSDT", True),
        "ETHUSDT": make_symbol_input("ETHUSDT", True),
    }
    current_prices = {"BTCUSDT": 90000.0, "ETHUSDT": 3200.0}
    atr_30m_values = {"BTCUSDT": 500.0, "ETHUSDT": 30.0}
    quant_packets: dict[str, dict[str, Any]] = {
        "BTCUSDT": {"atr_30m": 500.0, "timestamp_ms": 1731162600000},
        "ETHUSDT": {"atr_30m": 30.0, "timestamp_ms": 1731162600000},
    }

    summary = await use_case.execute_cycle(
        cycle_time_utc=cycle_time,
        market_inputs=market_inputs,
        current_prices=current_prices,
        atr_30m_values=atr_30m_values,
        quant_packets=quant_packets,
    )

    # 1. BTC was VETO_TO_SHADOW: is_shadow is True
    assert summary.btc_result.is_shadow is True
    assert summary.btc_result.order_id is None

    # 2. ETH was VETO: is_shadow is False
    assert summary.altcoin_results["ETHUSDT"].is_shadow is False
    assert summary.altcoin_results["ETHUSDT"].order_id is None

    # 3. Both candidates were registered in tracker
    active_orders = tracker.get_all_active_orders()
    assert len(active_orders) == 2
    symbols_in_tracker = {o.symbol for o in active_orders}
    assert symbols_in_tracker == {"BTCUSDT", "ETHUSDT"}

    # 4. Test post-SL registration helper
    sl_record = use_case.register_post_sl_position(
        symbol="SOLUSDT",
        side=TradeSide.LONG,
        entry_price=180.0,
        sl_price=175.0,
        tp_price=190.0,
        exit_timestamp_ms=1731162600000,
        quant_packet={"atr_30m": 3.5},
    )
    assert sl_record is not None
    assert sl_record.symbol == "SOLUSDT"
    assert sl_record.is_post_sl_tracker is True
    assert len(tracker.get_all_active_orders()) == 3
