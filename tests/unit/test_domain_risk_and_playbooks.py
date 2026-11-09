"""Unit tests for Pure Domain Services: RiskCalculator, PlaybookValidator, BarrierEvaluator."""

from __future__ import annotations

import pytest

from trading_bot.domain.entities.common.enums import (
    EntryOrderType,
    ExecutionMode,
    ExistingPositionDirective,
    OIPriceQuadrant,
    OutcomeCategory,
    PlaybookType,
    RiskDecisionType,
    TradeSide,
    VetoCategory,
)
from trading_bot.domain.entities.execution.trade_blueprint import RiskVerdict
from trading_bot.domain.entities.market.market_state import MicroBar1m
from trading_bot.domain.entities.risk.account_state import (
    AccountPortfolioContext,
    ActivePositionSnapshot,
    PendingOrderSnapshot,
)
from trading_bot.domain.services.execution.barrier_evaluator import BarrierConfig, BarrierEvaluator
from trading_bot.domain.services.execution.playbook_validator import (
    PlaybookConfig,
    PlaybookPreconditionsInput,
    PlaybookValidator,
)
from trading_bot.domain.services.risk.risk_calculator import (
    DeterministicRiskCalculator,
    RiskConfig,
)


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
        corr_scalar_one_same_dir=0.70,
        corr_scalar_two_plus_same_dir=0.40,
        daily_drawdown_halt_pct=0.04,
        max_drawdown_freeze_pct=0.08,
        min_barrier_distance_pct=0.003,
        min_leverage=1,
        max_leverage=10,
    )


@pytest.fixture
def playbook_config() -> PlaybookConfig:
    return PlaybookConfig(
        trend_hurst_min=0.54,
        trend_vpin_percentile_max=88.0,
        trend_min_calibrated_prob=0.57,
        reversal_hurst_max=0.49,
        reversal_liq_z_min=2.0,
        reversal_funding_z_abs_min=2.2,
        reversal_vpin_slope_max=0.0,
        lead_lag_btc_ret_z_min=1.8,
        lead_lag_btc_beta_min=0.55,
        lead_lag_catchup_gap_z_min=1.5,
        altcoin_symbols=["ETHUSDT", "SOLUSDT", "BNBUSDT"],
    )


@pytest.fixture
def barrier_config() -> BarrierConfig:
    return BarrierConfig(
        max_holding_bars=10,
        post_sl_tracking_bars=10,
    )


# -------------------------------------------------------------------------
# DeterministicRiskCalculator Tests
# -------------------------------------------------------------------------


def test_risk_calculator_approved_trade(risk_config: RiskConfig) -> None:
    calc = DeterministicRiskCalculator(risk_config)
    result = calc.evaluate_and_size_trade(
        nav_usd=10000.0,
        current_daily_dd_pct=0.01,
        current_max_dd_pct=0.02,
        effective_open_heat_pct=0.01,
        same_direction_open_count=0,
        entry_price=60000.0,
        stop_loss_price=59400.0,  # 1.0% SL
        take_profit_price=61500.0,  # 2.5% TP
        is_limit_entry=True,
        kyle_slippage_pct=0.0003,
        calibrated_win_prob=0.60,
        venn_abers_uncertainty_width=0.05,
        ood_shrinkage_factor=1.0,
        current_yz_vol=0.015,
        pre_event_blackout=False,
        available_margin_usd=10000.0,
        used_margin_usd=0.0,
        arbiter_size_multiplier=1.0,
    )
    assert result.approved is True
    assert result.rejection_reason is None
    assert result.net_rr >= 1.5
    assert result.risk_amount_usd > 0.0
    assert result.position_notional_usd > 0.0
    assert 1 <= result.recommended_leverage <= 10


def test_risk_calculator_daily_dd_breaker(risk_config: RiskConfig) -> None:
    calc = DeterministicRiskCalculator(risk_config)
    result = calc.evaluate_and_size_trade(
        nav_usd=10000.0,
        current_daily_dd_pct=0.05,  # Exceeds 0.04
        current_max_dd_pct=0.02,
        effective_open_heat_pct=0.0,
        same_direction_open_count=0,
        entry_price=60000.0,
        stop_loss_price=59400.0,
        take_profit_price=61500.0,
        is_limit_entry=True,
        kyle_slippage_pct=0.0003,
        calibrated_win_prob=0.60,
        venn_abers_uncertainty_width=0.05,
        ood_shrinkage_factor=1.0,
        current_yz_vol=0.015,
        pre_event_blackout=False,
        available_margin_usd=10000.0,
        used_margin_usd=0.0,
        arbiter_size_multiplier=1.0,
    )
    assert result.approved is False
    assert result.rejection_reason == "DAILY_DRAWDOWN_CIRCUIT_BREAKER_ACTIVE"


def test_risk_calculator_fee_gate_rejection(risk_config: RiskConfig) -> None:
    calc = DeterministicRiskCalculator(risk_config)
    result = calc.evaluate_and_size_trade(
        nav_usd=10000.0,
        current_daily_dd_pct=0.0,
        current_max_dd_pct=0.0,
        effective_open_heat_pct=0.0,
        same_direction_open_count=0,
        entry_price=60000.0,
        stop_loss_price=59400.0,
        take_profit_price=60300.0,  # TP 0.5% exceeds barrier 0.3% but is less than 3x roundtrip cost 0.9%
        is_limit_entry=False,
        kyle_slippage_pct=0.0010,
        calibrated_win_prob=0.60,
        venn_abers_uncertainty_width=0.05,
        ood_shrinkage_factor=1.0,
        current_yz_vol=0.015,
        pre_event_blackout=False,
        available_margin_usd=10000.0,
        used_margin_usd=0.0,
        arbiter_size_multiplier=1.0,
    )
    assert result.approved is False
    assert "TP_TOO_SMALL_VS_FEES" in str(result.rejection_reason)


def test_risk_calculator_portfolio_margin_capacity_capping(risk_config: RiskConfig) -> None:
    calc = DeterministicRiskCalculator(risk_config)
    # Available margin is constrained to 1500 USD (uncapped required margin would be ~9090 USD)
    result = calc.evaluate_and_size_trade(
        nav_usd=10000.0,
        current_daily_dd_pct=0.01,
        current_max_dd_pct=0.02,
        effective_open_heat_pct=0.01,
        same_direction_open_count=0,
        entry_price=60000.0,
        stop_loss_price=59400.0,
        take_profit_price=61500.0,
        is_limit_entry=True,
        kyle_slippage_pct=0.0003,
        calibrated_win_prob=0.60,
        venn_abers_uncertainty_width=0.05,
        ood_shrinkage_factor=1.0,
        current_yz_vol=0.015,
        pre_event_blackout=False,
        available_margin_usd=1500.0,
        used_margin_usd=500.0,
        arbiter_size_multiplier=1.0,
    )
    assert result.approved is True
    # Initial margin required = position_notional_usd / recommended_leverage must not exceed 1500 USD
    req_margin = result.position_notional_usd / result.recommended_leverage
    assert req_margin <= 1500.01


def test_risk_calculator_max_margin_utilization_capping(risk_config: RiskConfig) -> None:
    calc = DeterministicRiskCalculator(risk_config)
    # Max margin utilization is 80% (8000 on 10000 NAV). Used margin is 7000.
    # Headroom = 8000 - 7000 = 1000 USD.
    result = calc.evaluate_and_size_trade(
        nav_usd=10000.0,
        current_daily_dd_pct=0.01,
        current_max_dd_pct=0.02,
        effective_open_heat_pct=0.01,
        same_direction_open_count=0,
        entry_price=60000.0,
        stop_loss_price=59400.0,
        take_profit_price=61500.0,
        is_limit_entry=True,
        kyle_slippage_pct=0.0003,
        calibrated_win_prob=0.60,
        venn_abers_uncertainty_width=0.05,
        ood_shrinkage_factor=1.0,
        current_yz_vol=0.015,
        pre_event_blackout=False,
        available_margin_usd=5000.0,
        used_margin_usd=7000.0,
        arbiter_size_multiplier=1.0,
    )
    assert result.approved is True
    req_margin = result.position_notional_usd / result.recommended_leverage
    assert req_margin <= 1000.01


def test_risk_calculator_insufficient_margin_rejection(risk_config: RiskConfig) -> None:
    calc = DeterministicRiskCalculator(risk_config)
    # Zero available margin rejects
    result = calc.evaluate_and_size_trade(
        nav_usd=10000.0,
        current_daily_dd_pct=0.0,
        current_max_dd_pct=0.0,
        effective_open_heat_pct=0.0,
        same_direction_open_count=0,
        entry_price=60000.0,
        stop_loss_price=59400.0,
        take_profit_price=61500.0,
        is_limit_entry=True,
        kyle_slippage_pct=0.0003,
        calibrated_win_prob=0.60,
        venn_abers_uncertainty_width=0.05,
        ood_shrinkage_factor=1.0,
        current_yz_vol=0.015,
        pre_event_blackout=False,
        available_margin_usd=0.0,
        used_margin_usd=0.0,
        arbiter_size_multiplier=1.0,
    )
    assert result.approved is False
    assert result.rejection_reason == "INSUFFICIENT_AVAILABLE_MARGIN"


def test_risk_calculator_arbiter_multiplier_scaling(risk_config: RiskConfig) -> None:
    calc = DeterministicRiskCalculator(risk_config)
    res_full = calc.evaluate_and_size_trade(
        nav_usd=10000.0,
        current_daily_dd_pct=0.01,
        current_max_dd_pct=0.02,
        effective_open_heat_pct=0.0,
        same_direction_open_count=0,
        entry_price=60000.0,
        stop_loss_price=59400.0,
        take_profit_price=61500.0,
        is_limit_entry=True,
        kyle_slippage_pct=0.0003,
        calibrated_win_prob=0.60,
        venn_abers_uncertainty_width=0.05,
        ood_shrinkage_factor=1.0,
        current_yz_vol=0.015,
        pre_event_blackout=False,
        available_margin_usd=10000.0,
        used_margin_usd=0.0,
        arbiter_size_multiplier=1.0,
    )
    res_half = calc.evaluate_and_size_trade(
        nav_usd=10000.0,
        current_daily_dd_pct=0.01,
        current_max_dd_pct=0.02,
        effective_open_heat_pct=0.0,
        same_direction_open_count=0,
        entry_price=60000.0,
        stop_loss_price=59400.0,
        take_profit_price=61500.0,
        is_limit_entry=True,
        kyle_slippage_pct=0.0003,
        calibrated_win_prob=0.60,
        venn_abers_uncertainty_width=0.05,
        ood_shrinkage_factor=1.0,
        current_yz_vol=0.015,
        pre_event_blackout=False,
        available_margin_usd=10000.0,
        used_margin_usd=0.0,
        arbiter_size_multiplier=0.5,
    )
    assert res_full.approved is True
    assert res_half.approved is True
    # Half multiplier halves the risk amount
    assert pytest.approx(res_half.risk_amount_usd, rel=0.05) == res_full.risk_amount_usd * 0.5

    # 0.0 multiplier reduces size to zero and rejects
    res_zero = calc.evaluate_and_size_trade(
        nav_usd=10000.0,
        current_daily_dd_pct=0.01,
        current_max_dd_pct=0.02,
        effective_open_heat_pct=0.0,
        same_direction_open_count=0,
        entry_price=60000.0,
        stop_loss_price=59400.0,
        take_profit_price=61500.0,
        is_limit_entry=True,
        kyle_slippage_pct=0.0003,
        calibrated_win_prob=0.60,
        venn_abers_uncertainty_width=0.05,
        ood_shrinkage_factor=1.0,
        current_yz_vol=0.015,
        pre_event_blackout=False,
        available_margin_usd=10000.0,
        used_margin_usd=0.0,
        arbiter_size_multiplier=0.0,
    )
    assert res_zero.approved is False


def test_risk_calculator_breakeven_position_heat_exclusion(risk_config: RiskConfig) -> None:
    calc = DeterministicRiskCalculator(risk_config)
    positions = [
        ActivePositionSnapshot(
            position_id="pos_be",
            symbol="BTCUSDT",
            side=TradeSide.LONG,
            entry_price=60000.0,
            mark_price=62000.0,
            stop_loss_price=60050.0,
            take_profit_price=65000.0,
            liquidation_price=50000.0,
            quantity=0.1,
            notional_usd=6200.0,
            isolated_margin_usd=1200.0,
            leverage=5,
            unrealized_pnl_usd=200.0,
            unrealized_pnl_pct=0.033,
            unrealized_r_multiple=1.5,
            is_sl_at_breakeven=True,
            effective_at_risk_usd=0.0,
            btc_beta=1.0,
        ),
        ActivePositionSnapshot(
            position_id="pos_at_risk",
            symbol="ETHUSDT",
            side=TradeSide.LONG,
            entry_price=3000.0,
            mark_price=3010.0,
            stop_loss_price=2940.0,
            take_profit_price=3150.0,
            liquidation_price=2500.0,
            quantity=1.0,
            notional_usd=3010.0,
            isolated_margin_usd=600.0,
            leverage=5,
            unrealized_pnl_usd=10.0,
            unrealized_pnl_pct=0.003,
            unrealized_r_multiple=0.16,
            is_sl_at_breakeven=False,
            effective_at_risk_usd=60.0,
            btc_beta=1.2,
        ),
    ]

    effective_heat = calc.compute_effective_open_heat_pct(positions, equity_nav_usd=10000.0)
    assert effective_heat == pytest.approx(60.0 / 10000.0)  # 0.006

    raw_heat = calc.compute_raw_open_heat_pct(positions, equity_nav_usd=10000.0)
    assert raw_heat == pytest.approx(60.0 / 10000.0)


def test_evaluate_with_portfolio_context_integration(risk_config: RiskConfig) -> None:
    calc = DeterministicRiskCalculator(risk_config)
    context = AccountPortfolioContext(
        execution_mode=ExecutionMode.PAPER,
        timestamp_ms=1700000000000,
        wallet_balance_usd=10000.0,
        unrealized_pnl_usd=150.0,
        equity_nav_usd=10150.0,
        used_margin_usd=1200.0,
        available_margin_usd=8950.0,
        margin_utilization_pct=0.118,
        current_daily_pnl_usd=150.0,
        current_daily_dd_pct=0.0,
        remaining_daily_dd_budget_pct=0.04,
        current_max_dd_pct=0.01,
        remaining_max_dd_budget_pct=0.07,
        consecutive_losses_count=0,
        active_positions=[],
        pending_orders=[
            PendingOrderSnapshot(
                order_id="ord_1",
                symbol="ETHUSDT",
                side=TradeSide.LONG,
                order_type=EntryOrderType.LIMIT_PULLBACK_15M,
                limit_price=2950.0,
                notional_usd=2950.0,
                reserved_margin_usd=590.0,
                created_at_ms=1700000000000,
            )
        ],
        gross_exposure_usd=0.0,
        net_directional_delta_usd=0.0,
        net_btc_beta_exposure_usd=0.0,
        raw_open_heat_pct=0.0,
        effective_open_heat_pct=0.0,
        same_direction_active_count=0,
    )

    result = calc.evaluate_with_portfolio_context(
        context=context,
        arbiter_size_multiplier=1.0,
        entry_price=60000.0,
        stop_loss_price=59400.0,
        take_profit_price=61500.0,
        is_limit_entry=True,
        kyle_slippage_pct=0.0003,
        calibrated_win_prob=0.60,
        venn_abers_uncertainty_width=0.05,
        ood_shrinkage_factor=1.0,
        current_yz_vol=0.015,
        pre_event_blackout=False,
    )
    assert result.approved is True
    assert result.position_notional_usd > 0.0


def test_account_portfolio_context_and_verdict_models() -> None:
    verdict = RiskVerdict(
        decision=RiskDecisionType.APPROVE_FULL,
        approved_leverage=5,
        approved_notional_usd=5000.0,
        net_rr=2.1,
        reason=None,
        arbiter_size_multiplier=1.0,
        veto_category=None,
        existing_position_directive=ExistingPositionDirective.HOLD_EXISTING,
        portfolio_risk_rationale="Portfolio margin healthy and beta exposure balanced.",
    )
    assert verdict.arbiter_size_multiplier == 1.0
    assert verdict.veto_category is None
    assert verdict.existing_position_directive == ExistingPositionDirective.HOLD_EXISTING

    veto_verdict = RiskVerdict(
        decision=RiskDecisionType.VETO_TO_SHADOW,
        approved_leverage=1,
        approved_notional_usd=0.0,
        net_rr=1.8,
        reason="Account margin ceiling reached.",
        arbiter_size_multiplier=0.0,
        veto_category=VetoCategory.PORTFOLIO_CAPACITY_VETO,
        existing_position_directive=ExistingPositionDirective.FORBID_DUPLICATE_SYMBOL,
        portfolio_risk_rationale="Available margin exhausted by existing positions.",
    )
    assert veto_verdict.veto_category == VetoCategory.PORTFOLIO_CAPACITY_VETO
    assert (
        veto_verdict.existing_position_directive
        == ExistingPositionDirective.FORBID_DUPLICATE_SYMBOL
    )


# -------------------------------------------------------------------------
# PlaybookValidator Tests
# -------------------------------------------------------------------------


def test_trend_continuation_playbook_success(playbook_config: PlaybookConfig) -> None:
    validator = PlaybookValidator(playbook_config)
    inp = PlaybookPreconditionsInput(
        symbol="BTCUSDT",
        hurst_exp=0.58,
        oi_quadrant=OIPriceQuadrant.LONG_BUILDUP,
        vpin_percentile=75.0,
        p_long_cal=0.62,
        p_short_cal=0.38,
        liq_long_z=0.5,
        liq_short_z=0.2,
        funding_z=0.3,
        vpin_slope_3bars=0.02,
        btc_15m_return_z=0.0,
        btc_lead_beta=1.0,
        lead_lag_catchup_gap_z=0.0,
        is_ood_warning=False,
        pre_event_blackout=False,
    )
    res = validator.verify_playbook_preconditions(PlaybookType.TREND_CONTINUATION_BREAKOUT, inp)
    assert res.is_valid is True
    assert res.allowed_side == TradeSide.LONG
    assert res.playbook == PlaybookType.TREND_CONTINUATION_BREAKOUT


def test_liquidation_exhaustion_playbook_success(playbook_config: PlaybookConfig) -> None:
    validator = PlaybookValidator(playbook_config)
    inp = PlaybookPreconditionsInput(
        symbol="BTCUSDT",
        hurst_exp=0.45,
        oi_quadrant=OIPriceQuadrant.LONG_LIQUIDATION,
        vpin_percentile=80.0,
        p_long_cal=0.50,
        p_short_cal=0.50,
        liq_long_z=2.5,  # Long cascade
        liq_short_z=0.1,
        funding_z=-2.4,
        vpin_slope_3bars=-0.05,  # Exhausting toxicity
        btc_15m_return_z=0.0,
        btc_lead_beta=1.0,
        lead_lag_catchup_gap_z=0.0,
        is_ood_warning=False,
        pre_event_blackout=False,
    )
    res = validator.verify_playbook_preconditions(PlaybookType.LIQUIDATION_EXHAUSTION_REVERSAL, inp)
    assert res.is_valid is True
    assert res.allowed_side == TradeSide.LONG  # Mean-revert bounce


def test_playbook_circuit_breaker_forces_no_trade(playbook_config: PlaybookConfig) -> None:
    validator = PlaybookValidator(playbook_config)
    inp = PlaybookPreconditionsInput(
        symbol="BTCUSDT",
        hurst_exp=0.60,
        oi_quadrant=OIPriceQuadrant.LONG_BUILDUP,
        vpin_percentile=70.0,
        p_long_cal=0.65,
        p_short_cal=0.35,
        liq_long_z=0.0,
        liq_short_z=0.0,
        funding_z=0.0,
        vpin_slope_3bars=0.0,
        btc_15m_return_z=0.0,
        btc_lead_beta=1.0,
        lead_lag_catchup_gap_z=0.0,
        is_ood_warning=True,  # OOD Active
        pre_event_blackout=False,
    )
    res = validator.verify_playbook_preconditions(PlaybookType.TREND_CONTINUATION_BREAKOUT, inp)
    assert res.is_valid is False
    assert res.playbook == PlaybookType.NO_TRADE_REGIME
    assert res.allowed_side == TradeSide.FLAT


# -------------------------------------------------------------------------
# BarrierEvaluator Tests
# -------------------------------------------------------------------------


def test_barrier_evaluator_validated_win(barrier_config: BarrierConfig) -> None:
    evaluator = BarrierEvaluator(barrier_config)
    bars = [
        MicroBar1m(timestamp_ms=1000, open=100.0, high=101.0, low=99.5, close=100.5, volume=10.0),
        MicroBar1m(timestamp_ms=2000, open=100.5, high=103.0, low=100.0, close=102.5, volume=15.0),
    ]
    res = evaluator.evaluate_barriers(
        entry_price=100.0,
        stop_loss_price=98.0,
        take_profit_price=102.0,
        side=TradeSide.LONG,
        bars_1m=bars,
    )
    assert res.outcome == OutcomeCategory.VALIDATED_WIN
    assert res.touch_price == 102.0
    assert res.bars_held == 2
    assert res.post_sl_hit_tp is False


def test_barrier_evaluator_stop_hunted(barrier_config: BarrierConfig) -> None:
    evaluator = BarrierEvaluator(barrier_config)
    bars = [
        # Bar 1: Hit SL at 98.0
        MicroBar1m(timestamp_ms=1000, open=100.0, high=100.2, low=97.5, close=98.5, volume=20.0),
        # Bar 2: Immediate rebound to TP at 103.0
        MicroBar1m(timestamp_ms=2000, open=98.5, high=103.5, low=98.0, close=103.0, volume=35.0),
    ]
    res = evaluator.evaluate_barriers(
        entry_price=100.0,
        stop_loss_price=98.0,
        take_profit_price=102.0,
        side=TradeSide.LONG,
        bars_1m=bars,
    )
    assert res.outcome == OutcomeCategory.STOP_HUNTED_THEN_HIT_TP
    assert res.touch_price == 98.0
    assert res.post_sl_hit_tp is True
