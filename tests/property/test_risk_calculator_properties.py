"""Hypothesis property-based tests for DeterministicRiskCalculator."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from trading_bot.domain.services.risk_calculator import (
    DeterministicRiskCalculator,
    RiskConfig,
)

CONFIG = RiskConfig(
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


@settings(max_examples=100)
@given(
    nav_usd=st.floats(min_value=100.0, max_value=1_000_000.0),
    daily_dd=st.floats(min_value=0.0, max_value=0.20),
    max_dd=st.floats(min_value=0.0, max_value=0.20),
    open_heat=st.floats(min_value=0.0, max_value=0.10),
    same_dir_count=st.integers(min_value=0, max_value=5),
    entry_price=st.floats(min_value=10.0, max_value=100_000.0),
    sl_ratio=st.floats(min_value=0.005, max_value=0.05),
    tp_ratio=st.floats(min_value=0.01, max_value=0.15),
    win_prob=st.floats(min_value=0.30, max_value=0.90),
    uncertainty_width=st.floats(min_value=0.01, max_value=0.30),
    current_yz_vol=st.floats(min_value=0.001, max_value=0.10),
    pre_event_blackout=st.booleans(),
    available_margin=st.floats(min_value=0.0, max_value=1_000_000.0),
    used_margin=st.floats(min_value=0.0, max_value=1_000_000.0),
    arbiter_mult=st.floats(min_value=0.0, max_value=1.5),
)
def test_risk_calculator_invariants(
    nav_usd: float,
    daily_dd: float,
    max_dd: float,
    open_heat: float,
    same_dir_count: int,
    entry_price: float,
    sl_ratio: float,
    tp_ratio: float,
    win_prob: float,
    uncertainty_width: float,
    current_yz_vol: float,
    pre_event_blackout: bool,
    available_margin: float,
    used_margin: float,
    arbiter_mult: float,
) -> None:
    calc = DeterministicRiskCalculator(CONFIG)
    stop_loss = entry_price * (1.0 - sl_ratio)
    take_profit = entry_price * (1.0 + tp_ratio)

    result = calc.evaluate_and_size_trade(
        nav_usd=nav_usd,
        current_daily_dd_pct=daily_dd,
        current_max_dd_pct=max_dd,
        effective_open_heat_pct=open_heat,
        same_direction_open_count=same_dir_count,
        entry_price=entry_price,
        stop_loss_price=stop_loss,
        take_profit_price=take_profit,
        is_limit_entry=True,
        kyle_slippage_pct=0.0002,
        calibrated_win_prob=win_prob,
        venn_abers_uncertainty_width=uncertainty_width,
        ood_shrinkage_factor=1.0,
        current_yz_vol=current_yz_vol,
        pre_event_blackout=pre_event_blackout,
        available_margin_usd=available_margin,
        used_margin_usd=used_margin,
        arbiter_size_multiplier=arbiter_mult,
    )

    # Invariant 1: Recommended leverage must be strictly clamped
    assert CONFIG.min_leverage <= result.recommended_leverage <= CONFIG.max_leverage

    # Invariant 2: Circuit breaker halts must never approve
    if daily_dd >= CONFIG.daily_drawdown_halt_pct or max_dd >= CONFIG.max_drawdown_freeze_pct:
        assert result.approved is False

    # Invariant 3: Blackout window must never approve
    if pre_event_blackout:
        assert result.approved is False

    # Invariant 4: Notional position must never exceed max leverage * nav
    if result.approved:
        assert result.position_notional_usd <= nav_usd * CONFIG.max_leverage + 0.05
        assert result.net_rr >= CONFIG.min_net_rr
        assert result.risk_pct_of_nav <= CONFIG.max_risk_per_trade_pct + 0.0001
        # Invariant 5: Required margin must never exceed available margin
        req_margin = result.position_notional_usd / result.recommended_leverage
        assert req_margin <= available_margin + 0.05
        # Invariant 6: Margin utilization must never exceed max limit
        assert (used_margin + req_margin) / nav_usd <= CONFIG.max_margin_utilization_pct + 0.001
