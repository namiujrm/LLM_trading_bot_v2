"""Unit tests for ShadowOrderTracker application service.

Tests selective shadow order tracking on 1m bars, MAE/MFE updates,
post-SL lifecycle evaluation, outcome classifications, and PORTFOLIO_CAPACITY_VETO immunity.
Enforces Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
"""

from __future__ import annotations

import pytest

from trading_bot.application.services.shadow_order_tracker import ShadowOrderTracker
from trading_bot.domain.entities.common.enums import (
    EntryOrderType,
    ExistingPositionDirective,
    MarketRegime,
    OIPriceQuadrant,
    OutcomeCategory,
    PlaybookType,
    RiskDecisionType,
    ShadowOrderStatus,
    TradeSide,
    VetoCategory,
)
from trading_bot.domain.entities.execution.trade_blueprint import (
    MarketThesis,
    RiskVerdict,
    TradeBlueprint,
)
from trading_bot.domain.entities.market.market_state import (
    EnrichedMicroBarRecord,
    FeatureSignalItem,
    SemanticQuantPacket,
)


def create_sample_quant_packet(
    symbol: str,
    timestamp_ms: int,
    atr_30m: float,
) -> SemanticQuantPacket:
    """Helper to create SemanticQuantPacket with explicit features."""
    return SemanticQuantPacket(
        symbol=symbol,
        timestamp_ms=timestamp_ms,
        active_features=[
            FeatureSignalItem(
                feature_name="atr_30m",
                raw_value=atr_30m,
                z_score=0.5,
                is_statistically_active=True,
                economic_direction=1,
            )
        ],
        market_regime=MarketRegime.STRONG_TREND_BULL,
        oi_quadrant=OIPriceQuadrant.LONG_BUILDUP,
        trajectory_vector_16d=[0.1] * 16,
    )


def create_sample_blueprint(
    symbol: str,
    side: TradeSide,
    entry_price: float,
    stop_loss_price: float,
    take_profit_price: float,
) -> TradeBlueprint:
    """Helper to create TradeBlueprint with explicit parameters."""
    thesis = MarketThesis(
        symbol=symbol,
        selected_playbook=PlaybookType.TREND_CONTINUATION_BREAKOUT,
        proposed_side=side,
        supporting_active_features=["hurst_exp"],
        regime_alignment_score=0.85,
        thesis_summary="Bullish continuation setup",
    )
    return TradeBlueprint(
        thesis=thesis,
        order_type=EntryOrderType.LIMIT_PULLBACK_15M,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        rationale="Breakout entry with ATR stop",
    )


def create_sample_verdict(
    veto_category: VetoCategory | None,
) -> RiskVerdict:
    """Helper to create RiskVerdict with explicit parameters."""
    return RiskVerdict(
        decision=RiskDecisionType.VETO_TO_SHADOW,
        approved_leverage=1,
        approved_notional_usd=0.0,
        net_rr=2.0,
        reason="Risk Arbiter veto",
        arbiter_size_multiplier=0.0,
        veto_category=veto_category,
        existing_position_directive=ExistingPositionDirective.HOLD_EXISTING,
        portfolio_risk_rationale="Portfolio risk limit reached or setup defect",
    )


def create_micro_bar(
    symbol: str,
    open_time_ms: int,
    open_p: float,
    high_p: float,
    low_p: float,
    close_p: float,
) -> EnrichedMicroBarRecord:
    """Helper to create EnrichedMicroBarRecord with explicit parameters."""
    return EnrichedMicroBarRecord(
        symbol=symbol,
        open_time_ms=open_time_ms,
        open=open_p,
        high=high_p,
        low=low_p,
        close=close_p,
        volume=10.0,
        quote_volume=10.0 * close_p,
        taker_buy_vol=5.0,
        taker_buy_quote_vol=5.0 * close_p,
        trade_count=100,
        ofi_sum=0.1,
        obi_mean=0.05,
        obi_std=0.01,
        spread_bps_mean=1.5,
        liq_long_usd=0.0,
        liq_short_usd=0.0,
    )


def test_tracker_initialization_invalid_duration() -> None:
    """Verify ValueError is raised if duration is non-positive."""
    with pytest.raises(ValueError, match="max_tracking_duration_ms must be positive"):
        ShadowOrderTracker(max_tracking_duration_ms=0)

    with pytest.raises(ValueError, match="max_tracking_duration_ms must be positive"):
        ShadowOrderTracker(max_tracking_duration_ms=-1000)


def test_vetoed_candidate_hits_tp_setup_defect() -> None:
    """Vetoed candidate (SETUP_DEFECT) hitting TP before SL -> MISSED_OPPORTUNITY (-0.12 delta)."""
    tracker = ShadowOrderTracker(max_tracking_duration_ms=21_600_000)
    packet = create_sample_quant_packet("BTCUSDT", 1_700_000_000_000, 500.0)
    blueprint = create_sample_blueprint("BTCUSDT", TradeSide.LONG, 65000.0, 64000.0, 67000.0)
    verdict = create_sample_verdict(VetoCategory.SETUP_DEFECT)

    record = tracker.register_vetoed_candidate(blueprint, verdict, packet)
    assert record.symbol == "BTCUSDT"
    assert record.status == ShadowOrderStatus.TRACKING
    assert record.is_post_sl_tracker is False
    assert record.veto_category == VetoCategory.SETUP_DEFECT

    # Bar 1: Price goes up to 66000, low 64800
    bar1 = create_micro_bar("BTCUSDT", 1_700_000_060_000, 65000.0, 66000.0, 64800.0, 65900.0)
    concluded = tracker.on_1m_bar("BTCUSDT", bar1)
    assert len(concluded) == 0

    # Bar 2: Price spikes up to 67200, hitting TP!
    bar2 = create_micro_bar("BTCUSDT", 1_700_000_120_000, 65900.0, 67200.0, 65800.0, 67100.0)
    concluded = tracker.on_1m_bar("BTCUSDT", bar2)
    assert len(concluded) == 1

    concluded_order = concluded[0]
    assert concluded_order.status == ShadowOrderStatus.HIT_TP
    assert concluded_order.mfe_price == 67200.0
    assert concluded_order.mae_price == 64800.0
    # MFE diff = 67200 - 65000 = 2200, ATR = 500 -> 2200 / 500 = 4.4
    assert concluded_order.mfe_atr_multiple == 4.4
    # MAE diff = 65000 - 64800 = 200, ATR = 500 -> 200 / 500 = 0.4
    assert concluded_order.mae_atr_multiple == 0.4

    # Outcome & confidence delta
    outcome = tracker.classify_outcome(concluded_order)
    assert outcome == OutcomeCategory.MISSED_OPPORTUNITY
    delta = tracker.get_confidence_delta(outcome)
    assert delta == -0.12


def test_portfolio_capacity_veto_immunity_protection() -> None:
    """Vetoed candidate (PORTFOLIO_CAPACITY_VETO) hitting TP -> CAPACITY_SKIPPED_WIN (0.00 delta, IMMUNE)."""
    tracker = ShadowOrderTracker(max_tracking_duration_ms=21_600_000)
    packet = create_sample_quant_packet("ETHUSDT", 1_700_000_000_000, 50.0)
    blueprint = create_sample_blueprint("ETHUSDT", TradeSide.LONG, 3500.0, 3400.0, 3700.0)
    verdict = create_sample_verdict(VetoCategory.PORTFOLIO_CAPACITY_VETO)

    record = tracker.register_vetoed_candidate(blueprint, verdict, packet)
    assert record.veto_category == VetoCategory.PORTFOLIO_CAPACITY_VETO

    # Bar hitting TP
    bar = create_micro_bar("ETHUSDT", 1_700_000_060_000, 3500.0, 3750.0, 3480.0, 3720.0)
    concluded = tracker.on_1m_bar("ETHUSDT", bar)
    assert len(concluded) == 1

    concluded_order = concluded[0]
    assert concluded_order.status == ShadowOrderStatus.HIT_TP
    outcome = tracker.classify_outcome(concluded_order)
    assert outcome == OutcomeCategory.CAPACITY_SKIPPED_WIN

    # MUST have 0.00 confidence penalty (Immunity Protected!)
    delta = tracker.get_confidence_delta(outcome)
    assert delta == 0.00


def test_vetoed_candidate_hits_sl_validated_defense() -> None:
    """Vetoed candidate hitting SL first -> VALIDATED_DEFENSE (+0.08 delta)."""
    tracker = ShadowOrderTracker(max_tracking_duration_ms=21_600_000)
    packet = create_sample_quant_packet("SOLUSDT", 1_700_000_000_000, 5.0)
    blueprint = create_sample_blueprint("SOLUSDT", TradeSide.LONG, 150.0, 140.0, 170.0)
    verdict = create_sample_verdict(VetoCategory.SETUP_DEFECT)

    tracker.register_vetoed_candidate(blueprint, verdict, packet)

    # Bar hitting SL (low = 139.0)
    bar = create_micro_bar("SOLUSDT", 1_700_000_060_000, 150.0, 152.0, 139.0, 141.0)
    concluded = tracker.on_1m_bar("SOLUSDT", bar)
    assert len(concluded) == 1

    concluded_order = concluded[0]
    assert concluded_order.status == ShadowOrderStatus.HIT_SL
    outcome = tracker.classify_outcome(concluded_order)
    assert outcome == OutcomeCategory.VALIDATED_DEFENSE

    # Arbiter correctly saved capital: +0.08 boost!
    delta = tracker.get_confidence_delta(outcome)
    assert delta == 0.08


def test_vetoed_candidate_timeout_closed() -> None:
    """Candidate neither hitting TP nor SL within 6h -> STATISTICAL_NOISE_TIMEOUT (0.00 delta)."""
    tracker = ShadowOrderTracker(max_tracking_duration_ms=21_600_000)  # 6h = 21,600,000 ms
    start_ts = 1_700_000_000_000
    packet = create_sample_quant_packet("BNBUSDT", start_ts, 10.0)
    blueprint = create_sample_blueprint("BNBUSDT", TradeSide.LONG, 600.0, 580.0, 640.0)
    verdict = create_sample_verdict(VetoCategory.SETUP_DEFECT)

    tracker.register_vetoed_candidate(blueprint, verdict, packet)

    # Bar after 6 hours exactly without hitting TP or SL
    timeout_ts = start_ts + 21_600_000
    bar = create_micro_bar("BNBUSDT", timeout_ts, 605.0, 615.0, 595.0, 610.0)
    concluded = tracker.on_1m_bar("BNBUSDT", bar)
    assert len(concluded) == 1

    concluded_order = concluded[0]
    assert concluded_order.status == ShadowOrderStatus.TIMEOUT_CLOSED
    outcome = tracker.classify_outcome(concluded_order)
    assert outcome == OutcomeCategory.STATISTICAL_NOISE_TIMEOUT
    delta = tracker.get_confidence_delta(outcome)
    assert delta == 0.00


def test_post_sl_position_hits_tp_stop_hunted() -> None:
    """Post-SL position hits TP within 6h -> STOP_HUNTED_THEN_HIT_TP."""
    tracker = ShadowOrderTracker(max_tracking_duration_ms=21_600_000)
    exit_ts = 1_700_000_000_000
    packet = create_sample_quant_packet("BTCUSDT", exit_ts, 600.0)

    # Long trade stopped out at 64000 (entry was 65000, initial TP was 67000)
    record = tracker.register_post_sl_position(
        symbol="BTCUSDT",
        side=TradeSide.LONG,
        entry_price=65000.0,
        sl_price=64000.0,
        tp_price=67000.0,
        exit_timestamp_ms=exit_ts,
        quant_packet=packet,
    )
    assert record.is_post_sl_tracker is True
    assert record.status == ShadowOrderStatus.TRACKING
    assert record.mae_price == 64000.0

    # Intermediate bar: price dips slightly then begins recovery
    bar1 = create_micro_bar("BTCUSDT", exit_ts + 120_000, 64000.0, 64800.0, 63800.0, 64700.0)
    assert len(tracker.on_1m_bar("BTCUSDT", bar1)) == 0

    # Second bar (90 mins later): massive rally hitting original TP at 67100!
    bar2 = create_micro_bar("BTCUSDT", exit_ts + 5_400_000, 66500.0, 67150.0, 66400.0, 67050.0)
    concluded = tracker.on_1m_bar("BTCUSDT", bar2)
    assert len(concluded) == 1

    concluded_order = concluded[0]
    assert concluded_order.status == ShadowOrderStatus.STOP_HUNTED_THEN_TP
    outcome = tracker.classify_outcome(concluded_order)
    assert outcome == OutcomeCategory.STOP_HUNTED_THEN_HIT_TP
    delta = tracker.get_confidence_delta(outcome)
    assert delta == 0.00


def test_post_sl_position_timeout_false_positive_trap() -> None:
    """Post-SL position never recovers to TP within 6h -> FALSE_POSITIVE_TRAP (-0.12 delta)."""
    tracker = ShadowOrderTracker(max_tracking_duration_ms=21_600_000)
    exit_ts = 1_700_000_000_000
    packet = create_sample_quant_packet("ETHUSDT", exit_ts, 40.0)

    tracker.register_post_sl_position(
        symbol="ETHUSDT",
        side=TradeSide.LONG,
        entry_price=3500.0,
        sl_price=3450.0,
        tp_price=3650.0,
        exit_timestamp_ms=exit_ts,
        quant_packet=packet,
    )

    # 6 hours pass, price never reaches 3650.0
    timeout_bar = create_micro_bar("ETHUSDT", exit_ts + 21_600_000, 3400.0, 3430.0, 3380.0, 3390.0)
    concluded = tracker.on_1m_bar("ETHUSDT", timeout_bar)
    assert len(concluded) == 1

    concluded_order = concluded[0]
    assert concluded_order.status == ShadowOrderStatus.TIMEOUT_CLOSED
    outcome = tracker.classify_outcome(concluded_order)
    assert outcome == OutcomeCategory.FALSE_POSITIVE_TRAP
    delta = tracker.get_confidence_delta(outcome)
    assert delta == -0.12


def test_short_trade_tracking_and_both_barriers_touched() -> None:
    """Verify SHORT trade MAE/MFE logic and simultaneous barrier touch resolution."""
    tracker = ShadowOrderTracker(max_tracking_duration_ms=21_600_000)
    packet = create_sample_quant_packet("SOLUSDT", 1_700_000_000_000, 10.0)
    # SHORT: entry 150.0, SL 160.0 (above), TP 130.0 (below)
    blueprint = create_sample_blueprint("SOLUSDT", TradeSide.SHORT, 150.0, 160.0, 130.0)
    verdict = create_sample_verdict(VetoCategory.SETUP_DEFECT)

    tracker.register_vetoed_candidate(blueprint, verdict, packet)

    # Bar 1: Favorable move down to 142.0 (mfe), high 152.0 (mae)
    bar1 = create_micro_bar("SOLUSDT", 1_700_000_060_000, 150.0, 152.0, 142.0, 145.0)
    assert len(tracker.on_1m_bar("SOLUSDT", bar1)) == 0

    # Bar 2: Massive volatility bar touching both TP (128.0) and SL (162.0)
    # Conservative rule must classify as HIT_SL
    bar2 = create_micro_bar("SOLUSDT", 1_700_000_120_000, 145.0, 162.0, 128.0, 148.0)
    concluded = tracker.on_1m_bar("SOLUSDT", bar2)
    assert len(concluded) == 1
    assert concluded[0].status == ShadowOrderStatus.HIT_SL
    assert tracker.classify_outcome(concluded[0]) == OutcomeCategory.VALIDATED_DEFENSE


def test_tracker_accessors() -> None:
    """Verify get_all_active_orders, get_active_orders_for_symbol, and get_concluded_orders."""
    tracker = ShadowOrderTracker(max_tracking_duration_ms=21_600_000)
    packet_btc = create_sample_quant_packet("BTCUSDT", 1_700_000_000_000, 500.0)
    packet_eth = create_sample_quant_packet("ETHUSDT", 1_700_000_000_000, 50.0)

    bp_btc = create_sample_blueprint("BTCUSDT", TradeSide.LONG, 65000.0, 64000.0, 67000.0)
    bp_eth = create_sample_blueprint("ETHUSDT", TradeSide.LONG, 3500.0, 3400.0, 3700.0)
    verdict = create_sample_verdict(VetoCategory.SETUP_DEFECT)

    tracker.register_vetoed_candidate(bp_btc, verdict, packet_btc)
    tracker.register_vetoed_candidate(bp_eth, verdict, packet_eth)

    assert len(tracker.get_all_active_orders()) == 2
    assert len(tracker.get_active_orders_for_symbol("BTCUSDT")) == 1
    assert len(tracker.get_active_orders_for_symbol("ETHUSDT")) == 1
    assert len(tracker.get_active_orders_for_symbol("SOLUSDT")) == 0
    assert len(tracker.get_concluded_orders()) == 0
