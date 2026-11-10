"""Unit tests for ShadowOrderRecord, ShadowOrderStatus, and OutcomeCategory."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trading_bot.domain.entities.common.enums import (
    EntryOrderType,
    OutcomeCategory,
    PlaybookType,
    ShadowOrderStatus,
    TradeSide,
    VetoCategory,
)
from trading_bot.domain.entities.execution.shadow_order import (
    BarrierTouchResult,
    PostSLAutopsyPath,
    ShadowOrder,
    ShadowOrderRecord,
)
from trading_bot.domain.entities.execution.trade_blueprint import (
    MarketThesis,
    TradeBlueprint,
)


def test_outcome_category_capacity_skipped_win() -> None:
    """Verify OutcomeCategory includes CAPACITY_SKIPPED_WIN."""
    assert OutcomeCategory.CAPACITY_SKIPPED_WIN == "CAPACITY_SKIPPED_WIN"
    assert "CAPACITY_SKIPPED_WIN" in [o.value for o in OutcomeCategory]


def test_shadow_order_status_enum_values() -> None:
    """Verify ShadowOrderStatus contains all required lifecycle status values."""
    expected_values = {
        "TRACKING",
        "HIT_TP",
        "HIT_SL",
        "STOP_HUNTED_THEN_TP",
        "TIMEOUT_CLOSED",
    }
    actual_values = {s.value for s in ShadowOrderStatus}
    assert actual_values == expected_values
    assert ShadowOrderStatus.TRACKING == "TRACKING"
    assert ShadowOrderStatus.HIT_TP == "HIT_TP"
    assert ShadowOrderStatus.HIT_SL == "HIT_SL"
    assert ShadowOrderStatus.STOP_HUNTED_THEN_TP == "STOP_HUNTED_THEN_TP"
    assert ShadowOrderStatus.TIMEOUT_CLOSED == "TIMEOUT_CLOSED"


def test_shadow_order_record_creation_valid() -> None:
    """Verify valid instantiation of ShadowOrderRecord with explicit fields."""
    record = ShadowOrderRecord(
        shadow_id="shadow_btc_001",
        parent_order_id=None,
        symbol="BTCUSDT",
        side=TradeSide.LONG,
        entry_price=65000.0,
        stop_loss_price=64000.0,
        take_profit_price=67000.0,
        veto_category=VetoCategory.SETUP_DEFECT,
        is_post_sl_tracker=False,
        entry_timestamp_ms=1710000000000,
        max_tracking_duration_ms=21600000,
        mae_price=64500.0,
        mfe_price=66800.0,
        mae_atr_multiple=0.83,
        mfe_atr_multiple=3.0,
        status=ShadowOrderStatus.TRACKING,
    )

    assert record.shadow_id == "shadow_btc_001"
    assert record.parent_order_id is None
    assert record.symbol == "BTCUSDT"
    assert record.side == TradeSide.LONG
    assert record.entry_price == 65000.0
    assert record.stop_loss_price == 64000.0
    assert record.take_profit_price == 67000.0
    assert record.veto_category == VetoCategory.SETUP_DEFECT
    assert record.is_post_sl_tracker is False
    assert record.entry_timestamp_ms == 1710000000000
    assert record.max_tracking_duration_ms == 21600000
    assert record.mae_price == 64500.0
    assert record.mfe_price == 66800.0
    assert record.mae_atr_multiple == 0.83
    assert record.mfe_atr_multiple == 3.0
    assert record.status == ShadowOrderStatus.TRACKING


def test_shadow_order_record_post_sl_tracker() -> None:
    """Verify ShadowOrderRecord for post-SL tracker trade."""
    record = ShadowOrderRecord(
        shadow_id="shadow_post_sl_eth_002",
        parent_order_id="live_order_9988",
        symbol="ETHUSDT",
        side=TradeSide.SHORT,
        entry_price=3500.0,
        stop_loss_price=3550.0,
        take_profit_price=3400.0,
        veto_category=None,
        is_post_sl_tracker=True,
        entry_timestamp_ms=1710000000000,
        max_tracking_duration_ms=21600000,
        mae_price=3560.0,
        mfe_price=3390.0,
        mae_atr_multiple=0.4,
        mfe_atr_multiple=2.2,
        status=ShadowOrderStatus.STOP_HUNTED_THEN_TP,
    )

    assert record.parent_order_id == "live_order_9988"
    assert record.is_post_sl_tracker is True
    assert record.veto_category is None
    assert record.status == ShadowOrderStatus.STOP_HUNTED_THEN_TP


def test_shadow_order_record_zero_default_discipline() -> None:
    """Verify that omitting any field raises ValidationError (Invariant #9)."""
    with pytest.raises(ValidationError):
        # Missing status and mae/mfe fields
        ShadowOrderRecord(  # pyright: ignore[reportCallIssue]
            shadow_id="shadow_001",
            parent_order_id=None,
            symbol="BTCUSDT",
            side=TradeSide.LONG,
            entry_price=65000.0,
            stop_loss_price=64000.0,
            take_profit_price=67000.0,
            veto_category=None,
            is_post_sl_tracker=False,
            entry_timestamp_ms=1710000000000,
            max_tracking_duration_ms=21600000,
        )


def test_backward_compatibility_existing_shadow_entities() -> None:
    """Verify existing BarrierTouchResult, PostSLAutopsyPath, and ShadowOrder work intact."""
    touch = BarrierTouchResult(
        outcome=OutcomeCategory.VALIDATED_WIN,
        touch_timestamp_ms=1710001000000,
        touch_price=67000.0,
        mae_pct=0.005,
        mfe_pct=0.03,
        bars_held=20,
        exit_reason="TP Hit",
        post_sl_hit_tp=False,
    )
    assert touch.outcome == OutcomeCategory.VALIDATED_WIN

    autopsy = PostSLAutopsyPath(
        hit_tp_after_sl=True,
        min_price_after_sl=63800.0,
        max_price_after_sl=67100.0,
        bars_tracked=35,
    )
    assert autopsy.hit_tp_after_sl is True

    thesis = MarketThesis(
        symbol="BTCUSDT",
        selected_playbook=PlaybookType.TREND_CONTINUATION_BREAKOUT,
        proposed_side=TradeSide.LONG,
        supporting_active_features=["hurst_exp"],
        regime_alignment_score=0.9,
        thesis_summary="Bullish thesis",
    )
    blueprint = TradeBlueprint(
        thesis=thesis,
        order_type=EntryOrderType.LIMIT_PULLBACK_15M,
        entry_price=65000.0,
        stop_loss_price=64000.0,
        take_profit_price=67000.0,
        rationale="Breakout entry",
    )
    shadow = ShadowOrder(
        order_id="shadow_old_001",
        blueprint=blueprint,
        created_at_ms=1710000000000,
        active=True,
        touch_result=None,
    )
    assert shadow.active is True
