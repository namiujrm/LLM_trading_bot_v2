"""Shadow Order and Barrier Evaluation Entities."""

from __future__ import annotations

from pydantic import BaseModel

from trading_bot.domain.entities.common.enums import (
    OutcomeCategory,
    ShadowOrderStatus,
    TradeSide,
    VetoCategory,
)
from trading_bot.domain.entities.execution.trade_blueprint import TradeBlueprint


class ShadowOrderRecord(BaseModel):
    """Lifecycle record for a selectively tracked shadow order or post-SL trade."""

    shadow_id: str
    parent_order_id: str | None
    symbol: str
    side: TradeSide
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    veto_category: VetoCategory | None
    is_post_sl_tracker: bool
    entry_timestamp_ms: int
    max_tracking_duration_ms: int
    mae_price: float
    mfe_price: float
    mae_atr_multiple: float
    mfe_atr_multiple: float
    status: ShadowOrderStatus


class BarrierTouchResult(BaseModel):
    outcome: OutcomeCategory
    touch_timestamp_ms: int | None
    touch_price: float | None
    mae_pct: float
    mfe_pct: float
    bars_held: int
    exit_reason: str
    post_sl_hit_tp: bool


class PostSLAutopsyPath(BaseModel):
    hit_tp_after_sl: bool
    min_price_after_sl: float
    max_price_after_sl: float
    bars_tracked: int


class ShadowOrder(BaseModel):
    order_id: str
    blueprint: TradeBlueprint
    created_at_ms: int
    active: bool
    touch_result: BarrierTouchResult | None
