"""Shadow Order and Barrier Evaluation Entities."""

from __future__ import annotations

from pydantic import BaseModel

from trading_bot.domain.entities.enums import OutcomeCategory
from trading_bot.domain.entities.trade_blueprint import TradeBlueprint


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
