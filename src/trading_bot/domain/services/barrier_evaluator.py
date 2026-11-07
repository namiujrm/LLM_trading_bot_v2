"""Barrier Evaluator — Pure Triple-Barrier and Post-SL Autopsy Evaluation."""

from __future__ import annotations

from pydantic import BaseModel

from trading_bot.domain.entities.enums import OutcomeCategory, TradeSide
from trading_bot.domain.entities.market_state import MicroBar1m
from trading_bot.domain.entities.shadow_order import BarrierTouchResult


class BarrierConfig(BaseModel):
    """Configuration for barrier horizon and autopsy tracking (Zero Defaults)."""

    max_holding_bars: int
    post_sl_tracking_bars: int


class BarrierEvaluator:
    def __init__(self, config: BarrierConfig) -> None:
        self.cfg: BarrierConfig = config

    def evaluate_barriers(
        self,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        side: TradeSide,
        bars_1m: list[MicroBar1m],
    ) -> BarrierTouchResult:
        if entry_price <= 0.0 or stop_loss_price <= 0.0 or take_profit_price <= 0.0:
            raise ValueError("All barrier price levels must be positive")
        if side not in (TradeSide.LONG, TradeSide.SHORT):
            raise ValueError(f"Barrier evaluation only supports LONG or SHORT, got {side}")

        max_bars = min(len(bars_1m), self.cfg.max_holding_bars)
        max_mae_pct = 0.0
        max_mfe_pct = 0.0

        for i in range(max_bars):
            bar = bars_1m[i]
            if side == TradeSide.LONG:
                adv = max(0.0, (entry_price - bar.low) / entry_price)
                fav = max(0.0, (bar.high - entry_price) / entry_price)
            else:
                adv = max(0.0, (bar.high - entry_price) / entry_price)
                fav = max(0.0, (entry_price - bar.low) / entry_price)

            if adv > max_mae_pct:
                max_mae_pct = adv
            if fav > max_mfe_pct:
                max_mfe_pct = fav

            # Check Stop-Loss hit
            hit_sl = (side == TradeSide.LONG and bar.low <= stop_loss_price) or (
                side == TradeSide.SHORT and bar.high >= stop_loss_price
            )
            if hit_sl:
                # Track post-SL path to check for Stop-Hunt
                post_hit_tp = self._track_post_sl_tp_rebound(
                    bars_1m=bars_1m,
                    sl_index=i,
                    side=side,
                    take_profit_price=take_profit_price,
                )
                outcome = (
                    OutcomeCategory.STOP_HUNTED_THEN_HIT_TP
                    if post_hit_tp
                    else OutcomeCategory.FALSE_POSITIVE_TRAP
                )
                return BarrierTouchResult(
                    outcome=outcome,
                    touch_timestamp_ms=bar.timestamp_ms,
                    touch_price=stop_loss_price,
                    mae_pct=round(max_mae_pct, 5),
                    mfe_pct=round(max_mfe_pct, 5),
                    bars_held=i + 1,
                    exit_reason="STOP_LOSS_TOUCHED",
                    post_sl_hit_tp=post_hit_tp,
                )

            # Check Take-Profit hit
            hit_tp = (side == TradeSide.LONG and bar.high >= take_profit_price) or (
                side == TradeSide.SHORT and bar.low <= take_profit_price
            )
            if hit_tp:
                return BarrierTouchResult(
                    outcome=OutcomeCategory.VALIDATED_WIN,
                    touch_timestamp_ms=bar.timestamp_ms,
                    touch_price=take_profit_price,
                    mae_pct=round(max_mae_pct, 5),
                    mfe_pct=round(max_mfe_pct, 5),
                    bars_held=i + 1,
                    exit_reason="TAKE_PROFIT_TOUCHED",
                    post_sl_hit_tp=False,
                )

        # Timeout reached without hitting TP or SL
        return BarrierTouchResult(
            outcome=OutcomeCategory.STATISTICAL_NOISE_TIMEOUT,
            touch_timestamp_ms=None,
            touch_price=None,
            mae_pct=round(max_mae_pct, 5),
            mfe_pct=round(max_mfe_pct, 5),
            bars_held=max_bars,
            exit_reason="MAX_HOLDING_TIME_EXPIRED",
            post_sl_hit_tp=False,
        )

    def _track_post_sl_tp_rebound(
        self,
        bars_1m: list[MicroBar1m],
        sl_index: int,
        side: TradeSide,
        take_profit_price: float,
    ) -> bool:
        start_idx = sl_index + 1
        end_idx = min(len(bars_1m), start_idx + self.cfg.post_sl_tracking_bars)
        for j in range(start_idx, end_idx):
            bar = bars_1m[j]
            if side == TradeSide.LONG and bar.high >= take_profit_price:
                return True
            if side == TradeSide.SHORT and bar.low <= take_profit_price:
                return True
        return False
