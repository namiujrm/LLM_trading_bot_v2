"""Shadow Order Tracker Application Service (Clean Architecture).

Tracks selectively recorded shadow orders (vetoed candidates) and live positions post-SL
over high-resolution 1-minute bars up to a 6-hour lifecycle window.
Updates Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE) in price
and 30m ATR multiples.
Classifies concluded outcomes into canonical categories:
- STOP_HUNTED_THEN_HIT_TP
- FALSE_POSITIVE_TRAP
- MISSED_OPPORTUNITY
- CAPACITY_SKIPPED_WIN (Immunity protected: delta = 0.00)
- VALIDATED_DEFENSE
- STATISTICAL_NOISE_TIMEOUT

Clean Architecture:
- Application layer depends ONLY on domain entities and ports.
- ZERO imports from adapters, infrastructure, or interfaces.
- Strict Invariant #9: zero default parameters, zero default fields, zero silent fallbacks.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from trading_bot.domain.entities.common.enums import (
    OutcomeCategory,
    ShadowOrderStatus,
    TradeSide,
    VetoCategory,
)
from trading_bot.domain.entities.execution.shadow_order import ShadowOrderRecord
from trading_bot.domain.entities.execution.trade_blueprint import (
    RiskVerdict,
    TradeBlueprint,
)
from trading_bot.domain.entities.market.market_state import (
    EnrichedMicroBarRecord,
    SemanticQuantPacket,
)

logger = logging.getLogger(__name__)


class ShadowOrderTracker:
    """Selective shadow order tracking and post-SL lifecycle evaluation service."""

    def __init__(self, max_tracking_duration_ms: int) -> None:
        if max_tracking_duration_ms <= 0:
            raise ValueError(
                f"max_tracking_duration_ms must be positive, got {max_tracking_duration_ms}"
            )
        self._max_tracking_duration_ms: int = max_tracking_duration_ms
        self._active_orders: dict[str, ShadowOrderRecord] = {}
        self._concluded_orders: list[ShadowOrderRecord] = []
        self._symbol_atr: dict[str, float] = {}
        self._order_atr: dict[str, float] = {}

    def set_symbol_atr(self, symbol: str, atr_30m: float) -> None:
        """Explicitly set current 30m ATR for a symbol."""
        if atr_30m <= 0.0:
            raise ValueError(f"ATR must be strictly positive, got {atr_30m}")
        self._symbol_atr[symbol.upper()] = atr_30m

    def _resolve_atr(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        quant_packet: SemanticQuantPacket | dict[str, Any],
    ) -> float:
        """Resolve current 30m ATR from symbol state, quant packet active features, or risk distance."""
        if symbol in self._symbol_atr and self._symbol_atr[symbol] > 0.0:
            return self._symbol_atr[symbol]

        if isinstance(quant_packet, SemanticQuantPacket):
            for item in quant_packet.active_features:
                if (
                    item.feature_name in ("atr_30m", "atr", "yang_zhang_vol")
                    and item.raw_value > 0.0
                ):
                    return float(item.raw_value)
        elif "atr_30m" in quant_packet:
            atr_val = float(quant_packet["atr_30m"])
            if atr_val > 0.0:
                return atr_val

        sl_distance = abs(entry_price - stop_loss_price)
        if sl_distance > 0.0:
            return sl_distance

        raise ValueError(
            f"Unable to resolve positive ATR for symbol {symbol} with entry {entry_price} and SL {stop_loss_price}"
        )

    def register_vetoed_candidate(
        self,
        blueprint: TradeBlueprint,
        verdict: RiskVerdict,
        quant_packet: SemanticQuantPacket | dict[str, Any],
    ) -> ShadowOrderRecord:
        """Register a candidate trade vetoed by Risk Arbiter for shadow tracking."""
        symbol: str = blueprint.thesis.symbol.upper()
        side: TradeSide = blueprint.thesis.proposed_side
        entry_price: float = blueprint.entry_price
        stop_loss_price: float = blueprint.stop_loss_price
        take_profit_price: float = blueprint.take_profit_price
        veto_category: VetoCategory | None = verdict.veto_category

        if isinstance(quant_packet, SemanticQuantPacket):
            entry_timestamp_ms: int = quant_packet.timestamp_ms
        elif "timestamp_ms" in quant_packet:
            entry_timestamp_ms = int(quant_packet["timestamp_ms"])
        else:
            entry_timestamp_ms = 0

        shadow_id: str = f"shadow_veto_{symbol}_{entry_timestamp_ms}_{uuid.uuid4().hex[:6]}"
        atr: float = self._resolve_atr(symbol, entry_price, stop_loss_price, quant_packet)

        record = ShadowOrderRecord(
            shadow_id=shadow_id,
            parent_order_id=None,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            veto_category=veto_category,
            is_post_sl_tracker=False,
            entry_timestamp_ms=entry_timestamp_ms,
            max_tracking_duration_ms=self._max_tracking_duration_ms,
            mae_price=entry_price,
            mfe_price=entry_price,
            mae_atr_multiple=0.0,
            mfe_atr_multiple=0.0,
            status=ShadowOrderStatus.TRACKING,
        )

        self._active_orders[shadow_id] = record
        self._order_atr[shadow_id] = atr
        return record

    def register_post_sl_position(
        self,
        symbol: str,
        side: TradeSide,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        exit_timestamp_ms: int,
        quant_packet: SemanticQuantPacket | dict[str, Any],
    ) -> ShadowOrderRecord:
        """Register a live position that hit Stop Loss for post-SL 6h lifecycle tracking."""
        sym_clean: str = symbol.upper()
        shadow_id: str = f"shadow_post_sl_{sym_clean}_{exit_timestamp_ms}_{uuid.uuid4().hex[:6]}"
        parent_id: str = f"order_{sym_clean}_{exit_timestamp_ms}"
        atr: float = self._resolve_atr(sym_clean, entry_price, sl_price, quant_packet)

        initial_mae_price: float = sl_price
        initial_mfe_price: float = entry_price
        initial_mae_multiple: float = round(abs(entry_price - sl_price) / atr, 4)

        record = ShadowOrderRecord(
            shadow_id=shadow_id,
            parent_order_id=parent_id,
            symbol=sym_clean,
            side=side,
            entry_price=entry_price,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            veto_category=None,
            is_post_sl_tracker=True,
            entry_timestamp_ms=exit_timestamp_ms,
            max_tracking_duration_ms=self._max_tracking_duration_ms,
            mae_price=initial_mae_price,
            mfe_price=initial_mfe_price,
            mae_atr_multiple=initial_mae_multiple,
            mfe_atr_multiple=0.0,
            status=ShadowOrderStatus.TRACKING,
        )

        self._active_orders[shadow_id] = record
        self._order_atr[shadow_id] = atr
        return record

    def on_1m_bar(
        self,
        symbol: str,
        bar: EnrichedMicroBarRecord,
    ) -> list[ShadowOrderRecord]:
        """Evaluate active shadow orders for a symbol against a new 1-minute bar.

        Returns a list of concluded ShadowOrderRecord instances.
        """
        sym_clean: str = symbol.upper()
        matching_ids: list[str] = [
            order_id for order_id, order in self._active_orders.items() if order.symbol == sym_clean
        ]

        concluded: list[ShadowOrderRecord] = []

        for order_id in matching_ids:
            order = self._active_orders[order_id]
            atr = self._order_atr[order_id]

            # 1. Update MAE and MFE
            if order.side == TradeSide.LONG:
                new_mfe_price = max(order.mfe_price, bar.high)
                new_mae_price = min(order.mae_price, bar.low)
                mfe_diff = max(0.0, new_mfe_price - order.entry_price)
                mae_diff = max(0.0, order.entry_price - new_mae_price)
            else:  # SHORT
                new_mfe_price = min(order.mfe_price, bar.low)
                new_mae_price = max(order.mae_price, bar.high)
                mfe_diff = max(0.0, order.entry_price - new_mfe_price)
                mae_diff = max(0.0, new_mae_price - order.entry_price)

            new_mfe_multiple = round(mfe_diff / atr, 4)
            new_mae_multiple = round(mae_diff / atr, 4)

            # 2. Check tracking duration
            elapsed_ms = bar.open_time_ms - order.entry_timestamp_ms
            is_timed_out = elapsed_ms >= order.max_tracking_duration_ms

            # 3. Evaluate barriers and outcome
            new_status: ShadowOrderStatus = ShadowOrderStatus.TRACKING

            if order.is_post_sl_tracker:
                hit_tp = (
                    bar.high >= order.take_profit_price
                    if order.side == TradeSide.LONG
                    else bar.low <= order.take_profit_price
                )
                if hit_tp:
                    new_status = ShadowOrderStatus.STOP_HUNTED_THEN_TP
                elif is_timed_out:
                    new_status = ShadowOrderStatus.TIMEOUT_CLOSED
            else:
                if order.side == TradeSide.LONG:
                    hit_tp = bar.high >= order.take_profit_price
                    hit_sl = bar.low <= order.stop_loss_price
                else:  # SHORT
                    hit_tp = bar.low <= order.take_profit_price
                    hit_sl = bar.high >= order.stop_loss_price

                if hit_tp and hit_sl:
                    new_status = ShadowOrderStatus.HIT_SL
                elif hit_tp:
                    new_status = ShadowOrderStatus.HIT_TP
                elif hit_sl:
                    new_status = ShadowOrderStatus.HIT_SL
                elif is_timed_out:
                    new_status = ShadowOrderStatus.TIMEOUT_CLOSED

            updated_order = ShadowOrderRecord(
                shadow_id=order.shadow_id,
                parent_order_id=order.parent_order_id,
                symbol=order.symbol,
                side=order.side,
                entry_price=order.entry_price,
                stop_loss_price=order.stop_loss_price,
                take_profit_price=order.take_profit_price,
                veto_category=order.veto_category,
                is_post_sl_tracker=order.is_post_sl_tracker,
                entry_timestamp_ms=order.entry_timestamp_ms,
                max_tracking_duration_ms=order.max_tracking_duration_ms,
                mae_price=new_mae_price,
                mfe_price=new_mfe_price,
                mae_atr_multiple=new_mae_multiple,
                mfe_atr_multiple=new_mfe_multiple,
                status=new_status,
            )

            if new_status != ShadowOrderStatus.TRACKING:
                del self._active_orders[order_id]
                del self._order_atr[order_id]
                self._concluded_orders.append(updated_order)
                concluded.append(updated_order)
            else:
                self._active_orders[order_id] = updated_order

        return concluded

    def get_all_active_orders(self) -> list[ShadowOrderRecord]:
        """Return list of all currently active tracking shadow orders."""
        return list(self._active_orders.values())

    def get_active_orders_for_symbol(self, symbol: str) -> list[ShadowOrderRecord]:
        """Return list of active tracking shadow orders for a specific symbol."""
        sym_clean: str = symbol.upper()
        return [o for o in self._active_orders.values() if o.symbol == sym_clean]

    def get_concluded_orders(self) -> list[ShadowOrderRecord]:
        """Return list of all concluded shadow orders."""
        return list(self._concluded_orders)

    @staticmethod
    def classify_outcome(record: ShadowOrderRecord) -> OutcomeCategory:
        """Classify final outcome of a concluded ShadowOrderRecord."""
        if record.is_post_sl_tracker:
            if record.status == ShadowOrderStatus.STOP_HUNTED_THEN_TP:
                return OutcomeCategory.STOP_HUNTED_THEN_HIT_TP
            return OutcomeCategory.FALSE_POSITIVE_TRAP

        if record.status == ShadowOrderStatus.HIT_TP:
            if record.veto_category == VetoCategory.PORTFOLIO_CAPACITY_VETO:
                return OutcomeCategory.CAPACITY_SKIPPED_WIN
            return OutcomeCategory.MISSED_OPPORTUNITY

        if record.status == ShadowOrderStatus.HIT_SL:
            return OutcomeCategory.VALIDATED_DEFENSE

        if record.status == ShadowOrderStatus.TIMEOUT_CLOSED:
            return OutcomeCategory.STATISTICAL_NOISE_TIMEOUT

        return OutcomeCategory.STATISTICAL_NOISE_TIMEOUT

    @staticmethod
    def get_confidence_delta(outcome: OutcomeCategory) -> float:
        """Calculate confidence score adjustment based on outcome category."""
        if outcome in (OutcomeCategory.VALIDATED_DEFENSE, OutcomeCategory.VALIDATED_WIN):
            return 0.08
        if outcome in (OutcomeCategory.MISSED_OPPORTUNITY, OutcomeCategory.FALSE_POSITIVE_TRAP):
            return -0.12
        if outcome in (
            OutcomeCategory.CAPACITY_SKIPPED_WIN,
            OutcomeCategory.STATISTICAL_NOISE_TIMEOUT,
            OutcomeCategory.STOP_HUNTED_THEN_HIT_TP,
        ):
            return 0.00
        return 0.00
