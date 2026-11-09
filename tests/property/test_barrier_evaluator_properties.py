"""Hypothesis property-based tests for BarrierEvaluator."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from trading_bot.domain.entities.common.enums import OutcomeCategory, TradeSide
from trading_bot.domain.entities.market.market_state import MicroBar1m
from trading_bot.domain.services.execution.barrier_evaluator import BarrierConfig, BarrierEvaluator

CONFIG = BarrierConfig(
    max_holding_bars=15,
    post_sl_tracking_bars=15,
)


@settings(max_examples=100)
@given(
    entry_price=st.floats(min_value=10.0, max_value=1000.0),
    sl_dist=st.floats(min_value=0.005, max_value=0.05),
    tp_dist=st.floats(min_value=0.01, max_value=0.10),
    side=st.sampled_from([TradeSide.LONG, TradeSide.SHORT]),
    num_bars=st.integers(min_value=1, max_value=25),
)
def test_barrier_evaluator_invariants(
    entry_price: float,
    sl_dist: float,
    tp_dist: float,
    side: TradeSide,
    num_bars: int,
) -> None:
    evaluator = BarrierEvaluator(CONFIG)
    if side == TradeSide.LONG:
        sl_price = entry_price * (1.0 - sl_dist)
        tp_price = entry_price * (1.0 + tp_dist)
    else:
        sl_price = entry_price * (1.0 + sl_dist)
        tp_price = entry_price * (1.0 - tp_dist)

    # Generate synthetic sequence around entry
    bars: list[MicroBar1m] = []
    curr_price = entry_price
    for i in range(num_bars):
        high = curr_price * 1.002
        low = curr_price * 0.998
        bars.append(
            MicroBar1m(
                timestamp_ms=1000 * (i + 1),
                open=curr_price,
                high=high,
                low=low,
                close=curr_price,
                volume=10.0,
            )
        )

    res = evaluator.evaluate_barriers(
        entry_price=entry_price,
        stop_loss_price=sl_price,
        take_profit_price=tp_price,
        side=side,
        bars_1m=bars,
    )

    # Invariant 1: MAE and MFE are non-negative
    assert res.mae_pct >= 0.0
    assert res.mfe_pct >= 0.0

    # Invariant 2: Bars held never exceeds max_holding_bars or bar list length
    assert 1 <= res.bars_held <= min(num_bars, CONFIG.max_holding_bars)

    # Invariant 3: Outcome is a valid OutcomeCategory
    assert isinstance(res.outcome, OutcomeCategory)
