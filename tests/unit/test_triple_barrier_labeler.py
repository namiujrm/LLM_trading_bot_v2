"""Comprehensive Unit Tests for Triple Barrier Labeler (Zero Defaults, Invariant #9)."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from trading_bot.domain.entities.market.market_state import MicroBar1m
from trading_bot.domain.entities.ml.barrier_labels import BarrierType
from trading_bot.domain.entities.ml.ml_configs import TripleBarrierConfig
from trading_bot.domain.services.ml.triple_barrier_labeler import evaluate_triple_barrier


def _build_config() -> TripleBarrierConfig:
    return TripleBarrierConfig(
        pt_multiplier=1.8,
        sl_multiplier=1.8,
        horizon_bars_1m=360,
        taker_fee_rate=0.0005,
        slippage_rate=0.0001,
        min_tp_to_cost_multiple=1.5,
        dead_zone_weight=0.0,
    )


def _create_bar(
    ts_ms: int,
    open_p: float,
    high_p: float,
    low_p: float,
    close_p: float,
    vol: float,
) -> MicroBar1m:
    return MicroBar1m(
        timestamp_ms=ts_ms,
        open=open_p,
        high=high_p,
        low=low_p,
        close=close_p,
        volume=vol,
    )


def test_upper_profit_barrier_hit() -> None:
    config = _build_config()
    ref_time = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    entry_p = 50000.0
    vol_gk = 0.01  # 1% vol
    # roundtrip_cost = 2 * 0.0005 + 2 * 0.0001 = 0.0012
    # upper_barrier = 50000 * (1 + 1.8 * 0.01 + 0.0012) = 50000 * 1.0192 = 50960.0

    bars: list[MicroBar1m] = []
    for i in range(10):
        # Bar 5 (index 4) crosses upper barrier
        high_val = 51000.0 if i == 4 else 50100.0
        bars.append(_create_bar(1000 * (i + 1), 50000.0, high_val, 49900.0, 50050.0, 10.0))

    label = evaluate_triple_barrier(
        symbol="BTCUSDT",
        bar_time=ref_time,
        entry_price=entry_p,
        vol_gk=vol_gk,
        candles_1m=bars,
        config=config,
    )

    assert label.barrier_hit == BarrierType.UPPER_PROFIT
    assert label.y_long == 1
    assert label.y_short == 0
    assert label.bars_held == 5
    assert label.sample_weight == 1.0
    assert label.exit_price == pytest.approx(50960.0)
    assert label.net_return == pytest.approx(1.8 * vol_gk)


def test_lower_loss_barrier_hit() -> None:
    config = _build_config()
    ref_time = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    entry_p = 50000.0
    vol_gk = 0.01
    # lower_barrier = 50000 * (1 - 1.8 * 0.01 - 0.0012) = 50000 * 0.9808 = 49040.0

    bars: list[MicroBar1m] = []
    for i in range(10):
        # Bar 7 (index 6) crosses lower barrier
        low_val = 49000.0 if i == 6 else 49800.0
        bars.append(_create_bar(1000 * (i + 1), 50000.0, 50100.0, low_val, 49950.0, 10.0))

    label = evaluate_triple_barrier(
        symbol="BTCUSDT",
        bar_time=ref_time,
        entry_price=entry_p,
        vol_gk=vol_gk,
        candles_1m=bars,
        config=config,
    )

    assert label.barrier_hit == BarrierType.LOWER_LOSS
    assert label.y_long == 0
    assert label.y_short == 1
    assert label.bars_held == 7
    assert label.sample_weight == 1.0
    assert label.exit_price == pytest.approx(49040.0)
    assert label.net_return == pytest.approx(1.8 * vol_gk)


def test_intra_bar_tie_break_green_candle_hits_lower() -> None:
    """When both barriers crossed on a green candle, low was visited before high."""
    config = _build_config()
    ref_time = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    entry_p = 50000.0
    vol_gk = 0.01

    bars = [
        _create_bar(1000, 50000.0, 52000.0, 48000.0, 51500.0, 100.0)  # close > open
    ]

    label = evaluate_triple_barrier(
        symbol="BTCUSDT",
        bar_time=ref_time,
        entry_price=entry_p,
        vol_gk=vol_gk,
        candles_1m=bars,
        config=config,
    )

    assert label.barrier_hit == BarrierType.LOWER_LOSS
    assert label.y_long == 0
    assert label.y_short == 1
    assert label.bars_held == 1


def test_intra_bar_tie_break_red_candle_hits_upper() -> None:
    """When both barriers crossed on a red candle, high was visited before low."""
    config = _build_config()
    ref_time = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    entry_p = 50000.0
    vol_gk = 0.01

    bars = [
        _create_bar(1000, 50000.0, 52000.0, 48000.0, 48500.0, 100.0)  # close < open
    ]

    label = evaluate_triple_barrier(
        symbol="BTCUSDT",
        bar_time=ref_time,
        entry_price=entry_p,
        vol_gk=vol_gk,
        candles_1m=bars,
        config=config,
    )

    assert label.barrier_hit == BarrierType.UPPER_PROFIT
    assert label.y_long == 1
    assert label.y_short == 0
    assert label.bars_held == 1


def test_intra_bar_tie_break_doji_candle() -> None:
    """When both barriers crossed on a doji candle (close == open), check distances."""
    config = _build_config()
    ref_time = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    entry_p = 50000.0
    vol_gk = 0.01
    # upper_barrier = 50960.0, lower_barrier = 49040.0

    # Case A: open closer to upper barrier (open = 50600)
    bars_a = [_create_bar(1000, 50600.0, 52000.0, 48000.0, 50600.0, 100.0)]
    label_a = evaluate_triple_barrier("BTCUSDT", ref_time, entry_p, vol_gk, bars_a, config)
    assert label_a.barrier_hit == BarrierType.UPPER_PROFIT

    # Case B: open closer to lower barrier (open = 49400)
    bars_b = [_create_bar(1000, 49400.0, 52000.0, 48000.0, 49400.0, 100.0)]
    label_b = evaluate_triple_barrier("BTCUSDT", ref_time, entry_p, vol_gk, bars_b, config)
    assert label_b.barrier_hit == BarrierType.LOWER_LOSS


def test_timeout_dead_zone() -> None:
    """Price stays within +/- roundtrip cost band: dead zone."""
    config = _build_config()
    ref_time = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    entry_p = 50000.0
    vol_gk = 0.02
    roundtrip_cost = 0.0012

    # 3 bars, ends at 50020 (return = +0.0004 < 0.0012)
    bars = [
        _create_bar(1000, 50000.0, 50030.0, 49970.0, 50010.0, 5.0),
        _create_bar(2000, 50010.0, 50040.0, 49980.0, 50015.0, 5.0),
        _create_bar(3000, 50015.0, 50035.0, 49990.0, 50020.0, 5.0),
    ]

    label = evaluate_triple_barrier(
        symbol="BTCUSDT",
        bar_time=ref_time,
        entry_price=entry_p,
        vol_gk=vol_gk,
        candles_1m=bars,
        config=config,
    )

    assert label.barrier_hit == BarrierType.TIMEOUT
    assert label.y_long == 0
    assert label.y_short == 0
    assert label.sample_weight == 0.0
    assert label.bars_held == 3
    assert label.realized_return == pytest.approx(0.0004)
    assert label.net_return == pytest.approx(0.0004 - roundtrip_cost)


def test_timeout_profitable_long_drift() -> None:
    """Neither barrier touched, but price drifted higher beyond roundtrip cost."""
    config = _build_config()
    ref_time = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    entry_p = 50000.0
    vol_gk = 0.02

    # Exit at 50500 (+1% > 0.0012 cost)
    bars = [
        _create_bar(1000, 50000.0, 50200.0, 49900.0, 50100.0, 5.0),
        _create_bar(2000, 50100.0, 50600.0, 50000.0, 50500.0, 5.0),
    ]

    label = evaluate_triple_barrier(
        symbol="BTCUSDT",
        bar_time=ref_time,
        entry_price=entry_p,
        vol_gk=vol_gk,
        candles_1m=bars,
        config=config,
    )

    assert label.barrier_hit == BarrierType.TIMEOUT
    assert label.y_long == 1
    assert label.y_short == 0
    expected_realized = (50500.0 - 50000.0) / 50000.0
    assert label.realized_return == pytest.approx(expected_realized)
    assert label.sample_weight == pytest.approx(expected_realized / vol_gk)
    assert label.net_return == pytest.approx(expected_realized - 0.0012)


def test_timeout_profitable_short_drift() -> None:
    """Neither barrier touched, but price drifted lower beyond roundtrip cost."""
    config = _build_config()
    ref_time = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    entry_p = 50000.0
    vol_gk = 0.02

    # Exit at 49500 (-1% < -0.0012 cost)
    bars = [
        _create_bar(1000, 50000.0, 50100.0, 49700.0, 49800.0, 5.0),
        _create_bar(2000, 49800.0, 49900.0, 49400.0, 49500.0, 5.0),
    ]

    label = evaluate_triple_barrier(
        symbol="BTCUSDT",
        bar_time=ref_time,
        entry_price=entry_p,
        vol_gk=vol_gk,
        candles_1m=bars,
        config=config,
    )

    assert label.barrier_hit == BarrierType.TIMEOUT
    assert label.y_long == 0
    assert label.y_short == 1
    expected_realized = (49500.0 - 50000.0) / 50000.0
    assert label.realized_return == pytest.approx(expected_realized)
    assert label.sample_weight == pytest.approx(abs(expected_realized) / vol_gk)
    assert label.net_return == pytest.approx(abs(expected_realized) - 0.0012)


def test_fail_fast_validation_errors() -> None:
    config = _build_config()
    ref_time = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    bar = _create_bar(1000, 50000.0, 50100.0, 49900.0, 50000.0, 1.0)

    # Empty candles
    with pytest.raises(ValueError, match="candles_1m must not be empty"):
        evaluate_triple_barrier("BTCUSDT", ref_time, 50000.0, 0.01, [], config)

    # Non-positive entry price
    with pytest.raises(ValueError, match="entry_price must be positive"):
        evaluate_triple_barrier("BTCUSDT", ref_time, -100.0, 0.01, [bar], config)

    # Non-positive vol_gk
    with pytest.raises(ValueError, match="vol_gk must be positive"):
        evaluate_triple_barrier("BTCUSDT", ref_time, 50000.0, 0.0, [bar], config)

    # Non-positive multiplier
    bad_cfg = TripleBarrierConfig(
        pt_multiplier=0.0,
        sl_multiplier=1.8,
        horizon_bars_1m=360,
        taker_fee_rate=0.0005,
        slippage_rate=0.0001,
        min_tp_to_cost_multiple=1.5,
        dead_zone_weight=0.0,
    )
    with pytest.raises(ValueError, match="Multipliers"):
        evaluate_triple_barrier("BTCUSDT", ref_time, 50000.0, 0.01, [bar], bad_cfg)

    # Horizon bars <= 0
    bad_horizon_cfg = TripleBarrierConfig(
        pt_multiplier=1.8,
        sl_multiplier=1.8,
        horizon_bars_1m=0,
        taker_fee_rate=0.0005,
        slippage_rate=0.0001,
        min_tp_to_cost_multiple=1.5,
        dead_zone_weight=0.0,
    )
    with pytest.raises(ValueError, match="horizon_bars_1m must be positive"):
        evaluate_triple_barrier("BTCUSDT", ref_time, 50000.0, 0.01, [bar], bad_horizon_cfg)

    # Negative fee
    bad_fee_cfg = TripleBarrierConfig(
        pt_multiplier=1.8,
        sl_multiplier=1.8,
        horizon_bars_1m=360,
        taker_fee_rate=-0.0005,
        slippage_rate=0.0001,
        min_tp_to_cost_multiple=1.5,
        dead_zone_weight=0.0,
    )
    with pytest.raises(ValueError, match="Fee and slippage rates"):
        evaluate_triple_barrier("BTCUSDT", ref_time, 50000.0, 0.01, [bar], bad_fee_cfg)

    # Negative dead zone weight
    bad_dz_cfg = TripleBarrierConfig(
        pt_multiplier=1.8,
        sl_multiplier=1.8,
        horizon_bars_1m=360,
        taker_fee_rate=0.0005,
        slippage_rate=0.0001,
        min_tp_to_cost_multiple=1.5,
        dead_zone_weight=-0.5,
    )
    with pytest.raises(ValueError, match="dead_zone_weight must be non-negative"):
        evaluate_triple_barrier("BTCUSDT", ref_time, 50000.0, 0.01, [bar], bad_dz_cfg)

    # Extreme volatility causing lower barrier <= 0
    extreme_vol_cfg = TripleBarrierConfig(
        pt_multiplier=1.8,
        sl_multiplier=10.0,
        horizon_bars_1m=360,
        taker_fee_rate=0.0005,
        slippage_rate=0.0001,
        min_tp_to_cost_multiple=1.5,
        dead_zone_weight=0.0,
    )
    with pytest.raises(ValueError, match="Extreme volatility caused lower_barrier <= 0"):
        evaluate_triple_barrier("BTCUSDT", ref_time, 50000.0, 0.20, [bar], extreme_vol_cfg)


def test_timeout_exact_fee_breakeven_exits_produce_dead_zone() -> None:
    """Exact roundtrip fee breakeven exits (exit = entry * (1 +/- c_t)) must produce Dead Zone labels.

    Verifies that positions where realized return equals transaction costs exactly (r_net <= 0)
    are classified into the Dead Zone (y_long=0, y_short=0, sample_weight=0.0) without IEEE 754
    floating-point directional asymmetry.
    """
    config = _build_config()
    ref_time = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    entry_p = 50000.0
    vol_gk = 0.01  # upper_barrier = 50960.0, lower_barrier = 49040.0
    # roundtrip_cost = 2 * 0.0005 + 2 * 0.0001 = 0.0012

    # Case A: Nominal Long Breakeven Exit: exit_price = entry * (1 + c_t) = 50060.0
    exit_p_long_breakeven = entry_p * (1.0 + 0.0012)
    bars_long = [
        _create_bar(1000, entry_p, exit_p_long_breakeven, entry_p, exit_p_long_breakeven, 5.0)
    ]
    label_long = evaluate_triple_barrier(
        symbol="BTCUSDT",
        bar_time=ref_time,
        entry_price=entry_p,
        vol_gk=vol_gk,
        candles_1m=bars_long,
        config=config,
    )

    assert label_long.barrier_hit == BarrierType.TIMEOUT
    assert label_long.y_long == 0
    assert label_long.y_short == 0
    assert label_long.sample_weight == config.dead_zone_weight
    assert label_long.sample_weight == 0.0
    assert label_long.net_return <= 0.0
    assert label_long.net_return == pytest.approx(0.0, abs=1e-12)

    # Case B: Nominal Short Breakeven Exit: exit_price = entry * (1 - c_t) = 49940.0
    exit_p_short_breakeven = entry_p * (1.0 - 0.0012)
    bars_short = [
        _create_bar(1000, entry_p, entry_p, exit_p_short_breakeven, exit_p_short_breakeven, 5.0)
    ]
    label_short = evaluate_triple_barrier(
        symbol="BTCUSDT",
        bar_time=ref_time,
        entry_price=entry_p,
        vol_gk=vol_gk,
        candles_1m=bars_short,
        config=config,
    )

    assert label_short.barrier_hit == BarrierType.TIMEOUT
    assert label_short.y_long == 0
    assert label_short.y_short == 0
    assert label_short.sample_weight == config.dead_zone_weight
    assert label_short.sample_weight == 0.0
    assert label_short.net_return <= 0.0
    assert label_short.net_return == pytest.approx(0.0, abs=1e-12)


def test_timeout_drift_strictly_exceeding_fee_produces_directional_label() -> None:
    """Drift strictly exceeding roundtrip cost (exit = entry * (1 +/- (c_t + 1e-4))) triggers directional labels."""
    config = _build_config()
    ref_time = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    entry_p = 50000.0
    vol_gk = 0.01
    roundtrip = 0.0012
    excess = 1e-4

    # Case A: Long drift strictly exceeding fee: exit_price = entry * (1 + c_t + 1e-4) = 50065.0
    exit_p_long = entry_p * (1.0 + roundtrip + excess)
    bars_long = [_create_bar(1000, entry_p, exit_p_long, entry_p, exit_p_long, 5.0)]
    label_long = evaluate_triple_barrier(
        symbol="BTCUSDT",
        bar_time=ref_time,
        entry_price=entry_p,
        vol_gk=vol_gk,
        candles_1m=bars_long,
        config=config,
    )

    assert label_long.barrier_hit == BarrierType.TIMEOUT
    assert label_long.y_long == 1
    assert label_long.y_short == 0
    assert label_long.net_return == pytest.approx(excess, abs=1e-9)
    assert label_long.sample_weight > 0.0
    assert label_long.realized_return == pytest.approx(roundtrip + excess, abs=1e-9)

    # Case B: Short drift strictly exceeding fee: exit_price = entry * (1 - c_t - 1e-4) = 49935.0
    exit_p_short = entry_p * (1.0 - roundtrip - excess)
    bars_short = [_create_bar(1000, entry_p, entry_p, exit_p_short, exit_p_short, 5.0)]
    label_short = evaluate_triple_barrier(
        symbol="BTCUSDT",
        bar_time=ref_time,
        entry_price=entry_p,
        vol_gk=vol_gk,
        candles_1m=bars_short,
        config=config,
    )

    assert label_short.barrier_hit == BarrierType.TIMEOUT
    assert label_short.y_long == 0
    assert label_short.y_short == 1
    assert label_short.net_return == pytest.approx(excess, abs=1e-9)
    assert label_short.sample_weight > 0.0
    assert label_short.realized_return == pytest.approx(-(roundtrip + excess), abs=1e-9)


def test_timeout_ultra_low_volatility_clamps_soft_weight() -> None:
    """Ultra-low volatility (vol_gk = 1e-8) produces clamped soft weights rather than unbounded blowup."""
    config = _build_config()
    ref_time = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    entry_p = 50000.0
    vol_gk = 1e-8
    roundtrip = 0.0012

    # Timeout drift within [roundtrip, roundtrip + 1.8 * vol_gk]
    # upper_barrier is entry * (1 + 1.8 * 1e-8 + 0.0012) = 50060.0009
    # exit_price is entry * (1 + 0.001200009) = 50060.00045 (neither barrier hit)
    drift_return = roundtrip + 0.5 * 1.8 * vol_gk
    exit_p = entry_p * (1.0 + drift_return)
    bars = [_create_bar(1000, entry_p, exit_p, entry_p * 0.9999, exit_p, 5.0)]

    label = evaluate_triple_barrier(
        symbol="BTCUSDT",
        bar_time=ref_time,
        entry_price=entry_p,
        vol_gk=vol_gk,
        candles_1m=bars,
        config=config,
    )

    assert label.barrier_hit == BarrierType.TIMEOUT
    assert label.y_long == 1
    assert label.y_short == 0
    # Denominator clamped to 1e-6 -> soft_weight = drift_return / 1e-6 = 1200.009
    expected_clamped_weight = drift_return / 1e-6
    assert label.sample_weight == pytest.approx(expected_clamped_weight, rel=1e-5)
    # Strictly verify absence of unbounded blowup (which was 120,000.9)
    assert label.sample_weight < 2000.0
    assert math.isfinite(label.sample_weight)
