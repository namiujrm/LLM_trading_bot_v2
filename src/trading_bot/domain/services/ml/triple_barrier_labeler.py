"""Triple Barrier Labeler — Pure Domain Service for Directional Labeling (Zero Defaults, Invariant #9).

Pure Python domain service implementing Symmetrical Directional Triple-Barriers (Proposal P1):
- Symmetrical barriers scaled by volatility: m_pt * sigma_GK and m_sl * sigma_GK.
- Two-way roundtrip transaction cost deduction: c_t = 2 * f_taker + 2 * slippage.
- Intra-candle deterministic tie-breaking logic based on candle microstructure.
- Timeout resolution with soft return meta-weighting and dead zone filtering.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from trading_bot.domain.entities.market.market_state import MicroBar1m
from trading_bot.domain.entities.ml.barrier_labels import BarrierLabel, BarrierType
from trading_bot.domain.entities.ml.ml_configs import TripleBarrierConfig

# Numerical stability constants for float tolerance and division protection
_FLOAT_TOLERANCE_EPSILON: float = 1e-12
_MIN_VOLATILITY_EPSILON: float = 1e-6


def evaluate_triple_barrier(
    symbol: str,
    bar_time: datetime,
    entry_price: float,
    vol_gk: float,
    candles_1m: list[MicroBar1m],
    config: TripleBarrierConfig,
) -> BarrierLabel:
    """Evaluate symmetrical triple-barriers with fee deduction and 1m forward scan.

    Args:
        symbol: Trading pair symbol (e.g. "BTCUSDT").
        bar_time: Timestamp of decision entry bar.
        entry_price: Explicit execution entry price (> 0).
        vol_gk: Garman-Klass volatility from 30m bar (> 0).
        candles_1m: Sequence of forward 1m candles.
        config: Explicit TripleBarrierConfig instance.

    Returns:
        Deterministic BarrierLabel instance.

    Raises:
        ValueError: If any mathematical or structural invariant is violated.
    """
    if len(candles_1m) == 0:
        raise ValueError("candles_1m must not be empty")
    if not math.isfinite(entry_price) or entry_price <= 0.0:
        raise ValueError(f"entry_price must be positive, got {entry_price}")
    if not math.isfinite(vol_gk) or vol_gk <= 0.0:
        raise ValueError(f"vol_gk must be positive, got {vol_gk}")
    if config.pt_multiplier <= 0.0 or config.sl_multiplier <= 0.0:
        raise ValueError("Multipliers pt_multiplier and sl_multiplier must be positive")
    if config.horizon_bars_1m <= 0:
        raise ValueError(f"horizon_bars_1m must be positive, got {config.horizon_bars_1m}")
    if config.taker_fee_rate < 0.0 or config.slippage_rate < 0.0:
        raise ValueError("Fee and slippage rates must be non-negative")
    if config.dead_zone_weight < 0.0:
        raise ValueError(f"dead_zone_weight must be non-negative, got {config.dead_zone_weight}")

    # Calculate roundtrip transaction cost
    roundtrip_cost = 2.0 * config.taker_fee_rate + 2.0 * config.slippage_rate

    # Calculate upper and lower barrier price levels
    upper_barrier = entry_price * (1.0 + config.pt_multiplier * vol_gk + roundtrip_cost)
    lower_barrier = entry_price * (1.0 - config.sl_multiplier * vol_gk - roundtrip_cost)

    if lower_barrier <= 0.0:
        raise ValueError(f"Extreme volatility caused lower_barrier <= 0: {lower_barrier}")

    max_bars = min(len(candles_1m), config.horizon_bars_1m)

    for i in range(max_bars):
        bar = candles_1m[i]
        hit_upper = bar.high >= upper_barrier
        hit_lower = bar.low <= lower_barrier

        if hit_upper and hit_lower:
            # Deterministic intra-bar tie breaking
            if bar.close > bar.open:
                # Bullish candle: probed low first, then high -> Lower Barrier touched first
                first_hit = BarrierType.LOWER_LOSS
            elif bar.close < bar.open:
                # Bearish candle: probed high first, then low -> Upper Barrier touched first
                first_hit = BarrierType.UPPER_PROFIT
            else:
                # Doji: distance check from open
                dist_up = abs(upper_barrier - bar.open)
                dist_down = abs(bar.open - lower_barrier)
                if dist_up < dist_down:
                    first_hit = BarrierType.UPPER_PROFIT
                else:
                    first_hit = BarrierType.LOWER_LOSS
        elif hit_upper:
            first_hit = BarrierType.UPPER_PROFIT
        elif hit_lower:
            first_hit = BarrierType.LOWER_LOSS
        else:
            first_hit = None

        if first_hit is not None:
            exit_dt = _ms_to_datetime(bar.timestamp_ms, bar_time)
            bars_held = i + 1

            if first_hit == BarrierType.UPPER_PROFIT:
                exit_price = upper_barrier
                realized_return = (upper_barrier - entry_price) / entry_price
                net_return = config.pt_multiplier * vol_gk
                return BarrierLabel(
                    symbol=symbol,
                    bar_time=bar_time,
                    barrier_hit=BarrierType.UPPER_PROFIT,
                    exit_time=exit_dt,
                    bars_held=bars_held,
                    net_return=net_return,
                    sample_weight=1.0,
                    y_long=1,
                    y_short=0,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    realized_return=realized_return,
                    upper_barrier_price=upper_barrier,
                    lower_barrier_price=lower_barrier,
                    volatility_gk=vol_gk,
                    roundtrip_cost=roundtrip_cost,
                )
            else:
                exit_price = lower_barrier
                realized_return = (lower_barrier - entry_price) / entry_price
                net_return = config.sl_multiplier * vol_gk
                return BarrierLabel(
                    symbol=symbol,
                    bar_time=bar_time,
                    barrier_hit=BarrierType.LOWER_LOSS,
                    exit_time=exit_dt,
                    bars_held=bars_held,
                    net_return=net_return,
                    sample_weight=1.0,
                    y_long=0,
                    y_short=1,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    realized_return=realized_return,
                    upper_barrier_price=upper_barrier,
                    lower_barrier_price=lower_barrier,
                    volatility_gk=vol_gk,
                    roundtrip_cost=roundtrip_cost,
                )

    # Neither barrier touched -> Timeout occurs
    exit_bar = candles_1m[max_bars - 1]
    exit_dt = _ms_to_datetime(exit_bar.timestamp_ms, bar_time)
    exit_price = exit_bar.close
    realized_return = (exit_price - entry_price) / entry_price
    soft_weight = abs(realized_return) / max(vol_gk, _MIN_VOLATILITY_EPSILON)

    if abs(realized_return) <= roundtrip_cost + _FLOAT_TOLERANCE_EPSILON:
        # Dead zone: move did not strictly exceed transaction costs in either direction
        y_long = 0
        y_short = 0
        sample_weight = config.dead_zone_weight
        net_return = min(0.0, abs(realized_return) - roundtrip_cost)
    elif realized_return > roundtrip_cost:
        # Profitable long drift (strict r_net > 0)
        y_long = 1
        y_short = 0
        sample_weight = soft_weight
        net_return = realized_return - roundtrip_cost
    else:
        # Profitable short drift (strict r_net > 0)
        y_long = 0
        y_short = 1
        sample_weight = soft_weight
        net_return = -realized_return - roundtrip_cost

    return BarrierLabel(
        symbol=symbol,
        bar_time=bar_time,
        barrier_hit=BarrierType.TIMEOUT,
        exit_time=exit_dt,
        bars_held=max_bars,
        net_return=net_return,
        sample_weight=sample_weight,
        y_long=y_long,
        y_short=y_short,
        entry_price=entry_price,
        exit_price=exit_price,
        realized_return=realized_return,
        upper_barrier_price=upper_barrier,
        lower_barrier_price=lower_barrier,
        volatility_gk=vol_gk,
        roundtrip_cost=roundtrip_cost,
    )


def _ms_to_datetime(timestamp_ms: int, reference_dt: datetime) -> datetime:
    """Convert millisecond timestamp to datetime, matching reference timezone."""
    if timestamp_ms < 0:
        raise ValueError(f"timestamp_ms must be non-negative, got {timestamp_ms}")
    dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC)
    if reference_dt.tzinfo is None:
        return dt.replace(tzinfo=None)
    return dt.astimezone(reference_dt.tzinfo)
