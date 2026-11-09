"""Triple-Barrier Labels and Sample Weighting Entities (Zero Defaults).

Pure Python and Pydantic domain models for Causal ML directional labeling.
Adheres strictly to Invariant #9 (Zero Default Class Fields).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class BarrierType(StrEnum):
    """Triple-barrier event classification."""

    UPPER_PROFIT = "upper_profit"
    LOWER_LOSS = "lower_loss"
    TIMEOUT = "timeout"


class LabelSpan(BaseModel):
    """Discrete temporal horizon and financial return span for a sample (Zero Defaults)."""

    sample_id: str
    t_start: int
    t_end: int
    realized_return: float
    volatility: float


class BarrierLabel(BaseModel):
    """Deterministic directional label generated from 1m-resolved forward scanning.

    Adheres strictly to Invariant #9 (Zero Default Class Fields).
    """

    symbol: str
    bar_time: datetime
    barrier_hit: BarrierType
    exit_time: datetime
    bars_held: int
    net_return: float
    sample_weight: float
    y_long: int
    y_short: int
    entry_price: float
    exit_price: float
    realized_return: float
    upper_barrier_price: float
    lower_barrier_price: float
    volatility_gk: float
    roundtrip_cost: float


class SampleWeightRecord(BaseModel):
    """López de Prado sample uniqueness and composite weighting record.

    Adheres strictly to Invariant #9 (Zero Default Class Fields).
    """

    sample_id: str
    concurrency_avg: float
    uniqueness: float
    raw_weight: float
    normalized_weight: float
