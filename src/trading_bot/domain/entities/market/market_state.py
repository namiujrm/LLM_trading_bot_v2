"""Market State and Microstructure Domain Entities."""

from __future__ import annotations

from pydantic import BaseModel

from trading_bot.domain.entities.common.enums import MarketRegime, OIPriceQuadrant


class MicroBar1m(BaseModel):
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class EnrichedMicroBar1m(BaseModel):
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    taker_buy_volume: float
    quote_volume: float


class MultiTFBar(BaseModel):
    symbol: str
    timeframe: str
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class FeatureSignalItem(BaseModel):
    feature_name: str
    raw_value: float
    z_score: float
    is_statistically_active: bool
    economic_direction: int


class SemanticQuantPacket(BaseModel):
    symbol: str
    timestamp_ms: int
    active_features: list[FeatureSignalItem]
    market_regime: MarketRegime
    oi_quadrant: OIPriceQuadrant
    trajectory_vector_16d: list[float]
