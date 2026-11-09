"""Quantitative Features and Metrics Domain Entities (Zero Defaults).

Pure Python and Pydantic domain models representing microstructure metrics,
regime states, and the canonical 16-dimensional quantitative feature vector.
"""

from __future__ import annotations

from pydantic import BaseModel

from trading_bot.domain.entities.common.enums import MarketRegime


class MicrostructureMetrics(BaseModel):
    """High-frequency orderbook and trade flow microstructure metrics."""

    vpin: float
    kyle_lambda: float
    estimated_oneway_slippage_pct: float
    ofi_zscore: float
    obi_mean: float
    obi_std: float
    va_amihud: float
    corwin_schultz_spread_bps: float
    mnvr: float


class RegimeMetrics(BaseModel):
    """Multi-timeframe statistical physics, volatility, and information entropy metrics."""

    hurst_exponent: float
    yang_zhang_vol: float
    garman_klass_vol: float
    parkinson_vol: float
    bipower_variation: float
    jump_z_score: float
    has_volatility_jump: bool
    shannon_entropy: float
    permutation_entropy: float
    tsallis_entropy: float
    regime_label: MarketRegime


class QuantFeatureVector(BaseModel):
    """Canonical 16-dimensional normalized quantitative feature vector."""

    symbol: str
    timestamp_ms: int
    vector_16d: list[float]
    is_valid: bool
