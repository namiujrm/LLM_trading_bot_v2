"""Quant Engine Port Interface (Protocol) — Zero Defaults.

Domain Port defining abstract contracts for high-performance quantitative feature evaluation,
streaming microstructure processing, multi-timeframe regime compilation, and packet assembly.
Enforces Invariant #9: zero default parameters, zero hardcoded values.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from trading_bot.domain.entities.market.market_state import (
    MicroBar1m,
    MultiTFBar,
    SemanticQuantPacket,
)
from trading_bot.domain.entities.quant.quant_features import (
    MicrostructureMetrics,
    QuantFeatureVector,
    RegimeMetrics,
)


@runtime_checkable
class QuantEnginePort(Protocol):
    """Abstract interface for real-time and batch quantitative feature engine."""

    async def compute_microstructure_metrics(
        self,
        symbol: str,
        bars_1m: list[MicroBar1m],
    ) -> MicrostructureMetrics:
        """Compute real-time order flow and microstructure metrics from 1m micro-bars."""
        ...

    async def compute_regime_state(
        self,
        symbol: str,
        bars_multi_tf: list[MultiTFBar],
    ) -> RegimeMetrics:
        """Compute statistical physics, volatility, and regime state across timeframes."""
        ...

    async def evaluate_quant_packet(
        self,
        symbol: str,
        timestamp_ms: int,
    ) -> SemanticQuantPacket:
        """Assemble complete SemanticQuantPacket for decision cycle."""
        ...

    async def compute_feature_vector_batch(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> list[QuantFeatureVector]:
        """Compute historical batch of 16D feature vectors for ML training."""
        ...
