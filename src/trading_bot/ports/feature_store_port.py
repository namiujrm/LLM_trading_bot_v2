"""Feature Store Port Interface (Protocol) — Zero Defaults.

Domain Port defining abstract persistence contracts for 30m quant feature vectors,
episodic trajectory vectors, and assembled SemanticQuantPackets into TimescaleDB and Parquet.
Enforces Invariant #9: zero default parameters, zero hardcoded values.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from trading_bot.domain.entities.market.market_state import SemanticQuantPacket
from trading_bot.domain.entities.quant.quant_features import QuantFeatureVector


@runtime_checkable
class FeatureStorePort(Protocol):
    """Abstract interface for persisting and querying quant feature vectors and packets."""

    async def save_feature_vector(
        self,
        vector: QuantFeatureVector,
    ) -> None:
        """Persist a single 30m quant feature vector into the feature store."""
        ...

    async def save_feature_vectors_batch(
        self,
        vectors: list[QuantFeatureVector],
    ) -> int:
        """Bulk persist a batch of feature vectors, returning count saved."""
        ...

    async def fetch_latest_feature_vector(
        self,
        symbol: str,
    ) -> QuantFeatureVector | None:
        """Fetch the most recent 30m feature vector for a symbol."""
        ...

    async def save_quant_packet(
        self,
        packet: SemanticQuantPacket,
    ) -> None:
        """Persist an assembled SemanticQuantPacket."""
        ...

    async def fetch_latest_quant_packet(
        self,
        symbol: str,
    ) -> SemanticQuantPacket | None:
        """Fetch the most recent SemanticQuantPacket for a symbol."""
        ...

    async def export_parquet_features(
        self,
        symbol: str,
        year: int,
        month: int,
        destination_path: str,
    ) -> str:
        """Export feature records to partitioned Parquet format for ML training."""
        ...
