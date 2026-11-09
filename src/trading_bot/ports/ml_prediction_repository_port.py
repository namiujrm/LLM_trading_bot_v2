"""Machine Learning Prediction Repository Port Interface (Protocol) — Zero Defaults.

Domain Port defining persistence contracts for Stage 4 ML prediction packets,
calibrated barrier probabilities, Venn-Abers uncertainty intervals, OOD metrics,
and native C++ TreeSHAP explanations.

Clean Architecture: 100% pure Python standard typing and Domain Entities.
Zero heavy computation or database driver dependencies (NO asyncpg, NO polars, NO numpy).
Invariant #9: zero default parameters, zero class field defaults.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from trading_bot.domain.entities.ml.ml_predictions import UnifiedMLPredictionPacket


@runtime_checkable
class MLPredictionRepositoryPort(Protocol):
    """Abstract persistence contract for Stage 4 ML prediction packets (Zero Defaults)."""

    async def save_prediction_packet(
        self,
        packet: UnifiedMLPredictionPacket,
    ) -> None:
        """Persist a single ML prediction packet into the feature store."""
        ...

    async def save_prediction_packets_batch(
        self,
        packets: Sequence[UnifiedMLPredictionPacket],
    ) -> int:
        """Bulk upsert a batch of ML prediction packets into the feature store.

        Args:
            packets: Chronological sequence of UnifiedMLPredictionPacket instances.

        Returns:
            Count of successfully upserted prediction records.
        """
        ...

    async def get_prediction_packet(
        self,
        symbol: str,
    ) -> UnifiedMLPredictionPacket | None:
        """Fetch the most recent ML prediction packet for a given market symbol."""
        ...

    async def get_prediction_packets_range(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Sequence[UnifiedMLPredictionPacket]:
        """Fetch a chronological sequence of ML prediction packets within [start_time, end_time]."""
        ...
