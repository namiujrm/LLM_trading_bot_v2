"""Memory Repository Port Interface (Protocol)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from trading_bot.domain.entities.memory.episodic_case import (
    CohortBaseRateSummary,
    EpisodicCase,
    StratifiedMemoryPack,
)


@runtime_checkable
class MemoryRepositoryPort(Protocol):
    """Abstract interface for episodic memory and pgvector k-NN retrieval (Zero Defaults)."""

    async def query_knn_cohort(
        self,
        vector_16d: list[float],
        playbook: str,
        k: int,
    ) -> list[EpisodicCase]:
        """Retrieve the K most similar historical episodes using HNSW cosine distance."""
        ...

    async def compute_cohort_summary(
        self,
        vector_16d: list[float],
        playbook: str,
        k: int,
    ) -> CohortBaseRateSummary:
        """Compute empirical base-rate and 80th-percentile MAE from K=20 nearest neighbors."""
        ...

    async def fetch_stratified_pack(
        self,
        vector_16d: list[float],
        playbook: str,
        k: int,
    ) -> StratifiedMemoryPack:
        """Fetch both cohort base-rate summary and 6 representative stratified lessons."""
        ...

    async def upsert_case(self, case: EpisodicCase) -> None:
        """Persist or update an episodic case record."""
        ...

    async def boost_lesson_confidence(self, case_id: str, delta: float) -> None:
        """Adjust confidence score of an existing lesson after successful defense or validation."""
        ...
