"""Domain entities for ML Model Registry & Tournament Lifecycle (Pure Python/Pydantic).

Enforces Invariant #9:
- Zero default parameters in functions and methods
- Zero default values in class / Pydantic fields
- Zero silent fallbacks
- 100% pure domain: no I/O, no network, no external computation dependencies.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class ModelRole(StrEnum):
    """Lifecycle role of a machine learning model artifact."""

    CHAMPION = "CHAMPION"
    CHALLENGER = "CHALLENGER"
    ARCHIVE = "ARCHIVE"


class ModelArtifactMetadata(BaseModel):
    """Metadata schema for persistent ML model artifacts (Zero Defaults)."""

    model_id: str
    role: ModelRole
    feature_names: list[str]
    trained_at_utc: datetime
    training_sample_count: int
    net_sharpe: float
    brier_score: float
    brier_skill_score: float
    max_drawdown_pct: float
    file_checksum: str


class ModelPromotionRecord(BaseModel):
    """Immutable audit record for champion-challenger promotion (Zero Defaults)."""

    timestamp_utc: datetime
    previous_champion_id: str
    promoted_challenger_id: str
    promoted_by_user_id: str
    rationale: str
