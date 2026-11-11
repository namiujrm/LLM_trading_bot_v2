"""Unit tests for Model Registry Domain Entities (Milestone 1).

Enforces strictly ZERO default parameters, ZERO class field defaults, and ZERO silent fallbacks.
Tests:
- ModelRole enumeration.
- ModelArtifactMetadata instantiation and strict required fields.
- ModelPromotionRecord instantiation and strict required fields.
- Rejection of missing required fields (Pydantic validation).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from trading_bot.domain.entities.ml.model_registry_entities import (
    ModelArtifactMetadata,
    ModelPromotionRecord,
    ModelRole,
)


def test_model_role_enum_values() -> None:
    """Verify ModelRole enum values."""
    assert ModelRole.CHAMPION == "CHAMPION"
    assert ModelRole.CHALLENGER == "CHALLENGER"
    assert ModelRole.ARCHIVE == "ARCHIVE"
    assert len(ModelRole) == 3


def test_model_artifact_metadata_instantiation() -> None:
    """Verify ModelArtifactMetadata stores all quantitative and audit fields."""
    now_dt = datetime.now(tz=UTC)
    meta = ModelArtifactMetadata(
        model_id="champion_20261110_001",
        role=ModelRole.CHAMPION,
        feature_names=["f1", "f2", "f3"],
        trained_at_utc=now_dt,
        training_sample_count=5000,
        net_sharpe=2.45,
        brier_score=0.175,
        brier_skill_score=0.155,
        max_drawdown_pct=0.045,
        file_checksum="sha256:abc123def456",
        backtest_metrics=None,
    )

    assert meta.model_id == "champion_20261110_001"
    assert meta.role == ModelRole.CHAMPION
    assert meta.feature_names == ["f1", "f2", "f3"]
    assert meta.trained_at_utc == now_dt
    assert meta.training_sample_count == 5000
    assert meta.net_sharpe == 2.45
    assert meta.brier_score == 0.175
    assert meta.brier_skill_score == 0.155
    assert meta.max_drawdown_pct == 0.045
    assert meta.file_checksum == "sha256:abc123def456"


def test_model_artifact_metadata_missing_field_raises_validation_error() -> None:
    """Verify Invariant #9: zero defaults, missing fields fail immediately."""
    with pytest.raises(ValidationError):
        ModelArtifactMetadata(  # type: ignore[call-arg]
            model_id="test",
            role=ModelRole.CHAMPION,
        )


def test_model_promotion_record_instantiation() -> None:
    """Verify ModelPromotionRecord stores immutable audit record."""
    now_dt = datetime.now(tz=UTC)
    record = ModelPromotionRecord(
        timestamp_utc=now_dt,
        previous_champion_id="champion_v1",
        promoted_challenger_id="challenger_v2",
        promoted_by_user_id="user_admin_5577408890",
        rationale="Passed 5 quantitative tournament gates with Sharpe +0.35 and BSS +0.04",
    )

    assert record.timestamp_utc == now_dt
    assert record.previous_champion_id == "champion_v1"
    assert record.promoted_challenger_id == "challenger_v2"
    assert record.promoted_by_user_id == "user_admin_5577408890"
    assert (
        record.rationale == "Passed 5 quantitative tournament gates with Sharpe +0.35 and BSS +0.04"
    )


def test_model_promotion_record_missing_field_raises_validation_error() -> None:
    """Verify Invariant #9: zero defaults on promotion record."""
    with pytest.raises(ValidationError):
        ModelPromotionRecord(  # type: ignore[call-arg]
            timestamp_utc=datetime.now(tz=UTC),
        )
