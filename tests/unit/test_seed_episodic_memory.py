"""Unit tests for cold-start episodic memory seeding script."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.seed_episodic_memory import (
    CONFIDENCE_MAP,
    LESSON_TEMPLATES,
    PLAYBOOK_ORDER,
    TARGET_SYMBOLS,
    extract_16d_vector_from_row,
    load_database_config,
    select_stratified_cases,
)

from trading_bot.domain.entities.common.enums import OutcomeCategory
from trading_bot.infrastructure.config_schemas import DatabaseConfig


def test_load_database_config_valid() -> None:
    env_map = {
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5433",
        "POSTGRES_DB": "test_db",
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "password123",
        "POSTGRES_MAX_CONNECTIONS": "10",
    }
    cfg = load_database_config(env_map)
    assert isinstance(cfg, DatabaseConfig)
    assert cfg.host == "localhost"
    assert cfg.port == 5433
    assert cfg.db_name == "test_db"
    assert cfg.user == "postgres"
    assert cfg.password == "password123"
    assert cfg.max_connections == 10


def test_load_database_config_missing_key_raises_key_error() -> None:
    env_map = {
        "POSTGRES_HOST": "localhost",
        # missing POSTGRES_PORT
        "POSTGRES_DB": "test_db",
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "password123",
        "POSTGRES_MAX_CONNECTIONS": "10",
    }
    with pytest.raises(KeyError):
        load_database_config(env_map)


def test_extract_16d_vector_from_row_bounds() -> None:
    sample_row = {
        "symbol": "BTCUSDT",
        "hurst_exponent_100": 0.65,
        "volatility_gk": 0.025,
        "shannon_entropy_50": 0.75,
        "vpin_toxicity_pct": 35.0,
        "ofi_imbalance_z": 1.2,
        "taker_cvd_z_24": 0.8,
        "oi_directional_score": 1.5,
        "funding_crowded_pos_z": 0.5,
        "funding_crowded_neg_z": 0.0,
        "fracdiff_momentum_z": 1.1,
        "btc_lead_catchup_gap": 0.0,
    }
    vec = extract_16d_vector_from_row(sample_row)
    assert len(vec) == 16
    for val in vec:
        assert -4.0 <= val <= 4.0


def test_lesson_templates_coverage() -> None:
    for playbook in PLAYBOOK_ORDER:
        for outcome in OutcomeCategory:
            if outcome in CONFIDENCE_MAP:
                assert (playbook, outcome) in LESSON_TEMPLATES
                lesson = LESSON_TEMPLATES[(playbook, outcome)]
                assert isinstance(lesson, str)
                assert len(lesson) > 20


def test_select_stratified_cases_from_parquet() -> None:
    pred_path = Path("data/predictions_36m/all_4pairs_predictions_36m.parquet")
    feat_path = Path("data/features_36m/36m_features_labels_all_4pairs.parquet")

    if not pred_path.is_file() or not feat_path.is_file():
        pytest.skip(
            f"Required parquet datasets not found ({pred_path}, {feat_path}). Skipping live parquet selection test."
        )

    cases = select_stratified_cases(str(pred_path), str(feat_path))
    assert len(cases) == 72

    symbols_found = {c.symbol for c in cases}
    assert symbols_found == set(TARGET_SYMBOLS)

    outcomes_found = {c.outcome for c in cases}
    assert OutcomeCategory.VALIDATED_WIN in outcomes_found
    assert OutcomeCategory.FALSE_POSITIVE_TRAP in outcomes_found
    assert OutcomeCategory.STOP_HUNTED_THEN_HIT_TP in outcomes_found
    assert OutcomeCategory.MISSED_OPPORTUNITY in outcomes_found

    # Check confidence score bounds (0.60 to 0.85)
    for c in cases:
        assert 0.60 <= c.confidence_score <= 0.85
        assert len(c.vector_16d) == 16
        for v in c.vector_16d:
            assert -4.0 <= v <= 4.0
