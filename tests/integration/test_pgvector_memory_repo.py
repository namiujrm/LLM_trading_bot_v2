"""Integration Tests: PgvectorMemoryRepository against live Docker TimescaleDB + pgvector."""

from __future__ import annotations

import time
import uuid

import pytest

from trading_bot.adapters.persistence.database import DatabaseManager
from trading_bot.adapters.persistence.pgvector_memory_repo import (
    PgvectorMemoryRepository,
    build_16d_regime_trajectory_vector,
)
from trading_bot.domain.entities.enums import OutcomeCategory
from trading_bot.domain.entities.episodic_case import EpisodicCase
from trading_bot.infrastructure.config_schemas import DatabaseConfig


def get_test_db_config() -> DatabaseConfig:
    return DatabaseConfig(
        host="localhost",
        port=5433,
        db_name="trading_bot_db",
        user="postgres",
        password="postgres",
        max_connections=5,
    )


def test_build_16d_regime_trajectory_vector_bounding() -> None:
    vec = build_16d_regime_trajectory_vector(
        hurst_z=2.5,
        yang_zhang_vol_z=-1.2,
        shannon_entropy_z=0.8,
        vpin_z=10.0,  # Extreme outlier
        obi_mean_z=-0.4,
        taker_cvd_z=1.1,
        oi_directional_score_z=-15.0,  # Extreme outlier
        funding_rate_z=0.01,
        vpin_slope_3bars_z=0.5,
        oi_acceleration_3bars_z=0.3,
        cvd_acceleration_3bars_z=-0.2,
        fracdiff_momentum_slope_z=0.9,
        volatility_expansion_ratio_z=1.4,
        btc_15m_return_z=0.02,
        lead_lag_catchup_gap_z=-0.1,
        session_liquidity_score=1.0,
    )
    assert len(vec) == 16
    for val in vec:
        assert -4.0 <= val <= 4.0


@pytest.mark.asyncio
async def test_pgvector_memory_repo_upsert_query_and_boost() -> None:
    cfg = get_test_db_config()
    db = DatabaseManager(cfg)
    await db.initialize()
    repo = PgvectorMemoryRepository(db)

    try:
        playbook = f"PB_TEST_{int(time.time())}"
        now_ms = int(time.time() * 1000)

        # Base vector
        base_vec = [0.1 * i for i in range(16)]

        # 1. Upsert multiple cases
        case1_id = str(uuid.uuid4())
        case1 = EpisodicCase(
            case_id=case1_id,
            symbol="BTCUSDT",
            timestamp_ms=now_ms,
            vector_16d=base_vec,
            outcome=OutcomeCategory.VALIDATED_WIN,
            confidence_score=0.75,
            lesson_summary="Clean trend continuation above VPIN threshold",
            playbook=playbook,
        )

        case2_id = str(uuid.uuid4())
        # Slight variation in vector
        vec2 = [x + 0.05 for x in base_vec]
        case2 = EpisodicCase(
            case_id=case2_id,
            symbol="BTCUSDT",
            timestamp_ms=now_ms,
            vector_16d=vec2,
            outcome=OutcomeCategory.STOP_HUNTED_THEN_HIT_TP,
            confidence_score=0.60,
            lesson_summary="Wicked out by 0.5 ATR before running to TP",
            playbook=playbook,
        )

        case3_id = str(uuid.uuid4())
        vec3 = [x + 0.5 for x in base_vec]
        case3 = EpisodicCase(
            case_id=case3_id,
            symbol="BTCUSDT",
            timestamp_ms=now_ms,
            vector_16d=vec3,
            outcome=OutcomeCategory.FALSE_POSITIVE_TRAP,
            confidence_score=0.55,
            lesson_summary="Absorption failed at resistance",
            playbook=playbook,
        )

        await repo.upsert_case(case1)
        await repo.upsert_case(case2)
        await repo.upsert_case(case3)

        # 2. Query k-NN cohort
        cohort = await repo.query_knn_cohort(base_vec, playbook, 5)
        assert len(cohort) == 3
        # First neighbor should be case1 (exact match, distance ~ 0)
        assert cohort[0].case_id == case1_id

        # 3. Compute cohort summary
        summary = await repo.compute_cohort_summary(base_vec, playbook, 5)
        assert summary.has_historical_cohort is True
        assert summary.total_neighbors == 3
        assert summary.weighted_win_rate is not None
        assert summary.weighted_win_rate > 0.0
        assert summary.stop_hunted_then_tp_rate is not None
        assert summary.mae_target_percentile_atr is not None

        # 4. Fetch stratified pack
        pack = await repo.fetch_stratified_pack(base_vec, playbook, 5)
        assert pack.cohort_summary.has_historical_cohort is True
        assert len(pack.bucket1_bull_missed) >= 1
        assert pack.bucket1_bull_missed[0].outcome == OutcomeCategory.VALIDATED_WIN
        assert len(pack.bucket2_failure_traps) >= 1
        assert pack.bucket2_failure_traps[0].outcome == OutcomeCategory.FALSE_POSITIVE_TRAP
        assert len(pack.bucket3_stop_hunts) >= 1
        assert pack.bucket3_stop_hunts[0].outcome == OutcomeCategory.STOP_HUNTED_THEN_HIT_TP

        # 5. Boost confidence
        await repo.boost_lesson_confidence(case1_id, 0.08)
        updated_cohort = await repo.query_knn_cohort(base_vec, playbook, 5)
        boosted_case = next(c for c in updated_cohort if c.case_id == case1_id)
        assert boosted_case.confidence_score > 0.80

        # Penalize below 0.25 to trigger archive
        await repo.boost_lesson_confidence(case3_id, -0.40)
        active_cohort = await repo.query_knn_cohort(base_vec, playbook, 5)
        # Archived case should no longer be in active search results
        assert all(c.case_id != case3_id for c in active_cohort)
    finally:
        await db.close()
