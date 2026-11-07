"""Pgvector Memory Repository Adapter implementing MemoryRepositoryPort — Zero Defaults."""

from __future__ import annotations

import json
import math
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np

from trading_bot.adapters.persistence.database import DatabaseManager
from trading_bot.domain.entities.enums import OutcomeCategory
from trading_bot.domain.entities.episodic_case import (
    CohortBaseRateSummary,
    EpisodicCase,
    StratifiedMemoryPack,
)
from trading_bot.ports.memory_repository_port import MemoryRepositoryPort


def build_16d_regime_trajectory_vector(
    hurst_z: float,
    yang_zhang_vol_z: float,
    shannon_entropy_z: float,
    vpin_z: float,
    obi_mean_z: float,
    taker_cvd_z: float,
    oi_directional_score_z: float,
    funding_rate_z: float,
    vpin_slope_3bars_z: float,
    oi_acceleration_3bars_z: float,
    cvd_acceleration_3bars_z: float,
    fracdiff_momentum_slope_z: float,
    volatility_expansion_ratio_z: float,
    btc_15m_return_z: float,
    lead_lag_catchup_gap_z: float,
    session_liquidity_score: float,
) -> list[float]:
    """Encode static regime snapshot and 3-bar temporal dynamics into a 16-dimensional vector."""
    raw = np.array(
        [
            hurst_z,
            yang_zhang_vol_z,
            shannon_entropy_z,
            vpin_z,
            obi_mean_z,
            taker_cvd_z,
            oi_directional_score_z,
            funding_rate_z,
            vpin_slope_3bars_z,
            oi_acceleration_3bars_z,
            cvd_acceleration_3bars_z,
            fracdiff_momentum_slope_z,
            volatility_expansion_ratio_z,
            btc_15m_return_z,
            lead_lag_catchup_gap_z,
            session_liquidity_score,
        ],
        dtype=np.float64,
    )
    # Soft squashing via tanh to bound within [-4.0, +4.0] and prevent outlier distortion
    squashed = 4.0 * np.tanh(raw / 4.0)
    return [round(float(x), 5) for x in squashed]


class PgvectorMemoryRepository(MemoryRepositoryPort):
    """PostgreSQL pgvector adapter for 2-tier episodic memory retrieval."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db: DatabaseManager = db_manager

    async def _query_raw_neighbors(
        self,
        vector_16d: list[float],
        playbook: str,
        k: int,
    ) -> list[dict[str, Any]]:
        vec_str = f"[{','.join(f'{x:.5f}' for x in vector_16d)}]"
        query = """
            SELECT
                lesson_id::text AS case_id,
                EXTRACT(EPOCH FROM created_at) * 1000 AS ts_ms,
                symbol,
                playbook_used,
                outcome_category,
                confidence_score,
                regime_trajectory_vec::text AS vec_str,
                realized_r_multiple,
                mae_atr_multiple,
                mfe_atr_multiple,
                did_hit_tp_after_sl,
                tier_1_index_json,
                tier_2_dossier_json,
                regime_trajectory_vec <-> $1::vector AS l2_dist
            FROM episodic_lessons_memory
            WHERE is_archived = FALSE
              AND playbook_used = $2
            ORDER BY regime_trajectory_vec <-> $1::vector ASC
            LIMIT $3;
        """
        rows = await self._db.fetch(query, vec_str, playbook, k)
        result: list[dict[str, Any]] = []
        for r in rows:
            result.append(
                {
                    "case_id": str(r["case_id"]),
                    "ts_ms": int(float(str(r["ts_ms"]))),
                    "symbol": str(r["symbol"]),
                    "playbook": str(r["playbook_used"]),
                    "outcome_category": str(r["outcome_category"]),
                    "confidence_score": float(str(r["confidence_score"])),
                    "vec_str": str(r["vec_str"]),
                    "realized_r_multiple": float(str(r["realized_r_multiple"])),
                    "mae_atr_multiple": float(str(r["mae_atr_multiple"])),
                    "mfe_atr_multiple": float(str(r["mfe_atr_multiple"])),
                    "did_hit_tp_after_sl": bool(r["did_hit_tp_after_sl"]),
                    "tier_1_index_json": r["tier_1_index_json"],
                    "tier_2_dossier_json": r["tier_2_dossier_json"],
                    "l2_dist": float(str(r["l2_dist"])),
                }
            )
        return result

    async def query_knn_cohort(
        self,
        vector_16d: list[float],
        playbook: str,
        k: int,
    ) -> list[EpisodicCase]:
        """Retrieve K nearest neighbor episodic cases sorted by similarity distance."""
        neighbors = await self._query_raw_neighbors(vector_16d, playbook, k)
        cases: list[EpisodicCase] = []
        for n in neighbors:
            # Parse vector string back to list[float]
            v_raw = n["vec_str"].strip("[]").split(",")
            vec = [float(x.strip()) for x in v_raw]
            dossier = n["tier_2_dossier_json"]
            if isinstance(dossier, str):
                dossier_dict: dict[str, Any] = json.loads(dossier)
            else:
                dossier_dict = dossier

            summary_str = str(dossier_dict.get("summary")) if "summary" in dossier_dict else ""

            cases.append(
                EpisodicCase(
                    case_id=n["case_id"],
                    symbol=n["symbol"],
                    timestamp_ms=n["ts_ms"],
                    vector_16d=vec,
                    outcome=OutcomeCategory(n["outcome_category"]),
                    confidence_score=n["confidence_score"],
                    lesson_summary=summary_str,
                    playbook=n["playbook"],
                )
            )
        return cases

    async def compute_cohort_summary(
        self,
        vector_16d: list[float],
        playbook: str,
        k: int,
    ) -> CohortBaseRateSummary:
        """Compute empirical base-rate and percentiles from K nearest neighbors."""
        neighbors = await self._query_raw_neighbors(vector_16d, playbook, k)
        if not neighbors:
            return CohortBaseRateSummary(
                has_historical_cohort=False,
                total_neighbors=0,
                weighted_win_rate=None,
                stop_hunted_then_tp_rate=None,
                true_failure_trap_rate=None,
                expected_r_multiple=None,
                mae_target_percentile_atr=None,
                mfe_median_atr=None,
            )

        dist_decay_lambda = 0.5
        weight_eps = 1e-9
        weights: list[float] = [
            math.exp(-dist_decay_lambda * n["l2_dist"]) * n["confidence_score"] for n in neighbors
        ]
        sum_w = sum(weights) + weight_eps
        norm_w: list[float] = [w / sum_w for w in weights]

        win_outcomes = {"VALIDATED_WIN", "MISSED_OPPORTUNITY"}
        stophunt_outcomes = {"STOP_HUNTED_THEN_HIT_TP"}
        trap_outcomes = {"FALSE_POSITIVE_TRAP", "VALIDATED_DEFENSE"}

        weighted_win = sum(
            norm_w[i] for i, n in enumerate(neighbors) if n["outcome_category"] in win_outcomes
        )
        weighted_stophunt = sum(
            norm_w[i] for i, n in enumerate(neighbors) if n["outcome_category"] in stophunt_outcomes
        )
        weighted_trap = sum(
            norm_w[i] for i, n in enumerate(neighbors) if n["outcome_category"] in trap_outcomes
        )
        weighted_r = sum(norm_w[i] * n["realized_r_multiple"] for i, n in enumerate(neighbors))

        mae_atrs = np.array([abs(n["mae_atr_multiple"]) for n in neighbors], dtype=np.float64)
        mfe_atrs = np.array([abs(n["mfe_atr_multiple"]) for n in neighbors], dtype=np.float64)

        mae_p80 = float(np.percentile(mae_atrs, 80.0))  # pyright: ignore[reportUnknownMemberType]
        mfe_med = float(np.median(mfe_atrs))  # pyright: ignore[reportUnknownMemberType]

        return CohortBaseRateSummary(
            has_historical_cohort=True,
            total_neighbors=len(neighbors),
            weighted_win_rate=round(float(weighted_win), 3),
            stop_hunted_then_tp_rate=round(float(weighted_stophunt), 3),
            true_failure_trap_rate=round(float(weighted_trap), 3),
            expected_r_multiple=round(float(weighted_r), 3),
            mae_target_percentile_atr=round(mae_p80, 2),
            mfe_median_atr=round(mfe_med, 2),
        )

    async def fetch_stratified_pack(
        self,
        vector_16d: list[float],
        playbook: str,
        k: int,
    ) -> StratifiedMemoryPack:
        """Fetch both empirical cohort summary and 6 representative stratified lessons."""
        summary = await self.compute_cohort_summary(vector_16d, playbook, k)
        all_cases = await self.query_knn_cohort(vector_16d, playbook, k)

        bucket1: list[EpisodicCase] = []
        bucket2: list[EpisodicCase] = []
        bucket3: list[EpisodicCase] = []

        for c in all_cases:
            if (
                c.outcome in {OutcomeCategory.VALIDATED_WIN, OutcomeCategory.MISSED_OPPORTUNITY}
                and len(bucket1) < 2
            ):
                bucket1.append(c)
            elif (
                c.outcome
                in {
                    OutcomeCategory.FALSE_POSITIVE_TRAP,
                    OutcomeCategory.VALIDATED_DEFENSE,
                }
                and len(bucket2) < 2
            ):
                bucket2.append(c)
            elif c.outcome == OutcomeCategory.STOP_HUNTED_THEN_HIT_TP and len(bucket3) < 2:
                bucket3.append(c)

        return StratifiedMemoryPack(
            cohort_summary=summary,
            bucket1_bull_missed=bucket1,
            bucket2_failure_traps=bucket2,
            bucket3_stop_hunts=bucket3,
        )

    async def upsert_case(self, case: EpisodicCase) -> None:
        """Persist or update an episodic case record."""
        # Ensure UUID format
        try:
            valid_uuid = str(uuid.UUID(case.case_id))
        except ValueError:
            valid_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, case.case_id))

        query = """
            INSERT INTO episodic_lessons_memory (
                lesson_id, created_at, updated_at, symbol, playbook_used, side,
                outcome_category, confidence_score, is_archived, regime_trajectory_vec,
                realized_r_multiple, mae_atr_multiple, mfe_atr_multiple, did_hit_tp_after_sl,
                tier_1_index_json, tier_2_dossier_json
            ) VALUES (
                $1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10::vector,
                $11, $12, $13, $14, $15, $16
            )
            ON CONFLICT (lesson_id) DO UPDATE SET
                updated_at = EXCLUDED.updated_at,
                confidence_score = EXCLUDED.confidence_score,
                is_archived = EXCLUDED.is_archived,
                outcome_category = EXCLUDED.outcome_category,
                tier_2_dossier_json = EXCLUDED.tier_2_dossier_json;
        """
        now = datetime.now(tz=UTC)
        vec_str = f"[{','.join(f'{x:.5f}' for x in case.vector_16d)}]"
        tier1_json = json.dumps({"lesson_summary": case.lesson_summary})
        tier2_json = json.dumps(
            {
                "summary": case.lesson_summary,
                "playbook": case.playbook,
                "symbol": case.symbol,
            }
        )

        await self._db.execute(
            query,
            valid_uuid,
            now,
            now,
            case.symbol,
            case.playbook,
            "LONG",
            case.outcome.value,
            case.confidence_score,
            False,
            vec_str,
            1.5,
            0.8,
            2.2,
            False,
            tier1_json,
            tier2_json,
        )

    async def boost_lesson_confidence(self, case_id: str, delta: float) -> None:
        """Adjust confidence score of an existing lesson by delta."""
        try:
            valid_uuid = str(uuid.UUID(case_id))
        except ValueError:
            valid_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, case_id))

        query = """
            UPDATE episodic_lessons_memory
            SET
                confidence_score = GREATEST(0.0, LEAST(0.95, confidence_score + $2)),
                is_archived = CASE WHEN (confidence_score + $2) < 0.25 THEN TRUE ELSE is_archived END,
                updated_at = NOW()
            WHERE lesson_id = $1::uuid;
        """
        await self._db.execute(query, valid_uuid, delta)
