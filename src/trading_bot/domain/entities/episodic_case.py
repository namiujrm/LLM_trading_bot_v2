"""Episodic Memory Case and Cohort Entities."""

from __future__ import annotations

from pydantic import BaseModel

from trading_bot.domain.entities.enums import OutcomeCategory


class CohortBaseRateSummary(BaseModel):
    has_historical_cohort: bool
    total_neighbors: int
    weighted_win_rate: float | None
    stop_hunted_then_tp_rate: float | None
    true_failure_trap_rate: float | None
    expected_r_multiple: float | None
    mae_target_percentile_atr: float | None
    mfe_median_atr: float | None


class EpisodicCase(BaseModel):
    case_id: str
    symbol: str
    timestamp_ms: int
    vector_16d: list[float]
    outcome: OutcomeCategory
    confidence_score: float
    lesson_summary: str
    playbook: str


class StratifiedMemoryPack(BaseModel):
    cohort_summary: CohortBaseRateSummary
    bucket1_bull_missed: list[EpisodicCase]
    bucket2_failure_traps: list[EpisodicCase]
    bucket3_stop_hunts: list[EpisodicCase]
