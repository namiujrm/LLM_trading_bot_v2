"""Unit tests for ModelTournamentService (Milestone 7.3, Invariant #9)."""

from __future__ import annotations

from trading_bot.application.services.model_tournament_service import (
    ModelTournamentConfig,
    ModelTournamentService,
    TournamentEvaluationInput,
)
from trading_bot.domain.entities.quant.symbolic_alpha import TournamentVerdict


def _make_config() -> ModelTournamentConfig:
    """Helper to instantiate tournament config with institutional thresholds."""
    return ModelTournamentConfig(
        sharpe_delta_min=0.15,
        venn_abers_spread_max=0.12,
        max_sharpe_degradation_ratio=0.35,
        min_passed_gates_for_shadow=3,
    )


def _make_tournament_input(
    challenger_net_sharpe: float = 1.70,
    champion_net_sharpe: float = 1.50,
    challenger_max_drawdown_pct: float = 0.10,
    champion_max_drawdown_pct: float = 0.12,
    challenger_brier_skill_score: float = 0.11,
    champion_brier_skill_score: float = 0.08,
    challenger_venn_abers_spread: float = 0.08,
    champion_venn_abers_spread: float = 0.10,
    challenger_in_sample_sharpe: float = 2.00,
) -> TournamentEvaluationInput:
    """Helper to create tournament input with explicit parameters (Zero Defaults)."""
    return TournamentEvaluationInput(
        champion_model_id="champion_lightgbm_v1",
        challenger_model_id="challenger_lightgbm_v2",
        champion_feature_count=40,
        challenger_feature_count=42,
        champion_net_sharpe=champion_net_sharpe,
        challenger_net_sharpe=challenger_net_sharpe,
        champion_max_drawdown_pct=champion_max_drawdown_pct,
        challenger_max_drawdown_pct=challenger_max_drawdown_pct,
        champion_calmar_ratio=2.5,
        challenger_calmar_ratio=3.1,
        champion_win_rate=0.55,
        challenger_win_rate=0.58,
        champion_profit_factor=1.65,
        challenger_profit_factor=1.85,
        champion_expectancy_r=0.25,
        challenger_expectancy_r=0.32,
        champion_brier_score=0.18,
        challenger_brier_score=0.15,
        champion_brier_skill_score=champion_brier_skill_score,
        challenger_brier_skill_score=challenger_brier_skill_score,
        champion_log_loss=0.45,
        challenger_log_loss=0.41,
        champion_venn_abers_spread=champion_venn_abers_spread,
        challenger_venn_abers_spread=challenger_venn_abers_spread,
        challenger_in_sample_sharpe=challenger_in_sample_sharpe,
        created_at_utc="2026-11-10T12:00:00Z",
    )


def test_immediate_promotion_when_all_5_gates_pass() -> None:
    """All 5 mandatory quantitative approval gates satisfied -> Immediate Promotion."""
    service = ModelTournamentService(_make_config())
    inp = _make_tournament_input(
        challenger_net_sharpe=1.70,  # Delta = +0.20 >= +0.15 (Gate 1 PASS)
        champion_net_sharpe=1.50,
        challenger_max_drawdown_pct=0.10,  # 0.10 <= 0.12 (Gate 2 PASS)
        champion_max_drawdown_pct=0.12,
        challenger_brier_skill_score=0.11,  # 0.11 > 0.08 (Gate 3 PASS)
        champion_brier_skill_score=0.08,
        challenger_venn_abers_spread=0.08,  # 0.08 <= 0.12 (Gate 4 PASS)
        challenger_in_sample_sharpe=2.00,  # Degradation = (2.0 - 1.7) / 2.0 = 15% < 35% (Gate 5 PASS)
    )

    report = service.run_tournament(inp)

    assert report.is_promotable is True
    assert report.recommended_verdict == TournamentVerdict.PROMOTE_CHALLENGER_IMMEDIATELY
    assert report.approval_checklist == {
        "gate_1_sharpe_outperformance": True,
        "gate_2_max_drawdown_preservation": True,
        "gate_3_brier_skill_score": True,
        "gate_4_venn_abers_calibration": True,
        "gate_5_anti_overfitting": True,
    }
    assert "All 5 mandatory quantitative approval gates passed" in report.verdict_rationale


def test_rejection_when_drawdown_worsens_gate_2() -> None:
    """Gate 2 (Capital risk preservation) failure immediately forces KEEP_CHAMPION."""
    service = ModelTournamentService(_make_config())
    inp = _make_tournament_input(
        challenger_net_sharpe=1.90,  # +0.40 Sharpe outperformance!
        champion_net_sharpe=1.50,
        challenger_max_drawdown_pct=0.15,  # 0.15 > 0.12 (Gate 2 FAILS!)
        champion_max_drawdown_pct=0.12,
        challenger_brier_skill_score=0.15,
        champion_brier_skill_score=0.08,
        challenger_venn_abers_spread=0.05,
        challenger_in_sample_sharpe=2.20,
    )

    report = service.run_tournament(inp)

    assert report.is_promotable is False
    assert report.recommended_verdict == TournamentVerdict.KEEP_CHAMPION
    assert report.approval_checklist["gate_2_max_drawdown_preservation"] is False
    assert report.approval_checklist["gate_1_sharpe_outperformance"] is True
    assert "Gate 2 (Capital Risk Preservation failure)" in report.verdict_rationale


def test_rejection_when_overfitting_exceeds_threshold_gate_5() -> None:
    """Gate 5 (Anti-overfitting discipline) failure forces KEEP_CHAMPION."""
    service = ModelTournamentService(_make_config())
    # IS Sharpe = 3.0, OOS Sharpe = 1.70 -> degradation = (3.0 - 1.7) / 3.0 = 43.3% >= 35%
    inp = _make_tournament_input(
        challenger_net_sharpe=1.70,
        champion_net_sharpe=1.50,
        challenger_max_drawdown_pct=0.10,
        champion_max_drawdown_pct=0.12,
        challenger_brier_skill_score=0.12,
        champion_brier_skill_score=0.08,
        challenger_venn_abers_spread=0.07,
        challenger_in_sample_sharpe=3.00,  # Severe overfitting!
    )

    report = service.run_tournament(inp)

    assert report.is_promotable is False
    assert report.recommended_verdict == TournamentVerdict.KEEP_CHAMPION
    assert report.approval_checklist["gate_5_anti_overfitting"] is False
    assert "Gate 5 (Anti-overfitting discipline failure)" in report.verdict_rationale


def test_shadow_battle_recommendation_when_borderline_3_or_4_gates_pass() -> None:
    """When Gates 2 and 5 pass, but Sharpe delta is borderline (+0.10 < +0.15), recommend 48h Shadow Battle."""
    service = ModelTournamentService(_make_config())
    inp = _make_tournament_input(
        challenger_net_sharpe=1.58,  # Delta +0.08 < +0.15 (Gate 1 FAILS)
        champion_net_sharpe=1.50,
        challenger_max_drawdown_pct=0.10,  # Gate 2 PASS
        champion_max_drawdown_pct=0.12,
        challenger_brier_skill_score=0.10,  # Gate 3 PASS
        champion_brier_skill_score=0.08,
        challenger_venn_abers_spread=0.09,  # Gate 4 PASS
        challenger_in_sample_sharpe=1.80,  # Gate 5 PASS ((1.8 - 1.58)/1.8 = 12.2% < 35%)
    )

    report = service.run_tournament(inp)

    assert report.is_promotable is False
    assert report.recommended_verdict == TournamentVerdict.RUN_48H_SHADOW_BATTLE
    assert report.approval_checklist["gate_1_sharpe_outperformance"] is False
    assert report.approval_checklist["gate_2_max_drawdown_preservation"] is True
    assert report.approval_checklist["gate_5_anti_overfitting"] is True
    assert "Borderline performance: 4 of 5 approval gates passed" in report.verdict_rationale


def test_shadow_battle_recommendation_with_3_gates_passed() -> None:
    """When Gates 2 and 5 pass, plus Gate 1, but BSS and calibration fail (3 gates passed)."""
    service = ModelTournamentService(_make_config())
    inp = _make_tournament_input(
        challenger_net_sharpe=1.70,  # Gate 1 PASS (Delta +0.20)
        champion_net_sharpe=1.50,
        challenger_max_drawdown_pct=0.10,  # Gate 2 PASS
        champion_max_drawdown_pct=0.12,
        challenger_brier_skill_score=0.06,  # Gate 3 FAILS (0.06 < 0.08)
        champion_brier_skill_score=0.08,
        challenger_venn_abers_spread=0.15,  # Gate 4 FAILS (0.15 > 0.12)
        challenger_in_sample_sharpe=2.00,  # Gate 5 PASS
    )

    report = service.run_tournament(inp)

    assert report.is_promotable is False
    assert report.recommended_verdict == TournamentVerdict.RUN_48H_SHADOW_BATTLE
    assert sum(1 for p in report.approval_checklist.values() if p) == 3


def test_keep_champion_when_fewer_than_3_gates_pass() -> None:
    """When only 2 gates pass (even if Gate 2 and 5 pass), reject challenger with KEEP_CHAMPION."""
    service = ModelTournamentService(_make_config())
    inp = _make_tournament_input(
        challenger_net_sharpe=1.52,  # Gate 1 FAILS (Delta +0.02)
        champion_net_sharpe=1.50,
        challenger_max_drawdown_pct=0.11,  # Gate 2 PASS
        champion_max_drawdown_pct=0.12,
        challenger_brier_skill_score=0.05,  # Gate 3 FAILS (0.05 < 0.08)
        champion_brier_skill_score=0.08,
        challenger_venn_abers_spread=0.16,  # Gate 4 FAILS (0.16 > 0.12)
        challenger_in_sample_sharpe=1.60,  # Gate 5 PASS ((1.60 - 1.52)/1.60 = 5% < 35%)
    )

    report = service.run_tournament(inp)

    assert report.is_promotable is False
    assert report.recommended_verdict == TournamentVerdict.KEEP_CHAMPION
    assert sum(1 for p in report.approval_checklist.values() if p) == 2
    assert "Challenger rejected: Only 2 of 5 approval gates passed" in report.verdict_rationale


def test_boundary_exact_values() -> None:
    """Test exact equality boundaries: Sharpe delta exactly 0.15, DD exactly equal, spread exactly 0.12."""
    service = ModelTournamentService(_make_config())
    inp = _make_tournament_input(
        challenger_net_sharpe=1.65,  # Exactly 1.50 + 0.15
        champion_net_sharpe=1.50,
        challenger_max_drawdown_pct=0.12,  # Exactly 0.12 == 0.12
        champion_max_drawdown_pct=0.12,
        challenger_brier_skill_score=0.10,
        champion_brier_skill_score=0.08,
        challenger_venn_abers_spread=0.12,  # Exactly 0.12 == 0.12
        challenger_in_sample_sharpe=2.00,  # Degradation = 0.35 / 2.0 = 17.5% < 35%
    )

    report = service.run_tournament(inp)

    assert report.is_promotable is True
    assert report.recommended_verdict == TournamentVerdict.PROMOTE_CHALLENGER_IMMEDIATELY
    assert all(report.approval_checklist.values())


def test_non_positive_in_sample_sharpe_handling() -> None:
    """Verify robust degradation calculation when in-sample Sharpe is zero or negative."""
    service = ModelTournamentService(_make_config())
    # IS Sharpe = 0.0, OOS Sharpe = 0.20 -> no overfitting degradation
    inp_pos = _make_tournament_input(
        challenger_net_sharpe=0.20,
        champion_net_sharpe=0.02,
        challenger_max_drawdown_pct=0.10,
        champion_max_drawdown_pct=0.12,
        challenger_brier_skill_score=0.10,
        champion_brier_skill_score=0.08,
        challenger_venn_abers_spread=0.09,
        challenger_in_sample_sharpe=0.0,
    )
    report_pos = service.run_tournament(inp_pos)
    assert report_pos.approval_checklist["gate_5_anti_overfitting"] is True

    # IS Sharpe = 0.0, OOS Sharpe = -0.20 -> degradation = 1.0 >= 0.35
    inp_neg = _make_tournament_input(
        challenger_net_sharpe=-0.20,
        champion_net_sharpe=0.02,
        challenger_max_drawdown_pct=0.10,
        champion_max_drawdown_pct=0.12,
        challenger_brier_skill_score=0.10,
        champion_brier_skill_score=0.08,
        challenger_venn_abers_spread=0.09,
        challenger_in_sample_sharpe=0.0,
    )
    report_neg = service.run_tournament(inp_neg)
    assert report_neg.approval_checklist["gate_5_anti_overfitting"] is False
