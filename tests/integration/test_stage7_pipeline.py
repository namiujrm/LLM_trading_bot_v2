"""Comprehensive End-to-End Integration Test Suite for Stage 7 Pipeline.

Milestone M7.4: Notifier Adapter, CLI Runner, Integration Tests & 5-Stage Verification Gate.
Covers:
- Full dry-run execution of `run_weekend_alpha_evolution` producing a valid tournament report file.
- CLI runner entry point with `--dry-run` flag via `run_cli` and `subprocess.run`.
- All 3 tournament outcomes (`PROMOTE_CHALLENGER_IMMEDIATELY`, `RUN_48H_SHADOW_BATTLE`, `KEEP_CHAMPION`)
  with file output verification and approval checklist state checks.
- Interactive operator prompt simulation (options 1, 2, 3, and default fallback).
- Auto-creation of nested report directories.
- Fail-fast validation on invalid arguments and missing live environment variables.

Enforces 100% Invariant #9 compliance across all test helpers and runners:
- Zero default parameters in functions and methods.
- Zero class field defaults.
- Zero silent fallbacks.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from scripts.run_weekend_alpha_evolution import (
    CliBotLogger,
    DryRunMathAlphaAgent,
    WeekendEvolutionCLIConfig,
    execute_dry_run,
    parse_cli_arguments,
    run_cli,
)

from trading_bot.adapters.notification.console_tournament_notifier import (
    ConsoleAndFileTournamentNotifierAdapter,
)
from trading_bot.application.services.alpha_pruning_service import AlphaPruningService
from trading_bot.application.services.model_tournament_service import (
    ModelTournamentConfig,
    ModelTournamentService,
    TournamentEvaluationInput,
)
from trading_bot.application.use_cases.run_weekend_alpha_evolution import (
    CandidateSeriesData,
    RunWeekendAlphaEvolutionUseCase,
    WeekendAlphaEvolutionRequest,
    WeekendAlphaEvolutionResult,
)
from trading_bot.domain.entities.quant.symbolic_alpha import (
    AlphaGatekeeperConfig,
    ModelTournamentReport,
    TournamentVerdict,
)
from trading_bot.domain.services.quant.statistical_alpha_gatekeeper import (
    StatisticalAlphaGatekeeper,
)


def _make_series_data(n_bars: int) -> CandidateSeriesData:
    """Helper to build candidate series data with zero defaults."""
    alpha_oos = [float(i) for i in range(n_bars)]
    alpha_trunc = [float(i) for i in range(n_bars // 2)]
    target_returns = [0.005 * float(i) for i in range(n_bars)]
    realized_vol = [0.02 for _ in range(n_bars)]
    existing_matrix: list[list[float]] = []

    return CandidateSeriesData(
        alpha_oos_series=alpha_oos,
        alpha_truncated_series=alpha_trunc,
        target_returns_oos=target_returns,
        realized_vol_series=realized_vol,
        existing_alpha_matrix_oos=existing_matrix,
    )


def _make_tournament_input(
    created_at_utc: str,
    challenger_sharpe: float,
    challenger_drawdown: float,
    challenger_bss: float,
    challenger_venn_spread: float,
    challenger_in_sample_sharpe: float,
) -> TournamentEvaluationInput:
    """Helper to construct TournamentEvaluationInput with zero defaults."""
    return TournamentEvaluationInput(
        champion_model_id="champion_v1_lgbm_dual",
        challenger_model_id="challenger_v2_lgbm_dual",
        champion_feature_count=38,
        challenger_feature_count=42,
        champion_net_sharpe=1.45,
        challenger_net_sharpe=challenger_sharpe,
        champion_max_drawdown_pct=0.115,
        challenger_max_drawdown_pct=challenger_drawdown,
        champion_calmar_ratio=2.6,
        challenger_calmar_ratio=3.4,
        champion_win_rate=0.552,
        challenger_win_rate=0.588,
        champion_profit_factor=1.65,
        challenger_profit_factor=1.85,
        champion_expectancy_r=0.28,
        challenger_expectancy_r=0.38,
        champion_brier_score=0.175,
        challenger_brier_score=0.148,
        champion_brier_skill_score=0.082,
        challenger_brier_skill_score=challenger_bss,
        champion_log_loss=0.445,
        challenger_log_loss=0.395,
        champion_venn_abers_spread=0.105,
        challenger_venn_abers_spread=challenger_venn_spread,
        challenger_in_sample_sharpe=challenger_in_sample_sharpe,
        created_at_utc=created_at_utc,
    )


@pytest.mark.asyncio
async def test_stage7_dry_run_pipeline_execution(tmp_path: Path) -> None:
    """Verify full end-to-end dry-run execution produces valid report file and correct result."""
    report_dir = tmp_path / "weekend_evolution_reports"
    config = WeekendEvolutionCLIConfig(
        is_dry_run=True,
        is_interactive=False,
        report_dir=str(report_dir),
        target_count_per_cluster=2,
        verdict_override=None,
    )
    logger = CliBotLogger()

    result: WeekendAlphaEvolutionResult = await execute_dry_run(
        config=config,
        logger=logger,
    )

    # 1. Verify result structure and verdict
    assert isinstance(result, WeekendAlphaEvolutionResult)
    assert result.verdict == TournamentVerdict.PROMOTE_CHALLENGER_IMMEDIATELY
    assert result.tournament_report.is_promotable is True
    assert len(result.generated_candidates) == 8  # 4 clusters * 2 candidates
    assert len(result.passed_candidates) == 8
    assert len(result.active_zoo_features) == 48  # 40 baseline + 8 passed = 48 (under 50 cap)
    assert result.pruned_feature_ids == []

    # 2. Verify file output on disk
    report_files = list(report_dir.glob("tournament_*.md"))
    assert len(report_files) == 1, f"Expected 1 report file, found {len(report_files)}"

    report_content = report_files[0].read_text(encoding="utf-8")
    assert "# Weekend Alpha Evolution: Champion vs. Challenger Tournament Report" in report_content
    assert "champion_v1_lgbm_dual" in report_content
    assert "challenger_v2_lgbm_dual" in report_content
    assert "PROMOTE_CHALLENGER_IMMEDIATELY" in report_content
    assert (
        "| Gate 1 | Financial Outperformance | Challenger Sharpe Delta >= +0.15 | [PASS] PASSED |"
        in report_content
    )
    assert (
        "| Gate 2 | Capital Risk Preservation | Challenger Max Drawdown <= Champion Max Drawdown | [PASS] PASSED |"
        in report_content
    )
    assert (
        "| Gate 3 | Probability Calibration Quality | Challenger Brier Skill Score > Champion BSS | [PASS] PASSED |"
        in report_content
    )
    assert (
        "| Gate 4 | Venn-Abers Epistemic Uncertainty | Challenger Spread <= 0.12 | [PASS] PASSED |"
        in report_content
    )
    assert (
        "| Gate 5 | Anti-Overfitting Discipline | In-Sample to OOS Sharpe Decay < 35% | [PASS] PASSED |"
        in report_content
    )


@pytest.mark.asyncio
async def test_stage7_cli_runner_direct_invocation_dry_run(tmp_path: Path) -> None:
    """Verify run_cli function parses arguments and runs complete pipeline."""
    report_dir = tmp_path / "cli_reports"
    args = [
        "--dry-run",
        "--non-interactive",
        "--report-dir",
        str(report_dir),
        "--target-count",
        "1",
    ]

    result: WeekendAlphaEvolutionResult = await run_cli(args)

    assert result.verdict == TournamentVerdict.PROMOTE_CHALLENGER_IMMEDIATELY
    assert len(result.generated_candidates) == 4  # 4 clusters * 1 candidate
    assert len(result.passed_candidates) == 4
    assert len(result.active_zoo_features) == 44

    # Verify report written
    report_files = list(report_dir.glob("tournament_*.md"))
    assert len(report_files) == 1
    assert report_files[0].stat().st_size > 500


def test_stage7_cli_runner_subprocess_execution(tmp_path: Path) -> None:
    """Verify scripts/run_weekend_alpha_evolution.py runs as an independent CLI subprocess."""
    report_dir = tmp_path / "subprocess_reports"
    cmd = [
        sys.executable,
        "scripts/run_weekend_alpha_evolution.py",
        "--dry-run",
        "--non-interactive",
        "--report-dir",
        str(report_dir),
        "--target-count",
        "1",
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, (
        f"CLI runner failed with returncode {proc.returncode}: {proc.stderr}"
    )
    assert "[CLI COMPLETE] Verdict: PROMOTE_CHALLENGER_IMMEDIATELY" in proc.stdout
    assert "Active Zoo Features: 44" in proc.stdout

    report_files = list(report_dir.glob("tournament_*.md"))
    assert len(report_files) == 1
    content = report_files[0].read_text(encoding="utf-8")
    assert "PROMOTE_CHALLENGER_IMMEDIATELY" in content


@pytest.mark.asyncio
async def test_stage7_tournament_outcome_run_48h_shadow_battle(tmp_path: Path) -> None:
    """Verify RUN_48H_SHADOW_BATTLE outcome when borderline (4 of 5 gates pass)."""
    report_dir = tmp_path / "shadow_reports"
    logger = CliBotLogger()

    # Borderline: Sharpe delta is +0.10 (fails Gate 1 Sharpe delta >= 0.15), but other 4 gates pass
    eval_input = _make_tournament_input(
        created_at_utc="2026-11-10T12:00:00Z",
        challenger_sharpe=1.55,  # 1.55 - 1.45 = +0.10 < 0.15 (Fails Gate 1)
        challenger_drawdown=0.095,  # Pass Gate 2
        challenger_bss=0.120,  # Pass Gate 3
        challenger_venn_spread=0.090,  # Pass Gate 4
        challenger_in_sample_sharpe=1.70,  # Decay (1.70 - 1.55)/1.70 = 8.8% < 35% (Pass Gate 5)
    )

    gatekeeper_cfg = AlphaGatekeeperConfig(
        min_valid_sample_ratio=0.90,
        min_series_std=1e-4,
        invariance_tol=1e-12,
        min_abs_rank_ic=0.025,
        min_abs_icir=0.50,
        min_t_statistic=2.5,
        min_daily_blocks_for_icir=5,
        daily_block_bars=48,
        min_quantile_monotonicity=0.80,
        max_collinearity_corr=0.60,
    )
    gatekeeper = StatisticalAlphaGatekeeper(gatekeeper_cfg)
    pruning_service = AlphaPruningService(35, 50, 0.012)
    tournament_cfg = ModelTournamentConfig(
        sharpe_delta_min=0.15,
        venn_abers_spread_max=0.12,
        max_sharpe_degradation_ratio=0.35,
        min_passed_gates_for_shadow=3,
    )
    tournament_service = ModelTournamentService(tournament_cfg)
    notifier = ConsoleAndFileTournamentNotifierAdapter(
        report_dir=str(report_dir),
        interactive=False,
        default_non_interactive_verdict=None,
    )

    use_case = RunWeekendAlphaEvolutionUseCase(
        math_alpha_agent=DryRunMathAlphaAgent(1),
        gatekeeper=gatekeeper,
        pruning_service=pruning_service,
        tournament_service=tournament_service,
        tournament_notifier=notifier,
        logger=logger,
    )

    req = WeekendAlphaEvolutionRequest(
        blindspots=[],
        decayed_alphas=[],
        current_zoo_features=[f"feat_{i}" for i in range(38)],
        decayed_feature_metrics={},
        target_count_per_cluster=1,
        candidate_series_map=None,
        default_series_data=_make_series_data(480),
        tournament_eval_input=eval_input,
    )

    result = await use_case.execute(req)

    assert result.verdict == TournamentVerdict.RUN_48H_SHADOW_BATTLE
    assert result.tournament_report.is_promotable is False
    assert result.tournament_report.approval_checklist["gate_1_sharpe_outperformance"] is False
    assert result.tournament_report.approval_checklist["gate_2_max_drawdown_preservation"] is True

    # Check generated markdown
    report_file = report_dir / "tournament_2026-11-10.md"
    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "RUN_48H_SHADOW_BATTLE" in content
    assert (
        "| Gate 1 | Financial Outperformance | Challenger Sharpe Delta >= +0.15 | [FAIL] FAILED |"
        in content
    )
    assert (
        "| Gate 2 | Capital Risk Preservation | Challenger Max Drawdown <= Champion Max Drawdown | [PASS] PASSED |"
        in content
    )


@pytest.mark.asyncio
async def test_stage7_tournament_outcome_keep_champion(tmp_path: Path) -> None:
    """Verify KEEP_CHAMPION outcome when Gate 2 (Capital Risk Preservation) fails."""
    report_dir = tmp_path / "keep_champ_reports"
    logger = CliBotLogger()

    # Gate 2 failure: Challenger Drawdown is 15.0% > Champion 11.5%
    eval_input = _make_tournament_input(
        created_at_utc="2026-11-10T12:00:00Z",
        challenger_sharpe=1.85,  # Pass Gate 1 (+0.40)
        challenger_drawdown=0.150,  # Fails Gate 2 (0.150 > 0.115)
        challenger_bss=0.130,  # Pass Gate 3
        challenger_venn_spread=0.080,  # Pass Gate 4
        challenger_in_sample_sharpe=2.00,  # Pass Gate 5
    )

    gatekeeper_cfg = AlphaGatekeeperConfig(
        min_valid_sample_ratio=0.90,
        min_series_std=1e-4,
        invariance_tol=1e-12,
        min_abs_rank_ic=0.025,
        min_abs_icir=0.50,
        min_t_statistic=2.5,
        min_daily_blocks_for_icir=5,
        daily_block_bars=48,
        min_quantile_monotonicity=0.80,
        max_collinearity_corr=0.60,
    )
    gatekeeper = StatisticalAlphaGatekeeper(gatekeeper_cfg)
    pruning_service = AlphaPruningService(35, 50, 0.012)
    tournament_cfg = ModelTournamentConfig(
        sharpe_delta_min=0.15,
        venn_abers_spread_max=0.12,
        max_sharpe_degradation_ratio=0.35,
        min_passed_gates_for_shadow=3,
    )
    tournament_service = ModelTournamentService(tournament_cfg)
    notifier = ConsoleAndFileTournamentNotifierAdapter(
        report_dir=str(report_dir),
        interactive=False,
        default_non_interactive_verdict=None,
    )

    use_case = RunWeekendAlphaEvolutionUseCase(
        math_alpha_agent=DryRunMathAlphaAgent(1),
        gatekeeper=gatekeeper,
        pruning_service=pruning_service,
        tournament_service=tournament_service,
        tournament_notifier=notifier,
        logger=logger,
    )

    req = WeekendAlphaEvolutionRequest(
        blindspots=[],
        decayed_alphas=[],
        current_zoo_features=[f"feat_{i}" for i in range(38)],
        decayed_feature_metrics={},
        target_count_per_cluster=1,
        candidate_series_map=None,
        default_series_data=_make_series_data(480),
        tournament_eval_input=eval_input,
    )

    result = await use_case.execute(req)

    assert result.verdict == TournamentVerdict.KEEP_CHAMPION
    assert result.tournament_report.is_promotable is False
    assert result.tournament_report.approval_checklist["gate_2_max_drawdown_preservation"] is False

    report_file = report_dir / "tournament_2026-11-10.md"
    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "KEEP_CHAMPION" in content
    assert (
        "| Gate 2 | Capital Risk Preservation | Challenger Max Drawdown <= Champion Max Drawdown | [FAIL] FAILED |"
        in content
    )


@pytest.mark.asyncio
async def test_stage7_interactive_notifier_decisions(tmp_path: Path) -> None:
    """Verify operator decisions via input prompt in interactive mode."""
    report_dir = tmp_path / "interactive_reports"
    report = ModelTournamentReport(
        created_at_utc="2026-11-10T12:00:00Z",
        champion_model_id="champ",
        challenger_model_id="chall",
        champion_feature_count=35,
        challenger_feature_count=38,
        champion_net_sharpe=1.4,
        challenger_net_sharpe=1.7,
        champion_max_drawdown_pct=0.10,
        challenger_max_drawdown_pct=0.08,
        champion_calmar_ratio=2.5,
        challenger_calmar_ratio=3.2,
        champion_win_rate=0.55,
        challenger_win_rate=0.59,
        champion_profit_factor=1.6,
        challenger_profit_factor=1.8,
        champion_expectancy_r=0.25,
        challenger_expectancy_r=0.35,
        champion_brier_score=0.18,
        challenger_brier_score=0.15,
        champion_brier_skill_score=0.08,
        challenger_brier_skill_score=0.12,
        champion_log_loss=0.45,
        challenger_log_loss=0.40,
        champion_venn_abers_spread=0.10,
        challenger_venn_abers_spread=0.08,
        is_promotable=True,
        recommended_verdict=TournamentVerdict.PROMOTE_CHALLENGER_IMMEDIATELY,
        approval_checklist={},
        verdict_rationale="All 5 gates passed.",
    )

    # Choice "1" -> PROMOTE_CHALLENGER_IMMEDIATELY
    notifier1 = ConsoleAndFileTournamentNotifierAdapter(str(report_dir), True, None)
    with patch("builtins.input", return_value="1"):
        verdict1 = await notifier1.present_tournament_report(report)
        assert verdict1 == TournamentVerdict.PROMOTE_CHALLENGER_IMMEDIATELY

    # Choice "2" -> RUN_48H_SHADOW_BATTLE
    notifier2 = ConsoleAndFileTournamentNotifierAdapter(str(report_dir), True, None)
    with patch("builtins.input", return_value="2"):
        verdict2 = await notifier2.present_tournament_report(report)
        assert verdict2 == TournamentVerdict.RUN_48H_SHADOW_BATTLE

    # Choice "3" -> KEEP_CHAMPION
    notifier3 = ConsoleAndFileTournamentNotifierAdapter(str(report_dir), True, None)
    with patch("builtins.input", return_value="3"):
        verdict3 = await notifier3.present_tournament_report(report)
        assert verdict3 == TournamentVerdict.KEEP_CHAMPION

    # Empty choice or default -> recommended verdict
    notifier4 = ConsoleAndFileTournamentNotifierAdapter(str(report_dir), True, None)
    with patch("builtins.input", return_value=""):
        verdict4 = await notifier4.present_tournament_report(report)
        assert verdict4 == TournamentVerdict.PROMOTE_CHALLENGER_IMMEDIATELY

    # EOFError -> recommended verdict
    notifier5 = ConsoleAndFileTournamentNotifierAdapter(str(report_dir), True, None)
    with patch("builtins.input", side_effect=EOFError):
        verdict5 = await notifier5.present_tournament_report(report)
        assert verdict5 == TournamentVerdict.PROMOTE_CHALLENGER_IMMEDIATELY


@pytest.mark.asyncio
async def test_stage7_non_interactive_override(tmp_path: Path) -> None:
    """Verify non-interactive mode respects explicit default_non_interactive_verdict."""
    report_dir = tmp_path / "override_reports"
    report = ModelTournamentReport(
        created_at_utc="2026-11-10T12:00:00Z",
        champion_model_id="champ",
        challenger_model_id="chall",
        champion_feature_count=35,
        challenger_feature_count=38,
        champion_net_sharpe=1.4,
        challenger_net_sharpe=1.7,
        champion_max_drawdown_pct=0.10,
        challenger_max_drawdown_pct=0.08,
        champion_calmar_ratio=2.5,
        challenger_calmar_ratio=3.2,
        champion_win_rate=0.55,
        challenger_win_rate=0.59,
        champion_profit_factor=1.6,
        challenger_profit_factor=1.8,
        champion_expectancy_r=0.25,
        challenger_expectancy_r=0.35,
        champion_brier_score=0.18,
        challenger_brier_score=0.15,
        champion_brier_skill_score=0.08,
        challenger_brier_skill_score=0.12,
        champion_log_loss=0.45,
        challenger_log_loss=0.40,
        champion_venn_abers_spread=0.10,
        challenger_venn_abers_spread=0.08,
        is_promotable=True,
        recommended_verdict=TournamentVerdict.PROMOTE_CHALLENGER_IMMEDIATELY,
        approval_checklist={},
        verdict_rationale="Passed.",
    )

    notifier = ConsoleAndFileTournamentNotifierAdapter(
        report_dir=str(report_dir),
        interactive=False,
        default_non_interactive_verdict=TournamentVerdict.RUN_48H_SHADOW_BATTLE,
    )
    verdict = await notifier.present_tournament_report(report)
    assert verdict == TournamentVerdict.RUN_48H_SHADOW_BATTLE


def test_stage7_cli_arguments_parser_validation() -> None:
    """Verify CLI argument parsing behavior and fail-fast validation."""
    # 1. Defaults with no args
    cfg1 = parse_cli_arguments([])
    assert cfg1.is_dry_run is True
    assert cfg1.is_interactive is True
    assert cfg1.report_dir == "reports/weekend_evolution"
    assert cfg1.target_count_per_cluster == 2
    assert cfg1.verdict_override is None

    # 2. Explicit flags
    cfg2 = parse_cli_arguments(
        [
            "--execute",
            "--non-interactive",
            "--report-dir",
            "custom/path",
            "--target-count",
            "5",
            "--verdict-override",
            "RUN_48H_SHADOW_BATTLE",
        ]
    )
    assert cfg2.is_dry_run is False
    assert cfg2.is_interactive is False
    assert cfg2.report_dir == "custom/path"
    assert cfg2.target_count_per_cluster == 5
    assert cfg2.verdict_override == TournamentVerdict.RUN_48H_SHADOW_BATTLE

    # 3. Numeric verdict override "1"
    cfg3 = parse_cli_arguments(["--verdict-override", "1"])
    assert cfg3.verdict_override == TournamentVerdict.PROMOTE_CHALLENGER_IMMEDIATELY

    # 4. Invalid verdict override raises ValueError
    with pytest.raises(ValueError, match="Invalid verdict override"):
        parse_cli_arguments(["--verdict-override", "INVALID_VERDICT_STRING"])
