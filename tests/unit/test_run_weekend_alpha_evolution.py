"""Unit tests for RunWeekendAlphaEvolutionUseCase (Milestone 7.3, Invariant #9)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

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
    CLUSTER_0_REGIME,
    CLUSTER_1_MICRO,
    CLUSTER_2_DERIV,
    CLUSTER_3_CROSS,
    AlphaGatekeeperConfig,
    AlphaGateResult,
    ModelTournamentReport,
    SymbolicAlphaCandidate,
    TournamentVerdict,
)
from trading_bot.domain.exceptions import InsufficientDataException
from trading_bot.domain.services.quant.statistical_alpha_gatekeeper import (
    StatisticalAlphaGatekeeper,
)
from trading_bot.ports.logger_port import StructuredBotLoggerPort
from trading_bot.ports.math_alpha_agent_port import MathAlphaAgentPort
from trading_bot.ports.model_tournament_port import ModelTournamentNotifierPort


def _make_candidate(alpha_id: str, cluster: str) -> SymbolicAlphaCandidate:
    """Helper to create a SymbolicAlphaCandidate with zero defaults."""
    return SymbolicAlphaCandidate(
        alpha_id=alpha_id,
        dsl_expression=f"ts_zscore({alpha_id}, 16)",
        causal_cluster=cluster,
        long_monotonic_prior=1,
        short_monotonic_prior=-1,
        targeted_blindspot="false_positive_breakout",
        economic_rationale="Order flow imbalance momentum signal",
    )


def _make_series_data(n: int = 480) -> CandidateSeriesData:
    """Helper to create deterministic, valid screening series data."""
    alpha_oos = [float(i) for i in range(n)]
    alpha_trunc = [float(i) for i in range(n // 2)]
    target_returns = [0.001 * float(i) for i in range(n)]
    realized_vol = [0.02 for _ in range(n)]
    existing_matrix: list[list[float]] = []
    return CandidateSeriesData(
        alpha_oos_series=alpha_oos,
        alpha_truncated_series=alpha_trunc,
        target_returns_oos=target_returns,
        realized_vol_series=realized_vol,
        existing_alpha_matrix_oos=existing_matrix,
    )


def _make_tournament_input() -> TournamentEvaluationInput:
    """Helper to create tournament input."""
    return TournamentEvaluationInput(
        champion_model_id="champion_v1",
        challenger_model_id="challenger_v2",
        champion_feature_count=38,
        challenger_feature_count=40,
        champion_net_sharpe=1.50,
        challenger_net_sharpe=1.75,
        champion_max_drawdown_pct=0.12,
        challenger_max_drawdown_pct=0.10,
        champion_calmar_ratio=2.5,
        challenger_calmar_ratio=3.2,
        champion_win_rate=0.55,
        challenger_win_rate=0.59,
        champion_profit_factor=1.65,
        challenger_profit_factor=1.88,
        champion_expectancy_r=0.25,
        challenger_expectancy_r=0.34,
        champion_brier_score=0.18,
        challenger_brier_score=0.15,
        champion_brier_skill_score=0.08,
        challenger_brier_skill_score=0.12,
        champion_log_loss=0.45,
        challenger_log_loss=0.40,
        champion_venn_abers_spread=0.10,
        challenger_venn_abers_spread=0.08,
        challenger_in_sample_sharpe=2.00,
        created_at_utc="2026-11-10T12:00:00Z",
    )


@pytest.mark.asyncio
async def test_run_weekend_alpha_evolution_full_orchestration() -> None:
    """Verify full end-to-end orchestration across all 7 pipeline stages."""
    # Mocks
    mock_agent = AsyncMock(spec=MathAlphaAgentPort)
    mock_gatekeeper = MagicMock(spec=StatisticalAlphaGatekeeper)
    mock_notifier = AsyncMock(spec=ModelTournamentNotifierPort)
    mock_logger = MagicMock(spec=StructuredBotLoggerPort)

    # Return 1 candidate per cluster
    cand_regime = _make_candidate("cand_regime_1", CLUSTER_0_REGIME)
    cand_micro = _make_candidate("cand_micro_1", CLUSTER_1_MICRO)
    cand_deriv = _make_candidate("cand_deriv_1", CLUSTER_2_DERIV)
    cand_cross = _make_candidate("cand_cross_1", CLUSTER_3_CROSS)

    async def mock_gen(
        cluster_name: str,
        blindspots: list[dict[str, str]],
        decayed_alphas: list[str],
        target_count: int,
    ) -> list[SymbolicAlphaCandidate]:
        if cluster_name == CLUSTER_0_REGIME:
            return [cand_regime]
        elif cluster_name == CLUSTER_1_MICRO:
            return [cand_micro]
        elif cluster_name == CLUSTER_2_DERIV:
            return [cand_deriv]
        else:
            return [cand_cross]

    mock_agent.generate_candidates_for_cluster.side_effect = mock_gen

    # Gatekeeper passes regime, micro, deriv, but rejects cross
    def mock_evaluate(
        alpha_id: str,
        alpha_oos_series: list[float],
        alpha_truncated_series: list[float],
        target_returns_oos: list[float],
        realized_vol_series: list[float],
        expected_monotonic_sign: int,
        existing_alpha_matrix_oos: list[list[float]],
    ) -> AlphaGateResult:
        passed = alpha_id != "cand_cross_1"
        return AlphaGateResult(
            alpha_id=alpha_id,
            passed_all_gates=passed,
            failed_stage=None if passed else "STAGE_2_WEAK_IC",
            oos_rank_ic=0.035 if passed else 0.005,
            oos_icir=0.65 if passed else 0.10,
            t_statistic=2.8 if passed else 0.8,
            is_regime_invariant=True,
            quantile_monotonicity_corr=0.85 if passed else 0.20,
            max_corr_with_existing=0.25,
        )

    mock_gatekeeper.evaluate_candidate.side_effect = mock_evaluate

    # Real services
    pruning_service = AlphaPruningService(
        min_features=35,
        max_features=50,
        ic_decay_threshold=0.012,
    )
    tournament_config = ModelTournamentConfig(
        sharpe_delta_min=0.15,
        venn_abers_spread_max=0.12,
        max_sharpe_degradation_ratio=0.35,
        min_passed_gates_for_shadow=3,
    )
    tournament_service = ModelTournamentService(tournament_config)

    # Notifier returns immediate promotion
    mock_notifier.present_tournament_report.return_value = (
        TournamentVerdict.PROMOTE_CHALLENGER_IMMEDIATELY
    )

    use_case = RunWeekendAlphaEvolutionUseCase(
        math_alpha_agent=mock_agent,
        gatekeeper=mock_gatekeeper,
        pruning_service=pruning_service,
        tournament_service=tournament_service,
        tournament_notifier=mock_notifier,
        logger=mock_logger,
    )

    series_data = _make_series_data()
    req = WeekendAlphaEvolutionRequest(
        blindspots=[{"type": "FALSE_POSITIVE_TRAP", "symbol": "BTCUSDT"}],
        decayed_alphas=["old_alpha_1"],
        current_zoo_features=[f"zoo_feat_{i}" for i in range(40)],
        decayed_feature_metrics={"zoo_feat_0": 0.005},
        target_count_per_cluster=2,
        candidate_series_map=None,
        default_series_data=series_data,
        tournament_eval_input=_make_tournament_input(),
    )

    result = await use_case.execute(req)

    assert isinstance(result, WeekendAlphaEvolutionResult)
    assert result.verdict == TournamentVerdict.PROMOTE_CHALLENGER_IMMEDIATELY
    assert result.tournament_report.is_promotable is True
    assert len(result.generated_candidates) == 4
    assert len(result.passed_candidates) == 3  # cross was rejected
    assert "cand_cross_1" not in [c.alpha_id for c in result.passed_candidates]
    assert "cand_regime_1" in [c.alpha_id for c in result.passed_candidates]

    # Zoo features count: 40 + 3 passed = 43 (under 50 cap, so no features pruned)
    assert len(result.active_zoo_features) == 43
    assert result.pruned_feature_ids == []

    # Verify calls
    assert mock_agent.generate_candidates_for_cluster.call_count == 4
    assert mock_gatekeeper.evaluate_candidate.call_count == 4
    assert mock_notifier.present_tournament_report.call_count == 1
    assert mock_logger.info.call_count >= 5

    # Verify tuple unpacking
    report, verdict = result
    assert isinstance(report, ModelTournamentReport)
    assert verdict == TournamentVerdict.PROMOTE_CHALLENGER_IMMEDIATELY


@pytest.mark.asyncio
async def test_run_weekend_alpha_evolution_with_real_gatekeeper_and_pruning() -> None:
    """Verify integration with real StatisticalAlphaGatekeeper and real AlphaPruningService."""
    mock_agent = AsyncMock(spec=MathAlphaAgentPort)
    mock_notifier = AsyncMock(spec=ModelTournamentNotifierPort)
    mock_logger = MagicMock(spec=StructuredBotLoggerPort)

    cand_valid = _make_candidate("cand_valid", CLUSTER_0_REGIME)

    # Return cand_valid only for CLUSTER_0_REGIME, empty for other clusters
    async def mock_gen(cluster_name: str, **kwargs: object) -> list[SymbolicAlphaCandidate]:
        if cluster_name == CLUSTER_0_REGIME:
            return [cand_valid]
        return []

    mock_agent.generate_candidates_for_cluster.side_effect = mock_gen

    # Real Gatekeeper with standard quant config
    gate_cfg = AlphaGatekeeperConfig(
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
    gatekeeper = StatisticalAlphaGatekeeper(gate_cfg)

    pruning_service = AlphaPruningService(
        min_features=35,
        max_features=50,
        ic_decay_threshold=0.012,
    )
    tournament_config = ModelTournamentConfig(
        sharpe_delta_min=0.15,
        venn_abers_spread_max=0.12,
        max_sharpe_degradation_ratio=0.35,
        min_passed_gates_for_shadow=3,
    )
    tournament_service = ModelTournamentService(tournament_config)

    mock_notifier.present_tournament_report.return_value = TournamentVerdict.RUN_48H_SHADOW_BATTLE

    use_case = RunWeekendAlphaEvolutionUseCase(
        math_alpha_agent=mock_agent,
        gatekeeper=gatekeeper,
        pruning_service=pruning_service,
        tournament_service=tournament_service,
        tournament_notifier=mock_notifier,
        logger=mock_logger,
    )

    n = 480
    valid_series = CandidateSeriesData(
        alpha_oos_series=[float(i) for i in range(n)],
        alpha_truncated_series=[float(i) for i in range(n // 2)],
        target_returns_oos=[0.01 * float(i) for i in range(n)],
        realized_vol_series=[0.02 for _ in range(n)],
        existing_alpha_matrix_oos=[],
    )

    # 49 zoo features + decayed metrics
    current_zoo = [f"zoo_{i}" for i in range(49)]
    decayed_metrics = {"zoo_0": 0.005, "zoo_1": 0.008}

    req = WeekendAlphaEvolutionRequest(
        blindspots=[],
        decayed_alphas=["zoo_0"],
        current_zoo_features=current_zoo,
        decayed_feature_metrics=decayed_metrics,
        target_count_per_cluster=1,
        candidate_series_map=None,
        default_series_data=valid_series,
        tournament_eval_input=_make_tournament_input(),
    )

    result = await use_case.execute(req)

    assert result.verdict == TournamentVerdict.RUN_48H_SHADOW_BATTLE
    assert len(result.passed_candidates) == 1
    assert result.passed_candidates[0].alpha_id == "cand_valid"
    # Total before pruning: 49 + 1 = 50. Exactly at cap, no features pruned!
    assert len(result.active_zoo_features) == 50
    assert result.pruned_feature_ids == []


@pytest.mark.asyncio
async def test_run_weekend_alpha_evolution_fails_fast_on_missing_series() -> None:
    """Ensure InsufficientDataException is raised if no series data is provided for a candidate."""
    mock_agent = AsyncMock(spec=MathAlphaAgentPort)
    mock_gatekeeper = MagicMock(spec=StatisticalAlphaGatekeeper)
    mock_notifier = AsyncMock(spec=ModelTournamentNotifierPort)
    mock_logger = MagicMock(spec=StructuredBotLoggerPort)

    cand = _make_candidate("cand_unmapped", CLUSTER_0_REGIME)

    async def mock_gen(cluster_name: str, **kwargs: object) -> list[SymbolicAlphaCandidate]:
        if cluster_name == CLUSTER_0_REGIME:
            return [cand]
        return []

    mock_agent.generate_candidates_for_cluster.side_effect = mock_gen

    pruning_service = AlphaPruningService(35, 50, 0.012)
    tournament_config = ModelTournamentConfig(
        sharpe_delta_min=0.15,
        venn_abers_spread_max=0.12,
        max_sharpe_degradation_ratio=0.35,
        min_passed_gates_for_shadow=3,
    )
    tournament_service = ModelTournamentService(tournament_config)

    use_case = RunWeekendAlphaEvolutionUseCase(
        math_alpha_agent=mock_agent,
        gatekeeper=mock_gatekeeper,
        pruning_service=pruning_service,
        tournament_service=tournament_service,
        tournament_notifier=mock_notifier,
        logger=mock_logger,
    )

    # Neither candidate_series_map nor default_series_data is provided
    req = WeekendAlphaEvolutionRequest(
        blindspots=[],
        decayed_alphas=[],
        current_zoo_features=[f"zoo_{i}" for i in range(35)],
        decayed_feature_metrics={},
        target_count_per_cluster=1,
        candidate_series_map={},
        default_series_data=None,
        tournament_eval_input=_make_tournament_input(),
    )

    with pytest.raises(InsufficientDataException, match="No series evaluation data provided"):
        await use_case.execute(req)


@pytest.mark.asyncio
async def test_candidate_series_map_resolution() -> None:
    """Verify specific candidate series resolution from candidate_series_map."""
    mock_agent = AsyncMock(spec=MathAlphaAgentPort)
    mock_gatekeeper = MagicMock(spec=StatisticalAlphaGatekeeper)
    mock_notifier = AsyncMock(spec=ModelTournamentNotifierPort)
    mock_logger = MagicMock(spec=StructuredBotLoggerPort)

    cand_mapped = _make_candidate("cand_specific", CLUSTER_0_REGIME)

    async def mock_gen(cluster_name: str, **kwargs: object) -> list[SymbolicAlphaCandidate]:
        if cluster_name == CLUSTER_0_REGIME:
            return [cand_mapped]
        return []

    mock_agent.generate_candidates_for_cluster.side_effect = mock_gen

    mock_gatekeeper.evaluate_candidate.return_value = AlphaGateResult(
        alpha_id="cand_specific",
        passed_all_gates=True,
        failed_stage=None,
        oos_rank_ic=0.04,
        oos_icir=0.7,
        t_statistic=3.0,
        is_regime_invariant=True,
        quantile_monotonicity_corr=0.9,
        max_corr_with_existing=0.1,
    )

    pruning_service = AlphaPruningService(35, 50, 0.012)
    tournament_config = ModelTournamentConfig(
        sharpe_delta_min=0.15,
        venn_abers_spread_max=0.12,
        max_sharpe_degradation_ratio=0.35,
        min_passed_gates_for_shadow=3,
    )
    tournament_service = ModelTournamentService(tournament_config)
    mock_notifier.present_tournament_report.return_value = TournamentVerdict.KEEP_CHAMPION

    use_case = RunWeekendAlphaEvolutionUseCase(
        math_alpha_agent=mock_agent,
        gatekeeper=mock_gatekeeper,
        pruning_service=pruning_service,
        tournament_service=tournament_service,
        tournament_notifier=mock_notifier,
        logger=mock_logger,
    )

    specific_series = _make_series_data(n=300)
    req = WeekendAlphaEvolutionRequest(
        blindspots=[],
        decayed_alphas=[],
        current_zoo_features=[f"zoo_{i}" for i in range(35)],
        decayed_feature_metrics={},
        target_count_per_cluster=1,
        candidate_series_map={"cand_specific": specific_series},
        default_series_data=None,
        tournament_eval_input=_make_tournament_input(),
    )

    result = await use_case.execute(req)
    assert result.verdict == TournamentVerdict.KEEP_CHAMPION
    assert len(result.passed_candidates) == 1
    assert result.passed_candidates[0].alpha_id == "cand_specific"


@pytest.mark.asyncio
async def test_run_weekend_alpha_evolution_zero_candidates_pass_gates() -> None:
    """When all candidates fail screening, zoo remains unmodified and tournament still proceeds."""
    mock_agent = AsyncMock(spec=MathAlphaAgentPort)
    mock_gatekeeper = MagicMock(spec=StatisticalAlphaGatekeeper)
    mock_notifier = AsyncMock(spec=ModelTournamentNotifierPort)
    mock_logger = MagicMock(spec=StructuredBotLoggerPort)

    cand_bad = _make_candidate("cand_bad", CLUSTER_0_REGIME)
    mock_agent.generate_candidates_for_cluster.return_value = [cand_bad]

    mock_gatekeeper.evaluate_candidate.return_value = AlphaGateResult(
        alpha_id="cand_bad",
        passed_all_gates=False,
        failed_stage="STAGE_1_LOOKAHEAD_VIOLATION",
        oos_rank_ic=0.0,
        oos_icir=0.0,
        t_statistic=0.0,
        is_regime_invariant=False,
        quantile_monotonicity_corr=0.0,
        max_corr_with_existing=0.0,
    )

    pruning_service = AlphaPruningService(35, 50, 0.012)
    tournament_config = ModelTournamentConfig(
        sharpe_delta_min=0.15,
        venn_abers_spread_max=0.12,
        max_sharpe_degradation_ratio=0.35,
        min_passed_gates_for_shadow=3,
    )
    tournament_service = ModelTournamentService(tournament_config)
    mock_notifier.present_tournament_report.return_value = TournamentVerdict.KEEP_CHAMPION

    use_case = RunWeekendAlphaEvolutionUseCase(
        math_alpha_agent=mock_agent,
        gatekeeper=mock_gatekeeper,
        pruning_service=pruning_service,
        tournament_service=tournament_service,
        tournament_notifier=mock_notifier,
        logger=mock_logger,
    )

    current_zoo = [f"zoo_{i}" for i in range(38)]
    req = WeekendAlphaEvolutionRequest(
        blindspots=[],
        decayed_alphas=[],
        current_zoo_features=current_zoo,
        decayed_feature_metrics={},
        target_count_per_cluster=1,
        candidate_series_map=None,
        default_series_data=_make_series_data(),
        tournament_eval_input=_make_tournament_input(),
    )

    result = await use_case.execute(req)
    assert result.verdict == TournamentVerdict.KEEP_CHAMPION
    assert len(result.generated_candidates) == 4
    assert len(result.passed_candidates) == 0
    assert result.active_zoo_features == current_zoo
    assert result.pruned_feature_ids == []


@pytest.mark.asyncio
async def test_run_weekend_alpha_evolution_prunes_overcapacity_zoo() -> None:
    """When new candidates push total features beyond 50, decayed features are pruned."""
    mock_agent = AsyncMock(spec=MathAlphaAgentPort)
    mock_gatekeeper = MagicMock(spec=StatisticalAlphaGatekeeper)
    mock_notifier = AsyncMock(spec=ModelTournamentNotifierPort)
    mock_logger = MagicMock(spec=StructuredBotLoggerPort)

    cand_new = _make_candidate("cand_new_win", CLUSTER_0_REGIME)

    async def mock_gen(cluster_name: str, **kwargs: object) -> list[SymbolicAlphaCandidate]:
        if cluster_name == CLUSTER_0_REGIME:
            return [cand_new]
        return []

    mock_agent.generate_candidates_for_cluster.side_effect = mock_gen

    mock_gatekeeper.evaluate_candidate.return_value = AlphaGateResult(
        alpha_id="cand_new_win",
        passed_all_gates=True,
        failed_stage=None,
        oos_rank_ic=0.035,
        oos_icir=0.6,
        t_statistic=2.7,
        is_regime_invariant=True,
        quantile_monotonicity_corr=0.85,
        max_corr_with_existing=0.2,
    )

    pruning_service = AlphaPruningService(35, 50, 0.012)
    tournament_config = ModelTournamentConfig(
        sharpe_delta_min=0.15,
        venn_abers_spread_max=0.12,
        max_sharpe_degradation_ratio=0.35,
        min_passed_gates_for_shadow=3,
    )
    tournament_service = ModelTournamentService(tournament_config)
    mock_notifier.present_tournament_report.return_value = (
        TournamentVerdict.PROMOTE_CHALLENGER_IMMEDIATELY
    )

    use_case = RunWeekendAlphaEvolutionUseCase(
        math_alpha_agent=mock_agent,
        gatekeeper=mock_gatekeeper,
        pruning_service=pruning_service,
        tournament_service=tournament_service,
        tournament_notifier=mock_notifier,
        logger=mock_logger,
    )

    # 50 features currently in zoo, with zoo_0 decayed
    current_zoo = [f"zoo_{i}" for i in range(50)]
    decayed_metrics = {"zoo_0": 0.003}

    req = WeekendAlphaEvolutionRequest(
        blindspots=[],
        decayed_alphas=["zoo_0"],
        current_zoo_features=current_zoo,
        decayed_feature_metrics=decayed_metrics,
        target_count_per_cluster=1,
        candidate_series_map=None,
        default_series_data=_make_series_data(),
        tournament_eval_input=_make_tournament_input(),
    )

    result = await use_case.execute(req)
    assert len(result.active_zoo_features) == 50
    assert result.pruned_feature_ids == ["zoo_0"]
    assert "cand_new_win" in result.active_zoo_features
    assert "zoo_0" not in result.active_zoo_features
