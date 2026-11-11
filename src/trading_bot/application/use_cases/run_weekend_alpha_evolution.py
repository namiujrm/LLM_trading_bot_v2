"""Run Weekend Alpha Evolution Use Case (Clean Architecture, Invariant #9).

Orchestrates the weekly quantitative alpha evolution pipeline:
1. Ingests weekly blindspots and decayed alphas.
2. Calls Math Alpha Agent across 4 specialized causal clusters:
   (CLUSTER_0_REGIME, CLUSTER_1_MICRO, CLUSTER_2_DERIV, CLUSTER_3_CROSS).
3. Evaluates all generated candidate expressions through the 4-Stage
   Statistical Gatekeeper, filtering those passing all stages.
4. Updates the Alpha Zoo via AlphaPruningService (enforcing the 35-50 feature cap).
5. Executes the Champion vs. Challenger tournament evaluation via ModelTournamentService.
6. Presents the comprehensive tournament comparison report via ModelTournamentNotifierPort.
7. Returns the final ModelTournamentReport and promotion verdict.

Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks.
Application layer depends ONLY on Domain and Ports.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from trading_bot.application.services.alpha_pruning_service import AlphaPruningService
from trading_bot.application.services.model_tournament_service import (
    ModelTournamentService,
    TournamentEvaluationInput,
)
from trading_bot.domain.entities.quant.symbolic_alpha import (
    CLUSTER_0_REGIME,
    CLUSTER_1_MICRO,
    CLUSTER_2_DERIV,
    CLUSTER_3_CROSS,
    AlphaGateResult,
    ModelTournamentReport,
    SymbolicAlphaCandidate,
    TournamentVerdict,
)
from trading_bot.domain.exceptions import (
    InsufficientDataException,
    MalformedModelResponseException,
)
from trading_bot.domain.services.quant.statistical_alpha_gatekeeper import (
    StatisticalAlphaGatekeeper,
)
from trading_bot.ports.logger_port import StructuredBotLoggerPort
from trading_bot.ports.math_alpha_agent_port import MathAlphaAgentPort
from trading_bot.ports.model_tournament_port import ModelTournamentNotifierPort


class CandidateSeriesData(BaseModel):
    """Time-series data container for statistical gatekeeper candidate screening."""

    alpha_oos_series: list[float]
    alpha_truncated_series: list[float]
    target_returns_oos: list[float]
    realized_vol_series: list[float]
    existing_alpha_matrix_oos: list[list[float]]


class WeekendAlphaEvolutionRequest(BaseModel):
    """Input parameters for the weekend alpha evolution cycle."""

    blindspots: list[dict[str, Any]]
    decayed_alphas: list[str]
    current_zoo_features: list[str]
    decayed_feature_metrics: dict[str, float]
    target_count_per_cluster: int
    candidate_series_map: dict[str, CandidateSeriesData] | None
    default_series_data: CandidateSeriesData | None
    tournament_eval_input: TournamentEvaluationInput


class WeekendAlphaEvolutionResult(BaseModel):
    """Comprehensive result of the weekend alpha evolution and tournament process."""

    tournament_report: ModelTournamentReport
    verdict: TournamentVerdict
    generated_candidates: list[SymbolicAlphaCandidate]
    passed_candidates: list[SymbolicAlphaCandidate]
    active_zoo_features: list[str]
    pruned_feature_ids: list[str]

    def __iter__(self) -> Any:
        """Allow tuple unpacking (tournament_report, verdict)."""
        return iter((self.tournament_report, self.verdict))


class RunWeekendAlphaEvolutionUseCase:
    """Orchestrates weekend alpha evolution, statistical gating, pruning, and tournament."""

    def __init__(
        self,
        math_alpha_agent: MathAlphaAgentPort,
        gatekeeper: StatisticalAlphaGatekeeper,
        pruning_service: AlphaPruningService,
        tournament_service: ModelTournamentService,
        tournament_notifier: ModelTournamentNotifierPort,
        logger: StructuredBotLoggerPort,
    ) -> None:
        self._math_alpha_agent: MathAlphaAgentPort = math_alpha_agent
        self._gatekeeper: StatisticalAlphaGatekeeper = gatekeeper
        self._pruning_service: AlphaPruningService = pruning_service
        self._tournament_service: ModelTournamentService = tournament_service
        self._tournament_notifier: ModelTournamentNotifierPort = tournament_notifier
        self._logger: StructuredBotLoggerPort = logger

    def _resolve_candidate_series(
        self,
        candidate: SymbolicAlphaCandidate,
        candidate_series_map: dict[str, CandidateSeriesData] | None,
        default_series_data: CandidateSeriesData | None,
    ) -> CandidateSeriesData:
        """Resolve screening series data for a specific alpha candidate."""
        if candidate_series_map is not None and candidate.alpha_id in candidate_series_map:
            return candidate_series_map[candidate.alpha_id]
        if default_series_data is not None:
            return default_series_data
        raise InsufficientDataException(
            f"No series evaluation data provided for candidate {candidate.alpha_id}"
        )

    async def execute(
        self,
        request: WeekendAlphaEvolutionRequest,
    ) -> WeekendAlphaEvolutionResult:
        """Execute the complete weekend alpha evolution and tournament workflow."""
        self._logger.info(
            "Starting weekend alpha evolution cycle",
            {
                "blindspots_count": len(request.blindspots),
                "decayed_alphas_count": len(request.decayed_alphas),
                "current_zoo_count": len(request.current_zoo_features),
            },
        )

        # 1. Generate candidates across the 4 specialized causal clusters
        clusters: list[str] = [
            CLUSTER_0_REGIME,
            CLUSTER_1_MICRO,
            CLUSTER_2_DERIV,
            CLUSTER_3_CROSS,
        ]
        all_candidates: list[SymbolicAlphaCandidate] = []

        for cluster_name in clusters:
            try:
                cluster_candidates = await self._math_alpha_agent.generate_candidates_for_cluster(
                    cluster_name=cluster_name,
                    blindspots=request.blindspots,
                    decayed_alphas=request.decayed_alphas,
                    target_count=request.target_count_per_cluster,
                )
                all_candidates.extend(cluster_candidates)
                self._logger.info(
                    "Generated alpha candidates for cluster",
                    {
                        "cluster": cluster_name,
                        "count": len(cluster_candidates),
                    },
                )
            except Exception as exc:
                self._logger.warning(
                    "Failed to generate alpha candidates for cluster",
                    {
                        "cluster": cluster_name,
                        "error": str(exc),
                    },
                )

        if len(all_candidates) == 0:
            raise MalformedModelResponseException(
                "Failed to generate any valid alpha candidates across all causal clusters."
            )

        # 2. Evaluate all candidates through StatisticalAlphaGatekeeper
        passed_candidates: list[SymbolicAlphaCandidate] = []
        for candidate in all_candidates:
            series_data = self._resolve_candidate_series(
                candidate=candidate,
                candidate_series_map=request.candidate_series_map,
                default_series_data=request.default_series_data,
            )
            gate_result: AlphaGateResult = self._gatekeeper.evaluate_candidate(
                alpha_id=candidate.alpha_id,
                alpha_oos_series=series_data.alpha_oos_series,
                alpha_truncated_series=series_data.alpha_truncated_series,
                target_returns_oos=series_data.target_returns_oos,
                realized_vol_series=series_data.realized_vol_series,
                expected_monotonic_sign=candidate.long_monotonic_prior,
                existing_alpha_matrix_oos=series_data.existing_alpha_matrix_oos,
            )

            if gate_result.passed_all_gates:
                passed_candidates.append(candidate)
                self._logger.info(
                    "Alpha candidate passed statistical gates",
                    {
                        "alpha_id": candidate.alpha_id,
                        "cluster": candidate.causal_cluster,
                        "oos_rank_ic": gate_result.oos_rank_ic,
                        "oos_icir": gate_result.oos_icir,
                        "mono_corr": gate_result.quantile_monotonicity_corr,
                    },
                )
            else:
                self._logger.warning(
                    "Alpha candidate rejected by statistical gatekeeper",
                    {
                        "alpha_id": candidate.alpha_id,
                        "failed_stage": gate_result.failed_stage,
                    },
                )

        # 3. Update Alpha Zoo via AlphaPruningService
        pruning_result = self._pruning_service.prune_and_update_zoo(
            current_zoo_features=request.current_zoo_features,
            decayed_feature_metrics=request.decayed_feature_metrics,
            new_passed_candidates=passed_candidates,
        )
        self._logger.info(
            "Alpha zoo capacity updated",
            {
                "active_count": len(pruning_result.active_features),
                "pruned_count": len(pruning_result.pruned_features),
                "pruned_features": pruning_result.pruned_features,
            },
        )

        # 4. Run Champion vs. Challenger tournament comparison
        tournament_report: ModelTournamentReport = self._tournament_service.run_tournament(
            evaluation_input=request.tournament_eval_input
        )
        self._logger.info(
            "Model tournament evaluation complete",
            {
                "is_promotable": tournament_report.is_promotable,
                "recommended_verdict": str(tournament_report.recommended_verdict),
            },
        )

        # 5. Present report via ModelTournamentNotifierPort
        verdict: TournamentVerdict = await self._tournament_notifier.present_tournament_report(
            report=tournament_report
        )
        self._logger.info(
            "Tournament verdict received from notifier",
            {"verdict": str(verdict)},
        )

        return WeekendAlphaEvolutionResult(
            tournament_report=tournament_report,
            verdict=verdict,
            generated_candidates=all_candidates,
            passed_candidates=passed_candidates,
            active_zoo_features=pruning_result.active_features,
            pruned_feature_ids=pruning_result.pruned_features,
        )
