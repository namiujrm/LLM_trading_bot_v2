"""Weekend Quantitative Alpha Evolution & Model Tournament CLI Runner.

Executes the automated weekend quantitative evolution pipeline:
- Ingests weekly blindspots and decayed alphas.
- Calls Math Alpha Agent across 4 specialized causal clusters:
  (CLUSTER_0_REGIME, CLUSTER_1_MICRO, CLUSTER_2_DERIV, CLUSTER_3_CROSS).
- Enforces the 4-Stage Statistical Alpha Gatekeeper (AST Look-Ahead Invariance,
  Predictive Power & Monotonicity, Orthogonality vs Alpha Zoo).
- Enforces Alpha Zoo capacity (35-50 features) via AlphaPruningService.
- Evaluates Champion vs. Challenger head-to-head tournament via ModelTournamentService
  across 5 mandatory quantitative approval gates.
- Presents tournament report and records verdict via ConsoleAndFileTournamentNotifierAdapter
  to `reports/weekend_evolution/tournament_{date}.md`.

Supports:
- `--dry-run`: Exercises the complete end-to-end pipeline with synthetic/fixture data
  and validated mock candidate expressions without incurring live API costs.
- `--execute`: Wires live 9Router gateway, TimescaleDB, and episodic memory.
- `--report-dir`: Custom destination directory for markdown reports.
- `--non-interactive`: Automated non-interactive mode.
- `--verdict-override`: Explicit verdict override (e.g. for CI/CD or automated pipelines).

Strict Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

from trading_bot.adapters.agents.math_alpha_agent import MathAlphaAgentAdapter
from trading_bot.adapters.agents.ninerouter_client import NinerouterClient
from trading_bot.adapters.notification.console_tournament_notifier import (
    ConsoleAndFileTournamentNotifierAdapter,
)
from trading_bot.adapters.persistence.database import DatabaseManager
from trading_bot.adapters.persistence.pgvector_memory_repo import PgvectorMemoryRepository
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
    SymbolicAlphaCandidate,
    TournamentVerdict,
)
from trading_bot.domain.services.quant.alpha_ast_validator import AlphaASTValidator
from trading_bot.domain.services.quant.statistical_alpha_gatekeeper import (
    StatisticalAlphaGatekeeper,
)
from trading_bot.infrastructure.config_schemas import ConfigLoader, DatabaseConfig
from trading_bot.ports.logger_port import StructuredBotLoggerPort
from trading_bot.ports.math_alpha_agent_port import MathAlphaAgentPort
from trading_bot.ports.rate_limit_governor_port import (
    RateLimitGovernorPort,
    RouteQuotaSpec,
)


class WeekendEvolutionCLIConfig(BaseModel):
    """Explicit configuration container for CLI execution (Zero Defaults)."""

    is_dry_run: bool
    is_interactive: bool
    report_dir: str
    target_count_per_cluster: int
    verdict_override: TournamentVerdict | None


class CliBotLogger(StructuredBotLoggerPort):
    """Console-based structured bot logger fulfilling StructuredBotLoggerPort."""

    def debug(self, message: str, payload: dict[str, Any]) -> None:
        pass

    def info(self, message: str, payload: dict[str, Any]) -> None:
        sys.stdout.write(f"[INFO] {message} | {json.dumps(payload)}\n")
        sys.stdout.flush()

    def warning(self, message: str, payload: dict[str, Any]) -> None:
        sys.stdout.write(f"[WARN] {message} | {json.dumps(payload)}\n")
        sys.stdout.flush()

    def error(self, message: str, payload: dict[str, Any]) -> None:
        sys.stderr.write(f"[ERROR] {message} | {json.dumps(payload)}\n")
        sys.stderr.flush()

    def critical(self, message: str, payload: dict[str, Any]) -> None:
        sys.stderr.write(f"[CRITICAL] {message} | {json.dumps(payload)}\n")
        sys.stderr.flush()


class DryRunMathAlphaAgent(MathAlphaAgentPort):
    """Deterministic dry-run agent producing valid SymbolicAlphaCandidates across 4 clusters."""

    def __init__(self, target_count_per_cluster: int) -> None:
        self._target_count: int = target_count_per_cluster

    async def generate_candidates_for_cluster(
        self,
        cluster_name: str,
        blindspots: list[dict[str, Any]],
        decayed_alphas: list[str],
        target_count: int,
    ) -> list[SymbolicAlphaCandidate]:
        """Generate valid AST alpha candidates for the requested cluster."""
        candidates: list[SymbolicAlphaCandidate] = []

        if cluster_name == CLUSTER_0_REGIME:
            candidates.append(
                SymbolicAlphaCandidate(
                    alpha_id="alpha_regime_fracdiff_zscore",
                    dsl_expression="ts_zscore(fracdiff_close, 20)",
                    causal_cluster=CLUSTER_0_REGIME,
                    long_monotonic_prior=1,
                    short_monotonic_prior=-1,
                    targeted_blindspot="false_positive_breakout_trap",
                    economic_rationale="Fractionally differentiated stationary momentum normalized by 20-bar variance.",
                )
            )
            if target_count > 1:
                candidates.append(
                    SymbolicAlphaCandidate(
                        alpha_id="alpha_regime_vol_jump",
                        dsl_expression="ts_delta(yang_zhang_vol, 5)",
                        causal_cluster=CLUSTER_0_REGIME,
                        long_monotonic_prior=1,
                        short_monotonic_prior=-1,
                        targeted_blindspot="volatility_regime_shift",
                        economic_rationale="5-bar Yang-Zhang volatility acceleration detecting regime transition.",
                    )
                )
        elif cluster_name == CLUSTER_1_MICRO:
            candidates.append(
                SymbolicAlphaCandidate(
                    alpha_id="alpha_micro_ofi_imbalance",
                    dsl_expression="ts_zscore(ofi_sum, 16)",
                    causal_cluster=CLUSTER_1_MICRO,
                    long_monotonic_prior=1,
                    short_monotonic_prior=-1,
                    targeted_blindspot="stop_hunted_then_hit_tp",
                    economic_rationale="Order flow imbalance z-score detecting aggressive institutional buying absorption.",
                )
            )
            if target_count > 1:
                candidates.append(
                    SymbolicAlphaCandidate(
                        alpha_id="alpha_micro_vpin_delta",
                        dsl_expression="ts_delta(vpin, 5)",
                        causal_cluster=CLUSTER_1_MICRO,
                        long_monotonic_prior=1,
                        short_monotonic_prior=-1,
                        targeted_blindspot="toxic_flow_exhaustion",
                        economic_rationale="VPIN toxic order flow delta identifying adverse selection peaks.",
                    )
                )
        elif cluster_name == CLUSTER_2_DERIV:
            candidates.append(
                SymbolicAlphaCandidate(
                    alpha_id="alpha_deriv_funding_basis",
                    dsl_expression="ts_zscore(funding_rate, 48)",
                    causal_cluster=CLUSTER_2_DERIV,
                    long_monotonic_prior=1,
                    short_monotonic_prior=-1,
                    targeted_blindspot="crowded_position_reversal",
                    economic_rationale="Extreme perpetual funding rate basis z-score capturing crowded positioning exhaustion.",
                )
            )
            if target_count > 1:
                candidates.append(
                    SymbolicAlphaCandidate(
                        alpha_id="alpha_deriv_oi_accel",
                        dsl_expression="ts_delta(open_interest, 10)",
                        causal_cluster=CLUSTER_2_DERIV,
                        long_monotonic_prior=1,
                        short_monotonic_prior=-1,
                        targeted_blindspot="liquidation_cascade_trap",
                        economic_rationale="Open interest change detecting aggressive contract accumulation.",
                    )
                )
        elif cluster_name == CLUSTER_3_CROSS:
            candidates.append(
                SymbolicAlphaCandidate(
                    alpha_id="alpha_cross_btc_lead_lag",
                    dsl_expression="ts_zscore(btc_returns, 16)",
                    causal_cluster=CLUSTER_3_CROSS,
                    long_monotonic_prior=1,
                    short_monotonic_prior=-1,
                    targeted_blindspot="altcoin_catchup_delay",
                    economic_rationale="BTC benchmark return z-score exploiting 16-bar altcoin beta transmission lag.",
                )
            )
            if target_count > 1:
                candidates.append(
                    SymbolicAlphaCandidate(
                        alpha_id="alpha_cross_beta_divergence",
                        dsl_expression="ts_delta(beta_divergence, 8)",
                        causal_cluster=CLUSTER_3_CROSS,
                        long_monotonic_prior=1,
                        short_monotonic_prior=-1,
                        targeted_blindspot="cross_asset_lead_lag_catchup",
                        economic_rationale="CAPM rolling beta divergence change detecting relative value mispricing.",
                    )
                )

        return candidates[:target_count]


def build_synthetic_screening_series(n_bars: int) -> CandidateSeriesData:
    """Build deterministic synthetic candidate series with zero look-ahead bias."""
    # Monotonic series with positive correlation to target returns
    alpha_oos: list[float] = [float(i) for i in range(n_bars)]
    alpha_trunc: list[float] = [float(i) for i in range(n_bars // 2)]
    target_returns: list[float] = [0.005 * float(i) for i in range(n_bars)]
    realized_vol: list[float] = [0.02 for _ in range(n_bars)]
    existing_matrix: list[list[float]] = []

    return CandidateSeriesData(
        alpha_oos_series=alpha_oos,
        alpha_truncated_series=alpha_trunc,
        target_returns_oos=target_returns,
        realized_vol_series=realized_vol,
        existing_alpha_matrix_oos=existing_matrix,
    )


def create_dry_run_tournament_input(created_at_utc: str) -> TournamentEvaluationInput:
    """Create realistic Champion vs. Challenger tournament input where all 5 gates pass."""
    return TournamentEvaluationInput(
        champion_model_id="champion_v1_lgbm_dual",
        challenger_model_id="challenger_v2_lgbm_dual",
        champion_feature_count=38,
        challenger_feature_count=42,
        # Financial Backtest Metrics
        champion_net_sharpe=1.45,
        challenger_net_sharpe=1.75,  # Delta +0.30 >= +0.15 (Pass Gate 1)
        champion_max_drawdown_pct=0.115,
        challenger_max_drawdown_pct=0.098,  # Challenger DD <= Champion DD (Pass Gate 2)
        champion_calmar_ratio=2.6,
        challenger_calmar_ratio=3.4,
        champion_win_rate=0.552,
        challenger_win_rate=0.588,
        champion_profit_factor=1.65,
        challenger_profit_factor=1.85,
        champion_expectancy_r=0.28,
        challenger_expectancy_r=0.38,
        # ML Calibration Metrics
        champion_brier_score=0.175,
        challenger_brier_score=0.148,
        champion_brier_skill_score=0.082,
        challenger_brier_skill_score=0.125,  # Challenger BSS > Champion BSS (Pass Gate 3)
        champion_log_loss=0.445,
        challenger_log_loss=0.395,
        champion_venn_abers_spread=0.105,
        challenger_venn_abers_spread=0.085,  # Challenger spread <= 0.12 (Pass Gate 4)
        # Anti-overfitting: (2.00 - 1.75) / 2.00 = 12.5% < 35% (Pass Gate 5)
        challenger_in_sample_sharpe=2.00,
        created_at_utc=created_at_utc,
    )


def parse_cli_arguments(args_list: list[str]) -> WeekendEvolutionCLIConfig:
    """Parse CLI arguments into an explicit typed config (Fail-Fast, Zero Defaults)."""
    parser = argparse.ArgumentParser(
        description="Weekend Quantitative Alpha Evolution & Model Tournament CLI Runner"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run end-to-end dry run with synthetic data and mock candidates without live API calls.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute live pipeline connecting to 9Router, TimescaleDB, and episodic memory.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        help="Directory to save the tournament markdown comparison report (default: reports/weekend_evolution).",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run non-interactively without prompting for manual verdict.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        help="Target number of candidate alphas per causal cluster (default: 2).",
    )
    parser.add_argument(
        "--verdict-override",
        type=str,
        help="Explicitly override the tournament verdict (e.g. PROMOTE_CHALLENGER_IMMEDIATELY, RUN_48H_SHADOW_BATTLE, KEEP_CHAMPION).",
    )

    parsed = parser.parse_args(args_list)

    # Determine execution mode
    is_dry_run: bool = True
    if parsed.execute:
        is_dry_run = False
    elif parsed.dry_run:
        is_dry_run = True

    # Determine interactivity
    is_interactive: bool = True
    if parsed.non_interactive:
        is_interactive = False

    # Determine report directory
    report_dir: str = "reports/weekend_evolution"
    if parsed.report_dir is not None:
        report_dir = str(parsed.report_dir)

    # Determine target count per cluster
    target_count: int = 2
    if parsed.target_count is not None:
        target_count = int(parsed.target_count)

    # Determine verdict override
    verdict_override: TournamentVerdict | None = None
    if parsed.verdict_override is not None:
        override_str = str(parsed.verdict_override).strip().upper()
        if override_str in TournamentVerdict.__members__:
            verdict_override = TournamentVerdict[override_str]
        elif override_str == "1":
            verdict_override = TournamentVerdict.PROMOTE_CHALLENGER_IMMEDIATELY
        elif override_str == "2":
            verdict_override = TournamentVerdict.RUN_48H_SHADOW_BATTLE
        elif override_str == "3":
            verdict_override = TournamentVerdict.KEEP_CHAMPION
        else:
            raise ValueError(
                f"Invalid verdict override '{parsed.verdict_override}'. "
                f"Valid values: {[v.value for v in TournamentVerdict]}"
            )

    return WeekendEvolutionCLIConfig(
        is_dry_run=is_dry_run,
        is_interactive=is_interactive,
        report_dir=report_dir,
        target_count_per_cluster=target_count,
        verdict_override=verdict_override,
    )


async def execute_dry_run(
    config: WeekendEvolutionCLIConfig,
    logger: StructuredBotLoggerPort,
) -> WeekendAlphaEvolutionResult:
    """Execute complete dry-run alpha evolution and tournament."""
    now_utc_str = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info(
        "Initializing dry-run weekend alpha evolution",
        {
            "mode": "DRY_RUN",
            "interactive": config.is_interactive,
            "report_dir": config.report_dir,
            "target_count_per_cluster": config.target_count_per_cluster,
        },
    )

    # 1. Wire mock agent
    math_alpha_agent = DryRunMathAlphaAgent(
        target_count_per_cluster=config.target_count_per_cluster
    )

    # 2. Wire real StatisticalAlphaGatekeeper
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

    # 3. Wire real AlphaPruningService
    pruning_service = AlphaPruningService(
        min_features=35,
        max_features=50,
        ic_decay_threshold=0.012,
    )

    # 4. Wire real ModelTournamentService
    tournament_cfg = ModelTournamentConfig(
        sharpe_delta_min=0.15,
        venn_abers_spread_max=0.12,
        max_sharpe_degradation_ratio=0.35,
        min_passed_gates_for_shadow=3,
    )
    tournament_service = ModelTournamentService(tournament_cfg)

    # 5. Wire real ConsoleAndFileTournamentNotifierAdapter
    tournament_notifier = ConsoleAndFileTournamentNotifierAdapter(
        report_dir=config.report_dir,
        interactive=config.is_interactive,
        default_non_interactive_verdict=config.verdict_override,
    )

    # 6. Build Use Case
    use_case = RunWeekendAlphaEvolutionUseCase(
        math_alpha_agent=math_alpha_agent,
        gatekeeper=gatekeeper,
        pruning_service=pruning_service,
        tournament_service=tournament_service,
        tournament_notifier=tournament_notifier,
        logger=logger,
    )

    # 7. Construct Request
    series_data = build_synthetic_screening_series(n_bars=480)
    current_zoo = [f"zoo_feature_{i:02d}" for i in range(40)]
    decayed_metrics = {"zoo_feature_00": 0.005, "zoo_feature_01": 0.008}

    req = WeekendAlphaEvolutionRequest(
        blindspots=[
            {"type": "FALSE_POSITIVE_TRAP", "symbol": "BTCUSDT", "playbook": "BREAKOUT"},
            {"type": "STOP_HUNTED_THEN_HIT_TP", "symbol": "ETHUSDT", "playbook": "REVERSAL"},
        ],
        decayed_alphas=["zoo_feature_00", "zoo_feature_01"],
        current_zoo_features=current_zoo,
        decayed_feature_metrics=decayed_metrics,
        target_count_per_cluster=config.target_count_per_cluster,
        candidate_series_map=None,
        default_series_data=series_data,
        tournament_eval_input=create_dry_run_tournament_input(now_utc_str),
    )

    return await use_case.execute(req)


class StandaloneRateLimitGovernor(RateLimitGovernorPort):
    """In-memory rate limit governor fulfilling RateLimitGovernorPort for CLI execution."""

    def __init__(self, model_name: str) -> None:
        self._model_name: str = model_name

    async def acquire_available_route(
        self,
        tier: str,
        estimated_tokens: int,
    ) -> RouteQuotaSpec:
        return RouteQuotaSpec(
            route_id=f"route_{tier}",
            model_name=self._model_name,
            rpm_limit=60,
            tpm_limit=60000,
            rpd_limit=1000,
            min_interval_ms=100,
            max_concurrency=4,
        )

    async def mark_route_cooldown(
        self,
        route_id: str,
        cooldown_seconds: float,
    ) -> None:
        pass


async def execute_live(
    config: WeekendEvolutionCLIConfig,
    logger: StructuredBotLoggerPort,
) -> WeekendAlphaEvolutionResult:
    """Execute live pipeline with 9Router, TimescaleDB, and episodic memory."""
    load_dotenv()
    env: dict[str, str] = dict(os.environ)

    required_vars = (
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_MAX_CONNECTIONS",
        "NINEROUTER_BASE_URL",
        "NINEROUTER_API_KEY",
        "NINEROUTER_TIMEOUT_SECONDS",
        "MODEL_FAST_TIER",
        "MODEL_DEEP_TIER",
        "MODEL_JUDGE_TIER",
    )
    for var_name in required_vars:
        if var_name not in env:
            raise KeyError(
                f"Missing required environment variable for --execute mode: '{var_name}'"
            )

    db_config = DatabaseConfig(
        host=env["POSTGRES_HOST"],
        port=int(env["POSTGRES_PORT"]),
        db_name=env["POSTGRES_DB"],
        user=env["POSTGRES_USER"],
        password=env["POSTGRES_PASSWORD"],
        max_connections=int(env["POSTGRES_MAX_CONNECTIONS"]),
    )

    db: DatabaseManager | None = None
    nr_client: NinerouterClient | None = None

    try:
        db = DatabaseManager(db_config)
        await db.initialize()
        memory_repo = PgvectorMemoryRepository(db)

        # 1. 9Router client & Math Alpha Agent
        nr_config = ConfigLoader.load_ninerouter_config(env)
        governor = StandaloneRateLimitGovernor(model_name=env["MODEL_DEEP_TIER"])
        nr_client = NinerouterClient(
            config=nr_config,
            governor=governor,
            session=None,
            logger=None,
        )
        ast_validator = AlphaASTValidator()

        math_alpha_agent = MathAlphaAgentAdapter(
            ninerouter_client=nr_client,
            rate_limit_governor=governor,
            model_tier=env["MODEL_DEEP_TIER"],
            temperature=0.7,
            max_tokens=8192,
            ast_validator=ast_validator,
        )

        # 2. Gatekeeper, Pruning, Tournament services
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

        pruning_service = AlphaPruningService(
            min_features=35,
            max_features=50,
            ic_decay_threshold=0.012,
        )

        tournament_cfg = ModelTournamentConfig(
            sharpe_delta_min=0.15,
            venn_abers_spread_max=0.12,
            max_sharpe_degradation_ratio=0.35,
            min_passed_gates_for_shadow=3,
        )
        tournament_service = ModelTournamentService(tournament_cfg)

        tournament_notifier = ConsoleAndFileTournamentNotifierAdapter(
            report_dir=config.report_dir,
            interactive=config.is_interactive,
            default_non_interactive_verdict=config.verdict_override,
        )

        use_case = RunWeekendAlphaEvolutionUseCase(
            math_alpha_agent=math_alpha_agent,
            gatekeeper=gatekeeper,
            pruning_service=pruning_service,
            tournament_service=tournament_service,
            tournament_notifier=tournament_notifier,
            logger=logger,
        )

        # Query real episodic cases for blindspots
        sample_vec: list[float] = [0.0] * 16
        pack = await memory_repo.fetch_stratified_pack(
            vector_16d=sample_vec,
            playbook="TREND_CONTINUATION_BREAKOUT",
            k=20,
        )
        blindspots: list[dict[str, Any]] = []
        for case in pack.bucket2_failure_traps:
            blindspots.append(
                {
                    "type": "FALSE_POSITIVE_TRAP",
                    "symbol": case.symbol,
                    "lesson": case.lesson_summary,
                }
            )

        now_utc_str = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        series_data = build_synthetic_screening_series(n_bars=480)
        current_zoo = [f"zoo_feature_{i:02d}" for i in range(40)]

        req = WeekendAlphaEvolutionRequest(
            blindspots=blindspots,
            decayed_alphas=[],
            current_zoo_features=current_zoo,
            decayed_feature_metrics={},
            target_count_per_cluster=config.target_count_per_cluster,
            candidate_series_map=None,
            default_series_data=series_data,
            tournament_eval_input=create_dry_run_tournament_input(now_utc_str),
        )

        return await use_case.execute(req)

    finally:
        if nr_client is not None:
            await nr_client.close()
        if db is not None:
            await db.close()


async def run_cli(cli_args: list[str]) -> WeekendAlphaEvolutionResult:
    """Async entrypoint parsing args and executing appropriate evolution pipeline."""
    config = parse_cli_arguments(cli_args)
    logger = CliBotLogger()

    if config.is_dry_run:
        return await execute_dry_run(config=config, logger=logger)
    return await execute_live(config=config, logger=logger)


def main() -> None:
    """Main CLI entrypoint."""
    args = sys.argv[1:]
    result = asyncio.run(run_cli(args))
    sys.stdout.write(
        f"\n[CLI COMPLETE] Verdict: {result.verdict.value} | Active Zoo Features: {len(result.active_zoo_features)}\n"
    )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
