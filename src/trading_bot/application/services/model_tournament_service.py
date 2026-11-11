"""Model Tournament Service (Clean Architecture, Invariant #9).

Evaluates Champion vs. Challenger model tournament across Financial backtest metrics
and ML calibration metrics, enforcing the 5 Mandatory Quantitative Approval Gates:
1. Financial outperformance: Sharpe delta >= 0.15
2. Capital risk preservation: Challenger Max Drawdown <= Champion Max Drawdown
3. Brier Skill Score superiority: Challenger BSS > Champion BSS
4. Venn-Abers calibration stability: Challenger spread <= 0.12
5. Anti-overfitting discipline: In-Sample to OOS Sharpe degradation < 35%

Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks.
Application layer depends ONLY on domain entities and standard library.
"""

from __future__ import annotations

from pydantic import BaseModel

from trading_bot.domain.entities.quant.symbolic_alpha import (
    ModelTournamentReport,
    TournamentVerdict,
)


class ModelTournamentConfig(BaseModel):
    """Configuration thresholds for the 5 quantitative tournament approval gates."""

    sharpe_delta_min: float
    venn_abers_spread_max: float
    max_sharpe_degradation_ratio: float
    min_passed_gates_for_shadow: int


class TournamentEvaluationInput(BaseModel):
    """Input parameters comparing Champion and Challenger performance metrics."""

    champion_model_id: str
    challenger_model_id: str
    champion_feature_count: int
    challenger_feature_count: int
    # Financial Backtest Metrics
    champion_net_sharpe: float
    challenger_net_sharpe: float
    champion_max_drawdown_pct: float
    challenger_max_drawdown_pct: float
    champion_calmar_ratio: float
    challenger_calmar_ratio: float
    champion_win_rate: float
    challenger_win_rate: float
    champion_profit_factor: float
    challenger_profit_factor: float
    champion_expectancy_r: float
    challenger_expectancy_r: float
    # ML Calibration Metrics
    champion_brier_score: float
    challenger_brier_score: float
    champion_brier_skill_score: float
    challenger_brier_skill_score: float
    champion_log_loss: float
    challenger_log_loss: float
    champion_venn_abers_spread: float
    challenger_venn_abers_spread: float
    # In-Sample Sharpe of Challenger for overfitting calculation
    challenger_in_sample_sharpe: float
    # Report Timestamp
    created_at_utc: str


class ModelTournamentService:
    """Evaluator for Champion vs. Challenger head-to-head backtest and calibration gates."""

    def __init__(self, config: ModelTournamentConfig) -> None:
        self._config: ModelTournamentConfig = config

    @property
    def config(self) -> ModelTournamentConfig:
        return self._config

    def run_tournament(
        self,
        evaluation_input: TournamentEvaluationInput,
    ) -> ModelTournamentReport:
        """Evaluate Champion vs. Challenger metrics against the 5 mandatory approval gates."""
        inp = evaluation_input

        # 1. Gate 1: Sharpe outperformance
        gate_1_sharpe: bool = inp.challenger_net_sharpe >= (
            inp.champion_net_sharpe + self._config.sharpe_delta_min - 1e-9
        )

        # 2. Gate 2: Capital risk preservation (Challenger DD is not worse than Champion DD)
        gate_2_drawdown: bool = inp.challenger_max_drawdown_pct <= (
            inp.champion_max_drawdown_pct + 1e-9
        )

        # 3. Gate 3: Brier Skill Score superiority
        gate_3_bss: bool = inp.challenger_brier_skill_score > inp.champion_brier_skill_score

        # 4. Gate 4: Venn-Abers calibration spread
        gate_4_venn: bool = inp.challenger_venn_abers_spread <= (
            self._config.venn_abers_spread_max + 1e-9
        )

        # 5. Gate 5: Anti-overfitting discipline
        if inp.challenger_in_sample_sharpe > 0.0:
            sharpe_degradation: float = (
                inp.challenger_in_sample_sharpe - inp.challenger_net_sharpe
            ) / inp.challenger_in_sample_sharpe
        else:
            if inp.challenger_net_sharpe >= inp.challenger_in_sample_sharpe:
                sharpe_degradation = 0.0
            else:
                sharpe_degradation = 1.0

        gate_5_overfitting: bool = sharpe_degradation < self._config.max_sharpe_degradation_ratio

        checklist: dict[str, bool] = {
            "gate_1_sharpe_outperformance": gate_1_sharpe,
            "gate_2_max_drawdown_preservation": gate_2_drawdown,
            "gate_3_brier_skill_score": gate_3_bss,
            "gate_4_venn_abers_calibration": gate_4_venn,
            "gate_5_anti_overfitting": gate_5_overfitting,
        }

        passed_count: int = sum(1 for passed in checklist.values() if passed is True)

        is_promotable: bool
        recommended_verdict: TournamentVerdict
        verdict_rationale: str

        if passed_count == 5:
            is_promotable = True
            recommended_verdict = TournamentVerdict.PROMOTE_CHALLENGER_IMMEDIATELY
            verdict_rationale = (
                f"All 5 mandatory quantitative approval gates passed. Challenger outperforms Champion "
                f"by Sharpe +{round(inp.challenger_net_sharpe - inp.champion_net_sharpe, 4)} "
                f"(threshold +{self._config.sharpe_delta_min}), preserves capital risk (DD {round(inp.challenger_max_drawdown_pct, 4)} <= {round(inp.champion_max_drawdown_pct, 4)}), "
                f"superior BSS ({round(inp.challenger_brier_skill_score, 4)} > {round(inp.champion_brier_skill_score, 4)}), "
                f"tight Venn-Abers spread ({round(inp.challenger_venn_abers_spread, 4)} <= {self._config.venn_abers_spread_max}), "
                f"and low degradation ({round(sharpe_degradation * 100, 2)}% < {round(self._config.max_sharpe_degradation_ratio * 100, 2)}%). "
                f"Recommended verdict: PROMOTE_CHALLENGER_IMMEDIATELY."
            )
        elif not gate_2_drawdown:
            is_promotable = False
            recommended_verdict = TournamentVerdict.KEEP_CHAMPION
            verdict_rationale = (
                f"Promotion blocked by Gate 2 (Capital Risk Preservation failure): Challenger max drawdown "
                f"({round(inp.challenger_max_drawdown_pct, 4)}) exceeds Champion max drawdown ({round(inp.champion_max_drawdown_pct, 4)}). "
                f"Capital preservation is paramount. Recommended verdict: KEEP_CHAMPION."
            )
        elif not gate_5_overfitting:
            is_promotable = False
            recommended_verdict = TournamentVerdict.KEEP_CHAMPION
            verdict_rationale = (
                f"Promotion blocked by Gate 5 (Anti-overfitting discipline failure): In-Sample to OOS Sharpe degradation "
                f"({round(sharpe_degradation * 100, 2)}%) exceeds limit ({round(self._config.max_sharpe_degradation_ratio * 100, 2)}%). "
                f"Model exhibits severe out-of-sample decay. Recommended verdict: KEEP_CHAMPION."
            )
        elif passed_count >= self._config.min_passed_gates_for_shadow:
            is_promotable = False
            recommended_verdict = TournamentVerdict.RUN_48H_SHADOW_BATTLE
            verdict_rationale = (
                f"Borderline performance: {passed_count} of 5 approval gates passed. Core risk preservation "
                f"and anti-overfitting gates passed, but Sharpe, BSS, or calibration spread did not meet immediate promotion thresholds. "
                f"Recommended verdict: RUN_48H_SHADOW_BATTLE."
            )
        else:
            is_promotable = False
            recommended_verdict = TournamentVerdict.KEEP_CHAMPION
            verdict_rationale = (
                f"Challenger rejected: Only {passed_count} of 5 approval gates passed. Insufficient evidence of model superiority. "
                f"Recommended verdict: KEEP_CHAMPION."
            )

        return ModelTournamentReport(
            created_at_utc=inp.created_at_utc,
            champion_model_id=inp.champion_model_id,
            challenger_model_id=inp.challenger_model_id,
            champion_feature_count=inp.champion_feature_count,
            challenger_feature_count=inp.challenger_feature_count,
            champion_net_sharpe=inp.champion_net_sharpe,
            challenger_net_sharpe=inp.challenger_net_sharpe,
            champion_max_drawdown_pct=inp.champion_max_drawdown_pct,
            challenger_max_drawdown_pct=inp.challenger_max_drawdown_pct,
            champion_calmar_ratio=inp.champion_calmar_ratio,
            challenger_calmar_ratio=inp.challenger_calmar_ratio,
            champion_win_rate=inp.champion_win_rate,
            challenger_win_rate=inp.challenger_win_rate,
            champion_profit_factor=inp.champion_profit_factor,
            challenger_profit_factor=inp.challenger_profit_factor,
            champion_expectancy_r=inp.champion_expectancy_r,
            challenger_expectancy_r=inp.challenger_expectancy_r,
            champion_brier_score=inp.champion_brier_score,
            challenger_brier_score=inp.challenger_brier_score,
            champion_brier_skill_score=inp.champion_brier_skill_score,
            challenger_brier_skill_score=inp.challenger_brier_skill_score,
            champion_log_loss=inp.champion_log_loss,
            challenger_log_loss=inp.challenger_log_loss,
            champion_venn_abers_spread=inp.champion_venn_abers_spread,
            challenger_venn_abers_spread=inp.challenger_venn_abers_spread,
            is_promotable=is_promotable,
            recommended_verdict=recommended_verdict,
            approval_checklist=checklist,
            verdict_rationale=verdict_rationale,
        )
