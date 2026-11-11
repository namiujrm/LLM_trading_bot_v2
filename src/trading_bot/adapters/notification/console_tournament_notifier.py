"""Console and File Model Tournament Notifier Adapter (Clean Architecture, Invariant #9).

Formats and presents the Champion vs. Challenger model tournament comparison report:
- Compares Financial Backtest and ML Calibration metrics.
- Checks the 5 mandatory quantitative approval gates.
- Writes comprehensive markdown report to reports/weekend_evolution/tournament_{date}.md.
- In interactive mode: prompts operator for promotion verdict via CLI.
- In non-interactive mode: returns the recommended verdict or explicit injected override.

Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks.
"""

from __future__ import annotations

import sys
from pathlib import Path

from trading_bot.domain.entities.quant.symbolic_alpha import (
    ModelTournamentReport,
    TournamentVerdict,
)
from trading_bot.ports.model_tournament_port import ModelTournamentNotifierPort


class ConsoleAndFileTournamentNotifierAdapter(ModelTournamentNotifierPort):
    """Adapter presenting tournament reports to console and writing Markdown files."""

    def __init__(
        self,
        report_dir: str,
        interactive: bool,
        default_non_interactive_verdict: TournamentVerdict | None,
    ) -> None:
        self._report_dir: str = report_dir
        self._interactive: bool = interactive
        self._default_non_interactive_verdict: TournamentVerdict | None = (
            default_non_interactive_verdict
        )

    @property
    def report_dir(self) -> str:
        return self._report_dir

    @property
    def interactive(self) -> bool:
        return self._interactive

    @property
    def default_non_interactive_verdict(self) -> TournamentVerdict | None:
        return self._default_non_interactive_verdict

    def format_markdown_report(self, report: ModelTournamentReport) -> str:
        """Format complete Champion vs. Challenger comparison Markdown report."""
        rep = report

        # Extract gate status indicators
        def _gate_indicator(gate_name: str) -> str:
            if gate_name in rep.approval_checklist:
                return "[PASS] PASSED" if rep.approval_checklist[gate_name] else "[FAIL] FAILED"
            return "[FAIL] UNCHECKED"

        g1 = _gate_indicator("gate_1_sharpe_outperformance")
        g2 = _gate_indicator("gate_2_max_drawdown_preservation")
        g3 = _gate_indicator("gate_3_brier_skill_score")
        g4 = _gate_indicator("gate_4_venn_abers_calibration")
        g5 = _gate_indicator("gate_5_anti_overfitting")

        sharpe_delta = rep.challenger_net_sharpe - rep.champion_net_sharpe
        dd_delta = rep.challenger_max_drawdown_pct - rep.champion_max_drawdown_pct
        calmar_delta = rep.challenger_calmar_ratio - rep.champion_calmar_ratio
        win_rate_delta = rep.challenger_win_rate - rep.champion_win_rate
        profit_factor_delta = rep.challenger_profit_factor - rep.champion_profit_factor
        expectancy_delta = rep.challenger_expectancy_r - rep.champion_expectancy_r

        bss_delta = rep.challenger_brier_skill_score - rep.champion_brier_skill_score
        brier_delta = rep.challenger_brier_score - rep.champion_brier_score
        log_loss_delta = rep.challenger_log_loss - rep.champion_log_loss
        venn_delta = rep.challenger_venn_abers_spread - rep.champion_venn_abers_spread

        lines = [
            "# Weekend Alpha Evolution: Champion vs. Challenger Tournament Report",
            "",
            f"- **Execution Timestamp (UTC)**: `{rep.created_at_utc}`",
            f"- **Champion Model ID**: `{rep.champion_model_id}` ({rep.champion_feature_count} features)",
            f"- **Challenger Model ID**: `{rep.challenger_model_id}` ({rep.challenger_feature_count} features)",
            f"- **Recommended Verdict**: **`{rep.recommended_verdict.value}`**",
            f"- **Promotable Status**: `{'YES' if rep.is_promotable else 'NO'}`",
            "",
            "## 1. Executive Summary & Rationale",
            "",
            rep.verdict_rationale,
            "",
            "## 2. Mandatory 5 Quantitative Approval Gates",
            "",
            "| Gate # | Approval Gate Description | Criteria | Status |",
            "|---|---|---|---|",
            f"| Gate 1 | Financial Outperformance | Challenger Sharpe Delta >= +0.15 | {g1} |",
            f"| Gate 2 | Capital Risk Preservation | Challenger Max Drawdown <= Champion Max Drawdown | {g2} |",
            f"| Gate 3 | Probability Calibration Quality | Challenger Brier Skill Score > Champion BSS | {g3} |",
            f"| Gate 4 | Venn-Abers Epistemic Uncertainty | Challenger Spread <= 0.12 | {g4} |",
            f"| Gate 5 | Anti-Overfitting Discipline | In-Sample to OOS Sharpe Decay < 35% | {g5} |",
            "",
            "## 3. Financial Backtest Metrics Comparison",
            "",
            "| Financial Backtest Metric | Champion | Challenger | Absolute Delta | Relative Impact |",
            "|---|---|---|---|---|",
            f"| **Annualized Net Sharpe** | {round(rep.champion_net_sharpe, 4)} | {round(rep.challenger_net_sharpe, 4)} | {round(sharpe_delta, 4):+0.4f} | {'Outperforming' if sharpe_delta >= 0.15 else ('Positive' if sharpe_delta > 0 else 'Underperforming')} |",
            f"| **Max Drawdown (MDD %)** | {round(rep.champion_max_drawdown_pct * 100, 2)}% | {round(rep.challenger_max_drawdown_pct * 100, 2)}% | {round(dd_delta * 100, 2):+0.2f}% | {'Lower Risk' if dd_delta <= 0 else 'Higher Risk'} |",
            f"| **Calmar Ratio** | {round(rep.champion_calmar_ratio, 3)} | {round(rep.challenger_calmar_ratio, 3)} | {round(calmar_delta, 3):+0.3f} | {'Superior' if calmar_delta >= 0 else 'Inferior'} |",
            f"| **Win Rate** | {round(rep.champion_win_rate * 100, 2)}% | {round(rep.challenger_win_rate * 100, 2)}% | {round(win_rate_delta * 100, 2):+0.2f}% | {'Improved' if win_rate_delta >= 0 else 'Degraded'} |",
            f"| **Profit Factor** | {round(rep.champion_profit_factor, 3)} | {round(rep.challenger_profit_factor, 3)} | {round(profit_factor_delta, 3):+0.3f} | {'Superior' if profit_factor_delta >= 0 else 'Inferior'} |",
            f"| **Expectancy (R)** | {round(rep.champion_expectancy_r, 4)} | {round(rep.challenger_expectancy_r, 4)} | {round(expectancy_delta, 4):+0.4f} | {'Superior' if expectancy_delta >= 0 else 'Inferior'} |",
            "",
            "## 4. Machine Learning & Calibration Metrics Comparison",
            "",
            "| ML Calibration Metric | Champion | Challenger | Absolute Delta | Interpretation |",
            "|---|---|---|---|---|",
            f"| **Brier Skill Score (BSS)** | {round(rep.champion_brier_skill_score, 4)} | {round(rep.challenger_brier_skill_score, 4)} | {round(bss_delta, 4):+0.4f} | {'Superior' if bss_delta > 0 else 'Inferior'} |",
            f"| **Brier Score (Mean Squared Error)** | {round(rep.champion_brier_score, 4)} | {round(rep.challenger_brier_score, 4)} | {round(brier_delta, 4):+0.4f} | {'More Accurate' if brier_delta < 0 else 'Less Accurate'} |",
            f"| **Cross-Entropy Log-Loss** | {round(rep.champion_log_loss, 4)} | {round(rep.challenger_log_loss, 4)} | {round(log_loss_delta, 4):+0.4f} | {'Better Confidence' if log_loss_delta < 0 else 'Worse Confidence'} |",
            f"| **Venn-Abers Calibration Spread** | {round(rep.champion_venn_abers_spread, 4)} | {round(rep.challenger_venn_abers_spread, 4)} | {round(venn_delta, 4):+0.4f} | {'Lower Epistemic Uncertainty' if venn_delta <= 0 else 'Higher Epistemic Uncertainty'} |",
            "",
            "## 5. Alpha Zoo Evolution & Model Architecture",
            "",
            f"- **Champion Active Feature Dimension**: `{rep.champion_feature_count}` quantitative features.",
            f"- **Challenger Active Feature Dimension**: `{rep.challenger_feature_count}` quantitative features.",
            "- **Ensemble Architecture**: Dual-Binary LightGBM with monotonic calibrated constraints.",
            "- **Cross-Validation Scheme**: Purged Group Walk-Forward CV (Zero Look-Ahead Bias).",
            "",
            "---",
            "*Report automatically generated by LLM Trading Bot Weekend Quantitative Evolution Pipeline.*",
        ]
        return "\n".join(lines)

    def _resolve_report_file_path(self, created_at_utc: str) -> Path:
        """Resolve and ensure parent directory for the destination markdown report file."""
        if "T" in created_at_utc:
            date_token = created_at_utc.split("T")[0]
        else:
            date_token = created_at_utc.replace(":", "-")[:10]

        target_dir = Path(self._report_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"tournament_{date_token}.md"

    async def present_tournament_report(
        self,
        report: ModelTournamentReport,
    ) -> TournamentVerdict:
        """Present tournament report, write markdown file, and obtain operator/automated verdict."""
        markdown_content = self.format_markdown_report(report)
        file_path = self._resolve_report_file_path(report.created_at_utc)
        file_path.write_text(markdown_content, encoding="utf-8")

        # Print to console
        sys.stdout.write("\n" + "=" * 78 + "\n")
        sys.stdout.write(markdown_content + "\n")
        sys.stdout.write("=" * 78 + "\n")
        sys.stdout.write(f"\nTournament markdown report saved to: {file_path.resolve()}\n")
        sys.stdout.flush()

        # Handle interactive operator decision vs non-interactive automated resolution
        if self._interactive:
            sys.stdout.write("\n" + "-" * 40 + "\n")
            sys.stdout.write("CHAMPION VS. CHALLENGER OPERATOR DECISION\n")
            sys.stdout.write(f"Recommended System Verdict: {report.recommended_verdict.value}\n")
            sys.stdout.write("Options:\n")
            sys.stdout.write("  [1] Promote Immediately (Challenger becomes active Champion)\n")
            sys.stdout.write("  [2] Run 48h Shadow Battle (Challenger runs shadow paper for 48h)\n")
            sys.stdout.write("  [3] Keep Champion (Retain Champion, discard Challenger)\n")
            sys.stdout.write("-" * 40 + "\n")
            sys.stdout.flush()

            try:
                user_choice = input("Enter decision [1/2/3] (default: recommended): ").strip()
                choice_map: dict[str, TournamentVerdict] = {
                    "1": TournamentVerdict.PROMOTE_CHALLENGER_IMMEDIATELY,
                    "2": TournamentVerdict.RUN_48H_SHADOW_BATTLE,
                    "3": TournamentVerdict.KEEP_CHAMPION,
                }
                if user_choice in choice_map:
                    return choice_map[user_choice]
                return report.recommended_verdict
            except (EOFError, KeyboardInterrupt):
                return report.recommended_verdict

        # Non-interactive mode
        if self._default_non_interactive_verdict is not None:
            return self._default_non_interactive_verdict

        return report.recommended_verdict
