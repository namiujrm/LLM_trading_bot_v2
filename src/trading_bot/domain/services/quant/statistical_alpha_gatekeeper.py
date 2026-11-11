"""Statistical Alpha Gatekeeper Domain Service (Clean Architecture, Invariant #9).

Pure Python implementation of institutional 4-Stage Statistical Screening:
- Stage 1: AST Invariance (No Look-Ahead, 1e-12 tolerance) & Sample Stability
- Stage 2: Predictive Power (|Rank IC| >= 0.025, |ICIR| >= 0.50, |t| >= 2.5),
           Regime Invariance, and 5-Quantile Monotonicity (|rho_mono| >= 0.80)
- Stage 3: Orthogonality Filter (Max Spearman |rho| with Alpha Zoo < 0.60)
Strict Clean Architecture: 100% Pure Python (stdlib math only), zero numpy/scipy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from trading_bot.domain.entities.quant.symbolic_alpha import (
    AlphaGatekeeperConfig,
    AlphaGateResult,
)
from trading_bot.domain.exceptions import InsufficientDataException


class StatisticalAlphaGatekeeper:
    """Pure domain service for rigorous 4-stage statistical alpha screening."""

    def __init__(self, cfg: AlphaGatekeeperConfig) -> None:
        self.cfg: AlphaGatekeeperConfig = cfg

    def evaluate_candidate(
        self,
        alpha_id: str,
        alpha_oos_series: Sequence[float],
        alpha_truncated_series: Sequence[float],
        target_returns_oos: Sequence[float],
        realized_vol_series: Sequence[float],
        expected_monotonic_sign: int,
        existing_alpha_matrix_oos: Sequence[Sequence[float]],
    ) -> AlphaGateResult:
        """Evaluate an alpha candidate through Stages 1, 2, and 3 screening gates."""
        n = len(alpha_oos_series)
        if n == 0:
            raise InsufficientDataException("Empty alpha series provided for evaluation")
        if n != len(target_returns_oos) or n != len(realized_vol_series):
            raise InsufficientDataException(
                f"Length mismatch: alpha={n}, target={len(target_returns_oos)}, vol={len(realized_vol_series)}"
            )
        for col_series in existing_alpha_matrix_oos:
            if len(col_series) != n:
                raise InsufficientDataException(
                    f"Existing alpha series length {len(col_series)} mismatch with full series length {n}"
                )

        # ------------------------------------------------------------------
        # STAGE 1: AST Security & Zero Look-Ahead Invariance Check
        # ------------------------------------------------------------------
        n_trunc = len(alpha_truncated_series)
        if n_trunc > n:
            raise InsufficientDataException(
                f"Truncated series length {n_trunc} exceeds full series length {n}"
            )

        is_invariant = self.check_series_invariance(
            series_full=alpha_oos_series[:n_trunc],
            series_trunc=alpha_truncated_series,
            tol=self.cfg.invariance_tol,
        )
        if not is_invariant:
            return AlphaGateResult(
                alpha_id=alpha_id,
                passed_all_gates=False,
                failed_stage="STAGE_1_LOOKAHEAD_VIOLATION",
                oos_rank_ic=0.0,
                oos_icir=0.0,
                t_statistic=0.0,
                is_regime_invariant=False,
                quantile_monotonicity_corr=0.0,
                max_corr_with_existing=0.0,
            )

        # Filter valid samples where candidate alpha, target return, and volatility are finite
        valid_indices: list[int] = []
        valid_a: list[float] = []
        valid_y: list[float] = []
        valid_vol: list[float] = []
        for i in range(n):
            a_val = alpha_oos_series[i]
            y_val = target_returns_oos[i]
            vol_val = realized_vol_series[i]
            if math.isfinite(a_val) and math.isfinite(y_val) and math.isfinite(vol_val):
                valid_indices.append(i)
                valid_a.append(a_val)
                valid_y.append(y_val)
                valid_vol.append(vol_val)

        n_valid = len(valid_a)
        if (float(n_valid) / float(n)) < self.cfg.min_valid_sample_ratio:
            return AlphaGateResult(
                alpha_id=alpha_id,
                passed_all_gates=False,
                failed_stage="STAGE_1_DEGENERATE_OR_EXCESS_NAN",
                oos_rank_ic=0.0,
                oos_icir=0.0,
                t_statistic=0.0,
                is_regime_invariant=False,
                quantile_monotonicity_corr=0.0,
                max_corr_with_existing=0.0,
            )

        std_a = self.compute_std(valid_a)
        if std_a < self.cfg.min_series_std:
            return AlphaGateResult(
                alpha_id=alpha_id,
                passed_all_gates=False,
                failed_stage="STAGE_1_DEGENERATE_OR_EXCESS_NAN",
                oos_rank_ic=0.0,
                oos_icir=0.0,
                t_statistic=0.0,
                is_regime_invariant=False,
                quantile_monotonicity_corr=0.0,
                max_corr_with_existing=0.0,
            )

        # ------------------------------------------------------------------
        # STAGE 2: Predictive Power, Regime Invariance & 5-Quantile Monotonicity
        # ------------------------------------------------------------------
        overall_ic = self.compute_spearman_ic(valid_a, valid_y, self.cfg.min_series_std)
        if not math.isfinite(overall_ic):
            return AlphaGateResult(
                alpha_id=alpha_id,
                passed_all_gates=False,
                failed_stage="STAGE_2_NAN_SPEARMAN_IC",
                oos_rank_ic=0.0,
                oos_icir=0.0,
                t_statistic=0.0,
                is_regime_invariant=False,
                quantile_monotonicity_corr=0.0,
                max_corr_with_existing=0.0,
            )

        t_stat = self.compute_t_statistic(overall_ic, n_valid, self.cfg.min_series_std)
        icir, block_ics = self.compute_icir(
            valid_a,
            valid_y,
            daily_block_bars=self.cfg.daily_block_bars,
            min_blocks=self.cfg.min_daily_blocks_for_icir,
            eps=self.cfg.min_series_std,
        )

        if len(block_ics) < self.cfg.min_daily_blocks_for_icir:
            return AlphaGateResult(
                alpha_id=alpha_id,
                passed_all_gates=False,
                failed_stage="STAGE_2_INSUFFICIENT_DAILY_BLOCKS",
                oos_rank_ic=round(overall_ic, 4),
                oos_icir=0.0,
                t_statistic=round(t_stat, 2),
                is_regime_invariant=False,
                quantile_monotonicity_corr=0.0,
                max_corr_with_existing=0.0,
            )

        is_regime_invariant, _high_ic, _low_ic = self.check_regime_invariance(
            valid_a, valid_y, valid_vol, eps=self.cfg.min_series_std
        )

        mono_corr, _q_means = self.compute_5quantile_monotonicity(
            valid_a, valid_y, eps=self.cfg.min_series_std
        )
        if not math.isfinite(mono_corr):
            return AlphaGateResult(
                alpha_id=alpha_id,
                passed_all_gates=False,
                failed_stage="STAGE_2_NON_MONOTONIC_QUANTILES",
                oos_rank_ic=round(overall_ic, 4),
                oos_icir=round(icir, 3),
                t_statistic=round(t_stat, 2),
                is_regime_invariant=is_regime_invariant,
                quantile_monotonicity_corr=0.0,
                max_corr_with_existing=0.0,
            )

        # Check predictive significance gates: |Rank IC| >= min_abs_rank_ic, |t| >= min_t, |ICIR| >= min_icir
        if (
            abs(overall_ic) < self.cfg.min_abs_rank_ic
            or abs(t_stat) < self.cfg.min_t_statistic
            or abs(icir) < self.cfg.min_abs_icir
        ):
            return AlphaGateResult(
                alpha_id=alpha_id,
                passed_all_gates=False,
                failed_stage="STAGE_2_WEAK_IC_OR_ICIR",
                oos_rank_ic=round(overall_ic, 4),
                oos_icir=round(icir, 3),
                t_statistic=round(t_stat, 2),
                is_regime_invariant=is_regime_invariant,
                quantile_monotonicity_corr=round(mono_corr, 3),
                max_corr_with_existing=0.0,
            )

        if not is_regime_invariant:
            return AlphaGateResult(
                alpha_id=alpha_id,
                passed_all_gates=False,
                failed_stage="STAGE_2_REGIME_INVARIANCE_FAILED",
                oos_rank_ic=round(overall_ic, 4),
                oos_icir=round(icir, 3),
                t_statistic=round(t_stat, 2),
                is_regime_invariant=False,
                quantile_monotonicity_corr=round(mono_corr, 3),
                max_corr_with_existing=0.0,
            )

        if abs(mono_corr) < self.cfg.min_quantile_monotonicity:
            return AlphaGateResult(
                alpha_id=alpha_id,
                passed_all_gates=False,
                failed_stage="STAGE_2_NON_MONOTONIC_QUANTILES",
                oos_rank_ic=round(overall_ic, 4),
                oos_icir=round(icir, 3),
                t_statistic=round(t_stat, 2),
                is_regime_invariant=is_regime_invariant,
                quantile_monotonicity_corr=round(mono_corr, 3),
                max_corr_with_existing=0.0,
            )

        # Check monotonic prior direction match
        prior_sign = 0
        if expected_monotonic_sign > 0:
            prior_sign = 1
        elif expected_monotonic_sign < 0:
            prior_sign = -1

        mono_sign = 0
        if mono_corr > 0.0:
            mono_sign = 1
        elif mono_corr < 0.0:
            mono_sign = -1

        ic_sign = 0
        if overall_ic > 0.0:
            ic_sign = 1
        elif overall_ic < 0.0:
            ic_sign = -1

        if prior_sign != 0 and (mono_sign != prior_sign or ic_sign != prior_sign):
            return AlphaGateResult(
                alpha_id=alpha_id,
                passed_all_gates=False,
                failed_stage="STAGE_2_DIRECTION_MISMATCH",
                oos_rank_ic=round(overall_ic, 4),
                oos_icir=round(icir, 3),
                t_statistic=round(t_stat, 2),
                is_regime_invariant=is_regime_invariant,
                quantile_monotonicity_corr=round(mono_corr, 3),
                max_corr_with_existing=0.0,
            )

        # ------------------------------------------------------------------
        # STAGE 3: Orthogonality Filter (Max Spearman |rho| with Alpha Zoo)
        # ------------------------------------------------------------------
        max_corr = 0.0
        if len(existing_alpha_matrix_oos) > 0:
            for col_series in existing_alpha_matrix_oos:
                if len(col_series) != n:
                    raise InsufficientDataException(
                        f"Existing alpha series length {len(col_series)} mismatch with full series length {n}"
                    )
                joint_a: list[float] = []
                joint_ex: list[float] = []
                for idx in valid_indices:
                    val_ex = col_series[idx]
                    if math.isfinite(val_ex):
                        joint_a.append(alpha_oos_series[idx])
                        joint_ex.append(val_ex)

                if len(joint_a) >= 3 and self.compute_std(joint_ex) > self.cfg.min_series_std:
                    corr_ex = abs(
                        self.compute_spearman_ic(joint_a, joint_ex, self.cfg.min_series_std)
                    )
                    if corr_ex > max_corr:
                        max_corr = corr_ex

        if max_corr >= self.cfg.max_collinearity_corr:
            return AlphaGateResult(
                alpha_id=alpha_id,
                passed_all_gates=False,
                failed_stage="STAGE_3_COLLINEAR_DUPLICATE",
                oos_rank_ic=round(overall_ic, 4),
                oos_icir=round(icir, 3),
                t_statistic=round(t_stat, 2),
                is_regime_invariant=is_regime_invariant,
                quantile_monotonicity_corr=round(mono_corr, 3),
                max_corr_with_existing=round(max_corr, 3),
            )

        # ------------------------------------------------------------------
        # ALL GATES PASSED
        # ------------------------------------------------------------------
        return AlphaGateResult(
            alpha_id=alpha_id,
            passed_all_gates=True,
            failed_stage=None,
            oos_rank_ic=round(overall_ic, 4),
            oos_icir=round(icir, 3),
            t_statistic=round(t_stat, 2),
            is_regime_invariant=is_regime_invariant,
            quantile_monotonicity_corr=round(mono_corr, 3),
            max_corr_with_existing=round(max_corr, 3),
        )

    # ----------------------------------------------------------------------
    # Pure Python Mathematical Helpers (No numpy/scipy)
    # ----------------------------------------------------------------------

    def check_series_invariance(
        self,
        series_full: Sequence[float],
        series_trunc: Sequence[float],
        tol: float,
    ) -> bool:
        """Verify look-ahead invariance between truncated and full prefix series."""
        if len(series_full) != len(series_trunc):
            return False
        for f, t in zip(series_full, series_trunc, strict=True):
            if math.isnan(f) or math.isnan(t):
                if not (math.isnan(f) and math.isnan(t)):
                    return False
            elif not math.isclose(f, t, rel_tol=tol, abs_tol=tol):
                return False
        return True

    def compute_ranks(self, series: Sequence[float]) -> list[float]:
        """Compute fractional ranks for a numeric series with midrank tie resolution."""
        n = len(series)
        if n == 0:
            return []
        sorted_indices = sorted(range(n), key=lambda i: series[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and series[sorted_indices[j + 1]] == series[sorted_indices[i]]:
                j += 1
            avg_rank = (float(i + 1) + float(j + 1)) / 2.0
            for k in range(i, j + 1):
                ranks[sorted_indices[k]] = avg_rank
            i = j + 1
        return ranks

    def compute_spearman_ic(self, x: Sequence[float], y: Sequence[float], eps: float) -> float:
        """Compute Spearman rank correlation between two series."""
        n = len(x)
        if n != len(y) or n < 3:
            return 0.0
        rx = self.compute_ranks(x)
        ry = self.compute_ranks(y)
        mx = sum(rx) / float(n)
        my = sum(ry) / float(n)
        cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
        vx = sum((rx[i] - mx) ** 2 for i in range(n))
        vy = sum((ry[i] - my) ** 2 for i in range(n))
        denom = math.sqrt(vx * vy)
        if denom <= eps:
            return 0.0
        r = cov / denom
        if r > 1.0:
            return 1.0
        if r < -1.0:
            return -1.0
        return r

    def compute_std(self, series: Sequence[float]) -> float:
        """Compute sample standard deviation (N-1 denominator)."""
        n = len(series)
        if n < 2:
            return 0.0
        mean = sum(series) / float(n)
        var = sum((x - mean) ** 2 for x in series) / float(n - 1)
        return math.sqrt(var)

    def compute_t_statistic(self, ic: float, n: int, eps: float) -> float:
        """Compute Student t-statistic for a given Rank IC and sample size n."""
        if n <= 2:
            return 0.0
        if abs(ic) >= 1.0:
            return math.copysign(999.0, ic)
        df = float(n - 2)
        denom_sq = 1.0 - (ic * ic)
        if denom_sq <= eps:
            denom_sq = eps
        return (ic * math.sqrt(df)) / math.sqrt(denom_sq)

    def compute_icir(
        self,
        x: Sequence[float],
        y: Sequence[float],
        daily_block_bars: int,
        min_blocks: int,
        eps: float,
    ) -> tuple[float, list[float]]:
        """Compute Information Ratio of daily block ICs (mean / std)."""
        n = len(x)
        num_blocks = n // daily_block_bars
        if num_blocks < min_blocks:
            return 0.0, []
        daily_ics: list[float] = []
        for b in range(num_blocks):
            sub_x = x[b * daily_block_bars : (b + 1) * daily_block_bars]
            sub_y = y[b * daily_block_bars : (b + 1) * daily_block_bars]
            if self.compute_std(sub_x) > eps and self.compute_std(sub_y) > eps:
                ic_b = self.compute_spearman_ic(sub_x, sub_y, eps)
                daily_ics.append(ic_b)
        if len(daily_ics) < min_blocks:
            return 0.0, daily_ics
        mean_ic = sum(daily_ics) / float(len(daily_ics))
        std_ic = self.compute_std(daily_ics)
        icir = mean_ic / (std_ic + eps)
        return icir, daily_ics

    def check_regime_invariance(
        self,
        x: Sequence[float],
        y: Sequence[float],
        vol: Sequence[float],
        eps: float,
    ) -> tuple[bool, float, float]:
        """Verify correlation sign consistency across high and low volatility sub-samples."""
        n = len(x)
        if n < 10:
            return False, 0.0, 0.0
        sorted_indices = sorted(range(n), key=lambda i: vol[i])
        mid = n // 2
        low_x = [x[i] for i in sorted_indices[:mid]]
        low_y = [y[i] for i in sorted_indices[:mid]]
        high_x = [x[i] for i in sorted_indices[mid:]]
        high_y = [y[i] for i in sorted_indices[mid:]]
        low_ic = self.compute_spearman_ic(low_x, low_y, eps)
        high_ic = self.compute_spearman_ic(high_x, high_y, eps)
        is_invariant = (low_ic * high_ic) > 0.0
        return is_invariant, high_ic, low_ic

    def compute_5quantile_monotonicity(
        self,
        x: Sequence[float],
        y: Sequence[float],
        eps: float,
    ) -> tuple[float, list[float]]:
        """Compute Spearman rank correlation across 5 equal quantile forward return averages."""
        n = len(x)
        if n < 10:
            return 0.0, []
        sorted_x = sorted(x)
        q_cuts = [sorted_x[int(p * (n - 1))] for p in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
        buckets: list[list[float]] = [[] for _ in range(5)]
        for val_x, val_y in zip(x, y, strict=True):
            placed = False
            for q_idx in range(4):
                if q_cuts[q_idx] <= val_x < q_cuts[q_idx + 1]:
                    buckets[q_idx].append(val_y)
                    placed = True
                    break
            if not placed:
                buckets[4].append(val_y)

        if any(len(b) == 0 for b in buckets):
            return 0.0, []

        q_means = [sum(b) / float(len(b)) for b in buckets]
        mono_corr = self.compute_spearman_ic([1.0, 2.0, 3.0, 4.0, 5.0], q_means, eps)
        return mono_corr, q_means
