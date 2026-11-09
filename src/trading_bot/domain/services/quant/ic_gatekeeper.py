"""Rolling Spearman Rank Information Coefficient (IC) Gatekeeper Service (Zero Defaults).

Pure Domain Service implementing Invariant #9:
- Spearman Rank correlation between feature values and forward returns over rolling window
- Statistical significance filtering: |Rank IC| >= min_rank_ic, |t-stat| >= min_t_statistic, IC-IR >= min_ic_ir
- High/Low volatility regime invariance verification
- Economic direction locking to eliminate LLM confabulation
"""

from __future__ import annotations

import math

from pydantic import BaseModel

from trading_bot.domain.entities.quant.quant_configs import ICGatekeeperConfig
from trading_bot.domain.exceptions import InsufficientDataException, NumericalInstabilityException


class ICGatekeeperEvaluation(BaseModel):
    """Immutable evaluation verdict for a quantitative feature signal."""

    feature_name: str
    rank_ic: float
    t_statistic: float
    ic_ir: float
    high_vol_ic: float
    low_vol_ic: float
    is_regime_invariant: bool
    is_direction_valid: bool
    is_statistically_active: bool
    rejection_reason: str | None


class ICGatekeeperService:
    """Evaluates features against strict statistical significance and economic rationale gates."""

    def __init__(self, config: ICGatekeeperConfig) -> None:
        self._config: ICGatekeeperConfig = config

    def compute_ranks(self, series: list[float]) -> list[float]:
        """Compute fractional ranks for a numeric series handling ties with midranks."""
        n = len(series)
        if n == 0:
            return []

        for val in series:
            if not math.isfinite(val):
                raise NumericalInstabilityException(f"Non-finite value {val} in rank calculation")

        # Sort indices by value
        sorted_indices = sorted(range(n), key=lambda i: series[i])
        ranks = [0.0] * n

        i = 0
        while i < n:
            j = i
            while j + 1 < n and series[sorted_indices[j + 1]] == series[sorted_indices[i]]:
                j += 1
            # Fractional rank is 1-based average of indices from i to j
            avg_rank = (float(i + 1) + float(j + 1)) / 2.0
            for k in range(i, j + 1):
                ranks[sorted_indices[k]] = avg_rank
            i = j + 1

        return ranks

    def compute_spearman_ic(
        self,
        x: list[float],
        y: list[float],
        eps: float,
    ) -> float:
        """Compute Spearman Rank Correlation between two float series."""
        n = len(x)
        if n != len(y):
            raise InsufficientDataException(
                f"Length mismatch in Spearman IC calculation: x={n}, y={len(y)}"
            )
        if n < 3:
            raise InsufficientDataException(
                f"Sample size {n} insufficient for Spearman IC calculation (min 3)"
            )

        rank_x = self.compute_ranks(x)
        rank_y = self.compute_ranks(y)

        mean_x = sum(rank_x) / float(n)
        mean_y = sum(rank_y) / float(n)

        cov_xy = 0.0
        var_x = 0.0
        var_y = 0.0

        for i in range(n):
            dx = rank_x[i] - mean_x
            dy = rank_y[i] - mean_y
            cov_xy += dx * dy
            var_x += dx * dx
            var_y += dy * dy

        denom = math.sqrt(var_x * var_y)
        if denom <= eps:
            return 0.0

        corr = cov_xy / denom
        if corr > 1.0:
            corr = 1.0
        elif corr < -1.0:
            corr = -1.0
        return corr

    def compute_t_statistic(
        self,
        ic: float,
        n: int,
        eps: float,
    ) -> float:
        """Compute Student t-statistic for a given Rank IC and sample size n."""
        if n <= 2:
            return 0.0
        abs_ic = abs(ic)
        if abs_ic >= 1.0:
            return math.copysign(999.0, ic)

        df = float(n - 2)
        denom_sq = 1.0 - (ic * ic)
        if denom_sq <= eps:
            denom_sq = eps

        return (ic * math.sqrt(df)) / math.sqrt(denom_sq)

    def compute_ic_ir(
        self,
        sub_ics: list[float],
        eps: float,
    ) -> float:
        """Compute Information Ratio of ICs (mean / std)."""
        k = len(sub_ics)
        if k < 2:
            return 0.0

        mean_ic = sum(sub_ics) / float(k)
        var_ic = sum((val - mean_ic) ** 2 for val in sub_ics) / float(k - 1)
        std_ic = math.sqrt(var_ic)

        if std_ic <= eps:
            if abs(mean_ic) > eps:
                return math.copysign(999.0, mean_ic)
            return 0.0

        return mean_ic / std_ic

    def check_regime_invariance(
        self,
        x: list[float],
        y: list[float],
        vol_series: list[float],
        eps: float,
    ) -> tuple[bool, float, float]:
        """Verify correlation sign consistency across high and low volatility sub-samples."""
        n = len(x)
        if n != len(vol_series) or n < 10:
            raise InsufficientDataException(
                f"Sample size {n} insufficient for volatility regime invariance check"
            )

        sorted_indices = sorted(range(n), key=lambda idx: vol_series[idx])
        mid = n // 2
        low_indices = sorted_indices[:mid]
        high_indices = sorted_indices[mid:]

        low_x = [x[idx] for idx in low_indices]
        low_y = [y[idx] for idx in low_indices]
        high_x = [x[idx] for idx in high_indices]
        high_y = [y[idx] for idx in high_indices]

        if len(high_x) < 3 or len(low_x) < 3:
            return False, 0.0, 0.0

        high_ic = self.compute_spearman_ic(high_x, high_y, eps)
        low_ic = self.compute_spearman_ic(low_x, low_y, eps)

        # Invariance requires both sub-samples to share the same sign direction
        # and not cancel to opposite signs
        same_sign = (high_ic * low_ic) > 0.0
        return same_sign, high_ic, low_ic

    def evaluate_feature(
        self,
        feature_name: str,
        x: list[float],
        y: list[float],
        vol_series: list[float],
        locked_direction: int,
        sub_window_count: int,
        eps: float,
    ) -> ICGatekeeperEvaluation:
        """Run full gatekeeper evaluation pipeline on a feature signal series."""
        n = len(x)
        if n < self._config.rolling_eval_window_bars:
            raise InsufficientDataException(
                f"Feature '{feature_name}' has {n} bars, required {self._config.rolling_eval_window_bars}"
            )

        overall_ic = self.compute_spearman_ic(x, y, eps)
        t_stat = self.compute_t_statistic(overall_ic, n, eps)

        # Compute IC-IR over sub-windows
        k = max(2, sub_window_count)
        chunk_size = n // k
        sub_ics: list[float] = []
        for i in range(k):
            start_idx = i * chunk_size
            end_idx = n if i == k - 1 else (i + 1) * chunk_size
            chunk_x = x[start_idx:end_idx]
            chunk_y = y[start_idx:end_idx]
            if len(chunk_x) >= 3:
                sub_ic = self.compute_spearman_ic(chunk_x, chunk_y, eps)
                sub_ics.append(sub_ic)

        ic_ir = self.compute_ic_ir(sub_ics, eps)

        # Check regime invariance
        is_regime_invariant, high_vol_ic, low_vol_ic = self.check_regime_invariance(
            x, y, vol_series, eps
        )

        # Check economic direction alignment
        # locked_direction: +1 means higher feature expects higher forward return
        # -1 means higher feature expects lower forward return (or negative return)
        is_direction_valid = False
        if (locked_direction > 0 and overall_ic > 0) or (locked_direction < 0 and overall_ic < 0):
            is_direction_valid = True

        # Check all gates
        passed_ic = abs(overall_ic) >= self._config.min_rank_ic
        passed_t = abs(t_stat) >= self._config.min_t_statistic
        passed_ic_ir = abs(ic_ir) >= self._config.min_ic_ir

        is_active = (
            passed_ic and passed_t and passed_ic_ir and is_regime_invariant and is_direction_valid
        )

        rejection_reason: str | None = None
        if not is_direction_valid:
            rejection_reason = f"ECONOMIC_DIRECTION_MISMATCH: locked={locked_direction}, empirical_ic={overall_ic:.4f}"
        elif not passed_ic:
            rejection_reason = f"BELOW_MIN_RANK_IC: |{overall_ic:.4f}| < {self._config.min_rank_ic}"
        elif not passed_t:
            rejection_reason = f"BELOW_MIN_T_STAT: |{t_stat:.2f}| < {self._config.min_t_statistic}"
        elif not passed_ic_ir:
            rejection_reason = f"BELOW_MIN_IC_IR: |{ic_ir:.2f}| < {self._config.min_ic_ir}"
        elif not is_regime_invariant:
            rejection_reason = f"REGIME_INVARIANCE_FAILED: high_vol_ic={high_vol_ic:.4f}, low_vol_ic={low_vol_ic:.4f}"

        return ICGatekeeperEvaluation(
            feature_name=feature_name,
            rank_ic=overall_ic,
            t_statistic=t_stat,
            ic_ir=ic_ir,
            high_vol_ic=high_vol_ic,
            low_vol_ic=low_vol_ic,
            is_regime_invariant=is_regime_invariant,
            is_direction_valid=is_direction_valid,
            is_statistically_active=is_active,
            rejection_reason=rejection_reason,
        )
