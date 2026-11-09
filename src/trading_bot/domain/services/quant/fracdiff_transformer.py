"""Fixed-Window Fractional Differentiation (FracDiff) Pure Domain Service (Zero Defaults).

Pure Python implementation of fixed-window fractional differentiation, memory decay analysis,
and Dickey-Fuller stationarity evaluation:
- Binomial series weight generation: w_0 = 1.0, w_k = -w_{k-1} * (d - k + 1) / k.
- Memory truncation threshold: |w_k| < tau (tau from FracDiffConfig.weight_threshold).
- Fixed-width window convolution (L <= L_max) preserving strict causal invariance.
- First L - 1 values set to None (zero artificial padding, zero look-ahead bias).
- Augmented Dickey-Fuller (ADF) / Dickey-Fuller test statistic & MacKinnon p-value calculation.
- Minimum d* order search for stationarity (p < 0.05).

Strict Clean Architecture: 100% Pure Python (stdlib math only), zero numpy,
zero polars, zero numba, and zero default arguments (Invariant #9).
"""

from __future__ import annotations

import math

from trading_bot.domain.entities.quant.quant_configs import FracDiffConfig
from trading_bot.domain.exceptions import (
    InsufficientDataException,
    NumericalInstabilityException,
)


class FracDiffTransformerService:
    """Pure domain service for Fixed-Window Fractional Differentiation and stationarity."""

    def compute_weights(
        self,
        d: float,
        length: int,
        threshold: float,
    ) -> list[float]:
        """Compute FracDiff binomial expansion convolution weights.

        Expansion:
            w_0 = 1.0
            w_k = -w_{k-1} * (d - k + 1.0) / k
        Truncated when |w_k| < threshold or length is reached.

        Args:
            d: Fractional differentiation order (e.g., 0.40).
            length: Maximum window length L_max.
            threshold: Weight cutoff threshold tau (e.g., 1e-4).

        Returns:
            list[float]: Truncated weights array starting with w_0 = 1.0.
        """
        if length <= 0:
            raise InsufficientDataException(
                f"Window length must be strictly positive, received {length}."
            )
        if not math.isfinite(d):
            raise NumericalInstabilityException(
                f"Differentiation order d must be finite, received {d}."
            )
        if not math.isfinite(threshold) or threshold < 0.0:
            raise NumericalInstabilityException(
                f"Threshold must be non-negative and finite, received {threshold}."
            )

        weights: list[float] = [1.0]
        w_prev = 1.0

        for k in range(1, length):
            w_k = -w_prev * (d - float(k) + 1.0) / float(k)
            if not math.isfinite(w_k):
                raise NumericalInstabilityException(
                    f"FracDiff weight at index {k} diverged to non-finite value: {w_k}"
                )
            if abs(w_k) < threshold:
                break
            weights.append(w_k)
            w_prev = w_k

        return weights

    def transform_series(
        self,
        series: list[float],
        cfg: FracDiffConfig,
    ) -> list[float | None]:
        """Apply fixed-window fractional differentiation to a 1D time series.

        First L - 1 values produce None to preserve causality with zero look-ahead bias.

        Args:
            series: Input numerical sequence (e.g., log prices).
            cfg: FracDiffConfig with d_order, weight_threshold, max_window_length.

        Returns:
            list[float | None]: Stationarized series with first L - 1 elements as None.
        """
        n_samples = len(series)
        if n_samples == 0:
            raise InsufficientDataException(
                "Input series for fractional differentiation cannot be empty."
            )

        # Validate numerical inputs
        for idx, val in enumerate(series):
            if not math.isfinite(val):
                raise NumericalInstabilityException(
                    f"Series contains non-finite value at index {idx}: {val}"
                )

        weights = self.compute_weights(
            d=cfg.d_order,
            length=cfg.max_window_length,
            threshold=cfg.weight_threshold,
        )
        w_len = len(weights)

        output: list[float | None] = [None] * n_samples
        if n_samples < w_len:
            return output

        for t in range(w_len - 1, n_samples):
            accum = 0.0
            for k in range(w_len):
                accum += weights[k] * series[t - k]
            if not math.isfinite(accum):
                raise NumericalInstabilityException(
                    f"FracDiff transformation produced non-finite output at index {t}: {accum}"
                )
            output[t] = accum

        return output

    def compute_df_stat(
        self,
        series: list[float],
        division_eps: float,
    ) -> tuple[float, float]:
        """Compute Dickey-Fuller unit root test statistic and MacKinnon approximate p-value.

        Model: Delta y_t = alpha + beta * y_{t-1} + e_t.
        H_0: beta = 0 (unit root / non-stationary).
        H_1: beta < 0 (stationary).

        Args:
            series: 1D sequence of non-None values to test for stationarity.
            division_eps: Division protection epsilon.

        Returns:
            tuple[float, float]: (t-statistic, approximate p-value).
        """
        n = len(series)
        if n < 10:
            raise InsufficientDataException(
                f"Dickey-Fuller test requires at least 10 samples, received {n}."
            )

        # Differences and lagged levels
        x_lagged: list[float] = []
        delta_y: list[float] = []
        for t in range(1, n):
            x_lagged.append(series[t - 1])
            delta_y.append(series[t] - series[t - 1])

        t_len = len(x_lagged)
        mean_x = sum(x_lagged) / float(t_len)
        mean_y = sum(delta_y) / float(t_len)

        cov_xy = sum((x_lagged[i] - mean_x) * (delta_y[i] - mean_y) for i in range(t_len))
        var_x = sum((x - mean_x) ** 2 for x in x_lagged)

        if var_x <= division_eps:
            return (0.0, 1.0)

        beta_hat = cov_xy / var_x
        alpha_hat = mean_y - beta_hat * mean_x

        # Residual variance
        sse = sum((delta_y[i] - (alpha_hat + beta_hat * x_lagged[i])) ** 2 for i in range(t_len))
        df = max(t_len - 2, 1)
        s_squared = sse / float(df)

        se_beta = math.sqrt(s_squared / (var_x + division_eps))
        if se_beta <= division_eps:
            t_stat = 0.0
        else:
            t_stat = beta_hat / se_beta

        p_val = self._approx_mackinnon_p_value(t_stat=t_stat)
        return (t_stat, p_val)

    def _approx_mackinnon_p_value(
        self,
        t_stat: float,
    ) -> float:
        """MacKinnon (1994) response surface monotonic p-value approximation for DF test with constant.

        Critical lookup values for tau with constant:
        -3.43 -> 0.01, -2.86 -> 0.05, -2.57 -> 0.10, -1.94 -> 0.30,
        -1.45 -> 0.50, -0.93 -> 0.70, -0.18 -> 0.90, +0.60 -> 0.99.
        """
        # Interpolation anchors: (t_stat_threshold, p_value)
        anchors = (
            (-5.0, 0.0001),
            (-3.96, 0.001),
            (-3.43, 0.01),
            (-3.12, 0.025),
            (-2.86, 0.05),
            (-2.57, 0.10),
            (-2.23, 0.20),
            (-1.94, 0.30),
            (-1.69, 0.40),
            (-1.45, 0.50),
            (-1.20, 0.60),
            (-0.93, 0.70),
            (-0.61, 0.80),
            (-0.18, 0.90),
            (0.60, 0.99),
            (2.0, 0.9999),
        )
        if t_stat <= anchors[0][0]:
            return anchors[0][1]
        if t_stat >= anchors[-1][0]:
            return anchors[-1][1]

        for i in range(len(anchors) - 1):
            t_low, p_low = anchors[i]
            t_high, p_high = anchors[i + 1]
            if t_low <= t_stat <= t_high:
                ratio = (t_stat - t_low) / (t_high - t_low)
                return p_low + ratio * (p_high - p_low)

        return 1.0

    def find_min_d_for_stationarity(
        self,
        series: list[float],
        cfg: FracDiffConfig,
        d_candidates: list[float],
        division_eps: float,
    ) -> tuple[float, float, float]:
        """Find the minimum fractional differentiation order d that achieves stationarity (p < threshold).

        Args:
            series: Input non-stationary price/signal series.
            cfg: Base FracDiffConfig schema.
            d_candidates: List of candidate d values to evaluate (e.g., [0.0, 0.1, ..., 1.0]).
            division_eps: Numerical protection epsilon.

        Returns:
            tuple[float, float, float]: (min_stationary_d, t_statistic, p_value).
        """
        if len(d_candidates) == 0:
            raise InsufficientDataException("Candidate d list cannot be empty.")

        sorted_candidates = sorted(d_candidates)
        last_result = (sorted_candidates[-1], 0.0, 1.0)

        for d in sorted_candidates:
            trial_cfg = FracDiffConfig(
                d_order=d,
                weight_threshold=cfg.weight_threshold,
                max_window_length=cfg.max_window_length,
                adf_critical_pvalue=cfg.adf_critical_pvalue,
            )
            transformed = self.transform_series(series=series, cfg=trial_cfg)
            valid_vals = [v for v in transformed if v is not None]

            if len(valid_vals) < 10:
                continue

            t_stat, p_val = self.compute_df_stat(series=valid_vals, division_eps=division_eps)
            last_result = (d, t_stat, p_val)

            if p_val <= cfg.adf_critical_pvalue:
                return (d, t_stat, p_val)

        return last_result


# Standalone functional aliases for functional API consumers (all parameters explicit, zero defaults)
def compute_weights(
    d: float,
    length: int,
    threshold: float,
) -> list[float]:
    """Functional interface for computing FracDiff binomial expansion weights."""
    return FracDiffTransformerService().compute_weights(d=d, length=length, threshold=threshold)


def transform_series(
    series: list[float],
    cfg: FracDiffConfig,
) -> list[float | None]:
    """Functional interface for transforming a time series via FracDiff."""
    return FracDiffTransformerService().transform_series(series=series, cfg=cfg)


def find_min_d_for_stationarity(
    series: list[float],
    cfg: FracDiffConfig,
    d_candidates: list[float],
    division_eps: float,
) -> tuple[float, float, float]:
    """Functional interface for finding minimal stationary differentiation order d*."""
    return FracDiffTransformerService().find_min_d_for_stationarity(
        series=series,
        cfg=cfg,
        d_candidates=d_candidates,
        division_eps=division_eps,
    )
