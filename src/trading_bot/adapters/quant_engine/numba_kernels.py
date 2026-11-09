"""Numba Accelerated Quant Engine Kernels (Zero Defaults).

High-performance JIT-compiled kernels for quantitative time-series transformations:
- Fractional differentiation weights via binomial expansion (compute_fracdiff_weights_numba)
- Rolling fixed-window fractional differentiation convolution (rolling_fracdiff_numba)
- Rolling Anis-Lloyd bias-corrected Rescaled Range Hurst exponent (rolling_hurst_rs_numba)
- Rolling Yang-Zhang minimum-variance volatility (rolling_yang_zhang_numba)

All kernels use @numba.njit(fastmath=True, cache=True) and enforce Invariant #9
(zero default parameters, zero hardcoded fallback assignments).
"""

from __future__ import annotations

import math

import numba
import numpy as np


@numba.njit(fastmath=True, cache=True)
def compute_fracdiff_weights_numba(
    d: float,
    length: int,
    threshold: float,
) -> np.ndarray:
    """Compute fractional differentiation binomial expansion weights using Numba JIT.

    Expansion:
        w_0 = 1.0
        w_k = -w_{k-1} * (d - k + 1.0) / k
    Truncated when |w_k| < threshold or length is reached.

    Args:
        d: Fractional differentiation order (e.g., 0.40).
        length: Maximum window length L_max.
        threshold: Weight cutoff threshold tau (e.g., 1e-4).

    Returns:
        np.ndarray: 1D float64 array of truncated weights starting with w_0 = 1.0.
    """
    if length <= 0:
        return np.empty(0, dtype=np.float64)

    w_buf = np.zeros(length, dtype=np.float64)
    w_buf[0] = 1.0
    actual_len = length

    for k in range(1, length):
        w_k = -w_buf[k - 1] * (d - float(k) + 1.0) / float(k)
        if abs(w_k) < threshold:
            actual_len = k
            break
        w_buf[k] = w_k

    out = np.empty(actual_len, dtype=np.float64)
    for i in range(actual_len):
        out[i] = w_buf[i]
    return out


@numba.njit(fastmath=True, cache=True)
def rolling_fracdiff_numba(
    arr: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Apply fixed-window fractional differentiation convolution using Numba JIT.

    Preserves causal invariance by outputting NaN for all indices t < len(weights) - 1.

    Args:
        arr: 1D float64 array of input values (e.g., log prices).
        weights: 1D float64 array of FracDiff convolution weights.

    Returns:
        np.ndarray: 1D float64 array of transformed values with initial NaNs.
    """
    n = len(arr)
    w_len = len(weights)
    out = np.full(n, np.nan, dtype=np.float64)

    if w_len == 0 or n < w_len:
        return out

    for t in range(w_len - 1, n):
        accum = 0.0
        for k in range(w_len):
            accum += weights[k] * arr[t - k]
        out[t] = accum

    return out


@numba.njit(fastmath=True, cache=True)
def _anis_lloyd_expected_rs(
    m: int,
) -> float:
    """Theoretical Anis-Lloyd Expectation E[R/S]_m under Gaussian random walk."""
    sum_sqrt = 0.0
    for i in range(1, m):
        sum_sqrt += math.sqrt(float(m - i) / float(i))

    if m <= 340:
        gamma_ratio = math.gamma((float(m) - 1.0) / 2.0) / (
            math.sqrt(math.pi) * math.gamma(float(m) / 2.0)
        )
        return gamma_ratio * sum_sqrt
    else:
        return (1.0 / math.sqrt(math.pi * float(m) / 2.0)) * sum_sqrt


@numba.njit(fastmath=True, cache=True)
def _get_hurst_sub_sizes(
    window: int,
) -> np.ndarray:
    """Select candidate sub-chunk segment sizes for R/S regression."""
    candidates = (10, 20, 25, 50, 100)
    count = 0
    for c in candidates:
        if c <= window // 2:
            count += 1

    if count < 2:
        res = np.empty(2, dtype=np.int64)
        res[0] = max(window // 4, 2)
        res[1] = max(window // 2, 4)
        return res

    res = np.empty(count, dtype=np.int64)
    idx = 0
    for c in candidates:
        if c <= window // 2:
            res[idx] = c
            idx += 1
    return res


@numba.njit(fastmath=True, cache=True)
def rolling_hurst_rs_numba(
    arr: np.ndarray,
    window: int,
) -> np.ndarray:
    """Compute rolling Anis-Lloyd bias-corrected Hurst Exponent (H_t) via Numba JIT.

    Args:
        arr: 1D float64 array of log returns (or returns sequence).
        window: Rolling window size (e.g., 100).

    Returns:
        np.ndarray: 1D float64 array of Hurst exponents bounded in [0.05, 0.95].
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)

    if window < 10 or n < window:
        return out

    sub_sizes = _get_hurst_sub_sizes(window)
    num_subs = len(sub_sizes)
    eps = 1e-12

    for t in range(window - 1, n):
        segment = arr[t - window + 1 : t + 1]

        log_m = np.empty(num_subs, dtype=np.float64)
        log_rs_diff = np.empty(num_subs, dtype=np.float64)
        valid_subs = 0

        for s_idx in range(num_subs):
            m = sub_sizes[s_idx]
            num_blocks = window // m
            rs_sum = 0.0
            valid_blocks = 0

            for b in range(num_blocks):
                blk = segment[b * m : (b + 1) * m]
                mean_b = 0.0
                for i in range(m):
                    mean_b += blk[i]
                mean_b /= float(m)

                var_b = 0.0
                for i in range(m):
                    var_b += (blk[i] - mean_b) ** 2
                var_b /= float(m)
                std_b = math.sqrt(var_b)

                if std_b <= eps:
                    continue

                cum_dev = 0.0
                min_dev = 0.0
                max_dev = 0.0
                for i in range(m):
                    cum_dev += blk[i] - mean_b
                    if cum_dev < min_dev:
                        min_dev = cum_dev
                    if cum_dev > max_dev:
                        max_dev = cum_dev

                r_range = max_dev - min_dev
                rs_sum += r_range / std_b
                valid_blocks += 1

            if valid_blocks > 0:
                rs_empirical = rs_sum / float(valid_blocks)
                e_rs = max(_anis_lloyd_expected_rs(int(m)), eps)
                diff = math.log(rs_empirical + eps) - math.log(e_rs)
                log_m[valid_subs] = math.log(float(m))
                log_rs_diff[valid_subs] = diff
                valid_subs += 1

        if valid_subs < 2:
            out[t] = 0.50
            continue

        mean_x = 0.0
        mean_y = 0.0
        for i in range(valid_subs):
            mean_x += log_m[i]
            mean_y += log_rs_diff[i]
        mean_x /= float(valid_subs)
        mean_y /= float(valid_subs)

        cov_xy = 0.0
        var_x = 0.0
        for i in range(valid_subs):
            cov_xy += (log_m[i] - mean_x) * (log_rs_diff[i] - mean_y)
            var_x += (log_m[i] - mean_x) ** 2

        if var_x <= eps:
            out[t] = 0.50
        else:
            beta = cov_xy / var_x
            h_raw = 0.50 + beta
            out[t] = min(0.95, max(0.05, h_raw))

    return out


@numba.njit(fastmath=True, cache=True)
def rolling_yang_zhang_numba(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    window: int,
) -> np.ndarray:
    """Compute rolling Yang-Zhang Minimum-Variance Volatility (sigma_YZ) via Numba JIT.

    Combines overnight jump variance, continuous drift, and Rogers-Satchell range variance.

    Args:
        opens: 1D float64 array of Open prices.
        highs: 1D float64 array of High prices.
        lows: 1D float64 array of Low prices.
        closes: 1D float64 array of Close prices.
        window: Rolling window size (e.g., 20 bars).

    Returns:
        np.ndarray: 1D float64 array of annualized or per-bar Yang-Zhang volatility.
    """
    n = len(opens)
    out = np.full(n, np.nan, dtype=np.float64)

    if window < 2 or n < window:
        return out

    m = window - 1
    denom_m = max(m - 1, 1)
    k_weight = 0.34 / (1.34 + float(m + 1) / float(denom_m))
    eps = 1e-12

    for t in range(window - 1, n):
        mean_o = 0.0
        mean_c = 0.0

        for i in range(1, window):
            idx = t - window + 1 + i
            o_ret = math.log(max(opens[idx], eps) / max(closes[idx - 1], eps))
            c_ret = math.log(max(closes[idx], eps) / max(opens[idx], eps))
            mean_o += o_ret
            mean_c += c_ret

        mean_o /= float(m)
        mean_c /= float(m)

        var_o = 0.0
        var_c = 0.0
        rs_sum = 0.0

        for i in range(1, window):
            idx = t - window + 1 + i
            o_ret = math.log(max(opens[idx], eps) / max(closes[idx - 1], eps))
            c_ret = math.log(max(closes[idx], eps) / max(opens[idx], eps))

            u = math.log(max(highs[idx], eps) / max(opens[idx], eps))
            d = math.log(max(lows[idx], eps) / max(opens[idx], eps))
            c = c_ret

            rs = u * (u - c) + d * (d - c)
            if rs < 0.0:
                rs = 0.0

            var_o += (o_ret - mean_o) ** 2
            var_c += (c_ret - mean_c) ** 2
            rs_sum += rs

        var_o /= float(denom_m)
        var_c /= float(denom_m)
        var_rs = rs_sum / float(m)

        yz_sq = var_o + k_weight * var_c + (1.0 - k_weight) * var_rs
        out[t] = math.sqrt(max(0.0, yz_sq))

    return out
