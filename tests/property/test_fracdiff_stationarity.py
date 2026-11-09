"""Hypothesis property-based tests for Fractional Differentiation and Numba Kernels (Zero Defaults).

Verifies mathematical invariants:
1. Weight decay & monotonicity: |w_k| strictly decreases for k >= 1 when 0 < d < 1.
2. Stationarity achievement: Random walk series differentiated at d >= 0.40 achieves stationarity (p < 0.05).
3. Exact JIT parity: Pure domain transform_series and Numba rolling_fracdiff_numba are bitwise/numerically identical.
4. Volatility non-negativity: Yang-Zhang volatility sigma_YZ >= 0.0 across all generated OHLC paths.
5. Hurst bounds: Anis-Lloyd Hurst exponent is strictly bounded in [0.05, 0.95].
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from trading_bot.adapters.quant_engine.numba_kernels import (
    compute_fracdiff_weights_numba,
    rolling_fracdiff_numba,
    rolling_hurst_rs_numba,
    rolling_yang_zhang_numba,
)
from trading_bot.domain.entities.quant.quant_configs import FracDiffConfig
from trading_bot.domain.services.quant.fracdiff_transformer import (
    FracDiffTransformerService,
)


@settings(max_examples=50, deadline=None)
@given(
    d=st.floats(min_value=0.05, max_value=0.95),
    length=st.integers(min_value=10, max_value=80),
)
def test_property_fracdiff_weights_monotonic_decay(
    d: float,
    length: int,
) -> None:
    """Invariant: Fractional differentiation weights monotonically decay in magnitude for k >= 1."""
    service = FracDiffTransformerService()
    weights = service.compute_weights(d=d, length=length, threshold=1e-8)

    assert len(weights) >= 2
    assert math.isclose(weights[0], 1.0, abs_tol=1e-12)

    # For 0 < d < 1, all subsequent weights are negative and strictly decreasing in absolute magnitude
    for k in range(1, len(weights) - 1):
        assert weights[k] < 0.0
        assert abs(weights[k + 1]) <= abs(weights[k]) + 1e-12


@settings(max_examples=40, deadline=None)
@given(
    d_order=st.floats(min_value=0.40, max_value=0.85),
    noise_seed=st.integers(min_value=1, max_value=10_000),
)
def test_property_fracdiff_stationarity_on_random_walks(
    d_order: float,
    noise_seed: int,
) -> None:
    """Invariant: Random walk series differentiated with d >= 0.40 achieves stationarity (p < 0.05)."""
    service = FracDiffTransformerService()
    rng = np.random.default_rng(noise_seed)

    # Generate random walk path of length 250: X_t = X_0 + cumsum(eps)
    n_bars = 250
    innovations = rng.normal(0.0, 1.0, n_bars)
    random_walk: list[float] = [100.0]
    for inv in innovations:
        random_walk.append(random_walk[-1] + float(inv))

    cfg = FracDiffConfig(
        d_order=d_order,
        weight_threshold=1e-4,
        max_window_length=64,
        adf_critical_pvalue=0.05,
    )

    transformed = service.transform_series(series=random_walk, cfg=cfg)
    valid_series = [v for v in transformed if v is not None]

    assert len(valid_series) >= 150
    t_stat, p_val = service.compute_df_stat(series=valid_series, division_eps=1e-12)

    # For d >= 0.40, the series must reject the unit root null hypothesis at p < 0.05
    assert t_stat < -2.86
    assert p_val <= 0.05


@settings(max_examples=40, deadline=None)
@given(
    d_order=st.floats(min_value=0.10, max_value=0.90),
    window_length=st.integers(min_value=10, max_value=50),
    data_seed=st.integers(min_value=1, max_value=5_000),
)
def test_property_pure_domain_and_numba_jit_parity(
    d_order: float,
    window_length: int,
    data_seed: int,
) -> None:
    """Invariant: Pure domain FracDiff transformer and Numba JIT kernels produce identical outputs."""
    service = FracDiffTransformerService()
    rng = np.random.default_rng(data_seed)

    n_samples = 100
    series_arr = rng.uniform(50.0, 200.0, n_samples)
    series_list = series_arr.tolist()

    cfg = FracDiffConfig(
        d_order=d_order,
        weight_threshold=1e-4,
        max_window_length=window_length,
        adf_critical_pvalue=0.05,
    )

    # Pure domain calculation
    domain_result = service.transform_series(series=series_list, cfg=cfg)

    # Numba JIT calculation
    numba_weights = compute_fracdiff_weights_numba(
        d=cfg.d_order,
        length=cfg.max_window_length,
        threshold=cfg.weight_threshold,
    )
    numba_result = rolling_fracdiff_numba(arr=series_arr, weights=numba_weights)

    assert len(domain_result) == len(numba_result)
    for i in range(len(domain_result)):
        dom_val = domain_result[i]
        num_val = float(numba_result[i])

        if dom_val is None:
            assert np.isnan(num_val)
        else:
            assert not np.isnan(num_val)
            assert math.isclose(dom_val, num_val, abs_tol=1e-10)


@settings(max_examples=30, deadline=None)
@given(
    seed=st.integers(min_value=1, max_value=10_000),
    window=st.integers(min_value=10, max_value=30),
)
def test_property_numba_yang_zhang_invariants(
    seed: int,
    window: int,
) -> None:
    """Invariant: Yang-Zhang volatility sigma_YZ is always non-negative and finite."""
    rng = np.random.default_rng(seed)
    n_bars = 60

    opens = np.empty(n_bars, dtype=np.float64)
    highs = np.empty(n_bars, dtype=np.float64)
    lows = np.empty(n_bars, dtype=np.float64)
    closes = np.empty(n_bars, dtype=np.float64)

    curr_p = 100.0
    for i in range(n_bars):
        op = curr_p + rng.normal(0.0, 0.5)
        cl = op + rng.normal(0.0, 0.5)
        hi = max(op, cl) + abs(rng.normal(0.0, 0.3))
        lo = min(op, cl) - abs(rng.normal(0.0, 0.3))

        opens[i] = max(op, 1.0)
        highs[i] = max(hi, 1.0)
        lows[i] = max(lo, 0.5)
        closes[i] = max(cl, 1.0)
        curr_p = closes[i]

    yz = rolling_yang_zhang_numba(opens=opens, highs=highs, lows=lows, closes=closes, window=window)

    assert len(yz) == n_bars
    for i in range(window - 1):
        assert np.isnan(yz[i])

    for i in range(window - 1, n_bars):
        val = float(yz[i])
        assert not np.isnan(val)
        assert val >= 0.0
        assert math.isfinite(val)


@settings(max_examples=30, deadline=None)
@given(
    seed=st.integers(min_value=1, max_value=10_000),
    window=st.integers(min_value=30, max_value=80),
)
def test_property_numba_hurst_bounded(
    seed: int,
    window: int,
) -> None:
    """Invariant: Anis-Lloyd Hurst exponent is strictly bounded in [0.05, 0.95]."""
    rng = np.random.default_rng(seed)
    n_samples = 120
    returns = rng.normal(0.0, 0.02, n_samples)

    hurst = rolling_hurst_rs_numba(arr=returns, window=window)

    assert len(hurst) == n_samples
    for i in range(window - 1):
        assert np.isnan(hurst[i])

    for i in range(window - 1, n_samples):
        val = float(hurst[i])
        assert not np.isnan(val)
        assert 0.05 <= val <= 0.95
        assert math.isfinite(val)
