"""Unit tests for FracDiffTransformerService and Numba Kernels (Zero Defaults).

Verifies:
- Binomial series weights calculation against theoretical binomial coefficients
- Memory decay and threshold truncation (|w_k| < tau)
- Fixed-window FracDiff convolution and causal invariance (first L-1 values are None)
- Boundary orders (d = 0 -> identity, d = 1 -> first difference)
- Input validation (empty series, non-finite values, invalid lengths)
- Pure Python Dickey-Fuller stationarity test
- Minimal d* search for stationarity
- Exact numerical parity between Pure Domain service and Numba JIT kernels
- Numba accelerated rolling Hurst exponent and Yang-Zhang volatility
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from trading_bot.adapters.quant_engine.numba_kernels import (
    compute_fracdiff_weights_numba,
    rolling_fracdiff_numba,
    rolling_hurst_rs_numba,
    rolling_yang_zhang_numba,
)
from trading_bot.domain.entities.quant.quant_configs import FracDiffConfig
from trading_bot.domain.exceptions import (
    InsufficientDataException,
    NumericalInstabilityException,
)
from trading_bot.domain.services.quant.fracdiff_transformer import (
    FracDiffTransformerService,
    compute_weights,
    find_min_d_for_stationarity,
    transform_series,
)


@pytest.fixture
def fracdiff_service() -> FracDiffTransformerService:
    return FracDiffTransformerService()


@pytest.fixture
def default_fracdiff_config() -> FracDiffConfig:
    return FracDiffConfig(
        d_order=0.40,
        weight_threshold=1e-4,
        max_window_length=64,
        adf_critical_pvalue=0.05,
    )


def test_weights_analytical_binomial(
    fracdiff_service: FracDiffTransformerService,
) -> None:
    """Verify FracDiff weights match exact analytical binomial expansion coefficients."""
    d = 0.40
    weights = fracdiff_service.compute_weights(d=d, length=5, threshold=1e-8)

    assert len(weights) == 5
    assert math.isclose(weights[0], 1.0, abs_tol=1e-12)
    assert math.isclose(weights[1], -d, abs_tol=1e-12)
    assert math.isclose(weights[2], -d * (1.0 - d) / 2.0, abs_tol=1e-12)
    assert math.isclose(weights[3], -d * (1.0 - d) * (2.0 - d) / 6.0, abs_tol=1e-12)
    assert math.isclose(weights[4], -d * (1.0 - d) * (2.0 - d) * (3.0 - d) / 24.0, abs_tol=1e-12)


def test_weights_threshold_cutoff(
    fracdiff_service: FracDiffTransformerService,
) -> None:
    """Verify weight array is truncated when magnitude falls below threshold."""
    weights = fracdiff_service.compute_weights(d=0.40, length=100, threshold=0.05)
    # For d=0.40: w_0=1.0, w_1=-0.40, w_2=-0.12, w_3=-0.064, w_4=-0.0416 (< 0.05)
    assert len(weights) == 4
    for w in weights:
        assert abs(w) >= 0.05


def test_weights_boundary_orders(
    fracdiff_service: FracDiffTransformerService,
) -> None:
    """Verify boundary cases d=0.0 (identity) and d=1.0 (first difference)."""
    # d = 0.0 -> [1.0]
    w_0 = fracdiff_service.compute_weights(d=0.0, length=20, threshold=1e-4)
    assert len(w_0) == 1
    assert math.isclose(w_0[0], 1.0, abs_tol=1e-12)

    # d = 1.0 -> [1.0, -1.0]
    w_1 = fracdiff_service.compute_weights(d=1.0, length=20, threshold=1e-4)
    assert len(w_1) == 2
    assert math.isclose(w_1[0], 1.0, abs_tol=1e-12)
    assert math.isclose(w_1[1], -1.0, abs_tol=1e-12)


def test_weights_validation_exceptions(
    fracdiff_service: FracDiffTransformerService,
) -> None:
    """Verify invalid parameters raise typed domain exceptions."""
    with pytest.raises(InsufficientDataException):
        fracdiff_service.compute_weights(d=0.40, length=0, threshold=1e-4)

    with pytest.raises(NumericalInstabilityException):
        fracdiff_service.compute_weights(d=float("nan"), length=10, threshold=1e-4)

    with pytest.raises(NumericalInstabilityException):
        fracdiff_service.compute_weights(d=0.40, length=10, threshold=-1.0)


def test_transform_series_exact_values(
    fracdiff_service: FracDiffTransformerService,
) -> None:
    """Verify fixed-window convolution preserves causality and yields exact numerical values."""
    cfg = FracDiffConfig(
        d_order=0.50,
        weight_threshold=1e-4,
        max_window_length=3,
        adf_critical_pvalue=0.05,
    )
    # w_0 = 1.0, w_1 = -0.5, w_2 = -0.125
    series = [100.0, 102.0, 105.0, 110.0]
    result = fracdiff_service.transform_series(series=series, cfg=cfg)

    assert len(result) == 4
    # First L - 1 = 2 values are None
    assert result[0] is None
    assert result[1] is None

    # t = 2: w_0 * 105 + w_1 * 102 + w_2 * 100 = 105 - 51 - 12.5 = 41.5
    expected_2 = 1.0 * 105.0 - 0.5 * 102.0 - 0.125 * 100.0
    assert result[2] is not None
    assert math.isclose(result[2], expected_2, abs_tol=1e-10)

    # t = 3: w_0 * 110 + w_1 * 105 + w_2 * 102 = 110 - 52.5 - 12.75 = 44.75
    expected_3 = 1.0 * 110.0 - 0.5 * 105.0 - 0.125 * 102.0
    assert result[3] is not None
    assert math.isclose(result[3], expected_3, abs_tol=1e-10)


def test_transform_series_causal_invariance(
    fracdiff_service: FracDiffTransformerService,
    default_fracdiff_config: FracDiffConfig,
) -> None:
    """Verify zero look-ahead bias: future data modifications never affect past values."""
    series_a = [100.0 + float(i) for i in range(50)]
    series_b = list(series_a)
    # Alter only future elements (index >= 30)
    for i in range(30, 50):
        series_b[i] = series_b[i] * 2.0

    res_a = fracdiff_service.transform_series(series=series_a, cfg=default_fracdiff_config)
    res_b = fracdiff_service.transform_series(series=series_b, cfg=default_fracdiff_config)

    for i in range(30):
        val_a = res_a[i]
        val_b = res_b[i]
        if val_a is None:
            assert val_b is None
        else:
            assert val_b is not None
            assert math.isclose(val_a, val_b, abs_tol=1e-12)


def test_transform_series_short_and_empty(
    fracdiff_service: FracDiffTransformerService,
    default_fracdiff_config: FracDiffConfig,
) -> None:
    """Verify handling of series shorter than L_max and empty inputs."""
    short_series = [10.0, 20.0]
    res_short = fracdiff_service.transform_series(series=short_series, cfg=default_fracdiff_config)
    assert len(res_short) == 2
    assert res_short[0] is None
    assert res_short[1] is None

    with pytest.raises(InsufficientDataException):
        fracdiff_service.transform_series(series=[], cfg=default_fracdiff_config)

    with pytest.raises(NumericalInstabilityException):
        fracdiff_service.transform_series(
            series=[10.0, float("nan"), 12.0], cfg=default_fracdiff_config
        )


def test_transform_series_identity_and_first_difference(
    fracdiff_service: FracDiffTransformerService,
) -> None:
    """Verify identity transform at d=0.0 and first difference at d=1.0."""
    series = [10.0, 15.0, 12.0, 18.0, 20.0]

    # d = 0.0 -> weights = [1.0], no lookback needed (L=1)
    cfg_0 = FracDiffConfig(
        d_order=0.0,
        weight_threshold=1e-4,
        max_window_length=10,
        adf_critical_pvalue=0.05,
    )
    res_0 = fracdiff_service.transform_series(series=series, cfg=cfg_0)
    for i in range(len(series)):
        val_0 = res_0[i]
        assert val_0 is not None
        assert math.isclose(val_0, series[i], abs_tol=1e-12)

    # d = 1.0 -> weights = [1.0, -1.0], L=2
    cfg_1 = FracDiffConfig(
        d_order=1.0,
        weight_threshold=1e-4,
        max_window_length=10,
        adf_critical_pvalue=0.05,
    )
    res_1 = fracdiff_service.transform_series(series=series, cfg=cfg_1)
    assert res_1[0] is None
    for i in range(1, len(series)):
        expected_diff = series[i] - series[i - 1]
        val_1 = res_1[i]
        assert val_1 is not None
        assert math.isclose(val_1, expected_diff, abs_tol=1e-12)


def test_pure_python_df_stat(
    fracdiff_service: FracDiffTransformerService,
) -> None:
    """Verify Dickey-Fuller stationarity test distinguishes stationary from random walk series."""
    rng = random.Random(42)

    # Stationary series (mean-reverting AR(1) with phi=0.2)
    stationary_series: list[float] = [0.0]
    for _ in range(150):
        next_val = 0.2 * stationary_series[-1] + rng.gauss(0.0, 1.0)
        stationary_series.append(next_val)

    t_stat_stat, p_val_stat = fracdiff_service.compute_df_stat(
        series=stationary_series, division_eps=1e-12
    )
    # Stationary series has strongly negative t-statistic (t < -2.86) and low p-value (p < 0.05)
    assert t_stat_stat < -2.86
    assert p_val_stat <= 0.05

    # Random walk series (unit root: y_t = y_{t-1} + e_t)
    rw_series: list[float] = [100.0]
    for _ in range(150):
        rw_series.append(rw_series[-1] + rng.gauss(0.0, 1.0))

    t_stat_rw, p_val_rw = fracdiff_service.compute_df_stat(series=rw_series, division_eps=1e-12)
    # Random walk fails to reject unit root at 5% significance (t > -2.86, p > 0.05)
    assert t_stat_rw > -2.86
    assert p_val_rw > 0.05


def test_find_min_d_for_stationarity(
    fracdiff_service: FracDiffTransformerService,
) -> None:
    """Verify minimal d search finds stationary order on random walk series."""
    rng = random.Random(123)
    rw: list[float] = [50.0]
    for _ in range(200):
        rw.append(rw[-1] + rng.gauss(0.0, 1.0))

    cfg = FracDiffConfig(
        d_order=0.40,
        weight_threshold=1e-4,
        max_window_length=32,
        adf_critical_pvalue=0.05,
    )
    candidates = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    min_d, t_stat, p_val = fracdiff_service.find_min_d_for_stationarity(
        series=rw,
        cfg=cfg,
        d_candidates=candidates,
        division_eps=1e-12,
    )
    # Minimum d should be > 0.0 (random walk is not stationary at d=0.0)
    assert min_d > 0.0
    assert p_val <= 0.05
    assert t_stat < -2.86


def test_standalone_functional_interface() -> None:
    """Verify module-level standalone functions delegate cleanly to FracDiffTransformerService."""
    w = compute_weights(d=0.35, length=10, threshold=1e-4)
    assert len(w) > 1

    cfg = FracDiffConfig(
        d_order=0.35,
        weight_threshold=1e-4,
        max_window_length=10,
        adf_critical_pvalue=0.05,
    )
    transformed = transform_series(series=[1.0, 2.0, 3.0, 4.0, 5.0], cfg=cfg)
    assert len(transformed) == 5

    res_d, _, _ = find_min_d_for_stationarity(
        series=[float(i) for i in range(100)],
        cfg=cfg,
        d_candidates=[0.0, 0.5, 1.0],
        division_eps=1e-12,
    )
    assert res_d in [0.0, 0.5, 1.0]


def test_numba_compute_weights_parity(
    fracdiff_service: FracDiffTransformerService,
) -> None:
    """Verify Numba compute_fracdiff_weights_numba matches pure domain compute_weights exactly."""
    d = 0.45
    length = 50
    threshold = 1e-4

    domain_weights = fracdiff_service.compute_weights(d=d, length=length, threshold=threshold)
    numba_weights = compute_fracdiff_weights_numba(d=d, length=length, threshold=threshold)

    assert len(domain_weights) == len(numba_weights)
    for i in range(len(domain_weights)):
        assert math.isclose(domain_weights[i], float(numba_weights[i]), abs_tol=1e-12)


def test_numba_rolling_fracdiff_parity(
    fracdiff_service: FracDiffTransformerService,
) -> None:
    """Verify Numba rolling_fracdiff_numba matches pure domain transform_series exactly."""
    rng = random.Random(999)
    series_list = [100.0 + rng.gauss(0.0, 2.0) for _ in range(80)]
    series_arr = np.array(series_list, dtype=np.float64)

    cfg = FracDiffConfig(
        d_order=0.40,
        weight_threshold=1e-4,
        max_window_length=30,
        adf_critical_pvalue=0.05,
    )

    domain_res = fracdiff_service.transform_series(series=series_list, cfg=cfg)
    weights_arr = compute_fracdiff_weights_numba(
        d=cfg.d_order, length=cfg.max_window_length, threshold=cfg.weight_threshold
    )
    numba_res = rolling_fracdiff_numba(arr=series_arr, weights=weights_arr)

    assert len(domain_res) == len(numba_res)
    for i in range(len(domain_res)):
        val_domain = domain_res[i]
        val_numba = float(numba_res[i])
        if val_domain is None:
            assert np.isnan(val_numba)
        else:
            assert not np.isnan(val_numba)
            assert math.isclose(val_domain, val_numba, abs_tol=1e-10)


def test_numba_rolling_hurst_rs() -> None:
    """Verify Numba rolling Anis-Lloyd corrected Hurst exponent kernel."""
    rng = np.random.default_rng(42)
    # Pure Gaussian random walk returns (mean 0, std 1)
    returns = rng.normal(0.0, 1.0, 150)
    window = 100

    hurst_arr = rolling_hurst_rs_numba(arr=returns, window=window)

    assert len(hurst_arr) == 150
    # First window - 1 elements are NaN
    for i in range(window - 1):
        assert np.isnan(hurst_arr[i])

    # Valid values should be in [0.05, 0.95] and around 0.50 for random walk
    valid_hurst = hurst_arr[window - 1 :]
    assert len(valid_hurst) == 51
    for h in valid_hurst:
        assert 0.05 <= h <= 0.95
    # Mean of random walk Hurst should be close to 0.50
    assert 0.35 <= float(np.mean(valid_hurst)) <= 0.65


def test_numba_rolling_yang_zhang() -> None:
    """Verify Numba rolling Yang-Zhang minimum-variance volatility kernel."""
    n_bars = 60
    base_price = 100.0
    opens = np.empty(n_bars, dtype=np.float64)
    highs = np.empty(n_bars, dtype=np.float64)
    lows = np.empty(n_bars, dtype=np.float64)
    closes = np.empty(n_bars, dtype=np.float64)

    rng = np.random.default_rng(123)
    p = base_price
    for i in range(n_bars):
        p_open = p + rng.normal(0.0, 0.2)
        p_close = p_open + rng.normal(0.0, 0.5)
        p_high = max(p_open, p_close) + abs(rng.normal(0.0, 0.3))
        p_low = min(p_open, p_close) - abs(rng.normal(0.0, 0.3))
        opens[i] = p_open
        highs[i] = p_high
        lows[i] = p_low
        closes[i] = p_close
        p = p_close

    window = 20
    yz_arr = rolling_yang_zhang_numba(
        opens=opens, highs=highs, lows=lows, closes=closes, window=window
    )

    assert len(yz_arr) == n_bars
    for i in range(window - 1):
        assert np.isnan(yz_arr[i])

    valid_yz = yz_arr[window - 1 :]
    assert len(valid_yz) == 41
    for yz in valid_yz:
        assert not np.isnan(yz)
        assert yz > 0.0
        assert math.isfinite(yz)
