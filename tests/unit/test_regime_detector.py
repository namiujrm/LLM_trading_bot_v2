"""Unit tests for RegimeDetectorService (Zero Defaults).

Verifies Anis-Lloyd Hurst, Yang-Zhang, Garman-Klass, Parkinson, Bipower Variation Jump,
Shannon Entropy, Permutation Entropy, Tsallis Entropy, and Regime Classification.
"""

from __future__ import annotations

import math

import pytest

from trading_bot.domain.entities.common.enums import MarketRegime
from trading_bot.domain.entities.market.market_state import MicroBar1m, MultiTFBar
from trading_bot.domain.entities.quant.quant_configs import RegimeConfig
from trading_bot.domain.exceptions import InsufficientDataException
from trading_bot.domain.services.quant.regime_detector import RegimeDetectorService


@pytest.fixture
def regime_config() -> RegimeConfig:
    return RegimeConfig(
        hurst_window=100,
        yang_zhang_window=20,
        bipower_jump_confidence=2.58,
        entropy_bins=10,
        permutation_dimension=3,
        tsallis_q=1.5,
        division_eps=1e-12,
    )


@pytest.fixture
def regime_service() -> RegimeDetectorService:
    return RegimeDetectorService()


def test_anis_lloyd_hurst_persistent_trend(
    regime_service: RegimeDetectorService,
    regime_config: RegimeConfig,
) -> None:
    # Strongly persistent trending series (fractional long-range memory noise)
    import random

    rng = random.Random(100)
    wn = [rng.gauss(0.0, 1.0) for _ in range(200)]
    fgn: list[float] = []
    for i in range(50, 150):
        val = sum(((k + 1) ** (-0.2)) * wn[i - k] for k in range(40))
        fgn.append(val * 0.001)

    hurst = regime_service.compute_anis_lloyd_hurst(fgn, regime_config)
    assert 0.55 < hurst <= 0.95
    assert math.isfinite(hurst)


def test_anis_lloyd_hurst_mean_reverting(
    regime_service: RegimeDetectorService,
    regime_config: RegimeConfig,
) -> None:
    # Strongly mean-reverting alternating return series
    returns = [0.01 if i % 2 == 0 else -0.01 for i in range(100)]
    hurst = regime_service.compute_anis_lloyd_hurst(returns, regime_config)
    assert 0.05 <= hurst < 0.50
    assert math.isfinite(hurst)


def test_anis_lloyd_hurst_insufficient_data_raises(
    regime_service: RegimeDetectorService,
    regime_config: RegimeConfig,
) -> None:
    short_returns = [0.01, -0.01, 0.02]
    with pytest.raises(InsufficientDataException):
        regime_service.compute_anis_lloyd_hurst(short_returns, regime_config)


def test_yang_zhang_vol_calculation(
    regime_service: RegimeDetectorService,
    regime_config: RegimeConfig,
) -> None:
    base_ts = 1700000000000
    bars: list[MultiTFBar] = []
    for i in range(20):
        o = 50000.0 + i * 20.0
        h = o + 200.0
        low_p = o - 150.0
        c = o + 50.0
        bars.append(
            MultiTFBar(
                symbol="BTCUSDT",
                timeframe="30m",
                timestamp_ms=base_ts + i * 1800000,
                open=o,
                high=h,
                low=low_p,
                close=c,
                volume=500.0,
            )
        )

    yz_vol = regime_service.compute_yang_zhang_vol(bars, regime_config)
    assert yz_vol > 0.0
    assert math.isfinite(yz_vol)


def test_yang_zhang_vol_insufficient_bars_raises(
    regime_service: RegimeDetectorService,
    regime_config: RegimeConfig,
) -> None:
    bars = [
        MultiTFBar(
            symbol="BTCUSDT",
            timeframe="30m",
            timestamp_ms=1700000000000,
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
        )
    ]
    with pytest.raises(InsufficientDataException):
        regime_service.compute_yang_zhang_vol(bars, regime_config)


def test_garman_klass_vol_calculation(
    regime_service: RegimeDetectorService,
    regime_config: RegimeConfig,
) -> None:
    base_ts = 1700000000000
    bars: list[MultiTFBar] = []
    for i in range(10):
        bars.append(
            MultiTFBar(
                symbol="BTCUSDT",
                timeframe="30m",
                timestamp_ms=base_ts + i * 1800000,
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=10.0,
            )
        )

    gk_vol = regime_service.compute_garman_klass_vol(bars, regime_config)
    assert gk_vol > 0.0
    assert math.isfinite(gk_vol)


def test_parkinson_vol_calculation(
    regime_service: RegimeDetectorService,
    regime_config: RegimeConfig,
) -> None:
    base_ts = 1700000000000
    bars: list[MultiTFBar] = []
    for i in range(10):
        bars.append(
            MultiTFBar(
                symbol="BTCUSDT",
                timeframe="30m",
                timestamp_ms=base_ts + i * 1800000,
                open=100.0,
                high=110.0,
                low=90.0,
                close=100.0,
                volume=10.0,
            )
        )

    p_vol = regime_service.compute_parkinson_vol(bars, regime_config)
    assert p_vol > 0.0
    assert math.isfinite(p_vol)


def test_bipower_variation_continuous_flow_no_jump(
    regime_service: RegimeDetectorService,
    regime_config: RegimeConfig,
) -> None:
    # 30 bars with mild smooth diffusion
    base_ts = 1700000000000
    bars: list[MicroBar1m] = []
    for i in range(30):
        # Sine wave returns (no discrete jump)
        p = 50000.0 + 10.0 * math.sin(i * 0.2)
        bars.append(
            MicroBar1m(
                timestamp_ms=base_ts + i * 60000,
                open=p,
                high=p + 5.0,
                low=p - 5.0,
                close=p + 1.0,
                volume=10.0,
            )
        )

    bv, z_score, has_jump = regime_service.compute_bipower_variation_jump(bars, regime_config)
    assert bv >= 0.0
    assert math.isfinite(z_score)
    assert has_jump is False


def test_bipower_variation_detects_discrete_jump(
    regime_service: RegimeDetectorService,
    regime_config: RegimeConfig,
) -> None:
    # 30 bars with stable prices and a discrete jump (permanent step drop)
    base_ts = 1700000000000
    bars: list[MicroBar1m] = []
    for i in range(30):
        if i < 15:
            p = 50000.0 + (i % 3)
        else:
            p = 44000.0 + (i % 3)

        bars.append(
            MicroBar1m(
                timestamp_ms=base_ts + i * 60000,
                open=p,
                high=p + 2.0,
                low=p - 2.0,
                close=p,
                volume=50.0,
            )
        )

    bv, z_score, has_jump = regime_service.compute_bipower_variation_jump(bars, regime_config)
    assert bv >= 0.0
    assert z_score > 2.58
    assert has_jump is True


def test_shannon_entropy_bounds(
    regime_service: RegimeDetectorService,
    regime_config: RegimeConfig,
) -> None:
    # Flat distribution -> 0.0 entropy
    flat_returns = [0.001] * 50
    zero_entropy = regime_service.compute_shannon_entropy(flat_returns, regime_config)
    assert zero_entropy == 0.0

    # Diverse returns -> high entropy
    diverse_returns = [(i - 25) * 0.001 for i in range(50)]
    norm_entropy = regime_service.compute_shannon_entropy(diverse_returns, regime_config)
    assert 0.50 <= norm_entropy <= 1.0


def test_permutation_entropy_monotonic_vs_chaotic(
    regime_service: RegimeDetectorService,
    regime_config: RegimeConfig,
) -> None:
    # Strictly increasing monotonic series -> only 1 ordinal pattern (0, 1, 2) -> PE = 0.0
    monotonic_returns = [i * 0.001 for i in range(50)]
    pe_mono = regime_service.compute_permutation_entropy(monotonic_returns, regime_config)
    assert pe_mono == 0.0

    # Oscillating chaotic returns -> diverse patterns -> PE near 1.0
    chaotic_returns = [math.sin(i * 1.5) for i in range(50)]
    pe_chaotic = regime_service.compute_permutation_entropy(chaotic_returns, regime_config)
    assert pe_chaotic > 0.60


def test_tsallis_entropy_calculation(
    regime_service: RegimeDetectorService,
    regime_config: RegimeConfig,
) -> None:
    returns = [(i - 20) * 0.002 for i in range(40)]
    tsallis = regime_service.compute_tsallis_entropy(returns, regime_config)
    assert tsallis >= 0.0
    assert math.isfinite(tsallis)


def test_classify_regime(
    regime_service: RegimeDetectorService,
    regime_config: RegimeConfig,
) -> None:
    # 1. High Volatility Jump takes precedence
    regime_jump = regime_service.classify_regime(
        hurst=0.70, yz_vol=0.04, has_jump=True, cfg=regime_config
    )
    assert regime_jump == MarketRegime.HIGH_VOL_LIQUIDATION_CASCADE

    # 2. Mean reverting range
    regime_range = regime_service.classify_regime(
        hurst=0.38, yz_vol=0.01, has_jump=False, cfg=regime_config
    )
    assert regime_range == MarketRegime.MEAN_REVERTING_RANGE

    # 3. Persistent trend
    regime_trend = regime_service.classify_regime(
        hurst=0.62, yz_vol=0.02, has_jump=False, cfg=regime_config
    )
    assert regime_trend == MarketRegime.STRONG_TREND_BULL

    # 4. Random walk noise
    regime_noise = regime_service.classify_regime(
        hurst=0.50, yz_vol=0.015, has_jump=False, cfg=regime_config
    )
    assert regime_noise == MarketRegime.RANDOM_WALK_NOISE
