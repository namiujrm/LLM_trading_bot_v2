"""Unit tests for MicrostructureMathService (Zero Defaults).

Verifies VPIN, Intra-Bar Kyle's Lambda, VA-ILLIQ, Corwin-Schultz Spread,
and MNVR under normal, extreme, and degenerate boundary conditions.
"""

from __future__ import annotations

import math

import pytest

from trading_bot.domain.entities.market.market_state import (
    EnrichedMicroBar1m,
    MicroBar1m,
    MultiTFBar,
)
from trading_bot.domain.entities.quant.quant_configs import VPINConfig
from trading_bot.domain.exceptions import (
    InsufficientDataException,
    ZeroVolumeException,
)
from trading_bot.domain.services.quant.microstructure_math import MicrostructureMathService


@pytest.fixture
def vpin_config() -> VPINConfig:
    return VPINConfig(
        num_buckets=5,
        adv_lookback_days=20,
        volume_bucket_slice_ratio=0.02,
        toxicity_high_threshold=0.75,
        division_eps=1e-12,
    )


@pytest.fixture
def micro_service() -> MicrostructureMathService:
    return MicrostructureMathService()


def test_vpin_all_buy_flow_yields_maximum_toxicity(
    micro_service: MicrostructureMathService,
    vpin_config: VPINConfig,
) -> None:
    # 10 bars with 100% buyer volume
    adv_usd = 100000.0
    # bucket size = 100000.0 * 0.02 = 2000.0 USD
    bars: list[EnrichedMicroBar1m] = []
    base_ts = 1700000000000
    for i in range(10):
        bars.append(
            EnrichedMicroBar1m(
                timestamp_ms=base_ts + i * 60000,
                open=100.0 + i,
                high=102.0 + i,
                low=100.0 + i,
                close=102.0 + i,
                volume=20.0,
                taker_buy_volume=20.0,
                quote_volume=2040.0,
            )
        )

    vpin = micro_service.compute_vpin(bars, adv_usd, vpin_config)
    assert 0.99 <= vpin <= 1.0


def test_vpin_balanced_order_flow_yields_minimum_toxicity(
    micro_service: MicrostructureMathService,
    vpin_config: VPINConfig,
) -> None:
    adv_usd = 100000.0
    bars: list[EnrichedMicroBar1m] = []
    base_ts = 1700000000000
    for i in range(10):
        bars.append(
            EnrichedMicroBar1m(
                timestamp_ms=base_ts + i * 60000,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=20.0,
                taker_buy_volume=10.0,  # 50% buy, 50% sell
                quote_volume=2000.0,
            )
        )

    vpin = micro_service.compute_vpin(bars, adv_usd, vpin_config)
    assert 0.0 <= vpin <= 0.05


def test_vpin_standard_micro_bar_bvc_fallback(
    micro_service: MicrostructureMathService,
    vpin_config: VPINConfig,
) -> None:
    adv_usd = 10000.0
    bars: list[MicroBar1m] = []
    base_ts = 1700000000000
    for i in range(10):
        bars.append(
            MicroBar1m(
                timestamp_ms=base_ts + i * 60000,
                open=100.0,
                high=105.0,
                low=95.0,
                close=105.0,  # Close at high -> buy ratio = 1.0
                volume=10.0,
            )
        )

    vpin = micro_service.compute_vpin(bars, adv_usd, vpin_config)
    assert vpin > 0.80


def test_vpin_rejects_non_positive_adv(
    micro_service: MicrostructureMathService,
    vpin_config: VPINConfig,
) -> None:
    bars = [
        MicroBar1m(
            timestamp_ms=1700000000000,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10.0,
        )
    ]
    with pytest.raises(ZeroVolumeException):
        micro_service.compute_vpin(bars, 0.0, vpin_config)

    with pytest.raises(ZeroVolumeException):
        micro_service.compute_vpin(bars, -500.0, vpin_config)


def test_vpin_rejects_empty_bars(
    micro_service: MicrostructureMathService,
    vpin_config: VPINConfig,
) -> None:
    with pytest.raises(InsufficientDataException):
        micro_service.compute_vpin([], 100000.0, vpin_config)


def test_vpin_insufficient_volume_for_single_bucket(
    micro_service: MicrostructureMathService,
    vpin_config: VPINConfig,
) -> None:
    adv_usd = 100000000.0  # Bucket size = 2,000,000 USD
    bars = [
        MicroBar1m(
            timestamp_ms=1700000000000,
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=1.0,  # Only $1 volume
        )
    ]
    with pytest.raises(InsufficientDataException):
        micro_service.compute_vpin(bars, adv_usd, vpin_config)


def test_kyle_lambda_positive_impact_and_slippage(
    micro_service: MicrostructureMathService,
    vpin_config: VPINConfig,
) -> None:
    bars: list[MicroBar1m] = []
    base_ts = 1700000000000
    for i in range(30):
        bars.append(
            MicroBar1m(
                timestamp_ms=base_ts + i * 60000,
                open=100.0 + i * 0.1,
                high=100.5 + i * 0.1,
                low=99.8 + i * 0.1,
                close=100.4 + i * 0.1,
                volume=100.0,
            )
        )

    kyle_lambda, slippage = micro_service.compute_kyle_lambda(bars, vpin_config)
    assert kyle_lambda >= 0.0
    assert slippage >= 0.0001
    assert math.isfinite(kyle_lambda)
    assert math.isfinite(slippage)


def test_kyle_lambda_empty_bars_raises(
    micro_service: MicrostructureMathService,
    vpin_config: VPINConfig,
) -> None:
    with pytest.raises(InsufficientDataException):
        micro_service.compute_kyle_lambda([], vpin_config)


def test_estimate_execution_slippage(
    micro_service: MicrostructureMathService,
) -> None:
    kyle_lambda = 0.000005
    order_size_usd = 20000.0
    spread_bps = 4.0

    slippage = micro_service.estimate_execution_slippage(kyle_lambda, order_size_usd, spread_bps)
    # 0.5 * 4 * 1e-4 + 0.000005 * 20000 = 0.0002 + 0.1 = 0.1002
    assert slippage == pytest.approx(0.1002, rel=1e-4)

    # Floor test
    low_slippage = micro_service.estimate_execution_slippage(0.0, 0.0, 0.0)
    assert low_slippage == 0.0001


def test_va_amihud_calculation(
    micro_service: MicrostructureMathService,
    vpin_config: VPINConfig,
) -> None:
    bars: list[MicroBar1m] = []
    base_ts = 1700000000000
    for i in range(20):
        bars.append(
            MicroBar1m(
                timestamp_ms=base_ts + i * 60000,
                open=100.0,
                high=102.0,
                low=99.0,
                close=101.0,
                volume=50.0,
            )
        )

    yz_vol = 0.015
    va_amihud = micro_service.compute_va_amihud(bars, yz_vol, vpin_config)
    assert va_amihud > 0.0
    assert math.isfinite(va_amihud)


def test_corwin_schultz_spread_estimation(
    micro_service: MicrostructureMathService,
    vpin_config: VPINConfig,
) -> None:
    base_ts = 1700000000000
    bars = [
        MultiTFBar(
            symbol="BTCUSDT",
            timeframe="30m",
            timestamp_ms=base_ts,
            open=50000.0,
            high=50500.0,
            low=49500.0,
            close=50200.0,
            volume=1000.0,
        ),
        MultiTFBar(
            symbol="BTCUSDT",
            timeframe="30m",
            timestamp_ms=base_ts + 1800000,
            open=50200.0,
            high=50600.0,
            low=49800.0,
            close=50400.0,
            volume=1100.0,
        ),
    ]

    spread_bps = micro_service.compute_corwin_schultz_spread(bars, vpin_config)
    assert spread_bps >= 0.0
    assert math.isfinite(spread_bps)


def test_corwin_schultz_single_bar_raises(
    micro_service: MicrostructureMathService,
    vpin_config: VPINConfig,
) -> None:
    bars = [
        MultiTFBar(
            symbol="BTCUSDT",
            timeframe="30m",
            timestamp_ms=1700000000000,
            open=50000.0,
            high=50500.0,
            low=49500.0,
            close=50200.0,
            volume=1000.0,
        )
    ]
    with pytest.raises(InsufficientDataException):
        micro_service.compute_corwin_schultz_spread(bars, vpin_config)


def test_mnvr_bounds(
    micro_service: MicrostructureMathService,
    vpin_config: VPINConfig,
) -> None:
    # High noise ratio: rv_1m = 0.0010, rv_15m = 0.0004 -> (0.0010 - 0.0004) / 0.0010 = 0.60
    mnvr = micro_service.compute_mnvr(0.0010, 0.0004, vpin_config)
    assert mnvr == pytest.approx(0.60, abs=1e-4)

    # Negative excess (rv_1m < rv_15m) clamped to 0
    clamped_mnvr = micro_service.compute_mnvr(0.0002, 0.0005, vpin_config)
    assert clamped_mnvr == 0.0
