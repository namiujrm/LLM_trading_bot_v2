"""Unit tests for SemanticQuantPacketAssembler (M3.5).

Verifies:
- 16D trajectory vector normalization and bounding
- Strict 0 NaN / 0 Inf numerical stability checking
- SemanticQuantPacket assembly and validation
- Error handling for degenerate or malicious inputs
"""

from __future__ import annotations

import math

import pytest

from trading_bot.domain.entities.common.enums import MarketRegime, OIPriceQuadrant
from trading_bot.domain.entities.market.market_state import FeatureSignalItem, SemanticQuantPacket
from trading_bot.domain.exceptions import NumericalInstabilityException
from trading_bot.domain.services.quant.packet_assembler import SemanticQuantPacketAssembler


def test_build_trajectory_vector_16d_bounds_and_finite() -> None:
    assembler = SemanticQuantPacketAssembler()

    vec = assembler.build_trajectory_vector_16d(
        vpin=0.85,
        kyle_lambda=1.5e-5,
        ofi_zscore=2.5,
        obi_mean=-0.4,
        corwin_schultz_spread_bps=12.5,
        noise_variance_ratio=0.35,
        hurst_exponent=0.62,
        yang_zhang_vol_ratio=0.15,
        bipower_jump_z=1.2,
        permutation_entropy=0.78,
        fracdiff_return_d04=0.03,
        oi_delta_zscore=-1.8,
        funding_velocity_z=0.9,
        liq_imbalance_ratio=0.25,
        btc_lead_lag_gap=-0.6,
        composite_alpha_score=0.45,
    )

    assert len(vec) == 16
    for val in vec:
        assert math.isfinite(val)
        assert -1.0 <= val <= 1.0


def test_build_trajectory_vector_16d_rejects_nan_or_inf() -> None:
    assembler = SemanticQuantPacketAssembler()

    with pytest.raises(NumericalInstabilityException):
        assembler.build_trajectory_vector_16d(
            vpin=float("nan"),
            kyle_lambda=1.0,
            ofi_zscore=0.0,
            obi_mean=0.0,
            corwin_schultz_spread_bps=0.0,
            noise_variance_ratio=0.0,
            hurst_exponent=0.5,
            yang_zhang_vol_ratio=0.0,
            bipower_jump_z=0.0,
            permutation_entropy=0.5,
            fracdiff_return_d04=0.0,
            oi_delta_zscore=0.0,
            funding_velocity_z=0.0,
            liq_imbalance_ratio=0.0,
            btc_lead_lag_gap=0.0,
            composite_alpha_score=0.0,
        )

    with pytest.raises(NumericalInstabilityException):
        assembler.build_trajectory_vector_16d(
            vpin=0.5,
            kyle_lambda=float("inf"),
            ofi_zscore=0.0,
            obi_mean=0.0,
            corwin_schultz_spread_bps=0.0,
            noise_variance_ratio=0.0,
            hurst_exponent=0.5,
            yang_zhang_vol_ratio=0.0,
            bipower_jump_z=0.0,
            permutation_entropy=0.5,
            fracdiff_return_d04=0.0,
            oi_delta_zscore=0.0,
            funding_velocity_z=0.0,
            liq_imbalance_ratio=0.0,
            btc_lead_lag_gap=0.0,
            composite_alpha_score=0.0,
        )


def test_validate_vector_finite_checks_length_and_finiteness() -> None:
    assembler = SemanticQuantPacketAssembler()

    # Wrong length
    with pytest.raises(NumericalInstabilityException):
        assembler.validate_vector_finite([0.0] * 15)

    with pytest.raises(NumericalInstabilityException):
        assembler.validate_vector_finite([0.0] * 17)

    # Contains NaN
    bad_vec = [0.0] * 16
    bad_vec[5] = float("nan")
    with pytest.raises(NumericalInstabilityException):
        assembler.validate_vector_finite(bad_vec)


def test_assemble_packet_success() -> None:
    assembler = SemanticQuantPacketAssembler()

    vec_16d = [0.1 * i for i in range(16)]
    features = [
        FeatureSignalItem(
            feature_name="vpin_imbalance",
            raw_value=0.72,
            z_score=1.85,
            is_statistically_active=True,
            economic_direction=-1,
        )
    ]

    packet: SemanticQuantPacket = assembler.assemble_packet(
        symbol="BTCUSDT",
        timestamp_ms=1700000000000,
        active_features=features,
        market_regime=MarketRegime.STRONG_TREND_BULL,
        oi_quadrant=OIPriceQuadrant.LONG_BUILDUP,
        trajectory_vector_16d=vec_16d,
    )

    assert packet.symbol == "BTCUSDT"
    assert packet.timestamp_ms == 1700000000000
    assert packet.market_regime == MarketRegime.STRONG_TREND_BULL
    assert packet.oi_quadrant == OIPriceQuadrant.LONG_BUILDUP
    assert len(packet.trajectory_vector_16d) == 16
    assert len(packet.active_features) == 1


def test_assemble_packet_rejects_non_finite_features() -> None:
    assembler = SemanticQuantPacketAssembler()
    vec_16d = [0.0] * 16

    features_nan = [
        FeatureSignalItem(
            feature_name="bad_feature",
            raw_value=float("nan"),
            z_score=1.0,
            is_statistically_active=True,
            economic_direction=1,
        )
    ]

    with pytest.raises(NumericalInstabilityException):
        assembler.assemble_packet(
            symbol="ETHUSDT",
            timestamp_ms=1700000000000,
            active_features=features_nan,
            market_regime=MarketRegime.MEAN_REVERTING_RANGE,
            oi_quadrant=OIPriceQuadrant.NEUTRAL,
            trajectory_vector_16d=vec_16d,
        )
