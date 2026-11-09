"""Semantic Quant Packet Assembler Service (Zero Defaults).

Pure Domain Service implementing Invariant #9:
- Assembles immutable SemanticQuantPacket domain entity
- Constructs and normalizes the canonical 16-dimensional trajectory vector
- Enforces strict numerical stability verification: zero NaN, zero Inf, bounded floats
- Raises NumericalInstabilityException immediately upon encountering non-finite floats
"""

from __future__ import annotations

import math

from trading_bot.domain.entities.common.enums import MarketRegime, OIPriceQuadrant
from trading_bot.domain.entities.market.market_state import FeatureSignalItem, SemanticQuantPacket
from trading_bot.domain.exceptions import NumericalInstabilityException


class SemanticQuantPacketAssembler:
    """Pure domain assembler and validator for SemanticQuantPacket."""

    def build_trajectory_vector_16d(
        self,
        vpin: float,
        kyle_lambda: float,
        ofi_zscore: float,
        obi_mean: float,
        corwin_schultz_spread_bps: float,
        noise_variance_ratio: float,
        hurst_exponent: float,
        yang_zhang_vol_ratio: float,
        bipower_jump_z: float,
        permutation_entropy: float,
        fracdiff_return_d04: float,
        oi_delta_zscore: float,
        funding_velocity_z: float,
        liq_imbalance_ratio: float,
        btc_lead_lag_gap: float,
        composite_alpha_score: float,
    ) -> list[float]:
        """Normalize raw quantitative indicators into the canonical 16D trajectory vector."""
        raw_items = [
            vpin,
            kyle_lambda,
            ofi_zscore,
            obi_mean,
            corwin_schultz_spread_bps,
            noise_variance_ratio,
            hurst_exponent,
            yang_zhang_vol_ratio,
            bipower_jump_z,
            permutation_entropy,
            fracdiff_return_d04,
            oi_delta_zscore,
            funding_velocity_z,
            liq_imbalance_ratio,
            btc_lead_lag_gap,
            composite_alpha_score,
        ]

        for idx, val in enumerate(raw_items):
            if not math.isfinite(val):
                raise NumericalInstabilityException(
                    f"Non-finite value {val} at raw index {idx} in trajectory vector builder"
                )

        # Dimension 1: VPIN clipped to [0, 1]
        d1 = min(1.0, max(0.0, vpin))

        # Dimension 2: Kyle Lambda log-scaled and tanh bounded
        d2 = math.tanh(math.log1p(max(0.0, kyle_lambda * 1e6)))

        # Dimension 3: OFI Z-score scaled and clipped to [-1, 1]
        d3 = min(1.0, max(-1.0, ofi_zscore / 3.0))

        # Dimension 4: OBI Mean bounded to [-1, 1]
        d4 = min(1.0, max(-1.0, obi_mean))

        # Dimension 5: Corwin-Schultz Spread in bps normalized via tanh
        d5 = math.tanh(max(0.0, corwin_schultz_spread_bps) / 20.0)

        # Dimension 6: Microstructure Noise Ratio clipped to [0, 1]
        d6 = min(1.0, max(0.0, noise_variance_ratio))

        # Dimension 7: Hurst Exponent centered at 0.5 and clipped to [-1, 1]
        d7 = min(1.0, max(-1.0, (hurst_exponent - 0.5) / 0.5))

        # Dimension 8: Yang-Zhang Volatility Ratio normalized via tanh
        d8 = math.tanh(yang_zhang_vol_ratio)

        # Dimension 9: Bipower Jump Z-score scaled and clipped to [-1, 1]
        d9 = min(1.0, max(-1.0, bipower_jump_z / 3.0))

        # Dimension 10: Permutation Entropy clipped to [0, 1]
        d10 = min(1.0, max(0.0, permutation_entropy))

        # Dimension 11: FracDiff Return normalized via tanh
        d11 = math.tanh(fracdiff_return_d04)

        # Dimension 12: OI Delta Z-score scaled and clipped to [-1, 1]
        d12 = min(1.0, max(-1.0, oi_delta_zscore / 3.0))

        # Dimension 13: Funding Velocity Z-score scaled and clipped to [-1, 1]
        d13 = min(1.0, max(-1.0, funding_velocity_z / 3.0))

        # Dimension 14: Liquidation Imbalance Ratio bounded to [-1, 1]
        d14 = min(1.0, max(-1.0, liq_imbalance_ratio))

        # Dimension 15: BTC Lead-Lag Gap clipped to [-1, 1]
        d15 = min(1.0, max(-1.0, btc_lead_lag_gap / 2.0))

        # Dimension 16: Composite Alpha Score bounded via tanh
        d16 = math.tanh(composite_alpha_score)

        vector_16d = [
            d1,
            d2,
            d3,
            d4,
            d5,
            d6,
            d7,
            d8,
            d9,
            d10,
            d11,
            d12,
            d13,
            d14,
            d15,
            d16,
        ]

        self.validate_vector_finite(vector_16d)
        return vector_16d

    def validate_vector_finite(self, vector: list[float]) -> None:
        """Validate that a float vector has exactly 16 dimensions and contains 0 NaN and 0 Inf."""
        if len(vector) != 16:
            raise NumericalInstabilityException(
                f"Trajectory vector must have exactly 16 dimensions, got {len(vector)}"
            )

        for idx, val in enumerate(vector):
            if not math.isfinite(val):
                raise NumericalInstabilityException(
                    f"Non-finite float value {val} at index {idx} in trajectory vector"
                )

    def assemble_packet(
        self,
        symbol: str,
        timestamp_ms: int,
        active_features: list[FeatureSignalItem],
        market_regime: MarketRegime,
        oi_quadrant: OIPriceQuadrant,
        trajectory_vector_16d: list[float],
    ) -> SemanticQuantPacket:
        """Assemble and strictly validate a SemanticQuantPacket."""
        if not symbol:
            raise NumericalInstabilityException("Symbol cannot be empty in SemanticQuantPacket")
        if timestamp_ms <= 0:
            raise NumericalInstabilityException(
                f"Invalid timestamp_ms {timestamp_ms} in SemanticQuantPacket"
            )

        # Validate trajectory vector
        self.validate_vector_finite(trajectory_vector_16d)

        # Validate active features
        for item in active_features:
            if not math.isfinite(item.raw_value):
                raise NumericalInstabilityException(
                    f"Feature '{item.feature_name}' has non-finite raw_value {item.raw_value}"
                )
            if not math.isfinite(item.z_score):
                raise NumericalInstabilityException(
                    f"Feature '{item.feature_name}' has non-finite z_score {item.z_score}"
                )

        return SemanticQuantPacket(
            symbol=symbol,
            timestamp_ms=timestamp_ms,
            active_features=active_features,
            market_regime=market_regime,
            oi_quadrant=oi_quadrant,
            trajectory_vector_16d=trajectory_vector_16d,
        )
