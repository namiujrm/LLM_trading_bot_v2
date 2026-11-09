"""Dual-Binary Coherence Gate Domain Service — Zero Defaults (Invariant #9).

Pure Python domain service implementing dual-binary directional coherence enforcement (Proposal P3).
Enforces the probabilistic axiom that the upper profit barrier and lower loss barrier
are mutually exclusive events within the same evaluation horizon:
    P_long + P_short <= 1.0

When independent Long and Short models predict conflicting probabilities (sum > 1.0),
an attenuation factor is applied:
    penalty_factor = gamma_conflict * (S - 1.0)
    multiplier = max(min_floor, 1.0 - penalty_factor)
    P_long_adj = P_raw_long * multiplier
    P_short_adj = P_raw_short * multiplier

Strict Clean Architecture:
- 100% Pure Python (only `math`, `collections.abc.Sequence`, and domain entities).
- ZERO external computational dependencies (NO numpy, NO scipy, NO sklearn, NO lightgbm).
- Invariant #9: zero default parameters, zero class field defaults, zero silent fallbacks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from trading_bot.domain.entities.ml.ml_configs import CoherenceGateConfig
from trading_bot.domain.entities.ml.ml_predictions import BarrierPredictionResult


class CoherenceGate:
    """Pure domain service for enforcing dual-binary directional coherence.

    Invariant #9: All dependencies and configurations are explicitly injected.
    """

    def __init__(self, config: CoherenceGateConfig) -> None:
        """Initialize the CoherenceGate service.

        Args:
            config: Explicit CoherenceGateConfig instance.
        """
        self._config: CoherenceGateConfig = config

    @property
    def config(self) -> CoherenceGateConfig:
        """Return the injected CoherenceGateConfig."""
        return self._config

    def enforce_coherence(
        self,
        p_raw_long: float,
        p_raw_short: float,
    ) -> BarrierPredictionResult:
        """Enforce dual-binary coherence on a pair of directional probabilities.

        Args:
            p_raw_long: Raw predicted probability for Long upper barrier in [0, 1].
            p_raw_short: Raw predicted probability for Short lower barrier in [0, 1].

        Returns:
            BarrierPredictionResult containing adjusted probabilities and coherence metrics.

        Raises:
            ValueError: If input probabilities are not finite or outside [0, 1].
        """
        if not math.isfinite(p_raw_long):
            raise ValueError(f"p_raw_long must be a finite float, received {p_raw_long}")
        if not math.isfinite(p_raw_short):
            raise ValueError(f"p_raw_short must be a finite float, received {p_raw_short}")
        if not (0.0 <= p_raw_long <= 1.0):
            raise ValueError(f"p_raw_long must be in range [0, 1], received {p_raw_long}")
        if not (0.0 <= p_raw_short <= 1.0):
            raise ValueError(f"p_raw_short must be in range [0, 1], received {p_raw_short}")

        raw_sum = p_raw_long + p_raw_short
        max_comb = self._config.max_combined_probability
        _float_tol: float = 1e-9

        if raw_sum <= max_comb + _float_tol:
            # Probabilities are mutually coherent within numerical tolerance
            p_long_adj = min(p_raw_long, max_comb)
            p_short_adj = min(p_raw_short, max_comb)
            post_sum = p_long_adj + p_short_adj
            if post_sum > max_comb and post_sum > 0.0:
                scale = max_comb / post_sum
                p_long_adj = p_long_adj * scale
                p_short_adj = p_short_adj * scale
            return BarrierPredictionResult(
                p_long=p_long_adj,
                p_short=p_short_adj,
                p_raw_long=p_raw_long,
                p_raw_short=p_raw_short,
                is_coherent=True,
                coherence_penalty=0.0,
            )

        # Conflict detected: raw_sum > max_comb
        excess = raw_sum - max_comb
        penalty_factor = self._config.gamma_conflict * excess
        coherence_penalty = min(1.0, penalty_factor)

        attenuation_multiplier = max(
            self._config.min_probability_floor,
            1.0 - penalty_factor,
        )

        p_long_adj = max(self._config.min_probability_floor, p_raw_long * attenuation_multiplier)
        p_short_adj = max(self._config.min_probability_floor, p_raw_short * attenuation_multiplier)

        # Ensure post_sum <= max_comb invariant holds under any gamma
        post_sum = p_long_adj + p_short_adj
        if post_sum > max_comb and post_sum > 0.0:
            scale = max_comb / post_sum
            p_long_adj = p_long_adj * scale
            p_short_adj = p_short_adj * scale

        # Guarantee non-inflation
        p_long_adj = min(p_raw_long, p_long_adj)
        p_short_adj = min(p_raw_short, p_short_adj)

        return BarrierPredictionResult(
            p_long=p_long_adj,
            p_short=p_short_adj,
            p_raw_long=p_raw_long,
            p_raw_short=p_raw_short,
            is_coherent=False,
            coherence_penalty=coherence_penalty,
        )

    def enforce_batch(
        self,
        raw_pairs: Sequence[tuple[float, float]],
    ) -> list[BarrierPredictionResult]:
        """Enforce coherence on a sequence of (p_raw_long, p_raw_short) pairs.

        Args:
            raw_pairs: Sequence of (p_raw_long, p_raw_short) tuples.

        Returns:
            List of BarrierPredictionResult objects.
        """
        return [self.enforce_coherence(p_l, p_s) for p_l, p_s in raw_pairs]


def enforce_dual_binary_coherence(
    p_raw_long: float,
    p_raw_short: float,
    config: CoherenceGateConfig,
) -> BarrierPredictionResult:
    """Convenience pure functional entrypoint for single-prediction coherence gating.

    Args:
        p_raw_long: Raw Long barrier probability.
        p_raw_short: Raw Short barrier probability.
        config: Explicit CoherenceGateConfig instance.

    Returns:
        BarrierPredictionResult instance.
    """
    gate = CoherenceGate(config=config)
    return gate.enforce_coherence(p_raw_long=p_raw_long, p_raw_short=p_raw_short)
