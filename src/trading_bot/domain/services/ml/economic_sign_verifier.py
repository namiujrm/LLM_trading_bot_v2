"""Economic Sign Verifier Pure Python Domain Service (Clean Architecture, Invariant #9).

Strictly pure Python: ZERO dependencies on numpy, scipy, sklearn, lightgbm, or polars.
Enforces Invariant #9: zero defaults, zero class field defaults, zero silent fallbacks.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from trading_bot.domain.entities.ml.ml_configs import (
    LGBMMonotonicConstraints,
    TreeSHAPConfig,
)
from trading_bot.domain.entities.ml.ml_predictions import (
    SHAPDriverContribution,
    SHAPExplanationResult,
)


def verify_shap_sign_consistency(
    feature_value: float,
    shap_value: float,
    constraint_direction: int,
    baseline_value: float,
    feature_deadband_tol: float,
    attribution_deadband_tol: float,
) -> tuple[int, int, bool]:
    """Pure mathematical function evaluating single-feature economic sign consistency.

    Args:
        feature_value: Realized feature value x_j.
        shap_value: Realized TreeSHAP attribution phi_j.
        constraint_direction: Monotonic constraint M_j in {-1, 0, 1}.
        baseline_value: Neutral economic baseline b_j.
        feature_deadband_tol: Deadband tolerance for feature deviation.
        attribution_deadband_tol: Deadband tolerance for attribution magnitude.

    Returns:
        tuple (s_expected, s_actual, is_sign_consistent).
    """
    dev = feature_value - baseline_value
    if dev > feature_deadband_tol:
        d_j = 1
    elif dev < -feature_deadband_tol:
        d_j = -1
    else:
        d_j = 0

    s_expected = d_j * constraint_direction

    if shap_value > attribution_deadband_tol:
        s_actual = 1
    elif shap_value < -attribution_deadband_tol:
        s_actual = -1
    else:
        s_actual = 0

    is_consistent = s_expected * s_actual >= 0
    return s_expected, s_actual, is_consistent


def rank_top_drivers(
    contributions: Sequence[SHAPDriverContribution],
    top_k_bullish: int,
    top_k_bearish: int,
    attribution_deadband_tol: float,
) -> tuple[tuple[SHAPDriverContribution, ...], tuple[SHAPDriverContribution, ...]]:
    """Pure mathematical function ranking and extracting top bullish and bearish drivers.

    Deterministic tie-breaking:
    - Bullish: descending by shap_value, then ascending by feature_name.
    - Bearish: ascending by shap_value (most negative first), then ascending by feature_name.

    Args:
        contributions: Sequence of SHAPDriverContribution instances.
        top_k_bullish: Maximum number of bullish drivers to select.
        top_k_bearish: Maximum number of bearish drivers to select.
        attribution_deadband_tol: Magnitude threshold below which attributions are ignored.

    Returns:
        tuple (top_bullish_tuple, top_bearish_tuple).
    """
    bullish_candidates = [c for c in contributions if c.shap_value > attribution_deadband_tol]
    bearish_candidates = [c for c in contributions if c.shap_value < -attribution_deadband_tol]

    bullish_candidates.sort(key=lambda c: (-c.shap_value, c.feature_name))
    bearish_candidates.sort(key=lambda c: (c.shap_value, c.feature_name))

    return tuple(bullish_candidates[:top_k_bullish]), tuple(bearish_candidates[:top_k_bearish])


class EconomicSignVerifier:
    """100% pure Python domain service verifying economic sign invariants and ranking drivers.

    Zero dependencies on heavy computation libraries (NO numpy, NO scipy, NO sklearn, NO lightgbm).
    Adheres strictly to Invariant #9: zero defaults, explicit types, fail-fast validation.
    """

    def __init__(
        self,
        config: TreeSHAPConfig,
        monotonic_constraints: LGBMMonotonicConstraints,
    ) -> None:
        """Initialize domain verifier with explicit configuration and monotonic constraints."""
        self._config: TreeSHAPConfig = config
        self._constraints: LGBMMonotonicConstraints = monotonic_constraints

    @property
    def config(self) -> TreeSHAPConfig:
        """Return the injected TreeSHAPConfig."""
        return self._config

    @property
    def constraints(self) -> LGBMMonotonicConstraints:
        """Return the injected LGBMMonotonicConstraints."""
        return self._constraints

    def verify_and_rank(
        self,
        feature_names: Sequence[str],
        feature_values: Sequence[float],
        shap_values: Sequence[float],
    ) -> tuple[
        tuple[SHAPDriverContribution, ...],
        tuple[SHAPDriverContribution, ...],
        tuple[SHAPDriverContribution, ...],
        bool,
    ]:
        """Verify all feature attributions and rank top drivers.

        Returns:
            tuple of (all_contributions, top_bullish_drivers, top_bearish_drivers, is_economic_consistent).
        """
        if not (len(feature_names) == len(feature_values) == len(shap_values)):
            raise ValueError(
                f"Dimension mismatch: names={len(feature_names)}, "
                f"values={len(feature_values)}, shaps={len(shap_values)}"
            )
        if len(feature_names) == 0:
            raise ValueError("feature_names cannot be empty")

        contributions: list[SHAPDriverContribution] = []
        all_consistent: bool = True

        for name, val, phi in zip(feature_names, feature_values, shap_values, strict=True):
            if name not in self._constraints.constraints:
                raise ValueError(f"Unknown feature '{name}' not found in monotonic constraints")
            if name not in self._constraints.neutral_baselines:
                raise ValueError(f"Feature '{name}' missing neutral baseline")
            direction = self._constraints.constraints[name]
            baseline = self._constraints.neutral_baselines[name]

            s_expected, _, is_consistent = verify_shap_sign_consistency(
                feature_value=val,
                shap_value=phi,
                constraint_direction=direction,
                baseline_value=baseline,
                feature_deadband_tol=self._config.feature_deadband_tol,
                attribution_deadband_tol=self._config.attribution_deadband_tol,
            )

            if not is_consistent:
                all_consistent = False

            contributions.append(
                SHAPDriverContribution(
                    feature_name=name,
                    feature_value=float(val),
                    shap_value=float(phi),
                    expected_sign=int(s_expected),
                    is_sign_consistent=bool(is_consistent),
                )
            )

        top_bull, top_bear = rank_top_drivers(
            contributions=contributions,
            top_k_bullish=self._config.top_k_bullish,
            top_k_bearish=self._config.top_k_bearish,
            attribution_deadband_tol=self._config.attribution_deadband_tol,
        )

        return tuple(contributions), top_bull, top_bear, all_consistent

    def create_explanation_result(
        self,
        symbol: str,
        bar_time: datetime,
        model_type: str,
        base_value: float,
        raw_margin: float,
        feature_names: Sequence[str],
        feature_values: Sequence[float],
        shap_values: Sequence[float],
    ) -> SHAPExplanationResult:
        """Create a fully validated SHAPExplanationResult domain entity."""
        contribs, top_bull, top_bear, is_consistent = self.verify_and_rank(
            feature_names=feature_names,
            feature_values=feature_values,
            shap_values=shap_values,
        )
        return SHAPExplanationResult(
            symbol=symbol,
            bar_time=bar_time,
            model_type=model_type,
            base_value=base_value,
            raw_margin=raw_margin,
            contributions=contribs,
            top_bullish_drivers=top_bull,
            top_bearish_drivers=top_bear,
            is_economic_consistent=is_consistent,
        )
