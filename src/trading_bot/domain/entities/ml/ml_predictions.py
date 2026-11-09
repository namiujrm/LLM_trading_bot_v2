"""Machine Learning Prediction Entities and OOD Detection Results (Zero Defaults).

Pure Python and Pydantic domain models for Stage 4 machine learning inference.
Adheres strictly to Invariant #9 (Zero Default Class Fields).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, model_validator


class OODDetectionResult(BaseModel):
    """Result of 6D Subspace Conformal Mahalanobis Out-of-Distribution Detection.

    Adheres strictly to Invariant #9: zero default values, zero silent fallbacks.
    """

    is_ood: bool
    mahalanobis_distance: float
    empirical_quantile_threshold: float
    excess_distance: float
    gamma_ood: float
    feature_contributions: tuple[float, ...]

    @model_validator(mode="after")
    def _validate_invariants(self) -> OODDetectionResult:
        if self.mahalanobis_distance < 0.0:
            raise ValueError(
                f"mahalanobis_distance must be non-negative, received {self.mahalanobis_distance}"
            )
        if self.empirical_quantile_threshold <= 0.0:
            raise ValueError(
                "empirical_quantile_threshold must be strictly positive, "
                f"received {self.empirical_quantile_threshold}"
            )
        if self.excess_distance < 0.0:
            raise ValueError(
                f"excess_distance must be non-negative, received {self.excess_distance}"
            )
        if not (0.0 <= self.gamma_ood <= 1.0):
            raise ValueError(
                f"gamma_ood must be in closed interval [0.0, 1.0], received {self.gamma_ood}"
            )
        return self

    @property
    def threshold(self) -> float:
        """Alias for empirical_quantile_threshold."""
        return self.empirical_quantile_threshold

    @property
    def shrinkage_factor(self) -> float:
        """Alias for gamma_ood."""
        return self.gamma_ood


class BarrierPredictionResult(BaseModel):
    """Result of dual-binary causal barrier prediction with coherence enforcement.

    Adheres strictly to Invariant #9: zero defaults, explicit types, fail-fast validation.

    Fields:
        p_long: Post-coherence adjusted probability of Long upper barrier hit.
        p_short: Post-coherence adjusted probability of Short lower barrier hit.
        p_raw_long: Raw model probability for Long barrier before coherence attenuation.
        p_raw_short: Raw model probability for Short barrier before coherence attenuation.
        is_coherent: True if p_raw_long + p_raw_short <= 1.0, False if conflict was detected.
        coherence_penalty: Magnitude of attenuation penalty applied (0.0 if coherent).
    """

    p_long: float
    p_short: float
    p_raw_long: float
    p_raw_short: float
    is_coherent: bool
    coherence_penalty: float

    @model_validator(mode="after")
    def _validate_invariants(self) -> BarrierPredictionResult:
        # Range checks for raw probabilities
        if not (0.0 <= self.p_raw_long <= 1.0):
            raise ValueError(f"p_raw_long must be in range [0, 1], received {self.p_raw_long}")
        if not (0.0 <= self.p_raw_short <= 1.0):
            raise ValueError(f"p_raw_short must be in range [0, 1], received {self.p_raw_short}")

        # Range checks for adjusted probabilities
        if not (0.0 <= self.p_long <= 1.0):
            raise ValueError(f"p_long must be in range [0, 1], received {self.p_long}")
        if not (0.0 <= self.p_short <= 1.0):
            raise ValueError(f"p_short must be in range [0, 1], received {self.p_short}")

        # Non-negativity of penalty
        if self.coherence_penalty < 0.0:
            raise ValueError(
                f"coherence_penalty must be non-negative, received {self.coherence_penalty}"
            )

        # Non-inflation invariant: post-coherence probability cannot exceed raw probability
        _float_tol: float = 1e-7
        if self.p_long > self.p_raw_long + _float_tol:
            raise ValueError(
                f"p_long ({self.p_long}) cannot exceed raw probability p_raw_long ({self.p_raw_long})"
            )
        if self.p_short > self.p_raw_short + _float_tol:
            raise ValueError(
                f"p_short ({self.p_short}) cannot exceed raw probability p_raw_short ({self.p_raw_short})"
            )

        # Coherence guarantee invariant: sum of coherent probabilities must not exceed 1.0
        if self.p_long + self.p_short > 1.0 + _float_tol:
            raise ValueError(
                f"Coherence violation: p_long + p_short ({self.p_long + self.p_short}) > 1.0"
            )

        # Logical coherence state consistency
        raw_sum = self.p_raw_long + self.p_raw_short
        if self.is_coherent:
            if raw_sum > 1.0 + _float_tol:
                raise ValueError(f"is_coherent cannot be True when raw_sum ({raw_sum}) exceeds 1.0")
            if self.coherence_penalty > _float_tol:
                raise ValueError(
                    f"coherence_penalty must be 0.0 when coherent, received {self.coherence_penalty}"
                )

        return self

    @property
    def raw_sum(self) -> float:
        """Total unconstrained probability mass across both directions."""
        return self.p_raw_long + self.p_raw_short

    @property
    def coherent_sum(self) -> float:
        """Total regularized probability mass across both directions (<= 1.0)."""
        return self.p_long + self.p_short

    @property
    def raw_conflict_excess(self) -> float:
        """Excess probability above 1.0 before attenuation."""
        return max(0.0, (self.p_raw_long + self.p_raw_short) - 1.0)


DualBarrierPrediction = BarrierPredictionResult


class CalibratedProbability(BaseModel):
    """Calibrated probability point estimate and Venn-Abers uncertainty interval (Zero Defaults).

    Adheres strictly to Invariant #9: zero defaults, explicit types, fail-fast validation.

    Fields:
        p_calibrated: Point calibrated probability p_cal = p1 / (1 - p0 + p1) in [0, 1].
        p_lower: Lower Venn-Abers probability bound min(p0, p1) in [0, 1].
        p_upper: Upper Venn-Abers probability bound max(p0, p1) in [0, 1].
        uncertainty_width: Epistemic uncertainty interval width (p_upper - p_lower) in [0, 1].
        is_uncertain: True if uncertainty_width exceeds config uncertainty_threshold.
        raw_score: The uncalibrated decision score before calibration in [0, 1].
    """

    p_calibrated: float
    p_lower: float
    p_upper: float
    uncertainty_width: float
    is_uncertain: bool
    raw_score: float

    @model_validator(mode="after")
    def _validate_invariants(self) -> CalibratedProbability:
        _float_tol: float = 1e-7

        # Range checks
        if not (0.0 <= self.p_calibrated <= 1.0):
            raise ValueError(f"p_calibrated must be in range [0, 1], received {self.p_calibrated}")
        if not (0.0 <= self.p_lower <= 1.0):
            raise ValueError(f"p_lower must be in range [0, 1], received {self.p_lower}")
        if not (0.0 <= self.p_upper <= 1.0):
            raise ValueError(f"p_upper must be in range [0, 1], received {self.p_upper}")
        if not (0.0 <= self.raw_score <= 1.0):
            raise ValueError(f"raw_score must be in range [0, 1], received {self.raw_score}")

        # Bound ordering
        if self.p_lower > self.p_upper + _float_tol:
            raise ValueError(f"p_lower ({self.p_lower}) cannot exceed p_upper ({self.p_upper})")

        # Uncertainty width consistency
        if not (0.0 <= self.uncertainty_width <= 1.0):
            raise ValueError(
                f"uncertainty_width must be in range [0, 1], received {self.uncertainty_width}"
            )
        expected_width = self.p_upper - self.p_lower
        if abs(self.uncertainty_width - expected_width) > 1e-6:
            raise ValueError(
                f"uncertainty_width ({self.uncertainty_width}) must match p_upper - p_lower ({expected_width})"
            )

        # Venn-Abers mathematical containment guarantee: p_calibrated in [p_lower, p_upper]
        if not (self.p_lower - _float_tol <= self.p_calibrated <= self.p_upper + _float_tol):
            raise ValueError(
                f"p_calibrated ({self.p_calibrated}) must lie within interval [{self.p_lower}, {self.p_upper}]"
            )

        return self

    @property
    def is_confident(self) -> bool:
        """True if prediction uncertainty is within acceptable threshold."""
        return not self.is_uncertain

    @property
    def interval(self) -> tuple[float, float]:
        """Return the (p_lower, p_upper) conformal uncertainty interval."""
        return (self.p_lower, self.p_upper)


class ECEDriftRecord(BaseModel):
    """Record of Expected Calibration Error (ECE) and calibration drift assessment (Zero Defaults).

    Adheres strictly to Invariant #9: zero defaults, explicit types, fail-fast validation.

    Fields:
        eval_time: UTC timestamp when the drift assessment was evaluated.
        ece: Expected Calibration Error value across all bins in [0, 1].
        sample_count: Total number of evaluated prediction/outcome samples in the window.
        is_drift_detected: True if sample_count >= min_samples and ece > drift_warning_threshold.
        bin_accuracies: Tuple of empirical accuracies for each bin.
        bin_confidences: Tuple of mean predicted confidences for each bin.
        bin_counts: Tuple of sample counts in each bin.
    """

    eval_time: datetime
    ece: float
    sample_count: int
    is_drift_detected: bool
    bin_accuracies: tuple[float, ...]
    bin_confidences: tuple[float, ...]
    bin_counts: tuple[int, ...]

    @model_validator(mode="after")
    def _validate_invariants(self) -> ECEDriftRecord:
        if not (0.0 <= self.ece <= 1.0):
            raise ValueError(f"ece must be in range [0, 1], received {self.ece}")
        if self.sample_count < 0:
            raise ValueError(f"sample_count must be non-negative, received {self.sample_count}")

        # Equal length check across bin metrics
        n_accs = len(self.bin_accuracies)
        n_confs = len(self.bin_confidences)
        n_counts = len(self.bin_counts)
        if not (n_accs == n_confs == n_counts):
            raise ValueError(
                f"Mismatched bin tuple lengths: accuracies={n_accs}, confidences={n_confs}, counts={n_counts}"
            )

        # Sum of bin counts must equal sample_count
        if sum(self.bin_counts) != self.sample_count:
            raise ValueError(
                f"Sum of bin_counts ({sum(self.bin_counts)}) must equal sample_count ({self.sample_count})"
            )

        # Non-negativity and boundary checks
        for c in self.bin_counts:
            if c < 0:
                raise ValueError(f"Bin count must be non-negative, received {c}")
        for a in self.bin_accuracies:
            if not (0.0 <= a <= 1.0):
                raise ValueError(f"Bin accuracy must be in range [0, 1], received {a}")
        for cf in self.bin_confidences:
            if not (0.0 <= cf <= 1.0):
                raise ValueError(f"Bin confidence must be in range [0, 1], received {cf}")

        return self

    @property
    def n_bins(self) -> int:
        """Number of probability bins used in evaluation."""
        return len(self.bin_counts)

    @property
    def max_calibration_error(self) -> float:
        """Maximum calibration error (MCE) across all populated bins."""
        populated_errors = [
            abs(acc - conf)
            for acc, conf, count in zip(
                self.bin_accuracies, self.bin_confidences, self.bin_counts, strict=True
            )
            if count > 0
        ]
        if len(populated_errors) == 0:
            return 0.0
        return max(populated_errors)


class SHAPDriverContribution(BaseModel):
    """Individual feature SHAP attribution and economic sign verification status (Zero Defaults).

    Adheres strictly to Invariant #9: zero defaults, explicit types, fail-fast validation.

    Fields:
        feature_name: Canonical name of the feature.
        feature_value: Realized numerical value of the feature for this decision bar.
        shap_value: Additive TreeSHAP attribution value phi_j in margin (logit) space.
        expected_sign: Directional expected sign (+1, -1, or 0 if unconstrained/neutral).
        is_sign_consistent: True if the realized SHAP attribution satisfies economic sign constraint.
    """

    feature_name: str
    feature_value: float
    shap_value: float
    expected_sign: int
    is_sign_consistent: bool

    @model_validator(mode="after")
    def _validate_invariants(self) -> SHAPDriverContribution:
        if len(self.feature_name.strip()) == 0:
            raise ValueError("feature_name must not be empty or whitespace")
        if self.expected_sign not in (-1, 0, 1):
            raise ValueError(
                f"expected_sign must be -1, 0, or 1, received {self.expected_sign} "
                f"for feature '{self.feature_name}'"
            )
        return self

    @property
    def is_bullish(self) -> bool:
        """True if this feature exerts a positive attribution (phi_j > 0) towards the target."""
        return self.shap_value > 0.0

    @property
    def is_bearish(self) -> bool:
        """True if this feature exerts a negative attribution (phi_j < 0) against the target."""
        return self.shap_value < 0.0

    @property
    def abs_shap(self) -> float:
        """Absolute magnitude of the SHAP attribution."""
        return abs(self.shap_value)


class SHAPExplanationResult(BaseModel):
    """Complete TreeSHAP explanation result for a single decision bar (Zero Defaults).

    Adheres strictly to Invariant #9: zero defaults, explicit types, fail-fast validation.

    Fields:
        symbol: Market symbol (e.g. "BTCUSDT").
        bar_time: UTC timestamp of the decision bar.
        model_type: Model identifier (e.g. "long_barrier" or "short_barrier").
        base_value: Expected margin value across training distribution (phi_0).
        raw_margin: Total uncalibrated logit prediction margin f(x).
        contributions: Complete tuple of SHAPDriverContribution instances for all features.
        top_bullish_drivers: Top supportive drivers sorted descending by shap_value.
        top_bearish_drivers: Top opposing drivers sorted ascending by shap_value.
        is_economic_consistent: True if all features satisfy economic sign consistency.
    """

    symbol: str
    bar_time: datetime
    model_type: str
    base_value: float
    raw_margin: float
    contributions: tuple[SHAPDriverContribution, ...]
    top_bullish_drivers: tuple[SHAPDriverContribution, ...]
    top_bearish_drivers: tuple[SHAPDriverContribution, ...]
    is_economic_consistent: bool

    @model_validator(mode="after")
    def _validate_invariants(self) -> SHAPExplanationResult:
        if len(self.symbol.strip()) == 0:
            raise ValueError("symbol must not be empty")
        if len(self.model_type.strip()) == 0:
            raise ValueError("model_type must not be empty")
        if len(self.contributions) == 0:
            raise ValueError("contributions tuple must not be empty")

        # Local efficiency check: |sum(phi_j) + base_value - raw_margin| <= 1e-4
        _local_eff_tol: float = 1e-4
        phi_sum = sum(c.shap_value for c in self.contributions)
        discrepancy = abs(phi_sum + self.base_value - self.raw_margin)
        if discrepancy > _local_eff_tol:
            raise ValueError(
                f"TreeSHAP local efficiency violated: |sum(phi) ({phi_sum}) + base ({self.base_value}) "
                f"- raw ({self.raw_margin})| = {discrepancy:.2e} > {_local_eff_tol}"
            )

        # Validate bullish drivers ordering and signs
        for idx, bull in enumerate(self.top_bullish_drivers):
            if bull.shap_value <= 0.0:
                raise ValueError(
                    f"top_bullish_drivers contains non-positive attribution: {bull.shap_value} "
                    f"at index {idx}"
                )
            if idx > 0 and bull.shap_value > self.top_bullish_drivers[idx - 1].shap_value:
                raise ValueError(
                    f"top_bullish_drivers not sorted descending: index {idx} ({bull.shap_value}) "
                    f"> index {idx - 1} ({self.top_bullish_drivers[idx - 1].shap_value})"
                )

        # Validate bearish drivers ordering and signs
        for idx, bear in enumerate(self.top_bearish_drivers):
            if bear.shap_value >= 0.0:
                raise ValueError(
                    f"top_bearish_drivers contains non-negative attribution: {bear.shap_value} "
                    f"at index {idx}"
                )
            if idx > 0 and bear.shap_value < self.top_bearish_drivers[idx - 1].shap_value:
                raise ValueError(
                    f"top_bearish_drivers not sorted ascending: index {idx} ({bear.shap_value}) "
                    f"< index {idx - 1} ({self.top_bearish_drivers[idx - 1].shap_value})"
                )

        # Validate logical consistency
        expected_all_consistent = all(c.is_sign_consistent for c in self.contributions)
        if self.is_economic_consistent != expected_all_consistent:
            raise ValueError(
                f"is_economic_consistent ({self.is_economic_consistent}) contradicts "
                f"contributions evaluation ({expected_all_consistent})"
            )

        return self

    @property
    def total_feature_attribution(self) -> float:
        """Sum of all feature attributions sum(phi_j)."""
        return sum(c.shap_value for c in self.contributions)

    @property
    def feature_contribution_map(self) -> dict[str, float]:
        """Dictionary mapping feature name to its SHAP attribution value."""
        return {c.feature_name: c.shap_value for c in self.contributions}

    @property
    def num_bullish_drivers(self) -> int:
        """Number of bullish drivers extracted."""
        return len(self.top_bullish_drivers)

    @property
    def num_bearish_drivers(self) -> int:
        """Number of bearish drivers extracted."""
        return len(self.top_bearish_drivers)

    @property
    def inversion_count(self) -> int:
        """Number of feature attributions that violated economic sign consistency."""
        return sum(1 for c in self.contributions if not c.is_sign_consistent)


class UnifiedMLPredictionPacket(BaseModel):
    """Unified ML prediction record for a single 30m decision bar (Zero Defaults).

    Aggregates:
    - Directional barrier predictions (raw unconstrained probabilities)
    - Venn-Abers calibrated probabilities and epistemic uncertainty intervals
    - 6D Subspace Conformal Mahalanobis OOD detection and shrinkage factor
    - Top-5 Bullish and Top-3 Bearish native TreeSHAP attribution drivers
    - Economic sign consistency status
    - Raw feature snapshot for causal auditing

    Strict Clean Architecture & Invariant #9:
    - 100% pure Pydantic, zero default values, zero silent fallbacks.
    - Explicit fail-fast invariant checks via @model_validator.
    """

    symbol: str
    bar_time: datetime
    regime_label: str
    model_version: str
    p_long_raw: float
    p_short_raw: float
    p_long_calibrated: CalibratedProbability
    p_short_calibrated: CalibratedProbability
    ood_status: OODDetectionResult
    top_bullish_shap: tuple[SHAPDriverContribution, ...]
    top_bearish_shap: tuple[SHAPDriverContribution, ...]
    is_economic_consistent: bool
    features_snapshot: dict[str, float]

    @model_validator(mode="after")
    def _validate_invariants(self) -> UnifiedMLPredictionPacket:
        if len(self.symbol.strip()) == 0:
            raise ValueError("symbol must not be empty or whitespace")
        if self.bar_time.tzinfo is None:
            raise ValueError("bar_time must be timezone-aware UTC datetime")
        if len(self.regime_label.strip()) == 0:
            raise ValueError("regime_label must not be empty or whitespace")
        if len(self.model_version.strip()) == 0:
            raise ValueError("model_version must not be empty")
        if not (0.0 <= self.p_long_raw <= 1.0):
            raise ValueError(f"p_long_raw must be in [0, 1], received {self.p_long_raw}")
        if not (0.0 <= self.p_short_raw <= 1.0):
            raise ValueError(f"p_short_raw must be in [0, 1], received {self.p_short_raw}")
        if len(self.top_bullish_shap) > 5:
            raise ValueError(
                f"top_bullish_shap cannot exceed 5 drivers, received {len(self.top_bullish_shap)}"
            )
        if len(self.top_bearish_shap) > 3:
            raise ValueError(
                f"top_bearish_shap cannot exceed 3 drivers, received {len(self.top_bearish_shap)}"
            )
        for bull in self.top_bullish_shap:
            if bull.shap_value <= 0.0:
                raise ValueError(
                    f"Bullish SHAP driver '{bull.feature_name}' has non-positive attribution: {bull.shap_value}"
                )
        for bear in self.top_bearish_shap:
            if bear.shap_value >= 0.0:
                raise ValueError(
                    f"Bearish SHAP driver '{bear.feature_name}' has non-negative attribution: {bear.shap_value}"
                )
        if len(self.features_snapshot) == 0:
            raise ValueError("features_snapshot dictionary must not be empty")
        return self

    @property
    def features(self) -> dict[str, float]:
        """Alias for features_snapshot."""
        return self.features_snapshot

    @property
    def is_ood(self) -> bool:
        """Alias for OOD status flag."""
        return self.ood_status.is_ood

    @property
    def ood_shrinkage(self) -> float:
        """Continuous Kelly sizing multiplier gamma_OOD in [0, 1]."""
        return self.ood_status.gamma_ood

    @property
    def p_long_uncertainty(self) -> float:
        """Venn-Abers Long uncertainty interval width."""
        return self.p_long_calibrated.uncertainty_width

    @property
    def p_short_uncertainty(self) -> float:
        """Venn-Abers Short uncertainty interval width."""
        return self.p_short_calibrated.uncertainty_width

    @property
    def p_long_cal(self) -> float:
        """Calibrated probability of Long barrier."""
        return self.p_long_calibrated.p_calibrated

    @property
    def p_short_cal(self) -> float:
        """Calibrated probability of Short barrier."""
        return self.p_short_calibrated.p_calibrated

    @property
    def is_coherent(self) -> bool:
        """Indicates whether raw directional barrier probabilities are mutually coherent (sum <= 1.0)."""
        return (self.p_long_raw + self.p_short_raw) <= 1.0000001

    @property
    def coherence_penalty(self) -> float:
        """Magnitude of raw probability excess above 1.0 across directional models before regularization."""
        return max(0.0, (self.p_long_raw + self.p_short_raw) - 1.0)


class WalkForwardWindowSplit(BaseModel):
    """Temporal boundaries for a single walk-forward window (Zero Defaults).

    Adheres strictly to Invariant #9: zero default values, zero silent fallbacks.
    """

    window_id: int
    subtrain_start: datetime
    subtrain_end: datetime
    internal_embargo_start: datetime
    internal_embargo_end: datetime
    cal_start: datetime
    cal_end: datetime
    external_embargo_start: datetime
    external_embargo_end: datetime
    test_start: datetime
    test_end: datetime

    @model_validator(mode="after")
    def _validate_invariants(self) -> WalkForwardWindowSplit:
        if self.window_id < 1:
            raise ValueError(f"window_id must be >= 1, received {self.window_id}")

        datetimes = [
            self.subtrain_start,
            self.subtrain_end,
            self.internal_embargo_start,
            self.internal_embargo_end,
            self.cal_start,
            self.cal_end,
            self.external_embargo_start,
            self.external_embargo_end,
            self.test_start,
            self.test_end,
        ]
        for dt in datetimes:
            if dt.tzinfo is None:
                raise ValueError("All window split datetimes must be timezone-aware")

        if not (
            self.subtrain_start
            < self.subtrain_end
            <= self.internal_embargo_start
            <= self.internal_embargo_end
            <= self.cal_start
            < self.cal_end
            <= self.external_embargo_start
            <= self.external_embargo_end
            <= self.test_start
            < self.test_end
        ):
            raise ValueError(
                f"Invalid temporal ordering in window {self.window_id}: "
                f"subtrain: [{self.subtrain_start} -> {self.subtrain_end}], "
                f"internal embargo: [{self.internal_embargo_start} -> {self.internal_embargo_end}], "
                f"cal: [{self.cal_start} -> {self.cal_end}], "
                f"external embargo: [{self.external_embargo_start} -> {self.external_embargo_end}], "
                f"test: [{self.test_start} -> {self.test_end}]"
            )
        return self


class WindowEvaluationMetrics(BaseModel):
    """Evaluation metrics for a single walk-forward test block (Zero Defaults).

    Adheres strictly to Invariant #9: zero default values, zero silent fallbacks.
    """

    window_id: int
    symbol: str
    sample_count: int
    auc_long: float
    auc_short: float
    brier_long: float
    brier_short: float
    log_loss_long: float
    log_loss_short: float
    ece_long: float
    ece_short: float
    pct_ood: float
    pct_economic_consistent: float
    trade_count: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float
    cumulative_return: float

    @model_validator(mode="after")
    def _validate_invariants(self) -> WindowEvaluationMetrics:
        if self.window_id < 1:
            raise ValueError(f"window_id must be >= 1, received {self.window_id}")
        if len(self.symbol.strip()) == 0:
            raise ValueError("symbol must not be empty")
        if self.sample_count < 0:
            raise ValueError(f"sample_count must be non-negative, received {self.sample_count}")
        if self.trade_count < 0:
            raise ValueError(f"trade_count must be non-negative, received {self.trade_count}")
        if not (0.0 <= self.win_rate <= 1.0):
            raise ValueError(f"win_rate must be in [0, 1], received {self.win_rate}")
        if not (0.0 <= self.pct_ood <= 100.0):
            raise ValueError(f"pct_ood must be in [0, 100], received {self.pct_ood}")
        if not (0.0 <= self.pct_economic_consistent <= 100.0):
            raise ValueError(
                f"pct_economic_consistent must be in [0, 100], received {self.pct_economic_consistent}"
            )
        return self


class WalkForwardSummaryReport(BaseModel):
    """Comprehensive summary report across all walk-forward windows (Zero Defaults).

    Adheres strictly to Invariant #9: zero default values, zero silent fallbacks.
    """

    total_samples: int
    total_windows: int
    window_metrics: tuple[WindowEvaluationMetrics, ...]
    mean_auc_long: float
    mean_auc_short: float
    overall_sharpe: float
    overall_max_drawdown: float
    overall_profit_factor: float
    overall_win_rate: float
    overall_pct_ood: float
    overall_pct_economic_consistent: float
    model_version: str
    created_at: datetime

    @model_validator(mode="after")
    def _validate_invariants(self) -> WalkForwardSummaryReport:
        if self.total_samples < 0:
            raise ValueError(f"total_samples must be >= 0, received {self.total_samples}")
        if self.total_windows < 0:
            raise ValueError(f"total_windows must be >= 0, received {self.total_windows}")
        if len(self.model_version.strip()) == 0:
            raise ValueError("model_version must not be empty")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware UTC datetime")
        return self
