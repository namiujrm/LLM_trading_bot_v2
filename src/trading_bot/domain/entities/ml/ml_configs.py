"""Stage 4 Causal ML Explicit Configuration Schemas — Zero Defaults, Zero Hardcoding (Invariant #9).

Pure Python and Pydantic domain models for Stage 4 machine learning pipelines.
All fields are strictly required type annotations with zero default values.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, model_validator


class TripleBarrierConfig(BaseModel):
    """Configuration for Symmetrical 1m Triple-Barrier Labeling (Zero Defaults)."""

    pt_multiplier: float
    sl_multiplier: float
    horizon_bars_1m: int
    taker_fee_rate: float
    slippage_rate: float
    min_tp_to_cost_multiple: float
    dead_zone_weight: float


class SampleWeightConfig(BaseModel):
    """Configuration for López de Prado Sample Uniqueness Weighting (Zero Defaults)."""

    min_volatility_epsilon: float


class PurgedCVConfig(BaseModel):
    """Configuration for Purged K-Fold Cross-Validation with Dynamic Embargo (Zero Defaults)."""

    n_splits: int
    embargo_bars_30m: int
    max_label_holding_bars_30m: int

    @model_validator(mode="after")
    def _validate_invariants(self) -> PurgedCVConfig:
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, received {self.n_splits}")
        if self.embargo_bars_30m < 0:
            raise ValueError(
                f"embargo_bars_30m must be non-negative, received {self.embargo_bars_30m}"
            )
        if self.max_label_holding_bars_30m < 0:
            raise ValueError(
                f"max_label_holding_bars_30m must be non-negative, received {self.max_label_holding_bars_30m}"
            )
        return self

    @property
    def embargo_bars(self) -> int:
        """Alias for embargo_bars_30m."""
        return self.embargo_bars_30m

    @property
    def max_label_holding_bars(self) -> int:
        """Alias for max_label_holding_bars_30m."""
        return self.max_label_holding_bars_30m

    @classmethod
    def create_with_dynamic_embargo(
        cls,
        n_splits: int,
        feature_memory_lookback_bars: int,
        max_label_holding_bars: int,
    ) -> PurgedCVConfig:
        """Factory method computing h_embargo = max(holding, feature_lookback)."""
        embargo = max(max_label_holding_bars, feature_memory_lookback_bars)
        return cls(
            n_splits=n_splits,
            embargo_bars_30m=embargo,
            max_label_holding_bars_30m=max_label_holding_bars,
        )


class ConformalOODConfig(BaseModel):
    """Configuration for 6D Subspace Conformal Mahalanobis OOD Detector (Zero Defaults).

    Adheres strictly to Invariant #9: all parameters are explicit with zero defaults.
    """

    feature_names: tuple[str, ...]
    alpha_significance: float
    kappa_shrinkage: float
    max_condition_number: float
    min_eigenvalue_floor: float

    @model_validator(mode="after")
    def _validate_invariants(self) -> ConformalOODConfig:
        if not (0.0 < self.alpha_significance < 1.0):
            raise ValueError(
                f"alpha_significance must be in range (0, 1), received {self.alpha_significance}"
            )
        if self.kappa_shrinkage <= 0.0:
            raise ValueError(
                f"kappa_shrinkage must be strictly positive, received {self.kappa_shrinkage}"
            )
        if self.max_condition_number <= 1.0:
            raise ValueError(
                f"max_condition_number must be greater than 1.0, received {self.max_condition_number}"
            )
        if self.min_eigenvalue_floor <= 0.0:
            raise ValueError(
                f"min_eigenvalue_floor must be strictly positive, received {self.min_eigenvalue_floor}"
            )
        if len(self.feature_names) == 0:
            raise ValueError("feature_names tuple must not be empty")
        return self

    @property
    def decay_rate(self) -> float:
        """Alias for kappa_shrinkage."""
        return self.kappa_shrinkage

    @property
    def min_variance_clip(self) -> float:
        """Alias for min_eigenvalue_floor."""
        return self.min_eigenvalue_floor


class RidgePriorConfig(BaseModel):
    """Configuration for 5-Fold OOF Logistic Ridge Linear Prior (Zero Defaults)."""

    ridge_c: float
    ridge_max_iter: int
    oof_splits: int
    clip_eps: float
    embargo_bars_30m: int
    max_label_holding_bars_30m: int

    @model_validator(mode="after")
    def _validate_invariants(self) -> RidgePriorConfig:
        if self.ridge_c <= 0.0:
            raise ValueError(f"ridge_c must be strictly positive, received {self.ridge_c}")
        if self.ridge_max_iter < 1:
            raise ValueError(f"ridge_max_iter must be at least 1, received {self.ridge_max_iter}")
        if self.oof_splits < 2:
            raise ValueError(
                f"oof_splits must be at least 2 for cross-validation, received {self.oof_splits}"
            )
        if not (0.0 < self.clip_eps < 0.5):
            raise ValueError(f"clip_eps must be in range (0, 0.5), received {self.clip_eps}")
        if self.embargo_bars_30m < 0:
            raise ValueError(
                f"embargo_bars_30m must be non-negative, received {self.embargo_bars_30m}"
            )
        if self.max_label_holding_bars_30m < 0:
            raise ValueError(
                f"max_label_holding_bars_30m must be non-negative, received {self.max_label_holding_bars_30m}"
            )
        return self


class ConstrainedLGBMConfig(BaseModel):
    """Configuration for Shallow Monotonic LightGBM Ensemble (Zero Defaults)."""

    max_depth: int
    num_leaves: int
    min_child_samples: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    num_boost_round: int
    early_stopping_rounds: int
    coherence_penalty_gamma: float

    @model_validator(mode="after")
    def _validate_invariants(self) -> ConstrainedLGBMConfig:
        if self.max_depth < 1:
            raise ValueError(f"max_depth must be at least 1, received {self.max_depth}")
        if self.num_leaves < 2:
            raise ValueError(f"num_leaves must be at least 2, received {self.num_leaves}")
        if self.min_child_samples < 1:
            raise ValueError(
                f"min_child_samples must be at least 1, received {self.min_child_samples}"
            )
        if not (0.0 < self.learning_rate <= 1.0):
            raise ValueError(
                f"learning_rate must be in range (0, 1], received {self.learning_rate}"
            )
        if not (0.0 < self.subsample <= 1.0):
            raise ValueError(f"subsample must be in range (0, 1], received {self.subsample}")
        if not (0.0 < self.colsample_bytree <= 1.0):
            raise ValueError(
                f"colsample_bytree must be in range (0, 1], received {self.colsample_bytree}"
            )
        if self.reg_alpha < 0.0:
            raise ValueError(f"reg_alpha must be non-negative, received {self.reg_alpha}")
        if self.reg_lambda < 0.0:
            raise ValueError(f"reg_lambda must be non-negative, received {self.reg_lambda}")
        if self.num_boost_round < 1:
            raise ValueError(f"num_boost_round must be at least 1, received {self.num_boost_round}")
        if self.early_stopping_rounds < 1:
            raise ValueError(
                f"early_stopping_rounds must be at least 1, received {self.early_stopping_rounds}"
            )
        if self.coherence_penalty_gamma <= 0.0:
            raise ValueError(
                f"coherence_penalty_gamma must be strictly positive, received {self.coherence_penalty_gamma}"
            )
        return self


class CoherenceGateConfig(BaseModel):
    """Configuration for Dual-Binary Coherence Gate Domain Service (Zero Defaults).

    Controls the penalization and attenuation of conflicting directional probabilities
    when P_long + P_short > max_combined_probability (typically 1.0).

    Adheres strictly to Invariant #9: explicit types, zero defaults, fail-fast validation.
    """

    gamma_conflict: float
    max_combined_probability: float
    min_probability_floor: float

    @model_validator(mode="after")
    def _validate_invariants(self) -> CoherenceGateConfig:
        if self.gamma_conflict <= 0.0:
            raise ValueError(
                f"gamma_conflict must be strictly positive, received {self.gamma_conflict}"
            )
        if not (0.0 < self.max_combined_probability <= 1.0):
            raise ValueError(
                f"max_combined_probability must be in range (0, 1], received {self.max_combined_probability}"
            )
        if self.min_probability_floor < 0.0:
            raise ValueError(
                f"min_probability_floor must be non-negative, received {self.min_probability_floor}"
            )
        if self.min_probability_floor >= self.max_combined_probability:
            raise ValueError(
                f"min_probability_floor ({self.min_probability_floor}) must be less than "
                f"max_combined_probability ({self.max_combined_probability})"
            )
        return self


class FastVennAbersConfig(BaseModel):
    """Configuration for 500-Point Pre-Tabulated Venn-Abers Calibrator (Zero Defaults).

    Adheres strictly to Invariant #9: zero defaults, explicit types, fail-fast validation.

    Fields:
        grid_points: Number of uniform quantile evaluation grid points (typically 500).
        prob_clip_min: Minimum clipping floor for raw score input (typically 0.001).
        prob_clip_max: Maximum clipping ceiling for raw score input (typically 0.999).
        denom_eps: Epsilon floor to prevent denominator division-by-zero (1 - p0 + p1).
        uncertainty_threshold: Threshold above which a prediction is flagged as uncertain.
        calibration_floor: Minimum probability floor to avoid zero probabilities in Kelly sizing.
    """

    grid_points: int
    prob_clip_min: float
    prob_clip_max: float
    denom_eps: float
    uncertainty_threshold: float
    calibration_floor: float

    @model_validator(mode="after")
    def _validate_invariants(self) -> FastVennAbersConfig:
        if self.grid_points < 100:
            raise ValueError(
                f"grid_points must be at least 100 for adequate resolution, received {self.grid_points}"
            )
        if not (0.0 < self.prob_clip_min < self.prob_clip_max < 1.0):
            raise ValueError(
                f"Clipping bounds must satisfy 0.0 < prob_clip_min < prob_clip_max < 1.0, "
                f"received min={self.prob_clip_min}, max={self.prob_clip_max}"
            )
        if self.denom_eps <= 0.0:
            raise ValueError(f"denom_eps must be strictly positive, received {self.denom_eps}")
        if not (0.0 < self.uncertainty_threshold < 1.0):
            raise ValueError(
                f"uncertainty_threshold must be in open interval (0, 1), received {self.uncertainty_threshold}"
            )
        if not (0.0 <= self.calibration_floor < self.prob_clip_max):
            raise ValueError(
                f"calibration_floor must satisfy 0.0 <= calibration_floor < prob_clip_max ({self.prob_clip_max}), "
                f"received {self.calibration_floor}"
            )
        return self

    @property
    def n_grid_points(self) -> int:
        """Alias for grid_points for backward compatibility."""
        return self.grid_points


class ECEMonitorConfig(BaseModel):
    """Configuration for Rolling 7-Day ECE Drift Monitor (Zero Defaults).

    Adheres strictly to Invariant #9: zero defaults, explicit types, fail-fast validation.

    Fields:
        window_bars_30m: Rolling window capacity in 30m bars (typically 336 = 7 days * 48 bars/day).
        n_bins: Number of equal-width probability bins for ECE estimation (typically 10).
        drift_warning_threshold: Expected Calibration Error threshold triggering drift warning (typically 0.08).
        min_samples_to_evaluate: Minimum samples required in window before drift alerts activate (typically 30).
    """

    window_bars_30m: int
    n_bins: int
    drift_warning_threshold: float
    min_samples_to_evaluate: int

    @model_validator(mode="after")
    def _validate_invariants(self) -> ECEMonitorConfig:
        if self.window_bars_30m < 10:
            raise ValueError(
                f"window_bars_30m must be at least 10, received {self.window_bars_30m}"
            )
        if self.n_bins < 2:
            raise ValueError(f"n_bins must be at least 2, received {self.n_bins}")
        if not (0.0 < self.drift_warning_threshold < 1.0):
            raise ValueError(
                f"drift_warning_threshold must be in open interval (0, 1), received {self.drift_warning_threshold}"
            )
        if self.min_samples_to_evaluate < self.n_bins:
            raise ValueError(
                f"min_samples_to_evaluate ({self.min_samples_to_evaluate}) must be >= n_bins ({self.n_bins})"
            )
        return self

    @property
    def num_bins(self) -> int:
        """Alias for n_bins for backward compatibility."""
        return self.n_bins

    @property
    def drift_threshold(self) -> float:
        """Alias for drift_warning_threshold for backward compatibility."""
        return self.drift_warning_threshold


class TreeSHAPConfig(BaseModel):
    """Configuration for Native LightGBM C++ TreeSHAP Engine & Economic Sign Verifier (Zero Defaults).

    Adheres strictly to Invariant #9: zero defaults, explicit types, fail-fast validation.

    Fields:
        top_k_bullish: Maximum number of top positive SHAP attributions to extract (typically 5).
        top_k_bearish: Maximum number of top negative SHAP attributions to extract (typically 3).
        local_efficiency_tol: Floating point tolerance for local efficiency check |sum(phi) + base - raw| (typically 1e-4).
        sign_tolerance: Slack tolerance for numerical zero when verifying economic sign consistency (typically 1e-4).
    """

    top_k_bullish: int
    top_k_bearish: int
    local_efficiency_tol: float
    sign_tolerance: float

    @model_validator(mode="after")
    def _validate_invariants(self) -> TreeSHAPConfig:
        if self.top_k_bullish < 1:
            raise ValueError(f"top_k_bullish must be at least 1, received {self.top_k_bullish}")
        if self.top_k_bearish < 1:
            raise ValueError(f"top_k_bearish must be at least 1, received {self.top_k_bearish}")
        if self.local_efficiency_tol <= 0.0:
            raise ValueError(
                f"local_efficiency_tol must be strictly positive, received {self.local_efficiency_tol}"
            )
        if self.sign_tolerance < 0.0:
            raise ValueError(f"sign_tolerance must be non-negative, received {self.sign_tolerance}")
        return self

    @property
    def feature_deadband_tol(self) -> float:
        """Alias for sign_tolerance applied to feature deviation deadband."""
        return self.sign_tolerance

    @property
    def attribution_deadband_tol(self) -> float:
        """Alias for sign_tolerance applied to attribution magnitude deadband."""
        return self.sign_tolerance


SHAPExplainerConfig = TreeSHAPConfig


class LGBMMonotonicConstraints(BaseModel):
    """Domain specification of feature monotonic constraints and economic baselines (Zero Defaults).

    Adheres strictly to Invariant #9: zero defaults, explicit types, fail-fast validation.

    Fields:
        constraints: Mapping of feature_name to monotonic direction in {-1, 0, 1}.
        neutral_baselines: Mapping of feature_name to neutral economic baseline value.
    """

    constraints: dict[str, int]
    neutral_baselines: dict[str, float]

    @model_validator(mode="after")
    def _validate_invariants(self) -> LGBMMonotonicConstraints:
        if len(self.constraints) == 0:
            raise ValueError("constraints mapping must not be empty")
        for feature_name, direction in self.constraints.items():
            if direction not in (-1, 0, 1):
                raise ValueError(
                    f"Direction for '{feature_name}' must be -1, 0, or 1, received {direction}"
                )
        missing_baselines = set(self.constraints) - set(self.neutral_baselines)
        if len(missing_baselines) > 0:
            raise ValueError(
                f"neutral_baselines missing constraints for: {sorted(missing_baselines)}"
            )
        return self

    @classmethod
    def create_canonical_long(cls) -> LGBMMonotonicConstraints:
        """Factory for Long Barrier model's 11 canonical economic constraints and baselines."""
        constraints = {
            "hurst_exponent_100": 1,
            "shannon_entropy_50": -1,
            "fracdiff_momentum_z": 1,
            "vpin_toxicity_pct": -1,
            "taker_cvd_z_24": 1,
            "ofi_imbalance_z": 1,
            "kyle_lambda_log": -1,
            "oi_directional_score": 1,
            "funding_crowded_pos_z": -1,
            "funding_crowded_neg_z": 1,
            "btc_lead_catchup_gap": 1,
        }
        neutral_baselines = {
            "hurst_exponent_100": 0.50,
            "shannon_entropy_50": 0.50,
            "fracdiff_momentum_z": 0.00,
            "vpin_toxicity_pct": 0.50,
            "taker_cvd_z_24": 0.00,
            "ofi_imbalance_z": 0.00,
            "kyle_lambda_log": 0.00,
            "oi_directional_score": 0.00,
            "funding_crowded_pos_z": 0.00,
            "funding_crowded_neg_z": 0.00,
            "btc_lead_catchup_gap": 0.00,
        }
        return cls(constraints=constraints, neutral_baselines=neutral_baselines)

    @classmethod
    def create_canonical_short(cls) -> LGBMMonotonicConstraints:
        """Factory for Short Barrier model's 11 canonical economic constraints and baselines."""
        constraints = {
            "hurst_exponent_100": 1,
            "shannon_entropy_50": -1,
            "fracdiff_momentum_z": -1,
            "vpin_toxicity_pct": -1,
            "taker_cvd_z_24": -1,
            "ofi_imbalance_z": -1,
            "kyle_lambda_log": -1,
            "oi_directional_score": -1,
            "funding_crowded_pos_z": 1,
            "funding_crowded_neg_z": -1,
            "btc_lead_catchup_gap": -1,
        }
        neutral_baselines = {
            "hurst_exponent_100": 0.50,
            "shannon_entropy_50": 0.50,
            "fracdiff_momentum_z": 0.00,
            "vpin_toxicity_pct": 0.50,
            "taker_cvd_z_24": 0.00,
            "ofi_imbalance_z": 0.00,
            "kyle_lambda_log": 0.00,
            "oi_directional_score": 0.00,
            "funding_crowded_pos_z": 0.00,
            "funding_crowded_neg_z": 0.00,
            "btc_lead_catchup_gap": 0.00,
        }
        return cls(constraints=constraints, neutral_baselines=neutral_baselines)


class WalkForwardConfig(BaseModel):
    """Configuration for End-to-End Walk-Forward Evaluation Pipeline (Zero Defaults).

    Strict Clean Architecture & Invariant #9:
    - Zero default parameters, zero default class fields.
    - Explicit fail-fast invariant checks via @model_validator.
    """

    start_date: datetime
    end_date: datetime
    train_duration_months: int
    test_duration_months: int
    cal_ratio: float
    embargo_bars_30m: int
    decision_threshold: float
    symbols: tuple[str, ...]
    canonical_features: tuple[str, ...]
    subspace_features: tuple[str, ...]
    triple_barrier_config: TripleBarrierConfig
    sample_weight_config: SampleWeightConfig
    purged_cv_config: PurgedCVConfig
    conformal_ood_config: ConformalOODConfig
    ridge_prior_config: RidgePriorConfig
    constrained_lgbm_config: ConstrainedLGBMConfig
    coherence_gate_config: CoherenceGateConfig
    fast_venn_abers_config: FastVennAbersConfig
    ece_monitor_config: ECEMonitorConfig
    treeshap_config: TreeSHAPConfig
    random_state: int

    @model_validator(mode="after")
    def _validate_invariants(self) -> WalkForwardConfig:
        if self.start_date.tzinfo is None or self.end_date.tzinfo is None:
            raise ValueError("start_date and end_date must be timezone-aware UTC datetime")
        if self.start_date >= self.end_date:
            raise ValueError(
                f"start_date ({self.start_date}) must be strictly earlier than end_date ({self.end_date})"
            )
        if self.train_duration_months <= 0:
            raise ValueError(
                f"train_duration_months must be positive, received {self.train_duration_months}"
            )
        if self.test_duration_months <= 0:
            raise ValueError(
                f"test_duration_months must be positive, received {self.test_duration_months}"
            )
        if not (0.0 < self.cal_ratio < 1.0):
            raise ValueError(
                f"cal_ratio must be strictly between 0 and 1, received {self.cal_ratio}"
            )
        if self.embargo_bars_30m < 100:
            raise ValueError(
                f"embargo_bars_30m must be >= 100 to eliminate autoregressive feature memory leakage, received {self.embargo_bars_30m}"
            )
        if not (0.0 <= self.decision_threshold <= 1.0):
            raise ValueError(
                f"decision_threshold must be in [0, 1], received {self.decision_threshold}"
            )
        if len(self.symbols) == 0:
            raise ValueError("symbols tuple must not be empty")
        for sym in self.symbols:
            if len(sym.strip()) == 0:
                raise ValueError("symbol in symbols must not be empty")
        if len(self.canonical_features) == 0:
            raise ValueError("canonical_features tuple must not be empty")
        if len(self.subspace_features) == 0:
            raise ValueError("subspace_features tuple must not be empty")
        return self
