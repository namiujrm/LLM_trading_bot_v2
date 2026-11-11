# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Monotonic & Interaction-Constrained LightGBM Causal Barrier Model Adapter (Zero Defaults).

Strict Clean Architecture: Concrete adapter in adapters/ml_engine layer implementing CausalBarrierModelPort.
Enforces Invariant #9: zero default parameters, zero class field defaults,
zero silent fallbacks, fail-fast input validation.
"""

from __future__ import annotations

from collections.abc import Sequence

import lightgbm as lgb
import numpy as np
import scipy.special

from trading_bot.adapters.ml_engine.oof_ridge_prior import OOFRidgePrior
from trading_bot.domain.entities.ml.ml_configs import (
    CoherenceGateConfig,
    ConstrainedLGBMConfig,
    RidgePriorConfig,
)
from trading_bot.domain.entities.ml.ml_predictions import BarrierPredictionResult
from trading_bot.domain.exceptions import InsufficientDataException
from trading_bot.domain.services.ml.coherence_gate import CoherenceGate
from trading_bot.ports.ml_model_port import CausalBarrierModelPort

CANONICAL_FEATURES: tuple[str, ...] = (
    "hurst_exponent_100",  # 0
    "shannon_entropy_50",  # 1
    "fracdiff_momentum_z",  # 2
    "vpin_toxicity_pct",  # 3
    "taker_cvd_z_24",  # 4
    "ofi_imbalance_z",  # 5
    "kyle_lambda_log",  # 6
    "oi_directional_score",  # 7
    "funding_crowded_pos_z",  # 8
    "funding_crowded_neg_z",  # 9
    "btc_lead_catchup_gap",  # 10
)

CANONICAL_MONOTONE_LONG: dict[str, int] = {
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

CANONICAL_MONOTONE_SHORT: dict[str, int] = {
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

CANONICAL_INTERACTION_CLUSTERS: tuple[tuple[str, ...], ...] = (
    ("hurst_exponent_100", "shannon_entropy_50", "fracdiff_momentum_z"),
    ("vpin_toxicity_pct", "taker_cvd_z_24", "ofi_imbalance_z", "kyle_lambda_log"),
    ("oi_directional_score", "funding_crowded_pos_z", "funding_crowded_neg_z"),
    ("btc_lead_catchup_gap",),
)


class ConstrainedLGBMModel(CausalBarrierModelPort):
    """Dual-binary LightGBM Causal Barrier Model with 5-Fold OOF Ridge Prior."""

    def __init__(
        self,
        config: ConstrainedLGBMConfig,
        ridge_config: RidgePriorConfig,
        coherence_config: CoherenceGateConfig,
        feature_names: tuple[str, ...],
        random_state: int,
    ) -> None:
        """Initialize dual-binary model with explicit configuration schemas.

        Args:
            config: ConstrainedLGBMConfig (tree depth, leaves, regularization, rounds).
            ridge_config: RidgePriorConfig (penalty, iterations, splits, clip epsilon).
            coherence_config: CoherenceGateConfig (gamma penalty, max combined probability).
            feature_names: Explicit tuple of feature names to map monotonic/interaction constraints.
            random_state: Explicit integer seed for deterministic fitting.

        Raises:
            ValueError: If canonical features are missing or unknown features are provided.
        """
        missing_features = set(CANONICAL_FEATURES) - set(feature_names)
        if len(missing_features) > 0:
            raise ValueError(
                f"feature_names is missing required canonical features: {sorted(missing_features)}"
            )

        unknown_features = [f for f in feature_names if f not in CANONICAL_FEATURES]
        if len(unknown_features) > 0:
            raise ValueError(f"Unknown feature names provided: {unknown_features}")

        name_to_idx: dict[str, int] = {name: idx for idx, name in enumerate(feature_names)}

        self._config: ConstrainedLGBMConfig = config
        self._ridge_config: RidgePriorConfig = ridge_config
        self._coherence_config: CoherenceGateConfig = coherence_config
        self._feature_names: tuple[str, ...] = feature_names
        self._random_state: int = random_state

        self._monotone_long: list[int] = [CANONICAL_MONOTONE_LONG[f] for f in feature_names]
        self._monotone_short: list[int] = [CANONICAL_MONOTONE_SHORT[f] for f in feature_names]
        self._interaction_constraints: list[list[int]] = [
            [name_to_idx[f] for f in cluster if f in name_to_idx]
            for cluster in CANONICAL_INTERACTION_CLUSTERS
        ]

        self._ridge_long: OOFRidgePrior | None = None
        self._ridge_short: OOFRidgePrior | None = None
        self._booster_long: lgb.Booster | None = None
        self._booster_short: lgb.Booster | None = None
        self._coherence_gate: CoherenceGate = CoherenceGate(config=coherence_config)

    @property
    def is_fitted(self) -> bool:
        """Check whether dual boosters and ridge priors have been fitted."""
        return (
            self._booster_long is not None
            and self._booster_short is not None
            and self._ridge_long is not None
            and self._ridge_short is not None
        )

    @property
    def long_booster(self) -> lgb.Booster:
        """Return the fitted LightGBM booster for the Long barrier."""
        if self._booster_long is None:
            raise RuntimeError("ConstrainedLGBMModel has not been fitted.")
        return self._booster_long

    @property
    def short_booster(self) -> lgb.Booster:
        """Return the fitted LightGBM booster for the Short barrier."""
        if self._booster_short is None:
            raise RuntimeError("ConstrainedLGBMModel has not been fitted.")
        return self._booster_short

    @property
    def ridge_long(self) -> OOFRidgePrior:
        """Return the fitted OOFRidgePrior for the Long barrier."""
        if self._ridge_long is None:
            raise RuntimeError("ConstrainedLGBMModel has not been fitted.")
        return self._ridge_long

    @property
    def ridge_short(self) -> OOFRidgePrior:
        """Return the fitted OOFRidgePrior for the Short barrier."""
        if self._ridge_short is None:
            raise RuntimeError("ConstrainedLGBMModel has not been fitted.")
        return self._ridge_short

    @property
    def coherence_gate(self) -> CoherenceGate:
        """Return the injected CoherenceGate domain service."""
        return self._coherence_gate

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return the tuple of canonical feature names."""
        return self._feature_names

    @property
    def config(self) -> ConstrainedLGBMConfig:
        """Return the injected ConstrainedLGBMConfig."""
        return self._config

    @property
    def random_state(self) -> int:
        """Return the random seed."""
        return self._random_state

    @classmethod
    def from_trained_artifacts(
        cls,
        config: ConstrainedLGBMConfig,
        ridge_config: RidgePriorConfig,
        coherence_config: CoherenceGateConfig,
        feature_names: tuple[str, ...],
        random_state: int,
        booster_long: lgb.Booster,
        booster_short: lgb.Booster,
        ridge_long: OOFRidgePrior,
        ridge_short: OOFRidgePrior,
    ) -> ConstrainedLGBMModel:
        """Factory method to construct model from previously persisted boosters and ridge priors."""
        instance = cls(
            config=config,
            ridge_config=ridge_config,
            coherence_config=coherence_config,
            feature_names=feature_names,
            random_state=random_state,
        )
        instance._booster_long = booster_long
        instance._booster_short = booster_short
        instance._ridge_long = ridge_long
        instance._ridge_short = ridge_short
        return instance

    def fit(
        self,
        train_features: Sequence[Sequence[float]],
        y_long: Sequence[int],
        y_short: Sequence[int],
        sample_weights: Sequence[float],
        sample_intervals: Sequence[tuple[int, int]] | None,
    ) -> None:
        """Fit dual-binary causal models using 5-Fold OOF Ridge init_score stacking.

        Args:
            train_features: 2D sequence of shape (N, D).
            y_long: Binary target sequence of length N for Long barrier (1 = TP, 0 = otherwise).
            y_short: Binary target sequence of length N for Short barrier (1 = TP, 0 = otherwise).
            sample_weights: Sequence of length N containing López de Prado uniqueness weights.
            sample_intervals: Optional sequence of (t_start, t_end) intervals for Purged CV.

        Raises:
            ValueError: If dimensions mismatch, non-finite values present, or labels not binary.
            InsufficientDataException: If sample count is insufficient.
        """
        x_arr = np.asarray(train_features, dtype=np.float64)
        y_l = np.asarray(y_long, dtype=np.int64)
        y_s = np.asarray(y_short, dtype=np.int64)
        w_arr = np.asarray(sample_weights, dtype=np.float64)

        if x_arr.ndim != 2:
            raise ValueError(f"train_features must be 2D array, received ndim={x_arr.ndim}")
        if y_l.ndim != 1 or y_s.ndim != 1:
            raise ValueError("y_long and y_short must be 1D sequences")
        if w_arr.ndim != 1:
            raise ValueError("sample_weights must be 1D sequence")

        n_samples, n_feats = x_arr.shape
        if len(y_l) != n_samples or len(y_s) != n_samples or len(w_arr) != n_samples:
            raise ValueError(
                f"Dimension mismatch: features has {n_samples} rows, "
                f"y_long has {len(y_l)}, y_short has {len(y_s)}, sample_weights has {len(w_arr)}"
            )
        if n_feats != len(self._feature_names):
            raise ValueError(
                f"Feature count mismatch: expected {len(self._feature_names)}, received {n_feats}"
            )

        if not np.all(np.isfinite(x_arr)):
            raise ValueError("train_features contains NaN or non-finite values")
        if not np.all(np.isfinite(w_arr)):
            raise ValueError("sample_weights contains NaN or non-finite values")
        if np.any(w_arr <= 0.0):
            raise ValueError("sample_weights must contain strictly positive values")

        if not set(np.unique(y_l)).issubset({0, 1}):
            raise ValueError(f"y_long must be binary in {{0, 1}}, received {np.unique(y_l)}")
        if not set(np.unique(y_s)).issubset({0, 1}):
            raise ValueError(f"y_short must be binary in {{0, 1}}, received {np.unique(y_s)}")

        min_samples = max(20, self._config.min_child_samples * 2)
        if n_samples < min_samples:
            raise InsufficientDataException(
                f"Sample count ({n_samples}) is less than required minimum ({min_samples})."
            )

        # 1. Fit 5-Fold OOF Ridge Priors
        ridge_long = OOFRidgePrior(
            config=self._ridge_config,
            random_state=self._random_state,
        )
        oof_logits_long = ridge_long.fit(
            features=train_features,
            labels=y_long,
            sample_weights=sample_weights,
            sample_intervals=sample_intervals,
        )

        ridge_short = OOFRidgePrior(
            config=self._ridge_config,
            random_state=self._random_state + 1,
        )
        oof_logits_short = ridge_short.fit(
            features=train_features,
            labels=y_short,
            sample_weights=sample_weights,
            sample_intervals=sample_intervals,
        )

        # 2. Build LightGBM Datasets with init_score
        ds_long = lgb.Dataset(
            data=x_arr,
            label=y_l,
            weight=w_arr,
            init_score=oof_logits_long,
            feature_name=list(self._feature_names),
            free_raw_data=True,
        )
        ds_short = lgb.Dataset(
            data=x_arr,
            label=y_s,
            weight=w_arr,
            init_score=oof_logits_short,
            feature_name=list(self._feature_names),
            free_raw_data=True,
        )

        # 3. Parameters configuration
        params_base: dict[str, object] = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "max_depth": self._config.max_depth,
            "num_leaves": self._config.num_leaves,
            "min_child_samples": self._config.min_child_samples,
            "learning_rate": self._config.learning_rate,
            "subsample": self._config.subsample,
            "colsample_bytree": self._config.colsample_bytree,
            "reg_alpha": self._config.reg_alpha,
            "reg_lambda": self._config.reg_lambda,
            "interaction_constraints": self._interaction_constraints,
            "num_threads": 1,
            "verbosity": -1,
        }

        params_long = dict(params_base)
        params_long["monotone_constraints"] = self._monotone_long
        params_long["seed"] = self._random_state

        params_short = dict(params_base)
        params_short["monotone_constraints"] = self._monotone_short
        params_short["seed"] = self._random_state + 1

        # 4. Train Boosters
        booster_long = lgb.train(
            params=params_long,
            train_set=ds_long,
            num_boost_round=self._config.num_boost_round,
        )
        booster_short = lgb.train(
            params=params_short,
            train_set=ds_short,
            num_boost_round=self._config.num_boost_round,
        )

        self._ridge_long = ridge_long
        self._ridge_short = ridge_short
        self._booster_long = booster_long
        self._booster_short = booster_short

    def predict(
        self,
        sample_features: Sequence[float],
    ) -> BarrierPredictionResult:
        """Predict coherent directional barrier probabilities for a single feature vector.

        Args:
            sample_features: 1D sequence of length D.

        Returns:
            BarrierPredictionResult containing raw and coherent (p_long, p_short) probabilities.
        """
        results = self.predict_batch([sample_features])
        return results[0]

    def predict_batch(
        self,
        batch_features: Sequence[Sequence[float]],
    ) -> Sequence[BarrierPredictionResult]:
        """Predict coherent directional barrier probabilities for a batch of feature vectors.

        Args:
            batch_features: 2D sequence of shape (M, D).

        Returns:
            Sequence of BarrierPredictionResult objects.

        Raises:
            RuntimeError: If model is not fitted.
            ValueError: If input dimensions mismatch or non-finite values present.
        """
        if (
            self._booster_long is None
            or self._booster_short is None
            or self._ridge_long is None
            or self._ridge_short is None
        ):
            raise RuntimeError("ConstrainedLGBMModel has not been fitted.")

        x_batch = np.asarray(batch_features, dtype=np.float64)
        if x_batch.ndim != 2:
            raise ValueError(f"batch_features must be 2D array, received ndim={x_batch.ndim}")
        if x_batch.shape[1] != len(self._feature_names):
            raise ValueError(
                f"Feature dimension mismatch: expected {len(self._feature_names)}, received {x_batch.shape[1]}"
            )
        if not np.all(np.isfinite(x_batch)):
            raise ValueError("batch_features contains NaN or non-finite values")

        # 1. Ridge prior logit margins
        z_prior_long = self._ridge_long.predict_logits(x_batch)
        z_prior_short = self._ridge_short.predict_logits(x_batch)

        # 2. LightGBM tree additive margins (raw_score=True outputs sum of tree leaves)
        delta_z_long = np.asarray(
            self._booster_long.predict(x_batch, raw_score=True),
            dtype=np.float64,
        )
        delta_z_short = np.asarray(
            self._booster_short.predict(x_batch, raw_score=True),
            dtype=np.float64,
        )

        # 3. Additive margin combination: z_total = z_Ridge + delta_z_LGBM
        z_total_long = z_prior_long + delta_z_long
        z_total_short = z_prior_short + delta_z_short

        # 4. Numerically stable sigmoid transformation
        p_raw_long = scipy.special.expit(z_total_long)
        p_raw_short = scipy.special.expit(z_total_short)

        # 5. Dual-binary Coherence Gate regularization
        pairs: list[tuple[float, float]] = [
            (float(p_l), float(p_s)) for p_l, p_s in zip(p_raw_long, p_raw_short, strict=True)
        ]
        return self._coherence_gate.enforce_batch(pairs)
