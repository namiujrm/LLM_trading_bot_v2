# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Native LightGBM C++ TreeSHAP Explainer Adapter (Zero Defaults, Invariant #9).

Strict Clean Architecture:
- Concrete adapter in src/trading_bot/adapters/ml_engine/
- Implements TreeSHAPExplainerPort defined in src/trading_bot/ports/ml_model_port.py
- Calls LightGBM native C++ booster.predict(pred_contrib=True) (< 0.8ms SLA)
- Delegates economic sign verification to pure domain service EconomicSignVerifier
- Adheres strictly to Invariant #9: zero default parameters, zero default fields, zero fallbacks.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import lightgbm as lgb
import numpy as np

from trading_bot.domain.entities.ml.ml_configs import TreeSHAPConfig
from trading_bot.domain.entities.ml.ml_predictions import (
    SHAPExplanationResult,
)
from trading_bot.domain.services.ml.economic_sign_verifier import EconomicSignVerifier
from trading_bot.ports.ml_model_port import TreeSHAPExplainerPort


class NativeTreeSHAPExplainer(TreeSHAPExplainerPort):
    """High-throughput Native LightGBM C++ TreeSHAP Engine with Economic Sign Verification."""

    def __init__(
        self,
        config: TreeSHAPConfig,
        sign_verifier_long: EconomicSignVerifier,
        sign_verifier_short: EconomicSignVerifier,
    ) -> None:
        """Initialize adapter with explicit configuration and domain verifier instances.

        Args:
            config: Explicit TreeSHAPConfig (top_k_bullish, top_k_bearish, local_efficiency_tol, etc.).
            sign_verifier_long: Pure domain EconomicSignVerifier configured for Long barrier model.
            sign_verifier_short: Pure domain EconomicSignVerifier configured for Short barrier model.
        """
        self._config: TreeSHAPConfig = config
        self._sign_verifier_long: EconomicSignVerifier = sign_verifier_long
        self._sign_verifier_short: EconomicSignVerifier = sign_verifier_short
        self._booster_meta_cache: dict[int, tuple[int, tuple[str, ...]]] = {}

    @property
    def config(self) -> TreeSHAPConfig:
        """Return the injected TreeSHAPConfig schema."""
        return self._config

    @property
    def verifier_long(self) -> EconomicSignVerifier:
        """Return the injected Long barrier economic sign verifier."""
        return self._sign_verifier_long

    @property
    def verifier_short(self) -> EconomicSignVerifier:
        """Return the injected Short barrier economic sign verifier."""
        return self._sign_verifier_short

    def _get_booster_meta(self, booster: lgb.Booster) -> tuple[int, tuple[str, ...]]:
        """Retrieve and cache booster feature count and canonical feature names."""
        b_id = id(booster)
        if b_id in self._booster_meta_cache:
            return self._booster_meta_cache[b_id]

        expected_num_features = booster.num_feature()
        raw_names = booster.feature_name()
        if raw_names:
            booster_feature_names: tuple[str, ...] = tuple(raw_names)
        else:
            booster_feature_names = ()

        meta = (expected_num_features, booster_feature_names)
        self._booster_meta_cache[b_id] = meta
        return meta

    def _extract_booster(self, model: Any, model_type: str) -> lgb.Booster:
        """Extract underlying fitted lightgbm.Booster from model or return booster directly."""
        if isinstance(model, lgb.Booster):
            return model

        if model_type == "long_barrier" and hasattr(model, "long_booster"):
            booster = model.long_booster
            if isinstance(booster, lgb.Booster):
                return booster
        elif model_type == "short_barrier" and hasattr(model, "short_booster"):
            booster = model.short_booster
            if isinstance(booster, lgb.Booster):
                return booster

        raise TypeError(f"Cannot extract {model_type} booster from object of type {type(model)}")

    def explain(
        self,
        feature_vector: Sequence[float],
        feature_names: Sequence[str],
        model: Any,
        symbol: str,
        bar_time: datetime,
        model_type: str,
    ) -> SHAPExplanationResult:
        """Explain a single decision sample using native C++ TreeSHAP (< 0.8ms).

        Args:
            feature_vector: 1D sequence of length M containing feature values.
            feature_names: Sequence of length M containing canonical feature names.
            model: Underlying fitted lightgbm.Booster or model instance with long_booster/short_booster.
            symbol: Trading pair symbol (e.g. "BTCUSDT").
            bar_time: UTC timestamp of the decision bar.
            model_type: Directional barrier model identifier ("long_barrier" or "short_barrier").

        Returns:
            SHAPExplanationResult containing feature contributions, Top-K drivers, and sign check.

        Raises:
            ValueError: If feature dimensions mismatch, values are non-finite, or model_type unknown.
            TypeError: If booster cannot be extracted from model.
        """
        if len(feature_vector) == 0:
            raise ValueError("feature_vector cannot be empty")
        if len(feature_names) == 0:
            raise ValueError("feature_names cannot be empty")
        if len(feature_vector) != len(feature_names):
            raise ValueError(
                f"Feature dimension mismatch: expected {len(feature_names)} features, "
                f"received {len(feature_vector)}"
            )
        if model_type not in ("long_barrier", "short_barrier"):
            raise ValueError(
                f"Unknown model_type '{model_type}'; must be 'long_barrier' or 'short_barrier'"
            )

        booster = self._extract_booster(model=model, model_type=model_type)

        expected_num_features, booster_feature_names = self._get_booster_meta(booster)
        if len(feature_vector) != expected_num_features:
            raise ValueError(
                f"Feature dimension mismatch: booster expected {expected_num_features} features, "
                f"received {len(feature_vector)}"
            )

        if booster_feature_names and tuple(feature_names) != booster_feature_names:
            raise ValueError(
                f"Feature names mismatch: booster expected {list(booster_feature_names)}, "
                f"received {list(feature_names)}"
            )

        verifier = (
            self._sign_verifier_long if model_type == "long_barrier" else self._sign_verifier_short
        )

        x_sample = np.asarray([feature_vector], dtype=np.float64)
        if not np.all(np.isfinite(x_sample)):
            raise ValueError("feature_vector contains NaN or non-finite values")

        # Native C++ TreeSHAP Extraction & Model Margin Prediction
        # Output shapes: pred_contrib -> (1, M + 1), raw_score -> (1,)
        raw_pred = booster.predict(x_sample, pred_contrib=True)
        shap_matrix = np.asarray(raw_pred, dtype=np.float64)
        phi_vals = shap_matrix[0, :-1]
        base_value = float(shap_matrix[0, -1])

        raw_score_arr = booster.predict(x_sample, raw_score=True)
        raw_margin = float(np.asarray(raw_score_arr, dtype=np.float64)[0])

        # Verify Local Efficiency Axiom: sum(phi_j) + base_value == f(x)
        efficiency_diff = abs(float(np.sum(phi_vals) + base_value) - raw_margin)
        if efficiency_diff > self._config.local_efficiency_tol:
            raise ValueError(
                f"Local efficiency violated: diff={efficiency_diff:.2e} > "
                f"tol={self._config.local_efficiency_tol}"
            )

        # Economic Sign Verification & Driver Ranking via Pure Domain Service
        contributions, top_bullish, top_bearish, is_consistent = verifier.verify_and_rank(
            feature_names=feature_names,
            feature_values=feature_vector,
            shap_values=phi_vals.tolist(),
        )

        return SHAPExplanationResult(
            symbol=symbol,
            bar_time=bar_time,
            model_type=model_type,
            base_value=base_value,
            raw_margin=raw_margin,
            contributions=contributions,
            top_bullish_drivers=top_bullish,
            top_bearish_drivers=top_bearish,
            is_economic_consistent=is_consistent,
        )

    def explain_batch(
        self,
        batch_features: Sequence[Sequence[float]],
        feature_names: Sequence[str],
        model: Any,
        symbol: str,
        bar_times: Sequence[datetime],
        model_type: str,
    ) -> tuple[SHAPExplanationResult, ...]:
        """Explain a batch of decision samples using vectorized C++ TreeSHAP.

        Args:
            batch_features: 2D sequence of shape (N, M).
            feature_names: Sequence of length M containing canonical feature names.
            model: Underlying fitted lightgbm.Booster or model instance with long_booster/short_booster.
            symbol: Trading pair symbol (e.g. "BTCUSDT").
            bar_times: Sequence of UTC timestamps of length N.
            model_type: Directional barrier model identifier ("long_barrier" or "short_barrier").

        Returns:
            Tuple of length N containing SHAPExplanationResult entities.
        """
        if len(batch_features) == 0:
            raise ValueError("batch_features cannot be empty")
        if len(feature_names) == 0:
            raise ValueError("feature_names cannot be empty")
        if len(batch_features) != len(bar_times):
            raise ValueError(
                f"Batch length mismatch: {len(batch_features)} feature rows "
                f"!= {len(bar_times)} bar_times"
            )
        if model_type not in ("long_barrier", "short_barrier"):
            raise ValueError(
                f"Unknown model_type '{model_type}'; must be 'long_barrier' or 'short_barrier'"
            )

        booster = self._extract_booster(model=model, model_type=model_type)

        x_batch = np.asarray(batch_features, dtype=np.float64)
        if x_batch.ndim != 2:
            raise ValueError(f"batch_features must be 2D array, received ndim={x_batch.ndim}")
        if x_batch.shape[0] == 0:
            raise ValueError("batch_features cannot be empty")
        if x_batch.shape[1] != len(feature_names):
            raise ValueError(
                f"Feature dimension mismatch: expected {len(feature_names)}, got {x_batch.shape[1]}"
            )

        expected_num_features, booster_feature_names = self._get_booster_meta(booster)
        if x_batch.shape[1] != expected_num_features:
            raise ValueError(
                f"Feature dimension mismatch: booster expected {expected_num_features} features, "
                f"got {x_batch.shape[1]}"
            )

        if booster_feature_names and tuple(feature_names) != booster_feature_names:
            raise ValueError(
                f"Feature names mismatch: booster expected {list(booster_feature_names)}, "
                f"received {list(feature_names)}"
            )

        verifier = (
            self._sign_verifier_long if model_type == "long_barrier" else self._sign_verifier_short
        )
        for name in feature_names:
            if name not in verifier.constraints.constraints:
                raise ValueError(f"Unknown feature '{name}' not found in monotonic constraints")
            if name not in verifier.constraints.neutral_baselines:
                raise ValueError(f"Feature '{name}' missing neutral baseline")

        if not np.all(np.isfinite(x_batch)):
            raise ValueError("batch_features contains NaN or non-finite values")

        # Vectorized C++ TreeSHAP batch extraction & raw margin predictions
        raw_batch_pred = booster.predict(x_batch, pred_contrib=True)
        shap_matrix = np.asarray(raw_batch_pred, dtype=np.float64)

        raw_batch_scores = booster.predict(x_batch, raw_score=True)
        raw_scores_vec = np.asarray(raw_batch_scores, dtype=np.float64)

        results: list[SHAPExplanationResult] = []
        n_samples = x_batch.shape[0]

        for i in range(n_samples):
            phi_vals = shap_matrix[i, :-1]
            base_val = float(shap_matrix[i, -1])
            raw_margin = float(raw_scores_vec[i])

            # Verify Local Efficiency Axiom: sum(phi_j) + base_value == f(x)
            efficiency_diff = abs(float(np.sum(phi_vals) + base_val) - raw_margin)
            if efficiency_diff > self._config.local_efficiency_tol:
                raise ValueError(
                    f"Local efficiency violated at row {i}: diff={efficiency_diff:.2e} > "
                    f"tol={self._config.local_efficiency_tol}"
                )

            contribs, top_bull, top_bear, is_consistent = verifier.verify_and_rank(
                feature_names=feature_names,
                feature_values=x_batch[i].tolist(),
                shap_values=phi_vals.tolist(),
            )

            results.append(
                SHAPExplanationResult(
                    symbol=symbol,
                    bar_time=bar_times[i],
                    model_type=model_type,
                    base_value=base_val,
                    raw_margin=raw_margin,
                    contributions=contribs,
                    top_bullish_drivers=top_bull,
                    top_bearish_drivers=top_bear,
                    is_economic_consistent=is_consistent,
                )
            )

        return tuple(results)
