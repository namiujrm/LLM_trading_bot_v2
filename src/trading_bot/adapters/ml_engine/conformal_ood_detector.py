# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""6D Subspace Conformal Empirical Quantile OOD Detector Adapter (Zero Defaults).

Implements OODDetectorPort using scikit-learn Ledoit-Wolf covariance shrinkage,
spectral condition number regularization (cond <= 100.0), and non-parametric
empirical conformal quantile thresholding.

Strict Clean Architecture: Concrete adapter in adapters/ml_engine layer.
Enforces Invariant #9: zero default parameters, zero class field defaults,
zero silent fallbacks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from sklearn.covariance import LedoitWolf

from trading_bot.domain.entities.ml.ml_configs import ConformalOODConfig
from trading_bot.domain.entities.ml.ml_predictions import OODDetectionResult
from trading_bot.ports.ml_model_port import OODDetectorPort


class ConformalSubspaceOODDetector(OODDetectorPort):
    """High-throughput Conformal Subspace OOD Detector with Ledoit-Wolf shrinkage."""

    def __init__(self, config: ConformalOODConfig) -> None:
        """Initialize detector with an explicit configuration schema.

        Args:
            config: ConformalOODConfig containing feature_names, alpha_significance,
                    kappa_shrinkage, max_condition_number, and min_eigenvalue_floor.
        """
        self._config: ConformalOODConfig = config
        self._mean_vector: np.ndarray | None = None
        self._precision_matrix: np.ndarray | None = None
        self._covariance_matrix: np.ndarray | None = None
        self._threshold_ood: float | None = None
        self._condition_number: float | None = None
        self._n_features: int = len(config.feature_names)

    @property
    def is_fitted(self) -> bool:
        """Check whether the detector has been fitted with valid matrices."""
        return (
            self._mean_vector is not None
            and self._precision_matrix is not None
            and self._threshold_ood is not None
            and self._condition_number is not None
        )

    @property
    def config(self) -> ConformalOODConfig:
        """Return the injected configuration schema."""
        return self._config

    @property
    def threshold(self) -> float:
        """Return the fitted empirical conformal threshold."""
        if self._threshold_ood is None:
            raise RuntimeError("Detector has not been fitted.")
        return self._threshold_ood

    @property
    def condition_number(self) -> float:
        """Return the condition number of the regularized covariance matrix."""
        if self._condition_number is None:
            raise RuntimeError("Detector has not been fitted.")
        return self._condition_number

    @property
    def precision_matrix(self) -> np.ndarray:
        """Return the regularized precision matrix (inverse covariance)."""
        if self._precision_matrix is None:
            raise RuntimeError("Detector has not been fitted.")
        return self._precision_matrix

    @property
    def covariance_matrix(self) -> np.ndarray:
        """Return the regularized covariance matrix."""
        if self._covariance_matrix is None:
            raise RuntimeError("Detector has not been fitted.")
        return self._covariance_matrix

    @property
    def mean_vector(self) -> np.ndarray:
        """Return the estimated mean vector."""
        if self._mean_vector is None:
            raise RuntimeError("Detector has not been fitted.")
        return self._mean_vector

    def fit(
        self,
        train_features: Sequence[Sequence[float]],
        calibration_features: Sequence[Sequence[float]],
    ) -> None:
        """Fit Ledoit-Wolf covariance on train and compute conformal quantile on calibration.

        Args:
            train_features: Sequence of training feature vectors (shape N_train x d).
            calibration_features: Sequence of calibration feature vectors (shape N_cal x d).

        Raises:
            ValueError: If sample counts < n_features, dimensions mismatch, or inputs contain NaN/Inf.
        """
        if len(train_features) < self._n_features:
            raise ValueError(
                f"Insufficient training samples: N_train={len(train_features)} is less than feature dimension {self._n_features}"
            )
        if len(calibration_features) < self._n_features:
            raise ValueError(
                f"Insufficient calibration samples: N_cal={len(calibration_features)} is less than feature dimension {self._n_features}"
            )

        arr_train = np.asarray(train_features, dtype=np.float64)
        arr_cal = np.asarray(calibration_features, dtype=np.float64)

        if arr_train.ndim != 2:
            raise ValueError(f"train_features must be 2D array, received ndim={arr_train.ndim}")
        if arr_cal.ndim != 2:
            raise ValueError(f"calibration_features must be 2D array, received ndim={arr_cal.ndim}")

        if arr_train.shape[1] != self._n_features:
            raise ValueError(
                f"train_features feature dimension {arr_train.shape[1]} does not match "
                f"expected {self._n_features} features"
            )
        if arr_cal.shape[1] != self._n_features:
            raise ValueError(
                f"calibration_features feature dimension {arr_cal.shape[1]} does not match "
                f"expected {self._n_features} features"
            )

        if not np.all(np.isfinite(arr_train)):
            raise ValueError("train_features contains non-finite values (NaN or Inf)")
        if not np.all(np.isfinite(arr_cal)):
            raise ValueError("calibration_features contains non-finite values (NaN or Inf)")

        # Verify training set has non-zero variance
        variances = np.var(arr_train, axis=0)
        if np.all(variances <= self._config.min_eigenvalue_floor):
            raise ValueError("All features in train_features exhibit zero or near-zero variance")

        # 1. Ledoit-Wolf Covariance Shrinkage Estimation
        lw = LedoitWolf(assume_centered=False, store_precision=False)
        lw.fit(cast(Any, arr_train))
        mu = np.asarray(lw.location_, dtype=np.float64)
        sigma_lw = np.asarray(lw.covariance_, dtype=np.float64)

        # 2. Spectral Decomposition & Condition Number Regularization
        evals, evecs = np.linalg.eigh(sigma_lw)
        lambda_max = float(evals[-1])

        if lambda_max <= self._config.min_eigenvalue_floor:
            floor_eval = self._config.min_eigenvalue_floor
            clamped_evals = np.full_like(evals, floor_eval)
        else:
            floor_eval = max(
                lambda_max / self._config.max_condition_number,
                self._config.min_eigenvalue_floor,
            )
            clamped_evals = np.maximum(evals, floor_eval)

        cond_num = float(clamped_evals[-1] / clamped_evals[0])

        # Precision Matrix: Sigma^-1 = Q * diag(1 / lambda) * Q^T
        precision = (evecs / clamped_evals) @ evecs.T
        reg_cov = (evecs * clamped_evals) @ evecs.T

        # 3. Vectorized Mahalanobis Distance on Calibration Set
        diff_cal = arr_cal - mu
        prec_diff_cal = diff_cal @ precision
        d_sq_cal = np.sum(prec_diff_cal * diff_cal, axis=1)
        d_cal = np.sqrt(np.maximum(0.0, d_sq_cal))

        # 4. Empirical Conformal Quantile (1 - alpha)
        pct = 100.0 * (1.0 - self._config.alpha_significance)
        threshold_ood = float(np.percentile(d_cal, pct, method="linear"))

        if not np.isfinite(threshold_ood) or threshold_ood <= 0.0:
            raise ValueError(
                f"Empirical conformal threshold {threshold_ood} is non-positive. "
                "Calibration samples exhibit zero Mahalanobis distance or insufficient variance relative to training mean."
            )

        self._mean_vector = mu
        self._precision_matrix = precision
        self._covariance_matrix = reg_cov
        self._condition_number = cond_num
        self._threshold_ood = threshold_ood

    def predict(self, sample_features: Sequence[float]) -> OODDetectionResult:
        """Evaluate a single feature vector against empirical conformal threshold.

        Args:
            sample_features: 1D feature vector of length n_features.

        Returns:
            OODDetectionResult with distance, threshold, OOD flag, shrinkage, and feature contributions.
        """
        if (
            self._mean_vector is None
            or self._precision_matrix is None
            or self._threshold_ood is None
            or self._condition_number is None
        ):
            raise RuntimeError("ConformalSubspaceOODDetector has not been fitted.")

        x = np.asarray(sample_features, dtype=np.float64)
        if x.ndim != 1 or len(x) != self._n_features:
            raise ValueError(
                f"sample_features must be 1D vector of length {self._n_features}, received shape {x.shape}"
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("sample_features contains non-finite values (NaN or Inf)")

        return self.predict_batch([sample_features])[0]

    def predict_batch(
        self, batch_features: Sequence[Sequence[float]]
    ) -> Sequence[OODDetectionResult]:
        """Evaluate a batch of feature vectors using high-performance vectorization.

        Args:
            batch_features: 2D array-like of shape (M, n_features).

        Returns:
            Sequence of OODDetectionResult objects.
        """
        if (
            self._mean_vector is None
            or self._precision_matrix is None
            or self._threshold_ood is None
            or self._condition_number is None
        ):
            raise RuntimeError("ConformalSubspaceOODDetector has not been fitted.")

        arr = np.asarray(batch_features, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self._n_features:
            raise ValueError(
                f"batch_features must be 2D array of shape (M, {self._n_features}), received shape {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("batch_features contains non-finite values (NaN or Inf)")

        diff = arr - self._mean_vector
        prec_diff = diff @ self._precision_matrix
        d_sq = np.sum(prec_diff * diff, axis=1)
        d_m_array = np.sqrt(np.maximum(0.0, d_sq))

        threshold = self._threshold_ood
        is_ood_array = d_m_array > threshold
        excess_array = np.maximum(0.0, (d_m_array - threshold) / threshold)
        gamma_array = np.where(
            is_ood_array,
            np.clip(np.exp(-self._config.kappa_shrinkage * excess_array), 0.0, 1.0),
            1.0,
        )

        contrib_matrix = diff * prec_diff
        contrib_all: list[list[float]] = contrib_matrix.tolist()
        d_m_list: list[float] = d_m_array.tolist()
        is_ood_list: list[bool] = is_ood_array.tolist()
        excess_list: list[float] = excess_array.tolist()
        gamma_list: list[float] = gamma_array.tolist()

        return [
            OODDetectionResult(
                is_ood=is_ood,
                mahalanobis_distance=dm,
                empirical_quantile_threshold=threshold,
                excess_distance=excess,
                gamma_ood=gamma,
                feature_contributions=tuple(contrib),
            )
            for is_ood, dm, excess, gamma, contrib in zip(
                is_ood_list, d_m_list, excess_list, gamma_list, contrib_all, strict=True
            )
        ]

    def evaluate(self, sample_features: Sequence[float]) -> OODDetectionResult:
        """Alias for predict(sample_features) conforming to evaluation interface."""
        return self.predict(sample_features)
