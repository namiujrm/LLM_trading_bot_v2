# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""5-Fold Out-of-Fold (OOF) Logistic Ridge Prior Adapter (Zero Defaults).

Strict Clean Architecture: Concrete adapter in adapters/ml_engine layer.
Enforces Invariant #9: zero default parameters, zero class field defaults,
zero silent fallbacks, fail-fast input validation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from sklearn.linear_model import LogisticRegression

from trading_bot.domain.entities.ml.ml_configs import PurgedCVConfig, RidgePriorConfig
from trading_bot.domain.exceptions import InsufficientDataException
from trading_bot.domain.services.ml.purged_cv_splitter import PurgedCVSplitter


class OOFRidgePrior:
    """5-Fold Out-of-Fold Logistic Ridge Linear Prior generating init_score margins."""

    def __init__(
        self,
        config: RidgePriorConfig,
        random_state: int,
    ) -> None:
        """Initialize OOF Ridge Prior with explicit configuration and random seed.

        Args:
            config: RidgePriorConfig containing ridge_c, ridge_max_iter, oof_splits, clip_eps,
                embargo_bars_30m, max_label_holding_bars_30m.
            random_state: Explicit integer seed for deterministic fitting.
        """
        if config.ridge_c <= 0.0:
            raise ValueError(f"ridge_c must be strictly positive, received {config.ridge_c}")
        if config.ridge_max_iter <= 0:
            raise ValueError(
                f"ridge_max_iter must be strictly positive, received {config.ridge_max_iter}"
            )
        if config.oof_splits < 2:
            raise ValueError(f"oof_splits must be at least 2, received {config.oof_splits}")
        if not (0.0 < config.clip_eps < 0.5):
            raise ValueError(f"clip_eps must be in range (0.0, 0.5), received {config.clip_eps}")
        if config.embargo_bars_30m < 0:
            raise ValueError(
                f"embargo_bars_30m must be non-negative, received {config.embargo_bars_30m}"
            )
        if config.max_label_holding_bars_30m < 0:
            raise ValueError(
                f"max_label_holding_bars_30m must be non-negative, received {config.max_label_holding_bars_30m}"
            )

        self._config: RidgePriorConfig = config
        self._random_state: int = random_state
        self._full_model: LogisticRegression | None = None
        self._n_features: int | None = None
        self._oof_logits: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        """Check whether full model and OOF logits have been successfully fitted."""
        return self._full_model is not None and self._oof_logits is not None

    @property
    def config(self) -> RidgePriorConfig:
        """Return the injected RidgePriorConfig."""
        return self._config

    @property
    def full_model(self) -> LogisticRegression:
        """Return the full fitted LogisticRegression model."""
        if self._full_model is None:
            raise RuntimeError("OOFRidgePrior has not been fitted.")
        return self._full_model

    @property
    def oof_logits(self) -> np.ndarray:
        """Return the 1D array of OOF logit margins across the training dataset."""
        if self._oof_logits is None:
            raise RuntimeError("OOFRidgePrior has not been fitted.")
        return self._oof_logits

    @property
    def coefficients(self) -> np.ndarray:
        """Return coefficients of the full model (1D array of length n_features)."""
        if self._full_model is None or self._full_model.coef_ is None:
            raise RuntimeError("OOFRidgePrior has not been fitted.")
        return np.asarray(self._full_model.coef_[0], dtype=np.float64)

    @property
    def intercept(self) -> float:
        """Return scalar intercept of the full model."""
        if self._full_model is None or self._full_model.intercept_ is None:
            raise RuntimeError("OOFRidgePrior has not been fitted.")
        intercept_val = self._full_model.intercept_
        if isinstance(intercept_val, np.ndarray):
            return float(intercept_val[0])
        return float(intercept_val)

    @classmethod
    def from_saved_weights(
        cls,
        config: RidgePriorConfig,
        random_state: int,
        n_features: int,
        coefficients: Sequence[float],
        intercept: float,
    ) -> OOFRidgePrior:
        """Construct fitted OOFRidgePrior from previously saved linear weights and intercept."""
        prior = cls(config=config, random_state=random_state)
        clf = LogisticRegression(
            C=config.ridge_c,
            max_iter=config.ridge_max_iter,
            random_state=random_state,
            solver="lbfgs",
        )
        clf.coef_ = np.asarray(coefficients, dtype=np.float64).reshape(1, -1)
        clf.intercept_ = np.asarray([intercept], dtype=np.float64)
        clf.classes_ = np.array([0, 1], dtype=np.int64)

        prior._full_model = clf
        prior._n_features = n_features
        prior._oof_logits = np.zeros(1, dtype=np.float64)
        return prior

    def fit(
        self,
        features: Sequence[Sequence[float]],
        labels: Sequence[int],
        sample_weights: Sequence[float],
        sample_intervals: Sequence[tuple[int, int]] | None,
    ) -> np.ndarray:
        """Fit 5-fold OOF Ridge models to compute training logits, then fit full model.

        Args:
            features: 2D sequence of shape (N, D).
            labels: 1D sequence of binary labels {0, 1} of length N.
            sample_weights: 1D sequence of strictly positive weights of length N.
            sample_intervals: Optional sequence of (t_start, t_end) intervals for Purged CV.

        Returns:
            np.ndarray: 1D array of OOF logit margins of length N.

        Raises:
            ValueError: If dimensions mismatch, non-finite values present, or invalid labels.
            InsufficientDataException: If sample count < oof_splits * 2 or class count < 2.
        """
        x_arr = np.asarray(features, dtype=np.float64)
        y_arr = np.asarray(labels, dtype=np.int64)
        w_arr = np.asarray(sample_weights, dtype=np.float64)

        if x_arr.ndim != 2:
            raise ValueError(f"features must be 2D array, received ndim={x_arr.ndim}")
        if y_arr.ndim != 1:
            raise ValueError(f"labels must be 1D array, received ndim={y_arr.ndim}")
        if w_arr.ndim != 1:
            raise ValueError(f"sample_weights must be 1D array, received ndim={w_arr.ndim}")

        n_samples, n_feats = x_arr.shape
        if len(y_arr) != n_samples:
            raise ValueError(
                f"Dimension mismatch: features has {n_samples} rows, labels has {len(y_arr)} elements"
            )
        if len(w_arr) != n_samples:
            raise ValueError(
                f"Dimension mismatch: features has {n_samples} rows, sample_weights has {len(w_arr)} elements"
            )

        if not np.all(np.isfinite(x_arr)):
            raise ValueError("features contains NaN or non-finite values")
        if not np.all(np.isfinite(w_arr)):
            raise ValueError("sample_weights contains NaN or non-finite values")
        if np.any(w_arr <= 0.0):
            raise ValueError("sample_weights must contain strictly positive values")

        unique_labels = set(np.unique(y_arr))
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"labels must be binary in {{0, 1}}, received {unique_labels}")

        if len(unique_labels) < 2:
            raise InsufficientDataException(
                f"Training dataset contains only 1 class ({next(iter(unique_labels))}). Minimum 2 classes required."
            )

        k_splits = self._config.oof_splits
        if n_samples < k_splits * 2:
            raise InsufficientDataException(
                f"Sample count ({n_samples}) is too small for {k_splits}-fold OOF Ridge (minimum {k_splits * 2})."
            )

        # Build train/val splits
        splits_list: list[tuple[list[int], list[int]]] = []
        if sample_intervals is not None:
            if len(sample_intervals) != n_samples:
                raise ValueError(
                    f"sample_intervals length ({len(sample_intervals)}) does not match sample count ({n_samples})"
                )
            purged_splitter = PurgedCVSplitter(
                PurgedCVConfig(
                    n_splits=k_splits,
                    embargo_bars_30m=self._config.embargo_bars_30m,
                    max_label_holding_bars_30m=self._config.max_label_holding_bars_30m,
                )
            )
            purged_splits = purged_splitter.split(sample_intervals=sample_intervals)
            for split_obj in purged_splits:
                splits_list.append((split_obj.train_indices, split_obj.val_indices))
        else:
            fold_size = n_samples // k_splits
            remainder = n_samples % k_splits
            for k in range(k_splits):
                val_start = k * fold_size + min(k, remainder)
                val_end = (k + 1) * fold_size + min(k + 1, remainder)
                val_idx = list(range(val_start, val_end))
                val_set = set(val_idx)
                train_idx = [i for i in range(n_samples) if i not in val_set]
                splits_list.append((train_idx, val_idx))

        oof_logits = np.zeros(n_samples, dtype=np.float64)
        eps = self._config.clip_eps

        for fold_idx, (train_idx, val_idx) in enumerate(splits_list):
            if len(train_idx) == 0 or len(val_idx) == 0:
                raise InsufficientDataException(
                    f"Fold {fold_idx} has empty train or validation set ({len(train_idx)}, {len(val_idx)})."
                )

            x_train = x_arr[train_idx]
            y_train = y_arr[train_idx]
            w_train = w_arr[train_idx]
            x_val = x_arr[val_idx]

            fold_classes = np.unique(y_train)
            if len(fold_classes) < 2:
                class_str = str(fold_classes[0]) if len(fold_classes) > 0 else "0"
                raise InsufficientDataException(
                    f"Fold {fold_idx} contains only {len(fold_classes)} class ({class_str}). Minimum 2 classes required."
                )

            fold_model = LogisticRegression(
                C=self._config.ridge_c,
                max_iter=self._config.ridge_max_iter,
                random_state=self._random_state + fold_idx,
                solver="lbfgs",
            )
            fold_model.fit(x_train, y_train, sample_weight=w_train)

            # Predict probabilities on out-of-fold validation samples
            p_val = fold_model.predict_proba(cast(Any, x_val))[:, 1]
            p_clipped = np.clip(p_val, eps, 1.0 - eps)
            z_val = np.log(p_clipped) - np.log(1.0 - p_clipped)
            oof_logits[val_idx] = z_val

        # Fit Full Model on all data
        full_model = LogisticRegression(
            C=self._config.ridge_c,
            max_iter=self._config.ridge_max_iter,
            random_state=self._random_state,
            solver="lbfgs",
        )
        full_model.fit(x_arr, y_arr, sample_weight=w_arr)

        self._full_model = full_model
        self._n_features = n_feats
        self._oof_logits = oof_logits

        return oof_logits

    def predict_proba(
        self,
        sample_features: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        """Predict out-of-sample probabilities using full fitted Ridge model.

        Args:
            sample_features: 2D sequence or array of shape (M, D).

        Returns:
            np.ndarray: 1D array of probabilities of length M clipped to [clip_eps, 1 - clip_eps].

        Raises:
            RuntimeError: If model is not fitted.
            ValueError: If input dimensions mismatch or non-finite values present.
        """
        if self._full_model is None or self._n_features is None:
            raise RuntimeError("OOFRidgePrior has not been fitted.")

        x_test = np.asarray(sample_features, dtype=np.float64)
        if x_test.ndim != 2:
            raise ValueError(f"sample_features must be 2D array, received ndim={x_test.ndim}")
        if x_test.shape[1] != self._n_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self._n_features}, received {x_test.shape[1]}"
            )
        if not np.all(np.isfinite(x_test)):
            raise ValueError("sample_features contains NaN or non-finite values")

        p = self._full_model.predict_proba(cast(Any, x_test))[:, 1]
        eps = self._config.clip_eps
        return np.clip(p, eps, 1.0 - eps)

    def predict_logits(
        self,
        sample_features: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        """Predict out-of-sample logit margins using full fitted Ridge model.

        Args:
            sample_features: 2D sequence of shape (M, D).

        Returns:
            np.ndarray: 1D array of logit margins of length M.

        Raises:
            RuntimeError: If model is not fitted.
            ValueError: If input dimensions mismatch or non-finite values present.
        """
        p_clipped = self.predict_proba(sample_features)
        return np.log(p_clipped) - np.log(1.0 - p_clipped)
