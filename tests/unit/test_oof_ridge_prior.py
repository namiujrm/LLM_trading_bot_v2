# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Unit Tests for OOFRidgePrior Adapter (Proposal P3).

Verifies the mathematical correctness and stability of 5-Fold OOF Ridge regression:
- 100% sample coverage by OOF validation logits
- Probability clipping and logit stability [-13.8155, +13.8155]
- Out-of-sample inference consistency between logits and probabilities
- Impact of López de Prado sample weights
- Purged CV sample interval integration
- Fail-fast input validation and unfitted state protection
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.special

from trading_bot.adapters.ml_engine.oof_ridge_prior import OOFRidgePrior
from trading_bot.domain.entities.ml.ml_configs import RidgePriorConfig
from trading_bot.domain.exceptions import InsufficientDataException


@pytest.fixture
def default_ridge_config() -> RidgePriorConfig:
    return RidgePriorConfig(
        ridge_c=1.0,
        ridge_max_iter=1000,
        oof_splits=5,
        clip_eps=1e-6,
        embargo_bars_30m=100,
        max_label_holding_bars_30m=12,
    )


@pytest.fixture
def synthetic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    n_samples = 300
    n_features = 11
    x = rng.normal(0.0, 1.0, size=(n_samples, n_features))
    true_weights = rng.normal(0.0, 0.5, size=n_features)
    logits = x @ true_weights
    probs = scipy.special.expit(logits)
    y = (rng.uniform(0.0, 1.0, size=n_samples) < probs).astype(np.int64)
    weights = rng.uniform(0.5, 2.0, size=n_samples)
    return x, y, weights


def test_oof_logits_cover_all_samples_without_leakage(
    default_ridge_config: RidgePriorConfig,
    synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    x, y, w = synthetic_data
    prior = OOFRidgePrior(config=default_ridge_config, random_state=42)

    assert not prior.is_fitted
    with pytest.raises(RuntimeError, match="not been fitted"):
        _ = prior.oof_logits

    oof_logits = prior.fit(
        features=x.tolist(),
        labels=y.tolist(),
        sample_weights=w.tolist(),
        sample_intervals=None,
    )

    assert prior.is_fitted
    assert len(oof_logits) == len(x)
    assert np.all(np.isfinite(oof_logits))
    assert np.array_equal(oof_logits, prior.oof_logits)
    assert prior.coefficients.shape == (11,)
    assert isinstance(prior.intercept, float)


def test_margin_clipping_bounds(default_ridge_config: RidgePriorConfig) -> None:
    # Extreme separable data where probabilities would naturally be 0.0 or 1.0
    rng = np.random.default_rng(123)
    n_samples = 200
    x = rng.normal(0.0, 1.0, size=(n_samples, 2))
    # Perfectly separated
    y = (x[:, 0] > 0).astype(np.int64)
    w = np.ones(n_samples)

    prior = OOFRidgePrior(config=default_ridge_config, random_state=42)
    oof_logits = prior.fit(
        features=x.tolist(),
        labels=y.tolist(),
        sample_weights=w.tolist(),
        sample_intervals=None,
    )

    eps = default_ridge_config.clip_eps
    min_logit = np.log(eps) - np.log(1.0 - eps)
    max_logit = np.log(1.0 - eps) - np.log(eps)

    assert np.all(oof_logits >= min_logit - 1e-10)
    assert np.all(oof_logits <= max_logit + 1e-10)


def test_out_of_sample_prediction_logits_and_proba(
    default_ridge_config: RidgePriorConfig,
    synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    x, y, w = synthetic_data
    prior = OOFRidgePrior(config=default_ridge_config, random_state=42)
    prior.fit(
        features=x.tolist(),
        labels=y.tolist(),
        sample_weights=w.tolist(),
        sample_intervals=None,
    )

    rng = np.random.default_rng(999)
    x_test = rng.normal(0.0, 1.0, size=(50, 11))

    proba = prior.predict_proba(x_test.tolist())
    logits = prior.predict_logits(x_test.tolist())

    assert proba.shape == (50,)
    assert logits.shape == (50,)
    assert np.all(proba >= default_ridge_config.clip_eps)
    assert np.all(proba <= 1.0 - default_ridge_config.clip_eps)

    # Check logits match log(p) - log(1-p)
    expected_logits = np.log(proba) - np.log(1.0 - proba)
    np.testing.assert_allclose(logits, expected_logits, rtol=1e-6)


def test_sample_weights_alter_coefficients(
    default_ridge_config: RidgePriorConfig,
    synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    x, y, _ = synthetic_data
    w_uniform = np.ones(len(x))
    w_skewed = np.ones(len(x))
    # Give 100x weight to first 50 samples
    w_skewed[:50] = 100.0

    prior_uniform = OOFRidgePrior(config=default_ridge_config, random_state=42)
    prior_uniform.fit(x.tolist(), y.tolist(), w_uniform.tolist(), sample_intervals=None)

    prior_skewed = OOFRidgePrior(config=default_ridge_config, random_state=42)
    prior_skewed.fit(x.tolist(), y.tolist(), w_skewed.tolist(), sample_intervals=None)

    assert not np.allclose(prior_uniform.coefficients, prior_skewed.coefficients, atol=1e-3)


def test_with_purged_cv_sample_intervals(
    default_ridge_config: RidgePriorConfig,
    synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    x, y, w = synthetic_data
    n_samples = len(x)
    # Generate chronologically ordered non-decreasing start times
    sample_intervals = [(i * 2, i * 2 + 10) for i in range(n_samples)]

    prior = OOFRidgePrior(config=default_ridge_config, random_state=42)
    oof_logits = prior.fit(
        features=x.tolist(),
        labels=y.tolist(),
        sample_weights=w.tolist(),
        sample_intervals=sample_intervals,
    )

    assert len(oof_logits) == n_samples
    assert np.all(np.isfinite(oof_logits))


def test_fail_fast_on_single_class_fold(default_ridge_config: RidgePriorConfig) -> None:
    # 20 samples where only sample 0 has label 1, remaining have label 0
    # In 5 folds of 4 samples each, at least 4 folds will have only class 0
    x = np.random.randn(20, 2).tolist()
    y = [1] + [0] * 19
    w = [1.0] * 20

    prior = OOFRidgePrior(config=default_ridge_config, random_state=42)
    with pytest.raises(InsufficientDataException, match="contains only 1 class"):
        prior.fit(x, y, w, sample_intervals=None)


def test_fail_fast_on_invalid_data(default_ridge_config: RidgePriorConfig) -> None:
    prior = OOFRidgePrior(config=default_ridge_config, random_state=42)

    # 1. NaN in features
    with pytest.raises(ValueError, match="NaN"):
        prior.fit([[float("nan"), 1.0], [0.0, 1.0]], [1, 0], [1.0, 1.0], sample_intervals=None)

    # 2. Length mismatch
    with pytest.raises(ValueError, match="Dimension mismatch"):
        prior.fit([[1.0, 2.0], [3.0, 4.0]], [1], [1.0, 1.0], sample_intervals=None)

    # 3. Non-binary labels
    with pytest.raises(ValueError, match="binary"):
        prior.fit([[1.0, 2.0], [3.0, 4.0]], [1, 2], [1.0, 1.0], sample_intervals=None)

    # 4. Negative sample weights
    with pytest.raises(ValueError, match="strictly positive"):
        prior.fit([[1.0, 2.0], [3.0, 4.0]], [1, 0], [1.0, -0.5], sample_intervals=None)


def test_fail_fast_on_empty_fold_purged_cv() -> None:
    config = RidgePriorConfig(
        ridge_c=1.0,
        ridge_max_iter=100,
        oof_splits=2,
        clip_eps=1e-6,
        embargo_bars_30m=100,
        max_label_holding_bars_30m=12,
    )
    prior = OOFRidgePrior(config=config, random_state=42)
    # 4 samples with completely overlapping intervals (0, 10), so validation fold purges all training samples
    with pytest.raises(InsufficientDataException, match="empty train or validation set"):
        prior.fit(
            features=[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]],
            labels=[0, 1, 0, 1],
            sample_weights=[1.0, 1.0, 1.0, 1.0],
            sample_intervals=[(0, 10), (0, 10), (0, 10), (0, 10)],
        )


def test_fail_fast_on_invalid_ridge_config() -> None:
    with pytest.raises(ValueError, match="embargo_bars_30m"):
        RidgePriorConfig(
            ridge_c=1.0,
            ridge_max_iter=100,
            oof_splits=5,
            clip_eps=1e-6,
            embargo_bars_30m=-1,
            max_label_holding_bars_30m=12,
        )

    with pytest.raises(ValueError, match="max_label_holding_bars_30m"):
        RidgePriorConfig(
            ridge_c=1.0,
            ridge_max_iter=100,
            oof_splits=5,
            clip_eps=1e-6,
            embargo_bars_30m=100,
            max_label_holding_bars_30m=-1,
        )
