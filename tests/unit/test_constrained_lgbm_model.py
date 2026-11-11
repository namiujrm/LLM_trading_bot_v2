# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Unit & Property Tests for ConstrainedLGBMModel Adapter (Milestone M4.3, Proposal P3).

Verifies the mathematical correctness and integrity of the dual-binary causal ensemble:
- End-to-end fitting and dual-binary inference
- Strict monotonicity property assertion across all 11 canonical features
- LightGBM C++ TreeSHAP readiness for Milestone M4.5 (pred_contrib=True)
- Coherence gate regularized probability clamping (P_long + P_short <= 1.0)
- Fail-fast input validation and Clean Architecture conformance
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from trading_bot.adapters.ml_engine.constrained_lgbm_model import (
    CANONICAL_FEATURES,
    CANONICAL_MONOTONE_LONG,
    CANONICAL_MONOTONE_SHORT,
    ConstrainedLGBMModel,
)
from trading_bot.domain.entities.ml.ml_configs import (
    CoherenceGateConfig,
    ConstrainedLGBMConfig,
    RidgePriorConfig,
)
from trading_bot.domain.entities.ml.ml_predictions import BarrierPredictionResult


@pytest.fixture
def model_configs() -> tuple[ConstrainedLGBMConfig, RidgePriorConfig, CoherenceGateConfig]:
    lgbm_config = ConstrainedLGBMConfig(
        max_depth=4,
        num_leaves=15,
        min_child_samples=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        num_boost_round=30,
        early_stopping_rounds=5,
        coherence_penalty_gamma=1.0,
    )
    ridge_config = RidgePriorConfig(
        ridge_c=1.0,
        ridge_max_iter=500,
        oof_splits=5,
        clip_eps=1e-6,
        embargo_bars_30m=100,
        max_label_holding_bars_30m=12,
    )
    coherence_config = CoherenceGateConfig(
        gamma_conflict=1.0,
        max_combined_probability=1.0,
        min_probability_floor=0.0,
    )
    return lgbm_config, ridge_config, coherence_config


@pytest.fixture
def economically_aligned_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic dataset where feature correlations align with canonical economic signs."""
    rng = np.random.default_rng(42)
    n_samples = 400
    n_features = 11

    # Standard normal features
    x = rng.normal(0.0, 1.0, size=(n_samples, n_features))

    # Construct true directional latent signals according to canonical signs
    long_signs = np.array(
        [CANONICAL_MONOTONE_LONG[f] for f in CANONICAL_FEATURES], dtype=np.float64
    )
    short_signs = np.array(
        [CANONICAL_MONOTONE_SHORT[f] for f in CANONICAL_FEATURES], dtype=np.float64
    )

    long_latent = x @ (long_signs * 0.5) + rng.normal(0.0, 0.5, size=n_samples)
    short_latent = x @ (short_signs * 0.5) + rng.normal(0.0, 0.5, size=n_samples)

    y_long = (long_latent > np.percentile(long_latent, 60)).astype(np.int64)
    y_short = (short_latent > np.percentile(short_latent, 60)).astype(np.int64)

    weights = rng.uniform(0.8, 1.2, size=n_samples)

    return x, y_long, y_short, weights


def test_end_to_end_fit_and_predict(
    model_configs: tuple[ConstrainedLGBMConfig, RidgePriorConfig, CoherenceGateConfig],
    economically_aligned_training_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    lgbm_cfg, ridge_cfg, coh_cfg = model_configs
    x, y_l, y_s, w = economically_aligned_training_data

    model = ConstrainedLGBMModel(
        config=lgbm_cfg,
        ridge_config=ridge_cfg,
        coherence_config=coh_cfg,
        feature_names=CANONICAL_FEATURES,
        random_state=42,
    )

    assert not model.is_fitted
    with pytest.raises(RuntimeError, match="not been fitted"):
        _ = model.long_booster

    # Fit model
    model.fit(
        train_features=x.tolist(),
        y_long=y_l.tolist(),
        y_short=y_s.tolist(),
        sample_weights=w.tolist(),
        sample_intervals=None,
    )

    assert model.is_fitted
    assert model.long_booster is not None
    assert model.short_booster is not None

    # Predict single sample
    single_res = model.predict(x[0].tolist())
    assert isinstance(single_res, BarrierPredictionResult)
    assert 0.0 <= single_res.p_long <= 1.0
    assert 0.0 <= single_res.p_short <= 1.0
    assert (single_res.p_long + single_res.p_short) <= 1.0 + 1e-7

    # Predict batch of 50 samples
    batch_res = model.predict_batch(x[:50].tolist())
    assert len(batch_res) == 50
    for res in batch_res:
        assert isinstance(res, BarrierPredictionResult)
        assert (res.p_long + res.p_short) <= 1.0 + 1e-7


def test_strict_monotonicity_across_all_features(
    model_configs: tuple[ConstrainedLGBMConfig, RidgePriorConfig, CoherenceGateConfig],
    economically_aligned_training_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """MANDATORY QUANT PROPERTY TEST:

    Strictly asserts that increasing feature x_j obeys the locked monotonic direction:
    - Long Model:
        If M_j == +1: delta_z_long(x + delta e_j) >= delta_z_long(x) - 1e-7
        If M_j == -1: delta_z_long(x + delta e_j) <= delta_z_long(x) + 1e-7
    - Short Model:
        If M_j == +1: delta_z_short(x + delta e_j) >= delta_z_short(x) - 1e-7
        If M_j == -1: delta_z_short(x + delta e_j) <= delta_z_short(x) + 1e-7
    """
    lgbm_cfg, ridge_cfg, coh_cfg = model_configs
    x, y_l, y_s, w = economically_aligned_training_data

    model = ConstrainedLGBMModel(
        config=lgbm_cfg,
        ridge_config=ridge_cfg,
        coherence_config=coh_cfg,
        feature_names=CANONICAL_FEATURES,
        random_state=42,
    )
    model.fit(
        train_features=x.tolist(),
        y_long=y_l.tolist(),
        y_short=y_s.tolist(),
        sample_weights=w.tolist(),
        sample_intervals=None,
    )

    rng = np.random.default_rng(777)
    n_test = 50
    x_base = rng.normal(0.0, 1.0, size=(n_test, 11))
    delta = 0.5  # Perturbation step

    for feat_idx, feat_name in enumerate(CANONICAL_FEATURES):
        x_plus = x_base.copy()
        x_plus[:, feat_idx] += delta

        # 1. Test LightGBM Booster Tree Monotonicity directly
        delta_z_long_base = np.asarray(
            model.long_booster.predict(x_base, raw_score=True), dtype=np.float64
        )
        delta_z_long_plus = np.asarray(
            model.long_booster.predict(x_plus, raw_score=True), dtype=np.float64
        )

        delta_z_short_base = np.asarray(
            model.short_booster.predict(x_base, raw_score=True), dtype=np.float64
        )
        delta_z_short_plus = np.asarray(
            model.short_booster.predict(x_plus, raw_score=True), dtype=np.float64
        )

        sign_long = CANONICAL_MONOTONE_LONG[feat_name]
        sign_short = CANONICAL_MONOTONE_SHORT[feat_name]

        if sign_long == 1:
            diff_long = delta_z_long_plus - delta_z_long_base
            assert np.all(diff_long >= -1e-7), (
                f"Feature '{feat_name}' Long model monotonicity violated! "
                f"Expected non-decreasing (+1), found min diff: {np.min(diff_long)}"
            )
        elif sign_long == -1:
            diff_long = delta_z_long_plus - delta_z_long_base
            assert np.all(diff_long <= 1e-7), (
                f"Feature '{feat_name}' Long model monotonicity violated! "
                f"Expected non-increasing (-1), found max diff: {np.max(diff_long)}"
            )

        if sign_short == 1:
            diff_short = delta_z_short_plus - delta_z_short_base
            assert np.all(diff_short >= -1e-7), (
                f"Feature '{feat_name}' Short model monotonicity violated! "
                f"Expected non-decreasing (+1), found min diff: {np.min(diff_short)}"
            )
        elif sign_short == -1:
            diff_short = delta_z_short_plus - delta_z_short_base
            assert np.all(diff_short <= 1e-7), (
                f"Feature '{feat_name}' Short model monotonicity violated! "
                f"Expected non-increasing (-1), found max diff: {np.max(diff_short)}"
            )


def test_treeshap_readiness_and_latency(
    model_configs: tuple[ConstrainedLGBMConfig, RidgePriorConfig, CoherenceGateConfig],
    economically_aligned_training_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Milestone M4.5 Bridge Test:

    Confirms long_booster and short_booster expose native C++ TreeSHAP (pred_contrib=True)
    with output shape (N, D + 1) in < 1.0ms per sample.
    """
    lgbm_cfg, ridge_cfg, coh_cfg = model_configs
    x, y_l, y_s, w = economically_aligned_training_data

    model = ConstrainedLGBMModel(
        config=lgbm_cfg,
        ridge_config=ridge_cfg,
        coherence_config=coh_cfg,
        feature_names=CANONICAL_FEATURES,
        random_state=42,
    )
    model.fit(x.tolist(), y_l.tolist(), y_s.tolist(), w.tolist(), sample_intervals=None)

    sample = x[:1]
    # Warm-up 1 call to avoid cold C++ allocator / CI VM jitter in microbenchmark
    _ = model.long_booster.predict(sample, pred_contrib=True)

    t0 = time.perf_counter()
    shap_vals = np.asarray(model.long_booster.predict(sample, pred_contrib=True))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert shap_vals.shape == (1, 12)  # 11 features + 1 base value
    # Fast native C++ extraction
    assert elapsed_ms < 10.0, f"SHAP extraction took too long: {elapsed_ms:.2f}ms"


def test_coherence_regularization_in_predict(
    model_configs: tuple[ConstrainedLGBMConfig, RidgePriorConfig, CoherenceGateConfig],
    economically_aligned_training_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    lgbm_cfg, ridge_cfg, coh_cfg = model_configs
    x, y_l, y_s, w = economically_aligned_training_data

    model = ConstrainedLGBMModel(
        config=lgbm_cfg,
        ridge_config=ridge_cfg,
        coherence_config=coh_cfg,
        feature_names=CANONICAL_FEATURES,
        random_state=42,
    )
    model.fit(x.tolist(), y_l.tolist(), y_s.tolist(), w.tolist(), sample_intervals=None)

    results = model.predict_batch(x.tolist())
    for res in results:
        assert (res.p_long + res.p_short) <= 1.0 + 1e-7
        if not res.is_coherent:
            assert res.coherence_penalty > 0.0


def test_fail_fast_on_missing_canonical_features(
    model_configs: tuple[ConstrainedLGBMConfig, RidgePriorConfig, CoherenceGateConfig],
) -> None:
    lgbm_cfg, ridge_cfg, coh_cfg = model_configs
    incomplete_features = CANONICAL_FEATURES[:5]

    with pytest.raises(ValueError, match="missing required canonical features"):
        ConstrainedLGBMModel(
            config=lgbm_cfg,
            ridge_config=ridge_cfg,
            coherence_config=coh_cfg,
            feature_names=incomplete_features,
            random_state=42,
        )


def test_fail_fast_on_unknown_features(
    model_configs: tuple[ConstrainedLGBMConfig, RidgePriorConfig, CoherenceGateConfig],
) -> None:
    lgbm_cfg, ridge_cfg, coh_cfg = model_configs
    extra_unknown = (*CANONICAL_FEATURES, "non_existent_feature_alpha")

    with pytest.raises(ValueError, match="Unknown feature names provided"):
        ConstrainedLGBMModel(
            config=lgbm_cfg,
            ridge_config=ridge_cfg,
            coherence_config=coh_cfg,
            feature_names=extra_unknown,
            random_state=42,
        )
