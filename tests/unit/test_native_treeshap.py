# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Unit tests for NativeTreeSHAPExplainer Adapter (Milestone M4.5, Proposal P6)."""

from __future__ import annotations

import time
from datetime import UTC, datetime

import lightgbm as lgb
import numpy as np
import pytest

from trading_bot.adapters.ml_engine.constrained_lgbm_model import (
    CANONICAL_FEATURES,
    ConstrainedLGBMModel,
)
from trading_bot.adapters.ml_engine.native_treeshap_explainer import (
    NativeTreeSHAPExplainer,
)
from trading_bot.domain.entities.ml.ml_configs import (
    CoherenceGateConfig,
    ConstrainedLGBMConfig,
    LGBMMonotonicConstraints,
    RidgePriorConfig,
    TreeSHAPConfig,
)
from trading_bot.domain.entities.ml.ml_predictions import SHAPExplanationResult
from trading_bot.domain.services.ml.economic_sign_verifier import EconomicSignVerifier


@pytest.fixture
def treeshap_explainer() -> tuple[NativeTreeSHAPExplainer, TreeSHAPConfig]:
    config = TreeSHAPConfig(
        top_k_bullish=5,
        top_k_bearish=3,
        local_efficiency_tol=1e-4,
        sign_tolerance=1e-4,
    )
    long_constraints = LGBMMonotonicConstraints.create_canonical_long()
    short_constraints = LGBMMonotonicConstraints.create_canonical_short()

    verifier_long = EconomicSignVerifier(config=config, monotonic_constraints=long_constraints)
    verifier_short = EconomicSignVerifier(config=config, monotonic_constraints=short_constraints)

    explainer = NativeTreeSHAPExplainer(
        config=config,
        sign_verifier_long=verifier_long,
        sign_verifier_short=verifier_short,
    )
    return explainer, config


@pytest.fixture
def trained_models() -> tuple[lgb.Booster, ConstrainedLGBMModel]:
    rng = np.random.default_rng(42)
    n_samples = 300
    n_features = 11

    x_train = rng.normal(0.0, 1.0, size=(n_samples, n_features))
    # Synthetic target with strong linear component
    y_raw_long = (
        0.5 * x_train[:, 0]
        - 0.4 * x_train[:, 1]
        + 0.6 * x_train[:, 4]
        + rng.normal(0.0, 0.2, size=n_samples)
    )
    y_long = (y_raw_long > 0.0).astype(int)

    y_raw_short = -0.5 * x_train[:, 2] - 0.6 * x_train[:, 4] + rng.normal(0.0, 0.2, size=n_samples)
    y_short = (y_raw_short > 0.0).astype(int)

    # Train raw booster
    ds = lgb.Dataset(x_train, label=y_long, feature_name=list(CANONICAL_FEATURES))
    params = {
        "objective": "binary",
        "max_depth": 4,
        "num_leaves": 15,
        "learning_rate": 0.05,
        "verbosity": -1,
    }
    raw_booster = lgb.train(params, ds, num_boost_round=25)

    # Train ConstrainedLGBMModel
    lgbm_config = ConstrainedLGBMConfig(
        max_depth=4,
        num_leaves=15,
        min_child_samples=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        num_boost_round=25,
        early_stopping_rounds=5,
        coherence_penalty_gamma=1.0,
    )
    ridge_config = RidgePriorConfig(
        ridge_c=1.0,
        ridge_max_iter=500,
        oof_splits=5,
        clip_eps=1e-6,
        embargo_bars_30m=10,
        max_label_holding_bars_30m=5,
    )
    coherence_config = CoherenceGateConfig(
        gamma_conflict=1.0,
        max_combined_probability=1.0,
        min_probability_floor=0.0,
    )

    constrained_model = ConstrainedLGBMModel(
        config=lgbm_config,
        ridge_config=ridge_config,
        coherence_config=coherence_config,
        feature_names=CANONICAL_FEATURES,
        random_state=42,
    )

    weights = [1.0] * n_samples
    intervals = [(i, i + 1) for i in range(n_samples)]
    constrained_model.fit(
        train_features=x_train.tolist(),
        y_long=y_long.tolist(),
        y_short=y_short.tolist(),
        sample_weights=weights,
        sample_intervals=intervals,
    )

    return raw_booster, constrained_model


def _sample_time() -> datetime:
    return datetime(2026, 11, 8, 14, 0, tzinfo=UTC)


def test_adapter_properties_and_accessors(
    treeshap_explainer: tuple[NativeTreeSHAPExplainer, TreeSHAPConfig],
) -> None:
    """Test explainer properties and sub-verifier exposure."""
    explainer, config = treeshap_explainer
    assert explainer.config is config
    assert isinstance(explainer.verifier_long, EconomicSignVerifier)
    assert isinstance(explainer.verifier_short, EconomicSignVerifier)


def test_explain_single_sample_with_raw_booster(
    treeshap_explainer: tuple[NativeTreeSHAPExplainer, TreeSHAPConfig],
    trained_models: tuple[lgb.Booster, ConstrainedLGBMModel],
) -> None:
    """Test single sample explanation with direct lgb.Booster."""
    explainer, config = treeshap_explainer
    raw_booster, _ = trained_models

    feature_names = tuple(CANONICAL_FEATURES)
    feature_vector = (0.7, 0.3, 1.1, 0.4, 1.8, 0.5, -0.2, 0.3, -0.1, 0.4, 0.2)

    result = explainer.explain(
        feature_vector=feature_vector,
        feature_names=feature_names,
        model=raw_booster,
        symbol="BTCUSDT",
        bar_time=_sample_time(),
        model_type="long_barrier",
    )

    assert isinstance(result, SHAPExplanationResult)
    assert result.symbol == "BTCUSDT"
    assert result.model_type == "long_barrier"
    assert len(result.contributions) == 11

    # Local efficiency test
    eff_diff = abs(result.total_feature_attribution + result.base_value - result.raw_margin)
    assert eff_diff <= config.local_efficiency_tol

    # Top drivers ordering
    assert result.num_bullish_drivers <= config.top_k_bullish
    assert result.num_bearish_drivers <= config.top_k_bearish
    for idx, bull in enumerate(result.top_bullish_drivers):
        assert bull.shap_value > 0.0
        if idx > 0:
            assert bull.shap_value <= result.top_bullish_drivers[idx - 1].shap_value

    for idx, bear in enumerate(result.top_bearish_drivers):
        assert bear.shap_value < 0.0
        if idx > 0:
            assert bear.shap_value >= result.top_bearish_drivers[idx - 1].shap_value


def test_explain_with_constrained_lgbm_model(
    treeshap_explainer: tuple[NativeTreeSHAPExplainer, TreeSHAPConfig],
    trained_models: tuple[lgb.Booster, ConstrainedLGBMModel],
) -> None:
    """Test booster extraction from ConstrainedLGBMModel for Long and Short barriers."""
    explainer, _ = treeshap_explainer
    _, constrained_model = trained_models

    feature_names = tuple(CANONICAL_FEATURES)
    feature_vector = (0.55, 0.45, 0.2, 0.48, 0.8, 0.3, -0.1, 0.1, -0.05, 0.2, 0.1)

    # Test Long Barrier extraction
    res_long = explainer.explain(
        feature_vector=feature_vector,
        feature_names=feature_names,
        model=constrained_model,
        symbol="ETHUSDT",
        bar_time=_sample_time(),
        model_type="long_barrier",
    )
    assert res_long.model_type == "long_barrier"
    assert len(res_long.contributions) == 11

    # Test Short Barrier extraction
    res_short = explainer.explain(
        feature_vector=feature_vector,
        feature_names=feature_names,
        model=constrained_model,
        symbol="ETHUSDT",
        bar_time=_sample_time(),
        model_type="short_barrier",
    )
    assert res_short.model_type == "short_barrier"
    assert len(res_short.contributions) == 11


def test_explain_batch(
    treeshap_explainer: tuple[NativeTreeSHAPExplainer, TreeSHAPConfig],
    trained_models: tuple[lgb.Booster, ConstrainedLGBMModel],
) -> None:
    """Test vectorized batch explanation."""
    explainer, config = treeshap_explainer
    raw_booster, _ = trained_models

    n_samples = 20
    rng = np.random.default_rng(99)
    batch_features = rng.normal(0.0, 1.0, size=(n_samples, 11)).tolist()
    feature_names = tuple(CANONICAL_FEATURES)
    bar_times = [datetime(2026, 11, 8, 0, 0, tzinfo=UTC) for _ in range(n_samples)]

    results = explainer.explain_batch(
        batch_features=batch_features,
        feature_names=feature_names,
        model=raw_booster,
        symbol="SOLUSDT",
        bar_times=bar_times,
        model_type="long_barrier",
    )

    assert len(results) == n_samples
    for idx, r in enumerate(results):
        assert r.symbol == "SOLUSDT"
        assert r.bar_time == bar_times[idx]
        eff_diff = abs(r.total_feature_attribution + r.base_value - r.raw_margin)
        assert eff_diff <= config.local_efficiency_tol


def test_fail_fast_input_validation(
    treeshap_explainer: tuple[NativeTreeSHAPExplainer, TreeSHAPConfig],
    trained_models: tuple[lgb.Booster, ConstrainedLGBMModel],
) -> None:
    """Test fail-fast validation on dimension mismatch, NaNs, unknown types."""
    explainer, _ = treeshap_explainer
    raw_booster, _ = trained_models
    feature_names = tuple(CANONICAL_FEATURES)

    # 1. Feature dimension mismatch (10 features instead of 11)
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        explainer.explain(
            feature_vector=[0.0] * 10,
            feature_names=feature_names,
            model=raw_booster,
            symbol="BTCUSDT",
            bar_time=_sample_time(),
            model_type="long_barrier",
        )

    # 2. Unknown model type
    with pytest.raises(ValueError, match="Unknown model_type"):
        explainer.explain(
            feature_vector=[0.0] * 11,
            feature_names=feature_names,
            model=raw_booster,
            symbol="BTCUSDT",
            bar_time=_sample_time(),
            model_type="invalid_barrier",
        )

    # 3. Invalid model instance
    with pytest.raises(TypeError, match="Cannot extract"):
        explainer.explain(
            feature_vector=[0.0] * 11,
            feature_names=feature_names,
            model="not_a_model",
            symbol="BTCUSDT",
            bar_time=_sample_time(),
            model_type="long_barrier",
        )

    # 4. Non-finite values in sample
    with pytest.raises(ValueError, match="contains NaN or non-finite"):
        explainer.explain(
            feature_vector=[0.0] * 10 + [float("nan")],
            feature_names=feature_names,
            model=raw_booster,
            symbol="BTCUSDT",
            bar_time=_sample_time(),
            model_type="long_barrier",
        )

    # 5. Empty batch
    with pytest.raises(ValueError, match="batch_features cannot be empty"):
        explainer.explain_batch(
            batch_features=[],
            feature_names=feature_names,
            model=raw_booster,
            symbol="BTCUSDT",
            bar_times=[],
            model_type="long_barrier",
        )

    # 6. Batch length mismatch with bar_times
    with pytest.raises(ValueError, match="Batch length mismatch"):
        explainer.explain_batch(
            batch_features=[[0.0] * 11],
            feature_names=feature_names,
            model=raw_booster,
            symbol="BTCUSDT",
            bar_times=[_sample_time(), _sample_time()],
            model_type="long_barrier",
        )


def test_treeshap_latency_benchmark(
    treeshap_explainer: tuple[NativeTreeSHAPExplainer, TreeSHAPConfig],
    trained_models: tuple[lgb.Booster, ConstrainedLGBMModel],
) -> None:
    """Benchmark end-to-end single-sample explanation latency."""
    explainer, _ = treeshap_explainer
    raw_booster, _ = trained_models

    feature_names = tuple(CANONICAL_FEATURES)
    feature_vector = (0.7, 0.3, 1.1, 0.4, 1.8, 0.5, -0.2, 0.3, -0.1, 0.4, 0.2)
    sample_time = _sample_time()

    # Warm-up call
    _ = explainer.explain(
        feature_vector=feature_vector,
        feature_names=feature_names,
        model=raw_booster,
        symbol="BTCUSDT",
        bar_time=sample_time,
        model_type="long_barrier",
    )

    latencies_ms: list[float] = []
    n_runs = 50
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = explainer.explain(
            feature_vector=feature_vector,
            feature_names=feature_names,
            model=raw_booster,
            symbol="BTCUSDT",
            bar_time=sample_time,
            model_type="long_barrier",
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    median_latency = float(np.median(latencies_ms))
    # Median latency should easily be < 1.0ms SLA (typically ~0.2ms)
    assert median_latency < 1.0, f"Median latency {median_latency:.4f}ms exceeded 1.0ms SLA"
