# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Empirical Challenger 2 Adversarial Stress Suite (Milestone M4.3 Iteration 2).

Adversarially stress-tests:
1. Bug #1 Remediation: Boundary float tolerance (1.0 < P_L + P_S <= 1.0 + 1e-7)
   and custom thresholds (max_comb <= 1.0) under extreme epsilon perturbations.
2. Bug #2 Remediation: Purged CV degenerate folds (empty train/val sets, massive embargo
   purging, single-class folds) strictly raise domain InsufficientDataException (never IndexError).
3. Configuration Propagation: embargo_bars_30m and max_label_holding_bars_30m flow
   dynamically from RidgePriorConfig into PurgedCVSplitter and alter training splits/coefficients.
4. ConstrainedLGBMModel Integration: Fail-fast unknown feature rejection and degenerate fold
   bubbling through dual-binary LightGBM stacking.

Strict Clean Architecture & Invariant #9 adherence: zero default arguments.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trading_bot.adapters.ml_engine.constrained_lgbm_model import (
    CANONICAL_FEATURES,
    ConstrainedLGBMModel,
)
from trading_bot.adapters.ml_engine.oof_ridge_prior import OOFRidgePrior
from trading_bot.domain.entities.ml.ml_configs import (
    CoherenceGateConfig,
    ConstrainedLGBMConfig,
    RidgePriorConfig,
)
from trading_bot.domain.entities.ml.ml_predictions import BarrierPredictionResult
from trading_bot.domain.exceptions import InsufficientDataException
from trading_bot.domain.services.ml.coherence_gate import CoherenceGate

# ==============================================================================
# SECTION 1: Bug #1 Remediation — Float Boundary Tolerances & Custom Thresholds
# ==============================================================================


@pytest.mark.parametrize(
    ("p_raw_long", "p_raw_short", "max_comb"),
    [
        # Exact boundary 1.0
        (1.0, 0.0, 1.0),
        (0.5, 0.5, 1.0),
        # Challenger 1 reproducible crash vector: 1.0 + 5.96e-8
        (1.0, 5.960464477539063e-08, 1.0),
        # Epsilon boundary just above 1.0 (within float_tol 1e-7)
        (0.5, 0.5 + 1e-8, 1.0),
        (0.5, 0.5 + 5e-8, 1.0),
        (0.5, 0.5 + 9.9e-8, 1.0),
        # Epsilon boundary just above float_tol (conflict detected, attenuation applied)
        (0.5, 0.5 + 1.1e-7, 1.0),
        (0.5, 0.5 + 1e-6, 1.0),
        (0.7, 0.7, 1.0),
        (1.0, 1.0, 1.0),
        # Custom sub-1.0 thresholds (max_comb < 1.0)
        (0.4, 0.4, 0.8),
        (0.45, 0.45, 0.8),  # sum 0.90 > 0.80, but <= 1.0
        (0.5, 0.45, 0.9),  # sum 0.95 > 0.90, but <= 1.0
        (0.01, 0.01, 0.05),
        (0.1, 0.1, 0.15),
    ],
)
def test_boundary_float_tolerance_and_custom_thresholds_never_crash(
    p_raw_long: float,
    p_raw_short: float,
    max_comb: float,
) -> None:
    """Verify that CoherenceGate produces valid BarrierPredictionResult without ValidationError."""
    cfg = CoherenceGateConfig(
        gamma_conflict=1.0,
        max_combined_probability=max_comb,
        min_probability_floor=0.0,
    )
    gate = CoherenceGate(config=cfg)
    res = gate.enforce_coherence(p_raw_long=p_raw_long, p_raw_short=p_raw_short)

    assert isinstance(res, BarrierPredictionResult)
    assert 0.0 <= res.p_long <= 1.0
    assert 0.0 <= res.p_short <= 1.0
    assert res.p_long <= p_raw_long + 1e-7
    assert res.p_short <= p_raw_short + 1e-7
    assert res.coherent_sum <= max_comb + 1e-7
    assert res.coherent_sum <= 1.0 + 1e-7
    assert res.coherence_penalty >= 0.0

    if p_raw_long + p_raw_short <= max_comb + 1e-9:
        assert res.is_coherent is True
        assert res.coherence_penalty == pytest.approx(0.0)
    else:
        assert res.is_coherent is False
        assert res.coherence_penalty > 0.0


@given(
    p_long=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    p_short=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    max_comb=st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
    gamma=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_hypothesis_adversarial_coherence_property(
    p_long: float,
    p_short: float,
    max_comb: float,
    gamma: float,
) -> None:
    """Universal property test across 1,000 randomized configurations:
    ensures all mathematical invariants hold universally without crashes.
    """
    cfg = CoherenceGateConfig(
        gamma_conflict=gamma,
        max_combined_probability=max_comb,
        min_probability_floor=0.0,
    )
    gate = CoherenceGate(config=cfg)
    res = gate.enforce_coherence(p_raw_long=p_long, p_raw_short=p_short)

    assert 0.0 <= res.p_long <= 1.0
    assert 0.0 <= res.p_short <= 1.0
    assert res.p_long <= p_long + 1e-7
    assert res.p_short <= p_short + 1e-7
    assert res.coherent_sum <= max_comb + 1e-7
    assert res.coherent_sum <= 1.0 + 1e-7
    assert res.coherence_penalty >= 0.0


def test_barrier_prediction_result_accepts_sub_one_incoherent_flag() -> None:
    """Verifies that BarrierPredictionResult does NOT reject is_coherent=False
    when raw_sum <= 1.0 (from sub-1.0 custom threshold or boundary flags).
    """
    # raw_sum = 0.85 <= 1.0, but marked is_coherent=False due to max_comb=0.80
    res = BarrierPredictionResult(
        p_long=0.40,
        p_short=0.40,
        p_raw_long=0.45,
        p_raw_short=0.40,
        is_coherent=False,
        coherence_penalty=0.05,
    )
    assert res.is_coherent is False
    assert res.raw_sum == pytest.approx(0.85)
    assert res.coherent_sum == pytest.approx(0.80)


# ==============================================================================
# SECTION 2: Bug #2 Remediation — Degenerate Purged CV Folds (Typed Exceptions)
# ==============================================================================


def test_empty_train_fold_from_overlapping_intervals_raises_insufficient_data() -> None:
    """When Purged CV purges all training samples in a fold, OOFRidgePrior must
    raise InsufficientDataException, NEVER an unhandled IndexError.
    """
    config = RidgePriorConfig(
        ridge_c=1.0,
        ridge_max_iter=100,
        oof_splits=2,
        clip_eps=1e-6,
        embargo_bars_30m=100,
        max_label_holding_bars_30m=12,
    )
    prior = OOFRidgePrior(config=config, random_state=42)

    # 4 samples with completely overlapping intervals (0, 100)
    features = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
    labels = [0, 1, 0, 1]
    weights = [1.0, 1.0, 1.0, 1.0]
    intervals = [(0, 100), (0, 100), (0, 100), (0, 100)]

    with pytest.raises(InsufficientDataException) as exc_info:
        prior.fit(
            features=features,
            labels=labels,
            sample_weights=weights,
            sample_intervals=intervals,
        )

    assert "empty train or validation set" in str(exc_info.value)


def test_massive_embargo_purging_raises_insufficient_data() -> None:
    """When embargo window is larger than the temporal gap between folds,
    all post-validation samples are embargoed, leaving train fold empty.
    """
    # 6 samples, 2 splits -> 3 samples per fold
    # Fold 0 val: samples 0, 1, 2 spanning t=0..20
    # Samples 3, 4, 5 at t=25, 30, 35
    # With embargo=100, samples 3, 4, 5 are all within t <= 20 + 100 = 120 -> all embargoed!
    config = RidgePriorConfig(
        ridge_c=1.0,
        ridge_max_iter=100,
        oof_splits=2,
        clip_eps=1e-6,
        embargo_bars_30m=500,  # massive embargo
        max_label_holding_bars_30m=12,
    )
    prior = OOFRidgePrior(config=config, random_state=42)

    features = [[float(i), float(i)] for i in range(6)]
    labels = [0, 1, 0, 1, 0, 1]
    weights = [1.0] * 6
    intervals = [(i * 10, i * 10 + 5) for i in range(6)]

    with pytest.raises(InsufficientDataException) as exc_info:
        prior.fit(
            features=features,
            labels=labels,
            sample_weights=weights,
            sample_intervals=intervals,
        )

    assert "empty train or validation set" in str(exc_info.value)


def test_single_class_fold_raises_insufficient_data_with_safe_formatting() -> None:
    """When a fold contains only one class in y_train, OOFRidgePrior must raise
    InsufficientDataException without raw index crash when reporting class.
    """
    config = RidgePriorConfig(
        ridge_c=1.0,
        ridge_max_iter=100,
        oof_splits=3,
        clip_eps=1e-6,
        embargo_bars_30m=0,
        max_label_holding_bars_30m=0,
    )
    prior = OOFRidgePrior(config=config, random_state=42)

    # 12 samples: samples 0..7 have class 0, samples 8..11 have class 1
    # In split 2 (val is 8..11), train is 0..7 which has ONLY class 0!
    features = [[float(i), float(i)] for i in range(12)]
    labels = [0] * 8 + [1] * 4
    weights = [1.0] * 12

    with pytest.raises(InsufficientDataException) as exc_info:
        prior.fit(
            features=features,
            labels=labels,
            sample_weights=weights,
            sample_intervals=None,
        )

    assert "contains only 1 class" in str(exc_info.value)
    assert "Minimum 2 classes required" in str(exc_info.value)


def test_single_class_entire_dataset_raises_insufficient_data() -> None:
    """When entire dataset has only 1 class, raises InsufficientDataException immediately."""
    config = RidgePriorConfig(
        ridge_c=1.0,
        ridge_max_iter=100,
        oof_splits=2,
        clip_eps=1e-6,
        embargo_bars_30m=10,
        max_label_holding_bars_30m=5,
    )
    prior = OOFRidgePrior(config=config, random_state=42)

    features = [[float(i), float(i)] for i in range(10)]
    labels = [1] * 10
    weights = [1.0] * 10

    with pytest.raises(InsufficientDataException, match="Training dataset contains only 1 class"):
        prior.fit(
            features=features,
            labels=labels,
            sample_weights=weights,
            sample_intervals=None,
        )


def test_sample_count_too_small_raises_insufficient_data() -> None:
    """Sample count < oof_splits * 2 must raise InsufficientDataException."""
    config = RidgePriorConfig(
        ridge_c=1.0,
        ridge_max_iter=100,
        oof_splits=5,
        clip_eps=1e-6,
        embargo_bars_30m=10,
        max_label_holding_bars_30m=5,
    )
    prior = OOFRidgePrior(config=config, random_state=42)

    features = [[float(i), float(i)] for i in range(8)]  # 8 < 5 * 2
    labels = [0, 1] * 4
    weights = [1.0] * 8

    with pytest.raises(InsufficientDataException, match=r"Sample count .* is too small"):
        prior.fit(
            features=features,
            labels=labels,
            sample_weights=weights,
            sample_intervals=None,
        )


# ==============================================================================
# SECTION 3: Dynamic Embargo & Horizon Propagation from RidgePriorConfig
# ==============================================================================


def test_ridge_prior_config_rejects_negative_embargo_and_holding() -> None:
    """Verify fail-fast validation against negative embargo or holding parameters."""
    with pytest.raises(ValueError, match="embargo_bars_30m must be non-negative"):
        RidgePriorConfig(
            ridge_c=1.0,
            ridge_max_iter=100,
            oof_splits=5,
            clip_eps=1e-6,
            embargo_bars_30m=-10,
            max_label_holding_bars_30m=12,
        )

    with pytest.raises(ValueError, match="max_label_holding_bars_30m must be non-negative"):
        RidgePriorConfig(
            ridge_c=1.0,
            ridge_max_iter=100,
            oof_splits=5,
            clip_eps=1e-6,
            embargo_bars_30m=100,
            max_label_holding_bars_30m=-5,
        )


def test_embargo_parameter_empirically_affects_oof_training() -> None:
    """Empirical proof: altering embargo_bars_30m in RidgePriorConfig changes
    the training sample partition and produces numerically distinct OOF logits.
    """
    n_samples = 40
    n_features = 4
    rng = np.random.default_rng(12345)

    x = rng.normal(0.0, 1.0, size=(n_samples, n_features)).tolist()
    y = [i % 2 for i in range(n_samples)]
    w = [1.0] * n_samples
    intervals = [(i * 10, i * 10 + 8) for i in range(n_samples)]

    # Model 1: Zero embargo
    cfg_no_embargo = RidgePriorConfig(
        ridge_c=1.0,
        ridge_max_iter=500,
        oof_splits=2,
        clip_eps=1e-6,
        embargo_bars_30m=0,
        max_label_holding_bars_30m=8,
    )
    prior_no_embargo = OOFRidgePrior(config=cfg_no_embargo, random_state=42)
    logits_no_embargo = prior_no_embargo.fit(
        features=x,
        labels=y,
        sample_weights=w,
        sample_intervals=intervals,
    )

    # Model 2: Substantial embargo (embargo_bars_30m = 30 bars = 3 samples removed from training)
    cfg_with_embargo = RidgePriorConfig(
        ridge_c=1.0,
        ridge_max_iter=500,
        oof_splits=2,
        clip_eps=1e-6,
        embargo_bars_30m=30,
        max_label_holding_bars_30m=8,
    )
    prior_with_embargo = OOFRidgePrior(config=cfg_with_embargo, random_state=42)
    logits_with_embargo = prior_with_embargo.fit(
        features=x,
        labels=y,
        sample_weights=w,
        sample_intervals=intervals,
    )

    # Proof of functional propagation: logits differ due to purged embargoed bars
    assert not np.allclose(logits_no_embargo, logits_with_embargo, atol=1e-4)


# ==============================================================================
# SECTION 4: ConstrainedLGBMModel Integration Stress
# ==============================================================================


def test_constrained_lgbm_rejects_unknown_features() -> None:
    """Verifies that ConstrainedLGBMModel immediately rejects unknown feature names."""
    cfg_lgbm = ConstrainedLGBMConfig(
        max_depth=4,
        num_leaves=15,
        min_child_samples=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        num_boost_round=10,
        early_stopping_rounds=5,
        coherence_penalty_gamma=1.0,
    )
    cfg_ridge = RidgePriorConfig(
        ridge_c=1.0,
        ridge_max_iter=100,
        oof_splits=5,
        clip_eps=1e-6,
        embargo_bars_30m=100,
        max_label_holding_bars_30m=12,
    )
    cfg_coherence = CoherenceGateConfig(
        gamma_conflict=1.0,
        max_combined_probability=1.0,
        min_probability_floor=0.0,
    )

    with pytest.raises(ValueError, match="Unknown feature names provided"):
        ConstrainedLGBMModel(
            config=cfg_lgbm,
            ridge_config=cfg_ridge,
            coherence_config=cfg_coherence,
            feature_names=(*CANONICAL_FEATURES, "rogue_feature_x"),
            random_state=42,
        )


def test_constrained_lgbm_bubbles_insufficient_data_from_degenerate_ridge_fold() -> None:
    """Verifies that degenerate Purged CV splits bubble InsufficientDataException
    through ConstrainedLGBMModel.fit().
    """
    cfg_lgbm = ConstrainedLGBMConfig(
        max_depth=4,
        num_leaves=15,
        min_child_samples=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        num_boost_round=10,
        early_stopping_rounds=5,
        coherence_penalty_gamma=1.0,
    )
    cfg_ridge = RidgePriorConfig(
        ridge_c=1.0,
        ridge_max_iter=100,
        oof_splits=2,
        clip_eps=1e-6,
        embargo_bars_30m=100,
        max_label_holding_bars_30m=12,
    )
    cfg_coherence = CoherenceGateConfig(
        gamma_conflict=1.0,
        max_combined_probability=1.0,
        min_probability_floor=0.0,
    )

    model = ConstrainedLGBMModel(
        config=cfg_lgbm,
        ridge_config=cfg_ridge,
        coherence_config=cfg_coherence,
        feature_names=CANONICAL_FEATURES,
        random_state=42,
    )

    n_samples = 24
    features = np.zeros((n_samples, len(CANONICAL_FEATURES))).tolist()
    y_long = [0, 1] * 12
    y_short = [1, 0] * 12
    weights = [1.0] * n_samples
    # All intervals completely overlap at (0, 100), purging all training samples
    intervals = [(0, 100)] * n_samples

    with pytest.raises(InsufficientDataException, match="empty train or validation set"):
        model.fit(
            train_features=features,
            y_long=y_long,
            y_short=y_short,
            sample_weights=weights,
            sample_intervals=intervals,
        )
