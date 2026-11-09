# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Empirical Challenger 2 Adversarial Stress Suite (Milestone M4.5 — Proposal P6).

Adversarially tests:
1. Adversarial Sign Inversion & Fail-Fast Behavior:
   - Real-tree adversarial feature perturbations inducing sign inversions across all 4 sign permutations.
   - Assert EconomicSignVerifier and SHAPExplanationResult accurately flag is_sign_consistent=False
     and is_economic_consistent=False with exact inversion_count tracking.
   - Deadband boundary behavior for feature deviation and attribution magnitude.
2. Input Space Robustness & Degenerate Inputs:
   - Empty lists, NaN / Inf inputs, incorrect feature dimensions (10 instead of 11, 12 instead of 11).
   - Mismatched and permuted feature names.
   - Verification of typed exceptions (ValueError / TypeError) vs unhandled library crashes (LightGBMError).
3. Ranking Determinism & Boundary Conditions:
   - All 11 features positive (top_bearish == ()).
   - All 11 features negative (top_bullish == ()).
   - All 11 features within deadband (top_bullish == (), top_bearish == ()).
   - Exact numerical ties for bullish and bearish drivers (alphabetical ordering).
"""

from __future__ import annotations

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
from trading_bot.domain.entities.ml.ml_predictions import (
    SHAPDriverContribution,
    SHAPExplanationResult,
)
from trading_bot.domain.services.ml.economic_sign_verifier import (
    EconomicSignVerifier,
    rank_top_drivers,
    verify_shap_sign_consistency,
)


def _sample_time() -> datetime:
    return datetime(2026, 11, 8, 15, 0, tzinfo=UTC)


@pytest.fixture
def treeshap_env() -> tuple[
    NativeTreeSHAPExplainer, lgb.Booster, ConstrainedLGBMModel, TreeSHAPConfig
]:
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

    # Train a baseline raw booster with 11 features
    rng = np.random.default_rng(42)
    n_samples = 250
    x_train = rng.normal(0.0, 1.0, size=(n_samples, 11))
    y_train = (x_train[:, 0] > 0.0).astype(int)
    ds = lgb.Dataset(x_train, label=y_train, feature_name=list(CANONICAL_FEATURES))
    bst = lgb.train(
        {"objective": "binary", "max_depth": 4, "num_leaves": 15, "verbosity": -1},
        ds,
        num_boost_round=15,
    )

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
        num_boost_round=15,
        early_stopping_rounds=5,
        coherence_penalty_gamma=1.0,
    )
    ridge_config = RidgePriorConfig(
        ridge_c=1.0,
        ridge_max_iter=200,
        oof_splits=3,
        clip_eps=1e-6,
        embargo_bars_30m=5,
        max_label_holding_bars_30m=2,
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
    y_short = (x_train[:, 1] > 0.0).astype(int)
    constrained_model.fit(
        train_features=x_train.tolist(),
        y_long=y_train.tolist(),
        y_short=y_short.tolist(),
        sample_weights=[1.0] * n_samples,
        sample_intervals=[(i, i + 1) for i in range(n_samples)],
    )

    return explainer, bst, constrained_model, config


# ==============================================================================
# SECTION 1: Adversarial Sign Inversion & Fail-Fast Behavior
# ==============================================================================


def test_adversarial_sign_inversion_all_four_quadrants() -> None:
    """Test all four quadrants of economic sign inversion in pure domain math.

    Quadrant 1: M = +1, dev < 0 (expected -1), but phi > 0 (actual +1) -> INCONSISTENT
    Quadrant 2: M = +1, dev > 0 (expected +1), but phi < 0 (actual -1) -> INCONSISTENT
    Quadrant 3: M = -1, dev > 0 (expected -1), but phi > 0 (actual +1) -> INCONSISTENT
    Quadrant 4: M = -1, dev < 0 (expected +1), but phi < 0 (actual -1) -> INCONSISTENT
    """
    tol = 1e-4

    # Quadrant 1: Monotonic +1 (e.g. taker_cvd_z_24, baseline 0.0), dev < 0, phi > 0
    exp, act, is_cons = verify_shap_sign_consistency(
        feature_value=-1.5,
        shap_value=0.35,
        constraint_direction=1,
        baseline_value=0.0,
        feature_deadband_tol=tol,
        attribution_deadband_tol=tol,
    )
    assert exp == -1
    assert act == 1
    assert is_cons is False

    # Quadrant 2: Monotonic +1, dev > 0, phi < 0
    exp, act, is_cons = verify_shap_sign_consistency(
        feature_value=1.5,
        shap_value=-0.35,
        constraint_direction=1,
        baseline_value=0.0,
        feature_deadband_tol=tol,
        attribution_deadband_tol=tol,
    )
    assert exp == 1
    assert act == -1
    assert is_cons is False

    # Quadrant 3: Monotonic -1 (e.g. vpin_toxicity_pct, baseline 0.50), dev > 0 (high toxicity), phi > 0
    exp, act, is_cons = verify_shap_sign_consistency(
        feature_value=0.85,
        shap_value=0.20,
        constraint_direction=-1,
        baseline_value=0.50,
        feature_deadband_tol=tol,
        attribution_deadband_tol=tol,
    )
    assert exp == -1
    assert act == 1
    assert is_cons is False

    # Quadrant 4: Monotonic -1, dev < 0 (low toxicity), phi < 0
    exp, act, is_cons = verify_shap_sign_consistency(
        feature_value=0.15,
        shap_value=-0.20,
        constraint_direction=-1,
        baseline_value=0.50,
        feature_deadband_tol=tol,
        attribution_deadband_tol=tol,
    )
    assert exp == 1
    assert act == -1
    assert is_cons is False


def test_real_tree_background_shift_adversarial_sign_inversion(
    treeshap_env: tuple[NativeTreeSHAPExplainer, lgb.Booster, ConstrainedLGBMModel, TreeSHAPConfig],
) -> None:
    """Empirically induce sign inversion on a real LightGBM tree via background distribution shift.

    When training data has hurst centered at 0.10 (well below economic baseline 0.50),
    a test sample with hurst=0.25 is economically deficient (dev = 0.25 - 0.50 = -0.25 < 0,
    expected sign = -1 for M=+1), but relative to the training distribution it is above average,
    inducing a positive SHAP attribution (phi > 0).
    EconomicSignVerifier and SHAPExplanationResult MUST detect this inversion.
    """
    explainer, _, _, _ = treeshap_env

    rng = np.random.default_rng(2026)
    x_biased = rng.normal(0.0, 1.0, (300, 11))
    # Bias hurst to [0.0, 0.3] in training
    x_biased[:, 0] = rng.uniform(0.0, 0.3, size=300)
    y_biased = (x_biased[:, 0] > 0.15).astype(int)

    ds_biased = lgb.Dataset(x_biased, label=y_biased, feature_name=list(CANONICAL_FEATURES))
    monotone_long = [1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1]
    biased_booster = lgb.train(
        {"objective": "binary", "monotone_constraints": monotone_long, "verbosity": -1},
        ds_biased,
        num_boost_round=10,
    )

    # Test sample with hurst = 0.25 (below baseline 0.50, but above training mean 0.15)
    test_vector = [0.25] + [0.0] * 10

    result = explainer.explain(
        feature_vector=test_vector,
        feature_names=CANONICAL_FEATURES,
        model=biased_booster,
        symbol="BTCUSDT",
        bar_time=_sample_time(),
        model_type="long_barrier",
    )

    assert isinstance(result, SHAPExplanationResult)
    # At least hurst must be inverted
    hurst_contrib = next(c for c in result.contributions if c.feature_name == "hurst_exponent_100")
    assert hurst_contrib.feature_value == 0.25
    assert hurst_contrib.expected_sign == -1
    assert hurst_contrib.shap_value > 0.0
    assert hurst_contrib.is_sign_consistent is False

    # Result-level invariant: is_economic_consistent MUST be False
    assert result.is_economic_consistent is False
    assert result.inversion_count >= 1


def test_adversarial_deadband_neutral_invariants() -> None:
    """Stress-test deadband boundary conditions where no false inversions must trigger."""
    tol = 1e-4

    # 1. Feature exactly on deadband boundary (dev == tol)
    _, _, is_cons = verify_shap_sign_consistency(
        feature_value=0.50 + tol,
        shap_value=-0.5,
        constraint_direction=1,
        baseline_value=0.50,
        feature_deadband_tol=tol,
        attribution_deadband_tol=tol,
    )
    # dev is NOT > tol, so d_j == 0, expected == 0 -> consistent
    assert is_cons is True

    # 2. Feature just inside deadband
    _, _, is_cons = verify_shap_sign_consistency(
        feature_value=0.50 + 0.5 * tol,
        shap_value=-0.5,
        constraint_direction=1,
        baseline_value=0.50,
        feature_deadband_tol=tol,
        attribution_deadband_tol=tol,
    )
    assert is_cons is True

    # 3. Attribution exactly on deadband boundary (shap == tol)
    _, _, is_cons = verify_shap_sign_consistency(
        feature_value=1.0,
        shap_value=tol,
        constraint_direction=1,
        baseline_value=0.0,
        feature_deadband_tol=tol,
        attribution_deadband_tol=tol,
    )
    # shap is NOT > tol, so actual == 0 -> consistent
    assert is_cons is True

    # 4. Attribution just inside deadband
    _, _, is_cons = verify_shap_sign_consistency(
        feature_value=-1.0,
        shap_value=0.5 * tol,
        constraint_direction=1,
        baseline_value=0.0,
        feature_deadband_tol=tol,
        attribution_deadband_tol=tol,
    )
    assert is_cons is True


# ==============================================================================
# SECTION 2: Ranking Determinism & Boundary Conditions
# ==============================================================================


def test_ranking_boundary_all_eleven_features_positive() -> None:
    """Test boundary where all 11 features have positive attributions.

    top_bearish MUST be empty tuple (), top_bullish MUST contain top 5.
    """
    config = TreeSHAPConfig(
        top_k_bullish=5, top_k_bearish=3, local_efficiency_tol=1e-4, sign_tolerance=1e-4
    )
    long_c = LGBMMonotonicConstraints.create_canonical_long()
    verifier = EconomicSignVerifier(config=config, monotonic_constraints=long_c)

    names = tuple(long_c.constraints.keys())
    values = (0.7, 0.3, 1.0, 0.3, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0)
    shaps = tuple(0.10 + 0.05 * i for i in range(11))

    contribs, top_bull, top_bear, _ = verifier.verify_and_rank(names, values, shaps)

    assert len(contribs) == 11
    assert len(top_bull) == 5
    assert len(top_bear) == 0
    assert top_bear == ()

    # Assert descending order of top_bull
    for i in range(1, len(top_bull)):
        assert top_bull[i].shap_value <= top_bull[i - 1].shap_value


def test_ranking_boundary_all_eleven_features_negative() -> None:
    """Test boundary where all 11 features have negative attributions.

    top_bullish MUST be empty tuple (), top_bearish MUST contain top 3 (most negative first).
    """
    config = TreeSHAPConfig(
        top_k_bullish=5, top_k_bearish=3, local_efficiency_tol=1e-4, sign_tolerance=1e-4
    )
    long_c = LGBMMonotonicConstraints.create_canonical_long()
    verifier = EconomicSignVerifier(config=config, monotonic_constraints=long_c)

    names = tuple(long_c.constraints.keys())
    values = (0.3, 0.7, -1.0, 0.7, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0)
    # Shaps from -0.10 to -0.60
    shaps = tuple(-0.10 - 0.05 * i for i in range(11))

    contribs, top_bull, top_bear, _ = verifier.verify_and_rank(names, values, shaps)

    assert len(contribs) == 11
    assert len(top_bull) == 0
    assert top_bull == ()
    assert len(top_bear) == 3

    # Assert ascending order of top_bear (most negative first)
    for i in range(1, len(top_bear)):
        assert top_bear[i].shap_value >= top_bear[i - 1].shap_value


def test_ranking_boundary_all_features_in_deadband() -> None:
    """Test boundary where all 11 features have negligible attributions (|phi| <= deadband).

    Both top_bullish and top_bearish MUST be empty tuples ().
    """
    config = TreeSHAPConfig(
        top_k_bullish=5, top_k_bearish=3, local_efficiency_tol=1e-4, sign_tolerance=1e-4
    )
    long_c = LGBMMonotonicConstraints.create_canonical_long()
    verifier = EconomicSignVerifier(config=config, monotonic_constraints=long_c)

    names = tuple(long_c.constraints.keys())
    values = tuple(0.0 for _ in range(11))
    shaps = tuple(1e-5 for _ in range(11))

    contribs, top_bull, top_bear, _ = verifier.verify_and_rank(names, values, shaps)

    assert len(contribs) == 11
    assert top_bull == ()
    assert top_bear == ()


def test_ranking_exact_numerical_ties_alphabetical_order() -> None:
    """Test deterministic alphabetical tie-breaking on exact numerical ties."""
    contributions = [
        SHAPDriverContribution(
            feature_name="zeta_feat",
            feature_value=1.0,
            shap_value=0.25,
            expected_sign=1,
            is_sign_consistent=True,
        ),
        SHAPDriverContribution(
            feature_name="alpha_feat",
            feature_value=1.0,
            shap_value=0.25,
            expected_sign=1,
            is_sign_consistent=True,
        ),
        SHAPDriverContribution(
            feature_name="beta_feat",
            feature_value=1.0,
            shap_value=0.25,
            expected_sign=1,
            is_sign_consistent=True,
        ),
        SHAPDriverContribution(
            feature_name="omega_bear",
            feature_value=-1.0,
            shap_value=-0.35,
            expected_sign=-1,
            is_sign_consistent=True,
        ),
        SHAPDriverContribution(
            feature_name="charlie_bear",
            feature_value=-1.0,
            shap_value=-0.35,
            expected_sign=-1,
            is_sign_consistent=True,
        ),
    ]

    top_bull, top_bear = rank_top_drivers(
        contributions=contributions,
        top_k_bullish=5,
        top_k_bearish=3,
        attribution_deadband_tol=1e-4,
    )

    # Bullish ties: alpha_feat, beta_feat, zeta_feat
    assert [c.feature_name for c in top_bull] == ["alpha_feat", "beta_feat", "zeta_feat"]

    # Bearish ties: charlie_bear, omega_bear
    assert [c.feature_name for c in top_bear] == ["charlie_bear", "omega_bear"]


# ==============================================================================
# SECTION 3: Input Space Robustness & Degenerate Inputs (Empirical Challenge)
# ==============================================================================


def test_degenerate_nan_and_inf_inputs(
    treeshap_env: tuple[NativeTreeSHAPExplainer, lgb.Booster, ConstrainedLGBMModel, TreeSHAPConfig],
) -> None:
    """Assert explicit ValueError on NaN / Inf in explain and explain_batch."""
    explainer, bst, _, _ = treeshap_env
    names = tuple(CANONICAL_FEATURES)

    # NaN in single vector
    with pytest.raises(ValueError, match="NaN or non-finite"):
        explainer.explain(
            feature_vector=[float("nan")] + [0.0] * 10,
            feature_names=names,
            model=bst,
            symbol="BTCUSDT",
            bar_time=_sample_time(),
            model_type="long_barrier",
        )

    # Inf in single vector
    with pytest.raises(ValueError, match="NaN or non-finite"):
        explainer.explain(
            feature_vector=[float("inf")] + [0.0] * 10,
            feature_names=names,
            model=bst,
            symbol="BTCUSDT",
            bar_time=_sample_time(),
            model_type="long_barrier",
        )

    # NaN in batch
    with pytest.raises(ValueError, match="NaN or non-finite"):
        explainer.explain_batch(
            batch_features=[[float("nan")] + [0.0] * 10],
            feature_names=names,
            model=bst,
            symbol="BTCUSDT",
            bar_times=[_sample_time()],
            model_type="long_barrier",
        )


def test_degenerate_unknown_model_type_and_invalid_booster(
    treeshap_env: tuple[NativeTreeSHAPExplainer, lgb.Booster, ConstrainedLGBMModel, TreeSHAPConfig],
) -> None:
    """Assert explicit ValueError on unknown model_type and TypeError on invalid model."""
    explainer, bst, _, _ = treeshap_env
    names = tuple(CANONICAL_FEATURES)

    with pytest.raises(ValueError, match="Unknown model_type"):
        explainer.explain(
            feature_vector=[0.0] * 11,
            feature_names=names,
            model=bst,
            symbol="BTCUSDT",
            bar_time=_sample_time(),
            model_type="invalid_barrier",
        )

    with pytest.raises(TypeError, match="Cannot extract"):
        explainer.explain(
            feature_vector=[0.0] * 11,
            feature_names=names,
            model={"not": "a_booster"},
            symbol="BTCUSDT",
            bar_time=_sample_time(),
            model_type="long_barrier",
        )


def test_batch_length_mismatch_and_empty_batch(
    treeshap_env: tuple[NativeTreeSHAPExplainer, lgb.Booster, ConstrainedLGBMModel, TreeSHAPConfig],
) -> None:
    """Assert explicit ValueError on empty batch and bar_times length mismatch."""
    explainer, bst, _, _ = treeshap_env
    names = tuple(CANONICAL_FEATURES)

    with pytest.raises(ValueError, match="batch_features cannot be empty"):
        explainer.explain_batch(
            batch_features=[],
            feature_names=names,
            model=bst,
            symbol="BTCUSDT",
            bar_times=[],
            model_type="long_barrier",
        )

    with pytest.raises(ValueError, match="Batch length mismatch"):
        explainer.explain_batch(
            batch_features=[[0.0] * 11],
            feature_names=names,
            model=bst,
            symbol="BTCUSDT",
            bar_times=[_sample_time(), _sample_time()],
            model_type="long_barrier",
        )


def test_degenerate_empty_feature_vector_raises_value_error(
    treeshap_env: tuple[NativeTreeSHAPExplainer, lgb.Booster, ConstrainedLGBMModel, TreeSHAPConfig],
) -> None:
    """Challenge explainer with empty feature vector and names. Must raise explicit ValueError (never LightGBMError)."""
    explainer, bst, _, _ = treeshap_env

    with pytest.raises(ValueError):
        explainer.explain(
            feature_vector=[],
            feature_names=(),
            model=bst,
            symbol="BTCUSDT",
            bar_time=_sample_time(),
            model_type="long_barrier",
        )


def test_degenerate_incorrect_feature_dimensions_10_and_12(
    treeshap_env: tuple[NativeTreeSHAPExplainer, lgb.Booster, ConstrainedLGBMModel, TreeSHAPConfig],
) -> None:
    """Challenge explainer when both vector and names are 10 or 12 (mismatching model's 11 features).

    Must raise explicit ValueError before passing to LightGBM C++ (never fatal LightGBMError).
    """
    explainer, bst, _, _ = treeshap_env

    # 10 features instead of 11
    with pytest.raises(ValueError, match=r"(?i)dimension|expected 11"):
        explainer.explain(
            feature_vector=[0.0] * 10,
            feature_names=tuple(CANONICAL_FEATURES[:10]),
            model=bst,
            symbol="BTCUSDT",
            bar_time=_sample_time(),
            model_type="long_barrier",
        )

    # 12 features instead of 11
    names_12 = (*CANONICAL_FEATURES, "extra_feat")
    with pytest.raises(ValueError, match=r"(?i)dimension|expected 11"):
        explainer.explain(
            feature_vector=[0.0] * 12,
            feature_names=names_12,
            model=bst,
            symbol="BTCUSDT",
            bar_time=_sample_time(),
            model_type="long_barrier",
        )


def test_degenerate_mismatched_feature_names_fail_fast(
    treeshap_env: tuple[NativeTreeSHAPExplainer, lgb.Booster, ConstrainedLGBMModel, TreeSHAPConfig],
) -> None:
    """Challenge explainer with mismatched feature names.

    Must NOT silently fallback to unconstrained direction=0 and baseline=0.0.
    Must raise explicit ValueError.
    """
    explainer, bst, _, _ = treeshap_env
    bogus_names = tuple(f"bogus_feature_{i}" for i in range(11))

    with pytest.raises(ValueError, match=r"(?i)unknown|mismatch|unrecognized"):
        explainer.explain(
            feature_vector=[0.0] * 11,
            feature_names=bogus_names,
            model=bst,
            symbol="BTCUSDT",
            bar_time=_sample_time(),
            model_type="long_barrier",
        )


def test_economic_sign_verifier_rejects_unknown_feature_names() -> None:
    """Challenge EconomicSignVerifier with unknown feature names.

    Must NOT silently fall back to direction=0 and baseline=0.0.
    Must raise explicit ValueError (Fail-Fast Discipline under Invariant #9).
    """
    config = TreeSHAPConfig(
        top_k_bullish=5, top_k_bearish=3, local_efficiency_tol=1e-4, sign_tolerance=1e-4
    )
    long_c = LGBMMonotonicConstraints.create_canonical_long()
    verifier = EconomicSignVerifier(config=config, monotonic_constraints=long_c)

    bogus_names = tuple(f"unrecognized_feature_{i}" for i in range(11))
    values = tuple(0.0 for _ in range(11))
    shaps = tuple(0.1 for _ in range(11))

    with pytest.raises(ValueError, match=r"(?i)unknown|unrecognized|not found"):
        verifier.verify_and_rank(bogus_names, values, shaps)


def test_degenerate_batch_incorrect_dimensions_and_mismatched_names(
    treeshap_env: tuple[NativeTreeSHAPExplainer, lgb.Booster, ConstrainedLGBMModel, TreeSHAPConfig],
) -> None:
    """Challenge explain_batch with incorrect dimensions (10 or 12) and mismatched names.

    Must raise explicit ValueError before passing to LightGBM C++ and reject mismatched names.
    """
    explainer, bst, _, _ = treeshap_env
    times = [_sample_time()]

    # 1. 10 features in batch
    with pytest.raises(ValueError, match=r"(?i)dimension|expected 11"):
        explainer.explain_batch(
            batch_features=[[0.0] * 10],
            feature_names=tuple(CANONICAL_FEATURES[:10]),
            model=bst,
            symbol="BTCUSDT",
            bar_times=times,
            model_type="long_barrier",
        )

    # 2. 12 features in batch
    names_12 = (*CANONICAL_FEATURES, "extra_feat")
    with pytest.raises(ValueError, match=r"(?i)dimension|expected 11"):
        explainer.explain_batch(
            batch_features=[[0.0] * 12],
            feature_names=names_12,
            model=bst,
            symbol="BTCUSDT",
            bar_times=times,
            model_type="long_barrier",
        )

    # 3. Mismatched names in batch
    bogus_names = tuple(f"bogus_feature_{i}" for i in range(11))
    with pytest.raises(ValueError, match=r"(?i)unknown|mismatch|unrecognized"):
        explainer.explain_batch(
            batch_features=[[0.0] * 11],
            feature_names=bogus_names,
            model=bst,
            symbol="BTCUSDT",
            bar_times=times,
            model_type="long_barrier",
        )
