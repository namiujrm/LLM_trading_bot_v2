"""Unit tests for EconomicSignVerifier, TreeSHAPConfig, and SHAP Entities (Milestone M4.5)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trading_bot.domain.entities.ml.ml_configs import (
    LGBMMonotonicConstraints,
    SHAPExplainerConfig,
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
    return datetime(2026, 11, 8, 12, 0, tzinfo=UTC)


def test_tree_shap_config_validation() -> None:
    """Test valid and invalid parameters for TreeSHAPConfig."""
    cfg = TreeSHAPConfig(
        top_k_bullish=5,
        top_k_bearish=3,
        local_efficiency_tol=1e-4,
        sign_tolerance=1e-4,
    )
    assert cfg.top_k_bullish == 5
    assert cfg.top_k_bearish == 3
    assert cfg.local_efficiency_tol == 1e-4
    assert cfg.feature_deadband_tol == 1e-4
    assert cfg.attribution_deadband_tol == 1e-4
    assert SHAPExplainerConfig is TreeSHAPConfig

    with pytest.raises(ValueError, match="top_k_bullish must be at least 1"):
        TreeSHAPConfig(
            top_k_bullish=0,
            top_k_bearish=3,
            local_efficiency_tol=1e-4,
            sign_tolerance=1e-4,
        )

    with pytest.raises(ValueError, match="top_k_bearish must be at least 1"):
        TreeSHAPConfig(
            top_k_bullish=5,
            top_k_bearish=0,
            local_efficiency_tol=1e-4,
            sign_tolerance=1e-4,
        )

    with pytest.raises(ValueError, match="local_efficiency_tol must be strictly positive"):
        TreeSHAPConfig(
            top_k_bullish=5,
            top_k_bearish=3,
            local_efficiency_tol=0.0,
            sign_tolerance=1e-4,
        )

    with pytest.raises(ValueError, match="sign_tolerance must be non-negative"):
        TreeSHAPConfig(
            top_k_bullish=5,
            top_k_bearish=3,
            local_efficiency_tol=1e-4,
            sign_tolerance=-0.01,
        )


def test_lgbm_monotonic_constraints_factories_and_validation() -> None:
    """Test canonical long and short monotonic constraints factories and invariants."""
    long_constraints = LGBMMonotonicConstraints.create_canonical_long()
    assert len(long_constraints.constraints) == 11
    assert len(long_constraints.neutral_baselines) == 11
    assert long_constraints.constraints["hurst_exponent_100"] == 1
    assert long_constraints.constraints["shannon_entropy_50"] == -1
    assert long_constraints.neutral_baselines["hurst_exponent_100"] == 0.50

    short_constraints = LGBMMonotonicConstraints.create_canonical_short()
    assert len(short_constraints.constraints) == 11
    assert short_constraints.constraints["taker_cvd_z_24"] == -1
    assert short_constraints.constraints["funding_crowded_pos_z"] == 1

    with pytest.raises(ValueError, match="constraints mapping must not be empty"):
        LGBMMonotonicConstraints(constraints={}, neutral_baselines={})

    with pytest.raises(ValueError, match="Direction for 'f1' must be -1, 0, or 1"):
        LGBMMonotonicConstraints(constraints={"f1": 2}, neutral_baselines={"f1": 0.0})

    with pytest.raises(ValueError, match="neutral_baselines missing constraints"):
        LGBMMonotonicConstraints(constraints={"f1": 1}, neutral_baselines={})


def test_shap_driver_contribution_properties_and_validation() -> None:
    """Test SHAPDriverContribution invariants and helper properties."""
    c_pos = SHAPDriverContribution(
        feature_name="taker_cvd_z_24",
        feature_value=1.5,
        shap_value=0.25,
        expected_sign=1,
        is_sign_consistent=True,
    )
    assert c_pos.is_bullish is True
    assert c_pos.is_bearish is False
    assert c_pos.abs_shap == 0.25

    c_neg = SHAPDriverContribution(
        feature_name="shannon_entropy_50",
        feature_value=0.8,
        shap_value=-0.15,
        expected_sign=-1,
        is_sign_consistent=True,
    )
    assert c_neg.is_bullish is False
    assert c_neg.is_bearish is True
    assert c_neg.abs_shap == 0.15

    with pytest.raises(ValueError, match="feature_name must not be empty"):
        SHAPDriverContribution(
            feature_name="   ",
            feature_value=1.0,
            shap_value=0.1,
            expected_sign=1,
            is_sign_consistent=True,
        )

    with pytest.raises(ValueError, match="expected_sign must be -1, 0, or 1"):
        SHAPDriverContribution(
            feature_name="test_feat",
            feature_value=1.0,
            shap_value=0.1,
            expected_sign=5,
            is_sign_consistent=True,
        )


def test_shap_explanation_result_validators_and_properties() -> None:
    """Test SHAPExplanationResult invariant validations and properties."""
    c1 = SHAPDriverContribution(
        feature_name="feat_a",
        feature_value=1.0,
        shap_value=0.3,
        expected_sign=1,
        is_sign_consistent=True,
    )
    c2 = SHAPDriverContribution(
        feature_name="feat_b",
        feature_value=-1.0,
        shap_value=-0.2,
        expected_sign=-1,
        is_sign_consistent=True,
    )

    base = 0.5
    raw = 0.3 - 0.2 + base  # 0.6
    res = SHAPExplanationResult(
        symbol="BTCUSDT",
        bar_time=_sample_time(),
        model_type="long_barrier",
        base_value=base,
        raw_margin=raw,
        contributions=(c1, c2),
        top_bullish_drivers=(c1,),
        top_bearish_drivers=(c2,),
        is_economic_consistent=True,
    )

    assert res.total_feature_attribution == pytest.approx(0.1)
    assert res.feature_contribution_map == {"feat_a": 0.3, "feat_b": -0.2}
    assert res.num_bullish_drivers == 1
    assert res.num_bearish_drivers == 1
    assert res.inversion_count == 0

    # Local efficiency violation > 1e-4
    with pytest.raises(ValueError, match="local efficiency violated"):
        SHAPExplanationResult(
            symbol="BTCUSDT",
            bar_time=_sample_time(),
            model_type="long_barrier",
            base_value=base,
            raw_margin=10.0,
            contributions=(c1, c2),
            top_bullish_drivers=(c1,),
            top_bearish_drivers=(c2,),
            is_economic_consistent=True,
        )

    # Bullish drivers must be sorted descending
    c1_smaller = SHAPDriverContribution(
        feature_name="feat_small",
        feature_value=1.0,
        shap_value=0.1,
        expected_sign=1,
        is_sign_consistent=True,
    )
    with pytest.raises(ValueError, match="top_bullish_drivers not sorted descending"):
        SHAPExplanationResult(
            symbol="BTCUSDT",
            bar_time=_sample_time(),
            model_type="long_barrier",
            base_value=base,
            raw_margin=base + 0.3 + 0.1,
            contributions=(c1_smaller, c1),
            top_bullish_drivers=(c1_smaller, c1),
            top_bearish_drivers=(),
            is_economic_consistent=True,
        )

    # Bearish drivers must be sorted ascending (most negative first)
    c2_more_neg = SHAPDriverContribution(
        feature_name="feat_more_neg",
        feature_value=-2.0,
        shap_value=-0.4,
        expected_sign=-1,
        is_sign_consistent=True,
    )
    with pytest.raises(ValueError, match="top_bearish_drivers not sorted ascending"):
        SHAPExplanationResult(
            symbol="BTCUSDT",
            bar_time=_sample_time(),
            model_type="long_barrier",
            base_value=base,
            raw_margin=base - 0.2 - 0.4,
            contributions=(c2, c2_more_neg),
            top_bullish_drivers=(),
            top_bearish_drivers=(c2, c2_more_neg),
            is_economic_consistent=True,
        )

    # Inconsistency flag mismatch
    with pytest.raises(ValueError, match=r"is_economic_consistent .* contradicts"):
        SHAPExplanationResult(
            symbol="BTCUSDT",
            bar_time=_sample_time(),
            model_type="long_barrier",
            base_value=base,
            raw_margin=raw,
            contributions=(c1, c2),
            top_bullish_drivers=(c1,),
            top_bearish_drivers=(c2,),
            is_economic_consistent=False,
        )


def test_verify_shap_sign_consistency_truth_table() -> None:
    """Test verify_shap_sign_consistency truth table across deviations and constraints."""
    # 1. Monotonic +1 (e.g. taker_cvd_z_24, baseline 0.0)
    # Case A: dev > deadband, shap > deadband -> consistent (+1, +1, True)
    exp, act, cons = verify_shap_sign_consistency(
        feature_value=1.0,
        shap_value=0.5,
        constraint_direction=1,
        baseline_value=0.0,
        feature_deadband_tol=1e-4,
        attribution_deadband_tol=1e-4,
    )
    assert exp == 1 and act == 1 and cons is True

    # Case B: dev > deadband, shap < -deadband -> INCONSISTENT (+1, -1, False)
    exp, act, cons = verify_shap_sign_consistency(
        feature_value=1.0,
        shap_value=-0.5,
        constraint_direction=1,
        baseline_value=0.0,
        feature_deadband_tol=1e-4,
        attribution_deadband_tol=1e-4,
    )
    assert exp == 1 and act == -1 and cons is False

    # Case C: dev < -deadband, shap < -deadband -> consistent (-1, -1, True)
    exp, act, cons = verify_shap_sign_consistency(
        feature_value=-1.0,
        shap_value=-0.5,
        constraint_direction=1,
        baseline_value=0.0,
        feature_deadband_tol=1e-4,
        attribution_deadband_tol=1e-4,
    )
    assert exp == -1 and act == -1 and cons is True

    # Case D: dev < -deadband, shap > deadband -> INCONSISTENT (-1, +1, False)
    exp, act, cons = verify_shap_sign_consistency(
        feature_value=-1.0,
        shap_value=0.5,
        constraint_direction=1,
        baseline_value=0.0,
        feature_deadband_tol=1e-4,
        attribution_deadband_tol=1e-4,
    )
    assert exp == -1 and act == 1 and cons is False

    # 2. Monotonic -1 (e.g. vpin_toxicity_pct, baseline 0.50)
    # High toxicity dev > 0 -> expected_sign = 1 * -1 = -1
    # If shap < 0 -> consistent (-1, -1, True)
    exp, act, cons = verify_shap_sign_consistency(
        feature_value=0.80,
        shap_value=-0.2,
        constraint_direction=-1,
        baseline_value=0.50,
        feature_deadband_tol=1e-4,
        attribution_deadband_tol=1e-4,
    )
    assert exp == -1 and act == -1 and cons is True

    # High toxicity with positive attribution -> INCONSISTENT (-1, +1, False)
    exp, act, cons = verify_shap_sign_consistency(
        feature_value=0.80,
        shap_value=0.2,
        constraint_direction=-1,
        baseline_value=0.50,
        feature_deadband_tol=1e-4,
        attribution_deadband_tol=1e-4,
    )
    assert exp == -1 and act == 1 and cons is False

    # 3. Deadband neutral handling
    # Feature within deadband -> expected = 0 -> always consistent
    exp, act, cons = verify_shap_sign_consistency(
        feature_value=0.50005,
        shap_value=0.3,
        constraint_direction=1,
        baseline_value=0.50,
        feature_deadband_tol=1e-4,
        attribution_deadband_tol=1e-4,
    )
    assert exp == 0 and act == 1 and cons is True

    # Attribution within deadband -> actual = 0 -> always consistent
    exp, act, cons = verify_shap_sign_consistency(
        feature_value=1.5,
        shap_value=0.00005,
        constraint_direction=1,
        baseline_value=0.0,
        feature_deadband_tol=1e-4,
        attribution_deadband_tol=1e-4,
    )
    assert exp == 1 and act == 0 and cons is True


def test_rank_top_drivers_and_tie_breaking() -> None:
    """Test rank_top_drivers ranking, top-k limits, deadband exclusion, and alphabetical tie-breaking."""
    c_tie1 = SHAPDriverContribution(
        feature_name="zeta_feature",
        feature_value=1.0,
        shap_value=0.20,
        expected_sign=1,
        is_sign_consistent=True,
    )
    c_tie2 = SHAPDriverContribution(
        feature_name="alpha_feature",
        feature_value=1.0,
        shap_value=0.20,
        expected_sign=1,
        is_sign_consistent=True,
    )
    c_high = SHAPDriverContribution(
        feature_name="best_feature",
        feature_value=1.0,
        shap_value=0.50,
        expected_sign=1,
        is_sign_consistent=True,
    )
    c_neutral = SHAPDriverContribution(
        feature_name="neutral_feature",
        feature_value=1.0,
        shap_value=0.00001,
        expected_sign=1,
        is_sign_consistent=True,
    )
    c_neg1 = SHAPDriverContribution(
        feature_name="bad_feature_b",
        feature_value=-1.0,
        shap_value=-0.30,
        expected_sign=-1,
        is_sign_consistent=True,
    )
    c_neg2 = SHAPDriverContribution(
        feature_name="bad_feature_a",
        feature_value=-1.0,
        shap_value=-0.30,
        expected_sign=-1,
        is_sign_consistent=True,
    )
    c_worst = SHAPDriverContribution(
        feature_name="worst_feature",
        feature_value=-1.0,
        shap_value=-0.60,
        expected_sign=-1,
        is_sign_consistent=True,
    )

    contributions = [c_tie1, c_tie2, c_high, c_neutral, c_neg1, c_neg2, c_worst]

    top_bull, top_bear = rank_top_drivers(
        contributions=contributions,
        top_k_bullish=5,
        top_k_bearish=3,
        attribution_deadband_tol=1e-4,
    )

    # Bullish: best_feature (0.50), then alphabetical tie: alpha_feature (0.20) before zeta_feature (0.20)
    # Neutral feature excluded (< 1e-4)
    assert len(top_bull) == 3
    assert top_bull[0].feature_name == "best_feature"
    assert top_bull[1].feature_name == "alpha_feature"
    assert top_bull[2].feature_name == "zeta_feature"

    # Bearish: worst_feature (-0.60), then alphabetical tie: bad_feature_a (-0.30) before bad_feature_b (-0.30)
    assert len(top_bear) == 3
    assert top_bear[0].feature_name == "worst_feature"
    assert top_bear[1].feature_name == "bad_feature_a"
    assert top_bear[2].feature_name == "bad_feature_b"


def test_economic_sign_verifier_full_flow() -> None:
    """Test EconomicSignVerifier verify_and_rank and create_explanation_result."""
    config = TreeSHAPConfig(
        top_k_bullish=5,
        top_k_bearish=3,
        local_efficiency_tol=1e-4,
        sign_tolerance=1e-4,
    )
    constraints = LGBMMonotonicConstraints.create_canonical_long()
    verifier = EconomicSignVerifier(config=config, monotonic_constraints=constraints)

    assert verifier.config is config
    assert verifier.constraints is constraints

    feature_names = (
        "hurst_exponent_100",
        "shannon_entropy_50",
        "fracdiff_momentum_z",
        "vpin_toxicity_pct",
        "taker_cvd_z_24",
        "ofi_imbalance_z",
        "kyle_lambda_log",
        "oi_directional_score",
        "funding_crowded_pos_z",
        "funding_crowded_neg_z",
        "btc_lead_catchup_gap",
    )
    # Features consistent with Long model
    feature_values = (0.65, 0.40, 1.2, 0.30, 1.5, 0.8, -0.5, 0.4, -0.2, 0.6, 0.5)
    # Attributions aligned with features
    shap_values = (0.10, 0.05, 0.15, 0.08, 0.20, 0.12, 0.06, 0.07, 0.04, 0.09, 0.05)

    contribs, top_bull, top_bear, all_consistent = verifier.verify_and_rank(
        feature_names=feature_names,
        feature_values=feature_values,
        shap_values=shap_values,
    )

    assert len(contribs) == 11
    assert all_consistent is True
    assert len(top_bull) == 5
    assert len(top_bear) == 0

    base_val = -0.05
    raw_margin = sum(shap_values) + base_val
    result = verifier.create_explanation_result(
        symbol="ETHUSDT",
        bar_time=_sample_time(),
        model_type="long_barrier",
        base_value=base_val,
        raw_margin=raw_margin,
        feature_names=feature_names,
        feature_values=feature_values,
        shap_values=shap_values,
    )
    assert isinstance(result, SHAPExplanationResult)
    assert result.symbol == "ETHUSDT"
    assert result.is_economic_consistent is True
    assert result.inversion_count == 0

    # Dimension mismatch
    with pytest.raises(ValueError, match="Dimension mismatch"):
        verifier.verify_and_rank(
            feature_names=("f1",),
            feature_values=(1.0, 2.0),
            shap_values=(0.1,),
        )
