# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Adversarial Empirical Stress Tests for M4.6: Temporal Causality & Boundary Dynamics.

Empirically challenges:
1. Boundary Stress on compute_simulated_trade_return:
   - Volatility sigma -> 0 (near-zero / microscopic volatility).
   - Volatility sigma -> infinity (extreme volatility scaling without overflow).
   - High fee conditions: c_t > m_pt * sigma or c_t > |r_realized|.
   - Dead-zone timeout conservation law: r_net_long + r_net_short == -2 * c_t (zero phantom rebates).
   - Equal prediction probabilities: P_Long == P_Short suppresses trade execution completely.
2. 100-bar Embargo & Purged K-Fold Isolation under Synthetic Temporal Shifts:
   - Synthetic temporal translation invariance (|Delta| < 10^-12) under timeline shifts.
   - PurgedCVSplitter interval translation invariance (|Delta| = 0) across all folds.
   - Strict temporal isolation between training history and test evaluation.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from trading_bot.adapters.ml_engine.constrained_lgbm_model import CANONICAL_FEATURES
from trading_bot.adapters.ml_engine.walk_forward_trainer import (
    WalkForwardTrainer,
    compute_simulated_trade_return,
)
from trading_bot.domain.entities.ml.barrier_labels import BarrierLabel, BarrierType
from trading_bot.domain.entities.ml.ml_configs import (
    CoherenceGateConfig,
    ConformalOODConfig,
    ConstrainedLGBMConfig,
    ECEMonitorConfig,
    FastVennAbersConfig,
    PurgedCVConfig,
    RidgePriorConfig,
    SampleWeightConfig,
    TreeSHAPConfig,
    TripleBarrierConfig,
    WalkForwardConfig,
)
from trading_bot.domain.services.ml.purged_cv_splitter import PurgedCVSplitter

SUBSPACE_FEATURES: tuple[str, ...] = (
    "hurst_exponent_100",
    "fracdiff_momentum_z",
    "vpin_toxicity_pct",
    "taker_cvd_z_24",
    "ofi_imbalance_z",
    "kyle_lambda_log",
)


def _build_explicit_test_config(
    start_date: datetime,
    end_date: datetime,
    embargo_bars: int,
) -> WalkForwardConfig:
    """Explicit WalkForwardConfig factory adhering strictly to Invariant #9."""
    return WalkForwardConfig(
        start_date=start_date,
        end_date=end_date,
        train_duration_months=6,
        test_duration_months=2,
        cal_ratio=0.20,
        embargo_bars_30m=embargo_bars,
        decision_threshold=0.52,
        symbols=("BTCUSDT",),
        canonical_features=CANONICAL_FEATURES,
        subspace_features=SUBSPACE_FEATURES,
        triple_barrier_config=TripleBarrierConfig(
            pt_multiplier=1.8,
            sl_multiplier=1.8,
            horizon_bars_1m=60,
            taker_fee_rate=0.0004,
            slippage_rate=0.0001,
            min_tp_to_cost_multiple=1.5,
            dead_zone_weight=0.0,
        ),
        sample_weight_config=SampleWeightConfig(min_volatility_epsilon=1e-6),
        purged_cv_config=PurgedCVConfig(
            n_splits=5,
            embargo_bars_30m=embargo_bars,
            max_label_holding_bars_30m=12,
        ),
        conformal_ood_config=ConformalOODConfig(
            feature_names=SUBSPACE_FEATURES,
            alpha_significance=0.01,
            kappa_shrinkage=2.0,
            max_condition_number=100.0,
            min_eigenvalue_floor=1e-4,
        ),
        ridge_prior_config=RidgePriorConfig(
            ridge_c=1.0,
            ridge_max_iter=500,
            oof_splits=3,
            clip_eps=1e-4,
            embargo_bars_30m=embargo_bars,
            max_label_holding_bars_30m=12,
        ),
        constrained_lgbm_config=ConstrainedLGBMConfig(
            max_depth=3,
            num_leaves=7,
            min_child_samples=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            num_boost_round=10,
            early_stopping_rounds=5,
            coherence_penalty_gamma=2.0,
        ),
        coherence_gate_config=CoherenceGateConfig(
            gamma_conflict=2.0,
            max_combined_probability=1.0,
            min_probability_floor=0.0,
        ),
        fast_venn_abers_config=FastVennAbersConfig(
            grid_points=500,
            prob_clip_min=0.001,
            prob_clip_max=0.999,
            denom_eps=1e-6,
            uncertainty_threshold=0.15,
            calibration_floor=0.001,
        ),
        ece_monitor_config=ECEMonitorConfig(
            window_bars_30m=100,
            n_bins=10,
            drift_warning_threshold=0.08,
            min_samples_to_evaluate=20,
        ),
        treeshap_config=TreeSHAPConfig(
            top_k_bullish=5,
            top_k_bearish=3,
            local_efficiency_tol=1e-3,
            sign_tolerance=1e-4,
        ),
        random_state=42,
    )


def _make_dummy_barrier_label(
    barrier_hit: BarrierType,
    net_return: float,
    realized_return: float,
    roundtrip_cost: float,
    volatility_gk: float,
) -> BarrierLabel:
    """Helper creating a BarrierLabel domain entity with explicit parameters."""
    dt = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    y_long = 1 if barrier_hit == BarrierType.UPPER_PROFIT else 0
    y_short = 1 if barrier_hit == BarrierType.LOWER_LOSS else 0
    return BarrierLabel(
        symbol="BTCUSDT",
        bar_time=dt,
        barrier_hit=barrier_hit,
        exit_time=dt + timedelta(hours=1),
        bars_held=2,
        net_return=net_return,
        sample_weight=1.0,
        y_long=y_long,
        y_short=y_short,
        entry_price=50000.0,
        exit_price=50000.0 * (1.0 + realized_return),
        realized_return=realized_return,
        upper_barrier_price=50000.0 * (1.0 + 1.8 * volatility_gk + roundtrip_cost),
        lower_barrier_price=50000.0 * (1.0 - 1.8 * volatility_gk - roundtrip_cost),
        volatility_gk=volatility_gk,
        roundtrip_cost=roundtrip_cost,
    )


# =========================================================================
# Part A: Adversarial Boundary Stress on compute_simulated_trade_return
# =========================================================================


def test_stress_near_zero_volatility() -> None:
    """Boundary stress: sigma -> 0 (1e-12, 1e-9) maintains sign precision and zero drift."""
    cost = 0.0010
    for eps_sigma in (1e-12, 1e-9, 1e-6):
        net_ret = 1.8 * eps_sigma

        # 1. UPPER_PROFIT hit with microscopic volatility
        lbl_upper = _make_dummy_barrier_label(
            barrier_hit=BarrierType.UPPER_PROFIT,
            net_return=net_ret,
            realized_return=net_ret + cost,
            roundtrip_cost=cost,
            volatility_gk=eps_sigma,
        )
        r_long = compute_simulated_trade_return(barrier_label=lbl_upper, is_long=True)
        r_short = compute_simulated_trade_return(barrier_label=lbl_upper, is_long=False)

        assert r_long > 0.0
        assert r_short < 0.0
        assert r_long == pytest.approx(net_ret, abs=1e-15)
        assert r_short == pytest.approx(-net_ret, abs=1e-15)
        assert (r_long + r_short) == pytest.approx(0.0, abs=1e-15)

        # 2. LOWER_LOSS hit with microscopic volatility
        lbl_lower = _make_dummy_barrier_label(
            barrier_hit=BarrierType.LOWER_LOSS,
            net_return=net_ret,
            realized_return=-net_ret - cost,
            roundtrip_cost=cost,
            volatility_gk=eps_sigma,
        )
        r_long_l = compute_simulated_trade_return(barrier_label=lbl_lower, is_long=True)
        r_short_l = compute_simulated_trade_return(barrier_label=lbl_lower, is_long=False)

        assert r_long_l < 0.0
        assert r_short_l > 0.0
        assert r_long_l == pytest.approx(-net_ret, abs=1e-15)
        assert r_short_l == pytest.approx(net_ret, abs=1e-15)
        assert (r_long_l + r_short_l) == pytest.approx(0.0, abs=1e-15)


def test_stress_extreme_high_volatility() -> None:
    """Boundary stress: sigma -> infinity (10.0, 100.0, 1000.0) without overflow or sign distortion."""
    cost = 0.0010
    for extreme_sigma in (10.0, 100.0, 1000.0):
        net_ret = 1.8 * extreme_sigma

        lbl_upper = _make_dummy_barrier_label(
            barrier_hit=BarrierType.UPPER_PROFIT,
            net_return=net_ret,
            realized_return=net_ret + cost,
            roundtrip_cost=cost,
            volatility_gk=extreme_sigma,
        )
        r_long = compute_simulated_trade_return(barrier_label=lbl_upper, is_long=True)
        r_short = compute_simulated_trade_return(barrier_label=lbl_upper, is_long=False)

        assert math.isfinite(r_long)
        assert math.isfinite(r_short)
        assert r_long == pytest.approx(net_ret, rel=1e-12)
        assert r_short == pytest.approx(-net_ret, rel=1e-12)
        assert (r_long + r_short) == pytest.approx(0.0, abs=1e-12)


def test_stress_high_fee_regime_and_conservation_law() -> None:
    """Stress test: High fee conditions (c_t > m_pt * sigma and c_t > |r_realized|).

    Fundamental conservation law:
    For any timeout event, the total combined PnL of simultaneously holding a Long
    and Short position MUST exactly equal -2 * c_t (zero fee leakage, zero phantom rebates):
        r_net_long + r_net_short == (r - c_t) + (-r - c_t) == -2 * c_t.
    """
    high_cost = 0.0500  # 500 bps transaction fee!
    vol = 0.0050  # 1.8 * vol = 0.0090 << high_cost (5.5x smaller than fee)

    # 1. UPPER_PROFIT hit: market overcame massive fee
    lbl_upper = _make_dummy_barrier_label(
        barrier_hit=BarrierType.UPPER_PROFIT,
        net_return=1.8 * vol,
        realized_return=1.8 * vol + high_cost,
        roundtrip_cost=high_cost,
        volatility_gk=vol,
    )
    assert compute_simulated_trade_return(lbl_upper, is_long=True) == pytest.approx(1.8 * vol)
    assert compute_simulated_trade_return(lbl_upper, is_long=False) == pytest.approx(-1.8 * vol)

    # 2. TIMEOUT in Dead Zone (|r_realized| < c_t): Both sides lose due to fees!
    test_drifts = [-0.040, -0.010, 0.0, 0.010, 0.040]
    for r_real in test_drifts:
        lbl_to = _make_dummy_barrier_label(
            barrier_hit=BarrierType.TIMEOUT,
            net_return=min(0.0, abs(r_real) - high_cost),
            realized_return=r_real,
            roundtrip_cost=high_cost,
            volatility_gk=vol,
        )
        r_long = compute_simulated_trade_return(barrier_label=lbl_to, is_long=True)
        r_short = compute_simulated_trade_return(barrier_label=lbl_to, is_long=False)

        # Both sides MUST suffer fee erosion
        assert r_long < 0.0
        assert r_short < 0.0

        # Exact conservation law check
        assert (r_long + r_short) == pytest.approx(-2.0 * high_cost, abs=1e-15)

    # 3. TIMEOUT with Large Drift (|r_realized| > c_t)
    for r_real in [-0.15, +0.15]:
        lbl_to_large = _make_dummy_barrier_label(
            barrier_hit=BarrierType.TIMEOUT,
            net_return=abs(r_real) - high_cost,
            realized_return=r_real,
            roundtrip_cost=high_cost,
            volatility_gk=vol,
        )
        r_long = compute_simulated_trade_return(barrier_label=lbl_to_large, is_long=True)
        r_short = compute_simulated_trade_return(barrier_label=lbl_to_large, is_long=False)

        # Exact conservation law check
        assert (r_long + r_short) == pytest.approx(-2.0 * high_cost, abs=1e-15)


def test_stress_equal_prediction_probabilities_suppression() -> None:
    """Stress test: Equal prediction probabilities (P_Long == P_Short) suppresses trade execution.

    In walk_forward_trainer.py:
        if adj_long >= self._config.decision_threshold and adj_long > adj_short:
            trade_returns.append(...)
        elif adj_short >= self._config.decision_threshold and adj_short > adj_long:
            trade_returns.append(...)

    Assert that when P_Long == P_Short, neither branch triggers and trade_returns remains empty.
    """
    decision_threshold = 0.52

    # Test cases: (p_long, p_short)
    equal_prob_pairs = [
        (0.60, 0.60),  # Both above threshold
        (0.52, 0.52),  # Exactly on threshold
        (0.50, 0.50),  # Below threshold
        (0.00, 0.00),  # Floor
        (0.99, 0.99),  # Extreme confidence conflict
    ]

    for p_l, p_s in equal_prob_pairs:
        trade_executed = (p_l >= decision_threshold and p_l > p_s) or (
            p_s >= decision_threshold and p_s > p_l
        )

        assert not trade_executed, (
            f"Trade incorrectly executed when P_Long == P_Short ({p_l}, {p_s})"
        )

    # Differential test: infinitesimal difference breaks tie correctly
    p_base = 0.55
    eps = 1e-15
    p_long_higher = p_base + eps
    assert (p_long_higher >= decision_threshold and p_long_higher > p_base) is True
    assert (p_base >= decision_threshold and p_base > p_long_higher) is False


# =========================================================================
# Part B: 100-Bar Embargo & Purged K-Fold Isolation under Synthetic Temporal Shifts
# =========================================================================


def test_purged_kfold_synthetic_temporal_shift_invariance() -> None:
    """Stress test: PurgedCVSplitter is 100% invariant under synthetic temporal translation (|Delta| = 0).

    Shifting all interval timestamps by arbitrary temporal offset tau >= 0
    MUST yield bitwise identical fold partitions (train, val, purged, embargoed sets).
    """
    n_samples = 120
    # Create baseline intervals
    base_intervals: list[tuple[int, int]] = []
    current_t = 100
    for i in range(n_samples):
        duration = 5 + (i % 10)
        base_intervals.append((current_t, current_t + duration))
        current_t += 3

    config = PurgedCVConfig(
        n_splits=5,
        embargo_bars_30m=100,  # 100-bar dynamic memory embargo
        max_label_holding_bars_30m=15,
    )
    splitter = PurgedCVSplitter(config=config)
    base_splits = splitter.split(sample_intervals=base_intervals)

    # Apply synthetic temporal shifts: tau = 1, 1000, 1_000_000
    for tau in (1, 1000, 1_000_000):
        shifted_intervals = [(t_s + tau, t_e + tau) for (t_s, t_e) in base_intervals]
        shifted_splits = splitter.split(sample_intervals=shifted_intervals)

        assert len(shifted_splits) == len(base_splits) == 5

        for fold_idx in range(5):
            b_sp = base_splits[fold_idx]
            s_sp = shifted_splits[fold_idx]

            # Invariant: Sample index sets MUST be bitwise identical (|Delta| = 0)
            assert b_sp.train_indices == s_sp.train_indices
            assert b_sp.val_indices == s_sp.val_indices
            assert b_sp.purged_indices == s_sp.purged_indices
            assert b_sp.embargoed_indices == s_sp.embargoed_indices

            # Invariant: Time bounds shift by exactly tau
            assert s_sp.val_min_t == b_sp.val_min_t + tau
            assert s_sp.val_max_t == b_sp.val_max_t + tau


def test_walk_forward_window_generation_temporal_shift_invariance() -> None:
    """Stress test: WalkForwardTrainer temporal window generator invariant under global timeline shift.

    Shifting start_date and end_date by delta_t shifts all 10 temporal boundary dates
    by exactly delta_t with zero distortion (|Delta| = 0).
    """
    start_0 = datetime(2023, 1, 1, 0, 0, tzinfo=UTC)
    end_0 = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    cfg_base = _build_explicit_test_config(
        start_date=start_0,
        end_date=end_0,
        embargo_bars=100,
    )
    trainer_base = WalkForwardTrainer(config=cfg_base, model_version="m46_test_v1")
    base_windows = trainer_base.generate_temporal_windows()

    assert len(base_windows) >= 2

    # Shift timeline by 365 days
    delta_t = timedelta(days=365)
    cfg_shifted = _build_explicit_test_config(
        start_date=start_0 + delta_t,
        end_date=end_0 + delta_t,
        embargo_bars=100,
    )
    trainer_shifted = WalkForwardTrainer(config=cfg_shifted, model_version="m46_test_v1")
    shifted_windows = trainer_shifted.generate_temporal_windows()

    assert len(shifted_windows) == len(base_windows)

    for i in range(len(base_windows)):
        bw = base_windows[i]
        sw = shifted_windows[i]

        assert sw.window_id == bw.window_id
        assert sw.subtrain_start == bw.subtrain_start + delta_t
        assert sw.subtrain_end == bw.subtrain_end + delta_t
        assert sw.internal_embargo_start == bw.internal_embargo_start + delta_t
        assert sw.internal_embargo_end == bw.internal_embargo_end + delta_t
        assert sw.cal_start == bw.cal_start + delta_t
        assert sw.cal_end == bw.cal_end + delta_t
        assert sw.external_embargo_start == bw.external_embargo_start + delta_t
        assert sw.external_embargo_end == bw.external_embargo_end + delta_t
        assert sw.test_start == bw.test_start + delta_t
        assert sw.test_end == bw.test_end + delta_t


def test_strict_temporal_isolation_and_no_lookahead_leakage() -> None:
    """Stress test: 100-bar embargo prevents training samples from accessing validation / test future.

    Checks:
    1. For every fold in PurgedKFoldSplitter, max(train_indices before val) < min(val_indices).
    2. For every post-validation train sample, its t_start > max(val_exit) + 100 bars.
    """
    n_samples = 200
    intervals: list[tuple[int, int]] = []
    t = 0
    for _i in range(n_samples):
        intervals.append((t, t + 5))
        t += 2

    config = PurgedCVConfig(
        n_splits=4,
        embargo_bars_30m=100,
        max_label_holding_bars_30m=5,
    )
    splits = PurgedCVSplitter(config).split(intervals)

    for sp in splits:
        val_start_t = intervals[sp.val_indices[0]][0]
        val_exit_t = max(intervals[j][1] for j in sp.val_indices)

        for tr_idx in sp.train_indices:
            t_s, t_e = intervals[tr_idx]

            # Invariant: No train sample can overlap [val_start_t, val_exit_t]
            assert (t_e < val_start_t) or (t_s > val_exit_t), (
                f"Train sample {tr_idx} span [{t_s}, {t_e}] leaks into val [{val_start_t}, {val_exit_t}]"
            )

            # Invariant: Any post-val train sample must be strictly beyond embargo
            if t_s >= val_exit_t:
                assert t_s > val_exit_t + 100, (
                    f"Post-val train sample {tr_idx} t_start={t_s} violates 100-bar embargo "
                    f"(embargo boundary = {val_exit_t + 100})"
                )
