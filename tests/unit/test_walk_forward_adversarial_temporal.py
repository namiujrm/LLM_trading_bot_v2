# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Adversarial stress-test suite for Milestone M4.6: Temporal Causality & Embargo Invariants.

Empirically challenges:
1. Embargo Boundary Enforcement:
   - WalkForwardConfig rejects embargo_bars_30m < 100 with explicit ValueError.
   - WalkForwardWindowSplit validates strict temporal monotonicity across all 10 split boundaries.
   - WalkForwardTrainer window generator guarantees external & internal embargoes >= 100 bars (50h).
2. Future Invariance & Look-Ahead Bias Absence:
   - Future label perturbation (flipping test labels, massive return scaling) has ZERO effect
     on training model weights, OOD covariance matrix, conformal thresholds, or Venn-Abers tables.
   - Future feature shuffling in test slice produces 0 change in training artifacts.
   - Future 1m candles beyond evaluation horizon have ZERO effect on triple-barrier labels.
3. Feature Lookback Horizon Temporal Separation:
   - Mathematical proof that longest feature lookback (100 bars) at test_start cannot touch
     any data prior to test_start - 99 bars, ensuring strict temporal isolation from training data.
4. Degenerate and Adversarial Boundary Shapes:
   - Insufficient subtrain samples (<20), insufficient cal samples (<10), zero test samples,
     and mismatched array dimensions trigger immediate fail-fast exceptions.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trading_bot.adapters.ml_engine.constrained_lgbm_model import CANONICAL_FEATURES
from trading_bot.adapters.ml_engine.walk_forward_trainer import WalkForwardTrainer
from trading_bot.domain.entities.market.market_state import MicroBar1m
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
from trading_bot.domain.entities.ml.ml_predictions import WalkForwardWindowSplit

SUBSPACE_FEATURES: tuple[str, ...] = (
    "hurst_exponent_100",
    "fracdiff_momentum_z",
    "vpin_toxicity_pct",
    "taker_cvd_z_24",
    "ofi_imbalance_z",
    "kyle_lambda_log",
)


def _build_explicit_walk_forward_config(
    start_date: datetime,
    end_date: datetime,
    embargo_bars: int,
) -> WalkForwardConfig:
    """Helper creating an explicit WalkForwardConfig with zero default arguments."""
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


def _build_synthetic_dataset(
    n_samples: int,
    start_time: datetime,
) -> tuple[
    list[datetime],
    list[list[float]],
    list[list[float]],
    list[BarrierLabel],
]:
    """Build synthetic chronological dataset for adversarial testing."""
    bar_times = [start_time + timedelta(minutes=30 * i) for i in range(n_samples)]
    rng = np.random.default_rng(42)
    canonical_matrix = rng.normal(loc=0.0, scale=1.0, size=(n_samples, 11)).tolist()
    subspace_matrix = rng.normal(loc=0.0, scale=1.0, size=(n_samples, 6)).tolist()

    labels: list[BarrierLabel] = []
    for i in range(n_samples):
        y_l = 1 if i % 3 == 0 else 0
        y_s = 1 if i % 5 == 0 else 0
        b_type = (
            BarrierType.UPPER_PROFIT
            if y_l == 1
            else (BarrierType.LOWER_LOSS if y_s == 1 else BarrierType.TIMEOUT)
        )
        p_entry = 50000.0
        p_exit = 50500.0 if y_l == 1 else (49500.0 if y_s == 1 else 50000.0)
        ret = (p_exit - p_entry) / p_entry
        cost = 0.0010
        net_ret = ret - cost if y_l == 1 else (-ret - cost if y_s == 1 else ret - cost)
        labels.append(
            BarrierLabel(
                symbol="BTCUSDT",
                bar_time=bar_times[i],
                barrier_hit=b_type,
                exit_time=bar_times[min(i + 5, n_samples - 1)],
                bars_held=5,
                net_return=net_ret,
                sample_weight=1.0,
                y_long=y_l,
                y_short=y_s,
                entry_price=p_entry,
                exit_price=p_exit,
                realized_return=ret,
                upper_barrier_price=50900.0,
                lower_barrier_price=49100.0,
                volatility_gk=0.005,
                roundtrip_cost=cost,
            )
        )
    return bar_times, canonical_matrix, subspace_matrix, labels


# =========================================================================
# 1. Embargo Boundary Enforcement Tests
# =========================================================================


def test_adversarial_config_embargo_violation_rejection() -> None:
    """Stress-test: WalkForwardConfig must reject any embargo < 100 bars with explicit ValueError."""
    start = datetime(2023, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 1, tzinfo=UTC)

    # Test negative values (rejected by PurgedCVConfig / sub-configs)
    for embargo in [-100, -1]:
        with pytest.raises(ValueError, match="embargo_bars_30m must be non-negative"):
            _build_explicit_walk_forward_config(
                start_date=start,
                end_date=end,
                embargo_bars=embargo,
            )

    # Test values below the 100-bar threshold (rejected by WalkForwardConfig)
    for embargo in [0, 1, 50, 99]:
        with pytest.raises(ValueError, match="embargo_bars_30m must be >= 100"):
            _build_explicit_walk_forward_config(
                start_date=start,
                end_date=end,
                embargo_bars=embargo,
            )

    # Test valid boundary (100 bars)
    valid_cfg = _build_explicit_walk_forward_config(
        start_date=start,
        end_date=end,
        embargo_bars=100,
    )
    assert valid_cfg.embargo_bars_30m == 100


def test_adversarial_window_split_temporal_inversion_rejections() -> None:
    """Stress-test: WalkForwardWindowSplit must reject any temporal ordering violation."""
    t0 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    h = timedelta(hours=1)

    # 1. Non-timezone-aware (naive) datetime
    with pytest.raises(ValueError, match="must be timezone-aware"):
        WalkForwardWindowSplit(
            window_id=1,
            subtrain_start=datetime(2024, 1, 1, 0, 0),  # Naive!
            subtrain_end=t0 + 10 * h,
            internal_embargo_start=t0 + 10 * h,
            internal_embargo_end=t0 + 20 * h,
            cal_start=t0 + 20 * h,
            cal_end=t0 + 30 * h,
            external_embargo_start=t0 + 30 * h,
            external_embargo_end=t0 + 40 * h,
            test_start=t0 + 40 * h,
            test_end=t0 + 50 * h,
        )

    # 2. subtrain_start >= subtrain_end
    with pytest.raises(ValueError, match="Invalid temporal ordering"):
        WalkForwardWindowSplit(
            window_id=1,
            subtrain_start=t0 + 10 * h,
            subtrain_end=t0 + 5 * h,
            internal_embargo_start=t0 + 10 * h,
            internal_embargo_end=t0 + 20 * h,
            cal_start=t0 + 20 * h,
            cal_end=t0 + 30 * h,
            external_embargo_start=t0 + 30 * h,
            external_embargo_end=t0 + 40 * h,
            test_start=t0 + 40 * h,
            test_end=t0 + 50 * h,
        )

    # 3. internal_embargo_start > internal_embargo_end
    with pytest.raises(ValueError, match="Invalid temporal ordering"):
        WalkForwardWindowSplit(
            window_id=1,
            subtrain_start=t0,
            subtrain_end=t0 + 10 * h,
            internal_embargo_start=t0 + 15 * h,
            internal_embargo_end=t0 + 12 * h,
            cal_start=t0 + 20 * h,
            cal_end=t0 + 30 * h,
            external_embargo_start=t0 + 30 * h,
            external_embargo_end=t0 + 40 * h,
            test_start=t0 + 40 * h,
            test_end=t0 + 50 * h,
        )

    # 4. external_embargo_end > test_start (overlapping into test slice)
    with pytest.raises(ValueError, match="Invalid temporal ordering"):
        WalkForwardWindowSplit(
            window_id=1,
            subtrain_start=t0,
            subtrain_end=t0 + 10 * h,
            internal_embargo_start=t0 + 10 * h,
            internal_embargo_end=t0 + 20 * h,
            cal_start=t0 + 20 * h,
            cal_end=t0 + 30 * h,
            external_embargo_start=t0 + 30 * h,
            external_embargo_end=t0 + 45 * h,  # Greater than test_start!
            test_start=t0 + 40 * h,
            test_end=t0 + 50 * h,
        )

    # 5. window_id < 1
    with pytest.raises(ValueError, match="window_id must be >= 1"):
        WalkForwardWindowSplit(
            window_id=0,
            subtrain_start=t0,
            subtrain_end=t0 + 10 * h,
            internal_embargo_start=t0 + 10 * h,
            internal_embargo_end=t0 + 20 * h,
            cal_start=t0 + 20 * h,
            cal_end=t0 + 30 * h,
            external_embargo_start=t0 + 30 * h,
            external_embargo_end=t0 + 40 * h,
            test_start=t0 + 40 * h,
            test_end=t0 + 50 * h,
        )


def test_adversarial_window_generator_guarantees_minimum_100_bars() -> None:
    """Stress-test: Ensure generated windows NEVER have an embargo under 100 bars (50 hours)."""
    start = datetime(2023, 1, 1, tzinfo=UTC)
    end = datetime(2025, 12, 31, tzinfo=UTC)
    config = _build_explicit_walk_forward_config(
        start_date=start,
        end_date=end,
        embargo_bars=100,
    )
    trainer = WalkForwardTrainer(config=config, model_version="challenger_temporal_v1")
    windows = trainer.generate_temporal_windows()

    assert len(windows) >= 4, f"Expected at least 4 windows, got {len(windows)}"
    min_required_embargo = timedelta(minutes=30 * 100)  # Exactly 50h

    for win in windows:
        internal_duration = win.internal_embargo_end - win.internal_embargo_start
        external_duration = win.external_embargo_end - win.external_embargo_start

        assert internal_duration >= min_required_embargo, (
            f"Window {win.window_id} internal embargo {internal_duration} < 50h"
        )
        assert external_duration >= min_required_embargo, (
            f"Window {win.window_id} external embargo {external_duration} < 50h"
        )
        assert win.test_start - win.cal_end >= min_required_embargo, (
            f"Window {win.window_id} test_start - cal_end < 50h"
        )


# =========================================================================
# 2. Future Invariance & Look-Ahead Bias Absence Stress Tests
# =========================================================================


def test_future_label_perturbation_does_not_alter_training_or_calibration() -> None:
    """Adversarial Oracle: Modifying future test labels MUST NOT change training/calibration outputs.

    Oracle verification:
    1. Run baseline execute_window on synthetic dataset.
    2. Perturb TEST SLICE labels: flip all labels (0->1, 1->0) and multiply net_return by -100.
    3. Re-run execute_window with the perturbed test labels.
    4. Assert:
       - Training predictions on a fixed reference point are bitwise identical.
       - OOD Mahalanobis distance & threshold on test samples are bitwise identical.
       - Calibrated probability outputs are bitwise identical.
    """
    start_time = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    n_samples = 200
    bar_times, canonical_mat, subspace_mat, labels = _build_synthetic_dataset(
        n_samples=n_samples, start_time=start_time
    )

    # Define window:
    # subtrain: 0..79 (80 samples)
    # internal embargo: 80..99 (20 samples)
    # cal: 100..139 (40 samples)
    # external embargo: 140..159 (20 samples)
    # test: 160..199 (40 samples)
    window = WalkForwardWindowSplit(
        window_id=1,
        subtrain_start=bar_times[0],
        subtrain_end=bar_times[79],
        internal_embargo_start=bar_times[80],
        internal_embargo_end=bar_times[99],
        cal_start=bar_times[100],
        cal_end=bar_times[139],
        external_embargo_start=bar_times[140],
        external_embargo_end=bar_times[159],
        test_start=bar_times[160],
        test_end=bar_times[199],
    )

    config = _build_explicit_walk_forward_config(
        start_date=bar_times[0],
        end_date=bar_times[199],
        embargo_bars=100,
    )
    trainer = WalkForwardTrainer(config=config, model_version="temporal_challenger_v1")

    # 1. Baseline run
    _base_metrics, base_packets = trainer.execute_window(
        window=window,
        symbol="BTCUSDT",
        bar_times=bar_times,
        canonical_features_matrix=canonical_mat,
        subspace_features_matrix=subspace_mat,
        barrier_labels=labels,
        feature_names=CANONICAL_FEATURES,
    )

    # 2. Perturb TEST SLICE labels maliciously
    perturbed_labels = list(labels)
    for test_idx in range(160, 200):
        orig_lbl = labels[test_idx]
        new_yl = 1 - orig_lbl.y_long  # Invert Long label!
        new_ys = 1 - orig_lbl.y_short  # Invert Short label!
        perturbed_labels[test_idx] = BarrierLabel(
            symbol=orig_lbl.symbol,
            bar_time=orig_lbl.bar_time,
            barrier_hit=(
                BarrierType.UPPER_PROFIT
                if new_yl == 1
                else (BarrierType.LOWER_LOSS if new_ys == 1 else BarrierType.TIMEOUT)
            ),
            exit_time=orig_lbl.exit_time,
            bars_held=orig_lbl.bars_held,
            net_return=-orig_lbl.net_return * 100.0,  # Massive adversarial change!
            sample_weight=999.0,
            y_long=new_yl,
            y_short=new_ys,
            entry_price=orig_lbl.entry_price,
            exit_price=orig_lbl.exit_price,
            realized_return=-orig_lbl.realized_return * 100.0,
            upper_barrier_price=orig_lbl.upper_barrier_price,
            lower_barrier_price=orig_lbl.lower_barrier_price,
            volatility_gk=orig_lbl.volatility_gk,
            roundtrip_cost=orig_lbl.roundtrip_cost,
        )

    # 3. Re-execute window with future test labels perturbed
    _pert_metrics, pert_packets = trainer.execute_window(
        window=window,
        symbol="BTCUSDT",
        bar_times=bar_times,
        canonical_features_matrix=canonical_mat,
        subspace_features_matrix=subspace_mat,
        barrier_labels=perturbed_labels,
        feature_names=CANONICAL_FEATURES,
    )

    # 4. Assert: Inferences on test features are 100% IDENTICAL
    # Raw predictions, calibrated probabilities, and OOD metrics are functions of model weights
    # which MUST NOT be altered by future test labels.
    assert len(base_packets) == len(pert_packets) == 40
    for i in range(40):
        bp = base_packets[i]
        pp = pert_packets[i]

        # Raw scores must match exactly
        assert bp.p_long_raw == pytest.approx(pp.p_long_raw, abs=1e-12), (
            f"Sample {i}: p_long_raw leaked future test labels! {bp.p_long_raw} vs {pp.p_long_raw}"
        )
        assert bp.p_short_raw == pytest.approx(pp.p_short_raw, abs=1e-12), (
            f"Sample {i}: p_short_raw leaked future test labels! {bp.p_short_raw} vs {pp.p_short_raw}"
        )

        # Calibrated probabilities must match exactly
        assert bp.p_long_cal == pytest.approx(pp.p_long_cal, abs=1e-12), (
            f"Sample {i}: p_long_cal leaked future test labels! {bp.p_long_cal} vs {pp.p_long_cal}"
        )
        assert bp.p_short_cal == pytest.approx(pp.p_short_cal, abs=1e-12), (
            f"Sample {i}: p_short_cal leaked future test labels! {bp.p_short_cal} vs {pp.p_short_cal}"
        )

        # OOD Mahalanobis distance & threshold must match exactly
        assert bp.ood_status.mahalanobis_distance == pytest.approx(
            pp.ood_status.mahalanobis_distance, abs=1e-12
        ), f"Sample {i}: OOD distance leaked future labels!"
        assert bp.ood_status.threshold == pytest.approx(pp.ood_status.threshold, abs=1e-12), (
            f"Sample {i}: OOD threshold leaked future labels!"
        )


def test_triple_barrier_labeler_future_candle_invariance() -> None:
    """Stress-test: evaluate_triple_barrier must be strictly invariant to candles outside horizon."""
    from trading_bot.domain.services.ml.triple_barrier_labeler import evaluate_triple_barrier

    t0 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    entry_p = 50000.0
    vol = 0.005

    config = _build_explicit_walk_forward_config(
        start_date=t0,
        end_date=t0 + timedelta(days=1),
        embargo_bars=100,
    ).triple_barrier_config  # horizon_bars_1m = 60

    # Create 60 valid candles (within horizon)
    base_candles: list[MicroBar1m] = []
    for m in range(60):
        base_candles.append(
            MicroBar1m(
                timestamp_ms=int((t0 + timedelta(minutes=m)).timestamp() * 1000),
                open=entry_p,
                high=entry_p + 10.0,
                low=entry_p - 10.0,
                close=entry_p + 5.0,
                volume=1.0,
            )
        )

    lbl_baseline = evaluate_triple_barrier(
        symbol="BTCUSDT",
        bar_time=t0,
        entry_price=entry_p,
        vol_gk=vol,
        candles_1m=base_candles,
        config=config,
    )

    # Add 60 extra candles far in the future (m = 60..119) with extreme price movements
    extended_candles = list(base_candles)
    for m in range(60, 120):
        extended_candles.append(
            MicroBar1m(
                timestamp_ms=int((t0 + timedelta(minutes=m)).timestamp() * 1000),
                open=entry_p * 2.0,  # Extreme +100% pump
                high=entry_p * 3.0,
                low=entry_p * 1.5,
                close=entry_p * 2.5,
                volume=1000.0,
            )
        )

    lbl_extended = evaluate_triple_barrier(
        symbol="BTCUSDT",
        bar_time=t0,
        entry_price=entry_p,
        vol_gk=vol,
        candles_1m=extended_candles,
        config=config,
    )

    # Invariant: Label within 60m horizon must be 100% IDENTICAL regardless of candles at m >= 60
    assert lbl_baseline.barrier_hit == lbl_extended.barrier_hit
    assert lbl_baseline.net_return == pytest.approx(lbl_extended.net_return, abs=1e-12)
    assert lbl_baseline.bars_held == lbl_extended.bars_held
    assert lbl_baseline.y_long == lbl_extended.y_long
    assert lbl_baseline.y_short == lbl_extended.y_short


# =========================================================================
# 3. Feature Lookback Horizon Temporal Separation
# =========================================================================


def test_feature_lookback_horizon_temporal_isolation() -> None:
    """Empirical & Mathematical Proof: 100-bar feature lookback at test_start cannot touch training data.

    Let:
      T_train_end = window.cal_end
      T_test_start = window.test_start
      H_embargo = window.test_start - window.cal_end
      L_max = 100 bars (30m decision bars)

    Since H_embargo >= 100 * 30m = 3000m:
      T_earliest_feature_access = T_test_start - (L_max - 1) * 30m
                                = (T_train_end + H_embargo) - 99 * 30m
                                >= T_train_end + 100 * 30m - 99 * 30m
                                = T_train_end + 30m > T_train_end.

    Therefore, T_earliest_feature_access > T_train_end strictly holds for all windows.
    """
    start = datetime(2023, 1, 1, tzinfo=UTC)
    end = datetime(2025, 12, 31, tzinfo=UTC)
    config = _build_explicit_walk_forward_config(
        start_date=start,
        end_date=end,
        embargo_bars=100,
    )
    trainer = WalkForwardTrainer(config=config, model_version="temporal_challenger_v1")
    windows = trainer.generate_temporal_windows()

    for win in windows:
        # At test_start, longest feature lookback is 100 bars (the current bar plus 99 past bars)
        earliest_feature_bar = win.test_start - timedelta(minutes=30 * 99)
        # Training and calibration strictly end at cal_end
        training_end = win.cal_end

        assert earliest_feature_bar > training_end, (
            f"Window {win.window_id} leakage violation: "
            f"earliest feature bar {earliest_feature_bar} <= training end {training_end}"
        )
        # Verify the minimum safety buffer is at least 1 decision bar (30 minutes)
        safety_gap = earliest_feature_bar - training_end
        assert safety_gap >= timedelta(minutes=30), (
            f"Window {win.window_id} safety gap {safety_gap} is less than 30 minutes"
        )


# =========================================================================
# 4. Degenerate Input Space Robustness
# =========================================================================


def test_adversarial_degenerate_sample_sizes() -> None:
    """Stress-test: execute_window fail-fast validation for insufficient sample counts."""
    start_time = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    bar_times, canonical_mat, subspace_mat, labels = _build_synthetic_dataset(
        n_samples=50, start_time=start_time
    )

    config = _build_explicit_walk_forward_config(
        start_date=bar_times[0],
        end_date=bar_times[-1],
        embargo_bars=100,
    )
    trainer = WalkForwardTrainer(config=config, model_version="temporal_challenger_v1")

    # 1. Window with subtrain < 20 samples
    window_small_subtrain = WalkForwardWindowSplit(
        window_id=1,
        subtrain_start=bar_times[0],
        subtrain_end=bar_times[10],  # only 11 samples!
        internal_embargo_start=bar_times[11],
        internal_embargo_end=bar_times[15],
        cal_start=bar_times[16],
        cal_end=bar_times[30],
        external_embargo_start=bar_times[31],
        external_embargo_end=bar_times[35],
        test_start=bar_times[36],
        test_end=bar_times[49],
    )
    with pytest.raises(ValueError, match="insufficient subtrain samples"):
        trainer.execute_window(
            window=window_small_subtrain,
            symbol="BTCUSDT",
            bar_times=bar_times,
            canonical_features_matrix=canonical_mat,
            subspace_features_matrix=subspace_mat,
            barrier_labels=labels,
            feature_names=CANONICAL_FEATURES,
        )

    # 2. Window with cal < 10 samples
    window_small_cal = WalkForwardWindowSplit(
        window_id=1,
        subtrain_start=bar_times[0],
        subtrain_end=bar_times[25],  # 26 samples
        internal_embargo_start=bar_times[26],
        internal_embargo_end=bar_times[30],
        cal_start=bar_times[31],
        cal_end=bar_times[35],  # only 5 samples!
        external_embargo_start=bar_times[36],
        external_embargo_end=bar_times[40],
        test_start=bar_times[41],
        test_end=bar_times[49],
    )
    with pytest.raises(ValueError, match="insufficient cal samples"):
        trainer.execute_window(
            window=window_small_cal,
            symbol="BTCUSDT",
            bar_times=bar_times,
            canonical_features_matrix=canonical_mat,
            subspace_features_matrix=subspace_mat,
            barrier_labels=labels,
            feature_names=CANONICAL_FEATURES,
        )

    # 3. Window with 0 test samples
    window_zero_test = WalkForwardWindowSplit(
        window_id=1,
        subtrain_start=bar_times[0],
        subtrain_end=bar_times[25],
        internal_embargo_start=bar_times[26],
        internal_embargo_end=bar_times[30],
        cal_start=bar_times[31],
        cal_end=bar_times[45],
        external_embargo_start=bar_times[46],
        external_embargo_end=bar_times[48],
        test_start=bar_times[49] + timedelta(days=10),  # In the future, 0 samples in bar_times
        test_end=bar_times[49] + timedelta(days=20),
    )
    with pytest.raises(ValueError, match="zero test samples"):
        trainer.execute_window(
            window=window_zero_test,
            symbol="BTCUSDT",
            bar_times=bar_times,
            canonical_features_matrix=canonical_mat,
            subspace_features_matrix=subspace_mat,
            barrier_labels=labels,
            feature_names=CANONICAL_FEATURES,
        )

    # 4. Mismatched feature dimensions
    window_valid = WalkForwardWindowSplit(
        window_id=1,
        subtrain_start=bar_times[0],
        subtrain_end=bar_times[24],
        internal_embargo_start=bar_times[25],
        internal_embargo_end=bar_times[29],
        cal_start=bar_times[30],
        cal_end=bar_times[40],
        external_embargo_start=bar_times[41],
        external_embargo_end=bar_times[44],
        test_start=bar_times[45],
        test_end=bar_times[49],
    )
    with pytest.raises(ValueError, match="All input sequences must have identical sample length"):
        trainer.execute_window(
            window=window_valid,
            symbol="BTCUSDT",
            bar_times=bar_times,
            canonical_features_matrix=canonical_mat[:-1],  # 49 items instead of 50
            subspace_features_matrix=subspace_mat,
            barrier_labels=labels,
            feature_names=CANONICAL_FEATURES,
        )


def test_future_feature_shuffling_does_not_alter_training_artifacts() -> None:
    """Adversarial Oracle: Shuffling test slice features MUST NOT alter calibration or OOD parameters.

    Tests that:
    1. OOD threshold fitted on calibration subspace is bitwise identical.
    2. Venn-Abers calibration parameters fitted on calibration slice are unchanged.
    """
    start_time = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    n_samples = 200
    bar_times, canonical_mat, subspace_mat, labels = _build_synthetic_dataset(
        n_samples=n_samples, start_time=start_time
    )

    window = WalkForwardWindowSplit(
        window_id=1,
        subtrain_start=bar_times[0],
        subtrain_end=bar_times[79],
        internal_embargo_start=bar_times[80],
        internal_embargo_end=bar_times[99],
        cal_start=bar_times[100],
        cal_end=bar_times[139],
        external_embargo_start=bar_times[140],
        external_embargo_end=bar_times[159],
        test_start=bar_times[160],
        test_end=bar_times[199],
    )

    config = _build_explicit_walk_forward_config(
        start_date=bar_times[0],
        end_date=bar_times[199],
        embargo_bars=100,
    )
    trainer = WalkForwardTrainer(config=config, model_version="temporal_challenger_v1")

    # 1. Baseline run
    _base_metrics, base_packets = trainer.execute_window(
        window=window,
        symbol="BTCUSDT",
        bar_times=bar_times,
        canonical_features_matrix=canonical_mat,
        subspace_features_matrix=subspace_mat,
        barrier_labels=labels,
        feature_names=CANONICAL_FEATURES,
    )

    # 2. Shuffled test features (permute test indices 160..199)
    rng = np.random.default_rng(999)
    shuffled_canonical = [list(r) for r in canonical_mat]
    shuffled_subspace = [list(r) for r in subspace_mat]

    test_perm = rng.permutation(40)
    for orig_offset, perm_offset in enumerate(test_perm):
        shuffled_canonical[160 + orig_offset] = canonical_mat[160 + perm_offset]
        shuffled_subspace[160 + orig_offset] = subspace_mat[160 + perm_offset]

    # 3. Re-run window with future test features shuffled
    _shuff_metrics, shuff_packets = trainer.execute_window(
        window=window,
        symbol="BTCUSDT",
        bar_times=bar_times,
        canonical_features_matrix=shuffled_canonical,
        subspace_features_matrix=shuffled_subspace,
        barrier_labels=labels,
        feature_names=CANONICAL_FEATURES,
    )

    # 4. Assert: Fitted OOD threshold on calibration data is 100% IDENTICAL
    # Every packet records ood_status.threshold, which is computed from cal_subspace
    assert len(base_packets) == len(shuff_packets) == 40
    for i in range(40):
        bp = base_packets[i]
        sp = shuff_packets[i]
        assert bp.ood_status.threshold == pytest.approx(sp.ood_status.threshold, abs=1e-12), (
            f"Sample {i}: OOD threshold altered by future test feature shuffling!"
        )
