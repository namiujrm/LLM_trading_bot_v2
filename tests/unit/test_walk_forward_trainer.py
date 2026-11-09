"""Unit tests for WalkForwardTrainer: window splitting, embargo enforcement, and causal evaluation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trading_bot.adapters.ml_engine.constrained_lgbm_model import CANONICAL_FEATURES
from trading_bot.adapters.ml_engine.walk_forward_trainer import (
    WalkForwardTrainer,
    compute_brier_score_binary,
    compute_log_loss_binary,
    compute_roc_auc_binary,
)
from trading_bot.domain.entities.market.market_state import MicroBar1m
from trading_bot.domain.entities.ml.barrier_labels import BarrierType
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


def create_test_walk_forward_config(
    start_date: datetime,
    end_date: datetime,
    embargo_bars: int,
) -> WalkForwardConfig:
    """Helper creating an explicit WalkForwardConfig with 0 default values."""
    subspace_features = (
        "hurst_exponent_100",
        "fracdiff_momentum_z",
        "vpin_toxicity_pct",
        "taker_cvd_z_24",
        "ofi_imbalance_z",
        "kyle_lambda_log",
    )
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
        subspace_features=subspace_features,
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
            feature_names=subspace_features,
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


def test_walk_forward_config_embargo_invariant() -> None:
    """Invariant test: embargo_bars_30m < 100 must raise ValueError."""
    start = datetime(2023, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="embargo_bars_30m must be >= 100"):
        create_test_walk_forward_config(start_date=start, end_date=end, embargo_bars=50)


def test_walk_forward_window_generation_and_embargo_enforcement() -> None:
    """Test deterministic walk-forward window generation and 100-bar embargo verification."""
    start = datetime(2023, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 1, tzinfo=UTC)
    config = create_test_walk_forward_config(start_date=start, end_date=end, embargo_bars=100)
    trainer = WalkForwardTrainer(config=config, model_version="test_v1.0.0")

    windows = trainer.generate_temporal_windows()
    assert len(windows) >= 1

    embargo_50h = timedelta(minutes=30 * 100)  # Exactly 50 hours

    for win in windows:
        assert win.window_id >= 1
        # 1. Internal embargo checks
        assert win.internal_embargo_start == win.subtrain_end
        internal_duration = win.internal_embargo_end - win.internal_embargo_start
        assert internal_duration >= embargo_50h, (
            f"Internal embargo duration {internal_duration} is less than 50h"
        )
        assert win.cal_start >= win.internal_embargo_end
        assert win.cal_end > win.cal_start

        # 2. External embargo checks
        assert win.external_embargo_start == win.cal_end
        external_duration = win.external_embargo_end - win.external_embargo_start
        assert external_duration >= embargo_50h, (
            f"External embargo duration {external_duration} is less than 50h"
        )
        assert win.test_start >= win.external_embargo_end
        assert win.test_end > win.test_start

        # 3. Strict causal temporal monotonicity
        assert (
            win.subtrain_start
            < win.subtrain_end
            <= win.internal_embargo_start
            <= win.internal_embargo_end
            <= win.cal_start
            < win.cal_end
            <= win.external_embargo_start
            <= win.external_embargo_end
            <= win.test_start
            < win.test_end
        )

        # 4. Total separation between sub-train and test
        assert win.test_start - win.subtrain_end > embargo_50h


def test_label_dataset_execution() -> None:
    """Test labeling 30m decision bars with forward 1m candle scans."""
    start = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    end = datetime(2024, 1, 2, 0, 0, tzinfo=UTC)
    config = create_test_walk_forward_config(start_date=start, end_date=end, embargo_bars=100)
    trainer = WalkForwardTrainer(config=config, model_version="test_v1.0.0")

    bar_times = [start + timedelta(minutes=30 * i) for i in range(5)]
    entry_prices = [50000.0 + 100.0 * i for i in range(5)]
    vol_gks = [0.005 for _ in range(5)]

    # Mock 1m candles for each 30m bar (60 1m candles each)
    candles_1m_sets: list[list[MicroBar1m]] = []
    for b_idx in range(5):
        candle_set: list[MicroBar1m] = []
        base_p = entry_prices[b_idx]
        for m in range(60):
            candle_set.append(
                MicroBar1m(
                    timestamp_ms=int((bar_times[b_idx] + timedelta(minutes=m)).timestamp() * 1000),
                    open=base_p + m * 2.0,
                    high=base_p + m * 2.0 + 10.0,
                    low=base_p + m * 2.0 - 5.0,
                    close=base_p + m * 2.0 + 5.0,
                    volume=10.0,
                )
            )
        candles_1m_sets.append(candle_set)

    labels = trainer.label_dataset(
        symbol="BTCUSDT",
        bar_times=bar_times,
        entry_prices=entry_prices,
        vol_gks=vol_gks,
        candles_1m_by_time=candles_1m_sets,
    )

    assert len(labels) == 5
    for lbl in labels:
        assert lbl.symbol == "BTCUSDT"
        assert lbl.barrier_hit in (
            BarrierType.UPPER_PROFIT,
            BarrierType.LOWER_LOSS,
            BarrierType.TIMEOUT,
        )
        assert lbl.y_long in (0, 1)
        assert lbl.y_short in (0, 1)


def test_execute_window_causal_evaluation() -> None:
    """Test full execution of a single walk-forward window and invariant enforcement."""
    # Build 200 synthetic 30m decision samples spanning across sub-train, cal, and test
    start_time = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    bar_times = [start_time + timedelta(minutes=30 * i) for i in range(200)]

    # 100 bars subtrain: 0 -> 99
    subtrain_start = bar_times[0]
    subtrain_end = bar_times[79]

    # internal embargo: 80 -> 99 (20 bars)
    internal_embargo_start = bar_times[80]
    internal_embargo_end = bar_times[99]

    # cal: 100 -> 139 (40 bars)
    cal_start = bar_times[100]
    cal_end = bar_times[139]

    # external embargo: 140 -> 159 (20 bars)
    external_embargo_start = bar_times[140]
    external_embargo_end = bar_times[159]

    # test: 160 -> 199 (40 bars)
    test_start = bar_times[160]
    test_end = bar_times[199]

    from trading_bot.domain.entities.ml.ml_predictions import WalkForwardWindowSplit

    window = WalkForwardWindowSplit(
        window_id=1,
        subtrain_start=subtrain_start,
        subtrain_end=subtrain_end,
        internal_embargo_start=internal_embargo_start,
        internal_embargo_end=internal_embargo_end,
        cal_start=cal_start,
        cal_end=cal_end,
        external_embargo_start=external_embargo_start,
        external_embargo_end=external_embargo_end,
        test_start=test_start,
        test_end=test_end,
    )

    # Generate synthetic features
    np.random.seed(42)
    canonical_matrix = np.random.randn(200, 11).tolist()
    subspace_matrix = np.random.randn(200, 6).tolist()

    # Generate synthetic barrier labels
    from trading_bot.domain.entities.ml.barrier_labels import BarrierLabel, BarrierType

    labels: list[BarrierLabel] = []
    for i in range(200):
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
                exit_time=bar_times[min(i + 5, 199)],
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

    config = create_test_walk_forward_config(
        start_date=subtrain_start,
        end_date=test_end,
        embargo_bars=100,
    )
    trainer = WalkForwardTrainer(config=config, model_version="stage4_unit_test_v1")

    metrics, packets = trainer.execute_window(
        window=window,
        symbol="BTCUSDT",
        bar_times=bar_times,
        canonical_features_matrix=canonical_matrix,
        subspace_features_matrix=subspace_matrix,
        barrier_labels=labels,
        feature_names=CANONICAL_FEATURES,
    )

    # Validate window metrics
    assert metrics.window_id == 1
    assert metrics.symbol == "BTCUSDT"
    assert metrics.sample_count == len(packets) == 40  # bars 160..199
    assert 0.0 <= metrics.auc_long <= 1.0
    assert 0.0 <= metrics.auc_short <= 1.0
    assert metrics.brier_long >= 0.0
    assert metrics.brier_short >= 0.0

    # Validate generated packets invariants
    for p in packets:
        assert p.symbol == "BTCUSDT"
        assert p.model_version == "stage4_unit_test_v1"
        assert 0.0 <= p.p_long_cal <= 1.0
        assert 0.0 <= p.p_short_cal <= 1.0
        # Coherence enforcement check: P_long + P_short <= 1.0001
        assert p.p_long_cal + p.p_short_cal <= 1.0001
        assert p.ood_status.mahalanobis_distance >= 0.0
        assert 0.0 <= p.ood_shrinkage <= 1.0
        assert len(p.top_bullish_shap) <= 5
        assert len(p.top_bearish_shap) <= 3
        for bull in p.top_bullish_shap:
            assert bull.shap_value > 0.0
        for bear in p.top_bearish_shap:
            assert bear.shap_value < 0.0


def test_classification_metric_helpers() -> None:
    """Test pure mathematical metric helpers: ROC-AUC, Brier score, and Log-loss."""
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.7, 0.9]

    auc = compute_roc_auc_binary(y_true=y_true, y_scores=y_scores)
    assert auc == 1.0  # Perfectly separated

    bs = compute_brier_score_binary(y_true=y_true, y_scores=y_scores)
    assert bs < 0.10

    ll = compute_log_loss_binary(y_true=y_true, y_scores=y_scores, eps=1e-5)
    assert ll < 0.40

    # Test single-class fallback
    single_auc = compute_roc_auc_binary(y_true=[1, 1, 1], y_scores=[0.5, 0.6, 0.7])
    assert single_auc == 0.50


def test_compute_simulated_trade_return_directional_accuracy() -> None:
    """Verify exact directional return and transaction fee deduction across all 12 barrier scenarios."""
    from trading_bot.adapters.ml_engine.walk_forward_trainer import compute_simulated_trade_return
    from trading_bot.domain.entities.ml.barrier_labels import BarrierLabel, BarrierType

    dt = datetime(2024, 1, 1, tzinfo=UTC)
    cost = 0.0010  # 10 bps

    # 1. UPPER_PROFIT: Long wins (+1.8%), Short loses (-1.8%)
    lbl_upper = BarrierLabel(
        symbol="BTCUSDT",
        bar_time=dt,
        barrier_hit=BarrierType.UPPER_PROFIT,
        exit_time=dt,
        bars_held=5,
        net_return=0.0180,
        sample_weight=1.0,
        y_long=1,
        y_short=0,
        entry_price=50000.0,
        exit_price=50950.0,
        realized_return=0.0190,
        upper_barrier_price=50950.0,
        lower_barrier_price=49050.0,
        volatility_gk=0.01,
        roundtrip_cost=cost,
    )
    assert compute_simulated_trade_return(barrier_label=lbl_upper, is_long=True) == pytest.approx(
        0.0180, abs=1e-6
    )
    assert compute_simulated_trade_return(barrier_label=lbl_upper, is_long=False) == pytest.approx(
        -0.0180, abs=1e-6
    )

    # 2. LOWER_LOSS: Short wins (+1.8%), Long loses (-1.8%)
    lbl_lower = BarrierLabel(
        symbol="BTCUSDT",
        bar_time=dt,
        barrier_hit=BarrierType.LOWER_LOSS,
        exit_time=dt,
        bars_held=5,
        net_return=0.0180,
        sample_weight=1.0,
        y_long=0,
        y_short=1,
        entry_price=50000.0,
        exit_price=49050.0,
        realized_return=-0.0190,
        upper_barrier_price=50950.0,
        lower_barrier_price=49050.0,
        volatility_gk=0.01,
        roundtrip_cost=cost,
    )
    assert compute_simulated_trade_return(barrier_label=lbl_lower, is_long=True) == pytest.approx(
        -0.0180, abs=1e-6
    )
    assert compute_simulated_trade_return(barrier_label=lbl_lower, is_long=False) == pytest.approx(
        0.0180, abs=1e-6
    )

    # 3. TIMEOUT with Positive Drift (+1.2% price move):
    # Long net: +0.0120 - 0.0010 = +0.0110
    # Short net: -0.0120 - 0.0010 = -0.0130
    lbl_to_up = BarrierLabel(
        symbol="BTCUSDT",
        bar_time=dt,
        barrier_hit=BarrierType.TIMEOUT,
        exit_time=dt,
        bars_held=60,
        net_return=0.0110,
        sample_weight=0.5,
        y_long=1,
        y_short=0,
        entry_price=50000.0,
        exit_price=50600.0,
        realized_return=0.0120,
        upper_barrier_price=50950.0,
        lower_barrier_price=49050.0,
        volatility_gk=0.01,
        roundtrip_cost=cost,
    )
    assert compute_simulated_trade_return(barrier_label=lbl_to_up, is_long=True) == pytest.approx(
        0.0110, abs=1e-6
    )
    assert compute_simulated_trade_return(barrier_label=lbl_to_up, is_long=False) == pytest.approx(
        -0.0130, abs=1e-6
    )

    # 4. TIMEOUT with Negative Drift (-1.2% price move):
    # Long net: -0.0120 - 0.0010 = -0.0130
    # Short net: -(-0.0120) - 0.0010 = +0.0110
    lbl_to_down = BarrierLabel(
        symbol="BTCUSDT",
        bar_time=dt,
        barrier_hit=BarrierType.TIMEOUT,
        exit_time=dt,
        bars_held=60,
        net_return=0.0110,
        sample_weight=0.5,
        y_long=0,
        y_short=1,
        entry_price=50000.0,
        exit_price=49400.0,
        realized_return=-0.0120,
        upper_barrier_price=50950.0,
        lower_barrier_price=49050.0,
        volatility_gk=0.01,
        roundtrip_cost=cost,
    )
    assert compute_simulated_trade_return(barrier_label=lbl_to_down, is_long=True) == pytest.approx(
        -0.0130, abs=1e-6
    )
    assert compute_simulated_trade_return(
        barrier_label=lbl_to_down, is_long=False
    ) == pytest.approx(0.0110, abs=1e-6)

    # 5. TIMEOUT in Dead Zone Positive (+0.02% move, fee is 0.10%):
    # Long net: +0.0002 - 0.0010 = -0.0008
    # Short net: -0.0002 - 0.0010 = -0.0012
    lbl_to_dz_pos = BarrierLabel(
        symbol="BTCUSDT",
        bar_time=dt,
        barrier_hit=BarrierType.TIMEOUT,
        exit_time=dt,
        bars_held=60,
        net_return=-0.0008,
        sample_weight=0.0,
        y_long=0,
        y_short=0,
        entry_price=50000.0,
        exit_price=50010.0,
        realized_return=0.0002,
        upper_barrier_price=50950.0,
        lower_barrier_price=49050.0,
        volatility_gk=0.01,
        roundtrip_cost=cost,
    )
    assert compute_simulated_trade_return(
        barrier_label=lbl_to_dz_pos, is_long=True
    ) == pytest.approx(-0.0008, abs=1e-6)
    assert compute_simulated_trade_return(
        barrier_label=lbl_to_dz_pos, is_long=False
    ) == pytest.approx(-0.0012, abs=1e-6)

    # 6. TIMEOUT in Dead Zone Negative (-0.02% move, fee is 0.10%):
    # Long net: -0.0002 - 0.0010 = -0.0012
    # Short net: -(-0.0002) - 0.0010 = -0.0008
    lbl_to_dz_neg = BarrierLabel(
        symbol="BTCUSDT",
        bar_time=dt,
        barrier_hit=BarrierType.TIMEOUT,
        exit_time=dt,
        bars_held=60,
        net_return=-0.0008,
        sample_weight=0.0,
        y_long=0,
        y_short=0,
        entry_price=50000.0,
        exit_price=49990.0,
        realized_return=-0.0002,
        upper_barrier_price=50950.0,
        lower_barrier_price=49050.0,
        volatility_gk=0.01,
        roundtrip_cost=cost,
    )
    assert compute_simulated_trade_return(
        barrier_label=lbl_to_dz_neg, is_long=True
    ) == pytest.approx(-0.0012, abs=1e-6)
    assert compute_simulated_trade_return(
        barrier_label=lbl_to_dz_neg, is_long=False
    ) == pytest.approx(-0.0008, abs=1e-6)
