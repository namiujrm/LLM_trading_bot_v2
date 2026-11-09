"""Integration test evaluating walk-forward execution, zero look-ahead bias, and database persistence."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trading_bot.adapters.ml_engine.constrained_lgbm_model import CANONICAL_FEATURES
from trading_bot.adapters.ml_engine.walk_forward_trainer import WalkForwardTrainer
from trading_bot.adapters.persistence.database import DatabaseManager
from trading_bot.adapters.persistence.ml_prediction_repo import MLPredictionRepository
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
from trading_bot.infrastructure.config_schemas import DatabaseConfig


def get_integration_db_config() -> DatabaseConfig:
    """Return explicit DatabaseConfig for integration testing."""
    return DatabaseConfig(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        db_name=os.getenv("POSTGRES_DB", "trading_bot_db"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        max_connections=int(os.getenv("POSTGRES_MAX_CONNECTIONS", "5")),
    )


def create_integration_walk_forward_config(
    start_date: datetime,
    end_date: datetime,
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
        embargo_bars_30m=100,
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
            embargo_bars_30m=100,
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
            embargo_bars_30m=100,
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


@pytest.mark.asyncio
async def test_end_to_end_walk_forward_and_timescaledb_persistence() -> None:
    """Integration test: Walk-Forward execution across 2 windows and persistence into TimescaleDB."""
    db_cfg = get_integration_db_config()
    db = DatabaseManager(db_cfg)
    await db.initialize()
    test_symbol = f"BTC_M46_{int(time.time()) % 10000}"

    try:
        is_healthy = await db.healthcheck()
        assert is_healthy is True

        repo = MLPredictionRepository(db_manager=db, batch_chunk_size=50)

        # Build synthetic 30m dataset: 300 bars
        start_time = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        bar_times = [start_time + timedelta(minutes=30 * i) for i in range(300)]

        # Window 1: subtrain 0..79, cal 100..139, test 160..199
        # Window 2: subtrain 80..159, cal 180..219, test 240..279
        w1 = WalkForwardWindowSplit(
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

        w2 = WalkForwardWindowSplit(
            window_id=2,
            subtrain_start=bar_times[80],
            subtrain_end=bar_times[159],
            internal_embargo_start=bar_times[160],
            internal_embargo_end=bar_times[179],
            cal_start=bar_times[180],
            cal_end=bar_times[219],
            external_embargo_start=bar_times[220],
            external_embargo_end=bar_times[239],
            test_start=bar_times[240],
            test_end=bar_times[279],
        )

        np.random.seed(42)
        canonical_matrix = np.random.randn(300, 11).tolist()
        subspace_matrix = np.random.randn(300, 6).tolist()

        labels: list[BarrierLabel] = []
        for i in range(300):
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
                    symbol=test_symbol,
                    bar_time=bar_times[i],
                    barrier_hit=b_type,
                    exit_time=bar_times[min(i + 5, 299)],
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

        wf_config = create_integration_walk_forward_config(
            start_date=bar_times[0],
            end_date=bar_times[-1],
        )
        trainer = WalkForwardTrainer(
            config=wf_config,
            model_version="stage4_integration_v1.0.0",
        )

        # Execute 2-window walk-forward
        summary, all_packets = trainer.run_walk_forward(
            windows=[w1, w2],
            symbol=test_symbol,
            bar_times=bar_times,
            canonical_features_matrix=canonical_matrix,
            subspace_features_matrix=subspace_matrix,
            barrier_labels=labels,
            feature_names=CANONICAL_FEATURES,
        )

        assert summary.total_windows == 2
        assert summary.total_samples == 80  # 40 per window
        assert len(all_packets) == 80

        # Persist predictions into live TimescaleDB
        saved_count = await trainer.persist_predictions(repo=repo, packets=all_packets)
        assert saved_count == 80

        # Retrieve latest prediction packet from database
        latest_packet = await repo.get_prediction_packet(symbol=test_symbol)
        assert latest_packet is not None
        assert latest_packet.symbol == test_symbol
        assert latest_packet.model_version == "stage4_integration_v1.0.0"
        assert latest_packet.bar_time == all_packets[-1].bar_time

        # Retrieve prediction packets range
        range_packets = await repo.get_prediction_packets_range(
            symbol=test_symbol,
            start_time=w1.test_start,
            end_time=w1.test_end,
        )
        assert len(range_packets) == 40
        assert range_packets[0].bar_time == w1.test_start
        assert range_packets[-1].bar_time == w1.test_end

    finally:
        async with db.transaction() as conn:
            await conn.execute(
                "DELETE FROM quant_feature_vectors_30m WHERE symbol = $1;", test_symbol
            )
        await db.close()


def test_zero_look_ahead_bias_embargo_and_causal_ordering() -> None:
    """Integration test: Verify mathematical proof of zero look-ahead bias across all windows."""
    start = datetime(2023, 1, 1, tzinfo=UTC)
    end = datetime(2025, 12, 31, tzinfo=UTC)
    config = create_integration_walk_forward_config(start_date=start, end_date=end)
    trainer = WalkForwardTrainer(config=config, model_version="stage4_v1.0.0")

    windows = trainer.generate_temporal_windows()
    assert len(windows) >= 4

    embargo_min_duration = timedelta(minutes=30 * 100)

    for win in windows:
        # Assert test window is strictly in the future of train window
        assert win.test_start > win.cal_end
        assert win.test_start - win.cal_end >= embargo_min_duration, (
            f"External embargo violation: test_start {win.test_start} - cal_end {win.cal_end} "
            f"< {embargo_min_duration}"
        )

        # Assert subtrain and calibration are strictly separated by internal embargo
        assert win.cal_start > win.subtrain_end
        assert win.cal_start - win.subtrain_end >= embargo_min_duration, (
            f"Internal embargo violation: cal_start {win.cal_start} - subtrain_end {win.subtrain_end} "
            f"< {embargo_min_duration}"
        )

        # Assert zero overlap between any subtrain sample and test sample
        assert win.subtrain_end < win.test_start
