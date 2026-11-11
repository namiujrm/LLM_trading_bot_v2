# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Unit tests for ModelTrainingService (Milestone 3, Real Model Persistence).

Tests:
1. bootstrap_initial_champion: returns existing champion if present and valid.
2. bootstrap_initial_champion: auto-bootstraps genuine ConstrainedLGBMModel and OOD detector when missing.
3. bootstrap_initial_champion: auto-bootstraps when artifacts are corrupt.
4. train_challenger_model: trains challenger in "fast" mode and saves to registry under ModelRole.CHALLENGER.
5. train_challenger_model: trains challenger in "full" mode with custom features.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import polars as pl
import pytest

from trading_bot.adapters.ml_engine.constrained_lgbm_model import CANONICAL_FEATURES
from trading_bot.adapters.ml_engine.file_model_registry import FileModelRegistry
from trading_bot.adapters.ml_engine.model_trainer_adapter import LGBMModelTrainerAdapter
from trading_bot.application.services.model_training_service import ModelTrainingService
from trading_bot.domain.entities.ml.ml_configs import TreeSHAPConfig
from trading_bot.domain.entities.ml.model_registry_entities import ModelRole
from trading_bot.ports.market_data_port import MarketDataPort


@pytest.fixture
def test_setup(
    tmp_path: Path,
) -> tuple[FileModelRegistry, LGBMModelTrainerAdapter, ModelTrainingService]:
    tree_shap_cfg = TreeSHAPConfig(
        top_k_bullish=5,
        top_k_bearish=3,
        local_efficiency_tol=1e-4,
        sign_tolerance=1e-4,
    )
    registry = FileModelRegistry(
        base_dir=tmp_path,
        db_pool=None,
        tree_shap_config=tree_shap_cfg,
    )
    trainer = LGBMModelTrainerAdapter.create_canonical(tree_shap_config=tree_shap_cfg)
    service = ModelTrainingService(
        model_registry=registry,
        model_trainer=trainer,
        canonical_features=list(CANONICAL_FEATURES),
        default_sample_count=150,
        dataset_path=None,
    )
    return registry, trainer, service


@pytest.mark.asyncio
async def test_bootstrap_initial_champion_when_missing(
    tmp_path: Path,
    test_setup: tuple[FileModelRegistry, LGBMModelTrainerAdapter, ModelTrainingService],
) -> None:
    registry, _, service = test_setup

    # Champion does not exist initially
    champ_dir = tmp_path / "champion"
    assert not champ_dir.exists()

    meta = await service.bootstrap_initial_champion(
        market_data_port=None,
        historical_bars=None,
    )

    assert meta.role == ModelRole.CHAMPION
    assert "champion_bootstrap_" in meta.model_id
    assert len(meta.file_checksum) == 64
    assert meta.training_sample_count > 0

    # Verify physical artifacts on disk
    assert (champ_dir / "long_booster.txt").exists()
    assert (champ_dir / "short_booster.txt").exists()
    assert (champ_dir / "ridge_prior.json").exists()
    assert (champ_dir / "ood_calibration.json").exists()
    assert (champ_dir / "metadata.json").exists()

    # Verify load_champion_model works and returns genuine models
    model, ood, explainer, loaded_meta = await registry.load_champion_model()
    assert loaded_meta.model_id == meta.model_id
    assert explainer is not None

    # Verify predictions run without error
    test_vec = [0.1] * len(CANONICAL_FEATURES)
    ood_res = ood.predict(test_vec)
    assert isinstance(ood_res.is_ood, bool)
    pred = model.predict(test_vec)
    assert 0.0 <= pred.p_long <= 1.0
    assert 0.0 <= pred.p_short <= 1.0


@pytest.mark.asyncio
async def test_bootstrap_returns_existing_champion(
    test_setup: tuple[FileModelRegistry, LGBMModelTrainerAdapter, ModelTrainingService],
) -> None:
    _, _, service = test_setup

    # 1. First bootstrap creates champion
    meta1 = await service.bootstrap_initial_champion(
        market_data_port=None,
        historical_bars=None,
    )

    # 2. Second call should return existing metadata without retraining
    meta2 = await service.bootstrap_initial_champion(
        market_data_port=None,
        historical_bars=None,
    )

    assert meta1.model_id == meta2.model_id
    assert meta1.file_checksum == meta2.file_checksum


@pytest.mark.asyncio
async def test_bootstrap_recovers_from_corrupt_champion(
    tmp_path: Path,
    test_setup: tuple[FileModelRegistry, LGBMModelTrainerAdapter, ModelTrainingService],
) -> None:
    _, _, service = test_setup

    # 1. Bootstrap valid champion
    meta1 = await service.bootstrap_initial_champion(
        market_data_port=None,
        historical_bars=None,
    )
    assert meta1.role == ModelRole.CHAMPION

    # 2. Corrupt booster file
    champ_dir = tmp_path / "champion"
    (champ_dir / "long_booster.txt").write_text("CORRUPTED_WEIGHTS", encoding="utf-8")

    # 3. Bootstrap detects corruption and re-bootstraps fresh champion
    meta2 = await service.bootstrap_initial_champion(
        market_data_port=None,
        historical_bars=None,
    )

    assert meta2.role == ModelRole.CHAMPION
    assert len(meta2.file_checksum) == 64


@pytest.mark.asyncio
async def test_train_challenger_model_fast_and_full(
    tmp_path: Path,
    test_setup: tuple[FileModelRegistry, LGBMModelTrainerAdapter, ModelTrainingService],
) -> None:
    _registry, _, service = test_setup

    # 1. Train fast challenger
    meta_fast, model_fast, ood_fast, _ = await service.train_challenger_model(
        mode="fast",
        market_data_port=None,
        historical_bars=None,
        feature_names=None,
    )

    assert meta_fast.role == ModelRole.CHALLENGER
    assert "challenger_fast_" in meta_fast.model_id
    assert (tmp_path / "challenger" / "long_booster.txt").exists()
    test_vec = [0.1] * len(CANONICAL_FEATURES)
    pred_fast = model_fast.predict(test_vec)
    assert 0.0 <= pred_fast.p_long <= 1.0
    ood_res_fast = ood_fast.predict(test_vec)
    assert isinstance(ood_res_fast.is_ood, bool)

    # 2. Train full challenger with mock market data port
    mock_market_data = AsyncMock(spec=MarketDataPort)
    mock_market_data.fetch_multi_tf_bars.return_value = []

    meta_full, _, _, _ = await service.train_challenger_model(
        mode="full",
        market_data_port=mock_market_data,
        historical_bars=None,
        feature_names=list(CANONICAL_FEATURES),
    )

    assert meta_full.role == ModelRole.CHALLENGER
    assert "challenger_full_" in meta_full.model_id
    assert meta_full.net_sharpe >= meta_fast.net_sharpe


@pytest.mark.asyncio
async def test_bootstrap_upgrade_from_legacy_and_rolling_window_parquet(tmp_path: Path) -> None:
    """Verify legacy toy model auto-upgrade, float64 parquet loading, and FIFO rolling window."""
    tree_shap_cfg = TreeSHAPConfig(
        top_k_bullish=5,
        top_k_bearish=3,
        local_efficiency_tol=1e-4,
        sign_tolerance=1e-4,
    )
    registry = FileModelRegistry(
        base_dir=tmp_path / "models",
        db_pool=None,
        tree_shap_config=tree_shap_cfg,
    )
    trainer = LGBMModelTrainerAdapter.create_canonical(tree_shap_config=tree_shap_cfg)

    # 1. Create a legacy toy Champion first (<1000 samples)
    legacy_service = ModelTrainingService(
        model_registry=registry,
        model_trainer=trainer,
        canonical_features=list(CANONICAL_FEATURES),
        default_sample_count=250,
        dataset_path=None,
    )
    legacy_meta = await legacy_service.bootstrap_initial_champion(
        market_data_port=None,
        historical_bars=None,
    )
    assert legacy_meta.training_sample_count == 180

    # 2. Construct realistic float64 multi-symbol parquet dataset (2 symbols x 1000 bars = 2000 rows)
    rng = np.random.default_rng(123)
    n_per_sym = 1000
    base_time = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    symbols: list[str] = []
    timestamps: list[datetime] = []
    for i in range(n_per_sym):
        ts = base_time + timedelta(minutes=30 * i)
        symbols.extend(["BTCUSDT", "ETHUSDT"])
        timestamps.extend([ts, ts])

    n_total = len(symbols)
    feat_data: dict[str, object] = {
        "symbol": symbols,
        "timestamp": timestamps,
        "y_long": (rng.random(n_total) > 0.5).astype(int).tolist(),
        "y_short": (rng.random(n_total) > 0.5).astype(int).tolist(),
        "de_prado_weight": np.full(n_total, 1.0, dtype=np.float64).tolist(),
        "net_return": np.full(n_total, 0.008, dtype=np.float64).tolist(),
        "realized_return": rng.normal(0.0, 0.005, size=n_total).tolist(),
        "roundtrip_cost": np.full(n_total, 0.001, dtype=np.float64).tolist(),
        "volatility_gk": np.full(n_total, 0.004, dtype=np.float64).tolist(),
        "barrier_hit": ["UPPER_PROFIT" if i % 2 == 0 else "LOWER_LOSS" for i in range(n_total)],
    }
    for feat in CANONICAL_FEATURES:
        feat_data[feat] = rng.normal(0.0, 1.0, size=n_total).astype(np.float64).tolist()

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_file = data_dir / "36m_features_labels_all_4pairs.parquet"
    pl.DataFrame(feat_data).write_parquet(dataset_file)

    # 3. Bootstrapping with real dataset must automatically upgrade the 180-sample toy model
    real_service = ModelTrainingService(
        model_registry=registry,
        model_trainer=trainer,
        canonical_features=list(CANONICAL_FEATURES),
        default_sample_count=250,
        dataset_path=str(dataset_file),
    )
    upgraded_meta = await real_service.bootstrap_initial_champion(
        market_data_port=None,
        historical_bars=None,
    )
    assert upgraded_meta.training_sample_count == 1440  # 72% of 2000 rows
    assert upgraded_meta.brier_score > 0.0

    # 4. Verify roll_dataset_window FIFO mechanics with weekly updates
    weekly_rows = 20  # 10 new bars per symbol
    w_symbols: list[str] = []
    w_timestamps: list[datetime] = []
    for i in range(10):
        ts = base_time + timedelta(minutes=30 * (n_per_sym + i))
        w_symbols.extend(["BTCUSDT", "ETHUSDT"])
        w_timestamps.extend([ts, ts])
    weekly_data: dict[str, object] = {
        "symbol": w_symbols,
        "timestamp": w_timestamps,
        "y_long": (rng.random(weekly_rows) > 0.5).astype(int).tolist(),
        "y_short": (rng.random(weekly_rows) > 0.5).astype(int).tolist(),
        "de_prado_weight": np.full(weekly_rows, 1.0, dtype=np.float64).tolist(),
        "net_return": np.full(weekly_rows, 0.008, dtype=np.float64).tolist(),
        "realized_return": rng.normal(0.0, 0.005, size=weekly_rows).tolist(),
        "roundtrip_cost": np.full(weekly_rows, 0.001, dtype=np.float64).tolist(),
        "volatility_gk": np.full(weekly_rows, 0.004, dtype=np.float64).tolist(),
        "barrier_hit": ["UPPER_PROFIT"] * weekly_rows,
    }
    for feat in CANONICAL_FEATURES:
        weekly_data[feat] = rng.normal(0.0, 1.0, size=weekly_rows).astype(np.float64).tolist()

    weekly_file = data_dir / "weekly_updates.parquet"
    pl.DataFrame(weekly_data).write_parquet(weekly_file)

    rolled_total = trainer.roll_dataset_window(
        base_parquet_path=str(dataset_file),
        new_bars_parquet_path=str(weekly_file),
        max_bars_per_symbol=1000,
        output_parquet_path=str(dataset_file),
    )
    assert rolled_total == 2000
    rolled_df = pl.read_parquet(dataset_file)
    latest_ts = rolled_df["timestamp"].max()
    assert latest_ts == w_timestamps[-1]

    # 5. Train challenger on rolled dataset
    chall_meta, _, _, _ = await real_service.train_challenger_model(
        mode="full",
        market_data_port=None,
        historical_bars=None,
        feature_names=None,
    )
    assert chall_meta.training_sample_count == 1440
