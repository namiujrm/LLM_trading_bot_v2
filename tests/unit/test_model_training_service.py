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

from pathlib import Path
from unittest.mock import AsyncMock

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
