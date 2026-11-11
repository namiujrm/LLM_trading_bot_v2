# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Unit tests for ModelPromotionService (Milestone 3, Real Model Persistence).

Tests:
1. promote_and_reload: promotes challenger to champion with archive backup and hot-reload.
2. promote_and_reload: dispatches notification to TelegramNotifierPort.
3. promote_and_reload: raises ChallengerModelNotFoundError when no challenger exists.
4. get_champion_metadata: returns active champion metadata.
5. list_registered_models: lists models across roles.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from trading_bot.adapters.ml_engine.constrained_lgbm_model import CANONICAL_FEATURES
from trading_bot.adapters.ml_engine.file_model_registry import FileModelRegistry
from trading_bot.adapters.ml_engine.model_trainer_adapter import LGBMModelTrainerAdapter
from trading_bot.application.services.model_promotion_service import ModelPromotionService
from trading_bot.application.services.model_training_service import ModelTrainingService
from trading_bot.domain.entities.ml.ml_configs import TreeSHAPConfig
from trading_bot.domain.entities.ml.model_registry_entities import ModelRole
from trading_bot.domain.exceptions import ChallengerModelNotFoundError
from trading_bot.ports.telegram_port import TelegramNotifierPort


@pytest.fixture
def promotion_setup(
    tmp_path: Path,
) -> tuple[
    FileModelRegistry,
    ModelTrainingService,
    ModelPromotionService,
    MagicMock,
    AsyncMock,
]:
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
    training_service = ModelTrainingService(
        model_registry=registry,
        model_trainer=trainer,
        canonical_features=list(CANONICAL_FEATURES),
        default_sample_count=120,
    )

    mock_runner = MagicMock()
    mock_runner.reload_active_model = MagicMock()

    mock_notifier = AsyncMock(spec=TelegramNotifierPort)
    mock_notifier.send_text_message.return_value = True

    promotion_service = ModelPromotionService(
        model_registry=registry,
        live_paper_runner=mock_runner,
        notifier=mock_notifier,
    )

    return registry, training_service, promotion_service, mock_runner, mock_notifier


@pytest.mark.asyncio
async def test_promote_and_reload_success(
    tmp_path: Path,
    promotion_setup: tuple[
        FileModelRegistry,
        ModelTrainingService,
        ModelPromotionService,
        MagicMock,
        AsyncMock,
    ],
) -> None:
    registry, training_service, promotion_service, mock_runner, mock_notifier = promotion_setup

    # 1. Bootstrap initial Champion
    meta_champ = await training_service.bootstrap_initial_champion(
        market_data_port=None,
        historical_bars=None,
    )

    # 2. Train superior Challenger
    meta_chall, _, _, _ = await training_service.train_challenger_model(
        mode="fast",
        market_data_port=None,
        historical_bars=None,
        feature_names=None,
    )

    # 3. Execute promotion
    record = await promotion_service.promote_and_reload(
        user_id="lead_quant",
        rationale="Challenger achieved lower drawdown and higher Brier Skill Score.",
    )

    assert record.previous_champion_id == meta_champ.model_id
    assert record.promoted_challenger_id == meta_chall.model_id
    assert record.promoted_by_user_id == "lead_quant"

    # 4. Verify runner.reload_active_model was called with real components
    mock_runner.reload_active_model.assert_called_once()
    call_kwargs = mock_runner.reload_active_model.call_args[1]
    assert call_kwargs["new_metadata"].model_id == meta_chall.model_id

    # 5. Verify Telegram alert dispatched
    mock_notifier.send_text_message.assert_called_once()
    alert_text = mock_notifier.send_text_message.call_args[0][0]
    assert meta_chall.model_id in alert_text
    assert "PROMOTION COMPLETE" in alert_text

    # 6. Verify Champion directory has Challenger weights
    _, _, _, current_champ = await registry.load_champion_model()
    assert current_champ.model_id == meta_chall.model_id
    assert current_champ.role == ModelRole.CHAMPION

    # 7. Verify archive directory has old champion
    archive_dir = tmp_path / "archive"
    assert archive_dir.exists()
    archived_dirs = list(archive_dir.iterdir())
    assert len(archived_dirs) == 1
    assert meta_champ.model_id in archived_dirs[0].name


@pytest.mark.asyncio
async def test_promote_raises_when_no_challenger(
    promotion_setup: tuple[
        FileModelRegistry,
        ModelTrainingService,
        ModelPromotionService,
        MagicMock,
        AsyncMock,
    ],
) -> None:
    _, _, promotion_service, _, _ = promotion_setup

    with pytest.raises(ChallengerModelNotFoundError):
        await promotion_service.promote_and_reload(
            user_id="user1",
            rationale="No challenger on disk",
        )


@pytest.mark.asyncio
async def test_get_champion_metadata_and_list_models(
    promotion_setup: tuple[
        FileModelRegistry,
        ModelTrainingService,
        ModelPromotionService,
        MagicMock,
        AsyncMock,
    ],
) -> None:
    _, training_service, promotion_service, _, _ = promotion_setup

    # Bootstrap champion
    meta_champ = await training_service.bootstrap_initial_champion(
        market_data_port=None,
        historical_bars=None,
    )

    # Train challenger
    await training_service.train_challenger_model(
        mode="fast",
        market_data_port=None,
        historical_bars=None,
        feature_names=None,
    )

    fetched_meta = await promotion_service.get_champion_metadata()
    assert fetched_meta.model_id == meta_champ.model_id

    all_models = await promotion_service.list_registered_models()
    assert len(all_models) >= 2
    roles = {m.role for m in all_models}
    assert ModelRole.CHAMPION in roles
    assert ModelRole.CHALLENGER in roles
