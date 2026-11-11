# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Unit tests for FileModelRegistry Adapter (Milestone 2, Real Model Persistence).

Validates:
1. LightGBM booster native C++ saving and loading.
2. Ridge prior serialization and exact prediction reproduction.
3. Conformal Subspace OOD calibration matrix serialization and evaluation.
4. Native TreeSHAP Explainer reconstruction.
5. Atomic 2-step promotion from challenger to champion with archive backup.
6. DB audit trail table creation and insertion.
7. Fail-fast error handling for missing/corrupted files and checksum mismatches.
8. Listing registered models across champion, challenger, and archive.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from trading_bot.adapters.ml_engine.conformal_ood_detector import (
    ConformalSubspaceOODDetector,
)
from trading_bot.adapters.ml_engine.constrained_lgbm_model import (
    CANONICAL_FEATURES,
    ConstrainedLGBMModel,
)
from trading_bot.adapters.ml_engine.file_model_registry import (
    FileModelRegistry,
    compute_directory_artifacts_checksum,
)
from trading_bot.adapters.ml_engine.native_treeshap_explainer import (
    NativeTreeSHAPExplainer,
)
from trading_bot.adapters.ml_engine.tabulated_venn_abers import (
    TabulatedVennAbersCalibrator,
)
from trading_bot.domain.entities.ml.ml_configs import (
    CoherenceGateConfig,
    ConformalOODConfig,
    ConstrainedLGBMConfig,
    FastVennAbersConfig,
    LGBMMonotonicConstraints,
    RidgePriorConfig,
    TreeSHAPConfig,
)
from trading_bot.domain.entities.ml.model_registry_entities import (
    ModelArtifactMetadata,
    ModelRole,
)
from trading_bot.domain.exceptions import (
    ChallengerModelNotFoundError,
    ChampionModelNotFoundError,
    ModelArtifactCorruptError,
)
from trading_bot.domain.services.ml.economic_sign_verifier import EconomicSignVerifier
from trading_bot.ports.model_registry_port import ModelRegistryPort


@pytest.fixture
def treeshap_components() -> tuple[NativeTreeSHAPExplainer, TreeSHAPConfig]:
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
    return explainer, config


@pytest.fixture
def trained_model_and_ood() -> tuple[
    ConstrainedLGBMModel, ConformalSubspaceOODDetector, list[float]
]:
    rng = np.random.default_rng(42)
    n_samples = 250
    n_features = len(CANONICAL_FEATURES)

    x_train = rng.normal(0.0, 1.0, size=(n_samples, n_features))
    y_raw_long = (
        0.5 * x_train[:, 0]
        - 0.4 * x_train[:, 1]
        + 0.6 * x_train[:, 4]
        + rng.normal(0.0, 0.2, size=n_samples)
    )
    y_long = (y_raw_long > 0.0).astype(int)

    y_raw_short = -0.5 * x_train[:, 2] - 0.6 * x_train[:, 4] + rng.normal(0.0, 0.2, size=n_samples)
    y_short = (y_raw_short > 0.0).astype(int)
    sample_weights = np.ones(n_samples, dtype=np.float64)

    lgbm_config = ConstrainedLGBMConfig(
        max_depth=3,
        num_leaves=8,
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
        oof_splits=5,
        clip_eps=1e-5,
        embargo_bars_30m=5,
        max_label_holding_bars_30m=3,
    )
    coherence_config = CoherenceGateConfig(
        gamma_conflict=1.0,
        max_combined_probability=1.0,
        min_probability_floor=0.0,
    )

    model = ConstrainedLGBMModel(
        config=lgbm_config,
        ridge_config=ridge_config,
        coherence_config=coherence_config,
        feature_names=CANONICAL_FEATURES,
        random_state=42,
    )
    model.fit(
        train_features=x_train.tolist(),
        y_long=[int(v) for v in y_long],
        y_short=[int(v) for v in y_short],
        sample_weights=[float(v) for v in sample_weights],
        sample_intervals=None,
    )

    # Conformal OOD Detector
    ood_config = ConformalOODConfig(
        feature_names=CANONICAL_FEATURES,
        alpha_significance=0.05,
        kappa_shrinkage=2.0,
        max_condition_number=100.0,
        min_eigenvalue_floor=1e-5,
    )
    ood_detector = ConformalSubspaceOODDetector(config=ood_config)
    ood_detector.fit(
        train_features=x_train[:180].tolist(),
        calibration_features=x_train[180:].tolist(),
    )

    test_sample: list[float] = [float(val) for val in x_train[0]]
    return model, ood_detector, test_sample


def create_metadata(
    model_id: str,
    role: ModelRole,
) -> ModelArtifactMetadata:
    return ModelArtifactMetadata(
        model_id=model_id,
        role=role,
        feature_names=list(CANONICAL_FEATURES),
        trained_at_utc=datetime.now(UTC),
        training_sample_count=250,
        net_sharpe=2.15,
        brier_score=0.175,
        brier_skill_score=0.14,
        max_drawdown_pct=0.065,
        file_checksum="uncalculated",
        backtest_metrics=None,
    )


@pytest.mark.asyncio
async def test_save_model_artifacts_created(
    tmp_path: Path,
    trained_model_and_ood: tuple[ConstrainedLGBMModel, ConformalSubspaceOODDetector, list[float]],
    treeshap_components: tuple[NativeTreeSHAPExplainer, TreeSHAPConfig],
) -> None:
    model, ood_detector, _ = trained_model_and_ood
    explainer, shap_cfg = treeshap_components

    registry = FileModelRegistry(
        base_dir=tmp_path,
        db_pool=None,
        tree_shap_config=shap_cfg,
    )
    assert isinstance(registry, ModelRegistryPort)

    meta = create_metadata(model_id="challenger_001", role=ModelRole.CHALLENGER)
    returned_id = await registry.save_model(
        role=ModelRole.CHALLENGER,
        model=model,
        ood_detector=ood_detector,
        explainer=explainer,
        metadata=meta,
    )

    assert returned_id == "challenger_001"
    target_dir = tmp_path / "challenger"
    assert (target_dir / "long_booster.txt").exists()
    assert (target_dir / "short_booster.txt").exists()
    assert (target_dir / "ridge_prior.json").exists()
    assert (target_dir / "ood_calibration.json").exists()
    assert (target_dir / "metadata.json").exists()

    # Verify checksum was accurately written
    raw_meta = (target_dir / "metadata.json").read_text(encoding="utf-8")
    saved_meta = ModelArtifactMetadata.model_validate_json(raw_meta)
    assert saved_meta.role == ModelRole.CHALLENGER
    assert saved_meta.file_checksum != "uncalculated"
    assert len(saved_meta.file_checksum) == 64
    assert saved_meta.file_checksum == compute_directory_artifacts_checksum(target_dir)


@pytest.mark.asyncio
async def test_load_champion_model_prediction_parity(
    tmp_path: Path,
    trained_model_and_ood: tuple[ConstrainedLGBMModel, ConformalSubspaceOODDetector, list[float]],
    treeshap_components: tuple[NativeTreeSHAPExplainer, TreeSHAPConfig],
) -> None:
    model, ood_detector, test_sample = trained_model_and_ood
    explainer, shap_cfg = treeshap_components

    orig_pred = model.predict(test_sample)
    orig_ood = ood_detector.predict(test_sample)

    registry = FileModelRegistry(
        base_dir=tmp_path,
        db_pool=None,
        tree_shap_config=shap_cfg,
    )

    meta = create_metadata(model_id="champion_001", role=ModelRole.CHAMPION)
    await registry.save_model(
        role=ModelRole.CHAMPION,
        model=model,
        ood_detector=ood_detector,
        explainer=explainer,
        metadata=meta,
    )

    loaded_model, loaded_ood, loaded_shap, loaded_meta = await registry.load_champion_model()

    assert loaded_meta.model_id == "champion_001"
    assert loaded_meta.role == ModelRole.CHAMPION

    # Predict with loaded model
    loaded_pred = loaded_model.predict(test_sample)
    assert abs(orig_pred.p_long - loaded_pred.p_long) < 1e-6
    assert abs(orig_pred.p_short - loaded_pred.p_short) < 1e-6
    assert abs(orig_pred.p_raw_long - loaded_pred.p_raw_long) < 1e-6
    assert abs(orig_pred.p_raw_short - loaded_pred.p_raw_short) < 1e-6
    assert orig_pred.is_coherent == loaded_pred.is_coherent

    # Evaluate with loaded OOD detector
    loaded_ood_res = loaded_ood.predict(test_sample)
    assert abs(orig_ood.mahalanobis_distance - loaded_ood_res.mahalanobis_distance) < 1e-6
    assert abs(orig_ood.gamma_ood - loaded_ood_res.gamma_ood) < 1e-6
    assert orig_ood.is_ood == loaded_ood_res.is_ood

    # Explain with loaded TreeSHAP
    shap_res = loaded_shap.explain(
        feature_vector=test_sample,
        feature_names=CANONICAL_FEATURES,
        model=loaded_model,
        symbol="BTCUSDT",
        bar_time=datetime.now(UTC),
        model_type="long_barrier",
    )
    assert shap_res.symbol == "BTCUSDT"
    assert len(shap_res.top_bullish_drivers) > 0 or len(shap_res.top_bearish_drivers) > 0


@pytest.mark.asyncio
async def test_atomic_promotion_and_archive_backup(
    tmp_path: Path,
    trained_model_and_ood: tuple[ConstrainedLGBMModel, ConformalSubspaceOODDetector, list[float]],
    treeshap_components: tuple[NativeTreeSHAPExplainer, TreeSHAPConfig],
) -> None:
    model, ood_detector, _ = trained_model_and_ood
    explainer, shap_cfg = treeshap_components

    registry = FileModelRegistry(
        base_dir=tmp_path,
        db_pool=None,
        tree_shap_config=shap_cfg,
    )

    # 1. First save initial champion
    meta_champ = create_metadata(model_id="champion_v1", role=ModelRole.CHAMPION)
    await registry.save_model(
        role=ModelRole.CHAMPION,
        model=model,
        ood_detector=ood_detector,
        explainer=explainer,
        metadata=meta_champ,
    )

    # 2. Save superior challenger
    meta_chall = create_metadata(model_id="challenger_v2", role=ModelRole.CHALLENGER)
    await registry.save_model(
        role=ModelRole.CHALLENGER,
        model=model,
        ood_detector=ood_detector,
        explainer=explainer,
        metadata=meta_chall,
    )

    # 3. Promote Challenger to Champion
    record = await registry.promote_challenger_to_champion(
        user_id="lead_quant_42",
        rationale="Challenger achieved Brier Skill Score +0.14 and lower max drawdown.",
    )

    assert record.previous_champion_id == "champion_v1"
    assert record.promoted_challenger_id == "challenger_v2"
    assert record.promoted_by_user_id == "lead_quant_42"
    assert "Brier Skill Score" in record.rationale

    # 4. Verify Champion is now challenger_v2
    _, _, _, current_champ = await registry.load_champion_model()
    assert current_champ.model_id == "challenger_v2"
    assert current_champ.role == ModelRole.CHAMPION

    # 5. Verify Archive contains champion_v1 with role ARCHIVE
    archive_dir = tmp_path / "archive"
    assert archive_dir.exists()
    archived_subdirs = list(archive_dir.iterdir())
    assert len(archived_subdirs) == 1
    assert "champion_v1" in archived_subdirs[0].name

    arch_meta_raw = (archived_subdirs[0] / "metadata.json").read_text(encoding="utf-8")
    arch_meta = ModelArtifactMetadata.model_validate_json(arch_meta_raw)
    assert arch_meta.model_id == "champion_v1"
    assert arch_meta.role == ModelRole.ARCHIVE


@pytest.mark.asyncio
async def test_promotion_records_db_audit_trail(
    tmp_path: Path,
    trained_model_and_ood: tuple[ConstrainedLGBMModel, ConformalSubspaceOODDetector, list[float]],
    treeshap_components: tuple[NativeTreeSHAPExplainer, TreeSHAPConfig],
) -> None:
    model, ood_detector, _ = trained_model_and_ood
    explainer, shap_cfg = treeshap_components

    # Create mock asyncpg pool
    mock_conn = AsyncMock()
    mock_pool = MagicMock()

    class MockPoolContext:
        async def __aenter__(self) -> AsyncMock:
            return mock_conn

        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

    mock_pool.acquire.return_value = MockPoolContext()

    registry = FileModelRegistry(
        base_dir=tmp_path,
        db_pool=mock_pool,
        tree_shap_config=shap_cfg,
    )

    meta_chall = create_metadata(model_id="challenger_db_test", role=ModelRole.CHALLENGER)
    await registry.save_model(
        role=ModelRole.CHALLENGER,
        model=model,
        ood_detector=ood_detector,
        explainer=explainer,
        metadata=meta_chall,
    )

    record = await registry.promote_challenger_to_champion(
        user_id="audit_officer",
        rationale="Audit verification test promotion.",
    )

    assert record.promoted_challenger_id == "challenger_db_test"
    assert mock_conn.execute.call_count == 2
    # Verify SQL query executed
    call_args_list = mock_conn.execute.call_args_list
    create_sql = call_args_list[0][0][0]
    insert_sql = call_args_list[1][0][0]
    assert "CREATE TABLE IF NOT EXISTS ml_model_registry" in create_sql
    assert "INSERT INTO ml_model_registry" in insert_sql


@pytest.mark.asyncio
async def test_fail_fast_on_missing_champion(tmp_path: Path) -> None:
    shap_cfg = TreeSHAPConfig(
        top_k_bullish=5,
        top_k_bearish=3,
        local_efficiency_tol=1e-4,
        sign_tolerance=1e-4,
    )
    registry = FileModelRegistry(
        base_dir=tmp_path,
        db_pool=None,
        tree_shap_config=shap_cfg,
    )

    with pytest.raises(ChampionModelNotFoundError):
        await registry.load_champion_model()


@pytest.mark.asyncio
async def test_fail_fast_on_corrupt_files_and_checksum_mismatch(
    tmp_path: Path,
    trained_model_and_ood: tuple[ConstrainedLGBMModel, ConformalSubspaceOODDetector, list[float]],
    treeshap_components: tuple[NativeTreeSHAPExplainer, TreeSHAPConfig],
) -> None:
    model, ood_detector, _ = trained_model_and_ood
    explainer, shap_cfg = treeshap_components

    registry = FileModelRegistry(
        base_dir=tmp_path,
        db_pool=None,
        tree_shap_config=shap_cfg,
    )

    meta = create_metadata(model_id="corrupt_test", role=ModelRole.CHAMPION)
    await registry.save_model(
        role=ModelRole.CHAMPION,
        model=model,
        ood_detector=ood_detector,
        explainer=explainer,
        metadata=meta,
    )

    champ_dir = tmp_path / "champion"

    # 1. Test tampering booster content triggers checksum mismatch
    booster_path = champ_dir / "long_booster.txt"
    booster_path.write_text("TAMPERED_CONTENT", encoding="utf-8")

    with pytest.raises(ModelArtifactCorruptError, match="checksum mismatch"):
        await registry.load_champion_model()

    # 2. Test missing booster raises corrupt error
    booster_path.unlink()
    with pytest.raises(ModelArtifactCorruptError, match="Missing required model artifact file"):
        await registry.load_champion_model()

    # 3. Test malformed metadata.json
    (champ_dir / "metadata.json").write_text("{invalid_json", encoding="utf-8")
    with pytest.raises(ModelArtifactCorruptError, match="Corrupted model metadata"):
        await registry.load_champion_model()


@pytest.mark.asyncio
async def test_fail_fast_on_missing_challenger_for_promotion(tmp_path: Path) -> None:
    shap_cfg = TreeSHAPConfig(
        top_k_bullish=5,
        top_k_bearish=3,
        local_efficiency_tol=1e-4,
        sign_tolerance=1e-4,
    )
    registry = FileModelRegistry(
        base_dir=tmp_path,
        db_pool=None,
        tree_shap_config=shap_cfg,
    )

    with pytest.raises(ChallengerModelNotFoundError):
        await registry.promote_challenger_to_champion(
            user_id="user1",
            rationale="No challenger on disk",
        )


@pytest.mark.asyncio
async def test_list_registered_models(
    tmp_path: Path,
    trained_model_and_ood: tuple[ConstrainedLGBMModel, ConformalSubspaceOODDetector, list[float]],
    treeshap_components: tuple[NativeTreeSHAPExplainer, TreeSHAPConfig],
) -> None:
    model, ood_detector, _ = trained_model_and_ood
    explainer, shap_cfg = treeshap_components

    registry = FileModelRegistry(
        base_dir=tmp_path,
        db_pool=None,
        tree_shap_config=shap_cfg,
    )

    # Initially empty
    empty_list = await registry.list_registered_models()
    assert len(empty_list) == 0

    # Save champion
    await registry.save_model(
        role=ModelRole.CHAMPION,
        model=model,
        ood_detector=ood_detector,
        explainer=explainer,
        metadata=create_metadata("champ_1", ModelRole.CHAMPION),
    )

    # Save challenger
    await registry.save_model(
        role=ModelRole.CHALLENGER,
        model=model,
        ood_detector=ood_detector,
        explainer=explainer,
        metadata=create_metadata("chall_1", ModelRole.CHALLENGER),
    )

    # Save archive
    await registry.save_model(
        role=ModelRole.ARCHIVE,
        model=model,
        ood_detector=ood_detector,
        explainer=explainer,
        metadata=create_metadata("arch_1", ModelRole.ARCHIVE),
    )

    all_models = await registry.list_registered_models()
    assert len(all_models) == 3
    model_ids = {m.model_id for m in all_models}
    assert model_ids == {"champ_1", "chall_1", "arch_1"}


@pytest.mark.asyncio
async def test_save_and_promote_venn_abers_calibrators_roundtrip(
    tmp_path: Path,
    trained_model_and_ood: tuple[ConstrainedLGBMModel, ConformalSubspaceOODDetector, list[float]],
    treeshap_components: tuple[NativeTreeSHAPExplainer, TreeSHAPConfig],
) -> None:
    """Verify Venn-Abers calibrators serialize cleanly, survive Challenger->Champion promotion, and reproduce predictions."""
    model, ood_detector, _ = trained_model_and_ood
    explainer, shap_cfg = treeshap_components

    va_cfg = FastVennAbersConfig(
        grid_points=500,
        prob_clip_min=0.001,
        prob_clip_max=0.999,
        denom_eps=1e-6,
        uncertainty_threshold=0.15,
        calibration_floor=0.001,
    )
    va_long = TabulatedVennAbersCalibrator(config=va_cfg)
    va_short = TabulatedVennAbersCalibrator(config=va_cfg)
    scores = np.linspace(0.1, 0.9, 100).tolist()
    y_l = [1 if s > 0.55 else 0 for s in scores]
    y_s = [1 if s < 0.45 else 0 for s in scores]
    va_long.fit(scores=scores, labels=y_l)
    va_short.fit(scores=scores, labels=y_s)

    registry = FileModelRegistry(
        base_dir=tmp_path,
        db_pool=None,
        tree_shap_config=shap_cfg,
    )
    assert await registry.load_champion_calibrators() is None

    await registry.save_model(
        role=ModelRole.CHALLENGER,
        model=model,
        ood_detector=ood_detector,
        explainer=explainer,
        metadata=create_metadata("chall_va_001", ModelRole.CHALLENGER),
    )
    await registry.save_calibrators(
        role=ModelRole.CHALLENGER,
        calibrator_long=va_long,
        calibrator_short=va_short,
    )
    assert (tmp_path / "challenger" / "venn_abers_cal.json").exists()

    await registry.promote_challenger_to_champion(
        user_id="tester",
        rationale="Verify atomic promotion includes Venn-Abers calibrators",
    )
    loaded = await registry.load_champion_calibrators()
    assert loaded is not None
    loaded_l, loaded_s = loaded
    for probe in (0.35, 0.52, 0.74):
        orig_res = va_long.calibrate(probe)
        load_res = loaded_l.calibrate(probe)
        assert abs(orig_res.p_calibrated - load_res.p_calibrated) < 1e-12
        assert abs(orig_res.uncertainty_width - load_res.uncertainty_width) < 1e-12
        assert (
            abs(va_short.calibrate(probe).p_calibrated - loaded_s.calibrate(probe).p_calibrated)
            < 1e-12
        )
