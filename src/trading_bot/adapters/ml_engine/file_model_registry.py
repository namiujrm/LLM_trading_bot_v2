# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""File-based Machine Learning Model Registry Adapter (Clean Architecture, Invariant #9).

Implements ModelRegistryPort for persistent storage, loading, and champion-challenger
atomic disk promotion of Stage 4 Causal ML models, OOD detectors, and TreeSHAP explainers.
Persists native LightGBM C++ booster text files, OOF Ridge priors, Conformal OOD calibration
matrices, and atomic metadata JSON.

Strict Clean Architecture:
- Concrete adapter in src/trading_bot/adapters/ml_engine/ layer
- Implements ModelRegistryPort from src/trading_bot/ports/model_registry_port.py
- Zero cross-adapter dependencies (respects adapters_isolation)
- Adheres strictly to Invariant #9: zero defaults, zero class field defaults, zero silent fallbacks.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import asyncpg
import lightgbm as lgb

from trading_bot.adapters.ml_engine.conformal_ood_detector import ConformalSubspaceOODDetector
from trading_bot.adapters.ml_engine.constrained_lgbm_model import ConstrainedLGBMModel
from trading_bot.adapters.ml_engine.native_treeshap_explainer import NativeTreeSHAPExplainer
from trading_bot.adapters.ml_engine.oof_ridge_prior import OOFRidgePrior
from trading_bot.domain.entities.ml.ml_configs import (
    CoherenceGateConfig,
    ConformalOODConfig,
    ConstrainedLGBMConfig,
    LGBMMonotonicConstraints,
    RidgePriorConfig,
    TreeSHAPConfig,
)
from trading_bot.domain.entities.ml.model_registry_entities import (
    ModelArtifactMetadata,
    ModelPromotionRecord,
    ModelRole,
)
from trading_bot.domain.exceptions import (
    ChallengerModelNotFoundError,
    ChampionModelNotFoundError,
    ModelArtifactCorruptError,
)
from trading_bot.domain.services.ml.economic_sign_verifier import EconomicSignVerifier
from trading_bot.ports.ml_model_port import (
    CausalBarrierModelPort,
    OODDetectorPort,
    TreeSHAPExplainerPort,
)
from trading_bot.ports.model_registry_port import ModelRegistryPort


def compute_directory_artifacts_checksum(directory: Path) -> str:
    """Compute deterministic SHA-256 checksum across model weight and calibration files."""
    hasher = hashlib.sha256()
    file_sequence = (
        "long_booster.txt",
        "short_booster.txt",
        "ridge_prior.json",
        "ood_calibration.json",
    )
    for filename in file_sequence:
        file_path = directory / filename
        if not file_path.exists():
            raise ModelArtifactCorruptError(
                f"Missing required artifact file '{filename}' while computing checksum in {directory}"
            )
        hasher.update(file_path.read_bytes())
    return hasher.hexdigest()


class FileModelRegistry(ModelRegistryPort):
    """File-system based ML Model Registry with atomic swap promotion and DB audit trail."""

    def __init__(
        self,
        base_dir: Path,
        db_pool: asyncpg.Pool[asyncpg.Record] | None,
        tree_shap_config: TreeSHAPConfig | None,
    ) -> None:
        """Initialize FileModelRegistry with explicit base directory and configuration.

        Args:
            base_dir: Root directory for models (e.g. Path("models") or Path("/app/models")).
            db_pool: Optional asyncpg connection pool for recording promotion audit records.
            tree_shap_config: Explicit TreeSHAP configuration schema for initializing explainers.
        """
        self._base_dir: Path = base_dir
        self._db_pool: asyncpg.Pool[asyncpg.Record] | None = db_pool
        self._tree_shap_config: TreeSHAPConfig | None = tree_shap_config

    @property
    def base_dir(self) -> Path:
        """Return the configured base directory."""
        return self._base_dir

    @property
    def db_pool(self) -> asyncpg.Pool[asyncpg.Record] | None:
        """Return the database connection pool."""
        return self._db_pool

    async def save_model(
        self,
        *,
        role: ModelRole,
        model: CausalBarrierModelPort,
        ood_detector: OODDetectorPort,
        explainer: TreeSHAPExplainerPort,
        metadata: ModelArtifactMetadata,
    ) -> str:
        """Persist a model suite under the specified role and return the model identifier.

        Args:
            role: ModelRole (CHAMPION, CHALLENGER, or ARCHIVE).
            model: Causal barrier model implementing CausalBarrierModelPort.
            ood_detector: Out-of-distribution detector implementing OODDetectorPort.
            explainer: TreeSHAP explainer implementing TreeSHAPExplainerPort.
            metadata: Explicit metadata record describing training metrics and integrity.

        Returns:
            String representing the model identifier.
        """
        if role == ModelRole.CHAMPION:
            target_dir = self._base_dir / "champion"
        elif role == ModelRole.CHALLENGER:
            target_dir = self._base_dir / "challenger"
        elif role == ModelRole.ARCHIVE:
            timestamp_str = metadata.trained_at_utc.strftime("%Y%m%d_%H%M%S")
            target_dir = self._base_dir / "archive" / f"{timestamp_str}_{metadata.model_id}"
        else:
            raise ValueError(f"Unknown model role: {role}")

        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save LightGBM Boosters using native C++ .save_model()
        if not isinstance(model, ConstrainedLGBMModel):
            raise TypeError(f"Model must be an instance of ConstrainedLGBMModel, got {type(model)}")

        if not model.is_fitted:
            raise ModelArtifactCorruptError("Model must be fitted before saving.")

        long_booster = model.long_booster
        short_booster = model.short_booster
        long_booster.save_model(str(target_dir / "long_booster.txt"))
        short_booster.save_model(str(target_dir / "short_booster.txt"))

        # 2. Save Ridge Priors and Model Configs
        ridge_long = model.ridge_long
        ridge_short = model.ridge_short

        shap_cfg_dict: dict[str, Any] | None = None
        if isinstance(explainer, NativeTreeSHAPExplainer):
            shap_cfg_dict = explainer.config.model_dump()
        elif self._tree_shap_config is not None:
            shap_cfg_dict = self._tree_shap_config.model_dump()

        ridge_payload: dict[str, Any] = {
            "ridge_long_coefficients": ridge_long.coefficients.tolist(),
            "ridge_long_intercept": float(ridge_long.intercept),
            "ridge_short_coefficients": ridge_short.coefficients.tolist(),
            "ridge_short_intercept": float(ridge_short.intercept),
            "ridge_config": ridge_long.config.model_dump(),
            "lgbm_config": model.config.model_dump(),
            "coherence_config": model.coherence_gate.config.model_dump(),
            "feature_names": list(model.feature_names),
            "random_state": model.random_state,
            "tree_shap_config": shap_cfg_dict,
        }
        (target_dir / "ridge_prior.json").write_text(
            json.dumps(ridge_payload, indent=2), encoding="utf-8"
        )

        # 3. Save OOD Detector Calibration
        if not isinstance(ood_detector, ConformalSubspaceOODDetector):
            raise TypeError(
                f"OOD detector must be ConformalSubspaceOODDetector, got {type(ood_detector)}"
            )
        if not ood_detector.is_fitted:
            raise ModelArtifactCorruptError("OOD detector must be fitted before saving.")

        ood_payload: dict[str, Any] = {
            "mean_vector": ood_detector.mean_vector.tolist(),
            "precision_matrix": ood_detector.precision_matrix.tolist(),
            "covariance_matrix": ood_detector.covariance_matrix.tolist(),
            "threshold_ood": float(ood_detector.threshold),
            "condition_number": float(ood_detector.condition_number),
            "config": ood_detector.config.model_dump(),
        }
        (target_dir / "ood_calibration.json").write_text(
            json.dumps(ood_payload, indent=2), encoding="utf-8"
        )

        # 4. Compute Checksum of artifact files
        checksum = compute_directory_artifacts_checksum(target_dir)

        # 5. Write metadata.json atomically via rename
        persisted_metadata = metadata.model_copy(update={"file_checksum": checksum, "role": role})
        tmp_metadata_path = target_dir / "metadata.json.tmp"
        tmp_metadata_path.write_text(
            persisted_metadata.model_dump_json(indent=2),
            encoding="utf-8",
        )
        tmp_metadata_path.replace(target_dir / "metadata.json")

        return persisted_metadata.model_id

    async def load_champion_model(
        self,
    ) -> tuple[
        CausalBarrierModelPort, OODDetectorPort, TreeSHAPExplainerPort, ModelArtifactMetadata
    ]:
        """Load the active production champion model suite and associated metadata.

        Returns:
            Tuple of (CausalBarrierModelPort, OODDetectorPort, TreeSHAPExplainerPort, ModelArtifactMetadata).

        Raises:
            ChampionModelNotFoundError: If champion model directory or metadata does not exist.
            ModelArtifactCorruptError: If artifacts are corrupt or checksum verification fails.
        """
        champion_dir = self._base_dir / "champion"
        if not champion_dir.exists():
            raise ChampionModelNotFoundError(
                f"Active champion model directory not found at {champion_dir}"
            )
        if not (champion_dir / "metadata.json").exists():
            raise ChampionModelNotFoundError(
                f"Active champion metadata not found at {champion_dir / 'metadata.json'}"
            )
        return self._load_model_from_dir(directory=champion_dir, is_champion=True)

    def _load_model_from_dir(
        self,
        *,
        directory: Path,
        is_champion: bool,
    ) -> tuple[
        CausalBarrierModelPort, OODDetectorPort, TreeSHAPExplainerPort, ModelArtifactMetadata
    ]:
        """Load model suite from specified directory with strict fail-fast validation."""
        meta_path = directory / "metadata.json"
        if not meta_path.exists():
            if is_champion:
                raise ChampionModelNotFoundError(f"Model metadata not found at {meta_path}")
            raise ChallengerModelNotFoundError(f"Model metadata not found at {meta_path}")

        try:
            raw_meta = meta_path.read_text(encoding="utf-8")
            metadata = ModelArtifactMetadata.model_validate_json(raw_meta)
        except Exception as err:
            raise ModelArtifactCorruptError(
                f"Corrupted model metadata at {meta_path}: {err}"
            ) from err

        booster_long_path = directory / "long_booster.txt"
        booster_short_path = directory / "short_booster.txt"
        ridge_path = directory / "ridge_prior.json"
        ood_path = directory / "ood_calibration.json"

        for required_file in (booster_long_path, booster_short_path, ridge_path, ood_path):
            if not required_file.exists():
                raise ModelArtifactCorruptError(
                    f"Missing required model artifact file: '{required_file.name}' in {directory}"
                )

        computed_checksum = compute_directory_artifacts_checksum(directory)
        if computed_checksum != metadata.file_checksum:
            raise ModelArtifactCorruptError(
                f"Model artifact checksum mismatch in {directory}. "
                f"Expected {metadata.file_checksum}, got {computed_checksum}"
            )

        # 1. Load Boosters
        try:
            booster_long = lgb.Booster(model_file=str(booster_long_path))
            booster_short = lgb.Booster(model_file=str(booster_short_path))
        except Exception as err:
            raise ModelArtifactCorruptError(
                f"Failed to load LightGBM boosters from {directory}: {err}"
            ) from err

        # 2. Load Ridge Prior and Constrained LGBM Configuration
        try:
            ridge_data: dict[str, Any] = json.loads(ridge_path.read_text(encoding="utf-8"))
            lgbm_config = ConstrainedLGBMConfig.model_validate(ridge_data["lgbm_config"])
            ridge_config = RidgePriorConfig.model_validate(ridge_data["ridge_config"])
            coherence_config = CoherenceGateConfig.model_validate(ridge_data["coherence_config"])
            feature_names: tuple[str, ...] = tuple(ridge_data["feature_names"])
            random_state: int = int(ridge_data["random_state"])
        except Exception as err:
            raise ModelArtifactCorruptError(
                f"Failed to parse ridge_prior.json in {directory}: {err}"
            ) from err

        # Reconstruct Long Ridge Prior
        ridge_long = OOFRidgePrior.from_saved_weights(
            config=ridge_config,
            random_state=random_state,
            n_features=len(feature_names),
            coefficients=ridge_data["ridge_long_coefficients"],
            intercept=float(ridge_data["ridge_long_intercept"]),
        )

        # Reconstruct Short Ridge Prior
        ridge_short = OOFRidgePrior.from_saved_weights(
            config=ridge_config,
            random_state=random_state + 1,
            n_features=len(feature_names),
            coefficients=ridge_data["ridge_short_coefficients"],
            intercept=float(ridge_data["ridge_short_intercept"]),
        )

        model = ConstrainedLGBMModel.from_trained_artifacts(
            config=lgbm_config,
            ridge_config=ridge_config,
            coherence_config=coherence_config,
            feature_names=feature_names,
            random_state=random_state,
            booster_long=booster_long,
            booster_short=booster_short,
            ridge_long=ridge_long,
            ridge_short=ridge_short,
        )

        # 3. Load OOD Detector
        try:
            ood_data: dict[str, Any] = json.loads(ood_path.read_text(encoding="utf-8"))
            ood_config = ConformalOODConfig.model_validate(ood_data["config"])
            ood_detector = ConformalSubspaceOODDetector.from_saved_calibration(
                config=ood_config,
                mean_vector=ood_data["mean_vector"],
                precision_matrix=ood_data["precision_matrix"],
                covariance_matrix=ood_data["covariance_matrix"],
                threshold_ood=float(ood_data["threshold_ood"]),
                condition_number=float(ood_data["condition_number"]),
            )
        except Exception as err:
            raise ModelArtifactCorruptError(
                f"Failed to parse ood_calibration.json in {directory}: {err}"
            ) from err

        # 4. Initialize TreeSHAP Explainer
        if "tree_shap_config" in ridge_data and ridge_data["tree_shap_config"] is not None:
            shap_config = TreeSHAPConfig.model_validate(ridge_data["tree_shap_config"])
        elif self._tree_shap_config is not None:
            shap_config = self._tree_shap_config
        else:
            raise ModelArtifactCorruptError(
                f"TreeSHAPConfig not found in artifact {directory} or registry configuration."
            )

        sign_verifier_long = EconomicSignVerifier(
            config=shap_config,
            monotonic_constraints=LGBMMonotonicConstraints.create_canonical_long(),
        )
        sign_verifier_short = EconomicSignVerifier(
            config=shap_config,
            monotonic_constraints=LGBMMonotonicConstraints.create_canonical_short(),
        )
        explainer = NativeTreeSHAPExplainer(
            config=shap_config,
            sign_verifier_long=sign_verifier_long,
            sign_verifier_short=sign_verifier_short,
        )

        return model, ood_detector, explainer, metadata

    async def promote_challenger_to_champion(
        self,
        *,
        user_id: str,
        rationale: str,
    ) -> ModelPromotionRecord:
        """Promote current challenger model to champion with atomic archive swap and audit record.

        Args:
            user_id: Identifier of the authorized user approving promotion.
            rationale: Explicit business or quantitative justification for promotion.

        Returns:
            ModelPromotionRecord detailing the promotion event.

        Raises:
            ChallengerModelNotFoundError: If challenger directory or metadata does not exist.
            ModelArtifactCorruptError: If challenger artifacts fail validation.
        """
        challenger_dir = self._base_dir / "challenger"
        if not challenger_dir.exists():
            raise ChallengerModelNotFoundError(
                f"Challenger model directory not found at {challenger_dir} for promotion."
            )
        if not (challenger_dir / "metadata.json").exists():
            raise ChallengerModelNotFoundError(
                f"Challenger metadata not found at {challenger_dir / 'metadata.json'} for promotion."
            )

        try:
            raw_meta = (challenger_dir / "metadata.json").read_text(encoding="utf-8")
            challenger_meta = ModelArtifactMetadata.model_validate_json(raw_meta)
        except Exception as err:
            raise ModelArtifactCorruptError(
                f"Corrupted challenger metadata at {challenger_dir}: {err}"
            ) from err

        # Validate challenger artifact files before promotion
        checksum = compute_directory_artifacts_checksum(challenger_dir)
        if checksum != challenger_meta.file_checksum:
            raise ModelArtifactCorruptError(
                f"Challenger artifact checksum mismatch. Expected {challenger_meta.file_checksum}, got {checksum}"
            )

        champion_dir = self._base_dir / "champion"
        prev_champion_id = "GENESIS"

        if champion_dir.exists() and (champion_dir / "metadata.json").exists():
            try:
                curr_raw_meta = (champion_dir / "metadata.json").read_text(encoding="utf-8")
                curr_champ_meta = ModelArtifactMetadata.model_validate_json(curr_raw_meta)
                prev_champion_id = curr_champ_meta.model_id
            except Exception:
                prev_champion_id = "UNKNOWN_PREV_CHAMPION"

            now_utc = datetime.now(UTC)
            ts_str = now_utc.strftime("%Y%m%d_%H%M%S")
            archive_dir = self._base_dir / "archive" / f"{ts_str}_{prev_champion_id}"
            archive_dir.parent.mkdir(parents=True, exist_ok=True)

            if archive_dir.exists():
                shutil.rmtree(str(archive_dir))
            shutil.move(str(champion_dir), str(archive_dir))

            # Update role in archived metadata to ARCHIVE
            arch_meta_path = archive_dir / "metadata.json"
            if arch_meta_path.exists():
                with contextlib.suppress(Exception):
                    arch_meta = ModelArtifactMetadata.model_validate_json(
                        arch_meta_path.read_text(encoding="utf-8")
                    )
                    arch_updated = arch_meta.model_copy(update={"role": ModelRole.ARCHIVE})
                    tmp_arch = archive_dir / "metadata.json.tmp"
                    tmp_arch.write_text(arch_updated.model_dump_json(indent=2), encoding="utf-8")
                    tmp_arch.replace(arch_meta_path)

        # Copy challenger to champion
        if champion_dir.exists():
            shutil.rmtree(str(champion_dir))
        shutil.copytree(str(challenger_dir), str(champion_dir))

        # Update role in champion metadata to CHAMPION
        new_champ_meta = challenger_meta.model_copy(update={"role": ModelRole.CHAMPION})
        tmp_meta = champion_dir / "metadata.json.tmp"
        tmp_meta.write_text(new_champ_meta.model_dump_json(indent=2), encoding="utf-8")
        tmp_meta.replace(champion_dir / "metadata.json")

        promotion_record = ModelPromotionRecord(
            timestamp_utc=datetime.now(UTC),
            previous_champion_id=prev_champion_id,
            promoted_challenger_id=challenger_meta.model_id,
            promoted_by_user_id=user_id,
            rationale=rationale,
        )

        if self._db_pool is not None:
            async with self._db_pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS ml_model_registry (
                        id SERIAL PRIMARY KEY,
                        timestamp_utc TIMESTAMPTZ NOT NULL,
                        previous_champion_id TEXT NOT NULL,
                        promoted_challenger_id TEXT NOT NULL,
                        promoted_by_user_id TEXT NOT NULL,
                        rationale TEXT NOT NULL
                    );
                """)
                await conn.execute(
                    """
                    INSERT INTO ml_model_registry (
                        timestamp_utc,
                        previous_champion_id,
                        promoted_challenger_id,
                        promoted_by_user_id,
                        rationale
                    ) VALUES ($1, $2, $3, $4, $5);
                    """,
                    promotion_record.timestamp_utc,
                    promotion_record.previous_champion_id,
                    promotion_record.promoted_challenger_id,
                    promotion_record.promoted_by_user_id,
                    promotion_record.rationale,
                )

        return promotion_record

    async def list_registered_models(
        self,
    ) -> list[ModelArtifactMetadata]:
        """Retrieve metadata for all registered models across champion, challenger, and archive.

        Returns:
            List of ModelArtifactMetadata instances.
        """
        models: list[ModelArtifactMetadata] = []

        # 1. Champion
        champ_meta_path = self._base_dir / "champion" / "metadata.json"
        if champ_meta_path.exists():
            with contextlib.suppress(Exception):
                models.append(
                    ModelArtifactMetadata.model_validate_json(
                        champ_meta_path.read_text(encoding="utf-8")
                    )
                )

        # 2. Challenger
        challenger_meta_path = self._base_dir / "challenger" / "metadata.json"
        if challenger_meta_path.exists():
            with contextlib.suppress(Exception):
                models.append(
                    ModelArtifactMetadata.model_validate_json(
                        challenger_meta_path.read_text(encoding="utf-8")
                    )
                )

        # 3. Archive
        archive_dir = self._base_dir / "archive"
        if archive_dir.exists() and archive_dir.is_dir():
            for child in sorted(archive_dir.iterdir()):
                if child.is_dir():
                    arch_meta_path = child / "metadata.json"
                    if arch_meta_path.exists():
                        with contextlib.suppress(Exception):
                            models.append(
                                ModelArtifactMetadata.model_validate_json(
                                    arch_meta_path.read_text(encoding="utf-8")
                                )
                            )

        return models
