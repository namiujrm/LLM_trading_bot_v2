"""Machine Learning Model Registry & Storage Domain Port (Zero Defaults).

Defines the contract for persistent storage, loading, and champion-challenger promotion
of Stage 4 Causal ML models, OOD detectors, and TreeSHAP explainers.

Clean Architecture: Ports layer.
- Pure Protocol contracts with ZERO default parameters.
- Depends ONLY on pure domain entities and sibling port interfaces.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from trading_bot.domain.entities.ml.model_registry_entities import (
    ModelArtifactMetadata,
    ModelPromotionRecord,
    ModelRole,
)
from trading_bot.ports.ml_model_port import (
    CausalBarrierModelPort,
    OODDetectorPort,
    ProbabilityCalibratorPort,
    TreeSHAPExplainerPort,
)


@runtime_checkable
class ModelRegistryPort(Protocol):
    """Abstract interface for persistent Model Storage & Registry (Zero Defaults)."""

    async def load_champion_model(
        self,
    ) -> tuple[
        CausalBarrierModelPort, OODDetectorPort, TreeSHAPExplainerPort, ModelArtifactMetadata
    ]:
        """Load the active production champion model suite and associated metadata.

        Returns:
            Tuple of (CausalBarrierModelPort, OODDetectorPort, TreeSHAPExplainerPort, ModelArtifactMetadata).
        """
        ...

    async def save_model(
        self,
        *,
        role: ModelRole,
        model: CausalBarrierModelPort,
        ood_detector: OODDetectorPort,
        explainer: TreeSHAPExplainerPort,
        metadata: ModelArtifactMetadata,
    ) -> str:
        """Persist a model suite under the specified role and return the storage path or artifact ID.

        Args:
            role: ModelRole (CHAMPION, CHALLENGER, or ARCHIVE).
            model: Causal barrier model implementing CausalBarrierModelPort.
            ood_detector: Out-of-distribution detector implementing OODDetectorPort.
            explainer: TreeSHAP explainer implementing TreeSHAPExplainerPort.
            metadata: Explicit metadata record describing training metrics and integrity.

        Returns:
            String representing the artifact identifier or persistent path.
        """
        ...

    async def promote_challenger_to_champion(
        self,
        *,
        user_id: str,
        rationale: str,
    ) -> ModelPromotionRecord:
        """Promote the current challenger model to champion with atomic archive swap and audit record.

        Args:
            user_id: Identifier of the authorized user approving promotion.
            rationale: Explicit business or quantitative justification for promotion.

        Returns:
            ModelPromotionRecord detailing the promotion event.
        """
        ...

    async def list_registered_models(
        self,
    ) -> list[ModelArtifactMetadata]:
        """Retrieve metadata for all registered models across roles.

        Returns:
            List of ModelArtifactMetadata instances.
        """
        ...


@runtime_checkable
class ModelTrainerPort(Protocol):
    """Abstract interface for training real Stage 4 Causal ML models (Zero Defaults)."""

    def train_model_suite(
        self,
        *,
        train_features: Sequence[Sequence[float]],
        y_long: Sequence[int],
        y_short: Sequence[int],
        sample_weights: Sequence[float],
        calibration_features: Sequence[Sequence[float]],
        feature_names: Sequence[str],
        random_state: int,
    ) -> tuple[CausalBarrierModelPort, OODDetectorPort, TreeSHAPExplainerPort]:
        """Train and calibrate real ConstrainedLGBMModel, ConformalSubspaceOODDetector, and TreeSHAPExplainer.

        Args:
            train_features: 2D feature matrix for training.
            y_long: Long target labels.
            y_short: Short target labels.
            sample_weights: Sample uniqueness weights.
            calibration_features: 2D feature matrix for OOD calibration.
            feature_names: Names of feature columns.
            random_state: Random seed for deterministic training.

        Returns:
            Tuple of (CausalBarrierModelPort, OODDetectorPort, TreeSHAPExplainerPort).
        """
        ...

    def prepare_training_data_from_bars(
        self,
        *,
        bars: Any | None,
        feature_names: Sequence[str],
        sample_count: int,
    ) -> tuple[
        list[list[float]],
        list[int],
        list[int],
        list[float],
        list[list[float]],
    ]:
        """Extract or generate empirical training features, binary barrier targets, and weights.

        Args:
            bars: Raw historical bar sequence or DataFrame, or None for synthetic baseline.
            feature_names: Sequence of feature column names.
            sample_count: Number of samples to generate/extract.

        Returns:
            Tuple of (train_features, y_long, y_short, sample_weights, calibration_features).
        """
        ...

    def load_dataset_from_parquet(
        self,
        *,
        parquet_path: str,
        feature_names: Sequence[str],
        cal_ratio: float,
    ) -> tuple[
        list[list[float]],
        list[int],
        list[int],
        list[float],
        list[list[float]],
        int,
    ]:
        """Load real empirical dataset from parquet file and partition into train/calibration sets.

        Args:
            parquet_path: Path to parquet dataset containing features, targets, and weights.
            feature_names: Sequence of feature column names.
            cal_ratio: Proportion of dataset reserved for causal OOD calibration.

        Returns:
            Tuple of (train_features, y_long, y_short, sample_weights, calibration_features, total_samples).
        """
        ...

    def roll_dataset_window(
        self,
        *,
        base_parquet_path: str,
        new_bars_parquet_path: str | None,
        max_bars_per_symbol: int,
        output_parquet_path: str,
    ) -> int:
        """Roll FIFO dataset window by appending new weekly bars and trimming oldest bars.

        Args:
            base_parquet_path: Path to existing base parquet dataset.
            new_bars_parquet_path: Path to new weekly bar parquet, or None if no update.
            max_bars_per_symbol: Maximum number of historical bars retained per symbol (e.g. 52608 for 36M).
            output_parquet_path: Destination path for updated rolling dataset.

        Returns:
            Total row count of the rolled dataset.
        """
        ...

    def evaluate_model_on_parquet(
        self,
        *,
        model: CausalBarrierModelPort,
        parquet_path: str,
        feature_names: Sequence[str],
        cal_ratio: float,
        random_state: int,
    ) -> dict[str, float]:
        """Evaluate fitted model via full Walk-Forward Backtest (on 36M dataset) or holdout slice.

        Args:
            model: Fitted barrier model implementing CausalBarrierModelPort.
            parquet_path: Path to empirical parquet dataset.
            feature_names: Sequence of feature column names.
            cal_ratio: Proportion of trailing rows used as calibration/validation slice for small datasets.
            random_state: Deterministic random seed used in walk-forward windows.

        Returns:
            Dict containing comprehensive financial and ML calibration metrics.
        """
        ...

    def fit_calibrators_on_parquet(
        self,
        *,
        model: CausalBarrierModelPort,
        parquet_path: str,
        feature_names: Sequence[str],
        cal_ratio: float,
    ) -> tuple[ProbabilityCalibratorPort, ProbabilityCalibratorPort]:
        """Fit dual Long/Short Venn-Abers probability calibrators on parquet calibration split."""
        ...


@runtime_checkable
class CalibratorRegistryPort(Protocol):
    """Port for persisting and loading Venn-Abers calibrators alongside model roles (Zero Defaults)."""

    async def save_calibrators(
        self,
        *,
        role: ModelRole,
        calibrator_long: ProbabilityCalibratorPort,
        calibrator_short: ProbabilityCalibratorPort,
    ) -> str:
        """Persist fitted Long/Short probability calibrators for a registered model role."""
        ...

    async def load_champion_calibrators(
        self,
    ) -> tuple[ProbabilityCalibratorPort, ProbabilityCalibratorPort] | None:
        """Load active champion Venn-Abers calibrators if present, else return None."""
        ...
