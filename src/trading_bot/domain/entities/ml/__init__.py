"""Domain Entities for Stage 4 Machine Learning (Clean Architecture, Invariant #9)."""

from trading_bot.domain.entities.ml.barrier_labels import (
    BarrierLabel,
    BarrierType,
    LabelSpan,
    SampleWeightRecord,
)
from trading_bot.domain.entities.ml.ml_configs import (
    CoherenceGateConfig,
    ConformalOODConfig,
    ConstrainedLGBMConfig,
    ECEMonitorConfig,
    FastVennAbersConfig,
    LGBMMonotonicConstraints,
    PurgedCVConfig,
    RidgePriorConfig,
    SampleWeightConfig,
    SHAPExplainerConfig,
    TreeSHAPConfig,
    TripleBarrierConfig,
)
from trading_bot.domain.entities.ml.ml_predictions import (
    BarrierPredictionResult,
    CalibratedProbability,
    DualBarrierPrediction,
    ECEDriftRecord,
    OODDetectionResult,
    SHAPDriverContribution,
    SHAPExplanationResult,
)
from trading_bot.domain.entities.ml.model_registry_entities import (
    ModelArtifactMetadata,
    ModelPromotionRecord,
    ModelRole,
)

__all__ = [
    "BarrierLabel",
    "BarrierPredictionResult",
    "BarrierType",
    "CalibratedProbability",
    "CoherenceGateConfig",
    "ConformalOODConfig",
    "ConstrainedLGBMConfig",
    "DualBarrierPrediction",
    "ECEDriftRecord",
    "ECEMonitorConfig",
    "FastVennAbersConfig",
    "LGBMMonotonicConstraints",
    "LabelSpan",
    "ModelArtifactMetadata",
    "ModelPromotionRecord",
    "ModelRole",
    "OODDetectionResult",
    "PurgedCVConfig",
    "RidgePriorConfig",
    "SHAPDriverContribution",
    "SHAPExplainerConfig",
    "SHAPExplanationResult",
    "SampleWeightConfig",
    "SampleWeightRecord",
    "TreeSHAPConfig",
    "TripleBarrierConfig",
]
