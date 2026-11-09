"""Machine Learning Engine Adapters (Clean Architecture Layer 4)."""

from __future__ import annotations

from trading_bot.adapters.ml_engine.conformal_ood_detector import (
    ConformalSubspaceOODDetector,
)
from trading_bot.adapters.ml_engine.constrained_lgbm_model import (
    CANONICAL_FEATURES,
    CANONICAL_INTERACTION_CLUSTERS,
    CANONICAL_MONOTONE_LONG,
    CANONICAL_MONOTONE_SHORT,
    ConstrainedLGBMModel,
)
from trading_bot.adapters.ml_engine.native_treeshap_explainer import (
    NativeTreeSHAPExplainer,
)
from trading_bot.adapters.ml_engine.oof_ridge_prior import OOFRidgePrior
from trading_bot.adapters.ml_engine.tabulated_venn_abers import (
    TabulatedVennAbersCalibrator,
)

__all__ = [
    "CANONICAL_FEATURES",
    "CANONICAL_INTERACTION_CLUSTERS",
    "CANONICAL_MONOTONE_LONG",
    "CANONICAL_MONOTONE_SHORT",
    "ConformalSubspaceOODDetector",
    "ConstrainedLGBMModel",
    "NativeTreeSHAPExplainer",
    "OOFRidgePrior",
    "TabulatedVennAbersCalibrator",
]
