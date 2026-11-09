"""Machine Learning Model Domain Ports Interface (Protocols) — Zero Defaults.

Domain Ports defining abstract contracts for Stage 4 Causal ML components:
- Conformal Subspace OOD Detector (M4.2)
- Causal Barrier Model (M4.3)
- Probability Calibrator (M4.4)
- TreeSHAP Explainer (M4.5)

Clean Architecture: 100% pure typing and domain entities.
Zero heavy computation dependencies (NO numpy, NO scipy, NO sklearn).
Invariant #9: zero default parameters, zero hardcoded values.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from trading_bot.domain.entities.ml.ml_predictions import (
    BarrierPredictionResult,
    CalibratedProbability,
    OODDetectionResult,
    SHAPExplanationResult,
)


@runtime_checkable
class CausalBarrierModelPort(Protocol):
    """Abstract interface for Causal Dual-Barrier ML Ensemble (Zero Defaults).

    Defines the contract for the dual-binary causal model combining:
    - 5-Fold Out-of-Fold (OOF) Logistic Ridge Linear Prior
    - Monotonic & Interaction-Constrained LightGBM
    - Dual-Binary Coherence Gate

    Strict Clean Architecture:
    - Depends ONLY on pure Python sequences and Domain Entities.
    - Zero heavy computational dependencies in signatures (NO numpy, NO scipy, NO sklearn, NO lightgbm).
    - Invariant #9: zero default parameters, zero class field defaults.
    """

    def fit(
        self,
        train_features: Sequence[Sequence[float]],
        y_long: Sequence[int],
        y_short: Sequence[int],
        sample_weights: Sequence[float],
        sample_intervals: Sequence[tuple[int, int]] | None,
    ) -> None:
        """Fit the dual-binary causal ensemble on training features and de Prado weights.

        Args:
            train_features: 2D sequence of shape (N, D) containing causal features.
            y_long: Binary target sequence of length N for Long barrier (1 = TP hit, 0 = otherwise).
            y_short: Binary target sequence of length N for Short barrier (1 = TP hit, 0 = otherwise).
            sample_weights: Sequence of length N containing López de Prado uniqueness weights.
            sample_intervals: Optional sequence of (t_start, t_end) sample intervals for Purged CV.
        """
        ...

    def predict(
        self,
        sample_features: Sequence[float],
    ) -> BarrierPredictionResult:
        """Predict coherent directional barrier probabilities for a single feature vector.

        Args:
            sample_features: 1D sequence of length D representing a single 30m decision sample.

        Returns:
            BarrierPredictionResult containing raw and coherent (p_long, p_short) probabilities.
        """
        ...

    def predict_batch(
        self,
        batch_features: Sequence[Sequence[float]],
    ) -> Sequence[BarrierPredictionResult]:
        """Predict coherent directional barrier probabilities for a batch of feature vectors.

        Args:
            batch_features: 2D sequence of shape (M, D) containing M decision samples.

        Returns:
            Sequence of M BarrierPredictionResult instances.
        """
        ...


@runtime_checkable
class OODDetectorPort(Protocol):
    """Abstract interface for Conformal Subspace Out-of-Distribution Detector (Zero Defaults)."""

    def fit(
        self,
        train_features: Sequence[Sequence[float]],
        calibration_features: Sequence[Sequence[float]],
    ) -> None:
        """Estimate Ledoit-Wolf covariance on train and compute empirical conformal quantile on cal."""
        ...

    def predict(
        self,
        sample_features: Sequence[float],
    ) -> OODDetectionResult:
        """Evaluate a single feature vector against the empirical conformal threshold."""
        ...

    def predict_batch(
        self,
        batch_features: Sequence[Sequence[float]],
    ) -> Sequence[OODDetectionResult]:
        """Evaluate a batch of feature vectors against the empirical conformal threshold."""
        ...


@runtime_checkable
class ProbabilityCalibratorPort(Protocol):
    """Abstract interface for Probability Calibrator (Zero Defaults).

    Defines the contract for non-parametric probability calibration (Proposal P5: Fast Venn-Abers).

    Strict Clean Architecture:
    - Depends ONLY on pure Python sequences and Domain Entities (CalibratedProbability).
    - Zero heavy computational dependencies in signatures (NO numpy, NO scipy, NO sklearn).
    - Invariant #9: zero default parameters, zero class field defaults.
    """

    def fit(
        self,
        scores: Sequence[float],
        labels: Sequence[int],
    ) -> None:
        """Fit the calibrator on validation/calibration raw scores and true binary labels.

        Args:
            scores: Sequence of raw decision scores in [0, 1].
            labels: Sequence of binary ground truth labels in {0, 1}.
        """
        ...

    def calibrate(
        self,
        score: float,
    ) -> CalibratedProbability:
        """Calibrate a single raw score into a calibrated probability with uncertainty bounds.

        Args:
            score: Raw uncalibrated decision score in [0, 1].

        Returns:
            CalibratedProbability instance containing p_calibrated, p_lower, p_upper, uncertainty_width.
        """
        ...

    def calibrate_batch(
        self,
        scores: Sequence[float],
    ) -> tuple[CalibratedProbability, ...]:
        """Calibrate a batch of raw decision scores into calibrated probabilities.

        Args:
            scores: Sequence of raw decision scores in [0, 1].

        Returns:
            Tuple of CalibratedProbability instances of length len(scores).
        """
        ...


@runtime_checkable
class TreeSHAPExplainerPort(Protocol):
    """Abstract interface for Native LightGBM C++ TreeSHAP Engine (Zero Defaults).

    Defines the contract for high-speed TreeSHAP feature attribution extraction
    and economic sign verification (Proposal P6: Native LightGBM C++ TreeSHAP).

    Strict Clean Architecture:
    - Depends ONLY on pure Python sequences, standard typing, and Domain Entities.
    - Zero heavy computational dependencies in signatures (NO numpy, NO scipy, NO sklearn, NO lightgbm).
    - Invariant #9: zero default parameters, zero class field defaults.
    """

    def explain(
        self,
        feature_vector: Sequence[float],
        feature_names: Sequence[str],
        model: Any,
        symbol: str,
        bar_time: datetime,
        model_type: str,
    ) -> SHAPExplanationResult:
        """Explain a single decision feature vector using native C++ TreeSHAP attributions.

        Args:
            feature_vector: Sequence of length M containing numerical feature values.
            feature_names: Sequence of length M matching the feature columns.
            model: Trained booster or causal model instance.
            symbol: Target symbol (e.g. "BTCUSDT").
            bar_time: UTC timestamp of the decision bar.
            model_type: Model identifier ("long_barrier" or "short_barrier").

        Returns:
            SHAPExplanationResult containing full contributions, top drivers, and economic sign audit.
        """
        ...

    def explain_batch(
        self,
        batch_features: Sequence[Sequence[float]],
        feature_names: Sequence[str],
        model: Any,
        symbol: str,
        bar_times: Sequence[datetime],
        model_type: str,
    ) -> tuple[SHAPExplanationResult, ...]:
        """Explain a batch of decision feature vectors using native C++ TreeSHAP attributions.

        Args:
            batch_features: 2D sequence of shape (N, M) containing N feature vectors.
            feature_names: Sequence of length M matching the feature columns.
            model: Trained booster or causal model instance.
            symbol: Target symbol (e.g. "BTCUSDT").
            bar_times: Sequence of N UTC timestamps corresponding to each sample.
            model_type: Model identifier ("long_barrier" or "short_barrier").

        Returns:
            Tuple of N SHAPExplanationResult instances.
        """
        ...
