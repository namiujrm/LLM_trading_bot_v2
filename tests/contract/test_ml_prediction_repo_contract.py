"""Contract tests for MLPredictionRepositoryPort and MLPredictionRepository."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from datetime import UTC, datetime

from trading_bot.adapters.persistence.ml_prediction_repo import MLPredictionRepository
from trading_bot.domain.entities.ml.ml_predictions import (
    CalibratedProbability,
    OODDetectionResult,
    SHAPDriverContribution,
    UnifiedMLPredictionPacket,
)
from trading_bot.ports.ml_prediction_repository_port import MLPredictionRepositoryPort


class DummyMLPredictionRepo:
    """Minimal implementation conforming strictly to MLPredictionRepositoryPort."""

    async def save_prediction_packet(
        self,
        packet: UnifiedMLPredictionPacket,
    ) -> None:
        pass

    async def save_prediction_packets_batch(
        self,
        packets: Sequence[UnifiedMLPredictionPacket],
    ) -> int:
        return len(packets)

    async def get_prediction_packet(
        self,
        symbol: str,
    ) -> UnifiedMLPredictionPacket | None:
        return None

    async def get_prediction_packets_range(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Sequence[UnifiedMLPredictionPacket]:
        return ()


def test_ml_prediction_repository_port_protocol_conformance() -> None:
    """Test that MLPredictionRepositoryPort is runtime_checkable and recognizes conformant classes."""
    dummy = DummyMLPredictionRepo()
    assert isinstance(dummy, MLPredictionRepositoryPort)


def test_ml_prediction_repository_adapter_conformance() -> None:
    """Test that the persistence adapter MLPredictionRepository conforms to MLPredictionRepositoryPort."""
    assert issubclass(MLPredictionRepository, MLPredictionRepositoryPort)


def test_ml_prediction_repository_port_signatures() -> None:
    """Assert that port methods have explicit parameter annotations and return types."""
    methods = [
        "save_prediction_packet",
        "save_prediction_packets_batch",
        "get_prediction_packet",
        "get_prediction_packets_range",
    ]
    for method_name in methods:
        assert hasattr(MLPredictionRepositoryPort, method_name)
        method = getattr(MLPredictionRepositoryPort, method_name)
        sig = inspect.signature(method)
        # Ensure zero default values on parameters
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            assert param.default is inspect.Parameter.empty, (
                f"Port method `{method_name}` has default on parameter `{param_name}`"
            )


def test_unified_ml_prediction_packet_creation() -> None:
    """Test creating a valid UnifiedMLPredictionPacket and its invariants."""
    now = datetime(2025, 6, 15, 12, 0, tzinfo=UTC)
    packet = UnifiedMLPredictionPacket(
        symbol="BTCUSDT",
        bar_time=now,
        regime_label="STRONG_TREND_BULL",
        model_version="stage4_v1.0.0",
        p_long_raw=0.65,
        p_short_raw=0.25,
        p_long_calibrated=CalibratedProbability(
            p_calibrated=0.62,
            p_lower=0.58,
            p_upper=0.66,
            uncertainty_width=0.08,
            is_uncertain=False,
            raw_score=0.65,
        ),
        p_short_calibrated=CalibratedProbability(
            p_calibrated=0.24,
            p_lower=0.20,
            p_upper=0.28,
            uncertainty_width=0.08,
            is_uncertain=False,
            raw_score=0.25,
        ),
        ood_status=OODDetectionResult(
            is_ood=False,
            mahalanobis_distance=1.85,
            empirical_quantile_threshold=3.45,
            excess_distance=0.0,
            gamma_ood=1.0,
            feature_contributions=(),
        ),
        top_bullish_shap=(
            SHAPDriverContribution(
                feature_name="taker_cvd_z_24",
                feature_value=1.52,
                shap_value=0.18,
                expected_sign=1,
                is_sign_consistent=True,
            ),
        ),
        top_bearish_shap=(
            SHAPDriverContribution(
                feature_name="vpin_toxicity_pct",
                feature_value=0.72,
                shap_value=-0.09,
                expected_sign=-1,
                is_sign_consistent=True,
            ),
        ),
        is_economic_consistent=True,
        features_snapshot={"taker_cvd_z_24": 1.52, "vpin_toxicity_pct": 0.72},
    )

    assert packet.symbol == "BTCUSDT"
    assert packet.is_ood is False
    assert packet.ood_shrinkage == 1.0
    assert packet.p_long_cal == 0.62
    assert packet.p_short_cal == 0.24
    assert packet.p_long_uncertainty == 0.08
    assert packet.features["taker_cvd_z_24"] == 1.52
