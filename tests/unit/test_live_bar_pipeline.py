"""Unit tests for LiveBarPipelineService (Clean Architecture & Invariant #9 Compliant).

Verifies:
- 1m bar rolling buffer management (300-bar FIFO cap).
- Multi-timeframe resampling from 1m into 15m and 30m bars.
- Microstructure and regime feature evaluation via QuantEnginePort.
- Cross-asset return tracking, altcoin beta, and lead-lag catchup gap calculations.
- 30m candle close triggering causal ML inference, OOD detection, TreeSHAP explanation,
  and TieredFunnelInput assembly.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import pytest

from trading_bot.application.services.live_bar_pipeline import (
    LiveBarPipelineService,
)
from trading_bot.application.use_cases.evaluate_tiered_funnel import TieredFunnelInput
from trading_bot.domain.entities.common.enums import (
    MarketRegime,
    OIPriceQuadrant,
)
from trading_bot.domain.entities.market.market_state import (
    EnrichedMicroBarRecord,
    FeatureSignalItem,
    MicroBar1m,
    MultiTFBar,
    SemanticQuantPacket,
)
from trading_bot.domain.entities.ml.ml_predictions import (
    BarrierPredictionResult,
    OODDetectionResult,
    SHAPDriverContribution,
    SHAPExplanationResult,
)
from trading_bot.domain.entities.quant.quant_features import (
    MicrostructureMetrics,
    QuantFeatureVector,
    RegimeMetrics,
)
from trading_bot.ports.ml_model_port import (
    CausalBarrierModelPort,
    OODDetectorPort,
    TreeSHAPExplainerPort,
)
from trading_bot.ports.quant_engine_port import QuantEnginePort

# ==============================================================================
# Invariant #9 Compliant Test Doubles (Zero Defaults, Zero Class Field Defaults)
# ==============================================================================


class MockQuantEngine(QuantEnginePort):
    def __init__(self) -> None:
        self.call_count_micro: int = 0
        self.call_count_regime: int = 0
        self.call_count_packet: int = 0

    async def compute_microstructure_metrics(
        self,
        symbol: str,
        bars_1m: list[MicroBar1m],
    ) -> MicrostructureMetrics:
        self.call_count_micro += 1
        return MicrostructureMetrics(
            vpin=0.45,
            kyle_lambda=0.001,
            estimated_oneway_slippage_pct=0.0002,
            ofi_zscore=1.2,
            obi_mean=0.15,
            obi_std=0.05,
            va_amihud=0.0001,
            corwin_schultz_spread_bps=1.5,
            mnvr=0.25,
        )

    async def compute_regime_state(
        self,
        symbol: str,
        bars_multi_tf: list[MultiTFBar],
    ) -> RegimeMetrics:
        self.call_count_regime += 1
        return RegimeMetrics(
            hurst_exponent=0.58,
            yang_zhang_vol=0.018,
            garman_klass_vol=0.016,
            parkinson_vol=0.015,
            bipower_variation=0.017,
            jump_z_score=0.2,
            has_volatility_jump=False,
            shannon_entropy=0.62,
            permutation_entropy=0.65,
            tsallis_entropy=0.60,
            regime_label=MarketRegime.STRONG_TREND_BULL,
        )

    async def evaluate_quant_packet(
        self,
        symbol: str,
        timestamp_ms: int,
    ) -> SemanticQuantPacket:
        self.call_count_packet += 1
        return SemanticQuantPacket(
            symbol=symbol,
            timestamp_ms=timestamp_ms,
            active_features=[
                FeatureSignalItem(
                    feature_name="hurst_exponent_100",
                    raw_value=0.58,
                    z_score=1.2,
                    is_statistically_active=True,
                    economic_direction=1,
                )
            ],
            market_regime=MarketRegime.STRONG_TREND_BULL,
            oi_quadrant=OIPriceQuadrant.LONG_BUILDUP,
            trajectory_vector_16d=[
                0.58,
                0.62,
                0.5,
                0.45,
                0.8,
                1.2,
                -1.0,
                0.7,
                0.1,
                -0.1,
                0.4,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        )

    async def compute_feature_vector_batch(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> list[QuantFeatureVector]:
        return []


class MockBarrierModel(CausalBarrierModelPort):
    def __init__(self, p_long: float, p_short: float) -> None:
        self._p_long: float = p_long
        self._p_short: float = p_short
        self.predict_call_count: int = 0

    def fit(
        self,
        train_features: Sequence[Sequence[float]],
        y_long: Sequence[int],
        y_short: Sequence[int],
        sample_weights: Sequence[float],
        sample_intervals: Sequence[tuple[int, int]] | None,
    ) -> None:
        pass

    def predict(
        self,
        sample_features: Sequence[float],
    ) -> BarrierPredictionResult:
        self.predict_call_count += 1
        return BarrierPredictionResult(
            p_long=self._p_long,
            p_short=self._p_short,
            p_raw_long=self._p_long,
            p_raw_short=self._p_short,
            is_coherent=True,
            coherence_penalty=0.0,
        )

    def predict_batch(
        self,
        batch_features: Sequence[Sequence[float]],
    ) -> Sequence[BarrierPredictionResult]:
        return [self.predict(f) for f in batch_features]


class MockOODDetector(OODDetectorPort):
    def __init__(self, is_ood: bool) -> None:
        self._is_ood: bool = is_ood
        self.predict_call_count: int = 0

    def fit(
        self,
        train_features: Sequence[Sequence[float]],
        calibration_features: Sequence[Sequence[float]],
    ) -> None:
        pass

    def predict(
        self,
        sample_features: Sequence[float],
    ) -> OODDetectionResult:
        self.predict_call_count += 1
        return OODDetectionResult(
            is_ood=self._is_ood,
            mahalanobis_distance=1.8,
            empirical_quantile_threshold=3.2,
            excess_distance=0.0,
            gamma_ood=0.0,
            feature_contributions=(0.3, 0.2, 0.4, 0.5, 0.2, 0.2),
        )

    def predict_batch(
        self,
        batch_features: Sequence[Sequence[float]],
    ) -> Sequence[OODDetectionResult]:
        return [self.predict(f) for f in batch_features]


class MockTreeSHAPExplainer(TreeSHAPExplainerPort):
    def __init__(self) -> None:
        self.explain_call_count: int = 0

    def explain(
        self,
        feature_vector: Sequence[float],
        feature_names: Sequence[str],
        model: Any,
        symbol: str,
        bar_time: datetime,
        model_type: str,
    ) -> SHAPExplanationResult:
        self.explain_call_count += 1
        driver = SHAPDriverContribution(
            feature_name="hurst_exponent_100",
            feature_value=0.58,
            shap_value=0.15,
            expected_sign=1,
            is_sign_consistent=True,
        )
        return SHAPExplanationResult(
            symbol=symbol,
            bar_time=bar_time,
            model_type=model_type,
            base_value=0.0,
            raw_margin=0.15,
            contributions=(driver,),
            top_bullish_drivers=(driver,),
            top_bearish_drivers=(),
            is_economic_consistent=True,
        )

    def explain_batch(
        self,
        batch_features: Sequence[Sequence[float]],
        feature_names: Sequence[str],
        model: Any,
        symbol: str,
        bar_times: Sequence[datetime],
        model_type: str,
    ) -> tuple[SHAPExplanationResult, ...]:
        return tuple(
            self.explain(f, feature_names, model, symbol, t, model_type)
            for f, t in zip(batch_features, bar_times, strict=True)
        )


def make_sample_1m_bar(symbol: str, open_time_ms: int, price: float) -> EnrichedMicroBarRecord:
    return EnrichedMicroBarRecord(
        symbol=symbol,
        open_time_ms=open_time_ms,
        open=price,
        high=price + 10.0,
        low=price - 10.0,
        close=price + 2.0,
        volume=10.0,
        quote_volume=price * 10.0,
        taker_buy_vol=6.0,
        taker_buy_quote_vol=price * 6.0,
        trade_count=100,
        ofi_sum=15.0,
        obi_mean=0.12,
        obi_std=0.04,
        spread_bps_mean=1.5,
        liq_long_usd=1000.0,
        liq_short_usd=500.0,
    )


# ==============================================================================
# Unit Test Cases
# ==============================================================================


@pytest.mark.asyncio
async def test_1m_bar_buffer_caps_at_300_bars() -> None:
    quant_engine = MockQuantEngine()
    barrier_model = MockBarrierModel(p_long=0.65, p_short=0.20)
    ood_detector = MockOODDetector(is_ood=False)
    treeshap = MockTreeSHAPExplainer()

    pipeline = LiveBarPipelineService(
        symbols=["BTCUSDT"],
        quant_engine=quant_engine,
        barrier_model=barrier_model,
        ood_detector=ood_detector,
        treeshap_explainer=treeshap,
        on_30m_bar=None,
        on_cycle_ready=None,
    )

    base_ts = 1700000040000  # Minute-aligned
    # Feed 310 1m bars
    for i in range(310):
        bar = make_sample_1m_bar("BTCUSDT", base_ts + i * 60000, 40000.0 + float(i))
        await pipeline.on_1m_bar(bar)

    buffered = pipeline.get_bars_1m("BTCUSDT")
    assert len(buffered) == 300
    # First 10 bars (0 to 9) should have been evicted; earliest should be bar 10
    assert buffered[0].open_time_ms == base_ts + 10 * 60000
    assert buffered[-1].open_time_ms == base_ts + 309 * 60000


@pytest.mark.asyncio
async def test_resample_1m_into_15m_and_30m_bars() -> None:
    quant_engine = MockQuantEngine()
    barrier_model = MockBarrierModel(p_long=0.60, p_short=0.25)
    ood_detector = MockOODDetector(is_ood=False)
    treeshap = MockTreeSHAPExplainer()

    pipeline = LiveBarPipelineService(
        symbols=["BTCUSDT"],
        quant_engine=quant_engine,
        barrier_model=barrier_model,
        ood_detector=ood_detector,
        treeshap_explainer=treeshap,
        on_30m_bar=None,
        on_cycle_ready=None,
    )

    # Choose a base timestamp at exactly :00:00 boundary (divisible by 1800000)
    # 1700001000000 % 1800000 == 0
    boundary_30m = 1700001000000
    assert boundary_30m % 1800000 == 0

    # Feed exactly 30 1m bars
    for i in range(30):
        bar = make_sample_1m_bar("BTCUSDT", boundary_30m + i * 60000, 40000.0 + float(i))
        await pipeline.on_1m_bar(bar)

    bars_15m = pipeline.get_bars_15m("BTCUSDT")
    bars_30m = pipeline.get_bars_30m("BTCUSDT")

    # In 30 minutes, two 15m intervals close (minute 14 and minute 29)
    assert len(bars_15m) == 2
    assert bars_15m[0].timeframe == "15m"
    assert bars_15m[0].timestamp_ms == boundary_30m
    assert bars_15m[1].timeframe == "15m"
    assert bars_15m[1].timestamp_ms == boundary_30m + 900000

    # One 30m interval closes (minute 29)
    assert len(bars_30m) == 1
    assert bars_30m[0].timeframe == "30m"
    assert bars_30m[0].timestamp_ms == boundary_30m
    assert bars_30m[0].open == 40000.0
    assert bars_30m[0].close == 40000.0 + 29.0 + 2.0


@pytest.mark.asyncio
async def test_cross_asset_lead_lag_catchup_metrics() -> None:
    quant_engine = MockQuantEngine()
    barrier_model = MockBarrierModel(p_long=0.62, p_short=0.18)
    ood_detector = MockOODDetector(is_ood=False)
    treeshap = MockTreeSHAPExplainer()

    pipeline = LiveBarPipelineService(
        symbols=["BTCUSDT", "ETHUSDT"],
        quant_engine=quant_engine,
        barrier_model=barrier_model,
        ood_detector=ood_detector,
        treeshap_explainer=treeshap,
        on_30m_bar=None,
        on_cycle_ready=None,
    )

    base_ts = 1700001000000
    # Feed 30 bars for both BTC and ETH
    for i in range(30):
        btc_bar = make_sample_1m_bar("BTCUSDT", base_ts + i * 60000, 40000.0 * (1.0 + 0.001 * i))
        eth_bar = make_sample_1m_bar("ETHUSDT", base_ts + i * 60000, 2500.0 * (1.0 + 0.0008 * i))
        await pipeline.on_1m_bar(btc_bar)
        await pipeline.on_1m_bar(eth_bar)

    btc_input = pipeline.get_latest_funnel_input("BTCUSDT")
    eth_input = pipeline.get_latest_funnel_input("ETHUSDT")

    # For BTC lead asset, beta must be 1.0 and catchup gap 0.0
    assert btc_input.btc_lead_beta == 1.0
    assert btc_input.lead_lag_catchup_gap_z == 0.0

    # For altcoin ETH, beta and catchup gap must be populated
    assert eth_input.btc_lead_beta > 0.0
    assert isinstance(eth_input.lead_lag_catchup_gap_z, float)


@pytest.mark.asyncio
async def test_30m_candle_close_triggers_ml_ood_treeshap_emission() -> None:
    quant_engine = MockQuantEngine()
    barrier_model = MockBarrierModel(p_long=0.68, p_short=0.15)
    ood_detector = MockOODDetector(is_ood=False)
    treeshap = MockTreeSHAPExplainer()

    emitted_funnel_inputs: list[tuple[str, TieredFunnelInput, SemanticQuantPacket]] = []
    cycle_ready_events: list[tuple[datetime, dict[str, TieredFunnelInput]]] = []

    async def on_30m_bar(
        sym: str, funnel_inp: TieredFunnelInput, quant_pkt: SemanticQuantPacket
    ) -> None:
        emitted_funnel_inputs.append((sym, funnel_inp, quant_pkt))

    async def on_cycle(
        cycle_dt: datetime,
        inputs: dict[str, TieredFunnelInput],
        packets: dict[str, SemanticQuantPacket],
        prices: dict[str, float],
        atrs: dict[str, float],
    ) -> None:
        cycle_ready_events.append((cycle_dt, inputs))

    pipeline = LiveBarPipelineService(
        symbols=["BTCUSDT", "ETHUSDT"],
        quant_engine=quant_engine,
        barrier_model=barrier_model,
        ood_detector=ood_detector,
        treeshap_explainer=treeshap,
        on_30m_bar=on_30m_bar,
        on_cycle_ready=on_cycle,
    )

    base_ts = 1700001000000  # Divisible by 1800000
    # Feed 30 bars for both symbols up to 30m close
    for i in range(30):
        ts = base_ts + i * 60000
        await pipeline.on_1m_bar(make_sample_1m_bar("BTCUSDT", ts, 40000.0 + i))
        await pipeline.on_1m_bar(make_sample_1m_bar("ETHUSDT", ts, 2500.0 + i))

    # Verification: Both symbols emitted 30m bar
    assert len(emitted_funnel_inputs) == 2
    btc_entry = next(e for e in emitted_funnel_inputs if e[0] == "BTCUSDT")
    eth_entry = next(e for e in emitted_funnel_inputs if e[0] == "ETHUSDT")

    # ML barrier probability propagation
    assert btc_entry[1].p_long_cal == 0.68
    assert btc_entry[1].p_short_cal == 0.15
    assert not btc_entry[1].is_ood
    assert eth_entry[1].p_long_cal == 0.68
    assert eth_entry[1].p_short_cal == 0.15

    # Quant packet verification
    assert btc_entry[2].market_regime == MarketRegime.STRONG_TREND_BULL
    assert btc_entry[2].oi_quadrant == OIPriceQuadrant.LONG_BUILDUP

    # Both symbols done -> on_cycle_ready fired
    assert len(cycle_ready_events) == 1
    assert "BTCUSDT" in cycle_ready_events[0][1]
    assert "ETHUSDT" in cycle_ready_events[0][1]

    # Model invocations check
    assert barrier_model.predict_call_count == 2
    assert ood_detector.predict_call_count == 2
    assert treeshap.explain_call_count == 2

    # Getters
    assert pipeline.get_latest_funnel_input("BTCUSDT").symbol == "BTCUSDT"
    assert pipeline.get_latest_quant_packet("ETHUSDT").symbol == "ETHUSDT"
    assert pipeline.get_latest_shap_result("BTCUSDT") is not None
    assert pipeline.get_atr_30m_values()["BTCUSDT"] > 0.0


@pytest.mark.asyncio
async def test_30m_candle_cycle_deduplication_exact_once() -> None:
    """Verify that on_cycle_ready fires exactly once per 30m candle and does not duplicate."""
    quant_engine = MockQuantEngine()
    barrier_model = MockBarrierModel(p_long=0.68, p_short=0.15)
    ood_detector = MockOODDetector(is_ood=False)
    treeshap = MockTreeSHAPExplainer()

    cycle_dispatches: list[datetime] = []

    async def on_cycle(
        cycle_dt: datetime,
        inputs: dict[str, TieredFunnelInput],
        packets: dict[str, SemanticQuantPacket],
        prices: dict[str, float],
        atrs: dict[str, float],
    ) -> None:
        cycle_dispatches.append(cycle_dt)

    pipeline = LiveBarPipelineService(
        symbols=["BTCUSDT", "ETHUSDT"],
        quant_engine=quant_engine,
        barrier_model=barrier_model,
        ood_detector=ood_detector,
        treeshap_explainer=treeshap,
        on_30m_bar=None,
        on_cycle_ready=on_cycle,
    )

    base_ts = 1700001000000  # Divisible by 1800000
    # Feed 2 complete 30m cycles (60 1m bars for each symbol)
    for i in range(60):
        ts = base_ts + i * 60000
        await pipeline.on_1m_bar(make_sample_1m_bar("BTCUSDT", ts, 40000.0 + i))
        await pipeline.on_1m_bar(make_sample_1m_bar("ETHUSDT", ts, 2500.0 + i))

    # Exactly 2 cycles should have fired (one at bar 29, one at bar 59)
    assert len(cycle_dispatches) == 2

    # Duplicate / delayed bar at bar 59 should NOT fire a 3rd cycle
    delayed_ts = base_ts + 59 * 60000
    await pipeline.on_1m_bar(make_sample_1m_bar("BTCUSDT", delayed_ts, 40060.0))
    await pipeline.on_1m_bar(make_sample_1m_bar("ETHUSDT", delayed_ts, 2560.0))
    assert len(cycle_dispatches) == 2


@pytest.mark.asyncio
async def test_warmup_historical_bars_populates_buffers_without_triggering_cycles() -> None:
    """Verify that warmup_historical_bars pre-fills buffers without firing on_cycle_ready."""
    quant_engine = MockQuantEngine()
    barrier_model = MockBarrierModel(p_long=0.70, p_short=0.10)
    ood_detector = MockOODDetector(is_ood=False)
    treeshap = MockTreeSHAPExplainer()

    cycle_dispatches: list[datetime] = []

    async def on_cycle(
        cycle_dt: datetime,
        inputs: dict[str, TieredFunnelInput],
        packets: dict[str, SemanticQuantPacket],
        prices: dict[str, float],
        atrs: dict[str, float],
    ) -> None:
        cycle_dispatches.append(cycle_dt)

    pipeline = LiveBarPipelineService(
        symbols=["BTCUSDT"],
        quant_engine=quant_engine,
        barrier_model=barrier_model,
        ood_detector=ood_detector,
        treeshap_explainer=treeshap,
        on_30m_bar=None,
        on_cycle_ready=on_cycle,
    )

    base_ts = 1700001000000  # Exactly aligned with 15m and 30m candle boundaries
    bars = [make_sample_1m_bar("BTCUSDT", base_ts + i * 60000, 50000.0 + i) for i in range(120)]

    await pipeline.warmup_historical_bars("BTCUSDT", bars)

    # 1. Buffers populated
    bars_1m = pipeline.get_bars_1m("BTCUSDT")
    assert len(bars_1m) == 120
    assert len(pipeline.get_bars_15m("BTCUSDT")) > 0
    assert len(pipeline.get_bars_30m("BTCUSDT")) > 0

    # 2. ATR computed
    atrs = pipeline.get_atr_30m_values()
    assert atrs["BTCUSDT"] > 0.0

    # 3. ZERO live cycles dispatched during warmup
    assert len(cycle_dispatches) == 0
