"""Live Multi-Timeframe Bar & Feature Pipeline Application Service (Clean Architecture).

Consumes real-time 1m EnrichedMicroBarRecord streams, manages rolling bar buffers,
synthesizes 15m and 30m multi-timeframe candles, calculates microstructure and
regime metrics via QuantEnginePort, computes cross-asset lead-lag metrics, and
triggers Causal ML inference, OOD detection, and TreeSHAP attribution on 30m closes.

Strict Clean Architecture:
- Application layer depends ONLY on domain and ports.
- ZERO imports from adapters, infrastructure, or interfaces.
- Enforces Invariant #9: zero default parameters, zero default fields, zero silent fallbacks.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from trading_bot.application.use_cases.evaluate_tiered_funnel import TieredFunnelInput
from trading_bot.domain.entities.common.enums import MarketRegime
from trading_bot.domain.entities.market.market_state import (
    EnrichedMicroBarRecord,
    MicroBar1m,
    MultiTFBar,
    SemanticQuantPacket,
)
from trading_bot.domain.entities.ml.ml_configs import CoherenceGateConfig
from trading_bot.domain.entities.ml.ml_predictions import (
    SHAPExplanationResult,
)
from trading_bot.domain.entities.quant.quant_configs import RegimeConfig
from trading_bot.domain.entities.quant.quant_features import (
    MicrostructureMetrics,
    RegimeMetrics,
)
from trading_bot.domain.exceptions import InsufficientDataException
from trading_bot.domain.services.ml.coherence_gate import CoherenceGate
from trading_bot.domain.services.quant.fracdiff_transformer import FracDiffTransformerService
from trading_bot.domain.services.quant.regime_detector import RegimeDetectorService
from trading_bot.ports.ml_model_port import (
    CausalBarrierModelPort,
    OODDetectorPort,
    ProbabilityCalibratorPort,
    TreeSHAPExplainerPort,
)
from trading_bot.ports.quant_engine_port import QuantEnginePort

logger = logging.getLogger(__name__)

CANONICAL_FEATURE_NAMES: tuple[str, ...] = (
    "hurst_exponent_100",
    "shannon_entropy_50",
    "fracdiff_momentum_z",
    "vpin_toxicity_pct",
    "taker_cvd_z_24",
    "ofi_imbalance_z",
    "kyle_lambda_log",
    "oi_directional_score",
    "funding_crowded_pos_z",
    "funding_crowded_neg_z",
    "btc_lead_catchup_gap",
)


@dataclass
class _Enriched30mMeta:
    """Internal per-30m candle metadata supporting canonical 11-feature synthesis."""

    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    taker_buy_vol: float
    taker_buy_quote_vol: float
    ofi_sum: float
    vpin_bar: float
    kyle_lambda_log_bar: float
    vol_gk_ewma: float


def _compute_single_bar_gk_raw(
    *, open_p: float, high_p: float, low_p: float, close_p: float
) -> float:
    """Compute raw Garman-Klass volatility for a single OHLC bar."""
    co_factor = 2.0 * math.log(2.0) - 1.0
    log_hl = math.log(max(high_p / max(low_p, 1e-12), 1.0))
    log_co = math.log(max(close_p / max(open_p, 1e-12), 1e-12))
    gk_var = max(0.0, 0.5 * (log_hl**2) - co_factor * (log_co**2))
    return math.sqrt(gk_var)


def _compute_rolling_zscore_last(*, values: list[float], window: int) -> float:
    """Compute clipped [-3, 3] rolling z-score of the final element over window."""
    if len(values) < window or window <= 0:
        return 0.0
    tail = values[-window:]
    mean_v = sum(tail) / float(window)
    var_v = sum((x - mean_v) ** 2 for x in tail) / float(window)
    std_v = math.sqrt(var_v)
    z_val = (tail[-1] - mean_v) / (std_v + 1e-8)
    return max(-3.0, min(3.0, z_val))


class LiveBarPipelineService:
    """Production live multi-timeframe bar ingestion and online feature pipeline service."""

    def __init__(
        self,
        symbols: list[str],
        quant_engine: QuantEnginePort,
        barrier_model: CausalBarrierModelPort,
        ood_detector: OODDetectorPort,
        treeshap_explainer: TreeSHAPExplainerPort,
        on_30m_bar: (
            Callable[[str, TieredFunnelInput, SemanticQuantPacket], Awaitable[None]] | None
        ),
        on_cycle_ready: (
            Callable[
                [
                    datetime,
                    dict[str, TieredFunnelInput],
                    dict[str, SemanticQuantPacket],
                    dict[str, float],
                    dict[str, float],
                ],
                Awaitable[None],
            ]
            | None
        ),
    ) -> None:
        self._symbols: list[str] = [s.upper() for s in symbols]
        self._quant_engine: QuantEnginePort = quant_engine
        self._barrier_model: CausalBarrierModelPort = barrier_model
        self._ood_detector: OODDetectorPort = ood_detector
        self._treeshap_explainer: TreeSHAPExplainerPort = treeshap_explainer
        self._on_30m_bar: (
            Callable[[str, TieredFunnelInput, SemanticQuantPacket], Awaitable[None]] | None
        ) = on_30m_bar
        self._on_cycle_ready: (
            Callable[
                [
                    datetime,
                    dict[str, TieredFunnelInput],
                    dict[str, SemanticQuantPacket],
                    dict[str, float],
                    dict[str, float],
                ],
                Awaitable[None],
            ]
            | None
        ) = on_cycle_ready

        # Rolling buffers (max 300 1m bars, max 100 15m bars, max 100 30m bars)
        self._bars_1m: dict[str, list[EnrichedMicroBarRecord]] = {s: [] for s in self._symbols}
        self._bars_15m: dict[str, list[MultiTFBar]] = {s: [] for s in self._symbols}
        self._bars_30m: dict[str, list[MultiTFBar]] = {s: [] for s in self._symbols}
        self._meta_30m: dict[str, list[_Enriched30mMeta]] = {s: [] for s in self._symbols}
        self._va_long: ProbabilityCalibratorPort | None = None
        self._va_short: ProbabilityCalibratorPort | None = None
        self._fracdiff_weights: list[float] = FracDiffTransformerService().compute_weights(
            d=0.40,
            length=100,
            threshold=1e-4,
        )
        self._coherence_gate: CoherenceGate = CoherenceGate(
            config=CoherenceGateConfig(
                gamma_conflict=2.0,
                max_combined_probability=1.0,
                min_probability_floor=0.0,
            )
        )
        self._regime_svc: RegimeDetectorService = RegimeDetectorService()
        self._canonical_regime_cfg: RegimeConfig = RegimeConfig(
            hurst_window=100,
            yang_zhang_window=20,
            bipower_jump_confidence=0.99,
            entropy_bins=10,
            permutation_dimension=3,
            tsallis_q=2.0,
            division_eps=1e-12,
        )

        # Latest states and outputs
        self._latest_micro: dict[str, MicrostructureMetrics] = {}
        self._latest_regime: dict[str, RegimeMetrics] = {}
        self._latest_quant_packets: dict[str, SemanticQuantPacket] = {}
        self._latest_funnel_inputs: dict[str, TieredFunnelInput] = {}
        self._latest_shap_results: dict[str, SHAPExplanationResult] = {}
        self._current_prices: dict[str, float] = {}
        self._atr_30m: dict[str, float] = {}

        # History tracking for derived cross-features
        self._vpin_history: dict[str, list[float]] = {s: [] for s in self._symbols}
        self._btc_15m_returns: list[float] = []
        self._alt_15m_returns: dict[str, list[float]] = {s: [] for s in self._symbols}
        self._catchup_gaps: dict[str, list[float]] = {s: [] for s in self._symbols}

        # Cycle deduplication tracking (Zero Defaults, fail-fast)
        self._last_dispatched_cycle_ts_ms: int = 0
        self._cycle_closed_symbols: dict[int, set[str]] = {}

    async def on_1m_bar(self, bar: EnrichedMicroBarRecord) -> None:
        """Ingest a 1-minute enriched bar from the WebSocket ingestor."""
        symbol = bar.symbol.upper()
        if symbol not in self._bars_1m:
            self._bars_1m[symbol] = []
            self._bars_15m[symbol] = []
            self._bars_30m[symbol] = []
            self._meta_30m[symbol] = []
            self._vpin_history[symbol] = []
            self._alt_15m_returns[symbol] = []
            self._catchup_gaps[symbol] = []

        # 1. Append 1m bar to rolling buffer (capped at 300)
        self._bars_1m[symbol].append(bar)
        if len(self._bars_1m[symbol]) > 300:
            self._bars_1m[symbol].pop(0)

        self._current_prices[symbol] = bar.close

        # 2. Resample into 15m and 30m bars on each 1m close
        self._resample_15m(symbol, bar)
        self._resample_30m(symbol, bar)
        self._update_atr_30m(symbol)

        # 3. Compute microstructure metrics via QuantEnginePort
        micro_bars = [
            MicroBar1m(
                timestamp_ms=b.open_time_ms,
                open=b.open,
                high=b.high,
                low=b.low,
                close=b.close,
                volume=b.volume,
            )
            for b in self._bars_1m[symbol]
        ]
        micro_metrics = await self._quant_engine.compute_microstructure_metrics(symbol, micro_bars)
        self._latest_micro[symbol] = micro_metrics
        self._vpin_history[symbol].append(micro_metrics.vpin)
        if len(self._vpin_history[symbol]) > 50:
            self._vpin_history[symbol].pop(0)

        # 4. Compute regime state via QuantEnginePort
        # Select best available multi-TF bars (30m if >= 20, else 15m, else 1m wrapped)
        bars_for_regime = self._get_bars_for_regime(symbol)
        try:
            regime_metrics = await self._quant_engine.compute_regime_state(symbol, bars_for_regime)
        except InsufficientDataException:
            # Neutral default regime if insufficient history
            regime_metrics = RegimeMetrics(
                hurst_exponent=0.50,
                yang_zhang_vol=0.02,
                garman_klass_vol=0.02,
                parkinson_vol=0.02,
                bipower_variation=0.02,
                jump_z_score=0.0,
                has_volatility_jump=False,
                shannon_entropy=0.50,
                permutation_entropy=0.50,
                tsallis_entropy=0.50,
                regime_label=MarketRegime.RANDOM_WALK_NOISE,
            )
        self._latest_regime[symbol] = regime_metrics

        # 5. Update multi-asset returns and lead-lag statistics
        self._update_cross_asset_returns(symbol)

        # 6. Check if 30m candle boundary closed: (bar.open_time_ms + 60,000) % 1,800,000 == 0
        bar_close_ts = bar.open_time_ms + 60000
        if bar_close_ts % 1800000 == 0:
            await self._process_30m_candle_close(symbol, bar_close_ts)

    def set_calibrators(
        self,
        *,
        calibrator_long: ProbabilityCalibratorPort | None,
        calibrator_short: ProbabilityCalibratorPort | None,
    ) -> None:
        """Attach or hot-reload fitted Venn-Abers Long/Short calibrators."""
        self._va_long = calibrator_long
        self._va_short = calibrator_short

    async def warmup_historical_30m_bars(
        self,
        symbol: str,
        bars_30m: list[EnrichedMicroBarRecord],
    ) -> None:
        """Warm up rolling 30m buffers (350+ bars) for 336-bar funding z-score and 100-bar Hurst."""
        upper_sym = symbol.upper()
        if upper_sym not in self._bars_1m:
            self._bars_1m[upper_sym] = []
            self._bars_15m[upper_sym] = []
        self._bars_30m[upper_sym] = []
        self._meta_30m[upper_sym] = []
        if upper_sym not in self._vpin_history:
            self._vpin_history[upper_sym] = []
        if upper_sym not in self._alt_15m_returns:
            self._alt_15m_returns[upper_sym] = []
        if upper_sym not in self._catchup_gaps:
            self._catchup_gaps[upper_sym] = []

        now_ms = int(datetime.now(UTC).timestamp() * 1000.0)
        sorted_30m = sorted(bars_30m, key=lambda b: b.open_time_ms)
        for rec in sorted_30m:
            if rec.open_time_ms + 1800000 > now_ms:
                continue
            bar_30m = MultiTFBar(
                symbol=upper_sym,
                timeframe="30m",
                timestamp_ms=rec.open_time_ms,
                open=rec.open,
                high=rec.high,
                low=rec.low,
                close=rec.close,
                volume=rec.volume,
            )
            tot_usd = rec.volume * rec.close
            taker_usd = rec.taker_buy_vol * rec.close
            sv_flow = abs((2.0 * taker_usd) - tot_usd)
            dp = abs(rec.close - rec.open) / max(rec.open, 1e-12)
            kyle_raw = (dp * sv_flow) / ((sv_flow * sv_flow) + 1e-12)
            kyle_log = max(0.0, min(10.0, math.log(kyle_raw * 1e6 + 1.0)))
            vpin_val = max(
                0.0,
                min(
                    1.0,
                    abs((2.0 * rec.taker_buy_quote_vol) - rec.quote_volume)
                    / (rec.quote_volume + 1e-12),
                ),
            )
            self._upsert_30m_bar_and_meta(
                symbol=upper_sym,
                bar_30m=bar_30m,
                quote_volume=rec.quote_volume,
                taker_buy_vol=rec.taker_buy_vol,
                taker_buy_quote_vol=rec.taker_buy_quote_vol,
                ofi_sum=rec.ofi_sum,
                vpin_bar=vpin_val,
                kyle_lambda_log_bar=kyle_log,
            )

        if self._bars_30m[upper_sym]:
            self._current_prices[upper_sym] = self._bars_30m[upper_sym][-1].close
        self._update_atr_30m(upper_sym)

        logger.info(
            "Warmed up 30m buffer for %s with %d closed 30m bars.",
            upper_sym,
            len(self._bars_30m[upper_sym]),
        )

    async def warmup_historical_bars(self, symbol: str, bars: list[EnrichedMicroBarRecord]) -> None:
        """Warm up rolling buffers with historical 1m bars without triggering live trading cycles."""
        upper_sym = symbol.upper()
        self._bars_1m[upper_sym] = []
        self._bars_15m[upper_sym] = []
        if upper_sym not in self._bars_30m:
            self._bars_30m[upper_sym] = []
        if upper_sym not in self._meta_30m:
            self._meta_30m[upper_sym] = []
        self._vpin_history[upper_sym] = []
        self._alt_15m_returns[upper_sym] = []
        self._catchup_gaps[upper_sym] = []

        sorted_bars = sorted(bars, key=lambda b: b.open_time_ms)
        for b in sorted_bars:
            self._bars_1m[upper_sym].append(b)
            if len(self._bars_1m[upper_sym]) > 300:
                self._bars_1m[upper_sym].pop(0)

            self._current_prices[upper_sym] = b.close
            self._resample_15m(upper_sym, b)
            self._resample_30m(upper_sym, b)
            self._update_cross_asset_returns(upper_sym)

        self._update_atr_30m(upper_sym)

        if self._bars_1m[upper_sym]:
            micro_bars = [
                MicroBar1m(
                    timestamp_ms=b.open_time_ms,
                    open=b.open,
                    high=b.high,
                    low=b.low,
                    close=b.close,
                    volume=b.volume,
                )
                for b in self._bars_1m[upper_sym]
            ]
            micro_metrics = await self._quant_engine.compute_microstructure_metrics(
                upper_sym, micro_bars
            )
            self._latest_micro[upper_sym] = micro_metrics
            self._vpin_history[upper_sym].append(micro_metrics.vpin)
            if len(self._vpin_history[upper_sym]) > 50:
                self._vpin_history[upper_sym].pop(0)

            bars_for_regime = self._get_bars_for_regime(upper_sym)
            try:
                regime_metrics = await self._quant_engine.compute_regime_state(
                    upper_sym, bars_for_regime
                )
            except InsufficientDataException:
                regime_metrics = RegimeMetrics(
                    hurst_exponent=0.50,
                    yang_zhang_vol=0.02,
                    garman_klass_vol=0.02,
                    parkinson_vol=0.02,
                    bipower_variation=0.02,
                    jump_z_score=0.0,
                    has_volatility_jump=False,
                    shannon_entropy=0.50,
                    permutation_entropy=0.50,
                    tsallis_entropy=0.50,
                    regime_label=MarketRegime.RANDOM_WALK_NOISE,
                )
            self._latest_regime[upper_sym] = regime_metrics

        logger.info(
            "Warmed up rolling buffer for %s with %d 1m bars (15m: %d, 30m: %d)",
            upper_sym,
            len(self._bars_1m[upper_sym]),
            len(self._bars_15m[upper_sym]),
            len(self._bars_30m[upper_sym]),
        )

    def _get_bars_for_regime(self, symbol: str) -> list[MultiTFBar]:
        """Provide at least 20 MultiTFBars for regime calculations."""
        if len(self._bars_30m[symbol]) >= 20:
            return self._bars_30m[symbol]
        if len(self._bars_15m[symbol]) >= 20:
            return self._bars_15m[symbol]
        if len(self._bars_1m[symbol]) >= 20:
            return [
                MultiTFBar(
                    symbol=symbol,
                    timeframe="1m",
                    timestamp_ms=b.open_time_ms,
                    open=b.open,
                    high=b.high,
                    low=b.low,
                    close=b.close,
                    volume=b.volume,
                )
                for b in self._bars_1m[symbol]
            ]
        # If fewer than 20 1m bars exist, pad synthesized bars to prevent crash
        first_b = self._bars_1m[symbol][0]
        padded: list[MultiTFBar] = []
        for i in range(20 - len(self._bars_1m[symbol])):
            padded.append(
                MultiTFBar(
                    symbol=symbol,
                    timeframe="1m",
                    timestamp_ms=first_b.open_time_ms - (20 - i) * 60000,
                    open=first_b.open,
                    high=first_b.high,
                    low=first_b.low,
                    close=first_b.close,
                    volume=first_b.volume,
                )
            )
        for b in self._bars_1m[symbol]:
            padded.append(
                MultiTFBar(
                    symbol=symbol,
                    timeframe="1m",
                    timestamp_ms=b.open_time_ms,
                    open=b.open,
                    high=b.high,
                    low=b.low,
                    close=b.close,
                    volume=b.volume,
                )
            )
        return padded

    def _resample_15m(self, symbol: str, bar: EnrichedMicroBarRecord) -> None:
        """Resample 1m bars into 15m multi-timeframe bars."""
        interval_ms = 900000
        bucket_ts = (bar.open_time_ms // interval_ms) * interval_ms
        matching_1m = [
            b
            for b in self._bars_1m[symbol]
            if bucket_ts <= b.open_time_ms < bucket_ts + interval_ms
        ]
        if not matching_1m:
            return

        bar_15m = MultiTFBar(
            symbol=symbol,
            timeframe="15m",
            timestamp_ms=bucket_ts,
            open=matching_1m[0].open,
            high=max(b.high for b in matching_1m),
            low=min(b.low for b in matching_1m),
            close=matching_1m[-1].close,
            volume=sum(b.volume for b in matching_1m),
        )

        # On 15m close boundary, append bar to completed buffer
        if (bar.open_time_ms + 60000) % interval_ms == 0:
            self._bars_15m[symbol].append(bar_15m)
            if len(self._bars_15m[symbol]) > 100:
                self._bars_15m[symbol].pop(0)

    def _upsert_30m_bar_and_meta(
        self,
        *,
        symbol: str,
        bar_30m: MultiTFBar,
        quote_volume: float,
        taker_buy_vol: float,
        taker_buy_quote_vol: float,
        ofi_sum: float,
        vpin_bar: float,
        kyle_lambda_log_bar: float,
    ) -> None:
        """Append or update a closed 30m bar & aligned microstructure metadata (max 400 bars)."""
        bars_list = self._bars_30m[symbol]
        meta_list = self._meta_30m[symbol]
        gk_raw = _compute_single_bar_gk_raw(
            open_p=bar_30m.open,
            high_p=bar_30m.high,
            low_p=bar_30m.low,
            close_p=bar_30m.close,
        )
        ewma_alpha = 2.0 / 13.0

        # Check whether bucket already exists (e.g. historical warmup replaced by 1m aggregate)
        existing_idx: int | None = None
        for idx in range(len(bars_list) - 1, -1, -1):
            if bars_list[idx].timestamp_ms == bar_30m.timestamp_ms:
                existing_idx = idx
                break
            if bars_list[idx].timestamp_ms < bar_30m.timestamp_ms:
                break

        if existing_idx is not None:
            prev_ewma = meta_list[existing_idx - 1].vol_gk_ewma if existing_idx > 0 else gk_raw
            vol_gk_ewma = (
                max(0.0001, gk_raw)
                if existing_idx == 0
                else max(0.0001, ewma_alpha * gk_raw + (1.0 - ewma_alpha) * prev_ewma)
            )
            bars_list[existing_idx] = bar_30m
            meta_list[existing_idx] = _Enriched30mMeta(
                timestamp_ms=bar_30m.timestamp_ms,
                open=bar_30m.open,
                high=bar_30m.high,
                low=bar_30m.low,
                close=bar_30m.close,
                volume=bar_30m.volume,
                quote_volume=quote_volume,
                taker_buy_vol=taker_buy_vol,
                taker_buy_quote_vol=taker_buy_quote_vol,
                ofi_sum=ofi_sum,
                vpin_bar=vpin_bar,
                kyle_lambda_log_bar=kyle_lambda_log_bar,
                vol_gk_ewma=vol_gk_ewma,
            )
            return

        if meta_list:
            prev_ewma = meta_list[-1].vol_gk_ewma
            vol_gk_ewma = max(0.0001, ewma_alpha * gk_raw + (1.0 - ewma_alpha) * prev_ewma)
        else:
            vol_gk_ewma = max(0.0001, gk_raw)

        bars_list.append(bar_30m)
        meta_list.append(
            _Enriched30mMeta(
                timestamp_ms=bar_30m.timestamp_ms,
                open=bar_30m.open,
                high=bar_30m.high,
                low=bar_30m.low,
                close=bar_30m.close,
                volume=bar_30m.volume,
                quote_volume=quote_volume,
                taker_buy_vol=taker_buy_vol,
                taker_buy_quote_vol=taker_buy_quote_vol,
                ofi_sum=ofi_sum,
                vpin_bar=vpin_bar,
                kyle_lambda_log_bar=kyle_lambda_log_bar,
                vol_gk_ewma=vol_gk_ewma,
            )
        )
        if len(bars_list) > 400:
            bars_list.pop(0)
        if len(meta_list) > 400:
            meta_list.pop(0)

    def _resample_30m(self, symbol: str, bar: EnrichedMicroBarRecord) -> None:
        """Resample 1m bars into 30m multi-timeframe bars & compute 30m microstructure stats."""
        interval_ms = 1800000
        bucket_ts = (bar.open_time_ms // interval_ms) * interval_ms
        matching_1m = [
            b
            for b in self._bars_1m[symbol]
            if bucket_ts <= b.open_time_ms < bucket_ts + interval_ms
        ]
        if not matching_1m:
            return

        bar_30m = MultiTFBar(
            symbol=symbol,
            timeframe="30m",
            timestamp_ms=bucket_ts,
            open=matching_1m[0].open,
            high=max(b.high for b in matching_1m),
            low=min(b.low for b in matching_1m),
            close=matching_1m[-1].close,
            volume=sum(b.volume for b in matching_1m),
        )

        # On 30m close boundary, aggregate exact 1m VPIN, Kyle Lambda, QuoteVol, TakerBuy & OFI
        if (bar.open_time_ms + 60000) % interval_ms == 0:
            # Guard against overwriting a full 30m historical warmup bar with an incomplete leading 1m slice
            has_existing_bucket = any(b.timestamp_ms == bucket_ts for b in self._bars_30m[symbol])
            if has_existing_bucket and len(matching_1m) < 25:
                return
            q_vol_sum = sum(b.quote_volume for b in matching_1m)
            tb_vol_sum = sum(b.taker_buy_vol for b in matching_1m)
            tb_qvol_sum = sum(b.taker_buy_quote_vol for b in matching_1m)
            ofi_total = sum(b.ofi_sum for b in matching_1m)

            sv_q_sum = sum(abs((2.0 * b.taker_buy_quote_vol) - b.quote_volume) for b in matching_1m)
            vpin_val = max(0.0, min(1.0, sv_q_sum / (q_vol_sum + 1e-12)))

            sum_dp_sv = 0.0
            sum_sv_sq = 0.0
            for b in matching_1m:
                dp = abs(b.close - b.open) / max(b.open, 1e-12)
                tot_usd = b.volume * b.close
                taker_usd = b.taker_buy_vol * b.close
                sv = abs((2.0 * taker_usd) - tot_usd)
                sum_dp_sv += dp * sv
                sum_sv_sq += sv * sv
            kyle_raw = sum_dp_sv / (sum_sv_sq + 1e-12)
            kyle_log = max(0.0, min(10.0, math.log(kyle_raw * 1e6 + 1.0)))

            self._upsert_30m_bar_and_meta(
                symbol=symbol,
                bar_30m=bar_30m,
                quote_volume=q_vol_sum,
                taker_buy_vol=tb_vol_sum,
                taker_buy_quote_vol=tb_qvol_sum,
                ofi_sum=ofi_total,
                vpin_bar=vpin_val,
                kyle_lambda_log_bar=kyle_log,
            )

    def _update_atr_30m(self, symbol: str) -> None:
        """Compute rolling 30m Average True Range (ATR) for sizing."""
        bars_30m = self._bars_30m[symbol]
        if len(bars_30m) >= 2:
            recent_bars = bars_30m[-14:]
            trs: list[float] = []
            for i in range(1, len(recent_bars)):
                prev_c = recent_bars[i - 1].close
                cur_h = recent_bars[i].high
                cur_l = recent_bars[i].low
                tr = max(cur_h - cur_l, abs(cur_h - prev_c), abs(cur_l - prev_c))
                trs.append(tr)
            atr_val = (
                sum(trs) / float(len(trs))
                if trs
                else max(bars_30m[-1].high - bars_30m[-1].low, 1.0)
            )
            self._atr_30m[symbol] = atr_val
        else:
            # Fallback using 30m window of 1m bars
            recent_1m = self._bars_1m[symbol][-30:]
            h = max(b.high for b in recent_1m)
            low_val = min(b.low for b in recent_1m)
            self._atr_30m[symbol] = max(h - low_val, 1.0)

    def _update_cross_asset_returns(self, symbol: str) -> None:
        """Calculate BTC 15m return and altcoin lead-lag statistics."""
        if symbol == "BTCUSDT":
            if len(self._bars_1m["BTCUSDT"]) >= 15:
                p_now = self._bars_1m["BTCUSDT"][-1].close
                p_15m_ago = self._bars_1m["BTCUSDT"][-15].close
                if p_15m_ago > 0.0:
                    ret_15m = (p_now - p_15m_ago) / p_15m_ago
                    self._btc_15m_returns.append(ret_15m)
                    if len(self._btc_15m_returns) > 50:
                        self._btc_15m_returns.pop(0)
        else:
            if len(self._bars_1m[symbol]) >= 15:
                p_now = self._bars_1m[symbol][-1].close
                p_15m_ago = self._bars_1m[symbol][-15].close
                if p_15m_ago > 0.0:
                    ret_alt = (p_now - p_15m_ago) / p_15m_ago
                    self._alt_15m_returns[symbol].append(ret_alt)
                    if len(self._alt_15m_returns[symbol]) > 50:
                        self._alt_15m_returns[symbol].pop(0)

            # Update lead-lag catchup gap for altcoin
            if len(self._btc_15m_returns) >= 5 and len(self._alt_15m_returns[symbol]) >= 5:
                n = min(len(self._btc_15m_returns), len(self._alt_15m_returns[symbol]))
                b_rets = self._btc_15m_returns[-n:]
                a_rets = self._alt_15m_returns[symbol][-n:]
                m_b = sum(b_rets) / float(n)
                m_a = sum(a_rets) / float(n)
                cov = sum(
                    (b - m_b) * (a - m_a) for b, a in zip(b_rets, a_rets, strict=True)
                ) / float(n)
                var_b = sum((b - m_b) ** 2 for b in b_rets) / float(n)
                beta = max(0.1, min(3.0, cov / var_b)) if var_b > 1e-8 else 1.0
                gap = beta * b_rets[-1] - a_rets[-1]
                self._catchup_gaps[symbol].append(gap)
                if len(self._catchup_gaps[symbol]) > 50:
                    self._catchup_gaps[symbol].pop(0)

    async def _process_30m_candle_close(self, symbol: str, close_ts_ms: int) -> None:
        """Execute quant packet compilation, ML inference, OOD, and TreeSHAP attribution."""
        # 1. Assemble complete SemanticQuantPacket via QuantEnginePort
        quant_packet = await self._quant_engine.evaluate_quant_packet(symbol, close_ts_ms)
        self._latest_quant_packets[symbol] = quant_packet

        bar_dt = datetime.fromtimestamp(close_ts_ms / 1000.0, tz=UTC)
        canonical_features_11 = self._compute_canonical_features_11(symbol)
        # 2. Causal Barrier Model ML Inference & Attribution (Fail-Safe Capital Protection)
        try:
            barrier_pred = self._barrier_model.predict(canonical_features_11)

            # 3. Conformal Subspace OOD Detection (6D subspace when configured, else full 11D)
            ood_features: list[float] = canonical_features_11
            if hasattr(self._ood_detector, "config"):
                ood_obj: Any = self._ood_detector
                ood_cfg_obj: Any = ood_obj.config
                if hasattr(ood_cfg_obj, "feature_names"):
                    ood_feature_names: tuple[str, ...] = tuple(
                        str(f_name) for f_name in ood_cfg_obj.feature_names
                    )
                    if len(ood_feature_names) != len(canonical_features_11):
                        name_to_idx = {
                            name: idx for idx, name in enumerate(CANONICAL_FEATURE_NAMES)
                        }
                        ood_features = [
                            canonical_features_11[name_to_idx[f]] for f in ood_feature_names
                        ]
            ood_result = self._ood_detector.predict(ood_features)

            # 4. Native TreeSHAP Attribution
            shap_result = self._treeshap_explainer.explain(
                canonical_features_11,
                CANONICAL_FEATURE_NAMES,
                self._barrier_model,
                symbol,
                bar_dt,
                "long_barrier",
            )
            self._latest_shap_results[symbol] = shap_result

            if self._va_long is not None and self._va_short is not None:
                cal_l = self._va_long.calibrate(barrier_pred.p_raw_long)
                cal_s = self._va_short.calibrate(barrier_pred.p_raw_short)
                coherent_res = self._coherence_gate.enforce_coherence(
                    p_raw_long=cal_l.p_calibrated,
                    p_raw_short=cal_s.p_calibrated,
                )
                p_long_val = coherent_res.p_long
                p_short_val = coherent_res.p_short
                raw_unc = (
                    cal_l.uncertainty_width
                    if p_long_val >= p_short_val
                    else cal_s.uncertainty_width
                )
                unc_width = max(1e-4, min(0.50, raw_unc))
            else:
                p_long_val = barrier_pred.p_long
                p_short_val = barrier_pred.p_short
                unc_width = max(0.01, min(0.50, barrier_pred.coherence_penalty + 0.02))
            is_ood_val = ood_result.is_ood
        except Exception as exc:
            logger.error(
                "ML model inference failed for %s on 30m candle close: %s. Enforcing Fail-Safe Capital Protection.",
                symbol,
                exc,
            )
            p_long_val = 0.0
            p_short_val = 0.0
            unc_width = 1.0
            is_ood_val = True

        # 5. Extract microstructure and regime features
        micro_metrics = self._latest_micro[symbol]
        regime_metrics = self._latest_regime[symbol]

        # 6. Calculate derived z-scores
        vpin_slope = self._compute_vpin_slope(symbol)
        liq_long_z, liq_short_z = self._compute_liquidation_zscores(symbol)
        btc_15m_return_z = self._compute_btc_15m_return_z()
        btc_lead_beta, catchup_gap_z = self._compute_lead_lag_metrics(symbol)

        # 7. Construct TieredFunnelInput
        funnel_input = TieredFunnelInput(
            symbol=symbol,
            spread_bps=micro_metrics.corwin_schultz_spread_bps,
            yang_zhang_vol=regime_metrics.yang_zhang_vol,
            garman_klass_vol=regime_metrics.garman_klass_vol,
            is_blackout=False,
            is_account_frozen=False,
            p_long_cal=p_long_val,
            p_short_cal=p_short_val,
            uncertainty_width=unc_width,
            is_ood=is_ood_val,
            hurst_exp=regime_metrics.hurst_exponent,
            oi_quadrant=quant_packet.oi_quadrant,
            vpin_percentile=min(100.0, max(0.0, micro_metrics.vpin * 100.0)),
            liq_long_z=liq_long_z,
            liq_short_z=liq_short_z,
            funding_z=canonical_features_11[8] - canonical_features_11[9],
            vpin_slope_3bars=vpin_slope,
            btc_15m_return_z=btc_15m_return_z,
            btc_lead_beta=btc_lead_beta,
            lead_lag_catchup_gap_z=catchup_gap_z,
        )
        self._latest_funnel_inputs[symbol] = funnel_input

        # 8. Notify individual 30m bar subscriber if present
        if self._on_30m_bar is not None:
            await self._on_30m_bar(symbol, funnel_input, quant_packet)

        # 9. Track symbols that have completed this specific 30m candle close
        if close_ts_ms not in self._cycle_closed_symbols:
            self._cycle_closed_symbols[close_ts_ms] = set()
        self._cycle_closed_symbols[close_ts_ms].add(symbol)

        # Trigger full cycle callback exactly once when all symbols for this candle close are ready
        all_completed = self._cycle_closed_symbols[close_ts_ms].issuperset(self._symbols)
        is_new_cycle = close_ts_ms > self._last_dispatched_cycle_ts_ms

        if all_completed and is_new_cycle and self._on_cycle_ready is not None:
            self._last_dispatched_cycle_ts_ms = close_ts_ms
            # Clean up older cycle records to prevent memory leak
            stale_keys = [ts for ts in self._cycle_closed_symbols if ts < close_ts_ms]
            for old_ts in stale_keys:
                del self._cycle_closed_symbols[old_ts]

            await self._on_cycle_ready(
                bar_dt,
                dict(self._latest_funnel_inputs),
                dict(self._latest_quant_packets),
                dict(self._current_prices),
                dict(self._atr_30m),
            )

    def _compute_vpin_slope(self, symbol: str) -> float:
        history = self._vpin_history[symbol]
        if len(history) >= 3:
            return (history[-1] - history[-3]) / 2.0
        return 0.0

    def _compute_liquidation_zscores(self, symbol: str) -> tuple[float, float]:
        bars = self._bars_1m[symbol]
        if len(bars) < 5:
            return 0.0, 0.0

        long_liqs = [b.liq_long_usd for b in bars]
        short_liqs = [b.liq_short_usd for b in bars]

        m_long = sum(long_liqs) / float(len(long_liqs))
        var_long = sum((x - m_long) ** 2 for x in long_liqs) / float(len(long_liqs))
        std_long = math.sqrt(var_long)
        z_long = (long_liqs[-1] - m_long) / std_long if std_long > 1e-4 else 0.0

        m_short = sum(short_liqs) / float(len(short_liqs))
        var_short = sum((x - m_short) ** 2 for x in short_liqs) / float(len(short_liqs))
        std_short = math.sqrt(var_short)
        z_short = (short_liqs[-1] - m_short) / std_short if std_short > 1e-4 else 0.0

        return z_long, z_short

    def _compute_btc_15m_return_z(self) -> float:
        if len(self._btc_15m_returns) >= 5:
            m = sum(self._btc_15m_returns) / float(len(self._btc_15m_returns))
            var = sum((r - m) ** 2 for r in self._btc_15m_returns) / float(
                len(self._btc_15m_returns)
            )
            std = math.sqrt(var)
            return (self._btc_15m_returns[-1] - m) / std if std > 1e-6 else 0.0
        if self._btc_15m_returns:
            return self._btc_15m_returns[-1] / 0.01
        return 0.0

    def _compute_lead_lag_metrics(self, symbol: str) -> tuple[float, float]:
        if symbol == "BTCUSDT":
            return 1.0, 0.0

        gaps = self._catchup_gaps[symbol]
        if len(gaps) >= 5:
            m = sum(gaps) / float(len(gaps))
            var = sum((g - m) ** 2 for g in gaps) / float(len(gaps))
            std = math.sqrt(var)
            z_gap = (gaps[-1] - m) / std if std > 1e-6 else 0.0
        else:
            z_gap = 0.0

        if len(self._btc_15m_returns) >= 5 and len(self._alt_15m_returns[symbol]) >= 5:
            n = min(len(self._btc_15m_returns), len(self._alt_15m_returns[symbol]))
            b_rets = self._btc_15m_returns[-n:]
            a_rets = self._alt_15m_returns[symbol][-n:]
            m_b = sum(b_rets) / float(n)
            m_a = sum(a_rets) / float(n)
            cov = sum((b - m_b) * (a - m_a) for b, a in zip(b_rets, a_rets, strict=True)) / float(n)
            var_b = sum((b - m_b) ** 2 for b in b_rets) / float(n)
            beta = max(0.1, min(3.0, cov / var_b)) if var_b > 1e-8 else 1.0
        else:
            beta = 1.0

        return beta, z_gap

    def _compute_canonical_features_11(self, symbol: str) -> list[float]:
        """Synthesize the exact 11 CANONICAL_FEATURES matching 36M training pipeline."""
        meta_list = self._meta_30m[symbol]
        n_bars = len(meta_list)
        if n_bars == 0:
            return [0.50, 0.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        closes = [m.close for m in meta_list]
        log_rets = [
            math.log(max(closes[i], 1e-12) / max(closes[i - 1], 1e-12)) for i in range(1, n_bars)
        ]

        # 1. hurst_exponent_100 (100-bar 30m log returns via Anis-Lloyd)
        if len(log_rets) >= 20:
            hurst_win = log_rets[-100:]
            hurst_val = self._regime_svc.compute_anis_lloyd_hurst(
                hurst_win,
                self._canonical_regime_cfg,
            )
        else:
            hurst_val = 0.50

        # 2. shannon_entropy_50 (50-bar 30m log returns, 10 bins)
        if len(log_rets) >= 10:
            shannon_win = log_rets[-50:]
            shannon_val = self._regime_svc.compute_shannon_entropy(
                shannon_win,
                self._canonical_regime_cfg,
            )
        else:
            shannon_val = 0.50

        # 3. fracdiff_momentum_z (d=0.40, length<=100, thr=1e-4, z_window=24)
        w = self._fracdiff_weights
        w_len = len(w)
        z_window = 24
        if n_bars >= w_len + z_window:
            log_c = [math.log(max(c, 1e-12)) for c in closes]
            fd_vals: list[float] = []
            for t in range(n_bars - z_window, n_bars):
                acc = sum(w[k] * log_c[t - k] for k in range(w_len))
                fd_vals.append(acc)
            fracdiff_z = _compute_rolling_zscore_last(values=fd_vals, window=z_window)
        else:
            fracdiff_z = 0.0

        # 4. vpin_toxicity_pct (per-bar quote volume imbalance ratio)
        vpin_val = max(0.0, min(1.0, meta_list[-1].vpin_bar))

        # 5. taker_cvd_z_24 (24-bar z-score of 24-bar rolling net taker volume)
        net_taker_arr = [(2.0 * m.taker_buy_vol) - m.volume for m in meta_list]
        if n_bars >= z_window * 2:
            rolling_cvd: list[float] = []
            for idx in range(n_bars - z_window, n_bars):
                rolling_cvd.append(sum(net_taker_arr[idx - z_window + 1 : idx + 1]))
            cvd_z = _compute_rolling_zscore_last(values=rolling_cvd, window=z_window)
        else:
            cvd_z = 0.0

        # 6. ofi_imbalance_z (24-bar z-score of 30m ofi_sum)
        ofi_arr = [m.ofi_sum for m in meta_list]
        ofi_z = _compute_rolling_zscore_last(values=ofi_arr, window=z_window)

        # 7. kyle_lambda_log
        kyle_val = max(0.0, min(10.0, meta_list[-1].kyle_lambda_log_bar))

        # 8. oi_directional_score (6-bar 3h normalized price + net taker composite)
        mom_window = 6
        if n_bars > mom_window:
            cur = meta_list[-1]
            denom_vol = max(cur.vol_gk_ewma, 1e-4) * cur.close * math.sqrt(float(mom_window))
            zp = math.tanh((cur.close - closes[-1 - mom_window]) / max(denom_vol, 1e-12))
            vol_sum = sum(m.volume for m in meta_list[-mom_window:])
            net_sum = sum(net_taker_arr[-mom_window:])
            zf = net_sum / max(vol_sum, 1e-12)
            oi_score = max(-1.0, min(1.0, (0.5 * zp) + (0.5 * zf)))
        else:
            oi_score = 0.0

        # 9 & 10. funding_crowded_pos_z & neg_z (336-bar 7-day rolling VWAP premium z-score)
        funding_window = 336
        if n_bars >= funding_window:
            prem_arr = [
                ((m.quote_volume / max(m.volume, 1e-12)) - m.close) / max(m.close, 1e-12)
                for m in meta_list[-funding_window:]
            ]
            z_prem = _compute_rolling_zscore_last(values=prem_arr, window=funding_window)
            f_pos_z = max(0.0, min(3.0, z_prem))
            f_neg_z = max(0.0, min(3.0, -z_prem))
        else:
            f_pos_z = 0.0
            f_neg_z = 0.0

        # 11. btc_lead_catchup_gap (6h 12-bar BTC return minus asset return)
        if symbol != "BTCUSDT" and "BTCUSDT" in self._meta_30m:
            btc_meta = self._meta_30m["BTCUSDT"]
            if len(btc_meta) >= 13 and n_bars >= 13:
                btc_ret_6h = (btc_meta[-1].close / max(btc_meta[-13].close, 1e-12)) - 1.0
                sym_ret_6h = (closes[-1] / max(closes[-13], 1e-12)) - 1.0
                btc_gap = max(-3.0, min(3.0, btc_ret_6h - sym_ret_6h))
            else:
                btc_gap = 0.0
        else:
            btc_gap = 0.0

        return [
            hurst_val,
            shannon_val,
            fracdiff_z,
            vpin_val,
            cvd_z,
            ofi_z,
            kyle_val,
            oi_score,
            f_pos_z,
            f_neg_z,
            btc_gap,
        ]

    # -------------------------------------------------------------------------
    # Public Accessors & Status Queries
    # -------------------------------------------------------------------------
    def get_bars_1m(self, symbol: str) -> list[EnrichedMicroBarRecord]:
        """Fetch copy of rolling 1m bars for symbol."""
        upper_sym = symbol.upper()
        if upper_sym not in self._bars_1m:
            raise KeyError(f"No 1m bar buffer for symbol: {upper_sym}")
        return list(self._bars_1m[upper_sym])

    def get_bars_15m(self, symbol: str) -> list[MultiTFBar]:
        """Fetch copy of rolling 15m bars for symbol."""
        upper_sym = symbol.upper()
        if upper_sym not in self._bars_15m:
            raise KeyError(f"No 15m bar buffer for symbol: {upper_sym}")
        return list(self._bars_15m[upper_sym])

    def get_bars_30m(self, symbol: str) -> list[MultiTFBar]:
        """Fetch copy of rolling 30m bars for symbol."""
        upper_sym = symbol.upper()
        if upper_sym not in self._bars_30m:
            raise KeyError(f"No 30m bar buffer for symbol: {upper_sym}")
        return list(self._bars_30m[upper_sym])

    def get_latest_funnel_input(self, symbol: str) -> TieredFunnelInput:
        """Fetch latest TieredFunnelInput for symbol or raise KeyError."""
        upper_sym = symbol.upper()
        if upper_sym not in self._latest_funnel_inputs:
            raise KeyError(f"No 30m funnel input available yet for symbol: {upper_sym}")
        return self._latest_funnel_inputs[upper_sym]

    def get_latest_quant_packet(self, symbol: str) -> SemanticQuantPacket:
        """Fetch latest SemanticQuantPacket for symbol or raise KeyError."""
        upper_sym = symbol.upper()
        if upper_sym not in self._latest_quant_packets:
            raise KeyError(f"No 30m quant packet available yet for symbol: {upper_sym}")
        return self._latest_quant_packets[upper_sym]

    def get_all_latest_funnel_inputs(self) -> dict[str, TieredFunnelInput]:
        """Return shallow copy of latest funnel inputs across all symbols."""
        return dict(self._latest_funnel_inputs)

    def get_all_latest_quant_packets(self) -> dict[str, SemanticQuantPacket]:
        """Return shallow copy of latest quant packets across all symbols."""
        return dict(self._latest_quant_packets)

    def get_current_prices(self) -> dict[str, float]:
        """Return copy of latest close prices."""
        return dict(self._current_prices)

    def get_atr_30m_values(self) -> dict[str, float]:
        """Return copy of 30m ATR values."""
        return dict(self._atr_30m)

    def get_latest_shap_result(self, symbol: str) -> SHAPExplanationResult | None:
        """Return latest SHAPExplanationResult for symbol if available."""
        upper_sym = symbol.upper()
        if upper_sym in self._latest_shap_results:
            return self._latest_shap_results[upper_sym]
        return None

    def reload_models(
        self,
        barrier_model: CausalBarrierModelPort,
        ood_detector: OODDetectorPort,
        treeshap_explainer: TreeSHAPExplainerPort,
    ) -> None:
        """Hot-reload machine learning model ports in-memory with zero downtime."""
        self._barrier_model = barrier_model
        self._ood_detector = ood_detector
        self._treeshap_explainer = treeshap_explainer
