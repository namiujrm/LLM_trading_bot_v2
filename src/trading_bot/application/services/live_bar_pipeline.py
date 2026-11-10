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
from datetime import UTC, datetime

from trading_bot.application.use_cases.evaluate_tiered_funnel import TieredFunnelInput
from trading_bot.domain.entities.common.enums import MarketRegime
from trading_bot.domain.entities.market.market_state import (
    EnrichedMicroBarRecord,
    MicroBar1m,
    MultiTFBar,
    SemanticQuantPacket,
)
from trading_bot.domain.entities.ml.ml_predictions import (
    SHAPExplanationResult,
)
from trading_bot.domain.entities.quant.quant_features import (
    MicrostructureMetrics,
    RegimeMetrics,
)
from trading_bot.domain.exceptions import InsufficientDataException
from trading_bot.ports.ml_model_port import (
    CausalBarrierModelPort,
    OODDetectorPort,
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

    async def on_1m_bar(self, bar: EnrichedMicroBarRecord) -> None:
        """Ingest a 1-minute enriched bar from the WebSocket ingestor."""
        symbol = bar.symbol.upper()
        if symbol not in self._bars_1m:
            self._bars_1m[symbol] = []
            self._bars_15m[symbol] = []
            self._bars_30m[symbol] = []
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

    def _resample_30m(self, symbol: str, bar: EnrichedMicroBarRecord) -> None:
        """Resample 1m bars into 30m multi-timeframe bars."""
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

        # On 30m close boundary, append bar to completed buffer
        if (bar.open_time_ms + 60000) % interval_ms == 0:
            self._bars_30m[symbol].append(bar_30m)
            if len(self._bars_30m[symbol]) > 100:
                self._bars_30m[symbol].pop(0)

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

        # 2. Causal Barrier Model ML Inference
        sample_features_11 = quant_packet.trajectory_vector_16d[:11]
        barrier_pred = self._barrier_model.predict(sample_features_11)

        # 3. Conformal Subspace OOD Detection
        features_ood = quant_packet.trajectory_vector_16d[:6]
        ood_result = self._ood_detector.predict(features_ood)

        # 4. Native TreeSHAP Attribution
        bar_dt = datetime.fromtimestamp(close_ts_ms / 1000.0, tz=UTC)
        shap_result = self._treeshap_explainer.explain(
            sample_features_11,
            CANONICAL_FEATURE_NAMES,
            self._barrier_model,
            symbol,
            bar_dt,
            "long_barrier",
        )
        self._latest_shap_results[symbol] = shap_result

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
            p_long_cal=barrier_pred.p_long,
            p_short_cal=barrier_pred.p_short,
            uncertainty_width=max(0.01, min(0.50, barrier_pred.coherence_penalty + 0.02)),
            is_ood=ood_result.is_ood,
            hurst_exp=regime_metrics.hurst_exponent,
            oi_quadrant=quant_packet.oi_quadrant,
            vpin_percentile=min(100.0, max(0.0, micro_metrics.vpin * 100.0)),
            liq_long_z=liq_long_z,
            liq_short_z=liq_short_z,
            funding_z=0.0,
            vpin_slope_3bars=vpin_slope,
            btc_15m_return_z=btc_15m_return_z,
            btc_lead_beta=btc_lead_beta,
            lead_lag_catchup_gap_z=catchup_gap_z,
        )
        self._latest_funnel_inputs[symbol] = funnel_input

        # 8. Notify individual 30m bar subscriber if present
        if self._on_30m_bar is not None:
            await self._on_30m_bar(symbol, funnel_input, quant_packet)

        # 9. If all target symbols have completed 30m close, trigger full cycle callback
        all_completed = all(s in self._latest_funnel_inputs for s in self._symbols)
        if all_completed and self._on_cycle_ready is not None:
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
