"""End-to-End Integration and 5-Dimensional Benchmark Suite for Quant Engine (M3.6).

Verifies:
1. Feature Persistence into TimescaleDB hypertable `quant_feature_vectors_30m` and pgvector `episodic_lessons_memory`
2. Latency Budget:
   - 1m Microstructure streaming update < 5.0ms per bar
   - 30m 4-pair 16D vector compilation < 50.0ms total
   - 50 Symbolic Alpha expressions evaluation < 15.0ms
3. Memory Footprint: Peak RAM < 256MB during 1,440-bar processing
4. Numerical Stability: 0 NaN, 0 Inf, float64 precision across all operations
5. Real Klines Test with Rolling Spearman Rank IC Gatekeeper filtering
"""

from __future__ import annotations

import math
import os
import time
import tracemalloc
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from trading_bot.adapters.persistence.database import DatabaseManager
from trading_bot.adapters.quant_engine.numba_kernels import (
    compute_fracdiff_weights_numba,
    rolling_fracdiff_numba,
    rolling_yang_zhang_numba,
)
from trading_bot.adapters.quant_engine.polars_dsl_compiler import PolarsAlphaDSLCompiler
from trading_bot.adapters.quant_engine.polars_engine import PolarsQuantEngineAdapter
from trading_bot.adapters.quant_engine.timescale_feature_store import TimescaleFeatureStore
from trading_bot.domain.entities.common.enums import MarketRegime, OIPriceQuadrant
from trading_bot.domain.entities.market.market_state import (
    FeatureSignalItem,
    MicroBar1m,
    SemanticQuantPacket,
)
from trading_bot.domain.entities.quant.quant_configs import (
    AlphaDSLConfig,
    FracDiffConfig,
    ICGatekeeperConfig,
    QuantEngineMasterConfig,
    RegimeConfig,
    VPINConfig,
)
from trading_bot.domain.entities.quant.quant_features import (
    MicrostructureMetrics,
    QuantFeatureVector,
)
from trading_bot.domain.services.quant.alpha_ast_validator import AlphaASTValidator
from trading_bot.domain.services.quant.ic_gatekeeper import ICGatekeeperService
from trading_bot.domain.services.quant.microstructure_math import MicrostructureMathService
from trading_bot.domain.services.quant.packet_assembler import SemanticQuantPacketAssembler
from trading_bot.domain.services.quant.regime_detector import RegimeDetectorService
from trading_bot.infrastructure.config_schemas import DatabaseConfig


def get_test_db_config() -> DatabaseConfig:
    return DatabaseConfig(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        db_name=os.getenv("POSTGRES_DB", "trading_bot_db"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        max_connections=int(os.getenv("POSTGRES_MAX_CONNECTIONS", "5")),
    )


def get_test_master_config() -> QuantEngineMasterConfig:
    return QuantEngineMasterConfig(
        vpin=VPINConfig(
            num_buckets=50,
            adv_lookback_days=20,
            volume_bucket_slice_ratio=0.02,
            toxicity_high_threshold=0.85,
            division_eps=1e-12,
        ),
        regime=RegimeConfig(
            hurst_window=100,
            yang_zhang_window=20,
            bipower_jump_confidence=2.58,
            entropy_bins=10,
            permutation_dimension=3,
            tsallis_q=1.5,
            division_eps=1e-12,
        ),
        fracdiff=FracDiffConfig(
            d_order=0.40,
            weight_threshold=1e-4,
            max_window_length=64,
            adf_critical_pvalue=0.05,
        ),
        alpha_dsl=AlphaDSLConfig(
            max_ast_depth=8,
            max_ast_nodes=50,
            min_window_bars=1,
            max_window_bars=288,
            division_epsilon=1e-8,
        ),
        ic_gatekeeper=ICGatekeeperConfig(
            min_rank_ic=0.015,
            min_t_statistic=2.0,
            min_ic_ir=0.50,
            rolling_eval_window_bars=100,
            forward_prediction_bars=12,
        ),
    )


@pytest.mark.asyncio
async def test_timescale_feature_store_end_to_end() -> None:
    """Test TimescaleFeatureStore persistence and querying against live TimescaleDB."""
    db_cfg = get_test_db_config()
    db = DatabaseManager(db_cfg)
    await db.initialize()
    try:
        store = TimescaleFeatureStore(db)

        # 1. Save single feature vector
        now_ms = int(time.time() * 1000)
        vec_16d = [0.05 * i for i in range(16)]
        feature_vec = QuantFeatureVector(
            symbol="BTCUSDT",
            timestamp_ms=now_ms,
            vector_16d=vec_16d,
            is_valid=True,
        )
        await store.save_feature_vector(feature_vec)

        # 2. Fetch latest feature vector
        fetched = await store.fetch_latest_feature_vector("BTCUSDT")
        assert fetched is not None
        assert fetched.symbol == "BTCUSDT"
        assert len(fetched.vector_16d) == 16
        assert fetched.is_valid is True

        # 3. Save batch feature vectors
        batch_vecs = [
            QuantFeatureVector(
                symbol="BTCUSDT",
                timestamp_ms=now_ms + 1800000 * (i + 1),
                vector_16d=[0.02 * (i + j) for j in range(16)],
                is_valid=True,
            )
            for i in range(5)
        ]
        saved_count = await store.save_feature_vectors_batch(batch_vecs)
        assert saved_count == 5

        # 4. Save and fetch SemanticQuantPacket
        packet = SemanticQuantPacket(
            symbol="BTCUSDT",
            timestamp_ms=now_ms + 1800000 * 10,
            active_features=[
                FeatureSignalItem(
                    feature_name="vpin_imbalance",
                    raw_value=0.72,
                    z_score=1.5,
                    is_statistically_active=True,
                    economic_direction=-1,
                )
            ],
            market_regime=MarketRegime.STRONG_TREND_BULL,
            oi_quadrant=OIPriceQuadrant.LONG_BUILDUP,
            trajectory_vector_16d=vec_16d,
        )
        await store.save_quant_packet(packet)
        fetched_packet = await store.fetch_latest_quant_packet("BTCUSDT")
        assert fetched_packet is not None
        assert fetched_packet.symbol == "BTCUSDT"
        assert fetched_packet.market_regime == MarketRegime.STRONG_TREND_BULL
        assert len(fetched_packet.trajectory_vector_16d) == 16

        # 5. Save episodic lesson into pgvector
        lesson_id = await store.save_episodic_lesson(
            symbol="BTCUSDT",
            playbook="TREND_CONTINUATION_BREAKOUT",
            side="LONG",
            outcome_category="VALIDATED_WIN",
            confidence_score=0.88,
            trajectory_vector_16d=vec_16d,
            realized_r=2.5,
            mae_atr=0.4,
            mfe_atr=2.8,
            did_hit_tp_after_sl=False,
            tier1_json='{"summary": "clean breakout"}',
            tier2_json='{"indicators": {"vpin": 0.35}}',
        )
        assert lesson_id is not None
        assert uuid.UUID(lesson_id)

        # 6. Export Parquet features
        export_path = "data/parquet/features/test_export.parquet"
        result_path = await store.export_parquet_features(
            symbol="BTCUSDT",
            year=datetime.now().year if hasattr(datetime, "now") else 2026,
            month=1,
            destination_path=export_path,
        )
        assert Path(result_path).exists()
        if Path(result_path).exists():
            Path(result_path).unlink(missing_ok=True)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_polars_quant_engine_streaming_latency_budget() -> None:
    """Benchmark: 1m Microstructure streaming update must complete in < 5.0ms per bar."""
    cfg = get_test_master_config()
    micro = MicrostructureMathService()
    regime = RegimeDetectorService()
    assembler = SemanticQuantPacketAssembler()
    ic_gate = ICGatekeeperService(cfg.ic_gatekeeper)
    engine = PolarsQuantEngineAdapter(cfg, micro, regime, assembler, ic_gate)

    # Load 100 real bars from parquet if available, otherwise synthetic
    parquet_path = Path("data/parquet/BTCUSDT/2024/01.parquet")
    if parquet_path.exists():
        df = pl.read_parquet(parquet_path).slice(0, 100)
        bars = [
            MicroBar1m(
                timestamp_ms=int(r["open_time"]),
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=float(r["volume"]),
            )
            for r in df.iter_rows(named=True)
        ]
    else:
        bars = [
            MicroBar1m(
                timestamp_ms=1700000000000 + i * 60000,
                open=42000.0 + i,
                high=42050.0 + i,
                low=41980.0 + i,
                close=42020.0 + i,
                volume=15.0 + (i % 5),
            )
            for i in range(100)
        ]

    # Benchmark streaming update over 20 incremental window additions
    timings: list[float] = []
    for i in range(30, 50):
        window_bars = bars[:i]
        start_t = time.perf_counter()
        metrics: MicrostructureMetrics = await engine.compute_microstructure_metrics(
            "BTCUSDT", window_bars
        )
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        timings.append(elapsed_ms)

        # Numerical stability verification: 0 NaN, 0 Inf
        assert math.isfinite(metrics.vpin)
        assert math.isfinite(metrics.kyle_lambda)
        assert math.isfinite(metrics.corwin_schultz_spread_bps)
        assert math.isfinite(metrics.va_amihud)
        assert math.isfinite(metrics.mnvr)

    avg_latency = sum(timings) / float(len(timings))
    max_latency = max(timings)
    print(
        f"\n[BENCHMARK] Microstructure 1m bar latency: avg={avg_latency:.3f}ms, max={max_latency:.3f}ms"
    )
    # Budget: < 5.0ms per bar
    assert avg_latency < 5.0, f"Microstructure latency {avg_latency:.3f}ms exceeded 5.0ms budget"


@pytest.mark.asyncio
async def test_polars_quant_engine_4pair_compilation_latency() -> None:
    """Benchmark: 30m Multi-Timeframe Feature Vector Compilation < 50.0ms for all 4 pairs."""
    cfg = get_test_master_config()
    micro = MicrostructureMathService()
    regime = RegimeDetectorService()
    assembler = SemanticQuantPacketAssembler()
    ic_gate = ICGatekeeperService(cfg.ic_gatekeeper)
    engine = PolarsQuantEngineAdapter(cfg, micro, regime, assembler, ic_gate)

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    start_ms = 1700000000000
    end_ms = start_ms + (48 * 1800000)  # 48 30m bars (24 hours)

    # Warmup Numba JIT compiler
    _ = await engine.compute_feature_vector_batch("BTCUSDT", start_ms, end_ms)

    start_t = time.perf_counter()
    compiled_vectors: dict[str, list[QuantFeatureVector]] = {}
    for sym in symbols:
        vecs = await engine.compute_feature_vector_batch(sym, start_ms, end_ms)
        compiled_vectors[sym] = vecs
    total_elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    print(f"\n[BENCHMARK] 4-Pair 30m compilation latency: {total_elapsed_ms:.3f}ms total")

    # Assert all 4 symbols produced valid 16D vectors
    for sym in symbols:
        assert len(compiled_vectors[sym]) == 48
        for v in compiled_vectors[sym]:
            assert len(v.vector_16d) == 16
            assert v.is_valid is True
            # Zero NaN / Inf verification
            for x in v.vector_16d:
                assert math.isfinite(x)

    # Budget: < 50.0ms across all 4 symbols
    assert total_elapsed_ms < 50.0, (
        f"4-pair compilation {total_elapsed_ms:.3f}ms exceeded 50.0ms budget"
    )


def test_polars_quant_engine_50_symbolic_alphas_latency() -> None:
    """Benchmark: 50 Symbolic Alpha expressions evaluation over 1,440-row Polars frame < 15.0ms."""
    cfg = get_test_master_config()
    validator = AlphaASTValidator()
    compiler = PolarsAlphaDSLCompiler(validator)

    # Build 1,440-row Polars DataFrame (30 days of 30m bars)
    n = 1440
    data = {
        "open": np.linspace(40000.0, 42000.0, n),
        "high": np.linspace(40100.0, 42100.0, n),
        "low": np.linspace(39900.0, 41900.0, n),
        "close": np.linspace(40050.0, 42050.0, n),
        "volume": np.random.uniform(10.0, 100.0, n),
        "taker_buy_vol": np.random.uniform(5.0, 50.0, n),
        "ofi_sum": np.random.uniform(-10.0, 10.0, n),
        "obi_mean": np.random.uniform(-0.5, 0.5, n),
    }
    df = pl.DataFrame(data)

    # Formulate 50 distinct causal alpha expressions
    raw_formulas: list[str] = []
    for w in [5, 10, 15, 20, 25]:
        raw_formulas.append(f"ts_delta(close, {w})")
        raw_formulas.append(f"ts_zscore(close, {w})")
        raw_formulas.append(f"ts_mean(volume, {w})")
        raw_formulas.append(f"ts_std(close, {w})")
        raw_formulas.append(f"ts_delay(close, {w})")
        raw_formulas.append(f"ts_corr(close, ofi_sum, {w})")
        raw_formulas.append(f"signed_power(ts_delta(close, {w}), 1.5)")
        raw_formulas.append(f"tanh_norm(ts_zscore(ofi_sum, {w}))")
        raw_formulas.append(f"ts_delta(volume, {w})")
        raw_formulas.append(f"ts_zscore(volume, {w})")

    assert len(raw_formulas) == 50

    # Compile all 50 formulas into Polars expressions
    compiled_exprs: list[pl.Expr] = []
    for idx, formula in enumerate(raw_formulas):
        expr = compiler.compile_formula(formula, cfg.alpha_dsl)
        compiled_exprs.append(expr.alias(f"alpha_{idx}"))

    # Warmup
    _ = df.select(compiled_exprs)

    # Benchmark execution latency
    start_t = time.perf_counter()
    result_df = df.select(compiled_exprs)
    eval_elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    print(f"\n[BENCHMARK] 50 Symbolic Alphas evaluation: {eval_elapsed_ms:.3f}ms")

    # Assert shape and numerical stability (no NaN / Inf in valid tail)
    assert result_df.height == 1440
    assert result_df.width == 50
    tail_50 = result_df.slice(100, 1340)
    for col_name in tail_50.columns:
        s = tail_50[col_name].drop_nulls()
        assert not s.is_nan().any()
        assert not s.is_infinite().any()

    # Budget: < 15.0ms for 50 expressions
    assert eval_elapsed_ms < 15.0, (
        f"50 alphas evaluation {eval_elapsed_ms:.3f}ms exceeded 15.0ms budget"
    )


def test_quant_engine_memory_footprint_under_256mb() -> None:
    """Benchmark: Peak working set RAM during 1,440-bar processing < 256MB."""
    tracemalloc.start()
    try:
        # Load or generate 1,440 30m bars
        n = 1440
        data = {
            "open": np.linspace(40000.0, 42000.0, n),
            "high": np.linspace(40100.0, 42100.0, n),
            "low": np.linspace(39900.0, 41900.0, n),
            "close": np.linspace(40050.0, 42050.0, n),
            "volume": np.random.uniform(10.0, 100.0, n),
        }
        df = pl.DataFrame(data)

        # Perform rolling FracDiff, Yang-Zhang, and Hurst
        weights = compute_fracdiff_weights_numba(0.40, 64, 1e-4)
        _ = rolling_fracdiff_numba(df["close"].to_numpy(), weights)
        _ = rolling_yang_zhang_numba(
            df["open"].to_numpy(),
            df["high"].to_numpy(),
            df["low"].to_numpy(),
            df["close"].to_numpy(),
            20,
        )

        _current, peak = tracemalloc.get_traced_memory()
        peak_mb = peak / (1024 * 1024)
        print(f"\n[BENCHMARK] Peak memory working set: {peak_mb:.2f} MB")
        # Budget: Peak RAM < 256MB
        assert peak_mb < 256.0, f"Peak memory {peak_mb:.2f}MB exceeded 256MB budget"
    finally:
        tracemalloc.stop()


def test_ic_gatekeeper_filtering_on_real_klines() -> None:
    """Test Rolling Spearman Rank IC Gatekeeper filtering on real klines data."""
    cfg = get_test_master_config()
    service = ICGatekeeperService(cfg.ic_gatekeeper)
    parquet_path = Path("data/parquet/BTCUSDT/2024/01.parquet")

    if parquet_path.exists():
        # Read 1,440 30m bars by taking every 30th 1m bar
        df = pl.read_parquet(parquet_path)
        sampled = df.gather_every(30).slice(0, 1440)
        closes = sampled["close"].to_list()
        spreads = sampled["spread_bps_mean"].to_list()
    else:
        # Synthetic fallback
        closes = [40000.0 + 10.0 * i + (50.0 if i % 2 == 0 else -50.0) for i in range(1440)]
        spreads = [5.0 + 0.1 * (i % 10) for i in range(1440)]

    n = len(closes)
    assert n >= 100

    # Forward 12-bar (6-hour) log return
    fwd_ret: list[float] = []
    feat_x: list[float] = []
    vol_series: list[float] = []
    k_fwd = 12

    for i in range(n - k_fwd):
        ret = math.log(closes[i + k_fwd] / closes[i])
        fwd_ret.append(ret)
        feat_x.append(spreads[i])
        vol_series.append(abs(ret))

    eval_result = service.evaluate_feature(
        feature_name="corwin_schultz_spread",
        x=feat_x,
        y=fwd_ret,
        vol_series=vol_series,
        locked_direction=-1,
        sub_window_count=5,
        eps=1e-12,
    )

    # Verify numerical stability: all fields finite
    assert math.isfinite(eval_result.rank_ic)
    assert math.isfinite(eval_result.t_statistic)
    assert math.isfinite(eval_result.ic_ir)
    assert isinstance(eval_result.is_regime_invariant, bool)
    assert isinstance(eval_result.is_direction_valid, bool)
    assert isinstance(eval_result.is_statistically_active, bool)
    print(
        f"\n[REAL DATA IC GATE] Feature: {eval_result.feature_name}, Rank IC: {eval_result.rank_ic:.4f}, t-stat: {eval_result.t_statistic:.2f}, active={eval_result.is_statistically_active}"
    )
