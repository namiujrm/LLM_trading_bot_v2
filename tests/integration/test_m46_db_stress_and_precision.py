# pyright: reportPrivateUsage=false
"""Integration stress tests and numeric precision verification for TimescaleDB M4.6 persistence.

Tests:
1. High-throughput batch upsert with 5,000+ synthetic packets and conflict updates.
2. Concurrent read and write workloads on quant_feature_vectors_30m with zero deadlocks.
3. Numeric precision round-trip for calibrated probabilities, uncertainty intervals, OOD scores,
   and JSON TreeSHAP attributions.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tests.integration.test_walk_forward_evaluation import get_integration_db_config
from trading_bot.adapters.persistence.database import DatabaseManager
from trading_bot.adapters.persistence.ml_prediction_repo import (
    MLPredictionRepository,
    serialize_features_json,
    serialize_shap_explanation_json,
)
from trading_bot.domain.entities.ml.ml_predictions import (
    CalibratedProbability,
    OODDetectionResult,
    SHAPDriverContribution,
    UnifiedMLPredictionPacket,
)


def build_synthetic_packet(
    symbol: str,
    bar_time: datetime,
    raw_long: float,
    raw_short: float,
    cal_long: float,
    cal_short: float,
    uncertainty: float,
    m_dist: float,
    threshold: float,
    gamma_ood: float,
    shap_bull_val: float,
    shap_bear_val: float,
) -> UnifiedMLPredictionPacket:
    """Helper generating an explicit UnifiedMLPredictionPacket with zero defaults."""
    lower_l = max(0.0, cal_long - uncertainty * 0.5)
    upper_l = min(1.0, cal_long + uncertainty * 0.5)
    actual_unc_l = upper_l - lower_l

    lower_s = max(0.0, cal_short - uncertainty * 0.5)
    upper_s = min(1.0, cal_short + uncertainty * 0.5)
    actual_unc_s = upper_s - lower_s

    cal_prob_long = CalibratedProbability(
        p_calibrated=cal_long,
        p_lower=lower_l,
        p_upper=upper_l,
        uncertainty_width=actual_unc_l,
        is_uncertain=actual_unc_l > 0.15,
        raw_score=raw_long,
    )
    cal_prob_short = CalibratedProbability(
        p_calibrated=cal_short,
        p_lower=lower_s,
        p_upper=upper_s,
        uncertainty_width=actual_unc_s,
        is_uncertain=actual_unc_s > 0.15,
        raw_score=raw_short,
    )
    ood_result = OODDetectionResult(
        is_ood=m_dist > threshold,
        mahalanobis_distance=m_dist,
        empirical_quantile_threshold=threshold,
        excess_distance=max(0.0, m_dist - threshold),
        gamma_ood=gamma_ood,
        feature_contributions=(),
    )
    bull_driver = SHAPDriverContribution(
        feature_name="ofi_imbalance_z",
        feature_value=1.854321,
        shap_value=shap_bull_val,
        expected_sign=1,
        is_sign_consistent=True,
    )
    bear_driver = SHAPDriverContribution(
        feature_name="yang_zhang_vol_24",
        feature_value=0.012345,
        shap_value=shap_bear_val,
        expected_sign=-1,
        is_sign_consistent=True,
    )

    return UnifiedMLPredictionPacket(
        symbol=symbol,
        bar_time=bar_time,
        regime_label="STRESS_TEST",
        model_version="m46_stress_v1.0",
        p_long_raw=raw_long,
        p_short_raw=raw_short,
        p_long_calibrated=cal_prob_long,
        p_short_calibrated=cal_prob_short,
        ood_status=ood_result,
        top_bullish_shap=(bull_driver,),
        top_bearish_shap=(bear_driver,),
        is_economic_consistent=True,
        features_snapshot={
            "hurst_exponent_100": 0.584321,
            "yang_zhang_vol_24": 0.012345,
            "vpin_toxicity_pct": 0.421098,
            "kyle_lambda_log": 2.154321,
            "ofi_imbalance_z": 1.854321,
        },
    )


@pytest.mark.asyncio
async def test_high_throughput_batch_upsert_5000_packets() -> None:
    """Stress test: Ingest 5,000 synthetic prediction packets into TimescaleDB and measure latency."""
    db_cfg = get_integration_db_config()
    db = DatabaseManager(db_cfg)
    await db.initialize()

    test_symbol = "BTCUSDT_STR_5K"
    start_time = datetime(2023, 1, 1, 0, 0, tzinfo=UTC)
    packet_count = 5000
    chunk_size = 500

    try:
        repo = MLPredictionRepository(db_manager=db, batch_chunk_size=chunk_size)

        # 1. Generate 5,000 packets
        packets: list[UnifiedMLPredictionPacket] = []
        for i in range(packet_count):
            t = start_time + timedelta(minutes=30 * i)
            p = build_synthetic_packet(
                symbol=test_symbol,
                bar_time=t,
                raw_long=0.55,
                raw_short=0.35,
                cal_long=0.56,
                cal_short=0.34,
                uncertainty=0.08,
                m_dist=1.5,
                threshold=3.0,
                gamma_ood=1.0,
                shap_bull_val=0.05,
                shap_bear_val=-0.03,
            )
            packets.append(p)

        # 2. Time batch insertion
        t_start = time.perf_counter()
        saved_count = await repo.save_prediction_packets_batch(packets)
        t_end = time.perf_counter()
        elapsed_sec = t_end - t_start

        throughput = packet_count / elapsed_sec
        print(
            f"\n[STRESS 5K INSERT] {saved_count} packets in {elapsed_sec:.3f}s ({throughput:.1f} pkt/s)"
        )

        assert saved_count == packet_count
        assert elapsed_sec < 10.0, f"Batch upsert took too long: {elapsed_sec:.2f}s"

        # 3. Time ON CONFLICT upsert (update all 5,000 packets)
        updated_packets: list[UnifiedMLPredictionPacket] = []
        for i in range(packet_count):
            t = start_time + timedelta(minutes=30 * i)
            p_up = build_synthetic_packet(
                symbol=test_symbol,
                bar_time=t,
                raw_long=0.60,
                raw_short=0.30,
                cal_long=0.62,
                cal_short=0.28,
                uncertainty=0.06,
                m_dist=2.1,
                threshold=3.0,
                gamma_ood=0.95,
                shap_bull_val=0.08,
                shap_bear_val=-0.04,
            )
            updated_packets.append(p_up)

        t_up_start = time.perf_counter()
        upserted_count = await repo.save_prediction_packets_batch(updated_packets)
        t_up_end = time.perf_counter()
        elapsed_up_sec = t_up_end - t_up_start

        up_throughput = packet_count / elapsed_up_sec
        print(
            f"[STRESS 5K ON CONFLICT UPDATE] {upserted_count} packets in {elapsed_up_sec:.3f}s ({up_throughput:.1f} pkt/s)"
        )

        assert upserted_count == packet_count

        # 4. Verify total count remains exactly 5,000 (no duplicate rows)
        row = await db.fetchrow(
            "SELECT count(*) as cnt FROM quant_feature_vectors_30m WHERE symbol = $1;",
            test_symbol,
        )
        assert row is not None
        assert row["cnt"] == packet_count

        # 5. Verify updated values
        latest = await repo.get_prediction_packet(symbol=test_symbol)
        assert latest is not None
        assert abs(latest.p_long_calibrated.p_calibrated - 0.62) < 1e-4

    finally:
        async with db.transaction() as conn:
            await conn.execute(
                "DELETE FROM quant_feature_vectors_30m WHERE symbol = $1;", test_symbol
            )
        await db.close()


@pytest.mark.asyncio
async def test_concurrent_reads_and_writes() -> None:
    """Stress test: Simulate 4 concurrent writers and 4 concurrent readers without deadlocks."""
    db_cfg = get_integration_db_config()
    db = DatabaseManager(db_cfg)
    await db.initialize()

    num_symbols = 4
    symbols = [f"BTCUSDT_CONC_{i}" for i in range(num_symbols)]
    packets_per_symbol = 500
    start_time = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)

    try:
        repo = MLPredictionRepository(db_manager=db, batch_chunk_size=100)

        async def writer_task(sym: str, offset: int) -> int:
            pkts: list[UnifiedMLPredictionPacket] = []
            for j in range(packets_per_symbol):
                t = start_time + timedelta(minutes=30 * (offset * packets_per_symbol + j))
                p = build_synthetic_packet(
                    symbol=sym,
                    bar_time=t,
                    raw_long=0.51,
                    raw_short=0.42,
                    cal_long=0.52,
                    cal_short=0.40,
                    uncertainty=0.10,
                    m_dist=1.8,
                    threshold=3.0,
                    gamma_ood=1.0,
                    shap_bull_val=0.04,
                    shap_bear_val=-0.02,
                )
                pkts.append(p)
            return await repo.save_prediction_packets_batch(pkts)

        async def reader_task(sym: str, queries_count: int) -> int:
            success = 0
            for _ in range(queries_count):
                await asyncio.sleep(0.01)
                latest = await repo.get_prediction_packet(sym)
                if latest is not None:
                    success += 1
                range_pkts = await repo.get_prediction_packets_range(
                    symbol=sym,
                    start_time=start_time,
                    end_time=start_time + timedelta(days=5),
                )
                if len(range_pkts) > 0:
                    success += 1
            return success

        # Launch 4 writers and 4 readers concurrently
        writers = [writer_task(sym=s, offset=idx) for idx, s in enumerate(symbols)]
        readers = [reader_task(sym=s, queries_count=10) for s in symbols]

        t_start = time.perf_counter()
        results = await asyncio.gather(*writers, *readers, return_exceptions=False)
        t_end = time.perf_counter()

        write_results = results[:num_symbols]
        read_results = results[num_symbols:]

        print(f"\n[CONCURRENCY TEST] 4 writers + 4 readers completed in {t_end - t_start:.3f}s")
        print(f"Write outputs: {write_results}")
        print(f"Read outputs: {read_results}")

        for w_count in write_results:
            assert w_count == packets_per_symbol

    finally:
        async with db.transaction() as conn:
            for s in symbols:
                await conn.execute("DELETE FROM quant_feature_vectors_30m WHERE symbol = $1;", s)
        await db.close()


@pytest.mark.asyncio
async def test_numeric_precision_round_trip() -> None:
    """Verify numeric precision and identify any precision truncation or loss."""
    db_cfg = get_integration_db_config()
    db = DatabaseManager(db_cfg)
    await db.initialize()

    test_symbol = "BTCUSDT_PRECIS"
    t = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)

    # Use 8-decimal precision inputs
    raw_long_in = 0.54321987
    raw_short_in = 0.41238765
    cal_long_in = 0.55123456
    cal_short_in = 0.40123456
    unc_in = 0.08123456
    m_dist_in = 2.45678912
    threshold_in = 3.12345678
    gamma_ood_in = 0.87654321
    shap_bull_in = 0.04567891
    shap_bear_in = -0.02345678

    packet = build_synthetic_packet(
        symbol=test_symbol,
        bar_time=t,
        raw_long=raw_long_in,
        raw_short=raw_short_in,
        cal_long=cal_long_in,
        cal_short=cal_short_in,
        uncertainty=unc_in,
        m_dist=m_dist_in,
        threshold=threshold_in,
        gamma_ood=gamma_ood_in,
        shap_bull_val=shap_bull_in,
        shap_bear_val=shap_bear_in,
    )

    try:
        repo = MLPredictionRepository(db_manager=db, batch_chunk_size=10)
        await repo.save_prediction_packet(packet)

        retrieved = await repo.get_prediction_packet(test_symbol)
        assert retrieved is not None

        # Check calibrated probabilities in DOUBLE PRECISION columns
        cal_l_err = abs(retrieved.p_long_calibrated.p_calibrated - cal_long_in)
        cal_s_err = abs(retrieved.p_short_calibrated.p_calibrated - cal_short_in)
        unc_l_err = abs(
            retrieved.p_long_calibrated.uncertainty_width
            - packet.p_long_calibrated.uncertainty_width
        )
        m_dist_err = abs(retrieved.ood_status.mahalanobis_distance - m_dist_in)

        print("\n[PRECISION REPORT]")
        print(f"p_long_calibrated delta: {cal_l_err:.2e}")
        print(f"p_short_calibrated delta: {cal_s_err:.2e}")
        print(f"uncertainty_width delta: {unc_l_err:.2e}")
        print(f"mahalanobis_dist delta:  {m_dist_err:.2e}")

        # Check fields in JSON
        raw_l_err = abs(retrieved.p_long_raw - raw_long_in)
        raw_s_err = abs(retrieved.p_short_raw - raw_short_in)
        gamma_err = abs(retrieved.ood_status.gamma_ood - gamma_ood_in)
        thresh_err = abs(retrieved.ood_status.empirical_quantile_threshold - threshold_in)
        shap_bull_err = abs(retrieved.top_bullish_shap[0].shap_value - shap_bull_in)
        shap_bear_err = abs(retrieved.top_bearish_shap[0].shap_value - shap_bear_in)

        print(
            f"p_long_raw delta:        {raw_l_err:.2e} (input: {raw_long_in}, retrieved: {retrieved.p_long_raw})"
        )
        print(
            f"p_short_raw delta:       {raw_s_err:.2e} (input: {raw_short_in}, retrieved: {retrieved.p_short_raw})"
        )
        print(
            f"gamma_ood delta:         {gamma_err:.2e} (input: {gamma_ood_in}, retrieved: {retrieved.ood_status.gamma_ood})"
        )
        print(
            f"threshold delta:         {thresh_err:.2e} (input: {threshold_in}, retrieved: {retrieved.ood_status.empirical_quantile_threshold})"
        )
        print(
            f"shap_bull delta:         {shap_bull_err:.2e} (input: {shap_bull_in}, retrieved: {retrieved.top_bullish_shap[0].shap_value})"
        )
        print(
            f"shap_bear delta:         {shap_bear_err:.2e} (input: {shap_bear_in}, retrieved: {retrieved.top_bearish_shap[0].shap_value})"
        )

        # Double precision table columns and JSONB fields preserve exact float
        assert cal_l_err < 1e-12
        assert cal_s_err < 1e-12
        assert unc_l_err < 1e-12
        assert m_dist_err < 1e-12
        assert raw_l_err < 1e-12
        assert raw_s_err < 1e-12
        assert gamma_err < 1e-12
        assert thresh_err < 1e-12
        assert shap_bull_err < 1e-12
        assert shap_bear_err < 1e-12

    finally:
        async with db.transaction() as conn:
            await conn.execute(
                "DELETE FROM quant_feature_vectors_30m WHERE symbol = $1;", test_symbol
            )
        await db.close()


def test_small_shap_values_preservation_without_truncation() -> None:
    """Verify that small SHAP values (< 0.5e-6) are preserved exactly without truncation, satisfying domain invariants."""
    t = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)

    # Valid positive SHAP value, but smaller than 0.5e-6
    small_bull_shap = 3e-7
    small_bear_shap = -3e-7

    packet = build_synthetic_packet(
        symbol="BTCUSDT_SMALL_SHAP",
        bar_time=t,
        raw_long=0.55,
        raw_short=0.35,
        cal_long=0.56,
        cal_short=0.34,
        uncertainty=0.08,
        m_dist=1.5,
        threshold=3.0,
        gamma_ood=1.0,
        shap_bull_val=small_bull_shap,
        shap_bear_val=small_bear_shap,
    )

    shap_json = serialize_shap_explanation_json(packet)
    features_json = serialize_features_json(packet)

    mock_row: dict[str, Any] = {
        "bar_time": packet.bar_time,
        "symbol": packet.symbol,
        "regime_label": packet.regime_label,
        "model_version": packet.model_version,
        "ood_mahalanobis_dist": packet.ood_status.mahalanobis_distance,
        "is_ood_warning": packet.ood_status.is_ood,
        "p_long_win_cal": packet.p_long_calibrated.p_calibrated,
        "p_long_uncertainty": packet.p_long_calibrated.uncertainty_width,
        "p_short_win_cal": packet.p_short_calibrated.p_calibrated,
        "p_short_uncertainty": packet.p_short_calibrated.uncertainty_width,
        "shap_explanation_json": shap_json,
        "features_json": features_json,
        "hurst_exp": 0.584321,
    }

    repo = MLPredictionRepository.__new__(MLPredictionRepository)
    retrieved = repo._parse_row_to_packet(mock_row)

    assert retrieved.top_bullish_shap[0].shap_value == small_bull_shap
    assert retrieved.top_bearish_shap[0].shap_value == small_bear_shap


@pytest.mark.asyncio
async def test_extension_columns_populated_in_db() -> None:
    """Verify that columns added in 002_ml_predictions_extension.sql are properly populated and retrieved.

    Columns in 002_ml_predictions_extension.sql:
        gamma_ood, ood_threshold, is_economic_consistent, p_long_raw, p_short_raw, is_coherent, coherence_penalty
    """
    db_cfg = get_integration_db_config()
    db = DatabaseManager(db_cfg)
    await db.initialize()

    test_symbol = "BTCUSDT_EXT_COL"
    t = datetime(2024, 7, 1, 0, 0, tzinfo=UTC)

    packet = build_synthetic_packet(
        symbol=test_symbol,
        bar_time=t,
        raw_long=0.75,
        raw_short=0.25,
        cal_long=0.72,
        cal_short=0.22,
        uncertainty=0.04,
        m_dist=4.5,
        threshold=2.5,
        gamma_ood=0.35,  # Explicit non-default value!
        shap_bull_val=0.06,
        shap_bear_val=-0.04,
    )

    try:
        repo = MLPredictionRepository(db_manager=db, batch_chunk_size=10)
        await repo.save_prediction_packet(packet)

        # Check raw row values directly in PostgreSQL
        row = await db.fetchrow(
            "SELECT gamma_ood, ood_threshold, is_economic_consistent, p_long_raw, p_short_raw, is_coherent, coherence_penalty "
            "FROM quant_feature_vectors_30m WHERE symbol = $1;",
            test_symbol,
        )
        assert row is not None
        assert abs(row["gamma_ood"] - 0.35) < 1e-6
        assert abs(row["ood_threshold"] - 2.5) < 1e-6
        assert row["is_economic_consistent"] is True
        assert abs(row["p_long_raw"] - 0.75) < 1e-6
        assert abs(row["p_short_raw"] - 0.25) < 1e-6
        assert row["is_coherent"] is True
        assert abs(row["coherence_penalty"] - 0.0) < 1e-6

        # Check repository retrieval
        retrieved = await repo.get_prediction_packet(test_symbol)
        assert retrieved is not None
        assert abs(retrieved.ood_status.gamma_ood - 0.35) < 1e-6
        assert abs(retrieved.ood_status.empirical_quantile_threshold - 2.5) < 1e-6
        assert abs(retrieved.p_long_raw - 0.75) < 1e-6
        assert abs(retrieved.p_short_raw - 0.25) < 1e-6
        assert retrieved.is_economic_consistent is True

    finally:
        async with db.transaction() as conn:
            await conn.execute(
                "DELETE FROM quant_feature_vectors_30m WHERE symbol = $1;", test_symbol
            )
        await db.close()
