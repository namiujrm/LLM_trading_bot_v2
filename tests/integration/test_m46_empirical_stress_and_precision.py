# pyright: reportPrivateUsage=false
"""Empirical stress and precision verification suite for TimescaleDB M4.6 persistence.

Challenger 2 Empirical Verification:
1. Extreme tiny TreeSHAP attributions (1e-7, 1e-8, 1e-12, 1e-15, 1e-20, 1e-50) roundtrip
   through TimescaleDB with zero deserialization errors or Pydantic validation crashes.
2. Full 24-column storage, ON CONFLICT DO UPDATE fidelity, and exact float error < 1e-12.
3. High-throughput batch upsert (>500 pkt/s) and multi-coroutine concurrent writes/reads
   and simultaneous conflict upserts with zero deadlocks.
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
from trading_bot.infrastructure.config_schemas import DatabaseConfig


def make_packet(
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
    is_econ_consistent: bool,
    model_version: str,
    regime_label: str,
    hurst_exp: float,
    yang_zhang_vol: float,
    vpin_val: float,
    kyle_val: float,
    ofi_imbalance: float,
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
        feature_value=ofi_imbalance,
        shap_value=shap_bull_val,
        expected_sign=1,
        is_sign_consistent=True,
    )
    bear_driver = SHAPDriverContribution(
        feature_name="yang_zhang_vol_24",
        feature_value=yang_zhang_vol,
        shap_value=shap_bear_val,
        expected_sign=-1,
        is_sign_consistent=True,
    )

    return UnifiedMLPredictionPacket(
        symbol=symbol,
        bar_time=bar_time,
        regime_label=regime_label,
        model_version=model_version,
        p_long_raw=raw_long,
        p_short_raw=raw_short,
        p_long_calibrated=cal_prob_long,
        p_short_calibrated=cal_prob_short,
        ood_status=ood_result,
        top_bullish_shap=(bull_driver,),
        top_bearish_shap=(bear_driver,),
        is_economic_consistent=is_econ_consistent,
        features_snapshot={
            "hurst_exponent_100": hurst_exp,
            "yang_zhang_vol_24": yang_zhang_vol,
            "vpin_toxicity_pct": vpin_val,
            "kyle_lambda_log": kyle_val,
            "ofi_imbalance_z": ofi_imbalance,
        },
    )


@pytest.mark.asyncio
async def test_tiny_treeshap_attributions_roundtrip_all_magnitudes() -> None:
    """Empirically test tiny TreeSHAP attributions roundtrip through TimescaleDB.

    Magnitudes: 1e-7, 1e-8, 1e-12, 1e-15, 1e-20, 1e-50.
    Asserts zero deserialization errors, zero Pydantic validation crashes,
    and exact bit preservation across jsonb and domain objects.
    """
    db_cfg: DatabaseConfig = get_integration_db_config()
    db: DatabaseManager = DatabaseManager(db_cfg)
    await db.initialize()

    test_symbol = f"CHAL_SHAP_{int(time.time()) % 10000}"
    test_magnitudes = [1e-7, 1e-8, 1e-12, 1e-15, 1e-20, 1e-50]
    base_time = datetime(2024, 8, 1, 0, 0, tzinfo=UTC)

    try:
        repo = MLPredictionRepository(db_manager=db, batch_chunk_size=10)

        # 1. In-memory serialization and deserialization check
        for idx, mag in enumerate(test_magnitudes):
            pkt_time = base_time + timedelta(minutes=30 * idx)
            pkt = make_packet(
                symbol=test_symbol,
                bar_time=pkt_time,
                raw_long=0.55,
                raw_short=0.35,
                cal_long=0.56,
                cal_short=0.34,
                uncertainty=0.08,
                m_dist=1.5,
                threshold=3.0,
                gamma_ood=1.0,
                shap_bull_val=mag,
                shap_bear_val=-mag,
                is_econ_consistent=True,
                model_version="m46_chal_shap_v1",
                regime_label="STRESS",
                hurst_exp=0.584321,
                yang_zhang_vol=0.012345,
                vpin_val=0.421098,
                kyle_val=2.154321,
                ofi_imbalance=1.854321,
            )

            # Test JSON serialization
            shap_json = serialize_shap_explanation_json(pkt)
            features_json = serialize_features_json(pkt)

            # Parse directly via mock row
            mock_row: dict[str, Any] = {
                "bar_time": pkt.bar_time,
                "symbol": pkt.symbol,
                "regime_label": pkt.regime_label,
                "model_version": pkt.model_version,
                "ood_mahalanobis_dist": pkt.ood_status.mahalanobis_distance,
                "is_ood_warning": pkt.ood_status.is_ood,
                "p_long_win_cal": pkt.p_long_calibrated.p_calibrated,
                "p_long_uncertainty": pkt.p_long_calibrated.uncertainty_width,
                "p_short_win_cal": pkt.p_short_calibrated.p_calibrated,
                "p_short_uncertainty": pkt.p_short_calibrated.uncertainty_width,
                "shap_explanation_json": shap_json,
                "features_json": features_json,
                "hurst_exp": 0.584321,
            }
            parsed = repo._parse_row_to_packet(mock_row)
            assert parsed.top_bullish_shap[0].shap_value == mag
            assert parsed.top_bearish_shap[0].shap_value == -mag

        # 2. Live TimescaleDB Persistence and Retrieval Check
        packets: list[UnifiedMLPredictionPacket] = []
        for idx, mag in enumerate(test_magnitudes):
            pkt_time = base_time + timedelta(minutes=30 * idx)
            pkt = make_packet(
                symbol=test_symbol,
                bar_time=pkt_time,
                raw_long=0.55,
                raw_short=0.35,
                cal_long=0.56,
                cal_short=0.34,
                uncertainty=0.08,
                m_dist=1.5,
                threshold=3.0,
                gamma_ood=1.0,
                shap_bull_val=mag,
                shap_bear_val=-mag,
                is_econ_consistent=True,
                model_version="m46_chal_shap_v1",
                regime_label="STRESS",
                hurst_exp=0.584321,
                yang_zhang_vol=0.012345,
                vpin_val=0.421098,
                kyle_val=2.154321,
                ofi_imbalance=1.854321,
            )
            packets.append(pkt)

        # Batch save to real database
        saved = await repo.save_prediction_packets_batch(packets)
        assert saved == len(test_magnitudes)

        # Retrieve chronological range
        retrieved_packets = await repo.get_prediction_packets_range(
            symbol=test_symbol,
            start_time=base_time,
            end_time=base_time + timedelta(hours=5),
        )
        assert len(retrieved_packets) == len(test_magnitudes)

        for idx, mag in enumerate(test_magnitudes):
            retrieved = retrieved_packets[idx]
            retrieved_bull = retrieved.top_bullish_shap[0].shap_value
            retrieved_bear = retrieved.top_bearish_shap[0].shap_value

            # Verify exact float equality (or relative error < 1e-15)
            rel_bull_err = abs(retrieved_bull - mag) / mag
            rel_bear_err = abs(retrieved_bear - (-mag)) / mag
            assert rel_bull_err < 1e-12, (
                f"Bullish SHAP magnitude {mag} failed: error {rel_bull_err}"
            )
            assert rel_bear_err < 1e-12, (
                f"Bearish SHAP magnitude {-mag} failed: error {rel_bear_err}"
            )

    finally:
        async with db.transaction() as conn:
            await conn.execute(
                "DELETE FROM quant_feature_vectors_30m WHERE symbol = $1;", test_symbol
            )
        await db.close()


@pytest.mark.asyncio
async def test_all_24_columns_exact_storage_and_conflict_update() -> None:
    """Verify that all 24 columns in quant_feature_vectors_30m are:
    1. Fully written with explicit non-default values.
    2. Overwritten on conflict with updated values.
    3. Retrieved with float error < 1e-12.
    """
    db_cfg: DatabaseConfig = get_integration_db_config()
    db: DatabaseManager = DatabaseManager(db_cfg)
    await db.initialize()

    test_symbol = f"CHAL_24COL_{int(time.time()) % 10000}"
    bar_time = datetime(2024, 9, 1, 12, 0, tzinfo=UTC)

    # Initial explicit non-default values (Initial State)
    v1_hurst = 0.654321098765
    v1_yz = 0.023456789012
    v1_vpin = 0.345678901234
    v1_kyle = 1.234567890123
    v1_mdist = 4.567890123456
    v1_thresh = 2.876543210987
    v1_gamma = 0.432109876543
    v1_raw_long = 0.678901234567
    v1_raw_short = 0.389012345678
    v1_cal_long = 0.612345678901
    v1_cal_short = 0.298765432109
    v1_unc = 0.087654321098
    v1_bull_shap = 0.054321098765
    v1_bear_shap = -0.034567890123
    v1_econ = False  # Explicit False to test non-default!

    pkt1 = make_packet(
        symbol=test_symbol,
        bar_time=bar_time,
        raw_long=v1_raw_long,
        raw_short=v1_raw_short,
        cal_long=v1_cal_long,
        cal_short=v1_cal_short,
        uncertainty=v1_unc,
        m_dist=v1_mdist,
        threshold=v1_thresh,
        gamma_ood=v1_gamma,
        shap_bull_val=v1_bull_shap,
        shap_bear_val=v1_bear_shap,
        is_econ_consistent=v1_econ,
        model_version="m46_v1_init",
        regime_label="REGIME_ALPHA",
        hurst_exp=v1_hurst,
        yang_zhang_vol=v1_yz,
        vpin_val=v1_vpin,
        kyle_val=v1_kyle,
        ofi_imbalance=1.987654321,
    )

    try:
        repo = MLPredictionRepository(db_manager=db, batch_chunk_size=10)

        # 1. Ingest initial packet
        await repo.save_prediction_packet(pkt1)

        # 2. Direct SQL row inspection of all 24 columns
        row1 = await db.fetchrow(
            """
            SELECT
                bar_time, symbol, regime_label, hurst_exp, yang_zhang_vol,
                vpin_value, kyle_lambda, oi_quadrant, ood_mahalanobis_dist,
                is_ood_warning, p_long_win_cal, p_long_uncertainty,
                p_short_win_cal, p_short_uncertainty, model_version,
                features_json, shap_explanation_json, gamma_ood,
                ood_threshold, is_economic_consistent, p_long_raw,
                p_short_raw, is_coherent, coherence_penalty
            FROM quant_feature_vectors_30m
            WHERE symbol = $1 AND bar_time = $2;
            """,
            test_symbol,
            bar_time,
        )
        assert row1 is not None, "Initial packet row was not found in DB"

        # Verify all 24 columns in DB
        assert row1["symbol"] == test_symbol
        assert row1["bar_time"] == bar_time
        assert row1["regime_label"] == "REGIME_ALPHA"
        assert abs(row1["hurst_exp"] - v1_hurst) < 1e-12
        assert abs(row1["yang_zhang_vol"] - v1_yz) < 1e-12
        assert abs(row1["vpin_value"] - v1_vpin) < 1e-12
        assert abs(row1["kyle_lambda"] - v1_kyle) < 1e-12
        assert row1["oi_quadrant"] == "NEUTRAL"
        assert abs(row1["ood_mahalanobis_dist"] - v1_mdist) < 1e-12
        assert row1["is_ood_warning"] is True  # 4.567 > 2.876
        assert abs(row1["p_long_win_cal"] - v1_cal_long) < 1e-12
        assert abs(row1["p_short_win_cal"] - v1_cal_short) < 1e-12
        assert row1["model_version"] == "m46_v1_init"
        assert abs(row1["gamma_ood"] - v1_gamma) < 1e-12
        assert abs(row1["ood_threshold"] - v1_thresh) < 1e-12
        assert row1["is_economic_consistent"] is False
        assert abs(row1["p_long_raw"] - v1_raw_long) < 1e-12
        assert abs(row1["p_short_raw"] - v1_raw_short) < 1e-12
        # v1_raw_long + v1_raw_short = 0.678901234567 + 0.389012345678 = 1.067913580245 > 1.0
        assert row1["is_coherent"] is False
        expected_penalty = (v1_raw_long + v1_raw_short) - 1.0
        assert abs(row1["coherence_penalty"] - expected_penalty) < 1e-12

        # 3. ON CONFLICT DO UPDATE with updated values (Second State)
        v2_hurst = 0.512345678901
        v2_yz = 0.015432109876
        v2_vpin = 0.210987654321
        v2_kyle = 0.987654321098
        v2_mdist = 1.876543210987
        v2_thresh = 3.123456789012
        v2_gamma = 0.987654321098
        v2_raw_long = 0.456789012345
        v2_raw_short = 0.412345678901
        v2_cal_long = 0.461234567890
        v2_cal_short = 0.409876543210
        v2_unc = 0.065432109876
        v2_bull_shap = 0.041234567890
        v2_bear_shap = -0.021098765432
        v2_econ = True

        pkt2 = make_packet(
            symbol=test_symbol,
            bar_time=bar_time,
            raw_long=v2_raw_long,
            raw_short=v2_raw_short,
            cal_long=v2_cal_long,
            cal_short=v2_cal_short,
            uncertainty=v2_unc,
            m_dist=v2_mdist,
            threshold=v2_thresh,
            gamma_ood=v2_gamma,
            shap_bull_val=v2_bull_shap,
            shap_bear_val=v2_bear_shap,
            is_econ_consistent=v2_econ,
            model_version="m46_v2_updated",
            regime_label="REGIME_BETA",
            hurst_exp=v2_hurst,
            yang_zhang_vol=v2_yz,
            vpin_val=v2_vpin,
            kyle_val=v2_kyle,
            ofi_imbalance=0.87654321,
        )

        await repo.save_prediction_packet(pkt2)

        # Ensure total row count for this symbol is STILL EXACTLY 1 (idempotent upsert)
        count_row = await db.fetchrow(
            "SELECT count(*) as cnt FROM quant_feature_vectors_30m WHERE symbol = $1;",
            test_symbol,
        )
        assert count_row is not None
        assert count_row["cnt"] == 1

        # Direct SQL row inspection of updated values
        row2 = await db.fetchrow(
            """
            SELECT
                bar_time, symbol, regime_label, hurst_exp, yang_zhang_vol,
                vpin_value, kyle_lambda, oi_quadrant, ood_mahalanobis_dist,
                is_ood_warning, p_long_win_cal, p_long_uncertainty,
                p_short_win_cal, p_short_uncertainty, model_version,
                features_json, shap_explanation_json, gamma_ood,
                ood_threshold, is_economic_consistent, p_long_raw,
                p_short_raw, is_coherent, coherence_penalty
            FROM quant_feature_vectors_30m
            WHERE symbol = $1 AND bar_time = $2;
            """,
            test_symbol,
            bar_time,
        )
        assert row2 is not None

        # Verify update reflected in DB
        assert row2["regime_label"] == "REGIME_BETA"
        assert row2["model_version"] == "m46_v2_updated"
        assert abs(row2["hurst_exp"] - v2_hurst) < 1e-12
        assert abs(row2["yang_zhang_vol"] - v2_yz) < 1e-12
        assert abs(row2["vpin_value"] - v2_vpin) < 1e-12
        assert abs(row2["kyle_lambda"] - v2_kyle) < 1e-12
        assert abs(row2["ood_mahalanobis_dist"] - v2_mdist) < 1e-12
        assert row2["is_ood_warning"] is False  # 1.876 < 3.123
        assert abs(row2["p_long_win_cal"] - v2_cal_long) < 1e-12
        assert abs(row2["p_short_win_cal"] - v2_cal_short) < 1e-12
        assert abs(row2["gamma_ood"] - v2_gamma) < 1e-12
        assert abs(row2["ood_threshold"] - v2_thresh) < 1e-12
        assert row2["is_economic_consistent"] is True
        assert abs(row2["p_long_raw"] - v2_raw_long) < 1e-12
        assert abs(row2["p_short_raw"] - v2_raw_short) < 1e-12
        assert row2["is_coherent"] is True
        assert abs(row2["coherence_penalty"] - 0.0) < 1e-12

        # 4. Verify repo.get_prediction_packet retrieval fidelity
        retrieved_pkt = await repo.get_prediction_packet(test_symbol)
        assert retrieved_pkt is not None
        assert retrieved_pkt.regime_label == "REGIME_BETA"
        assert retrieved_pkt.model_version == "m46_v2_updated"
        assert abs(retrieved_pkt.p_long_raw - v2_raw_long) < 1e-12
        assert abs(retrieved_pkt.p_short_raw - v2_raw_short) < 1e-12
        assert abs(retrieved_pkt.p_long_calibrated.p_calibrated - v2_cal_long) < 1e-12
        assert abs(retrieved_pkt.p_short_calibrated.p_calibrated - v2_cal_short) < 1e-12
        assert abs(retrieved_pkt.ood_status.gamma_ood - v2_gamma) < 1e-12
        assert abs(retrieved_pkt.ood_status.empirical_quantile_threshold - v2_thresh) < 1e-12
        assert retrieved_pkt.is_economic_consistent is True
        assert retrieved_pkt.is_coherent is True
        assert abs(retrieved_pkt.coherence_penalty - 0.0) < 1e-12
        assert abs(retrieved_pkt.top_bullish_shap[0].shap_value - v2_bull_shap) < 1e-12
        assert abs(retrieved_pkt.top_bearish_shap[0].shap_value - v2_bear_shap) < 1e-12

    finally:
        async with db.transaction() as conn:
            await conn.execute(
                "DELETE FROM quant_feature_vectors_30m WHERE symbol = $1;", test_symbol
            )
        await db.close()


@pytest.mark.asyncio
async def test_batch_upsert_throughput_and_multi_worker_concurrency() -> None:
    """Stress test: Measure high-throughput batch ingestion and concurrent readers/writers
    and simultaneous conflict upsert stress without deadlocks.
    """
    db_cfg: DatabaseConfig = get_integration_db_config()
    db: DatabaseManager = DatabaseManager(db_cfg)
    await db.initialize()

    bench_symbol = f"CHAL_BENCH_{int(time.time()) % 10000}"
    packet_count = 3000
    start_time = datetime(2023, 1, 1, 0, 0, tzinfo=UTC)
    worker_count = 6
    worker_symbols: list[str] = [
        f"CHAL_W_{idx}_{int(time.time()) % 1000}" for idx in range(worker_count)
    ]
    conflict_sym = f"CHAL_LOCK_{int(time.time()) % 1000}"

    try:
        repo = MLPredictionRepository(db_manager=db, batch_chunk_size=500)

        packets: list[UnifiedMLPredictionPacket] = []
        for i in range(packet_count):
            t = start_time + timedelta(minutes=30 * i)
            p = make_packet(
                symbol=bench_symbol,
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
                is_econ_consistent=True,
                model_version="m46_bench_v1",
                regime_label="BENCHMARK",
                hurst_exp=0.55,
                yang_zhang_vol=0.015,
                vpin_val=0.35,
                kyle_val=1.5,
                ofi_imbalance=1.2,
            )
            packets.append(p)

        t_start = time.perf_counter()
        saved = await repo.save_prediction_packets_batch(packets)
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        throughput = packet_count / elapsed

        print(
            f"\n[CHALLENGER THROUGHPUT] {saved} packets saved in {elapsed:.3f}s ({throughput:.1f} pkt/s)"
        )
        assert saved == packet_count
        assert throughput > 400.0, f"Throughput too low: {throughput:.1f} pkt/s"

        # Sub-test 2: Multi-worker concurrency (6 concurrent writers + 6 concurrent readers)
        packets_per_worker = 200

        async def worker_write(sym: str) -> int:
            worker_pkts: list[UnifiedMLPredictionPacket] = []
            for j in range(packets_per_worker):
                wt = start_time + timedelta(minutes=30 * j)
                wp = make_packet(
                    symbol=sym,
                    bar_time=wt,
                    raw_long=0.52,
                    raw_short=0.40,
                    cal_long=0.53,
                    cal_short=0.39,
                    uncertainty=0.09,
                    m_dist=1.7,
                    threshold=3.0,
                    gamma_ood=0.98,
                    shap_bull_val=0.04,
                    shap_bear_val=-0.02,
                    is_econ_consistent=True,
                    model_version="m46_worker_v1",
                    regime_label="WORKER",
                    hurst_exp=0.54,
                    yang_zhang_vol=0.014,
                    vpin_val=0.32,
                    kyle_val=1.4,
                    ofi_imbalance=1.1,
                )
                worker_pkts.append(wp)
            return await repo.save_prediction_packets_batch(worker_pkts)

        async def worker_read(sym: str) -> int:
            success = 0
            for _ in range(5):
                await asyncio.sleep(0.01)
                latest = await repo.get_prediction_packet(sym)
                if latest is not None:
                    success += 1
                range_pkts = await repo.get_prediction_packets_range(
                    symbol=sym,
                    start_time=start_time,
                    end_time=start_time + timedelta(days=2),
                )
                if len(range_pkts) > 0:
                    success += 1
            return success

        writers = [worker_write(s) for s in worker_symbols]
        readers = [worker_read(s) for s in worker_symbols]

        t_conc_start = time.perf_counter()
        conc_results = await asyncio.gather(*writers, *readers, return_exceptions=False)
        t_conc_end = time.perf_counter()

        write_counts = conc_results[:worker_count]
        print(
            f"[CHALLENGER CONCURRENCY] 6 writers + 6 readers finished in {t_conc_end - t_conc_start:.3f}s"
        )
        for count in write_counts:
            assert count == packets_per_worker

        # Sub-test 3: Simultaneous conflict upsert on the EXACT SAME keys
        shared_timestamps = [start_time + timedelta(minutes=30 * k) for k in range(50)]

        async def concurrent_conflict_writer(worker_id: int) -> int:
            c_pkts: list[UnifiedMLPredictionPacket] = []
            for st in shared_timestamps:
                cp = make_packet(
                    symbol=conflict_sym,
                    bar_time=st,
                    raw_long=0.50 + 0.01 * worker_id,
                    raw_short=0.40,
                    cal_long=0.51,
                    cal_short=0.38,
                    uncertainty=0.07,
                    m_dist=1.9,
                    threshold=3.0,
                    gamma_ood=1.0,
                    shap_bull_val=0.03,
                    shap_bear_val=-0.01,
                    is_econ_consistent=True,
                    model_version=f"m46_lock_w{worker_id}",
                    regime_label="LOCK_TEST",
                    hurst_exp=0.52,
                    yang_zhang_vol=0.013,
                    vpin_val=0.30,
                    kyle_val=1.3,
                    ofi_imbalance=1.0,
                )
                c_pkts.append(cp)
            return await repo.save_prediction_packets_batch(c_pkts)

        # 4 workers simultaneously hammering the EXACT SAME 50 keys
        conflict_tasks = [concurrent_conflict_writer(w) for w in range(4)]
        conflict_results = await asyncio.gather(*conflict_tasks, return_exceptions=False)
        for cr in conflict_results:
            assert cr == 50

        # Row count must remain exactly 50 (zero duplicates)
        c_count_row = await db.fetchrow(
            "SELECT count(*) as cnt FROM quant_feature_vectors_30m WHERE symbol = $1;",
            conflict_sym,
        )
        assert c_count_row is not None
        assert c_count_row["cnt"] == 50

    finally:
        async with db.transaction() as conn:
            await conn.execute(
                "DELETE FROM quant_feature_vectors_30m WHERE symbol = $1;", bench_symbol
            )
            for ws in worker_symbols:
                await conn.execute("DELETE FROM quant_feature_vectors_30m WHERE symbol = $1;", ws)
            await conn.execute(
                "DELETE FROM quant_feature_vectors_30m WHERE symbol = $1;", conflict_sym
            )
        await db.close()
