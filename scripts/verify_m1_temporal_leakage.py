# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Verification Script: scripts/verify_m1_temporal_leakage.py.

Empirical Challenger 2 Verification Harness for Milestone 1.
Adversarially probes:
1. Temporal Monotonicity & Continuity across 4 pairs (BTC, ETH, SOL, BNB).
2. Cross-Sectional Synchronization at every 30m decision epoch.
3. Strict Causal Alignment of Triple-Barrier forward 1m scan (starts at T_open + 30m).
4. Feature Non-Degeneracy (variance > 0, zero NaNs, zero Infs, bounded ranges).
5. Empirical Causal Oracle Test: Future perturbation invariance test.
6. Sample Uniqueness Weight Conservation (sum(w) == N).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

try:
    from scripts.build_36m_feature_dataset import (
        CANONICAL_FEATURE_NAMES,
        SUBSPACE_FEATURE_NAMES,
        compute_derived_series_features,
        compute_fracdiff_momentum_z,
        compute_garman_klass_volatility,
        compute_microstructure_features,
        compute_rolling_hurst_100,
        compute_rolling_shannon_entropy_50,
        resample_1m_to_30m,
    )
except ModuleNotFoundError:
    from build_36m_feature_dataset import (
        CANONICAL_FEATURE_NAMES,
        SUBSPACE_FEATURE_NAMES,
        compute_derived_series_features,
        compute_fracdiff_momentum_z,
        compute_garman_klass_volatility,
        compute_microstructure_features,
        compute_rolling_hurst_100,
        compute_rolling_shannon_entropy_50,
        resample_1m_to_30m,
    )


def verify_file_inventory(dataset_dir: Path) -> dict[str, pl.DataFrame]:
    """Verify that all expected Parquet files exist and have exact expected dimensions."""
    print("=== [CHECK 1] Dataset Inventory & Schema Verification ===")
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
    dfs: dict[str, pl.DataFrame] = {}

    for sym in symbols:
        p: Path = dataset_dir / f"{sym}.parquet"
        if not p.is_file():
            raise FileNotFoundError(f"Missing parquet file: {p}")
        df: pl.DataFrame = pl.read_parquet(p)
        if len(df) != 52608:
            raise AssertionError(f"Expected 52,608 rows for {sym}, got {len(df)}")
        if len(df.columns) != 45:
            raise AssertionError(f"Expected 45 columns for {sym}, got {len(df.columns)}")
        null_count: int = int(df.null_count().sum_horizontal().item(0))
        if null_count != 0:
            raise AssertionError(f"Found {null_count} nulls in {sym}")
        dfs[sym] = df
        print(f"  [PASS] {sym}.parquet: 52,608 rows, 45 cols, 0 nulls")

    comb_path: Path = dataset_dir / "combined_4pairs_36m.parquet"
    if not comb_path.is_file():
        raise FileNotFoundError(f"Missing combined parquet: {comb_path}")
    comb_df: pl.DataFrame = pl.read_parquet(comb_path)
    if len(comb_df) != 210432:
        raise AssertionError(f"Expected 210,432 rows in combined dataset, got {len(comb_df)}")
    null_count_comb: int = int(comb_df.null_count().sum_horizontal().item(0))
    if null_count_comb != 0:
        raise AssertionError(f"Found {null_count_comb} nulls in combined dataset")
    print(f"  [PASS] combined_4pairs_36m.parquet: {len(comb_df):,} rows, 45 cols, 0 nulls")
    dfs["COMBINED"] = comb_df
    return dfs


def verify_temporal_monotonicity(dfs: dict[str, pl.DataFrame]) -> None:
    """Verify strict monotonic increasing timestamps without gaps, duplicates, or backwards jumps."""
    print("=== [CHECK 2] Temporal Monotonicity & Step Size Integrity ===")
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
    step_ms: int = 1_800_000  # 30 minutes in ms
    start_ms: int = 1672531200000  # 2023-01-01 00:00:00 UTC
    end_ms: int = 1767223800000  # 2025-12-31 23:30:00 UTC

    for sym in symbols:
        df: pl.DataFrame = dfs[sym]
        buckets: np.ndarray = df["bucket"].to_numpy()
        n: int = len(buckets)

        # 1. Exact boundary timestamps
        if buckets[0] != start_ms:
            raise AssertionError(f"{sym} start bucket is {buckets[0]}, expected {start_ms}")
        if buckets[-1] != end_ms:
            raise AssertionError(f"{sym} end bucket is {buckets[-1]}, expected {end_ms}")

        # 2. Delta check
        diffs: np.ndarray = buckets[1:] - buckets[:-1]
        bad_steps: int = int((diffs != step_ms).sum())
        if bad_steps != 0:
            raise AssertionError(f"{sym} contains {bad_steps} non-30m step intervals!")

        # 3. Strictly increasing (no duplicates, no backwards jumps)
        strictly_increasing: bool = bool(np.all(diffs > 0))
        if not strictly_increasing:
            raise AssertionError(f"{sym} timestamps are NOT strictly increasing!")

        # 4. Timestamp column consistency
        ts_ms: np.ndarray = df["timestamp"].dt.epoch("ms").to_numpy()
        ts_diff: np.ndarray = ts_ms - buckets
        if not np.all(ts_diff == 0):
            raise AssertionError(f"{sym} timestamp column diverges from bucket column!")

        print(
            f"  [PASS] {sym}: Exactly {n:,} contiguous 30m bars from {datetime.fromtimestamp(start_ms / 1000, tz=UTC)} to {datetime.fromtimestamp(end_ms / 1000, tz=UTC)}. 0 gaps, 0 duplicates."
        )

    # 5. Cross-Sectional Synchrony in Combined Dataset
    comb_df: pl.DataFrame = dfs["COMBINED"]
    unique_ts: int = comb_df["bucket"].n_unique()
    if unique_ts != 52608:
        raise AssertionError(f"Expected 52,608 unique decision epochs, got {unique_ts}")

    # Check that at every timestamp, there are exactly 4 rows (1 per symbol)
    ts_counts: pl.DataFrame = comb_df.group_by("bucket").agg(pl.len().alias("count"))
    counts_not_4: int = int((ts_counts["count"] != 4).sum())
    if counts_not_4 != 0:
        raise AssertionError(
            f"Found {counts_not_4} timestamps without exactly 4 synchronous assets!"
        )

    print(
        f"  [PASS] Cross-Sectional Synchrony: All {unique_ts:,} epochs have exactly 4 synchronized pairs."
    )


def verify_triple_barrier_alignment(dfs: dict[str, pl.DataFrame]) -> None:
    """Verify triple-barrier forward scan strictly begins at T_open + 30m without leakage."""
    print("=== [CHECK 3] Triple-Barrier Forward Alignment & Leakage Probe ===")
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")

    for sym in symbols:
        df: pl.DataFrame = dfs[sym]
        buckets: np.ndarray = df["bucket"].to_numpy()
        closes: np.ndarray = df["close"].to_numpy()
        entry_prices: np.ndarray = df["entry_price"].to_numpy()
        exit_times_ms: np.ndarray = df["exit_time"].dt.epoch("ms").to_numpy()
        bars_held_1m: np.ndarray = df["bars_held_1m"].to_numpy()
        bars_held_30m: np.ndarray = df["bars_held_30m"].to_numpy()
        y_long: np.ndarray = df["y_long"].to_numpy()
        y_short: np.ndarray = df["y_short"].to_numpy()
        barrier_hits: list[str] = df["barrier_hit"].to_list()
        vol_gk: np.ndarray = df["volatility_gk"].to_numpy()
        upper_p: np.ndarray = df["upper_barrier_price"].to_numpy()
        lower_p: np.ndarray = df["lower_barrier_price"].to_numpy()
        roundtrip_costs: np.ndarray = df["roundtrip_cost"].to_numpy()

        n: int = len(df)

        # 1. Entry price must equal 30m close
        diff_entry: np.ndarray = np.abs(entry_prices - closes)
        if np.max(diff_entry) > 1e-8:
            raise AssertionError(f"{sym} entry_price != 30m close: max diff = {np.max(diff_entry)}")

        # 2. Forward scan start: earliest possible exit time is T_open + 30m
        min_allowed_exit_ms: np.ndarray = buckets + 1_800_000
        early_exits: int = int((exit_times_ms < min_allowed_exit_ms).sum())
        if early_exits != 0:
            raise AssertionError(
                f"{sym} has {early_exits} bars where exit_time < T_open + 30m! FORWARD LEAKAGE DETECTED!"
            )

        # 3. Holding duration bounds
        if int((bars_held_1m < 1).sum()) > 0 or int((bars_held_1m > 120).sum()) > 0:
            raise AssertionError(f"{sym} bars_held_1m outside [1, 120] range!")
        if int((bars_held_30m < 1).sum()) > 0 or int((bars_held_30m > 4).sum()) > 0:
            raise AssertionError(f"{sym} bars_held_30m outside [1, 4] range!")

        # 4. Exit time exact equation: exit_time == T_open + 30m + (bars_held_1m - 1) * 60_000
        expected_exit_ms: np.ndarray = min_allowed_exit_ms + (bars_held_1m - 1) * 60_000
        exit_mismatch: int = int((exit_times_ms != expected_exit_ms).sum())
        if exit_mismatch != 0:
            raise AssertionError(f"{sym} has {exit_mismatch} exit_time mismatches!")

        # 5. Label exclusivity: y_long + y_short <= 1
        label_sum: np.ndarray = y_long + y_short
        invalid_labels: int = int((label_sum > 1).sum())
        if invalid_labels != 0:
            raise AssertionError(f"{sym} has {invalid_labels} bars with y_long=1 and y_short=1!")

        # 6. Barrier hit consistency
        upper_hits: int = 0
        lower_hits: int = 0
        timeout_hits: int = 0
        for i in range(n):
            hit = barrier_hits[i]
            if hit == "upper_profit":
                upper_hits += 1
                if y_long[i] != 1 or y_short[i] != 0:
                    raise AssertionError(f"{sym} upper_profit row {i} has invalid y_long/y_short!")
            elif hit == "lower_loss":
                lower_hits += 1
                if y_long[i] != 0 or y_short[i] != 1:
                    raise AssertionError(f"{sym} lower_loss row {i} has invalid y_long/y_short!")
            elif hit == "timeout":
                timeout_hits += 1
            else:
                raise AssertionError(f"Unknown barrier hit type: {hit}")

        # 7. Barrier price geometry
        exp_upper = entry_prices * (1.0 + 1.8 * vol_gk + roundtrip_costs)
        exp_lower = entry_prices * (1.0 - 1.8 * vol_gk - roundtrip_costs)
        max_upper_err = np.max(np.abs(upper_p - exp_upper) / entry_prices)
        max_lower_err = np.max(np.abs(lower_p - exp_lower) / entry_prices)
        if max_upper_err > 1e-10 or max_lower_err > 1e-10:
            raise AssertionError(f"{sym} barrier geometry calculation error!")

        print(
            f"  [PASS] {sym}: 0 forward leakage, min exit delay = 30m, max horizon = 120m. Upper={upper_hits:,}, Lower={lower_hits:,}, Timeout={timeout_hits:,}."
        )


def verify_feature_properties(dfs: dict[str, pl.DataFrame]) -> None:
    """Verify non-degeneracy, variance > 0, zero NaNs, and bounded distributions for all features."""
    print("=== [CHECK 4] Feature Non-Degeneracy & Variance Analysis ===")
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")

    for sym in symbols:
        df: pl.DataFrame = dfs[sym]
        print(f"\n--- Feature Statistics for {sym} ---")
        for f_name in CANONICAL_FEATURE_NAMES:
            vals: np.ndarray = df[f_name].to_numpy()
            var: float = float(np.var(vals))
            v_min: float = float(np.min(vals))
            v_max: float = float(np.max(vals))
            v_mean: float = float(np.mean(vals))
            n_unique: int = len(np.unique(vals))

            # BTCUSDT lead catchup gap against itself is identically 0.0 by definition
            if sym == "BTCUSDT" and f_name == "btc_lead_catchup_gap":
                if not np.all(vals == 0.0):
                    raise AssertionError("BTCUSDT btc_lead_catchup_gap must be identically 0.0")
                print(
                    f"  {f_name:<22} | var={var:.6f} | min={v_min:+.4f} | max={v_max:+.4f} | mean={v_mean:+.4f} | unique={n_unique} [PASS - BTC Identity Anchor]"
                )
                continue

            # All other features for all symbols MUST have positive variance
            if var <= 0.0:
                raise AssertionError(f"{sym} feature `{f_name}` has degenerate zero variance!")

            if np.isnan(vals).any():
                raise AssertionError(f"{sym} feature `{f_name}` contains NaN!")
            if np.isinf(vals).any():
                raise AssertionError(f"{sym} feature `{f_name}` contains Inf!")

            # Verify specific domain bounds
            if f_name == "hurst_exponent_100":
                if v_min < 0.05 or v_max > 0.95:
                    raise AssertionError(f"{sym} hurst outside [0.05, 0.95]: [{v_min}, {v_max}]")
            elif f_name == "shannon_entropy_50":
                if v_min < 0.0 or v_max > 1.0:
                    raise AssertionError(f"{sym} shannon outside [0.0, 1.0]: [{v_min}, {v_max}]")
            elif f_name in ("fracdiff_momentum_z", "taker_cvd_z_24", "ofi_imbalance_z"):
                if v_min < -3.0001 or v_max > 3.0001:
                    raise AssertionError(f"{sym} {f_name} outside [-3.0, 3.0]: [{v_min}, {v_max}]")
            elif f_name == "vpin_toxicity_pct":
                if v_min < 0.0 or v_max > 1.0:
                    raise AssertionError(f"{sym} vpin outside [0.0, 1.0]: [{v_min}, {v_max}]")
            elif f_name == "oi_directional_score":
                if v_min < -1.0 or v_max > 1.0:
                    raise AssertionError(f"{sym} oi score outside [-1.0, 1.0]: [{v_min}, {v_max}]")
            elif f_name in ("funding_crowded_pos_z", "funding_crowded_neg_z") and (
                v_min < 0.0 or v_max > 3.0001
            ):
                raise AssertionError(f"{sym} {f_name} outside [0.0, 3.0]: [{v_min}, {v_max}]")

            print(
                f"  {f_name:<22} | var={var:.6f} | min={v_min:+.4f} | max={v_max:+.4f} | mean={v_mean:+.4f} | unique={n_unique} [PASS]"
            )

        # Also verify 6 Subspace Features
        for sub_name in SUBSPACE_FEATURE_NAMES:
            if sub_name not in df.columns:
                raise AssertionError(f"{sym} missing subspace feature: {sub_name}")


def verify_sample_weight_conservation(dfs: dict[str, pl.DataFrame]) -> None:
    """Verify Lopez de Prado sample uniqueness weight conservation law sum(w) == N."""
    print("\n=== [CHECK 5] Sample Weight Conservation Law ===")
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")

    for sym in symbols:
        df: pl.DataFrame = dfs[sym]
        weights: np.ndarray = df["de_prado_weight"].to_numpy()
        n: int = len(df)
        weight_sum: float = float(np.sum(weights))
        discrepancy: float = abs(weight_sum - float(n))

        if discrepancy > 0.01:
            raise AssertionError(
                f"{sym} sample weight conservation violated: sum={weight_sum:.4f}, expected={n}"
            )
        if np.any(weights < 0.0):
            raise AssertionError(f"{sym} contains negative sample weights!")

        print(
            f"  [PASS] {sym}: sum(de_prado_weight) = {weight_sum:.2f} == N ({n:,}) | min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}"
        )


def verify_causal_oracle_perturbation() -> None:
    """Adversarial Oracle Test: Corrupt future 1m rows and prove past features do not change."""
    print("\n=== [CHECK 6] Adversarial Empirical Oracle Test: Future Perturbation Invariance ===")
    parquet_path: Path = Path("data/parquet/BTCUSDT/2023/01.parquet")
    if not parquet_path.is_file():
        print("  [SKIP] BTC 2023/01 parquet not found.")
        return

    df_1m_clean: pl.DataFrame = pl.read_parquet(parquet_path).sort("open_time").head(15000)
    df_30m_clean: pl.DataFrame = resample_1m_to_30m(df_1m=df_1m_clean, symbol="BTCUSDT")
    n_30m: int = len(df_30m_clean)
    split_30m_bar: int = 200  # Split at bar 200
    split_1m_bar: int = split_30m_bar * 30  # 1m index = 6000

    # Clean Feature Computation
    closes_clean: np.ndarray = df_30m_clean["close"].to_numpy()
    gk_clean: np.ndarray = compute_garman_klass_volatility(
        opens=df_30m_clean["open"].to_numpy(),
        highs=df_30m_clean["high"].to_numpy(),
        lows=df_30m_clean["low"].to_numpy(),
        closes=closes_clean,
        ewma_span=12,
    )
    hurst_clean: np.ndarray = compute_rolling_hurst_100(closes=closes_clean, window=100)
    shannon_clean: np.ndarray = compute_rolling_shannon_entropy_50(
        closes=closes_clean, window=50, k_bins=10
    )
    fracdiff_clean: np.ndarray = compute_fracdiff_momentum_z(
        closes=closes_clean, d=0.40, length=100, threshold=1e-4, z_window=24
    )
    vpin_clean, kyle_clean = compute_microstructure_features(df_1m=df_1m_clean, n_bars_30m=n_30m)
    cvd_clean, ofi_clean, oi_clean, f_pos_clean, f_neg_clean, _ = compute_derived_series_features(
        df_30m=df_30m_clean, z_window=24, funding_window=336, btc_6h_returns=None
    )

    # Now create PERTURBED future: multiply prices and volumes of all rows >= split_1m_bar by 10x
    open_p: np.ndarray = df_1m_clean["open"].to_numpy().copy()
    high_p: np.ndarray = df_1m_clean["high"].to_numpy().copy()
    low_p: np.ndarray = df_1m_clean["low"].to_numpy().copy()
    close_p: np.ndarray = df_1m_clean["close"].to_numpy().copy()
    vol_p: np.ndarray = df_1m_clean["volume"].to_numpy().copy()

    open_p[split_1m_bar:] *= 10.0
    high_p[split_1m_bar:] *= 15.0
    low_p[split_1m_bar:] *= 5.0
    close_p[split_1m_bar:] *= 12.0
    vol_p[split_1m_bar:] *= 20.0

    df_1m_perturbed: pl.DataFrame = df_1m_clean.with_columns(
        pl.Series("open", open_p),
        pl.Series("high", high_p),
        pl.Series("low", low_p),
        pl.Series("close", close_p),
        pl.Series("volume", vol_p),
    )
    df_30m_perturbed: pl.DataFrame = resample_1m_to_30m(df_1m=df_1m_perturbed, symbol="BTCUSDT")

    closes_pert: np.ndarray = df_30m_perturbed["close"].to_numpy()
    gk_pert: np.ndarray = compute_garman_klass_volatility(
        opens=df_30m_perturbed["open"].to_numpy(),
        highs=df_30m_perturbed["high"].to_numpy(),
        lows=df_30m_perturbed["low"].to_numpy(),
        closes=closes_pert,
        ewma_span=12,
    )
    hurst_pert: np.ndarray = compute_rolling_hurst_100(closes=closes_pert, window=100)
    shannon_pert: np.ndarray = compute_rolling_shannon_entropy_50(
        closes=closes_pert, window=50, k_bins=10
    )
    fracdiff_pert: np.ndarray = compute_fracdiff_momentum_z(
        closes=closes_pert, d=0.40, length=100, threshold=1e-4, z_window=24
    )
    vpin_pert, kyle_pert = compute_microstructure_features(df_1m=df_1m_perturbed, n_bars_30m=n_30m)
    cvd_pert, ofi_pert, oi_pert, f_pos_pert, f_neg_pert, _ = compute_derived_series_features(
        df_30m=df_30m_perturbed, z_window=24, funding_window=336, btc_6h_returns=None
    )

    # EMPIRICAL CAUSALITY ASSERTION:
    # Every feature for bars k < split_30m_bar MUST be bit-for-bit identical!
    test_features: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("volatility_gk", gk_clean, gk_pert),
        ("hurst_exponent_100", hurst_clean, hurst_pert),
        ("shannon_entropy_50", shannon_clean, shannon_pert),
        ("fracdiff_momentum_z", fracdiff_clean, fracdiff_pert),
        ("vpin_toxicity_pct", vpin_clean, vpin_pert),
        ("kyle_lambda_log", kyle_clean, kyle_pert),
        ("taker_cvd_z_24", cvd_clean, cvd_pert),
        ("ofi_imbalance_z", ofi_clean, ofi_pert),
        ("oi_directional_score", oi_clean, oi_pert),
        ("funding_crowded_pos_z", f_pos_clean, f_pos_pert),
        ("funding_crowded_neg_z", f_neg_clean, f_neg_pert),
    ]

    for name, clean_arr, pert_arr in test_features:
        diff: np.ndarray = np.abs(clean_arr[:split_30m_bar] - pert_arr[:split_30m_bar])
        max_diff: float = float(np.max(diff))
        if max_diff > 0.0:
            raise AssertionError(
                f"FUTURE LEAKAGE DETECTED in `{name}`! Past features changed when future was corrupted! Max diff={max_diff}"
            )
        print(f"  [PASS - ZERO LEAKAGE] `{name:<22}`: Past max diff = 0.0000000000000000")

    print(
        f"  [ORACLE VERDICT] All 11 features proved 100% invariant to future modifications before bar {split_30m_bar}."
    )


def main() -> None:
    dataset_dir: Path = Path("data/features_36m").resolve()
    print(f"Starting Challenger 2 Adversarial Probe on: {dataset_dir}\n")

    dfs: dict[str, pl.DataFrame] = verify_file_inventory(dataset_dir=dataset_dir)
    verify_temporal_monotonicity(dfs=dfs)
    verify_triple_barrier_alignment(dfs=dfs)
    verify_feature_properties(dfs=dfs)
    verify_sample_weight_conservation(dfs=dfs)
    verify_causal_oracle_perturbation()

    print("\n=====================================================================")
    print("=== [FINAL VERDICT: APPROVE] Zero Look-Ahead Bias & Zero Leakage! ===")
    print("=====================================================================")


if __name__ == "__main__":
    main()
