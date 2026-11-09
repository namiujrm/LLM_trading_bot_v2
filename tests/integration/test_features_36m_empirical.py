# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Empirical Challenger stress-testing for Milestone 1 36-Month 4-Pair Feature and Label Datasets.

Adversarially validates:
1. Numerical integrity: 0 NaNs, 0 Infs, 0 nulls across BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT.
2. Dimensions & Grid Synchronicity: Exactly 52,608 rows per symbol, 210,432 rows total.
3. De Prado Weight Conservation: abs(sum(de_prado_weight) - N) < 1e-4 for all symbols.
4. Cross-Asset Feature: btc_lead_catchup_gap == 0.0 for BTCUSDT, non-zero distribution for ETH, SOL, BNB.
5. Triple-Barrier Labeling Sanity: Long & Short balance, no degenerate all-one/all-zero, bars held <= 120.
6. Causal Integrity & Temporal Invariants: Zero look-ahead bias, exit_time > timestamp, horizon bounds.
7. Feature Domain Invariants: Bounds checks on Hurst, Shannon, Fracdiff, VPIN, Kyle's Lambda, OFI, CVD, OI.
8. Financial Math Consistency: Exact relation between realized_return, net_return, and roundtrip_cost.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

DATA_DIR: Path = Path("data/features_36m")
SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
EXPECTED_ROWS_PER_SYMBOL: int = 52608
EXPECTED_TOTAL_ROWS: int = 210432

CANONICAL_FEATURES: tuple[str, ...] = (
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


def load_parquet_or_fail(file_path: Path) -> pl.DataFrame:
    """Load parquet file ensuring path exists."""
    assert file_path.is_file(), f"File does not exist: {file_path}"
    df: pl.DataFrame = pl.read_parquet(file_path)
    return df


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
def test_empirical_dataset_dimensions_and_nulls(symbol: str) -> None:
    """Stress-test dimensions and assert 0 nulls, 0 NaNs, 0 Infs per symbol."""
    file_path: Path = DATA_DIR / f"{symbol}.parquet"
    df: pl.DataFrame = load_parquet_or_fail(file_path)

    # 1. Exact row count check
    assert len(df) == EXPECTED_ROWS_PER_SYMBOL, (
        f"Row count mismatch for {symbol}: expected {EXPECTED_ROWS_PER_SYMBOL}, got {len(df)}"
    )

    # 2. Total null check across all columns
    null_count: int = int(df.null_count().sum_horizontal().item(0))
    assert null_count == 0, f"Found {null_count} nulls in {symbol} dataset!"

    # 3. NaNs and Infs check across all float columns
    for col_name in df.columns:
        col_type = df.schema[col_name]
        if col_type.is_float():
            vals: np.ndarray = df[col_name].to_numpy()
            nan_count: int = int(np.isnan(vals).sum())
            inf_count: int = int(np.isinf(vals).sum())
            assert nan_count == 0, f"Column {col_name} in {symbol} has {nan_count} NaNs!"
            assert inf_count == 0, f"Column {col_name} in {symbol} has {inf_count} Infs!"

    # 4. Monotonic timestamps check (exactly 30-minute steps)
    ts = df["timestamp"].to_numpy().astype("datetime64[ms]").astype(np.int64)
    diffs = np.diff(ts)
    assert np.all(diffs == 1_800_000), f"{symbol} timestamp steps are not all 30 minutes!"

    # 5. OHLC validity
    o_arr, h_arr, l_arr, c_arr = (
        df["open"].to_numpy(),
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        df["close"].to_numpy(),
    )
    assert (o_arr > 0).all() and (h_arr > 0).all() and (l_arr > 0).all() and (c_arr > 0).all(), (
        f"{symbol} non-positive prices"
    )
    assert (h_arr >= l_arr).all(), f"{symbol} high < low"
    assert (h_arr >= o_arr).all(), f"{symbol} high < open"
    assert (h_arr >= c_arr).all(), f"{symbol} high < close"
    assert (l_arr <= o_arr).all(), f"{symbol} low > open"
    assert (l_arr <= c_arr).all(), f"{symbol} low > close"


def test_empirical_combined_dataset_synchronicity() -> None:
    """Stress-test the combined 4-pair dataset for exact alignment across all 4 symbols."""
    file_path: Path = DATA_DIR / "combined_4pairs_36m.parquet"
    df: pl.DataFrame = load_parquet_or_fail(file_path)

    assert len(df) == EXPECTED_TOTAL_ROWS, (
        f"Combined dataset rows mismatch: expected {EXPECTED_TOTAL_ROWS}, got {len(df)}"
    )

    # Verify all 4 symbols present with equal counts
    sym_counts = df.group_by("symbol").len()
    assert len(sym_counts) == 4, f"Expected 4 symbols, found {len(sym_counts)}"
    for row in sym_counts.iter_rows():
        sym_name: str = str(row[0])
        count: int = int(row[1])
        assert count == EXPECTED_ROWS_PER_SYMBOL, (
            f"Symbol {sym_name} has {count} rows, expected {EXPECTED_ROWS_PER_SYMBOL}"
        )

    # Verify temporal alignment: exactly 52,608 unique timestamps
    unique_timestamps: int = df["timestamp"].n_unique()
    assert unique_timestamps == EXPECTED_ROWS_PER_SYMBOL, (
        f"Timestamp mismatch: expected {EXPECTED_ROWS_PER_SYMBOL} unique timestamps, got {unique_timestamps}"
    )


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
def test_empirical_de_prado_weight_conservation(symbol: str) -> None:
    """Stress-test López de Prado sample uniqueness weight conservation: sum(w_i) == N."""
    file_path: Path = DATA_DIR / f"{symbol}.parquet"
    df: pl.DataFrame = load_parquet_or_fail(file_path)

    weights: np.ndarray = df["de_prado_weight"].to_numpy()
    weight_sum: float = float(weights.sum())
    n_bars: float = float(EXPECTED_ROWS_PER_SYMBOL)
    discrepancy: float = abs(weight_sum - n_bars)

    assert discrepancy < 1e-4, (
        f"De Prado weight conservation violated for {symbol}: sum={weight_sum:.6f}, N={n_bars}, delta={discrepancy:.6e}"
    )

    # Assert all weights are strictly non-negative
    assert (weights >= 0.0).all(), f"Found negative De Prado weights in {symbol}!"

    # Assert weights are non-trivial (not all identical or constant 1.0)
    assert float(weights.std()) > 0.0, f"De Prado weights are degenerate constant for {symbol}!"


def test_empirical_cross_asset_btc_lead_catchup_gap() -> None:
    """Stress-test btc_lead_catchup_gap: identically 0.0 for BTCUSDT, non-zero for altcoins."""
    # 1. BTCUSDT must be identically 0.0
    btc_df: pl.DataFrame = load_parquet_or_fail(DATA_DIR / "BTCUSDT.parquet")
    btc_gap: np.ndarray = btc_df["btc_lead_catchup_gap"].to_numpy()

    assert np.all(btc_gap == 0.0), (
        f"BTCUSDT btc_lead_catchup_gap is not identically 0.0! Max abs: {np.max(np.abs(btc_gap))}"
    )

    # 2. Altcoins must have non-zero variance and dynamic distribution
    for sym in ("ETHUSDT", "SOLUSDT", "BNBUSDT"):
        alt_df: pl.DataFrame = load_parquet_or_fail(DATA_DIR / f"{sym}.parquet")
        alt_gap: np.ndarray = alt_df["btc_lead_catchup_gap"].to_numpy()

        gap_std: float = float(alt_gap.std())
        gap_min: float = float(alt_gap.min())
        gap_max: float = float(alt_gap.max())

        assert gap_std > 0.005, f"{sym} btc_lead_catchup_gap has degenerate variance: {gap_std:.6f}"
        assert gap_min < -0.01, f"{sym} btc_lead_catchup_gap min is too tight: {gap_min}"
        assert gap_max > 0.01, f"{sym} btc_lead_catchup_gap max is too tight: {gap_max}"


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
def test_empirical_triple_barrier_label_sanity(symbol: str) -> None:
    """Stress-test triple-barrier labels: both Longs and Shorts, non-degenerate, bars held <= 120."""
    file_path: Path = DATA_DIR / f"{symbol}.parquet"
    df: pl.DataFrame = load_parquet_or_fail(file_path)

    y_long: np.ndarray = df["y_long"].to_numpy()
    y_short: np.ndarray = df["y_short"].to_numpy()
    bars_held: np.ndarray = df["bars_held_1m"].to_numpy()
    barrier_hits: list[str] = df["barrier_hit"].to_list()

    # 1. Bars held strictly between 1 and 120
    assert (bars_held >= 1).all(), f"Found bars_held_1m < 1 in {symbol}!"
    assert (bars_held <= 120).all(), f"Found bars_held_1m > 120 in {symbol}! Max: {bars_held.max()}"

    # 2. Non-degenerate labels: Longs and Shorts present
    n_long: int = int(y_long.sum())
    n_short: int = int(y_short.sum())
    n_total: int = len(df)

    long_pct: float = n_long / float(n_total)
    short_pct: float = n_short / float(n_total)

    # Long and short hit rates should be balanced between 20% and 70%
    assert 0.20 < long_pct < 0.70, (
        f"{symbol} Long rate out of expected bounds: {long_pct:.3%} ({n_long}/{n_total})"
    )
    assert 0.20 < short_pct < 0.70, (
        f"{symbol} Short rate out of expected bounds: {short_pct:.3%} ({n_short}/{n_total})"
    )

    # 3. Check barrier hit distribution
    hits_upper: int = sum(1 for h in barrier_hits if h == "upper_profit")
    hits_lower: int = sum(1 for h in barrier_hits if h == "lower_loss")
    hits_timeout: int = sum(1 for h in barrier_hits if h == "timeout")

    assert hits_upper > 5000, f"{symbol} too few upper hits: {hits_upper}"
    assert hits_lower > 5000, f"{symbol} too few lower hits: {hits_lower}"
    assert hits_timeout > 5000, f"{symbol} too few timeouts: {hits_timeout}"
    assert hits_upper + hits_lower + hits_timeout == n_total, (
        f"{symbol} barrier hit sum mismatch: {hits_upper + hits_lower + hits_timeout} vs {n_total}"
    )


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
def test_empirical_causal_temporality_and_leakage(symbol: str) -> None:
    """Stress-test zero look-ahead bias and causal temporal ordering."""
    file_path: Path = DATA_DIR / f"{symbol}.parquet"
    df: pl.DataFrame = load_parquet_or_fail(file_path)

    timestamps: np.ndarray = df["timestamp"].to_numpy()
    exit_times: np.ndarray = df["exit_time"].to_numpy()
    bars_held_1m: np.ndarray = df["bars_held_1m"].to_numpy()

    # 1. Exit time must strictly occur after decision bar open timestamp
    assert (exit_times > timestamps).all(), f"Found exit_time <= timestamp in {symbol}!"

    # 2. Maximum forward duration is 120 minutes + 29 minutes offset from bar open
    diff_ms: np.ndarray = (exit_times - timestamps).astype("timedelta64[ms]").astype(np.int64)
    min_diff_ms: int = int(diff_ms.min())
    max_diff_ms: int = int(diff_ms.max())

    # Forward 1m candles start at T + 30m. First candle open is T + 30m (diff = 1,800,000 ms).
    # Last candle (candle 119) open is T + 30m + 119m = T + 149m (diff = 8,940,000 ms).
    assert min_diff_ms >= 1_800_000, (
        f"{symbol} min forward exit difference too small: {min_diff_ms} ms (expected >= 1,800,000 ms)"
    )
    assert max_diff_ms <= 8_940_000, (
        f"{symbol} max forward exit difference too large: {max_diff_ms} ms (expected <= 8,940,000 ms)"
    )

    # 3. Check bars held correlation with diff_ms:
    # diff_ms equals (bars_held_1m - 1 + 30) * 60,000 ms
    expected_diff_ms: np.ndarray = (bars_held_1m - 1 + 30) * 60_000
    mismatch_count: int = int((diff_ms != expected_diff_ms).sum())
    assert mismatch_count == 0, (
        f"Found {mismatch_count} discrepancies between exit_time and bars_held_1m for {symbol}!"
    )


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
def test_empirical_canonical_feature_bounds(symbol: str) -> None:
    """Stress-test numerical range boundaries on all 11 canonical features."""
    file_path: Path = DATA_DIR / f"{symbol}.parquet"
    df: pl.DataFrame = load_parquet_or_fail(file_path)

    # 1. Hurst Exponent in [0.05, 0.95]
    h: np.ndarray = df["hurst_exponent_100"].to_numpy()
    assert (h >= 0.05).all() and (h <= 0.95).all(), (
        f"{symbol} Hurst out of bounds: [{h.min()}, {h.max()}]"
    )

    # 2. Shannon Entropy in [0.0, 1.0]
    s: np.ndarray = df["shannon_entropy_50"].to_numpy()
    assert (s >= 0.0).all() and (s <= 1.0).all(), (
        f"{symbol} Shannon out of bounds: [{s.min()}, {s.max()}]"
    )

    # 3. Fracdiff Momentum Z-score in [-3.0, 3.0]
    fd: np.ndarray = df["fracdiff_momentum_z"].to_numpy()
    assert (fd >= -3.0).all() and (fd <= 3.0).all(), (
        f"{symbol} FracDiff Z out of bounds: [{fd.min()}, {fd.max()}]"
    )

    # 4. VPIN toxicity in [0.0, 1.0]
    vpin: np.ndarray = df["vpin_toxicity_pct"].to_numpy()
    assert (vpin >= 0.0).all() and (vpin <= 1.0).all(), (
        f"{symbol} VPIN out of bounds: [{vpin.min()}, {vpin.max()}]"
    )

    # 5. Kyle's Lambda log in [0.0, 10.0]
    kl: np.ndarray = df["kyle_lambda_log"].to_numpy()
    assert (kl >= 0.0).all() and (kl <= 10.0).all(), (
        f"{symbol} Kyle Lambda out of bounds: [{kl.min()}, {kl.max()}]"
    )

    # 6. CVD Z in [-3.0, 3.0]
    cvd: np.ndarray = df["taker_cvd_z_24"].to_numpy()
    assert (cvd >= -3.0).all() and (cvd <= 3.0).all(), (
        f"{symbol} CVD Z out of bounds: [{cvd.min()}, {cvd.max()}]"
    )

    # 7. OFI Z in [-3.0, 3.0]
    ofi: np.ndarray = df["ofi_imbalance_z"].to_numpy()
    assert (ofi >= -3.0).all() and (ofi <= 3.0).all(), (
        f"{symbol} OFI Z out of bounds: [{ofi.min()}, {ofi.max()}]"
    )

    # 8. Funding Z in [0.0, 3.0]
    fp: np.ndarray = df["funding_crowded_pos_z"].to_numpy()
    fn: np.ndarray = df["funding_crowded_neg_z"].to_numpy()
    assert (fp >= 0.0).all() and (fp <= 3.0).all(), f"{symbol} Funding Pos Z out of bounds"
    assert (fn >= 0.0).all() and (fn <= 3.0).all(), f"{symbol} Funding Neg Z out of bounds"

    # 9. OI score in [-1.0, 1.0]
    oi: np.ndarray = df["oi_directional_score"].to_numpy()
    assert (oi >= -1.0).all() and (oi <= 1.0).all(), f"{symbol} OI score out of bounds"


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
def test_empirical_financial_math_consistency(symbol: str) -> None:
    """Stress-test financial math relations and barrier geometry."""
    file_path: Path = DATA_DIR / f"{symbol}.parquet"
    df: pl.DataFrame = load_parquet_or_fail(file_path)

    realized: np.ndarray = df["realized_return"].to_numpy()
    net: np.ndarray = df["net_return"].to_numpy()
    cost: np.ndarray = df["roundtrip_cost"].to_numpy()
    entry: np.ndarray = df["entry_price"].to_numpy()
    exit_p: np.ndarray = df["exit_price"].to_numpy()
    upper: np.ndarray = df["upper_barrier_price"].to_numpy()
    lower: np.ndarray = df["lower_barrier_price"].to_numpy()
    hits: list[str] = df["barrier_hit"].to_list()

    # 1. Roundtrip cost must be strictly positive (10 bps = 0.0010)
    assert (cost > 0.0).all(), f"Found non-positive roundtrip cost in {symbol}!"
    assert np.allclose(cost, 0.0010, atol=1e-8), f"{symbol} unexpected roundtrip cost values"

    # 2. Barrier geometry: upper > entry > lower
    assert (upper > entry).all(), f"Upper barrier not above entry price in {symbol}!"
    assert (lower < entry).all(), f"Lower barrier not below entry price in {symbol}!"

    # 3. Realized return definition check: realized == (exit_price - entry_price) / entry_price
    expected_realized: np.ndarray = (exit_p - entry) / entry
    realized_diff: np.ndarray = np.abs(realized - expected_realized)
    max_realized_diff: float = float(realized_diff.max())
    assert max_realized_diff < 1e-9, (
        f"{symbol} Realized return discrepancy: {max_realized_diff:.3e}"
    )

    # 4. Triple-Barrier Directional Net Return Invariants:
    # - Upper profit hit: net_return == realized_return - roundtrip_cost
    # - Lower loss hit: net_return == -realized_return - roundtrip_cost
    # - Timeout: net_return conforms to directional drift excess over transaction cost
    for i, h in enumerate(hits):
        r: float = float(realized[i])
        c: float = float(cost[i])
        n: float = float(net[i])

        if h == "upper_profit":
            expected_net = r - c
            assert abs(n - expected_net) < 1e-6, (
                f"Row {i} Upper profit net mismatch: got {n}, expected {expected_net}"
            )
        elif h == "lower_loss":
            expected_net = -r - c
            assert abs(n - expected_net) < 1e-6, (
                f"Row {i} Lower loss net mismatch: got {n}, expected {expected_net}"
            )
        elif h == "timeout":
            if abs(r) <= c + 1e-8:
                assert n <= 0.0, f"Row {i} Timeout dead zone should be <= 0.0, got {n}"
            elif r > c:
                expected_net = r - c
                assert abs(n - expected_net) < 1e-6, (
                    f"Row {i} Timeout long net mismatch: got {n}, expected {expected_net}"
                )
            else:
                expected_net = -r - c
                assert abs(n - expected_net) < 1e-6, (
                    f"Row {i} Timeout short net mismatch: got {n}, expected {expected_net}"
                )
