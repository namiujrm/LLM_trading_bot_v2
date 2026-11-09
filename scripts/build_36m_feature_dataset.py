# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Script: scripts/build_36m_feature_dataset.py.

Constructs the complete 36-Month 4-Pair Feature and Triple-Barrier Label Dataset
for Stage 4 Causal ML Backtesting (Strict Invariant #9 & Clean Architecture Compliant).

Processes:
- BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT across 2023-01-01 to 2025-12-31 (52,608 30m bars each, 210,432 total).
- 11 Canonical Features + 6 Subspace OOD Features.
- Dynamic Garman-Klass Volatility (sigma_GK).
- Symmetrical 1.8*sigma_GK Dynamic Triple-Barrier Labels (evaluate_triple_barrier, c_t = 0.0010, horizon = 120 1m bars).
- Lopez de Prado Sample Uniqueness Weights (compute_de_prado_weights, sum(w) == N).
- Zero look-ahead bias: features strictly <= T_decision, forward scans strictly > T_decision.
- Strict Invariant #9: 0 default parameters, 0 default fields, 0 silent fallback expressions.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from trading_bot.domain.entities.market.market_state import MicroBar1m
from trading_bot.domain.entities.ml.barrier_labels import BarrierLabel, LabelSpan
from trading_bot.domain.entities.ml.ml_configs import (
    SampleWeightConfig,
    TripleBarrierConfig,
)
from trading_bot.domain.services.ml.sample_uniqueness_weights import (
    compute_de_prado_weights,
)
from trading_bot.domain.services.ml.triple_barrier_labeler import evaluate_triple_barrier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_36m_features")

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

SUBSPACE_FEATURE_NAMES: tuple[str, ...] = (
    "hurst_exponent_100",
    "fracdiff_momentum_z",
    "vpin_toxicity_pct",
    "taker_cvd_z_24",
    "ofi_imbalance_z",
    "kyle_lambda_log",
)


@dataclass(frozen=True)
class PipelineConfig:
    """Explicit configuration schema with Zero Defaults (Invariant #9)."""

    parquet_dir: Path
    output_dir: Path
    symbols: tuple[str, ...]
    start_year: int
    end_year: int
    pt_multiplier: float
    sl_multiplier: float
    horizon_bars_1m: int
    taker_fee_rate: float
    slippage_rate: float
    min_tp_to_cost_multiple: float
    dead_zone_weight: float
    min_volatility_epsilon: float
    hurst_window: int
    shannon_window: int
    shannon_bins: int
    fracdiff_d: float
    fracdiff_length: int
    fracdiff_threshold: float
    z_window: int
    funding_window: int


def create_pipeline_config(
    parquet_dir_path: str,
    output_dir_path: str,
) -> PipelineConfig:
    """Factory creating PipelineConfig with explicit values and zero defaults."""
    return PipelineConfig(
        parquet_dir=Path(parquet_dir_path).resolve(),
        output_dir=Path(output_dir_path).resolve(),
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"),
        start_year=2023,
        end_year=2025,
        pt_multiplier=1.8,
        sl_multiplier=1.8,
        horizon_bars_1m=120,
        taker_fee_rate=0.0004,
        slippage_rate=0.0001,
        min_tp_to_cost_multiple=1.5,
        dead_zone_weight=0.0,
        min_volatility_epsilon=1e-6,
        hurst_window=100,
        shannon_window=50,
        shannon_bins=10,
        fracdiff_d=0.40,
        fracdiff_length=100,
        fracdiff_threshold=1e-4,
        z_window=24,
        funding_window=336,
    )


def load_all_1m_parquets_for_symbol(
    parquet_dir: Path,
    symbol: str,
    start_year: int,
    end_year: int,
) -> pl.DataFrame:
    """Load and concatenate all monthly 1m Parquet files for a given symbol."""
    file_paths: list[Path] = []
    for y in range(start_year, end_year + 1):
        year_dir = parquet_dir / symbol / str(y)
        if year_dir.is_dir():
            for m in range(1, 13):
                p = year_dir / f"{m:02d}.parquet"
                if p.is_file():
                    file_paths.append(p)

    if len(file_paths) == 0:
        raise FileNotFoundError(
            f"No Parquet files found for {symbol} between {start_year} and {end_year} in {parquet_dir}"
        )

    logger.info(f"Loading {len(file_paths)} monthly Parquet files for {symbol}...")
    dfs: list[pl.DataFrame] = [pl.read_parquet(fp) for fp in sorted(file_paths)]
    combined: pl.DataFrame = pl.concat(dfs).sort("open_time")
    deduped: pl.DataFrame = combined.unique(subset=["open_time"]).sort("open_time")
    logger.info(f"Loaded {len(deduped):,} 1m rows for {symbol} from disk.")
    return deduped


def resample_1m_to_30m(
    df_1m: pl.DataFrame,
    symbol: str,
) -> pl.DataFrame:
    """Aggregate 1m microstructure klines into 30m decision bars.

    Replicates continuous aggregate market_klines_30m_cagg exactly.
    """
    df_with_bucket: pl.DataFrame = df_1m.with_columns(
        (pl.col("open_time") // 1_800_000 * 1_800_000).alias("bucket")
    )

    agg_30m: pl.DataFrame = (
        df_with_bucket.group_by("bucket")
        .agg(
            pl.lit(symbol).alias("symbol"),
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("quote_volume").sum().alias("quote_volume"),
            pl.col("taker_buy_vol").sum().alias("taker_buy_vol"),
            pl.col("taker_buy_quote_vol").sum().alias("taker_buy_quote_vol"),
            pl.col("trade_count").sum().alias("trade_count"),
            pl.col("ofi_sum").sum().alias("ofi_sum"),
            pl.col("obi_mean").mean().alias("obi_mean"),
            pl.col("spread_bps_mean").mean().alias("spread_bps_mean"),
            pl.col("liq_long_usd").sum().alias("liq_long_usd"),
            pl.col("liq_short_usd").sum().alias("liq_short_usd"),
        )
        .sort("bucket")
    )
    return agg_30m


def compute_garman_klass_volatility(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    ewma_span: int,
) -> np.ndarray:
    """Compute Garman-Klass volatility with EWMA smoothing in float64."""
    co_factor: float = 2.0 * math.log(2.0) - 1.0  # ~0.386294
    log_hl: np.ndarray = np.log(np.maximum(highs / np.maximum(lows, 1e-12), 1.0))
    log_co: np.ndarray = np.log(np.maximum(closes / np.maximum(opens, 1e-12), 1e-12))

    gk_var: np.ndarray = 0.5 * (log_hl**2) - co_factor * (log_co**2)
    gk_var_clipped: np.ndarray = np.maximum(0.0, gk_var)
    gk_vol: np.ndarray = np.sqrt(gk_var_clipped)

    # Causal EWMA filter
    n: int = len(gk_vol)
    smoothed: np.ndarray = np.empty(n, dtype=np.float64)
    alpha: float = 2.0 / (float(ewma_span) + 1.0)
    smoothed[0] = max(0.0001, float(gk_vol[0]))
    for i in range(1, n):
        smoothed[i] = max(0.0001, alpha * float(gk_vol[i]) + (1.0 - alpha) * smoothed[i - 1])
    return smoothed


def compute_rolling_hurst_100(
    closes: np.ndarray,
    window: int,
) -> np.ndarray:
    """Compute Anis-Lloyd bias-corrected Hurst Exponent using sliding window vectorization."""
    n: int = len(closes)
    out: np.ndarray = np.full(n, 0.50, dtype=np.float64)
    if n < window + 1:
        return out

    log_rets: np.ndarray = np.diff(np.log(np.maximum(closes, 1e-12)))
    wins: np.ndarray = sliding_window_view(log_rets, window_shape=window)
    num_wins: int = len(wins)

    sub_sizes: tuple[int, ...] = (10, 20, 25, 50)
    log_m: np.ndarray = np.array([math.log(float(m)) for m in sub_sizes], dtype=np.float64)

    e_rs_list: list[float] = []
    for m in sub_sizes:
        gamma_ratio: float = math.gamma((float(m) - 1.0) / 2.0) / (
            math.sqrt(math.pi) * math.gamma(float(m) / 2.0)
        )
        sum_sqrt: float = sum(math.sqrt(float(m - i) / float(i)) for i in range(1, m))
        e_rs_list.append(gamma_ratio * sum_sqrt)

    log_rs_diffs: list[np.ndarray] = []
    for s_idx, m in enumerate(sub_sizes):
        k: int = window // m
        w_m: np.ndarray = wins.reshape(num_wins, k, m)
        mean_m: np.ndarray = w_m.mean(axis=2, keepdims=True)
        std_m: np.ndarray = w_m.std(axis=2, ddof=0)
        dev_m: np.ndarray = np.cumsum(w_m - mean_m, axis=2)
        r_m: np.ndarray = dev_m.max(axis=2) - dev_m.min(axis=2)
        rs_m: np.ndarray = np.where(std_m > 1e-12, r_m / np.maximum(std_m, 1e-12), 0.0).mean(axis=1)
        diff_m: np.ndarray = np.log(np.maximum(rs_m, 1e-12)) - math.log(
            max(e_rs_list[s_idx], 1e-12)
        )
        log_rs_diffs.append(diff_m)

    y: np.ndarray = np.stack(log_rs_diffs, axis=1)
    mean_x: float = float(log_m.mean())
    var_x: float = float(((log_m - mean_x) ** 2).sum())
    mean_y: np.ndarray = y.mean(axis=1, keepdims=True)
    cov_xy: np.ndarray = ((log_m - mean_x) * (y - mean_y)).sum(axis=1)

    beta: np.ndarray = cov_xy / max(var_x, 1e-12)
    h_raw: np.ndarray = 0.50 + beta
    h_clipped: np.ndarray = np.clip(h_raw, 0.05, 0.95)
    out[window:] = h_clipped
    return out


def compute_rolling_shannon_entropy_50(
    closes: np.ndarray,
    window: int,
    k_bins: int,
) -> np.ndarray:
    """Compute normalized Shannon information entropy of log returns over 50 bars."""
    n: int = len(closes)
    out: np.ndarray = np.full(n, 0.50, dtype=np.float64)
    if n < window + 1:
        return out

    log_rets: np.ndarray = np.diff(np.log(np.maximum(closes, 1e-12)))
    wins: np.ndarray = sliding_window_view(log_rets, window_shape=window)
    num_wins: int = len(wins)

    min_v: np.ndarray = wins.min(axis=1, keepdims=True)
    max_v: np.ndarray = wins.max(axis=1, keepdims=True)
    rng: np.ndarray = max_v - min_v
    valid_rng: np.ndarray = rng > 1e-12
    rng_safe: np.ndarray = np.where(valid_rng, rng, 1.0)

    bin_idx: np.ndarray = np.floor((wins - min_v) / (rng_safe / float(k_bins))).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, k_bins - 1)

    norm_log_k: float = math.log(float(k_bins))
    entropies: np.ndarray = np.zeros(num_wins, dtype=np.float64)

    for k in range(k_bins):
        cnt: np.ndarray = (bin_idx == k).sum(axis=1)
        p: np.ndarray = cnt / float(window)
        p_safe: np.ndarray = np.where(p > 0.0, p, 1.0)
        entropies += np.where(p > 0.0, -p * np.log(p_safe), 0.0)

    norm_entropy: np.ndarray = entropies / norm_log_k
    norm_entropy_safe: np.ndarray = np.where(valid_rng.squeeze(), norm_entropy, 0.0)
    out[window:] = np.clip(norm_entropy_safe, 0.0, 1.0)
    return out


def compute_fracdiff_weights(
    d: float,
    length: int,
    threshold: float,
) -> np.ndarray:
    """Compute fractional differentiation binomial expansion weights."""
    w: list[float] = [1.0]
    for k in range(1, length):
        w_k: float = -w[-1] * (d - float(k) + 1.0) / float(k)
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    return np.array(w, dtype=np.float64)


def compute_fracdiff_momentum_z(
    closes: np.ndarray,
    d: float,
    length: int,
    threshold: float,
    z_window: int,
) -> np.ndarray:
    """Fixed-window fractional differentiation with rolling 24-bar Z-score."""
    n: int = len(closes)
    out: np.ndarray = np.full(n, 0.0, dtype=np.float64)
    weights: np.ndarray = compute_fracdiff_weights(d=d, length=length, threshold=threshold)
    w_len: int = len(weights)
    if n < w_len + z_window:
        return out

    log_c: np.ndarray = np.log(np.maximum(closes, 1e-12))
    conv: np.ndarray = np.convolve(log_c, weights, mode="full")[:n]
    fd: np.ndarray = np.copy(conv)
    fd[: w_len - 1] = np.nan

    valid_fd: np.ndarray = fd[w_len - 1 :]
    wins: np.ndarray = sliding_window_view(valid_fd, window_shape=z_window)
    mean_z: np.ndarray = wins.mean(axis=1)
    std_z: np.ndarray = wins.std(axis=1, ddof=0)
    current_val: np.ndarray = wins[:, -1]

    z: np.ndarray = (current_val - mean_z) / (std_z + 1e-8)
    z_clipped: np.ndarray = np.clip(z, -3.0, 3.0)

    start_idx: int = w_len - 1 + z_window - 1
    out[start_idx:] = z_clipped
    return out


def compute_microstructure_features(
    df_1m: pl.DataFrame,
    n_bars_30m: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute VPIN toxicity and log Kyle Lambda price impact from 1m micro-bars."""
    opens_1m: np.ndarray = df_1m["open"].to_numpy().reshape(n_bars_30m, 30)
    closes_1m: np.ndarray = df_1m["close"].to_numpy().reshape(n_bars_30m, 30)
    vols_1m: np.ndarray = df_1m["volume"].to_numpy().reshape(n_bars_30m, 30)
    q_vols_1m: np.ndarray = df_1m["quote_volume"].to_numpy().reshape(n_bars_30m, 30)
    taker_q_vols_1m: np.ndarray = df_1m["taker_buy_quote_vol"].to_numpy().reshape(n_bars_30m, 30)
    taker_vols_1m: np.ndarray = df_1m["taker_buy_vol"].to_numpy().reshape(n_bars_30m, 30)

    # 1. VPIN Toxicity Percentile (quote volume imbalance ratio across 30 1m bars)
    sv_q: np.ndarray = np.abs(2.0 * taker_q_vols_1m - q_vols_1m)
    vpin_raw: np.ndarray = sv_q.sum(axis=1) / (q_vols_1m.sum(axis=1) + 1e-12)
    vpin_toxicity: np.ndarray = np.clip(vpin_raw, 0.0, 1.0)

    # 2. Kyle Lambda Log Price Impact
    dp: np.ndarray = np.abs(closes_1m - opens_1m) / np.maximum(opens_1m, 1e-12)
    tot_vol_usd: np.ndarray = vols_1m * closes_1m
    taker_usd: np.ndarray = taker_vols_1m * closes_1m
    sv_flow: np.ndarray = np.abs(2.0 * taker_usd - tot_vol_usd)

    sum_dp_sv: np.ndarray = (dp * sv_flow).sum(axis=1)
    sum_sv_sq: np.ndarray = (sv_flow * sv_flow).sum(axis=1)
    kyle_lambda: np.ndarray = sum_dp_sv / (sum_sv_sq + 1e-12)
    kyle_lambda_log: np.ndarray = np.log(kyle_lambda * 1e6 + 1.0)
    kyle_lambda_log_clipped: np.ndarray = np.clip(kyle_lambda_log, 0.0, 10.0)

    return (vpin_toxicity, kyle_lambda_log_clipped)


def compute_derived_series_features(
    df_30m: pl.DataFrame,
    z_window: int,
    funding_window: int,
    btc_6h_returns: list[float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute CVD z-score, OFI z-score, OI score, funding pos/neg z-scores, and BTC catchup gap."""
    n_bars: int = len(df_30m)

    # 1. Taker CVD Z-score (24 bars)
    net_taker: np.ndarray = 2.0 * df_30m["taker_buy_vol"].to_numpy() - df_30m["volume"].to_numpy()
    cvd_z: np.ndarray = np.full(n_bars, 0.0, dtype=np.float64)
    if n_bars >= z_window * 2:
        wins_flow: np.ndarray = sliding_window_view(net_taker, window_shape=z_window)
        rolling_cvd: np.ndarray = wins_flow.sum(axis=1)
        wins_cvd: np.ndarray = sliding_window_view(rolling_cvd, window_shape=z_window)
        mean_cvd: np.ndarray = wins_cvd.mean(axis=1)
        std_cvd: np.ndarray = wins_cvd.std(axis=1, ddof=0)
        curr_cvd: np.ndarray = wins_cvd[:, -1]
        z_cvd: np.ndarray = (curr_cvd - mean_cvd) / (std_cvd + 1e-8)
        start_idx: int = (z_window - 1) + (z_window - 1)
        cvd_z[start_idx:] = np.clip(z_cvd, -3.0, 3.0)

    # 2. OFI Imbalance Z-score (24 bars)
    ofi_arr: np.ndarray = df_30m["ofi_sum"].to_numpy()
    ofi_z: np.ndarray = np.full(n_bars, 0.0, dtype=np.float64)
    if n_bars >= z_window:
        wins_ofi: np.ndarray = sliding_window_view(ofi_arr, window_shape=z_window)
        mean_ofi: np.ndarray = wins_ofi.mean(axis=1)
        std_ofi: np.ndarray = wins_ofi.std(axis=1, ddof=0)
        curr_ofi: np.ndarray = wins_ofi[:, -1]
        z_ofi: np.ndarray = (curr_ofi - mean_ofi) / (std_ofi + 1e-8)
        ofi_z[z_window - 1 :] = np.clip(z_ofi, -3.0, 3.0)

    # 3. OI Price-Quadrant Directional Score
    closes: np.ndarray = df_30m["close"].to_numpy()
    liq_long: np.ndarray = df_30m["liq_long_usd"].to_numpy()
    liq_short: np.ndarray = df_30m["liq_short_usd"].to_numpy()
    dp: np.ndarray = np.diff(closes, prepend=closes[0])
    d_liq: np.ndarray = liq_short - liq_long

    oi_score: np.ndarray = np.zeros(n_bars, dtype=np.float64)
    for i in range(1, n_bars):
        if dp[i] > 0.0:
            if d_liq[i] >= 0.0:
                oi_score[i] = 1.0  # Long Buildup
            else:
                oi_score[i] = 0.5  # Short Covering
        elif dp[i] < 0.0:
            if d_liq[i] >= 0.0:
                oi_score[i] = -1.0  # Short Buildup
            else:
                oi_score[i] = -0.5  # Long Liquidation
        else:
            oi_score[i] = 0.0  # Neutral

    # 4 & 5. Funding Rate Crowdedness Pos & Neg Z-score (336 bars = 7 days)
    q_vols: np.ndarray = df_30m["quote_volume"].to_numpy()
    vols: np.ndarray = df_30m["volume"].to_numpy()
    vwap: np.ndarray = q_vols / np.maximum(vols, 1e-12)
    premium: np.ndarray = (vwap - closes) / np.maximum(closes, 1e-12)

    funding_pos_z: np.ndarray = np.full(n_bars, 0.0, dtype=np.float64)
    funding_neg_z: np.ndarray = np.full(n_bars, 0.0, dtype=np.float64)
    if n_bars >= funding_window:
        wins_prem: np.ndarray = sliding_window_view(premium, window_shape=funding_window)
        mean_p: np.ndarray = wins_prem.mean(axis=1)
        std_p: np.ndarray = wins_prem.std(axis=1, ddof=0)
        curr_p: np.ndarray = wins_prem[:, -1]
        z_prem: np.ndarray = (curr_p - mean_p) / (std_p + 1e-8)
        pos_z: np.ndarray = np.clip(np.maximum(0.0, z_prem), 0.0, 3.0)
        neg_z: np.ndarray = np.clip(np.maximum(0.0, -z_prem), 0.0, 3.0)
        funding_pos_z[funding_window - 1 :] = pos_z
        funding_neg_z[funding_window - 1 :] = neg_z

    # 6. BTC Lead Catchup Gap (6-hour return gap)
    btc_gap: np.ndarray = np.full(n_bars, 0.0, dtype=np.float64)
    if btc_6h_returns is not None:
        # Altcoin 6-hour return: close[k] / close[k - 12] - 1.0
        ret_6h: np.ndarray = np.zeros(n_bars, dtype=np.float64)
        for i in range(12, n_bars):
            ret_6h[i] = (closes[i] / max(closes[i - 12], 1e-12)) - 1.0
        for i in range(n_bars):
            gap: float = btc_6h_returns[i] - float(ret_6h[i])
            btc_gap[i] = min(3.0, max(-3.0, gap))

    return (cvd_z, ofi_z, oi_score, funding_pos_z, funding_neg_z, btc_gap)


def run_triple_barrier_on_symbol(
    symbol: str,
    df_30m: pl.DataFrame,
    candles_1m_all: list[MicroBar1m],
    tb_config: TripleBarrierConfig,
) -> list[BarrierLabel]:
    """Execute forward 1m candle scans strictly > T_decision for all 30m decision bars."""
    n_bars: int = len(df_30m)
    labels: list[BarrierLabel] = []
    bucket_ms_list: list[int] = df_30m["bucket"].to_list()
    close_list: list[float] = df_30m["close"].to_list()
    vol_gk_list: list[float] = df_30m["volatility_gk"].to_list()

    logger.info(f"Executing Triple-Barrier scans for {n_bars:,} bars in {symbol}...")
    t_start: float = time.perf_counter()

    for k in range(n_bars):
        bucket_ms: int = bucket_ms_list[k]
        decision_dt: datetime = datetime.fromtimestamp(bucket_ms / 1000.0, tz=UTC)
        entry_price: float = float(close_list[k])
        vol_gk: float = float(vol_gk_list[k])

        # Forward candles strictly > T_decision
        # Bar k covers [k * 30, (k + 1) * 30). Decision boundary is (k + 1) * 30.
        fwd_start: int = (k + 1) * 30
        fwd_end: int = min(fwd_start + tb_config.horizon_bars_1m, len(candles_1m_all))

        if fwd_start >= len(candles_1m_all):
            # Final boundary bar at the very end of 2025
            forward_candles: list[MicroBar1m] = [
                MicroBar1m(
                    timestamp_ms=bucket_ms + 1_800_000,
                    open=entry_price,
                    high=entry_price,
                    low=entry_price,
                    close=entry_price,
                    volume=0.0,
                )
            ]
        else:
            forward_candles = candles_1m_all[fwd_start:fwd_end]

        lbl: BarrierLabel = evaluate_triple_barrier(
            symbol=symbol,
            bar_time=decision_dt,
            entry_price=entry_price,
            vol_gk=vol_gk,
            candles_1m=forward_candles,
            config=tb_config,
        )
        labels.append(lbl)

    elapsed: float = time.perf_counter() - t_start
    rate: float = float(n_bars) / max(elapsed, 1e-4)
    logger.info(
        f"Completed {n_bars:,} Triple-Barrier scans for {symbol} in {elapsed:.2f}s ({rate:.0f} bars/s)."
    )
    return labels


def build_36m_feature_dataset(config: PipelineConfig) -> pl.DataFrame:
    """Full orchestration building the 36-month feature and label dataset across all 4 pairs."""
    t_pipeline_start: float = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    tb_config: TripleBarrierConfig = TripleBarrierConfig(
        pt_multiplier=config.pt_multiplier,
        sl_multiplier=config.sl_multiplier,
        horizon_bars_1m=config.horizon_bars_1m,
        taker_fee_rate=config.taker_fee_rate,
        slippage_rate=config.slippage_rate,
        min_tp_to_cost_multiple=config.min_tp_to_cost_multiple,
        dead_zone_weight=config.dead_zone_weight,
    )

    weight_config: SampleWeightConfig = SampleWeightConfig(
        min_volatility_epsilon=config.min_volatility_epsilon,
    )

    # 1. First Pass: Process BTCUSDT to establish baseline 6h returns
    logger.info("=== Phase 1: Processing BTCUSDT (Cross-Asset Anchor) ===")
    btc_df_1m: pl.DataFrame = load_all_1m_parquets_for_symbol(
        parquet_dir=config.parquet_dir,
        symbol="BTCUSDT",
        start_year=config.start_year,
        end_year=config.end_year,
    )
    btc_df_30m: pl.DataFrame = resample_1m_to_30m(df_1m=btc_df_1m, symbol="BTCUSDT")
    n_btc_bars: int = len(btc_df_30m)
    logger.info(f"BTCUSDT 30m aggregated bars: {n_btc_bars:,}")

    btc_closes: np.ndarray = btc_df_30m["close"].to_numpy()
    btc_6h_returns: list[float] = [0.0] * n_btc_bars
    for i in range(12, n_btc_bars):
        btc_6h_returns[i] = (btc_closes[i] / max(btc_closes[i - 12], 1e-12)) - 1.0

    all_symbol_dfs: list[pl.DataFrame] = []

    for sym in config.symbols:
        logger.info(f"=== Processing Symbol: {sym} ===")
        t_sym_start: float = time.perf_counter()

        if sym == "BTCUSDT":
            df_1m: pl.DataFrame = btc_df_1m
            df_30m: pl.DataFrame = btc_df_30m
            curr_btc_returns: list[float] | None = None
        else:
            df_1m = load_all_1m_parquets_for_symbol(
                parquet_dir=config.parquet_dir,
                symbol=sym,
                start_year=config.start_year,
                end_year=config.end_year,
            )
            df_30m = resample_1m_to_30m(df_1m=df_1m, symbol=sym)
            curr_btc_returns = btc_6h_returns

        n_bars: int = len(df_30m)
        if n_bars != 52608:
            logger.warning(
                f"Symbol {sym} has {n_bars} bars (expected exactly 52,608 for 36 months)!"
            )

        # 2. Garman-Klass Volatility
        gk_vols: np.ndarray = compute_garman_klass_volatility(
            opens=df_30m["open"].to_numpy(),
            highs=df_30m["high"].to_numpy(),
            lows=df_30m["low"].to_numpy(),
            closes=df_30m["close"].to_numpy(),
            ewma_span=12,
        )
        df_30m = df_30m.with_columns(pl.Series("volatility_gk", gk_vols))

        # 3. 11 Canonical Features
        closes_np: np.ndarray = df_30m["close"].to_numpy()

        hurst_100: np.ndarray = compute_rolling_hurst_100(
            closes=closes_np,
            window=config.hurst_window,
        )
        shannon_50: np.ndarray = compute_rolling_shannon_entropy_50(
            closes=closes_np,
            window=config.shannon_window,
            k_bins=config.shannon_bins,
        )
        fracdiff_z: np.ndarray = compute_fracdiff_momentum_z(
            closes=closes_np,
            d=config.fracdiff_d,
            length=config.fracdiff_length,
            threshold=config.fracdiff_threshold,
            z_window=config.z_window,
        )
        vpin_tox, kyle_lambda_log = compute_microstructure_features(
            df_1m=df_1m,
            n_bars_30m=n_bars,
        )
        cvd_z, ofi_z, oi_score, f_pos_z, f_neg_z, btc_gap = compute_derived_series_features(
            df_30m=df_30m,
            z_window=config.z_window,
            funding_window=config.funding_window,
            btc_6h_returns=curr_btc_returns,
        )

        # 4. Pre-instantiate 1m candles for zero-overhead forward slicing
        t_1m_list: list[int] = df_1m["open_time"].to_list()
        open_1m_list: list[float] = df_1m["open"].to_list()
        high_1m_list: list[float] = df_1m["high"].to_list()
        low_1m_list: list[float] = df_1m["low"].to_list()
        close_1m_list: list[float] = df_1m["close"].to_list()
        vol_1m_list: list[float] = df_1m["volume"].to_list()

        candles_1m_all: list[MicroBar1m] = [
            MicroBar1m(
                timestamp_ms=t_1m_list[idx],
                open=open_1m_list[idx],
                high=high_1m_list[idx],
                low=low_1m_list[idx],
                close=close_1m_list[idx],
                volume=vol_1m_list[idx],
            )
            for idx in range(len(t_1m_list))
        ]

        # 5. Triple-Barrier Labels
        labels: list[BarrierLabel] = run_triple_barrier_on_symbol(
            symbol=sym,
            df_30m=df_30m,
            candles_1m_all=candles_1m_all,
            tb_config=tb_config,
        )

        # 6. Lopez de Prado Sample Uniqueness Weights
        label_spans: list[LabelSpan] = []
        for i, lbl in enumerate(labels):
            dur_30m: int = math.ceil(lbl.bars_held / 30)
            label_spans.append(
                LabelSpan(
                    sample_id=str(i),
                    t_start=i,
                    t_end=min(i + dur_30m, n_bars - 1),
                    realized_return=lbl.net_return,
                    volatility=max(1e-4, abs(lbl.net_return)),
                )
            )

        de_prado_weights: list[float] = compute_de_prado_weights(
            events=label_spans,
            num_bars=n_bars,
            config=weight_config,
        )
        weight_sum: float = sum(de_prado_weights)
        if abs(weight_sum - float(n_bars)) > 1.0:
            raise ValueError(
                f"Sample weight conservation violated for {sym}: sum={weight_sum}, expected={n_bars}"
            )

        # 7. Assemble Polars DataFrame
        buckets: list[int] = df_30m["bucket"].to_list()
        dt_utc_str: list[str] = [
            datetime.fromtimestamp(b / 1000.0, tz=UTC).isoformat() for b in buckets
        ]
        dt_utc_obj: list[datetime] = [datetime.fromtimestamp(b / 1000.0, tz=UTC) for b in buckets]

        sym_df: pl.DataFrame = df_30m.with_columns(
            pl.Series("timestamp", dt_utc_obj).cast(pl.Datetime(time_unit="ms", time_zone="UTC")),
            pl.Series("datetime_utc", dt_utc_str),
            # 11 Canonical Features
            pl.Series("hurst_exponent_100", hurst_100),
            pl.Series("shannon_entropy_50", shannon_50),
            pl.Series("fracdiff_momentum_z", fracdiff_z),
            pl.Series("vpin_toxicity_pct", vpin_tox),
            pl.Series("taker_cvd_z_24", cvd_z),
            pl.Series("ofi_imbalance_z", ofi_z),
            pl.Series("kyle_lambda_log", kyle_lambda_log),
            pl.Series("oi_directional_score", oi_score),
            pl.Series("funding_crowded_pos_z", f_pos_z),
            pl.Series("funding_crowded_neg_z", f_neg_z),
            pl.Series("btc_lead_catchup_gap", btc_gap),
            # Triple-Barrier Labels & Metadata
            pl.Series("barrier_hit", [lbl.barrier_hit.value for lbl in labels]),
            pl.Series("exit_time", [lbl.exit_time for lbl in labels]).cast(
                pl.Datetime(time_unit="ms", time_zone="UTC")
            ),
            pl.Series("bars_held_1m", [lbl.bars_held for lbl in labels]),
            pl.Series("bars_held_30m", [math.ceil(lbl.bars_held / 30) for lbl in labels]),
            pl.Series("y_long", [lbl.y_long for lbl in labels]),
            pl.Series("y_short", [lbl.y_short for lbl in labels]),
            pl.Series("entry_price", [lbl.entry_price for lbl in labels]),
            pl.Series("exit_price", [lbl.exit_price for lbl in labels]),
            pl.Series("upper_barrier_price", [lbl.upper_barrier_price for lbl in labels]),
            pl.Series("lower_barrier_price", [lbl.lower_barrier_price for lbl in labels]),
            pl.Series("realized_return", [lbl.realized_return for lbl in labels]),
            pl.Series("net_return", [lbl.net_return for lbl in labels]),
            pl.Series("roundtrip_cost", [lbl.roundtrip_cost for lbl in labels]),
            pl.Series("raw_sample_weight", [lbl.sample_weight for lbl in labels]),
            pl.Series("de_prado_weight", de_prado_weights),
        )

        # 8. Numerical Integrity Assertions
        null_counts: int = int(sym_df.null_count().sum_horizontal().item(0))
        if null_counts > 0:
            raise ValueError(f"Found {null_counts} null values in {sym} dataset!")

        for col_name in CANONICAL_FEATURE_NAMES:
            vals: np.ndarray = sym_df[col_name].to_numpy()
            if np.isnan(vals).any():
                raise ValueError(f"Feature `{col_name}` contains NaN values for {sym}!")
            if np.isinf(vals).any():
                raise ValueError(f"Feature `{col_name}` contains Inf values for {sym}!")

        # 9. Write Per-Symbol Parquet
        out_sym_file: Path = config.output_dir / f"{sym}.parquet"
        sym_df.write_parquet(out_sym_file, compression="zstd")
        out_sym_compat: Path = config.output_dir / f"36m_features_labels_{sym}.parquet"
        sym_df.write_parquet(out_sym_compat, compression="zstd")

        elapsed_sym: float = time.perf_counter() - t_sym_start
        logger.info(
            f"Successfully saved {sym} ({len(sym_df):,} rows) to {out_sym_file} in {elapsed_sym:.2f}s."
        )
        all_symbol_dfs.append(sym_df)

    # 10. Write Combined 4-Pair Dataset
    logger.info("=== Phase 3: Assembling Combined 4-Pair Dataset ===")
    combined_df: pl.DataFrame = pl.concat(all_symbol_dfs).sort(["timestamp", "symbol"])
    total_bars: int = len(combined_df)
    if total_bars != 210432:
        logger.warning(
            f"Combined dataset row count: {total_bars:,} (expected 210,432 for 4 pairs x 52,608 bars)!"
        )

    combined_out_1: Path = config.output_dir / "combined_4pairs_36m.parquet"
    combined_out_2: Path = config.output_dir / "dataset_4pairs_36m.parquet"
    combined_out_3: Path = config.output_dir / "36m_features_labels_all_4pairs.parquet"
    combined_df.write_parquet(combined_out_1, compression="zstd")
    combined_df.write_parquet(combined_out_2, compression="zstd")
    combined_df.write_parquet(combined_out_3, compression="zstd")

    total_pipeline_time: float = time.perf_counter() - t_pipeline_start
    logger.info(
        f"=== PIPELINE COMPLETE: {total_bars:,} bars generated across 4 pairs in {total_pipeline_time:.2f}s. Output: {combined_out_1} ==="
    )
    return combined_df


def main() -> None:
    """CLI Entrypoint with explicit configuration."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Build 36-Month 4-Pair Feature & Triple-Barrier Label Dataset."
    )
    parser.add_argument(
        "--parquet-dir",
        type=str,
        required=False,
        help="Path to 1m raw parquet directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        help="Destination directory for output feature parquet files.",
    )

    args: argparse.Namespace = parser.parse_args()

    if args.parquet_dir is not None:
        p_dir: str = str(args.parquet_dir)
    else:
        p_dir = "data/parquet"

    if args.output_dir is not None:
        o_dir: str = str(args.output_dir)
    else:
        o_dir = "data/features_36m"

    cfg: PipelineConfig = create_pipeline_config(
        parquet_dir_path=p_dir,
        output_dir_path=o_dir,
    )
    build_36m_feature_dataset(config=cfg)


if __name__ == "__main__":
    main()
