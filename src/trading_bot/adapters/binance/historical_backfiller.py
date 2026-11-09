"""Historical Backfiller streaming Binance Vision ZIP archives into Polars and Parquet (Zero Defaults)."""

from __future__ import annotations

import io
import logging
import math
import zipfile
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import asyncpg
import polars as pl

logger = logging.getLogger(__name__)

ENRICHED_COLUMNS: list[str] = [
    "open_time",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_vol",
    "taker_buy_quote_vol",
    "trade_count",
    "ofi_sum",
    "obi_mean",
    "obi_std",
    "spread_bps_mean",
    "liq_long_usd",
    "liq_short_usd",
]


class HistoricalBackfiller:
    """Streams historical klines from data.binance.vision directly in RAM, avoids disk clutter."""

    def __init__(
        self,
        base_vision_url: str,
        persister: Callable[[str, list[tuple[Any, ...]]], Awaitable[int]] | None,
        pool: asyncpg.Pool[asyncpg.Record] | None,
    ) -> None:
        self._base_vision_url: str = base_vision_url
        self._persister: Callable[[str, list[tuple[Any, ...]]], Awaitable[int]] | None = persister
        self._pool: asyncpg.Pool[asyncpg.Record] | None = pool

    def build_monthly_url(self, symbol: str, timeframe: str, year: int, month: int) -> str:
        """Construct data.binance.vision URL for monthly kline ZIP archive."""
        sym_upper = symbol.upper()
        month_str = f"{month:02d}"
        file_name = f"{sym_upper}-{timeframe}-{year}-{month_str}.zip"
        return f"{self._base_vision_url}/data/futures/um/monthly/klines/{sym_upper}/{timeframe}/{file_name}"

    def build_daily_url(self, symbol: str, timeframe: str, date_iso: str) -> str:
        """Construct data.binance.vision URL for daily kline ZIP archive."""
        sym_upper = symbol.upper()
        file_name = f"{sym_upper}-{timeframe}-{date_iso}.zip"
        return f"{self._base_vision_url}/data/futures/um/daily/klines/{sym_upper}/{timeframe}/{file_name}"

    async def fetch_zip_bytes(self, session: aiohttp.ClientSession, url: str) -> bytes:
        """Download remote ZIP archive into RAM without temporary disk writes."""
        async with session.get(url) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Failed to fetch archive from {url} with HTTP status {response.status}"
                )
            return await response.read()

    def parse_zip_to_polars(self, zip_content: bytes, symbol: str) -> pl.DataFrame:
        """Decompress in-memory ZIP and parse CSV payload into enriched Polars DataFrame."""
        sym_upper = symbol.upper()
        with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
            namelist = z.namelist()
            csv_files = [n for n in namelist if n.endswith(".csv")]
            if len(csv_files) == 0:
                raise ValueError("No CSV file found inside Binance ZIP archive.")

            first_csv = csv_files[0]
            with z.open(first_csv) as f:
                csv_bytes = f.read()

        # Adaptive header sniffer: detect whether first line starts with "open_time"
        first_line = csv_bytes.split(b"\n", 1)[0]
        has_header = first_line.startswith(b"open_time")

        if has_header:
            schema_overrides_header = {
                "open_time": pl.Int64,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "close_time": pl.Int64,
                "quote_volume": pl.Float64,
                "count": pl.Int64,
                "taker_buy_volume": pl.Float64,
                "taker_buy_quote_volume": pl.Float64,
                "ignore": pl.String,
            }
            df = pl.read_csv(
                io.BytesIO(csv_bytes),
                has_header=True,
                schema_overrides=schema_overrides_header,
            )
            rename_map: dict[str, str] = {}
            if "count" in df.columns:
                rename_map["count"] = "trade_count"
            if "taker_buy_volume" in df.columns:
                rename_map["taker_buy_volume"] = "taker_buy_vol"
            if "taker_buy_quote_volume" in df.columns:
                rename_map["taker_buy_quote_volume"] = "taker_buy_quote_vol"
            if len(rename_map) > 0:
                df = df.rename(rename_map)
        else:
            standard_cols = [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trade_count",
                "taker_buy_vol",
                "taker_buy_quote_vol",
                "ignore",
            ]
            schema_overrides_no_header = {
                "open_time": pl.Int64,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "close_time": pl.Int64,
                "quote_volume": pl.Float64,
                "trade_count": pl.Int64,
                "taker_buy_vol": pl.Float64,
                "taker_buy_quote_vol": pl.Float64,
                "ignore": pl.String,
            }
            df = pl.read_csv(
                io.BytesIO(csv_bytes),
                has_header=False,
                new_columns=standard_cols,
                schema_overrides=schema_overrides_no_header,
            )

        df = df.with_columns(
            [
                pl.col("open_time").cast(pl.Int64),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
                pl.col("close_time").cast(pl.Int64),
                pl.col("quote_volume").cast(pl.Float64),
                pl.col("trade_count").cast(pl.Int64),
                pl.col("taker_buy_vol").cast(pl.Float64),
                pl.col("taker_buy_quote_vol").cast(pl.Float64),
            ]
        )

        # Vectorized Microstructure Reconstruction:
        # 1. Net Taker Order Flow Imbalance (OFI Proxy): 2 * taker_buy_vol - volume
        ofi_expr = (2.0 * pl.col("taker_buy_vol") - pl.col("volume")).alias("ofi_sum")

        # 2. Corwin-Schultz (2012) High-Low spread estimator:
        # beta = [ln(H_{t}/L_{t})]^2 + [ln(H_{t-1}/L_{t-1})]^2
        # gamma = [ln(max(H_t, H_{t-1}) / min(L_t, L_{t-1}))]^2
        # alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))
        # S_t = max(0.0, 2*(exp(alpha) - 1) / (1 + exp(alpha)))
        sqrt_2 = math.sqrt(2.0)
        denom = 3.0 - 2.0 * sqrt_2

        high_col = pl.col("high")
        low_col = pl.col("low")
        high_prev = pl.col("high").shift(1)
        low_prev = pl.col("low").shift(1)

        hl_sq = (high_col / low_col).log() ** 2
        hl_prev_sq = (high_prev / low_prev).log() ** 2
        beta = hl_sq + hl_prev_sq

        max_h2 = pl.max_horizontal(high_col, high_prev)
        min_l2 = pl.min_horizontal(low_col, low_prev)
        gamma = (max_h2 / min_l2).log() ** 2

        alpha = (beta.mul(2.0).sqrt() - beta.sqrt()) / denom - (gamma / denom).sqrt()
        spread_raw = 2.0 * (alpha.exp() - 1.0) / (alpha.exp() + 1.0)
        spread_bps_expr = (
            pl.when(spread_raw > 0.0)
            .then(spread_raw)
            .otherwise(0.0)
            .mul(10000.0)
            .clip(lower_bound=0.5)
            .fill_null(strategy="backward")
            .fill_null(0.5)
            .alias("spread_bps_mean")
        )

        enriched_df = df.with_columns(
            [
                pl.lit(sym_upper).alias("symbol"),
                ofi_expr,
                pl.lit(0.0).alias("obi_mean"),
                pl.lit(0.0).alias("obi_std"),
                spread_bps_expr,
                pl.lit(0.0).alias("liq_long_usd"),
                pl.lit(0.0).alias("liq_short_usd"),
            ]
        )
        return enriched_df

    def prepare_records_for_copy(self, df: pl.DataFrame) -> list[tuple[Any, ...]]:
        """Transform Polars DataFrame into list of tuples matching market_klines_1m_enriched schema."""
        times = df["open_time"].to_list()
        symbols = df["symbol"].to_list()
        opens = df["open"].to_list()
        highs = df["high"].to_list()
        lows = df["low"].to_list()
        closes = df["close"].to_list()
        volumes = df["volume"].to_list()
        quote_volumes = df["quote_volume"].to_list()
        taker_buy_vols = df["taker_buy_vol"].to_list()
        taker_buy_quote_vols = df["taker_buy_quote_vol"].to_list()
        trade_counts = df["trade_count"].to_list()
        ofi_sums = df["ofi_sum"].to_list()
        obi_means = df["obi_mean"].to_list()
        obi_stds = df["obi_std"].to_list()
        spread_bps_means = df["spread_bps_mean"].to_list()
        liq_long_usds = df["liq_long_usd"].to_list()
        liq_short_usds = df["liq_short_usd"].to_list()

        row_count = len(times)
        records: list[tuple[Any, ...]] = [
            (
                datetime.fromtimestamp(times[i] / 1000.0, tz=UTC),
                symbols[i],
                opens[i],
                highs[i],
                lows[i],
                closes[i],
                volumes[i],
                quote_volumes[i],
                taker_buy_vols[i],
                taker_buy_quote_vols[i],
                trade_counts[i],
                ofi_sums[i],
                obi_means[i],
                obi_stds[i],
                spread_bps_means[i],
                liq_long_usds[i],
                liq_short_usds[i],
            )
            for i in range(row_count)
        ]
        return records

    async def copy_records_to_timescale(
        self,
        symbol: str,
        records: list[tuple[Any, ...]],
    ) -> int:
        """High-throughput bulk copy into TimescaleDB market_klines_1m_enriched using asyncpg."""
        if self._pool is None:
            raise RuntimeError("Database pool not configured on HistoricalBackfiller.")
        if len(records) == 0:
            return 0

        start_dt: datetime = records[0][0]
        end_dt: datetime = records[-1][0]
        async with self._pool.acquire() as conn:
            # Clean existing records in the window to guarantee idempotency and avoid UniqueViolationError
            await conn.execute(
                "DELETE FROM market_klines_1m_enriched WHERE symbol = $1 AND open_time >= $2 AND open_time <= $3;",
                symbol,
                start_dt,
                end_dt,
            )
            await conn.copy_records_to_table(
                "market_klines_1m_enriched",
                records=records,
                columns=ENRICHED_COLUMNS,
            )
        return len(records)

    async def refresh_continuous_aggregates(
        self,
        start_ts: str,
        end_ts: str,
    ) -> None:
        """Execute explicit PL/pgSQL CALL to refresh 15m, 30m, 1h, 4h continuous aggregates."""
        if self._pool is None:
            raise RuntimeError("Database pool not configured on HistoricalBackfiller.")
        cagg_views = [
            "market_klines_15m_cagg",
            "market_klines_30m_cagg",
            "market_klines_1h_cagg",
            "market_klines_4h_cagg",
        ]
        async with self._pool.acquire() as conn:
            for view_name in cagg_views:
                logger.info(
                    "Calling refresh_continuous_aggregate for %s from %s to %s",
                    view_name,
                    start_ts,
                    end_ts,
                )
                query = f"CALL refresh_continuous_aggregate('{view_name}', '{start_ts}'::timestamptz, '{end_ts}'::timestamptz);"
                await conn.execute(query)

    async def compress_chunks(
        self,
        older_than_days: int,
    ) -> list[str]:
        """Compress hypertable chunks older than the given number of days."""
        if self._pool is None:
            raise RuntimeError("Database pool not configured on HistoricalBackfiller.")
        compressed_chunks: list[str] = []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT chunk_schema, chunk_name
                FROM timescaledb_information.chunks
                WHERE hypertable_name = 'market_klines_1m_enriched'
                  AND is_compressed = FALSE
                  AND range_end < now() - make_interval(days => $1);
                """,
                older_than_days,
            )
            for r in rows:
                full_chunk_name = f"{r['chunk_schema']}.{r['chunk_name']}"
                await conn.fetchval(
                    f"SELECT compress_chunk('{full_chunk_name}', if_not_compressed => TRUE);"
                )
                compressed_chunks.append(full_chunk_name)
        return compressed_chunks

    def export_to_parquet(self, df: pl.DataFrame, target_path: Path) -> None:
        """Export Polars DataFrame to Parquet file compressed with zstandard level 9."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(target_path, compression="zstd", compression_level=9)

    async def backfill_monthly(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        timeframe: str,
        year: int,
        month: int,
        parquet_output_path: Path | None,
    ) -> tuple[int, pl.DataFrame]:
        """Download, parse in RAM, persist to database, and optionally export Parquet."""
        url = self.build_monthly_url(symbol, timeframe, year, month)
        zip_bytes = await self.fetch_zip_bytes(session, url)
        df = self.parse_zip_to_polars(zip_bytes, symbol)

        records = self.prepare_records_for_copy(df)
        inserted_count = 0
        if self._pool is not None:
            inserted_count = await self.copy_records_to_timescale(symbol, records)
        elif self._persister is not None:
            inserted_count = await self._persister(symbol, records)

        if parquet_output_path is not None:
            self.export_to_parquet(df, parquet_output_path)

        return (inserted_count, df)
