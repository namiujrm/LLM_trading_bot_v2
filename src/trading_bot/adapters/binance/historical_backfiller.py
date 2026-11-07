"""Historical Backfiller streaming Binance Vision ZIP archives into Polars and Parquet (Zero Defaults)."""

from __future__ import annotations

import io
import logging
import zipfile
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import polars as pl

logger = logging.getLogger(__name__)


class HistoricalBackfiller:
    """Streams historical klines from data.binance.vision directly in RAM, avoids disk clutter."""

    def __init__(
        self,
        base_vision_url: str,
        persister: Callable[[str, list[tuple[Any, ...]]], Awaitable[int]] | None,
    ) -> None:
        self._base_vision_url: str = base_vision_url
        self._persister: Callable[[str, list[tuple[Any, ...]]], Awaitable[int]] | None = persister

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
            if not csv_files:
                raise ValueError("No CSV file found inside Binance ZIP archive.")

            first_csv = csv_files[0]
            with z.open(first_csv) as f:
                csv_bytes = f.read()

        # Binance klines CSV columns standard:
        # open_time, open, high, low, close, volume, close_time, quote_volume, count,
        # taker_buy_vol, taker_buy_quote_vol, ignore
        raw_df = pl.read_csv(
            io.BytesIO(csv_bytes),
            has_header=False,
            new_columns=[
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
            ],
            schema_overrides={
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
            },
        )

        # Enrich with symbol and default zero microstructure placeholders for historical data
        enriched_df = raw_df.with_columns(
            [
                pl.lit(sym_upper).alias("symbol"),
                pl.lit(0.0).alias("ofi_sum"),
                pl.lit(0.0).alias("obi_mean"),
                pl.lit(0.0).alias("obi_std"),
                pl.lit(0.0).alias("spread_bps_mean"),
                pl.lit(0.0).alias("liq_long_usd"),
                pl.lit(0.0).alias("liq_short_usd"),
            ]
        )
        return enriched_df

    def prepare_records_for_copy(self, df: pl.DataFrame) -> list[tuple[Any, ...]]:
        """Transform Polars DataFrame into list of tuples matching market_klines_1m_enriched schema."""
        records: list[tuple[Any, ...]] = []
        for row in df.iter_rows(named=True):
            ts_ms: int = row["open_time"]
            dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
            records.append(
                (
                    dt,
                    str(row["symbol"]),
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                    float(row["quote_volume"]),
                    float(row["taker_buy_vol"]),
                    float(row["taker_buy_quote_vol"]),
                    int(row["trade_count"]),
                    float(row["ofi_sum"]),
                    float(row["obi_mean"]),
                    float(row["obi_std"]),
                    float(row["spread_bps_mean"]),
                    float(row["liq_long_usd"]),
                    float(row["liq_short_usd"]),
                )
            )
        return records

    def export_to_parquet(self, df: pl.DataFrame, target_path: Path) -> None:
        """Export Polars DataFrame to Parquet file compressed with zstandard."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(target_path, compression="zstd")

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

        inserted_count = 0
        if self._persister is not None:
            records = self.prepare_records_for_copy(df)
            inserted_count = await self._persister(symbol, records)

        if parquet_output_path is not None:
            self.export_to_parquet(df, parquet_output_path)

        return (inserted_count, df)
