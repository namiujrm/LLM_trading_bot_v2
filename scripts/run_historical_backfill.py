"""Historical Backfill Runner for 4 Binance Futures pairs across 36 months (Invariant #9 Compliant)."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import asyncpg
import polars as pl

from trading_bot.adapters.binance.historical_backfiller import HistoricalBackfiller

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("historical_backfill_runner")


@dataclass
class BackfillRunnerConfig:
    """Explicit configuration for the historical backfill runner (Zero Defaults)."""

    symbols: list[str]
    start_year: int
    start_month: int
    end_year: int
    end_month: int
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    max_db_connections: int
    parquet_base_dir: Path
    base_vision_url: str


def load_backfill_config_from_env() -> BackfillRunnerConfig:
    """Load configuration strictly from environment variables without silent fallbacks."""
    host = os.environ.get("POSTGRES_HOST")
    if host is None:
        host = "localhost"

    port_str = os.environ.get("POSTGRES_PORT")
    if port_str is None:
        port_str = "5433"

    db_name = os.environ.get("POSTGRES_DB")
    if db_name is None:
        db_name = "trading_bot_db"

    user = os.environ.get("POSTGRES_USER")
    if user is None:
        user = "postgres"

    password = os.environ.get("POSTGRES_PASSWORD")
    if password is None:
        password = "postgres"

    max_conns_str = os.environ.get("POSTGRES_MAX_CONNECTIONS")
    if max_conns_str is None:
        max_conns_str = "10"

    symbols_env = os.environ.get("BACKFILL_SYMBOLS")
    if symbols_env is not None:
        symbols = [s.strip().upper() for s in symbols_env.split(",") if len(s.strip()) > 0]
    else:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

    start_year_str = os.environ.get("BACKFILL_START_YEAR")
    if start_year_str is not None:
        start_year = int(start_year_str)
    else:
        start_year = 2023

    start_month_str = os.environ.get("BACKFILL_START_MONTH")
    if start_month_str is not None:
        start_month = int(start_month_str)
    else:
        start_month = 1

    end_year_str = os.environ.get("BACKFILL_END_YEAR")
    if end_year_str is not None:
        end_year = int(end_year_str)
    else:
        end_year = 2025

    end_month_str = os.environ.get("BACKFILL_END_MONTH")
    if end_month_str is not None:
        end_month = int(end_month_str)
    else:
        end_month = 12

    parquet_dir_env = os.environ.get("BACKFILL_PARQUET_DIR")
    if parquet_dir_env is not None:
        parquet_base_dir = Path(parquet_dir_env).resolve()
    else:
        parquet_base_dir = Path("data/parquet").resolve()

    base_vision_url = os.environ.get("BINANCE_VISION_URL")
    if base_vision_url is None:
        base_vision_url = "https://data.binance.vision"

    return BackfillRunnerConfig(
        symbols=symbols,
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
        db_host=host,
        db_port=int(port_str),
        db_name=db_name,
        db_user=user,
        db_password=password,
        max_db_connections=int(max_conns_str),
        parquet_base_dir=parquet_base_dir,
        base_vision_url=base_vision_url,
    )


def generate_year_month_tuples(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> list[tuple[int, int]]:
    """Generate sequential (year, month) pairs for the backfill window."""
    result: list[tuple[int, int]] = []
    current_year = start_year
    current_month = start_month

    while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
        result.append((current_year, current_month))
        if current_month == 12:
            current_year += 1
            current_month = 1
        else:
            current_month += 1
    return result


async def run_backfill(config: BackfillRunnerConfig) -> None:
    """Execute complete end-to-end historical backfill pipeline."""
    logger.info(
        "Initializing TimescaleDB connection pool at %s:%s...", config.db_host, config.db_port
    )
    pool = await asyncpg.create_pool(
        host=config.db_host,
        port=config.db_port,
        database=config.db_name,
        user=config.db_user,
        password=config.db_password,
        min_size=2,
        max_size=config.max_db_connections,
    )

    backfiller = HistoricalBackfiller(
        base_vision_url=config.base_vision_url,
        persister=None,
        pool=pool,
    )

    months_to_process = generate_year_month_tuples(
        config.start_year,
        config.start_month,
        config.end_year,
        config.end_month,
    )

    logger.info(
        "Starting Historical Backfill: symbols=%s, months_count=%d (from %d-%02d to %d-%02d)",
        config.symbols,
        len(months_to_process),
        config.start_year,
        config.start_month,
        config.end_year,
        config.end_month,
    )

    total_inserted = 0
    t_start = time.perf_counter()

    timeout = aiohttp.ClientTimeout(total=60.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for symbol in config.symbols:
            logger.info("=== Processing Symbol: %s ===", symbol)
            sym_inserted = 0

            for year, month in months_to_process:
                month_str = f"{month:02d}"
                parquet_path = config.parquet_base_dir / symbol / str(year) / f"{month_str}.parquet"

                t_month_start = time.perf_counter()
                try:
                    count, _df = await backfiller.backfill_monthly(
                        session=session,
                        symbol=symbol,
                        timeframe="1m",
                        year=year,
                        month=month,
                        parquet_output_path=parquet_path,
                    )
                    sym_inserted += count
                    total_inserted += count
                    elapsed = time.perf_counter() - t_month_start
                    logger.info(
                        "[%s] %d-%s: Ingested %d bars in %.2fs -> %s",
                        symbol,
                        year,
                        month_str,
                        count,
                        elapsed,
                        parquet_path,
                    )
                except Exception as exc:
                    logger.error(
                        "[%s] %d-%s: Failed to backfill month: %s",
                        symbol,
                        year,
                        month_str,
                        exc,
                    )
                    raise

            logger.info(
                "Completed Symbol %s: %d bars total ingested.",
                symbol,
                sym_inserted,
            )

    logger.info(
        "=== All monthly batches ingested. Total 1m bars: %d. Starting Continuous Aggregates Refresh ===",
        total_inserted,
    )

    # Continuous Aggregates Refresh over full backfill window
    start_ts_str = f"{config.start_year}-{config.start_month:02d}-01 00:00:00+00"
    # End timestamp 1 day after end of window or up to next year
    refresh_end_year = config.end_year + 1
    end_ts_str = f"{refresh_end_year}-01-01 00:00:00+00"

    t_cagg_start = time.perf_counter()
    logger.info(
        "Executing CALL refresh_continuous_aggregate from %s to %s...", start_ts_str, end_ts_str
    )
    await backfiller.refresh_continuous_aggregates(start_ts_str, end_ts_str)
    cagg_elapsed = time.perf_counter() - t_cagg_start
    logger.info("Continuous aggregates refreshed successfully in %.2fs.", cagg_elapsed)

    # Chunk Compression for chunks older than 7 days
    logger.info("Compressing historical TimescaleDB chunks older than 7 days...")
    t_comp_start = time.perf_counter()
    compressed = await backfiller.compress_chunks(7)
    comp_elapsed = time.perf_counter() - t_comp_start
    logger.info("Compressed %d chunks in %.2fs.", len(compressed), comp_elapsed)

    # Verification Queries
    logger.info("=== Verifying Final Database & Parquet State ===")
    async with pool.acquire() as conn:
        for sym in config.symbols:
            row_count_1m = await conn.fetchval(
                "SELECT count(*) FROM market_klines_1m_enriched WHERE symbol = $1;",
                sym,
            )
            row_count_30m = await conn.fetchval(
                "SELECT count(*) FROM market_klines_30m_cagg WHERE symbol = $1;",
                sym,
            )
            logger.info(
                "[%s] Verified DB: 1m rows = %d | 30m CAGG rows = %d",
                sym,
                row_count_1m,
                row_count_30m,
            )

    # Verify Parquet Files on Disk
    total_parquet_files = 0
    total_parquet_rows = 0
    for sym in config.symbols:
        sym_dir = config.parquet_base_dir / sym
        if sym_dir.exists():
            files = list(sym_dir.rglob("*.parquet"))
            total_parquet_files += len(files)
            for pf in files:
                read_df = pl.read_parquet(pf)
                total_parquet_rows += read_df.height

    logger.info(
        "Verified Parquet Store: %d files, %d total rows in %s",
        total_parquet_files,
        total_parquet_rows,
        config.parquet_base_dir,
    )

    total_pipeline_time = time.perf_counter() - t_start
    logger.info(
        "=== Historical Backfill Pipeline Completed Successfully in %.2fs ===",
        total_pipeline_time,
    )

    await pool.close()


def main() -> None:
    """CLI entry point."""
    config = load_backfill_config_from_env()
    asyncio.run(run_backfill(config))


if __name__ == "__main__":
    main()
