"""Integration Tests: HistoricalBackfiller (RAM Streaming, Polars, Parquet Zstd) & MicroAccumulator."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import polars as pl

from trading_bot.adapters.binance.historical_backfiller import HistoricalBackfiller
from trading_bot.adapters.binance.micro_accumulator import (
    MicroIngestorConfig,
    MinuteMicroAccumulator,
)


def create_sample_binance_zip() -> bytes:
    """Create a sample Binance Vision monthly klines ZIP archive in memory."""
    csv_content = (
        "1704067200000,42000.0,42050.0,41980.0,42020.0,10.5,1704067259999,441210.0,150,5.2,218504.0,0\n"
        "1704067260000,42020.0,42080.0,42010.0,42070.0,12.0,1704067319999,504840.0,180,7.0,294490.0,0\n"
        "1704067320000,42070.0,42090.0,42040.0,42060.0,8.5,1704067379999,357510.0,110,3.5,147210.0,0\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("BTCUSDT-1m-2024-01.csv", csv_content)
    return buffer.getvalue()


def create_sample_binance_zip_with_header() -> bytes:
    """Create a sample Binance Vision monthly klines ZIP archive with real header in memory."""
    csv_content = (
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
        "1704067200000,42000.0,42050.0,41980.0,42020.0,10.5,1704067259999,441210.0,150,5.2,218504.0,0\n"
        "1704067260000,42020.0,42080.0,42010.0,42070.0,12.0,1704067319999,504840.0,180,7.0,294490.0,0\n"
        "1704067320000,42070.0,42090.0,42040.0,42060.0,8.5,1704067379999,357510.0,110,3.5,147210.0,0\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("BTCUSDT-1m-2024-01.csv", csv_content)
    return buffer.getvalue()


def test_historical_backfiller_url_builders() -> None:
    backfiller = HistoricalBackfiller(
        base_vision_url="https://data.binance.vision",
        persister=None,
        pool=None,
    )
    m_url = backfiller.build_monthly_url("BTCUSDT", "1m", 2024, 1)
    assert (
        m_url
        == "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip"
    )

    d_url = backfiller.build_daily_url("ETHUSDT", "1m", "2024-01-15")
    assert (
        d_url
        == "https://data.binance.vision/data/futures/um/daily/klines/ETHUSDT/1m/ETHUSDT-1m-2024-01-15.zip"
    )


def test_historical_backfiller_ram_stream_and_polars_parsing(tmp_path: Path) -> None:
    backfiller = HistoricalBackfiller(
        base_vision_url="https://data.binance.vision",
        persister=None,
        pool=None,
    )
    zip_bytes = create_sample_binance_zip()

    # 1. Parse in memory without temporary files (headerless CSV)
    df = backfiller.parse_zip_to_polars(zip_bytes, "BTCUSDT")
    assert df.height == 3
    assert "symbol" in df.columns
    assert "ofi_sum" in df.columns
    assert "spread_bps_mean" in df.columns
    assert df["close"][0] == 42020.0
    # Microstructure reconstruction checks
    # OFI = 2 * 5.2 - 10.5 = -0.1
    assert abs(df["ofi_sum"][0] - (-0.1)) < 1e-6
    assert df["spread_bps_mean"][0] >= 0.5

    # 2. Prepare tuples for binary copy
    records = backfiller.prepare_records_for_copy(df)
    assert len(records) == 3
    assert records[0][1] == "BTCUSDT"
    assert records[0][5] == 42020.0

    # 3. Export to Parquet with zstd compression
    parquet_file = tmp_path / "btcusdt_1m.parquet"
    backfiller.export_to_parquet(df, parquet_file)
    assert parquet_file.exists()

    read_back = pl.read_parquet(parquet_file)
    assert read_back.height == 3
    assert read_back["close"][1] == 42070.0


def test_historical_backfiller_adaptive_header_sniffer(tmp_path: Path) -> None:
    backfiller = HistoricalBackfiller(
        base_vision_url="https://data.binance.vision",
        persister=None,
        pool=None,
    )
    zip_bytes = create_sample_binance_zip_with_header()

    # Parse in memory with header row present
    df = backfiller.parse_zip_to_polars(zip_bytes, "BTCUSDT")
    assert df.height == 3
    assert df["symbol"][0] == "BTCUSDT"
    assert df["trade_count"][0] == 150
    assert df["taker_buy_vol"][0] == 5.2
    assert abs(df["ofi_sum"][0] - (-0.1)) < 1e-6
    assert df["spread_bps_mean"][0] >= 0.5


def test_micro_accumulator_ofi_and_obi_computation() -> None:
    cfg = MicroIngestorConfig(
        obi_depth_levels=5,
        obi_decay_rate=0.2,
        bps_scale_factor=10000.0,
    )
    acc = MinuteMicroAccumulator.create_empty("BTCUSDT", 1704067200000, cfg)

    # 1. Feed order book quotes
    bids1 = [(42000.0, 10.0), (41990.0, 15.0), (41980.0, 20.0)]
    asks1 = [(42005.0, 8.0), (42010.0, 12.0), (42020.0, 25.0)]
    acc.on_depth_update(bids1, asks1)

    # Next update: Bid price moved up -> positive OFI impact
    bids2 = [(42002.0, 12.0), (42000.0, 10.0), (41990.0, 15.0)]
    asks2 = [(42005.0, 6.0), (42010.0, 12.0), (42020.0, 25.0)]
    acc.on_depth_update(bids2, asks2)

    assert acc.ofi_sum > 0.0
    assert len(acc.obi_samples) == 2
    assert len(acc.spread_bps_samples) == 2

    # 2. Feed trades
    acc.on_agg_trade(price=42003.0, qty=1.5, is_buyer_maker=False, trade_ts_ms=1704067210000)
    acc.on_agg_trade(price=42005.0, qty=2.0, is_buyer_maker=False, trade_ts_ms=1704067220000)
    assert acc.trade_count == 2
    assert acc.high_price == 42005.0
    assert acc.taker_buy_vol == 3.5

    # 3. Feed liquidation
    acc.on_force_order(side="SELL", price=41995.0, qty=1.0)
    assert acc.liq_long_usd == 41995.0

    # 4. Flush record
    record = acc.flush_record()
    assert record.symbol == "BTCUSDT"
    assert record.trade_count == 2
    assert record.liq_long_usd == 41995.0
    assert record.ofi_sum > 0.0
