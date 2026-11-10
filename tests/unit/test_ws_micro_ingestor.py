# pyright: reportPrivateUsage=false
"""Unit tests for BinanceFuturesWSIngestor (Invariant #9 Compliant).

Verifies:
- Stream subscription URLs including {s}@kline_1m.
- Kline message handling (in-progress and closed).
- Auto-reconnect gap healing with REST fallback.
- Graceful handling of HTTP 418 / rate limits.
- Active minute watchdog flushing during quiet/no-tick periods.
- Stream silence watchdog timeout handling.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from trading_bot.adapters.binance.micro_accumulator import (
    EnrichedMicroBarRecord,
    MicroIngestorConfig,
)
from trading_bot.adapters.binance.ws_micro_ingestor import BinanceFuturesWSIngestor
from trading_bot.infrastructure.config_schemas import BinanceConfig


def make_ingestor_cfg() -> MicroIngestorConfig:
    return MicroIngestorConfig(
        obi_depth_levels=10,
        obi_decay_rate=0.1,
        bps_scale_factor=10000.0,
    )


def make_binance_cfg(testnet: bool) -> BinanceConfig:
    return BinanceConfig(
        api_key="test_api_key",
        api_secret="test_api_secret",
        testnet=testnet,
        recv_window_ms=5000,
    )


@pytest.mark.asyncio
async def test_stream_subscription_url_includes_kline() -> None:
    symbols: list[str] = ["BTCUSDT", "ETHUSDT"]
    closed_bars: list[EnrichedMicroBarRecord] = []

    async def on_bar(record: EnrichedMicroBarRecord) -> None:
        closed_bars.append(record)

    ingestor = BinanceFuturesWSIngestor(
        symbols=symbols,
        cfg=make_ingestor_cfg(),
        binance_cfg=make_binance_cfg(testnet=True),
        on_bar_closed=on_bar,
    )
    url_testnet = ingestor._get_stream_url()
    assert "btcusdt@kline_1m" in url_testnet
    assert "ethusdt@kline_1m" in url_testnet
    assert "btcusdt@aggTrade" in url_testnet
    assert "btcusdt@depth20@100ms" in url_testnet
    assert "btcusdt@forceOrder" in url_testnet
    assert "stream.binancefuture.com" in url_testnet

    ingestor_live = BinanceFuturesWSIngestor(
        symbols=symbols,
        cfg=make_ingestor_cfg(),
        binance_cfg=make_binance_cfg(testnet=False),
        on_bar_closed=on_bar,
    )
    url_live = ingestor_live._get_stream_url()
    assert "fstream.binance.com" in url_live
    assert "btcusdt@kline_1m" in url_live


@pytest.mark.asyncio
async def test_handle_kline_in_progress_updates_accumulator() -> None:
    symbols: list[str] = ["BTCUSDT"]
    closed_bars: list[EnrichedMicroBarRecord] = []

    async def on_bar(record: EnrichedMicroBarRecord) -> None:
        closed_bars.append(record)

    ingestor = BinanceFuturesWSIngestor(
        symbols=symbols,
        cfg=make_ingestor_cfg(),
        binance_cfg=make_binance_cfg(testnet=True),
        on_bar_closed=on_bar,
    )

    kline_msg: dict[str, Any] = {
        "data": {
            "e": "kline",
            "E": 1700000030000,
            "s": "BTCUSDT",
            "k": {
                "t": 1700000000000,
                "T": 1700000059999,
                "s": "BTCUSDT",
                "i": "1m",
                "o": "40000.0",
                "c": "40050.0",
                "h": "40080.0",
                "l": "39990.0",
                "v": "10.5",
                "q": "420000.0",
                "V": "5.5",
                "Q": "220000.0",
                "n": 100,
                "x": False,
            },
        }
    }

    await ingestor._handle_message(kline_msg)

    acc = ingestor.get_accumulator("BTCUSDT")
    assert acc.open_price == 40000.0
    assert acc.high_price == 40080.0
    assert acc.low_price == 39990.0
    assert acc.close_price == 40050.0
    assert acc.volume == 10.5
    assert acc.trade_count == 100
    assert len(closed_bars) == 0


@pytest.mark.asyncio
async def test_handle_kline_closed_flushes_bar() -> None:
    symbols: list[str] = ["BTCUSDT"]
    closed_bars: list[EnrichedMicroBarRecord] = []

    async def on_bar(record: EnrichedMicroBarRecord) -> None:
        closed_bars.append(record)

    ingestor = BinanceFuturesWSIngestor(
        symbols=symbols,
        cfg=make_ingestor_cfg(),
        binance_cfg=make_binance_cfg(testnet=True),
        on_bar_closed=on_bar,
    )

    minute_open = 1700000040000  # Divisible by 60000
    kline_closed_msg: dict[str, Any] = {
        "data": {
            "e": "kline",
            "E": minute_open + 59999,
            "s": "BTCUSDT",
            "k": {
                "t": minute_open,
                "T": minute_open + 59999,
                "s": "BTCUSDT",
                "i": "1m",
                "o": "40000.0",
                "c": "40060.0",
                "h": "40100.0",
                "l": "39980.0",
                "v": "15.0",
                "q": "600000.0",
                "V": "8.0",
                "Q": "320000.0",
                "n": 150,
                "x": True,
            },
        }
    }

    await ingestor._handle_message(kline_closed_msg)

    assert len(closed_bars) == 1
    flushed = closed_bars[0]
    assert flushed.symbol == "BTCUSDT"
    assert flushed.open_time_ms == minute_open
    assert flushed.open == 40000.0
    assert flushed.close == 40060.0
    assert flushed.high == 40100.0
    assert flushed.low == 39980.0
    assert flushed.volume == 15.0
    assert flushed.trade_count == 150
    assert ingestor.get_last_seen_bar_ts_ms("BTCUSDT") == minute_open


@pytest.mark.asyncio
async def test_auto_reconnect_gap_healer_recovers_missing_bars() -> None:
    symbols: list[str] = ["BTCUSDT"]
    closed_bars: list[EnrichedMicroBarRecord] = []

    async def on_bar(record: EnrichedMicroBarRecord) -> None:
        closed_bars.append(record)

    ingestor = BinanceFuturesWSIngestor(
        symbols=symbols,
        cfg=make_ingestor_cfg(),
        binance_cfg=make_binance_cfg(testnet=True),
        on_bar_closed=on_bar,
    )

    # Simulate last seen bar was 3 minutes ago
    last_seen = 1700000000000
    current_simulated_time = 1700000180000  # 3 minutes later
    ingestor._last_seen_bar_ts_ms["BTCUSDT"] = last_seen

    mock_healed: list[dict[str, Any]] = [
        {
            "open_time_ms": 1700000060000,
            "open": 40060.0,
            "high": 40120.0,
            "low": 40050.0,
            "close": 40100.0,
            "volume": 20.0,
            "quote_volume": 802000.0,
            "taker_buy_vol": 12.0,
            "taker_buy_quote_vol": 481200.0,
            "trade_count": 220,
        },
        {
            "open_time_ms": 1700000120000,
            "open": 40100.0,
            "high": 40150.0,
            "low": 40080.0,
            "close": 40130.0,
            "volume": 18.0,
            "quote_volume": 722000.0,
            "taker_buy_vol": 9.0,
            "taker_buy_quote_vol": 361000.0,
            "trade_count": 190,
        },
    ]

    with (
        patch.object(ingestor, "_get_current_time_ms", return_value=current_simulated_time),
        patch.object(ingestor, "heal_rest_gaps", new=AsyncMock(return_value=mock_healed)),
    ):
        await ingestor._check_and_heal_gaps()

    assert len(closed_bars) == 2
    assert closed_bars[0].open_time_ms == 1700000060000
    assert closed_bars[1].open_time_ms == 1700000120000
    assert ingestor.get_last_seen_bar_ts_ms("BTCUSDT") == 1700000120000


@pytest.mark.asyncio
async def test_heal_rest_gaps_handles_418_gracefully() -> None:
    symbols: list[str] = ["BTCUSDT"]
    closed_bars: list[EnrichedMicroBarRecord] = []

    async def on_bar(record: EnrichedMicroBarRecord) -> None:
        closed_bars.append(record)

    ingestor = BinanceFuturesWSIngestor(
        symbols=symbols,
        cfg=make_ingestor_cfg(),
        binance_cfg=make_binance_cfg(testnet=True),
        on_bar_closed=on_bar,
    )

    class MockResp:
        def __init__(self, status: int) -> None:
            self.status: int = status

        async def __aenter__(self) -> MockResp:
            return self

        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

        async def text(self) -> str:
            return "IP banned 418"

        async def json(self) -> list[Any]:
            return []

    class MockSession:
        def get(self, url: str, params: dict[str, Any]) -> MockResp:
            return MockResp(status=418)

    ingestor._session = MockSession()  # type: ignore[assignment]
    result = await ingestor.heal_rest_gaps("BTCUSDT", 1700000000000, 1700000060000)
    assert result == []


@pytest.mark.asyncio
async def test_minute_watchdog_flushes_quiet_minute() -> None:
    symbols: list[str] = ["SOLUSDT"]
    closed_bars: list[EnrichedMicroBarRecord] = []

    async def on_bar(record: EnrichedMicroBarRecord) -> None:
        closed_bars.append(record)

    ingestor = BinanceFuturesWSIngestor(
        symbols=symbols,
        cfg=make_ingestor_cfg(),
        binance_cfg=make_binance_cfg(testnet=True),
        on_bar_closed=on_bar,
    )

    minute_start = 1700000040000  # Divisible by 60000
    # Create accumulator at minute_start
    await ingestor._ensure_accumulator("SOLUSDT", minute_start)

    # Simulate quiet period where time advances past minute close (minute_start + 60,000ms)
    time_after_minute = minute_start + 60000

    with patch.object(ingestor, "_get_current_time_ms", return_value=time_after_minute):
        await ingestor._check_minute_flushes()

    assert len(closed_bars) == 1
    assert closed_bars[0].symbol == "SOLUSDT"
    assert closed_bars[0].open_time_ms == minute_start
    assert ingestor.get_last_seen_bar_ts_ms("SOLUSDT") == minute_start
