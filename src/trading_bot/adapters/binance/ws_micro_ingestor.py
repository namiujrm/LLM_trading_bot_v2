"""Binance Futures Real-Time WebSocket Ingestor and Gap Healer (Zero Defaults)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp

from trading_bot.adapters.binance.micro_accumulator import (
    EnrichedMicroBarRecord,
    MicroIngestorConfig,
    MinuteMicroAccumulator,
)
from trading_bot.infrastructure.config_schemas import BinanceConfig

logger = logging.getLogger(__name__)


class BinanceFuturesWSIngestor:
    """Consumes real-time Binance Futures WS streams, accumulating 1m enriched bars."""

    def __init__(
        self,
        symbols: list[str],
        cfg: MicroIngestorConfig,
        binance_cfg: BinanceConfig,
        on_bar_closed: Callable[[EnrichedMicroBarRecord], Awaitable[None]],
    ) -> None:
        self._symbols: list[str] = [s.lower() for s in symbols]
        self._cfg: MicroIngestorConfig = cfg
        self._binance_cfg: BinanceConfig = binance_cfg
        self._on_bar_closed: Callable[[EnrichedMicroBarRecord], Awaitable[None]] = on_bar_closed
        self._accumulators: dict[str, MinuteMicroAccumulator] = {}
        self._is_running: bool = False
        self._session: aiohttp.ClientSession | None = None
        self._task: asyncio.Task[None] | None = None

    def get_accumulator(self, symbol: str) -> MinuteMicroAccumulator:
        """Fetch current minute accumulator for symbol or raise KeyError."""
        upper_sym = symbol.upper()
        if upper_sym not in self._accumulators:
            raise KeyError(f"No active accumulator for symbol: {upper_sym}")
        return self._accumulators[upper_sym]

    def _get_stream_url(self) -> str:
        streams: list[str] = []
        for s in self._symbols:
            streams.append(f"{s}@aggTrade")
            streams.append(f"{s}@depth20@100ms")
            streams.append(f"{s}@forceOrder")
        stream_path = "/".join(streams)
        if self._binance_cfg.testnet:
            return f"wss://stream.binancefuture.com/stream?streams={stream_path}"
        return f"wss://fstream.binance.com/stream?streams={stream_path}"

    def _get_rest_base_url(self) -> str:
        if self._binance_cfg.testnet:
            return "https://testnet.binancefuture.com"
        return "https://fapi.binance.com"

    async def start(self) -> None:
        """Start WebSocket listener in background task."""
        self._is_running = True
        self._session = aiohttp.ClientSession()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop WebSocket connection and flush all remaining accumulators."""
        self._is_running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

        if self._session is not None:
            await self._session.close()
            self._session = None

        # Flush any remaining bars on shutdown
        for acc in self._accumulators.values():
            if acc.trade_count > 0 or len(acc.spread_bps_samples) > 0:
                record = acc.flush_record()
                await self._on_bar_closed(record)
        self._accumulators.clear()

    async def _run_loop(self) -> None:
        ws_url = self._get_stream_url()
        backoff_delay = 1.0
        max_backoff = 30.0

        while self._is_running:
            try:
                if self._session is None:
                    self._session = aiohttp.ClientSession()

                async with self._session.ws_connect(ws_url, heartbeat=15.0) as ws:
                    backoff_delay = 1.0  # Reset backoff on successful connect
                    logger.info("Connected to Binance Futures WebSocket multiplexed stream.")
                    async for msg in ws:
                        if not self._is_running:
                            break

                        if msg.type == aiohttp.WSMsgType.TEXT:
                            payload: dict[str, Any] = json.loads(msg.data)
                            await self._handle_message(payload)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(
                    "Binance WS connection error: %s. Reconnecting in %ss...",
                    e,
                    backoff_delay,
                )
                await asyncio.sleep(backoff_delay)
                backoff_delay = min(max_backoff, backoff_delay * 2.0)

    async def _handle_message(self, payload: dict[str, Any]) -> None:
        if "data" not in payload:
            return

        data: dict[str, Any] = payload["data"]
        if "e" not in data:
            return
        event_type = str(data["e"])

        if event_type == "aggTrade":
            symbol = str(data["s"]).upper()
            ts_ms = int(data["E"])
            price = float(data["p"])
            qty = float(data["q"])
            is_buyer_maker = bool(data["m"])

            acc = await self._ensure_accumulator(symbol, ts_ms)
            acc.on_agg_trade(price, qty, is_buyer_maker, ts_ms)

        elif event_type == "depthUpdate":
            symbol = str(data["s"]).upper()
            ts_ms = int(data["E"])
            if "b" not in data:
                return
            if "a" not in data:
                return
            raw_bids: list[list[str]] = data["b"]
            raw_asks: list[list[str]] = data["a"]

            bids = [(float(p), float(q)) for p, q in raw_bids if float(q) > 0.0]
            asks = [(float(p), float(q)) for p, q in raw_asks if float(q) > 0.0]

            if bids and asks:
                acc = await self._ensure_accumulator(symbol, ts_ms)
                acc.on_depth_update(bids, asks)

        elif event_type == "forceOrder":
            if "o" not in data:
                return
            o: dict[str, Any] = data["o"]
            symbol = str(o["s"]).upper()
            ts_ms = int(data["E"])
            side = str(o["S"])
            price = float(o["p"])
            qty = float(o["q"])

            acc = await self._ensure_accumulator(symbol, ts_ms)
            acc.on_force_order(side, price, qty)

    async def _ensure_accumulator(self, symbol: str, event_ts_ms: int) -> MinuteMicroAccumulator:
        minute_open = (event_ts_ms // 60000) * 60000
        acc = self._accumulators.get(symbol)

        if acc is None:
            new_acc = MinuteMicroAccumulator.create_empty(symbol, minute_open, self._cfg)
            self._accumulators[symbol] = new_acc
            return new_acc

        if event_ts_ms >= acc.minute_open_ts_ms + 60000:
            # Minute closed -> flush and notify
            record = acc.flush_record()
            await self._on_bar_closed(record)

            new_acc = MinuteMicroAccumulator.create_empty(symbol, minute_open, self._cfg)
            self._accumulators[symbol] = new_acc
            return new_acc

        return acc

    async def heal_rest_gaps(
        self,
        symbol: str,
        start_ts_ms: int,
        end_ts_ms: int,
    ) -> list[dict[str, Any]]:
        """Fetch missing 1m klines via REST API to heal connection gaps."""
        url = f"{self._get_rest_base_url()}/fapi/v1/klines"
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": "1m",
            "startTime": str(start_ts_ms),
            "endTime": str(end_ts_ms),
            "limit": "1000",
        }
        if self._session is None:
            self._session = aiohttp.ClientSession()

        async with self._session.get(url, params=params) as resp:
            if resp.status != 200:
                raise RuntimeError(
                    f"Binance REST klines request failed with status {resp.status}: {await resp.text()}"
                )
            raw_klines: list[list[Any]] = await resp.json()

        healed: list[dict[str, Any]] = []
        for k in raw_klines:
            healed.append(
                {
                    "open_time_ms": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "quote_volume": float(k[7]),
                    "taker_buy_vol": float(k[9]),
                    "taker_buy_quote_vol": float(k[10]),
                    "trade_count": int(k[8]),
                }
            )
        return healed
