"""Binance Futures Real-Time WebSocket Ingestor and Gap Healer (Zero Defaults)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
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
        self._last_seen_bar_ts_ms: dict[str, int] = {}
        self._is_running: bool = False
        self._session: aiohttp.ClientSession | None = None
        self._task: asyncio.Task[None] | None = None
        self._watchdog_task: asyncio.Task[None] | None = None

    def get_accumulator(self, symbol: str) -> MinuteMicroAccumulator:
        """Fetch current minute accumulator for symbol or raise KeyError."""
        upper_sym = symbol.upper()
        if upper_sym not in self._accumulators:
            raise KeyError(f"No active accumulator for symbol: {upper_sym}")
        return self._accumulators[upper_sym]

    def get_last_seen_bar_ts_ms(self, symbol: str) -> int | None:
        """Return the last seen 1m bar open timestamp in ms for the symbol."""
        upper_sym = symbol.upper()
        if upper_sym in self._last_seen_bar_ts_ms:
            return self._last_seen_bar_ts_ms[upper_sym]
        return None

    def _get_current_time_ms(self) -> int:
        return int(time.time() * 1000)

    def _get_stream_url(self) -> str:
        streams: list[str] = []
        for s in self._symbols:
            streams.append(f"{s}@aggTrade")
            streams.append(f"{s}@depth20@100ms")
            streams.append(f"{s}@forceOrder")
            streams.append(f"{s}@kline_1m")
        stream_path = "/".join(streams)
        if self._binance_cfg.testnet:
            return f"wss://stream.binancefuture.com/stream?streams={stream_path}"
        return f"wss://fstream.binance.com/stream?streams={stream_path}"

    def _get_rest_base_url(self) -> str:
        if self._binance_cfg.testnet:
            return "https://testnet.binancefuture.com"
        return "https://fapi.binance.com"

    async def start(self) -> None:
        """Start WebSocket listener and active minute watchdog in background tasks."""
        self._is_running = True
        self._session = aiohttp.ClientSession()
        now_ms = self._get_current_time_ms()
        minute_open = (now_ms // 60000) * 60000
        for s in self._symbols:
            upper_sym = s.upper()
            if upper_sym not in self._accumulators:
                self._accumulators[upper_sym] = MinuteMicroAccumulator.create_empty(
                    upper_sym, minute_open, self._cfg
                )
        self._task = asyncio.create_task(self._run_loop())
        self._watchdog_task = asyncio.create_task(self._run_minute_watchdog())

    async def stop(self) -> None:
        """Stop WebSocket connection and flush all remaining accumulators."""
        self._is_running = False
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watchdog_task
            self._watchdog_task = None

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
                self._last_seen_bar_ts_ms[record.symbol] = record.open_time_ms
                await self._on_bar_closed(record)
        self._accumulators.clear()

    async def _run_minute_watchdog(self) -> None:
        """Active minute-watchdog timer to ensure minute bars are flushed at second :00."""
        while self._is_running:
            try:
                await asyncio.sleep(1.0)
                await self._check_minute_flushes()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error in minute watchdog loop: %s", exc)

    async def _check_minute_flushes(self) -> None:
        """Check if any symbol's active minute window has elapsed and flush."""
        now_ms = self._get_current_time_ms()
        current_minute_open = (now_ms // 60000) * 60000
        for s in self._symbols:
            upper_sym = s.upper()
            if upper_sym in self._accumulators:
                acc = self._accumulators[upper_sym]
                if now_ms >= acc.minute_open_ts_ms + 60000:
                    await self._flush_symbol_bar(upper_sym, current_minute_open)

    async def _flush_symbol_bar(self, symbol: str, next_minute_open_ts_ms: int) -> None:
        """Flush current minute accumulator and initialize a fresh window."""
        upper_sym = symbol.upper()
        if upper_sym not in self._accumulators:
            return
        acc = self._accumulators[upper_sym]
        record = acc.flush_record()
        self._last_seen_bar_ts_ms[upper_sym] = record.open_time_ms
        self._accumulators[upper_sym] = MinuteMicroAccumulator.create_empty(
            upper_sym, next_minute_open_ts_ms, self._cfg
        )
        await self._on_bar_closed(record)

    async def _check_and_heal_gaps(self) -> None:
        """Auto-reconnect gap healer: verify continuity and fetch missing bars via REST."""
        now_ms = self._get_current_time_ms()
        for s in self._symbols:
            upper_sym = s.upper()
            if upper_sym in self._last_seen_bar_ts_ms:
                last_seen_ts = self._last_seen_bar_ts_ms[upper_sym]
                gap_ms = now_ms - (last_seen_ts + 60000)
                if gap_ms > 60000:
                    start_ts = last_seen_ts + 60000
                    end_ts = (now_ms // 60000) * 60000 - 1
                    if start_ts <= end_ts:
                        logger.info(
                            "Detected %sms gap for %s after reconnect (start=%s, end=%s). Invoking heal_rest_gaps().",
                            gap_ms,
                            upper_sym,
                            start_ts,
                            end_ts,
                        )
                        try:
                            healed_klines = await self.heal_rest_gaps(upper_sym, start_ts, end_ts)
                            for k in healed_klines:
                                healed_record = EnrichedMicroBarRecord(
                                    symbol=upper_sym,
                                    open_time_ms=k["open_time_ms"],
                                    open=k["open"],
                                    high=k["high"],
                                    low=k["low"],
                                    close=k["close"],
                                    volume=k["volume"],
                                    quote_volume=k["quote_volume"],
                                    taker_buy_vol=k["taker_buy_vol"],
                                    taker_buy_quote_vol=k["taker_buy_quote_vol"],
                                    trade_count=k["trade_count"],
                                    ofi_sum=0.0,
                                    obi_mean=0.0,
                                    obi_std=0.0,
                                    spread_bps_mean=0.0,
                                    liq_long_usd=0.0,
                                    liq_short_usd=0.0,
                                )
                                self._last_seen_bar_ts_ms[upper_sym] = healed_record.open_time_ms
                                await self._on_bar_closed(healed_record)
                        except Exception as exc:
                            logger.warning(
                                "Gracefully handled gap healing failure for %s: %s",
                                upper_sym,
                                exc,
                            )

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
                    await self._check_and_heal_gaps()

                    while self._is_running:
                        try:
                            msg = await asyncio.wait_for(ws.receive(), timeout=15.0)
                        except TimeoutError:
                            logger.warning(
                                "Binance WS stream silent for > 15s. Reconnection watchdog triggered."
                            )
                            await ws.close()
                            break

                        if msg.type == aiohttp.WSMsgType.TEXT:
                            payload: dict[str, Any] = json.loads(msg.data)
                            await self._handle_message(payload)
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                            aiohttp.WSMsgType.CLOSING,
                        ):
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

        elif event_type == "kline":
            if "k" not in data:
                return
            k: dict[str, Any] = data["k"]
            symbol = str(data["s"]).upper()
            kline_open_ts = int(k["t"])
            open_p = float(k["o"])
            high_p = float(k["h"])
            low_p = float(k["l"])
            close_p = float(k["c"])
            vol = float(k["v"])
            q_vol = float(k["q"])
            tb_vol = float(k["V"])
            tb_q_vol = float(k["Q"])
            trades = int(k["n"])
            is_closed = bool(k["x"])

            acc = await self._ensure_accumulator(symbol, kline_open_ts)
            acc.on_kline(
                open_p, high_p, low_p, close_p, vol, q_vol, tb_vol, tb_q_vol, trades, is_closed
            )
            if is_closed:
                await self._flush_symbol_bar(symbol, kline_open_ts + 60000)

    async def _ensure_accumulator(self, symbol: str, event_ts_ms: int) -> MinuteMicroAccumulator:
        minute_open = (event_ts_ms // 60000) * 60000
        upper_sym = symbol.upper()
        if upper_sym not in self._accumulators:
            new_acc = MinuteMicroAccumulator.create_empty(upper_sym, minute_open, self._cfg)
            self._accumulators[upper_sym] = new_acc
            return new_acc

        acc = self._accumulators[upper_sym]
        if event_ts_ms >= acc.minute_open_ts_ms + 60000:
            await self._flush_symbol_bar(upper_sym, minute_open)
            return self._accumulators[upper_sym]

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

        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status in (418, 429):
                    logger.warning(
                        "Binance REST rate limit or IP ban (HTTP %s) when healing %s. Skipping REST gap.",
                        resp.status,
                        symbol,
                    )
                    return []
                if resp.status != 200:
                    logger.warning(
                        "Binance REST klines request failed with status %s: %s",
                        resp.status,
                        await resp.text(),
                    )
                    return []
                raw_klines: list[list[Any]] = await resp.json()
        except Exception as exc:
            logger.warning("Network or client error during heal_rest_gaps for %s: %s", symbol, exc)
            return []

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
