"""Binance Futures User Data Stream WebSocket Adapter (Invariant #9 Compliant).

Clean Architecture: Technical Adapter Layer.
Listens to real-time order fill and position update events (ORDER_TRADE_UPDATE, ACCOUNT_UPDATE)
and dispatches callbacks for immediate OCO bracket order cancellation.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, cast

import aiohttp

from trading_bot.infrastructure.config_schemas import BinanceConfig
from trading_bot.ports.execution_gateway_port import ExecutionGatewayPort

logger = logging.getLogger(__name__)


class BinanceUserDataStream:
    """Manages Binance Futures WebSocket User Data Stream connection and OCO triggers."""

    def __init__(
        self,
        binance_config: BinanceConfig,
        gateway: ExecutionGatewayPort,
        session: aiohttp.ClientSession | None,
        on_bracket_leg_filled: (Callable[[str, str, str, str], Awaitable[None]] | None),
    ) -> None:
        self._config: BinanceConfig = binance_config
        self._gateway: ExecutionGatewayPort = gateway
        self._session: aiohttp.ClientSession | None = session
        self._owns_session: bool = session is None
        self._on_bracket_leg_filled: Callable[[str, str, str, str], Awaitable[None]] | None = (
            on_bracket_leg_filled
        )

        self._listen_key: str | None = None
        self._is_running: bool = False
        self._stop_event: asyncio.Event = asyncio.Event()
        self._ws_task: asyncio.Task[None] | None = None
        self._keepalive_task: asyncio.Task[None] | None = None

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def listen_key(self) -> str | None:
        return self._listen_key

    def _resolve_ws_endpoint(self, key: str) -> str:
        """Resolve WebSocket endpoint based on testnet setting."""
        if self._config.testnet:
            return f"wss://stream.binancefuture.com/ws/{key}"
        return f"wss://fstream.binance.com/ws/{key}"

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def start(self) -> None:
        """Acquire listenKey, start keepalive scheduler and begin WebSocket message loop."""
        if self._is_running:
            return

        self._is_running = True
        self._stop_event.clear()

        # 1. Acquire initial listenKey from gateway
        self._listen_key = await self._gateway.create_user_data_stream()
        logger.info("Acquired Binance User Data Stream listenKey: %s...", self._listen_key[:12])

        # 2. Spawn background tasks
        self._keepalive_task = asyncio.create_task(self._run_keepalive_loop())
        self._ws_task = asyncio.create_task(self._run_ws_loop())

    async def stop(self) -> None:
        """Stop WebSocket, cancel keepalive task, and release listenKey on exchange."""
        if not self._is_running:
            return

        self._is_running = False
        self._stop_event.set()

        if self._keepalive_task is not None and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._keepalive_task
            self._keepalive_task = None

        if self._ws_task is not None and not self._ws_task.done():
            self._ws_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ws_task
            self._ws_task = None

        if self._listen_key is not None:
            try:
                await self._gateway.close_user_data_stream(self._listen_key)
                logger.info("Closed Binance User Data Stream listenKey")
            except Exception as exc:
                logger.warning("Error closing listenKey on exchange: %s", exc)
            self._listen_key = None

        if self._owns_session and self._session is not None and not self._session.closed:
            await self._session.close()

    async def _run_keepalive_loop(self) -> None:
        """Send PUT /fapi/v1/listenKey every 30 minutes to keep listenKey valid."""
        keepalive_interval_seconds: float = 1800.0  # 30m
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(keepalive_interval_seconds)
                if self._listen_key is not None and not self._stop_event.is_set():
                    await self._gateway.keepalive_user_data_stream(self._listen_key)
                    logger.debug("Successfully refreshed User Data Stream listenKey")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Failed to refresh User Data Stream listenKey: %s", exc)

    async def _run_ws_loop(self) -> None:
        """Main WebSocket read and dispatch loop with exponential backoff reconnect."""
        reconnect_delay_seconds: float = 2.0
        max_reconnect_delay: float = 30.0

        while not self._stop_event.is_set():
            if self._listen_key is None:
                try:
                    self._listen_key = await self._gateway.create_user_data_stream()
                except Exception as exc:
                    logger.error("Failed to acquire listenKey during reconnect: %s", exc)
                    await asyncio.sleep(reconnect_delay_seconds)
                    reconnect_delay_seconds = min(
                        reconnect_delay_seconds * 1.5, max_reconnect_delay
                    )
                    continue

            ws_url = self._resolve_ws_endpoint(self._listen_key)
            session = await self._get_session()

            try:
                logger.info("Connecting to Binance User Data Stream WebSocket...")
                async with session.ws_connect(ws_url, heartbeat=20.0) as ws:
                    logger.info("Connected to Binance User Data Stream WebSocket")
                    reconnect_delay_seconds = 2.0

                    async for msg in ws:
                        if self._stop_event.is_set():
                            break

                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            await self.handle_ws_message(data)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            logger.warning("User Data Stream WebSocket connection closed: %s", msg)
                            break
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(
                    "User Data Stream WebSocket error: %s. Reconnecting in %.1fs...",
                    exc,
                    reconnect_delay_seconds,
                )
                # Re-acquire listen key on next attempt in case expired
                self._listen_key = None
                await asyncio.sleep(reconnect_delay_seconds)
                reconnect_delay_seconds = min(reconnect_delay_seconds * 1.5, max_reconnect_delay)

    async def handle_ws_message(self, data: dict[str, Any]) -> None:
        """Parse incoming WebSocket message and handle ORDER_TRADE_UPDATE."""
        if "e" not in data:
            return

        event_type = str(data["e"])
        if event_type != "ORDER_TRADE_UPDATE":
            return

        if "o" not in data:
            return
        if not isinstance(data["o"], dict):
            return

        order_data: dict[str, Any] = cast(dict[str, Any], data["o"])
        if "s" not in order_data:
            return
        if "c" not in order_data:
            return
        if "X" not in order_data:
            return

        symbol: str = str(order_data["s"]).upper()
        client_order_id: str = str(order_data["c"])
        order_status: str = str(order_data["X"])

        if order_status != "FILLED":
            return

        # Check if the filled order is a bracket order leg (SL or TP)
        if client_order_id.endswith("_sl"):
            parent_id = client_order_id[:-3]
            opposing_algo_id = f"{parent_id}_tp"
            logger.info(
                "Bracket SL order %s FILLED for %s. Triggering immediate cancellation of TP leg %s",
                client_order_id,
                symbol,
                opposing_algo_id,
            )
            if self._on_bracket_leg_filled is not None:
                try:
                    await self._on_bracket_leg_filled(symbol, "SL", parent_id, opposing_algo_id)
                except Exception as exc:
                    logger.error("Error in on_bracket_leg_filled callback for SL: %s", exc)

        elif client_order_id.endswith("_tp"):
            parent_id = client_order_id[:-3]
            opposing_algo_id = f"{parent_id}_sl"
            logger.info(
                "Bracket TP order %s FILLED for %s. Triggering immediate cancellation of SL leg %s",
                client_order_id,
                symbol,
                opposing_algo_id,
            )
            if self._on_bracket_leg_filled is not None:
                try:
                    await self._on_bracket_leg_filled(symbol, "TP", parent_id, opposing_algo_id)
                except Exception as exc:
                    logger.error("Error in on_bracket_leg_filled callback for TP: %s", exc)
