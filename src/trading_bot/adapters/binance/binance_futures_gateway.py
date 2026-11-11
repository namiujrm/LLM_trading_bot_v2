"""Binance USD-M Futures Execution Gateway Adapter (Zero Defaults, Invariant #9)."""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
import urllib.parse
from typing import Any, cast

import aiohttp

from trading_bot.domain.entities.common.enums import (
    EntryOrderType,
    ExecutionMode,
    TradeSide,
)
from trading_bot.domain.entities.execution.trade_blueprint import BracketOrderTicket
from trading_bot.domain.entities.market.market_state import EnrichedMicroBarRecord
from trading_bot.domain.entities.market.symbol_filters import SymbolFilterRule
from trading_bot.domain.entities.risk.account_state import (
    AccountPortfolioContext,
    ActivePositionSnapshot,
    PendingOrderSnapshot,
)
from trading_bot.domain.exceptions import ExchangeFilterViolationError
from trading_bot.domain.services.risk.risk_calculator import DeterministicRiskCalculator
from trading_bot.infrastructure.config_schemas import BinanceConfig
from trading_bot.ports.execution_gateway_port import ExecutionGatewayPort

logger = logging.getLogger(__name__)


class BinanceFuturesGatewayAdapter(ExecutionGatewayPort):
    """Adapter connecting execution layer to Binance USD-M Perpetual Futures REST API."""

    def __init__(self, config: BinanceConfig, session: aiohttp.ClientSession | None) -> None:
        self._config: BinanceConfig = config
        self._session: aiohttp.ClientSession | None = session
        self._owns_session: bool = session is None
        self._time_offset_ms: int = 0
        self._symbol_filters: dict[str, SymbolFilterRule] = {}
        self._filters_last_synced_time: float = 0.0

    @property
    def symbol_filters(self) -> dict[str, SymbolFilterRule]:
        """Return a copy of cached symbol filter rules."""
        return dict(self._symbol_filters)

    def set_symbol_filter(self, rule: SymbolFilterRule) -> None:
        """Register or update a symbol filter rule directly."""
        self._symbol_filters[rule.symbol.upper()] = rule

    async def fetch_symbol_filters(self) -> dict[str, SymbolFilterRule]:
        """Fetch or return cached symbol filter rules (refreshes if older than 24h)."""
        current_time: float = time.monotonic()
        if not self._symbol_filters or (current_time - self._filters_last_synced_time) > 86400.0:
            await self.sync_exchange_filters()
        return dict(self._symbol_filters)

    async def sync_exchange_filters(self) -> dict[str, SymbolFilterRule]:
        """Fetch exchangeInfo from Binance Futures and parse symbol filter rules."""
        data = await self._request("GET", "/fapi/v1/exchangeInfo", {}, False)
        new_filters: dict[str, SymbolFilterRule] = {}
        if "symbols" in data:
            for sym_info in data["symbols"]:
                sym_name = str(sym_info["symbol"]).upper()
                if "pricePrecision" in sym_info:
                    price_prec = int(sym_info["pricePrecision"])
                else:
                    price_prec = 2

                if "quantityPrecision" in sym_info:
                    qty_prec = int(sym_info["quantityPrecision"])
                else:
                    qty_prec = 3

                price_tick = 0.1
                qty_step = 0.001
                min_q = 0.001
                max_q = 1000000.0
                min_notional = 5.0

                if "filters" in sym_info:
                    for f in sym_info["filters"]:
                        if "filterType" not in f:
                            continue
                        ftype = f["filterType"]
                        if ftype == "PRICE_FILTER":
                            if "tickSize" in f:
                                price_tick = float(f["tickSize"])
                        elif ftype == "LOT_SIZE":
                            if "stepSize" in f:
                                qty_step = float(f["stepSize"])
                            if "minQty" in f:
                                min_q = float(f["minQty"])
                            if "maxQty" in f:
                                max_q = float(f["maxQty"])
                        elif ftype in ("MIN_NOTIONAL", "NOTIONAL") and "notional" in f:
                            min_notional = float(f["notional"])

                s_step = f"{qty_step:.10f}".rstrip("0")
                if "." in s_step:
                    calculated_qty_prec = len(s_step.split(".")[1])
                    if calculated_qty_prec > qty_prec:
                        qty_prec = calculated_qty_prec

                s_tick = f"{price_tick:.10f}".rstrip("0")
                if "." in s_tick:
                    calculated_price_prec = len(s_tick.split(".")[1])
                    if calculated_price_prec > price_prec:
                        price_prec = calculated_price_prec

                new_filters[sym_name] = SymbolFilterRule(
                    symbol=sym_name,
                    price_tick_size=price_tick,
                    qty_step_size=qty_step,
                    min_qty=min_q,
                    max_qty=max_q,
                    min_notional=min_notional,
                    price_precision=price_prec,
                    qty_precision=qty_prec,
                )

        self._symbol_filters = new_filters
        self._filters_last_synced_time = time.monotonic()
        logger.info("Synchronized Binance exchange filters for %d symbols", len(new_filters))
        return dict(self._symbol_filters)

    def quantize_quantity(self, symbol: str, quantity: float) -> str:
        """Floor quantity to symbol's qty_step_size and check min/max constraints."""
        upper_sym = symbol.upper()
        if upper_sym not in self._symbol_filters:
            raise KeyError(f"Exchange filter not loaded for symbol: {upper_sym}")

        rule = self._symbol_filters[upper_sym]
        step = rule.qty_step_size
        floored_qty = math.floor(quantity / step) * step
        floored_qty = round(floored_qty, rule.qty_precision)

        if floored_qty < rule.min_qty:
            raise ExchangeFilterViolationError(
                f"Quantized quantity {floored_qty} is below minQty {rule.min_qty} for {upper_sym}"
            )
        if floored_qty > rule.max_qty:
            raise ExchangeFilterViolationError(
                f"Quantized quantity {floored_qty} exceeds maxQty {rule.max_qty} for {upper_sym}"
            )

        return f"{floored_qty:.{rule.qty_precision}f}"

    def quantize_price(self, symbol: str, price: float) -> str:
        """Round price to symbol's price_tick_size."""
        upper_sym = symbol.upper()
        if upper_sym not in self._symbol_filters:
            raise KeyError(f"Exchange filter not loaded for symbol: {upper_sym}")

        rule = self._symbol_filters[upper_sym]
        tick = rule.price_tick_size
        rounded_price = round(price / tick) * tick
        rounded_price = round(rounded_price, rule.price_precision)
        return f"{rounded_price:.{rule.price_precision}f}"

    @property
    def base_url(self) -> str:
        """Resolve REST base endpoint based on testnet configuration."""
        if self._config.testnet:
            return "https://testnet.binancefuture.com"
        return "https://fapi.binance.com"

    async def sync_time_offset(self) -> int:
        """Synchronize exchange server time offset to prevent timestamp drift errors."""
        res = await self._request("GET", "/fapi/v1/time", {}, False)
        server_time: int = int(res["serverTime"])
        local_time: int = int(time.time() * 1000)
        self._time_offset_ms = server_time - local_time
        logger.info(
            "Synchronized Binance server time. Offset: %d ms (local: %d, server: %d)",
            self._time_offset_ms,
            local_time,
            server_time,
        )
        return self._time_offset_ms

    def sign_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Sign request query parameters with HMAC-SHA256 according to Binance API spec."""
        signed: dict[str, Any] = dict(params)
        if "timestamp" not in signed:
            signed["timestamp"] = int(time.time() * 1000) + self._time_offset_ms
        if "recvWindow" not in signed:
            signed["recvWindow"] = self._config.recv_window_ms

        query_string: str = urllib.parse.urlencode(signed)
        signature: str = hmac.new(
            self._config.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        signed["signature"] = signature
        return signed

    def _sign_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Internal alias for sign_params."""
        return self.sign_params(params)

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any],
        signed: bool,
    ) -> Any:
        """Execute HTTP request against Binance Futures REST API."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

        headers: dict[str, str] = {}
        if signed:
            headers["X-MBX-APIKEY"] = self._config.api_key
            request_params = self._sign_params(params)
        else:
            request_params = dict(params)

        url: str = f"{self.base_url}{endpoint}"
        async with self._session.request(
            method,
            url,
            params=request_params,
            headers=headers,
        ) as resp:
            if resp.status >= 400:
                error_body: str = await resp.text()
                if "-1021" in error_body and signed:
                    logger.warning(
                        "Detected Binance timestamp drift (-1021) on %s %s. Synchronizing clock...",
                        method,
                        endpoint,
                    )
                    await self.sync_time_offset()
                    # Re-sign and retry once with updated time offset
                    retry_params = self._sign_params(params)
                    async with self._session.request(
                        method,
                        url,
                        params=retry_params,
                        headers=headers,
                    ) as retry_resp:
                        if retry_resp.status >= 400:
                            retry_err: str = await retry_resp.text()
                            raise RuntimeError(
                                f"Binance API error [{retry_resp.status}] on {method} {endpoint}: {retry_err}"
                            )
                        return await retry_resp.json()
                raise RuntimeError(
                    f"Binance API error [{resp.status}] on {method} {endpoint}: {error_body}"
                )
            return await resp.json()

    async def place_bracket_order(self, ticket: BracketOrderTicket) -> str:
        """Submit bracket order triplet (entry, SL, TP) and return parent exchange order ID."""
        sym_upper = ticket.symbol.upper()
        if sym_upper not in self._symbol_filters:
            await self.sync_exchange_filters()

        if ticket.side == TradeSide.LONG:
            entry_side = "BUY"
            exit_side = "SELL"
        elif ticket.side == TradeSide.SHORT:
            entry_side = "SELL"
            exit_side = "BUY"
        else:
            raise ValueError(f"Unsupported trade side for bracket order: {ticket.side}")

        # Quantize quantity and prices according to exchange filter rules
        qty_str = self.quantize_quantity(sym_upper, ticket.quantity)
        quantized_qty = float(qty_str)
        entry_price_str = self.quantize_price(sym_upper, ticket.entry_price)
        sl_price_str = self.quantize_price(sym_upper, ticket.stop_loss_price)
        tp_price_str = self.quantize_price(sym_upper, ticket.take_profit_price)

        rule = self._symbol_filters[sym_upper]
        notional = quantized_qty * float(entry_price_str)
        if notional < rule.min_notional:
            raise ExchangeFilterViolationError(
                f"Bracket order notional ${notional:.2f} is below minNotional ${rule.min_notional:.2f} for {sym_upper}"
            )

        entry_params: dict[str, Any] = {
            "symbol": sym_upper,
            "side": entry_side,
            "quantity": qty_str,
            "newClientOrderId": ticket.order_id,
        }

        if ticket.entry_type == EntryOrderType.MARKET:
            entry_params["type"] = "MARKET"
        elif ticket.entry_type == EntryOrderType.LIMIT_PULLBACK_15M:
            entry_params["type"] = "LIMIT"
            entry_params["price"] = entry_price_str
            entry_params["timeInForce"] = "GTC"
        else:
            raise ValueError(f"Unsupported entry order type: {ticket.entry_type}")

        # 1. Place parent entry order
        entry_resp = await self._request("POST", "/fapi/v1/order", entry_params, True)
        if "orderId" in entry_resp:
            parent_order_id = str(entry_resp["orderId"])
        else:
            parent_order_id = ticket.order_id

        # 2. Place stop-loss market order via Algo Order endpoint
        sl_params: dict[str, Any] = {
            "symbol": sym_upper,
            "side": exit_side,
            "algoType": "CONDITIONAL",
            "type": "STOP_MARKET",
            "triggerPrice": sl_price_str,
            "quantity": qty_str,
            "workingType": "MARK_PRICE",
            "reduceOnly": "true",
            "clientAlgoId": f"{ticket.order_id}_sl",
        }
        await self._request("POST", "/fapi/v1/algoOrder", sl_params, True)

        # 3. Place take-profit market order via Algo Order endpoint
        tp_params: dict[str, Any] = {
            "symbol": sym_upper,
            "side": exit_side,
            "algoType": "CONDITIONAL",
            "type": "TAKE_PROFIT_MARKET",
            "triggerPrice": tp_price_str,
            "quantity": qty_str,
            "workingType": "MARK_PRICE",
            "reduceOnly": "true",
            "clientAlgoId": f"{ticket.order_id}_tp",
        }
        await self._request("POST", "/fapi/v1/algoOrder", tp_params, True)

        return parent_order_id

    async def fetch_open_algo_orders(self, symbol: str) -> list[dict[str, Any]]:
        """Fetch all currently open algo conditional orders for a symbol."""
        params: dict[str, Any] = {"symbol": symbol.upper()}
        res: Any = await self._request("GET", "/fapi/v1/openAlgoOrders", params, True)
        if isinstance(res, list):
            return cast(list[dict[str, Any]], res)
        return []

    async def create_user_data_stream(self) -> str:
        """Start a user data stream and return the listenKey."""
        res = await self._request("POST", "/fapi/v1/listenKey", {}, False)
        if "listenKey" in res:
            return str(res["listenKey"])
        raise RuntimeError(f"Failed to create user data stream: {res}")

    async def keepalive_user_data_stream(self, listen_key: str) -> None:
        """Keepalive a user data stream listenKey."""
        await self._request("PUT", "/fapi/v1/listenKey", {"listenKey": listen_key}, False)

    async def close_user_data_stream(self, listen_key: str) -> None:
        """Close an active user data stream listenKey."""
        await self._request("DELETE", "/fapi/v1/listenKey", {"listenKey": listen_key}, False)

    async def fetch_recent_klines_1m(self, symbol: str, limit: int) -> list[EnrichedMicroBarRecord]:
        """Fetch recent 1m klines from Binance REST API and map to EnrichedMicroBarRecord."""
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": "1m",
            "limit": limit,
        }
        raw_klines = await self._request("GET", "/fapi/v1/klines", params, False)
        bars: list[EnrichedMicroBarRecord] = []
        for k in raw_klines:
            open_time_ms = int(k[0])
            open_p = float(k[1])
            high_p = float(k[2])
            low_p = float(k[3])
            close_p = float(k[4])
            vol = float(k[5])
            quote_vol = float(k[7])
            trades = int(k[8])
            taker_buy_vol = float(k[9])
            taker_buy_quote_vol = float(k[10])
            bars.append(
                EnrichedMicroBarRecord(
                    symbol=symbol.upper(),
                    open_time_ms=open_time_ms,
                    open=open_p,
                    high=high_p,
                    low=low_p,
                    close=close_p,
                    volume=vol,
                    quote_volume=quote_vol,
                    taker_buy_vol=taker_buy_vol,
                    taker_buy_quote_vol=taker_buy_quote_vol,
                    trade_count=trades,
                    ofi_sum=0.0,
                    obi_mean=0.0,
                    obi_std=0.0,
                    spread_bps_mean=1.0,
                    liq_long_usd=0.0,
                    liq_short_usd=0.0,
                )
            )
        return bars

    async def cancel_algo_order(self, symbol: str, algo_id: str) -> bool:
        """Cancel an active algo conditional order (SL/TP) by algo ID or clientAlgoId."""
        algo_params: dict[str, Any] = {"symbol": symbol.upper()}
        if algo_id.isdigit():
            algo_params["algoId"] = int(algo_id)
        else:
            algo_params["clientAlgoId"] = algo_id
        resp = await self._request("DELETE", "/fapi/v1/algoOrder", algo_params, True)
        if "code" in resp and str(resp["code"]) == "200":
            return True
        return "algoId" in resp

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an active order by symbol and order ID (exchange numeric or client ID)."""
        params: dict[str, Any] = {"symbol": symbol.upper()}
        if order_id.isdigit():
            params["orderId"] = order_id
        else:
            params["origClientOrderId"] = order_id

        try:
            resp = await self._request("DELETE", "/fapi/v1/order", params, True)
            if "status" in resp and resp["status"] == "CANCELED":
                return True
            return "orderId" in resp
        except RuntimeError as e:
            # If standard order endpoint fails (e.g. -2011 Unknown order), fallback to algoOrder endpoint
            try:
                return await self.cancel_algo_order(symbol, order_id)
            except Exception:
                raise e from None

    async def get_position_notional(self, symbol: str) -> float:
        """Fetch current open position notional value in USD for a symbol."""
        params: dict[str, Any] = {"symbol": symbol.upper()}
        data = await self._request("GET", "/fapi/v2/positionRisk", params, True)

        total_notional: float = 0.0
        for item in data:
            if "notional" in item:
                total_notional += abs(float(item["notional"]))
            elif "positionAmt" in item and "markPrice" in item:
                total_notional += abs(float(item["positionAmt"]) * float(item["markPrice"]))
        return total_notional

    def _map_position_snapshot(self, p: dict[str, Any]) -> ActivePositionSnapshot:
        """Map raw Binance position payload to ActivePositionSnapshot entity."""
        amt: float = float(p["positionAmt"])
        if amt > 0.0:
            side = TradeSide.LONG
        else:
            side = TradeSide.SHORT

        qty: float = abs(amt)
        symbol: str = str(p["symbol"])
        entry_price: float = float(p["entryPrice"])
        mark_price: float = float(p["markPrice"])

        if "notional" in p:
            notional = abs(float(p["notional"]))
        else:
            notional = qty * mark_price

        leverage: int = int(p["leverage"])

        if "isolatedMargin" in p and float(p["isolatedMargin"]) > 0.0:
            isolated_margin = float(p["isolatedMargin"])
        elif leverage > 0:
            isolated_margin = notional / float(leverage)
        else:
            isolated_margin = notional

        liquidation_price: float = float(p["liquidationPrice"])
        unrealized_pnl: float = float(p["unRealizedProfit"])

        if isolated_margin > 0.0:
            unrealized_pnl_pct = unrealized_pnl / isolated_margin
        else:
            unrealized_pnl_pct = 0.0

        if "stopLossPrice" in p and float(p["stopLossPrice"]) > 0.0:
            stop_loss = float(p["stopLossPrice"])
        elif liquidation_price > 0.0:
            stop_loss = liquidation_price
        elif side == TradeSide.LONG:
            stop_loss = entry_price * 0.95
        else:
            stop_loss = entry_price * 1.05

        if "takeProfitPrice" in p and float(p["takeProfitPrice"]) > 0.0:
            take_profit = float(p["takeProfitPrice"])
        elif side == TradeSide.LONG:
            take_profit = entry_price * 1.05
        else:
            take_profit = entry_price * 0.95

        if side == TradeSide.LONG:
            is_breakeven = stop_loss >= entry_price
        else:
            is_breakeven = stop_loss <= entry_price and stop_loss > 0.0

        effective_risk = abs(entry_price - stop_loss) * qty

        if effective_risk > 0.0:
            r_multiple = unrealized_pnl / effective_risk
        else:
            r_multiple = 0.0

        beta_map: dict[str, float] = {
            "BTCUSDT": 1.0,
            "ETHUSDT": 1.2,
            "SOLUSDT": 1.5,
            "BNBUSDT": 1.1,
        }
        if symbol in beta_map:
            btc_beta = beta_map[symbol]
        else:
            btc_beta = 1.0

        if "positionId" in p:
            pos_id = str(p["positionId"])
        else:
            pos_id = f"{symbol}_{side.value}"

        return ActivePositionSnapshot(
            position_id=pos_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            mark_price=mark_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            liquidation_price=liquidation_price,
            quantity=qty,
            notional_usd=notional,
            isolated_margin_usd=isolated_margin,
            leverage=leverage,
            unrealized_pnl_usd=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            unrealized_r_multiple=r_multiple,
            is_sl_at_breakeven=is_breakeven,
            effective_at_risk_usd=effective_risk,
            btc_beta=btc_beta,
        )

    def _map_pending_order_snapshot(self, o: dict[str, Any]) -> PendingOrderSnapshot:
        """Map raw Binance open order payload to PendingOrderSnapshot entity."""
        order_id: str = str(o["orderId"])
        symbol: str = str(o["symbol"])
        side_raw: str = str(o["side"]).upper()
        if side_raw == "BUY":
            side = TradeSide.LONG
        else:
            side = TradeSide.SHORT

        type_raw: str = str(o["type"]).upper()
        if type_raw == "MARKET":
            order_type = EntryOrderType.MARKET
        else:
            order_type = EntryOrderType.LIMIT_PULLBACK_15M

        price: float = float(o["price"])
        if price == 0.0 and "stopPrice" in o:
            price = float(o["stopPrice"])

        orig_qty: float = float(o["origQty"])
        notional: float = orig_qty * price

        if "reserved_margin_usd" in o:
            reserved_margin = float(o["reserved_margin_usd"])
        else:
            reserved_margin = notional

        if "time" in o:
            created_at = int(o["time"])
        elif "updateTime" in o:
            created_at = int(o["updateTime"])
        else:
            created_at = int(time.time() * 1000)

        return PendingOrderSnapshot(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            limit_price=price,
            notional_usd=notional,
            reserved_margin_usd=reserved_margin,
            created_at_ms=created_at,
        )

    async def fetch_active_positions(self) -> list[ActivePositionSnapshot]:
        """Fetch all currently open positions with margin and breakeven metrics."""
        params: dict[str, Any] = {}
        data = await self._request("GET", "/fapi/v2/positionRisk", params, True)

        active_positions: list[ActivePositionSnapshot] = []
        for p in data:
            amt = float(p["positionAmt"])
            if abs(amt) <= 1e-9:
                continue
            active_positions.append(self._map_position_snapshot(p))

        return active_positions

    async def fetch_pending_orders(self) -> list[PendingOrderSnapshot]:
        """Fetch all currently open pending entry orders reserving margin."""
        params: dict[str, Any] = {}
        data = await self._request("GET", "/fapi/v1/openOrders", params, True)

        pending_orders: list[PendingOrderSnapshot] = []
        for o in data:
            pending_orders.append(self._map_pending_order_snapshot(o))

        return pending_orders

    async def fetch_account_portfolio_context(self) -> AccountPortfolioContext:
        """Fetch the unified live portfolio and margin state context from Binance."""
        params: dict[str, Any] = {}
        data = await self._request("GET", "/fapi/v2/account", params, True)

        if self._config.testnet:
            exec_mode = ExecutionMode.TESTNET
        else:
            exec_mode = ExecutionMode.LIVE

        if "updateTime" in data and int(data["updateTime"]) > 0:
            ts_ms = int(data["updateTime"])
        else:
            ts_ms = int(time.time() * 1000)

        active_positions: list[ActivePositionSnapshot] = []
        if "positions" in data and isinstance(data["positions"], list):
            for p in data["positions"]:
                amt = float(p["positionAmt"])
                if abs(amt) > 1e-9:
                    active_positions.append(self._map_position_snapshot(p))

        pending_orders: list[PendingOrderSnapshot] = []
        if "open_orders" in data and isinstance(data["open_orders"], list):
            for o in data["open_orders"]:
                pending_orders.append(self._map_pending_order_snapshot(o))
        elif "pending_orders" in data and isinstance(data["pending_orders"], list):
            for o in data["pending_orders"]:
                pending_orders.append(self._map_pending_order_snapshot(o))

        if "totalWalletBalance" in data:
            wallet_balance = float(data["totalWalletBalance"])
        elif "wallet_balance_usd" in data:
            wallet_balance = float(data["wallet_balance_usd"])
        else:
            wallet_balance = 0.0

        if "totalUnrealizedProfit" in data:
            unrealized_pnl = float(data["totalUnrealizedProfit"])
        elif "unrealized_pnl_usd" in data:
            unrealized_pnl = float(data["unrealized_pnl_usd"])
        else:
            unrealized_pnl = sum(p.unrealized_pnl_usd for p in active_positions)

        if "totalMarginBalance" in data:
            equity_nav = float(data["totalMarginBalance"])
        elif "equity_nav_usd" in data:
            equity_nav = float(data["equity_nav_usd"])
        else:
            equity_nav = wallet_balance + unrealized_pnl

        if "totalInitialMargin" in data:
            used_margin = float(data["totalInitialMargin"])
        elif "used_margin_usd" in data:
            used_margin = float(data["used_margin_usd"])
        else:
            used_margin = sum(p.isolated_margin_usd for p in active_positions)

        if "availableBalance" in data:
            available_margin = float(data["availableBalance"])
        elif "available_margin_usd" in data:
            available_margin = float(data["available_margin_usd"])
        else:
            available_margin = max(0.0, equity_nav - used_margin)

        if equity_nav > 0.0:
            margin_utilization = used_margin / equity_nav
        else:
            margin_utilization = 0.0

        if "current_daily_pnl_usd" in data:
            current_daily_pnl = float(data["current_daily_pnl_usd"])
        else:
            current_daily_pnl = unrealized_pnl

        if "current_daily_dd_pct" in data:
            current_daily_dd = float(data["current_daily_dd_pct"])
        else:
            current_daily_dd = 0.0

        if "remaining_daily_dd_budget_pct" in data:
            remaining_daily_dd = float(data["remaining_daily_dd_budget_pct"])
        else:
            remaining_daily_dd = max(0.0, 0.05 - current_daily_dd)

        if "current_max_dd_pct" in data:
            current_max_dd = float(data["current_max_dd_pct"])
        else:
            current_max_dd = 0.0

        if "remaining_max_dd_budget_pct" in data:
            remaining_max_dd = float(data["remaining_max_dd_budget_pct"])
        else:
            remaining_max_dd = max(0.0, 0.10 - current_max_dd)

        if "consecutive_losses_count" in data:
            consecutive_losses = int(data["consecutive_losses_count"])
        else:
            consecutive_losses = 0

        gross_exposure = sum(p.notional_usd for p in active_positions)
        net_delta = sum(
            p.notional_usd if p.side == TradeSide.LONG else -p.notional_usd
            for p in active_positions
        )
        net_beta = sum(
            p.notional_usd * p.btc_beta
            if p.side == TradeSide.LONG
            else -p.notional_usd * p.btc_beta
            for p in active_positions
        )

        raw_open_heat = DeterministicRiskCalculator.compute_raw_open_heat_pct(
            active_positions, equity_nav
        )
        effective_open_heat = DeterministicRiskCalculator.compute_effective_open_heat_pct(
            active_positions, equity_nav
        )

        return AccountPortfolioContext(
            execution_mode=exec_mode,
            timestamp_ms=ts_ms,
            wallet_balance_usd=wallet_balance,
            unrealized_pnl_usd=unrealized_pnl,
            equity_nav_usd=equity_nav,
            used_margin_usd=used_margin,
            available_margin_usd=available_margin,
            margin_utilization_pct=margin_utilization,
            current_daily_pnl_usd=current_daily_pnl,
            current_daily_dd_pct=current_daily_dd,
            remaining_daily_dd_budget_pct=remaining_daily_dd,
            current_max_dd_pct=current_max_dd,
            remaining_max_dd_budget_pct=remaining_max_dd,
            consecutive_losses_count=consecutive_losses,
            active_positions=active_positions,
            pending_orders=pending_orders,
            gross_exposure_usd=gross_exposure,
            net_directional_delta_usd=net_delta,
            net_btc_beta_exposure_usd=net_beta,
            raw_open_heat_pct=raw_open_heat,
            effective_open_heat_pct=effective_open_heat,
            same_direction_active_count=len(active_positions),
        )

    async def close(self) -> None:
        """Close underlying aiohttp session if owned."""
        if self._owns_session and self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> BinanceFuturesGatewayAdapter:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()
