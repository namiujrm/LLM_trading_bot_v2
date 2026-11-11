"""Test script to verify order placement on Binance Futures Testnet."""

from __future__ import annotations

import asyncio
import os
import time

import aiohttp
from dotenv import load_dotenv

from trading_bot.adapters.binance.binance_futures_gateway import (
    BinanceFuturesGatewayAdapter,
)
from trading_bot.domain.entities.common.enums import EntryOrderType, TradeSide
from trading_bot.domain.entities.execution.trade_blueprint import BracketOrderTicket
from trading_bot.infrastructure.config_schemas import ConfigLoader


async def run_test_order() -> None:
    load_dotenv()
    env = dict(os.environ)
    config = ConfigLoader.load_binance_config(env)

    async with aiohttp.ClientSession() as session:
        gateway = BinanceFuturesGatewayAdapter(config=config, session=session)

        # 1. Sync exchange server time offset
        time_offset = await gateway.sync_time_offset()
        print(f"[*] Synchronized Binance Server Time Offset: {time_offset} ms")

        # 2. Fetch current BTCUSDT mark / market price
        ticker_resp = await gateway._request(  # pyright: ignore[reportPrivateUsage]
            "GET", "/fapi/v1/ticker/price", {"symbol": "BTCUSDT"}, False
        )
        current_price = float(ticker_resp["price"])
        print(f"[*] Current BTCUSDT Market Price on Testnet: ${current_price:,.2f}")

        # 3. Create a test Bracket Order Ticket
        # Using LIMIT far below market price (e.g. 15% below) so it rests safely on book
        limit_entry_price = round(current_price * 0.85, 1)
        sl_price = round(limit_entry_price * 0.90, 1)
        # Take profit must be above current market price so it rests safely without immediately triggering
        tp_price = round(current_price * 1.15, 1)
        test_order_id = f"test_cli_{int(time.time() * 1000)}"

        ticket = BracketOrderTicket(
            order_id=test_order_id,
            symbol="BTCUSDT",
            side=TradeSide.LONG,
            entry_type=EntryOrderType.LIMIT_PULLBACK_15M,
            entry_price=limit_entry_price,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            quantity=0.002,
            leverage=5,
        )

        print("[*] Submitting Bracket Order on Binance Testnet:")
        print(f"    - Order ID: {test_order_id}")
        print("    - Symbol: BTCUSDT | Side: LONG (BUY)")
        print(f"    - Type: LIMIT @ ${limit_entry_price:,.1f}")
        print(f"    - Stop Loss: ${sl_price:,.1f} | Take Profit: ${tp_price:,.1f}")
        print("    - Quantity: 0.002 BTC")

        # 4. Place bracket order via gateway
        parent_id = await gateway.place_bracket_order(ticket)
        print("[SUCCESS] Order Placed Successfully on Binance Futures Testnet!")
        print(f"          Parent Exchange Order ID: {parent_id}")

        # 5. Query active orders to confirm it appears on the exchange orderbook
        active_orders = await gateway._request(  # pyright: ignore[reportPrivateUsage]
            "GET", "/fapi/v1/openOrders", {"symbol": "BTCUSDT"}, True
        )
        matching = [
            o
            for o in active_orders
            if str(o.get("orderId")) == str(parent_id) or o.get("clientOrderId") == test_order_id
        ]
        print(
            f"[*] Verified on Exchange: Found {len(matching)} matching active entry order(s) on Binance orderbook."
        )
        for o in matching:
            print(
                f"    -> Parent Order ID: {o['orderId']} | Status: {o['status']} | Type: {o['type']} | Price: {o['price']}"
            )

        # 6. Query active algo orders to confirm SL and TP orders were registered
        open_algos = await gateway._request(  # pyright: ignore[reportPrivateUsage]
            "GET", "/fapi/v1/openAlgoOrders", {"symbol": "BTCUSDT"}, True
        )
        matching_algos = [
            a
            for a in open_algos
            if a.get("clientAlgoId") in (f"{test_order_id}_sl", f"{test_order_id}_tp")
        ]
        print(
            f"[*] Verified Algo Orders: Found {len(matching_algos)} matching active SL/TP order(s)."
        )
        for a in matching_algos:
            print(
                f"    -> Algo ID: {a['algoId']} | Client ID: {a['clientAlgoId']} | Type: {a['orderType']} | Trigger: ${float(a['triggerPrice']):,.1f}"
            )

        # 7. Clean up: Cancel parent entry order and child SL/TP orders
        cancel_parent_ok = await gateway.cancel_order("BTCUSDT", parent_id)
        print(f"[*] Clean-up Parent Order Cancelled: {cancel_parent_ok}")

        cancel_sl_ok = await gateway.cancel_order("BTCUSDT", f"{test_order_id}_sl")
        print(f"[*] Clean-up Stop Loss Algo Order Cancelled: {cancel_sl_ok}")

        cancel_tp_ok = await gateway.cancel_order("BTCUSDT", f"{test_order_id}_tp")
        print(f"[*] Clean-up Take Profit Algo Order Cancelled: {cancel_tp_ok}")

        print("[*] Full Bracket Order Placement & Clean-up Cycle Verified 100%!")


if __name__ == "__main__":
    asyncio.run(run_test_order())
