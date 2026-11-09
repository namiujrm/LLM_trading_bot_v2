"""Standalone Verification Script for 9Router and Binance Live Connections."""

from __future__ import annotations

import asyncio
import os

import aiohttp
from dotenv import load_dotenv

from trading_bot.adapters.binance.binance_futures_gateway import BinanceFuturesGatewayAdapter
from trading_bot.infrastructure.config_schemas import BinanceConfig

load_dotenv()


async def verify_9router(session: aiohttp.ClientSession) -> bool:
    """Verify 9Router endpoints and test inference."""
    nr_url = os.getenv("NINEROUTER_BASE_URL", "http://localhost:20128/v1").rstrip("/")
    nr_key = os.getenv("NINEROUTER_API_KEY", "")

    print(f"[1/2] Testing 9Router Gateway at {nr_url}...")
    headers = {
        "Authorization": f"Bearer {nr_key}",
        "Content-Type": "application/json",
    }

    try:
        async with session.get(
            f"{nr_url}/models", headers=headers, timeout=aiohttp.ClientTimeout(total=10.0)
        ) as resp:
            if resp.status != 200:
                err_text = await resp.text()
                print(f"  [FAIL] 9Router /models failed [HTTP {resp.status}]: {err_text}")
                return False

            data = await resp.json()
            models_list = [m["id"] for m in data.get("data", [])]
            print(f"  [OK] 9Router Connected! {len(models_list)} models available.")
            print(f"       Registered models preview: {models_list[:5]}")

        # Display configured tiers
        fast_tier = os.getenv("MODEL_FAST_TIER", "ds/deepseek-v4-flash")
        deep_tier = os.getenv("MODEL_DEEP_TIER", "ds/deepseek-v4-pro")
        print(f"       Configured Fast Tier:     {fast_tier}")
        print(f"       Configured Deep Tier:     {deep_tier}")

        # Test simple chat completion
        test_model = fast_tier
        print(f"  Testing chat inference with model: '{test_model}'...")
        payload = {
            "model": test_model,
            "messages": [{"role": "user", "content": "Ping test! Reply with PONG."}],
            "max_tokens": 15,
            "stream": False,
        }
        async with session.post(
            f"{nr_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30.0),
        ) as chat_resp:
            if chat_resp.status != 200:
                err_text = await chat_resp.text()
                print(f"  [FAIL] Inference failed [HTTP {chat_resp.status}]: {err_text}")
                return False

            chat_json = await chat_resp.json(content_type=None)
            msg = chat_json["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning_content") or ""
            usage = chat_json.get("usage", {})
            print(f'  [OK] Inference SUCCESS! Response: "{content.strip()[:60]}"')
            print(
                f"       Tokens used: prompt={usage.get('prompt_tokens', 0)}, completion={usage.get('completion_tokens', 0)}"
            )
            return True

    except Exception as exc:
        print(f"  [FAIL] 9Router connection exception: {exc}")
        return False


async def verify_binance(session: aiohttp.ClientSession) -> bool:
    """Verify Binance Futures REST API authentication and account context."""
    print("[2/2] Testing Binance Futures Gateway...")
    b_key = os.getenv("BINANCE_API_KEY", "")
    b_sec = os.getenv("BINANCE_API_SECRET", "")
    testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

    if not b_key or not b_sec:
        print("  [FAIL] Missing BINANCE_API_KEY or BINANCE_API_SECRET in .env")
        return False

    binance_cfg = BinanceConfig(
        api_key=b_key,
        api_secret=b_sec,
        testnet=testnet,
        recv_window_ms=5000,
    )

    gateway = BinanceFuturesGatewayAdapter(config=binance_cfg, session=session)
    try:
        context = await gateway.fetch_account_portfolio_context()
        print("  [OK] Binance Futures Gateway Connected & Authenticated!")
        print(f"       Execution Mode:        {context.execution_mode.value.upper()}")
        print(f"       Base URL:              {gateway.base_url}")
        print(f"       Wallet Balance:        ${context.wallet_balance_usd:,.2f} USDT")
        print(f"       Equity NAV:            ${context.equity_nav_usd:,.2f} USDT")
        print(f"       Available Margin:      ${context.available_margin_usd:,.2f} USDT")
        print(f"       Used Margin:           ${context.used_margin_usd:,.2f} USDT")
        print(f"       Active Open Positions: {len(context.active_positions)}")
        print(f"       Pending Limit Orders:  {len(context.pending_orders)}")
        return True
    except Exception as exc:
        print(f"  [FAIL] Binance gateway exception: {exc}")
        return False


async def main() -> None:
    print("================================================================")
    print("               LIVE TRADING BOT CONNECTION AUDIT                ")
    print("================================================================")
    async with aiohttp.ClientSession() as session:
        nr_ok = await verify_9router(session)
        print("----------------------------------------------------------------")
        bn_ok = await verify_binance(session)

    print("================================================================")
    if nr_ok and bn_ok:
        print("[SUCCESS] ALL SYSTEMS READY: 9Router and Binance Live Connections Verified 100%!")
    else:
        print("[WARNING] One or more systems failed connection. Please inspect logs above.")
    print("================================================================")


if __name__ == "__main__":
    asyncio.run(main())
