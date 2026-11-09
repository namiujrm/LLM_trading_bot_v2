"""Integration Tests: RedisClient & Lua Rate Limiting against live Docker Redis container."""

from __future__ import annotations

import os
import time

import pytest

from trading_bot.adapters.persistence.redis_client import RedisClient
from trading_bot.domain.entities.common.enums import ExecutionMode
from trading_bot.domain.entities.risk.account_state import AccountPortfolioContext
from trading_bot.infrastructure.config_schemas import RedisConfig


def get_test_redis_config() -> RedisConfig:
    return RedisConfig(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_DB", "0")),
        password=os.getenv("REDIS_PASSWORD", ""),
        ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
    )


@pytest.mark.asyncio
async def test_redis_client_lifecycle_and_ping() -> None:
    cfg = get_test_redis_config()
    client = RedisClient(cfg)
    await client.initialize()
    try:
        is_alive = await client.ping()
        assert is_alive is True
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_redis_client_acquire_route_permit_lua() -> None:
    cfg = get_test_redis_config()
    client = RedisClient(cfg)
    await client.initialize()
    try:
        route_id = f"test_route_{int(time.time() * 1000)}"
        now_ms = int(time.time() * 1000)

        # 1. First request should succeed
        allowed, wait_ms, reason = await client.acquire_route_permit(
            route_id=route_id,
            now_ms=now_ms,
            est_tokens=500,
            rpm_limit=5,
            tpm_limit=5000,
            rpd_limit=100,
            min_interval_ms=1000,
        )
        assert allowed is True
        assert wait_ms == 0
        assert reason == "OK"

        # 2. Immediate second request within 1000ms should trigger MIN_INTERVAL_WAIT
        allowed2, wait_ms2, reason2 = await client.acquire_route_permit(
            route_id=route_id,
            now_ms=now_ms + 100,
            est_tokens=500,
            rpm_limit=5,
            tpm_limit=5000,
            rpd_limit=100,
            min_interval_ms=1000,
        )
        assert allowed2 is False
        assert wait_ms2 > 0
        assert reason2 == "MIN_INTERVAL_WAIT"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_redis_client_live_account_context_caching() -> None:
    cfg = get_test_redis_config()
    client = RedisClient(cfg)
    await client.initialize()
    try:
        now_ms = int(time.time() * 1000)
        context = AccountPortfolioContext(
            execution_mode=ExecutionMode.PAPER,
            timestamp_ms=now_ms,
            wallet_balance_usd=25000.0,
            unrealized_pnl_usd=-120.0,
            equity_nav_usd=24880.0,
            used_margin_usd=3000.0,
            available_margin_usd=21880.0,
            margin_utilization_pct=0.1205,
            current_daily_pnl_usd=-120.0,
            current_daily_dd_pct=0.0048,
            remaining_daily_dd_budget_pct=0.0452,
            current_max_dd_pct=0.012,
            remaining_max_dd_budget_pct=0.088,
            consecutive_losses_count=2,
            active_positions=[],
            pending_orders=[],
            gross_exposure_usd=15000.0,
            net_directional_delta_usd=-5000.0,
            net_btc_beta_exposure_usd=-6000.0,
            raw_open_heat_pct=0.06,
            effective_open_heat_pct=0.04,
            same_direction_active_count=1,
        )

        await client.set_live_account_context(context, 60)
        cached = await client.get_live_account_context(ExecutionMode.PAPER)

        assert cached is not None
        assert cached.execution_mode == ExecutionMode.PAPER
        assert cached.wallet_balance_usd == 25000.0
        assert cached.equity_nav_usd == 24880.0
        assert cached.remaining_daily_dd_budget_pct == 0.0452
        assert cached.consecutive_losses_count == 2

        # Non-existent execution mode returns None
        non_existent = await client.get_live_account_context(ExecutionMode.LIVE)
        assert non_existent is None
    finally:
        await client.close()
