"""Integration Tests: Database Connection Pool & Repositories against live Docker TimescaleDB."""

from __future__ import annotations

import os
import time
import uuid

import pytest

from trading_bot.adapters.persistence.account_portfolio_repo import AccountPortfolioRepository
from trading_bot.adapters.persistence.database import DatabaseManager
from trading_bot.adapters.persistence.market_data_repo import MarketDataRepository
from trading_bot.domain.entities.common.enums import ExecutionMode
from trading_bot.domain.entities.market.market_state import MicroBar1m
from trading_bot.domain.entities.risk.account_state import AccountPortfolioContext
from trading_bot.infrastructure.config_schemas import DatabaseConfig


def get_test_db_config() -> DatabaseConfig:
    return DatabaseConfig(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        db_name=os.getenv("POSTGRES_DB", "trading_bot_db"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        max_connections=int(os.getenv("POSTGRES_MAX_CONNECTIONS", "5")),
    )


@pytest.mark.asyncio
async def test_database_manager_lifecycle_and_healthcheck() -> None:
    cfg = get_test_db_config()
    db = DatabaseManager(cfg)
    await db.initialize()
    try:
        is_healthy = await db.healthcheck()
        assert is_healthy is True
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_database_transaction_commit_and_rollback() -> None:
    cfg = get_test_db_config()
    db = DatabaseManager(cfg)
    await db.initialize()
    try:
        order_id = str(uuid.uuid4())
        # Test rollback on exception
        with pytest.raises(RuntimeError):
            async with db.transaction() as conn:
                await conn.execute(
                    """
                    INSERT INTO orders_and_positions (
                        order_id, created_at, bar_time_30m, symbol, execution_mode,
                        playbook_used, side, entry_type, entry_price, stop_loss_price,
                        take_profit_price, vertical_barrier_time, notional_usd, leverage,
                        isolated_margin_usd, is_sl_at_breakeven, effective_at_risk_usd,
                        account_context_json, status, mae_pct, mfe_pct, agent_debate_json
                    ) VALUES (
                        $1::uuid, NOW(), NOW(), 'BTCUSDT', 'PAPER', 'PB_TEST', 'LONG',
                        'MARKET', 50000.0, 49000.0, 52000.0, NOW(), 5000.0, 5, 1000.0,
                        FALSE, 100.0, '{}'::jsonb, 'OPEN', 0.0, 0.0, '{}'::jsonb
                    );
                    """,
                    order_id,
                )
                raise RuntimeError("Intentional rollback trigger")

        row = await db.fetchrow(
            "SELECT * FROM orders_and_positions WHERE order_id = $1::uuid;",
            order_id,
        )
        assert row is None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_market_data_repo_save_and_fetch() -> None:
    cfg = get_test_db_config()
    db = DatabaseManager(cfg)
    await db.initialize()
    repo = MarketDataRepository(db)

    try:
        now_ms = int(time.time() * 1000)
        bars: list[MicroBar1m] = [
            MicroBar1m(
                timestamp_ms=now_ms - 120000,
                open=65000.0,
                high=65100.0,
                low=64950.0,
                close=65050.0,
                volume=12.5,
            ),
            MicroBar1m(
                timestamp_ms=now_ms - 60000,
                open=65050.0,
                high=65200.0,
                low=65000.0,
                close=65180.0,
                volume=15.0,
            ),
        ]

        await repo.save_klines_1m("BTCUSDT", bars)
        fetched = await repo.fetch_recent_klines_1m("BTCUSDT", 2)
        assert len(fetched) >= 2

        latest_price = await repo.get_latest_price("BTCUSDT")
        assert latest_price > 0.0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_account_portfolio_repository_assemble_context() -> None:
    cfg = get_test_db_config()
    db = DatabaseManager(cfg)
    await db.initialize()
    repo = AccountPortfolioRepository(db)
    market_repo = MarketDataRepository(db)

    try:
        now_ms = int(time.time() * 1000)
        # Create and persist a dummy equity snapshot
        dummy_context = AccountPortfolioContext(
            execution_mode=ExecutionMode.PAPER,
            timestamp_ms=now_ms,
            wallet_balance_usd=10000.0,
            unrealized_pnl_usd=150.0,
            equity_nav_usd=10150.0,
            used_margin_usd=2000.0,
            available_margin_usd=8150.0,
            margin_utilization_pct=0.197,
            current_daily_pnl_usd=150.0,
            current_daily_dd_pct=0.015,
            remaining_daily_dd_budget_pct=0.035,
            current_max_dd_pct=0.02,
            remaining_max_dd_budget_pct=0.08,
            consecutive_losses_count=1,
            active_positions=[],
            pending_orders=[],
            gross_exposure_usd=10000.0,
            net_directional_delta_usd=10000.0,
            net_btc_beta_exposure_usd=10000.0,
            raw_open_heat_pct=0.05,
            effective_open_heat_pct=0.05,
            same_direction_active_count=0,
        )
        await market_repo.save_account_equity_snapshot(dummy_context)

        # Assemble context
        assembled = await repo.assemble_context(
            execution_mode=ExecutionMode.PAPER,
            now_timestamp_ms=now_ms,
            default_wallet_balance_usd=10000.0,
            daily_dd_limit_pct=0.05,
            max_dd_limit_pct=0.10,
        )

        assert assembled.execution_mode == ExecutionMode.PAPER
        assert assembled.wallet_balance_usd == 10000.0
        assert assembled.current_daily_dd_pct == 0.015
        assert assembled.remaining_daily_dd_budget_pct == 0.035
        assert assembled.consecutive_losses_count == 1
    finally:
        await db.close()
