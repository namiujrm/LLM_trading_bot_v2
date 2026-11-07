"""Database Connection Pool Adapter using asyncpg — Zero Defaults."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import asyncpg

from trading_bot.infrastructure.config_schemas import DatabaseConfig


class DatabaseManager:
    """Manages asyncpg connection pool with explicit configuration injection."""

    def __init__(self, config: DatabaseConfig) -> None:
        self._config: DatabaseConfig = config
        self._pool: asyncpg.Pool[asyncpg.Record] | None = None

    async def initialize(self) -> None:
        """Initialize the asyncpg connection pool."""
        min_conns = 1
        if self._config.max_connections < min_conns:
            min_conns = self._config.max_connections

        self._pool = await asyncpg.create_pool(
            host=self._config.host,
            port=self._config.port,
            user=self._config.user,
            password=self._config.password,
            database=self._config.db_name,
            min_size=min_conns,
            max_size=self._config.max_connections,
        )

    async def close(self) -> None:
        """Gracefully terminate all connections in the pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def get_pool(self) -> asyncpg.Pool[asyncpg.Record]:
        """Return the active connection pool or raise RuntimeError."""
        if self._pool is None:
            raise RuntimeError(
                "DatabaseManager pool has not been initialized. Call initialize() first."
            )
        return self._pool

    async def healthcheck(self) -> bool:
        """Verify database connectivity via a lightweight ping query."""
        pool = self.get_pool()
        async with pool.acquire() as conn:
            val = await conn.fetchval("SELECT 1")
            return val == 1

    @asynccontextmanager
    async def connection(
        self,
    ) -> AsyncGenerator[asyncpg.pool.PoolConnectionProxy[asyncpg.Record], None]:
        """Context manager for acquiring a single connection from the pool."""
        pool = self.get_pool()
        async with pool.acquire() as conn:
            yield conn

    @asynccontextmanager
    async def transaction(
        self,
    ) -> AsyncGenerator[asyncpg.pool.PoolConnectionProxy[asyncpg.Record], None]:
        """Context manager executing block inside an atomic transaction."""
        pool = self.get_pool()
        async with pool.acquire() as conn, conn.transaction():
            yield conn

    async def execute(self, query: str, *args: Any) -> str:
        """Execute a query without returning result rows."""
        pool = self.get_pool()
        async with pool.acquire() as conn:
            return str(await conn.execute(query, *args))

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        """Fetch multiple rows matching the query."""
        pool = self.get_pool()
        async with pool.acquire() as conn:
            records = await conn.fetch(query, *args)
            return list(records)

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        """Fetch a single row or None if not found."""
        pool = self.get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        """Fetch a scalar value or None."""
        pool = self.get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def copy_records_to_table(
        self,
        table_name: str,
        records: list[tuple[Any, ...]],
        columns: list[str],
    ) -> str:
        """Bulk binary copy into TimescaleDB/PostgreSQL table."""
        pool = self.get_pool()
        async with pool.acquire() as conn:
            result = await conn.copy_records_to_table(
                table_name,
                records=records,
                columns=columns,
            )
            return str(result)
