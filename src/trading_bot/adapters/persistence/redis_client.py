"""Redis Connection Adapter and Rate Governor — Zero Defaults."""

from __future__ import annotations

from typing import Any

import redis.asyncio as aioredis

from trading_bot.domain.entities.account_state import AccountPortfolioContext
from trading_bot.domain.entities.enums import ExecutionMode
from trading_bot.infrastructure.config_schemas import RedisConfig

ACQUIRE_ROUTE_LUA = """
local route_id = ARGV[1]
local now_ms = tonumber(ARGV[2])
local est_tokens = tonumber(ARGV[3])
local rpm_limit = tonumber(ARGV[4])
local tpm_limit = tonumber(ARGV[5])
local rpd_limit = tonumber(ARGV[6])
local min_interval_ms = tonumber(ARGV[7])

local cooldown_key = "rl:cooldown:" .. route_id
local last_ts_key  = "rl:last_ts:" .. route_id
local rpm_key      = "rl:rpm:" .. route_id .. ":" .. math.floor(now_ms / 60000)
local tpm_key      = "rl:tpm:" .. route_id .. ":" .. math.floor(now_ms / 60000)
local rpd_key      = "rl:rpd:" .. route_id .. ":" .. math.floor(now_ms / 86400000)

-- 1. Check Cooldown
local cooldown_until = tonumber(redis.call("GET", cooldown_key) or "0")
if now_ms < cooldown_until then
    return {0, cooldown_until - now_ms, "COOLDOWN_ACTIVE"}
end

-- 2. Check minimum interval between requests
local last_ts = tonumber(redis.call("GET", last_ts_key) or "0")
if (now_ms - last_ts) < min_interval_ms then
    return {0, min_interval_ms - (now_ms - last_ts), "MIN_INTERVAL_WAIT"}
end

-- 3. Check RPD, RPM and TPM limits
local cur_rpd = tonumber(redis.call("GET", rpd_key) or "0")
if cur_rpd + 1 > rpd_limit then
    return {0, 60000, "RPD_EXHAUSTED"}
end

local cur_rpm = tonumber(redis.call("GET", rpm_key) or "0")
if cur_rpm + 1 > rpm_limit then
    return {0, 60000 - (now_ms % 60000), "RPM_EXHAUSTED"}
end

local cur_tpm = tonumber(redis.call("GET", tpm_key) or "0")
if cur_tpm + est_tokens > tpm_limit then
    return {0, 60000 - (now_ms % 60000), "TPM_EXHAUSTED"}
end

-- 4. Allocation success: increment counters
redis.call("SET", last_ts_key, tostring(now_ms), "PX", 120000)
redis.call("INCR", rpm_key)
redis.call("EXPIRE", rpm_key, 120)
redis.call("INCRBY", tpm_key, est_tokens)
redis.call("EXPIRE", tpm_key, 120)
redis.call("INCR", rpd_key)
redis.call("EXPIRE", rpd_key, 172800)

return {1, 0, "OK"}
"""


class RedisClient:
    """Async Redis client wrapper with rate limiting Lua and live portfolio context caching."""

    def __init__(self, config: RedisConfig) -> None:
        self._config: RedisConfig = config
        self._client: aioredis.Redis | None = None
        self._lua_script_sha: str | None = None

    async def initialize(self) -> None:
        """Connect to Redis instance and register Lua scripts."""
        self._client = aioredis.Redis(
            host=self._config.host,
            port=self._config.port,
            db=self._config.db,
            password=self._config.password,
            ssl=self._config.ssl,
            decode_responses=True,
        )
        # Register Lua script SHA for high-speed atomic calls
        self._lua_script_sha = str(await self._client.script_load(ACQUIRE_ROUTE_LUA))

    async def close(self) -> None:
        """Close Redis connection pool."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._lua_script_sha = None

    def get_client(self) -> aioredis.Redis:
        """Return raw active redis client or raise RuntimeError."""
        if self._client is None:
            raise RuntimeError("RedisClient has not been initialized. Call initialize() first.")
        return self._client

    async def ping(self) -> bool:
        """Check Redis connectivity."""
        client = self.get_client()
        result: Any = await client.ping()  # pyright: ignore[reportUnknownMemberType]
        return bool(result)

    async def acquire_route_permit(
        self,
        route_id: str,
        now_ms: int,
        est_tokens: int,
        rpm_limit: int,
        tpm_limit: int,
        rpd_limit: int,
        min_interval_ms: int,
    ) -> tuple[bool, int, str]:
        """Atomically evaluate rate limit quotas and cooldowns via Lua."""
        client = self.get_client()
        args: list[Any] = [
            route_id,
            str(now_ms),
            str(est_tokens),
            str(rpm_limit),
            str(tpm_limit),
            str(rpd_limit),
            str(min_interval_ms),
        ]
        if self._lua_script_sha is not None:
            raw_result = await client.evalsha(self._lua_script_sha, 0, *args)
        else:
            raw_result = await client.eval(ACQUIRE_ROUTE_LUA, 0, *args)

        result_list = list(raw_result)
        allowed = int(result_list[0]) == 1
        wait_ms = int(result_list[1])
        reason = str(result_list[2])
        return (allowed, wait_ms, reason)

    async def set_live_account_context(
        self,
        context: AccountPortfolioContext,
        ttl_seconds: int,
    ) -> None:
        """Store serialized AccountPortfolioContext snapshot in Redis with explicit TTL."""
        client = self.get_client()
        key = f"live_account_context:{context.execution_mode.value}"
        payload_json = context.model_dump_json()
        await client.set(key, payload_json, ex=ttl_seconds)

    async def get_live_account_context(
        self,
        execution_mode: ExecutionMode,
    ) -> AccountPortfolioContext | None:
        """Fetch and deserialize live AccountPortfolioContext from Redis, or return None."""
        client = self.get_client()
        key = f"live_account_context:{execution_mode.value}"
        raw_val = await client.get(key)
        if raw_val is None:
            return None
        return AccountPortfolioContext.model_validate_json(raw_val)
