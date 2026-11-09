"""Per-Provider Redis Rate-Limit Governor — Controls RPM, TPM, RPD & Min-Interval for each API.

Enforces Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, cast

import redis.asyncio as redis
from pydantic import BaseModel

from trading_bot.ports.rate_limit_governor_port import (
    RateLimitGovernorPort,
    RouteQuotaSpec,
)

# Atomic Lua script executed on Redis to evaluate and allocate API quota
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

-- 1. Kiểm tra xem API này có đang bị phạt Cooldown (do 429 trước đó) không
local cooldown_until = tonumber(redis.call("GET", cooldown_key) or "0")
if now_ms < cooldown_until then
    return {0, cooldown_until - now_ms, "COOLDOWN_ACTIVE"}
end

-- 2. Kiểm tra khoảng cách tối thiểu giữa 2 lần gọi liên tiếp (min_interval_ms)
local last_ts = tonumber(redis.call("GET", last_ts_key) or "0")
if (now_ms - last_ts) < min_interval_ms then
    return {0, min_interval_ms - (now_ms - last_ts), "MIN_INTERVAL_WAIT"}
end

-- 3. Kiểm tra hạn mức RPD (Requests Per Day), RPM (Per Minute) và TPM (Tokens Per Minute)
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

-- 4. Cấp phát thành công -> Cập nhật bộ đếm nguyên tử
redis.call("SET", last_ts_key, tostring(now_ms), "PX", 120000)
redis.call("INCR", rpm_key)
redis.call("EXPIRE", rpm_key, 120)
redis.call("INCRBY", tpm_key, est_tokens)
redis.call("EXPIRE", tpm_key, 120)
redis.call("INCR", rpd_key)
redis.call("EXPIRE", rpd_key, 172800)

return {1, 0, "OK"}
"""


class GovernorPacerConfig(BaseModel):
    """Pacing configuration for Rate Limit Governor backoff and polling (Zero Defaults)."""

    initial_max_wait_ms: int
    min_poll_step_ms: int


class PerProviderRateLimitGovernor(RateLimitGovernorPort):
    """Redis-backed atomic rate limit governor allocating routes across model tiers."""

    def __init__(
        self,
        redis_client: redis.Redis,
        tier_routes: dict[str, list[RouteQuotaSpec]],
        pacer_cfg: GovernorPacerConfig,
    ) -> None:
        self.redis: redis.Redis = redis_client
        self.tier_routes: dict[str, list[RouteQuotaSpec]] = tier_routes
        self.pacer_cfg: GovernorPacerConfig = pacer_cfg
        self._rr_cursor: dict[str, int] = dict.fromkeys(tier_routes, 0)

    @property
    def rr_cursor(self) -> dict[str, int]:
        """Return a copy of the round-robin cursor positions for inspection."""
        return dict(self._rr_cursor)

    async def acquire_available_route(self, tier: str, estimated_tokens: int) -> RouteQuotaSpec:
        """Acquire an available route within the requested tier or pace until available."""
        if tier not in self.tier_routes:
            raise KeyError(f"Tier '{tier}' is not configured in rate limit governor.")
        routes: list[RouteQuotaSpec] = self.tier_routes[tier]
        n: int = len(routes)
        if n == 0:
            raise ValueError(f"No routes available in tier '{tier}'.")

        while True:
            min_wait_ms: int = self.pacer_cfg.initial_max_wait_ms
            if tier not in self._rr_cursor:
                self._rr_cursor[tier] = 0
            start_idx: int = self._rr_cursor[tier]

            for offset in range(n):
                idx: int = (start_idx + offset) % n
                spec: RouteQuotaSpec = routes[idx]
                now_ms: int = int(time.time() * 1000)
                eval_res = await self.redis.eval(
                    ACQUIRE_ROUTE_LUA,
                    0,
                    spec.route_id,
                    str(now_ms),
                    str(estimated_tokens),
                    str(spec.rpm_limit),
                    str(spec.tpm_limit),
                    str(spec.rpd_limit),
                    str(spec.min_interval_ms),
                )
                res = cast(list[Any], eval_res)
                allowed: int = int(res[0])
                wait_ms: int = int(res[1])
                if allowed == 1:
                    self._rr_cursor[tier] = (idx + 1) % n
                    return spec
                pacer_step: int = max(self.pacer_cfg.min_poll_step_ms, wait_ms)
                min_wait_ms = min(min_wait_ms, pacer_step)

            await asyncio.sleep(min_wait_ms / 1000.0)

    async def mark_route_cooldown(self, route_id: str, cooldown_seconds: float) -> None:
        """Set cooldown timestamp in Redis for a specific route."""
        until_ms: int = int((time.time() + cooldown_seconds) * 1000)
        cooldown_key: str = f"rl:cooldown:{route_id}"
        ttl_px: int = max(1, int(cooldown_seconds * 1000))
        await self.redis.set(cooldown_key, str(until_ms), px=ttl_px)
