"""Unit tests for PerProviderRateLimitGovernor, RateLimitGovernorPort, and Invariant #9."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
from scripts.check_no_defaults_or_hardcode import ZeroDefaultAndHardcodeVisitor

from trading_bot.adapters.agents.rate_limit_governor import (
    ACQUIRE_ROUTE_LUA,
    GovernorPacerConfig,
    PerProviderRateLimitGovernor,
)
from trading_bot.ports.rate_limit_governor_port import (
    RateLimitGovernorPort,
    RouteQuotaSpec,
)


class MockRedisClient:
    """Mock Redis client for testing Rate Limit Governor (Zero Defaults)."""

    def __init__(self, eval_return_values: list[Any]) -> None:
        self.eval_return_values: list[Any] = eval_return_values
        self.eval_calls: list[tuple[Any, ...]] = []
        self.set_calls: list[dict[str, Any]] = []

    async def eval(self, script: str, numkeys: int, *args: Any) -> Any:
        self.eval_calls.append((script, numkeys, *args))
        if len(self.eval_return_values) > 0:
            return self.eval_return_values.pop(0)
        return [1, 0, "OK"]

    async def set(self, name: str, value: Any, px: int | None) -> bool:
        self.set_calls.append({"name": name, "value": value, "px": px})
        return True


def create_sample_spec(route_id: str, model_name: str) -> RouteQuotaSpec:
    """Helper to create RouteQuotaSpec with explicit parameters."""
    return RouteQuotaSpec(
        route_id=route_id,
        model_name=model_name,
        rpm_limit=15,
        tpm_limit=100000,
        rpd_limit=1500,
        min_interval_ms=1000,
        max_concurrency=2,
    )


def test_route_quota_spec_and_pacer_config_fields() -> None:
    """Validate RouteQuotaSpec and GovernorPacerConfig fields and constraints."""
    spec = create_sample_spec("route_1", "gemini-2.5-flash")
    assert spec.route_id == "route_1"
    assert spec.model_name == "gemini-2.5-flash"
    assert spec.rpm_limit == 15
    assert spec.tpm_limit == 100000
    assert spec.rpd_limit == 1500
    assert spec.min_interval_ms == 1000
    assert spec.max_concurrency == 2

    pacer = GovernorPacerConfig(
        initial_max_wait_ms=2000,
        min_poll_step_ms=100,
    )
    assert pacer.initial_max_wait_ms == 2000
    assert pacer.min_poll_step_ms == 100


def test_lua_script_integrity() -> None:
    """Ensure ACQUIRE_ROUTE_LUA contains expected atomic rate limiting steps."""
    assert 'local cooldown_key = "rl:cooldown:" .. route_id' in ACQUIRE_ROUTE_LUA
    assert 'local last_ts_key  = "rl:last_ts:" .. route_id' in ACQUIRE_ROUTE_LUA
    assert "COOLDOWN_ACTIVE" in ACQUIRE_ROUTE_LUA
    assert "MIN_INTERVAL_WAIT" in ACQUIRE_ROUTE_LUA
    assert "RPD_EXHAUSTED" in ACQUIRE_ROUTE_LUA
    assert "RPM_EXHAUSTED" in ACQUIRE_ROUTE_LUA
    assert "TPM_EXHAUSTED" in ACQUIRE_ROUTE_LUA
    assert 'return {1, 0, "OK"}' in ACQUIRE_ROUTE_LUA


@pytest.mark.asyncio
async def test_acquire_available_route_success() -> None:
    """Validate successful route acquisition on first pass."""
    spec1 = create_sample_spec("route_1", "model_1")
    spec2 = create_sample_spec("route_2", "model_2")
    tier_routes = {"fast_tactical": [spec1, spec2]}
    pacer = GovernorPacerConfig(initial_max_wait_ms=500, min_poll_step_ms=50)

    # Redis returns [1, 0, "OK"]
    mock_redis = MockRedisClient([[1, 0, "OK"]])
    governor = PerProviderRateLimitGovernor(
        mock_redis,  # pyright: ignore[reportArgumentType]
        tier_routes,
        pacer,
    )

    acquired = await governor.acquire_available_route("fast_tactical", 1500)
    assert acquired.route_id == "route_1"
    assert len(mock_redis.eval_calls) == 1
    # Check that cursor progressed to index 1
    assert governor.rr_cursor["fast_tactical"] == 1


@pytest.mark.asyncio
async def test_round_robin_cursor_progression() -> None:
    """Validate round-robin cursor across multiple routes in a tier."""
    spec_a = create_sample_spec("route_a", "model_a")
    spec_b = create_sample_spec("route_b", "model_b")
    spec_c = create_sample_spec("route_c", "model_c")
    tier_routes = {"fast_tactical": [spec_a, spec_b, spec_c]}
    pacer = GovernorPacerConfig(initial_max_wait_ms=500, min_poll_step_ms=50)

    # All three requests succeed immediately
    mock_redis = MockRedisClient(
        [
            [1, 0, "OK"],
            [1, 0, "OK"],
            [1, 0, "OK"],
            [1, 0, "OK"],
        ]
    )
    governor = PerProviderRateLimitGovernor(
        mock_redis,  # pyright: ignore[reportArgumentType]
        tier_routes,
        pacer,
    )

    # Call 1 -> route_a, cursor becomes 1
    r1 = await governor.acquire_available_route("fast_tactical", 1000)
    assert r1.route_id == "route_a"
    assert governor.rr_cursor["fast_tactical"] == 1

    # Call 2 -> route_b, cursor becomes 2
    r2 = await governor.acquire_available_route("fast_tactical", 1000)
    assert r2.route_id == "route_b"
    assert governor.rr_cursor["fast_tactical"] == 2

    # Call 3 -> route_c, cursor wraps around to 0
    r3 = await governor.acquire_available_route("fast_tactical", 1000)
    assert r3.route_id == "route_c"
    assert governor.rr_cursor["fast_tactical"] == 0

    # Call 4 -> route_a again, cursor becomes 1
    r4 = await governor.acquire_available_route("fast_tactical", 1000)
    assert r4.route_id == "route_a"
    assert governor.rr_cursor["fast_tactical"] == 1


@pytest.mark.asyncio
async def test_skip_route_on_cooldown_or_exhaustion() -> None:
    """Test skipping route on cooldown and taking the next available route immediately."""
    spec_a = create_sample_spec("route_a", "model_a")
    spec_b = create_sample_spec("route_b", "model_b")
    tier_routes = {"deep_reasoning": [spec_a, spec_b]}
    pacer = GovernorPacerConfig(initial_max_wait_ms=500, min_poll_step_ms=50)

    # route_a is on COOLDOWN, route_b is OK
    mock_redis = MockRedisClient(
        [
            [0, 25000, "COOLDOWN_ACTIVE"],
            [1, 0, "OK"],
        ]
    )
    governor = PerProviderRateLimitGovernor(
        mock_redis,  # pyright: ignore[reportArgumentType]
        tier_routes,
        pacer,
    )

    acquired = await governor.acquire_available_route("deep_reasoning", 2000)
    assert acquired.route_id == "route_b"
    assert len(mock_redis.eval_calls) == 2
    # Cursor now set to (1 + 1) % 2 = 0
    assert governor.rr_cursor["deep_reasoning"] == 0


@pytest.mark.asyncio
async def test_quota_exhaustion_pacing_and_subsequent_success() -> None:
    """Test pacing when all routes initially exhausted, succeeding on retry."""
    spec_a = create_sample_spec("route_a", "model_a")
    tier_routes = {"fast_tactical": [spec_a]}
    # Minimal poll step and wait to speed up unit test
    pacer = GovernorPacerConfig(initial_max_wait_ms=10, min_poll_step_ms=5)

    # First attempt: RPM exhausted; Second attempt: OK
    mock_redis = MockRedisClient(
        [
            [0, 10, "RPM_EXHAUSTED"],
            [1, 0, "OK"],
        ]
    )
    governor = PerProviderRateLimitGovernor(
        mock_redis,  # pyright: ignore[reportArgumentType]
        tier_routes,
        pacer,
    )

    acquired = await governor.acquire_available_route("fast_tactical", 1000)
    assert acquired.route_id == "route_a"
    assert len(mock_redis.eval_calls) == 2


@pytest.mark.asyncio
async def test_mark_route_cooldown() -> None:
    """Test mark_route_cooldown sets TTL key in Redis."""
    spec = create_sample_spec("route_cooldown_test", "model_x")
    tier_routes = {"tier_x": [spec]}
    pacer = GovernorPacerConfig(initial_max_wait_ms=100, min_poll_step_ms=10)

    mock_redis = MockRedisClient([])
    governor = PerProviderRateLimitGovernor(
        mock_redis,  # pyright: ignore[reportArgumentType]
        tier_routes,
        pacer,
    )

    await governor.mark_route_cooldown("route_cooldown_test", 30.0)

    assert len(mock_redis.set_calls) == 1
    call = mock_redis.set_calls[0]
    assert call["name"] == "rl:cooldown:route_cooldown_test"
    assert call["px"] == 30000


@pytest.mark.asyncio
async def test_governor_invalid_tier_and_empty_routes() -> None:
    """Test error handling when tier is not configured or has zero routes."""
    pacer = GovernorPacerConfig(initial_max_wait_ms=100, min_poll_step_ms=10)
    mock_redis = MockRedisClient([])

    governor = PerProviderRateLimitGovernor(
        mock_redis,  # pyright: ignore[reportArgumentType]
        {"empty_tier": []},
        pacer,
    )

    with pytest.raises(KeyError, match="not configured"):
        await governor.acquire_available_route("non_existent_tier", 1000)

    with pytest.raises(ValueError, match="No routes available"):
        await governor.acquire_available_route("empty_tier", 1000)


def test_port_protocol_conformance() -> None:
    """Verify PerProviderRateLimitGovernor implements RateLimitGovernorPort Protocol."""
    spec = create_sample_spec("route_proto", "model_p")
    pacer = GovernorPacerConfig(initial_max_wait_ms=100, min_poll_step_ms=10)
    mock_redis = MockRedisClient([])

    governor = PerProviderRateLimitGovernor(
        mock_redis,  # pyright: ignore[reportArgumentType]
        {"fast": [spec]},
        pacer,
    )

    assert isinstance(governor, RateLimitGovernorPort)


def test_ast_compliance_zero_defaults_zero_fallbacks() -> None:
    """Validate rate limit governor files strictly comply with Invariant #9."""
    repo_root = Path(__file__).resolve().parent.parent.parent

    target_files = [
        repo_root / "src" / "trading_bot" / "ports" / "rate_limit_governor_port.py",
        repo_root / "src" / "trading_bot" / "adapters" / "agents" / "rate_limit_governor.py",
        Path(__file__).resolve(),
    ]

    all_violations: list[str] = []
    for f in target_files:
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        visitor = ZeroDefaultAndHardcodeVisitor(f)
        visitor.visit(tree)
        all_violations.extend(visitor.violations)

    assert not all_violations, "Invariant #9 violations found:\n" + "\n".join(all_violations)
