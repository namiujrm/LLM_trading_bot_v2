"""Unit tests for NinerouterClient, resilience, fallback, and Invariant #9 compliance."""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from scripts.check_no_defaults_or_hardcode import ZeroDefaultAndHardcodeVisitor

from trading_bot.adapters.agents.ninerouter_client import (
    NinerouterClient,
    NinerouterResponse,
)
from trading_bot.infrastructure.config_schemas import NinerouterConfig
from trading_bot.ports.rate_limit_governor_port import (
    RouteQuotaSpec,
)


class MockResponse:
    """Mock aiohttp HTTP response context manager (Zero Defaults)."""

    def __init__(self, status: int, data: Any, text_data: str) -> None:
        self.status: int = status
        self._data: Any = data
        self._text_data: str = text_data

    async def json(self) -> Any:
        return self._data

    async def text(self) -> str:
        return self._text_data

    async def __aenter__(self) -> MockResponse:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        pass


class MockClientSession:
    """Mock aiohttp ClientSession (Zero Defaults)."""

    def __init__(self, request_handler: Callable[..., Any]) -> None:
        self._request_handler: Callable[..., Any] = request_handler
        self.closed: bool = False
        self.post_calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        json: dict[str, Any] | None,
        headers: dict[str, str] | None,
        timeout: Any,
    ) -> Any:
        self.post_calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return self._request_handler(url, json, headers, timeout)

    async def close(self) -> None:
        self.closed = True


class MockGovernor:
    """Mock RateLimitGovernorPort for testing NinerouterClient (Zero Defaults)."""

    def __init__(self, specs_to_return: list[RouteQuotaSpec]) -> None:
        self._specs: list[RouteQuotaSpec] = specs_to_return
        self.cooldown_calls: list[tuple[str, float]] = []

    async def acquire_available_route(self, tier: str, estimated_tokens: int) -> RouteQuotaSpec:
        if len(self._specs) > 0:
            return self._specs.pop(0)
        raise RuntimeError("No routes available in mock governor")

    async def mark_route_cooldown(self, route_id: str, cooldown_seconds: float) -> None:
        self.cooldown_calls.append((route_id, cooldown_seconds))


class MockStructuredLogger:
    """Mock StructuredBotLogger for verifying critical fallback logging (Zero Defaults)."""

    def __init__(self) -> None:
        self.critical_logs: list[tuple[str, dict[str, Any]]] = []

    def critical(self, message: str, payload: dict[str, Any]) -> None:
        self.critical_logs.append((message, payload))


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


def create_ninerouter_config() -> NinerouterConfig:
    """Helper to create standard test NinerouterConfig."""
    return NinerouterConfig(
        base_url="http://9router:20128/v1",
        api_key="test-api-key-12345",
        timeout_seconds=15.0,
        fast_tier_model="ds/deepseek-v4-flash",
        deep_tier_model="ds/deepseek-v4-pro",
        judge_tier_model="gemini-2.5-flash",
    )


def test_ninerouter_response_entity() -> None:
    """Validate NinerouterResponse fields and constraints."""
    resp = NinerouterResponse(
        content="Market thesis: Bullish breakout confirmed.",
        model="gemini-cli/gemini-2.5-flash",
        finish_reason="stop",
        prompt_tokens=150,
        completion_tokens=45,
        is_fallback=False,
    )
    assert resp.content == "Market thesis: Bullish breakout confirmed."
    assert resp.model == "gemini-cli/gemini-2.5-flash"
    assert resp.finish_reason == "stop"
    assert resp.prompt_tokens == 150
    assert resp.completion_tokens == 45
    assert resp.is_fallback is False


@pytest.mark.asyncio
async def test_session_lifecycle_and_ownership() -> None:
    """Validate session closure behavior when session is owned vs caller-provided."""
    cfg = create_ninerouter_config()
    spec = create_sample_spec("route_1", "model_1")
    gov = MockGovernor([spec])

    # Case 1: Caller-provided session is NOT closed by NinerouterClient
    caller_session = MockClientSession(lambda *args: MockResponse(200, {}, "OK"))
    client = NinerouterClient(
        cfg,
        gov,
        caller_session,  # pyright: ignore[reportArgumentType]
        None,
    )
    await client.close()
    assert caller_session.closed is False

    # Case 2: Async context manager
    async with client:
        pass
    assert caller_session.closed is False


@pytest.mark.asyncio
async def test_successful_chat_completion_with_token_tracking() -> None:
    """Validate successful chat completion through 9Router with token usage tracking."""
    cfg = create_ninerouter_config()
    spec = create_sample_spec("gemini_slot_1", "gemini-cli/gemini-2.5-flash")
    gov = MockGovernor([spec])

    response_payload: dict[str, Any] = {
        "id": "chatcmpl-test-001",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Breakout detected on SOLUSDT. Long entry advised.",
                },
                "finish_reason": "stop",
            }
        ],
        "model": "gemini-cli/gemini-2.5-flash",
        "usage": {
            "prompt_tokens": 128,
            "completion_tokens": 36,
            "total_tokens": 164,
        },
    }

    def handle_request(
        url: str,
        json_body: dict[str, Any] | None,
        headers: dict[str, str] | None,
        timeout: Any,
    ) -> MockResponse:
        assert url == "http://9router:20128/v1/chat/completions"
        assert headers is not None
        assert headers["Authorization"] == "Bearer test-api-key-12345"
        assert headers["Content-Type"] == "application/json"
        assert json_body is not None
        assert json_body["model"] == "gemini-cli/gemini-2.5-flash"
        assert json_body["temperature"] == 0.2
        assert json_body["max_tokens"] == 1000
        return MockResponse(200, response_payload, "OK")

    mock_session = MockClientSession(handle_request)
    mock_logger = MockStructuredLogger()

    client = NinerouterClient(
        cfg,
        gov,
        mock_session,  # pyright: ignore[reportArgumentType]
        mock_logger,  # pyright: ignore[reportArgumentType]
    )

    messages = [{"role": "user", "content": "Analyze SOLUSDT 30m chart."}]
    result = await client.generate_chat_completion(
        tier="fast_tactical",
        messages=messages,
        estimated_tokens=200,
        temperature=0.2,
        max_tokens=1000,
    )

    assert result.is_fallback is False
    assert result.content == "Breakout detected on SOLUSDT. Long entry advised."
    assert result.model == "gemini-cli/gemini-2.5-flash"
    assert result.finish_reason == "stop"
    assert result.prompt_tokens == 128
    assert result.completion_tokens == 36
    assert len(mock_logger.critical_logs) == 0


@pytest.mark.asyncio
async def test_429_cooldown_trigger_and_route_retry() -> None:
    """Validate HTTP 429 triggers cooldown and retries next route successfully."""
    cfg = create_ninerouter_config()
    spec1 = create_sample_spec("route_rate_limited", "model_1")
    spec2 = create_sample_spec("route_backup", "model_2")
    gov = MockGovernor([spec1, spec2])

    call_count: list[int] = [0]

    def handle_request(
        url: str,
        json_body: dict[str, Any] | None,
        headers: dict[str, str] | None,
        timeout: Any,
    ) -> MockResponse:
        call_count[0] += 1
        if call_count[0] == 1:
            # First attempt hits 429
            return MockResponse(429, {"error": "Rate limit exceeded"}, "Too Many Requests")
        # Second attempt succeeds
        success_body = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Backup route succeeded."},
                    "finish_reason": "stop",
                }
            ],
            "model": "model_2",
            "usage": {"prompt_tokens": 50, "completion_tokens": 20},
        }
        return MockResponse(200, success_body, "OK")

    mock_session = MockClientSession(handle_request)
    mock_logger = MockStructuredLogger()

    client = NinerouterClient(
        cfg,
        gov,
        mock_session,  # pyright: ignore[reportArgumentType]
        mock_logger,  # pyright: ignore[reportArgumentType]
    )

    result = await client.generate_chat_completion(
        tier="fast_tactical",
        messages=[{"role": "user", "content": "Hello"}],
        estimated_tokens=100,
        temperature=0.1,
        max_tokens=500,
    )

    assert result.is_fallback is False
    assert result.content == "Backup route succeeded."
    assert result.model == "model_2"

    # Verify governor mark_route_cooldown was called for route_rate_limited with 30.0s
    assert len(gov.cooldown_calls) == 1
    assert gov.cooldown_calls[0] == ("route_rate_limited", 30.0)


@pytest.mark.asyncio
async def test_deterministic_fallback_on_network_connection_error() -> None:
    """Validate deterministic fallback and CRITICAL log emission on connection error."""
    cfg = create_ninerouter_config()
    spec = create_sample_spec("route_1", "model_1")
    gov = MockGovernor([spec])

    def handle_request(
        url: str,
        json_body: dict[str, Any] | None,
        headers: dict[str, str] | None,
        timeout: Any,
    ) -> MockResponse:
        raise aiohttp.ClientConnectionError("Connection refused to 9router:20128")

    mock_session = MockClientSession(handle_request)
    mock_logger = MockStructuredLogger()

    client = NinerouterClient(
        cfg,
        gov,
        mock_session,  # pyright: ignore[reportArgumentType]
        mock_logger,  # pyright: ignore[reportArgumentType]
    )

    result = await client.generate_chat_completion(
        tier="deep_reasoning",
        messages=[{"role": "user", "content": "Deliberate risk profile"}],
        estimated_tokens=500,
        temperature=0.0,
        max_tokens=800,
    )

    assert result.is_fallback is True
    assert result.content == ""
    assert result.model == "deterministic_fallback"
    assert result.finish_reason == "fallback"
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0

    # Verify CRITICAL log was emitted with exact message
    assert len(mock_logger.critical_logs) == 1
    msg, payload = mock_logger.critical_logs[0]
    assert msg == "9Router unavailable/exhausted, triggering Deterministic Fallback"
    assert payload["tier"] == "deep_reasoning"
    assert payload["is_fallback"] is True


@pytest.mark.asyncio
async def test_deterministic_fallback_on_5xx_server_error() -> None:
    """Validate deterministic fallback on 500 Internal Server Error."""
    cfg = create_ninerouter_config()
    spec = create_sample_spec("route_1", "model_1")
    gov = MockGovernor([spec])

    def handle_request(
        url: str,
        json_body: dict[str, Any] | None,
        headers: dict[str, str] | None,
        timeout: Any,
    ) -> MockResponse:
        return MockResponse(500, {}, "Internal Server Error")

    mock_session = MockClientSession(handle_request)
    mock_logger = MockStructuredLogger()

    client = NinerouterClient(
        cfg,
        gov,
        mock_session,  # pyright: ignore[reportArgumentType]
        mock_logger,  # pyright: ignore[reportArgumentType]
    )

    result = await client.generate_chat_completion(
        tier="fast_tactical",
        messages=[{"role": "user", "content": "Test"}],
        estimated_tokens=100,
        temperature=0.5,
        max_tokens=256,
    )

    assert result.is_fallback is True
    assert result.model == "deterministic_fallback"
    assert len(mock_logger.critical_logs) == 1


@pytest.mark.asyncio
async def test_deterministic_fallback_on_timeout_error() -> None:
    """Validate deterministic fallback on asyncio.TimeoutError."""
    cfg = create_ninerouter_config()
    spec = create_sample_spec("route_1", "model_1")
    gov = MockGovernor([spec])

    def handle_request(
        url: str,
        json_body: dict[str, Any] | None,
        headers: dict[str, str] | None,
        timeout: Any,
    ) -> MockResponse:
        raise TimeoutError("Request timed out after 15.0s")

    mock_session = MockClientSession(handle_request)
    mock_logger = MockStructuredLogger()

    client = NinerouterClient(
        cfg,
        gov,
        mock_session,  # pyright: ignore[reportArgumentType]
        mock_logger,  # pyright: ignore[reportArgumentType]
    )

    result = await client.generate_chat_completion(
        tier="fast_tactical",
        messages=[{"role": "user", "content": "Test timeout"}],
        estimated_tokens=100,
        temperature=0.5,
        max_tokens=256,
    )

    assert result.is_fallback is True
    assert result.model == "deterministic_fallback"
    assert len(mock_logger.critical_logs) == 1


@pytest.mark.asyncio
async def test_deterministic_fallback_on_governor_exhaustion() -> None:
    """Validate fallback when rate limit governor fails to acquire route."""
    cfg = create_ninerouter_config()
    # Empty specs list in mock governor -> raises RuntimeError
    gov = MockGovernor([])

    mock_session = MockClientSession(lambda *args: MockResponse(200, {}, "OK"))
    mock_logger = MockStructuredLogger()

    client = NinerouterClient(
        cfg,
        gov,
        mock_session,  # pyright: ignore[reportArgumentType]
        mock_logger,  # pyright: ignore[reportArgumentType]
    )

    result = await client.generate_chat_completion(
        tier="fast_tactical",
        messages=[{"role": "user", "content": "Test governor failure"}],
        estimated_tokens=100,
        temperature=0.5,
        max_tokens=256,
    )

    assert result.is_fallback is True
    assert result.model == "deterministic_fallback"
    assert len(mock_logger.critical_logs) == 1


def test_ast_compliance_zero_defaults_zero_fallbacks() -> None:
    """Validate NinerouterClient and test file strictly comply with Invariant #9."""
    repo_root = Path(__file__).resolve().parent.parent.parent

    target_files = [
        repo_root / "src" / "trading_bot" / "adapters" / "agents" / "ninerouter_client.py",
        Path(__file__).resolve(),
    ]

    all_violations: list[str] = []
    for f in target_files:
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        visitor = ZeroDefaultAndHardcodeVisitor(f)
        visitor.visit(tree)
        all_violations.extend(visitor.violations)

    assert not all_violations, "Invariant #9 violations found:\n" + "\n".join(all_violations)
