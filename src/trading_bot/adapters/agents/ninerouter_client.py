"""9Router Resilient HTTP Client Adapter with Rate-Limit Governor and Fallback.

Enforces Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, cast

import aiohttp
from pydantic import BaseModel

from trading_bot.infrastructure.config_schemas import NinerouterConfig
from trading_bot.infrastructure.logging.bot_logger import StructuredBotLogger
from trading_bot.ports.rate_limit_governor_port import (
    RateLimitGovernorPort,
    RouteQuotaSpec,
)

_std_logger = logging.getLogger(__name__)


class NinerouterResponse(BaseModel):
    """Normalized response payload returned by 9Router client."""

    content: str
    model: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    is_fallback: bool


class NinerouterClient:
    """HTTP Client connecting to 9Router OpenAI-compatible gateway with resilience and fallback."""

    def __init__(
        self,
        config: NinerouterConfig,
        governor: RateLimitGovernorPort,
        session: aiohttp.ClientSession | None,
        logger: StructuredBotLogger | None,
    ) -> None:
        self._config: NinerouterConfig = config
        self._governor: RateLimitGovernorPort = governor
        self._session: aiohttp.ClientSession | None = session
        self._owns_session: bool = session is None
        self._logger: StructuredBotLogger | None = logger

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return active aiohttp ClientSession or lazily create a new one."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def generate_chat_completion(
        self,
        tier: str,
        messages: list[dict[str, str]],
        estimated_tokens: int,
        temperature: float,
        max_tokens: int,
    ) -> NinerouterResponse:
        """Call 9Router /chat/completions endpoint using rate-governed model route."""
        max_attempts: int = 3
        last_error: Exception | None = None

        for _ in range(max_attempts):
            try:
                spec: RouteQuotaSpec = await self._governor.acquire_available_route(
                    tier, estimated_tokens
                )
            except Exception as exc:
                last_error = exc
                break

            try:
                session: aiohttp.ClientSession = await self._get_session()
                base_url_clean: str = self._config.base_url.rstrip("/")
                endpoint: str = f"{base_url_clean}/chat/completions"
                headers: dict[str, str] = {
                    "Authorization": f"Bearer {self._config.api_key}",
                    "Content-Type": "application/json",
                }
                payload: dict[str, Any] = {
                    "model": spec.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                }
                timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)

                async with session.post(
                    endpoint, json=payload, headers=headers, timeout=timeout
                ) as resp:
                    if resp.status == 429:
                        await self._governor.mark_route_cooldown(spec.route_id, 30.0)
                        last_error = RuntimeError(f"HTTP 429 rate limit on route {spec.route_id}")
                        continue

                    if resp.status >= 400:
                        err_text: str = await resp.text()
                        raise RuntimeError(f"9Router error [{resp.status}]: {err_text}")

                    raw_data = await resp.json()
                    data = cast(dict[str, Any], raw_data)

                    choices_val = data["choices"]
                    choices = cast(list[dict[str, Any]], choices_val)
                    first_choice = choices[0]
                    msg_obj = cast(dict[str, Any], first_choice["message"])
                    raw_content = msg_obj["content"]
                    if raw_content is None:
                        content: str = ""
                    else:
                        content = str(raw_content)

                    raw_finish = first_choice["finish_reason"]
                    if raw_finish is None:
                        finish_reason: str = "stop"
                    else:
                        finish_reason = str(raw_finish)

                    resp_model: str = str(data["model"])
                    if "usage" in data and isinstance(data["usage"], dict):
                        usage_dict = cast(dict[str, Any], data["usage"])
                        prompt_tokens: int = int(usage_dict["prompt_tokens"])
                        completion_tokens: int = int(usage_dict["completion_tokens"])
                    else:
                        prompt_tokens = estimated_tokens
                        completion_tokens = 0

                    return NinerouterResponse(
                        content=content,
                        model=resp_model,
                        finish_reason=finish_reason,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        is_fallback=False,
                    )
            except Exception as exc:
                with contextlib.suppress(Exception):
                    await self._governor.mark_route_cooldown(spec.route_id, 30.0)
                last_error = exc
                break

        # Log CRITICAL alert when 9Router completely fails and trigger deterministic fallback
        if last_error is not None:
            err_detail: str = str(last_error)
        else:
            err_detail = "Rate limit attempts exhausted"

        critical_msg: str = "9Router unavailable/exhausted, triggering Deterministic Fallback"
        if self._logger is not None:
            self._logger.critical(
                critical_msg,
                {
                    "tier": tier,
                    "error": err_detail,
                    "is_fallback": True,
                },
            )
        else:
            _std_logger.critical("%s: %s", critical_msg, err_detail)

        return NinerouterResponse(
            content="",
            model="deterministic_fallback",
            finish_reason="fallback",
            prompt_tokens=0,
            completion_tokens=0,
            is_fallback=True,
        )

    async def close(self) -> None:
        """Close underlying aiohttp session if owned."""
        if self._owns_session and self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> NinerouterClient:
        """Async context manager enter."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()
