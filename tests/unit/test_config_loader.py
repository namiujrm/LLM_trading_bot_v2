"""Unit tests for ConfigLoader and NinerouterConfig schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trading_bot.infrastructure.config_schemas import ConfigLoader, NinerouterConfig


def test_ninerouter_config_explicit_instantiation() -> None:
    """Verify NinerouterConfig requires all 6 fields including judge_tier_model."""
    cfg = NinerouterConfig(
        base_url="http://localhost:20128/v1",
        api_key="sk-test-key",
        timeout_seconds=45.0,
        fast_tier_model="ds/deepseek-v4-flash",
        deep_tier_model="ds/deepseek-v4-pro",
        judge_tier_model="gemini-2.5-flash",
    )
    assert cfg.base_url == "http://localhost:20128/v1"
    assert cfg.api_key == "sk-test-key"
    assert cfg.timeout_seconds == 45.0
    assert cfg.fast_tier_model == "ds/deepseek-v4-flash"
    assert cfg.deep_tier_model == "ds/deepseek-v4-pro"
    assert cfg.judge_tier_model == "gemini-2.5-flash"


def test_ninerouter_config_zero_default_discipline() -> None:
    """Verify omitting judge_tier_model or any other field raises ValidationError."""
    with pytest.raises(ValidationError):
        NinerouterConfig(  # pyright: ignore[reportCallIssue]
            base_url="http://localhost:20128/v1",
            api_key="sk-test-key",
            timeout_seconds=45.0,
            fast_tier_model="ds/deepseek-v4-flash",
            deep_tier_model="ds/deepseek-v4-pro",
        )


def test_config_loader_from_mapping() -> None:
    """Verify ConfigLoader loads NinerouterConfig from a dictionary mapping."""
    env_map = {
        "NINEROUTER_BASE_URL": "http://127.0.0.1:20128/v1",
        "NINEROUTER_API_KEY": "sk-mapped-key",
        "NINEROUTER_TIMEOUT_SECONDS": "55.0",
        "MODEL_FAST_TIER": "fast-model",
        "MODEL_DEEP_TIER": "deep-model",
        "MODEL_JUDGE_TIER": "gemini-2.5-flash",
    }
    cfg = ConfigLoader.load_ninerouter_config(env_map)
    assert cfg.base_url == "http://127.0.0.1:20128/v1"
    assert cfg.api_key == "sk-mapped-key"
    assert cfg.timeout_seconds == 55.0
    assert cfg.fast_tier_model == "fast-model"
    assert cfg.deep_tier_model == "deep-model"
    assert cfg.judge_tier_model == "gemini-2.5-flash"


def test_config_loader_missing_key_fail_fast() -> None:
    """Verify ConfigLoader raises KeyError immediately when any required env var is missing."""
    incomplete_env = {
        "NINEROUTER_BASE_URL": "http://127.0.0.1:20128/v1",
        "NINEROUTER_API_KEY": "sk-mapped-key",
        "NINEROUTER_TIMEOUT_SECONDS": "55.0",
        "MODEL_FAST_TIER": "fast-model",
        "MODEL_DEEP_TIER": "deep-model",
        # Missing MODEL_JUDGE_TIER
    }
    with pytest.raises(KeyError) as exc_info:
        ConfigLoader.load_ninerouter_config(incomplete_env)
    assert "MODEL_JUDGE_TIER" in str(exc_info.value)


def test_config_loader_from_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ConfigLoader.load_ninerouter_config_from_env reads from os.environ."""
    monkeypatch.setenv("NINEROUTER_BASE_URL", "http://env-host:20128/v1")
    monkeypatch.setenv("NINEROUTER_API_KEY", "env-secret-key")
    monkeypatch.setenv("NINEROUTER_TIMEOUT_SECONDS", "30.0")
    monkeypatch.setenv("MODEL_FAST_TIER", "env-fast")
    monkeypatch.setenv("MODEL_DEEP_TIER", "env-deep")
    monkeypatch.setenv("MODEL_JUDGE_TIER", "gemini-2.5-flash")

    cfg = ConfigLoader.load_ninerouter_config_from_env()
    assert cfg.base_url == "http://env-host:20128/v1"
    assert cfg.judge_tier_model == "gemini-2.5-flash"
