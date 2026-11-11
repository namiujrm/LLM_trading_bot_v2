"""Explicit Configuration Schemas — Enforcing Invariant #9 (Zero Defaults)."""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import BaseModel


class DatabaseConfig(BaseModel):
    """PostgreSQL / TimescaleDB configuration with required fields (Zero Defaults)."""

    host: str
    port: int
    db_name: str
    user: str
    password: str
    max_connections: int


class RedisConfig(BaseModel):
    """Redis store and rate limiter configuration (Zero Defaults)."""

    host: str
    port: int
    db: int
    password: str
    ssl: bool


class BinanceConfig(BaseModel):
    """Binance Futures API credentials and networking parameters (Zero Defaults)."""

    api_key: str
    api_secret: str
    testnet: bool
    recv_window_ms: int


class NinerouterConfig(BaseModel):
    """9router LLM gateway configuration (Zero Defaults)."""

    base_url: str
    api_key: str
    timeout_seconds: float
    fast_tier_model: str
    deep_tier_model: str
    judge_tier_model: str


class TelegramConfig(BaseModel):
    """Telegram alerting and command bot configuration (Zero Defaults)."""

    bot_token: str
    chat_id: str


class LoggingConfig(BaseModel):
    """Structured logging configuration (Zero Defaults)."""

    log_dir: str
    retention_days: int
    min_level: str
    console_output: bool


class ConfigLoader:
    """Factory loading configurations from environment mappings with zero defaults and fail-fast discipline."""

    @staticmethod
    def load_ninerouter_config(env_mapping: Mapping[str, str]) -> NinerouterConfig:
        """Construct NinerouterConfig from an explicit environment mapping (fail-fast KeyError on missing keys)."""
        return NinerouterConfig(
            base_url=env_mapping["NINEROUTER_BASE_URL"],
            api_key=env_mapping["NINEROUTER_API_KEY"],
            timeout_seconds=float(env_mapping["NINEROUTER_TIMEOUT_SECONDS"]),
            fast_tier_model=env_mapping["MODEL_FAST_TIER"],
            deep_tier_model=env_mapping["MODEL_DEEP_TIER"],
            judge_tier_model=env_mapping["MODEL_JUDGE_TIER"],
        )

    @staticmethod
    def load_ninerouter_config_from_env() -> NinerouterConfig:
        """Construct NinerouterConfig directly from os.environ."""
        return ConfigLoader.load_ninerouter_config(os.environ)

    @staticmethod
    def load_telegram_config(env_mapping: Mapping[str, str]) -> TelegramConfig:
        """Construct TelegramConfig from an explicit environment mapping (fail-fast KeyError on missing keys)."""
        return TelegramConfig(
            bot_token=env_mapping["TELEGRAM_BOT_TOKEN"],
            chat_id=env_mapping["TELEGRAM_CHAT_ID"],
        )

    @staticmethod
    def load_telegram_config_from_env() -> TelegramConfig:
        """Construct TelegramConfig directly from os.environ."""
        return ConfigLoader.load_telegram_config(os.environ)

    @staticmethod
    def load_binance_config(env_mapping: Mapping[str, str]) -> BinanceConfig:
        """Construct BinanceConfig from an explicit environment mapping (fail-fast KeyError on missing keys)."""
        api_key: str = env_mapping["BINANCE_API_KEY"]
        api_secret: str = env_mapping["BINANCE_API_SECRET"]
        raw_testnet: str = env_mapping["BINANCE_TESTNET"].strip().lower()
        testnet_val: bool = raw_testnet in ("true", "1", "yes")
        recv_window_ms: int = int(env_mapping["BINANCE_RECV_WINDOW_MS"])
        return BinanceConfig(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet_val,
            recv_window_ms=recv_window_ms,
        )

    @staticmethod
    def load_binance_config_from_env() -> BinanceConfig:
        """Construct BinanceConfig directly from os.environ."""
        return ConfigLoader.load_binance_config(os.environ)
