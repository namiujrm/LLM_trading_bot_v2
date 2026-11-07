"""Explicit Configuration Schemas — Enforcing Invariant #9 (Zero Defaults)."""

from __future__ import annotations

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


class TelegramConfig(BaseModel):
    """Telegram alerting and command bot configuration (Zero Defaults)."""

    bot_token: str
    chat_id: str
