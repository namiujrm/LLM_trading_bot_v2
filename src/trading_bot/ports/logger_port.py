"""Structured Bot Logger Port (Clean Architecture, Invariant #9).

Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks.
Depends ONLY on standard typing.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StructuredBotLoggerPort(Protocol):
    """Protocol port interface for structured bot logging."""

    def debug(self, message: str, payload: dict[str, Any]) -> None:
        """Emit a DEBUG structured log record."""
        ...

    def info(self, message: str, payload: dict[str, Any]) -> None:
        """Emit an INFO structured log record."""
        ...

    def warning(self, message: str, payload: dict[str, Any]) -> None:
        """Emit a WARNING structured log record."""
        ...

    def error(self, message: str, payload: dict[str, Any]) -> None:
        """Emit an ERROR structured log record."""
        ...

    def critical(self, message: str, payload: dict[str, Any]) -> None:
        """Emit a CRITICAL structured log record."""
        ...
