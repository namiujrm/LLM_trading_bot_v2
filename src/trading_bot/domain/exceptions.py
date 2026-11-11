"""Domain Exceptions — Pure Python.

Fail-Fast typed exceptions for quantitative and architectural validation.
"""

from __future__ import annotations


class DomainException(Exception):
    """Base domain exception for LLM trading bot domain layer."""


class ZeroVolumeException(DomainException):
    """Raised when trading volume or ADV is zero or non-positive."""


class InsufficientDataException(DomainException):
    """Raised when sample size is insufficient to compute a quantitative metric."""


class NumericalInstabilityException(DomainException):
    """Raised when a calculation produces NaN, Inf, or invalid degenerate bounds."""


class DegeneratePriceSeriesException(DomainException):
    """Raised when OHLC price series is invalid or flat."""


class UnsafeAlphaExpressionError(DomainException, ValueError):
    """Raised when an alpha formula violates AST safety, complexity, or causality invariants."""


class MalformedModelResponseException(DomainException, ValueError):
    """Raised when LLM response is unparseable, malformed, or missing required fields."""


class InvalidCausalClusterException(DomainException, ValueError):
    """Raised when an unknown or invalid causal cluster is specified."""


class ChampionModelNotFoundError(DomainException):
    """Raised when active champion model artifacts cannot be found on disk."""


class ChallengerModelNotFoundError(DomainException):
    """Raised when challenger model artifacts cannot be found on disk."""


class ModelArtifactCorruptError(DomainException):
    """Raised when persistent model artifacts are corrupt, invalid, or fail checksum validation."""


class ExchangeFilterViolationError(DomainException, ValueError):
    """Raised when order parameters violate exchange precision, lot size, or min notional rules."""
