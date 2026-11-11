"""Symbol Filter Rule Domain Entity (Invariant #9 Compliant).

Clean Architecture: Pure Domain Layer.
Strictly zero external IO or third-party computing libraries.
"""

from __future__ import annotations

from pydantic import BaseModel


class SymbolFilterRule(BaseModel):
    """Normalized exchange trading rules and precision constraints for a symbol.

    Enforces Invariant #9: All fields are required with zero defaults.
    """

    symbol: str
    price_tick_size: float
    qty_step_size: float
    min_qty: float
    max_qty: float
    min_notional: float
    price_precision: int
    qty_precision: int
