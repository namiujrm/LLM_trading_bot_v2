"""Agent Ports Layer — Clean Architecture Protocol interfaces for Multi-Agent Crew.

Enforces Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
Clean Architecture: Depends ONLY on domain entities and ports.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from trading_bot.domain.entities.common.enums import PlaybookType
from trading_bot.domain.entities.execution.trade_blueprint import (
    MarketThesis,
    RiskVerdict,
    TradeBlueprint,
)
from trading_bot.domain.entities.memory.episodic_case import (
    CohortBaseRateSummary,
    StratifiedMemoryPack,
)
from trading_bot.domain.entities.risk.account_state import AccountPortfolioContext
from trading_bot.ports.macro_sentinel_port import MacroBlackoutState


@runtime_checkable
class MarketScoutPort(Protocol):
    """Port interface for Market Scout agent."""

    async def analyze_market(
        self,
        symbol: str,
        quant_packet: dict[str, Any],
        playbook: PlaybookType,
        macro_state: MacroBlackoutState,
        btc_thesis: MarketThesis | None,
    ) -> MarketThesis:
        """Analyze market using quant features, playbook rules, and macro sentiment."""
        ...


@runtime_checkable
class TradeArchitectPort(Protocol):
    """Port interface for Trade Architect agent."""

    async def construct_trade_blueprint(
        self,
        thesis: MarketThesis,
        current_price: float,
        atr_30m: float,
        cohort_summary: CohortBaseRateSummary | None,
    ) -> TradeBlueprint:
        """Construct structured trade blueprint with order type, entry, SL, and TP."""
        ...


@runtime_checkable
class RiskArbiterPort(Protocol):
    """Port interface for Risk Arbiter agent."""

    async def evaluate_risk(
        self,
        blueprint: TradeBlueprint,
        account_context: AccountPortfolioContext,
        memory_pack: StratifiedMemoryPack | None,
    ) -> RiskVerdict:
        """Evaluate trade risk against account context and historical failure memory."""
        ...
