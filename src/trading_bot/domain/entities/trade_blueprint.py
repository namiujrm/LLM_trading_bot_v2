"""Trade Blueprint, Thesis, and Execution Ticket Entities."""

from __future__ import annotations

from pydantic import BaseModel

from trading_bot.domain.entities.enums import (
    EntryOrderType,
    ExistingPositionDirective,
    PlaybookType,
    RiskDecisionType,
    TradeSide,
    VetoCategory,
)


class MarketThesis(BaseModel):
    symbol: str
    selected_playbook: PlaybookType
    proposed_side: TradeSide
    supporting_active_features: list[str]
    regime_alignment_score: float
    thesis_summary: str


class TradeBlueprint(BaseModel):
    thesis: MarketThesis
    order_type: EntryOrderType
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    rationale: str


class RiskVerdict(BaseModel):
    decision: RiskDecisionType
    approved_leverage: int
    approved_notional_usd: float
    net_rr: float
    reason: str | None
    arbiter_size_multiplier: float
    veto_category: VetoCategory | None
    existing_position_directive: ExistingPositionDirective
    portfolio_risk_rationale: str


class BracketOrderTicket(BaseModel):
    order_id: str
    symbol: str
    side: TradeSide
    entry_type: EntryOrderType
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    quantity: float
    leverage: int
