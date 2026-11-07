"""Account Portfolio Context and Position/Order Snapshot Entities — Pure Domain."""

from __future__ import annotations

from pydantic import BaseModel

from trading_bot.domain.entities.enums import (
    EntryOrderType,
    ExecutionMode,
    TradeSide,
)


class ActivePositionSnapshot(BaseModel):
    """Snapshot of an active open position with risk and breakeven metrics."""

    position_id: str
    symbol: str
    side: TradeSide
    entry_price: float
    mark_price: float
    stop_loss_price: float
    take_profit_price: float
    liquidation_price: float
    quantity: float
    notional_usd: float
    isolated_margin_usd: float
    leverage: int
    unrealized_pnl_usd: float
    unrealized_pnl_pct: float
    unrealized_r_multiple: float
    is_sl_at_breakeven: bool
    effective_at_risk_usd: float
    btc_beta: float


class PendingOrderSnapshot(BaseModel):
    """Snapshot of a pending unexecuted entry limit order reserving margin."""

    order_id: str
    symbol: str
    side: TradeSide
    order_type: EntryOrderType
    limit_price: float
    notional_usd: float
    reserved_margin_usd: float
    created_at_ms: int


class AccountPortfolioContext(BaseModel):
    """Comprehensive portfolio and margin context injected into Risk Arbiter and Calculator."""

    execution_mode: ExecutionMode
    timestamp_ms: int
    wallet_balance_usd: float
    unrealized_pnl_usd: float
    equity_nav_usd: float
    used_margin_usd: float
    available_margin_usd: float
    margin_utilization_pct: float
    current_daily_pnl_usd: float
    current_daily_dd_pct: float
    remaining_daily_dd_budget_pct: float
    current_max_dd_pct: float
    remaining_max_dd_budget_pct: float
    consecutive_losses_count: int
    active_positions: list[ActivePositionSnapshot]
    pending_orders: list[PendingOrderSnapshot]
    gross_exposure_usd: float
    net_directional_delta_usd: float
    net_btc_beta_exposure_usd: float
    raw_open_heat_pct: float
    effective_open_heat_pct: float
    same_direction_active_count: int
