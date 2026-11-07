"""Account Portfolio Repository Adapter — Builder for AccountPortfolioContext."""

from __future__ import annotations

from trading_bot.adapters.persistence.database import DatabaseManager
from trading_bot.domain.entities.account_state import (
    AccountPortfolioContext,
    ActivePositionSnapshot,
    PendingOrderSnapshot,
)
from trading_bot.domain.entities.enums import ExecutionMode, TradeSide
from trading_bot.domain.services.risk_calculator import DeterministicRiskCalculator


class AccountPortfolioRepository:
    """Assembles unified AccountPortfolioContext from database persistence layer."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db: DatabaseManager = db_manager

    async def assemble_context(
        self,
        execution_mode: ExecutionMode,
        now_timestamp_ms: int,
        default_wallet_balance_usd: float,
        daily_dd_limit_pct: float,
        max_dd_limit_pct: float,
    ) -> AccountPortfolioContext:
        """Query orders_and_positions and account_equity_snapshots to assemble live context."""
        # 1. Fetch active open positions
        pos_query = """
            SELECT
                order_id, symbol, side, entry_price, stop_loss_price,
                take_profit_price, notional_usd, leverage, isolated_margin_usd,
                is_sl_at_breakeven, effective_at_risk_usd, net_pnl_usd, net_pnl_pct
            FROM orders_and_positions
            WHERE execution_mode = $1 AND status = 'OPEN';
        """
        pos_rows = await self._db.fetch(pos_query, execution_mode.value)
        active_positions: list[ActivePositionSnapshot] = []

        beta_map: dict[str, float] = {
            "BTCUSDT": 1.0,
            "ETHUSDT": 1.2,
            "SOLUSDT": 1.5,
            "BNBUSDT": 1.1,
        }

        for r in pos_rows:
            symbol = str(r["symbol"])
            side = TradeSide(str(r["side"]))
            entry_price = float(str(r["entry_price"]))
            stop_loss = float(str(r["stop_loss_price"]))
            take_profit = float(str(r["take_profit_price"]))
            notional = float(str(r["notional_usd"]))
            leverage = int(str(r["leverage"]))
            isolated_margin = float(str(r["isolated_margin_usd"]))
            is_breakeven = bool(r["is_sl_at_breakeven"])
            effective_risk = float(str(r["effective_at_risk_usd"]))
            pnl_usd = float(str(r["net_pnl_usd"]))
            pnl_pct = float(str(r["net_pnl_pct"]))

            qty = notional / entry_price if entry_price > 0.0 else 0.0
            mark_price = entry_price

            if leverage > 0:
                liq_buffer = 1.0 / float(leverage)
            else:
                liq_buffer = 1.0

            if side == TradeSide.LONG:
                liq_price = entry_price * (1.0 - liq_buffer)
            else:
                liq_price = entry_price * (1.0 + liq_buffer)

            if effective_risk > 0.0:
                r_mult = pnl_usd / effective_risk
            else:
                r_mult = 0.0

            if symbol in beta_map:
                btc_beta = beta_map[symbol]
            else:
                btc_beta = 1.0

            active_positions.append(
                ActivePositionSnapshot(
                    position_id=str(r["order_id"]),
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    mark_price=mark_price,
                    stop_loss_price=stop_loss,
                    take_profit_price=take_profit,
                    liquidation_price=liq_price,
                    quantity=qty,
                    notional_usd=notional,
                    isolated_margin_usd=isolated_margin,
                    leverage=leverage,
                    unrealized_pnl_usd=pnl_usd,
                    unrealized_pnl_pct=pnl_pct,
                    unrealized_r_multiple=r_mult,
                    is_sl_at_breakeven=is_breakeven,
                    effective_at_risk_usd=effective_risk,
                    btc_beta=btc_beta,
                )
            )

        # 2. Fetch latest equity snapshot
        snap_query = """
            SELECT
                wallet_balance_usd, daily_dd_pct, max_dd_pct, consecutive_losses
            FROM account_equity_snapshots
            WHERE execution_mode = $1
            ORDER BY snapshot_time DESC
            LIMIT 1;
        """
        snap_row = await self._db.fetchrow(snap_query, execution_mode.value)
        if snap_row is not None:
            wallet_balance = float(str(snap_row["wallet_balance_usd"]))
            current_daily_dd = float(str(snap_row["daily_dd_pct"]))
            current_max_dd = float(str(snap_row["max_dd_pct"]))
            consecutive_losses = int(str(snap_row["consecutive_losses"]))
        else:
            wallet_balance = default_wallet_balance_usd
            current_daily_dd = 0.0
            current_max_dd = 0.0
            consecutive_losses = 0

        # 3. Aggregate portfolio metrics
        unrealized_pnl_usd = sum(p.unrealized_pnl_usd for p in active_positions)
        equity_nav_usd = wallet_balance + unrealized_pnl_usd
        used_margin_usd = sum(p.isolated_margin_usd for p in active_positions)
        available_margin_usd = max(0.0, equity_nav_usd - used_margin_usd)

        if equity_nav_usd > 0.0:
            margin_utilization_pct = used_margin_usd / equity_nav_usd
        else:
            margin_utilization_pct = 0.0

        raw_open_heat_pct = DeterministicRiskCalculator.compute_raw_open_heat_pct(
            active_positions, equity_nav_usd
        )
        effective_open_heat_pct = DeterministicRiskCalculator.compute_effective_open_heat_pct(
            active_positions, equity_nav_usd
        )

        gross_exposure = sum(p.notional_usd for p in active_positions)
        net_delta = sum(
            p.notional_usd if p.side == TradeSide.LONG else -p.notional_usd
            for p in active_positions
        )
        net_beta = sum(
            p.notional_usd * p.btc_beta
            if p.side == TradeSide.LONG
            else -p.notional_usd * p.btc_beta
            for p in active_positions
        )

        remaining_daily_dd = max(0.0, daily_dd_limit_pct - current_daily_dd)
        remaining_max_dd = max(0.0, max_dd_limit_pct - current_max_dd)

        pending_orders: list[PendingOrderSnapshot] = []

        return AccountPortfolioContext(
            execution_mode=execution_mode,
            timestamp_ms=now_timestamp_ms,
            wallet_balance_usd=wallet_balance,
            unrealized_pnl_usd=unrealized_pnl_usd,
            equity_nav_usd=equity_nav_usd,
            used_margin_usd=used_margin_usd,
            available_margin_usd=available_margin_usd,
            margin_utilization_pct=margin_utilization_pct,
            current_daily_pnl_usd=unrealized_pnl_usd,
            current_daily_dd_pct=current_daily_dd,
            remaining_daily_dd_budget_pct=remaining_daily_dd,
            current_max_dd_pct=current_max_dd,
            remaining_max_dd_budget_pct=remaining_max_dd,
            consecutive_losses_count=consecutive_losses,
            active_positions=active_positions,
            pending_orders=pending_orders,
            gross_exposure_usd=gross_exposure,
            net_directional_delta_usd=net_delta,
            net_btc_beta_exposure_usd=net_beta,
            raw_open_heat_pct=raw_open_heat_pct,
            effective_open_heat_pct=effective_open_heat_pct,
            same_direction_active_count=len(active_positions),
        )
