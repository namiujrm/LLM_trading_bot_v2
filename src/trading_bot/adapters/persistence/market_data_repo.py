"""Market Data Repository Adapter implementing MarketDataPort — Zero Defaults."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from trading_bot.adapters.persistence.database import DatabaseManager
from trading_bot.domain.entities.market.market_state import MicroBar1m, MultiTFBar
from trading_bot.domain.entities.risk.account_state import AccountPortfolioContext
from trading_bot.ports.market_data_port import MarketDataPort


class MarketDataRepository(MarketDataPort):
    """PostgreSQL / TimescaleDB repository implementing MarketDataPort interface."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db: DatabaseManager = db_manager

    async def fetch_recent_klines_1m(self, symbol: str, limit: int) -> list[MicroBar1m]:
        """Fetch the most recent N 1-minute bars for a symbol sorted chronologically."""
        query = """
            SELECT
                EXTRACT(EPOCH FROM open_time) * 1000 AS ts_ms,
                open, high, low, close, volume
            FROM market_klines_1m_enriched
            WHERE symbol = $1
            ORDER BY open_time DESC
            LIMIT $2;
        """
        rows = await self._db.fetch(query, symbol, limit)
        bars: list[MicroBar1m] = []
        for r in reversed(rows):
            bars.append(
                MicroBar1m(
                    timestamp_ms=int(float(str(r["ts_ms"]))),
                    open=float(str(r["open"])),
                    high=float(str(r["high"])),
                    low=float(str(r["low"])),
                    close=float(str(r["close"])),
                    volume=float(str(r["volume"])),
                )
            )
        return bars

    async def save_klines_1m(self, symbol: str, bars: list[MicroBar1m]) -> None:
        """Persist a batch of 1-minute bars into TimescaleDB hypertable."""
        if not bars:
            return

        query = """
            INSERT INTO market_klines_1m_enriched (
                open_time, symbol, open, high, low, close, volume,
                quote_volume, taker_buy_vol, taker_buy_quote_vol, trade_count,
                ofi_sum, obi_mean, obi_std, spread_bps_mean, liq_long_usd, liq_short_usd
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
            )
            ON CONFLICT (symbol, open_time) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                quote_volume = EXCLUDED.quote_volume;
        """
        records: list[tuple[Any, ...]] = []
        for b in bars:
            dt = datetime.fromtimestamp(b.timestamp_ms / 1000.0, tz=UTC)
            quote_vol = b.volume * b.close
            records.append(
                (
                    dt,
                    symbol,
                    b.open,
                    b.high,
                    b.low,
                    b.close,
                    b.volume,
                    quote_vol,
                    0.5 * b.volume,
                    0.5 * quote_vol,
                    1,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            )

        async with self._db.transaction() as conn:
            await conn.executemany(query, records)

    async def get_latest_price(self, symbol: str) -> float:
        """Get the latest close price for a symbol from 1m enriched bars."""
        query = """
            SELECT close FROM market_klines_1m_enriched
            WHERE symbol = $1
            ORDER BY open_time DESC
            LIMIT 1;
        """
        row = await self._db.fetchrow(query, symbol)
        if row is None:
            raise ValueError(f"No price data available for symbol: {symbol}")
        return float(str(row["close"]))

    async def fetch_multi_tf_bars(
        self, symbol: str, timeframe: str, limit: int
    ) -> list[MultiTFBar]:
        """Fetch multi-timeframe aggregated bars from continuous aggregates."""
        tf_view_map: dict[str, str] = {
            "15m": "market_klines_15m_cagg",
            "30m": "market_klines_30m_cagg",
            "1h": "market_klines_1h_cagg",
            "4h": "market_klines_4h_cagg",
        }
        if timeframe not in tf_view_map:
            raise ValueError(f"Unsupported timeframe for multi-tf query: {timeframe}")

        table_view = tf_view_map[timeframe]
        query = f"""
            SELECT
                EXTRACT(EPOCH FROM bucket) * 1000 AS ts_ms,
                open, high, low, close, volume
            FROM {table_view}
            WHERE symbol = $1
            ORDER BY bucket DESC
            LIMIT $2;
        """
        rows = await self._db.fetch(query, symbol, limit)
        bars: list[MultiTFBar] = []
        for r in reversed(rows):
            bars.append(
                MultiTFBar(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp_ms=int(float(str(r["ts_ms"]))),
                    open=float(str(r["open"])),
                    high=float(str(r["high"])),
                    low=float(str(r["low"])),
                    close=float(str(r["close"])),
                    volume=float(str(r["volume"])),
                )
            )
        return bars

    async def save_account_equity_snapshot(self, context: AccountPortfolioContext) -> None:
        """Persist account equity snapshot point-in-time."""
        query = """
            INSERT INTO account_equity_snapshots (
                snapshot_time, execution_mode, wallet_balance_usd, unrealized_pnl_usd,
                equity_nav_usd, used_margin_usd, available_margin_usd, margin_utilization_pct,
                open_positions_count, raw_open_heat_pct, effective_open_heat_pct, net_btc_beta_usd,
                daily_dd_pct, max_dd_pct, consecutive_losses, positions_snapshot_json
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
            )
            ON CONFLICT (execution_mode, snapshot_time) DO UPDATE SET
                wallet_balance_usd = EXCLUDED.wallet_balance_usd,
                unrealized_pnl_usd = EXCLUDED.unrealized_pnl_usd,
                equity_nav_usd = EXCLUDED.equity_nav_usd,
                used_margin_usd = EXCLUDED.used_margin_usd,
                available_margin_usd = EXCLUDED.available_margin_usd,
                margin_utilization_pct = EXCLUDED.margin_utilization_pct,
                open_positions_count = EXCLUDED.open_positions_count,
                raw_open_heat_pct = EXCLUDED.raw_open_heat_pct,
                effective_open_heat_pct = EXCLUDED.effective_open_heat_pct,
                net_btc_beta_usd = EXCLUDED.net_btc_beta_usd,
                daily_dd_pct = EXCLUDED.daily_dd_pct,
                max_dd_pct = EXCLUDED.max_dd_pct,
                consecutive_losses = EXCLUDED.consecutive_losses,
                positions_snapshot_json = EXCLUDED.positions_snapshot_json;
        """
        snapshot_dt = datetime.fromtimestamp(context.timestamp_ms / 1000.0, tz=UTC)
        positions_json = json.dumps([pos.model_dump() for pos in context.active_positions])

        await self._db.execute(
            query,
            snapshot_dt,
            context.execution_mode.value,
            context.wallet_balance_usd,
            context.unrealized_pnl_usd,
            context.equity_nav_usd,
            context.used_margin_usd,
            context.available_margin_usd,
            context.margin_utilization_pct,
            len(context.active_positions),
            context.raw_open_heat_pct,
            context.effective_open_heat_pct,
            context.net_btc_beta_exposure_usd,
            context.current_daily_dd_pct,
            context.current_max_dd_pct,
            context.consecutive_losses_count,
            positions_json,
        )
