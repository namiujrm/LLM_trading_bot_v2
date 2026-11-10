"""Minute Microstructure Accumulator — OFI, OBI, Spreads & Liquidations (Zero Defaults)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from trading_bot.domain.entities.market.market_state import (
    EnrichedMicroBarRecord as EnrichedMicroBarRecord,
)


@dataclass
class MicroIngestorConfig:
    """Explicit parameters for microstructure accumulation and metrics scaling."""

    obi_depth_levels: int
    obi_decay_rate: float
    bps_scale_factor: float


class MinuteMicroAccumulator:
    """In-memory microstructure accumulator aggregating order book events over 1 minute."""

    def __init__(
        self,
        symbol: str,
        minute_open_ts_ms: int,
        cfg: MicroIngestorConfig,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        has_first_trade: bool,
        volume: float,
        quote_volume: float,
        taker_buy_vol: float,
        taker_buy_quote_vol: float,
        trade_count: int,
        ofi_sum: float,
        obi_samples: list[float],
        spread_bps_samples: list[float],
        liq_long_usd: float,
        liq_short_usd: float,
        prev_best_bid_p: float,
        prev_best_bid_q: float,
        prev_best_ask_p: float,
        prev_best_ask_q: float,
        has_prev_quote: bool,
    ) -> None:
        self.symbol: str = symbol
        self.minute_open_ts_ms: int = minute_open_ts_ms
        self.cfg: MicroIngestorConfig = cfg
        self.open_price: float = open_price
        self.high_price: float = high_price
        self.low_price: float = low_price
        self.close_price: float = close_price
        self.has_first_trade: bool = has_first_trade
        self.volume: float = volume
        self.quote_volume: float = quote_volume
        self.taker_buy_vol: float = taker_buy_vol
        self.taker_buy_quote_vol: float = taker_buy_quote_vol
        self.trade_count: int = trade_count
        self.ofi_sum: float = ofi_sum
        self.obi_samples: list[float] = obi_samples
        self.spread_bps_samples: list[float] = spread_bps_samples
        self.liq_long_usd: float = liq_long_usd
        self.liq_short_usd: float = liq_short_usd
        self.prev_best_bid_p: float = prev_best_bid_p
        self.prev_best_bid_q: float = prev_best_bid_q
        self.prev_best_ask_p: float = prev_best_ask_p
        self.prev_best_ask_q: float = prev_best_ask_q
        self.has_prev_quote: bool = has_prev_quote

    @classmethod
    def create_empty(
        cls,
        symbol: str,
        minute_open_ts_ms: int,
        cfg: MicroIngestorConfig,
    ) -> MinuteMicroAccumulator:
        """Create a fresh zero-initialized accumulator for a new 1-minute window."""
        return cls(
            symbol=symbol,
            minute_open_ts_ms=minute_open_ts_ms,
            cfg=cfg,
            open_price=0.0,
            high_price=0.0,
            low_price=0.0,
            close_price=0.0,
            has_first_trade=False,
            volume=0.0,
            quote_volume=0.0,
            taker_buy_vol=0.0,
            taker_buy_quote_vol=0.0,
            trade_count=0,
            ofi_sum=0.0,
            obi_samples=[],
            spread_bps_samples=[],
            liq_long_usd=0.0,
            liq_short_usd=0.0,
            prev_best_bid_p=0.0,
            prev_best_bid_q=0.0,
            prev_best_ask_p=0.0,
            prev_best_ask_q=0.0,
            has_prev_quote=False,
        )

    def on_agg_trade(
        self,
        price: float,
        qty: float,
        is_buyer_maker: bool,
        trade_ts_ms: int,
    ) -> None:
        """Process real-time aggregated trade from Binance Futures."""
        if not self.has_first_trade:
            self.open_price = price
            self.high_price = price
            self.low_price = price
            self.close_price = price
            self.has_first_trade = True
        else:
            if price > self.high_price:
                self.high_price = price
            if price < self.low_price:
                self.low_price = price
            self.close_price = price

        self.volume += qty
        quote_val = price * qty
        self.quote_volume += quote_val
        self.trade_count += 1

        # If buyer is NOT maker, then the taker is buyer (buyer-initiated aggressive trade)
        if not is_buyer_maker:
            self.taker_buy_vol += qty
            self.taker_buy_quote_vol += quote_val

    def on_kline(
        self,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float,
        quote_volume: float,
        taker_buy_vol: float,
        taker_buy_quote_vol: float,
        trade_count: int,
        is_closed: bool,
    ) -> None:
        """Process official Binance 1m kline event to reconcile or update bar OHLCV."""
        if not self.has_first_trade:
            self.open_price = open_price
            self.high_price = high_price
            self.low_price = low_price
            self.close_price = close_price
            self.has_first_trade = True
            self.volume = volume
            self.quote_volume = quote_volume
            self.taker_buy_vol = taker_buy_vol
            self.taker_buy_quote_vol = taker_buy_quote_vol
            self.trade_count = trade_count
        else:
            if high_price > self.high_price:
                self.high_price = high_price
            if low_price < self.low_price and low_price > 0.0:
                self.low_price = low_price
            self.close_price = close_price
            if volume > self.volume:
                self.volume = volume
            if quote_volume > self.quote_volume:
                self.quote_volume = quote_volume
            if taker_buy_vol > self.taker_buy_vol:
                self.taker_buy_vol = taker_buy_vol
            if taker_buy_quote_vol > self.taker_buy_quote_vol:
                self.taker_buy_quote_vol = taker_buy_quote_vol
            if trade_count > self.trade_count:
                self.trade_count = trade_count

    def on_depth_update(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
    ) -> None:
        """Process 100ms order book snapshot for Cont-Kukanov-Stoikov OFI and weighted OBI."""
        if not bids:
            raise ValueError(f"[{self.symbol}] Received empty bids orderbook snapshot.")
        if not asks:
            raise ValueError(f"[{self.symbol}] Received empty asks orderbook snapshot.")

        b_p, b_q = bids[0]
        a_p, a_q = asks[0]

        # 1. Cont-Kukanov-Stoikov Order Flow Imbalance (OFI) update
        if self.has_prev_quote:
            e_bid = 0.0
            if b_p >= self.prev_best_bid_p:
                e_bid += b_q
            if b_p <= self.prev_best_bid_p:
                e_bid -= self.prev_best_bid_q

            e_ask = 0.0
            if a_p <= self.prev_best_ask_p:
                e_ask += a_q
            if a_p >= self.prev_best_ask_p:
                e_ask -= self.prev_best_ask_q

            self.ofi_sum += e_bid - e_ask

        self.prev_best_bid_p = b_p
        self.prev_best_bid_q = b_q
        self.prev_best_ask_p = a_p
        self.prev_best_ask_q = a_q
        self.has_prev_quote = True

        # 2. Weighted multi-level OBI & Spread (bps)
        mid = 0.5 * (b_p + a_p)
        if mid <= 0.0:
            raise ValueError(f"[{self.symbol}] Non-positive mid price: {mid}")

        spread_bps = ((a_p - b_p) / mid) * self.cfg.bps_scale_factor
        self.spread_bps_samples.append(spread_bps)

        levels = self.cfg.obi_depth_levels
        decay = self.cfg.obi_decay_rate
        w_bid = sum(math.exp(-decay * float(i)) * q for i, (_, q) in enumerate(bids[:levels]))
        w_ask = sum(math.exp(-decay * float(i)) * q for i, (_, q) in enumerate(asks[:levels]))
        denom = w_bid + w_ask
        if denom > 0.0:
            self.obi_samples.append((w_bid - w_ask) / denom)

    def on_force_order(self, side: str, price: float, qty: float) -> None:
        """Accumulate liquidated margin positions from Binance forceOrder stream."""
        usd_val = price * qty
        if side == "SELL":
            self.liq_long_usd += usd_val
        elif side == "BUY":
            self.liq_short_usd += usd_val
        else:
            raise ValueError(f"[{self.symbol}] Invalid liquidation side: {side}")

    def flush_record(self) -> EnrichedMicroBarRecord:
        """Compute final 1-minute summary statistics and return immutable enriched bar."""
        if self.obi_samples:
            obi_mean = sum(self.obi_samples) / float(len(self.obi_samples))
            if len(self.obi_samples) > 1:
                variance = sum((x - obi_mean) ** 2 for x in self.obi_samples) / float(
                    len(self.obi_samples) - 1
                )
                obi_std = math.sqrt(variance)
            else:
                obi_std = 0.0
        else:
            obi_mean = 0.0
            obi_std = 0.0

        if self.spread_bps_samples:
            spread_mean = sum(self.spread_bps_samples) / float(len(self.spread_bps_samples))
        else:
            spread_mean = 0.0

        if not self.has_first_trade:
            open_p = self.prev_best_bid_p
            high_p = self.prev_best_bid_p
            low_p = self.prev_best_bid_p
            close_p = self.prev_best_bid_p
        else:
            open_p = self.open_price
            high_p = self.high_price
            low_p = self.low_price
            close_p = self.close_price

        return EnrichedMicroBarRecord(
            symbol=self.symbol,
            open_time_ms=self.minute_open_ts_ms,
            open=open_p,
            high=high_p,
            low=low_p,
            close=close_p,
            volume=self.volume,
            quote_volume=self.quote_volume,
            taker_buy_vol=self.taker_buy_vol,
            taker_buy_quote_vol=self.taker_buy_quote_vol,
            trade_count=self.trade_count,
            ofi_sum=round(self.ofi_sum, 4),
            obi_mean=round(obi_mean, 4),
            obi_std=round(obi_std, 4),
            spread_bps_mean=round(spread_mean, 4),
            liq_long_usd=round(self.liq_long_usd, 2),
            liq_short_usd=round(self.liq_short_usd, 2),
        )
