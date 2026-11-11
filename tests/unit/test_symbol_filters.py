"""Unit tests for SymbolFilterRule and Binance Gateway Quantization (Invariant #9 Compliant)."""

from __future__ import annotations

import pytest

from trading_bot.adapters.binance.binance_futures_gateway import (
    BinanceFuturesGatewayAdapter,
)
from trading_bot.domain.entities.market.symbol_filters import SymbolFilterRule
from trading_bot.domain.exceptions import ExchangeFilterViolationError
from trading_bot.infrastructure.config_schemas import BinanceConfig


def make_gateway() -> BinanceFuturesGatewayAdapter:
    cfg = BinanceConfig(
        api_key="mock_key",
        api_secret="mock_secret",
        testnet=True,
        recv_window_ms=5000,
    )
    gw = BinanceFuturesGatewayAdapter(cfg, None)
    gw.set_symbol_filter(
        SymbolFilterRule(
            symbol="BTCUSDT",
            price_tick_size=0.1,
            qty_step_size=0.001,
            min_qty=0.001,
            max_qty=1000.0,
            min_notional=5.0,
            price_precision=1,
            qty_precision=3,
        )
    )
    gw.set_symbol_filter(
        SymbolFilterRule(
            symbol="SOLUSDT",
            price_tick_size=0.01,
            qty_step_size=1.0,
            min_qty=1.0,
            max_qty=10000.0,
            min_notional=5.0,
            price_precision=2,
            qty_precision=0,
        )
    )
    return gw


def test_quantize_quantity_btc_flooring() -> None:
    gw = make_gateway()
    # 0.0026281648... should floor to 0.002 with stepSize 0.001
    res = gw.quantize_quantity("BTCUSDT", 0.0026281648)
    assert res == "0.002"


def test_quantize_quantity_sol_integer_step() -> None:
    gw = make_gateway()
    # 3.85 SOL should floor to 3 with stepSize 1.0
    res = gw.quantize_quantity("SOLUSDT", 3.85)
    assert res == "3"


def test_quantize_quantity_below_min_qty() -> None:
    gw = make_gateway()
    with pytest.raises(ExchangeFilterViolationError, match="below minQty"):
        gw.quantize_quantity("BTCUSDT", 0.0005)


def test_quantize_quantity_above_max_qty() -> None:
    gw = make_gateway()
    with pytest.raises(ExchangeFilterViolationError, match="exceeds maxQty"):
        gw.quantize_quantity("BTCUSDT", 1500.0)


def test_quantize_price_rounding() -> None:
    gw = make_gateway()
    # 95123.456 on BTCUSDT (tick 0.1) -> 95123.5
    res = gw.quantize_price("BTCUSDT", 95123.456)
    assert res == "95123.5"

    # 185.342 on SOLUSDT (tick 0.01) -> 185.34
    res_sol = gw.quantize_price("SOLUSDT", 185.342)
    assert res_sol == "185.34"


def test_quantize_unloaded_symbol_raises_key_error() -> None:
    gw = make_gateway()
    with pytest.raises(KeyError, match="Exchange filter not loaded"):
        gw.quantize_quantity("XRPUSDT", 10.0)
