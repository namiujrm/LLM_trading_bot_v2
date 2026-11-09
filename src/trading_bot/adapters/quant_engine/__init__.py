"""Quant Engine Adapters module."""

from __future__ import annotations

from trading_bot.adapters.quant_engine.numba_kernels import (
    compute_fracdiff_weights_numba,
    rolling_fracdiff_numba,
    rolling_hurst_rs_numba,
    rolling_yang_zhang_numba,
)
from trading_bot.adapters.quant_engine.polars_dsl_compiler import PolarsAlphaDSLCompiler
from trading_bot.adapters.quant_engine.polars_engine import PolarsQuantEngineAdapter
from trading_bot.adapters.quant_engine.timescale_feature_store import TimescaleFeatureStore

__all__ = [
    "PolarsAlphaDSLCompiler",
    "PolarsQuantEngineAdapter",
    "TimescaleFeatureStore",
    "compute_fracdiff_weights_numba",
    "rolling_fracdiff_numba",
    "rolling_hurst_rs_numba",
    "rolling_yang_zhang_numba",
]
