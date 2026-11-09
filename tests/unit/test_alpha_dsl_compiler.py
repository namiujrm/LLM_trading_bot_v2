"""Unit and Hypothesis tests for Symbolic Alpha DSL Validator and Polars Compiler.

Tests:
1. Conformance to AlphaDSLCompilerPort.
2. Strict AST node whitelist and malicious code rejection.
3. Complexity limit rejection (max depth, max nodes).
4. Strictly causal positive lag enforcement (rejection of d < 1, lookahead).
5. Canonical AST hashing and deduplication caching.
6. Exact mathematical parity on point and rolling operators.
7. Zero lookahead bias across time slices (10^-12 float parity).
8. WorldQuant seed alpha expressions execution.
9. Hypothesis property-based fuzz tests.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trading_bot.adapters.quant_engine.polars_dsl_compiler import PolarsAlphaDSLCompiler
from trading_bot.domain.entities.quant.alpha_dsl import AlphaFormulaAST
from trading_bot.domain.entities.quant.quant_configs import AlphaDSLConfig
from trading_bot.domain.exceptions import (
    UnsafeAlphaExpressionError,
)
from trading_bot.domain.services.quant.alpha_ast_validator import AlphaASTValidator
from trading_bot.ports.alpha_dsl_port import AlphaDSLCompilerPort


def create_test_config() -> AlphaDSLConfig:
    """Create a standard AlphaDSLConfig instance without default arguments."""
    return AlphaDSLConfig(
        max_ast_depth=8,
        max_ast_nodes=50,
        min_window_bars=1,
        max_window_bars=288,
        division_epsilon=1e-8,
    )


def create_test_dataframe(num_rows: int) -> pl.DataFrame:
    """Create a deterministic synthetic market DataFrame."""
    rng = np.random.default_rng(42)
    step_vals: list[float] = [float(x) for x in rng.normal(0.0, 50.0, size=num_rows)]
    close_vals: list[float] = []
    current_close = 50000.0
    for step in step_vals:
        current_close += step
        close_vals.append(current_close)

    high_vals = [c + float(abs(rng.normal(10.0, 5.0))) for c in close_vals]
    low_vals = [c - float(abs(rng.normal(10.0, 5.0))) for c in close_vals]
    open_vals = [
        low_vals[i] + (high_vals[i] - low_vals[i]) * float(rng.uniform(0.1, 0.9))
        for i in range(num_rows)
    ]
    volume_vals = [float(x) for x in rng.uniform(10.0, 500.0, size=num_rows)]
    quote_volume_vals = [volume_vals[i] * close_vals[i] for i in range(num_rows)]
    taker_buy_vol_vals = [volume_vals[i] * float(rng.uniform(0.3, 0.7)) for i in range(num_rows)]
    vwap_vals = [(high_vals[i] + low_vals[i] + close_vals[i]) / 3.0 for i in range(num_rows)]
    returns_vals: list[float] = [0.0]
    for i in range(1, num_rows):
        returns_vals.append((close_vals[i] - close_vals[i - 1]) / close_vals[i - 1])
    ofi_sum_vals = [float(x) for x in rng.normal(0.0, 20.0, size=num_rows)]
    spread_bps_vals = [float(x) for x in rng.uniform(1.0, 5.0, size=num_rows)]

    return pl.DataFrame(
        {
            "open": open_vals,
            "high": high_vals,
            "low": low_vals,
            "close": close_vals,
            "volume": volume_vals,
            "quote_volume": quote_volume_vals,
            "taker_buy_vol": taker_buy_vol_vals,
            "vwap": vwap_vals,
            "returns": returns_vals,
            "ofi_sum": ofi_sum_vals,
            "spread_bps_mean": spread_bps_vals,
        }
    )


def test_compiler_implements_port_protocol() -> None:
    validator = AlphaASTValidator()
    compiler = PolarsAlphaDSLCompiler(validator=validator)
    assert isinstance(compiler, AlphaDSLCompilerPort)


@pytest.mark.parametrize(
    "malicious_formula",
    [
        "__import__('os').system('ls')",
        "exec('import os')",
        "eval('2 + 2')",
        "open('/etc/passwd')",
        "__builtins__",
        "close.__class__",
        "close.foo",
        "close[0]",
        "lambda x: x",
        "x = 5",
        "[close, open]",
        "{'a': 1}",
        "def bad(): pass",
        "close > open",
        "close if open else 0",
        "not close",
        "~close",
        "close % 2",
        "close // 2",
        "close + 'string_literal'",
        "close + None",
        "close + True",
        "secret_field + 1",
        "unknown_function(close)",
        "",
        "   ",
        "close + +",
        "(close",
    ],
)
def test_ast_validator_rejects_unsafe_and_forbidden_syntax(malicious_formula: str) -> None:
    validator = AlphaASTValidator()
    cfg = create_test_config()
    with pytest.raises(UnsafeAlphaExpressionError):
        validator.validate_formula(formula=malicious_formula, cfg=cfg)


@pytest.mark.parametrize(
    "causal_violation_formula",
    [
        "ts_delay(close, -5)",
        "ts_delay(close, 0)",
        "ts_delay(close, -1)",
        "ts_delta(close, -2)",
        "ts_delta(close, 0)",
        "ts_mean(close, -10)",
        "ts_mean(close, 0)",
        "ts_std(close, -3)",
        "ts_zscore(close, -5)",
        "ts_rank(close, -1)",
        "ts_min(close, 0)",
        "ts_max(close, -4)",
        "decay_linear(close, 0)",
        "decay_linear(close, -2)",
        "ts_corr(close, volume, -5)",
        "ts_corr(close, volume, 0)",
        "ts_delay(close, 1.5)",
        "ts_delay(close, open)",
        "ts_delay(close, 300)",
        "abs(close, open)",
        "ts_delay(close)",
        "ts_corr(close, volume)",
        "abs(x=close)",
        "signed_power(close, -1.0)",
        "signed_power(close, 20.0)",
    ],
)
def test_ast_validator_rejects_causal_and_parameter_violations(
    causal_violation_formula: str,
) -> None:
    validator = AlphaASTValidator()
    cfg = create_test_config()
    with pytest.raises(UnsafeAlphaExpressionError):
        validator.validate_formula(formula=causal_violation_formula, cfg=cfg)


def test_ast_complexity_limits_depth_and_node_count() -> None:
    validator = AlphaASTValidator()

    strict_depth_cfg = AlphaDSLConfig(
        max_ast_depth=3,
        max_ast_nodes=50,
        min_window_bars=1,
        max_window_bars=288,
        division_epsilon=1e-8,
    )
    deep_formula = "abs(abs(abs(abs(close))))"
    with pytest.raises(UnsafeAlphaExpressionError) as exc_info:
        validator.validate_formula(formula=deep_formula, cfg=strict_depth_cfg)
    assert "depth" in str(exc_info.value).lower()

    strict_node_cfg = AlphaDSLConfig(
        max_ast_depth=8,
        max_ast_nodes=5,
        min_window_bars=1,
        max_window_bars=288,
        division_epsilon=1e-8,
    )
    many_nodes_formula = "close + open + high + low + volume"
    with pytest.raises(UnsafeAlphaExpressionError) as exc_info:
        validator.validate_formula(formula=many_nodes_formula, cfg=strict_node_cfg)
    assert "node count" in str(exc_info.value).lower()


def test_deterministic_canonical_hashing_and_deduplication() -> None:
    validator = AlphaASTValidator()
    cfg = create_test_config()

    f1 = "close  +   open"
    f2 = "(close+open)"
    f3 = " close + open "

    ast1 = validator.validate_formula(formula=f1, cfg=cfg)
    ast2 = validator.validate_formula(formula=f2, cfg=cfg)
    ast3 = validator.validate_formula(formula=f3, cfg=cfg)

    assert ast1.canonical_formula == "(close + open)"
    assert ast1.ast_hash == ast2.ast_hash
    assert ast2.ast_hash == ast3.ast_hash
    assert ast1.referenced_columns == ("close", "open")


def test_compiler_hash_cache() -> None:
    validator = AlphaASTValidator()
    compiler = PolarsAlphaDSLCompiler(validator=validator)
    cfg = create_test_config()

    assert compiler.cache_size == 0
    expr1 = compiler.compile_formula(formula="ts_delay(close, 5)", cfg=cfg)
    assert compiler.cache_size == 1

    expr2 = compiler.compile_formula(formula="ts_delay( close , 5 )", cfg=cfg)
    assert compiler.cache_size == 1
    assert expr1 is expr2

    compiler.clear_cache()
    assert compiler.cache_size == 0


def test_mathematical_operations_numeric_parity() -> None:
    df = create_test_dataframe(num_rows=100)
    validator = AlphaASTValidator()
    compiler = PolarsAlphaDSLCompiler(validator=validator)
    cfg = create_test_config()

    expr_delay = compiler.compile_formula(formula="ts_delay(close, 3)", cfg=cfg)
    res_delay = compiler.execute_alpha(df=df, expr=expr_delay).to_list()
    close_list: list[float] = [float(x) for x in df["close"].to_list()]
    assert res_delay[3:] == close_list[:-3]

    expr_delta = compiler.compile_formula(formula="ts_delta(close, 4)", cfg=cfg)
    res_delta = compiler.execute_alpha(df=df, expr=expr_delta).to_list()
    for i in range(4, 20):
        expected_diff = close_list[i] - close_list[i - 4]
        res_val = res_delta[i]
        assert res_val is not None
        assert abs(float(res_val) - expected_diff) < 1e-8

    expr_mean = compiler.compile_formula(formula="ts_mean(close, 5)", cfg=cfg)
    res_mean = compiler.execute_alpha(df=df, expr=expr_mean).to_list()
    for i in range(4, 15):
        slice_vals = close_list[i - 4 : i + 1]
        expected_m = sum(slice_vals) / len(slice_vals)
        res_val = res_mean[i]
        assert res_val is not None
        assert abs(float(res_val) - expected_m) < 1e-8

    expr_std = compiler.compile_formula(formula="ts_std(close, 6)", cfg=cfg)
    res_std = compiler.execute_alpha(df=df, expr=expr_std).to_list()
    for i in range(5, 15):
        slice_vals = close_list[i - 5 : i + 1]
        mean_val = sum(slice_vals) / len(slice_vals)
        var_val = sum((x - mean_val) ** 2 for x in slice_vals) / (len(slice_vals) - 1)
        expected_s = math.sqrt(var_val)
        res_val = res_std[i]
        assert res_val is not None
        assert abs(float(res_val) - expected_s) < 1e-8

    expr_zscore = compiler.compile_formula(formula="ts_zscore(close, 10)", cfg=cfg)
    res_zscore = compiler.execute_alpha(df=df, expr=expr_zscore)
    assert res_zscore.slice(9).is_finite().all()

    expr_min = compiler.compile_formula(formula="ts_min(low, 8)", cfg=cfg)
    res_min = compiler.execute_alpha(df=df, expr=expr_min).to_list()
    low_list: list[float] = [float(x) for x in df["low"].to_list()]
    for i in range(7, 15):
        expected_min = min(low_list[i - 7 : i + 1])
        res_val = res_min[i]
        assert res_val is not None
        assert abs(float(res_val) - expected_min) < 1e-8

    expr_max = compiler.compile_formula(formula="ts_max(high, 8)", cfg=cfg)
    res_max = compiler.execute_alpha(df=df, expr=expr_max).to_list()
    high_list: list[float] = [float(x) for x in df["high"].to_list()]
    for i in range(7, 15):
        expected_max = max(high_list[i - 7 : i + 1])
        res_val = res_max[i]
        assert res_val is not None
        assert abs(float(res_val) - expected_max) < 1e-8

    expr_rank = compiler.compile_formula(formula="ts_rank(close, 5)", cfg=cfg)
    res_rank = compiler.execute_alpha(df=df, expr=expr_rank).slice(4)
    assert (res_rank >= 0.2).all()
    assert (res_rank <= 1.0).all()

    expr_decay = compiler.compile_formula(formula="decay_linear(close, 3)", cfg=cfg)
    res_decay = compiler.execute_alpha(df=df, expr=expr_decay).to_list()
    w1, w2, w3 = 1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0
    for i in range(2, 10):
        expected_decay = w1 * close_list[i - 2] + w2 * close_list[i - 1] + w3 * close_list[i]
        res_val = res_decay[i]
        assert res_val is not None
        assert abs(float(res_val) - expected_decay) < 1e-8


def test_division_by_zero_guard_eliminates_nan_and_inf() -> None:
    df = create_test_dataframe(num_rows=20)
    validator = AlphaASTValidator()
    compiler = PolarsAlphaDSLCompiler(validator=validator)
    cfg = create_test_config()

    expr_div_zero = compiler.compile_formula(formula="close / 0.0", cfg=cfg)
    series = compiler.execute_alpha(df=df, expr=expr_div_zero)
    assert series.is_finite().all()
    assert not series.is_nan().any()


def test_strictly_causal_no_lookahead_bias_float_parity() -> None:
    rng = np.random.default_rng(12345)
    n_total = 500
    t_check = 300

    step_raw: list[float] = [float(x) for x in rng.normal(0.0, 100.0, size=n_total)]
    close_raw: list[float] = []
    c_acc = 50000.0
    for step in step_raw:
        c_acc += step
        close_raw.append(c_acc)

    volume_raw = [float(x) for x in rng.uniform(10.0, 1000.0, size=n_total)]
    ofi_raw = [float(x) for x in rng.normal(0.0, 50.0, size=n_total)]

    returns_raw: list[float] = [0.0]
    for i in range(1, n_total):
        returns_raw.append((close_raw[i] - close_raw[i - 1]) / close_raw[i - 1])

    df_full = pl.DataFrame(
        {
            "open": [c - 5.0 for c in close_raw],
            "high": [c + 20.0 for c in close_raw],
            "low": [c - 20.0 for c in close_raw],
            "close": close_raw,
            "volume": volume_raw,
            "quote_volume": [volume_raw[i] * close_raw[i] for i in range(n_total)],
            "taker_buy_vol": [volume_raw[i] * 0.5 for i in range(n_total)],
            "vwap": close_raw,
            "returns": returns_raw,
            "ofi_sum": ofi_raw,
            "spread_bps_mean": [float(x) for x in rng.uniform(2.0, 6.0, size=n_total)],
        }
    )

    df_corrupted = df_full.clone()
    corrupted_close = list(close_raw)
    for i in range(t_check + 1, n_total):
        corrupted_close[i] = 999_999_999.0
    corrupted_vol = list(volume_raw)
    for i in range(t_check + 1, n_total):
        corrupted_vol[i] = 0.00001
    corrupted_ofi = list(ofi_raw)
    for i in range(t_check + 1, n_total):
        corrupted_ofi[i] = -9999999.0

    df_corrupted = df_corrupted.with_columns(
        [
            pl.Series("close", corrupted_close),
            pl.Series("volume", corrupted_vol),
            pl.Series("ofi_sum", corrupted_ofi),
        ]
    )

    cfg = create_test_config()
    compiler = PolarsAlphaDSLCompiler(validator=AlphaASTValidator())

    test_formulas = [
        "tanh_norm(ts_zscore(ts_delta(close, 8), 24) + ts_corr(close, ofi_sum, 16))",
        "decay_linear(ts_rank(close, 10) - ts_rank(vwap, 10), 5)",
        "ts_mean(close, 20) / (ts_std(close, 20) + 1.0)",
        "signed_power(ts_delta(close, 5), 0.5)",
    ]

    for formula in test_formulas:
        expr = compiler.compile_formula(formula=formula, cfg=cfg)

        res_truncated = df_full.slice(0, t_check + 1).select(expr.alias("alpha"))["alpha"].to_list()
        res_corrupted = df_corrupted.select(expr.alias("alpha"))["alpha"].to_list()[: t_check + 1]

        for i in range(t_check + 1):
            val_trunc = res_truncated[i]
            val_corr = res_corrupted[i]
            if val_trunc is None or val_corr is None:
                assert val_trunc is None and val_corr is None
            else:
                diff = abs(float(val_trunc) - float(val_corr))
                assert diff < 1e-12, f"Discrepancy {diff} at bar {i} for formula {formula}"


def test_worldquant_alpha_seed_execution() -> None:
    df = create_test_dataframe(num_rows=200)
    lazy_df = df.lazy()
    validator = AlphaASTValidator()
    compiler = PolarsAlphaDSLCompiler(validator=validator)
    cfg = create_test_config()

    seed1 = "ts_rank(ts_delta(log(volume), 1), 5)"
    expr1 = compiler.compile_formula(formula=seed1, cfg=cfg)
    series1_df = compiler.execute_alpha(df=df, expr=expr1)
    series1_lazy = compiler.execute_alpha(df=lazy_df, expr=expr1)

    assert len(series1_df) == 200
    assert len(series1_lazy) == 200
    list_df = series1_df.drop_nulls().to_list()
    list_lazy = series1_lazy.drop_nulls().to_list()
    assert len(list_df) == len(list_lazy)
    for i in range(len(list_df)):
        assert abs(float(list_df[i]) - float(list_lazy[i])) < 1e-12

    seed2 = "tanh_norm(ts_zscore(ts_delta(close, 8), 24) + ts_corr(close, ofi_sum, 16))"
    expr2 = compiler.compile_formula(formula=seed2, cfg=cfg)
    series2 = compiler.execute_alpha(df=df, expr=expr2)
    valid_vals = series2.drop_nulls()
    assert (valid_vals >= -1.0).all()
    assert (valid_vals <= 1.0).all()
    assert valid_vals.is_finite().all()


@given(
    col_name=st.sampled_from(["open", "high", "low", "close", "volume", "vwap", "returns"]),
    window=st.integers(min_value=1, max_value=50),
    fn_name=st.sampled_from(["ts_delay", "ts_delta", "ts_mean", "ts_std", "ts_zscore", "ts_rank"]),
)
@settings(max_examples=30, deadline=None)
def test_hypothesis_fuzz_valid_expressions(col_name: str, window: int, fn_name: str) -> None:
    formula = f"{fn_name}({col_name}, {window})"
    cfg = create_test_config()
    validator = AlphaASTValidator()
    compiler = PolarsAlphaDSLCompiler(validator=validator)

    ast_obj = validator.validate_formula(formula=formula, cfg=cfg)
    assert isinstance(ast_obj, AlphaFormulaAST)
    assert ast_obj.node_count >= 3
    assert ast_obj.max_depth >= 2
    assert col_name in ast_obj.referenced_columns

    expr = compiler.compile_to_polars_expr(ast_obj=ast_obj, cfg=cfg)
    assert isinstance(expr, pl.Expr)


@given(
    invalid_str=st.text(
        alphabet=st.characters(blacklist_categories=("Cs",)),
        min_size=1,
        max_size=30,
    )
)
@settings(max_examples=30, deadline=None)
def test_hypothesis_fuzz_invalid_strings(invalid_str: str) -> None:
    cfg = create_test_config()
    validator = AlphaASTValidator()
    try:
        validator.validate_formula(formula=invalid_str, cfg=cfg)
    except UnsafeAlphaExpressionError:
        pass
    except Exception as exc:
        pytest.fail(f"Unexpected exception {type(exc).__name__} on input '{invalid_str}': {exc}")
