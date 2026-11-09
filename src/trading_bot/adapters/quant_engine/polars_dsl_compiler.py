"""Polars Symbolic Alpha DSL AST Compiler Adapter (Zero Defaults).

Translates validated AlphaFormulaAST into native Polars pl.Expr constructs.
Zero eval(), zero exec(), zero defaults, zero lookahead.
Maintains deterministic SHA-256 AST hash cache.
"""

from __future__ import annotations

import ast

import polars as pl

from trading_bot.domain.entities.quant.alpha_dsl import AlphaFormulaAST
from trading_bot.domain.entities.quant.quant_configs import AlphaDSLConfig
from trading_bot.domain.exceptions import (
    NumericalInstabilityException,
    UnsafeAlphaExpressionError,
)
from trading_bot.domain.services.quant.alpha_ast_validator import AlphaASTValidator


class PolarsAlphaDSLCompiler:
    """Polars expression compiler conforming to AlphaDSLCompilerPort."""

    def __init__(self, validator: AlphaASTValidator) -> None:
        self.validator: AlphaASTValidator = validator
        self._ast_cache: dict[str, pl.Expr] = {}

    def validate_formula(self, formula: str, cfg: AlphaDSLConfig) -> AlphaFormulaAST:
        """Parse and validate formula against AST node whitelist and causality invariants."""
        return self.validator.validate_formula(formula=formula, cfg=cfg)

    def compile_to_polars_expr(self, ast_obj: AlphaFormulaAST, cfg: AlphaDSLConfig) -> pl.Expr:
        """Translate validated AST representation into native Polars expression."""
        if ast_obj.ast_hash in self._ast_cache:
            return self._ast_cache[ast_obj.ast_hash]

        expr = self._compile_node(node=ast_obj.root_node.body, cfg=cfg)
        self._ast_cache[ast_obj.ast_hash] = expr
        return expr

    def compile_formula(self, formula: str, cfg: AlphaDSLConfig) -> pl.Expr:
        """Convenience method to validate and compile formula in a single step."""
        ast_obj = self.validate_formula(formula=formula, cfg=cfg)
        return self.compile_to_polars_expr(ast_obj=ast_obj, cfg=cfg)

    def execute_alpha(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        expr: pl.Expr,
    ) -> pl.Series:
        """Evaluate compiled Polars expression over a DataFrame or LazyFrame."""
        if isinstance(df, pl.LazyFrame):
            evaluated = df.select(expr.alias("alpha")).collect(engine="streaming")
            series = evaluated["alpha"]
        else:
            evaluated = df.select(expr.alias("alpha"))
            series = evaluated["alpha"]

        non_null = series.drop_nulls()
        if len(non_null) > 0 and (non_null.is_nan().any() or non_null.is_infinite().any()):
            raise NumericalInstabilityException(
                "Alpha expression produced NaN or Infinite values during evaluation."
            )

        return series

    def clear_cache(self) -> None:
        """Clear internal compiled expression cache."""
        self._ast_cache.clear()

    @property
    def cache_size(self) -> int:
        """Return number of cached compiled expressions."""
        return len(self._ast_cache)

    def _extract_window(self, node: ast.expr) -> int:
        """Extract an integer window constant with strict type validation."""
        if not isinstance(node, ast.Constant):
            raise UnsafeAlphaExpressionError("Window parameter must be an integer constant.")
        if not isinstance(node.value, int):
            raise UnsafeAlphaExpressionError("Window parameter must be an integer constant.")
        return int(node.value)

    def _compile_node(self, node: ast.AST, cfg: AlphaDSLConfig) -> pl.Expr:
        if isinstance(node, ast.Name):
            return pl.col(node.id)

        if isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, (int, float)):
                return pl.lit(float(val))
            raise UnsafeAlphaExpressionError(f"Constant value must be numeric, got {val!r}")

        if isinstance(node, ast.UnaryOp):
            operand = self._compile_node(node=node.operand, cfg=cfg)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return operand
            raise UnsafeAlphaExpressionError(
                f"Unsupported unary operator: {type(node.op).__name__}"
            )

        if isinstance(node, ast.BinOp):
            left = self._compile_node(node=node.left, cfg=cfg)
            right = self._compile_node(node=node.right, cfg=cfg)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                sign_expr = pl.when(right >= 0.0).then(1.0).otherwise(-1.0)
                safe_den = right.abs().clip(lower_bound=cfg.division_epsilon) * sign_expr
                return left / safe_den
            if isinstance(node.op, ast.Pow):
                return left.pow(right)
            raise UnsafeAlphaExpressionError(
                f"Unsupported binary operator: {type(node.op).__name__}"
            )

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise UnsafeAlphaExpressionError("Function call target must be an identifier.")
            fn = node.func.id
            args = node.args

            if fn == "abs":
                return self._compile_node(node=args[0], cfg=cfg).abs()

            if fn == "sign":
                return self._compile_node(node=args[0], cfg=cfg).sign()

            if fn == "log":
                x = self._compile_node(node=args[0], cfg=cfg)
                return x.abs().clip(lower_bound=cfg.division_epsilon).log()

            if fn == "sqrt":
                x = self._compile_node(node=args[0], cfg=cfg)
                return x.abs().clip(lower_bound=0.0).sqrt()

            if fn == "tanh_norm":
                return self._compile_node(node=args[0], cfg=cfg).tanh()

            if fn == "signed_power":
                x = self._compile_node(node=args[0], cfg=cfg)
                exp_node = args[1]
                if not isinstance(exp_node, ast.Constant):
                    raise UnsafeAlphaExpressionError(
                        "signed_power exponent must be a constant number."
                    )
                if not isinstance(exp_node.value, (int, float)):
                    raise UnsafeAlphaExpressionError(
                        "signed_power exponent must be a constant number."
                    )
                a = float(exp_node.value)
                return x.sign() * (x.abs().clip(lower_bound=0.0).pow(a))

            if fn == "ts_delay":
                x = self._compile_node(node=args[0], cfg=cfg)
                d = self._extract_window(node=args[1])
                return x.shift(d)

            if fn == "ts_delta":
                x = self._compile_node(node=args[0], cfg=cfg)
                d = self._extract_window(node=args[1])
                return x - x.shift(d)

            if fn == "ts_mean":
                x = self._compile_node(node=args[0], cfg=cfg)
                d = self._extract_window(node=args[1])
                return x.rolling_mean(window_size=d)

            if fn == "ts_std":
                x = self._compile_node(node=args[0], cfg=cfg)
                d = self._extract_window(node=args[1])
                return x.rolling_std(window_size=d)

            if fn == "ts_zscore":
                x = self._compile_node(node=args[0], cfg=cfg)
                d = self._extract_window(node=args[1])
                mean = x.rolling_mean(window_size=d)
                std = x.rolling_std(window_size=d)
                safe_std = std.abs().clip(lower_bound=cfg.division_epsilon)
                return (x - mean) / safe_std

            if fn == "ts_min":
                x = self._compile_node(node=args[0], cfg=cfg)
                d = self._extract_window(node=args[1])
                return x.rolling_min(window_size=d)

            if fn == "ts_max":
                x = self._compile_node(node=args[0], cfg=cfg)
                d = self._extract_window(node=args[1])
                return x.rolling_max(window_size=d)

            if fn == "ts_corr":
                x = self._compile_node(node=args[0], cfg=cfg)
                y = self._compile_node(node=args[1], cfg=cfg)
                d = self._extract_window(node=args[2])
                return pl.rolling_corr(x, y, window_size=d)

            if fn == "ts_rank":
                x = self._compile_node(node=args[0], cfg=cfg)
                d = self._extract_window(node=args[1])
                raw_rank = pl.sum_horizontal(
                    [(x >= x.shift(i)).cast(pl.Float64) for i in range(d)]
                ) / float(d)
                return pl.when(x.shift(d - 1).is_null()).then(None).otherwise(raw_rank)

            if fn == "decay_linear":
                x = self._compile_node(node=args[0], cfg=cfg)
                d = self._extract_window(node=args[1])
                w_sum = float(d * (d + 1) // 2)
                terms = [x.shift(d - i) * (float(i) / w_sum) for i in range(1, d + 1)]
                raw_decay = pl.sum_horizontal(terms)
                return pl.when(x.shift(d - 1).is_null()).then(None).otherwise(raw_decay)

        raise UnsafeAlphaExpressionError(
            f"Unsupported AST node during compilation: {type(node).__name__}"
        )
