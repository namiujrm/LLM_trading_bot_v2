"""Symbolic Alpha DSL Port Interface (Protocol)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from trading_bot.domain.entities.quant.alpha_dsl import AlphaFormulaAST
from trading_bot.domain.entities.quant.quant_configs import AlphaDSLConfig

if TYPE_CHECKING:
    import polars as pl


@runtime_checkable
class AlphaDSLCompilerPort(Protocol):
    """Abstract interface for validating, compiling, and executing Symbolic Alpha DSL expressions (Zero Defaults)."""

    def validate_formula(self, formula: str, cfg: AlphaDSLConfig) -> AlphaFormulaAST:
        """Parse and validate formula against AST node whitelist and causality invariants."""
        ...

    def compile_to_polars_expr(self, ast_obj: AlphaFormulaAST, cfg: AlphaDSLConfig) -> pl.Expr:
        """Translate validated AST representation into native Polars expression."""
        ...

    def execute_alpha(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        expr: pl.Expr,
    ) -> pl.Series:
        """Evaluate compiled Polars expression over a DataFrame or LazyFrame."""
        ...
