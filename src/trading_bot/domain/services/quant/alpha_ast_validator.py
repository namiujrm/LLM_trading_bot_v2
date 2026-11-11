"""Symbolic Alpha DSL AST Validator — Pure Domain Service (Zero Defaults).

Validates symbolic alpha expressions against strict security whitelists,
causal positive-lag invariants, and complexity limits.
Zero eval(), zero exec(), 100% pure Python AST inspection.
"""

from __future__ import annotations

import ast
import hashlib
import math

from trading_bot.domain.entities.quant.alpha_dsl import AlphaFormulaAST
from trading_bot.domain.entities.quant.quant_configs import AlphaDSLConfig
from trading_bot.domain.exceptions import UnsafeAlphaExpressionError

ALLOWED_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_vol",
        "vwap",
        "returns",
        "ofi_sum",
        "spread_bps_mean",
        "vpin",
        "obi",
        "hurst",
        "yang_zhang_vol",
        "fracdiff_close",
        "funding_rate",
        "open_interest",
        "basis_bps",
        "liquidation_vol",
        "btc_returns",
        "beta_divergence",
    }
)

ALLOWED_UNARY_FUNCS: frozenset[str] = frozenset(
    {
        "abs",
        "sign",
        "log",
        "sqrt",
        "tanh_norm",
    }
)

ALLOWED_ROLLING_FUNCS: frozenset[str] = frozenset(
    {
        "ts_delay",
        "ts_delta",
        "ts_mean",
        "ts_std",
        "ts_zscore",
        "ts_rank",
        "ts_min",
        "ts_max",
        "decay_linear",
    }
)

ALLOWED_MATH_FUNCS: frozenset[str] = (
    ALLOWED_UNARY_FUNCS | ALLOWED_ROLLING_FUNCS | frozenset({"signed_power", "ts_corr"})
)

ALLOWED_AST_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.Name,
    ast.Constant,
    ast.UnaryOp,
    ast.BinOp,
    ast.Call,
)


class AlphaASTValidator:
    """Pure domain service for validating and canonicalizing symbolic Alpha DSL ASTs."""

    def __init__(self) -> None:
        pass

    def validate_formula(self, formula: str, cfg: AlphaDSLConfig) -> AlphaFormulaAST:
        """Parse formula string and validate against safety, complexity, and causality invariants."""
        formula_clean = formula.strip()
        if len(formula_clean) == 0:
            raise UnsafeAlphaExpressionError("Formula string cannot be empty or whitespace.")

        try:
            tree = ast.parse(formula_clean, mode="eval")
        except SyntaxError as exc:
            raise UnsafeAlphaExpressionError(
                f"Syntax error in formula '{formula_clean}': {exc}"
            ) from exc
        except Exception as exc:
            raise UnsafeAlphaExpressionError(
                f"Failed to parse formula '{formula_clean}': {exc}"
            ) from exc

        return self.validate_ast(tree=tree, raw_formula=formula, cfg=cfg)

    def validate_ast(
        self,
        tree: ast.AST,
        raw_formula: str,
        cfg: AlphaDSLConfig,
    ) -> AlphaFormulaAST:
        """Validate an already parsed ast.Expression tree."""
        if not isinstance(tree, ast.Expression):
            raise UnsafeAlphaExpressionError(
                f"Root AST node must be ast.Expression, got {type(tree).__name__}."
            )

        all_nodes = list(ast.walk(tree))
        node_count = len(all_nodes)
        if node_count > cfg.max_ast_nodes:
            raise UnsafeAlphaExpressionError(
                f"AST node count {node_count} exceeds maximum allowed limit {cfg.max_ast_nodes}."
            )

        depth = self._compute_depth(tree.body)
        if depth > cfg.max_ast_depth:
            raise UnsafeAlphaExpressionError(
                f"AST depth {depth} exceeds maximum allowed limit {cfg.max_ast_depth}."
            )

        referenced_columns: list[str] = []
        self._validate_node(node=tree.body, cfg=cfg, referenced_cols=referenced_columns)

        canonical_formula, ast_hash = self.compute_canonical_hash(tree=tree)
        unique_referenced = tuple(sorted(set(referenced_columns)))

        return AlphaFormulaAST(
            raw_formula=raw_formula,
            canonical_formula=canonical_formula,
            ast_hash=ast_hash,
            node_count=node_count,
            max_depth=depth,
            referenced_columns=unique_referenced,
            root_node=tree,
        )

    def compute_canonical_hash(self, tree: ast.Expression) -> tuple[str, str]:
        """Compute canonical string representation and SHA-256 hash."""
        canonical_str = self._to_canonical_str(tree.body)
        ast_hash = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
        return canonical_str, ast_hash

    def _compute_depth(self, node: ast.AST) -> int:
        children = list(ast.iter_child_nodes(node))
        if len(children) == 0:
            return 1
        return 1 + max(self._compute_depth(child) for child in children)

    def _validate_node(
        self,
        node: ast.AST,
        cfg: AlphaDSLConfig,
        referenced_cols: list[str],
    ) -> None:
        if not isinstance(node, ALLOWED_AST_NODES):
            raise UnsafeAlphaExpressionError(
                f"Forbidden AST node type '{type(node).__name__}'. Only expressions, names, "
                f"numeric constants, unary ops, binary ops, and whitelisted calls are permitted."
            )

        if isinstance(node, ast.Name):
            if node.id not in ALLOWED_FEATURE_COLUMNS:
                raise UnsafeAlphaExpressionError(
                    f"Forbidden identifier '{node.id}'. Permitted columns: "
                    f"{sorted(ALLOWED_FEATURE_COLUMNS)}"
                )
            referenced_cols.append(node.id)
            return

        if isinstance(node, ast.Constant):
            val = node.value
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise UnsafeAlphaExpressionError(
                    f"Constant values must be numeric (int or float), got {type(val).__name__}."
                )
            num_val = float(val)
            if math.isnan(num_val) or math.isinf(num_val):
                raise UnsafeAlphaExpressionError("Numeric constant cannot be NaN or Infinite.")
            return

        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                raise UnsafeAlphaExpressionError(
                    f"Forbidden unary operator '{type(node.op).__name__}'. Only unary +/- permitted."
                )
            self._validate_node(node=node.operand, cfg=cfg, referenced_cols=referenced_cols)
            return

        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
                raise UnsafeAlphaExpressionError(
                    f"Forbidden binary operator '{type(node.op).__name__}'. "
                    f"Only +, -, *, /, ** are permitted."
                )
            self._validate_node(node=node.left, cfg=cfg, referenced_cols=referenced_cols)
            self._validate_node(node=node.right, cfg=cfg, referenced_cols=referenced_cols)
            return

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise UnsafeAlphaExpressionError(
                    "Function call target must be a direct identifier name."
                )
            fn_name = node.func.id
            if fn_name not in ALLOWED_MATH_FUNCS:
                raise UnsafeAlphaExpressionError(
                    f"Forbidden function call '{fn_name}'. Whitelisted functions: "
                    f"{sorted(ALLOWED_MATH_FUNCS)}"
                )
            if len(node.keywords) > 0:
                raise UnsafeAlphaExpressionError(
                    f"Keyword arguments are forbidden in alpha function '{fn_name}'."
                )

            if fn_name in ALLOWED_UNARY_FUNCS:
                if len(node.args) != 1:
                    raise UnsafeAlphaExpressionError(
                        f"Function '{fn_name}' requires exactly 1 argument, got {len(node.args)}."
                    )
                self._validate_node(node=node.args[0], cfg=cfg, referenced_cols=referenced_cols)
                return

            if fn_name == "signed_power":
                if len(node.args) != 2:
                    raise UnsafeAlphaExpressionError(
                        f"Function 'signed_power' requires exactly 2 arguments (x, a), got {len(node.args)}."
                    )
                self._validate_node(node=node.args[0], cfg=cfg, referenced_cols=referenced_cols)
                exponent_node = node.args[1]
                if not isinstance(exponent_node, ast.Constant):
                    raise UnsafeAlphaExpressionError(
                        "Exponent parameter 'a' in signed_power(x, a) must be a numeric constant."
                    )
                exp_raw = exponent_node.value
                if not isinstance(exp_raw, (int, float)) or isinstance(exp_raw, bool):
                    raise UnsafeAlphaExpressionError(
                        "Exponent parameter 'a' in signed_power(x, a) must be a numeric constant."
                    )
                exp_val = float(exp_raw)
                if exp_val <= 0.0 or exp_val > 10.0:
                    raise UnsafeAlphaExpressionError(
                        f"Exponent 'a' in signed_power must be in range (0, 10], got {exp_val}."
                    )
                return

            if fn_name in ALLOWED_ROLLING_FUNCS:
                if len(node.args) != 2:
                    raise UnsafeAlphaExpressionError(
                        f"Function '{fn_name}' requires exactly 2 arguments (x, d), got {len(node.args)}."
                    )
                self._validate_node(node=node.args[0], cfg=cfg, referenced_cols=referenced_cols)
                window_node = node.args[1]
                if not isinstance(window_node, ast.Constant):
                    raise UnsafeAlphaExpressionError(
                        f"Window parameter 'd' in '{fn_name}(x, d)' must be an integer constant."
                    )
                win_raw = window_node.value
                if not isinstance(win_raw, int) or isinstance(win_raw, bool):
                    raise UnsafeAlphaExpressionError(
                        f"Window parameter 'd' in '{fn_name}(x, d)' must be an integer constant."
                    )
                d = int(win_raw)
                if d < 1:
                    raise UnsafeAlphaExpressionError(
                        f"Causal invariance violation: window d={d} must be >= 1 (strictly positive lag, zero look-ahead)."
                    )
                if d < cfg.min_window_bars or d > cfg.max_window_bars:
                    raise UnsafeAlphaExpressionError(
                        f"Window d={d} in '{fn_name}' violates allowed range [{cfg.min_window_bars}, {cfg.max_window_bars}]."
                    )
                return

            if fn_name == "ts_corr":
                if len(node.args) != 3:
                    raise UnsafeAlphaExpressionError(
                        f"Function 'ts_corr' requires exactly 3 arguments (x, y, d), got {len(node.args)}."
                    )
                self._validate_node(node=node.args[0], cfg=cfg, referenced_cols=referenced_cols)
                self._validate_node(node=node.args[1], cfg=cfg, referenced_cols=referenced_cols)
                window_node = node.args[2]
                if not isinstance(window_node, ast.Constant):
                    raise UnsafeAlphaExpressionError(
                        "Window parameter 'd' in 'ts_corr(x, y, d)' must be an integer constant."
                    )
                win_raw = window_node.value
                if not isinstance(win_raw, int) or isinstance(win_raw, bool):
                    raise UnsafeAlphaExpressionError(
                        "Window parameter 'd' in 'ts_corr(x, y, d)' must be an integer constant."
                    )
                d = int(win_raw)
                if d < 1:
                    raise UnsafeAlphaExpressionError(
                        f"Causal invariance violation: window d={d} must be >= 1 in ts_corr."
                    )
                if d < cfg.min_window_bars or d > cfg.max_window_bars:
                    raise UnsafeAlphaExpressionError(
                        f"Window d={d} in 'ts_corr' violates allowed range [{cfg.min_window_bars}, {cfg.max_window_bars}]."
                    )
                return

    def _to_canonical_str(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id.lower()
        if isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, int) and not isinstance(val, bool):
                return str(val)
            if isinstance(val, (int, float)):
                return str(float(val))
            raise UnsafeAlphaExpressionError(
                f"Cannot canonicalize constant of type {type(val).__name__}"
            )
        if isinstance(node, ast.UnaryOp):
            op_sym = "-" if isinstance(node.op, ast.USub) else "+"
            return f"({op_sym}{self._to_canonical_str(node.operand)})"
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                op_sym = "+"
            elif isinstance(node.op, ast.Sub):
                op_sym = "-"
            elif isinstance(node.op, ast.Mult):
                op_sym = "*"
            elif isinstance(node.op, ast.Div):
                op_sym = "/"
            elif isinstance(node.op, ast.Pow):
                op_sym = "^"
            else:
                raise UnsafeAlphaExpressionError(
                    f"Unsupported binary operator: {type(node.op).__name__}"
                )
            left_str = self._to_canonical_str(node.left)
            right_str = self._to_canonical_str(node.right)
            return f"({left_str} {op_sym} {right_str})"
        if isinstance(node, ast.Call):
            fn_name = node.func.id if isinstance(node.func, ast.Name) else "unknown"
            args_str = ", ".join(self._to_canonical_str(arg) for arg in node.args)
            return f"{fn_name}({args_str})"
        raise UnsafeAlphaExpressionError(f"Cannot canonicalize node type: {type(node).__name__}")
