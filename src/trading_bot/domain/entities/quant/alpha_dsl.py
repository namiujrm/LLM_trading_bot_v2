"""Symbolic Alpha DSL Domain Entities (Zero Defaults).

Pure Python domain models representing parsed and validated Alpha AST representations.
"""

from __future__ import annotations

import ast


class AlphaFormulaAST:
    """Domain entity representing a validated, canonicalized Alpha DSL AST."""

    def __init__(
        self,
        raw_formula: str,
        canonical_formula: str,
        ast_hash: str,
        node_count: int,
        max_depth: int,
        referenced_columns: tuple[str, ...],
        root_node: ast.Expression,
    ) -> None:
        self.raw_formula: str = raw_formula
        self.canonical_formula: str = canonical_formula
        self.ast_hash: str = ast_hash
        self.node_count: int = node_count
        self.max_depth: int = max_depth
        self.referenced_columns: tuple[str, ...] = referenced_columns
        self.root_node: ast.Expression = root_node

    def __repr__(self) -> str:
        return (
            f"AlphaFormulaAST(canonical='{self.canonical_formula}', "
            f"hash='{self.ast_hash[:8]}', nodes={self.node_count}, depth={self.max_depth})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AlphaFormulaAST):
            return False
        return self.ast_hash == other.ast_hash

    def __hash__(self) -> int:
        return hash(self.ast_hash)
