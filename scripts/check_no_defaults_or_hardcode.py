"""AST Static Gate: Enforce Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


class ZeroDefaultAndHardcodeVisitor(ast.NodeVisitor):
    def __init__(self, file_path: Path) -> None:
        self.file_path: Path = file_path
        self.violations: list[str] = []
        self._in_enum_class: bool = False

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        base_names: set[str] = {
            b.id if isinstance(b, ast.Name) else (b.attr if isinstance(b, ast.Attribute) else "")
            for b in node.bases
        }
        prev_in_enum: bool = self._in_enum_class
        self._in_enum_class = any("Enum" in name for name in base_names)

        # In non-Enum classes (BaseModel, Dataclass, standard class), prohibit default field assignments
        if not self._in_enum_class:
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                    target_name = stmt.target.id if isinstance(stmt.target, ast.Name) else "field"
                    self.violations.append(
                        f"{self.file_path}:{stmt.lineno} — [NO_DEFAULT_CLASS_FIELD] "
                        f"Field `{node.name}.{target_name}` must not have a default value. "
                        f"Explicitly inject values via constructor or config schema."
                    )
                elif isinstance(stmt, ast.Assign):
                    for t in stmt.targets:
                        target_name = t.id if isinstance(t, ast.Name) else "target"
                        self.violations.append(
                            f"{self.file_path}:{stmt.lineno} — [NO_DEFAULT_CLASS_FIELD] "
                            f"Class attribute `{node.name}.{target_name}` must not have a default assignment."
                        )

        self.generic_visit(node)
        self._in_enum_class = prev_in_enum

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_func_args(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_func_args(node)
        self.generic_visit(node)

    def _check_func_args(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # 1. Prohibit positional default arguments: def fn(a: int = 10)
        if len(node.args.defaults) > 0:
            self.violations.append(
                f"{self.file_path}:{node.lineno} — [NO_DEFAULT_FUNC_PARAM] "
                f"Function `{node.name}()` contains positional default parameters. "
                f"All arguments must be explicitly passed by the caller."
            )
        # 2. Prohibit keyword-only default arguments: def fn(*, a: int = 10)
        if any(d is not None for d in node.args.kw_defaults):
            self.violations.append(
                f"{self.file_path}:{node.lineno} — [NO_DEFAULT_KW_PARAM] "
                f"Function `{node.name}()` contains keyword-only default parameters."
            )

    def visit_Call(self, node: ast.Call) -> None:
        # 3. Prohibit dict.get("key", default_value)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get" and len(node.args) >= 2:
            self.violations.append(
                f"{self.file_path}:{node.lineno} — [NO_DICT_GET_FALLBACK] "
                f"Prohibited `.get(key, default)` fallback. Access directly or raise explicit exception."
            )
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        # 4. Prohibit ternary fallback: `val if val is not None else HARD_CONSTANT` or `x if x else CONST`
        if isinstance(node.orelse, (ast.Constant, ast.UnaryOp)):
            is_none_check = isinstance(node.test, ast.Compare) and any(
                isinstance(op, (ast.Is, ast.IsNot)) for op in node.test.ops
            )
            is_truthy_name = isinstance(node.test, ast.Name)
            if is_none_check or is_truthy_name:
                self.violations.append(
                    f"{self.file_path}:{node.lineno} — [NO_TERNARY_FALLBACK_DEFAULT] "
                    f"Prohibited ternary fallback expression `... else <hard_val>`. Raise exception on missing data."
                )
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # 5. Prohibit boolean Or fallback: `val = raw_val or 0.50`
        if isinstance(node.op, ast.Or) and isinstance(node.values[-1], (ast.Constant, ast.UnaryOp)):
            self.violations.append(
                f"{self.file_path}:{node.lineno} — [NO_OR_FALLBACK_DEFAULT] "
                f"Prohibited silent fallback `x or <hard_val>`."
            )
        self.generic_visit(node)


def scan_directory(target_dir: Path, repo_root: Path) -> list[str]:
    violations: list[str] = []
    for py_file in sorted(target_dir.rglob("*.py")):
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except Exception as exc:
            violations.append(f"{py_file}:1 — [PARSE_ERROR] Failed to parse AST: {exc}")
            continue

        rel_path = py_file.relative_to(repo_root) if py_file.is_relative_to(repo_root) else py_file
        visitor = ZeroDefaultAndHardcodeVisitor(rel_path)
        visitor.visit(tree)
        violations.extend(visitor.violations)
    return violations


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    target_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else repo_root / "src"

    if not target_path.exists():
        print(f"Target path does not exist: {target_path}. Skipping AST check.")
        sys.exit(0)

    if target_path.is_file():
        violations = scan_directory(target_path.parent, repo_root)
    else:
        violations = scan_directory(target_path, repo_root)

    if violations:
        print("=== [AST GATE FAILED] Invariant #9 Violations Detected ===")
        for v in violations:
            print(f"  [X] {v}")
        sys.exit(1)

    print("=== [AST GATE PASSED] 0 Default Parameters, 0 Default Fields, 0 Silent Fallbacks ===")


if __name__ == "__main__":
    main()
