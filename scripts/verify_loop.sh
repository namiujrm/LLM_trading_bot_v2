#!/usr/bin/env bash
set -euo pipefail

echo "=== [1/5] Running Ruff Linter & Formatter Check ==="
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts

echo "=== [2/5] Running Pyright Static Type Checker ==="
uv run pyright src tests scripts

echo "=== [3/5] Running AST Zero-Default & Anti-Hardcode Gate ==="
uv run python scripts/check_no_defaults_or_hardcode.py

echo "=== [4/5] Running Import-Linter Clean Architecture Contracts ==="
uv run lint-imports

echo "=== [5/5] Running Pytest Unit, Property, and Contract Suite ==="
uv run pytest tests -v

echo "=== ALL 5 VERIFICATION GATES PASSED 100% ==="
