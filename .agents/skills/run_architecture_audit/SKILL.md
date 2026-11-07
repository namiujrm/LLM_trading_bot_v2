---
name: run_architecture_audit
description: >-
  Audits the codebase against Clean Architecture contracts, Invariant #9 (no defaults/hardcoding),
  and type safety constraints across all layers.
---

# Architecture Audit Runbook (`run_architecture_audit`)

Execute a comprehensive architectural audit:

## 1. Run Import Contracts Audit
```powershell
uv run lint-imports
```
Verifies 4 architectural contracts:
- `domain_is_pure`: Domain layer imports no outer layers or I/O libraries.
- `ports_depend_only_on_domain`: Ports only import from Domain.
- `application_depends_only_on_domain_and_ports`: Application uses only Domain and Ports.
- `adapters_isolation`: Concrete adapters never cross-import each other.

## 2. Run Invariant #9 AST Scanner
```powershell
uv run python scripts/check_no_defaults_or_hardcode.py
```
Checks for:
- Positional & keyword-only default parameters (`[NO_DEFAULT_FUNC_PARAM]`, `[NO_DEFAULT_KW_PARAM]`).
- Default values in classes / Pydantic models (`[NO_DEFAULT_CLASS_FIELD]`).
- Silent fallback patterns (`[NO_DICT_GET_FALLBACK]`, `[NO_TERNARY_FALLBACK_DEFAULT]`, `[NO_OR_FALLBACK_DEFAULT]`).

## 3. Run Strict Type Checking
```powershell
uv run pyright src tests scripts
```
Ensures 100% strict type safety with zero type errors.
