# Global Architecture & Development Rules — LLM_trading_bot_v2

## 1. Clean Architecture Strict Hierarchy
- The project strictly follows 4-layer Clean Architecture:
  `src/trading_bot/domain/` <- `src/trading_bot/ports/` <- `src/trading_bot/application/` <- `src/trading_bot/adapters/` & `infrastructure/` & `interfaces/`.
- The Domain layer is 100% pure Python and Pydantic: no I/O, no network calls, no database access, no third-party heavy computational dependencies (`polars`, `numpy`, `lightgbm`, `httpx`, `redis`, `asyncpg`, `telegram`).
- Inter-layer communication MUST flow through Domain Ports (`Protocol` / `ABC`) defined in `src/trading_bot/ports/`.
- All `[tool.importlinter]` contracts (`domain_is_pure`, `ports_depend_only_on_domain`, `application_depends_only_on_domain_and_ports`, `adapters_isolation`) MUST be kept 100%.

## 2. Invariant #9: Zero Defaults, Zero Hardcoding, Zero Silent Fallbacks (Fail-Fast Discipline)
1. **Zero Default Parameters in Functions and Methods (`[NO_DEFAULT_FUNC_PARAM]`, `[NO_DEFAULT_KW_PARAM]`):**
   - Strictly prohibited: `def fn(x: int = 10):` or `def fn(*, timeout: float = 5.0):`.
   - Every parameter must have an explicit type annotation and must be passed explicitly by the caller or injected via a Config object.
2. **Zero Default Values in Class / Dataclass / Pydantic Fields (`[NO_DEFAULT_CLASS_FIELD]`):**
   - In classes, `@dataclass`, and Pydantic `BaseModel`, all fields must be required type annotations (e.g., `maker_fee_rate: float`, `rejection_reason: str | None`).
   - Gaining default values via `= ...` or `= Field(...)` is strictly forbidden.
   - Exception: `Enum` classes inheriting from `Enum`.
3. **Zero Silent Fallback Expressions (`[NO_DICT_GET_FALLBACK]`, `[NO_TERNARY_FALLBACK_DEFAULT]`, `[NO_OR_FALLBACK_DEFAULT]`):
   - Prohibited patterns:
     - ❌ `val = payload.get("key", default_val)` (use `payload["key"]` or raise explicit exception)
     - ❌ `val = x if x is not None else 0.5`
     - ❌ `val = raw_val or 1.0`
   - When required data is missing, raise an explicit typed domain exception immediately (Fail-Fast).
4. **Zero Hardcoded Magic Literals:**
   - Quantitative risk thresholds, model names, endpoints, and multipliers must be loaded from explicit configuration schemas.

## 3. Quantitative & Deterministic Invariants
- **Zero LLM Arithmetic:** LLMs never compute sizing, probability calibrations, or indicators. All math resides in deterministic domain services or quant engine.
- **Zero Look-Ahead Bias:** Causal temporal ordering is strictly preserved. No negative shifts (`shift(-k)` with `k > 0`).

## 4. Mandatory 5-Stage Verification Gate (`verify_loop`)
Before completing any task or reporting progress, you must pass all 5 gates:
1. `uv run ruff check src tests scripts` & `uv run ruff format --check src tests scripts`
2. `uv run pyright src tests scripts` (strict type checking)
3. `uv run python scripts/check_no_defaults_or_hardcode.py` (AST gate)
4. `uv run lint-imports` (4 architecture contracts)
5. `uv run pytest tests -v` (unit, property, and contract tests)
