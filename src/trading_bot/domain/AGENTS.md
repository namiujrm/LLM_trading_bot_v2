# Domain Layer Rules (`src/trading_bot/domain/`)

1. **100% Pure Python & Pydantic:**
   - In `src/trading_bot/domain/`, only standard library modules (`math`, `enum`, `typing`, `dataclasses`, `datetime`, `abc`, `uuid`) and `pydantic` are allowed.
   - Strictly FORBIDDEN to import: `polars`, `numpy`, `lightgbm`, `catboost`, `httpx`, `redis`, `asyncpg`, `telegram`, or any outer layer modules (`ports`, `application`, `adapters`, `infrastructure`, `interfaces`).

2. **Absolute Zero Default Values & Zero Hardcoding In Entities & Services:**
   - All fields in `BaseModel` and `@dataclass` are Required Fields (pure type annotations without `= ...` or `= Field(...)`).
   - All quantitative thresholds and risk limits in `DeterministicRiskCalculator`, `PlaybookValidator`, and `BarrierEvaluator` MUST be accessed from explicitly injected configuration objects (`self.cfg.<param_name>`).
   - Magic numbers and literals embedded inside function bodies are strictly prohibited.

3. **Deterministic Financial Math:**
   - All mathematical formulas (Quarter-Kelly, fee calculations, risk sizing, barrier crossing) must be purely functional and deterministic without random or LLM-based estimation.
