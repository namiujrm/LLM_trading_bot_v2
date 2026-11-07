# Project: LLM_trading_bot_v2 (Production Foundation)

## Architecture Overview
The system implements a strict Clean Architecture under `src/trading_bot/`:
1. **Domain** (`src/trading_bot/domain/`): Pure business entities, value objects, domain invariants, and pure deterministic services. Strictly zero dependencies on outer layers or external I/O libraries.
2. **Ports** (`src/trading_bot/ports/`): Abstract interface definitions (`MarketDataPort`, `MemoryRepositoryPort`, `ExecutionGatewayPort`). Depends ONLY on Domain.
3. **Application** (`src/trading_bot/application/`): Use cases and orchestration DTOs. Depends ONLY on Domain and Ports.
4. **Adapters** (`src/trading_bot/adapters/`): Concrete technical adapters (persistence, binance, telegram). Sub-adapters are strictly isolated from each other.
5. **Infrastructure** (`src/trading_bot/infrastructure/`): Low-level platform utilities, logging, connection management.
6. **Interfaces** (`src/trading_bot/interfaces/`): User/external entry points (CLI, main entry).

### Invariant #9 Absolute Constraint
- Zero Hardcoding, Zero Default Parameters (`def f(x=1):` is strictly prohibited).
- Zero Default Fields in classes, Pydantic `BaseModel`, and `@dataclass` (all fields require explicit type annotation with no assigned default or `= Field(...)`).
- Zero Silent Fallbacks (`x or default`, `dict.get(k, default)`, `x if cond else default` are prohibited; must raise explicit exceptions on missing data).
- Enforced automatically via AST scanner: `scripts/check_no_defaults_or_hardcode.py`.

---

## Feature Inventory

| # | Feature | Description | Milestone | Owner | Source | Status |
|---|---------|-------------|-----------|-------|--------|--------|
| 1 | Clean Architecture Directory Tree | `src/trading_bot/{domain,ports,application,adapters,infrastructure,interfaces}`, `tests/{unit,property,contract,architecture,integration}` | M1 | Agent 1 | User Decision Gate 1 | **DONE** |
| 2 | Python & Tooling Environment (`pyproject.toml`) | `uv`, `ruff`, `basedpyright` (`typeCheckingMode = "strict"`), `import-linter`, `pytest`, `hypothesis` | M1 | Agent 1 | User Decision Gate 1 | **DONE** |
| 3 | Import-Linter 4 Contracts | `domain_is_pure`, `ports_depend_only_on_domain`, `application_depends_only_on_domain_and_ports`, `adapters_isolation` | M1 | Agent 1 | User Decision Gate 1 | **DONE (4 kept, 0 broken)** |
| 4 | Vibecoding Agent Harness | `AGENTS.md` hierarchy (root, `domain/AGENTS.md`, `adapters/AGENTS.md`) and 3 Skills (`add_domain_port`, `run_architecture_audit`, `verify_financial_math`) | M1 | Agent 1 | User Decision Gate 1 | **DONE** |
| 5 | AST Invariant #9 Scanner | `scripts/check_no_defaults_or_hardcode.py` catching 6 violation codes + unit tests | M1 | Agent 1 | User Decision Gate 1 | **DONE (0 violations)** |
| 6 | Dev Verification Loop Scripts & Hooks | `scripts/verify_loop.ps1` (5-stage verification gate) and `.agents/hooks.json` | M1 | Agent 1 | User Decision Gate 1 | **DONE** |
| 7 | Pure Domain Entities, Value Objects & Services | `DeterministicRiskCalculator`, `PlaybookValidator`, `BarrierEvaluator` + Hypothesis property tests | M1 | Agent 1 | User Decision Gate 1 | **DONE** |
| 8 | Domain Ports & Explicit Config Schemas | `MarketDataPort`, `MemoryRepositoryPort`, `ExecutionGatewayPort`, `DatabaseConfig`, `RedisConfig`, `BinanceConfig`, `.env.example` | M1 | Agent 1 | User Decision Gate 1 | **DONE** |
| 9 | Docker Compose Data Infrastructure | `timescale/timescaledb-ha:pg16` + `redis:7-alpine` with healthchecks and persistent volumes | M2A | Agent 2 | User Decision Gate 1 | **DONE (healthy)** |
| 10 | TimescaleDB SQL Schema (`sql/001_init_schema.sql`) | 5 tables (`market_klines_1m_enriched`, `quant_feature_vectors_30m`, `orders_and_positions`, `shadow_orders_lifecycle`, `episodic_lessons_memory` with `VECTOR(16)` HNSW index) + Continuous Aggregates (`15m`, `30m`, `1h`, `4h`) | M2A | Agent 2 | User Decision Gate 1 | **DONE (verified)** |
| 11 | AccountPortfolioContext Domain Delta | `account_state.py`, `VetoCategory`, `ExistingPositionDirective`, `trade_blueprint.py` updates, `DeterministicRiskCalculator` margin sizing & arbiter multiplier, Port updates | M1-Delta | Agent 1 | User Decision Gate 2 | **DONE** |
| 12 | TimescaleDB SQL Delta & Account Snapshots | `account_equity_snapshots` hypertable (30d), delta columns in `orders_and_positions` and `shadow_orders_lifecycle`, applied to live container | M2-Delta | Agent 2 | User Decision Gate 2 | **DONE (verified)** |
| 13 | Database & Redis Adapters (T2.3) | `database.py` (asyncpg pool), `redis_client.py` with `ACQUIRE_ROUTE_LUA` and live context cache | M2B | Agent 2 | User Decision Gate 2 | **DONE** |
| 14 | Market Data & Portfolio Repository (T2.4) | `MarketDataRepository` (implementing `MarketDataPort`) and `AccountPortfolioRepository` (builder for `AccountPortfolioContext`) | M2B | Agent 2 | User Decision Gate 2 | **DONE** |
| 15 | Episodic Memory pgvector Adapter (T2.5) | `PgvectorMemoryRepository` implementing `MemoryRepositoryPort` with 16D regime trajectory vector | M2B | Agent 2 | User Decision Gate 2 | **DONE** |
| 16 | WebSocket Ingestor Pipeline (T2.6) | `BinanceFuturesWSIngestor` & `MinuteMicroAccumulator` (OFI, OBI, Spreads, Liquidations) | M2B | Agent 2 | User Decision Gate 2 | **DONE** |
| 17 | Historical Backfiller & Live E2E Tests (T2.7) | `HistoricalBackfiller` (ZIP streaming in RAM, binary COPY to DB, Zstandard Parquet export) + integration tests | M2B | Agent 2 | User Decision Gate 2 | **DONE** |

---

## Milestones

| # | Milestone Name | Scope | Dependencies | Status | Key Outputs |
|---|----------------|-------|-------------|--------|-------------|
| M1 | Project Skeleton & Vibecoding Harness | Tasks T1.1–T1.8 | None | **DONE** | 43 files, 25 tests pass, 5 gates pass 100% |
| M2A | Independent Database Infrastructure & Schema | Tasks T2.1–T2.2 | None | **DONE** | Containers up & healthy, schema verified |
| M1-Delta | AccountPortfolioContext & Domain Updates | Task T1.9: `account_state.py`, `enums.py`, `trade_blueprint.py`, `risk_calculator.py`, Ports, tests | M1 | **DONE** | Domain context & margin risk controls |
| M2B | SQL Delta & Persistence/Ingestion Adapters | Tasks T2.3–T2.7: Schema migration, DB/Redis adapters, MarketDataRepo, MemoryRepo, WS Ingestor, Backfiller | M1-Delta, M2A | **DONE** | 56 files, 44 tests pass, live Docker verified |
| M3 | Final Integrated Verification & Acceptance | Full 5-stage verification loop + live integration tests against Docker containers | M1-Delta, M2B | **PAUSED** | Pausing at Decision Gate 3 for User Approval |

---

## Code Layout

```
LLM_trading_bot_v2/
├── .agents/
│   ├── hooks.json
│   ├── skills/
│   │   ├── add_domain_port/SKILL.md
│   │   ├── run_architecture_audit/SKILL.md
│   │   └── verify_financial_math/SKILL.md
│   └── teamwork/
├── scripts/
│   ├── check_no_defaults_or_hardcode.py
│   ├── verify_stop_hook.py
│   ├── verify_loop.ps1
│   └── verify_loop.sh
├── sql/
│   └── 001_init_schema.sql
├── src/
│   └── trading_bot/
│       ├── domain/
│       │   ├── AGENTS.md
│       │   ├── entities/
│       │   │   ├── account_state.py
│       │   │   ├── enums.py
│       │   │   ├── market_data.py
│       │   │   ├── trade_blueprint.py
│       │   │   └── episodic_memory.py
│       │   ├── value_objects/
│       │   └── services/
│       │       ├── risk_calculator.py
│       │       ├── playbook_validator.py
│       │       └── barrier_evaluator.py
│       ├── ports/
│       │   ├── market_data_port.py
│       │   ├── memory_repository_port.py
│       │   └── execution_gateway_port.py
│       ├── application/
│       ├── adapters/
│       │   ├── AGENTS.md
│       │   ├── persistence/
│       │   │   ├── database.py
│       │   │   ├── redis_client.py
│       │   │   ├── market_data_repo.py
│       │   │   ├── account_portfolio_repo.py
│       │   │   └── pgvector_memory_repo.py
│       │   └── binance/
│       │       ├── ws_micro_ingestor.py
│       │       ├── micro_accumulator.py
│       │       └── historical_backfiller.py
│       ├── infrastructure/
│       │   └── config_schemas.py
│       └── interfaces/
├── tests/
│   ├── unit/
│   ├── property/
│   ├── contract/
│   ├── architecture/
│   └── integration/
├── docker-compose.yml
├── pyproject.toml
├── .env.example
├── .gitignore
└── AGENTS.md
```
