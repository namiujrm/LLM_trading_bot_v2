# 01 — Thiết Kế Khung Dự Án Clean Architecture & Vibecoding Harness

> [!IMPORTANT]
> **Mục tiêu Pattern 1:** Thiết lập nền móng **Clean Architecture 4 tầng** bất biến kết hợp bộ công cụ **Full Autonomous Vibecoding Harness** (`uv`, `import-linter`, `AGENTS.md` phân tầng, `3 Workspace Skills`, và `.agents/hooks.json` `Stop` Hook). Mục đích là đảm bảo mỗi khi AI Agent viết hoặc sửa bất kỳ dòng code nào, hệ thống tự động kiểm tra: **(1) Format & Lint (`ruff`)**, **(2) Kiểu dữ liệu tĩnh (`pyright`)**, **(3) Ranh giới kiến trúc (`lint-imports`)**, và **(4) Toàn bộ Unit Tests chống Look-ahead Bias (`pytest`)** trước khi cho phép kết thúc lượt.

---

## 1. Cấu Trúc Thư Mục Chi Tiết Toàn Dự Án (`Directory Tree`)

```text
LLM_trading_bot_v2/
├── AGENTS.md                                       # [Rule Gốc] Quy tắc toàn cục & bắt buộc chạy verify_loop
├── pyproject.toml                                  # Quản lý bằng uv: dependencies, ruff, pyright, import-linter, pytest
├── uv.lock                                         # Lockfile tự động sinh bởi uv
├── .env.example                                    # Mẫu biến môi trường (Binance, Postgres, Redis, 9router, Telegram)
├── docker-compose.yml                              # Khởi chạy TimescaleDB+pgvector, Redis 7, 9router, TradingBot
│
├── .agents/                                        # Hệ thống Customization cho Vibecoding tự hành
│   ├── hooks.json                                  # Lifecycle Stop Hook chặn kết thúc lượt nếu verify_loop lỗi
│   └── skills/
│       ├── verify-loop/
│       │   └── SKILL.md                            # Skill chạy 4 bước kiểm증: ruff -> pyright -> import-linter -> pytest
│       ├── add-quant-alpha/
│       │   └── SKILL.md                            # Skill thêm chỉ số nghiên cứu / Symbolic Alpha chuẩn chống Look-ahead
│       └── new-clean-slice/
│           └── SKILL.md                            # Skill tạo tính năng dọc từ Domain Port -> UseCase -> Adapter
│
├── scripts/
│   ├── verify_loop.ps1                             # Script chạy kiểm định 5 bước trên Windows PowerShell
│   ├── verify_loop.sh                              # Script chạy kiểm định 5 bước trên Linux / Docker
│   ├── check_no_defaults_or_hardcode.py            # [AST Gate] Chặn đứng default params, fallback ngầm & hardcode
│   ├── verify_stop_hook.py                         # Handler Python cho .agents/hooks.json (Stop event)
│   └── backfill_historical.py                      # CLI tải dữ liệu lịch sử từ data.binance.vision & REST
│
├── config/
│   ├── settings.yaml                               # Cấu hình thị trường, khung thời gian, hạn mức rủi ro
│   ├── rate_limits.yaml                            # Cấu hình RPM/TPM/RPD/min_interval_ms cho từng API sau 9router
│   └── alpha_zoo_seed.yaml                         # Danh mục hạt giống Research Indicators & WorldQuant Alphas
│
├── docs/
│   └── system_design/                              # Bộ 9 tài liệu thiết kế hệ thống (00 -> 08)
│
├── src/
│   ├── __init__.py
│   │
│   ├── domain/                                     # === TẦNG 1: DOMAIN (100% Pure Python + Pydantic, 0 I/O) ===
│   │   ├── AGENTS.md                               # [Rule Tầng Domain] Cấm import thư viện ngoài, khóa công thức Risk
│   │   ├── __init__.py
│   │   ├── entities/
│   │   │   ├── __init__.py
│   │   │   ├── enums.py                            # Side, RegimeType, PlaybookType, OIQuadrant, OutcomeCategory
│   │   │   ├── market_state.py                     # MicroBar1m, MultiTFBar, FeatureSignalItem, SemanticQuantPacket
│   │   │   ├── ml_prediction.py                    # DualBinaryPrediction, VennAbersInterval, SHAPFactor, OODState
│   │   │   ├── trade_blueprint.py                  # MarketThesis, TradeBlueprint, RiskVerdict, BracketOrderTicket
│   │   │   ├── shadow_order.py                     # ShadowOrder, BarrierTouchResult, PostSLAutopsyPath
│   │   │   ├── episodic_case.py                    # EpisodicCase (2-Tier), CohortBaseRateSummary, StratifiedMemoryPack
│   │   │   └── alpha_candidate.py                  # SymbolicAlphaSpec, GatekeeperReport, ChallengerComparison
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── risk_calculator.py                  # Fee-to-Edge Gate, Uncertainty Quarter-Kelly, Correlation Cap, DD Breaker
│   │   │   ├── playbook_validator.py               # Kiểm tra điều kiện biên bằng code cho 4 Pre-Verified Playbooks
│   │   │   └── barrier_evaluator.py                # Logic xét chạm biên Triple-Barrier & tính MAE/MFE trên chuỗi 1m
│   │   └── ports/
│   │       ├── __init__.py
│   │       ├── market_data_port.py                 # Interface đọc/ghi nến, OI, Funding, Parquet
│   │       ├── quant_engine_port.py                # Interface tính SemanticQuantPacket & Symbolic Alphas
│   │       ├── ml_model_port.py                    # Interface dự báo Dual-Binary Causal GBDT & SHAP
│   │       ├── agent_gateway_port.py               # Interface gọi 9router có kiểm soát Rate-Limit & Structured Output
│   │       ├── memory_repository_port.py           # Interface truy xuất k-NN Cohort (K=20) & 6 Stratified Lessons
│   │       ├── execution_gateway_port.py           # Interface đặt lệnh Bracket Order (Live / Testnet / Paper)
│   │       └── notification_port.py                # Interface gửi báo cáo & nhận lệnh điều khiển Telegram
│   │
│   ├── application/                                # === TẦNG 2: APPLICATION USE CASES (Điều phối nghiệp vụ) ===
│   │   ├── __init__.py
│   │   ├── dto/
│   │   │   ├── __init__.py
│   │   │   └── cycle_report_dto.py                 # DTO tổng hợp kết quả chu kỳ 30m gửi ra Telegram & DB
│   │   └── use_cases/
│   │       ├── __init__.py
│   │       ├── run_30m_trading_cycle.py            # Điều phối BTC-Lead Cascade -> ML -> 3 Agents -> Risk -> Execution/Shadow
│   │       ├── evaluate_open_and_shadow_orders.py  # Quét nến 1m cập nhật MAE/MFE, chạm biên TP/SL & theo dõi Post-SL
│   │       ├── reflect_and_update_memory.py        # Chạy Post-Mortem Judge tạo EpisodicCase 2 tầng & cập nhật Confidence
│   │       ├── run_weekend_alpha_mining.py         # Chạy Math Agent + 4-Stage Gatekeeper + Champion/Challenger
│   │       └── handle_emergency_controls.py        # Xử lý Kill-Switch, Breakeven SL, Pause/Resume, Promote Model
│   │
│   ├── infrastructure/                             # === TẦNG 3: INFRASTRUCTURE (Triển khai kỹ thuật chi tiết) ===
│   │   ├── __init__.py
│   │   ├── quant_engine/
│   │   │   ├── AGENTS.md                           # [Rule Quant Engine] Bắt buộc Causal Polars/Numba, cấm shift(-k)
│   │   │   ├── __init__.py
│   │   │   ├── microstructure.py                   # VPIN, OFI, OBI, Kyle's Lambda, Taker CVD Z-Score
│   │   │   ├── regime_physics.py                   # Hurst (Numba R/S), Yang-Zhang Vol, Shannon Entropy, FracDiff
│   │   │   ├── derivatives_alpha.py                # OI x Price 4-Quadrant, Funding Z-Score, BTC Lead-Lag Beta
│   │   │   ├── symbolic_dsl.py                     # Trình biên dịch & kiểm định biểu thức Polars AST (WorldQuant + Math Agent)
│   │   │   └── packet_builder.py                   # Lọc Rolling 30d Rank IC >= 0.015 & đóng gói SemanticQuantPacket
│   │   ├── ml/
│   │   │   ├── __init__.py
│   │   │   ├── triple_barrier_labeler.py           # Gán nhãn y_long_win, y_short_win có trừ phí & trượt giá trên nến 1m
│   │   │   ├── purged_cv.py                        # Purged K-Fold Cross-Validation & Embargo
│   │   │   ├── mahalanobis_ood.py                  # Ledoit-Wolf Mahalanobis Out-of-Distribution Detector
│   │   │   ├── venn_abers.py                       # Inductive Venn-Abers Calibrator (P_cal + Uncertainty Width)
│   │   │   ├── dual_binary_gbdt.py                 # Long Model & Short Model (Ridge Prior + Monotonic LightGBM/CatBoost)
│   │   │   └── shap_explainer.py                   # Trích xuất Economic-Locked TreeSHAP
│   │   ├── agents/
│   │   │   ├── AGENTS.md                           # [Rule Agents] Bắt buộc qua Rate-Limit Governor, Structured Pydantic
│   │   │   ├── __init__.py
│   │   │   ├── rate_limit_governor.py              # Bộ điều phối Sliding-Window RPM/TPM/RPD/Interval trên Redis
│   │   │   ├── ninerouter_client.py                # Adapter gọi 9router, tách thẻ <think>, tự phục hồi JSON Schema
│   │   │   ├── macro_news_sentinel.py              # Dịch vụ nền canh lịch CPI/FOMC & lọc tin SimHash vào SharedMacroCache
│   │   │   ├── scout_agent.py                      # Agent 1: Market Scout (Khớp 4 Playbooks)
│   │   │   ├── architect_agent.py                  # Agent 2: Trade Architect (Tối ưu Entry 15m & SL theo Cohort MAE)
│   │   │   ├── arbiter_agent.py                    # Agent 3: Semantic Risk Arbiter (Kiểm toán IC & phản biện)
│   │   │   ├── post_mortem_agent.py                # Giải phẫu lệnh đóng/lệnh ảo tạo EpisodicCase 2 tầng
│   │   │   └── math_alpha_agent.py                 # Weekend Math Agent sinh công thức Symbolic AST
│   │   ├── binance/
│   │   │   ├── __init__.py
│   │   │   ├── ws_micro_ingestor.py                # Nhận luồng aggTrade, depth20@100ms, kline_1m, forceOrder -> gom nến 1m
│   │   │   ├── futures_rest_gateway.py             # Đặt lệnh Bracket Order (Entry + Hard SL + TP), lấy OI & Funding
│   │   │   ├── historical_backfiller.py            # Tải ZIP từ data.binance.vision & đồng bộ Parquet
│   │   │   └── paper_simulator.py                  # Khớp lệnh giả lập có tính phí & trượt giá Kyle's Lambda trên nến 1m
│   │   └── persistence/
│   │       ├── __init__.py
│   │       ├── database.py                         # AsyncPG / SQLAlchemy Async Engine kết nối TimescaleDB
│   │       ├── redis_store.py                      # Quản lý trạng thái WebSocket, SharedMacroCache & Quota
│   │       ├── market_data_repo.py                 # Đọc/ghi Hypertables & đồng bộ Parquet Feature Store
│   │       └── pgvector_memory_repo.py             # Truy xuất k-NN Cohort (K=20) & 6 Stratified Multi-Bucket Cases
│   │
│   └── presentation/                               # === TẦNG 4: PRESENTATION & ENTRYPOINT ===
│       ├── __init__.py
│       ├── telegram/
│       │   ├── __init__.py
│       │   ├── bot_controller.py                   # Xử lý /status, /quota, /mode, /positions, /shadow, /panic_close
│       │   ├── keyboards.py                        # Inline Keyboards (Kill-Switch, Breakeven SL, Approve Challenger)
│       │   └── formatters.py                       # Render báo cáo nến 30m, SHAP, Cohort 20-Case & Debate Digest
│       ├── scheduler/
│       │   ├── __init__.py
│       │   └── orchestrator_loop.py                # Nhịp đóng nến 30m, quét lệnh 1m, Macro Sentinel 5m, Math Agent Chủ Nhật
│       └── main.py                                 # Dependency Injection Wiring & Khởi động toàn hệ thống
│
└── tests/
    ├── conftest.py                                 # Fixtures dữ liệu nến mẫu, mock Redis/9router/Binance
    ├── unit/
    │   ├── test_domain_risk_and_playbooks.py       # Test công thức Quarter-Kelly, Fee Gate, 4 Playbooks, Drawdown Breaker
    │   ├── test_quant_no_lookahead.py              # Test bất biến thời gian (Zero Look-ahead) cho Quant Engine & Symbolic AST
    │   ├── test_causal_ml_monotonicity.py          # Test khóa đơn điệu Long/Short Model, Mahalanobis OOD, Venn-Abers
    │   ├── test_rate_limit_governor.py             # Test giới hạn RPM/TPM/RPD/min_interval_ms và tự xoay vòng API
    │   └── test_cohort_memory_and_autopsy.py       # Test thống kê K=20 Cohort, chia 6 Bucket & phát hiện Stop-Hunt
    └── integration/
        └── test_full_30m_cycle_paper.py            # Test luồng End-to-End từ nến 30m -> ML -> 3 Agents -> Paper Bracket Order
```

---

## 2. Đặc Tả File Cấu Hình Trung Tâm `pyproject.toml` (Quản Lý Bằng `uv`)

File `pyproject.toml` tích hợp sẵn cấu hình của **`ruff`**, **`pyright`**, **`pytest`**, và **`import-linter`** để khóa chặt biên giới Clean Architecture:

```toml
[project]
name = "llm-trading-bot-v2"
version = "0.1.0"
description = "Quant-LLM Multi-Agent Crypto Trading System with Causal GBDT, 9router & 2-Tier Episodic Memory"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.9.0",
    "pydantic-settings>=2.5.0",
    "pyyaml>=6.0.2",
    "polars>=1.9.0",
    "pyarrow>=17.0.0",
    "numpy>=2.0.0",
    "scipy>=1.14.0",
    "numba>=0.60.0",
    "scikit-learn>=1.5.2",
    "lightgbm>=4.5.0",
    "catboost>=1.2.7",
    "shap>=0.46.0",
    "httpx>=0.27.2",
    "websockets>=13.1",
    "asyncpg>=0.30.0",
    "pgvector>=0.3.5",
    "redis>=5.1.1",
    "python-telegram-bot>=21.6",
    "apscheduler>=3.10.4",
    "structlog>=24.4.0",
]

[dependency-groups]
dev = [
    "pytest>=8.3.3",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=5.0.0",
    "ruff>=0.7.0",
    "pyright>=1.1.385",
    "import-linter>=2.1",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]

# =====================================================================
# 1. CẤU HÌNH RUFF (LINTER & FORMATTER SIÊU TỐC)
# =====================================================================
[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
    "SIM", # flake8-simplify
    "RUF", # Ruff-specific rules
]
ignore = ["E501"]

# =====================================================================
# 2. CẤU HÌNH PYRIGHT (STATIC TYPE CHECKING NGHIÊM NGẶT)
# =====================================================================
[tool.pyright]
include = ["src", "tests"]
pythonVersion = "3.11"
typeCheckingMode = "standard"
reportMissingImports = true
reportMissingTypeStubs = false
reportUnusedImport = true
reportUnusedVariable = true

# =====================================================================
# 3. CẤU HÌNH PYTEST
# =====================================================================
[tool.pytest.ini_options]
minversion = "8.0"
addopts = "-ra -q --strict-markers"
testpaths = ["tests"]
pythonpath = ["."]
asyncio_mode = "auto"

# =====================================================================
# 4. CẤU HÌNH IMPORT-LINTER (KHÓA CỨNG RANH GIỚI CLEAN ARCHITECTURE)
# =====================================================================
[tool.importlinter]
root_packages = ["src"]
include_external_packages = true

[[tool.importlinter.contracts]]
name = "Contract 1: Strict Layered Clean Architecture (Presentation -> Infrastructure/Application -> Domain)"
type = "layers"
layers = [
    "src.presentation",
    "src.infrastructure",
    "src.application",
    "src.domain",
]

[[tool.importlinter.contracts]]
name = "Contract 2: Pure Domain Isolation (Cấm Domain import thư viện I/O, ML hoặc DataFrame)"
type = "forbidden"
source_modules = ["src.domain"]
forbidden_modules = [
    "src.application",
    "src.infrastructure",
    "src.presentation",
    "polars",
    "lightgbm",
    "catboost",
    "shap",
    "httpx",
    "websockets",
    "asyncpg",
    "redis",
    "telegram",
]

[[tool.importlinter.contracts]]
name = "Contract 3: Application Layer Purity (Application chỉ được phụ thuộc vào Domain)"
type = "forbidden"
source_modules = ["src.application"]
forbidden_modules = [
    "src.infrastructure",
    "src.presentation",
    "httpx",
    "websockets",
    "asyncpg",
    "redis",
    "telegram",
]

[[tool.importlinter.contracts]]
name = "Contract 4: Infrastructure Submodule Independence (Các adapter hạ tầng không gọi chéo nhau)"
type = "independence"
modules = [
    "src.infrastructure.quant_engine",
    "src.infrastructure.ml",
    "src.infrastructure.agents",
    "src.infrastructure.binance",
    "src.infrastructure.persistence",
]
```

---

## 3. Thiết Kế Hệ Thống `AGENTS.md` Phân Tầng (Hierarchical Rules)

Theo cơ chế khám phá của Antigravity, khi Agent mở hoặc sửa một file, hệ thống sẽ đi ngược từ thư mục chứa file đó lên gốc dự án và tự động nạp các file `AGENTS.md` liên quan.

### 3.1. File Gốc: `/AGENTS.md`
```markdown
# Global Project Rules — LLM_trading_bot_v2

## 1. Kiến Trúc Cốt Lõi (Clean Architecture)
- Dự án tuân thủ nghiêm ngặt Clean Architecture 4 tầng: `src/domain` <- `src/application` <- `src/infrastructure` <- `src/presentation`.
- Mọi giao tiếp giữa `application` và `infrastructure` BẮT BUỘC đi qua các Interfaces (`Protocol` / `ABC`) định nghĩa tại `src/domain/ports/`.
- Tuyệt đối không phá vỡ các hợp đồng trong `[tool.importlinter]` của `pyproject.toml`.

## 2. Quy Tắc Thép: CẤM HARDCODE & CẤM GÁN DỮ LIỆU MẶC ĐỊNH (Fail-Fast Discipline)
1. **Cấm Tham Số Mặc Định Trong Hàm & Method (`No Default Parameters`):**
   - TUYỆT ĐỐI CẤM định nghĩa hàm hoặc method có giá trị mặc định: ví dụ CẤM viết `def calc(window: int = 100, d: float = 0.4):` hoặc `def fetch(timeout: float = 5.0):`.
   - Mọi tham số trong `def ...(...)` bắt buộc phải khai báo kiểu và truyền giá trị tường minh 100% từ nơi gọi hoặc từ đối tượng Config.
2. **Cấm Giá Trị Mặc Định Trong Class / Pydantic Model / Dataclass (`No Default Field Values`):**
   - Trong mọi `BaseModel` và `@dataclass` thuộc `src/`, CẤM gán giá trị mặc định cho các trường cấu hình hoặc dữ liệu (ví dụ CẤM viết `maker_fee: float = 0.0002` hay `leverage: int = 1`). Nếu thiếu trường khi khởi tạo hoặc đọc YAML/JSON, hệ thống phải lập tức báo lỗi `ValidationError` (`Fail-Fast`).
3. **Cấm Biểu Thức Fallback Ngầm (`No Silent Fallback: if sth else hardvalue / dict.get / or`):**
   - TUYỆT ĐỐI CẤM viết các cấu trúc lấp liếm dữ liệu thiếu bằng giá trị cứng như:
     - ❌ `x = val if val is not None else 0.50`
     - ❌ `price = payload.get("price", 0.0)`
     - ❌ `vol = raw_vol or 1.0`
   - Khi một trường dữ liệu bắt buộc bị thiếu, `None`, hoặc `NaN` bất hợp lệ, BẮT BUỘC phải `raise` Exception tường minh (`MissingMarketDataError`, `InvalidConfigurationError`, `CorruptedStateError`) để dừng luồng đó ngay lập tức!
4. **Cấm Hardcode Magic Numbers & Strings:**
   - Cấm nhúng trực tiếp các con số ngưỡng định lượng (như `0.55`, `0.015`, `60000`, `1.5`), tên mã coin, URL hay tên model vào thân hàm nghiệp vụ. Tất cả phải được nạp từ `config/settings.yaml`, `config/rate_limits.yaml`, hoặc `config/alpha_zoo_seed.yaml`.

## 3. Nguyên Tắc Định Lượng & AI Bất Biến
- **Zero LLM Math:** LLM (`Scout`, `Trade Architect`, `Risk Arbiter`, `Math Agent`) KHÔNG BAO GIỜ tự tính chỉ số kỹ thuật, xác suất hay khối lượng lệnh. Toàn bộ tính toán số học phải nằm ở `src/domain/services/` hoặc `src/infrastructure/quant_engine/`.
- **Zero Look-Ahead Bias:** Mọi phép tính trên chuỗi thời gian (`Polars`, `Numba`, `Symbolic DSL`) tại thời điểm `t` chỉ được phép truy cập dữ liệu `<= t`. Cấm tuyệt đối dùng `.shift(-k)` (với `k > 0`) hoặc chuẩn hóa toàn cục (`global mean/std`) trong tính toán đặc trưng; chỉ dùng cửa sổ trượt nhân quả (`rolling_*`, `ewm_*`).

## 4. Quy Trình Kiểm Định 5 Bước Bắt Buộc (`verify_loop`)
- Sau mỗi cụm thay đổi code, bạn BẮT BUỘC chạy lệnh kiểm định toàn diện:
  `powershell -ExecutionPolicy Bypass -File ./scripts/verify_loop.ps1`
- Không bao giờ báo cáo hoàn thành công việc nếu bất kỳ bước nào trong 5 bước (`ruff`, `pyright`, `check_no_defaults_or_hardcode`, `lint-imports`, `pytest`) còn báo lỗi.
```

### 3.2. File Tầng Domain: `/src/domain/AGENTS.md`
```markdown
# Domain Layer Rules (`src/domain/`)

1. **100% Pure Python & Pydantic:**
   - Trong `src/domain/`, chỉ được phép import thư viện chuẩn Python (`math`, `enum`, `typing`, `dataclasses`, `datetime`, `abc`, `uuid`) và `pydantic`.
   - CẤM import `polars`, `numpy`, `lightgbm`, `catboost`, `httpx`, `redis`, `asyncpg`, `telegram`.
2. **Cấm Tuyệt Đối Default Values & Hardcode Trong Entities / Services:**
   - Mọi trường trong `BaseModel` (`RiskConfig`, `SizingResult`, `MarketThesis`, `TradeBlueprint`, `EpisodicCase`) đều là **Required Fields** (không gán `= ...`).
   - Mọi ngưỡng tính toán trong `risk_calculator.py` và `playbook_validator.py` phải lấy từ `self.cfg.<field_name>`, tuyệt đối không viết số hằng trực tiếp trong câu lệnh `if`.
```

### 3.3. File Tầng Quant Engine: `/src/infrastructure/quant_engine/AGENTS.md`
```markdown
# Quant Engine & Symbolic DSL Rules (`src/infrastructure/quant_engine/`)

1. **Causal Time-Series Invariance:**
   - Mọi hàm tính chỉ số (`VPIN`, `OFI`, `Kyle's Lambda`, `Hurst`, `Yang-Zhang Vol`, `FracDiff`, `OI 4-Quadrant`, `Symbolic AST`) phải thỏa mãn bài test bất biến nhân quả:
     $$\text{Feature}(X_{1..t})[t] \equiv \text{Feature}(X_{1..T})[t] \quad \forall t < T$$
2. **Không Tham Số Cửa Sổ Mặc Định (`No Default Window/Param`):**
   - Cấm viết `def rolling_hurst_rs_numba(log_returns: np.ndarray, window: int = 100):`. Tham số `window`, `d`, `ic_threshold` bắt buộc phải truyền tường minh từ `QuantEngineConfig` đọc từ `config/settings.yaml`.
3. **Bắt buộc chạy Test sau khi sửa:**
   - Chạy `uv run pytest tests/unit/test_quant_no_lookahead.py -v` ngay sau khi thêm hoặc sửa bất kỳ công thức nào.
```

### 3.4. File Tầng Multi-Agents: `/src/infrastructure/agents/AGENTS.md`
```markdown
# Multi-Agent & 9router Rules (`src/infrastructure/agents/`)

1. **Bắt Buộc Đi Qua `Per-Provider Redis Rate-Limit Governor`:**
   - Không bao giờ gọi trực tiếp HTTP tới `9router` mà bỏ qua `RateLimitGovernor.acquire_available_route()`.
   - Phải tôn trọng `rpm_limit`, `tpm_limit`, `rpd_limit`, và `min_interval_ms` của từng API con trong `config/rate_limits.yaml`.
2. **Cấm Fallback Dữ Liệu Mặc Định Khi Parse Output Của LLM:**
   - Cấm dùng `.get("field", default_value)` khi đọc JSON của Agent. Nếu LLM trả thiếu trường hoặc sai kiểu, `Pydantic ValidationError` phải được ném ra để kích hoạt cơ chế retry hoặc báo lỗi rõ ràng.
3. **Không Tự Sáng Tác Chiến Thuật Ngoài 4 Playbooks:**
   - `MarketThesis` của `Scout` và `TradeBlueprint` của `Architect` phải khớp với `PlaybookValidator.verify_playbook_preconditions()`.
```

---

## 4. Thiết Kế 3 Workspace Skills (`.agents/skills/`), AST Gate & `Stop` Hook Tự Động

### 4.1. Skill 1: `.agents/skills/verify-loop/SKILL.md`
```markdown
---
name: verify-loop
description: >-
  Run the mandatory 5-stage verification loop (Ruff Lint/Format, Pyright Typecheck,
  AST Zero-Default/No-Hardcode Gate, Import-Linter Clean Architecture Check, and Pytest Suite).
  Use this skill after writing or modifying any Python code.
---

# Verification Loop Skill (`verify-loop`)

Khi kích hoạt skill này, hãy thực thi script tổng hợp:

## Lệnh chạy trên Windows PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/verify_loop.ps1
```

## Ý nghĩa 5 cổng kiểm tra:
1. `uv run ruff check src tests` & `uv run ruff format --check src tests`: Đảm bảo code sạch và chuẩn PEP8.
2. `uv run pyright src tests`: Đảm bảo không có lỗi kiểu dữ liệu tĩnh.
3. `uv run python scripts/check_no_defaults_or_hardcode.py`: Quét cây cú pháp AST trong `src/`, chặn đứng mọi hàm có tham số mặc định (`def f(x=1)`), trường class gán mặc định, `.get(k, default)` hoặc `if cond else hard_literal`.
4. `uv run lint-imports`: Kiểm tra 4 hợp đồng Clean Architecture trong `pyproject.toml`.
5. `uv run pytest tests/ -v`: Chạy toàn bộ Unit & Integration tests.
```

### 4.2. Skill 2: `.agents/skills/add-quant-alpha/SKILL.md`
```markdown
---
name: add-quant-alpha
description: >-
  Step-by-step workflow for adding a new Research Indicator or Symbolic Formulaic Alpha
  to the Hybrid Quant Engine while guaranteeing zero look-ahead bias, zero hardcoding, and economic monotonicity.
---

# Add Quant Alpha Runbook (`add-quant-alpha`)

Khi thêm một chỉ số định lượng mới hoặc một toán tử vào `src/infrastructure/quant_engine/`:
1. **Khai báo Metadata Kinh tế & Tham số trong YAML:** Đăng ký tên biến, tham số cửa sổ, nhóm tương tác (`Regime` / `Microstructure` / `Derivatives` / `CrossAsset`), và dấu kỳ vọng đơn điệu (`long_monotonic_prior`, `short_monotonic_prior`) vào `config/alpha_zoo_seed.yaml`. Tuyệt đối không gán tham số mặc định trong chữ ký hàm Python.
2. **Viết hàm tính toán nhân quả:** Chỉ sử dụng các phép tính cửa sổ trượt (`rolling_*`, `ewm_*`, `ts_rank`, `decay_linear`) trong `Polars` hoặc `Numba`.
3. **Đăng ký vào `test_quant_no_lookahead.py`:** Thêm tên hàm/alpha vào danh sách kiểm thử cắt chuỗi thời gian tại điểm $t = 300$ vs $T = 500$ và chạy `verify-loop`.
```

### 4.3. Skill 3: `.agents/skills/new-clean-slice/SKILL.md`
```markdown
---
name: new-clean-slice
description: >-
  Scaffold a new vertical feature slice adhering strictly to Clean Architecture
  (Domain Entity/Port -> Application UseCase -> Infrastructure Adapter -> Unit Test) with zero default arguments.
---

# New Clean Architecture Slice Runbook (`new-clean-slice`)

Luôn triển khai tính năng mới theo đúng 4 bước từ trong ra ngoài:
1. **Bước 1 (`src/domain/`):** Định nghĩa Entity/ValueObject bằng `Pydantic BaseModel` (tất cả các trường đều bắt buộc, không gán giá trị mặc định) trong `src/domain/entities/` và khai báo `Protocol` Interface trong `src/domain/ports/`.
2. **Bước 2 (`src/application/`):** Viết UseCase trong `src/application/use_cases/` nhận các `Port` và `Config` qua constructor (Dependency Injection).
3. **Bước 3 (`src/infrastructure/`):** Viết class Adapter thực thi `Port` trong thư mục con tương ứng của `src/infrastructure/`.
4. **Bước 4 (`tests/`):** Viết Unit Test cho UseCase và chạy `verify-loop`.
```

### 4.4. Bộ Quét AST Chống Default Arguments & Hardcode Fallback: `scripts/check_no_defaults_or_hardcode.py`

Script này phân tích trực tiếp cây cú pháp trừu tượng (`Python AST`) của mọi file trong `src/` để cưỡng chế tự động quy tắc của bạn:

```python
"""AST Static Gate: Cấm tham số mặc định, trường mặc định và biểu thức gán mặc định dự phòng trong src/."""
from __future__ import annotations

import ast
import sys
from pathlib import Path


class ZeroDefaultAndHardcodeVisitor(ast.NodeVisitor):
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.violations: list[str] = []
        self._in_enum_class = False

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        base_names = {
            b.id if isinstance(b, ast.Name) else (b.attr if isinstance(b, ast.Attribute) else "")
            for b in node.bases
        }
        prev_in_enum = self._in_enum_class
        self._in_enum_class = "Enum" in base_names

        # Nếu là Pydantic BaseModel hoặc Dataclass/Class thường (không phải Enum), cấm gán giá trị mặc định cho field
        if not self._in_enum_class:
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                    target_name = stmt.target.id if isinstance(stmt.target, ast.Name) else "field"
                    self.violations.append(
                        f"{self.file_path}:{stmt.lineno} — [NO_DEFAULT_CLASS_FIELD] "
                        f"Trường `{node.name}.{target_name}` không được gán giá trị mặc định. "
                        f"Hãy truyền tường minh từ file config YAML hoặc constructor."
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
        # 1. Cấm positional default arguments: def fn(a: int = 10)
        if node.args.defaults:
            self.violations.append(
                f"{self.file_path}:{node.lineno} — [NO_DEFAULT_FUNC_PARAM] "
                f"Hàm `{node.name}()` chứa tham số mặc định (positional defaults). "
                f"Bắt buộc bỏ giá trị mặc định và truyền tham số tường minh."
            )
        # 2. Cấm keyword-only default arguments: def fn(*, a: int = 10)
        if any(d is not None for d in node.args.kw_defaults):
            self.violations.append(
                f"{self.file_path}:{node.lineno} — [NO_DEFAULT_KW_PARAM] "
                f"Hàm `{node.name}()` chứa keyword-only default parameter."
            )

    def visit_Call(self, node: ast.Call) -> None:
        # 3. Cấm dict.get("key", default_value) có truyền giá trị fallback mặc định
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get" and len(node.args) >= 2:
            self.violations.append(
                f"{self.file_path}:{node.lineno} — [NO_DICT_GET_FALLBACK] "
                f"Cấm dùng `.get(key, default)` để gán dữ liệu mặc định ngầm. "
                f"Hãy truy cập trực tiếp hoặc kiểm tra khóa và raise Exception tường minh."
            )
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        # 4. Cấm dạng `val if val is not None else HARD_CONSTANT` hoặc `x if x else HARD_CONSTANT`
        if isinstance(node.orelse, ast.Constant | ast.UnaryOp):
            is_none_check = isinstance(node.test, ast.Compare) and any(
                isinstance(op, ast.Is | ast.IsNot) for op in node.test.ops
            )
            is_truthy_name = isinstance(node.test, ast.Name)
            if is_none_check or is_truthy_name:
                self.violations.append(
                    f"{self.file_path}:{node.lineno} — [NO_TERNARY_FALLBACK_DEFAULT] "
                    f"Cấm dùng biểu thức `if ... else <hard_value>` để gán mặc định khi thiếu dữ liệu. "
                    f"Hãy dùng cấu trúc kiểm tra rõ ràng và raise Exception nếu dữ liệu bị None."
                )
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # 5. Cấm dạng `val = raw_val or 0.50`
        if isinstance(node.op, ast.Or) and isinstance(node.values[-1], ast.Constant | ast.UnaryOp):
            self.violations.append(
                f"{self.file_path}:{node.lineno} — [NO_OR_FALLBACK_DEFAULT] "
                f"Cấm dùng toán tử `x or <hard_value>` để gán giá trị mặc định ngầm."
            )
        self.generic_visit(node)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"
    if not src_dir.exists():
        sys.exit(0)

    all_violations: list[str] = []
    for py_file in sorted(src_dir.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        visitor = ZeroDefaultAndHardcodeVisitor(py_file.relative_to(repo_root))
        visitor.visit(tree)
        all_violations.extend(visitor.violations)

    if all_violations:
        print("=== [AST GATE FAILED] PHÁT HIỆN VI PHẠM QUY TẮC CẤM DEFAULT & HARDCODE FALLBACK ===")
        for v in all_violations:
            print(f"  ❌ {v}")
        sys.exit(1)

    print("=== [AST GATE PASSED] 0 Default Parameters, 0 Default Fields, 0 Silent Fallbacks ===")


if __name__ == "__main__":
    main()
```

### 4.5. Cấu Hình Lifecycle Hook Tự Động: `.agents/hooks.json` & `scripts/verify_stop_hook.py`

File `.agents/hooks.json`:
```json
{
  "clean-arch-verify-gate": {
    "enabled": true,
    "Stop": [
      {
        "type": "command",
        "command": "uv run python scripts/verify_stop_hook.py",
        "timeout": 60
      }
    ]
  }
}
```

Nội dung `scripts/verify_stop_hook.py`:
```python
"""Stop Hook Handler: Chặn Agent dừng lượt nếu code trong src/ hoặc tests/ chưa vượt qua 5 cổng verify_loop."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    raw_stdin = sys.stdin.read()
    _payload = json.loads(raw_stdin) if raw_stdin.strip() else {}
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"

    if not src_dir.exists() or not any(src_dir.rglob("*.py")):
        print(json.dumps({"decision": "allow"}))
        return

    commands = [
        ["uv", "run", "ruff", "check", "src", "tests"],
        ["uv", "run", "pyright", "src", "tests"],
        ["uv", "run", "python", "scripts/check_no_defaults_or_hardcode.py"],
        ["uv", "run", "lint-imports"],
        ["uv", "run", "pytest", "tests/unit", "-q"],
    ]

    for cmd in commands:
        proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
        if proc.returncode != 0:
            err_snippet = (proc.stdout + "\n" + proc.stderr).strip()[-1200:]
            print(
                json.dumps(
                    {
                        "decision": "continue",
                        "reason": (
                            f"[Verify-Loop Gate Blocked Stop] Lệnh `{' '.join(cmd)}` thất bại.\n"
                            f"Hãy sửa triệt để lỗi sau trước khi kết thúc lượt:\n{err_snippet}"
                        ),
                    }
                )
            )
            return

    print(json.dumps({"decision": "allow"}))


if __name__ == "__main__":
    main()
```

Nội dung `scripts/verify_loop.ps1`:
```powershell
$ErrorActionPreference = "Stop"
Write-Host "=== [1/5] Running Ruff Linter & Formatter Check ===" -ForegroundColor Cyan
uv run ruff check src tests
uv run ruff format --check src tests

Write-Host "=== [2/5] Running Pyright Static Type Checker ===" -ForegroundColor Cyan
uv run pyright src tests

Write-Host "=== [3/5] Running AST Zero-Default & Anti-Hardcode Gate ===" -ForegroundColor Cyan
uv run python scripts/check_no_defaults_or_hardcode.py

Write-Host "=== [4/5] Running Import-Linter Clean Architecture Contracts ===" -ForegroundColor Cyan
uv run lint-imports

Write-Host "=== [5/5] Running Pytest Unit & Integration Suite ===" -ForegroundColor Cyan
uv run pytest tests -v

Write-Host "=== ALL 5 VERIFICATION GATES PASSED 100% ===" -ForegroundColor Green
```

---

## 5. Đặc Tả Code Tầng 1: Core Domain Entities, Services & Ports (`src/domain/`) — Tuân Thủ 100% Zero Default & Zero Hardcode

### 5.1. Enum Chuẩn Hóa Nghiệp Vụ (`src/domain/entities/enums.py`)

```python
from enum import Enum


class TradeSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class EntryOrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT_PULLBACK_15M = "LIMIT_PULLBACK_15M"


class MarketRegime(str, Enum):
    STRONG_TREND_BULL = "STRONG_TREND_BULL"
    STRONG_TREND_BEAR = "STRONG_TREND_BEAR"
    MEAN_REVERTING_RANGE = "MEAN_REVERTING_RANGE"
    HIGH_VOL_LIQUIDATION_CASCADE = "HIGH_VOL_LIQUIDATION_CASCADE"
    RANDOM_WALK_NOISE = "RANDOM_WALK_NOISE"


class OIPriceQuadrant(str, Enum):
    LONG_BUILDUP = "LONG_BUILDUP"           # Delta P > 0, Delta OI > 0
    SHORT_COVERING = "SHORT_COVERING"       # Delta P > 0, Delta OI < 0
    SHORT_BUILDUP = "SHORT_BUILDUP"         # Delta P < 0, Delta OI > 0
    LONG_LIQUIDATION = "LONG_LIQUIDATION"   # Delta P < 0, Delta OI < 0
    NEUTRAL = "NEUTRAL"


class PlaybookType(str, Enum):
    TREND_CONTINUATION_BREAKOUT = "TREND_CONTINUATION_BREAKOUT"
    LIQUIDATION_EXHAUSTION_REVERSAL = "LIQUIDATION_EXHAUSTION_REVERSAL"
    CROSS_ASSET_LEAD_LAG_CATCHUP = "CROSS_ASSET_LEAD_LAG_CATCHUP"
    NO_TRADE_REGIME = "NO_TRADE_REGIME"


class RiskDecisionType(str, Enum):
    APPROVE_FULL = "APPROVE_FULL"
    APPROVE_REDUCED_SIZE = "APPROVE_REDUCED_SIZE"
    VETO_TO_SHADOW = "VETO_TO_SHADOW"


class OutcomeCategory(str, Enum):
    VALIDATED_WIN = "VALIDATED_WIN"                     # Lệnh thật chạm TP
    FALSE_POSITIVE_TRAP = "FALSE_POSITIVE_TRAP"         # Lệnh thật chạm SL và đi ngược luôn
    STOP_HUNTED_THEN_HIT_TP = "STOP_HUNTED_THEN_HIT_TP" # Lệnh thật chạm SL do râu nến rồi quay lại chạm TP
    MISSED_OPPORTUNITY = "MISSED_OPPORTUNITY"           # Lệnh ảo (bị Veto/Skip) chạm TP ngọt ngào
    VALIDATED_DEFENSE = "VALIDATED_DEFENSE"             # Lệnh ảo chạm SL (Risk Arbiter cứu thua đúng)
    STATISTICAL_NOISE_TIMEOUT = "STATISTICAL_NOISE_TIMEOUT"


class ExecutionMode(str, Enum):
    PAPER = "PAPER"
    TESTNET = "TESTNET"
    LIVE = "LIVE"
```

### 5.2. Domain Service: Quản Trị Rủi Ro Định Lượng (`src/domain/services/risk_calculator.py`)

Mọi trường trong `RiskConfig` và `SizingResult`, cũng như mọi tham số trong `DeterministicRiskCalculator` đều **bắt buộc truyền tường minh 100% (không có giá trị mặc định, không có hằng số hardcode trong logic)**:

```python
"""Deterministic Risk Calculator — 100% Explicit Config Injection, Zero Default Params, Zero Hardcode."""
from __future__ import annotations

import math
from pydantic import BaseModel


class RiskConfig(BaseModel):
    """Toàn bộ tham số bắt buộc phải được nạp từ config/settings.yaml (Không gán mặc định)."""
    maker_fee_rate: float
    taker_fee_rate: float
    min_tp_to_cost_multiple: float
    min_net_rr: float
    kelly_fraction: float
    max_risk_per_trade_pct: float
    min_executable_risk_pct: float
    max_total_open_heat_pct: float
    target_30m_yz_vol: float
    min_vol_floor: float
    max_vol_scalar: float
    uncertainty_penalty_weight: float
    prior_neutral_prob: float
    corr_scalar_zero_same_dir: float
    corr_scalar_one_same_dir: float
    corr_scalar_two_plus_same_dir: float
    daily_drawdown_halt_pct: float
    max_drawdown_freeze_pct: float
    max_margin_utilization_pct: float
    min_barrier_distance_pct: float
    min_leverage: int
    max_leverage: int


class SizingResult(BaseModel):
    """Kết quả tính toán khối lượng lệnh (Mọi trường đều được truyền tường minh)."""
    approved: bool
    rejection_reason: str | None
    net_rr: float
    roundtrip_cost_pct: float
    risk_pct_of_nav: float
    risk_amount_usd: float
    position_notional_usd: float
    recommended_leverage: int


class DeterministicRiskCalculator:
    def __init__(self, config: RiskConfig) -> None:
        self.cfg = config

    @staticmethod
    def _rejected(reason: str, net_rr: float, roundtrip_cost_pct: float, min_lev: int) -> SizingResult:
        return SizingResult(
            approved=False,
            rejection_reason=reason,
            net_rr=net_rr,
            roundtrip_cost_pct=roundtrip_cost_pct,
            risk_pct_of_nav=0.0,
            risk_amount_usd=0.0,
            position_notional_usd=0.0,
            recommended_leverage=min_lev,
        )

    def compute_roundtrip_cost_pct(
        self,
        is_limit_entry: bool,
        estimated_oneway_slippage_pct: float,
    ) -> float:
        if is_limit_entry:
            entry_fee = self.cfg.maker_fee_rate
            entry_slip = 0.0
        else:
            entry_fee = self.cfg.taker_fee_rate
            entry_slip = estimated_oneway_slippage_pct
        exit_fee = self.cfg.taker_fee_rate
        exit_slip = estimated_oneway_slippage_pct
        return entry_fee + exit_fee + entry_slip + exit_slip

    def _resolve_correlation_scalar(self, same_direction_open_count: int) -> float:
        if same_direction_open_count >= 2:
            return self.cfg.corr_scalar_two_plus_same_dir
        if same_direction_open_count == 1:
            return self.cfg.corr_scalar_one_same_dir
        return self.cfg.corr_scalar_zero_same_dir

    def evaluate_and_size_trade(
        self,
        nav_usd: float,
        current_daily_dd_pct: float,
        current_max_dd_pct: float,
        current_open_heat_pct: float,
        same_direction_open_count: int,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        is_limit_entry: bool,
        kyle_slippage_pct: float,
        calibrated_win_prob: float,
        venn_abers_uncertainty_width: float,
        ood_shrinkage_factor: float,
        current_yz_vol: float,
        pre_event_blackout: bool,
    ) -> SizingResult:
        min_lev = self.cfg.min_leverage
        if current_max_dd_pct >= self.cfg.max_drawdown_freeze_pct:
            return self._rejected("MAX_DRAWDOWN_CIRCUIT_BREAKER_ACTIVE", 0.0, 0.0, min_lev)
        if current_daily_dd_pct >= self.cfg.daily_drawdown_halt_pct:
            return self._rejected("DAILY_DRAWDOWN_CIRCUIT_BREAKER_ACTIVE", 0.0, 0.0, min_lev)
        if pre_event_blackout:
            return self._rejected("MACRO_EVENT_BLACKOUT_WINDOW", 0.0, 0.0, min_lev)

        raw_sl_pct = abs(entry_price - stop_loss_price) / entry_price
        raw_tp_pct = abs(take_profit_price - entry_price) / entry_price
        if raw_sl_pct <= self.cfg.min_barrier_distance_pct or raw_tp_pct <= self.cfg.min_barrier_distance_pct:
            return self._rejected("INVALID_SL_OR_TP_DISTANCE", 0.0, 0.0, min_lev)

        cost_pct = self.compute_roundtrip_cost_pct(is_limit_entry, kyle_slippage_pct)
        min_required_tp = self.cfg.min_tp_to_cost_multiple * cost_pct
        if raw_tp_pct < min_required_tp:
            return self._rejected(
                f"TP_TOO_SMALL_VS_FEES ({raw_tp_pct:.4f} < {min_required_tp:.4f})",
                0.0,
                cost_pct,
                min_lev,
            )

        net_tp_pct = raw_tp_pct - cost_pct
        net_sl_pct = raw_sl_pct + cost_pct
        net_rr = net_tp_pct / net_sl_pct
        if net_rr < self.cfg.min_net_rr:
            return self._rejected(
                f"INSUFFICIENT_NET_RR ({net_rr:.2f} < {self.cfg.min_net_rr:.2f})",
                net_rr,
                cost_pct,
                min_lev,
            )

        p_base = self.cfg.prior_neutral_prob
        clamped_shrinkage = max(0.0, min(1.0, ood_shrinkage_factor))
        p_eff = p_base + (calibrated_win_prob - p_base) * clamped_shrinkage
        q_eff = 1.0 - p_eff
        full_kelly = (p_eff * net_rr - q_eff) / net_rr
        if full_kelly <= 0.0:
            return self._rejected(
                f"NON_POSITIVE_KELLY_EXPECTANCY (p_eff={p_eff:.3f}, net_rr={net_rr:.2f})",
                net_rr,
                cost_pct,
                min_lev,
            )

        uncertainty_penalty = max(0.0, 1.0 - self.cfg.uncertainty_penalty_weight * venn_abers_uncertainty_width)
        safe_vol = max(current_yz_vol, self.cfg.min_vol_floor)
        vol_scalar = min(self.cfg.max_vol_scalar, self.cfg.target_30m_yz_vol / safe_vol)
        raw_risk_pct = self.cfg.kelly_fraction * full_kelly * uncertainty_penalty * vol_scalar

        corr_multiplier = self._resolve_correlation_scalar(same_direction_open_count)
        capped_risk_pct = min(self.cfg.max_risk_per_trade_pct * corr_multiplier, raw_risk_pct)
        remaining_heat_pct = max(0.0, self.cfg.max_total_open_heat_pct - current_open_heat_pct)
        final_risk_pct = min(capped_risk_pct, remaining_heat_pct)

        if final_risk_pct < self.cfg.min_executable_risk_pct:
            return self._rejected(
                "PORTFOLIO_HEAT_LIMIT_REACHED_OR_SIZE_TOO_SMALL",
                net_rr,
                cost_pct,
                min_lev,
            )

        risk_usd = nav_usd * final_risk_pct
        notional_usd = risk_usd / net_sl_pct
        max_allowed_notional = nav_usd * self.cfg.max_leverage
        if notional_usd > max_allowed_notional:
            notional_usd = max_allowed_notional
            risk_usd = notional_usd * net_sl_pct
            final_risk_pct = risk_usd / nav_usd

        raw_lev = math.ceil(notional_usd / nav_usd)
        rec_leverage = max(self.cfg.min_leverage, min(self.cfg.max_leverage, raw_lev))
        return SizingResult(
            approved=True,
            rejection_reason=None,
            net_rr=round(net_rr, 3),
            roundtrip_cost_pct=round(cost_pct, 5),
            risk_pct_of_nav=round(final_risk_pct, 5),
            risk_amount_usd=round(risk_usd, 2),
            position_notional_usd=round(notional_usd, 2),
            recommended_leverage=rec_leverage,
        )
```

---

## 6. Checklist Nghiệm Thu Stage 1 (`Doc 01`)

- [ ] Khởi tạo `uv init` & cài đặt toàn bộ dependencies trong `pyproject.toml`.
- [ ] Tạo đầy đủ 4 file `AGENTS.md`, 3 Skills trong `.agents/skills/`, `scripts/check_no_defaults_or_hardcode.py` và `.agents/hooks.json`.
- [ ] Viết toàn bộ Entities, Services (`DeterministicRiskCalculator`, `PlaybookValidator`, `BarrierEvaluator`) và Ports trong `src/domain/` tuân thủ 100% quy tắc không tham số mặc định và không hardcode.
- [ ] Viết `tests/unit/test_domain_risk_and_playbooks.py` và chạy `powershell -ExecutionPolicy Bypass -File ./scripts/verify_loop.ps1` vượt qua cả 5 cổng 100%.
