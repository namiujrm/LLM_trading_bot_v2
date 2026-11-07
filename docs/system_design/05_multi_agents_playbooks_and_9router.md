# 05 — Thiết Kế Đội 3 Multi-Agents, `BTC-Lead Cascade`, `4 Playbooks` & Bộ Điều Phối Rate-Limit Từng API Cho `9router`

> [!IMPORTANT]
> **Mục tiêu Pattern 5:** Thiết kế luồng suy luận của đội 3 Agents (`Market Scout` $\rightarrow$ `Trade Architect` $\rightarrow$ `Risk Arbiter`) kết hợp dịch vụ chạy nền `Event-Driven Macro & News Sentinel`. Mọi lời gọi LLM đều đi qua cổng **`Per-Provider Redis Rate-Limit Governor`** đặt phía trước **`9router`** để kiểm soát chặt chẽ `RPM`, `TPM`, `RPD` và `min_interval_ms` của từng API con, đồng thời điều phối 4 mã theo chiến lược **`BTC-Lead Cascade`** và khóa cứng suy luận vào **`4 Pre-Verified Quant Playbooks`**.

---

## 1. Sơ Đồ Điều Phối `BTC-Lead Cascade` & Kiểm Soát Rate-Limit Trước `9router`

```mermaid
sequenceDiagram
    autonumber
    participant CRON as 30m Candle Trigger (:00 / :30)
    participant QML as Polars Quant + Dual-Binary ML
    participantSENT as Redis SharedMacroCache
    participant BTC_S as BTCUSDT Scout (Priority P0)
    participant ALT as ETH / SOL / BNB Crew (Priority P1)
    participant GOV as Redis Per-API Rate-Limit Governor
    participant R9 as 9router OpenAI-Compatible Endpoint

    CRON->>QML: Đóng nến 30m cho cả 4 mã (BTC, ETH, SOL, BNB)
    Note over QML: Vectorized Polars + TreeSHAP + Cohort K=20<br/>Hoàn tất cả 4 mã trong ~250ms
    QML->>BTC_S: Gửi BTC QuantPacket + SHAP + Cohort Stats
    SENT->>BTC_S: Đọc Macro/News State từ RAM/Redis (~0.5ms)
    BTC_S->>GOV: Yêu cầu Route thuộc nhóm [fast_tier] (Priority P0)
    GOV->>R9: Chọn Route #1 đang rảnh (RPM/TPM/Interval OK)
    R9-->>BTC_S: Trả về BTC MarketThesis (~3.0s)

    par Chạy song song tiếp theo: Hoàn thiện BTC & Khởi chạy ETH, SOL, BNB có kèm BTC_Realtime_Thesis
        BTC_S->>GOV: Nếu BTC có Playbook != NO_TRADE -> Chạy BTC Architect & Arbiter
    and
        BTC_S->>ALT: Phát sóng BTC_Realtime_Thesis (Bias, Volatility Shock, Lead Direction)
        ALT->>GOV: Yêu cầu Route cho ETH, SOL, BNB (Xoay vòng sang Route #2, #3, #4 đang rảnh)
        GOV->>R9: Phân tải thông minh không chạm Rate Limit của từng API
        R9-->>ALT: Trả về MarketThesis -> TradeBlueprint -> RiskVerdict
    end
```

---

## 2. Thiết Kế `Per-Provider Redis Rate-Limit Governor` Phía Trước `9router`

Khi dùng `9router` để xoay vòng nhiều tài khoản/nhà cung cấp (Gemini, Claude, DeepSeek, OpenRouter, Groq...), nếu chỉ gọi chung một tên model alias mà không kiểm soát nhịp độ ở phía Bot, các luồng chạy song song khi đóng nến `30m` rất dễ bắn dồn dập vào 1 API gây lỗi `429 Too Many Requests` hoặc khóa tạm thời tài khoản.

### 2.1. File Cấu Hình Hạn Mức Từng API: `config/rate_limits.yaml`

```yaml
ninerouter_base_url: "http://9router:20128/v1"
default_timeout_seconds: 15.0

tiers:
  fast_tactical: # Dùng cho Scout, Trade Architect, Macro News Sentinel
    routes:
      - route_id: "gemini_flash_slot_1"
        model_name: "gemini-cli/gemini-2.5-flash"
        rpm_limit: 12            # Tối đa 12 requests / phút
        tpm_limit: 120000        # Tối đa 120,000 tokens / phút
        rpd_limit: 1200          # Tối đa 1,200 requests / ngày
        min_interval_ms: 2200    # Nghỉ tối thiểu 2.2s giữa 2 lần gọi liên tiếp vào slot này
        max_concurrency: 2

      - route_id: "deepseek_v3_slot_1"
        model_name: "deepseek/deepseek-chat"
        rpm_limit: 25
        tpm_limit: 200000
        rpd_limit: 3000
        min_interval_ms: 1000
        max_concurrency: 3

      - route_id: "claude_haiku_slot_1"
        model_name: "claude/claude-3-5-haiku"
        rpm_limit: 15
        tpm_limit: 100000
        rpd_limit: 1500
        min_interval_ms: 1800
        max_concurrency: 2

  deep_reasoning: # Dùng cho Risk Arbiter, Post-Mortem Judge, Weekend Math Agent
    routes:
      - route_id: "gemini_pro_slot_1"
        model_name: "gemini-cli/gemini-2.5-pro"
        rpm_limit: 5
        tpm_limit: 80000
        rpd_limit: 300
        min_interval_ms: 5000
        max_concurrency: 1

      - route_id: "claude_sonnet_slot_1"
        model_name: "claude/claude-3-7-sonnet"
        rpm_limit: 8
        tpm_limit: 80000
        rpd_limit: 500
        min_interval_ms: 3500
        max_concurrency: 2

      - route_id: "deepseek_r1_slot_1"
        model_name: "deepseek/deepseek-reasoner"
        rpm_limit: 10
        tpm_limit: 100000
        rpd_limit: 800
        min_interval_ms: 3000
        max_concurrency: 2
```

### 2.2. Thuật Toán Kiểm Soát Nguyên Tử Trên Redis (`src/infrastructure/agents/rate_limit_governor.py`)

```python
"""Per-Provider Redis Rate-Limit Governor — Kiểm soát RPM, TPM, RPD & Min-Interval cho từng API."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
import redis.asyncio as redis


@dataclass
class RouteQuotaSpec:
    route_id: str
    model_name: str
    rpm_limit: int
    tpm_limit: int
    rpd_limit: int
    min_interval_ms: int
    max_concurrency: int


# Lua Script chạy nguyên tử (Atomic) trên Redis để kiểm tra và cấp phát quyền gọi API
ACQUIRE_ROUTE_LUA = """
local route_id = ARGV[1]
local now_ms = tonumber(ARGV[2])
local est_tokens = tonumber(ARGV[3])
local rpm_limit = tonumber(ARGV[4])
local tpm_limit = tonumber(ARGV[5])
local rpd_limit = tonumber(ARGV[6])
local min_interval_ms = tonumber(ARGV[7])

local cooldown_key = "rl:cooldown:" .. route_id
local last_ts_key  = "rl:last_ts:" .. route_id
local rpm_key      = "rl:rpm:" .. route_id .. ":" .. math.floor(now_ms / 60000)
local tpm_key      = "rl:tpm:" .. route_id .. ":" .. math.floor(now_ms / 60000)
local rpd_key      = "rl:rpd:" .. route_id .. ":" .. math.floor(now_ms / 86400000)

-- 1. Kiểm tra xem API này có đang bị phạt Cooldown (do 429 trước đó) không
local cooldown_until = tonumber(redis.call("GET", cooldown_key) or "0")
if now_ms < cooldown_until then
    return {0, cooldown_until - now_ms, "COOLDOWN_ACTIVE"}
end

-- 2. Kiểm tra khoảng cách tối thiểu giữa 2 lần gọi liên tiếp (min_interval_ms)
local last_ts = tonumber(redis.call("GET", last_ts_key) or "0")
if (now_ms - last_ts) < min_interval_ms then
    return {0, min_interval_ms - (now_ms - last_ts), "MIN_INTERVAL_WAIT"}
end

-- 3. Kiểm tra hạn mức RPD (Requests Per Day), RPM (Per Minute) và TPM (Tokens Per Minute)
local cur_rpd = tonumber(redis.call("GET", rpd_key) or "0")
if cur_rpd + 1 > rpd_limit then
    return {0, 60000, "RPD_EXHAUSTED"}
end

local cur_rpm = tonumber(redis.call("GET", rpm_key) or "0")
if cur_rpm + 1 > rpm_limit then
    return {0, 60000 - (now_ms % 60000), "RPM_EXHAUSTED"}
end

local cur_tpm = tonumber(redis.call("GET", tpm_key) or "0")
if cur_tpm + est_tokens > tpm_limit then
    return {0, 60000 - (now_ms % 60000), "TPM_EXHAUSTED"}
end

-- 4. Cấp phát thành công -> Cập nhật bộ đếm nguyên tử
redis.call("SET", last_ts_key, tostring(now_ms), "PX", 120000)
redis.call("INCR", rpm_key)
redis.call("EXPIRE", rpm_key, 120)
redis.call("INCRBY", tpm_key, est_tokens)
redis.call("EXPIRE", tpm_key, 120)
redis.call("INCR", rpd_key)
redis.call("EXPIRE", rpd_key, 172800)

return {1, 0, "OK"}
"""


@dataclass
class GovernorPacerConfig:
    initial_max_wait_ms: int
    min_poll_step_ms: int


class PerProviderRateLimitGovernor:
    def __init__(
        self,
        redis_client: redis.Redis,
        tier_routes: dict[str, list[RouteQuotaSpec]],
        pacer_cfg: GovernorPacerConfig,
    ) -> None:
        self.redis = redis_client
        self.tier_routes = tier_routes
        self.pacer_cfg = pacer_cfg
        self._rr_cursor: dict[str, int]
        self._rr_cursor = {tier: 0 for tier in tier_routes}

    async def acquire_available_route(self, tier: str, estimated_tokens: int) -> RouteQuotaSpec:
        routes = self.tier_routes[tier]
        n = len(routes)
        while True:
            min_wait_ms = self.pacer_cfg.initial_max_wait_ms
            start_idx = self._rr_cursor[tier]
            for offset in range(n):
                idx = (start_idx + offset) % n
                spec = routes[idx]
                now_ms = int(time.time() * 1000)
                res = await self.redis.eval(
                    ACQUIRE_ROUTE_LUA,
                    0,
                    spec.route_id,
                    str(now_ms),
                    str(estimated_tokens),
                    str(spec.rpm_limit),
                    str(spec.tpm_limit),
                    str(spec.rpd_limit),
                    str(spec.min_interval_ms),
                )
                allowed, wait_ms, _reason = int(res[0]), int(res[1]), res[2]
                if allowed == 1:
                    self._rr_cursor[tier] = (idx + 1) % n
                    return spec
                min_wait_ms = min(min_wait_ms, max(self.pacer_cfg.min_poll_step_ms, wait_ms))

            await asyncio.sleep(min_wait_ms / 1000.0)

    async def mark_route_cooldown(self, route_id: str, cooldown_seconds: float) -> None:
        until_ms = int((time.time() + cooldown_seconds) * 1000)
        await self.redis.set(f"rl:cooldown:{route_id}", str(until_ms), px=int(cooldown_seconds * 1000))
```

---

## 3. Đặc Tả Điều Kiện Biên Bằng Code Cho `4 Pre-Verified Quant Playbooks` (`playbook_validator.py`)

Để đảm bảo `Market Scout` và `Trade Architect` **không bao giờ tự bịa ra lý thuyết giao dịch ngoài kiểm chứng**, tầng Domain kiểm tra điều kiện biên bằng code cho từng Playbook:

| Playbook | Điều Kiện Biên Bắt Buộc Trong Code (`PlaybookValidator`) | Hướng Cho Phép |
| :--- | :--- | :--- |
| **1. `TREND_CONTINUATION_BREAKOUT`** | `hurst_exp >= 0.54` **VÀ** `oi_quadrant in [LONG_BUILDUP, SHORT_BUILDUP]` **VÀ** `vpin_percentile <= 88.0` **VÀ** `max(p_long_cal, p_short_cal) >= 0.57` | Thuận theo hướng `OI Build-up` & `ML Model` |
| **2. `LIQUIDATION_EXHAUSTION_REVERSAL`** | `hurst_exp <= 0.49` **VÀ** (`liq_long_z >= 2.0` hoặc `liq_short_z >= 2.0` hoặc `|funding_z| >= 2.2`) **VÀ** `vpin_slope_3bars <= 0.0` (độc tố xả đã đạt đỉnh và đang dịu xuống) | Đảo chiều ngược hướng bị ép thanh lý |
| **3. `CROSS_ASSET_LEAD_LAG_CATCHUP`** | Mã là Altcoin (`ETH`, `SOL`, `BNB`) **VÀ** `|btc_15m_return_z| >= 1.8` **VÀ** `btc_lead_beta >= 0.55` **VÀ** `|lead_lag_catchup_gap_z| >= 1.5` | Thuận theo hướng vừa bứt phá của `BTCUSDT` |
| **4. `NO_TRADE_REGIME`** | Mặc định khi không thỏa 3 Playbooks trên, hoặc `is_ood_warning == True`, hoặc `pre_event_blackout == True` | `FLAT` (Đứng ngoài 100%) |

---

## 4. Thiết Kế Chi Tiết Đội 3 Agents (`Scout` $\rightarrow$ `Trade Architect` $\rightarrow$ `Risk Arbiter`)

### 4.1. Agent 1: `Market Scout` (`src/infrastructure/agents/scout_agent.py`)
- **Đầu vào:**
  1. `SemanticQuantPacket` (chỉ chứa các biến có `|Rolling 30d IC| >= 0.015` và nhãn `locked_economic_direction`).
  2. `DualBinaryPrediction` (`p_long_cal`, `p_short_cal`, `uncertainty_width`, `ood_mahalanobis_dist`, `Top-5 Bullish SHAP`, `Top-3 Bearish SHAP`).
  3. `SharedMacroCache` (từ `Macro & News Sentinel`) + `BTC_Realtime_Thesis` (nếu đang phân tích `ETH, SOL, BNB`).
  4. `CohortBaseRateSummary` (`K = 20` ca lịch sử gần nhất) + `2 Bull/Missed-Opp Lessons` từ `pgvector`.
- **Đầu ra (`MarketThesis` Pydantic Schema):**
  - `selected_playbook: PlaybookType`
  - `proposed_side: TradeSide` (`LONG` | `SHORT` | `FLAT`)
  - `supporting_active_features: list[str]` (bắt buộc nằm trong danh sách `is_statistically_active == True`)
  - `regime_alignment_score: float` $\in [0.0, 1.0]$
  - `thesis_summary: str`

### 4.2. Agent 2: `Trade Architect` (`src/infrastructure/agents/architect_agent.py`)
- **Chỉ kích hoạt khi:** `MarketThesis.selected_playbook != PlaybookType.NO_TRADE_REGIME` và `PlaybookValidator` xác nhận hợp lệ.
- **Đầu vào:**
  1. `MarketThesis` của `Scout`.
  2. Cấu trúc vi mô khung `15m` (`VWAP_15m`, `Micro_Support_Resistance`, `Yang_Zhang_ATR_Price`, `Spread_bps`, `Kyle_Slippage_Est`).
  3. **`Cohort MAE/MFE Distribution`** (từ `K = 20` ca tương đồng: biết phân vị thứ 80 của `MAE` là bao nhiêu nhân `ATR`) + **`2 Execution / Stop-Hunt Lessons`** từ `pgvector`.
- **Nhiệm vụ cốt lõi:**
  - Chọn **`MARKET`** (nếu Momentum đang bứt phá mạnh và `VPIN` thấp) hoặc **`LIMIT_PULLBACK_15M`** (đặt Limit chờ giá hồi về vùng `0.3x – 0.5x ATR` trên khung `15m` để hưởng phí Maker `0.02%` và tránh bị quét râu nến đầu phiên).
  - Đặt `stop_loss_price` rộng hơn `80th-percentile MAE` của quần thể `K = 20` ca tương đồng (nằm trong dải `[1.0x, 2.2x] Yang-Zhang ATR`) và đặt `take_profit_price` đảm bảo `Net R:R >= 1.5`.

### 4.3. Agent 3: `Risk Arbiter` (`src/infrastructure/agents/arbiter_agent.py`)
- **Lớp 1 — `Portfolio-Aware Semantic Risk Prosecutor` (Chạy trên `deep_reasoning` tier của `9router`):**
  - **Đầu vào bắt buộc (Đầy đủ Ngữ cảnh Kỹ thuật + Ngữ cảnh Tài khoản & Danh mục `AccountPortfolioContext`):**
    1. `TradeBlueprint` của `Trade Architect` + `SemanticQuantPacket` + `DualBinaryPrediction` (`SHAP` âm & dương).
    2. **`2 True Failure Trap Lessons`** từ `pgvector`.
    3. **Gói Ngữ Cảnh Tài Khoản & Vị Thế Đang Mở (`AccountPortfolioContext`):**
       - **Số dư & Ký quỹ (`Wallet & Margin Health`):** `wallet_balance_usd`, `unrealized_pnl_usd`, `equity_nav_usd`, `total_initial_margin_usd`, `available_margin_usd`, `margin_utilization_pct`.
       - **Chi tiết Vị thế đang mở (`active_positions: list[ActivePositionSnapshot]`):** Danh sách từng lệnh đang giữ (`symbol`, `side`, `notional_usd`, `leverage`, `isolated_margin_usd`, `entry_price`, `mark_price`, `liquidation_price`, `unrealized_pnl_usd`, `unrealized_r_multiple`, `is_sl_at_breakeven`, `effective_at_risk_usd`, `btc_beta`).
       - **Lệnh chờ đang treo (`pending_orders: list[PendingOrderSnapshot]`):** Các lệnh `LIMIT_PULLBACK_15M` đang chờ khớp và phần `reserved_margin_usd`.
       - **Tương quan & Độ nóng thực tế (`Effective Heat & Beta Exposure`):** `raw_open_heat_pct` vs. **`effective_open_heat_pct`** (chỉ cộng rủi ro của các vị thế **chưa** dời SL về hòa vốn `is_sl_at_breakeven == False`!), `net_directional_delta_usd`, `net_btc_beta_exposure_usd`, `same_direction_active_count`.
       - **Ngân sách Drawdown & Chuỗi phong độ (`Drawdown Headroom & Streak`):** `current_daily_dd_pct`, `remaining_daily_dd_budget_pct`, `current_max_dd_pct`, `remaining_max_dd_budget_pct`, `consecutive_losses_count`.
  - **Bộ 4 Quy Tắc Xét Duyệt Có Ngữ Cảnh Tài Khoản (`Portfolio-Aware Prosecutor Rubric`):**
    1. *Quy tắc 1 (`Effective Open Heat vs. Breakeven Lock`):* Phân biệt rạch ròi vị thế đang chịu rủi ro (`is_sl_at_breakeven == False`) với vị thế đã chốt `TP1` và khóa SL ở hòa vốn (`is_sl_at_breakeven == True`, `effective_at_risk_usd == 0.0`). Nếu các lệnh cũ đã khóa hòa vốn và `available_margin_usd` dồi dào, không được VETO oan lệnh mới chỉ vì số lượng vị thế mở.
    2. *Quy tắc 2 (`Correlated Beta Stacking vs. Natural Portfolio Hedge`):* Nếu danh mục đang giữ $\ge 2$ vị thế cùng chiều chưa khóa Breakeven và đang âm trạng thái (`unrealized_pnl_usd < 0`), phải siết chặt: chọn `APPROVE_REDUCED_SIZE` (`arbiter_size_multiplier = 0.5`) hoặc `VETO_TO_SHADOW`. Ngược lại, nếu lệnh đề xuất mới ngược chiều với `net_btc_beta_exposure_usd` hiện tại (đóng vai trò **Natural Hedge** giảm tổng rủi ro danh mục) và có cấu trúc định lượng tốt, ưu tiên duyệt!
    3. *Quy tắc 3 (`Drawdown Headroom & Loss-Streak Throttling`):* Khi ngân sách lỗ ngày còn lại thấp (`remaining_daily_dd_budget_pct <= 1.2%`) hoặc tài khoản đang dính chuỗi thua liên tiếp (`consecutive_losses_count >= 2`), tự động kích hoạt chế độ **`DEFENSIVE_CAPITAL_PRESERVATION`**: bắt buộc hạ bậc xuống `APPROVE_REDUCED_SIZE` (`0.5x`) hoặc `VETO_TO_SHADOW` đối với mọi kèo có `uncertainty_width` cao.
    4. *Quy tắc 4 (`Same-Symbol Conflict & Veto Attribution`):* Cấm mở chồng lệnh cùng mã (`Pyramiding`) nếu vị thế hiện tại trên mã đó chưa khóa SL về Breakeven. Khi ra quyết định `VETO_TO_SHADOW`, bắt buộc phân loại rõ `veto_category`:
       - `SETUP_DEFECT`: Từ chối vì bẫy phân tích kỹ thuật / vi cấu trúc / SHAP ngược chiều.
       - `PORTFOLIO_CAPACITY_VETO`: Setup kỹ thuật tốt nhưng từ chối vì tài khoản cạn Margin, chạm trần Beta cùng chiều hoặc sát ngưỡng Drawdown ngày (giúp `Post-Mortem Judge` không phạt nhầm bài học của `Scout/Architect`!).
  - **Đầu ra (`RiskVerdict` Pydantic Schema):**
    - `decision: RiskDecisionType` (`APPROVE_FULL` | `APPROVE_REDUCED_SIZE` | `VETO_TO_SHADOW`)
    - `arbiter_size_multiplier: float` (`1.0`, `0.5`, hoặc `0.0`)
    - `veto_category: str | None` (`SETUP_DEFECT` | `PORTFOLIO_CAPACITY_VETO` | `None`)
    - `existing_position_directive: str` (`HOLD_EXISTING` | `TIGHTEN_EXISTING_SL_TO_BE` | `FORBID_DUPLICATE_SYMBOL`)
    - `portfolio_risk_rationale: str`
- **Lớp 2 — `DeterministicRiskCalculator` (Code cứng trong `src/domain/services/risk_calculator.py`):**
  - Nhận `AccountPortfolioContext` và `arbiter_size_multiplier` từ Lớp 1 để chạy kiểm tra toán học bất biến cuối cùng về `Fee-to-Edge`, `Quarter-Kelly * arbiter_size_multiplier`, `Effective Open Heat Cap`, **`Available Margin & Max Margin Utilization Cap`**, và `Drawdown Circuit Breaker`.

---

## 5. Checklist Nghiệm Thu Stage 4 (`Doc 05`)

- [ ] Viết Unit Test `test_rate_limit_governor.py` giả lập bắn 20 requests đồng thời vào tier có 3 API với `min_interval_ms = 2000ms` và `rpm_limit = 5`, xác nhận hệ thống tự xoay vòng qua cả 3 API và tự động sleep điều tiết mà không để lọt bất kỳ lỗi vượt quota nào.
- [ ] Kiểm thử `ResilientStructuredOutputParser` với chuỗi phản hồi có chứa `<think>...</think>` và khối markdown JSON, xác nhận bóc tách sạch sẽ vào `Pydantic Schema`.
- [ ] Kiểm thử luồng `BTC-Lead Cascade` cho 4 mã hoàn tất toàn bộ chu kỳ trong `< 15 giây`.
