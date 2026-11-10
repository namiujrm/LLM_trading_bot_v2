"""Comprehensive End-to-End Integration Test Suite for Stage 5 Pipeline.

Milestone M5.7: Full Test Suite & 5-Stage Verification Gate.
Covers 4 Key Operational Scenarios:
- Scenario 1 (Normal Deliberation & Order Placement):
  BTC P0 evaluates via Tiered Funnel -> Scout generates bullish thesis -> Altcoins (ETH, SOL)
  evaluate in parallel receiving BTC thesis -> Architect builds bracket blueprints -> Arbiter evaluates
  risk -> Bracket orders placed via BinanceFuturesGatewayAdapter (mocked network) -> Structured logs recorded.
- Scenario 2 (Deterministic Rule-Only Fallback & CRITICAL Logging):
  9Router fails or is offline (simulated timeout/connection error) -> Rate Limit Governor marks cooldown ->
  Scout & Architect provide fallback -> Arbiter falls back to DeterministicRiskCalculator applying
  Quarter-Kelly and Open Heat <= 6% limits -> Emits CRITICAL log
  "9Router offline/exhausted. Deterministic Rule-Only Fallback applied".
- Scenario 3 (Macro News Blackout Halting):
  Active CPI/FOMC blackout window -> Funnel and cycle halt immediately without invoking LLMs or placing trades.
- Scenario 4 (Shadow Veto on Excess Heat / Risk):
  Account is near maximum heat limit -> Arbiter vetoes new trade to VETO_TO_SHADOW -> Flagged as
  shadow order for Stage 6 without live placement.

Enforces 100% Invariant #9 compliance across all test mock classes and helpers:
- Zero default parameters in functions and methods.
- Zero class field defaults in classes/dataclasses/BaseModel.
- Zero silent fallbacks (.get(k, default), a if a is not None else b, x or default).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import pytest

from trading_bot.adapters.agents.arbiter_agent import RiskArbiterAdapter
from trading_bot.adapters.agents.architect_agent import TradeArchitectAdapter
from trading_bot.adapters.agents.macro_news_sentinel import (
    MacroEventSchedule,
    MacroNewsSentinel,
)
from trading_bot.adapters.agents.ninerouter_client import NinerouterClient
from trading_bot.adapters.agents.scout_agent import MarketScoutAdapter
from trading_bot.adapters.binance.binance_futures_gateway import (
    BinanceFuturesGatewayAdapter,
)
from trading_bot.application.use_cases.evaluate_tiered_funnel import (
    TieredFunnelEvaluator,
    TieredFunnelInput,
)
from trading_bot.application.use_cases.run_30m_trading_cycle import (
    Run30mTradingCycleUseCase,
    TradingCycleSummary,
)
from trading_bot.domain.entities.common.enums import (
    EntryOrderType,
    OIPriceQuadrant,
    RiskDecisionType,
    TradeSide,
    VetoCategory,
)
from trading_bot.domain.services.execution.fee_to_edge_gate import (
    FeeToEdgeConfig,
    FeeToEdgeGatingService,
)
from trading_bot.domain.services.execution.playbook_validator import (
    PlaybookConfig,
    PlaybookValidator,
)
from trading_bot.domain.services.risk.risk_calculator import (
    DeterministicRiskCalculator,
    RiskConfig,
)
from trading_bot.infrastructure.config_schemas import (
    BinanceConfig,
    LoggingConfig,
    NinerouterConfig,
)
from trading_bot.infrastructure.logging.bot_logger import (
    LogCategory,
    StructuredBotLogger,
    close_bot_logger,
)
from trading_bot.ports.rate_limit_governor_port import (
    RateLimitGovernorPort,
    RouteQuotaSpec,
)

# ==============================================================================
# Invariant #9 Compliant Test Doubles (Zero Defaults, Zero Class Field Defaults)
# ==============================================================================


class MockHttpResponse:
    """Mock aiohttp HTTP response context manager (Zero Defaults)."""

    def __init__(self, status: int, data: Any, text_data: str) -> None:
        self.status: int = status
        self._data: Any = data
        self._text_data: str = text_data

    async def json(self) -> Any:
        return self._data

    async def text(self) -> str:
        return self._text_data

    async def __aenter__(self) -> MockHttpResponse:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        pass


class MockAiohttpSession:
    """Mock aiohttp ClientSession intercepting requests (Zero Defaults)."""

    def __init__(
        self,
        handler: Callable[
            [str, str, dict[str, Any] | None, dict[str, Any] | None, dict[str, str] | None],
            MockHttpResponse,
        ],
    ) -> None:
        self.handler: Callable[
            [str, str, dict[str, Any] | None, dict[str, Any] | None, dict[str, str] | None],
            MockHttpResponse,
        ] = handler
        self.closed: bool = False
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        json: dict[str, Any] | None,
        headers: dict[str, str] | None,
        timeout: Any,
    ) -> MockHttpResponse:
        record: dict[str, Any] = {
            "method": "POST",
            "url": url,
            "json": json,
            "params": None,
            "headers": headers,
            "timeout": timeout,
        }
        self.calls.append(record)
        return self.handler("POST", url, None, json, headers)

    def request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> MockHttpResponse:
        record: dict[str, Any] = {
            "method": method,
            "url": url,
            "json": None,
            "params": params,
            "headers": headers,
            "timeout": None,
        }
        self.calls.append(record)
        return self.handler(method, url, params, None, headers)

    async def close(self) -> None:
        self.closed = True


class MockRateLimitGovernor(RateLimitGovernorPort):
    """In-memory rate limit governor implementing RateLimitGovernorPort (Zero Defaults)."""

    def __init__(self, specs_by_tier: dict[str, list[RouteQuotaSpec]]) -> None:
        self._specs_by_tier: dict[str, list[RouteQuotaSpec]] = specs_by_tier
        self.cooldown_calls: list[tuple[str, float]] = []
        self.acquired_calls: list[tuple[str, int]] = []

    async def acquire_available_route(
        self,
        tier: str,
        estimated_tokens: int,
    ) -> RouteQuotaSpec:
        self.acquired_calls.append((tier, estimated_tokens))
        if tier in self._specs_by_tier and len(self._specs_by_tier[tier]) > 0:
            return self._specs_by_tier[tier][0]
        raise RuntimeError(f"No routes available in mock governor for tier {tier}")

    async def mark_route_cooldown(
        self,
        route_id: str,
        cooldown_seconds: float,
    ) -> None:
        self.cooldown_calls.append((route_id, cooldown_seconds))


# ==============================================================================
# Helper Factories (Zero Defaults)
# ==============================================================================


def make_route_spec(route_id: str, model_name: str) -> RouteQuotaSpec:
    return RouteQuotaSpec(
        route_id=route_id,
        model_name=model_name,
        rpm_limit=20,
        tpm_limit=150000,
        rpd_limit=2000,
        min_interval_ms=1000,
        max_concurrency=3,
    )


def make_ninerouter_config() -> NinerouterConfig:
    return NinerouterConfig(
        base_url="http://9router:20128/v1",
        api_key="integration-test-api-key",
        timeout_seconds=10.0,
        fast_tier_model="ds/deepseek-v4-flash",
        deep_tier_model="ds/deepseek-v4-pro",
        judge_tier_model="gemini-2.5-flash",
    )


def make_binance_config() -> BinanceConfig:
    return BinanceConfig(
        api_key="binance-test-key",
        api_secret="binance-test-secret-12345",
        testnet=True,
        recv_window_ms=5000,
    )


def make_risk_config() -> RiskConfig:
    return RiskConfig(
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0005,
        min_tp_to_cost_multiple=3.0,
        min_net_rr=1.30,
        kelly_fraction=0.25,
        max_risk_per_trade_pct=0.02,
        min_executable_risk_pct=0.001,
        max_total_open_heat_pct=0.06,
        max_margin_utilization_pct=0.80,
        target_30m_yz_vol=0.015,
        min_vol_floor=0.005,
        max_vol_scalar=2.0,
        uncertainty_penalty_weight=1.0,
        prior_neutral_prob=0.50,
        corr_scalar_zero_same_dir=1.0,
        corr_scalar_one_same_dir=0.6,
        corr_scalar_two_plus_same_dir=0.3,
        daily_drawdown_halt_pct=0.04,
        max_drawdown_freeze_pct=0.08,
        min_barrier_distance_pct=0.003,
        min_leverage=1,
        max_leverage=10,
    )


def make_logging_config(log_dir: str) -> LoggingConfig:
    return LoggingConfig(
        log_dir=log_dir,
        retention_days=3,
        min_level="INFO",
        console_output=False,
    )


def make_funnel_evaluator() -> TieredFunnelEvaluator:
    fee_cfg = FeeToEdgeConfig(
        multiplier=1.0,
        roundtrip_cost_rate=0.0006,
        min_calibrated_prob=0.52,
    )
    fee_service = FeeToEdgeGatingService(fee_cfg)

    pb_cfg = PlaybookConfig(
        trend_hurst_min=0.50,
        trend_vpin_percentile_max=88.0,
        trend_min_calibrated_prob=0.52,
        reversal_hurst_max=0.49,
        reversal_liq_z_min=2.0,
        reversal_funding_z_abs_min=2.2,
        reversal_vpin_slope_max=0.0,
        lead_lag_btc_ret_z_min=1.8,
        lead_lag_btc_beta_min=0.55,
        lead_lag_catchup_gap_z_min=1.5,
        altcoin_symbols=["ETHUSDT", "SOLUSDT", "BNBUSDT"],
    )
    pb_validator = PlaybookValidator(pb_cfg)

    return TieredFunnelEvaluator(
        fee_to_edge_service=fee_service,
        playbook_validator=pb_validator,
        max_spread_bps=2.5,
        min_yz_vol=0.005,
        max_uncertainty_width=0.10,
    )


def make_funnel_input(
    symbol: str,
    p_long: float,
    p_short: float,
    hurst: float,
    oi_quad: OIPriceQuadrant,
    btc_15m_z: float,
    btc_beta: float,
    catchup_z: float,
    is_blackout: bool,
) -> TieredFunnelInput:
    return TieredFunnelInput(
        symbol=symbol,
        spread_bps=1.2,
        yang_zhang_vol=0.015,
        garman_klass_vol=0.015,
        is_blackout=is_blackout,
        is_account_frozen=False,
        p_long_cal=p_long,
        p_short_cal=p_short,
        uncertainty_width=0.03,
        is_ood=False,
        hurst_exp=hurst,
        oi_quadrant=oi_quad,
        vpin_percentile=50.0,
        liq_long_z=0.5,
        liq_short_z=0.5,
        funding_z=0.2,
        vpin_slope_3bars=-0.1,
        btc_15m_return_z=btc_15m_z,
        btc_lead_beta=btc_beta,
        lead_lag_catchup_gap_z=catchup_z,
    )


# ==============================================================================
# Scenario 1: Normal Deliberation & Order Placement
# ==============================================================================


@pytest.mark.asyncio
async def test_scenario_1_normal_deliberation_and_order_placement(tmp_path: Path) -> None:
    """Scenario 1: Full pipeline normal deliberation and bracket order placement.

    Flow:
    - BTC P0 evaluates via Tiered Funnel -> Scout generates bullish thesis.
    - Altcoins (ETH, SOL) evaluate in parallel receiving BTC thesis.
    - Architect builds bracket blueprints.
    - Arbiter evaluates risk (Dual-shield approve full).
    - Bracket orders placed via BinanceFuturesGatewayAdapter.
    - Structured logs recorded in daily JSON Lines files.
    """
    log_dir_str = str(tmp_path / "logs")
    log_cfg = make_logging_config(log_dir_str)
    agent_logger = StructuredBotLogger(LogCategory.AGENTS, log_cfg)
    trading_logger = StructuredBotLogger(LogCategory.TRADING, log_cfg)

    # 1. Setup Rate Limit Governor and Route Specs
    specs_fast = [make_route_spec("gemini_slot_1", "gemini-2.5-flash")]
    specs_deep = [make_route_spec("claude_slot_1", "claude-3-7-sonnet")]
    governor = MockRateLimitGovernor(
        {
            "fast_tactical": specs_fast,
            "deep_reasoning": specs_deep,
        }
    )

    # 2. Setup 9Router HTTP Handler simulating LLM agent responses
    ninerouter_calls: list[dict[str, Any]] = []

    def ninerouter_handler(
        method: str,
        url: str,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> MockHttpResponse:
        assert json_body is not None
        messages: list[dict[str, str]] = json_body["messages"]
        system_content = messages[0]["content"]
        user_content = messages[1]["content"]
        user_dict = json.loads(user_content)
        ninerouter_calls.append({"messages": messages, "user_dict": user_dict})

        # Market Scout Deliberation
        if "Market Scout" in system_content:
            sym = user_dict["symbol"]
            if sym == "BTCUSDT":
                resp_content = json.dumps(
                    {
                        "symbol": "BTCUSDT",
                        "selected_playbook": "TREND_CONTINUATION_BREAKOUT",
                        "proposed_side": "LONG",
                        "supporting_active_features": ["hurst_exp", "vpin"],
                        "regime_alignment_score": 0.92,
                        "thesis_summary": "BTC bullish breakout confirmed across volume and trend metrics.",
                    }
                )
            elif sym == "ETHUSDT":
                # Ensure altcoin scout received lead BTC thesis context
                assert "btc_context" in user_dict
                assert user_dict["btc_context"]["btc_proposed_side"] == "LONG"
                resp_content = json.dumps(
                    {
                        "symbol": "ETHUSDT",
                        "selected_playbook": "CROSS_ASSET_LEAD_LAG_CATCHUP",
                        "proposed_side": "LONG",
                        "supporting_active_features": ["btc_lead_beta", "catchup_gap"],
                        "regime_alignment_score": 0.88,
                        "thesis_summary": "ETH catching up to BTC breakout lead.",
                    }
                )
            elif sym == "SOLUSDT":
                assert "btc_context" in user_dict
                resp_content = json.dumps(
                    {
                        "symbol": "SOLUSDT",
                        "selected_playbook": "GENERAL_ALPHA_MOMENTUM",
                        "proposed_side": "LONG",
                        "supporting_active_features": ["alpha_momentum"],
                        "regime_alignment_score": 0.85,
                        "thesis_summary": "SOL alpha momentum continuation aligned with BTC.",
                    }
                )
            else:
                raise ValueError(f"Unexpected symbol {sym}")

        # Trade Architect Tactical Blueprint Construction
        elif "Trade Architect" in system_content:
            thesis_dict = user_dict["thesis"]
            sym = thesis_dict["symbol"]
            curr_px = float(user_dict["market_context"]["current_price"])
            atr = float(user_dict["market_context"]["atr_30m"])
            resp_content = json.dumps(
                {
                    "order_type": "LIMIT_PULLBACK_15M",
                    "entry_price": curr_px,
                    "stop_loss_price": curr_px - 2.0 * atr,
                    "take_profit_price": curr_px + 3.5 * atr,
                    "rationale": f"ATR bracket limit order for {sym}",
                }
            )

        # Risk Arbiter Deliberation
        elif "Risk Arbiter" in system_content:
            bp_dict = user_dict["blueprint"]
            sym = bp_dict["symbol"]
            resp_content = json.dumps(
                {
                    "decision": "APPROVE_FULL",
                    "approved_leverage": 4,
                    "approved_notional_usd": 12000.0,
                    "net_rr": 1.75,
                    "reason": None,
                    "arbiter_size_multiplier": 1.0,
                    "veto_category": None,
                    "existing_position_directive": "HOLD_EXISTING",
                    "portfolio_risk_rationale": f"Approved trade for {sym} within portfolio capacity.",
                }
            )
        else:
            raise ValueError(f"Unknown system prompt: {system_content}")

        return MockHttpResponse(
            200,
            {
                "id": "chatcmpl-test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": resp_content},
                        "finish_reason": "stop",
                    }
                ],
                "model": "test-model",
                "usage": {"prompt_tokens": 500, "completion_tokens": 120},
            },
            "OK",
        )

    mock_ninerouter_session = MockAiohttpSession(ninerouter_handler)
    nr_client = NinerouterClient(
        make_ninerouter_config(),
        governor,
        mock_ninerouter_session,  # pyright: ignore[reportArgumentType]
        agent_logger,
    )

    scout = MarketScoutAdapter(nr_client, "fast_tactical")
    architect = TradeArchitectAdapter(nr_client, "fast_tactical")

    risk_cfg = make_risk_config()
    calc = DeterministicRiskCalculator(risk_cfg)
    arbiter = RiskArbiterAdapter(nr_client, "deep_reasoning", calc, agent_logger)

    # 3. Setup Binance Futures Gateway with Mocked REST Network
    binance_order_posts: list[dict[str, Any]] = []

    def binance_handler(
        method: str,
        url: str,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> MockHttpResponse:
        if method == "GET" and "/fapi/v2/account" in url:
            # Healthy account with zero open heat
            account_data: dict[str, Any] = {
                "updateTime": 1710000000000,
                "totalWalletBalance": "50000.0",
                "totalUnrealizedProfit": "0.0",
                "totalMarginBalance": "50000.0",
                "totalInitialMargin": "0.0",
                "availableBalance": "50000.0",
                "positions": [],
                "open_orders": [],
            }
            return MockHttpResponse(200, account_data, "OK")

        if method == "POST" and "/fapi/v1/order" in url:
            assert params is not None
            binance_order_posts.append(dict(params))
            order_id_num = 900000 + len(binance_order_posts)
            return MockHttpResponse(200, {"orderId": order_id_num, "status": "NEW"}, "OK")

        raise ValueError(f"Unhandled Binance request: {method} {url}")

    mock_binance_session = MockAiohttpSession(binance_handler)
    gateway = BinanceFuturesGatewayAdapter(
        make_binance_config(),
        mock_binance_session,  # pyright: ignore[reportArgumentType]
    )

    # 4. Macro Sentinel: Clear (No blackout active)
    sentinel = MacroNewsSentinel([])

    # 5. Wire Orchestrator
    funnel_eval = make_funnel_evaluator()
    orchestrator = Run30mTradingCycleUseCase(
        funnel_evaluator=funnel_eval,
        scout=scout,
        architect=architect,
        arbiter=arbiter,
        execution_gateway=gateway,
        macro_sentinel=sentinel,
        memory_repo=None,
        shadow_order_tracker=None,
    )

    # 6. Execute 30-Minute Cascade Trading Cycle
    cycle_time = datetime.now(UTC)
    market_inputs = {
        "BTCUSDT": make_funnel_input(
            "BTCUSDT",
            0.65,
            0.35,
            0.62,
            OIPriceQuadrant.LONG_BUILDUP,
            1.0,
            1.0,
            0.0,
            False,
        ),
        "ETHUSDT": make_funnel_input(
            "ETHUSDT",
            0.60,
            0.40,
            0.52,
            OIPriceQuadrant.LONG_BUILDUP,
            2.2,
            0.95,
            1.8,
            False,
        ),
        "SOLUSDT": make_funnel_input(
            "SOLUSDT",
            0.58,
            0.42,
            0.51,
            OIPriceQuadrant.LONG_BUILDUP,
            1.2,
            1.4,
            0.5,
            False,
        ),
    }
    current_prices = {"BTCUSDT": 60000.0, "ETHUSDT": 3000.0, "SOLUSDT": 150.0}
    atr_values = {"BTCUSDT": 600.0, "ETHUSDT": 35.0, "SOLUSDT": 2.5}
    quant_packets = {
        "BTCUSDT": {"hurst_exp": 0.62, "vpin": 0.25},
        "ETHUSDT": {"btc_lead_beta": 0.95, "catchup_gap": 1.8},
        "SOLUSDT": {"alpha_momentum": 0.72},
    }

    summary: TradingCycleSummary = await orchestrator.execute_cycle(
        cycle_time_utc=cycle_time,
        market_inputs=market_inputs,
        current_prices=current_prices,
        atr_30m_values=atr_values,
        quant_packets=quant_packets,
    )

    # 7. Verification of Operational Results
    assert summary.executed_orders_count == 3
    assert summary.shadow_orders_count == 0

    # BTC P0 Check
    assert summary.btc_result.funnel_passed is True
    assert summary.btc_result.order_id is not None
    assert summary.btc_result.is_shadow is False
    assert summary.btc_result.thesis is not None
    assert summary.btc_result.thesis.proposed_side == TradeSide.LONG
    assert summary.btc_result.verdict is not None
    assert summary.btc_result.verdict.decision == RiskDecisionType.APPROVE_FULL

    # Altcoin P1 Check
    assert "ETHUSDT" in summary.altcoin_results
    assert summary.altcoin_results["ETHUSDT"].order_id is not None
    assert summary.altcoin_results["ETHUSDT"].is_shadow is False
    assert summary.altcoin_results["ETHUSDT"].thesis is not None
    assert summary.altcoin_results["ETHUSDT"].thesis.proposed_side == TradeSide.LONG

    assert "SOLUSDT" in summary.altcoin_results
    assert summary.altcoin_results["SOLUSDT"].order_id is not None
    assert summary.altcoin_results["SOLUSDT"].is_shadow is False

    # Binance Gateway Orders Check: 3 symbols * 3 legs (entry, SL, TP) = 9 orders placed
    assert len(binance_order_posts) == 9

    # 8. Verification of Structured Logs
    trading_logger.info(
        "Trading cycle completed successfully",
        {
            "cycle_ms": summary.cycle_timestamp_ms,
            "executed": summary.executed_orders_count,
            "shadow": summary.shadow_orders_count,
        },
    )
    close_bot_logger(agent_logger.logger)
    close_bot_logger(trading_logger.logger)

    date_str = cycle_time.strftime("%Y-%m-%d")
    trading_log_file = tmp_path / "logs" / "trading" / f"orders_{date_str}.jsonl"
    assert trading_log_file.exists()
    with open(trading_log_file, encoding="utf-8") as f:
        log_lines = f.readlines()
        assert len(log_lines) > 0
        last_rec = json.loads(log_lines[-1])
        assert last_rec["category"] == "trading"
        assert "executed" in last_rec["payload"]
        assert last_rec["payload"]["executed"] == 3


# ==============================================================================
# Scenario 2: Deterministic Rule-Only Fallback & CRITICAL Logging
# ==============================================================================


@pytest.mark.asyncio
async def test_scenario_2_deterministic_fallback_and_critical_logging(tmp_path: Path) -> None:
    """Scenario 2: 9Router offline/fails -> Rate limit governor marks cooldown ->

    Scout & Architect provide fallback -> Arbiter falls back to DeterministicRiskCalculator
    applying Quarter-Kelly and heat limits -> Emits CRITICAL log
    "9Router offline/exhausted. Deterministic Rule-Only Fallback applied".
    """
    log_dir_str = str(tmp_path / "logs")
    log_cfg = make_logging_config(log_dir_str)
    agent_logger = StructuredBotLogger(LogCategory.AGENTS, log_cfg)

    # 1. Setup Governor
    spec = make_route_spec("primary_route", "test-model")
    governor = MockRateLimitGovernor(
        {
            "fast_tactical": [spec],
            "deep_reasoning": [spec],
        }
    )

    # 2. Simulate 9Router offline: Mock session raises ClientConnectionError
    def failing_ninerouter_handler(
        method: str,
        url: str,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> MockHttpResponse:
        raise aiohttp.ClientConnectionError("Connection refused to 9Router at :20128")

    mock_nr_session = MockAiohttpSession(failing_ninerouter_handler)
    nr_client = NinerouterClient(
        make_ninerouter_config(),
        governor,
        mock_nr_session,  # pyright: ignore[reportArgumentType]
        agent_logger,
    )

    scout = MarketScoutAdapter(nr_client, "fast_tactical")
    architect = TradeArchitectAdapter(nr_client, "fast_tactical")

    risk_cfg = make_risk_config()
    calc = DeterministicRiskCalculator(risk_cfg)
    arbiter = RiskArbiterAdapter(nr_client, "deep_reasoning", calc, agent_logger)

    # 3. Setup Binance Gateway Mock
    binance_posts: list[dict[str, Any]] = []

    def binance_handler(
        method: str,
        url: str,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> MockHttpResponse:
        if method == "GET" and "/fapi/v2/account" in url:
            account_data: dict[str, Any] = {
                "updateTime": 1710000000000,
                "totalWalletBalance": "50000.0",
                "totalUnrealizedProfit": "0.0",
                "totalMarginBalance": "50000.0",
                "totalInitialMargin": "0.0",
                "availableBalance": "50000.0",
                "positions": [],
                "open_orders": [],
            }
            return MockHttpResponse(200, account_data, "OK")
        if method == "POST" and "/fapi/v1/order" in url:
            assert params is not None
            binance_posts.append(dict(params))
            return MockHttpResponse(200, {"orderId": 881122, "status": "NEW"}, "OK")
        raise ValueError(f"Unhandled: {url}")

    mock_binance_session = MockAiohttpSession(binance_handler)
    gateway = BinanceFuturesGatewayAdapter(
        make_binance_config(),
        mock_binance_session,  # pyright: ignore[reportArgumentType]
    )

    sentinel = MacroNewsSentinel([])
    orchestrator = Run30mTradingCycleUseCase(
        funnel_evaluator=make_funnel_evaluator(),
        scout=scout,
        architect=architect,
        arbiter=arbiter,
        execution_gateway=gateway,
        macro_sentinel=sentinel,
        memory_repo=None,
        shadow_order_tracker=None,
    )

    cycle_time = datetime.now(UTC)
    market_inputs = {
        "BTCUSDT": make_funnel_input(
            "BTCUSDT",
            0.64,
            0.36,
            0.60,
            OIPriceQuadrant.LONG_BUILDUP,
            1.0,
            1.0,
            0.0,
            False,
        ),
    }
    quant_packets = {
        "BTCUSDT": {
            "p_long_cal": 0.64,
            "p_short_cal": 0.36,
            "proposed_side": TradeSide.LONG,
        }
    }

    # 4. Execute Trading Cycle during 9Router outage
    summary = await orchestrator.execute_cycle(
        cycle_time_utc=cycle_time,
        market_inputs=market_inputs,
        current_prices={"BTCUSDT": 60000.0},
        atr_30m_values={"BTCUSDT": 600.0},
        quant_packets=quant_packets,
    )

    # 5. Verify Rate Limit Governor recorded cooldown for failing route
    assert len(governor.cooldown_calls) > 0
    route_id, cooldown_s = governor.cooldown_calls[0]
    assert route_id == "primary_route"
    assert cooldown_s == 30.0

    # 6. Verify Deterministic Fallback Order Execution
    assert summary.executed_orders_count == 1
    assert summary.shadow_orders_count == 0
    assert summary.btc_result.funnel_passed is True
    assert summary.btc_result.order_id is not None
    assert summary.btc_result.is_shadow is False
    assert summary.btc_result.thesis is not None
    assert summary.btc_result.thesis.proposed_side == TradeSide.LONG
    assert summary.btc_result.blueprint is not None
    assert summary.btc_result.blueprint.order_type == EntryOrderType.LIMIT_PULLBACK_15M
    assert summary.btc_result.verdict is not None
    assert summary.btc_result.verdict.decision == RiskDecisionType.APPROVE_FULL
    assert (
        "deterministic rule-only fallback"
        in summary.btc_result.verdict.portfolio_risk_rationale.lower()
    )

    # 7. Verify CRITICAL log was emitted to JSON Lines file with exact message
    close_bot_logger(agent_logger.logger)
    date_str = cycle_time.strftime("%Y-%m-%d")
    agent_log_file = tmp_path / "logs" / "agents" / f"deliberation_{date_str}.jsonl"
    assert agent_log_file.exists()

    found_critical_arbiter_log = False
    with open(agent_log_file, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if (
                rec["level"] == "CRITICAL"
                and "9Router offline/exhausted. Deterministic Rule-Only Fallback applied"
                in rec["message"]
            ):
                found_critical_arbiter_log = True
                assert rec["payload"]["approved"] is True
                assert rec["payload"]["position_notional_usd"] > 0.0

    assert found_critical_arbiter_log is True, "Expected CRITICAL fallback log was not found"


# ==============================================================================
# Scenario 3: Macro News Blackout Halting
# ==============================================================================


@pytest.mark.asyncio
async def test_scenario_3_macro_news_blackout_halting() -> None:
    """Scenario 3: Active CPI/FOMC blackout window halts funnel and cycle immediately

    without invoking LLMs or placing any orders.
    """
    now = datetime(2026, 11, 9, 15, 0, 0, tzinfo=UTC)

    # 1. Schedule a CPI Release event exactly at current cycle time with 30m blackout buffer
    schedule = MacroEventSchedule(
        event_name="CPI_RELEASE_MOM",
        event_time_utc=now,
        blackout_before_minutes=30.0,
        blackout_after_minutes=30.0,
    )
    sentinel = MacroNewsSentinel([schedule])

    # 2. Mock 9Router and Binance to verify zero network requests
    nr_calls: list[str] = []

    def mock_nr_handler(
        method: str,
        url: str,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> MockHttpResponse:
        nr_calls.append(url)
        return MockHttpResponse(200, {}, "OK")

    binance_calls: list[str] = []

    def mock_binance_handler(
        method: str,
        url: str,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> MockHttpResponse:
        binance_calls.append(url)
        return MockHttpResponse(200, {}, "OK")

    governor = MockRateLimitGovernor(
        {
            "fast_tactical": [make_route_spec("r1", "m1")],
            "deep_reasoning": [make_route_spec("r2", "m2")],
        }
    )
    nr_client = NinerouterClient(
        make_ninerouter_config(),
        governor,
        MockAiohttpSession(mock_nr_handler),  # pyright: ignore[reportArgumentType]
        None,
    )
    scout = MarketScoutAdapter(nr_client, "fast_tactical")
    architect = TradeArchitectAdapter(nr_client, "fast_tactical")
    calc = DeterministicRiskCalculator(make_risk_config())
    arbiter = RiskArbiterAdapter(nr_client, "deep_reasoning", calc, None)
    gateway = BinanceFuturesGatewayAdapter(
        make_binance_config(),
        MockAiohttpSession(mock_binance_handler),  # pyright: ignore[reportArgumentType]
    )

    orchestrator = Run30mTradingCycleUseCase(
        funnel_evaluator=make_funnel_evaluator(),
        scout=scout,
        architect=architect,
        arbiter=arbiter,
        execution_gateway=gateway,
        macro_sentinel=sentinel,
        memory_repo=None,
        shadow_order_tracker=None,
    )

    market_inputs = {
        "BTCUSDT": make_funnel_input(
            "BTCUSDT",
            0.65,
            0.35,
            0.60,
            OIPriceQuadrant.LONG_BUILDUP,
            1.0,
            1.0,
            0.0,
            False,
        ),
        "ETHUSDT": make_funnel_input(
            "ETHUSDT",
            0.60,
            0.40,
            0.52,
            OIPriceQuadrant.LONG_BUILDUP,
            2.0,
            1.0,
            1.5,
            False,
        ),
    }

    summary = await orchestrator.execute_cycle(
        cycle_time_utc=now,
        market_inputs=market_inputs,
        current_prices={"BTCUSDT": 60000.0, "ETHUSDT": 3000.0},
        atr_30m_values={"BTCUSDT": 600.0, "ETHUSDT": 35.0},
        quant_packets={"BTCUSDT": {}, "ETHUSDT": {}},
    )

    # 3. Verify Cycle Halted Immediately
    assert summary.executed_orders_count == 0
    assert summary.shadow_orders_count == 0
    assert summary.btc_result.funnel_passed is False
    assert summary.btc_result.order_id is None
    assert summary.btc_result.rejection_reason is not None
    assert "MACRO_BLACKOUT" in summary.btc_result.rejection_reason
    assert "CPI_RELEASE_MOM" in summary.btc_result.rejection_reason

    assert "ETHUSDT" in summary.altcoin_results
    assert summary.altcoin_results["ETHUSDT"].funnel_passed is False
    assert "MACRO_BLACKOUT" in str(summary.altcoin_results["ETHUSDT"].rejection_reason)

    # 4. Invariant: 0 LLM calls and 0 exchange orders
    assert len(nr_calls) == 0
    assert len(binance_calls) == 0


# ==============================================================================
# Scenario 4: Shadow Veto on Excess Heat / Risk
# ==============================================================================


@pytest.mark.asyncio
async def test_scenario_4_shadow_veto_on_excess_heat() -> None:
    """Scenario 4: Account is near maximum heat limit (6%) -> Arbiter vetoes new trade

    to VETO_TO_SHADOW with VetoCategory.PORTFOLIO_CAPACITY_VETO -> Flagged as shadow
    order for Stage 6 without live Binance placement.
    """
    governor = MockRateLimitGovernor(
        {
            "fast_tactical": [make_route_spec("r1", "m1")],
            "deep_reasoning": [make_route_spec("r2", "m2")],
        }
    )

    def nr_handler(
        method: str,
        url: str,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> MockHttpResponse:
        assert json_body is not None
        sys_prompt = json_body["messages"][0]["content"]

        if "Market Scout" in sys_prompt:
            content = json.dumps(
                {
                    "symbol": "BTCUSDT",
                    "selected_playbook": "TREND_CONTINUATION_BREAKOUT",
                    "proposed_side": "LONG",
                    "supporting_active_features": ["hurst_exp"],
                    "regime_alignment_score": 0.90,
                    "thesis_summary": "Bullish breakout setup.",
                }
            )
        elif "Trade Architect" in sys_prompt:
            content = json.dumps(
                {
                    "order_type": "LIMIT_PULLBACK_15M",
                    "entry_price": 60000.0,
                    "stop_loss_price": 59000.0,
                    "take_profit_price": 62500.0,
                    "rationale": "Tight ATR limit entry.",
                }
            )
        elif "Risk Arbiter" in sys_prompt:
            # Arbiter LLM itself recognizes portfolio heat saturation
            content = json.dumps(
                {
                    "decision": "VETO_TO_SHADOW",
                    "approved_leverage": 1,
                    "approved_notional_usd": 0.0,
                    "net_rr": 2.5,
                    "reason": "Effective open heat saturated near 6% cap",
                    "arbiter_size_multiplier": 0.0,
                    "veto_category": "PORTFOLIO_CAPACITY_VETO",
                    "existing_position_directive": "HOLD_EXISTING",
                    "portfolio_risk_rationale": "Capacity full; routing to Stage 6 shadow paper order.",
                }
            )
        else:
            raise ValueError(f"Unknown prompt: {sys_prompt}")

        return MockHttpResponse(
            200,
            {
                "choices": [
                    {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                ],
                "model": "model_1",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
            "OK",
        )

    nr_client = NinerouterClient(
        make_ninerouter_config(),
        governor,
        MockAiohttpSession(nr_handler),  # pyright: ignore[reportArgumentType]
        None,
    )
    scout = MarketScoutAdapter(nr_client, "fast_tactical")
    architect = TradeArchitectAdapter(nr_client, "fast_tactical")
    calc = DeterministicRiskCalculator(make_risk_config())
    arbiter = RiskArbiterAdapter(nr_client, "deep_reasoning", calc, None)

    # Setup Binance Gateway: Account has existing open position causing 6% effective heat
    binance_posts: list[dict[str, Any]] = []

    def binance_handler(
        method: str,
        url: str,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> MockHttpResponse:
        if method == "GET" and "/fapi/v2/account" in url:
            # 10,000 USD equity with active position risking 600 USD (6.0% heat)
            account_data: dict[str, Any] = {
                "updateTime": 1710000000000,
                "totalWalletBalance": "10000.0",
                "totalUnrealizedProfit": "0.0",
                "totalMarginBalance": "10000.0",
                "totalInitialMargin": "2000.0",
                "availableBalance": "8000.0",
                "positions": [
                    {
                        "symbol": "ETHUSDT",
                        "positionAmt": "1.0",
                        "entryPrice": "3000.0",
                        "markPrice": "3000.0",
                        "unRealizedProfit": "0.0",
                        "liquidationPrice": "2400.0",
                        "leverage": "5",
                        "isolatedMargin": "600.0",
                        "notional": "3000.0",
                        "stopLossPrice": "2400.0",  # risk = (3000 - 2400) * 1.0 = 600.0 USD = 6.0%
                        "takeProfitPrice": "4000.0",
                    }
                ],
                "open_orders": [],
            }
            return MockHttpResponse(200, account_data, "OK")

        if method == "POST" and "/fapi/v1/order" in url:
            assert params is not None
            binance_posts.append(dict(params))
            return MockHttpResponse(200, {"orderId": 99999, "status": "NEW"}, "OK")

        raise ValueError(f"Unhandled url: {url}")

    gateway = BinanceFuturesGatewayAdapter(
        make_binance_config(),
        MockAiohttpSession(binance_handler),  # pyright: ignore[reportArgumentType]
    )

    sentinel = MacroNewsSentinel([])
    orchestrator = Run30mTradingCycleUseCase(
        funnel_evaluator=make_funnel_evaluator(),
        scout=scout,
        architect=architect,
        arbiter=arbiter,
        execution_gateway=gateway,
        macro_sentinel=sentinel,
        memory_repo=None,
        shadow_order_tracker=None,
    )

    cycle_time = datetime(2026, 11, 9, 15, 30, 0, tzinfo=UTC)
    market_inputs = {
        "BTCUSDT": make_funnel_input(
            "BTCUSDT",
            0.65,
            0.35,
            0.60,
            OIPriceQuadrant.LONG_BUILDUP,
            1.0,
            1.0,
            0.0,
            False,
        ),
    }

    summary = await orchestrator.execute_cycle(
        cycle_time_utc=cycle_time,
        market_inputs=market_inputs,
        current_prices={"BTCUSDT": 60000.0},
        atr_30m_values={"BTCUSDT": 600.0},
        quant_packets={"BTCUSDT": {"hurst_exp": 0.60}},
    )

    # Verify Trade was Vetoed to Shadow Order
    assert summary.executed_orders_count == 0
    assert summary.shadow_orders_count == 1
    assert summary.btc_result.funnel_passed is True
    assert summary.btc_result.is_shadow is True
    assert summary.btc_result.order_id is None
    assert summary.btc_result.verdict is not None
    assert summary.btc_result.verdict.decision == RiskDecisionType.VETO_TO_SHADOW
    assert summary.btc_result.verdict.veto_category == VetoCategory.PORTFOLIO_CAPACITY_VETO

    # Invariant: No live order placed on Binance
    assert len(binance_posts) == 0
