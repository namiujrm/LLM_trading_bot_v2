"""Contract tests for Domain Ports."""

from __future__ import annotations

from trading_bot.domain.entities.account_state import (
    AccountPortfolioContext,
    ActivePositionSnapshot,
    PendingOrderSnapshot,
)
from trading_bot.domain.entities.enums import ExecutionMode
from trading_bot.domain.entities.episodic_case import (
    CohortBaseRateSummary,
    EpisodicCase,
    StratifiedMemoryPack,
)
from trading_bot.domain.entities.market_state import MicroBar1m, MultiTFBar
from trading_bot.domain.entities.trade_blueprint import BracketOrderTicket
from trading_bot.ports.execution_gateway_port import ExecutionGatewayPort
from trading_bot.ports.market_data_port import MarketDataPort
from trading_bot.ports.memory_repository_port import MemoryRepositoryPort


def test_market_data_port_structural_conformance() -> None:
    class DummyMarketData:
        async def fetch_recent_klines_1m(self, symbol: str, limit: int) -> list[MicroBar1m]:
            return []

        async def save_klines_1m(self, symbol: str, bars: list[MicroBar1m]) -> None:
            pass

        async def get_latest_price(self, symbol: str) -> float:
            return 100.0

        async def fetch_multi_tf_bars(
            self, symbol: str, timeframe: str, limit: int
        ) -> list[MultiTFBar]:
            return []

        async def save_account_equity_snapshot(self, context: AccountPortfolioContext) -> None:
            pass

    dummy = DummyMarketData()
    assert isinstance(dummy, MarketDataPort)


def test_memory_repository_port_structural_conformance() -> None:
    class DummyMemoryRepo:
        async def query_knn_cohort(
            self, vector_16d: list[float], playbook: str, k: int
        ) -> list[EpisodicCase]:
            return []

        async def compute_cohort_summary(
            self, vector_16d: list[float], playbook: str, k: int
        ) -> CohortBaseRateSummary:
            return CohortBaseRateSummary(
                has_historical_cohort=False,
                total_neighbors=0,
                weighted_win_rate=None,
                stop_hunted_then_tp_rate=None,
                true_failure_trap_rate=None,
                expected_r_multiple=None,
                mae_target_percentile_atr=None,
                mfe_median_atr=None,
            )

        async def fetch_stratified_pack(
            self, vector_16d: list[float], playbook: str, k: int
        ) -> StratifiedMemoryPack:
            summary = await self.compute_cohort_summary(vector_16d, playbook, k)
            return StratifiedMemoryPack(
                cohort_summary=summary,
                bucket1_bull_missed=[],
                bucket2_failure_traps=[],
                bucket3_stop_hunts=[],
            )

        async def upsert_case(self, case: EpisodicCase) -> None:
            pass

        async def boost_lesson_confidence(self, case_id: str, delta: float) -> None:
            pass

    dummy = DummyMemoryRepo()
    assert isinstance(dummy, MemoryRepositoryPort)


def test_execution_gateway_port_structural_conformance() -> None:
    class DummyExecutionGateway:
        async def place_bracket_order(self, ticket: BracketOrderTicket) -> str:
            return ticket.order_id

        async def cancel_order(self, symbol: str, order_id: str) -> bool:
            return True

        async def get_position_notional(self, symbol: str) -> float:
            return 0.0

        async def fetch_account_portfolio_context(self) -> AccountPortfolioContext:
            return AccountPortfolioContext(
                execution_mode=ExecutionMode.PAPER,
                timestamp_ms=1000,
                wallet_balance_usd=10000.0,
                unrealized_pnl_usd=0.0,
                equity_nav_usd=10000.0,
                used_margin_usd=0.0,
                available_margin_usd=10000.0,
                margin_utilization_pct=0.0,
                current_daily_pnl_usd=0.0,
                current_daily_dd_pct=0.0,
                remaining_daily_dd_budget_pct=0.04,
                current_max_dd_pct=0.0,
                remaining_max_dd_budget_pct=0.08,
                consecutive_losses_count=0,
                active_positions=[],
                pending_orders=[],
                gross_exposure_usd=0.0,
                net_directional_delta_usd=0.0,
                net_btc_beta_exposure_usd=0.0,
                raw_open_heat_pct=0.0,
                effective_open_heat_pct=0.0,
                same_direction_active_count=0,
            )

        async def fetch_active_positions(self) -> list[ActivePositionSnapshot]:
            return []

        async def fetch_pending_orders(self) -> list[PendingOrderSnapshot]:
            return []

    dummy = DummyExecutionGateway()
    assert isinstance(dummy, ExecutionGatewayPort)
