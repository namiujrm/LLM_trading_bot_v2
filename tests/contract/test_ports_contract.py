"""Contract tests for Domain Ports."""

from __future__ import annotations

from trading_bot.domain.entities.common.enums import ExecutionMode
from trading_bot.domain.entities.execution.trade_blueprint import BracketOrderTicket
from trading_bot.domain.entities.market.market_state import MicroBar1m, MultiTFBar
from trading_bot.domain.entities.memory.episodic_case import (
    CohortBaseRateSummary,
    EpisodicCase,
    StratifiedMemoryPack,
)
from trading_bot.domain.entities.risk.account_state import (
    AccountPortfolioContext,
    ActivePositionSnapshot,
    PendingOrderSnapshot,
)
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


def test_quant_engine_port_structural_conformance() -> None:
    from trading_bot.domain.entities.common.enums import MarketRegime, OIPriceQuadrant
    from trading_bot.domain.entities.market.market_state import SemanticQuantPacket
    from trading_bot.domain.entities.quant.quant_features import (
        MicrostructureMetrics,
        QuantFeatureVector,
        RegimeMetrics,
    )
    from trading_bot.ports.quant_engine_port import QuantEnginePort

    class DummyQuantEngine:
        async def compute_microstructure_metrics(
            self, symbol: str, bars_1m: list[MicroBar1m]
        ) -> MicrostructureMetrics:
            return MicrostructureMetrics(
                vpin=0.5,
                kyle_lambda=0.0,
                estimated_oneway_slippage_pct=0.0001,
                ofi_zscore=0.0,
                obi_mean=0.0,
                obi_std=0.0,
                va_amihud=0.0,
                corwin_schultz_spread_bps=5.0,
                mnvr=0.2,
            )

        async def compute_regime_state(
            self, symbol: str, bars_multi_tf: list[MultiTFBar]
        ) -> RegimeMetrics:
            return RegimeMetrics(
                hurst_exponent=0.5,
                yang_zhang_vol=0.02,
                garman_klass_vol=0.02,
                parkinson_vol=0.02,
                bipower_variation=0.02,
                jump_z_score=0.0,
                has_volatility_jump=False,
                shannon_entropy=0.5,
                permutation_entropy=0.5,
                tsallis_entropy=0.5,
                regime_label=MarketRegime.RANDOM_WALK_NOISE,
            )

        async def evaluate_quant_packet(
            self, symbol: str, timestamp_ms: int
        ) -> SemanticQuantPacket:
            return SemanticQuantPacket(
                symbol=symbol,
                timestamp_ms=timestamp_ms,
                active_features=[],
                market_regime=MarketRegime.RANDOM_WALK_NOISE,
                oi_quadrant=OIPriceQuadrant.NEUTRAL,
                trajectory_vector_16d=[0.0] * 16,
            )

        async def compute_feature_vector_batch(
            self, symbol: str, start_time_ms: int, end_time_ms: int
        ) -> list[QuantFeatureVector]:
            return []

    dummy = DummyQuantEngine()
    assert isinstance(dummy, QuantEnginePort)


def test_feature_store_port_structural_conformance() -> None:
    from trading_bot.domain.entities.market.market_state import SemanticQuantPacket
    from trading_bot.domain.entities.quant.quant_features import QuantFeatureVector
    from trading_bot.ports.feature_store_port import FeatureStorePort

    class DummyFeatureStore:
        async def save_feature_vector(self, vector: QuantFeatureVector) -> None:
            pass

        async def save_feature_vectors_batch(self, vectors: list[QuantFeatureVector]) -> int:
            return len(vectors)

        async def fetch_latest_feature_vector(self, symbol: str) -> QuantFeatureVector | None:
            return None

        async def save_quant_packet(self, packet: SemanticQuantPacket) -> None:
            pass

        async def fetch_latest_quant_packet(self, symbol: str) -> SemanticQuantPacket | None:
            return None

        async def export_parquet_features(
            self, symbol: str, year: int, month: int, destination_path: str
        ) -> str:
            return destination_path

    dummy = DummyFeatureStore()
    assert isinstance(dummy, FeatureStorePort)


def test_alpha_dsl_compiler_port_structural_conformance() -> None:
    import ast

    import polars as pl

    from trading_bot.domain.entities.quant.alpha_dsl import AlphaFormulaAST
    from trading_bot.domain.entities.quant.quant_configs import AlphaDSLConfig
    from trading_bot.ports.alpha_dsl_port import AlphaDSLCompilerPort

    class DummyAlphaDSLCompiler:
        def validate_formula(self, formula: str, cfg: AlphaDSLConfig) -> AlphaFormulaAST:
            return AlphaFormulaAST(
                raw_formula=formula,
                canonical_formula=formula,
                ast_hash="dummy_hash",
                node_count=1,
                max_depth=1,
                referenced_columns=(),
                root_node=ast.Expression(body=ast.Constant(value=1.0)),
            )

        def compile_to_polars_expr(self, ast_obj: AlphaFormulaAST, cfg: AlphaDSLConfig) -> pl.Expr:
            return pl.lit(1.0)

        def execute_alpha(
            self,
            df: pl.DataFrame | pl.LazyFrame,
            expr: pl.Expr,
        ) -> pl.Series:
            return pl.Series("alpha", [1.0])

    dummy = DummyAlphaDSLCompiler()
    assert isinstance(dummy, AlphaDSLCompilerPort)


def test_ood_detector_port_structural_conformance() -> None:
    from collections.abc import Sequence

    from trading_bot.domain.entities.ml.ml_predictions import OODDetectionResult
    from trading_bot.ports.ml_model_port import OODDetectorPort

    class DummyOODDetector:
        def fit(
            self,
            train_features: Sequence[Sequence[float]],
            calibration_features: Sequence[Sequence[float]],
        ) -> None:
            pass

        def predict(
            self,
            sample_features: Sequence[float],
        ) -> OODDetectionResult:
            return OODDetectionResult(
                is_ood=False,
                mahalanobis_distance=1.5,
                empirical_quantile_threshold=3.2,
                excess_distance=0.0,
                gamma_ood=1.0,
                feature_contributions=(0.25, 0.25, 0.25, 0.25, 0.25, 0.25),
            )

        def predict_batch(
            self,
            batch_features: Sequence[Sequence[float]],
        ) -> Sequence[OODDetectionResult]:
            return [self.predict(f) for f in batch_features]

    dummy = DummyOODDetector()
    assert isinstance(dummy, OODDetectorPort)


def test_causal_barrier_model_port_structural_conformance() -> None:
    from collections.abc import Sequence

    from trading_bot.adapters.ml_engine.constrained_lgbm_model import ConstrainedLGBMModel
    from trading_bot.domain.entities.ml.ml_predictions import BarrierPredictionResult
    from trading_bot.ports.ml_model_port import CausalBarrierModelPort

    class DummyCausalBarrierModel:
        def fit(
            self,
            train_features: Sequence[Sequence[float]],
            y_long: Sequence[int],
            y_short: Sequence[int],
            sample_weights: Sequence[float],
            sample_intervals: Sequence[tuple[int, int]] | None,
        ) -> None:
            pass

        def predict(
            self,
            sample_features: Sequence[float],
        ) -> BarrierPredictionResult:
            return BarrierPredictionResult(
                p_long=0.4,
                p_short=0.3,
                p_raw_long=0.4,
                p_raw_short=0.3,
                is_coherent=True,
                coherence_penalty=0.0,
            )

        def predict_batch(
            self,
            batch_features: Sequence[Sequence[float]],
        ) -> Sequence[BarrierPredictionResult]:
            return [self.predict(f) for f in batch_features]

    dummy = DummyCausalBarrierModel()
    assert isinstance(dummy, CausalBarrierModelPort)
    assert issubclass(ConstrainedLGBMModel, CausalBarrierModelPort)


def test_probability_calibrator_port_structural_conformance() -> None:
    from collections.abc import Sequence

    from trading_bot.adapters.ml_engine.tabulated_venn_abers import (
        TabulatedVennAbersCalibrator,
    )
    from trading_bot.domain.entities.ml.ml_predictions import CalibratedProbability
    from trading_bot.ports.ml_model_port import ProbabilityCalibratorPort

    class DummyProbabilityCalibrator:
        def fit(
            self,
            scores: Sequence[float],
            labels: Sequence[int],
        ) -> None:
            pass

        def calibrate(
            self,
            score: float,
        ) -> CalibratedProbability:
            return CalibratedProbability(
                p_calibrated=0.60,
                p_lower=0.55,
                p_upper=0.65,
                uncertainty_width=0.10,
                is_uncertain=False,
                raw_score=score,
            )

        def calibrate_batch(
            self,
            scores: Sequence[float],
        ) -> tuple[CalibratedProbability, ...]:
            return tuple(self.calibrate(s) for s in scores)

    dummy = DummyProbabilityCalibrator()
    assert isinstance(dummy, ProbabilityCalibratorPort)
    assert issubclass(TabulatedVennAbersCalibrator, ProbabilityCalibratorPort)


def test_tree_shap_explainer_port_structural_conformance() -> None:
    from collections.abc import Sequence
    from datetime import datetime
    from typing import Any

    from trading_bot.adapters.ml_engine.native_treeshap_explainer import (
        NativeTreeSHAPExplainer,
    )
    from trading_bot.domain.entities.ml.ml_predictions import (
        SHAPDriverContribution,
        SHAPExplanationResult,
    )
    from trading_bot.ports.ml_model_port import TreeSHAPExplainerPort

    class DummyTreeSHAPExplainer:
        def explain(
            self,
            feature_vector: Sequence[float],
            feature_names: Sequence[str],
            model: Any,
            symbol: str,
            bar_time: datetime,
            model_type: str,
        ) -> SHAPExplanationResult:
            contribs = tuple(
                SHAPDriverContribution(
                    feature_name=f,
                    feature_value=v,
                    shap_value=0.1,
                    expected_sign=1,
                    is_sign_consistent=True,
                )
                for f, v in zip(feature_names, feature_vector, strict=True)
            )
            top_bull = tuple(c for c in contribs if c.shap_value > 0.0)[:5]
            raw_margin = sum(c.shap_value for c in contribs) + 0.0
            return SHAPExplanationResult(
                symbol=symbol,
                bar_time=bar_time,
                model_type=model_type,
                base_value=0.0,
                raw_margin=raw_margin,
                contributions=contribs,
                top_bullish_drivers=top_bull,
                top_bearish_drivers=(),
                is_economic_consistent=True,
            )

        def explain_batch(
            self,
            batch_features: Sequence[Sequence[float]],
            feature_names: Sequence[str],
            model: Any,
            symbol: str,
            bar_times: Sequence[datetime],
            model_type: str,
        ) -> tuple[SHAPExplanationResult, ...]:
            return tuple(
                self.explain(row, feature_names, model, symbol, bt, model_type)
                for row, bt in zip(batch_features, bar_times, strict=True)
            )

    dummy = DummyTreeSHAPExplainer()
    assert isinstance(dummy, TreeSHAPExplainerPort)
    assert issubclass(NativeTreeSHAPExplainer, TreeSHAPExplainerPort)
