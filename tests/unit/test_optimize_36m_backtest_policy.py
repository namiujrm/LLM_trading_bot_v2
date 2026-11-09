from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scripts.optimize_36m_backtest_policy import (
    CandidatePolicy,
    OptimizationRunConfig,
    TradeMetrics,
    compute_trade_metrics,
    create_optimization_config,
    evaluate_policy,
)


def test_compute_trade_metrics_empty() -> None:
    """Empty return array must return zero trades and neutral metrics."""
    rets = np.array([], dtype=np.float64)
    metrics: TradeMetrics = compute_trade_metrics(
        returns=rets,
        fee_rate=0.0010,
        annualization_factor=17520.0,
    )
    assert metrics.trade_count == 0
    assert metrics.win_rate == pytest.approx(0.0)
    assert metrics.profit_factor == pytest.approx(1.0)
    assert metrics.sharpe_ratio == pytest.approx(0.0)
    assert metrics.max_drawdown_pct == pytest.approx(0.0)
    assert metrics.cumulative_return == pytest.approx(0.0)
    assert metrics.fee_drag == pytest.approx(0.0)


def test_compute_trade_metrics_all_positive() -> None:
    """All positive returns must produce 100% win rate and positive Sharpe."""
    rets = np.array([0.01, 0.02, 0.015, 0.005], dtype=np.float64)
    metrics: TradeMetrics = compute_trade_metrics(
        returns=rets,
        fee_rate=0.0010,
        annualization_factor=17520.0,
    )
    assert metrics.trade_count == 4
    assert metrics.win_rate == pytest.approx(1.0)
    assert metrics.profit_factor == pytest.approx(2.0)
    assert metrics.sharpe_ratio > 0.0
    assert metrics.max_drawdown_pct == pytest.approx(0.0)
    assert metrics.cumulative_return == pytest.approx(0.05)
    assert metrics.fee_drag == pytest.approx(0.004)
    assert metrics.gross_return == pytest.approx(0.054)


def test_compute_trade_metrics_known_drawdown() -> None:
    """Known sequence of returns must calculate exact max drawdown."""
    # Cumsum: 0.10, 0.05 (dd=0.05), 0.03 (dd=0.07), 0.12 (peak=0.12, dd=0.0)
    rets = np.array([0.10, -0.05, -0.02, 0.09], dtype=np.float64)
    metrics: TradeMetrics = compute_trade_metrics(
        returns=rets,
        fee_rate=0.0010,
        annualization_factor=17520.0,
    )
    assert metrics.trade_count == 4
    assert metrics.win_rate == pytest.approx(0.50)
    assert metrics.profit_factor == pytest.approx(0.19 / 0.07, abs=1e-4)
    assert metrics.max_drawdown_pct == pytest.approx(7.0, abs=0.01)
    assert metrics.cumulative_return == pytest.approx(0.12)


def test_create_optimization_config_factory() -> None:
    """Factory must return strictly typed OptimizationRunConfig with zero defaults."""
    cfg: OptimizationRunConfig = create_optimization_config(
        predictions_path_str="data/predictions_36m/all_4pairs_predictions_36m.parquet",
        features_dir_str="data/features_36m",
        output_report_str="reports/backtest_36m/lead_quant_evaluation_and_optimization_report.md",
        output_json_str="reports/backtest_36m/optimization_summary.json",
    )
    assert (
        cfg.predictions_file
        == Path("data/predictions_36m/all_4pairs_predictions_36m.parquet").resolve()
    )
    assert cfg.roundtrip_fee_rate == pytest.approx(0.0010)
    assert cfg.gamma_conflict == pytest.approx(2.0)
    assert cfg.annualization_factor == pytest.approx(17520.0)
    assert len(cfg.symbols) == 4


def test_evaluate_policy_filtering() -> None:
    """Test that Fee-to-Edge and Uncertainty filters eliminate sub-par trades."""
    cfg: OptimizationRunConfig = create_optimization_config(
        predictions_path_str="data/predictions_36m/all_4pairs_predictions_36m.parquet",
        features_dir_str="data/features_36m",
        output_report_str="reports/backtest_36m/lead_quant_evaluation_and_optimization_report.md",
        output_json_str="reports/backtest_36m/optimization_summary.json",
    )

    # Synthetic toy dataset
    # Row 0: high edge, low unc -> PASS
    # Row 1: low prob -> FAIL threshold
    # Row 2: high prob, but high unc -> FAIL unc filter
    # Row 3: high prob, low unc, but tiny vol -> FAIL fee-to-edge
    data = {
        "symbol": ["BTCUSDT", "BTCUSDT", "ETHUSDT", "ETHUSDT"],
        "is_ood": [False, False, False, False],
        "adj_long": [0.60, 0.53, 0.65, 0.60],
        "adj_short": [0.40, 0.47, 0.35, 0.40],
        "p_long_uncertainty": [0.05, 0.05, 0.18, 0.04],
        "p_short_uncertainty": [0.05, 0.05, 0.18, 0.04],
        "volatility_gk": [0.0050, 0.0050, 0.0050, 0.0005],  # 1.8*vol = 0.009 vs 0.0009
        "roundtrip_cost": [0.0010, 0.0010, 0.0010, 0.0010],
        "sim_long_return": [0.015, -0.010, 0.020, -0.005],
        "sim_short_return": [-0.015, 0.010, -0.020, 0.005],
        "timestamp": pl.datetime_range(
            start=pl.datetime(2024, 1, 1),
            end=pl.datetime(2024, 1, 1, 1, 30),
            interval="30m",
            eager=True,
        ),
    }
    df = pl.DataFrame(data)

    policy = CandidatePolicy(
        name="TestPolicy",
        description="Test Filter",
        threshold=0.56,
        fee_to_edge_mult=2.0,  # Required edge >= 2.0 * 0.001 = 0.002
        max_uncertainty=0.10,
        min_volatility=None,
        allowed_symbols=None,
    )

    res = evaluate_policy(df=df, policy=policy, config=cfg)
    assert res.portfolio_metrics.trade_count == 0
    # Row 0: (2*0.6-1)*(1.8*0.005) = 0.2 * 0.009 = 0.0018 < 0.0020 (FAIL edge)
    # Wait, 0.2 * 0.009 = 0.0018 < 0.0020!
    # If edge mult is 1.5: 0.0018 >= 1.5 * 0.0010 = 0.0015 (PASS)
    policy_15 = CandidatePolicy(
        name="TestPolicy15",
        description="Test Filter 1.5x",
        threshold=0.56,
        fee_to_edge_mult=1.5,
        max_uncertainty=0.10,
        min_volatility=None,
        allowed_symbols=None,
    )
    res_15 = evaluate_policy(df=df, policy=policy_15, config=cfg)
    assert res_15.portfolio_metrics.trade_count == 1
    assert res_15.portfolio_metrics.cumulative_return == pytest.approx(0.015)
