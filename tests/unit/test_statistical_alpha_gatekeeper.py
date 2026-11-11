"""Unit tests for StatisticalAlphaGatekeeper (Milestone 7.1).

Verifies institutional 4-Stage Statistical Screening:
- Stage 1: Zero Look-Ahead Invariance (tol=1e-12) & Degeneracy / NaN ratios
- Stage 2: Predictive Power (|Rank IC| >= 0.025, |ICIR| >= 0.50, |t| >= 2.5),
           Regime Invariance (low vs. high volatility), and 5-Quantile Monotonicity
- Stage 3: Orthogonality Filter (max pairwise |rho| < 0.60 with existing Alpha Zoo)
- Fail-Fast discipline with typed domain exceptions
- 100% Pure Python implementation with zero default arguments
"""

from __future__ import annotations

import math

import pytest

from trading_bot.domain.entities.quant.symbolic_alpha import (
    AlphaGatekeeperConfig,
    AlphaGateResult,
    ModelTournamentReport,
    SymbolicAlphaCandidate,
    TournamentVerdict,
)
from trading_bot.domain.exceptions import InsufficientDataException
from trading_bot.domain.services.quant.statistical_alpha_gatekeeper import (
    StatisticalAlphaGatekeeper,
)


def create_test_config() -> AlphaGatekeeperConfig:
    """Create strict test configuration without default arguments."""
    return AlphaGatekeeperConfig(
        min_valid_sample_ratio=0.90,
        min_series_std=1e-4,
        invariance_tol=1e-12,
        min_abs_rank_ic=0.025,
        min_abs_icir=0.50,
        min_t_statistic=2.5,
        min_daily_blocks_for_icir=10,
        daily_block_bars=48,
        min_quantile_monotonicity=0.80,
        max_collinearity_corr=0.60,
    )


def test_domain_entities_instantiation() -> None:
    """Verify symbolic alpha entities and tournament models instantiate correctly."""
    candidate = SymbolicAlphaCandidate(
        alpha_id="alpha_m71_01",
        dsl_expression="ts_corr(close, volume, 20)",
        causal_cluster="CLUSTER_1_MICRO",
        long_monotonic_prior=1,
        short_monotonic_prior=-1,
        targeted_blindspot="orderbook_illiquidity_spikes",
        economic_rationale="Volume-price correlation signals aggressive institutional accumulation",
    )
    assert candidate.alpha_id == "alpha_m71_01"
    assert candidate.long_monotonic_prior == 1

    report = ModelTournamentReport(
        created_at_utc="2026-11-10T10:00:00Z",
        champion_model_id="champ_v1",
        challenger_model_id="challenger_v2",
        champion_feature_count=35,
        challenger_feature_count=38,
        champion_net_sharpe=1.85,
        challenger_net_sharpe=2.15,
        champion_max_drawdown_pct=0.082,
        challenger_max_drawdown_pct=0.065,
        champion_calmar_ratio=2.25,
        challenger_calmar_ratio=2.85,
        champion_win_rate=0.54,
        challenger_win_rate=0.58,
        champion_profit_factor=1.65,
        challenger_profit_factor=1.92,
        champion_expectancy_r=0.45,
        challenger_expectancy_r=0.62,
        champion_brier_score=0.185,
        challenger_brier_score=0.162,
        champion_brier_skill_score=0.12,
        challenger_brier_skill_score=0.18,
        champion_log_loss=0.52,
        challenger_log_loss=0.48,
        champion_venn_abers_spread=0.08,
        challenger_venn_abers_spread=0.05,
        is_promotable=True,
        recommended_verdict=TournamentVerdict.PROMOTE_CHALLENGER_IMMEDIATELY,
        approval_checklist={"sharpe_superior": True, "brier_superior": True},
        verdict_rationale="Challenger outperformed champion on all metrics.",
    )
    assert report.is_promotable is True
    assert report.recommended_verdict == TournamentVerdict.PROMOTE_CHALLENGER_IMMEDIATELY


def test_stage1_lookahead_violation() -> None:
    """Stage 1: Verify lookahead leakage is detected when truncated series diverges."""
    cfg = create_test_config()
    gatekeeper = StatisticalAlphaGatekeeper(cfg)
    n = 480
    n_trunc = 240

    full_alpha = [float(i) for i in range(n)]
    target_returns = [0.01 * float(i) for i in range(n)]
    vol_series = [0.02 for _ in range(n)]

    # Truncated series where the last bar diverges (future lookahead leakage)
    trunc_alpha = [float(i) for i in range(n_trunc)]
    trunc_alpha[-1] = trunc_alpha[-1] + 0.05

    result: AlphaGateResult = gatekeeper.evaluate_candidate(
        alpha_id="lookahead_alpha",
        alpha_oos_series=full_alpha,
        alpha_truncated_series=trunc_alpha,
        target_returns_oos=target_returns,
        realized_vol_series=vol_series,
        expected_monotonic_sign=1,
        existing_alpha_matrix_oos=[],
    )

    assert result.passed_all_gates is False
    assert result.failed_stage == "STAGE_1_LOOKAHEAD_VIOLATION"
    assert result.oos_rank_ic == 0.0


def test_stage1_degenerate_or_excess_nan() -> None:
    """Stage 1: Verify rejection on excess NaN ratio or degenerate zero-variance series."""
    cfg = create_test_config()
    gatekeeper = StatisticalAlphaGatekeeper(cfg)
    n = 480
    n_trunc = 240

    # Case A: Excess NaN ratio (> 10% NaNs)
    nan_alpha = [float(i) for i in range(n)]
    for i in range(60):  # 60 / 480 = 12.5% NaNs
        nan_alpha[i] = float("nan")

    target_returns = [0.01 * float(i) for i in range(n)]
    vol_series = [0.02 for _ in range(n)]
    trunc_alpha = nan_alpha[:n_trunc]

    res_nan: AlphaGateResult = gatekeeper.evaluate_candidate(
        alpha_id="excess_nan_alpha",
        alpha_oos_series=nan_alpha,
        alpha_truncated_series=trunc_alpha,
        target_returns_oos=target_returns,
        realized_vol_series=vol_series,
        expected_monotonic_sign=1,
        existing_alpha_matrix_oos=[],
    )
    assert res_nan.passed_all_gates is False
    assert res_nan.failed_stage == "STAGE_1_DEGENERATE_OR_EXCESS_NAN"

    # Case B: Degenerate constant series (std < 1e-4)
    flat_alpha = [5.0 for _ in range(n)]
    trunc_flat = flat_alpha[:n_trunc]

    res_flat: AlphaGateResult = gatekeeper.evaluate_candidate(
        alpha_id="flat_alpha",
        alpha_oos_series=flat_alpha,
        alpha_truncated_series=trunc_flat,
        target_returns_oos=target_returns,
        realized_vol_series=vol_series,
        expected_monotonic_sign=1,
        existing_alpha_matrix_oos=[],
    )
    assert res_flat.passed_all_gates is False
    assert res_flat.failed_stage == "STAGE_1_DEGENERATE_OR_EXCESS_NAN"


def test_stage2_weak_ic_or_icir() -> None:
    """Stage 2: Verify rejection on weak Rank IC or low ICIR."""
    cfg = create_test_config()
    gatekeeper = StatisticalAlphaGatekeeper(cfg)
    n = 480
    n_trunc = 240

    # Periodic wave with negligible correlation to monotonically increasing returns
    alpha_series = [math.sin(float(i) * 2.0 * math.pi / 4.0) for i in range(n)]
    target_returns = [float(i) / float(n) for i in range(n)]
    vol_series = [0.02 for _ in range(n)]
    trunc_alpha = alpha_series[:n_trunc]

    result: AlphaGateResult = gatekeeper.evaluate_candidate(
        alpha_id="weak_ic_alpha",
        alpha_oos_series=alpha_series,
        alpha_truncated_series=trunc_alpha,
        target_returns_oos=target_returns,
        realized_vol_series=vol_series,
        expected_monotonic_sign=1,
        existing_alpha_matrix_oos=[],
    )

    assert result.passed_all_gates is False
    assert result.failed_stage == "STAGE_2_WEAK_IC_OR_ICIR"


def test_stage2_regime_invariance_failure() -> None:
    """Stage 2: Verify rejection when correlation flips sign across volatility regimes."""
    cfg = create_test_config()
    gatekeeper = StatisticalAlphaGatekeeper(cfg)
    n = 480
    n_trunc = 240

    alpha_series = [float(i) for i in range(n)]
    trunc_alpha = alpha_series[:n_trunc]

    # Interleaved even bars (low vol, positive IC) and odd bars (high vol, negative IC)
    # Calibrated so overall Rank IC ~ 0.28, t-stat ~ 6.3, and ICIR ~ 4.2 (passing weak IC gate),
    # while low_ic = +1.0 and high_ic = -0.98 fails regime invariance ((low_ic * high_ic) < 0).
    vol_series: list[float] = [0.01 if i % 2 == 0 else 0.10 for i in range(n)]
    target_returns: list[float] = [0.0] * n
    for b in range(10):
        for i in range(48):
            idx = b * 48 + i
            if i % 2 == 0:
                target_returns[idx] = float(idx)
            else:
                target_returns[idx] = 200.0 - float(b) * 10.0 + float(i) * 0.05

    result: AlphaGateResult = gatekeeper.evaluate_candidate(
        alpha_id="regime_fragile_alpha",
        alpha_oos_series=alpha_series,
        alpha_truncated_series=trunc_alpha,
        target_returns_oos=target_returns,
        realized_vol_series=vol_series,
        expected_monotonic_sign=1,
        existing_alpha_matrix_oos=[],
    )

    assert result.passed_all_gates is False
    assert result.failed_stage == "STAGE_2_REGIME_INVARIANCE_FAILED"
    assert result.is_regime_invariant is False


def test_stage2_non_monotonic_quantiles_u_shape() -> None:
    """Stage 2: Verify rejection when quantile return averages are non-monotonic."""
    cfg = create_test_config()
    gatekeeper = StatisticalAlphaGatekeeper(cfg)
    n = 480
    n_trunc = 240

    # Alpha is linear [0..1]
    alpha_series = [float(i) / float(n) for i in range(n)]
    trunc_alpha = alpha_series[:n_trunc]
    vol_series = [0.02 for _ in range(n)]

    # Generate return profile with positive overall slope but non-monotonic quantiles:
    # Q0: mean ~ 0.02
    # Q1: mean ~ 0.04
    # Q2: mean ~ 0.01 (dip / valley)
    # Q3: mean ~ 0.03
    # Q4: mean ~ 0.05
    # Quantile ranks: [2, 4, 1, 3, 5] -> rho_mono = 0.50 (< 0.80)
    target_returns: list[float] = []
    for x in alpha_series:
        if x < 0.20:
            target_returns.append(0.020 + 0.005 * x)
        elif x < 0.40:
            target_returns.append(0.040 + 0.005 * x)
        elif x < 0.60:
            target_returns.append(0.010 + 0.005 * x)
        elif x < 0.80:
            target_returns.append(0.030 + 0.005 * x)
        else:
            target_returns.append(0.050 + 0.005 * x)

    result: AlphaGateResult = gatekeeper.evaluate_candidate(
        alpha_id="non_monotonic_alpha",
        alpha_oos_series=alpha_series,
        alpha_truncated_series=trunc_alpha,
        target_returns_oos=target_returns,
        realized_vol_series=vol_series,
        expected_monotonic_sign=1,
        existing_alpha_matrix_oos=[],
    )

    assert result.passed_all_gates is False
    assert result.failed_stage == "STAGE_2_NON_MONOTONIC_QUANTILES"
    assert result.quantile_monotonicity_corr < cfg.min_quantile_monotonicity


def test_stage2_direction_mismatch() -> None:
    """Stage 2: Verify rejection when correlation direction opposes declared prior."""
    cfg = create_test_config()
    gatekeeper = StatisticalAlphaGatekeeper(cfg)
    n = 480
    n_trunc = 240

    # Alpha series positively correlated with block index
    alpha_series = [float(i) / float(n) for i in range(n)]
    trunc_alpha = alpha_series[:n_trunc]
    vol_series = [0.02 for _ in range(n)]

    # Target returns strictly inversely correlated: y = -x
    target_returns = [-alpha_series[i] for i in range(n)]

    # Declare expected prior as +1 (positive), while empirical is strongly negative
    result: AlphaGateResult = gatekeeper.evaluate_candidate(
        alpha_id="opposing_prior_alpha",
        alpha_oos_series=alpha_series,
        alpha_truncated_series=trunc_alpha,
        target_returns_oos=target_returns,
        realized_vol_series=vol_series,
        expected_monotonic_sign=1,
        existing_alpha_matrix_oos=[],
    )

    assert result.passed_all_gates is False
    assert result.failed_stage == "STAGE_2_DIRECTION_MISMATCH"
    assert result.oos_rank_ic < 0.0


def test_stage3_collinear_duplicate() -> None:
    """Stage 3: Verify rejection of candidate collinear with existing Alpha Zoo (|rho| >= 0.60)."""
    cfg = create_test_config()
    gatekeeper = StatisticalAlphaGatekeeper(cfg)
    n = 480
    n_trunc = 240

    # Strong valid monotonic alpha
    alpha_series = [float(i) / float(n) + 0.001 * float(i % 3) for i in range(n)]
    target_returns = [0.8 * alpha_series[i] + 0.001 * float(i % 5) for i in range(n)]
    vol_series = [0.01 + 0.01 * (float(i) / float(n)) for i in range(n)]
    trunc_alpha = alpha_series[:n_trunc]

    # Existing Alpha Zoo contains a nearly identical feature (correlation ~0.99)
    collinear_zoo_col = [x + 0.0001 * float(i % 2) for i, x in enumerate(alpha_series)]
    existing_zoo = [collinear_zoo_col]

    result: AlphaGateResult = gatekeeper.evaluate_candidate(
        alpha_id="duplicate_alpha",
        alpha_oos_series=alpha_series,
        alpha_truncated_series=trunc_alpha,
        target_returns_oos=target_returns,
        realized_vol_series=vol_series,
        expected_monotonic_sign=1,
        existing_alpha_matrix_oos=existing_zoo,
    )

    assert result.passed_all_gates is False
    assert result.failed_stage == "STAGE_3_COLLINEAR_DUPLICATE"
    assert result.max_corr_with_existing >= cfg.max_collinearity_corr


def test_all_gates_pass_valid_orthogonal_alpha() -> None:
    """Stages 1-3: Verify valid, monotonic, orthogonal alpha passes all gates."""
    cfg = create_test_config()
    gatekeeper = StatisticalAlphaGatekeeper(cfg)
    n = 480
    n_trunc = 240

    # Strong monotonic alpha
    alpha_series = [float(i) / float(n) + 0.001 * float(i % 3) for i in range(n)]
    target_returns = [0.5 * alpha_series[i] + 0.001 * float(i % 5) for i in range(n)]
    vol_series = [0.01 + 0.02 * (float(i) / float(n)) for i in range(n)]
    trunc_alpha = alpha_series[:n_trunc]

    # Existing Zoo contains an orthogonal alternating signal (low correlation < 0.20)
    orthogonal_zoo_col = [math.sin(float(i) * 2.0 * math.pi / 24.0) for i in range(n)]
    existing_zoo = [orthogonal_zoo_col]

    result: AlphaGateResult = gatekeeper.evaluate_candidate(
        alpha_id="approved_alpha_01",
        alpha_oos_series=alpha_series,
        alpha_truncated_series=trunc_alpha,
        target_returns_oos=target_returns,
        realized_vol_series=vol_series,
        expected_monotonic_sign=1,
        existing_alpha_matrix_oos=existing_zoo,
    )

    assert result.passed_all_gates is True
    assert result.failed_stage is None
    assert result.oos_rank_ic >= cfg.min_abs_rank_ic
    assert abs(result.t_statistic) >= cfg.min_t_statistic
    assert abs(result.oos_icir) >= cfg.min_abs_icir
    assert result.is_regime_invariant is True
    assert result.quantile_monotonicity_corr >= cfg.min_quantile_monotonicity
    assert result.max_corr_with_existing < cfg.max_collinearity_corr


def test_fail_fast_validation_errors() -> None:
    """Verify Fail-Fast InsufficientDataException on invalid dimensions."""
    cfg = create_test_config()
    gatekeeper = StatisticalAlphaGatekeeper(cfg)

    # Empty alpha series
    with pytest.raises(InsufficientDataException, match="Empty alpha series"):
        gatekeeper.evaluate_candidate(
            alpha_id="empty_alpha",
            alpha_oos_series=[],
            alpha_truncated_series=[],
            target_returns_oos=[],
            realized_vol_series=[],
            expected_monotonic_sign=1,
            existing_alpha_matrix_oos=[],
        )

    # Length mismatch
    with pytest.raises(InsufficientDataException, match="Length mismatch"):
        gatekeeper.evaluate_candidate(
            alpha_id="mismatch_alpha",
            alpha_oos_series=[1.0, 2.0, 3.0],
            alpha_truncated_series=[1.0],
            target_returns_oos=[1.0, 2.0],
            realized_vol_series=[1.0, 2.0, 3.0],
            expected_monotonic_sign=1,
            existing_alpha_matrix_oos=[],
        )

    # Truncated length exceeds full series
    with pytest.raises(InsufficientDataException, match="exceeds full series length"):
        gatekeeper.evaluate_candidate(
            alpha_id="excess_trunc_alpha",
            alpha_oos_series=[1.0, 2.0],
            alpha_truncated_series=[1.0, 2.0, 3.0],
            target_returns_oos=[1.0, 2.0],
            realized_vol_series=[1.0, 2.0],
            expected_monotonic_sign=1,
            existing_alpha_matrix_oos=[],
        )

    # Existing zoo series length mismatch
    with pytest.raises(InsufficientDataException, match="Existing alpha series length"):
        gatekeeper.evaluate_candidate(
            alpha_id="zoo_mismatch_alpha",
            alpha_oos_series=[1.0, 2.0, 3.0, 4.0, 5.0],
            alpha_truncated_series=[1.0, 2.0],
            target_returns_oos=[1.0, 2.0, 3.0, 4.0, 5.0],
            realized_vol_series=[1.0, 2.0, 3.0, 4.0, 5.0],
            expected_monotonic_sign=1,
            existing_alpha_matrix_oos=[[1.0, 2.0]],
        )


def test_pure_python_helpers() -> None:
    """Verify accuracy and numerical stability of Pure Python helpers."""
    cfg = create_test_config()
    gatekeeper = StatisticalAlphaGatekeeper(cfg)

    # compute_ranks
    assert gatekeeper.compute_ranks([]) == []
    assert gatekeeper.compute_ranks([10.0, 20.0, 30.0]) == [1.0, 2.0, 3.0]
    assert gatekeeper.compute_ranks([10.0, 20.0, 20.0, 30.0]) == [1.0, 2.5, 2.5, 4.0]

    # compute_spearman_ic
    assert gatekeeper.compute_spearman_ic([1.0, 2.0], [1.0, 2.0], eps=1e-4) == 0.0
    ic_perfect = gatekeeper.compute_spearman_ic(
        [1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0], eps=1e-4
    )
    assert math.isclose(ic_perfect, 1.0, rel_tol=1e-9)

    ic_neg = gatekeeper.compute_spearman_ic(
        [1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0], eps=1e-4
    )
    assert math.isclose(ic_neg, -1.0, rel_tol=1e-9)

    # compute_std
    assert gatekeeper.compute_std([1.0]) == 0.0
    std_val = gatekeeper.compute_std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert math.isclose(std_val, 2.138089935299395, rel_tol=1e-6)

    # compute_t_statistic
    assert gatekeeper.compute_t_statistic(0.5, 2, eps=1e-4) == 0.0
    t_stat = gatekeeper.compute_t_statistic(0.5, 102, eps=1e-4)
    assert t_stat > 5.0

    # check_series_invariance
    assert gatekeeper.check_series_invariance([1.0, 2.0], [1.0, 2.0], tol=1e-12) is True
    assert (
        gatekeeper.check_series_invariance([1.0, float("nan")], [1.0, float("nan")], tol=1e-12)
        is True
    )
    assert gatekeeper.check_series_invariance([1.0, 2.0], [1.0, 2.1], tol=1e-12) is False
