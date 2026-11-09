"""Unit tests for ICGatekeeperService (M3.5).

Verifies:
- Spearman Rank IC calculation
- Student t-statistic calculation
- IC-IR calculation across sub-windows
- High/Low volatility regime invariance
- Economic direction locking
- Edge cases and Fail-Fast exceptions
"""

from __future__ import annotations

import pytest

from trading_bot.domain.entities.quant.quant_configs import ICGatekeeperConfig
from trading_bot.domain.exceptions import InsufficientDataException, NumericalInstabilityException
from trading_bot.domain.services.quant.ic_gatekeeper import (
    ICGatekeeperEvaluation,
    ICGatekeeperService,
)


def get_default_test_config() -> ICGatekeeperConfig:
    return ICGatekeeperConfig(
        min_rank_ic=0.015,
        min_t_statistic=2.0,
        min_ic_ir=0.50,
        rolling_eval_window_bars=100,
        forward_prediction_bars=12,
    )


def test_compute_ranks_simple_and_ties() -> None:
    cfg = get_default_test_config()
    service = ICGatekeeperService(cfg)

    # Simple sorted
    ranks = service.compute_ranks([10.0, 20.0, 30.0])
    assert ranks == [1.0, 2.0, 3.0]

    # Ties: midrank
    ranks_ties = service.compute_ranks([10.0, 20.0, 20.0, 30.0])
    assert ranks_ties == [1.0, 2.5, 2.5, 4.0]

    # All identical
    ranks_all_same = service.compute_ranks([5.0, 5.0, 5.0])
    assert ranks_all_same == [2.0, 2.0, 2.0]

    # Empty
    assert service.compute_ranks([]) == []


def test_compute_ranks_rejects_non_finite() -> None:
    cfg = get_default_test_config()
    service = ICGatekeeperService(cfg)
    with pytest.raises(NumericalInstabilityException):
        service.compute_ranks([1.0, float("nan"), 3.0])


def test_compute_spearman_ic_perfect_correlations() -> None:
    cfg = get_default_test_config()
    service = ICGatekeeperService(cfg)
    eps = 1e-12

    x = [float(i) for i in range(50)]
    y = [float(i) for i in range(50)]
    ic_pos = service.compute_spearman_ic(x, y, eps)
    assert pytest.approx(ic_pos, rel=1e-5) == 1.0

    y_rev = [float(-i) for i in range(50)]
    ic_neg = service.compute_spearman_ic(x, y_rev, eps)
    assert pytest.approx(ic_neg, rel=1e-5) == -1.0


def test_compute_spearman_ic_flat_series() -> None:
    cfg = get_default_test_config()
    service = ICGatekeeperService(cfg)
    eps = 1e-12

    x = [5.0] * 20
    y = [float(i) for i in range(20)]
    ic_zero = service.compute_spearman_ic(x, y, eps)
    assert ic_zero == 0.0


def test_compute_t_statistic() -> None:
    cfg = get_default_test_config()
    service = ICGatekeeperService(cfg)
    eps = 1e-12

    t_val = service.compute_t_statistic(0.10, 100, eps)
    # df = 98, sqrt(98) ~= 9.899, sqrt(1 - 0.01) = sqrt(0.99) ~= 0.99498
    # t = 0.1 * 9.899 / 0.99498 ~= 0.9949
    assert pytest.approx(t_val, rel=1e-2) == 0.995

    # Extreme IC
    assert service.compute_t_statistic(1.0, 50, eps) == 999.0
    assert service.compute_t_statistic(-1.0, 50, eps) == -999.0


def test_compute_ic_ir() -> None:
    cfg = get_default_test_config()
    service = ICGatekeeperService(cfg)
    eps = 1e-12

    # High consistency: sub-ICs all near 0.05
    sub_ics = [0.048, 0.052, 0.050, 0.049, 0.051]
    ir = service.compute_ic_ir(sub_ics, eps)
    assert ir > 10.0

    # Low consistency / sign flip
    sub_ics_noisy = [0.05, -0.04, 0.03, -0.02, 0.01]
    ir_noisy = service.compute_ic_ir(sub_ics_noisy, eps)
    assert ir_noisy < 0.50


def test_regime_invariance_detection() -> None:
    cfg = get_default_test_config()
    service = ICGatekeeperService(cfg)
    eps = 1e-12

    # High-vol has positive IC, Low-vol has positive IC
    x = [float(i) for i in range(100)]
    y = [float(i) for i in range(100)]
    vol = [0.01] * 50 + [0.05] * 50

    is_invariant, high_ic, low_ic = service.check_regime_invariance(x, y, vol, eps)
    assert is_invariant is True
    assert high_ic > 0.0
    assert low_ic > 0.0

    # Sign flip: high-vol positive, low-vol negative
    y_flipped = [-float(i) for i in range(50)] + [float(i) for i in range(50, 100)]
    is_inv_flip, high_ic_f, low_ic_f = service.check_regime_invariance(x, y_flipped, vol, eps)
    assert is_inv_flip is False
    assert (high_ic_f * low_ic_f) <= 0.0


def test_evaluate_feature_passes_all_gates() -> None:
    cfg = get_default_test_config()
    service = ICGatekeeperService(cfg)
    eps = 1e-12

    # Construct feature with steady positive correlation to forward return
    n = 100
    x = [float(i) + (0.1 if i % 2 == 0 else -0.1) for i in range(n)]
    y = [float(i) for i in range(n)]
    vol = [0.01] * (n // 2) + [0.04] * (n // 2)

    eval_result: ICGatekeeperEvaluation = service.evaluate_feature(
        feature_name="momentum_score",
        x=x,
        y=y,
        vol_series=vol,
        locked_direction=1,
        sub_window_count=4,
        eps=eps,
    )

    assert eval_result.is_statistically_active is True
    assert eval_result.is_regime_invariant is True
    assert eval_result.is_direction_valid is True
    assert eval_result.rejection_reason is None
    assert eval_result.rank_ic > cfg.min_rank_ic
    assert eval_result.t_statistic > cfg.min_t_statistic
    assert eval_result.ic_ir > cfg.min_ic_ir


def test_evaluate_feature_rejects_economic_direction_mismatch() -> None:
    cfg = get_default_test_config()
    service = ICGatekeeperService(cfg)
    eps = 1e-12

    # Negative correlation but locked_direction is +1
    n = 100
    x = [float(-i) for i in range(n)]
    y = [float(i) for i in range(n)]
    vol = [0.02] * (n // 2) + [0.05] * (n // 2)

    eval_result = service.evaluate_feature(
        feature_name="vpin_imbalance",
        x=x,
        y=y,
        vol_series=vol,
        locked_direction=1,  # expects positive, but empirical is negative
        sub_window_count=4,
        eps=eps,
    )

    assert eval_result.is_statistically_active is False
    assert eval_result.is_direction_valid is False
    assert eval_result.rejection_reason is not None
    assert "ECONOMIC_DIRECTION_MISMATCH" in eval_result.rejection_reason


def test_evaluate_feature_rejects_insufficient_data() -> None:
    cfg = get_default_test_config()
    service = ICGatekeeperService(cfg)
    eps = 1e-12

    with pytest.raises(InsufficientDataException):
        service.evaluate_feature(
            feature_name="short_feature",
            x=[1.0, 2.0],
            y=[1.0, 2.0],
            vol_series=[0.01, 0.02],
            locked_direction=1,
            sub_window_count=2,
            eps=eps,
        )
