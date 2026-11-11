"""Unit tests for AlphaPruningService (Milestone 7.3, Invariant #9)."""

from __future__ import annotations

import pytest

from trading_bot.application.services.alpha_pruning_service import (
    AlphaPruningConfig,
    AlphaPruningResult,
    AlphaPruningService,
)
from trading_bot.domain.entities.quant.symbolic_alpha import (
    CLUSTER_1_MICRO,
    SymbolicAlphaCandidate,
)


def _make_candidate(alpha_id: str) -> SymbolicAlphaCandidate:
    """Helper to create a valid SymbolicAlphaCandidate with zero defaults."""
    return SymbolicAlphaCandidate(
        alpha_id=alpha_id,
        dsl_expression=f"ts_zscore({alpha_id}, 16)",
        causal_cluster=CLUSTER_1_MICRO,
        long_monotonic_prior=1,
        short_monotonic_prior=-1,
        targeted_blindspot="false_positive_breakout",
        economic_rationale="Order flow imbalance momentum signal",
    )


def test_alpha_pruning_service_initialization_and_validation() -> None:
    """Verify constructor bounds checking and property accessors."""
    service = AlphaPruningService(
        min_features=35,
        max_features=50,
        ic_decay_threshold=0.012,
    )
    assert service.min_features == 35
    assert service.max_features == 50
    assert service.ic_decay_threshold == 0.012

    # Negative/zero min_features
    with pytest.raises(ValueError, match="min_features must be positive"):
        AlphaPruningService(
            min_features=0,
            max_features=50,
            ic_decay_threshold=0.012,
        )

    # max_features < min_features
    with pytest.raises(ValueError, match="cannot be less than min_features"):
        AlphaPruningService(
            min_features=50,
            max_features=35,
            ic_decay_threshold=0.012,
        )

    # negative ic_decay_threshold
    with pytest.raises(ValueError, match="cannot be negative"):
        AlphaPruningService(
            min_features=35,
            max_features=50,
            ic_decay_threshold=-0.01,
        )


def test_from_config() -> None:
    """Verify instantiation via AlphaPruningConfig."""
    cfg = AlphaPruningConfig(
        min_features=35,
        max_features=50,
        ic_decay_threshold=0.012,
    )
    service = AlphaPruningService.from_config(cfg)
    assert service.min_features == 35
    assert service.max_features == 50
    assert service.ic_decay_threshold == 0.012


def test_under_capacity_keeps_all_core_features() -> None:
    """Zoo cap requirement: When under max_features, all core features are kept."""
    service = AlphaPruningService(
        min_features=35,
        max_features=50,
        ic_decay_threshold=0.012,
    )
    current_zoo = [f"feat_{i}" for i in range(30)]
    decayed_metrics = {"feat_0": 0.005, "feat_1": 0.008}  # decayed, but under capacity
    new_candidates = [_make_candidate(f"cand_{i}") for i in range(5)]

    result = service.prune_and_update_zoo(
        current_zoo_features=current_zoo,
        decayed_feature_metrics=decayed_metrics,
        new_passed_candidates=new_candidates,
    )

    assert isinstance(result, AlphaPruningResult)
    assert len(result.active_features) == 35
    assert result.pruned_features == []
    assert len(result.added_features) == 5
    assert "feat_0" in result.active_features
    assert "cand_0" in result.active_features


def test_zoo_cap_enforcement_and_pruning_priority_for_decayed_features() -> None:
    """Zoo cap enforcement (35-50) and pruning priority for lowest rolling IC features."""
    service = AlphaPruningService(
        min_features=35,
        max_features=50,
        ic_decay_threshold=0.012,
    )
    # 48 existing features
    current_zoo = [f"feat_{i}" for i in range(48)]
    # 3 features are severely decayed, in order 0.001 < 0.004 < 0.009
    decayed_metrics = {
        "feat_2": 0.009,
        "feat_0": 0.001,
        "feat_1": 0.004,
        "feat_3": 0.025,  # healthy
    }
    # 4 new candidates added -> total 52 > 50
    new_candidates = [_make_candidate(f"cand_{i}") for i in range(4)]

    result = service.prune_and_update_zoo(
        current_zoo_features=current_zoo,
        decayed_feature_metrics=decayed_metrics,
        new_passed_candidates=new_candidates,
    )

    # Must prune at least 2 features to reach <= 50
    # Decayed features (feat_0, feat_1, feat_2) should all be pruned in ascending score order
    assert len(result.active_features) <= 50
    assert len(result.active_features) >= 35
    assert result.pruned_features == ["feat_0", "feat_1", "feat_2"]
    assert "feat_0" not in result.active_features
    assert "feat_1" not in result.active_features
    assert "feat_2" not in result.active_features
    assert "feat_3" in result.active_features
    assert len(result.active_features) == 49


def test_pruning_more_decayed_features_within_bounds() -> None:
    """When multiple decayed features exist and capacity allows, prune all of them."""
    service = AlphaPruningService(
        min_features=35,
        max_features=50,
        ic_decay_threshold=0.012,
    )
    current_zoo = [f"feat_{i}" for i in range(48)]
    decayed_metrics = {f"feat_{i}": 0.002 + 0.001 * i for i in range(6)}  # feat_0..feat_5 < 0.012
    new_candidates = [_make_candidate(f"cand_{i}") for i in range(4)]
    # total = 52. 6 decayed features pruned -> 46 remaining (>= 35).

    result = service.prune_and_update_zoo(
        current_zoo_features=current_zoo,
        decayed_feature_metrics=decayed_metrics,
        new_passed_candidates=new_candidates,
    )

    assert len(result.pruned_features) == 6
    assert result.pruned_features == [f"feat_{i}" for i in range(6)]
    assert len(result.active_features) == 46


def test_prunes_lowest_healthy_when_insufficient_decayed_to_satisfy_cap() -> None:
    """When decayed features alone cannot bring zoo to max_features, prune lowest healthy features."""
    service = AlphaPruningService(
        min_features=35,
        max_features=50,
        ic_decay_threshold=0.012,
    )
    current_zoo = [f"feat_{i}" for i in range(49)]
    # Only 1 feature is decayed, but 4 new candidates added (total 53). Need 3 pruned.
    decayed_metrics = {
        "feat_0": 0.005,  # decayed (< 0.012)
        "feat_1": 0.015,  # healthy
        "feat_2": 0.018,  # healthy
        "feat_3": 0.025,  # healthy
    }
    new_candidates = [_make_candidate(f"cand_{i}") for i in range(4)]

    result = service.prune_and_update_zoo(
        current_zoo_features=current_zoo,
        decayed_feature_metrics=decayed_metrics,
        new_passed_candidates=new_candidates,
    )

    assert len(result.active_features) == 50
    assert len(result.pruned_features) == 3
    # feat_0 (decayed) + feat_1, feat_2 (lowest healthy)
    assert result.pruned_features == ["feat_0", "feat_1", "feat_2"]


def test_min_features_bound_never_violated() -> None:
    """Ensure pruning never drops zoo feature count below min_features."""
    service = AlphaPruningService(
        min_features=35,
        max_features=36,
        ic_decay_threshold=0.012,
    )
    current_zoo = [f"feat_{i}" for i in range(35)]
    # All 35 features are decayed
    decayed_metrics = {f"feat_{i}": 0.001 for i in range(35)}
    # 2 candidates added -> total 37. Max prunable is 37 - 35 = 2.
    new_candidates = [_make_candidate(f"cand_{i}") for i in range(2)]

    result = service.prune_and_update_zoo(
        current_zoo_features=current_zoo,
        decayed_feature_metrics=decayed_metrics,
        new_passed_candidates=new_candidates,
    )

    assert len(result.active_features) == 35
    assert len(result.pruned_features) == 2


def test_candidate_deduplication_and_tuple_unpacking() -> None:
    """Verify candidate deduplication against existing zoo and tuple unpacking."""
    service = AlphaPruningService(
        min_features=35,
        max_features=50,
        ic_decay_threshold=0.012,
    )
    current_zoo = ["alpha_existing_1", "alpha_existing_2"]
    new_candidates = [
        _make_candidate("alpha_existing_1"),  # already exists
        _make_candidate("alpha_new_1"),
    ]

    active, pruned = service.prune_and_update_zoo(
        current_zoo_features=current_zoo,
        decayed_feature_metrics={},
        new_passed_candidates=new_candidates,
    )

    assert active == ["alpha_existing_1", "alpha_existing_2", "alpha_new_1"]
    assert pruned == []
