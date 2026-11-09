"""Deterministic Unit Tests for Purged K-Fold Cross-Validation Splitter (Zero Defaults)."""

from __future__ import annotations

import pytest

from trading_bot.domain.entities.ml.ml_configs import PurgedCVConfig
from trading_bot.domain.exceptions import InsufficientDataException
from trading_bot.domain.services.ml.purged_cv_splitter import (
    PurgedCVSplitter,
)


def test_handcrafted_5fold_split_purging_and_embargo() -> None:
    """Test with a deterministic 10-sample sequence with known holding and embargo."""
    # 10 samples at t_start = 0, 10, 20, 30, 40, 50, 60, 70, 80, 90
    # durations = 5 bars for each sample (t_end = t_start + 5)
    intervals = [(i * 10, i * 10 + 5) for i in range(10)]
    config = PurgedCVConfig(n_splits=5, embargo_bars_30m=15, max_label_holding_bars_30m=12)
    splitter = PurgedCVSplitter(config)
    splits = splitter.split(intervals)

    assert len(splits) == 5

    # Fold 0: val = [0, 1] -> t_start in [0, 10], t_end in [5, 15]
    # val_min_t = 0, val_max_t = 15
    # Embargo interval: [15, 15 + 15] = [15, 30]
    # Sample 2: [20, 25] -> starts at 20 <= 30 -> EMBARGOED
    # Sample 3: [30, 35] -> starts at 30 <= 30 -> EMBARGOED
    # Sample 4: [40, 45] -> starts at 40 > 30 -> TRAIN
    s0 = splits[0]
    assert s0.val_indices == [0, 1]
    assert s0.val_min_t == 0
    assert s0.val_max_t == 15
    assert s0.purged_indices == []
    assert s0.embargoed_indices == [2, 3]
    assert s0.train_indices == [4, 5, 6, 7, 8, 9]
    assert s0.num_train == 6
    assert s0.num_val == 2
    assert s0.num_purged == 0
    assert s0.num_embargoed == 2
    assert s0.total_accounted == 10


def test_label_overlap_purging_before_validation() -> None:
    """Test sample before validation that holds until after validation starts."""
    intervals = [
        (0, 15),  # Index 0: enters at 0, holds until 15 (OVERLAPS with sample 1 entering at 10)
        (10, 18),  # Index 1: enters at 10, holds until 18
        (30, 35),  # Index 2: post-val
        (50, 55),  # Index 3: post-val
    ]
    config = PurgedCVConfig(n_splits=2, embargo_bars_30m=5, max_label_holding_bars_30m=20)
    splits = PurgedCVSplitter(config).split(intervals)

    # Fold 1: val = [2, 3] -> val_min_t = 30, val_max_t = 55
    # Fold 0: val = [0, 1] -> val_min_t = 0, val_max_t = 18
    assert splits[0].val_indices == [0, 1]
    # In Fold 0: sample 2 enters at 30 > 18 + 5 = 23 -> train
    assert splits[0].train_indices == [2, 3]

    # In Fold 1: val = [2, 3], val_min_t = 30, val_max_t = 55
    # Sample 0: [0, 15] -> t_end = 15 < val_min_t = 30 -> no overlap -> TRAIN
    # Sample 1: [10, 18] -> t_end = 18 < val_min_t = 30 -> no overlap -> TRAIN
    assert splits[1].train_indices == [0, 1]
    assert splits[1].purged_indices == []


def test_fail_fast_on_insufficient_samples() -> None:
    config = PurgedCVConfig(n_splits=5, embargo_bars_30m=10, max_label_holding_bars_30m=10)
    splitter = PurgedCVSplitter(config)
    with pytest.raises(
        InsufficientDataException, match="must be greater than or equal to n_splits"
    ):
        splitter.split([(0, 5), (1, 6)])


def test_fail_fast_on_causality_violation() -> None:
    config = PurgedCVConfig(n_splits=2, embargo_bars_30m=10, max_label_holding_bars_30m=10)
    splitter = PurgedCVSplitter(config)
    with pytest.raises(ValueError, match="violates causality"):
        splitter.split([(10, 5), (20, 25)])


def test_fail_fast_on_unordered_samples() -> None:
    config = PurgedCVConfig(n_splits=2, embargo_bars_30m=10, max_label_holding_bars_30m=10)
    splitter = PurgedCVSplitter(config)
    with pytest.raises(ValueError, match="chronologically ordered"):
        splitter.split([(20, 25), (10, 15)])


def test_fail_fast_on_negative_time() -> None:
    config = PurgedCVConfig(n_splits=2, embargo_bars_30m=10, max_label_holding_bars_30m=10)
    splitter = PurgedCVSplitter(config)
    with pytest.raises(ValueError, match="negative start time"):
        splitter.split([(-1, 5), (10, 15)])


def test_fail_fast_on_invalid_config() -> None:
    with pytest.raises(ValueError, match="n_splits must be at least 2"):
        PurgedCVConfig(n_splits=1, embargo_bars_30m=10, max_label_holding_bars_30m=10)

    with pytest.raises(ValueError, match="embargo_bars_30m must be non-negative"):
        PurgedCVConfig(n_splits=5, embargo_bars_30m=-1, max_label_holding_bars_30m=10)

    with pytest.raises(ValueError, match="max_label_holding_bars_30m must be non-negative"):
        PurgedCVConfig(n_splits=5, embargo_bars_30m=10, max_label_holding_bars_30m=-5)


def test_purged_cv_config_factory_and_properties() -> None:
    config = PurgedCVConfig.create_with_dynamic_embargo(
        n_splits=5,
        feature_memory_lookback_bars=100,
        max_label_holding_bars=12,
    )
    assert config.embargo_bars == 100
    assert config.max_label_holding_bars == 12
    assert config.n_splits == 5


def test_tuple_unpacking_iteration() -> None:
    """Test standard unpack: train_idx, val_idx = split."""
    intervals = [(i * 10, i * 10 + 2) for i in range(10)]
    config = PurgedCVConfig(n_splits=5, embargo_bars_30m=5, max_label_holding_bars_30m=5)
    splits = PurgedCVSplitter(config).split(intervals)

    for split in splits:
        train_idx, val_idx = split
        assert train_idx == split.train_indices
        assert val_idx == split.val_indices
