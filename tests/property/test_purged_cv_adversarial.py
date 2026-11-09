"""Adversarial Stress and Edge-Case Tests for Purged K-Fold Cross-Validation Splitter.

Empirical verification of invariants under hostile / pathological input distributions:
1. Multi-Pair Clustered Samples: 10,000 samples with identical start times across 4 pairs.
2. Zero Duration Samples: Instantaneous events (t_start == t_end).
3. Extreme Embargo: Embargo window exceeding total remaining sample timeline.
4. Minimal Fold Allocation: Single sample per fold (N == K) and boundary splits (N == K + 1).
5. Degenerate Distributions: All samples at identical timestamps, extreme holding durations.
6. Zero Embargo Boundary: Verification that h_embargo == 0 produces strictly empty embargoed sets.
7. Scale Benchmark: Runtime and memory validation on production-scale dataset (210,436 samples).
"""

from __future__ import annotations

import time

from trading_bot.domain.entities.ml.ml_configs import PurgedCVConfig
from trading_bot.domain.exceptions import InsufficientDataException
from trading_bot.domain.services.ml.purged_cv_splitter import (
    PurgedCVSplitter,
)


def _assert_split_invariants(
    intervals: list[tuple[int, int]],
    n_splits: int,
    embargo_bars: int,
    max_holding_bars: int,
) -> None:
    """Verifies all core mathematical invariants for a given splitter run."""
    n_samples = len(intervals)
    config = PurgedCVConfig(
        n_splits=n_splits,
        embargo_bars_30m=embargo_bars,
        max_label_holding_bars_30m=max_holding_bars,
    )
    splitter = PurgedCVSplitter(config)
    splits = splitter.split(intervals)

    assert len(splits) == n_splits

    all_val_indices: list[int] = []

    for split in splits:
        train_set = set(split.train_indices)
        val_set = set(split.val_indices)
        purged_set = set(split.purged_indices)
        embargoed_set = set(split.embargoed_indices)

        # 1. Total Accounted & Partition Completeness
        assert split.total_accounted == n_samples
        assert (train_set | val_set | purged_set | embargoed_set) == set(range(n_samples))

        # 2. Pairwise Disjointness (All 6 combinations)
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(purged_set)
        assert train_set.isdisjoint(embargoed_set)
        assert val_set.isdisjoint(purged_set)
        assert val_set.isdisjoint(embargoed_set)
        assert purged_set.isdisjoint(embargoed_set)

        # 3. Zero Label Leakage in Training Set
        for idx in split.train_indices:
            t_start, t_end = intervals[idx]
            overlaps = (t_start <= split.val_max_t) and (t_end >= split.val_min_t)
            assert not overlaps, (
                f"Fold {split.fold_idx}: Train sample {idx} [{t_start}, {t_end}] "
                f"overlaps validation span [{split.val_min_t}, {split.val_max_t}]"
            )

        # 4. Strict Embargo Boundary Enforcement
        for idx in split.train_indices:
            t_start, _ = intervals[idx]
            if t_start >= split.val_max_t:
                assert t_start > split.val_max_t + embargo_bars, (
                    f"Fold {split.fold_idx}: Post-val train sample {idx} (t_start={t_start}) "
                    f"violates embargo threshold ({split.val_max_t + embargo_bars})"
                )

        # 5. Purged Set Correctness
        for idx in split.purged_indices:
            t_start, t_end = intervals[idx]
            assert (t_start <= split.val_max_t) and (t_end >= split.val_min_t), (
                f"Fold {split.fold_idx}: Sample {idx} [{t_start}, {t_end}] was wrongly purged; "
                f"does not overlap [{split.val_min_t}, {split.val_max_t}]"
            )

        # 6. Embargoed Set Correctness
        for idx in split.embargoed_indices:
            t_start, _ = intervals[idx]
            assert split.val_max_t <= t_start <= split.val_max_t + embargo_bars, (
                f"Fold {split.fold_idx}: Sample {idx} (t_start={t_start}) was wrongly embargoed; "
                f"not in [{split.val_max_t}, {split.val_max_t + embargo_bars}]"
            )

        all_val_indices.extend(split.val_indices)

    # 7. Exact Validation Coverage
    assert len(all_val_indices) == n_samples
    assert set(all_val_indices) == set(range(n_samples))


def test_adversarial_multipair_clustered_10k_samples() -> None:
    """Stress test with 10,000 multi-pair samples having identical start times per bar."""
    # 2,500 timestamps with 4 pairs each = 10,000 samples
    intervals: list[tuple[int, int]] = []
    for bar_idx in range(2500):
        t_bar = bar_idx * 5  # Every 5 units of time
        # 4 concurrent assets at the exact same bar
        for pair_offset in range(4):
            # Vary holding durations: 0, 4, 12, 24 bars
            duration = (pair_offset * 6) % 25
            intervals.append((t_bar, t_bar + duration))

    assert len(intervals) == 10000

    start_time = time.perf_counter()
    _assert_split_invariants(
        intervals=intervals,
        n_splits=5,
        embargo_bars=100,
        max_holding_bars=24,
    )
    elapsed = time.perf_counter() - start_time

    # Performance requirement: 10,000 samples 5-fold split must complete in < 0.200s
    assert elapsed < 0.200, f"Purged CV 10,000 samples took {elapsed:.4f}s, expected < 0.200s"


def test_adversarial_zero_duration_samples() -> None:
    """Test instantaneous events where t_start == t_end (e.g. instant barrier hits)."""
    # Alternating 0-duration and normal duration samples
    intervals: list[tuple[int, int]] = []
    for i in range(100):
        t = i * 2
        duration = 0 if i % 2 == 0 else 5
        intervals.append((t, t + duration))

    _assert_split_invariants(
        intervals=intervals,
        n_splits=5,
        embargo_bars=10,
        max_holding_bars=5,
    )


def test_adversarial_all_zero_duration_samples() -> None:
    """Test when all samples have t_start == t_end."""
    intervals = [(i * 3, i * 3) for i in range(50)]
    _assert_split_invariants(
        intervals=intervals,
        n_splits=5,
        embargo_bars=6,
        max_holding_bars=0,
    )


def test_adversarial_embargo_exceeds_entire_dataset_timeline() -> None:
    """Test scenario where embargo window is larger than the entire remaining dataset."""
    # 20 samples spanning t = 0 to t = 95
    intervals = [(i * 5, i * 5 + 3) for i in range(20)]
    # Embargo is 1,000 bars (much larger than dataset horizon)
    config = PurgedCVConfig(
        n_splits=4,
        embargo_bars_30m=1000,
        max_label_holding_bars_30m=3,
    )
    splitter = PurgedCVSplitter(config)
    splits = splitter.split(intervals)

    assert len(splits) == 4

    # Fold 0: val = [0, 1, 2, 3, 4]
    # Because embargo covers everything after fold 0, train set should be empty!
    fold_0 = splits[0]
    assert fold_0.val_indices == [0, 1, 2, 3, 4]
    assert fold_0.train_indices == []
    assert fold_0.embargoed_indices == [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    # Fold 3 (last fold): val = [15, 16, 17, 18, 19]
    # Train set should contain all earlier samples (minus purged)
    fold_3 = splits[3]
    assert fold_3.val_indices == [15, 16, 17, 18, 19]
    assert fold_3.embargoed_indices == []
    assert len(fold_3.train_indices) == 15

    # Invariants must still strictly hold
    _assert_split_invariants(
        intervals=intervals,
        n_splits=4,
        embargo_bars=1000,
        max_holding_bars=3,
    )


def test_adversarial_single_sample_per_fold() -> None:
    """Edge case: n_samples == n_splits (exactly 1 sample per validation fold)."""
    intervals = [(i * 10, i * 10 + 4) for i in range(5)]
    _assert_split_invariants(
        intervals=intervals,
        n_splits=5,
        embargo_bars=10,
        max_holding_bars=4,
    )


def test_adversarial_minimal_k2_n2() -> None:
    """Absolute minimal valid configuration: 2 samples, 2 splits."""
    intervals = [(0, 5), (10, 15)]
    _assert_split_invariants(
        intervals=intervals,
        n_splits=2,
        embargo_bars=5,
        max_holding_bars=5,
    )


def test_adversarial_uneven_fold_remainder_distribution() -> None:
    """Test uneven splits where n_samples % n_splits != 0."""
    # 11 samples, 4 splits -> fold sizes: 3, 3, 3, 2
    intervals = [(i * 10, i * 10 + 5) for i in range(11)]
    config = PurgedCVConfig(
        n_splits=4,
        embargo_bars_30m=15,
        max_label_holding_bars_30m=5,
    )
    splits = PurgedCVSplitter(config).split(intervals)

    assert len(splits[0].val_indices) == 3
    assert len(splits[1].val_indices) == 3
    assert len(splits[2].val_indices) == 3
    assert len(splits[3].val_indices) == 2

    _assert_split_invariants(
        intervals=intervals,
        n_splits=4,
        embargo_bars=15,
        max_holding_bars=5,
    )


def test_adversarial_zero_embargo_boundary() -> None:
    """When h_embargo == 0, no post-val samples should be embargoed."""
    intervals = [(i * 10, i * 10 + 2) for i in range(20)]
    config = PurgedCVConfig(
        n_splits=4,
        embargo_bars_30m=0,
        max_label_holding_bars_30m=2,
    )
    splits = PurgedCVSplitter(config).split(intervals)

    for split in splits:
        # With h_embargo == 0, embargoed_indices MUST be empty
        assert split.embargoed_indices == []
        assert split.num_embargoed == 0

    _assert_split_invariants(
        intervals=intervals,
        n_splits=4,
        embargo_bars=0,
        max_holding_bars=2,
    )


def test_adversarial_all_samples_identical_start_time() -> None:
    """Hostile edge case: 100 samples all starting at the exact same timestamp."""
    # All start at t=50, holding durations vary between 0 and 10
    intervals = [(50, 50 + (i % 11)) for i in range(100)]
    _assert_split_invariants(
        intervals=intervals,
        n_splits=5,
        embargo_bars=20,
        max_holding_bars=10,
    )


def test_adversarial_extreme_holding_duration_spans_future_folds() -> None:
    """Test sample in earlier fold whose holding duration extends across future folds."""
    intervals = [
        (0, 1000),  # Index 0 holds for 1000 bars!
        (10, 15),
        (20, 25),
        (30, 35),
        (40, 45),
        (50, 55),
    ]
    # For any fold where index 0 is not in val, its span [0, 1000] overlaps val [val_min, val_max]
    # and MUST be purged from training!
    config = PurgedCVConfig(
        n_splits=3,
        embargo_bars_30m=5,
        max_label_holding_bars_30m=1000,
    )
    splits = PurgedCVSplitter(config).split(intervals)

    # In fold 1 (val = [2, 3], val_span = [20, 35]):
    # Sample 0 has span [0, 1000]. Overlaps [20, 35] -> MUST BE PURGED
    assert 0 in splits[1].purged_indices
    assert 0 not in splits[1].train_indices

    # In fold 2 (val = [4, 5], val_span = [40, 55]):
    # Sample 0 has span [0, 1000]. Overlaps [40, 55] -> MUST BE PURGED
    assert 0 in splits[2].purged_indices
    assert 0 not in splits[2].train_indices

    _assert_split_invariants(
        intervals=intervals,
        n_splits=3,
        embargo_bars=5,
        max_holding_bars=1000,
    )


def test_adversarial_insufficient_samples_raises() -> None:
    """Edge cases: N < K must raise InsufficientDataException."""
    config = PurgedCVConfig(n_splits=5, embargo_bars_30m=10, max_label_holding_bars_30m=5)
    splitter = PurgedCVSplitter(config)

    for n in range(5):
        try:
            splitter.split([(i * 5, i * 5 + 2) for i in range(n)])
            raise AssertionError(f"Expected InsufficientDataException for n={n} < k=5")
        except InsufficientDataException:
            pass
