"""Hypothesis Property-Based Tests for Purged K-Fold Cross-Validation (Zero Defaults).

Verifies mathematical invariants:
1. Zero Label Overlap Leakage: No training sample overlaps validation active span.
2. Embargo Integrity: No training sample after validation starts within [val_max_t, val_max_t + h_embargo].
3. Partition Completeness: Train, Val, Purged, and Embargoed sets form an exact partition of all samples.
4. Validation Coverage: Every sample is evaluated in exactly one validation fold.
5. Embargo Monotonicity: Increasing h_embargo monotonically shrinks or maintains train set size.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from trading_bot.domain.entities.ml.ml_configs import PurgedCVConfig
from trading_bot.domain.services.ml.purged_cv_splitter import (
    PurgedCVSplitter,
)


@st.composite
def sample_intervals_strategy(draw: st.DrawFn) -> list[tuple[int, int]]:
    """Generate chronologically sorted valid sample intervals (t_start, t_end)."""
    n_samples = draw(st.integers(min_value=5, max_value=200))
    # Generate non-decreasing start times (allowing concurrent bars with step=0)
    steps = draw(
        st.lists(
            st.integers(min_value=0, max_value=5),
            min_size=n_samples,
            max_size=n_samples,
        )
    )
    start_times: list[int] = []
    current_t = draw(st.integers(min_value=0, max_value=1000))
    for step in steps:
        current_t += step
        start_times.append(current_t)

    # Generate holding durations >= 0
    durations = draw(
        st.lists(
            st.integers(min_value=0, max_value=25),
            min_size=n_samples,
            max_size=n_samples,
        )
    )

    intervals = [(start_times[i], start_times[i] + durations[i]) for i in range(n_samples)]
    return intervals


@settings(max_examples=100, deadline=None)
@given(
    intervals=sample_intervals_strategy(),
    n_splits=st.integers(min_value=2, max_value=8),
    embargo_bars=st.integers(min_value=0, max_value=100),
    max_holding=st.integers(min_value=0, max_value=50),
)
def test_property_zero_leakage_and_embargo_integrity(
    intervals: list[tuple[int, int]],
    n_splits: int,
    embargo_bars: int,
    max_holding: int,
) -> None:
    """Invariant: Zero label overlap leakage and 100% embargo integrity across all splits."""
    if len(intervals) < n_splits:
        return

    config = PurgedCVConfig(
        n_splits=n_splits,
        embargo_bars_30m=embargo_bars,
        max_label_holding_bars_30m=max_holding,
    )
    splitter = PurgedCVSplitter(config)
    splits = splitter.split(intervals)

    assert len(splits) == n_splits
    n_samples = len(intervals)

    for split in splits:
        train_set = set(split.train_indices)
        val_set = set(split.val_indices)
        purged_set = set(split.purged_indices)
        embargoed_set = set(split.embargoed_indices)

        # Property 1a: Train and Val are strictly disjoint
        assert train_set.isdisjoint(val_set)

        # Property 1b: Zero Label Overlap Leakage
        for idx in split.train_indices:
            t_start, t_end = intervals[idx]
            overlaps = (t_start <= split.val_max_t) and (t_end >= split.val_min_t)
            assert not overlaps, (
                f"Fold {split.fold_idx}: Train sample {idx} span [{t_start}, {t_end}] "
                f"overlaps validation span [{split.val_min_t}, {split.val_max_t}]"
            )
            # Equivalent formulation: sample must be strictly before or strictly after
            assert (t_end < split.val_min_t) or (t_start > split.val_max_t)

        # Property 2: Embargo Integrity for post-validation training samples
        for idx in split.train_indices:
            t_start, _ = intervals[idx]
            if t_start >= split.val_max_t:
                assert t_start > split.val_max_t + embargo_bars, (
                    f"Fold {split.fold_idx}: Train sample {idx} start {t_start} "
                    f"violates embargo [{split.val_max_t}, {split.val_max_t + embargo_bars}]"
                )

        # Property 3: Partition Completeness & Pairwise Disjointness
        assert split.total_accounted == n_samples
        union_all = train_set | val_set | purged_set | embargoed_set
        assert union_all == set(range(n_samples))
        assert train_set.isdisjoint(purged_set)
        assert train_set.isdisjoint(embargoed_set)
        assert val_set.isdisjoint(purged_set)
        assert val_set.isdisjoint(embargoed_set)
        assert purged_set.isdisjoint(embargoed_set)

        # Property 4: Purged indices verification
        for p_idx in split.purged_indices:
            t_start, t_end = intervals[p_idx]
            assert (t_start <= split.val_max_t) and (t_end >= split.val_min_t)

        # Property 5: Embargoed indices verification
        for e_idx in split.embargoed_indices:
            t_start, _ = intervals[e_idx]
            assert split.val_max_t <= t_start <= split.val_max_t + embargo_bars


@settings(max_examples=50, deadline=None)
@given(
    intervals=sample_intervals_strategy(),
    n_splits=st.integers(min_value=2, max_value=8),
)
def test_property_validation_coverage(
    intervals: list[tuple[int, int]],
    n_splits: int,
) -> None:
    """Invariant: Every sample appears in the validation set of exactly one fold."""
    if len(intervals) < n_splits:
        return

    config = PurgedCVConfig(
        n_splits=n_splits,
        embargo_bars_30m=50,
        max_label_holding_bars_30m=12,
    )
    splitter = PurgedCVSplitter(config)
    splits = splitter.split(intervals)

    val_seen: list[int] = []
    for split in splits:
        val_seen.extend(split.val_indices)

    assert len(val_seen) == len(intervals)
    assert set(val_seen) == set(range(len(intervals)))


@settings(max_examples=50, deadline=None)
@given(
    intervals=sample_intervals_strategy(),
    n_splits=st.integers(min_value=2, max_value=5),
    h1=st.integers(min_value=0, max_value=50),
    h2_delta=st.integers(min_value=1, max_value=50),
)
def test_property_embargo_monotonicity(
    intervals: list[tuple[int, int]],
    n_splits: int,
    h1: int,
    h2_delta: int,
) -> None:
    """Invariant: Increasing embargo window monotonically shrinks or preserves train set."""
    if len(intervals) < n_splits:
        return

    h2 = h1 + h2_delta
    cfg1 = PurgedCVConfig(n_splits=n_splits, embargo_bars_30m=h1, max_label_holding_bars_30m=12)
    cfg2 = PurgedCVConfig(n_splits=n_splits, embargo_bars_30m=h2, max_label_holding_bars_30m=12)

    splits1 = PurgedCVSplitter(cfg1).split(intervals)
    splits2 = PurgedCVSplitter(cfg2).split(intervals)

    for s1, s2 in zip(splits1, splits2, strict=True):
        # Samples purged due to label overlap are identical (independent of h_embargo)
        assert s1.purged_indices == s2.purged_indices
        # Train set under larger embargo must be a subset of train set under smaller embargo
        assert set(s2.train_indices).issubset(set(s1.train_indices))
        # Embargoed set under larger embargo must be a superset of smaller embargo
        assert set(s1.embargoed_indices).issubset(set(s2.embargoed_indices))
