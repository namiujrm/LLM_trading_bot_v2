"""Purged K-Fold Cross-Validation Splitter with Dynamic Memory Embargo.

Pure Python domain service implementing López de Prado's Purged K-Fold Cross-Validation:
- Removes label overlap leakage by purging all training samples whose evaluation span
  [t_start_i, t_end_i] overlaps with the validation interval [min(val_start), max(val_end)].
- Removes autoregressive feature memory leakage by embargoing all training samples
  occurring within h_embargo bars after the validation fold exit time.

Strict Clean Architecture: 100% Pure Python, zero numpy/polars, zero default arguments (Invariant #9).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from trading_bot.domain.entities.ml.ml_configs import PurgedCVConfig
from trading_bot.domain.exceptions import InsufficientDataException


@dataclass(frozen=True)
class PurgedCVSplit:
    """Represents a single fold split in Purged K-Fold Cross-Validation.

    Invariant #9: All fields are strictly required without defaults.
    """

    fold_idx: int
    train_indices: list[int]
    val_indices: list[int]
    val_min_t: int
    val_max_t: int
    purged_indices: list[int]
    embargoed_indices: list[int]

    def __iter__(self) -> Iterator[list[int]]:
        """Allow standard tuple unpacking: train_idx, val_idx = split."""
        yield self.train_indices
        yield self.val_indices

    @property
    def num_train(self) -> int:
        return len(self.train_indices)

    @property
    def num_val(self) -> int:
        return len(self.val_indices)

    @property
    def num_purged(self) -> int:
        return len(self.purged_indices)

    @property
    def num_embargoed(self) -> int:
        return len(self.embargoed_indices)

    @property
    def total_accounted(self) -> int:
        return (
            len(self.train_indices)
            + len(self.val_indices)
            + len(self.purged_indices)
            + len(self.embargoed_indices)
        )


class PurgedCVSplitter:
    """Pure domain service for generating Purged & Embargoed Cross-Validation splits."""

    def __init__(self, config: PurgedCVConfig) -> None:
        """Initialize the splitter with an explicit configuration schema.

        Args:
            config: PurgedCVConfig specifying n_splits, embargo_bars_30m, max_label_holding_bars_30m.
        """
        if config.n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, received {config.n_splits}")
        if config.embargo_bars_30m < 0:
            raise ValueError(
                f"embargo_bars_30m must be non-negative, received {config.embargo_bars_30m}"
            )
        if config.max_label_holding_bars_30m < 0:
            raise ValueError(
                f"max_label_holding_bars_30m must be non-negative, received {config.max_label_holding_bars_30m}"
            )
        self._config: PurgedCVConfig = config

    @property
    def config(self) -> PurgedCVConfig:
        return self._config

    def split(
        self,
        sample_intervals: Sequence[tuple[int, int]],
    ) -> list[PurgedCVSplit]:
        """Generate purged and embargoed K-Fold train/val splits.

        Args:
            sample_intervals: Sequence of (t_start_i, t_end_i) integer intervals for N samples,
                              sorted by t_start_i in non-decreasing order.

        Returns:
            list[PurgedCVSplit]: Exactly K split objects, each containing train, val, purged,
                                and embargoed sample indices, along with validation time bounds.

        Raises:
            InsufficientDataException: If sample count N is less than n_splits.
            ValueError: If sample intervals violate causality (t_end < t_start),
                        negative time, or temporal monotonicity (t_start[i] < t_start[i-1]).
        """
        n_samples = len(sample_intervals)
        n_splits = self._config.n_splits

        if n_samples < n_splits:
            raise InsufficientDataException(
                f"Number of samples ({n_samples}) must be greater than or equal to n_splits ({n_splits})."
            )

        # Validate causality and chronological ordering across all intervals
        for i, (t_start, t_end) in enumerate(sample_intervals):
            if t_start < 0:
                raise ValueError(f"Sample at index {i} has negative start time: {t_start}.")
            if t_end < t_start:
                raise ValueError(
                    f"Sample at index {i} violates causality: exit time {t_end} is before entry time {t_start}."
                )
            if i > 0 and t_start < sample_intervals[i - 1][0]:
                raise ValueError(
                    f"Sample intervals must be chronologically ordered by start time. "
                    f"Sample at index {i} (t_start={t_start}) precedes index {i - 1} "
                    f"(t_start={sample_intervals[i - 1][0]})."
                )

        fold_size = n_samples // n_splits
        remainder = n_samples % n_splits
        h_embargo = self._config.embargo_bars_30m

        splits: list[PurgedCVSplit] = []

        for fold_idx in range(n_splits):
            val_start_idx = fold_idx * fold_size + min(fold_idx, remainder)
            val_end_idx = (fold_idx + 1) * fold_size + min(fold_idx + 1, remainder)
            val_indices = list(range(val_start_idx, val_end_idx))
            val_indices_set = set(val_indices)

            val_min_t = sample_intervals[val_indices[0]][0]
            val_max_t = max(sample_intervals[j][1] for j in val_indices)

            train_indices: list[int] = []
            purged_indices: list[int] = []
            embargoed_indices: list[int] = []

            for i, (t_start_i, t_end_i) in enumerate(sample_intervals):
                if i in val_indices_set:
                    continue

                # Purging: Label span overlap with validation interval [val_min_t, val_max_t]
                # Closed intervals [t_start_i, t_end_i] and [val_min_t, val_max_t] overlap
                # if and only if: (t_start_i <= val_max_t) and (t_end_i >= val_min_t)
                if (t_start_i <= val_max_t) and (t_end_i >= val_min_t):
                    purged_indices.append(i)
                # Dynamic Memory Embargo: Non-purged post-validation samples starting within h_embargo bars after validation exit
                elif val_max_t <= t_start_i <= (val_max_t + h_embargo):
                    embargoed_indices.append(i)
                else:
                    train_indices.append(i)

            splits.append(
                PurgedCVSplit(
                    fold_idx=fold_idx,
                    train_indices=train_indices,
                    val_indices=val_indices,
                    val_min_t=val_min_t,
                    val_max_t=val_max_t,
                    purged_indices=purged_indices,
                    embargoed_indices=embargoed_indices,
                )
            )

        return splits
