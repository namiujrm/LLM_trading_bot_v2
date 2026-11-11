"""Alpha Pruning Service (Clean Architecture, Invariant #9).

Enforces institutional quantitative Alpha Zoo feature capacity (35-50 features)
and prunes lowest-ranking features exhibiting IC decay (rolling IC < ic_decay_threshold)
or lowest SHAP/importance scores.

Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks.
Application layer depends ONLY on domain entities and standard library.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from trading_bot.domain.entities.quant.symbolic_alpha import SymbolicAlphaCandidate


class AlphaPruningConfig(BaseModel):
    """Configuration for Alpha Pruning feature zoo capacity."""

    min_features: int
    max_features: int
    ic_decay_threshold: float


class AlphaPruningResult(BaseModel):
    """Result of alpha zoo pruning and candidate integration."""

    active_features: list[str]
    pruned_features: list[str]
    added_features: list[str]

    def __iter__(self) -> Any:
        """Allow tuple unpacking (active_features, pruned_features)."""
        return iter((self.active_features, self.pruned_features))


class AlphaPruningService:
    """Application service for Alpha Zoo capacity management and decayed feature pruning."""

    def __init__(
        self,
        min_features: int,
        max_features: int,
        ic_decay_threshold: float,
    ) -> None:
        if min_features <= 0:
            raise ValueError(f"min_features must be positive, got {min_features}")
        if max_features < min_features:
            raise ValueError(
                f"max_features ({max_features}) cannot be less than min_features ({min_features})"
            )
        if ic_decay_threshold < 0.0:
            raise ValueError(f"ic_decay_threshold cannot be negative, got {ic_decay_threshold}")
        self._min_features: int = min_features
        self._max_features: int = max_features
        self._ic_decay_threshold: float = ic_decay_threshold

    @classmethod
    def from_config(cls, config: AlphaPruningConfig) -> AlphaPruningService:
        """Instantiate service from explicit configuration schema."""
        return cls(
            min_features=config.min_features,
            max_features=config.max_features,
            ic_decay_threshold=config.ic_decay_threshold,
        )

    @property
    def min_features(self) -> int:
        return self._min_features

    @property
    def max_features(self) -> int:
        return self._max_features

    @property
    def ic_decay_threshold(self) -> float:
        return self._ic_decay_threshold

    def prune_and_update_zoo(
        self,
        current_zoo_features: list[str],
        decayed_feature_metrics: dict[str, float],
        new_passed_candidates: list[SymbolicAlphaCandidate],
    ) -> AlphaPruningResult:
        """Update Alpha Zoo with new candidates and prune lowest-ranking decayed features.

        - Adds new passed candidates (deduplicated against existing zoo).
        - If total features <= max_features, keeps all core features without pruning.
        - If total features > max_features, prunes features prioritizing those with
          rolling IC < ic_decay_threshold and lowest importance scores, maintaining
          at least min_features and not exceeding max_features.
        """
        combined_features: list[str] = list(current_zoo_features)
        added_features: list[str] = []

        for candidate in new_passed_candidates:
            if candidate.alpha_id not in combined_features:
                combined_features.append(candidate.alpha_id)
                added_features.append(candidate.alpha_id)

        # 1. Under-capacity: Keep all core features
        if len(combined_features) <= self._max_features:
            return AlphaPruningResult(
                active_features=combined_features,
                pruned_features=[],
                added_features=added_features,
            )

        # 2. Over-capacity: Pruning required
        min_to_prune: int = len(combined_features) - self._max_features
        max_to_prune: int = len(combined_features) - self._min_features

        if max_to_prune <= 0:
            # Cannot prune without dropping below min_features; cap to max_features
            pruned_ids = combined_features[self._max_features :]
            active_ids = combined_features[: self._max_features]
            return AlphaPruningResult(
                active_features=active_ids,
                pruned_features=pruned_ids,
                added_features=[a for a in added_features if a in active_ids],
            )

        # Classify existing zoo features by metric score
        decayed_features: list[tuple[str, float]] = []
        healthy_features: list[tuple[str, float]] = []
        unflagged_features: list[str] = []

        for feat in current_zoo_features:
            if feat in decayed_feature_metrics:
                score = decayed_feature_metrics[feat]
                if score < self._ic_decay_threshold:
                    decayed_features.append((feat, score))
                else:
                    healthy_features.append((feat, score))
            else:
                unflagged_features.append(feat)

        # Sort decayed features ascending (lowest rolling IC / SHAP pruned first)
        decayed_features.sort(key=lambda item: item[1])
        # Sort healthy features ascending (lowest metric pruned first if needed)
        healthy_features.sort(key=lambda item: item[1])

        # Candidate order for pruning: decayed first, then healthy, then unflagged
        prune_candidates: list[str] = (
            [f for f, _ in decayed_features] + [f for f, _ in healthy_features] + unflagged_features
        )

        # Determine prune count: prune all decayed features if possible,
        # but at least min_to_prune and at most max_to_prune.
        num_decayed = len(decayed_features)
        if num_decayed >= min_to_prune:
            prune_count = min(num_decayed, max_to_prune)
        else:
            prune_count = min(min_to_prune, max_to_prune)

        pruned_set: set[str] = set(prune_candidates[:prune_count])
        active_features: list[str] = [f for f in combined_features if f not in pruned_set]
        pruned_list: list[str] = list(prune_candidates[:prune_count])

        return AlphaPruningResult(
            active_features=active_features,
            pruned_features=pruned_list,
            added_features=[a for a in added_features if a in active_features],
        )
