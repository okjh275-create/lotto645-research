"""Revision-aware read repository for M6-005 adaptive weights."""

from __future__ import annotations

from typing import Mapping, Sequence

from .adaptive_models import AdaptiveWeight, AdaptiveWeightDataset, StrategyKey
from .ranking import StrategyRanking
from .ranking_repository import RankingRepository


class AdaptiveWeightRepository:
    """Build adaptive datasets and retain only the latest derived weights.

    RC1 is intentionally memory-only. No SQLite schema or append-only
    learning record is changed.
    """

    def __init__(self, ranking_repository: RankingRepository) -> None:
        self.ranking_repository = ranking_repository
        self._dataset_cache_key: tuple[
            tuple[int, int],
            str | None,
            int,
        ] | None = None
        self._dataset_cache_value: AdaptiveWeightDataset | None = None
        self._latest_weights: dict[
            tuple[str | None, int],
            tuple[AdaptiveWeight, ...],
        ] = {}

    def build_dataset(
        self,
        *,
        rankings: Sequence[StrategyRanking],
        strategy_type: str | None,
        history_limit: int,
    ) -> AdaptiveWeightDataset:
        if history_limit <= 0:
            raise ValueError("history_limit must be positive")

        normalized_type = (
            None
            if strategy_type is None
            else self._required_text(
                strategy_type,
                field_name="strategy_type",
            )
        )
        revision = self.ranking_repository.repository_revision()
        cache_key = (revision, normalized_type, int(history_limit))

        if (
            self._dataset_cache_key == cache_key
            and self._dataset_cache_value is not None
        ):
            return self._dataset_cache_value

        dataset = AdaptiveWeightDataset(
            revision=revision,
            rankings=tuple(rankings),
            previous_weights=self._previous_weight_map(
                strategy_type=normalized_type,
                history_limit=history_limit,
            ),
        )
        self._dataset_cache_key = cache_key
        self._dataset_cache_value = dataset
        return dataset

    def remember(
        self,
        *,
        strategy_type: str | None,
        history_limit: int,
        weights: Sequence[AdaptiveWeight],
    ) -> None:
        normalized_type = (
            None
            if strategy_type is None
            else self._required_text(
                strategy_type,
                field_name="strategy_type",
            )
        )
        if history_limit <= 0:
            raise ValueError("history_limit must be positive")
        self._latest_weights[(normalized_type, int(history_limit))] = tuple(weights)

    def latest(
        self,
        *,
        strategy_type: str | None = None,
        history_limit: int = 100,
    ) -> tuple[AdaptiveWeight, ...]:
        normalized_type = (
            None
            if strategy_type is None
            else self._required_text(
                strategy_type,
                field_name="strategy_type",
            )
        )
        return self._latest_weights.get(
            (normalized_type, int(history_limit)),
            (),
        )

    def invalidate_cache(self) -> None:
        self._dataset_cache_key = None
        self._dataset_cache_value = None

    def _previous_weight_map(
        self,
        *,
        strategy_type: str | None,
        history_limit: int,
    ) -> Mapping[StrategyKey, float]:
        latest = self._latest_weights.get(
            (strategy_type, int(history_limit)),
            (),
        )
        return {
            item.strategy_key: item.current_weight
            for item in latest
        }

    @staticmethod
    def _required_text(value: str, *, field_name: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"{field_name} must not be empty")
        return normalized
