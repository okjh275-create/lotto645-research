"""Derived learning-domain orchestration for ranking and adaptive weights."""

from __future__ import annotations

from .adaptive_engine import AdaptiveWeightEngine
from .adaptive_models import AdaptiveWeight
from .adaptive_repository import AdaptiveWeightRepository
from .ranking import StrategyRanking, StrategyRankingEngine
from .ranking_repository import RankingRepository


class LearningFacade:
    """Coordinate derived ranking and adaptive-weight layers."""

    def __init__(
        self,
        *,
        ranking_repository: RankingRepository,
        ranking_engine: StrategyRankingEngine,
        adaptive_repository: AdaptiveWeightRepository,
        adaptive_engine: AdaptiveWeightEngine,
    ) -> None:
        self.ranking_repository = ranking_repository
        self.ranking_engine = ranking_engine
        self.adaptive_repository = adaptive_repository
        self.adaptive_engine = adaptive_engine

    def rank_strategies(
        self,
        *,
        strategy_type: str | None = None,
        history_limit: int = 100,
    ) -> tuple[StrategyRanking, ...]:
        dataset = self.ranking_repository.build_dataset(
            strategy_type=strategy_type,
            history_limit=history_limit,
        )
        return self.ranking_engine.rank(
            dataset.statistics,
            dataset.histories,
        )

    def get_adaptive_weights(
        self,
        *,
        strategy_type: str | None = None,
        history_limit: int = 100,
    ) -> tuple[AdaptiveWeight, ...]:
        rankings = self.rank_strategies(
            strategy_type=strategy_type,
            history_limit=history_limit,
        )
        dataset = self.adaptive_repository.build_dataset(
            rankings=rankings,
            strategy_type=strategy_type,
            history_limit=history_limit,
        )
        weights = self.adaptive_engine.calculate(dataset)
        self.adaptive_repository.remember(
            strategy_type=strategy_type,
            history_limit=history_limit,
            weights=weights,
        )
        return weights

    def invalidate_derived_caches(self) -> None:
        self.ranking_repository.invalidate_cache()
        self.adaptive_repository.invalidate_cache()
