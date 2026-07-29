"""Incremental learning, aggregation, ranking, and adaptive service."""

from __future__ import annotations

from dataclasses import dataclass
import time

from .adaptive_engine import AdaptiveWeightConfig, AdaptiveWeightEngine
from .adaptive_models import AdaptiveWeight
from .adaptive_repository import AdaptiveWeightRepository
from .aggregator import StrategyAggregationSummary, StrategyAggregator
from .evaluator import evaluate_prediction
from .learning_facade import LearningFacade
from .ranking import RankingConfig, StrategyRanking, StrategyRankingEngine
from .ranking_repository import RankingRepository
from .repository import LearningRepository
from .strategy_stats import StrategyStatistics


@dataclass(frozen=True, slots=True)
class IncrementalReviewSummary:
    scanned: int
    created: int
    skipped: int
    elapsed_seconds: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "scanned": self.scanned,
            "created": self.created,
            "skipped": self.skipped,
            "elapsed_seconds": self.elapsed_seconds,
        }


class LearningService:
    """Stable entry point for learning workflows."""

    def __init__(
        self,
        repository: LearningRepository,
        *,
        ranking_config: RankingConfig | None = None,
        adaptive_config: AdaptiveWeightConfig | None = None,
    ) -> None:
        self.repository = repository
        self.strategy_aggregator = StrategyAggregator(repository)
        self.ranking_repository = RankingRepository(repository)
        self.ranking_engine = StrategyRankingEngine(ranking_config)
        self.adaptive_repository = AdaptiveWeightRepository(
            self.ranking_repository
        )
        self.adaptive_engine = AdaptiveWeightEngine(adaptive_config)
        self.learning_facade = LearningFacade(
            ranking_repository=self.ranking_repository,
            ranking_engine=self.ranking_engine,
            adaptive_repository=self.adaptive_repository,
            adaptive_engine=self.adaptive_engine,
        )

    def run_incremental_review(
        self,
        *,
        round_no: int | None = None,
        limit: int | None = None,
        reviewed_at_kst: str | None = None,
    ) -> IncrementalReviewSummary:
        started = time.perf_counter()
        pending = self.repository.pending_predictions_with_results(
            round_no=round_no,
            limit=limit,
        )

        created = 0
        skipped = 0
        result_cache = {}

        for prediction in pending:
            result = result_cache.get(prediction.round_no)
            if result is None:
                result = self.repository.get_result(prediction.round_no)
                if result is None:
                    skipped += 1
                    continue
                result_cache[prediction.round_no] = result

            review = evaluate_prediction(
                prediction,
                result,
                reviewed_at_kst=reviewed_at_kst,
            )
            if self.repository.add_review(review):
                created += 1
            else:
                skipped += 1

        return IncrementalReviewSummary(
            scanned=len(pending),
            created=created,
            skipped=skipped,
            elapsed_seconds=round(time.perf_counter() - started, 6),
        )

    def run_strategy_aggregation(
        self,
        *,
        limit: int | None = None,
        aggregated_at_kst: str | None = None,
    ) -> StrategyAggregationSummary:
        summary = self.strategy_aggregator.run(
            limit=limit,
            aggregated_at_kst=aggregated_at_kst,
        )
        if summary.created_events:
            self.learning_facade.invalidate_derived_caches()
        return summary

    def get_strategy_statistics(
        self,
        *,
        strategy_type: str | None = None,
    ) -> tuple[StrategyStatistics, ...]:
        return self.repository.get_strategy_statistics(
            strategy_type=strategy_type
        )

    def rank_strategies(
        self,
        *,
        strategy_type: str | None = None,
        history_limit: int = 100,
    ) -> tuple[StrategyRanking, ...]:
        """Return deterministic rankings from current aggregated evidence."""
        return self.learning_facade.rank_strategies(
            strategy_type=strategy_type,
            history_limit=history_limit,
        )

    def get_adaptive_weights(
        self,
        *,
        strategy_type: str | None = None,
        history_limit: int = 100,
    ) -> tuple[AdaptiveWeight, ...]:
        """Return bounded and normalized adaptive strategy priorities."""
        return self.learning_facade.get_adaptive_weights(
            strategy_type=strategy_type,
            history_limit=history_limit,
        )
