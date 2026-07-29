"""Incremental learning, aggregation, and strategy-ranking service."""

from __future__ import annotations

from dataclasses import dataclass
import time

from .aggregator import (
    StrategyAggregationSummary,
    StrategyAggregator,
)
from .evaluator import evaluate_prediction
from .ranking import (
    RankingConfig,
    StrategyRanking,
    StrategyRankingEngine,
)
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
    ) -> None:
        self.repository = repository
        self.strategy_aggregator = StrategyAggregator(
            repository
        )
        self.ranking_repository = RankingRepository(
            repository
        )
        self.ranking_engine = StrategyRankingEngine(
            ranking_config
        )

    def run_incremental_review(
        self,
        *,
        round_no: int | None = None,
        limit: int | None = None,
        reviewed_at_kst: str | None = None,
    ) -> IncrementalReviewSummary:
        started = time.perf_counter()

        pending = (
            self.repository
            .pending_predictions_with_results(
                round_no=round_no,
                limit=limit,
            )
        )

        created = 0
        skipped = 0
        result_cache = {}

        for prediction in pending:
            result = result_cache.get(
                prediction.round_no
            )

            if result is None:
                result = self.repository.get_result(
                    prediction.round_no
                )

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
            elapsed_seconds=round(
                time.perf_counter() - started,
                6,
            ),
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
            self.ranking_repository.invalidate_cache()

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

        dataset = self.ranking_repository.build_dataset(
            strategy_type=strategy_type,
            history_limit=history_limit,
        )

        return self.ranking_engine.rank(
            dataset.statistics,
            dataset.histories,
        )
