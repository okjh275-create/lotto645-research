"""Incremental strategy-statistics aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import time
from zoneinfo import ZoneInfo

from .repository import LearningRepository


_KST = ZoneInfo("Asia/Seoul")


@dataclass(frozen=True, slots=True)
class StrategyAggregationSummary:
    scanned: int
    created_events: int
    skipped_events: int
    elapsed_seconds: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "scanned": self.scanned,
            "created_events": self.created_events,
            "skipped_events": self.skipped_events,
            "elapsed_seconds": self.elapsed_seconds,
        }


class StrategyAggregator:
    """Incrementally aggregate reviewed prediction performance."""

    def __init__(
        self,
        repository: LearningRepository,
    ) -> None:
        self.repository = repository

    def run(
        self,
        *,
        limit: int | None = None,
        aggregated_at_kst: str | None = None,
    ) -> StrategyAggregationSummary:
        started = time.perf_counter()

        timestamp = aggregated_at_kst
        if timestamp is None:
            timestamp = datetime.now(_KST).isoformat(
                timespec="seconds"
            )

        pending = self.repository.pending_strategy_reviews(
            limit=limit
        )

        created = 0
        skipped = 0

        for prediction, review in pending:
            strategy_keys = [
                ("model", prediction.model_name),
            ]

            scenario = prediction.parameters.get("scenario")
            if scenario:
                strategy_keys.append(
                    ("scenario", str(scenario))
                )

            for strategy_type, strategy_name in strategy_keys:
                inserted = (
                    self.repository
                    .apply_strategy_stat_event(
                        prediction_id=(
                            prediction.prediction_id
                        ),
                        strategy_type=strategy_type,
                        strategy_name=strategy_name,
                        match_count=review.match_count,
                        prediction_score=prediction.score,
                        prize_rank=review.prize_rank,
                        aggregated_at_kst=timestamp,
                    )
                )

                if inserted:
                    created += 1
                else:
                    skipped += 1

        return StrategyAggregationSummary(
            scanned=len(pending),
            created_events=created,
            skipped_events=skipped,
            elapsed_seconds=round(
                time.perf_counter() - started,
                6,
            ),
        )
