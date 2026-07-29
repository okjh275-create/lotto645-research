"""Incremental learning and automatic review service."""

from __future__ import annotations

from dataclasses import dataclass
import time

from .evaluator import evaluate_prediction
from .repository import LearningRepository


@dataclass(frozen=True, slots=True)
class IncrementalReviewSummary:
    """Summary of one incremental review execution."""

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
    ) -> None:
        self.repository = repository

    def run_incremental_review(
        self,
        *,
        round_no: int | None = None,
        limit: int | None = None,
        reviewed_at_kst: str | None = None,
    ) -> IncrementalReviewSummary:
        """Review only predictions having results and no review."""

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
