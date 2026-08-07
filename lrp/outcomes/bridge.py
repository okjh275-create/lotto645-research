"""Bridge prediction artifacts and draw results into learning reviews."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from lrp.learning import LearningRepository, LearningService, ResultRecord
from .importer import OutcomeImporter


@dataclass(frozen=True, slots=True)
class OutcomeBridgeResult:
    round_no: int
    imported_predictions: int
    created_predictions: int
    existing_predictions: int
    result_created: bool
    reviews_scanned: int
    reviews_created: int
    reviews_skipped: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_no": self.round_no,
            "imported_predictions": self.imported_predictions,
            "created_predictions": self.created_predictions,
            "existing_predictions": self.existing_predictions,
            "result_created": self.result_created,
            "reviews_scanned": self.reviews_scanned,
            "reviews_created": self.reviews_created,
            "reviews_skipped": self.reviews_skipped,
        }


class OutcomeBridge:
    def __init__(self, *, repository: LearningRepository, model_name: str) -> None:
        if not isinstance(repository, LearningRepository):
            raise TypeError("repository must be a LearningRepository")
        self._repository = repository
        self._importer = OutcomeImporter(model_name=model_name)
        self._service = LearningService(repository)

    @property
    def repository(self) -> LearningRepository:
        return self._repository

    @property
    def importer(self) -> OutcomeImporter:
        return self._importer

    @property
    def service(self) -> LearningService:
        return self._service

    def process(
        self,
        prediction_payload: Mapping[str, Any],
        *,
        winning_numbers: tuple[int, ...],
        bonus: int,
        recorded_at_kst: str,
        reviewed_at_kst: str | None = None,
    ) -> OutcomeBridgeResult:
        records = self.importer.import_predictions(prediction_payload)
        if not records:
            raise ValueError("prediction artifact produced no records")

        round_numbers = {record.round_no for record in records}
        if len(round_numbers) != 1:
            raise ValueError("all predictions must have the same round")

        round_no = next(iter(round_numbers))
        created_predictions = 0
        for record in records:
            if self.repository.add_prediction(record):
                created_predictions += 1

        result = ResultRecord(
            round_no=round_no,
            numbers=winning_numbers,
            bonus=bonus,
            recorded_at_kst=recorded_at_kst,
        )
        result_created = self.repository.add_result(result)
        review_result = self.service.run_incremental_review(
            reviewed_at_kst=reviewed_at_kst
        )
        return OutcomeBridgeResult(
            round_no=round_no,
            imported_predictions=len(records),
            created_predictions=created_predictions,
            existing_predictions=len(records) - created_predictions,
            result_created=result_created,
            reviews_scanned=review_result.scanned,
            reviews_created=review_result.created,
            reviews_skipped=review_result.skipped,
        )
