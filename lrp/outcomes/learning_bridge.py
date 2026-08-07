"""Bridge reviewed outcomes into the existing evolution learning service."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lrp.evolution.contracts.learning_context import LearningContext
from lrp.evolution.contracts.review_learning import ReviewLearningResult
from lrp.evolution.services.review_learning_service import ReviewLearningService


@dataclass(frozen=True, slots=True)
class OutcomeLearningBridgeResult:
    """Result of one outcome-to-learning bridge execution."""

    round_no: int
    snapshot_id: str
    feedback_count: int
    policy: str | None
    learning: ReviewLearningResult

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_no": self.round_no,
            "snapshot_id": self.snapshot_id,
            "feedback_count": self.feedback_count,
            "policy": self.policy,
        }


class OutcomeLearningBridge:
    """Delegate reviewed outcome learning to ReviewLearningService."""

    def __init__(
        self,
        *,
        service: ReviewLearningService,
    ) -> None:
        if not isinstance(service, ReviewLearningService):
            raise TypeError(
                "service must be a ReviewLearningService"
            )

        self._service = service

    @property
    def service(self) -> ReviewLearningService:
        return self._service

    def learn(
        self,
        *,
        context: LearningContext,
        review_payload: Mapping[str, Any],
        snapshot_id: str,
        prediction_payload: Mapping[str, Any] | None = None,
        winning_numbers: tuple[int, ...] | None = None,
        policy: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> OutcomeLearningBridgeResult:
        if not isinstance(context, LearningContext):
            raise TypeError(
                "context must be a LearningContext"
            )

        if not isinstance(review_payload, Mapping):
            raise TypeError(
                "review_payload must be a mapping"
            )

        result = self.service.learn(
            context=context,
            review_payload=review_payload,
            prediction_payload=prediction_payload,
            winning_numbers=winning_numbers,
            snapshot_id=snapshot_id,
            policy=policy,
            metadata=metadata,
            overwrite=overwrite,
        )

        return OutcomeLearningBridgeResult(
            round_no=context.round_no,
            snapshot_id=snapshot_id,
            feedback_count=result.feedback_count,
            policy=result.policy,
            learning=result,
        )
