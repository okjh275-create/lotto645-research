"""Round-completion orchestration pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lrp.evolution.contracts.learning_context import LearningContext
from lrp.evolution.services.review_profile_evolution_service import (
    ReviewProfileEvolutionService,
)
from lrp.operations import review_prediction
from lrp.outcomes import (
    OutcomeBridge,
    OutcomeLearningBridge,
)


def _load_prediction(
    source: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)

    path = Path(source)

    if not path.is_file():
        raise FileNotFoundError(path)

    payload = json.loads(
        path.read_text(encoding="utf-8-sig")
    )

    if not isinstance(payload, dict):
        raise TypeError(
            "prediction artifact must be a JSON object"
        )

    return payload


@dataclass(frozen=True, slots=True)
class RoundCompletionResult:
    """Complete result of one round-closing workflow."""

    round_no: int
    review: Mapping[str, Any]
    outcome: Mapping[str, Any]
    learning_snapshot_id: str
    feedback_count: int
    final_context_version: int
    profile_applied: bool
    profile_revision: int | None
    profile_snapshot_saved: bool
    profile_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_no": self.round_no,
            "review": dict(self.review),
            "outcome": dict(self.outcome),
            "learning": {
                "snapshot_id": self.learning_snapshot_id,
                "feedback_count": self.feedback_count,
                "final_context_version": (
                    self.final_context_version
                ),
            },
            "profile": {
                "applied": self.profile_applied,
                "revision": self.profile_revision,
                "snapshot_saved": (
                    self.profile_snapshot_saved
                ),
                "reasons": list(self.profile_reasons),
            },
        }


class RoundCompletionPipeline:
    """Close one prediction round through review and learning."""

    def __init__(
        self,
        *,
        outcome_bridge: OutcomeBridge,
        learning_bridge: OutcomeLearningBridge,
        profile_service: ReviewProfileEvolutionService,
    ) -> None:
        if not isinstance(outcome_bridge, OutcomeBridge):
            raise TypeError(
                "outcome_bridge must be an OutcomeBridge"
            )

        if not isinstance(
            learning_bridge,
            OutcomeLearningBridge,
        ):
            raise TypeError(
                "learning_bridge must be an "
                "OutcomeLearningBridge"
            )

        if not isinstance(
            profile_service,
            ReviewProfileEvolutionService,
        ):
            raise TypeError(
                "profile_service must be a "
                "ReviewProfileEvolutionService"
            )

        self._outcome_bridge = outcome_bridge
        self._learning_bridge = learning_bridge
        self._profile_service = profile_service

    @property
    def outcome_bridge(self) -> OutcomeBridge:
        return self._outcome_bridge

    @property
    def learning_bridge(self) -> OutcomeLearningBridge:
        return self._learning_bridge

    @property
    def profile_service(
        self,
    ) -> ReviewProfileEvolutionService:
        return self._profile_service

    def run(
        self,
        prediction: str | Path | Mapping[str, Any],
        *,
        winning_numbers: tuple[int, ...],
        bonus: int,
        snapshot_id: str | None = None,
        policy: str | None = None,
        confidence: float = 0.80,
        recorded_at_kst: str,
        reviewed_at_kst: str | None = None,
        overwrite_learning: bool = False,
        generated_at_utc: datetime | None = None,
    ) -> RoundCompletionResult:
        prediction_payload = _load_prediction(
            prediction
        )

        review_payload = review_prediction(
            prediction_payload,
            winning_numbers=winning_numbers,
            bonus=bonus,
        )

        round_no = int(review_payload["round"])
        effective_snapshot_id = (
            snapshot_id
            or f"review-{round_no}"
        )

        effective_reviewed_at = (
            reviewed_at_kst
            or str(review_payload["reviewed_at_kst"])
        )

        outcome = self.outcome_bridge.process(
            prediction_payload,
            winning_numbers=winning_numbers,
            bonus=bonus,
            recorded_at_kst=recorded_at_kst,
            reviewed_at_kst=effective_reviewed_at,
        )

        context = LearningContext(
            cycle_id=f"round-completion-{round_no}",
            round_no=round_no,
        )

        feature_attribution_available = (
            "probability_vector"
            in prediction_payload
        )

        learning = self.learning_bridge.learn(
            context=context,
            review_payload=review_payload,
            prediction_payload=(
                prediction_payload
                if feature_attribution_available
                else None
            ),
            winning_numbers=(
                winning_numbers
                if feature_attribution_available
                else None
            ),
            snapshot_id=effective_snapshot_id,
            policy=policy,
            metadata={
                "round": round_no,
                "source": "round_completion",
                "feature_attribution_available": (
                    feature_attribution_available
                ),
            },
            overwrite=overwrite_learning,
        )

        profile_result = self.profile_service.evolve(
            context=learning.learning.final_context,
            generated_at=(
                generated_at_utc
                or datetime.now(timezone.utc)
            ),
            confidence=confidence,
        )

        profile = profile_result.decision.profile

        return RoundCompletionResult(
            round_no=round_no,
            review=review_payload,
            outcome=outcome.as_dict(),
            learning_snapshot_id=(
                learning.snapshot_id
            ),
            feedback_count=(
                learning.feedback_count
            ),
            final_context_version=(
                learning.learning.final_context.version
            ),
            profile_applied=(
                profile_result.decision.applied
            ),
            profile_revision=(
                profile.revision
                if profile is not None
                else None
            ),
            profile_snapshot_saved=(
                profile_result.snapshot is not None
            ),
            profile_reasons=tuple(
                profile_result.decision.reasons
            ),
        )
