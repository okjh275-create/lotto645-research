"""Aggregate read-only round-completion operational statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .round_completion_repository import (
    RoundCompletionRepository,
)


@dataclass(frozen=True, slots=True)
class RoundCompletionSummary:
    completion_count: int
    latest_round: int | None
    average_feedback_count: float
    profile_apply_rate: float
    manifest_pass_rate: float
    latest_snapshot_id: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "completion_count": (
                self.completion_count
            ),
            "latest_round": self.latest_round,
            "average_feedback_count": (
                self.average_feedback_count
            ),
            "profile_apply_rate": (
                self.profile_apply_rate
            ),
            "manifest_pass_rate": (
                self.manifest_pass_rate
            ),
            "latest_snapshot_id": (
                self.latest_snapshot_id
            ),
        }


def summarize_round_completions(
    repository: RoundCompletionRepository,
    *,
    limit: int = 20,
) -> RoundCompletionSummary:
    if not isinstance(
        repository,
        RoundCompletionRepository,
    ):
        raise TypeError(
            "repository must be a "
            "RoundCompletionRepository"
        )

    records = repository.recent(limit)

    if not records:
        return RoundCompletionSummary(
            completion_count=0,
            latest_round=None,
            average_feedback_count=0.0,
            profile_apply_rate=0.0,
            manifest_pass_rate=0.0,
            latest_snapshot_id=None,
        )

    feedback_counts: list[int] = []
    profile_applied = 0
    manifest_passed = 0

    for record in records:
        learning = record.get("learning", {})
        profile = record.get("profile", {})

        feedback = (
            learning.get("feedback_count", 0)
            if isinstance(learning, dict)
            else 0
        )

        if (
            isinstance(feedback, int)
            and not isinstance(feedback, bool)
            and feedback >= 0
        ):
            feedback_counts.append(feedback)
        else:
            feedback_counts.append(0)

        if (
            isinstance(profile, dict)
            and profile.get("applied") is True
        ):
            profile_applied += 1

        round_no = record.get("round_no")

        if isinstance(round_no, int):
            verification = (
                repository.verify_round(
                    round_no
                )
            )

            if verification.get("status") == "PASS":
                manifest_passed += 1

    latest = records[0]
    latest_learning = latest.get(
        "learning",
        {},
    )

    latest_snapshot_id = (
        latest_learning.get("snapshot_id")
        if isinstance(latest_learning, dict)
        else None
    )

    count = len(records)

    return RoundCompletionSummary(
        completion_count=count,
        latest_round=int(latest["round_no"]),
        average_feedback_count=round(
            sum(feedback_counts) / count,
            6,
        ),
        profile_apply_rate=round(
            profile_applied / count,
            6,
        ),
        manifest_pass_rate=round(
            manifest_passed / count,
            6,
        ),
        latest_snapshot_id=(
            str(latest_snapshot_id)
            if latest_snapshot_id is not None
            else None
        ),
    )
