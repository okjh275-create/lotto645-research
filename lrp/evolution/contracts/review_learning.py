from __future__ import annotations

from dataclasses import dataclass

from lrp.evolution.contracts.persistent_learning import (
    PersistentLearningRunResult,
)


@dataclass(frozen=True, slots=True)
class ReviewLearningResult:
    """Result of converting one review into persisted learning."""

    run_result: PersistentLearningRunResult
    feedback_count: int
    policy: str | None

    def __post_init__(self) -> None:
        if not isinstance(
            self.run_result,
            PersistentLearningRunResult,
        ):
            raise TypeError(
                "run_result must be a "
                "PersistentLearningRunResult"
            )

        if (
            isinstance(self.feedback_count, bool)
            or not isinstance(
                self.feedback_count,
                int,
            )
        ):
            raise TypeError(
                "feedback_count must be an integer"
            )

        if self.feedback_count < 1:
            raise ValueError(
                "feedback_count must be greater than "
                "or equal to 1"
            )

        normalized_policy = self._normalize_policy(
            self.policy
        )

        object.__setattr__(
            self,
            "policy",
            normalized_policy,
        )

    @property
    def snapshot_id(self) -> str:
        return self.run_result.snapshot_id

    @property
    def learning_result(self):
        return self.run_result.learning_result

    @property
    def snapshot(self):
        return self.run_result.snapshot

    @property
    def final_context(self):
        return self.run_result.final_context

    @property
    def step_count(self) -> int:
        return self.run_result.step_count

    @staticmethod
    def _normalize_policy(
        policy: str | None,
    ) -> str | None:
        if policy is None:
            return None

        if not isinstance(policy, str):
            raise TypeError(
                "policy must be a string or None"
            )

        normalized = policy.strip()

        if not normalized:
            raise ValueError(
                "policy must not be empty"
            )

        return normalized
