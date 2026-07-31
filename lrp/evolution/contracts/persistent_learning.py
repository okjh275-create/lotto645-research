from __future__ import annotations

from dataclasses import dataclass

from lrp.evolution.contracts.learning_cycle import (
    LearningCycleResult,
)
from lrp.evolution.contracts.snapshot_schema import (
    LearningCycleSnapshot,
)


@dataclass(frozen=True, slots=True)
class PersistentLearningRunResult:
    """Combined result of a learning cycle and its snapshot."""

    learning_result: LearningCycleResult
    snapshot: LearningCycleSnapshot

    def __post_init__(self) -> None:
        if not isinstance(
            self.learning_result,
            LearningCycleResult,
        ):
            raise TypeError(
                "learning_result must be a "
                "LearningCycleResult"
            )

        if not isinstance(
            self.snapshot,
            LearningCycleSnapshot,
        ):
            raise TypeError(
                "snapshot must be a "
                "LearningCycleSnapshot"
            )

        if (
            self.snapshot.result
            != self.learning_result
        ):
            raise ValueError(
                "snapshot result must match "
                "learning_result"
            )

    @property
    def snapshot_id(self) -> str:
        return self.snapshot.snapshot_id

    @property
    def initial_context(self):
        return self.learning_result.initial_context

    @property
    def final_context(self):
        return self.learning_result.final_context

    @property
    def steps(self):
        return self.learning_result.steps

    @property
    def step_count(self) -> int:
        return self.learning_result.step_count

    @property
    def version_delta(self) -> int:
        return self.learning_result.version_delta
