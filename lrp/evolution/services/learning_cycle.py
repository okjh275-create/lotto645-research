from __future__ import annotations

from collections.abc import Iterable

from lrp.evolution.algorithms.reinforcement import (
    ReinforcementUpdater,
)
from lrp.evolution.contracts.learning_context import (
    LearningContext,
)
from lrp.evolution.contracts.learning_cycle import (
    LearningCycleResult,
    LearningCycleStep,
)
from lrp.evolution.contracts.reinforcement import (
    RewardFeedback,
)


class LearningCycle:
    """Coordinate immutable reinforcement updates."""

    def __init__(
        self,
        updater: ReinforcementUpdater | None = None,
    ) -> None:
        if (
            updater is not None
            and not isinstance(
                updater,
                ReinforcementUpdater,
            )
        ):
            raise TypeError(
                "updater must be a "
                "ReinforcementUpdater"
            )

        self._updater = (
            updater
            if updater is not None
            else ReinforcementUpdater()
        )

    def run(
        self,
        *,
        context: LearningContext,
        feedbacks: Iterable[RewardFeedback],
    ) -> LearningCycleResult:
        if not isinstance(
            context,
            LearningContext,
        ):
            raise TypeError(
                "context must be a LearningContext"
            )

        normalized_feedbacks = (
            self._normalize_feedbacks(
                feedbacks
            )
        )

        current = context
        steps: list[LearningCycleStep] = []

        for index, feedback in enumerate(
            normalized_feedbacks,
            start=1,
        ):
            version_before = current.version

            current = self._updater.apply(
                context=current,
                feedback=feedback,
            )

            steps.append(
                LearningCycleStep(
                    index=index,
                    name="reinforcement_feedback",
                    version_before=version_before,
                    version_after=current.version,
                    reward_key=self._reward_key(
                        feedback
                    ),
                )
            )

        return LearningCycleResult(
            initial_context=context,
            final_context=current,
            steps=tuple(steps),
            metadata={
                "feedback_count": len(
                    normalized_feedbacks
                ),
                "cycle_completed": True,
            },
        )

    @staticmethod
    def _normalize_feedbacks(
        feedbacks: Iterable[RewardFeedback],
    ) -> tuple[RewardFeedback, ...]:
        if isinstance(
            feedbacks,
            (str, bytes),
        ):
            raise TypeError(
                "feedbacks must be an iterable of "
                "RewardFeedback values"
            )

        try:
            normalized = tuple(feedbacks)
        except TypeError as exc:
            raise TypeError(
                "feedbacks must be iterable"
            ) from exc

        for feedback in normalized:
            if not isinstance(
                feedback,
                RewardFeedback,
            ):
                raise TypeError(
                    "feedbacks must contain only "
                    "RewardFeedback values"
                )

        return normalized

    @staticmethod
    def _reward_key(
        feedback: RewardFeedback,
    ) -> str:
        if feedback.policy is None:
            return (
                f"{feedback.source}:"
                f"{feedback.arm}"
            )

        return (
            f"{feedback.source}:"
            f"{feedback.policy}:"
            f"{feedback.arm}"
        )
