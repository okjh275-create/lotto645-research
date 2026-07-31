from __future__ import annotations

from lrp.evolution.contracts.learning_context import (
    LearningContext,
)
from lrp.evolution.contracts.reinforcement import (
    RewardFeedback,
)


class ReinforcementUpdater:
    """Apply immutable reward feedback to a learning context."""

    def apply(
        self,
        *,
        context: LearningContext,
        feedback: RewardFeedback,
    ) -> LearningContext:
        if not isinstance(
            context,
            LearningContext,
        ):
            raise TypeError(
                "context must be a LearningContext"
            )

        if not isinstance(
            feedback,
            RewardFeedback,
        ):
            raise TypeError(
                "feedback must be a RewardFeedback"
            )

        reward_key = self._reward_key(
            feedback
        )

        rewards = dict(context.rewards)
        rewards[reward_key] = feedback.reward

        metadata = dict(context.metadata)
        metadata.update(
            {
                "feedback_source": feedback.source,
                "feedback_arm": feedback.arm,
                "feedback_observation_count": (
                    feedback.observation_count
                ),
            }
        )

        if feedback.policy is not None:
            metadata["feedback_policy"] = (
                feedback.policy
            )

        updated = context.with_rewards(
            rewards
        ).with_metadata(
            metadata
        )

        if feedback.policy is not None:
            updated = updated.with_selection(
                policy=feedback.policy,
                arm=feedback.arm,
            )

        return updated.advance_version()

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
