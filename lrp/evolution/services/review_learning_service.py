from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from lrp.evolution.contracts.learning_context import (
    LearningContext,
)
from lrp.evolution.contracts.review_learning import (
    ReviewLearningResult,
)
from lrp.evolution.integration.prediction_reward_mapper import (
    PredictionRewardMapper,
)
from lrp.evolution.services.persistent_learning_runner import (
    PersistentLearningRunner,
)


class ReviewLearningService:
    """Convert prediction reviews into persisted learning cycles."""

    def __init__(
        self,
        runner: PersistentLearningRunner,
        reward_mapper: PredictionRewardMapper | None = None,
    ) -> None:
        if not isinstance(
            runner,
            PersistentLearningRunner,
        ):
            raise TypeError(
                "runner must be a "
                "PersistentLearningRunner"
            )

        if (
            reward_mapper is not None
            and not isinstance(
                reward_mapper,
                PredictionRewardMapper,
            )
        ):
            raise TypeError(
                "reward_mapper must be a "
                "PredictionRewardMapper"
            )

        self._runner = runner
        self._reward_mapper = (
            reward_mapper
            if reward_mapper is not None
            else PredictionRewardMapper()
        )

    @property
    def runner(self) -> PersistentLearningRunner:
        return self._runner

    @property
    def reward_mapper(
        self,
    ) -> PredictionRewardMapper:
        return self._reward_mapper

    def learn(
        self,
        *,
        context: LearningContext,
        review_payload: Mapping[str, Any],
        snapshot_id: str,
        policy: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> ReviewLearningResult:
        if not isinstance(
            context,
            LearningContext,
        ):
            raise TypeError(
                "context must be a LearningContext"
            )

        if not isinstance(
            review_payload,
            Mapping,
        ):
            raise TypeError(
                "review_payload must be a mapping"
            )

        if not isinstance(overwrite, bool):
            raise TypeError(
                "overwrite must be a boolean"
            )

        feedbacks = self.reward_mapper.map(
            review_payload,
            policy=policy,
        )
        reward_vector = self.reward_mapper.vector(
            review_payload,
            policy=policy,
        )

        review_set_count = (
            self._review_set_count(
                review_payload
            )
        )

        snapshot_metadata = self._build_metadata(
            metadata=metadata,
            policy=policy,
            feedback_count=len(feedbacks),
            review_set_count=review_set_count,
        )
        snapshot_metadata.update(
            self._reward_vector_metadata(
                reward_vector
            )
        )

        previous_cumulative = (
            context.metadata.get(
                "cumulative_review_set_count",
                0,
            )
        )
        previous_review_count = (
            context.metadata.get(
                "review_count",
                0,
            )
        )

        if (
            isinstance(previous_cumulative, bool)
            or not isinstance(
                previous_cumulative,
                int,
            )
            or previous_cumulative < 0
        ):
            previous_cumulative = 0

        if (
            isinstance(previous_review_count, bool)
            or not isinstance(
                previous_review_count,
                int,
            )
            or previous_review_count < 0
        ):
            previous_review_count = 0

        reward_vector_metadata = {
            key: value
            for key, value
            in snapshot_metadata.items()
            if key.startswith(
                "reward_vector_"
            )
        }

        enriched_context = replace(
            context,
            metadata={
                **dict(context.metadata),
                "review_set_count": (
                    review_set_count
                ),
                "cumulative_review_set_count": (
                    previous_cumulative
                    + review_set_count
                ),
                "review_count": (
                    previous_review_count + 1
                ),
                **reward_vector_metadata,
            },
        )

        run_result = self.runner.run(
            context=enriched_context,
            feedbacks=feedbacks,
            snapshot_id=snapshot_id,
            metadata=snapshot_metadata,
            overwrite=overwrite,
        )

        return ReviewLearningResult(
            run_result=run_result,
            feedback_count=len(feedbacks),
            policy=policy,
        )

    @staticmethod
    def _review_set_count(
        review_payload: Mapping[str, Any],
    ) -> int:
        summary = review_payload.get(
            "summary",
            review_payload,
        )

        if not isinstance(summary, Mapping):
            raise TypeError(
                "review summary must be a mapping"
            )

        value = summary.get("set_count")

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                "set_count must be an integer"
            )

        if value < 1:
            raise ValueError(
                "set_count must be greater than "
                "or equal to 1"
            )

        return value

    @staticmethod
    def _reward_vector_metadata(
        reward_vector: object,
    ) -> dict[str, Any]:
        payload = reward_vector.as_dict()

        metadata = payload.get(
            "metadata",
            {},
        )

        result: dict[str, Any] = {
            "reward_vector_portfolio_hit": (
                payload["portfolio_hit"]
            ),
            "reward_vector_practical_hit": (
                payload["practical_hit"]
            ),
            "reward_vector_rank_quality": (
                payload["rank_quality"]
            ),
            "reward_vector_coverage": (
                payload["coverage"]
            ),
            "reward_vector_diversity": (
                payload["diversity"]
            ),
            "reward_vector_stability": (
                payload["stability"]
            ),
            "reward_vector_sample_size": (
                payload["sample_size"]
            ),
        }

        if isinstance(metadata, Mapping):
            source = metadata.get("source")
            policy = metadata.get("policy")
            round_no = metadata.get("round")

            if isinstance(source, str):
                result[
                    "reward_vector_source"
                ] = source

            if isinstance(policy, str):
                result[
                    "reward_vector_policy"
                ] = policy

            if (
                isinstance(round_no, int)
                and not isinstance(round_no, bool)
            ):
                result[
                    "reward_vector_round"
                ] = round_no

        return result

    @staticmethod
    def _build_metadata(
        *,
        metadata: Mapping[str, Any] | None,
        policy: str | None,
        feedback_count: int,
        review_set_count: int,
    ) -> dict[str, Any]:
        if metadata is None:
            normalized: dict[str, Any] = {}
        else:
            if not isinstance(metadata, Mapping):
                raise TypeError(
                    "metadata must be a mapping"
                )

            normalized = dict(metadata)

        normalized.update(
            {
                "learning_source": (
                    "prediction_review"
                ),
                "feedback_count": feedback_count,
                "review_set_count": (
                    review_set_count
                ),
            }
        )

        if policy is not None:
            if not isinstance(policy, str):
                raise TypeError(
                    "policy must be a string or None"
                )

            normalized_policy = policy.strip()

            if not normalized_policy:
                raise ValueError(
                    "policy must not be empty"
                )

            normalized["policy"] = normalized_policy

        return normalized
