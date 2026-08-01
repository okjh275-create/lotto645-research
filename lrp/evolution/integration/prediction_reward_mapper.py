from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lrp.evolution.contracts.reinforcement import (
    RewardFeedback,
)


class PredictionRewardMapper:
    """Map prediction-review summaries to reinforcement feedback."""

    HIT_REWARDS = {
        0: -1.00,
        1: -0.70,
        2: -0.30,
        3: 0.20,
        4: 0.55,
        5: 0.85,
        6: 1.00,
    }

    def map(
        self,
        review_payload: Mapping[str, Any],
        *,
        policy: str | None = None,
    ) -> tuple[RewardFeedback, ...]:
        if not isinstance(
            review_payload,
            Mapping,
        ):
            raise TypeError(
                "review_payload must be a mapping"
            )

        summary = self._summary(
            review_payload
        )
        normalized_policy = (
            self._normalize_policy(policy)
        )

        best_hits = self._hit_count(
            summary,
            "best_main_hits",
        )
        practical_hits = self._hit_count(
            summary,
            "practical_best_hits",
        )
        set_count = self._positive_count(
            summary,
            "set_count",
        )

        return (
            RewardFeedback(
                source="prediction_review",
                policy=normalized_policy,
                arm="portfolio_top_k",
                reward=self.reward_for_hits(
                    best_hits
                ),
                observation_count=set_count,
            ),
            RewardFeedback(
                source="prediction_review",
                policy=normalized_policy,
                arm="practical_top5",
                reward=self.reward_for_hits(
                    practical_hits
                ),
                observation_count=1,
            ),
        )

    @classmethod
    def reward_for_hits(
        cls,
        main_hits: int,
    ) -> float:
        if isinstance(main_hits, bool):
            raise TypeError(
                "main_hits must be an integer"
            )

        if not isinstance(main_hits, int):
            raise TypeError(
                "main_hits must be an integer"
            )

        if main_hits not in cls.HIT_REWARDS:
            raise ValueError(
                "main_hits must be between 0 and 6"
            )

        return cls.HIT_REWARDS[main_hits]

    @staticmethod
    def _summary(
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if "summary" in payload:
            summary = payload["summary"]

            if not isinstance(
                summary,
                Mapping,
            ):
                raise TypeError(
                    "review summary must be a mapping"
                )

            return summary

        return payload

    @staticmethod
    def _hit_count(
        summary: Mapping[str, Any],
        field_name: str,
    ) -> int:
        if field_name not in summary:
            raise ValueError(
                f"missing review field: {field_name}"
            )

        value = summary[field_name]

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                f"{field_name} must be an integer"
            )

        if not 0 <= value <= 6:
            raise ValueError(
                f"{field_name} must be between 0 and 6"
            )

        return value

    @staticmethod
    def _positive_count(
        summary: Mapping[str, Any],
        field_name: str,
    ) -> int:
        if field_name not in summary:
            raise ValueError(
                f"missing review field: {field_name}"
            )

        value = summary[field_name]

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                f"{field_name} must be an integer"
            )

        if value < 1:
            raise ValueError(
                f"{field_name} must be greater than "
                "or equal to 1"
            )

        return value

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
