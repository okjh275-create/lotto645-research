from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lrp.evolution.contracts.reinforcement import (
    RewardFeedback,
)
from lrp.evolution.contracts.review_reward_vector import (
    ReviewRewardVector,
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

    def vector(
        self,
        review_payload: Mapping[str, Any],
        *,
        policy: str | None = None,
    ) -> ReviewRewardVector:
        """Build a structured reward vector from one review."""

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

        rank_quality = self._rank_quality(
            summary,
            set_count=set_count,
        )
        coverage = self._coverage(
            summary,
            set_count=set_count,
        )

        metadata: dict[str, Any] = {
            "source": "prediction_review",
            "best_main_hits": best_hits,
            "practical_best_hits": practical_hits,
        }

        if normalized_policy is not None:
            metadata["policy"] = normalized_policy

        round_no = review_payload.get("round")

        if (
            isinstance(round_no, int)
            and not isinstance(round_no, bool)
            and round_no > 0
        ):
            metadata["round"] = round_no

        return ReviewRewardVector(
            portfolio_hit=self.reward_for_hits(
                best_hits
            ),
            practical_hit=self.reward_for_hits(
                practical_hits
            ),
            rank_quality=rank_quality,
            coverage=coverage,
            diversity=0.0,
            stability=0.0,
            sample_size=set_count,
            metadata=metadata,
        )

    @classmethod
    def _rank_quality(
        cls,
        summary: Mapping[str, Any],
        *,
        set_count: int,
    ) -> float:
        raw_counts = summary.get(
            "winning_rank_counts"
        )

        if raw_counts is None:
            return 0.0

        if not isinstance(raw_counts, Mapping):
            raise TypeError(
                "winning_rank_counts must be a mapping"
            )

        rank_weights = {
            "1": 1.0,
            "2": 0.8,
            "3": 0.6,
            "4": 0.4,
            "5": 0.2,
        }

        weighted_total = 0.0
        ranked_count = 0

        for rank, weight in rank_weights.items():
            count = cls._non_negative_count(
                raw_counts,
                rank,
            )
            weighted_total += count * weight
            ranked_count += count

        if ranked_count > set_count:
            raise ValueError(
                "winning rank count exceeds set_count"
            )

        normalized = weighted_total / set_count

        return (2.0 * normalized) - 1.0

    @classmethod
    def _coverage(
        cls,
        summary: Mapping[str, Any],
        *,
        set_count: int,
    ) -> float:
        raw_distribution = summary.get(
            "hit_distribution"
        )

        if raw_distribution is None:
            return 0.0

        if not isinstance(
            raw_distribution,
            Mapping,
        ):
            raise TypeError(
                "hit_distribution must be a mapping"
            )

        weighted_hits = 0
        observed_count = 0

        for hits in range(7):
            count = cls._non_negative_count(
                raw_distribution,
                str(hits),
                default=0,
            )
            weighted_hits += hits * count
            observed_count += count

        if observed_count != set_count:
            raise ValueError(
                "hit_distribution total must equal "
                "set_count"
            )

        mean_hits = weighted_hits / set_count

        return (mean_hits / 3.0) - 1.0

    @staticmethod
    def _non_negative_count(
        values: Mapping[str, Any],
        field_name: str,
        *,
        default: int | None = None,
    ) -> int:
        if field_name not in values:
            if default is not None:
                return default

            raise ValueError(
                f"missing count field: {field_name}"
            )

        value = values[field_name]

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                f"{field_name} count must be an integer"
            )

        if value < 0:
            raise ValueError(
                f"{field_name} count must be non-negative"
            )

        return value

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
