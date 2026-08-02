from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from lrp.evolution.contracts.learning_context import (
    LearningContext,
)


class ReviewSignalExtractor:
    """Extract conservative adaptive signals from review rewards."""

    COMPONENTS = (
        "hot",
        "cold",
        "gap",
        "trend",
        "transition",
        "learning",
        "adaptive",
    )

    PORTFOLIO_SUFFIX = "portfolio_top_k"
    PRACTICAL_SUFFIX = "practical_top5"

    def extract(
        self,
        context: LearningContext,
    ) -> dict[str, float]:
        if not isinstance(
            context,
            LearningContext,
        ):
            raise TypeError(
                "context must be a LearningContext"
            )

        legacy_portfolio_reward = self._find_reward(
            context.rewards,
            suffix=self.PORTFOLIO_SUFFIX,
        )
        legacy_practical_reward = self._find_reward(
            context.rewards,
            suffix=self.PRACTICAL_SUFFIX,
        )

        structured_keys = (
            "reward_vector_portfolio_hit",
            "reward_vector_practical_hit",
            "reward_vector_rank_quality",
            "reward_vector_coverage",
            "reward_vector_diversity",
            "reward_vector_stability",
        )
        has_structured_vector = any(
            key in context.metadata
            for key in structured_keys
        )

        if has_structured_vector:
            portfolio_reward = self._metadata_reward(
                context,
                key="reward_vector_portfolio_hit",
                fallback=legacy_portfolio_reward,
            )
            practical_reward = self._metadata_reward(
                context,
                key="reward_vector_practical_hit",
                fallback=legacy_practical_reward,
            )
            rank_quality = self._metadata_reward(
                context,
                key="reward_vector_rank_quality",
                fallback=0.0,
            )
            coverage = self._metadata_reward(
                context,
                key="reward_vector_coverage",
                fallback=0.0,
            )
            diversity = self._metadata_reward(
                context,
                key="reward_vector_diversity",
                fallback=0.0,
            )
            stability = self._metadata_reward(
                context,
                key="reward_vector_stability",
                fallback=0.0,
            )

            learning_signal = self._bounded_mean(
                portfolio_reward,
                rank_quality,
                coverage,
            )
            adaptive_signal = self._bounded_mean(
                practical_reward,
                diversity,
                stability,
            )
        else:
            learning_signal = legacy_portfolio_reward
            adaptive_signal = legacy_practical_reward

        return {
            "hot": 0.0,
            "cold": 0.0,
            "gap": 0.0,
            "trend": 0.0,
            "transition": 0.0,
            "learning": learning_signal,
            "adaptive": adaptive_signal,
        }

    @staticmethod
    def _metadata_reward(
        context: LearningContext,
        *,
        key: str,
        fallback: float,
    ) -> float:
        if key not in context.metadata:
            return fallback

        value = context.metadata[key]

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise TypeError(
                f"{key} must be numeric"
            )

        normalized = float(value)

        if not isfinite(normalized):
            raise ValueError(
                f"{key} must be finite"
            )

        if not -1.0 <= normalized <= 1.0:
            raise ValueError(
                f"{key} must be between "
                "-1.0 and 1.0"
            )

        return normalized

    @staticmethod
    def _bounded_mean(
        *values: float,
    ) -> float:
        if not values:
            return 0.0

        result = sum(values) / len(values)

        return max(
            -1.0,
            min(1.0, result),
        )

    @staticmethod
    def sample_size(
        context: LearningContext,
    ) -> int:
        if not isinstance(
            context,
            LearningContext,
        ):
            raise TypeError(
                "context must be a LearningContext"
            )

        value = context.metadata.get(
            "cumulative_review_set_count",
            context.metadata.get(
                "review_set_count",
                context.metadata.get(
                    "feedback_observation_count",
                    1,
                ),
            ),
        )

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            return 1

        return value

    @staticmethod
    def _find_reward(
        rewards: Mapping[str, Any],
        *,
        suffix: str,
    ) -> float:
        matches = [
            value
            for key, value in rewards.items()
            if key.endswith(f":{suffix}")
        ]

        if not matches:
            return 0.0

        value = matches[-1]

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise TypeError(
                f"reward for {suffix} must be numeric"
            )

        normalized = float(value)

        if not isfinite(normalized):
            raise ValueError(
                f"reward for {suffix} must be finite"
            )

        if not -1.0 <= normalized <= 1.0:
            raise ValueError(
                f"reward for {suffix} must be "
                "between -1.0 and 1.0"
            )

        return normalized
