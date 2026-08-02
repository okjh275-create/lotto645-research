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

        portfolio_reward = self._find_reward(
            context.rewards,
            suffix=self.PORTFOLIO_SUFFIX,
        )
        practical_reward = self._find_reward(
            context.rewards,
            suffix=self.PRACTICAL_SUFFIX,
        )

        return {
            "hot": 0.0,
            "cold": 0.0,
            "gap": 0.0,
            "trend": 0.0,
            "transition": 0.0,
            "learning": portfolio_reward,
            "adaptive": practical_reward,
        }

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
