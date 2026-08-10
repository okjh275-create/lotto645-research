from __future__ import annotations

from collections.abc import Mapping

from lrp.evolution.contracts.review_reward_vector import (
    ReviewRewardVector,
)
from lrp.regimes.reward import RegimeReward


class RegimeRewardCalculator:
    """Map an existing review reward vector to one regime reward."""

    PORTFOLIO_WEIGHT = 0.45
    PRACTICAL_WEIGHT = 0.25
    RANK_QUALITY_WEIGHT = 0.20
    COVERAGE_WEIGHT = 0.10
    TARGET_SAMPLE_SIZE = 10

    def calculate(
        self,
        reward_vector: ReviewRewardVector,
        *,
        global_regime: Mapping[str, object],
    ) -> RegimeReward:
        if not isinstance(
            reward_vector,
            ReviewRewardVector,
        ):
            raise TypeError(
                "reward_vector must be a ReviewRewardVector"
            )

        if not isinstance(global_regime, Mapping):
            raise TypeError(
                "global_regime must be a mapping"
            )

        regime = global_regime.get("primary")
        confidence = global_regime.get("confidence")

        if not isinstance(regime, str):
            raise TypeError(
                "global_regime.primary must be a string"
            )

        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
        ):
            raise TypeError(
                "global_regime.confidence must be numeric"
            )

        raw_reward = (
            self.PORTFOLIO_WEIGHT
            * reward_vector.portfolio_hit
            + self.PRACTICAL_WEIGHT
            * reward_vector.practical_hit
            + self.RANK_QUALITY_WEIGHT
            * reward_vector.rank_quality
            + self.COVERAGE_WEIGHT
            * reward_vector.coverage
        )

        raw_reward = max(
            -1.0,
            min(1.0, raw_reward),
        )

        sample_weight = min(
            reward_vector.sample_size
            / self.TARGET_SAMPLE_SIZE,
            1.0,
        )

        return RegimeReward(
            regime=regime,
            reward=raw_reward,
            confidence=float(confidence),
            sample_weight=sample_weight,
        )
