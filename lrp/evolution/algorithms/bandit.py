from __future__ import annotations

from math import isfinite, log, sqrt
from typing import Iterable

from lrp.evolution.contracts.bandit import (
    ArmStatistics,
    BanditDecision,
)


class UCB1Bandit:
    """Deterministic Upper Confidence Bound policy."""

    def __init__(
        self,
        *,
        exploration_strength: float = sqrt(2.0),
    ) -> None:
        self._exploration_strength = (
            self._validate_exploration_strength(
                exploration_strength
            )
        )

    @property
    def exploration_strength(self) -> float:
        return self._exploration_strength

    def select(
        self,
        arms: Iterable[ArmStatistics],
    ) -> BanditDecision:
        normalized_arms = self._normalize_arms(
            arms
        )
        total_pulls = sum(
            arm.pulls
            for arm in normalized_arms
        )

        untried = [
            arm
            for arm in normalized_arms
            if arm.pulls == 0
        ]

        if untried:
            selected = min(
                untried,
                key=lambda arm: arm.arm,
            )

            return BanditDecision(
                arm=selected.arm,
                score=0.0,
                reason="untried_arm",
                total_pulls=total_pulls,
            )

        scored = [
            (
                self.score(
                    arm=arm,
                    total_pulls=total_pulls,
                ),
                arm,
            )
            for arm in normalized_arms
        ]

        selected_score, selected_arm = max(
            scored,
            key=lambda item: (
                item[0],
                -normalized_arms.index(item[1]),
            ),
        )

        return BanditDecision(
            arm=selected_arm.arm,
            score=selected_score,
            reason="highest_ucb_score",
            total_pulls=total_pulls,
        )

    def score(
        self,
        *,
        arm: ArmStatistics,
        total_pulls: int,
    ) -> float:
        if not isinstance(
            arm,
            ArmStatistics,
        ):
            raise TypeError(
                "arm must be an ArmStatistics"
            )

        if isinstance(total_pulls, bool):
            raise TypeError(
                "total_pulls must be an integer"
            )

        if not isinstance(total_pulls, int):
            raise TypeError(
                "total_pulls must be an integer"
            )

        if total_pulls < 1:
            raise ValueError(
                "total_pulls must be greater "
                "than or equal to 1"
            )

        if arm.pulls < 1:
            raise ValueError(
                "cannot calculate UCB score "
                "for an untried arm"
            )

        if total_pulls < arm.pulls:
            raise ValueError(
                "total_pulls must not be less "
                "than arm pulls"
            )

        exploration_bonus = (
            self.exploration_strength
            * sqrt(
                log(total_pulls)
                / arm.pulls
            )
        )

        return (
            arm.mean_reward
            + exploration_bonus
        )

    @staticmethod
    def _normalize_arms(
        arms: Iterable[ArmStatistics],
    ) -> tuple[ArmStatistics, ...]:
        if isinstance(
            arms,
            (str, bytes),
        ):
            raise TypeError(
                "arms must be an iterable of "
                "ArmStatistics"
            )

        try:
            normalized = tuple(arms)
        except TypeError as exc:
            raise TypeError(
                "arms must be an iterable of "
                "ArmStatistics"
            ) from exc

        if not normalized:
            raise ValueError(
                "arms must contain at least one arm"
            )

        seen: set[str] = set()

        for arm in normalized:
            if not isinstance(
                arm,
                ArmStatistics,
            ):
                raise TypeError(
                    "every arm must be an "
                    "ArmStatistics"
                )

            if arm.arm in seen:
                raise ValueError(
                    f"duplicate arm: {arm.arm}"
                )

            seen.add(arm.arm)

        return normalized

    @staticmethod
    def _validate_exploration_strength(
        exploration_strength: float,
    ) -> float:
        if isinstance(
            exploration_strength,
            bool,
        ):
            raise TypeError(
                "exploration_strength must "
                "be numeric"
            )

        if not isinstance(
            exploration_strength,
            (int, float),
        ):
            raise TypeError(
                "exploration_strength must "
                "be numeric"
            )

        value = float(
            exploration_strength
        )

        if not isfinite(value):
            raise ValueError(
                "exploration_strength must "
                "be finite"
            )

        if value < 0.0:
            raise ValueError(
                "exploration_strength must be "
                "greater than or equal to 0"
            )

        return value
