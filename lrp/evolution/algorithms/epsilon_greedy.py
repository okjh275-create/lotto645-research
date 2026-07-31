from __future__ import annotations

from math import isfinite
from random import Random
from typing import Iterable

from lrp.evolution.contracts.bandit import (
    ArmStatistics,
    BanditDecision,
)


class EpsilonGreedyBandit:
    """Seeded epsilon-greedy bandit policy."""

    def __init__(
        self,
        *,
        epsilon: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self._epsilon = self._validate_epsilon(
            epsilon
        )
        self._seed = self._validate_seed(
            seed
        )
        self._random = Random(self.seed)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def seed(self) -> int | None:
        return self._seed

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

        if self._should_explore():
            selected = self._random.choice(
                normalized_arms
            )

            return BanditDecision(
                arm=selected.arm,
                score=selected.mean_reward,
                reason="random_exploration",
                total_pulls=total_pulls,
            )

        selected = max(
            enumerate(normalized_arms),
            key=lambda item: (
                item[1].mean_reward,
                -item[0],
            ),
        )[1]

        return BanditDecision(
            arm=selected.arm,
            score=selected.mean_reward,
            reason="greedy_exploitation",
            total_pulls=total_pulls,
        )

    def reset(
        self,
    ) -> None:
        """Reset the random sequence to the configured seed."""

        self._random = Random(self.seed)

    def _should_explore(
        self,
    ) -> bool:
        return self._random.random() < self.epsilon

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
    def _validate_epsilon(
        epsilon: float,
    ) -> float:
        if isinstance(epsilon, bool):
            raise TypeError(
                "epsilon must be numeric"
            )

        if not isinstance(
            epsilon,
            (int, float),
        ):
            raise TypeError(
                "epsilon must be numeric"
            )

        value = float(epsilon)

        if not isfinite(value):
            raise ValueError(
                "epsilon must be finite"
            )

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "epsilon must be between "
                "0.0 and 1.0 inclusive"
            )

        return value

    @staticmethod
    def _validate_seed(
        seed: int | None,
    ) -> int | None:
        if seed is None:
            return None

        if isinstance(seed, bool):
            raise TypeError(
                "seed must be an integer or None"
            )

        if not isinstance(seed, int):
            raise TypeError(
                "seed must be an integer or None"
            )

        return seed
