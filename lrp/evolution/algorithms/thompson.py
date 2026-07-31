from __future__ import annotations

from random import Random
from typing import Iterable

from lrp.evolution.contracts.thompson import (
    BetaArmStatistics,
    ThompsonDecision,
)


class ThompsonBandit:
    """Seeded Thompson Sampling policy."""

    def __init__(
        self,
        *,
        seed: int | None = None,
    ) -> None:
        if seed is not None:
            if isinstance(seed, bool):
                raise TypeError(
                    "seed must be an integer or None"
                )

            if not isinstance(seed, int):
                raise TypeError(
                    "seed must be an integer or None"
                )

        self._seed = seed

    @property
    def seed(self) -> int | None:
        return self._seed

    def select(
        self,
        arms: Iterable[BetaArmStatistics],
    ) -> ThompsonDecision:
        normalized_arms = self._normalize_arms(arms)
        random = Random(self.seed)

        sampled = [
            (
                random.betavariate(
                    arm.alpha,
                    arm.beta,
                ),
                index,
                arm,
            )
            for index, arm in enumerate(normalized_arms)
        ]

        sample, _, selected = max(
            sampled,
            key=lambda item: (
                item[0],
                -item[1],
            ),
        )

        return ThompsonDecision(
            arm=selected.arm,
            sample=sample,
            seed=self.seed,
            observations=selected.observations,
        )

    @staticmethod
    def _normalize_arms(
        arms: Iterable[BetaArmStatistics],
    ) -> tuple[BetaArmStatistics, ...]:
        if isinstance(arms, (str, bytes)):
            raise TypeError(
                "arms must be an iterable of "
                "BetaArmStatistics"
            )

        try:
            normalized = tuple(arms)
        except TypeError as exc:
            raise TypeError(
                "arms must be an iterable of "
                "BetaArmStatistics"
            ) from exc

        if not normalized:
            raise ValueError(
                "arms must contain at least one arm"
            )

        seen: set[str] = set()

        for arm in normalized:
            if not isinstance(
                arm,
                BetaArmStatistics,
            ):
                raise TypeError(
                    "every arm must be a "
                    "BetaArmStatistics"
                )

            if arm.arm in seen:
                raise ValueError(
                    f"duplicate arm: {arm.arm}"
                )

            seen.add(arm.arm)

        return normalized
