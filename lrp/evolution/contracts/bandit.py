from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class ArmStatistics:
    """Accumulated observations for one bandit arm."""

    arm: str
    pulls: int = 0
    reward_sum: float = 0.0

    def __post_init__(self) -> None:
        normalized_arm = self._normalize_arm(
            self.arm
        )
        normalized_pulls = self._normalize_pulls(
            self.pulls
        )
        normalized_reward_sum = (
            self._normalize_reward_sum(
                self.reward_sum
            )
        )

        if (
            normalized_pulls == 0
            and normalized_reward_sum != 0.0
        ):
            raise ValueError(
                "reward_sum must be 0.0 when "
                "pulls is 0"
            )

        object.__setattr__(
            self,
            "arm",
            normalized_arm,
        )
        object.__setattr__(
            self,
            "pulls",
            normalized_pulls,
        )
        object.__setattr__(
            self,
            "reward_sum",
            normalized_reward_sum,
        )

    @property
    def mean_reward(self) -> float:
        if self.pulls == 0:
            return 0.0

        return self.reward_sum / self.pulls

    def record(
        self,
        reward: float,
    ) -> ArmStatistics:
        normalized_reward = (
            self._normalize_reward(
                reward
            )
        )

        return ArmStatistics(
            arm=self.arm,
            pulls=self.pulls + 1,
            reward_sum=(
                self.reward_sum
                + normalized_reward
            ),
        )

    @staticmethod
    def _normalize_arm(
        arm: str,
    ) -> str:
        if not isinstance(arm, str):
            raise TypeError(
                "arm must be a string"
            )

        normalized = arm.strip()

        if not normalized:
            raise ValueError(
                "arm must not be empty"
            )

        return normalized

    @staticmethod
    def _normalize_pulls(
        pulls: int,
    ) -> int:
        if isinstance(pulls, bool):
            raise TypeError(
                "pulls must be an integer"
            )

        if not isinstance(pulls, int):
            raise TypeError(
                "pulls must be an integer"
            )

        if pulls < 0:
            raise ValueError(
                "pulls must be greater than or "
                "equal to 0"
            )

        return pulls

    @staticmethod
    def _normalize_reward_sum(
        reward_sum: float,
    ) -> float:
        if isinstance(reward_sum, bool):
            raise TypeError(
                "reward_sum must be numeric"
            )

        if not isinstance(
            reward_sum,
            (int, float),
        ):
            raise TypeError(
                "reward_sum must be numeric"
            )

        value = float(reward_sum)

        if not isfinite(value):
            raise ValueError(
                "reward_sum must be finite"
            )

        return value

    @staticmethod
    def _normalize_reward(
        reward: float,
    ) -> float:
        if isinstance(reward, bool):
            raise TypeError(
                "reward must be numeric"
            )

        if not isinstance(
            reward,
            (int, float),
        ):
            raise TypeError(
                "reward must be numeric"
            )

        value = float(reward)

        if not isfinite(value):
            raise ValueError(
                "reward must be finite"
            )

        if not -1.0 <= value <= 1.0:
            raise ValueError(
                "reward must be between "
                "-1.0 and 1.0 inclusive"
            )

        return value


@dataclass(frozen=True, slots=True)
class BanditDecision:
    """Selection result returned by a bandit policy."""

    arm: str
    score: float
    reason: str
    total_pulls: int

    def __post_init__(self) -> None:
        normalized_arm = ArmStatistics._normalize_arm(
            self.arm
        )

        if isinstance(self.score, bool):
            raise TypeError(
                "score must be numeric"
            )

        if not isinstance(
            self.score,
            (int, float),
        ):
            raise TypeError(
                "score must be numeric"
            )

        normalized_score = float(self.score)

        if not isfinite(normalized_score):
            raise ValueError(
                "score must be finite"
            )

        if not isinstance(self.reason, str):
            raise TypeError(
                "reason must be a string"
            )

        normalized_reason = self.reason.strip()

        if not normalized_reason:
            raise ValueError(
                "reason must not be empty"
            )

        normalized_total_pulls = (
            ArmStatistics._normalize_pulls(
                self.total_pulls
            )
        )

        object.__setattr__(
            self,
            "arm",
            normalized_arm,
        )
        object.__setattr__(
            self,
            "score",
            normalized_score,
        )
        object.__setattr__(
            self,
            "reason",
            normalized_reason,
        )
        object.__setattr__(
            self,
            "total_pulls",
            normalized_total_pulls,
        )
