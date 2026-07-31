from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class RewardFeedback:
    """Validated feedback applied to one learning context."""

    source: str
    arm: str
    reward: float
    policy: str | None = None
    observation_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source",
            self._normalize_required_text(
                self.source,
                field_name="source",
            ),
        )
        object.__setattr__(
            self,
            "arm",
            self._normalize_required_text(
                self.arm,
                field_name="arm",
            ),
        )
        object.__setattr__(
            self,
            "reward",
            self._normalize_reward(
                self.reward,
            ),
        )
        object.__setattr__(
            self,
            "policy",
            self._normalize_optional_text(
                self.policy,
                field_name="policy",
            ),
        )
        object.__setattr__(
            self,
            "observation_count",
            self._normalize_positive_integer(
                self.observation_count,
                field_name="observation_count",
            ),
        )

    @staticmethod
    def _normalize_required_text(
        value: str,
        *,
        field_name: str,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} must be a string"
            )

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                f"{field_name} must not be empty"
            )

        return normalized

    @classmethod
    def _normalize_optional_text(
        cls,
        value: str | None,
        *,
        field_name: str,
    ) -> str | None:
        if value is None:
            return None

        return cls._normalize_required_text(
            value,
            field_name=field_name,
        )

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

    @staticmethod
    def _normalize_positive_integer(
        value: int,
        *,
        field_name: str,
    ) -> int:
        if isinstance(value, bool):
            raise TypeError(
                f"{field_name} must be an integer"
            )

        if not isinstance(value, int):
            raise TypeError(
                f"{field_name} must be an integer"
            )

        if value < 1:
            raise ValueError(
                f"{field_name} must be greater than "
                "or equal to 1"
            )

        return value
