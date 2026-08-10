from __future__ import annotations

from dataclasses import dataclass
import math

from lrp.contracts import ContractError
from lrp.regimes.contracts import (
    SUPPORTED_REGIMES,
)


@dataclass(frozen=True, slots=True)
class RegimeReward:
    """Validated reward signal for one global regime."""

    regime: str
    reward: float
    confidence: float
    sample_weight: float

    def __post_init__(self) -> None:
        if self.regime not in SUPPORTED_REGIMES:
            raise ContractError(
                f"unsupported regime: {self.regime}"
            )

        for field_name in (
            "reward",
            "confidence",
            "sample_weight",
        ):
            value = getattr(self, field_name)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise TypeError(
                    f"{field_name} must be numeric"
                )

            normalized = float(value)

            if not math.isfinite(normalized):
                raise ValueError(
                    f"{field_name} must be finite"
                )

            object.__setattr__(
                self,
                field_name,
                normalized,
            )

        if not -1.0 <= self.reward <= 1.0:
            raise ValueError(
                "reward must be between -1.0 and 1.0"
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0.0 and 1.0"
            )

        if not 0.0 <= self.sample_weight <= 1.0:
            raise ValueError(
                "sample_weight must be between 0.0 and 1.0"
            )

    @property
    def effective_reward(self) -> float:
        return (
            self.reward
            * self.confidence
            * self.sample_weight
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "regime": self.regime,
            "reward": self.reward,
            "confidence": self.confidence,
            "sample_weight": self.sample_weight,
            "effective_reward": self.effective_reward,
        }
