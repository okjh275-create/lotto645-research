from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class AdaptivePolicyConfig:
    """Safety limits for applying an adaptive weight profile."""

    min_confidence: float = 0.60
    min_sample_size: int = 20
    max_component_delta: float = 0.08
    fail_open: bool = False

    def __post_init__(self) -> None:
        if not isfinite(self.min_confidence):
            raise ValueError("min_confidence must be finite")

        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(
                "min_confidence must be between 0.0 and 1.0"
            )

        if isinstance(self.min_sample_size, bool):
            raise TypeError(
                "min_sample_size must be an integer"
            )

        if not isinstance(self.min_sample_size, int):
            raise TypeError(
                "min_sample_size must be an integer"
            )

        if self.min_sample_size < 0:
            raise ValueError(
                "min_sample_size must be greater than "
                "or equal to 0"
            )

        if not isfinite(self.max_component_delta):
            raise ValueError(
                "max_component_delta must be finite"
            )

        if not 0.0 <= self.max_component_delta < 1.0:
            raise ValueError(
                "max_component_delta must be between "
                "0.0 inclusive and 1.0 exclusive"
            )

        if not isinstance(self.fail_open, bool):
            raise TypeError("fail_open must be a bool")