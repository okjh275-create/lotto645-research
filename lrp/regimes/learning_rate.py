from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class AdaptiveLearningRatePolicy:
    """Decay regime learning as calibration evidence matures."""

    base_rate: float = 0.10
    min_rate: float = 0.02
    sample_scale: int = 20

    def __post_init__(self) -> None:
        for field_name in (
            "base_rate",
            "min_rate",
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

            if not 0.0 < normalized <= 1.0:
                raise ValueError(
                    f"{field_name} must be greater than 0 "
                    "and less than or equal to 1"
                )

            object.__setattr__(
                self,
                field_name,
                normalized,
            )

        if self.min_rate > self.base_rate:
            raise ValueError(
                "min_rate must not exceed base_rate"
            )

        if (
            isinstance(self.sample_scale, bool)
            or not isinstance(self.sample_scale, int)
        ):
            raise TypeError(
                "sample_scale must be an integer"
            )

        if self.sample_scale < 1:
            raise ValueError(
                "sample_scale must be greater than or equal to 1"
            )

    def rate(
        self,
        *,
        revision: int,
        sample_size: int,
    ) -> float:
        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
        ):
            raise TypeError(
                "revision must be an integer"
            )

        if revision < 0:
            raise ValueError(
                "revision must be greater than or equal to 0"
            )

        if (
            isinstance(sample_size, bool)
            or not isinstance(sample_size, int)
        ):
            raise TypeError(
                "sample_size must be an integer"
            )

        if sample_size < 0:
            raise ValueError(
                "sample_size must be greater than or equal to 0"
            )

        revision_factor = (
            1.0
            / math.sqrt(
                max(1, revision)
            )
        )

        sample_factor = (
            1.0
            / math.sqrt(
                1.0
                + (
                    sample_size
                    / self.sample_scale
                )
            )
        )

        maturity_factor = min(
            revision_factor,
            sample_factor,
        )

        rate = (
            self.base_rate
            * maturity_factor
        )

        return max(
            self.min_rate,
            min(self.base_rate, rate),
        )
