from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class RegimeCalibration:
    """Adaptive calibration factors for global regime boosts."""

    gap_recovery: float = 1.0
    cluster_rotation: float = 1.0
    high_band_expansion: float = 1.0
    low_band_expansion: float = 1.0

    def __post_init__(self) -> None:
        for field_name in (
            "gap_recovery",
            "cluster_rotation",
            "high_band_expansion",
            "low_band_expansion",
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

            if not 0.50 <= normalized <= 1.50:
                raise ValueError(
                    f"{field_name} must be between 0.50 and 1.50"
                )

            object.__setattr__(
                self,
                field_name,
                normalized,
            )

    @classmethod
    def neutral(cls) -> "RegimeCalibration":
        return cls()

    def get(self, regime: str) -> float:
        if regime == "gap_recovery":
            return self.gap_recovery
        if regime == "cluster_rotation":
            return self.cluster_rotation
        if regime == "high_band_expansion":
            return self.high_band_expansion
        if regime == "low_band_expansion":
            return self.low_band_expansion

        return 1.0

    def as_dict(self) -> dict[str, float]:
        return {
            "gap_recovery": self.gap_recovery,
            "cluster_rotation": self.cluster_rotation,
            "high_band_expansion": self.high_band_expansion,
            "low_band_expansion": self.low_band_expansion,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> "RegimeCalibration":
        return cls(
            gap_recovery=payload.get(
                "gap_recovery",
                1.0,
            ),
            cluster_rotation=payload.get(
                "cluster_rotation",
                1.0,
            ),
            high_band_expansion=payload.get(
                "high_band_expansion",
                1.0,
            ),
            low_band_expansion=payload.get(
                "low_band_expansion",
                1.0,
            ),
        )
