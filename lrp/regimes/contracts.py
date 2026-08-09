"""Contracts for prediction-regime analysis."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

from lrp.contracts import ContractError


SUPPORTED_REGIMES = (
    "neutral",
    "mixed",
    "gap_recovery",
    "cluster_rotation",
    "high_band_expansion",
    "low_band_expansion",
)


def _unit_float(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ContractError(f"{field} must be numeric")

    result = float(value)

    if not math.isfinite(result):
        raise ContractError(f"{field} must be finite")

    if not 0.0 <= result <= 1.0:
        raise ContractError(
            f"{field} must be between 0 and 1"
        )

    return result


@dataclass(frozen=True, slots=True)
class RegimeFeatureSnapshot:
    """Global features aggregated from number-level signals."""

    average_recency: float
    average_frequency: float
    average_gap_reversion: float
    pair_density: float
    frequency_dispersion: float
    recency_variance: float
    pair_variance: float
    low_band_ratio: float
    high_band_ratio: float

    def __post_init__(self) -> None:
        for field in (
            "average_recency",
            "average_frequency",
            "average_gap_reversion",
            "pair_density",
            "frequency_dispersion",
            "recency_variance",
            "pair_variance",
            "low_band_ratio",
            "high_band_ratio",
        ):
            object.__setattr__(
                self,
                field,
                _unit_float(
                    getattr(self, field),
                    field=field,
                ),
            )

    def as_dict(self) -> dict[str, float]:
        return {
            "average_recency": self.average_recency,
            "average_frequency": self.average_frequency,
            "average_gap_reversion": (
                self.average_gap_reversion
            ),
            "pair_density": self.pair_density,
            "frequency_dispersion": (
                self.frequency_dispersion
            ),
            "recency_variance": self.recency_variance,
            "pair_variance": self.pair_variance,
            "low_band_ratio": self.low_band_ratio,
            "high_band_ratio": self.high_band_ratio,
        }


@dataclass(frozen=True, slots=True)
class RegimeDecision:
    """Validated output of a regime detector."""

    primary: str
    confidence: float
    features: RegimeFeatureSnapshot
    scores: Mapping[str, float]
    secondary: str | None = None
    secondary_confidence: float | None = None

    def __post_init__(self) -> None:
        if self.primary not in SUPPORTED_REGIMES:
            raise ContractError(
                f"unsupported primary regime: {self.primary}"
            )

        confidence = _unit_float(
            self.confidence,
            field="confidence",
        )

        if not isinstance(
            self.features,
            RegimeFeatureSnapshot,
        ):
            raise ContractError(
                "features must be a RegimeFeatureSnapshot"
            )

        if not isinstance(self.scores, Mapping):
            raise ContractError("scores must be a mapping")

        normalized_scores: dict[str, float] = {}

        for name, score in self.scores.items():
            if name not in SUPPORTED_REGIMES:
                raise ContractError(
                    f"unsupported score regime: {name}"
                )

            normalized_scores[name] = _unit_float(
                score,
                field=f"scores[{name}]",
            )

        if self.primary not in normalized_scores:
            raise ContractError(
                "scores must contain the primary regime"
            )

        secondary = self.secondary
        secondary_confidence = (
            self.secondary_confidence
        )

        if secondary is None:
            if secondary_confidence is not None:
                raise ContractError(
                    "secondary_confidence requires secondary"
                )
        else:
            if secondary not in SUPPORTED_REGIMES:
                raise ContractError(
                    f"unsupported secondary regime: {secondary}"
                )

            if secondary == self.primary:
                raise ContractError(
                    "secondary regime must differ from primary"
                )

            if secondary not in normalized_scores:
                raise ContractError(
                    "scores must contain the secondary regime"
                )

            if secondary_confidence is None:
                raise ContractError(
                    "secondary regime requires confidence"
                )

            secondary_confidence = _unit_float(
                secondary_confidence,
                field="secondary_confidence",
            )

        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(
            self,
            "secondary_confidence",
            secondary_confidence,
        )
        object.__setattr__(
            self,
            "scores",
            MappingProxyType(normalized_scores),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "primary": self.primary,
            "confidence": self.confidence,
            "secondary": self.secondary,
            "secondary_confidence": (
                self.secondary_confidence
            ),
            "scores": dict(self.scores),
            "features": self.features.as_dict(),
        }
