from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from lrp.evolution.contracts.models import (
    AdaptiveWeightProfile,
)


@dataclass(frozen=True, slots=True)
class EvolutionPipelineRequest:
    """Validated input passed to an evolution pipeline."""

    signals: Mapping[str, float]
    confidence: float
    sample_size: int
    revision: int
    generated_at: datetime
    previous_profile: AdaptiveWeightProfile | None = None

    def __post_init__(self) -> None:
        normalized_signals = self._normalize_signals(
            self.signals
        )

        object.__setattr__(
            self,
            "signals",
            MappingProxyType(normalized_signals),
        )

        self._validate_confidence(self.confidence)
        self._validate_sample_size(self.sample_size)
        self._validate_revision(self.revision)
        self._validate_generated_at(self.generated_at)

        if (
            self.previous_profile is not None
            and not isinstance(
                self.previous_profile,
                AdaptiveWeightProfile,
            )
        ):
            raise TypeError(
                "previous_profile must be an "
                "AdaptiveWeightProfile or None"
            )

    @staticmethod
    def _normalize_signals(
        signals: Mapping[str, float],
    ) -> dict[str, float]:
        if not isinstance(signals, Mapping):
            raise TypeError("signals must be a mapping")

        if not signals:
            raise ValueError(
                "signals must contain at least one item"
            )

        normalized: dict[str, float] = {}

        for name, value in signals.items():
            if not isinstance(name, str):
                raise TypeError(
                    "signal names must be strings"
                )

            normalized_name = name.strip()

            if not normalized_name:
                raise ValueError(
                    "signal names must not be empty"
                )

            if isinstance(value, bool):
                raise TypeError(
                    f"signal '{normalized_name}' "
                    "must be numeric"
                )

            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"signal '{normalized_name}' "
                    "must be numeric"
                )

            normalized_value = float(value)

            if not isfinite(normalized_value):
                raise ValueError(
                    f"signal '{normalized_name}' "
                    "must be finite"
                )

            normalized[normalized_name] = (
                normalized_value
            )

        return normalized

    @staticmethod
    def _validate_confidence(
        confidence: float,
    ) -> None:
        if isinstance(confidence, bool):
            raise TypeError(
                "confidence must be numeric"
            )

        if not isinstance(confidence, (int, float)):
            raise TypeError(
                "confidence must be numeric"
            )

        numeric_confidence = float(confidence)

        if not isfinite(numeric_confidence):
            raise ValueError(
                "confidence must be finite"
            )

        if not 0.0 <= numeric_confidence <= 1.0:
            raise ValueError(
                "confidence must be between "
                "0.0 and 1.0"
            )

    @staticmethod
    def _validate_sample_size(
        sample_size: int,
    ) -> None:
        if isinstance(sample_size, bool):
            raise TypeError(
                "sample_size must be an integer"
            )

        if not isinstance(sample_size, int):
            raise TypeError(
                "sample_size must be an integer"
            )

        if sample_size < 0:
            raise ValueError(
                "sample_size must be greater than "
                "or equal to 0"
            )

    @staticmethod
    def _validate_revision(
        revision: int,
    ) -> None:
        if isinstance(revision, bool):
            raise TypeError(
                "revision must be an integer"
            )

        if not isinstance(revision, int):
            raise TypeError(
                "revision must be an integer"
            )

        if revision < 1:
            raise ValueError(
                "revision must be greater than "
                "or equal to 1"
            )

    @staticmethod
    def _validate_generated_at(
        generated_at: datetime,
    ) -> None:
        if not isinstance(generated_at, datetime):
            raise TypeError(
                "generated_at must be a datetime"
            )

        if generated_at.tzinfo is None:
            raise ValueError(
                "generated_at must be timezone-aware"
            )

        if generated_at.utcoffset() is None:
            raise ValueError(
                "generated_at must have a valid "
                "UTC offset"
            )