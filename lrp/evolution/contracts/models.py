from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isclose, isfinite
from typing import Any, ClassVar, Mapping


@dataclass(frozen=True, slots=True)
class AdaptiveWeightProfile:
    """Immutable adaptive weights used by probability fusion."""

    hot_weight: float
    cold_weight: float
    gap_weight: float
    trend_weight: float
    transition_weight: float
    learning_weight: float
    adaptive_weight: float

    confidence: float
    sample_size: int
    revision: int
    generated_at: datetime

    WEIGHT_FIELDS: ClassVar[tuple[str, ...]] = (
        "hot_weight",
        "cold_weight",
        "gap_weight",
        "trend_weight",
        "transition_weight",
        "learning_weight",
        "adaptive_weight",
    )

    DEFAULT_TOLERANCE: ClassVar[float] = 1e-9

    def __post_init__(self) -> None:
        self._validate_weights()
        self._validate_metadata()

    @classmethod
    def default(
        cls,
        *,
        revision: int = 1,
        generated_at: datetime | None = None,
    ) -> AdaptiveWeightProfile:
        """Return the Project F baseline probability weights."""

        return cls(
            hot_weight=0.35,
            cold_weight=0.15,
            gap_weight=0.15,
            trend_weight=0.15,
            transition_weight=0.10,
            learning_weight=0.05,
            adaptive_weight=0.05,
            confidence=0.0,
            sample_size=0,
            revision=revision,
            generated_at=generated_at or datetime.now(timezone.utc),
        )

    @property
    def weights(self) -> dict[str, float]:
        return {
            field_name: float(getattr(self, field_name))
            for field_name in self.WEIGHT_FIELDS
        }

    @property
    def total_weight(self) -> float:
        return sum(self.weights.values())

    def to_probability_weights(self) -> dict[str, float]:
        """Return keys compatible with probability-fusion components."""

        return {
            "hot": self.hot_weight,
            "cold": self.cold_weight,
            "gap": self.gap_weight,
            "trend": self.trend_weight,
            "transition": self.transition_weight,
            "learning": self.learning_weight,
            "adaptive": self.adaptive_weight,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "weights": self.weights,
            "confidence": self.confidence,
            "sample_size": self.sample_size,
            "revision": self.revision,
            "generated_at": self.generated_at.isoformat(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> AdaptiveWeightProfile:
        """Restore a profile from its serialized representation."""

        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")

        raw_weights = payload.get("weights")

        if not isinstance(raw_weights, Mapping):
            raise ValueError("payload.weights must be a mapping")

        missing_fields = [
            field_name
            for field_name in cls.WEIGHT_FIELDS
            if field_name not in raw_weights
        ]

        if missing_fields:
            missing = ", ".join(missing_fields)
            raise ValueError(
                f"missing adaptive weight fields: {missing}"
            )

        generated_at_raw = payload.get("generated_at")

        if not isinstance(generated_at_raw, str):
            raise ValueError(
                "generated_at must be an ISO-8601 string"
            )

        try:
            generated_at = datetime.fromisoformat(
                generated_at_raw
            )
        except ValueError as exc:
            raise ValueError(
                "generated_at must be a valid ISO-8601 string"
            ) from exc

        required_metadata = (
            "confidence",
            "sample_size",
            "revision",
        )

        missing_metadata = [
            field_name
            for field_name in required_metadata
            if field_name not in payload
        ]

        if missing_metadata:
            missing = ", ".join(missing_metadata)
            raise ValueError(
                f"missing adaptive metadata fields: {missing}"
            )

        return cls(
            hot_weight=float(raw_weights["hot_weight"]),
            cold_weight=float(raw_weights["cold_weight"]),
            gap_weight=float(raw_weights["gap_weight"]),
            trend_weight=float(raw_weights["trend_weight"]),
            transition_weight=float(
                raw_weights["transition_weight"]
            ),
            learning_weight=float(
                raw_weights["learning_weight"]
            ),
            adaptive_weight=float(
                raw_weights["adaptive_weight"]
            ),
            confidence=float(payload["confidence"]),
            sample_size=int(payload["sample_size"]),
            revision=int(payload["revision"]),
            generated_at=generated_at,
        )

    def _validate_weights(self) -> None:
        for field_name, value in self.weights.items():
            if not isfinite(value):
                raise ValueError(
                    f"{field_name} must be finite"
                )

            if value < 0.0:
                raise ValueError(
                    f"{field_name} must be greater than or equal to 0"
                )

        if not isclose(
            self.total_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=self.DEFAULT_TOLERANCE,
        ):
            raise ValueError(
                "adaptive weights must sum to 1.0; "
                f"received {self.total_weight:.12f}"
            )

    def _validate_metadata(self) -> None:
        if not isfinite(self.confidence):
            raise ValueError("confidence must be finite")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0.0 and 1.0"
            )

        if isinstance(self.sample_size, bool):
            raise TypeError("sample_size must be an integer")

        if not isinstance(self.sample_size, int):
            raise TypeError("sample_size must be an integer")

        if self.sample_size < 0:
            raise ValueError(
                "sample_size must be greater than or equal to 0"
            )

        if isinstance(self.revision, bool):
            raise TypeError("revision must be an integer")

        if not isinstance(self.revision, int):
            raise TypeError("revision must be an integer")

        if self.revision < 1:
            raise ValueError(
                "revision must be greater than or equal to 1"
            )

        if not isinstance(self.generated_at, datetime):
            raise TypeError("generated_at must be a datetime")

        if self.generated_at.tzinfo is None:
            raise ValueError(
                "generated_at must be timezone-aware"
            )

        if self.generated_at.utcoffset() is None:
            raise ValueError(
                "generated_at must have a valid UTC offset"
            )