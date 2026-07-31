"""Prediction accuracy models for Project F."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from lrp.contracts import ContractError


def _unit_score(
    value: object,
    *,
    field_name: str,
) -> float:
    """Validate and normalize a finite score in the range 0..1."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ContractError(
            f"{field_name} must be numeric"
        )

    result = float(value)

    if not 0.0 <= result <= 1.0:
        raise ContractError(
            f"{field_name} must be between 0 and 1"
        )

    return result


@dataclass(frozen=True, slots=True)
class NumberRegime:
    """Normalized regime signals for one Lotto 6/45 number."""

    number: int
    hot_score: float
    cold_score: float
    gap_score: float
    trend_score: float
    transition_score: float
    confidence: float
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            isinstance(self.number, bool)
            or not isinstance(self.number, int)
            or not 1 <= self.number <= 45
        ):
            raise ContractError(
                "number must be an integer between 1 and 45"
            )

        for field_name in (
            "hot_score",
            "cold_score",
            "gap_score",
            "trend_score",
            "transition_score",
            "confidence",
        ):
            normalized = _unit_score(
                getattr(self, field_name),
                field_name=field_name,
            )
            object.__setattr__(
                self,
                field_name,
                normalized,
            )

        if not isinstance(self.metadata, Mapping):
            raise ContractError(
                "metadata must be a mapping"
            )

        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )

    @property
    def dominant_regime(self) -> str:
        """Return the strongest primary regime label."""

        scores = {
            "hot": self.hot_score,
            "cold": self.cold_score,
            "gap": self.gap_score,
            "transition": self.transition_score,
        }

        return max(
            scores,
            key=lambda name: (
                scores[name],
                name,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "hot_score": self.hot_score,
            "cold_score": self.cold_score,
            "gap_score": self.gap_score,
            "trend_score": self.trend_score,
            "transition_score": self.transition_score,
            "confidence": self.confidence,
            "dominant_regime": self.dominant_regime,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RegimeProfile:
    """Immutable collection of number-level regime signals."""

    round_no: int | None
    generated_at_kst: str | None
    regimes: tuple[NumberRegime, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.round_no is not None and (
            isinstance(self.round_no, bool)
            or not isinstance(self.round_no, int)
            or self.round_no <= 0
        ):
            raise ContractError(
                "round_no must be a positive integer or None"
            )

        if (
            self.generated_at_kst is not None
            and (
                not isinstance(
                    self.generated_at_kst,
                    str,
                )
                or not self.generated_at_kst.strip()
            )
        ):
            raise ContractError(
                "generated_at_kst must be a non-empty string or None"
            )

        regimes = tuple(self.regimes)

        if len(regimes) != 45:
            raise ContractError(
                "regimes must contain exactly 45 number records"
            )

        if not all(
            isinstance(item, NumberRegime)
            for item in regimes
        ):
            raise ContractError(
                "regimes must contain NumberRegime values"
            )

        numbers = tuple(
            item.number
            for item in regimes
        )

        if numbers != tuple(range(1, 46)):
            raise ContractError(
                "regimes must be ordered from number 1 through 45"
            )

        if not isinstance(self.metadata, Mapping):
            raise ContractError(
                "metadata must be a mapping"
            )

        object.__setattr__(
            self,
            "regimes",
            regimes,
        )
        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )

    def get(
        self,
        number: int,
    ) -> NumberRegime:
        if (
            isinstance(number, bool)
            or not isinstance(number, int)
            or not 1 <= number <= 45
        ):
            raise ContractError(
                "number must be an integer between 1 and 45"
            )

        return self.regimes[number - 1]

    def top(
        self,
        *,
        metric: str,
        limit: int = 10,
    ) -> tuple[NumberRegime, ...]:
        allowed = {
            "hot_score",
            "cold_score",
            "gap_score",
            "trend_score",
            "transition_score",
            "confidence",
        }

        if metric not in allowed:
            raise ContractError(
                "unsupported regime metric: "
                f"{metric!r}"
            )

        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit <= 0
        ):
            raise ContractError(
                "limit must be a positive integer"
            )

        return tuple(
            sorted(
                self.regimes,
                key=lambda item: (
                    -getattr(item, metric),
                    item.number,
                ),
            )[:limit]
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_no": self.round_no,
            "generated_at_kst": (
                self.generated_at_kst
            ),
            "regime_count": len(self.regimes),
            "regimes": [
                item.as_dict()
                for item in self.regimes
            ],
            "metadata": dict(self.metadata),
        }


def ordered_regimes(
    values: Iterable[NumberRegime],
) -> tuple[NumberRegime, ...]:
    """Return number regimes in canonical number order."""

    regimes = tuple(values)

    if not all(
        isinstance(item, NumberRegime)
        for item in regimes
    ):
        raise ContractError(
            "values must contain NumberRegime records"
        )

    return tuple(
        sorted(
            regimes,
            key=lambda item: item.number,
        )
    )
