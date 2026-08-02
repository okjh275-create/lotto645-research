"""Structured reward vector derived from prediction reviews."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Mapping

from lrp.contracts import ContractError


_REWARD_FIELDS = (
    "portfolio_hit",
    "practical_hit",
    "rank_quality",
    "coverage",
    "diversity",
    "stability",
)


def _reward(
    value: object,
    *,
    field_name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ContractError(
            f"{field_name} must be numeric"
        )

    normalized = float(value)

    if not isfinite(normalized):
        raise ContractError(
            f"{field_name} must be finite"
        )

    if not -1.0 <= normalized <= 1.0:
        raise ContractError(
            f"{field_name} must be between -1 and 1"
        )

    return normalized


@dataclass(frozen=True, slots=True)
class ReviewRewardVector:
    """Normalized reward components for one reviewed prediction."""

    portfolio_hit: float
    practical_hit: float
    rank_quality: float
    coverage: float
    diversity: float
    stability: float
    sample_size: int
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        for field_name in _REWARD_FIELDS:
            object.__setattr__(
                self,
                field_name,
                _reward(
                    getattr(self, field_name),
                    field_name=field_name,
                ),
            )

        if (
            isinstance(self.sample_size, bool)
            or not isinstance(self.sample_size, int)
            or self.sample_size < 1
        ):
            raise ContractError(
                "sample_size must be a positive integer"
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

    @classmethod
    def neutral(
        cls,
        *,
        sample_size: int = 1,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ReviewRewardVector":
        return cls(
            portfolio_hit=0.0,
            practical_hit=0.0,
            rank_quality=0.0,
            coverage=0.0,
            diversity=0.0,
            stability=0.0,
            sample_size=sample_size,
            metadata=(
                {}
                if metadata is None
                else metadata
            ),
        )

    def weighted_score(
        self,
        weights: Mapping[str, float] | None = None,
    ) -> float:
        normalized_weights = self._weights(weights)

        return sum(
            getattr(self, field_name)
            * normalized_weights[field_name]
            for field_name in _REWARD_FIELDS
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def _weights(
        weights: Mapping[str, float] | None,
    ) -> dict[str, float]:
        if weights is None:
            return {
                "portfolio_hit": 0.30,
                "practical_hit": 0.25,
                "rank_quality": 0.15,
                "coverage": 0.10,
                "diversity": 0.10,
                "stability": 0.10,
            }

        if not isinstance(weights, Mapping):
            raise ContractError(
                "weights must be a mapping or None"
            )

        missing = tuple(
            field_name
            for field_name in _REWARD_FIELDS
            if field_name not in weights
        )
        extra = tuple(
            key
            for key in weights
            if key not in _REWARD_FIELDS
        )

        if missing or extra:
            raise ContractError(
                "reward weights must contain exactly: "
                + ", ".join(_REWARD_FIELDS)
            )

        normalized: dict[str, float] = {}

        for field_name in _REWARD_FIELDS:
            value = weights[field_name]

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ContractError(
                    f"{field_name} weight must be "
                    "finite and non-negative"
                )

            normalized[field_name] = float(value)

        total = sum(normalized.values())

        if total <= 0.0:
            raise ContractError(
                "reward weight total must be positive"
            )

        return {
            field_name: value / total
            for field_name, value
            in normalized.items()
        }
