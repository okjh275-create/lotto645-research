"""Probability fusion models for Project F-002."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

from lrp.contracts import ContractError


def _probability(
    value: object,
    *,
    field_name: str,
) -> float:
    """Validate a finite probability in the inclusive 0..1 range."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ContractError(
            f"{field_name} must be numeric"
        )

    result = float(value)

    if not math.isfinite(result):
        raise ContractError(
            f"{field_name} must be finite"
        )

    if not 0.0 <= result <= 1.0:
        raise ContractError(
            f"{field_name} must be between 0 and 1"
        )

    return result


@dataclass(frozen=True, slots=True)
class NumberProbability:
    """Final probability score for one Lotto number."""

    number: int
    probability: float
    raw_score: float
    rank: int
    components: Mapping[str, float]
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

        probability = _probability(
            self.probability,
            field_name="probability",
        )
        object.__setattr__(
            self,
            "probability",
            probability,
        )

        if (
            isinstance(self.raw_score, bool)
            or not isinstance(self.raw_score, (int, float))
            or not math.isfinite(float(self.raw_score))
            or float(self.raw_score) < 0.0
        ):
            raise ContractError(
                "raw_score must be a finite non-negative number"
            )

        object.__setattr__(
            self,
            "raw_score",
            float(self.raw_score),
        )

        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or not 1 <= self.rank <= 45
        ):
            raise ContractError(
                "rank must be an integer between 1 and 45"
            )

        if not isinstance(self.components, Mapping):
            raise ContractError(
                "components must be a mapping"
            )

        normalized_components = {
            str(name): _probability(
                value,
                field_name=f"components[{name!r}]",
            )
            for name, value in self.components.items()
        }

        if not isinstance(self.metadata, Mapping):
            raise ContractError(
                "metadata must be a mapping"
            )

        object.__setattr__(
            self,
            "components",
            normalized_components,
        )
        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "probability": self.probability,
            "raw_score": self.raw_score,
            "rank": self.rank,
            "components": dict(self.components),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ProbabilityVector:
    """Normalized probability distribution across numbers 1 through 45."""

    round_no: int | None
    generated_at_kst: str | None
    probabilities: tuple[NumberProbability, ...]
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
                not isinstance(self.generated_at_kst, str)
                or not self.generated_at_kst.strip()
            )
        ):
            raise ContractError(
                "generated_at_kst must be a non-empty string or None"
            )

        probabilities = tuple(self.probabilities)

        if len(probabilities) != 45:
            raise ContractError(
                "probabilities must contain exactly 45 records"
            )

        if not all(
            isinstance(item, NumberProbability)
            for item in probabilities
        ):
            raise ContractError(
                "probabilities must contain NumberProbability values"
            )

        numbers = tuple(
            item.number
            for item in probabilities
        )

        if numbers != tuple(range(1, 46)):
            raise ContractError(
                "probabilities must be ordered from number 1 through 45"
            )

        ranks = tuple(
            item.rank
            for item in probabilities
        )

        if set(ranks) != set(range(1, 46)):
            raise ContractError(
                "probability ranks must uniquely cover 1 through 45"
            )

        total_probability = sum(
            item.probability
            for item in probabilities
        )

        if not math.isclose(
            total_probability,
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ContractError(
                "probabilities must sum to 1.0"
            )

        if not isinstance(self.metadata, Mapping):
            raise ContractError(
                "metadata must be a mapping"
            )

        object.__setattr__(
            self,
            "probabilities",
            probabilities,
        )
        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )

    def get(
        self,
        number: int,
    ) -> NumberProbability:
        if (
            isinstance(number, bool)
            or not isinstance(number, int)
            or not 1 <= number <= 45
        ):
            raise ContractError(
                "number must be an integer between 1 and 45"
            )

        return self.probabilities[number - 1]

    def top(
        self,
        limit: int = 10,
    ) -> tuple[NumberProbability, ...]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 45
        ):
            raise ContractError(
                "limit must be an integer between 1 and 45"
            )

        return tuple(
            sorted(
                self.probabilities,
                key=lambda item: item.rank,
            )[:limit]
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_no": self.round_no,
            "generated_at_kst": self.generated_at_kst,
            "probability_count": len(self.probabilities),
            "probability_sum": sum(
                item.probability
                for item in self.probabilities
            ),
            "probabilities": [
                item.as_dict()
                for item in self.probabilities
            ],
            "metadata": dict(self.metadata),
        }


def ordered_probabilities(
    values: Iterable[NumberProbability],
) -> tuple[NumberProbability, ...]:
    """Return probability records in canonical number order."""

    probabilities = tuple(values)

    if not all(
        isinstance(item, NumberProbability)
        for item in probabilities
    ):
        raise ContractError(
            "values must contain NumberProbability records"
        )

    return tuple(
        sorted(
            probabilities,
            key=lambda item: item.number,
        )
    )

from dataclasses import dataclass

from .models import RegimeProfile


@dataclass(frozen=True, slots=True)
class ProbabilityFusionConfig:
    """Weight configuration for probability fusion."""

    hot_weight: float = 0.35
    cold_weight: float = 0.15
    gap_weight: float = 0.15
    trend_weight: float = 0.15
    transition_weight: float = 0.10
    learning_weight: float = 0.05
    adaptive_weight: float = 0.05


class ProbabilityFusionEngine:
    """Fuse regime scores into a probability vector."""

    def __init__(
        self,
        config: ProbabilityFusionConfig | None = None,
    ) -> None:
        self._config = (
            config
            if config is not None
            else ProbabilityFusionConfig()
        )

    @property
    def config(self) -> ProbabilityFusionConfig:
        return self._config

    def _validate_weights(self) -> dict[str, float]:
        weights = {
            "hot": self._config.hot_weight,
            "cold": self._config.cold_weight,
            "gap": self._config.gap_weight,
            "trend": self._config.trend_weight,
            "transition": self._config.transition_weight,
            "learning": self._config.learning_weight,
            "adaptive": self._config.adaptive_weight,
        }

        normalized: dict[str, float] = {}

        for name, value in weights.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ContractError(
                    f"{name}_weight must be a finite "
                    "non-negative number"
                )

            normalized[name] = float(value)

        total = sum(normalized.values())

        if not math.isclose(
            total,
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ContractError(
                "probability fusion weights must sum to 1.0"
            )

        return normalized

    @staticmethod
    def _external_score(
        values: Mapping[int, float] | None,
        *,
        number: int,
        field_name: str,
    ) -> float:
        if values is None:
            return 0.5

        if not isinstance(values, Mapping):
            raise ContractError(
                f"{field_name} must be a mapping or None"
            )

        value = values.get(number, 0.5)

        return _probability(
            value,
            field_name=f"{field_name}[{number}]",
        )

    def build(
        self,
        profile: RegimeProfile,
        *,
        learning_scores: Mapping[int, float] | None = None,
        adaptive_scores: Mapping[int, float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ProbabilityVector:
        """Fuse regime and external scores into one distribution."""

        if not isinstance(profile, RegimeProfile):
            raise ContractError(
                "profile must be a RegimeProfile"
            )

        if metadata is None:
            metadata = {}

        if not isinstance(metadata, Mapping):
            raise ContractError(
                "metadata must be a mapping"
            )

        weights = self._validate_weights()
        raw_records: list[
            tuple[int, float, dict[str, float]]
        ] = []

        for regime in profile.regimes:
            components = {
                "hot": regime.hot_score,
                "cold": regime.cold_score,
                "gap": regime.gap_score,
                "trend": regime.trend_score,
                "transition": regime.transition_score,
                "learning": self._external_score(
                    learning_scores,
                    number=regime.number,
                    field_name="learning_scores",
                ),
                "adaptive": self._external_score(
                    adaptive_scores,
                    number=regime.number,
                    field_name="adaptive_scores",
                ),
            }

            raw_score = sum(
                components[name] * weights[name]
                for name in weights
            )

            raw_records.append(
                (
                    regime.number,
                    raw_score,
                    components,
                )
            )

        raw_total = sum(
            record[1]
            for record in raw_records
        )

        if raw_total <= 0.0:
            raise ContractError(
                "fused raw probability total must be positive"
            )

        ranked = sorted(
            raw_records,
            key=lambda record: (
                -record[1],
                record[0],
            ),
        )
        rank_by_number = {
            record[0]: rank
            for rank, record in enumerate(
                ranked,
                start=1,
            )
        }

        probabilities = ordered_probabilities(
            NumberProbability(
                number=number,
                probability=raw_score / raw_total,
                raw_score=raw_score,
                rank=rank_by_number[number],
                components=components,
                metadata={
                    "regime_confidence": (
                        profile.get(number).confidence
                    ),
                    "dominant_regime": (
                        profile.get(number).dominant_regime
                    ),
                },
            )
            for number, raw_score, components
            in raw_records
        )

        vector_metadata = {
            "engine": "F-002",
            "fusion": type(self).__name__,
            "weights": dict(weights),
            "source_engine": profile.metadata.get(
                "engine"
            ),
        }
        vector_metadata.update(
            dict(metadata)
        )

        return ProbabilityVector(
            round_no=profile.round_no,
            generated_at_kst=profile.generated_at_kst,
            probabilities=probabilities,
            metadata=vector_metadata,
        )
