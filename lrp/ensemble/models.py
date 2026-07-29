"""Public data models for the ensemble engine."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from lrp.contracts import ContractError


_ALLOWED_STRATEGY_TYPES = frozenset(
    {
        "model",
        "scenario",
        "rule",
        "engine",
    }
)

_ALLOWED_TRENDS = frozenset(
    {
        "UP",
        "FLAT",
        "DOWN",
        "UNKNOWN",
    }
)


def _finite_number(
    value: object,
    *,
    field_name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ContractError(
            f"{field_name} must be a number"
        )

    normalized = float(value)

    if not math.isfinite(normalized):
        raise ContractError(
            f"{field_name} must be finite"
        )

    return normalized


def _unit_interval(
    value: object,
    *,
    field_name: str,
) -> float:
    normalized = _finite_number(
        value,
        field_name=field_name,
    )

    if not 0.0 <= normalized <= 1.0:
        raise ContractError(
            f"{field_name} must be between 0 and 1"
        )

    return normalized


def _non_negative_integer(
    value: object,
    *,
    field_name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ContractError(
            f"{field_name} must be a non-negative integer"
        )

    return value


def _normalized_text(
    value: object,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise ContractError(
            f"{field_name} must be a string"
        )

    normalized = value.strip()

    if not normalized:
        raise ContractError(
            f"{field_name} must not be empty"
        )

    return normalized


@dataclass(frozen=True, slots=True)
class StrategyWeight:
    """Normalized M6 strategy weight consumed by Project E."""

    strategy_type: str
    strategy_name: str
    current_weight: float
    normalized_weight: float
    confidence: float = 0.0
    stability: float = 0.0
    trend: str = "UNKNOWN"
    sample_count: int = 0
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        strategy_type = _normalized_text(
            self.strategy_type,
            field_name="strategy_type",
        ).lower()

        if strategy_type not in _ALLOWED_STRATEGY_TYPES:
            raise ContractError(
                "strategy_type must be one of: "
                + ", ".join(sorted(_ALLOWED_STRATEGY_TYPES))
            )

        strategy_name = _normalized_text(
            self.strategy_name,
            field_name="strategy_name",
        )

        current_weight = _finite_number(
            self.current_weight,
            field_name="current_weight",
        )
        if current_weight < 0.0:
            raise ContractError(
                "current_weight must be non-negative"
            )

        normalized_weight = _unit_interval(
            self.normalized_weight,
            field_name="normalized_weight",
        )

        confidence = _unit_interval(
            self.confidence,
            field_name="confidence",
        )
        stability = _unit_interval(
            self.stability,
            field_name="stability",
        )

        trend = _normalized_text(
            self.trend,
            field_name="trend",
        ).upper()

        if trend not in _ALLOWED_TRENDS:
            raise ContractError(
                "trend must be one of: "
                + ", ".join(sorted(_ALLOWED_TRENDS))
            )

        sample_count = _non_negative_integer(
            self.sample_count,
            field_name="sample_count",
        )

        if not isinstance(self.metadata, Mapping):
            raise ContractError(
                "metadata must be a mapping"
            )

        object.__setattr__(
            self,
            "strategy_type",
            strategy_type,
        )
        object.__setattr__(
            self,
            "strategy_name",
            strategy_name,
        )
        object.__setattr__(
            self,
            "current_weight",
            current_weight,
        )
        object.__setattr__(
            self,
            "normalized_weight",
            normalized_weight,
        )
        object.__setattr__(
            self,
            "confidence",
            confidence,
        )
        object.__setattr__(
            self,
            "stability",
            stability,
        )
        object.__setattr__(
            self,
            "trend",
            trend,
        )
        object.__setattr__(
            self,
            "sample_count",
            sample_count,
        )
        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )

    @property
    def key(self) -> tuple[str, str]:
        return (
            self.strategy_type,
            self.strategy_name,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_type": self.strategy_type,
            "strategy_name": self.strategy_name,
            "current_weight": self.current_weight,
            "normalized_weight": self.normalized_weight,
            "confidence": self.confidence,
            "stability": self.stability,
            "trend": self.trend,
            "sample_count": self.sample_count,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class EnsembleConfig:
    """Configuration for Project E candidate re-evaluation."""

    base_score_weight: float = 1.0
    adaptive_weight: float = 0.0
    confidence_weight: float = 0.0
    stability_weight: float = 0.0
    trend_weight: float = 0.0
    top_k: int | None = None

    def __post_init__(self) -> None:
        values = {
            "base_score_weight": self.base_score_weight,
            "adaptive_weight": self.adaptive_weight,
            "confidence_weight": self.confidence_weight,
            "stability_weight": self.stability_weight,
            "trend_weight": self.trend_weight,
        }

        normalized: dict[str, float] = {}

        for name, value in values.items():
            number = _finite_number(
                value,
                field_name=name,
            )

            if number < 0.0:
                raise ContractError(
                    f"{name} must be non-negative"
                )

            normalized[name] = number

        if sum(normalized.values()) <= 0.0:
            raise ContractError(
                "at least one ensemble weight must be positive"
            )

        if self.top_k is not None:
            if (
                isinstance(self.top_k, bool)
                or not isinstance(self.top_k, int)
                or self.top_k <= 0
            ):
                raise ContractError(
                    "top_k must be a positive integer or None"
                )

        for name, value in normalized.items():
            object.__setattr__(
                self,
                name,
                value,
            )

    @property
    def total_weight(self) -> float:
        return (
            self.base_score_weight
            + self.adaptive_weight
            + self.confidence_weight
            + self.stability_weight
            + self.trend_weight
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_score_weight": self.base_score_weight,
            "adaptive_weight": self.adaptive_weight,
            "confidence_weight": self.confidence_weight,
            "stability_weight": self.stability_weight,
            "trend_weight": self.trend_weight,
            "top_k": self.top_k,
        }


@dataclass(frozen=True, slots=True)
class EnsembleCandidateScore:
    """One Project E re-evaluated candidate."""

    source: object
    source_index: int
    base_score: float
    ensemble_score: float
    contributions: Mapping[str, float] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_index, bool)
            or not isinstance(self.source_index, int)
            or self.source_index < 0
        ):
            raise ContractError(
                "source_index must be a non-negative integer"
            )

        base_score = _unit_interval(
            self.base_score,
            field_name="base_score",
        )
        ensemble_score = _unit_interval(
            self.ensemble_score,
            field_name="ensemble_score",
        )

        if not isinstance(self.contributions, Mapping):
            raise ContractError(
                "contributions must be a mapping"
            )

        normalized_contributions: dict[str, float] = {}

        for name, value in self.contributions.items():
            key = _normalized_text(
                name,
                field_name="contribution name",
            )
            normalized_contributions[key] = _finite_number(
                value,
                field_name=f"contributions[{key}]",
            )

        object.__setattr__(
            self,
            "base_score",
            base_score,
        )
        object.__setattr__(
            self,
            "ensemble_score",
            ensemble_score,
        )
        object.__setattr__(
            self,
            "contributions",
            normalized_contributions,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_index": self.source_index,
            "base_score": self.base_score,
            "ensemble_score": self.ensemble_score,
            "contributions": dict(self.contributions),
        }


@dataclass(frozen=True, slots=True)
class EnsembleResult:
    """Complete Project E foundation result."""

    round_no: int
    items: tuple[EnsembleCandidateScore, ...]
    strategy_weights: tuple[StrategyWeight, ...]
    config: EnsembleConfig
    engine_version: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.round_no, bool)
            or not isinstance(self.round_no, int)
            or self.round_no <= 0
        ):
            raise ContractError(
                "round_no must be a positive integer"
            )

        items = tuple(self.items)
        weights = tuple(self.strategy_weights)

        if not all(
            isinstance(item, EnsembleCandidateScore)
            for item in items
        ):
            raise ContractError(
                "items must contain EnsembleCandidateScore values"
            )

        if not all(
            isinstance(weight, StrategyWeight)
            for weight in weights
        ):
            raise ContractError(
                "strategy_weights must contain StrategyWeight values"
            )

        if not isinstance(self.config, EnsembleConfig):
            raise ContractError(
                "config must be an EnsembleConfig"
            )

        engine_version = _normalized_text(
            self.engine_version,
            field_name="engine_version",
        )

        object.__setattr__(
            self,
            "items",
            items,
        )
        object.__setattr__(
            self,
            "strategy_weights",
            weights,
        )
        object.__setattr__(
            self,
            "engine_version",
            engine_version,
        )

    @property
    def count(self) -> int:
        return len(self.items)

    @property
    def selected_sources(self) -> tuple[object, ...]:
        return tuple(
            item.source
            for item in self.items
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_no": self.round_no,
            "count": self.count,
            "engine_version": self.engine_version,
            "config": self.config.to_dict(),
            "strategy_weights": [
                weight.to_dict()
                for weight in self.strategy_weights
            ],
            "items": [
                item.to_dict()
                for item in self.items
            ],
        }


def normalize_strategy_weights(
    values: Sequence[StrategyWeight],
) -> tuple[StrategyWeight, ...]:
    """Validate and return deterministic strategy order."""

    normalized = tuple(values)

    if not all(
        isinstance(value, StrategyWeight)
        for value in normalized
    ):
        raise ContractError(
            "values must contain StrategyWeight objects"
        )

    keys = [
        value.key
        for value in normalized
    ]

    if len(keys) != len(set(keys)):
        raise ContractError(
            "strategy weights contain duplicate keys"
        )

    return tuple(
        sorted(
            normalized,
            key=lambda value: (
                value.strategy_type,
                value.strategy_name,
            ),
        )
    )
