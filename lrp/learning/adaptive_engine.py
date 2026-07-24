"""Deterministic adaptive-weight calculation for M6-005."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .adaptive_models import AdaptiveWeight, AdaptiveWeightDataset
from .ranking import StrategyRanking


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(frozen=True, slots=True)
class AdaptiveWeightFactors:
    """Factors used to derive one target weight."""

    rank_score: float = 0.60
    confidence: float = 0.25
    stability: float = 0.15

    def __post_init__(self) -> None:
        values = (self.rank_score, self.confidence, self.stability)
        if any(value < 0.0 for value in values):
            raise ValueError("adaptive weight factors must be non-negative")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
            raise ValueError("adaptive weight factors must sum to 1.0")


@dataclass(frozen=True, slots=True)
class AdaptiveWeightConfig:
    """Stable RC1 calculation parameters."""

    base_weight: float = 1.0
    learning_rate: float = 0.20
    minimum_weight: float = 0.50
    maximum_weight: float = 1.50
    factors: AdaptiveWeightFactors = field(default_factory=AdaptiveWeightFactors)

    def __post_init__(self) -> None:
        if self.base_weight <= 0.0:
            raise ValueError("base_weight must be positive")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be between 0 and 1")
        if self.minimum_weight <= 0.0:
            raise ValueError("minimum_weight must be positive")
        if self.maximum_weight < self.minimum_weight:
            raise ValueError("maximum_weight must be >= minimum_weight")


class AdaptiveWeightEngine:
    """Convert rankings into bounded, normalized adaptive weights."""

    def __init__(self, config: AdaptiveWeightConfig | None = None) -> None:
        self.config = config or AdaptiveWeightConfig()

    def calculate(
        self,
        dataset: AdaptiveWeightDataset,
    ) -> tuple[AdaptiveWeight, ...]:
        if not dataset.rankings:
            return ()

        provisional = [
            self._calculate_one(ranking=item, dataset=dataset)
            for item in dataset.rankings
        ]
        total = sum(float(item["current_weight"]) for item in provisional)
        if total <= 0.0:
            raise RuntimeError("adaptive current-weight total must be positive")

        results = tuple(
            AdaptiveWeight(
                **item,
                normalized_weight=float(item["current_weight"]) / total,
            )
            for item in provisional
        )
        return tuple(
            sorted(
                results,
                key=lambda item: (
                    -item.normalized_weight,
                    item.rank_position,
                    item.strategy_type,
                    item.strategy_name,
                ),
            )
        )

    def _calculate_one(
        self,
        *,
        ranking: StrategyRanking,
        dataset: AdaptiveWeightDataset,
    ) -> dict[str, object]:
        factors = self.config.factors
        unit_target = (
            factors.rank_score * ranking.rank_score
            + factors.confidence * ranking.confidence
            + factors.stability * ranking.stability
        )
        target_weight = self.config.minimum_weight + unit_target * (
            self.config.maximum_weight - self.config.minimum_weight
        )
        target_weight = _clamp(
            target_weight,
            self.config.minimum_weight,
            self.config.maximum_weight,
        )

        key = (ranking.strategy_type, ranking.strategy_name)
        previous_weight = float(
            dataset.previous_weights.get(key, self.config.base_weight)
        )
        previous_weight = _clamp(
            previous_weight,
            self.config.minimum_weight,
            self.config.maximum_weight,
        )
        current_weight = (
            (1.0 - self.config.learning_rate) * previous_weight
            + self.config.learning_rate * target_weight
        )
        current_weight = _clamp(
            current_weight,
            self.config.minimum_weight,
            self.config.maximum_weight,
        )

        return {
            "strategy_type": ranking.strategy_type,
            "strategy_name": ranking.strategy_name,
            "rank_position": ranking.rank_position,
            "rank_score": ranking.rank_score,
            "target_weight": target_weight,
            "previous_weight": previous_weight,
            "current_weight": current_weight,
            "confidence": ranking.confidence,
            "stability": ranking.stability,
            "trend": ranking.trend,
            "sample_count": ranking.sample_count,
            "revision": dataset.revision,
        }
