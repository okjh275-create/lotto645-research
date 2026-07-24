"""Immutable models for M6-005 adaptive strategy weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .ranking import StrategyRanking


AdaptiveRevision = tuple[int, int]
StrategyKey = tuple[str, str]


@dataclass(frozen=True, slots=True)
class AdaptiveWeightDataset:
    """Immutable inputs consumed by AdaptiveWeightEngine."""

    revision: AdaptiveRevision
    rankings: tuple[StrategyRanking, ...]
    previous_weights: Mapping[StrategyKey, float]

    @property
    def strategy_count(self) -> int:
        return len(self.rankings)


@dataclass(frozen=True, slots=True)
class AdaptiveWeight:
    """Derived adaptive priority for one strategy."""

    strategy_type: str
    strategy_name: str
    rank_position: int
    rank_score: float
    target_weight: float
    previous_weight: float
    current_weight: float
    normalized_weight: float
    confidence: float
    stability: float
    trend: str
    sample_count: int
    revision: AdaptiveRevision

    def __post_init__(self) -> None:
        if not self.strategy_type.strip():
            raise ValueError("strategy_type must not be empty")
        if not self.strategy_name.strip():
            raise ValueError("strategy_name must not be empty")
        if self.rank_position <= 0:
            raise ValueError("rank_position must be positive")
        if self.sample_count < 0:
            raise ValueError("sample_count must not be negative")
        if self.trend not in {"UP", "DOWN", "FLAT"}:
            raise ValueError("trend must be UP, DOWN, or FLAT")

        bounded = (
            self.rank_score,
            self.target_weight,
            self.previous_weight,
            self.current_weight,
            self.normalized_weight,
            self.confidence,
            self.stability,
        )
        if any(value < 0.0 for value in bounded):
            raise ValueError("adaptive weight values must be non-negative")

    @property
    def strategy_key(self) -> StrategyKey:
        return (self.strategy_type, self.strategy_name)

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy_type": self.strategy_type,
            "strategy_name": self.strategy_name,
            "rank_position": self.rank_position,
            "rank_score": round(self.rank_score, 6),
            "target_weight": round(self.target_weight, 6),
            "previous_weight": round(self.previous_weight, 6),
            "current_weight": round(self.current_weight, 6),
            "normalized_weight": round(self.normalized_weight, 6),
            "confidence": round(self.confidence, 6),
            "stability": round(self.stability, 6),
            "trend": self.trend,
            "sample_count": self.sample_count,
            "revision": list(self.revision),
        }
