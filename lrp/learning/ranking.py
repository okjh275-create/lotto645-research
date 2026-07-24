"""Strategy ranking models and deterministic ranking engine.

M6-004 keeps ranking as a derived calculation layer.  No ranking table is
required: the engine consumes incrementally accumulated StrategyStatistics
plus a bounded recent-performance window supplied by the repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import fmean, pstdev
from typing import Iterable, Mapping, Sequence

from .strategy_stats import StrategyStatistics


_VALID_TRENDS = {"UP", "DOWN", "FLAT"}


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(fmean(values))


@dataclass(frozen=True, slots=True)
class StrategyPerformancePoint:
    """One reviewed prediction used by bounded rolling-window calculations."""

    prediction_id: str
    round_no: int
    match_count: int
    prediction_score: float
    prize_rank: int | None

    def __post_init__(self) -> None:
        if not self.prediction_id.strip():
            raise ValueError("prediction_id must not be empty")
        if self.round_no <= 0:
            raise ValueError("round_no must be positive")
        if self.match_count < 0 or self.match_count > 6:
            raise ValueError("match_count must be between 0 and 6")
        if not 0.0 <= float(self.prediction_score) <= 1.0:
            raise ValueError("prediction_score must be between 0 and 1")
        if self.prize_rank is not None and not 1 <= self.prize_rank <= 5:
            raise ValueError("prize_rank must be 1 to 5 or None")


@dataclass(frozen=True, slots=True)
class RankingWeights:
    """Weights used to build one normalized rank score."""

    performance: float = 0.42
    recent_gain: float = 0.18
    prize_rate: float = 0.12
    stability: float = 0.10
    confidence: float = 0.10
    prediction_quality: float = 0.08

    def __post_init__(self) -> None:
        values = (
            self.performance,
            self.recent_gain,
            self.prize_rate,
            self.stability,
            self.confidence,
            self.prediction_quality,
        )
        if any(value < 0.0 for value in values):
            raise ValueError("ranking weights must be non-negative")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
            raise ValueError("ranking weights must sum to 1.0")


@dataclass(frozen=True, slots=True)
class RankingConfig:
    """Deterministic ranking parameters."""

    windows: tuple[int, ...] = (10, 20, 50, 100)
    trend_short_window: int = 10
    trend_long_window: int = 30
    trend_threshold: float = 0.20
    confidence_scale: float = 20.0
    weights: RankingWeights = field(default_factory=RankingWeights)

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError("windows must not be empty")
        if any(window <= 0 for window in self.windows):
            raise ValueError("windows must contain positive integers")
        if len(set(self.windows)) != len(self.windows):
            raise ValueError("windows must not contain duplicates")
        if self.trend_short_window <= 0 or self.trend_long_window <= 0:
            raise ValueError("trend windows must be positive")
        if self.trend_short_window > self.trend_long_window:
            raise ValueError(
                "trend_short_window must be less than or equal to "
                "trend_long_window"
            )
        if self.trend_threshold < 0.0:
            raise ValueError("trend_threshold must be non-negative")
        if self.confidence_scale <= 0.0:
            raise ValueError("confidence_scale must be positive")


@dataclass(frozen=True, slots=True)
class StrategyRanking:
    """Derived ranking output for one model or scenario strategy."""

    strategy_type: str
    strategy_name: str
    rank_position: int
    rank_score: float
    confidence: float
    stability: float
    trend: str
    recent_gain: float
    sample_count: int
    average_match_count: float
    average_prediction_score: float
    prize_rate: float
    rolling_matches: Mapping[int, float]
    rolling_prize_rates: Mapping[int, float]

    def __post_init__(self) -> None:
        if not self.strategy_type.strip():
            raise ValueError("strategy_type must not be empty")
        if not self.strategy_name.strip():
            raise ValueError("strategy_name must not be empty")
        if self.rank_position <= 0:
            raise ValueError("rank_position must be positive")
        if self.trend not in _VALID_TRENDS:
            raise ValueError("trend must be UP, DOWN, or FLAT")

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy_type": self.strategy_type,
            "strategy_name": self.strategy_name,
            "rank_position": self.rank_position,
            "rank_score": round(self.rank_score, 6),
            "confidence": round(self.confidence, 6),
            "stability": round(self.stability, 6),
            "trend": self.trend,
            "recent_gain": round(self.recent_gain, 6),
            "sample_count": self.sample_count,
            "average_match_count": round(self.average_match_count, 6),
            "average_prediction_score": round(
                self.average_prediction_score,
                6,
            ),
            "prize_rate": round(self.prize_rate, 6),
            "rolling_matches": {
                str(window): round(value, 6)
                for window, value in sorted(self.rolling_matches.items())
            },
            "rolling_prize_rates": {
                str(window): round(value, 6)
                for window, value in sorted(
                    self.rolling_prize_rates.items()
                )
            },
        }


class StrategyRankingEngine:
    """Build deterministic, normalized rankings from bounded history."""

    def __init__(
        self,
        config: RankingConfig | None = None,
    ) -> None:
        self.config = config or RankingConfig()

    def rank(
        self,
        statistics: Sequence[StrategyStatistics],
        histories: Mapping[
            tuple[str, str],
            Sequence[StrategyPerformancePoint],
        ],
    ) -> tuple[StrategyRanking, ...]:
        """Rank strategies, then assign stable one-based positions."""

        provisional: list[dict[str, object]] = []

        for item in statistics:
            key = (item.strategy_type, item.strategy_name)
            history = tuple(histories.get(key, ()))
            provisional.append(
                self._calculate_one(
                    statistics=item,
                    history=history,
                )
            )

        provisional.sort(
            key=lambda row: (
                -float(row["rank_score"]),
                -float(row["confidence"]),
                -int(row["sample_count"]),
                str(row["strategy_type"]),
                str(row["strategy_name"]),
            )
        )

        return tuple(
            StrategyRanking(
                rank_position=index,
                **row,
            )
            for index, row in enumerate(provisional, start=1)
        )

    def _calculate_one(
        self,
        *,
        statistics: StrategyStatistics,
        history: Sequence[StrategyPerformancePoint],
    ) -> dict[str, object]:
        ordered_history = tuple(
            sorted(
                history,
                key=lambda point: (
                    point.round_no,
                    point.prediction_id,
                ),
            )
        )

        match_values = [
            float(point.match_count)
            for point in ordered_history
        ]
        prize_values = [
            1.0 if point.prize_rank is not None else 0.0
            for point in ordered_history
        ]

        rolling_matches = {
            window: _safe_mean(match_values[-window:])
            for window in self.config.windows
        }
        rolling_prize_rates = {
            window: _safe_mean(prize_values[-window:])
            for window in self.config.windows
        }

        confidence = self._confidence(statistics.sample_count)
        stability = self._stability(match_values)
        trend, recent_gain = self._trend(match_values)

        performance_component = _clamp(
            statistics.average_match_count / 6.0
        )
        recent_component = _clamp(
            0.5 + (recent_gain / 6.0)
        )
        prize_component = _clamp(statistics.prize_rate)
        prediction_quality_component = _clamp(
            statistics.average_prediction_score
        )

        weights = self.config.weights
        rank_score = _clamp(
            weights.performance * performance_component
            + weights.recent_gain * recent_component
            + weights.prize_rate * prize_component
            + weights.stability * stability
            + weights.confidence * confidence
            + weights.prediction_quality
            * prediction_quality_component
        )

        return {
            "strategy_type": statistics.strategy_type,
            "strategy_name": statistics.strategy_name,
            "rank_score": rank_score,
            "confidence": confidence,
            "stability": stability,
            "trend": trend,
            "recent_gain": recent_gain,
            "sample_count": statistics.sample_count,
            "average_match_count": statistics.average_match_count,
            "average_prediction_score": (
                statistics.average_prediction_score
            ),
            "prize_rate": statistics.prize_rate,
            "rolling_matches": rolling_matches,
            "rolling_prize_rates": rolling_prize_rates,
        }

    def _confidence(self, sample_count: int) -> float:
        if sample_count <= 0:
            return 0.0

        # Smoothly approaches 1.0 without allowing tiny samples to dominate.
        return _clamp(
            1.0 - math.exp(
                -float(sample_count)
                / self.config.confidence_scale
            )
        )

    @staticmethod
    def _stability(match_values: Sequence[float]) -> float:
        if not match_values:
            return 0.0
        if len(match_values) == 1:
            return 0.5

        deviation = float(pstdev(match_values))
        # Match counts are bounded to 0..6.  A deviation of three or more is
        # treated as maximally unstable.
        return _clamp(1.0 - deviation / 3.0)

    def _trend(
        self,
        match_values: Sequence[float],
    ) -> tuple[str, float]:
        if not match_values:
            return "FLAT", 0.0

        short_values = match_values[
            -self.config.trend_short_window:
        ]
        long_values = match_values[
            -self.config.trend_long_window:
        ]

        short_average = _safe_mean(short_values)
        long_average = _safe_mean(long_values)
        recent_gain = short_average - long_average

        if recent_gain > self.config.trend_threshold:
            return "UP", recent_gain
        if recent_gain < -self.config.trend_threshold:
            return "DOWN", recent_gain
        return "FLAT", recent_gain
