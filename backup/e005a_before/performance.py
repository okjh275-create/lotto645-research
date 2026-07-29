"""Read-only strategy performance analysis for Project E E-005A."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from .ranking import (
    RankingConfig,
    StrategyPerformancePoint,
    StrategyRanking,
    StrategyRankingEngine,
)
from .ranking_repository import RankingRepository
from .strategy_stats import StrategyStatistics


_KST = ZoneInfo("Asia/Seoul")
_VALID_TRENDS = {"UP", "DOWN", "FLAT"}


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _finite_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _unit_interval(value: object, *, field_name: str) -> float:
    normalized = _finite_number(value, field_name=field_name)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return normalized


def _non_negative_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _positive_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _normalize_window_mapping(
    values: Mapping[int, float],
    *,
    field_name: str,
    unit_interval: bool,
) -> dict[int, float]:
    normalized: dict[int, float] = {}
    for raw_window, raw_value in values.items():
        try:
            window = int(raw_window)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} windows must be integers") from exc
        _positive_integer(window, field_name=f"{field_name} window")
        if window in normalized:
            raise ValueError(f"{field_name} contains duplicate windows")
        if unit_interval:
            value = _unit_interval(
                raw_value,
                field_name=f"{field_name}[{window}]",
            )
        else:
            value = _finite_number(
                raw_value,
                field_name=f"{field_name}[{window}]",
            )
            if not 0.0 <= value <= 6.0:
                raise ValueError(
                    f"{field_name}[{window}] must be between 0 and 6"
                )
        normalized[window] = value
    return normalized


@dataclass(frozen=True, slots=True)
class StrategyPerformanceSummary:
    """Standard read-only performance view for one strategy."""

    strategy_type: str
    strategy_name: str
    sample_count: int
    history_count: int
    average_match_count: float
    best_match_count: int
    worst_match_count: int
    average_prediction_score: float
    hit3_plus_rate: float
    prize_rate: float
    confidence: float
    stability: float
    trend: str
    recent_gain: float
    rank_score: float
    rank_position: int
    rolling_matches: Mapping[int, float] = field(default_factory=dict)
    rolling_prize_rates: Mapping[int, float] = field(default_factory=dict)
    history: tuple[StrategyPerformancePoint, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        strategy_type = _required_text(
            self.strategy_type,
            field_name="strategy_type",
        ).lower()
        strategy_name = _required_text(
            self.strategy_name,
            field_name="strategy_name",
        )
        sample_count = _non_negative_integer(
            self.sample_count,
            field_name="sample_count",
        )
        history_count = _non_negative_integer(
            self.history_count,
            field_name="history_count",
        )
        average_match_count = _finite_number(
            self.average_match_count,
            field_name="average_match_count",
        )
        if not 0.0 <= average_match_count <= 6.0:
            raise ValueError("average_match_count must be between 0 and 6")
        best_match_count = _non_negative_integer(
            self.best_match_count,
            field_name="best_match_count",
        )
        worst_match_count = _non_negative_integer(
            self.worst_match_count,
            field_name="worst_match_count",
        )
        if best_match_count > 6:
            raise ValueError("best_match_count must be between 0 and 6")
        if worst_match_count > 6:
            raise ValueError("worst_match_count must be between 0 and 6")
        if worst_match_count > best_match_count:
            raise ValueError(
                "worst_match_count must not exceed best_match_count"
            )
        average_prediction_score = _unit_interval(
            self.average_prediction_score,
            field_name="average_prediction_score",
        )
        hit3_plus_rate = _unit_interval(
            self.hit3_plus_rate,
            field_name="hit3_plus_rate",
        )
        prize_rate = _unit_interval(
            self.prize_rate,
            field_name="prize_rate",
        )
        confidence = _unit_interval(
            self.confidence,
            field_name="confidence",
        )
        stability = _unit_interval(
            self.stability,
            field_name="stability",
        )
        rank_score = _unit_interval(
            self.rank_score,
            field_name="rank_score",
        )
        trend = _required_text(self.trend, field_name="trend").upper()
        if trend not in _VALID_TRENDS:
            raise ValueError("trend must be UP, DOWN, or FLAT")
        recent_gain = _finite_number(
            self.recent_gain,
            field_name="recent_gain",
        )
        rank_position = _positive_integer(
            self.rank_position,
            field_name="rank_position",
        )
        history = tuple(self.history)
        if not all(
            isinstance(point, StrategyPerformancePoint)
            for point in history
        ):
            raise ValueError("history contains invalid performance points")
        if history_count != len(history):
            raise ValueError("history_count must equal history length")
        ordered_history = tuple(
            sorted(
                history,
                key=lambda point: (
                    point.round_no,
                    point.prediction_id,
                ),
            )
        )
        if history != ordered_history:
            raise ValueError("history must be in chronological order")
        observed_matches = tuple(point.match_count for point in history)
        if observed_matches:
            if best_match_count != max(observed_matches):
                raise ValueError("best_match_count must match history")
            if worst_match_count != min(observed_matches):
                raise ValueError("worst_match_count must match history")
        elif best_match_count != 0 or worst_match_count != 0:
            raise ValueError(
                "empty history requires zero best/worst match counts"
            )
        rolling_matches = _normalize_window_mapping(
            self.rolling_matches,
            field_name="rolling_matches",
            unit_interval=False,
        )
        rolling_prize_rates = _normalize_window_mapping(
            self.rolling_prize_rates,
            field_name="rolling_prize_rates",
            unit_interval=True,
        )
        object.__setattr__(self, "strategy_type", strategy_type)
        object.__setattr__(self, "strategy_name", strategy_name)
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "history_count", history_count)
        object.__setattr__(self, "average_match_count", average_match_count)
        object.__setattr__(self, "best_match_count", best_match_count)
        object.__setattr__(self, "worst_match_count", worst_match_count)
        object.__setattr__(
            self,
            "average_prediction_score",
            average_prediction_score,
        )
        object.__setattr__(self, "hit3_plus_rate", hit3_plus_rate)
        object.__setattr__(self, "prize_rate", prize_rate)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "stability", stability)
        object.__setattr__(self, "trend", trend)
        object.__setattr__(self, "recent_gain", recent_gain)
        object.__setattr__(self, "rank_score", rank_score)
        object.__setattr__(self, "rank_position", rank_position)
        object.__setattr__(self, "rolling_matches", rolling_matches)
        object.__setattr__(self, "rolling_prize_rates", rolling_prize_rates)
        object.__setattr__(self, "history", history)

    @property
    def key(self) -> tuple[str, str]:
        return self.strategy_type, self.strategy_name

    def as_dict(self, *, include_history: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "strategy_type": self.strategy_type,
            "strategy_name": self.strategy_name,
            "sample_count": self.sample_count,
            "history_count": self.history_count,
            "average_match_count": round(self.average_match_count, 6),
            "best_match_count": self.best_match_count,
            "worst_match_count": self.worst_match_count,
            "average_prediction_score": round(
                self.average_prediction_score,
                6,
            ),
            "hit3_plus_rate": round(self.hit3_plus_rate, 6),
            "prize_rate": round(self.prize_rate, 6),
            "confidence": round(self.confidence, 6),
            "stability": round(self.stability, 6),
            "trend": self.trend,
            "recent_gain": round(self.recent_gain, 6),
            "rank_score": round(self.rank_score, 6),
            "rank_position": self.rank_position,
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
        if include_history:
            payload["history"] = [
                {
                    "prediction_id": point.prediction_id,
                    "round_no": point.round_no,
                    "match_count": point.match_count,
                    "prediction_score": round(point.prediction_score, 6),
                    "prize_rank": point.prize_rank,
                }
                for point in self.history
            ]
        return payload


@dataclass(frozen=True, slots=True)
class StrategyPerformanceReport:
    """Immutable E-005A performance-analysis result."""

    revision: tuple[int, int]
    strategy_type: str | None
    history_limit: int
    generated_at_kst: str
    summaries: tuple[StrategyPerformanceSummary, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        revision = tuple(self.revision)
        if (
            len(revision) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in revision
            )
        ):
            raise ValueError(
                "revision must contain two non-negative integers"
            )
        strategy_type = self.strategy_type
        if strategy_type is not None:
            strategy_type = _required_text(
                strategy_type,
                field_name="strategy_type",
            ).lower()
        history_limit = _positive_integer(
            self.history_limit,
            field_name="history_limit",
        )
        generated_at_kst = _required_text(
            self.generated_at_kst,
            field_name="generated_at_kst",
        )
        summaries = tuple(self.summaries)
        if not all(
            isinstance(item, StrategyPerformanceSummary)
            for item in summaries
        ):
            raise ValueError("summaries contains invalid items")
        keys = [item.key for item in summaries]
        if len(keys) != len(set(keys)):
            raise ValueError("summaries contains duplicate strategies")
        expected_order = tuple(
            sorted(
                summaries,
                key=lambda item: (
                    item.rank_position,
                    item.strategy_type,
                    item.strategy_name,
                ),
            )
        )
        if summaries != expected_order:
            raise ValueError("summaries must be ordered by rank position")
        if strategy_type is not None and any(
            item.strategy_type != strategy_type
            for item in summaries
        ):
            raise ValueError(
                "summary strategy_type does not match report filter"
            )
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "strategy_type", strategy_type)
        object.__setattr__(self, "history_limit", history_limit)
        object.__setattr__(self, "generated_at_kst", generated_at_kst)
        object.__setattr__(self, "summaries", summaries)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def strategy_count(self) -> int:
        return len(self.summaries)

    @property
    def total_samples(self) -> int:
        return sum(item.sample_count for item in self.summaries)

    @property
    def total_history_points(self) -> int:
        return sum(item.history_count for item in self.summaries)

    def get(
        self,
        strategy_type: str,
        strategy_name: str,
    ) -> StrategyPerformanceSummary | None:
        key = (
            _required_text(
                strategy_type,
                field_name="strategy_type",
            ).lower(),
            _required_text(
                strategy_name,
                field_name="strategy_name",
            ),
        )
        return next(
            (
                item
                for item in self.summaries
                if item.key == key
            ),
            None,
        )

    def as_dict(self, *, include_history: bool = True) -> dict[str, Any]:
        return {
            "revision": list(self.revision),
            "strategy_type": self.strategy_type,
            "history_limit": self.history_limit,
            "generated_at_kst": self.generated_at_kst,
            "strategy_count": self.strategy_count,
            "total_samples": self.total_samples,
            "total_history_points": self.total_history_points,
            "summaries": [
                item.as_dict(include_history=include_history)
                for item in self.summaries
            ],
            "metadata": dict(self.metadata),
        }


class PerformanceAnalyzer:
    """Build deterministic reports without modifying learning data."""

    def __init__(
        self,
        repository: RankingRepository,
        *,
        ranking_config: RankingConfig | None = None,
    ) -> None:
        if not isinstance(repository, RankingRepository):
            raise TypeError("repository must be a RankingRepository")
        self.repository = repository
        self.ranking_engine = StrategyRankingEngine(ranking_config)

    def analyze(
        self,
        *,
        strategy_type: str | None = None,
        history_limit: int = 100,
        generated_at_kst: str | None = None,
    ) -> StrategyPerformanceReport:
        history_limit = _positive_integer(
            history_limit,
            field_name="history_limit",
        )
        normalized_type = (
            None
            if strategy_type is None
            else _required_text(
                strategy_type,
                field_name="strategy_type",
            ).lower()
        )
        dataset = self.repository.build_dataset(
            strategy_type=normalized_type,
            history_limit=history_limit,
        )
        rankings = self.ranking_engine.rank(
            dataset.statistics,
            dataset.histories,
        )
        statistics_by_key = {
            (item.strategy_type, item.strategy_name): item
            for item in dataset.statistics
        }
        summaries = tuple(
            self._build_summary(
                ranking=ranking,
                statistics=statistics_by_key[
                    (
                        ranking.strategy_type,
                        ranking.strategy_name,
                    )
                ],
                history=tuple(
                    dataset.histories.get(
                        (
                            ranking.strategy_type,
                            ranking.strategy_name,
                        ),
                        (),
                    )
                ),
            )
            for ranking in rankings
        )
        timestamp = generated_at_kst
        if timestamp is None:
            timestamp = datetime.now(_KST).isoformat(timespec="seconds")
        else:
            timestamp = _required_text(
                timestamp,
                field_name="generated_at_kst",
            )
        return StrategyPerformanceReport(
            revision=dataset.revision,
            strategy_type=normalized_type,
            history_limit=history_limit,
            generated_at_kst=timestamp,
            summaries=summaries,
            metadata={
                "source": "lrp.learning",
                "analyzer": "E-005A",
                "read_only": True,
                "history_order": "chronological",
                "ranking_windows": list(
                    self.ranking_engine.config.windows
                ),
                "trend_short_window": (
                    self.ranking_engine.config.trend_short_window
                ),
                "trend_long_window": (
                    self.ranking_engine.config.trend_long_window
                ),
                "trend_threshold": (
                    self.ranking_engine.config.trend_threshold
                ),
            },
        )

    @staticmethod
    def _build_summary(
        *,
        ranking: StrategyRanking,
        statistics: StrategyStatistics,
        history: tuple[StrategyPerformancePoint, ...],
    ) -> StrategyPerformanceSummary:
        match_values = tuple(point.match_count for point in history)
        return StrategyPerformanceSummary(
            strategy_type=ranking.strategy_type,
            strategy_name=ranking.strategy_name,
            sample_count=statistics.sample_count,
            history_count=len(history),
            average_match_count=statistics.average_match_count,
            best_match_count=max(match_values) if match_values else 0,
            worst_match_count=min(match_values) if match_values else 0,
            average_prediction_score=(
                statistics.average_prediction_score
            ),
            hit3_plus_rate=statistics.hit3_plus_rate,
            prize_rate=statistics.prize_rate,
            confidence=ranking.confidence,
            stability=ranking.stability,
            trend=ranking.trend,
            recent_gain=ranking.recent_gain,
            rank_score=ranking.rank_score,
            rank_position=ranking.rank_position,
            rolling_matches=dict(ranking.rolling_matches),
            rolling_prize_rates=dict(ranking.rolling_prize_rates),
            history=history,
        )
