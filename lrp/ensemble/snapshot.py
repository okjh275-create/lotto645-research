"""Immutable learning snapshot models for Project E."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from lrp.contracts import ContractError

from .models import StrategyWeight


def _required_text(
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


@dataclass(frozen=True, slots=True)
class StrategyStatisticSnapshot:
    """Serializable Project E view of one M6 strategy statistic."""

    strategy_type: str
    strategy_name: str
    sample_count: int
    average_match_count: float
    average_prediction_score: float
    hit3_plus_rate: float
    prize_rate: float
    updated_at_kst: str
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

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

        average_match_count = _finite_number(
            self.average_match_count,
            field_name="average_match_count",
        )

        if not 0.0 <= average_match_count <= 6.0:
            raise ContractError(
                "average_match_count must be between 0 and 6"
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

        updated_at_kst = _required_text(
            self.updated_at_kst,
            field_name="updated_at_kst",
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
            "sample_count",
            sample_count,
        )
        object.__setattr__(
            self,
            "average_match_count",
            average_match_count,
        )
        object.__setattr__(
            self,
            "average_prediction_score",
            average_prediction_score,
        )
        object.__setattr__(
            self,
            "hit3_plus_rate",
            hit3_plus_rate,
        )
        object.__setattr__(
            self,
            "prize_rate",
            prize_rate,
        )
        object.__setattr__(
            self,
            "updated_at_kst",
            updated_at_kst,
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
            "sample_count": self.sample_count,
            "average_match_count": self.average_match_count,
            "average_prediction_score": (
                self.average_prediction_score
            ),
            "hit3_plus_rate": self.hit3_plus_rate,
            "prize_rate": self.prize_rate,
            "updated_at_kst": self.updated_at_kst,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class StrategyRankingSnapshot:
    """Serializable Project E view of one M6 ranking."""

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
    rolling_matches: Mapping[int, float] = field(
        default_factory=dict
    )
    rolling_prize_rates: Mapping[int, float] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        strategy_type = _required_text(
            self.strategy_type,
            field_name="strategy_type",
        ).lower()

        strategy_name = _required_text(
            self.strategy_name,
            field_name="strategy_name",
        )

        if (
            isinstance(self.rank_position, bool)
            or not isinstance(self.rank_position, int)
            or self.rank_position <= 0
        ):
            raise ContractError(
                "rank_position must be a positive integer"
            )

        rank_score = _unit_interval(
            self.rank_score,
            field_name="rank_score",
        )

        confidence = _unit_interval(
            self.confidence,
            field_name="confidence",
        )

        stability = _unit_interval(
            self.stability,
            field_name="stability",
        )

        trend = _required_text(
            self.trend,
            field_name="trend",
        ).upper()

        if trend not in {
            "UP",
            "FLAT",
            "DOWN",
            "UNKNOWN",
        }:
            raise ContractError(
                "trend must be UP, FLAT, DOWN, or UNKNOWN"
            )

        recent_gain = _finite_number(
            self.recent_gain,
            field_name="recent_gain",
        )

        sample_count = _non_negative_integer(
            self.sample_count,
            field_name="sample_count",
        )

        average_match_count = _finite_number(
            self.average_match_count,
            field_name="average_match_count",
        )

        average_prediction_score = _unit_interval(
            self.average_prediction_score,
            field_name="average_prediction_score",
        )

        prize_rate = _unit_interval(
            self.prize_rate,
            field_name="prize_rate",
        )

        rolling_matches = {
            int(window): _finite_number(
                value,
                field_name=(
                    f"rolling_matches[{window}]"
                ),
            )
            for window, value
            in self.rolling_matches.items()
        }

        rolling_prize_rates = {
            int(window): _unit_interval(
                value,
                field_name=(
                    f"rolling_prize_rates[{window}]"
                ),
            )
            for window, value
            in self.rolling_prize_rates.items()
        }

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
            "rank_score",
            rank_score,
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
            "recent_gain",
            recent_gain,
        )
        object.__setattr__(
            self,
            "sample_count",
            sample_count,
        )
        object.__setattr__(
            self,
            "average_match_count",
            average_match_count,
        )
        object.__setattr__(
            self,
            "average_prediction_score",
            average_prediction_score,
        )
        object.__setattr__(
            self,
            "prize_rate",
            prize_rate,
        )
        object.__setattr__(
            self,
            "rolling_matches",
            rolling_matches,
        )
        object.__setattr__(
            self,
            "rolling_prize_rates",
            rolling_prize_rates,
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
            "rank_position": self.rank_position,
            "rank_score": self.rank_score,
            "confidence": self.confidence,
            "stability": self.stability,
            "trend": self.trend,
            "recent_gain": self.recent_gain,
            "sample_count": self.sample_count,
            "average_match_count": (
                self.average_match_count
            ),
            "average_prediction_score": (
                self.average_prediction_score
            ),
            "prize_rate": self.prize_rate,
            "rolling_matches": {
                str(window): value
                for window, value
                in sorted(self.rolling_matches.items())
            },
            "rolling_prize_rates": {
                str(window): value
                for window, value
                in sorted(
                    self.rolling_prize_rates.items()
                )
            },
        }


@dataclass(frozen=True, slots=True)
class LearningSnapshot:
    """Complete immutable M6 learning input for Project E."""

    round_no: int
    revision: tuple[int, int]
    statistics: tuple[
        StrategyStatisticSnapshot,
        ...,
    ]
    rankings: tuple[
        StrategyRankingSnapshot,
        ...,
    ]
    strategy_weights: tuple[
        StrategyWeight,
        ...,
    ]
    source: str = "lrp.learning"
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.round_no, bool)
            or not isinstance(self.round_no, int)
            or self.round_no <= 0
        ):
            raise ContractError(
                "round_no must be a positive integer"
            )

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
            raise ContractError(
                "revision must contain two "
                "non-negative integers"
            )

        statistics = tuple(self.statistics)
        rankings = tuple(self.rankings)
        strategy_weights = tuple(
            self.strategy_weights
        )

        if not all(
            isinstance(
                item,
                StrategyStatisticSnapshot,
            )
            for item in statistics
        ):
            raise ContractError(
                "statistics contains invalid items"
            )

        if not all(
            isinstance(
                item,
                StrategyRankingSnapshot,
            )
            for item in rankings
        ):
            raise ContractError(
                "rankings contains invalid items"
            )

        if not all(
            isinstance(item, StrategyWeight)
            for item in strategy_weights
        ):
            raise ContractError(
                "strategy_weights contains invalid items"
            )

        source = _required_text(
            self.source,
            field_name="source",
        )

        if not isinstance(self.metadata, Mapping):
            raise ContractError(
                "metadata must be a mapping"
            )

        statistic_keys = [
            item.key
            for item in statistics
        ]
        ranking_keys = [
            item.key
            for item in rankings
        ]
        weight_keys = [
            item.key
            for item in strategy_weights
        ]

        if len(statistic_keys) != len(
            set(statistic_keys)
        ):
            raise ContractError(
                "statistics contains duplicate strategies"
            )

        if len(ranking_keys) != len(
            set(ranking_keys)
        ):
            raise ContractError(
                "rankings contains duplicate strategies"
            )

        if len(weight_keys) != len(
            set(weight_keys)
        ):
            raise ContractError(
                "strategy_weights contains duplicate strategies"
            )

        object.__setattr__(
            self,
            "revision",
            revision,
        )
        object.__setattr__(
            self,
            "statistics",
            statistics,
        )
        object.__setattr__(
            self,
            "rankings",
            rankings,
        )
        object.__setattr__(
            self,
            "strategy_weights",
            strategy_weights,
        )
        object.__setattr__(
            self,
            "source",
            source,
        )
        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )

    @property
    def strategy_count(self) -> int:
        return len(
            {
                item.key
                for item in (
                    *self.statistics,
                    *self.rankings,
                    *self.strategy_weights,
                )
            }
        )

    def statistic(
        self,
        strategy_type: str,
        strategy_name: str,
    ) -> StrategyStatisticSnapshot | None:
        key = (
            strategy_type.strip().lower(),
            strategy_name.strip(),
        )

        return next(
            (
                item
                for item in self.statistics
                if item.key == key
            ),
            None,
        )

    def ranking(
        self,
        strategy_type: str,
        strategy_name: str,
    ) -> StrategyRankingSnapshot | None:
        key = (
            strategy_type.strip().lower(),
            strategy_name.strip(),
        )

        return next(
            (
                item
                for item in self.rankings
                if item.key == key
            ),
            None,
        )

    def weight(
        self,
        strategy_type: str,
        strategy_name: str,
    ) -> StrategyWeight | None:
        key = (
            strategy_type.strip().lower(),
            strategy_name.strip(),
        )

        return next(
            (
                item
                for item in self.strategy_weights
                if item.key == key
            ),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_no": self.round_no,
            "revision": list(self.revision),
            "source": self.source,
            "strategy_count": self.strategy_count,
            "statistics": [
                item.to_dict()
                for item in self.statistics
            ],
            "rankings": [
                item.to_dict()
                for item in self.rankings
            ],
            "strategy_weights": [
                item.to_dict()
                for item in self.strategy_weights
            ],
            "metadata": dict(self.metadata),
        }


def ordered_statistics(
    values: Sequence[
        StrategyStatisticSnapshot
    ],
) -> tuple[
    StrategyStatisticSnapshot,
    ...,
]:
    return tuple(
        sorted(
            values,
            key=lambda item: (
                item.strategy_type,
                item.strategy_name,
            ),
        )
    )


def ordered_rankings(
    values: Sequence[
        StrategyRankingSnapshot
    ],
) -> tuple[
    StrategyRankingSnapshot,
    ...,
]:
    return tuple(
        sorted(
            values,
            key=lambda item: (
                item.strategy_type,
                item.rank_position,
                item.strategy_name,
            ),
        )
    )
