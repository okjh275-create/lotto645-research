"""Strategy feature-vector construction for Project E."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from lrp.contracts import ContractError

from .snapshot import LearningSnapshot


def _finite(
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

    result = float(value)

    if not math.isfinite(result):
        raise ContractError(
            f"{field_name} must be finite"
        )

    return result


def _unit(
    value: object,
    *,
    field_name: str,
) -> float:
    result = _finite(
        value,
        field_name=field_name,
    )

    if not 0.0 <= result <= 1.0:
        raise ContractError(
            f"{field_name} must be between 0 and 1"
        )

    return result


def trend_value(trend: str) -> float:
    normalized = str(trend).strip().upper()

    return {
        "UP": 1.0,
        "FLAT": 0.5,
        "DOWN": 0.0,
        "UNKNOWN": 0.5,
    }.get(normalized, 0.5)


@dataclass(frozen=True, slots=True)
class StrategyFeatureVector:
    """Normalized learning features for one strategy."""

    strategy_type: str
    strategy_name: str

    adaptive_weight: float
    rank_score: float
    confidence: float
    stability: float
    trend_score: float

    average_match_score: float
    average_prediction_score: float
    prize_rate: float
    sample_confidence: float

    evidence_count: int
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        strategy_type = str(
            self.strategy_type
        ).strip().lower()

        strategy_name = str(
            self.strategy_name
        ).strip()

        if not strategy_type:
            raise ContractError(
                "strategy_type must not be empty"
            )

        if not strategy_name:
            raise ContractError(
                "strategy_name must not be empty"
            )

        for field_name in (
            "adaptive_weight",
            "rank_score",
            "confidence",
            "stability",
            "trend_score",
            "average_match_score",
            "average_prediction_score",
            "prize_rate",
            "sample_confidence",
        ):
            _unit(
                getattr(self, field_name),
                field_name=field_name,
            )

        if (
            isinstance(self.evidence_count, bool)
            or not isinstance(
                self.evidence_count,
                int,
            )
            or self.evidence_count < 0
        ):
            raise ContractError(
                "evidence_count must be a "
                "non-negative integer"
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
            "adaptive_weight": self.adaptive_weight,
            "rank_score": self.rank_score,
            "confidence": self.confidence,
            "stability": self.stability,
            "trend_score": self.trend_score,
            "average_match_score": (
                self.average_match_score
            ),
            "average_prediction_score": (
                self.average_prediction_score
            ),
            "prize_rate": self.prize_rate,
            "sample_confidence": (
                self.sample_confidence
            ),
            "evidence_count": self.evidence_count,
            "metadata": dict(self.metadata),
        }


def build_strategy_feature(
    snapshot: LearningSnapshot,
    *,
    strategy_type: str,
    strategy_name: str,
    sample_saturation: int = 20,
) -> StrategyFeatureVector | None:
    """Merge M6 weight, ranking and statistic records."""

    if sample_saturation <= 0:
        raise ContractError(
            "sample_saturation must be positive"
        )

    strategy_type = str(
        strategy_type
    ).strip().lower()
    strategy_name = str(
        strategy_name
    ).strip()

    weight = snapshot.weight(
        strategy_type,
        strategy_name,
    )
    ranking = snapshot.ranking(
        strategy_type,
        strategy_name,
    )
    statistic = snapshot.statistic(
        strategy_type,
        strategy_name,
    )

    if (
        weight is None
        and ranking is None
        and statistic is None
    ):
        return None

    sample_counts = [
        int(item.sample_count)
        for item in (
            weight,
            ranking,
            statistic,
        )
        if item is not None
    ]

    sample_count = max(
        sample_counts,
        default=0,
    )

    adaptive_weight = (
        weight.normalized_weight
        if weight is not None
        else 0.0
    )

    rank_score = (
        ranking.rank_score
        if ranking is not None
        else 0.0
    )

    confidence_values = [
        float(item.confidence)
        for item in (
            weight,
            ranking,
        )
        if item is not None
    ]

    stability_values = [
        float(item.stability)
        for item in (
            weight,
            ranking,
        )
        if item is not None
    ]

    trend = (
        ranking.trend
        if ranking is not None
        else (
            weight.trend
            if weight is not None
            else "UNKNOWN"
        )
    )

    average_match_count = (
        statistic.average_match_count
        if statistic is not None
        else (
            ranking.average_match_count
            if ranking is not None
            else 0.0
        )
    )

    average_prediction_score = (
        statistic.average_prediction_score
        if statistic is not None
        else (
            ranking.average_prediction_score
            if ranking is not None
            else 0.0
        )
    )

    prize_rate = (
        statistic.prize_rate
        if statistic is not None
        else (
            ranking.prize_rate
            if ranking is not None
            else 0.0
        )
    )

    return StrategyFeatureVector(
        strategy_type=strategy_type,
        strategy_name=strategy_name,
        adaptive_weight=max(
            0.0,
            min(1.0, adaptive_weight),
        ),
        rank_score=max(
            0.0,
            min(1.0, rank_score),
        ),
        confidence=max(
            0.0,
            min(
                1.0,
                (
                    sum(confidence_values)
                    / len(confidence_values)
                    if confidence_values
                    else 0.0
                ),
            ),
        ),
        stability=max(
            0.0,
            min(
                1.0,
                (
                    sum(stability_values)
                    / len(stability_values)
                    if stability_values
                    else 0.0
                ),
            ),
        ),
        trend_score=trend_value(trend),
        average_match_score=max(
            0.0,
            min(
                1.0,
                float(average_match_count) / 6.0,
            ),
        ),
        average_prediction_score=max(
            0.0,
            min(
                1.0,
                float(average_prediction_score),
            ),
        ),
        prize_rate=max(
            0.0,
            min(1.0, float(prize_rate)),
        ),
        sample_confidence=max(
            0.0,
            min(
                1.0,
                sample_count / sample_saturation,
            ),
        ),
        evidence_count=sum(
            item is not None
            for item in (
                weight,
                ranking,
                statistic,
            )
        ),
        metadata={
            "round_no": snapshot.round_no,
            "revision": list(
                snapshot.revision
            ),
            "sample_count": sample_count,
            "has_weight": weight is not None,
            "has_ranking": ranking is not None,
            "has_statistic": statistic is not None,
        },
    )


def build_feature_catalog(
    snapshot: LearningSnapshot,
    *,
    sample_saturation: int = 20,
) -> dict[
    tuple[str, str],
    StrategyFeatureVector,
]:
    keys = {
        item.key
        for item in (
            *snapshot.statistics,
            *snapshot.rankings,
            *snapshot.strategy_weights,
        )
    }

    result: dict[
        tuple[str, str],
        StrategyFeatureVector,
    ] = {}

    for strategy_type, strategy_name in sorted(
        keys
    ):
        vector = build_strategy_feature(
            snapshot,
            strategy_type=strategy_type,
            strategy_name=strategy_name,
            sample_saturation=sample_saturation,
        )

        if vector is not None:
            result[vector.key] = vector

    return result
