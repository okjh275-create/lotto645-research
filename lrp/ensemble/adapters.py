"""Adapters from M6 learning objects to Project E snapshots."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from lrp.contracts import ContractError

from .models import (
    StrategyWeight,
    normalize_strategy_weights,
)
from .snapshot import (
    StrategyRankingSnapshot,
    StrategyStatisticSnapshot,
    ordered_rankings,
    ordered_statistics,
)


def read_value(
    value: object,
    *names: str,
    default: Any = None,
) -> Any:
    for name in names:
        if isinstance(value, Mapping):
            if name in value:
                return value[name]

        if hasattr(value, name):
            return getattr(value, name)

    return default


def _required_text(
    value: object,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        value = str(value)

    normalized = value.strip()

    if not normalized:
        raise ContractError(
            f"{field_name} must not be empty"
        )

    return normalized


def _number(
    value: object,
    *,
    field_name: str,
    default: float = 0.0,
) -> float:
    if value is None:
        value = default

    if isinstance(value, bool):
        raise ContractError(
            f"{field_name} must be numeric"
        )

    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(
            f"{field_name} must be numeric"
        ) from exc

    if not math.isfinite(normalized):
        raise ContractError(
            f"{field_name} must be finite"
        )

    return normalized


def _integer(
    value: object,
    *,
    field_name: str,
    default: int = 0,
) -> int:
    if value is None:
        value = default

    if isinstance(value, bool):
        raise ContractError(
            f"{field_name} must be an integer"
        )

    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(
            f"{field_name} must be an integer"
        ) from exc

    return normalized


def _mapping(
    value: object,
) -> dict[Any, Any]:
    if value is None:
        return {}

    if not isinstance(value, Mapping):
        raise ContractError(
            "expected a mapping"
        )

    return dict(value)


def statistic_from_m6(
    value: object,
) -> StrategyStatisticSnapshot:
    metadata: dict[str, Any] = {}

    for name in (
        "hit3_count",
        "hit4_count",
        "hit5_count",
        "hit6_count",
        "prize_count",
    ):
        item = read_value(
            value,
            name,
            default=None,
        )

        if item is not None:
            metadata[name] = _integer(
                item,
                field_name=name,
            )

    return StrategyStatisticSnapshot(
        strategy_type=_required_text(
            read_value(value, "strategy_type"),
            field_name="strategy_type",
        ),
        strategy_name=_required_text(
            read_value(value, "strategy_name"),
            field_name="strategy_name",
        ),
        sample_count=_integer(
            read_value(
                value,
                "sample_count",
                default=0,
            ),
            field_name="sample_count",
        ),
        average_match_count=_number(
            read_value(
                value,
                "average_match_count",
                default=0.0,
            ),
            field_name="average_match_count",
        ),
        average_prediction_score=_number(
            read_value(
                value,
                "average_prediction_score",
                default=0.0,
            ),
            field_name="average_prediction_score",
        ),
        hit3_plus_rate=_number(
            read_value(
                value,
                "hit3_plus_rate",
                default=0.0,
            ),
            field_name="hit3_plus_rate",
        ),
        prize_rate=_number(
            read_value(
                value,
                "prize_rate",
                default=0.0,
            ),
            field_name="prize_rate",
        ),
        updated_at_kst=_required_text(
            read_value(
                value,
                "updated_at_kst",
                default="UNKNOWN",
            ),
            field_name="updated_at_kst",
        ),
        metadata=metadata,
    )


def ranking_from_m6(
    value: object,
) -> StrategyRankingSnapshot:
    rolling_matches = {
        int(window): _number(
            score,
            field_name=(
                f"rolling_matches[{window}]"
            ),
        )
        for window, score
        in _mapping(
            read_value(
                value,
                "rolling_matches",
                default={},
            )
        ).items()
    }

    rolling_prize_rates = {
        int(window): _number(
            score,
            field_name=(
                f"rolling_prize_rates[{window}]"
            ),
        )
        for window, score
        in _mapping(
            read_value(
                value,
                "rolling_prize_rates",
                default={},
            )
        ).items()
    }

    return StrategyRankingSnapshot(
        strategy_type=_required_text(
            read_value(value, "strategy_type"),
            field_name="strategy_type",
        ),
        strategy_name=_required_text(
            read_value(value, "strategy_name"),
            field_name="strategy_name",
        ),
        rank_position=_integer(
            read_value(
                value,
                "rank_position",
                default=1,
            ),
            field_name="rank_position",
        ),
        rank_score=_number(
            read_value(
                value,
                "rank_score",
                default=0.0,
            ),
            field_name="rank_score",
        ),
        confidence=_number(
            read_value(
                value,
                "confidence",
                default=0.0,
            ),
            field_name="confidence",
        ),
        stability=_number(
            read_value(
                value,
                "stability",
                default=0.0,
            ),
            field_name="stability",
        ),
        trend=_required_text(
            read_value(
                value,
                "trend",
                default="UNKNOWN",
            ),
            field_name="trend",
        ),
        recent_gain=_number(
            read_value(
                value,
                "recent_gain",
                default=0.0,
            ),
            field_name="recent_gain",
        ),
        sample_count=_integer(
            read_value(
                value,
                "sample_count",
                default=0,
            ),
            field_name="sample_count",
        ),
        average_match_count=_number(
            read_value(
                value,
                "average_match_count",
                default=0.0,
            ),
            field_name="average_match_count",
        ),
        average_prediction_score=_number(
            read_value(
                value,
                "average_prediction_score",
                default=0.0,
            ),
            field_name="average_prediction_score",
        ),
        prize_rate=_number(
            read_value(
                value,
                "prize_rate",
                default=0.0,
            ),
            field_name="prize_rate",
        ),
        rolling_matches=rolling_matches,
        rolling_prize_rates=(
            rolling_prize_rates
        ),
    )


def weight_from_m6(
    value: object,
) -> StrategyWeight:
    metadata: dict[str, Any] = {}

    for name in (
        "rank_position",
        "rank_score",
        "target_weight",
        "previous_weight",
        "recent_gain",
        "average_match_count",
        "average_prediction_score",
        "prize_rate",
        "revision",
    ):
        item = read_value(
            value,
            name,
            default=None,
        )

        if item is not None:
            metadata[name] = item

    current_weight = _number(
        read_value(
            value,
            "current_weight",
            "weight",
            default=1.0,
        ),
        field_name="current_weight",
    )

    normalized_weight = _number(
        read_value(
            value,
            "normalized_weight",
            default=0.0,
        ),
        field_name="normalized_weight",
    )

    normalized_weight = max(
        0.0,
        min(1.0, normalized_weight),
    )

    return StrategyWeight(
        strategy_type=_required_text(
            read_value(value, "strategy_type"),
            field_name="strategy_type",
        ),
        strategy_name=_required_text(
            read_value(value, "strategy_name"),
            field_name="strategy_name",
        ),
        current_weight=max(
            0.0,
            current_weight,
        ),
        normalized_weight=normalized_weight,
        confidence=max(
            0.0,
            min(
                1.0,
                _number(
                    read_value(
                        value,
                        "confidence",
                        default=0.0,
                    ),
                    field_name="confidence",
                ),
            ),
        ),
        stability=max(
            0.0,
            min(
                1.0,
                _number(
                    read_value(
                        value,
                        "stability",
                        default=0.0,
                    ),
                    field_name="stability",
                ),
            ),
        ),
        trend=_required_text(
            read_value(
                value,
                "trend",
                default="UNKNOWN",
            ),
            field_name="trend",
        ),
        sample_count=max(
            0,
            _integer(
                read_value(
                    value,
                    "sample_count",
                    default=0,
                ),
                field_name="sample_count",
            ),
        ),
        metadata=metadata,
    )


def weights_from_rankings(
    rankings: Sequence[
        StrategyRankingSnapshot
    ],
) -> tuple[StrategyWeight, ...]:
    """Build deterministic fallback weights from M6 rankings."""

    grouped: dict[
        str,
        list[StrategyRankingSnapshot],
    ] = {}

    for ranking in rankings:
        grouped.setdefault(
            ranking.strategy_type,
            [],
        ).append(ranking)

    result: list[StrategyWeight] = []

    for strategy_type, items in grouped.items():
        positive_scores = [
            max(0.0, item.rank_score)
            for item in items
        ]
        total = sum(positive_scores)

        if total <= 0.0:
            normalized = [
                1.0 / len(items)
                for _ in items
            ]
        else:
            normalized = [
                score / total
                for score in positive_scores
            ]

        for item, normalized_weight in zip(
            items,
            normalized,
        ):
            result.append(
                StrategyWeight(
                    strategy_type=strategy_type,
                    strategy_name=(
                        item.strategy_name
                    ),
                    current_weight=max(
                        0.0,
                        item.rank_score,
                    ),
                    normalized_weight=(
                        normalized_weight
                    ),
                    confidence=item.confidence,
                    stability=item.stability,
                    trend=item.trend,
                    sample_count=(
                        item.sample_count
                    ),
                    metadata={
                        "source": (
                            "ranking_fallback"
                        ),
                        "rank_position": (
                            item.rank_position
                        ),
                        "rank_score": (
                            item.rank_score
                        ),
                        "recent_gain": (
                            item.recent_gain
                        ),
                    },
                )
            )

    return normalize_strategy_weights(
        result
    )


def statistics_from_m6(
    values: Iterable[object],
) -> tuple[
    StrategyStatisticSnapshot,
    ...,
]:
    return ordered_statistics(
        tuple(
            statistic_from_m6(value)
            for value in values
        )
    )


def rankings_from_m6(
    values: Iterable[object],
) -> tuple[
    StrategyRankingSnapshot,
    ...,
]:
    return ordered_rankings(
        tuple(
            ranking_from_m6(value)
            for value in values
        )
    )


def weights_from_m6(
    values: Iterable[object],
) -> tuple[StrategyWeight, ...]:
    return normalize_strategy_weights(
        tuple(
            weight_from_m6(value)
            for value in values
        )
    )
