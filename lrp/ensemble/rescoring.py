"""Explainable candidate rescoring for Project E."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping, Sequence

from lrp.contracts import ContractError

from .features import (
    StrategyFeatureVector,
    build_feature_catalog,
)
from .snapshot import LearningSnapshot


StrategyKey = tuple[str, str]
StrategyResolver = Callable[
    [object],
    Sequence[StrategyKey],
]
BaseScoreReader = Callable[[object], float]


def _read(
    value: object,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)

    return getattr(value, name, default)


def default_base_score_reader(
    candidate: object,
) -> float:
    for name in (
        "normalized_score",
        "score",
        "base_score",
    ):
        value = _read(
            candidate,
            name,
            None,
        )

        if value is not None:
            try:
                result = float(value)
            except (TypeError, ValueError):
                continue

            if math.isfinite(result):
                return result

    raise ContractError(
        "candidate does not expose a finite score"
    )


def default_strategy_resolver(
    candidate: object,
) -> tuple[StrategyKey, ...]:
    """Resolve optional model/scenario evidence from a candidate."""

    keys: list[StrategyKey] = []

    strategy_type = _read(
        candidate,
        "strategy_type",
        None,
    )
    strategy_name = _read(
        candidate,
        "strategy_name",
        None,
    )

    if strategy_type and strategy_name:
        keys.append(
            (
                str(strategy_type).strip().lower(),
                str(strategy_name).strip(),
            )
        )

    model_name = _read(
        candidate,
        "model_name",
        None,
    )

    if model_name:
        keys.append(
            (
                "model",
                str(model_name).strip(),
            )
        )

    scenario_name = _read(
        candidate,
        "scenario_name",
        None,
    )

    if scenario_name:
        keys.append(
            (
                "scenario",
                str(scenario_name).strip(),
            )
        )

    scenario_names = _read(
        candidate,
        "scenario_names",
        (),
    )

    if scenario_names:
        for name in scenario_names:
            keys.append(
                (
                    "scenario",
                    str(name).strip(),
                )
            )

    features = _read(
        candidate,
        "features",
        None,
    )

    if isinstance(features, Mapping):
        feature_model = features.get(
            "model_name"
        )

        if feature_model:
            keys.append(
                (
                    "model",
                    str(feature_model).strip(),
                )
            )

        feature_scenarios = features.get(
            "scenario_names",
            (),
        )

        for name in feature_scenarios or ():
            keys.append(
                (
                    "scenario",
                    str(name).strip(),
                )
            )

    return tuple(
        dict.fromkeys(
            key
            for key in keys
            if key[0] and key[1]
        )
    )


@dataclass(frozen=True, slots=True)
class RescoringConfig:
    """Weights for E-003 candidate rescoring."""

    base_score_weight: float = 0.75
    adaptive_weight: float = 0.07
    ranking_weight: float = 0.05
    confidence_weight: float = 0.03
    stability_weight: float = 0.03
    trend_weight: float = 0.02
    performance_weight: float = 0.04
    sample_weight: float = 0.01

    preserve_without_evidence: bool = True

    def __post_init__(self) -> None:
        values = (
            self.base_score_weight,
            self.adaptive_weight,
            self.ranking_weight,
            self.confidence_weight,
            self.stability_weight,
            self.trend_weight,
            self.performance_weight,
            self.sample_weight,
        )

        if any(
            isinstance(value, bool)
            or not isinstance(
                value,
                (int, float),
            )
            or not math.isfinite(
                float(value)
            )
            or float(value) < 0.0
            for value in values
        ):
            raise ContractError(
                "rescoring weights must be "
                "finite non-negative numbers"
            )

        total = sum(
            float(value)
            for value in values
        )

        if abs(total - 1.0) > 1e-9:
            raise ContractError(
                "rescoring weights must sum to 1"
            )


@dataclass(frozen=True, slots=True)
class ScoreContribution:
    base: float
    adaptive: float
    ranking: float
    confidence: float
    stability: float
    trend: float
    performance: float
    sample: float

    @property
    def total(self) -> float:
        return (
            self.base
            + self.adaptive
            + self.ranking
            + self.confidence
            + self.stability
            + self.trend
            + self.performance
            + self.sample
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "base": self.base,
            "adaptive": self.adaptive,
            "ranking": self.ranking,
            "confidence": self.confidence,
            "stability": self.stability,
            "trend": self.trend,
            "performance": self.performance,
            "sample": self.sample,
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class RescoredCandidate:
    source: object
    base_score: float
    ensemble_score: float
    rank_before: int
    rank_after: int
    strategy_keys: tuple[StrategyKey, ...]
    contributions: ScoreContribution
    feature_vectors: tuple[
        StrategyFeatureVector,
        ...,
    ]

    @property
    def rank_change(self) -> int:
        return self.rank_before - self.rank_after

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_score": self.base_score,
            "ensemble_score": self.ensemble_score,
            "rank_before": self.rank_before,
            "rank_after": self.rank_after,
            "rank_change": self.rank_change,
            "strategy_keys": [
                list(key)
                for key in self.strategy_keys
            ],
            "contributions": (
                self.contributions.to_dict()
            ),
            "feature_vectors": [
                vector.to_dict()
                for vector in self.feature_vectors
            ],
        }


@dataclass(frozen=True, slots=True)
class RescoringResult:
    round_no: int
    items: tuple[RescoredCandidate, ...]
    snapshot_revision: tuple[int, int]
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    @property
    def count(self) -> int:
        return len(self.items)

    @property
    def changed_rank_count(self) -> int:
        return sum(
            item.rank_change != 0
            for item in self.items
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_no": self.round_no,
            "snapshot_revision": list(
                self.snapshot_revision
            ),
            "count": self.count,
            "changed_rank_count": (
                self.changed_rank_count
            ),
            "items": [
                item.to_dict()
                for item in self.items
            ],
            "metadata": dict(self.metadata),
        }


def _average(
    vectors: Sequence[
        StrategyFeatureVector
    ],
    field_name: str,
) -> float:
    if not vectors:
        return 0.0

    return sum(
        float(
            getattr(vector, field_name)
        )
        for vector in vectors
    ) / len(vectors)


def _performance_score(
    vectors: Sequence[
        StrategyFeatureVector
    ],
) -> float:
    if not vectors:
        return 0.0

    return sum(
        (
            vector.average_match_score
            + vector.average_prediction_score
            + vector.prize_rate
        )
        / 3.0
        for vector in vectors
    ) / len(vectors)


class CandidateRescorer:
    """Apply candidate-specific M6 learning evidence."""

    def __init__(
        self,
        *,
        config: RescoringConfig | None = None,
        score_reader: BaseScoreReader = (
            default_base_score_reader
        ),
        strategy_resolver: StrategyResolver = (
            default_strategy_resolver
        ),
    ) -> None:
        self.config = (
            config
            if config is not None
            else RescoringConfig()
        )
        self.score_reader = score_reader
        self.strategy_resolver = (
            strategy_resolver
        )

    def evaluate(
        self,
        candidates: Sequence[object],
        *,
        snapshot: LearningSnapshot,
    ) -> RescoringResult:
        catalog = build_feature_catalog(
            snapshot
        )

        base_rows = [
            (
                index,
                candidate,
                max(
                    0.0,
                    min(
                        1.0,
                        float(
                            self.score_reader(
                                candidate
                            )
                        ),
                    ),
                ),
            )
            for index, candidate
            in enumerate(candidates)
        ]

        base_rows.sort(
            key=lambda row: (
                -row[2],
                row[0],
            )
        )

        base_rank = {
            original_index: rank
            for rank, (
                original_index,
                _,
                _,
            ) in enumerate(
                base_rows,
                start=1,
            )
        }

        pending: list[
            tuple[
                int,
                object,
                float,
                tuple[StrategyKey, ...],
                tuple[
                    StrategyFeatureVector,
                    ...,
                ],
                ScoreContribution,
                float,
            ]
        ] = []

        for (
            original_index,
            candidate,
            base_score,
        ) in base_rows:
            keys = tuple(
                self.strategy_resolver(
                    candidate
                )
            )

            vectors = tuple(
                catalog[key]
                for key in keys
                if key in catalog
            )

            if (
                not vectors
                and self.config
                .preserve_without_evidence
            ):
                contributions = (
                    ScoreContribution(
                        base=base_score,
                        adaptive=0.0,
                        ranking=0.0,
                        confidence=0.0,
                        stability=0.0,
                        trend=0.0,
                        performance=0.0,
                        sample=0.0,
                    )
                )
                final_score = base_score
            else:
                contributions = (
                    ScoreContribution(
                        base=(
                            base_score
                            * self.config
                            .base_score_weight
                        ),
                        adaptive=(
                            _average(
                                vectors,
                                "adaptive_weight",
                            )
                            * self.config
                            .adaptive_weight
                        ),
                        ranking=(
                            _average(
                                vectors,
                                "rank_score",
                            )
                            * self.config
                            .ranking_weight
                        ),
                        confidence=(
                            _average(
                                vectors,
                                "confidence",
                            )
                            * self.config
                            .confidence_weight
                        ),
                        stability=(
                            _average(
                                vectors,
                                "stability",
                            )
                            * self.config
                            .stability_weight
                        ),
                        trend=(
                            _average(
                                vectors,
                                "trend_score",
                            )
                            * self.config
                            .trend_weight
                        ),
                        performance=(
                            _performance_score(
                                vectors
                            )
                            * self.config
                            .performance_weight
                        ),
                        sample=(
                            _average(
                                vectors,
                                "sample_confidence",
                            )
                            * self.config
                            .sample_weight
                        ),
                    )
                )

                final_score = max(
                    0.0,
                    min(
                        1.0,
                        contributions.total,
                    ),
                )

            pending.append(
                (
                    original_index,
                    candidate,
                    base_score,
                    keys,
                    vectors,
                    contributions,
                    final_score,
                )
            )

        pending.sort(
            key=lambda row: (
                -row[6],
                -row[2],
                row[0],
            )
        )

        items = tuple(
            RescoredCandidate(
                source=candidate,
                base_score=base_score,
                ensemble_score=final_score,
                rank_before=base_rank[
                    original_index
                ],
                rank_after=rank_after,
                strategy_keys=keys,
                contributions=contributions,
                feature_vectors=vectors,
            )
            for rank_after, (
                original_index,
                candidate,
                base_score,
                keys,
                vectors,
                contributions,
                final_score,
            ) in enumerate(
                pending,
                start=1,
            )
        )

        return RescoringResult(
            round_no=snapshot.round_no,
            items=items,
            snapshot_revision=(
                snapshot.revision
            ),
            metadata={
                "feature_count": len(catalog),
                "candidate_count": len(
                    candidates
                ),
                "evidence_candidate_count": (
                    sum(
                        bool(
                            item.feature_vectors
                        )
                        for item in items
                    )
                ),
                "preserve_without_evidence": (
                    self.config
                    .preserve_without_evidence
                ),
            },
        )
