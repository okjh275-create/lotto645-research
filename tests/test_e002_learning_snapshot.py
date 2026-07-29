"""Regression test for Project E E-002 learning snapshots."""

from __future__ import annotations

from dataclasses import dataclass

from lrp.ensemble import (
    EnsembleConfig,
    EnsembleEngine,
    LearningSnapshotWeightRepository,
    M6LearningSnapshotRepository,
)


@dataclass(frozen=True, slots=True)
class StatisticFixture:
    strategy_type: str
    strategy_name: str
    sample_count: int
    average_match_count: float
    average_prediction_score: float
    hit3_plus_rate: float
    prize_rate: float
    updated_at_kst: str
    hit3_count: int = 0
    hit4_count: int = 0
    hit5_count: int = 0
    hit6_count: int = 0
    prize_count: int = 0


@dataclass(frozen=True, slots=True)
class RankingFixture:
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
    rolling_matches: dict[int, float]
    rolling_prize_rates: dict[int, float]


@dataclass(frozen=True, slots=True)
class WeightFixture:
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
    revision: tuple[int, int]


@dataclass(frozen=True, slots=True)
class CandidateFixture:
    name: str
    normalized_score: float


class RankingRepositoryFixture:
    def __init__(self) -> None:
        self.revision = (8, 8)

    def repository_revision(
        self,
    ) -> tuple[int, int]:
        return self.revision


class ServiceFixture:
    def __init__(self) -> None:
        self.ranking_repository = (
            RankingRepositoryFixture()
        )

    def get_strategy_statistics(
        self,
        *,
        strategy_type: str,
    ) -> tuple[StatisticFixture, ...]:
        if strategy_type == "model":
            return (
                StatisticFixture(
                    strategy_type="model",
                    strategy_name="GPT-v3.3",
                    sample_count=2,
                    average_match_count=5.0,
                    average_prediction_score=0.90,
                    hit3_plus_rate=1.0,
                    prize_rate=1.0,
                    updated_at_kst=(
                        "2026-07-25T21:02:00+09:00"
                    ),
                    hit6_count=1,
                    prize_count=2,
                ),
                StatisticFixture(
                    strategy_type="model",
                    strategy_name="Gemini-v7.1",
                    sample_count=2,
                    average_match_count=2.5,
                    average_prediction_score=0.83,
                    hit3_plus_rate=0.5,
                    prize_rate=0.5,
                    updated_at_kst=(
                        "2026-07-25T21:02:00+09:00"
                    ),
                    hit4_count=1,
                    prize_count=1,
                ),
            )

        return (
            StatisticFixture(
                strategy_type="scenario",
                strategy_name="gap",
                sample_count=2,
                average_match_count=5.0,
                average_prediction_score=0.90,
                hit3_plus_rate=1.0,
                prize_rate=1.0,
                updated_at_kst=(
                    "2026-07-25T21:02:00+09:00"
                ),
                hit6_count=1,
                prize_count=2,
            ),
            StatisticFixture(
                strategy_type="scenario",
                strategy_name="pair",
                sample_count=2,
                average_match_count=2.5,
                average_prediction_score=0.83,
                hit3_plus_rate=0.5,
                prize_rate=0.5,
                updated_at_kst=(
                    "2026-07-25T21:02:00+09:00"
                ),
                hit4_count=1,
                prize_count=1,
            ),
        )

    def rank_strategies(
        self,
        *,
        strategy_type: str,
        history_limit: int,
    ) -> tuple[RankingFixture, ...]:
        assert history_limit == 100

        names = (
            ("GPT-v3.3", "Gemini-v7.1")
            if strategy_type == "model"
            else ("gap", "pair")
        )

        return (
            RankingFixture(
                strategy_type=strategy_type,
                strategy_name=names[0],
                rank_position=1,
                rank_score=0.708183,
                confidence=0.095163,
                stability=0.666667,
                trend="FLAT",
                recent_gain=0.0,
                sample_count=2,
                average_match_count=5.0,
                average_prediction_score=0.90,
                prize_rate=1.0,
                rolling_matches={
                    10: 5.0,
                    20: 5.0,
                    50: 5.0,
                    100: 5.0,
                },
                rolling_prize_rates={
                    10: 1.0,
                    20: 1.0,
                    50: 1.0,
                    100: 1.0,
                },
            ),
            RankingFixture(
                strategy_type=strategy_type,
                strategy_name=names[1],
                rank_position=2,
                rank_score=0.48425,
                confidence=0.095163,
                stability=0.833333,
                trend="FLAT",
                recent_gain=0.0,
                sample_count=2,
                average_match_count=2.5,
                average_prediction_score=0.83,
                prize_rate=0.5,
                rolling_matches={
                    10: 2.5,
                    20: 2.5,
                    50: 2.5,
                    100: 2.5,
                },
                rolling_prize_rates={
                    10: 0.5,
                    20: 0.5,
                    50: 0.5,
                    100: 0.5,
                },
            ),
        )

    def adaptive_weights(
        self,
        *,
        strategy_type: str,
        history_limit: int,
    ) -> tuple[WeightFixture, ...]:
        assert history_limit == 100

        names = (
            ("GPT-v3.3", "Gemini-v7.1")
            if strategy_type == "model"
            else ("gap", "pair")
        )

        return (
            WeightFixture(
                strategy_type=strategy_type,
                strategy_name=names[0],
                rank_position=1,
                rank_score=0.758477,
                target_weight=1.042279,
                previous_weight=1.0,
                current_weight=1.008456,
                normalized_weight=0.512439,
                confidence=0.048771,
                stability=0.5,
                trend="FLAT",
                sample_count=1,
                revision=(8, 8),
            ),
            WeightFixture(
                strategy_type=strategy_type,
                strategy_name=names[1],
                rank_position=2,
                rank_score=0.350477,
                target_weight=0.797479,
                previous_weight=1.0,
                current_weight=0.959496,
                normalized_weight=0.487561,
                confidence=0.048771,
                stability=0.5,
                trend="FLAT",
                sample_count=1,
                revision=(8, 8),
            ),
        )


class RankingOnlyServiceFixture(
    ServiceFixture
):
    adaptive_weights = None


def test_adaptive_snapshot() -> None:
    repository = M6LearningSnapshotRepository(
        service=ServiceFixture(),
        history_limit=100,
    )

    first = repository.load_snapshot(
        round_no=1220
    )
    second = repository.load_snapshot(
        round_no=1220
    )

    assert first is second
    assert first.round_no == 1220
    assert first.revision == (8, 8)
    assert first.strategy_count == 4
    assert len(first.statistics) == 4
    assert len(first.rankings) == 4
    assert len(first.strategy_weights) == 4

    assert (
        first.metadata["weight_source"]
        == "m6_adaptive"
    )

    gpt = first.weight(
        "model",
        "GPT-v3.3",
    )

    assert gpt is not None
    assert gpt.current_weight == 1.008456
    assert gpt.normalized_weight == 0.512439
    assert gpt.metadata["revision"] == (8, 8)

    gap = first.ranking(
        "scenario",
        "gap",
    )

    assert gap is not None
    assert gap.rank_position == 1
    assert gap.rolling_matches[10] == 5.0

    payload = first.to_dict()

    assert payload["strategy_count"] == 4
    assert payload["revision"] == [8, 8]
    assert len(payload["statistics"]) == 4
    assert len(payload["rankings"]) == 4
    assert len(
        payload["strategy_weights"]
    ) == 4


def test_ranking_fallback() -> None:
    repository = M6LearningSnapshotRepository(
        service=RankingOnlyServiceFixture(),
    )

    snapshot = repository.load_snapshot(
        round_no=1220
    )

    assert (
        snapshot.metadata["weight_source"]
        == "m6_ranking_fallback"
    )

    model_weights = [
        item
        for item in snapshot.strategy_weights
        if item.strategy_type == "model"
    ]

    assert len(model_weights) == 2

    assert abs(
        sum(
            item.normalized_weight
            for item in model_weights
        )
        - 1.0
    ) < 1e-12

    assert (
        model_weights[0].metadata["source"]
        == "ranking_fallback"
    )


def test_ensemble_connection() -> None:
    snapshot_repository = (
        M6LearningSnapshotRepository(
            service=ServiceFixture()
        )
    )

    weight_repository = (
        LearningSnapshotWeightRepository(
            snapshot_repository
        )
    )

    engine = EnsembleEngine(
        repository=weight_repository
    )

    result = engine.evaluate(
        (
            CandidateFixture("A", 0.91),
            CandidateFixture("B", 0.82),
            CandidateFixture("C", 0.73),
        ),
        round_no=1220,
        config=EnsembleConfig(
            base_score_weight=0.75,
            adaptive_weight=0.10,
            confidence_weight=0.05,
            stability_weight=0.05,
            trend_weight=0.05,
        ),
    )

    assert result.count == 3
    assert len(result.strategy_weights) == 4

    assert [
        item.source.name
        for item in result.items
    ] == [
        "A",
        "B",
        "C",
    ]

    assert (
        result.items[0].ensemble_score
        > result.items[1].ensemble_score
        > result.items[2].ensemble_score
    )


def main() -> None:
    test_adaptive_snapshot()
    test_ranking_fallback()
    test_ensemble_connection()

    print(
        "PASS: Project E E-002 learning snapshot"
    )
    print("m6_statistics_adapter: PASS")
    print("m6_ranking_adapter: PASS")
    print("m6_adaptive_adapter: PASS")
    print("ranking_fallback: PASS")
    print("snapshot_cache: PASS")
    print("ensemble_connection: PASS")


if __name__ == "__main__":
    main()
