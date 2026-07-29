"""Regression test for Project E E-003 feature rescoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lrp.ensemble import (
    CandidateRescorer,
    LearningSnapshot,
    RescoredCandidate,
    RescoringConfig,
    StrategyRankingSnapshot,
    StrategyStatisticSnapshot,
    StrategyWeight,
    build_feature_catalog,
    explain_result,
)


@dataclass(frozen=True, slots=True)
class CandidateFixture:
    """Minimal candidate fixture for E-003 regression tests."""

    name: str
    normalized_score: float
    model_name: str | None = None
    scenario_name: str | None = None


def build_snapshot() -> LearningSnapshot:
    """Build deterministic strong/weak learning evidence."""

    return LearningSnapshot(
        round_no=1220,
        revision=(8, 8),
        statistics=(
            StrategyStatisticSnapshot(
                strategy_type="model",
                strategy_name="strong",
                sample_count=20,
                average_match_count=5.0,
                average_prediction_score=0.92,
                hit3_plus_rate=1.0,
                prize_rate=1.0,
                updated_at_kst=(
                    "2026-07-27T18:00:00+09:00"
                ),
            ),
            StrategyStatisticSnapshot(
                strategy_type="model",
                strategy_name="weak",
                sample_count=20,
                average_match_count=1.0,
                average_prediction_score=0.40,
                hit3_plus_rate=0.0,
                prize_rate=0.0,
                updated_at_kst=(
                    "2026-07-27T18:00:00+09:00"
                ),
            ),
        ),
        rankings=(
            StrategyRankingSnapshot(
                strategy_type="model",
                strategy_name="strong",
                rank_position=1,
                rank_score=0.95,
                confidence=0.90,
                stability=0.90,
                trend="UP",
                recent_gain=0.10,
                sample_count=20,
                average_match_count=5.0,
                average_prediction_score=0.92,
                prize_rate=1.0,
            ),
            StrategyRankingSnapshot(
                strategy_type="model",
                strategy_name="weak",
                rank_position=2,
                rank_score=0.20,
                confidence=0.30,
                stability=0.40,
                trend="DOWN",
                recent_gain=-0.10,
                sample_count=20,
                average_match_count=1.0,
                average_prediction_score=0.40,
                prize_rate=0.0,
            ),
        ),
        strategy_weights=(
            StrategyWeight(
                strategy_type="model",
                strategy_name="strong",
                current_weight=1.10,
                normalized_weight=0.80,
                confidence=0.90,
                stability=0.90,
                trend="UP",
                sample_count=20,
            ),
            StrategyWeight(
                strategy_type="model",
                strategy_name="weak",
                current_weight=0.90,
                normalized_weight=0.20,
                confidence=0.30,
                stability=0.40,
                trend="DOWN",
                sample_count=20,
            ),
        ),
    )


def item_map(
    items: tuple[RescoredCandidate, ...],
) -> dict[str, RescoredCandidate]:
    """Index result items by fixture candidate name."""

    result = {
        item.source.name: item
        for item in items
    }

    assert len(result) == len(items)

    return result


def assert_unit_interval(
    value: float,
) -> None:
    assert 0.0 <= value <= 1.0


def test_feature_catalog() -> None:
    snapshot = build_snapshot()
    catalog = build_feature_catalog(snapshot)

    assert len(catalog) == 2
    assert set(catalog) == {
        ("model", "strong"),
        ("model", "weak"),
    }

    strong = catalog[
        ("model", "strong")
    ]
    weak = catalog[
        ("model", "weak")
    ]

    assert strong.adaptive_weight == 0.80
    assert strong.rank_score == 0.95
    assert strong.trend_score == 1.0
    assert strong.average_match_score == (
        5.0 / 6.0
    )
    assert strong.average_prediction_score == 0.92
    assert strong.prize_rate == 1.0
    assert strong.sample_confidence == 1.0
    assert strong.evidence_count == 3

    assert weak.adaptive_weight == 0.20
    assert weak.rank_score == 0.20
    assert weak.trend_score == 0.0
    assert weak.average_match_score == (
        1.0 / 6.0
    )
    assert weak.average_prediction_score == 0.40
    assert weak.prize_rate == 0.0
    assert weak.sample_confidence == 1.0
    assert weak.evidence_count == 3

    assert (
        strong.adaptive_weight
        > weak.adaptive_weight
    )
    assert strong.rank_score > weak.rank_score
    assert (
        strong.average_match_score
        > weak.average_match_score
    )


def test_preserve_without_evidence() -> None:
    candidates = (
        CandidateFixture(
            name="A",
            normalized_score=0.90,
        ),
        CandidateFixture(
            name="B",
            normalized_score=0.80,
        ),
    )

    result = CandidateRescorer().evaluate(
        candidates,
        snapshot=build_snapshot(),
    )

    items = item_map(result.items)

    assert result.count == 2
    assert result.changed_rank_count == 0

    assert items["A"].base_score == 0.90
    assert items["A"].ensemble_score == 0.90
    assert items["A"].rank_before == 1
    assert items["A"].rank_after == 1
    assert items["A"].rank_change == 0
    assert not items["A"].strategy_keys
    assert not items["A"].feature_vectors

    assert items["B"].base_score == 0.80
    assert items["B"].ensemble_score == 0.80
    assert items["B"].rank_before == 2
    assert items["B"].rank_after == 2
    assert items["B"].rank_change == 0
    assert not items["B"].strategy_keys
    assert not items["B"].feature_vectors

    assert [
        item.source.name
        for item in result.items
    ] == [
        "A",
        "B",
    ]

    assert (
        result.metadata[
            "evidence_candidate_count"
        ]
        == 0
    )
    assert (
        result.metadata[
            "preserve_without_evidence"
        ]
        is True
    )


def test_candidate_specific_rescoring() -> None:
    candidates = (
        CandidateFixture(
            name="weak-high-base",
            normalized_score=0.90,
            model_name="weak",
        ),
        CandidateFixture(
            name="strong-lower-base",
            normalized_score=0.86,
            model_name="strong",
        ),
        CandidateFixture(
            name="neutral",
            normalized_score=0.70,
        ),
    )

    config = RescoringConfig(
        base_score_weight=0.60,
        adaptive_weight=0.10,
        ranking_weight=0.08,
        confidence_weight=0.04,
        stability_weight=0.04,
        trend_weight=0.04,
        performance_weight=0.08,
        sample_weight=0.02,
    )

    result = CandidateRescorer(
        config=config
    ).evaluate(
        candidates,
        snapshot=build_snapshot(),
    )

    items = item_map(result.items)

    strong = items["strong-lower-base"]
    weak = items["weak-high-base"]
    neutral = items["neutral"]

    assert result.count == 3
    assert (
        result.metadata[
            "candidate_count"
        ]
        == 3
    )
    assert (
        result.metadata[
            "feature_count"
        ]
        == 2
    )
    assert (
        result.metadata[
            "evidence_candidate_count"
        ]
        == 2
    )

    # 기존 점수에서는 weak가 1위, strong이 2위였다.
    assert weak.rank_before == 1
    assert strong.rank_before == 2
    assert neutral.rank_before == 3

    # 강한 학습 증거가 있는 후보가 최종 1위로 상승한다.
    assert strong.rank_after == 1
    assert strong.rank_change == 1

    # 약한 학습 증거 후보는 strong보다 낮게 재평가된다.
    assert (
        strong.ensemble_score
        > weak.ensemble_score
    )

    # 근거 없는 후보는 기존 점수를 그대로 유지한다.
    assert neutral.base_score == 0.70
    assert neutral.ensemble_score == 0.70
    assert not neutral.strategy_keys
    assert not neutral.feature_vectors

    # 현재 설정에서는 neutral이 weak보다 높을 수 있다.
    # 특정 2·3위 순서를 강제하지 않고 점수 계약을 검증한다.
    assert (
        neutral.ensemble_score
        > weak.ensemble_score
    )

    assert strong.strategy_keys == (
        ("model", "strong"),
    )
    assert weak.strategy_keys == (
        ("model", "weak"),
    )

    assert len(strong.feature_vectors) == 1
    assert len(weak.feature_vectors) == 1

    assert (
        strong.feature_vectors[0]
        .strategy_name
        == "strong"
    )
    assert (
        weak.feature_vectors[0]
        .strategy_name
        == "weak"
    )

    assert_unit_interval(
        strong.ensemble_score
    )
    assert_unit_interval(
        weak.ensemble_score
    )
    assert_unit_interval(
        neutral.ensemble_score
    )

    # 기여도 합계가 최종 점수와 일치해야 한다.
    assert abs(
        strong.contributions.total
        - strong.ensemble_score
    ) < 1e-12

    assert abs(
        weak.contributions.total
        - weak.ensemble_score
    ) < 1e-12

    assert (
        strong.contributions.adaptive
        > weak.contributions.adaptive
    )
    assert (
        strong.contributions.ranking
        > weak.contributions.ranking
    )
    assert (
        strong.contributions.confidence
        > weak.contributions.confidence
    )
    assert (
        strong.contributions.performance
        > weak.contributions.performance
    )

    # 결과는 최종 점수 내림차순이어야 한다.
    scores = [
        item.ensemble_score
        for item in result.items
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    # strong과 weak의 순위가 모두 변경된다.
    assert result.changed_rank_count >= 2


def test_explainable_result() -> None:
    candidates = (
        CandidateFixture(
            name="weak-high-base",
            normalized_score=0.90,
            model_name="weak",
        ),
        CandidateFixture(
            name="strong-lower-base",
            normalized_score=0.86,
            model_name="strong",
        ),
        CandidateFixture(
            name="neutral",
            normalized_score=0.70,
        ),
    )

    result = CandidateRescorer(
        config=RescoringConfig(
            base_score_weight=0.60,
            adaptive_weight=0.10,
            ranking_weight=0.08,
            confidence_weight=0.04,
            stability_weight=0.04,
            trend_weight=0.04,
            performance_weight=0.08,
            sample_weight=0.02,
        )
    ).evaluate(
        candidates,
        snapshot=build_snapshot(),
    )

    explanation = explain_result(result)

    assert explanation["round_no"] == 1220
    assert explanation[
        "snapshot_revision"
    ] == [
        8,
        8,
    ]
    assert explanation[
        "candidate_count"
    ] == 3
    assert explanation[
        "changed_rank_count"
    ] == result.changed_rank_count
    assert len(explanation["items"]) == 3

    for item in explanation["items"]:
        assert set(item) == {
            "base_score",
            "ensemble_score",
            "rank_before",
            "rank_after",
            "rank_change",
            "strategy_evidence",
            "contributions",
        }

        contributions: dict[
            str,
            Any,
        ] = item["contributions"]

        assert set(contributions) == {
            "base",
            "adaptive",
            "ranking",
            "confidence",
            "stability",
            "trend",
            "performance",
            "sample",
            "total",
        }

        assert_unit_interval(
            float(
                item["ensemble_score"]
            )
        )


def main() -> None:
    test_feature_catalog()
    test_preserve_without_evidence()
    test_candidate_specific_rescoring()
    test_explainable_result()

    print(
        "PASS: Project E E-003 feature rescoring"
    )
    print("strategy_feature_vector: PASS")
    print("snapshot_feature_fusion: PASS")
    print("no_evidence_preservation: PASS")
    print("candidate_specific_rescoring: PASS")
    print("rank_change_tracking: PASS")
    print("explainable_breakdown: PASS")
    print("order_independent_assertions: PASS")


if __name__ == "__main__":
    main()
