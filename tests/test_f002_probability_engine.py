"""Tests for Project F-002 probability fusion engine."""

from __future__ import annotations

from dataclasses import dataclass
import math

from lrp.prediction import (
    ProbabilityFusionConfig,
    ProbabilityFusionEngine,
    RegimeDetector,
)


@dataclass(frozen=True)
class SampleFeature:
    number: int
    freq_all: int
    freq10: int
    freq20: int
    freq50: int
    gap: int


def _build_features() -> list[SampleFeature]:
    return [
        SampleFeature(
            number=number,
            freq_all=30 + number,
            freq10=number % 4,
            freq20=(number * 2) % 7,
            freq50=(number * 3) % 12,
            gap=number % 16,
        )
        for number in range(1, 46)
    ]


def _build_profile():
    return RegimeDetector().detect(
        _build_features(),
        round_no=1220,
        generated_at_kst="2026-07-31T18:45:00+09:00",
    )


def test_probability_engine_builds_complete_vector() -> None:
    vector = ProbabilityFusionEngine().build(
        _build_profile(),
        metadata={"source": "unit-test"},
    )

    assert vector.round_no == 1220
    assert len(vector.probabilities) == 45
    assert vector.probabilities[0].number == 1
    assert vector.probabilities[-1].number == 45
    assert vector.metadata["engine"] == "F-002"
    assert vector.metadata["source"] == "unit-test"


def test_probabilities_sum_to_one() -> None:
    vector = ProbabilityFusionEngine().build(
        _build_profile()
    )

    total = sum(
        item.probability
        for item in vector.probabilities
    )

    assert math.isclose(
        total,
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )

    for item in vector.probabilities:
        assert 0.0 <= item.probability <= 1.0
        assert item.raw_score >= 0.0


def test_probability_ranks_are_unique() -> None:
    vector = ProbabilityFusionEngine().build(
        _build_profile()
    )

    ranks = {
        item.rank
        for item in vector.probabilities
    }

    assert ranks == set(range(1, 46))
    assert vector.top(5)[0].rank == 1
    assert len(vector.top(5)) == 5


def test_external_scores_affect_ranking() -> None:
    profile = _build_profile()

    config = ProbabilityFusionConfig(
        hot_weight=0.10,
        cold_weight=0.05,
        gap_weight=0.05,
        trend_weight=0.05,
        transition_weight=0.05,
        learning_weight=0.35,
        adaptive_weight=0.35,
    )

    learning_scores = {
        number: 0.0
        for number in range(1, 46)
    }
    adaptive_scores = {
        number: 0.0
        for number in range(1, 46)
    }

    learning_scores[45] = 1.0
    adaptive_scores[45] = 1.0

    vector = ProbabilityFusionEngine(
        config
    ).build(
        profile,
        learning_scores=learning_scores,
        adaptive_scores=adaptive_scores,
    )

    assert vector.get(45).rank == 1


def test_same_input_produces_same_distribution() -> None:
    profile = _build_profile()
    engine = ProbabilityFusionEngine()

    first = engine.build(profile)
    second = engine.build(profile)

    assert [
        item.probability
        for item in first.probabilities
    ] == [
        item.probability
        for item in second.probabilities
    ]


def main() -> None:
    test_probability_engine_builds_complete_vector()
    test_probabilities_sum_to_one()
    test_probability_ranks_are_unique()
    test_external_scores_affect_ranking()
    test_same_input_produces_same_distribution()
    print("OK: F-002 probability engine tests")


if __name__ == "__main__":
    main()
