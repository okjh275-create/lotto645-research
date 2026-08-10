from __future__ import annotations

import math

import pytest

from lrp.prediction.probability import (
    NumberProbability,
    ProbabilityVector,
)
from lrp.regimes.integration.active_adjustment import (
    ProbabilityVectorAdjuster,
)


def make_vector() -> ProbabilityVector:
    raw_scores = {
        number: float(number)
        for number in range(1, 46)
    }

    total = sum(raw_scores.values())

    ranked_numbers = sorted(
        raw_scores,
        key=lambda number: (
            -raw_scores[number],
            number,
        ),
    )

    rank_by_number = {
        number: rank
        for rank, number
        in enumerate(ranked_numbers, start=1)
    }

    probabilities = tuple(
        NumberProbability(
            number=number,
            probability=raw_scores[number] / total,
            raw_score=raw_scores[number],
            rank=rank_by_number[number],
            components={
                "hot": number / 45.0,
            },
            metadata={
                "source": "test",
            },
        )
        for number in range(1, 46)
    )

    return ProbabilityVector(
        round_no=1220,
        generated_at_kst="2026-08-09T21:45:00+09:00",
        probabilities=probabilities,
        metadata={
            "source": "unit-test",
        },
    )


def test_identity_adjustment_preserves_scores_and_ranks() -> None:
    vector = make_vector()

    adjusted = ProbabilityVectorAdjuster.adjust(
        vector,
        multipliers={},
    )

    for original, result in zip(
        vector.probabilities,
        adjusted.probabilities,
        strict=True,
    ):
        assert result.number == original.number
        assert result.raw_score == original.raw_score
        assert result.probability == original.probability
        assert result.rank == original.rank
        assert result.components == original.components
        assert result.metadata == original.metadata


def test_adjustment_renormalizes_probability_sum() -> None:
    vector = make_vector()

    adjusted = ProbabilityVectorAdjuster.adjust(
        vector,
        multipliers={45: 2.0},
    )

    total = sum(
        item.probability
        for item in adjusted.probabilities
    )

    assert math.isclose(
        total,
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )

    for item in adjusted.probabilities:
        assert 0.0 <= item.probability <= 1.0


def test_adjustment_recalculates_rank() -> None:
    vector = make_vector()

    adjusted = ProbabilityVectorAdjuster.adjust(
        vector,
        multipliers={1: 100.0},
    )

    assert adjusted.get(1).rank == 1
    assert adjusted.top(1)[0].number == 1


def test_adjustment_preserves_number_order() -> None:
    adjusted = ProbabilityVectorAdjuster.adjust(
        make_vector(),
        multipliers={1: 100.0, 45: 0.5},
    )

    assert tuple(
        item.number
        for item in adjusted.probabilities
    ) == tuple(range(1, 46))


def test_adjustment_preserves_vector_metadata() -> None:
    vector = make_vector()

    adjusted = ProbabilityVectorAdjuster.adjust(
        vector,
        multipliers={45: 1.02},
    )

    assert adjusted.metadata["source"] == "unit-test"
    assert (
        adjusted.metadata["global_regime_adjusted"]
        is True
    )
    assert adjusted.round_no == vector.round_no
    assert (
        adjusted.generated_at_kst
        == vector.generated_at_kst
    )


def test_negative_multiplier_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        ProbabilityVectorAdjuster.adjust(
            make_vector(),
            multipliers={1: -0.01},
        )


def test_invalid_vector_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="ProbabilityVector",
    ):
        ProbabilityVectorAdjuster.adjust(
            object(),
            multipliers={},
        )
