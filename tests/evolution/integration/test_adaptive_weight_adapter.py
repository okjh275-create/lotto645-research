from __future__ import annotations

from datetime import datetime, timezone
from math import isclose

import pytest

from lrp.contracts import ContractError
from lrp.evolution.contracts.models import (
    AdaptiveWeightProfile,
)
from lrp.evolution.integration import (
    AdaptiveEvolutionWeightAdapter,
)
from lrp.prediction.probability import (
    NumberProbability,
    ProbabilityVector,
)


FIXED_TIME = datetime(
    2026,
    8,
    2,
    0,
    0,
    tzinfo=timezone.utc,
)


def make_profile() -> AdaptiveWeightProfile:
    return AdaptiveWeightProfile(
        hot_weight=0.50,
        cold_weight=0.10,
        gap_weight=0.10,
        trend_weight=0.10,
        transition_weight=0.10,
        learning_weight=0.05,
        adaptive_weight=0.05,
        confidence=0.80,
        sample_size=40,
        revision=7,
        generated_at=FIXED_TIME,
    )


def make_vector() -> ProbabilityVector:
    raw_records: list[
        tuple[int, float, dict[str, float]]
    ] = []

    for number in range(1, 46):
        hot = number / 45.0
        cold = 1.0 - hot

        components = {
            "hot": hot,
            "cold": cold,
            "gap": 0.5,
            "trend": 0.5,
            "transition": 0.5,
            "learning": 0.5,
            "adaptive": 0.5,
        }

        raw_records.append(
            (
                number,
                1.0,
                components,
            )
        )

    probabilities = tuple(
        NumberProbability(
            number=number,
            probability=1.0 / 45.0,
            raw_score=raw_score,
            rank=number,
            components=components,
            metadata={
                "original": True,
            },
        )
        for number, raw_score, components
        in raw_records
    )

    return ProbabilityVector(
        round_no=1220,
        generated_at_kst=(
            "2026-08-02T09:00:00+09:00"
        ),
        probabilities=probabilities,
        metadata={
            "engine": "F-002",
        },
    )


def test_adapter_accepts_profile() -> None:
    profile = make_profile()

    adapter = AdaptiveEvolutionWeightAdapter(
        profile
    )

    assert adapter.profile is profile


def test_invalid_profile_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="AdaptiveWeightProfile",
    ):
        AdaptiveEvolutionWeightAdapter(
            object(),  # type: ignore[arg-type]
        )


def test_adjust_returns_new_vector() -> None:
    original = make_vector()

    adjusted = (
        AdaptiveEvolutionWeightAdapter(
            make_profile()
        ).adjust(
            original,
            round_no=1220,
            seed=20260802,
        )
    )

    assert adjusted is not original
    assert adjusted.probabilities is not (
        original.probabilities
    )


def test_adjust_preserves_original_vector() -> None:
    original = make_vector()
    original_payload = original.as_dict()

    AdaptiveEvolutionWeightAdapter(
        make_profile()
    ).adjust(
        original,
        round_no=1220,
        seed=20260802,
    )

    assert original.as_dict() == (
        original_payload
    )


def test_adjust_preserves_number_order() -> None:
    adjusted = (
        AdaptiveEvolutionWeightAdapter(
            make_profile()
        ).adjust(
            make_vector(),
            round_no=1220,
            seed=20260802,
        )
    )

    assert tuple(
        item.number
        for item in adjusted.probabilities
    ) == tuple(range(1, 46))


def test_adjusted_probabilities_sum_to_one() -> None:
    adjusted = (
        AdaptiveEvolutionWeightAdapter(
            make_profile()
        ).adjust(
            make_vector(),
            round_no=1220,
            seed=20260802,
        )
    )

    assert isclose(
        sum(
            item.probability
            for item in adjusted.probabilities
        ),
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def test_hot_weight_changes_ranking() -> None:
    adjusted = (
        AdaptiveEvolutionWeightAdapter(
            make_profile()
        ).adjust(
            make_vector(),
            round_no=1220,
            seed=20260802,
        )
    )

    assert adjusted.get(45).rank == 1
    assert adjusted.get(1).rank == 45
    assert (
        adjusted.get(45).probability
        > adjusted.get(1).probability
    )


def test_adjust_preserves_components() -> None:
    original = make_vector()

    adjusted = (
        AdaptiveEvolutionWeightAdapter(
            make_profile()
        ).adjust(
            original,
            round_no=1220,
            seed=20260802,
        )
    )

    assert adjusted.get(10).components == (
        original.get(10).components
    )


def test_adjust_adds_record_metadata() -> None:
    adjusted = (
        AdaptiveEvolutionWeightAdapter(
            make_profile()
        ).adjust(
            make_vector(),
            round_no=1220,
            seed=20260802,
        )
    )

    assert adjusted.get(1).metadata[
        "original"
    ] is True
    assert adjusted.get(1).metadata[
        "evolution_adjusted"
    ] is True
    assert adjusted.get(1).metadata[
        "evolution_revision"
    ] == 7


def test_adjust_adds_vector_metadata() -> None:
    adjusted = (
        AdaptiveEvolutionWeightAdapter(
            make_profile()
        ).adjust(
            make_vector(),
            round_no=1220,
            seed=20260802,
        )
    )

    assert adjusted.metadata["engine"] == (
        "F-002"
    )
    assert adjusted.metadata[
        "evolution_revision"
    ] == 7
    assert adjusted.metadata[
        "evolution_seed"
    ] == 20260802
    assert adjusted.metadata[
        "evolution_weights"
    ] == pytest.approx(
        make_profile()
        .to_probability_weights()
    )


def test_round_mismatch_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="round_no does not match",
    ):
        AdaptiveEvolutionWeightAdapter(
            make_profile()
        ).adjust(
            make_vector(),
            round_no=1221,
            seed=20260802,
        )


def test_invalid_vector_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="ProbabilityVector",
    ):
        AdaptiveEvolutionWeightAdapter(
            make_profile()
        ).adjust(
            object(),  # type: ignore[arg-type]
            round_no=1220,
            seed=20260802,
        )


def test_missing_component_is_rejected() -> None:
    vector = make_vector()
    items = list(vector.probabilities)
    first = items[0]

    items[0] = NumberProbability(
        number=first.number,
        probability=first.probability,
        raw_score=first.raw_score,
        rank=first.rank,
        components={
            key: value
            for key, value
            in first.components.items()
            if key != "adaptive"
        },
        metadata=first.metadata,
    )

    invalid_vector = ProbabilityVector(
        round_no=vector.round_no,
        generated_at_kst=(
            vector.generated_at_kst
        ),
        probabilities=tuple(items),
        metadata=vector.metadata,
    )

    with pytest.raises(
        ContractError,
        match="components are missing",
    ):
        AdaptiveEvolutionWeightAdapter(
            make_profile()
        ).adjust(
            invalid_vector,
            round_no=1220,
            seed=20260802,
        )
