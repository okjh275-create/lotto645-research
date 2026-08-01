from __future__ import annotations

import pytest

from lrp.evolution.integration import (
    NoOpEvolutionWeightAdapter,
)


def test_returns_same_object() -> None:
    adapter = NoOpEvolutionWeightAdapter()
    probability_vector = object()

    result = adapter.adjust(
        probability_vector,
        round_no=1220,
        seed=20260802,
    )

    assert result is probability_vector


def test_preserves_mapping_identity() -> None:
    adapter = NoOpEvolutionWeightAdapter()
    probability_vector = {
        1: 0.1,
        2: 0.2,
    }

    result = adapter.adjust(
        probability_vector,
        round_no=1220,
        seed=20260802,
    )

    assert result is probability_vector
    assert result == {
        1: 0.1,
        2: 0.2,
    }


def test_preserves_immutable_object_identity() -> None:
    adapter = NoOpEvolutionWeightAdapter()
    probability_vector = (
        (1, 0.1),
        (2, 0.2),
    )

    result = adapter.adjust(
        probability_vector,
        round_no=1220,
        seed=20260802,
    )

    assert result is probability_vector


@pytest.mark.parametrize(
    "round_no",
    [0, -1],
)
def test_invalid_round_number_is_rejected(
    round_no: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="greater than or equal to 1",
    ):
        NoOpEvolutionWeightAdapter().adjust(
            object(),
            round_no=round_no,
            seed=1,
        )


@pytest.mark.parametrize(
    "round_no",
    [True, 1.5, "1220"],
)
def test_invalid_round_number_type_is_rejected(
    round_no: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="round_no must be an integer",
    ):
        NoOpEvolutionWeightAdapter().adjust(
            object(),
            round_no=round_no,  # type: ignore[arg-type]
            seed=1,
        )


@pytest.mark.parametrize(
    "seed",
    [True, 1.5, "1"],
)
def test_invalid_seed_is_rejected(
    seed: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="seed must be an integer",
    ):
        NoOpEvolutionWeightAdapter().adjust(
            object(),
            round_no=1220,
            seed=seed,  # type: ignore[arg-type]
        )


def test_none_probability_vector_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="must not be None",
    ):
        NoOpEvolutionWeightAdapter().adjust(
            None,
            round_no=1220,
            seed=1,
        )
