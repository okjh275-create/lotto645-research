from __future__ import annotations

import pytest

from lrp.contracts import ContractError
from lrp.evolution.contracts.review_reward_vector import (
    ReviewRewardVector,
)


def make_vector(
    **overrides,
) -> ReviewRewardVector:
    values = {
        "portfolio_hit": 0.60,
        "practical_hit": 0.50,
        "rank_quality": 0.30,
        "coverage": 0.20,
        "diversity": 0.10,
        "stability": -0.10,
        "sample_size": 20,
        "metadata": {
            "round": 1231,
        },
    }
    values.update(overrides)

    return ReviewRewardVector(**values)


def test_reward_vector_normalizes_values() -> None:
    vector = make_vector()

    assert vector.portfolio_hit == 0.60
    assert vector.practical_hit == 0.50
    assert vector.sample_size == 20
    assert vector.metadata["round"] == 1231


def test_reward_vector_is_immutable() -> None:
    vector = make_vector()

    with pytest.raises(
        AttributeError,
    ):
        vector.portfolio_hit = 0.0  # type: ignore[misc]


def test_reward_vector_rejects_out_of_range() -> None:
    with pytest.raises(
        ContractError,
        match="between -1 and 1",
    ):
        make_vector(
            diversity=1.1,
        )


def test_reward_vector_neutral_factory() -> None:
    vector = ReviewRewardVector.neutral(
        sample_size=20,
    )

    assert vector.weighted_score() == 0.0
    assert vector.sample_size == 20


def test_weighted_score_uses_default_weights() -> None:
    vector = make_vector()

    expected = (
        0.60 * 0.30
        + 0.50 * 0.25
        + 0.30 * 0.15
        + 0.20 * 0.10
        + 0.10 * 0.10
        - 0.10 * 0.10
    )

    assert vector.weighted_score() == pytest.approx(
        expected
    )


def test_weighted_score_normalizes_custom_weights() -> None:
    vector = make_vector()

    score = vector.weighted_score(
        {
            "portfolio_hit": 1.0,
            "practical_hit": 1.0,
            "rank_quality": 0.0,
            "coverage": 0.0,
            "diversity": 0.0,
            "stability": 0.0,
        }
    )

    assert score == pytest.approx(0.55)


def test_invalid_sample_size_is_rejected() -> None:
    with pytest.raises(
        ContractError,
        match="sample_size",
    ):
        make_vector(
            sample_size=0,
        )


def test_as_dict_is_serializable_shape() -> None:
    payload = make_vector().as_dict()

    assert payload["portfolio_hit"] == 0.60
    assert payload["sample_size"] == 20
    assert payload["metadata"]["round"] == 1231
