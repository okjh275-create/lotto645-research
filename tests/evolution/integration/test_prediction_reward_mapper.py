from __future__ import annotations

import pytest

from lrp.evolution.contracts.reinforcement import (
    RewardFeedback,
)
from lrp.evolution.integration import (
    PredictionRewardMapper,
)


def make_review(
    *,
    best_hits: int = 4,
    practical_hits: int = 3,
    set_count: int = 10,
) -> dict[str, object]:
    return {
        "summary": {
            "set_count": set_count,
            "best_main_hits": best_hits,
            "best_set_ids": ["S1"],
            "hit_distribution": {
                "0": 2,
                "1": 3,
                "2": 3,
                "3": 1,
                "4": 1,
            },
            "practical_best_hits": (
                practical_hits
            ),
            "winning_rank_counts": {
                "1": 0,
                "2": 0,
                "3": 0,
                "4": 1,
                "5": 1,
                "none": 8,
            },
        }
    }


def test_mapper_creates_two_feedbacks() -> None:
    feedbacks = PredictionRewardMapper().map(
        make_review()
    )

    assert len(feedbacks) == 2
    assert all(
        isinstance(item, RewardFeedback)
        for item in feedbacks
    )


def test_portfolio_feedback() -> None:
    feedback = PredictionRewardMapper().map(
        make_review()
    )[0]

    assert feedback.source == (
        "prediction_review"
    )
    assert feedback.arm == (
        "portfolio_top_k"
    )
    assert feedback.policy is None
    assert feedback.reward == pytest.approx(
        0.55
    )
    assert feedback.observation_count == 10


def test_practical_feedback() -> None:
    feedback = PredictionRewardMapper().map(
        make_review()
    )[1]

    assert feedback.arm == (
        "practical_top5"
    )
    assert feedback.reward == pytest.approx(
        0.20
    )
    assert feedback.observation_count == 1


def test_policy_is_applied_to_both_feedbacks() -> None:
    feedbacks = PredictionRewardMapper().map(
        make_review(),
        policy=" thompson ",
    )

    assert tuple(
        item.policy
        for item in feedbacks
    ) == (
        "thompson",
        "thompson",
    )


@pytest.mark.parametrize(
    ("hits", "expected"),
    [
        (0, -1.00),
        (1, -0.70),
        (2, -0.30),
        (3, 0.20),
        (4, 0.55),
        (5, 0.85),
        (6, 1.00),
    ],
)
def test_reward_scale(
    hits: int,
    expected: float,
) -> None:
    assert (
        PredictionRewardMapper
        .reward_for_hits(hits)
        == pytest.approx(expected)
    )


def test_flat_summary_is_supported() -> None:
    payload = {
        "set_count": 10,
        "best_main_hits": 2,
        "practical_best_hits": 1,
    }

    feedbacks = PredictionRewardMapper().map(
        payload
    )

    assert feedbacks[0].reward == pytest.approx(
        -0.30
    )
    assert feedbacks[1].reward == pytest.approx(
        -0.70
    )


def test_invalid_payload_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="must be a mapping",
    ):
        PredictionRewardMapper().map(
            object(),  # type: ignore[arg-type]
        )


def test_missing_required_field_is_rejected() -> None:
    payload = make_review()
    summary = payload["summary"]

    assert isinstance(summary, dict)
    del summary["best_main_hits"]

    with pytest.raises(
        ValueError,
        match="missing review field",
    ):
        PredictionRewardMapper().map(payload)


@pytest.mark.parametrize(
    "hits",
    [-1, 7],
)
def test_invalid_hits_are_rejected(
    hits: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="between 0 and 6",
    ):
        PredictionRewardMapper.reward_for_hits(
            hits
        )


def test_empty_policy_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        PredictionRewardMapper().map(
            make_review(),
            policy=" ",
        )
