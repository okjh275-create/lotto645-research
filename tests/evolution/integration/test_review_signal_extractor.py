from __future__ import annotations

import pytest

from lrp.evolution.contracts.learning_context import (
    LearningContext,
)
from lrp.evolution.integration import (
    ReviewSignalExtractor,
)


def make_context() -> LearningContext:
    return LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        rewards={
            (
                "prediction_review:thompson:"
                "portfolio_top_k"
            ): 0.55,
            (
                "prediction_review:thompson:"
                "practical_top5"
            ): 0.20,
        },
        metadata={
            "feedback_observation_count": 10,
        },
    )


def test_extract_maps_review_rewards() -> None:
    signals = ReviewSignalExtractor().extract(
        make_context()
    )

    assert signals == {
        "hot": 0.0,
        "cold": 0.0,
        "gap": 0.0,
        "trend": 0.0,
        "transition": 0.0,
        "learning": 0.55,
        "adaptive": 0.20,
    }


def test_missing_rewards_default_to_zero() -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
    )

    signals = ReviewSignalExtractor().extract(
        context
    )

    assert signals["learning"] == 0.0
    assert signals["adaptive"] == 0.0


def test_policy_free_keys_are_supported() -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        rewards={
            (
                "prediction_review:"
                "portfolio_top_k"
            ): 0.85,
            (
                "prediction_review:"
                "practical_top5"
            ): 0.55,
        },
    )

    signals = ReviewSignalExtractor().extract(
        context
    )

    assert signals["learning"] == pytest.approx(
        0.85
    )
    assert signals["adaptive"] == pytest.approx(
        0.55
    )


def test_sample_size_uses_metadata() -> None:
    assert ReviewSignalExtractor.sample_size(
        make_context()
    ) == 10


def test_invalid_sample_size_falls_back_to_one() -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        metadata={
            "feedback_observation_count": 0,
        },
    )

    assert ReviewSignalExtractor.sample_size(
        context
    ) == 1


def test_invalid_context_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="LearningContext",
    ):
        ReviewSignalExtractor().extract(
            object()  # type: ignore[arg-type]
        )


def test_out_of_range_reward_is_rejected() -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        rewards={
            (
                "prediction_review:"
                "portfolio_top_k"
            ): 1.5,
        },
    )

    with pytest.raises(
        ValueError,
        match="between -1.0 and 1.0",
    ):
        ReviewSignalExtractor().extract(
            context
        )


def test_sample_size_prefers_review_set_count() -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        metadata={
            "review_set_count": 10,
            "feedback_observation_count": 1,
        },
    )

    assert ReviewSignalExtractor.sample_size(
        context
    ) == 10
