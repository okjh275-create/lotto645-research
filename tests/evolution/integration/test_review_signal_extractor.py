from __future__ import annotations

from dataclasses import replace

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


def test_sample_size_prefers_cumulative_review_count() -> None:
    extractor = ReviewSignalExtractor()

    context = replace(
        make_context(),
        metadata={
            "review_set_count": 20,
            "cumulative_review_set_count": 80,
        },
    )

    assert extractor.sample_size(context) == 80


def test_structured_reward_vector_is_preferred() -> None:
    extractor = ReviewSignalExtractor()

    context = replace(
        make_context(),
        metadata={
            "reward_vector_portfolio_hit": 0.60,
            "reward_vector_practical_hit": 0.30,
            "reward_vector_rank_quality": 0.30,
            "reward_vector_coverage": 0.00,
            "reward_vector_diversity": 0.00,
            "reward_vector_stability": 0.00,
        },
    )

    signals = extractor.extract(context)

    assert signals["learning"] == pytest.approx(
        0.30
    )
    assert signals["adaptive"] == pytest.approx(
        0.10
    )


def test_reward_vector_falls_back_to_legacy_rewards() -> None:
    extractor = ReviewSignalExtractor()

    context = make_context()

    signals = extractor.extract(context)

    assert signals["learning"] == pytest.approx(
        0.55
    )
    assert signals["adaptive"] == pytest.approx(
        0.20
    )


def test_invalid_structured_reward_is_rejected() -> None:
    extractor = ReviewSignalExtractor()

    context = replace(
        make_context(),
        metadata={
            "reward_vector_portfolio_hit": 2.0,
        },
    )

    with pytest.raises(
        ValueError,
        match="between -1.0 and 1.0",
    ):
        extractor.extract(context)


def test_sample_size_preserves_cumulative_review_count() -> None:
    extractor = ReviewSignalExtractor()

    context = replace(
        make_context(),
        metadata={
            "review_set_count": 20,
            "cumulative_review_set_count": 80,
            "reward_vector_sample_size": 100,
        },
    )

    assert extractor.sample_size(context) == 80


def test_feature_attribution_signals_are_extracted() -> None:
    extractor = ReviewSignalExtractor()

    context = replace(
        make_context(),
        metadata={
            "feature_signal_hot": 0.25,
            "feature_signal_cold": -0.10,
            "feature_signal_gap": 0.15,
            "feature_signal_trend": 0.05,
            "feature_signal_transition": -0.20,
        },
    )

    signals = extractor.extract(context)

    assert signals["hot"] == pytest.approx(0.25)
    assert signals["cold"] == pytest.approx(-0.10)
    assert signals["gap"] == pytest.approx(0.15)
    assert signals["trend"] == pytest.approx(0.05)
    assert signals["transition"] == pytest.approx(-0.20)


def test_missing_feature_signals_remain_neutral() -> None:
    signals = ReviewSignalExtractor().extract(
        make_context()
    )

    assert signals["hot"] == 0.0
    assert signals["cold"] == 0.0
    assert signals["gap"] == 0.0
    assert signals["trend"] == 0.0
    assert signals["transition"] == 0.0


def test_invalid_feature_signal_is_rejected() -> None:
    context = replace(
        make_context(),
        metadata={
            "feature_signal_hot": 1.5,
        },
    )

    with pytest.raises(
        ValueError,
        match="between -1.0 and 1.0",
    ):
        ReviewSignalExtractor().extract(
            context
        )
