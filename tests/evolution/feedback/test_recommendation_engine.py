from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lrp.evolution.feedback import (
    AdaptiveAction,
    AdaptiveFeedback,
    AdaptiveRecommendationEngine,
)


CURRENT_WEIGHTS = {
    "hot_weight": 0.30,
    "cold_weight": 0.17,
    "gap_weight": 0.17,
    "trend_weight": 0.14,
    "transition_weight": 0.12,
    "learning_weight": 0.05,
    "adaptive_weight": 0.05,
}


def make_feedback(
    *,
    component: str = "hot",
    direction: str = "stable",
    p_value: float = 0.01,
    significant: bool = True,
    practical_code: float = 1.0,
    best_code: float = 1.0,
    practical_delta: float = 0.04,
    best_delta: float = 0.03,
) -> AdaptiveFeedback:
    return AdaptiveFeedback(
        policy_name="floor",
        component=component,
        window_count=3,
        total_round_count=300,
        direction=direction,
        p_value=p_value,
        significant=significant,
        metrics={
            "practical_direction_code": (
                practical_code
            ),
            "best_direction_code": best_code,
            "practical_hit_mean_delta": (
                practical_delta
            ),
            "best_hit_mean_delta": best_delta,
        },
    )


def test_non_significant_feedback_keeps_weight() -> None:
    recommendation = (
        AdaptiveRecommendationEngine()
        .recommend(
            recommendation_id="rec-1",
            feedback=(
                make_feedback(
                    significant=False,
                    p_value=0.40,
                ),
            ),
            current_weights=CURRENT_WEIGHTS,
            created_at_utc=datetime(
                2026,
                8,
                4,
                tzinfo=timezone.utc,
            ),
        )
    )

    decision = recommendation.decisions[0]

    assert decision.action is (
        AdaptiveAction.KEEP
    )
    assert decision.proposed_value == (
        pytest.approx(0.30)
    )


def test_positive_evidence_increases_weight() -> None:
    recommendation = (
        AdaptiveRecommendationEngine(
            step_size=0.01
        )
        .recommend(
            recommendation_id="rec-2",
            feedback=(
                make_feedback(),
            ),
            current_weights=CURRENT_WEIGHTS,
        )
    )

    decision = recommendation.decisions[0]

    assert decision.action is (
        AdaptiveAction.INCREASE
    )
    assert decision.proposed_value == (
        pytest.approx(0.31)
    )


def test_negative_evidence_decreases_weight() -> None:
    recommendation = (
        AdaptiveRecommendationEngine()
        .recommend(
            recommendation_id="rec-3",
            feedback=(
                make_feedback(
                    practical_code=-1.0,
                    best_code=-1.0,
                    practical_delta=-0.02,
                    best_delta=-0.01,
                ),
            ),
            current_weights=CURRENT_WEIGHTS,
        )
    )

    decision = recommendation.decisions[0]

    assert decision.action is (
        AdaptiveAction.DECREASE
    )
    assert decision.proposed_value == (
        pytest.approx(0.29)
    )


def test_severe_negative_evidence_rolls_back() -> None:
    recommendation = (
        AdaptiveRecommendationEngine(
            rollback_threshold=-0.05
        )
        .recommend(
            recommendation_id="rec-4",
            feedback=(
                make_feedback(
                    practical_code=-1.0,
                    best_code=-1.0,
                    practical_delta=-0.08,
                    best_delta=-0.06,
                ),
            ),
            current_weights=CURRENT_WEIGHTS,
        )
    )

    decision = recommendation.decisions[0]

    assert decision.action is (
        AdaptiveAction.ROLLBACK
    )
    assert decision.proposed_value == (
        pytest.approx(0.30)
    )


def test_multiple_components_are_recommended() -> None:
    recommendation = (
        AdaptiveRecommendationEngine()
        .recommend(
            recommendation_id="rec-5",
            feedback=(
                make_feedback(
                    component="hot"
                ),
                make_feedback(
                    component="gap",
                    significant=False,
                ),
            ),
            current_weights=CURRENT_WEIGHTS,
        )
    )

    assert len(
        recommendation.decisions
    ) == 2

    assert tuple(
        decision.component
        for decision in recommendation.decisions
    ) == (
        "hot_weight",
        "gap_weight",
    )


def test_summary_feedback_is_created() -> None:
    recommendation = (
        AdaptiveRecommendationEngine()
        .recommend(
            recommendation_id="rec-6",
            feedback=(
                make_feedback(
                    component="hot"
                ),
                make_feedback(
                    component="gap",
                    significant=False,
                ),
            ),
            current_weights=CURRENT_WEIGHTS,
        )
    )

    assert recommendation.feedback.component == (
        "portfolio"
    )
    assert recommendation.feedback.metrics[
        "component_count"
    ] == pytest.approx(2.0)
    assert recommendation.feedback.metrics[
        "significant_component_count"
    ] == pytest.approx(1.0)


def test_invalid_weight_total_is_rejected() -> None:
    invalid = dict(CURRENT_WEIGHTS)
    invalid["hot_weight"] = 0.40

    with pytest.raises(
        ValueError,
        match="sum to 1.0",
    ):
        AdaptiveRecommendationEngine().recommend(
            recommendation_id="rec-7",
            feedback=(
                make_feedback(),
            ),
            current_weights=invalid,
        )


def test_duplicate_components_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="components must be unique",
    ):
        AdaptiveRecommendationEngine().recommend(
            recommendation_id="rec-8",
            feedback=(
                make_feedback(),
                make_feedback(),
            ),
            current_weights=CURRENT_WEIGHTS,
        )


def test_empty_feedback_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        AdaptiveRecommendationEngine().recommend(
            recommendation_id="rec-9",
            feedback=(),
            current_weights=CURRENT_WEIGHTS,
        )


def test_public_exports_include_engine() -> None:
    import lrp.evolution.feedback as feedback

    assert feedback.__all__ == [
        "AdaptiveAction",
        "AdaptiveDecision",
        "AdaptiveFeedback",
        "AdaptiveFeedbackAnalyzer",
        "AdaptiveRecommendation",
        "AdaptiveRecommendationEngine",
        "AdaptiveSafetyGuard",
        "AdaptiveSafetyResult",
    ]
