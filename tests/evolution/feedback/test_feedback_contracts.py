from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lrp.evolution.feedback import (
    AdaptiveAction,
    AdaptiveDecision,
    AdaptiveFeedback,
    AdaptiveRecommendation,
)


def make_feedback() -> AdaptiveFeedback:
    return AdaptiveFeedback(
        policy_name="floor",
        component="learning",
        window_count=3,
        total_round_count=300,
        direction="adaptive_better",
        p_value=0.12,
        significant=False,
        metrics={
            "best_hit_mean_delta": 0.04,
            "practical_hit_mean_delta": 0.03,
        },
    )


def make_decision(
    *,
    component: str = "learning_weight",
) -> AdaptiveDecision:
    return AdaptiveDecision(
        component=component,
        action=AdaptiveAction.KEEP,
        current_value=0.03,
        proposed_value=0.03,
        confidence=0.80,
        reason=(
            "Cross-window evidence does not "
            "justify a parameter change."
        ),
    )


def test_feedback_normalizes_values() -> None:
    feedback = AdaptiveFeedback(
        policy_name=" floor ",
        component=" learning ",
        window_count=3,
        total_round_count=300,
        direction=" stable ",
        p_value=0.5,
        significant=False,
        metrics={
            "delta": 0.01,
        },
    )

    assert feedback.policy_name == "floor"
    assert feedback.component == "learning"
    assert feedback.direction == "stable"
    assert feedback.metrics["delta"] == (
        pytest.approx(0.01)
    )


def test_feedback_metrics_are_immutable() -> None:
    feedback = make_feedback()

    with pytest.raises(TypeError):
        feedback.metrics["new"] = 1.0  # type: ignore[index]


def test_feedback_rejects_invalid_p_value() -> None:
    with pytest.raises(
        ValueError,
        match="between 0.0 and 1.0",
    ):
        AdaptiveFeedback(
            policy_name="floor",
            component="learning",
            window_count=3,
            total_round_count=300,
            direction="stable",
            p_value=1.1,
            significant=False,
            metrics={},
        )


def test_decision_exposes_delta() -> None:
    decision = AdaptiveDecision(
        component="hot_weight",
        action=AdaptiveAction.DECREASE,
        current_value=0.30,
        proposed_value=0.28,
        confidence=0.75,
        reason="Repeated negative evidence.",
    )

    assert decision.delta == pytest.approx(
        -0.02
    )
    assert decision.as_dict()["action"] == (
        "decrease"
    )


def test_decision_rejects_plain_action_string() -> None:
    with pytest.raises(
        TypeError,
        match="AdaptiveAction",
    ):
        AdaptiveDecision(
            component="hot_weight",
            action="keep",  # type: ignore[arg-type]
            current_value=0.30,
            proposed_value=0.30,
            confidence=0.50,
            reason="No change.",
        )


def test_recommendation_serialization() -> None:
    recommendation = AdaptiveRecommendation(
        recommendation_id="feedback-1232",
        created_at_utc=datetime(
            2026,
            8,
            4,
            6,
            30,
            tzinfo=timezone.utc,
        ),
        feedback=make_feedback(),
        decisions=(
            make_decision(),
        ),
    )

    payload = recommendation.as_dict()

    assert payload["recommendation_id"] == (
        "feedback-1232"
    )
    assert payload["created_at_utc"] == (
        "2026-08-04T06:30:00+00:00"
    )
    assert payload["feedback"][
        "policy_name"
    ] == "floor"
    assert len(payload["decisions"]) == 1


def test_recommendation_rejects_naive_time() -> None:
    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        AdaptiveRecommendation(
            recommendation_id="feedback-1232",
            created_at_utc=datetime(
                2026,
                8,
                4,
            ),
            feedback=make_feedback(),
            decisions=(
                make_decision(),
            ),
        )


def test_recommendation_rejects_empty_decisions() -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        AdaptiveRecommendation(
            recommendation_id="feedback-1232",
            created_at_utc=datetime.now(
                timezone.utc
            ),
            feedback=make_feedback(),
            decisions=(),
        )


def test_recommendation_rejects_duplicate_components() -> None:
    with pytest.raises(
        ValueError,
        match="components must be unique",
    ):
        AdaptiveRecommendation(
            recommendation_id="feedback-1232",
            created_at_utc=datetime.now(
                timezone.utc
            ),
            feedback=make_feedback(),
            decisions=(
                make_decision(),
                make_decision(),
            ),
        )


def test_feedback_public_exports() -> None:
    import lrp.evolution.feedback as feedback

    assert feedback.__all__ == [
        "AdaptiveAction",
        "AdaptiveAutomationResult",
        "AdaptiveAutomationService",
        "AdaptiveDecision",
        "AdaptiveFeedback",
        "AdaptiveFeedbackAnalyzer",
        "AdaptiveProfileUpdatePlan",
        "AdaptiveProfileUpdatePlanner",
        "AdaptiveRecommendation",
        "AdaptiveRecommendationEngine",
        "AdaptiveSafetyGuard",
        "AdaptiveSafetyResult",
    ]
