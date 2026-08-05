from __future__ import annotations

import pytest

from lrp.evolution.feedback import (
    AdaptiveFeedbackAnalyzer,
)


def make_report() -> dict[str, object]:
    weights = {
        field: {
            "values": [0.03, 0.031],
            "first": 0.03,
            "last": 0.031,
            "net_change": 0.001,
            "mean": 0.0305,
            "direction": "stable",
            "increase_steps": 0,
            "decrease_steps": 0,
            "stable_steps": 1,
        }
        for field in (
            "hot_weight",
            "cold_weight",
            "gap_weight",
            "trend_weight",
            "transition_weight",
            "learning_weight",
            "adaptive_weight",
        )
    }

    return {
        "policies": {
            "floor": {
                "window_count": 2,
                "total_round_count": 200,
                "best_hit_mean_delta": 0.04,
                "practical_hit_mean_delta": 0.03,
                "average_probability_l1_delta": 0.05,
                "average_changed_set_count": 15.0,
            }
        },
        "weight_trends": {
            "policies": {
                "floor": {
                    "weights": weights,
                }
            }
        },
        "significance": {
            "policies": {
                "floor": {
                    "best": {
                        "adaptive_wins": 25,
                        "noop_wins": 17,
                        "ties": 158,
                        "direction": (
                            "adaptive_better"
                        ),
                        "p_value": 0.28,
                        "significant": False,
                    },
                    "practical": {
                        "adaptive_wins": 24,
                        "noop_wins": 18,
                        "ties": 158,
                        "direction": (
                            "adaptive_better"
                        ),
                        "p_value": 0.44,
                        "significant": False,
                    },
                }
            }
        },
    }


def test_analyzes_all_components() -> None:
    feedback = AdaptiveFeedbackAnalyzer().analyze(
        make_report(),
        policy_name="floor",
    )

    assert len(feedback) == 7
    assert tuple(
        item.component
        for item in feedback
    ) == (
        "hot",
        "cold",
        "gap",
        "trend",
        "transition",
        "learning",
        "adaptive",
    )


def test_feedback_contains_policy_metrics() -> None:
    feedback = AdaptiveFeedbackAnalyzer().analyze(
        make_report(),
        policy_name="floor",
    )

    learning = next(
        item
        for item in feedback
        if item.component == "learning"
    )

    assert learning.policy_name == "floor"
    assert learning.window_count == 2
    assert learning.total_round_count == 200
    assert learning.direction == "stable"
    assert learning.p_value == pytest.approx(
        0.28
    )
    assert learning.significant is False
    assert learning.metrics[
        "best_hit_mean_delta"
    ] == pytest.approx(0.04)
    assert learning.metrics[
        "trend_net_change"
    ] == pytest.approx(0.001)


def test_direction_codes_are_recorded() -> None:
    feedback = AdaptiveFeedbackAnalyzer().analyze(
        make_report(),
        policy_name="floor",
    )

    item = feedback[0]

    assert item.metrics[
        "practical_direction_code"
    ] == pytest.approx(1.0)
    assert item.metrics[
        "best_direction_code"
    ] == pytest.approx(1.0)


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="unknown policy",
    ):
        AdaptiveFeedbackAnalyzer().analyze(
            make_report(),
            policy_name="missing",
        )


def test_missing_trend_component_is_rejected() -> None:
    report = make_report()

    trends = report["weight_trends"]

    assert isinstance(trends, dict)

    policies = trends["policies"]

    assert isinstance(policies, dict)

    floor = policies["floor"]

    assert isinstance(floor, dict)

    weights = floor["weights"]

    assert isinstance(weights, dict)

    del weights["hot_weight"]

    with pytest.raises(
        ValueError,
        match="missing trend data",
    ):
        AdaptiveFeedbackAnalyzer().analyze(
            report,
            policy_name="floor",
        )


def test_invalid_report_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="report must be a mapping",
    ):
        AdaptiveFeedbackAnalyzer().analyze(
            object(),  # type: ignore[arg-type]
            policy_name="floor",
        )


def test_public_exports_include_analyzer() -> None:
    import lrp.evolution.feedback as feedback

    assert feedback.__all__ == [
        "AdaptiveAction",
        "AdaptiveAutomationRepository",
        "AdaptiveAutomationResult",
        "AdaptiveAutomationSaveResult",
        "AdaptiveAutomationService",
        "AdaptiveDecision",
        "AdaptiveFeedback",
        "AdaptiveFeedbackAnalyzer",
        "AdaptiveProfileUpdatePlan",
        "AdaptiveProfileUpdatePlanner",
        "AdaptiveRecommendation",
        "AdaptiveRecommendationEngine",
        "AdaptiveRepositoryStatusAnalyzer",
        "AdaptiveRollbackDiff",
        "AdaptiveRollbackManager",
        "AdaptiveRollbackPlan",
        "AdaptiveRollbackRepository",
        "AdaptiveRollbackSaveResult",
        "AdaptiveSafetyGuard",
        "AdaptiveSafetyResult",
        "AdaptiveStatusIssue",
        "AdaptiveStatusReport",
        "RevisionAwareAutomationResult",
        "RevisionAwareAutomationRunner",
    ]
