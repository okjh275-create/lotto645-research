from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lrp.evolution.contracts import (
    AdaptiveWeightProfile,
)
from lrp.evolution.feedback import (
    AdaptiveAction,
    AdaptiveAutomationResult,
    AdaptiveAutomationService,
)


def current_profile() -> AdaptiveWeightProfile:
    return AdaptiveWeightProfile(
        hot_weight=0.30,
        cold_weight=0.17,
        gap_weight=0.17,
        trend_weight=0.14,
        transition_weight=0.12,
        learning_weight=0.05,
        adaptive_weight=0.05,
        confidence=0.80,
        sample_size=300,
        revision=12,
        generated_at=datetime(
            2026,
            8,
            4,
            tzinfo=timezone.utc,
        ),
    )


def make_report(
    *,
    significant: bool = False,
    p_value: float = 0.40,
) -> dict[str, object]:
    trend_weights = {
        field: {
            "values": [0.05, 0.05],
            "first": 0.05,
            "last": 0.05,
            "net_change": 0.0,
            "mean": 0.05,
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
                "window_count": 3,
                "total_round_count": 300,
                "best_hit_mean_delta": 0.04,
                "practical_hit_mean_delta": 0.03,
                "average_probability_l1_delta": 0.05,
                "average_changed_set_count": 15.0,
            }
        },
        "weight_trends": {
            "policies": {
                "floor": {
                    "weights": trend_weights,
                }
            }
        },
        "significance": {
            "policies": {
                "floor": {
                    "best": {
                        "adaptive_wins": 30,
                        "noop_wins": 20,
                        "ties": 250,
                        "direction": (
                            "adaptive_better"
                        ),
                        "p_value": p_value,
                        "significant": significant,
                    },
                    "practical": {
                        "adaptive_wins": 32,
                        "noop_wins": 18,
                        "ties": 250,
                        "direction": (
                            "adaptive_better"
                        ),
                        "p_value": p_value,
                        "significant": significant,
                    },
                }
            }
        },
    }


def test_run_returns_complete_result() -> None:
    result = AdaptiveAutomationService().run(
        report=make_report(),
        policy_name="floor",
        recommendation_id="auto-1",
        current_profile=current_profile(),
        created_at_utc=datetime(
            2026,
            8,
            5,
            tzinfo=timezone.utc,
        ),
    )

    assert isinstance(
        result,
        AdaptiveAutomationResult,
    )
    assert len(result.feedback) == 7
    assert len(
        result.recommendation.decisions
    ) == 7
    assert result.safety_result.approved is True
    assert result.update_plan.approved is True


def test_non_significant_report_keeps_weights() -> None:
    result = AdaptiveAutomationService().run(
        report=make_report(
            significant=False,
            p_value=0.40,
        ),
        policy_name="floor",
        recommendation_id="auto-2",
        current_profile=current_profile(),
    )

    assert all(
        decision.action
        is AdaptiveAction.KEEP
        for decision
        in result.recommendation.decisions
    )
    assert result.update_plan.target_revision == 13
    assert result.update_plan.profile.hot_weight == (
        pytest.approx(0.30)
    )


def test_significant_report_generates_changes() -> None:
    result = AdaptiveAutomationService().run(
        report=make_report(
            significant=True,
            p_value=0.01,
        ),
        policy_name="floor",
        recommendation_id="auto-3",
        current_profile=current_profile(),
    )

    assert any(
        decision.action
        is AdaptiveAction.INCREASE
        for decision
        in result.recommendation.decisions
    )


def test_metadata_overrides_are_applied() -> None:
    result = AdaptiveAutomationService().run(
        report=make_report(),
        policy_name="floor",
        recommendation_id="auto-4",
        current_profile=current_profile(),
        target_confidence=0.90,
        target_sample_size=450,
    )

    assert result.update_plan.profile.confidence == (
        pytest.approx(0.90)
    )
    assert result.update_plan.profile.sample_size == 450


def test_result_serialization() -> None:
    result = AdaptiveAutomationService().run(
        report=make_report(),
        policy_name="floor",
        recommendation_id="auto-5",
        current_profile=current_profile(),
        created_at_utc=datetime(
            2026,
            8,
            5,
            1,
            2,
            3,
            tzinfo=timezone.utc,
        ),
    )

    payload = result.as_dict()

    assert len(payload["feedback"]) == 7
    assert payload["recommendation"][
        "recommendation_id"
    ] == "auto-5"
    assert payload["safety_result"][
        "approved"
    ] is True
    assert payload["update_plan"][
        "target_revision"
    ] == 13


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="unknown policy",
    ):
        AdaptiveAutomationService().run(
            report=make_report(),
            policy_name="missing",
            recommendation_id="auto-6",
            current_profile=current_profile(),
        )


def test_invalid_profile_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="AdaptiveWeightProfile",
    ):
        AdaptiveAutomationService().run(
            report=make_report(),
            policy_name="floor",
            recommendation_id="auto-7",
            current_profile=object(),  # type: ignore[arg-type]
        )


def test_public_exports_include_service() -> None:
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
        "AdaptiveProfileIntegrityDoctor",
        "AdaptiveProfileIntegrityReport",
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
