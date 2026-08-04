from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lrp.evolution.contracts import (
    AdaptiveWeightProfile,
)
from lrp.evolution.feedback import (
    AdaptiveProfileUpdatePlan,
    AdaptiveProfileUpdatePlanner,
    AdaptiveSafetyResult,
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


def current_profile() -> AdaptiveWeightProfile:
    return AdaptiveWeightProfile(
        **CURRENT_WEIGHTS,
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


def safety_result(
    *,
    approved: bool = True,
) -> AdaptiveSafetyResult:
    safe = dict(CURRENT_WEIGHTS)

    if approved:
        safe["hot_weight"] = 0.305
        safe["cold_weight"] = 0.165

    return AdaptiveSafetyResult(
        approved=approved,
        proposed_weights=safe,
        safe_weights=(
            safe
            if approved
            else CURRENT_WEIGHTS
        ),
        violations=(
            ()
            if approved
            else ("change rejected",)
        ),
    )


def test_approved_plan_creates_next_revision() -> None:
    plan = AdaptiveProfileUpdatePlanner().plan(
        current_profile=current_profile(),
        safety_result=safety_result(),
        generated_at=datetime(
            2026,
            8,
            5,
            tzinfo=timezone.utc,
        ),
    )

    assert isinstance(
        plan,
        AdaptiveProfileUpdatePlan,
    )
    assert plan.approved is True
    assert plan.source_revision == 12
    assert plan.target_revision == 13
    assert plan.profile.revision == 13
    assert plan.profile.hot_weight == (
        pytest.approx(0.305)
    )
    assert plan.profile.cold_weight == (
        pytest.approx(0.165)
    )


def test_rejected_plan_keeps_revision() -> None:
    plan = AdaptiveProfileUpdatePlanner().plan(
        current_profile=current_profile(),
        safety_result=safety_result(
            approved=False
        ),
    )

    assert plan.approved is False
    assert plan.source_revision == 12
    assert plan.target_revision == 12
    assert plan.profile.revision == 12
    assert plan.profile.hot_weight == (
        pytest.approx(0.30)
    )
    assert plan.violations == (
        "change rejected",
    )


def test_plan_preserves_profile_metadata() -> None:
    plan = AdaptiveProfileUpdatePlanner().plan(
        current_profile=current_profile(),
        safety_result=safety_result(),
    )

    assert plan.profile.confidence == (
        pytest.approx(0.80)
    )
    assert plan.profile.sample_size == 300


def test_metadata_can_be_overridden() -> None:
    plan = AdaptiveProfileUpdatePlanner().plan(
        current_profile=current_profile(),
        safety_result=safety_result(),
        confidence=0.90,
        sample_size=450,
    )

    assert plan.profile.confidence == (
        pytest.approx(0.90)
    )
    assert plan.profile.sample_size == 450


def test_profile_weights_sum_to_one() -> None:
    plan = AdaptiveProfileUpdatePlanner().plan(
        current_profile=current_profile(),
        safety_result=safety_result(),
    )

    total = sum(
        plan.profile
        .to_probability_weights()
        .values()
    )

    assert total == pytest.approx(1.0)


def test_plan_serialization() -> None:
    plan = AdaptiveProfileUpdatePlanner().plan(
        current_profile=current_profile(),
        safety_result=safety_result(),
        generated_at=datetime(
            2026,
            8,
            5,
            1,
            2,
            3,
            tzinfo=timezone.utc,
        ),
    )

    payload = plan.as_dict()

    assert payload["approved"] is True
    assert payload["source_revision"] == 12
    assert payload["target_revision"] == 13
    assert payload["profile"][
        "revision"
    ] == 13
    assert payload["profile"][
        "generated_at"
    ] == "2026-08-05T01:02:03+00:00"


def test_naive_time_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        AdaptiveProfileUpdatePlanner().plan(
            current_profile=current_profile(),
            safety_result=safety_result(),
            generated_at=datetime(
                2026,
                8,
                5,
            ),
        )


def test_invalid_confidence_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="between 0.0 and 1.0",
    ):
        AdaptiveProfileUpdatePlanner().plan(
            current_profile=current_profile(),
            safety_result=safety_result(),
            confidence=1.1,
        )


def test_invalid_sample_size_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="greater than or equal",
    ):
        AdaptiveProfileUpdatePlanner().plan(
            current_profile=current_profile(),
            safety_result=safety_result(),
            sample_size=-1,
        )


def test_public_exports_include_planner() -> None:
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
        "AdaptiveSafetyGuard",
        "AdaptiveSafetyResult",
    ]
