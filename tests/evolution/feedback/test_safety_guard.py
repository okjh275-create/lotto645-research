from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lrp.evolution.feedback import (
    AdaptiveAction,
    AdaptiveDecision,
    AdaptiveFeedback,
    AdaptiveRecommendation,
    AdaptiveSafetyGuard,
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


def recommendation(
    *decisions: AdaptiveDecision,
) -> AdaptiveRecommendation:
    return AdaptiveRecommendation(
        recommendation_id="safe-1",
        created_at_utc=datetime(
            2026,
            8,
            4,
            tzinfo=timezone.utc,
        ),
        feedback=AdaptiveFeedback(
            policy_name="floor",
            component="portfolio",
            window_count=3,
            total_round_count=300,
            direction="mixed",
            p_value=0.10,
            significant=False,
            metrics={
                "component_count": float(
                    len(decisions)
                ),
            },
        ),
        decisions=decisions,
    )


def decision(
    *,
    component: str = "hot_weight",
    action: AdaptiveAction = AdaptiveAction.KEEP,
    current_value: float = 0.30,
    proposed_value: float = 0.30,
) -> AdaptiveDecision:
    return AdaptiveDecision(
        component=component,
        action=action,
        current_value=current_value,
        proposed_value=proposed_value,
        confidence=0.80,
        reason="Safety validation test.",
    )


def test_safe_recommendation_is_approved() -> None:
    result = AdaptiveSafetyGuard(
        max_delta=0.02,
    ).validate(
        recommendation=recommendation(
            decision(
                action=AdaptiveAction.KEEP,
            )
        ),
        current_weights=CURRENT_WEIGHTS,
    )

    assert isinstance(
        result,
        AdaptiveSafetyResult,
    )
    assert result.approved is True
    assert result.violations == ()
    assert sum(
        result.safe_weights.values()
    ) == pytest.approx(1.0)


def test_small_increase_is_normalized() -> None:
    result = AdaptiveSafetyGuard(
        max_delta=0.02,
    ).validate(
        recommendation=recommendation(
            decision(
                action=AdaptiveAction.INCREASE,
                proposed_value=0.31,
            )
        ),
        current_weights=CURRENT_WEIGHTS,
    )

    assert result.approved is True
    assert sum(
        result.proposed_weights.values()
    ) == pytest.approx(1.0)
    assert result.proposed_weights[
        "hot_weight"
    ] > CURRENT_WEIGHTS["hot_weight"]


def test_excessive_delta_is_rejected() -> None:
    result = AdaptiveSafetyGuard(
        max_delta=0.02,
    ).validate(
        recommendation=recommendation(
            decision(
                action=AdaptiveAction.INCREASE,
                proposed_value=0.35,
            )
        ),
        current_weights=CURRENT_WEIGHTS,
    )

    assert result.approved is False
    assert any(
        "delta exceeds maximum"
        in violation
        for violation in result.violations
    )
    assert dict(result.safe_weights) == (
        CURRENT_WEIGHTS
    )


def test_weight_below_floor_is_rejected() -> None:
    result = AdaptiveSafetyGuard(
        minimum_weight=0.03,
    ).validate(
        recommendation=recommendation(
            decision(
                component="learning_weight",
                action=AdaptiveAction.DECREASE,
                current_value=0.05,
                proposed_value=0.02,
            )
        ),
        current_weights=CURRENT_WEIGHTS,
    )

    assert result.approved is False
    assert any(
        "below minimum weight"
        in violation
        for violation in result.violations
    )


def test_rollback_is_rejected_by_default() -> None:
    result = AdaptiveSafetyGuard().validate(
        recommendation=recommendation(
            decision(
                action=AdaptiveAction.ROLLBACK,
            )
        ),
        current_weights=CURRENT_WEIGHTS,
    )

    assert result.approved is False
    assert any(
        "rollback is not allowed"
        in violation
        for violation in result.violations
    )


def test_rollback_can_be_allowed() -> None:
    result = AdaptiveSafetyGuard(
        allow_rollback=True,
    ).validate(
        recommendation=recommendation(
            decision(
                action=AdaptiveAction.ROLLBACK,
            )
        ),
        current_weights=CURRENT_WEIGHTS,
    )

    assert result.approved is True


def test_current_value_mismatch_is_rejected() -> None:
    result = AdaptiveSafetyGuard().validate(
        recommendation=recommendation(
            decision(
                current_value=0.29,
                proposed_value=0.29,
            )
        ),
        current_weights=CURRENT_WEIGHTS,
    )

    assert result.approved is False
    assert any(
        "current value mismatch"
        in violation
        for violation in result.violations
    )


def test_invalid_current_total_is_rejected() -> None:
    invalid = dict(CURRENT_WEIGHTS)
    invalid["hot_weight"] = 0.40

    with pytest.raises(
        ValueError,
        match="sum to 1.0",
    ):
        AdaptiveSafetyGuard().validate(
            recommendation=recommendation(
                decision()
            ),
            current_weights=invalid,
        )


def test_result_mappings_are_immutable() -> None:
    result = AdaptiveSafetyGuard().validate(
        recommendation=recommendation(
            decision()
        ),
        current_weights=CURRENT_WEIGHTS,
    )

    with pytest.raises(TypeError):
        result.safe_weights[
            "hot_weight"
        ] = 0.50  # type: ignore[index]


def test_public_exports_include_safety() -> None:
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
