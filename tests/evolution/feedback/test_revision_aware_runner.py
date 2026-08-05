from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from lrp.evolution.contracts import (
    AdaptiveWeightProfile,
)
from lrp.evolution.feedback import (
    AdaptiveAutomationRepository,
    RevisionAwareAutomationResult,
    RevisionAwareAutomationRunner,
)


def profile(
    *,
    revision: int = 12,
    hot_weight: float = 0.30,
) -> AdaptiveWeightProfile:
    cold_weight = (
        0.47 - hot_weight
    )

    return AdaptiveWeightProfile(
        hot_weight=hot_weight,
        cold_weight=cold_weight,
        gap_weight=0.17,
        trend_weight=0.14,
        transition_weight=0.12,
        learning_weight=0.05,
        adaptive_weight=0.05,
        confidence=0.80,
        sample_size=300,
        revision=revision,
        generated_at=datetime(
            2026,
            8,
            4,
            tzinfo=timezone.utc,
        ),
    )


def report() -> dict[str, object]:
    trends = {
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
                    "weights": trends,
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
                        "p_value": 0.40,
                        "significant": False,
                    },
                    "practical": {
                        "adaptive_wins": 32,
                        "noop_wins": 18,
                        "ties": 250,
                        "direction": (
                            "adaptive_better"
                        ),
                        "p_value": 0.40,
                        "significant": False,
                    },
                }
            }
        },
    }


def seed_repository(
    repository: AdaptiveAutomationRepository,
) -> AdaptiveWeightProfile:
    current = profile(
        revision=12
    )

    result = RevisionAwareAutomationRunner(
        repository=repository,
    ).run(
        report=report(),
        policy_name="floor",
        recommendation_id="seed-13",
        current_profile=current,
        created_at_utc=datetime(
            2026,
            8,
            5,
            tzinfo=timezone.utc,
        ),
    )

    return result.automation_result.update_plan.profile


def test_empty_repository_run_is_allowed(
    tmp_path: Path,
) -> None:
    repository = (
        AdaptiveAutomationRepository(
            tmp_path
        )
    )

    result = RevisionAwareAutomationRunner(
        repository=repository,
    ).run(
        report=report(),
        policy_name="floor",
        recommendation_id="auto-13",
        current_profile=profile(),
    )

    assert isinstance(
        result,
        RevisionAwareAutomationResult,
    )
    assert (
        result.repository_revision_before
        is None
    )
    assert (
        result.repository_revision_after
        == 13
    )
    assert result.save_result.profile_created is True


def test_matching_repository_head_is_accepted(
    tmp_path: Path,
) -> None:
    repository = (
        AdaptiveAutomationRepository(
            tmp_path
        )
    )

    current = seed_repository(
        repository
    )

    result = RevisionAwareAutomationRunner(
        repository=repository,
    ).run(
        report=report(),
        policy_name="floor",
        recommendation_id="auto-14",
        current_profile=current,
        created_at_utc=datetime(
            2026,
            8,
            6,
            tzinfo=timezone.utc,
        ),
    )

    assert (
        result.repository_revision_before
        == 13
    )
    assert (
        result.repository_revision_after
        == 14
    )


def test_stale_revision_is_rejected(
    tmp_path: Path,
) -> None:
    repository = (
        AdaptiveAutomationRepository(
            tmp_path
        )
    )

    seed_repository(repository)

    with pytest.raises(
        RuntimeError,
        match="revision does not match",
    ):
        RevisionAwareAutomationRunner(
            repository=repository,
        ).run(
            report=report(),
            policy_name="floor",
            recommendation_id="stale",
            current_profile=profile(
                revision=12
            ),
        )


def test_same_revision_with_different_weights_is_rejected(
    tmp_path: Path,
) -> None:
    repository = (
        AdaptiveAutomationRepository(
            tmp_path
        )
    )

    current = seed_repository(
        repository
    )

    mismatched = profile(
        revision=current.revision,
        hot_weight=0.31,
    )

    with pytest.raises(
        RuntimeError,
        match="does not match repository head",
    ):
        RevisionAwareAutomationRunner(
            repository=repository,
        ).run(
            report=report(),
            policy_name="floor",
            recommendation_id="mismatch",
            current_profile=mismatched,
        )


def test_empty_repository_can_be_rejected(
    tmp_path: Path,
) -> None:
    repository = (
        AdaptiveAutomationRepository(
            tmp_path
        )
    )

    with pytest.raises(
        RuntimeError,
        match="repository is empty",
    ):
        RevisionAwareAutomationRunner(
            repository=repository,
            allow_empty_repository=False,
        ).run(
            report=report(),
            policy_name="floor",
            recommendation_id="empty",
            current_profile=profile(),
        )


def test_runner_result_serialization(
    tmp_path: Path,
) -> None:
    repository = (
        AdaptiveAutomationRepository(
            tmp_path
        )
    )

    result = RevisionAwareAutomationRunner(
        repository=repository,
    ).run(
        report=report(),
        policy_name="floor",
        recommendation_id="serialize",
        current_profile=profile(),
    )

    payload = result.as_dict()

    assert payload[
        "repository_revision_before"
    ] is None
    assert payload[
        "repository_revision_after"
    ] == 13
    assert payload["save_result"][
        "profile_created"
    ] is True


def test_invalid_repository_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="AdaptiveAutomationRepository",
    ):
        RevisionAwareAutomationRunner(
            repository=object(),  # type: ignore[arg-type]
        )


def test_public_exports_include_runner() -> None:
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
        "AdaptiveRollbackDiff",
        "AdaptiveRollbackManager",
        "AdaptiveRollbackPlan",
        "AdaptiveRollbackRepository",
        "AdaptiveRollbackSaveResult",
        "AdaptiveSafetyGuard",
        "AdaptiveSafetyResult",
        "RevisionAwareAutomationResult",
        "RevisionAwareAutomationRunner",
    ]
