from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lrp.evolution.contracts import (
    AdaptiveWeightProfile,
)
from lrp.evolution.feedback import (
    AdaptiveAutomationRepository,
    AdaptiveAutomationResult,
    AdaptiveAutomationSaveResult,
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
) -> dict[str, object]:
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
                        "p_value": (
                            0.01
                            if significant
                            else 0.40
                        ),
                        "significant": significant,
                    },
                    "practical": {
                        "adaptive_wins": 32,
                        "noop_wins": 18,
                        "ties": 250,
                        "direction": (
                            "adaptive_better"
                        ),
                        "p_value": (
                            0.01
                            if significant
                            else 0.40
                        ),
                        "significant": significant,
                    },
                }
            }
        },
    }


def automation_result(
    *,
    recommendation_id: str = "auto-1232",
) -> AdaptiveAutomationResult:
    return AdaptiveAutomationService().run(
        report=make_report(),
        policy_name="floor",
        recommendation_id=(
            recommendation_id
        ),
        current_profile=current_profile(),
        created_at_utc=datetime(
            2026,
            8,
            5,
            tzinfo=timezone.utc,
        ),
    )


def test_save_writes_automation_and_profile(
    tmp_path: Path,
) -> None:
    repository = (
        AdaptiveAutomationRepository(
            tmp_path
        )
    )

    saved = repository.save(
        automation_result()
    )

    assert isinstance(
        saved,
        AdaptiveAutomationSaveResult,
    )
    assert saved.automation_created is True
    assert saved.profile_created is True
    assert saved.automation_path.is_file()
    assert saved.profile_path is not None
    assert saved.profile_path.is_file()


def test_save_is_idempotent(
    tmp_path: Path,
) -> None:
    repository = (
        AdaptiveAutomationRepository(
            tmp_path
        )
    )
    result = automation_result()

    first = repository.save(result)
    second = repository.save(result)

    assert first.automation_created is True
    assert first.profile_created is True
    assert second.automation_created is False
    assert second.profile_created is False


def test_automation_json_is_deterministic(
    tmp_path: Path,
) -> None:
    repository = (
        AdaptiveAutomationRepository(
            tmp_path
        )
    )

    saved = repository.save(
        automation_result()
    )

    data = saved.automation_path.read_bytes()

    assert not data.startswith(
        b"\xef\xbb\xbf"
    )
    assert data.endswith(b"\n")

    payload = json.loads(
        data.decode("utf-8")
    )

    assert payload[
        "recommendation"
    ]["recommendation_id"] == "auto-1232"


def test_load_automation(
    tmp_path: Path,
) -> None:
    repository = (
        AdaptiveAutomationRepository(
            tmp_path
        )
    )
    repository.save(
        automation_result()
    )

    payload = repository.load_automation(
        "auto-1232"
    )

    assert payload[
        "update_plan"
    ]["target_revision"] == 13


def test_load_profile_revision(
    tmp_path: Path,
) -> None:
    repository = (
        AdaptiveAutomationRepository(
            tmp_path
        )
    )
    repository.save(
        automation_result()
    )

    payload = (
        repository
        .load_profile_revision(13)
    )

    assert payload[
        "target_revision"
    ] == 13
    assert payload["profile"][
        "revision"
    ] == 13


def test_lists_saved_records(
    tmp_path: Path,
) -> None:
    repository = (
        AdaptiveAutomationRepository(
            tmp_path
        )
    )
    repository.save(
        automation_result()
    )

    assert (
        repository.list_automation_ids()
    ) == ("auto-1232",)

    assert (
        repository.list_profile_revisions()
    ) == (13,)


def test_latest_profile(
    tmp_path: Path,
) -> None:
    repository = (
        AdaptiveAutomationRepository(
            tmp_path
        )
    )
    repository.save(
        automation_result()
    )

    latest = repository.latest_profile()

    assert latest is not None
    assert latest["target_revision"] == 13


def test_invalid_recommendation_id_is_rejected(
    tmp_path: Path,
) -> None:
    repository = (
        AdaptiveAutomationRepository(
            tmp_path
        )
    )

    with pytest.raises(
        ValueError,
        match="unsupported characters",
    ):
        repository.save(
            automation_result(
                recommendation_id=(
                    "../invalid"
                )
            )
        )


def test_existing_different_automation_collides(
    tmp_path: Path,
) -> None:
    repository = (
        AdaptiveAutomationRepository(
            tmp_path
        )
    )
    saved = repository.save(
        automation_result()
    )

    saved.automation_path.write_text(
        "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError,
        match="automation recommendation",
    ):
        repository.save(
            automation_result()
        )


def test_missing_record_is_rejected(
    tmp_path: Path,
) -> None:
    repository = (
        AdaptiveAutomationRepository(
            tmp_path
        )
    )

    with pytest.raises(
        FileNotFoundError,
    ):
        repository.load_automation(
            "missing"
        )


def test_empty_repository_lists_nothing(
    tmp_path: Path,
) -> None:
    repository = (
        AdaptiveAutomationRepository(
            tmp_path
        )
    )

    assert (
        repository.list_automation_ids()
    ) == ()
    assert (
        repository.list_profile_revisions()
    ) == ()
    assert repository.latest_profile() is None


def test_public_exports_include_repository() -> None:
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
        "AdaptiveSafetyGuard",
        "AdaptiveSafetyResult",
        "RevisionAwareAutomationResult",
        "RevisionAwareAutomationRunner",
    ]
