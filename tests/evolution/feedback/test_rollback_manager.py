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
    AdaptiveRollbackManager,
    AdaptiveRollbackPlan,
)


def make_profile(
    *,
    revision: int,
    hot_weight: float,
    generated_at: datetime,
) -> AdaptiveWeightProfile:
    cold_weight = 0.47 - hot_weight

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
        generated_at=generated_at,
    )


def write_profile_revision(
    repository: AdaptiveAutomationRepository,
    *,
    revision: int,
    profile: AdaptiveWeightProfile,
) -> None:
    path = (
        repository.profile_root
        / f"revision-{revision:08d}.json"
    )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": 1,
        "recommendation_id": (
            f"fixture-{revision}"
        ),
        "source_revision": (
            max(0, revision - 1)
        ),
        "target_revision": revision,
        "profile": {
            "hot_weight": profile.hot_weight,
            "cold_weight": profile.cold_weight,
            "gap_weight": profile.gap_weight,
            "trend_weight": profile.trend_weight,
            "transition_weight": (
                profile.transition_weight
            ),
            "learning_weight": (
                profile.learning_weight
            ),
            "adaptive_weight": (
                profile.adaptive_weight
            ),
            "confidence": profile.confidence,
            "sample_size": profile.sample_size,
            "revision": profile.revision,
            "generated_at": (
                profile.generated_at
                .isoformat()
            ),
        },
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def prepare_repository(
    tmp_path: Path,
) -> tuple[
    AdaptiveAutomationRepository,
    AdaptiveWeightProfile,
]:
    repository = AdaptiveAutomationRepository(
        tmp_path
    )

    old = make_profile(
        revision=12,
        hot_weight=0.28,
        generated_at=datetime(
            2026,
            8,
            1,
            tzinfo=timezone.utc,
        ),
    )
    current = make_profile(
        revision=15,
        hot_weight=0.31,
        generated_at=datetime(
            2026,
            8,
            4,
            tzinfo=timezone.utc,
        ),
    )

    write_profile_revision(
        repository,
        revision=12,
        profile=old,
    )
    write_profile_revision(
        repository,
        revision=15,
        profile=current,
    )

    return repository, current


def test_plans_historical_weight_restore(
    tmp_path: Path,
) -> None:
    repository, current = prepare_repository(
        tmp_path
    )

    plan = AdaptiveRollbackManager(
        repository=repository
    ).plan(
        current_profile=current,
        rollback_revision=12,
        generated_at=datetime(
            2026,
            8,
            5,
            tzinfo=timezone.utc,
        ),
    )

    assert isinstance(
        plan,
        AdaptiveRollbackPlan,
    )
    assert plan.source_revision == 15
    assert plan.rollback_revision == 12
    assert plan.target_revision == 16
    assert plan.profile.revision == 16
    assert plan.profile.hot_weight == (
        pytest.approx(0.28)
    )


def test_rollback_does_not_modify_historical_revision(
    tmp_path: Path,
) -> None:
    repository, current = prepare_repository(
        tmp_path
    )

    original = (
        repository
        .load_profile_revision(12)
    )

    AdaptiveRollbackManager(
        repository=repository
    ).plan(
        current_profile=current,
        rollback_revision=12,
    )

    after = (
        repository
        .load_profile_revision(12)
    )

    assert after == original
    assert not (
        repository.profile_root
        / "revision-00000016.json"
    ).exists()


def test_diff_reports_changed_components(
    tmp_path: Path,
) -> None:
    repository, current = prepare_repository(
        tmp_path
    )

    plan = AdaptiveRollbackManager(
        repository=repository
    ).plan(
        current_profile=current,
        rollback_revision=12,
    )

    hot = next(
        item
        for item in plan.differences
        if item.component == "hot_weight"
    )

    assert hot.current_value == (
        pytest.approx(0.31)
    )
    assert hot.target_value == (
        pytest.approx(0.28)
    )
    assert hot.delta == pytest.approx(
        -0.03
    )
    assert plan.changed_component_count == 2


def test_plan_preserves_current_metadata_by_default(
    tmp_path: Path,
) -> None:
    repository, current = prepare_repository(
        tmp_path
    )

    plan = AdaptiveRollbackManager(
        repository=repository
    ).plan(
        current_profile=current,
        rollback_revision=12,
    )

    assert plan.profile.confidence == (
        pytest.approx(current.confidence)
    )
    assert plan.profile.sample_size == (
        current.sample_size
    )


def test_current_revision_must_match_repository(
    tmp_path: Path,
) -> None:
    repository, _ = prepare_repository(
        tmp_path
    )

    stale = make_profile(
        revision=14,
        hot_weight=0.30,
        generated_at=datetime(
            2026,
            8,
            3,
            tzinfo=timezone.utc,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="does not match repository head",
    ):
        AdaptiveRollbackManager(
            repository=repository
        ).plan(
            current_profile=stale,
            rollback_revision=12,
        )


def test_future_or_current_revision_is_rejected(
    tmp_path: Path,
) -> None:
    repository, current = prepare_repository(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="must be less",
    ):
        AdaptiveRollbackManager(
            repository=repository
        ).plan(
            current_profile=current,
            rollback_revision=15,
        )


def test_missing_historical_revision_is_rejected(
    tmp_path: Path,
) -> None:
    repository, current = prepare_repository(
        tmp_path
    )

    with pytest.raises(
        FileNotFoundError,
    ):
        AdaptiveRollbackManager(
            repository=repository
        ).plan(
            current_profile=current,
            rollback_revision=10,
        )


def test_plan_serialization(
    tmp_path: Path,
) -> None:
    repository, current = prepare_repository(
        tmp_path
    )

    plan = AdaptiveRollbackManager(
        repository=repository
    ).plan(
        current_profile=current,
        rollback_revision=12,
    )

    payload = plan.as_dict()

    assert payload["source_revision"] == 15
    assert payload["rollback_revision"] == 12
    assert payload["target_revision"] == 16
    assert payload["profile"]["revision"] == 16
    assert len(payload["differences"]) == 7
