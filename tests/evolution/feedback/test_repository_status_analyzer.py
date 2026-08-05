from __future__ import annotations

import json
from pathlib import Path

from lrp.evolution.feedback import (
    AdaptiveAutomationRepository,
    AdaptiveRepositoryStatusAnalyzer,
)


def profile_payload(
    *,
    revision: int,
    record_type: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "target_revision": revision,
        "profile": {
            "hot_weight": 0.30,
            "cold_weight": 0.17,
            "gap_weight": 0.17,
            "trend_weight": 0.14,
            "transition_weight": 0.12,
            "learning_weight": 0.05,
            "adaptive_weight": 0.05,
            "confidence": 0.80,
            "sample_size": 300,
            "revision": revision,
            "generated_at": (
                "2026-08-06T00:00:00+00:00"
            ),
        },
    }

    if record_type == "rollback":
        payload.update(
            {
                "record_type": "rollback",
                "source_revision": (
                    revision - 1
                ),
                "rollback_revision": (
                    revision - 3
                ),
            }
        )

    return payload


def write_profile(
    root: Path,
    *,
    revision: int,
    record_type: str | None = None,
) -> None:
    path = (
        root
        / "profiles"
        / f"revision-{revision:08d}.json"
    )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            profile_payload(
                revision=revision,
                record_type=record_type,
            )
        ),
        encoding="utf-8",
    )


def write_automation(
    root: Path,
    *,
    recommendation_id: str,
) -> None:
    path = (
        root
        / "automation"
        / f"{recommendation_id}.json"
    )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            {
                "recommendation": {
                    "recommendation_id": (
                        recommendation_id
                    )
                },
                "update_plan": {
                    "approved": True,
                },
            }
        ),
        encoding="utf-8",
    )


def test_healthy_repository_passes(
    tmp_path: Path,
) -> None:
    write_profile(
        tmp_path,
        revision=12,
    )
    write_profile(
        tmp_path,
        revision=13,
    )
    write_profile(
        tmp_path,
        revision=14,
        record_type="rollback",
    )
    write_automation(
        tmp_path,
        recommendation_id="auto-13",
    )

    report = (
        AdaptiveRepositoryStatusAnalyzer()
        .analyze(
            AdaptiveAutomationRepository(
                tmp_path
            )
        )
    )

    assert report.latest_revision == 14
    assert report.repository_ok is True
    assert report.profile_ok is True
    assert report.validation_ok is True
    assert report.rollback_count == 1
    assert report.recommendation_count == 1
    assert report.overall_ok is True
    assert report.issues == ()


def test_revision_gap_is_reported(
    tmp_path: Path,
) -> None:
    write_profile(
        tmp_path,
        revision=12,
    )
    write_profile(
        tmp_path,
        revision=14,
    )

    report = (
        AdaptiveRepositoryStatusAnalyzer()
        .analyze(
            AdaptiveAutomationRepository(
                tmp_path
            )
        )
    )

    assert report.repository_ok is False
    assert report.overall_ok is False
    assert any(
        issue.category == "revision"
        and "gap detected" in issue.message
        for issue in report.issues
    )


def test_invalid_weight_total_is_reported(
    tmp_path: Path,
) -> None:
    payload = profile_payload(
        revision=12
    )

    profile = payload["profile"]

    assert isinstance(profile, dict)

    profile["hot_weight"] = 0.40

    path = (
        tmp_path
        / "profiles"
        / "revision-00000012.json"
    )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    report = (
        AdaptiveRepositoryStatusAnalyzer()
        .analyze(
            AdaptiveAutomationRepository(
                tmp_path
            )
        )
    )

    assert report.profile_ok is False
    assert any(
        "weights sum" in issue.message
        for issue in report.issues
    )


def test_empty_repository_has_no_profile(
    tmp_path: Path,
) -> None:
    report = (
        AdaptiveRepositoryStatusAnalyzer()
        .analyze(
            AdaptiveAutomationRepository(
                tmp_path
            )
        )
    )

    assert report.latest_revision == 0
    assert report.repository_ok is True
    assert report.profile_ok is False
    assert report.overall_ok is False


def test_mismatched_automation_id_is_reported(
    tmp_path: Path,
) -> None:
    write_profile(
        tmp_path,
        revision=12,
    )

    path = (
        tmp_path
        / "automation"
        / "auto-12.json"
    )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            {
                "recommendation": {
                    "recommendation_id": (
                        "different-id"
                    )
                },
                "update_plan": {
                    "approved": True,
                },
            }
        ),
        encoding="utf-8",
    )

    report = (
        AdaptiveRepositoryStatusAnalyzer()
        .analyze(
            AdaptiveAutomationRepository(
                tmp_path
            )
        )
    )

    assert report.repository_ok is False
    assert report.recommendation_count == 1
    assert any(
        issue.category == "automation"
        for issue in report.issues
    )


def test_invalid_repository_type_is_rejected() -> None:
    import pytest

    with pytest.raises(
        TypeError,
        match="AdaptiveAutomationRepository",
    ):
        AdaptiveRepositoryStatusAnalyzer().analyze(
            object(),  # type: ignore[arg-type]
        )
