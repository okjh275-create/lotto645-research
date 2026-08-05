from __future__ import annotations

import json
from pathlib import Path

from lrp.evolution.feedback import (
    AdaptiveAutomationRepository,
    AdaptiveProfileIntegrityDoctor,
    AdaptiveProfileIntegrityReport,
)


def write_revision(
    root: Path,
    *,
    revision: int,
    source_revision: int | None,
    generated_at: str,
    hot_weight: float = 0.30,
    record_type: str | None = None,
    rollback_revision: int | None = None,
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

    payload: dict[str, object] = {
        "schema_version": 1,
        "target_revision": revision,
        "profile": {
            "hot_weight": hot_weight,
            "cold_weight": (
                0.47 - hot_weight
            ),
            "gap_weight": 0.17,
            "trend_weight": 0.14,
            "transition_weight": 0.12,
            "learning_weight": 0.05,
            "adaptive_weight": 0.05,
            "confidence": 0.80,
            "sample_size": 300,
            "revision": revision,
            "generated_at": generated_at,
        },
    }

    if source_revision is not None:
        payload["source_revision"] = (
            source_revision
        )

    if record_type is not None:
        payload["record_type"] = (
            record_type
        )

    if rollback_revision is not None:
        payload["rollback_revision"] = (
            rollback_revision
        )

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_valid_lineage_passes(
    tmp_path: Path,
) -> None:
    write_revision(
        tmp_path,
        revision=12,
        source_revision=11,
        generated_at=(
            "2026-08-01T00:00:00+00:00"
        ),
        hot_weight=0.28,
    )
    write_revision(
        tmp_path,
        revision=13,
        source_revision=12,
        generated_at=(
            "2026-08-02T00:00:00+00:00"
        ),
        hot_weight=0.29,
    )
    write_revision(
        tmp_path,
        revision=14,
        source_revision=13,
        generated_at=(
            "2026-08-03T00:00:00+00:00"
        ),
        hot_weight=0.28,
        record_type="rollback",
        rollback_revision=12,
    )

    report = AdaptiveProfileIntegrityDoctor().inspect(
        AdaptiveAutomationRepository(
            tmp_path
        )
    )

    assert isinstance(
        report,
        AdaptiveProfileIntegrityReport,
    )
    assert report.revision_count == 3
    assert report.first_revision == 12
    assert report.latest_revision == 14
    assert report.rollback_count == 1
    assert report.duplicate_profile_count == 0
    assert report.overall_ok is True
    assert report.issues == ()


def test_revision_lineage_failure_is_reported(
    tmp_path: Path,
) -> None:
    write_revision(
        tmp_path,
        revision=12,
        source_revision=10,
        generated_at=(
            "2026-08-01T00:00:00+00:00"
        ),
    )

    report = AdaptiveProfileIntegrityDoctor().inspect(
        AdaptiveAutomationRepository(
            tmp_path
        )
    )

    assert report.overall_ok is False
    assert any(
        "does not lead" in issue.message
        for issue in report.issues
    )


def test_timestamp_reversal_is_reported(
    tmp_path: Path,
) -> None:
    write_revision(
        tmp_path,
        revision=12,
        source_revision=11,
        generated_at=(
            "2026-08-02T00:00:00+00:00"
        ),
        hot_weight=0.28,
    )
    write_revision(
        tmp_path,
        revision=13,
        source_revision=12,
        generated_at=(
            "2026-08-01T00:00:00+00:00"
        ),
        hot_weight=0.29,
    )

    report = AdaptiveProfileIntegrityDoctor().inspect(
        AdaptiveAutomationRepository(
            tmp_path
        )
    )

    assert report.overall_ok is False
    assert any(
        "earlier than" in issue.message
        for issue in report.issues
    )


def test_naive_timestamp_is_rejected(
    tmp_path: Path,
) -> None:
    write_revision(
        tmp_path,
        revision=12,
        source_revision=11,
        generated_at=(
            "2026-08-01T00:00:00"
        ),
    )

    report = AdaptiveProfileIntegrityDoctor().inspect(
        AdaptiveAutomationRepository(
            tmp_path
        )
    )

    assert report.overall_ok is False
    assert any(
        "timezone-aware" in issue.message
        for issue in report.issues
    )


def test_duplicate_profile_is_warning(
    tmp_path: Path,
) -> None:
    write_revision(
        tmp_path,
        revision=12,
        source_revision=11,
        generated_at=(
            "2026-08-01T00:00:00+00:00"
        ),
    )
    write_revision(
        tmp_path,
        revision=13,
        source_revision=12,
        generated_at=(
            "2026-08-02T00:00:00+00:00"
        ),
    )

    report = AdaptiveProfileIntegrityDoctor().inspect(
        AdaptiveAutomationRepository(
            tmp_path
        )
    )

    assert report.overall_ok is True
    assert report.duplicate_profile_count == 1
    assert any(
        issue.severity == "warning"
        for issue in report.issues
    )


def test_invalid_rollback_lineage_is_reported(
    tmp_path: Path,
) -> None:
    write_revision(
        tmp_path,
        revision=14,
        source_revision=13,
        generated_at=(
            "2026-08-03T00:00:00+00:00"
        ),
        record_type="rollback",
        rollback_revision=13,
    )

    report = AdaptiveProfileIntegrityDoctor().inspect(
        AdaptiveAutomationRepository(
            tmp_path
        )
    )

    assert report.overall_ok is False
    assert any(
        "must be older" in issue.message
        for issue in report.issues
    )


def test_empty_repository_passes_with_no_revisions(
    tmp_path: Path,
) -> None:
    report = AdaptiveProfileIntegrityDoctor().inspect(
        AdaptiveAutomationRepository(
            tmp_path
        )
    )

    assert report.revision_count == 0
    assert report.first_revision is None
    assert report.latest_revision is None
    assert report.overall_ok is True


def test_invalid_repository_type_is_rejected() -> None:
    import pytest

    with pytest.raises(
        TypeError,
        match="AdaptiveAutomationRepository",
    ):
        AdaptiveProfileIntegrityDoctor().inspect(
            object(),  # type: ignore[arg-type]
        )
