from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from lrp.evolution.feedback import (
    AdaptiveAutomationDoctor,
    AdaptiveAutomationDoctorMarkdownRenderer,
    AdaptiveAutomationDoctorReportWriter,
    AdaptiveAutomationRepository,
)


WEIGHT_FIELDS = (
    "hot_weight",
    "cold_weight",
    "gap_weight",
    "trend_weight",
    "transition_weight",
    "learning_weight",
    "adaptive_weight",
)


def write_profile_revision(
    repository_root: Path,
    *,
    revision: int,
    generated_at_day: int,
) -> None:
    path = (
        repository_root
        / "profiles"
        / f"revision-{revision:08d}.json"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    hot_weight = (
        0.30
        + ((revision % 5) * 0.001)
    )
    cold_weight = (
        0.47 - hot_weight
    )

    profile = {
        "hot_weight": hot_weight,
        "cold_weight": cold_weight,
        "gap_weight": 0.17,
        "trend_weight": 0.14,
        "transition_weight": 0.12,
        "learning_weight": 0.05,
        "adaptive_weight": 0.05,
        "confidence": 0.80,
        "sample_size": (
            revision * 20
        ),
        "revision": revision,
        "generated_at": (
            "2026-01-01T00:00:00+00:00"
        ),
    }

    payload = {
        "schema_version": 1,
        "source_revision": (
            revision - 1
        ),
        "target_revision": revision,
        "profile": profile,
    }

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_automation_record(
    repository_root: Path,
    *,
    revision: int,
) -> None:
    recommendation_id = (
        f"performance-{revision:08d}"
    )

    path = (
        repository_root
        / "automation"
        / f"{recommendation_id}.json"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": 1,
        "recommendation": {
            "recommendation_id": (
                recommendation_id
            ),
        },
        "update_plan": {
            "approved": True,
            "source_revision": (
                revision - 1
            ),
            "target_revision": revision,
        },
    }

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def build_repository(
    root: Path,
    *,
    revision_count: int,
    automation_stride: int = 10,
) -> AdaptiveAutomationRepository:
    first_revision = 1

    for offset in range(
        revision_count
    ):
        revision = (
            first_revision + offset
        )

        write_profile_revision(
            root,
            revision=revision,
            generated_at_day=offset,
        )

        if (
            revision
            % automation_stride
            == 0
        ):
            write_automation_record(
                root,
                revision=revision,
            )

    return AdaptiveAutomationRepository(
        root
    )


def elapsed_seconds(
    operation,
) -> tuple[float, object]:
    started = perf_counter()
    result = operation()
    elapsed = (
        perf_counter() - started
    )

    return elapsed, result


def test_doctor_scans_100_revisions_within_baseline(
    tmp_path: Path,
) -> None:
    repository = build_repository(
        tmp_path / "repository",
        revision_count=100,
    )

    elapsed, report = elapsed_seconds(
        lambda: (
            AdaptiveAutomationDoctor()
            .inspect(repository)
        )
    )

    assert report.latest_revision == 100
    assert report.profile.revision_count == 100
    assert report.recommendation_count == 10
    assert report.error_count == 0
    assert elapsed < 5.0, (
        "100-revision doctor scan exceeded "
        f"baseline: {elapsed:.6f}s"
    )


def test_doctor_scans_500_revisions_within_baseline(
    tmp_path: Path,
) -> None:
    repository = build_repository(
        tmp_path / "repository",
        revision_count=500,
    )

    elapsed, report = elapsed_seconds(
        lambda: (
            AdaptiveAutomationDoctor()
            .inspect(repository)
        )
    )

    assert report.latest_revision == 500
    assert report.profile.revision_count == 500
    assert report.recommendation_count == 50
    assert report.error_count == 0
    assert elapsed < 15.0, (
        "500-revision doctor scan exceeded "
        f"baseline: {elapsed:.6f}s"
    )


def test_repeated_report_rendering_within_baseline(
    tmp_path: Path,
) -> None:
    repository = build_repository(
        tmp_path / "repository",
        revision_count=100,
    )

    report = AdaptiveAutomationDoctor().inspect(
        repository
    )

    writer = (
        AdaptiveAutomationDoctorReportWriter()
    )
    renderer = (
        AdaptiveAutomationDoctorMarkdownRenderer()
    )

    output_root = tmp_path / "reports"

    def render_repeatedly() -> None:
        for index in range(100):
            json_path = (
                output_root
                / f"doctor-{index:03d}.json"
            )

            writer.write(
                report,
                json_path,
            )

            markdown = renderer.render(
                report
            )

            markdown_path = (
                output_root
                / f"doctor-{index:03d}.md"
            )

            markdown_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            markdown_path.write_text(
                markdown,
                encoding="utf-8",
                newline="\n",
            )

    elapsed, _ = elapsed_seconds(
        render_repeatedly
    )

    json_files = tuple(
        output_root.glob(
            "doctor-*.json"
        )
    )
    markdown_files = tuple(
        output_root.glob(
            "doctor-*.md"
        )
    )

    assert len(json_files) == 100
    assert len(markdown_files) == 100
    assert elapsed < 10.0, (
        "100 repeated report writes exceeded "
        f"baseline: {elapsed:.6f}s"
    )


def test_repeated_doctor_runs_are_stable(
    tmp_path: Path,
) -> None:
    repository = build_repository(
        tmp_path / "repository",
        revision_count=100,
    )

    doctor = AdaptiveAutomationDoctor()

    first = doctor.inspect(
        repository
    ).as_dict()

    started = perf_counter()

    for _ in range(20):
        current = doctor.inspect(
            repository
        ).as_dict()

        assert current == first

    elapsed = (
        perf_counter() - started
    )

    assert elapsed < 10.0, (
        "20 repeated doctor scans exceeded "
        f"baseline: {elapsed:.6f}s"
    )
