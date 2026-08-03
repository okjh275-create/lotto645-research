from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools.validation.validation_report_models import (
    ValidationReportBuilder,
)
from tools.validation.validation_run_discovery import (
    ValidationRunRecord,
)


def make_run(
    tmp_path: Path,
    *,
    run_type: str = "replay",
    start_round: int = 100,
    end_round: int = 109,
    status: str = "PASS",
    suffix: str = "a",
) -> ValidationRunRecord:
    return ValidationRunRecord(
        run_type=run_type,
        root=tmp_path / suffix,
        start_round=start_round,
        end_round=end_round,
        round_count=(
            end_round - start_round + 1
        ),
        policy_name=None,
        files={},
        missing_files=(
            ()
            if status == "PASS"
            else ("missing.json",)
        ),
        status=status,
    )


def test_builds_empty_report(
    tmp_path: Path,
) -> None:
    report = ValidationReportBuilder().build(
        source_root=tmp_path,
        runs=(),
        generated_at_utc=datetime(
            2026,
            8,
            4,
            tzinfo=timezone.utc,
        ),
    )

    assert report.summary.run_count == 0
    assert (
        report.summary.earliest_round
        is None
    )
    assert (
        report.summary.latest_round
        is None
    )
    assert report.duplicate_windows == ()


def test_builds_summary_counts(
    tmp_path: Path,
) -> None:
    runs = (
        make_run(
            tmp_path,
            run_type="replay",
            start_round=100,
            end_round=109,
            suffix="replay",
        ),
        make_run(
            tmp_path,
            run_type="policy_comparison",
            start_round=110,
            end_round=119,
            status="INCOMPLETE",
            suffix="policy",
        ),
    )

    report = ValidationReportBuilder().build(
        source_root=tmp_path,
        runs=runs,
        generated_at_utc=datetime(
            2026,
            8,
            4,
            tzinfo=timezone.utc,
        ),
    )

    summary = report.summary

    assert summary.run_count == 2
    assert summary.replay_count == 1
    assert (
        summary.policy_comparison_count
        == 1
    )
    assert summary.pass_count == 1
    assert summary.incomplete_count == 1
    assert summary.earliest_round == 100
    assert summary.latest_round == 119


def test_duplicate_windows_are_detected(
    tmp_path: Path,
) -> None:
    runs = (
        make_run(
            tmp_path,
            suffix="first",
        ),
        make_run(
            tmp_path,
            suffix="second",
        ),
    )

    report = ValidationReportBuilder().build(
        source_root=tmp_path,
        runs=runs,
    )

    assert report.duplicate_windows == (
        (100, 109, "replay"),
    )
    assert (
        report.summary
        .duplicate_window_count
        == 1
    )


def test_different_run_types_are_not_duplicates(
    tmp_path: Path,
) -> None:
    runs = (
        make_run(
            tmp_path,
            run_type="replay",
            suffix="first",
        ),
        make_run(
            tmp_path,
            run_type="policy_comparison",
            suffix="second",
        ),
    )

    report = ValidationReportBuilder().build(
        source_root=tmp_path,
        runs=runs,
    )

    assert report.duplicate_windows == ()


def test_runs_are_sorted_deterministically(
    tmp_path: Path,
) -> None:
    runs = (
        make_run(
            tmp_path,
            start_round=200,
            end_round=209,
            suffix="later",
        ),
        make_run(
            tmp_path,
            start_round=100,
            end_round=109,
            suffix="earlier",
        ),
    )

    report = ValidationReportBuilder().build(
        source_root=tmp_path,
        runs=runs,
    )

    assert tuple(
        run.start_round
        for run in report.runs
    ) == (
        100,
        200,
    )


def test_report_serialization(
    tmp_path: Path,
) -> None:
    report = ValidationReportBuilder().build(
        source_root=tmp_path,
        runs=(
            make_run(tmp_path),
        ),
        generated_at_utc=datetime(
            2026,
            8,
            4,
            1,
            2,
            3,
            tzinfo=timezone.utc,
        ),
    )

    payload = report.as_dict()

    assert payload["schema_version"] == 1
    assert payload["source_root"] == str(
        tmp_path
    )
    assert payload[
        "generated_at_utc"
    ] == "2026-08-04T01:02:03+00:00"
    assert payload["summary"][
        "run_count"
    ] == 1
    assert len(payload["runs"]) == 1


def test_naive_timestamp_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        ValidationReportBuilder().build(
            source_root=tmp_path,
            runs=(),
            generated_at_utc=datetime(
                2026,
                8,
                4,
            ),
        )


def test_invalid_run_value_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        TypeError,
        match="ValidationRunRecord",
    ):
        ValidationReportBuilder().build(
            source_root=tmp_path,
            runs=(
                object(),  # type: ignore[arg-type]
            ),
        )
