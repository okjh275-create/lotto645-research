from __future__ import annotations

import json
from pathlib import Path

import pytest

from lrp.evolution.feedback import (
    AdaptiveAutomationDoctor,
    AdaptiveAutomationDoctorReport,
    AdaptiveAutomationDoctorReportWriter,
    AdaptiveAutomationRepository,
    AdaptiveDoctorReportWriteResult,
)


def doctor_report(
    tmp_path: Path,
) -> AdaptiveAutomationDoctorReport:
    return AdaptiveAutomationDoctor().inspect(
        AdaptiveAutomationRepository(
            tmp_path / "repository"
        )
    )


def test_writer_creates_json_report(
    tmp_path: Path,
) -> None:
    report = doctor_report(tmp_path)
    path = (
        tmp_path
        / "reports"
        / "adaptive_doctor.json"
    )

    result = (
        AdaptiveAutomationDoctorReportWriter()
        .write(
            report,
            path,
        )
    )

    assert isinstance(
        result,
        AdaptiveDoctorReportWriteResult,
    )
    assert result.path == path
    assert result.created is True
    assert result.changed is True
    assert result.byte_count > 0
    assert path.is_file()


def test_written_json_matches_report(
    tmp_path: Path,
) -> None:
    report = doctor_report(tmp_path)
    path = tmp_path / "doctor.json"

    AdaptiveAutomationDoctorReportWriter().write(
        report,
        path,
    )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert payload == report.as_dict()
    assert payload["overall_ok"] is False
    assert payload["latest_revision"] == 0
    assert payload["error_count"] == 0
    assert payload["warning_count"] == 0


def test_output_is_utf8_without_bom_and_has_newline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "doctor.json"

    AdaptiveAutomationDoctorReportWriter().write(
        doctor_report(tmp_path),
        path,
    )

    data = path.read_bytes()

    assert not data.startswith(
        b"\xef\xbb\xbf"
    )
    assert data.endswith(b"\n")
    assert data.decode("utf-8")


def test_serialization_is_deterministic(
    tmp_path: Path,
) -> None:
    report = doctor_report(tmp_path)
    writer = (
        AdaptiveAutomationDoctorReportWriter()
    )

    first = writer.serialize(report)
    second = writer.serialize(report)

    assert first == second

    path = tmp_path / "doctor.json"

    first_result = writer.write(
        report,
        path,
    )
    first_bytes = path.read_bytes()

    second_result = writer.write(
        report,
        path,
    )
    second_bytes = path.read_bytes()

    assert first_result.created is True
    assert first_result.changed is True
    assert second_result.created is False
    assert second_result.changed is False
    assert first_bytes == second_bytes


def test_writer_overwrites_different_existing_content(
    tmp_path: Path,
) -> None:
    path = tmp_path / "doctor.json"
    path.write_text(
        '{"stale": true}\n',
        encoding="utf-8",
    )

    result = (
        AdaptiveAutomationDoctorReportWriter()
        .write(
            doctor_report(tmp_path),
            path,
        )
    )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert result.created is False
    assert result.changed is True
    assert "stale" not in payload
    assert "overall_ok" in payload


def test_parent_directories_are_created(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "nested"
        / "doctor"
        / "report.json"
    )

    AdaptiveAutomationDoctorReportWriter().write(
        doctor_report(tmp_path),
        path,
    )

    assert path.is_file()


def test_json_keys_are_sorted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "doctor.json"

    AdaptiveAutomationDoctorReportWriter().write(
        doctor_report(tmp_path),
        path,
    )

    text = path.read_text(
        encoding="utf-8"
    )

    top_level_lines = [
        line
        for line in text.splitlines()
        if line.startswith('  "')
    ]

    keys = [
        line.split('"', 2)[1]
        for line in top_level_lines
    ]

    assert keys == sorted(keys)


def test_invalid_report_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        TypeError,
        match="AdaptiveAutomationDoctorReport",
    ):
        AdaptiveAutomationDoctorReportWriter().write(
            object(),  # type: ignore[arg-type]
            tmp_path / "doctor.json",
        )


def test_directory_output_is_rejected(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(
        IsADirectoryError,
    ):
        AdaptiveAutomationDoctorReportWriter().write(
            doctor_report(tmp_path),
            output,
        )


def test_public_exports_include_writer() -> None:
    import lrp.evolution.feedback as feedback

    assert (
        "AdaptiveAutomationDoctorReportWriter"
        in feedback.__all__
    )
    assert (
        "AdaptiveDoctorReportWriteResult"
        in feedback.__all__
    )
