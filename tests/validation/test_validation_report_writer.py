from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools.validation.validation_report_writer import (
    ValidationReportWriter,
)


def write_replay(
    root: Path,
) -> None:
    run_root = root / "replay_100_109"

    run_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        run_root / "replay_summary.json"
    ).write_text(
        json.dumps(
            {
                "config": {
                    "start_round": 100,
                    "end_round": 109,
                },
                "summary": {
                    "round_count": 10,
                },
            }
        ),
        encoding="utf-8",
    )

    (
        run_root / "replay_rounds.jsonl"
    ).write_text(
        "{}\n",
        encoding="utf-8",
    )


def test_build_discovers_runs(
    tmp_path: Path,
) -> None:
    write_replay(tmp_path)

    report = ValidationReportWriter().build(
        source_root=tmp_path,
        generated_at_utc=datetime(
            2026,
            8,
            4,
            tzinfo=timezone.utc,
        ),
    )

    assert report.summary.run_count == 1
    assert report.summary.replay_count == 1
    assert report.summary.pass_count == 1


def test_write_json(
    tmp_path: Path,
) -> None:
    write_replay(tmp_path)

    writer = ValidationReportWriter()

    report = writer.build(
        source_root=tmp_path,
        generated_at_utc=datetime(
            2026,
            8,
            4,
            tzinfo=timezone.utc,
        ),
    )

    output = writer.write_json(
        report=report,
        output=(
            tmp_path
            / "reports"
            / "validation_report.json"
        ),
    )

    payload = json.loads(
        output.read_text(
            encoding="utf-8"
        )
    )

    assert output.is_file()
    assert payload["schema_version"] == 1
    assert payload["summary"][
        "run_count"
    ] == 1


def test_build_and_write_json(
    tmp_path: Path,
) -> None:
    write_replay(tmp_path)

    report, output = (
        ValidationReportWriter()
        .build_and_write_json(
            source_root=tmp_path,
            output=(
                tmp_path
                / "validation_report.json"
            ),
            generated_at_utc=datetime(
                2026,
                8,
                4,
                tzinfo=timezone.utc,
            ),
        )
    )

    assert report.summary.run_count == 1
    assert output.is_file()


def test_json_is_deterministic(
    tmp_path: Path,
) -> None:
    write_replay(tmp_path)

    writer = ValidationReportWriter()

    timestamp = datetime(
        2026,
        8,
        4,
        tzinfo=timezone.utc,
    )

    report = writer.build(
        source_root=tmp_path,
        generated_at_utc=timestamp,
    )

    first = writer.write_json(
        report=report,
        output=tmp_path / "first.json",
    )
    second = writer.write_json(
        report=report,
        output=tmp_path / "second.json",
    )

    assert first.read_bytes() == (
        second.read_bytes()
    )


def test_invalid_report_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        TypeError,
        match="ValidationReport",
    ):
        ValidationReportWriter().write_json(
            report=object(),  # type: ignore[arg-type]
            output=tmp_path / "report.json",
        )


def test_output_directory_is_rejected(
    tmp_path: Path,
) -> None:
    write_replay(tmp_path)

    writer = ValidationReportWriter()

    report = writer.build(
        source_root=tmp_path,
    )

    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(
        IsADirectoryError,
    ):
        writer.write_json(
            report=report,
            output=output,
        )


def test_json_report_preserves_regime_learning_provenance(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "replay_100_109"
    run_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        run_root / "replay_summary.json"
    ).write_text(
        json.dumps(
            {
                "config": {
                    "start_round": 100,
                    "end_round": 109,
                },
                "summary": {
                    "round_count": 10,
                    "regime_calibration_applied_count": 8,
                    "final_regime_calibration_revision": 7,
                    "final_regime_calibration_sample_size": 70,
                    "regime_bayesian_applied_count": 8,
                    "final_regime_bayesian_revision": 7,
                    "final_regime_bayesian_sample_size": 70,
                },
            }
        ),
        encoding="utf-8",
    )

    (
        run_root / "replay_rounds.jsonl"
    ).write_text(
        "{}\n",
        encoding="utf-8",
    )

    writer = ValidationReportWriter()

    report = writer.build(
        source_root=tmp_path,
        generated_at_utc=datetime(
            2026,
            8,
            11,
            tzinfo=timezone.utc,
        ),
    )

    output = writer.write_json(
        report=report,
        output=tmp_path / "validation_report.json",
    )

    payload = json.loads(
        output.read_text(
            encoding="utf-8"
        )
    )

    run = payload["runs"][0]

    assert (
        run["regime_calibration_applied_count"]
        == 8
    )
    assert (
        run["regime_calibration_latest_revision"]
        == 7
    )
    assert (
        run[
            "regime_calibration_latest_sample_size"
        ]
        == 70
    )

    assert (
        run["regime_bayesian_applied_count"]
        == 8
    )
    assert (
        run["regime_bayesian_latest_revision"]
        == 7
    )
    assert (
        run[
            "regime_bayesian_latest_sample_size"
        ]
        == 70
    )
