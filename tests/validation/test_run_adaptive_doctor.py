from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validation.run_adaptive_doctor import (
    build_parser,
    run,
)


def write_profile(
    repository: Path,
    *,
    revision: int,
) -> None:
    path = (
        repository
        / "profiles"
        / f"revision-{revision:08d}.json"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": 1,
        "source_revision": revision - 1,
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

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_parser_reads_arguments() -> None:
    args = build_parser().parse_args(
        [
            "--repository",
            "repository",
            "--output",
            "reports",
            "--stem",
            "doctor",
        ]
    )

    assert args.repository == Path(
        "repository"
    )
    assert args.output == Path(
        "reports"
    )
    assert args.stem == "doctor"
    assert args.fail_on_issues is False


def test_empty_repository_generates_reports(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = (
        tmp_path / "repository"
    )
    output = tmp_path / "reports"

    exit_code = run(
        [
            "--repository",
            str(repository),
            "--output",
            str(output),
            "--stem",
            "adaptive_doctor",
        ]
    )

    payload = json.loads(
        capsys.readouterr().out
    )

    assert exit_code == 0
    assert payload["status"] == "FAIL"
    assert payload["overall_ok"] is False
    assert payload["latest_revision"] == 0
    assert payload["json_created"] is True
    assert payload[
        "markdown_created"
    ] is True

    assert (
        output
        / "adaptive_doctor.json"
    ).is_file()

    assert (
        output
        / "adaptive_doctor.md"
    ).is_file()


def test_healthy_repository_passes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = (
        tmp_path / "repository"
    )
    output = tmp_path / "reports"

    write_profile(
        repository,
        revision=12,
    )

    exit_code = run(
        [
            "--repository",
            str(repository),
            "--output",
            str(output),
            "--stem",
            "healthy",
        ]
    )

    payload = json.loads(
        capsys.readouterr().out
    )

    assert exit_code == 0
    assert payload["status"] == "PASS"
    assert payload["overall_ok"] is True
    assert payload["latest_revision"] == 12
    assert payload["error_count"] == 0


def test_fail_on_issues_returns_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "reports"

    exit_code = run(
        [
            "--repository",
            str(tmp_path / "empty"),
            "--output",
            str(output),
            "--stem",
            "failed",
            "--fail-on-issues",
        ]
    )

    payload = json.loads(
        capsys.readouterr().out
    )

    assert exit_code == 1
    assert payload["status"] == "FAIL"

    assert (
        output / "failed.json"
    ).is_file()

    assert (
        output / "failed.md"
    ).is_file()


def test_repeat_run_is_idempotent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = (
        tmp_path / "repository"
    )
    output = tmp_path / "reports"

    write_profile(
        repository,
        revision=12,
    )

    arguments = [
        "--repository",
        str(repository),
        "--output",
        str(output),
        "--stem",
        "doctor",
    ]

    assert run(arguments) == 0
    first_payload = json.loads(
        capsys.readouterr().out
    )

    assert run(arguments) == 0
    second_payload = json.loads(
        capsys.readouterr().out
    )

    assert first_payload[
        "json_created"
    ] is True

    assert second_payload[
        "json_created"
    ] is False

    assert second_payload[
        "json_changed"
    ] is False

    assert second_payload[
        "markdown_created"
    ] is False

    assert second_payload[
        "markdown_changed"
    ] is False


def test_invalid_stem_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        SystemExit,
    ) as error:
        run(
            [
                "--repository",
                str(tmp_path / "repository"),
                "--output",
                str(tmp_path / "reports"),
                "--stem",
                "../invalid",
            ]
        )

    assert error.value.code == 2


def test_output_file_is_rejected(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.write_text(
        "not a directory",
        encoding="utf-8",
    )

    with pytest.raises(
        SystemExit,
    ) as error:
        run(
            [
                "--repository",
                str(tmp_path / "repository"),
                "--output",
                str(output),
                "--stem",
                "doctor",
            ]
        )

    assert error.value.code == 2


def test_markdown_contains_expected_sections(
    tmp_path: Path,
) -> None:
    repository = (
        tmp_path / "repository"
    )
    output = tmp_path / "reports"

    write_profile(
        repository,
        revision=12,
    )

    assert run(
        [
            "--repository",
            str(repository),
            "--output",
            str(output),
            "--stem",
            "doctor",
        ]
    ) == 0

    text = (
        output / "doctor.md"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "# Adaptive Automation Doctor Report"
        in text
    )
    assert "## Summary" in text
    assert "## Conclusion" in text
    assert text.endswith("\n")


def test_json_report_matches_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = (
        tmp_path / "repository"
    )
    output = tmp_path / "reports"

    write_profile(
        repository,
        revision=12,
    )

    assert run(
        [
            "--repository",
            str(repository),
            "--output",
            str(output),
            "--stem",
            "doctor",
        ]
    ) == 0

    summary = json.loads(
        capsys.readouterr().out
    )

    report = json.loads(
        (
            output / "doctor.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert report[
        "overall_ok"
    ] == summary["overall_ok"]

    assert report[
        "latest_revision"
    ] == summary["latest_revision"]
