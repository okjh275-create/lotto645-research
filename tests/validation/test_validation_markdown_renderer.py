from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools.validation.validation_markdown_renderer import (
    ValidationMarkdownRenderer,
)
from tools.validation.validation_report_models import (
    ValidationReportBuilder,
)
from tools.validation.validation_run_discovery import (
    ValidationRunRecord,
)


def make_report(
    tmp_path: Path,
    *,
    status: str = "PASS",
):
    run = ValidationRunRecord(
        run_type="replay",
        root=tmp_path / "replay_100_109",
        start_round=100,
        end_round=109,
        round_count=10,
        policy_name=None,
        files={},
        missing_files=(
            ()
            if status == "PASS"
            else ("replay_rounds.jsonl",)
        ),
        status=status,
    )

    return ValidationReportBuilder().build(
        source_root=tmp_path,
        runs=(run,),
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


def test_render_contains_summary(
    tmp_path: Path,
) -> None:
    report = make_report(tmp_path)

    text = ValidationMarkdownRenderer().render(
        report
    )

    assert "# Validation Report" in text
    assert "- Total runs: 1" in text
    assert "- Replay runs: 1" in text
    assert "- Complete runs: 1" in text
    assert "100–109" in text


def test_render_contains_incomplete_run(
    tmp_path: Path,
) -> None:
    report = make_report(
        tmp_path,
        status="INCOMPLETE",
    )

    text = ValidationMarkdownRenderer().render(
        report
    )

    assert "INCOMPLETE" in text
    assert "replay_rounds.jsonl" in text


def test_render_empty_report(
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

    text = ValidationMarkdownRenderer().render(
        report
    )

    assert (
        "No validation runs were discovered."
        in text
    )
    assert (
        "No duplicate validation windows "
        "were found."
        in text
    )


def test_render_is_deterministic(
    tmp_path: Path,
) -> None:
    report = make_report(tmp_path)
    renderer = ValidationMarkdownRenderer()

    assert renderer.render(report) == (
        renderer.render(report)
    )


def test_write_markdown(
    tmp_path: Path,
) -> None:
    report = make_report(tmp_path)

    output = (
        ValidationMarkdownRenderer()
        .write(
            report=report,
            output=(
                tmp_path
                / "reports"
                / "validation_report.md"
            ),
        )
    )

    assert output.is_file()
    assert output.read_text(
        encoding="utf-8"
    ).startswith("# Validation Report")


def test_invalid_report_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="ValidationReport",
    ):
        ValidationMarkdownRenderer().render(
            object(),  # type: ignore[arg-type]
        )


def test_output_directory_is_rejected(
    tmp_path: Path,
) -> None:
    report = make_report(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(
        IsADirectoryError,
    ):
        ValidationMarkdownRenderer().write(
            report=report,
            output=output,
        )
