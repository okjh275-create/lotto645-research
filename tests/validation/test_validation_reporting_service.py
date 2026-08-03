from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools.validation.validation_reporting_service import (
    ValidationReportingResult,
    ValidationReportingService,
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


def test_generate_writes_both_formats(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "reports"

    source_root.mkdir()
    write_replay(source_root)

    result = (
        ValidationReportingService()
        .generate(
            source_root=source_root,
            output_root=output_root,
            generated_at_utc=datetime(
                2026,
                8,
                4,
                tzinfo=timezone.utc,
            ),
        )
    )

    assert isinstance(
        result,
        ValidationReportingResult,
    )
    assert result.json_path.is_file()
    assert result.markdown_path.is_file()
    assert result.report.summary.run_count == 1


def test_generate_uses_custom_stem(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    write_replay(source_root)

    result = (
        ValidationReportingService()
        .generate(
            source_root=source_root,
            output_root=tmp_path / "reports",
            stem="project_h",
        )
    )

    assert result.json_path.name == (
        "project_h.json"
    )
    assert result.markdown_path.name == (
        "project_h.md"
    )


def test_result_serialization(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    write_replay(source_root)

    result = (
        ValidationReportingService()
        .generate(
            source_root=source_root,
            output_root=tmp_path / "reports",
        )
    )

    payload = result.as_dict()

    assert payload["report"]["summary"][
        "run_count"
    ] == 1
    assert payload["json_path"].endswith(
        "validation_report.json"
    )
    assert payload["markdown_path"].endswith(
        "validation_report.md"
    )


@pytest.mark.parametrize(
    "stem",
    [
        "",
        " ",
        ".",
        "..",
        "nested/report",
        r"nested\report",
    ],
)
def test_invalid_stem_is_rejected(
    tmp_path: Path,
    stem: str,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()

    with pytest.raises(
        ValueError,
    ):
        ValidationReportingService().generate(
            source_root=source_root,
            output_root=tmp_path / "reports",
            stem=stem,
        )


def test_non_string_stem_is_rejected(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()

    with pytest.raises(
        TypeError,
        match="stem must be a string",
    ):
        ValidationReportingService().generate(
            source_root=source_root,
            output_root=tmp_path / "reports",
            stem=123,  # type: ignore[arg-type]
        )
