from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validation.generate_validation_report import (
    build_parser,
    run,
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


def test_parser_defaults() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--source",
            "source",
            "--output",
            "output",
        ]
    )

    assert args.source == Path("source")
    assert args.output == Path("output")
    assert args.stem == "validation_report"


def test_run_generates_reports(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "reports"

    source_root.mkdir()
    write_replay(source_root)

    exit_code = run(
        [
            "--source",
            str(source_root),
            "--output",
            str(output_root),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0

    payload = json.loads(
        captured.out
    )

    assert payload["status"] == "PASS"
    assert payload["run_count"] == 1
    assert payload["pass_count"] == 1
    assert payload["incomplete_count"] == 0

    assert (
        output_root
        / "validation_report.json"
    ).is_file()

    assert (
        output_root
        / "validation_report.md"
    ).is_file()


def test_run_supports_custom_stem(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "reports"

    source_root.mkdir()
    write_replay(source_root)

    exit_code = run(
        [
            "--source",
            str(source_root),
            "--output",
            str(output_root),
            "--stem",
            "project_h",
        ]
    )

    assert exit_code == 0
    assert (
        output_root / "project_h.json"
    ).is_file()
    assert (
        output_root / "project_h.md"
    ).is_file()


def test_missing_source_exits_with_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        SystemExit,
    ) as error:
        run(
            [
                "--source",
                str(tmp_path / "missing"),
                "--output",
                str(tmp_path / "reports"),
            ]
        )

    assert error.value.code == 2


def test_invalid_stem_exits_with_error(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()

    with pytest.raises(
        SystemExit,
    ) as error:
        run(
            [
                "--source",
                str(source_root),
                "--output",
                str(tmp_path / "reports"),
                "--stem",
                "nested/report",
            ]
        )

    assert error.value.code == 2
