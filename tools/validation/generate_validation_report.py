"""Command-line entry point for validation reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from tools.validation.validation_reporting_service import (
    ValidationReportingService,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="validation-report",
        description=(
            "Discover validation runs and generate "
            "JSON and Markdown reports."
        ),
    )

    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help=(
            "Root directory containing validation "
            "run artifacts."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help=(
            "Directory where report files will "
            "be written."
        ),
    )
    parser.add_argument(
        "--stem",
        default="validation_report",
        help=(
            "Output file stem. Defaults to "
            "'validation_report'."
        ),
    )

    return parser


def run(
    argv: Sequence[str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    service = ValidationReportingService()

    try:
        result = service.generate(
            source_root=args.source,
            output_root=args.output,
            stem=args.stem,
        )
    except (
        FileNotFoundError,
        NotADirectoryError,
        IsADirectoryError,
        TypeError,
        ValueError,
    ) as exc:
        parser.exit(
            status=2,
            message=f"error: {exc}\n",
        )

    payload = {
        "status": "PASS",
        "run_count": (
            result.report.summary.run_count
        ),
        "pass_count": (
            result.report.summary.pass_count
        ),
        "incomplete_count": (
            result.report.summary
            .incomplete_count
        ),
        "json_path": str(
            result.json_path
        ),
        "markdown_path": str(
            result.markdown_path
        ),
    }

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
