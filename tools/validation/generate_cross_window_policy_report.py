"""Command-line entry point for cross-window policy reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from tools.validation.cross_window_policy_reporting_service import (
    CrossWindowPolicyReportingService,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cross-window-policy-report",
        description=(
            "Aggregate policy comparison reports "
            "across non-overlapping validation windows."
        ),
    )

    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help=(
            "Root directory containing "
            "policy_comparison.json files."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help=(
            "Directory where JSON and Markdown "
            "reports will be written."
        ),
    )
    parser.add_argument(
        "--stem",
        default="cross_window_policy_report",
        help=(
            "Output file stem. Defaults to "
            "'cross_window_policy_report'."
        ),
    )

    return parser


def run(
    argv: Sequence[str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = (
            CrossWindowPolicyReportingService()
            .discover_and_generate(
                source_root=args.source,
                output_root=args.output,
                stem=args.stem,
            )
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

    ranking = result.report["ranking"]

    winner = (
        ranking[0]["policy_name"]
        if ranking
        else None
    )

    payload = {
        "status": "PASS",
        "window_count": (
            result.report["window_count"]
        ),
        "total_round_count": (
            result.report[
                "total_round_count"
            ]
        ),
        "policy_count": (
            result.report["policy_count"]
        ),
        "winner": winner,
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
