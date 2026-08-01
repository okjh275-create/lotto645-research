"""History export command."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from lrp.io.history_export import (
    export_history,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lrp export-history",
        description=(
            "Export Lotto645 draw history "
            "from SQLite to CSV or JSON."
        ),
    )
    parser.add_argument(
        "--db",
        required=True,
        help="SQLite database path",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "json"),
        default="json",
        dest="file_format",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV or JSON path",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    arguments = _parser().parse_args(argv)

    try:
        output = export_history(
            database=arguments.db,
            output=arguments.output,
            file_format=arguments.file_format,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "ERROR",
                    "error_type": (
                        type(exc).__name__
                    ),
                    "message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            {
                "status": "PASS",
                "format": arguments.file_format,
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0
