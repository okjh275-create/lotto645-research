"""LRP command-line dispatcher."""

from __future__ import annotations

import argparse
from typing import Sequence

from .predict import main as predict_main


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lrp",
        description="Lotto645 Research Platform",
    )

    subparsers = parser.add_subparsers(
        dest="command",
    )

    subparsers.add_parser(
        "predict",
        help="Generate prediction artifacts",
        add_help=False,
    )

    namespace, remaining = parser.parse_known_args(argv)

    if namespace.command == "predict":
        return predict_main(remaining)

    parser.print_help()
    return 0
