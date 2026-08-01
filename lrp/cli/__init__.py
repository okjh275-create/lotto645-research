"""LRP command-line dispatcher."""

from __future__ import annotations

import argparse
from typing import Sequence

from .backup import main as backup_main
from .doctor import main as doctor_main
from .export_history import main as export_history_main
from .predict import main as predict_main
from .restore import main as restore_main
from .review import main as review_main
from .status import main as status_main
from .verify import main as verify_main
from .weekly import main as weekly_main


_COMMANDS = {
    "predict": predict_main,
    "weekly": weekly_main,
    "review": review_main,
    "verify": verify_main,
    "backup": backup_main,
    "restore": restore_main,
    "status": status_main,
    "doctor": doctor_main,
    "export-history": export_history_main,
}


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lrp",
        description="Lotto645 Research Platform",
    )
    subparsers = parser.add_subparsers(
        dest="command"
    )

    for name, help_text in (
        (
            "predict",
            "Generate prediction artifacts",
        ),
        (
            "weekly",
            "Run the weekly prediction workflow",
        ),
        (
            "review",
            "Review a saved prediction",
        ),
        (
            "verify",
            "Verify an artifact manifest",
        ),
        (
            "backup",
            "Create a repository ZIP backup",
        ),
        (
            "restore",
            "Restore a repository ZIP backup",
        ),
        (
            "status",
            "Show lightweight platform status",
        ),
        (
            "doctor",
            "Run lightweight platform diagnostics",
        ),
        (
            "export-history",
            "Export draw history from SQLite",
        ),
    ):
        subparsers.add_parser(
            name,
            help=help_text,
            add_help=False,
        )

    namespace, remaining = parser.parse_known_args(
        argv
    )
    command = _COMMANDS.get(namespace.command)

    if command is None:
        parser.print_help()
        return 0

    return command(remaining)
