"""LRP command-line dispatcher."""

from __future__ import annotations

import argparse
from typing import Sequence

from .production_lifecycle import main as production_lifecycle_main
from .audit_champion import main as audit_champion_main
from .backup import main as backup_main
from .doctor import main as doctor_main
from .export_history import main as export_history_main
from .model_evaluation import main as model_evaluation_main
from .predict import main as predict_main
from .publish_champion import (
    main as publish_champion_main,
)
from .rollback_champion import main as rollback_champion_main
from .restore import main as restore_main
from .review import main as review_main
from .round_complete import main as round_complete_main
from .status import main as status_main
from .verify import main as verify_main
from .weekly import main as weekly_main


_COMMANDS = {
    "model-evaluation": model_evaluation_main,
    "audit-champion": audit_champion_main,
    "predict": predict_main,
    "weekly": weekly_main,
    "review": review_main,
    "round-complete": round_complete_main,
    "verify": verify_main,
    "backup": backup_main,
    "restore": restore_main,
    "status": status_main,
    "doctor": doctor_main,
    "export-history": export_history_main,
    "publish-champion": publish_champion_main,
    "rollback-champion": rollback_champion_main,
    "production-lifecycle": production_lifecycle_main,
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
            "model-evaluation",
            "Run historical model evaluation",
        ),
        (
            "audit-champion",
            "Audit active production champion state",
        ),
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
            "round-complete",
            "Complete review and learning for a draw",
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
    (
        "publish-champion",
        "Publish evaluated champion to production registry",
    ),
        (
            "rollback-champion",
            "Plan or execute production champion rollback",
        ),
        (
            "production-lifecycle",
            "Run production release lifecycle orchestration",
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
