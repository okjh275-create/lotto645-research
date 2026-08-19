"""Production champion rollback CLI."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Sequence

from lrp.production.champion_rollback import (
    ChampionRollbackService,
)


def _parser() -> argparse.ArgumentParser:
    """Build the production champion rollback parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Plan or execute a production "
            "champion rollback."
        )
    )

    parser.add_argument(
        "--production-registry",
        type=Path,
        required=True,
        help=(
            "Production champion registry root."
        ),
    )

    parser.add_argument(
        "--revision-id",
        required=True,
        help=(
            "Immutable publication revision "
            "identifier to restore."
        ),
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Execute the rollback. "
            "Without this flag only a plan "
            "is produced."
        ),
    )

    return parser


def run_rollback(
    *,
    production_registry: str | Path,
    revision_id: str,
    execute: bool = False,
) -> dict[str, object]:
    """Plan or execute one verified champion rollback."""

    service = ChampionRollbackService(
        registry_root=production_registry,
    )

    plan = service.plan(
        revision_id
    )

    if not execute:
        return {
            "status": "PASS",
            "mode": "PLAN",
            **asdict(plan),
        }

    result = service.execute(
        plan
    )

    return {
        "status": "PASS",
        "mode": "EXECUTE",
        **asdict(result),
    }


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the production champion rollback CLI."""

    parser = _parser()

    arguments = parser.parse_args(
        argv
    )

    try:
        summary = run_rollback(
            production_registry=(
                arguments.production_registry
            ),
            revision_id=(
                arguments.revision_id
            ),
            execute=(
                arguments.execute
            ),
        )

    except Exception as exc:
        error = {
            "status": "ERROR",
            "error_type": (
                type(exc).__name__
            ),
            "message": str(exc),
        }

        print(
            json.dumps(
                error,
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )

        return 1

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
