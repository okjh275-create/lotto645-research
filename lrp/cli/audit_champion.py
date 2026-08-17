"""Audit active production champion state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from lrp.production import (
    ProductionChampionAudit,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lrp audit-champion",
        description=(
            "Audit the active production "
            "champion registry."
        ),
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
        "--snapshot-root",
        type=Path,
        required=True,
        help=(
            "Production model snapshot root."
        ),
    )

    return parser


def run_audit(
    *,
    production_registry: str | Path,
    snapshot_root: str | Path,
) -> dict[str, object]:
    result = (
        ProductionChampionAudit()
        .audit(
            registry_root=production_registry,
            snapshot_root=snapshot_root,
        )
    )

    return result.as_dict()


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = _parser()

    arguments = parser.parse_args(
        argv
    )

    try:
        summary = run_audit(
            production_registry=(
                arguments.production_registry
            ),
            snapshot_root=(
                arguments.snapshot_root
            ),
        )

        print(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
            )
        )

        if summary["status"] == "FAIL":
            return 1

        return 0

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
                indent=2,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
