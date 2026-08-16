"""Production champion publication CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from lrp.production import (
    ProductionChampionRegistryPublisher,
)


def _parser() -> argparse.ArgumentParser:
    """Build the champion publication parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Publish a persisted champion decision "
            "to the active production registry."
        )
    )

    parser.add_argument(
        "--champion-decision",
        type=Path,
        required=True,
        help=(
            "Persisted champion decision artifact "
            "to publish."
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

    return parser


def run_publish(
    *,
    champion_decision: str | Path,
    production_registry: str | Path,
) -> dict[str, object]:
    """Publish a champion decision and return its summary."""

    result = (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=champion_decision,
            registry_root=production_registry,
        )
    )

    summary = result.as_dict()

    return {
        "status": "PASS",
        **summary,
    }


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the production champion publication CLI."""

    parser = _parser()

    arguments = parser.parse_args(
        argv
    )

    try:
        summary = run_publish(
            champion_decision=(
                arguments.champion_decision
            ),
            production_registry=(
                arguments.production_registry
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