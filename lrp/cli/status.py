"""Platform status command."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from lrp.management import collect_platform_status
from lrp.operations import (
    RoundCompletionRepository,
    summarize_round_completions,
)


def _round_completion_root(
    *,
    project_root: str | Path,
    snapshots_root: str | Path,
) -> Path:
    root = Path(project_root).resolve()
    snapshots = Path(snapshots_root)

    if snapshots.is_absolute():
        return (
            snapshots
            / "round-completion"
        )

    return (
        root
        / snapshots
        / "round-completion"
    )


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lrp status",
        description="Show lightweight LRP platform status.",
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--predictions", default="predictions")
    parser.add_argument("--snapshots", default="snapshots")
    parser.add_argument("--backups", default="backups")
    parser.add_argument(
        "--round-completion",
        action="store_true",
        help=(
            "Include persisted round-completion "
            "operational summary."
        ),
    )
    parser.add_argument(
        "--round-limit",
        type=int,
        default=20,
        help=(
            "Number of recent round-completion "
            "records to summarize. Default: 20"
        ),
    )

    arguments = parser.parse_args(argv)

    if arguments.round_limit < 1:
        parser.error(
            "--round-limit must be greater than "
            "or equal to 1"
        )

    started = time.perf_counter()

    result = collect_platform_status(
        project_root=arguments.root,
        predictions_root=arguments.predictions,
        snapshots_root=arguments.snapshots,
        backups_root=arguments.backups,
    )

    if arguments.round_completion:
        repository = RoundCompletionRepository(
            _round_completion_root(
                project_root=arguments.root,
                snapshots_root=arguments.snapshots,
            )
        )

        summary = summarize_round_completions(
            repository,
            limit=arguments.round_limit,
        )

        result["round_completion"] = {
            **summary.as_dict(),
            "limit": arguments.round_limit,
            "root": str(
                repository.root.resolve()
            ),
        }

    result["elapsed_seconds"] = round(
        time.perf_counter() - started,
        6,
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0 if result["status"] == "PASS" else 1
