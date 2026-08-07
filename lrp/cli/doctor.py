"""Platform diagnostic command."""

from __future__ import annotations

import argparse
import json
import time
from typing import Sequence

from lrp.management import run_doctor


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lrp doctor",
        description=(
            "Run lightweight LRP platform diagnostics."
        ),
    )
    parser.add_argument(
        "--root",
        default=".",
    )
    parser.add_argument(
        "--round-completion",
        action="store_true",
        help=(
            "Verify persisted round-completion "
            "artifacts and manifests."
        ),
    )
    parser.add_argument(
        "--snapshots",
        default="snapshots",
    )
    parser.add_argument(
        "--round-limit",
        type=int,
        default=20,
    )
    arguments = parser.parse_args(argv)

    started = time.perf_counter()

    result = run_doctor(
        project_root=arguments.root,
        round_completion=arguments.round_completion,
        snapshots_root=arguments.snapshots,
        round_limit=arguments.round_limit,
    )
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

    return (
        0
        if result["status"] == "PASS"
        else 1
    )
