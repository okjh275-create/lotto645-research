"""Platform status command."""

from __future__ import annotations

import argparse
import json
import time
from typing import Sequence

from lrp.management import collect_platform_status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lrp status",
        description="Show lightweight LRP platform status.",
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--predictions", default="predictions")
    parser.add_argument("--snapshots", default="snapshots")
    parser.add_argument("--backups", default="backups")
    arguments = parser.parse_args(argv)

    started = time.perf_counter()

    result = collect_platform_status(
        project_root=arguments.root,
        predictions_root=arguments.predictions,
        snapshots_root=arguments.snapshots,
        backups_root=arguments.backups,
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
    return 0 if result["status"] == "PASS" else 1
