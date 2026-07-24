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
    arguments = parser.parse_args(argv)

    started = time.perf_counter()

    result = run_doctor(
        project_root=arguments.root,
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
