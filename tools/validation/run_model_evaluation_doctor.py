"""Command-line doctor for Project M model-evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from tools.validation.model_evaluation_doctor import (
    ModelEvaluationDoctor,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect historical model-evaluation "
            "artifacts for operational readiness."
        )
    )

    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help=(
            "Root containing Project M "
            "evaluation artifacts."
        ),
    )

    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help=(
            "Return exit code 1 when the "
            "doctor reports FAIL."
        ),
    )

    return parser


def run(
    argv: Sequence[str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = (
            ModelEvaluationDoctor()
            .inspect(args.root)
        )
    except (
        FileNotFoundError,
        NotADirectoryError,
    ) as exc:
        payload = {
            "status": "ERROR",
            "overall_ok": False,
            "error": str(exc),
        }

        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            )
        )

        return 2

    print(
        json.dumps(
            result.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    if (
        args.fail_on_issues
        and not result.overall_ok
    ):
        return 1

    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
