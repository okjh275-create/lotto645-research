"""Review command."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from lrp.operations import review_prediction, write_operation_artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lrp review")
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--numbers", nargs=6, required=True, type=int)
    parser.add_argument("--bonus", type=int)
    parser.add_argument("--output", default="snapshots")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        payload = review_prediction(
            arguments.prediction,
            winning_numbers=arguments.numbers,
            bonus=arguments.bonus,
        )
        artifact = write_operation_artifact(
            payload,
            output_root=arguments.output,
            artifact_type="reviews",
            round_no=int(payload["round"]),
            filename="review.json",
        )
        print(json.dumps({"status": "PASS", "summary": payload["summary"], "artifact": artifact}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
