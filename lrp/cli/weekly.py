"""Weekly prediction command with operation snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from lrp.cli.predict import run_predict
from lrp.operations import write_operation_artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lrp weekly")
    parser.add_argument("--history", required=True)
    parser.add_argument("--round", dest="round_no", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--candidate-count", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--practical-k", type=int, default=5)
    parser.add_argument("--long-gap-window", type=int, default=5)
    parser.add_argument("--mode", choices=("fast", "full"), default="fast")
    parser.add_argument("--output", default="predictions")
    parser.add_argument("--snapshots", default="snapshots")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    arguments.print_json = False
    try:
        summary = run_predict(arguments)
        snapshot_payload = {
            "schema_version": "1.0",
            "artifact_type": "weekly_run",
            "round": summary["round"],
            "seed": summary["seed"],
            "mode": summary["mode"],
            "history_draws": summary["history_draws"],
            "generated_candidates": summary["generated_candidates"],
            "selected_sets": summary["selected_sets"],
            "top5_practical": summary["top5_practical"],
            "diversity": summary["diversity"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "prediction_artifact": summary["artifact"],
        }
        operation_artifact = write_operation_artifact(
            snapshot_payload,
            output_root=arguments.snapshots,
            artifact_type="weekly",
            round_no=arguments.round_no,
            filename="weekly.json",
        )
        public = {key: value for key, value in summary.items() if key != "payload"}
        public["weekly_snapshot"] = operation_artifact
        print(json.dumps(public, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
