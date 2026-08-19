"""Production release lifecycle orchestration command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from lrp.production.production_lifecycle import (
    ProductionLifecycleRequest,
    ProductionLifecycleService,
)
from lrp.production.production_lifecycle_adapters import (
    run_audit_stage,
    run_model_evaluation_stage,
    run_prediction_stage,
    run_publication_stage,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the production lifecycle command parser."""

    parser = argparse.ArgumentParser(
        prog="python -m lrp production-lifecycle",
        description=(
            "Run the production lifecycle orchestration."
        ),
    )

    parser.add_argument(
        "--history",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--evaluation-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--production-registry",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--production-snapshot-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--prediction-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--round",
        dest="round_no",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--practical-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--mode",
        default="fast",
    )
    parser.add_argument(
        "--evaluation-start-round",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--evaluation-end-round",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--long-gap-window",
        type=int,
        default=5,
    )

    return parser


def _build_request(
    args: argparse.Namespace,
) -> ProductionLifecycleRequest:
    """Map CLI arguments to the lifecycle request."""

    return ProductionLifecycleRequest(
        history_path=args.history,
        evaluation_output_root=(
            args.evaluation_output
        ),
        production_registry_root=(
            args.production_registry
        ),
        production_snapshot_root=(
            args.production_snapshot_root
        ),
        prediction_output_root=(
            args.prediction_output
        ),
        round_no=args.round_no,
        seed=args.seed,
        temperature=args.temperature,
        candidate_count=args.candidate_count,
        top_k=args.top_k,
        practical_k=args.practical_k,
        mode=args.mode,
        evaluation_start_round=(
            args.evaluation_start_round
        ),
        evaluation_end_round=(
            args.evaluation_end_round
        ),
        long_gap_window=(
            args.long_gap_window
        ),
    )


def _stage_to_dict(
    stage: object,
) -> dict[str, object]:
    """Serialize a lifecycle stage without changing its contract."""

    if isinstance(stage, dict):
        return dict(stage)

    payload = getattr(
        stage,
        "__dict__",
        None,
    )

    if isinstance(payload, dict):
        return dict(payload)

    return {
        "value": str(stage),
    }


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the production lifecycle orchestration."""

    parser = build_parser()
    args = parser.parse_args(argv)

    request = _build_request(args)

    service = ProductionLifecycleService(
        model_evaluation=(
            run_model_evaluation_stage
        ),
        publication=run_publication_stage,
        audit=run_audit_stage,
        prediction=run_prediction_stage,
    )

    result = service.run(
        request
    )

    payload = {
        "status": result.status,
        "stages": [
            _stage_to_dict(stage)
            for stage in result.stages
        ],
    }

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    return (
        1
        if result.status == "ERROR"
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
