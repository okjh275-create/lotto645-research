"""Command-line prediction application."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Sequence

from lrp.io import (
    history_until_round,
    load_history,
    long_gap_numbers,
    previous_numbers,
    to_statistics_draws,
    write_prediction_artifacts,
)
from lrp.pipelines import (
    PredictionPipeline,
    PredictionRequest,
    prediction_to_dict,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lrp predict",
        description=(
            "Generate reproducible Lotto645 prediction artifacts."
        ),
    )

    parser.add_argument(
        "--history",
        required=True,
        help="CSV or JSON draw-history file",
    )
    parser.add_argument(
        "--round",
        dest="round_no",
        required=True,
        type=int,
        help="Target prediction round",
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        help="Deterministic random seed",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=10_000,
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
        "--long-gap-window",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--mode",
        choices=("fast", "full"),
        default="fast",
        help=(
            "fast keeps the same statistics snapshot while reducing "
            "uncertainty/backtest overhead"
        ),
    )
    parser.add_argument(
        "--output",
        default="predictions",
        help="Prediction artifact root directory",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print full prediction JSON",
    )

    return parser


def _analysis_config(
    pipeline: PredictionPipeline,
    *,
    mode: str,
    draw_count: int,
) -> object:
    if mode == "full":
        iterations = 1000
    else:
        iterations = 100

    return pipeline.statistics.create_config(
        short_window=10,
        mid_window=20,
        long_window=50,
        bootstrap_iterations=iterations,
        confidence_level=0.95,
        seed=20260719,
        top_n=10,
        backtest_minimum_history=max(
            50,
            draw_count + 1,
        ),
        serialization_precision=8,
    )


def run_predict(
    arguments: argparse.Namespace,
) -> dict[str, object]:
    started = time.perf_counter()

    all_history = load_history(arguments.history)
    bounded_history = history_until_round(
        all_history,
        target_round=arguments.round_no,
    )

    pipeline = PredictionPipeline.load()
    draw_type = getattr(
        pipeline.statistics.module,
        "DrawRecord",
    )

    draws = to_statistics_draws(
        bounded_history,
        draw_type=draw_type,
    )

    request = PredictionRequest(
        round_no=arguments.round_no,
        seed=arguments.seed,
        temperature=arguments.temperature,
        candidate_count=arguments.candidate_count,
        max_attempts_multiplier=50,
        top_k=arguments.top_k,
        practical_k=arguments.practical_k,
        previous_numbers=previous_numbers(
            bounded_history
        ),
        long_gap_numbers=long_gap_numbers(
            bounded_history,
            recent_draw_count=(
                arguments.long_gap_window
            ),
        ),
    )

    analysis_config = _analysis_config(
        pipeline,
        mode=arguments.mode,
        draw_count=len(draws),
    )

    result = pipeline.run(
        draws,
        request,
        analysis_config=analysis_config,
    )

    payload = prediction_to_dict(result)

    artifact = write_prediction_artifacts(
        payload,
        output_root=arguments.output,
    )

    elapsed = time.perf_counter() - started

    return {
        "status": "PASS",
        "round": arguments.round_no,
        "seed": arguments.seed,
        "mode": arguments.mode,
        "history_draws": len(draws),
        "generated_candidates": (
            result.generated_count
        ),
        "selected_sets": len(payload["sets"]),
        "top5_practical": payload[
            "top5_practical"
        ],
        "diversity": payload["diversity"],
        "elapsed_seconds": round(elapsed, 3),
        "artifact": artifact,
        "payload": payload,
    }


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)

    try:
        summary = run_predict(arguments)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "ERROR",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    if arguments.print_json:
        print(
            json.dumps(
                summary["payload"],
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        public_summary = {
            key: value
            for key, value in summary.items()
            if key != "payload"
        }
        print(
            json.dumps(
                public_summary,
                ensure_ascii=False,
                indent=2,
            )
        )

    return 0
