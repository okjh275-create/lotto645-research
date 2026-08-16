"""Command-line entry point for historical model evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from lrp.contracts import ContractError
from lrp.evaluation import EvaluationWindow
from lrp.io import HistoryRow, load_history

from tools.validation.historical_replay_models import (
    ReplayConfig,
)
from tools.validation.model_evaluation_orchestration_service import (
    HistoricalModelEvaluationOrchestrationService,
)


DEFAULT_MODELS = (
    "baseline",
    "calibration",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-evaluation",
        description=(
            "Run historical model evaluation and "
            "write champion decision artifacts."
        ),
    )

    parser.add_argument(
        "--history",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--replay-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--start-round",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--end-round",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
    )

    parser.add_argument(
        "--seed-base",
        type=int,
        default=20260802,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
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
        "--confidence",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--mode",
        default="fast",
    )

    return parser


def _load_history(
    path: Path,
    parser: argparse.ArgumentParser,
) -> tuple[HistoryRow, ...]:
    if not path.is_file():
        parser.error(
            f"history file not found: {path}"
        )

    try:
        return load_history(path)
    except (
        OSError,
        ContractError,
    ) as exc:
        parser.error(
            f"failed to read history: {exc}"
        )


def _build_windows(
    *,
    start_round: int,
    end_round: int,
    window_size: int,
) -> tuple[EvaluationWindow, ...]:
    windows: list[EvaluationWindow] = []

    current = start_round
    index = 1

    while current <= end_round:
        window_end = min(
            current + window_size - 1,
            end_round,
        )

        windows.append(
            EvaluationWindow(
                name=f"window-{index:03d}",
                start_round=current,
                end_round=window_end,
            )
        )

        current = window_end + 1
        index += 1

    return tuple(windows)


def run(
    argv: Sequence[str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.start_round > args.end_round:
        parser.error(
            "start-round must be <= end-round"
        )

    if args.window_size <= 0:
        parser.error(
            "window-size must be greater than zero"
        )

    history = _load_history(
        args.history,
        parser,
    )

    windows = _build_windows(
        start_round=args.start_round,
        end_round=args.end_round,
        window_size=args.window_size,
    )

    config = ReplayConfig(
        start_round=args.start_round,
        end_round=args.end_round,
        seed_base=args.seed_base,
        temperature=args.temperature,
        candidate_count=args.candidate_count,
        top_k=args.top_k,
        practical_k=args.practical_k,
        long_gap_window=args.long_gap_window,
        confidence=args.confidence,
        mode=args.mode,
    )

    service = (
        HistoricalModelEvaluationOrchestrationService()
    )

    result = service.run(
        history=history,
        model_names=tuple(args.models),
        windows=windows,
        replay_output_root=args.replay_output,
        report_output_root=args.report_output,
        base_config=config,
    )

    payload = {
        "status": "PASS",
        "ranking_champion": (
            result.matrix.ranking.champion
        ),
        "selected_model": (
            result.champion.selection.selected_model
        ),
        "promoted": (
            result.champion.selection.promotion.promoted
        ),
        "artifact_path": str(
            result.artifact_path
        ),
    }

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
