"""Execution framework for historical replay validation."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from lrp.contracts import ContractError

from .historical_replay_models import (
    ReplayConfig,
    ReplayRoundResult,
    ReplaySummary,
    summarize_replay,
    validate_round_coverage,
)


RoundExecutor = Callable[
    [int, int, object, object | None],
    tuple[ReplayRoundResult, object | None],
]


@dataclass(frozen=True, slots=True)
class HistoricalReplayResult:
    """Completed replay output and artifact paths."""

    config: ReplayConfig
    rounds: tuple[ReplayRoundResult, ...]
    summary: ReplaySummary
    final_state: object | None
    rounds_path: Path
    summary_path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.as_dict(),
            "summary": self.summary.as_dict(),
            "rounds_path": str(self.rounds_path),
            "summary_path": str(self.summary_path),
        }


class HistoricalReplayRunner:
    """Run deterministic rounds and persist validation metrics."""

    def __init__(
        self,
        *,
        executor: RoundExecutor,
    ) -> None:
        if not callable(executor):
            raise TypeError(
                "executor must be callable"
            )

        self._executor = executor

    @property
    def executor(self) -> RoundExecutor:
        return self._executor

    def run(
        self,
        *,
        config: ReplayConfig,
        draw_by_round: Mapping[int, object],
        output_root: str | Path,
        initial_state: object | None = None,
        overwrite: bool = False,
    ) -> HistoricalReplayResult:
        if not isinstance(config, ReplayConfig):
            raise TypeError(
                "config must be a ReplayConfig"
            )

        if not isinstance(draw_by_round, Mapping):
            raise TypeError(
                "draw_by_round must be a mapping"
            )

        if not isinstance(overwrite, bool):
            raise TypeError(
                "overwrite must be a boolean"
            )

        validate_round_coverage(
            config=config,
            draw_by_round=draw_by_round,
        )

        root = Path(output_root)
        rounds_path = root / "replay_rounds.jsonl"
        summary_path = root / "replay_summary.json"

        self._prepare_output(
            root=root,
            rounds_path=rounds_path,
            summary_path=summary_path,
            overwrite=overwrite,
        )

        state = initial_state
        rows: list[ReplayRoundResult] = []

        started = perf_counter()

        try:
            with rounds_path.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as stream:
                for round_no in config.rounds:
                    seed = config.seed_for_round(
                        round_no
                    )

                    row, state = self._executor(
                        round_no,
                        seed,
                        draw_by_round[round_no],
                        state,
                    )

                    self._validate_executor_result(
                        expected_round=round_no,
                        expected_seed=seed,
                        row=row,
                    )

                    rows.append(row)

                    stream.write(
                        json.dumps(
                            row.as_dict(),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                    stream.write("\n")
                    stream.flush()

        except Exception:
            self._write_failure(
                root=root,
                config=config,
                completed=tuple(rows),
            )
            raise

        summary = summarize_replay(
            tuple(rows)
        )

        elapsed = perf_counter() - started

        payload = {
            "status": "PASS",
            "config": config.as_dict(),
            "summary": summary.as_dict(),
            "measured_runner_seconds": elapsed,
            "rounds_path": str(rounds_path),
        }

        summary_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        failure_path = root / "replay_failure.json"
        if failure_path.exists():
            failure_path.unlink()

        return HistoricalReplayResult(
            config=config,
            rounds=tuple(rows),
            summary=summary,
            final_state=state,
            rounds_path=rounds_path,
            summary_path=summary_path,
        )

    @staticmethod
    def _prepare_output(
        *,
        root: Path,
        rounds_path: Path,
        summary_path: Path,
        overwrite: bool,
    ) -> None:
        existing = tuple(
            path
            for path in (
                rounds_path,
                summary_path,
                root / "replay_failure.json",
            )
            if path.exists()
        )

        if existing and not overwrite:
            raise FileExistsError(
                "replay output already exists: "
                + ", ".join(
                    str(path)
                    for path in existing
                )
            )

        root.mkdir(
            parents=True,
            exist_ok=True,
        )

        if overwrite:
            for path in existing:
                path.unlink()

    @staticmethod
    def _validate_executor_result(
        *,
        expected_round: int,
        expected_seed: int,
        row: ReplayRoundResult,
    ) -> None:
        if not isinstance(
            row,
            ReplayRoundResult,
        ):
            raise TypeError(
                "executor must return "
                "ReplayRoundResult"
            )

        if row.round_no != expected_round:
            raise ContractError(
                "executor returned unexpected round"
            )

        if row.seed != expected_seed:
            raise ContractError(
                "executor returned unexpected seed"
            )

    @staticmethod
    def _write_failure(
        *,
        root: Path,
        config: ReplayConfig,
        completed: tuple[ReplayRoundResult, ...],
    ) -> None:
        payload = {
            "status": "ERROR",
            "config": config.as_dict(),
            "completed_rounds": [
                row.round_no
                for row in completed
            ],
            "completed_count": len(completed),
        }

        failure_path = (
            root / "replay_failure.json"
        )

        failure_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
