from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProductionLifecycleRequest:
    """Inputs required by a production lifecycle run."""

    history_path: Path
    evaluation_output_root: Path
    production_registry_root: Path
    production_snapshot_root: Path
    prediction_output_root: Path
    round_no: int
    seed: int
    temperature: float
    candidate_count: int
    top_k: int
    practical_k: int
    mode: str
    evaluation_start_round: int | None = None
    evaluation_end_round: int | None = None
    long_gap_window: int = 5


@dataclass(frozen=True)
class ProductionLifecycleStageResult:
    """Result of one ordered lifecycle stage."""

    name: str
    status: str
    detail: dict[str, Any]


@dataclass(frozen=True)
class ProductionLifecycleResult:
    """Aggregate lifecycle result."""

    status: str
    stages: tuple[
        ProductionLifecycleStageResult,
        ...,
    ]


class ProductionLifecycleService:
    """Application boundary for production orchestration."""

    def __init__(
        self,
        *,
        model_evaluation=None,
        publication=None,
        audit=None,
        prediction=None,
    ) -> None:
        self._model_evaluation = (
            model_evaluation
        )
        self._publication = publication
        self._audit = audit
        self._prediction = prediction

    def run(
        self,
        request: ProductionLifecycleRequest,
    ) -> ProductionLifecycleResult:
        stages = []

        runners = (
            self._model_evaluation,
            self._publication,
            self._audit,
            self._prediction,
        )

        if any(
            runner is None
            for runner in runners
        ):
            raise NotImplementedError(
                "production lifecycle orchestration "
                "dependencies are not configured"
            )

        overall_status = "PASS"

        for runner in runners:
            stage = runner(request)
            stages.append(stage)

            if stage.status == "ERROR":
                overall_status = "ERROR"
                break

            if stage.status == "WARN":
                overall_status = "WARN"

        return ProductionLifecycleResult(
            status=overall_status,
            stages=tuple(stages),
        )
