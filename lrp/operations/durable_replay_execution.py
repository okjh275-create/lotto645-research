from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lrp.contracts.exceptions import ContractError
from lrp.evaluation.contracts import EvaluationWindow
from lrp.evaluation.topk_replay_evaluation import (
    TopKReplayEvaluationResult,
)
from lrp.io.draws import (
    HistoryRow,
    history_until_round,
    load_history,
)
from lrp.operations.durable_replay_evaluation_orchestrator import (
    DurableReplayEvaluationOrchestrator,
    DurableReplayEvaluationSourceSpec,
)


@dataclass(frozen=True)
class DurableReplayExecutionSource:
    artifact_path: str | Path
    round_no: int
    model_name: str
    regime_id: str | None = None
    strategy_name: str | None = None


@dataclass(frozen=True)
class DurableReplayExecutionRequest:
    history_path: str | Path
    window_name: str
    start_round: int
    end_round: int
    candidate_sources: tuple[
        DurableReplayExecutionSource,
        ...,
    ]
    baseline_sources: tuple[
        DurableReplayExecutionSource,
        ...,
    ]


class DurableReplayExecutionService:
    def execute(
        self,
        *,
        request: DurableReplayExecutionRequest,
    ) -> TopKReplayEvaluationResult:
        if not isinstance(
            request,
            DurableReplayExecutionRequest,
        ):
            raise ContractError(
                "request must be "
                "DurableReplayExecutionRequest"
            )

        history_rows = load_history(
            request.history_path
        )

        window = EvaluationWindow(
            name=request.window_name,
            start_round=request.start_round,
            end_round=request.end_round,
        )

        candidate_specs = tuple(
            self._source_spec(
                source=source,
                history_rows=history_rows,
                label="candidate",
            )
            for source in request.candidate_sources
        )

        baseline_specs = tuple(
            self._source_spec(
                source=source,
                history_rows=history_rows,
                label="baseline",
            )
            for source in request.baseline_sources
        )

        actual_draws = tuple(
            row
            for row in history_rows
            if (
                request.start_round
                <= row.round_no
                <= request.end_round
            )
        )

        return (
            DurableReplayEvaluationOrchestrator()
            .evaluate(
                window=window,
                candidate_sources=candidate_specs,
                baseline_sources=baseline_specs,
                actual_draws=actual_draws,
            )
        )

    @staticmethod
    def _source_spec(
        *,
        source: DurableReplayExecutionSource,
        history_rows: tuple[HistoryRow, ...],
        label: str,
    ) -> DurableReplayEvaluationSourceSpec:
        if not isinstance(
            source,
            DurableReplayExecutionSource,
        ):
            raise ContractError(
                f"{label} source must be "
                "DurableReplayExecutionSource"
            )

        source_history = history_until_round(
            history_rows,
            target_round=source.round_no,
        )

        history_rounds = tuple(
            row.round_no
            for row in source_history
        )

        return DurableReplayEvaluationSourceSpec(
            artifact_path=source.artifact_path,
            history_rounds=history_rounds,
            model_name=source.model_name,
            regime_id=source.regime_id,
            strategy_name=source.strategy_name,
        )
