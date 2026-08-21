from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lrp.contracts.exceptions import ContractError
from lrp.io.draws import HistoryRow
from lrp.evaluation.topk_live_prediction_binding import (
    TopKLivePredictionBinder,
    TopKLivePredictionBindingRequest,
)
from lrp.evaluation.topk_prediction_source_adapter import (
    TopKPredictionSourceAdapter,
)
from lrp.evaluation.topk_replay_evaluation import (
    TopKReplayEvaluationRequest,
    TopKReplayEvaluationResult,
    TopKReplayEvaluationService,
)
from lrp.pipelines.models import PredictionResult


def _require_non_empty_string(
    value: Any,
    *,
    field_name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ContractError(
            f"{field_name} must be a non-empty string"
        )
    return value


def _require_prediction_result(
    value: Any,
    *,
    field_name: str,
) -> PredictionResult:
    if not isinstance(value, PredictionResult):
        raise ContractError(
            f"{field_name} must be PredictionResult"
        )
    return value


def _require_history_rows(
    value: Any,
    *,
    field_name: str,
) -> tuple[HistoryRow, ...]:
    if not isinstance(value, tuple):
        raise ContractError(
            f"{field_name} must be a tuple"
        )

    for row in value:
        if not isinstance(row, HistoryRow):
            raise ContractError(
                f"{field_name} must contain HistoryRow"
            )

    return value


def _require_actual_draws(
    value: Any,
) -> tuple[Any, ...]:
    if not isinstance(value, tuple):
        raise ContractError(
            "actual_draws must be a tuple"
        )

    for draw in value:
        if not hasattr(draw, "round_no"):
            raise ContractError(
                "actual draw must expose round_no"
            )
        if not hasattr(draw, "numbers"):
            raise ContractError(
                "actual draw must expose numbers"
            )

    return value


def _prediction_round(
    prediction_result: PredictionResult,
) -> int:
    generation = prediction_result.generation

    request = getattr(
        generation,
        "request",
        None,
    )

    round_no = getattr(
        request,
        "round_no",
        None,
    )

    if (
        isinstance(round_no, bool)
        or not isinstance(round_no, int)
    ):
        raise ContractError(
            "prediction round_no must be an integer"
        )

    return round_no


def _validate_history_before_round(
    history_rows: tuple[HistoryRow, ...],
    *,
    prediction_round: int,
    field_name: str,
) -> None:
    for row in history_rows:
        if row.round_no >= prediction_round:
            raise ContractError(
                f"{field_name} must contain only prior rounds"
            )


@dataclass(frozen=True)
class TopKLiveEvaluationRequest:
    window: Any
    candidate_prediction_result: PredictionResult
    candidate_history_rows: tuple[HistoryRow, ...]
    candidate_model_name: str
    baseline_prediction_result: PredictionResult
    baseline_history_rows: tuple[HistoryRow, ...]
    baseline_model_name: str
    actual_draws: tuple[Any, ...]
    candidate_regime_id: str | None = None
    candidate_strategy_name: str | None = None
    baseline_regime_id: str | None = None
    baseline_strategy_name: str | None = None


@dataclass(frozen=True)
class TopKLiveEvaluationResult:
    evaluation: TopKReplayEvaluationResult
    candidate_binding: Any
    baseline_binding: Any
    candidate_replay_prediction: Any
    baseline_replay_prediction: Any


class TopKLiveEvaluationOrchestrator:
    def evaluate(
        self,
        *,
        request: TopKLiveEvaluationRequest,
    ) -> TopKLiveEvaluationResult:
        if not isinstance(
            request,
            TopKLiveEvaluationRequest,
        ):
            raise ContractError(
                "request must be TopKLiveEvaluationRequest"
            )

        candidate_prediction_result = (
            _require_prediction_result(
                request.candidate_prediction_result,
                field_name="candidate_prediction_result",
            )
        )
        baseline_prediction_result = (
            _require_prediction_result(
                request.baseline_prediction_result,
                field_name="baseline_prediction_result",
            )
        )

        candidate_history_rows = _require_history_rows(
            request.candidate_history_rows,
            field_name="candidate_history_rows",
        )
        baseline_history_rows = _require_history_rows(
            request.baseline_history_rows,
            field_name="baseline_history_rows",
        )

        actual_draws = _require_actual_draws(
            request.actual_draws
        )

        candidate_model_name = _require_non_empty_string(
            request.candidate_model_name,
            field_name="candidate_model_name",
        )
        baseline_model_name = _require_non_empty_string(
            request.baseline_model_name,
            field_name="baseline_model_name",
        )

        for field_name, value in (
            (
                "candidate_regime_id",
                request.candidate_regime_id,
            ),
            (
                "baseline_regime_id",
                request.baseline_regime_id,
            ),
            (
                "candidate_strategy_name",
                request.candidate_strategy_name,
            ),
            (
                "baseline_strategy_name",
                request.baseline_strategy_name,
            ),
        ):
            if value is not None:
                _require_non_empty_string(
                    value,
                    field_name=field_name,
                )

        candidate_round = _prediction_round(
            candidate_prediction_result
        )
        baseline_round = _prediction_round(
            baseline_prediction_result
        )

        if candidate_round != baseline_round:
            raise ContractError(
                "candidate and baseline prediction rounds must match"
            )

        _validate_history_before_round(
            candidate_history_rows,
            prediction_round=candidate_round,
            field_name="candidate_history_rows",
        )
        _validate_history_before_round(
            baseline_history_rows,
            prediction_round=baseline_round,
            field_name="baseline_history_rows",
        )

        binder = TopKLivePredictionBinder()

        candidate_binding = binder.bind(
            request=TopKLivePredictionBindingRequest(
                prediction_result=candidate_prediction_result,
                history_rows=candidate_history_rows,
                model_name=candidate_model_name,
                regime_id=request.candidate_regime_id,
                strategy_name=request.candidate_strategy_name,
            )
        )

        baseline_binding = binder.bind(
            request=TopKLivePredictionBindingRequest(
                prediction_result=baseline_prediction_result,
                history_rows=baseline_history_rows,
                model_name=baseline_model_name,
                regime_id=request.baseline_regime_id,
                strategy_name=request.baseline_strategy_name,
            )
        )

        source_adapter = TopKPredictionSourceAdapter()

        candidate_replay_prediction = source_adapter.adapt(
            source=candidate_binding.source,
        )
        baseline_replay_prediction = source_adapter.adapt(
            source=baseline_binding.source,
        )

        replay_result = TopKReplayEvaluationService().evaluate(
            request=TopKReplayEvaluationRequest(
                window=request.window,
                candidate_predictions=(
                    candidate_replay_prediction,
                ),
                baseline_predictions=(
                    baseline_replay_prediction,
                ),
                actual_draws=actual_draws,
            )
        )

        return TopKLiveEvaluationResult(
            evaluation=replay_result,
            candidate_binding=candidate_binding,
            baseline_binding=baseline_binding,
            candidate_replay_prediction=candidate_replay_prediction,
            baseline_replay_prediction=baseline_replay_prediction,
        )
