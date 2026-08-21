from __future__ import annotations

import inspect
from dataclasses import is_dataclass

from lrp.evaluation.topk_live_evaluation_orchestrator import (
    TopKLiveEvaluationOrchestrator,
    TopKLiveEvaluationRequest,
    TopKLiveEvaluationResult,
)


def _parameter_names(
    value: object,
) -> tuple[str, ...]:
    return tuple(
        inspect.signature(
            value
        ).parameters
    )


def test_request_public_signature_is_exact() -> None:
    assert _parameter_names(
        TopKLiveEvaluationRequest
    ) == (
        "window",
        "candidate_prediction_result",
        "candidate_history_rows",
        "candidate_model_name",
        "baseline_prediction_result",
        "baseline_history_rows",
        "baseline_model_name",
        "actual_draws",
        "candidate_regime_id",
        "candidate_strategy_name",
        "baseline_regime_id",
        "baseline_strategy_name",
    )


def test_request_is_frozen_dataclass() -> None:
    assert is_dataclass(
        TopKLiveEvaluationRequest
    )

    assert (
        TopKLiveEvaluationRequest.__dataclass_params__.frozen
        is True
    )


def test_result_public_signature_is_exact() -> None:
    assert _parameter_names(
        TopKLiveEvaluationResult
    ) == (
        "evaluation",
        "candidate_binding",
        "baseline_binding",
        "candidate_replay_prediction",
        "baseline_replay_prediction",
    )


def test_result_is_frozen_dataclass() -> None:
    assert is_dataclass(
        TopKLiveEvaluationResult
    )

    assert (
        TopKLiveEvaluationResult.__dataclass_params__.frozen
        is True
    )


def test_orchestrator_constructor_is_parameterless() -> None:
    assert _parameter_names(
        TopKLiveEvaluationOrchestrator
    ) == ()


def test_orchestrator_evaluate_signature_is_exact() -> None:
    signature = inspect.signature(
        TopKLiveEvaluationOrchestrator.evaluate
    )

    assert tuple(
        signature.parameters
    ) == (
        "self",
        "request",
    )

    assert (
        signature.parameters["request"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )