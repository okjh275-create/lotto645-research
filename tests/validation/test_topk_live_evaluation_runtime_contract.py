from __future__ import annotations

import importlib
import inspect
from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from lrp.evaluation import EvaluationWindow
from lrp.contracts.exceptions import ContractError
from lrp.io.draws import HistoryRow
from lrp.pipelines.models import (
    PredictionGenerationResult,
    PredictionRequest,
    PredictionResult,
)


PRODUCT_MODULE = (
    "lrp.evaluation."
    "topk_live_evaluation_runtime"
)


def _product():
    return importlib.import_module(
        PRODUCT_MODULE
    )


class _SelectedItem:
    def __init__(
        self,
        numbers: tuple[int, ...],
    ) -> None:
        self.numbers = numbers


class _Diversity:
    def __init__(
        self,
        selected: tuple[_SelectedItem, ...],
    ) -> None:
        self.selected = selected


def _prediction_result(
    *,
    round_no: int = 1233,
    top_k: int = 10,
    generated_at_kst: datetime | None = None,
) -> PredictionResult:

    request = PredictionRequest(
        round_no=round_no,
        seed=20260821,
        top_k=top_k,
        practical_k=1,
        long_gap_numbers=frozenset({45}),
    )

    generation = PredictionGenerationResult(
        request=request,
        windows=(10, 20, 50),
        probabilities={},
        statistics_contract=object(),
        number_signals={},
        candidates=(),
        statistics_version="stats-v1",
        candidate_version="candidate-v1",
    )

    return PredictionResult(
        generation=generation,
        scored_candidates=(),
        ranking=object(),
        diversity=_Diversity(
            (
                _SelectedItem(
                    (1, 7, 13, 24, 32, 41),
                ),
                _SelectedItem(
                    (3, 9, 18, 27, 35, 44),
                ),
                _SelectedItem(
                    (2, 10, 16, 23, 31, 45),
                ),
                _SelectedItem(
                    (4, 11, 19, 26, 33, 42),
                ),
                _SelectedItem(
                    (5, 12, 17, 28, 36, 43),
                ),
                _SelectedItem(
                    (6, 14, 20, 29, 34, 40),
                ),
                _SelectedItem(
                    (8, 15, 21, 25, 37, 44),
                ),
                _SelectedItem(
                    (1, 16, 22, 30, 38, 45),
                ),
                _SelectedItem(
                    (7, 17, 23, 31, 39, 42),
                ),
                _SelectedItem(
                    (9, 18, 24, 32, 40, 43),
                ),
            )
        ),
        practical=object(),
        generated_at_kst=(
            generated_at_kst
            if generated_at_kst is not None
            else datetime.fromisoformat(
                "2026-08-21T17:00:00+09:00"
            )
        ),
    )


def _history_rows() -> tuple[HistoryRow, ...]:
    return (
        HistoryRow(
            round_no=1230,
            numbers=(1, 2, 3, 4, 5, 6),
            bonus=7,
        ),
        HistoryRow(
            round_no=1231,
            numbers=(8, 9, 10, 11, 12, 13),
            bonus=14,
        ),
        HistoryRow(
            round_no=1232,
            numbers=(15, 16, 17, 18, 19, 20),
            bonus=21,
        ),
    )


def _actual_draws() -> tuple[object, ...]:
    return (
        SimpleNamespace(
            round_no=1233,
            numbers=(2, 8, 16, 23, 31, 44),
        ),
    )


def _request(
    *,
    candidate_prediction_result: PredictionResult | None = None,
    baseline_prediction_result: PredictionResult | None = None,
    candidate_model_name: str = "candidate-v1",
    baseline_model_name: str = "baseline-v1",
    candidate_source_artifact_sha256: str = "a" * 64,
    baseline_source_artifact_sha256: str = "b" * 64,
):
    product = _product()

    return product.TopKLiveEvaluationRuntimeRequest(
        window=EvaluationWindow(
            name="ag20-live-runtime",
            start_round=1233,
            end_round=1233,
        ),
        candidate_prediction_result=(
            candidate_prediction_result
            if candidate_prediction_result is not None
            else _prediction_result()
        ),
        candidate_history_rows=_history_rows(),
        candidate_model_name=candidate_model_name,
        candidate_source_artifact_sha256=(
            candidate_source_artifact_sha256
        ),
        baseline_prediction_result=(
            baseline_prediction_result
            if baseline_prediction_result is not None
            else _prediction_result()
        ),
        baseline_history_rows=_history_rows(),
        baseline_model_name=baseline_model_name,
        baseline_source_artifact_sha256=(
            baseline_source_artifact_sha256
        ),
        actual_draws=_actual_draws(),
        candidate_regime_id="regime-a",
        candidate_strategy_name="strategy-a",
        baseline_regime_id="regime-b",
        baseline_strategy_name="strategy-b",
    )


def _parameter_names(value: object) -> tuple[str, ...]:
    return tuple(
        inspect.signature(value)
        .parameters
        .keys()
    )


def test_runtime_request_public_signature_is_exact() -> None:
    product = _product()

    assert _parameter_names(
        product.TopKLiveEvaluationRuntimeRequest
    ) == (
        "window",
        "candidate_prediction_result",
        "candidate_history_rows",
        "candidate_model_name",
        "candidate_source_artifact_sha256",
        "baseline_prediction_result",
        "baseline_history_rows",
        "baseline_model_name",
        "baseline_source_artifact_sha256",
        "actual_draws",
        "candidate_regime_id",
        "candidate_strategy_name",
        "baseline_regime_id",
        "baseline_strategy_name",
    )


def test_runtime_result_public_signature_is_exact() -> None:
    product = _product()

    assert _parameter_names(
        product.TopKLiveEvaluationRuntimeResult
    ) == (
        "evaluation",
        "source_pair",
    )


def test_runtime_service_is_parameterless() -> None:
    product = _product()

    assert _parameter_names(
        product.TopKLiveEvaluationRuntimeService
    ) == ()


def test_runtime_execute_signature_is_exact() -> None:
    product = _product()

    signature = inspect.signature(
        product.TopKLiveEvaluationRuntimeService.execute
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


def test_runtime_request_is_immutable() -> None:
    request = _request()

    with pytest.raises(FrozenInstanceError):
        request.candidate_model_name = "changed"


def test_runtime_composes_dual_source_evaluation() -> None:
    product = _product()

    result = (
        product.TopKLiveEvaluationRuntimeService()
        .execute(
            request=_request()
        )
    )

    assert result.evaluation is not None
    assert result.source_pair is not None


def test_runtime_source_pair_preserves_identity() -> None:
    product = _product()

    result = (
        product.TopKLiveEvaluationRuntimeService()
        .execute(
            request=_request()
        )
    )

    assert (
        result.source_pair.candidate.model_name
        == "candidate-v1"
    )

    assert (
        result.source_pair.baseline.model_name
        == "baseline-v1"
    )


def test_runtime_source_pair_preserves_sha256() -> None:
    product = _product()

    result = (
        product.TopKLiveEvaluationRuntimeService()
        .execute(
            request=_request()
        )
    )

    assert (
        result.source_pair
        .candidate
        .source_artifact_sha256
        == "a" * 64
    )

    assert (
        result.source_pair
        .baseline
        .source_artifact_sha256
        == "b" * 64
    )


def test_runtime_preserves_optional_provenance() -> None:
    product = _product()

    result = (
        product.TopKLiveEvaluationRuntimeService()
        .execute(
            request=_request()
        )
    )

    assert (
        result.source_pair.candidate.regime_id
        == "regime-a"
    )

    assert (
        result.source_pair.candidate.strategy_name
        == "strategy-a"
    )

    assert (
        result.source_pair.baseline.regime_id
        == "regime-b"
    )

    assert (
        result.source_pair.baseline.strategy_name
        == "strategy-b"
    )


def test_runtime_repeated_execute_is_semantically_stable() -> None:
    product = _product()

    request = _request()
    service = (
        product.TopKLiveEvaluationRuntimeService()
    )

    first = service.execute(
        request=request
    )

    second = service.execute(
        request=request
    )

    assert first == second


def test_runtime_does_not_mutate_prediction_results() -> None:
    product = _product()

    candidate = _prediction_result()
    baseline = _prediction_result()

    candidate_generation = candidate.generation
    candidate_diversity = candidate.diversity

    baseline_generation = baseline.generation
    baseline_diversity = baseline.diversity

    product.TopKLiveEvaluationRuntimeService().execute(
        request=_request(
            candidate_prediction_result=candidate,
            baseline_prediction_result=baseline,
        )
    )

    assert candidate.generation is candidate_generation
    assert candidate.diversity is candidate_diversity
    assert baseline.generation is baseline_generation
    assert baseline.diversity is baseline_diversity


def test_runtime_rejects_invalid_request_type() -> None:
    product = _product()

    with pytest.raises(ContractError):
        product.TopKLiveEvaluationRuntimeService().execute(
            request=object(),
        )


def test_runtime_rejects_candidate_round_mismatch() -> None:
    product = _product()

    with pytest.raises(ContractError):
        product.TopKLiveEvaluationRuntimeService().execute(
            request=_request(
                candidate_prediction_result=(
                    _prediction_result(
                        round_no=1234,
                    )
                )
            )
        )


def test_runtime_rejects_same_model_identity() -> None:
    product = _product()

    with pytest.raises(ContractError):
        product.TopKLiveEvaluationRuntimeService().execute(
            request=_request(
                candidate_model_name="same-model",
                baseline_model_name="same-model",
            )
        )


def test_runtime_rejects_invalid_candidate_sha256() -> None:
    with pytest.raises(ContractError):
        _request(
            candidate_source_artifact_sha256="abc"
        )


def test_runtime_rejects_invalid_baseline_sha256() -> None:
    with pytest.raises(ContractError):
        _request(
            baseline_source_artifact_sha256=(
                "ABC" * 21 + "A"
            )
        )


def test_runtime_product_has_no_filesystem_dependency() -> None:
    path = Path(
        "lrp/evaluation/"
        "topk_live_evaluation_runtime.py"
    )

    source = path.read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "open(",
        "Path(",
        "write_text(",
        "write_bytes(",
        "mkdir(",
        "write_operation_artifact",
        "write_prediction_artifacts",
        "sqlite3",
        "subprocess",
    )

    assert not any(
        token in source
        for token in forbidden
    )


def test_runtime_product_has_no_runtime_nondeterminism() -> None:
    path = Path(
        "lrp/evaluation/"
        "topk_live_evaluation_runtime.py"
    )

    source = path.read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "datetime.now",
        "datetime.utcnow",
        "random",
        "secrets",
        "uuid",
        "time.time",
    )

    assert not any(
        token in source
        for token in forbidden
    )


def test_runtime_does_not_import_round_completion() -> None:
    path = Path(
        "lrp/evaluation/"
        "topk_live_evaluation_runtime.py"
    )

    source = path.read_text(
        encoding="utf-8-sig"
    )

    assert "round_completion" not in source


def test_runtime_does_not_import_production_lifecycle() -> None:
    path = Path(
        "lrp/evaluation/"
        "topk_live_evaluation_runtime.py"
    )

    source = path.read_text(
        encoding="utf-8-sig"
    )

    assert "production_lifecycle" not in source