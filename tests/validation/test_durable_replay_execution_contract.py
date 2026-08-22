from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.evaluation.contracts import EvaluationWindow
from lrp.evaluation.topk_replay_evaluation import (
    TopKReplayEvaluationResult,
)
from lrp.io.draws import HistoryRow


_PRODUCT_MODULE = "lrp.operations.durable_replay_execution"


def _product():
    return importlib.import_module(
        _PRODUCT_MODULE
    )


def _source(
    *,
    artifact_path: str | Path = "candidate.json",
    round_no: int = 1233,
    model_name: str = "candidate-model",
    regime_id: str | None = None,
    strategy_name: str | None = None,
):
    product = _product()

    return product.DurableReplayExecutionSource(
        artifact_path=artifact_path,
        round_no=round_no,
        model_name=model_name,
        regime_id=regime_id,
        strategy_name=strategy_name,
    )


def _request(
    *,
    history_path: str | Path = "history.json",
    window_name: str = "window-001",
    start_round: int = 1233,
    end_round: int = 1234,
    candidate_sources=None,
    baseline_sources=None,
):
    product = _product()

    if candidate_sources is None:
        candidate_sources = (
            _source(
                artifact_path="candidate.json",
                round_no=1233,
                model_name="candidate-model",
            ),
        )

    if baseline_sources is None:
        baseline_sources = (
            _source(
                artifact_path="baseline.json",
                round_no=1233,
                model_name="baseline-model",
            ),
        )

    return product.DurableReplayExecutionRequest(
        history_path=history_path,
        window_name=window_name,
        start_round=start_round,
        end_round=end_round,
        candidate_sources=candidate_sources,
        baseline_sources=baseline_sources,
    )


def _history() -> tuple[HistoryRow, ...]:
    return (
        HistoryRow(
            round_no=1230,
            numbers=(1, 2, 3, 4, 5, 6),
        ),
        HistoryRow(
            round_no=1231,
            numbers=(7, 8, 9, 10, 11, 12),
        ),
        HistoryRow(
            round_no=1232,
            numbers=(13, 14, 15, 16, 17, 18),
        ),
        HistoryRow(
            round_no=1233,
            numbers=(19, 20, 21, 22, 23, 24),
        ),
        HistoryRow(
            round_no=1234,
            numbers=(25, 26, 27, 28, 29, 30),
        ),
    )


_FAKE_EVALUATION = "evaluation-A"


def _fake_result() -> TopKReplayEvaluationResult:
    return SimpleNamespace(
        evaluation=_FAKE_EVALUATION,
        candidate_model_name="candidate-model",
        baseline_model_name="baseline-model",
        round_count=2,
    )


def test_execution_source_is_frozen() -> None:
    product = _product()
    cls = product.DurableReplayExecutionSource

    assert is_dataclass(cls)
    assert cls.__dataclass_params__.frozen is True


def test_execution_source_fields_are_exact() -> None:
    product = _product()
    cls = product.DurableReplayExecutionSource

    assert tuple(
        field.name
        for field in fields(cls)
    ) == (
        "artifact_path",
        "round_no",
        "model_name",
        "regime_id",
        "strategy_name",
    )


def test_execution_request_is_frozen() -> None:
    product = _product()
    cls = product.DurableReplayExecutionRequest

    assert is_dataclass(cls)
    assert cls.__dataclass_params__.frozen is True


def test_execution_request_fields_are_exact() -> None:
    product = _product()
    cls = product.DurableReplayExecutionRequest

    assert tuple(
        field.name
        for field in fields(cls)
    ) == (
        "history_path",
        "window_name",
        "start_round",
        "end_round",
        "candidate_sources",
        "baseline_sources",
    )


def test_service_is_parameterless() -> None:
    product = _product()

    assert str(
        inspect.signature(
            product.DurableReplayExecutionService
        )
    ) == "()"


def test_execute_public_signature_is_exact() -> None:
    product = _product()

    assert str(
        inspect.signature(
            product.DurableReplayExecutionService.execute
        )
    ) == (
        "(self, *, request: "
        "'DurableReplayExecutionRequest') "
        "-> 'TopKReplayEvaluationResult'"
    )


def test_execute_returns_topk_replay_evaluation_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()
    expected = _fake_result()

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: _history(),
    )

    monkeypatch.setattr(
        product.DurableReplayEvaluationOrchestrator,
        "evaluate",
        lambda self, **kwargs: expected,
    )

    result = (
        product.DurableReplayExecutionService()
        .execute(
            request=_request()
        )
    )

    assert result is expected


def test_execute_loads_history_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()
    calls: list[Any] = []

    def fake_load_history(path):
        calls.append(path)
        return _history()

    monkeypatch.setattr(
        product,
        "load_history",
        fake_load_history,
    )

    monkeypatch.setattr(
        product.DurableReplayEvaluationOrchestrator,
        "evaluate",
        lambda self, **kwargs: _fake_result(),
    )

    request = _request(
        history_path="history.json"
    )

    product.DurableReplayExecutionService().execute(
        request=request
    )

    assert calls == ["history.json"]


def test_execute_constructs_exact_evaluation_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()
    captured = {}

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: _history(),
    )

    def fake_evaluate(self, **kwargs):
        captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(
        product.DurableReplayEvaluationOrchestrator,
        "evaluate",
        fake_evaluate,
    )

    product.DurableReplayExecutionService().execute(
        request=_request(
            window_name="window-A",
            start_round=1233,
            end_round=1234,
        )
    )

    assert captured["window"] == EvaluationWindow(
        name="window-A",
        start_round=1233,
        end_round=1234,
    )


def test_execute_projects_candidate_history_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()
    captured = {}

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: _history(),
    )

    def fake_evaluate(self, **kwargs):
        captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(
        product.DurableReplayEvaluationOrchestrator,
        "evaluate",
        fake_evaluate,
    )

    product.DurableReplayExecutionService().execute(
        request=_request(
            candidate_sources=(
                _source(
                    artifact_path="candidate.json",
                    round_no=1233,
                    model_name="candidate-model",
                ),
            )
        )
    )

    spec = captured["candidate_sources"][0]

    assert spec.history_rounds == (
        1230,
        1231,
        1232,
    )


def test_execute_projects_baseline_history_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()
    captured = {}

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: _history(),
    )

    def fake_evaluate(self, **kwargs):
        captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(
        product.DurableReplayEvaluationOrchestrator,
        "evaluate",
        fake_evaluate,
    )

    product.DurableReplayExecutionService().execute(
        request=_request(
            baseline_sources=(
                _source(
                    artifact_path="baseline.json",
                    round_no=1234,
                    model_name="baseline-model",
                ),
            )
        )
    )

    spec = captured["baseline_sources"][0]

    assert spec.history_rounds == (
        1230,
        1231,
        1232,
        1233,
    )


def test_execute_preserves_candidate_source_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()
    captured = {}

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: _history(),
    )

    def fake_evaluate(self, **kwargs):
        captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(
        product.DurableReplayEvaluationOrchestrator,
        "evaluate",
        fake_evaluate,
    )

    request = _request(
        candidate_sources=(
            _source(
                artifact_path="c2.json",
                round_no=1234,
                model_name="c2",
            ),
            _source(
                artifact_path="c1.json",
                round_no=1233,
                model_name="c1",
            ),
        )
    )

    product.DurableReplayExecutionService().execute(
        request=request
    )

    assert tuple(
        spec.model_name
        for spec in captured["candidate_sources"]
    ) == (
        "c2",
        "c1",
    )


def test_execute_preserves_baseline_source_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()
    captured = {}

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: _history(),
    )

    def fake_evaluate(self, **kwargs):
        captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(
        product.DurableReplayEvaluationOrchestrator,
        "evaluate",
        fake_evaluate,
    )

    request = _request(
        baseline_sources=(
            _source(
                artifact_path="b2.json",
                round_no=1234,
                model_name="b2",
            ),
            _source(
                artifact_path="b1.json",
                round_no=1233,
                model_name="b1",
            ),
        )
    )

    product.DurableReplayExecutionService().execute(
        request=request
    )

    assert tuple(
        spec.model_name
        for spec in captured["baseline_sources"]
    ) == (
        "b2",
        "b1",
    )


def test_execute_preserves_candidate_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()
    captured = {}

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: _history(),
    )

    def fake_evaluate(self, **kwargs):
        captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(
        product.DurableReplayEvaluationOrchestrator,
        "evaluate",
        fake_evaluate,
    )

    source = _source(
        artifact_path="candidate.json",
        round_no=1233,
        model_name="candidate",
        regime_id="regime-A",
        strategy_name="strategy-A",
    )

    product.DurableReplayExecutionService().execute(
        request=_request(
            candidate_sources=(source,)
        )
    )

    spec = captured["candidate_sources"][0]

    assert spec.artifact_path == source.artifact_path
    assert spec.model_name == source.model_name
    assert spec.regime_id == source.regime_id
    assert spec.strategy_name == source.strategy_name


def test_execute_preserves_baseline_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()
    captured = {}

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: _history(),
    )

    def fake_evaluate(self, **kwargs):
        captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(
        product.DurableReplayEvaluationOrchestrator,
        "evaluate",
        fake_evaluate,
    )

    source = _source(
        artifact_path="baseline.json",
        round_no=1233,
        model_name="baseline",
        regime_id="regime-B",
        strategy_name="strategy-B",
    )

    product.DurableReplayExecutionService().execute(
        request=_request(
            baseline_sources=(source,)
        )
    )

    spec = captured["baseline_sources"][0]

    assert spec.artifact_path == source.artifact_path
    assert spec.model_name == source.model_name
    assert spec.regime_id == source.regime_id
    assert spec.strategy_name == source.strategy_name


def test_execute_projects_actual_draws_from_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()
    captured = {}

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: _history(),
    )

    def fake_evaluate(self, **kwargs):
        captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(
        product.DurableReplayEvaluationOrchestrator,
        "evaluate",
        fake_evaluate,
    )

    product.DurableReplayExecutionService().execute(
        request=_request(
            start_round=1233,
            end_round=1234,
        )
    )

    assert tuple(
        row.round_no
        for row in captured["actual_draws"]
    ) == (
        1233,
        1234,
    )


def test_execute_invokes_ak_orchestrator_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()
    calls = []

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: _history(),
    )

    def fake_evaluate(self, **kwargs):
        calls.append(kwargs)
        return _fake_result()

    monkeypatch.setattr(
        product.DurableReplayEvaluationOrchestrator,
        "evaluate",
        fake_evaluate,
    )

    product.DurableReplayExecutionService().execute(
        request=_request()
    )

    assert len(calls) == 1


def test_execute_delegates_history_file_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()
    failure = FileNotFoundError("history missing")

    def fake_load_history(path):
        raise failure

    monkeypatch.setattr(
        product,
        "load_history",
        fake_load_history,
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        product.DurableReplayExecutionService().execute(
            request=_request()
        )

    assert exc_info.value is failure


def test_execute_delegates_window_contract_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: _history(),
    )

    with pytest.raises(ContractError):
        product.DurableReplayExecutionService().execute(
            request=_request(
                window_name="",
            )
        )


def test_execute_delegates_history_projection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()
    failure = ContractError(
        "history projection failure"
    )

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: _history(),
    )

    monkeypatch.setattr(
        product,
        "history_until_round",
        lambda rows, *, target_round: (_ for _ in ()).throw(
            failure
        ),
    )

    with pytest.raises(ContractError) as exc_info:
        product.DurableReplayExecutionService().execute(
            request=_request()
        )

    assert exc_info.value is failure


def test_execute_delegates_ak_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()
    failure = ContractError("AK failure")

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: _history(),
    )

    def fake_evaluate(self, **kwargs):
        raise failure

    monkeypatch.setattr(
        product.DurableReplayEvaluationOrchestrator,
        "evaluate",
        fake_evaluate,
    )

    with pytest.raises(ContractError) as exc_info:
        product.DurableReplayExecutionService().execute(
            request=_request()
        )

    assert exc_info.value is failure


def test_execute_rejects_invalid_request_type() -> None:
    product = _product()

    with pytest.raises(
        ContractError,
        match=(
            "request must be "
            "DurableReplayExecutionRequest"
        ),
    ):
        product.DurableReplayExecutionService().execute(
            request=object()
        )


@pytest.mark.parametrize(
    "value",
    (
        None,
        object(),
        {},
        [],
        (),
        "source",
        1,
        True,
    ),
)
def test_execute_rejects_invalid_candidate_source_item(
    monkeypatch: pytest.MonkeyPatch,
    value,
) -> None:
    product = _product()

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: _history(),
    )

    request = _request(
        candidate_sources=(value,)
    )

    with pytest.raises(
        ContractError,
        match=(
            "candidate source must be "
            "DurableReplayExecutionSource"
        ),
    ):
        product.DurableReplayExecutionService().execute(
            request=request
        )


@pytest.mark.parametrize(
    "value",
    (
        None,
        object(),
        {},
        [],
        (),
        "source",
        1,
        True,
    ),
)
def test_execute_rejects_invalid_baseline_source_item(
    monkeypatch: pytest.MonkeyPatch,
    value,
) -> None:
    product = _product()

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: _history(),
    )

    request = _request(
        baseline_sources=(value,)
    )

    with pytest.raises(
        ContractError,
        match=(
            "baseline source must be "
            "DurableReplayExecutionSource"
        ),
    ):
        product.DurableReplayExecutionService().execute(
            request=request
        )


def test_execute_does_not_mutate_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: _history(),
    )

    monkeypatch.setattr(
        product.DurableReplayEvaluationOrchestrator,
        "evaluate",
        lambda self, **kwargs: _fake_result(),
    )

    request = _request()
    before = repr(request)

    product.DurableReplayExecutionService().execute(
        request=request
    )

    assert repr(request) == before


def test_execute_is_semantically_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()
    calls = []

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: _history(),
    )

    def fake_evaluate(self, **kwargs):
        calls.append(kwargs)
        return _fake_result()

    monkeypatch.setattr(
        product.DurableReplayEvaluationOrchestrator,
        "evaluate",
        fake_evaluate,
    )

    service = product.DurableReplayExecutionService()
    request = _request()

    first = service.execute(
        request=request
    )
    second = service.execute(
        request=request
    )

    assert first == second
    assert calls[0] == calls[1]


def _product_source() -> str:
    module_path = (
        Path(__file__).parents[2]
        / "lrp"
        / "operations"
        / "durable_replay_execution.py"
    )

    return module_path.read_text(
        encoding="utf-8-sig"
    )


def test_product_has_no_direct_json_dependency() -> None:
    source = _product_source()

    assert "import json" not in source
    assert "json.loads" not in source
    assert "json.load" not in source


def test_product_has_no_direct_filesystem_read_dependency() -> None:
    source = _product_source()

    forbidden = (
        ".read_text(",
        ".read_bytes(",
        "open(",
    )

    assert not any(
        token in source
        for token in forbidden
    )


def test_product_has_no_filesystem_write_dependency() -> None:
    source = _product_source()

    forbidden = (
        ".write_text(",
        ".write_bytes(",
        ".mkdir(",
        "write_operation_artifact",
        "write_prediction_artifacts",
    )

    assert not any(
        token in source
        for token in forbidden
    )


def test_product_has_no_artifact_discovery_dependency() -> None:
    source = _product_source()

    forbidden = (
        ".glob(",
        ".rglob(",
        ".iterdir(",
        "operation_log",
        "manifest",
        "evaluation_source.json",
        "prediction-evaluation-sources",
    )

    assert not any(
        token in source
        for token in forbidden
    )


def test_product_has_no_predictionresult_dependency() -> None:
    source = _product_source()

    assert "PredictionResult" not in source


def test_product_has_no_direct_replay_service_dependency() -> None:
    source = _product_source()

    assert "TopKReplayEvaluationService" not in source


def test_product_has_no_cli_dependency() -> None:
    source = _product_source()

    assert "lrp.cli" not in source
