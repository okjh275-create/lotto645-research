from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.io.draws import HistoryRow
from lrp.operations import durable_replay_execution as product


def _history() -> tuple[HistoryRow, ...]:
    return tuple(
        HistoryRow(
            round_no=round_no,
            numbers=(
                1,
                2,
                3,
                4,
                5,
                ((round_no - 1) % 45) + 1,
            ),
        )
        for round_no in range(100, 107)
    )


def _source(
    *,
    artifact_path: str = "source.json",
    round_no: int = 105,
    model_name: str = "model-A",
    regime_id: str | None = None,
    strategy_name: str | None = None,
):
    return product.DurableReplayExecutionSource(
        artifact_path=artifact_path,
        round_no=round_no,
        model_name=model_name,
        regime_id=regime_id,
        strategy_name=strategy_name,
    )


def _request(
    *,
    candidate_sources=None,
    baseline_sources=None,
    start_round: int = 102,
    end_round: int = 105,
):
    if candidate_sources is None:
        candidate_sources = (
            _source(
                artifact_path="candidate.json",
                round_no=105,
                model_name="candidate",
            ),
        )

    if baseline_sources is None:
        baseline_sources = (
            _source(
                artifact_path="baseline.json",
                round_no=104,
                model_name="baseline",
            ),
        )

    return product.DurableReplayExecutionRequest(
        history_path="history.json",
        window_name="window-A",
        start_round=start_round,
        end_round=end_round,
        candidate_sources=candidate_sources,
        baseline_sources=baseline_sources,
    )


def _result():
    return SimpleNamespace(
        evaluation="evaluation-A",
        candidate_model_name="candidate",
        baseline_model_name="baseline",
        round_count=4,
    )


def test_mixed_round_sources_project_exact_history_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: _history(),
    )

    def fake_evaluate(self, **kwargs):
        captured.update(kwargs)
        return _result()

    monkeypatch.setattr(
        product.DurableReplayEvaluationOrchestrator,
        "evaluate",
        fake_evaluate,
    )

    request = _request(
        candidate_sources=(
            _source(
                artifact_path="c106.json",
                round_no=106,
                model_name="c106",
            ),
            _source(
                artifact_path="c102.json",
                round_no=102,
                model_name="c102",
            ),
        ),
        baseline_sources=(
            _source(
                artifact_path="b104.json",
                round_no=104,
                model_name="b104",
            ),
            _source(
                artifact_path="b103.json",
                round_no=103,
                model_name="b103",
            ),
        ),
    )

    product.DurableReplayExecutionService().execute(
        request=request
    )

    assert tuple(
        spec.history_rounds
        for spec in captured["candidate_sources"]
    ) == (
        (100, 101, 102, 103, 104, 105),
        (100, 101),
    )

    assert tuple(
        spec.history_rounds
        for spec in captured["baseline_sources"]
    ) == (
        (100, 101, 102, 103),
        (100, 101, 102),
    )


def test_actual_draw_projection_is_exactly_inclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: _history(),
    )

    def fake_evaluate(self, **kwargs):
        captured.update(kwargs)
        return _result()

    monkeypatch.setattr(
        product.DurableReplayEvaluationOrchestrator,
        "evaluate",
        fake_evaluate,
    )

    product.DurableReplayExecutionService().execute(
        request=_request(
            start_round=102,
            end_round=105,
        )
    )

    assert tuple(
        row.round_no
        for row in captured["actual_draws"]
    ) == (
        102,
        103,
        104,
        105,
    )


@pytest.mark.parametrize(
    (
        "candidate_sources",
        "baseline_sources",
        "expected_counts",
    ),
    (
        ((), (), (0, 0)),
        (
            (_source(model_name="candidate"),),
            (),
            (1, 0),
        ),
        (
            (),
            (_source(model_name="baseline"),),
            (0, 1),
        ),
    ),
)
def test_empty_source_shapes_are_forwarded_downstream(
    monkeypatch: pytest.MonkeyPatch,
    candidate_sources,
    baseline_sources,
    expected_counts,
) -> None:
    captured = {}

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: _history(),
    )

    def fake_evaluate(self, **kwargs):
        captured.update(kwargs)
        return _result()

    monkeypatch.setattr(
        product.DurableReplayEvaluationOrchestrator,
        "evaluate",
        fake_evaluate,
    )

    product.DurableReplayExecutionService().execute(
        request=_request(
            candidate_sources=candidate_sources,
            baseline_sources=baseline_sources,
        )
    )

    assert (
        len(captured["candidate_sources"]),
        len(captured["baseline_sources"]),
    ) == expected_counts


def test_non_monotonic_source_round_order_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: _history(),
    )

    def fake_evaluate(self, **kwargs):
        captured.update(kwargs)
        return _result()

    monkeypatch.setattr(
        product.DurableReplayEvaluationOrchestrator,
        "evaluate",
        fake_evaluate,
    )

    product.DurableReplayExecutionService().execute(
        request=_request(
            candidate_sources=(
                _source(
                    artifact_path="c106.json",
                    round_no=106,
                    model_name="c106",
                ),
                _source(
                    artifact_path="c102.json",
                    round_no=102,
                    model_name="c102",
                ),
                _source(
                    artifact_path="c105.json",
                    round_no=105,
                    model_name="c105",
                ),
            ),
            baseline_sources=(
                _source(
                    artifact_path="b105.json",
                    round_no=105,
                    model_name="b105",
                ),
                _source(
                    artifact_path="b103.json",
                    round_no=103,
                    model_name="b103",
                ),
            ),
        )
    )

    assert tuple(
        spec.model_name
        for spec in captured["candidate_sources"]
    ) == (
        "c106",
        "c102",
        "c105",
    )

    assert tuple(
        spec.model_name
        for spec in captured["baseline_sources"]
    ) == (
        "b105",
        "b103",
    )


def test_load_history_failure_identity_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = FileNotFoundError(
        "history failure"
    )

    def fail(path):
        raise failure

    monkeypatch.setattr(
        product,
        "load_history",
        fail,
    )

    with pytest.raises(
        FileNotFoundError
    ) as exc_info:
        product.DurableReplayExecutionService().execute(
            request=_request()
        )

    assert exc_info.value is failure


def test_history_projection_failure_identity_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = ContractError(
        "projection failure"
    )

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: _history(),
    )

    def fail(
        rows,
        *,
        target_round,
    ):
        raise failure

    monkeypatch.setattr(
        product,
        "history_until_round",
        fail,
    )

    with pytest.raises(
        ContractError
    ) as exc_info:
        product.DurableReplayExecutionService().execute(
            request=_request()
        )

    assert exc_info.value is failure


def test_ak_failure_identity_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = ContractError(
        "AK failure"
    )

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: _history(),
    )

    def fail(self, **kwargs):
        raise failure

    monkeypatch.setattr(
        product.DurableReplayEvaluationOrchestrator,
        "evaluate",
        fail,
    )

    with pytest.raises(
        ContractError
    ) as exc_info:
        product.DurableReplayExecutionService().execute(
            request=_request()
        )

    assert exc_info.value is failure


def test_request_and_history_are_not_mutated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = _history()
    request = _request()

    history_before = copy.deepcopy(history)
    request_before = copy.deepcopy(request)

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: history,
    )

    monkeypatch.setattr(
        product.DurableReplayEvaluationOrchestrator,
        "evaluate",
        lambda self, **kwargs: _result(),
    )

    product.DurableReplayExecutionService().execute(
        request=request
    )

    assert history == history_before
    assert request == request_before


def test_repeated_execution_is_semantically_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    monkeypatch.setattr(
        product,
        "load_history",
        lambda path: _history(),
    )

    def fake_evaluate(self, **kwargs):
        calls.append(kwargs)
        return _result()

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


def test_product_has_exact_two_owned_raise_sites() -> None:
    path = (
        Path(__file__).parents[2]
        / "lrp"
        / "operations"
        / "durable_replay_execution.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    raises = [
        ast.unparse(node.exc)
        for node in ast.walk(tree)
        if isinstance(node, ast.Raise)
        and node.exc is not None
    ]

    assert raises == [
        (
            "ContractError("
            "'request must be DurableReplayExecutionRequest'"
            ")"
        ),
        (
            "ContractError("
            "f'{label} source must be "
            "DurableReplayExecutionSource'"
            ")"
        ),
    ]


def test_product_has_no_exception_normalization_layer() -> None:
    path = (
        Path(__file__).parents[2]
        / "lrp"
        / "operations"
        / "durable_replay_execution.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    handlers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
    ]

    assert handlers == []


def test_product_static_dependency_boundary_is_exact() -> None:
    path = (
        Path(__file__).parents[2]
        / "lrp"
        / "operations"
        / "durable_replay_execution.py"
    )

    source = path.read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                alias.name
                for alias in node.names
            )

        elif isinstance(
            node,
            ast.ImportFrom,
        ):
            if node.module:
                imports.append(
                    node.module
                )

    assert set(imports) == {
        "__future__",
        "dataclasses",
        "pathlib",
        "lrp.contracts.exceptions",
        "lrp.evaluation.contracts",
        "lrp.evaluation.topk_replay_evaluation",
        "lrp.io.draws",
        (
            "lrp.operations."
            "durable_replay_evaluation_orchestrator"
        ),
    }


def test_product_has_no_lower_layer_ownership_leak() -> None:
    path = (
        Path(__file__).parents[2]
        / "lrp"
        / "operations"
        / "durable_replay_execution.py"
    )

    source = path.read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "import json",
        "json.loads",
        "json.load",
        ".read_text(",
        ".read_bytes(",
        "open(",
        ".write_text(",
        ".write_bytes(",
        ".mkdir(",
        ".glob(",
        ".rglob(",
        ".iterdir(",
        "operation_log",
        "manifest",
        "evaluation_source.json",
        "prediction-evaluation-sources",
        "PredictionResult",
        "source_from_json",
        "source_from_dict",
        "TopKDurableReplayAdapter",
        "TopKReplayEvaluationService",
        "write_operation_artifact",
        "write_prediction_artifacts",
        "lrp.cli",
    )

    assert not any(
        token in source
        for token in forbidden
    )


def test_structural_call_contract_is_exact() -> None:
    path = (
        Path(__file__).parents[2]
        / "lrp"
        / "operations"
        / "durable_replay_execution.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    calls = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            try:
                calls.append(
                    ast.unparse(node.func)
                )
            except Exception:
                pass

    assert calls.count(
        "load_history"
    ) == 1

    assert calls.count(
        "EvaluationWindow"
    ) == 1

    assert calls.count(
        "history_until_round"
    ) == 1

    assert calls.count(
        "DurableReplayEvaluationSourceSpec"
    ) == 1

    assert calls.count(
        "DurableReplayEvaluationOrchestrator"
    ) == 1

    assert calls.count(
        "DurableReplayEvaluationOrchestrator().evaluate"
    ) == 1


def test_product_public_surface_remains_minimal() -> None:
    path = (
        Path(__file__).parents[2]
        / "lrp"
        / "operations"
        / "durable_replay_execution.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    classes = tuple(
        node.name
        for node in tree.body
        if isinstance(
            node,
            ast.ClassDef,
        )
    )

    functions = tuple(
        node.name
        for node in tree.body
        if isinstance(
            node,
            ast.FunctionDef,
        )
    )

    assert classes == (
        "DurableReplayExecutionSource",
        "DurableReplayExecutionRequest",
        "DurableReplayExecutionService",
    )

    assert functions == ()
