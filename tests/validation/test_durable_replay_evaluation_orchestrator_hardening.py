from __future__ import annotations

from pathlib import Path
import ast

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.evaluation.contracts import EvaluationWindow
from lrp.operations.durable_replay_evaluation_orchestrator import (
    DurableReplayEvaluationOrchestrator,
    DurableReplayEvaluationSourceSpec,
)
import lrp.operations.durable_replay_evaluation_orchestrator as product


PRODUCT_PATH = Path(
    "lrp/operations/"
    "durable_replay_evaluation_orchestrator.py"
)


def _window() -> EvaluationWindow:
    return EvaluationWindow(
        name="ak-hardening",
        start_round=1233,
        end_round=1234,
    )


def _spec(
    *,
    artifact_path: str | Path,
    model_name: str,
    history_rounds: tuple[int, ...] = (1200, 1201),
    regime_id: str | None = None,
    strategy_name: str | None = None,
) -> DurableReplayEvaluationSourceSpec:
    return DurableReplayEvaluationSourceSpec(
        artifact_path=artifact_path,
        history_rounds=history_rounds,
        model_name=model_name,
        regime_id=regime_id,
        strategy_name=strategy_name,
    )


def _source_text() -> str:
    return PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )


def test_candidate_source_order_is_preserved(
    monkeypatch,
) -> None:
    observed = []

    def fake_load(self, **kwargs):
        observed.append(
            kwargs["artifact_path"]
        )
        return object()

    monkeypatch.setattr(
        product.DurableReplayOperationalConsumer,
        "load",
        fake_load,
    )

    monkeypatch.setattr(
        product.TopKReplayEvaluationService,
        "evaluate",
        lambda self, *, request: request,
    )

    sources = (
        _spec(
            artifact_path="c-3.json",
            model_name="candidate",
        ),
        _spec(
            artifact_path="c-1.json",
            model_name="candidate",
        ),
        _spec(
            artifact_path="c-2.json",
            model_name="candidate",
        ),
    )

    DurableReplayEvaluationOrchestrator().evaluate(
        window=_window(),
        candidate_sources=sources,
        baseline_sources=(),
        actual_draws=(),
    )

    assert observed == [
        "c-3.json",
        "c-1.json",
        "c-2.json",
    ]


def test_baseline_source_order_is_preserved(
    monkeypatch,
) -> None:
    observed = []

    def fake_load(self, **kwargs):
        observed.append(
            kwargs["artifact_path"]
        )
        return object()

    monkeypatch.setattr(
        product.DurableReplayOperationalConsumer,
        "load",
        fake_load,
    )

    monkeypatch.setattr(
        product.TopKReplayEvaluationService,
        "evaluate",
        lambda self, *, request: request,
    )

    sources = (
        _spec(
            artifact_path="b-2.json",
            model_name="baseline",
        ),
        _spec(
            artifact_path="b-1.json",
            model_name="baseline",
        ),
    )

    DurableReplayEvaluationOrchestrator().evaluate(
        window=_window(),
        candidate_sources=(),
        baseline_sources=sources,
        actual_draws=(),
    )

    assert observed == [
        "b-2.json",
        "b-1.json",
    ]


def test_candidate_context_is_passed_exactly_to_consumer(
    monkeypatch,
) -> None:
    observed = []

    def fake_load(self, **kwargs):
        observed.append(kwargs)
        return object()

    monkeypatch.setattr(
        product.DurableReplayOperationalConsumer,
        "load",
        fake_load,
    )

    monkeypatch.setattr(
        product.TopKReplayEvaluationService,
        "evaluate",
        lambda self, *, request: request,
    )

    source = _spec(
        artifact_path=Path("candidate.json"),
        history_rounds=(100, 101, 102),
        model_name="candidate-model",
        regime_id="candidate-regime",
        strategy_name="candidate-strategy",
    )

    DurableReplayEvaluationOrchestrator().evaluate(
        window=_window(),
        candidate_sources=(source,),
        baseline_sources=(),
        actual_draws=(),
    )

    assert observed == [
        {
            "artifact_path": Path(
                "candidate.json"
            ),
            "history_rounds": (
                100,
                101,
                102,
            ),
            "model_name": "candidate-model",
            "regime_id": "candidate-regime",
            "strategy_name": "candidate-strategy",
        }
    ]


def test_baseline_context_is_passed_exactly_to_consumer(
    monkeypatch,
) -> None:
    observed = []

    def fake_load(self, **kwargs):
        observed.append(kwargs)
        return object()

    monkeypatch.setattr(
        product.DurableReplayOperationalConsumer,
        "load",
        fake_load,
    )

    monkeypatch.setattr(
        product.TopKReplayEvaluationService,
        "evaluate",
        lambda self, *, request: request,
    )

    source = _spec(
        artifact_path=Path("baseline.json"),
        history_rounds=(200, 201),
        model_name="baseline-model",
        regime_id="baseline-regime",
        strategy_name="baseline-strategy",
    )

    DurableReplayEvaluationOrchestrator().evaluate(
        window=_window(),
        candidate_sources=(),
        baseline_sources=(source,),
        actual_draws=(),
    )

    assert observed == [
        {
            "artifact_path": Path(
                "baseline.json"
            ),
            "history_rounds": (
                200,
                201,
            ),
            "model_name": "baseline-model",
            "regime_id": "baseline-regime",
            "strategy_name": "baseline-strategy",
        }
    ]


def test_exact_request_projection_is_preserved(
    monkeypatch,
) -> None:
    loaded = [
        object(),
        object(),
        object(),
    ]

    iterator = iter(loaded)
    captured = {}

    def fake_load(self, **kwargs):
        return next(iterator)

    def fake_evaluate(self, *, request):
        captured["request"] = request
        return request

    monkeypatch.setattr(
        product.DurableReplayOperationalConsumer,
        "load",
        fake_load,
    )

    monkeypatch.setattr(
        product.TopKReplayEvaluationService,
        "evaluate",
        fake_evaluate,
    )

    window = _window()
    actual_draws = (
        object(),
        object(),
    )

    result = DurableReplayEvaluationOrchestrator().evaluate(
        window=window,
        candidate_sources=(
            _spec(
                artifact_path="c1.json",
                model_name="candidate",
            ),
            _spec(
                artifact_path="c2.json",
                model_name="candidate",
            ),
        ),
        baseline_sources=(
            _spec(
                artifact_path="b1.json",
                model_name="baseline",
            ),
        ),
        actual_draws=actual_draws,
    )

    request = captured["request"]

    assert request.window is window
    assert (
        request.candidate_predictions
        == tuple(loaded[:2])
    )
    assert (
        request.baseline_predictions
        == tuple(loaded[2:])
    )
    assert request.actual_draws is actual_draws
    assert result is request


def test_replay_service_is_invoked_exactly_once(
    monkeypatch,
) -> None:
    calls = []

    def fake_evaluate(self, *, request):
        calls.append(request)
        return object()

    monkeypatch.setattr(
        product.TopKReplayEvaluationService,
        "evaluate",
        fake_evaluate,
    )

    DurableReplayEvaluationOrchestrator().evaluate(
        window=_window(),
        candidate_sources=(),
        baseline_sources=(),
        actual_draws=(),
    )

    assert len(calls) == 1


def test_empty_sources_are_forwarded_to_replay_service(
    monkeypatch,
) -> None:
    captured = {}

    def fake_evaluate(self, *, request):
        captured["request"] = request
        return request

    monkeypatch.setattr(
        product.TopKReplayEvaluationService,
        "evaluate",
        fake_evaluate,
    )

    result = DurableReplayEvaluationOrchestrator().evaluate(
        window=_window(),
        candidate_sources=(),
        baseline_sources=(),
        actual_draws=(),
    )

    assert result is captured["request"]
    assert result.candidate_predictions == ()
    assert result.baseline_predictions == ()


def test_orchestrator_has_no_explicit_empty_source_guard() -> None:
    source = _source_text()

    forbidden = (
        "if not candidate_sources",
        "if not baseline_sources",
        "len(candidate_sources)",
        "len(baseline_sources)",
    )

    assert not any(
        token in source
        for token in forbidden
    )


def test_candidate_consumer_failure_identity_is_preserved(
    monkeypatch,
) -> None:
    error = FileNotFoundError(
        "candidate source missing"
    )

    def fail(self, **kwargs):
        raise error

    monkeypatch.setattr(
        product.DurableReplayOperationalConsumer,
        "load",
        fail,
    )

    with pytest.raises(
        FileNotFoundError
    ) as captured:
        DurableReplayEvaluationOrchestrator().evaluate(
            window=_window(),
            candidate_sources=(
                _spec(
                    artifact_path="missing.json",
                    model_name="candidate",
                ),
            ),
            baseline_sources=(),
            actual_draws=(),
        )

    assert captured.value is error


def test_baseline_consumer_failure_identity_is_preserved(
    monkeypatch,
) -> None:
    error = PermissionError(
        "baseline unreadable"
    )

    calls = 0

    def fake_load(self, **kwargs):
        nonlocal calls
        calls += 1

        if calls == 1:
            return object()

        raise error

    monkeypatch.setattr(
        product.DurableReplayOperationalConsumer,
        "load",
        fake_load,
    )

    with pytest.raises(
        PermissionError
    ) as captured:
        DurableReplayEvaluationOrchestrator().evaluate(
            window=_window(),
            candidate_sources=(
                _spec(
                    artifact_path="candidate.json",
                    model_name="candidate",
                ),
            ),
            baseline_sources=(
                _spec(
                    artifact_path="baseline.json",
                    model_name="baseline",
                ),
            ),
            actual_draws=(),
        )

    assert captured.value is error


def test_replay_service_failure_identity_is_preserved(
    monkeypatch,
) -> None:
    error = ContractError(
        "replay service failure"
    )

    def fail(self, *, request):
        raise error

    monkeypatch.setattr(
        product.TopKReplayEvaluationService,
        "evaluate",
        fail,
    )

    with pytest.raises(
        ContractError
    ) as captured:
        DurableReplayEvaluationOrchestrator().evaluate(
            window=_window(),
            candidate_sources=(),
            baseline_sources=(),
            actual_draws=(),
        )

    assert captured.value is error


def test_repeated_execution_is_semantically_stable(
    monkeypatch,
) -> None:
    def fake_load(self, **kwargs):
        return (
            str(kwargs["artifact_path"]),
            kwargs["history_rounds"],
            kwargs["model_name"],
            kwargs["regime_id"],
            kwargs["strategy_name"],
        )

    def fake_evaluate(self, *, request):
        return (
            request.window,
            request.candidate_predictions,
            request.baseline_predictions,
            request.actual_draws,
        )

    monkeypatch.setattr(
        product.DurableReplayOperationalConsumer,
        "load",
        fake_load,
    )

    monkeypatch.setattr(
        product.TopKReplayEvaluationService,
        "evaluate",
        fake_evaluate,
    )

    kwargs = {
        "window": _window(),
        "candidate_sources": (
            _spec(
                artifact_path="candidate.json",
                model_name="candidate",
            ),
        ),
        "baseline_sources": (
            _spec(
                artifact_path="baseline.json",
                model_name="baseline",
            ),
        ),
        "actual_draws": (
            "draw-A",
            "draw-B",
        ),
    }

    first = (
        DurableReplayEvaluationOrchestrator()
        .evaluate(**kwargs)
    )

    second = (
        DurableReplayEvaluationOrchestrator()
        .evaluate(**kwargs)
    )

    assert first == second


def test_inputs_are_not_mutated(
    monkeypatch,
) -> None:
    candidate = (
        _spec(
            artifact_path="candidate.json",
            model_name="candidate",
        ),
    )

    baseline = (
        _spec(
            artifact_path="baseline.json",
            model_name="baseline",
        ),
    )

    actual_draws = (
        object(),
        object(),
    )

    candidate_before = tuple(candidate)
    baseline_before = tuple(baseline)
    actual_before = tuple(actual_draws)

    monkeypatch.setattr(
        product.DurableReplayOperationalConsumer,
        "load",
        lambda self, **kwargs: object(),
    )

    monkeypatch.setattr(
        product.TopKReplayEvaluationService,
        "evaluate",
        lambda self, *, request: request,
    )

    DurableReplayEvaluationOrchestrator().evaluate(
        window=_window(),
        candidate_sources=candidate,
        baseline_sources=baseline,
        actual_draws=actual_draws,
    )

    assert candidate == candidate_before
    assert baseline == baseline_before
    assert actual_draws == actual_before


def test_product_has_exact_dependency_boundary() -> None:
    tree = ast.parse(
        _source_text()
    )

    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                alias.name
                for alias in node.names
            )

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(
                    node.module
                )

    assert set(imports) == {
        "__future__",
        "dataclasses",
        "pathlib",
        "typing",
        "lrp.contracts.exceptions",
        "lrp.evaluation.contracts",
        "lrp.evaluation.topk_replay_evaluation",
        "lrp.operations.durable_replay_consumer",
    }


def test_product_has_no_direct_filesystem_access() -> None:
    source = _source_text()

    forbidden = (
        "read_text(",
        "read_bytes(",
        "open(",
        "write_text(",
        "write_bytes(",
        "mkdir(",
        "unlink(",
        "rename(",
        "replace(",
    )

    assert not any(
        token in source
        for token in forbidden
    )


def test_product_has_no_exception_normalization_layer() -> None:
    tree = ast.parse(
        _source_text()
    )

    handlers = [
        node
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.ExceptHandler,
        )
    ]

    assert handlers == []


def test_product_has_exact_single_owned_raise() -> None:
    tree = ast.parse(
        _source_text()
    )

    raises = [
        node
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.Raise,
        )
    ]

    assert len(raises) == 1


def test_product_has_no_lower_layer_ownership_leak() -> None:
    source = _source_text()

    forbidden = (
        "DurablePredictionEvaluationSource",
        "source_from_json",
        "TopKDurableReplayAdapter",
        "PredictionResult",
        "TopKLiveEvaluationOrchestrator",
        "evaluation_source.json",
        "prediction-evaluation-sources",
        "write_operation_artifact",
        "write_prediction_artifacts",
        "load_history",
        "load_actual",
        "lrp.cli",
    )

    assert not any(
        token in source
        for token in forbidden
    )