from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.operations.durable_replay_artifact_discovery import (
    DurableReplayArtifactDiscoveryRequest,
    DurableReplayArtifactSelector,
)
from lrp.operations.durable_replay_execution import (
    DurableReplayExecutionRequest,
    DurableReplayExecutionSource,
)


def _product():
    return importlib.import_module(
        "lrp.operations.durable_replay_composition"
    )


def _selector(
    *,
    round_no: int,
    model_name: str,
    regime_id: str | None = None,
    strategy_name: str | None = None,
):
    return DurableReplayArtifactSelector(
        round_no=round_no,
        model_name=model_name,
        regime_id=regime_id,
        strategy_name=strategy_name,
    )


def _request():
    product = _product()

    return product.DurableReplayCompositionRequest(
        artifact_root="artifacts",
        history_path="data/history.csv",
        window_name="window-001",
        start_round=1001,
        end_round=1002,
        candidate_selectors=(
            _selector(
                round_no=1001,
                model_name="candidate-a",
                regime_id="regime-a",
                strategy_name="strategy-a",
            ),
            _selector(
                round_no=1002,
                model_name="candidate-b",
            ),
        ),
        baseline_selectors=(
            _selector(
                round_no=1001,
                model_name="baseline-a",
            ),
        ),
    )


def _candidate_sources():
    return (
        DurableReplayExecutionSource(
            artifact_path=Path(
                "artifacts/prediction-evaluation-sources/"
                "round_1001/evaluation_source.json"
            ),
            round_no=1001,
            model_name="candidate-a",
            regime_id="regime-a",
            strategy_name="strategy-a",
        ),
        DurableReplayExecutionSource(
            artifact_path=Path(
                "artifacts/prediction-evaluation-sources/"
                "round_1002/evaluation_source.json"
            ),
            round_no=1002,
            model_name="candidate-b",
        ),
    )


def _baseline_sources():
    return (
        DurableReplayExecutionSource(
            artifact_path=Path(
                "artifacts/prediction-evaluation-sources/"
                "round_1001/evaluation_source.json"
            ),
            round_no=1001,
            model_name="baseline-a",
        ),
    )


def _fake_result():
    return SimpleNamespace(
        evaluation="evaluation",
        candidate_model_name="candidate-model",
        baseline_model_name="baseline-model",
        round_count=2,
    )


def test_request_is_frozen() -> None:
    request = _request()

    with pytest.raises(
        dataclasses.FrozenInstanceError
    ):
        request.window_name = "changed"


def test_request_fields_are_exact() -> None:
    product = _product()

    fields = tuple(
        field.name
        for field in dataclasses.fields(
            product.DurableReplayCompositionRequest
        )
    )

    assert fields == (
        "artifact_root",
        "history_path",
        "window_name",
        "start_round",
        "end_round",
        "candidate_selectors",
        "baseline_selectors",
    )


def test_request_public_signature_is_exact() -> None:
    product = _product()

    assert str(
        inspect.signature(
            product.DurableReplayCompositionRequest
        )
    ) == (
        "(artifact_root: 'str | Path', "
        "history_path: 'str | Path', "
        "window_name: 'str', "
        "start_round: 'int', "
        "end_round: 'int', "
        "candidate_selectors: "
        "'tuple[DurableReplayArtifactSelector, ...]', "
        "baseline_selectors: "
        "'tuple[DurableReplayArtifactSelector, ...]') -> None"
    )


def test_service_is_parameterless() -> None:
    product = _product()

    assert str(
        inspect.signature(
            product.DurableReplayCompositionService
        )
    ) == "()"


def test_execute_public_signature_is_exact() -> None:
    product = _product()

    assert str(
        inspect.signature(
            product.DurableReplayCompositionService.execute
        )
    ) == (
        "(self, *, request: "
        "'DurableReplayCompositionRequest') "
        "-> 'TopKReplayEvaluationResult'"
    )


def test_execute_returns_downstream_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()
    result = _fake_result()

    monkeypatch.setattr(
        product.DurableReplayArtifactDiscoveryService,
        "discover",
        lambda self, **kwargs: (
            _candidate_sources(),
            _baseline_sources(),
        ),
    )

    monkeypatch.setattr(
        product.DurableReplayExecutionService,
        "execute",
        lambda self, **kwargs: result,
    )

    actual = (
        product.DurableReplayCompositionService()
        .execute(
            request=_request()
        )
    )

    assert actual is result


@pytest.mark.parametrize(
    "invalid",
    (
        None,
        object(),
        {},
        [],
        (),
        "request",
        1,
        True,
    ),
)
def test_execute_rejects_invalid_request_type(
    invalid,
) -> None:
    product = _product()

    with pytest.raises(
        ContractError,
        match=(
            "^request must be "
            "DurableReplayCompositionRequest$"
        ),
    ):
        product.DurableReplayCompositionService().execute(
            request=invalid
        )


def test_exact_an_request_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()
    observed = []

    def fake_discover(self, *, request):
        observed.append(request)
        return (
            _candidate_sources(),
            _baseline_sources(),
        )

    monkeypatch.setattr(
        product.DurableReplayArtifactDiscoveryService,
        "discover",
        fake_discover,
    )

    monkeypatch.setattr(
        product.DurableReplayExecutionService,
        "execute",
        lambda self, **kwargs: _fake_result(),
    )

    request = _request()

    product.DurableReplayCompositionService().execute(
        request=request
    )

    assert len(observed) == 1

    assert observed[0] == (
        DurableReplayArtifactDiscoveryRequest(
            artifact_root=request.artifact_root,
            candidate_selectors=request.candidate_selectors,
            baseline_selectors=request.baseline_selectors,
        )
    )


def test_an_service_is_constructed_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()
    builds = []

    class FakeDiscoveryService:
        def __init__(self):
            builds.append(object())

        def discover(self, *, request):
            return (
                _candidate_sources(),
                _baseline_sources(),
            )

    monkeypatch.setattr(
        product,
        "DurableReplayArtifactDiscoveryService",
        FakeDiscoveryService,
    )

    monkeypatch.setattr(
        product.DurableReplayExecutionService,
        "execute",
        lambda self, **kwargs: _fake_result(),
    )

    product.DurableReplayCompositionService().execute(
        request=_request()
    )

    assert len(builds) == 1


def test_an_discover_is_invoked_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()
    calls = []

    def fake_discover(self, *, request):
        calls.append(request)
        return (
            _candidate_sources(),
            _baseline_sources(),
        )

    monkeypatch.setattr(
        product.DurableReplayArtifactDiscoveryService,
        "discover",
        fake_discover,
    )

    monkeypatch.setattr(
        product.DurableReplayExecutionService,
        "execute",
        lambda self, **kwargs: _fake_result(),
    )

    product.DurableReplayCompositionService().execute(
        request=_request()
    )

    assert len(calls) == 1


def test_exact_al_request_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()

    candidate_sources = _candidate_sources()
    baseline_sources = _baseline_sources()
    observed = []

    monkeypatch.setattr(
        product.DurableReplayArtifactDiscoveryService,
        "discover",
        lambda self, **kwargs: (
            candidate_sources,
            baseline_sources,
        ),
    )

    def fake_execute(self, *, request):
        observed.append(request)
        return _fake_result()

    monkeypatch.setattr(
        product.DurableReplayExecutionService,
        "execute",
        fake_execute,
    )

    request = _request()

    product.DurableReplayCompositionService().execute(
        request=request
    )

    assert len(observed) == 1

    assert observed[0] == DurableReplayExecutionRequest(
        history_path=request.history_path,
        window_name=request.window_name,
        start_round=request.start_round,
        end_round=request.end_round,
        candidate_sources=candidate_sources,
        baseline_sources=baseline_sources,
    )


def test_al_service_is_constructed_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()
    builds = []

    monkeypatch.setattr(
        product.DurableReplayArtifactDiscoveryService,
        "discover",
        lambda self, **kwargs: (
            _candidate_sources(),
            _baseline_sources(),
        ),
    )

    class FakeExecutionService:
        def __init__(self):
            builds.append(object())

        def execute(self, *, request):
            return _fake_result()

    monkeypatch.setattr(
        product,
        "DurableReplayExecutionService",
        FakeExecutionService,
    )

    product.DurableReplayCompositionService().execute(
        request=_request()
    )

    assert len(builds) == 1


def test_al_execute_is_invoked_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()
    calls = []

    monkeypatch.setattr(
        product.DurableReplayArtifactDiscoveryService,
        "discover",
        lambda self, **kwargs: (
            _candidate_sources(),
            _baseline_sources(),
        ),
    )

    def fake_execute(self, *, request):
        calls.append(request)
        return _fake_result()

    monkeypatch.setattr(
        product.DurableReplayExecutionService,
        "execute",
        fake_execute,
    )

    product.DurableReplayCompositionService().execute(
        request=_request()
    )

    assert len(calls) == 1


def test_candidate_sources_are_forwarded_by_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()

    candidate_sources = _candidate_sources()
    baseline_sources = _baseline_sources()
    observed = []

    monkeypatch.setattr(
        product.DurableReplayArtifactDiscoveryService,
        "discover",
        lambda self, **kwargs: (
            candidate_sources,
            baseline_sources,
        ),
    )

    def fake_execute(self, *, request):
        observed.append(request)
        return _fake_result()

    monkeypatch.setattr(
        product.DurableReplayExecutionService,
        "execute",
        fake_execute,
    )

    product.DurableReplayCompositionService().execute(
        request=_request()
    )

    assert (
        observed[0].candidate_sources
        is candidate_sources
    )


def test_baseline_sources_are_forwarded_by_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()

    candidate_sources = _candidate_sources()
    baseline_sources = _baseline_sources()
    observed = []

    monkeypatch.setattr(
        product.DurableReplayArtifactDiscoveryService,
        "discover",
        lambda self, **kwargs: (
            candidate_sources,
            baseline_sources,
        ),
    )

    def fake_execute(self, *, request):
        observed.append(request)
        return _fake_result()

    monkeypatch.setattr(
        product.DurableReplayExecutionService,
        "execute",
        fake_execute,
    )

    product.DurableReplayCompositionService().execute(
        request=_request()
    )

    assert (
        observed[0].baseline_sources
        is baseline_sources
    )


def test_candidate_order_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()

    original = _candidate_sources()

    candidate_sources = (
        original[1],
        original[0],
    )

    observed = []

    monkeypatch.setattr(
        product.DurableReplayArtifactDiscoveryService,
        "discover",
        lambda self, **kwargs: (
            candidate_sources,
            _baseline_sources(),
        ),
    )

    def fake_execute(self, *, request):
        observed.append(request)
        return _fake_result()

    monkeypatch.setattr(
        product.DurableReplayExecutionService,
        "execute",
        fake_execute,
    )

    product.DurableReplayCompositionService().execute(
        request=_request()
    )

    assert (
        observed[0].candidate_sources
        == candidate_sources
    )


def test_baseline_order_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()

    baseline_sources = (
        DurableReplayExecutionSource(
            artifact_path="b2.json",
            round_no=2,
            model_name="b2",
        ),
        DurableReplayExecutionSource(
            artifact_path="b1.json",
            round_no=1,
            model_name="b1",
        ),
    )

    observed = []

    monkeypatch.setattr(
        product.DurableReplayArtifactDiscoveryService,
        "discover",
        lambda self, **kwargs: (
            _candidate_sources(),
            baseline_sources,
        ),
    )

    def fake_execute(self, *, request):
        observed.append(request)
        return _fake_result()

    monkeypatch.setattr(
        product.DurableReplayExecutionService,
        "execute",
        fake_execute,
    )

    product.DurableReplayCompositionService().execute(
        request=_request()
    )

    assert (
        observed[0].baseline_sources
        == baseline_sources
    )


def test_empty_sources_are_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()
    observed = []

    monkeypatch.setattr(
        product.DurableReplayArtifactDiscoveryService,
        "discover",
        lambda self, **kwargs: (
            (),
            (),
        ),
    )

    def fake_execute(self, *, request):
        observed.append(request)
        return _fake_result()

    monkeypatch.setattr(
        product.DurableReplayExecutionService,
        "execute",
        fake_execute,
    )

    product.DurableReplayCompositionService().execute(
        request=_request()
    )

    assert observed[0].candidate_sources == ()
    assert observed[0].baseline_sources == ()


def test_an_failure_identity_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()

    failure = RuntimeError(
        "AN failure"
    )

    def fake_discover(self, *, request):
        raise failure

    monkeypatch.setattr(
        product.DurableReplayArtifactDiscoveryService,
        "discover",
        fake_discover,
    )

    with pytest.raises(
        RuntimeError
    ) as captured:
        product.DurableReplayCompositionService().execute(
            request=_request()
        )

    assert captured.value is failure


def test_al_failure_identity_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()

    failure = RuntimeError(
        "AL failure"
    )

    monkeypatch.setattr(
        product.DurableReplayArtifactDiscoveryService,
        "discover",
        lambda self, **kwargs: (
            _candidate_sources(),
            _baseline_sources(),
        ),
    )

    def fake_execute(self, *, request):
        raise failure

    monkeypatch.setattr(
        product.DurableReplayExecutionService,
        "execute",
        fake_execute,
    )

    with pytest.raises(
        RuntimeError
    ) as captured:
        product.DurableReplayCompositionService().execute(
            request=_request()
        )

    assert captured.value is failure


def test_request_is_not_mutated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()

    monkeypatch.setattr(
        product.DurableReplayArtifactDiscoveryService,
        "discover",
        lambda self, **kwargs: (
            _candidate_sources(),
            _baseline_sources(),
        ),
    )

    monkeypatch.setattr(
        product.DurableReplayExecutionService,
        "execute",
        lambda self, **kwargs: _fake_result(),
    )

    request = _request()

    before = dataclasses.asdict(
        request
    )

    product.DurableReplayCompositionService().execute(
        request=request
    )

    after = dataclasses.asdict(
        request
    )

    assert before == after


def test_execute_is_semantically_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _product()

    candidate_sources = _candidate_sources()
    baseline_sources = _baseline_sources()

    monkeypatch.setattr(
        product.DurableReplayArtifactDiscoveryService,
        "discover",
        lambda self, **kwargs: (
            candidate_sources,
            baseline_sources,
        ),
    )

    result = _fake_result()

    monkeypatch.setattr(
        product.DurableReplayExecutionService,
        "execute",
        lambda self, **kwargs: result,
    )

    service = (
        product.DurableReplayCompositionService()
    )

    request = _request()

    first = service.execute(
        request=request
    )

    second = service.execute(
        request=request
    )

    assert first == second


def test_product_has_no_direct_filesystem_io() -> None:
    source = Path(
        "lrp/operations/durable_replay_composition.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    calls = tuple(
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    )

    forbidden = {
        "open",
        "Path.open",
        "Path.read_text",
        "Path.read_bytes",
        "Path.write_text",
        "Path.write_bytes",
        "Path.mkdir",
    }

    assert not (
        forbidden
        & set(calls)
    )


def test_product_has_no_direct_json_dependency() -> None:
    source = Path(
        "lrp/operations/durable_replay_composition.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    modules = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(
                alias.name
                for alias in node.names
            )

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(
                    node.module
                )

    assert "json" not in modules


def test_product_has_no_history_io_ownership() -> None:
    source = Path(
        "lrp/operations/durable_replay_composition.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    assert "load_history" not in source
    assert "history_until_round" not in source
    assert "lrp.io.draws" not in source


def test_product_has_no_evaluation_window_ownership() -> None:
    source = Path(
        "lrp/operations/durable_replay_composition.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    assert "EvaluationWindow" not in source


def test_product_has_no_replay_core_ownership() -> None:
    source = Path(
        "lrp/operations/durable_replay_composition.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "DurableReplayOperationalConsumer",
        "DurableReplayEvaluationOrchestrator",
        "TopKReplayEvaluationService",
        "PredictionResult",
    )

    assert not any(
        token in source
        for token in forbidden
    )


def test_product_has_no_cli_dependency() -> None:
    source = Path(
        "lrp/operations/durable_replay_composition.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    assert "argparse" not in source
    assert "lrp.cli" not in source


def test_product_has_no_validation_tool_dependency() -> None:
    source = Path(
        "lrp/operations/durable_replay_composition.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    assert "tools.validation" not in source


def test_product_has_no_exception_normalization_layer() -> None:
    source = Path(
        "lrp/operations/durable_replay_composition.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    handlers = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.ExceptHandler,
        )
    )

    assert handlers == ()


def test_product_has_exact_one_owned_raise_site() -> None:
    source = Path(
        "lrp/operations/durable_replay_composition.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    raises = tuple(
        ast.unparse(node.exc)
        for node in ast.walk(tree)
        if isinstance(node, ast.Raise)
        and node.exc is not None
    )

    assert raises == (
        "ContractError('request must be DurableReplayCompositionRequest')",
    )


def test_product_structural_call_contract_is_exact() -> None:
    source = Path(
        "lrp/operations/durable_replay_composition.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    calls = tuple(
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    )

    assert (
        calls.count(
            "DurableReplayArtifactDiscoveryRequest"
        )
        == 1
    )

    assert (
        calls.count(
            "DurableReplayArtifactDiscoveryService"
        )
        == 1
    )

    assert (
        sum(
            call.endswith(".discover")
            for call in calls
        )
        == 1
    )

    assert (
        calls.count(
            "DurableReplayExecutionRequest"
        )
        == 1
    )

    assert (
        calls.count(
            "DurableReplayExecutionService"
        )
        == 1
    )

    assert (
        sum(
            call.endswith(".execute")
            for call in calls
        )
        == 1
    )


def test_product_public_surface_remains_minimal() -> None:
    source = Path(
        "lrp/operations/durable_replay_composition.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(source)

    classes = tuple(
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    )

    functions = tuple(
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    )

    methods = {}

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods[node.name] = tuple(
                child.name
                for child in node.body
                if isinstance(
                    child,
                    ast.FunctionDef,
                )
                and not child.name.startswith("_")
            )

    assert classes == (
        "DurableReplayCompositionRequest",
        "DurableReplayCompositionService",
    )

    assert functions == ()

    assert methods == {
        "DurableReplayCompositionRequest": (),
        "DurableReplayCompositionService": (
            "execute",
        ),
    }
