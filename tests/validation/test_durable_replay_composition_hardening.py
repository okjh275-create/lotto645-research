from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path

import pytest

from lrp.operations import durable_replay_composition as product
from lrp.operations.durable_replay_artifact_discovery import (
    DurableReplayArtifactDiscoveryRequest,
    DurableReplayArtifactSelector,
)
from lrp.operations.durable_replay_execution import (
    DurableReplayExecutionRequest,
    DurableReplayExecutionSource,
)


def _selector(
    round_no: int,
    model_name: str,
    regime_id: str | None = None,
    strategy_name: str | None = None,
) -> DurableReplayArtifactSelector:
    return DurableReplayArtifactSelector(
        round_no=round_no,
        model_name=model_name,
        regime_id=regime_id,
        strategy_name=strategy_name,
    )


def _request(
    *,
    candidate_selectors=None,
    baseline_selectors=None,
) -> product.DurableReplayCompositionRequest:
    if candidate_selectors is None:
        candidate_selectors = (
            _selector(
                1300,
                "candidate-z",
                "regime-z",
                "strategy-z",
            ),
            _selector(
                1001,
                "candidate-a",
            ),
        )

    if baseline_selectors is None:
        baseline_selectors = (
            _selector(
                1250,
                "baseline-b",
            ),
            _selector(
                1050,
                "baseline-a",
                "baseline-regime",
                "baseline-strategy",
            ),
        )

    return product.DurableReplayCompositionRequest(
        artifact_root="artifact-root",
        history_path="history.json",
        window_name="window-hardening",
        start_round=1000,
        end_round=1300,
        candidate_selectors=candidate_selectors,
        baseline_selectors=baseline_selectors,
    )


def _candidate_sources():
    return (
        DurableReplayExecutionSource(
            artifact_path="candidate-z.json",
            round_no=1300,
            model_name="candidate-z",
            regime_id="regime-z",
            strategy_name="strategy-z",
        ),
        DurableReplayExecutionSource(
            artifact_path="candidate-a.json",
            round_no=1001,
            model_name="candidate-a",
        ),
    )


def _baseline_sources():
    return (
        DurableReplayExecutionSource(
            artifact_path="baseline-b.json",
            round_no=1250,
            model_name="baseline-b",
        ),
        DurableReplayExecutionSource(
            artifact_path="baseline-a.json",
            round_no=1050,
            model_name="baseline-a",
            regime_id="baseline-regime",
            strategy_name="baseline-strategy",
        ),
    )


def _run_with_projection(
    monkeypatch: pytest.MonkeyPatch,
    *,
    request,
    candidate_sources,
    baseline_sources,
):
    discovery_requests = []
    execution_requests = []
    result = object()

    def fake_discover(self, *, request):
        discovery_requests.append(request)
        return candidate_sources, baseline_sources

    def fake_execute(self, *, request):
        execution_requests.append(request)
        return result

    monkeypatch.setattr(
        product.DurableReplayArtifactDiscoveryService,
        "discover",
        fake_discover,
    )

    monkeypatch.setattr(
        product.DurableReplayExecutionService,
        "execute",
        fake_execute,
    )

    actual = (
        product.DurableReplayCompositionService()
        .execute(
            request=request
        )
    )

    return (
        actual,
        result,
        discovery_requests,
        execution_requests,
    )


def test_hardening_selector_tuple_identity_into_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = (
        _selector(1300, "c-z"),
        _selector(1001, "c-a"),
    )

    baseline = (
        _selector(1250, "b-b"),
        _selector(1050, "b-a"),
    )

    request = _request(
        candidate_selectors=candidate,
        baseline_selectors=baseline,
    )

    _, _, discovery_requests, _ = _run_with_projection(
        monkeypatch,
        request=request,
        candidate_sources=(),
        baseline_sources=(),
    )

    projected = discovery_requests[0]

    assert projected.candidate_selectors is candidate
    assert projected.baseline_selectors is baseline


def test_hardening_source_tuple_identity_into_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_sources = _candidate_sources()
    baseline_sources = _baseline_sources()

    _, _, _, execution_requests = _run_with_projection(
        monkeypatch,
        request=_request(),
        candidate_sources=candidate_sources,
        baseline_sources=baseline_sources,
    )

    projected = execution_requests[0]

    assert projected.candidate_sources is candidate_sources
    assert projected.baseline_sources is baseline_sources


def test_hardening_mixed_selector_order_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = (
        _selector(1400, "c4"),
        _selector(900, "c0"),
        _selector(1200, "c2"),
    )

    baseline = (
        _selector(1350, "b3"),
        _selector(950, "b0"),
        _selector(1100, "b1"),
    )

    request = _request(
        candidate_selectors=candidate,
        baseline_selectors=baseline,
    )

    _, _, discovery_requests, _ = _run_with_projection(
        monkeypatch,
        request=request,
        candidate_sources=(),
        baseline_sources=(),
    )

    projected = discovery_requests[0]

    assert tuple(
        item.round_no
        for item in projected.candidate_selectors
    ) == (
        1400,
        900,
        1200,
    )

    assert tuple(
        item.round_no
        for item in projected.baseline_selectors
    ) == (
        1350,
        950,
        1100,
    )


def test_hardening_mixed_source_order_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_sources = (
        DurableReplayExecutionSource(
            artifact_path="c3.json",
            round_no=1400,
            model_name="c3",
        ),
        DurableReplayExecutionSource(
            artifact_path="c0.json",
            round_no=900,
            model_name="c0",
        ),
        DurableReplayExecutionSource(
            artifact_path="c2.json",
            round_no=1200,
            model_name="c2",
        ),
    )

    baseline_sources = (
        DurableReplayExecutionSource(
            artifact_path="b3.json",
            round_no=1350,
            model_name="b3",
        ),
        DurableReplayExecutionSource(
            artifact_path="b0.json",
            round_no=950,
            model_name="b0",
        ),
    )

    _, _, _, execution_requests = _run_with_projection(
        monkeypatch,
        request=_request(),
        candidate_sources=candidate_sources,
        baseline_sources=baseline_sources,
    )

    projected = execution_requests[0]

    assert tuple(
        item.round_no
        for item in projected.candidate_sources
    ) == (
        1400,
        900,
        1200,
    )

    assert tuple(
        item.round_no
        for item in projected.baseline_sources
    ) == (
        1350,
        950,
    )


def test_hardening_empty_source_tuple_shape_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, execution_requests = _run_with_projection(
        monkeypatch,
        request=_request(
            candidate_selectors=(),
            baseline_selectors=(),
        ),
        candidate_sources=(),
        baseline_sources=(),
    )

    projected = execution_requests[0]

    assert projected.candidate_sources == ()
    assert projected.baseline_sources == ()
    assert isinstance(
        projected.candidate_sources,
        tuple,
    )
    assert isinstance(
        projected.baseline_sources,
        tuple,
    )


def test_hardening_discovery_failure_identity_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError(
        "discovery sentinel"
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


def test_hardening_execution_failure_identity_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError(
        "execution sentinel"
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


def test_hardening_repeated_execution_returns_downstream_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = object()

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

    request = _request()

    service = (
        product.DurableReplayCompositionService()
    )

    first = service.execute(
        request=request
    )

    second = service.execute(
        request=request
    )

    assert first is result
    assert second is result


def test_hardening_request_and_selector_tuples_are_not_mutated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = (
        _selector(1300, "c-z"),
        _selector(1001, "c-a"),
    )

    baseline = (
        _selector(1250, "b-b"),
        _selector(1050, "b-a"),
    )

    request = _request(
        candidate_selectors=candidate,
        baseline_selectors=baseline,
    )

    before = dataclasses.asdict(
        request
    )

    candidate_before = tuple(
        candidate
    )

    baseline_before = tuple(
        baseline
    )

    _run_with_projection(
        monkeypatch,
        request=request,
        candidate_sources=_candidate_sources(),
        baseline_sources=_baseline_sources(),
    )

    after = dataclasses.asdict(
        request
    )

    assert before == after
    assert candidate == candidate_before
    assert baseline == baseline_before
    assert request.candidate_selectors is candidate
    assert request.baseline_selectors is baseline


def test_hardening_exact_dependency_boundary() -> None:
    source = Path(
        "lrp/operations/durable_replay_composition.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(
        source
    )

    imports = []

    for node in ast.walk(tree):
        if isinstance(
            node,
            ast.Import,
        ):
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
        "lrp.evaluation.topk_replay_evaluation",
        (
            "lrp.operations."
            "durable_replay_artifact_discovery"
        ),
        (
            "lrp.operations."
            "durable_replay_execution"
        ),
    }


def test_hardening_exact_single_owned_raise_site() -> None:
    source = Path(
        "lrp/operations/durable_replay_composition.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(
        source
    )

    raises = tuple(
        ast.unparse(node.exc)
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.Raise,
        )
        and node.exc is not None
    )

    assert raises == (
        "ContractError("
        "'request must be DurableReplayCompositionRequest'"
        ")",
    )


def test_hardening_no_exception_normalization() -> None:
    source = Path(
        "lrp/operations/durable_replay_composition.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(
        source
    )

    handlers = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.ExceptHandler,
        )
    )

    assert handlers == ()


def test_hardening_exact_structural_call_counts() -> None:
    source = Path(
        "lrp/operations/durable_replay_composition.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(
        source
    )

    calls = tuple(
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.Call,
        )
    )

    assert calls.count(
        "DurableReplayArtifactDiscoveryRequest"
    ) == 1

    assert calls.count(
        "DurableReplayArtifactDiscoveryService"
    ) == 1

    assert sum(
        call.endswith(".discover")
        for call in calls
    ) == 1

    assert calls.count(
        "DurableReplayExecutionRequest"
    ) == 1

    assert calls.count(
        "DurableReplayExecutionService"
    ) == 1

    assert sum(
        call.endswith(".execute")
        for call in calls
    ) == 1


def test_hardening_minimal_public_surface() -> None:
    source = Path(
        "lrp/operations/durable_replay_composition.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(
        source
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

    public_methods = {}

    for node in tree.body:
        if isinstance(
            node,
            ast.ClassDef,
        ):
            public_methods[
                node.name
            ] = tuple(
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

    assert public_methods == {
        "DurableReplayCompositionRequest": (),
        "DurableReplayCompositionService": (
            "execute",
        ),
    }


def test_hardening_no_filesystem_or_json_side_effects() -> None:
    source = Path(
        "lrp/operations/durable_replay_composition.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    tree = ast.parse(
        source
    )

    calls = {
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.Call,
        )
    }

    imports = set()

    for node in ast.walk(tree):
        if isinstance(
            node,
            ast.Import,
        ):
            imports.update(
                alias.name
                for alias in node.names
            )

        elif isinstance(
            node,
            ast.ImportFrom,
        ):
            if node.module:
                imports.add(
                    node.module
                )

    assert "json" not in imports

    forbidden_calls = {
        "open",
        "Path.open",
        "Path.read_text",
        "Path.read_bytes",
        "Path.write_text",
        "Path.write_bytes",
        "Path.mkdir",
        "Path.glob",
        "Path.rglob",
        "Path.iterdir",
    }

    assert not (
        forbidden_calls
        & calls
    )


def test_hardening_no_lower_layer_ownership_leak() -> None:
    source = Path(
        "lrp/operations/durable_replay_composition.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "load_history",
        "history_until_round",
        "EvaluationWindow",
        "DurableReplayOperationalConsumer",
        "DurableReplayEvaluationOrchestrator",
        "TopKReplayEvaluationService",
        "PredictionResult",
        "argparse",
        "subprocess",
        "tools.validation",
        "lrp.cli",
    )

    assert not any(
        token in source
        for token in forbidden
    )


def test_hardening_discovery_request_projection_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()

    _, _, discovery_requests, _ = _run_with_projection(
        monkeypatch,
        request=request,
        candidate_sources=(),
        baseline_sources=(),
    )

    assert discovery_requests == [
        DurableReplayArtifactDiscoveryRequest(
            artifact_root=request.artifact_root,
            candidate_selectors=request.candidate_selectors,
            baseline_selectors=request.baseline_selectors,
        )
    ]


def test_hardening_execution_request_projection_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()

    candidate_sources = _candidate_sources()
    baseline_sources = _baseline_sources()

    _, _, _, execution_requests = _run_with_projection(
        monkeypatch,
        request=request,
        candidate_sources=candidate_sources,
        baseline_sources=baseline_sources,
    )

    assert execution_requests == [
        DurableReplayExecutionRequest(
            history_path=request.history_path,
            window_name=request.window_name,
            start_round=request.start_round,
            end_round=request.end_round,
            candidate_sources=candidate_sources,
            baseline_sources=baseline_sources,
        )
    ]


def test_hardening_execute_return_annotation_is_frozen() -> None:
    signature = inspect.signature(
        product.DurableReplayCompositionService.execute
    )

    assert (
        str(
            signature.return_annotation
        )
        ==
        "TopKReplayEvaluationResult"
    )
