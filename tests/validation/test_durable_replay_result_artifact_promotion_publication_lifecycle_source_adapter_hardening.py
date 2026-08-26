from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import lrp.operations.durable_replay_result_artifact_promotion_publication_lifecycle_source_adapter as product


class _Source:
    def __init__(
        self,
        *,
        result: object | None = None,
        failure: BaseException | None = None,
    ) -> None:
        self.result = result
        self.failure = failure
        self.calls: list[dict[str, object]] = []

    def adapt(
        self,
        artifact_root: object,
        end_round: object,
        *,
        source_decision: object,
        registry_root: object,
    ) -> object:
        self.calls.append(
            {
                "artifact_root": artifact_root,
                "end_round": end_round,
                "source_decision": source_decision,
                "registry_root": registry_root,
            }
        )

        if self.failure is not None:
            raise self.failure

        return self.result


class _Entrypoint:
    def __init__(
        self,
        *,
        result: object | None = None,
        failure: BaseException | None = None,
    ) -> None:
        self.result = result
        self.failure = failure
        self.calls: list[object] = []

    def run(self, request: object) -> object:
        self.calls.append(request)

        if self.failure is not None:
            raise self.failure

        return self.result


@pytest.mark.parametrize(
    ("artifact_root", "end_round"),
    [
        ("", -1),
        ("  artifact root  ", 0),
        (r".\relative\artifact-root", 1),
        (r"..\relative\artifact-root", 1234),
        (r"%USERPROFILE%\artifact-root", 999999),
        (Path("artifact/../artifact-root"), 7),
    ],
)
def test_artifact_identity_inputs_are_forwarded_unchanged(
    artifact_root: object,
    end_round: int,
) -> None:
    request = object()
    source = _Source(result=request)
    entrypoint = _Entrypoint(result=object())

    adapter = (
        product
        .DurableReplayResultArtifactPromotionPublicationLifecycleSourceAdapter(
            source_adapter=source,
            lifecycle_entrypoint=entrypoint,
        )
    )

    adapter.run(
        artifact_root,
        end_round,
        source_decision="decision",
        registry_root="registry",
    )

    assert len(source.calls) == 1
    call = source.calls[0]

    assert call["artifact_root"] is artifact_root
    assert call["end_round"] == end_round


@pytest.mark.parametrize(
    ("source_decision", "registry_root"),
    [
        (" decision ", " registry "),
        (
            Path("decision/../decision.json"),
            Path("registry/../registry"),
        ),
        ("", ""),
    ],
)
def test_publication_identity_inputs_are_forwarded_by_identity(
    source_decision: object,
    registry_root: object,
) -> None:
    request = object()
    source = _Source(result=request)
    entrypoint = _Entrypoint(result=object())

    adapter = (
        product
        .DurableReplayResultArtifactPromotionPublicationLifecycleSourceAdapter(
            source_adapter=source,
            lifecycle_entrypoint=entrypoint,
        )
    )

    adapter.run(
        "artifact",
        1234,
        source_decision=source_decision,
        registry_root=registry_root,
    )

    call = source.calls[0]

    assert call["source_decision"] is source_decision
    assert call["registry_root"] is registry_root


def test_source_request_identity_is_forwarded_exactly_to_entrypoint() -> None:
    request = object()
    result = object()

    source = _Source(result=request)
    entrypoint = _Entrypoint(result=result)

    adapter = (
        product
        .DurableReplayResultArtifactPromotionPublicationLifecycleSourceAdapter(
            source_adapter=source,
            lifecycle_entrypoint=entrypoint,
        )
    )

    actual = adapter.run(
        "artifact",
        1234,
        source_decision="decision",
        registry_root="registry",
    )

    assert entrypoint.calls == [request]
    assert entrypoint.calls[0] is request
    assert actual is result


def test_source_failure_prevents_lifecycle_entrypoint_call() -> None:
    failure = RuntimeError("source-owned")

    source = _Source(failure=failure)
    entrypoint = _Entrypoint(result=object())

    adapter = (
        product
        .DurableReplayResultArtifactPromotionPublicationLifecycleSourceAdapter(
            source_adapter=source,
            lifecycle_entrypoint=entrypoint,
        )
    )

    with pytest.raises(RuntimeError) as exc_info:
        adapter.run(
            "artifact",
            1234,
            source_decision="decision",
            registry_root="registry",
        )

    assert exc_info.value is failure
    assert len(source.calls) == 1
    assert entrypoint.calls == []


def test_lifecycle_failure_propagates_by_identity() -> None:
    request = object()
    failure = RuntimeError("lifecycle-owned")

    source = _Source(result=request)
    entrypoint = _Entrypoint(failure=failure)

    adapter = (
        product
        .DurableReplayResultArtifactPromotionPublicationLifecycleSourceAdapter(
            source_adapter=source,
            lifecycle_entrypoint=entrypoint,
        )
    )

    with pytest.raises(RuntimeError) as exc_info:
        adapter.run(
            "artifact",
            1234,
            source_decision="decision",
            registry_root="registry",
        )

    assert exc_info.value is failure
    assert entrypoint.calls == [request]


def test_repeated_composition_is_semantically_deterministic() -> None:
    request = object()
    result = object()

    source = _Source(result=request)
    entrypoint = _Entrypoint(result=result)

    adapter = (
        product
        .DurableReplayResultArtifactPromotionPublicationLifecycleSourceAdapter(
            source_adapter=source,
            lifecycle_entrypoint=entrypoint,
        )
    )

    first = adapter.run(
        "artifact",
        1234,
        source_decision="decision",
        registry_root="registry",
    )

    second = adapter.run(
        "artifact",
        1234,
        source_decision="decision",
        registry_root="registry",
    )

    assert first is result
    assert second is result

    assert source.calls == [
        {
            "artifact_root": "artifact",
            "end_round": 1234,
            "source_decision": "decision",
            "registry_root": "registry",
        },
        {
            "artifact_root": "artifact",
            "end_round": 1234,
            "source_decision": "decision",
            "registry_root": "registry",
        },
    ]

    assert entrypoint.calls == [
        request,
        request,
    ]


def test_constructor_requires_exact_existing_dependencies() -> None:
    cls = (
        product
        .DurableReplayResultArtifactPromotionPublicationLifecycleSourceAdapter
    )

    params = list(inspect.signature(cls).parameters.values())

    assert [p.name for p in params] == [
        "source_adapter",
        "lifecycle_entrypoint",
    ]

    assert all(
        p.default is inspect.Parameter.empty
        for p in params
    )


def test_product_has_no_dependency_default_construction() -> None:
    source = inspect.getsource(product)

    forbidden = (
        "DurableReplayResultArtifactPromotionPublicationRequestSourceAdapter()",
        "DurableReplayPublicationLifecycleEntrypoint(",
        "DurableReplayPublicationLifecycleAdaptationService(",
        "DurableReplayPromotionPublicationExecutionService(",
        "ProductionChampionRegistryPublisher(",
    )

    for token in forbidden:
        assert token not in source


def test_product_has_no_exception_translation_or_cleanup_surface() -> None:
    tree = ast.parse(inspect.getsource(product))

    assert not any(
        isinstance(
            node,
            (
                ast.Try,
                ast.Raise,
                ast.With,
                ast.AsyncWith,
            ),
        )
        for node in ast.walk(tree)
    )


def test_product_has_no_input_rewrite_assignments() -> None:
    tree = ast.parse(inspect.getsource(product))

    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name
        == (
            "DurableReplayResultArtifactPromotion"
            "PublicationLifecycleSourceAdapter"
        )
    )

    run = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run"
    )

    assigned_names = {
        target.id
        for node in ast.walk(run)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
        )
        if isinstance(target, ast.Name)
    }

    assert "artifact_root" not in assigned_names
    assert "end_round" not in assigned_names
    assert "source_decision" not in assigned_names
    assert "registry_root" not in assigned_names


def test_product_call_graph_is_exact_and_ordered() -> None:
    tree = ast.parse(inspect.getsource(product))

    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name
        == (
            "DurableReplayResultArtifactPromotion"
            "PublicationLifecycleSourceAdapter"
        )
    )

    run = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run"
    )

    calls = [
        ast.unparse(node.func)
        for node in ast.walk(run)
        if isinstance(node, ast.Call)
    ]

    assert calls == [
        "self._source_adapter.adapt",
        "self._lifecycle_entrypoint.run",
    ]


def test_product_has_exact_operational_dependency_boundary() -> None:
    tree = ast.parse(inspect.getsource(product))

    operational_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("lrp.operations.")
    }

    assert operational_imports == {
        (
            "lrp.operations."
            "durable_replay_publication_lifecycle_entrypoint"
        ),
        (
            "lrp.operations."
            "durable_replay_result_artifact_promotion_"
            "publication_request_source_adapter"
        ),
    }


def test_product_has_no_forbidden_execution_or_transport_surface() -> None:
    source = inspect.getsource(product).lower()

    forbidden = (
        "productionchampionregistrypublisher",
        ".publish(",
        ".execute(",
        "durablereplaypromotionpublicationexecutionservice",
        "durablereplaypublicationlifecycleadaptationservice",
        "durablereplaypublicationinvocationtransport",
        "json.",
        "argparse",
        "lrp.cli",
        "discover",
        "latest",
        "getenv",
        "environ",
        "resolve(",
        "absolute(",
        "expanduser",
        "open(",
        "read_text(",
        "read_bytes(",
        "write_text(",
        "write_bytes(",
        "mkdir(",
        "unlink(",
        "rollback",
    )

    for token in forbidden:
        assert token not in source