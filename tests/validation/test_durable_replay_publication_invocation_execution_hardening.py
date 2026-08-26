from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import lrp.operations.durable_replay_publication_invocation_execution as product


class _Carrier:
    def __init__(
        self,
        *,
        result: object | None = None,
        failure: BaseException | None = None,
    ) -> None:
        self.result = result
        self.failure = failure
        self.calls: list[object] = []

    def read(self, path: object) -> object:
        self.calls.append(path)

        if self.failure is not None:
            raise self.failure

        return self.result


class _Codec:
    def __init__(
        self,
        *,
        result: object | None = None,
        failure: BaseException | None = None,
    ) -> None:
        self.result = result
        self.failure = failure
        self.calls: list[object] = []

    def decode(self, transport: object) -> object:
        self.calls.append(transport)

        if self.failure is not None:
            raise self.failure

        return self.result


class _Lifecycle:
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


def _service(
    *,
    carrier: _Carrier,
    codec: _Codec,
    lifecycle: _Lifecycle,
):
    return product.DurableReplayPublicationInvocationExecutionService(
        file_carrier=carrier,
        transport_codec=codec,
        lifecycle_entrypoint=lifecycle,
    )


@pytest.mark.parametrize(
    "path",
    [
        "",
        " ",
        "invocation.json",
        r".\invocation.json",
        r"..\payload\invocation.json",
        r"%USERPROFILE%\invocation.json",
        Path("payload/../payload/invocation.json"),
        Path("."),
    ],
)
def test_explicit_path_is_forwarded_without_rewrite(path: object) -> None:
    transport = object()
    request = object()
    result = object()

    carrier = _Carrier(result=transport)
    codec = _Codec(result=request)
    lifecycle = _Lifecycle(result=result)

    service = _service(
        carrier=carrier,
        codec=codec,
        lifecycle=lifecycle,
    )

    actual = service.execute(path)

    assert actual is result

    assert carrier.calls == [path]
    assert carrier.calls[0] is path

    assert codec.calls == [transport]
    assert codec.calls[0] is transport

    assert lifecycle.calls == [request]
    assert lifecycle.calls[0] is request


def test_transport_identity_is_preserved_into_decode() -> None:
    transport = object()

    carrier = _Carrier(result=transport)
    codec = _Codec(result=object())
    lifecycle = _Lifecycle(result=object())

    service = _service(
        carrier=carrier,
        codec=codec,
        lifecycle=lifecycle,
    )

    service.execute("invocation.json")

    assert codec.calls == [transport]
    assert codec.calls[0] is transport


def test_request_identity_is_preserved_into_lifecycle() -> None:
    transport = object()
    request = object()

    carrier = _Carrier(result=transport)
    codec = _Codec(result=request)
    lifecycle = _Lifecycle(result=object())

    service = _service(
        carrier=carrier,
        codec=codec,
        lifecycle=lifecycle,
    )

    service.execute("invocation.json")

    assert lifecycle.calls == [request]
    assert lifecycle.calls[0] is request


def test_result_identity_is_preserved() -> None:
    result = object()

    carrier = _Carrier(result=object())
    codec = _Codec(result=object())
    lifecycle = _Lifecycle(result=result)

    service = _service(
        carrier=carrier,
        codec=codec,
        lifecycle=lifecycle,
    )

    actual = service.execute("invocation.json")

    assert actual is result


def test_carrier_failure_short_circuits_decode_and_lifecycle() -> None:
    failure = RuntimeError("carrier")

    carrier = _Carrier(failure=failure)
    codec = _Codec(result=object())
    lifecycle = _Lifecycle(result=object())

    service = _service(
        carrier=carrier,
        codec=codec,
        lifecycle=lifecycle,
    )

    with pytest.raises(RuntimeError) as exc_info:
        service.execute("invocation.json")

    assert exc_info.value is failure
    assert carrier.calls == ["invocation.json"]
    assert codec.calls == []
    assert lifecycle.calls == []


def test_decode_failure_short_circuits_lifecycle() -> None:
    transport = object()
    failure = RuntimeError("decode")

    carrier = _Carrier(result=transport)
    codec = _Codec(failure=failure)
    lifecycle = _Lifecycle(result=object())

    service = _service(
        carrier=carrier,
        codec=codec,
        lifecycle=lifecycle,
    )

    with pytest.raises(RuntimeError) as exc_info:
        service.execute("invocation.json")

    assert exc_info.value is failure
    assert carrier.calls == ["invocation.json"]
    assert codec.calls == [transport]
    assert lifecycle.calls == []


def test_lifecycle_failure_propagates_by_identity() -> None:
    transport = object()
    request = object()
    failure = RuntimeError("lifecycle")

    carrier = _Carrier(result=transport)
    codec = _Codec(result=request)
    lifecycle = _Lifecycle(failure=failure)

    service = _service(
        carrier=carrier,
        codec=codec,
        lifecycle=lifecycle,
    )

    with pytest.raises(RuntimeError) as exc_info:
        service.execute("invocation.json")

    assert exc_info.value is failure
    assert carrier.calls == ["invocation.json"]
    assert codec.calls == [transport]
    assert lifecycle.calls == [request]


def test_repeated_execution_is_composition_deterministic() -> None:
    transport = object()
    request = object()
    result = object()

    carrier = _Carrier(result=transport)
    codec = _Codec(result=request)
    lifecycle = _Lifecycle(result=result)

    service = _service(
        carrier=carrier,
        codec=codec,
        lifecycle=lifecycle,
    )

    first = service.execute("invocation.json")
    second = service.execute("invocation.json")

    assert first is result
    assert second is result

    assert carrier.calls == [
        "invocation.json",
        "invocation.json",
    ]

    assert codec.calls == [
        transport,
        transport,
    ]

    assert lifecycle.calls == [
        request,
        request,
    ]


def test_constructor_requires_exact_three_dependencies() -> None:
    cls = product.DurableReplayPublicationInvocationExecutionService

    params = list(
        inspect.signature(cls).parameters.values()
    )

    assert [param.name for param in params] == [
        "file_carrier",
        "transport_codec",
        "lifecycle_entrypoint",
    ]

    assert all(
        param.default is inspect.Parameter.empty
        for param in params
    )


def test_execute_signature_remains_minimal() -> None:
    cls = product.DurableReplayPublicationInvocationExecutionService

    assert str(inspect.signature(cls.execute)) == (
        "(self, path: 'str | Path') "
        "-> 'ProductionLifecycleStageResult'"
    )


def test_product_call_graph_remains_exact_and_ordered() -> None:
    tree = ast.parse(inspect.getsource(product))

    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name
        == "DurableReplayPublicationInvocationExecutionService"
    )

    execute = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "execute"
    )

    calls = [
        ast.unparse(node.func)
        for node in ast.walk(execute)
        if isinstance(node, ast.Call)
    ]

    assert calls == [
        "self._file_carrier.read",
        "self._transport_codec.decode",
        "self._lifecycle_entrypoint.run",
    ]


def test_execute_has_no_path_rewrite_assignments() -> None:
    tree = ast.parse(inspect.getsource(product))

    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name
        == "DurableReplayPublicationInvocationExecutionService"
    )

    execute = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "execute"
    )

    assigned_names = {
        target.id
        for node in ast.walk(execute)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
        )
        if isinstance(target, ast.Name)
    }

    assert "path" not in assigned_names


def test_product_has_no_dependency_default_construction() -> None:
    source = inspect.getsource(product)

    forbidden = (
        "DurableReplayPublicationInvocationJsonFileCarrier()",
        "DurableReplayPublicationInvocationTransportCodec()",
        "DurableReplayPublicationLifecycleEntrypoint(",
    )

    for token in forbidden:
        assert token not in source


def test_product_has_no_exception_translation_or_cleanup() -> None:
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


def test_product_operational_import_boundary_is_exact() -> None:
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
            "durable_replay_publication_invocation_json_file_carrier"
        ),
        (
            "lrp.operations."
            "durable_replay_publication_invocation_transport"
        ),
        (
            "lrp.operations."
            "durable_replay_publication_lifecycle_entrypoint"
        ),
    }


def test_product_has_no_policy_recomputation_surface() -> None:
    source = inspect.getsource(product).lower()

    forbidden = (
        "promotioneligibility",
        "promotionactionplan",
        "candidate_advantage_count",
        "baseline_advantage_count",
        "baseline_delta",
        "recommendation",
        "promote_candidate",
    )

    for token in forbidden:
        assert token not in source


def test_product_has_no_direct_publication_owner_surface() -> None:
    source = inspect.getsource(product).lower()

    forbidden = (
        "productionchampionregistrypublisher",
        "run_publication_stage",
        ".publish(",
        "publish_champion",
        "rollback",
        "registry mutation",
    )

    for token in forbidden:
        assert token not in source


def test_product_has_no_discovery_or_path_normalization_surface() -> None:
    source = inspect.getsource(product).lower()

    forbidden = (
        "discover",
        "latest",
        "glob(",
        "rglob(",
        "resolve(",
        "absolute(",
        "expanduser",
        "getenv",
        "environ",
        "cwd",
        "home(",
    )

    for token in forbidden:
        assert token not in source


def test_product_has_no_cli_stdio_or_inspection_surface() -> None:
    source = inspect.getsource(product).lower()

    forbidden = (
        "argparse",
        "lrp.cli",
        "sys.stdin",
        "sys.stdout",
        "input(",
        "print(",
        "inspect-replay-invocation",
    )

    for token in forbidden:
        assert token not in source