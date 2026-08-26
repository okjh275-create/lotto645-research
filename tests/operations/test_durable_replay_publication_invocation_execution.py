from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from pathlib import Path

import pytest


MODULE_NAME = (
    "lrp.operations."
    "durable_replay_publication_invocation_execution"
)

CLASS_NAME = (
    "DurableReplayPublicationInvocationExecutionService"
)

PRODUCT_PATH = Path(
    "lrp/operations/"
    "durable_replay_publication_invocation_execution.py"
)


def _module():
    return importlib.import_module(MODULE_NAME)


def _service_class():
    return getattr(_module(), CLASS_NAME)


class _CarrierStub:
    def __init__(
        self,
        *,
        result: object | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[object] = []

    def read(self, path: object) -> object:
        self.calls.append(path)

        if self.error is not None:
            raise self.error

        return self.result


class _CodecStub:
    def __init__(
        self,
        *,
        result: object | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[object] = []

    def decode(self, transport: object) -> object:
        self.calls.append(transport)

        if self.error is not None:
            raise self.error

        return self.result


class _LifecycleEntrypointStub:
    def __init__(
        self,
        *,
        result: object | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[object] = []

    def run(self, request: object) -> object:
        self.calls.append(request)

        if self.error is not None:
            raise self.error

        return self.result


def test_product_module_exists() -> None:
    assert importlib.util.find_spec(MODULE_NAME) is not None


def test_product_service_exists() -> None:
    assert hasattr(_module(), CLASS_NAME)


def test_public_surface_is_exact() -> None:
    cls = _service_class()

    public = {
        name
        for name, value in cls.__dict__.items()
        if callable(value) and not name.startswith("_")
    }

    assert public == {"execute"}


def test_constructor_has_exact_three_required_dependencies() -> None:
    cls = _service_class()
    params = list(inspect.signature(cls).parameters.values())

    assert [p.name for p in params] == [
        "file_carrier",
        "transport_codec",
        "lifecycle_entrypoint",
    ]

    assert all(
        p.default is inspect.Parameter.empty
        for p in params
    )


def test_execute_signature_is_exact() -> None:
    cls = _service_class()

    assert str(inspect.signature(cls.execute)) == (
        "(self, path: 'str | Path') "
        "-> 'ProductionLifecycleStageResult'"
    )


def test_execute_composes_exact_chain_once_and_preserves_identity() -> None:
    transport = object()
    request = object()
    result = object()

    carrier = _CarrierStub(result=transport)
    codec = _CodecStub(result=request)
    lifecycle = _LifecycleEntrypointStub(result=result)

    service = _service_class()(
        file_carrier=carrier,
        transport_codec=codec,
        lifecycle_entrypoint=lifecycle,
    )

    path = Path("explicit/invocation.json")

    actual = service.execute(path)

    assert actual is result

    assert carrier.calls == [path]
    assert carrier.calls[0] is path

    assert codec.calls == [transport]
    assert codec.calls[0] is transport

    assert lifecycle.calls == [request]
    assert lifecycle.calls[0] is request


@pytest.mark.parametrize(
    "owner",
    [
        "carrier",
        "codec",
        "lifecycle",
    ],
)
def test_dependency_failures_propagate_by_identity(owner: str) -> None:
    failure = RuntimeError(owner)
    transport = object()
    request = object()

    if owner == "carrier":
        carrier = _CarrierStub(error=failure)
        codec = _CodecStub(result=request)
        lifecycle = _LifecycleEntrypointStub(result=object())

    elif owner == "codec":
        carrier = _CarrierStub(result=transport)
        codec = _CodecStub(error=failure)
        lifecycle = _LifecycleEntrypointStub(result=object())

    else:
        carrier = _CarrierStub(result=transport)
        codec = _CodecStub(result=request)
        lifecycle = _LifecycleEntrypointStub(error=failure)

    service = _service_class()(
        file_carrier=carrier,
        transport_codec=codec,
        lifecycle_entrypoint=lifecycle,
    )

    with pytest.raises(RuntimeError) as exc_info:
        service.execute("explicit/invocation.json")

    assert exc_info.value is failure

    if owner == "carrier":
        assert codec.calls == []
        assert lifecycle.calls == []

    elif owner == "codec":
        assert carrier.calls == ["explicit/invocation.json"]
        assert codec.calls == [transport]
        assert lifecycle.calls == []

    else:
        assert carrier.calls == ["explicit/invocation.json"]
        assert codec.calls == [transport]
        assert lifecycle.calls == [request]


def test_product_has_exact_operational_imports() -> None:
    tree = ast.parse(
        PRODUCT_PATH.read_text(encoding="utf-8-sig")
    )

    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("lrp.operations.")
    }

    assert imports == {
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


def test_product_imports_existing_lifecycle_result_only() -> None:
    tree = ast.parse(
        PRODUCT_PATH.read_text(encoding="utf-8-sig")
    )

    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
    }

    assert "lrp.production.production_lifecycle" in imports


def test_product_declares_one_class_only() -> None:
    tree = ast.parse(
        PRODUCT_PATH.read_text(encoding="utf-8-sig")
    )

    classes = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    ]

    assert classes == [CLASS_NAME]


def test_product_has_no_exception_translation() -> None:
    tree = ast.parse(
        PRODUCT_PATH.read_text(encoding="utf-8-sig")
    )

    assert not any(
        isinstance(
            node,
            (
                ast.Try,
                ast.Raise,
            ),
        )
        for node in ast.walk(tree)
    )


def test_product_has_no_path_rewrite_or_discovery() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    ).lower()

    forbidden = (
        "resolve(",
        "absolute(",
        "expanduser",
        "getenv",
        "environ",
        "glob(",
        "rglob(",
        "discover",
        "latest",
        "default",
        "cwd",
        "home(",
    )

    for token in forbidden:
        assert token not in source


def test_product_has_no_policy_or_direct_publication_surface() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    ).lower()

    forbidden = (
        "productionchampionregistrypublisher",
        "run_publication_stage",
        ".publish(",
        ".execute(",
        "promotioneligibility",
        "promotionactionplan",
        "candidate_advantage_count",
        "baseline_advantage_count",
        "baseline_delta",
        "rollback",
        "publish_champion",
        "production_lifecycle_adapters",
    )

    for token in forbidden:
        assert token not in source


def test_product_has_no_cli_or_stdio_surface() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    ).lower()

    forbidden = (
        "argparse",
        "sys.stdin",
        "sys.stdout",
        "lrp.cli",
        "input(",
        "print(",
    )

    for token in forbidden:
        assert token not in source


def test_call_graph_is_exact_and_ordered() -> None:
    tree = ast.parse(
        PRODUCT_PATH.read_text(encoding="utf-8-sig")
    )

    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == CLASS_NAME
    )

    execute = next(
        node
        for node in class_node.body
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


def test_execute_does_not_rewrite_input_path() -> None:
    tree = ast.parse(
        PRODUCT_PATH.read_text(encoding="utf-8-sig")
    )

    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == CLASS_NAME
    )

    execute = next(
        node
        for node in class_node.body
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


def test_service_does_not_default_construct_authoritative_dependencies() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    forbidden = (
        "DurableReplayPublicationInvocationJsonFileCarrier()",
        "DurableReplayPublicationInvocationTransportCodec()",
        "DurableReplayPublicationLifecycleEntrypoint(",
    )

    for token in forbidden:
        assert token not in source


def test_existing_inspection_cli_is_not_imported() -> None:
    source = PRODUCT_PATH.read_text(
        encoding="utf-8-sig"
    )

    assert (
        "durable_replay_publication_invocation_json_file"
        not in source.replace(
            "durable_replay_publication_invocation_json_file_carrier",
            "",
        )
    )