from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from pathlib import Path

import pytest


MODULE_NAME = (
    "lrp.operations."
    "durable_replay_result_artifact_promotion_publication_lifecycle_source_adapter"
)

CLASS_NAME = (
    "DurableReplayResultArtifactPromotionPublicationLifecycleSourceAdapter"
)

PRODUCT_PATH = Path(
    "lrp/operations/"
    "durable_replay_result_artifact_promotion_publication_lifecycle_source_adapter.py"
)


def _module():
    return importlib.import_module(MODULE_NAME)


def _product_class():
    return getattr(_module(), CLASS_NAME)


class _SourceAdapterStub:
    def __init__(
        self,
        *,
        result: object | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.result = result
        self.error = error
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


def test_product_class_exists() -> None:
    assert hasattr(_module(), CLASS_NAME)


def test_public_surface_is_exact() -> None:
    cls = _product_class()

    public = {
        name
        for name, value in cls.__dict__.items()
        if callable(value) and not name.startswith("_")
    }

    assert public == {"run"}


def test_constructor_has_exact_two_required_dependencies() -> None:
    cls = _product_class()
    params = list(inspect.signature(cls).parameters.values())

    assert [p.name for p in params] == [
        "source_adapter",
        "lifecycle_entrypoint",
    ]

    assert all(p.default is inspect.Parameter.empty for p in params)


def test_run_signature_is_exact() -> None:
    cls = _product_class()

    assert str(inspect.signature(cls.run)) == (
        "(self, artifact_root: 'str | Path', end_round: 'int', *, "
        "source_decision: 'str | Path', registry_root: 'str | Path') "
        "-> 'ProductionLifecycleStageResult'"
    )


def test_run_composes_exact_dependencies_once_and_preserves_identity() -> None:
    request = object()
    result = object()

    source = _SourceAdapterStub(result=request)
    entrypoint = _LifecycleEntrypointStub(result=result)

    adapter = _product_class()(
        source_adapter=source,
        lifecycle_entrypoint=entrypoint,
    )

    artifact_root = Path("artifact-root")
    source_decision = Path("decision.json")
    registry_root = Path("registry")

    actual = adapter.run(
        artifact_root,
        1234,
        source_decision=source_decision,
        registry_root=registry_root,
    )

    assert actual is result

    assert source.calls == [
        {
            "artifact_root": artifact_root,
            "end_round": 1234,
            "source_decision": source_decision,
            "registry_root": registry_root,
        }
    ]

    assert entrypoint.calls == [request]
    assert entrypoint.calls[0] is request


@pytest.mark.parametrize(
    "owner",
    [
        "source_adapter",
        "lifecycle_entrypoint",
    ],
)
def test_dependency_failures_propagate_by_identity(owner: str) -> None:
    failure = RuntimeError(owner)
    request = object()

    if owner == "source_adapter":
        source = _SourceAdapterStub(error=failure)
        entrypoint = _LifecycleEntrypointStub(result=object())
    else:
        source = _SourceAdapterStub(result=request)
        entrypoint = _LifecycleEntrypointStub(error=failure)

    adapter = _product_class()(
        source_adapter=source,
        lifecycle_entrypoint=entrypoint,
    )

    with pytest.raises(RuntimeError) as exc_info:
        adapter.run(
            "artifact-root",
            1234,
            source_decision="decision.json",
            registry_root="registry",
        )

    assert exc_info.value is failure

    if owner == "source_adapter":
        assert entrypoint.calls == []
    else:
        assert entrypoint.calls == [request]


def test_product_has_exact_operational_imports() -> None:
    tree = ast.parse(PRODUCT_PATH.read_text(encoding="utf-8-sig"))

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
            "durable_replay_result_artifact_promotion_publication_request_source_adapter"
        ),
        (
            "lrp.operations."
            "durable_replay_publication_lifecycle_entrypoint"
        ),
    }


def test_product_imports_existing_lifecycle_result_only() -> None:
    tree = ast.parse(PRODUCT_PATH.read_text(encoding="utf-8-sig"))

    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
    }

    assert "lrp.production.production_lifecycle" in imports


def test_product_declares_one_class_only() -> None:
    tree = ast.parse(PRODUCT_PATH.read_text(encoding="utf-8-sig"))

    classes = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    ]

    assert classes == [CLASS_NAME]


def test_product_has_no_exception_translation() -> None:
    tree = ast.parse(PRODUCT_PATH.read_text(encoding="utf-8-sig"))

    assert not any(
        isinstance(node, (ast.Try, ast.Raise))
        for node in ast.walk(tree)
    )


def test_product_has_no_policy_execution_transport_or_io_surface() -> None:
    source = PRODUCT_PATH.read_text(encoding="utf-8-sig").lower()

    forbidden = (
        "argparse",
        "lrp.cli",
        "production_lifecycle_adapters",
        "durablereplaypublicationlifecycleadaptationservice",
        "durablereplaypromotionpublicationexecutionservice",
        "productionchampionregistrypublisher",
        "run_publication_stage",
        ".publish(",
        ".execute(",
        "candidate_advantage_count",
        "baseline_advantage_count",
        "baseline_delta",
        "promotion_policy",
        "rollback",
        "discover",
        "latest",
        "getenv",
        "environ",
        "resolve()",
        "absolute()",
        "expanduser",
        "read_text(",
        "read_bytes(",
        "write_text(",
        "write_bytes(",
        "json.load",
        "json.dump",
        "mkdir(",
        "unlink(",
        "os.replace",
    )

    for token in forbidden:
        assert token not in source


def test_call_graph_is_exact() -> None:
    tree = ast.parse(PRODUCT_PATH.read_text(encoding="utf-8-sig"))

    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == CLASS_NAME
    )

    run = next(
        node
        for node in class_node.body
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


def test_run_does_not_rewrite_inputs() -> None:
    tree = ast.parse(PRODUCT_PATH.read_text(encoding="utf-8-sig"))

    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == CLASS_NAME
    )

    run = next(
        node
        for node in class_node.body
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