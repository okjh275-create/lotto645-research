from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

MODULE_NAME = "lrp.operations.durable_replay_result_artifact_source_adapter"
CLASS_NAME = "DurableReplayResultArtifactSourceAdapter"


def _module():
    return importlib.import_module(MODULE_NAME)


def _class():
    return getattr(_module(), CLASS_NAME)


def _source() -> str:
    return Path(
        "lrp/operations/durable_replay_result_artifact_source_adapter.py"
    ).read_text(encoding="utf-8-sig")


@pytest.mark.parametrize(
    "artifact_root",
    [
        r"relative\artifact-root",
        r".\relative\artifact-root",
        r"..\relative\artifact-root",
        r"folder with spaces\artifact-root",
        r"%USERPROFILE%\artifact-root",
        r"~\artifact-root",
        Path(r"relative\path-object"),
    ],
)
def test_artifact_root_identity_is_forwarded_exactly(
    monkeypatch,
    artifact_root: str | Path,
) -> None:
    module = _module()
    owner = _class()

    seen: list[object] = []
    sentinel = object()

    class FakeRequest:
        def __init__(self, artifact_root: object, end_round: object) -> None:
            seen.append(artifact_root)
            self.artifact_root = artifact_root
            self.end_round = end_round

    class FakeInspectionService:
        def inspect(self, *, request: object):
            return sentinel

    monkeypatch.setattr(
        module,
        "DurableReplayResultArtifactConsumerRequest",
        FakeRequest,
    )

    result = owner(
        inspection_service=FakeInspectionService()
    ).adapt(
        artifact_root=artifact_root,
        end_round=123,
    )

    assert seen == [artifact_root]
    assert seen[0] is artifact_root or seen[0] == artifact_root
    assert result is sentinel


@pytest.mark.parametrize(
    "end_round",
    [-1, 0, 1, 2, 999, 1234, 999999],
)
def test_end_round_identity_is_forwarded_without_adapter_validation(
    monkeypatch,
    end_round: int,
) -> None:
    module = _module()
    owner = _class()

    seen: list[object] = []
    sentinel = object()

    class FakeRequest:
        def __init__(self, artifact_root: object, end_round: object) -> None:
            seen.append(end_round)
            self.artifact_root = artifact_root
            self.end_round = end_round

    class FakeInspectionService:
        def inspect(self, *, request: object):
            return sentinel

    monkeypatch.setattr(
        module,
        "DurableReplayResultArtifactConsumerRequest",
        FakeRequest,
    )

    result = owner(
        inspection_service=FakeInspectionService()
    ).adapt(
        artifact_root="artifact-root",
        end_round=end_round,
    )

    assert seen == [end_round]
    assert result is sentinel


def test_request_constructor_failure_propagates_unchanged(
    monkeypatch,
) -> None:
    module = _module()
    owner = _class()

    failure = TypeError("request-construction-failure")

    class FailingRequest:
        def __init__(self, artifact_root: object, end_round: object) -> None:
            raise failure

    class FakeInspectionService:
        def inspect(self, *, request: object):
            raise AssertionError("inspection must not run")

    monkeypatch.setattr(
        module,
        "DurableReplayResultArtifactConsumerRequest",
        FailingRequest,
    )

    with pytest.raises(TypeError) as exc_info:
        owner(
            inspection_service=FakeInspectionService()
        ).adapt(
            artifact_root="artifact-root",
            end_round=1,
        )

    assert exc_info.value is failure


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("missing"),
        IsADirectoryError("directory"),
        TypeError("type"),
        ValueError("value"),
        RuntimeError("runtime"),
    ],
)
def test_inspection_failures_propagate_by_identity(
    monkeypatch,
    failure: Exception,
) -> None:
    module = _module()
    owner = _class()

    class FakeRequest:
        def __init__(self, artifact_root: object, end_round: object) -> None:
            self.artifact_root = artifact_root
            self.end_round = end_round

    class FakeInspectionService:
        def inspect(self, *, request: object):
            raise failure

    monkeypatch.setattr(
        module,
        "DurableReplayResultArtifactConsumerRequest",
        FakeRequest,
    )

    with pytest.raises(type(failure)) as exc_info:
        owner(
            inspection_service=FakeInspectionService()
        ).adapt(
            artifact_root="artifact-root",
            end_round=1,
        )

    assert exc_info.value is failure


def test_adapter_does_not_catch_or_translate_failures() -> None:
    tree = ast.parse(_source())

    assert not any(isinstance(node, ast.Try) for node in ast.walk(tree))
    assert not any(isinstance(node, ast.Raise) for node in ast.walk(tree))


def test_adapter_has_exact_two_operational_imports() -> None:
    tree = ast.parse(_source())

    operational: set[str] = set()

    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("lrp."):
                operational.add(node.module)

    assert operational == {
        "lrp.operations.durable_replay_result_artifact_consumer",
        "lrp.operations.durable_replay_result_artifact_inspection",
    }


def test_adapter_has_no_artifact_file_io_surface() -> None:
    source = _source()

    forbidden = [
        "read_text(",
        "read_bytes(",
        "write_text(",
        "write_bytes(",
        "open(",
        "json.loads",
        "json.dumps",
        "verify_manifest",
        "manifest.json",
        "evaluation_result.json",
    ]

    for token in forbidden:
        assert token not in source


def test_adapter_has_no_path_discovery_normalization_or_expansion() -> None:
    source = _source()

    forbidden = [
        "resolve()",
        "absolute()",
        "expanduser",
        "getenv(",
        "environ",
        "glob(",
        "rglob(",
        "cwd(",
        "home(",
        "latest",
        "discover",
    ]

    for token in forbidden:
        assert token not in source


def test_adapter_has_no_downstream_comparison_or_publication_logic() -> None:
    source = _source()

    forbidden = [
        "DurableReplayResultComparisonSummary",
        "DurableReplayResultComparisonAssessment",
        "DurableReplayResultPromotionEligibility",
        "DurableReplayResultPromotionActionPlan",
        "DurableReplayPromotionPublicationRequest",
        "DurableReplayPublicationLifecycleEntrypoint",
        "DurableReplayPromotionPublicationExecutionService",
        "ProductionChampionRegistryPublisher",
        "run_publication_stage",
        ".publish(",
    ]

    for token in forbidden:
        assert token not in source


def test_adapter_has_no_cli_or_stream_surface() -> None:
    source = _source()

    forbidden = [
        "argparse",
        "sys.stdin",
        "sys.stdout",
        "stdin",
        "stdout",
        "lrp.cli",
    ]

    for token in forbidden:
        assert token not in source


def test_adapter_has_no_registry_or_rollback_surface() -> None:
    source = _source()

    forbidden = [
        "registry_root",
        "ProductionChampionRegistry",
        "rollback",
        "champion_registry",
    ]

    for token in forbidden:
        assert token not in source


def test_adapter_declares_only_one_product_class() -> None:
    tree = ast.parse(_source())

    classes = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    ]

    assert classes == [CLASS_NAME]


def test_adapter_call_graph_is_minimal() -> None:
    tree = ast.parse(_source())

    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            try:
                calls.append(ast.unparse(node.func))
            except Exception:
                pass

    assert calls.count("DurableReplayResultArtifactConsumerRequest") == 1
    assert calls.count("self._inspection_service.inspect") == 1
    assert calls.count("DurableReplayResultArtifactInspectionService") == 1

    allowed = {
        "DurableReplayResultArtifactConsumerRequest",
        "self._inspection_service.inspect",
        "DurableReplayResultArtifactInspectionService",
    }

    assert set(calls) <= allowed


def test_adapt_has_no_local_identity_rewrite_assignments() -> None:
    tree = ast.parse(_source())

    adapt = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == CLASS_NAME:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "adapt":
                    adapt = item
                    break

    assert adapt is not None

    assigned_names = set()
    for node in ast.walk(adapt):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned_names.add(target.id)

    assert assigned_names == {"request"}


def test_constructor_dependency_is_optional_and_owned() -> None:
    owner = _class()
    signature = inspect.signature(owner)

    param = signature.parameters["inspection_service"]
    assert param.default is None

    source = _source()
    assert "if inspection_service is not None" in source
    assert "DurableReplayResultArtifactInspectionService()" in source
