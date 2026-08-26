from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Any

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


def _inspection():
    from lrp.operations.durable_replay_result_artifact_inspection import (
        DurableReplayResultArtifactInspection,
    )

    return DurableReplayResultArtifactInspection(
        status="PASS",
        round_count=12,
        candidate_model_name="candidate",
        baseline_model_name="baseline",
        evaluation={
            "top3": {},
            "top5": {},
            "top10": {},
            "window": {},
        },
    )


def test_source_adapter_product_module_exists() -> None:
    assert importlib.util.find_spec(MODULE_NAME) is not None


def test_source_adapter_class_exists() -> None:
    module = _module()
    assert hasattr(module, CLASS_NAME)


def test_public_methods_are_exact() -> None:
    owner = _class()
    public = {
        name
        for name, value in inspect.getmembers(owner, inspect.isfunction)
        if not name.startswith("_")
    }
    assert public == {"adapt"}


def test_init_signature_is_minimal() -> None:
    owner = _class()
    signature = inspect.signature(owner)
    params = list(signature.parameters.values())

    assert len(params) == 1
    assert params[0].name == "inspection_service"
    assert params[0].default is None


def test_adapt_signature_is_exact() -> None:
    owner = _class()
    signature = inspect.signature(owner.adapt)
    params = list(signature.parameters.values())

    assert [p.name for p in params] == [
        "self",
        "artifact_root",
        "end_round",
    ]
    assert str(params[1].annotation) in {
        "str | Path",
        "'str | Path'",
    }
    assert str(params[2].annotation) in {
        "int",
        "'int'",
    }
    assert (
        "DurableReplayResultArtifactInspection"
        in str(signature.return_annotation)
    )


def test_adapter_owns_or_receives_inspection_service_dependency() -> None:
    source = _source()
    assert "DurableReplayResultArtifactInspectionService" in source


def test_adapt_constructs_consumer_request_exactly_once(monkeypatch) -> None:
    module = _module()
    owner = _class()

    seen: list[tuple[Any, Any]] = []

    class FakeRequest:
        def __init__(self, artifact_root: object, end_round: object) -> None:
            seen.append((artifact_root, end_round))
            self.artifact_root = artifact_root
            self.end_round = end_round

    class FakeInspectionService:
        def inspect(self, *, request: object):
            return _inspection()

    monkeypatch.setattr(
        module,
        "DurableReplayResultArtifactConsumerRequest",
        FakeRequest,
    )

    adapter = owner(inspection_service=FakeInspectionService())
    result = adapter.adapt(
        artifact_root=r"relative\artifact-root",
        end_round=1234,
    )

    assert result == _inspection()
    assert seen == [(r"relative\artifact-root", 1234)]


def test_adapt_delegates_request_to_inspection_exactly_once(monkeypatch) -> None:
    module = _module()
    owner = _class()

    requests: list[object] = []

    class FakeRequest:
        def __init__(self, artifact_root: object, end_round: object) -> None:
            self.artifact_root = artifact_root
            self.end_round = end_round

    class FakeInspectionService:
        def inspect(self, *, request: object):
            requests.append(request)
            return _inspection()

    monkeypatch.setattr(
        module,
        "DurableReplayResultArtifactConsumerRequest",
        FakeRequest,
    )

    service = FakeInspectionService()
    adapter = owner(inspection_service=service)

    result = adapter.adapt(
        artifact_root=r".\explicit\artifact-root",
        end_round=77,
    )

    assert result == _inspection()
    assert len(requests) == 1
    assert requests[0].artifact_root == r".\explicit\artifact-root"
    assert requests[0].end_round == 77


@pytest.mark.parametrize(
    "artifact_root",
    [
        r"relative\artifact-root",
        r".\relative\artifact-root",
        r"..\relative\artifact-root",
        r"folder with spaces\artifact-root",
        r"%USERPROFILE%\artifact-root",
        r"~\artifact-root",
    ],
)
def test_artifact_root_text_is_forwarded_unchanged(
    monkeypatch,
    artifact_root: str,
) -> None:
    module = _module()
    owner = _class()

    seen: list[object] = []

    class FakeRequest:
        def __init__(self, artifact_root: object, end_round: object) -> None:
            seen.append(artifact_root)
            self.artifact_root = artifact_root
            self.end_round = end_round

    class FakeInspectionService:
        def inspect(self, *, request: object):
            return _inspection()

    monkeypatch.setattr(
        module,
        "DurableReplayResultArtifactConsumerRequest",
        FakeRequest,
    )

    owner(inspection_service=FakeInspectionService()).adapt(
        artifact_root=artifact_root,
        end_round=100,
    )

    assert seen == [artifact_root]


@pytest.mark.parametrize("end_round", [1, 9, 10, 999, 1234])
def test_end_round_is_forwarded_unchanged(
    monkeypatch,
    end_round: int,
) -> None:
    module = _module()
    owner = _class()

    seen: list[object] = []

    class FakeRequest:
        def __init__(self, artifact_root: object, end_round: object) -> None:
            seen.append(end_round)
            self.artifact_root = artifact_root
            self.end_round = end_round

    class FakeInspectionService:
        def inspect(self, *, request: object):
            return _inspection()

    monkeypatch.setattr(
        module,
        "DurableReplayResultArtifactConsumerRequest",
        FakeRequest,
    )

    owner(inspection_service=FakeInspectionService()).adapt(
        artifact_root="artifact-root",
        end_round=end_round,
    )

    assert seen == [end_round]


def test_inspection_failure_propagates_unchanged(monkeypatch) -> None:
    module = _module()
    owner = _class()

    failure = ValueError("inspection-failure")

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

    with pytest.raises(ValueError) as exc_info:
        owner(inspection_service=FakeInspectionService()).adapt(
            artifact_root="artifact-root",
            end_round=1,
        )

    assert exc_info.value is failure


def test_adapter_does_not_read_or_parse_artifact_files() -> None:
    source = _source()
    forbidden = [
        "json.loads",
        "json.dumps",
        "read_text(",
        "read_bytes(",
        "open(",
        "verify_manifest",
        "manifest.json",
        "evaluation_result.json",
    ]
    for token in forbidden:
        assert token not in source


def test_adapter_does_not_duplicate_downstream_domain_layers() -> None:
    source = _source()
    forbidden = [
        "DurableReplayResultComparisonSummary",
        "DurableReplayResultComparisonAssessment",
        "DurableReplayResultPromotionEligibility",
        "DurableReplayResultPromotionActionPlan",
        "DurableReplayPromotionPublicationRequest",
    ]
    for token in forbidden:
        assert token not in source


def test_adapter_has_no_cli_execution_or_mutation_surface() -> None:
    source = _source()
    forbidden = [
        "argparse",
        "lrp.cli",
        "stdin",
        "stdout",
        "DurableReplayPublicationLifecycleEntrypoint",
        "DurableReplayPromotionPublicationExecutionService",
        "ProductionChampionRegistryPublisher",
        "run_publication_stage",
        ".publish(",
        "rollback",
    ]
    for token in forbidden:
        assert token not in source


def test_adapter_has_no_discovery_or_path_normalization_surface() -> None:
    source = _source()
    forbidden = [
        "resolve()",
        "expanduser",
        "getenv(",
        "environ",
        "glob(",
        "rglob(",
        "latest",
    ]
    for token in forbidden:
        assert token not in source


def test_adapter_import_boundary_is_exact() -> None:
    source = _source()
    tree = ast.parse(source)

    imports: set[str] = set()

    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imports.add(node.module)

    allowed = {
        "__future__",
        "pathlib",
        "lrp.operations.durable_replay_result_artifact_consumer",
        "lrp.operations.durable_replay_result_artifact_inspection",
    }

    assert imports <= allowed
    assert (
        "lrp.operations.durable_replay_result_artifact_consumer"
        in imports
    )
    assert (
        "lrp.operations.durable_replay_result_artifact_inspection"
        in imports
    )


def test_adapter_ast_has_exact_request_and_inspection_calls() -> None:
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


def test_adapter_declares_no_second_result_model() -> None:
    tree = ast.parse(_source())

    classes = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    ]

    assert classes == [CLASS_NAME]
