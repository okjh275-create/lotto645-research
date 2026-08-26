from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import Any

import pytest


MODULE_NAME = "lrp.operations.durable_replay_result_artifact_comparison_source_adapter"
CLASS_NAME = "DurableReplayResultArtifactComparisonSourceAdapter"
PRODUCT_PATH = Path(
    "lrp/operations/durable_replay_result_artifact_comparison_source_adapter.py"
)


def _module():
    return importlib.import_module(MODULE_NAME)


def _class():
    return getattr(_module(), CLASS_NAME)


def _source() -> str:
    return PRODUCT_PATH.read_text(encoding="utf-8-sig")


def _tree() -> ast.Module:
    return ast.parse(_source(), filename=str(PRODUCT_PATH))


@pytest.mark.parametrize(
    "artifact_root",
    [
        r"relative\artifact-root",
        r".\relative\artifact-root",
        r"..\relative\artifact-root",
        r"folder with spaces\artifact-root",
        r"%USERPROFILE%\artifact-root",
        r"~\artifact-root",
        object(),
    ],
)
def test_artifact_root_identity_is_forwarded_exactly(artifact_root: Any) -> None:
    cls = _class()

    class Source:
        def __init__(self) -> None:
            self.received = None

        def adapt(self, root: Any, end_round: int) -> object:
            self.received = root
            return object()

    class Summary:
        def summarize(self, inspection: object) -> object:
            return object()

    class Assessment:
        def assess(self, summary: object) -> object:
            return object()

    source = Source()
    adapter = cls(
        source_adapter=source,
        summary_service=Summary(),
        assessment_service=Assessment(),
    )

    adapter.adapt(artifact_root, 1234)

    if isinstance(artifact_root, str):
        assert source.received == artifact_root
    else:
        assert source.received is artifact_root


@pytest.mark.parametrize(
    "end_round",
    [-1, 0, 1, 2, 9, 999, 1234, 999999],
)
def test_end_round_identity_is_forwarded_without_adapter_validation(
    end_round: int,
) -> None:
    cls = _class()

    class Source:
        def __init__(self) -> None:
            self.received = None

        def adapt(self, root: Any, value: int) -> object:
            self.received = value
            return object()

    class Summary:
        def summarize(self, inspection: object) -> object:
            return object()

    class Assessment:
        def assess(self, summary: object) -> object:
            return object()

    source = Source()
    adapter = cls(
        source_adapter=source,
        summary_service=Summary(),
        assessment_service=Assessment(),
    )

    adapter.adapt("root", end_round)
    assert source.received == end_round


def test_inspection_identity_is_forwarded_exactly() -> None:
    cls = _class()
    inspection = object()

    class Source:
        def adapt(self, root: Any, end_round: int) -> object:
            return inspection

    class Summary:
        def __init__(self) -> None:
            self.received = None

        def summarize(self, value: object) -> object:
            self.received = value
            return object()

    class Assessment:
        def assess(self, summary: object) -> object:
            return object()

    summary_service = Summary()
    adapter = cls(
        source_adapter=Source(),
        summary_service=summary_service,
        assessment_service=Assessment(),
    )

    adapter.adapt("root", 1234)
    assert summary_service.received is inspection


def test_summary_identity_is_forwarded_exactly() -> None:
    cls = _class()
    summary = object()

    class Source:
        def adapt(self, root: Any, end_round: int) -> object:
            return object()

    class Summary:
        def summarize(self, inspection: object) -> object:
            return summary

    class Assessment:
        def __init__(self) -> None:
            self.received = None

        def assess(self, value: object) -> object:
            self.received = value
            return object()

    assessment_service = Assessment()
    adapter = cls(
        source_adapter=Source(),
        summary_service=Summary(),
        assessment_service=assessment_service,
    )

    adapter.adapt("root", 1234)
    assert assessment_service.received is summary


def test_assessment_return_identity_is_preserved() -> None:
    cls = _class()
    assessment = object()

    class Source:
        def adapt(self, root: Any, end_round: int) -> object:
            return object()

    class Summary:
        def summarize(self, inspection: object) -> object:
            return object()

    class Assessment:
        def assess(self, summary: object) -> object:
            return assessment

    adapter = cls(
        source_adapter=Source(),
        summary_service=Summary(),
        assessment_service=Assessment(),
    )

    assert adapter.adapt("root", 1234) is assessment


@pytest.mark.parametrize("owner", ["source", "summary", "assessment"])
def test_dependency_failures_propagate_by_identity(owner: str) -> None:
    cls = _class()
    failure = RuntimeError(owner)

    class Source:
        def adapt(self, root: Any, end_round: int) -> object:
            if owner == "source":
                raise failure
            return object()

    class Summary:
        def summarize(self, inspection: object) -> object:
            if owner == "summary":
                raise failure
            return object()

    class Assessment:
        def assess(self, summary: object) -> object:
            if owner == "assessment":
                raise failure
            return object()

    adapter = cls(
        source_adapter=Source(),
        summary_service=Summary(),
        assessment_service=Assessment(),
    )

    with pytest.raises(RuntimeError) as exc:
        adapter.adapt("root", 1234)

    assert exc.value is failure


def test_adapter_does_not_catch_or_translate_failures() -> None:
    tree = _tree()

    try_nodes = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Try)
    ]
    raise_nodes = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Raise)
    ]

    assert try_nodes == []
    assert raise_nodes == []


def test_adapter_constructor_has_exact_three_optional_dependencies() -> None:
    cls = _class()
    signature = inspect.signature(cls)

    assert list(signature.parameters) == [
        "source_adapter",
        "summary_service",
        "assessment_service",
    ]

    for parameter in signature.parameters.values():
        assert parameter.default is None


def test_adapter_owns_default_dependencies_once_each() -> None:
    source = _source()
    tree = _tree()

    calls = [
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]

    assert calls.count("DurableReplayResultArtifactSourceAdapter") == 1
    assert calls.count("DurableReplayResultComparisonSummaryService") == 1
    assert calls.count("DurableReplayResultComparisonAssessmentService") == 1


def test_adapter_has_exact_three_operational_imports() -> None:
    tree = _tree()

    imported_modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module not in {"__future__", "pathlib"}:
                imported_modules.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in {"pathlib"}:
                    imported_modules.add(alias.name)

    assert imported_modules == {
        "lrp.operations.durable_replay_result_artifact_source_adapter",
        "lrp.operations.durable_replay_result_comparison_summary",
        "lrp.operations.durable_replay_result_comparison_assessment",
    }


def test_adapter_has_no_direct_artifact_consumer_or_inspection_dependency() -> None:
    source = _source()

    forbidden = [
        "durable_replay_result_artifact_consumer",
        "DurableReplayResultArtifactConsumer",
        "DurableReplayResultArtifactConsumerRequest",
        "durable_replay_result_artifact_inspection",
        "DurableReplayResultArtifactInspectionService",
        "DurableReplayResultArtifactInspection(",
    ]

    for token in forbidden:
        assert token not in source


def test_adapter_has_no_artifact_file_io_surface() -> None:
    source = _source()

    forbidden = [
        "open(",
        "read_text(",
        "read_bytes(",
        "write_text(",
        "write_bytes(",
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
        "latest",
        "discover",
    ]

    for token in forbidden:
        assert token not in source


def test_adapter_has_no_local_summary_or_assessment_reconstruction() -> None:
    source = _source()

    forbidden = [
        "DurableReplayResultComparisonSummary(",
        "DurableReplayResultComparisonAssessment(",
        "baseline_delta_mean_best_hits",
        "baseline_delta_3plus_rate",
        "baseline_delta_4plus_rate",
        "recommendation=",
        "status=",
        "round_count=",
        "candidate_model_name=",
        "baseline_model_name=",
        "window=",
    ]

    for token in forbidden:
        assert token not in source


def test_adapter_has_no_promotion_or_publication_logic() -> None:
    source = _source()

    forbidden = [
        "DurableReplayResultPromotionEligibility",
        "DurableReplayResultPromotionActionPlan",
        "DurableReplayPromotionPublicationRequest",
        "DurableReplayPublicationLifecycleEntrypoint",
        "DurableReplayPromotionPublicationExecutionService",
        "ProductionChampionRegistryPublisher",
        "run_publication_stage",
        ".publish(",
        "registry_root",
        "source_decision",
        "eligibility",
        "promotion_policy",
        "rollback",
    ]

    for token in forbidden:
        assert token not in source


def test_adapter_has_no_cli_or_stream_surface() -> None:
    source = _source()

    forbidden = [
        "argparse",
        "lrp.cli",
        "stdin",
        "stdout",
        "sys.stdin",
        "sys.stdout",
    ]

    for token in forbidden:
        assert token not in source


def test_adapt_call_graph_is_exact_and_ordered() -> None:
    tree = _tree()

    adapt = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == CLASS_NAME
        for node in node.body
        if isinstance(node, ast.FunctionDef) and node.name == "adapt"
    )

    calls = [
        ast.unparse(node.func)
        for node in ast.walk(adapt)
        if isinstance(node, ast.Call)
    ]

    assert calls.count("self._source_adapter.adapt") == 1
    assert calls.count("self._summary_service.summarize") == 1
    assert calls.count("self._assessment_service.assess") == 1

    source_index = calls.index("self._source_adapter.adapt")
    summary_index = calls.index("self._summary_service.summarize")
    assessment_index = calls.index("self._assessment_service.assess")

    assert source_index < summary_index < assessment_index


def test_adapt_has_no_local_input_rewrite_assignments() -> None:
    tree = _tree()

    adapt = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == CLASS_NAME
        for node in node.body
        if isinstance(node, ast.FunctionDef) and node.name == "adapt"
    )

    forbidden_targets = {"artifact_root", "end_round"}

    for node in ast.walk(adapt):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assert target.id not in forbidden_targets
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assert node.target.id not in forbidden_targets


def test_adapter_declares_only_one_product_class() -> None:
    tree = _tree()

    classes = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    ]

    assert classes == [CLASS_NAME]
