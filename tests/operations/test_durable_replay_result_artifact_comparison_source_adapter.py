from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


MODULE_NAME = "lrp.operations.durable_replay_result_artifact_comparison_source_adapter"
CLASS_NAME = "DurableReplayResultArtifactComparisonSourceAdapter"
PRODUCT_PATH = Path(
    "lrp/operations/durable_replay_result_artifact_comparison_source_adapter.py"
)

SOURCE_MODULE = "lrp.operations.durable_replay_result_artifact_source_adapter"
SUMMARY_MODULE = "lrp.operations.durable_replay_result_comparison_summary"
ASSESSMENT_MODULE = "lrp.operations.durable_replay_result_comparison_assessment"


def _module() -> ModuleType:
    return importlib.import_module(MODULE_NAME)


def _class():
    module = _module()
    return getattr(module, CLASS_NAME)


def _source() -> str:
    return PRODUCT_PATH.read_text(encoding="utf-8-sig")


def _ast() -> ast.Module:
    return ast.parse(_source(), filename=str(PRODUCT_PATH))


def _call_name(node: ast.Call) -> str:
    return ast.unparse(node.func)


def test_product_module_exists() -> None:
    assert importlib.util.find_spec(MODULE_NAME) is not None


def test_product_class_exists() -> None:
    cls = _class()
    assert cls.__name__ == CLASS_NAME


def test_public_methods_are_exact() -> None:
    cls = _class()
    public = {
        name
        for name, value in cls.__dict__.items()
        if callable(value) and not name.startswith("_")
    }
    assert public == {"adapt"}


def test_init_signature_is_minimal() -> None:
    cls = _class()
    signature = inspect.signature(cls)
    assert list(signature.parameters) == [
        "source_adapter",
        "summary_service",
        "assessment_service",
    ]
    for parameter in signature.parameters.values():
        assert parameter.default is None


def test_adapt_signature_is_exact() -> None:
    cls = _class()
    signature = inspect.signature(cls.adapt)

    assert list(signature.parameters) == [
        "self",
        "artifact_root",
        "end_round",
    ]

    hints = inspect.get_annotations(cls.adapt, eval_str=True)

    assert str(hints["artifact_root"]) in {
        "str | pathlib.Path",
        "str | Path",
    } or hints["artifact_root"].__class__.__name__ == "UnionType"

    assert hints["end_round"] is int
    assert hints["return"].__name__ == "DurableReplayResultComparisonAssessment"


def test_constructor_owns_or_receives_exact_three_dependencies() -> None:
    cls = _class()
    source_adapter = object()
    summary_service = object()
    assessment_service = object()

    instance = cls(
        source_adapter=source_adapter,
        summary_service=summary_service,
        assessment_service=assessment_service,
    )

    assert instance._source_adapter is source_adapter
    assert instance._summary_service is summary_service
    assert instance._assessment_service is assessment_service


def test_adapt_calls_source_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    cls = _class()

    inspection = object()

    class Source:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, int]] = []

        def adapt(self, artifact_root: Any, end_round: int) -> Any:
            self.calls.append((artifact_root, end_round))
            return inspection

    class Summary:
        def summarize(self, value: Any) -> Any:
            return object()

    class Assessment:
        def assess(self, value: Any) -> Any:
            return object()

    source = Source()
    adapter = cls(
        source_adapter=source,
        summary_service=Summary(),
        assessment_service=Assessment(),
    )

    root = object()
    adapter.adapt(root, 1234)

    assert source.calls == [(root, 1234)]


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
def test_artifact_root_is_forwarded_unchanged(artifact_root: str) -> None:
    cls = _class()

    class Source:
        def __init__(self) -> None:
            self.received = None

        def adapt(self, root, end_round):
            self.received = root
            return object()

    class Summary:
        def summarize(self, value):
            return object()

    class Assessment:
        def assess(self, value):
            return object()

    source = Source()
    adapter = cls(
        source_adapter=source,
        summary_service=Summary(),
        assessment_service=Assessment(),
    )

    adapter.adapt(artifact_root, 1234)
    assert source.received == artifact_root


@pytest.mark.parametrize("end_round", [-1, 0, 1, 9, 999, 1234, 999999])
def test_end_round_is_forwarded_unchanged(end_round: int) -> None:
    cls = _class()

    class Source:
        def __init__(self) -> None:
            self.received = None

        def adapt(self, artifact_root, round_value):
            self.received = round_value
            return object()

    class Summary:
        def summarize(self, value):
            return object()

    class Assessment:
        def assess(self, value):
            return object()

    source = Source()
    adapter = cls(
        source_adapter=source,
        summary_service=Summary(),
        assessment_service=Assessment(),
    )

    adapter.adapt("root", end_round)
    assert source.received == end_round


def test_inspection_is_forwarded_to_summary_exactly_once() -> None:
    cls = _class()

    inspection = object()
    summary = object()

    class Source:
        def adapt(self, artifact_root, end_round):
            return inspection

    class Summary:
        def __init__(self) -> None:
            self.calls = []

        def summarize(self, value):
            self.calls.append(value)
            return summary

    class Assessment:
        def assess(self, value):
            return object()

    summary_service = Summary()

    adapter = cls(
        source_adapter=Source(),
        summary_service=summary_service,
        assessment_service=Assessment(),
    )

    adapter.adapt("root", 1234)

    assert summary_service.calls == [inspection]


def test_summary_is_forwarded_to_assessment_exactly_once() -> None:
    cls = _class()

    inspection = object()
    summary = object()

    class Source:
        def adapt(self, artifact_root, end_round):
            return inspection

    class Summary:
        def summarize(self, value):
            return summary

    class Assessment:
        def __init__(self) -> None:
            self.calls = []

        def assess(self, value):
            self.calls.append(value)
            return object()

    assessment_service = Assessment()

    adapter = cls(
        source_adapter=Source(),
        summary_service=Summary(),
        assessment_service=assessment_service,
    )

    adapter.adapt("root", 1234)

    assert assessment_service.calls == [summary]


def test_assessment_is_returned_unchanged() -> None:
    cls = _class()

    assessment = object()

    class Source:
        def adapt(self, artifact_root, end_round):
            return object()

    class Summary:
        def summarize(self, value):
            return object()

    class Assessment:
        def assess(self, value):
            return assessment

    adapter = cls(
        source_adapter=Source(),
        summary_service=Summary(),
        assessment_service=Assessment(),
    )

    result = adapter.adapt("root", 1234)
    assert result is assessment


@pytest.mark.parametrize(
    "owner",
    ["source", "summary", "assessment"],
)
def test_failures_propagate_unchanged(owner: str) -> None:
    cls = _class()

    failure = RuntimeError(owner)

    class Source:
        def adapt(self, artifact_root, end_round):
            if owner == "source":
                raise failure
            return object()

    class Summary:
        def summarize(self, value):
            if owner == "summary":
                raise failure
            return object()

    class Assessment:
        def assess(self, value):
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


def test_adapter_does_not_construct_lower_layer_artifact_objects() -> None:
    source = _source()

    forbidden = [
        "DurableReplayResultArtifactConsumer(",
        "DurableReplayResultArtifactConsumerRequest(",
        "DurableReplayResultArtifactInspectionService(",
        "DurableReplayResultArtifactInspection(",
    ]

    for token in forbidden:
        assert token not in source


def test_adapter_does_not_duplicate_summary_or_assessment_models() -> None:
    source = _source()

    forbidden = [
        "DurableReplayResultComparisonSummary(",
        "DurableReplayResultComparisonAssessment(",
        "baseline_delta_mean_best_hits",
        "baseline_delta_3plus_rate",
        "baseline_delta_4plus_rate",
        "recommendation=",
    ]

    for token in forbidden:
        assert token not in source


def test_adapter_has_no_promotion_or_publication_surface() -> None:
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
        "rollback",
    ]

    for token in forbidden:
        assert token not in source


def test_adapter_has_no_cli_file_io_or_discovery_surface() -> None:
    source = _source()

    forbidden = [
        "argparse",
        "lrp.cli",
        "stdin",
        "stdout",
        "open(",
        "read_text(",
        "read_bytes(",
        "write_text(",
        "write_bytes(",
        "json.loads",
        "json.dumps",
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


def test_import_boundary_is_exact() -> None:
    tree = _ast()

    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)

    allowed = {
        "__future__",
        "pathlib",
        SOURCE_MODULE,
        SUMMARY_MODULE,
        ASSESSMENT_MODULE,
    }

    assert imports == allowed


def test_ast_call_graph_is_exact() -> None:
    tree = _ast()

    calls = [
        _call_name(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]

    assert calls.count("DurableReplayResultArtifactSourceAdapter") == 1
    assert calls.count("DurableReplayResultComparisonSummaryService") == 1
    assert calls.count("DurableReplayResultComparisonAssessmentService") == 1

    assert calls.count("self._source_adapter.adapt") == 1
    assert calls.count("self._summary_service.summarize") == 1
    assert calls.count("self._assessment_service.assess") == 1

    forbidden_calls = {
        "DurableReplayResultPromotionEligibilityService",
        "DurableReplayResultPromotionActionPlanService",
        "DurableReplayPromotionPublicationRequestService",
        "DurableReplayPublicationLifecycleEntrypoint",
        "ProductionChampionRegistryPublisher",
    }

    assert set(calls).isdisjoint(forbidden_calls)


def test_adapter_declares_only_one_product_class() -> None:
    tree = _ast()

    classes = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    ]

    assert classes == [CLASS_NAME]
