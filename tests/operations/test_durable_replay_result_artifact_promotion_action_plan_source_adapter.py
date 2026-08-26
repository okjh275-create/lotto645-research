from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from pathlib import Path
from types import ModuleType

import pytest

MODULE_NAME = (
    "lrp.operations."
    "durable_replay_result_artifact_promotion_action_plan_source_adapter"
)
CLASS_NAME = (
    "DurableReplayResultArtifactPromotionActionPlanSourceAdapter"
)
PRODUCT_PATH = Path(
    "lrp/operations/"
    "durable_replay_result_artifact_promotion_action_plan_source_adapter.py"
)


def _module() -> ModuleType:
    return importlib.import_module(MODULE_NAME)


def _product_class():
    return getattr(_module(), CLASS_NAME)


def _source_text() -> str:
    return PRODUCT_PATH.read_text(encoding="utf-8-sig")


def test_product_module_exists() -> None:
    assert importlib.util.find_spec(MODULE_NAME) is not None


def test_product_class_exists() -> None:
    module = _module()
    assert hasattr(module, CLASS_NAME)


def test_public_methods_are_exact() -> None:
    cls = _product_class()
    methods = {
        name
        for name, member in cls.__dict__.items()
        if callable(member) and not name.startswith("_")
    }
    assert methods == {"adapt"}


def test_init_signature_is_minimal() -> None:
    cls = _product_class()
    assert str(inspect.signature(cls)) == (
        "(source_adapter: "
        "'DurableReplayResultArtifactPromotionEligibilitySourceAdapter | None' = None, "
        "action_plan_service: "
        "'DurableReplayResultPromotionActionPlanService | None' = None"
        ") -> 'None'"
    )


def test_adapt_signature_is_exact() -> None:
    cls = _product_class()
    assert str(inspect.signature(cls.adapt)) == (
        "(self, artifact_root: 'str | Path', end_round: 'int') "
        "-> 'DurableReplayResultPromotionActionPlan'"
    )


def test_constructor_owns_or_receives_exact_two_dependencies() -> None:
    cls = _product_class()
    params = list(inspect.signature(cls).parameters.values())
    assert [p.name for p in params] == [
        "source_adapter",
        "action_plan_service",
    ]
    assert all(p.default is None for p in params)


class _SourceStub:
    def __init__(self, result=None, failure=None):
        self.result = result
        self.failure = failure
        self.calls = []

    def adapt(self, artifact_root, end_round):
        self.calls.append((artifact_root, end_round))
        if self.failure is not None:
            raise self.failure
        return self.result


class _PlanStub:
    def __init__(self, result=None, failure=None):
        self.result = result
        self.failure = failure
        self.calls = []

    def plan(self, eligibility):
        self.calls.append(eligibility)
        if self.failure is not None:
            raise self.failure
        return self.result


def test_adapt_calls_source_exactly_once() -> None:
    eligibility = object()
    result = object()
    source = _SourceStub(result=eligibility)
    planner = _PlanStub(result=result)
    product = _product_class()(source, planner)

    returned = product.adapt("root", 1234)

    assert source.calls == [("root", 1234)]
    assert planner.calls == [eligibility]
    assert returned is result


@pytest.mark.parametrize(
    "artifact_root",
    [
        r"relative\artifact-root",
        r".\relative\artifact-root",
        r"..\relative\artifact-root",
        r"folder with spaces\artifact-root",
        r"%USERPROFILE%\artifact-root",
        r"~\artifact-root",
        Path(r"artifact_root6"),
    ],
)
def test_artifact_root_is_forwarded_unchanged(artifact_root) -> None:
    eligibility = object()
    source = _SourceStub(result=eligibility)
    planner = _PlanStub(result=object())
    product = _product_class()(source, planner)

    product.adapt(artifact_root, 1234)

    assert source.calls[0][0] is artifact_root


@pytest.mark.parametrize(
    "end_round",
    [-1, 0, 1, 2, 9, 999, 1234, 999999],
)
def test_end_round_is_forwarded_unchanged(end_round: int) -> None:
    eligibility = object()
    source = _SourceStub(result=eligibility)
    planner = _PlanStub(result=object())
    product = _product_class()(source, planner)

    product.adapt("root", end_round)

    assert source.calls == [("root", end_round)]


def test_eligibility_is_forwarded_to_action_plan_exactly_once() -> None:
    eligibility = object()
    source = _SourceStub(result=eligibility)
    planner = _PlanStub(result=object())
    product = _product_class()(source, planner)

    product.adapt("root", 1234)

    assert planner.calls == [eligibility]
    assert planner.calls[0] is eligibility


def test_action_plan_is_returned_unchanged() -> None:
    eligibility = object()
    result = object()
    source = _SourceStub(result=eligibility)
    planner = _PlanStub(result=result)
    product = _product_class()(source, planner)

    assert product.adapt("root", 1234) is result


@pytest.mark.parametrize("owner", ["source", "action_plan"])
def test_failures_propagate_unchanged(owner: str) -> None:
    failure = RuntimeError(owner)
    eligibility = object()

    if owner == "source":
        source = _SourceStub(failure=failure)
        planner = _PlanStub(result=object())
    else:
        source = _SourceStub(result=eligibility)
        planner = _PlanStub(failure=failure)

    product = _product_class()(source, planner)

    with pytest.raises(RuntimeError) as exc_info:
        product.adapt("root", 1234)

    assert exc_info.value is failure


def test_adapter_does_not_construct_lower_layer_objects() -> None:
    text = _source_text()
    forbidden = [
        "DurableReplayResultArtifactConsumer(",
        "DurableReplayResultArtifactConsumerRequest(",
        "DurableReplayResultArtifactInspectionService(",
        "DurableReplayResultComparisonSummaryService(",
        "DurableReplayResultComparisonAssessmentService(",
        "DurableReplayResultPromotionEligibilityService(",
        "DurableReplayResultPromotionEligibility(",
    ]
    for token in forbidden:
        assert token not in text


def test_adapter_does_not_duplicate_action_plan_model_or_policy() -> None:
    text = _source_text()
    forbidden = [
        "DurableReplayResultPromotionActionPlan(",
        "recommendation=",
        "action=",
        "eligibility.recommendation",
        "eligibility.status",
    ]
    for token in forbidden:
        assert token not in text


def test_adapter_has_no_publication_request_or_execution_surface() -> None:
    text = _source_text()
    forbidden = [
        "DurableReplayPromotionPublicationRequest",
        "DurableReplayPromotionPublicationRequestService",
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
        assert token not in text


def test_adapter_has_no_cli_file_io_or_discovery_surface() -> None:
    text = _source_text()
    forbidden = [
        "argparse",
        "lrp.cli",
        "stdin",
        "stdout",
        "json.loads",
        "json.dumps",
        "open(",
        "read_text(",
        "read_bytes(",
        "write_text(",
        "write_bytes(",
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
        assert token not in text


def test_import_boundary_is_exact() -> None:
    tree = ast.parse(_source_text())

    imports = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            imports.append(node.module)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)

    assert set(imports) == {
        "__future__",
        "pathlib",
        (
            "lrp.operations."
            "durable_replay_result_artifact_promotion_eligibility_source_adapter"
        ),
        "lrp.operations.durable_replay_result_promotion_action_plan",
    }


def test_ast_call_graph_is_exact() -> None:
    tree = ast.parse(_source_text())

    calls = [
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]

    assert calls.count(
        "DurableReplayResultArtifactPromotionEligibilitySourceAdapter"
    ) == 1
    assert calls.count(
        "DurableReplayResultPromotionActionPlanService"
    ) == 1
    assert calls.count("self._source_adapter.adapt") == 1
    assert calls.count("self._action_plan_service.plan") == 1

    assert len(calls) == 4


def test_adapter_declares_only_one_product_class() -> None:
    tree = ast.parse(_source_text())
    classes = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    ]
    assert classes == [CLASS_NAME]
