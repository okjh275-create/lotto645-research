from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import get_type_hints

import pytest

MODULE_NAME = (
    "lrp.operations."
    "durable_replay_result_artifact_promotion_eligibility_source_adapter"
)
CLASS_NAME = "DurableReplayResultArtifactPromotionEligibilitySourceAdapter"
PRODUCT_PATH = Path(
    "lrp/operations/"
    "durable_replay_result_artifact_promotion_eligibility_source_adapter.py"
)


def _module():
    return importlib.import_module(MODULE_NAME)


def _class():
    return getattr(_module(), CLASS_NAME)


class _Source:
    def __init__(self, *, result=None, failure=None):
        self.result = result
        self.failure = failure
        self.calls = []

    def adapt(self, artifact_root, end_round):
        self.calls.append((artifact_root, end_round))
        if self.failure is not None:
            raise self.failure
        return self.result


class _Eligibility:
    def __init__(self, *, result=None, failure=None):
        self.result = result
        self.failure = failure
        self.calls = []

    def evaluate(self, assessment):
        self.calls.append(assessment)
        if self.failure is not None:
            raise self.failure
        return self.result


def _adapter(source, eligibility):
    cls = _class()
    return cls(
        source_adapter=source,
        eligibility_service=eligibility,
    )


def test_product_module_exists() -> None:
    assert importlib.util.find_spec(MODULE_NAME) is not None


def test_product_class_exists() -> None:
    assert hasattr(_module(), CLASS_NAME)


def test_public_methods_are_exact() -> None:
    names = {
        name
        for name, value in vars(_class()).items()
        if callable(value) and not name.startswith("_")
    }
    assert names == {"adapt"}


def test_init_signature_is_minimal() -> None:
    signature = inspect.signature(_class())
    assert list(signature.parameters) == [
        "source_adapter",
        "eligibility_service",
    ]
    assert all(
        parameter.default is None
        for parameter in signature.parameters.values()
    )


def test_adapt_signature_is_exact() -> None:
    signature = inspect.signature(_class().adapt)
    assert list(signature.parameters) == [
        "self",
        "artifact_root",
        "end_round",
    ]

    hints = get_type_hints(_class().adapt)
    assert hints["end_round"] is int
    assert hints["return"].__name__ == "DurableReplayResultPromotionEligibility"


def test_constructor_owns_or_receives_exact_two_dependencies() -> None:
    source = _Source()
    eligibility = _Eligibility()
    adapter = _adapter(source, eligibility)

    assert adapter._source_adapter is source
    assert adapter._eligibility_service is eligibility


def test_adapt_calls_source_exactly_once() -> None:
    assessment = object()
    eligibility_result = object()
    source = _Source(result=assessment)
    eligibility = _Eligibility(result=eligibility_result)

    result = _adapter(source, eligibility).adapt("artifact-root", 1234)

    assert source.calls == [("artifact-root", 1234)]
    assert result is eligibility_result


@pytest.mark.parametrize(
    "artifact_root",
    [
        r"relative\artifact-root",
        r".\relative\artifact-root",
        r"..\relative\artifact-root",
        r"folder with spaces\artifact-root",
        r"%USERPROFILE%\artifact-root",
        r"~\artifact-root",
        Path("path-object") / "artifact-root",
    ],
)
def test_artifact_root_is_forwarded_unchanged(artifact_root) -> None:
    assessment = object()
    source = _Source(result=assessment)
    eligibility = _Eligibility(result=object())

    _adapter(source, eligibility).adapt(artifact_root, 1234)

    forwarded_root, _ = source.calls[0]
    assert forwarded_root is artifact_root


@pytest.mark.parametrize(
    "end_round",
    [-1, 0, 1, 2, 9, 999, 1234, 999999],
)
def test_end_round_is_forwarded_unchanged(end_round: int) -> None:
    source = _Source(result=object())
    eligibility = _Eligibility(result=object())

    _adapter(source, eligibility).adapt("artifact-root", end_round)

    _, forwarded_round = source.calls[0]
    assert forwarded_round == end_round


def test_assessment_is_forwarded_to_eligibility_exactly_once() -> None:
    assessment = object()
    source = _Source(result=assessment)
    eligibility = _Eligibility(result=object())

    _adapter(source, eligibility).adapt("artifact-root", 1234)

    assert eligibility.calls == [assessment]
    assert eligibility.calls[0] is assessment


def test_eligibility_is_returned_unchanged() -> None:
    eligibility_result = object()
    source = _Source(result=object())
    eligibility = _Eligibility(result=eligibility_result)

    result = _adapter(source, eligibility).adapt("artifact-root", 1234)

    assert result is eligibility_result


@pytest.mark.parametrize("owner", ["source", "eligibility"])
def test_failures_propagate_unchanged(owner: str) -> None:
    failure = RuntimeError(owner)

    if owner == "source":
        source = _Source(failure=failure)
        eligibility = _Eligibility(result=object())
    else:
        source = _Source(result=object())
        eligibility = _Eligibility(failure=failure)

    with pytest.raises(RuntimeError) as captured:
        _adapter(source, eligibility).adapt("artifact-root", 1234)

    assert captured.value is failure


def test_adapter_does_not_construct_lower_layer_artifact_objects() -> None:
    text = PRODUCT_PATH.read_text(encoding="utf-8-sig")
    forbidden = [
        "DurableReplayResultArtifactConsumer(",
        "DurableReplayResultArtifactConsumerRequest(",
        "DurableReplayResultArtifactInspectionService(",
        "DurableReplayResultComparisonSummaryService(",
        "DurableReplayResultComparisonAssessmentService(",
    ]
    assert all(token not in text for token in forbidden)


def test_adapter_does_not_duplicate_eligibility_model_or_policy() -> None:
    text = PRODUCT_PATH.read_text(encoding="utf-8-sig")
    forbidden = [
        "DurableReplayResultPromotionEligibility(",
        "recommendation=",
        "eligible=",
        "reason=",
        "policy",
    ]
    assert all(token not in text for token in forbidden)


def test_adapter_has_no_action_plan_or_publication_surface() -> None:
    text = PRODUCT_PATH.read_text(encoding="utf-8-sig")
    forbidden = [
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
    assert all(token not in text for token in forbidden)


def test_adapter_has_no_cli_file_io_or_discovery_surface() -> None:
    text = PRODUCT_PATH.read_text(encoding="utf-8-sig")
    forbidden = [
        "json.loads",
        "json.dumps",
        "open(",
        "read_text(",
        "read_bytes(",
        "write_text(",
        "write_bytes(",
        "argparse",
        "lrp.cli",
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
    assert all(token not in text for token in forbidden)


def test_import_boundary_is_exact() -> None:
    tree = ast.parse(PRODUCT_PATH.read_text(encoding="utf-8-sig"))
    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)

    assert imports == {
        "__future__",
        "pathlib",
        (
            "lrp.operations."
            "durable_replay_result_artifact_comparison_source_adapter"
        ),
        "lrp.operations.durable_replay_result_promotion_eligibility",
    }


def test_ast_call_graph_is_exact() -> None:
    tree = ast.parse(PRODUCT_PATH.read_text(encoding="utf-8-sig"))

    def dotted(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            left = dotted(node.value)
            return f"{left}.{node.attr}" if left else node.attr
        return None

    calls = [
        dotted(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]

    assert calls.count(
        "DurableReplayResultArtifactComparisonSourceAdapter"
    ) == 1
    assert calls.count("DurableReplayResultPromotionEligibilityService") == 1
    assert calls.count("self._source_adapter.adapt") == 1
    assert calls.count("self._eligibility_service.evaluate") == 1
    assert len(calls) == 4


def test_adapter_declares_only_one_product_class() -> None:
    tree = ast.parse(PRODUCT_PATH.read_text(encoding="utf-8-sig"))
    classes = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    ]
    assert classes == [CLASS_NAME]
