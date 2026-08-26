from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from lrp.operations.durable_replay_result_artifact_promotion_eligibility_source_adapter import (
    DurableReplayResultArtifactPromotionEligibilitySourceAdapter,
)

PRODUCT_PATH = Path(
    "lrp/operations/"
    "durable_replay_result_artifact_promotion_eligibility_source_adapter.py"
)


class Source:
    def __init__(self, *, result=None, failure=None):
        self.result = result
        self.failure = failure
        self.calls = []

    def adapt(self, artifact_root, end_round):
        self.calls.append((artifact_root, end_round))
        if self.failure is not None:
            raise self.failure
        return self.result


class Eligibility:
    def __init__(self, *, result=None, failure=None):
        self.result = result
        self.failure = failure
        self.calls = []

    def evaluate(self, assessment):
        self.calls.append(assessment)
        if self.failure is not None:
            raise self.failure
        return self.result


def adapter(source, eligibility):
    return DurableReplayResultArtifactPromotionEligibilitySourceAdapter(
        source_adapter=source,
        eligibility_service=eligibility,
    )


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
def test_artifact_root_identity_is_forwarded_exactly(artifact_root) -> None:
    source = Source(result=object())
    eligibility = Eligibility(result=object())

    adapter(source, eligibility).adapt(artifact_root, 1234)

    forwarded_root, _ = source.calls[0]
    assert forwarded_root is artifact_root


@pytest.mark.parametrize(
    "end_round",
    [-1, 0, 1, 2, 9, 999, 1234, 999999],
)
def test_end_round_identity_is_forwarded_without_adapter_validation(
    end_round: int,
) -> None:
    source = Source(result=object())
    eligibility = Eligibility(result=object())

    adapter(source, eligibility).adapt("artifact-root", end_round)

    _, forwarded_round = source.calls[0]
    assert forwarded_round == end_round


def test_assessment_identity_is_forwarded_exactly() -> None:
    assessment = object()
    source = Source(result=assessment)
    eligibility = Eligibility(result=object())

    adapter(source, eligibility).adapt("artifact-root", 1234)

    assert eligibility.calls == [assessment]
    assert eligibility.calls[0] is assessment


def test_eligibility_return_identity_is_preserved() -> None:
    result = object()
    source = Source(result=object())
    eligibility = Eligibility(result=result)

    returned = adapter(source, eligibility).adapt("artifact-root", 1234)

    assert returned is result


@pytest.mark.parametrize("owner", ["source", "eligibility"])
def test_dependency_failures_propagate_by_identity(owner: str) -> None:
    failure = RuntimeError(owner)

    if owner == "source":
        source = Source(failure=failure)
        eligibility = Eligibility(result=object())
    else:
        source = Source(result=object())
        eligibility = Eligibility(failure=failure)

    with pytest.raises(RuntimeError) as captured:
        adapter(source, eligibility).adapt("artifact-root", 1234)

    assert captured.value is failure


def test_adapter_does_not_catch_or_translate_failures() -> None:
    tree = ast.parse(PRODUCT_PATH.read_text(encoding="utf-8-sig"))

    assert not any(isinstance(node, ast.Try) for node in ast.walk(tree))
    assert not any(isinstance(node, ast.Raise) for node in ast.walk(tree))


def test_adapter_constructor_has_exact_two_optional_dependencies() -> None:
    signature = inspect.signature(
        DurableReplayResultArtifactPromotionEligibilitySourceAdapter
    )
    assert list(signature.parameters) == [
        "source_adapter",
        "eligibility_service",
    ]
    assert all(
        parameter.default is None
        for parameter in signature.parameters.values()
    )


def test_adapter_owns_default_dependencies_once_each() -> None:
    tree = ast.parse(PRODUCT_PATH.read_text(encoding="utf-8-sig"))

    calls = [
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]

    assert calls.count(
        "DurableReplayResultArtifactComparisonSourceAdapter"
    ) == 1
    assert calls.count(
        "DurableReplayResultPromotionEligibilityService"
    ) == 1


def test_adapter_has_exact_two_operational_imports() -> None:
    tree = ast.parse(PRODUCT_PATH.read_text(encoding="utf-8-sig"))

    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module not in {"__future__", "pathlib"}
    }

    assert modules == {
        (
            "lrp.operations."
            "durable_replay_result_artifact_comparison_source_adapter"
        ),
        "lrp.operations.durable_replay_result_promotion_eligibility",
    }


def test_adapter_has_no_direct_lower_layer_comparison_dependency() -> None:
    text = PRODUCT_PATH.read_text(encoding="utf-8-sig")
    forbidden = [
        "durable_replay_result_artifact_consumer",
        "durable_replay_result_artifact_inspection",
        "durable_replay_result_comparison_summary",
        "durable_replay_result_comparison_assessment",
    ]
    assert all(token not in text for token in forbidden)


def test_adapter_has_no_local_eligibility_reconstruction() -> None:
    text = PRODUCT_PATH.read_text(encoding="utf-8-sig")
    forbidden = [
        "DurableReplayResultPromotionEligibility(",
        "eligible=",
        "recommendation=",
        "reason=",
        "policy",
    ]
    assert all(token not in text for token in forbidden)


def test_adapter_has_no_action_plan_or_publication_logic() -> None:
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


def test_adapter_has_no_artifact_file_io_surface() -> None:
    text = PRODUCT_PATH.read_text(encoding="utf-8-sig")
    forbidden = [
        "json.loads",
        "json.dumps",
        "open(",
        "read_text(",
        "read_bytes(",
        "write_text(",
        "write_bytes(",
        "verify_manifest",
        "manifest.json",
        "evaluation_result.json",
    ]
    assert all(token not in text for token in forbidden)


def test_adapter_has_no_path_discovery_normalization_or_expansion() -> None:
    text = PRODUCT_PATH.read_text(encoding="utf-8-sig")
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
    assert all(token not in text for token in forbidden)


def test_adapter_has_no_cli_or_stream_surface() -> None:
    text = PRODUCT_PATH.read_text(encoding="utf-8-sig")
    forbidden = [
        "argparse",
        "lrp.cli",
        "stdin",
        "stdout",
        "sys.stdin",
        "sys.stdout",
    ]
    assert all(token not in text for token in forbidden)


def test_adapt_call_graph_is_exact_and_ordered() -> None:
    tree = ast.parse(PRODUCT_PATH.read_text(encoding="utf-8-sig"))

    adapt = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        for node in node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "adapt"
    )

    calls = [
        ast.unparse(node.value.func)
        for node in adapt.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
    ]
    returns = [
        ast.unparse(node.value.func)
        for node in adapt.body
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Call)
    ]

    assert calls == ["self._source_adapter.adapt"]
    assert returns == ["self._eligibility_service.evaluate"]


def test_adapt_has_no_local_input_rewrite_assignments() -> None:
    tree = ast.parse(PRODUCT_PATH.read_text(encoding="utf-8-sig"))

    adapt = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        for node in node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "adapt"
    )

    assigned_names = {
        target.id
        for node in ast.walk(adapt)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    assert "artifact_root" not in assigned_names
    assert "end_round" not in assigned_names


def test_adapter_declares_only_one_product_class() -> None:
    tree = ast.parse(PRODUCT_PATH.read_text(encoding="utf-8-sig"))
    classes = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    ]
    assert classes == [
        "DurableReplayResultArtifactPromotionEligibilitySourceAdapter"
    ]
