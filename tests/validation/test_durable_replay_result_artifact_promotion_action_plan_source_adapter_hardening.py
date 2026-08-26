from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from lrp.operations.durable_replay_result_artifact_promotion_action_plan_source_adapter import (
    DurableReplayResultArtifactPromotionActionPlanSourceAdapter,
)

PRODUCT_PATH = Path(
    "lrp/operations/"
    "durable_replay_result_artifact_promotion_action_plan_source_adapter.py"
)


def _source_text() -> str:
    return PRODUCT_PATH.read_text(encoding="utf-8-sig")


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


class _Planner:
    def __init__(self, *, result=None, failure=None):
        self.result = result
        self.failure = failure
        self.calls = []

    def plan(self, eligibility):
        self.calls.append(eligibility)
        if self.failure is not None:
            raise self.failure
        return self.result


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
def test_artifact_root_identity_is_forwarded_exactly(artifact_root) -> None:
    eligibility = object()
    source = _Source(result=eligibility)
    planner = _Planner(result=object())

    adapter = DurableReplayResultArtifactPromotionActionPlanSourceAdapter(
        source_adapter=source,
        action_plan_service=planner,
    )
    adapter.adapt(artifact_root, 1234)

    assert source.calls[0][0] is artifact_root


@pytest.mark.parametrize(
    "end_round",
    [-1, 0, 1, 2, 9, 999, 1234, 999999],
)
def test_end_round_identity_is_forwarded_without_adapter_validation(
    end_round: int,
) -> None:
    source = _Source(result=object())
    planner = _Planner(result=object())

    adapter = DurableReplayResultArtifactPromotionActionPlanSourceAdapter(
        source_adapter=source,
        action_plan_service=planner,
    )
    adapter.adapt("root", end_round)

    assert source.calls == [("root", end_round)]


def test_eligibility_identity_is_forwarded_exactly() -> None:
    eligibility = object()
    source = _Source(result=eligibility)
    planner = _Planner(result=object())

    adapter = DurableReplayResultArtifactPromotionActionPlanSourceAdapter(
        source_adapter=source,
        action_plan_service=planner,
    )
    adapter.adapt("root", 1234)

    assert planner.calls == [eligibility]
    assert planner.calls[0] is eligibility


def test_action_plan_return_identity_is_preserved() -> None:
    eligibility = object()
    action_plan = object()
    source = _Source(result=eligibility)
    planner = _Planner(result=action_plan)

    adapter = DurableReplayResultArtifactPromotionActionPlanSourceAdapter(
        source_adapter=source,
        action_plan_service=planner,
    )

    assert adapter.adapt("root", 1234) is action_plan


@pytest.mark.parametrize("owner", ["source", "action_plan"])
def test_dependency_failures_propagate_by_identity(owner: str) -> None:
    failure = RuntimeError(owner)

    if owner == "source":
        source = _Source(failure=failure)
        planner = _Planner(result=object())
    else:
        source = _Source(result=object())
        planner = _Planner(failure=failure)

    adapter = DurableReplayResultArtifactPromotionActionPlanSourceAdapter(
        source_adapter=source,
        action_plan_service=planner,
    )

    with pytest.raises(RuntimeError) as exc_info:
        adapter.adapt("root", 1234)

    assert exc_info.value is failure


def test_adapter_does_not_catch_or_translate_failures() -> None:
    tree = ast.parse(_source_text())
    forbidden = (
        ast.Try,
        ast.Raise,
    )
    assert not any(isinstance(node, forbidden) for node in ast.walk(tree))


def test_adapter_constructor_has_exact_two_optional_dependencies() -> None:
    sig = inspect.signature(
        DurableReplayResultArtifactPromotionActionPlanSourceAdapter
    )
    params = list(sig.parameters.values())

    assert [p.name for p in params] == [
        "source_adapter",
        "action_plan_service",
    ]
    assert all(p.default is None for p in params)


def test_adapter_owns_default_dependencies_once_each() -> None:
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


def test_adapter_has_exact_two_operational_imports() -> None:
    tree = ast.parse(_source_text())

    operational_imports = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("lrp.operations.")
    }

    assert operational_imports == {
        (
            "lrp.operations."
            "durable_replay_result_artifact_promotion_eligibility_source_adapter"
        ),
        "lrp.operations.durable_replay_result_promotion_action_plan",
    }


def test_adapter_has_no_direct_lower_layer_dependency() -> None:
    text = _source_text()
    forbidden = [
        "durable_replay_result_artifact_consumer",
        "durable_replay_result_artifact_inspection",
        "durable_replay_result_comparison_summary",
        "durable_replay_result_comparison_assessment",
        "durable_replay_result_promotion_eligibility.py",
    ]
    for token in forbidden:
        assert token not in text


def test_adapter_has_no_local_action_plan_reconstruction() -> None:
    text = _source_text()
    forbidden = [
        "DurableReplayResultPromotionActionPlan(",
        "recommendation=",
        "action=",
        "eligibility.status",
        "eligibility.recommendation",
    ]
    for token in forbidden:
        assert token not in text


def test_adapter_has_no_publication_request_or_execution_logic() -> None:
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


def test_adapter_has_no_artifact_file_io_surface() -> None:
    text = _source_text()
    forbidden = [
        "json.loads",
        "json.dumps",
        "open(",
        "read_text(",
        "read_bytes(",
        "write_text(",
        "write_bytes(",
        "manifest.json",
        "evaluation_result.json",
    ]
    for token in forbidden:
        assert token not in text


def test_adapter_has_no_path_discovery_normalization_or_expansion() -> None:
    text = _source_text()
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
        assert token not in text


def test_adapter_has_no_cli_or_stream_surface() -> None:
    text = _source_text()
    forbidden = [
        "argparse",
        "lrp.cli",
        "stdin",
        "stdout",
    ]
    for token in forbidden:
        assert token not in text


def test_adapt_call_graph_is_exact_and_ordered() -> None:
    tree = ast.parse(_source_text())

    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name
        == "DurableReplayResultArtifactPromotionActionPlanSourceAdapter"
    )
    adapt = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "adapt"
    )

    calls = [
        ast.unparse(node.func)
        for node in ast.walk(adapt)
        if isinstance(node, ast.Call)
    ]

    assert calls == [
        "self._source_adapter.adapt",
        "self._action_plan_service.plan",
    ]


def test_adapt_has_no_local_input_rewrite_assignments() -> None:
    tree = ast.parse(_source_text())

    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name
        == "DurableReplayResultArtifactPromotionActionPlanSourceAdapter"
    )
    adapt = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "adapt"
    )

    assigned_names = {
        target.id
        for node in ast.walk(adapt)
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


def test_adapter_declares_only_one_product_class() -> None:
    tree = ast.parse(_source_text())
    classes = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    ]
    assert classes == [
        "DurableReplayResultArtifactPromotionActionPlanSourceAdapter"
    ]
