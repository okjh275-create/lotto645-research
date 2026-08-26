from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import lrp.operations.durable_replay_result_artifact_promotion_publication_request_source_adapter as product
from lrp.operations.durable_replay_promotion_publication_request import (
    DurableReplayPromotionPublicationRequestService,
)
from lrp.operations.durable_replay_result_promotion_action_plan import (
    DurableReplayResultPromotionActionPlan,
)


class _CapturingSource:
    def __init__(self, action_plan: object) -> None:
        self.action_plan = action_plan
        self.calls: list[tuple[object, object]] = []

    def adapt(self, artifact_root: object, end_round: object) -> object:
        self.calls.append((artifact_root, end_round))
        return self.action_plan


class _CapturingService:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def build(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return self.result


@pytest.mark.parametrize(
    ("artifact_root", "end_round"),
    [
        ("  artifact root  ", -7),
        (Path("a/../b"), 0),
        ("", 999999999),
    ],
)
def test_unusual_upstream_inputs_are_forwarded_unchanged(
    artifact_root: object,
    end_round: int,
) -> None:
    action_plan = object()
    result = object()

    source = _CapturingSource(action_plan)
    service = _CapturingService(result)

    adapter = (
        product
        .DurableReplayResultArtifactPromotionPublicationRequestSourceAdapter(
            source_adapter=source,
            publication_request_service=service,
        )
    )

    actual = adapter.adapt(
        artifact_root,
        end_round,
        source_decision="decision",
        registry_root="registry",
    )

    assert actual is result
    assert source.calls == [(artifact_root, end_round)]


@pytest.mark.parametrize(
    ("source_decision", "registry_root"),
    [
        (" decision ", " registry "),
        (Path("decision/../decision.json"), Path("registry/../registry")),
    ],
)
def test_publication_identity_values_are_forwarded_by_identity(
    source_decision: object,
    registry_root: object,
) -> None:
    action_plan = object()
    result = object()

    source = _CapturingSource(action_plan)
    service = _CapturingService(result)

    adapter = (
        product
        .DurableReplayResultArtifactPromotionPublicationRequestSourceAdapter(
            source_adapter=source,
            publication_request_service=service,
        )
    )

    actual = adapter.adapt(
        "artifact",
        1234,
        source_decision=source_decision,
        registry_root=registry_root,
    )

    assert actual is result

    call = service.calls[0]
    assert call["source_decision"] is source_decision
    assert call["registry_root"] is registry_root


@pytest.mark.parametrize(
    ("source_decision", "registry_root"),
    [
        ("", "registry"),
        ("decision", ""),
        (Path(""), Path("registry")),
        (Path("decision"), Path("")),
    ],
)
def test_invalid_publication_identity_is_delegated_downstream(
    source_decision: object,
    registry_root: object,
) -> None:
    action_plan = DurableReplayResultPromotionActionPlan(
        status="eligible",
        round_count=1,
        candidate_model_name="candidate",
        baseline_model_name="baseline",
        recommendation="promote",
        action="prepare_publish",
        window={"start_round": 1, "end_round": 1},
    )

    source = _CapturingSource(action_plan)

    adapter = (
        product
        .DurableReplayResultArtifactPromotionPublicationRequestSourceAdapter(
            source_adapter=source,
            publication_request_service=(
                DurableReplayPromotionPublicationRequestService()
            ),
        )
    )

    with pytest.raises(ValueError):
        adapter.adapt(
            "artifact",
            1,
            source_decision=source_decision,
            registry_root=registry_root,
        )

    assert source.calls == [("artifact", 1)]


@pytest.mark.parametrize(
    "action",
    [
        "hold",
        "block",
    ],
)
def test_non_publish_action_rejection_is_delegated_downstream(
    action: str,
) -> None:
    action_plan = DurableReplayResultPromotionActionPlan(
        status="ineligible",
        round_count=1,
        candidate_model_name="candidate",
        baseline_model_name="baseline",
        recommendation="hold",
        action=action,
        window={"start_round": 1, "end_round": 1},
    )

    source = _CapturingSource(action_plan)

    adapter = (
        product
        .DurableReplayResultArtifactPromotionPublicationRequestSourceAdapter(
            source_adapter=source,
            publication_request_service=(
                DurableReplayPromotionPublicationRequestService()
            ),
        )
    )

    with pytest.raises(ValueError) as exc_info:
        adapter.adapt(
            "artifact",
            1,
            source_decision="decision",
            registry_root="registry",
        )

    assert "prepare_publish" in str(exc_info.value)
    assert source.calls == [("artifact", 1)]


def test_product_has_no_exception_translation() -> None:
    tree = ast.parse(inspect.getsource(product))

    assert not any(
        isinstance(node, ast.Try)
        for node in ast.walk(tree)
    )

    assert not any(
        isinstance(node, ast.Raise)
        for node in ast.walk(tree)
    )


def test_product_has_no_input_rewriting_or_hidden_policy() -> None:
    source = inspect.getsource(product)

    forbidden = (
        ".strip(",
        ".resolve(",
        ".absolute(",
        ".expanduser(",
        "os.getenv",
        "os.environ",
        "default_registry",
        "load_latest",
        ".latest(",
        ".discover(",
    )

    for token in forbidden:
        assert token not in source


def test_product_call_graph_is_exact() -> None:
    tree = ast.parse(inspect.getsource(product))

    calls: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func

        if isinstance(func, ast.Name):
            calls.append(func.id)

        elif isinstance(func, ast.Attribute):
            calls.append(func.attr)

    assert sorted(calls) == sorted(
        [
            "DurableReplayResultArtifactPromotionActionPlanSourceAdapter",
            "DurableReplayPromotionPublicationRequestService",
            "adapt",
            "build",
        ]
    )


def test_composition_is_semantically_deterministic() -> None:
    action_plan = object()
    result = object()

    source = _CapturingSource(action_plan)
    service = _CapturingService(result)

    adapter = (
        product
        .DurableReplayResultArtifactPromotionPublicationRequestSourceAdapter(
            source_adapter=source,
            publication_request_service=service,
        )
    )

    first = adapter.adapt(
        "artifact",
        1234,
        source_decision="decision",
        registry_root="registry",
    )

    second = adapter.adapt(
        "artifact",
        1234,
        source_decision="decision",
        registry_root="registry",
    )

    assert first is result
    assert second is result

    assert source.calls == [
        ("artifact", 1234),
        ("artifact", 1234),
    ]

    assert service.calls == [
        {
            "action_plan": action_plan,
            "source_decision": "decision",
            "registry_root": "registry",
        },
        {
            "action_plan": action_plan,
            "source_decision": "decision",
            "registry_root": "registry",
        },
    ]
