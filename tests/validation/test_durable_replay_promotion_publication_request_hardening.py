from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType

import pytest

from lrp.operations.durable_replay_promotion_publication_request import (
    DurableReplayPromotionPublicationRequestService,
)
from lrp.operations.durable_replay_result_promotion_action_plan import (
    DurableReplayResultPromotionActionPlan,
)


def _plan(
    *,
    action: str = "prepare_publish",
    window: object | None = None,
) -> DurableReplayResultPromotionActionPlan:
    if window is None:
        window = {
            "name": "hardening-window",
            "start_round": 1231,
            "end_round": 1231,
        }

    recommendation = {
        "prepare_publish": "eligible",
        "hold": "insufficient_evidence",
        "block": "ineligible",
    }.get(action, "eligible")

    return DurableReplayResultPromotionActionPlan(
        status="PASS",
        round_count=1,
        candidate_model_name="candidate-model",
        baseline_model_name="baseline-model",
        recommendation=recommendation,
        action=action,  # type: ignore[arg-type]
        window=window,  # type: ignore[arg-type]
    )


def test_valid_publication_request_round_trip() -> None:
    result = DurableReplayPromotionPublicationRequestService().build(
        action_plan=_plan(),
        source_decision="artifacts/decision.json",
        registry_root="production/registry",
    )
    assert result.action == "prepare_publish"
    assert str(result.source_decision) == "artifacts/decision.json"
    assert str(result.registry_root) == "production/registry"


@pytest.mark.parametrize("action", ["hold", "block", "unknown"])
def test_non_prepare_publish_action_fails(action: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        DurableReplayPromotionPublicationRequestService().build(
            action_plan=_plan(action=action),
            source_decision="artifacts/decision.json",
            registry_root="production/registry",
        )


@pytest.mark.parametrize(
    "source_decision",
    ["", "   ", Path("")],
)
def test_empty_source_decision_fails(source_decision: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        DurableReplayPromotionPublicationRequestService().build(
            action_plan=_plan(),
            source_decision=source_decision,  # type: ignore[arg-type]
            registry_root="production/registry",
        )


@pytest.mark.parametrize(
    "registry_root",
    ["", "   ", Path("")],
)
def test_empty_registry_root_fails(registry_root: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        DurableReplayPromotionPublicationRequestService().build(
            action_plan=_plan(),
            source_decision="artifacts/decision.json",
            registry_root=registry_root,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("source_decision", None),
        ("source_decision", 123),
        ("source_decision", []),
        ("registry_root", None),
        ("registry_root", 123),
        ("registry_root", []),
    ],
)
def test_invalid_publication_identity_type_fails(
    field_name: str,
    value: object,
) -> None:
    kwargs = {
        "action_plan": _plan(),
        "source_decision": "artifacts/decision.json",
        "registry_root": "production/registry",
    }
    kwargs[field_name] = value

    with pytest.raises((TypeError, ValueError)):
        DurableReplayPromotionPublicationRequestService().build(**kwargs)


def test_non_mapping_window_fails() -> None:
    with pytest.raises(TypeError):
        DurableReplayPromotionPublicationRequestService().build(
            action_plan=_plan(window=[]),
            source_decision="artifacts/decision.json",
            registry_root="production/registry",
        )


def test_result_is_immutable() -> None:
    result = DurableReplayPromotionPublicationRequestService().build(
        action_plan=_plan(),
        source_decision="artifacts/decision.json",
        registry_root="production/registry",
    )
    with pytest.raises(FrozenInstanceError):
        result.registry_root = "other"  # type: ignore[misc]


def test_window_is_read_only_and_detached() -> None:
    source_window = {
        "name": "hardening-window",
        "start_round": 1231,
        "end_round": 1231,
    }
    plan = _plan(window=source_window)

    result = DurableReplayPromotionPublicationRequestService().build(
        action_plan=plan,
        source_decision="artifacts/decision.json",
        registry_root="production/registry",
    )

    assert isinstance(result.window, MappingProxyType)
    assert result.window is not plan.window

    source_window["name"] = "changed-after-build"
    assert result.window["name"] == "hardening-window"

    with pytest.raises(TypeError):
        result.window["name"] = "mutated"  # type: ignore[index]


def test_request_is_deterministic_for_same_input() -> None:
    plan = _plan()
    service = DurableReplayPromotionPublicationRequestService()

    first = service.build(
        action_plan=plan,
        source_decision="artifacts/decision.json",
        registry_root="production/registry",
    )
    second = service.build(
        action_plan=plan,
        source_decision="artifacts/decision.json",
        registry_root="production/registry",
    )

    assert first == second


def test_build_does_not_mutate_action_plan() -> None:
    plan = _plan()
    before = (
        plan.status,
        plan.round_count,
        plan.candidate_model_name,
        plan.baseline_model_name,
        plan.recommendation,
        plan.action,
        dict(plan.window),
    )

    DurableReplayPromotionPublicationRequestService().build(
        action_plan=plan,
        source_decision="artifacts/decision.json",
        registry_root="production/registry",
    )

    after = (
        plan.status,
        plan.round_count,
        plan.candidate_model_name,
        plan.baseline_model_name,
        plan.recommendation,
        plan.action,
        dict(plan.window),
    )

    assert after == before