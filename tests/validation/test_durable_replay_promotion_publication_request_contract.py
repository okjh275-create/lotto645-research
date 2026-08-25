from __future__ import annotations

import ast
import dataclasses
import importlib
import importlib.util
import inspect
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, get_type_hints

import pytest

from lrp.operations.durable_replay_result_promotion_action_plan import (
    DurableReplayResultPromotionActionPlan,
)

MODULE_NAME = "lrp.operations.durable_replay_promotion_publication_request"
PRODUCT_PATH = Path(
    "lrp/operations/durable_replay_promotion_publication_request.py"
)

EXPECTED_FIELDS = (
    "status",
    "round_count",
    "candidate_model_name",
    "baseline_model_name",
    "recommendation",
    "action",
    "window",
    "source_decision",
    "registry_root",
)


def _plan(
    action: str = "prepare_publish",
) -> DurableReplayResultPromotionActionPlan:
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
        window=MappingProxyType(
            {
                "name": "test-window",
                "start_round": 1231,
                "end_round": 1231,
            }
        ),
    )


def _module():
    return importlib.import_module(MODULE_NAME)


def test_publication_request_product_module_exists() -> None:
    assert importlib.util.find_spec(MODULE_NAME) is not None


def test_publication_request_result_is_frozen_dataclass() -> None:
    module = _module()
    result_type = module.DurableReplayPromotionPublicationRequest
    assert dataclasses.is_dataclass(result_type)
    assert result_type.__dataclass_params__.frozen is True


def test_publication_request_result_fields_are_exact() -> None:
    module = _module()
    result_type = module.DurableReplayPromotionPublicationRequest
    assert tuple(
        field.name for field in dataclasses.fields(result_type)
    ) == EXPECTED_FIELDS


def test_publication_request_service_class_exists() -> None:
    module = _module()
    assert inspect.isclass(
        module.DurableReplayPromotionPublicationRequestService
    )


def test_publication_request_public_method_is_build() -> None:
    module = _module()
    service = module.DurableReplayPromotionPublicationRequestService
    public = [
        name
        for name, value in service.__dict__.items()
        if callable(value) and not name.startswith("_")
    ]
    assert public == ["build"]


def test_publication_request_method_accepts_exact_inputs() -> None:
    module = _module()
    method = module.DurableReplayPromotionPublicationRequestService.build
    signature = inspect.signature(method)
    params = list(signature.parameters.values())

    assert [param.name for param in params] == [
        "self",
        "action_plan",
        "source_decision",
        "registry_root",
    ]
    assert params[1].kind is inspect.Parameter.KEYWORD_ONLY
    assert params[2].kind is inspect.Parameter.KEYWORD_ONLY
    assert params[3].kind is inspect.Parameter.KEYWORD_ONLY

    hints = get_type_hints(method)
    assert hints["action_plan"] is DurableReplayResultPromotionActionPlan


def test_publication_request_return_annotation_is_exact_result_type() -> None:
    module = _module()
    method = module.DurableReplayPromotionPublicationRequestService.build
    hints = get_type_hints(method)
    assert hints["return"] is module.DurableReplayPromotionPublicationRequest


def test_publication_request_projects_action_plan_fields_exactly() -> None:
    module = _module()
    source = _plan("prepare_publish")
    result = module.DurableReplayPromotionPublicationRequestService().build(
        action_plan=source,
        source_decision="artifacts/decision.json",
        registry_root="production/registry",
    )

    assert result.status == source.status
    assert result.round_count == source.round_count
    assert result.candidate_model_name == source.candidate_model_name
    assert result.baseline_model_name == source.baseline_model_name
    assert result.recommendation == source.recommendation
    assert result.action == source.action


def test_publication_request_preserves_explicit_publication_identity() -> None:
    module = _module()
    result = module.DurableReplayPromotionPublicationRequestService().build(
        action_plan=_plan(),
        source_decision="artifacts/decision.json",
        registry_root="production/registry",
    )

    assert str(result.source_decision) == "artifacts/decision.json"
    assert str(result.registry_root) == "production/registry"


@pytest.mark.parametrize("action", ["hold", "block"])
def test_publication_request_requires_prepare_publish(action: str) -> None:
    module = _module()
    with pytest.raises((TypeError, ValueError)):
        module.DurableReplayPromotionPublicationRequestService().build(
            action_plan=_plan(action),
            source_decision="artifacts/decision.json",
            registry_root="production/registry",
        )


def test_publication_request_window_is_mapping_compatible() -> None:
    module = _module()
    result = module.DurableReplayPromotionPublicationRequestService().build(
        action_plan=_plan(),
        source_decision="artifacts/decision.json",
        registry_root="production/registry",
    )
    assert isinstance(result.window, Mapping)


def test_publication_request_window_is_read_only_projection() -> None:
    module = _module()
    source = _plan()
    result = module.DurableReplayPromotionPublicationRequestService().build(
        action_plan=source,
        source_decision="artifacts/decision.json",
        registry_root="production/registry",
    )

    assert isinstance(result.window, MappingProxyType)
    assert result.window is not source.window
    assert dict(result.window) == dict(source.window)

    with pytest.raises(TypeError):
        result.window["end_round"] = 9999  # type: ignore[index]


def test_publication_request_has_no_execution_or_result_fields() -> None:
    module = _module()
    fields = {
        field.name
        for field in dataclasses.fields(
            module.DurableReplayPromotionPublicationRequest
        )
    }
    forbidden = {
        "published",
        "publication_result",
        "registry_written",
        "champion_id",
        "rollback_result",
    }
    assert fields.isdisjoint(forbidden)


def test_publication_request_has_no_runtime_io_or_production_dependency() -> None:
    tree = ast.parse(PRODUCT_PATH.read_text(encoding="utf-8-sig"))
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    forbidden_prefixes = (
        "os",
        "sqlite3",
        "random",
        "time",
        "datetime",
        "lrp.production",
        "lrp.cli",
    )

    assert not any(
        imported == prefix or imported.startswith(prefix + ".")
        for imported in imports
        for prefix in forbidden_prefixes
    )


def test_publication_request_depends_on_az_action_plan_owner() -> None:
    source = PRODUCT_PATH.read_text(encoding="utf-8-sig")
    assert "durable_replay_result_promotion_action_plan" in source
    assert "DurableReplayResultPromotionActionPlan" in source


def test_publication_request_declares_no_discovery_or_mutation_surface() -> None:
    source = PRODUCT_PATH.read_text(encoding="utf-8-sig").lower()
    forbidden = (
        "discover",
        "latest",
        "cross_round",
        "cross-round",
        "publish(",
        "championregistrypublisher",
        "write_operation_artifact",
        "production_lifecycle",
        "default_registry",
        "environ",
        "getenv",
    )
    assert all(token not in source for token in forbidden)