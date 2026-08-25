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

from lrp.operations.durable_replay_result_promotion_eligibility import (
    DurableReplayResultPromotionEligibility,
)

MODULE_NAME = "lrp.operations.durable_replay_result_promotion_action_plan"
PRODUCT_PATH = Path(
    "lrp/operations/durable_replay_result_promotion_action_plan.py"
)

EXPECTED_FIELDS = (
    "status",
    "round_count",
    "candidate_model_name",
    "baseline_model_name",
    "recommendation",
    "action",
    "window",
)


def _eligibility(
    recommendation: str,
) -> DurableReplayResultPromotionEligibility:
    counts = {
        "eligible": (2, 7, 0),
        "insufficient_evidence": (1, 8, 0),
        "ineligible": (0, 8, 1),
    }
    candidate, neutral, baseline = counts[recommendation]
    return DurableReplayResultPromotionEligibility(
        status="PASS",
        round_count=1,
        candidate_model_name="candidate-model",
        baseline_model_name="baseline-model",
        recommendation=recommendation,  # type: ignore[arg-type]
        candidate_advantage_count=candidate,
        neutral_count=neutral,
        baseline_advantage_count=baseline,
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


def test_promotion_action_plan_product_module_exists() -> None:
    assert importlib.util.find_spec(MODULE_NAME) is not None


def test_action_plan_result_is_frozen_dataclass() -> None:
    module = _module()
    result_type = module.DurableReplayResultPromotionActionPlan
    assert dataclasses.is_dataclass(result_type)
    assert result_type.__dataclass_params__.frozen is True


def test_action_plan_result_fields_are_exact() -> None:
    module = _module()
    result_type = module.DurableReplayResultPromotionActionPlan
    assert tuple(
        field.name for field in dataclasses.fields(result_type)
    ) == EXPECTED_FIELDS


def test_action_plan_service_class_exists() -> None:
    module = _module()
    assert inspect.isclass(
        module.DurableReplayResultPromotionActionPlanService
    )


def test_action_plan_public_method_is_plan() -> None:
    module = _module()
    service = module.DurableReplayResultPromotionActionPlanService
    public = [
        name
        for name, value in service.__dict__.items()
        if callable(value) and not name.startswith("_")
    ]
    assert public == ["plan"]


def test_action_plan_method_accepts_ay_eligibility_directly() -> None:
    module = _module()
    method = module.DurableReplayResultPromotionActionPlanService.plan
    hints = get_type_hints(method)
    assert hints["eligibility"] is DurableReplayResultPromotionEligibility


def test_action_plan_return_annotation_is_exact_result_type() -> None:
    module = _module()
    method = module.DurableReplayResultPromotionActionPlanService.plan
    hints = get_type_hints(method)
    assert hints["return"] is module.DurableReplayResultPromotionActionPlan


def test_action_plan_projects_identity_fields_exactly() -> None:
    module = _module()
    source = _eligibility("eligible")
    result = module.DurableReplayResultPromotionActionPlanService().plan(source)

    assert result.status == source.status
    assert result.round_count == source.round_count
    assert result.candidate_model_name == source.candidate_model_name
    assert result.baseline_model_name == source.baseline_model_name
    assert result.recommendation == source.recommendation


@pytest.mark.parametrize(
    ("recommendation", "expected_action"),
    [
        ("eligible", "prepare_publish"),
        ("insufficient_evidence", "hold"),
        ("ineligible", "block"),
    ],
)
def test_action_plan_uses_exact_frozen_recommendation_mapping(
    recommendation: str,
    expected_action: str,
) -> None:
    module = _module()
    result = module.DurableReplayResultPromotionActionPlanService().plan(
        _eligibility(recommendation)
    )
    assert result.action == expected_action


def test_action_plan_labels_are_literal_compatible() -> None:
    module = _module()
    annotation = get_type_hints(
        module.DurableReplayResultPromotionActionPlan
    )["action"]
    assert set(annotation.__args__) == {
        "prepare_publish",
        "hold",
        "block",
    }


def test_action_plan_window_is_mapping_compatible() -> None:
    module = _module()
    result = module.DurableReplayResultPromotionActionPlanService().plan(
        _eligibility("eligible")
    )
    assert isinstance(result.window, Mapping)


def test_action_plan_window_is_read_only_projection() -> None:
    module = _module()
    source = _eligibility("eligible")
    result = module.DurableReplayResultPromotionActionPlanService().plan(source)

    assert isinstance(result.window, MappingProxyType)
    assert result.window is not source.window
    assert dict(result.window) == dict(source.window)

    with pytest.raises(TypeError):
        result.window["end_round"] = 9999  # type: ignore[index]


def test_action_plan_has_no_execution_or_registry_fields() -> None:
    module = _module()
    fields = {
        field.name
        for field in dataclasses.fields(
            module.DurableReplayResultPromotionActionPlan
        )
    }
    forbidden = {
        "published",
        "rolled_back",
        "champion_id",
        "registry_path",
        "production_state",
        "winner",
    }
    assert fields.isdisjoint(forbidden)


def test_action_plan_has_no_runtime_io_or_production_dependency() -> None:
    tree = ast.parse(PRODUCT_PATH.read_text(encoding="utf-8-sig"))
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    forbidden_prefixes = (
        "os",
        "pathlib",
        "sqlite3",
        "random",
        "time",
        "datetime",
        "lrp.production",
        "lrp.cli",
        "lrp.evaluation.promotion",
    )

    assert not any(
        imported == prefix or imported.startswith(prefix + ".")
        for imported in imports
        for prefix in forbidden_prefixes
    )


def test_action_plan_depends_on_ay_eligibility_owner() -> None:
    source = PRODUCT_PATH.read_text(encoding="utf-8-sig")
    assert "durable_replay_result_promotion_eligibility" in source
    assert "DurableReplayResultPromotionEligibility" in source


def test_action_plan_declares_no_discovery_persistence_or_mutation_surface() -> None:
    source = PRODUCT_PATH.read_text(encoding="utf-8-sig").lower()
    forbidden = (
        "discover",
        "latest",
        "cross_round",
        "cross-round",
        "write_operation_artifact",
        "publish_champion",
        "rollback_champion",
        "production_lifecycle",
        "champion_registry",
        "baseline_delta_mean_best_hits",
        "candidate_advantage_count",
        "neutral_count",
        "baseline_advantage_count",
    )
    assert all(token not in source for token in forbidden)