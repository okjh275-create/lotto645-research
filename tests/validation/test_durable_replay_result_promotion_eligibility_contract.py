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

from lrp.operations.durable_replay_result_comparison_assessment import (
    DurableReplayResultComparisonAssessment,
)

MODULE_NAME = "lrp.operations.durable_replay_result_promotion_eligibility"
PRODUCT_PATH = Path("lrp/operations/durable_replay_result_promotion_eligibility.py")

EXPECTED_FIELDS = (
    "status",
    "round_count",
    "candidate_model_name",
    "baseline_model_name",
    "recommendation",
    "candidate_advantage_count",
    "neutral_count",
    "baseline_advantage_count",
    "window",
)


def _assessment(
    candidate: int,
    neutral: int,
    baseline: int,
) -> DurableReplayResultComparisonAssessment:
    labels = (
        ["candidate_advantage"] * candidate
        + ["neutral"] * neutral
        + ["baseline_advantage"] * baseline
    )
    assert len(labels) == 9
    return DurableReplayResultComparisonAssessment(
        status="PASS",
        round_count=1,
        candidate_model_name="candidate-model",
        baseline_model_name="baseline-model",
        top3_baseline_delta_mean_best_hits_assessment=labels[0],
        top5_baseline_delta_mean_best_hits_assessment=labels[1],
        top10_baseline_delta_mean_best_hits_assessment=labels[2],
        top3_baseline_delta_3plus_rate_assessment=labels[3],
        top5_baseline_delta_3plus_rate_assessment=labels[4],
        top10_baseline_delta_3plus_rate_assessment=labels[5],
        top3_baseline_delta_4plus_rate_assessment=labels[6],
        top5_baseline_delta_4plus_rate_assessment=labels[7],
        top10_baseline_delta_4plus_rate_assessment=labels[8],
        candidate_advantage_count=candidate,
        neutral_count=neutral,
        baseline_advantage_count=baseline,
        window=MappingProxyType(
            {"name": "test-window", "start_round": 1231, "end_round": 1231}
        ),
    )


def _module():
    return importlib.import_module(MODULE_NAME)


def test_promotion_eligibility_product_module_exists() -> None:
    assert importlib.util.find_spec(MODULE_NAME) is not None


def test_eligibility_result_is_frozen_dataclass() -> None:
    module = _module()
    result_type = module.DurableReplayResultPromotionEligibility
    assert dataclasses.is_dataclass(result_type)
    assert result_type.__dataclass_params__.frozen is True


def test_eligibility_result_fields_are_exact() -> None:
    module = _module()
    result_type = module.DurableReplayResultPromotionEligibility
    assert tuple(field.name for field in dataclasses.fields(result_type)) == EXPECTED_FIELDS


def test_eligibility_service_class_exists() -> None:
    module = _module()
    assert inspect.isclass(module.DurableReplayResultPromotionEligibilityService)


def test_eligibility_public_method_is_evaluate() -> None:
    module = _module()
    service = module.DurableReplayResultPromotionEligibilityService
    public = [
        name
        for name, value in service.__dict__.items()
        if callable(value) and not name.startswith("_")
    ]
    assert public == ["evaluate"]


def test_eligibility_method_accepts_ax_assessment_directly() -> None:
    module = _module()
    method = module.DurableReplayResultPromotionEligibilityService.evaluate
    hints = get_type_hints(method)
    assert hints["assessment"] is DurableReplayResultComparisonAssessment


def test_eligibility_return_annotation_is_exact_result_type() -> None:
    module = _module()
    method = module.DurableReplayResultPromotionEligibilityService.evaluate
    hints = get_type_hints(method)
    assert hints["return"] is module.DurableReplayResultPromotionEligibility


def test_eligibility_projects_identity_fields_exactly() -> None:
    module = _module()
    source = _assessment(2, 7, 0)
    result = module.DurableReplayResultPromotionEligibilityService().evaluate(source)
    assert result.status == source.status
    assert result.round_count == source.round_count
    assert result.candidate_model_name == source.candidate_model_name
    assert result.baseline_model_name == source.baseline_model_name
    assert result.candidate_advantage_count == source.candidate_advantage_count
    assert result.neutral_count == source.neutral_count
    assert result.baseline_advantage_count == source.baseline_advantage_count


@pytest.mark.parametrize(
    ("candidate", "neutral", "baseline", "expected"),
    [
        (2, 7, 0, "eligible"),
        (3, 5, 1, "eligible"),
        (1, 8, 0, "insufficient_evidence"),
        (1, 7, 1, "insufficient_evidence"),
        (0, 9, 0, "insufficient_evidence"),
        (2, 5, 2, "insufficient_evidence"),
        (0, 8, 1, "ineligible"),
        (1, 6, 2, "ineligible"),
    ],
)
def test_eligibility_recommendation_uses_exact_frozen_count_policy(
    candidate: int,
    neutral: int,
    baseline: int,
    expected: str,
) -> None:
    module = _module()
    result = module.DurableReplayResultPromotionEligibilityService().evaluate(
        _assessment(candidate, neutral, baseline)
    )
    assert result.recommendation == expected


def test_eligibility_labels_are_literal_compatible() -> None:
    module = _module()
    annotation = get_type_hints(module.DurableReplayResultPromotionEligibility)["recommendation"]
    assert set(annotation.__args__) == {
        "eligible",
        "insufficient_evidence",
        "ineligible",
    }


def test_eligibility_window_is_mapping_compatible() -> None:
    module = _module()
    result = module.DurableReplayResultPromotionEligibilityService().evaluate(
        _assessment(2, 7, 0)
    )
    assert isinstance(result.window, Mapping)


def test_eligibility_window_is_read_only_projection() -> None:
    module = _module()
    source = _assessment(2, 7, 0)
    result = module.DurableReplayResultPromotionEligibilityService().evaluate(source)
    assert isinstance(result.window, MappingProxyType)
    assert dict(result.window) == dict(source.window)
    assert result.window is not source.window
    with pytest.raises(TypeError):
        result.window["end_round"] = 9999  # type: ignore[index]


def test_eligibility_has_no_winner_or_production_mutation_fields() -> None:
    module = _module()
    fields = {
        field.name
        for field in dataclasses.fields(module.DurableReplayResultPromotionEligibility)
    }
    forbidden = {
        "winner",
        "winner_model",
        "promoted",
        "promoted_model",
        "publish",
        "rollback",
        "champion",
        "production_state",
    }
    assert fields.isdisjoint(forbidden)


def test_eligibility_has_no_runtime_io_or_production_dependency() -> None:
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


def test_eligibility_depends_on_ax_assessment_owner() -> None:
    source = PRODUCT_PATH.read_text(encoding="utf-8-sig")
    assert "durable_replay_result_comparison_assessment" in source
    assert "DurableReplayResultComparisonAssessment" in source


def test_eligibility_declares_no_discovery_raw_delta_or_mutation_surface() -> None:
    source = PRODUCT_PATH.read_text(encoding="utf-8-sig").lower()
    forbidden = (
        "discover",
        "latest",
        "cross_round",
        "cross-round",
        "baseline_delta_mean_best_hits",
        "baseline_delta_3plus_rate",
        "baseline_delta_4plus_rate",
        "publish_champion",
        "rollback_champion",
        "evaluate_champion_promotion",
    )
    assert all(token not in source for token in forbidden)