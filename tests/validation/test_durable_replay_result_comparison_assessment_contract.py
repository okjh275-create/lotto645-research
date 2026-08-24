from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, get_args, get_origin, get_type_hints

import pytest

from lrp.operations.durable_replay_result_comparison_summary import (
    DurableReplayResultComparisonSummary,
)

MODULE_NAME = "lrp.operations.durable_replay_result_comparison_assessment"
PRODUCT_PATH = Path("lrp/operations/durable_replay_result_comparison_assessment.py")

EXPECTED_FIELDS = (
    "status",
    "round_count",
    "candidate_model_name",
    "baseline_model_name",
    "top3_baseline_delta_mean_best_hits_assessment",
    "top5_baseline_delta_mean_best_hits_assessment",
    "top10_baseline_delta_mean_best_hits_assessment",
    "top3_baseline_delta_3plus_rate_assessment",
    "top5_baseline_delta_3plus_rate_assessment",
    "top10_baseline_delta_3plus_rate_assessment",
    "top3_baseline_delta_4plus_rate_assessment",
    "top5_baseline_delta_4plus_rate_assessment",
    "top10_baseline_delta_4plus_rate_assessment",
    "candidate_advantage_count",
    "neutral_count",
    "baseline_advantage_count",
    "window",
)

ALLOWED_LABELS = {
    "candidate_advantage",
    "neutral",
    "baseline_advantage",
}


def _module():
    return importlib.import_module(MODULE_NAME)


def _summary() -> DurableReplayResultComparisonSummary:
    return DurableReplayResultComparisonSummary(
        status="PASS",
        round_count=1,
        candidate_model_name="candidate-model",
        baseline_model_name="baseline-model",
        top3_baseline_delta_mean_best_hits=1.0,
        top5_baseline_delta_mean_best_hits=0.0,
        top10_baseline_delta_mean_best_hits=-1.0,
        top3_baseline_delta_3plus_rate=0.5,
        top5_baseline_delta_3plus_rate=0.0,
        top10_baseline_delta_3plus_rate=-0.5,
        top3_baseline_delta_4plus_rate=0.25,
        top5_baseline_delta_4plus_rate=0.0,
        top10_baseline_delta_4plus_rate=-0.25,
        window=MappingProxyType(
            {
                "name": "sample",
                "start_round": 1231,
                "end_round": 1231,
            }
        ),
    )


def test_comparison_assessment_product_module_exists() -> None:
    assert importlib.util.find_spec(MODULE_NAME) is not None


def test_assessment_result_is_frozen_dataclass() -> None:
    module = _module()
    result_type = module.DurableReplayResultComparisonAssessment
    assert is_dataclass(result_type)
    assert result_type.__dataclass_params__.frozen is True


def test_assessment_result_fields_are_exact() -> None:
    module = _module()
    result_type = module.DurableReplayResultComparisonAssessment
    assert tuple(field.name for field in fields(result_type)) == EXPECTED_FIELDS


def test_assessment_service_class_exists() -> None:
    module = _module()
    assert hasattr(module, "DurableReplayResultComparisonAssessmentService")


def test_assessment_public_method_is_assess() -> None:
    module = _module()
    service_type = module.DurableReplayResultComparisonAssessmentService
    public = [
        name
        for name, value in inspect.getmembers(
            service_type,
            predicate=inspect.isfunction,
        )
        if not name.startswith("_")
    ]
    assert public == ["assess"]


def test_assessment_method_accepts_aw_summary_directly() -> None:
    module = _module()
    method = module.DurableReplayResultComparisonAssessmentService.assess
    signature = inspect.signature(method)
    parameters = list(signature.parameters.values())
    assert [parameter.name for parameter in parameters] == [
        "self",
        "summary",
    ]
    hints = get_type_hints(method)
    assert hints["summary"] is DurableReplayResultComparisonSummary


def test_assessment_return_annotation_is_exact_result_type() -> None:
    module = _module()
    method = module.DurableReplayResultComparisonAssessmentService.assess
    hints = get_type_hints(method)
    assert hints["return"] is module.DurableReplayResultComparisonAssessment


def test_assessment_projects_identity_fields_exactly() -> None:
    module = _module()
    result = module.DurableReplayResultComparisonAssessmentService().assess(
        _summary()
    )
    assert result.status == "PASS"
    assert result.round_count == 1
    assert result.candidate_model_name == "candidate-model"
    assert result.baseline_model_name == "baseline-model"


def test_assessment_classifies_positive_zero_negative_by_sign_only() -> None:
    module = _module()
    result = module.DurableReplayResultComparisonAssessmentService().assess(
        _summary()
    )
    assert result.top3_baseline_delta_mean_best_hits_assessment == "candidate_advantage"
    assert result.top5_baseline_delta_mean_best_hits_assessment == "neutral"
    assert result.top10_baseline_delta_mean_best_hits_assessment == "baseline_advantage"
    assert result.top3_baseline_delta_3plus_rate_assessment == "candidate_advantage"
    assert result.top5_baseline_delta_3plus_rate_assessment == "neutral"
    assert result.top10_baseline_delta_3plus_rate_assessment == "baseline_advantage"
    assert result.top3_baseline_delta_4plus_rate_assessment == "candidate_advantage"
    assert result.top5_baseline_delta_4plus_rate_assessment == "neutral"
    assert result.top10_baseline_delta_4plus_rate_assessment == "baseline_advantage"


def test_assessment_aggregate_counts_are_exact_and_sum_to_nine() -> None:
    module = _module()
    result = module.DurableReplayResultComparisonAssessmentService().assess(
        _summary()
    )
    assert result.candidate_advantage_count == 3
    assert result.neutral_count == 3
    assert result.baseline_advantage_count == 3
    assert (
        result.candidate_advantage_count
        + result.neutral_count
        + result.baseline_advantage_count
    ) == 9


def test_assessment_labels_are_literal_compatible() -> None:
    module = _module()
    hints = get_type_hints(module.DurableReplayResultComparisonAssessment)
    assessment_fields = [
        name
        for name in EXPECTED_FIELDS
        if name.endswith("_assessment")
    ]
    for name in assessment_fields:
        annotation = hints[name]
        args = set(get_args(annotation))
        if args:
            assert args == ALLOWED_LABELS


def test_assessment_window_is_mapping_compatible() -> None:
    module = _module()
    hints = get_type_hints(module.DurableReplayResultComparisonAssessment)
    annotation = hints["window"]
    origin = get_origin(annotation)
    assert (
        annotation == Mapping[str, object]
        or origin in (Mapping, dict)
        or str(origin) == "<class 'collections.abc.Mapping'>"
    )


def test_assessment_window_is_read_only_projection() -> None:
    module = _module()
    result = module.DurableReplayResultComparisonAssessmentService().assess(
        _summary()
    )
    assert isinstance(result.window, Mapping)
    with pytest.raises(TypeError):
        result.window["name"] = "mutated"  # type: ignore[index]


def test_assessment_has_no_winner_or_promotion_fields() -> None:
    module = _module()
    names = {
        field.name
        for field in fields(module.DurableReplayResultComparisonAssessment)
    }
    forbidden = {
        "winner",
        "winner_label",
        "recommendation",
        "promotion",
        "promote",
        "champion",
        "decision",
    }
    assert names.isdisjoint(forbidden)


def test_assessment_has_no_runtime_io_or_production_dependency() -> None:
    source = PRODUCT_PATH.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    forbidden_prefixes = (
        "pathlib",
        "sqlite3",
        "random",
        "time",
        "datetime",
        "os",
        "lrp.operations.runtime",
        "lrp.production",
        "lrp.cli",
        "lrp.evaluation.promotion",
    )
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in imports
        for prefix in forbidden_prefixes
    )


def test_assessment_depends_on_aw_summary_owner() -> None:
    source = PRODUCT_PATH.read_text(encoding="utf-8-sig")
    assert "lrp.operations.durable_replay_result_comparison_summary" in source


def test_assessment_declares_no_threshold_discovery_or_mutation_surface() -> None:
    source = PRODUCT_PATH.read_text(encoding="utf-8-sig").lower()
    forbidden_tokens = (
        "latest",
        "discover",
        "selector",
        "threshold",
        "epsilon",
        "write_operation_artifact",
        "publish_champion",
        "rollback_champion",
        "winner",
        "recommendation",
        "promotion",
    )
    for token in forbidden_tokens:
        assert token not in source
