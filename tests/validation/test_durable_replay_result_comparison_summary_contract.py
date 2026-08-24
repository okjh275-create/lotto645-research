from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, get_type_hints

import pytest

from lrp.operations.durable_replay_result_artifact_inspection import (
    DurableReplayResultArtifactInspection,
)

MODULE_NAME = "lrp.operations.durable_replay_result_comparison_summary"
PRODUCT_PATH = Path("lrp/operations/durable_replay_result_comparison_summary.py")

EXPECTED_FIELDS = (
    "status",
    "round_count",
    "candidate_model_name",
    "baseline_model_name",
    "top3_baseline_delta_mean_best_hits",
    "top5_baseline_delta_mean_best_hits",
    "top10_baseline_delta_mean_best_hits",
    "top3_baseline_delta_3plus_rate",
    "top5_baseline_delta_3plus_rate",
    "top10_baseline_delta_3plus_rate",
    "top3_baseline_delta_4plus_rate",
    "top5_baseline_delta_4plus_rate",
    "top10_baseline_delta_4plus_rate",
    "window",
)


def _module():
    return importlib.import_module(MODULE_NAME)


def _inspection() -> DurableReplayResultArtifactInspection:
    evaluation = MappingProxyType(
        {
            "top3": {
                "baseline_delta_mean_best_hits": 0.10,
                "baseline_delta_3plus_rate": 0.20,
                "baseline_delta_4plus_rate": 0.30,
            },
            "top5": {
                "baseline_delta_mean_best_hits": 0.40,
                "baseline_delta_3plus_rate": 0.50,
                "baseline_delta_4plus_rate": 0.60,
            },
            "top10": {
                "baseline_delta_mean_best_hits": 0.70,
                "baseline_delta_3plus_rate": 0.80,
                "baseline_delta_4plus_rate": 0.90,
            },
            "window": {
                "name": "test-window",
                "start_round": 1200,
                "end_round": 1231,
            },
        }
    )
    return DurableReplayResultArtifactInspection(
        status="PASS",
        round_count=32,
        candidate_model_name="candidate-model",
        baseline_model_name="baseline-model",
        evaluation=evaluation,
    )


def test_comparison_summary_product_module_exists() -> None:
    assert importlib.util.find_spec(MODULE_NAME) is not None


def test_summary_result_is_frozen_dataclass() -> None:
    module = _module()
    result_type = module.DurableReplayResultComparisonSummary
    assert is_dataclass(result_type)
    assert result_type.__dataclass_params__.frozen is True


def test_summary_result_fields_are_exact() -> None:
    module = _module()
    result_type = module.DurableReplayResultComparisonSummary
    assert tuple(field.name for field in fields(result_type)) == EXPECTED_FIELDS


def test_summary_service_class_exists() -> None:
    module = _module()
    assert hasattr(module, "DurableReplayResultComparisonSummaryService")


def test_summary_public_method_is_summarize() -> None:
    module = _module()
    service_type = module.DurableReplayResultComparisonSummaryService
    public = [
        name
        for name, value in inspect.getmembers(service_type, predicate=inspect.isfunction)
        if not name.startswith("_")
    ]
    assert public == ["summarize"]


def test_summary_method_accepts_av_inspection_directly() -> None:
    module = _module()
    method = module.DurableReplayResultComparisonSummaryService.summarize
    signature = inspect.signature(method)
    parameters = list(signature.parameters.values())
    assert [parameter.name for parameter in parameters] == ["self", "inspection"]
    hints = get_type_hints(method)
    assert hints["inspection"] is DurableReplayResultArtifactInspection


def test_summary_return_annotation_is_exact_result_type() -> None:
    module = _module()
    method = module.DurableReplayResultComparisonSummaryService.summarize
    hints = get_type_hints(method)
    assert hints["return"] is module.DurableReplayResultComparisonSummary


def test_summary_projects_identity_fields_exactly() -> None:
    module = _module()
    summary = module.DurableReplayResultComparisonSummaryService().summarize(_inspection())
    assert summary.status == "PASS"
    assert summary.round_count == 32
    assert summary.candidate_model_name == "candidate-model"
    assert summary.baseline_model_name == "baseline-model"


def test_summary_projects_exact_frozen_topk_delta_fields() -> None:
    module = _module()
    summary = module.DurableReplayResultComparisonSummaryService().summarize(_inspection())
    assert summary.top3_baseline_delta_mean_best_hits == 0.10
    assert summary.top5_baseline_delta_mean_best_hits == 0.40
    assert summary.top10_baseline_delta_mean_best_hits == 0.70
    assert summary.top3_baseline_delta_3plus_rate == 0.20
    assert summary.top5_baseline_delta_3plus_rate == 0.50
    assert summary.top10_baseline_delta_3plus_rate == 0.80
    assert summary.top3_baseline_delta_4plus_rate == 0.30
    assert summary.top5_baseline_delta_4plus_rate == 0.60
    assert summary.top10_baseline_delta_4plus_rate == 0.90


def test_summary_window_is_mapping_compatible() -> None:
    module = _module()
    hints = get_type_hints(module.DurableReplayResultComparisonSummary)
    annotation = hints["window"]
    origin = getattr(annotation, "__origin__", None)
    assert (
        annotation == Mapping[str, object]
        or origin in (Mapping, dict)
        or str(origin) == "<class 'collections.abc.Mapping'>"
    )


def test_summary_window_is_read_only_projection() -> None:
    module = _module()
    summary = module.DurableReplayResultComparisonSummaryService().summarize(_inspection())
    assert isinstance(summary.window, Mapping)
    with pytest.raises(TypeError):
        summary.window["name"] = "mutated"  # type: ignore[index]


def test_summary_has_no_winner_or_promotion_fields() -> None:
    module = _module()
    names = {field.name for field in fields(module.DurableReplayResultComparisonSummary)}
    forbidden = {
        "winner", "winner_label", "recommendation", "promotion",
        "promote", "champion", "decision",
    }
    assert names.isdisjoint(forbidden)


def test_summary_has_no_filesystem_database_or_runtime_dependency() -> None:
    source = PRODUCT_PATH.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    forbidden_prefixes = (
        "pathlib", "sqlite3", "random", "time", "datetime", "os",
        "lrp.operations.runtime", "lrp.production", "lrp.cli",
    )
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in imports
        for prefix in forbidden_prefixes
    )


def test_summary_depends_on_av_inspection_owner() -> None:
    source = PRODUCT_PATH.read_text(encoding="utf-8-sig")
    assert "lrp.operations.durable_replay_result_artifact_inspection" in source


def test_summary_declares_no_discovery_or_mutation_surface() -> None:
    source = PRODUCT_PATH.read_text(encoding="utf-8-sig").lower()
    forbidden_tokens = (
        "latest", "discover", "selector", "write_operation_artifact",
        "publish_champion", "rollback_champion",
    )
    for token in forbidden_tokens:
        assert token not in source
