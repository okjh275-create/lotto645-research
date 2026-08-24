from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from lrp.operations.durable_replay_result_artifact_inspection import (
    DurableReplayResultArtifactInspection,
)
from lrp.operations.durable_replay_result_comparison_summary import (
    DurableReplayResultComparisonSummaryService,
)


def _inspection(
    *,
    top3: object | None = None,
    top5: object | None = None,
    top10: object | None = None,
    window: object | None = None,
) -> DurableReplayResultArtifactInspection:
    if top3 is None:
        top3 = {
            "baseline_delta_mean_best_hits": 0.1,
            "baseline_delta_3plus_rate": 0.2,
            "baseline_delta_4plus_rate": 0.3,
        }
    if top5 is None:
        top5 = {
            "baseline_delta_mean_best_hits": 0.4,
            "baseline_delta_3plus_rate": 0.5,
            "baseline_delta_4plus_rate": 0.6,
        }
    if top10 is None:
        top10 = {
            "baseline_delta_mean_best_hits": 0.7,
            "baseline_delta_3plus_rate": 0.8,
            "baseline_delta_4plus_rate": 0.9,
        }
    if window is None:
        window = {
            "name": "sample",
            "start_round": 1200,
            "end_round": 1231,
        }
    return DurableReplayResultArtifactInspection(
        status="PASS",
        round_count=32,
        candidate_model_name="candidate-model",
        baseline_model_name="baseline-model",
        evaluation=MappingProxyType(
            {
                "top3": top3,
                "top5": top5,
                "top10": top10,
                "window": window,
            }
        ),
    )


def test_valid_summary_round_trip() -> None:
    summary = DurableReplayResultComparisonSummaryService().summarize(
        _inspection()
    )
    assert summary.status == "PASS"
    assert summary.round_count == 32
    assert summary.top3_baseline_delta_mean_best_hits == 0.1
    assert summary.top10_baseline_delta_4plus_rate == 0.9


@pytest.mark.parametrize("key", ["top3", "top5", "top10", "window"])
def test_missing_required_summary_block_fails(key: str) -> None:
    evaluation = {
        "top3": {
            "baseline_delta_mean_best_hits": 0.1,
            "baseline_delta_3plus_rate": 0.2,
            "baseline_delta_4plus_rate": 0.3,
        },
        "top5": {
            "baseline_delta_mean_best_hits": 0.4,
            "baseline_delta_3plus_rate": 0.5,
            "baseline_delta_4plus_rate": 0.6,
        },
        "top10": {
            "baseline_delta_mean_best_hits": 0.7,
            "baseline_delta_3plus_rate": 0.8,
            "baseline_delta_4plus_rate": 0.9,
        },
        "window": {"name": "sample"},
    }
    del evaluation[key]
    inspection = DurableReplayResultArtifactInspection(
        status="PASS",
        round_count=1,
        candidate_model_name="candidate-model",
        baseline_model_name="baseline-model",
        evaluation=MappingProxyType(evaluation),
    )
    with pytest.raises(KeyError):
        DurableReplayResultComparisonSummaryService().summarize(
            inspection
        )


@pytest.mark.parametrize("key", ["top3", "top5", "top10"])
def test_non_mapping_topk_block_fails(key: str) -> None:
    kwargs = {
        "top3": None,
        "top5": None,
        "top10": None,
    }
    kwargs[key] = []
    inspection = _inspection(**kwargs)
    with pytest.raises(TypeError):
        DurableReplayResultComparisonSummaryService().summarize(
            inspection
        )


def test_non_mapping_window_fails() -> None:
    inspection = _inspection(window=[])
    with pytest.raises(TypeError):
        DurableReplayResultComparisonSummaryService().summarize(
            inspection
        )


@pytest.mark.parametrize(
    "metric",
    [
        "baseline_delta_mean_best_hits",
        "baseline_delta_3plus_rate",
        "baseline_delta_4plus_rate",
    ],
)
def test_missing_delta_metric_fails(metric: str) -> None:
    top3 = {
        "baseline_delta_mean_best_hits": 0.1,
        "baseline_delta_3plus_rate": 0.2,
        "baseline_delta_4plus_rate": 0.3,
    }
    del top3[metric]
    with pytest.raises(KeyError):
        DurableReplayResultComparisonSummaryService().summarize(
            _inspection(top3=top3)
        )


@pytest.mark.parametrize("value", ["x", None, [], {}])
def test_invalid_numeric_delta_fails(value: object) -> None:
    top3 = {
        "baseline_delta_mean_best_hits": value,
        "baseline_delta_3plus_rate": 0.2,
        "baseline_delta_4plus_rate": 0.3,
    }
    with pytest.raises((TypeError, ValueError)):
        DurableReplayResultComparisonSummaryService().summarize(
            _inspection(top3=top3)
        )


def test_summary_result_is_immutable() -> None:
    summary = DurableReplayResultComparisonSummaryService().summarize(
        _inspection()
    )
    with pytest.raises(FrozenInstanceError):
        summary.status = "FAIL"  # type: ignore[misc]


def test_window_projection_is_read_only_and_detached() -> None:
    source_window = {
        "name": "sample",
        "start_round": 1200,
        "end_round": 1231,
    }
    summary = DurableReplayResultComparisonSummaryService().summarize(
        _inspection(window=source_window)
    )
    assert isinstance(summary.window, MappingProxyType)
    source_window["name"] = "changed-after-summary"
    assert summary.window["name"] == "sample"
    with pytest.raises(TypeError):
        summary.window["name"] = "mutated"  # type: ignore[index]


def test_summary_is_deterministic_for_same_input() -> None:
    inspection = _inspection()
    service = DurableReplayResultComparisonSummaryService()
    first = service.summarize(inspection)
    second = service.summarize(inspection)
    assert first == second


def test_summary_does_not_mutate_input_evaluation() -> None:
    inspection = _inspection()
    before = {
        key: value.copy() if isinstance(value, dict) else value
        for key, value in inspection.evaluation.items()
    }
    DurableReplayResultComparisonSummaryService().summarize(inspection)
    after = {
        key: value.copy() if isinstance(value, dict) else value
        for key, value in inspection.evaluation.items()
    }
    assert after == before
