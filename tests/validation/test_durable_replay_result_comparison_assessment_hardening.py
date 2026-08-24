from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from lrp.operations.durable_replay_result_comparison_summary import (
    DurableReplayResultComparisonSummary,
)
from lrp.operations.durable_replay_result_comparison_assessment import (
    DurableReplayResultComparisonAssessmentService,
)


def _summary(
    *,
    values: tuple[object, ...] | None = None,
    window: object | None = None,
) -> DurableReplayResultComparisonSummary:
    if values is None:
        values = (1.0, 0.0, -1.0, 0.5, 0.0, -0.5, 0.25, 0.0, -0.25)
    if window is None:
        window = {
            "name": "sample",
            "start_round": 1200,
            "end_round": 1231,
        }
    return DurableReplayResultComparisonSummary(
        status="PASS",
        round_count=32,
        candidate_model_name="candidate-model",
        baseline_model_name="baseline-model",
        top3_baseline_delta_mean_best_hits=values[0],  # type: ignore[arg-type]
        top5_baseline_delta_mean_best_hits=values[1],  # type: ignore[arg-type]
        top10_baseline_delta_mean_best_hits=values[2],  # type: ignore[arg-type]
        top3_baseline_delta_3plus_rate=values[3],  # type: ignore[arg-type]
        top5_baseline_delta_3plus_rate=values[4],  # type: ignore[arg-type]
        top10_baseline_delta_3plus_rate=values[5],  # type: ignore[arg-type]
        top3_baseline_delta_4plus_rate=values[6],  # type: ignore[arg-type]
        top5_baseline_delta_4plus_rate=values[7],  # type: ignore[arg-type]
        top10_baseline_delta_4plus_rate=values[8],  # type: ignore[arg-type]
        window=window,  # type: ignore[arg-type]
    )


def test_valid_assessment_round_trip() -> None:
    result = DurableReplayResultComparisonAssessmentService().assess(_summary())
    assert result.candidate_advantage_count == 3
    assert result.neutral_count == 3
    assert result.baseline_advantage_count == 3
    assert (
        result.candidate_advantage_count
        + result.neutral_count
        + result.baseline_advantage_count
    ) == 9


@pytest.mark.parametrize(
    "value",
    [True, False, "1", None, [], {}, complex(1, 0)],
)
def test_invalid_numeric_evidence_fails(value: object) -> None:
    values = [1.0] * 9
    values[0] = value
    with pytest.raises((TypeError, ValueError)):
        DurableReplayResultComparisonAssessmentService().assess(
            _summary(values=tuple(values))
        )


@pytest.mark.parametrize("value", [math.nan, float("nan")])
def test_nan_evidence_fails(value: float) -> None:
    values = [1.0] * 9
    values[4] = value
    with pytest.raises((TypeError, ValueError)):
        DurableReplayResultComparisonAssessmentService().assess(
            _summary(values=tuple(values))
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1e-300, "candidate_advantage"),
        (0.0, "neutral"),
        (-0.0, "neutral"),
        (-1e-300, "baseline_advantage"),
    ],
)
def test_zero_threshold_is_exact_sign_only(
    value: float,
    expected: str,
) -> None:
    values = [0.0] * 9
    values[0] = value
    result = DurableReplayResultComparisonAssessmentService().assess(
        _summary(values=tuple(values))
    )
    assert result.top3_baseline_delta_mean_best_hits_assessment == expected


def test_all_positive_counts_are_exact() -> None:
    result = DurableReplayResultComparisonAssessmentService().assess(
        _summary(values=(1.0,) * 9)
    )
    assert result.candidate_advantage_count == 9
    assert result.neutral_count == 0
    assert result.baseline_advantage_count == 0


def test_all_zero_counts_are_exact() -> None:
    result = DurableReplayResultComparisonAssessmentService().assess(
        _summary(values=(0.0,) * 9)
    )
    assert result.candidate_advantage_count == 0
    assert result.neutral_count == 9
    assert result.baseline_advantage_count == 0


def test_all_negative_counts_are_exact() -> None:
    result = DurableReplayResultComparisonAssessmentService().assess(
        _summary(values=(-1.0,) * 9)
    )
    assert result.candidate_advantage_count == 0
    assert result.neutral_count == 0
    assert result.baseline_advantage_count == 9


def test_non_mapping_window_fails() -> None:
    with pytest.raises(TypeError):
        DurableReplayResultComparisonAssessmentService().assess(
            _summary(window=[])
        )


def test_assessment_result_is_immutable() -> None:
    result = DurableReplayResultComparisonAssessmentService().assess(_summary())
    with pytest.raises(FrozenInstanceError):
        result.status = "FAIL"  # type: ignore[misc]


def test_window_projection_is_read_only_and_detached() -> None:
    source_window = {
        "name": "sample",
        "start_round": 1200,
        "end_round": 1231,
    }
    result = DurableReplayResultComparisonAssessmentService().assess(
        _summary(window=source_window)
    )
    assert isinstance(result.window, MappingProxyType)
    source_window["name"] = "changed-after-assessment"
    assert result.window["name"] == "sample"
    with pytest.raises(TypeError):
        result.window["name"] = "mutated"  # type: ignore[index]


def test_assessment_is_deterministic_for_same_input() -> None:
    summary = _summary()
    service = DurableReplayResultComparisonAssessmentService()
    first = service.assess(summary)
    second = service.assess(summary)
    assert first == second


def test_assessment_does_not_mutate_input_summary() -> None:
    summary = _summary()
    before = (
        summary.status,
        summary.round_count,
        summary.candidate_model_name,
        summary.baseline_model_name,
        summary.top3_baseline_delta_mean_best_hits,
        summary.top5_baseline_delta_mean_best_hits,
        summary.top10_baseline_delta_mean_best_hits,
        summary.top3_baseline_delta_3plus_rate,
        summary.top5_baseline_delta_3plus_rate,
        summary.top10_baseline_delta_3plus_rate,
        summary.top3_baseline_delta_4plus_rate,
        summary.top5_baseline_delta_4plus_rate,
        summary.top10_baseline_delta_4plus_rate,
        dict(summary.window),
    )
    DurableReplayResultComparisonAssessmentService().assess(summary)
    after = (
        summary.status,
        summary.round_count,
        summary.candidate_model_name,
        summary.baseline_model_name,
        summary.top3_baseline_delta_mean_best_hits,
        summary.top5_baseline_delta_mean_best_hits,
        summary.top10_baseline_delta_mean_best_hits,
        summary.top3_baseline_delta_3plus_rate,
        summary.top5_baseline_delta_3plus_rate,
        summary.top10_baseline_delta_3plus_rate,
        summary.top3_baseline_delta_4plus_rate,
        summary.top5_baseline_delta_4plus_rate,
        summary.top10_baseline_delta_4plus_rate,
        dict(summary.window),
    )
    assert after == before
