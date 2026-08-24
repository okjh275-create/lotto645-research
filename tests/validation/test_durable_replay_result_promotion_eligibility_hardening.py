from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from lrp.operations.durable_replay_result_comparison_assessment import (
    DurableReplayResultComparisonAssessment,
)
from lrp.operations.durable_replay_result_promotion_eligibility import (
    DurableReplayResultPromotionEligibilityService,
)


def _assessment(
    candidate: object = 2,
    neutral: object = 7,
    baseline: object = 0,
    *,
    window: object | None = None,
) -> DurableReplayResultComparisonAssessment:
    if window is None:
        window = {
            "name": "hardening-window",
            "start_round": 1231,
            "end_round": 1231,
        }

    labels = [
        "candidate_advantage",
        "candidate_advantage",
        "neutral",
        "neutral",
        "neutral",
        "neutral",
        "neutral",
        "neutral",
        "neutral",
    ]

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
        candidate_advantage_count=candidate,  # type: ignore[arg-type]
        neutral_count=neutral,  # type: ignore[arg-type]
        baseline_advantage_count=baseline,  # type: ignore[arg-type]
        window=window,  # type: ignore[arg-type]
    )


def test_valid_eligibility_round_trip() -> None:
    result = DurableReplayResultPromotionEligibilityService().evaluate(
        _assessment()
    )
    assert result.recommendation == "eligible"
    assert result.candidate_advantage_count == 2
    assert result.neutral_count == 7
    assert result.baseline_advantage_count == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate", True),
        ("neutral", False),
        ("baseline", True),
        ("candidate", "2"),
        ("neutral", None),
        ("baseline", 1.5),
    ],
)
def test_invalid_count_type_fails(field: str, value: object) -> None:
    kwargs = {"candidate": 2, "neutral": 7, "baseline": 0}
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError)):
        DurableReplayResultPromotionEligibilityService().evaluate(
            _assessment(**kwargs)
        )


@pytest.mark.parametrize(
    ("candidate", "neutral", "baseline"),
    [
        (-1, 10, 0),
        (2, -1, 8),
        (2, 8, -1),
    ],
)
def test_negative_count_fails(
    candidate: int,
    neutral: int,
    baseline: int,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        DurableReplayResultPromotionEligibilityService().evaluate(
            _assessment(candidate, neutral, baseline)
        )


@pytest.mark.parametrize(
    ("candidate", "neutral", "baseline"),
    [
        (2, 6, 0),
        (2, 8, 0),
        (9, 1, 0),
        (0, 0, 0),
    ],
)
def test_count_sum_other_than_nine_fails(
    candidate: int,
    neutral: int,
    baseline: int,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        DurableReplayResultPromotionEligibilityService().evaluate(
            _assessment(candidate, neutral, baseline)
        )


def test_non_mapping_window_fails() -> None:
    with pytest.raises(TypeError):
        DurableReplayResultPromotionEligibilityService().evaluate(
            _assessment(window=[])
        )


def test_result_is_immutable() -> None:
    result = DurableReplayResultPromotionEligibilityService().evaluate(
        _assessment()
    )
    with pytest.raises(FrozenInstanceError):
        result.recommendation = "ineligible"  # type: ignore[misc]


def test_window_is_read_only_and_detached() -> None:
    source_window = {
        "name": "hardening-window",
        "start_round": 1231,
        "end_round": 1231,
    }
    assessment = _assessment(window=source_window)
    result = DurableReplayResultPromotionEligibilityService().evaluate(
        assessment
    )

    assert isinstance(result.window, MappingProxyType)
    assert result.window is not assessment.window
    source_window["name"] = "changed-after-evaluate"
    assert result.window["name"] == "hardening-window"

    with pytest.raises(TypeError):
        result.window["name"] = "mutated"  # type: ignore[index]


def test_eligibility_is_deterministic_for_same_input() -> None:
    assessment = _assessment()
    service = DurableReplayResultPromotionEligibilityService()
    first = service.evaluate(assessment)
    second = service.evaluate(assessment)
    assert first == second


def test_evaluate_does_not_mutate_input_assessment() -> None:
    assessment = _assessment()
    before = (
        assessment.status,
        assessment.round_count,
        assessment.candidate_model_name,
        assessment.baseline_model_name,
        assessment.candidate_advantage_count,
        assessment.neutral_count,
        assessment.baseline_advantage_count,
        dict(assessment.window),
    )

    DurableReplayResultPromotionEligibilityService().evaluate(assessment)

    after = (
        assessment.status,
        assessment.round_count,
        assessment.candidate_model_name,
        assessment.baseline_model_name,
        assessment.candidate_advantage_count,
        assessment.neutral_count,
        assessment.baseline_advantage_count,
        dict(assessment.window),
    )
    assert after == before