from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from lrp.operations.durable_replay_result_promotion_action_plan import (
    DurableReplayResultPromotionActionPlanService,
)
from lrp.operations.durable_replay_result_promotion_eligibility import (
    DurableReplayResultPromotionEligibility,
)


def _eligibility(
    recommendation: str = "eligible",
    *,
    window: object | None = None,
) -> DurableReplayResultPromotionEligibility:
    if window is None:
        window = {
            "name": "hardening-window",
            "start_round": 1231,
            "end_round": 1231,
        }

    counts = {
        "eligible": (2, 7, 0),
        "insufficient_evidence": (1, 8, 0),
        "ineligible": (0, 8, 1),
    }
    candidate, neutral, baseline = counts.get(
        recommendation,
        (2, 7, 0),
    )

    return DurableReplayResultPromotionEligibility(
        status="PASS",
        round_count=1,
        candidate_model_name="candidate-model",
        baseline_model_name="baseline-model",
        recommendation=recommendation,  # type: ignore[arg-type]
        candidate_advantage_count=candidate,
        neutral_count=neutral,
        baseline_advantage_count=baseline,
        window=window,  # type: ignore[arg-type]
    )


def test_valid_action_plan_round_trip() -> None:
    result = DurableReplayResultPromotionActionPlanService().plan(
        _eligibility("eligible")
    )
    assert result.recommendation == "eligible"
    assert result.action == "prepare_publish"


@pytest.mark.parametrize(
    ("recommendation", "expected"),
    [
        ("eligible", "prepare_publish"),
        ("insufficient_evidence", "hold"),
        ("ineligible", "block"),
    ],
)
def test_exact_recommendation_mapping(
    recommendation: str,
    expected: str,
) -> None:
    result = DurableReplayResultPromotionActionPlanService().plan(
        _eligibility(recommendation)
    )
    assert result.action == expected


@pytest.mark.parametrize(
    "recommendation",
    [
        "",
        "publish",
        "approved",
        "candidate_advantage",
        "ELIGIBLE",
        "unknown",
    ],
)
def test_unknown_recommendation_fails(
    recommendation: str,
) -> None:
    with pytest.raises(ValueError):
        DurableReplayResultPromotionActionPlanService().plan(
            _eligibility(recommendation)
        )


def test_non_mapping_window_fails() -> None:
    with pytest.raises(TypeError):
        DurableReplayResultPromotionActionPlanService().plan(
            _eligibility(window=[])
        )


def test_result_is_immutable() -> None:
    result = DurableReplayResultPromotionActionPlanService().plan(
        _eligibility()
    )
    with pytest.raises(FrozenInstanceError):
        result.action = "block"  # type: ignore[misc]


def test_window_is_read_only_and_detached() -> None:
    source_window = {
        "name": "hardening-window",
        "start_round": 1231,
        "end_round": 1231,
    }
    eligibility = _eligibility(window=source_window)
    result = DurableReplayResultPromotionActionPlanService().plan(
        eligibility
    )

    assert isinstance(result.window, MappingProxyType)
    assert result.window is not eligibility.window

    source_window["name"] = "changed-after-plan"
    assert result.window["name"] == "hardening-window"

    with pytest.raises(TypeError):
        result.window["name"] = "mutated"  # type: ignore[index]


def test_action_plan_is_deterministic_for_same_input() -> None:
    eligibility = _eligibility()
    service = DurableReplayResultPromotionActionPlanService()
    first = service.plan(eligibility)
    second = service.plan(eligibility)
    assert first == second


def test_plan_does_not_mutate_input_eligibility() -> None:
    eligibility = _eligibility()
    before = (
        eligibility.status,
        eligibility.round_count,
        eligibility.candidate_model_name,
        eligibility.baseline_model_name,
        eligibility.recommendation,
        eligibility.candidate_advantage_count,
        eligibility.neutral_count,
        eligibility.baseline_advantage_count,
        dict(eligibility.window),
    )

    DurableReplayResultPromotionActionPlanService().plan(eligibility)

    after = (
        eligibility.status,
        eligibility.round_count,
        eligibility.candidate_model_name,
        eligibility.baseline_model_name,
        eligibility.recommendation,
        eligibility.candidate_advantage_count,
        eligibility.neutral_count,
        eligibility.baseline_advantage_count,
        dict(eligibility.window),
    )

    assert after == before