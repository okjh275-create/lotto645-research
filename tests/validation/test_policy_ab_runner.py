from __future__ import annotations

import pytest

from tools.validation.policy_ab_runner import (
    DEFAULT_SCENARIOS,
    PolicyScenario,
    _rank_results,
)


def test_default_scenarios_are_fixed() -> None:
    assert tuple(
        scenario.name
        for scenario in DEFAULT_SCENARIOS
    ) == (
        "baseline",
        "damped",
        "floor",
        "conservative",
    )

    assert DEFAULT_SCENARIOS[0] == (
        PolicyScenario(
            name="baseline",
            adjustment_scale=0.25,
            minimum_weight=0.01,
        )
    )

    assert DEFAULT_SCENARIOS[1] == (
        PolicyScenario(
            name="damped",
            adjustment_scale=0.0625,
            minimum_weight=0.01,
        )
    )


def make_result(
    name: str,
    *,
    practical: float,
    best: float,
    wins: int,
    l1: float,
) -> dict[str, object]:
    return {
        "scenario": {
            "name": name,
        },
        "effectiveness": {
            "summary": {
                "practical_hit_mean_delta": practical,
                "best_hit_mean_delta": best,
                "practical_adaptive_wins": wins,
                "practical_noop_wins": 10,
                "average_probability_l1_delta": l1,
                "average_changed_set_count": 15.0,
            }
        },
    }


def test_ranking_prefers_practical_delta() -> None:
    ranking = _rank_results(
        [
            make_result(
                "a",
                practical=0.10,
                best=0.00,
                wins=20,
                l1=0.05,
            ),
            make_result(
                "b",
                practical=0.20,
                best=-0.10,
                wins=18,
                l1=0.04,
            ),
        ]
    )

    assert ranking[0]["scenario"] == "b"
    assert ranking[0]["rank"] == 1


def test_ranking_uses_best_as_tiebreaker() -> None:
    ranking = _rank_results(
        [
            make_result(
                "a",
                practical=0.10,
                best=-0.05,
                wins=20,
                l1=0.04,
            ),
            make_result(
                "b",
                practical=0.10,
                best=0.05,
                wins=18,
                l1=0.05,
            ),
        ]
    )

    assert ranking[0]["scenario"] == "b"


def test_policy_scenario_serialization() -> None:
    scenario = PolicyScenario(
        name="custom",
        adjustment_scale=0.1,
        minimum_weight=0.02,
    )

    assert scenario.as_dict() == {
        "name": "custom",
        "adjustment_scale": 0.1,
        "minimum_weight": 0.02,
    }


def test_ranking_accepts_flat_effectiveness() -> None:
    result = make_result(
        "flat",
        practical=0.15,
        best=0.05,
        wins=22,
        l1=0.03,
    )

    effectiveness = result["effectiveness"]

    assert isinstance(effectiveness, dict)

    summary = effectiveness["summary"]

    assert isinstance(summary, dict)

    result["effectiveness"] = summary

    ranking = _rank_results([result])

    assert ranking[0]["scenario"] == "flat"
    assert ranking[0][
        "practical_hit_mean_delta"
    ] == pytest.approx(0.15)
