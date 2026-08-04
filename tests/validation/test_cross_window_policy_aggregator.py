from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validation.cross_window_policy_aggregator import (
    CrossWindowPolicyAggregator,
)


def result(
    name: str,
    *,
    round_count: int,
    best: float,
    practical: float,
    practical_wins: int,
) -> dict[str, object]:
    return {
        "scenario": {
            "name": name,
            "adjustment_scale": 0.25,
            "minimum_weight": 0.03,
        },
        "effectiveness": {
            "round_count": round_count,
            "best_hit_mean_delta": best,
            "practical_hit_mean_delta": practical,
            "best_adaptive_wins": 10,
            "best_noop_wins": 8,
            "best_ties": round_count - 18,
            "practical_adaptive_wins": (
                practical_wins
            ),
            "practical_noop_wins": 8,
            "practical_ties": (
                round_count
                - practical_wins
                - 8
            ),
            "average_probability_l1_delta": 0.05,
            "average_changed_set_count": 15.0,
            "changed_portfolio_round_count": (
                round_count
            ),
        },
        "final_profile": {
            "weights": {
                "hot_weight": 0.30,
                "cold_weight": 0.17,
                "gap_weight": 0.17,
                "trend_weight": 0.14,
                "transition_weight": 0.12,
                "learning_weight": 0.05,
                "adaptive_weight": 0.05,
            }
        },
    }


def write_window(
    path: Path,
    *,
    start_round: int,
    end_round: int,
    baseline_practical: float,
    floor_practical: float,
) -> None:
    round_count = (
        end_round - start_round + 1
    )

    payload = {
        "config": {
            "start_round": start_round,
            "end_round": end_round,
        },
        "scenario_count": 2,
        "results": [
            result(
                "baseline",
                round_count=round_count,
                best=-0.05,
                practical=(
                    baseline_practical
                ),
                practical_wins=20,
            ),
            result(
                "floor",
                round_count=round_count,
                best=0.05,
                practical=floor_practical,
                practical_wins=24,
            ),
        ],
        "ranking": [
            {
                "rank": 1,
                "scenario": "floor",
            },
            {
                "rank": 2,
                "scenario": "baseline",
            },
        ],
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_aggregates_multiple_windows(
    tmp_path: Path,
) -> None:
    first = (
        tmp_path
        / "window_1"
        / "policy_comparison.json"
    )
    second = (
        tmp_path
        / "window_2"
        / "policy_comparison.json"
    )

    write_window(
        first,
        start_round=100,
        end_round=199,
        baseline_practical=0.10,
        floor_practical=0.20,
    )
    write_window(
        second,
        start_round=200,
        end_round=299,
        baseline_practical=0.00,
        floor_practical=0.10,
    )

    report = (
        CrossWindowPolicyAggregator()
        .aggregate((first, second))
    )

    assert report["window_count"] == 2
    assert report["total_round_count"] == 200
    assert report["policy_count"] == 2

    floor = report["policies"]["floor"]

    assert floor[
        "practical_hit_mean_delta"
    ] == pytest.approx(0.15)
    assert floor["first_place_count"] == 2
    assert floor["total_round_count"] == 200


def test_ranking_prefers_practical_delta(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "window"
        / "policy_comparison.json"
    )

    write_window(
        path,
        start_round=100,
        end_round=199,
        baseline_practical=0.10,
        floor_practical=0.20,
    )

    report = (
        CrossWindowPolicyAggregator()
        .aggregate((path,))
    )

    assert report["ranking"][0][
        "policy_name"
    ] == "floor"


def test_discover_and_aggregate(
    tmp_path: Path,
) -> None:
    write_window(
        tmp_path
        / "first"
        / "policy_comparison.json",
        start_round=100,
        end_round=199,
        baseline_practical=0.10,
        floor_practical=0.20,
    )

    report = (
        CrossWindowPolicyAggregator()
        .discover_and_aggregate(
            tmp_path
        )
    )

    assert report["window_count"] == 1


def test_overlapping_windows_are_rejected(
    tmp_path: Path,
) -> None:
    first = (
        tmp_path
        / "first"
        / "policy_comparison.json"
    )
    second = (
        tmp_path
        / "second"
        / "policy_comparison.json"
    )

    write_window(
        first,
        start_round=100,
        end_round=199,
        baseline_practical=0.1,
        floor_practical=0.2,
    )
    write_window(
        second,
        start_round=150,
        end_round=249,
        baseline_practical=0.1,
        floor_practical=0.2,
    )

    with pytest.raises(
        ValueError,
        match="must not overlap",
    ):
        CrossWindowPolicyAggregator().aggregate(
            (first, second)
        )


def test_duplicate_paths_are_rejected(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "window"
        / "policy_comparison.json"
    )

    write_window(
        path,
        start_round=100,
        end_round=199,
        baseline_practical=0.1,
        floor_practical=0.2,
    )

    with pytest.raises(
        ValueError,
        match="paths must be unique",
    ):
        CrossWindowPolicyAggregator().aggregate(
            (path, path)
        )


def test_write_json(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "window"
        / "policy_comparison.json"
    )

    write_window(
        path,
        start_round=100,
        end_round=199,
        baseline_practical=0.1,
        floor_practical=0.2,
    )

    aggregator = (
        CrossWindowPolicyAggregator()
    )

    report = aggregator.aggregate((path,))

    output = aggregator.write_json(
        report=report,
        output=(
            tmp_path
            / "reports"
            / "cross_window.json"
        ),
    )

    payload = json.loads(
        output.read_text(
            encoding="utf-8"
        )
    )

    assert output.is_file()
    assert payload["window_count"] == 1
    assert payload["policy_count"] == 2


def test_missing_reports_are_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="no policy comparison",
    ):
        CrossWindowPolicyAggregator(
        ).discover_and_aggregate(
            tmp_path
        )
