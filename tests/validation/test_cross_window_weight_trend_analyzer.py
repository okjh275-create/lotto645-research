from __future__ import annotations

import pytest

from tools.validation.cross_window_weight_trend_analyzer import (
    CrossWindowWeightTrendAnalyzer,
)


def weights(
    *,
    hot: float,
    learning: float,
    adaptive: float,
) -> dict[str, float]:
    remaining = (
        1.0
        - hot
        - learning
        - adaptive
    )

    return {
        "hot_weight": hot,
        "cold_weight": remaining * 0.25,
        "gap_weight": remaining * 0.25,
        "trend_weight": remaining * 0.20,
        "transition_weight": remaining * 0.30,
        "learning_weight": learning,
        "adaptive_weight": adaptive,
    }


def make_report() -> dict[str, object]:
    return {
        "windows": [
            {
                "start_round": 100,
                "end_round": 199,
            },
            {
                "start_round": 200,
                "end_round": 299,
            },
            {
                "start_round": 300,
                "end_round": 399,
            },
        ],
        "policies": {
            "floor": {
                "windows": [
                    {
                        "start_round": 100,
                        "end_round": 199,
                        "final_weights": weights(
                            hot=0.30,
                            learning=0.030,
                            adaptive=0.030,
                        ),
                    },
                    {
                        "start_round": 200,
                        "end_round": 299,
                        "final_weights": weights(
                            hot=0.31,
                            learning=0.031,
                            adaptive=0.030,
                        ),
                    },
                    {
                        "start_round": 300,
                        "end_round": 399,
                        "final_weights": weights(
                            hot=0.32,
                            learning=0.032,
                            adaptive=0.030,
                        ),
                    },
                ]
            }
        },
    }


def test_analyzes_increasing_weight() -> None:
    report = CrossWindowWeightTrendAnalyzer(
        stable_tolerance=0.005,
    ).analyze(make_report())

    hot = report[
        "policies"
    ]["floor"]["weights"]["hot_weight"]

    assert hot["direction"] == "increasing"
    assert hot["first"] == pytest.approx(
        0.30
    )
    assert hot["last"] == pytest.approx(
        0.32
    )
    assert hot["net_change"] == pytest.approx(
        0.02
    )


def test_analyzes_stable_weight() -> None:
    report = CrossWindowWeightTrendAnalyzer(
        stable_tolerance=0.005,
    ).analyze(make_report())

    adaptive = report[
        "policies"
    ]["floor"]["weights"][
        "adaptive_weight"
    ]

    assert adaptive["direction"] == "stable"
    assert adaptive["stable_steps"] == 2


def test_learning_small_change_is_stable() -> None:
    report = CrossWindowWeightTrendAnalyzer(
        stable_tolerance=0.005,
    ).analyze(make_report())

    learning = report[
        "policies"
    ]["floor"]["weights"][
        "learning_weight"
    ]

    assert learning["direction"] == "stable"
    assert learning["net_change"] == (
        pytest.approx(0.002)
    )


def test_single_window_is_insufficient() -> None:
    report = make_report()

    policies = report["policies"]

    assert isinstance(policies, dict)

    floor = policies["floor"]

    assert isinstance(floor, dict)

    policy_windows = floor["windows"]

    assert isinstance(policy_windows, list)

    floor["windows"] = policy_windows[:1]

    result = (
        CrossWindowWeightTrendAnalyzer()
        .analyze(report)
    )

    hot = result[
        "policies"
    ]["floor"]["weights"]["hot_weight"]

    assert hot["direction"] == (
        "insufficient_data"
    )
    assert hot["increase_steps"] == 0
    assert hot["decrease_steps"] == 0
    assert hot["stable_steps"] == 0


def test_report_metadata_is_recorded() -> None:
    result = (
        CrossWindowWeightTrendAnalyzer(
            stable_tolerance=0.005,
        )
        .analyze(make_report())
    )

    assert result["schema_version"] == 1
    assert result["stable_tolerance"] == (
        pytest.approx(0.005)
    )
    assert result["window_count"] == 3
    assert result["policy_count"] == 1


def test_invalid_tolerance_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="greater than or equal",
    ):
        CrossWindowWeightTrendAnalyzer(
            stable_tolerance=-0.1
        )


def test_non_numeric_tolerance_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="must be numeric",
    ):
        CrossWindowWeightTrendAnalyzer(
            stable_tolerance="invalid"  # type: ignore[arg-type]
        )


def test_invalid_report_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="report must be a mapping",
    ):
        CrossWindowWeightTrendAnalyzer().analyze(
            object(),  # type: ignore[arg-type]
        )


def test_invalid_weight_total_is_rejected() -> None:
    report = make_report()

    policies = report["policies"]

    assert isinstance(policies, dict)

    floor = policies["floor"]

    assert isinstance(floor, dict)

    policy_windows = floor["windows"]

    assert isinstance(policy_windows, list)

    first = policy_windows[0]

    assert isinstance(first, dict)

    final_weights = first["final_weights"]

    assert isinstance(final_weights, dict)

    final_weights["hot_weight"] = 0.90

    with pytest.raises(
        ValueError,
        match="sum to 1.0",
    ):
        CrossWindowWeightTrendAnalyzer().analyze(
            report
        )
