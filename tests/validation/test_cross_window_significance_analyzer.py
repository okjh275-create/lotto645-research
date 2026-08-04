from __future__ import annotations

import pytest

from tools.validation.cross_window_significance_analyzer import (
    CrossWindowSignificanceAnalyzer,
)


def make_report() -> dict[str, object]:
    return {
        "policies": {
            "floor": {
                "best_adaptive_wins": 63,
                "best_noop_wins": 54,
                "best_ties": 183,
                "practical_adaptive_wins": 73,
                "practical_noop_wins": 66,
                "practical_ties": 161,
            },
            "baseline": {
                "best_adaptive_wins": 62,
                "best_noop_wins": 61,
                "best_ties": 177,
                "practical_adaptive_wins": 70,
                "practical_noop_wins": 66,
                "practical_ties": 164,
            },
        }
    }


def test_analyzes_policy_significance() -> None:
    result = (
        CrossWindowSignificanceAnalyzer()
        .analyze(make_report())
    )

    assert result["policy_count"] == 2

    floor = result[
        "policies"
    ]["floor"]

    assert floor["best"][
        "adaptive_wins"
    ] == 63
    assert floor["best"][
        "noop_wins"
    ] == 54
    assert floor["best"][
        "non_tie_count"
    ] == 117
    assert floor["best"][
        "direction"
    ] == "adaptive_better"


def test_equal_wins_return_tie() -> None:
    outcome = (
        CrossWindowSignificanceAnalyzer()
        ._outcome(
            adaptive_wins=10,
            noop_wins=10,
            ties=5,
        )
    )

    assert outcome["direction"] == "tie"
    assert outcome["p_value"] == pytest.approx(
        1.0
    )
    assert outcome["significant"] is False


def test_zero_non_ties_return_one() -> None:
    outcome = (
        CrossWindowSignificanceAnalyzer()
        ._outcome(
            adaptive_wins=0,
            noop_wins=0,
            ties=100,
        )
    )

    assert outcome["p_value"] == pytest.approx(
        1.0
    )


def test_strong_imbalance_is_significant() -> None:
    outcome = (
        CrossWindowSignificanceAnalyzer()
        ._outcome(
            adaptive_wins=20,
            noop_wins=2,
            ties=0,
        )
    )

    assert outcome["direction"] == (
        "adaptive_better"
    )
    assert outcome["p_value"] < 0.05
    assert outcome["significant"] is True


def test_invalid_report_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="report must be a mapping",
    ):
        CrossWindowSignificanceAnalyzer().analyze(
            object(),  # type: ignore[arg-type]
        )


def test_negative_count_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="greater than or equal",
    ):
        CrossWindowSignificanceAnalyzer()._outcome(
            adaptive_wins=-1,
            noop_wins=2,
            ties=0,
        )
