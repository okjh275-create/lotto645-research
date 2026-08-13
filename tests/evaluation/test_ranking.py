from __future__ import annotations

import pytest

from lrp.evaluation import (
    EvaluationWindow,
    WindowEvaluation,
    build_model_evaluation,
    rank_model_evaluations,
)


def make_window(
    *,
    name: str,
    start_round: int,
    practical_delta: float,
    best_delta: float,
    practical_pvalue: float = 0.20,
    best_pvalue: float = 0.20,
) -> WindowEvaluation:
    window = EvaluationWindow(
        name=name,
        start_round=start_round,
        end_round=start_round + 9,
    )

    return WindowEvaluation(
        window=window,
        round_count=10,
        average_best_hits=2.0,
        average_practical_hits=1.5,
        baseline_best_hit_delta=best_delta,
        baseline_practical_hit_delta=practical_delta,
        best_hit_stddev=0.3,
        practical_hit_stddev=0.3,
        best_win_count=4,
        best_loss_count=3,
        best_tie_count=3,
        best_sign_test_pvalue=best_pvalue,
        practical_win_count=4,
        practical_loss_count=3,
        practical_tie_count=3,
        practical_sign_test_pvalue=practical_pvalue,
        average_jaccard=0.2,
    )


def make_model(
    *,
    name: str,
    practical: tuple[float, float],
    best: tuple[float, float],
) :
    windows = (
        make_window(
            name="recent",
            start_round=1200,
            practical_delta=practical[0],
            best_delta=best[0],
        ),
        make_window(
            name="mid",
            start_round=1180,
            practical_delta=practical[1],
            best_delta=best[1],
        ),
    )

    return build_model_evaluation(
        model_name=name,
        windows=windows,
    )


def test_ranking_prefers_balanced_practical_performance() -> None:
    balanced = make_model(
        name="balanced",
        practical=(0.20, 0.10),
        best=(0.10, 0.10),
    )

    volatile = make_model(
        name="volatile",
        practical=(0.50, -0.20),
        best=(0.30, -0.10),
    )

    result = rank_model_evaluations(
        (volatile, balanced)
    )

    assert result.champion == "balanced"
    assert tuple(
        entry.model_name
        for entry in result.entries
    ) == (
        "balanced",
        "volatile",
    )


def test_model_below_worst_window_floor_is_ineligible() -> None:
    stable = make_model(
        name="stable",
        practical=(0.10, 0.05),
        best=(0.10, 0.05),
    )

    unstable = make_model(
        name="unstable",
        practical=(0.80, -0.40),
        best=(0.50, -0.20),
    )

    result = rank_model_evaluations(
        (unstable, stable)
    )

    unstable_entry = next(
        entry
        for entry in result.entries
        if entry.model_name == "unstable"
    )

    assert unstable_entry.eligible is False
    assert (
        "worst_practical_delta_below_floor"
        in unstable_entry.exclusion_reasons
    )
    assert result.champion == "stable"


def test_all_models_can_be_ineligible() -> None:
    first = make_model(
        name="first",
        practical=(-0.40, -0.50),
        best=(-0.60, -0.70),
    )

    second = make_model(
        name="second",
        practical=(-0.30, -0.40),
        best=(-0.60, -0.70),
    )

    result = rank_model_evaluations(
        (first, second)
    )

    assert result.champion is None
    assert all(
        not entry.eligible
        for entry in result.entries
    )


def test_ranking_is_deterministic_for_exact_ties() -> None:
    alpha = make_model(
        name="alpha",
        practical=(0.1, 0.1),
        best=(0.1, 0.1),
    )

    beta = make_model(
        name="beta",
        practical=(0.1, 0.1),
        best=(0.1, 0.1),
    )

    result = rank_model_evaluations(
        (beta, alpha)
    )

    assert tuple(
        entry.model_name
        for entry in result.entries
    ) == (
        "alpha",
        "beta",
    )
