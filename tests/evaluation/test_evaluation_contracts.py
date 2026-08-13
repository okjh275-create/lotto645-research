from __future__ import annotations

import pytest

from lrp.contracts import ContractError
from lrp.evaluation import (
    EvaluationWindow,
    WindowEvaluation,
    build_model_evaluation,
)


def make_window(
    *,
    name: str = "recent",
    start_round: int = 1200,
    end_round: int = 1209,
    best_delta: float = 0.2,
    practical_delta: float = 0.1,
    best_pvalue: float = 0.04,
    practical_pvalue: float = 0.03,
) -> WindowEvaluation:
    window = EvaluationWindow(
        name=name,
        start_round=start_round,
        end_round=end_round,
    )

    return WindowEvaluation(
        window=window,
        round_count=window.round_count,
        average_best_hits=2.2,
        average_practical_hits=1.8,
        baseline_best_hit_delta=best_delta,
        baseline_practical_hit_delta=practical_delta,
        best_hit_stddev=0.5,
        practical_hit_stddev=0.4,
        best_win_count=4,
        best_loss_count=2,
        best_tie_count=4,
        best_sign_test_pvalue=best_pvalue,
        practical_win_count=5,
        practical_loss_count=2,
        practical_tie_count=3,
        practical_sign_test_pvalue=practical_pvalue,
        average_jaccard=0.21,
    )


def test_evaluation_window_round_count() -> None:
    window = EvaluationWindow(
        name="recent",
        start_round=1200,
        end_round=1209,
    )

    assert window.round_count == 10
    assert window.as_dict()["round_count"] == 10


def test_window_evaluation_requires_matching_round_count() -> None:
    with pytest.raises(
        ContractError,
        match="round_count must match",
    ):
        WindowEvaluation(
            window=EvaluationWindow(
                name="recent",
                start_round=1200,
                end_round=1209,
            ),
            round_count=9,
            average_best_hits=2.0,
            average_practical_hits=1.5,
            baseline_best_hit_delta=0.1,
            baseline_practical_hit_delta=0.1,
            best_hit_stddev=0.2,
            practical_hit_stddev=0.2,
            best_win_count=4,
            best_loss_count=2,
            best_tie_count=3,
            best_sign_test_pvalue=0.1,
            practical_win_count=4,
            practical_loss_count=2,
            practical_tie_count=3,
            practical_sign_test_pvalue=0.1,
            average_jaccard=0.2,
        )


def test_build_model_evaluation_aggregates_windows() -> None:
    recent = make_window(
        name="recent",
        start_round=1200,
        end_round=1209,
        best_delta=0.3,
        practical_delta=0.2,
        best_pvalue=0.01,
        practical_pvalue=0.02,
    )

    mid = make_window(
        name="mid",
        start_round=1180,
        end_round=1189,
        best_delta=-0.1,
        practical_delta=0.0,
        best_pvalue=0.5,
        practical_pvalue=0.8,
    )

    evaluation = build_model_evaluation(
        model_name="candidate-a",
        windows=(recent, mid),
    )

    assert evaluation.total_round_count == 20
    assert evaluation.mean_best_hit_delta == pytest.approx(0.1)
    assert evaluation.mean_practical_hit_delta == pytest.approx(0.1)
    assert evaluation.worst_best_hit_delta == pytest.approx(-0.1)
    assert evaluation.worst_practical_hit_delta == pytest.approx(0.0)
    assert evaluation.significant_best_window_count == 1
    assert evaluation.significant_practical_window_count == 1


def test_model_evaluation_rejects_duplicate_window_names() -> None:
    first = make_window()
    second = make_window(
        start_round=1180,
        end_round=1189,
    )

    with pytest.raises(
        ContractError,
        match="window names must be unique",
    ):
        build_model_evaluation(
            model_name="candidate-a",
            windows=(first, second),
        )


def test_significance_level_is_validated() -> None:
    with pytest.raises(
        ContractError,
        match="significance_level",
    ):
        build_model_evaluation(
            model_name="candidate-a",
            windows=(make_window(),),
            significance_level=1.0,
        )
