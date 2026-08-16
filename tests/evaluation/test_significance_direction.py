from __future__ import annotations

from lrp.evaluation import (
    EvaluationWindow,
    WindowEvaluation,
    build_model_evaluation,
)


def make_window(
    *,
    name: str,
    best_delta: float,
    practical_delta: float,
    best_p: float,
    practical_p: float,
) -> WindowEvaluation:
    window = EvaluationWindow(
        name=name,
        start_round=1200,
        end_round=1209,
    )

    return WindowEvaluation(
        window=window,
        round_count=10,
        average_best_hits=2.0,
        average_practical_hits=2.0,
        baseline_best_hit_delta=best_delta,
        baseline_practical_hit_delta=practical_delta,
        best_hit_stddev=0.5,
        practical_hit_stddev=0.5,
        best_win_count=5,
        best_loss_count=5,
        best_tie_count=0,
        best_sign_test_pvalue=best_p,
        practical_win_count=5,
        practical_loss_count=5,
        practical_tie_count=0,
        practical_sign_test_pvalue=practical_p,
        average_jaccard=0.2,
    )


def test_significant_negative_delta_is_not_counted_as_improvement() -> None:
    evaluation = build_model_evaluation(
        model_name="candidate",
        windows=(
            make_window(
                name="negative",
                best_delta=-0.20,
                practical_delta=-0.30,
                best_p=0.01,
                practical_p=0.01,
            ),
        ),
    )

    assert evaluation.significant_best_window_count == 0
    assert evaluation.significant_practical_window_count == 0


def test_significant_positive_delta_is_counted_as_improvement() -> None:
    evaluation = build_model_evaluation(
        model_name="candidate",
        windows=(
            make_window(
                name="positive",
                best_delta=0.20,
                practical_delta=0.30,
                best_p=0.01,
                practical_p=0.01,
            ),
        ),
    )

    assert evaluation.significant_best_window_count == 1
    assert evaluation.significant_practical_window_count == 1


def test_non_significant_positive_delta_is_not_counted() -> None:
    evaluation = build_model_evaluation(
        model_name="candidate",
        windows=(
            make_window(
                name="nonsignificant",
                best_delta=0.20,
                practical_delta=0.30,
                best_p=0.20,
                practical_p=0.20,
            ),
        ),
    )

    assert evaluation.significant_best_window_count == 0
    assert evaluation.significant_practical_window_count == 0


def test_zero_delta_is_not_counted_as_improvement() -> None:
    evaluation = build_model_evaluation(
        model_name="candidate",
        windows=(
            make_window(
                name="zero",
                best_delta=0.0,
                practical_delta=0.0,
                best_p=0.01,
                practical_p=0.01,
            ),
        ),
    )

    assert evaluation.significant_best_window_count == 0
    assert evaluation.significant_practical_window_count == 0
