from __future__ import annotations

import pytest

from lrp.contracts import ContractError
from lrp.evaluation import (
    EvaluationWindow,
    WindowEvaluation,
)
from tools.validation.model_evaluation_matrix import (
    HistoricalEvaluationMatrix,
    build_evaluation_matrix,
)


def make_result(
    *,
    model_name: str,
    window: EvaluationWindow,
    practical_delta: float,
    best_delta: float,
) -> tuple[str, WindowEvaluation]:
    return (
        model_name,
        WindowEvaluation(
            window=window,
            round_count=window.round_count,
            average_best_hits=2.0,
            average_practical_hits=1.5,
            baseline_best_hit_delta=best_delta,
            baseline_practical_hit_delta=practical_delta,
            best_hit_stddev=0.3,
            practical_hit_stddev=0.3,
            best_win_count=4,
            best_loss_count=3,
            best_tie_count=(
                window.round_count - 7
            ),
            best_sign_test_pvalue=0.20,
            practical_win_count=4,
            practical_loss_count=3,
            practical_tie_count=(
                window.round_count - 7
            ),
            practical_sign_test_pvalue=0.20,
            average_jaccard=0.20,
        ),
    )


def test_matrix_builds_model_evaluations_and_ranking() -> None:
    recent = EvaluationWindow(
        name="recent",
        start_round=1222,
        end_round=1231,
    )
    prior = EvaluationWindow(
        name="prior",
        start_round=1212,
        end_round=1221,
    )

    result = build_evaluation_matrix(
        windows=(recent, prior),
        results=(
            make_result(
                model_name="baseline",
                window=recent,
                practical_delta=0.00,
                best_delta=0.00,
            ),
            make_result(
                model_name="baseline",
                window=prior,
                practical_delta=0.00,
                best_delta=0.00,
            ),
            make_result(
                model_name="combined",
                window=recent,
                practical_delta=0.20,
                best_delta=0.10,
            ),
            make_result(
                model_name="combined",
                window=prior,
                practical_delta=0.10,
                best_delta=0.05,
            ),
        ),
    )

    assert isinstance(
        result,
        HistoricalEvaluationMatrix,
    )
    assert result.model_names == (
        "baseline",
        "combined",
    )
    assert result.window_names == (
        "recent",
        "prior",
    )
    assert len(result.evaluations) == 2
    assert result.ranking.champion == "combined"


def test_matrix_requires_complete_model_window_coverage() -> None:
    recent = EvaluationWindow(
        name="recent",
        start_round=1222,
        end_round=1231,
    )
    prior = EvaluationWindow(
        name="prior",
        start_round=1212,
        end_round=1221,
    )

    with pytest.raises(
        ContractError,
        match="complete model-window coverage",
    ):
        build_evaluation_matrix(
            windows=(recent, prior),
            results=(
                make_result(
                    model_name="baseline",
                    window=recent,
                    practical_delta=0.0,
                    best_delta=0.0,
                ),
                make_result(
                    model_name="baseline",
                    window=prior,
                    practical_delta=0.0,
                    best_delta=0.0,
                ),
                make_result(
                    model_name="combined",
                    window=recent,
                    practical_delta=0.1,
                    best_delta=0.1,
                ),
            ),
        )


def test_matrix_rejects_duplicate_model_window_cells() -> None:
    recent = EvaluationWindow(
        name="recent",
        start_round=1222,
        end_round=1231,
    )

    duplicate = make_result(
        model_name="combined",
        window=recent,
        practical_delta=0.1,
        best_delta=0.1,
    )

    with pytest.raises(
        ContractError,
        match="duplicate model-window cell",
    ):
        build_evaluation_matrix(
            windows=(recent,),
            results=(
                duplicate,
                duplicate,
            ),
        )


def test_matrix_rejects_result_for_unknown_window() -> None:
    recent = EvaluationWindow(
        name="recent",
        start_round=1222,
        end_round=1231,
    )
    other = EvaluationWindow(
        name="other",
        start_round=1202,
        end_round=1211,
    )

    with pytest.raises(
        ContractError,
        match="unknown evaluation window",
    ):
        build_evaluation_matrix(
            windows=(recent,),
            results=(
                make_result(
                    model_name="combined",
                    window=other,
                    practical_delta=0.1,
                    best_delta=0.1,
                ),
            ),
        )
