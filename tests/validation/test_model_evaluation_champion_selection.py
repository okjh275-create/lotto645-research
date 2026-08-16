from __future__ import annotations

from lrp.evaluation import (
    ChampionPromotionPolicy,
    ChampionSelection,
    EvaluationWindow,
    WindowEvaluation,
)

from tools.validation.model_evaluation_champion import (
    HistoricalChampionSelection,
    select_historical_champion,
)

from tools.validation.model_evaluation_matrix import (
    build_evaluation_matrix,
)


def _window(
    name: str,
    start: int,
    end: int,
) -> EvaluationWindow:
    return EvaluationWindow(
        name=name,
        start_round=start,
        end_round=end,
    )


def _evaluation(
    window: EvaluationWindow,
    *,
    best_delta: float,
    practical_delta: float,
    best_p: float = 0.01,
    practical_p: float = 0.01,
) -> WindowEvaluation:
    return WindowEvaluation(
        window=window,
        round_count=window.round_count,
        average_best_hits=2.0 + best_delta,
        average_practical_hits=2.0 + practical_delta,
        baseline_best_hit_delta=best_delta,
        baseline_practical_hit_delta=practical_delta,
        best_hit_stddev=0.5,
        practical_hit_stddev=0.5,
        best_win_count=10,
        best_loss_count=5,
        best_tie_count=5,
        best_sign_test_pvalue=best_p,
        practical_win_count=10,
        practical_loss_count=5,
        practical_tie_count=5,
        practical_sign_test_pvalue=practical_p,
        average_jaccard=0.2,
    )


def _matrix():
    windows = (
        _window(
            "short",
            1001,
            1020,
        ),
        _window(
            "mid",
            1021,
            1040,
        ),
        _window(
            "long",
            1041,
            1060,
        ),
    )

    results = []

    for window in windows:
        results.extend(
            (
                (
                    "baseline",
                    _evaluation(
                        window,
                        best_delta=0.05,
                        practical_delta=0.05,
                    ),
                ),
                (
                    "challenger",
                    _evaluation(
                        window,
                        best_delta=0.80,
                        practical_delta=0.90,
                    ),
                ),
            )
        )

    return build_evaluation_matrix(
        windows=windows,
        results=results,
    )


def test_select_historical_champion_returns_contract():
    matrix = _matrix()

    result = select_historical_champion(
        matrix=matrix,
        policy=ChampionPromotionPolicy(
            minimum_composite_margin=0.0,
            minimum_significance_score=0.0,
        ),
    )

    assert isinstance(
        result,
        HistoricalChampionSelection,
    )

    assert result.matrix is matrix

    assert isinstance(
        result.selection,
        ChampionSelection,
    )

    assert (
        result.selection.ranking_champion
        == matrix.ranking.champion
    )

    assert (
        result.selection.selected_model
        == "challenger"
    )


def test_historical_selection_serializes_matrix_and_selection():
    matrix = _matrix()

    result = select_historical_champion(
        matrix=matrix,
        policy=ChampionPromotionPolicy(
            minimum_composite_margin=0.0,
            minimum_significance_score=0.0,
        ),
    )

    payload = result.as_dict()

    assert payload["matrix"] == matrix.as_dict()

    assert (
        payload["selection"]
        == result.selection.as_dict()
    )

    assert (
        payload["ranking_champion"]
        == matrix.ranking.champion
    )

    assert (
        payload["selected_model"]
        == result.selection.selected_model
    )


def test_historical_selection_uses_matrix_ranking_entries():
    matrix = _matrix()

    result = select_historical_champion(
        matrix=matrix,
        policy=ChampionPromotionPolicy(
            minimum_composite_margin=0.0,
            minimum_significance_score=0.0,
        ),
    )

    assert (
        result.selection.ranking_champion
        == next(
            entry.model_name
            for entry in matrix.ranking.entries
            if entry.eligible
        )
    )


def test_historical_selection_rejects_invalid_matrix():
    try:
        select_historical_champion(
            matrix=object(),
        )
    except TypeError as exc:
        assert (
            "HistoricalEvaluationMatrix"
            in str(exc)
        )
    else:
        raise AssertionError(
            "invalid matrix must raise TypeError"
        )
