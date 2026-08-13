"""Adapters from replay validation results to Project M evaluation contracts."""

from __future__ import annotations

from statistics import mean

from lrp.contracts import ContractError

from lrp.evaluation.contracts import (
    EvaluationWindow,
    WindowEvaluation,
)

from .historical_replay_models import (
    ReplayRoundResult,
)
from .replay_effectiveness import (
    EffectivenessSummary,
    evaluate_effectiveness,
)


def window_evaluation_from_replay(
    *,
    window: EvaluationWindow,
    rows: tuple[ReplayRoundResult, ...],
) -> WindowEvaluation:
    """Build one model-evaluation window from paired replay rows."""

    if not isinstance(window, EvaluationWindow):
        raise TypeError(
            "window must be EvaluationWindow"
        )

    if not rows:
        raise ContractError(
            "rows must not be empty"
        )

    if any(
        not isinstance(row, ReplayRoundResult)
        for row in rows
    ):
        raise TypeError(
            "rows must contain ReplayRoundResult values"
        )

    round_numbers = tuple(
        row.round_no
        for row in rows
    )

    expected_rounds = tuple(
        range(
            window.start_round,
            window.end_round + 1,
        )
    )

    if round_numbers != expected_rounds:
        raise ContractError(
            "replay rows must exactly match evaluation window"
        )

    effectiveness = evaluate_effectiveness(
        rows
    )

    return window_evaluation_from_effectiveness(
        window=window,
        rows=rows,
        effectiveness=effectiveness,
    )


def window_evaluation_from_effectiveness(
    *,
    window: EvaluationWindow,
    rows: tuple[ReplayRoundResult, ...],
    effectiveness: EffectivenessSummary,
) -> WindowEvaluation:
    """Convert validated replay effectiveness into WindowEvaluation."""

    if not isinstance(window, EvaluationWindow):
        raise TypeError(
            "window must be EvaluationWindow"
        )

    if not isinstance(
        effectiveness,
        EffectivenessSummary,
    ):
        raise TypeError(
            "effectiveness must be EffectivenessSummary"
        )

    if not rows:
        raise ContractError(
            "rows must not be empty"
        )

    if len(rows) != window.round_count:
        raise ContractError(
            "row count must match evaluation window"
        )

    if effectiveness.round_count != len(rows):
        raise ContractError(
            "effectiveness round_count must match rows"
        )

    average_best_hits = mean(
        row.adaptive_best_hits
        for row in rows
    )

    average_practical_hits = mean(
        row.adaptive_practical_hits
        for row in rows
    )

    average_jaccard = mean(
        row.adaptive_avg_jaccard
        for row in rows
    )

    return WindowEvaluation(
        window=window,
        round_count=len(rows),
        average_best_hits=average_best_hits,
        average_practical_hits=average_practical_hits,
        baseline_best_hit_delta=(
            effectiveness.best_hit_mean_delta
        ),
        baseline_practical_hit_delta=(
            effectiveness.practical_hit_mean_delta
        ),
        best_hit_stddev=(
            effectiveness.best_hit_delta_stddev
        ),
        practical_hit_stddev=(
            effectiveness.practical_hit_delta_stddev
        ),
        best_win_count=(
            effectiveness.best_adaptive_wins
        ),
        best_loss_count=(
            effectiveness.best_noop_wins
        ),
        best_tie_count=(
            effectiveness.best_ties
        ),
        best_sign_test_pvalue=(
            effectiveness.best_sign_test_pvalue
        ),
        practical_win_count=(
            effectiveness.practical_adaptive_wins
        ),
        practical_loss_count=(
            effectiveness.practical_noop_wins
        ),
        practical_tie_count=(
            effectiveness.practical_ties
        ),
        practical_sign_test_pvalue=(
            effectiveness.practical_sign_test_pvalue
        ),
        average_jaccard=average_jaccard,
    )

