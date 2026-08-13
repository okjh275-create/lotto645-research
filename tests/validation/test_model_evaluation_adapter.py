from __future__ import annotations

import pytest

from lrp.contracts import ContractError
from lrp.evaluation import EvaluationWindow
from tools.validation.model_evaluation_adapter import (
    window_evaluation_from_replay,
)
from tools.validation.historical_replay_models import (
    ReplayRoundResult,
)


def make_row(
    *,
    round_no: int,
    noop_best: int,
    adaptive_best: int,
    noop_practical: int,
    adaptive_practical: int,
    adaptive_jaccard: float,
) -> ReplayRoundResult:
    return ReplayRoundResult(
        round_no=round_no,
        seed=20260000 + round_no,
        history_draws=100,
        noop_best_hits=noop_best,
        adaptive_best_hits=adaptive_best,
        noop_practical_hits=noop_practical,
        adaptive_practical_hits=adaptive_practical,
        noop_avg_jaccard=0.20,
        adaptive_avg_jaccard=adaptive_jaccard,
        probability_l1_delta=0.10,
        probability_max_delta=0.02,
        changed_probability_count=4,
        changed_set_count=2,
        profile_applied=True,
        profile_revision=1,
        profile_sample_size=10,
        elapsed_seconds=0.01,
    )


def test_window_evaluation_from_replay() -> None:
    rows = (
        make_row(
            round_no=1200,
            noop_best=1,
            adaptive_best=2,
            noop_practical=1,
            adaptive_practical=2,
            adaptive_jaccard=0.20,
        ),
        make_row(
            round_no=1201,
            noop_best=2,
            adaptive_best=2,
            noop_practical=1,
            adaptive_practical=1,
            adaptive_jaccard=0.30,
        ),
        make_row(
            round_no=1202,
            noop_best=2,
            adaptive_best=1,
            noop_practical=2,
            adaptive_practical=1,
            adaptive_jaccard=0.25,
        ),
    )

    result = window_evaluation_from_replay(
        window=EvaluationWindow(
            name="recent",
            start_round=1200,
            end_round=1202,
        ),
        rows=rows,
    )

    assert result.round_count == 3
    assert result.average_best_hits == pytest.approx(
        5 / 3
    )
    assert result.average_practical_hits == pytest.approx(
        4 / 3
    )
    assert result.baseline_best_hit_delta == pytest.approx(
        0.0
    )
    assert result.baseline_practical_hit_delta == pytest.approx(
        0.0
    )
    assert result.best_win_count == 1
    assert result.best_loss_count == 1
    assert result.best_tie_count == 1
    assert result.practical_win_count == 1
    assert result.practical_loss_count == 1
    assert result.practical_tie_count == 1
    assert result.average_jaccard == pytest.approx(
        0.25
    )


def test_replay_rows_must_match_window_exactly() -> None:
    rows = (
        make_row(
            round_no=1200,
            noop_best=1,
            adaptive_best=2,
            noop_practical=1,
            adaptive_practical=2,
            adaptive_jaccard=0.20,
        ),
        make_row(
            round_no=1202,
            noop_best=1,
            adaptive_best=2,
            noop_practical=1,
            adaptive_practical=2,
            adaptive_jaccard=0.20,
        ),
    )

    with pytest.raises(
        ContractError,
        match="exactly match evaluation window",
    ):
        window_evaluation_from_replay(
            window=EvaluationWindow(
                name="recent",
                start_round=1200,
                end_round=1201,
            ),
            rows=rows,
        )


def test_replay_rows_must_not_be_empty() -> None:
    with pytest.raises(
        ContractError,
        match="rows must not be empty",
    ):
        window_evaluation_from_replay(
            window=EvaluationWindow(
                name="recent",
                start_round=1200,
                end_round=1200,
            ),
            rows=(),
        )

