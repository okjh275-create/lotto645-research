from __future__ import annotations

import pytest

from lrp.contracts import ContractError
from lrp.evaluation import (
    EvaluationWindow,
    WindowEvaluation,
)

from tools.validation.historical_replay_models import (
    ReplayRoundResult,
)

from tools.validation.model_replay_evaluator import (
    HistoricalModelReplayEvaluator,
)


def make_row(
    *,
    round_no: int,
    adaptive_best_hits: int = 2,
    adaptive_practical_hits: int = 2,
    noop_best_hits: int = 1,
    noop_practical_hits: int = 1,
) -> ReplayRoundResult:
    return ReplayRoundResult(
        round_no=round_no,
        seed=20260802 + round_no,
        history_draws=100,
        noop_best_hits=noop_best_hits,
        adaptive_best_hits=adaptive_best_hits,
        noop_practical_hits=noop_practical_hits,
        adaptive_practical_hits=adaptive_practical_hits,
        noop_avg_jaccard=0.20,
        adaptive_avg_jaccard=0.18,
        probability_l1_delta=0.10,
        probability_max_delta=0.05,
        changed_probability_count=10,
        changed_set_count=3,
        profile_applied=True,
        profile_revision=1,
        profile_sample_size=10,
    )


def test_replay_evaluator_builds_window_evaluation() -> None:
    window = EvaluationWindow(
        name="recent",
        start_round=1220,
        end_round=1222,
    )

    calls: list[
        tuple[str, EvaluationWindow]
    ] = []

    def replay_rows(
        model_name: str,
        requested_window: EvaluationWindow,
    ) -> tuple[ReplayRoundResult, ...]:
        calls.append(
            (
                model_name,
                requested_window,
            )
        )

        return (
            make_row(round_no=1220),
            make_row(round_no=1221),
            make_row(round_no=1222),
        )

    evaluator = HistoricalModelReplayEvaluator(
        replay_rows=replay_rows,
    )

    result = evaluator(
        "combined",
        window,
    )

    assert isinstance(
        result,
        WindowEvaluation,
    )

    assert result.window == window
    assert result.round_count == 3

    assert result.average_best_hits == pytest.approx(
        2.0
    )

    assert result.average_practical_hits == pytest.approx(
        2.0
    )

    assert result.baseline_best_hit_delta == pytest.approx(
        1.0
    )

    assert (
        result.baseline_practical_hit_delta
        == pytest.approx(1.0)
    )

    assert calls == [
        (
            "combined",
            window,
        )
    ]


def test_replay_evaluator_rejects_blank_model_name() -> None:
    evaluator = HistoricalModelReplayEvaluator(
        replay_rows=lambda model_name, window: (),
    )

    with pytest.raises(
        ValueError,
        match="model_name",
    ):
        evaluator(
            " ",
            EvaluationWindow(
                name="recent",
                start_round=1220,
                end_round=1222,
            ),
        )


def test_replay_evaluator_requires_evaluation_window() -> None:
    evaluator = HistoricalModelReplayEvaluator(
        replay_rows=lambda model_name, window: (),
    )

    with pytest.raises(
        TypeError,
        match="EvaluationWindow",
    ):
        evaluator(
            "combined",
            object(),
        )


def test_replay_evaluator_rejects_empty_rows() -> None:
    window = EvaluationWindow(
        name="recent",
        start_round=1220,
        end_round=1222,
    )

    evaluator = HistoricalModelReplayEvaluator(
        replay_rows=lambda model_name, window: (),
    )

    with pytest.raises(
        ContractError,
        match="rows must not be empty",
    ):
        evaluator(
            "combined",
            window,
        )


def test_replay_evaluator_rejects_incomplete_window() -> None:
    window = EvaluationWindow(
        name="recent",
        start_round=1220,
        end_round=1222,
    )

    evaluator = HistoricalModelReplayEvaluator(
        replay_rows=lambda model_name, window: (
            make_row(round_no=1220),
            make_row(round_no=1221),
        ),
    )

    with pytest.raises(
        ContractError,
        match="exactly match evaluation window",
    ):
        evaluator(
            "combined",
            window,
        )


def test_replay_evaluator_rejects_wrong_row_type() -> None:
    window = EvaluationWindow(
        name="recent",
        start_round=1220,
        end_round=1220,
    )

    evaluator = HistoricalModelReplayEvaluator(
        replay_rows=lambda model_name, window: (
            object(),
        ),
    )

    with pytest.raises(
        TypeError,
        match="ReplayRoundResult",
    ):
        evaluator(
            "combined",
            window,
        )
