from __future__ import annotations

from dataclasses import dataclass

import pytest

from lrp.evaluation import (
    EvaluationWindow,
    WindowEvaluation,
)


@dataclass(frozen=True)
class FakeWindowResult:
    model_name: str
    window: EvaluationWindow
    evaluation: WindowEvaluation


def make_window(
    name: str,
    start_round: int,
    end_round: int,
) -> EvaluationWindow:
    return EvaluationWindow(
        name=name,
        start_round=start_round,
        end_round=end_round,
    )


def make_evaluation(
    *,
    window: EvaluationWindow,
    practical_delta: float,
    best_delta: float,
) -> WindowEvaluation:
    round_count = window.round_count

    return WindowEvaluation(
        window=window,
        round_count=round_count,
        average_best_hits=2.0 + best_delta,
        average_practical_hits=1.0 + practical_delta,
        baseline_best_hit_delta=best_delta,
        baseline_practical_hit_delta=practical_delta,
        best_hit_stddev=0.30,
        practical_hit_stddev=0.25,
        best_win_count=4,
        best_loss_count=3,
        best_tie_count=round_count - 7,
        best_sign_test_pvalue=0.20,
        practical_win_count=4,
        practical_loss_count=3,
        practical_tie_count=round_count - 7,
        practical_sign_test_pvalue=0.20,
        average_jaccard=0.20,
    )


def test_runner_executes_every_model_window_pair() -> None:
    from tools.validation.model_evaluation_runner import (
        HistoricalModelEvaluationRunner,
    )

    windows = (
        make_window("short", 1200, 1210),
        make_window("mid", 1180, 1210),
    )

    calls: list[tuple[str, str]] = []

    def evaluator(
        model_name: str,
        window: EvaluationWindow,
    ) -> WindowEvaluation:
        calls.append((model_name, window.name))

        bonus = (
            0.10
            if model_name == "candidate"
            else 0.00
        )

        return make_evaluation(
            window=window,
            practical_delta=bonus,
            best_delta=bonus,
        )

    runner = HistoricalModelEvaluationRunner(
        evaluator=evaluator,
    )

    result = runner.run(
        model_names=("baseline", "candidate"),
        windows=windows,
    )

    assert calls == [
        ("baseline", "short"),
        ("baseline", "mid"),
        ("candidate", "short"),
        ("candidate", "mid"),
    ]

    assert result.windows == windows
    assert len(result.evaluations) == 2


def test_runner_builds_ranked_historical_matrix() -> None:
    from tools.validation.model_evaluation_runner import (
        HistoricalModelEvaluationRunner,
    )

    windows = (
        make_window("w1", 1200, 1210),
        make_window("w2", 1211, 1220),
    )

    def evaluator(
        model_name: str,
        window: EvaluationWindow,
    ) -> WindowEvaluation:
        if model_name == "candidate":
            return make_evaluation(
                window=window,
                practical_delta=0.40,
                best_delta=0.40,
            )

        return make_evaluation(
            window=window,
            practical_delta=0.00,
            best_delta=0.00,
        )

    result = HistoricalModelEvaluationRunner(
        evaluator=evaluator,
    ).run(
        model_names=("baseline", "candidate"),
        windows=windows,
    )

    assert result.ranking.entries[0].model_name == (
        "candidate"
    )

    assert {
        item.model_name
        for item in result.evaluations
    } == {
        "baseline",
        "candidate",
    }


def test_runner_is_deterministic() -> None:
    from tools.validation.model_evaluation_runner import (
        HistoricalModelEvaluationRunner,
    )

    windows = (
        make_window("w1", 1200, 1210),
    )

    def evaluator(
        model_name: str,
        window: EvaluationWindow,
    ) -> WindowEvaluation:
        del model_name

        return make_evaluation(
            window=window,
            practical_delta=0.00,
            best_delta=0.00,
        )

    runner = HistoricalModelEvaluationRunner(
        evaluator=evaluator,
    )

    first = runner.run(
        model_names=("baseline", "candidate"),
        windows=windows,
    )

    second = runner.run(
        model_names=("baseline", "candidate"),
        windows=windows,
    )

    assert first == second


def test_runner_rejects_empty_models() -> None:
    from tools.validation.model_evaluation_runner import (
        HistoricalModelEvaluationRunner,
    )

    runner = HistoricalModelEvaluationRunner(
        evaluator=lambda model_name, window: (
            make_evaluation(
                window=window,
                practical_delta=0.00,
                best_delta=0.00,
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="model_names must not be empty",
    ):
        runner.run(
            model_names=(),
            windows=(
                make_window(
                    "w1",
                    1200,
                    1210,
                ),
            ),
        )


def test_runner_rejects_empty_windows() -> None:
    from tools.validation.model_evaluation_runner import (
        HistoricalModelEvaluationRunner,
    )

    runner = HistoricalModelEvaluationRunner(
        evaluator=lambda model_name, window: (
            make_evaluation(
                window=window,
                practical_delta=0.00,
                best_delta=0.00,
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="windows must not be empty",
    ):
        runner.run(
            model_names=("baseline",),
            windows=(),
        )


def test_runner_rejects_duplicate_model_names() -> None:
    from tools.validation.model_evaluation_runner import (
        HistoricalModelEvaluationRunner,
    )

    runner = HistoricalModelEvaluationRunner(
        evaluator=lambda model_name, window: (
            make_evaluation(
                window=window,
                practical_delta=0.00,
                best_delta=0.00,
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="model_names must be unique",
    ):
        runner.run(
            model_names=("baseline", "baseline"),
            windows=(
                make_window(
                    "w1",
                    1200,
                    1210,
                ),
            ),
        )


def test_runner_rejects_non_callable_evaluator() -> None:
    from tools.validation.model_evaluation_runner import (
        HistoricalModelEvaluationRunner,
    )

    with pytest.raises(
        TypeError,
        match="evaluator must be callable",
    ):
        HistoricalModelEvaluationRunner(
            evaluator=object(),
        )




def test_runner_rejects_non_window_evaluator_result() -> None:
    from tools.validation.model_evaluation_runner import (
        HistoricalModelEvaluationRunner,
    )

    runner = HistoricalModelEvaluationRunner(
        evaluator=lambda model_name, window: object(),
    )

    with pytest.raises(
        TypeError,
        match="evaluator must return WindowEvaluation",
    ):
        runner.run(
            model_names=("baseline",),
            windows=(
                make_window(
                    "w1",
                    1200,
                    1210,
                ),
            ),
        )


def test_runner_rejects_mismatched_evaluation_window() -> None:
    from tools.validation.model_evaluation_runner import (
        HistoricalModelEvaluationRunner,
    )

    requested = make_window(
        "requested",
        1200,
        1210,
    )

    other = make_window(
        "other",
        1211,
        1221,
    )

    runner = HistoricalModelEvaluationRunner(
        evaluator=lambda model_name, window: (
            make_evaluation(
                window=other,
                practical_delta=0.0,
                best_delta=0.0,
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="result window must match",
    ):
        runner.run(
            model_names=("baseline",),
            windows=(requested,),
        )


def test_runner_rejects_blank_model_name() -> None:
    from tools.validation.model_evaluation_runner import (
        HistoricalModelEvaluationRunner,
    )

    runner = HistoricalModelEvaluationRunner(
        evaluator=lambda model_name, window: (
            make_evaluation(
                window=window,
                practical_delta=0.0,
                best_delta=0.0,
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="non-empty strings",
    ):
        runner.run(
            model_names=("baseline", " "),
            windows=(
                make_window(
                    "w1",
                    1200,
                    1210,
                ),
            ),
        )
