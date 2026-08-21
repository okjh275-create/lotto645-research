from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from lrp.evaluation import EvaluationWindow
from lrp.evaluation.topk_replay_adapter import TopKReplayPrediction
from lrp.evaluation.topk_replay_evaluation import (
    TopKReplayEvaluationRequest,
    TopKReplayEvaluationResult,
)
from lrp.evaluation.topk_walkforward import (
    HitDistribution,
    TopKEvaluation,
    WalkForwardEvaluation,
    WalkForwardRoundEvaluation,
)


class Draw:
    def __init__(
        self,
        round_no: int = 1200,
        numbers: tuple[int, ...] = (
            1, 2, 3, 4, 5, 6
        ),
    ) -> None:
        self.round_no = round_no
        self.numbers = numbers


def _prediction(
    *,
    round_no: int = 1200,
    model_name: str = "candidate",
) -> TopKReplayPrediction:
    return TopKReplayPrediction(
        round_no=round_no,
        history_rounds=(
            round_no - 3,
            round_no - 2,
            round_no - 1,
        ),
        predictions=(
            (1, 2, 3, 4, 5, 6),
            (7, 8, 9, 10, 11, 12),
            (13, 14, 15, 16, 17, 18),
            (19, 20, 21, 22, 23, 24),
            (25, 26, 27, 28, 29, 30),
            (31, 32, 33, 34, 35, 36),
            (37, 38, 39, 40, 41, 42),
            (1, 2, 3, 43, 44, 45),
            (10, 11, 12, 13, 14, 15),
            (20, 21, 22, 23, 24, 25),
        ),
        model_name=model_name,
        regime_id="R1",
        strategy_name="S1",
    )


def _evaluation() -> WalkForwardEvaluation:
    def distribution(
        count: int,
    ) -> HitDistribution:
        return HitDistribution(
            hit_0=count,
            hit_1=0,
            hit_2=0,
            hit_3=0,
            hit_4=0,
            hit_5=0,
            hit_6=0,
        )

    def topk(
        k: int,
    ) -> TopKEvaluation:
        return TopKEvaluation(
            k=k,
            round_count=1,
            set_count=k,
            mean_best_hits=0.0,
            mean_set_hits=0.0,
            best_hit_distribution=(
                distribution(1)
            ),
            set_hit_distribution=(
                distribution(k)
            ),
            baseline_delta_mean_best_hits=0.0,
            baseline_delta_3plus_rate=0.0,
            baseline_delta_4plus_rate=0.0,
        )

    top3 = topk(3)
    top5 = topk(5)
    top10 = topk(10)

    round_row = WalkForwardRoundEvaluation(
        round_no=1200,
        history_end_round=1199,
        actual_numbers=(
            1, 2, 3, 4, 5, 6
        ),
        model_name="candidate",
        regime_id="R1",
        strategy_name="S1",
        top3=top3,
        top5=top5,
        top10=top10,
    )

    return WalkForwardEvaluation(
        window=EvaluationWindow(
            name="ac04-model",
            start_round=1200,
            end_round=1200,
        ),
        rounds=(
            round_row,
        ),
        top3=top3,
        top5=top5,
        top10=top10,
        model_name="candidate",
        regime_slices=(),
        strategy_slices=(),
    )


def test_replay_evaluation_request_contract() -> None:
    request = TopKReplayEvaluationRequest(
        window=EvaluationWindow(
            name="ac04",
            start_round=1200,
            end_round=1200,
        ),
        candidate_predictions=(
            _prediction(
                model_name="candidate"
            ),
        ),
        baseline_predictions=(
            _prediction(
                model_name="baseline"
            ),
        ),
        actual_draws=(
            Draw(),
        ),
    )

    assert [
        field.name
        for field in fields(
            TopKReplayEvaluationRequest
        )
    ] == [
        "window",
        "candidate_predictions",
        "baseline_predictions",
        "actual_draws",
    ]

    assert request.window.name == "ac04"
    assert len(request.candidate_predictions) == 1
    assert len(request.baseline_predictions) == 1
    assert len(request.actual_draws) == 1


def test_replay_evaluation_request_is_immutable() -> None:
    request = TopKReplayEvaluationRequest(
        window=EvaluationWindow(
            name="ac04",
            start_round=1200,
            end_round=1200,
        ),
        candidate_predictions=(
            _prediction(
                model_name="candidate"
            ),
        ),
        baseline_predictions=(
            _prediction(
                model_name="baseline"
            ),
        ),
        actual_draws=(
            Draw(),
        ),
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        request.window = EvaluationWindow(  # type: ignore[misc]
            name="mutated",
            start_round=1200,
            end_round=1200,
        )


def test_replay_evaluation_result_contract() -> None:
    result = TopKReplayEvaluationResult(
        evaluation=_evaluation(),
        candidate_model_name="candidate",
        baseline_model_name="baseline",
        round_count=1,
    )

    assert [
        field.name
        for field in fields(
            TopKReplayEvaluationResult
        )
    ] == [
        "evaluation",
        "candidate_model_name",
        "baseline_model_name",
        "round_count",
    ]

    assert result.candidate_model_name == "candidate"
    assert result.baseline_model_name == "baseline"
    assert result.round_count == 1


def test_replay_evaluation_result_is_immutable() -> None:
    result = TopKReplayEvaluationResult(
        evaluation=_evaluation(),
        candidate_model_name="candidate",
        baseline_model_name="baseline",
        round_count=1,
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        result.round_count = 2  # type: ignore[misc]
