from __future__ import annotations

from dataclasses import dataclass

import pytest

from lrp.contracts import ContractError
from lrp.evaluation import EvaluationWindow
from lrp.evaluation.topk_replay_adapter import (
    TopKReplayPrediction,
)
from lrp.evaluation.topk_replay_evaluation import (
    TopKReplayEvaluationRequest,
    TopKReplayEvaluationService,
)


@dataclass(frozen=True)
class Draw:
    round_no: int
    numbers: tuple[int, ...]


def _prediction_sets() -> tuple[
    tuple[int, ...],
    ...
]:
    return (
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
    )


def _prediction(
    round_no: int,
    *,
    model_name: str,
    regime_id: str | None = "R1",
    strategy_name: str | None = "S1",
) -> TopKReplayPrediction:
    return TopKReplayPrediction(
        round_no=round_no,
        history_rounds=(
            round_no - 3,
            round_no - 2,
            round_no - 1,
        ),
        predictions=_prediction_sets(),
        model_name=model_name,
        regime_id=regime_id,
        strategy_name=strategy_name,
    )


def _draw(
    round_no: int,
) -> Draw:
    return Draw(
        round_no=round_no,
        numbers=(
            1, 2, 3, 4, 5, 6
        ),
    )


def _request(
    *,
    candidate_rounds: tuple[int, ...] = (
        1200,
    ),
    baseline_rounds: tuple[int, ...] = (
        1200,
    ),
    candidate_models: tuple[str, ...] | None = None,
    baseline_models: tuple[str, ...] | None = None,
    draw_rounds: tuple[int, ...] = (
        1200,
    ),
    window: EvaluationWindow | None = None,
) -> TopKReplayEvaluationRequest:
    if candidate_models is None:
        candidate_models = tuple(
            "candidate"
            for _ in candidate_rounds
        )

    if baseline_models is None:
        baseline_models = tuple(
            "baseline"
            for _ in baseline_rounds
        )

    if window is None:
        all_rounds = (
            candidate_rounds
            + baseline_rounds
        )

        start_round = min(
            all_rounds
        ) if all_rounds else 1200

        end_round = max(
            all_rounds
        ) if all_rounds else 1200

        window = EvaluationWindow(
            name="ac04",
            start_round=start_round,
            end_round=end_round,
        )

    return TopKReplayEvaluationRequest(
        window=window,
        candidate_predictions=tuple(
            _prediction(
                round_no,
                model_name=model_name,
            )
            for round_no, model_name
            in zip(
                candidate_rounds,
                candidate_models,
            )
        ),
        baseline_predictions=tuple(
            _prediction(
                round_no,
                model_name=model_name,
            )
            for round_no, model_name
            in zip(
                baseline_rounds,
                baseline_models,
            )
        ),
        actual_draws=tuple(
            _draw(
                round_no
            )
            for round_no in draw_rounds
        ),
    )


def test_service_evaluates_single_round() -> None:
    result = TopKReplayEvaluationService().evaluate(
        request=_request()
    )

    assert result.round_count == 1
    assert len(result.evaluation.rounds) == 1
    assert result.evaluation.rounds[0].round_no == 1200


def test_service_evaluates_multiple_rounds() -> None:
    result = TopKReplayEvaluationService().evaluate(
        request=_request(
            candidate_rounds=(
                1200,
                1201,
                1202,
            ),
            baseline_rounds=(
                1200,
                1201,
                1202,
            ),
            draw_rounds=(
                1200,
                1201,
                1202,
            ),
        )
    )

    assert result.round_count == 3

    assert tuple(
        row.round_no
        for row in result.evaluation.rounds
    ) == (
        1200,
        1201,
        1202,
    )


def test_service_adapts_candidate_predictions() -> None:
    result = TopKReplayEvaluationService().evaluate(
        request=_request()
    )

    assert result.evaluation.model_name == "candidate"
    assert result.evaluation.rounds[0].model_name == "candidate"


def test_service_adapts_baseline_predictions() -> None:
    result = TopKReplayEvaluationService().evaluate(
        request=_request()
    )

    assert result.baseline_model_name == "baseline"


def test_service_constructs_baseline_provider() -> None:
    result = TopKReplayEvaluationService().evaluate(
        request=_request()
    )

    assert result.round_count == 1
    assert result.evaluation.top3 is not None
    assert result.evaluation.top5 is not None
    assert result.evaluation.top10 is not None


def test_service_invokes_walkforward_evaluator() -> None:
    result = TopKReplayEvaluationService().evaluate(
        request=_request()
    )

    assert result.evaluation.window.name == "ac04"
    assert len(result.evaluation.rounds) == 1


def test_service_preserves_candidate_model_identity() -> None:
    result = TopKReplayEvaluationService().evaluate(
        request=_request()
    )

    assert result.candidate_model_name == "candidate"
    assert result.evaluation.model_name == "candidate"


def test_service_preserves_baseline_model_identity() -> None:
    result = TopKReplayEvaluationService().evaluate(
        request=_request()
    )

    assert result.baseline_model_name == "baseline"


def test_service_forwards_window() -> None:
    window = EvaluationWindow(
        name="exact-window",
        start_round=1200,
        end_round=1200,
    )

    request = _request(
        window=window
    )

    result = TopKReplayEvaluationService().evaluate(
        request=request
    )

    assert result.evaluation.window == window


def test_service_derives_round_count() -> None:
    result = TopKReplayEvaluationService().evaluate(
        request=_request(
            candidate_rounds=(
                1200,
                1202,
            ),
            baseline_rounds=(
                1200,
                1202,
            ),
            draw_rounds=(
                1200,
                1202,
            ),
            window=EvaluationWindow(
                name="sparse",
                start_round=1200,
                end_round=1202,
            ),
        )
    )

    assert result.round_count == len(
        result.evaluation.rounds
    )

    assert result.round_count == 2


def test_service_rejects_empty_candidate_predictions() -> None:
    with pytest.raises(
        ContractError
    ):
        TopKReplayEvaluationService().evaluate(
            request=_request(
                candidate_rounds=(),
                baseline_rounds=(
                    1200,
                ),
                draw_rounds=(
                    1200,
                ),
            )
        )


def test_service_rejects_empty_baseline_predictions() -> None:
    with pytest.raises(
        ContractError
    ):
        TopKReplayEvaluationService().evaluate(
            request=_request(
                candidate_rounds=(
                    1200,
                ),
                baseline_rounds=(),
                draw_rounds=(
                    1200,
                ),
            )
        )


def test_service_rejects_candidate_baseline_round_mismatch() -> None:
    with pytest.raises(
        ContractError
    ):
        TopKReplayEvaluationService().evaluate(
            request=_request(
                candidate_rounds=(
                    1200,
                    1201,
                ),
                baseline_rounds=(
                    1200,
                    1202,
                ),
                draw_rounds=(
                    1200,
                    1201,
                    1202,
                ),
                window=EvaluationWindow(
                    name="mismatch",
                    start_round=1200,
                    end_round=1202,
                ),
            )
        )


def test_service_rejects_round_outside_window() -> None:
    with pytest.raises(
        ContractError
    ):
        TopKReplayEvaluationService().evaluate(
            request=_request(
                candidate_rounds=(
                    1200,
                    1201,
                ),
                baseline_rounds=(
                    1200,
                    1201,
                ),
                draw_rounds=(
                    1200,
                    1201,
                ),
                window=EvaluationWindow(
                    name="narrow",
                    start_round=1200,
                    end_round=1200,
                ),
            )
        )


def test_service_rejects_missing_actual_draw() -> None:
    with pytest.raises(
        ContractError
    ):
        TopKReplayEvaluationService().evaluate(
            request=_request(
                candidate_rounds=(
                    1200,
                    1201,
                ),
                baseline_rounds=(
                    1200,
                    1201,
                ),
                draw_rounds=(
                    1200,
                ),
            )
        )


def test_service_rejects_mixed_candidate_model_names() -> None:
    with pytest.raises(
        ContractError
    ):
        TopKReplayEvaluationService().evaluate(
            request=_request(
                candidate_rounds=(
                    1200,
                    1201,
                ),
                baseline_rounds=(
                    1200,
                    1201,
                ),
                candidate_models=(
                    "candidate-a",
                    "candidate-b",
                ),
                baseline_models=(
                    "baseline",
                    "baseline",
                ),
                draw_rounds=(
                    1200,
                    1201,
                ),
            )
        )


def test_service_rejects_mixed_baseline_model_names() -> None:
    with pytest.raises(
        ContractError
    ):
        TopKReplayEvaluationService().evaluate(
            request=_request(
                candidate_rounds=(
                    1200,
                    1201,
                ),
                baseline_rounds=(
                    1200,
                    1201,
                ),
                candidate_models=(
                    "candidate",
                    "candidate",
                ),
                baseline_models=(
                    "baseline-a",
                    "baseline-b",
                ),
                draw_rounds=(
                    1200,
                    1201,
                ),
            )
        )
