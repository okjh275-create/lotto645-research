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


class DrawWithoutNumbers:
    def __init__(
        self,
        round_no: int,
    ) -> None:
        self.round_no = round_no


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
        regime_id="R1",
        strategy_name="S1",
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
    candidate_predictions: tuple[
        TopKReplayPrediction,
        ...,
    ] | tuple[object, ...] | None = None,
    baseline_predictions: tuple[
        TopKReplayPrediction,
        ...,
    ] | tuple[object, ...] | None = None,
    actual_draws: tuple[object, ...] | None = None,
    window: EvaluationWindow | None = None,
) -> TopKReplayEvaluationRequest:

    if candidate_predictions is None:
        candidate_predictions = (
            _prediction(
                1200,
                model_name="candidate",
            ),
        )

    if baseline_predictions is None:
        baseline_predictions = (
            _prediction(
                1200,
                model_name="baseline",
            ),
        )

    if actual_draws is None:
        actual_draws = (
            _draw(1200),
        )

    if window is None:
        window = EvaluationWindow(
            name="ac06",
            start_round=1200,
            end_round=1200,
        )

    return TopKReplayEvaluationRequest(
        window=window,
        candidate_predictions=candidate_predictions,  # type: ignore[arg-type]
        baseline_predictions=baseline_predictions,  # type: ignore[arg-type]
        actual_draws=actual_draws,
    )


def test_service_rejects_duplicate_candidate_round() -> None:
    request = _request(
        candidate_predictions=(
            _prediction(
                1200,
                model_name="candidate",
            ),
            _prediction(
                1200,
                model_name="candidate",
            ),
        )
    )

    with pytest.raises(
        ContractError
    ):
        TopKReplayEvaluationService().evaluate(
            request=request
        )


def test_service_rejects_duplicate_baseline_round() -> None:
    request = _request(
        baseline_predictions=(
            _prediction(
                1200,
                model_name="baseline",
            ),
            _prediction(
                1200,
                model_name="baseline",
            ),
        )
    )

    with pytest.raises(
        ContractError
    ):
        TopKReplayEvaluationService().evaluate(
            request=request
        )


def test_service_rejects_duplicate_actual_draw_round() -> None:
    request = _request(
        actual_draws=(
            _draw(1200),
            _draw(1200),
        )
    )

    with pytest.raises(
        ContractError
    ):
        TopKReplayEvaluationService().evaluate(
            request=request
        )


def test_service_rejects_actual_draw_missing_numbers() -> None:
    request = _request(
        actual_draws=(
            DrawWithoutNumbers(
                1200
            ),
        )
    )

    with pytest.raises(
        ContractError
    ):
        TopKReplayEvaluationService().evaluate(
            request=request
        )


def test_service_rejects_wrong_candidate_item_type() -> None:
    request = _request(
        candidate_predictions=(
            object(),
        )
    )

    with pytest.raises(
        ContractError
    ):
        TopKReplayEvaluationService().evaluate(
            request=request
        )


def test_service_rejects_wrong_baseline_item_type() -> None:
    request = _request(
        baseline_predictions=(
            object(),
        )
    )

    with pytest.raises(
        ContractError
    ):
        TopKReplayEvaluationService().evaluate(
            request=request
        )


def test_service_rejects_non_tuple_candidate_source() -> None:
    valid = _prediction(
        1200,
        model_name="candidate",
    )

    request = TopKReplayEvaluationRequest(
        window=EvaluationWindow(
            name="ac06-invalid-candidate-source",
            start_round=1200,
            end_round=1200,
        ),
        candidate_predictions=[  # type: ignore[arg-type]
            valid
        ],
        baseline_predictions=(
            _prediction(
                1200,
                model_name="baseline",
            ),
        ),
        actual_draws=(
            _draw(1200),
        ),
    )

    with pytest.raises(
        ContractError
    ):
        TopKReplayEvaluationService().evaluate(
            request=request
        )



def test_service_rejects_non_tuple_baseline_source() -> None:
    valid = _prediction(
        1200,
        model_name="baseline",
    )

    request = TopKReplayEvaluationRequest(
        window=EvaluationWindow(
            name="ac06-invalid-baseline-source",
            start_round=1200,
            end_round=1200,
        ),
        candidate_predictions=(
            _prediction(
                1200,
                model_name="candidate",
            ),
        ),
        baseline_predictions=[  # type: ignore[arg-type]
            valid
        ],
        actual_draws=(
            _draw(1200),
        ),
    )

    with pytest.raises(
        ContractError
    ):
        TopKReplayEvaluationService().evaluate(
            request=request
        )



def test_reversed_prediction_order_is_normalized() -> None:
    request = _request(
        candidate_predictions=(
            _prediction(
                1202,
                model_name="candidate",
            ),
            _prediction(
                1200,
                model_name="candidate",
            ),
        ),
        baseline_predictions=(
            _prediction(
                1202,
                model_name="baseline",
            ),
            _prediction(
                1200,
                model_name="baseline",
            ),
        ),
        actual_draws=(
            _draw(1200),
            _draw(1202),
        ),
        window=EvaluationWindow(
            name="ac06-reversed",
            start_round=1200,
            end_round=1202,
        ),
    )

    result = TopKReplayEvaluationService().evaluate(
        request=request
    )

    assert tuple(
        row.round_no
        for row in result.evaluation.rounds
    ) == (
        1200,
        1202,
    )

    assert result.round_count == 2


def test_reversed_actual_draw_order_is_semantically_stable() -> None:
    candidate = (
        _prediction(
            1200,
            model_name="candidate",
        ),
        _prediction(
            1201,
            model_name="candidate",
        ),
    )

    baseline = (
        _prediction(
            1200,
            model_name="baseline",
        ),
        _prediction(
            1201,
            model_name="baseline",
        ),
    )

    window = EvaluationWindow(
        name="ac06-draw-order",
        start_round=1200,
        end_round=1201,
    )

    first = TopKReplayEvaluationService().evaluate(
        request=_request(
            candidate_predictions=candidate,
            baseline_predictions=baseline,
            actual_draws=(
                _draw(1200),
                _draw(1201),
            ),
            window=window,
        )
    )

    second = TopKReplayEvaluationService().evaluate(
        request=_request(
            candidate_predictions=candidate,
            baseline_predictions=baseline,
            actual_draws=(
                _draw(1201),
                _draw(1200),
            ),
            window=window,
        )
    )

    assert first == second


def test_extra_actual_draw_is_allowed() -> None:
    request = _request(
        actual_draws=(
            _draw(1199),
            _draw(1200),
            _draw(1201),
        )
    )

    result = TopKReplayEvaluationService().evaluate(
        request=request
    )

    assert result.round_count == 1

    assert (
        result.evaluation.rounds[0].round_no
        == 1200
    )


def test_same_candidate_and_baseline_model_name_is_allowed() -> None:
    request = _request(
        candidate_predictions=(
            _prediction(
                1200,
                model_name="shared",
            ),
        ),
        baseline_predictions=(
            _prediction(
                1200,
                model_name="shared",
            ),
        ),
    )

    result = TopKReplayEvaluationService().evaluate(
        request=request
    )

    assert result.candidate_model_name == "shared"
    assert result.baseline_model_name == "shared"
    assert result.round_count == 1


def test_sparse_matching_rounds_inside_window_are_allowed() -> None:
    request = _request(
        candidate_predictions=(
            _prediction(
                1200,
                model_name="candidate",
            ),
            _prediction(
                1202,
                model_name="candidate",
            ),
        ),
        baseline_predictions=(
            _prediction(
                1200,
                model_name="baseline",
            ),
            _prediction(
                1202,
                model_name="baseline",
            ),
        ),
        actual_draws=(
            _draw(1200),
            _draw(1202),
        ),
        window=EvaluationWindow(
            name="ac06-sparse",
            start_round=1200,
            end_round=1202,
        ),
    )

    result = TopKReplayEvaluationService().evaluate(
        request=request
    )

    assert result.round_count == 2

    assert tuple(
        row.round_no
        for row in result.evaluation.rounds
    ) == (
        1200,
        1202,
    )


def test_service_rejects_invalid_request_type() -> None:
    with pytest.raises(
        ContractError
    ):
        TopKReplayEvaluationService().evaluate(
            request=object()  # type: ignore[arg-type]
        )


def test_repeated_evaluation_is_semantically_stable() -> None:
    service = TopKReplayEvaluationService()

    request = _request()

    first = service.evaluate(
        request=request
    )

    second = service.evaluate(
        request=request
    )

    assert first == second
    assert repr(first) == repr(second)


def test_result_round_count_matches_evaluation_rounds() -> None:
    request = _request(
        candidate_predictions=(
            _prediction(
                1200,
                model_name="candidate",
            ),
            _prediction(
                1201,
                model_name="candidate",
            ),
            _prediction(
                1202,
                model_name="candidate",
            ),
        ),
        baseline_predictions=(
            _prediction(
                1200,
                model_name="baseline",
            ),
            _prediction(
                1201,
                model_name="baseline",
            ),
            _prediction(
                1202,
                model_name="baseline",
            ),
        ),
        actual_draws=(
            _draw(1200),
            _draw(1201),
            _draw(1202),
        ),
        window=EvaluationWindow(
            name="ac06-count",
            start_round=1200,
            end_round=1202,
        ),
    )

    result = TopKReplayEvaluationService().evaluate(
        request=request
    )

    assert result.round_count == len(
        result.evaluation.rounds
    )

    assert result.round_count == 3
