from __future__ import annotations

from types import SimpleNamespace

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.io.draws import HistoryRow
from lrp.pipelines.models import (
    PredictionGenerationResult,
    PredictionRequest,
    PredictionResult,
)
from lrp.evaluation.contracts import EvaluationWindow

from lrp.evaluation.topk_live_evaluation_orchestrator import (
    TopKLiveEvaluationOrchestrator,
    TopKLiveEvaluationRequest,
    TopKLiveEvaluationResult,
)


def _prediction_result(
    round_no: int = 1200,
) -> PredictionResult:
    request = PredictionRequest(
        round_no=round_no,
        seed=20260821,
        long_gap_numbers=frozenset(
            {
                1,
            }
        ),
    )

    generation = object.__new__(
        PredictionGenerationResult
    )

    object.__setattr__(
        generation,
        "request",
        request,
    )

    result = object.__new__(
        PredictionResult
    )

    object.__setattr__(
        result,
        "generation",
        generation,
    )

    selected_numbers = (
        (1, 2, 3, 4, 5, 6),
        (7, 8, 9, 10, 11, 12),
        (13, 14, 15, 16, 17, 18),
        (19, 20, 21, 22, 23, 24),
        (25, 26, 27, 28, 29, 30),
        (31, 32, 33, 34, 35, 36),
        (37, 38, 39, 40, 41, 42),
        (1, 8, 15, 22, 29, 36),
        (2, 9, 16, 23, 30, 37),
        (3, 10, 17, 24, 31, 38),
    )

    object.__setattr__(
        result,
        "diversity",
        SimpleNamespace(
            selected=tuple(
                SimpleNamespace(
                    numbers=numbers,
                )
                for numbers in selected_numbers
            )
        ),
    )

    return result


def _history() -> tuple[HistoryRow, ...]:
    return (
        HistoryRow(
            round_no=1197,
            numbers=(1, 2, 3, 4, 5, 6),
        ),
        HistoryRow(
            round_no=1198,
            numbers=(7, 8, 9, 10, 11, 12),
        ),
        HistoryRow(
            round_no=1199,
            numbers=(13, 14, 15, 16, 17, 18),
        ),
    )


def _draw(
    round_no: int = 1200,
) -> SimpleNamespace:
    return SimpleNamespace(
        round_no=round_no,
        numbers=(1, 2, 3, 4, 5, 6),
    )


def _window() -> EvaluationWindow:
    return EvaluationWindow(
        name="af04b",
        start_round=1200,
        end_round=1200,
    )


def _request(
    **overrides: object,
) -> TopKLiveEvaluationRequest:
    values: dict[str, object] = {
        "window":
            _window(),

        "candidate_prediction_result":
            _prediction_result(1200),

        "candidate_history_rows":
            _history(),

        "candidate_model_name":
            "candidate",

        "baseline_prediction_result":
            _prediction_result(1200),

        "baseline_history_rows":
            _history(),

        "baseline_model_name":
            "baseline",

        "actual_draws":
            (
                _draw(1200),
            ),

        "candidate_regime_id":
            "gap-recovery",

        "candidate_strategy_name":
            "candidate-main",

        "baseline_regime_id":
            "stable",

        "baseline_strategy_name":
            "baseline-main",
    }

    values.update(
        overrides
    )

    return TopKLiveEvaluationRequest(
        **values  # type: ignore[arg-type]
    )


def _evaluate(
    **overrides: object,
) -> TopKLiveEvaluationResult:
    return TopKLiveEvaluationOrchestrator().evaluate(
        request=_request(
            **overrides
        )
    )


# ---------------------------------------------------------------------------
# 21 failure / fail-closed contracts
# ---------------------------------------------------------------------------


def test_orchestrator_rejects_invalid_request_type() -> None:
    with pytest.raises(
        ContractError
    ):
        TopKLiveEvaluationOrchestrator().evaluate(
            request=object()  # type: ignore[arg-type]
        )


def test_rejects_wrong_candidate_prediction_result_type() -> None:
    with pytest.raises(
        ContractError
    ):
        _evaluate(
            candidate_prediction_result=object()
        )


def test_rejects_wrong_baseline_prediction_result_type() -> None:
    with pytest.raises(
        ContractError
    ):
        _evaluate(
            baseline_prediction_result=object()
        )


def test_rejects_non_tuple_candidate_history_rows() -> None:
    with pytest.raises(
        ContractError
    ):
        _evaluate(
            candidate_history_rows=list(
                _history()
            )
        )


def test_rejects_non_tuple_baseline_history_rows() -> None:
    with pytest.raises(
        ContractError
    ):
        _evaluate(
            baseline_history_rows=list(
                _history()
            )
        )


def test_rejects_invalid_candidate_history_item() -> None:
    with pytest.raises(
        ContractError
    ):
        _evaluate(
            candidate_history_rows=(
                object(),
            )
        )


def test_rejects_invalid_baseline_history_item() -> None:
    with pytest.raises(
        ContractError
    ):
        _evaluate(
            baseline_history_rows=(
                object(),
            )
        )


def test_rejects_non_tuple_actual_draws() -> None:
    with pytest.raises(
        ContractError
    ):
        _evaluate(
            actual_draws=[
                _draw()
            ]
        )


def test_rejects_actual_draw_missing_round_no() -> None:
    with pytest.raises(
        ContractError
    ):
        _evaluate(
            actual_draws=(
                SimpleNamespace(
                    numbers=(1, 2, 3, 4, 5, 6)
                ),
            )
        )


def test_rejects_actual_draw_missing_numbers() -> None:
    with pytest.raises(
        ContractError
    ):
        _evaluate(
            actual_draws=(
                SimpleNamespace(
                    round_no=1200
                ),
            )
        )


def test_rejects_blank_candidate_model_name() -> None:
    with pytest.raises(
        ContractError
    ):
        _evaluate(
            candidate_model_name="   "
        )


def test_rejects_blank_baseline_model_name() -> None:
    with pytest.raises(
        ContractError
    ):
        _evaluate(
            baseline_model_name="   "
        )


def test_rejects_blank_candidate_regime_id() -> None:
    with pytest.raises(
        ContractError
    ):
        _evaluate(
            candidate_regime_id="   "
        )


def test_rejects_blank_baseline_regime_id() -> None:
    with pytest.raises(
        ContractError
    ):
        _evaluate(
            baseline_regime_id="   "
        )


def test_rejects_blank_candidate_strategy_name() -> None:
    with pytest.raises(
        ContractError
    ):
        _evaluate(
            candidate_strategy_name="   "
        )


def test_rejects_blank_baseline_strategy_name() -> None:
    with pytest.raises(
        ContractError
    ):
        _evaluate(
            baseline_strategy_name="   "
        )


def test_rejects_candidate_history_current_round() -> None:
    with pytest.raises(
        ContractError
    ):
        _evaluate(
            candidate_history_rows=(
                *_history(),
                HistoryRow(
                    round_no=1200,
                    numbers=(19, 20, 21, 22, 23, 24),
                ),
            )
        )


def test_rejects_baseline_history_current_round() -> None:
    with pytest.raises(
        ContractError
    ):
        _evaluate(
            baseline_history_rows=(
                *_history(),
                HistoryRow(
                    round_no=1200,
                    numbers=(19, 20, 21, 22, 23, 24),
                ),
            )
        )


def test_rejects_candidate_history_future_round() -> None:
    with pytest.raises(
        ContractError
    ):
        _evaluate(
            candidate_history_rows=(
                *_history(),
                HistoryRow(
                    round_no=1201,
                    numbers=(19, 20, 21, 22, 23, 24),
                ),
            )
        )


def test_rejects_baseline_history_future_round() -> None:
    with pytest.raises(
        ContractError
    ):
        _evaluate(
            baseline_history_rows=(
                *_history(),
                HistoryRow(
                    round_no=1201,
                    numbers=(19, 20, 21, 22, 23, 24),
                ),
            )
        )


def test_rejects_candidate_baseline_prediction_round_mismatch() -> None:
    with pytest.raises(
        ContractError
    ):
        _evaluate(
            baseline_prediction_result=_prediction_result(
                1201
            )
        )


# ---------------------------------------------------------------------------
# 3 positive composition contracts
# ---------------------------------------------------------------------------


def test_orchestrator_composes_existing_live_evaluation_chain() -> None:
    result = _evaluate()

    assert isinstance(
        result,
        TopKLiveEvaluationResult,
    )

    assert (
        result.evaluation.candidate_model_name
        == "candidate"
    )

    assert (
        result.evaluation.baseline_model_name
        == "baseline"
    )

    assert result.evaluation.round_count == 1


def test_orchestrator_preserves_candidate_and_baseline_identity() -> None:
    result = _evaluate()

    assert (
        result.candidate_binding.model_name
        == "candidate"
    )

    assert (
        result.baseline_binding.model_name
        == "baseline"
    )

    assert (
        result.candidate_replay_prediction.model_name
        == "candidate"
    )

    assert (
        result.baseline_replay_prediction.model_name
        == "baseline"
    )


def test_orchestrator_preserves_explicit_provenance() -> None:
    result = _evaluate()

    assert (
        result.candidate_binding.source.regime_id
        == "gap-recovery"
    )

    assert (
        result.candidate_binding.source.strategy_name
        == "candidate-main"
    )

    assert (
        result.baseline_binding.source.regime_id
        == "stable"
    )

    assert (
        result.baseline_binding.source.strategy_name
        == "baseline-main"
    )