from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from lrp.io.draws import HistoryRow
from lrp.pipelines.models import (
    PredictionGenerationResult,
    PredictionRequest,
    PredictionResult,
)
from lrp.evaluation.topk_prediction_source_adapter import (
    TopKPredictionSourceRecord,
)
from lrp.evaluation.topk_live_prediction_binding import (
    TopKLivePredictionBinder,
    TopKLivePredictionBindingRequest,
    TopKLivePredictionBindingResult,
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

    return result


def _history() -> tuple[HistoryRow, ...]:
    return (
        HistoryRow(
            round_no=1198,
            numbers=(1, 2, 3, 4, 5, 6),
        ),
        HistoryRow(
            round_no=1199,
            numbers=(7, 8, 9, 10, 11, 12),
        ),
    )


def _source() -> TopKPredictionSourceRecord:
    return TopKPredictionSourceRecord(
        prediction_result=_prediction_result(),
        model_name="champion-v1",
        history_rounds=(1198, 1199),
        regime_id="gap_recovery",
        strategy_name="ensemble-main",
    )


def test_binding_request_contract() -> None:
    request = TopKLivePredictionBindingRequest(
        prediction_result=_prediction_result(),
        history_rows=_history(),
        model_name="champion-v1",
        regime_id="gap_recovery",
        strategy_name="ensemble-main",
    )

    assert isinstance(
        request.prediction_result,
        PredictionResult,
    )
    assert request.history_rows == _history()
    assert request.model_name == "champion-v1"
    assert request.regime_id == "gap_recovery"
    assert request.strategy_name == "ensemble-main"


def test_binding_request_is_immutable() -> None:
    request = TopKLivePredictionBindingRequest(
        prediction_result=_prediction_result(),
        history_rows=_history(),
        model_name="champion-v1",
    )

    with pytest.raises(
        (FrozenInstanceError, AttributeError),
    ):
        request.model_name = "changed"  # type: ignore[misc]


def test_binding_result_contract() -> None:
    source = _source()

    result = TopKLivePredictionBindingResult(
        source=source,
        prediction_round=1200,
        history_rounds=(1198, 1199),
        model_name="champion-v1",
    )

    assert result.source is source
    assert result.prediction_round == 1200
    assert result.history_rounds == (1198, 1199)
    assert result.model_name == "champion-v1"


def test_binding_result_is_immutable() -> None:
    result = TopKLivePredictionBindingResult(
        source=_source(),
        prediction_round=1200,
        history_rounds=(1198, 1199),
        model_name="champion-v1",
    )

    with pytest.raises(
        (FrozenInstanceError, AttributeError),
    ):
        result.model_name = "changed"  # type: ignore[misc]


def test_binding_request_preserves_optional_provenance() -> None:
    request = TopKLivePredictionBindingRequest(
        prediction_result=_prediction_result(),
        history_rows=_history(),
        model_name="champion-v1",
        regime_id="cluster_rotation",
        strategy_name="strategy-a",
    )

    assert request.regime_id == "cluster_rotation"
    assert request.strategy_name == "strategy-a"

    none_request = TopKLivePredictionBindingRequest(
        prediction_result=_prediction_result(),
        history_rows=_history(),
        model_name="champion-v1",
    )

    assert none_request.regime_id is None
    assert none_request.strategy_name is None


def test_binding_result_exposes_canonical_identity() -> None:
    result = TopKLivePredictionBinder().bind(
        request=TopKLivePredictionBindingRequest(
            prediction_result=_prediction_result(),
            history_rows=_history(),
            model_name="champion-v1",
        )
    )

    assert result.prediction_round == 1200
    assert result.model_name == "champion-v1"
    assert result.history_rounds == (1198, 1199)
