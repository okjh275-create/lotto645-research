from __future__ import annotations

from types import SimpleNamespace

from dataclasses import FrozenInstanceError

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.io.draws import HistoryRow
from lrp.pipelines.models import (
    PredictionGenerationResult,
    PredictionRequest,
    PredictionResult,
)
from lrp.evaluation.topk_prediction_source_adapter import (
    TopKPredictionSourceAdapter,
    TopKPredictionSourceRecord,
)
from lrp.evaluation.topk_live_prediction_binding import (
    TopKLivePredictionBinder,
    TopKLivePredictionBindingRequest,
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

    diversity = SimpleNamespace(
        selected=tuple(
            SimpleNamespace(
                numbers=numbers,
            )
            for numbers in selected_numbers
        )
    )

    object.__setattr__(
        result,
        "diversity",
        diversity,
    )

    return result


def _row(
    round_no: int,
    *,
    offset: int = 0,
) -> HistoryRow:
    start = 1 + offset

    return HistoryRow(
        round_no=round_no,
        numbers=(
            start,
            start + 1,
            start + 2,
            start + 3,
            start + 4,
            start + 5,
        ),
    )


def _history() -> tuple[HistoryRow, ...]:
    return (
        _row(
            1197,
            offset=0,
        ),
        _row(
            1198,
            offset=6,
        ),
        _row(
            1199,
            offset=12,
        ),
    )


def _request(
    *,
    prediction_result: PredictionResult | None = None,
    history_rows: tuple[HistoryRow, ...] | None = None,
    model_name: str = "champion-v1",
    regime_id: str | None = None,
    strategy_name: str | None = None,
) -> TopKLivePredictionBindingRequest:
    return TopKLivePredictionBindingRequest(
        prediction_result=(
            _prediction_result()
            if prediction_result is None
            else prediction_result
        ),
        history_rows=(
            _history()
            if history_rows is None
            else history_rows
        ),
        model_name=model_name,
        regime_id=regime_id,
        strategy_name=strategy_name,
    )


def _malformed_request(
    **values: object,
) -> TopKLivePredictionBindingRequest:
    request = object.__new__(
        TopKLivePredictionBindingRequest
    )

    defaults: dict[str, object] = {
        "prediction_result":
            _prediction_result(),

        "history_rows":
            _history(),

        "model_name":
            "champion-v1",

        "regime_id":
            None,

        "strategy_name":
            None,
    }

    defaults.update(
        values
    )

    for name, value in defaults.items():
        object.__setattr__(
            request,
            name,
            value,
        )

    return request


def test_binder_produces_ad_compatible_source_record() -> None:
    binding = TopKLivePredictionBinder().bind(
        request=_request(
            regime_id="gap_recovery",
            strategy_name="ensemble-main",
        )
    )

    assert isinstance(
        binding.source,
        TopKPredictionSourceRecord,
    )

    replay_prediction = (
        TopKPredictionSourceAdapter()
        .adapt(
            source=binding.source
        )
    )

    assert replay_prediction.round_no == 1200
    assert replay_prediction.model_name == "champion-v1"
    assert replay_prediction.history_rounds == (
        1197,
        1198,
        1199,
    )


def test_binder_derives_prediction_round_from_result() -> None:
    binding = TopKLivePredictionBinder().bind(
        request=_request(
            prediction_result=_prediction_result(
                1225
            ),
            history_rows=(
                _row(
                    1223,
                    offset=0,
                ),
                _row(
                    1224,
                    offset=6,
                ),
            ),
        )
    )

    assert binding.prediction_round == 1225
    assert binding.source.prediction_result.request.round_no == 1225


def test_binder_derives_prior_history_rounds() -> None:
    binding = TopKLivePredictionBinder().bind(
        request=_request()
    )

    assert binding.history_rounds == (
        1197,
        1198,
        1199,
    )

    assert binding.source.history_rounds == (
        1197,
        1198,
        1199,
    )

    assert all(
        round_no < binding.prediction_round
        for round_no in binding.history_rounds
    )


def test_binder_normalizes_history_order() -> None:
    history_rows = (
        _row(
            1199,
            offset=12,
        ),
        _row(
            1197,
            offset=0,
        ),
        _row(
            1198,
            offset=6,
        ),
    )

    binding = TopKLivePredictionBinder().bind(
        request=_request(
            history_rows=history_rows
        )
    )

    assert binding.history_rounds == (
        1197,
        1198,
        1199,
    )


def test_binder_preserves_model_name() -> None:
    binding = TopKLivePredictionBinder().bind(
        request=_request(
            model_name="resolved-champion-v7"
        )
    )

    assert binding.model_name == "resolved-champion-v7"
    assert binding.source.model_name == "resolved-champion-v7"


def test_binder_preserves_optional_regime_id() -> None:
    binding = TopKLivePredictionBinder().bind(
        request=_request(
            regime_id="cluster_rotation"
        )
    )

    assert binding.source.regime_id == "cluster_rotation"


def test_binder_preserves_optional_strategy_name() -> None:
    binding = TopKLivePredictionBinder().bind(
        request=_request(
            strategy_name="ensemble-main"
        )
    )

    assert binding.source.strategy_name == "ensemble-main"


def test_binder_rejects_wrong_request_type() -> None:
    with pytest.raises(
        ContractError
    ):
        TopKLivePredictionBinder().bind(
            request=object()  # type: ignore[arg-type]
        )


def test_binder_rejects_wrong_prediction_result_type() -> None:
    request = _malformed_request(
        prediction_result=object()
    )

    with pytest.raises(
        ContractError
    ):
        TopKLivePredictionBinder().bind(
            request=request
        )


def test_binder_rejects_empty_history() -> None:
    request = _malformed_request(
        history_rows=()
    )

    with pytest.raises(
        ContractError
    ):
        TopKLivePredictionBinder().bind(
            request=request
        )


def test_binder_rejects_wrong_history_item_type() -> None:
    request = _malformed_request(
        history_rows=(
            _row(
                1199
            ),
            object(),
        )
    )

    with pytest.raises(
        ContractError
    ):
        TopKLivePredictionBinder().bind(
            request=request
        )


def test_binder_rejects_blank_model_name() -> None:
    request = _malformed_request(
        model_name="   "
    )

    with pytest.raises(
        ContractError
    ):
        TopKLivePredictionBinder().bind(
            request=request
        )


def test_binder_rejects_blank_optional_regime_id() -> None:
    request = _malformed_request(
        regime_id="   "
    )

    with pytest.raises(
        ContractError
    ):
        TopKLivePredictionBinder().bind(
            request=request
        )


def test_binder_rejects_blank_optional_strategy_name() -> None:
    request = _malformed_request(
        strategy_name="   "
    )

    with pytest.raises(
        ContractError
    ):
        TopKLivePredictionBinder().bind(
            request=request
        )


def test_binder_rejects_duplicate_history_round() -> None:
    request = _malformed_request(
        history_rows=(
            _row(
                1198,
                offset=0,
            ),
            _row(
                1198,
                offset=6,
            ),
            _row(
                1199,
                offset=12,
            ),
        )
    )

    with pytest.raises(
        ContractError
    ):
        TopKLivePredictionBinder().bind(
            request=request
        )


def test_binder_rejects_current_round_in_history() -> None:
    request = _malformed_request(
        history_rows=(
            _row(
                1199,
                offset=0,
            ),
            _row(
                1200,
                offset=6,
            ),
        )
    )

    with pytest.raises(
        ContractError
    ):
        TopKLivePredictionBinder().bind(
            request=request
        )


def test_binder_rejects_future_round_in_history() -> None:
    request = _malformed_request(
        history_rows=(
            _row(
                1199,
                offset=0,
            ),
            _row(
                1201,
                offset=6,
            ),
        )
    )

    with pytest.raises(
        ContractError
    ):
        TopKLivePredictionBinder().bind(
            request=request
        )


def test_binder_rejects_no_prior_history() -> None:
    request = _malformed_request(
        history_rows=(
            _row(
                1200,
                offset=0,
            ),
            _row(
                1201,
                offset=6,
            ),
        )
    )

    with pytest.raises(
        ContractError
    ):
        TopKLivePredictionBinder().bind(
            request=request
        )
