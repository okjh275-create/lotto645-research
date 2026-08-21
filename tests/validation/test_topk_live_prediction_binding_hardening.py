from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.io.draws import HistoryRow
from lrp.pipelines.models import (
    PredictionGenerationResult,
    PredictionRequest,
    PredictionResult,
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


def _request(
    *,
    prediction_result: PredictionResult | None = None,
    history_rows: object | None = None,
    model_name: str = "champion-v1",
    regime_id: str | None = None,
    strategy_name: str | None = None,
) -> TopKLivePredictionBindingRequest:
    if history_rows is None:
        history_rows = _history()

    return TopKLivePredictionBindingRequest(
        prediction_result=(
            _prediction_result()
            if prediction_result is None
            else prediction_result
        ),
        history_rows=history_rows,  # type: ignore[arg-type]
        model_name=model_name,
        regime_id=regime_id,
        strategy_name=strategy_name,
    )


def _malformed_prediction_round(
    value: object,
) -> PredictionResult:
    result = _prediction_result()

    request = object.__new__(
        PredictionRequest
    )

    object.__setattr__(
        request,
        "round_no",
        value,
    )

    generation = object.__new__(
        PredictionGenerationResult
    )

    object.__setattr__(
        generation,
        "request",
        request,
    )

    object.__setattr__(
        result,
        "generation",
        generation,
    )

    return result


def _malformed_history_row(
    value: object,
) -> HistoryRow:
    row = object.__new__(
        HistoryRow
    )

    object.__setattr__(
        row,
        "round_no",
        value,
    )

    object.__setattr__(
        row,
        "numbers",
        (1, 2, 3, 4, 5, 6),
    )

    return row


def _semantic_payload(
    result: object,
) -> tuple[object, ...]:
    return (
        result.prediction_round,  # type: ignore[attr-defined]
        result.history_rounds,  # type: ignore[attr-defined]
        result.model_name,  # type: ignore[attr-defined]
        result.source.model_name,  # type: ignore[attr-defined]
        result.source.history_rounds,  # type: ignore[attr-defined]
        result.source.regime_id,  # type: ignore[attr-defined]
        result.source.strategy_name,  # type: ignore[attr-defined]
    )


def test_request_rejects_non_tuple_history_rows() -> None:
    with pytest.raises(
        ContractError
    ):
        _request(
            history_rows=list(
                _history()
            )
        )


def test_request_rejects_blank_model_name() -> None:
    with pytest.raises(
        ContractError
    ):
        _request(
            model_name="   "
        )


def test_request_rejects_blank_regime_id() -> None:
    with pytest.raises(
        ContractError
    ):
        _request(
            regime_id="   "
        )


def test_request_rejects_blank_strategy_name() -> None:
    with pytest.raises(
        ContractError
    ):
        _request(
            strategy_name="   "
        )


def test_binder_rejects_boolean_prediction_round() -> None:
    request = _request(
        prediction_result=_malformed_prediction_round(
            True
        )
    )

    with pytest.raises(
        ContractError
    ):
        TopKLivePredictionBinder().bind(
            request=request
        )


def test_binder_rejects_non_integer_prediction_round() -> None:
    request = _request(
        prediction_result=_malformed_prediction_round(
            "1200"
        )
    )

    with pytest.raises(
        ContractError
    ):
        TopKLivePredictionBinder().bind(
            request=request
        )


def test_binder_rejects_boolean_history_round() -> None:
    history = (
        _malformed_history_row(
            True
        ),
        *_history(),
    )

    request = _request(
        history_rows=history
    )

    with pytest.raises(
        ContractError
    ):
        TopKLivePredictionBinder().bind(
            request=request
        )


def test_binder_rejects_non_integer_history_round() -> None:
    history = (
        _malformed_history_row(
            "1196"
        ),
        *_history(),
    )

    request = _request(
        history_rows=history
    )

    with pytest.raises(
        ContractError
    ):
        TopKLivePredictionBinder().bind(
            request=request
        )


def test_history_input_order_is_semantically_normalized() -> None:
    binder = TopKLivePredictionBinder()

    forward = binder.bind(
        request=_request(
            history_rows=_history()
        )
    )

    reverse = binder.bind(
        request=_request(
            history_rows=tuple(
                reversed(
                    _history()
                )
            )
        )
    )

    assert _semantic_payload(
        forward
    ) == _semantic_payload(
        reverse
    )


def test_repeated_binding_is_semantically_stable() -> None:
    binder = TopKLivePredictionBinder()

    request = _request(
        regime_id="gap-recovery",
        strategy_name="ensemble-main",
    )

    first = binder.bind(
        request=request
    )

    second = binder.bind(
        request=request
    )

    assert _semantic_payload(
        first
    ) == _semantic_payload(
        second
    )


def test_binding_does_not_mutate_history_rows() -> None:
    history = _history()

    before = tuple(
        (
            row.round_no,
            row.numbers,
        )
        for row in history
    )

    TopKLivePredictionBinder().bind(
        request=_request(
            history_rows=history
        )
    )

    after = tuple(
        (
            row.round_no,
            row.numbers,
        )
        for row in history
    )

    assert after == before


def test_binding_does_not_mutate_prediction_result() -> None:
    prediction_result = _prediction_result()

    generation_before = (
        prediction_result.generation
    )

    diversity_before = (
        prediction_result.diversity
    )

    TopKLivePredictionBinder().bind(
        request=_request(
            prediction_result=prediction_result
        )
    )

    assert (
        prediction_result.generation
        is generation_before
    )

    assert (
        prediction_result.diversity
        is diversity_before
    )


def test_result_source_identity_is_stable() -> None:
    prediction_result = _prediction_result()

    result = TopKLivePredictionBinder().bind(
        request=_request(
            prediction_result=prediction_result,
            model_name="champion-v9",
        )
    )

    assert (
        result.source.prediction_result
        is prediction_result
    )

    assert (
        result.source.model_name
        == result.model_name
        == "champion-v9"
    )

    assert (
        result.source.history_rounds
        == result.history_rounds
    )


def test_optional_none_provenance_is_preserved() -> None:
    result = TopKLivePredictionBinder().bind(
        request=_request(
            regime_id=None,
            strategy_name=None,
        )
    )

    assert result.source.regime_id is None
    assert result.source.strategy_name is None


def test_extra_prior_history_round_is_preserved() -> None:
    history = (
        HistoryRow(
            round_no=1196,
            numbers=(19, 20, 21, 22, 23, 24),
        ),
        *_history(),
    )

    result = TopKLivePredictionBinder().bind(
        request=_request(
            history_rows=history
        )
    )

    assert result.history_rounds == (
        1196,
        1197,
        1198,
        1199,
    )


def test_product_has_no_runtime_nondeterminism_dependencies() -> None:
    product = Path(
        "lrp/evaluation/topk_live_prediction_binding.py"
    )

    tree = ast.parse(
        product.read_text(
            encoding="utf-8-sig"
        )
    )

    imports: list[str] = []

    for node in ast.walk(
        tree
    ):
        if isinstance(
            node,
            ast.Import,
        ):
            imports.extend(
                alias.name
                for alias in node.names
            )

        elif isinstance(
            node,
            ast.ImportFrom,
        ):
            imports.append(
                node.module or ""
            )

    forbidden = (
        "random",
        "secrets",
        "uuid",
        "time",
        "datetime",
    )

    violations = [
        name
        for name in imports
        if any(
            name == token
            or name.startswith(
                token + "."
            )
            for token in forbidden
        )
    ]

    assert violations == []