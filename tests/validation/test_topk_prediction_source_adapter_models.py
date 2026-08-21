from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from lrp.contracts.exceptions import ContractError
from lrp.pipelines.models import (
    PredictionGenerationResult,
    PredictionRequest,
    PredictionResult,
)
from lrp.evaluation.topk_prediction_source_adapter import (
    TopKPredictionSourceRecord,
)


def _prediction_result(
    *,
    round_no: int = 1200,
    top_k: int = 2,
) -> PredictionResult:
    request = PredictionRequest(
        round_no=round_no,
        seed=20260821,
        long_gap_numbers=frozenset({45}),
        top_k=top_k,
        practical_k=min(2, top_k),
    )

    generation = PredictionGenerationResult(
        request=request,
        windows=(10, 20, 50),
        probabilities={1: 1.0},
        statistics_contract=object(),
        number_signals={},
        candidates=(),
        statistics_version="test-statistics",
        candidate_version="test-candidate",
    )

    selected = (
        SimpleNamespace(
            numbers=(1, 2, 3, 4, 5, 6),
        ),
        SimpleNamespace(
            numbers=(7, 8, 9, 10, 11, 12),
        ),
    )[:top_k]

    return PredictionResult(
        generation=generation,
        scored_candidates=(),
        ranking=object(),
        diversity=SimpleNamespace(
            selected=selected,
        ),
        practical=object(),
        generated_at_kst=__import__(
            "datetime"
        ).datetime.now(
            __import__(
                "datetime"
            ).timezone.utc
        ),
    )


def _record(
    **overrides: object,
) -> TopKPredictionSourceRecord:
    values = {
        "prediction_result":
            _prediction_result(),

        "model_name":
            "candidate",

        "history_rounds":
            (1197, 1198, 1199),

        "regime_id":
            "regime-a",

        "strategy_name":
            "strategy-a",
    }

    values.update(
        overrides
    )

    return TopKPredictionSourceRecord(
        **values
    )


def test_source_record_contract() -> None:
    result = _prediction_result()

    source = TopKPredictionSourceRecord(
        prediction_result=result,
        model_name="candidate",
        history_rounds=(
            1197,
            1198,
            1199,
        ),
        regime_id="regime-a",
        strategy_name="strategy-a",
    )

    assert source.prediction_result is result
    assert source.model_name == "candidate"
    assert source.history_rounds == (
        1197,
        1198,
        1199,
    )
    assert source.regime_id == "regime-a"
    assert source.strategy_name == "strategy-a"


def test_source_record_is_immutable() -> None:
    source = _record()

    with pytest.raises(
        FrozenInstanceError
    ):
        source.model_name = "changed"  # type: ignore[misc]


def test_source_record_rejects_invalid_prediction_result() -> None:
    with pytest.raises(
        ContractError
    ):
        _record(
            prediction_result=object(),
        )


def test_source_record_rejects_blank_model_name() -> None:
    for value in (
        "",
        "   ",
    ):
        with pytest.raises(
            ContractError
        ):
            _record(
                model_name=value,
            )


def test_source_record_rejects_invalid_optional_provenance() -> None:
    invalid_values = (
        "",
        "   ",
        1,
        False,
        object(),
    )

    for value in invalid_values:
        with pytest.raises(
            ContractError
        ):
            _record(
                regime_id=value,
            )

        with pytest.raises(
            ContractError
        ):
            _record(
                strategy_name=value,
            )


def test_source_record_rejects_invalid_history_rounds() -> None:
    invalid_values = (
        (),
        (False,),
        ("1199",),
        (0,),
        (-1,),
        (1198, 1198),
        (1199, 1198),
        (1197, 1200),
        (1197, 1201),
    )

    for value in invalid_values:
        with pytest.raises(
            ContractError
        ):
            _record(
                history_rounds=value,
            )
