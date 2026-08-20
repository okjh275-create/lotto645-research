from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from lrp.contracts import ContractError
from lrp.evaluation.topk_replay_adapter import (
    TopKReplayBaselineProvider,
    TopKReplayPrediction,
    TopKReplayRow,
)


def _predictions() -> tuple[tuple[int, ...], ...]:
    return (
        (6, 5, 4, 3, 2, 1),
        (12, 11, 10, 9, 8, 7),
        (18, 17, 16, 15, 14, 13),
        (24, 23, 22, 21, 20, 19),
        (30, 29, 28, 27, 26, 25),
        (36, 35, 34, 33, 32, 31),
        (42, 41, 40, 39, 38, 37),
        (45, 44, 43, 3, 2, 1),
        (15, 14, 13, 12, 11, 10),
        (25, 24, 23, 22, 21, 20),
    )


def _row(
    round_no: int = 1200,
    model_name: str = "combined",
) -> TopKReplayRow:
    return TopKReplayRow(
        round_no=round_no,
        history_rounds=(
            round_no - 3,
            round_no - 2,
            round_no - 1,
        ),
        actual_numbers=(6, 5, 4, 3, 2, 1),
        predictions=_predictions(),
        model_name=model_name,
        regime_id="R1",
        strategy_name="S1",
    )


def test_topk_replay_prediction_contract() -> None:
    value = TopKReplayPrediction(
        round_no=1200,
        history_rounds=(1197, 1198, 1199),
        predictions=_predictions(),
        model_name="combined",
        regime_id="R1",
        strategy_name="S1",
    )

    assert value.round_no == 1200
    assert value.history_rounds == (1197, 1198, 1199)
    assert value.model_name == "combined"
    assert value.regime_id == "R1"
    assert value.strategy_name == "S1"

    with pytest.raises(FrozenInstanceError):
        value.model_name = "other"  # type: ignore[misc]


def test_topk_replay_row_contract() -> None:
    value = _row()

    assert value.round_no == 1200
    assert value.history_rounds == (1197, 1198, 1199)
    assert value.actual_numbers == (1, 2, 3, 4, 5, 6)

    assert value.predictions[0] == (
        1,
        2,
        3,
        4,
        5,
        6,
    )

    assert len(value.predictions) == 10
    assert value.model_name == "combined"
    assert value.regime_id == "R1"
    assert value.strategy_name == "S1"

    with pytest.raises(FrozenInstanceError):
        value.round_no = 1201  # type: ignore[misc]


def test_topk_replay_baseline_provider_contract() -> None:
    first = _row(
        1200,
        model_name="baseline",
    )

    second = _row(
        1201,
        model_name="baseline",
    )

    provider = TopKReplayBaselineProvider(
        (
            first,
            second,
        )
    )

    assert provider.get(1200) == first
    assert provider.get(1201) == second


def test_prediction_model_rejects_invalid_round() -> None:
    with pytest.raises(ContractError):
        TopKReplayPrediction(
            round_no=0,
            history_rounds=(1,),
            predictions=_predictions(),
            model_name="combined",
            regime_id=None,
            strategy_name=None,
        )


def test_replay_row_rejects_invalid_actual_numbers() -> None:
    with pytest.raises(ContractError):
        TopKReplayRow(
            round_no=1200,
            history_rounds=(1197, 1198, 1199),
            actual_numbers=(1, 1, 2, 3, 4, 5),
            predictions=_predictions(),
            model_name="combined",
            regime_id=None,
            strategy_name=None,
        )


def test_replay_models_reject_non_string_optional_provenance() -> None:
    with pytest.raises(ContractError):
        TopKReplayPrediction(
            round_no=1200,
            history_rounds=(1197, 1198, 1199),
            predictions=_predictions(),
            model_name="combined",
            regime_id=123,  # type: ignore[arg-type]
            strategy_name=None,
        )
