from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import pytest

from lrp.contracts import ContractError
from lrp.evaluation import EvaluationWindow
from lrp.evaluation.topk_walkforward import (
    HitDistribution,
    TopKWalkForwardEvaluator,
)


@dataclass(frozen=True)
class ReplayRow:
    round_no: int
    history_rounds: tuple[int, ...]
    actual_numbers: tuple[int, ...]
    predictions: tuple[tuple[int, ...], ...]
    model_name: str = "combined"
    regime_id: str | None = None
    strategy_name: str | None = None


class BaselineProvider:
    def __init__(
        self,
        rows: dict[int, ReplayRow],
    ) -> None:
        self.rows = dict(rows)

    def get(
        self,
        round_no: int,
    ) -> ReplayRow:
        try:
            return self.rows[round_no]
        except KeyError as exc:
            raise ContractError(
                "baseline round missing"
            ) from exc


def _predictions() -> tuple[tuple[int, ...], ...]:
    return (
        (1, 2, 3, 20, 21, 22),
        (1, 2, 7, 8, 9, 10),
        (1, 11, 12, 13, 14, 15),
        (1, 2, 3, 4, 16, 17),
        (1, 2, 3, 4, 5, 6),
        (7, 8, 9, 22, 23, 24),
        (10, 11, 12, 25, 26, 27),
        (13, 14, 15, 28, 29, 30),
        (16, 17, 18, 31, 32, 33),
        (19, 20, 21, 34, 35, 36),
    )


def _baseline_predictions() -> tuple[tuple[int, ...], ...]:
    return (
        (1, 7, 8, 9, 10, 11),
        (2, 12, 13, 14, 15, 16),
        (3, 17, 18, 19, 20, 21),
        (4, 22, 23, 24, 25, 26),
        (5, 27, 28, 29, 30, 31),
        (6, 32, 33, 34, 35, 36),
        (7, 8, 37, 38, 39, 40),
        (9, 10, 11, 41, 42, 43),
        (12, 13, 14, 15, 44, 45),
        (16, 17, 18, 19, 20, 21),
    )


def _row(
    round_no: int,
    *,
    history_rounds: tuple[int, ...] | None = None,
    actual_numbers: tuple[int, ...] = (
        1,
        2,
        3,
        4,
        5,
        6,
    ),
    predictions: tuple[tuple[int, ...], ...] | None = None,
    model_name: str = "combined",
    regime_id: str | None = None,
    strategy_name: str | None = None,
) -> ReplayRow:
    if history_rounds is None:
        history_rounds = (
            round_no - 4,
            round_no - 3,
            round_no - 2,
            round_no - 1,
        )

    if predictions is None:
        predictions = _predictions()

    return ReplayRow(
        round_no=round_no,
        history_rounds=history_rounds,
        actual_numbers=actual_numbers,
        predictions=predictions,
        model_name=model_name,
        regime_id=regime_id,
        strategy_name=strategy_name,
    )


def _baseline(
    round_no: int,
    *,
    history_rounds: tuple[int, ...] | None = None,
) -> ReplayRow:
    if history_rounds is None:
        history_rounds = (
            round_no - 4,
            round_no - 3,
            round_no - 2,
            round_no - 1,
        )

    return ReplayRow(
        round_no=round_no,
        history_rounds=history_rounds,
        actual_numbers=(
            1,
            2,
            3,
            4,
            5,
            6,
        ),
        predictions=_baseline_predictions(),
        model_name="baseline",
        regime_id=None,
        strategy_name=None,
    )


def _evaluate(
    rows: tuple[ReplayRow, ...],
    *,
    baseline_rows: dict[int, ReplayRow] | None = None,
    start_round: int | None = None,
    end_round: int | None = None,
):
    if baseline_rows is None:
        baseline_rows = {
            row.round_no: _baseline(
                row.round_no
            )
            for row in rows
        }

    if start_round is None:
        start_round = min(
            row.round_no
            for row in rows
        )

    if end_round is None:
        end_round = max(
            row.round_no
            for row in rows
        )

    evaluator = TopKWalkForwardEvaluator(
        baseline_provider=BaselineProvider(
            baseline_rows
        )
    )

    return evaluator.evaluate(
        window=EvaluationWindow(
            name="aa07",
            start_round=start_round,
            end_round=end_round,
        ),
        replay_rows=rows,
    )


def test_empty_hit_distribution_rates_are_zero() -> None:
    value = HitDistribution(
        hit_0=0,
        hit_1=0,
        hit_2=0,
        hit_3=0,
        hit_4=0,
        hit_5=0,
        hit_6=0,
    )

    assert value.total_count == 0
    assert value.at_least_3_rate == 0.0
    assert value.at_least_4_rate == 0.0
    assert value.at_least_5_rate == 0.0
    assert value.six_hit_rate == 0.0


def test_duplicate_actual_numbers_fail_closed() -> None:
    row = _row(
        1200,
        actual_numbers=(
            1,
            1,
            2,
            3,
            4,
            5,
        ),
    )

    with pytest.raises(ContractError):
        _evaluate(
            (row,)
        )


def test_out_of_range_prediction_number_fails_closed() -> None:
    predictions = list(
        _predictions()
    )

    predictions[0] = (
        1,
        2,
        3,
        20,
        21,
        46,
    )

    row = _row(
        1200,
        predictions=tuple(
            predictions
        ),
    )

    with pytest.raises(ContractError):
        _evaluate(
            (row,)
        )


def test_missing_baseline_round_fails_closed() -> None:
    row = _row(
        1200
    )

    with pytest.raises(ContractError):
        _evaluate(
            (row,),
            baseline_rows={},
        )


def test_mixed_model_names_fail_closed() -> None:
    rows = (
        _row(
            1200,
            model_name="combined",
        ),
        _row(
            1201,
            model_name="alternative",
        ),
    )

    with pytest.raises(ContractError):
        _evaluate(
            rows
        )


def test_round_outside_window_fails_closed() -> None:
    row = _row(
        1200
    )

    with pytest.raises(ContractError):
        _evaluate(
            (row,),
            start_round=1201,
            end_round=1201,
        )


def test_baseline_history_boundary_mismatch_fails_closed() -> None:
    row = _row(
        1200,
        history_rounds=(
            1196,
            1197,
            1198,
            1199,
        ),
    )

    bad_baseline = _baseline(
        1200,
        history_rounds=(
            1195,
            1196,
            1197,
            1198,
        ),
    )

    with pytest.raises(ContractError):
        _evaluate(
            (row,),
            baseline_rows={
                1200: bad_baseline,
            },
        )


def test_slice_order_is_deterministic() -> None:
    rows = (
        _row(
            1200,
            regime_id="R2",
            strategy_name="B",
        ),
        _row(
            1201,
            regime_id="R1",
            strategy_name="A",
        ),
        _row(
            1202,
            regime_id="R3",
            strategy_name="C",
        ),
    )

    result = _evaluate(
        rows
    )

    assert tuple(
        item.name
        for item in result.regime_slices
    ) == (
        "R1",
        "R2",
        "R3",
    )

    assert tuple(
        item.name
        for item in result.strategy_slices
    ) == (
        "A",
        "B",
        "C",
    )


def test_evaluation_does_not_mutate_replay_rows() -> None:
    rows = (
        _row(
            1200,
            regime_id="R1",
            strategy_name="A",
        ),
        _row(
            1201,
            regime_id="R2",
            strategy_name="B",
        ),
    )

    before = copy.deepcopy(
        rows
    )

    _evaluate(
        rows
    )

    assert rows == before


def test_repeated_evaluation_payload_is_byte_semantically_stable() -> None:
    rows = (
        _row(
            1200,
            regime_id="R2",
            strategy_name="B",
        ),
        _row(
            1201,
            regime_id="R1",
            strategy_name="A",
        ),
    )

    first = _evaluate(
        rows
    )

    second = _evaluate(
        rows
    )

    first_payload = json.dumps(
        first.as_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )

    second_payload = json.dumps(
        second.as_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )

    assert first_payload == second_payload
