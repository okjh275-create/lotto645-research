from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from lrp.contracts import ContractError
from lrp.evaluation.topk_replay_adapter import (
    TopKReplayAdapter,
    TopKReplayBaselineProvider,
    TopKReplayPrediction,
    TopKReplayRow,
)


class Draw:
    def __init__(
        self,
        *,
        round_no=1200,
        numbers=(1, 2, 3, 4, 5, 6),
    ) -> None:
        self.round_no = round_no
        self.numbers = numbers


class DrawWithoutRound:
    def __init__(self) -> None:
        self.numbers = (1, 2, 3, 4, 5, 6)


class DrawWithoutNumbers:
    def __init__(self) -> None:
        self.round_no = 1200


def _prediction_sets():
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
    round_no: int = 1200,
    *,
    history_rounds=None,
    predictions=None,
    model_name: str = "candidate",
    regime_id: str | None = "R1",
    strategy_name: str | None = "S1",
):
    if history_rounds is None:
        history_rounds = (
            round_no - 3,
            round_no - 2,
            round_no - 1,
        )

    if predictions is None:
        predictions = _prediction_sets()

    return TopKReplayPrediction(
        round_no=round_no,
        history_rounds=history_rounds,
        predictions=predictions,
        model_name=model_name,
        regime_id=regime_id,
        strategy_name=strategy_name,
    )


def _row(
    round_no: int = 1200,
    *,
    model_name: str = "baseline",
):
    return TopKReplayRow(
        round_no=round_no,
        history_rounds=(
            round_no - 3,
            round_no - 2,
            round_no - 1,
        ),
        actual_numbers=(
            1,
            2,
            3,
            4,
            5,
            6,
        ),
        predictions=_prediction_sets(),
        model_name=model_name,
        regime_id="R1",
        strategy_name="S1",
    )


def test_adapter_empty_prediction_source_returns_empty_rows() -> None:
    result = TopKReplayAdapter().adapt(
        prediction_rows=(),
        actual_draws=(),
    )

    assert result == ()


def test_adapter_ignores_extra_actual_draws() -> None:
    result = TopKReplayAdapter().adapt(
        prediction_rows=(
            _prediction(1200),
        ),
        actual_draws=(
            Draw(
                round_no=1200,
            ),
            Draw(
                round_no=1201,
                numbers=(
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                ),
            ),
        ),
    )

    assert len(result) == 1
    assert result[0].round_no == 1200


def test_adapter_rejects_draw_missing_round_no() -> None:
    with pytest.raises(ContractError):
        TopKReplayAdapter().adapt(
            prediction_rows=(
                _prediction(),
            ),
            actual_draws=(
                DrawWithoutRound(),
            ),
        )


def test_adapter_rejects_draw_missing_numbers() -> None:
    with pytest.raises(ContractError):
        TopKReplayAdapter().adapt(
            prediction_rows=(
                _prediction(),
            ),
            actual_draws=(
                DrawWithoutNumbers(),
            ),
        )


def test_adapter_rejects_non_integer_draw_round() -> None:
    with pytest.raises(ContractError):
        TopKReplayAdapter().adapt(
            prediction_rows=(
                _prediction(),
            ),
            actual_draws=(
                Draw(
                    round_no="1200",
                ),
            ),
        )


def test_adapter_rejects_boolean_draw_round() -> None:
    with pytest.raises(ContractError):
        TopKReplayAdapter().adapt(
            prediction_rows=(
                _prediction(),
            ),
            actual_draws=(
                Draw(
                    round_no=True,
                ),
            ),
        )


def test_adapter_rejects_non_iterable_draw_numbers() -> None:
    with pytest.raises(ContractError):
        TopKReplayAdapter().adapt(
            prediction_rows=(
                _prediction(),
            ),
            actual_draws=(
                Draw(
                    numbers=123,
                ),
            ),
        )


def test_adapter_rejects_boolean_prediction_number() -> None:
    bad = list(
        _prediction_sets()
    )

    bad[0] = (
        True,
        2,
        3,
        4,
        5,
        6,
    )

    with pytest.raises(ContractError):
        _prediction(
            predictions=tuple(bad)
        )


def test_adapter_rejects_non_integer_history_round() -> None:
    with pytest.raises(ContractError):
        _prediction(
            history_rounds=(
                1197,
                1198,
                "1199",
            )
        )


def test_baseline_provider_rejects_non_replay_row() -> None:
    with pytest.raises(ContractError):
        TopKReplayBaselineProvider(
            (
                object(),
            )
        )


def test_baseline_provider_lookup_is_stable() -> None:
    row = _row()

    provider = TopKReplayBaselineProvider(
        (
            row,
        )
    )

    first = provider.get(
        1200
    )

    second = provider.get(
        1200
    )

    assert first is row
    assert second is row
    assert first == second


def test_repeated_adaptation_semantics_are_stable() -> None:
    adapter = TopKReplayAdapter()

    prediction = _prediction()

    draw = Draw()

    first = adapter.adapt(
        prediction_rows=(
            prediction,
        ),
        actual_draws=(
            draw,
        ),
    )

    second = adapter.adapt(
        prediction_rows=(
            prediction,
        ),
        actual_draws=(
            draw,
        ),
    )

    assert first == second
    assert repr(first) == repr(second)


def test_output_rows_are_immutable() -> None:
    row = TopKReplayAdapter().adapt(
        prediction_rows=(
            _prediction(),
        ),
        actual_draws=(
            Draw(),
        ),
    )[0]

    with pytest.raises(
        FrozenInstanceError
    ):
        row.model_name = "mutated"  # type: ignore[misc]


def test_extra_draw_does_not_change_output_semantics() -> None:
    adapter = TopKReplayAdapter()

    prediction_rows = (
        _prediction(),
    )

    primary = (
        Draw(
            round_no=1200,
        ),
    )

    extended = (
        Draw(
            round_no=1200,
        ),
        Draw(
            round_no=1300,
            numbers=(
                40,
                41,
                42,
                43,
                44,
                45,
            ),
        ),
    )

    first = adapter.adapt(
        prediction_rows=prediction_rows,
        actual_draws=primary,
    )

    second = adapter.adapt(
        prediction_rows=prediction_rows,
        actual_draws=extended,
    )

    assert first == second


def test_optional_none_provenance_is_preserved() -> None:
    result = TopKReplayAdapter().adapt(
        prediction_rows=(
            _prediction(
                regime_id=None,
                strategy_name=None,
            ),
        ),
        actual_draws=(
            Draw(),
        ),
    )

    assert result[0].regime_id is None
    assert result[0].strategy_name is None


def test_adapter_does_not_depend_on_topk_evaluator() -> None:
    path = Path(
        "lrp/evaluation/topk_replay_adapter.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    imports = []

    for node in ast.walk(tree):
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
                node.module
                or ""
            )

    forbidden = [
        name
        for name in imports
        if (
            name
            == "lrp.evaluation.topk_walkforward"
            or name.startswith(
                "lrp.evaluation.topk_walkforward."
            )
        )
    ]

    assert forbidden == []
