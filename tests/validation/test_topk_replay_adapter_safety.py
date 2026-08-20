from __future__ import annotations

import ast
import copy
from pathlib import Path

from lrp.evaluation.topk_replay_adapter import (
    TopKReplayAdapter,
    TopKReplayPrediction,
)


class Draw:
    def __init__(
        self,
        round_no: int,
        numbers: tuple[int, ...],
    ) -> None:
        self.round_no = round_no
        self.numbers = numbers
        self.bonus = 45


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


def _prediction() -> TopKReplayPrediction:
    return TopKReplayPrediction(
        round_no=1200,
        history_rounds=(1197, 1198, 1199),
        predictions=_predictions(),
        model_name="combined",
        regime_id="R1",
        strategy_name="S1",
    )


def test_adapter_is_deterministic() -> None:
    prediction = _prediction()

    draw = Draw(
        round_no=1200,
        numbers=(6, 5, 4, 3, 2, 1),
    )

    adapter = TopKReplayAdapter()

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


def test_adapter_does_not_mutate_inputs() -> None:
    prediction = _prediction()

    draw = Draw(
        round_no=1200,
        numbers=(6, 5, 4, 3, 2, 1),
    )

    prediction_before = copy.deepcopy(
        prediction
    )

    draw_before = copy.deepcopy(
        draw.__dict__
    )

    TopKReplayAdapter().adapt(
        prediction_rows=(
            prediction,
        ),
        actual_draws=(
            draw,
        ),
    )

    assert prediction == prediction_before
    assert draw.__dict__ == draw_before


def test_adapter_has_no_mutating_or_production_dependencies() -> None:
    path = Path(
        "lrp/evaluation/topk_replay_adapter.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    imports = []
    calls = set()

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

        elif isinstance(
            node,
            ast.Call,
        ):
            func = node.func

            if isinstance(
                func,
                ast.Name,
            ):
                calls.add(
                    func.id
                )

            elif isinstance(
                func,
                ast.Attribute,
            ):
                calls.add(
                    func.attr
                )

    forbidden_imports = [
        name
        for name in imports
        if (
            name.startswith(
                "lrp.production"
            )
            or name.startswith(
                "lrp.cli"
            )
            or name == "tools"
            or name.startswith(
                "tools."
            )
        )
    ]

    forbidden_calls = {
        "write_text",
        "write_bytes",
        "unlink",
        "mkdir",
        "rmdir",
        "rename",
        "remove",
    }

    assert forbidden_imports == []
    assert calls.isdisjoint(
        forbidden_calls
    )
