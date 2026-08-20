from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

from lrp.contracts import ContractError
from lrp.evaluation import EvaluationWindow


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


def _api():
    module = importlib.import_module(
        "lrp.evaluation.topk_walkforward"
    )

    return module.TopKWalkForwardEvaluator


def _predictions() -> tuple[tuple[int, ...], ...]:
    return tuple(
        (
            index,
            index + 1,
            index + 2,
            index + 20,
            index + 21,
            index + 22,
        )
        for index in range(1, 11)
    )


def test_evaluator_is_read_only(
    tmp_path: Path,
) -> None:
    Evaluator = _api()

    state_root = tmp_path / "state"

    state_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    marker = state_root / "marker.json"

    marker.write_text(
        '{"stable": true}\n',
        encoding="utf-8",
    )

    before = {
        path.relative_to(
            state_root
        ).as_posix(): path.read_bytes()
        for path in state_root.rglob("*")
        if path.is_file()
    }

    candidate = ReplayRow(
        round_no=1200,
        history_rounds=(
            1196,
            1197,
            1198,
            1199,
        ),
        actual_numbers=(
            1,
            2,
            3,
            4,
            5,
            6,
        ),
        predictions=_predictions(),
    )

    baseline = ReplayRow(
        round_no=1200,
        history_rounds=(
            1196,
            1197,
            1198,
            1199,
        ),
        actual_numbers=(
            1,
            2,
            3,
            4,
            5,
            6,
        ),
        predictions=_predictions(),
        model_name="baseline",
    )

    evaluator = Evaluator(
        baseline_provider=BaselineProvider(
            {
                1200: baseline,
            }
        )
    )

    evaluator.evaluate(
        window=EvaluationWindow(
            name="w1",
            start_round=1200,
            end_round=1200,
        ),
        replay_rows=(
            candidate,
        ),
    )

    after = {
        path.relative_to(
            state_root
        ).as_posix(): path.read_bytes()
        for path in state_root.rglob("*")
        if path.is_file()
    }

    assert after == before
