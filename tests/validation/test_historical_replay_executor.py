from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from tools.validation.historical_replay_executor import (
    HistoricalReplayExecutor,
    ReplayState,
)
from tools.validation.historical_replay_models import (
    ReplayConfig,
)


@dataclass(frozen=True)
class ProbabilityItemStub:
    number: int
    probability: float


@dataclass(frozen=True)
class ProbabilityVectorStub:
    probabilities: tuple[
        ProbabilityItemStub,
        ...
    ]


@dataclass(frozen=True)
class GenerationStub:
    probability_vector: ProbabilityVectorStub


@dataclass(frozen=True)
class ResultStub:
    generation: GenerationStub


@dataclass(frozen=True)
class DrawStub:
    numbers: tuple[int, ...]
    bonus: int


def make_result(
    *,
    delta: float = 0.0,
) -> ResultStub:
    values = []

    base = 1.0 / 45.0

    for number in range(1, 46):
        adjustment = (
            delta
            if number == 1
            else (
                -delta
                if number == 2
                else 0.0
            )
        )

        values.append(
            ProbabilityItemStub(
                number=number,
                probability=(
                    base + adjustment
                ),
            )
        )

    return ResultStub(
        generation=GenerationStub(
            probability_vector=(
                ProbabilityVectorStub(
                    probabilities=tuple(
                        values
                    )
                )
            )
        )
    )


def make_executor(
    tmp_path: Path,
) -> HistoricalReplayExecutor:
    return HistoricalReplayExecutor(
        history=(object(),),
        config=ReplayConfig(
            start_round=1222,
            end_round=1222,
        ),
        learning_root=tmp_path / "learning",
        profile_root=tmp_path / "profiles",
    )


def test_probability_metrics(
    tmp_path: Path,
) -> None:
    executor = make_executor(tmp_path)

    metrics = executor._probability_metrics(
        make_result(),
        make_result(delta=0.001),
    )

    assert metrics["changed"] == 2
    assert metrics["l1"] == pytest.approx(0.002)
    assert metrics["max"] == pytest.approx(0.001)


def test_changed_set_count(
    tmp_path: Path,
) -> None:
    executor = make_executor(tmp_path)

    noop = {
        "sets": [
            {
                "id": "S1",
                "numbers": [1, 2, 3, 4, 5, 6],
            },
            {
                "id": "S2",
                "numbers": [7, 8, 9, 10, 11, 12],
            },
        ]
    }

    adaptive = {
        "sets": [
            {
                "id": "S1",
                "numbers": [1, 2, 3, 4, 5, 6],
            },
            {
                "id": "S2",
                "numbers": [7, 8, 9, 10, 11, 13],
            },
        ]
    }

    assert executor._changed_set_count(
        noop,
        adaptive,
    ) == 1


def test_draw_accessors(
    tmp_path: Path,
) -> None:
    executor = make_executor(tmp_path)

    draw = DrawStub(
        numbers=(
            4,
            13,
            14,
            18,
            31,
            38,
        ),
        bonus=15,
    )

    assert executor._draw_numbers(draw) == (
        4,
        13,
        14,
        18,
        31,
        38,
    )
    assert executor._draw_bonus(draw) == 15


def test_replay_state_holds_context(
    tmp_path: Path,
) -> None:
    executor = make_executor(tmp_path)

    state = ReplayState(
        learning_context=object()
    )

    assert state.learning_context is not None
    assert executor.policy == "thompson"
