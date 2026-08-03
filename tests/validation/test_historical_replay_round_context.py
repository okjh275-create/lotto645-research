from __future__ import annotations

from pathlib import Path

from lrp.evolution.contracts.learning_context import (
    LearningContext,
)
from tools.validation.historical_replay_executor import (
    HistoricalReplayExecutor,
    ReplayState,
)
from tools.validation.historical_replay_models import (
    ReplayConfig,
)


def make_executor(
    tmp_path: Path,
) -> HistoricalReplayExecutor:
    return HistoricalReplayExecutor(
        history=(object(),),
        config=ReplayConfig(
            start_round=1132,
            end_round=1231,
            seed_base=20260802,
            candidate_count=1000,
            top_k=20,
            practical_k=5,
            mode="fast",
        ),
        learning_root=tmp_path / "learning",
        profile_root=tmp_path / "profiles",
    )


def test_carried_learning_context_advances_round(
    tmp_path: Path,
) -> None:
    executor = make_executor(tmp_path)

    previous = LearningContext(
        cycle_id=(
            "historical-replay-1132-1231"
        ),
        round_no=1132,
        metadata={
            "review_count": 1,
        },
    )

    current = executor._learning_context(
        state=ReplayState(
            learning_context=previous
        ),
        round_no=1133,
    )

    assert current.round_no == 1133
    assert current.cycle_id == previous.cycle_id
    assert current.metadata == previous.metadata
    assert current is not previous


def test_initial_learning_context_uses_current_round(
    tmp_path: Path,
) -> None:
    executor = make_executor(tmp_path)

    context = executor._learning_context(
        state=None,
        round_no=1132,
    )

    assert context.round_no == 1132
    assert context.cycle_id == (
        "historical-replay-1132-1231"
    )
