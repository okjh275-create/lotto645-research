from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from lrp.evolution.algorithms.reinforcement import (
    ReinforcementUpdater,
)
from lrp.evolution.contracts.learning_context import (
    LearningContext,
)
from lrp.evolution.contracts.learning_cycle import (
    LearningCycleResult,
    LearningCycleStep,
)
from lrp.evolution.contracts.reinforcement import (
    RewardFeedback,
)
from lrp.evolution.services.learning_cycle import (
    LearningCycle,
)


def test_default_cycle_creation() -> None:
    cycle = LearningCycle()

    assert isinstance(
        cycle._updater,  # noqa: SLF001
        ReinforcementUpdater,
    )


def test_custom_updater_is_accepted() -> None:
    updater = ReinforcementUpdater()

    cycle = LearningCycle(
        updater=updater,
    )

    assert cycle._updater is updater  # noqa: SLF001


def test_invalid_updater_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="ReinforcementUpdater",
    ):
        LearningCycle(
            updater=object(),  # type: ignore[arg-type]
        )


def test_empty_cycle_preserves_context() -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        version=3,
    )

    result = LearningCycle().run(
        context=context,
        feedbacks=(),
    )

    assert result.initial_context is context
    assert result.final_context is context
    assert result.steps == ()
    assert result.step_count == 0
    assert result.version_delta == 0


def test_single_feedback_cycle() -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
    )
    feedback = RewardFeedback(
        source="draw_result",
        policy="ucb1",
        arm="strategy_a",
        reward=0.75,
    )

    result = LearningCycle().run(
        context=context,
        feedbacks=(feedback,),
    )

    assert result.final_context.rewards == {
        "draw_result:ucb1:strategy_a": 0.75,
    }
    assert result.final_context.version == 2
    assert result.step_count == 1
    assert result.version_delta == 1


def test_multiple_feedbacks_are_applied_in_order() -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        version=4,
    )
    feedbacks = (
        RewardFeedback(
            source="draw_result",
            policy="ucb1",
            arm="strategy_a",
            reward=0.75,
        ),
        RewardFeedback(
            source="validation",
            policy="thompson",
            arm="strategy_b",
            reward=0.25,
        ),
    )

    result = LearningCycle().run(
        context=context,
        feedbacks=feedbacks,
    )

    assert result.final_context.rewards == {
        "draw_result:ucb1:strategy_a": 0.75,
        "validation:thompson:strategy_b": 0.25,
    }
    assert result.final_context.version == 6
    assert result.final_context.selected_policy == (
        "thompson"
    )
    assert result.final_context.selected_arm == (
        "strategy_b"
    )


def test_cycle_creates_contiguous_steps() -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        version=7,
    )
    feedbacks = (
        RewardFeedback(
            source="result",
            arm="strategy_a",
            reward=0.4,
        ),
        RewardFeedback(
            source="result",
            arm="strategy_b",
            reward=0.6,
        ),
    )

    result = LearningCycle().run(
        context=context,
        feedbacks=feedbacks,
    )

    assert result.steps == (
        LearningCycleStep(
            index=1,
            name="reinforcement_feedback",
            version_before=7,
            version_after=8,
            reward_key="result:strategy_a",
        ),
        LearningCycleStep(
            index=2,
            name="reinforcement_feedback",
            version_before=8,
            version_after=9,
            reward_key="result:strategy_b",
        ),
    )


def test_cycle_does_not_mutate_initial_context() -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        rewards={
            "existing:strategy_c": 0.1,
        },
    )

    result = LearningCycle().run(
        context=context,
        feedbacks=(
            RewardFeedback(
                source="result",
                arm="strategy_a",
                reward=0.7,
            ),
        ),
    )

    assert context.rewards == {
        "existing:strategy_c": 0.1,
    }
    assert context.version == 1
    assert result.final_context is not context


def test_cycle_accepts_generator() -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
    )

    feedbacks = (
        RewardFeedback(
            source="result",
            arm=f"strategy_{index}",
            reward=0.5,
        )
        for index in range(2)
    )

    result = LearningCycle().run(
        context=context,
        feedbacks=feedbacks,
    )

    assert result.step_count == 2
    assert result.final_context.version == 3


def test_result_metadata_is_read_only() -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
    )

    result = LearningCycle().run(
        context=context,
        feedbacks=(),
    )

    assert isinstance(
        result.metadata,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        result.metadata["feedback_count"] = 1  # type: ignore[index]


def test_result_is_frozen() -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
    )

    result = LearningCycle().run(
        context=context,
        feedbacks=(),
    )

    with pytest.raises(FrozenInstanceError):
        result.steps = ()  # type: ignore[misc]


def test_result_snapshot_is_detached() -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
    )

    result = LearningCycle().run(
        context=context,
        feedbacks=(
            RewardFeedback(
                source="result",
                arm="strategy_a",
                reward=0.7,
            ),
        ),
    )

    snapshot = result.snapshot()

    assert snapshot["step_count"] == 1
    assert snapshot["version_delta"] == 1

    final_context = snapshot["final_context"]
    assert isinstance(final_context, dict)

    rewards = final_context["rewards"]
    assert isinstance(rewards, dict)

    rewards["result:strategy_a"] = 0.1

    assert result.final_context.rewards[
        "result:strategy_a"
    ] == pytest.approx(0.7)


def test_invalid_context_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="context must be",
    ):
        LearningCycle().run(
            context=object(),  # type: ignore[arg-type]
            feedbacks=(),
        )


def test_non_iterable_feedbacks_are_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="feedbacks must be iterable",
    ):
        LearningCycle().run(
            context=LearningContext(
                cycle_id="cycle-1220",
                round_no=1220,
            ),
            feedbacks=1,  # type: ignore[arg-type]
        )


def test_string_feedbacks_are_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="iterable of RewardFeedback",
    ):
        LearningCycle().run(
            context=LearningContext(
                cycle_id="cycle-1220",
                round_no=1220,
            ),
            feedbacks="invalid",  # type: ignore[arg-type]
        )


def test_invalid_feedback_item_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="contain only RewardFeedback",
    ):
        LearningCycle().run(
            context=LearningContext(
                cycle_id="cycle-1220",
                round_no=1220,
            ),
            feedbacks=(
                object(),
            ),  # type: ignore[arg-type]
        )


def test_step_rejects_non_contiguous_versions() -> None:
    with pytest.raises(
        ValueError,
        match="version_after must be greater",
    ):
        LearningCycleStep(
            index=1,
            name="reinforcement_feedback",
            version_before=3,
            version_after=3,
            reward_key="result:strategy_a",
        )


def test_result_rejects_changed_cycle_id() -> None:
    with pytest.raises(
        ValueError,
        match="cycle_id must not change",
    ):
        LearningCycleResult(
            initial_context=LearningContext(
                cycle_id="cycle-a",
                round_no=1220,
            ),
            final_context=LearningContext(
                cycle_id="cycle-b",
                round_no=1220,
            ),
            steps=(),
        )


def test_result_rejects_changed_round_number() -> None:
    with pytest.raises(
        ValueError,
        match="round_no must not change",
    ):
        LearningCycleResult(
            initial_context=LearningContext(
                cycle_id="cycle-a",
                round_no=1220,
            ),
            final_context=LearningContext(
                cycle_id="cycle-a",
                round_no=1221,
            ),
            steps=(),
        )


def test_result_rejects_non_contiguous_steps() -> None:
    initial = LearningContext(
        cycle_id="cycle-a",
        round_no=1220,
        version=1,
    )
    final = LearningContext(
        cycle_id="cycle-a",
        round_no=1220,
        version=4,
    )

    with pytest.raises(
        ValueError,
        match="must be contiguous",
    ):
        LearningCycleResult(
            initial_context=initial,
            final_context=final,
            steps=(
                LearningCycleStep(
                    index=1,
                    name="reinforcement_feedback",
                    version_before=1,
                    version_after=2,
                    reward_key="result:strategy_a",
                ),
                LearningCycleStep(
                    index=2,
                    name="reinforcement_feedback",
                    version_before=3,
                    version_after=4,
                    reward_key="result:strategy_b",
                ),
            ),
        )
