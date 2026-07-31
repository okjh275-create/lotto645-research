from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from lrp.evolution.algorithms.reinforcement import (
    ReinforcementUpdater,
)
from lrp.evolution.contracts.learning_context import (
    LearningContext,
)
from lrp.evolution.contracts.reinforcement import (
    RewardFeedback,
)


def test_feedback_normalizes_text() -> None:
    feedback = RewardFeedback(
        source=" result ",
        arm=" strategy_a ",
        reward=0.5,
        policy=" ucb1 ",
    )

    assert feedback.source == "result"
    assert feedback.arm == "strategy_a"
    assert feedback.policy == "ucb1"


def test_feedback_is_frozen() -> None:
    feedback = RewardFeedback(
        source="result",
        arm="strategy_a",
        reward=0.5,
    )

    with pytest.raises(FrozenInstanceError):
        feedback.reward = 0.1  # type: ignore[misc]


@pytest.mark.parametrize(
    "reward",
    [-1.0, 0.0, 1.0],
)
def test_reward_boundaries_are_valid(
    reward: float,
) -> None:
    feedback = RewardFeedback(
        source="result",
        arm="strategy_a",
        reward=reward,
    )

    assert feedback.reward == pytest.approx(
        reward
    )


@pytest.mark.parametrize(
    "reward",
    [-1.1, 1.1],
)
def test_out_of_range_reward_is_rejected(
    reward: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="between -1.0 and 1.0",
    ):
        RewardFeedback(
            source="result",
            arm="strategy_a",
            reward=reward,
        )


def test_boolean_reward_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="reward must be numeric",
    ):
        RewardFeedback(
            source="result",
            arm="strategy_a",
            reward=True,
        )


def test_invalid_observation_count_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="greater than or equal to 1",
    ):
        RewardFeedback(
            source="result",
            arm="strategy_a",
            reward=0.5,
            observation_count=0,
        )


def test_apply_returns_new_context() -> None:
    original = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
    )
    feedback = RewardFeedback(
        source="draw_result",
        arm="strategy_a",
        reward=0.75,
    )

    updated = ReinforcementUpdater().apply(
        context=original,
        feedback=feedback,
    )

    assert updated is not original
    assert original.rewards == {}
    assert original.version == 1


def test_apply_stores_reward_without_policy() -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
    )
    feedback = RewardFeedback(
        source="draw_result",
        arm="strategy_a",
        reward=0.75,
    )

    updated = ReinforcementUpdater().apply(
        context=context,
        feedback=feedback,
    )

    assert updated.rewards == {
        "draw_result:strategy_a": 0.75,
    }


def test_apply_stores_reward_with_policy() -> None:
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

    updated = ReinforcementUpdater().apply(
        context=context,
        feedback=feedback,
    )

    assert updated.rewards == {
        "draw_result:ucb1:strategy_a": 0.75,
    }


def test_apply_preserves_existing_rewards() -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        rewards={
            "existing:strategy_b": 0.25,
        },
    )
    feedback = RewardFeedback(
        source="draw_result",
        arm="strategy_a",
        reward=0.75,
    )

    updated = ReinforcementUpdater().apply(
        context=context,
        feedback=feedback,
    )

    assert updated.rewards == {
        "existing:strategy_b": 0.25,
        "draw_result:strategy_a": 0.75,
    }


def test_apply_updates_selection_when_policy_exists() -> None:
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

    updated = ReinforcementUpdater().apply(
        context=context,
        feedback=feedback,
    )

    assert updated.selected_policy == "ucb1"
    assert updated.selected_arm == "strategy_a"


def test_apply_does_not_replace_selection_without_policy() -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        selected_policy="epsilon_greedy",
        selected_arm="strategy_b",
    )
    feedback = RewardFeedback(
        source="draw_result",
        arm="strategy_a",
        reward=0.75,
    )

    updated = ReinforcementUpdater().apply(
        context=context,
        feedback=feedback,
    )

    assert updated.selected_policy == (
        "epsilon_greedy"
    )
    assert updated.selected_arm == "strategy_b"


def test_apply_records_feedback_metadata() -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        metadata={
            "seed": 20260731,
        },
    )
    feedback = RewardFeedback(
        source="draw_result",
        policy="ucb1",
        arm="strategy_a",
        reward=0.75,
        observation_count=6,
    )

    updated = ReinforcementUpdater().apply(
        context=context,
        feedback=feedback,
    )

    assert updated.metadata == {
        "seed": 20260731,
        "feedback_source": "draw_result",
        "feedback_arm": "strategy_a",
        "feedback_observation_count": 6,
        "feedback_policy": "ucb1",
    }


def test_apply_advances_version_once() -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        version=4,
    )
    feedback = RewardFeedback(
        source="draw_result",
        arm="strategy_a",
        reward=0.75,
    )

    updated = ReinforcementUpdater().apply(
        context=context,
        feedback=feedback,
    )

    assert updated.version == 5


def test_same_reward_key_is_replaced() -> None:
    context = LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        rewards={
            "draw_result:strategy_a": 0.25,
        },
    )
    feedback = RewardFeedback(
        source="draw_result",
        arm="strategy_a",
        reward=0.8,
    )

    updated = ReinforcementUpdater().apply(
        context=context,
        feedback=feedback,
    )

    assert updated.rewards[
        "draw_result:strategy_a"
    ] == pytest.approx(0.8)


def test_invalid_context_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="context must be",
    ):
        ReinforcementUpdater().apply(
            context=object(),  # type: ignore[arg-type]
            feedback=RewardFeedback(
                source="result",
                arm="strategy_a",
                reward=0.5,
            ),
        )


def test_invalid_feedback_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="feedback must be",
    ):
        ReinforcementUpdater().apply(
            context=LearningContext(
                cycle_id="cycle-1220",
                round_no=1220,
            ),
            feedback=object(),  # type: ignore[arg-type]
        )
