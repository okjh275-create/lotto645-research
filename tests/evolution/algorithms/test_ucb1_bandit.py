from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import log, sqrt

import pytest

from lrp.evolution.algorithms.bandit import (
    UCB1Bandit,
)
from lrp.evolution.contracts.bandit import (
    ArmStatistics,
    BanditDecision,
)


def test_arm_statistics_normalizes_name() -> None:
    statistics = ArmStatistics(
        arm=" hot ",
    )

    assert statistics.arm == "hot"


def test_arm_statistics_is_frozen() -> None:
    statistics = ArmStatistics(
        arm="hot",
    )

    with pytest.raises(FrozenInstanceError):
        statistics.pulls = 1  # type: ignore[misc]


def test_untried_arm_has_zero_mean() -> None:
    statistics = ArmStatistics(
        arm="hot",
    )

    assert statistics.mean_reward == 0.0


def test_arm_mean_reward() -> None:
    statistics = ArmStatistics(
        arm="hot",
        pulls=4,
        reward_sum=2.0,
    )

    assert statistics.mean_reward == pytest.approx(
        0.5
    )


def test_record_returns_new_statistics() -> None:
    original = ArmStatistics(
        arm="hot",
        pulls=2,
        reward_sum=0.5,
    )

    updated = original.record(0.75)

    assert original.pulls == 2
    assert updated.pulls == 3
    assert updated.reward_sum == pytest.approx(
        1.25
    )


@pytest.mark.parametrize(
    "reward",
    [-1.1, 1.1],
)
def test_record_rejects_out_of_range_reward(
    reward: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="between -1.0 and 1.0",
    ):
        ArmStatistics(
            arm="hot",
        ).record(reward)


def test_empty_arm_name_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="arm must not be empty",
    ):
        ArmStatistics(
            arm=" ",
        )


def test_negative_pulls_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="greater than or equal to 0",
    ):
        ArmStatistics(
            arm="hot",
            pulls=-1,
        )


def test_reward_requires_pulls() -> None:
    with pytest.raises(
        ValueError,
        match="reward_sum must be 0.0",
    ):
        ArmStatistics(
            arm="hot",
            pulls=0,
            reward_sum=1.0,
        )


def test_bandit_decision_is_frozen() -> None:
    decision = BanditDecision(
        arm="hot",
        score=1.0,
        reason="test",
        total_pulls=3,
    )

    with pytest.raises(FrozenInstanceError):
        decision.arm = "gap"  # type: ignore[misc]


def test_select_returns_decision() -> None:
    decision = UCB1Bandit().select(
        [
            ArmStatistics(
                arm="hot",
                pulls=2,
                reward_sum=1.0,
            ),
        ]
    )

    assert isinstance(
        decision,
        BanditDecision,
    )


def test_untried_arm_is_selected_first() -> None:
    decision = UCB1Bandit().select(
        [
            ArmStatistics(
                arm="hot",
                pulls=10,
                reward_sum=8.0,
            ),
            ArmStatistics(
                arm="gap",
            ),
        ]
    )

    assert decision.arm == "gap"
    assert decision.reason == "untried_arm"


def test_untried_selection_is_deterministic() -> None:
    decision = UCB1Bandit().select(
        [
            ArmStatistics(arm="trend"),
            ArmStatistics(arm="gap"),
            ArmStatistics(arm="hot"),
        ]
    )

    assert decision.arm == "gap"


def test_total_pulls_are_reported() -> None:
    decision = UCB1Bandit().select(
        [
            ArmStatistics(
                arm="hot",
                pulls=3,
                reward_sum=1.0,
            ),
            ArmStatistics(
                arm="gap",
            ),
        ]
    )

    assert decision.total_pulls == 3


def test_ucb_score_formula() -> None:
    bandit = UCB1Bandit(
        exploration_strength=sqrt(2.0),
    )
    arm = ArmStatistics(
        arm="hot",
        pulls=5,
        reward_sum=3.0,
    )

    score = bandit.score(
        arm=arm,
        total_pulls=20,
    )

    expected = (
        0.6
        + sqrt(2.0)
        * sqrt(log(20) / 5)
    )

    assert score == pytest.approx(expected)


def test_high_mean_arm_can_be_selected() -> None:
    decision = UCB1Bandit(
        exploration_strength=0.0,
    ).select(
        [
            ArmStatistics(
                arm="hot",
                pulls=10,
                reward_sum=8.0,
            ),
            ArmStatistics(
                arm="gap",
                pulls=10,
                reward_sum=4.0,
            ),
        ]
    )

    assert decision.arm == "hot"
    assert decision.reason == (
        "highest_ucb_score"
    )


def test_less_sampled_arm_gets_exploration_bonus() -> None:
    decision = UCB1Bandit().select(
        [
            ArmStatistics(
                arm="stable",
                pulls=90,
                reward_sum=54.0,
            ),
            ArmStatistics(
                arm="explore",
                pulls=10,
                reward_sum=6.0,
            ),
        ]
    )

    assert decision.arm == "explore"


def test_input_order_breaks_equal_score_tie() -> None:
    decision = UCB1Bandit(
        exploration_strength=0.0,
    ).select(
        [
            ArmStatistics(
                arm="first",
                pulls=2,
                reward_sum=1.0,
            ),
            ArmStatistics(
                arm="second",
                pulls=2,
                reward_sum=1.0,
            ),
        ]
    )

    assert decision.arm == "first"


def test_duplicate_arms_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="duplicate arm",
    ):
        UCB1Bandit().select(
            [
                ArmStatistics(arm="hot"),
                ArmStatistics(arm="hot"),
            ]
        )


def test_empty_arms_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="at least one arm",
    ):
        UCB1Bandit().select([])


def test_invalid_arm_item_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="every arm",
    ):
        UCB1Bandit().select(
            [object()]  # type: ignore[list-item]
        )


def test_untried_arm_score_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="untried arm",
    ):
        UCB1Bandit().score(
            arm=ArmStatistics(arm="hot"),
            total_pulls=1,
        )


def test_total_pulls_cannot_be_too_small() -> None:
    with pytest.raises(
        ValueError,
        match="must not be less",
    ):
        UCB1Bandit().score(
            arm=ArmStatistics(
                arm="hot",
                pulls=3,
                reward_sum=1.0,
            ),
            total_pulls=2,
        )


@pytest.mark.parametrize(
    "exploration_strength",
    [-0.1, float("inf")],
)
def test_invalid_exploration_strength(
    exploration_strength: float,
) -> None:
    with pytest.raises(ValueError):
        UCB1Bandit(
            exploration_strength=(
                exploration_strength
            )
        )
