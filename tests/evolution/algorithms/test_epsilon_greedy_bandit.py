from __future__ import annotations

from typing import cast

import pytest

from lrp.evolution.algorithms.bandit import (
    UCB1Bandit,
)
from lrp.evolution.algorithms.epsilon_greedy import (
    EpsilonGreedyBandit,
)
from lrp.evolution.algorithms.thompson import (
    ThompsonBandit,
)
from lrp.evolution.contracts.bandit import (
    ArmStatistics,
    BanditDecision,
)
from lrp.evolution.contracts.policy import (
    BanditPolicy,
)
from lrp.evolution.contracts.thompson import (
    BetaArmStatistics,
    ThompsonDecision,
)


def test_default_configuration() -> None:
    bandit = EpsilonGreedyBandit()

    assert bandit.epsilon == pytest.approx(0.1)
    assert bandit.seed is None


def test_select_returns_decision() -> None:
    decision = EpsilonGreedyBandit(
        epsilon=0.0,
        seed=1,
    ).select(
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


def test_zero_epsilon_always_exploits() -> None:
    decision = EpsilonGreedyBandit(
        epsilon=0.0,
        seed=7,
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
        "greedy_exploitation"
    )
    assert decision.score == pytest.approx(0.8)


def test_one_epsilon_always_explores() -> None:
    decision = EpsilonGreedyBandit(
        epsilon=1.0,
        seed=7,
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

    assert decision.reason == (
        "random_exploration"
    )
    assert decision.arm in {
        "hot",
        "gap",
    }


def test_greedy_selection_uses_mean_reward() -> None:
    decision = EpsilonGreedyBandit(
        epsilon=0.0,
        seed=1,
    ).select(
        [
            ArmStatistics(
                arm="many_pulls",
                pulls=100,
                reward_sum=60.0,
            ),
            ArmStatistics(
                arm="few_pulls",
                pulls=2,
                reward_sum=1.5,
            ),
        ]
    )

    assert decision.arm == "few_pulls"
    assert decision.score == pytest.approx(0.75)


def test_input_order_breaks_exploitation_tie() -> None:
    decision = EpsilonGreedyBandit(
        epsilon=0.0,
        seed=1,
    ).select(
        [
            ArmStatistics(
                arm="first",
                pulls=2,
                reward_sum=1.0,
            ),
            ArmStatistics(
                arm="second",
                pulls=4,
                reward_sum=2.0,
            ),
        ]
    )

    assert decision.arm == "first"


def test_untried_arm_has_zero_greedy_score() -> None:
    decision = EpsilonGreedyBandit(
        epsilon=0.0,
        seed=1,
    ).select(
        [
            ArmStatistics(
                arm="untried",
            ),
            ArmStatistics(
                arm="negative",
                pulls=2,
                reward_sum=-1.0,
            ),
        ]
    )

    assert decision.arm == "untried"
    assert decision.score == pytest.approx(0.0)


def test_total_pulls_are_reported() -> None:
    decision = EpsilonGreedyBandit(
        epsilon=0.0,
        seed=1,
    ).select(
        [
            ArmStatistics(
                arm="hot",
                pulls=3,
                reward_sum=1.0,
            ),
            ArmStatistics(
                arm="gap",
                pulls=7,
                reward_sum=2.0,
            ),
        ]
    )

    assert decision.total_pulls == 10


def test_same_seed_produces_same_sequence() -> None:
    arms = [
        ArmStatistics(
            arm="hot",
            pulls=2,
            reward_sum=1.0,
        ),
        ArmStatistics(
            arm="gap",
            pulls=2,
            reward_sum=1.0,
        ),
        ArmStatistics(
            arm="trend",
            pulls=2,
            reward_sum=1.0,
        ),
    ]

    first = EpsilonGreedyBandit(
        epsilon=1.0,
        seed=42,
    )
    second = EpsilonGreedyBandit(
        epsilon=1.0,
        seed=42,
    )

    first_sequence = [
        first.select(arms).arm
        for _ in range(10)
    ]
    second_sequence = [
        second.select(arms).arm
        for _ in range(10)
    ]

    assert first_sequence == second_sequence


def test_reset_restarts_random_sequence() -> None:
    arms = [
        ArmStatistics(arm="hot"),
        ArmStatistics(arm="gap"),
        ArmStatistics(arm="trend"),
    ]
    bandit = EpsilonGreedyBandit(
        epsilon=1.0,
        seed=123,
    )

    first_sequence = [
        bandit.select(arms).arm
        for _ in range(8)
    ]

    bandit.reset()

    second_sequence = [
        bandit.select(arms).arm
        for _ in range(8)
    ]

    assert first_sequence == second_sequence


def test_duplicate_arms_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="duplicate arm",
    ):
        EpsilonGreedyBandit().select(
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
        EpsilonGreedyBandit().select([])


def test_invalid_item_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="every arm",
    ):
        EpsilonGreedyBandit().select(
            [object()]  # type: ignore[list-item]
        )


@pytest.mark.parametrize(
    "epsilon",
    [-0.1, 1.1, float("inf")],
)
def test_invalid_epsilon_is_rejected(
    epsilon: float,
) -> None:
    with pytest.raises(ValueError):
        EpsilonGreedyBandit(
            epsilon=epsilon
        )


@pytest.mark.parametrize(
    "epsilon",
    [True, "0.1", object()],
)
def test_non_numeric_epsilon_is_rejected(
    epsilon: object,
) -> None:
    with pytest.raises(TypeError):
        EpsilonGreedyBandit(
            epsilon=epsilon,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "seed",
    [True, 1.5, "1"],
)
def test_invalid_seed_is_rejected(
    seed: object,
) -> None:
    with pytest.raises(TypeError):
        EpsilonGreedyBandit(
            seed=seed,  # type: ignore[arg-type]
        )


def test_epsilon_greedy_matches_policy_protocol() -> None:
    policy = cast(
        BanditPolicy[
            ArmStatistics,
            BanditDecision,
        ],
        EpsilonGreedyBandit(
            epsilon=0.0,
            seed=1,
        ),
    )

    decision = policy.select(
        [
            ArmStatistics(
                arm="hot",
                pulls=1,
                reward_sum=1.0,
            ),
        ]
    )

    assert decision.arm == "hot"


def test_ucb1_matches_policy_protocol() -> None:
    policy = cast(
        BanditPolicy[
            ArmStatistics,
            BanditDecision,
        ],
        UCB1Bandit(),
    )

    decision = policy.select(
        [
            ArmStatistics(
                arm="hot",
                pulls=1,
                reward_sum=1.0,
            ),
        ]
    )

    assert decision.arm == "hot"


def test_thompson_matches_policy_protocol() -> None:
    policy = cast(
        BanditPolicy[
            BetaArmStatistics,
            ThompsonDecision,
        ],
        ThompsonBandit(seed=1),
    )

    decision = policy.select(
        [
            BetaArmStatistics(
                arm="hot",
            ),
        ]
    )

    assert decision.arm == "hot"


def test_runtime_protocol_check() -> None:
    assert isinstance(
        EpsilonGreedyBandit(),
        BanditPolicy,
    )
    assert isinstance(
        UCB1Bandit(),
        BanditPolicy,
    )
    assert isinstance(
        ThompsonBandit(seed=1),
        BanditPolicy,
    )
