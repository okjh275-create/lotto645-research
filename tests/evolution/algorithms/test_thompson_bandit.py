from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from lrp.evolution.algorithms.thompson import (
    ThompsonBandit,
)
from lrp.evolution.contracts.thompson import (
    BetaArmStatistics,
    ThompsonDecision,
)


def test_beta_arm_normalizes_name() -> None:
    arm = BetaArmStatistics(arm=" hot ")

    assert arm.arm == "hot"


def test_beta_arm_is_frozen() -> None:
    arm = BetaArmStatistics(arm="hot")

    with pytest.raises(FrozenInstanceError):
        arm.successes = 1.0  # type: ignore[misc]


def test_default_posterior() -> None:
    arm = BetaArmStatistics(arm="hot")

    assert arm.alpha == pytest.approx(1.0)
    assert arm.beta == pytest.approx(1.0)
    assert arm.posterior_mean == pytest.approx(0.5)


def test_posterior_parameters() -> None:
    arm = BetaArmStatistics(
        arm="hot",
        successes=4.0,
        failures=2.0,
        prior_alpha=2.0,
        prior_beta=3.0,
    )

    assert arm.alpha == pytest.approx(6.0)
    assert arm.beta == pytest.approx(5.0)
    assert arm.observations == pytest.approx(6.0)


def test_record_success() -> None:
    original = BetaArmStatistics(arm="hot")
    updated = original.record(1.0)

    assert original.successes == pytest.approx(0.0)
    assert updated.successes == pytest.approx(1.0)
    assert updated.failures == pytest.approx(0.0)


def test_record_failure() -> None:
    updated = BetaArmStatistics(
        arm="hot"
    ).record(0.0)

    assert updated.successes == pytest.approx(0.0)
    assert updated.failures == pytest.approx(1.0)


def test_record_fractional_reward() -> None:
    updated = BetaArmStatistics(
        arm="hot"
    ).record(0.25)

    assert updated.successes == pytest.approx(0.25)
    assert updated.failures == pytest.approx(0.75)


@pytest.mark.parametrize("reward", [-0.1, 1.1])
def test_invalid_reward_is_rejected(
    reward: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="between 0.0 and 1.0",
    ):
        BetaArmStatistics(
            arm="hot"
        ).record(reward)


def test_negative_successes_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="greater than or equal to 0",
    ):
        BetaArmStatistics(
            arm="hot",
            successes=-1.0,
        )


def test_zero_prior_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="greater than 0",
    ):
        BetaArmStatistics(
            arm="hot",
            prior_alpha=0.0,
        )


def test_select_returns_decision() -> None:
    decision = ThompsonBandit(seed=7).select(
        [
            BetaArmStatistics(arm="hot"),
        ]
    )

    assert isinstance(decision, ThompsonDecision)


def test_same_seed_is_reproducible() -> None:
    arms = [
        BetaArmStatistics(
            arm="hot",
            successes=4.0,
            failures=2.0,
        ),
        BetaArmStatistics(
            arm="gap",
            successes=2.0,
            failures=4.0,
        ),
    ]

    first = ThompsonBandit(seed=42).select(arms)
    second = ThompsonBandit(seed=42).select(arms)

    assert first == second


def test_decision_records_seed() -> None:
    decision = ThompsonBandit(seed=123).select(
        [
            BetaArmStatistics(arm="hot"),
        ]
    )

    assert decision.seed == 123


def test_decision_sample_is_probability() -> None:
    decision = ThompsonBandit(seed=1).select(
        [
            BetaArmStatistics(arm="hot"),
        ]
    )

    assert 0.0 <= decision.sample <= 1.0


def test_strong_arm_is_selected_for_fixed_seed() -> None:
    decision = ThompsonBandit(seed=10).select(
        [
            BetaArmStatistics(
                arm="strong",
                successes=50.0,
                failures=2.0,
            ),
            BetaArmStatistics(
                arm="weak",
                successes=2.0,
                failures=50.0,
            ),
        ]
    )

    assert decision.arm == "strong"


def test_selected_observations_are_reported() -> None:
    decision = ThompsonBandit(seed=10).select(
        [
            BetaArmStatistics(
                arm="strong",
                successes=5.0,
                failures=2.0,
            ),
        ]
    )

    assert decision.observations == pytest.approx(7.0)


def test_empty_arms_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="at least one arm",
    ):
        ThompsonBandit(seed=1).select([])


def test_duplicate_arms_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="duplicate arm",
    ):
        ThompsonBandit(seed=1).select(
            [
                BetaArmStatistics(arm="hot"),
                BetaArmStatistics(arm="hot"),
            ]
        )


def test_invalid_item_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="every arm",
    ):
        ThompsonBandit(seed=1).select(
            [object()]  # type: ignore[list-item]
        )


@pytest.mark.parametrize(
    "seed",
    [1.2, "1", True],
)
def test_invalid_seed_is_rejected(
    seed: object,
) -> None:
    with pytest.raises(TypeError):
        ThompsonBandit(
            seed=seed  # type: ignore[arg-type]
        )


def test_input_order_breaks_exact_sample_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lrp.evolution.algorithms.thompson."
        "Random.betavariate",
        lambda self, alpha, beta: 0.5,
    )

    decision = ThompsonBandit(seed=1).select(
        [
            BetaArmStatistics(arm="first"),
            BetaArmStatistics(arm="second"),
        ]
    )

    assert decision.arm == "first"
