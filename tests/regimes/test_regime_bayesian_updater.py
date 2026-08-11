from __future__ import annotations

import pytest

from lrp.regimes.bayesian_evidence import (
    RegimeBayesianEvidenceAdapter,
)
from lrp.regimes.bayesian_state import (
    RegimeBayesianState,
)
from lrp.regimes.bayesian_updater import (
    RegimeBayesianUpdater,
)
from lrp.regimes.reward import RegimeReward


def make_reward(
    regime: str,
    reward: float,
) -> RegimeReward:
    return RegimeReward(
        regime=regime,
        reward=reward,
        confidence=1.0,
        sample_weight=1.0,
    )


def test_positive_reward_updates_target_posterior() -> None:
    updater = RegimeBayesianUpdater(
        evidence_adapter=(
            RegimeBayesianEvidenceAdapter(
                evidence_scale=10
            )
        )
    )

    state = RegimeBayesianState.default()

    updated = updater.update(
        state,
        make_reward(
            "gap_recovery",
            0.8,
        ),
    )

    posterior = updated.posteriors[
        "gap_recovery"
    ]

    assert posterior.alpha == 9.0
    assert posterior.beta == 1.0


def test_negative_reward_updates_failure_evidence() -> None:
    updater = RegimeBayesianUpdater(
        evidence_adapter=(
            RegimeBayesianEvidenceAdapter(
                evidence_scale=10
            )
        )
    )

    updated = updater.update(
        RegimeBayesianState.default(),
        make_reward(
            "cluster_rotation",
            -0.6,
        ),
    )

    posterior = updated.posteriors[
        "cluster_rotation"
    ]

    assert posterior.alpha == 1.0
    assert posterior.beta == 7.0


def test_only_target_regime_changes() -> None:
    updater = RegimeBayesianUpdater()

    state = RegimeBayesianState.default()

    updated = updater.update(
        state,
        make_reward(
            "high_band_expansion",
            1.0,
        ),
    )

    assert (
        updated.posteriors["gap_recovery"]
        == state.posteriors["gap_recovery"]
    )
    assert (
        updated.posteriors["cluster_rotation"]
        == state.posteriors["cluster_rotation"]
    )
    assert (
        updated.posteriors["low_band_expansion"]
        == state.posteriors["low_band_expansion"]
    )
    assert (
        updated.posteriors["high_band_expansion"]
        != state.posteriors["high_band_expansion"]
    )


def test_zero_reward_preserves_posterior_values() -> None:
    updater = RegimeBayesianUpdater()

    state = RegimeBayesianState.default()

    updated = updater.update(
        state,
        make_reward(
            "gap_recovery",
            0.0,
        ),
    )

    assert updated.posteriors == state.posteriors
    assert updated is not state


@pytest.mark.parametrize(
    "regime",
    [
        "neutral",
        "mixed",
    ],
)
def test_non_adaptive_regime_is_noop(
    regime: str,
) -> None:
    updater = RegimeBayesianUpdater()

    state = RegimeBayesianState.default()

    updated = updater.update(
        state,
        make_reward(
            regime,
            1.0,
        ),
    )

    assert updated.posteriors == state.posteriors


def test_posterior_signal_moves_positive() -> None:
    updater = RegimeBayesianUpdater()

    updated = updater.update(
        RegimeBayesianState.default(),
        make_reward(
            "gap_recovery",
            1.0,
        ),
    )

    assert (
        updated.to_signals()[
            "gap_recovery"
        ]
        > 0.0
    )


def test_invalid_state_is_rejected() -> None:
    updater = RegimeBayesianUpdater()

    with pytest.raises(
        TypeError,
        match="RegimeBayesianState",
    ):
        updater.update(
            object(),  # type: ignore[arg-type]
            make_reward(
                "gap_recovery",
                1.0,
            ),
        )


def test_invalid_reward_is_rejected() -> None:
    updater = RegimeBayesianUpdater()

    with pytest.raises(
        TypeError,
        match="RegimeReward",
    ):
        updater.update(
            RegimeBayesianState.default(),
            object(),  # type: ignore[arg-type]
        )