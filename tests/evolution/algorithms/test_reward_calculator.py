from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from lrp.evolution.algorithms.reward import (
    RewardCalculator,
)
from lrp.evolution.algorithms.signals import (
    SignalAggregator,
)
from lrp.evolution.contracts.reward import (
    RewardObservation,
)
from lrp.evolution.contracts.signals import (
    SignalFrame,
)


def test_observation_normalizes_source() -> None:
    observation = RewardObservation(
        source=" weekly ",
        successes={"hot": 3},
        trials={"hot": 5},
    )

    assert observation.source == "weekly"


def test_observation_mappings_are_immutable() -> None:
    observation = RewardObservation(
        source="weekly",
        successes={"hot": 3},
        trials={"hot": 5},
    )

    assert isinstance(
        observation.successes,
        MappingProxyType,
    )
    assert isinstance(
        observation.trials,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        observation.successes["hot"] = 4  # type: ignore[index]


def test_observation_is_frozen() -> None:
    observation = RewardObservation(
        source="weekly",
        successes={"hot": 3},
        trials={"hot": 5},
    )

    with pytest.raises(FrozenInstanceError):
        observation.source = "changed"  # type: ignore[misc]


def test_observation_total_trials() -> None:
    observation = RewardObservation(
        source="weekly",
        successes={
            "hot": 3,
            "gap": 2,
        },
        trials={
            "hot": 5,
            "gap": 4,
        },
    )

    assert observation.total_trials == 9


def test_observation_component_order() -> None:
    observation = RewardObservation(
        source="weekly",
        successes={
            "adaptive": 1,
            "hot": 1,
            "gap": 1,
        },
        trials={
            "adaptive": 2,
            "hot": 2,
            "gap": 2,
        },
    )

    assert observation.observed_components == (
        "hot",
        "gap",
        "adaptive",
    )


def test_success_rate() -> None:
    observation = RewardObservation(
        source="weekly",
        successes={"hot": 3},
        trials={"hot": 4},
    )

    assert observation.success_rate(
        "hot"
    ) == pytest.approx(0.75)


def test_unobserved_success_rate_is_rejected() -> None:
    observation = RewardObservation(
        source="weekly",
        successes={"hot": 3},
        trials={"hot": 4},
    )

    with pytest.raises(
        ValueError,
        match="component was not observed",
    ):
        observation.success_rate("cold")


def test_empty_source_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="source must not be empty",
    ):
        RewardObservation(
            source=" ",
            successes={"hot": 1},
            trials={"hot": 2},
        )


def test_unknown_component_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="unknown reward component",
    ):
        RewardObservation(
            source="weekly",
            successes={"unknown": 1},
            trials={"unknown": 2},
        )


def test_zero_trials_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="greater than or equal to 1",
    ):
        RewardObservation(
            source="weekly",
            successes={"hot": 0},
            trials={"hot": 0},
        )


def test_negative_successes_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="greater than or equal to 0",
    ):
        RewardObservation(
            source="weekly",
            successes={"hot": -1},
            trials={"hot": 2},
        )


def test_successes_cannot_exceed_trials() -> None:
    with pytest.raises(
        ValueError,
        match="must not exceed trials",
    ):
        RewardObservation(
            source="weekly",
            successes={"hot": 3},
            trials={"hot": 2},
        )


def test_component_sets_must_match() -> None:
    with pytest.raises(
        ValueError,
        match="missing success counts",
    ):
        RewardObservation(
            source="weekly",
            successes={"hot": 1},
            trials={
                "hot": 2,
                "gap": 2,
            },
        )


@pytest.mark.parametrize(
    "baseline",
    [0.0, 1.0, -0.1, 1.1],
)
def test_invalid_baseline_is_rejected(
    baseline: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="between 0.0 and 1.0 exclusive",
    ):
        RewardObservation(
            source="weekly",
            successes={"hot": 1},
            trials={"hot": 2},
            baseline_rate=baseline,
        )


def test_calculator_returns_signal_frame() -> None:
    observation = RewardObservation(
        source="weekly",
        successes={"hot": 3},
        trials={"hot": 4},
    )

    frame = RewardCalculator().calculate(
        observation
    )

    assert isinstance(frame, SignalFrame)
    assert frame.source == "reward:weekly"


def test_baseline_performance_is_neutral() -> None:
    observation = RewardObservation(
        source="weekly",
        successes={"hot": 5},
        trials={"hot": 10},
        baseline_rate=0.5,
    )

    frame = RewardCalculator().calculate(
        observation
    )

    assert frame.signals["hot"] == pytest.approx(
        0.0
    )


def test_positive_performance_creates_positive_signal() -> None:
    observation = RewardObservation(
        source="weekly",
        successes={"hot": 8},
        trials={"hot": 10},
    )

    frame = RewardCalculator(
        prior_strength=0.0
    ).calculate(observation)

    assert frame.signals["hot"] == pytest.approx(
        0.6
    )


def test_negative_performance_creates_negative_signal() -> None:
    observation = RewardObservation(
        source="weekly",
        successes={"cold": 2},
        trials={"cold": 10},
    )

    frame = RewardCalculator(
        prior_strength=0.0
    ).calculate(observation)

    assert frame.signals["cold"] == pytest.approx(
        -0.6
    )


def test_prior_strength_shrinks_signal() -> None:
    observation = RewardObservation(
        source="weekly",
        successes={"hot": 1},
        trials={"hot": 1},
    )

    unsmoothed = RewardCalculator(
        prior_strength=0.0
    ).calculate(observation)

    smoothed = RewardCalculator(
        prior_strength=4.0
    ).calculate(observation)

    assert unsmoothed.signals["hot"] == pytest.approx(
        1.0
    )
    assert (
        smoothed.signals["hot"]
        < unsmoothed.signals["hot"]
    )


def test_reliability_uses_total_trial_count() -> None:
    observation = RewardObservation(
        source="weekly",
        successes={
            "hot": 3,
            "gap": 2,
        },
        trials={
            "hot": 5,
            "gap": 5,
        },
    )

    frame = RewardCalculator(
        target_sample_size=20
    ).calculate(observation)

    assert frame.reliability == pytest.approx(
        0.5
    )


def test_reliability_is_capped_at_one() -> None:
    observation = RewardObservation(
        source="weekly",
        successes={"hot": 20},
        trials={"hot": 30},
    )

    frame = RewardCalculator(
        target_sample_size=20
    ).calculate(observation)

    assert frame.reliability == pytest.approx(
        1.0
    )


def test_custom_source_prefix() -> None:
    observation = RewardObservation(
        source="weekly",
        successes={"hot": 1},
        trials={"hot": 2},
    )

    frame = RewardCalculator(
        source_prefix="performance"
    ).calculate(observation)

    assert frame.source == "performance:weekly"


def test_reward_frame_integrates_with_aggregator() -> None:
    reward_frame = RewardCalculator(
        prior_strength=0.0,
    ).calculate(
        RewardObservation(
            source="weekly",
            successes={"hot": 8},
            trials={"hot": 10},
        )
    )

    bayesian_frame = SignalFrame(
        source="bayesian",
        signals={"hot": 0.2},
        reliability=1.0,
    )

    result = SignalAggregator().aggregate(
        [
            reward_frame,
            bayesian_frame,
        ]
    )

    assert result.signals["hot"] == pytest.approx(
        1.0 / 3.0
    )


def test_invalid_observation_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="RewardObservation",
    ):
        RewardCalculator().calculate(
            object()  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "prior_strength",
    [-0.1, float("inf")],
)
def test_invalid_prior_strength_is_rejected(
    prior_strength: float,
) -> None:
    with pytest.raises(ValueError):
        RewardCalculator(
            prior_strength=prior_strength
        )


@pytest.mark.parametrize(
    "target_sample_size",
    [0, -1],
)
def test_invalid_target_sample_size_is_rejected(
    target_sample_size: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="greater than or equal to 1",
    ):
        RewardCalculator(
            target_sample_size=target_sample_size
        )


def test_empty_source_prefix_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="source_prefix must not be empty",
    ):
        RewardCalculator(
            source_prefix=" "
        )
