from __future__ import annotations

import math

import pytest

from lrp.regimes.learning_rate import (
    AdaptiveLearningRatePolicy,
)


def test_fresh_calibration_uses_base_rate() -> None:
    policy = AdaptiveLearningRatePolicy()

    assert policy.rate(
        revision=0,
        sample_size=0,
    ) == 0.10


def test_revision_maturity_reduces_rate() -> None:
    policy = AdaptiveLearningRatePolicy()

    fresh = policy.rate(
        revision=1,
        sample_size=0,
    )
    mature = policy.rate(
        revision=25,
        sample_size=0,
    )

    assert mature < fresh


def test_sample_maturity_reduces_rate() -> None:
    policy = AdaptiveLearningRatePolicy()

    fresh = policy.rate(
        revision=1,
        sample_size=0,
    )
    mature = policy.rate(
        revision=1,
        sample_size=80,
    )

    assert mature < fresh


def test_rate_never_exceeds_base_rate() -> None:
    policy = AdaptiveLearningRatePolicy(
        base_rate=0.15,
        min_rate=0.02,
    )

    assert policy.rate(
        revision=0,
        sample_size=0,
    ) == 0.15


def test_rate_is_clamped_to_minimum() -> None:
    policy = AdaptiveLearningRatePolicy(
        base_rate=0.10,
        min_rate=0.02,
    )

    rate = policy.rate(
        revision=1_000_000,
        sample_size=1_000_000,
    )

    assert rate == 0.02


def test_custom_sample_scale_changes_decay() -> None:
    fast_decay = AdaptiveLearningRatePolicy(
        sample_scale=10
    )
    slow_decay = AdaptiveLearningRatePolicy(
        sample_scale=100
    )

    fast_rate = fast_decay.rate(
        revision=1,
        sample_size=100,
    )
    slow_rate = slow_decay.rate(
        revision=1,
        sample_size=100,
    )

    assert fast_rate < slow_rate


def test_expected_maturity_rate() -> None:
    policy = AdaptiveLearningRatePolicy(
        base_rate=0.10,
        min_rate=0.02,
        sample_scale=20,
    )

    rate = policy.rate(
        revision=4,
        sample_size=60,
    )

    assert math.isclose(
        rate,
        0.05,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_rate": 0.0},
        {"base_rate": -0.1},
        {"base_rate": 1.1},
        {"min_rate": 0.0},
        {"min_rate": -0.1},
        {"min_rate": 1.1},
        {
            "base_rate": 0.10,
            "min_rate": 0.20,
        },
    ],
)
def test_invalid_rate_configuration_is_rejected(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        AdaptiveLearningRatePolicy(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_rate": True},
        {"base_rate": "0.1"},
        {"min_rate": True},
        {"min_rate": "0.02"},
    ],
)
def test_invalid_rate_type_is_rejected(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(TypeError):
        AdaptiveLearningRatePolicy(**kwargs)


@pytest.mark.parametrize(
    "sample_scale",
    [0, -1],
)
def test_invalid_sample_scale_is_rejected(
    sample_scale: int,
) -> None:
    with pytest.raises(ValueError):
        AdaptiveLearningRatePolicy(
            sample_scale=sample_scale
        )


@pytest.mark.parametrize(
    "sample_scale",
    [True, 10.5, "20"],
)
def test_invalid_sample_scale_type_is_rejected(
    sample_scale: object,
) -> None:
    with pytest.raises(TypeError):
        AdaptiveLearningRatePolicy(
            sample_scale=sample_scale
        )


@pytest.mark.parametrize(
    ("revision", "sample_size"),
    [
        (-1, 0),
        (0, -1),
    ],
)
def test_negative_maturity_is_rejected(
    revision: int,
    sample_size: int,
) -> None:
    policy = AdaptiveLearningRatePolicy()

    with pytest.raises(ValueError):
        policy.rate(
            revision=revision,
            sample_size=sample_size,
        )


@pytest.mark.parametrize(
    ("revision", "sample_size"),
    [
        (True, 0),
        ("1", 0),
        (0, True),
        (0, "1"),
    ],
)
def test_invalid_maturity_type_is_rejected(
    revision: object,
    sample_size: object,
) -> None:
    policy = AdaptiveLearningRatePolicy()

    with pytest.raises(TypeError):
        policy.rate(
            revision=revision,
            sample_size=sample_size,
        )