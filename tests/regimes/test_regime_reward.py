from __future__ import annotations

import math

import pytest

from lrp.contracts import ContractError
from lrp.regimes.reward import RegimeReward


def test_regime_reward_builds_effective_reward() -> None:
    reward = RegimeReward(
        regime="gap_recovery",
        reward=0.5,
        confidence=0.8,
        sample_weight=0.6,
    )

    assert math.isclose(
        reward.effective_reward,
        0.24,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


@pytest.mark.parametrize(
    "value",
    [-1.0, 0.0, 1.0],
)
def test_reward_accepts_boundary_values(
    value: float,
) -> None:
    result = RegimeReward(
        regime="gap_recovery",
        reward=value,
        confidence=1.0,
        sample_weight=1.0,
    )

    assert result.reward == value


@pytest.mark.parametrize(
    "value",
    [-1.01, 1.01],
)
def test_reward_rejects_out_of_range_values(
    value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="reward must be between -1.0 and 1.0",
    ):
        RegimeReward(
            regime="gap_recovery",
            reward=value,
            confidence=1.0,
            sample_weight=1.0,
        )


@pytest.mark.parametrize(
    "value",
    [-0.1, 1.1],
)
def test_confidence_rejects_out_of_range_values(
    value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="confidence must be between 0.0 and 1.0",
    ):
        RegimeReward(
            regime="gap_recovery",
            reward=0.0,
            confidence=value,
            sample_weight=1.0,
        )


@pytest.mark.parametrize(
    "value",
    [-0.01, 1.2],
)
def test_sample_weight_rejects_out_of_range_values(
    value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="sample_weight must be between 0.0 and 1.0",
    ):
        RegimeReward(
            regime="gap_recovery",
            reward=0.0,
            confidence=1.0,
            sample_weight=value,
        )


def test_unknown_regime_is_rejected() -> None:
    with pytest.raises(
        ContractError,
        match="unsupported regime",
    ):
        RegimeReward(
            regime="unknown",
            reward=0.0,
            confidence=1.0,
            sample_weight=1.0,
        )


@pytest.mark.parametrize(
    "field_name",
    ["reward", "confidence", "sample_weight"],
)
@pytest.mark.parametrize(
    "value",
    [True, None, "0.5"],
)
def test_numeric_fields_reject_invalid_types(
    field_name: str,
    value: object,
) -> None:
    kwargs = {
        "regime": "gap_recovery",
        "reward": 0.0,
        "confidence": 1.0,
        "sample_weight": 1.0,
    }
    kwargs[field_name] = value

    with pytest.raises(
        TypeError,
        match=f"{field_name} must be numeric",
    ):
        RegimeReward(**kwargs)


def test_as_dict_contains_effective_reward() -> None:
    reward = RegimeReward(
        regime="cluster_rotation",
        reward=-0.4,
        confidence=0.5,
        sample_weight=0.25,
    )

    payload = reward.as_dict()

    assert payload["regime"] == "cluster_rotation"
    assert payload["reward"] == -0.4
    assert payload["confidence"] == 0.5
    assert payload["sample_weight"] == 0.25
    assert math.isclose(
        payload["effective_reward"],
        -0.05,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
