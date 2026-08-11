from __future__ import annotations

import pytest

from lrp.evolution.contracts.bayesian import (
    BayesianEvidence,
)
from lrp.regimes.bayesian_evidence import (
    RegimeBayesianEvidenceAdapter,
)
from lrp.regimes.reward import RegimeReward


def make_reward(
    reward: float,
    *,
    confidence: float = 1.0,
    sample_weight: float = 1.0,
) -> RegimeReward:
    return RegimeReward(
        regime="gap_recovery",
        reward=reward,
        confidence=confidence,
        sample_weight=sample_weight,
    )


def test_positive_reward_becomes_success_evidence() -> None:
    adapter = RegimeBayesianEvidenceAdapter(
        evidence_scale=10,
    )

    evidence = adapter.convert(
        make_reward(0.8)
    )

    assert evidence == BayesianEvidence(
        successes=8,
        failures=0,
    )


def test_negative_reward_becomes_failure_evidence() -> None:
    adapter = RegimeBayesianEvidenceAdapter(
        evidence_scale=10,
    )

    evidence = adapter.convert(
        make_reward(-0.6)
    )

    assert evidence == BayesianEvidence(
        successes=0,
        failures=6,
    )


def test_zero_reward_produces_zero_evidence() -> None:
    adapter = RegimeBayesianEvidenceAdapter(
        evidence_scale=10,
    )

    evidence = adapter.convert(
        make_reward(0.0)
    )

    assert evidence == BayesianEvidence(
        successes=0,
        failures=0,
    )


def test_effective_reward_controls_evidence_strength() -> None:
    adapter = RegimeBayesianEvidenceAdapter(
        evidence_scale=10,
    )

    evidence = adapter.convert(
        make_reward(
            1.0,
            confidence=0.5,
            sample_weight=0.4,
        )
    )

    assert evidence == BayesianEvidence(
        successes=2,
        failures=0,
    )


def test_small_nonzero_reward_preserves_evidence() -> None:
    adapter = RegimeBayesianEvidenceAdapter(
        evidence_scale=10,
    )

    evidence = adapter.convert(
        make_reward(0.01)
    )

    assert evidence == BayesianEvidence(
        successes=1,
        failures=0,
    )


def test_default_scale_is_ten() -> None:
    adapter = RegimeBayesianEvidenceAdapter()

    evidence = adapter.convert(
        make_reward(0.5)
    )

    assert evidence.successes == 5
    assert evidence.failures == 0


@pytest.mark.parametrize(
    "value",
    [
        True,
        0,
        -1,
        1.5,
        "10",
        None,
    ],
)
def test_invalid_evidence_scale_is_rejected(
    value: object,
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
        match="evidence_scale",
    ):
        RegimeBayesianEvidenceAdapter(
            evidence_scale=value,  # type: ignore[arg-type]
        )


def test_invalid_reward_is_rejected() -> None:
    adapter = RegimeBayesianEvidenceAdapter()

    with pytest.raises(
        TypeError,
        match="RegimeReward",
    ):
        adapter.convert(
            object()  # type: ignore[arg-type]
        )
