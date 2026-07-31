from __future__ import annotations

from datetime import datetime, timezone
from math import isclose

import pytest

from lrp.evolution import (
    AdaptivePolicyConfig,
    AdaptivePolicyDecision,
    AdaptiveWeightPolicy,
    AdaptiveWeightProfile,
)


FIXED_TIME = datetime(
    2026,
    7,
    31,
    11,
    0,
    0,
    tzinfo=timezone.utc,
)


def make_profile(
    *,
    confidence: float = 0.80,
    sample_size: int = 40,
    revision: int = 2,
    hot_weight: float = 0.35,
    cold_weight: float = 0.15,
    gap_weight: float = 0.15,
    trend_weight: float = 0.15,
    transition_weight: float = 0.10,
    learning_weight: float = 0.05,
    adaptive_weight: float = 0.05,
) -> AdaptiveWeightProfile:
    return AdaptiveWeightProfile(
        hot_weight=hot_weight,
        cold_weight=cold_weight,
        gap_weight=gap_weight,
        trend_weight=trend_weight,
        transition_weight=transition_weight,
        learning_weight=learning_weight,
        adaptive_weight=adaptive_weight,
        confidence=confidence,
        sample_size=sample_size,
        revision=revision,
        generated_at=FIXED_TIME,
    )


def test_default_config_values() -> None:
    config = AdaptivePolicyConfig()

    assert config.min_confidence == 0.60
    assert config.min_sample_size == 20
    assert config.max_component_delta == 0.08
    assert config.fail_open is False


def test_valid_candidate_is_applied() -> None:
    candidate = make_profile()
    policy = AdaptiveWeightPolicy()

    decision = policy.evaluate(candidate)

    assert isinstance(
        decision,
        AdaptivePolicyDecision,
    )
    assert decision.applied is True
    assert decision.rejected is False
    assert decision.profile == candidate
    assert decision.reasons == ()
    assert decision.clamped_components == ()


def test_low_confidence_is_rejected() -> None:
    candidate = make_profile(confidence=0.59)
    policy = AdaptiveWeightPolicy()

    decision = policy.evaluate(candidate)

    assert decision.rejected is True
    assert "confidence_below_threshold" in (
        decision.reasons
    )


def test_small_sample_is_rejected() -> None:
    candidate = make_profile(sample_size=19)
    policy = AdaptiveWeightPolicy()

    decision = policy.evaluate(candidate)

    assert decision.rejected is True
    assert "sample_size_below_threshold" in (
        decision.reasons
    )


def test_multiple_rejection_reasons_are_recorded() -> None:
    candidate = make_profile(
        confidence=0.20,
        sample_size=2,
    )
    policy = AdaptiveWeightPolicy()

    decision = policy.evaluate(candidate)

    assert decision.reasons == (
        "confidence_below_threshold",
        "sample_size_below_threshold",
    )


def test_rejection_uses_previous_profile_as_fallback() -> None:
    previous = make_profile(
        revision=3,
        confidence=0.90,
        sample_size=80,
    )
    candidate = make_profile(
        revision=4,
        confidence=0.20,
    )

    decision = AdaptiveWeightPolicy().evaluate(
        candidate,
        previous=previous,
    )

    assert decision.rejected is True
    assert decision.profile == previous


def test_rejection_without_previous_uses_default() -> None:
    candidate = make_profile(
        confidence=0.10,
        revision=5,
    )

    decision = AdaptiveWeightPolicy().evaluate(
        candidate
    )

    assert decision.rejected is True
    assert decision.profile.to_probability_weights() == (
        AdaptiveWeightProfile.default(
            generated_at=FIXED_TIME,
        ).to_probability_weights()
    )


def test_revision_must_be_newer_than_previous() -> None:
    previous = make_profile(revision=4)
    candidate = make_profile(revision=4)

    decision = AdaptiveWeightPolicy().evaluate(
        candidate,
        previous=previous,
    )

    assert decision.rejected is True
    assert "revision_not_newer" in decision.reasons


def test_fail_open_applies_low_confidence_candidate() -> None:
    config = AdaptivePolicyConfig(fail_open=True)
    policy = AdaptiveWeightPolicy(config)

    candidate = make_profile(
        confidence=0.10,
        sample_size=1,
    )

    decision = policy.evaluate(candidate)

    assert decision.applied is True
    assert decision.profile == candidate
    assert "fail_open_applied" in decision.reasons


def test_large_component_change_is_clamped() -> None:
    previous = make_profile(
        revision=2,
        hot_weight=0.35,
        cold_weight=0.15,
        gap_weight=0.15,
        trend_weight=0.15,
        transition_weight=0.10,
        learning_weight=0.05,
        adaptive_weight=0.05,
    )

    candidate = make_profile(
        revision=3,
        hot_weight=0.55,
        cold_weight=0.10,
        gap_weight=0.10,
        trend_weight=0.10,
        transition_weight=0.05,
        learning_weight=0.05,
        adaptive_weight=0.05,
    )

    policy = AdaptiveWeightPolicy(
        AdaptivePolicyConfig(
            max_component_delta=0.08,
        )
    )

    decision = policy.evaluate(
        candidate,
        previous=previous,
    )

    assert decision.applied is True
    assert decision.was_clamped is True
    assert "hot" in decision.clamped_components
    assert "component_delta_clamped" in (
        decision.reasons
    )
    assert decision.profile.hot_weight < (
        candidate.hot_weight
    )


def test_small_component_changes_are_not_clamped() -> None:
    previous = make_profile(revision=2)

    candidate = make_profile(
        revision=3,
        hot_weight=0.37,
        cold_weight=0.14,
        gap_weight=0.14,
        trend_weight=0.15,
        transition_weight=0.10,
        learning_weight=0.05,
        adaptive_weight=0.05,
    )

    decision = AdaptiveWeightPolicy().evaluate(
        candidate,
        previous=previous,
    )

    assert decision.applied is True
    assert decision.profile == candidate
    assert decision.was_clamped is False


def test_clamped_profile_remains_normalized() -> None:
    previous = make_profile(revision=2)

    candidate = make_profile(
        revision=3,
        hot_weight=0.60,
        cold_weight=0.08,
        gap_weight=0.08,
        trend_weight=0.08,
        transition_weight=0.06,
        learning_weight=0.05,
        adaptive_weight=0.05,
    )

    decision = AdaptiveWeightPolicy().evaluate(
        candidate,
        previous=previous,
    )

    assert isclose(
        decision.profile.total_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    )


def test_candidate_type_is_validated() -> None:
    policy = AdaptiveWeightPolicy()

    with pytest.raises(
        TypeError,
        match="candidate must be",
    ):
        policy.evaluate(  # type: ignore[arg-type]
            {"hot": 0.35}
        )


@pytest.mark.parametrize(
    "confidence",
    [-0.01, 1.01],
)
def test_invalid_min_confidence_is_rejected(
    confidence: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="between 0.0 and 1.0",
    ):
        AdaptivePolicyConfig(
            min_confidence=confidence
        )


def test_negative_min_sample_size_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="greater than or equal to 0",
    ):
        AdaptivePolicyConfig(
            min_sample_size=-1
        )


def test_invalid_max_component_delta_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="between",
    ):
        AdaptivePolicyConfig(
            max_component_delta=1.0
        )


def test_public_api_exports_policy_types() -> None:
    config = AdaptivePolicyConfig()
    policy = AdaptiveWeightPolicy(config)

    assert policy.config is config