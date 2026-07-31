from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lrp.evolution import AdaptiveWeightProfile


FIXED_TIME = datetime(
    2026,
    7,
    31,
    0,
    0,
    0,
    tzinfo=timezone.utc,
)


def make_profile(
    **overrides: object,
) -> AdaptiveWeightProfile:
    values: dict[str, object] = {
        "hot_weight": 0.35,
        "cold_weight": 0.15,
        "gap_weight": 0.15,
        "trend_weight": 0.15,
        "transition_weight": 0.10,
        "learning_weight": 0.05,
        "adaptive_weight": 0.05,
        "confidence": 0.75,
        "sample_size": 24,
        "revision": 3,
        "generated_at": FIXED_TIME,
    }
    values.update(overrides)

    return AdaptiveWeightProfile(**values)  # type: ignore[arg-type]


def test_default_profile_matches_probability_defaults() -> None:
    profile = AdaptiveWeightProfile.default(
        revision=1,
        generated_at=FIXED_TIME,
    )

    assert profile.to_probability_weights() == {
        "hot": 0.35,
        "cold": 0.15,
        "gap": 0.15,
        "trend": 0.15,
        "transition": 0.10,
        "learning": 0.05,
        "adaptive": 0.05,
    }
    assert profile.total_weight == pytest.approx(1.0)
    assert profile.confidence == 0.0
    assert profile.sample_size == 0
    assert profile.revision == 1
    assert profile.generated_at == FIXED_TIME


def test_profile_is_immutable() -> None:
    profile = make_profile()

    with pytest.raises(AttributeError):
        profile.hot_weight = 0.30  # type: ignore[misc]


def test_serialization_round_trip() -> None:
    profile = make_profile()

    payload = profile.to_dict()
    restored = AdaptiveWeightProfile.from_dict(payload)

    assert restored == profile
    assert restored.to_dict() == payload


def test_serialized_weight_order_is_stable() -> None:
    profile = make_profile()

    assert tuple(profile.to_dict()["weights"]) == (
        "hot_weight",
        "cold_weight",
        "gap_weight",
        "trend_weight",
        "transition_weight",
        "learning_weight",
        "adaptive_weight",
    )


@pytest.mark.parametrize(
    "field_name",
    AdaptiveWeightProfile.WEIGHT_FIELDS,
)
def test_negative_weight_is_rejected(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="greater than or equal to 0",
    ):
        make_profile(**{field_name: -0.01})


def test_weight_sum_below_one_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must sum to 1.0",
    ):
        make_profile(hot_weight=0.34)


def test_weight_sum_above_one_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must sum to 1.0",
    ):
        make_profile(hot_weight=0.36)


@pytest.mark.parametrize(
    "confidence",
    [-0.01, 1.01],
)
def test_invalid_confidence_is_rejected(
    confidence: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="between 0.0 and 1.0",
    ):
        make_profile(confidence=confidence)


def test_negative_sample_size_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="greater than or equal to 0",
    ):
        make_profile(sample_size=-1)


@pytest.mark.parametrize(
    "revision",
    [0, -1],
)
def test_invalid_revision_is_rejected(
    revision: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="greater than or equal to 1",
    ):
        make_profile(revision=revision)


def test_naive_generated_at_is_rejected() -> None:
    naive_time = datetime(2026, 7, 31, 0, 0, 0)

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        make_profile(generated_at=naive_time)


def test_missing_serialized_weight_is_rejected() -> None:
    payload = make_profile().to_dict()
    del payload["weights"]["gap_weight"]

    with pytest.raises(
        ValueError,
        match="missing adaptive weight fields",
    ):
        AdaptiveWeightProfile.from_dict(payload)


def test_missing_serialized_metadata_is_rejected() -> None:
    payload = make_profile().to_dict()
    del payload["confidence"]

    with pytest.raises(
        ValueError,
        match="missing adaptive metadata fields",
    ):
        AdaptiveWeightProfile.from_dict(payload)