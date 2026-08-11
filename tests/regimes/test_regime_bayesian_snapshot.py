from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lrp.evolution.contracts.bayesian import (
    BayesianPosterior,
)
from lrp.regimes.bayesian_snapshot import (
    RegimeBayesianSnapshot,
)
from lrp.regimes.bayesian_state import (
    RegimeBayesianState,
)


FIXED_TIME = datetime(
    2026,
    8,
    11,
    12,
    30,
    tzinfo=timezone.utc,
)


def make_state() -> RegimeBayesianState:
    return RegimeBayesianState.from_posteriors(
        {
            "gap_recovery": BayesianPosterior(
                alpha=3.0,
                beta=1.0,
            ),
            "cluster_rotation": BayesianPosterior(
                alpha=1.0,
                beta=4.0,
            ),
            "high_band_expansion": BayesianPosterior(
                alpha=5.0,
                beta=2.0,
            ),
            "low_band_expansion": BayesianPosterior(
                alpha=2.0,
                beta=3.0,
            ),
        }
    )


def test_create_builds_valid_snapshot() -> None:
    snapshot = RegimeBayesianSnapshot.create(
        make_state(),
        revision=3,
        sample_size=40,
        saved_at=FIXED_TIME,
    )

    assert snapshot.state == make_state()
    assert snapshot.revision == 3
    assert snapshot.sample_size == 40
    assert snapshot.saved_at == FIXED_TIME
    assert snapshot.schema_version == 1


def test_to_dict_and_from_dict_round_trip() -> None:
    original = RegimeBayesianSnapshot.create(
        make_state(),
        revision=7,
        sample_size=80,
        saved_at=FIXED_TIME,
    )

    restored = RegimeBayesianSnapshot.from_dict(
        original.to_dict()
    )

    assert restored == original


def test_payload_preserves_alpha_beta() -> None:
    payload = RegimeBayesianSnapshot.create(
        make_state(),
        revision=2,
        sample_size=10,
        saved_at=FIXED_TIME,
    ).to_dict()

    posterior = payload["posteriors"][
        "gap_recovery"
    ]

    assert posterior == {
        "alpha": 3.0,
        "beta": 1.0,
    }


def test_payload_does_not_persist_derived_signals() -> None:
    payload = RegimeBayesianSnapshot.create(
        make_state(),
        saved_at=FIXED_TIME,
    ).to_dict()

    for posterior in payload["posteriors"].values():
        assert "mean" not in posterior
        assert "adaptive_signal" not in posterior


@pytest.mark.parametrize(
    "field_name",
    [
        "schema_version",
        "saved_at",
        "revision",
        "sample_size",
        "posteriors",
    ],
)
def test_from_dict_requires_all_fields(
    field_name: str,
) -> None:
    payload = RegimeBayesianSnapshot.create(
        make_state(),
        saved_at=FIXED_TIME,
    ).to_dict()

    payload.pop(field_name)

    with pytest.raises(
        ValueError,
        match="missing regime Bayesian snapshot fields",
    ):
        RegimeBayesianSnapshot.from_dict(payload)


def test_snapshot_requires_timezone_aware_saved_at() -> None:
    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        RegimeBayesianSnapshot(
            state=make_state(),
            saved_at=datetime(2026, 8, 11, 12, 30),
            revision=1,
            sample_size=0,
        )


@pytest.mark.parametrize(
    "revision",
    [0, -1],
)
def test_snapshot_rejects_invalid_revision(
    revision: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="revision must be greater than or equal to 1",
    ):
        RegimeBayesianSnapshot.create(
            make_state(),
            revision=revision,
            saved_at=FIXED_TIME,
        )


@pytest.mark.parametrize(
    "sample_size",
    [-1, -10],
)
def test_snapshot_rejects_negative_sample_size(
    sample_size: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="sample_size must be greater than or equal to 0",
    ):
        RegimeBayesianSnapshot.create(
            make_state(),
            sample_size=sample_size,
            saved_at=FIXED_TIME,
        )


def test_snapshot_rejects_unsupported_schema_version() -> None:
    with pytest.raises(
        ValueError,
        match="unsupported regime Bayesian schema version",
    ):
        RegimeBayesianSnapshot(
            state=make_state(),
            saved_at=FIXED_TIME,
            revision=1,
            sample_size=0,
            schema_version=2,
        )


def test_from_dict_rejects_invalid_saved_at() -> None:
    payload = RegimeBayesianSnapshot.create(
        make_state(),
        saved_at=FIXED_TIME,
    ).to_dict()

    payload["saved_at"] = "not-a-date"

    with pytest.raises(
        ValueError,
        match="valid ISO-8601",
    ):
        RegimeBayesianSnapshot.from_dict(payload)


def test_from_dict_rejects_missing_posterior_field() -> None:
    payload = RegimeBayesianSnapshot.create(
        make_state(),
        saved_at=FIXED_TIME,
    ).to_dict()

    del payload["posteriors"][
        "gap_recovery"
    ]["beta"]

    with pytest.raises(
        ValueError,
        match="alpha and beta",
    ):
        RegimeBayesianSnapshot.from_dict(payload)


def test_from_dict_rejects_missing_regime() -> None:
    payload = RegimeBayesianSnapshot.create(
        make_state(),
        saved_at=FIXED_TIME,
    ).to_dict()

    del payload["posteriors"]["gap_recovery"]

    with pytest.raises(
        ValueError,
        match="missing regime posteriors",
    ):
        RegimeBayesianSnapshot.from_dict(payload)