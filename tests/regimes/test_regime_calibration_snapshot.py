from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lrp.evolution.contracts.regime_calibration import (
    RegimeCalibration,
)
from lrp.regimes.calibration_snapshot import (
    RegimeCalibrationSnapshot,
)


FIXED_TIME = datetime(
    2026,
    8,
    9,
    13,
    30,
    tzinfo=timezone.utc,
)


def make_calibration() -> RegimeCalibration:
    return RegimeCalibration(
        gap_recovery=1.10,
        cluster_rotation=0.90,
        high_band_expansion=1.20,
        low_band_expansion=0.80,
    )


def test_create_builds_valid_snapshot() -> None:
    snapshot = RegimeCalibrationSnapshot.create(
        make_calibration(),
        revision=3,
        sample_size=40,
        saved_at=FIXED_TIME,
    )

    assert snapshot.calibration == make_calibration()
    assert snapshot.revision == 3
    assert snapshot.sample_size == 40
    assert snapshot.saved_at == FIXED_TIME
    assert snapshot.schema_version == 1


def test_to_dict_and_from_dict_round_trip() -> None:
    original = RegimeCalibrationSnapshot.create(
        make_calibration(),
        revision=7,
        sample_size=80,
        saved_at=FIXED_TIME,
    )

    restored = RegimeCalibrationSnapshot.from_dict(
        original.to_dict()
    )

    assert restored == original


def test_serialized_payload_contains_expected_fields() -> None:
    payload = RegimeCalibrationSnapshot.create(
        make_calibration(),
        revision=2,
        sample_size=10,
        saved_at=FIXED_TIME,
    ).to_dict()

    assert payload["schema_version"] == 1
    assert payload["revision"] == 2
    assert payload["sample_size"] == 10
    assert payload["saved_at"] == FIXED_TIME.isoformat()
    assert payload["calibration"] == make_calibration().as_dict()


@pytest.mark.parametrize(
    "field_name",
    [
        "schema_version",
        "saved_at",
        "revision",
        "sample_size",
        "calibration",
    ],
)
def test_from_dict_requires_all_fields(
    field_name: str,
) -> None:
    payload = RegimeCalibrationSnapshot.create(
        make_calibration(),
        saved_at=FIXED_TIME,
    ).to_dict()

    payload.pop(field_name)

    with pytest.raises(
        ValueError,
        match="missing regime calibration snapshot fields",
    ):
        RegimeCalibrationSnapshot.from_dict(payload)


def test_snapshot_requires_timezone_aware_saved_at() -> None:
    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        RegimeCalibrationSnapshot(
            calibration=make_calibration(),
            saved_at=datetime(2026, 8, 9, 13, 30),
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
        RegimeCalibrationSnapshot.create(
            make_calibration(),
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
        RegimeCalibrationSnapshot.create(
            make_calibration(),
            sample_size=sample_size,
            saved_at=FIXED_TIME,
        )


def test_snapshot_rejects_unsupported_schema_version() -> None:
    with pytest.raises(
        ValueError,
        match="unsupported regime calibration schema version",
    ):
        RegimeCalibrationSnapshot(
            calibration=make_calibration(),
            saved_at=FIXED_TIME,
            revision=1,
            sample_size=0,
            schema_version=2,
        )


def test_from_dict_rejects_invalid_saved_at() -> None:
    payload = RegimeCalibrationSnapshot.create(
        make_calibration(),
        saved_at=FIXED_TIME,
    ).to_dict()
    payload["saved_at"] = "not-a-date"

    with pytest.raises(
        ValueError,
        match="valid ISO-8601",
    ):
        RegimeCalibrationSnapshot.from_dict(payload)
