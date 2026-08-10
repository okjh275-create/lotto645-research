from __future__ import annotations

import pytest

from lrp.evolution.contracts.regime_calibration import (
    RegimeCalibration,
)


def test_neutral_calibration_defaults_to_one() -> None:
    calibration = RegimeCalibration.neutral()

    assert calibration.gap_recovery == 1.0
    assert calibration.cluster_rotation == 1.0
    assert calibration.high_band_expansion == 1.0
    assert calibration.low_band_expansion == 1.0


def test_as_dict_and_from_dict_round_trip() -> None:
    original = RegimeCalibration(
        gap_recovery=1.10,
        cluster_rotation=0.90,
        high_band_expansion=1.20,
        low_band_expansion=0.80,
    )

    restored = RegimeCalibration.from_dict(
        original.as_dict()
    )

    assert restored == original


def test_from_dict_uses_neutral_defaults() -> None:
    calibration = RegimeCalibration.from_dict({})

    assert calibration == RegimeCalibration.neutral()


@pytest.mark.parametrize(
    "field_name",
    [
        "gap_recovery",
        "cluster_rotation",
        "high_band_expansion",
        "low_band_expansion",
    ],
)
@pytest.mark.parametrize(
    "value",
    [0.49, 1.51],
)
def test_calibration_rejects_out_of_range_values(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="between 0.50 and 1.50",
    ):
        RegimeCalibration(
            **{field_name: value}
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "gap_recovery",
        "cluster_rotation",
        "high_band_expansion",
        "low_band_expansion",
    ],
)
@pytest.mark.parametrize(
    "value",
    [True, "1.0", None],
)
def test_calibration_rejects_invalid_types(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be numeric",
    ):
        RegimeCalibration(
            **{field_name: value}
        )


def test_get_returns_regime_value() -> None:
    calibration = RegimeCalibration(
        gap_recovery=1.25,
        cluster_rotation=0.85,
        high_band_expansion=1.10,
        low_band_expansion=0.95,
    )

    assert calibration.get("gap_recovery") == 1.25
    assert calibration.get("cluster_rotation") == 0.85
    assert calibration.get("high_band_expansion") == 1.10
    assert calibration.get("low_band_expansion") == 0.95


def test_get_returns_neutral_for_unhandled_regime() -> None:
    calibration = RegimeCalibration()

    assert calibration.get("neutral") == 1.0
    assert calibration.get("mixed") == 1.0
    assert calibration.get("unknown") == 1.0
