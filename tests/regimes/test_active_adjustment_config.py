from __future__ import annotations

import pytest

from lrp.regimes.integration.active_adjustment import (
    RegimeAdjustmentConfig,
)


def test_default_config_is_valid() -> None:
    config = RegimeAdjustmentConfig()

    assert config.gap_recovery_max_boost == 0.02
    assert config.cluster_rotation_max_boost == 0.01
    assert config.high_band_max_boost == 0.02
    assert config.low_band_max_boost == 0.02


def test_numeric_values_are_normalized_to_float() -> None:
    config = RegimeAdjustmentConfig(
        gap_recovery_max_boost=0,
        cluster_rotation_max_boost=1 / 100,
        high_band_max_boost=2 / 100,
        low_band_max_boost=3 / 100,
    )

    assert isinstance(
        config.gap_recovery_max_boost,
        float,
    )
    assert isinstance(
        config.cluster_rotation_max_boost,
        float,
    )
    assert isinstance(
        config.high_band_max_boost,
        float,
    )
    assert isinstance(
        config.low_band_max_boost,
        float,
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "gap_recovery_max_boost",
        "cluster_rotation_max_boost",
        "high_band_max_boost",
        "low_band_max_boost",
    ],
)
@pytest.mark.parametrize(
    "value",
    [-0.01, 0.11],
)
def test_config_rejects_out_of_range_values(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="between 0 and 0.10",
    ):
        RegimeAdjustmentConfig(
            **{field_name: value}
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "gap_recovery_max_boost",
        "cluster_rotation_max_boost",
        "high_band_max_boost",
        "low_band_max_boost",
    ],
)
@pytest.mark.parametrize(
    "value",
    [True, "0.02"],
)
def test_config_rejects_non_numeric_values(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be numeric",
    ):
        RegimeAdjustmentConfig(
            **{field_name: value}
        )
