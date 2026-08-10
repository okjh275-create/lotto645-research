from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lrp.evolution.contracts.regime_calibration import (
    RegimeCalibration,
)
from lrp.regimes.calibration_repository import (
    RegimeCalibrationRepository,
)
from lrp.regimes.integration.calibration_provider import (
    RepositoryRegimeCalibrationProvider,
    StaticRegimeCalibrationProvider,
)


def test_static_provider_returns_calibration() -> None:
    calibration = RegimeCalibration.neutral()

    provider = StaticRegimeCalibrationProvider(
        calibration
    )

    assert (
        provider.get_calibration(round_no=1221)
        is calibration
    )


def test_static_provider_can_return_none() -> None:
    provider = StaticRegimeCalibrationProvider(None)

    assert (
        provider.get_calibration(round_no=1221)
        is None
    )


@pytest.mark.parametrize(
    "round_no",
    [
        True,
        1.5,
        "1221",
    ],
)
def test_static_provider_rejects_invalid_round_type(
    round_no: object,
) -> None:
    provider = StaticRegimeCalibrationProvider(None)

    with pytest.raises(
        TypeError,
        match="round_no",
    ):
        provider.get_calibration(
            round_no=round_no,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "round_no",
    [
        0,
        -1,
    ],
)
def test_static_provider_rejects_invalid_round_value(
    round_no: int,
) -> None:
    provider = StaticRegimeCalibrationProvider(None)

    with pytest.raises(
        ValueError,
        match="round_no",
    ):
        provider.get_calibration(
            round_no=round_no
        )


def test_repository_provider_returns_none_without_snapshot(
    tmp_path,
) -> None:
    repository = RegimeCalibrationRepository(
        tmp_path / "regime-calibration"
    )
    provider = RepositoryRegimeCalibrationProvider(
        repository
    )

    assert (
        provider.get_calibration(round_no=1221)
        is None
    )


def test_repository_provider_loads_latest_calibration(
    tmp_path,
) -> None:
    repository = RegimeCalibrationRepository(
        tmp_path / "regime-calibration"
    )

    first = RegimeCalibration.neutral()

    second = RegimeCalibration(
        gap_recovery=1.10,
        cluster_rotation=0.95,
        high_band_expansion=1.05,
        low_band_expansion=0.90,
    )

    repository.save(
        first,
        revision=1,
        sample_size=10,
        saved_at=datetime(
            2026,
            8,
            10,
            tzinfo=timezone.utc,
        ),
    )
    repository.save(
        second,
        revision=2,
        sample_size=20,
        saved_at=datetime(
            2026,
            8,
            11,
            tzinfo=timezone.utc,
        ),
    )

    provider = RepositoryRegimeCalibrationProvider(
        repository
    )

    assert (
        provider.get_calibration(round_no=1221)
        == second
    )


def test_repository_provider_rejects_invalid_repository() -> None:
    with pytest.raises(
        TypeError,
        match="repository",
    ):
        RepositoryRegimeCalibrationProvider(
            object()  # type: ignore[arg-type]
        )