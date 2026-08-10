from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from lrp.evolution.contracts.regime_calibration import (
    RegimeCalibration,
)
from lrp.regimes.calibration_repository import (
    RegimeCalibrationNotFoundError,
    RegimeCalibrationRepository,
)


FIXED_TIME = datetime(
    2026, 8, 9, 14, 0, tzinfo=timezone.utc
)


def make_calibration(
    value: float = 1.0,
) -> RegimeCalibration:
    return RegimeCalibration(
        gap_recovery=value,
        cluster_rotation=1.0,
        high_band_expansion=1.0,
        low_band_expansion=1.0,
    )


def test_repository_saves_and_loads_revision(
    tmp_path: Path,
) -> None:
    repository = RegimeCalibrationRepository(tmp_path)

    saved = repository.save(
        make_calibration(1.10),
        revision=1,
        sample_size=20,
        saved_at=FIXED_TIME,
    )

    loaded = repository.load_revision(1)

    assert loaded == saved
    assert repository.exists(1) is True
    assert repository.revisions() == (1,)


def test_repository_loads_latest_revision(
    tmp_path: Path,
) -> None:
    repository = RegimeCalibrationRepository(tmp_path)

    repository.save(
        make_calibration(0.90),
        revision=1,
        saved_at=FIXED_TIME,
    )
    repository.save(
        make_calibration(1.20),
        revision=2,
        saved_at=FIXED_TIME,
    )

    latest = repository.load_latest()

    assert latest.revision == 2
    assert latest.calibration.gap_recovery == 1.20


def test_repository_history_is_revision_ordered(
    tmp_path: Path,
) -> None:
    repository = RegimeCalibrationRepository(tmp_path)

    repository.save(
        make_calibration(1.00),
        revision=1,
        saved_at=FIXED_TIME,
    )
    repository.save(
        make_calibration(1.10),
        revision=2,
        saved_at=FIXED_TIME,
    )
    repository.save(
        make_calibration(1.20),
        revision=3,
        saved_at=FIXED_TIME,
    )

    history = repository.history()

    assert tuple(
        snapshot.revision
        for snapshot in history
    ) == (1, 2, 3)


def test_repository_rejects_duplicate_revision(
    tmp_path: Path,
) -> None:
    repository = RegimeCalibrationRepository(tmp_path)

    repository.save(
        make_calibration(),
        revision=1,
        saved_at=FIXED_TIME,
    )

    with pytest.raises(FileExistsError):
        repository.save(
            make_calibration(1.10),
            revision=1,
            saved_at=FIXED_TIME,
        )


def test_repository_missing_revision_raises(
    tmp_path: Path,
) -> None:
    repository = RegimeCalibrationRepository(tmp_path)

    with pytest.raises(
        RegimeCalibrationNotFoundError
    ):
        repository.load_revision(1)


def test_repository_missing_latest_raises(
    tmp_path: Path,
) -> None:
    repository = RegimeCalibrationRepository(tmp_path)

    with pytest.raises(
        RegimeCalibrationNotFoundError
    ):
        repository.load_latest()


def test_load_latest_skips_corrupt_newest(
    tmp_path: Path,
) -> None:
    repository = RegimeCalibrationRepository(tmp_path)

    repository.save(
        make_calibration(1.10),
        revision=1,
        saved_at=FIXED_TIME,
    )

    corrupt = tmp_path / "revision-00000002.json"
    corrupt.write_text(
        "{not-json",
        encoding="utf-8",
    )

    latest = repository.load_latest()

    assert latest.revision == 1


def test_history_skips_corrupt_snapshot(
    tmp_path: Path,
) -> None:
    repository = RegimeCalibrationRepository(tmp_path)

    repository.save(
        make_calibration(1.00),
        revision=1,
        saved_at=FIXED_TIME,
    )

    corrupt = tmp_path / "revision-00000002.json"
    corrupt.write_text(
        "{broken",
        encoding="utf-8",
    )

    repository.save(
        make_calibration(1.20),
        revision=3,
        saved_at=FIXED_TIME,
    )

    history = repository.history()

    assert tuple(
        snapshot.revision
        for snapshot in history
    ) == (1, 3)
