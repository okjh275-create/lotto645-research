from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from lrp.evolution.contracts.bayesian import (
    BayesianPosterior,
)
from lrp.regimes.bayesian_repository import (
    RegimeBayesianNotFoundError,
    RegimeBayesianRepository,
)
from lrp.regimes.bayesian_serializer import (
    RegimeBayesianSerializationError,
)
from lrp.regimes.bayesian_state import (
    RegimeBayesianState,
)


FIXED_TIME = datetime(
    2026,
    8,
    11,
    13,
    0,
    tzinfo=timezone.utc,
)


def make_state(
    alpha: float = 1.0,
) -> RegimeBayesianState:
    return RegimeBayesianState.from_posteriors(
        {
            "gap_recovery": BayesianPosterior(
                alpha=alpha,
                beta=1.0,
            ),
            "cluster_rotation": BayesianPosterior(
                alpha=1.0,
                beta=1.0,
            ),
            "high_band_expansion": BayesianPosterior(
                alpha=1.0,
                beta=1.0,
            ),
            "low_band_expansion": BayesianPosterior(
                alpha=1.0,
                beta=1.0,
            ),
        }
    )


def test_repository_saves_and_loads_revision(
    tmp_path: Path,
) -> None:
    repository = RegimeBayesianRepository(
        tmp_path
    )

    saved = repository.save(
        make_state(3.0),
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
    repository = RegimeBayesianRepository(
        tmp_path
    )

    repository.save(
        make_state(2.0),
        revision=1,
        saved_at=FIXED_TIME,
    )
    repository.save(
        make_state(4.0),
        revision=2,
        saved_at=FIXED_TIME,
    )

    latest = repository.load_latest()

    assert latest.revision == 2
    assert (
        latest.state
        .posteriors["gap_recovery"]
        .alpha
        == 4.0
    )


def test_repository_history_is_revision_ordered(
    tmp_path: Path,
) -> None:
    repository = RegimeBayesianRepository(
        tmp_path
    )

    for revision in (1, 2, 3):
        repository.save(
            make_state(float(revision)),
            revision=revision,
            saved_at=FIXED_TIME,
        )

    assert tuple(
        snapshot.revision
        for snapshot in repository.history()
    ) == (1, 2, 3)


def test_repository_rejects_duplicate_revision(
    tmp_path: Path,
) -> None:
    repository = RegimeBayesianRepository(
        tmp_path
    )

    repository.save(
        make_state(),
        revision=1,
        saved_at=FIXED_TIME,
    )

    with pytest.raises(FileExistsError):
        repository.save(
            make_state(2.0),
            revision=1,
            saved_at=FIXED_TIME,
        )


def test_repository_missing_revision_raises(
    tmp_path: Path,
) -> None:
    repository = RegimeBayesianRepository(
        tmp_path
    )

    with pytest.raises(
        RegimeBayesianNotFoundError
    ):
        repository.load_revision(1)


def test_repository_missing_latest_raises(
    tmp_path: Path,
) -> None:
    repository = RegimeBayesianRepository(
        tmp_path
    )

    with pytest.raises(
        RegimeBayesianNotFoundError
    ):
        repository.load_latest()


def test_load_latest_skips_corrupt_newest(
    tmp_path: Path,
) -> None:
    repository = RegimeBayesianRepository(
        tmp_path
    )

    repository.save(
        make_state(2.0),
        revision=1,
        saved_at=FIXED_TIME,
    )

    (
        tmp_path / "revision-00000002.json"
    ).write_text(
        "{not-json",
        encoding="utf-8",
    )

    latest = repository.load_latest()

    assert latest.revision == 1


def test_history_skips_corrupt_snapshot(
    tmp_path: Path,
) -> None:
    repository = RegimeBayesianRepository(
        tmp_path
    )

    repository.save(
        make_state(1.0),
        revision=1,
        saved_at=FIXED_TIME,
    )

    (
        tmp_path / "revision-00000002.json"
    ).write_text(
        "{broken",
        encoding="utf-8",
    )

    repository.save(
        make_state(3.0),
        revision=3,
        saved_at=FIXED_TIME,
    )

    assert tuple(
        snapshot.revision
        for snapshot in repository.history()
    ) == (1, 3)


def test_load_latest_can_refuse_corrupt_newest(
    tmp_path: Path,
) -> None:
    repository = RegimeBayesianRepository(
        tmp_path
    )

    repository.save(
        make_state(),
        revision=1,
        saved_at=FIXED_TIME,
    )

    (
        tmp_path / "revision-00000002.json"
    ).write_text(
        "{broken",
        encoding="utf-8",
    )

    with pytest.raises(
        RegimeBayesianSerializationError
    ):
        repository.load_latest(
            skip_corrupt=False
        )