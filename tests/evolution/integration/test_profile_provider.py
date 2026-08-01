from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from lrp.evolution.contracts.models import (
    AdaptiveWeightProfile,
)
from lrp.evolution.integration import (
    AdaptiveWeightProfileProvider,
    SnapshotProfileProvider,
    StaticProfileProvider,
)
from lrp.evolution.storage import (
    SnapshotRepository,
)


FIXED_TIME = datetime(
    2026,
    8,
    2,
    tzinfo=timezone.utc,
)


def make_profile(
    revision: int = 1,
) -> AdaptiveWeightProfile:
    return AdaptiveWeightProfile.default(
        revision=revision,
        generated_at=FIXED_TIME,
    )


def test_static_provider_returns_profile() -> None:
    profile = make_profile()

    provider = StaticProfileProvider(
        profile
    )

    assert provider.get_profile(
        round_no=1220
    ) is profile


def test_static_provider_can_return_none() -> None:
    assert StaticProfileProvider(
        None
    ).get_profile(
        round_no=1220
    ) is None


def test_static_provider_matches_protocol() -> None:
    assert isinstance(
        StaticProfileProvider(None),
        AdaptiveWeightProfileProvider,
    )


def test_snapshot_provider_returns_none_when_empty(
    tmp_path: Path,
) -> None:
    provider = SnapshotProfileProvider(
        SnapshotRepository(tmp_path)
    )

    assert provider.get_profile(
        round_no=1220
    ) is None


def test_snapshot_provider_loads_latest_profile(
    tmp_path: Path,
) -> None:
    repository = SnapshotRepository(
        tmp_path
    )
    repository.save(make_profile(1))
    repository.save(make_profile(2))

    provider = SnapshotProfileProvider(
        repository
    )

    profile = provider.get_profile(
        round_no=1220
    )

    assert profile is not None
    assert profile.revision == 2


def test_invalid_repository_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="SnapshotRepository",
    ):
        SnapshotProfileProvider(
            object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "round_no",
    [0, -1],
)
def test_invalid_round_is_rejected(
    round_no: int,
) -> None:
    with pytest.raises(ValueError):
        StaticProfileProvider(None).get_profile(
            round_no=round_no
        )
