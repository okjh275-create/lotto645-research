from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from lrp.evolution.contracts.models import (
    AdaptiveWeightProfile,
)
from lrp.evolution.integration import (
    EvolutionAdapterFactory,
    NoOpEvolutionWeightAdapter,
    ProviderEvolutionWeightAdapter,
)
from lrp.evolution.storage import (
    SnapshotRepository,
)


def make_profile(
    revision: int,
) -> AdaptiveWeightProfile:
    return AdaptiveWeightProfile.default(
        revision=revision,
        generated_at=datetime(
            2026,
            8,
            2,
            tzinfo=timezone.utc,
        ),
    )


def test_build_returns_noop_by_default() -> None:
    adapter = EvolutionAdapterFactory.build()

    assert isinstance(
        adapter,
        NoOpEvolutionWeightAdapter,
    )


def test_explicit_adapter_has_priority(
    tmp_path: Path,
) -> None:
    explicit = NoOpEvolutionWeightAdapter()

    adapter = EvolutionAdapterFactory.build(
        evolution=explicit,
        snapshot_root=tmp_path,
    )

    assert adapter is explicit


def test_builds_provider_from_path(
    tmp_path: Path,
) -> None:
    adapter = EvolutionAdapterFactory.build(
        snapshot_root=tmp_path
    )

    assert isinstance(
        adapter,
        ProviderEvolutionWeightAdapter,
    )


def test_builds_provider_from_string(
    tmp_path: Path,
) -> None:
    adapter = EvolutionAdapterFactory.build(
        snapshot_root=str(tmp_path)
    )

    assert isinstance(
        adapter,
        ProviderEvolutionWeightAdapter,
    )


def test_provider_loads_latest_profile(
    tmp_path: Path,
) -> None:
    repository = SnapshotRepository(
        tmp_path
    )
    repository.save(make_profile(1))
    repository.save(make_profile(2))

    adapter = EvolutionAdapterFactory.build(
        snapshot_root=tmp_path
    )

    assert isinstance(
        adapter,
        ProviderEvolutionWeightAdapter,
    )

    profile = adapter.provider.get_profile(
        round_no=1220
    )

    assert profile is not None
    assert profile.revision == 2


def test_empty_repository_returns_none(
    tmp_path: Path,
) -> None:
    adapter = EvolutionAdapterFactory.build(
        snapshot_root=tmp_path
    )

    assert isinstance(
        adapter,
        ProviderEvolutionWeightAdapter,
    )
    assert adapter.provider.get_profile(
        round_no=1220
    ) is None


@pytest.mark.parametrize(
    "snapshot_root",
    ["", "   "],
)
def test_empty_root_is_rejected(
    snapshot_root: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        EvolutionAdapterFactory.build(
            snapshot_root=snapshot_root
        )


def test_invalid_root_type_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="string or Path",
    ):
        EvolutionAdapterFactory.build(
            snapshot_root=1,  # type: ignore[arg-type]
        )


def test_invalid_explicit_adapter_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="EvolutionWeightAdapter",
    ):
        EvolutionAdapterFactory.build(
            evolution=object(),  # type: ignore[arg-type]
        )
