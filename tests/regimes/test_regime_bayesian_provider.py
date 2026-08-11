from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lrp.evolution.contracts.bayesian import (
    BayesianPosterior,
)
from lrp.regimes.bayesian_repository import (
    RegimeBayesianRepository,
)
from lrp.regimes.bayesian_state import (
    RegimeBayesianState,
)
from lrp.regimes.integration.bayesian_provider import (
    RepositoryRegimeBayesianProvider,
    StaticRegimeBayesianProvider,
)


def test_static_bayesian_provider_returns_state() -> None:
    state = RegimeBayesianState.default()

    provider = StaticRegimeBayesianProvider(state)

    assert (
        provider.get_bayesian_state(round_no=1220)
        is state
    )


def test_static_bayesian_provider_accepts_none() -> None:
    provider = StaticRegimeBayesianProvider(None)

    assert (
        provider.get_bayesian_state(round_no=1220)
        is None
    )


def test_static_bayesian_provider_rejects_invalid_state() -> None:
    with pytest.raises(
        TypeError,
        match="state",
    ):
        StaticRegimeBayesianProvider(object())


def test_repository_bayesian_provider_returns_none_when_empty(
    tmp_path,
) -> None:
    repository = RegimeBayesianRepository(
        tmp_path / "regime-bayesian"
    )
    provider = RepositoryRegimeBayesianProvider(
        repository
    )

    assert (
        provider.get_bayesian_state(round_no=1220)
        is None
    )


def test_repository_bayesian_provider_loads_latest_state(
    tmp_path,
) -> None:
    repository = RegimeBayesianRepository(
        tmp_path / "regime-bayesian"
    )

    first = RegimeBayesianState.default()

    second = RegimeBayesianState.from_posteriors(
        {
            "gap_recovery": BayesianPosterior(
                alpha=3.0,
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

    provider = RepositoryRegimeBayesianProvider(
        repository
    )

    loaded = provider.get_bayesian_state(
        round_no=1220
    )

    assert loaded == second
    assert (
        loaded.posteriors[
            "gap_recovery"
        ].alpha
        == 3.0
    )


@pytest.mark.parametrize(
    "round_no",
    [
        0,
        -1,
    ],
)
def test_bayesian_provider_rejects_invalid_round(
    round_no: int,
) -> None:
    provider = StaticRegimeBayesianProvider(
        RegimeBayesianState.default()
    )

    with pytest.raises(ValueError):
        provider.get_bayesian_state(
            round_no=round_no
        )

def test_repository_bayesian_provider_returns_none_when_all_snapshots_corrupt(
    tmp_path,
) -> None:
    repository = RegimeBayesianRepository(
        tmp_path / "regime-bayesian"
    )

    root = tmp_path / "regime-bayesian"
    root.mkdir(parents=True, exist_ok=True)

    (
        root / "revision-00000001.json"
    ).write_text(
        "{broken",
        encoding="utf-8",
    )

    (
        root / "revision-00000002.json"
    ).write_text(
        "{also-broken",
        encoding="utf-8",
    )

    provider = RepositoryRegimeBayesianProvider(
        repository
    )

    assert (
        provider.get_bayesian_state(round_no=1220)
        is None
    )


def test_repository_bayesian_provider_falls_back_from_corrupt_latest(
    tmp_path,
) -> None:
    repository = RegimeBayesianRepository(
        tmp_path / "regime-bayesian"
    )

    expected = RegimeBayesianState.default()

    repository.save(
        expected,
        revision=1,
        sample_size=10,
        saved_at=datetime(
            2026,
            8,
            11,
            tzinfo=timezone.utc,
        ),
    )

    (
        tmp_path
        / "regime-bayesian"
        / "revision-00000002.json"
    ).write_text(
        "{broken",
        encoding="utf-8",
    )

    provider = RepositoryRegimeBayesianProvider(
        repository
    )

    assert (
        provider.get_bayesian_state(round_no=1220)
        == expected
    )
