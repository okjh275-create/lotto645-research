from __future__ import annotations

from typing import Protocol, runtime_checkable

from lrp.evolution.contracts.models import (
    AdaptiveWeightProfile,
)
from lrp.evolution.storage import (
    SnapshotNotFoundError,
    SnapshotRepository,
)


@runtime_checkable
class AdaptiveWeightProfileProvider(Protocol):
    """Supply an adaptive profile for prediction."""

    def get_profile(
        self,
        *,
        round_no: int,
    ) -> AdaptiveWeightProfile | None:
        """Return the active profile or None."""
        ...


class StaticProfileProvider:
    """Return one fixed profile."""

    def __init__(
        self,
        profile: AdaptiveWeightProfile | None,
    ) -> None:
        if (
            profile is not None
            and not isinstance(
                profile,
                AdaptiveWeightProfile,
            )
        ):
            raise TypeError(
                "profile must be an "
                "AdaptiveWeightProfile or None"
            )

        self._profile = profile

    def get_profile(
        self,
        *,
        round_no: int,
    ) -> AdaptiveWeightProfile | None:
        self._validate_round_no(round_no)
        return self._profile

    @staticmethod
    def _validate_round_no(
        round_no: int,
    ) -> None:
        if (
            isinstance(round_no, bool)
            or not isinstance(round_no, int)
        ):
            raise TypeError(
                "round_no must be an integer"
            )

        if round_no < 1:
            raise ValueError(
                "round_no must be greater than "
                "or equal to 1"
            )


class SnapshotProfileProvider:
    """Load the latest persisted adaptive profile."""

    def __init__(
        self,
        repository: SnapshotRepository,
    ) -> None:
        if not isinstance(
            repository,
            SnapshotRepository,
        ):
            raise TypeError(
                "repository must be a "
                "SnapshotRepository"
            )

        self._repository = repository

    @property
    def repository(self) -> SnapshotRepository:
        return self._repository

    def get_profile(
        self,
        *,
        round_no: int,
    ) -> AdaptiveWeightProfile | None:
        StaticProfileProvider._validate_round_no(
            round_no
        )

        try:
            snapshot = (
                self.repository.load_latest()
            )
        except SnapshotNotFoundError:
            return None

        return snapshot.profile
