from __future__ import annotations

from typing import Protocol, runtime_checkable

from lrp.regimes.bayesian_repository import (
    RegimeBayesianNotFoundError,
    RegimeBayesianRepository,
)
from lrp.regimes.bayesian_state import (
    RegimeBayesianState,
)


@runtime_checkable
class RegimeBayesianProvider(Protocol):
    """Supply regime Bayesian state for prediction."""

    def get_bayesian_state(
        self,
        *,
        round_no: int,
    ) -> RegimeBayesianState | None:
        """Return the active Bayesian state or None."""
        ...


class StaticRegimeBayesianProvider:
    """Return one fixed regime Bayesian state."""

    def __init__(
        self,
        state: RegimeBayesianState | None,
    ) -> None:
        if (
            state is not None
            and not isinstance(
                state,
                RegimeBayesianState,
            )
        ):
            raise TypeError(
                "state must be a "
                "RegimeBayesianState or None"
            )

        self._state = state

    def get_bayesian_state(
        self,
        *,
        round_no: int,
    ) -> RegimeBayesianState | None:
        self._validate_round_no(round_no)
        return self._state

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


class RepositoryRegimeBayesianProvider:
    """Load the latest persisted regime Bayesian state."""

    def __init__(
        self,
        repository: RegimeBayesianRepository,
    ) -> None:
        if not isinstance(
            repository,
            RegimeBayesianRepository,
        ):
            raise TypeError(
                "repository must be a "
                "RegimeBayesianRepository"
            )

        self._repository = repository

    @property
    def repository(
        self,
    ) -> RegimeBayesianRepository:
        return self._repository

    def get_bayesian_state(
        self,
        *,
        round_no: int,
    ) -> RegimeBayesianState | None:
        StaticRegimeBayesianProvider._validate_round_no(
            round_no
        )

        try:
            snapshot = self.repository.load_latest()
        except RegimeBayesianNotFoundError:
            return None

        return snapshot.state
