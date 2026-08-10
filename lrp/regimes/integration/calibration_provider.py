from __future__ import annotations

from typing import Protocol, runtime_checkable

from lrp.evolution.contracts.regime_calibration import (
    RegimeCalibration,
)
from lrp.regimes.calibration_repository import (
    RegimeCalibrationNotFoundError,
    RegimeCalibrationRepository,
)


@runtime_checkable
class RegimeCalibrationProvider(Protocol):
    """Supply regime calibration for prediction."""

    def get_calibration(
        self,
        *,
        round_no: int,
    ) -> RegimeCalibration | None:
        """Return the active calibration or None."""
        ...


class StaticRegimeCalibrationProvider:
    """Return one fixed regime calibration."""

    def __init__(
        self,
        calibration: RegimeCalibration | None,
    ) -> None:
        if (
            calibration is not None
            and not isinstance(
                calibration,
                RegimeCalibration,
            )
        ):
            raise TypeError(
                "calibration must be a "
                "RegimeCalibration or None"
            )

        self._calibration = calibration

    def get_calibration(
        self,
        *,
        round_no: int,
    ) -> RegimeCalibration | None:
        self._validate_round_no(round_no)
        return self._calibration

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


class RepositoryRegimeCalibrationProvider:
    """Load the latest persisted regime calibration."""

    def __init__(
        self,
        repository: RegimeCalibrationRepository,
    ) -> None:
        if not isinstance(
            repository,
            RegimeCalibrationRepository,
        ):
            raise TypeError(
                "repository must be a "
                "RegimeCalibrationRepository"
            )

        self._repository = repository

    @property
    def repository(
        self,
    ) -> RegimeCalibrationRepository:
        return self._repository

    def get_calibration(
        self,
        *,
        round_no: int,
    ) -> RegimeCalibration | None:
        StaticRegimeCalibrationProvider._validate_round_no(
            round_no
        )

        try:
            snapshot = self.repository.load_latest()
        except RegimeCalibrationNotFoundError:
            return None

        return snapshot.calibration