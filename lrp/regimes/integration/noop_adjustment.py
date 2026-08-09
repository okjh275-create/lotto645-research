from __future__ import annotations

from typing import TypeVar

from .adjustment import (
    GlobalRegimeAdjustmentAdapter,
)


ProbabilityVectorT = TypeVar(
    "ProbabilityVectorT",
)


class NoOpGlobalRegimeAdjustmentAdapter(
    GlobalRegimeAdjustmentAdapter[ProbabilityVectorT],
):
    """Preserve probability vector without regime modification."""

    def adjust(
        self,
        probability_vector: ProbabilityVectorT,
        *,
        global_regime: object | None,
        round_no: int,
        seed: int,
    ) -> ProbabilityVectorT:
        self._validate_round_no(round_no)
        self._validate_seed(seed)

        if probability_vector is None:
            raise TypeError(
                "probability_vector must not be None"
            )

        return probability_vector

    @staticmethod
    def _validate_round_no(
        round_no: int,
    ) -> None:
        if isinstance(round_no, bool):
            raise TypeError(
                "round_no must be an integer"
            )

        if not isinstance(round_no, int):
            raise TypeError(
                "round_no must be an integer"
            )

        if round_no < 1:
            raise ValueError(
                "round_no must be greater than "
                "or equal to 1"
            )

    @staticmethod
    def _validate_seed(
        seed: int,
    ) -> None:
        if isinstance(seed, bool):
            raise TypeError(
                "seed must be an integer"
            )

        if not isinstance(seed, int):
            raise TypeError(
                "seed must be an integer"
            )
