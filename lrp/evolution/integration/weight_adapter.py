from __future__ import annotations

from typing import (
    Protocol,
    TypeVar,
    runtime_checkable,
)


ProbabilityVectorT = TypeVar(
    "ProbabilityVectorT",
)


@runtime_checkable
class EvolutionWeightAdapter(
    Protocol[ProbabilityVectorT],
):
    """Adjust a prediction probability vector before sampling."""

    def adjust(
        self,
        probability_vector: ProbabilityVectorT,
        *,
        round_no: int,
        seed: int,
    ) -> ProbabilityVectorT:
        """Return an adjusted probability vector."""
        ...
