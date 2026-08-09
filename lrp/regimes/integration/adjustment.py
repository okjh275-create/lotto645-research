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
class GlobalRegimeAdjustmentAdapter(
    Protocol[ProbabilityVectorT],
):
    """Adjust a probability vector using global regime context."""

    def adjust(
        self,
        probability_vector: ProbabilityVectorT,
        *,
        global_regime: object | None,
        round_no: int,
        seed: int,
    ) -> ProbabilityVectorT:
        """Return a regime-adjusted probability vector."""
        ...
