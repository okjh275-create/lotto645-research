from __future__ import annotations

from typing import (
    Iterable,
    Protocol,
    TypeVar,
    runtime_checkable,
)


ArmT = TypeVar(
    "ArmT",
    contravariant=True,
)
DecisionT = TypeVar(
    "DecisionT",
    covariant=True,
)


@runtime_checkable
class BanditPolicy(
    Protocol[ArmT, DecisionT],
):
    """Structural interface for bandit selection policies."""

    def select(
        self,
        arms: Iterable[ArmT],
    ) -> DecisionT:
        """Select one arm and return a decision."""
        ...
