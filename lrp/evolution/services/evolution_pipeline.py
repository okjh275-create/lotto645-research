from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from lrp.evolution.contracts.models import (
    AdaptiveWeightProfile,
)
from lrp.evolution.contracts.pipeline import (
    EvolutionPipelineRequest,
)


@runtime_checkable
class EvolutionPipeline(Protocol):
    """Common interface for evolution algorithms."""

    def calculate(
        self,
        request: EvolutionPipelineRequest,
    ) -> AdaptiveWeightProfile:
        """Calculate a candidate adaptive profile."""
        ...


class CallableEvolutionPipeline:
    """Pipeline adapter backed by a validated callable."""

    def __init__(
        self,
        calculator: Callable[
            [EvolutionPipelineRequest],
            AdaptiveWeightProfile,
        ],
    ) -> None:
        if not callable(calculator):
            raise TypeError(
                "calculator must be callable"
            )

        self._calculator = calculator

    def calculate(
        self,
        request: EvolutionPipelineRequest,
    ) -> AdaptiveWeightProfile:
        if not isinstance(
            request,
            EvolutionPipelineRequest,
        ):
            raise TypeError(
                "request must be an "
                "EvolutionPipelineRequest"
            )

        profile = self._calculator(request)

        if not isinstance(
            profile,
            AdaptiveWeightProfile,
        ):
            raise TypeError(
                "pipeline calculator must return an "
                "AdaptiveWeightProfile"
            )

        return profile