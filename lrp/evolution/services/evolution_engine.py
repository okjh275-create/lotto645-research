from __future__ import annotations

from lrp.evolution.contracts.execution import (
    EvolutionRunResult,
)
from lrp.evolution.contracts.pipeline import (
    EvolutionPipelineRequest,
)
from lrp.evolution.services.coordinator import (
    EvolutionCoordinator,
)


class EvolutionEngine:
    """Public facade for executing evolution runs."""

    def __init__(
        self,
        coordinator: EvolutionCoordinator,
    ) -> None:
        if not isinstance(
            coordinator,
            EvolutionCoordinator,
        ):
            raise TypeError(
                "coordinator must be an "
                "EvolutionCoordinator"
            )

        self._coordinator = coordinator

    @property
    def coordinator(self) -> EvolutionCoordinator:
        return self._coordinator

    def run(
        self,
        request: EvolutionPipelineRequest,
    ) -> EvolutionRunResult:
        if not isinstance(
            request,
            EvolutionPipelineRequest,
        ):
            raise TypeError(
                "request must be an "
                "EvolutionPipelineRequest"
            )

        return self.coordinator.execute(request)