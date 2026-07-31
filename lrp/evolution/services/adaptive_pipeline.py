from __future__ import annotations

from lrp.evolution.algorithms.adaptive import (
    AdaptiveWeightCalculator,
)
from lrp.evolution.contracts.models import (
    AdaptiveWeightProfile,
)
from lrp.evolution.contracts.pipeline import (
    EvolutionPipelineRequest,
)


class AdaptiveEvolutionPipeline:
    """Adapt AdaptiveWeightCalculator to EvolutionPipeline."""

    def __init__(
        self,
        calculator: AdaptiveWeightCalculator | None = None,
    ) -> None:
        if (
            calculator is not None
            and not isinstance(
                calculator,
                AdaptiveWeightCalculator,
            )
        ):
            raise TypeError(
                "calculator must be an "
                "AdaptiveWeightCalculator or None"
            )

        self._calculator = (
            calculator
            if calculator is not None
            else AdaptiveWeightCalculator()
        )

    @property
    def calculator(
        self,
    ) -> AdaptiveWeightCalculator:
        return self._calculator

    def calculate(
        self,
        request: EvolutionPipelineRequest,
    ) -> AdaptiveWeightProfile:
        """Calculate a profile from a pipeline request."""

        if not isinstance(
            request,
            EvolutionPipelineRequest,
        ):
            raise TypeError(
                "request must be an "
                "EvolutionPipelineRequest"
            )

        profile = self.calculator.calculate(
            request.signals,
            confidence=request.confidence,
            sample_size=request.sample_size,
            revision=request.revision,
            generated_at=request.generated_at,
            baseline=request.previous_profile,
        )

        if not isinstance(
            profile,
            AdaptiveWeightProfile,
        ):
            raise TypeError(
                "adaptive calculator must return an "
                "AdaptiveWeightProfile"
            )

        return profile
