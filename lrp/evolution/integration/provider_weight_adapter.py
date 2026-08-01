from __future__ import annotations

from lrp.evolution.integration.adaptive_weight_adapter import (
    AdaptiveEvolutionWeightAdapter,
)
from lrp.evolution.integration.profile_provider import (
    AdaptiveWeightProfileProvider,
)
from lrp.evolution.integration.weight_adapter import (
    EvolutionWeightAdapter,
)
from lrp.prediction.probability import (
    ProbabilityVector,
)


class ProviderEvolutionWeightAdapter(
    EvolutionWeightAdapter[ProbabilityVector]
):
    """Adjust probabilities using a provider-supplied profile."""

    def __init__(
        self,
        provider: AdaptiveWeightProfileProvider,
    ) -> None:
        if not isinstance(
            provider,
            AdaptiveWeightProfileProvider,
        ):
            raise TypeError(
                "provider must implement "
                "AdaptiveWeightProfileProvider"
            )

        self._provider = provider

    @property
    def provider(
        self,
    ) -> AdaptiveWeightProfileProvider:
        return self._provider

    def adjust(
        self,
        probability_vector: ProbabilityVector,
        *,
        round_no: int,
        seed: int,
    ) -> ProbabilityVector:
        profile = self.provider.get_profile(
            round_no=round_no
        )

        if profile is None:
            return probability_vector

        return AdaptiveEvolutionWeightAdapter(
            profile
        ).adjust(
            probability_vector,
            round_no=round_no,
            seed=seed,
        )
