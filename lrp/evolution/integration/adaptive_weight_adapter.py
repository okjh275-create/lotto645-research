from __future__ import annotations

import math

from lrp.contracts import ContractError
from lrp.evolution.contracts.models import (
    AdaptiveWeightProfile,
)
from lrp.evolution.integration.weight_adapter import (
    EvolutionWeightAdapter,
)
from lrp.prediction.probability import (
    NumberProbability,
    ProbabilityVector,
    ordered_probabilities,
)


class AdaptiveEvolutionWeightAdapter(
    EvolutionWeightAdapter[ProbabilityVector]
):
    """Rebuild a probability vector with adaptive fusion weights."""

    REQUIRED_COMPONENTS = (
        "hot",
        "cold",
        "gap",
        "trend",
        "transition",
        "learning",
        "adaptive",
    )

    def __init__(
        self,
        profile: AdaptiveWeightProfile,
    ) -> None:
        if not isinstance(
            profile,
            AdaptiveWeightProfile,
        ):
            raise TypeError(
                "profile must be an "
                "AdaptiveWeightProfile"
            )

        self._profile = profile

    @property
    def profile(self) -> AdaptiveWeightProfile:
        return self._profile

    def adjust(
        self,
        probability_vector: ProbabilityVector,
        *,
        round_no: int,
        seed: int,
    ) -> ProbabilityVector:
        self._validate_round_no(round_no)
        self._validate_seed(seed)

        if not isinstance(
            probability_vector,
            ProbabilityVector,
        ):
            raise TypeError(
                "probability_vector must be a "
                "ProbabilityVector"
            )

        if (
            probability_vector.round_no is not None
            and probability_vector.round_no
            != round_no
        ):
            raise ValueError(
                "probability_vector round_no does not "
                "match requested round_no"
            )

        weights = (
            self.profile.to_probability_weights()
        )

        raw_records: list[
            tuple[
                NumberProbability,
                float,
            ]
        ] = []

        for item in probability_vector.probabilities:
            self._validate_components(item)

            raw_score = sum(
                item.components[name]
                * weights[name]
                for name in self.REQUIRED_COMPONENTS
            )

            if (
                not math.isfinite(raw_score)
                or raw_score < 0.0
            ):
                raise ContractError(
                    "adaptive raw score must be "
                    "finite and non-negative"
                )

            raw_records.append(
                (
                    item,
                    raw_score,
                )
            )

        raw_total = sum(
            raw_score
            for _, raw_score in raw_records
        )

        if (
            not math.isfinite(raw_total)
            or raw_total <= 0.0
        ):
            raise ContractError(
                "adaptive raw score total must be "
                "finite and positive"
            )

        ranked = sorted(
            raw_records,
            key=lambda record: (
                -record[1],
                record[0].number,
            ),
        )

        rank_by_number = {
            item.number: rank
            for rank, (
                item,
                _,
            ) in enumerate(
                ranked,
                start=1,
            )
        }

        adjusted_records = (
            NumberProbability(
                number=item.number,
                probability=(
                    raw_score / raw_total
                ),
                raw_score=raw_score,
                rank=rank_by_number[
                    item.number
                ],
                components=dict(
                    item.components
                ),
                metadata={
                    **dict(item.metadata),
                    "evolution_adjusted": True,
                    "evolution_revision": (
                        self.profile.revision
                    ),
                },
            )
            for item, raw_score in raw_records
        )

        metadata = {
            **dict(probability_vector.metadata),
            "evolution_adapter": (
                type(self).__name__
            ),
            "evolution_revision": (
                self.profile.revision
            ),
            "evolution_confidence": (
                self.profile.confidence
            ),
            "evolution_sample_size": (
                self.profile.sample_size
            ),
            "evolution_seed": seed,
            "evolution_weights": dict(
                weights
            ),
        }

        return ProbabilityVector(
            round_no=(
                probability_vector.round_no
            ),
            generated_at_kst=(
                probability_vector
                .generated_at_kst
            ),
            probabilities=(
                ordered_probabilities(
                    adjusted_records
                )
            ),
            metadata=metadata,
        )

    @classmethod
    def _validate_components(
        cls,
        item: NumberProbability,
    ) -> None:
        missing = tuple(
            name
            for name in cls.REQUIRED_COMPONENTS
            if name not in item.components
        )

        if missing:
            raise ContractError(
                "probability components are missing: "
                + ", ".join(missing)
            )

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
