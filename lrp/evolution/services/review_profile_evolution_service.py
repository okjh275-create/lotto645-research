from __future__ import annotations

from datetime import datetime
from math import isfinite

from lrp.evolution.contracts.execution import (
    EvolutionRunResult,
)
from lrp.evolution.contracts.learning_context import (
    LearningContext,
)
from lrp.evolution.contracts.pipeline import (
    EvolutionPipelineRequest,
)
from lrp.evolution.integration.review_signal_extractor import (
    ReviewSignalExtractor,
)
from lrp.evolution.services.coordinator import (
    EvolutionCoordinator,
)
from lrp.evolution.storage import (
    SnapshotNotFoundError,
)


class ReviewProfileEvolutionService:
    """Promote review-learning rewards into an adaptive profile."""

    def __init__(
        self,
        coordinator: EvolutionCoordinator,
        signal_extractor: ReviewSignalExtractor | None = None,
    ) -> None:
        if not isinstance(
            coordinator,
            EvolutionCoordinator,
        ):
            raise TypeError(
                "coordinator must be an "
                "EvolutionCoordinator"
            )

        if (
            signal_extractor is not None
            and not isinstance(
                signal_extractor,
                ReviewSignalExtractor,
            )
        ):
            raise TypeError(
                "signal_extractor must be a "
                "ReviewSignalExtractor"
            )

        self._coordinator = coordinator
        self._signal_extractor = (
            signal_extractor
            if signal_extractor is not None
            else ReviewSignalExtractor()
        )

    @property
    def coordinator(self) -> EvolutionCoordinator:
        return self._coordinator

    @property
    def signal_extractor(
        self,
    ) -> ReviewSignalExtractor:
        return self._signal_extractor

    def evolve(
        self,
        *,
        context: LearningContext,
        generated_at: datetime,
        confidence: float = 0.50,
    ) -> EvolutionRunResult:
        if not isinstance(
            context,
            LearningContext,
        ):
            raise TypeError(
                "context must be a LearningContext"
            )

        normalized_confidence = (
            self._validate_confidence(confidence)
        )
        revision = self._next_revision()

        request = EvolutionPipelineRequest(
            signals=self.signal_extractor.extract(
                context
            ),
            confidence=normalized_confidence,
            sample_size=(
                self.signal_extractor.sample_size(
                    context
                )
            ),
            revision=revision,
            generated_at=generated_at,
        )

        return self.coordinator.execute(request)

    def _next_revision(self) -> int:
        try:
            snapshot = (
                self.coordinator
                .repository
                .load_latest()
            )
        except SnapshotNotFoundError:
            return 1

        return snapshot.profile.revision + 1

    @staticmethod
    def _validate_confidence(
        confidence: float,
    ) -> float:
        if isinstance(confidence, bool):
            raise TypeError(
                "confidence must be numeric"
            )

        if not isinstance(
            confidence,
            (int, float),
        ):
            raise TypeError(
                "confidence must be numeric"
            )

        normalized = float(confidence)

        if not isfinite(normalized):
            raise ValueError(
                "confidence must be finite"
            )

        if not 0.0 <= normalized <= 1.0:
            raise ValueError(
                "confidence must be between "
                "0.0 and 1.0"
            )

        return normalized
