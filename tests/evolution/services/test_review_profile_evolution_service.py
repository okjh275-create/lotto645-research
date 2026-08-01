from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from lrp.evolution.contracts.learning_context import (
    LearningContext,
)
from lrp.evolution.policies import (
    AdaptivePolicyConfig,
    AdaptiveWeightPolicy,
)
from lrp.evolution.services.adaptive_pipeline import (
    AdaptiveEvolutionPipeline,
)
from lrp.evolution.services.coordinator import (
    EvolutionCoordinator,
)
from lrp.evolution.services.review_profile_evolution_service import (
    ReviewProfileEvolutionService,
)
from lrp.evolution.storage import (
    SnapshotRepository,
)


FIXED_TIME = datetime(
    2026,
    8,
    2,
    1,
    0,
    tzinfo=timezone.utc,
)


def make_context(
    *,
    portfolio: float = 0.55,
    practical: float = 0.20,
) -> LearningContext:
    return LearningContext(
        cycle_id="cycle-1220",
        round_no=1220,
        rewards={
            (
                "prediction_review:thompson:"
                "portfolio_top_k"
            ): portfolio,
            (
                "prediction_review:thompson:"
                "practical_top5"
            ): practical,
        },
        metadata={
            "feedback_observation_count": 10,
        },
    )


def make_service(
    tmp_path: Path,
) -> ReviewProfileEvolutionService:
    coordinator = EvolutionCoordinator(
        pipeline=AdaptiveEvolutionPipeline(),
        policy=AdaptiveWeightPolicy(
            AdaptivePolicyConfig(
                min_confidence=0.0,
                min_sample_size=1,
            )
        ),
        repository=SnapshotRepository(
            tmp_path
        ),
    )

    return ReviewProfileEvolutionService(
        coordinator
    )


def test_evolve_creates_first_profile(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)

    result = service.evolve(
        context=make_context(),
        generated_at=FIXED_TIME,
        confidence=0.80,
    )

    assert result.snapshot is not None
    assert result.snapshot.profile.revision == 1
    assert (
        result.snapshot.profile.confidence
        == pytest.approx(0.80)
    )
    assert (
        result.snapshot.profile.sample_size
        == 10
    )


def test_evolve_persists_profile(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)

    result = service.evolve(
        context=make_context(),
        generated_at=FIXED_TIME,
    )

    assert result.snapshot is not None

    restored = (
        service.coordinator
        .repository
        .load_latest()
    )

    assert restored == result.snapshot


def test_evolve_advances_revision(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path)

    first = service.evolve(
        context=make_context(),
        generated_at=FIXED_TIME,
    )
    second = service.evolve(
        context=make_context(
            portfolio=0.85,
            practical=0.55,
        ),
        generated_at=datetime(
            2026,
            8,
            3,
            1,
            0,
            tzinfo=timezone.utc,
        ),
    )

    assert first.snapshot is not None
    assert first.snapshot.profile.revision == 1

    if second.snapshot is not None:
        assert second.snapshot.profile.revision == 2
    else:
        assert second.decision.applied is False


def test_evolve_maps_review_signals(
    tmp_path: Path,
) -> None:
    result = make_service(tmp_path).evolve(
        context=make_context(),
        generated_at=FIXED_TIME,
    )

    assert result.decision.profile is not None

    profile = result.decision.profile

    assert profile.learning_weight >= 0.0
    assert profile.adaptive_weight >= 0.0
    assert profile.revision == 1


def test_invalid_coordinator_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="EvolutionCoordinator",
    ):
        ReviewProfileEvolutionService(
            object()  # type: ignore[arg-type]
        )


def test_invalid_context_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        TypeError,
        match="LearningContext",
    ):
        make_service(tmp_path).evolve(
            context=object(),  # type: ignore[arg-type]
            generated_at=FIXED_TIME,
        )


@pytest.mark.parametrize(
    "confidence",
    [-0.1, 1.1],
)
def test_invalid_confidence_is_rejected(
    tmp_path: Path,
    confidence: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="between 0.0 and 1.0",
    ):
        make_service(tmp_path).evolve(
            context=make_context(),
            generated_at=FIXED_TIME,
            confidence=confidence,
        )


def test_boolean_confidence_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        TypeError,
        match="confidence must be numeric",
    ):
        make_service(tmp_path).evolve(
            context=make_context(),
            generated_at=FIXED_TIME,
            confidence=True,  # type: ignore[arg-type]
        )
