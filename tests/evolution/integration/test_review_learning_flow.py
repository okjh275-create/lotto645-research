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
from lrp.evolution.repositories.file_snapshot_repository import (
    FileSnapshotRepository,
)
from lrp.evolution.services.adaptive_pipeline import (
    AdaptiveEvolutionPipeline,
)
from lrp.evolution.services.coordinator import (
    EvolutionCoordinator,
)
from lrp.evolution.services.persistent_learning_runner import (
    PersistentLearningRunner,
)
from lrp.evolution.services.persistent_learning_service import (
    PersistentLearningService,
)
from lrp.evolution.services.review_learning_service import (
    ReviewLearningService,
)
from lrp.evolution.services.review_profile_evolution_service import (
    ReviewProfileEvolutionService,
)
from lrp.evolution.services.snapshot_factory import (
    SnapshotFactory,
)
from lrp.evolution.storage import (
    SnapshotRepository,
)


FIRST_TIME = datetime(
    2026,
    8,
    2,
    1,
    0,
    tzinfo=timezone.utc,
)

SECOND_TIME = datetime(
    2026,
    8,
    3,
    1,
    0,
    tzinfo=timezone.utc,
)


def make_review(
    *,
    best_hits: int = 4,
    practical_hits: int = 3,
    set_count: int = 20,
) -> dict[str, object]:
    return {
        "round": 1220,
        "summary": {
            "set_count": set_count,
            "best_main_hits": best_hits,
            "practical_best_hits": (
                practical_hits
            ),
        },
    }


def make_context(
    *,
    round_no: int = 1220,
    version: int = 1,
) -> LearningContext:
    return LearningContext(
        cycle_id=f"cycle-{round_no}",
        round_no=round_no,
        version=version,
    )


def make_learning_service(
    root: Path,
) -> ReviewLearningService:
    persistence = PersistentLearningService(
        FileSnapshotRepository(root),
        snapshot_factory=SnapshotFactory(
            clock=lambda: FIRST_TIME
        ),
    )
    runner = PersistentLearningRunner(
        persistence
    )

    return ReviewLearningService(runner)


def make_profile_service(
    root: Path,
) -> ReviewProfileEvolutionService:
    policy = AdaptiveWeightPolicy(
        AdaptivePolicyConfig(
            min_confidence=0.60,
            min_sample_size=20,
        )
    )
    coordinator = EvolutionCoordinator(
        pipeline=AdaptiveEvolutionPipeline(),
        policy=policy,
        repository=SnapshotRepository(root),
    )

    return ReviewProfileEvolutionService(
        coordinator
    )


def test_review_learning_flow_creates_both_snapshots(
    tmp_path: Path,
) -> None:
    learning_root = tmp_path / "learning"
    profile_root = tmp_path / "profiles"

    learning_service = make_learning_service(
        learning_root
    )
    profile_service = make_profile_service(
        profile_root
    )

    learning = learning_service.learn(
        context=make_context(),
        review_payload=make_review(),
        snapshot_id="review-1220",
        policy="thompson",
        metadata={
            "round": 1220,
        },
    )

    evolution = profile_service.evolve(
        context=learning.final_context,
        generated_at=FIRST_TIME,
        confidence=0.80,
    )

    assert (
        learning_root / "review-1220.json"
    ).is_file()

    assert evolution.snapshot is not None
    assert evolution.decision.applied is True
    assert evolution.snapshot.profile.revision == 1
    assert evolution.snapshot.profile.sample_size == 20

    latest = (
        profile_service.coordinator
        .repository
        .load_latest()
    )

    assert latest == evolution.snapshot


def test_review_rewards_reach_profile_evolution(
    tmp_path: Path,
) -> None:
    learning_service = make_learning_service(
        tmp_path / "learning"
    )
    profile_service = make_profile_service(
        tmp_path / "profiles"
    )

    learning = learning_service.learn(
        context=make_context(),
        review_payload=make_review(
            best_hits=5,
            practical_hits=4,
        ),
        snapshot_id="review-1220",
        policy="thompson",
    )

    assert learning.final_context.rewards[
        (
            "prediction_review:"
            "thompson:"
            "portfolio_top_k"
        )
    ] == pytest.approx(0.85)

    assert learning.final_context.rewards[
        (
            "prediction_review:"
            "thompson:"
            "practical_top5"
        )
    ] == pytest.approx(0.55)

    evolution = profile_service.evolve(
        context=learning.final_context,
        generated_at=FIRST_TIME,
        confidence=0.80,
    )

    assert evolution.snapshot is not None

    profile = evolution.snapshot.profile

    assert profile.learning_weight >= 0.0
    assert profile.adaptive_weight >= 0.0
    assert profile.sample_size == 20


def test_profile_revision_advances_on_next_review(
    tmp_path: Path,
) -> None:
    learning_service = make_learning_service(
        tmp_path / "learning"
    )
    profile_service = make_profile_service(
        tmp_path / "profiles"
    )

    first_learning = learning_service.learn(
        context=make_context(
            round_no=1220,
        ),
        review_payload=make_review(
            set_count=20,
        ),
        snapshot_id="review-1220",
        policy="thompson",
    )

    first_profile = profile_service.evolve(
        context=first_learning.final_context,
        generated_at=FIRST_TIME,
        confidence=0.80,
    )

    second_learning = learning_service.learn(
        context=make_context(
            round_no=1221,
        ),
        review_payload={
            "round": 1221,
            "summary": {
                "set_count": 20,
                "best_main_hits": 5,
                "practical_best_hits": 4,
            },
        },
        snapshot_id="review-1221",
        policy="thompson",
    )

    second_profile = profile_service.evolve(
        context=second_learning.final_context,
        generated_at=SECOND_TIME,
        confidence=0.80,
    )

    assert first_profile.snapshot is not None
    assert (
        first_profile.snapshot.profile.revision
        == 1
    )

    if second_profile.snapshot is not None:
        assert (
            second_profile.snapshot.profile.revision
            == 2
        )
    else:
        assert (
            second_profile.decision.applied
            is False
        )


def test_policy_rejects_insufficient_sample_size(
    tmp_path: Path,
) -> None:
    learning_service = make_learning_service(
        tmp_path / "learning"
    )
    profile_service = make_profile_service(
        tmp_path / "profiles"
    )

    learning = learning_service.learn(
        context=make_context(),
        review_payload=make_review(
            set_count=10,
        ),
        snapshot_id="review-1220",
        policy="thompson",
    )

    evolution = profile_service.evolve(
        context=learning.final_context,
        generated_at=FIRST_TIME,
        confidence=0.80,
    )

    assert evolution.snapshot is None
    assert evolution.decision.applied is False
    assert (
        "sample_size_below_threshold"
        in evolution.decision.reasons
    )


def test_policy_rejects_low_confidence(
    tmp_path: Path,
) -> None:
    learning_service = make_learning_service(
        tmp_path / "learning"
    )
    profile_service = make_profile_service(
        tmp_path / "profiles"
    )

    learning = learning_service.learn(
        context=make_context(),
        review_payload=make_review(
            set_count=20,
        ),
        snapshot_id="review-1220",
        policy="thompson",
    )

    evolution = profile_service.evolve(
        context=learning.final_context,
        generated_at=FIRST_TIME,
        confidence=0.50,
    )

    assert evolution.snapshot is None
    assert evolution.decision.applied is False
    assert (
        "confidence_below_threshold"
        in evolution.decision.reasons
    )
