from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from lrp.evolution.policies import AdaptiveWeightPolicy
from lrp.evolution.repositories.file_snapshot_repository import FileSnapshotRepository
from lrp.evolution.services.adaptive_pipeline import AdaptiveEvolutionPipeline
from lrp.evolution.services.coordinator import EvolutionCoordinator
from lrp.evolution.services.persistent_learning_runner import PersistentLearningRunner
from lrp.evolution.services.persistent_learning_service import PersistentLearningService
from lrp.evolution.services.review_learning_service import ReviewLearningService
from lrp.evolution.services.review_profile_evolution_service import ReviewProfileEvolutionService
from lrp.evolution.storage import SnapshotRepository
from lrp.learning import LearningRepository
from lrp.outcomes import OutcomeBridge, OutcomeLearningBridge
from lrp.pipelines.round_completion import RoundCompletionPipeline


NOW = datetime(2026, 8, 8, 1, 0, tzinfo=timezone.utc)


def prediction_payload() -> dict[str, object]:
    return {
        "round": 1232,
        "generated_at_kst": "2026-08-08T20:30:00+09:00",
        "seed": 20260808,
        "params": {"temperature": 0.85},
        "sets": [
            {
                "id": "S1",
                "numbers": [3, 8, 14, 22, 35, 41],
                "score": 0.91,
                "risk_flags": [],
                "features": {"sum": 123},
            },
            {
                "id": "S2",
                "numbers": [4, 11, 19, 27, 34, 42],
                "score": 0.84,
                "risk_flags": [],
                "features": {"sum": 137},
            },
        ],
        "top5_practical": ["S1"],
        "metadata": {"statistics_version": "1.0.0"},
    }


def make_pipeline(tmp_path: Path) -> RoundCompletionPipeline:
    learning_root = tmp_path / "learning"
    profile_root = tmp_path / "profiles"

    repository = LearningRepository(learning_root / "learning.db")
    outcome_bridge = OutcomeBridge(
        repository=repository,
        model_name="lrp-v4.0.0",
    )

    persistence = PersistentLearningService(
        FileSnapshotRepository(learning_root)
    )
    learning_bridge = OutcomeLearningBridge(
        service=ReviewLearningService(
            PersistentLearningRunner(persistence)
        )
    )

    profile_service = ReviewProfileEvolutionService(
        EvolutionCoordinator(
            pipeline=AdaptiveEvolutionPipeline(),
            policy=AdaptiveWeightPolicy(),
            repository=SnapshotRepository(profile_root),
        )
    )

    return RoundCompletionPipeline(
        outcome_bridge=outcome_bridge,
        learning_bridge=learning_bridge,
        profile_service=profile_service,
    )


def test_round_completion_runs_full_flow(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path)

    result = pipeline.run(
        prediction_payload(),
        winning_numbers=(3, 8, 14, 22, 35, 41),
        bonus=9,
        policy="thompson",
        confidence=0.80,
        recorded_at_kst="2026-08-08T21:00:00+09:00",
        reviewed_at_kst="2026-08-08T21:01:00+09:00",
        generated_at_utc=NOW,
    )

    assert result.round_no == 1232
    assert result.outcome["created_predictions"] == 2
    assert result.outcome["reviews_created"] == 2
    assert result.feedback_count > 0
    assert result.learning_snapshot_id == "review-1232"
    assert result.final_context_version >= 1

    assert (tmp_path / "learning" / "review-1232.json").is_file()


def test_round_completion_result_is_serializable(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path)

    result = pipeline.run(
        prediction_payload(),
        winning_numbers=(3, 8, 14, 22, 35, 41),
        bonus=9,
        recorded_at_kst="2026-08-08T21:00:00+09:00",
        reviewed_at_kst="2026-08-08T21:01:00+09:00",
        generated_at_utc=NOW,
    )

    payload = result.as_dict()

    assert payload["round_no"] == 1232
    assert payload["learning"]["snapshot_id"] == "review-1232"
    assert payload["profile"]["snapshot_saved"] in (True, False)
