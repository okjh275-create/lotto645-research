"""Regression tests for Project E E-005D Learning Coordinator."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from lrp.learning import (
    LearningCoordinator,
    LearningCoordinatorConfig,
    LearningCoordinatorResult,
    LearningRepository,
    LearningService,
    LearningSnapshotWriter,
    PredictionRecord,
    ResultRecord,
)


GENERATED_AT_KST = "2026-07-29T22:30:00+09:00"
RECORDED_AT_KST = "2026-07-29T22:31:00+09:00"
REVIEWED_AT_KST = "2026-07-29T22:32:00+09:00"
AGGREGATED_AT_KST = "2026-07-29T22:33:00+09:00"


def _prepare_pending_learning(
    service: LearningService,
    *,
    round_no: int,
) -> None:
    """Create one prediction and its matching draw result."""

    repository = service.repository

    prediction = PredictionRecord(
        prediction_id=f"{round_no}:GPT-v3.3:S1",
        round_no=round_no,
        set_id="S1",
        numbers=(1, 2, 3, 20, 21, 22),
        score=0.90,
        model_name="GPT-v3.3",
        seed=20260729,
        generated_at_kst=GENERATED_AT_KST,
        features={
            "gap_mix": 0.70,
            "pair_affinity": 0.65,
        },
        parameters={
            "temperature": 0.85,
            "scenario": "gap",
        },
    )

    result = ResultRecord(
        round_no=round_no,
        numbers=(1, 2, 3, 4, 5, 6),
        bonus=7,
        recorded_at_kst=RECORDED_AT_KST,
    )

    assert repository.add_prediction(prediction) is True
    assert repository.add_result(result) is True


def test_coordinated_learning_run() -> None:
    """Review, aggregate, and persist one learning snapshot."""

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repository = LearningRepository(
            root / "learning.db"
        )
        service = LearningService(repository)
        writer = LearningSnapshotWriter(service)
        coordinator = LearningCoordinator(
            service,
            writer,
        )

        _prepare_pending_learning(
            service,
            round_no=1244,
        )

        result = coordinator.run(
            round_no=1244,
            output_root=root / "snapshots",
            config=LearningCoordinatorConfig(
                strategy_type="MODEL",
                history_limit=10,
                review_limit=100,
                aggregation_limit=100,
            ),
            reviewed_at_kst=REVIEWED_AT_KST,
            aggregated_at_kst=AGGREGATED_AT_KST,
            generated_at_kst=GENERATED_AT_KST,
            metadata={
                "project_version": "4.0",
                "seed": 20260729,
            },
        )

        assert isinstance(
            result,
            LearningCoordinatorResult,
        )
        assert result.round_no == 1244

        assert result.review_summary.scanned == 1
        assert result.review_summary.created == 1
        assert result.review_summary.skipped == 0

        assert (
            result.aggregation_summary.created_events
            == 2
        )

        assert result.revision == (2, 2)
        assert result.directory == (
            root / "snapshots" / "1244"
        )
        assert result.directory.is_dir()
        assert result.elapsed_seconds >= 0.0

        assert result.metadata["coordinator"] == "E-005D"
        assert result.metadata["snapshot_written"] is True
        assert result.metadata["seed"] == 20260729
        assert (
            result.metadata["config"]["strategy_type"]
            == "model"
        )

        expected_files = {
            "rankings.json",
            "adaptive_weights.json",
            "performance.json",
            "adaptive_report.json",
            "metadata.json",
            "SHA256SUMS.txt",
        }

        assert set(result.snapshot.files) == expected_files
        assert all(
            (result.directory / filename).is_file()
            for filename in expected_files
        )

        snapshot_metadata = json.loads(
            (
                result.directory / "metadata.json"
            ).read_text(encoding="utf-8")
        )

        embedded = snapshot_metadata["metadata"]

        assert embedded["coordinator"] == "E-005D"
        assert embedded["project_version"] == "4.0"
        assert embedded["seed"] == 20260729
        assert (
            embedded["coordinator_config"][
                "strategy_type"
            ]
            == "model"
        )
        assert (
            embedded["review_summary"]["created"]
            == 1
        )
        assert (
            embedded["aggregation_summary"][
                "created_events"
            ]
            == 2
        )

        payload = result.as_dict()

        assert payload["round_no"] == 1244
        assert payload["revision"] == [2, 2]
        assert payload["review"]["created"] == 1
        assert (
            payload["aggregation"]["created_events"]
            == 2
        )


def test_coordinator_is_incremental() -> None:
    """A repeated run must not recreate completed learning."""

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repository = LearningRepository(
            root / "learning.db"
        )
        service = LearningService(repository)
        writer = LearningSnapshotWriter(service)
        coordinator = LearningCoordinator(
            service,
            writer,
        )

        _prepare_pending_learning(
            service,
            round_no=1245,
        )

        first = coordinator.run(
            round_no=1245,
            output_root=root / "first",
            reviewed_at_kst=REVIEWED_AT_KST,
            aggregated_at_kst=AGGREGATED_AT_KST,
            generated_at_kst=GENERATED_AT_KST,
        )

        second = coordinator.run(
            round_no=1245,
            output_root=root / "second",
            reviewed_at_kst=REVIEWED_AT_KST,
            aggregated_at_kst=AGGREGATED_AT_KST,
            generated_at_kst=GENERATED_AT_KST,
        )

        assert first.review_summary.created == 1
        assert (
            first.aggregation_summary.created_events
            == 2
        )

        assert second.review_summary.scanned == 0
        assert second.review_summary.created == 0
        assert (
            second.aggregation_summary.created_events
            == 0
        )

        assert first.revision == (2, 2)
        assert second.revision == (2, 2)


def test_coordinator_overwrite_guard() -> None:
    """Existing snapshots require explicit overwrite."""

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repository = LearningRepository(
            root / "learning.db"
        )
        service = LearningService(repository)
        writer = LearningSnapshotWriter(service)
        coordinator = LearningCoordinator(
            service,
            writer,
        )

        coordinator.run(
            round_no=1246,
            output_root=root / "snapshots",
            generated_at_kst=GENERATED_AT_KST,
        )

        try:
            coordinator.run(
                round_no=1246,
                output_root=root / "snapshots",
                generated_at_kst=GENERATED_AT_KST,
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError(
                "existing snapshots must require "
                "overwrite_snapshot=True"
            )

        overwritten = coordinator.run(
            round_no=1246,
            output_root=root / "snapshots",
            config=LearningCoordinatorConfig(
                overwrite_snapshot=True
            ),
            generated_at_kst=GENERATED_AT_KST,
        )

        assert overwritten.revision == (0, 0)
        assert overwritten.directory.is_dir()


def test_coordinator_config_validation() -> None:
    """Configuration values must be normalized and validated."""

    config = LearningCoordinatorConfig(
        strategy_type="  MODEL  ",
        history_limit=25,
        review_limit=10,
        aggregation_limit=20,
        overwrite_snapshot=True,
    )

    assert config.strategy_type == "model"
    assert config.as_dict() == {
        "strategy_type": "model",
        "history_limit": 25,
        "review_limit": 10,
        "aggregation_limit": 20,
        "overwrite_snapshot": True,
    }

    invalid_values = (
        {"history_limit": 0},
        {"review_limit": 0},
        {"aggregation_limit": -1},
        {"strategy_type": "   "},
        {"overwrite_snapshot": 1},
    )

    for values in invalid_values:
        try:
            LearningCoordinatorConfig(**values)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"invalid configuration accepted: {values}"
            )


def main() -> None:
    test_coordinated_learning_run()
    test_coordinator_is_incremental()
    test_coordinator_overwrite_guard()
    test_coordinator_config_validation()

    print(
        "PASS: Project E E-005D "
        "learning coordinator"
    )
    print("workflow_order: PASS")
    print("incremental_review: PASS")
    print("strategy_aggregation: PASS")
    print("snapshot_persistence: PASS")
    print("metadata_propagation: PASS")
    print("overwrite_guard: PASS")
    print("configuration_validation: PASS")
    print("public_api_compatibility: PASS")


if __name__ == "__main__":
    main()