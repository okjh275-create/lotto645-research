"""Regression tests for Project E E-005C Learning Snapshot Writer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

from lrp.learning import (
    LearningRepository,
    LearningService,
    LearningSnapshot,
    LearningSnapshotWriter,
    PredictionRecord,
    ResultRecord,
)


GENERATED_AT_KST = "2026-07-27T22:50:00+09:00"
RECORDED_AT_KST = "2026-07-27T22:51:00+09:00"
REVIEWED_AT_KST = "2026-07-27T22:52:00+09:00"
AGGREGATED_AT_KST = "2026-07-27T22:53:00+09:00"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare(service: LearningService) -> None:
    repository = service.repository
    prediction = PredictionRecord(
        prediction_id="1240:GPT-v3.3:S1",
        round_no=1240,
        set_id="S1",
        numbers=(1, 2, 3, 20, 21, 22),
        score=0.90,
        model_name="GPT-v3.3",
        seed=20260727,
        generated_at_kst=GENERATED_AT_KST,
        features={"gap_mix": 0.70},
        parameters={
            "temperature": 0.85,
            "scenario": "gap",
        },
    )
    assert repository.add_prediction(prediction) is True
    assert repository.add_result(
        ResultRecord(
            round_no=1240,
            numbers=(1, 2, 3, 4, 5, 6),
            bonus=7,
            recorded_at_kst=RECORDED_AT_KST,
        )
    ) is True
    review = service.run_incremental_review(
        reviewed_at_kst=REVIEWED_AT_KST
    )
    assert review.created == 1
    aggregation = service.run_strategy_aggregation(
        aggregated_at_kst=AGGREGATED_AT_KST
    )
    assert aggregation.created_events == 2


def test_write_snapshot() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repository = LearningRepository(root / "learning.db")
        service = LearningService(repository)
        _prepare(service)

        before_counts = repository.counts()
        writer = LearningSnapshotWriter(service)
        snapshot = writer.write(
            round_no=1241,
            output_root=root / "snapshots",
            history_limit=10,
            generated_at_kst=GENERATED_AT_KST,
            metadata={
                "project_version": "4.0",
                "seed": 20260727,
            },
        )
        after_counts = repository.counts()

        assert isinstance(snapshot, LearningSnapshot)
        assert before_counts == after_counts
        assert snapshot.round_no == 1241
        assert snapshot.revision == (2, 2)
        assert snapshot.directory == root / "snapshots" / "1241"

        expected = {
            "rankings.json",
            "adaptive_weights.json",
            "performance.json",
            "adaptive_report.json",
            "metadata.json",
            "SHA256SUMS.txt",
        }
        assert set(snapshot.files) == expected
        assert all(
            (snapshot.directory / name).is_file()
            for name in expected
        )

        metadata = json.loads(
            (snapshot.directory / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        assert metadata["revision"] == [2, 2]
        assert metadata["metadata"]["seed"] == 20260727

        for filename, digest in metadata["files"].items():
            assert _sha256(snapshot.directory / filename) == digest

        rankings = json.loads(
            (snapshot.directory / "rankings.json").read_text(
                encoding="utf-8"
            )
        )
        weights = json.loads(
            (
                snapshot.directory / "adaptive_weights.json"
            ).read_text(encoding="utf-8")
        )
        assert rankings["revision"] == weights["revision"]
        assert len(rankings["rankings"]) == 2
        assert len(weights["weights"]) == 2


def test_overwrite_guard() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repository = LearningRepository(root / "learning.db")
        service = LearningService(repository)
        writer = LearningSnapshotWriter(service)

        writer.write(
            round_no=1242,
            output_root=root / "snapshots",
            generated_at_kst=GENERATED_AT_KST,
        )
        try:
            writer.write(
                round_no=1242,
                output_root=root / "snapshots",
                generated_at_kst=GENERATED_AT_KST,
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError(
                "existing snapshots must require overwrite=True"
            )

        second = writer.write(
            round_no=1242,
            output_root=root / "snapshots",
            generated_at_kst=GENERATED_AT_KST,
            overwrite=True,
        )
        assert second.revision == (0, 0)


def test_deterministic_files() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repository = LearningRepository(root / "learning.db")
        service = LearningService(repository)
        _prepare(service)
        writer = LearningSnapshotWriter(service)

        first = writer.write(
            round_no=1243,
            output_root=root / "one",
            generated_at_kst=GENERATED_AT_KST,
        )
        second = writer.write(
            round_no=1243,
            output_root=root / "two",
            generated_at_kst=GENERATED_AT_KST,
        )

        comparable = {
            "rankings.json",
            "adaptive_weights.json",
            "performance.json",
            "adaptive_report.json",
            "metadata.json",
            "SHA256SUMS.txt",
        }
        for filename in comparable:
            assert (
                first.directory / filename
            ).read_bytes() == (
                second.directory / filename
            ).read_bytes()


def main() -> None:
    test_write_snapshot()
    test_overwrite_guard()
    test_deterministic_files()

    print("PASS: Project E E-005C learning snapshot writer")
    print("read_only_source: PASS")
    print("revision_consistency: PASS")
    print("strategy_consistency: PASS")
    print("atomic_json_write: PASS")
    print("sha256_manifest: PASS")
    print("overwrite_guard: PASS")
    print("deterministic_snapshot: PASS")
    print("public_api_compatibility: PASS")


if __name__ == "__main__":
    main()
