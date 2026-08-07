from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from lrp.evolution.contracts.learning_context import LearningContext
from lrp.evolution.repositories.file_snapshot_repository import FileSnapshotRepository
from lrp.evolution.services.persistent_learning_runner import PersistentLearningRunner
from lrp.evolution.services.persistent_learning_service import PersistentLearningService
from lrp.evolution.services.review_learning_service import ReviewLearningService
from lrp.evolution.services.snapshot_factory import SnapshotFactory
from lrp.outcomes import (
    OutcomeLearningBridge,
    OutcomeLearningBridgeResult,
)


NOW = datetime(
    2026,
    8,
    8,
    1,
    0,
    tzinfo=timezone.utc,
)


def make_service(root: Path) -> ReviewLearningService:
    persistence = PersistentLearningService(
        FileSnapshotRepository(root),
        snapshot_factory=SnapshotFactory(
            clock=lambda: NOW
        ),
    )
    runner = PersistentLearningRunner(persistence)
    return ReviewLearningService(runner)


def make_context() -> LearningContext:
    return LearningContext(
        cycle_id="cycle-1232",
        round_no=1232,
    )


def make_review() -> dict[str, object]:
    return {
        "round": 1232,
        "summary": {
            "set_count": 20,
            "best_main_hits": 4,
            "practical_best_hits": 3,
        },
    }


def test_learns_review_into_snapshot(tmp_path: Path) -> None:
    service = make_service(tmp_path / "learning")
    bridge = OutcomeLearningBridge(service=service)

    result = bridge.learn(
        context=make_context(),
        review_payload=make_review(),
        snapshot_id="review-1232",
        policy="thompson",
        metadata={"round": 1232},
    )

    assert isinstance(result, OutcomeLearningBridgeResult)
    assert result.round_no == 1232
    assert result.snapshot_id == "review-1232"
    assert result.feedback_count > 0
    assert result.policy == "thompson"
    assert (
        tmp_path
        / "learning"
        / "review-1232.json"
    ).is_file()


def test_exposes_final_context(tmp_path: Path) -> None:
    bridge = OutcomeLearningBridge(
        service=make_service(tmp_path / "learning")
    )

    result = bridge.learn(
        context=make_context(),
        review_payload=make_review(),
        snapshot_id="review-1232",
        policy="thompson",
    )

    final_context = result.learning.final_context

    assert final_context.round_no == 1232
    assert final_context.metadata["review_count"] == 1
    assert final_context.metadata["review_set_count"] == 20


def test_result_as_dict(tmp_path: Path) -> None:
    bridge = OutcomeLearningBridge(
        service=make_service(tmp_path / "learning")
    )

    result = bridge.learn(
        context=make_context(),
        review_payload=make_review(),
        snapshot_id="review-1232",
        policy="thompson",
    )

    assert result.as_dict() == {
        "round_no": 1232,
        "snapshot_id": "review-1232",
        "feedback_count": result.feedback_count,
        "policy": "thompson",
    }


def test_rejects_invalid_service() -> None:
    with pytest.raises(TypeError, match="ReviewLearningService"):
        OutcomeLearningBridge(service=object())


def test_rejects_invalid_context(tmp_path: Path) -> None:
    bridge = OutcomeLearningBridge(
        service=make_service(tmp_path / "learning")
    )

    with pytest.raises(TypeError, match="LearningContext"):
        bridge.learn(
            context=object(),
            review_payload=make_review(),
            snapshot_id="review-1232",
        )


def test_public_exports() -> None:
    import lrp.outcomes as outcomes

    assert "OutcomeLearningBridge" in outcomes.__all__
    assert "OutcomeLearningBridgeResult" in outcomes.__all__
