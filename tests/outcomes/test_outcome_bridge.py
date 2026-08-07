from __future__ import annotations

from pathlib import Path
import pytest

from lrp.learning import LearningRepository
from lrp.outcomes import OutcomeBridge, OutcomeBridgeResult


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


def make_bridge(tmp_path: Path):
    repository = LearningRepository(tmp_path / "learning.db")
    bridge = OutcomeBridge(
        repository=repository,
        model_name="lrp-v4.0.0",
    )
    return bridge, repository


def test_processes_predictions_result_and_reviews(tmp_path: Path) -> None:
    bridge, repository = make_bridge(tmp_path)

    result = bridge.process(
        prediction_payload(),
        winning_numbers=(3, 8, 14, 22, 35, 41),
        bonus=9,
        recorded_at_kst="2026-08-08T21:00:00+09:00",
        reviewed_at_kst="2026-08-08T21:01:00+09:00",
    )

    assert isinstance(result, OutcomeBridgeResult)
    assert result.round_no == 1232
    assert result.imported_predictions == 2
    assert result.created_predictions == 2
    assert result.existing_predictions == 0
    assert result.result_created is True
    assert result.reviews_scanned == 2
    assert result.reviews_created == 2
    assert result.reviews_skipped == 0

    counts = repository.counts()
    assert counts["predictions"] == 2
    assert counts["results"] == 1
    assert counts["reviews"] == 2
    assert repository.pending_prediction_ids() == ()


def test_process_is_idempotent(tmp_path: Path) -> None:
    bridge, repository = make_bridge(tmp_path)
    kwargs = {
        "winning_numbers": (3, 8, 14, 22, 35, 41),
        "bonus": 9,
        "recorded_at_kst": "2026-08-08T21:00:00+09:00",
        "reviewed_at_kst": "2026-08-08T21:01:00+09:00",
    }

    first = bridge.process(prediction_payload(), **kwargs)
    second = bridge.process(prediction_payload(), **kwargs)

    assert first.created_predictions == 2
    assert first.result_created is True
    assert first.reviews_created == 2
    assert second.created_predictions == 0
    assert second.existing_predictions == 2
    assert second.result_created is False
    assert second.reviews_scanned == 0
    assert second.reviews_created == 0

    counts = repository.counts()
    assert counts["predictions"] == 2
    assert counts["results"] == 1
    assert counts["reviews"] == 2


def test_rejects_conflicting_result(tmp_path: Path) -> None:
    bridge, repository = make_bridge(tmp_path)

    bridge.process(
        prediction_payload(),
        winning_numbers=(3, 8, 14, 22, 35, 41),
        bonus=9,
        recorded_at_kst="2026-08-08T21:00:00+09:00",
    )

    with pytest.raises(ValueError):
        bridge.process(
            prediction_payload(),
            winning_numbers=(1, 2, 3, 4, 5, 6),
            bonus=7,
            recorded_at_kst="2026-08-08T21:00:00+09:00",
        )

    counts = repository.counts()
    assert counts["results"] == 1
    assert counts["reviews"] == 2


def test_rejects_invalid_bonus(tmp_path: Path) -> None:
    bridge, repository = make_bridge(tmp_path)

    with pytest.raises(ValueError, match="bonus"):
        bridge.process(
            prediction_payload(),
            winning_numbers=(3, 8, 14, 22, 35, 41),
            bonus=41,
            recorded_at_kst="2026-08-08T21:00:00+09:00",
        )

    counts = repository.counts()
    assert counts["results"] == 0
    assert counts["reviews"] == 0


def test_result_as_dict(tmp_path: Path) -> None:
    bridge, _ = make_bridge(tmp_path)

    result = bridge.process(
        prediction_payload(),
        winning_numbers=(3, 8, 14, 22, 35, 41),
        bonus=9,
        recorded_at_kst="2026-08-08T21:00:00+09:00",
    )

    assert result.as_dict() == {
        "round_no": 1232,
        "imported_predictions": 2,
        "created_predictions": 2,
        "existing_predictions": 0,
        "result_created": True,
        "reviews_scanned": 2,
        "reviews_created": 2,
        "reviews_skipped": 0,
    }


def test_public_exports() -> None:
    import lrp.outcomes as outcomes

    assert "OutcomeBridge" in outcomes.__all__
    assert "OutcomeBridgeResult" in outcomes.__all__
    assert "OutcomeImporter" in outcomes.__all__
    assert "OutcomeImportError" in outcomes.__all__
