from __future__ import annotations

from pathlib import Path

import pytest

from lrp.learning import LearningRepository, ResultRecord


def make_result(recorded_at_kst: str) -> ResultRecord:
    return ResultRecord(
        round_no=1232,
        numbers=(3, 8, 14, 22, 35, 41),
        bonus=9,
        recorded_at_kst=recorded_at_kst,
    )


def test_same_result_with_different_recorded_time_is_idempotent(
    tmp_path: Path,
) -> None:
    repository = LearningRepository(
        tmp_path / "learning.db"
    )

    assert repository.add_result(
        make_result("2026-08-08T21:00:00+09:00")
    ) is True

    assert repository.add_result(
        make_result("2026-08-08T21:01:00+09:00")
    ) is False

    stored = repository.get_result(1232)

    assert stored is not None
    assert stored.recorded_at_kst == (
        "2026-08-08T21:00:00+09:00"
    )


def test_conflicting_numbers_still_raise(
    tmp_path: Path,
) -> None:
    repository = LearningRepository(
        tmp_path / "learning.db"
    )

    assert repository.add_result(
        make_result("2026-08-08T21:00:00+09:00")
    ) is True

    conflicting = ResultRecord(
        round_no=1232,
        numbers=(1, 2, 3, 4, 5, 6),
        bonus=7,
        recorded_at_kst="2026-08-08T21:01:00+09:00",
    )

    with pytest.raises(
        ValueError,
        match="different content",
    ):
        repository.add_result(conflicting)
