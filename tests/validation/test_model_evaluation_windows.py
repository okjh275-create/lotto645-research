from __future__ import annotations

import pytest

from lrp.evaluation import EvaluationWindow

from tools.validation.model_evaluation_windows import (
    build_non_overlapping_windows,
)


def test_builds_default_three_window_suite() -> None:
    windows = build_non_overlapping_windows(
        end_round=1231,
    )

    assert windows == (
        EvaluationWindow(
            name="recent",
            start_round=1212,
            end_round=1231,
        ),
        EvaluationWindow(
            name="prior",
            start_round=1192,
            end_round=1211,
        ),
        EvaluationWindow(
            name="older",
            start_round=1172,
            end_round=1191,
        ),
    )


def test_windows_have_equal_sizes() -> None:
    windows = build_non_overlapping_windows(
        end_round=1231,
        window_size=20,
    )

    assert tuple(
        window.round_count
        for window in windows
    ) == (
        20,
        20,
        20,
    )


def test_windows_do_not_overlap() -> None:
    windows = build_non_overlapping_windows(
        end_round=1231,
        window_size=20,
    )

    occupied: set[int] = set()

    for window in windows:
        current = set(
            range(
                window.start_round,
                window.end_round + 1,
            )
        )

        assert occupied.isdisjoint(current)

        occupied.update(current)

    assert len(occupied) == 60


def test_supports_custom_window_count() -> None:
    windows = build_non_overlapping_windows(
        end_round=1231,
        window_size=10,
        names=(
            "recent",
            "prior",
            "older",
            "oldest",
        ),
    )

    assert tuple(
        (item.name, item.start_round, item.end_round)
        for item in windows
    ) == (
        ("recent", 1222, 1231),
        ("prior", 1212, 1221),
        ("older", 1202, 1211),
        ("oldest", 1192, 1201),
    )


def test_rejects_empty_names() -> None:
    with pytest.raises(
        ValueError,
        match="names must not be empty",
    ):
        build_non_overlapping_windows(
            end_round=1231,
            names=(),
        )


def test_rejects_duplicate_names() -> None:
    with pytest.raises(
        ValueError,
        match="window names must be unique",
    ):
        build_non_overlapping_windows(
            end_round=1231,
            names=(
                "recent",
                "recent",
            ),
        )


@pytest.mark.parametrize(
    "window_size",
    (
        0,
        -1,
    ),
)
def test_rejects_invalid_window_size(
    window_size: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="window_size must be positive",
    ):
        build_non_overlapping_windows(
            end_round=1231,
            window_size=window_size,
        )


def test_rejects_insufficient_round_range() -> None:
    with pytest.raises(
        ValueError,
        match="insufficient round range",
    ):
        build_non_overlapping_windows(
            end_round=40,
            window_size=20,
            names=(
                "recent",
                "prior",
                "older",
            ),
        )


def test_is_deterministic() -> None:
    first = build_non_overlapping_windows(
        end_round=1231,
    )

    second = build_non_overlapping_windows(
        end_round=1231,
    )

    assert first == second
