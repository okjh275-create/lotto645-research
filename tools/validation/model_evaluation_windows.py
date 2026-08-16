"""Historical evaluation window construction for Project M."""

from __future__ import annotations

from collections.abc import Iterable

from lrp.evaluation import EvaluationWindow


DEFAULT_WINDOW_NAMES = (
    "recent",
    "prior",
    "older",
)


def build_non_overlapping_windows(
    *,
    end_round: int,
    window_size: int = 20,
    names: Iterable[str] = DEFAULT_WINDOW_NAMES,
) -> tuple[EvaluationWindow, ...]:
    """Build equal-sized adjacent historical windows newest to oldest."""

    normalized_names = tuple(names)

    if not normalized_names:
        raise ValueError(
            "names must not be empty"
        )

    if any(
        not isinstance(name, str)
        or not name.strip()
        for name in normalized_names
    ):
        raise ValueError(
            "window names must be non-empty strings"
        )

    if len(normalized_names) != len(
        set(normalized_names)
    ):
        raise ValueError(
            "window names must be unique"
        )

    if not isinstance(window_size, int):
        raise TypeError(
            "window_size must be int"
        )

    if window_size <= 0:
        raise ValueError(
            "window_size must be positive"
        )

    if not isinstance(end_round, int):
        raise TypeError(
            "end_round must be int"
        )

    required_rounds = (
        window_size
        * len(normalized_names)
    )

    first_round = (
        end_round
        - required_rounds
        + 1
    )

    if first_round < 1:
        raise ValueError(
            "insufficient round range"
        )

    windows: list[
        EvaluationWindow
    ] = []

    current_end = end_round

    for name in normalized_names:
        current_start = (
            current_end
            - window_size
            + 1
        )

        windows.append(
            EvaluationWindow(
                name=name,
                start_round=current_start,
                end_round=current_end,
            )
        )

        current_end = (
            current_start - 1
        )

    return tuple(windows)
