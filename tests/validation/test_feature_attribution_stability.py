from __future__ import annotations

from tools.validation.feature_attribution_stability import (
    analyze_blocks,
    pearson,
)


def test_pearson_perfect_positive() -> None:
    assert pearson(
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
    ) == 1.0


def test_pearson_constant_returns_zero() -> None:
    assert pearson(
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 3.0],
    ) == 0.0


def test_block_analysis() -> None:
    rows = []

    for index in range(8):
        signal = (
            1.0
            if index % 2 == 0
            else -1.0
        )

        rows.append(
            {
                "round_no": 100 + index,
                "hot": signal,
                "cold": signal,
                "gap": signal,
                "trend": signal,
                "transition": signal,
                "best_hit_delta": signal,
                "practical_hit_delta": signal,
            }
        )

    report = analyze_blocks(
        rows,
        block_size=4,
    )

    assert report["block_count"] == 2
    assert report[
        "stability"
    ]["hot"]["best_hit_delta"][
        "consistent_direction"
    ] is True
