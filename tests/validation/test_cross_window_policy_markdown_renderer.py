from __future__ import annotations

from pathlib import Path

import pytest

from tools.validation.cross_window_policy_markdown_renderer import (
    CrossWindowPolicyMarkdownRenderer,
)


def make_report() -> dict[str, object]:
    weights = {
        "hot_weight": 0.30,
        "cold_weight": 0.17,
        "gap_weight": 0.17,
        "trend_weight": 0.14,
        "transition_weight": 0.12,
        "learning_weight": 0.05,
        "adaptive_weight": 0.05,
    }

    return {
        "schema_version": 1,
        "window_count": 2,
        "total_round_count": 200,
        "policy_count": 2,
        "windows": [
            {
                "start_round": 100,
                "end_round": 199,
                "round_count": 100,
                "path": "first",
                "ranking": [
                    {
                        "rank": 1,
                        "scenario": "floor",
                    }
                ],
            },
            {
                "start_round": 200,
                "end_round": 299,
                "round_count": 100,
                "path": "second",
                "ranking": [
                    {
                        "rank": 1,
                        "scenario": "baseline",
                    }
                ],
            },
        ],
        "policies": {
            "baseline": {
                "policy_name": "baseline",
                "window_count": 2,
                "total_round_count": 200,
                "best_hit_mean_delta": 0.01,
                "practical_hit_mean_delta": 0.02,
                "best_adaptive_wins": 20,
                "best_noop_wins": 18,
                "best_ties": 162,
                "practical_adaptive_wins": 22,
                "practical_noop_wins": 20,
                "practical_ties": 158,
                "changed_portfolio_round_count": 190,
                "first_place_count": 1,
                "average_probability_l1_delta": 0.05,
                "average_changed_set_count": 15.0,
                "mean_final_weights": weights,
                "windows": [],
            },
            "floor": {
                "policy_name": "floor",
                "window_count": 2,
                "total_round_count": 200,
                "best_hit_mean_delta": 0.04,
                "practical_hit_mean_delta": 0.03,
                "best_adaptive_wins": 25,
                "best_noop_wins": 17,
                "best_ties": 158,
                "practical_adaptive_wins": 24,
                "practical_noop_wins": 18,
                "practical_ties": 158,
                "changed_portfolio_round_count": 188,
                "first_place_count": 1,
                "average_probability_l1_delta": 0.04,
                "average_changed_set_count": 14.0,
                "mean_final_weights": weights,
                "windows": [],
            },
        },
        "ranking": [
            {
                "rank": 1,
                "policy_name": "floor",
                "practical_hit_mean_delta": 0.03,
                "best_hit_mean_delta": 0.04,
                "first_place_count": 1,
                "average_probability_l1_delta": 0.04,
            },
            {
                "rank": 2,
                "policy_name": "baseline",
                "practical_hit_mean_delta": 0.02,
                "best_hit_mean_delta": 0.01,
                "first_place_count": 1,
                "average_probability_l1_delta": 0.05,
            },
        ],
    }


def test_render_contains_overview() -> None:
    text = (
        CrossWindowPolicyMarkdownRenderer()
        .render(make_report())
    )

    assert "# Cross-Window Policy Report" in text
    assert "- Validation windows: 2" in text
    assert "- Total rounds: 200" in text
    assert "- Policies: 2" in text


def test_render_contains_ranking() -> None:
    text = (
        CrossWindowPolicyMarkdownRenderer()
        .render(make_report())
    )

    assert "| 1 | floor |" in text
    assert "| 2 | baseline |" in text


def test_render_contains_weights() -> None:
    text = (
        CrossWindowPolicyMarkdownRenderer()
        .render(make_report())
    )

    assert "### floor" in text
    assert "| learning_weight | 0.050000 |" in text
    assert "| adaptive_weight | 0.050000 |" in text


def test_render_contains_window_winners() -> None:
    text = (
        CrossWindowPolicyMarkdownRenderer()
        .render(make_report())
    )

    assert "| 100–199 | 100 | floor |" in text
    assert "| 200–299 | 100 | baseline |" in text


def test_write_markdown(
    tmp_path: Path,
) -> None:
    output = (
        CrossWindowPolicyMarkdownRenderer()
        .write(
            report=make_report(),
            output=(
                tmp_path
                / "reports"
                / "cross_window.md"
            ),
        )
    )

    assert output.is_file()
    assert output.read_text(
        encoding="utf-8"
    ).startswith(
        "# Cross-Window Policy Report"
    )


def test_invalid_report_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="report must be a mapping",
    ):
        CrossWindowPolicyMarkdownRenderer().render(
            object(),  # type: ignore[arg-type]
        )


def test_output_directory_is_rejected(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(
        IsADirectoryError,
    ):
        CrossWindowPolicyMarkdownRenderer().write(
            report=make_report(),
            output=output,
        )


def test_render_contains_weight_trends() -> None:
    report = make_report()

    report["weight_trends"] = {
        "policies": {
            "floor": {
                "weights": {
                    field: {
                        "direction": "stable",
                        "first": 0.05,
                        "last": 0.05,
                        "net_change": 0.0,
                    }
                    for field in (
                        "hot_weight",
                        "cold_weight",
                        "gap_weight",
                        "trend_weight",
                        "transition_weight",
                        "learning_weight",
                        "adaptive_weight",
                    )
                }
            }
        }
    }

    text = (
        CrossWindowPolicyMarkdownRenderer()
        .render(report)
    )

    assert "## Weight Trends" in text
    assert "### floor" in text
    assert (
        "| learning_weight | stable | "
        "0.050000 | 0.050000 | 0.000000 |"
        in text
    )
