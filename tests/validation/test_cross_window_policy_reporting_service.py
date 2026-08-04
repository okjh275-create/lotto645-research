from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validation.cross_window_policy_reporting_service import (
    CrossWindowPolicyReportingResult,
    CrossWindowPolicyReportingService,
)


def result(
    name: str,
    *,
    round_count: int,
    best: float,
    practical: float,
) -> dict[str, object]:
    return {
        "scenario": {
            "name": name,
            "adjustment_scale": 0.25,
            "minimum_weight": 0.03,
        },
        "effectiveness": {
            "round_count": round_count,
            "best_hit_mean_delta": best,
            "practical_hit_mean_delta": practical,
            "best_adaptive_wins": 10,
            "best_noop_wins": 8,
            "best_ties": round_count - 18,
            "practical_adaptive_wins": 12,
            "practical_noop_wins": 8,
            "practical_ties": round_count - 20,
            "average_probability_l1_delta": 0.05,
            "average_changed_set_count": 15.0,
            "changed_portfolio_round_count": round_count,
        },
        "final_profile": {
            "weights": {
                "hot_weight": 0.30,
                "cold_weight": 0.17,
                "gap_weight": 0.17,
                "trend_weight": 0.14,
                "transition_weight": 0.12,
                "learning_weight": 0.05,
                "adaptive_weight": 0.05,
            }
        },
    }


def write_window(
    path: Path,
    *,
    start_round: int,
    end_round: int,
) -> None:
    round_count = (
        end_round - start_round + 1
    )

    payload = {
        "config": {
            "start_round": start_round,
            "end_round": end_round,
        },
        "scenario_count": 2,
        "results": [
            result(
                "baseline",
                round_count=round_count,
                best=-0.05,
                practical=0.01,
            ),
            result(
                "floor",
                round_count=round_count,
                best=0.05,
                practical=0.03,
            ),
        ],
        "ranking": [
            {
                "rank": 1,
                "scenario": "floor",
            },
            {
                "rank": 2,
                "scenario": "baseline",
            },
        ],
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_generate_from_paths_writes_both_formats(
    tmp_path: Path,
) -> None:
    first = (
        tmp_path
        / "source"
        / "first"
        / "policy_comparison.json"
    )
    second = (
        tmp_path
        / "source"
        / "second"
        / "policy_comparison.json"
    )

    write_window(
        first,
        start_round=100,
        end_round=199,
    )
    write_window(
        second,
        start_round=200,
        end_round=299,
    )

    result_value = (
        CrossWindowPolicyReportingService()
        .generate_from_paths(
            paths=(first, second),
            output_root=tmp_path / "reports",
        )
    )

    assert isinstance(
        result_value,
        CrossWindowPolicyReportingResult,
    )
    assert result_value.json_path.is_file()
    assert result_value.markdown_path.is_file()
    assert result_value.report[
        "window_count"
    ] == 2


def test_discover_and_generate(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"

    write_window(
        source_root
        / "first"
        / "policy_comparison.json",
        start_round=100,
        end_round=199,
    )

    result_value = (
        CrossWindowPolicyReportingService()
        .discover_and_generate(
            source_root=source_root,
            output_root=tmp_path / "reports",
            stem="project_h",
        )
    )

    assert result_value.json_path.name == (
        "project_h.json"
    )
    assert result_value.markdown_path.name == (
        "project_h.md"
    )


def test_result_serialization(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"

    write_window(
        source_root
        / "first"
        / "policy_comparison.json",
        start_round=100,
        end_round=199,
    )

    result_value = (
        CrossWindowPolicyReportingService()
        .discover_and_generate(
            source_root=source_root,
            output_root=tmp_path / "reports",
        )
    )

    payload = result_value.as_dict()

    assert payload["report"][
        "window_count"
    ] == 1
    assert payload["json_path"].endswith(
        "cross_window_policy_report.json"
    )
    assert payload["markdown_path"].endswith(
        "cross_window_policy_report.md"
    )


@pytest.mark.parametrize(
    "stem",
    [
        "",
        " ",
        ".",
        "..",
        "nested/report",
        r"nested\report",
    ],
)
def test_invalid_stem_is_rejected(
    tmp_path: Path,
    stem: str,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()

    with pytest.raises(
        ValueError,
    ):
        CrossWindowPolicyReportingService(
        ).discover_and_generate(
            source_root=source_root,
            output_root=tmp_path / "reports",
            stem=stem,
        )


def test_non_string_stem_is_rejected(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()

    with pytest.raises(
        TypeError,
        match="stem must be a string",
    ):
        CrossWindowPolicyReportingService(
        ).discover_and_generate(
            source_root=source_root,
            output_root=tmp_path / "reports",
            stem=123,  # type: ignore[arg-type]
        )
