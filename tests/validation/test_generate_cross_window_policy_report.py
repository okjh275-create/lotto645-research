from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validation.generate_cross_window_policy_report import (
    build_parser,
    run,
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


def test_parser_defaults() -> None:
    args = build_parser().parse_args(
        [
            "--source",
            "source",
            "--output",
            "output",
        ]
    )

    assert args.source == Path("source")
    assert args.output == Path("output")
    assert args.stem == (
        "cross_window_policy_report"
    )


def test_run_generates_reports(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "reports"

    write_window(
        source_root
        / "first"
        / "policy_comparison.json",
        start_round=100,
        end_round=199,
    )
    write_window(
        source_root
        / "second"
        / "policy_comparison.json",
        start_round=200,
        end_round=299,
    )

    exit_code = run(
        [
            "--source",
            str(source_root),
            "--output",
            str(output_root),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "PASS"
    assert payload["window_count"] == 2
    assert payload["total_round_count"] == 200
    assert payload["policy_count"] == 2
    assert payload["winner"] == "floor"

    assert (
        output_root
        / "cross_window_policy_report.json"
    ).is_file()

    assert (
        output_root
        / "cross_window_policy_report.md"
    ).is_file()


def test_custom_stem(
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

    exit_code = run(
        [
            "--source",
            str(source_root),
            "--output",
            str(tmp_path / "reports"),
            "--stem",
            "project_h_policy",
        ]
    )

    assert exit_code == 0

    assert (
        tmp_path
        / "reports"
        / "project_h_policy.json"
    ).is_file()


def test_missing_source_exits_with_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        SystemExit,
    ) as error:
        run(
            [
                "--source",
                str(tmp_path / "missing"),
                "--output",
                str(tmp_path / "reports"),
            ]
        )

    assert error.value.code == 2


def test_no_policy_reports_exits_with_error(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()

    with pytest.raises(
        SystemExit,
    ) as error:
        run(
            [
                "--source",
                str(source_root),
                "--output",
                str(tmp_path / "reports"),
            ]
        )

    assert error.value.code == 2


def test_invalid_stem_exits_with_error(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()

    with pytest.raises(
        SystemExit,
    ) as error:
        run(
            [
                "--source",
                str(source_root),
                "--output",
                str(tmp_path / "reports"),
                "--stem",
                "nested/report",
            ]
        )

    assert error.value.code == 2
