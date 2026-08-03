from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validation.feature_attribution_effectiveness import (
    analyze_feature_attribution,
    write_feature_attribution_report,
)


def write_rounds(
    path: Path,
) -> None:
    rows = [
        {
            "round_no": 100,
            "best_hit_delta": 1,
            "practical_hit_delta": 1,
            "probability_l1_delta": 0.1,
            "changed_set_count": 10,
        },
        {
            "round_no": 101,
            "best_hit_delta": -1,
            "practical_hit_delta": -1,
            "probability_l1_delta": 0.2,
            "changed_set_count": 20,
        },
        {
            "round_no": 102,
            "best_hit_delta": 1,
            "practical_hit_delta": 1,
            "probability_l1_delta": 0.3,
            "changed_set_count": 30,
        },
    ]

    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def write_snapshot(
    root: Path,
    *,
    round_no: int,
    signal: float,
) -> None:
    payload = {
        "round_no": round_no,
        "metadata": {
            "feature_signal_hot": signal,
            "feature_signal_cold": signal,
            "feature_signal_gap": signal,
            "feature_signal_trend": signal,
            "feature_signal_transition": signal,
        },
    }

    (root / f"review-{round_no}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def make_inputs(
    tmp_path: Path,
) -> tuple[Path, Path]:
    rounds_path = (
        tmp_path / "replay_rounds.jsonl"
    )
    learning_root = (
        tmp_path / "learning"
    )
    learning_root.mkdir()

    write_rounds(rounds_path)

    write_snapshot(
        learning_root,
        round_no=100,
        signal=1.0,
    )
    write_snapshot(
        learning_root,
        round_no=101,
        signal=-1.0,
    )
    write_snapshot(
        learning_root,
        round_no=102,
        signal=1.0,
    )

    return rounds_path, learning_root


def test_analysis_joins_all_rounds(
    tmp_path: Path,
) -> None:
    rounds_path, learning_root = (
        make_inputs(tmp_path)
    )

    result = analyze_feature_attribution(
        rounds_path=rounds_path,
        learning_root=learning_root,
    )

    assert result["round_count"] == 3
    assert result["lagged_round_count"] == 2


def test_contemporaneous_correlation(
    tmp_path: Path,
) -> None:
    rounds_path, learning_root = (
        make_inputs(tmp_path)
    )

    result = analyze_feature_attribution(
        rounds_path=rounds_path,
        learning_root=learning_root,
    )

    hot = result[
        "contemporaneous"
    ]["hot"]

    assert hot["correlations"][
        "best_hit_delta"
    ] == pytest.approx(1.0)


def test_lagged_analysis_uses_previous_signal(
    tmp_path: Path,
) -> None:
    rounds_path, learning_root = (
        make_inputs(tmp_path)
    )

    result = analyze_feature_attribution(
        rounds_path=rounds_path,
        learning_root=learning_root,
    )

    hot = result[
        "lagged_one_round"
    ]["hot"]

    assert hot["observation_count"] == 2


def test_missing_snapshot_is_rejected(
    tmp_path: Path,
) -> None:
    rounds_path, learning_root = (
        make_inputs(tmp_path)
    )

    (
        learning_root / "review-101.json"
    ).unlink()

    with pytest.raises(
        ValueError,
        match="missing feature signals",
    ):
        analyze_feature_attribution(
            rounds_path=rounds_path,
            learning_root=learning_root,
        )


def test_report_is_written(
    tmp_path: Path,
) -> None:
    rounds_path, learning_root = (
        make_inputs(tmp_path)
    )

    report = analyze_feature_attribution(
        rounds_path=rounds_path,
        learning_root=learning_root,
    )

    output = write_feature_attribution_report(
        report=report,
        output=tmp_path / "report.json",
    )

    payload = json.loads(
        output.read_text(
            encoding="utf-8"
        )
    )

    assert payload["round_count"] == 3
    assert "lagged_one_round" in payload
