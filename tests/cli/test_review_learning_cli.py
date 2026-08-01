from __future__ import annotations

import argparse
from pathlib import Path

from lrp.cli.review import (
    _parser,
    _run_learning,
)


def make_payload(
    *,
    set_count: int = 20,
) -> dict[str, object]:
    return {
        "round": 1220,
        "summary": {
            "set_count": set_count,
            "best_main_hits": 4,
            "practical_best_hits": 3,
        },
    }


def make_arguments(
    tmp_path: Path,
    *,
    confidence: float = 0.80,
) -> argparse.Namespace:
    return argparse.Namespace(
        prediction="prediction.json",
        numbers=[
            1,
            2,
            3,
            4,
            5,
            6,
        ],
        bonus=None,
        output=str(tmp_path / "output"),
        learn=True,
        learning_snapshots=str(
            tmp_path / "learning"
        ),
        profile_snapshots=str(
            tmp_path / "profiles"
        ),
        learning_policy="thompson",
        learning_snapshot_id=None,
        overwrite_learning=False,
        learning_confidence=confidence,
    )


def test_parser_preserves_default_review_mode() -> None:
    arguments = _parser().parse_args(
        [
            "--prediction",
            "prediction.json",
            "--numbers",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
        ]
    )

    assert arguments.learn is False
    assert arguments.output == "snapshots"
    assert arguments.learning_confidence == 0.80


def test_learning_flow_saves_both_snapshots(
    tmp_path: Path,
) -> None:
    arguments = make_arguments(tmp_path)

    result = _run_learning(
        payload=make_payload(),
        arguments=arguments,
    )

    assert result[
        "learning_snapshot_id"
    ] == "review-1220"
    assert result["feedback_count"] == 2
    assert result["step_count"] == 2
    assert result["profile_applied"] is True
    assert (
        result["profile_snapshot_saved"]
        is True
    )
    assert result["profile_revision"] == 1

    assert (
        tmp_path
        / "learning"
        / "review-1220.json"
    ).is_file()


def test_profile_policy_rejects_small_sample(
    tmp_path: Path,
) -> None:
    result = _run_learning(
        payload=make_payload(
            set_count=10
        ),
        arguments=make_arguments(tmp_path),
    )

    assert result["profile_applied"] is False
    assert (
        result["profile_snapshot_saved"]
        is False
    )
    assert (
        "sample_size_below_threshold"
        in result["profile_reasons"]
    )


def test_custom_snapshot_id_is_used(
    tmp_path: Path,
) -> None:
    arguments = make_arguments(tmp_path)
    arguments.learning_snapshot_id = (
        "custom-review-1220"
    )

    result = _run_learning(
        payload=make_payload(),
        arguments=arguments,
    )

    assert result[
        "learning_snapshot_id"
    ] == "custom-review-1220"

    assert (
        tmp_path
        / "learning"
        / "custom-review-1220.json"
    ).is_file()


def test_default_learning_paths_follow_output(
    tmp_path: Path,
) -> None:
    arguments = make_arguments(tmp_path)
    arguments.learning_snapshots = None
    arguments.profile_snapshots = None

    result = _run_learning(
        payload=make_payload(),
        arguments=arguments,
    )

    assert result[
        "learning_snapshot_root"
    ] == str(
        tmp_path / "output" / "learning"
    )

    assert result[
        "profile_snapshot_root"
    ] == str(
        tmp_path / "output" / "profiles"
    )
