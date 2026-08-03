from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validation.validation_run_discovery import (
    ValidationRunDiscovery,
)


def write_json(
    path: Path,
    payload: object,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def make_replay(
    root: Path,
    *,
    start_round: int = 100,
    end_round: int = 109,
) -> Path:
    run_root = (
        root
        / f"replay_{start_round}_{end_round}"
    )

    write_json(
        run_root / "replay_summary.json",
        {
            "config": {
                "start_round": start_round,
                "end_round": end_round,
            },
            "summary": {
                "round_count": (
                    end_round
                    - start_round
                    + 1
                ),
            },
        },
    )

    (
        run_root / "replay_rounds.jsonl"
    ).write_text(
        "{}\n",
        encoding="utf-8",
    )

    return run_root


def make_policy_comparison(
    root: Path,
) -> Path:
    run_root = (
        root / "policy_ab_100_109"
    )

    write_json(
        run_root / "policy_comparison.json",
        {
            "config": {
                "start_round": 100,
                "end_round": 109,
            },
            "scenario_count": 2,
            "results": [],
            "ranking": [],
        },
    )

    return run_root


def test_discovers_replay_run(
    tmp_path: Path,
) -> None:
    run_root = make_replay(tmp_path)

    records = (
        ValidationRunDiscovery()
        .discover(tmp_path)
    )

    assert len(records) == 1

    record = records[0]

    assert record.run_type == "replay"
    assert record.root == run_root
    assert record.start_round == 100
    assert record.end_round == 109
    assert record.round_count == 10
    assert record.status == "PASS"
    assert record.missing_files == ()


def test_discovers_policy_comparison(
    tmp_path: Path,
) -> None:
    run_root = make_policy_comparison(
        tmp_path
    )

    records = (
        ValidationRunDiscovery()
        .discover(tmp_path)
    )

    assert len(records) == 1

    record = records[0]

    assert record.run_type == (
        "policy_comparison"
    )
    assert record.root == run_root
    assert record.round_count == 10
    assert record.status == "PASS"


def test_incomplete_replay_is_reported(
    tmp_path: Path,
) -> None:
    run_root = make_replay(tmp_path)

    (
        run_root / "replay_rounds.jsonl"
    ).unlink()

    record = (
        ValidationRunDiscovery()
        .discover(tmp_path)[0]
    )

    assert record.status == "INCOMPLETE"
    assert record.missing_files == (
        "replay_rounds.jsonl",
    )


def test_optional_files_are_recorded(
    tmp_path: Path,
) -> None:
    run_root = make_replay(tmp_path)

    write_json(
        run_root
        / "effectiveness_report.json",
        {},
    )

    record = (
        ValidationRunDiscovery()
        .discover(tmp_path)[0]
    )

    assert (
        "effectiveness_report.json"
        in record.files
    )


def test_round_count_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    run_root = make_replay(tmp_path)

    write_json(
        run_root / "replay_summary.json",
        {
            "config": {
                "start_round": 100,
                "end_round": 109,
            },
            "summary": {
                "round_count": 9,
            },
        },
    )

    with pytest.raises(
        ValueError,
        match="round_count does not match",
    ):
        ValidationRunDiscovery().discover(
            tmp_path
        )


def test_missing_root_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
    ):
        ValidationRunDiscovery().discover(
            tmp_path / "missing"
        )


def test_record_serialization(
    tmp_path: Path,
) -> None:
    make_replay(tmp_path)

    record = (
        ValidationRunDiscovery()
        .discover(tmp_path)[0]
    )

    payload = record.as_dict()

    assert payload["run_type"] == "replay"
    assert payload["status"] == "PASS"
    assert payload["missing_files"] == []
