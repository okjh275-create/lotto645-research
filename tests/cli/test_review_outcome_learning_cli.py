from __future__ import annotations

import json
from pathlib import Path

from lrp.cli.review import main
from lrp.learning import LearningRepository


def prediction_payload() -> dict[str, object]:
    return {
        "round": 1232,
        "generated_at_kst": "2026-08-08T20:30:00+09:00",
        "seed": 20260808,
        "params": {
            "temperature": 0.85,
        },
        "sets": [
            {
                "id": "S1",
                "numbers": [3, 8, 14, 22, 35, 41],
                "score": 0.91,
                "risk_flags": [],
                "features": {"sum": 123},
            },
            {
                "id": "S2",
                "numbers": [4, 11, 19, 27, 34, 42],
                "score": 0.84,
                "risk_flags": [],
                "features": {"sum": 137},
            },
        ],
        "top5_practical": ["S1"],
        "metadata": {
            "statistics_version": "1.0.0",
        },
    }


def write_prediction(tmp_path: Path) -> Path:
    path = tmp_path / "prediction.json"
    path.write_text(
        json.dumps(
            prediction_payload(),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def test_cli_runs_outcome_and_learning_flow(
    tmp_path: Path,
    capsys,
) -> None:
    prediction = write_prediction(tmp_path)
    output = tmp_path / "output"

    exit_code = main(
        [
            "--prediction",
            str(prediction),
            "--numbers",
            "3",
            "8",
            "14",
            "22",
            "35",
            "41",
            "--bonus",
            "9",
            "--output",
            str(output),
            "--learn",
            "--learning-policy",
            "thompson",
        ]
    )

    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["status"] == "PASS"
    assert payload["outcome"]["status"] == "PASS"
    assert payload["outcome"]["round_no"] == 1232
    assert payload["outcome"]["created_predictions"] == 2
    assert payload["outcome"]["reviews_created"] == 2

    assert payload["learning"]["learning_snapshot_id"] == (
        "review-1232"
    )
    assert payload["learning"]["feedback_count"] > 0

    database = (
        output
        / "learning"
        / "learning.db"
    )

    repository = LearningRepository(database)
    counts = repository.counts()

    assert counts["predictions"] == 2
    assert counts["results"] == 1
    assert counts["reviews"] == 2

    assert (
        output
        / "learning"
        / "review-1232.json"
    ).is_file()


def test_cli_is_idempotent_for_outcome_repository(
    tmp_path: Path,
    capsys,
) -> None:
    prediction = write_prediction(tmp_path)
    output = tmp_path / "output"

    args = [
        "--prediction",
        str(prediction),
        "--numbers",
        "3",
        "8",
        "14",
        "22",
        "35",
        "41",
        "--bonus",
        "9",
        "--output",
        str(output),
        "--learn",
        "--learning-policy",
        "thompson",
        "--overwrite-learning",
    ]

    first_code = main(args)
    first = json.loads(
        capsys.readouterr().out
    )

    second_code = main(args)
    second_capture = capsys.readouterr()

    assert second_code == 0, (
        "second CLI execution failed:`n"
        f"stdout={second_capture.out!r}`n"
        f"stderr={second_capture.err!r}"
    )

    second = json.loads(
        second_capture.out
    )

    assert first_code == 0

    assert first["outcome"]["created_predictions"] == 2
    assert first["outcome"]["result_created"] is True

    assert second["outcome"]["created_predictions"] == 0
    assert second["outcome"]["existing_predictions"] == 2
    assert second["outcome"]["result_created"] is False
    assert second["outcome"]["reviews_created"] == 0


def test_cli_skips_outcome_persistence_without_bonus(
    tmp_path: Path,
    capsys,
) -> None:
    prediction = write_prediction(tmp_path)
    output = tmp_path / "output"

    exit_code = main(
        [
            "--prediction",
            str(prediction),
            "--numbers",
            "3",
            "8",
            "14",
            "22",
            "35",
            "41",
            "--output",
            str(output),
            "--learn",
        ]
    )

    assert exit_code == 0

    payload = json.loads(
        capsys.readouterr().out
    )

    assert payload["outcome"]["status"] == "SKIPPED"
    assert payload["outcome"]["reason"] == "bonus_required"
