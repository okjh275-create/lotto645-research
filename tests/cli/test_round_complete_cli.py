from __future__ import annotations

import json
from pathlib import Path

from lrp.cli.round_complete import main


def write_prediction(tmp_path: Path) -> Path:
    payload = {
        "round": 1232,
        "generated_at_kst": "2026-08-08T20:30:00+09:00",
        "seed": 20260808,
        "params": {"temperature": 0.85},
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
        "metadata": {"statistics_version": "1.0.0"},
    }

    path = tmp_path / "prediction.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def test_round_complete_cli_runs(tmp_path: Path, capsys) -> None:
    prediction = write_prediction(tmp_path)
    output = tmp_path / "output"

    code = main(
        [
            "--prediction",
            str(prediction),
            "--numbers",
            "3", "8", "14", "22", "35", "41",
            "--bonus",
            "9",
            "--output",
            str(output),
            "--policy",
            "thompson",
        ]
    )

    assert code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["status"] == "PASS"
    assert payload["round_no"] == 1232
    assert payload["outcome"]["created_predictions"] == 2
    assert payload["learning"]["feedback_count"] > 0

    assert (
        output
        / "learning"
        / "review-1232.json"
    ).is_file()
