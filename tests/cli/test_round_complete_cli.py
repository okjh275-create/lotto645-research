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

def test_round_complete_writes_operation_artifact(
    tmp_path: Path,
    capsys,
) -> None:
    prediction = write_prediction(tmp_path)
    output = tmp_path / "output"

    code = main(
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
            "--policy",
            "thompson",
        ]
    )

    assert code == 0

    response = json.loads(
        capsys.readouterr().out
    )

    data_path = (
        output
        / "round-completion"
        / "round_1232"
        / "round_completion.json"
    )
    manifest_path = (
        output
        / "round-completion"
        / "round_1232"
        / "manifest.json"
    )
    operation_log = (
        output
        / "operation_log.jsonl"
    )

    assert data_path.is_file()
    assert manifest_path.is_file()
    assert operation_log.is_file()

    saved = json.loads(
        data_path.read_text(
            encoding="utf-8"
        )
    )

    assert saved["round_no"] == 1232
    assert saved["platform_version"] == "4.0.0"
    assert saved["completed_at_kst"]
    assert saved["learning"]["feedback_count"] > 0

    assert (
        response["artifact"]["data_path"]
        == str(data_path.resolve())
    )
    assert response["artifact"]["sha256"]
    assert response["verification"]["status"] == "PASS"
    assert response["verification"]["failures"] == []
    assert response["verification"]["checked"]

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    assert manifest["schema_version"] == "1.0"
    assert manifest["artifact_type"] == (
        "round-completion"
    )
    assert manifest["round"] == 1232
    assert (
        "round_completion.json"
        in manifest["files"]
    )

def test_round_complete_manifest_detects_tampering(
    tmp_path: Path,
    capsys,
) -> None:
    from lrp.operations import verify_manifest

    prediction = write_prediction(tmp_path)
    output = tmp_path / "output"

    code = main(
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
            "--policy",
            "thompson",
        ]
    )

    assert code == 0

    response = json.loads(
        capsys.readouterr().out
    )

    assert response["verification"]["status"] == "PASS"

    data_path = Path(
        response["artifact"]["data_path"]
    )
    manifest_path = Path(
        response["artifact"]["manifest_path"]
    )

    data_path.write_text(
        '{"tampered": true}\n',
        encoding="utf-8",
    )

    verification = verify_manifest(
        manifest_path
    )

    assert verification["status"] == "FAIL"
    assert verification["failures"]
    assert (
        verification["failures"][0]["reason"]
        == "sha256_mismatch"
    )

def test_round_complete_success_contract(
    tmp_path: Path,
    capsys,
) -> None:
    prediction = write_prediction(tmp_path)

    code = main(
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
            str(tmp_path / "output"),
        ]
    )

    assert code == 0

    payload = json.loads(
        capsys.readouterr().out
    )

    assert payload["schema_version"] == "1.0"
    assert payload["status"] == "PASS"
    assert payload["warnings"] == []
    assert payload["verification"]["status"] == "PASS"
    assert "artifact" in payload
    assert "learning" in payload
    assert "profile" in payload


def test_round_complete_error_contract(
    tmp_path: Path,
    capsys,
) -> None:
    missing = tmp_path / "missing.json"

    code = main(
        [
            "--prediction",
            str(missing),
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
            str(tmp_path / "output"),
        ]
    )

    assert code == 1

    captured = capsys.readouterr()

    assert captured.out == ""

    payload = json.loads(
        captured.err
    )

    assert payload["schema_version"] == "1.0"
    assert payload["status"] == "ERROR"
    assert payload["error_type"] == "FileNotFoundError"
    assert payload["message"]
    assert payload["warnings"] == []
