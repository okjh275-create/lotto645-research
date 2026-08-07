from __future__ import annotations

import json
from pathlib import Path

from lrp.cli.doctor import main as doctor_main
from lrp.cli.round_complete import main as round_complete_main
from lrp.cli.status import main as status_main
from lrp.learning import LearningRepository
from lrp.operations import verify_manifest


def prepare_project(root: Path) -> None:
    (root / "lrp").mkdir()
    (root / "tests").mkdir()
    (root / "tools").mkdir()
    (root / "config.yaml").write_text(
        "project: test\n",
        encoding="utf-8",
    )


def write_prediction(root: Path) -> Path:
    payload = {
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

    path = root / "prediction.json"
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


def test_round_completion_operational_flow(
    tmp_path: Path,
    capsys,
) -> None:
    prepare_project(tmp_path)
    prediction = write_prediction(tmp_path)
    snapshots = tmp_path / "snapshots"

    round_code = round_complete_main(
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
            str(snapshots),
            "--policy",
            "thompson",
        ]
    )

    assert round_code == 0

    round_response = json.loads(
        capsys.readouterr().out
    )

    assert round_response["schema_version"] == "1.0"
    assert round_response["status"] == "PASS"
    assert round_response["round_no"] == 1232
    assert (
        round_response["verification"]["status"]
        == "PASS"
    )
    assert round_response["warnings"] == []

    artifact = round_response["artifact"]
    data_path = Path(artifact["data_path"])
    manifest_path = Path(
        artifact["manifest_path"]
    )

    assert data_path.is_file()
    assert manifest_path.is_file()

    verification = verify_manifest(
        manifest_path
    )

    assert verification["status"] == "PASS"
    assert verification["failures"] == []

    learning_db = (
        snapshots
        / "learning"
        / "learning.db"
    )

    repository = LearningRepository(
        learning_db
    )

    counts = repository.counts()

    assert counts["predictions"] == 2
    assert counts["results"] == 1
    assert counts["reviews"] == 2

    assert (
        snapshots
        / "learning"
        / "review-1232.json"
    ).is_file()

    status_code = status_main(
        [
            "--root",
            str(tmp_path),
            "--snapshots",
            str(snapshots),
            "--round-completion",
            "--round-limit",
            "20",
        ]
    )

    assert status_code == 0

    status_payload = json.loads(
        capsys.readouterr().out
    )

    assert status_payload["status"] == "PASS"

    completion = (
        status_payload["round_completion"]
    )

    assert completion["completion_count"] == 1
    assert completion["latest_round"] == 1232
    assert completion["manifest_pass_rate"] == 1.0
    assert completion["latest_snapshot_id"] == (
        "review-1232"
    )

    doctor_code = doctor_main(
        [
            "--root",
            str(tmp_path),
            "--snapshots",
            str(snapshots),
            "--round-completion",
            "--round-limit",
            "20",
        ]
    )

    assert doctor_code == 0

    doctor_payload = json.loads(
        capsys.readouterr().out
    )

    assert doctor_payload["status"] == "PASS"
    assert doctor_payload["failure_count"] == 0

    assert any(
        item["name"] == "round_completion:1232"
        and item["status"] == "PASS"
        for item in doctor_payload["checks"]
    )


def test_operational_flow_detects_post_completion_tampering(
    tmp_path: Path,
    capsys,
) -> None:
    prepare_project(tmp_path)
    prediction = write_prediction(tmp_path)
    snapshots = tmp_path / "snapshots"

    code = round_complete_main(
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
            str(snapshots),
        ]
    )

    assert code == 0

    response = json.loads(
        capsys.readouterr().out
    )

    data_path = Path(
        response["artifact"]["data_path"]
    )

    data_path.write_text(
        '{"tampered": true}\n',
        encoding="utf-8",
    )

    doctor_code = doctor_main(
        [
            "--root",
            str(tmp_path),
            "--snapshots",
            str(snapshots),
            "--round-completion",
        ]
    )

    assert doctor_code == 1

    doctor_payload = json.loads(
        capsys.readouterr().out
    )

    assert doctor_payload["status"] == "FAIL"
    assert doctor_payload["failure_count"] >= 1

    assert any(
        item["name"] == "round_completion:1232"
        and item["status"] == "FAIL"
        for item in doctor_payload["checks"]
    )
