from __future__ import annotations

import json
from pathlib import Path

from lrp.cli.doctor import main
from lrp.operations import write_operation_artifact


def prepare_project(root: Path) -> None:
    (root / "lrp").mkdir()
    (root / "tests").mkdir()
    (root / "tools").mkdir()
    (root / "config.yaml").write_text(
        "project: test\n",
        encoding="utf-8",
    )


def write_round(root: Path, round_no: int) -> None:
    write_operation_artifact(
        {
            "round_no": round_no,
            "learning": {
                "snapshot_id": f"review-{round_no}",
                "feedback_count": 2,
            },
            "profile": {
                "applied": True,
            },
        },
        output_root=root / "snapshots",
        artifact_type="round-completion",
        round_no=round_no,
        filename="round_completion.json",
    )


def test_doctor_verifies_round_completion(
    tmp_path: Path,
    capsys,
) -> None:
    prepare_project(tmp_path)
    write_round(tmp_path, 1232)

    code = main([
        "--root",
        str(tmp_path),
        "--round-completion",
    ])

    assert code == 0

    payload = json.loads(
        capsys.readouterr().out
    )

    assert payload["status"] == "PASS"
    assert any(
        item["name"] == "round_completion:1232"
        and item["status"] == "PASS"
        for item in payload["checks"]
    )


def test_doctor_detects_tampered_round_completion(
    tmp_path: Path,
    capsys,
) -> None:
    prepare_project(tmp_path)
    write_round(tmp_path, 1232)

    data_path = (
        tmp_path
        / "snapshots"
        / "round-completion"
        / "round_1232"
        / "round_completion.json"
    )

    data_path.write_text(
        '{"tampered": true}\n',
        encoding="utf-8",
    )

    code = main([
        "--root",
        str(tmp_path),
        "--round-completion",
    ])

    assert code == 1

    payload = json.loads(
        capsys.readouterr().out
    )

    assert payload["status"] == "FAIL"
    assert payload["failure_count"] >= 1
    assert any(
        item["name"] == "round_completion:1232"
        and item["status"] == "FAIL"
        for item in payload["checks"]
    )
