from __future__ import annotations

import json
from pathlib import Path

from lrp.cli.status import main
from lrp.operations import write_operation_artifact


def prepare_project(root: Path) -> None:
    (root / "lrp").mkdir()
    (root / "tests").mkdir()
    (root / "tools").mkdir()
    (root / "config.yaml").write_text(
        "project: test\n",
        encoding="utf-8",
    )


def write_round(
    root: Path,
    *,
    round_no: int,
    feedback_count: int,
    applied: bool,
) -> None:
    payload = {
        "round_no": round_no,
        "learning": {
            "snapshot_id": f"review-{round_no}",
            "feedback_count": feedback_count,
            "final_context_version": 2,
        },
        "profile": {
            "applied": applied,
            "revision": 1 if applied else None,
            "snapshot_saved": applied,
            "reasons": [],
        },
    }

    write_operation_artifact(
        payload,
        output_root=root / "snapshots",
        artifact_type="round-completion",
        round_no=round_no,
        filename="round_completion.json",
    )


def test_default_status_does_not_add_round_completion(
    tmp_path: Path,
    capsys,
) -> None:
    prepare_project(tmp_path)

    code = main([
        "--root",
        str(tmp_path),
    ])

    assert code == 0

    payload = json.loads(
        capsys.readouterr().out
    )

    assert "round_completion" not in payload


def test_status_includes_round_completion_summary(
    tmp_path: Path,
    capsys,
) -> None:
    prepare_project(tmp_path)

    write_round(
        tmp_path,
        round_no=1231,
        feedback_count=2,
        applied=True,
    )
    write_round(
        tmp_path,
        round_no=1232,
        feedback_count=4,
        applied=False,
    )

    code = main([
        "--root",
        str(tmp_path),
        "--round-completion",
        "--round-limit",
        "20",
    ])

    assert code == 0

    payload = json.loads(
        capsys.readouterr().out
    )

    summary = payload["round_completion"]

    assert summary["completion_count"] == 2
    assert summary["latest_round"] == 1232
    assert summary["average_feedback_count"] == 3.0
    assert summary["profile_apply_rate"] == 0.5
    assert summary["manifest_pass_rate"] == 1.0
    assert summary["latest_snapshot_id"] == "review-1232"
    assert summary["limit"] == 20


def test_status_round_limit_is_applied(
    tmp_path: Path,
    capsys,
) -> None:
    prepare_project(tmp_path)

    for round_no in (1230, 1231, 1232):
        write_round(
            tmp_path,
            round_no=round_no,
            feedback_count=round_no - 1229,
            applied=True,
        )

    code = main([
        "--root",
        str(tmp_path),
        "--round-completion",
        "--round-limit",
        "2",
    ])

    assert code == 0

    payload = json.loads(
        capsys.readouterr().out
    )

    assert (
        payload["round_completion"]["completion_count"]
        == 2
    )
    assert (
        payload["round_completion"]["latest_round"]
        == 1232
    )
