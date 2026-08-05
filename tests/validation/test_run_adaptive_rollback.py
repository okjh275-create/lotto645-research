from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lrp.evolution.contracts import (
    AdaptiveWeightProfile,
)
from lrp.evolution.feedback import (
    AdaptiveAutomationRepository,
    AdaptiveRollbackManager,
    AdaptiveRollbackRepository,
)
from tools.validation.run_adaptive_rollback import (
    run,
)


def profile(
    *,
    revision: int,
    hot: float,
) -> dict[str, object]:
    return {
        "hot_weight": hot,
        "cold_weight": 0.47 - hot,
        "gap_weight": 0.17,
        "trend_weight": 0.14,
        "transition_weight": 0.12,
        "learning_weight": 0.05,
        "adaptive_weight": 0.05,
        "confidence": 0.80,
        "sample_size": 300,
        "revision": revision,
        "generated_at": (
            "2026-08-04T00:00:00+00:00"
        ),
    }


def write_revision(
    root: Path,
    *,
    revision: int,
    hot: float,
) -> Path:
    path = (
        root
        / "profiles"
        / f"revision-{revision:08d}.json"
    )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": 1,
        "target_revision": revision,
        "profile": profile(
            revision=revision,
            hot=hot,
        ),
    }

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    return path


def test_repository_saves_new_rollback_revision(
    tmp_path: Path,
) -> None:
    repository = AdaptiveAutomationRepository(
        tmp_path
    )

    old_path = write_revision(
        tmp_path,
        revision=12,
        hot=0.28,
    )
    current_path = write_revision(
        tmp_path,
        revision=15,
        hot=0.31,
    )

    current_payload = json.loads(
        current_path.read_text(
            encoding="utf-8"
        )
    )["profile"]

    current = AdaptiveWeightProfile(
        **{
            **current_payload,
            "generated_at": datetime.fromisoformat(
                current_payload[
                    "generated_at"
                ]
            ),
        }
    )

    plan = AdaptiveRollbackManager(
        repository=repository
    ).plan(
        current_profile=current,
        rollback_revision=12,
    )

    saved = AdaptiveRollbackRepository(
        repository=repository
    ).save(
        plan,
        rollback_id="rollback-16",
    )

    assert saved.created is True
    assert saved.path.name == (
        "revision-00000016.json"
    )
    assert old_path.is_file()


def test_cli_dry_run_does_not_write(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_revision(
        tmp_path,
        revision=12,
        hot=0.28,
    )
    current_path = write_revision(
        tmp_path,
        revision=15,
        hot=0.31,
    )

    assert run(
        [
            "--profile",
            str(current_path),
            "--repository",
            str(tmp_path),
            "--rollback-revision",
            "12",
            "--rollback-id",
            "dry-16",
            "--dry-run",
        ]
    ) == 0

    payload = json.loads(
        capsys.readouterr().out
    )

    assert payload["mode"] == "dry_run"
    assert payload["created"] is False
    assert not (
        tmp_path
        / "profiles"
        / "revision-00000016.json"
    ).exists()


def test_cli_requires_explicit_approval(
    tmp_path: Path,
) -> None:
    current_path = write_revision(
        tmp_path,
        revision=15,
        hot=0.31,
    )

    with pytest.raises(SystemExit) as error:
        run(
            [
                "--profile",
                str(current_path),
                "--repository",
                str(tmp_path),
                "--rollback-revision",
                "12",
                "--rollback-id",
                "rollback-16",
            ]
        )

    assert error.value.code == 2
