from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import lrp.cli as root_cli
from lrp.production import (
    ProductionChampionRegistryPublisher,
)


def _write_decision(
    path: Path,
    *,
    model: str,
) -> Path:
    payload = {
        "selection": {
            "selected_model": model,
        },
    }

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return path


def _publication_revision_id(
    registry: Path,
    *,
    source_sha256: str,
) -> str:
    history = (
        registry
        / "history"
    )

    for path in history.glob(
        "*.json"
    ):
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if (
            payload.get(
                "source_sha256"
            )
            == source_sha256
        ):
            return path.stem

    raise AssertionError(
        "publication revision not found"
    )


def _active_snapshot(
    registry: Path,
) -> dict[str, bytes]:
    active = (
        registry
        / "active"
    )

    return {
        path.name: path.read_bytes()
        for path in active.iterdir()
        if path.is_file()
    }


def _prepare_registry(
    tmp_path: Path,
):
    registry = (
        tmp_path
        / "registry"
    )

    publisher = (
        ProductionChampionRegistryPublisher()
    )

    first = publisher.publish(
        source_decision=(
            _write_decision(
                tmp_path / "a.json",
                model="model-a",
            )
        ),
        registry_root=registry,
    )

    second = publisher.publish(
        source_decision=(
            _write_decision(
                tmp_path / "b.json",
                model="model-b",
            )
        ),
        registry_root=registry,
    )

    revision_id = (
        _publication_revision_id(
            registry,
            source_sha256=(
                first.source_sha256
            ),
        )
    )

    return (
        registry,
        first,
        second,
        revision_id,
    )


def test_root_commands_include_rollback_champion(
) -> None:
    assert (
        "rollback-champion"
        in root_cli._COMMANDS
    )


def test_root_command_count_increases_to_sixteen(
) -> None:
    assert len(
        root_cli._COMMANDS
    ) == 16


def test_root_help_lists_rollback_champion(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(
        SystemExit
    ) as exc_info:
        root_cli.main(
            [
                "--help",
            ]
        )

    assert (
        exc_info.value.code
        == 0
    )

    output = (
        capsys.readouterr()
        .out
    )

    assert (
        "rollback-champion"
        in output
    )


def test_rollback_champion_help_is_available(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(
        SystemExit
    ) as exc_info:
        root_cli.main(
            [
                "rollback-champion",
                "--help",
            ]
        )

    assert (
        exc_info.value.code
        == 0
    )

    output = (
        capsys.readouterr()
        .out
    )

    assert (
        "--production-registry"
        in output
    )

    assert (
        "--revision-id"
        in output
    )

    assert (
        "--execute"
        in output
    )


def test_default_rollback_command_is_plan_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (
        registry,
        first,
        second,
        revision_id,
    ) = _prepare_registry(
        tmp_path
    )

    active_before = (
        _active_snapshot(
            registry
        )
    )

    exit_code = root_cli.main(
        [
            "rollback-champion",
            "--production-registry",
            str(registry),
            "--revision-id",
            revision_id,
        ]
    )

    assert exit_code == 0

    captured = (
        capsys.readouterr()
    )

    assert captured.err == ""

    payload = json.loads(
        captured.out
    )

    assert (
        payload["status"]
        == "PASS"
    )

    assert (
        payload["mode"]
        == "PLAN"
    )

    assert (
        payload["target_revision_id"]
        == revision_id
    )

    assert (
        payload["target_source_sha256"]
        == first.source_sha256
    )

    assert (
        payload["target_selected_model"]
        == "model-a"
    )

    assert (
        payload["active_source_sha256"]
        == second.source_sha256
    )

    assert (
        _active_snapshot(
            registry
        )
        == active_before
    )


def test_execute_flag_performs_rollback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (
        registry,
        first,
        second,
        revision_id,
    ) = _prepare_registry(
        tmp_path
    )

    exit_code = root_cli.main(
        [
            "rollback-champion",
            "--production-registry",
            str(registry),
            "--revision-id",
            revision_id,
            "--execute",
        ]
    )

    assert exit_code == 0

    captured = (
        capsys.readouterr()
    )

    assert captured.err == ""

    payload = json.loads(
        captured.out
    )

    assert (
        payload["status"]
        == "PASS"
    )

    assert (
        payload["mode"]
        == "EXECUTE"
    )

    assert (
        payload["target_revision_id"]
        == revision_id
    )

    assert (
        payload["source_sha256"]
        == first.source_sha256
    )

    assert (
        payload["selected_model"]
        == "model-a"
    )

    active_decision = json.loads(
        (
            registry
            / "active"
            / "champion_decision.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        active_decision[
            "selection"
        ][
            "selected_model"
        ]
        == "model-a"
    )


def test_invalid_revision_returns_error_exit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (
        registry,
        _,
        _,
        _,
    ) = _prepare_registry(
        tmp_path
    )

    exit_code = root_cli.main(
        [
            "rollback-champion",
            "--production-registry",
            str(registry),
            "--revision-id",
            "0" * 64,
        ]
    )

    assert exit_code == 1

    captured = (
        capsys.readouterr()
    )

    assert captured.out == ""

    payload = json.loads(
        captured.err
    )

    assert (
        payload["status"]
        == "ERROR"
    )

    assert (
        isinstance(
            payload["error_type"],
            str,
        )
    )

    assert (
        isinstance(
            payload["message"],
            str,
        )
    )

    assert payload["message"]


def test_plan_mode_creates_no_rollback_audit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (
        registry,
        _,
        _,
        revision_id,
    ) = _prepare_registry(
        tmp_path
    )

    rollback_root = (
        registry
        / "history"
        / "rollbacks"
    )

    before = (
        sorted(
            path.name
            for path
            in rollback_root.glob(
                "*.json"
            )
        )
        if rollback_root.exists()
        else []
    )

    exit_code = root_cli.main(
        [
            "rollback-champion",
            "--production-registry",
            str(registry),
            "--revision-id",
            revision_id,
        ]
    )

    assert exit_code == 0

    capsys.readouterr()

    after = (
        sorted(
            path.name
            for path
            in rollback_root.glob(
                "*.json"
            )
        )
        if rollback_root.exists()
        else []
    )

    assert after == before
