from __future__ import annotations

import json
from pathlib import Path

import pytest

import lrp.cli


def _write_decision(
    path: Path,
    *,
    selected_model: object,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            {
                "selection": {
                    "selected_model": selected_model,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_root_commands_include_publish_champion() -> None:
    assert (
        "publish-champion"
        in lrp.cli._COMMANDS
    )


def test_root_publish_champion_routes_successfully(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = (
        tmp_path
        / "champion_decision.json"
    )

    registry = (
        tmp_path
        / "registry"
    )

    _write_decision(
        source,
        selected_model="combined",
    )

    exit_code = lrp.cli.main(
        [
            "publish-champion",
            "--champion-decision",
            str(source),
            "--production-registry",
            str(registry),
        ]
    )

    assert exit_code == 0

    captured = capsys.readouterr()

    payload = json.loads(
        captured.out
    )

    assert payload[
        "status"
    ] == "PASS"

    assert payload[
        "selected_model"
    ] == "combined"

    assert (
        registry
        / "active"
        / "champion_decision.json"
    ).is_file()

    assert (
        registry
        / "active"
        / "publication.json"
    ).is_file()


def test_root_publish_champion_preserves_none_model(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = (
        tmp_path
        / "champion_decision.json"
    )

    registry = (
        tmp_path
        / "registry"
    )

    _write_decision(
        source,
        selected_model=None,
    )

    exit_code = lrp.cli.main(
        [
            "publish-champion",
            "--champion-decision",
            str(source),
            "--production-registry",
            str(registry),
        ]
    )

    assert exit_code == 0

    captured = capsys.readouterr()

    payload = json.loads(
        captured.out
    )

    assert payload[
        "selected_model"
    ] is None


def test_root_publish_champion_propagates_error_exit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = lrp.cli.main(
        [
            "publish-champion",
            "--champion-decision",
            str(
                tmp_path
                / "missing.json"
            ),
            "--production-registry",
            str(
                tmp_path
                / "registry"
            ),
        ]
    )

    assert exit_code == 1

    captured = capsys.readouterr()

    payload = json.loads(
        captured.err
    )

    assert payload[
        "status"
    ] == "ERROR"

    assert payload[
        "error_type"
    ] == "FileNotFoundError"


def test_root_help_lists_publish_champion(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(
        SystemExit,
    ) as exc_info:
        lrp.cli.main(
            [
                "--help",
            ]
        )

    assert exc_info.value.code == 0

    captured = capsys.readouterr()

    assert (
        "publish-champion"
        in captured.out
    )


def test_existing_predict_command_remains_registered() -> None:
    assert "predict" in lrp.cli._COMMANDS


def test_existing_command_count_increases_by_one() -> None:
    expected = {
        "predict",
        "weekly",
        "review",
        "round-complete",
        "verify",
        "backup",
        "restore",
        "status",
        "doctor",
        "export-history",
        "publish-champion",
        "audit-champion",
        "model-evaluation",
    }

    assert (
        set(lrp.cli._COMMANDS)
        == expected
    )
