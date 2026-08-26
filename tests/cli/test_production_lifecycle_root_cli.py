from __future__ import annotations

import subprocess
import sys

import lrp.cli as root_cli


EXPECTED_COMMAND = "production-lifecycle"


def _run_root(
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "lrp",
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_root_command_registry_contains_production_lifecycle(
) -> None:
    assert (
        EXPECTED_COMMAND
        in root_cli._COMMANDS
    )


def test_root_help_exposes_production_lifecycle(
) -> None:
    result = _run_root(
        "--help"
    )

    assert result.returncode == 0
    assert (
        EXPECTED_COMMAND
        in result.stdout
    )


def test_production_lifecycle_help_is_available(
) -> None:
    result = _run_root(
        EXPECTED_COMMAND,
        "--help",
    )

    assert result.returncode == 0

    output = (
        result.stdout
        + result.stderr
    )

    assert (
        "production lifecycle"
        in output.lower()
    )


def test_existing_root_commands_are_preserved(
) -> None:
    expected_existing = {
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

    assert expected_existing.issubset(
        set(root_cli._COMMANDS)
    )


def test_root_command_count_becomes_eighteen(
) -> None:
    assert len(
        root_cli._COMMANDS
    ) == 18
