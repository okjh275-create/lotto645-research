from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import lrp.cli


def test_root_commands_include_model_evaluation() -> None:
    assert "model-evaluation" in lrp.cli._COMMANDS


def test_root_help_lists_model_evaluation() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lrp",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "model-evaluation" in result.stdout


def test_model_evaluation_help_routes_from_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lrp",
            "model-evaluation",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0

    output = result.stdout

    assert "--history" in output
    assert "--replay-output" in output
    assert "--report-output" in output
    assert "--start-round" in output
    assert "--end-round" in output


def test_lrp_package_does_not_import_tools_layer() -> None:
    root = Path("lrp")

    violations: list[str] = []

    for path in root.rglob("*.py"):
        source = path.read_text(
            encoding="utf-8-sig",
        )

        for line_number, line in enumerate(
            source.splitlines(),
            start=1,
        ):
            stripped = line.strip()

            if (
                stripped.startswith("from tools")
                or stripped.startswith("import tools")
            ):
                violations.append(
                    f"{path}:{line_number}:{stripped}"
                )

    assert violations == []
