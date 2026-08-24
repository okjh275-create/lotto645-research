from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import lrp.cli as cli


def test_root_commands_mapping_contains_durable_replay() -> None:
    assert "durable-replay-evaluation" in cli._COMMANDS


def test_root_main_subparser_surface_contains_durable_replay() -> None:
    source = Path("lrp/cli/__init__.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    main = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    segment = ast.get_source_segment(source, main) or ""
    assert "durable-replay-evaluation" in segment


def test_root_help_exposes_durable_replay_command() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "lrp", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "durable-replay-evaluation" in completed.stdout


def test_root_durable_replay_help_dispatches_successfully() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "lrp",
            "durable-replay-evaluation",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    output = completed.stdout + completed.stderr
    for option in (
        "--history",
        "--window-name",
        "--start-round",
        "--end-round",
        "--artifact-root",
        "--candidate-selector",
        "--baseline-selector",
    ):
        assert option in output


def test_root_command_dispatch_identity_is_exact() -> None:
    assert (
        cli._COMMANDS["durable-replay-evaluation"]
        is cli.durable_replay_evaluation_main
    )


def test_root_entrypoint_remains_lrp_cli_main() -> None:
    source = Path("lrp/__main__.py").read_text(encoding="utf-8-sig")
    assert "from lrp.cli import main" in source
    assert "SystemExit(main())" in source