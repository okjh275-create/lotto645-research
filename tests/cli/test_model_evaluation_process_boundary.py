from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import lrp.cli.model_evaluation as cli


def test_adapter_targets_project_m_module() -> None:
    assert (
        cli._TARGET_MODULE
        == "tools.validation.run_model_evaluation"
    )


def test_adapter_delegates_through_python_module_process(
    monkeypatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(
        command,
        *,
        check,
    ):
        observed["command"] = command
        observed["check"] = check

        return SimpleNamespace(
            returncode=7,
        )

    monkeypatch.setattr(
        cli.subprocess,
        "run",
        fake_run,
    )

    result = cli.main(
        [
            "--help",
        ]
    )

    assert result == 7

    command = observed[
        "command"
    ]

    assert command[0] == cli.sys.executable
    assert command[1:3] == [
        "-m",
        "tools.validation.run_model_evaluation",
    ]
    assert command[3:] == [
        "--help",
    ]

    assert observed["check"] is False


def test_adapter_has_no_tools_python_import() -> None:
    path = Path(
        "lrp/cli/model_evaluation.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8-sig",
        )
    )

    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(
            node,
            ast.ImportFrom,
        ):
            module = node.module or ""

            if (
                module == "tools"
                or module.startswith(
                    "tools."
                )
            ):
                violations.append(
                    module
                )

        if isinstance(
            node,
            ast.Import,
        ):
            for alias in node.names:
                if (
                    alias.name == "tools"
                    or alias.name.startswith(
                        "tools."
                    )
                ):
                    violations.append(
                        alias.name
                    )

    assert violations == []


def test_lrp_tree_has_no_tools_python_import() -> None:
    root = Path("lrp")

    violations: list[str] = []

    for path in root.rglob("*.py"):
        tree = ast.parse(
            path.read_text(
                encoding="utf-8-sig",
            )
        )

        for node in ast.walk(tree):
            if isinstance(
                node,
                ast.ImportFrom,
            ):
                module = node.module or ""

                if (
                    module == "tools"
                    or module.startswith(
                        "tools."
                    )
                ):
                    violations.append(
                        f"{path}:{node.lineno}"
                    )

            if isinstance(
                node,
                ast.Import,
            ):
                for alias in node.names:
                    if (
                        alias.name == "tools"
                        or alias.name.startswith(
                            "tools."
                        )
                    ):
                        violations.append(
                            f"{path}:{node.lineno}"
                        )

    assert violations == []