"""Export exact source contracts required for Project E E-004 integration.

Read-only utility. It does not modify project source code, databases,
predictions, manifests, or build artifacts.
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = ROOT / "project_e004_contracts.txt"

TARGET_FILES = (
    "lrp/adapters/candidate.py",
    "lrp/ensemble/__init__.py",
    "lrp/ensemble/adapters.py",
    "lrp/ensemble/engine.py",
    "lrp/ensemble/models.py",
    "lrp/ensemble/rescoring.py",
    "lrp/ensemble/snapshot.py",
    "lrp/pipelines/models.py",
    "lrp/pipelines/prediction.py",
    "lrp/pipelines/serializer.py",
    "tests/test_candidate.py",
    "tests/test_e001_ensemble_foundation.py",
    "tests/test_e002_learning_snapshot.py",
    "tests/test_e003_feature_rescoring.py",
    "tests/test_prediction_orchestrator.py",
)


def read_source(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(
            encoding="cp949",
            errors="replace",
        )


def safe_unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""

    try:
        return ast.unparse(node)
    except Exception:
        return node.__class__.__name__


def assigned_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]

    if isinstance(node, ast.Attribute):
        return [safe_unparse(node)]

    if isinstance(node, (ast.Tuple, ast.List)):
        result: list[str] = []

        for item in node.elts:
            result.extend(assigned_names(item))

        return result

    return []


def class_fields(node: ast.ClassDef) -> list[str]:
    result: list[str] = []

    for child in node.body:
        if isinstance(child, ast.AnnAssign):
            result.extend(
                assigned_names(child.target)
            )

        elif isinstance(child, ast.Assign):
            for target in child.targets:
                result.extend(
                    assigned_names(target)
                )

        elif isinstance(
            child,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ) and child.name == "__init__":
            for item in ast.walk(child):
                if isinstance(item, ast.Assign):
                    targets = item.targets
                elif isinstance(item, ast.AnnAssign):
                    targets = [item.target]
                else:
                    continue

                for target in targets:
                    for name in assigned_names(target):
                        if name.startswith("self."):
                            result.append(name)

    return sorted(set(result))


def function_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    prefix = (
        "async def"
        if isinstance(node, ast.AsyncFunctionDef)
        else "def"
    )

    arguments: list[str] = []

    positional = (
        *node.args.posonlyargs,
        *node.args.args,
    )

    positional_defaults = (
        [None]
        * (
            len(positional)
            - len(node.args.defaults)
        )
        + list(node.args.defaults)
    )

    for argument, default in zip(
        positional,
        positional_defaults,
        strict=True,
    ):
        rendered = argument.arg

        annotation = safe_unparse(
            argument.annotation
        )

        if annotation:
            rendered += f": {annotation}"

        if default is not None:
            rendered += (
                f" = {safe_unparse(default)}"
            )

        arguments.append(rendered)

    if node.args.vararg is not None:
        rendered = (
            f"*{node.args.vararg.arg}"
        )

        annotation = safe_unparse(
            node.args.vararg.annotation
        )

        if annotation:
            rendered += f": {annotation}"

        arguments.append(rendered)
    elif node.args.kwonlyargs:
        arguments.append("*")

    for argument, default in zip(
        node.args.kwonlyargs,
        node.args.kw_defaults,
        strict=True,
    ):
        rendered = argument.arg

        annotation = safe_unparse(
            argument.annotation
        )

        if annotation:
            rendered += f": {annotation}"

        if default is not None:
            rendered += (
                f" = {safe_unparse(default)}"
            )

        arguments.append(rendered)

    if node.args.kwarg is not None:
        rendered = (
            f"**{node.args.kwarg.arg}"
        )

        annotation = safe_unparse(
            node.args.kwarg.annotation
        )

        if annotation:
            rendered += f": {annotation}"

        arguments.append(rendered)

    returns = safe_unparse(node.returns)

    return (
        f"{prefix} {node.name}"
        f"({', '.join(arguments)})"
        + (
            f" -> {returns}"
            if returns
            else ""
        )
    )


def render_ast_contracts(
    relative: str,
    source: str,
) -> list[str]:
    lines: list[str] = []

    try:
        tree = ast.parse(
            source,
            filename=relative,
        )
    except SyntaxError as exc:
        return [
            (
                "PARSE ERROR: "
                f"{exc.msg}, "
                f"line={exc.lineno}, "
                f"offset={exc.offset}"
            )
        ]

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            lines.append(
                f"class {node.name}"
            )

            bases = [
                safe_unparse(base)
                for base in node.bases
            ]

            lines.append(
                "  bases: "
                + (
                    ", ".join(bases)
                    if bases
                    else "(none)"
                )
            )

            decorators = [
                safe_unparse(item)
                for item in node.decorator_list
            ]

            lines.append(
                "  decorators: "
                + (
                    ", ".join(decorators)
                    if decorators
                    else "(none)"
                )
            )

            fields = class_fields(node)

            lines.append(
                "  fields: "
                + (
                    ", ".join(fields)
                    if fields
                    else "(none)"
                )
            )

            methods = [
                child
                for child in node.body
                if isinstance(
                    child,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                    ),
                )
            ]

            if methods:
                lines.append("  methods:")

                for method in methods:
                    lines.append(
                        "    "
                        + function_signature(method)
                    )
            else:
                lines.append(
                    "  methods: (none)"
                )

            lines.append("")

        elif isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            lines.append(
                function_signature(node)
            )

    return lines


def numbered_source(source: str) -> Iterable[str]:
    for line_number, line in enumerate(
        source.splitlines(),
        start=1,
    ):
        yield f"{line_number:>5}: {line}"


def main() -> None:
    report: list[str] = [
        "=" * 88,
        "LRP Project E-004 Exact Contract Export",
        "=" * 88,
        "",
        f"project_root: {ROOT}",
        f"python_version: {sys.version}",
        f"target_files: {len(TARGET_FILES)}",
        "",
    ]

    found_count = 0
    missing_count = 0

    for relative in TARGET_FILES:
        path = ROOT / relative

        report.extend(
            [
                "",
                "#" * 88,
                f"FILE: {relative}",
                "#" * 88,
                "",
            ]
        )

        if not path.exists():
            missing_count += 1
            report.append("STATUS: MISSING")
            continue

        found_count += 1
        source = read_source(path)

        report.extend(
            [
                "STATUS: FOUND",
                "",
                "-" * 88,
                "AST CONTRACT SUMMARY",
                "-" * 88,
                "",
            ]
        )

        report.extend(
            render_ast_contracts(
                relative,
                source,
            )
        )

        report.extend(
            [
                "",
                "-" * 88,
                "FULL SOURCE",
                "-" * 88,
                "",
            ]
        )

        report.extend(
            numbered_source(source)
        )

    report.extend(
        [
            "",
            "=" * 88,
            "EXPORT SUMMARY",
            "=" * 88,
            f"found: {found_count}",
            f"missing: {missing_count}",
            f"output: {OUTPUT_PATH}",
            "",
        ]
    )

    OUTPUT_PATH.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(
        "PASS: Project E E-004 contract export"
    )
    print(f"found: {found_count}")
    print(f"missing: {missing_count}")
    print(f"output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
