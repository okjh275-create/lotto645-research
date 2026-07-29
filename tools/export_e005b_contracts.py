from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
import platform
from zoneinfo import ZoneInfo


ROOT = Path.cwd()
KST = ZoneInfo("Asia/Seoul")

TARGETS = (
    "lrp/learning/adaptive_engine.py",
    "lrp/learning/adaptive_models.py",
    "lrp/learning/adaptive_repository.py",
    "lrp/learning/learning_facade.py",
    "lrp/learning/service.py",
    "lrp/learning/performance.py",
    "lrp/learning/ranking.py",
    "lrp/learning/ranking_repository.py",
    "lrp/learning/__init__.py",
    "tests/test_m6_adaptive_weights.py",
    "tests/test_e005a_performance_analyzer.py",
)


def signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    arguments = []

    positional = list(node.args.posonlyargs) + list(node.args.args)
    defaults = [None] * (
        len(positional) - len(node.args.defaults)
    ) + list(node.args.defaults)

    for argument, default in zip(positional, defaults):
        text = argument.arg

        if argument.annotation is not None:
            text += ": " + ast.unparse(argument.annotation)

        if default is not None:
            text += " = " + ast.unparse(default)

        arguments.append(text)

    if node.args.vararg is not None:
        arguments.append("*" + node.args.vararg.arg)
    elif node.args.kwonlyargs:
        arguments.append("*")

    for argument, default in zip(
        node.args.kwonlyargs,
        node.args.kw_defaults,
    ):
        text = argument.arg

        if argument.annotation is not None:
            text += ": " + ast.unparse(argument.annotation)

        if default is not None:
            text += " = " + ast.unparse(default)

        arguments.append(text)

    if node.args.kwarg is not None:
        arguments.append("**" + node.args.kwarg.arg)

    result = f"{node.name}({', '.join(arguments)})"

    if node.returns is not None:
        result += " -> " + ast.unparse(node.returns)

    return result


def ast_summary(source: str) -> list[str]:
    tree = ast.parse(source)
    lines: list[str] = []

    for node in tree.body:
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            lines.append("def " + signature(node))

        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(
                ast.unparse(base)
                for base in node.bases
            ) or "(none)"

            decorators = ", ".join(
                ast.unparse(item)
                for item in node.decorator_list
            ) or "(none)"

            lines.append("")
            lines.append(f"class {node.name}")
            lines.append(f"  bases: {bases}")
            lines.append(f"  decorators: {decorators}")

            for item in node.body:
                if isinstance(
                    item,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    lines.append(
                        "  method: " + signature(item)
                    )

    return lines or ["(no public AST contracts found)"]


output = ROOT / "project_e005b_contracts.txt"

with output.open("w", encoding="utf-8") as stream:
    stream.write("=" * 88 + "\n")
    stream.write(
        "LRP Project E-005B Exact Contract Export\n"
    )
    stream.write("=" * 88 + "\n\n")
    stream.write(
        "generated_at_kst: "
        + datetime.now(KST).isoformat(
            timespec="seconds"
        )
        + "\n"
    )
    stream.write(f"project_root: {ROOT}\n")
    stream.write(
        f"python_version: {platform.python_version()}\n"
    )
    stream.write(f"target_files: {len(TARGETS)}\n")

    for relative in TARGETS:
        path = ROOT / relative

        stream.write("\n\n")
        stream.write("#" * 88 + "\n")
        stream.write(f"FILE: {relative}\n")
        stream.write("#" * 88 + "\n\n")

        if not path.exists():
            stream.write("STATUS: MISSING\n")
            continue

        source = path.read_text(encoding="utf-8")

        stream.write("STATUS: FOUND\n\n")
        stream.write("-" * 88 + "\n")
        stream.write("AST CONTRACT SUMMARY\n")
        stream.write("-" * 88 + "\n\n")

        try:
            for line in ast_summary(source):
                stream.write(line + "\n")
        except SyntaxError as exc:
            stream.write(
                f"AST ERROR: {type(exc).__name__}: {exc}\n"
            )

        stream.write("\n")
        stream.write("-" * 88 + "\n")
        stream.write("FULL SOURCE\n")
        stream.write("-" * 88 + "\n\n")

        for number, line in enumerate(
            source.splitlines(),
            start=1,
        ):
            stream.write(f"{number:5}: {line}\n")

print(output.resolve())
print(f"bytes: {output.stat().st_size}")
