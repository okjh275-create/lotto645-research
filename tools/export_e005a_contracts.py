"""Export exact contracts required by Project E E-005A."""

from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
import platform
from zoneinfo import ZoneInfo


_KST = ZoneInfo("Asia/Seoul")

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "project_e005a_contracts.txt"

TARGETS = (
    "lrp/learning/__init__.py",
    "lrp/learning/models.py",
    "lrp/learning/repository.py",
    "lrp/learning/review.py",
    "lrp/learning/service.py",
    "lrp/learning/aggregator.py",
    "lrp/learning/strategy_stats.py",
    "lrp/learning/ranking.py",
    "lrp/learning/adaptive.py",
    "lrp/ensemble/repository.py",
    "lrp/ensemble/snapshot.py",
    "lrp/ensemble/features.py",
    "lrp/ensemble/rescoring.py",
    "tests/test_m6_learning_foundation.py",
    "tests/test_m6_automatic_review.py",
    "tests/test_m6_strategy_statistics.py",
    "tests/test_m6_strategy_ranking.py",
    "tests/test_m6_adaptive_weights.py",
    "tests/test_e002_learning_snapshot.py",
    "tests/test_e003_feature_rescoring.py",
    "tests/test_e004_pipeline_integration.py",
)


def annotation_text(
    node: ast.expr | None,
) -> str:
    if node is None:
        return "Any"

    try:
        return ast.unparse(node)
    except Exception:
        return "<unavailable>"


def function_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    parts: list[str] = []

    positional = [
        *node.args.posonlyargs,
        *node.args.args,
    ]

    default_offset = (
        len(positional)
        - len(node.args.defaults)
    )

    for index, argument in enumerate(positional):
        text = argument.arg

        if argument.annotation is not None:
            text += (
                ": "
                + annotation_text(
                    argument.annotation
                )
            )

        default_index = index - default_offset

        if default_index >= 0:
            try:
                default_text = ast.unparse(
                    node.args.defaults[
                        default_index
                    ]
                )
            except Exception:
                default_text = "<default>"

            text += f" = {default_text}"

        parts.append(text)

    if node.args.vararg is not None:
        text = "*" + node.args.vararg.arg

        if node.args.vararg.annotation is not None:
            text += (
                ": "
                + annotation_text(
                    node.args.vararg.annotation
                )
            )

        parts.append(text)
    elif node.args.kwonlyargs:
        parts.append("*")

    for argument, default in zip(
        node.args.kwonlyargs,
        node.args.kw_defaults,
    ):
        text = argument.arg

        if argument.annotation is not None:
            text += (
                ": "
                + annotation_text(
                    argument.annotation
                )
            )

        if default is not None:
            try:
                text += " = " + ast.unparse(default)
            except Exception:
                text += " = <default>"

        parts.append(text)

    if node.args.kwarg is not None:
        text = "**" + node.args.kwarg.arg

        if node.args.kwarg.annotation is not None:
            text += (
                ": "
                + annotation_text(
                    node.args.kwarg.annotation
                )
            )

        parts.append(text)

    return_type = annotation_text(node.returns)

    return (
        f"{node.name}("
        + ", ".join(parts)
        + f") -> {return_type}"
    )


def class_summary(
    node: ast.ClassDef,
) -> list[str]:
    lines = [f"class {node.name}"]

    bases = []

    for base in node.bases:
        try:
            bases.append(ast.unparse(base))
        except Exception:
            bases.append("<unknown>")

    lines.append(
        "  bases: "
        + (
            ", ".join(bases)
            if bases
            else "(none)"
        )
    )

    decorators = []

    for decorator in node.decorator_list:
        try:
            decorators.append(
                ast.unparse(decorator)
            )
        except Exception:
            decorators.append("<unknown>")

    lines.append(
        "  decorators: "
        + (
            ", ".join(decorators)
            if decorators
            else "(none)"
        )
    )

    fields: list[str] = []
    methods: list[str] = []

    for child in node.body:
        if isinstance(child, ast.AnnAssign):
            if isinstance(child.target, ast.Name):
                fields.append(
                    f"{child.target.id}: "
                    f"{annotation_text(child.annotation)}"
                )

        elif isinstance(
            child,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            methods.append(
                function_signature(child)
            )

    lines.append(
        "  fields:"
        if fields
        else "  fields: (none)"
    )

    lines.extend(
        f"    {field}"
        for field in fields
    )

    lines.append(
        "  methods:"
        if methods
        else "  methods: (none)"
    )

    lines.extend(
        f"    {method}"
        for method in methods
    )

    return lines


def module_summary(
    tree: ast.Module,
) -> list[str]:
    lines: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            lines.extend(class_summary(node))
            lines.append("")

        elif isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            lines.append(
                "def " + function_signature(node)
            )

    if not lines:
        lines.append("(no public AST contracts found)")

    return lines


def numbered_source(
    text: str,
) -> list[str]:
    return [
        f"{index:5d}: {line}"
        for index, line
        in enumerate(
            text.splitlines(),
            start=1,
        )
    ]


def export_file(
    relative: str,
) -> list[str]:
    path = ROOT / relative

    lines = [
        "#" * 88,
        f"FILE: {relative}",
        "#" * 88,
        "",
    ]

    if not path.is_file():
        lines.extend(
            [
                "STATUS: MISSING",
                "",
            ]
        )
        return lines

    text = path.read_text(
        encoding="utf-8-sig"
    )

    lines.extend(
        [
            "STATUS: FOUND",
            "",
            "-" * 88,
            "AST CONTRACT SUMMARY",
            "-" * 88,
            "",
        ]
    )

    try:
        tree = ast.parse(
            text,
            filename=str(path),
        )
    except SyntaxError as exc:
        lines.extend(
            [
                "PARSE ERROR:",
                str(exc),
            ]
        )
    else:
        lines.extend(module_summary(tree))

    lines.extend(
        [
            "",
            "-" * 88,
            "FULL SOURCE",
            "-" * 88,
            "",
            *numbered_source(text),
            "",
        ]
    )

    return lines


def main() -> None:
    lines = [
        "=" * 88,
        "LRP Project E-005A Exact Contract Export",
        "=" * 88,
        "",
        f"generated_at_kst: "
        f"{datetime.now(_KST).isoformat(timespec='seconds')}",
        f"project_root: {ROOT}",
        f"python_version: {platform.python_version()}",
        f"target_files: {len(TARGETS)}",
        "",
    ]

    found = 0
    missing = 0

    for relative in TARGETS:
        if (ROOT / relative).is_file():
            found += 1
        else:
            missing += 1

        lines.extend(export_file(relative))

    lines.extend(
        [
            "=" * 88,
            "E-005A CONTRACT QUESTIONS",
            "=" * 88,
            "",
            "Q1. What is the exact PredictionRecord contract?",
            "Q2. What is the exact ReviewRecord contract?",
            "Q3. How are model and scenario strategy keys resolved?",
            "Q4. Which repository method exposes bounded history?",
            "Q5. Is history returned newest-first or chronological?",
            "Q6. Which fields represent confidence, stability, and trend?",
            "Q7. What existing score normalization is already applied?",
            "Q8. What is the lowest-risk read-only analyzer insertion point?",
            "Q9. Which existing public exports must remain compatible?",
            "Q10. Which tests must be added to the full build runner?",
            "",
            "=" * 88,
            "EXPORT SUMMARY",
            "=" * 88,
            "",
            f"found: {found}",
            f"missing: {missing}",
            f"output: {OUTPUT}",
            "",
        ]
    )

    OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        "PASS: Project E E-005A contract export"
    )
    print(f"found: {found}")
    print(f"missing: {missing}")
    print(f"output: {OUTPUT}")


if __name__ == "__main__":
    main()
