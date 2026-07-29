"""Inspect the Project D-to-Prediction pipeline for E-004 integration.

This script performs read-only static inspection. It does not modify
source files, databases, predictions, manifests, or build artifacts.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
import re
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]

SOURCE_ROOTS = (
    ROOT / "lrp",
    ROOT / "tests",
)

OUTPUT_PATH = (
    ROOT / "project_e004_pipeline_inventory.txt"
)

SOURCE_SUFFIXES = {".py"}


TARGET_TERMS = (
    "candidate",
    "ranking",
    "rank",
    "score",
    "diversity",
    "practical",
    "prediction",
    "orchestrator",
    "pipeline",
    "top5",
    "top10",
    "selected_sets",
    "generated_candidates",
    "normalized_score",
    "ensemble_score",
    "model_name",
    "scenario_name",
    "scenario_names",
)


@dataclass(slots=True)
class ClassRecord:
    file: str
    line: int
    name: str
    bases: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FunctionRecord:
    file: str
    line: int
    name: str
    arguments: list[str] = field(default_factory=list)
    returns: str = ""
    decorators: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AssignmentRecord:
    file: str
    line: int
    target: str
    expression: str


@dataclass(slots=True)
class ImportRecord:
    file: str
    line: int
    statement: str


@dataclass(slots=True)
class FileRecord:
    path: Path
    relative: str
    source: str
    tree: ast.AST | None
    parse_error: str | None


def iter_python_files() -> Iterable[Path]:
    seen: set[Path] = set()

    for source_root in SOURCE_ROOTS:
        if not source_root.exists():
            continue

        for path in source_root.rglob("*.py"):
            resolved = path.resolve()

            if resolved in seen:
                continue

            seen.add(resolved)
            yield path


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_file(path: Path) -> FileRecord:
    relative = relative_path(path)

    try:
        source = path.read_text(
            encoding="utf-8-sig"
        )
    except UnicodeDecodeError:
        source = path.read_text(
            encoding="cp949",
            errors="replace",
        )

    try:
        tree = ast.parse(
            source,
            filename=relative,
        )
        parse_error = None
    except SyntaxError as exc:
        tree = None
        parse_error = (
            f"{exc.__class__.__name__}: "
            f"{exc.msg} "
            f"(line={exc.lineno}, "
            f"offset={exc.offset})"
        )

    return FileRecord(
        path=path,
        relative=relative,
        source=source,
        tree=tree,
        parse_error=parse_error,
    )


def safe_unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""

    try:
        return ast.unparse(node)
    except Exception:
        return node.__class__.__name__


def decorator_names(
    decorators: list[ast.expr],
) -> list[str]:
    return [
        safe_unparse(item)
        for item in decorators
    ]


def argument_names(
    args: ast.arguments,
) -> list[str]:
    result: list[str] = []

    for arg in (
        *args.posonlyargs,
        *args.args,
        *args.kwonlyargs,
    ):
        annotation = safe_unparse(
            arg.annotation
        )

        if annotation:
            result.append(
                f"{arg.arg}: {annotation}"
            )
        else:
            result.append(arg.arg)

    if args.vararg is not None:
        result.append(
            f"*{args.vararg.arg}"
        )

    if args.kwarg is not None:
        result.append(
            f"**{args.kwarg.arg}"
        )

    return result


def assigned_names(
    node: ast.AST,
) -> list[str]:
    names: list[str] = []

    if isinstance(node, ast.Name):
        names.append(node.id)

    elif isinstance(node, ast.Attribute):
        names.append(
            safe_unparse(node)
        )

    elif isinstance(
        node,
        (ast.Tuple, ast.List),
    ):
        for element in node.elts:
            names.extend(
                assigned_names(element)
            )

    return names


def class_fields(
    node: ast.ClassDef,
) -> list[str]:
    fields: list[str] = []

    for child in node.body:
        if isinstance(child, ast.AnnAssign):
            fields.extend(
                assigned_names(child.target)
            )

        elif isinstance(child, ast.Assign):
            for target in child.targets:
                fields.extend(
                    assigned_names(target)
                )

        elif isinstance(
            child,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ) and child.name == "__init__":
            for init_child in ast.walk(child):
                if not isinstance(
                    init_child,
                    (ast.Assign, ast.AnnAssign),
                ):
                    continue

                targets = (
                    init_child.targets
                    if isinstance(
                        init_child,
                        ast.Assign,
                    )
                    else [init_child.target]
                )

                for target in targets:
                    for name in assigned_names(
                        target
                    ):
                        if name.startswith(
                            "self."
                        ):
                            fields.append(name)

    return sorted(set(fields))


def class_methods(
    node: ast.ClassDef,
) -> list[str]:
    return [
        child.name
        for child in node.body
        if isinstance(
            child,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        )
    ]


def is_relevant_text(text: str) -> bool:
    lowered = text.lower()

    return any(
        term in lowered
        for term in TARGET_TERMS
    )


def collect_classes(
    record: FileRecord,
) -> list[ClassRecord]:
    if record.tree is None:
        return []

    result: list[ClassRecord] = []

    for node in ast.walk(record.tree):
        if not isinstance(node, ast.ClassDef):
            continue

        rendered = " ".join(
            (
                node.name,
                *[
                    safe_unparse(base)
                    for base in node.bases
                ],
                *class_fields(node),
                *class_methods(node),
            )
        )

        if not is_relevant_text(rendered):
            continue

        result.append(
            ClassRecord(
                file=record.relative,
                line=node.lineno,
                name=node.name,
                bases=[
                    safe_unparse(base)
                    for base in node.bases
                ],
                decorators=decorator_names(
                    node.decorator_list
                ),
                fields=class_fields(node),
                methods=class_methods(node),
            )
        )

    return sorted(
        result,
        key=lambda item: (
            item.file,
            item.line,
            item.name,
        ),
    )


def collect_functions(
    record: FileRecord,
) -> list[FunctionRecord]:
    if record.tree is None:
        return []

    result: list[FunctionRecord] = []

    for node in ast.walk(record.tree):
        if not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue

        rendered = " ".join(
            (
                node.name,
                safe_unparse(node.returns),
                *argument_names(node.args),
            )
        )

        if not is_relevant_text(rendered):
            continue

        result.append(
            FunctionRecord(
                file=record.relative,
                line=node.lineno,
                name=node.name,
                arguments=argument_names(
                    node.args
                ),
                returns=safe_unparse(
                    node.returns
                ),
                decorators=decorator_names(
                    node.decorator_list
                ),
            )
        )

    return sorted(
        result,
        key=lambda item: (
            item.file,
            item.line,
            item.name,
        ),
    )


def collect_assignments(
    record: FileRecord,
) -> list[AssignmentRecord]:
    if record.tree is None:
        return []

    result: list[AssignmentRecord] = []

    for node in ast.walk(record.tree):
        targets: list[ast.AST]
        value: ast.AST | None

        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue

        for target_node in targets:
            target = safe_unparse(
                target_node
            )

            if not is_relevant_text(target):
                continue

            expression = safe_unparse(value)

            if len(expression) > 500:
                expression = (
                    expression[:497] + "..."
                )

            result.append(
                AssignmentRecord(
                    file=record.relative,
                    line=getattr(
                        node,
                        "lineno",
                        0,
                    ),
                    target=target,
                    expression=expression,
                )
            )

    return sorted(
        result,
        key=lambda item: (
            item.file,
            item.line,
            item.target,
        ),
    )


def collect_imports(
    record: FileRecord,
) -> list[ImportRecord]:
    if record.tree is None:
        return []

    result: list[ImportRecord] = []

    for node in ast.walk(record.tree):
        if not isinstance(
            node,
            (ast.Import, ast.ImportFrom),
        ):
            continue

        statement = safe_unparse(node)

        if not is_relevant_text(statement):
            continue

        result.append(
            ImportRecord(
                file=record.relative,
                line=node.lineno,
                statement=statement,
            )
        )

    return sorted(
        result,
        key=lambda item: (
            item.file,
            item.line,
            item.statement,
        ),
    )


def matching_lines(
    record: FileRecord,
    *,
    maximum: int = 250,
) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []

    pattern = re.compile(
        "|".join(
            re.escape(term)
            for term in TARGET_TERMS
        ),
        flags=re.IGNORECASE,
    )

    for line_number, line in enumerate(
        record.source.splitlines(),
        start=1,
    ):
        if not pattern.search(line):
            continue

        stripped = line.strip()

        if not stripped:
            continue

        matches.append(
            (line_number, stripped)
        )

        if len(matches) >= maximum:
            break

    return matches


def render_list(
    values: Iterable[str],
) -> str:
    rendered = list(values)

    if not rendered:
        return "(none)"

    return ", ".join(rendered)


def write_report(
    records: list[FileRecord],
) -> None:
    classes = [
        item
        for record in records
        for item in collect_classes(record)
    ]

    functions = [
        item
        for record in records
        for item in collect_functions(record)
    ]

    assignments = [
        item
        for record in records
        for item in collect_assignments(record)
    ]

    imports = [
        item
        for record in records
        for item in collect_imports(record)
    ]

    relevant_files = [
        record
        for record in records
        if (
            is_relevant_text(record.relative)
            or is_relevant_text(
                record.source
            )
        )
    ]

    lines: list[str] = []

    lines.extend(
        [
            "=" * 78,
            "LRP Project E-004 Pipeline Inventory",
            "=" * 78,
            "",
            f"project_root: {ROOT}",
            f"python_version: {sys.version}",
            f"python_files_scanned: {len(records)}",
            f"relevant_files: {len(relevant_files)}",
            f"relevant_classes: {len(classes)}",
            f"relevant_functions: {len(functions)}",
            f"relevant_assignments: {len(assignments)}",
            "",
        ]
    )

    parse_errors = [
        record
        for record in records
        if record.parse_error is not None
    ]

    lines.extend(
        [
            "-" * 78,
            "1. Parse Status",
            "-" * 78,
        ]
    )

    if not parse_errors:
        lines.append("PASS: all scanned files parsed")
    else:
        for record in parse_errors:
            lines.append(
                f"ERROR: {record.relative}: "
                f"{record.parse_error}"
            )

    lines.append("")

    lines.extend(
        [
            "-" * 78,
            "2. Relevant Files",
            "-" * 78,
        ]
    )

    for record in sorted(
        relevant_files,
        key=lambda item: item.relative,
    ):
        lines.append(record.relative)

    lines.append("")

    lines.extend(
        [
            "-" * 78,
            "3. Candidate / Ranking / Pipeline Classes",
            "-" * 78,
        ]
    )

    if not classes:
        lines.append("(none)")
    else:
        for item in classes:
            lines.extend(
                [
                    (
                        f"{item.file}:{item.line} "
                        f"class {item.name}"
                    ),
                    (
                        "  bases: "
                        + render_list(item.bases)
                    ),
                    (
                        "  decorators: "
                        + render_list(
                            item.decorators
                        )
                    ),
                    (
                        "  fields: "
                        + render_list(item.fields)
                    ),
                    (
                        "  methods: "
                        + render_list(item.methods)
                    ),
                    "",
                ]
            )

    lines.extend(
        [
            "-" * 78,
            "4. Candidate / Ranking / Pipeline Functions",
            "-" * 78,
        ]
    )

    if not functions:
        lines.append("(none)")
    else:
        for item in functions:
            lines.extend(
                [
                    (
                        f"{item.file}:{item.line} "
                        f"def {item.name}"
                    ),
                    (
                        "  arguments: "
                        + render_list(
                            item.arguments
                        )
                    ),
                    (
                        "  returns: "
                        + (
                            item.returns
                            or "(none)"
                        )
                    ),
                    (
                        "  decorators: "
                        + render_list(
                            item.decorators
                        )
                    ),
                    "",
                ]
            )

    lines.extend(
        [
            "-" * 78,
            "5. Relevant Imports",
            "-" * 78,
        ]
    )

    if not imports:
        lines.append("(none)")
    else:
        for item in imports:
            lines.append(
                f"{item.file}:{item.line} "
                f"{item.statement}"
            )

    lines.append("")

    lines.extend(
        [
            "-" * 78,
            "6. Score / Ranking / Selection Assignments",
            "-" * 78,
        ]
    )

    if not assignments:
        lines.append("(none)")
    else:
        for item in assignments:
            lines.extend(
                [
                    (
                        f"{item.file}:{item.line} "
                        f"{item.target}"
                    ),
                    (
                        "  expression: "
                        + (
                            item.expression
                            or "(none)"
                        )
                    ),
                ]
            )

    lines.append("")

    lines.extend(
        [
            "-" * 78,
            "7. Source Evidence by File",
            "-" * 78,
        ]
    )

    for record in sorted(
        relevant_files,
        key=lambda item: item.relative,
    ):
        matches = matching_lines(record)

        if not matches:
            continue

        lines.append("")
        lines.append(
            f"[{record.relative}]"
        )

        for line_number, text in matches:
            lines.append(
                f"{line_number:>5}: {text}"
            )

    lines.extend(
        [
            "",
            "-" * 78,
            "8. E-004 Questions to Resolve",
            "-" * 78,
            (
                "Q1. What is the concrete Project D "
                "candidate class?"
            ),
            (
                "Q2. Which candidate field is the "
                "authoritative base score?"
            ),
            (
                "Q3. Where is candidate sorting or "
                "ranking performed?"
            ),
            (
                "Q4. Where is diversity compression "
                "performed?"
            ),
            (
                "Q5. Where are Top10 and Top5 "
                "selected?"
            ),
            (
                "Q6. Does a candidate carry model or "
                "scenario provenance?"
            ),
            (
                "Q7. Can ranking accept a wrapper, or "
                "must E-004 expose the original "
                "candidate interface?"
            ),
            (
                "Q8. What is the lowest-risk insertion "
                "point between Project D scoring and "
                "diversity selection?"
            ),
            "",
            "=" * 78,
            "END",
            "=" * 78,
            "",
        ]
    )

    OUTPUT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    records = [
        load_file(path)
        for path in sorted(
            iter_python_files()
        )
    ]

    write_report(records)

    print(
        "PASS: Project E E-004 pipeline inventory"
    )
    print(
        f"files_scanned: {len(records)}"
    )
    print(
        f"output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
