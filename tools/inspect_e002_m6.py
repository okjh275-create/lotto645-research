"""Inspect M6 adaptive-weight integration points for Project E."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import pkgutil
import sys
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SEARCH_TERMS = (
    "adaptive",
    "weight",
    "strategy",
    "ranking",
    "learning",
    "repository",
)


def heading(title: str) -> None:
    print()
    print("=" * 79)
    print(title)
    print("=" * 79)


def safe_signature(value: Any) -> str:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return "<signature unavailable>"


def module_matches(name: str) -> bool:
    lowered = name.lower()

    return any(
        term in lowered
        for term in SEARCH_TERMS
    )


def discover_lrp_modules() -> list[str]:
    import lrp

    names: list[str] = []

    for module_info in pkgutil.walk_packages(
        lrp.__path__,
        prefix="lrp.",
    ):
        if module_matches(module_info.name):
            names.append(module_info.name)

    return sorted(set(names))


def print_module(module: ModuleType) -> None:
    heading(f"MODULE: {module.__name__}")

    path = getattr(module, "__file__", None)
    print("file:", path)

    members = inspect.getmembers(module)

    classes = [
        (name, value)
        for name, value in members
        if inspect.isclass(value)
        and value.__module__ == module.__name__
    ]

    functions = [
        (name, value)
        for name, value in members
        if inspect.isfunction(value)
        and value.__module__ == module.__name__
    ]

    constants = [
        (name, value)
        for name, value in members
        if name.isupper()
        and not inspect.ismodule(value)
        and not inspect.isroutine(value)
        and not inspect.isclass(value)
    ]

    if classes:
        print()
        print("CLASSES")

        for name, value in classes:
            print(f"- {name}{safe_signature(value)}")

            annotations = getattr(
                value,
                "__annotations__",
                None,
            )

            if annotations:
                print(
                    "  annotations:",
                    annotations,
                )

            for method_name, method in inspect.getmembers(
                value,
                predicate=inspect.isfunction,
            ):
                if method_name.startswith("_"):
                    continue

                if method.__module__ != module.__name__:
                    continue

                print(
                    f"  method: "
                    f"{method_name}{safe_signature(method)}"
                )

    if functions:
        print()
        print("FUNCTIONS")

        for name, value in functions:
            print(f"- {name}{safe_signature(value)}")

    if constants:
        print()
        print("CONSTANTS")

        for name, value in constants:
            rendered = repr(value)

            if len(rendered) > 300:
                rendered = rendered[:297] + "..."

            print(f"- {name} = {rendered}")


def print_source_files() -> None:
    heading("SOURCE FILE CANDIDATES")

    candidates: list[Path] = []

    for path in ROOT.rglob("*.py"):
        lowered = str(path.relative_to(ROOT)).lower()

        if any(
            term in lowered
            for term in SEARCH_TERMS
        ):
            candidates.append(path)

    for path in sorted(candidates):
        print(path.relative_to(ROOT))


def print_database_files() -> None:
    heading("DATABASE / DATA FILE CANDIDATES")

    suffixes = {
        ".db",
        ".sqlite",
        ".sqlite3",
        ".json",
        ".jsonl",
        ".csv",
    }

    candidates: list[Path] = []

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in suffixes:
            continue

        lowered = str(path.relative_to(ROOT)).lower()

        if any(
            term in lowered
            for term in SEARCH_TERMS
        ):
            candidates.append(path)

    if not candidates:
        print("<none found by filename>")
        return

    for path in sorted(candidates):
        print(path.relative_to(ROOT))


def print_test_imports() -> None:
    heading("M6 TEST IMPORTS")

    test_names = (
        "test_m6_adaptive_weight.py",
        "test_m6_strategy_ranking.py",
        "test_m6_strategy_statistics.py",
        "test_m6_learning_foundation.py",
    )

    for test_name in test_names:
        path = ROOT / "tests" / test_name

        print()
        print(f"FILE: {path.relative_to(ROOT)}")

        if not path.exists():
            print("<missing>")
            continue

        for line in path.read_text(
            encoding="utf-8",
        ).splitlines():
            stripped = line.strip()

            if (
                stripped.startswith("from ")
                or stripped.startswith("import ")
            ):
                print(stripped)


def main() -> None:
    heading("PROJECT E E-002 M6 INVENTORY")
    print("root:", ROOT)

    print_source_files()
    print_database_files()
    print_test_imports()

    heading("DISCOVERED LRP MODULES")

    module_names = discover_lrp_modules()

    if not module_names:
        print("<none>")
        return

    for name in module_names:
        print(name)

    for name in module_names:
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            heading(f"MODULE IMPORT FAILED: {name}")
            print(type(exc).__name__, str(exc))
            continue

        print_module(module)


if __name__ == "__main__":
    main()
