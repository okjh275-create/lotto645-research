"""Fast development environment checks for LRP."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sqlite3
import sys
import time


REQUIRED_PATHS = (
    "config.yaml",
    "lrp",
    "lrp/contracts",
    "lrp/core",
    "lrp/adapters",
    "lrp/pipelines",
    "tests",
    "tools",
    "tools/lrp_build.py",
)

REQUIRED_MODULES = (
    "lrp",
    "lrp.contracts",
    "lrp.core",
    "lrp.adapters",
    "lrp.pipelines",
)


def main() -> int:
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    checks: list[dict[str, object]] = []

    def record(
        name: str,
        passed: bool,
        detail: str,
        elapsed: float = 0.0,
    ) -> None:
        checks.append(
            {
                "name": name,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
                "elapsed_seconds": round(elapsed, 6),
            }
        )

    for relative in REQUIRED_PATHS:
        target = root / relative
        record(
            f"path:{relative}",
            target.exists(),
            str(target),
        )

    for module_name in REQUIRED_MODULES:
        check_started = time.perf_counter()

        try:
            importlib.import_module(module_name)
        except Exception as exc:
            record(
                f"import:{module_name}",
                False,
                f"{type(exc).__name__}: {exc}",
                time.perf_counter() - check_started,
            )
        else:
            record(
                f"import:{module_name}",
                True,
                "importable",
                time.perf_counter() - check_started,
            )

    sqlite_started = time.perf_counter()

    try:
        connection = sqlite3.connect(":memory:")
        value = connection.execute("SELECT 1").fetchone()
        connection.close()

        if value != (1,):
            raise RuntimeError(
                f"unexpected sqlite result: {value}"
            )
    except Exception as exc:
        record(
            "sqlite",
            False,
            f"{type(exc).__name__}: {exc}",
            time.perf_counter() - sqlite_started,
        )
    else:
        record(
            "sqlite",
            True,
            sqlite3.sqlite_version,
            time.perf_counter() - sqlite_started,
        )

    failures = [
        item
        for item in checks
        if item["status"] == "FAIL"
    ]

    result = {
        "schema_version": "1.0",
        "python": sys.version.split()[0],
        "project_root": str(root),
        "status": "PASS" if not failures else "FAIL",
        "failure_count": len(failures),
        "elapsed_seconds": round(
            time.perf_counter() - started,
            6,
        ),
        "checks": checks,
    }

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
