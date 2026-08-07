"""Fast, read-only platform diagnostics."""

from __future__ import annotations

from datetime import datetime
import importlib
from pathlib import Path
import platform
import sqlite3
import sys
import tempfile
from typing import Any
from zoneinfo import ZoneInfo

from lrp.operations import RoundCompletionRepository


_KST = ZoneInfo("Asia/Seoul")


def run_doctor(
    *,
    project_root: str | Path = ".",
    round_completion: bool = False,
    snapshots_root: str | Path = "snapshots",
    round_limit: int = 20,
) -> dict[str, Any]:
    """Run lightweight diagnostics without prediction workloads."""
    root = Path(project_root).resolve()
    checks: list[dict[str, str]] = []

    def add(
        name: str,
        passed: bool,
        detail: str,
    ) -> None:
        checks.append(
            {
                "name": name,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    add(
        "python_version",
        sys.version_info >= (3, 11),
        platform.python_version(),
    )

    for relative in (
        "config.yaml",
        "lrp",
        "tests",
        "tools",
    ):
        target = root / relative
        add(
            f"path:{relative}",
            target.exists(),
            str(target),
        )

    for module_name in (
        "lrp",
        "lrp.contracts",
        "lrp.core",
        "lrp.adapters",
        "lrp.pipelines",
        "lrp.io",
        "lrp.operations",
        "lrp.management",
    ):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            add(
                f"import:{module_name}",
                False,
                f"{type(exc).__name__}: {exc}",
            )
        else:
            add(
                f"import:{module_name}",
                True,
                "importable",
            )

    probe_dir = root / "build" / "doctor_probe"

    try:
        probe_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        with tempfile.NamedTemporaryFile(
            dir=probe_dir,
            delete=True,
        ):
            pass
    except OSError as exc:
        add(
            "filesystem_write",
            False,
            str(exc),
        )
    else:
        add(
            "filesystem_write",
            True,
            str(probe_dir),
        )

    try:
        connection = sqlite3.connect(":memory:")
        connection.execute("SELECT 1").fetchone()
        connection.close()
    except sqlite3.Error as exc:
        add(
            "sqlite",
            False,
            str(exc),
        )
    else:
        add(
            "sqlite",
            True,
            sqlite3.sqlite_version,
        )

    if round_completion:
        if (
            isinstance(round_limit, bool)
            or not isinstance(round_limit, int)
        ):
            raise TypeError(
                "round_limit must be an integer"
            )

        if round_limit < 1:
            raise ValueError(
                "round_limit must be greater than or equal to 1"
            )

        snapshots = Path(snapshots_root)

        if snapshots.is_absolute():
            completion_root = (
                snapshots / "round-completion"
            )
        else:
            completion_root = (
                root
                / snapshots
                / "round-completion"
            )

        repository = RoundCompletionRepository(
            completion_root
        )

        rounds = repository.list_rounds()

        if not completion_root.exists():
            add(
                "round_completion:repository",
                True,
                "repository not present",
            )
        elif not rounds:
            add(
                "round_completion:repository",
                False,
                "repository exists but has no valid rounds",
            )
        else:
            selected = rounds[-round_limit:]

            add(
                "round_completion:repository",
                True,
                (
                    f"{len(rounds)} valid rounds, "
                    f"checking {len(selected)}"
                ),
            )

            for round_no in selected:
                verification = (
                    repository.verify_round(
                        round_no
                    )
                )

                passed = (
                    verification.get("status")
                    == "PASS"
                )

                detail = (
                    "manifest verified"
                    if passed
                    else str(
                        verification.get(
                            "reason",
                            verification.get(
                                "failures",
                                "verification failed",
                            ),
                        )
                    )
                )

                add(
                    f"round_completion:{round_no}",
                    passed,
                    detail,
                )
    failure_count = sum(
        item["status"] == "FAIL"
        for item in checks
    )

    return {
        "schema_version": "1.0",
        "checked_at_kst": datetime.now(_KST).isoformat(
            timespec="seconds"
        ),
        "project_root": str(root),
        "status": (
            "PASS"
            if failure_count == 0
            else "FAIL"
        ),
        "failure_count": failure_count,
        "checks": checks,
    }
