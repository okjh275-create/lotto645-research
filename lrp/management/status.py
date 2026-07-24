"""Fast, read-only platform status collection."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from lrp import PROJECT_NAME, __version__


_KST = ZoneInfo("Asia/Seoul")


def _files(root: Path, filename: str) -> tuple[Path, ...]:
    """Return matching files without reading their contents."""
    if not root.is_dir():
        return ()
    return tuple(root.rglob(filename))


def _latest_path(paths: tuple[Path, ...]) -> str | None:
    """Return the most recently modified path."""
    if not paths:
        return None

    latest = max(
        paths,
        key=lambda path: path.stat().st_mtime_ns,
    )
    return str(latest.resolve())


def collect_platform_status(
    *,
    project_root: str | Path = ".",
    predictions_root: str | Path = "predictions",
    snapshots_root: str | Path = "snapshots",
    backups_root: str | Path = "backups",
) -> dict[str, Any]:
    """Collect lightweight operational status.

    This function does not execute statistics, candidate generation,
    prediction, review, backup, or manifest verification.
    """
    root = Path(project_root).resolve()
    predictions = root / predictions_root
    snapshots = root / snapshots_root
    backups = root / backups_root

    prediction_files = _files(
        predictions,
        "prediction.json",
    )
    review_files = _files(
        snapshots,
        "review.json",
    )
    weekly_files = _files(
        snapshots,
        "weekly.json",
    )
    prediction_manifests = _files(
        predictions,
        "manifest.json",
    )
    snapshot_manifests = _files(
        snapshots,
        "manifest.json",
    )

    backup_files = (
        tuple(backups.glob("*.zip"))
        if backups.is_dir()
        else ()
    )

    required_paths = {
        "config": (root / "config.yaml").is_file(),
        "lrp": (root / "lrp").is_dir(),
        "tests": (root / "tests").is_dir(),
        "tools": (root / "tools").is_dir(),
    }

    return {
        "schema_version": "1.0",
        "project": PROJECT_NAME,
        "platform_version": __version__,
        "collected_at_kst": datetime.now(_KST).isoformat(
            timespec="seconds"
        ),
        "project_root": str(root),
        "status": (
            "PASS"
            if all(required_paths.values())
            else "FAIL"
        ),
        "required_paths": required_paths,
        "counts": {
            "predictions": len(prediction_files),
            "reviews": len(review_files),
            "weekly_snapshots": len(weekly_files),
            "manifests": (
                len(prediction_manifests)
                + len(snapshot_manifests)
            ),
            "backups": len(backup_files),
        },
        "latest": {
            "prediction": _latest_path(prediction_files),
            "review": _latest_path(review_files),
            "weekly_snapshot": _latest_path(weekly_files),
            "backup": _latest_path(backup_files),
        },
    }
