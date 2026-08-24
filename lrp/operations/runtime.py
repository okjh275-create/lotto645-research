"""Operational snapshots, manifests, verification, and backup."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any, Mapping
from zoneinfo import ZoneInfo
import hashlib
import json
import os
import tempfile
import zipfile

_KST = ZoneInfo("Asia/Seoul")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def atomic_write(path: str | Path, content: bytes) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_operation_artifact(
    payload: Mapping[str, Any],
    *,
    output_root: str | Path,
    artifact_type: str,
    round_no: int,
    filename: str,
    artifact_key: str | None = None,
) -> dict[str, Any]:
    directory = Path(output_root) / artifact_type / f"round_{round_no:04d}"
    if artifact_key is not None:
        if not isinstance(artifact_key, str):
            raise ValueError("artifact_key must be str or None")
        if (not artifact_key or len(artifact_key) > 128 or artifact_key in {".", ".."} or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", artifact_key) is None):
            raise ValueError("invalid artifact_key")
        directory = directory / artifact_key
    data_path = atomic_write(directory / filename, _json_bytes(payload))
    manifest = {
        "schema_version": "1.0",
        "artifact_type": artifact_type,
        "round": round_no,
        "created_at_kst": datetime.now(_KST).isoformat(timespec="seconds"),
        "files": {
            filename: {
                "sha256": sha256_file(data_path),
                "bytes": data_path.stat().st_size,
            }
        },
    }
    if artifact_key is not None:
        manifest["artifact_key"] = artifact_key
    manifest_path = atomic_write(directory / "manifest.json", _json_bytes(manifest))
    append_operation_log(
        Path(output_root) / "operation_log.jsonl",
        {
            "at_kst": datetime.now(_KST).isoformat(timespec="seconds"),
            "artifact_type": artifact_type,
            "round": round_no,
            "status": "PASS",
            "path": str(data_path.resolve()),
        },
    )
    return {
        "directory": str(directory.resolve()),
        "data_path": str(data_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "sha256": manifest["files"][filename]["sha256"],
    }


def append_operation_log(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")) + "\n"
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()


def verify_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    failures: list[dict[str, Any]] = []
    checked: list[str] = []
    for name, metadata in manifest.get("files", {}).items():
        target = manifest_path.parent / name
        checked.append(str(target))
        if not target.is_file():
            failures.append({"file": name, "reason": "missing"})
            continue
        actual = sha256_file(target)
        expected = str(metadata.get("sha256", ""))
        if actual != expected:
            failures.append({"file": name, "reason": "sha256_mismatch", "expected": expected, "actual": actual})
    return {
        "status": "PASS" if not failures else "FAIL",
        "manifest": str(manifest_path.resolve()),
        "checked": checked,
        "failures": failures,
    }


def create_backup(
    *,
    source_root: str | Path,
    destination_root: str | Path,
    label: str = "lrp",
) -> dict[str, Any]:
    source = Path(source_root).resolve()
    destination = Path(destination_root)
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(_KST).strftime("%Y%m%d_%H%M%S")
    archive = destination / f"{label}_{stamp}.zip"
    excluded_parts = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            if any(part in excluded_parts for part in relative.parts):
                continue
            if archive.resolve() == path.resolve():
                continue
            bundle.write(path, relative.as_posix())
    return {
        "status": "PASS",
        "archive": str(archive.resolve()),
        "sha256": sha256_file(archive),
        "bytes": archive.stat().st_size,
    }


def restore_backup(
    *,
    archive_path: str | Path,
    destination_root: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Safely restore a ZIP backup into a destination directory."""
    archive = Path(archive_path).resolve()
    destination = Path(destination_root).resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    destination.mkdir(parents=True, exist_ok=True)

    restored: list[str] = []
    skipped: list[str] = []
    with zipfile.ZipFile(archive, "r") as bundle:
        for member in bundle.infolist():
            if member.is_dir():
                continue
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe archive member: {member.filename}")
            target = (destination / relative).resolve()
            try:
                target.relative_to(destination)
            except ValueError as exc:
                raise ValueError(f"unsafe archive member: {member.filename}") from exc
            if target.exists() and not overwrite:
                skipped.append(relative.as_posix())
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member, "r") as source, target.open("wb") as output:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
            restored.append(relative.as_posix())

    return {
        "status": "PASS",
        "archive": str(archive),
        "destination": str(destination),
        "restored_count": len(restored),
        "skipped_count": len(skipped),
        "restored": restored,
        "skipped": skipped,
    }
