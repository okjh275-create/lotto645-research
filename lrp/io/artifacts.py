"""Atomic prediction artifact and manifest storage."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
from zoneinfo import ZoneInfo


_KST = ZoneInfo("Asia/Seoul")


def _json_bytes(
    payload: Mapping[str, Any],
) -> bytes:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
    )
    return (text + "\n").encode("utf-8")


def atomic_write_bytes(
    path: str | Path,
    content: bytes,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )

    temporary = Path(temporary_name)

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
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def write_prediction_artifacts(
    payload: Mapping[str, Any],
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    round_no = int(payload["round"])
    seed = int(payload["seed"])

    directory = (
        Path(output_root)
        / f"round_{round_no:04d}"
        / f"seed_{seed}"
    )

    prediction_path = atomic_write_bytes(
        directory / "prediction.json",
        _json_bytes(payload),
    )

    manifest = {
        "schema_version": "1.0",
        "artifact_type": "lotto645_prediction",
        "round": round_no,
        "seed": seed,
        "created_at_kst": datetime.now(_KST).isoformat(
            timespec="seconds"
        ),
        "files": {
            "prediction.json": {
                "sha256": sha256_file(prediction_path),
                "bytes": prediction_path.stat().st_size,
            }
        },
        "metadata": dict(payload.get("metadata", {})),
    }

    manifest_path = atomic_write_bytes(
        directory / "manifest.json",
        _json_bytes(manifest),
    )

    return {
        "directory": str(directory.resolve()),
        "prediction_path": str(
            prediction_path.resolve()
        ),
        "manifest_path": str(
            manifest_path.resolve()
        ),
        "prediction_sha256": (
            manifest["files"]["prediction.json"]["sha256"]
        ),
    }
