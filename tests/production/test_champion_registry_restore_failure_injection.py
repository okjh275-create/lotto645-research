from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lrp.production.champion_registry_recovery import (
    ProductionRegistryBackupService,
    ProductionRegistryRestoreService,
)


def _write_registry(
    root: Path,
    *,
    model: str,
    revision_digit: str,
) -> Path:
    registry = root / "registry"

    decision_payload = {
        "selected_model":
            model,

        "source_sha256":
            "1" * 64,
    }

    decision_raw = (
        json.dumps(
            decision_payload,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")

    decision_sha = (
        hashlib.sha256(
            decision_raw
        )
        .hexdigest()
    )

    revision_id = (
        revision_digit
        * 64
    )

    publication = {
        "selected_model":
            model,

        "source_sha256":
            decision_sha,

        "revision_id":
            revision_id,
    }

    active = (
        registry
        / "active"
    )

    active.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        active
        / "champion_decision.json"
    ).write_bytes(
        decision_raw
    )

    (
        active
        / "publication.json"
    ).write_text(
        json.dumps(
            publication,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    history = (
        registry
        / "history"
    )

    history.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        history
        / f"{revision_id}.json"
    ).write_text(
        json.dumps(
            publication,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    decisions = (
        history
        / "decisions"
    )

    decisions.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        decisions
        / f"{decision_sha}.json"
    ).write_bytes(
        decision_raw
    )

    rollbacks = (
        history
        / "rollbacks"
    )

    rollbacks.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        rollbacks
        / "rollback-001.json"
    ).write_text(
        json.dumps(
            {
                "event":
                    "rollback",

                "revision_id":
                    revision_id,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return registry

def _tree(
    root: Path,
) -> dict[str, bytes]:
    if not root.exists():
        return {}

    result: dict[str, bytes] = {}

    for path in sorted(
        root.rglob("*")
    ):
        if not path.is_file():
            continue

        relative = (
            path.relative_to(root)
            .as_posix()
        )

        if relative == ".writer.lock":
            continue

        result[
            relative
        ] = path.read_bytes()

    return result


def _make_backup(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
]:
    source = _write_registry(
        tmp_path / "source",
        model="x13-source",
        revision_digit="7",
    )

    result = (
        ProductionRegistryBackupService(
            source
        )
        .backup(
            tmp_path / "backups"
        )
    )

    return (
        source,
        result.backup_root,
    )


def test_restore_delete_failure_compensates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, backup_root = (
        _make_backup(
            tmp_path
        )
    )

    destination = _write_registry(
        tmp_path / "destination",
        model="x13-original",
        revision_digit="9",
    )

    before = _tree(
        destination
    )

    original_unlink = (
        Path.unlink
    )

    injected = False

    def fail_one_delete(
        self: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal injected

        try:
            relative = (
                self.relative_to(
                    destination
                )
                .as_posix()
            )

        except ValueError:
            return original_unlink(
                self,
                *args,
                **kwargs,
            )

        if (
            not injected
            and relative != ".writer.lock"
        ):
            injected = True

            raise OSError(
                "injected restore delete failure"
            )

        return original_unlink(
            self,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        Path,
        "unlink",
        fail_one_delete,
    )

    with pytest.raises(
        OSError,
        match="injected restore delete failure",
    ):
        ProductionRegistryRestoreService(
            destination
        ).restore(
            backup_root
        )

    assert _tree(
        destination
    ) == before


def test_post_restore_verification_failure_compensates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, backup_root = (
        _make_backup(
            tmp_path
        )
    )

    destination = _write_registry(
        tmp_path / "destination",
        model="x13-original",
        revision_digit="9",
    )

    before = _tree(
        destination
    )

    service = (
        ProductionRegistryRestoreService(
            destination
        )
    )

    original_verify = (
        service._verify_restored
    )

    calls = 0

    def fail_verify(
        payload: dict[str, bytes],
    ) -> None:
        nonlocal calls

        calls += 1

        if calls == 1:
            raise RuntimeError(
                "injected post-restore verification failure"
            )

        original_verify(
            payload
        )

    monkeypatch.setattr(
        service,
        "_verify_restored",
        fail_verify,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "injected post-restore "
            "verification failure"
        ),
    ):
        service.restore(
            backup_root
        )

    assert _tree(
        destination
    ) == before
