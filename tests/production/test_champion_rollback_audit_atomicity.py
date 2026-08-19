from __future__ import annotations

import json
from pathlib import Path

import pytest

from lrp.production.champion_registry_publisher import (
    ProductionChampionRegistryPublisher,
)
from lrp.production.champion_rollback import (
    ChampionRollbackService,
)


def _write_decision(
    path: Path,
    *,
    model: str,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "selection": {
                    "selected_model":
                        model,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return path


def _publication_revision_id(
    registry_root: Path,
    *,
    source_sha256: str,
) -> str:
    matches = []

    for path in (
        registry_root
        / "history"
    ).glob(
        "*.json"
    ):
        if not path.is_file():
            continue

        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if (
            payload.get(
                "source_sha256"
            )
            == source_sha256
        ):
            matches.append(
                path.stem
            )

    assert len(matches) == 1

    return matches[0]


def _active_snapshot(
    registry_root: Path,
) -> dict[str, bytes]:
    active_root = (
        registry_root
        / "active"
    )

    return {
        path.name:
            path.read_bytes()
        for path in active_root.iterdir()
        if path.is_file()
    }


def test_audit_write_failure_restores_original_active_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = (
        tmp_path
        / "registry"
    )

    publisher = (
        ProductionChampionRegistryPublisher()
    )

    first = publisher.publish(
        source_decision=(
            _write_decision(
                tmp_path / "a.json",
                model="model-a",
            )
        ),
        registry_root=registry,
    )

    second = publisher.publish(
        source_decision=(
            _write_decision(
                tmp_path / "b.json",
                model="model-b",
            )
        ),
        registry_root=registry,
    )

    target_revision = (
        _publication_revision_id(
            registry,
            source_sha256=(
                first.source_sha256
            ),
        )
    )

    service = ChampionRollbackService(
        registry_root=registry
    )

    plan = service.plan(
        target_revision
    )

    active_before = (
        _active_snapshot(
            registry
        )
    )

    def fail_audit(
        *args,
        **kwargs,
    ):
        raise OSError(
            "simulated rollback audit failure"
        )

    monkeypatch.setattr(
        service,
        "_write_rollback_provenance",
        fail_audit,
    )

    with pytest.raises(
        OSError,
        match="simulated rollback audit failure",
    ):
        service.execute(
            plan
        )

    active_after = (
        _active_snapshot(
            registry
        )
    )

    assert (
        active_after
        == active_before
    )

    active_publication = json.loads(
        (
            registry
            / "active"
            / "publication.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        active_publication[
            "source_sha256"
        ]
        == second.source_sha256
    )

    rollback_root = (
        registry
        / "history"
        / "rollbacks"
    )

    if rollback_root.exists():
        assert list(
            rollback_root.glob(
                "*.json"
            )
        ) == []
