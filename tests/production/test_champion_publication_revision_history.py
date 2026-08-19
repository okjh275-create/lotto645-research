from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lrp.production.champion_registry_publisher import (
    ProductionChampionRegistryPublisher,
)


def _write_decision(
    path: Path,
    *,
    selected_model: str,
) -> Path:
    payload = {
        "selection": {
            "selected_model": selected_model,
        },
    }

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return path


def _revision_files(
    registry_root: Path,
) -> list[Path]:
    revision_root = (
        registry_root
        / "history"
    )

    if not revision_root.exists():
        return []

    return sorted(
        path
        for path in revision_root.iterdir()
        if path.is_file()
        and path.suffix == ".json"
    )


def test_first_publish_persists_immutable_revision(
    tmp_path: Path,
) -> None:
    registry_root = tmp_path / "registry"

    source = _write_decision(
        tmp_path / "decision-a.json",
        selected_model="model-a",
    )

    source_bytes = source.read_bytes()
    source_sha256 = hashlib.sha256(
        source_bytes
    ).hexdigest()

    result = (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source,
            registry_root=registry_root,
        )
    )

    revisions = _revision_files(
        registry_root
    )

    assert len(revisions) == 1

    revision = json.loads(
        revisions[0].read_text(
            encoding="utf-8"
        )
    )

    assert revision["selected_model"] == (
        "model-a"
    )
    assert revision["source_sha256"] == (
        source_sha256
    )
    assert revision["published_at_kst"] == (
        result.published_at_kst
    )

    assert revision["source_path"] == str(
        source
    )

    assert (
        revision["published_path"]
        == str(
            registry_root
            / "active"
            / "champion_decision.json"
        )
    )


def test_repeated_publish_preserves_both_revisions(
    tmp_path: Path,
) -> None:
    registry_root = tmp_path / "registry"

    source_a = _write_decision(
        tmp_path / "decision-a.json",
        selected_model="model-a",
    )

    source_b = _write_decision(
        tmp_path / "decision-b.json",
        selected_model="model-b",
    )

    publisher = (
        ProductionChampionRegistryPublisher()
    )

    publisher.publish(
        source_decision=source_a,
        registry_root=registry_root,
    )

    publisher.publish(
        source_decision=source_b,
        registry_root=registry_root,
    )

    revisions = _revision_files(
        registry_root
    )

    assert len(revisions) == 2

    payloads = [
        json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
        for path in revisions
    ]

    models = {
        payload["selected_model"]
        for payload in payloads
    }

    assert models == {
        "model-a",
        "model-b",
    }


def test_repeated_publish_still_replaces_active_state(
    tmp_path: Path,
) -> None:
    registry_root = tmp_path / "registry"

    source_a = _write_decision(
        tmp_path / "decision-a.json",
        selected_model="model-a",
    )

    source_b = _write_decision(
        tmp_path / "decision-b.json",
        selected_model="model-b",
    )

    publisher = (
        ProductionChampionRegistryPublisher()
    )

    publisher.publish(
        source_decision=source_a,
        registry_root=registry_root,
    )

    publisher.publish(
        source_decision=source_b,
        registry_root=registry_root,
    )

    active_decision = json.loads(
        (
            registry_root
            / "active"
            / "champion_decision.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    active_publication = json.loads(
        (
            registry_root
            / "active"
            / "publication.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        active_decision[
            "selection"
        ]["selected_model"]
        == "model-b"
    )

    assert (
        active_publication[
            "selected_model"
        ]
        == "model-b"
    )

    assert len(
        _revision_files(
            registry_root
        )
    ) == 2


def test_revision_records_are_not_active_aliases(
    tmp_path: Path,
) -> None:
    registry_root = tmp_path / "registry"

    source_a = _write_decision(
        tmp_path / "decision-a.json",
        selected_model="model-a",
    )

    source_b = _write_decision(
        tmp_path / "decision-b.json",
        selected_model="model-b",
    )

    publisher = (
        ProductionChampionRegistryPublisher()
    )

    publisher.publish(
        source_decision=source_a,
        registry_root=registry_root,
    )

    first_revisions = _revision_files(
        registry_root
    )

    assert len(first_revisions) == 1

    first_path = first_revisions[0]
    first_bytes = first_path.read_bytes()

    publisher.publish(
        source_decision=source_b,
        registry_root=registry_root,
    )

    assert first_path.exists()
    assert first_path.read_bytes() == (
        first_bytes
    )

    assert len(
        _revision_files(
            registry_root
        )
    ) == 2
