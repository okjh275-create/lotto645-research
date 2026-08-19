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
    marker: str,
) -> Path:
    payload = {
        "selection": {
            "selected_model": selected_model,
        },
        "metadata": {
            "marker": marker,
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


def _decision_history_path(
    registry_root: Path,
    source: Path,
) -> Path:
    source_sha256 = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()

    return (
        registry_root
        / "history"
        / "decisions"
        / f"{source_sha256}.json"
    )


def test_publish_persists_byte_identical_decision_snapshot(
    tmp_path: Path,
) -> None:
    registry_root = (
        tmp_path
        / "registry"
    )

    source = _write_decision(
        tmp_path / "decision.json",
        selected_model="model-a",
        marker="first",
    )

    source_bytes = (
        source.read_bytes()
    )

    ProductionChampionRegistryPublisher().publish(
        source_decision=source,
        registry_root=registry_root,
    )

    snapshot = (
        _decision_history_path(
            registry_root,
            source,
        )
    )

    assert snapshot.exists()

    assert (
        snapshot.read_bytes()
        == source_bytes
    )


def test_decision_snapshot_filename_is_source_sha256(
    tmp_path: Path,
) -> None:
    registry_root = (
        tmp_path
        / "registry"
    )

    source = _write_decision(
        tmp_path / "decision.json",
        selected_model="model-a",
        marker="sha-contract",
    )

    source_sha256 = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()

    ProductionChampionRegistryPublisher().publish(
        source_decision=source,
        registry_root=registry_root,
    )

    snapshot = (
        registry_root
        / "history"
        / "decisions"
        / f"{source_sha256}.json"
    )

    assert snapshot.exists()

    assert (
        snapshot.stem
        == source_sha256
    )


def test_multiple_publications_preserve_all_decision_snapshots(
    tmp_path: Path,
) -> None:
    registry_root = (
        tmp_path
        / "registry"
    )

    source_a = _write_decision(
        tmp_path / "a.json",
        selected_model="model-a",
        marker="a",
    )

    source_b = _write_decision(
        tmp_path / "b.json",
        selected_model="model-b",
        marker="b",
    )

    publisher = (
        ProductionChampionRegistryPublisher()
    )

    publisher.publish(
        source_decision=source_a,
        registry_root=registry_root,
    )

    snapshot_a = (
        _decision_history_path(
            registry_root,
            source_a,
        )
    )

    snapshot_a_bytes = (
        snapshot_a.read_bytes()
        if snapshot_a.exists()
        else None
    )

    publisher.publish(
        source_decision=source_b,
        registry_root=registry_root,
    )

    snapshot_b = (
        _decision_history_path(
            registry_root,
            source_b,
        )
    )

    assert snapshot_a.exists()
    assert snapshot_b.exists()

    assert (
        snapshot_a.read_bytes()
        == source_a.read_bytes()
    )

    assert (
        snapshot_b.read_bytes()
        == source_b.read_bytes()
    )

    assert (
        snapshot_a_bytes
        == snapshot_a.read_bytes()
    )


def test_republishing_same_decision_does_not_mutate_snapshot(
    tmp_path: Path,
) -> None:
    registry_root = (
        tmp_path
        / "registry"
    )

    source = _write_decision(
        tmp_path / "decision.json",
        selected_model="model-a",
        marker="immutable",
    )

    publisher = (
        ProductionChampionRegistryPublisher()
    )

    publisher.publish(
        source_decision=source,
        registry_root=registry_root,
    )

    snapshot = (
        _decision_history_path(
            registry_root,
            source,
        )
    )

    assert snapshot.exists()

    before = snapshot.read_bytes()

    publisher.publish(
        source_decision=source,
        registry_root=registry_root,
    )

    assert snapshot.exists()

    assert (
        snapshot.read_bytes()
        == before
    )


def test_decision_history_survives_source_deletion(
    tmp_path: Path,
) -> None:
    registry_root = (
        tmp_path
        / "registry"
    )

    source = _write_decision(
        tmp_path / "decision.json",
        selected_model="model-a",
        marker="deletion",
    )

    source_bytes = (
        source.read_bytes()
    )

    publisher = (
        ProductionChampionRegistryPublisher()
    )

    publisher.publish(
        source_decision=source,
        registry_root=registry_root,
    )

    snapshot = (
        _decision_history_path(
            registry_root,
            source,
        )
    )

    source.unlink()

    assert not source.exists()

    assert snapshot.exists()

    assert (
        snapshot.read_bytes()
        == source_bytes
    )
