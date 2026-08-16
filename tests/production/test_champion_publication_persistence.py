from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from lrp.production import (
    ProductionChampionRegistryPublisher,
)


def _write_decision(
    path: Path,
    *,
    selected_model: object,
) -> bytes:
    raw = (
        json.dumps(
            {
                "selection": {
                    "selected_model": selected_model,
                },
            },
            indent=2,
        )
        + "\n"
    ).encode("utf-8")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_bytes(raw)

    return raw


def _publication_path(
    registry_root: Path,
) -> Path:
    return (
        registry_root
        / "active"
        / "publication.json"
    )


def test_publish_persists_publication_record(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "source.json"
    )

    registry = (
        tmp_path
        / "registry"
    )

    _write_decision(
        source,
        selected_model="combined",
    )

    (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source,
            registry_root=registry,
        )
    )

    path = _publication_path(
        registry
    )

    assert path.is_file()


def test_persisted_record_matches_result(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "source.json"
    )

    registry = (
        tmp_path
        / "registry"
    )

    _write_decision(
        source,
        selected_model="calibration",
    )

    result = (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source,
            registry_root=registry,
        )
    )

    payload = json.loads(
        _publication_path(
            registry
        ).read_text(
            encoding="utf-8",
        )
    )

    assert payload == result.as_dict()


def test_persisted_record_preserves_none_model(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "source.json"
    )

    registry = (
        tmp_path
        / "registry"
    )

    _write_decision(
        source,
        selected_model=None,
    )

    (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source,
            registry_root=registry,
        )
    )

    payload = json.loads(
        _publication_path(
            registry
        ).read_text(
            encoding="utf-8",
        )
    )

    assert payload[
        "selected_model"
    ] is None


def test_persisted_record_contains_source_sha256(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "source.json"
    )

    registry = (
        tmp_path
        / "registry"
    )

    raw = _write_decision(
        source,
        selected_model="bayesian",
    )

    (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source,
            registry_root=registry,
        )
    )

    payload = json.loads(
        _publication_path(
            registry
        ).read_text(
            encoding="utf-8",
        )
    )

    assert (
        payload["source_sha256"]
        == hashlib.sha256(
            raw
        ).hexdigest()
    )


def test_persisted_record_timestamp_is_kst(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "source.json"
    )

    registry = (
        tmp_path
        / "registry"
    )

    _write_decision(
        source,
        selected_model="baseline",
    )

    (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source,
            registry_root=registry,
        )
    )

    payload = json.loads(
        _publication_path(
            registry
        ).read_text(
            encoding="utf-8",
        )
    )

    timestamp = datetime.fromisoformat(
        payload[
            "published_at_kst"
        ]
    )

    assert timestamp.utcoffset() is not None

    assert (
        timestamp.utcoffset()
        .total_seconds()
        == 9 * 60 * 60
    )


def test_repeated_publish_replaces_publication_record(
    tmp_path: Path,
) -> None:
    source_a = (
        tmp_path
        / "a.json"
    )

    source_b = (
        tmp_path
        / "b.json"
    )

    registry = (
        tmp_path
        / "registry"
    )

    _write_decision(
        source_a,
        selected_model="baseline",
    )

    first = (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source_a,
            registry_root=registry,
        )
    )

    first_payload = json.loads(
        _publication_path(
            registry
        ).read_text(
            encoding="utf-8",
        )
    )

    _write_decision(
        source_b,
        selected_model="combined",
    )

    second = (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source_b,
            registry_root=registry,
        )
    )

    second_payload = json.loads(
        _publication_path(
            registry
        ).read_text(
            encoding="utf-8",
        )
    )

    assert (
        first_payload
        == first.as_dict()
    )

    assert (
        second_payload
        == second.as_dict()
    )

    assert (
        second_payload[
            "selected_model"
        ]
        == "combined"
    )

    assert (
        second_payload[
            "source_path"
        ]
        == str(source_b)
    )


def test_publication_record_has_no_temp_file_leftover(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "source.json"
    )

    registry = (
        tmp_path
        / "registry"
    )

    _write_decision(
        source,
        selected_model="combined",
    )

    (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source,
            registry_root=registry,
        )
    )

    active = (
        registry
        / "active"
    )

    leftovers = list(
        active.glob(
            ".publication.json.*.tmp"
        )
    )

    assert leftovers == []


def test_persistence_does_not_change_active_decision(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "source.json"
    )

    registry = (
        tmp_path
        / "registry"
    )

    raw = _write_decision(
        source,
        selected_model="calibration",
    )

    result = (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source,
            registry_root=registry,
        )
    )

    assert (
        result.published_path
        .read_bytes()
        == raw
    )


def test_registry_reader_still_reads_active_decision(
    tmp_path: Path,
) -> None:
    from lrp.production import (
        ProductionChampionRegistryReader,
    )

    source = (
        tmp_path
        / "source.json"
    )

    registry = (
        tmp_path
        / "registry"
    )

    _write_decision(
        source,
        selected_model="combined",
    )

    (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source,
            registry_root=registry,
        )
    )

    decision = (
        ProductionChampionRegistryReader()
        .read(registry)
    )

    assert (
        decision.selected_model
        == "combined"
    )