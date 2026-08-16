from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from lrp.production import (
    ProductionChampionPublicationResult,
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


def test_publication_result_records_source_path(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "source"
        / "champion_decision.json"
    )

    registry = (
        tmp_path
        / "registry"
    )

    _write_decision(
        source,
        selected_model="combined",
    )

    result = (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source,
            registry_root=registry,
        )
    )

    assert result.source_path == source


def test_publication_result_records_source_sha256(
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

    expected_sha256 = (
        hashlib.sha256(raw)
        .hexdigest()
    )

    result = (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source,
            registry_root=registry,
        )
    )

    assert (
        result.source_sha256
        == expected_sha256
    )


def test_publication_result_records_published_path(
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

    result = (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source,
            registry_root=registry,
        )
    )

    assert result.published_path == (
        registry
        / "active"
        / "champion_decision.json"
    )


def test_publication_result_records_kst_timestamp(
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
        selected_model="bayesian",
    )

    result = (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source,
            registry_root=registry,
        )
    )

    timestamp = datetime.fromisoformat(
        result.published_at_kst
    )

    assert timestamp.utcoffset() is not None

    assert (
        timestamp.utcoffset()
        .total_seconds()
        == 9 * 60 * 60
    )


def test_publication_result_records_selected_model(
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

    result = (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source,
            registry_root=registry,
        )
    )

    assert (
        result.selected_model
        == "combined"
    )


def test_publication_result_preserves_none_model(
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

    result = (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source,
            registry_root=registry,
        )
    )

    assert result.selected_model is None


def test_publication_result_serializes_provenance(
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
        selected_model="combined",
    )

    result = (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source,
            registry_root=registry,
        )
    )

    payload = result.as_dict()

    assert payload[
        "source_path"
    ] == str(source)

    assert payload[
        "source_sha256"
    ] == hashlib.sha256(
        raw
    ).hexdigest()

    assert payload[
        "published_path"
    ] == str(
        registry
        / "active"
        / "champion_decision.json"
    )

    assert payload[
        "selected_model"
    ] == "combined"

    assert isinstance(
        payload[
            "published_at_kst"
        ],
        str,
    )


def test_publication_result_is_frozen() -> None:
    from dataclasses import (
        FrozenInstanceError,
    )

    result = (
        ProductionChampionPublicationResult(
            source_path=Path(
                "source.json"
            ),
            source_sha256="a" * 64,
            published_path=Path(
                "registry/active/"
                "champion_decision.json"
            ),
            published_at_kst=(
                "2026-08-16T12:00:00+09:00"
            ),
            selected_model="baseline",
        )
    )

    try:
        result.selected_model = "combined"
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError(
            "publication result is mutable"
        )