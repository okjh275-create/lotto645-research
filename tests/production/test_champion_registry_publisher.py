from __future__ import annotations

import json
from pathlib import Path

import pytest

from lrp.production import (
    ProductionChampionRegistryPublisher,
)


def _write_decision(
    path: Path,
    *,
    selected_model: object,
) -> bytes:
    payload = {
        "selection": {
            "selected_model": selected_model,
        },
    }

    raw = (
        json.dumps(
            payload,
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


def test_publisher_publishes_active_decision(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "source"
        / "champion_decision.json"
    )

    registry_root = (
        tmp_path
        / "registry"
    )

    source_bytes = _write_decision(
        source,
        selected_model="combined",
    )

    result = (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source,
            registry_root=registry_root,
        )
    )

    expected = (
        registry_root
        / "active"
        / "champion_decision.json"
    )

    assert result.published_path == expected
    assert result.selected_model == "combined"

    assert expected.is_file()

    assert (
        expected.read_bytes()
        == source_bytes
    )


def test_publisher_preserves_none_selected_model(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "source.json"
    )

    registry_root = (
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
            registry_root=registry_root,
        )
    )

    assert result.selected_model is None


def test_publisher_accepts_string_paths(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "source.json"
    )

    registry_root = (
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
            source_decision=str(source),
            registry_root=str(
                registry_root
            ),
        )
    )

    assert (
        result.published_path
        == registry_root
        / "active"
        / "champion_decision.json"
    )


def test_publisher_rejects_missing_source(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
    ):
        (
            ProductionChampionRegistryPublisher()
            .publish(
                source_decision=(
                    tmp_path
                    / "missing.json"
                ),
                registry_root=(
                    tmp_path
                    / "registry"
                ),
            )
        )


def test_publisher_rejects_directory_source(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "source"
    )

    source.mkdir()

    with pytest.raises(
        IsADirectoryError,
    ):
        (
            ProductionChampionRegistryPublisher()
            .publish(
                source_decision=source,
                registry_root=(
                    tmp_path
                    / "registry"
                ),
            )
        )


def test_publisher_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "source.json"
    )

    source.write_text(
        "not-json\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
    ):
        (
            ProductionChampionRegistryPublisher()
            .publish(
                source_decision=source,
                registry_root=(
                    tmp_path
                    / "registry"
                ),
            )
        )


def test_publisher_rejects_invalid_decision_schema(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "source.json"
    )

    source.write_text(
        json.dumps(
            {
                "selection": {
                    "selected_model": 123,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
    ):
        (
            ProductionChampionRegistryPublisher()
            .publish(
                source_decision=source,
                registry_root=(
                    tmp_path
                    / "registry"
                ),
            )
        )


def test_publisher_rejects_file_registry_root(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "source.json"
    )

    registry_root = (
        tmp_path
        / "registry"
    )

    _write_decision(
        source,
        selected_model="baseline",
    )

    registry_root.write_text(
        "not-directory\n",
        encoding="utf-8",
    )

    with pytest.raises(
        NotADirectoryError,
    ):
        (
            ProductionChampionRegistryPublisher()
            .publish(
                source_decision=source,
                registry_root=registry_root,
            )
        )


def test_publisher_replaces_existing_active_decision(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "source.json"
    )

    registry_root = (
        tmp_path
        / "registry"
    )

    existing = (
        registry_root
        / "active"
        / "champion_decision.json"
    )

    _write_decision(
        existing,
        selected_model="baseline",
    )

    replacement = _write_decision(
        source,
        selected_model="combined",
    )

    result = (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source,
            registry_root=registry_root,
        )
    )

    assert result.selected_model == "combined"

    assert (
        existing.read_bytes()
        == replacement
    )


def test_publisher_does_not_modify_source(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "source.json"
    )

    registry_root = (
        tmp_path
        / "registry"
    )

    original = _write_decision(
        source,
        selected_model="calibration",
    )

    (
        ProductionChampionRegistryPublisher()
        .publish(
            source_decision=source,
            registry_root=registry_root,
        )
    )

    assert (
        source.read_bytes()
        == original
    )