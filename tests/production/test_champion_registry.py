from __future__ import annotations

from pathlib import Path

import pytest

from lrp.production import (
    ProductionChampionRegistry,
)


def test_registry_normalizes_root_to_path() -> None:
    registry = ProductionChampionRegistry(
        root="production-registry",
    )

    assert registry.root == Path(
        "production-registry"
    )


def test_registry_exposes_deterministic_active_path() -> None:
    registry = ProductionChampionRegistry(
        root=Path("production-registry"),
    )

    assert registry.active_decision_path == (
        Path("production-registry")
        / "active"
        / "champion_decision.json"
    )


def test_registry_resolves_existing_active_decision(
    tmp_path: Path,
) -> None:
    root = (
        tmp_path
        / "registry"
    )

    decision_path = (
        root
        / "active"
        / "champion_decision.json"
    )

    decision_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    decision_path.write_text(
        "{}\n",
        encoding="utf-8",
    )

    registry = ProductionChampionRegistry(
        root=root,
    )

    assert registry.decision_path() == decision_path


def test_registry_rejects_missing_root(
    tmp_path: Path,
) -> None:
    root = (
        tmp_path
        / "missing"
    )

    registry = ProductionChampionRegistry(
        root=root,
    )

    with pytest.raises(
        FileNotFoundError,
    ):
        registry.decision_path()


def test_registry_rejects_file_root(
    tmp_path: Path,
) -> None:
    root = (
        tmp_path
        / "registry"
    )

    root.write_text(
        "not a directory\n",
        encoding="utf-8",
    )

    registry = ProductionChampionRegistry(
        root=root,
    )

    with pytest.raises(
        NotADirectoryError,
    ):
        registry.decision_path()


def test_registry_rejects_missing_active_directory(
    tmp_path: Path,
) -> None:
    root = (
        tmp_path
        / "registry"
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    registry = ProductionChampionRegistry(
        root=root,
    )

    with pytest.raises(
        FileNotFoundError,
    ):
        registry.decision_path()


def test_registry_rejects_missing_decision_file(
    tmp_path: Path,
) -> None:
    root = (
        tmp_path
        / "registry"
    )

    (
        root
        / "active"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    registry = ProductionChampionRegistry(
        root=root,
    )

    with pytest.raises(
        FileNotFoundError,
    ):
        registry.decision_path()


def test_registry_rejects_directory_decision_path(
    tmp_path: Path,
) -> None:
    root = (
        tmp_path
        / "registry"
    )

    decision_path = (
        root
        / "active"
        / "champion_decision.json"
    )

    decision_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    registry = ProductionChampionRegistry(
        root=root,
    )

    with pytest.raises(
        IsADirectoryError,
    ):
        registry.decision_path()


def test_registry_does_not_parse_decision_contents(
    tmp_path: Path,
) -> None:
    root = (
        tmp_path
        / "registry"
    )

    decision_path = (
        root
        / "active"
        / "champion_decision.json"
    )

    decision_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    decision_path.write_text(
        "this is not json\n",
        encoding="utf-8",
    )

    registry = ProductionChampionRegistry(
        root=root,
    )

    assert registry.decision_path() == decision_path
