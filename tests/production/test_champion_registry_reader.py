from __future__ import annotations

import json
from pathlib import Path

import pytest

from lrp.production import (
    ProductionChampionDecision,
    ProductionChampionRegistryReader,
)


def _write_active_decision(
    root: Path,
    *,
    selected_model: object,
) -> Path:
    path = (
        root
        / "active"
        / "champion_decision.json"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            {
                "selection": {
                    "selected_model": (
                        selected_model
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    return path


def test_reader_reads_active_registry_decision(
    tmp_path: Path,
) -> None:
    root = (
        tmp_path
        / "registry"
    )

    _write_active_decision(
        root,
        selected_model="combined",
    )

    reader = (
        ProductionChampionRegistryReader()
    )

    decision = reader.read(
        root
    )

    assert isinstance(
        decision,
        ProductionChampionDecision,
    )

    assert (
        decision.selected_model
        == "combined"
    )


def test_reader_preserves_none_selected_model(
    tmp_path: Path,
) -> None:
    root = (
        tmp_path
        / "registry"
    )

    _write_active_decision(
        root,
        selected_model=None,
    )

    decision = (
        ProductionChampionRegistryReader()
        .read(root)
    )

    assert decision.selected_model is None


def test_reader_accepts_string_root(
    tmp_path: Path,
) -> None:
    root = (
        tmp_path
        / "registry"
    )

    _write_active_decision(
        root,
        selected_model="baseline",
    )

    decision = (
        ProductionChampionRegistryReader()
        .read(
            str(root)
        )
    )

    assert (
        decision.selected_model
        == "baseline"
    )


def test_reader_propagates_missing_registry(
    tmp_path: Path,
) -> None:
    root = (
        tmp_path
        / "missing"
    )

    with pytest.raises(
        FileNotFoundError,
    ):
        (
            ProductionChampionRegistryReader()
            .read(root)
        )


def test_reader_propagates_file_root(
    tmp_path: Path,
) -> None:
    root = (
        tmp_path
        / "registry"
    )

    root.write_text(
        "not-directory\n",
        encoding="utf-8",
    )

    with pytest.raises(
        NotADirectoryError,
    ):
        (
            ProductionChampionRegistryReader()
            .read(root)
        )


def test_reader_propagates_missing_active_decision(
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

    with pytest.raises(
        FileNotFoundError,
    ):
        (
            ProductionChampionRegistryReader()
            .read(root)
        )


def test_reader_delegates_invalid_json_to_decision_reader(
    tmp_path: Path,
) -> None:
    root = (
        tmp_path
        / "registry"
    )

    path = (
        root
        / "active"
        / "champion_decision.json"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "not-json\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
    ):
        (
            ProductionChampionRegistryReader()
            .read(root)
        )


def test_reader_rejects_directory_decision_path(
    tmp_path: Path,
) -> None:
    root = (
        tmp_path
        / "registry"
    )

    (
        root
        / "active"
        / "champion_decision.json"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    with pytest.raises(
        IsADirectoryError,
    ):
        (
            ProductionChampionRegistryReader()
            .read(root)
        )
