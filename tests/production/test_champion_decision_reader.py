import json
from pathlib import Path

import pytest

from lrp.production import (
    ProductionChampionDecisionReader,
)


def _write_json(
    path: Path,
    payload: object,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_reader_loads_selected_model(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "champion_decision.json"
    )

    _write_json(
        path,
        {
            "selection": {
                "selected_model": "combined",
            },
        },
    )

    decision = (
        ProductionChampionDecisionReader()
        .read(path)
    )

    assert decision.selected_model == "combined"


def test_reader_preserves_none_selected_model(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "champion_decision.json"
    )

    _write_json(
        path,
        {
            "selection": {
                "selected_model": None,
            },
        },
    )

    decision = (
        ProductionChampionDecisionReader()
        .read(path)
    )

    assert decision.selected_model is None


def test_reader_accepts_string_path(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "champion_decision.json"
    )

    _write_json(
        path,
        {
            "selection": {
                "selected_model": "calibration",
            },
        },
    )

    decision = (
        ProductionChampionDecisionReader()
        .read(str(path))
    )

    assert decision.selected_model == "calibration"


def test_reader_rejects_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
    ):
        ProductionChampionDecisionReader().read(
            tmp_path
            / "missing.json"
        )


def test_reader_rejects_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        IsADirectoryError,
    ):
        ProductionChampionDecisionReader().read(
            tmp_path
        )


def test_reader_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "champion_decision.json"
    )

    path.write_text(
        "{invalid",
        encoding="utf-8",
    )

    with pytest.raises(
        json.JSONDecodeError,
    ):
        ProductionChampionDecisionReader().read(
            path
        )


def test_reader_delegates_payload_validation(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "champion_decision.json"
    )

    _write_json(
        path,
        {
            "selection": {
                "selected_model": 123,
            },
        },
    )

    with pytest.raises(
        TypeError,
        match="selected_model",
    ):
        ProductionChampionDecisionReader().read(
            path
        )


def test_reader_ignores_non_production_fields(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "champion_decision.json"
    )

    _write_json(
        path,
        {
            "selection": {
                "ranking_champion": "bayesian",
                "selected_model": "bayesian",
                "promotion": {
                    "promoted": True,
                },
            },
            "matrix": {
                "ignored": True,
            },
        },
    )

    decision = (
        ProductionChampionDecisionReader()
        .read(path)
    )

    assert decision.as_dict() == {
        "selected_model": "bayesian",
    }
