from __future__ import annotations

import json
from pathlib import Path

import pytest


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
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _run_payload(
    *,
    run_id: str = "abc123def4567890",
    start_round: int = 1200,
    end_round: int = 1228,
    champion_artifact: str = (
        "report/champion_decision.json"
    ),
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "history_path": "data/history.json",
        "model_names": [
            "baseline",
            "calibration",
        ],
        "round_range": {
            "start_round": start_round,
            "end_round": end_round,
        },
        "windows": [
            {
                "name": "window-001",
                "start_round": 1200,
                "end_round": 1209,
                "round_count": 10,
            },
            {
                "name": "window-002",
                "start_round": 1210,
                "end_round": 1219,
                "round_count": 10,
            },
            {
                "name": "window-003",
                "start_round": 1220,
                "end_round": 1228,
                "round_count": 9,
            },
        ],
        "replay_config": {
            "seed_base": 20260802,
            "temperature": 0.85,
            "candidate_count": 1000,
            "top_k": 20,
            "practical_k": 5,
            "long_gap_window": 5,
            "confidence": 0.8,
            "mode": "fast",
        },
        "champion": {
            "ranking_champion": "baseline",
            "selected_model": None,
            "promoted": False,
        },
        "champion_artifact": champion_artifact,
    }


def _make_run(
    root: Path,
    *,
    name: str = "run-001",
    payload: dict[str, object] | None = None,
    create_champion: bool = True,
) -> Path:
    run_root = root / name

    selected_payload = (
        _run_payload()
        if payload is None
        else payload
    )

    _write_json(
        run_root / "evaluation_run.json",
        selected_payload,
    )

    if create_champion:
        _write_json(
            run_root
            / "report"
            / "champion_decision.json",
            {
                "ranking_champion": "baseline",
                "selected_model": None,
                "selection": {
                    "ranking_champion": "baseline",
                    "selected_model": None,
                    "promotion": {
                        "candidate": "baseline",
                        "promoted": False,
                        "promoted_model": None,
                        "composite_margin": 0.0,
                        "rejection_reasons": [],
                    },
                },
                "matrix": {},
            },
        )

    return run_root


def test_discovers_model_evaluation_run(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_discovery import (
        ModelEvaluationRunDiscovery,
    )

    run_root = _make_run(tmp_path)

    records = (
        ModelEvaluationRunDiscovery()
        .discover(tmp_path)
    )

    assert len(records) == 1

    record = records[0]

    assert record.run_id == (
        "abc123def4567890"
    )
    assert record.root == run_root
    assert record.history_path == Path(
        "data/history.json"
    )
    assert record.model_names == (
        "baseline",
        "calibration",
    )
    assert record.start_round == 1200
    assert record.end_round == 1228
    assert record.window_count == 3

    assert record.ranking_champion == (
        "baseline"
    )
    assert record.selected_model is None
    assert record.promoted is False

    assert record.status == "PASS"
    assert record.missing_files == ()


def test_discovers_multiple_runs_deterministically(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_discovery import (
        ModelEvaluationRunDiscovery,
    )

    _make_run(
        tmp_path,
        name="run-b",
        payload=_run_payload(
            run_id="bbbbbbbbbbbbbbbb",
            start_round=1210,
            end_round=1228,
        ),
    )

    _make_run(
        tmp_path,
        name="run-a",
        payload=_run_payload(
            run_id="aaaaaaaaaaaaaaaa",
            start_round=1200,
            end_round=1228,
        ),
    )

    records = (
        ModelEvaluationRunDiscovery()
        .discover(tmp_path)
    )

    assert tuple(
        record.run_id
        for record in records
    ) == (
        "aaaaaaaaaaaaaaaa",
        "bbbbbbbbbbbbbbbb",
    )


def test_missing_champion_artifact_is_incomplete(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_discovery import (
        ModelEvaluationRunDiscovery,
    )

    _make_run(
        tmp_path,
        create_champion=False,
    )

    record = (
        ModelEvaluationRunDiscovery()
        .discover(tmp_path)[0]
    )

    assert record.status == "INCOMPLETE"

    assert record.missing_files == (
        "report/champion_decision.json",
    )


def test_record_serialization(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_discovery import (
        ModelEvaluationRunDiscovery,
    )

    _make_run(tmp_path)

    record = (
        ModelEvaluationRunDiscovery()
        .discover(tmp_path)[0]
    )

    payload = record.as_dict()

    assert payload["run_id"] == (
        "abc123def4567890"
    )

    assert payload["root"] == str(
        record.root
    )

    assert payload["history_path"] == (
        "data/history.json"
    )

    assert payload["model_names"] == [
        "baseline",
        "calibration",
    ]

    assert payload["round_range"] == {
        "start_round": 1200,
        "end_round": 1228,
    }

    assert payload["window_count"] == 3

    assert payload["champion"] == {
        "ranking_champion": "baseline",
        "selected_model": None,
        "promoted": False,
    }

    assert payload["status"] == "PASS"
    assert payload["missing_files"] == []


def test_invalid_round_range_is_rejected(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_discovery import (
        ModelEvaluationRunDiscovery,
    )

    payload = _run_payload()

    payload["round_range"] = {
        "start_round": 1228,
        "end_round": 1200,
    }

    _make_run(
        tmp_path,
        payload=payload,
    )

    with pytest.raises(
        ValueError,
        match="end_round",
    ):
        ModelEvaluationRunDiscovery().discover(
            tmp_path
        )


def test_empty_model_names_are_rejected(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_discovery import (
        ModelEvaluationRunDiscovery,
    )

    payload = _run_payload()
    payload["model_names"] = []

    _make_run(
        tmp_path,
        payload=payload,
    )

    with pytest.raises(
        ValueError,
        match="model_names",
    ):
        ModelEvaluationRunDiscovery().discover(
            tmp_path
        )


def test_empty_windows_are_rejected(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_discovery import (
        ModelEvaluationRunDiscovery,
    )

    payload = _run_payload()
    payload["windows"] = []

    _make_run(
        tmp_path,
        payload=payload,
    )

    with pytest.raises(
        ValueError,
        match="windows",
    ):
        ModelEvaluationRunDiscovery().discover(
            tmp_path
        )


def test_missing_root_is_rejected(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_discovery import (
        ModelEvaluationRunDiscovery,
    )

    with pytest.raises(
        FileNotFoundError,
    ):
        ModelEvaluationRunDiscovery().discover(
            tmp_path / "missing"
        )


def test_file_root_is_rejected(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_discovery import (
        ModelEvaluationRunDiscovery,
    )

    path = tmp_path / "not-a-directory"

    path.write_text(
        "x",
        encoding="utf-8",
    )

    with pytest.raises(
        NotADirectoryError,
    ):
        ModelEvaluationRunDiscovery().discover(
            path
        )


def test_repository_relative_champion_artifact_is_resolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.validation.model_evaluation_run_discovery import (
        ModelEvaluationRunDiscovery,
    )

    repository_root = (
        tmp_path
        / "repository"
    )

    run_root = (
        repository_root
        / "artifacts"
        / "validation"
        / "project_m"
        / "report"
    )

    run_record_path = (
        run_root
        / "evaluation_run.json"
    )

    champion_path = (
        run_root
        / "champion_decision.json"
    )

    repository_relative_artifact = (
        "artifacts/validation/"
        "project_m/report/"
        "champion_decision.json"
    )

    _write_json(
        run_record_path,
        _run_payload(
            champion_artifact=(
                repository_relative_artifact
            ),
        ),
    )

    _write_json(
        champion_path,
        {
            "ranking_champion": "baseline",
            "selected_model": None,
            "selection": {
                "ranking_champion": "baseline",
                "selected_model": None,
                "promotion": {
                    "candidate": "baseline",
                    "promoted": False,
                    "promoted_model": None,
                    "composite_margin": 0.0,
                    "rejection_reasons": [],
                },
            },
            "matrix": {},
        },
    )

    monkeypatch.chdir(
        repository_root
    )

    record = (
        ModelEvaluationRunDiscovery()
        .discover(
            repository_root
            / "artifacts"
        )[0]
    )

    assert record.status == "PASS"
    assert record.missing_files == ()

    assert record.champion_artifact == Path(
        repository_relative_artifact
    )
