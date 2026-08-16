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
    run_id: str,
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
            "start_round": 1231,
            "end_round": 1231,
        },
        "windows": [
            {
                "name": "window-001",
                "start_round": 1231,
                "end_round": 1231,
                "round_count": 1,
            },
        ],
        "replay_config": {
            "seed_base": 20260802,
            "temperature": 0.85,
            "candidate_count": 100,
            "top_k": 10,
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


def _decision_payload(
    *,
    ranking_champion: str = "baseline",
    selected_model: str | None = None,
    promoted: bool = False,
) -> dict[str, object]:
    return {
        "selection": {
            "ranking_champion": ranking_champion,
            "selected_model": selected_model,
            "promotion": {
                "candidate": ranking_champion,
                "promoted": promoted,
                "promoted_model": (
                    selected_model
                    if promoted
                    else None
                ),
                "composite_margin": 0.0,
                "rejection_reasons": [],
            },
        },
        "matrix": {},
    }


def _make_run(
    root: Path,
    *,
    name: str,
    run_id: str,
    decision: dict[str, object] | None = None,
    create_decision: bool = True,
) -> Path:
    run_root = root / name

    _write_json(
        run_root / "evaluation_run.json",
        _run_payload(
            run_id=run_id,
        ),
    )

    if create_decision:
        _write_json(
            run_root
            / "report"
            / "champion_decision.json",
            (
                _decision_payload()
                if decision is None
                else decision
            ),
        )

    return run_root


def test_service_audits_discovered_runs(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_audit_service import (
        ModelEvaluationRunAuditService,
    )

    _make_run(
        tmp_path,
        name="run-a",
        run_id="aaaaaaaaaaaaaaaa",
    )

    _make_run(
        tmp_path,
        name="run-b",
        run_id="bbbbbbbbbbbbbbbb",
    )

    report = (
        ModelEvaluationRunAuditService()
        .run(tmp_path)
    )

    assert report.total_count == 2
    assert report.pass_count == 2
    assert report.fail_count == 0
    assert report.incomplete_count == 0

    assert tuple(
        result.run_id
        for result in report.results
    ) == (
        "aaaaaaaaaaaaaaaa",
        "bbbbbbbbbbbbbbbb",
    )


def test_service_aggregates_status_counts(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_audit_service import (
        ModelEvaluationRunAuditService,
    )

    _make_run(
        tmp_path,
        name="pass",
        run_id="aaaaaaaaaaaaaaaa",
    )

    _make_run(
        tmp_path,
        name="fail",
        run_id="bbbbbbbbbbbbbbbb",
        decision=_decision_payload(
            ranking_champion="calibration",
        ),
    )

    _make_run(
        tmp_path,
        name="incomplete",
        run_id="cccccccccccccccc",
        create_decision=False,
    )

    report = (
        ModelEvaluationRunAuditService()
        .run(tmp_path)
    )

    assert report.total_count == 3
    assert report.pass_count == 1
    assert report.fail_count == 1
    assert report.incomplete_count == 1


def test_service_report_serialization(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_audit_service import (
        ModelEvaluationRunAuditService,
    )

    _make_run(
        tmp_path,
        name="run-a",
        run_id="aaaaaaaaaaaaaaaa",
    )

    report = (
        ModelEvaluationRunAuditService()
        .run(tmp_path)
    )

    payload = report.as_dict()

    assert payload["summary"] == {
        "total_count": 1,
        "pass_count": 1,
        "fail_count": 0,
        "incomplete_count": 0,
    }

    assert len(
        payload["results"]
    ) == 1

    assert (
        payload["results"][0]["run_id"]
        == "aaaaaaaaaaaaaaaa"
    )

    assert (
        payload["results"][0]["status"]
        == "PASS"
    )


def test_service_returns_empty_report(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_audit_service import (
        ModelEvaluationRunAuditService,
    )

    report = (
        ModelEvaluationRunAuditService()
        .run(tmp_path)
    )

    assert report.results == ()
    assert report.total_count == 0
    assert report.pass_count == 0
    assert report.fail_count == 0
    assert report.incomplete_count == 0


def test_service_rejects_missing_root(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_audit_service import (
        ModelEvaluationRunAuditService,
    )

    with pytest.raises(
        FileNotFoundError,
    ):
        ModelEvaluationRunAuditService().run(
            tmp_path / "missing"
        )


def test_service_rejects_file_root(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_audit_service import (
        ModelEvaluationRunAuditService,
    )

    path = (
        tmp_path
        / "not-a-directory"
    )

    path.write_text(
        "x",
        encoding="utf-8",
    )

    with pytest.raises(
        NotADirectoryError,
    ):
        ModelEvaluationRunAuditService().run(
            path
        )
