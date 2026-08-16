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


def _run_payload() -> dict[str, object]:
    return {
        "run_id": "0123456789abcdef",
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
        "champion_artifact": (
            "report/champion_decision.json"
        ),
    }


def _decision_payload(
    *,
    ranking_champion: str | None = "baseline",
    selected_model: str | None = None,
    promoted: bool = False,
) -> dict[str, object]:
    return {
        "ranking_champion": ranking_champion,
        "selected_model": selected_model,
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
    decision: dict[str, object] | None = None,
    create_decision: bool = True,
) -> Path:
    run_root = root / "run"

    _write_json(
        run_root / "evaluation_run.json",
        _run_payload(),
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


def test_audit_passes_matching_artifacts(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_audit import (
        ModelEvaluationRunAudit,
    )

    run_root = _make_run(tmp_path)

    result = ModelEvaluationRunAudit().audit(
        run_root / "evaluation_run.json"
    )

    assert result.run_id == (
        "0123456789abcdef"
    )

    assert result.status == "PASS"
    assert result.issues == ()


def test_audit_detects_ranking_champion_mismatch(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_audit import (
        ModelEvaluationRunAudit,
    )

    run_root = _make_run(
        tmp_path,
        decision=_decision_payload(
            ranking_champion="calibration",
        ),
    )

    result = ModelEvaluationRunAudit().audit(
        run_root / "evaluation_run.json"
    )

    assert result.status == "FAIL"

    assert (
        "ranking_champion_mismatch"
        in result.issues
    )


def test_audit_detects_selected_model_mismatch(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_audit import (
        ModelEvaluationRunAudit,
    )

    run_root = _make_run(
        tmp_path,
        decision=_decision_payload(
            selected_model="calibration",
        ),
    )

    result = ModelEvaluationRunAudit().audit(
        run_root / "evaluation_run.json"
    )

    assert result.status == "FAIL"

    assert (
        "selected_model_mismatch"
        in result.issues
    )


def test_audit_detects_promotion_mismatch(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_audit import (
        ModelEvaluationRunAudit,
    )

    run_root = _make_run(
        tmp_path,
        decision=_decision_payload(
            selected_model="baseline",
            promoted=True,
        ),
    )

    result = ModelEvaluationRunAudit().audit(
        run_root / "evaluation_run.json"
    )

    assert result.status == "FAIL"

    assert (
        "promoted_mismatch"
        in result.issues
    )


def test_audit_reports_missing_champion_artifact(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_audit import (
        ModelEvaluationRunAudit,
    )

    run_root = _make_run(
        tmp_path,
        create_decision=False,
    )

    result = ModelEvaluationRunAudit().audit(
        run_root / "evaluation_run.json"
    )

    assert result.status == "INCOMPLETE"

    assert result.issues == (
        "champion_artifact_missing",
    )


def test_audit_result_serialization(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_audit import (
        ModelEvaluationRunAudit,
    )

    run_root = _make_run(tmp_path)

    result = ModelEvaluationRunAudit().audit(
        run_root / "evaluation_run.json"
    )

    payload = result.as_dict()

    assert payload == {
        "run_id": "0123456789abcdef",
        "status": "PASS",
        "issues": [],
        "evaluation_run": str(
            run_root
            / "evaluation_run.json"
        ),
        "champion_artifact": (
            "report/champion_decision.json"
        ),
    }


def test_audit_rejects_missing_run_record(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_audit import (
        ModelEvaluationRunAudit,
    )

    with pytest.raises(
        FileNotFoundError,
    ):
        ModelEvaluationRunAudit().audit(
            tmp_path / "missing.json"
        )


def test_audit_rejects_directory_run_record(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_audit import (
        ModelEvaluationRunAudit,
    )

    with pytest.raises(
        IsADirectoryError,
    ):
        ModelEvaluationRunAudit().audit(
            tmp_path
        )
