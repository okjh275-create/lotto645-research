from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from lrp.production.production_lifecycle import (
    ProductionLifecycleStageResult,
)


def _request(
    tmp_path: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        history_path=(
            tmp_path
            / "history.json"
        ),
        evaluation_output_root=(
            tmp_path
            / "evaluation"
        ),
        production_registry_root=(
            tmp_path
            / "registry"
        ),
        production_snapshot_root=(
            tmp_path
            / "snapshots"
        ),
        prediction_output_root=(
            tmp_path
            / "prediction"
        ),
        round_no=1232,
        seed=20260818,
        temperature=0.85,
        candidate_count=100,
        top_k=10,
        practical_k=5,
        mode="fast",
        evaluation_start_round=1212,
        evaluation_end_round=1231,
        long_gap_window=5,
    )


def test_adapter_module_is_directly_importable(
) -> None:
    code = """
import lrp.production.production_lifecycle_adapters
print("IMPORT OK")
"""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0

    assert (
        "IMPORT OK"
        in result.stdout
    )


def test_model_evaluation_returns_stage_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapters = importlib.import_module(
        "lrp.production."
        "production_lifecycle_adapters"
    )

    monkeypatch.setattr(
        adapters.model_evaluation_cli,
        "main",
        lambda argv=None: 0,
    )

    request = _request(
        tmp_path
    )

    result = (
        adapters
        .run_model_evaluation_stage(
            request
        )
    )

    assert isinstance(
        result,
        ProductionLifecycleStageResult,
    )

    assert result.name == (
        "model_evaluation"
    )

    assert result.status == "PASS"

    assert result.detail[
        "returncode"
    ] == 0

    assert result.detail[
        "replay_output"
    ] == (
        request.evaluation_output_root
        / "replay"
    )

    assert result.detail[
        "report_output"
    ] == (
        request.evaluation_output_root
        / "report"
    )


def test_publication_returns_stage_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapters = importlib.import_module(
        "lrp.production."
        "production_lifecycle_adapters"
    )

    monkeypatch.setattr(
        adapters.publish_champion_cli,
        "run_publish",
        lambda **kwargs: {
            "status": "PASS",
            "selected_model": None,
        },
    )

    result = (
        adapters
        .run_publication_stage(
            _request(tmp_path)
        )
    )

    assert isinstance(
        result,
        ProductionLifecycleStageResult,
    )

    assert result.name == "publication"
    assert result.status == "PASS"

    assert result.detail[
        "selected_model"
    ] is None


def test_audit_returns_warn_stage_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapters = importlib.import_module(
        "lrp.production."
        "production_lifecycle_adapters"
    )

    monkeypatch.setattr(
        adapters.audit_champion_cli,
        "run_audit",
        lambda **kwargs: {
            "status": "WARN",
            "resolved_model":
                "baseline",
        },
    )

    result = (
        adapters
        .run_audit_stage(
            _request(tmp_path)
        )
    )

    assert isinstance(
        result,
        ProductionLifecycleStageResult,
    )

    assert result.name == "audit"
    assert result.status == "WARN"

    assert result.detail[
        "resolved_model"
    ] == "baseline"


def test_prediction_returns_stage_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapters = importlib.import_module(
        "lrp.production."
        "production_lifecycle_adapters"
    )

    monkeypatch.setattr(
        adapters.predict_cli,
        "run_predict",
        lambda arguments: {
            "status": "PASS",
            "round": 1232,
            "production_activation": {
                "enabled": True,
                "resolved_model":
                    "baseline",
            },
        },
    )

    result = (
        adapters
        .run_prediction_stage(
            _request(tmp_path)
        )
    )

    assert isinstance(
        result,
        ProductionLifecycleStageResult,
    )

    assert result.name == "prediction"
    assert result.status == "PASS"

    assert result.detail[
        "round"
    ] == 1232


def test_error_status_is_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapters = importlib.import_module(
        "lrp.production."
        "production_lifecycle_adapters"
    )

    monkeypatch.setattr(
        adapters.publish_champion_cli,
        "run_publish",
        lambda **kwargs: {
            "status": "ERROR",
            "message": "failed",
        },
    )

    result = (
        adapters
        .run_publication_stage(
            _request(tmp_path)
        )
    )

    assert result.status == "ERROR"

    assert result.detail[
        "message"
    ] == "failed"
