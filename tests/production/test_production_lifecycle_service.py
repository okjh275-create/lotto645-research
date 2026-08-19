from __future__ import annotations

from pathlib import Path

import pytest


def _load_contract():
    from lrp.production.production_lifecycle import (
        ProductionLifecycleRequest,
        ProductionLifecycleResult,
        ProductionLifecycleService,
        ProductionLifecycleStageResult,
    )

    return (
        ProductionLifecycleRequest,
        ProductionLifecycleResult,
        ProductionLifecycleService,
        ProductionLifecycleStageResult,
    )


def test_production_lifecycle_contract_is_importable(
) -> None:
    (
        ProductionLifecycleRequest,
        ProductionLifecycleResult,
        ProductionLifecycleService,
        ProductionLifecycleStageResult,
    ) = _load_contract()

    assert ProductionLifecycleRequest is not None
    assert ProductionLifecycleResult is not None
    assert ProductionLifecycleService is not None
    assert ProductionLifecycleStageResult is not None


def test_request_preserves_release_inputs(
    tmp_path: Path,
) -> None:
    (
        ProductionLifecycleRequest,
        _,
        _,
        _,
    ) = _load_contract()

    request = ProductionLifecycleRequest(
        history_path=tmp_path / "history.json",
        evaluation_output_root=(
            tmp_path / "evaluation"
        ),
        production_registry_root=(
            tmp_path / "registry"
        ),
        production_snapshot_root=(
            tmp_path / "snapshots"
        ),
        prediction_output_root=(
            tmp_path / "prediction"
        ),
        round_no=1232,
        seed=20260818,
        temperature=0.85,
        candidate_count=100,
        top_k=10,
        practical_k=5,
        mode="fast",
    )

    assert request.history_path == (
        tmp_path / "history.json"
    )

    assert request.evaluation_output_root == (
        tmp_path / "evaluation"
    )

    assert request.production_registry_root == (
        tmp_path / "registry"
    )

    assert request.production_snapshot_root == (
        tmp_path / "snapshots"
    )

    assert request.prediction_output_root == (
        tmp_path / "prediction"
    )

    assert request.round_no == 1232
    assert request.seed == 20260818
    assert request.temperature == 0.85
    assert request.candidate_count == 100
    assert request.top_k == 10
    assert request.practical_k == 5
    assert request.mode == "fast"


@pytest.mark.parametrize(
    ("name", "status"),
    [
        ("model_evaluation", "PASS"),
        ("publication", "PASS"),
        ("audit", "WARN"),
        ("prediction", "PASS"),
    ],
)
def test_stage_result_preserves_stage_contract(
    name: str,
    status: str,
) -> None:
    (
        _,
        _,
        _,
        ProductionLifecycleStageResult,
    ) = _load_contract()

    stage = ProductionLifecycleStageResult(
        name=name,
        status=status,
        detail={
            "marker": name,
        },
    )

    assert stage.name == name
    assert stage.status == status
    assert stage.detail == {
        "marker": name,
    }


def test_result_preserves_ordered_lifecycle_stages(
) -> None:
    (
        _,
        ProductionLifecycleResult,
        _,
        ProductionLifecycleStageResult,
    ) = _load_contract()

    stages = (
        ProductionLifecycleStageResult(
            name="model_evaluation",
            status="PASS",
            detail={},
        ),
        ProductionLifecycleStageResult(
            name="publication",
            status="PASS",
            detail={},
        ),
        ProductionLifecycleStageResult(
            name="audit",
            status="WARN",
            detail={},
        ),
        ProductionLifecycleStageResult(
            name="prediction",
            status="PASS",
            detail={},
        ),
    )

    result = ProductionLifecycleResult(
        status="WARN",
        stages=stages,
    )

    assert result.status == "WARN"

    assert tuple(
        stage.name
        for stage in result.stages
    ) == (
        "model_evaluation",
        "publication",
        "audit",
        "prediction",
    )


def test_service_exposes_run_boundary(
) -> None:
    (
        _,
        _,
        ProductionLifecycleService,
        _,
    ) = _load_contract()

    service = ProductionLifecycleService()

    assert callable(
        service.run
    )
