from __future__ import annotations

from pathlib import Path

import pytest

from lrp.production.production_lifecycle import (
    ProductionLifecycleRequest,
    ProductionLifecycleService,
    ProductionLifecycleStageResult,
)


def _request(
    tmp_path: Path,
) -> ProductionLifecycleRequest:
    return ProductionLifecycleRequest(
        history_path=(
            tmp_path / "history.json"
        ),
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


def _stage(
    name: str,
    status: str = "PASS",
) -> ProductionLifecycleStageResult:
    return ProductionLifecycleStageResult(
        name=name,
        status=status,
        detail={
            "stage": name,
        },
    )


def test_run_executes_stages_in_release_order(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def model_evaluation(request):
        calls.append(
            "model_evaluation"
        )
        return _stage(
            "model_evaluation"
        )

    def publication(request):
        calls.append(
            "publication"
        )
        return _stage(
            "publication"
        )

    def audit(request):
        calls.append(
            "audit"
        )
        return _stage(
            "audit",
            "WARN",
        )

    def prediction(request):
        calls.append(
            "prediction"
        )
        return _stage(
            "prediction"
        )

    service = ProductionLifecycleService(
        model_evaluation=model_evaluation,
        publication=publication,
        audit=audit,
        prediction=prediction,
    )

    result = service.run(
        _request(tmp_path)
    )

    assert calls == [
        "model_evaluation",
        "publication",
        "audit",
        "prediction",
    ]

    assert [
        stage.name
        for stage in result.stages
    ] == calls


def test_audit_warn_does_not_stop_prediction(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def stage(
        name: str,
        status: str = "PASS",
    ):
        def run(request):
            calls.append(name)
            return _stage(
                name,
                status,
            )

        return run

    service = ProductionLifecycleService(
        model_evaluation=stage(
            "model_evaluation"
        ),
        publication=stage(
            "publication"
        ),
        audit=stage(
            "audit",
            "WARN",
        ),
        prediction=stage(
            "prediction"
        ),
    )

    result = service.run(
        _request(tmp_path)
    )

    assert calls == [
        "model_evaluation",
        "publication",
        "audit",
        "prediction",
    ]

    assert result.status == "WARN"

    assert [
        item.status
        for item in result.stages
    ] == [
        "PASS",
        "PASS",
        "WARN",
        "PASS",
    ]


def test_all_pass_produces_pass_result(
    tmp_path: Path,
) -> None:
    def stage(name: str):
        def run(request):
            return _stage(name)

        return run

    service = ProductionLifecycleService(
        model_evaluation=stage(
            "model_evaluation"
        ),
        publication=stage(
            "publication"
        ),
        audit=stage(
            "audit"
        ),
        prediction=stage(
            "prediction"
        ),
    )

    result = service.run(
        _request(tmp_path)
    )

    assert result.status == "PASS"

    assert len(result.stages) == 4


@pytest.mark.parametrize(
    "failed_stage",
    [
        "model_evaluation",
        "publication",
        "audit",
        "prediction",
    ],
)
def test_error_stage_fails_fast(
    tmp_path: Path,
    failed_stage: str,
) -> None:
    calls: list[str] = []

    ordered = [
        "model_evaluation",
        "publication",
        "audit",
        "prediction",
    ]

    def stage(name: str):
        def run(request):
            calls.append(name)

            status = (
                "ERROR"
                if name == failed_stage
                else "PASS"
            )

            return _stage(
                name,
                status,
            )

        return run

    service = ProductionLifecycleService(
        model_evaluation=stage(
            "model_evaluation"
        ),
        publication=stage(
            "publication"
        ),
        audit=stage(
            "audit"
        ),
        prediction=stage(
            "prediction"
        ),
    )

    result = service.run(
        _request(tmp_path)
    )

    failure_index = ordered.index(
        failed_stage
    )

    expected_calls = ordered[
        : failure_index + 1
    ]

    assert calls == expected_calls

    assert [
        item.name
        for item in result.stages
    ] == expected_calls

    assert result.status == "ERROR"


def test_request_object_is_forwarded_unchanged(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)

    observed = []

    def stage(name: str):
        def run(received):
            observed.append(received)
            return _stage(name)

        return run

    service = ProductionLifecycleService(
        model_evaluation=stage(
            "model_evaluation"
        ),
        publication=stage(
            "publication"
        ),
        audit=stage(
            "audit"
        ),
        prediction=stage(
            "prediction"
        ),
    )

    service.run(request)

    assert observed == [
        request,
        request,
        request,
        request,
    ]

    assert all(
        item is request
        for item in observed
    )
