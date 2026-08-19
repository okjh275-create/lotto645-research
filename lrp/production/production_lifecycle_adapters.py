"""Adapters for production lifecycle stages."""

from __future__ import annotations

import argparse
import importlib
from typing import Any

from lrp.production.production_lifecycle import (
    ProductionLifecycleStageResult,
)


class _LazyModuleProxy:
    """Resolve a CLI module only when an adapter actually uses it."""

    def __init__(
        self,
        module_name: str,
    ) -> None:
        self._module_name = module_name

    def __getattr__(
        self,
        name: str,
    ) -> Any:
        module = importlib.import_module(
            self._module_name
        )

        return getattr(
            module,
            name,
        )


model_evaluation_cli = _LazyModuleProxy(
    "lrp.cli.model_evaluation"
)

publish_champion_cli = _LazyModuleProxy(
    "lrp.cli.publish_champion"
)

audit_champion_cli = _LazyModuleProxy(
    "lrp.cli.audit_champion"
)

predict_cli = _LazyModuleProxy(
    "lrp.cli.predict"
)


def _payload_stage_result(
    *,
    name: str,
    payload: dict[str, Any],
) -> ProductionLifecycleStageResult:
    """Normalize an existing application payload to a lifecycle stage."""

    raw_status = str(
        payload.get(
            "status",
            "ERROR",
        )
    ).upper()

    if raw_status == "FAIL":
        status = "ERROR"
    elif raw_status in {
        "PASS",
        "WARN",
        "ERROR",
    }:
        status = raw_status
    else:
        status = "ERROR"

    return ProductionLifecycleStageResult(
        name=name,
        status=status,
        detail=dict(payload),
    )


def run_model_evaluation_stage(
    request: argparse.Namespace,
) -> ProductionLifecycleStageResult:
    """Run the existing model-evaluation process boundary."""

    if request.evaluation_start_round is None:
        raise ValueError(
            "evaluation_start_round is required "
            "for model evaluation"
        )

    if request.evaluation_end_round is None:
        raise ValueError(
            "evaluation_end_round is required "
            "for model evaluation"
        )

    replay_output = (
        request.evaluation_output_root
        / "replay"
    )

    report_output = (
        request.evaluation_output_root
        / "report"
    )

    argv = [
        "--history",
        str(request.history_path),
        "--replay-output",
        str(replay_output),
        "--report-output",
        str(report_output),
        "--start-round",
        str(
            request.evaluation_start_round
        ),
        "--end-round",
        str(
            request.evaluation_end_round
        ),
        "--seed-base",
        str(request.seed),
        "--temperature",
        str(request.temperature),
        "--candidate-count",
        str(request.candidate_count),
        "--top-k",
        str(request.top_k),
        "--practical-k",
        str(request.practical_k),
        "--long-gap-window",
        str(request.long_gap_window),
        "--mode",
        str(request.mode),
    ]

    returncode = (
        model_evaluation_cli.main(
            argv
        )
    )

    status = (
        "PASS"
        if returncode == 0
        else "ERROR"
    )

    return ProductionLifecycleStageResult(
        name="model_evaluation",
        status=status,
        detail={
            "status": status,
            "returncode": returncode,
            "replay_output": replay_output,
            "report_output": report_output,
        },
    )


def run_publication_stage(
    request: argparse.Namespace,
) -> ProductionLifecycleStageResult:
    """Run the existing champion publication boundary."""

    champion_decision = (
        request.evaluation_output_root
        / "report"
        / "champion_decision.json"
    )

    payload = (
        publish_champion_cli.run_publish(
            champion_decision=(
                champion_decision
            ),
            production_registry=(
                request
                .production_registry_root
            ),
        )
    )

    return _payload_stage_result(
        name="publication",
        payload=payload,
    )


def run_audit_stage(
    request: argparse.Namespace,
) -> ProductionLifecycleStageResult:
    """Run the existing champion audit boundary."""

    payload = (
        audit_champion_cli.run_audit(
            production_registry=(
                request
                .production_registry_root
            ),
            snapshot_root=(
                request
                .production_snapshot_root
            ),
        )
    )

    return _payload_stage_result(
        name="audit",
        payload=payload,
    )


def run_prediction_stage(
    request: argparse.Namespace,
) -> ProductionLifecycleStageResult:
    """Run the existing production prediction boundary."""

    arguments = argparse.Namespace(
        history=request.history_path,
        round_no=request.round_no,
        seed=request.seed,
        temperature=request.temperature,
        candidate_count=(
            request.candidate_count
        ),
        top_k=request.top_k,
        practical_k=request.practical_k,
        long_gap_window=(
            request.long_gap_window
        ),
        mode=request.mode,
        output=(
            request.prediction_output_root
        ),
        production_registry=(
            request.production_registry_root
        ),
        production_snapshot_root=(
            request.production_snapshot_root
        ),
    )

    payload = predict_cli.run_predict(
        arguments
    )

    return _payload_stage_result(
        name="prediction",
        payload=payload,
    )
