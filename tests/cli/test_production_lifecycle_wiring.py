from __future__ import annotations

from pathlib import Path

import pytest

from lrp.cli import (
    production_lifecycle as cli,
)


def _argv(
    tmp_path: Path,
) -> list[str]:
    return [
        "--history",
        str(
            tmp_path
            / "history.json"
        ),
        "--evaluation-output",
        str(
            tmp_path
            / "evaluation"
        ),
        "--production-registry",
        str(
            tmp_path
            / "registry"
        ),
        "--production-snapshot-root",
        str(
            tmp_path
            / "snapshots"
        ),
        "--prediction-output",
        str(
            tmp_path
            / "prediction"
        ),
        "--round",
        "1232",
        "--seed",
        "20260818",
        "--temperature",
        "0.85",
        "--candidate-count",
        "100",
        "--top-k",
        "10",
        "--practical-k",
        "5",
        "--mode",
        "fast",
        "--evaluation-start-round",
        "1212",
        "--evaluation-end-round",
        "1231",
        "--long-gap-window",
        "5",
    ]


def test_parser_accepts_real_lifecycle_arguments(
    tmp_path: Path,
) -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        _argv(tmp_path)
    )

    assert args.history == (
        tmp_path
        / "history.json"
    )

    assert (
        args.evaluation_output
        == tmp_path
        / "evaluation"
    )

    assert (
        args.production_registry
        == tmp_path
        / "registry"
    )

    assert (
        args.production_snapshot_root
        == tmp_path
        / "snapshots"
    )

    assert (
        args.prediction_output
        == tmp_path
        / "prediction"
    )

    assert args.round_no == 1232
    assert args.seed == 20260818

    assert (
        args.evaluation_start_round
        == 1212
    )

    assert (
        args.evaluation_end_round
        == 1231
    )

    assert args.long_gap_window == 5


def test_main_constructs_request_and_wires_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {}

    class FakeResult:
        status = "PASS"
        stages = ()

    class FakeService:
        def __init__(
            self,
            *,
            model_evaluation,
            publication,
            audit,
            prediction,
        ) -> None:
            observed[
                "dependencies"
            ] = (
                model_evaluation,
                publication,
                audit,
                prediction,
            )

        def run(
            self,
            request,
        ):
            observed[
                "request"
            ] = request

            return FakeResult()

    monkeypatch.setattr(
        cli,
        "ProductionLifecycleService",
        FakeService,
    )

    exit_code = cli.main(
        _argv(tmp_path)
    )

    assert exit_code == 0

    dependencies = observed[
        "dependencies"
    ]

    assert dependencies == (
        cli.run_model_evaluation_stage,
        cli.run_publication_stage,
        cli.run_audit_stage,
        cli.run_prediction_stage,
    )

    request = observed[
        "request"
    ]

    assert request.history_path == (
        tmp_path
        / "history.json"
    )

    assert (
        request.evaluation_output_root
        == tmp_path
        / "evaluation"
    )

    assert (
        request.production_registry_root
        == tmp_path
        / "registry"
    )

    assert (
        request.production_snapshot_root
        == tmp_path
        / "snapshots"
    )

    assert (
        request.prediction_output_root
        == tmp_path
        / "prediction"
    )

    assert request.round_no == 1232
    assert request.seed == 20260818

    assert (
        request.evaluation_start_round
        == 1212
    )

    assert (
        request.evaluation_end_round
        == 1231
    )

    assert request.long_gap_window == 5


def test_main_prints_success_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeResult:
        status = "WARN"
        stages = ()

    class FakeService:
        def __init__(
            self,
            **kwargs,
        ) -> None:
            pass

        def run(
            self,
            request,
        ):
            return FakeResult()

    monkeypatch.setattr(
        cli,
        "ProductionLifecycleService",
        FakeService,
    )

    exit_code = cli.main(
        _argv(tmp_path)
    )

    captured = capsys.readouterr()

    assert exit_code == 0

    assert '"status": "WARN"' in (
        captured.out
    )


def test_main_returns_nonzero_for_error_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResult:
        status = "ERROR"
        stages = ()

    class FakeService:
        def __init__(
            self,
            **kwargs,
        ) -> None:
            pass

        def run(
            self,
            request,
        ):
            return FakeResult()

    monkeypatch.setattr(
        cli,
        "ProductionLifecycleService",
        FakeService,
    )

    exit_code = cli.main(
        _argv(tmp_path)
    )

    assert exit_code == 1
