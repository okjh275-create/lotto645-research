from __future__ import annotations

from pathlib import Path

import pytest

from lrp.production import (
    production_lifecycle_adapters
    as adapters,
)

from lrp.production.production_lifecycle import (
    ProductionLifecycleRequest,
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
        evaluation_start_round=1212,
        evaluation_end_round=1231,
        long_gap_window=5,
    )


def test_request_contains_real_execution_fields(
    tmp_path: Path,
) -> None:
    request = _request(
        tmp_path
    )

    assert (
        request.evaluation_start_round
        == 1212
    )

    assert (
        request.evaluation_end_round
        == 1231
    )

    assert (
        request.long_gap_window
        == 5
    )


def test_model_evaluation_adapter_maps_real_cli_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(
        tmp_path
    )

    calls: list[
        list[str]
    ] = []

    def fake_main(
        argv=None,
    ) -> int:
        assert argv is not None

        calls.append(
            list(argv)
        )

        return 0

    monkeypatch.setattr(
        adapters.model_evaluation_cli,
        "main",
        fake_main,
    )

    result = (
        adapters
        .run_model_evaluation_stage(
            request
        )
    )

    assert len(calls) == 1

    argv = calls[0]

    expected_pairs = {
        "--history": str(
            request.history_path
        ),
        "--replay-output": str(
            request.evaluation_output_root
            / "replay"
        ),
        "--report-output": str(
            request.evaluation_output_root
            / "report"
        ),
        "--start-round": str(
            request.evaluation_start_round
        ),
        "--end-round": str(
            request.evaluation_end_round
        ),
        "--seed-base": str(
            request.seed
        ),
        "--temperature": str(
            request.temperature
        ),
        "--candidate-count": str(
            request.candidate_count
        ),
        "--top-k": str(
            request.top_k
        ),
        "--practical-k": str(
            request.practical_k
        ),
        "--long-gap-window": str(
            request.long_gap_window
        ),
        "--mode": request.mode,
    }

    assert "--output" not in argv

    for option, value in (
        expected_pairs.items()
    ):
        index = argv.index(
            option
        )

        assert (
            argv[index + 1]
            == value
        )

    assert result.name == "model_evaluation"
    assert result.status == "PASS"
    assert result.detail["status"] == "PASS"


def test_publication_adapter_derives_champion_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(
        tmp_path
    )

    calls = []

    def fake_run_publish(
        *,
        champion_decision,
        production_registry,
    ):
        calls.append(
            {
                "champion_decision":
                champion_decision,
                "production_registry":
                production_registry,
            }
        )

        return {
            "status": "PASS",
            "selected_model": None,
        }

    monkeypatch.setattr(
        adapters.publish_champion_cli,
        "run_publish",
        fake_run_publish,
    )

    result = (
        adapters
        .run_publication_stage(
            request
        )
    )

    assert calls == [
        {
            "champion_decision": (
                request
                .evaluation_output_root
                / "report"
                / "champion_decision.json"
            ),
            "production_registry": (
                request
                .production_registry_root
            ),
        }
    ]

    assert result.name == "publication"
    assert result.status == "PASS"
    assert result.detail["status"] == "PASS"


def test_audit_adapter_maps_registry_and_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(
        tmp_path
    )

    calls = []

    def fake_run_audit(
        *,
        production_registry,
        snapshot_root,
    ):
        calls.append(
            {
                "production_registry":
                production_registry,
                "snapshot_root":
                snapshot_root,
            }
        )

        return {
            "status": "WARN",
            "resolved_model":
                "baseline",
        }

    monkeypatch.setattr(
        adapters.audit_champion_cli,
        "run_audit",
        fake_run_audit,
    )

    result = (
        adapters
        .run_audit_stage(
            request
        )
    )

    assert calls == [
        {
            "production_registry": (
                request
                .production_registry_root
            ),
            "snapshot_root": (
                request
                .production_snapshot_root
            ),
        }
    ]

    assert result.name == "audit"
    assert result.status == "WARN"
    assert result.detail["status"] == "WARN"


def test_prediction_adapter_builds_production_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(
        tmp_path
    )

    observed = []

    def fake_run_predict(
        arguments,
    ):
        observed.append(
            arguments
        )

        return {
            "status": "PASS",
            "round": request.round_no,
            "production_activation": {
                "enabled": True,
                "resolved_model":
                    "baseline",
            },
        }

    monkeypatch.setattr(
        adapters.predict_cli,
        "run_predict",
        fake_run_predict,
    )

    result = (
        adapters
        .run_prediction_stage(
            request
        )
    )

    assert len(observed) == 1

    arguments = observed[0]

    expected = {
        "history":
            request.history_path,
        "round_no":
            request.round_no,
        "seed":
            request.seed,
        "temperature":
            request.temperature,
        "candidate_count":
            request.candidate_count,
        "top_k":
            request.top_k,
        "practical_k":
            request.practical_k,
        "long_gap_window":
            request.long_gap_window,
        "mode":
            request.mode,
        "output":
            request.prediction_output_root,
        "production_registry":
            request.production_registry_root,
        "production_snapshot_root":
            request.production_snapshot_root,
    }

    for name, value in (
        expected.items()
    ):
        assert hasattr(
            arguments,
            name,
        )

        assert (
            getattr(
                arguments,
                name,
            )
            == value
        )

    assert result.name == "prediction"
    assert result.status == "PASS"
    assert result.detail["status"] == "PASS"


def test_evaluation_artifact_paths_are_deterministic(
    tmp_path: Path,
) -> None:
    request = _request(
        tmp_path
    )

    replay = (
        request
        .evaluation_output_root
        / "replay"
    )

    report = (
        request
        .evaluation_output_root
        / "report"
    )

    decision = (
        report
        / "champion_decision.json"
    )

    assert replay == (
        tmp_path
        / "evaluation"
        / "replay"
    )

    assert report == (
        tmp_path
        / "evaluation"
        / "report"
    )

    assert decision == (
        tmp_path
        / "evaluation"
        / "report"
        / "champion_decision.json"
    )
