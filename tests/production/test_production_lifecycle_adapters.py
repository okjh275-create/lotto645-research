from __future__ import annotations

import argparse
import inspect

import pytest


EXPECTED_ADAPTERS = (
    "run_model_evaluation_stage",
    "run_publication_stage",
    "run_audit_stage",
    "run_prediction_stage",
)


def _load_module():
    from lrp.production import (
        production_lifecycle_adapters,
    )

    return production_lifecycle_adapters


def test_adapter_module_is_available(
) -> None:
    module = _load_module()

    assert module is not None


@pytest.mark.parametrize(
    "name",
    EXPECTED_ADAPTERS,
)
def test_adapter_surface_is_callable(
    name: str,
) -> None:
    module = _load_module()

    assert callable(
        getattr(
            module,
            name,
        )
    )


def test_adapter_surface_is_exact(
) -> None:
    module = _load_module()

    public = {
        name
        for name, value
        in vars(module).items()
        if (
            not name.startswith("_")
            and inspect.isfunction(value)
            and value.__module__
            == module.__name__
        )
    }

    assert public == set(
        EXPECTED_ADAPTERS
    )


def test_model_evaluation_adapter_preserves_process_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()

    calls = []

    def fake_main(
        argv=None,
    ):
        calls.append(
            argv
        )
        return 0

    monkeypatch.setattr(
        module.model_evaluation_cli,
        "main",
        fake_main,
    )

    evaluation_root = (
        tmp_path
        / "evaluation"
    )

    request = argparse.Namespace(
        history_path=(
            tmp_path
            / "history.json"
        ),
        evaluation_output_root=(
            evaluation_root
        ),
        evaluation_start_round=1212,
        evaluation_end_round=1231,
        seed=20260818,
        temperature=0.85,
        candidate_count=100,
        top_k=10,
        practical_k=5,
        long_gap_window=5,
        mode="fast",
    )

    result = (
        module.run_model_evaluation_stage(
            request
        )
    )

    assert len(calls) == 1

    argv = calls[0]

    assert isinstance(
        argv,
        list,
    )

    assert argv == [
        "--history",
        str(request.history_path),
        "--replay-output",
        str(
            evaluation_root
            / "replay"
        ),
        "--report-output",
        str(
            evaluation_root
            / "report"
        ),
        "--start-round",
        "1212",
        "--end-round",
        "1231",
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
        request.mode,
    ]

    assert result.name == "model_evaluation"
    assert result.status == "PASS"
    assert result.detail["status"] == "PASS"
    assert result.detail["returncode"] == 0

    assert result.detail["replay_output"] == (
        evaluation_root
        / "replay"
    )

    assert result.detail["report_output"] == (
        evaluation_root
        / "report"
    )


def test_publication_adapter_uses_run_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()

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
        module.publish_champion_cli,
        "run_publish",
        fake_run_publish,
    )

    request = argparse.Namespace(
        evaluation_output_root=(
            tmp_path
            / "evaluation"
        ),
        production_registry_root=(
            tmp_path
            / "registry"
        ),
    )

    result = (
        module.run_publication_stage(
            request
        )
    )

    assert calls == [
        {
            "champion_decision": (
                request.evaluation_output_root
                / "report"
                / "champion_decision.json"
            ),
            "production_registry": (
                request.production_registry_root
            ),
        }
    ]

    assert result.name == "publication"
    assert result.status == "PASS"
    assert result.detail["status"] == "PASS"


def test_audit_adapter_uses_run_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()

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
            "resolved_model": "baseline",
        }

    monkeypatch.setattr(
        module.audit_champion_cli,
        "run_audit",
        fake_run_audit,
    )

    request = argparse.Namespace(
        production_registry_root=(
            tmp_path
            / "registry"
        ),
        production_snapshot_root=(
            tmp_path
            / "snapshots"
        ),
    )

    result = (
        module.run_audit_stage(
            request
        )
    )

    assert calls == [
        {
            "production_registry": (
                request.production_registry_root
            ),
            "snapshot_root": (
                request.production_snapshot_root
            ),
        }
    ]

    assert result.name == "audit"
    assert result.status == "WARN"
    assert result.detail["status"] == "WARN"


def test_prediction_adapter_uses_run_predict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()

    calls = []

    def fake_run_predict(
        arguments,
    ):
        calls.append(
            arguments
        )

        return {
            "status": "PASS",
            "round": 1232,
        }

    monkeypatch.setattr(
        module.predict_cli,
        "run_predict",
        fake_run_predict,
    )

    request = argparse.Namespace(
        history_path=(
            tmp_path
            / "history.json"
        ),
        round_no=1232,
        seed=20260818,
        temperature=0.85,
        candidate_count=100,
        top_k=10,
        practical_k=5,
        long_gap_window=5,
        mode="fast",
        prediction_output_root=(
            tmp_path
            / "prediction"
        ),
        production_registry_root=(
            tmp_path
            / "registry"
        ),
        production_snapshot_root=(
            tmp_path
            / "snapshots"
        ),
    )

    result = (
        module.run_prediction_stage(
            request
        )
    )

    assert len(calls) == 1

    arguments = calls[0]

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

        assert getattr(
            arguments,
            name,
        ) == value

    assert result.name == "prediction"
    assert result.status == "PASS"
    assert result.detail["status"] == "PASS"
