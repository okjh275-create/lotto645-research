from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import lrp.cli.predict as cli


class _FakeStatistics:
    class _Module:
        class DrawRecord:
            pass

    module = _Module()

    def create_config(
        self,
        **kwargs: object,
    ) -> object:
        return object()


class _FakeResult:
    generated_count = 100


class _FakePipeline:
    statistics = _FakeStatistics()

    def run(
        self,
        *args: object,
        **kwargs: object,
    ) -> _FakeResult:
        return _FakeResult()


def _arguments(
    *,
    champion_decision: Path | None = None,
    production_snapshot_root: Path | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        history="history.json",
        round_no=1232,
        seed=20260816,
        temperature=0.85,
        candidate_count=100,
        top_k=10,
        practical_k=5,
        long_gap_window=5,
        mode="fast",
        output="predictions",
        print_json=False,
        champion_decision=champion_decision,
        production_snapshot_root=(
            production_snapshot_root
        ),
    )


def _patch_prediction_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "load_history",
        lambda path: (),
    )

    monkeypatch.setattr(
        cli,
        "history_until_round",
        lambda history, target_round: (),
    )

    monkeypatch.setattr(
        cli,
        "to_statistics_draws",
        lambda history, draw_type: (),
    )

    monkeypatch.setattr(
        cli,
        "previous_numbers",
        lambda history: (),
    )

    monkeypatch.setattr(
        cli,
        "long_gap_numbers",
        lambda history, recent_draw_count: (45,),
    )

    monkeypatch.setattr(
        cli,
        "prediction_to_dict",
        lambda result: {
            "sets": [],
            "top5_practical": [],
            "diversity": {},
        },
    )

    monkeypatch.setattr(
        cli,
        "write_prediction_artifacts",
        lambda payload, output_root: {
            "prediction_path": "prediction.json",
        },
    )

    monkeypatch.setattr(
        cli.PredictionPipeline,
        "load",
        lambda **kwargs: _FakePipeline(),
    )


def test_default_run_reports_activation_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_prediction_runtime(
        monkeypatch
    )

    summary = cli.run_predict(
        _arguments()
    )

    assert summary[
        "production_activation"
    ] == {
        "enabled": False,
    }


def test_opt_in_run_reports_activation_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_prediction_runtime(
        monkeypatch
    )

    decision_path = (
        tmp_path
        / "champion_decision.json"
    )

    decision_path.write_text(
        json.dumps(
            {
                "selection": {
                    "selected_model": "combined",
                },
            }
        ),
        encoding="utf-8",
    )

    summary = cli.run_predict(
        _arguments(
            champion_decision=decision_path,
            production_snapshot_root=(
                tmp_path
                / "snapshots"
            ),
        )
    )

    assert summary[
        "production_activation"
    ] == {
        "enabled": True,
        "requested_model": "combined",
        "resolved_model": "combined",
        "fallback_applied": False,
        "fallback_reason": None,
    }


def test_fallback_run_reports_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_prediction_runtime(
        monkeypatch
    )

    decision_path = (
        tmp_path
        / "champion_decision.json"
    )

    decision_path.write_text(
        json.dumps(
            {
                "selection": {
                    "selected_model": None,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = cli.run_predict(
        _arguments(
            champion_decision=decision_path,
            production_snapshot_root=(
                tmp_path
                / "snapshots"
            ),
        )
    )

    assert summary[
        "production_activation"
    ] == {
        "enabled": True,
        "requested_model": None,
        "resolved_model": "baseline",
        "fallback_applied": True,
        "fallback_reason": "no_selected_model",
    }


def test_provenance_does_not_leak_into_prediction_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_prediction_runtime(
        monkeypatch
    )

    decision_path = (
        tmp_path
        / "champion_decision.json"
    )

    decision_path.write_text(
        json.dumps(
            {
                "selection": {
                    "selected_model": "baseline",
                },
            }
        ),
        encoding="utf-8",
    )

    summary = cli.run_predict(
        _arguments(
            champion_decision=decision_path,
            production_snapshot_root=(
                tmp_path
                / "snapshots"
            ),
        )
    )

    assert (
        "production_activation"
        in summary
    )

    assert (
        "production_activation"
        not in summary["payload"]
    )
