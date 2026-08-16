from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _history_payload() -> list[dict[str, object]]:
    return [
        {
            "round": 1227,
            "nums": [
                1,
                2,
                3,
                4,
                5,
                6,
            ],
            "bonus": 7,
        },
        {
            "round": 1228,
            "nums": [
                8,
                9,
                10,
                11,
                12,
                13,
            ],
            "bonus": 14,
        },
    ]


def _fake_result(
    artifact_path: Path,
) -> object:
    promotion = SimpleNamespace(
        promoted=False,
    )

    selection = SimpleNamespace(
        selected_model=None,
        promotion=promotion,
    )

    champion = SimpleNamespace(
        selection=selection,
    )

    ranking = SimpleNamespace(
        champion="baseline",
    )

    matrix = SimpleNamespace(
        ranking=ranking,
    )

    return SimpleNamespace(
        matrix=matrix,
        champion=champion,
        artifact_path=artifact_path,
    )


def test_cli_writes_evaluation_run_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.validation.run_model_evaluation as cli

    history_path = (
        tmp_path
        / "history.json"
    )

    history_path.write_text(
        json.dumps(
            _history_payload(),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    replay_root = (
        tmp_path
        / "replay"
    )

    report_root = (
        tmp_path
        / "report"
    )

    champion_artifact = (
        report_root
        / "champion_decision.json"
    )

    champion_artifact.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    champion_artifact.write_text(
        "{}\n",
        encoding="utf-8",
    )

    class FakeService:
        def run(
            self,
            **kwargs: object,
        ) -> object:
            return _fake_result(
                champion_artifact
            )

    monkeypatch.setattr(
        cli,
        "HistoricalModelEvaluationOrchestrationService",
        FakeService,
    )

    exit_code = cli.run(
        [
            "--history",
            str(history_path),
            "--replay-output",
            str(replay_root),
            "--report-output",
            str(report_root),
            "--start-round",
            "1228",
            "--end-round",
            "1228",
            "--window-size",
            "1",
            "--models",
            "baseline",
            "calibration",
            "--candidate-count",
            "100",
        ]
    )

    assert exit_code == 0

    run_record_path = (
        report_root
        / "evaluation_run.json"
    )

    assert run_record_path.is_file()

    payload = json.loads(
        run_record_path.read_text(
            encoding="utf-8"
        )
    )

    assert payload["history_path"] == (
        history_path.as_posix()
    )

    assert payload["model_names"] == [
        "baseline",
        "calibration",
    ]

    assert payload["round_range"] == {
        "start_round": 1228,
        "end_round": 1228,
    }

    assert payload["windows"] == [
        {
            "name": "window-001",
            "start_round": 1228,
            "end_round": 1228,
            "round_count": 1,
        }
    ]

    assert payload["replay_config"][
        "candidate_count"
    ] == 100

    assert payload["champion"] == {
        "ranking_champion": "baseline",
        "selected_model": None,
        "promoted": False,
    }

    assert payload["champion_artifact"] == (
        champion_artifact.as_posix()
    )

    assert isinstance(
        payload["run_id"],
        str,
    )

    assert len(
        payload["run_id"]
    ) == 16


def test_cli_stdout_reports_run_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import tools.validation.run_model_evaluation as cli

    history_path = (
        tmp_path
        / "history.json"
    )

    history_path.write_text(
        json.dumps(
            _history_payload(),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report_root = (
        tmp_path
        / "report"
    )

    champion_artifact = (
        report_root
        / "champion_decision.json"
    )

    champion_artifact.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    champion_artifact.write_text(
        "{}\n",
        encoding="utf-8",
    )

    class FakeService:
        def run(
            self,
            **kwargs: object,
        ) -> object:
            return _fake_result(
                champion_artifact
            )

    monkeypatch.setattr(
        cli,
        "HistoricalModelEvaluationOrchestrationService",
        FakeService,
    )

    exit_code = cli.run(
        [
            "--history",
            str(history_path),
            "--replay-output",
            str(tmp_path / "replay"),
            "--report-output",
            str(report_root),
            "--start-round",
            "1228",
            "--end-round",
            "1228",
            "--window-size",
            "1",
        ]
    )

    assert exit_code == 0

    captured = capsys.readouterr()

    payload = json.loads(
        captured.out
    )

    assert isinstance(
        payload["run_id"],
        str,
    )

    assert len(
        payload["run_id"]
    ) == 16

    assert payload[
        "run_record_path"
    ] == str(
        report_root
        / "evaluation_run.json"
    )


def test_identical_cli_inputs_produce_same_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import tools.validation.run_model_evaluation as cli

    history_path = (
        tmp_path
        / "history.json"
    )

    history_path.write_text(
        json.dumps(
            _history_payload(),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report_root = (
        tmp_path
        / "report"
    )

    champion_artifact = (
        report_root
        / "champion_decision.json"
    )

    champion_artifact.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    champion_artifact.write_text(
        "{}\n",
        encoding="utf-8",
    )

    class FakeService:
        def run(
            self,
            **kwargs: object,
        ) -> object:
            return _fake_result(
                champion_artifact
            )

    monkeypatch.setattr(
        cli,
        "HistoricalModelEvaluationOrchestrationService",
        FakeService,
    )

    argv = [
        "--history",
        str(history_path),
        "--replay-output",
        str(tmp_path / "replay"),
        "--report-output",
        str(report_root),
        "--start-round",
        "1228",
        "--end-round",
        "1228",
        "--window-size",
        "1",
    ]

    assert cli.run(argv) == 0

    first_output = json.loads(
        capsys.readouterr().out
    )

    assert cli.run(argv) == 0

    second_output = json.loads(
        capsys.readouterr().out
    )

    assert (
        first_output["run_id"]
        == second_output["run_id"]
    )

    run_record_path = (
        report_root
        / "evaluation_run.json"
    )

    assert run_record_path.is_file()

    stored = json.loads(
        run_record_path.read_text(
            encoding="utf-8"
        )
    )

    assert stored["run_id"] == (
        first_output["run_id"]
    )
