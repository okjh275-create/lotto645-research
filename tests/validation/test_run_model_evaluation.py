from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_parser_exposes_operational_contract() -> None:
    from tools.validation.run_model_evaluation import (
        build_parser,
    )

    parser = build_parser()

    assert parser.prog == "model-evaluation"

    args = parser.parse_args(
        [
            "--history",
            "history.json",
            "--replay-output",
            "replay",
            "--report-output",
            "report",
            "--start-round",
            "1200",
            "--end-round",
            "1228",
            "--window-size",
            "10",
            "--models",
            "baseline",
            "calibration",
        ]
    )

    assert args.history == Path("history.json")
    assert args.replay_output == Path("replay")
    assert args.report_output == Path("report")
    assert args.start_round == 1200
    assert args.end_round == 1228
    assert args.window_size == 10

    assert args.models == [
        "baseline",
        "calibration",
    ]


def test_parser_exposes_replay_defaults() -> None:
    from tools.validation.run_model_evaluation import (
        build_parser,
    )

    args = build_parser().parse_args(
        [
            "--history",
            "history.json",
            "--replay-output",
            "replay",
            "--report-output",
            "report",
            "--start-round",
            "1228",
            "--end-round",
            "1228",
        ]
    )

    assert args.seed_base == 20260802
    assert args.temperature == pytest.approx(0.85)
    assert args.candidate_count == 1000
    assert args.top_k == 20
    assert args.practical_k == 5
    assert args.long_gap_window == 5
    assert args.confidence == pytest.approx(0.8)
    assert args.mode == "fast"

    assert args.models == [
        "baseline",
        "calibration",
    ]


def test_run_delegates_to_orchestration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import tools.validation.run_model_evaluation as cli

    history_path = tmp_path / "history.json"

    history_payload = [
        {
            "round": 1227,
            "nums": [1, 2, 3, 4, 5, 6],
            "bonus": 7,
        },
        {
            "round": 1228,
            "nums": [8, 9, 10, 11, 12, 13],
            "bonus": 14,
        },
    ]

    history_path.write_text(
        json.dumps(
            history_payload,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    artifact_path = (
        tmp_path
        / "report"
        / "champion_decision.json"
    )

    artifact_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_path.write_text(
        "{}\n",
        encoding="utf-8",
    )

    calls: dict[str, object] = {}

    class FakePromotion:
        promoted = False

    class FakeSelection:
        selected_model = "calibration"
        promotion = FakePromotion()

    class FakeChampion:
        selection = FakeSelection()

    class FakeRanking:
        champion = "calibration"

    class FakeMatrix:
        ranking = FakeRanking()

    class FakeResult:
        matrix = FakeMatrix()
        champion = FakeChampion()

        def __init__(
            self,
            result_artifact_path: Path,
        ) -> None:
            self.artifact_path = (
                result_artifact_path
            )

    class FakeService:
        def run(
            self,
            **kwargs: object,
        ) -> FakeResult:
            calls.update(kwargs)
            return FakeResult(
                artifact_path
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
            str(tmp_path / "report"),
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

    assert tuple(calls["model_names"]) == (
        "baseline",
        "calibration",
    )

    history = tuple(calls["history"])

    assert len(history) == 2
    assert history[0].round_no == 1227
    assert history[0].numbers == (
        1,
        2,
        3,
        4,
        5,
        6,
    )
    assert history[0].bonus == 7

    assert history[1].round_no == 1228
    assert history[1].numbers == (
        8,
        9,
        10,
        11,
        12,
        13,
    )
    assert history[1].bonus == 14

    windows = tuple(calls["windows"])

    assert len(windows) == 1
    assert windows[0].start_round == 1228
    assert windows[0].end_round == 1228

    config = calls["base_config"]

    assert config.start_round == 1228
    assert config.end_round == 1228
    assert config.candidate_count == 100

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["status"] == "PASS"
    assert payload["ranking_champion"] == "calibration"
    assert payload["selected_model"] == "calibration"
    assert payload["promoted"] is False

    assert payload["artifact_path"] == str(
        artifact_path
    )


def test_run_rejects_missing_history(
    tmp_path: Path,
) -> None:
    from tools.validation.run_model_evaluation import run

    with pytest.raises(SystemExit) as exc_info:
        run(
            [
                "--history",
                str(tmp_path / "missing.json"),
                "--replay-output",
                str(tmp_path / "replay"),
                "--report-output",
                str(tmp_path / "report"),
                "--start-round",
                "1228",
                "--end-round",
                "1228",
            ]
        )

    assert exc_info.value.code == 2


def test_run_rejects_invalid_round_range(
    tmp_path: Path,
) -> None:
    from tools.validation.run_model_evaluation import run

    history_path = tmp_path / "history.json"

    history_path.write_text(
        "[]\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        run(
            [
                "--history",
                str(history_path),
                "--replay-output",
                str(tmp_path / "replay"),
                "--report-output",
                str(tmp_path / "report"),
                "--start-round",
                "1228",
                "--end-round",
                "1200",
            ]
        )

    assert exc_info.value.code == 2


def test_run_rejects_invalid_window_size(
    tmp_path: Path,
) -> None:
    from tools.validation.run_model_evaluation import run

    history_path = tmp_path / "history.json"

    history_path.write_text(
        "[]\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        run(
            [
                "--history",
                str(history_path),
                "--replay-output",
                str(tmp_path / "replay"),
                "--report-output",
                str(tmp_path / "report"),
                "--start-round",
                "1228",
                "--end-round",
                "1228",
                "--window-size",
                "0",
            ]
        )

    assert exc_info.value.code == 2
