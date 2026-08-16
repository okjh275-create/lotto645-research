from __future__ import annotations

from pathlib import Path

import pytest

from lrp.evaluation import EvaluationWindow

from tools.validation.historical_replay_models import (
    ReplayConfig,
)


def _config() -> ReplayConfig:
    return ReplayConfig(
        start_round=1200,
        end_round=1228,
        seed_base=20260802,
        temperature=0.85,
        candidate_count=1000,
        top_k=20,
        practical_k=5,
        long_gap_window=5,
        confidence=0.8,
        mode="fast",
    )


def _windows() -> tuple[EvaluationWindow, ...]:
    return (
        EvaluationWindow(
            name="window-001",
            start_round=1200,
            end_round=1209,
        ),
        EvaluationWindow(
            name="window-002",
            start_round=1210,
            end_round=1219,
        ),
        EvaluationWindow(
            name="window-003",
            start_round=1220,
            end_round=1228,
        ),
    )


def test_run_record_is_deterministic() -> None:
    from tools.validation.model_evaluation_run_record import (
        ModelEvaluationRunRecord,
    )

    first = ModelEvaluationRunRecord.build(
        history_path=Path("history.json"),
        model_names=(
            "baseline",
            "calibration",
        ),
        windows=_windows(),
        replay_config=_config(),
        ranking_champion="baseline",
        selected_model=None,
        promoted=False,
        champion_artifact=Path(
            "report/champion_decision.json"
        ),
    )

    second = ModelEvaluationRunRecord.build(
        history_path=Path("history.json"),
        model_names=(
            "baseline",
            "calibration",
        ),
        windows=_windows(),
        replay_config=_config(),
        ranking_champion="baseline",
        selected_model=None,
        promoted=False,
        champion_artifact=Path(
            "report/champion_decision.json"
        ),
    )

    assert first == second
    assert first.run_id == second.run_id


def test_run_id_changes_when_inputs_change() -> None:
    from tools.validation.model_evaluation_run_record import (
        ModelEvaluationRunRecord,
    )

    first = ModelEvaluationRunRecord.build(
        history_path=Path("history.json"),
        model_names=(
            "baseline",
            "calibration",
        ),
        windows=_windows(),
        replay_config=_config(),
        ranking_champion="baseline",
        selected_model=None,
        promoted=False,
        champion_artifact=Path(
            "report/champion_decision.json"
        ),
    )

    changed_config = ReplayConfig(
        start_round=1200,
        end_round=1228,
        seed_base=20260803,
        temperature=0.85,
        candidate_count=1000,
        top_k=20,
        practical_k=5,
        long_gap_window=5,
        confidence=0.8,
        mode="fast",
    )

    second = ModelEvaluationRunRecord.build(
        history_path=Path("history.json"),
        model_names=(
            "baseline",
            "calibration",
        ),
        windows=_windows(),
        replay_config=changed_config,
        ranking_champion="baseline",
        selected_model=None,
        promoted=False,
        champion_artifact=Path(
            "report/champion_decision.json"
        ),
    )

    assert first.run_id != second.run_id


def test_run_record_exposes_complete_provenance() -> None:
    from tools.validation.model_evaluation_run_record import (
        ModelEvaluationRunRecord,
    )

    record = ModelEvaluationRunRecord.build(
        history_path=Path("history.json"),
        model_names=(
            "baseline",
            "calibration",
        ),
        windows=_windows(),
        replay_config=_config(),
        ranking_champion="baseline",
        selected_model=None,
        promoted=False,
        champion_artifact=Path(
            "report/champion_decision.json"
        ),
    )

    payload = record.as_dict()

    assert payload["history_path"] == "history.json"

    assert payload["model_names"] == [
        "baseline",
        "calibration",
    ]

    assert payload["round_range"] == {
        "start_round": 1200,
        "end_round": 1228,
    }

    assert payload["windows"] == [
        {
            "name": "window-001",
            "start_round": 1200,
            "end_round": 1209,
            "round_count": 10,
        },
        {
            "name": "window-002",
            "start_round": 1210,
            "end_round": 1219,
            "round_count": 10,
        },
        {
            "name": "window-003",
            "start_round": 1220,
            "end_round": 1228,
            "round_count": 9,
        },
    ]

    assert payload["replay_config"] == {
        "seed_base": 20260802,
        "temperature": 0.85,
        "candidate_count": 1000,
        "top_k": 20,
        "practical_k": 5,
        "long_gap_window": 5,
        "confidence": 0.8,
        "mode": "fast",
    }

    assert payload["champion"] == {
        "ranking_champion": "baseline",
        "selected_model": None,
        "promoted": False,
    }

    assert payload["champion_artifact"] == (
        "report/champion_decision.json"
    )

    assert payload["run_id"] == record.run_id


def test_run_record_rejects_empty_models() -> None:
    from tools.validation.model_evaluation_run_record import (
        ModelEvaluationRunRecord,
    )

    with pytest.raises(
        ValueError,
        match="model_names must not be empty",
    ):
        ModelEvaluationRunRecord.build(
            history_path=Path("history.json"),
            model_names=(),
            windows=_windows(),
            replay_config=_config(),
            ranking_champion="baseline",
            selected_model=None,
            promoted=False,
            champion_artifact=Path(
                "report/champion_decision.json"
            ),
        )


def test_run_record_rejects_empty_windows() -> None:
    from tools.validation.model_evaluation_run_record import (
        ModelEvaluationRunRecord,
    )

    with pytest.raises(
        ValueError,
        match="windows must not be empty",
    ):
        ModelEvaluationRunRecord.build(
            history_path=Path("history.json"),
            model_names=("baseline",),
            windows=(),
            replay_config=_config(),
            ranking_champion="baseline",
            selected_model=None,
            promoted=False,
            champion_artifact=Path(
                "report/champion_decision.json"
            ),
        )


def test_run_record_requires_replay_config() -> None:
    from tools.validation.model_evaluation_run_record import (
        ModelEvaluationRunRecord,
    )

    with pytest.raises(
        TypeError,
        match="ReplayConfig",
    ):
        ModelEvaluationRunRecord.build(
            history_path=Path("history.json"),
            model_names=("baseline",),
            windows=_windows(),
            replay_config=object(),
            ranking_champion="baseline",
            selected_model=None,
            promoted=False,
            champion_artifact=Path(
                "report/champion_decision.json"
            ),
        )
