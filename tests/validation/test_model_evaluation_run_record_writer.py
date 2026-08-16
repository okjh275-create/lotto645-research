from __future__ import annotations

import json
from pathlib import Path

import pytest

from lrp.evaluation import EvaluationWindow

from tools.validation.historical_replay_models import (
    ReplayConfig,
)
from tools.validation.model_evaluation_run_record import (
    ModelEvaluationRunRecord,
)


def _record() -> ModelEvaluationRunRecord:
    return ModelEvaluationRunRecord.build(
        history_path=Path(
            "artifacts/history.json"
        ),
        model_names=(
            "baseline",
            "calibration",
        ),
        windows=(
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
        ),
        replay_config=ReplayConfig(
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
        ),
        ranking_champion="baseline",
        selected_model=None,
        promoted=False,
        champion_artifact=Path(
            "report/champion_decision.json"
        ),
    )


def test_writer_writes_deterministic_json(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_record_writer import (
        ModelEvaluationRunRecordWriter,
    )

    record = _record()

    output = (
        tmp_path
        / "nested"
        / "evaluation_run.json"
    )

    result = (
        ModelEvaluationRunRecordWriter()
        .write_json(
            record=record,
            output=output,
        )
    )

    assert result == output
    assert output.is_file()

    expected = (
        json.dumps(
            record.as_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    assert output.read_text(
        encoding="utf-8"
    ) == expected


def test_writer_creates_parent_directory(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_record_writer import (
        ModelEvaluationRunRecordWriter,
    )

    output = (
        tmp_path
        / "a"
        / "b"
        / "evaluation_run.json"
    )

    ModelEvaluationRunRecordWriter().write_json(
        record=_record(),
        output=output,
    )

    assert output.is_file()


def test_writer_preserves_run_id(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_record_writer import (
        ModelEvaluationRunRecordWriter,
    )

    record = _record()

    output = (
        tmp_path
        / "evaluation_run.json"
    )

    ModelEvaluationRunRecordWriter().write_json(
        record=record,
        output=output,
    )

    payload = json.loads(
        output.read_text(
            encoding="utf-8"
        )
    )

    assert payload["run_id"] == (
        record.run_id
    )


def test_writer_rejects_wrong_record_type(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_record_writer import (
        ModelEvaluationRunRecordWriter,
    )

    with pytest.raises(
        TypeError,
        match="ModelEvaluationRunRecord",
    ):
        ModelEvaluationRunRecordWriter().write_json(
            record={},
            output=(
                tmp_path
                / "evaluation_run.json"
            ),
        )


def test_writer_rejects_directory_output(
    tmp_path: Path,
) -> None:
    from tools.validation.model_evaluation_run_record_writer import (
        ModelEvaluationRunRecordWriter,
    )

    with pytest.raises(
        IsADirectoryError,
    ):
        ModelEvaluationRunRecordWriter().write_json(
            record=_record(),
            output=tmp_path,
        )
