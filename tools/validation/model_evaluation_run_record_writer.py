"""Deterministic JSON writer for model-evaluation run records."""

from __future__ import annotations

import json
from pathlib import Path

from tools.validation.model_evaluation_run_record import (
    ModelEvaluationRunRecord,
)


class ModelEvaluationRunRecordWriter:
    """Persist one model-evaluation run record as canonical JSON."""

    def write_json(
        self,
        *,
        record: ModelEvaluationRunRecord,
        output: str | Path,
    ) -> Path:
        if not isinstance(
            record,
            ModelEvaluationRunRecord,
        ):
            raise TypeError(
                "record must be a "
                "ModelEvaluationRunRecord"
            )

        path = Path(output)

        if path.exists() and path.is_dir():
            raise IsADirectoryError(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        text = (
            json.dumps(
                record.as_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        path.write_text(
            text,
            encoding="utf-8",
            newline="\n",
        )

        return path
