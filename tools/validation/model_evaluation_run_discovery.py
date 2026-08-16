"""Discover historical model-evaluation run artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelEvaluationRunDiscoveryRecord:
    """Metadata for one discovered model-evaluation run."""

    run_id: str
    root: Path
    history_path: Path
    model_names: tuple[str, ...]
    start_round: int
    end_round: int
    window_count: int
    ranking_champion: str | None
    selected_model: str | None
    promoted: bool
    champion_artifact: Path
    missing_files: tuple[str, ...]
    status: str

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "root": str(self.root),
            "history_path": self.history_path.as_posix(),
            "model_names": list(
                self.model_names
            ),
            "round_range": {
                "start_round": self.start_round,
                "end_round": self.end_round,
            },
            "window_count": self.window_count,
            "champion": {
                "ranking_champion": (
                    self.ranking_champion
                ),
                "selected_model": (
                    self.selected_model
                ),
                "promoted": self.promoted,
            },
            "champion_artifact": (
                self.champion_artifact.as_posix()
            ),
            "missing_files": list(
                self.missing_files
            ),
            "status": self.status,
        }


class ModelEvaluationRunDiscovery:
    """Discover model-evaluation provenance records."""

    RUN_RECORD_NAME = "evaluation_run.json"

    def discover(
        self,
        root: str | Path,
    ) -> tuple[
        ModelEvaluationRunDiscoveryRecord,
        ...,
    ]:
        root = Path(root)

        if not root.exists():
            raise FileNotFoundError(root)

        if not root.is_dir():
            raise NotADirectoryError(root)

        records = [
            self._record(path)
            for path in root.rglob(
                self.RUN_RECORD_NAME
            )
        ]

        records.sort(
            key=lambda record: (
                record.start_round,
                record.end_round,
                record.run_id,
                str(record.root),
            )
        )

        return tuple(records)

    def _record(
        self,
        path: Path,
    ) -> ModelEvaluationRunDiscoveryRecord:
        payload = self._load_object(path)

        run_id = self._non_empty_string(
            payload,
            "run_id",
        )

        history_path = Path(
            self._non_empty_string(
                payload,
                "history_path",
            )
        )

        model_names = self._model_names(
            payload
        )

        round_range = self._mapping(
            payload,
            "round_range",
        )

        start_round = self._integer(
            round_range,
            "start_round",
        )

        end_round = self._integer(
            round_range,
            "end_round",
        )

        if start_round < 1:
            raise ValueError(
                "start_round must be greater "
                "than or equal to 1"
            )

        if end_round < start_round:
            raise ValueError(
                "end_round must be greater "
                "than or equal to start_round"
            )

        windows = payload.get("windows")

        if not isinstance(windows, list):
            raise TypeError(
                "windows must be a list"
            )

        if not windows:
            raise ValueError(
                "windows must not be empty"
            )

        for window in windows:
            if not isinstance(window, dict):
                raise TypeError(
                    "each window must be an object"
                )

        champion = self._mapping(
            payload,
            "champion",
        )

        ranking_champion = (
            self._optional_string(
                champion,
                "ranking_champion",
            )
        )

        selected_model = (
            self._optional_string(
                champion,
                "selected_model",
            )
        )

        promoted = champion.get(
            "promoted"
        )

        if not isinstance(promoted, bool):
            raise TypeError(
                "promoted must be a boolean"
            )

        champion_artifact_value = (
            self._non_empty_string(
                payload,
                "champion_artifact",
            )
        )

        champion_artifact = Path(
            champion_artifact_value
        )

        missing_files = (
            self._missing_artifacts(
                run_record_path=path,
                artifact=champion_artifact,
            )
        )

        return ModelEvaluationRunDiscoveryRecord(
            run_id=run_id,
            root=path.parent,
            history_path=history_path,
            model_names=model_names,
            start_round=start_round,
            end_round=end_round,
            window_count=len(windows),
            ranking_champion=(
                ranking_champion
            ),
            selected_model=selected_model,
            promoted=promoted,
            champion_artifact=(
                champion_artifact
            ),
            missing_files=missing_files,
            status=(
                "PASS"
                if not missing_files
                else "INCOMPLETE"
            ),
        )

    @staticmethod
    def _missing_artifacts(
        *,
        run_record_path: Path,
        artifact: Path,
    ) -> tuple[str, ...]:
        if artifact.is_absolute():
            candidates = (
                artifact,
            )
        else:
            candidates = (
                artifact,
                run_record_path.parent
                / artifact,
            )

        if any(
            candidate.is_file()
            for candidate in candidates
        ):
            return ()

        return (
            artifact.as_posix(),
        )

    @staticmethod
    def _load_object(
        path: Path,
    ) -> dict[str, Any]:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(payload, dict):
            raise TypeError(
                f"{path.name} must contain "
                "a JSON object"
            )

        return payload

    @staticmethod
    def _mapping(
        values: dict[str, Any],
        key: str,
    ) -> dict[str, Any]:
        value = values.get(key)

        if not isinstance(value, dict):
            raise TypeError(
                f"{key} must be an object"
            )

        return value

    @staticmethod
    def _integer(
        values: dict[str, Any],
        key: str,
    ) -> int:
        if key not in values:
            raise ValueError(
                f"missing field: {key}"
            )

        value = values[key]

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                f"{key} must be an integer"
            )

        return value

    @staticmethod
    def _non_empty_string(
        values: dict[str, Any],
        key: str,
    ) -> str:
        value = values.get(key)

        if not isinstance(value, str):
            raise TypeError(
                f"{key} must be a string"
            )

        if not value.strip():
            raise ValueError(
                f"{key} must not be empty"
            )

        return value

    @staticmethod
    def _optional_string(
        values: dict[str, Any],
        key: str,
    ) -> str | None:
        value = values.get(key)

        if value is None:
            return None

        if not isinstance(value, str):
            raise TypeError(
                f"{key} must be a string "
                "or None"
            )

        if not value.strip():
            raise ValueError(
                f"{key} must not be empty"
            )

        return value

    @staticmethod
    def _model_names(
        payload: dict[str, Any],
    ) -> tuple[str, ...]:
        values = payload.get(
            "model_names"
        )

        if not isinstance(values, list):
            raise TypeError(
                "model_names must be a list"
            )

        if not values:
            raise ValueError(
                "model_names must not be empty"
            )

        result: list[str] = []

        for value in values:
            if not isinstance(value, str):
                raise TypeError(
                    "model_names entries "
                    "must be strings"
                )

            if not value.strip():
                raise ValueError(
                    "model_names entries "
                    "must not be empty"
                )

            result.append(value)

        return tuple(result)
