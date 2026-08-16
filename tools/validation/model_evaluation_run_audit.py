"""Audit model-evaluation provenance against champion decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelEvaluationRunAuditResult:
    run_id: str
    status: str
    issues: tuple[str, ...]
    evaluation_run: Path
    champion_artifact: Path

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "issues": list(self.issues),
            "evaluation_run": str(
                self.evaluation_run
            ),
            "champion_artifact": (
                self.champion_artifact.as_posix()
            ),
        }


class ModelEvaluationRunAudit:
    """Audit one evaluation_run.json against its champion artifact."""

    def audit(
        self,
        evaluation_run: str | Path,
    ) -> ModelEvaluationRunAuditResult:
        run_path = Path(
            evaluation_run
        )

        if not run_path.exists():
            raise FileNotFoundError(
                run_path
            )

        if run_path.is_dir():
            raise IsADirectoryError(
                run_path
            )

        run_payload = self._load_object(
            run_path
        )

        run_id = self._non_empty_string(
            run_payload,
            "run_id",
        )

        champion = self._mapping(
            run_payload,
            "champion",
        )

        artifact_value = (
            self._non_empty_string(
                run_payload,
                "champion_artifact",
            )
        )

        artifact = Path(
            artifact_value
        )

        artifact_path = (
            self._resolve_artifact(
                run_path=run_path,
                artifact=artifact,
            )
        )

        if artifact_path is None:
            return ModelEvaluationRunAuditResult(
                run_id=run_id,
                status="INCOMPLETE",
                issues=(
                    "champion_artifact_missing",
                ),
                evaluation_run=run_path,
                champion_artifact=artifact,
            )

        decision_payload = self._load_object(
            artifact_path
        )

        selection = self._mapping(
            decision_payload,
            "selection",
        )

        promotion = self._mapping(
            selection,
            "promotion",
        )

        issues: list[str] = []

        if (
            champion.get(
                "ranking_champion"
            )
            != selection.get(
                "ranking_champion"
            )
        ):
            issues.append(
                "ranking_champion_mismatch"
            )

        if (
            champion.get(
                "selected_model"
            )
            != selection.get(
                "selected_model"
            )
        ):
            issues.append(
                "selected_model_mismatch"
            )

        if (
            champion.get(
                "promoted"
            )
            != promotion.get(
                "promoted"
            )
        ):
            issues.append(
                "promoted_mismatch"
            )

        return ModelEvaluationRunAuditResult(
            run_id=run_id,
            status=(
                "PASS"
                if not issues
                else "FAIL"
            ),
            issues=tuple(issues),
            evaluation_run=run_path,
            champion_artifact=artifact,
        )

    @staticmethod
    def _resolve_artifact(
        *,
        run_path: Path,
        artifact: Path,
    ) -> Path | None:
        if artifact.is_absolute():
            candidates = (
                artifact,
            )
        else:
            candidates = (
                artifact,
                run_path.parent / artifact,
            )

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        return None

    @staticmethod
    def _load_object(
        path: Path,
    ) -> dict[str, Any]:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            payload,
            dict,
        ):
            raise TypeError(
                f"{path.name} must contain "
                "a JSON object"
            )

        return payload

    @staticmethod
    def _mapping(
        payload: dict[str, Any],
        key: str,
    ) -> dict[str, Any]:
        value = payload.get(
            key
        )

        if not isinstance(
            value,
            dict,
        ):
            raise TypeError(
                f"{key} must be an object"
            )

        return value

    @staticmethod
    def _non_empty_string(
        payload: dict[str, Any],
        key: str,
    ) -> str:
        value = payload.get(
            key
        )

        if not isinstance(
            value,
            str,
        ):
            raise TypeError(
                f"{key} must be a string"
            )

        if not value.strip():
            raise ValueError(
                f"{key} must not be empty"
            )

        return value
