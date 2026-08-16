"""Operational provenance record for historical model evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from lrp.evaluation import EvaluationWindow

from tools.validation.historical_replay_models import (
    ReplayConfig,
)


@dataclass(frozen=True)
class ModelEvaluationRunRecord:
    run_id: str
    history_path: Path
    model_names: tuple[str, ...]
    windows: tuple[EvaluationWindow, ...]
    replay_config: ReplayConfig
    ranking_champion: str | None
    selected_model: str | None
    promoted: bool
    champion_artifact: Path

    @classmethod
    def build(
        cls,
        *,
        history_path: str | Path,
        model_names: Iterable[str],
        windows: Iterable[EvaluationWindow],
        replay_config: ReplayConfig,
        ranking_champion: str | None,
        selected_model: str | None,
        promoted: bool,
        champion_artifact: str | Path,
    ) -> "ModelEvaluationRunRecord":
        normalized_models = tuple(
            model_names
        )

        if not normalized_models:
            raise ValueError(
                "model_names must not be empty"
            )

        if any(
            not isinstance(name, str)
            or not name.strip()
            for name in normalized_models
        ):
            raise ValueError(
                "model_names must contain "
                "non-empty strings"
            )

        normalized_windows = tuple(
            windows
        )

        if not normalized_windows:
            raise ValueError(
                "windows must not be empty"
            )

        if any(
            not isinstance(
                window,
                EvaluationWindow,
            )
            for window in normalized_windows
        ):
            raise TypeError(
                "windows must contain "
                "EvaluationWindow values"
            )

        if not isinstance(
            replay_config,
            ReplayConfig,
        ):
            raise TypeError(
                "replay_config must be ReplayConfig"
            )

        history = Path(
            history_path
        )

        artifact = Path(
            champion_artifact
        )

        payload = cls._canonical_payload(
            history_path=history,
            model_names=normalized_models,
            windows=normalized_windows,
            replay_config=replay_config,
            ranking_champion=ranking_champion,
            selected_model=selected_model,
            promoted=promoted,
            champion_artifact=artifact,
        )

        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        ).encode(
            "utf-8"
        )

        run_id = hashlib.sha256(
            encoded
        ).hexdigest()[:16]

        return cls(
            run_id=run_id,
            history_path=history,
            model_names=normalized_models,
            windows=normalized_windows,
            replay_config=replay_config,
            ranking_champion=ranking_champion,
            selected_model=selected_model,
            promoted=bool(promoted),
            champion_artifact=artifact,
        )

    def as_dict(self) -> dict[str, object]:
        payload = self._canonical_payload(
            history_path=self.history_path,
            model_names=self.model_names,
            windows=self.windows,
            replay_config=self.replay_config,
            ranking_champion=(
                self.ranking_champion
            ),
            selected_model=(
                self.selected_model
            ),
            promoted=self.promoted,
            champion_artifact=(
                self.champion_artifact
            ),
        )

        return {
            "run_id": self.run_id,
            **payload,
        }

    @staticmethod
    def _canonical_payload(
        *,
        history_path: Path,
        model_names: tuple[str, ...],
        windows: tuple[
            EvaluationWindow,
            ...,
        ],
        replay_config: ReplayConfig,
        ranking_champion: str | None,
        selected_model: str | None,
        promoted: bool,
        champion_artifact: Path,
    ) -> dict[str, object]:
        return {
            "history_path": history_path.as_posix(),
            "model_names": list(
                model_names
            ),
            "round_range": {
                "start_round": (
                    replay_config.start_round
                ),
                "end_round": (
                    replay_config.end_round
                ),
            },
            "windows": [
                {
                    "name": window.name,
                    "start_round": (
                        window.start_round
                    ),
                    "end_round": (
                        window.end_round
                    ),
                    "round_count": (
                        window.round_count
                    ),
                }
                for window in windows
            ],
            "replay_config": {
                "seed_base": (
                    replay_config.seed_base
                ),
                "temperature": (
                    replay_config.temperature
                ),
                "candidate_count": (
                    replay_config.candidate_count
                ),
                "top_k": (
                    replay_config.top_k
                ),
                "practical_k": (
                    replay_config.practical_k
                ),
                "long_gap_window": (
                    replay_config.long_gap_window
                ),
                "confidence": (
                    replay_config.confidence
                ),
                "mode": (
                    replay_config.mode
                ),
            },
            "champion": {
                "ranking_champion": (
                    ranking_champion
                ),
                "selected_model": (
                    selected_model
                ),
                "promoted": bool(
                    promoted
                ),
            },
            "champion_artifact": champion_artifact.as_posix(),
        }
