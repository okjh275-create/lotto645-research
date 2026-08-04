"""Persist adaptive-automation results as deterministic JSON."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lrp.evolution.feedback.automation import (
    AdaptiveAutomationResult,
)


@dataclass(frozen=True, slots=True)
class AdaptiveAutomationSaveResult:
    """Paths written by the automation repository."""

    automation_path: Path
    profile_path: Path | None
    automation_created: bool
    profile_created: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "automation_path": str(
                self.automation_path
            ),
            "profile_path": (
                str(self.profile_path)
                if self.profile_path is not None
                else None
            ),
            "automation_created": (
                self.automation_created
            ),
            "profile_created": (
                self.profile_created
            ),
        }


class AdaptiveAutomationRepository:
    """Store automation decisions and approved profile plans."""

    _ID_PATTERN = re.compile(
        r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
    )

    def __init__(
        self,
        root: Path,
    ) -> None:
        self._root = Path(root)

        if (
            self._root.exists()
            and not self._root.is_dir()
        ):
            raise NotADirectoryError(
                self._root
            )

    @property
    def root(self) -> Path:
        return self._root

    @property
    def automation_root(self) -> Path:
        return self.root / "automation"

    @property
    def profile_root(self) -> Path:
        return self.root / "profiles"

    def save(
        self,
        result: AdaptiveAutomationResult,
    ) -> AdaptiveAutomationSaveResult:
        if not isinstance(
            result,
            AdaptiveAutomationResult,
        ):
            raise TypeError(
                "result must be an "
                "AdaptiveAutomationResult"
            )

        recommendation_id = self._validate_id(
            result.recommendation
            .recommendation_id
        )

        automation_payload = (
            result.as_dict()
        )
        automation_path = (
            self.automation_root
            / f"{recommendation_id}.json"
        )

        automation_created = (
            self._write_deterministic(
                path=automation_path,
                payload=automation_payload,
                collision_name=(
                    "automation recommendation"
                ),
            )
        )

        profile_path: Path | None = None
        profile_created = False

        if result.update_plan.approved:
            revision = (
                result.update_plan
                .target_revision
            )

            if revision < 0:
                raise ValueError(
                    "target revision must be "
                    "greater than or equal to 0"
                )

            profile_path = (
                self.profile_root
                / (
                    "revision-"
                    f"{revision:08d}.json"
                )
            )

            profile_payload = {
                "schema_version": 1,
                "recommendation_id": (
                    recommendation_id
                ),
                "source_revision": (
                    result.update_plan
                    .source_revision
                ),
                "target_revision": (
                    revision
                ),
                "profile": (
                    result.update_plan
                    .as_dict()["profile"]
                ),
            }

            profile_created = (
                self._write_deterministic(
                    path=profile_path,
                    payload=profile_payload,
                    collision_name=(
                        "adaptive profile revision"
                    ),
                )
            )

        return AdaptiveAutomationSaveResult(
            automation_path=automation_path,
            profile_path=profile_path,
            automation_created=(
                automation_created
            ),
            profile_created=profile_created,
        )

    def load_automation(
        self,
        recommendation_id: str,
    ) -> dict[str, Any]:
        normalized = self._validate_id(
            recommendation_id
        )
        path = (
            self.automation_root
            / f"{normalized}.json"
        )

        return self._load_object(path)

    def load_profile_revision(
        self,
        revision: int,
    ) -> dict[str, Any]:
        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
        ):
            raise TypeError(
                "revision must be an integer"
            )

        if revision < 0:
            raise ValueError(
                "revision must be greater "
                "than or equal to 0"
            )

        path = (
            self.profile_root
            / f"revision-{revision:08d}.json"
        )

        return self._load_object(path)

    def list_automation_ids(
        self,
    ) -> tuple[str, ...]:
        if not self.automation_root.is_dir():
            return ()

        return tuple(
            path.stem
            for path in sorted(
                self.automation_root.glob(
                    "*.json"
                )
            )
        )

    def list_profile_revisions(
        self,
    ) -> tuple[int, ...]:
        if not self.profile_root.is_dir():
            return ()

        revisions: list[int] = []

        for path in sorted(
            self.profile_root.glob(
                "revision-*.json"
            )
        ):
            suffix = path.stem.removeprefix(
                "revision-"
            )

            if suffix.isdigit():
                revisions.append(
                    int(suffix)
                )

        return tuple(revisions)

    def latest_profile(
        self,
    ) -> dict[str, Any] | None:
        revisions = (
            self.list_profile_revisions()
        )

        if not revisions:
            return None

        return self.load_profile_revision(
            revisions[-1]
        )

    def _write_deterministic(
        self,
        *,
        path: Path,
        payload: Mapping[str, Any],
        collision_name: str,
    ) -> bool:
        serialized = self._serialize(
            payload
        )

        if path.exists():
            existing = path.read_bytes()

            if existing == serialized:
                return False

            raise FileExistsError(
                f"{collision_name} collision: "
                f"{path}"
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = path.with_suffix(
            path.suffix + ".tmp"
        )

        if temporary.exists():
            temporary.unlink()

        temporary.write_bytes(
            serialized
        )
        temporary.replace(path)

        return True

    @staticmethod
    def _serialize(
        payload: Mapping[str, Any],
    ) -> bytes:
        if not isinstance(
            payload,
            Mapping,
        ):
            raise TypeError(
                "payload must be a mapping"
            )

        text = (
            json.dumps(
                dict(payload),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )

        return text.encode("utf-8")

    @staticmethod
    def _load_object(
        path: Path,
    ) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(path)

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

    @classmethod
    def _validate_id(
        cls,
        value: object,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(
                "recommendation_id must be "
                "a string"
            )

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "recommendation_id must not "
                "be empty"
            )

        if not cls._ID_PATTERN.fullmatch(
            normalized
        ):
            raise ValueError(
                "recommendation_id contains "
                "unsupported characters"
            )

        return normalized
