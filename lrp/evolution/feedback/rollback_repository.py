"""Persist approved adaptive rollback plans."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lrp.evolution.feedback.repository import (
    AdaptiveAutomationRepository,
)
from lrp.evolution.feedback.rollback import (
    AdaptiveRollbackPlan,
)


@dataclass(frozen=True, slots=True)
class AdaptiveRollbackSaveResult:
    """Result of persisting a rollback plan."""

    path: Path
    created: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "created": self.created,
        }


class AdaptiveRollbackRepository:
    """Persist rollback plans as new profile revisions."""

    def __init__(
        self,
        *,
        repository: AdaptiveAutomationRepository,
    ) -> None:
        if not isinstance(
            repository,
            AdaptiveAutomationRepository,
        ):
            raise TypeError(
                "repository must be an "
                "AdaptiveAutomationRepository"
            )

        self._repository = repository

    @property
    def repository(
        self,
    ) -> AdaptiveAutomationRepository:
        return self._repository

    def save(
        self,
        plan: AdaptiveRollbackPlan,
        *,
        rollback_id: str,
    ) -> AdaptiveRollbackSaveResult:
        if not isinstance(
            plan,
            AdaptiveRollbackPlan,
        ):
            raise TypeError(
                "plan must be an AdaptiveRollbackPlan"
            )

        normalized_id = self._validate_id(
            rollback_id
        )

        latest = self.repository.latest_profile()

        if latest is None:
            raise RuntimeError(
                "adaptive profile repository is empty"
            )

        repository_revision = latest.get(
            "target_revision"
        )

        if (
            isinstance(repository_revision, bool)
            or not isinstance(
                repository_revision,
                int,
            )
        ):
            raise TypeError(
                "repository target_revision "
                "must be an integer"
            )

        if (
            repository_revision
            != plan.source_revision
        ):
            raise RuntimeError(
                "rollback source revision does not "
                "match repository head"
            )

        path = (
            self.repository.profile_root
            / (
                "revision-"
                f"{plan.target_revision:08d}.json"
            )
        )

        payload = {
            "schema_version": 1,
            "record_type": "rollback",
            "rollback_id": normalized_id,
            "source_revision": (
                plan.source_revision
            ),
            "rollback_revision": (
                plan.rollback_revision
            ),
            "target_revision": (
                plan.target_revision
            ),
            "changed_component_count": (
                plan.changed_component_count
            ),
            "differences": [
                item.as_dict()
                for item in plan.differences
            ],
            "profile": (
                plan.as_dict()["profile"]
            ),
        }

        serialized = (
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

        if path.exists():
            if path.read_bytes() == serialized:
                return AdaptiveRollbackSaveResult(
                    path=path,
                    created=False,
                )

            raise FileExistsError(
                "rollback profile revision collision: "
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

        temporary.write_bytes(serialized)
        temporary.replace(path)

        return AdaptiveRollbackSaveResult(
            path=path,
            created=True,
        )

    @staticmethod
    def _validate_id(
        value: object,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(
                "rollback_id must be a string"
            )

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "rollback_id must not be empty"
            )

        if any(
            character not in (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789._-"
            )
            for character in normalized
        ):
            raise ValueError(
                "rollback_id contains "
                "unsupported characters"
            )

        return normalized
