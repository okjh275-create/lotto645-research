from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChampionRollbackTarget:
    """Verified immutable production rollback target."""

    revision_id: str
    source_sha256: str
    selected_model: str
    published_at_kst: str
    publication_path: Path
    decision_path: Path
    publication: dict[str, Any]


class ChampionRollbackHistoryReader:
    """Read-only resolver for production champion history."""

    def __init__(
        self,
        *,
        registry_root: str | Path,
    ) -> None:
        self._registry_root = Path(
            registry_root
        )

    @property
    def registry_root(
        self,
    ) -> Path:
        return self._registry_root

    @property
    def history_root(
        self,
    ) -> Path:
        return (
            self._registry_root
            / "history"
        )

    @property
    def decision_history_root(
        self,
    ) -> Path:
        return (
            self.history_root
            / "decisions"
        )

    def list_revisions(
        self,
    ) -> tuple[
        ChampionRollbackTarget,
        ...,
    ]:
        history_root = (
            self.history_root
        )

        if not history_root.exists():
            return ()

        targets: list[
            ChampionRollbackTarget
        ] = []

        for path in history_root.glob(
            "*.json"
        ):
            if not path.is_file():
                continue

            target = (
                self._resolve_path(
                    path
                )
            )

            targets.append(
                target
            )

        targets.sort(
            key=lambda item: (
                item.published_at_kst,
                item.revision_id,
            )
        )

        return tuple(
            targets
        )

    def resolve(
        self,
        revision_id: str,
        *,
        reject_active: bool = False,
    ) -> ChampionRollbackTarget:
        normalized = str(
            revision_id
        ).strip().lower()

        if (
            len(normalized) != 64
            or any(
                char not in "0123456789abcdef"
                for char in normalized
            )
        ):
            raise ValueError(
                "invalid revision id"
            )

        path = (
            self.history_root
            / f"{normalized}.json"
        )

        if not path.is_file():
            raise ValueError(
                "rollback revision not found"
            )

        target = (
            self._resolve_path(
                path
            )
        )

        if (
            reject_active
            and self._is_active_target(
                target
            )
        ):
            raise ValueError(
                "rollback target is already active"
            )

        return target

    def _resolve_path(
        self,
        path: Path,
    ) -> ChampionRollbackTarget:
        publication_bytes = (
            path.read_bytes()
        )

        actual_revision_id = (
            hashlib.sha256(
                publication_bytes
            ).hexdigest()
        )

        expected_revision_id = (
            path.stem.lower()
        )

        if (
            actual_revision_id
            != expected_revision_id
        ):
            raise ValueError(
                "publication revision sha256 mismatch"
            )

        try:
            payload = json.loads(
                publication_bytes
                .decode(
                    "utf-8"
                )
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError(
                "invalid publication revision"
            ) from exc

        if not isinstance(
            payload,
            dict,
        ):
            raise ValueError(
                "invalid publication revision"
            )

        source_sha256 = (
            payload.get(
                "source_sha256"
            )
        )

        selected_model = (
            payload.get(
                "selected_model"
            )
        )

        published_at_kst = (
            payload.get(
                "published_at_kst"
            )
        )

        if (
            not isinstance(
                source_sha256,
                str,
            )
            or len(
                source_sha256
            ) != 64
            or any(
                char not in
                "0123456789abcdef"
                for char
                in source_sha256.lower()
            )
        ):
            raise ValueError(
                "invalid decision source sha256"
            )

        source_sha256 = (
            source_sha256.lower()
        )

        if (
            not isinstance(
                selected_model,
                str,
            )
            or not selected_model
        ):
            raise ValueError(
                "invalid selected model"
            )

        if (
            not isinstance(
                published_at_kst,
                str,
            )
            or not published_at_kst
        ):
            raise ValueError(
                "invalid publication timestamp"
            )

        decision_path = (
            self.decision_history_root
            / f"{source_sha256}.json"
        )

        if not decision_path.is_file():
            raise ValueError(
                "decision snapshot not found"
            )

        decision_bytes = (
            decision_path
            .read_bytes()
        )

        actual_source_sha256 = (
            hashlib.sha256(
                decision_bytes
            ).hexdigest()
        )

        if (
            actual_source_sha256
            != source_sha256
        ):
            raise ValueError(
                "decision snapshot sha256 mismatch"
            )

        try:
            decision_payload = (
                json.loads(
                    decision_bytes
                    .decode(
                        "utf-8"
                    )
                )
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError(
                "invalid decision snapshot"
            ) from exc

        if not isinstance(
            decision_payload,
            dict,
        ):
            raise ValueError(
                "invalid decision snapshot"
            )

        selection = (
            decision_payload.get(
                "selection"
            )
        )

        if not isinstance(
            selection,
            dict,
        ):
            raise ValueError(
                "invalid decision snapshot"
            )

        decision_model = (
            selection.get(
                "selected_model"
            )
        )

        if (
            decision_model
            != selected_model
        ):
            raise ValueError(
                "decision snapshot model mismatch"
            )

        return ChampionRollbackTarget(
            revision_id=(
                actual_revision_id
            ),
            source_sha256=(
                source_sha256
            ),
            selected_model=(
                selected_model
            ),
            published_at_kst=(
                published_at_kst
            ),
            publication_path=path,
            decision_path=(
                decision_path
            ),
            publication=dict(
                payload
            ),
        )

    def _is_active_target(
        self,
        target: ChampionRollbackTarget,
    ) -> bool:
        active_publication = (
            self._registry_root
            / "active"
            / "publication.json"
        )

        if not active_publication.is_file():
            return False

        try:
            payload = json.loads(
                active_publication.read_text(
                    encoding="utf-8"
                )
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError(
                "invalid active publication"
            ) from exc

        if not isinstance(
            payload,
            dict,
        ):
            raise ValueError(
                "invalid active publication"
            )

        active_sha256 = (
            payload.get(
                "source_sha256"
            )
        )

        if not isinstance(
            active_sha256,
            str,
        ):
            raise ValueError(
                "invalid active publication"
            )

        return (
            active_sha256.lower()
            == target.source_sha256
        )
