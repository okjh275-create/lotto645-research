from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

from lrp.production.champion_rollback_history import (
    ChampionRollbackHistoryReader,
)
from lrp.production.production_registry_lock import (
    ProductionRegistryWriterLock,
)


@dataclass(frozen=True)
class ChampionRollbackPlan:
    """Validated rollback intent bound to one active state."""

    target_revision_id: str
    target_source_sha256: str
    target_selected_model: str
    active_source_sha256: str


@dataclass(frozen=True)
class ChampionRollbackResult:
    """Result of one explicit production rollback."""

    status: str
    target_revision_id: str
    source_sha256: str
    selected_model: str


class ChampionRollbackService:
    """Plan and execute verified production champion rollback."""

    def __init__(
        self,
        *,
        registry_root: str | Path,
    ) -> None:
        self._registry_root = Path(
            registry_root
        )

        self._reader = (
            ChampionRollbackHistoryReader(
                registry_root=(
                    self._registry_root
                )
            )
        )

    def plan(
        self,
        revision_id: str,
    ) -> ChampionRollbackPlan:
        target = self._reader.resolve(
            revision_id,
            reject_active=True,
        )

        active_source_sha256 = (
            self._read_active_source_sha256()
        )

        return ChampionRollbackPlan(
            target_revision_id=(
                target.revision_id
            ),
            target_source_sha256=(
                target.source_sha256
            ),
            target_selected_model=(
                target.selected_model
            ),
            active_source_sha256=(
                active_source_sha256
            ),
        )

    def execute(
        self,
        plan: ChampionRollbackPlan,
    ) -> ChampionRollbackResult:
        if not isinstance(
            plan,
            ChampionRollbackPlan,
        ):
            raise TypeError(
                "rollback plan is required"
            )

        with ProductionRegistryWriterLock(
            self._registry_root,
        ):
            current_active_sha256 = (
                self._read_active_source_sha256()
            )

            if (
                current_active_sha256
                != plan.active_source_sha256
            ):
                raise ValueError(
                    "stale rollback plan"
                )

            # Re-resolve immediately before any write.
            # This revalidates both publication history
            # and the immutable decision snapshot.
            target = self._reader.resolve(
                plan.target_revision_id,
                reject_active=False,
            )

            if (
                target.source_sha256
                != plan.target_source_sha256
                or target.selected_model
                != plan.target_selected_model
            ):
                raise ValueError(
                    "stale rollback target"
                )

            if (
                target.source_sha256
                == current_active_sha256
            ):
                raise ValueError(
                    "rollback target is already active"
                )

            decision_bytes = (
                target.decision_path
                .read_bytes()
            )

            publication_bytes = (
                target.publication_path
                .read_bytes()
            )

            active_root = (
                self._registry_root
                / "active"
            )

            previous_decision_bytes = (
                active_root
                / "champion_decision.json"
            ).read_bytes()

            previous_publication_bytes = (
                active_root
                / "publication.json"
            ).read_bytes()

            self._replace_active_pair(
                decision_bytes=(
                    decision_bytes
                ),
                publication_bytes=(
                    publication_bytes
                ),
            )

            try:
                self._write_rollback_provenance(
                    from_source_sha256=(
                        plan.active_source_sha256
                    ),
                    to_source_sha256=(
                        target.source_sha256
                    ),
                    target_revision_id=(
                        target.revision_id
                    ),
                    selected_model=(
                        target.selected_model
                    ),
                )
            except Exception:
                self._replace_active_pair(
                    decision_bytes=(
                        previous_decision_bytes
                    ),
                    publication_bytes=(
                        previous_publication_bytes
                    ),
                )
                raise


        return ChampionRollbackResult(
            status="PASS",
            target_revision_id=(
                target.revision_id
            ),
            source_sha256=(
                target.source_sha256
            ),
            selected_model=(
                target.selected_model
            ),
        )

    def _read_active_source_sha256(
        self,
    ) -> str:
        active_root = (
            self._registry_root
            / "active"
        )

        decision_path = (
            active_root
            / "champion_decision.json"
        )

        publication_path = (
            active_root
            / "publication.json"
        )

        if not decision_path.is_file():
            raise ValueError(
                "active champion decision not found"
            )

        if not publication_path.is_file():
            raise ValueError(
                "active publication not found"
            )

        try:
            publication = json.loads(
                publication_path.read_text(
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
            publication,
            dict,
        ):
            raise ValueError(
                "invalid active publication"
            )

        source_sha256 = (
            publication.get(
                "source_sha256"
            )
        )

        if (
            not isinstance(
                source_sha256,
                str,
            )
            or len(source_sha256) != 64
            or any(
                char not in
                "0123456789abcdef"
                for char
                in source_sha256.lower()
            )
        ):
            raise ValueError(
                "invalid active source sha256"
            )

        source_sha256 = (
            source_sha256.lower()
        )

        import hashlib

        actual_sha256 = hashlib.sha256(
            decision_path.read_bytes()
        ).hexdigest()

        if (
            actual_sha256
            != source_sha256
        ):
            raise ValueError(
                "active decision sha256 mismatch"
            )

        return source_sha256

    def _replace_active_pair(
        self,
        *,
        decision_bytes: bytes,
        publication_bytes: bytes,
    ) -> None:
        active_root = (
            self._registry_root
            / "active"
        )

        decision_path = (
            active_root
            / "champion_decision.json"
        )

        publication_path = (
            active_root
            / "publication.json"
        )

        if not active_root.is_dir():
            raise ValueError(
                "active registry not found"
            )

        old_decision = (
            decision_path.read_bytes()
        )

        old_publication = (
            publication_path.read_bytes()
        )

        decision_temp = (
            self._prepare_temp_file(
                active_root,
                decision_bytes,
            )
        )

        publication_temp = (
            self._prepare_temp_file(
                active_root,
                publication_bytes,
            )
        )

        decision_replaced = False

        try:
            os.replace(
                decision_temp,
                decision_path,
            )

            decision_replaced = True

            os.replace(
                publication_temp,
                publication_path,
            )

        except Exception:
            # Best-effort compensation if the first
            # atomic replacement succeeded but the
            # second replacement failed.
            if decision_replaced:
                self._atomic_write_bytes(
                    decision_path,
                    old_decision,
                )

            self._atomic_write_bytes(
                publication_path,
                old_publication,
            )

            raise

        finally:
            for temp_path in (
                decision_temp,
                publication_temp,
            ):
                try:
                    temp_path.unlink(
                        missing_ok=True
                    )
                except OSError:
                    pass

    @staticmethod
    def _prepare_temp_file(
        directory: Path,
        payload: bytes,
    ) -> Path:
        handle = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=directory,
            prefix=".rollback-",
            suffix=".tmp",
            delete=False,
        )

        path = Path(
            handle.name
        )

        try:
            with handle:
                handle.write(
                    payload
                )
                handle.flush()
                os.fsync(
                    handle.fileno()
                )

        except Exception:
            path.unlink(
                missing_ok=True
            )
            raise

        return path

    @classmethod
    def _atomic_write_bytes(
        cls,
        path: Path,
        payload: bytes,
    ) -> None:
        temp_path = (
            cls._prepare_temp_file(
                path.parent,
                payload,
            )
        )

        try:
            os.replace(
                temp_path,
                path,
            )
        finally:
            temp_path.unlink(
                missing_ok=True
            )

    def _write_rollback_provenance(
        self,
        *,
        from_source_sha256: str,
        to_source_sha256: str,
        target_revision_id: str,
        selected_model: str,
    ) -> Path:
        from datetime import datetime, timezone
        import hashlib

        executed_at = (
            datetime.now(
                timezone.utc
            )
            .astimezone()
            .isoformat()
        )

        payload = {
            "executed_at": executed_at,
            "from_source_sha256": from_source_sha256,
            "to_source_sha256": to_source_sha256,
            "target_revision_id": target_revision_id,
            "selected_model": selected_model,
        }

        record_bytes = (
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        ).encode(
            "utf-8"
        )

        record_sha256 = (
            hashlib.sha256(
                record_bytes
            ).hexdigest()
        )

        rollback_root = (
            self._registry_root
            / "history"
            / "rollbacks"
        )

        rollback_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        record_path = (
            rollback_root
            / f"{record_sha256}.json"
        )

        if record_path.exists():
            if (
                record_path.read_bytes()
                != record_bytes
            ):
                raise RuntimeError(
                    "rollback provenance hash collision"
                )

            return record_path

        with record_path.open(
            "xb"
        ) as handle:
            handle.write(
                record_bytes
            )

        return record_path
