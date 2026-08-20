"""Production champion registry backup contracts and service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

from lrp.production.production_registry_lock import (
    ProductionRegistryWriterLock,
)


_SHA256_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)

_BACKUP_FILE_ROLES = frozenset(
    {
        "active_decision",
        "active_publication",
        "history_revision",
        "history_decision",
        "rollback_provenance",
    }
)


def _validate_sha256(
    value: str,
    *,
    field: str,
) -> None:
    if (
        not isinstance(value, str)
        or _SHA256_PATTERN.fullmatch(
            value
        )
        is None
    ):
        raise ValueError(
            f"{field} must be a lowercase sha256 hex digest"
        )


def _canonical_json_bytes(
    payload: object,
) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(
    frozen=True,
    slots=True,
)
class ProductionRegistryBackupFile:
    relative_path: str
    size: int
    sha256: str
    role: str

    def __post_init__(
        self,
    ) -> None:
        if (
            not isinstance(
                self.relative_path,
                str,
            )
            or not self.relative_path
        ):
            raise ValueError(
                "relative_path must not be empty"
            )

        if "\\" in self.relative_path:
            raise ValueError(
                "relative_path must use POSIX separators"
            )

        relative = PurePosixPath(
            self.relative_path
        )

        if (
            relative.is_absolute()
            or self.relative_path.startswith(
                "/"
            )
            or any(
                part in {
                    "",
                    ".",
                    "..",
                }
                for part in relative.parts
            )
        ):
            raise ValueError(
                "relative_path must be a safe relative path"
            )

        if (
            isinstance(
                self.size,
                bool,
            )
            or not isinstance(
                self.size,
                int,
            )
            or self.size < 0
        ):
            raise ValueError(
                "size must be a non-negative integer"
            )

        _validate_sha256(
            self.sha256,
            field="sha256",
        )

        if (
            not isinstance(
                self.role,
                str,
            )
            or self.role
            not in _BACKUP_FILE_ROLES
        ):
            raise ValueError(
                "role is not supported"
            )

    def as_dict(
        self,
    ) -> dict[str, object]:
        return {
            "relative_path":
                self.relative_path,

            "size":
                self.size,

            "sha256":
                self.sha256,

            "role":
                self.role,
        }


@dataclass(
    frozen=True,
    slots=True,
)
class ProductionRegistryBackupManifest:
    schema_version: int
    created_at: str
    source_registry_root: str
    active_model: str | None
    active_source_sha256: str
    active_revision_id: str
    files: tuple[
        ProductionRegistryBackupFile,
        ...,
    ]

    def __post_init__(
        self,
    ) -> None:
        if self.schema_version != 1:
            raise ValueError(
                "schema_version must be 1"
            )

        if (
            not isinstance(
                self.created_at,
                str,
            )
            or not self.created_at
        ):
            raise ValueError(
                "created_at must not be empty"
            )

        if (
            not isinstance(
                self.source_registry_root,
                str,
            )
            or not self.source_registry_root
        ):
            raise ValueError(
                "source_registry_root must not be empty"
            )

        if (
            self.active_model is not None
            and (
                not isinstance(
                    self.active_model,
                    str,
                )
                or not self.active_model
            )
        ):
            raise ValueError(
                "active_model must be a string or None"
            )

        _validate_sha256(
            self.active_source_sha256,
            field="active_source_sha256",
        )

        _validate_sha256(
            self.active_revision_id,
            field="active_revision_id",
        )

        if not isinstance(
            self.files,
            tuple,
        ):
            object.__setattr__(
                self,
                "files",
                tuple(
                    self.files
                ),
            )

        if not all(
            isinstance(
                item,
                ProductionRegistryBackupFile,
            )
            for item in self.files
        ):
            raise TypeError(
                "files must contain "
                "ProductionRegistryBackupFile records"
            )

        relative_paths = [
            item.relative_path
            for item in self.files
        ]

        if (
            relative_paths
            != sorted(
                relative_paths
            )
            or len(
                set(
                    relative_paths
                )
            )
            != len(
                relative_paths
            )
        ):
            raise ValueError(
                "files must be sorted and unique"
            )

    def as_dict(
        self,
    ) -> dict[str, object]:
        return {
            "schema_version":
                self.schema_version,

            "created_at":
                self.created_at,

            "source_registry_root":
                self.source_registry_root,

            "active_model":
                self.active_model,

            "active_source_sha256":
                self.active_source_sha256,

            "active_revision_id":
                self.active_revision_id,

            "files":
                [
                    item.as_dict()
                    for item in self.files
                ],
        }

    @property
    def backup_id(
        self,
    ) -> str:
        return hashlib.sha256(
            _canonical_json_bytes(
                self.as_dict()
            )
        ).hexdigest()


@dataclass(
    frozen=True,
    slots=True,
)
class ProductionRegistryBackupResult:
    backup_root: Path
    backup_id: str
    manifest_path: Path
    file_count: int

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "backup_root",
            Path(
                self.backup_root
            ),
        )

        object.__setattr__(
            self,
            "manifest_path",
            Path(
                self.manifest_path
            ),
        )

        _validate_sha256(
            self.backup_id,
            field="backup_id",
        )

        if (
            isinstance(
                self.file_count,
                bool,
            )
            or not isinstance(
                self.file_count,
                int,
            )
            or self.file_count < 0
        ):
            raise ValueError(
                "file_count must be a non-negative integer"
            )


class ProductionRegistryBackupService:
    """Create a coherent read-only backup of a production registry."""

    def __init__(
        self,
        registry_root: str | Path,
    ) -> None:
        self._registry_root = Path(
            registry_root
        )

    def backup(
        self,
        destination_root: str | Path,
    ) -> ProductionRegistryBackupResult:
        destination = Path(
            destination_root
        )

        with ProductionRegistryWriterLock(
            self._registry_root
        ):
            (
                captured,
                manifest,
            ) = self._capture_locked()

        backup_id = manifest.backup_id

        backup_root = (
            destination
            / backup_id
        )

        manifest_path = (
            backup_root
            / "manifest.json"
        )

        complete_path = (
            backup_root
            / "COMPLETE"
        )

        if backup_root.exists():
            raise FileExistsError(
                f"backup already exists: {backup_root}"
            )

        backup_root.mkdir(
            parents=True,
            exist_ok=False,
        )

        payload_root = (
            backup_root
            / "payload"
        )

        try:
            for (
                relative_path,
                data,
            ) in captured:
                target = (
                    payload_root
                    / Path(
                        *PurePosixPath(
                            relative_path
                        ).parts
                    )
                )

                target.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                target.write_bytes(
                    data
                )

            manifest_bytes = (
                json.dumps(
                    manifest.as_dict(),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")

            manifest_path.write_bytes(
                manifest_bytes
            )

            # COMPLETE is deliberately published last.
            complete_path.write_text(
                backup_id + "\n",
                encoding="utf-8",
            )

        except BaseException:
            # Partial material may remain for diagnosis, but it
            # must never be advertised as a complete backup.
            try:
                if complete_path.exists():
                    complete_path.unlink()
            except OSError:
                pass

            raise

        return ProductionRegistryBackupResult(
            backup_root=backup_root,
            backup_id=backup_id,
            manifest_path=manifest_path,
            file_count=len(
                manifest.files
            ),
        )

    def _capture_locked(
        self,
    ) -> tuple[
        tuple[
            tuple[str, bytes],
            ...,
        ],
        ProductionRegistryBackupManifest,
    ]:
        registry = (
            self._registry_root
        )

        if not registry.exists():
            raise FileNotFoundError(
                registry
            )

        if not registry.is_dir():
            raise NotADirectoryError(
                registry
            )

        files: list[
            tuple[
                str,
                bytes,
                str,
            ]
        ] = []

        for path in sorted(
            registry.rglob("*")
        ):
            if not path.is_file():
                continue

            relative = (
                path.relative_to(
                    registry
                )
                .as_posix()
            )

            if relative == ".writer.lock":
                continue

            role = self._classify_path(
                relative
            )

            data = path.read_bytes()

            files.append(
                (
                    relative,
                    data,
                    role,
                )
            )

        captured_by_path = {
            relative: data
            for (
                relative,
                data,
                _,
            ) in files
        }

        decision_relative = (
            "active/champion_decision.json"
        )

        publication_relative = (
            "active/publication.json"
        )

        decision_present = (
            decision_relative
            in captured_by_path
        )

        publication_present = (
            publication_relative
            in captured_by_path
        )

        if (
            decision_present
            != publication_present
        ):
            raise ValueError(
                "active pair is partial"
            )

        if not (
            decision_present
            and publication_present
        ):
            raise ValueError(
                "active pair is missing"
            )

        decision_bytes = (
            captured_by_path[
                decision_relative
            ]
        )

        publication_bytes = (
            captured_by_path[
                publication_relative
            ]
        )

        try:
            decision_payload = json.loads(
                decision_bytes.decode(
                    "utf-8-sig"
                )
            )

        except Exception as exc:
            raise ValueError(
                "active decision is invalid"
            ) from exc

        try:
            publication_payload = json.loads(
                publication_bytes.decode(
                    "utf-8-sig"
                )
            )

        except Exception as exc:
            raise ValueError(
                "active publication is invalid"
            ) from exc

        if not isinstance(
            decision_payload,
            dict,
        ):
            raise ValueError(
                "active decision must be an object"
            )

        if not isinstance(
            publication_payload,
            dict,
        ):
            raise ValueError(
                "active publication must be an object"
            )

        active_model = (
            self._selected_model(
                decision_payload
            )
        )

        publication_model = (
            publication_payload.get(
                "selected_model"
            )
        )

        if (
            active_model
            != publication_model
        ):
            raise ValueError(
                "active pair selected_model mismatch"
            )

        decision_sha = (
            hashlib.sha256(
                decision_bytes
            )
            .hexdigest()
        )

        publication_source_sha = (
            publication_payload.get(
                "source_sha256"
            )
        )

        if (
            publication_source_sha
            != decision_sha
        ):
            raise ValueError(
                "active pair source_sha256 mismatch"
            )

        revision_id = (
            publication_payload.get(
                "revision_id"
            )
        )

        if not isinstance(
            revision_id,
            str,
        ):
            raise ValueError(
                "active publication revision_id is invalid"
            )

        _validate_sha256(
            revision_id,
            field="active_revision_id",
        )

        records = tuple(
            ProductionRegistryBackupFile(
                relative_path=relative,
                size=len(
                    data
                ),
                sha256=(
                    hashlib.sha256(
                        data
                    )
                    .hexdigest()
                ),
                role=role,
            )
            for (
                relative,
                data,
                role,
            ) in files
        )

        # rglob() ordering is filesystem-specific; the manifest
        # contract is explicitly path-sorted.
        records = tuple(
            sorted(
                records,
                key=lambda item:
                    item.relative_path,
            )
        )

        captured = tuple(
            (
                relative,
                data,
            )
            for (
                relative,
                data,
                _,
            ) in sorted(
                files,
                key=lambda item:
                    item[0],
            )
        )

        manifest = (
            ProductionRegistryBackupManifest(
                schema_version=1,
                created_at=(
                    datetime.now(
                        timezone.utc
                    )
                    .isoformat()
                ),
                source_registry_root=str(
                    registry.resolve()
                ),
                active_model=active_model,
                active_source_sha256=(
                    decision_sha
                ),
                active_revision_id=(
                    revision_id
                ),
                files=records,
            )
        )

        return (
            captured,
            manifest,
        )

    @staticmethod
    def _selected_model(
        payload: dict[str, Any],
    ) -> str | None:
        if "selected_model" in payload:
            value = payload[
                "selected_model"
            ]

        else:
            selection = payload.get(
                "selection"
            )

            if isinstance(
                selection,
                dict,
            ):
                value = selection.get(
                    "selected_model"
                )

            else:
                value = None

        if (
            value is not None
            and not isinstance(
                value,
                str,
            )
        ):
            raise ValueError(
                "active decision selected_model is invalid"
            )

        return value

    @staticmethod
    def _classify_path(
        relative: str,
    ) -> str:
        path = PurePosixPath(
            relative
        )

        if (
            relative
            == "active/champion_decision.json"
        ):
            return "active_decision"

        if (
            relative
            == "active/publication.json"
        ):
            return "active_publication"

        parts = path.parts

        if (
            len(parts) == 2
            and parts[0] == "history"
            and path.suffix == ".json"
            and len(
                path.stem
            ) == 64
            and _SHA256_PATTERN.fullmatch(
                path.stem
            )
            is not None
        ):
            return "history_revision"

        if (
            len(parts) == 3
            and parts[0] == "history"
            and parts[1] == "decisions"
            and path.suffix == ".json"
            and len(
                path.stem
            ) == 64
            and _SHA256_PATTERN.fullmatch(
                path.stem
            )
            is not None
        ):
            return "history_decision"

        if (
            len(parts) == 3
            and parts[0] == "history"
            and parts[1] == "rollbacks"
            and path.suffix == ".json"
        ):
            return "rollback_provenance"

        raise ValueError(
            "unknown production registry file: "
            f"{relative}"
        )

class ProductionRegistryRestoreAtomicityError(
    RuntimeError
):
    """Raised when restore compensation cannot fully recover state."""


@dataclass(
    frozen=True,
    slots=True,
)
class ProductionRegistryRestoreResult:
    registry_root: Path
    backup_id: str
    restored_file_count: int

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "registry_root",
            Path(
                self.registry_root
            ),
        )

        _validate_sha256(
            self.backup_id,
            field="backup_id",
        )

        if (
            isinstance(
                self.restored_file_count,
                bool,
            )
            or not isinstance(
                self.restored_file_count,
                int,
            )
            or self.restored_file_count < 0
        ):
            raise ValueError(
                "restored_file_count must be "
                "a non-negative integer"
            )


class ProductionRegistryRestoreService:
    """Restore a production registry from a verified backup."""

    def __init__(
        self,
        registry_root: str | Path,
    ) -> None:
        self._registry_root = Path(
            registry_root
        )

    def restore(
        self,
        backup_root: str | Path,
    ) -> ProductionRegistryRestoreResult:
        backup = Path(
            backup_root
        )

        (
            backup_id,
            payload,
        ) = self._validate_backup(
            backup
        )

        with ProductionRegistryWriterLock(
            self._registry_root
        ):
            self._validate_destination()

            original = (
                self._capture_destination()
            )

            try:
                self._replace_registry(
                    payload
                )

                self._verify_restored(
                    payload
                )

            except BaseException as primary_error:
                try:
                    self._compensate_restore(
                        original
                    )

                    self._verify_destination_bytes(
                        original
                    )

                except BaseException as compensation_error:
                    raise (
                        ProductionRegistryRestoreAtomicityError(
                            "restore failed and "
                            "compensation could not "
                            "recover the original registry"
                        )
                    ) from compensation_error

                raise primary_error

        return ProductionRegistryRestoreResult(
            registry_root=self._registry_root,
            backup_id=backup_id,
            restored_file_count=len(
                payload
            ),
        )

    def _capture_destination(
        self,
    ) -> dict[str, bytes]:
        registry = (
            self._registry_root
        )

        if not registry.exists():
            return {}

        captured: dict[
            str,
            bytes,
        ] = {}

        for path in sorted(
            registry.rglob("*")
        ):
            if not path.is_file():
                continue

            relative = (
                path.relative_to(
                    registry
                )
                .as_posix()
            )

            if relative == ".writer.lock":
                continue

            captured[
                relative
            ] = path.read_bytes()

        return captured

    def _compensate_restore(
        self,
        original: dict[str, bytes],
    ) -> None:
        registry = (
            self._registry_root
        )

        registry.mkdir(
            parents=True,
            exist_ok=True,
        )

        current_files = [
            path
            for path in sorted(
                registry.rglob("*"),
                key=lambda item:
                    len(item.parts),
                reverse=True,
            )
            if (
                path.is_file()
                and path.name
                != ".writer.lock"
            )
        ]

        for path in current_files:
            path.unlink()

        current_dirs = [
            path
            for path in sorted(
                registry.rglob("*"),
                key=lambda item:
                    len(item.parts),
                reverse=True,
            )
            if path.is_dir()
        ]

        for path in current_dirs:
            try:
                path.rmdir()

            except OSError:
                pass

        for relative in sorted(
            original
        ):
            target = (
                registry
                / Path(
                    *PurePosixPath(
                        relative
                    ).parts
                )
            )

            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            target.write_bytes(
                original[
                    relative
                ]
            )

    def _verify_destination_bytes(
        self,
        expected: dict[str, bytes],
    ) -> None:
        actual = (
            self._capture_destination()
        )

        if actual != expected:
            raise RuntimeError(
                "restore compensation verification failed"
            )

    def _validate_backup(
        self,
        backup_root: Path,
    ) -> tuple[
        str,
        dict[str, bytes],
    ]:
        if not backup_root.exists():
            raise FileNotFoundError(
                backup_root
            )

        if not backup_root.is_dir():
            raise NotADirectoryError(
                backup_root
            )

        manifest_path = (
            backup_root
            / "manifest.json"
        )

        complete_path = (
            backup_root
            / "COMPLETE"
        )

        payload_root = (
            backup_root
            / "payload"
        )

        if not manifest_path.is_file():
            raise FileNotFoundError(
                manifest_path
            )

        if not complete_path.is_file():
            raise FileNotFoundError(
                complete_path
            )

        if not payload_root.is_dir():
            raise FileNotFoundError(
                payload_root
            )

        try:
            manifest = json.loads(
                manifest_path.read_text(
                    encoding="utf-8-sig"
                )
            )

        except Exception as exc:
            raise ValueError(
                "backup manifest is invalid"
            ) from exc

        if not isinstance(
            manifest,
            dict,
        ):
            raise ValueError(
                "backup manifest must be an object"
            )

        if (
            manifest.get(
                "schema_version"
            )
            != 1
        ):
            raise ValueError(
                "backup manifest schema_version "
                "must be 1"
            )

        computed_backup_id = (
            hashlib.sha256(
                _canonical_json_bytes(
                    manifest
                )
            )
            .hexdigest()
        )

        expected_backup_id = (
            backup_root.name
        )

        _validate_sha256(
            expected_backup_id,
            field="backup_id",
        )

        if (
            computed_backup_id
            != expected_backup_id
        ):
            raise ValueError(
                "backup manifest identity mismatch"
            )

        complete_id = (
            complete_path
            .read_text(
                encoding="utf-8"
            )
            .strip()
        )

        if (
            complete_id
            != expected_backup_id
        ):
            raise ValueError(
                "COMPLETE backup identity mismatch"
            )

        records = manifest.get(
            "files"
        )

        if not isinstance(
            records,
            list,
        ):
            raise ValueError(
                "backup manifest files must be a list"
            )

        expected_paths: set[
            str
        ] = set()

        payload: dict[
            str,
            bytes,
        ] = {}

        for record in records:
            if not isinstance(
                record,
                dict,
            ):
                raise ValueError(
                    "backup file record is invalid"
                )

            relative = record.get(
                "relative_path"
            )

            size = record.get(
                "size"
            )

            expected_sha = record.get(
                "sha256"
            )

            role = record.get(
                "role"
            )

            file_record = (
                ProductionRegistryBackupFile(
                    relative_path=relative,
                    size=size,
                    sha256=expected_sha,
                    role=role,
                )
            )

            if (
                file_record.relative_path
                in expected_paths
            ):
                raise ValueError(
                    "duplicate backup payload path"
                )

            expected_paths.add(
                file_record.relative_path
            )

            path = (
                payload_root
                / Path(
                    *PurePosixPath(
                        file_record.relative_path
                    ).parts
                )
            )

            if not path.is_file():
                raise FileNotFoundError(
                    path
                )

            data = path.read_bytes()

            if (
                len(data)
                != file_record.size
            ):
                raise ValueError(
                    "backup payload size mismatch"
                )

            actual_sha = (
                hashlib.sha256(
                    data
                )
                .hexdigest()
            )

            if (
                actual_sha
                != file_record.sha256
            ):
                raise ValueError(
                    "backup payload sha256 mismatch"
                )

            payload[
                file_record.relative_path
            ] = data

        actual_paths = {
            path.relative_to(
                payload_root
            ).as_posix()
            for path in payload_root.rglob("*")
            if path.is_file()
        }

        if actual_paths != expected_paths:
            raise ValueError(
                "backup payload path set mismatch"
            )

        if ".writer.lock" in actual_paths:
            raise ValueError(
                "backup payload must not contain "
                ".writer.lock"
            )

        self._validate_active_pair(
            payload,
            manifest,
        )

        return (
            expected_backup_id,
            payload,
        )

    def _validate_active_pair(
        self,
        payload: dict[str, bytes],
        manifest: dict[str, object],
    ) -> None:
        decision_path = (
            "active/champion_decision.json"
        )

        publication_path = (
            "active/publication.json"
        )

        decision_present = (
            decision_path in payload
        )

        publication_present = (
            publication_path in payload
        )

        if (
            decision_present
            != publication_present
        ):
            raise ValueError(
                "backup active pair is partial"
            )

        if not (
            decision_present
            and publication_present
        ):
            raise ValueError(
                "backup active pair is missing"
            )

        try:
            decision = json.loads(
                payload[
                    decision_path
                ].decode(
                    "utf-8-sig"
                )
            )

            publication = json.loads(
                payload[
                    publication_path
                ].decode(
                    "utf-8-sig"
                )
            )

        except Exception as exc:
            raise ValueError(
                "backup active pair is invalid"
            ) from exc

        if (
            not isinstance(
                decision,
                dict,
            )
            or not isinstance(
                publication,
                dict,
            )
        ):
            raise ValueError(
                "backup active pair must contain objects"
            )

        active_model = (
            ProductionRegistryBackupService
            ._selected_model(
                decision
            )
        )

        publication_model = (
            publication.get(
                "selected_model"
            )
        )

        if (
            active_model
            != publication_model
        ):
            raise ValueError(
                "backup active pair selected_model mismatch"
            )

        decision_sha = (
            hashlib.sha256(
                payload[
                    decision_path
                ]
            )
            .hexdigest()
        )

        if (
            publication.get(
                "source_sha256"
            )
            != decision_sha
        ):
            raise ValueError(
                "backup active pair source_sha256 mismatch"
            )

        revision_id = (
            publication.get(
                "revision_id"
            )
        )

        if not isinstance(
            revision_id,
            str,
        ):
            raise ValueError(
                "backup active revision_id is invalid"
            )

        _validate_sha256(
            revision_id,
            field="active_revision_id",
        )

        if (
            manifest.get(
                "active_model"
            )
            != active_model
        ):
            raise ValueError(
                "manifest active_model mismatch"
            )

        if (
            manifest.get(
                "active_source_sha256"
            )
            != decision_sha
        ):
            raise ValueError(
                "manifest active_source_sha256 mismatch"
            )

        if (
            manifest.get(
                "active_revision_id"
            )
            != revision_id
        ):
            raise ValueError(
                "manifest active_revision_id mismatch"
            )

    def _validate_destination(
        self,
    ) -> None:
        registry = (
            self._registry_root
        )

        if registry.exists():
            if not registry.is_dir():
                raise NotADirectoryError(
                    registry
                )

            for path in sorted(
                registry.rglob("*")
            ):
                if not path.is_file():
                    continue

                relative = (
                    path.relative_to(
                        registry
                    )
                    .as_posix()
                )

                if relative == ".writer.lock":
                    continue

                ProductionRegistryBackupService._classify_path(
                    relative
                )

    def _replace_registry(
        self,
        payload: dict[str, bytes],
    ) -> None:
        registry = (
            self._registry_root
        )

        registry.mkdir(
            parents=True,
            exist_ok=True,
        )

        existing_files = [
            path
            for path in sorted(
                registry.rglob("*"),
                reverse=True,
            )
            if (
                path.is_file()
                and path.name
                != ".writer.lock"
            )
        ]

        for path in existing_files:
            path.unlink()

        existing_dirs = [
            path
            for path in sorted(
                registry.rglob("*"),
                key=lambda item:
                    len(
                        item.parts
                    ),
                reverse=True,
            )
            if path.is_dir()
        ]

        for path in existing_dirs:
            try:
                path.rmdir()

            except OSError:
                pass

        for relative in sorted(
            payload
        ):
            target = (
                registry
                / Path(
                    *PurePosixPath(
                        relative
                    ).parts
                )
            )

            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            target.write_bytes(
                payload[
                    relative
                ]
            )

    def _verify_restored(
        self,
        payload: dict[str, bytes],
    ) -> None:
        actual = {
            path.relative_to(
                self._registry_root
            ).as_posix():
                path.read_bytes()
            for path
            in self._registry_root.rglob("*")
            if (
                path.is_file()
                and path.name
                != ".writer.lock"
            )
        }

        if actual != payload:
            raise RuntimeError(
                "post-restore verification failed"
            )
