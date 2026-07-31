from __future__ import annotations

import os
import re
from pathlib import Path
from uuid import uuid4

from lrp.evolution.contracts.snapshot_schema import (
    LearningCycleSnapshot,
)
from lrp.evolution.repositories.snapshot_repository import (
    SnapshotRepository,
)
from lrp.evolution.serialization.json_snapshot_serializer import (
    JsonSnapshotSerializer,
)


class FileSnapshotRepository(
    SnapshotRepository
):
    """Store learning snapshots as UTF-8 JSON files."""

    _SNAPSHOT_ID_PATTERN = re.compile(
        r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
    )

    def __init__(
        self,
        root_directory: str | Path,
        serializer: (
            JsonSnapshotSerializer | None
        ) = None,
    ) -> None:
        self._root_directory = (
            self._normalize_root_directory(
                root_directory
            )
        )

        if (
            serializer is not None
            and not isinstance(
                serializer,
                JsonSnapshotSerializer,
            )
        ):
            raise TypeError(
                "serializer must be a "
                "JsonSnapshotSerializer"
            )

        self._serializer = (
            serializer
            if serializer is not None
            else JsonSnapshotSerializer()
        )

        self._root_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self._root_directory.is_dir():
            raise NotADirectoryError(
                str(self._root_directory)
            )

    @property
    def root_directory(self) -> Path:
        return self._root_directory

    def save(
        self,
        snapshot: LearningCycleSnapshot,
        *,
        overwrite: bool = False,
    ) -> None:
        if not isinstance(
            snapshot,
            LearningCycleSnapshot,
        ):
            raise TypeError(
                "snapshot must be a "
                "LearningCycleSnapshot"
            )

        if not isinstance(overwrite, bool):
            raise TypeError(
                "overwrite must be a boolean"
            )

        snapshot_id = self._normalize_snapshot_id(
            snapshot.snapshot_id
        )
        target_path = self._snapshot_path(
            snapshot_id
        )

        if (
            target_path.exists()
            and not overwrite
        ):
            raise FileExistsError(
                f"snapshot already exists: "
                f"{snapshot_id}"
            )

        serialized = self._serializer.serialize(
            snapshot
        )

        temporary_path = (
            self._temporary_path(snapshot_id)
        )

        try:
            self._write_temporary_file(
                path=temporary_path,
                serialized=serialized,
            )

            if (
                target_path.exists()
                and not overwrite
            ):
                raise FileExistsError(
                    f"snapshot already exists: "
                    f"{snapshot_id}"
                )

            os.replace(
                temporary_path,
                target_path,
            )
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def load(
        self,
        snapshot_id: str,
    ) -> LearningCycleSnapshot:
        normalized_id = (
            self._normalize_snapshot_id(
                snapshot_id
            )
        )
        path = self._snapshot_path(
            normalized_id
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"snapshot does not exist: "
                f"{normalized_id}"
            )

        serialized = path.read_text(
            encoding="utf-8"
        )
        snapshot = (
            self._serializer.deserialize(
                serialized
            )
        )

        if snapshot.snapshot_id != normalized_id:
            raise ValueError(
                "stored snapshot_id does not match "
                "requested snapshot_id"
            )

        return snapshot

    def exists(
        self,
        snapshot_id: str,
    ) -> bool:
        normalized_id = (
            self._normalize_snapshot_id(
                snapshot_id
            )
        )

        return self._snapshot_path(
            normalized_id
        ).is_file()

    def list_ids(self) -> tuple[str, ...]:
        snapshot_ids: list[str] = []

        for path in self._root_directory.glob(
            "*.json"
        ):
            if not path.is_file():
                continue

            snapshot_id = path.stem

            try:
                normalized_id = (
                    self._normalize_snapshot_id(
                        snapshot_id
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            snapshot_ids.append(
                normalized_id
            )

        return tuple(sorted(snapshot_ids))

    def delete(
        self,
        snapshot_id: str,
    ) -> bool:
        normalized_id = (
            self._normalize_snapshot_id(
                snapshot_id
            )
        )
        path = self._snapshot_path(
            normalized_id
        )

        try:
            path.unlink()
        except FileNotFoundError:
            return False

        return True

    @classmethod
    def _normalize_snapshot_id(
        cls,
        snapshot_id: str,
    ) -> str:
        if not isinstance(snapshot_id, str):
            raise TypeError(
                "snapshot_id must be a string"
            )

        normalized = snapshot_id.strip()

        if not normalized:
            raise ValueError(
                "snapshot_id must not be empty"
            )

        if normalized in {".", ".."}:
            raise ValueError(
                "snapshot_id contains invalid "
                "path content"
            )

        if (
            cls._SNAPSHOT_ID_PATTERN.fullmatch(
                normalized
            )
            is None
        ):
            raise ValueError(
                "snapshot_id may contain only "
                "letters, numbers, dots, "
                "underscores, and hyphens"
            )

        return normalized

    @staticmethod
    def _normalize_root_directory(
        root_directory: str | Path,
    ) -> Path:
        if not isinstance(
            root_directory,
            (str, Path),
        ):
            raise TypeError(
                "root_directory must be a "
                "string or Path"
            )

        if isinstance(root_directory, str):
            normalized_text = (
                root_directory.strip()
            )

            if not normalized_text:
                raise ValueError(
                    "root_directory must not "
                    "be empty"
                )

            path = Path(normalized_text)
        else:
            path = root_directory

        return path.expanduser().resolve()

    def _snapshot_path(
        self,
        snapshot_id: str,
    ) -> Path:
        return self._root_directory / (
            f"{snapshot_id}.json"
        )

    def _temporary_path(
        self,
        snapshot_id: str,
    ) -> Path:
        return self._root_directory / (
            f".{snapshot_id}."
            f"{uuid4().hex}.tmp"
        )

    @staticmethod
    def _write_temporary_file(
        *,
        path: Path,
        serialized: str,
    ) -> None:
        with path.open(
            mode="x",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
