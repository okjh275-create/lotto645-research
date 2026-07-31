from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

from lrp.evolution.contracts import AdaptiveWeightProfile
from lrp.evolution.storage.filesystem import (
    AtomicTextFileSystem,
)
from lrp.evolution.storage.serializer import (
    EvolutionSnapshotSerializer,
    SnapshotSerializationError,
)
from lrp.evolution.storage.snapshot import (
    EvolutionSnapshot,
)


class SnapshotNotFoundError(FileNotFoundError):
    """Raised when an evolution snapshot cannot be found."""


class SnapshotRepository:
    """Revision-based repository for evolution snapshots."""

    FILE_PATTERN = "revision-*.json"
    FILE_NAME_PATTERN = re.compile(
        r"^revision-(\d{8})\.json$"
    )

    def __init__(
        self,
        root: Path | str,
        *,
        serializer: EvolutionSnapshotSerializer | None = None,
        filesystem: AtomicTextFileSystem | None = None,
    ) -> None:
        self._root = Path(root)
        self._serializer = (
            serializer or EvolutionSnapshotSerializer()
        )
        self._filesystem = (
            filesystem or AtomicTextFileSystem()
        )

    @property
    def root(self) -> Path:
        return self._root

    def save(
        self,
        profile: AdaptiveWeightProfile,
        *,
        saved_at: datetime | None = None,
    ) -> EvolutionSnapshot:
        if not isinstance(
            profile,
            AdaptiveWeightProfile,
        ):
            raise TypeError(
                "profile must be an AdaptiveWeightProfile"
            )

        snapshot = EvolutionSnapshot.create(
            profile,
            saved_at=saved_at,
        )

        target = self._path_for_revision(
            snapshot.revision
        )

        content = self._serializer.dumps(snapshot)

        self._filesystem.write_atomic(
            target,
            content,
            overwrite=False,
        )

        return snapshot

    def load_revision(
        self,
        revision: int,
    ) -> EvolutionSnapshot:
        self._validate_revision(revision)

        path = self._path_for_revision(revision)

        if not path.exists():
            raise SnapshotNotFoundError(
                f"snapshot revision not found: {revision}"
            )

        return self._load_path(path)

    def load_latest(
        self,
        *,
        skip_corrupt: bool = True,
    ) -> EvolutionSnapshot:
        paths = self._revision_paths(descending=True)

        if not paths:
            raise SnapshotNotFoundError(
                "no evolution snapshots found"
            )

        last_error: SnapshotSerializationError | None = None

        for path in paths:
            try:
                return self._load_path(path)
            except SnapshotSerializationError as exc:
                last_error = exc

                if not skip_corrupt:
                    raise

        raise SnapshotSerializationError(
            "no valid evolution snapshots found"
        ) from last_error

    def history(
        self,
        *,
        skip_corrupt: bool = True,
    ) -> tuple[EvolutionSnapshot, ...]:
        snapshots: list[EvolutionSnapshot] = []

        for path in self._revision_paths(
            descending=False,
        ):
            try:
                snapshots.append(
                    self._load_path(path)
                )
            except SnapshotSerializationError:
                if not skip_corrupt:
                    raise

        return tuple(snapshots)

    def revisions(self) -> tuple[int, ...]:
        return tuple(
            self._revision_from_path(path)
            for path in self._revision_paths(
                descending=False,
            )
        )

    def exists(
        self,
        revision: int,
    ) -> bool:
        self._validate_revision(revision)

        return self._path_for_revision(
            revision
        ).exists()

    def _load_path(
        self,
        path: Path,
    ) -> EvolutionSnapshot:
        content = self._filesystem.read_text(path)
        snapshot = self._serializer.loads(content)

        expected_revision = self._revision_from_path(path)

        if snapshot.revision != expected_revision:
            raise SnapshotSerializationError(
                "snapshot revision does not match filename"
            )

        return snapshot

    def _revision_paths(
        self,
        *,
        descending: bool,
    ) -> tuple[Path, ...]:
        candidates = self._filesystem.list_files(
            self.root,
            pattern=self.FILE_PATTERN,
        )

        valid = [
            path
            for path in candidates
            if self.FILE_NAME_PATTERN.match(path.name)
        ]

        valid.sort(
            key=self._revision_from_path,
            reverse=descending,
        )

        return tuple(valid)

    def _path_for_revision(
        self,
        revision: int,
    ) -> Path:
        self._validate_revision(revision)

        return self.root / (
            f"revision-{revision:08d}.json"
        )

    @classmethod
    def _revision_from_path(
        cls,
        path: Path,
    ) -> int:
        match = cls.FILE_NAME_PATTERN.match(
            path.name
        )

        if match is None:
            raise ValueError(
                f"invalid snapshot filename: {path.name}"
            )

        return int(match.group(1))

    @staticmethod
    def _validate_revision(
        revision: int,
    ) -> None:
        if isinstance(revision, bool):
            raise TypeError(
                "revision must be an integer"
            )

        if not isinstance(revision, int):
            raise TypeError(
                "revision must be an integer"
            )

        if revision < 1:
            raise ValueError(
                "revision must be greater than "
                "or equal to 1"
            )