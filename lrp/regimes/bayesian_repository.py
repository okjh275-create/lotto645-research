from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

from lrp.evolution.storage.filesystem import (
    AtomicTextFileSystem,
)
from lrp.regimes.bayesian_serializer import (
    RegimeBayesianSerializationError,
    RegimeBayesianSnapshotSerializer,
)
from lrp.regimes.bayesian_snapshot import (
    RegimeBayesianSnapshot,
)
from lrp.regimes.bayesian_state import (
    RegimeBayesianState,
)


class RegimeBayesianNotFoundError(FileNotFoundError):
    """Raised when a regime Bayesian snapshot cannot be found."""


class RegimeBayesianRepository:
    """Revision-based repository for regime Bayesian snapshots."""

    FILE_PATTERN = "revision-*.json"
    FILE_NAME_PATTERN = re.compile(
        r"^revision-(\d{8})\.json$"
    )

    def __init__(
        self,
        root: Path | str,
        *,
        serializer: RegimeBayesianSnapshotSerializer | None = None,
        filesystem: AtomicTextFileSystem | None = None,
    ) -> None:
        self._root = Path(root)
        self._serializer = (
            serializer
            or RegimeBayesianSnapshotSerializer()
        )
        self._filesystem = (
            filesystem
            or AtomicTextFileSystem()
        )

    @property
    def root(self) -> Path:
        return self._root

    def save(
        self,
        state: RegimeBayesianState,
        *,
        revision: int,
        sample_size: int = 0,
        saved_at: datetime | None = None,
    ) -> RegimeBayesianSnapshot:
        if not isinstance(
            state,
            RegimeBayesianState,
        ):
            raise TypeError(
                "state must be a RegimeBayesianState"
            )

        snapshot = RegimeBayesianSnapshot.create(
            state,
            revision=revision,
            sample_size=sample_size,
            saved_at=saved_at,
        )

        target = self._path_for_revision(
            snapshot.revision
        )

        self._filesystem.write_atomic(
            target,
            self._serializer.dumps(snapshot),
            overwrite=False,
        )

        return snapshot

    def load_revision(
        self,
        revision: int,
    ) -> RegimeBayesianSnapshot:
        self._validate_revision(revision)

        path = self._path_for_revision(revision)

        if not path.exists():
            raise RegimeBayesianNotFoundError(
                f"regime Bayesian revision not found: {revision}"
            )

        return self._load_path(path)

    def load_latest(
        self,
        *,
        skip_corrupt: bool = True,
    ) -> RegimeBayesianSnapshot:
        paths = self._revision_paths(
            descending=True
        )

        if not paths:
            raise RegimeBayesianNotFoundError(
                "no regime Bayesian snapshots found"
            )

        last_error: (
            RegimeBayesianSerializationError | None
        ) = None

        for path in paths:
            try:
                return self._load_path(path)
            except RegimeBayesianSerializationError as exc:
                last_error = exc

                if not skip_corrupt:
                    raise

        raise RegimeBayesianSerializationError(
            "no valid regime Bayesian snapshots found"
        ) from last_error

    def history(
        self,
        *,
        skip_corrupt: bool = True,
    ) -> tuple[RegimeBayesianSnapshot, ...]:
        snapshots: list[
            RegimeBayesianSnapshot
        ] = []

        for path in self._revision_paths(
            descending=False
        ):
            try:
                snapshots.append(
                    self._load_path(path)
                )
            except RegimeBayesianSerializationError:
                if not skip_corrupt:
                    raise

        return tuple(snapshots)

    def revisions(self) -> tuple[int, ...]:
        return tuple(
            self._revision_from_path(path)
            for path in self._revision_paths(
                descending=False
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
    ) -> RegimeBayesianSnapshot:
        snapshot = self._serializer.loads(
            self._filesystem.read_text(path)
        )

        expected_revision = (
            self._revision_from_path(path)
        )

        if snapshot.revision != expected_revision:
            raise RegimeBayesianSerializationError(
                "regime Bayesian revision does not match filename"
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
            if self.FILE_NAME_PATTERN.match(
                path.name
            )
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
                f"invalid regime Bayesian filename: {path.name}"
            )

        return int(match.group(1))

    @staticmethod
    def _validate_revision(
        revision: int,
    ) -> None:
        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
        ):
            raise TypeError(
                "revision must be an integer"
            )

        if revision < 1:
            raise ValueError(
                "revision must be greater than or equal to 1"
            )