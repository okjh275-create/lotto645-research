from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

from lrp.evolution.contracts.regime_calibration import (
    RegimeCalibration,
)
from lrp.evolution.storage.filesystem import (
    AtomicTextFileSystem,
)
from lrp.regimes.calibration_serializer import (
    RegimeCalibrationSerializationError,
    RegimeCalibrationSnapshotSerializer,
)
from lrp.regimes.calibration_snapshot import (
    RegimeCalibrationSnapshot,
)


class RegimeCalibrationNotFoundError(FileNotFoundError):
    """Raised when a regime calibration snapshot cannot be found."""


class RegimeCalibrationRepository:
    """Revision-based repository for regime calibration snapshots."""

    FILE_PATTERN = "revision-*.json"
    FILE_NAME_PATTERN = re.compile(
        r"^revision-(\d{8})\.json$"
    )

    def __init__(
        self,
        root: Path | str,
        *,
        serializer: RegimeCalibrationSnapshotSerializer | None = None,
        filesystem: AtomicTextFileSystem | None = None,
    ) -> None:
        self._root = Path(root)
        self._serializer = (
            serializer
            or RegimeCalibrationSnapshotSerializer()
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
        calibration: RegimeCalibration,
        *,
        revision: int,
        sample_size: int = 0,
        saved_at: datetime | None = None,
    ) -> RegimeCalibrationSnapshot:
        if not isinstance(
            calibration,
            RegimeCalibration,
        ):
            raise TypeError(
                "calibration must be a RegimeCalibration"
            )

        snapshot = RegimeCalibrationSnapshot.create(
            calibration,
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
    ) -> RegimeCalibrationSnapshot:
        self._validate_revision(revision)
        path = self._path_for_revision(revision)

        if not path.exists():
            raise RegimeCalibrationNotFoundError(
                f"regime calibration revision not found: {revision}"
            )

        return self._load_path(path)

    def load_latest(
        self,
        *,
        skip_corrupt: bool = True,
    ) -> RegimeCalibrationSnapshot:
        paths = self._revision_paths(descending=True)

        if not paths:
            raise RegimeCalibrationNotFoundError(
                "no regime calibration snapshots found"
            )

        last_error: RegimeCalibrationSerializationError | None = None

        for path in paths:
            try:
                return self._load_path(path)
            except RegimeCalibrationSerializationError as exc:
                last_error = exc
                if not skip_corrupt:
                    raise

        raise RegimeCalibrationSerializationError(
            "no valid regime calibration snapshots found"
        ) from last_error

    def history(
        self,
        *,
        skip_corrupt: bool = True,
    ) -> tuple[RegimeCalibrationSnapshot, ...]:
        snapshots: list[RegimeCalibrationSnapshot] = []

        for path in self._revision_paths(descending=False):
            try:
                snapshots.append(self._load_path(path))
            except RegimeCalibrationSerializationError:
                if not skip_corrupt:
                    raise

        return tuple(snapshots)

    def revisions(self) -> tuple[int, ...]:
        return tuple(
            self._revision_from_path(path)
            for path in self._revision_paths(descending=False)
        )

    def exists(self, revision: int) -> bool:
        self._validate_revision(revision)
        return self._path_for_revision(revision).exists()

    def _load_path(
        self,
        path: Path,
    ) -> RegimeCalibrationSnapshot:
        snapshot = self._serializer.loads(
            self._filesystem.read_text(path)
        )

        expected_revision = self._revision_from_path(path)

        if snapshot.revision != expected_revision:
            raise RegimeCalibrationSerializationError(
                "regime calibration revision does not match filename"
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

    def _path_for_revision(self, revision: int) -> Path:
        self._validate_revision(revision)
        return self.root / f"revision-{revision:08d}.json"

    @classmethod
    def _revision_from_path(cls, path: Path) -> int:
        match = cls.FILE_NAME_PATTERN.match(path.name)

        if match is None:
            raise ValueError(
                f"invalid regime calibration filename: {path.name}"
            )

        return int(match.group(1))

    @staticmethod
    def _validate_revision(revision: int) -> None:
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
