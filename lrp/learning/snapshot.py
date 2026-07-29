"""Revision-consistent learning snapshot writer for Project E E-005C."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from .adaptive_models import AdaptiveWeight
from .adaptive_report import AdaptiveWeightReport, AdaptiveWeightReporter
from .performance import PerformanceAnalyzer, StrategyPerformanceReport
from .ranking import StrategyRanking
from .service import LearningService


_KST = ZoneInfo("Asia/Seoul")
_SNAPSHOT_SCHEMA = "lrp.learning.snapshot.v1"
_JSON_OPTIONS = {
    "ensure_ascii": False,
    "indent": 2,
    "sort_keys": True,
}


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _positive_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _json_bytes(payload: Any) -> bytes:
    text = json.dumps(payload, **_JSON_OPTIONS) + "\n"
    return text.encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class LearningSnapshot:
    """Immutable description of one persisted learning snapshot."""

    round_no: int
    revision: tuple[int, int]
    strategy_type: str | None
    history_limit: int
    generated_at_kst: str
    directory: Path
    files: Mapping[str, str]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        round_no = _positive_integer(self.round_no, field_name="round_no")
        revision = tuple(self.revision)
        if (
            len(revision) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in revision
            )
        ):
            raise ValueError(
                "revision must contain two non-negative integers"
            )
        strategy_type = self.strategy_type
        if strategy_type is not None:
            strategy_type = _required_text(
                strategy_type,
                field_name="strategy_type",
            ).lower()
        history_limit = _positive_integer(
            self.history_limit,
            field_name="history_limit",
        )
        generated_at_kst = _required_text(
            self.generated_at_kst,
            field_name="generated_at_kst",
        )
        directory = Path(self.directory)
        if not isinstance(self.files, Mapping):
            raise ValueError("files must be a mapping")
        files = {
            _required_text(name, field_name="file name"):
            _required_text(digest, field_name="sha256")
            for name, digest in self.files.items()
        }
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")

        object.__setattr__(self, "round_no", round_no)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "strategy_type", strategy_type)
        object.__setattr__(self, "history_limit", history_limit)
        object.__setattr__(self, "generated_at_kst", generated_at_kst)
        object.__setattr__(self, "directory", directory)
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": _SNAPSHOT_SCHEMA,
            "round_no": self.round_no,
            "revision": list(self.revision),
            "strategy_type": self.strategy_type,
            "history_limit": self.history_limit,
            "generated_at_kst": self.generated_at_kst,
            "directory": str(self.directory),
            "files": dict(sorted(self.files.items())),
            "metadata": dict(self.metadata),
        }


class LearningSnapshotWriter:
    """Persist a complete read-only learning state for one Lotto round."""

    def __init__(
        self,
        service: LearningService,
        *,
        performance_analyzer: PerformanceAnalyzer | None = None,
        adaptive_reporter: AdaptiveWeightReporter | None = None,
    ) -> None:
        if not isinstance(service, LearningService):
            raise TypeError("service must be a LearningService")
        self.service = service
        self.performance_analyzer = (
            performance_analyzer
            or PerformanceAnalyzer(service.ranking_repository)
        )
        self.adaptive_reporter = (
            adaptive_reporter or AdaptiveWeightReporter()
        )

    def write(
        self,
        *,
        round_no: int,
        output_root: str | Path,
        strategy_type: str | None = None,
        history_limit: int = 100,
        generated_at_kst: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> LearningSnapshot:
        round_no = _positive_integer(round_no, field_name="round_no")
        history_limit = _positive_integer(
            history_limit,
            field_name="history_limit",
        )
        normalized_type = (
            None
            if strategy_type is None
            else _required_text(
                strategy_type,
                field_name="strategy_type",
            ).lower()
        )
        timestamp = (
            datetime.now(_KST).isoformat(timespec="seconds")
            if generated_at_kst is None
            else _required_text(
                generated_at_kst,
                field_name="generated_at_kst",
            )
        )
        extra_metadata = {} if metadata is None else dict(metadata)
        json.dumps(extra_metadata, ensure_ascii=False)

        directory = Path(output_root) / str(round_no)
        if directory.exists() and any(directory.iterdir()) and not overwrite:
            raise FileExistsError(
                f"snapshot directory already exists: {directory}"
            )
        directory.mkdir(parents=True, exist_ok=True)

        rankings = self.service.rank_strategies(
            strategy_type=normalized_type,
            history_limit=history_limit,
        )
        adaptive_weights = self.service.get_adaptive_weights(
            strategy_type=normalized_type,
            history_limit=history_limit,
        )
        performance = self.performance_analyzer.analyze(
            strategy_type=normalized_type,
            history_limit=history_limit,
            generated_at_kst=timestamp,
        )
        adaptive_report = self.adaptive_reporter.build(
            adaptive_weights,
            strategy_type=normalized_type,
            history_limit=history_limit,
            generated_at_kst=timestamp,
        )

        revision = self._validate_consistency(
            rankings=rankings,
            adaptive_weights=adaptive_weights,
            performance=performance,
            adaptive_report=adaptive_report,
        )

        payloads: dict[str, Any] = {
            "rankings.json": {
                "schema": _SNAPSHOT_SCHEMA,
                "round_no": round_no,
                "revision": list(revision),
                "generated_at_kst": timestamp,
                "strategy_type": normalized_type,
                "history_limit": history_limit,
                "rankings": [item.as_dict() for item in rankings],
            },
            "adaptive_weights.json": {
                "schema": _SNAPSHOT_SCHEMA,
                "round_no": round_no,
                "revision": list(revision),
                "generated_at_kst": timestamp,
                "strategy_type": normalized_type,
                "history_limit": history_limit,
                "weights": [item.as_dict() for item in adaptive_weights],
            },
            "performance.json": performance.as_dict(
                include_history=True
            ),
            "adaptive_report.json": adaptive_report.as_dict(),
        }

        digests: dict[str, str] = {}
        for filename, payload in payloads.items():
            content = _json_bytes(payload)
            _atomic_write(directory / filename, content)
            digests[filename] = _sha256_bytes(content)

        metadata_payload = {
            "schema": _SNAPSHOT_SCHEMA,
            "round_no": round_no,
            "revision": list(revision),
            "strategy_type": normalized_type,
            "history_limit": history_limit,
            "generated_at_kst": timestamp,
            "files": dict(sorted(digests.items())),
            "metadata": {
                "source": "lrp.learning",
                "writer": "E-005C",
                "read_only_source": True,
                **extra_metadata,
            },
        }
        metadata_content = _json_bytes(metadata_payload)
        _atomic_write(directory / "metadata.json", metadata_content)
        digests["metadata.json"] = _sha256_bytes(metadata_content)

        manifest_lines = [
            f"{digest}  {filename}"
            for filename, digest in sorted(digests.items())
        ]
        manifest_content = (
            "\n".join(manifest_lines) + "\n"
        ).encode("utf-8")
        _atomic_write(directory / "SHA256SUMS.txt", manifest_content)
        digests["SHA256SUMS.txt"] = _sha256_bytes(manifest_content)

        return LearningSnapshot(
            round_no=round_no,
            revision=revision,
            strategy_type=normalized_type,
            history_limit=history_limit,
            generated_at_kst=timestamp,
            directory=directory,
            files=digests,
            metadata=metadata_payload["metadata"],
        )

    @staticmethod
    def _validate_consistency(
        *,
        rankings: tuple[StrategyRanking, ...],
        adaptive_weights: tuple[AdaptiveWeight, ...],
        performance: StrategyPerformanceReport,
        adaptive_report: AdaptiveWeightReport,
    ) -> tuple[int, int]:
        revisions = {
            tuple(performance.revision),
            tuple(adaptive_report.revision),
        }
        revisions.update(
            tuple(item.revision) for item in adaptive_weights
        )
        if len(revisions) != 1:
            raise RuntimeError(
                "learning snapshot contains mixed repository revisions"
            )
        revision = next(iter(revisions))

        ranking_keys = {
            (item.strategy_type, item.strategy_name)
            for item in rankings
        }
        weight_keys = {
            (item.strategy_type, item.strategy_name)
            for item in adaptive_weights
        }
        performance_keys = {
            item.key for item in performance.summaries
        }
        report_keys = {
            item.key for item in adaptive_report.summaries
        }

        if not (
            ranking_keys
            == weight_keys
            == performance_keys
            == report_keys
        ):
            raise RuntimeError(
                "learning snapshot strategy sets are inconsistent"
            )

        return revision
