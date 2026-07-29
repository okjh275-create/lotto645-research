"""Learning workflow coordination for Project E-005D."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping

from .aggregator import StrategyAggregationSummary
from .service import IncrementalReviewSummary
from .snapshot import LearningSnapshot


def _positive_integer(
    value: object,
    *,
    field_name: str,
) -> int:
    """Validate and return a positive integer."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            f"{field_name} must be a positive integer"
        )

    return value


def _optional_positive_integer(
    value: object,
    *,
    field_name: str,
) -> int | None:
    """Validate an optional positive integer."""

    if value is None:
        return None

    return _positive_integer(
        value,
        field_name=field_name,
    )


def _optional_text(
    value: object,
    *,
    field_name: str,
) -> str | None:
    """Validate and normalize optional non-empty text."""

    if value is None:
        return None

    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be a string"
        )

    normalized = value.strip()

    if not normalized:
        raise ValueError(
            f"{field_name} must not be empty"
        )

    return normalized


@dataclass(frozen=True, slots=True)
class LearningCoordinatorConfig:
    """Validated options for one coordinated learning run."""

    strategy_type: str | None = None
    history_limit: int = 100
    review_limit: int | None = None
    aggregation_limit: int | None = None
    overwrite_snapshot: bool = False

    def __post_init__(self) -> None:
        strategy_type = _optional_text(
            self.strategy_type,
            field_name="strategy_type",
        )

        if strategy_type is not None:
            strategy_type = strategy_type.lower()

        history_limit = _positive_integer(
            self.history_limit,
            field_name="history_limit",
        )

        review_limit = _optional_positive_integer(
            self.review_limit,
            field_name="review_limit",
        )

        aggregation_limit = _optional_positive_integer(
            self.aggregation_limit,
            field_name="aggregation_limit",
        )

        if not isinstance(
            self.overwrite_snapshot,
            bool,
        ):
            raise ValueError(
                "overwrite_snapshot must be a boolean"
            )

        object.__setattr__(
            self,
            "strategy_type",
            strategy_type,
        )
        object.__setattr__(
            self,
            "history_limit",
            history_limit,
        )
        object.__setattr__(
            self,
            "review_limit",
            review_limit,
        )
        object.__setattr__(
            self,
            "aggregation_limit",
            aggregation_limit,
        )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible configuration mapping."""

        return {
            "strategy_type": self.strategy_type,
            "history_limit": self.history_limit,
            "review_limit": self.review_limit,
            "aggregation_limit": self.aggregation_limit,
            "overwrite_snapshot": self.overwrite_snapshot,
        }


@dataclass(frozen=True, slots=True)
class LearningCoordinatorResult:
    """Immutable result of one coordinated learning run."""

    round_no: int
    review_summary: IncrementalReviewSummary
    aggregation_summary: StrategyAggregationSummary
    snapshot: LearningSnapshot
    elapsed_seconds: float
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        round_no = _positive_integer(
            self.round_no,
            field_name="round_no",
        )

        if not isinstance(
            self.review_summary,
            IncrementalReviewSummary,
        ):
            raise TypeError(
                "review_summary must be an "
                "IncrementalReviewSummary"
            )

        if not isinstance(
            self.aggregation_summary,
            StrategyAggregationSummary,
        ):
            raise TypeError(
                "aggregation_summary must be a "
                "StrategyAggregationSummary"
            )

        if not isinstance(
            self.snapshot,
            LearningSnapshot,
        ):
            raise TypeError(
                "snapshot must be a LearningSnapshot"
            )

        if self.snapshot.round_no != round_no:
            raise ValueError(
                "snapshot round_no does not match "
                "result round_no"
            )

        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(
                self.elapsed_seconds,
                (int, float),
            )
        ):
            raise TypeError(
                "elapsed_seconds must be numeric"
            )

        elapsed_seconds = float(
            self.elapsed_seconds
        )

        if (
            not math.isfinite(elapsed_seconds)
            or elapsed_seconds < 0.0
        ):
            raise ValueError(
                "elapsed_seconds must be finite "
                "and non-negative"
            )

        if not isinstance(self.metadata, Mapping):
            raise TypeError(
                "metadata must be a mapping"
            )

        object.__setattr__(
            self,
            "round_no",
            round_no,
        )
        object.__setattr__(
            self,
            "elapsed_seconds",
            elapsed_seconds,
        )
        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )

    @property
    def revision(self) -> tuple[int, int]:
        """Return the persisted repository revision."""

        return self.snapshot.revision

    @property
    def directory(self) -> Path:
        """Return the snapshot directory."""

        return self.snapshot.directory

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result mapping."""

        return {
            "round_no": self.round_no,
            "revision": list(self.revision),
            "review": self.review_summary.as_dict(),
            "aggregation": (
                self.aggregation_summary.as_dict()
            ),
            "snapshot": self.snapshot.as_dict(),
            "elapsed_seconds": self.elapsed_seconds,
            "metadata": dict(self.metadata),
        }


from .service import LearningService
from .snapshot import LearningSnapshotWriter


class LearningCoordinator:
    """Coordinates the complete learning workflow."""

    def __init__(
        self,
        service: LearningService,
        snapshot_writer: LearningSnapshotWriter,
    ) -> None:
        if not isinstance(service, LearningService):
            raise TypeError(
                "service must be a LearningService"
            )

        if not isinstance(
            snapshot_writer,
            LearningSnapshotWriter,
        ):
            raise TypeError(
                "snapshot_writer must be a "
                "LearningSnapshotWriter"
            )

        self._service = service
        self._snapshot_writer = snapshot_writer
