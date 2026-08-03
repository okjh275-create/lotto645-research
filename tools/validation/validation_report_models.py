"""Immutable validation-report contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tools.validation.validation_run_discovery import (
    ValidationRunRecord,
)


@dataclass(frozen=True, slots=True)
class ValidationReportSummary:
    """Aggregate validation-run counts and coverage."""

    run_count: int
    replay_count: int
    policy_comparison_count: int
    pass_count: int
    incomplete_count: int
    earliest_round: int | None
    latest_round: int | None
    duplicate_window_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_count": self.run_count,
            "replay_count": self.replay_count,
            "policy_comparison_count": (
                self.policy_comparison_count
            ),
            "pass_count": self.pass_count,
            "incomplete_count": (
                self.incomplete_count
            ),
            "earliest_round": (
                self.earliest_round
            ),
            "latest_round": self.latest_round,
            "duplicate_window_count": (
                self.duplicate_window_count
            ),
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Deterministic report of discovered validation runs."""

    schema_version: int
    generated_at_utc: datetime
    source_root: Path
    summary: ValidationReportSummary
    runs: tuple[ValidationRunRecord, ...]
    duplicate_windows: tuple[
        tuple[int, int, str],
        ...,
    ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "generated_at_utc": (
                self.generated_at_utc
                .astimezone(timezone.utc)
                .isoformat()
            ),
            "source_root": str(
                self.source_root
            ),
            "summary": self.summary.as_dict(),
            "runs": [
                run.as_dict()
                for run in self.runs
            ],
            "duplicate_windows": [
                {
                    "start_round": start_round,
                    "end_round": end_round,
                    "run_type": run_type,
                }
                for (
                    start_round,
                    end_round,
                    run_type,
                ) in self.duplicate_windows
            ],
        }


class ValidationReportBuilder:
    """Build aggregate validation reports from discovered runs."""

    SCHEMA_VERSION = 1

    def build(
        self,
        *,
        source_root: Path,
        runs: Iterable[
            ValidationRunRecord
        ],
        generated_at_utc: (
            datetime | None
        ) = None,
    ) -> ValidationReport:
        source_root = Path(source_root)

        if not source_root.exists():
            raise FileNotFoundError(
                source_root
            )

        if not source_root.is_dir():
            raise NotADirectoryError(
                source_root
            )

        normalized_runs = tuple(runs)

        for run in normalized_runs:
            if not isinstance(
                run,
                ValidationRunRecord,
            ):
                raise TypeError(
                    "runs must contain only "
                    "ValidationRunRecord values"
                )

        timestamp = (
            generated_at_utc
            or datetime.now(timezone.utc)
        )

        if not isinstance(
            timestamp,
            datetime,
        ):
            raise TypeError(
                "generated_at_utc must be "
                "a datetime or None"
            )

        if timestamp.tzinfo is None:
            raise ValueError(
                "generated_at_utc must be "
                "timezone-aware"
            )

        sorted_runs = tuple(
            sorted(
                normalized_runs,
                key=lambda item: (
                    item.start_round,
                    item.end_round,
                    item.run_type,
                    item.policy_name or "",
                    str(item.root),
                ),
            )
        )

        duplicate_windows = (
            self._duplicate_windows(
                sorted_runs
            )
        )

        summary = self._summary(
            sorted_runs,
            duplicate_windows,
        )

        return ValidationReport(
            schema_version=(
                self.SCHEMA_VERSION
            ),
            generated_at_utc=timestamp,
            source_root=source_root,
            summary=summary,
            runs=sorted_runs,
            duplicate_windows=(
                duplicate_windows
            ),
        )

    @staticmethod
    def _summary(
        runs: tuple[
            ValidationRunRecord,
            ...,
        ],
        duplicate_windows: tuple[
            tuple[int, int, str],
            ...,
        ],
    ) -> ValidationReportSummary:
        replay_count = sum(
            run.run_type == "replay"
            for run in runs
        )
        policy_count = sum(
            run.run_type
            == "policy_comparison"
            for run in runs
        )
        pass_count = sum(
            run.status == "PASS"
            for run in runs
        )
        incomplete_count = sum(
            run.status == "INCOMPLETE"
            for run in runs
        )

        if runs:
            earliest_round = min(
                run.start_round
                for run in runs
            )
            latest_round = max(
                run.end_round
                for run in runs
            )
        else:
            earliest_round = None
            latest_round = None

        return ValidationReportSummary(
            run_count=len(runs),
            replay_count=replay_count,
            policy_comparison_count=(
                policy_count
            ),
            pass_count=pass_count,
            incomplete_count=(
                incomplete_count
            ),
            earliest_round=earliest_round,
            latest_round=latest_round,
            duplicate_window_count=len(
                duplicate_windows
            ),
        )

    @staticmethod
    def _duplicate_windows(
        runs: tuple[
            ValidationRunRecord,
            ...,
        ],
    ) -> tuple[
        tuple[int, int, str],
        ...,
    ]:
        counts: dict[
            tuple[int, int, str],
            int,
        ] = {}

        for run in runs:
            key = (
                run.start_round,
                run.end_round,
                run.run_type,
            )

            counts[key] = (
                counts.get(key, 0) + 1
            )

        duplicates = [
            key
            for key, count
            in counts.items()
            if count > 1
        ]

        duplicates.sort()

        return tuple(duplicates)
