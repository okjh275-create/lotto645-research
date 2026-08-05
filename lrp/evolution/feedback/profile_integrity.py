"""Inspect adaptive profile revision integrity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Any

from lrp.evolution.feedback.repository import (
    AdaptiveAutomationRepository,
)
from lrp.evolution.feedback.status import (
    AdaptiveStatusIssue,
)


@dataclass(frozen=True, slots=True)
class AdaptiveProfileIntegrityReport:
    """Detailed profile revision integrity report."""

    revision_count: int
    first_revision: int | None
    latest_revision: int | None
    rollback_count: int
    duplicate_profile_count: int
    issues: tuple[AdaptiveStatusIssue, ...]

    @property
    def overall_ok(self) -> bool:
        return not any(
            issue.severity == "error"
            for issue in self.issues
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "revision_count": self.revision_count,
            "first_revision": self.first_revision,
            "latest_revision": self.latest_revision,
            "rollback_count": self.rollback_count,
            "duplicate_profile_count": (
                self.duplicate_profile_count
            ),
            "overall_ok": self.overall_ok,
            "issues": [
                issue.as_dict()
                for issue in self.issues
            ],
        }


class AdaptiveProfileIntegrityDoctor:
    """Validate profile revision lineage and metadata."""

    WEIGHT_FIELDS = (
        "hot_weight",
        "cold_weight",
        "gap_weight",
        "trend_weight",
        "transition_weight",
        "learning_weight",
        "adaptive_weight",
    )

    def inspect(
        self,
        repository: AdaptiveAutomationRepository,
    ) -> AdaptiveProfileIntegrityReport:
        if not isinstance(
            repository,
            AdaptiveAutomationRepository,
        ):
            raise TypeError(
                "repository must be an "
                "AdaptiveAutomationRepository"
            )

        revisions = (
            repository.list_profile_revisions()
        )
        issues: list[AdaptiveStatusIssue] = []

        previous_revision: int | None = None
        previous_timestamp: datetime | None = None
        previous_signature: tuple[float, ...] | None = None

        rollback_count = 0
        duplicate_profile_count = 0

        for revision in revisions:
            try:
                payload = (
                    repository
                    .load_profile_revision(
                        revision
                    )
                )

                result = self._inspect_revision(
                    payload=payload,
                    filename_revision=revision,
                    previous_revision=(
                        previous_revision
                    ),
                    previous_timestamp=(
                        previous_timestamp
                    ),
                    previous_signature=(
                        previous_signature
                    ),
                    issues=issues,
                )

                if result["is_rollback"]:
                    rollback_count += 1

                if result["is_duplicate"]:
                    duplicate_profile_count += 1

                previous_revision = revision
                previous_timestamp = result[
                    "timestamp"
                ]
                previous_signature = result[
                    "signature"
                ]

            except (
                FileNotFoundError,
                TypeError,
                ValueError,
            ) as exc:
                issues.append(
                    AdaptiveStatusIssue(
                        category="profile_integrity",
                        severity="error",
                        message=(
                            f"revision {revision}: "
                            f"{exc}"
                        ),
                    )
                )

        return AdaptiveProfileIntegrityReport(
            revision_count=len(revisions),
            first_revision=(
                revisions[0]
                if revisions
                else None
            ),
            latest_revision=(
                revisions[-1]
                if revisions
                else None
            ),
            rollback_count=rollback_count,
            duplicate_profile_count=(
                duplicate_profile_count
            ),
            issues=tuple(issues),
        )

    def _inspect_revision(
        self,
        *,
        payload: Mapping[str, Any],
        filename_revision: int,
        previous_revision: int | None,
        previous_timestamp: datetime | None,
        previous_signature: tuple[
            float,
            ...,
        ] | None,
        issues: list[AdaptiveStatusIssue],
    ) -> dict[str, Any]:
        target_revision = self._integer(
            payload,
            "target_revision",
        )

        if target_revision != filename_revision:
            self._error(
                issues,
                (
                    f"revision filename "
                    f"{filename_revision} does not "
                    f"match target_revision "
                    f"{target_revision}"
                ),
            )

        profile = payload.get("profile")

        if not isinstance(
            profile,
            Mapping,
        ):
            raise TypeError(
                "profile must be an object"
            )

        profile_revision = self._integer(
            profile,
            "revision",
        )

        if profile_revision != filename_revision:
            self._error(
                issues,
                (
                    f"profile revision "
                    f"{profile_revision} does not "
                    f"match filename revision "
                    f"{filename_revision}"
                ),
            )

        source_revision = payload.get(
            "source_revision"
        )

        if source_revision is not None:
            if (
                isinstance(source_revision, bool)
                or not isinstance(
                    source_revision,
                    int,
                )
            ):
                raise TypeError(
                    "source_revision must be "
                    "an integer or None"
                )

            if (
                source_revision + 1
                != target_revision
            ):
                self._error(
                    issues,
                    (
                        f"source_revision "
                        f"{source_revision} does not "
                        f"lead to target_revision "
                        f"{target_revision}"
                    ),
                )

        if (
            previous_revision is not None
            and filename_revision
            != previous_revision + 1
        ):
            self._error(
                issues,
                (
                    "revision continuity failure: "
                    f"{previous_revision} to "
                    f"{filename_revision}"
                ),
            )

        is_rollback = (
            payload.get("record_type")
            == "rollback"
        )

        if is_rollback:
            rollback_revision = self._integer(
                payload,
                "rollback_revision",
            )

            if source_revision is None:
                self._error(
                    issues,
                    (
                        f"rollback revision "
                        f"{filename_revision} is "
                        "missing source_revision"
                    ),
                )
            elif rollback_revision >= source_revision:
                self._error(
                    issues,
                    (
                        f"rollback_revision "
                        f"{rollback_revision} must be "
                        "older than source_revision "
                        f"{source_revision}"
                    ),
                )

        timestamp = self._timestamp(
            profile,
            "generated_at",
        )

        if (
            previous_timestamp is not None
            and timestamp < previous_timestamp
        ):
            self._error(
                issues,
                (
                    f"revision {filename_revision} "
                    "generated_at is earlier than "
                    "the previous revision"
                ),
            )

        signature = tuple(
            self._number(
                profile,
                field,
            )
            for field in self.WEIGHT_FIELDS
        )

        for field, value in zip(
            self.WEIGHT_FIELDS,
            signature,
        ):
            if not 0.0 <= value <= 1.0:
                self._error(
                    issues,
                    (
                        f"revision "
                        f"{filename_revision} "
                        f"{field} is outside "
                        "0.0 to 1.0"
                    ),
                )

        total = sum(signature)

        if abs(total - 1.0) > 1e-9:
            self._error(
                issues,
                (
                    f"revision "
                    f"{filename_revision} "
                    f"weights sum to {total}"
                ),
            )

        confidence = self._number(
            profile,
            "confidence",
        )

        if not 0.0 <= confidence <= 1.0:
            self._error(
                issues,
                (
                    f"revision "
                    f"{filename_revision} "
                    "confidence is outside "
                    "0.0 to 1.0"
                ),
            )

        sample_size = self._integer(
            profile,
            "sample_size",
        )

        if sample_size < 0:
            self._error(
                issues,
                (
                    f"revision "
                    f"{filename_revision} "
                    "sample_size is negative"
                ),
            )

        is_duplicate = (
            previous_signature is not None
            and signature == previous_signature
        )

        if is_duplicate:
            issues.append(
                AdaptiveStatusIssue(
                    category=(
                        "profile_integrity"
                    ),
                    severity="warning",
                    message=(
                        f"revision "
                        f"{filename_revision} "
                        "duplicates the previous "
                        "revision weights"
                    ),
                )
            )

        return {
            "timestamp": timestamp,
            "signature": signature,
            "is_rollback": is_rollback,
            "is_duplicate": is_duplicate,
        }

    @staticmethod
    def _error(
        issues: list[AdaptiveStatusIssue],
        message: str,
    ) -> None:
        issues.append(
            AdaptiveStatusIssue(
                category="profile_integrity",
                severity="error",
                message=message,
            )
        )

    @staticmethod
    def _integer(
        values: Mapping[str, Any],
        key: str,
    ) -> int:
        value = values.get(key)

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                f"{key} must be an integer"
            )

        return value

    @staticmethod
    def _number(
        values: Mapping[str, Any],
        key: str,
    ) -> float:
        value = values.get(key)

        if (
            isinstance(value, bool)
            or not isinstance(
                value,
                (int, float),
            )
        ):
            raise TypeError(
                f"{key} must be numeric"
            )

        normalized = float(value)

        if not isfinite(normalized):
            raise ValueError(
                f"{key} must be finite"
            )

        return normalized

    @staticmethod
    def _timestamp(
        values: Mapping[str, Any],
        key: str,
    ) -> datetime:
        value = values.get(key)

        if not isinstance(value, str):
            raise TypeError(
                f"{key} must be a string"
            )

        timestamp = datetime.fromisoformat(
            value
        )

        if timestamp.tzinfo is None:
            raise ValueError(
                f"{key} must be timezone-aware"
            )

        return timestamp
