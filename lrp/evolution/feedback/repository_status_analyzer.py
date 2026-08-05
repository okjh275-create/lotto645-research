"""Analyze adaptive automation repository health."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lrp.evolution.feedback.repository import (
    AdaptiveAutomationRepository,
)
from lrp.evolution.feedback.status import (
    AdaptiveStatusIssue,
    AdaptiveStatusReport,
)


class AdaptiveRepositoryStatusAnalyzer:
    """Inspect repository revisions and automation records."""

    WEIGHT_FIELDS = (
        "hot_weight",
        "cold_weight",
        "gap_weight",
        "trend_weight",
        "transition_weight",
        "learning_weight",
        "adaptive_weight",
    )

    def analyze(
        self,
        repository: AdaptiveAutomationRepository,
    ) -> AdaptiveStatusReport:
        if not isinstance(
            repository,
            AdaptiveAutomationRepository,
        ):
            raise TypeError(
                "repository must be an "
                "AdaptiveAutomationRepository"
            )

        issues: list[AdaptiveStatusIssue] = []

        revisions = (
            repository.list_profile_revisions()
        )
        automation_ids = (
            repository.list_automation_ids()
        )

        rollback_count = 0
        valid_revisions: list[int] = []

        for revision in revisions:
            try:
                payload = (
                    repository
                    .load_profile_revision(
                        revision
                    )
                )
                self._validate_profile_record(
                    payload=payload,
                    revision=revision,
                    issues=issues,
                )

                if payload.get(
                    "record_type"
                ) == "rollback":
                    rollback_count += 1

                valid_revisions.append(
                    revision
                )
            except (
                FileNotFoundError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ) as exc:
                issues.append(
                    AdaptiveStatusIssue(
                        category="profile",
                        severity="error",
                        message=(
                            f"revision {revision}: "
                            f"{exc}"
                        ),
                    )
                )

        self._check_revision_continuity(
            valid_revisions,
            issues,
        )

        valid_recommendations = 0

        for recommendation_id in automation_ids:
            try:
                payload = (
                    repository.load_automation(
                        recommendation_id
                    )
                )
                self._validate_automation_record(
                    payload=payload,
                    recommendation_id=(
                        recommendation_id
                    ),
                    issues=issues,
                )
                valid_recommendations += 1
            except (
                FileNotFoundError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ) as exc:
                issues.append(
                    AdaptiveStatusIssue(
                        category="automation",
                        severity="error",
                        message=(
                            f"{recommendation_id}: "
                            f"{exc}"
                        ),
                    )
                )

        latest_revision = (
            valid_revisions[-1]
            if valid_revisions
            else 0
        )

        repository_ok = not any(
            issue.severity == "error"
            and issue.category
            in {
                "repository",
                "revision",
                "automation",
            }
            for issue in issues
        )

        profile_ok = bool(
            valid_revisions
        ) and not any(
            issue.severity == "error"
            and issue.category == "profile"
            for issue in issues
        )

        return AdaptiveStatusReport(
            latest_revision=latest_revision,
            repository_ok=repository_ok,
            profile_ok=profile_ok,
            validation_ok=True,
            rollback_count=rollback_count,
            recommendation_count=(
                valid_recommendations
            ),
            issues=tuple(issues),
        )

    def _validate_profile_record(
        self,
        *,
        payload: Mapping[str, Any],
        revision: int,
        issues: list[AdaptiveStatusIssue],
    ) -> None:
        target_revision = self._integer(
            payload,
            "target_revision",
        )

        if target_revision != revision:
            issues.append(
                AdaptiveStatusIssue(
                    category="profile",
                    severity="error",
                    message=(
                        f"revision filename {revision} "
                        "does not match "
                        f"target_revision "
                        f"{target_revision}"
                    ),
                )
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

        if profile_revision != revision:
            issues.append(
                AdaptiveStatusIssue(
                    category="profile",
                    severity="error",
                    message=(
                        f"profile revision "
                        f"{profile_revision} does not "
                        f"match filename revision "
                        f"{revision}"
                    ),
                )
            )

        weights = [
            self._number(
                profile,
                field,
            )
            for field in self.WEIGHT_FIELDS
        ]

        if any(
            value < 0.0
            for value in weights
        ):
            issues.append(
                AdaptiveStatusIssue(
                    category="profile",
                    severity="error",
                    message=(
                        f"revision {revision} contains "
                        "a negative weight"
                    ),
                )
            )

        total = sum(weights)

        if abs(total - 1.0) > 1e-9:
            issues.append(
                AdaptiveStatusIssue(
                    category="profile",
                    severity="error",
                    message=(
                        f"revision {revision} weights "
                        f"sum to {total}"
                    ),
                )
            )

        confidence = self._number(
            profile,
            "confidence",
        )

        if not 0.0 <= confidence <= 1.0:
            issues.append(
                AdaptiveStatusIssue(
                    category="profile",
                    severity="error",
                    message=(
                        f"revision {revision} "
                        "confidence is outside "
                        "0.0 to 1.0"
                    ),
                )
            )

        sample_size = self._integer(
            profile,
            "sample_size",
        )

        if sample_size < 0:
            issues.append(
                AdaptiveStatusIssue(
                    category="profile",
                    severity="error",
                    message=(
                        f"revision {revision} "
                        "sample_size is negative"
                    ),
                )
            )

        generated_at = profile.get(
            "generated_at"
        )

        if not isinstance(
            generated_at,
            str,
        ) or not generated_at.strip():
            issues.append(
                AdaptiveStatusIssue(
                    category="profile",
                    severity="error",
                    message=(
                        f"revision {revision} "
                        "generated_at is missing"
                    ),
                )
            )

        if payload.get(
            "record_type"
        ) == "rollback":
            self._validate_rollback_record(
                payload=payload,
                revision=revision,
                issues=issues,
            )

    def _validate_rollback_record(
        self,
        *,
        payload: Mapping[str, Any],
        revision: int,
        issues: list[AdaptiveStatusIssue],
    ) -> None:
        source_revision = self._integer(
            payload,
            "source_revision",
        )
        rollback_revision = self._integer(
            payload,
            "rollback_revision",
        )

        if source_revision + 1 != revision:
            issues.append(
                AdaptiveStatusIssue(
                    category="profile",
                    severity="error",
                    message=(
                        f"rollback revision {revision} "
                        "does not follow source "
                        f"revision {source_revision}"
                    ),
                )
            )

        if rollback_revision >= source_revision:
            issues.append(
                AdaptiveStatusIssue(
                    category="profile",
                    severity="error",
                    message=(
                        f"rollback target "
                        f"{rollback_revision} must be "
                        "older than source revision "
                        f"{source_revision}"
                    ),
                )
            )

    @staticmethod
    def _validate_automation_record(
        *,
        payload: Mapping[str, Any],
        recommendation_id: str,
        issues: list[AdaptiveStatusIssue],
    ) -> None:
        recommendation = payload.get(
            "recommendation"
        )

        if not isinstance(
            recommendation,
            Mapping,
        ):
            raise TypeError(
                "recommendation must be an object"
            )

        stored_id = recommendation.get(
            "recommendation_id"
        )

        if stored_id != recommendation_id:
            issues.append(
                AdaptiveStatusIssue(
                    category="automation",
                    severity="error",
                    message=(
                        f"automation filename "
                        f"{recommendation_id} does not "
                        "match stored recommendation "
                        f"{stored_id}"
                    ),
                )
            )

        update_plan = payload.get(
            "update_plan"
        )

        if not isinstance(
            update_plan,
            Mapping,
        ):
            raise TypeError(
                "update_plan must be an object"
            )

        approved = update_plan.get(
            "approved"
        )

        if not isinstance(
            approved,
            bool,
        ):
            raise TypeError(
                "update_plan approved must "
                "be boolean"
            )

    @staticmethod
    def _check_revision_continuity(
        revisions: list[int],
        issues: list[AdaptiveStatusIssue],
    ) -> None:
        for previous, current in zip(
            revisions,
            revisions[1:],
        ):
            if current != previous + 1:
                issues.append(
                    AdaptiveStatusIssue(
                        category="revision",
                        severity="error",
                        message=(
                            "revision gap detected: "
                            f"{previous} to {current}"
                        ),
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

        return float(value)
