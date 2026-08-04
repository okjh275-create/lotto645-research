"""Run adaptive automation against a revision-aware repository."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from lrp.evolution.contracts import (
    AdaptiveWeightProfile,
)
from lrp.evolution.feedback.automation import (
    AdaptiveAutomationResult,
    AdaptiveAutomationService,
)
from lrp.evolution.feedback.repository import (
    AdaptiveAutomationRepository,
    AdaptiveAutomationSaveResult,
)


@dataclass(frozen=True, slots=True)
class RevisionAwareAutomationResult:
    """Automation result with repository revision context."""

    repository_revision_before: int | None
    repository_revision_after: int | None
    automation_result: AdaptiveAutomationResult
    save_result: AdaptiveAutomationSaveResult

    def as_dict(self) -> dict[str, Any]:
        return {
            "repository_revision_before": (
                self.repository_revision_before
            ),
            "repository_revision_after": (
                self.repository_revision_after
            ),
            "automation_result": (
                self.automation_result.as_dict()
            ),
            "save_result": (
                self.save_result.as_dict()
            ),
        }


class RevisionAwareAutomationRunner:
    """Validate repository head, run automation, and persist it."""

    def __init__(
        self,
        *,
        service: AdaptiveAutomationService | None = None,
        repository: AdaptiveAutomationRepository,
        allow_empty_repository: bool = True,
    ) -> None:
        if (
            service is not None
            and not isinstance(
                service,
                AdaptiveAutomationService,
            )
        ):
            raise TypeError(
                "service must be an "
                "AdaptiveAutomationService or None"
            )

        if not isinstance(
            repository,
            AdaptiveAutomationRepository,
        ):
            raise TypeError(
                "repository must be an "
                "AdaptiveAutomationRepository"
            )

        if not isinstance(
            allow_empty_repository,
            bool,
        ):
            raise TypeError(
                "allow_empty_repository must "
                "be boolean"
            )

        self._service = (
            service
            if service is not None
            else AdaptiveAutomationService()
        )
        self._repository = repository
        self._allow_empty_repository = (
            allow_empty_repository
        )

    @property
    def service(
        self,
    ) -> AdaptiveAutomationService:
        return self._service

    @property
    def repository(
        self,
    ) -> AdaptiveAutomationRepository:
        return self._repository

    @property
    def allow_empty_repository(
        self,
    ) -> bool:
        return self._allow_empty_repository

    def run(
        self,
        *,
        report: Mapping[str, Any],
        policy_name: str,
        recommendation_id: str,
        current_profile: AdaptiveWeightProfile,
        created_at_utc: datetime | None = None,
        target_confidence: float | None = None,
        target_sample_size: int | None = None,
    ) -> RevisionAwareAutomationResult:
        if not isinstance(
            current_profile,
            AdaptiveWeightProfile,
        ):
            raise TypeError(
                "current_profile must be an "
                "AdaptiveWeightProfile"
            )

        latest = self.repository.latest_profile()

        revision_before = self._validate_head(
            latest=latest,
            current_profile=current_profile,
        )

        automation_result = self.service.run(
            report=report,
            policy_name=policy_name,
            recommendation_id=recommendation_id,
            current_profile=current_profile,
            created_at_utc=created_at_utc,
            target_confidence=target_confidence,
            target_sample_size=target_sample_size,
        )

        save_result = self.repository.save(
            automation_result
        )

        latest_after = (
            self.repository.latest_profile()
        )

        revision_after = (
            self._revision_from_payload(
                latest_after
            )
            if latest_after is not None
            else None
        )

        return RevisionAwareAutomationResult(
            repository_revision_before=(
                revision_before
            ),
            repository_revision_after=(
                revision_after
            ),
            automation_result=automation_result,
            save_result=save_result,
        )

    def _validate_head(
        self,
        *,
        latest: Mapping[str, Any] | None,
        current_profile: AdaptiveWeightProfile,
    ) -> int | None:
        if latest is None:
            if not self.allow_empty_repository:
                raise RuntimeError(
                    "adaptive profile repository "
                    "is empty"
                )

            return None

        repository_revision = (
            self._revision_from_payload(
                latest
            )
        )

        if (
            repository_revision
            != current_profile.revision
        ):
            raise RuntimeError(
                "current profile revision does not "
                "match repository head: "
                f"current={current_profile.revision}, "
                f"repository={repository_revision}"
            )

        stored_profile = latest.get(
            "profile"
        )

        if not isinstance(
            stored_profile,
            Mapping,
        ):
            raise TypeError(
                "repository profile must be an object"
            )

        expected = self._profile_payload(
            current_profile
        )

        for key, expected_value in (
            expected.items()
        ):
            if key not in stored_profile:
                raise ValueError(
                    f"repository profile is missing "
                    f"{key}"
                )

            actual_value = stored_profile[key]

            if isinstance(
                expected_value,
                float,
            ):
                if (
                    isinstance(actual_value, bool)
                    or not isinstance(
                        actual_value,
                        (int, float),
                    )
                    or abs(
                        float(actual_value)
                        - expected_value
                    ) > 1e-12
                ):
                    raise RuntimeError(
                        "current profile does not match "
                        f"repository head field: {key}"
                    )
            elif actual_value != expected_value:
                raise RuntimeError(
                    "current profile does not match "
                    f"repository head field: {key}"
                )

        return repository_revision

    @staticmethod
    def _revision_from_payload(
        payload: Mapping[str, Any],
    ) -> int:
        revision = payload.get(
            "target_revision"
        )

        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
        ):
            raise TypeError(
                "repository target_revision "
                "must be an integer"
            )

        if revision < 0:
            raise ValueError(
                "repository target_revision "
                "must be non-negative"
            )

        return revision

    @staticmethod
    def _profile_payload(
        profile: AdaptiveWeightProfile,
    ) -> dict[str, Any]:
        return {
            "hot_weight": profile.hot_weight,
            "cold_weight": profile.cold_weight,
            "gap_weight": profile.gap_weight,
            "trend_weight": (
                profile.trend_weight
            ),
            "transition_weight": (
                profile.transition_weight
            ),
            "learning_weight": (
                profile.learning_weight
            ),
            "adaptive_weight": (
                profile.adaptive_weight
            ),
            "confidence": profile.confidence,
            "sample_size": profile.sample_size,
            "revision": profile.revision,
            "generated_at": (
                profile.generated_at
                .astimezone(timezone.utc)
                .isoformat()
            ),
        }
