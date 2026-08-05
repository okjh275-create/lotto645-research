"""Contracts for adaptive automation status reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AdaptiveStatusIssue:
    category: str
    severity: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveStatusReport:
    latest_revision: int
    repository_ok: bool
    profile_ok: bool
    validation_ok: bool
    rollback_count: int
    recommendation_count: int
    issues: tuple[
        AdaptiveStatusIssue,
        ...
    ] = field(default_factory=tuple)

    @property
    def overall_ok(self) -> bool:
        return (
            self.repository_ok
            and self.profile_ok
            and self.validation_ok
            and not self.issues
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "latest_revision":
                self.latest_revision,
            "repository_ok":
                self.repository_ok,
            "profile_ok":
                self.profile_ok,
            "validation_ok":
                self.validation_ok,
            "rollback_count":
                self.rollback_count,
            "recommendation_count":
                self.recommendation_count,
            "overall_ok":
                self.overall_ok,
            "issues": [
                item.as_dict()
                for item in self.issues
            ],
        }
