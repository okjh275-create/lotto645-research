"""Read-only production registry reconciliation."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
import hashlib
import json
from pathlib import Path
from typing import Any
from typing import Mapping


_DOMAINS = (
    "active_pair",
    "revision_history",
    "decision_history",
    "rollback_provenance",
    "cross_store_identity",
)

_STATUSES = {
    "PASS",
    "WARN",
    "FAIL",
}

_SEVERITIES = {
    "WARN",
    "FAIL",
}

_DOMAIN_INDEX = {
    domain: index
    for index, domain in enumerate(
        _DOMAINS
    )
}


@dataclass(frozen=True)
class ProductionReconciliationCheck:
    """One reconciliation-domain outcome."""

    name: str
    domain: str
    status: str
    detail: str

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.name,
            str,
        ) or not self.name:
            raise ValueError(
                "name must be a non-empty string"
            )

        if self.domain not in _DOMAINS:
            raise ValueError(
                "unknown reconciliation domain"
            )

        if self.status not in _STATUSES:
            raise ValueError(
                "invalid reconciliation status"
            )

        if not isinstance(
            self.detail,
            str,
        ) or not self.detail:
            raise ValueError(
                "detail must be a non-empty string"
            )


@dataclass(frozen=True)
class ProductionReconciliationIssue:
    """One reconciliation issue."""

    code: str
    domain: str
    severity: str
    message: str
    path: str | None = None

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.code,
            str,
        ) or not self.code:
            raise ValueError(
                "code must be a non-empty string"
            )

        if self.domain not in _DOMAINS:
            raise ValueError(
                "unknown reconciliation domain"
            )

        if self.severity not in _SEVERITIES:
            raise ValueError(
                "invalid reconciliation severity"
            )

        if not isinstance(
            self.message,
            str,
        ) or not self.message:
            raise ValueError(
                "message must be a non-empty string"
            )

        if (
            self.path is not None
            and (
                not isinstance(
                    self.path,
                    str,
                )
                or not self.path
            )
        ):
            raise ValueError(
                "path must be None or a non-empty string"
            )


@dataclass(frozen=True)
class ProductionReconciliationResult:
    """Deterministic reconciliation result."""

    schema_version: int
    generated_at: str
    status: str
    checks: tuple[
        ProductionReconciliationCheck,
        ...,
    ]
    issues: tuple[
        ProductionReconciliationIssue,
        ...,
    ]
    active_model: str | None
    active_source_sha256: str | None
    active_revision_id: str | None
    domains: Mapping[str, object]

    def __post_init__(
        self,
    ) -> None:
        if self.schema_version != 1:
            raise ValueError(
                "schema_version must be 1"
            )

        if not isinstance(
            self.generated_at,
            str,
        ) or not self.generated_at:
            raise ValueError(
                "generated_at must be a non-empty string"
            )

        if self.status not in _STATUSES:
            raise ValueError(
                "invalid reconciliation status"
            )

    def to_payload(
        self,
    ) -> dict[str, object]:
        return {
            "schema_version":
                self.schema_version,

            "generated_at":
                self.generated_at,

            "status":
                self.status,

            "checks": [
                asdict(check)
                for check in self.checks
            ],

            "issues": [
                asdict(issue)
                for issue in self.issues
            ],

            "active_model":
                self.active_model,

            "active_source_sha256":
                self.active_source_sha256,

            "active_revision_id":
                self.active_revision_id,

            "domains": {
                key: value
                for key, value
                in self.domains.items()
            },
        }

    def as_dict(
        self,
    ) -> dict[str, object]:
        return self.to_payload()


class ProductionReconciliationService:
    """Reconcile persisted production registry identities."""

    def __init__(
        self,
        *,
        registry_root: str | Path,
    ) -> None:
        self._registry_root = Path(
            registry_root
        )

    def reconcile(
        self,
    ) -> ProductionReconciliationResult:
        issues: list[
            ProductionReconciliationIssue
        ] = []

        active_decision_path = (
            self._registry_root
            / "active"
            / "champion_decision.json"
        )

        active_publication_path = (
            self._registry_root
            / "active"
            / "publication.json"
        )

        decision_payload: dict[
            str,
            Any,
        ] | None = None

        publication_payload: dict[
            str,
            Any,
        ] | None = None

        decision_raw: bytes | None = None

        active_model: str | None = None
        active_source_sha256: str | None = None
        active_revision_id: str | None = None

        active_pair_status = "PASS"

        if not active_decision_path.is_file():
            active_pair_status = "FAIL"

            issues.append(
                self._issue(
                    code="ACTIVE_PAIR_MISSING",
                    domain="active_pair",
                    severity="FAIL",
                    message=(
                        "active champion decision "
                        "is missing"
                    ),
                    path=self._relative(
                        active_decision_path
                    ),
                )
            )

        else:
            try:
                decision_raw = (
                    active_decision_path
                    .read_bytes()
                )

                decision_payload = (
                    self._decode_object(
                        decision_raw
                    )
                )

                selection = (
                    decision_payload
                    .get(
                        "selection"
                    )
                )

                if isinstance(
                    selection,
                    dict,
                ):
                    selected = (
                        selection
                        .get(
                            "selected_model"
                        )
                    )

                    if isinstance(
                        selected,
                        str,
                    ):
                        active_model = selected

            except Exception:
                active_pair_status = "FAIL"

                issues.append(
                    self._issue(
                        code="ACTIVE_PAIR_MISMATCH",
                        domain="active_pair",
                        severity="FAIL",
                        message=(
                            "active champion decision "
                            "is malformed"
                        ),
                        path=self._relative(
                            active_decision_path
                        ),
                    )
                )

        if not active_publication_path.is_file():
            active_pair_status = "FAIL"

            issues.append(
                self._issue(
                    code="ACTIVE_PAIR_MISSING",
                    domain="active_pair",
                    severity="FAIL",
                    message=(
                        "active publication "
                        "is missing"
                    ),
                    path=self._relative(
                        active_publication_path
                    ),
                )
            )

        else:
            try:
                publication_payload = (
                    self._decode_object(
                        active_publication_path
                        .read_bytes()
                    )
                )

                value = (
                    publication_payload
                    .get(
                        "source_sha256"
                    )
                )

                if isinstance(
                    value,
                    str,
                ):
                    active_source_sha256 = value

                value = (
                    publication_payload
                    .get(
                        "revision_id"
                    )
                )

                if isinstance(
                    value,
                    str,
                ):
                    active_revision_id = value

                publication_model = (
                    publication_payload
                    .get(
                        "selected_model"
                    )
                )

                if (
                    active_model is None
                    and isinstance(
                        publication_model,
                        str,
                    )
                ):
                    active_model = (
                        publication_model
                    )

            except Exception:
                active_pair_status = "FAIL"

                issues.append(
                    self._issue(
                        code="ACTIVE_PAIR_MISMATCH",
                        domain="active_pair",
                        severity="FAIL",
                        message=(
                            "active publication "
                            "is malformed"
                        ),
                        path=self._relative(
                            active_publication_path
                        ),
                    )
                )

        if (
            decision_raw is not None
            and publication_payload
            is not None
        ):
            actual_source_sha = (
                hashlib.sha256(
                    decision_raw
                ).hexdigest()
            )

            publication_source_sha = (
                publication_payload
                .get(
                    "source_sha256"
                )
            )

            publication_model = (
                publication_payload
                .get(
                    "selected_model"
                )
            )

            if (
                publication_source_sha
                != actual_source_sha
            ):
                active_pair_status = "FAIL"

                issues.append(
                    self._issue(
                        code="ACTIVE_PAIR_MISMATCH",
                        domain="active_pair",
                        severity="FAIL",
                        message=(
                            "active publication source "
                            "hash does not match "
                            "active decision bytes"
                        ),
                        path=self._relative(
                            active_publication_path
                        ),
                    )
                )

            if (
                active_model is not None
                and isinstance(
                    publication_model,
                    str,
                )
                and publication_model
                != active_model
            ):
                active_pair_status = "FAIL"

                issues.append(
                    self._issue(
                        code="ACTIVE_PAIR_MISMATCH",
                        domain="active_pair",
                        severity="FAIL",
                        message=(
                            "active publication model "
                            "does not match active "
                            "decision model"
                        ),
                        path=self._relative(
                            active_publication_path
                        ),
                    )
                )

        revision_status = "PASS"
        active_revision_payload: dict[
            str,
            Any,
        ] | None = None

        history_root = (
            self._registry_root
            / "history"
        )

        active_revision_path: Path | None = None

        if active_revision_id:
            active_revision_path = (
                history_root
                / f"{active_revision_id}.json"
            )

            if not active_revision_path.is_file():
                revision_status = "FAIL"

                issues.append(
                    self._issue(
                        code="REVISION_MISSING",
                        domain="revision_history",
                        severity="FAIL",
                        message=(
                            "active revision history "
                            "record is missing"
                        ),
                        path=self._relative(
                            active_revision_path
                        ),
                    )
                )

            else:
                try:
                    active_revision_payload = (
                        self._decode_object(
                            active_revision_path
                            .read_bytes()
                        )
                    )

                except Exception:
                    revision_status = "FAIL"

                    issues.append(
                        self._issue(
                            code=(
                                "REVISION_HASH_MISMATCH"
                            ),
                            domain=(
                                "revision_history"
                            ),
                            severity="FAIL",
                            message=(
                                "active revision "
                                "record is malformed"
                            ),
                            path=self._relative(
                                active_revision_path
                            ),
                        )
                    )

                if (
                    active_revision_payload
                    is not None
                    and active_source_sha256
                    is not None
                    and active_revision_payload
                    .get(
                        "source_sha256"
                    )
                    != active_source_sha256
                ):
                    revision_status = "FAIL"

                    issues.append(
                        self._issue(
                            code=(
                                "REVISION_HASH_MISMATCH"
                            ),
                            domain=(
                                "revision_history"
                            ),
                            severity="FAIL",
                            message=(
                                "active revision source "
                                "hash does not match "
                                "active publication"
                            ),
                            path=self._relative(
                                active_revision_path
                            ),
                        )
                    )

        elif publication_payload is not None:
            revision_status = "FAIL"

            issues.append(
                self._issue(
                    code="REVISION_MISSING",
                    domain="revision_history",
                    severity="FAIL",
                    message=(
                        "active publication has no "
                        "revision identity"
                    ),
                    path=self._relative(
                        active_publication_path
                    ),
                )
            )

        if history_root.is_dir():
            for path in sorted(
                history_root.glob(
                    "*.json"
                ),
                key=lambda item:
                    item.as_posix(),
            ):
                if (
                    active_revision_path
                    is not None
                    and path
                    == active_revision_path
                ):
                    continue

                if revision_status != "FAIL":
                    revision_status = "WARN"

                issues.append(
                    self._issue(
                        code="ORPHAN_REVISION",
                        domain="revision_history",
                        severity="WARN",
                        message=(
                            "revision record is not "
                            "the active revision"
                        ),
                        path=self._relative(
                            path
                        ),
                    )
                )

        decision_history_status = "PASS"

        decisions_root = (
            history_root
            / "decisions"
        )

        active_decision_history_path: (
            Path | None
        ) = None

        if active_source_sha256:
            active_decision_history_path = (
                decisions_root
                / (
                    active_source_sha256
                    + ".json"
                )
            )

            if (
                not active_decision_history_path
                .is_file()
            ):
                decision_history_status = "FAIL"

                issues.append(
                    self._issue(
                        code=(
                            "DECISION_HISTORY_MISSING"
                        ),
                        domain="decision_history",
                        severity="FAIL",
                        message=(
                            "active decision history "
                            "record is missing"
                        ),
                        path=self._relative(
                            active_decision_history_path
                        ),
                    )
                )

            else:
                history_raw = (
                    active_decision_history_path
                    .read_bytes()
                )

                history_sha = (
                    hashlib.sha256(
                        history_raw
                    ).hexdigest()
                )

                if (
                    history_sha
                    != active_source_sha256
                ):
                    decision_history_status = (
                        "FAIL"
                    )

                    issues.append(
                        self._issue(
                            code=(
                                "DECISION_HISTORY_"
                                "HASH_MISMATCH"
                            ),
                            domain=(
                                "decision_history"
                            ),
                            severity="FAIL",
                            message=(
                                "active decision "
                                "history hash does "
                                "not match identity"
                            ),
                            path=self._relative(
                                active_decision_history_path
                            ),
                        )
                    )

        elif publication_payload is not None:
            decision_history_status = "FAIL"

            issues.append(
                self._issue(
                    code=(
                        "DECISION_HISTORY_MISSING"
                    ),
                    domain="decision_history",
                    severity="FAIL",
                    message=(
                        "active source identity "
                        "is unavailable"
                    ),
                    path=None,
                )
            )

        if decisions_root.is_dir():
            for path in sorted(
                decisions_root.glob(
                    "*.json"
                ),
                key=lambda item:
                    item.as_posix(),
            ):
                if (
                    active_decision_history_path
                    is not None
                    and path
                    == active_decision_history_path
                ):
                    continue

                if decision_history_status != "FAIL":
                    decision_history_status = "WARN"

                issues.append(
                    self._issue(
                        code="ORPHAN_DECISION",
                        domain="decision_history",
                        severity="WARN",
                        message=(
                            "decision history record "
                            "is not referenced by "
                            "the active publication"
                        ),
                        path=self._relative(
                            path
                        ),
                    )
                )

        rollback_status = "PASS"

        rollbacks_root = (
            history_root
            / "rollbacks"
        )

        if rollbacks_root.is_dir():
            for path in sorted(
                rollbacks_root.glob(
                    "*.json"
                ),
                key=lambda item:
                    item.as_posix(),
            ):
                try:
                    payload = (
                        self._decode_object(
                            path.read_bytes()
                        )
                    )

                except Exception:
                    if rollback_status != "FAIL":
                        rollback_status = "WARN"

                    issues.append(
                        self._issue(
                            code=(
                                "ROLLBACK_RECORD_"
                                "MALFORMED"
                            ),
                            domain=(
                                "rollback_provenance"
                            ),
                            severity="WARN",
                            message=(
                                "rollback provenance "
                                "record is malformed"
                            ),
                            path=self._relative(
                                path
                            ),
                        )
                    )

                    continue

                source_revision_id = (
                    payload.get(
                        "source_revision_id"
                    )
                )

                target_revision_id = (
                    payload.get(
                        "target_revision_id"
                    )
                )

                target_source_sha256 = (
                    payload.get(
                        "target_source_sha256"
                    )
                )

                if not isinstance(
                    source_revision_id,
                    str,
                ) or not source_revision_id:
                    if rollback_status != "FAIL":
                        rollback_status = "WARN"

                    issues.append(
                        self._issue(
                            code="ORPHAN_ROLLBACK",
                            domain=(
                                "rollback_provenance"
                            ),
                            severity="WARN",
                            message=(
                                "rollback provenance "
                                "has no source "
                                "revision identity"
                            ),
                            path=self._relative(
                                path
                            ),
                        )
                    )

                    continue

                if not isinstance(
                    target_revision_id,
                    str,
                ) or not target_revision_id:
                    rollback_status = "FAIL"

                    issues.append(
                        self._issue(
                            code=(
                                "ROLLBACK_TARGET_MISSING"
                            ),
                            domain=(
                                "rollback_provenance"
                            ),
                            severity="FAIL",
                            message=(
                                "rollback target "
                                "revision is missing"
                            ),
                            path=self._relative(
                                path
                            ),
                        )
                    )

                    continue

                target_path = (
                    history_root
                    / (
                        target_revision_id
                        + ".json"
                    )
                )

                if not target_path.is_file():
                    rollback_status = "FAIL"

                    issues.append(
                        self._issue(
                            code=(
                                "ROLLBACK_TARGET_MISSING"
                            ),
                            domain=(
                                "rollback_provenance"
                            ),
                            severity="FAIL",
                            message=(
                                "rollback target "
                                "revision does not exist"
                            ),
                            path=self._relative(
                                path
                            ),
                        )
                    )

                    continue

                try:
                    target_payload = (
                        self._decode_object(
                            target_path.read_bytes()
                        )
                    )

                except Exception:
                    rollback_status = "FAIL"

                    issues.append(
                        self._issue(
                            code=(
                                "ROLLBACK_HASH_MISMATCH"
                            ),
                            domain=(
                                "rollback_provenance"
                            ),
                            severity="FAIL",
                            message=(
                                "rollback target "
                                "revision is malformed"
                            ),
                            path=self._relative(
                                path
                            ),
                        )
                    )

                    continue

                if (
                    isinstance(
                        target_source_sha256,
                        str,
                    )
                    and target_payload.get(
                        "source_sha256"
                    )
                    != target_source_sha256
                ):
                    rollback_status = "FAIL"

                    issues.append(
                        self._issue(
                            code=(
                                "ROLLBACK_HASH_MISMATCH"
                            ),
                            domain=(
                                "rollback_provenance"
                            ),
                            severity="FAIL",
                            message=(
                                "rollback target source "
                                "hash does not match "
                                "target revision"
                            ),
                            path=self._relative(
                                path
                            ),
                        )
                    )

        cross_store_status = "PASS"

        if (
            active_revision_payload
            is not None
        ):
            revision_model = (
                active_revision_payload
                .get(
                    "selected_model"
                )
            )

            if (
                active_model is not None
                and isinstance(
                    revision_model,
                    str,
                )
                and revision_model
                != active_model
            ):
                cross_store_status = "FAIL"

                issues.append(
                    self._issue(
                        code=(
                            "CROSS_STORE_IDENTITY_"
                            "MISMATCH"
                        ),
                        domain=(
                            "cross_store_identity"
                        ),
                        severity="FAIL",
                        message=(
                            "active model does not "
                            "match revision-history "
                            "model"
                        ),
                        path=(
                            self._relative(
                                active_revision_path
                            )
                            if active_revision_path
                            is not None
                            else None
                        ),
                    )
                )

            revision_source = (
                active_revision_payload
                .get(
                    "source_sha256"
                )
            )

            if (
                active_source_sha256
                is not None
                and isinstance(
                    revision_source,
                    str,
                )
                and revision_source
                != active_source_sha256
            ):
                cross_store_status = "FAIL"

                issues.append(
                    self._issue(
                        code=(
                            "CROSS_STORE_IDENTITY_"
                            "MISMATCH"
                        ),
                        domain=(
                            "cross_store_identity"
                        ),
                        severity="FAIL",
                        message=(
                            "publication and revision "
                            "source identities differ"
                        ),
                        path=(
                            self._relative(
                                active_revision_path
                            )
                            if active_revision_path
                            is not None
                            else None
                        ),
                    )
                )

        checks = (
            ProductionReconciliationCheck(
                name="active_pair",
                domain="active_pair",
                status=active_pair_status,
                detail=(
                    "active decision and publication "
                    "identity reconciliation"
                ),
            ),
            ProductionReconciliationCheck(
                name="revision_history",
                domain="revision_history",
                status=revision_status,
                detail=(
                    "revision history identity "
                    "reconciliation"
                ),
            ),
            ProductionReconciliationCheck(
                name="decision_history",
                domain="decision_history",
                status=decision_history_status,
                detail=(
                    "decision history identity "
                    "reconciliation"
                ),
            ),
            ProductionReconciliationCheck(
                name="rollback_provenance",
                domain="rollback_provenance",
                status=rollback_status,
                detail=(
                    "rollback provenance "
                    "reconciliation"
                ),
            ),
            ProductionReconciliationCheck(
                name="cross_store_identity",
                domain="cross_store_identity",
                status=cross_store_status,
                detail=(
                    "cross-store identity "
                    "reconciliation"
                ),
            ),
        )

        ordered_issues = tuple(
            sorted(
                issues,
                key=lambda issue: (
                    _DOMAIN_INDEX[
                        issue.domain
                    ],
                    issue.code,
                    issue.path or "",
                ),
            )
        )

        status = self._aggregate_status(
            tuple(
                check.status
                for check in checks
            )
        )

        domains = {
            check.domain: {
                "status":
                    check.status,
            }
            for check in checks
        }

        return ProductionReconciliationResult(
            schema_version=1,
            generated_at=(
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            status=status,
            checks=checks,
            issues=ordered_issues,
            active_model=active_model,
            active_source_sha256=(
                active_source_sha256
            ),
            active_revision_id=(
                active_revision_id
            ),
            domains=domains,
        )

    @staticmethod
    def _aggregate_status(
        statuses: tuple[str, ...],
    ) -> str:
        if "FAIL" in statuses:
            return "FAIL"

        if "WARN" in statuses:
            return "WARN"

        return "PASS"

    @staticmethod
    def _decode_object(
        raw: bytes,
    ) -> dict[str, Any]:
        payload = json.loads(
            raw.decode(
                "utf-8"
            )
        )

        if not isinstance(
            payload,
            dict,
        ):
            raise ValueError(
                "JSON payload must be an object"
            )

        return payload

    def _relative(
        self,
        path: Path,
    ) -> str:
        try:
            return (
                path.relative_to(
                    self._registry_root
                )
                .as_posix()
            )

        except ValueError:
            return path.as_posix()

    @staticmethod
    def _issue(
        *,
        code: str,
        domain: str,
        severity: str,
        message: str,
        path: str | None,
    ) -> ProductionReconciliationIssue:
        return ProductionReconciliationIssue(
            code=code,
            domain=domain,
            severity=severity,
            message=message,
            path=path,
        )
