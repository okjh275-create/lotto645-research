from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
)
from datetime import (
    datetime,
    timezone,
)
import json
from pathlib import Path
from typing import Any

from lrp.production.champion_active_snapshot import (
    ProductionChampionActiveSnapshotReader,
)
from lrp.production.champion_audit import (
    ProductionChampionAudit,
)
from lrp.production.champion_history_retention import (
    ChampionHistoryRetentionAtomicityError,
    ChampionHistoryRetentionExecutor,
)
from lrp.production.champion_registry_recovery import (
    ProductionRegistryBackupService,
    ProductionRegistryRestoreAtomicityError,
    ProductionRegistryRestoreService,
)
from lrp.production.champion_rollback import (
    ChampionRollbackService,
)
from lrp.production.champion_rollback_history import (
    ChampionRollbackHistoryReader,
)
from lrp.production.production_lifecycle import (
    ProductionLifecycleRequest,
    ProductionLifecycleResult,
    ProductionLifecycleService,
    ProductionLifecycleStageResult,
)


_STATUS_VALUES = {
    "PASS",
    "WARN",
    "FAIL",
}

_ISSUE_SEVERITIES = {
    "WARN",
    "FAIL",
}

_DOMAINS = (
    "active_champion",
    "writer_lock",
    "recovery_readiness",
    "history_safety",
    "lifecycle_readiness",
)

_DOMAIN_INDEX = {
    domain: index
    for index, domain
    in enumerate(_DOMAINS)
}


def _require_non_empty(
    value: str,
    *,
    field: str,
) -> None:
    if (
        not isinstance(
            value,
            str,
        )
        or not value.strip()
    ):
        raise ValueError(
            f"{field} must be a non-empty string"
        )


def _require_domain(
    value: str,
) -> None:
    if value not in _DOMAINS:
        raise ValueError(
            "unknown production health domain: "
            f"{value}"
        )


@dataclass(
    frozen=True,
)
class ProductionHealthCheck:
    name: str
    domain: str
    status: str
    detail: str

    def __post_init__(
        self,
    ) -> None:
        _require_non_empty(
            self.name,
            field="name",
        )

        _require_domain(
            self.domain
        )

        if (
            self.status
            not in _STATUS_VALUES
        ):
            raise ValueError(
                "invalid health check status"
            )

        if not isinstance(
            self.detail,
            str,
        ):
            raise TypeError(
                "detail must be a string"
            )

    def to_payload(
        self,
    ) -> dict[str, object]:
        return {
            "name":
                self.name,
            "domain":
                self.domain,
            "status":
                self.status,
            "detail":
                self.detail,
        }


@dataclass(
    frozen=True,
)
class ProductionHealthIssue:
    code: str
    domain: str
    severity: str
    message: str

    def __post_init__(
        self,
    ) -> None:
        _require_non_empty(
            self.code,
            field="code",
        )

        _require_domain(
            self.domain
        )

        if (
            self.severity
            not in _ISSUE_SEVERITIES
        ):
            raise ValueError(
                "invalid health issue severity"
            )

        _require_non_empty(
            self.message,
            field="message",
        )

    def to_payload(
        self,
    ) -> dict[str, object]:
        return {
            "code":
                self.code,
            "domain":
                self.domain,
            "severity":
                self.severity,
            "message":
                self.message,
        }


@dataclass(
    frozen=True,
)
class ProductionHealthSnapshot:
    schema_version: int
    generated_at: str
    status: str
    checks: tuple[
        ProductionHealthCheck,
        ...,
    ]
    issues: tuple[
        ProductionHealthIssue,
        ...,
    ]
    active_model: str | None
    active_revision_id: str | None
    domains: dict[
        str,
        object,
    ]

    def __post_init__(
        self,
    ) -> None:
        if (
            isinstance(
                self.schema_version,
                bool,
            )
            or self.schema_version != 1
        ):
            raise ValueError(
                "schema_version must be 1"
            )

        _require_non_empty(
            self.generated_at,
            field="generated_at",
        )

        if (
            self.status
            not in _STATUS_VALUES
        ):
            raise ValueError(
                "invalid snapshot status"
            )

        if not isinstance(
            self.checks,
            tuple,
        ):
            raise TypeError(
                "checks must be a tuple"
            )

        if not isinstance(
            self.issues,
            tuple,
        ):
            raise TypeError(
                "issues must be a tuple"
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
                item.to_payload()
                for item in self.checks
            ],
            "issues": [
                item.to_payload()
                for item in self.issues
            ],
            "active_model":
                self.active_model,
            "active_revision_id":
                self.active_revision_id,
            "domains":
                self.domains,
        }


class ProductionHealthService:
    def __init__(
        self,
        *,
        registry_root: str | Path,
        snapshot_root: str | Path,
    ) -> None:
        self._registry_root = Path(
            registry_root
        )

        self._snapshot_root = Path(
            snapshot_root
        )

    def snapshot(
        self,
    ) -> ProductionHealthSnapshot:
        checks: list[
            ProductionHealthCheck
        ] = []

        issues: list[
            ProductionHealthIssue
        ] = []

        domains: dict[
            str,
            object,
        ] = {}

        (
            active_model,
            active_revision_id,
            active_domain,
            active_check,
            active_issues,
        ) = self._active_champion_domain()

        domains[
            "active_champion"
        ] = active_domain

        checks.append(
            active_check
        )

        issues.extend(
            active_issues
        )

        (
            writer_domain,
            writer_check,
            writer_issues,
        ) = self._writer_lock_domain()

        domains[
            "writer_lock"
        ] = writer_domain

        checks.append(
            writer_check
        )

        issues.extend(
            writer_issues
        )

        (
            recovery_domain,
            recovery_check,
            recovery_issues,
        ) = self._recovery_domain()

        domains[
            "recovery_readiness"
        ] = recovery_domain

        checks.append(
            recovery_check
        )

        issues.extend(
            recovery_issues
        )

        (
            history_domain,
            history_check,
            history_issues,
        ) = self._history_domain()

        domains[
            "history_safety"
        ] = history_domain

        checks.append(
            history_check
        )

        issues.extend(
            history_issues
        )

        (
            lifecycle_domain,
            lifecycle_check,
            lifecycle_issues,
        ) = self._lifecycle_domain()

        domains[
            "lifecycle_readiness"
        ] = lifecycle_domain

        checks.append(
            lifecycle_check
        )

        issues.extend(
            lifecycle_issues
        )

        ordered_checks = tuple(
            sorted(
                checks,
                key=lambda item: (
                    _DOMAIN_INDEX[
                        item.domain
                    ],
                    item.name,
                ),
            )
        )

        ordered_issues = tuple(
            sorted(
                issues,
                key=lambda item: (
                    _DOMAIN_INDEX[
                        item.domain
                    ],
                    item.code,
                ),
            )
        )

        status = self._aggregate_status(
            [
                item.status
                for item
                in ordered_checks
            ]
        )

        return ProductionHealthSnapshot(
            schema_version=1,
            generated_at=(
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            status=status,
            checks=ordered_checks,
            issues=ordered_issues,
            active_model=active_model,
            active_revision_id=(
                active_revision_id
            ),
            domains=domains,
        )

    def _active_champion_domain(
        self,
    ) -> tuple[
        str | None,
        str | None,
        dict[str, object],
        ProductionHealthCheck,
        tuple[
            ProductionHealthIssue,
            ...,
        ],
    ]:
        try:
            audit = (
                ProductionChampionAudit()
                .audit(
                    registry_root=(
                        self._registry_root
                    ),
                    snapshot_root=(
                        self._snapshot_root
                    ),
                )
            )

            audit_status = audit.status

            active_model = getattr(
                audit,
                "selected_model",
                None,
            )

            active_revision_id = None

            try:
                active = (
                    ProductionChampionActiveSnapshotReader()
                    .read(
                        self._registry_root
                    )
                )

                active_model = (
                    getattr(
                        active,
                        "selected_model",
                        active_model,
                    )
                )

                active_revision_id = (
                    getattr(
                        active,
                        "revision_id",
                        None,
                    )
                )

            except Exception:
                if audit_status == "PASS":
                    audit_status = "FAIL"

            if audit_status not in {
                "PASS",
                "WARN",
                "FAIL",
            }:
                audit_status = "FAIL"

            domain = {
                "status":
                    audit_status,
                "audit_status":
                    getattr(
                        audit,
                        "status",
                        "FAIL",
                    ),
            }

            issue_list: list[
                ProductionHealthIssue
            ] = []

            if audit_status == "WARN":
                issue_list.append(
                    ProductionHealthIssue(
                        code=(
                            "ACTIVE_CHAMPION_WARN"
                        ),
                        domain=(
                            "active_champion"
                        ),
                        severity="WARN",
                        message=(
                            "active champion "
                            "audit reported warning"
                        ),
                    )
                )

            elif audit_status == "FAIL":
                issue_list.append(
                    ProductionHealthIssue(
                        code=(
                            "ACTIVE_CHAMPION_FAIL"
                        ),
                        domain=(
                            "active_champion"
                        ),
                        severity="FAIL",
                        message=(
                            "active champion "
                            "health check failed"
                        ),
                    )
                )

            return (
                active_model,
                active_revision_id,
                domain,
                ProductionHealthCheck(
                    name="active_champion",
                    domain="active_champion",
                    status=audit_status,
                    detail=(
                        "active champion "
                        f"status={audit_status}"
                    ),
                ),
                tuple(
                    issue_list
                ),
            )

        except Exception as exc:
            return (
                None,
                None,
                {
                    "status":
                        "FAIL",
                    "error_type":
                        type(exc).__name__,
                },
                ProductionHealthCheck(
                    name="active_champion",
                    domain="active_champion",
                    status="FAIL",
                    detail=(
                        "active champion "
                        "health unavailable"
                    ),
                ),
                (
                    ProductionHealthIssue(
                        code=(
                            "ACTIVE_CHAMPION_FAIL"
                        ),
                        domain=(
                            "active_champion"
                        ),
                        severity="FAIL",
                        message=(
                            "active champion "
                            "health unavailable"
                        ),
                    ),
                ),
            )

    def _writer_lock_domain(
        self,
    ) -> tuple[
        dict[str, object],
        ProductionHealthCheck,
        tuple[
            ProductionHealthIssue,
            ...,
        ],
    ]:
        path = (
            self._registry_root
            / ".writer.lock"
        )

        if not path.exists():
            return (
                {
                    "status":
                        "PASS",
                    "present":
                        False,
                },
                ProductionHealthCheck(
                    name="writer_lock",
                    domain="writer_lock",
                    status="PASS",
                    detail=(
                        "writer lock sidecar absent"
                    ),
                ),
                (),
            )

        try:
            raw_text = path.read_text(
                encoding="utf-8"
            )

            payload = json.loads(
                raw_text.lstrip(
                    "\x00"
                )
            )

        except Exception as exc:
            return (
                {
                    "status":
                        "WARN",
                    "present":
                        True,
                    "metadata_valid":
                        False,
                    "error_type":
                        type(exc).__name__,
                },
                ProductionHealthCheck(
                    name="writer_lock",
                    domain="writer_lock",
                    status="WARN",
                    detail=(
                        "writer lock sidecar "
                        "is unreadable or malformed"
                    ),
                ),
                (
                    ProductionHealthIssue(
                        code=(
                            "WRITER_LOCK_MALFORMED"
                        ),
                        domain="writer_lock",
                        severity="WARN",
                        message=(
                            "writer lock diagnostic "
                            "metadata is unreadable "
                            "or malformed"
                        ),
                    ),
                ),
            )

        valid = (
            isinstance(
                payload,
                dict,
            )
            and payload.get(
                "pid"
            )
            is not None
            and payload.get(
                "acquired_at"
            )
            is not None
        )

        if not valid:
            return (
                {
                    "status":
                        "WARN",
                    "present":
                        True,
                    "metadata_valid":
                        False,
                },
                ProductionHealthCheck(
                    name="writer_lock",
                    domain="writer_lock",
                    status="WARN",
                    detail=(
                        "writer lock diagnostic "
                        "metadata is malformed"
                    ),
                ),
                (
                    ProductionHealthIssue(
                        code=(
                            "WRITER_LOCK_MALFORMED"
                        ),
                        domain="writer_lock",
                        severity="WARN",
                        message=(
                            "writer lock diagnostic "
                            "metadata is malformed"
                        ),
                    ),
                ),
            )

        # The writer-lock file is persistent diagnostic
        # metadata. Its readable presence alone does not
        # prove that a writer currently owns the lock.
        return (
            {
                "status":
                    "PASS",
                "present":
                    True,
                "metadata_valid":
                    True,
                "diagnostic_sidecar":
                    True,
            },
            ProductionHealthCheck(
                name="writer_lock",
                domain="writer_lock",
                status="PASS",
                detail=(
                    "writer lock diagnostic "
                    "metadata is readable"
                ),
            ),
            (),
        )

    def _recovery_domain(
        self,
    ) -> tuple[
        dict[str, object],
        ProductionHealthCheck,
        tuple[
            ProductionHealthIssue,
            ...,
        ],
    ]:
        backup = (
            ProductionRegistryBackupService
            is not None
        )

        restore = (
            ProductionRegistryRestoreService
            is not None
        )

        atomicity = (
            ProductionRegistryRestoreAtomicityError
            is not None
        )

        ready = (
            backup
            and restore
            and atomicity
        )

        status = (
            "PASS"
            if ready
            else "FAIL"
        )

        issues = (
            ()
            if ready
            else (
                ProductionHealthIssue(
                    code=(
                        "RECOVERY_NOT_READY"
                    ),
                    domain=(
                        "recovery_readiness"
                    ),
                    severity="FAIL",
                    message=(
                        "production registry "
                        "recovery is not ready"
                    ),
                ),
            )
        )

        return (
            {
                "status":
                    status,
                "backup":
                    backup,
                "restore":
                    restore,
                "atomicity":
                    atomicity,
            },
            ProductionHealthCheck(
                name=(
                    "recovery_readiness"
                ),
                domain=(
                    "recovery_readiness"
                ),
                status=status,
                detail=(
                    "production registry "
                    f"recovery status={status}"
                ),
            ),
            issues,
        )

    def _history_domain(
        self,
    ) -> tuple[
        dict[str, object],
        ProductionHealthCheck,
        tuple[
            ProductionHealthIssue,
            ...,
        ],
    ]:
        retention = (
            ChampionHistoryRetentionExecutor
            is not None
            and
            ChampionHistoryRetentionAtomicityError
            is not None
        )

        rollback = (
            ChampionRollbackService
            is not None
            and
            ChampionRollbackHistoryReader
            is not None
        )

        ready = (
            retention
            and rollback
        )

        status = (
            "PASS"
            if ready
            else "FAIL"
        )

        issues = (
            ()
            if ready
            else (
                ProductionHealthIssue(
                    code=(
                        "HISTORY_SAFETY_NOT_READY"
                    ),
                    domain=(
                        "history_safety"
                    ),
                    severity="FAIL",
                    message=(
                        "history safety "
                        "foundation is not ready"
                    ),
                ),
            )
        )

        return (
            {
                "status":
                    status,
                "retention":
                    retention,
                "rollback":
                    rollback,
            },
            ProductionHealthCheck(
                name="history_safety",
                domain="history_safety",
                status=status,
                detail=(
                    "history safety "
                    f"status={status}"
                ),
            ),
            issues,
        )

    def _lifecycle_domain(
        self,
    ) -> tuple[
        dict[str, object],
        ProductionHealthCheck,
        tuple[
            ProductionHealthIssue,
            ...,
        ],
    ]:
        service = (
            ProductionLifecycleService
            is not None
        )

        contracts = all(
            item is not None
            for item in (
                ProductionLifecycleRequest,
                ProductionLifecycleStageResult,
                ProductionLifecycleResult,
            )
        )

        ready = (
            service
            and contracts
        )

        status = (
            "PASS"
            if ready
            else "FAIL"
        )

        issues = (
            ()
            if ready
            else (
                ProductionHealthIssue(
                    code=(
                        "LIFECYCLE_NOT_READY"
                    ),
                    domain=(
                        "lifecycle_readiness"
                    ),
                    severity="FAIL",
                    message=(
                        "production lifecycle "
                        "foundation is not ready"
                    ),
                ),
            )
        )

        return (
            {
                "status":
                    status,
                "service":
                    service,
                "contracts":
                    contracts,
            },
            ProductionHealthCheck(
                name=(
                    "lifecycle_readiness"
                ),
                domain=(
                    "lifecycle_readiness"
                ),
                status=status,
                detail=(
                    "production lifecycle "
                    f"status={status}"
                ),
            ),
            issues,
        )

    @staticmethod
    def _aggregate_status(
        statuses: list[str],
    ) -> str:
        if "FAIL" in statuses:
            return "FAIL"

        if "WARN" in statuses:
            return "WARN"

        return "PASS"
